from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from .domain import PriceBar
from .features import bars_by_symbol, training_rows
from .manifest import create_manifest
from .model import create_model


@dataclass(frozen=True)
class Fold:
    number: int
    train_dates: tuple[date, ...]
    calibration_dates: tuple[date, ...]
    test_dates: tuple[date, ...]

    def __post_init__(self) -> None:
        if not self.train_dates or not self.calibration_dates or not self.test_dates:
            raise ValueError("fold segments cannot be empty")
        if max(self.train_dates) >= min(self.calibration_dates):
            raise ValueError("training must precede calibration")
        if max(self.calibration_dates) >= min(self.test_dates):
            raise ValueError("calibration must precede testing")


class PlattCalibrator:
    def __init__(self, learning_rate: float = 0.05, iterations: int = 500):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = np.zeros(2)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        if len(probabilities) < 10 or len(np.unique(labels)) < 2:
            raise ValueError("calibration requires at least 10 rows across two classes")
        logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))
        design = np.column_stack([np.ones(len(logits)), logits])
        for _ in range(self.iterations):
            predicted = 1 / (1 + np.exp(-np.clip(design @ self.weights, -30, 30)))
            self.weights -= self.learning_rate * (design.T @ (predicted - labels) / len(labels))
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))
        design = np.column_stack([np.ones(len(logits)), logits])
        return 1 / (1 + np.exp(-np.clip(design @ self.weights, -30, 30)))


def expanding_folds(
    dates: list[date], min_train_days: int = 120, calibration_days: int = 40, test_days: int = 40,
) -> list[Fold]:
    unique = sorted(set(dates))
    folds: list[Fold] = []
    cursor = min_train_days
    while cursor + calibration_days + test_days <= len(unique):
        folds.append(Fold(
            number=len(folds) + 1,
            train_dates=tuple(unique[:cursor]),
            calibration_dates=tuple(unique[cursor:cursor + calibration_days]),
            test_dates=tuple(unique[cursor + calibration_days:cursor + calibration_days + test_days]),
        ))
        cursor += test_days
    if not folds:
        required = min_train_days + calibration_days + test_days
        raise ValueError(f"walk-forward evaluation requires at least {required} trading days")
    return folds


def _panel_rows(bars: list[PriceBar]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_blocks, label_blocks, return_blocks, date_blocks = [], [], [], []
    for history in bars_by_symbol(bars).values():
        x, y, returns = training_rows(history, date.max)
        target_dates = np.asarray([bar.trading_date for bar in history[11:]], dtype=object)
        feature_blocks.append(x)
        label_blocks.append(y)
        return_blocks.append(returns)
        date_blocks.append(target_dates)
    return (
        np.concatenate(feature_blocks), np.concatenate(label_blocks),
        np.concatenate(return_blocks), np.concatenate(date_blocks),
    )


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
    predicted = probabilities >= 0.5
    positives = labels == 1
    negatives = ~positives
    true_positive_rate = float(np.mean(predicted[positives])) if positives.any() else 0.0
    true_negative_rate = float(np.mean(~predicted[negatives])) if negatives.any() else 0.0
    bins = np.minimum((probabilities * 10).astype(int), 9)
    calibration_error = 0.0
    for bin_number in range(10):
        selected = bins == bin_number
        if selected.any():
            calibration_error += float(np.mean(selected)) * abs(float(np.mean(probabilities[selected]) - np.mean(labels[selected])))
    return {
        "rows": len(labels),
        "accuracy": round(float(np.mean(predicted == positives)), 8),
        "balanced_accuracy": round((true_positive_rate + true_negative_rate) / 2, 8),
        "brier_score": round(float(np.mean((probabilities - labels) ** 2)), 8),
        "log_loss": round(float(-np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))), 8),
        "expected_calibration_error": round(calibration_error, 8),
    }


def evaluate_models(bars: list[PriceBar], models: tuple[str, ...] = ("logistic", "xgboost"), seed: int = 7) -> dict:
    x, y, returns, target_dates = _panel_rows(bars)
    folds = expanding_folds(list(target_dates))
    report: dict = {"models": {}, "folds": []}
    for fold in folds:
        report["folds"].append({
            "number": fold.number,
            "train_end": max(fold.train_dates).isoformat(),
            "calibration_start": min(fold.calibration_dates).isoformat(),
            "calibration_end": max(fold.calibration_dates).isoformat(),
            "test_start": min(fold.test_dates).isoformat(),
            "test_end": max(fold.test_dates).isoformat(),
        })
    for model_name in models:
        raw_predictions, calibrated_predictions, test_labels = [], [], []
        for fold in folds:
            train_mask = np.isin(target_dates, fold.train_dates)
            calibration_mask = np.isin(target_dates, fold.calibration_dates)
            test_mask = np.isin(target_dates, fold.test_dates)
            model = create_model(model_name, seed).fit(x[train_mask], y[train_mask], returns[train_mask])
            calibration_raw = model.predict_probabilities(x[calibration_mask])
            calibrator = PlattCalibrator().fit(calibration_raw, y[calibration_mask])
            test_raw = model.predict_probabilities(x[test_mask])
            raw_predictions.append(test_raw)
            calibrated_predictions.append(calibrator.predict(test_raw))
            test_labels.append(y[test_mask])
        labels = np.concatenate(test_labels)
        report["models"][model_name] = {
            "uncalibrated": _metrics(labels, np.concatenate(raw_predictions)),
            "calibrated": _metrics(labels, np.concatenate(calibrated_predictions)),
        }
    report["promotion_candidate"] = min(
        report["models"], key=lambda name: report["models"][name]["calibrated"]["brier_score"]
    )
    return report


def write_model_report(report: dict, bars: list[PriceBar], source: str, path: str | Path) -> Path:
    manifest = create_manifest(bars, source)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": manifest.version,
        "dataset_sha256": manifest.content_sha256,
        **report,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
