from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import numpy as np

ACTION_LABELS = ("A", "B", "C")


def _validated(
    probabilities: np.ndarray, labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(ACTION_LABELS):
        raise ValueError("probabilities must have shape (rows, 3)")
    if labels.ndim != 1 or len(labels) != len(probabilities) or not len(labels):
        raise ValueError("labels must be a non-empty vector matching probabilities")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("each probability row must sum to one")
    if np.any(labels < 0) or np.any(labels >= len(ACTION_LABELS)):
        raise ValueError("labels must use class indexes 0, 1, or 2")
    return probabilities, labels


def classification_report(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    probabilities, labels = _validated(probabilities, labels)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    one_hot = np.eye(len(ACTION_LABELS))[labels]
    confusion = np.zeros((len(ACTION_LABELS), len(ACTION_LABELS)), dtype=int)
    for actual, selected in zip(labels, predicted, strict=True):
        confusion[actual, selected] += 1

    per_class = {}
    recalls = []
    f1_scores = []
    for index, label in enumerate(ACTION_LABELS):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        selected = int(confusion[:, index].sum())
        precision = true_positive / selected if selected else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            recalls.append(recall)
            f1_scores.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index, (lower, upper) in enumerate(pairwise(edges)):
        selected = (confidence >= lower if index == 0 else confidence > lower) & (
            confidence <= upper
        )
        if selected.any():
            ece += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())

    counts = np.bincount(labels, minlength=len(ACTION_LABELS))
    accuracy = float(correct.mean())
    return {
        "examples": len(labels),
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)))),
        "ece": float(ece),
        "majority_baseline_accuracy": float(counts.max() / len(labels)),
        "accuracy_lift_over_majority": float(accuracy - counts.max() / len(labels)),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def selective_report(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float,
) -> dict[str, Any]:
    probabilities, labels = _validated(probabilities, labels)
    if not 0 <= threshold <= 1:
        raise ValueError("selective threshold must be between zero and one")
    confidence = probabilities.max(axis=1)
    selected = confidence >= threshold
    count = int(selected.sum())
    return {
        "threshold": threshold,
        "coverage": float(selected.mean()),
        "selected_examples": count,
        "accuracy": (
            float((probabilities[selected].argmax(axis=1) == labels[selected]).mean())
            if count else None
        ),
    }


@dataclass(frozen=True)
class SelectivePolicy:
    threshold: float
    target_accuracy: float
    minimum_coverage: float
    calibration_accuracy: float | None
    calibration_coverage: float
    calibration_examples: int
    target_met: bool
    confidence_kind: str = "answer_probability"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_selective_policy(
    probabilities: np.ndarray, labels: np.ndarray,
    target_accuracy: float = 0.85, minimum_coverage: float = 0.10,
) -> SelectivePolicy:
    probabilities, labels = _validated(probabilities, labels)
    if not 0 < target_accuracy <= 1 or not 0 < minimum_coverage <= 1:
        raise ValueError("target accuracy and minimum coverage must be in (0, 1]")
    confidence = probabilities.max(axis=1)
    candidates = sorted({0.0, *confidence.tolist()})
    eligible = []
    for threshold in candidates:
        report = selective_report(probabilities, labels, threshold)
        if (
            report["coverage"] >= minimum_coverage
            and report["accuracy"] is not None
            and report["accuracy"] >= target_accuracy
        ):
            eligible.append(report)
    if not eligible:
        report = selective_report(probabilities, labels, 1.0)
        return SelectivePolicy(
            threshold=1.0,
            target_accuracy=target_accuracy,
            minimum_coverage=minimum_coverage,
            calibration_accuracy=report["accuracy"],
            calibration_coverage=report["coverage"],
            calibration_examples=report["selected_examples"],
            target_met=False,
        )
    best = max(eligible, key=lambda item: (item["coverage"], -item["threshold"]))
    return SelectivePolicy(
        threshold=float(best["threshold"]),
        target_accuracy=target_accuracy,
        minimum_coverage=minimum_coverage,
        calibration_accuracy=float(best["accuracy"]),
        calibration_coverage=float(best["coverage"]),
        calibration_examples=int(best["selected_examples"]),
        target_met=True,
    )


def promotion_gate(
    overall: dict[str, Any], selective: dict[str, Any], baseline_brier: float | None,
    target_accuracy: float = 0.85, minimum_coverage: float = 0.10,
    minimum_test_examples: int = 500,
) -> dict[str, Any]:
    checks = {
        "enough_untouched_test_examples": overall["examples"] >= minimum_test_examples,
        "beats_majority_baseline": overall["accuracy_lift_over_majority"] > 0,
        "brier_improved": baseline_brier is not None and overall["brier"] < baseline_brier,
        "selective_accuracy_met": (
            selective["accuracy"] is not None and selective["accuracy"] >= target_accuracy
        ),
        "selective_coverage_met": selective["coverage"] >= minimum_coverage,
    }
    return {
        "target_selective_accuracy": target_accuracy,
        "minimum_selective_coverage": minimum_coverage,
        "minimum_untouched_test_examples": minimum_test_examples,
        "checks": checks,
        "passed": all(checks.values()),
    }
