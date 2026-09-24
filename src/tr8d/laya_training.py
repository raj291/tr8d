from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .decision import LayaDecisionProvider
from .domain import PriceBar
from .features import FEATURE_NAMES, bars_by_symbol, build_features, training_rows
from .laya_quality import (
    classification_report,
    fit_selective_policy,
    promotion_gate,
    selective_report,
)
from .model import LogisticBaseline

ACTION_LABELS = ("A", "B", "C")
RLCD_QUESTION = {
    "action": {
        "type": "choice",
        "instructions": "Choose A for BUY, B for SELL, or C for HOLD.",
        "criteria": {
            "A": "BUY when the evidence supports positive risk-adjusted edge.",
            "B": "SELL only when an existing position has negative risk-adjusted edge.",
            "C": "HOLD when evidence or edge is insufficient.",
        },
    },
}


@dataclass(frozen=True)
class LayaTrainingExample:
    state: dict[str, Any]
    target_label: str
    target_probabilities: dict[str, float]
    symbol: str
    decision_date: str
    outcome_date: str
    realized_return: float
    feature_source_dates: tuple[str, ...]
    prediction_source: str = "feature_heuristic"
    portfolio_context: str = "flat"
    label_source: str = "realized_open_to_close"

    def __post_init__(self) -> None:
        if self.target_label not in ACTION_LABELS:
            raise ValueError("training target must be A, B, or C")
        if set(self.target_probabilities) != set(ACTION_LABELS):
            raise ValueError("training probabilities must cover A, B, and C")
        if not math.isclose(sum(self.target_probabilities.values()), 1.0, abs_tol=1e-8):
            raise ValueError("training probabilities must sum to one")


def _target(realized_return: float, threshold: float) -> str:
    if realized_return > threshold:
        return "A"
    if realized_return < -threshold:
        return "B"
    return "C"


def build_laya_examples(
    bars: list[PriceBar], threshold: float = 0.0025,
    prediction_source: str = "expanding_logistic",
) -> list[LayaTrainingExample]:
    if threshold <= 0:
        raise ValueError("training threshold must be positive")
    if prediction_source not in {"expanding_logistic", "feature_heuristic"}:
        raise ValueError("prediction source must be expanding_logistic or feature_heuristic")
    examples: list[LayaTrainingExample] = []
    for symbol, history in bars_by_symbol(bars).items():
        for target_bar in history[11:]:
            features = build_features(history, target_bar.trading_date)
            feature_map = dict(zip(FEATURE_NAMES, features.values, strict=True))
            if prediction_source == "expanding_logistic":
                training_x, training_y, realized_returns = training_rows(
                    history, target_bar.trading_date,
                )
                if len(training_x) < 30 or len(np.unique(training_y)) < 2:
                    continue
                prediction = LogisticBaseline().fit(
                    training_x, training_y, realized_returns,
                ).predict(features.values)
                bull_probability = prediction.bull_probability
                expected_return = prediction.expected_return
            else:
                volatility = max(feature_map["volatility_5d"], 1e-6)
                signal = feature_map["return_5d"] / (4 * volatility)
                bull_probability = float(np.clip(0.5 + signal, 0.05, 0.95))
                expected_return = float(0.35 * feature_map["return_1d"])
            realized_return = target_bar.close / target_bar.open - 1
            direction_label = _target(realized_return, threshold)
            contexts = (
                ("flat", 10.0, 0.0, "A" if direction_label == "A" else "C"),
                ("holding", 5.0, 0.05, direction_label),
            )
            for portfolio_context, cash, held_quantity, label in contexts:
                state = {
                    "instruction": "Choose a conservative paper-trading action; C is the safe default.",
                    "symbol": symbol,
                    "bull_probability": bull_probability,
                    "expected_return": expected_return,
                    "cash": cash,
                    "held_quantity": held_quantity,
                    "evidence": [],
                }
                examples.append(LayaTrainingExample(
                    state=state,
                    target_label=label,
                    target_probabilities={item: float(item == label) for item in ACTION_LABELS},
                    symbol=symbol,
                    decision_date=target_bar.trading_date.isoformat(),
                    outcome_date=target_bar.trading_date.isoformat(),
                    realized_return=realized_return,
                    feature_source_dates=tuple(
                        item.isoformat() for item in features.source_dates
                    ),
                    prediction_source=prediction_source,
                    portfolio_context=portfolio_context,
                ))
    return sorted(
        examples,
        key=lambda item: (item.decision_date, item.symbol, item.portfolio_context),
    )


def temporal_split(
    examples: list[LayaTrainingExample], train_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
) -> dict[str, list[LayaTrainingExample]]:
    if not examples:
        raise ValueError("cannot split an empty training dataset")
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("training and calibration fractions must leave a test split")
    dates = sorted({item.decision_date for item in examples})
    if len(dates) < 3:
        raise ValueError("at least three decision dates are required")
    train_end = min(max(1, int(len(dates) * train_fraction)), len(dates) - 2)
    calibration_end = max(train_end + 1, int(len(dates) * (train_fraction + calibration_fraction)))
    calibration_end = min(calibration_end, len(dates) - 1)
    train_dates = set(dates[:train_end])
    calibration_dates = set(dates[train_end:calibration_end])
    test_dates = set(dates[calibration_end:])
    return {
        "train": [item for item in examples if item.decision_date in train_dates],
        "calibration": [item for item in examples if item.decision_date in calibration_dates],
        "test": [item for item in examples if item.decision_date in test_dates],
    }


def _digest(examples: list[LayaTrainingExample]) -> str:
    canonical = "\n".join(json.dumps(asdict(item), sort_keys=True) for item in examples)
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_laya_dataset(
    examples: list[LayaTrainingExample], output_directory: str | Path,
    source: str = "synthetic", synthetic: bool = True,
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    splits = temporal_split(examples)
    for name, rows in splits.items():
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in rows),
            encoding="utf-8",
        )
        (output / f"{name}_rlcd.jsonl").write_text(
            "".join(
                json.dumps({
                    "state": json.dumps(item.state, sort_keys=True),
                    "questions": json.dumps(RLCD_QUESTION, sort_keys=True),
                    "gold": json.dumps({
                        "action": {"probabilities": item.target_probabilities},
                    }, sort_keys=True),
                    "decision_date": item.decision_date,
                    "symbol": item.symbol,
                }, sort_keys=True) + "\n"
                for item in rows
            ),
            encoding="utf-8",
        )
    manifest = {
        "format": "tr8d-laya-choice-v1",
        "schema": LayaDecisionProvider._schema,
        "labels": {"A": "BUY", "B": "SELL", "C": "HOLD"},
        "digest": _digest(examples),
        "examples": len(examples),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_date_ranges": {
            name: [rows[0].decision_date, rows[-1].decision_date] for name, rows in splits.items()
        },
        "label_counts": {
            label: sum(item.target_label == label for item in examples) for label in ACTION_LABELS
        },
        "source": source,
        "synthetic": synthetic,
        "point_in_time_features": True,
        "prediction_sources": sorted({item.prediction_source for item in examples}),
        "portfolio_context_counts": {
            context: sum(item.portfolio_context == context for item in examples)
            for context in ("flat", "holding")
        },
        "rlcd_compatible_exports": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def load_laya_split(dataset_directory: str | Path, split: str) -> list[LayaTrainingExample]:
    path = Path(dataset_directory) / f"{split}.jsonl"
    if split not in {"train", "calibration", "test"}:
        raise ValueError("split must be train, calibration, or test")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            payload["feature_source_dates"] = tuple(payload["feature_source_dates"])
            rows.append(LayaTrainingExample(**payload))
    return rows


def _tokenize(agent: Any, examples: list[LayaTrainingExample]) -> list[dict[str, Any]]:
    from laya.common import QTYPES, build_sequence
    from laya.structured import questions_from_json_schema

    question = questions_from_json_schema(LayaDecisionProvider._schema)["action"]
    internal = {
        "t": question["type"],
        "ins": question["instructions"],
        "crit": question["criteria"],
    }
    items = []
    for example in examples:
        sequence, markers = build_sequence(
            agent.tok, example.state, internal,
            agent.cfg["max_len"], agent.cfg["head_max_len"],
        )
        if len(markers) != len(ACTION_LABELS):
            raise ValueError("Laya tokenizer did not produce three action markers")
        items.append({
            "ids": sequence,
            "markers": markers,
            "qtype": QTYPES["choice"],
            "label": ACTION_LABELS.index(example.target_label),
        })
    return items


def _batches(items: list[dict[str, Any]], batch_size: int, pad_id: int):
    import torch

    for start in range(0, len(items), batch_size):
        rows = items[start:start + batch_size]
        length = max(len(item["ids"]) for item in rows)
        input_ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
        attention = torch.zeros((len(rows), length), dtype=torch.long)
        marker_positions = torch.tensor([item["markers"] for item in rows], dtype=torch.long)
        marker_mask = torch.ones_like(marker_positions, dtype=torch.bool)
        for index, item in enumerate(rows):
            size = len(item["ids"])
            input_ids[index, :size] = torch.tensor(item["ids"], dtype=torch.long)
            attention[index, :size] = 1
        yield {
            "input_ids": input_ids,
            "attention_mask": attention,
            "marker_pos": marker_positions,
            "marker_mask": marker_mask,
            "qtype": torch.tensor([item["qtype"] for item in rows], dtype=torch.long),
            "label": torch.tensor([item["label"] for item in rows], dtype=torch.long),
        }


def _logits(model: Any, items: list[dict[str, Any]], batch_size: int, pad_id: int, device: Any):
    import torch

    collected, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in _batches(items, batch_size, pad_id):
            values, _ = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
            collected.append(values.float().cpu())
            labels.append(batch["label"])
    return torch.cat(collected), torch.cat(labels)


def _probabilities(logits: Any, temperature: float = 1.0) -> np.ndarray:
    import torch

    return torch.softmax(logits / temperature, dim=-1).cpu().numpy()


def _metrics(logits: Any, labels: Any, temperature: float = 1.0) -> dict[str, Any]:
    return classification_report(_probabilities(logits, temperature), labels.cpu().numpy())


def _quality_report(
    calibration_logits: Any, calibration_labels: Any,
    test_logits: Any, test_labels: Any, temperature: float,
    baseline_brier: float | None, target_accuracy: float,
    minimum_coverage: float, minimum_test_examples: int,
) -> dict[str, Any]:
    calibration_probabilities = _probabilities(calibration_logits, temperature)
    test_probabilities = _probabilities(test_logits, temperature)
    calibration_array = calibration_labels.cpu().numpy()
    test_array = test_labels.cpu().numpy()
    policy = fit_selective_policy(
        calibration_probabilities, calibration_array,
        target_accuracy=target_accuracy, minimum_coverage=minimum_coverage,
    )
    overall = classification_report(test_probabilities, test_array)
    selective = selective_report(test_probabilities, test_array, policy.threshold)
    gate = promotion_gate(
        overall, selective, baseline_brier,
        target_accuracy=target_accuracy,
        minimum_coverage=minimum_coverage,
        minimum_test_examples=minimum_test_examples,
    )
    return {
        "confidence_policy": policy.as_dict(),
        "untouched_test": overall,
        "untouched_test_selective": selective,
        "promotion_gate": gate,
    }


def _fit_temperature(logits: Any, labels: Any) -> float:
    import torch

    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=75)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_temperature.exp(), 0.5, 5.0).item())


def _safe_temperature(value: Any) -> float:
    try:
        return float(np.clip(float(value), 0.5, 5.0))
    except (TypeError, ValueError):
        return 1.0


def train_laya_head(
    dataset_directory: str | Path, output_directory: str | Path,
    base_model: str = "convaiinnovations/laya", device: str = "cpu",
    epochs: int = 1, batch_size: int = 8, learning_rate: float = 1e-4,
    max_train_examples: int = 0, seed: int = 7,
    target_accuracy: float = 0.85, minimum_coverage: float = 0.10,
    minimum_test_examples: int = 500, patience: int = 2,
) -> dict[str, Any]:
    """CPU-safe domain adaptation of Laya's decision head using temporal labels."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or patience < 1:
        raise ValueError("positive epochs, batch size, and learning rate are required")
    if not 0 < target_accuracy <= 1 or not 0 < minimum_coverage <= 1:
        raise ValueError("target accuracy and minimum coverage must be in (0, 1]")
    if minimum_test_examples < 1:
        raise ValueError("minimum test examples must be positive")
    import laya
    import torch
    from safetensors.torch import save_file

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_examples = load_laya_split(dataset_directory, "train")
    calibration_examples = load_laya_split(dataset_directory, "calibration")
    test_examples = load_laya_split(dataset_directory, "test")
    manifest = json.loads(
        (Path(dataset_directory) / "manifest.json").read_text(encoding="utf-8")
    )
    if max_train_examples > 0:
        train_examples = train_examples[:max_train_examples]
    if not train_examples or not calibration_examples or not test_examples:
        raise ValueError("training, calibration, and test splits must all be non-empty")
    if len(train_examples) < 10:
        raise ValueError("at least ten training examples are required")
    validation_count = max(1, int(len(train_examples) * 0.10))
    validation_examples = train_examples[-validation_count:]
    train_examples = train_examples[:-validation_count]
    agent = laya.load(base_model, device=device)
    train_items = _tokenize(agent, train_examples)
    validation_items = _tokenize(agent, validation_examples)
    calibration_items = _tokenize(agent, calibration_examples)
    test_items = _tokenize(agent, test_examples)
    target_device = agent.device
    pad_id = agent.tok.pad_token_id

    before_logits, test_labels = _logits(
        agent.model, test_items, batch_size, pad_id, target_device,
    )
    baseline = _metrics(before_logits, test_labels)

    for parameter in agent.model.parameters():
        parameter.requires_grad = False
    for component in (agent.model.head, agent.model.type_emb, agent.model.scorer):
        for parameter in component.parameters():
            parameter.requires_grad = True
    trainable_named = [
        (name, parameter)
        for name, parameter in agent.model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in trainable_named]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    label_counts = np.bincount([item["label"] for item in train_items], minlength=3)
    imbalance_ratio = float(label_counts.max() / max(1, label_counts.min()))
    class_weights = (
        torch.tensor(
            [len(train_items) / max(1, 3 * count) for count in label_counts],
            dtype=torch.float32, device=target_device,
        )
        if imbalance_ratio >= 1.5 else None
    )
    losses = []
    initial_validation_logits, initial_validation_labels = _logits(
        agent.model, validation_items, batch_size, pad_id, target_device,
    )
    best_validation_loss = float(
        torch.nn.functional.cross_entropy(
            initial_validation_logits, initial_validation_labels,
        ).item()
    )
    epoch_history = [{"epoch": 0, "validation_loss": best_validation_loss}]
    best_epoch = 0
    best_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in trainable_named
    }
    stale_epochs = 0
    for epoch in range(epochs):
        agent.model.train()
        agent.model.encoder.eval()
        random.Random(seed + epoch).shuffle(train_items)
        for batch in _batches(train_items, batch_size, pad_id):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = agent.model(
                batch["input_ids"].to(target_device),
                batch["attention_mask"].to(target_device),
                batch["marker_pos"].to(target_device),
                batch["marker_mask"].to(target_device),
                batch["qtype"].to(target_device),
            )
            loss = torch.nn.functional.cross_entropy(
                logits.float(), batch["label"].to(target_device), weight=class_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation_logits, validation_labels = _logits(
            agent.model, validation_items, batch_size, pad_id, target_device,
        )
        validation_loss = float(
            torch.nn.functional.cross_entropy(validation_logits, validation_labels).item()
        )
        epoch_history.append({
            "epoch": epoch + 1,
            "validation_loss": validation_loss,
        })
        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in trainable_named
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    with torch.no_grad():
        for name, parameter in trainable_named:
            parameter.copy_(best_state[name].to(target_device))

    calibration_logits, calibration_labels = _logits(
        agent.model, calibration_items, batch_size, pad_id, target_device,
    )
    temperature = _fit_temperature(calibration_logits, calibration_labels)
    after_logits, test_labels = _logits(
        agent.model, test_items, batch_size, pad_id, target_device,
    )
    trained = _metrics(after_logits, test_labels, temperature)
    quality = _quality_report(
        calibration_logits, calibration_labels, after_logits, test_labels, temperature,
        baseline["brier"], target_accuracy, minimum_coverage, minimum_test_examples,
    )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    state = {
        name: parameter.detach().cpu().half().contiguous()
        for name, parameter in agent.model.state_dict().items()
    }
    save_file(state, str(output / "model.safetensors"))
    agent.model.encoder.config.save_pretrained(output / "encoder")
    agent.tok.save_pretrained(output / "tokenizer")
    config = dict(agent.cfg)
    temperatures = list(config.get("temperature", [1.0, 1.0, 1.0]))
    temperatures = [_safe_temperature(value) for value in (temperatures + [1.0] * 3)[:3]]
    temperatures[0] = temperature
    config["temperature"] = temperatures
    config.pop("temperature_by_options", None)
    config["fine_tuned"] = True
    config["model_name"] = "tr8d-laya-head-v1"
    config["training"] = {
        "method": "supervised_head_only",
        "base_model": base_model,
        "dataset_digest": manifest["digest"],
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "early_stopping_patience": patience,
        "class_weights_used": class_weights is not None,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    (output / "rl_agent_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    policy_payload = {
        **quality["confidence_policy"],
        "promotion_gate_passed": quality["promotion_gate"]["passed"],
        "dataset_digest": manifest["digest"],
    }
    (output / "tr8d_policy.json").write_text(
        json.dumps(policy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    report = {
        "status": "completed",
        "method": "supervised_head_only",
        "base_model": base_model,
        "dataset_digest": manifest["digest"],
        "dataset_source": manifest["source"],
        "device": str(target_device),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "calibration_examples": len(calibration_examples),
        "test_examples": len(test_examples),
        "epochs": epochs,
        "epochs_completed": len(epoch_history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "early_stopping_patience": patience,
        "class_weights_used": class_weights is not None,
        "training_label_counts": {
            label: int(label_counts[index]) for index, label in enumerate(ACTION_LABELS)
        },
        "epoch_history": epoch_history,
        "mean_training_loss": float(np.mean(losses)),
        "temperature": temperature,
        "before": baseline,
        "after": trained,
        **quality,
        "promoted": quality["promotion_gate"]["passed"],
        "output": str(output),
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return report


def evaluate_laya_checkpoint(
    dataset_directory: str | Path, model: str | Path,
    device: str = "cpu", batch_size: int = 8,
    target_accuracy: float = 0.85, minimum_coverage: float = 0.10,
    minimum_test_examples: int = 500, write_policy: bool = False,
) -> dict[str, Any]:
    """Audit a saved checkpoint without fitting anything on the untouched test split."""
    import laya

    agent = laya.load(str(model), device=device)
    calibration_examples = load_laya_split(dataset_directory, "calibration")
    test_examples = load_laya_split(dataset_directory, "test")
    calibration_items = _tokenize(agent, calibration_examples)
    test_items = _tokenize(agent, test_examples)
    calibration_logits, calibration_labels = _logits(
        agent.model, calibration_items, batch_size, agent.tok.pad_token_id, agent.device,
    )
    test_logits, test_labels = _logits(
        agent.model, test_items, batch_size, agent.tok.pad_token_id, agent.device,
    )
    temperatures = list(agent.cfg.get("temperature", [1.0, 1.0, 1.0]))
    temperature = _safe_temperature(
        agent.cfg.get("temperature_by_options", {}).get(
            "choice:3-5", temperatures[0] if temperatures else 1.0,
        )
    )
    baseline_brier = None
    model_path = Path(model)
    prior_report_path = model_path / "training_report.json"
    if prior_report_path.exists():
        prior_report = json.loads(prior_report_path.read_text(encoding="utf-8"))
        baseline_brier = prior_report.get("before", {}).get("brier")
    quality = _quality_report(
        calibration_logits, calibration_labels, test_logits, test_labels, temperature,
        baseline_brier, target_accuracy, minimum_coverage, minimum_test_examples,
    )
    manifest = json.loads(
        (Path(dataset_directory) / "manifest.json").read_text(encoding="utf-8")
    )
    report = {
        "status": "completed",
        "model": str(model),
        "dataset_digest": manifest["digest"],
        "temperature": temperature,
        **quality,
        "promoted": quality["promotion_gate"]["passed"],
    }
    if write_policy:
        if not model_path.is_dir():
            raise ValueError("--write-policy requires a local model directory")
        policy_payload = {
            **quality["confidence_policy"],
            "promotion_gate_passed": quality["promotion_gate"]["passed"],
            "dataset_digest": manifest["digest"],
        }
        (model_path / "tr8d_policy.json").write_text(
            json.dumps(policy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    return report
