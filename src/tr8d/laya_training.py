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
from .features import FEATURE_NAMES, bars_by_symbol, build_features

ACTION_LABELS = ("A", "B", "C")


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
) -> list[LayaTrainingExample]:
    if threshold <= 0:
        raise ValueError("training threshold must be positive")
    examples: list[LayaTrainingExample] = []
    for symbol, history in bars_by_symbol(bars).items():
        for target_bar in history[11:]:
            features = build_features(history, target_bar.trading_date)
            feature_map = dict(zip(FEATURE_NAMES, features.values, strict=True))
            volatility = max(feature_map["volatility_5d"], 1e-6)
            signal = feature_map["return_5d"] / (4 * volatility)
            bull_probability = float(np.clip(0.5 + signal, 0.05, 0.95))
            expected_return = float(0.35 * feature_map["return_1d"])
            state = {
                "instruction": "Choose a conservative paper-trading action; C is the safe default.",
                "symbol": symbol,
                "bull_probability": bull_probability,
                "expected_return": expected_return,
                "cash": 10.0,
                "held_quantity": 0.0,
                "evidence": [],
            }
            realized_return = target_bar.close / target_bar.open - 1
            label = _target(realized_return, threshold)
            examples.append(LayaTrainingExample(
                state=state,
                target_label=label,
                target_probabilities={item: float(item == label) for item in ACTION_LABELS},
                symbol=symbol,
                decision_date=target_bar.trading_date.isoformat(),
                outcome_date=target_bar.trading_date.isoformat(),
                realized_return=realized_return,
                feature_source_dates=tuple(item.isoformat() for item in features.source_dates),
            ))
    return sorted(examples, key=lambda item: (item.decision_date, item.symbol))


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


def _metrics(logits: Any, labels: Any, temperature: float = 1.0) -> dict[str, float]:
    import torch

    probabilities = torch.softmax(logits / temperature, dim=-1)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=len(ACTION_LABELS)).float()
    confidence, predicted = probabilities.max(dim=-1)
    accuracy = (predicted == labels).float().mean().item()
    brier = ((probabilities - one_hot) ** 2).sum(dim=-1).mean().item()
    ece = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1)
        if selected.any():
            observed = (predicted[selected] == labels[selected]).float().mean().item()
            ece += selected.float().mean().item() * abs(confidence[selected].mean().item() - observed)
    return {"accuracy": accuracy, "brier": brier, "ece": ece}


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


def train_laya_head(
    dataset_directory: str | Path, output_directory: str | Path,
    base_model: str = "convaiinnovations/laya", device: str = "cpu",
    epochs: int = 1, batch_size: int = 8, learning_rate: float = 1e-4,
    max_train_examples: int = 0, seed: int = 7,
) -> dict[str, Any]:
    """CPU-safe domain adaptation of Laya's decision head using temporal labels."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("positive epochs, batch size, and learning rate are required")
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
    agent = laya.load(base_model, device=device)
    train_items = _tokenize(agent, train_examples)
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
    trainable = [parameter for parameter in agent.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    label_counts = np.bincount([item["label"] for item in train_items], minlength=3)
    class_weights = torch.tensor(
        [len(train_items) / max(1, 3 * count) for count in label_counts],
        dtype=torch.float32, device=target_device,
    )
    losses = []
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

    calibration_logits, calibration_labels = _logits(
        agent.model, calibration_items, batch_size, pad_id, target_device,
    )
    temperature = _fit_temperature(calibration_logits, calibration_labels)
    after_logits, test_labels = _logits(
        agent.model, test_items, batch_size, pad_id, target_device,
    )
    trained = _metrics(after_logits, test_labels, temperature)

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
    def safe_temperature(value: Any) -> float:
        try:
            return float(np.clip(float(value), 0.5, 5.0))
        except (TypeError, ValueError):
            return 1.0

    temperatures = list(config.get("temperature", [1.0, 1.0, 1.0]))
    temperatures = [safe_temperature(value) for value in (temperatures + [1.0] * 3)[:3]]
    temperatures[0] = temperature
    config["temperature"] = temperatures
    buckets = {
        name: safe_temperature(value)
        for name, value in config.get("temperature_by_options", {}).items()
    }
    buckets["choice:3-5"] = temperature
    config["temperature_by_options"] = buckets
    config["fine_tuned"] = True
    config["model_name"] = "tr8d-laya-head-v1"
    config["training"] = {
        "method": "supervised_head_only",
        "base_model": base_model,
        "dataset_digest": manifest["digest"],
        "train_examples": len(train_examples),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    (output / "rl_agent_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    report = {
        "status": "completed",
        "method": "supervised_head_only",
        "base_model": base_model,
        "dataset_digest": manifest["digest"],
        "dataset_source": manifest["source"],
        "device": str(target_device),
        "train_examples": len(train_examples),
        "calibration_examples": len(calibration_examples),
        "test_examples": len(test_examples),
        "epochs": epochs,
        "mean_training_loss": float(np.mean(losses)),
        "temperature": temperature,
        "before": baseline,
        "after": trained,
        "promoted": trained["accuracy"] > baseline["accuracy"] and trained["brier"] < baseline["brier"],
        "output": str(output),
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return report
