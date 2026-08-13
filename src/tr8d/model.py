from __future__ import annotations

import numpy as np

from .domain import Prediction


class LogisticBaseline:
    """Small deterministic logistic baseline with standardized features."""

    def __init__(self, learning_rate: float = 0.08, iterations: int = 350, l2: float = 0.02):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2 = l2
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.average_returns = (0.0, 0.0)

    def fit(self, x: np.ndarray, y: np.ndarray, realized_returns: np.ndarray) -> "LogisticBaseline":
        if len(x) < 30 or len(np.unique(y)) < 2:
            raise ValueError("training requires at least 30 rows across two classes")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-9] = 1.0
        standardized = (x - self.mean) / self.scale
        design = np.column_stack([np.ones(len(x)), standardized])
        self.weights = np.zeros(design.shape[1])
        for _ in range(self.iterations):
            logits = np.clip(design @ self.weights, -30, 30)
            probabilities = 1 / (1 + np.exp(-logits))
            gradient = design.T @ (probabilities - y) / len(x)
            gradient[1:] += self.l2 * self.weights[1:]
            self.weights -= self.learning_rate * gradient
        positive = realized_returns[y == 1]
        negative = realized_returns[y == 0]
        self.average_returns = (float(negative.mean()), float(positive.mean()))
        return self

    def predict(self, values: tuple[float, ...]) -> Prediction:
        if self.mean is None or self.scale is None or self.weights is None:
            raise RuntimeError("model is not fitted")
        standardized = (np.asarray(values) - self.mean) / self.scale
        design = np.concatenate([[1.0], standardized])
        probability = float(1 / (1 + np.exp(-np.clip(design @ self.weights, -30, 30))))
        bearish, bullish = self.average_returns
        expected = probability * bullish + (1 - probability) * bearish
        return Prediction(bull_probability=probability, expected_return=expected)

    def predict_probabilities(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or self.weights is None:
            raise RuntimeError("model is not fitted")
        standardized = (values - self.mean) / self.scale
        design = np.column_stack([np.ones(len(values)), standardized])
        return 1 / (1 + np.exp(-np.clip(design @ self.weights, -30, 30)))


class XGBoostBaseline:
    """Optional boosted-tree model; imported lazily to keep the core lightweight."""

    def __init__(self, seed: int = 7):
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError("XGBoost is optional; install with `pip install -e '.[ml]'`") from error
        self.model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=seed,
        )

    def fit(self, x: np.ndarray, y: np.ndarray, realized_returns: np.ndarray) -> "XGBoostBaseline":
        if len(x) < 30 or len(np.unique(y)) < 2:
            raise ValueError("training requires at least 30 rows across two classes")
        self.model.fit(x, y)
        return self

    def predict_probabilities(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(values)[:, 1], dtype=float)


def create_model(name: str, seed: int = 7):
    if name == "logistic":
        return LogisticBaseline()
    if name == "xgboost":
        return XGBoostBaseline(seed)
    raise ValueError(f"unknown model: {name}")
