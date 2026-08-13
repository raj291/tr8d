from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


class Action(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class PriceBar:
    symbol: str
    trading_date: date
    open: float
    close: float
    available_at: datetime

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        if self.open <= 0 or self.close <= 0:
            raise ValueError("prices must be positive")


@dataclass(frozen=True)
class Features:
    symbol: str
    decision_date: date
    values: tuple[float, ...]
    source_dates: tuple[date, ...]


@dataclass(frozen=True)
class Prediction:
    bull_probability: float
    expected_return: float


@dataclass(frozen=True)
class Proposal:
    symbol: str
    action: Action
    notional: float
    confidence: float
    reason: str


@dataclass(frozen=True)
class Wallet:
    cash: float
    quantities: dict[str, float]

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + sum(qty * marks.get(symbol, 0.0) for symbol, qty in self.quantities.items())


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    approved_notional: float = 0.0
