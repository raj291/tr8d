from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .domain import PriceBar
from .features import bars_by_symbol


@dataclass(frozen=True)
class Performance:
    start_equity: float
    end_equity: float
    total_return: float
    max_drawdown: float
    annualized_volatility: float
    trades: int
    turnover: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def performance(equities: list[float], trades: int, traded_notional: float) -> Performance:
    if not equities:
        raise ValueError("equity history is required")
    values = np.asarray(equities, dtype=float)
    peaks = np.maximum.accumulate(values)
    drawdowns = values / peaks - 1
    returns = values[1:] / values[:-1] - 1
    volatility = float(np.std(returns) * math.sqrt(252)) if len(returns) else 0.0
    return Performance(
        start_equity=round(float(values[0]), 6),
        end_equity=round(float(values[-1]), 6),
        total_return=round(float(values[-1] / values[0] - 1), 8),
        max_drawdown=round(float(drawdowns.min()), 8),
        annualized_volatility=round(volatility, 8),
        trades=trades,
        turnover=round(traded_notional / float(values[0]), 8),
    )


def baselines(bars: list[PriceBar], replay_dates: list) -> dict[str, dict[str, float]]:
    grouped = bars_by_symbol(bars)
    if not replay_dates:
        return {}
    start, end = replay_dates[0], replay_dates[-1]
    returns: dict[str, float] = {}
    for symbol, history in grouped.items():
        start_bar = next(bar for bar in history if bar.trading_date >= start)
        end_bar = next(bar for bar in reversed(history) if bar.trading_date <= end)
        returns[symbol] = end_bar.close / start_bar.open - 1
    equal_weight = sum(returns.values()) / len(returns)
    result = {"cash": {"end_equity": 10.0, "total_return": 0.0}}
    result["equal_weight"] = {"end_equity": round(10 * (1 + equal_weight), 6), "total_return": round(equal_weight, 8)}
    for symbol, value in returns.items():
        result[f"buy_hold_{symbol.lower()}"] = {
            "end_equity": round(10 * (1 + value), 6),
            "total_return": round(value, 8),
        }
    return result
