from __future__ import annotations

from datetime import UTC, date, datetime, time

import numpy as np

from .domain import Features, PriceBar

FEATURE_NAMES = ("return_1d", "return_3d", "return_5d", "return_10d", "volatility_5d", "distance_5d")


def decision_cutoff(decision_date: date) -> datetime:
    """Canonical pre-open cutoff for v1 (09:20 ET during daylight time)."""
    return datetime.combine(decision_date, time(13, 20), tzinfo=UTC)


def bars_by_symbol(bars: list[PriceBar]) -> dict[str, list[PriceBar]]:
    grouped: dict[str, list[PriceBar]] = {}
    for bar in sorted(bars, key=lambda item: (item.symbol, item.trading_date)):
        grouped.setdefault(bar.symbol, []).append(bar)
    return grouped


def _values_from_closes(closes: np.ndarray) -> tuple[float, ...]:
    daily = closes[1:] / closes[:-1] - 1

    def period_return(window: int) -> float:
        return float(closes[-1] / closes[-1 - window] - 1)

    return (
        period_return(1),
        period_return(3),
        period_return(5),
        period_return(10),
        float(np.std(daily[-5:])),
        float(closes[-1] / np.mean(closes[-5:]) - 1),
    )


def build_features(history: list[PriceBar], decision_date: date) -> Features:
    cutoff = decision_cutoff(decision_date)
    eligible = [
        bar for bar in history
        if bar.trading_date < decision_date and bar.available_at <= cutoff
    ]
    if len(eligible) < 11:
        raise ValueError("at least 11 completed prior bars are required")
    closes = np.array([bar.close for bar in eligible], dtype=float)
    values = _values_from_closes(closes)
    assert all(source_date < decision_date for source_date in (bar.trading_date for bar in eligible[-11:]))
    return Features(
        symbol=eligible[-1].symbol,
        decision_date=decision_date,
        values=values,
        source_dates=tuple(bar.trading_date for bar in eligible[-11:]),
    )


def training_rows(history: list[PriceBar], before: date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = [
        bar for bar in history
        if bar.trading_date < before and bar.available_at <= decision_cutoff(before)
    ]
    rows: list[tuple[float, ...]] = []
    labels: list[float] = []
    returns: list[float] = []
    closes = np.asarray([bar.close for bar in eligible], dtype=float)
    for index in range(11, len(eligible)):
        target = eligible[index]
        ret = target.close / target.open - 1
        rows.append(_values_from_closes(closes[:index]))
        labels.append(1.0 if ret > 0 else 0.0)
        returns.append(ret)
    if not rows:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0), np.empty(0)
    return np.asarray(rows), np.asarray(labels), np.asarray(returns)
