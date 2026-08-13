from __future__ import annotations

import csv
import math
import random
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from .domain import PriceBar

UTC = timezone.utc


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def load_price_csv(path: str | Path) -> list[PriceBar]:
    bars: list[PriceBar] = []
    seen: set[tuple[str, date]] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "date", "open", "close"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain {sorted(required)}")
        for line_no, row in enumerate(reader, start=2):
            trading_date = date.fromisoformat(row["date"])
            key = (row["symbol"].strip().upper(), trading_date)
            if key in seen:
                raise ValueError(f"duplicate price row at line {line_no}: {key}")
            seen.add(key)
            available = row.get("available_at", "").strip()
            available_at = (
                _parse_datetime(available)
                if available
                else datetime.combine(trading_date, time(21, 0), tzinfo=UTC)
            )
            if available_at.date() < trading_date:
                raise ValueError(f"available_at precedes trading date at line {line_no}")
            bars.append(
                PriceBar(
                    symbol=key[0],
                    trading_date=trading_date,
                    open=float(row["open"]),
                    close=float(row["close"]),
                    available_at=available_at,
                )
            )
    return sorted(bars, key=lambda bar: (bar.trading_date, bar.symbol))


def synthetic_prices(days: int = 800, seed: int = 7) -> list[PriceBar]:
    """Create deterministic weekday bars for an offline smoke test."""
    if days < 120:
        raise ValueError("demo requires at least 120 days")
    rng = random.Random(seed)
    symbols = {"XLK": (100.0, 0.00035), "XLE": (55.0, 0.00015), "XLF": (35.0, 0.00022)}
    current = date(2020, 1, 2)
    bars: list[PriceBar] = []
    generated = 0
    previous_close = {symbol: start for symbol, (start, _) in symbols.items()}
    phase = {symbol: rng.random() * math.tau for symbol in symbols}
    while generated < days:
        if current.weekday() < 5:
            market_shock = rng.gauss(0, 0.004)
            for index, (symbol, (_, drift)) in enumerate(symbols.items()):
                gap = rng.gauss(0, 0.002)
                open_price = previous_close[symbol] * (1 + gap)
                cycle = 0.0015 * math.sin(generated / 19 + phase[symbol])
                sector_noise = rng.gauss(0, 0.006 + index * 0.0005)
                intraday_return = drift + 0.45 * market_shock + cycle + sector_noise
                close_price = open_price * (1 + intraday_return)
                bars.append(
                    PriceBar(
                        symbol=symbol,
                        trading_date=current,
                        open=round(open_price, 6),
                        close=round(close_price, 6),
                        available_at=datetime.combine(current, time(21, 0), tzinfo=UTC),
                    )
                )
                previous_close[symbol] = close_price
            generated += 1
        current += timedelta(days=1)
    return bars
