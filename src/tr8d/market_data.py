from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class MarketDataError(RuntimeError):
    """A safe, user-facing market-data failure."""


def normalize_symbols(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    symbols = tuple(dict.fromkeys(value.strip().upper() for value in values if value.strip()))
    if not symbols or len(symbols) > 30:
        raise ValueError("provide between 1 and 30 symbols")
    if any(not symbol.replace(".", "").replace("-", "").isalnum() for symbol in symbols):
        raise ValueError("symbols may contain only letters, numbers, dots, and hyphens")
    return symbols


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float
    change: float
    change_percent: float
    bid: float | None
    ask: float | None
    volume: int
    as_of: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Bar:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class MarketDataProvider(Protocol):
    @property
    def metadata(self) -> dict[str, object]: ...

    def quotes(self, symbols: tuple[str, ...]) -> list[Quote]: ...

    def history(
        self, symbol: str, *, timeframe: str, start: datetime, end: datetime, limit: int
    ) -> list[Bar]: ...


class AlpacaMarketDataProvider:
    """Read-only Alpaca equities client. It contains no order/trading methods."""

    BASE_URL = "https://data.alpaca.markets"
    ALLOWED_FEEDS = {"iex", "sip", "delayed_sip"}
    ALLOWED_TIMEFRAMES = {"1Min", "5Min", "15Min", "1Hour", "1Day"}

    def __init__(
        self, key_id: str | None = None, secret_key: str | None = None,
        *, feed: str | None = None, timeout: float = 12.0,
    ) -> None:
        self.key_id = key_id or os.environ.get("ALPACA_API_KEY_ID", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_API_SECRET_KEY", "")
        self.feed = feed or os.environ.get("ALPACA_DATA_FEED", "iex")
        self.timeout = timeout
        if not self.key_id or not self.secret_key:
            raise MarketDataError(
                "Alpaca credentials are missing; set ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY or start with --provider demo"
            )
        if self.feed not in self.ALLOWED_FEEDS:
            raise MarketDataError(f"unsupported Alpaca feed: {self.feed}")

    @property
    def metadata(self) -> dict[str, object]:
        consolidated = self.feed in {"sip", "delayed_sip"}
        return {
            "provider": "alpaca",
            "feed": self.feed,
            "live": self.feed in {"iex", "sip"},
            "synthetic": False,
            "consolidated": consolidated,
            "coverage": "All US exchanges (SIP)" if consolidated else "IEX exchange only",
            "notice": (
                "Consolidated SIP market data"
                if consolidated else "Free-tier real-time IEX data; not the consolidated US market"
            ),
        }

    def _get(self, path: str, params: dict[str, object]) -> dict[str, object]:
        query = urlencode({key: value for key, value in params.items() if value is not None})
        request = Request(
            f"{self.BASE_URL}{path}?{query}",
            headers={
                "APCA-API-KEY-ID": self.key_id,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Accept": "application/json",
                "User-Agent": "tr8d-market-data/0.5",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise MarketDataError(f"Alpaca returned HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise MarketDataError(f"Alpaca request failed: {error}") from error

    def quotes(self, symbols: tuple[str, ...]) -> list[Quote]:
        payload = self._get(
            "/v2/stocks/snapshots", {"symbols": ",".join(symbols), "feed": self.feed},
        )
        snapshots = payload.get("snapshots", payload)
        output: list[Quote] = []
        for symbol in symbols:
            item = snapshots.get(symbol, {}) if isinstance(snapshots, dict) else {}
            trade, quote = item.get("latestTrade", {}), item.get("latestQuote", {})
            daily, previous = item.get("dailyBar", {}), item.get("prevDailyBar", {})
            price = float(trade.get("p") or daily.get("c") or 0)
            prior = float(previous.get("c") or price)
            change = price - prior
            output.append(Quote(
                symbol=symbol, price=round(price, 6), change=round(change, 6),
                change_percent=round((change / prior * 100) if prior else 0, 4),
                bid=float(quote["bp"]) if quote.get("bp") is not None else None,
                ask=float(quote["ap"]) if quote.get("ap") is not None else None,
                volume=int(daily.get("v") or 0),
                as_of=str(trade.get("t") or daily.get("t") or datetime.now(UTC).isoformat()),
            ))
        return output

    def history(
        self, symbol: str, *, timeframe: str, start: datetime, end: datetime, limit: int
    ) -> list[Bar]:
        if timeframe not in self.ALLOWED_TIMEFRAMES:
            raise ValueError(f"unsupported timeframe: {timeframe}")
        payload = self._get(f"/v2/stocks/{symbol}/bars", {
            "timeframe": timeframe, "start": start.isoformat(), "end": end.isoformat(),
            "limit": min(max(limit, 1), 10000), "adjustment": "all", "feed": self.feed,
            "sort": "asc",
        })
        return [Bar(
            timestamp=str(item["t"]), open=float(item["o"]), high=float(item["h"]),
            low=float(item["l"]), close=float(item["c"]), volume=int(item["v"]),
        ) for item in payload.get("bars", [])]


class DemoMarketDataProvider:
    """Deterministic synthetic provider for local UI development and tests."""

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "provider": "demo", "feed": "synthetic", "live": False,
            "synthetic": True, "consolidated": False, "coverage": "Demo symbols only",
            "notice": "Synthetic demo data — never use it for trading decisions",
        }

    @staticmethod
    def _series(symbol: str, points: int) -> list[float]:
        rng = random.Random(f"tr8d:{symbol}")
        base = 80 + sum(ord(char) for char in symbol) % 180
        values = [float(base)]
        for index in range(1, points):
            cycle = math.sin(index / 7) * 0.003
            values.append(values[-1] * (1 + 0.0007 + cycle + rng.gauss(0, 0.008)))
        return values

    def quotes(self, symbols: tuple[str, ...]) -> list[Quote]:
        now = datetime.now(UTC).replace(microsecond=0).isoformat()
        output = []
        for symbol in symbols:
            prices = self._series(symbol, 92)
            price, prior = prices[-1], prices[-2]
            output.append(Quote(
                symbol, round(price, 2), round(price - prior, 2),
                round((price / prior - 1) * 100, 2), round(price - 0.03, 2),
                round(price + 0.03, 2), 1_000_000 + sum(map(ord, symbol)) * 10_000, now,
            ))
        return output

    def history(
        self, symbol: str, *, timeframe: str, start: datetime, end: datetime, limit: int
    ) -> list[Bar]:
        count = min(max(limit, 1), 365)
        prices = self._series(symbol, count + 1)
        first = end - timedelta(days=count - 1)
        return [Bar(
            timestamp=(first + timedelta(days=index)).replace(microsecond=0).isoformat(),
            open=round(prices[index] * 0.998, 2), high=round(prices[index] * 1.008, 2),
            low=round(prices[index] * 0.992, 2), close=round(prices[index], 2),
            volume=800_000 + index * 7_913,
        ) for index in range(count)]


def create_provider(name: str) -> MarketDataProvider:
    if name == "demo":
        return DemoMarketDataProvider()
    if name == "alpaca":
        return AlpacaMarketDataProvider()
    if name == "nyse":
        raise MarketDataError(
            "NYSE TOP is a Trade Operations Portal, not a free retail price feed. "
            "A licensed NYSE market-data product and its specific credentials are required."
        )
    raise MarketDataError(f"unknown provider: {name}")
