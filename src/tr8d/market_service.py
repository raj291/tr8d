from __future__ import annotations

import json
import mimetypes
import threading
import time
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import parse_qs, urlparse

from .market_data import MarketDataError, MarketDataProvider, normalize_symbols


class QuoteCache:
    def __init__(self, ttl_seconds: int = 10) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._values: dict[tuple[str, ...], tuple[float, list[dict[str, object]]]] = {}

    def get(self, key: tuple[str, ...]) -> list[dict[str, object]] | None:
        with self._lock:
            item = self._values.get(key)
            if item and time.monotonic() - item[0] < self.ttl_seconds:
                return item[1]
        return None

    def put(self, key: tuple[str, ...], value: list[dict[str, object]]) -> None:
        with self._lock:
            self._values[key] = (time.monotonic(), value)


def make_handler(provider: MarketDataProvider) -> type[BaseHTTPRequestHandler]:
    cache = QuoteCache()
    asset_root = files("tr8d").joinpath("dashboard")

    class Handler(BaseHTTPRequestHandler):
        server_version = "TR8DMarketData/0.5"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _asset(self, name: str) -> None:
            target = asset_root.joinpath(name)
            try:
                body = target.read_bytes()
            except (FileNotFoundError, IsADirectoryError):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            try:
                if parsed.path == "/api/health":
                    self._json({"status": "ok", "service": "tr8d-market-data"})
                elif parsed.path == "/api/market/status":
                    self._json({**provider.metadata, "server_time": datetime.now(UTC).isoformat()})
                elif parsed.path == "/api/quotes":
                    symbols = normalize_symbols(params.get("symbols", ["AAPL,MSFT,NVDA,SPY,QQQ"])[0].split(","))
                    quotes = cache.get(symbols)
                    cached = quotes is not None
                    if quotes is None:
                        quotes = [quote.as_dict() for quote in provider.quotes(symbols)]
                        cache.put(symbols, quotes)
                    self._json({"quotes": quotes, "cached": cached, "feed": provider.metadata})
                elif parsed.path.startswith("/api/history/"):
                    symbol = normalize_symbols([parsed.path.rsplit("/", 1)[-1]])[0]
                    end = datetime.fromisoformat(params.get("end", [datetime.now(UTC).isoformat()])[0].replace("Z", "+00:00"))
                    days = min(max(int(params.get("days", ["90"])[0]), 1), 1825)
                    limit = min(max(int(params.get("limit", [str(days)])[0]), 1), 10000)
                    timeframe = params.get("timeframe", ["1Day"])[0]
                    bars = provider.history(
                        symbol, timeframe=timeframe, start=end - timedelta(days=days),
                        end=end, limit=limit,
                    )
                    self._json({
                        "symbol": symbol, "timeframe": timeframe,
                        "bars": [bar.as_dict() for bar in bars], "feed": provider.metadata,
                    })
                elif parsed.path == "/" or parsed.path == "/index.html":
                    self._asset("index.html")
                elif parsed.path in {"/app.js", "/styles.css"}:
                    self._asset(parsed.path[1:])
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (MarketDataError, ValueError, OverflowError) as error:
                status = HTTPStatus.BAD_GATEWAY if isinstance(error, MarketDataError) else HTTPStatus.BAD_REQUEST
                self._json({"error": str(error)}, status)

    return Handler


def serve(provider: MarketDataProvider, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(provider))
    print(f"TR8D dashboard: http://{host}:{server.server_port}")
    print(f"Feed: {provider.metadata['notice']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
