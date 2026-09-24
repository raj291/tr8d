from __future__ import annotations

import json
import threading
import unittest
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

from tr8d.market_data import DemoMarketDataProvider, normalize_symbols
from tr8d.market_service import make_handler


class MarketDataTests(unittest.TestCase):
    def test_symbol_normalization_deduplicates_and_rejects_unsafe_values(self) -> None:
        self.assertEqual(normalize_symbols([" aapl ", "MSFT", "AAPL"]), ("AAPL", "MSFT"))
        with self.assertRaises(ValueError):
            normalize_symbols(["AAPL&feed=sip"])
        with self.assertRaises(ValueError):
            normalize_symbols([])

    def test_demo_data_is_deterministic_and_truthfully_labeled(self) -> None:
        provider = DemoMarketDataProvider()
        first = provider.quotes(("AAPL",))[0]
        second = provider.quotes(("AAPL",))[0]
        self.assertEqual(first.price, second.price)
        self.assertTrue(provider.metadata["synthetic"])
        self.assertFalse(provider.metadata["live"])
        bars = provider.history(
            "AAPL", timeframe="1Day", start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 3, 1, tzinfo=UTC), limit=30,
        )
        self.assertEqual(len(bars), 30)
        self.assertTrue(all(bar.low <= bar.close <= bar.high for bar in bars))

    def test_http_service_exposes_health_quotes_history_and_assets(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(DemoMarketDataProvider()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/health") as response:
                self.assertEqual(json.load(response)["status"], "ok")
            with urlopen(base + "/api/quotes?symbols=AAPL,MSFT") as response:
                payload = json.load(response)
                self.assertEqual(len(payload["quotes"]), 2)
                self.assertEqual(payload["feed"]["feed"], "synthetic")
            with urlopen(base + "/api/history/AAPL?days=10&limit=10") as response:
                self.assertEqual(len(json.load(response)["bars"]), 10)
            with urlopen(base + "/") as response:
                self.assertIn(b"TR8D Market Desk", response.read())
            with self.assertRaises(HTTPError) as context:
                urlopen(base + "/api/history/AAPL%26feed=sip")
            self.assertEqual(context.exception.code, 400)
            context.exception.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
