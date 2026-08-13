import json
import tempfile
import unittest
from pathlib import Path

from tr8d.data import load_stooq_csv, synthetic_prices, validate_price_panel
from tr8d.manifest import create_manifest
from tr8d.metrics import performance


class ManifestAndMetricsTests(unittest.TestCase):
    def test_manifest_is_stable_for_same_rows(self):
        bars = synthetic_prices(120)
        first = create_manifest(bars, "test")
        second = create_manifest(list(reversed(bars)), "test")
        self.assertEqual(first.content_sha256, second.content_sha256)
        self.assertEqual(first.version, second.version)
        with tempfile.TemporaryDirectory() as directory:
            path = first.write(directory)
            stored = json.loads(Path(path).read_text())
            self.assertEqual(stored["row_count"], 360)

    def test_performance_computes_drawdown_and_turnover(self):
        result = performance([10.0, 11.0, 9.9, 12.0], trades=3, traded_notional=5.0)
        self.assertAlmostEqual(result.total_return, 0.2)
        self.assertAlmostEqual(result.max_drawdown, -0.1)
        self.assertEqual(result.turnover, 0.5)

    def test_stooq_csv_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "xlk.csv"
            source.write_text("Date,Open,High,Low,Close,Volume\n2024-01-02,100,102,99,101,50\n")
            bars = load_stooq_csv(source, "xlk")
            self.assertEqual(bars[0].symbol, "XLK")
            self.assertEqual(bars[0].close, 101.0)

    def test_unaligned_symbol_calendars_fail_closed(self):
        bars = synthetic_prices(120)
        with self.assertRaisesRegex(ValueError, "aligned"):
            validate_price_panel(bars[:-1])


if __name__ == "__main__":
    unittest.main()
