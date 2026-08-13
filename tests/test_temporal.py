import unittest
from dataclasses import replace
from datetime import timedelta

from tr8d.data import synthetic_prices
from tr8d.features import bars_by_symbol, build_features, training_rows


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.history = bars_by_symbol(synthetic_prices(150))["XLK"]

    def test_snapshot_excludes_decision_day(self):
        decision_date = self.history[100].trading_date
        features = build_features(self.history, decision_date)
        self.assertTrue(all(item < decision_date for item in features.source_dates))

    def test_training_target_excludes_decision_day(self):
        decision_date = self.history[100].trading_date
        x, y, returns = training_rows(self.history, decision_date)
        self.assertEqual(len(x), 89)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(y), len(returns))

    def test_future_available_prior_bar_is_excluded(self):
        decision_date = self.history[100].trading_date
        tainted = list(self.history)
        tainted[99] = replace(tainted[99], available_at=tainted[99].available_at + timedelta(days=10))
        features = build_features(tainted, decision_date)
        self.assertNotIn(tainted[99].trading_date, features.source_dates)


if __name__ == "__main__":
    unittest.main()
