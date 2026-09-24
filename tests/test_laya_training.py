import json
import tempfile
import unittest
from pathlib import Path

from tr8d.data import synthetic_prices
from tr8d.laya_training import (
    build_laya_examples,
    load_laya_split,
    temporal_split,
    write_laya_dataset,
)


class LayaTrainingDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = build_laya_examples(synthetic_prices(120, seed=11))

    def test_examples_use_only_prior_features_and_realized_labels(self):
        self.assertGreater(len(self.examples), 400)
        expected_state_keys = {
            "instruction", "symbol", "bull_probability", "expected_return",
            "cash", "held_quantity", "evidence",
        }
        for example in self.examples:
            self.assertEqual(set(example.state), expected_state_keys)
            self.assertTrue(all(day < example.decision_date for day in example.feature_source_dates))
            direction = (
                "A" if example.realized_return > 0.0025
                else "B" if example.realized_return < -0.0025
                else "C"
            )
            expected = direction if example.portfolio_context == "holding" else (
                "A" if direction == "A" else "C"
            )
            self.assertEqual(example.target_label, expected)
            self.assertEqual(example.target_probabilities[expected], 1.0)
            self.assertEqual(example.prediction_source, "expanding_logistic")
            self.assertEqual(
                example.state["held_quantity"] > 0,
                example.portfolio_context == "holding",
            )

    def test_temporal_splits_are_strictly_ordered(self):
        splits = temporal_split(self.examples)
        train_dates = {row.decision_date for row in splits["train"]}
        calibration_dates = {row.decision_date for row in splits["calibration"]}
        test_dates = {row.decision_date for row in splits["test"]}
        self.assertTrue(train_dates.isdisjoint(calibration_dates | test_dates))
        self.assertTrue(calibration_dates.isdisjoint(test_dates))
        self.assertLess(max(train_dates), min(calibration_dates))
        self.assertLess(max(calibration_dates), min(test_dates))

    def test_round_trip_writes_manifest_and_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_laya_dataset(
                self.examples, directory, source="test-fixture", synthetic=True,
            )
            loaded = load_laya_split(directory, "test")
            stored = json.loads(Path(directory, "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, manifest)
            self.assertEqual(manifest["examples"], len(self.examples))
            self.assertEqual(manifest["source"], "test-fixture")
            self.assertTrue(manifest["point_in_time_features"])
            self.assertEqual(manifest["prediction_sources"], ["expanding_logistic"])
            self.assertEqual(
                manifest["portfolio_context_counts"],
                {"flat": len(self.examples) // 2, "holding": len(self.examples) // 2},
            )
            self.assertTrue(manifest["rlcd_compatible_exports"])
            self.assertEqual(len(loaded), manifest["split_counts"]["test"])
            self.assertEqual(loaded[0].feature_source_dates, self.examples[-len(loaded)].feature_source_dates)
            rlcd = json.loads(
                Path(directory, "train_rlcd.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(set(json.loads(rlcd["gold"])["action"]["probabilities"]), {"A", "B", "C"})


if __name__ == "__main__":
    unittest.main()
