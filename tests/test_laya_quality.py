import unittest

import numpy as np

from tr8d.laya_quality import (
    classification_report,
    fit_selective_policy,
    promotion_gate,
    selective_report,
)


class LayaQualityTests(unittest.TestCase):
    def setUp(self):
        self.probabilities = np.array([
            [0.95, 0.03, 0.02],
            [0.02, 0.94, 0.04],
            [0.03, 0.02, 0.95],
            [0.91, 0.05, 0.04],
            [0.40, 0.35, 0.25],
            [0.38, 0.34, 0.28],
            [0.36, 0.33, 0.31],
            [0.35, 0.34, 0.31],
            [0.34, 0.33, 0.33],
            [0.34, 0.33, 0.33],
        ])
        self.labels = np.array([0, 1, 2, 0, 0, 2, 1, 2, 1, 2])

    def test_classification_report_exposes_baselines_and_class_metrics(self):
        report = classification_report(self.probabilities, self.labels)
        self.assertEqual(report["examples"], 10)
        self.assertEqual(set(report["per_class"]), {"A", "B", "C"})
        self.assertEqual(len(report["confusion_matrix"]), 3)
        self.assertGreater(report["accuracy"], report["majority_baseline_accuracy"])

    def test_selective_threshold_is_fit_without_using_test_results(self):
        policy = fit_selective_policy(
            self.probabilities, self.labels, target_accuracy=0.85, minimum_coverage=0.30,
        )
        self.assertTrue(policy.target_met)
        self.assertGreaterEqual(policy.calibration_accuracy, 0.85)
        self.assertGreaterEqual(policy.calibration_coverage, 0.30)
        held_out = selective_report(self.probabilities, self.labels, policy.threshold)
        self.assertEqual(held_out["threshold"], policy.threshold)

    def test_promotion_gate_refuses_small_or_unproven_test_sets(self):
        overall = classification_report(self.probabilities, self.labels)
        selective = selective_report(self.probabilities, self.labels, 0.90)
        gate = promotion_gate(
            overall, selective, baseline_brier=1.0,
            target_accuracy=0.85, minimum_coverage=0.10, minimum_test_examples=500,
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["enough_untouched_test_examples"])


if __name__ == "__main__":
    unittest.main()
