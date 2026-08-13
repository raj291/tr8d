import unittest
from datetime import date, timedelta

import numpy as np

from tr8d.evaluation import PlattCalibrator, expanding_folds


class EvaluationTests(unittest.TestCase):
    def test_folds_are_strictly_temporal(self):
        dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(240)]
        folds = expanding_folds(dates, min_train_days=120, calibration_days=40, test_days=40)
        self.assertEqual(len(folds), 2)
        for fold in folds:
            self.assertLess(max(fold.train_dates), min(fold.calibration_dates))
            self.assertLess(max(fold.calibration_dates), min(fold.test_dates))
            self.assertFalse(set(fold.train_dates) & set(fold.test_dates))

    def test_platt_calibrator_returns_probabilities(self):
        raw = np.linspace(0.1, 0.9, 20)
        labels = np.asarray([0] * 10 + [1] * 10, dtype=float)
        calibrated = PlattCalibrator().fit(raw, labels).predict(raw)
        self.assertTrue(np.all(calibrated >= 0))
        self.assertTrue(np.all(calibrated <= 1))
        self.assertGreater(calibrated[-1], calibrated[0])


if __name__ == "__main__":
    unittest.main()
