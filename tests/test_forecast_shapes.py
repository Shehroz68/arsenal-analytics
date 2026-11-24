import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import log_loss_per_match, brier_per_match, rps_per_match, summary


class ForecastShapeTests(unittest.TestCase):
    def test_mismatched_shapes_cannot_broadcast(self):
        for p in ([[0.2, 0.3, 0.5]], [[0.2, 0.8], [0.3, 0.7]], [0.2, 0.3, 0.5]):
            for metric in (log_loss_per_match, brier_per_match, rps_per_match, summary):
                with self.subTest(p=p, metric=metric.__name__):
                    with self.assertRaises(ValueError):
                        metric(p, [0, 2])

    def test_lists_and_arrays_produce_same_summary(self):
        p = [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]
        self.assertEqual(summary(p, [0, 2]), summary(np.array(p), np.array([0, 2])))
        self.assertEqual(summary(p, [0, 2])["accuracy"], 1)


class ProbabilityValueTests(unittest.TestCase):
    def test_invalid_probabilities_rejected(self):
        for row in ([np.nan, 0.5, 0.5], [np.inf, 0, 0], [-0.1, 0.5, 0.6],
                    [1.1, 0, 0], [0.2, 0.2, 0.2], ["0.2", "0.3", "0.5"]):
            for metric in (log_loss_per_match, brier_per_match, rps_per_match, summary):
                with self.subTest(row=row, metric=metric.__name__):
                    with self.assertRaises(ValueError):
                        metric([row], [0])

    def test_float32_roundoff_accepted(self):
        result = summary(np.full((1, 3), 1 / 3, dtype=np.float32), [0])
        self.assertAlmostEqual(result["brier"], 2 / 3)

    def test_certain_predictions_remain_valid(self):
        result = summary([[1, 0, 0]], [0])
        self.assertEqual(result["brier"], 0)
        self.assertEqual(result["log_loss"], 0)
