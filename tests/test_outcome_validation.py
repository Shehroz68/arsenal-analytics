import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import one_hot, log_loss_per_match, brier_per_match, rps_per_match


class OutcomeTests(unittest.TestCase):
    def test_invalid_outcomes_rejected_by_every_metric(self):
        for y in ([-1], [3], [0.5], [True], [[0]], [np.nan]):
            for metric in (log_loss_per_match, brier_per_match, rps_per_match):
                with self.subTest(y=y, metric=metric.__name__):
                    with self.assertRaises(ValueError):
                        metric(np.array([[0.2, 0.3, 0.5]]), y)

    def test_valid_integer_types_and_lists(self):
        for y in ([0, 1, 2], np.array([0, 1, 2], dtype=np.uint8)):
            np.testing.assert_array_equal(one_hot(y), np.eye(3))
            np.testing.assert_allclose(log_loss_per_match(np.eye(3), y), 0)

    def test_empty_one_hot(self):
        self.assertEqual(one_hot([]).shape, (0, 3))
