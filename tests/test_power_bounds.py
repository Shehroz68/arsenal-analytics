import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market import power


class PowerBoundsTests(unittest.TestCase):
    def test_roots_outside_old_bracket(self):
        for odds in ([[100, 100, 100]], [[1.01, 1.01, 1.01]]):
            np.testing.assert_allclose(power(np.array(odds)), [[1 / 3] * 3])

    def test_invalid_decimal_odds(self):
        for value in (0, -1, 1, np.inf):
            with self.subTest(value=value):
                with np.errstate(divide="ignore", invalid="ignore"):
                    with self.assertRaises(ValueError):
                        power(np.array([[value, 3, 4]], dtype=float))

    def test_missing_odds_keep_nan_row(self):
        self.assertTrue(np.isnan(power(np.array([[np.nan, 3, 4]]))).all())
