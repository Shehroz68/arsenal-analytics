import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from elo import expected


class EloStabilityTests(unittest.TestCase):
    def test_extreme_rating_gaps_do_not_overflow(self):
        with np.errstate(over="raise", invalid="raise"):
            self.assertEqual(expected(-1000000, 1000000, 0), 0)
            self.assertEqual(expected(1000000, -1000000, 0), 1)

    def test_matches_original_formula_for_normal_ratings(self):
        for gap in (-800, -400, -60, 0, 60, 400, 800):
            original = 1 / (1 + 10 ** (-gap / 400))
            self.assertAlmostEqual(expected(1500 + gap, 1500, 0), original, places=14)

    def test_symmetry_with_home_advantage(self):
        self.assertAlmostEqual(expected(1600, 1400, 60) + expected(1400, 1600, -60), 1)
