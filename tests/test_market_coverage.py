import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market import market_probs, PCOLS


class MarketCoverageTests(unittest.TestCase):
    def test_missing_bookmaker_columns_return_empty_schema(self):
        for frame in (pd.DataFrame(), pd.DataFrame({"match_id": [1], "season": ["0001"], "b365h": [2.0]})):
            result = market_probs(frame)
            self.assertTrue(result.empty)
            self.assertEqual(list(result.columns), ["match_id", "season", "model", *PCOLS, "overround"])

    def test_single_bookmaker_still_produces_both_methods(self):
        frame = pd.DataFrame({"match_id": [1], "season": ["0001"],
                              "b365h": [2.1], "b365d": [3.4], "b365a": [3.6]})
        result = market_probs(frame)
        self.assertEqual(set(result.model), {"market_b365_power", "market_b365_proportional"})
        self.assertEqual(result.match_id.tolist(), [1, 1])
        np.testing.assert_allclose(result[PCOLS].sum(axis=1), 1)
