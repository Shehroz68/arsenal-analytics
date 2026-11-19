"""Check result validation against an isolated in-memory database."""
from pathlib import Path
import sqlite3
import unittest

SQL = (Path(__file__).resolve().parents[1] / "sql/validate.sql").read_text()
QUERY = next(query for query in SQL.split(";") if "AS bad_results" in query)


class ResultValidationTests(unittest.TestCase):
    def count_bad(self, rows):
        with sqlite3.connect(":memory:") as con:
            con.execute("CREATE TABLE matches (fthg INTEGER, ftag INTEGER, ftr TEXT)")
            con.executemany("INSERT INTO matches VALUES (?, ?, ?)", rows)
            return con.execute(QUERY).fetchone()[0]

    def test_valid_results(self):
        self.assertEqual(self.count_bad([(2, 1, "H"), (0, 0, "D"), (0, 1, "A")]), 0)

    def test_mismatched_results(self):
        self.assertEqual(self.count_bad([(2, 1, "A"), (0, 0, "H"), (0, 1, "D")]), 3)

    def test_missing_and_unknown_results(self):
        self.assertEqual(self.count_bad([(1, 0, None), (1, 0, ""), (1, 0, "X")]), 3)

    def test_missing_scores(self):
        self.assertEqual(self.count_bad([(None, 0, "H"), (1, None, "H"), (None, None, "D")]), 3)
