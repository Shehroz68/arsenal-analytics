"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DB_PATH = PROCESSED / "epl.db"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

for p in (RAW, PROCESSED, FIGURES, MODELS):
    p.mkdir(parents=True, exist_ok=True)

TEAM = "Arsenal"

# StatsBomb Open Data: Premier League seasons (checked in Step 1)
STATSBOMB_EPL = [
    {"competition_id": 2, "season_id": 27, "season": "1516"},  # full 2015/16 season
    {"competition_id": 2, "season_id": 44, "season": "0304"},  # Arsenal's 38 matches, 2003/04
]

# Arsenal managerial eras (by match date; interim spells folded into the era they ended)
ERAS = [
    ("Wenger I (Highbury)", "2000-07-01", "2006-06-30"),
    ("Wenger II (Emirates)", "2006-07-01", "2018-06-30"),
    ("Emery", "2018-07-01", "2019-12-19"),
    ("Arteta", "2019-12-20", "2099-12-31"),
]

# First season scored out-of-sample in the walk-forward evaluation.
FIRST_TEST_SEASON = "0506"
