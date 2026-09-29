"""Step 4a - pull every shot (plus key-pass context) from StatsBomb Open Data EPL matches.

Output: data/raw/statsbomb/shots.parquet (one row per shot) and matches.parquet.
Downloads are cached per match, so re-running is cheap.
"""
import warnings

import pandas as pd
from statsbombpy import sb

from config import RAW, STATSBOMB_EPL

warnings.filterwarnings("ignore")
OUT = RAW / "statsbomb"
CACHE = OUT / "events"
CACHE.mkdir(parents=True, exist_ok=True)

SHOT_COLS = [
    "id", "match_id", "period", "minute", "second", "team", "player", "location",
    "play_pattern", "under_pressure", "shot_type", "shot_body_part", "shot_technique",
    "shot_outcome", "shot_first_time", "shot_one_on_one", "shot_open_goal",
    "shot_deflected", "shot_key_pass_id", "shot_statsbomb_xg",
]
PASS_COLS = ["id", "pass_cross", "pass_through_ball", "pass_cut_back", "pass_switch",
             "pass_height", "pass_type"]


def match_shots(match_id: int) -> pd.DataFrame:
    f = CACHE / f"{match_id}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    ev = sb.events(match_id=match_id)
    shots = ev[ev["type"] == "Shot"].reindex(columns=SHOT_COLS)
    passes = ev[ev["type"] == "Pass"].reindex(columns=PASS_COLS).add_prefix("kp_")
    shots = shots.merge(passes, left_on="shot_key_pass_id", right_on="kp_id", how="left")
    shots["x"] = shots["location"].str[0]
    shots["y"] = shots["location"].str[1]
    shots = shots.drop(columns=["location", "kp_id"])
    # parquet dislikes mixed object columns
    for c in shots.columns:
        if shots[c].dtype == object:
            shots[c] = shots[c].astype("string")
    shots.to_parquet(f, index=False)
    return shots


def main():
    all_matches, all_shots = [], []
    for s in STATSBOMB_EPL:
        m = sb.matches(competition_id=s["competition_id"], season_id=s["season_id"])
        m["season"] = s["season"]
        all_matches.append(m)
        for i, mid in enumerate(m["match_id"]):
            all_shots.append(match_shots(int(mid)).assign(season=s["season"]))
            if i % 50 == 0:
                print(f"{s['season']}: {i}/{len(m)} matches")
    matches = pd.concat(all_matches, ignore_index=True)
    keep = ["match_id", "season", "match_date", "home_team", "away_team", "home_score", "away_score"]
    matches[keep].to_parquet(OUT / "matches.parquet", index=False)
    shots = pd.concat(all_shots, ignore_index=True)
    shots.to_parquet(OUT / "shots.parquet", index=False)
    print(f"{len(matches)} matches, {len(shots)} shots")


if __name__ == "__main__":
    main()
