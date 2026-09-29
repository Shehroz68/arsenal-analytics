"""Step 5 - pre-match features with no data leakage.

Every feature for a match uses only information available before kick-off:
  * rolling averages over each team's previous 5 and 10 league matches (shift(1) before rolling)
  * season-to-date points per game and goal difference per game (also shifted)
  * days of rest since the team's previous league match
  * pre-match Elo ratings from Step 3
Rolling windows run across season boundaries on purpose (form carries over the summer);
`matches_this_season` lets the model learn to trust early-season numbers less.

Output table: features (one row per match; home_*, away_* and diff_* columns)
"""
import sqlite3

import numpy as np
import pandas as pd

from config import DB_PATH

STATS = ["gf", "ga", "shots_f", "shots_a", "sot_f", "sot_a", "corners_f", "corners_a", "points"]
WINDOWS = [5, 10]


def team_features(tm: pd.DataFrame) -> pd.DataFrame:
    tm = tm.sort_values(["team", "date", "match_id"]).copy()
    g = tm.groupby("team", group_keys=False)
    feats = pd.DataFrame(index=tm.index)
    for w in WINDOWS:
        for s in STATS:
            feats[f"{s}_r{w}"] = g[s].transform(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
    # shot-quality proxies over the last 10
    feats["sot_ratio_r10"] = feats["sot_f_r10"] / (feats["sot_f_r10"] + feats["sot_a_r10"])
    feats["conv_r10"] = feats["gf_r10"] / feats["sot_f_r10"].replace(0, np.nan)

    gs = tm.groupby(["team", "season"], group_keys=False)
    feats["matches_this_season"] = gs.cumcount()
    feats["ppg_season"] = gs["points"].transform(lambda x: x.shift(1).expanding().mean())
    tm["gd"] = tm["gf"] - tm["ga"]
    feats["gdpg_season"] = tm.groupby(["team", "season"])["gd"].transform(lambda x: x.shift(1).expanding().mean())

    dates = pd.to_datetime(tm["date"])
    feats["rest_days"] = dates.groupby(tm["team"]).diff().dt.days.clip(upper=21)

    return pd.concat([tm[["match_id", "team", "is_home"]], feats], axis=1)


def build(con) -> pd.DataFrame:
    tm = pd.read_sql("SELECT * FROM team_matches", con)
    tf = team_features(tm)
    feat_cols = [c for c in tf.columns if c not in ("match_id", "team", "is_home")]

    home = tf[tf.is_home == 1].drop(columns=["team", "is_home"]).add_prefix("home_").rename(columns={"home_match_id": "match_id"})
    away = tf[tf.is_home == 0].drop(columns=["team", "is_home"]).add_prefix("away_").rename(columns={"away_match_id": "match_id"})
    m = pd.read_sql("SELECT match_id, season, date, hometeam, awayteam, ftr FROM matches", con)
    elo = pd.read_sql("SELECT match_id, elo_home, elo_away, elo_diff, elo_exp_home FROM elo", con)
    f = m.merge(home, on="match_id").merge(away, on="match_id").merge(elo, on="match_id")
    for c in feat_cols:
        if c != "matches_this_season":
            f[f"diff_{c}"] = f[f"home_{c}"] - f[f"away_{c}"]
    return f.sort_values(["date", "match_id"]).reset_index(drop=True)


def main():
    con = sqlite3.connect(DB_PATH)
    f = build(con)
    f.to_sql("features", con, if_exists="replace", index=False)
    print(f"features: {f.shape[0]} matches x {f.shape[1]} columns")
    con.close()


if __name__ == "__main__":
    main()
