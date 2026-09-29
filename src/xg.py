"""Step 4 - build an expected-goals (xG) model from raw StatsBomb shots.

Pipeline
  1. Geometry features from shot location: distance to goal centre, shooting angle (angle the
     goal mouth subtends), plus context: body part, shot type, play pattern, pressure, first
     time, one-on-one, open goal, and how the chance was created (cross / through ball / cut back).
  2. Penalties are handled separately (empirical conversion rate from training folds).
  3. Two models, logistic regression and XGBoost, evaluated with GroupKFold by match so shots
     from the same game never sit on both sides of a split.
  4. Benchmark against StatsBomb's own xG on the same shots.
  5. Aggregate to team-match xG and join onto football-data matches (table `match_xg`).

Outputs: data/processed/shots_xg.parquet, models/xg_logreg.joblib + xg_xgb.json, DB tables xg_metrics + match_xg,
         reports/figures/xg_calibration.png, reports/figures/arsenal_shotmap_*.png
"""
import sqlite3
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import DB_PATH, FIGURES, MODELS, PROCESSED, RAW

warnings.filterwarnings("ignore")

GOAL_X, GOAL_Y, POST_HALF_WIDTH = 120.0, 40.0, 4.0  # StatsBomb pitch is 120 x 80, goal is 8 yards wide

# StatsBomb -> football-data.co.uk team names (needed to join xG onto the matches table)
SB_TO_FD = {
    "AFC Bournemouth": "Bournemouth", "Leicester City": "Leicester", "Manchester City": "Man City",
    "Manchester United": "Man United", "Newcastle United": "Newcastle", "Norwich City": "Norwich",
    "Stoke City": "Stoke", "Swansea City": "Swansea", "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom", "West Ham United": "West Ham", "Birmingham City": "Birmingham",
    "Blackburn Rovers": "Blackburn", "Bolton Wanderers": "Bolton", "Charlton Athletic": "Charlton",
    "Leeds United": "Leeds", "Wolverhampton Wanderers": "Wolves",
}

NUM = ["distance", "angle", "distance_x_angle", "under_pressure", "first_time", "one_on_one",
       "open_goal", "deflected", "from_cross", "from_through_ball", "from_cut_back", "header"]
CAT = ["body_part", "shot_type", "play_pattern", "technique"]


def to_flag(s: pd.Series) -> pd.Series:
    return s.astype("string").str.lower().isin(["true", "1", "1.0"]).astype(int)


def engineer(shots: pd.DataFrame) -> pd.DataFrame:
    d = shots.copy()
    dx = GOAL_X - d.x
    dy = (d.y - GOAL_Y).abs()
    d["distance"] = np.hypot(dx, dy)
    # angle subtended by the two posts, in radians
    a1 = np.arctan2(GOAL_Y + POST_HALF_WIDTH - d.y, dx)
    a2 = np.arctan2(GOAL_Y - POST_HALF_WIDTH - d.y, dx)
    d["angle"] = (a1 - a2).abs()
    d["distance_x_angle"] = d.distance * d.angle
    d["under_pressure"] = to_flag(d.under_pressure)
    d["first_time"] = to_flag(d.shot_first_time)
    d["one_on_one"] = to_flag(d.shot_one_on_one)
    d["open_goal"] = to_flag(d.shot_open_goal)
    d["deflected"] = to_flag(d.shot_deflected)
    d["from_cross"] = to_flag(d.kp_pass_cross)
    d["from_through_ball"] = to_flag(d.kp_pass_through_ball)
    d["from_cut_back"] = to_flag(d.kp_pass_cut_back)
    d["header"] = (d.shot_body_part == "Head").astype(int)
    d["body_part"] = d.shot_body_part.fillna("Other")
    d["shot_type"] = d.shot_type.fillna("Open Play")
    d["play_pattern"] = d.play_pattern.fillna("Other")
    d["technique"] = d.shot_technique.fillna("Normal")
    d["goal"] = (d.shot_outcome == "Goal").astype(int)
    d["is_penalty"] = (d.shot_type == "Penalty").astype(int)
    return d


def logreg():
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))


def xgb():
    # shallow + heavily regularised: deeper trees overfit ~10k shots (checked with grouped CV)
    return XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=2, min_child_weight=30,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                         eval_metric="logloss", enable_categorical=True, tree_method="hist")


def as_xgb_frame(d: pd.DataFrame) -> pd.DataFrame:
    X = d[NUM + CAT].copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    return X


def cross_validate(d: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Out-of-fold predictions for both models; penalties get the fold's training conversion rate."""
    d = d.copy()
    d["xg_logreg"] = np.nan
    d["xg_xgb"] = np.nan
    for tr, te in GroupKFold(n_splits=n_splits).split(d, groups=d.match_id):
        train, test = d.iloc[tr], d.iloc[te]
        np_train = train[train.is_penalty == 0]
        pen_rate = train.loc[train.is_penalty == 1, "goal"].mean()

        lr = logreg().fit(np_train[NUM + CAT], np_train.goal)
        xb = xgb().fit(as_xgb_frame(np_train), np_train.goal)

        pen = test[test.is_penalty == 1].index
        npt = test[test.is_penalty == 0]
        d.loc[pen, ["xg_logreg", "xg_xgb"]] = pen_rate
        d.loc[npt.index, "xg_logreg"] = lr.predict_proba(npt[NUM + CAT])[:, 1]
        d.loc[npt.index, "xg_xgb"] = xb.predict_proba(as_xgb_frame(npt))[:, 1]
    return d


def metrics(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, name in [("xg_logreg", "Logistic regression"), ("xg_xgb", "XGBoost"),
                      ("shot_statsbomb_xg", "StatsBomb xG (benchmark)")]:
        p = d[col].clip(1e-6, 1 - 1e-6)
        rows.append({"model": name, "log_loss": log_loss(d.goal, p), "brier": brier_score_loss(d.goal, p),
                     "auc": roc_auc_score(d.goal, p), "total_xg": p.sum(), "actual_goals": d.goal.sum()})
    base = d.goal.mean()
    rows.append({"model": "Baseline (league conversion rate)", "log_loss": log_loss(d.goal, np.full(len(d), base)),
                 "brier": brier_score_loss(d.goal, np.full(len(d), base)), "auc": 0.5,
                 "total_xg": base * len(d), "actual_goals": d.goal.sum()})
    return pd.DataFrame(rows)


def calibration_plot(d: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    bins = np.array([0, .03, .06, .1, .15, .2, .3, .45, .6, 1.0])
    for col, label in [("xg_logreg", "Logistic"), ("xg_xgb", "XGBoost"), ("shot_statsbomb_xg", "StatsBomb")]:
        g = d.groupby(pd.cut(d[col], bins), observed=True).agg(pred=(col, "mean"), obs=("goal", "mean"))
        ax.plot(g.pred, g.obs, marker="o", label=label)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set(xlabel="Predicted xG", ylabel="Observed goal rate", title="xG calibration (out-of-fold)",
           xlim=(0, .8), ylim=(0, .8))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "xg_calibration.png", dpi=150)
    plt.close(fig)


def shot_map(d: pd.DataFrame, season: str):
    s = d[(d.team == "Arsenal") & (d.season == season) & (d.is_penalty == 0)]
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor("#1f5f2c")
    ax.plot([102, 102, 120, 120, 102], [18, 62, 62, 18, 18], c="white", lw=1)  # penalty box
    ax.plot([114, 114, 120, 120, 114], [30, 50, 50, 30, 30], c="white", lw=1)  # six-yard box
    ax.plot([120, 120], [36, 44], c="white", lw=4)
    miss = s[s.goal == 0]
    goal = s[s.goal == 1]
    ax.scatter(miss.x, miss.y, s=miss.xg * 400 + 5, c="none", edgecolors="white", alpha=.5, lw=.8)
    ax.scatter(goal.x, goal.y, s=goal.xg * 400 + 5, c="#ef0107", edgecolors="white", lw=.8)
    ax.set(xlim=(80, 121), ylim=(80, 0), xticks=[], yticks=[],
           title=f"Arsenal non-penalty shots 20{season[:2]}/{season[2:]}  "
                 f"({goal.shape[0]} goals, {s.xg.sum():.1f} xG)")
    fig.tight_layout()
    fig.savefig(FIGURES / f"arsenal_shotmap_{season}.png", dpi=150)
    plt.close(fig)


def match_level(d: pd.DataFrame, sb_matches: pd.DataFrame) -> pd.DataFrame:
    """Team-match xG, then mapped onto football-data match_id via date + team names."""
    agg = d.groupby(["match_id", "team"]).agg(xg=("xg", "sum"), sb_xg=("shot_statsbomb_xg", "sum"),
                                               shots=("goal", "size"), goals=("goal", "sum")).reset_index()
    m = sb_matches.copy()
    home = agg.rename(columns=lambda c: c if c in ("match_id", "team") else f"home_{c}")
    away = agg.rename(columns=lambda c: c if c in ("match_id", "team") else f"away_{c}")
    m = m.merge(home, left_on=["match_id", "home_team"], right_on=["match_id", "team"], how="left").drop(columns="team")
    m = m.merge(away, left_on=["match_id", "away_team"], right_on=["match_id", "team"], how="left").drop(columns="team")
    m = m.fillna({c: 0 for c in m.columns if c.startswith(("home_", "away_")) and c not in ("home_team", "away_team")})
    m["hometeam"] = m.home_team.replace(SB_TO_FD)
    m["awayteam"] = m.away_team.replace(SB_TO_FD)
    m["date"] = pd.to_datetime(m.match_date).dt.strftime("%Y-%m-%d")
    return m.rename(columns={"match_id": "sb_match_id"})


def main():
    shots = pd.read_parquet(RAW / "statsbomb" / "shots.parquet")
    sb_matches = pd.read_parquet(RAW / "statsbomb" / "matches.parquet")
    d = engineer(shots).dropna(subset=["x", "y"]).reset_index(drop=True)
    print(f"{len(d)} shots, {d.goal.sum()} goals, {d.is_penalty.sum()} penalties")

    d = cross_validate(d)
    met = metrics(d)
    print(met.round(4).to_string(index=False))

    # use whichever of our two models scored better out-of-fold for everything downstream
    best = met.iloc[:2].sort_values("log_loss").iloc[0]["model"]
    d["xg"] = d["xg_logreg"] if best == "Logistic regression" else d["xg_xgb"]
    print(f"using {best} as the project xG model")

    # final models on all non-penalty shots, saved for reuse
    npd = d[d.is_penalty == 0]
    xgb().fit(as_xgb_frame(npd), npd.goal).save_model(MODELS / "xg_xgb.json")
    joblib.dump(logreg().fit(npd[NUM + CAT], npd.goal), MODELS / "xg_logreg.joblib")

    calibration_plot(d)
    for season in d.season.unique():
        shot_map(d, season)

    d.drop(columns=[c for c in d.columns if c.startswith("kp_")]).to_parquet(PROCESSED / "shots_xg.parquet", index=False)

    mx = match_level(d, sb_matches)
    con = sqlite3.connect(DB_PATH)
    met.to_sql("xg_metrics", con, if_exists="replace", index=False)
    try:
        fd = pd.read_sql("SELECT match_id, date, hometeam, awayteam FROM matches", con)
        mx = mx.merge(fd, on=["date", "hometeam", "awayteam"], how="left")
        unmatched = mx.match_id.isna().sum()
        print(f"joined StatsBomb matches to football-data: {len(mx) - unmatched}/{len(mx)} matched")
        if unmatched:
            print(mx.loc[mx.match_id.isna(), ["date", "hometeam", "awayteam"]].head(10).to_string(index=False))
    except Exception as e:  # matches table not built yet
        print("matches table not available, storing StatsBomb ids only:", e)
    mx.to_sql("match_xg", con, if_exists="replace", index=False)
    con.close()


if __name__ == "__main__":
    main()
