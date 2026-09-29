"""Step 6 - time-aware match prediction (walk-forward, season by season).

For every test season S (from FIRST_TEST_SEASON onwards):
    train on all seasons before S (2000/01 is burn-in for Elo/rolling features, never trained on)
    predict every match in S
So every prediction is out-of-sample and uses only the past, like a real forecaster.

Models
    base_rates   historical home/draw/away frequencies (the "know nothing" benchmark)
    elo_logit    multinomial logistic regression on Elo difference only
    full_logit   multinomial logistic regression on all Step 5 features
    xgb          gradient-boosted trees on all features; number of trees chosen by early
                 stopping on the most recent training season, then refit on all training data

Outputs: DB tables predictions (long format) and model_metrics; feature importance figure.
"""
import sqlite3
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import DB_PATH, FIGURES, FIRST_TEST_SEASON
from elo import season_order
from metrics import result_to_int, summary

warnings.filterwarnings("ignore")

ELO = ["elo_diff"]


def feature_columns(f: pd.DataFrame) -> list[str]:
    cols = ["elo_diff", "elo_home", "elo_away",
            "home_rest_days", "away_rest_days", "home_matches_this_season"]
    cols += [c for c in f.columns if c.startswith("diff_")]
    cols += [c for c in f.columns if c.startswith(("home_", "away_")) and c.endswith(("_r10", "ppg_season"))]
    return list(dict.fromkeys(c for c in cols if c in f.columns))


def logit():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=3000, C=0.1))


XGB_PARAMS = dict(objective="multi:softprob", num_class=3, learning_rate=0.02, max_depth=3,
                  min_child_weight=20, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
                  eval_metric="mlogloss", tree_method="hist")


def fit_xgb(Xtr, ytr, seasons_tr):
    last = seasons_tr == seasons_tr.max()
    probe = XGBClassifier(n_estimators=1500, early_stopping_rounds=100, **XGB_PARAMS)
    probe.fit(Xtr[~last], ytr[~last], eval_set=[(Xtr[last], ytr[last])], verbose=False)
    n = max(50, probe.best_iteration + 1)
    return XGBClassifier(n_estimators=n, **XGB_PARAMS).fit(Xtr, ytr)


def walk_forward(f: pd.DataFrame) -> tuple[pd.DataFrame, XGBClassifier, list[str]]:
    f = f.copy()
    f["y"] = result_to_int(f.ftr)
    f["season_n"] = f.season.map(season_order)
    cols = feature_columns(f)
    first_test = season_order(FIRST_TEST_SEASON)
    burn_in = f.season_n.min()

    preds, model = [], None
    for s in sorted(f.season_n.unique()):
        if s < first_test:
            continue
        tr = f[(f.season_n < s) & (f.season_n > burn_in)]
        te = f[f.season_n == s]
        ytr = tr.y.values

        out = {}
        rates = np.bincount(ytr, minlength=3) / len(ytr)
        out["base_rates"] = np.tile(rates, (len(te), 1))
        out["elo_logit"] = logit().fit(tr[ELO], ytr).predict_proba(te[ELO])
        out["full_logit"] = logit().fit(tr[cols], ytr).predict_proba(te[cols])
        model = fit_xgb(tr[cols].values, ytr, tr.season_n.values)
        out["xgb"] = model.predict_proba(te[cols].values)

        for name, p in out.items():
            preds.append(pd.DataFrame({"match_id": te.match_id.values, "season": te.season.values, "model": name,
                                       "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2]}))
        print(f"season {s}/{(s + 1) % 100:02d}: trained on {len(tr)}, predicted {len(te)}")
    return pd.concat(preds, ignore_index=True), model, cols


def evaluate(preds: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    y = f.set_index("match_id").ftr
    rows = []
    for (model,), g in preds.groupby(["model"]):
        p = g[["p_home", "p_draw", "p_away"]].values
        rows.append({"model": model, **summary(p, result_to_int(y.loc[g.match_id].values))})
    return pd.DataFrame(rows).sort_values("log_loss")


def importance_plot(model: XGBClassifier, cols: list[str]):
    imp = pd.Series(model.get_booster().get_score(importance_type="gain"))
    imp.index = [cols[int(k[1:])] for k in imp.index]
    imp = imp.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax, color="#ef0107")
    ax.set(title="XGBoost feature importance (gain), final season model", xlabel="gain")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=150)
    plt.close(fig)


def main():
    con = sqlite3.connect(DB_PATH)
    f = pd.read_sql("SELECT * FROM features", con)
    preds, model, cols = walk_forward(f)
    preds.to_sql("predictions", con, if_exists="replace", index=False)
    met = evaluate(preds, f)
    met.to_sql("model_metrics", con, if_exists="replace", index=False)
    print(met.round(4).to_string(index=False))
    importance_plot(model, cols)
    con.close()


if __name__ == "__main__":
    main()
