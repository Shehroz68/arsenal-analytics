"""Step 7 - benchmark the models against the betting market.

1. Bookmaker odds -> probabilities. Raw 1/odds sum to > 1 (the bookmaker's margin, or
   "overround"). We remove it two ways:
      proportional  p_i = (1/o_i) / sum(1/o)
      power         p_i = (1/o_i)^k with k solved so the p_i sum to 1 (corrects the
                    favourite-longshot bias better than proportional)
2. Score models vs market on exactly the same matches (log loss, Brier, RPS) and bootstrap a
   confidence interval for the log-loss gap: is any difference real or noise?
3. Stacking: does the model add information the market doesn't already have? A walk-forward
   logistic regression on [log p_market, log p_model] is trained on past seasons only.
4. Betting simulation: bet 1 unit at Bet365 prices whenever model_prob * odds - 1 > edge.
   Report ROI with a bootstrap 95% CI. A positive point estimate with a CI spanning zero is
   *not* evidence of beating the market.

Outputs: DB tables market_probs, market_metrics, betting_results; figures.
"""
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

from config import DB_PATH, FIGURES
from elo import season_order
from metrics import log_loss_per_match, result_to_int, summary

RNG = np.random.default_rng(42)
PCOLS = ["p_home", "p_draw", "p_away"]


def proportional(odds: np.ndarray) -> np.ndarray:
    inv = 1 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def power(odds: np.ndarray) -> np.ndarray:
    inv = 1 / odds
    out = np.full_like(inv, np.nan)
    for i, row in enumerate(inv):
        if np.isnan(row).any():
            continue
        k = brentq(lambda k: (row ** k).sum() - 1, 0.5, 3.0)
        out[i] = row ** k
    return out


def market_probs(m: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for book, cols in [("b365", ["b365h", "b365d", "b365a"]), ("pinnacle", ["psh", "psd", "psa"])]:
        if not set(cols) <= set(m.columns):
            continue
        sub = m.dropna(subset=cols)
        odds = sub[cols].values.astype(float)
        for method, fn in [("power", power), ("proportional", proportional)]:
            p = fn(odds)
            frames.append(pd.DataFrame({"match_id": sub.match_id.values, "season": sub.season.values,
                                        "model": f"market_{book}_{method}",
                                        "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2],
                                        "overround": (1 / odds).sum(axis=1) - 1}))
    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(x: np.ndarray, n: int = 2000, stat=np.mean) -> tuple[float, float]:
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    s = stat(x[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def compare(preds: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics per model on the matches every core model + Bet365 covers.

    Pinnacle (from 2012/13) and the stacked model (from the third test season) cover fewer
    matches, so they are scored on their own subsets and flagged in `sample`.
    """
    wide = {k: g.set_index("match_id")[PCOLS] for k, g in preds.groupby("model")}
    partial = lambda k: k.startswith("market_pinnacle") or k == "stacked"
    common = sorted(set.intersection(*(set(v.index) for k, v in wide.items() if not partial(k))))

    rows = []
    for k, v in wide.items():
        idx = sorted(v.index) if partial(k) else common
        rows.append({"model": k, "sample": "own subset" if partial(k) else "common",
                     **summary(v.loc[idx].values, result_to_int(y.loc[idx].values))})
    table = pd.DataFrame(rows).sort_values(["sample", "log_loss"])

    gaps = []
    for ref in ("market_b365_power", "market_pinnacle_power"):
        for k in ("xgb", "full_logit", "elo_logit", "stacked"):
            if k not in wide or ref not in wide:
                continue
            idx = sorted(set(wide[k].index) & set(wide[ref].index))
            yy = result_to_int(y.loc[idx].values)
            d = log_loss_per_match(wide[k].loc[idx].values, yy) - log_loss_per_match(wide[ref].loc[idx].values, yy)
            lo, hi = bootstrap_ci(d)
            gaps.append({"model": k, "vs": ref, "n": len(idx), "mean_logloss_gap": d.mean(), "ci_low": lo,
                         "ci_high": hi,
                         "verdict": "model better" if hi < 0 else "market better" if lo > 0 else "no significant difference"})
    return table, pd.DataFrame(gaps)


def stack(preds: pd.DataFrame, f: pd.DataFrame, model: str = "xgb", market: str = "market_b365_power") -> pd.DataFrame:
    a = preds[preds.model == model].set_index("match_id")[PCOLS].add_prefix("m_")
    b = preds[preds.model == market].set_index("match_id")[PCOLS].add_prefix("k_")
    d = a.join(b, how="inner").join(f.set_index("match_id")[["season", "ftr"]])
    d["season_n"] = d.season.map(season_order)
    X = np.log(d[[*a.columns, *b.columns]].clip(1e-6))
    y = result_to_int(d.ftr.values)
    out = []
    seasons = sorted(d.season_n.unique())
    for s in seasons[2:]:  # need at least two seasons of out-of-sample predictions to learn weights
        tr, te = d.season_n < s, d.season_n == s
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
        p = lr.predict_proba(X[te])
        out.append(pd.DataFrame({"match_id": d.index[te], "season": d.season[te].values, "model": "stacked",
                                 "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def betting(preds: pd.DataFrame, m: pd.DataFrame, model: str = "xgb") -> tuple[pd.DataFrame, pd.DataFrame]:
    p = preds[preds.model == model].merge(m[["match_id", "date", "ftr", "b365h", "b365d", "b365a"]], on="match_id")
    p = p.dropna(subset=["b365h", "b365d", "b365a"]).sort_values("date")
    rows, curves = [], {}
    for edge in [0.0, 0.02, 0.05, 0.10, 0.15]:
        bets = []
        for pc, oc, res in [("p_home", "b365h", "H"), ("p_draw", "b365d", "D"), ("p_away", "b365a", "A")]:
            ev = p[pc] * p[oc] - 1
            sel = p[ev > edge]
            bets.append(pd.DataFrame({"date": sel.date, "profit": np.where(sel.ftr == res, sel[oc] - 1, -1.0)}))
        b = pd.concat(bets).sort_values("date")
        if b.empty:
            continue
        lo, hi = bootstrap_ci(b.profit.values)
        rows.append({"model": model, "min_edge": edge, "bets": len(b), "profit_units": b.profit.sum(),
                     "roi": b.profit.mean(), "roi_ci_low": lo, "roi_ci_high": hi})
        curves[edge] = b.profit.cumsum().values
    fig, ax = plt.subplots(figsize=(7, 4))
    for edge, c in curves.items():
        ax.plot(c, label=f"edge > {edge:.0%}")
    ax.axhline(0, c="grey", lw=1)
    ax.set(title=f"Flat-stake betting at Bet365 prices using {model} probabilities",
           xlabel="bets placed (chronological)", ylabel="cumulative profit (units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "betting_curve.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows), p


def calibration(preds: pd.DataFrame, y: pd.Series, models: list[str]):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    bins = np.linspace(0, 1, 11)
    for k in models:
        g = preds[preds.model == k]
        if g.empty:
            continue
        obs = (y.loc[g.match_id].values == "H").astype(float)
        dd = pd.DataFrame({"p": g.p_home.values, "o": obs})
        c = dd.groupby(pd.cut(dd.p, bins), observed=True).mean()
        ax.plot(c.p, c.o, marker="o", label=k)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set(xlabel="predicted P(home win)", ylabel="observed home-win rate", title="Calibration: model vs market")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_model_vs_market.png", dpi=150)
    plt.close(fig)


def main():
    con = sqlite3.connect(DB_PATH)
    m = pd.read_sql("SELECT * FROM matches", con)
    f = pd.read_sql("SELECT match_id, season, ftr FROM features", con)
    preds = pd.read_sql("SELECT * FROM predictions", con)

    mk = market_probs(m[m.match_id.isin(preds.match_id)])
    mk.to_sql("market_probs", con, if_exists="replace", index=False)
    print(f"mean Bet365 overround: {mk.loc[mk.model == 'market_b365_power', 'overround'].mean():.3%}")

    allp = pd.concat([preds, mk.drop(columns="overround")], ignore_index=True)
    st = stack(allp, f)
    allp = pd.concat([allp, st], ignore_index=True)
    st.to_sql("predictions_stacked", con, if_exists="replace", index=False)

    y = m.set_index("match_id").ftr
    table, gaps = compare(allp, y)
    table.to_sql("market_metrics", con, if_exists="replace", index=False)
    gaps.to_sql("market_gaps", con, if_exists="replace", index=False)
    print(table.round(4).to_string(index=False))
    print(gaps.round(4).to_string(index=False))

    bet, _ = betting(allp, m)
    bet.to_sql("betting_results", con, if_exists="replace", index=False)
    print(bet.round(3).to_string(index=False))

    calibration(allp, y, ["xgb", "full_logit", "market_b365_power"])
    con.close()


if __name__ == "__main__":
    main()
