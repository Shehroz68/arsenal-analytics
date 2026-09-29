"""Step 8a - why does Arsenal over- or under-perform, era by era?

1. Expected points (xPts) per match = 3 * P(win) + P(draw), from
      the market (Bet365, falling back to William Hill; margin removed with the power method)
      our XGBoost model (out-of-sample seasons only)
   Over-performance = actual points - xPts. The market already prices in squad quality, so a
   consistent positive gap means Arsenal beat expectations *given* how good they looked.
2. Performance diagnosis per season, as z-scores against the rest of the league that season:
      shot volume        shots on target for per game          (attack creation)
      finishing          goals per shot on target               (conversion / luck)
      suppression        shots on target against per game (neg) (defensive structure)
      shot stopping      goals conceded per SoT against (neg)   (goalkeeping / luck)
   Finishing and shot stopping are the noisiest; big swings there usually regress.
3. StatsBomb xG for the two seasons we have event data for (2003/04 and 2015/16).

Outputs: DB tables arsenal_season_diag, arsenal_era_summary, arsenal_xg_compare;
         figures; reports/arsenal_findings.md
"""
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DB_PATH, ERAS, FIGURES, ROOT, TEAM
from market import power

RED = "#ef0107"


def era_of(date: str) -> str:
    for name, start, end in ERAS:
        if start <= date <= end:
            return name
    return "Unknown"


def season_label(s: str) -> str:
    return f"20{s[:2]}/{s[2:]}"


def match_xpts(con) -> pd.DataFrame:
    m = pd.read_sql(f"""SELECT match_id, season, date, hometeam, awayteam, ftr,
                               b365h, b365d, b365a, whh, whd, wha
                        FROM matches WHERE hometeam = '{TEAM}' OR awayteam = '{TEAM}'""", con)
    odds = m[["b365h", "b365d", "b365a"]].to_numpy(dtype=float, copy=True)
    fallback = m[["whh", "whd", "wha"]].to_numpy(dtype=float, copy=True)
    missing = np.isnan(odds).any(axis=1)
    odds[missing] = fallback[missing]
    pm = power(odds)

    home = m.hometeam == TEAM
    m["p_win_mkt"] = np.where(home, pm[:, 0], pm[:, 2])
    m["p_draw_mkt"] = pm[:, 1]
    m["xpts_mkt"] = 3 * m.p_win_mkt + m.p_draw_mkt
    m["pts"] = np.select([m.ftr == "D", (m.ftr == "H") == home], [1, 3], 0)

    pr = pd.read_sql("SELECT match_id, p_home, p_draw, p_away FROM predictions WHERE model = 'xgb'", con)
    m = m.merge(pr, on="match_id", how="left")
    m["xpts_model"] = 3 * np.where(home, m.p_home, m.p_away) + m.p_draw
    m["era"] = m.date.map(era_of)
    return m


def league_zscores(con) -> pd.DataFrame:
    tm = pd.read_sql("SELECT season, team, gf, ga, sot_f, sot_a, shots_f FROM team_matches", con)
    t = tm.groupby(["season", "team"]).agg(gp=("gf", "size"), gf=("gf", "sum"), ga=("ga", "sum"),
                                           sot_f=("sot_f", "sum"), sot_a=("sot_a", "sum")).reset_index()
    t["shot_volume"] = t.sot_f / t.gp
    t["finishing"] = t.gf / t.sot_f
    t["suppression"] = -(t.sot_a / t.gp)
    t["shot_stopping"] = -(t.ga / t.sot_a)
    dims = ["shot_volume", "finishing", "suppression", "shot_stopping"]
    z = t.groupby("season")[dims].transform(lambda x: (x - x.mean()) / x.std())
    return pd.concat([t[["season", "team"]], z.add_prefix("z_")], axis=1)


def figures(season_diag: pd.DataFrame, con):
    # 1. points vs expected points
    d = season_diag
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, d.pts, color=RED, alpha=.85, label="actual points")
    ax.plot(x, d.xpts_mkt, "k-o", ms=4, label="market xPts")
    if d.xpts_model.notna().any():
        ax.plot(x, d.xpts_model, "--o", c="#063672", ms=4, label="model xPts (out-of-sample)")
    ax.set_xticks(x, [season_label(s) for s in d.season], rotation=60, fontsize=8)
    ax.set(ylabel="points", title="Arsenal: actual vs expected points per season")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "arsenal_points_vs_xpts.png", dpi=150)
    plt.close(fig)

    # 2. diagnosis heatmap
    z = d.set_index("season")[["z_shot_volume", "z_finishing", "z_suppression", "z_shot_stopping"]]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(z.T.values, cmap="RdBu", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(4), ["shot volume", "finishing", "suppression", "shot stopping"])
    ax.set_xticks(range(len(z)), [season_label(s) for s in z.index], rotation=60, fontsize=8)
    ax.set_title("Arsenal vs league (z-score, blue = better than average)")
    fig.colorbar(im, ax=ax, fraction=.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "arsenal_diagnosis_heatmap.png", dpi=150)
    plt.close(fig)

    # 3. Elo over time with eras shaded
    e = pd.read_sql(f"""SELECT m.date, CASE WHEN m.hometeam = '{TEAM}' THEN e.elo_home ELSE e.elo_away END AS elo
                        FROM matches m JOIN elo e USING (match_id)
                        WHERE m.hometeam = '{TEAM}' OR m.awayteam = '{TEAM}' ORDER BY m.date""", con)
    e["date"] = pd.to_datetime(e.date)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(e.date, e.elo, c=RED)
    for i, (name, start, end) in enumerate(ERAS):
        ax.axvspan(pd.Timestamp(start), min(pd.Timestamp(end), e.date.max()), alpha=.08 if i % 2 else .16, color="grey")
        ax.text(pd.Timestamp(start), 0.98, f" {name}", fontsize=8, va="top", transform=ax.get_xaxis_transform())
    ax.set(ylabel="pre-match Elo", title="Arsenal Elo rating, 2000-2026")
    fig.tight_layout()
    fig.savefig(FIGURES / "arsenal_elo.png", dpi=150)
    plt.close(fig)


def xg_compare(con) -> pd.DataFrame:
    try:
        mx = pd.read_sql("SELECT * FROM match_xg", con)
    except Exception:
        return pd.DataFrame()
    rows = []
    for season, g in mx.groupby("season"):
        home = g.hometeam == TEAM
        away = g.awayteam == TEAM
        g = g[home | away]
        h = g.hometeam == TEAM
        xg_f = np.where(h, g.home_xg, g.away_xg)
        xg_a = np.where(h, g.away_xg, g.home_xg)
        gf = np.where(h, g.home_score, g.away_score)
        ga = np.where(h, g.away_score, g.home_score)
        rows.append({"season": season, "matches": len(g), "xg_for_pg": xg_f.mean(), "goals_for_pg": gf.mean(),
                     "xg_against_pg": xg_a.mean(), "goals_against_pg": ga.mean(),
                     "finishing_over_xg": gf.sum() - xg_f.sum(), "conceded_under_xg": xg_a.sum() - ga.sum()})
    return pd.DataFrame(rows)


def write_report(season_diag, eras, xgc, con):
    def tbl(df, digits=2):
        return df.round(digits).to_markdown(index=False)

    metrics = pd.read_sql("SELECT * FROM market_metrics", con) if _has(con, "market_metrics") else pd.DataFrame()
    gaps = pd.read_sql("SELECT * FROM market_gaps", con) if _has(con, "market_gaps") else pd.DataFrame()
    bets = pd.read_sql("SELECT * FROM betting_results", con) if _has(con, "betting_results") else pd.DataFrame()
    xgm = pd.read_sql("SELECT * FROM xg_metrics", con) if _has(con, "xg_metrics") else pd.DataFrame()

    best = season_diag.loc[season_diag.over_mkt.idxmax()]
    worst = season_diag.loc[season_diag.over_mkt.idxmin()]
    lines = [
        "# Findings (auto-generated by `src/arsenal.py`)",
        "",
        "Numbers here are regenerated on every run of `python run_all.py`.",
        "",
        "## 1. Can a model beat the betting market?",
        "",
        tbl(metrics, 4) if len(metrics) else "_run src/market.py first_",
        "",
        "Log-loss gap vs market (negative = model better), 95% bootstrap CI:",
        "",
        tbl(gaps, 4) if len(gaps) else "",
        "",
        "Flat-stake betting simulation at Bet365 prices:",
        "",
        tbl(bets, 3) if len(bets) else "",
        "",
        "## 2. Arsenal: points vs expectation by era",
        "",
        tbl(eras),
        "",
        f"Biggest over-performance: **{season_label(best.season)}** ({best.pts:.0f} pts vs {best.xpts_mkt:.1f} market xPts). "
        f"Biggest under-performance: **{season_label(worst.season)}** ({worst.pts:.0f} pts vs {worst.xpts_mkt:.1f}).",
        "",
        "## 3. Season-by-season diagnosis",
        "",
        tbl(season_diag[["season", "era", "position", "pts", "xpts_mkt", "over_mkt", "z_shot_volume",
                         "z_finishing", "z_suppression", "z_shot_stopping"]]),
        "",
        "## 4. xG model (StatsBomb shots)",
        "",
        tbl(xgm, 4) if len(xgm) else "",
        "",
        "Arsenal xG, the two seasons with event data:",
        "",
        tbl(xgc) if len(xgc) else "_run src/xg.py first_",
        "",
        "![points](figures/arsenal_points_vs_xpts.png)",
        "![diagnosis](figures/arsenal_diagnosis_heatmap.png)",
        "![elo](figures/arsenal_elo.png)",
    ]
    (ROOT / "reports" / "arsenal_findings.md").write_text("\n".join(lines))


def _has(con, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone() is not None


def main():
    con = sqlite3.connect(DB_PATH)
    mx = match_xpts(con)
    season = mx.groupby("season").agg(era=("era", lambda s: s.mode()[0]), pts=("pts", "sum"),
                                      xpts_mkt=("xpts_mkt", "sum"), xpts_model=("xpts_model", lambda s: s.sum(min_count=len(s)))).reset_index()
    season["over_mkt"] = season.pts - season.xpts_mkt
    season["over_model"] = season.pts - season.xpts_model
    pos = pd.read_sql(f"SELECT season, position FROM season_table WHERE team = '{TEAM}'", con)
    z = league_zscores(con)
    season = season.merge(pos, on="season").merge(z[z.team == TEAM].drop(columns="team"), on="season")
    season.to_sql("arsenal_season_diag", con, if_exists="replace", index=False)

    era_order = {name: i for i, (name, _, _) in enumerate(ERAS)}
    eras = (mx.groupby("era").agg(matches=("pts", "size"), ppg=("pts", "mean"), xppg_mkt=("xpts_mkt", "mean"))
            .assign(over_per_38=lambda d: (d.ppg - d.xppg_mkt) * 38)
            .join(season.groupby("era")[["z_shot_volume", "z_finishing", "z_suppression", "z_shot_stopping"]].mean())
            .reset_index().sort_values("era", key=lambda s: s.map(era_order)))
    eras.to_sql("arsenal_era_summary", con, if_exists="replace", index=False)

    xgc = xg_compare(con)
    if not xgc.empty:
        xgc.to_sql("arsenal_xg_compare", con, if_exists="replace", index=False)

    figures(season, con)
    write_report(season, eras, xgc, con)
    print(eras.round(2).to_string(index=False))
    print(xgc.round(2).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
