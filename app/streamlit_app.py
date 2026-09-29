"""Step 8b - interactive dashboard.  Run:  streamlit run app/streamlit_app.py"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import DB_PATH, FIGURES  # noqa: E402

st.set_page_config(page_title="Arsenal Under the Hood", layout="wide")


@st.cache_data
def q(sql: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(sql, con)


def has(table: str) -> bool:
    return not q(f"SELECT name FROM sqlite_master WHERE name = '{table}'").empty


def label(s: str) -> str:
    return f"20{s[:2]}/{s[2:]}"


if not DB_PATH.exists():
    st.error("No database found. Run `python run_all.py` first.")
    st.stop()

st.title("Arsenal Under the Hood")
st.caption("Premier League match prediction vs the betting market, and what drives Arsenal's over/under-performance. "
           "Data: football-data.co.uk, StatsBomb Open Data.")

tab1, tab2, tab3, tab4 = st.tabs(["Model vs market", "Arsenal by era", "Elo explorer", "xG"])

with tab1:
    if has("market_metrics"):
        st.subheader("Out-of-sample forecast quality (lower log loss / RPS is better)")
        st.dataframe(q("SELECT * FROM market_metrics").round(4), width="stretch", hide_index=True)
        st.subheader("Is the gap real? (bootstrap 95% CI on log-loss difference)")
        st.dataframe(q("SELECT * FROM market_gaps").round(4), width="stretch", hide_index=True)
        st.subheader("Betting simulation")
        st.dataframe(q("SELECT * FROM betting_results").round(3), width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        for c, f in zip((c1, c2), ("calibration_model_vs_market.png", "betting_curve.png")):
            if (FIGURES / f).exists():
                c.image(str(FIGURES / f))
        if (FIGURES / "feature_importance.png").exists():
            st.image(str(FIGURES / "feature_importance.png"), width=550)

with tab2:
    if has("arsenal_season_diag"):
        d = q("SELECT * FROM arsenal_season_diag")
        e = q("SELECT * FROM arsenal_era_summary")
        st.subheader("Eras")
        st.dataframe(e.round(2), width="stretch", hide_index=True)
        st.subheader("Actual vs expected points")
        chart = d.assign(season=d.season.map(label)).set_index("season")[["pts", "xpts_mkt", "xpts_model"]]
        st.line_chart(chart)
        st.subheader("Diagnosis (z-score vs league, positive = better)")
        z = d.assign(season=d.season.map(label)).set_index("season")[
            ["z_shot_volume", "z_finishing", "z_suppression", "z_shot_stopping"]]
        st.bar_chart(z)
        season = st.selectbox("Season detail", d.season, index=len(d) - 1, format_func=label)
        st.dataframe(q(f"""SELECT date, hometeam, awayteam, fthg, ftag, ftr FROM matches
                           WHERE season = '{season}' AND (hometeam = 'Arsenal' OR awayteam = 'Arsenal')
                           ORDER BY date"""), width="stretch", hide_index=True)

with tab3:
    teams = q("SELECT DISTINCT team FROM elo_season_end ORDER BY team").team.tolist()
    pick = st.multiselect("Teams", teams, default=[t for t in ["Arsenal", "Man United", "Chelsea", "Liverpool", "Man City"] if t in teams])
    if pick:
        e = q("SELECT * FROM elo_season_end")
        e = e[e.team.isin(pick)].assign(season=lambda d: d.season.map(label))
        st.line_chart(e.pivot(index="season", columns="team", values="elo"))
    st.caption("End-of-season Elo (rating going into each team's final match).")

with tab4:
    if has("xg_metrics"):
        st.subheader("xG model vs StatsBomb's own xG (5-fold, grouped by match)")
        st.dataframe(q("SELECT * FROM xg_metrics").round(4), width="stretch", hide_index=True)
    if has("arsenal_xg_compare"):
        st.dataframe(q("SELECT * FROM arsenal_xg_compare").round(2), width="stretch", hide_index=True)
    cols = st.columns(3)
    for c, f in zip(cols, ["xg_calibration.png", "arsenal_shotmap_0304.png", "arsenal_shotmap_1516.png"]):
        if (FIGURES / f).exists():
            c.image(str(FIGURES / f))
