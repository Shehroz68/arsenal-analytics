"""Step 2 - clean the raw matches table and build the SQL modelling layer.

- Normalises team names (football-data occasionally changes spellings between seasons).
- Drops matches with missing results, flags duplicates.
- Creates views: team_matches, season_table, arsenal_seasons (sql/02_views.sql).
"""
import sqlite3

import pandas as pd

from config import DB_PATH, ROOT

# football-data.co.uk is consistent across seasons, so this starts empty. If validate.py ever shows
# two spellings of one club, add {"variant": "canonical"} here.
TEAM_ALIASES: dict[str, str] = {}


def main():
    con = sqlite3.connect(DB_PATH)
    m = pd.read_sql("SELECT * FROM matches", con)
    n0 = len(m)

    for col in ("hometeam", "awayteam"):
        m[col] = m[col].str.strip().replace(TEAM_ALIASES)

    m = m.dropna(subset=["date", "fthg", "ftag"])
    dupes = m.duplicated(subset=["date", "hometeam", "awayteam"])
    if dupes.any():
        print(f"dropping {dupes.sum()} duplicate rows")
        m = m[~dupes]

    for c in ["fthg", "ftag", "hs", "as", "hst", "ast", "hf", "af", "hc", "ac", "hy", "ay", "hr", "ar"]:
        if c in m:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    for c in ["b365h", "b365d", "b365a", "psh", "psd", "psa", "whh", "whd", "wha"]:
        if c in m:
            m[c] = pd.to_numeric(m[c], errors="coerce")
            m.loc[m[c] <= 1.0, c] = None  # impossible decimal odds

    m = m.sort_values(["date", "hometeam"]).reset_index(drop=True)
    m.to_sql("matches", con, if_exists="replace", index=False)
    con.executescript(open(ROOT / "sql" / "02_views.sql").read())
    con.execute("CREATE INDEX IF NOT EXISTS ix_matches_date ON matches(date)")
    con.commit()

    print(f"matches: {n0} -> {len(m)} rows")
    print(pd.read_sql("SELECT season, position, pts, gf, ga, goals_per_sot FROM arsenal_seasons", con).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
