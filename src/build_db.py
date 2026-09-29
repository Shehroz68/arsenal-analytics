"""Combine the raw season CSVs into one SQLite table: data/processed/epl.db -> matches."""
import glob
import re
import sqlite3

import pandas as pd

frames = []
for f in sorted(glob.glob("data/raw/fd/E0_*.csv")):
    season = re.search(r"E0_(\d{4})", f).group(1)
    df = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip").dropna(how="all")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    df["season"] = season
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    frames.append(df)

matches = pd.concat(frames, ignore_index=True)

# keep a core set of columns (odds columns vary by season)
core = ["season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
        "Referee", "B365H", "B365D", "B365A"]
matches = matches[[c for c in core if c in matches.columns]]
matches.columns = [c.lower() for c in matches.columns]
matches = matches.sort_values(["date", "hometeam"]).reset_index(drop=True)
matches.insert(0, "match_id", matches.index + 1)
matches["date"] = matches["date"].dt.strftime("%Y-%m-%d")

con = sqlite3.connect("data/processed/epl.db")
matches.to_sql("matches", con, if_exists="replace", index=False)
con.close()
print(matches.shape)
