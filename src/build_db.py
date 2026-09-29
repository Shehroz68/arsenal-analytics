"""Combine the raw season CSVs into one SQLite table: data/processed/epl.db -> matches.

Some football-data.co.uk files (e.g. 2003/04, 2004/05) have rows with more commas than the
header. pandas would skip those rows, silently losing matches, so each file is read with the
csv module and every row is trimmed or padded to the header length.
"""
import csv
import glob
import re
import sqlite3

import pandas as pd

CORE = ["season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
        "Referee", "B365H", "B365D", "B365A",
        "PSH", "PSD", "PSA",  # Pinnacle (from 2012/13): a sharper market than Bet365
        "WHH", "WHD", "WHA"]  # William Hill: fallback for early seasons


def read_season(path: str) -> pd.DataFrame:
    with open(path, encoding="latin-1", newline="") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    while header and header[-1] == "":  # trailing empty header cells
        header.pop()
    n = len(header)
    body = [(r + [""] * n)[:n] for r in rows[1:] if any(cell.strip() for cell in r)]
    df = pd.DataFrame(body, columns=header).replace("", None)
    return df.dropna(subset=["HomeTeam", "AwayTeam"])


def parse_dates(s: pd.Series) -> pd.Series:
    """Early seasons use dd/mm/yy, later ones dd/mm/yyyy."""
    long = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return long.fillna(short)


frames = []
for f in sorted(glob.glob("data/raw/fd/E0_*.csv")):
    season = re.search(r"E0_(\d{4})", f).group(1)
    df = read_season(f)
    df = df[[c for c in CORE if c in df.columns]].copy()
    df["season"] = season
    df["Date"] = parse_dates(df["Date"])
    frames.append(df)
    print(f"{season}: {len(df)} matches")

matches = pd.concat(frames, ignore_index=True)
matches = matches[[c for c in CORE if c in matches.columns]]
matches.columns = [c.lower() for c in matches.columns]
matches = matches.sort_values(["date", "hometeam"]).reset_index(drop=True)
matches.insert(0, "match_id", matches.index + 1)
matches["date"] = matches["date"].dt.strftime("%Y-%m-%d")

con = sqlite3.connect("data/processed/epl.db")
matches.to_sql("matches", con, if_exists="replace", index=False)
con.close()
print(matches.shape)
