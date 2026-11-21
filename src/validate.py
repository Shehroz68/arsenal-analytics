"""Run sql/validate.sql against the database and print each result."""
import sqlite3
from pathlib import Path


def connect_existing(path):
    """Open an existing database without creating or modifying it."""
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)

def main():
    import pandas as pd

    con = connect_existing("data/processed/epl.db")
    queries = [q.strip() for q in open("sql/validate.sql").read().split(";") if "SELECT" in q]
    for q in queries:
        title = q.splitlines()[0]
        body = "\n".join(l for l in q.splitlines() if not l.startswith("--"))
        print(f"\n{title}")
        print(pd.read_sql(body, con).to_string(index=False))


if __name__ == "__main__":
    main()
