"""Run sql/validate.sql against the database and print each result."""
from contextlib import closing
import sqlite3
from pathlib import Path


def connect_existing(path):
    """Open an existing database without creating or modifying it."""
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)

def main():
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    queries = [q.strip() for q in (root / "sql/validate.sql").read_text().split(";") if "SELECT" in q]
    with closing(connect_existing(root / "data/processed/epl.db")) as con:
        for q in queries:
            title = q.splitlines()[0]
            body = "\n".join(l for l in q.splitlines() if not l.startswith("--"))
            print(f"\n{title}")
            print(pd.read_sql(body, con).to_string(index=False))



if __name__ == "__main__":
    main()
