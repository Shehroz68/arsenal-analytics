"""Run the full pipeline end to end:  python run_all.py  [--skip-download]"""
import argparse
from pathlib import Path
import subprocess
import sys

STEPS = [
    ("1  download football-data.co.uk", "src/download.py"),
    ("1  build SQLite database", "src/build_db.py"),
    ("2  clean + SQL views", "src/transform.py"),
    ("1  validation checks", "src/validate.py"),
    ("3  Elo ratings", "src/elo.py"),
    ("4a StatsBomb shots", "src/statsbomb_shots.py"),
    ("4b xG model", "src/xg.py"),
    ("5  features", "src/features.py"),
    ("6  walk-forward models", "src/train.py"),
    ("7  market benchmark", "src/market.py"),
    ("8  Arsenal diagnosis", "src/arsenal.py"),
]

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Arsenal analytics pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Reuse existing football-data CSV files")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    for name, script in STEPS:
        if args.skip_download and script == "src/download.py":
            continue
        print(f"\n=== Step {name} ===", flush=True)
        subprocess.run([sys.executable, str(root / script)], cwd=root, check=True)
    print("\nDone. Dashboard:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
