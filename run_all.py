"""Run the full pipeline end to end:  python run_all.py  [--skip-download]"""
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

skip = {"src/download.py"} if "--skip-download" in sys.argv else set()
for name, script in STEPS:
    if script in skip:
        continue
    print(f"\n=== Step {name} ===", flush=True)
    subprocess.run([sys.executable, script], check=True)
print("\nDone. Dashboard:  streamlit run app/streamlit_app.py")
