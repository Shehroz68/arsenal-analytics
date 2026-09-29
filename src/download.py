"""Download every Premier League season (2000/01 - 2025/26) from football-data.co.uk."""
import pathlib
import requests

out = pathlib.Path("data/raw/fd")
out.mkdir(parents=True, exist_ok=True)
seasons = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2000, 2026)]  # 0001 ... 2526

failed = []
for s in seasons:
    url = f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv"
    try:
        r = requests.get(url, timeout=30)
        ok = r.ok and len(r.content) > 1000
    except requests.RequestException as e:
        ok, r = False, e
    if ok:
        (out / f"E0_{s}.csv").write_bytes(r.content)
        print("ok", s)
    else:
        print("FAILED", s, getattr(r, "status_code", r))
        failed.append(s)

print("failed seasons:", failed or "none")
