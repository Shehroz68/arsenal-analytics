# Arsenal Under the Hood

Match prediction and performance diagnosis in the Premier League.

**Core question:** Can a custom model beat the betting market at predicting Premier League results, and what does it say about why Arsenal over- or under-performs in different eras?

## Data sources

- **[football-data.co.uk](https://www.football-data.co.uk/englandm.php)**: every EPL match from 2000/01 to 2025/26 with results, shots, cards, fouls, corners, referee and bookmaker odds. Data courtesy of football-data.co.uk.
- **[StatsBomb Open Data](https://github.com/statsbomb/open-data)**: event-level data (shots with location, passes, pressures). Data provided by StatsBomb.

## StatsBomb Premier League coverage

| competition_id | season_id | season | matches | Arsenal matches |
|---|---|---|---|---|
| 2 | 27 | 2015/16 | 380 | 38 |
| 2 | 44 | 2003/04 | 38 | 38 (Invincibles season, Arsenal only) |

## Repo layout

```
data/raw/         raw CSVs (git-ignored)
data/processed/   epl.db (git-ignored) and small derived tables
sql/              SQL queries (validation, analysis)
notebooks/        exploration
src/              pipeline scripts
```

## Reproduce Step 1

```bash
pip install -r requirements.txt
python src/download.py         # 26 seasons -> data/raw/fd/
python src/build_db.py         # -> data/processed/epl.db, table `matches`
python src/validate.py         # runs sql/validate.sql
python src/statsbomb_check.py  # StatsBomb EPL coverage
```

## Stack

Python, SQLite, scikit-learn, XGBoost, Streamlit / Tableau Public.
