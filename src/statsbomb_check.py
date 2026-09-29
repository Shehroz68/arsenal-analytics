"""List Premier League coverage in StatsBomb Open Data and whether Arsenal is included."""
import warnings
import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")  # statsbombpy warns about missing credentials for open data

comps = sb.competitions()
epl = comps[comps.competition_name.str.contains("Premier League", case=False)
            & (comps.country_name == "England")]
rows = []
for _, c in epl.iterrows():
    m = sb.matches(competition_id=c.competition_id, season_id=c.season_id)
    arsenal = m[(m.home_team == "Arsenal") | (m.away_team == "Arsenal")]
    rows.append({"competition_id": c.competition_id, "season_id": c.season_id,
                 "season": c.season_name, "matches": len(m), "arsenal_matches": len(arsenal)})

report = pd.DataFrame(rows)
print(report.to_string(index=False))
report.to_csv("data/processed/statsbomb_epl_coverage.csv", index=False)
