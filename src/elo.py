"""Step 3 - custom Elo rating system for the Premier League.

Design choices (each one is a tunable parameter):
  * home advantage (hfa) added to the home side's rating when computing expectations
  * margin-of-victory multiplier, damped for favourites (FiveThirtyEight-style) so big
    wins by strong teams don't inflate ratings
  * between seasons, ratings regress towards the league mean by `regress`
  * promoted teams enter at the average end-of-season rating of the relegated sides
    (they replace those teams in the league)

Leakage rule: each match stores the ratings *before* kick-off. The update happens after.
Tuning uses only 2001/02-2004/05, which comes before the first test season in Step 6.

Output table: elo (match_id, elo_home, elo_away, elo_diff, elo_exp_home)
"""
import itertools
import sqlite3

import numpy as np
import pandas as pd

from config import DB_PATH, FIRST_TEST_SEASON

BASE = 1500.0


def expected(r_home: float, r_away: float, hfa: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_home + hfa - r_away) / 400.0))


def mov_multiplier(goal_diff: int, winner_elo_edge: float) -> float:
    if goal_diff == 0:
        return 1.0
    return np.log(abs(goal_diff) + 1) * (2.2 / (winner_elo_edge * 0.001 + 2.2))


def run_elo(matches: pd.DataFrame, k: float = 20, hfa: float = 60, regress: float = 0.2) -> pd.DataFrame:
    """matches must be sorted by date and contain season, hometeam, awayteam, fthg, ftag."""
    ratings: dict[str, float] = {}
    out = np.zeros((len(matches), 3))
    current_season = None
    season_teams: set[str] = set()
    prev_season_teams: set[str] = set()

    for i, row in enumerate(matches.itertuples(index=False)):
        if row.season != current_season:
            # season rollover: regress to mean and seed promoted clubs
            if current_season is not None:
                mean = np.mean([ratings[t] for t in season_teams])
                for t in season_teams:
                    ratings[t] = mean + (1 - regress) * (ratings[t] - mean)
                prev_season_teams = season_teams
            in_season = matches.season == row.season
            new_season_teams = set(matches.loc[in_season, "hometeam"]) | set(matches.loc[in_season, "awayteam"])
            relegated = prev_season_teams - new_season_teams
            promoted = new_season_teams - prev_season_teams
            seed = np.mean([ratings[t] for t in relegated]) if relegated and prev_season_teams else BASE
            for t in promoted:
                ratings[t] = seed
            season_teams = new_season_teams
            current_season = row.season

        h, a = row.hometeam, row.awayteam
        rh, ra = ratings.get(h, BASE), ratings.get(a, BASE)
        e_home = expected(rh, ra, hfa)
        out[i] = (rh, ra, e_home)

        gd = int(row.fthg - row.ftag)
        s_home = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        winner_edge = (rh + hfa - ra) if gd > 0 else (ra - rh - hfa) if gd < 0 else 0.0
        delta = k * mov_multiplier(gd, winner_edge) * (s_home - e_home)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

    res = pd.DataFrame(out, columns=["elo_home", "elo_away", "elo_exp_home"], index=matches.index)
    res["elo_diff"] = res["elo_home"] - res["elo_away"]
    return res


def score(matches: pd.DataFrame, elo: pd.DataFrame, seasons: list[str]) -> float:
    """Brier score of the Elo expected score vs actual (1 / 0.5 / 0) on the given seasons."""
    mask = matches.season.isin(seasons)
    actual = np.select([matches.fthg > matches.ftag, matches.fthg == matches.ftag], [1.0, 0.5], 0.0)
    return float(np.mean((elo.loc[mask, "elo_exp_home"] - actual[mask]) ** 2))


def tune(matches: pd.DataFrame) -> dict:
    seasons = sorted(matches.season.unique(), key=season_order)
    # 2000/01 is burn-in; tune on the seasons before the first out-of-sample season
    tune_seasons = [s for s in seasons[1:] if season_order(s) < season_order(FIRST_TEST_SEASON)]
    best = None
    for k, hfa, reg in itertools.product([10, 15, 20, 25, 30, 40], [30, 45, 60, 75, 90], [0.0, 0.1, 0.2, 0.33, 0.5]):
        s = score(matches, run_elo(matches, k, hfa, reg), tune_seasons)
        if best is None or s < best[0]:
            best = (s, dict(k=k, hfa=hfa, regress=reg))
    print(f"best Elo params {best[1]} (Brier {best[0]:.4f} on {tune_seasons[0]}-{tune_seasons[-1]})")
    return best[1]


def season_order(s: str) -> int:
    """'0001' -> 2000, '9900' would be 1999; all our seasons are 2000+."""
    return 2000 + int(s[:2])


def main():
    con = sqlite3.connect(DB_PATH)
    m = pd.read_sql("SELECT match_id, season, date, hometeam, awayteam, fthg, ftag FROM matches ORDER BY date, match_id", con)
    params = tune(m)
    elo = run_elo(m, **params)
    out = pd.concat([m[["match_id"]], elo], axis=1)
    out.to_sql("elo", con, if_exists="replace", index=False)
    pd.DataFrame([params]).to_sql("elo_params", con, if_exists="replace", index=False)

    # handy: end-of-season rating per team
    final = (pd.concat([m, elo], axis=1)
             .melt(id_vars=["season", "date"], value_vars=["hometeam", "awayteam"], value_name="team", ignore_index=False)
             .join(elo[["elo_home", "elo_away"]]))
    final["elo"] = np.where(final.variable == "hometeam", final.elo_home, final.elo_away)
    final = final.sort_values("date").groupby(["season", "team"]).elo.last().reset_index()
    final.to_sql("elo_season_end", con, if_exists="replace", index=False)
    print(final[final.team == "Arsenal"].tail(10).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
