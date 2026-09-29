import pandas as pd

from elo import expected, run_elo


def _matches(rows):
    return pd.DataFrame(rows, columns=["season", "hometeam", "awayteam", "fthg", "ftag"])


def test_expected_symmetry():
    assert abs(expected(1500, 1500, 0) - 0.5) < 1e-12
    assert expected(1500, 1500, 60) > 0.5


def test_ratings_are_pre_match():
    m = _matches([("0001", "A", "B", 3, 0), ("0001", "A", "B", 0, 0)])
    e = run_elo(m, k=20, hfa=0, regress=0)
    # first match must be rated before any result is known
    assert e.elo_home.iloc[0] == e.elo_away.iloc[0] == 1500
    # the winner's rating goes up before the second match
    assert e.elo_home.iloc[1] > 1500 > e.elo_away.iloc[1]


def test_zero_sum():
    m = _matches([("0001", "A", "B", 2, 1), ("0001", "A", "B", 1, 1)])
    e = run_elo(m, k=30, hfa=50, regress=0)
    # whatever A gained in match 1, B lost
    assert abs((e.elo_home.iloc[1] - 1500) + (e.elo_away.iloc[1] - 1500)) < 1e-9


def test_promoted_team_seeded_from_relegated():
    m = _matches([("0001", "A", "B", 5, 0), ("0001", "B", "A", 0, 5),
                  ("0102", "A", "C", 1, 1)])  # B relegated, C promoted
    e = run_elo(m, k=20, hfa=0, regress=0)
    b_final = e.elo_away.iloc[1]  # B's rating going into its last match; B then lost again
    assert e.elo_away.iloc[2] < 1500  # C inherits a below-average (relegated) rating
    assert e.elo_away.iloc[2] < b_final
