"""The most important property of the whole project: features never see the match they predict."""
import numpy as np
import pandas as pd

from features import team_features


def _team_matches():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        gf, ga = rng.integers(0, 4, 2)
        rows.append(dict(match_id=i, season="0001", date=f"2000-{8 + i // 10:02d}-{1 + i % 10 * 2:02d}",
                         team="A", opponent="B", is_home=i % 2, gf=gf, ga=ga, shots_f=10, shots_a=10,
                         sot_f=4, sot_a=4, corners_f=5, corners_a=5,
                         points=3 if gf > ga else 1 if gf == ga else 0))
    return pd.DataFrame(rows)


def test_rolling_features_ignore_current_match():
    tm = _team_matches()
    base = team_features(tm).set_index("match_id")
    # change the result of match 20 drastically; its own features must not move
    tm2 = tm.copy()
    tm2.loc[tm2.match_id == 20, ["gf", "points"]] = [9, 3]
    after = team_features(tm2).set_index("match_id")
    cols = [c for c in base.columns if c not in ("team", "is_home")]
    pd.testing.assert_frame_equal(base.loc[:20, cols], after.loc[:20, cols])
    # ...but later matches should see it
    assert after.loc[21, "gf_r5"] != base.loc[21, "gf_r5"]


def test_first_match_has_no_history():
    f = team_features(_team_matches())
    assert np.isnan(f.iloc[0]["gf_r5"])
    assert f.iloc[0]["matches_this_season"] == 0
