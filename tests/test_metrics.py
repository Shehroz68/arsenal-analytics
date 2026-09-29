import numpy as np

from metrics import brier_per_match, result_to_int, rps_per_match


def test_rps_perfect_and_worst():
    y = np.array([0, 2])
    perfect = np.array([[1, 0, 0], [0, 0, 1]])
    worst = np.array([[0, 0, 1], [1, 0, 0]])
    assert np.allclose(rps_per_match(perfect, y), 0)
    assert np.allclose(rps_per_match(worst, y), 1)


def test_rps_rewards_near_misses():
    # outcome was a home win; predicting a draw is "closer" than predicting an away win
    y = np.array([0])
    assert rps_per_match(np.array([[0, 1, 0]]), y) < rps_per_match(np.array([[0, 0, 1]]), y)


def test_brier_uniform():
    p = np.full((1, 3), 1 / 3)
    assert np.isclose(brier_per_match(p, np.array([1]))[0], 2 / 3)


def test_result_to_int():
    assert list(result_to_int(["H", "D", "A"])) == [0, 1, 2]
