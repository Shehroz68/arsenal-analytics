import numpy as np

from market import power, proportional


def test_margin_removed():
    odds = np.array([[2.1, 3.4, 3.6], [1.3, 5.5, 11.0]])
    for fn in (power, proportional):
        p = fn(odds)
        assert np.allclose(p.sum(axis=1), 1)
        assert (p > 0).all()


def test_power_shrinks_longshots_more():
    odds = np.array([[1.3, 5.5, 11.0]])
    assert power(odds)[0, 2] < proportional(odds)[0, 2]
