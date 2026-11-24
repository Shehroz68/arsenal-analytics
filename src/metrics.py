"""Probability-forecast metrics for 3-way (home / draw / away) outcomes.

Probabilities are arrays of shape (n, 3) in the order [home, draw, away];
outcomes are integers 0 = home win, 1 = draw, 2 = away win.
"""
import numpy as np

EPS = 1e-12


def _validate_outcomes(y):
    y = np.asarray(y)
    if y.ndim != 1 or (y.size and (y.dtype.kind not in "iu" or not np.isin(y, [0, 1, 2]).all())):
        raise ValueError("Outcomes must be a one-dimensional array of integers 0, 1, or 2")
    return y.astype(int, copy=False)


def _validate_forecasts(p, y):
    y = _validate_outcomes(y)
    p = np.asarray(p)
    if p.shape != (len(y), 3):
        raise ValueError("Forecasts must have shape (number of outcomes, 3)")
    if p.dtype.kind not in "iuf" or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Forecast probabilities must be finite numbers between 0 and 1")
    if not np.allclose(p.sum(axis=1), 1, rtol=1e-5, atol=1e-8):
        raise ValueError("Forecast probabilities must sum to 1 for each match")
    return p, y


def one_hot(y: np.ndarray) -> np.ndarray:
    y = _validate_outcomes(y)
    out = np.zeros((len(y), 3))
    out[np.arange(len(y)), y] = 1
    return out


def log_loss_per_match(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p, y = _validate_forecasts(p, y)
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))


def brier_per_match(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p, y = _validate_forecasts(p, y)
    return ((p - one_hot(y)) ** 2).sum(axis=1)


def rps_per_match(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked probability score: respects the ordering home > draw > away (lower is better)."""
    p, y = _validate_forecasts(p, y)
    cp = np.cumsum(p, axis=1)[:, :2]
    co = np.cumsum(one_hot(y), axis=1)[:, :2]
    return ((cp - co) ** 2).sum(axis=1) / 2


def summary(p: np.ndarray, y: np.ndarray) -> dict:
    p, y = _validate_forecasts(p, y)
    return {
        "n": len(y),
        "log_loss": log_loss_per_match(p, y).mean(),
        "brier": brier_per_match(p, y).mean(),
        "rps": rps_per_match(p, y).mean(),
        "accuracy": (p.argmax(axis=1) == y).mean(),
    }


def result_to_int(ftr) -> np.ndarray:
    """Map H/D/A to 0/1/2; reject missing or unknown match results."""
    ftr = np.asarray(ftr)
    if not np.isin(ftr, ["H", "D", "A"]).all():
        raise ValueError("Match results must contain only H, D, or A")
    return np.select([ftr == "H", ftr == "D"], [0, 1], 2)
