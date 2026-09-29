"""Probability-forecast metrics for 3-way (home / draw / away) outcomes.

Probabilities are arrays of shape (n, 3) in the order [home, draw, away];
outcomes are integers 0 = home win, 1 = draw, 2 = away win.
"""
import numpy as np

EPS = 1e-12


def one_hot(y: np.ndarray) -> np.ndarray:
    out = np.zeros((len(y), 3))
    out[np.arange(len(y)), y] = 1
    return out


def log_loss_per_match(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))


def brier_per_match(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((p - one_hot(y)) ** 2).sum(axis=1)


def rps_per_match(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked probability score: respects the ordering home > draw > away (lower is better)."""
    cp = np.cumsum(p, axis=1)[:, :2]
    co = np.cumsum(one_hot(y), axis=1)[:, :2]
    return ((cp - co) ** 2).sum(axis=1) / 2


def summary(p: np.ndarray, y: np.ndarray) -> dict:
    return {
        "n": len(y),
        "log_loss": log_loss_per_match(p, y).mean(),
        "brier": brier_per_match(p, y).mean(),
        "rps": rps_per_match(p, y).mean(),
        "accuracy": (p.argmax(axis=1) == y).mean(),
    }


def result_to_int(ftr) -> np.ndarray:
    """'H' -> 0, 'D' -> 1, 'A' -> 2"""
    ftr = np.asarray(ftr)
    return np.select([ftr == "H", ftr == "D"], [0, 1], 2)
