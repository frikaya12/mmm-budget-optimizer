"""Media transformations.

Adstock: an ad keeps working after the week it is shown.   A_t = x_t + lam * A_{t-1}
Saturation (Hill): the 10th million brings less than the 1st.   h(a) = a^s / (K^s + a^s)
Adstock is applied first (exposure accumulates), then saturation.
"""
import numpy as np


def adstock(x: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(np.asarray(x, dtype=float))
    carry = 0.0
    for t, v in enumerate(x):
        carry = v + lam * carry
        out[t] = carry
    return out


def hill(a: np.ndarray, K: float, s: float) -> np.ndarray:
    a = np.maximum(np.asarray(a, dtype=float), 0.0)
    return a ** s / (K ** s + a ** s)


def half_life(lam: float) -> float:
    """Weeks for the effect to halve."""
    return float("inf") if lam >= 1 else (0.0 if lam <= 0 else np.log(0.5) / np.log(lam))


def media_feature(x: np.ndarray, lam: float, K: float, s: float) -> np.ndarray:
    return hill(adstock(x, lam), K, s)
