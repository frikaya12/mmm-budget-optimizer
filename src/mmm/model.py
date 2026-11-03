"""Marketing mix model: additive model with adstock + Hill saturation per channel.

sales_t = intercept + trend + seasonality (Fourier) + price + promo + sum_c beta_c * hill(adstock(spend_c))

- Media coefficients are constrained to be >= 0 (an ad does not destroy sales).
- Adstock and saturation parameters (lam, K, s) are searched to minimise the error on the LAST
  weeks of data, which the model never sees when it is fitted (out-of-sample validation).
- A light ridge penalty stabilises coefficients when channels move together (collinearity).
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from .transforms import media_feature

LAM_RANGE = (0.0, 0.85)
S_RANGE = (0.8, 2.5)


@dataclass
class MMMFit:
    channels: list[str]
    hyper: dict                      # {channel: (lam, K, s)}
    coef: np.ndarray
    columns: list[str]
    val_mape: float | None = None
    train_mape: float | None = None
    extra: dict = field(default_factory=dict)


def controls(df: pd.DataFrame, n_fourier: int = 2) -> dict[str, np.ndarray]:
    t = df["week"].to_numpy(dtype=float)
    cols = {"intercept": np.ones(len(df)), "trend": t / 52.0, "price": df["price"].to_numpy() - 1.0,
            "promo": df["promo"].to_numpy()}
    for k in range(1, n_fourier + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t / 52)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / 52)
    return cols


def design(df: pd.DataFrame, channels: list[str], hyper: dict) -> tuple[np.ndarray, list[str]]:
    cols = controls(df)
    for c in channels:
        lam, K, s = hyper[c]
        cols[f"media_{c}"] = media_feature(df[f"spend_{c}"].to_numpy(), lam, K, s)
    return np.column_stack(list(cols.values())), list(cols)


def solve(X: np.ndarray, y: np.ndarray, columns: list[str], ridge: float = 1e-3) -> np.ndarray:
    lo = np.array([0.0 if c.startswith("media_") else -np.inf for c in columns])
    hi = np.full(len(columns), np.inf)
    scale = np.maximum(np.abs(X).max(axis=0), 1e-9)
    Xs = X / scale
    pen = np.sqrt(ridge * len(y)) * np.eye(X.shape[1])
    pen[columns.index("intercept"), columns.index("intercept")] = 0.0  # do not shrink the intercept
    A = np.vstack([Xs, pen])
    b = np.concatenate([y, np.zeros(X.shape[1])])
    res = lsq_linear(A, b, bounds=(lo, hi), lsmr_tol="auto")
    return res.x / scale


def mape(y, yhat) -> float:
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def predict(fit: MMMFit, df: pd.DataFrame) -> np.ndarray:
    X, _ = design(df, fit.channels, fit.hyper)
    return X @ fit.coef


def fit_fixed(df: pd.DataFrame, channels: list[str], hyper: dict, ridge: float = 1e-3) -> MMMFit:
    X, cols = design(df, channels, hyper)
    coef = solve(X, df["sales"].to_numpy(), cols, ridge)
    f = MMMFit(channels, hyper, coef, cols)
    f.train_mape = mape(df["sales"].to_numpy(), X @ coef)
    return f


def random_hyper(df, channels, rng) -> dict:
    h = {}
    for c in channels:
        x = df[f"spend_{c}"].to_numpy()
        pos = x[x > 0]
        lam = rng.uniform(*LAM_RANGE)
        scale = np.median(pos) / (1 - lam) if pos.size else 1.0   # typical adstocked level
        h[c] = (lam, float(scale * rng.uniform(0.3, 3.0)), rng.uniform(*S_RANGE))
    return h


def search(df: pd.DataFrame, channels: list[str], n_iter: int = 1500, holdout: int = 26,
           ridge: float = 1e-3, seed: int = 0, refine: int = 600, extra_loss=None) -> MMMFit:
    """Random search then local refinement of (lam, K, s), scored on the last `holdout` weeks.

    extra_loss(fit, df) -> float is added to the validation MAPE; it is how field-test results
    (lift tests) calibrate the model (see calibration.py)."""
    rng = np.random.default_rng(seed)
    train, val = df.iloc[:-holdout], df.iloc[-holdout:]
    yv = val["sales"].to_numpy()

    def score(h):
        f = fit_fixed(train, channels, h, ridge)
        m = mape(yv, predict(f, val))
        f.val_mape = m
        return m + (extra_loss(f, df) if extra_loss else 0.0), f

    best_h, best_s, best_f = None, np.inf, None
    for _ in range(n_iter):
        h = random_hyper(df, channels, rng)
        s, f = score(h)
        if s < best_s:
            best_h, best_s, best_f = h, s, f
    for _ in range(refine):  # perturb one channel at a time around the best point
        h = dict(best_h)
        c = channels[rng.integers(len(channels))]
        lam, K, s_ = h[c]
        h[c] = (float(np.clip(lam + rng.normal(0, 0.05), *LAM_RANGE)), float(K * np.exp(rng.normal(0, 0.15))),
                float(np.clip(s_ + rng.normal(0, 0.1), *S_RANGE)))
        sc, f = score(h)
        if sc < best_s:
            best_h, best_s, best_f = h, sc, f
    final = fit_fixed(df, channels, best_h, ridge)  # refit on all weeks with the chosen transforms
    final.val_mape = best_f.val_mape
    final.extra["objective"] = best_s
    return final
