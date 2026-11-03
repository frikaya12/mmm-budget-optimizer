"""What the client needs: contributions, ROI, marginal ROI, uncertainty, stability, and a
budget reallocation under realistic constraints."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .model import MMMFit, fit_fixed, predict


def contributions(fit: MMMFit, df: pd.DataFrame) -> dict[str, float]:
    """Sales due to each channel = prediction - prediction with that channel switched off."""
    base = predict(fit, df)
    out = {}
    for c in fit.channels:
        d = df.copy()
        d[f"spend_{c}"] = 0.0
        out[c] = float((base - predict(fit, d)).sum())
    return out


def roi(fit: MMMFit, df: pd.DataFrame) -> dict[str, float]:
    contrib = contributions(fit, df)
    return {c: contrib[c] / df[f"spend_{c}"].sum() for c in fit.channels}


def marginal_roi(fit: MMMFit, df: pd.DataFrame, step: float = 0.01) -> dict[str, float]:
    """Sales from the NEXT euro: spend the same pattern +1 %, measure the extra sales."""
    base = predict(fit, df).sum()
    out = {}
    for c in fit.channels:
        d = df.copy()
        d[f"spend_{c}"] = d[f"spend_{c}"] * (1 + step)
        out[c] = float((predict(fit, d).sum() - base) / (df[f"spend_{c}"].sum() * step))
    return out


def response_curve(fit: MMMFit, df: pd.DataFrame, channel: str, multipliers=np.linspace(0, 2, 21)) -> pd.DataFrame:
    rows = []
    off = df.copy()
    off[f"spend_{channel}"] = 0.0
    base = predict(fit, off).sum()
    for m in multipliers:
        d = df.copy()
        d[f"spend_{channel}"] = d[f"spend_{channel}"] * m
        rows.append({"multiplier": m, "spend": d[f"spend_{channel}"].sum(), "incremental_sales": predict(fit, d).sum() - base})
    return pd.DataFrame(rows)


def bootstrap_roi(fit: MMMFit, df: pd.DataFrame, n: int = 200, block: int = 8, seed: int = 0) -> dict:
    """Uncertainty: resample residuals by blocks of weeks (keeps autocorrelation), refit the
    coefficients with the same transforms, recompute ROI. Returns a 90 % interval centred on
    the estimate (the spread of the bootstrap draws around their median)."""
    rng = np.random.default_rng(seed)
    yhat = predict(fit, df)
    resid = df["sales"].to_numpy() - yhat
    T = len(df)
    draws = {c: [] for c in fit.channels}
    for _ in range(n):
        starts = rng.integers(0, T - block, size=T // block + 1)
        r = np.concatenate([resid[s:s + block] for s in starts])[:T]
        d = df.copy()
        d["sales"] = yhat + r
        f = fit_fixed(d, fit.channels, fit.hyper)
        for c, v in roi(f, df).items():
            draws[c].append(v)
    point = roi(fit, df)
    # Interval centred on the model's estimate: point + (spread of the bootstrap around its median)
    return {c: (max(0.0, point[c] + float(np.percentile(v, 5) - np.median(v))),
                point[c] + float(np.percentile(v, 95) - np.median(v))) for c, v in draws.items()}


def stability(fit: MMMFit, df: pd.DataFrame, cut: int = 13) -> dict:
    """Refit without the first / last `cut` weeks. Large ROI swings = not defensible."""
    full = roi(fit, df)
    out = {}
    for name, d in {"sans_debut": df.iloc[cut:], "sans_fin": df.iloc[:-cut]}.items():
        f = fit_fixed(d.reset_index(drop=True), fit.channels, fit.hyper)
        r = roi(f, d.reset_index(drop=True))
        out[name] = {c: r[c] / full[c] - 1 if full[c] else 0.0 for c in fit.channels}
    out["max_abs_change"] = max(abs(v) for k in ("sans_debut", "sans_fin") for v in out[k].values())
    return out


def optimise_budget(fit: MMMFit, df: pd.DataFrame, max_change: float = 0.3) -> dict:
    """Same total budget, each channel scaled by a factor in [1 - max_change, 1 + max_change]
    (keeping its weekly pattern). Maximise predicted sales. Bounds keep the plan realistic:
    the response curves are uncertain far from observed spend, and media plans have constraints."""
    chans = fit.channels
    spend = np.array([df[f"spend_{c}"].sum() for c in chans])
    budget = spend.sum()

    def sales(m):
        d = df.copy()
        for c, k in zip(chans, m):
            d[f"spend_{c}"] = d[f"spend_{c}"] * k
        return predict(fit, d).sum()

    cons = [{"type": "eq", "fun": lambda m: (m * spend).sum() - budget}]
    res = minimize(lambda m: -sales(m) / 1e3, np.ones(len(chans)), method="SLSQP",
                   bounds=[(1 - max_change, 1 + max_change)] * len(chans), constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    m = res.x
    new = m * spend
    base_sales, new_sales = sales(np.ones(len(chans))), sales(m)
    return {"multipliers": dict(zip(chans, m.round(3))), "old_spend": dict(zip(chans, spend.round(1))),
            "new_spend": dict(zip(chans, new.round(1))),
            "reallocated_share": float(np.abs(new - spend).sum() / 2 / budget),
            "sales_uplift": float(new_sales - base_sales), "sales_uplift_pct": float((new_sales - base_sales) / base_sales),
            "success": bool(res.success)}
