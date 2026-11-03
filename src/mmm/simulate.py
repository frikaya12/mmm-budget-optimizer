"""Simulated weekly data with a KNOWN ground truth.

Real MMM data never tells you the true ROI, so you cannot check that a model is right. Here we
generate sales from known effects (adstock, saturation, ROI per channel, seasonality, price,
promotions, noise), fit the model blind, and measure how well it recovers the truth.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .transforms import media_feature

CHANNELS = ["tv", "meta", "youtube", "search", "ooh"]


@dataclass
class Truth:
    lam: dict
    K: dict
    s: dict
    beta: dict
    base: float
    trend: float
    season_amp: float
    price_elasticity: float
    promo_lift: float
    noise_sd: float
    extra: dict = field(default_factory=dict)


DEFAULT_TRUTH = Truth(
    lam={"tv": 0.65, "meta": 0.30, "youtube": 0.45, "search": 0.05, "ooh": 0.55},
    K={"tv": 180.0, "meta": 60.0, "youtube": 55.0, "search": 35.0, "ooh": 70.0},
    s={"tv": 1.6, "meta": 1.1, "youtube": 1.3, "search": 1.0, "ooh": 1.4},
    beta={"tv": 520.0, "meta": 300.0, "youtube": 160.0, "search": 260.0, "ooh": 70.0},
    base=1500.0, trend=1.2, season_amp=180.0, price_elasticity=-900.0, promo_lift=160.0, noise_sd=45.0,
)


def simulate(weeks: int = 156, truth: Truth = DEFAULT_TRUTH, seed: int = 7) -> tuple[pd.DataFrame, Truth]:
    rng = np.random.default_rng(seed)
    t = np.arange(weeks)
    # Spend patterns (k€ per week): TV and OOH in campaign bursts, digital always-on with noise
    tv = np.where((t % 13) < 5, rng.uniform(120, 220, weeks), 0.0)
    ooh = np.where((t % 26) < 6, rng.uniform(50, 90, weeks), 0.0)
    meta = np.clip(40 + 15 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 8, weeks), 5, None)
    youtube = np.where((t % 8) < 4, rng.uniform(30, 60, weeks), rng.uniform(0, 10, weeks))
    search = np.clip(25 + 6 * np.sin(2 * np.pi * (t + 6) / 52) + rng.normal(0, 4, weeks), 5, None)
    spend = {"tv": tv, "meta": meta, "youtube": youtube, "search": search, "ooh": ooh}

    price = 1 + 0.03 * np.sin(2 * np.pi * t / 26) + rng.normal(0, 0.01, weeks)   # price index
    promo = (rng.uniform(size=weeks) < 0.12).astype(float)                       # promo weeks
    season = truth.season_amp * np.sin(2 * np.pi * (t - 10) / 52)
    base = truth.base + truth.trend * t + season + truth.price_elasticity * (price - 1) + truth.promo_lift * promo
    contrib = {c: truth.beta[c] * media_feature(spend[c], truth.lam[c], truth.K[c], truth.s[c]) for c in CHANNELS}
    sales = base + sum(contrib.values()) + rng.normal(0, truth.noise_sd, weeks)

    df = pd.DataFrame({"week": t, "sales": sales, "price": price, "promo": promo,
                       **{f"spend_{c}": spend[c] for c in CHANNELS}})
    truth.extra = {"true_contribution": {c: float(contrib[c].sum()) for c in CHANNELS},
                   "true_roi": {c: float(contrib[c].sum() / spend[c].sum()) for c in CHANNELS}}
    return df, truth


def true_media_sales(df: pd.DataFrame, truth: Truth, multipliers: dict | None = None) -> float:
    """Media-driven sales under the TRUE model for a given budget plan (channel multipliers).
    Only possible with simulated data: it tells us whether a recommendation really helps."""
    multipliers = multipliers or {}
    total = 0.0
    for c in CHANNELS:
        x = df[f"spend_{c}"].to_numpy() * float(multipliers.get(c, 1.0))
        total += float((truth.beta[c] * media_feature(x, truth.lam[c], truth.K[c], truth.s[c])).sum())
    return total
