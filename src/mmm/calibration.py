"""Calibrate the model with field tests (lift tests).

A model can predict sales very well and still attribute them to the wrong channels: when
channels move together, many combinations of effects fit the data equally well. Field tests
(e.g. switching a channel off in some regions) measure the true incremental effect of a channel.
We add a penalty when the model's ROI for a tested channel is far from the measured one, in
units of the test's standard error.
"""
from dataclasses import dataclass
import numpy as np
from .analysis import roi


@dataclass
class LiftTest:
    channel: str
    roi: float      # measured incremental sales per euro
    se: float       # standard error of the measurement


def simulate_lift_test(true_roi: float, channel: str, rel_se: float = 0.15, seed: int = 0) -> LiftTest:
    """A noisy measurement of the true ROI, as a real geo-test would give."""
    rng = np.random.default_rng(seed)
    se = rel_se * true_roi
    return LiftTest(channel, float(true_roi + rng.normal(0, se)), float(se))


def calibration_loss(tests: list[LiftTest], weight: float = 1.0):
    def loss(fit, df):
        r = roi(fit, df)
        return weight * float(np.mean([((r[t.channel] - t.roi) / t.se) ** 2 for t in tests]))
    return loss


# ---------------------------------------------------------------------------------------------
# Marginal tests. A classic lift test (channel switched off) measures the AVERAGE return of the
# money already spent. A budget decision needs the MARGINAL return: what the next (or the last)
# euros bring. A "scale" test changes a channel's spend by `step` (e.g. +30 % in test regions)
# and measures the incremental sales per incremental euro.

@dataclass
class ScaleTest:
    channel: str
    step: float     # relative change of spend tested, e.g. +0.3 or -0.3
    mroi: float     # measured incremental sales per incremental euro
    se: float


def _scaled_sales(predict_fn, df, channel: str, step: float) -> float:
    d = df.copy()
    d[f"spend_{channel}"] = d[f"spend_{channel}"] * (1 + step)
    return predict_fn(d)


def true_scale_mroi(df, truth, channel: str, step: float) -> float:
    from .simulate import true_media_sales
    base = true_media_sales(df, truth)
    new = true_media_sales(df, truth, {channel: 1 + step})
    return (new - base) / (step * df[f"spend_{channel}"].sum())


def simulate_scale_test(df, truth, channel: str, step: float = 0.3, rel_se: float = 0.15, seed: int = 0) -> ScaleTest:
    rng = np.random.default_rng(seed)
    m = true_scale_mroi(df, truth, channel, step)
    se = rel_se * abs(m) + 1e-6
    return ScaleTest(channel, step, float(m + rng.normal(0, se)), float(se))


def model_scale_mroi(fit, df, channel: str, step: float) -> float:
    from .model import predict
    base = predict(fit, df).sum()
    new = _scaled_sales(lambda d: predict(fit, d).sum(), df, channel, step)
    return (new - base) / (step * df[f"spend_{channel}"].sum())


def combined_loss(lift: list[LiftTest], scale: list[ScaleTest], weight: float = 1.0):
    """Penalty for both kinds of tests, in standard errors."""
    def loss(fit, df):
        terms = []
        if lift:
            r = roi(fit, df)
            terms += [((r[t.channel] - t.roi) / t.se) ** 2 for t in lift]
        terms += [((model_scale_mroi(fit, df, t.channel, t.step) - t.mroi) / t.se) ** 2 for t in scale]
        return weight * float(np.mean(terms)) if terms else 0.0
    return loss
