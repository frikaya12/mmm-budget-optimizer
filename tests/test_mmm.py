import numpy as np
import pytest
from mmm.analysis import bootstrap_roi, optimise_budget, roi, stability
from mmm.calibration import calibration_loss, simulate_lift_test
from mmm.model import fit_fixed, search
from mmm.simulate import CHANNELS, simulate, true_media_sales
from mmm.transforms import adstock, half_life, hill

DF, TRUTH = simulate()
TRUE_HYPER = {c: (TRUTH.lam[c], TRUTH.K[c], TRUTH.s[c]) for c in CHANNELS}


def test_adstock_hill_half_life():
    assert adstock(np.array([1.0, 0, 0]), 0.5).tolist() == [1.0, 0.5, 0.25]
    assert hill(np.array([50.0]), 50.0, 2.0)[0] == pytest.approx(0.5)
    assert half_life(0.5) == pytest.approx(1.0)


def test_true_media_sales_matches_the_simulated_contributions():
    assert true_media_sales(DF, TRUTH) == pytest.approx(sum(TRUTH.extra["true_contribution"].values()))


def test_media_coefficients_are_never_negative():
    f = fit_fixed(DF, CHANNELS, TRUE_HYPER)
    media = [v for v, c in zip(f.coef, f.columns) if c.startswith("media_")]
    assert min(media) >= 0


def test_with_the_true_transforms_most_roi_are_recovered():
    """With the right adstock/saturation, ROI are recovered within 10 %, except META: its spend
    follows the seasons, so part of its effect is confounded with seasonality (documented)."""
    r = roi(fit_fixed(DF, CHANNELS, TRUE_HYPER), DF)
    for c in ["tv", "youtube", "search", "ooh"]:
        assert abs(r[c] / TRUTH.extra["true_roi"][c] - 1) < 0.10


def test_calibration_with_lift_tests_reduces_roi_error():
    tr = TRUTH.extra["true_roi"]
    err = lambda f: np.mean([abs(roi(f, DF)[c] / tr[c] - 1) for c in CHANNELS])
    blind = search(DF, CHANNELS, n_iter=250, refine=120)
    tests = [simulate_lift_test(tr[c], c, seed=i) for i, c in enumerate(["tv", "meta", "search"])]
    calibrated = search(DF, CHANNELS, n_iter=250, refine=120, extra_loss=calibration_loss(tests))
    assert err(calibrated) < err(blind)


def test_budget_optimiser_keeps_budget_and_bounds():
    f = fit_fixed(DF, CHANNELS, TRUE_HYPER)
    plan = optimise_budget(f, DF, max_change=0.2)
    assert sum(plan["new_spend"].values()) == pytest.approx(sum(plan["old_spend"].values()), rel=1e-3)
    assert all(0.8 - 1e-6 <= m <= 1.2 + 1e-6 for m in plan["multipliers"].values())
    assert plan["sales_uplift"] >= -1e-6


def test_bootstrap_and_stability_outputs():
    f = fit_fixed(DF, CHANNELS, TRUE_HYPER)
    ci = bootstrap_roi(f, DF, n=10)
    assert all(lo <= hi for lo, hi in ci.values())
    assert stability(f, DF)["max_abs_change"] >= 0


def test_scale_tests_measure_the_marginal_return():
    from mmm.calibration import combined_loss, model_scale_mroi, simulate_scale_test, true_scale_mroi
    # Saturation: the next euros bring less than the average euro
    up = true_scale_mroi(DF, TRUTH, "search", 0.3)
    assert 0 < up < TRUTH.extra["true_roi"]["search"]
    # The model with the true transforms reproduces the marginal return of the truth
    f = fit_fixed(DF, CHANNELS, TRUE_HYPER)
    assert model_scale_mroi(f, DF, "search", 0.3) == pytest.approx(up, rel=0.25)
    t = simulate_scale_test(DF, TRUTH, "search", 0.3, rel_se=0.15, seed=1)
    assert combined_loss([], [t])(f, DF) < 4          # within 2 standard errors
