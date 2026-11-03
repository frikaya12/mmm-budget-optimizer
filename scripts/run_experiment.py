"""Full experiment: simulate data with a known truth, fit the MMM blind, then calibrate it with
lift tests, and check every recommendation against the truth. Writes reports/report.md + figures.

  python scripts/run_experiment.py            # about 1 minute
  python scripts/run_experiment.py --fast     # fewer search iterations, for a quick look
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmm.analysis import bootstrap_roi, marginal_roi, optimise_budget, response_curve, roi, stability  # noqa: E402
from mmm.calibration import calibration_loss, simulate_lift_test  # noqa: E402
from mmm.model import predict, search  # noqa: E402
from mmm.simulate import CHANNELS, simulate, true_media_sales  # noqa: E402
from mmm.transforms import half_life  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--fast", action="store_true")
p.add_argument("--tested", default="tv,meta,search", help="channels with a lift test")
p.add_argument("--max-change", type=float, default=0.3)
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
n_iter, refine, n_boot = (300, 150, 40) if a.fast else (800, 400, 150)
out = ROOT / "reports"; out.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#1f4e79", "#d9822b", "#9a9a9a"

df, truth = simulate(seed=a.seed)
true_roi = truth.extra["true_roi"]
tested = [c for c in a.tested.split(",") if c]
tests = [simulate_lift_test(true_roi[c], c, seed=i) for i, c in enumerate(tested)]

models = {"Sans calibration": search(df, CHANNELS, n_iter=n_iter, refine=refine),
          "Calibré (tests terrain)": search(df, CHANNELS, n_iter=n_iter, refine=refine,
                                            extra_loss=calibration_loss(tests))}
base_true = true_media_sales(df, truth)
res = {}
for name, f in models.items():
    r = roi(f, df)
    plan = optimise_budget(f, df, a.max_change)
    res[name] = {"val_mape": f.val_mape, "roi": r, "mroi": marginal_roi(f, df),
                 "roi_error": float(np.mean([abs(r[c] / true_roi[c] - 1) for c in CHANNELS])),
                 "ci": bootstrap_roi(f, df, n=n_boot), "stability": stability(f, df)["max_abs_change"],
                 "plan": plan, "true_gain": true_media_sales(df, truth, plan["multipliers"]) - base_true,
                 "hyper": f.hyper}

# Best plan under the truth (only possible with simulated data)
from scipy.optimize import minimize  # noqa: E402
sp = np.array([df[f"spend_{c}"].sum() for c in CHANNELS])
opt = minimize(lambda m: -true_media_sales(df, truth, dict(zip(CHANNELS, m))) / 1e3, np.ones(len(CHANNELS)),
               method="SLSQP", bounds=[(1 - a.max_change, 1 + a.max_change)] * len(CHANNELS),
               constraints=[{"type": "eq", "fun": lambda m: (m * sp).sum() - sp.sum()}])
best_true_gain = true_media_sales(df, truth, dict(zip(CHANNELS, opt.x))) - base_true

# ---- Figures
f_cal = models["Calibré (tests terrain)"]
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.plot(df["week"], df["sales"], color=GREY, lw=1, label="Ventes observées")
ax.plot(df["week"], predict(f_cal, df), color=BLUE, lw=1.2, label="Modèle calibré")
ax.axvspan(len(df) - 26, len(df), color=ORANGE, alpha=0.12, label="Semaines de validation")
ax.set_xlabel("Semaine"); ax.set_ylabel("Ventes (k€)"); ax.legend(frameon=False, ncol=3, fontsize=8)
fig.tight_layout(); fig.savefig(out / "fit.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3))
x = np.arange(len(CHANNELS)); w = 0.27
ax.bar(x - w, [true_roi[c] for c in CHANNELS], w, color=GREY, label="ROI réel (vérité simulée)")
for k, (name, col) in enumerate([("Sans calibration", ORANGE), ("Calibré (tests terrain)", BLUE)]):
    vals = [res[name]["roi"][c] for c in CHANNELS]
    lo = [vals[i] - res[name]["ci"][c][0] for i, c in enumerate(CHANNELS)]
    hi = [res[name]["ci"][c][1] - vals[i] for i, c in enumerate(CHANNELS)]
    ax.bar(x + k * w, vals, w, color=col, label=name, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], capsize=2)
ax.set_xticks(x); ax.set_xticklabels([c.upper() for c in CHANNELS]); ax.set_ylabel("Ventes par euro")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(out / "roi.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3))
for c in CHANNELS:
    rc = response_curve(f_cal, df, c)
    ax.plot(rc["spend"] / 1e3, rc["incremental_sales"] / 1e3, label=c.upper())
ax.set_xlabel("Dépense totale du canal (M€)"); ax.set_ylabel("Ventes incrémentales (M€)")
ax.legend(frameon=False, ncol=5, fontsize=8); fig.tight_layout(); fig.savefig(out / "response_curves.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 2.8))
names = list(res) + ["Meilleur plan possible"]
promised = [res[n]["plan"]["sales_uplift"] for n in res] + [np.nan]
real = [res[n]["true_gain"] for n in res] + [best_true_gain]
xx = np.arange(len(names))
ax.bar(xx - 0.2, promised, 0.4, color=GREY, label="Gain promis par le modèle")
ax.bar(xx + 0.2, real, 0.4, color=[ORANGE, BLUE, "#5a8f3d"], label="Gain réel (vérité simulée)")
ax.axhline(0, color="black", lw=0.6); ax.set_xticks(xx); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("Ventes (k€)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(out / "plans.png", dpi=150); plt.close(fig)

# ---- Report
def pct(v): return f"{v * 100:.0f} %"
def keur(v): return f"{v:,.0f} k€".replace(",", "\u202f")
L = ["# Résultats de l'expérience", "",
     f"Données simulées : {len(df)} semaines, {len(CHANNELS)} canaux. Tests terrain simulés sur : "
     f"{', '.join(tested)} (erreur de mesure 15 %). Réallocation limitée à ±{int(a.max_change * 100)} % par canal.", "",
     "## Qualité de prédiction et justesse des ROI", "",
     "| Modèle | Erreur de prédiction (MAPE, 26 dernières semaines) | Erreur moyenne sur les ROI | Variation max. des ROI (stabilité) |",
     "|---|---|---|---|"]
for n, r in res.items():
    L.append(f"| {n} | {r['val_mape']:.1f} % | {pct(r['roi_error'])} | {pct(r['stability'])} |")
L += ["", "![](fit.png)", "", "## ROI par canal", "",
      "| Canal | ROI réel | " + " | ".join(f"{n} (IC 90 %)" for n in res) + " |", "|---" * (2 + len(res)) + "|"]
for c in CHANNELS:
    cells = [f"{res[n]['roi'][c]:.2f} [{res[n]['ci'][c][0]:.2f} ; {res[n]['ci'][c][1]:.2f}]" for n in res]
    L.append(f"| {c.upper()} | {true_roi[c]:.2f} | " + " | ".join(cells) + " |")
L += ["", "![](roi.png)", "", "## Rémanence (adstock) retrouvée par le modèle calibré", "",
      "| Canal | λ réel | λ estimé | Demi-vie estimée (semaines) |", "|---|---|---|---|"]
for c in CHANNELS:
    lam = res["Calibré (tests terrain)"]["hyper"][c][0]
    L.append(f"| {c.upper()} | {truth.lam[c]:.2f} | {lam:.2f} | {half_life(lam):.1f} |")
L += ["", "![](response_curves.png)", "", "## Réallocation du budget (budget constant)", "",
      "| Modèle | Part du budget réallouée | Gain promis | Gain réel |", "|---|---|---|---|"]
for n, r in res.items():
    L.append(f"| {n} | {pct(r['plan']['reallocated_share'])} | {keur(r['plan']['sales_uplift'])} | {keur(r['true_gain'])} |")
L.append(f"| Meilleur plan possible (connaissant la vérité) | | | {keur(best_true_gain)} |")
L += ["", "![](plans.png)", ""]
(out / "report.md").write_text("\n".join(L), encoding="utf-8")
(out / "results.json").write_text(json.dumps({n: {k: v for k, v in r.items() if k != "hyper"} for n, r in res.items()},
                                             indent=2, default=float), encoding="utf-8")
print("\n".join(L))
