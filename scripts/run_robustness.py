"""Is the main result a lucky draw? Repeat the experiment on many simulated markets.

Each seed draws new spend patterns, noise, promotions and test errors (same true effects).
Three models are compared on each market:
  raw   fitted on sales only
  cal   + classic lift tests (channel switched off: measures the AVERAGE return)
  marg  + scale tests (spend changed by +30 % and -30 %: measures the MARGINAL return)
For each one: compare the ROI to the truth, and check whether the recommended budget plan
really increases sales.

  python scripts/run_robustness.py --seeds 20        # about 6 minutes
Writes reports/robustness.md, reports/robustness.json and reports/robustness.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmm.analysis import optimise_budget, roi  # noqa: E402
from mmm.calibration import calibration_loss, combined_loss, simulate_lift_test, simulate_scale_test  # noqa: E402
from mmm.model import search  # noqa: E402
from mmm.simulate import CHANNELS, simulate, true_media_sales  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=20)
p.add_argument("--first-seed", type=int, default=100, help="seed 7 is the showcase run; use others")
p.add_argument("--tested", default="tv,meta,youtube,search,ooh")
p.add_argument("--max-change", type=float, default=0.3)
p.add_argument("--fast", action="store_true")
a = p.parse_args()
n_iter, refine = (300, 150) if a.fast else (800, 400)
out = ROOT / "reports"; out.mkdir(exist_ok=True)
tested = [c for c in a.tested.split(",") if c]
MODELS = {"raw": "Sans calibration", "cal": "Calibré, ROI moyen (tests d'arrêt)",
          "marg": "Calibré, ROI marginal (tests ±30 %)"}

rows = []
for s in range(a.first_seed, a.first_seed + a.seeds):
    df, truth = simulate(seed=s)
    true_roi = dict(truth.extra["true_roi"])
    lift = [simulate_lift_test(true_roi[c], c, seed=1000 * s + i) for i, c in enumerate(tested)]
    scale = [simulate_scale_test(df, truth, c, st, seed=2000 * s + 10 * i + j)
             for i, c in enumerate(tested) for j, st in enumerate((0.3, -0.3))]
    base_true = true_media_sales(df, truth)
    sp = np.array([df[f"spend_{c}"].sum() for c in CHANNELS])
    opt = minimize(lambda m: -true_media_sales(df, truth, dict(zip(CHANNELS, m))) / 1e3, np.ones(len(CHANNELS)),
                   method="SLSQP", bounds=[(1 - a.max_change, 1 + a.max_change)] * len(CHANNELS),
                   constraints=[{"type": "eq", "fun": lambda m: (m * sp).sum() - sp.sum()}])
    row = {"seed": s, "best_gain": true_media_sales(df, truth, dict(zip(CHANNELS, opt.x))) - base_true}
    for name, extra in (("raw", None), ("cal", calibration_loss(lift)), ("marg", combined_loss(lift, scale))):
        f = search(df, CHANNELS, n_iter=n_iter, refine=refine, seed=s, extra_loss=extra)
        r = roi(f, df)
        plan = optimise_budget(f, df, a.max_change)
        row[name] = {"val_mape": f.val_mape,
                     "roi_error": float(np.mean([abs(r[c] / true_roi[c] - 1) for c in CHANNELS])),
                     "promised": plan["sales_uplift"],
                     "real": true_media_sales(df, truth, plan["multipliers"]) - base_true}
    rows.append(row)
    print(s, {k: (round(row[k]["val_mape"], 1), round(row[k]["roi_error"], 2), round(row[k]["real"])) for k in MODELS},
          round(row["best_gain"]), flush=True)

(out / "robustness.json").write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")


def col(m, k):
    return np.array([r[m][k] for r in rows])


N = len(rows)
best = np.array([r["best_gain"] for r in rows])


def q(v, pct=False):
    m, lo, hi = np.percentile(v, [50, 10, 90])
    if pct:
        return f"{m * 100:.0f} % ({lo * 100:.0f} ; {hi * 100:.0f})"
    return f"{m:.1f} % ({lo:.1f} ; {hi:.1f})"


def keur(v):
    return f"{v:,.0f}".replace(",", " ")


L = ["# Robustesse : le résultat tient-il sur d'autres marchés simulés ?", "",
     f"{N} marchés simulés (graines {a.first_seed} à {a.first_seed + N - 1}) : nouvelles dépenses, promotions, bruit "
     f"et erreurs de mesure des tests (15 %) à chaque fois. Tests sur : {', '.join(tested)}. "
     f"Réallocation limitée à ±{int(a.max_change * 100)} % par canal, budget total constant.", "",
     "- **Test d'arrêt** (lift test classique) : on coupe le canal dans des régions tests ; mesure le ROI **moyen**.",
     "- **Test ±30 %** : on augmente ou baisse la dépense du canal de 30 % dans des régions tests ; mesure le ROI "
     "**marginal**, c'est-à-dire ce que rapportent les euros qu'on envisage de déplacer.", "",
     "| Médiane (P10 ; P90) sur les marchés | " + " | ".join(MODELS.values()) + " |", "|---" * (1 + len(MODELS)) + "|",
     "| Erreur de prédiction (MAPE, 26 semaines non vues) | " + " | ".join(q(col(m, "val_mape")) for m in MODELS) + " |",
     "| Erreur moyenne sur les ROI | " + " | ".join(q(col(m, "roi_error"), pct=True) for m in MODELS) + " |",
     "| Marchés où le plan recommandé **fait perdre** des ventes | " + " | ".join(f"{int((col(m, 'real') < 0).sum())} / {N}" for m in MODELS) + " |",
     "| Gain réel du plan, médiane (k€) | " + " | ".join(keur(np.median(col(m, "real"))) for m in MODELS) + " |",
     "| Gain promis par le modèle, médiane (k€) | " + " | ".join(keur(np.median(col(m, "promised"))) for m in MODELS) + " |",
     "| Part du meilleur gain possible obtenue, médiane | " + " | ".join(
         f"{np.median(np.clip(col(m, 'real') / best, -5, 1)) * 100:.0f} %" for m in MODELS) + " |",
     f"| Meilleur gain possible, médiane (k€, connaissant la vérité) | {keur(np.median(best))} | | |", "",
     "![](robustness.png)"]
(out / "robustness.md").write_text("\n".join(L), encoding="utf-8")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
colors = {"raw": "#d9822b", "cal": "#9a9a9a", "marg": "#1f4e79"}
short = {"raw": "Sans\ncalibration", "cal": "ROI moyen\ncalibré", "marg": "ROI marginal\ncalibré"}
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3))
ax[0].boxplot([col(m, "roi_error") * 100 for m in MODELS], tick_labels=[short[m] for m in MODELS])
ax[0].set_ylabel("Erreur moyenne sur les ROI (%)")
for i, m in enumerate(MODELS):
    ax[1].scatter(np.full(N, i) + np.linspace(-0.15, 0.15, N), col(m, "real"), color=colors[m], s=12)
ax[1].axhline(0, color="black", lw=0.6)
ax[1].set_xticks(range(len(MODELS))); ax[1].set_xticklabels([short[m] for m in MODELS])
ax[1].set_ylabel("Gain réel du plan (k€)")
fig.tight_layout(); fig.savefig(out / "robustness.png", dpi=150); plt.close(fig)
print("\n".join(L))
