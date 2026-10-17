"""Equal-force-evaluation figure for EB vs E1 (descriptive; unpaired allocations).
    python scripts/convergence_fr/fig_equal_budget.py -> figures/convergence_fr/equal_budget_ite300.{pdf,png}"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eri_campaign"))
from figstyle import ABF, FR, INK2, SERIES, setup  # noqa: E402

plt = setup()
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
p = os.path.join(ROOT, "results", "convergence_fr_campaign", "analysis", "EB_equal_budget")
R = json.load(open(p + ".json"))
z = np.load(p + "_curves.npz")
labels = [str(x) for x in z["labels"]]
cols = [SERIES[2], ABF, FR]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9))
ax = axs[0]
for i, (lab, col) in enumerate(zip(labels, cols)):
    B, E = z[f"B_{i}"] / 1e8, z[f"eF_{i}"]
    ax.fill_between(B, np.quantile(E, 0.25, 1), np.quantile(E, 0.75, 1), color=col, alpha=0.15, lw=0)
    ax.plot(B, np.median(E, 1), color=col, lw=1.6, label=lab.split(" (")[0])
ax.set_yscale("log")
ax.set_ylim(0.025, 2)
ax.set_xlabel("molecular force evaluations (10⁸)")
ax.set_ylabel("e_F (kJ/mol), median and 25–75 %")
ax.legend(fontsize=7.5)
ax = axs[1]
eps = [r["eps_kJmol"] for r in R["rungs"]]
for i, (lab, col) in enumerate(zip(labels, cols)):
    y = [r[lab]["median_restricted"] for r in R["rungs"]]
    ax.plot(range(len(eps)), y, "o-", color=col, lw=1.4, ms=5, label=lab.split(" (")[0])
ax.set_xticks(range(len(eps)))
ax.set_xticklabels([f"{e:.2f}" for e in eps])
ax.set_xlabel("tolerance ε (kJ/mol)")
ax.set_ylabel("median sustained τ_ε (10⁸ force evals)")
ax.axhline(R["budget_force_evals"] / 1e8, color=INK2, lw=0.8, ls=":")
ax.legend(fontsize=7.5)
fig.suptitle("ITE 300 K at an equal budget of 5.12×10⁸ force evaluations: ABF K 256 × 200 t.u. (EB) vs ABF and "
             "ABF + FR (stopped at 10 t.u.) at K 1024 × 50 t.u. (E1)", x=0.01, y=1.03, ha="left", fontsize=9.5)
fig.text(0.01, -0.03, "Allocations are unpaired (different seeds and K); 16 seeds each. Dotted: the budget "
         "(censoring). Equilibration excluded.", fontsize=7, color=INK2)
fig.tight_layout()
out = os.path.join(ROOT, "figures", "convergence_fr", "equal_budget_ite300")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight")
print("wrote", out)
