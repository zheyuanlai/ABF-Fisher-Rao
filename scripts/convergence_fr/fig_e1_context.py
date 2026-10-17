"""DESCRIPTIVE context figure: E1 (ABF vs ABF + FR stopped at 10 t.u., confirmatory seeds) next to X1 (ABF vs ABF +
continuous FR, exploratory seeds).  Two separate paired two-arm experiments on different seed blocks: the panels
show each block's OWN paired effect; no cross-block pairing is implied.

    python scripts/convergence_fr/fig_e1_context.py  -> figures/convergence_fr/early_stopping_vs_continuous.{pdf,png}
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "eri_campaign"))
import atlas_lib as L  # noqa: E402
from figstyle import ABF, FR, INK2, MUTED, SERIES, setup  # noqa: E402

plt = setup()
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
AN = os.path.join(ROOT, "results", "convergence_fr_campaign", "analysis")
CONT = SERIES[6]          # continuous-FR block colour (violet), early-stopping block uses FR orange


def main():
    E = json.load(open(os.path.join(AN, "E1.json")))
    X = json.load(open(os.path.join(AN, "X1.json")))
    ze = np.load(os.path.join(AN, "E1_curves.npz"))
    zx = np.load(os.path.join(AN, "X1_curves.npz"))
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    ax = axs[0]
    for z, lab, col, ls in ((zx, "X1: ABF", ABF, ":"), (zx, "X1: ABF + continuous FR", CONT, ":"),
                            (ze, "E1: ABF", ABF, "-"), (ze, "E1: ABF + FR stopped at 10 t.u.", FR, "-")):
        key = "eF_abf" if "ABF" == lab.split(": ")[1] else "eF_fr"
        ax.plot(z["t"], np.median(z[key], 1), color=col, ls=ls, lw=1.5, label=lab)
    ax.set_yscale("log")
    ax.set_ylabel("median e_F(t) (kJ/mol)")
    ax.legend(fontsize=7, loc="upper right")
    ax = axs[1]
    for z, lab, col in ((zx, "X1: continuous FR vs its ABF", CONT), (ze, "E1: FR stopped at 10 t.u. vs its ABF", FR)):
        rel = 100 * (z["eF_fr"] / z["eF_abf"] - 1)
        med, lo, hi = L.boot_median_curve(rel)
        ax.fill_between(z["t"], lo, hi, color=col, alpha=0.2, lw=0)
        ax.plot(z["t"], med, color=col, lw=1.5, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("paired e_F(FR)/e_F(ABF) − 1 (%), median and 95 % CI")
    ax.legend(fontsize=7, loc="lower right")
    for ax in axs[:2]:
        ax.axvline(1, color=MUTED, ls=":", lw=1)
        ax.axvline(10, color=FR, ls="--", lw=1)
        ax.set_xlabel("t (t.u.)")
    ax = axs[2]
    rows_e = E["time_to_accuracy"]["sustained"]
    rows_x = X["time_to_accuracy"]["sustained"]
    eps = [r["eps_kJmol"] for r in rows_e]
    for rows, col, lab, off in ((rows_x, CONT, "X1 continuous", -0.06), (rows_e, FR, "E1 stopped at 10", 0.06)):
        y = np.array([r["S_median"] for r in rows])
        lo = np.array([r["S_ci95"][0] for r in rows])
        hi = np.array([r["S_ci95"][1] for r in rows])
        x = np.arange(len(rows)) + off
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", color=col, ms=5, capsize=3, lw=1.2, label=lab)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xticks(range(len(eps)))
    ax.set_xticklabels([f"{e:.2f}" for e in eps])
    ax.set_xlabel("tolerance ε (kJ/mol)")
    ax.set_ylabel("sustained speedup S = τ_ABF/τ_FR (median per seed)")
    ax.set_title(f"primary S_geo: X1 {X['primary']['median']:.2f} [{X['primary']['ci95'][0]:.2f}, {X['primary']['ci95'][1]:.2f}]"
                 f";  E1 {E['primary']['median']:.2f} [{E['primary']['ci95'][0]:.2f}, {E['primary']['ci95'][1]:.2f}]",
                 fontsize=8.5, loc="left")
    ax.legend(fontsize=7)
    fig.suptitle("ITE 300 K, K 1024, T 50, 180 bins, FR rate 2.0 from 1 t.u.: early-stopping FR (E1, confirmatory, "
                 "seeds 1100000–015) and continuous FR (X1, exploratory, seeds 980000–015)", x=0.01, ha="left", fontsize=9.5)
    fig.text(0.01, -0.03, "Each block is its own paired two-arm experiment (ABF vs one FR schedule on identical seeds). "
             "Comparing X1 with E1 is a cross-block, descriptive comparison: different seeds, no pairing.",
             fontsize=7, color=INK2)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "convergence_fr", "early_stopping_vs_continuous")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
