"""Figure for one analysed two-arm block (ABF vs ABF + FR), from results/convergence_fr_campaign/analysis/<block>.json
and <block>_curves.npz.

    python scripts/convergence_fr/fig_block.py E1 --title "ITE 300 K · K 1024 · early-stopping FR (t_stop 10 t.u.)"

Panels: (a) median e_F(t) with IQR and 95 % CI of the median; (b) per-seed e_F(FR)/e_F(ABF) - 1, median and CI;
(c) sustained time to accuracy per seed on the frozen ladder; (d) circulation c(t) (read-out Gamma mean).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "eri_campaign"))
import atlas_lib as L  # noqa: E402
from figstyle import ABF, FR, INK2, MUTED, setup  # noqa: E402

plt = setup()
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


def bands(ax, t, X, col, lab):
    med, lo, hi = L.boot_median_curve(X)
    ax.fill_between(t, np.nanquantile(X, 0.25, 1), np.nanquantile(X, 0.75, 1), color=col, alpha=0.13, lw=0)
    ax.fill_between(t, lo, hi, color=col, alpha=0.35, lw=0)
    ax.plot(t, med, color=col, lw=1.6, label=lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("block")
    ap.add_argument("--title", default="")
    ap.add_argument("--frlabel", default="ABF + FR")
    a = ap.parse_args()
    base = os.path.join(ROOT, "results", "convergence_fr_campaign", "analysis", a.block)
    R = json.load(open(base + ".json"))
    z = np.load(base + "_curves.npz")
    t, T = z["t"], R["T"]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.7))
    ax = axs[0]
    bands(ax, t, z["eF_abf"], ABF, "ABF")
    bands(ax, t, z["eF_fr"], FR, a.frlabel)
    ax.set_yscale("log")
    ax.set_ylabel("e_F(t) (kJ/mol, RMS vs exact MC)")
    ax.legend(loc="upper right")
    ax = axs[1]
    rel = 100 * (z["eF_fr"] / z["eF_abf"] - 1)
    med, lo, hi = L.boot_median_curve(rel)
    ax.fill_between(t, lo, hi, color=FR, alpha=0.25, lw=0)
    ax.plot(t, med, color=FR, lw=1.5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("per-seed e_F(FR)/e_F(ABF) − 1 (%)")
    sel = t >= R["t_fr"]
    ax.set_ylim(np.nanmin(lo[sel]) - 5, max(15, np.nanmax(hi[sel]) + 5))
    for ax in axs[:2]:
        ax.axvline(R["t_fr"], color=MUTED, ls=":", lw=1)
        if R.get("t_stop"):
            ax.axvline(R["t_stop"], color=FR, ls="--", lw=1)
        ax.set_xlabel("t (t.u.)")
    ax = axs[2]
    rows = R["time_to_accuracy"]["sustained"]
    for i, st in enumerate(rows):
        for off, arm, col in ((-0.17, "abf", ABF), (0.17, "fr", FR)):
            tau = np.array(st[f"tau_{arm}"], float)
            fin = np.isfinite(tau)
            x = i + off + np.linspace(-0.07, 0.07, len(tau))
            ax.scatter(x[fin], tau[fin], s=10, color=col, lw=0)
            ax.scatter(x[~fin], np.full((~fin).sum(), T * 1.04), s=12, facecolor="white", edgecolor=col, lw=0.8)
            if fin.sum() * 2 > len(tau):
                ax.hlines(np.median(np.minimum(tau, T)), i + off - 0.12, i + off + 0.12, color=col, lw=2.2)
        ax.text(i, T * 1.12, f"S {st['S_median']:.2f}", ha="center", fontsize=7, color=INK2)
    ax.axhline(T, color=MUTED, lw=0.8)
    ax.set_ylim(0, T * 1.18)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{st['eps_kJmol']:.3g}" for st in rows])
    ax.set_xlabel("tolerance ε (kJ/mol)")
    ax.set_ylabel("sustained τ_ε (t.u.); open = censored")
    p = R["primary"]
    ax.set_title(f"primary S_geo {p['median']:.2f} [{p['ci95'][0]:.2f}, {p['ci95'][1]:.2f}]", fontsize=9, loc="left")
    ax = axs[3]
    for key, col, lab in (("c_abf", ABF, "ABF"), ("c_fr", FR, a.frlabel)):
        X = z[key]
        ax.fill_between(t, np.quantile(X, 0.25, 1), np.quantile(X, 0.75, 1), color=col, alpha=0.15, lw=0)
        ax.plot(t, np.median(X, 1), color=col, lw=1.4, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(R["t_fr"], color=MUTED, ls=":", lw=1)
    if R.get("t_stop"):
        ax.axvline(R["t_stop"], color=FR, ls="--", lw=1)
    sel = t >= 2
    ax.set_ylim(np.quantile(z["c_abf"][sel], 0.02) - 0.01, np.quantile(z["c_fr"][sel], 0.98) + 0.01)
    ax.set_xlabel("t (t.u.)")
    ax.set_ylabel("circulation c(t) = ⟨Γ_t⟩_φ (kJ/mol/rad)")
    fig.suptitle(a.title or a.block, x=0.01, ha="left", fontsize=10.5)
    fig.text(0.01, -0.03, f"{R['n_seeds']} paired seeds; K {R['K']}, T {T:g} t.u., {R['bins']} bins, FR rate "
             f"{R['rate']:g} from {R['t_fr']:g} t.u. (dotted)" + (f", stopped at {R['t_stop']:g} t.u. (dashed)" if R.get("t_stop") else "")
             + ". Bands: light 25–75 % of seeds, dark 95 % bootstrap CI of the median. S: median per-seed restricted "
             "sustained speedup τ_ABF/τ_FR.", fontsize=7, color=INK2)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "convergence_fr", f"block_{a.block}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
