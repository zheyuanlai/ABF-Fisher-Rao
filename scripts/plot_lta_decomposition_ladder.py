#!/usr/bin/env python
"""Ethane/LTA: the free-energy decomposition F(z) = U(z) - T S(z) across temperature.

Reads every available umbrella/WHAM reference (results/uniform_campaign/lta/reference/reference_T*.npz)
and draws (a) F, (b) U, (c) -TS vs z in kT with one line per temperature (one hue, light -> dark with T),
and (d) the barrier heights dF, dU, -TdS (kT) and the entropic share vs T.  Static companion of the movies
and of the sweep figures; no simulation.

    python scripts/plot_lta_decomposition_ladder.py [--out results/lta_histogram/figures]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from publication_style import PALETTE, apply_publication_style, save_figure  # noqa: E402

REF_DIR = os.path.join(ROOT, "results/uniform_campaign/lta/reference")
PI = math.pi


def load_refs():
    refs = {}
    for p in glob.glob(os.path.join(REF_DIR, "reference_T*.npz")):
        T = float(re.search(r"reference_T([\d.]+)\.npz", p).group(1))
        refs[T] = np.load(p, allow_pickle=True)
    return dict(sorted(refs.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results/lta_histogram/figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    refs = load_refs()
    assert refs, "no references found"
    Ts = list(refs)
    apply_publication_style()
    # sequential hue (blue family), light -> dark with T
    cmap = plt.get_cmap("Blues")
    cols = {T: cmap(0.35 + 0.6 * i / max(len(Ts) - 1, 1)) for i, T in enumerate(Ts)}
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.45), layout="constrained")
    titles = [r"$\beta F(z)$", r"$\beta U(z)$", r"$-S(z)/k_B$  ($= \beta F - \beta U$)"]
    for T, r in refs.items():
        z = np.asarray(r["z"], float); beta = 1.0 / float(r["kT"])
        F = np.asarray(r["F"], float); U = np.asarray(r["U"], float)
        F = F - F[np.abs(np.abs(z) - z.max()) < 0.6].mean()      # gauge: cage centre = 0
        U = U - U[np.abs(np.abs(z) - z.max()) < 0.6].mean()
        mTS = F - U
        for ax, y in zip(axes[:3], (beta * F, beta * U, beta * mTS)):
            ax.plot(z, y, color=cols[T], lw=1.4, label=f"{T:g} K")
    for ax, t in zip(axes[:3], titles):
        ax.set_title(t, fontsize=8.5)
        ax.set_xlabel("z (Å)")
        ax.axvline(0, color=PALETTE["light_gray"], lw=0.8, zorder=0)
    axes[0].set_ylabel("$k_BT$ units (cage centre = 0)")
    axes[1].text(0.5, -0.33, "window at z = 0, cage centres at z = ±a/2 = ±5.96 Å", transform=axes[1].transAxes, ha="center", va="top", fontsize=7, color=PALETTE["gray"])
    axes[2].legend(frameon=False, fontsize=6.5, ncol=1, loc="upper right")
    ax = axes[3]
    dF = [float(refs[T]["dF_barrier"]) / float(refs[T]["kT"]) for T in Ts]
    dU = [float(refs[T]["dU_barrier"]) / float(refs[T]["kT"]) for T in Ts]
    dS = [float(refs[T]["mTdS_barrier"]) / float(refs[T]["kT"]) for T in Ts]
    ax.plot(Ts, dF, "o-", color=PALETTE["black"], lw=1.4, ms=3.5, label=r"$\Delta F^\ddagger$")
    ax.plot(Ts, dU, "s--", color=PALETTE["blue"], lw=1.2, ms=3.2, label=r"$\Delta U^\ddagger$")
    ax.plot(Ts, dS, "^-.", color=PALETTE["vermillion"], lw=1.2, ms=3.2, label=r"$-T\Delta S^\ddagger$")
    for T, f, s in zip(Ts, dF, dS):
        ax.annotate(f"{100 * s / f:.0f}%", (T, f), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=6.5, color=PALETTE["gray"])
    ax.set_xlabel("T (K)"); ax.set_ylabel("barrier ($k_BT$)")
    ax.set_title("barrier heights", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_ylim(0, max(dF) * 1.18)
    save_figure(fig, os.path.join(a.out, "fig_lta_decomposition_vs_T"))
    print("wrote fig_lta_decomposition_vs_T for T =", Ts)
    for T in Ts:
        r = refs[T]; kT = float(r["kT"])
        print(f"  {T:5.0f} K: dF {float(r['dF_barrier'])/kT:5.2f} kT, dU {float(r['dU_barrier'])/kT:5.2f}, "
              f"-TdS {float(r['mTdS_barrier'])/kT:5.2f} ({100*float(r['mTdS_barrier'])/float(r['dF_barrier']):.0f} % entropic)")


if __name__ == "__main__":
    main()
