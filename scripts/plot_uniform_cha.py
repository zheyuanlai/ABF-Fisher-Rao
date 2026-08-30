#!/usr/bin/env python
"""Figures for one olefin/CHA cell (abf vs fr_uniform), campaign style, plus the
three-panel aligned mechanism figure (KL -> e_F' -> e_F, FR start marked).

    python scripts/plot_uniform_cha.py --guest ethene --temperature 450
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from publication_style import PALETTE, apply_publication_style, save_figure  # noqa: E402
from analyze_uniform_cha import error_series  # noqa: E402

ROOT = os.path.join(SCRIPTS, "..")
CHA = os.path.join(ROOT, "results/uniform_campaign/cha")
OUT = os.path.join(CHA, "figures")
C_ABF = PALETTE["blue"]
C_UNI = PALETTE["vermillion"]


def med_iqr(a, axis):
    return (np.median(a, axis), np.percentile(a, 25, axis), np.percentile(a, 75, axis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guest", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    a = ap.parse_args()
    tag = f"{a.guest}_{a.temperature:g}"
    os.makedirs(OUT, exist_ok=True)
    apply_publication_style()
    ref = np.load(os.path.join(CHA, "reference", f"reference_{tag}.npz"),
                  allow_pickle=True)
    runs = {m: np.load(os.path.join(CHA, f"production_{tag}", f"{m}.npz"),
                       allow_pickle=True) for m in ("abf", "fr_uniform")}
    grid = runs["abf"]["grid"]
    xi_A, xi_B = float(ref["xi_A"]), float(ref["xi_B"])
    mask = (grid >= xi_A - 1.0) & (grid <= xi_B + 1.0)
    F_ref = np.asarray(ref["F"], dtype=float)
    dF_ref = np.gradient(F_ref, grid)
    t = np.asarray(runs["abf"]["times"], dtype=float)
    fr_start = 25_000 * 2e-4
    err = {m: error_series(np.asarray(runs[m]["pmf"], dtype=float), F_ref, mask)
           for m in runs}
    errp = {m: error_series(np.asarray(runs[m]["mean_force"], dtype=float),
                            dF_ref, mask) for m in runs}

    # ---- convergence + ratio ----
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
    ax = axes[0]
    for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "ABF + uniform mFR")):
        md, lo, hi = med_iqr(err[m], 1)
        ax.plot(t, md, color=c, lw=1.4, label=lab)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axvline(fr_start, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("t (BD units)")
    ax.set_ylabel(r"$e_F(t)$ (kJ/mol)")
    ax.set_title(f"{a.guest} {a.temperature:g} K", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ratios = err["fr_uniform"] / err["abf"]
    for r in range(ratios.shape[1]):
        ax.plot(t, ratios[:, r], color=C_UNI, alpha=0.18, lw=0.5)
    ax.plot(t, np.median(ratios, 1), color=C_UNI, lw=1.7)
    ax.axhline(1.0, color=PALETTE["black"], lw=0.8, ls=":")
    ax.axvline(fr_start, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_xlabel("t (BD units)")
    ax.set_ylabel(r"$R_F(t)$")
    ax.set_title("below 1 = uniform mFR ahead", fontsize=9.5)
    save_figure(fig, os.path.join(OUT, f"fig_cha_convergence_{tag}"))
    plt.close(fig)

    # ---- THE mechanism figure: KL -> e_F' -> e_F aligned ----
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 6.2), sharex=True,
                             layout="constrained")
    panels = (("kl_uniform", r"$D_{\rm KL}(\hat p_t\|\,u)$", None),
              (None, r"$e_{F'}(t)$", errp),
              (None, r"$e_F(t)$", err))
    for ax, (key, lab, series) in zip(axes, panels):
        for m, c, lname in (("abf", C_ABF, "ABF"),
                            ("fr_uniform", C_UNI, "ABF + uniform mFR")):
            y = (np.asarray(runs[m][key], dtype=float) if key else series[m])
            md, lo, hi = med_iqr(y, 1)
            ax.plot(t, md, color=c, lw=1.4, label=lname)
            ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
        ax.axvline(fr_start, color=PALETTE["gray"], lw=0.8, ls="--")
        ax.set_ylabel(lab)
        ax.set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title(f"{a.guest} {a.temperature:g} K: marginal -> mean force -> F",
                      fontsize=9.5)
    axes[2].set_xlabel("t (BD units)  (dashed line: FR start)")
    save_figure(fig, os.path.join(OUT, f"fig_cha_mechanism_{tag}"))
    plt.close(fig)

    # ---- profiles: F / mean force / marginal snapshots ----
    snaps = [int(round(f * (len(t) - 1))) for f in (0.15, 0.3, 0.6, 1.0)]
    fig, axes = plt.subplots(3, len(snaps), figsize=(6.9, 6.4), sharex=True,
                             layout="constrained")
    for j, sidx in enumerate(snaps):
        for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "uniform mFR")):
            F = np.median(np.asarray(runs[m]["pmf"], dtype=float)[sidx], 0)
            F = F - (F - F_ref)[mask].mean()
            mf = np.median(np.asarray(runs[m]["mean_force"], dtype=float)[sidx], 0)
            p = np.median(np.asarray(runs[m]["p_hat"], dtype=float)[sidx], 0)
            axes[0, j].plot(grid, F, color=c, lw=1.2, label=lab)
            axes[1, j].plot(grid, mf, color=c, lw=1.2)
            axes[2, j].plot(grid, p, color=c, lw=1.2)
        axes[0, j].plot(grid, F_ref, color=PALETTE["black"], lw=0.9, ls=":",
                        label="umbrella/WHAM ref")
        axes[1, j].plot(grid, dF_ref, color=PALETTE["black"], lw=0.9, ls=":")
        axes[0, j].set_title(f"t = {t[sidx]:g}", fontsize=9)
        axes[2, j].set_xlabel(r"$\xi$ (A)")
        for ax in (axes[0, j], axes[1, j], axes[2, j]):
            ax.set_xlim(xi_A - 2, xi_B + 2)
    axes[0, 0].set_ylabel(r"$\hat F_t(\xi)$ (kJ/mol)")
    axes[1, 0].set_ylabel(r"$d\hat F_t/d\xi$")
    axes[2, 0].set_ylabel(r"marginal $\hat p_t(\xi)$")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{a.guest} {a.temperature:g} K: F / mean force / xi-marginal "
                 "(median over 16 labels; window at 0)", fontsize=9.5)
    save_figure(fig, os.path.join(OUT, f"fig_cha_profiles_{tag}"))
    plt.close(fig)

    # ---- genealogy + paired ----
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9), layout="constrained")
    N = 1024
    ess = np.asarray(runs["fr_uniform"]["ancestor_ess"], dtype=float) / N
    md, lo, hi = med_iqr(ess, 1)
    axes[0].plot(t, md, color=C_UNI, lw=1.4)
    axes[0].fill_between(t, lo, hi, color=C_UNI, alpha=0.18, lw=0)
    axes[0].axhline(0.30, color=PALETTE["black"], lw=0.8, ls=":", label="floor 0.30")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("ancestor ESS / N")
    axes[0].legend(frameon=False, fontsize=7)
    for ax, vals, lab in ((axes[1], {m: np.trapezoid(err[m], t, axis=0) for m in err},
                           r"$I_F$"),
                          (axes[2], {m: err[m][-1] for m in err}, r"$e_F(T)$")):
        aa, uu = vals["abf"], vals["fr_uniform"]
        for i in range(len(aa)):
            ax.plot([0, 1], [aa[i], uu[i]], color=PALETTE["gray"], alpha=0.4, lw=0.7)
        ax.plot([0, 1], [np.median(aa), np.median(uu)], color=PALETTE["black"],
                lw=2.0, marker="o", ms=4)
        d = 100.0 * (uu - aa) / aa
        ax.set_title(f"median {np.median(d):+.1f}%", fontsize=9)
        ax.set_xticks([0, 1], ["ABF", "uniform\nmFR"])
        ax.set_ylabel(lab)
        ax.set_xlim(-0.3, 1.3)
    fig.suptitle(f"{a.guest} {a.temperature:g} K: genealogy and paired endpoints",
                 fontsize=9.5)
    save_figure(fig, os.path.join(OUT, f"fig_cha_paired_{tag}"))
    plt.close(fig)
    print(f"wrote figures -> {OUT}")


if __name__ == "__main__":
    main()
