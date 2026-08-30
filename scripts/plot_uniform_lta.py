#!/usr/bin/env python
"""Figures for Stage 4 of the uniform-FR campaign (ethane/LTA).

Under results/uniform_campaign/lta/figures/ (png+pdf):

  fig_lta_convergence   e_F(t) vs the umbrella/WHAM reference, both arms
  fig_lta_ratio         R_F(t) per-seed spaghetti + median
  fig_lta_mechanism     KL(p||uniform) + cage/window occupancy vs t
  fig_lta_genealogy     ancestor ESS/N, max lineage share, cumulative events
  fig_lta_paired        paired per-seed I_F and final error
  fig_lta_profiles      F(z) / mean force / marginal snapshots vs reference

    python scripts/plot_uniform_lta.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from publication_style import PALETTE, apply_publication_style, save_figure  # noqa: E402
from analyze_uniform_lta import circular_interp_ref, error_series  # noqa: E402

ROOT = os.path.join(SCRIPTS, "..")
PROD = os.path.join(ROOT, "results/uniform_campaign/lta/production")
REF = os.path.join(ROOT, "results/uniform_campaign/lta/reference/reference_T300.npz")
OUT = os.path.join(ROOT, "results/uniform_campaign/lta/figures")

PI = math.pi
C_ABF = PALETTE["blue"]
C_UNI = PALETTE["vermillion"]


def med_iqr(a, axis):
    return (np.median(a, axis), np.percentile(a, 25, axis), np.percentile(a, 75, axis))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=None,
                    help="sweep mode: read production_T{T}/ + reference_T{T}.npz, suffix figures")
    a_cli = ap.parse_args()
    global OUT
    if a_cli.temperature is not None:
        tkey = f"{a_cli.temperature:g}"
        prod = os.path.join(ROOT, f"results/uniform_campaign/lta/production_T{tkey}")
        ref_path = os.path.join(ROOT,
                                f"results/uniform_campaign/lta/reference/reference_T{tkey}.npz")
        suffix = f"_T{tkey}"
    else:
        prod, ref_path, suffix = PROD, REF, ""
    os.makedirs(OUT, exist_ok=True)
    apply_publication_style()
    ref = np.load(ref_path, allow_pickle=True)
    runs = {m: np.load(os.path.join(prod, f"{m}.npz"), allow_pickle=True)
            for m in ("abf", "fr_uniform")}
    grid = runs["abf"]["grid"]
    a = float(runs["abf"]["a_pseudo"])
    z = grid * a / (2 * PI)
    t = np.asarray(runs["abf"]["times"], dtype=float)
    F_ref = circular_interp_ref(ref["F"], ref["grid_phi"], grid)
    Fp_ref = np.gradient(F_ref, grid)          # dF/dphi for the mean-force panel
    err = {m: error_series(np.asarray(runs[m]["pmf"], dtype=float), F_ref) for m in runs}
    meta = json.loads(str(runs["abf"]["meta"]))
    fr_start_steps = 20000 if "sweep" in str(meta.get("prereg", "")) else 40000
    fr_start_t = t[np.searchsorted(np.asarray(runs["abf"]["steps"]), fr_start_steps)]

    # ---- convergence ----
    fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
    for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "ABF + uniform mFR")):
        md, lo, hi = med_iqr(err[m], 1)
        ax.plot(t, md, color=c, lw=1.4, label=lab)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axvline(fr_start_t, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("t (BD units)")
    ax.set_ylabel(r"$e_F(t)$ (kJ/mol, full-circle RMS)")
    ttl = f"Ethane/LTA {float(ref['temperature']):g} K (16 paired seed labels, N=1024)"
    ax.set_title(ttl, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, os.path.join(OUT, "fig_lta_convergence" + suffix))
    plt.close(fig)

    # ---- ratio ----
    fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
    ratios = err["fr_uniform"] / err["abf"]
    for r in range(ratios.shape[1]):
        ax.plot(t, ratios[:, r], color=C_UNI, alpha=0.18, lw=0.5)
    ax.plot(t, np.median(ratios, 1), color=C_UNI, lw=1.7, label="median ratio")
    ax.axhline(1.0, color=PALETTE["black"], lw=0.8, ls=":")
    ax.axvline(fr_start_t, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_xlabel("t (BD units)")
    ax.set_ylabel(r"$R_F(t) = e_F^{\rm uni}/e_F^{\rm abf}$")
    ax.set_title("below 1 = uniform mFR ahead", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, os.path.join(OUT, "fig_lta_ratio" + suffix))
    plt.close(fig)

    # ---- mechanism ----
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
    ax = axes[0]
    for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "ABF + uniform mFR")):
        kl = np.asarray(runs[m]["kl_uniform"], dtype=float)
        md, lo, hi = med_iqr(kl, 1)
        ax.plot(t, md, color=c, lw=1.4, label=lab)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axvline(fr_start_t, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_xlabel("t"); ax.set_ylabel(r"$D_{\rm KL}(\hat p_t\|\,{\rm uniform})$")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("marginal establishment", fontsize=9)
    ax = axes[1]
    for m, c in (("abf", C_ABF), ("fr_uniform", C_UNI)):
        for key, ls in (("frac_cage", "-"), ("frac_window", "--")):
            md, lo, hi = med_iqr(np.asarray(runs[m][key], dtype=float), 1)
            ax.plot(t, md, color=c, lw=1.2, ls=ls)
            ax.fill_between(t, lo, hi, color=c, alpha=0.12, lw=0)
    ax.axvline(fr_start_t, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_xlabel("t"); ax.set_ylabel("population fraction")
    ax.set_yscale("log")
    ax.set_title("cage (solid) / window (dashed)", fontsize=9)
    save_figure(fig, os.path.join(OUT, "fig_lta_mechanism" + suffix))
    plt.close(fig)

    # ---- genealogy ----
    N = 1024
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9), layout="constrained")
    ess = np.asarray(runs["fr_uniform"]["ancestor_ess"], dtype=float) / N
    md, lo, hi = med_iqr(ess, 1)
    axes[0].plot(t, md, color=C_UNI, lw=1.4)
    axes[0].fill_between(t, lo, hi, color=C_UNI, alpha=0.18, lw=0)
    axes[0].axhline(0.30, color=PALETTE["black"], lw=0.8, ls=":", label="floor 0.30")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("ancestor ESS / N (uniform arm)")
    axes[0].legend(frameon=False, fontsize=7)
    wmax = np.asarray(runs["fr_uniform"]["max_ancestor_frac"], dtype=float)
    md, lo, hi = med_iqr(wmax, 1)
    axes[1].plot(t, md, color=C_UNI, lw=1.4)
    axes[1].fill_between(t, lo, hi, color=C_UNI, alpha=0.18, lw=0)
    axes[1].axhline(0.05, color=PALETTE["black"], lw=0.8, ls=":", label="cap 0.05")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("max lineage share")
    axes[1].legend(frameon=False, fontsize=7)
    ev = np.asarray(runs["fr_uniform"]["repl_cumulative"], dtype=float) / N
    md, lo, hi = med_iqr(ev, 1)
    axes[2].plot(t, md, color=C_UNI, lw=1.4)
    axes[2].fill_between(t, lo, hi, color=C_UNI, alpha=0.18, lw=0)
    axes[2].set_xlabel("t"); axes[2].set_ylabel("cumulative events / N")
    fig.suptitle("Ethane/LTA genealogy (uniform arm)", fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_lta_genealogy" + suffix))
    plt.close(fig)

    # ---- paired ----
    I = {m: np.trapezoid(err[m], t, axis=0) for m in err}
    fin = {m: err[m][-1] for m in err}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), layout="constrained")
    for ax, vals, lab in ((axes[0], I, r"$I_F$ (integrated error)"),
                          (axes[1], fin, r"$e_F(T)$ (final error)")):
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
    fig.suptitle("Ethane/LTA: paired per-seed endpoints (16 labels)", fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_lta_paired" + suffix))
    plt.close(fig)

    # ---- profiles ----
    snaps_t = [0.2, 0.4, 0.6, 1.0]
    snaps = [int(round(f * (len(t) - 1))) for f in snaps_t]
    fig, axes = plt.subplots(3, len(snaps), figsize=(6.9, 6.4), sharex=True,
                             layout="constrained")
    for j, sidx in enumerate(snaps):
        for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "uniform mFR")):
            F = np.median(np.asarray(runs[m]["pmf"], dtype=float)[sidx], 0)
            F = F - (F - F_ref).mean()
            mf = np.median(np.asarray(runs[m]["mean_force"], dtype=float)[sidx], 0)
            p = np.median(np.asarray(runs[m]["p_hat"], dtype=float)[sidx], 0)
            axes[0, j].plot(z, F, color=c, lw=1.2, label=lab)
            axes[1, j].plot(z, mf, color=c, lw=1.2)
            axes[2, j].plot(z, p, color=c, lw=1.2)
        axes[0, j].plot(z, F_ref, color=PALETTE["black"], lw=0.9, ls=":",
                        label="umbrella/WHAM ref")
        axes[1, j].plot(z, Fp_ref, color=PALETTE["black"], lw=0.9, ls=":")
        axes[2, j].axhline(1.0 / (2 * PI), color=PALETTE["black"], lw=0.9, ls=":")
        axes[0, j].set_title(f"t = {t[sidx]:g}", fontsize=9)
        axes[2, j].set_xlabel("z (A)")
    axes[0, 0].set_ylabel(r"$\hat F_t(z)$ (kJ/mol)")
    axes[1, 0].set_ylabel(r"$d\hat F_t/d\varphi$")
    axes[2, 0].set_ylabel(r"marginal $\hat p_t(\varphi)$")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Ethane/LTA: F / mean force / xi-marginal convergence\n"
                 "(median over 16 seed labels; window at z=0, cages at $\\pm$a/2)",
                 fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_lta_profiles" + suffix))
    plt.close(fig)
    print(f"wrote figures -> {OUT}")


if __name__ == "__main__":
    main()
