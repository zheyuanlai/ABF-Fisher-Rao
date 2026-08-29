#!/usr/bin/env python
"""Figures for Stage 2 of the uniform-FR campaign (WCA Case IX, abf vs fr_uniform).

Under results/uniform_campaign/wca/figures/ (png+pdf):

  fig_wca_convergence   e_F(t) median + IQR, log-y; FR onset marked
  fig_wca_ratio         R_F(t) per-seed spaghetti + median
  fig_wca_mechanism     region occupancy vs t + KL(p||uniform) of the uniform arm
  fig_wca_genealogy     ancestor ESS/N and max lineage share vs t, floors marked
  fig_wca_paired        paired per-seed I_F and final-error slopegraphs
  fig_wca_profiles      F(z), F'(z) snapshots + final marginal vs the corrected
                        TI reference (free energy / mean force / xi-marginal view)

    python scripts/plot_uniform_wca.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from publication_style import PALETTE, apply_publication_style, save_figure  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "results/uniform_campaign/wca/uniform/raw")
OUT = os.path.join(ROOT, "results/uniform_campaign/wca/figures")

C_ABF = PALETTE["blue"]
C_UNI = PALETTE["vermillion"]
T_FR = 40.0
N_REPLICAS = 1024


def load():
    runs = {}
    for path in sorted(glob.glob(os.path.join(RAW, "*.npz"))):
        with np.load(path, allow_pickle=True) as z:
            spec = json.loads(str(z["spec_json"]))
            d = {k: np.asarray(z[k]) for k in
                 ("l2_f_t", "l2_fp_t", "times", "ancestor_ess_t", "max_ancestor_frac_t",
                  "frac_compact", "frac_transition", "frac_stretched", "kl_pq_t",
                  "pmf_t", "mean_force_t", "grid", "reference_free_energy",
                  "reference_mean_force", "final_p_hat", "profile_times")}
            d["int_lf"] = float(z["integrated_l2_f"])
            d["l2_f"] = float(z["l2_f"])
            runs[(spec["method"], spec["seed"])] = d
    seeds = sorted({s for (m, s) in runs if m == "abf" and ("fr_uniform", s) in runs})
    return runs, seeds


def med_iqr(stack):
    a = np.asarray(stack)
    return np.median(a, 0), np.percentile(a, 25, 0), np.percentile(a, 75, 0)


def fig_convergence(runs, seeds):
    t = runs[("abf", seeds[0])]["times"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
    for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "ABF + uniform mFR")):
        md, lo, hi = med_iqr([runs[(m, s)]["l2_f_t"] for s in seeds])
        ax.plot(t, md, color=c, lw=1.4, label=lab)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axvline(T_FR, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.text(T_FR + 2, ax.get_ylim()[1] * 0.6, "FR on", fontsize=7, color=PALETTE["gray"])
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$e_F(t)$  (window RMS)")
    ax.set_title(f"WCA Case IX (corrected reference, {len(seeds)} seeds)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, os.path.join(OUT, "fig_wca_convergence"))
    plt.close(fig)


def fig_ratio(runs, seeds):
    t = runs[("abf", seeds[0])]["times"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
    ratios = []
    for s in seeds:
        r = runs[("fr_uniform", s)]["l2_f_t"] / runs[("abf", s)]["l2_f_t"]
        ratios.append(r)
        ax.plot(t, r, color=C_UNI, alpha=0.18, lw=0.5)
    ax.plot(t, np.median(ratios, 0), color=C_UNI, lw=1.7, label="median ratio")
    ax.axhline(1.0, color=PALETTE["black"], lw=0.8, ls=":")
    ax.axvline(T_FR, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$R_F(t) = e_F^{\rm uni}/e_F^{\rm abf}$")
    ax.set_title("below 1 = uniform mFR ahead", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, os.path.join(OUT, "fig_wca_ratio"))
    plt.close(fig)


def fig_mechanism(runs, seeds):
    t = runs[("abf", seeds[0])]["times"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
    ax = axes[0]
    for m, c in (("abf", C_ABF), ("fr_uniform", C_UNI)):
        for key, ls in (("frac_compact", "-"), ("frac_stretched", "--")):
            md, lo, hi = med_iqr([runs[(m, s)][key] for s in seeds])
            ax.plot(t, md, color=c, lw=1.2, ls=ls)
            ax.fill_between(t, lo, hi, color=c, alpha=0.12, lw=0)
    ax.axvline(T_FR, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_xlabel("t")
    ax.set_ylabel("population fraction")
    ax.set_title("compact (solid) / stretched (dashed)\nblue=ABF, red=uniform mFR",
                 fontsize=8.5)

    ax = axes[1]
    md, lo, hi = med_iqr([runs[("fr_uniform", s)]["kl_pq_t"] for s in seeds])
    ax.plot(t, md, color=C_UNI, lw=1.4, label="uniform arm")
    ax.fill_between(t, lo, hi, color=C_UNI, alpha=0.18, lw=0)
    ax.axvline(T_FR, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$D_{\rm KL}(\hat p_t\,\|\,{\rm uniform})$")
    ax.set_title("marginal establishment (uniform arm's own score KL)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, os.path.join(OUT, "fig_wca_mechanism"))
    plt.close(fig)


def fig_genealogy(runs, seeds):
    t = runs[("abf", seeds[0])]["times"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
    ax = axes[0]
    for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "ABF + uniform mFR")):
        md, lo, hi = med_iqr([runs[(m, s)]["ancestor_ess_t"] / N_REPLICAS for s in seeds])
        ax.plot(t, md, color=c, lw=1.4, label=lab)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axhline(0.10, color=PALETTE["black"], lw=0.8, ls=":", label="declared floor 0.10")
    ax.axvline(T_FR, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_xlabel("t"); ax.set_ylabel("ancestor ESS / N")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for m, c in (("abf", C_ABF), ("fr_uniform", C_UNI)):
        md, lo, hi = med_iqr([runs[(m, s)]["max_ancestor_frac_t"] for s in seeds])
        ax.plot(t, md, color=c, lw=1.4)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axhline(0.05, color=PALETTE["black"], lw=0.8, ls=":", label="declared cap 0.05")
    ax.axvline(T_FR, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_xlabel("t"); ax.set_ylabel("max lineage share")
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle("WCA genealogy", fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_wca_genealogy"))
    plt.close(fig)


def fig_paired(runs, seeds):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), layout="constrained")
    for ax, key, lab in ((axes[0], "int_lf", r"$I_F$ (integrated error)"),
                         (axes[1], "l2_f", r"$e_F(T)$ (final error)")):
        a = np.array([runs[("abf", s)][key] for s in seeds])
        u = np.array([runs[("fr_uniform", s)][key] for s in seeds])
        for i in range(len(a)):
            ax.plot([0, 1], [a[i], u[i]], color=PALETTE["gray"], alpha=0.4, lw=0.7)
        ax.plot([0, 1], [np.median(a), np.median(u)], color=PALETTE["black"], lw=2.0,
                marker="o", ms=4)
        d = 100.0 * (u - a) / a
        ax.set_title(f"median {np.median(d):+.1f}%", fontsize=9)
        ax.set_xticks([0, 1], ["ABF", "uniform\nmFR"])
        ax.set_ylabel(lab)
        ax.set_xlim(-0.3, 1.3)
    fig.suptitle(f"WCA: paired per-seed endpoints ({len(seeds)} seeds)", fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_wca_paired"))
    plt.close(fig)


def fig_profiles(runs, seeds):
    d0 = runs[("abf", seeds[0])]
    z = d0["grid"]
    pt = d0["profile_times"]
    F_ref = d0["reference_free_energy"]
    Fp_ref = d0["reference_mean_force"]
    snaps = [np.argmin(np.abs(pt - v)) for v in (60.0, 120.0, 180.0, pt[-1])]
    win = (z >= -0.1) & (z <= 1.1)

    fig, axes = plt.subplots(3, len(snaps), figsize=(6.9, 6.4), sharex=True,
                             layout="constrained")
    for j, sidx in enumerate(snaps):
        for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "uniform mFR")):
            F = np.median([runs[(m, s)]["pmf_t"][sidx] for s in seeds], 0)
            Fp = np.median([runs[(m, s)]["mean_force_t"][sidx] for s in seeds], 0)
            F = F - (F - F_ref)[win].mean()
            axes[0, j].plot(z, F, color=c, lw=1.2, label=lab)
            axes[1, j].plot(z, Fp, color=c, lw=1.2)
        axes[0, j].plot(z, F_ref, color=PALETTE["black"], lw=0.9, ls=":", label="TI reference")
        axes[1, j].plot(z, Fp_ref, color=PALETTE["black"], lw=0.9, ls=":")
        axes[0, j].set_title(f"t = {pt[sidx]:g}", fontsize=9)
        for m, c in (("abf", C_ABF), ("fr_uniform", C_UNI)):
            p = np.median([runs[(m, s)]["final_p_hat"] for s in seeds], 0)
            axes[2, j].plot(z, p, color=c, lw=1.2)
        axes[2, j].axhline(1.0 / (z[-1] - z[0]), color=PALETTE["black"], lw=0.9, ls=":")
        axes[2, j].set_xlabel("z")
    axes[0, 0].set_ylabel(r"$\hat F_t(z)$")
    axes[1, 0].set_ylabel(r"$\hat F'_t(z)$")
    axes[2, 0].set_ylabel(r"final marginal $\hat p_T(z)$")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("WCA: F / mean force snapshots vs corrected TI reference\n"
                 f"(median over {len(seeds)} seeds; bottom row: FINAL marginal only -- "
                 "per-checkpoint marginals are not stored)", fontsize=9)
    save_figure(fig, os.path.join(OUT, "fig_wca_profiles"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    apply_publication_style()
    runs, seeds = load()
    print(f"{len(seeds)} complete pairs")
    fig_convergence(runs, seeds)
    fig_ratio(runs, seeds)
    fig_mechanism(runs, seeds)
    fig_genealogy(runs, seeds)
    fig_paired(runs, seeds)
    fig_profiles(runs, seeds)
    print(f"wrote figures -> {OUT}")


if __name__ == "__main__":
    main()
