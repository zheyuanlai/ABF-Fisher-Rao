#!/usr/bin/env python
"""Figures for Stage 1 of the uniform-FR campaign (gateway, abf vs fr_uniform).

Produces, under results/uniform_campaign/gateway/figures/ (png+pdf each):

  fig_gw_convergence   e_F(t) median + IQR band, one panel per init
  fig_gw_ratio         R_F(t) = e_F^uni / e_F^abf, per-seed spaghetti + median
  fig_gw_mechanism     KL(p_t || uniform) and gate/right-basin occupancy vs t
  fig_gw_genealogy     ancestor ESS/N and max lineage share vs t, floors marked
  fig_gw_paired        paired per-seed I_F, final error, frozen-bias slopegraphs
  fig_gw_profiles      F(x), F'(x) and marginal p(x) snapshots vs time (the
                       free-energy / mean-force / xi-marginal convergence view)

    python scripts/plot_uniform_gateway.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from publication_style import PALETTE, apply_publication_style, save_figure  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "results/uniform_campaign/gateway/raw.npz")
OUT = os.path.join(ROOT, "results/uniform_campaign/gateway/figures")

C_ABF = PALETTE["blue"]
C_UNI = PALETTE["vermillion"]
INITS = ("left", "one_right")
INIT_LABEL = {"left": "init: left basin", "one_right": "init: one walker right"}


def load():
    z = np.load(RAW, allow_pickle=True)
    method = np.array([str(m) for m in z["method"]])
    init = np.array([str(i) for i in z["init"]])
    seed = z["seed"].astype(int)
    rows = {}
    for i in range(len(method)):
        rows.setdefault((init[i], seed[i]), {})[method[i]] = i
    return z, rows


def med_iqr(x):
    return (np.median(x, axis=0), np.percentile(x, 25, axis=0),
            np.percentile(x, 75, axis=0))


def idxs(rows, ini, m):
    return [rows[k][m] for k in sorted(rows) if k[0] == ini]


def fig_convergence(z, rows):
    t = z["t"][0]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=True, layout="constrained")
    for ax, ini in zip(axes, INITS):
        for m, c, lab in ((("abf"), C_ABF, "ABF"),
                          (("fr_uniform"), C_UNI, "ABF + uniform mFR")):
            e = z["l2_f_t"][idxs(rows, ini, m)]
            md, lo, hi = med_iqr(e)
            ax.plot(t, md, color=c, lw=1.4, label=lab)
            ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
        ax.set_xlabel("t (simulation time)")
        ax.set_yscale("log")
        ax.set_title(INIT_LABEL[ini], fontsize=9)
    axes[0].set_ylabel(r"$e_F(t)$  (window RMS)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Gateway: free-energy error vs time (median, IQR band; 32 seeds)",
                 fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_gw_convergence"))
    plt.close(fig)


def fig_ratio(z, rows):
    t = z["t"][0]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=True, layout="constrained")
    for ax, ini in zip(axes, INITS):
        keys = [k for k in sorted(rows) if k[0] == ini]
        ratios = []
        for k in keys:
            r = z["l2_f_t"][rows[k]["fr_uniform"]] / z["l2_f_t"][rows[k]["abf"]]
            ratios.append(r)
            ax.plot(t, r, color=C_UNI, alpha=0.12, lw=0.5)
        ax.plot(t, np.median(ratios, axis=0), color=C_UNI, lw=1.6, label="median ratio")
        ax.axhline(1.0, color=PALETTE["black"], lw=0.8, ls=":")
        ax.set_xlabel("t")
        ax.set_title(INIT_LABEL[ini], fontsize=9)
        ax.set_ylim(0, 2.0)
    axes[0].set_ylabel(r"$R_F(t) = e_F^{\rm uni}/e_F^{\rm abf}$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Gateway: error ratio -- below 1 = uniform mFR ahead", fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_gw_ratio"))
    plt.close(fig)


def fig_mechanism(z, rows):
    t = z["t"][0]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
    ax = axes[0]
    for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "ABF + uniform mFR")):
        kl = z["kl_uniform_t"][idxs(rows, "left", m)]
        md, lo, hi = med_iqr(kl)
        ax.plot(t, md, color=c, lw=1.4, label=lab)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$D_{\rm KL}(\hat p_t \,\|\, {\rm uniform})$")
    ax.set_title("marginal establishment (init: left)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for m, c in (("abf", C_ABF), ("fr_uniform", C_UNI)):
        P = z["P_regions"][idxs(rows, "left", m)]           # (n, T, 3)
        md, lo, hi = med_iqr(P[:, :, 2])
        ax.plot(t, md, color=c, lw=1.4)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
        md_g, _, _ = med_iqr(P[:, :, 1])
        ax.plot(t, md_g, color=c, lw=1.0, ls="--")
    Q = z["Q_regions"][idxs(rows, "left", "abf")]
    ax.plot(t, np.median(Q[:, :, 2], axis=0), color=PALETTE["gray"], lw=1.0, ls=":",
            label=r"bias-aware target $Q^*_{\rm right}$")
    ax.set_xlabel("t")
    ax.set_ylabel("population fraction")
    ax.set_title("right basin (solid) / gate (dashed)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    save_figure(fig, os.path.join(OUT, "fig_gw_mechanism"))
    plt.close(fig)


def fig_genealogy(z, rows):
    t = z["t"][0]
    N = json.loads(str(z["config_json"][0]))["N"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
    ax = axes[0]
    for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "ABF + uniform mFR")):
        e = z["ess_t"][idxs(rows, "left", m)] / N
        md, lo, hi = med_iqr(e)
        ax.plot(t, md, color=c, lw=1.4, label=lab)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axhline(0.30, color=PALETTE["black"], lw=0.8, ls=":", label="declared floor 0.30")
    ax.set_xlabel("t"); ax.set_ylabel("ancestor ESS / N")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for m, c in (("abf", C_ABF), ("fr_uniform", C_UNI)):
        w = z["wmax_t"][idxs(rows, "left", m)]
        md, lo, hi = med_iqr(w)
        ax.plot(t, md, color=c, lw=1.4)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axhline(0.05, color=PALETTE["black"], lw=0.8, ls=":", label="declared cap 0.05")
    ax.set_xlabel("t"); ax.set_ylabel("max lineage share")
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Gateway genealogy (init: left; 4000-step ancestry window)", fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_gw_genealogy"))
    plt.close(fig)


def fig_paired(z, rows):
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.0), layout="constrained")
    panels = (("int_l2_f", r"$I_F$ (integrated error)", "int"),
              ("final_l2_f", r"$e_F(T)$ (final error)", "final"),
              ("frozen_l2_f_kT", "frozen-bias error (kT)", "frozen"))
    for ax, (key, lab, _) in zip(axes, panels):
        a = np.array([z[key][rows[k]["abf"]] for k in sorted(rows)], dtype=float)
        u = np.array([z[key][rows[k]["fr_uniform"]] for k in sorted(rows)], dtype=float)
        for i in range(len(a)):
            ax.plot([0, 1], [a[i], u[i]], color=PALETTE["gray"], alpha=0.35, lw=0.6)
        ax.plot([0, 1], [np.median(a), np.median(u)], color=PALETTE["black"], lw=2.0,
                marker="o", ms=4)
        d = 100.0 * (u - a) / a
        ax.set_title(f"median {np.median(d):+.1f}%", fontsize=9)
        ax.set_xticks([0, 1], ["ABF", "uniform\nmFR"])
        ax.set_ylabel(lab)
        ax.set_xlim(-0.3, 1.3)
    fig.suptitle("Gateway: paired per-seed endpoints (64 pairs, both inits)", fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_gw_paired"))
    plt.close(fig)


def fig_profiles(z, rows):
    t = z["t"][0]
    x = z["x_grid"][0]
    snaps = [np.argmin(np.abs(t - v)) for v in (2.0, 5.0, 10.0, 40.0)]
    keys = [k for k in sorted(rows) if k[0] == "left"]
    F_ref = z["F_ref"][rows[keys[0]]["abf"]]
    Fp_ref = z["Fp_ref"][rows[keys[0]]["abf"]]

    fig, axes = plt.subplots(3, len(snaps), figsize=(6.9, 6.4), sharex=True, layout="constrained")
    for j, sidx in enumerate(snaps):
        for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "uniform mFR")):
            ii = [rows[k][m] for k in keys]
            F = np.median(z["F_prof_t"][ii, sidx], axis=0)
            Fp = np.median(z["Fp_prof_t"][ii, sidx], axis=0)
            p = np.median(z["phat_t"][ii, sidx], axis=0)
            axes[0, j].plot(x, F - F[np.abs(x) <= 1.5].mean(), color=c, lw=1.2, label=lab)
            axes[1, j].plot(x, Fp, color=c, lw=1.2)
            axes[2, j].plot(x, p, color=c, lw=1.2)
        axes[0, j].plot(x, F_ref, color=PALETTE["black"], lw=0.9, ls=":", label="reference")
        axes[1, j].plot(x, Fp_ref, color=PALETTE["black"], lw=0.9, ls=":")
        axes[2, j].axhline(1.0 / (x[-1] - x[0]), color=PALETTE["black"], lw=0.9, ls=":")
        axes[0, j].set_title(f"t = {round(float(t[sidx])):g}", fontsize=9)
        axes[2, j].set_xlabel("x")
    axes[0, 0].set_ylabel(r"$\hat F_t(x)$")
    axes[1, 0].set_ylabel(r"$\hat F'_t(x)$")
    axes[2, 0].set_ylabel(r"marginal $\hat p_t(x)$")
    for j in range(len(snaps)):
        axes[1, j].set_ylim(-8, 8)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Gateway: F / mean force / xi-marginal convergence\n"
                 "(median over 32 seeds, init: left; dotted = reference / uniform)",
                 fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_gw_profiles"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    apply_publication_style()
    z, rows = load()
    fig_convergence(z, rows)
    fig_ratio(z, rows)
    fig_mechanism(z, rows)
    fig_genealogy(z, rows)
    fig_paired(z, rows)
    fig_profiles(z, rows)
    print(f"wrote figures -> {OUT}")


if __name__ == "__main__":
    main()
