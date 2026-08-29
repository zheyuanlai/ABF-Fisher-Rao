#!/usr/bin/env python
"""Figures for Stage 3 of the uniform-FR campaign (alanine, abf vs fr_uniform).

Under results/uniform_campaign/alanine/figures/ (png+pdf):

  fig_ala_convergence   kernel-matched aligned-L2 FES error vs time, both arms
  fig_ala_ratio         R_F(t) per-seed spaghetti + median
  fig_ala_mechanism     KL(p_t || uniform-on-torus) and basin fractions vs t
  fig_ala_genealogy     age-aware ancestor ESS/N, max lineage share, cumulative events
  fig_ala_paired        paired per-seed integrated FES error (eval window)
  fig_ala_marginals     2-D (phi, psi) walker-marginal snapshots, ABF vs uniform mFR

    python scripts/plot_uniform_alanine.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "..", "src"))
from publication_style import PALETTE, apply_publication_style, save_figure  # noqa: E402
from alanine.metrics_ala import aligned_l2, build_masks, smooth_reference  # noqa: E402

ROOT = os.path.join(SCRIPTS, "..")
RAW = os.path.join(ROOT, "results/uniform_campaign/alanine/N2048_uniform/raw")
REF = os.path.join(ROOT, "results/alanine/reference/reference.npz")
OUT = os.path.join(ROOT, "results/uniform_campaign/alanine/figures")

C_ABF = PALETTE["blue"]
C_UNI = PALETTE["vermillion"]
T_FR = 20.0     # ps
WINDOW = (20.0, 100.0)


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RAW, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        runs[meta["method"]] = {k: np.asarray(d[k]) for k in d.files if k != "meta"}
    assert set(runs) == {"abf", "fr_uniform"}, f"found {sorted(runs)}"
    return runs


def fes_error_series(runs):
    """Kernel-matched equilibrium-weighted aligned L2 per save per seed, both arms."""
    refd = np.load(REF, allow_pickle=True)
    F_ref = refd["F"]
    meta = json.load(open(os.path.join(os.path.dirname(REF), "meta.json")))
    kT = float(meta["kT_kJ"])
    n_grid = int(meta["n_grid"])
    pack = build_masks(F_ref, kT)
    F_sm = smooth_reference(F_ref, 0.08, n_grid)
    w = pack["weights"]["equilibrium"]
    err = {}
    for m, d in runs.items():
        pmf = d["pmf"]                       # (T, R, n, n)
        T, R = pmf.shape[:2]
        e = np.zeros((T, R))
        for ti in range(T):
            for r in range(R):
                e[ti, r] = aligned_l2(pmf[ti, r], F_sm, w)
        err[m] = e
    return err, np.asarray(runs["abf"]["times"], dtype=float)


def med_iqr(a, axis):
    return (np.median(a, axis), np.percentile(a, 25, axis), np.percentile(a, 75, axis))


def fig_convergence(err, t):
    fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
    for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "ABF + uniform mFR")):
        md, lo, hi = med_iqr(err[m], 1)
        ax.plot(t, md, color=c, lw=1.4, label=lab)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axvline(T_FR, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.text(T_FR + 1, ax.get_ylim()[1] * 0.7, "FR on", fontsize=7, color=PALETTE["gray"])
    ax.set_yscale("log")
    ax.set_xlabel("t (ps)")
    ax.set_ylabel("kernel-matched aligned $L_2$ FES error (kJ/mol)")
    ax.set_title("Alanine (vacuum), 16 paired seeds, N=2048", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, os.path.join(OUT, "fig_ala_convergence"))
    plt.close(fig)


def fig_ratio(err, t):
    fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
    R = err["abf"].shape[1]
    ratios = err["fr_uniform"] / err["abf"]
    for r in range(R):
        ax.plot(t, ratios[:, r], color=C_UNI, alpha=0.18, lw=0.5)
    ax.plot(t, np.median(ratios, 1), color=C_UNI, lw=1.7, label="median ratio")
    ax.axhline(1.0, color=PALETTE["black"], lw=0.8, ls=":")
    ax.axvline(T_FR, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_xlabel("t (ps)")
    ax.set_ylabel(r"$R_F(t) = e_F^{\rm uni}/e_F^{\rm abf}$")
    ax.set_title("below 1 = uniform mFR ahead", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, os.path.join(OUT, "fig_ala_ratio"))
    plt.close(fig)


def fig_mechanism(runs, t):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
    ax = axes[0]
    for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "ABF + uniform mFR")):
        kl = runs[m]["kl_uniform"]           # (T, R)
        md, lo, hi = med_iqr(kl, 1)
        ax.plot(t, md, color=c, lw=1.4, label=lab)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axvline(T_FR, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_xlabel("t (ps)")
    ax.set_ylabel(r"$D_{\rm KL}(\hat p_t\,\|\,{\rm uniform})$")
    ax.set_title("marginal establishment on the torus", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    names = ("C7eq", "C5", "C7ax")
    styles = ("-", "--", ":")
    for m, c in (("abf", C_ABF), ("fr_uniform", C_UNI)):
        bf = runs[m]["basin_frac"]           # (T, R, 3)
        for k, ls in enumerate(styles):
            ax.plot(t, np.median(bf[:, :, k], 1), color=c, lw=1.2, ls=ls)
    for k, (nm, ls) in enumerate(zip(names, styles)):
        ax.plot([], [], color=PALETTE["black"], ls=ls, label=nm)
    ax.axvline(T_FR, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_xlabel("t (ps)")
    ax.set_ylabel("basin fraction")
    ax.set_yscale("log")
    ax.set_title("basin occupancy (blue=ABF, red=uniform mFR)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    save_figure(fig, os.path.join(OUT, "fig_ala_mechanism"))
    plt.close(fig)


def fig_genealogy(runs, t):
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9), layout="constrained")
    ax = axes[0]
    for m, c, lab in (("abf", C_ABF, "ABF"), ("fr_uniform", C_UNI, "uniform mFR")):
        md, lo, hi = med_iqr(runs[m]["ess_age"], 1)
        ax.plot(t, md, color=c, lw=1.4, label=lab)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axhline(0.30, color=PALETTE["black"], lw=0.8, ls=":", label="floor 0.30")
    ax.set_xlabel("t (ps)"); ax.set_ylabel("age-aware ESS / N")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for m, c in (("abf", C_ABF), ("fr_uniform", C_UNI)):
        md, lo, hi = med_iqr(runs[m]["wmax"], 1)
        ax.plot(t, md, color=c, lw=1.4)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axhline(0.05, color=PALETTE["black"], lw=0.8, ls=":", label="cap 0.05")
    ax.set_xlabel("t (ps)"); ax.set_ylabel("max lineage share")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[2]
    N = float(runs["fr_uniform"]["n_replicas"]) if "n_replicas" in runs["fr_uniform"] else 2048.0
    ev = runs["fr_uniform"]["events_cum"] / N
    md, lo, hi = med_iqr(ev, 1)
    ax.plot(t, md, color=C_UNI, lw=1.4)
    ax.fill_between(t, lo, hi, color=C_UNI, alpha=0.18, lw=0)
    ax.set_xlabel("t (ps)"); ax.set_ylabel("cumulative events / N")
    ax.set_title("uniform arm only", fontsize=9)
    fig.suptitle("Alanine genealogy and event budget", fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_ala_genealogy"))
    plt.close(fig)


def fig_paired(err, t):
    sel = (t >= WINDOW[0]) & (t <= WINDOW[1])
    ia = np.trapezoid(err["abf"][sel], t[sel], axis=0)
    iu = np.trapezoid(err["fr_uniform"][sel], t[sel], axis=0)
    fig, ax = plt.subplots(figsize=(3.4, 3.0), layout="constrained")
    for i in range(len(ia)):
        ax.plot([0, 1], [ia[i], iu[i]], color=PALETTE["gray"], alpha=0.4, lw=0.7)
    ax.plot([0, 1], [np.median(ia), np.median(iu)], color=PALETTE["black"], lw=2.0,
            marker="o", ms=4)
    d = 100.0 * (iu - ia) / ia
    ax.set_title(f"paired $I_F$ ({WINDOW[0]:g}-{WINDOW[1]:g} ps)\n"
                 f"median {np.median(d):+.2f}%", fontsize=9)
    ax.set_xticks([0, 1], ["ABF", "uniform\nmFR"])
    ax.set_ylabel("integrated FES error (kJ/mol ps)")
    ax.set_xlim(-0.3, 1.3)
    save_figure(fig, os.path.join(OUT, "fig_ala_paired"))
    plt.close(fig)


def fig_marginals(runs, t):
    times = (20.0, 30.0, 60.0, 100.0)
    snaps = [int(np.argmin(np.abs(t - v))) for v in times]
    fig, axes = plt.subplots(2, len(snaps), figsize=(6.9, 4.2), layout="constrained",
                             sharex=True, sharey=True)
    ext = (-180, 180, -180, 180)
    vmax = None
    for j, sidx in enumerate(snaps):
        for i, (m, lab) in enumerate((("abf", "ABF"), ("fr_uniform", "uniform mFR"))):
            h = np.median(runs[m]["marg_hist"][sidx], 0)          # (n, n) over seeds
            if vmax is None:
                vmax = np.percentile(h, 99.5)
            im = axes[i, j].imshow(h.T, origin="lower", extent=ext, cmap="viridis",
                                   vmin=0, vmax=vmax, aspect="auto")
            if j == 0:
                axes[i, j].set_ylabel(f"{lab}\n$\\psi$ (deg)")
        axes[0, j].set_title(f"t = {t[sidx]:g} ps", fontsize=9)
        axes[1, j].set_xlabel(r"$\phi$ (deg)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="walker marginal (bin probability)")
    fig.suptitle("Alanine: walker marginal on (phi, psi) -- toward uniform?", fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_ala_marginals"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    apply_publication_style()
    runs = load()
    t = np.asarray(runs["abf"]["times"], dtype=float)
    err, t2 = fes_error_series(runs)
    assert np.allclose(t, t2)
    fig_convergence(err, t)
    fig_ratio(err, t)
    fig_mechanism(runs, t)
    fig_genealogy(runs, t)
    fig_paired(err, t)
    fig_marginals(runs, t)
    print(f"wrote figures -> {OUT}")


if __name__ == "__main__":
    main()
