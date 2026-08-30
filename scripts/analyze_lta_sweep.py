#!/usr/bin/env python
"""Sweep-level analysis for the LTA temperature sweep (Stage 5).

Collects the per-T summaries and references and produces the discriminating
comparison the sweep was preregistered for: benefit vs T plotted against BOTH
candidate predictors -- entropy share (from the references) and establishment
starvation (measured on the ABF arm itself).  Also writes the sweep figure set:

  fig_lta_sweep_profiles       F(z)/kT per T + barrier decomposition vs T
  fig_lta_sweep_benefit        Delta I_F and final Delta vs T (CI bars)
  fig_lta_sweep_predictors     benefit vs entropy share AND vs starvation

    python scripts/analyze_lta_sweep.py
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
LTA = os.path.join(ROOT, "results/uniform_campaign/lta")
OUT = os.path.join(LTA, "figures")
PREREG = os.path.join(ROOT, "configs/uniform_campaign/lta_sweep_prereg.json")
TEMPS = (80, 150, 225, 300)
PI = math.pi


def main():
    pre = json.load(open(PREREG))
    fr_start = int(pre["sampler"]["fr_start_steps"])
    rows = []
    for T in TEMPS:
        tkey = f"{T:g}"
        summ = json.load(open(os.path.join(LTA, f"summary_T{tkey}.json")))
        ref = np.load(os.path.join(LTA, "reference", f"reference_T{tkey}.npz"),
                      allow_pickle=True)
        # starvation measured on the ABF arm: relative error still to go when FR
        # came on, e_F(fr_start)/e_F(0), and tau_abf(e0/8) from the summary
        z = np.load(os.path.join(LTA, f"production_T{tkey}", "abf.npz"),
                    allow_pickle=True)
        F_ref = circular_interp_ref(ref["F"], ref["grid_phi"], z["grid"])
        err = error_series(np.asarray(z["pmf"], dtype=float), F_ref)
        steps = np.asarray(z["steps"])
        i_fr = int(np.searchsorted(steps, fr_start))
        med = np.median(err, axis=1)
        starv_frac = float(med[i_fr] / med[0])
        # the direct dynamical starvation measure: how much window traffic the ABF
        # arm generates on its own, per replica, over the whole budget
        abf_cross_per_rep = float(np.asarray(z["n_cage_crossings"]).sum() / (16 * 1024))
        rows.append(dict(
            T=T, kT=float(ref["kT"]),
            dF_kT=summ["reference"]["dF_barrier_kT"],
            dU_kT=summ["reference"]["dU_barrier_kT"],
            mTdS_kT=summ["reference"]["mTdS_barrier_kT"],
            entropic_fraction=summ["reference"]["entropic_fraction"],
            d_int=summ["d_int_pct"]["median"], d_int_ci=summ["d_int_pct"]["ci95"],
            d_fin=summ["d_final_pct"]["median"], d_fin_ci=summ["d_final_pct"]["ci95"],
            wins=summ["d_int_pct"]["wins"], n=summ["n_pairs"],
            tau_abf_e08=summ["time_to_accuracy"]["e0/8"]["tau_abf"],
            starv_err_frac_at_fr_start=starv_frac,
            abf_crossings_per_replica=abf_cross_per_rep,
            verdict=summ["verdict"], fr_rate=summ["fr_rate"],
            ess=summ["health"]["median_min_ess_frac"]))
        print(f"T={T:>3} K: dF={rows[-1]['dF_kT']:.1f} kT "
              f"(-TdS {rows[-1]['mTdS_kT']:.1f}, {100*rows[-1]['entropic_fraction']:.0f}%)  "
              f"dI_F {rows[-1]['d_int']:+.2f}% {rows[-1]['d_int_ci']}  "
              f"final {rows[-1]['d_fin']:+.2f}%  "
              f"starv(eF@FRstart/e0) {starv_frac:.3f}  tau_abf(e0/8) "
              f"{rows[-1]['tau_abf_e08']:.1f}  {rows[-1]['verdict']}")

    with open(os.path.join(LTA, "sweep_summary.json"), "w") as fh:
        json.dump(dict(rows=rows, fr_start_steps=fr_start,
                       prereg=os.path.relpath(PREREG, ROOT)), fh, indent=2)

    apply_publication_style()
    Ts = [r["T"] for r in rows]

    # ---- profiles + decomposition ----
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
    cmap = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["vermillion"]]
    ax = axes[0]
    for r, c in zip(rows, cmap):
        ref = np.load(os.path.join(LTA, "reference", f"reference_T{r['T']:g}.npz"),
                      allow_pickle=True)
        ax.plot(ref["z"], ref["F"] / r["kT"], color=c, lw=1.3, label=f"{r['T']} K")
    ax.set_xlabel("z (A)")
    ax.set_ylabel(r"$F(z)/k_BT$")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("reference free energy per T", fontsize=9)
    ax = axes[1]
    for key, c, lab in (("dF_kT", PALETTE["black"], r"$\Delta F^\ddagger$"),
                        ("dU_kT", PALETTE["blue"], r"$\Delta U^\ddagger$"),
                        ("mTdS_kT", PALETTE["vermillion"], r"$-T\Delta S^\ddagger$")):
        ax.plot(Ts, [r[key] for r in rows], color=c, marker="o", ms=4, lw=1.3, label=lab)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("barrier (kT)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("decomposition: dU grows as 1/T, -TdS ~ const", fontsize=9)
    save_figure(fig, os.path.join(OUT, "fig_lta_sweep_profiles"))
    plt.close(fig)

    # ---- benefit vs T ----
    fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
    y = [r["d_int"] for r in rows]
    yerr = np.array([[r["d_int"] - r["d_int_ci"][0] for r in rows],
                     [r["d_int_ci"][1] - r["d_int"] for r in rows]])
    ax.errorbar(Ts, y, yerr=yerr, color=PALETTE["vermillion"], marker="o", ms=5,
                lw=1.4, capsize=3, label=r"$\Delta I_F$ (integrated)")
    yf = [r["d_fin"] for r in rows]
    yferr = np.array([[r["d_fin"] - r["d_fin_ci"][0] for r in rows],
                      [r["d_fin_ci"][1] - r["d_fin"] for r in rows]])
    ax.errorbar(Ts, yf, yerr=yferr, color=PALETTE["blue"], marker="s", ms=4,
                lw=1.2, ls="--", capsize=3, label=r"final $\Delta e_F(T)$")
    ax.axhline(0, color=PALETTE["black"], lw=0.8, ls=":")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("paired median change (%)  [negative = uniform better]")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("uniform-FR benefit vs temperature", fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_lta_sweep_benefit"))
    plt.close(fig)

    # ---- benefit vs the two predictors ----
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
    for ax, xkey, xlab in (
            (axes[0], "entropic_fraction", r"entropy share $-T\Delta S^\ddagger/\Delta F^\ddagger$"),
            (axes[1], "abf_crossings_per_replica",
             "starvation: ABF cage crossings per replica (log)")):
        x = [r[xkey] for r in rows]
        ax.errorbar(x, y, yerr=yerr, color=PALETTE["vermillion"], marker="o", ms=5,
                    lw=0, elinewidth=1.2, capsize=3)
        for r, xi, yi in zip(rows, x, y):
            ax.annotate(f"{r['T']}K", (xi, yi), textcoords="offset points",
                        xytext=(5, 4), fontsize=7)
        ax.axhline(0, color=PALETTE["black"], lw=0.8, ls=":")
        if "log" in xlab:
            ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"$\Delta I_F$ (%)")
    fig.suptitle("which predictor tracks the benefit? (preregistered contrast)",
                 fontsize=9.5)
    save_figure(fig, os.path.join(OUT, "fig_lta_sweep_predictors"))
    plt.close(fig)
    print(f"wrote sweep summary + figures -> {OUT}")


if __name__ == "__main__":
    main()
