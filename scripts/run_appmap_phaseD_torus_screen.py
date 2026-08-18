"""Application-map Phase D1: plain-SHUS screen on the analytic torus (NO FR).

Design frozen in docs/PREREGISTRATION_APPLICATION_MAP.md before this run: four
cells x 8 seeds, one batch, frozen numerics (72x72 grid, eps 0.06, eta 0.25,
K = 1024, dt = 1e-3, T = 200). Gates: T_hit = all four basins persistently
occupied (hold 0.05); T_est = trailing-median KL rule with
D_tol = 1.5 x (KL* + noise95_2D); eligibility = hit <= 0.2 AND gap >= 0.25.

Usage: python scripts/run_appmap_phaseD_torus_screen.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from abpfr.diagnostics import (establishment_time_median, first_persistent,
                               kde_noise_floor2)
from abpfr.io import save_run
from abpfr.systems import torus2d as t2
from abpfr.systems.gateway import Method

SEEDS = list(range(8))
COMMON = dict(K=1024, dt=1e-3, n_steps=200_000, block=20, eps_bw=0.06,
              eta_bw=0.25, n_saves=400, profile_every=8, ess_window_steps=4000)
CELLS = {
    "t_easy":   dict(beta=1.0, H1=0.8, H2=0.8, Hc=0.2),
    "t_mid":    dict(beta=4.0, H1=1.5, H2=1.5, Hc=0.5),
    "t_cold":   dict(beta=8.0, H1=1.0, H2=1.0, Hc=0.25),
    "t_anchor": dict(beta=8.0, H1=1.5, H2=1.5, Hc=0.75),
}
BATCH_SEED = 20260827
OUT = "results/appmap_phaseD_torus_screen"


def classify(rows):
    hit = np.array([r["hit_frac"] for r in rows])
    est = np.array([r["est_frac"] for r in rows])
    late = ~np.isfinite(hit) | (hit > 0.2)
    if late.mean() >= 0.25:
        return "discovery-limited"
    gap = np.where(np.isfinite(est), est, np.inf) - hit
    med_gap = float(np.nanmedian(gap))
    if med_gap >= 0.25:
        return "establishment-limited"
    med_est = float(np.nanmedian(np.where(np.isfinite(est), est, np.inf)))
    if med_est <= 0.2:
        return "SHUS-sufficient"
    return "intermediate"


def main():
    device = t2.DEVICE
    print(f"device: {device}")
    cfgs, seeds, names = [], [], []
    for name, cell in CELLS.items():
        for sd in SEEDS:
            cfgs.append(t2.Torus2DConfig(**cell, **COMMON))
            seeds.append(sd)
            names.append(name)
    T = cfgs[0].T_total
    print(f"Phase D1: {len(CELLS)} cells x {len(SEEDS)} seeds = {len(cfgs)} rows, "
          f"T={T:.0f}, K={COMMON['K']}, plain SHUS only")
    t0 = time.time()
    recs = t2.simulate_batch(cfgs, seeds, [Method("shus")], batch_seed=BATCH_SEED,
                             device=device, progress=25_000)
    print(f"wall {time.time()-t0:.0f}s")

    os.makedirs(OUT, exist_ok=True)
    for i, rec in enumerate(recs):
        arrays = {k: rec[k] for k in
                  ("time", "profile_time", "pmf_t", "marginal_t", "x1_grid",
                   "x2_grid", "F_ref", "l2_f_t", "kl_u_t", "tv_u_t", "ess_anc_t",
                   "wmax_t", "n_anc_t", "dep_ref_l2_t", "dep_self_l2_t",
                   "P_regions")}
        meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
                "config": rec["config"], "method": rec["method"], "seed": rec["seed"],
                "cell": names[i], "stage": "appmap_phaseD_torus_screen"}
        save_run(os.path.join(OUT, f"{names[i]}_seed{rec['seed']}"), arrays, meta)

    noise95 = float(np.quantile(
        kde_noise_floor2(COMMON["K"], COMMON["eta_bw"], t2.GRID2, n_rep=256,
                         seed=777), 0.95))
    print(f"noise95_2D = {noise95:.5f}")
    floors = {name: t2.analytic_floors(t2.Torus2DConfig(**cell, **COMMON))
              for name, cell in CELLS.items()}
    D_tols = {name: 1.5 * (fp["kl_star"] + noise95) for name, fp in floors.items()}

    summary = {"noise95": noise95, "common": COMMON, "batch_seed": BATCH_SEED,
               "cells": {}}
    print(f"\n{'cell':<9s} {'barrier':>7s} {'split':>6s} {'D_tol':>8s} "
          f"{'T_hit/T':>9s} {'T_est/T':>10s} {'gap/T':>7s} {'e_F(T)':>8s} "
          f"{'eligible':>9s}  classification")
    for name, cell in CELLS.items():
        rows = []
        for i, rec in enumerate(recs):
            if names[i] != name:
                continue
            t = rec["time"]
            occupied_all = (rec["P_regions"] > 0).all(axis=1)
            th = first_persistent(occupied_all, t, hold_frac=0.05)
            te = establishment_time_median(rec["kl_u_t"], t, D_tols[name],
                                           hold_frac=0.10)
            gap = (np.nan if not np.isfinite(th)
                   else (np.inf if not np.isfinite(te) else (te - th) / T))
            rows.append(dict(seed=rec["seed"], T_hit=th, T_est=te,
                             hit_frac=th / T if np.isfinite(th) else np.nan,
                             est_frac=te / T if np.isfinite(te) else np.nan,
                             gap_frac=gap, I_F=float(rec["int_l2_f"]),
                             eT=float(rec["l2_f_t"][-1]),
                             D_T=float(rec["kl_u_t"][-1]),
                             overshoot=[float(rec["P_regions"][:, k].max())
                                        for k in range(4)]))
        cls = classify(rows)
        med = lambda k: float(np.nanmedian([r[k] for r in rows]))
        med_gap = float(np.nanmedian([r["gap_frac"] for r in rows]))
        eligible = bool(med("hit_frac") <= 0.2 and med_gap >= 0.25)
        cfg = t2.Torus2DConfig(**cell, **COMMON)
        n_c = sum(not np.isfinite(r["est_frac"]) for r in rows)
        print(f"{name:<9s} {cfg.barrier_kT():>6.1f}k "
              f"{2*cell['Hc']*cell['beta']:>5.1f}k {D_tols[name]:>8.4f} "
              f"{med('hit_frac'):>9.3f} {med('est_frac'):>9.3f}({n_c}c) "
              f"{med_gap:>7.3f} {med('eT'):>8.4f} {str(eligible):>9s}  {cls}")
        summary["cells"][name] = {
            "config": cell, "barrier_kT": cfg.barrier_kT(), "rows": rows,
            "classification": cls, "eligible": eligible, "D_tol": D_tols[name],
            "e_star": floors[name]["e_star"], "kl_star": floors[name]["kl_star"],
            "median": {k: med(k) for k in
                       ("hit_frac", "est_frac", "I_F", "eT", "D_T")},
            "median_gap": med_gap}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nsummary -> {OUT}/summary.json")
    make_figure(recs, names, D_tols)


def make_figure(recs, names, D_tols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = {c: f"C{i}" for i, c in enumerate(CELLS)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for c in CELLS:
        rs = [r for i, r in enumerate(recs) if names[i] == c]
        t = rs[0]["time"]
        for key, ax in (("l2_f_t", axes[0, 0]), ("kl_u_t", axes[0, 1])):
            curves = np.stack([r[key] for r in rs])
            ax.plot(t, np.median(curves, 0), color=cmap[c], label=c)
            ax.fill_between(t, *np.quantile(curves, [0.25, 0.75], axis=0),
                            color=cmap[c], alpha=0.15, lw=0)
            ax.set_yscale("log")
        # slowest basin to fill: the minimum region occupancy over time
        occ = np.stack([r["P_regions"].min(axis=1) for r in rs])
        axes[1, 0].plot(t, np.median(occ, 0), color=cmap[c])
        dep = np.stack([r["dep_self_l2_t"] for r in rs])
        axes[1, 1].plot(t, np.median(dep, 0), color=cmap[c])
    for c in CELLS:
        axes[0, 1].axhline(D_tols[c], color=cmap[c], ls="--", lw=0.8, alpha=0.7)
    axes[0, 0].set_ylabel(r"$e_F(t)$ (median, IQR)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_ylabel(r"KL$(\hat p_t\|u)$ (dashed: $D_{\rm tol}$)")
    axes[1, 0].set_ylabel("min basin occupancy")
    axes[1, 1].set_ylabel(r"$\|d_n - r_n\|$ (self-feed)")
    for ax in axes[1]:
        ax.set_xlabel("t")
    fig.suptitle("Phase D1: plain SHUS across torus cells (8 seeds; no FR)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "phaseD1_overview.png"), dpi=130)
    print(f"figure -> {OUT}/phaseD1_overview.png")


if __name__ == "__main__":
    main()
