"""Stage 1: plain-SHUS-only calibration screen on the gateway (NO FR anywhere).

Runs 5 preregistered cells x 8 calibration seeds in one batched GPU call, stores full
records, then applies the Gate-1 analysis frozen in docs/PREREGISTRATION_GATEWAY.md:

    T_hit  : first persistent time the right basin holds >= 1 walker (hold 0.05)
    T_est  : first time the TRAILING-WINDOW MEDIAN of D_t = KL(p_hat||u) <= D_tol
             (hold 0.10; median rule is robust to single-save KL spikes)
    D_tol  : 1.5 x (KL* + noise95), where KL* is the ANALYTIC marginal floor of the
             mollified-SHUS fixed point on that cell (gw.mollified_fixed_point) and
             noise95 the finite-K KDE noise floor -- both computable before any run
    eligible for FR  <=>  median T_hit/T <= 0.2  AND  median T_est/T >= 0.4

Cells: an easy negative control plus a betaH = 8 kT family whose geometry (s, r)
varies; the anchor (beta=16, s=0.10, r=32) was establishment-limited under ABF but
must EARN that label again under SHUS.

Usage:
    python scripts/run_stage1_calibration.py                 # the screen
    python scripts/run_stage1_calibration.py --dt-check      # dt-halving stability
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from abpfr.diagnostics import (establishment_time_median, hit_time, kde_noise_floor)
from abpfr.io import save_run
from abpfr.metrics import cosine_modes, l2_error_gauge
from abpfr.systems import gateway as gw

SEEDS = list(range(8))                      # calibration seeds, frozen
COMMON = dict(K=1024, dt=2e-4, n_steps=500_000, block=20, n_saves=400,
              ess_window_steps=4000)
CELLS = {
    # name          beta   H     s     r      note
    "easy_A":  dict(beta=1.0, H=1.0, s=0.15, r=8.0),    # negative control (smoke cell)
    "mid_B":   dict(beta=4.0, H=2.0, s=0.15, r=8.0),    # betaH=8, wide/mild gate
    "cold_C":  dict(beta=16.0, H=0.5, s=0.15, r=8.0),   # betaH=8, cold, wide gate
    "anchor_D": dict(beta=16.0, H=0.5, s=0.10, r=32.0),  # old ABF anchor: narrow+severe
    "hot_E":   dict(beta=8.0, H=1.0, s=0.10, r=32.0),   # narrow+severe, warmer
}
OUT = "results/stage1_calibration"


def gate1_row(rec, D_tol, F_star, eval_mask):
    t, T = rec["time"], rec["time"][-1]
    th = hit_time(rec["P_regions"][:, 2], t, hold_frac=0.05)
    te = establishment_time_median(rec["kl_u_t"], t, D_tol, hold_frac=0.10)
    # error against the estimator's own analytic limit F* (estimator-consistent
    # convergence, separated from the irreducible mollifier bias)
    eT_eps = float(l2_error_gauge(rec["pmf_t"][-1], F_star, eval_mask))
    return dict(seed=rec["seed"], T=T, T_hit=th, T_est=te,
                hit_frac=th / T if np.isfinite(th) else np.nan,
                est_frac=te / T if np.isfinite(te) else np.nan,
                e0=float(rec["l2_f_t"][0]), eT=float(rec["l2_f_t"][-1]),
                eT_eps=eT_eps,
                I_F=float(rec["int_l2_f"]), D_T=float(rec["kl_u_t"][-1]))


def classify(rows):
    hit = np.array([r["hit_frac"] for r in rows])
    est = np.array([r["est_frac"] for r in rows])
    late_or_missing = ~np.isfinite(hit) | (hit > 0.2)
    if late_or_missing.mean() >= 0.25:
        return "discovery-limited"
    med_est = np.nanmedian(np.where(np.isfinite(est), est, np.inf))
    if med_est >= 0.4:
        return "establishment-limited"
    if med_est <= 0.2:
        return "SHUS-sufficient"
    return "intermediate"


def run_screen(device):
    cfgs, seeds, names = [], [], []
    for name, cell in CELLS.items():
        for sd in SEEDS:
            cfgs.append(gw.GatewayConfig(**cell, **COMMON))
            seeds.append(sd)
            names.append(name)
    print(f"screen: {len(CELLS)} cells x {len(SEEDS)} seeds = {len(cfgs)} rows, "
          f"T={cfgs[0].T_total:.0f}, K={cfgs[0].K}, plain SHUS only")
    t0 = time.time()
    recs = gw.simulate_batch(cfgs, seeds, [gw.SHUS], batch_seed=20260818,
                             device=device, progress=50_000)
    print(f"wall {time.time()-t0:.0f}s")

    os.makedirs(OUT, exist_ok=True)
    D_noise = kde_noise_floor(COMMON["K"], 0.10, gw.GRID, n_rep=512, seed=777)
    noise95 = float(np.quantile(D_noise, 0.95))
    print(f"D_noise 95th pct = {noise95:.5f}")
    fps = {name: gw.mollified_fixed_point(gw.GatewayConfig(**cell, **COMMON))
           for name, cell in CELLS.items()}
    D_tols = {name: 1.5 * (fp["kl_star"] + noise95) for name, fp in fps.items()}
    for name, fp in fps.items():
        print(f"  {name}: analytic floor e*={fp['e_star']:.4f} "
              f"(beta*e*={CELLS[name]['beta']*fp['e_star']:.3f} kT), "
              f"KL*={fp['kl_star']:.5f} -> D_tol={D_tols[name]:.5f}")

    summary = {"noise95": noise95, "common": COMMON, "cells": {}}
    for i, rec in enumerate(recs):
        arrays = {k: rec[k] for k in
                  ("time", "pmf_t", "marginal_t", "x_grid", "F_ref", "Fp_ref",
                   "l2_f_t", "l2_fp_t", "kl_u_t", "tv_u_t", "ess_anc_t", "wmax_t",
                   "n_anc_t", "dep_ref_l2_t", "dep_self_l2_t", "P_regions",
                   "Q_regions")}
        meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
                "config": rec["config"], "method": rec["method"], "seed": rec["seed"],
                "cell": names[i], "stage": "stage1_calibration"}
        save_run(os.path.join(OUT, f"{names[i]}_seed{rec['seed']}"), arrays, meta)

    eval_mask = gw.GRID.eval_mask(torch.device("cpu")).numpy()
    print(f"\n{'cell':<10s} {'barrier':>7s} {'T_hit/T':>12s} {'T_est/T':>12s} "
          f"{'I_F':>8s} {'e_F(T)':>8s} {'e_eps(T)':>9s} {'eps*':>8s}  classification")
    for name, cell in CELLS.items():
        F_star = fps[name]["F_star"].numpy()
        rows = [gate1_row(r, D_tols[name], F_star, eval_mask)
                for i, r in enumerate(recs) if names[i] == name]
        cls = classify(rows)
        med = lambda k: float(np.nanmedian([r[k] for r in rows]))
        n_nan_hit = sum(not np.isfinite(r["hit_frac"]) for r in rows)
        n_nan_est = sum(not np.isfinite(r["est_frac"]) for r in rows)
        eps_star = med("eT")     # prereg: median final plain-SHUS error (per cell)
        cfg = gw.GatewayConfig(**cell, **COMMON)
        print(f"{name:<10s} {cfg.barrier_kT():>6.1f}k "
              f"{med('hit_frac'):>9.3f}({n_nan_hit}c) "
              f"{med('est_frac'):>9.3f}({n_nan_est}c) "
              f"{med('I_F'):>8.2f} {med('eT'):>8.4f} {med('eT_eps'):>9.4f} "
              f"{eps_star:>8.4f}  {cls}")
        summary["cells"][name] = {
            "config": cell, "barrier_kT": cfg.barrier_kT(), "rows": rows,
            "classification": cls, "eps_star": eps_star,
            "D_tol": D_tols[name], "e_star": fps[name]["e_star"],
            "kl_star": fps[name]["kl_star"],
            "median": {k: med(k) for k in
                       ("hit_frac", "est_frac", "I_F", "e0", "eT", "eT_eps", "D_T")}}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nsummary -> {OUT}/summary.json")
    make_figures(recs, names, D_tols)


def make_figures(recs, names, D_tols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = list(CELLS)
    cmap = {c: f"C{i}" for i, c in enumerate(cells)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for c in cells:
        rs = [r for i, r in enumerate(recs) if names[i] == c]
        t = rs[0]["time"]
        for key, ax, log in (("l2_f_t", axes[0, 0], True), ("kl_u_t", axes[0, 1], True)):
            curves = np.stack([r[key] for r in rs])
            med = np.median(curves, 0)
            lo, hi = np.quantile(curves, [0.25, 0.75], axis=0)
            ax.plot(t, med, color=cmap[c], label=c)
            ax.fill_between(t, lo, hi, color=cmap[c], alpha=0.15, lw=0)
            if log:
                ax.set_yscale("log")
        occ = np.stack([r["P_regions"][:, 2] for r in rs])
        axes[1, 0].plot(t, np.median(occ, 0), color=cmap[c])
        a1 = np.stack([np.abs(cosine_modes(r["pmf_t"] - r["F_ref"], r["x_grid"],
                                           gw.GRID.eval_lo, gw.GRID.eval_hi,
                                           k_max=1))[:, 0] for r in rs])
        axes[1, 1].plot(t, np.median(a1, 0), color=cmap[c])
        axes[1, 1].set_yscale("log")
    for c in cells:
        axes[0, 1].axhline(D_tols[c], color=cmap[c], ls="--", lw=0.8, alpha=0.7)
    axes[0, 0].set_ylabel(r"$e_F(t)$ (median, IQR)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_ylabel(r"KL$(\hat p_t\|u)$  (dashed: per-cell $D_{\rm tol}$)")
    axes[1, 0].set_ylabel(r"right-basin occupancy $P_+$")
    axes[1, 1].set_ylabel(r"slow bias-error mode $|a_1(t)|$")
    for ax in axes[1]:
        ax.set_xlabel("t")
    fig.suptitle("Stage 1: plain SHUS across gateway cells (8 seeds; no FR)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stage1_overview.png"), dpi=130)
    print(f"figure -> {OUT}/stage1_overview.png")


def run_dt_check(device):
    """dt-halving on the stiffest cell: the discretization floor must be well below
    the errors the campaign will interpret."""
    cell = CELLS["anchor_D"]
    rows = []
    for dt, n in ((2e-4, 500_000), (1e-4, 1_000_000)):
        cfgs = [gw.GatewayConfig(**cell, **{**COMMON, "dt": dt, "n_steps": n})
                for _ in range(4)]
        recs = gw.simulate_batch(cfgs, [100, 101, 102, 103], [gw.SHUS],
                                 batch_seed=555, device=device, progress=100_000)
        eT = [float(r["l2_f_t"][-1]) for r in recs]
        rows.append((dt, float(np.median(eT)), eT))
        print(f"dt={dt:g}: e_F(T) per seed {['%.4f' % e for e in eT]}, "
              f"median {np.median(eT):.4f}")
    d = abs(rows[0][1] - rows[1][1])
    print(f"dt-halving shift of median e_F(T): {d:.4f} "
          f"({'OK' if d < 0.5 * rows[1][1] or d < 0.01 else 'INVESTIGATE'})")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "dt_check.json"), "w") as f:
        json.dump({"cell": cell, "rows": rows}, f, indent=2, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt-check", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu") if a.cpu else gw.DEVICE
    print(f"device: {dev}")
    if a.dt_check:
        run_dt_check(dev)
    else:
        run_screen(dev)
