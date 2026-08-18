"""Application-map Phase B: WCA population-size / resource-scaling map (NO FR).

Design frozen in docs/PREREGISTRATION_APPLICATION_MAP.md (commit ae1a998, before this
run): plain SHUS on b2h6 (stress) + b1h2 (control) at K in {32, 64, 128, 256},
seeds 0..7, all Stage-4 conventions unchanged; the frozen Stage-4 screen provides the
K = 1024 anchor (same seeds/protocol, results/stage4_wca_screen). D_tol(K) adapts
only through the analytic finite-K KDE noise floor. Eligibility per (cell, K):
median T_hit/T <= 0.2 AND median (T_est - T_hit)/T >= 0.25.

Usage:
    python scripts/run_appmap_phaseB_kscaling.py --K 32     # one K batch (both cells)
    python scripts/run_appmap_phaseB_kscaling.py --analyze  # map + eligibility
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from abpfr.diagnostics import (establishment_time_median, hit_time,
                               kde_noise_floor)
from abpfr.io import save_run
from abpfr.systems import wca
from abpfr.systems.gateway import SHUS

SEEDS = list(range(8))                       # same seeds as the Stage-4 anchor
COMMON = dict(dt=2e-3, n_steps=250_000, block=20, n_saves=400,
              ess_window_steps=4000)         # K supplied per batch
CELLS = {
    "b1h2": dict(beta=1.0, h=2.0),           # easy control
    "b2h6": dict(beta=2.0, h=6.0),           # primary stress cell (~12 kT barrier)
}
K_LADDER = (32, 64, 128, 256)                # new runs; 1024 = frozen Stage-4 anchor
STAGE4 = "results/stage4_wca_screen"
OUT = "results/appmap_phaseB_kscaling"


def kl_floors():
    """Per-cell analytic KL* of the mollified fixed point (K-independent)."""
    floors = {}
    for name, cell in CELLS.items():
        cfg = wca.WCAConfig(**cell, K=64, **COMMON)
        if name == "b1h2":
            F_ref, _ = wca.load_reference(cfg, device="cpu")
        else:
            F_ref = wca.load_gate_proxy(name, device="cpu")
        e_star, kl_star = wca.mollified_marginal_floor(
            F_ref, cell["beta"], cfg.eps_bw, device="cpu")
        floors[name] = dict(e_star=e_star, kl_star=kl_star)
    return floors


def d_tol(name, K, floors, noise_cache={}):
    """D_tol(cell, K): only the finite-K noise component varies with K (prereg)."""
    if K not in noise_cache:
        eta = wca.WCAConfig(**CELLS["b1h2"], K=K, **COMMON).eta_bw
        noise_cache[K] = float(np.quantile(
            kde_noise_floor(K, eta, wca.GRID, n_rep=512, seed=777), 0.95))
    return 1.5 * (floors[name]["kl_star"] + noise_cache[K]), noise_cache[K]


def classify(rows):
    """Stage-4 classification, unchanged (frozen vocabulary)."""
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


def run_K(K):
    device = wca.DEVICE
    cfgs, seeds, names = [], [], []
    for name, cell in CELLS.items():
        for sd in SEEDS:
            cfgs.append(wca.WCAConfig(**cell, K=K, **COMMON))
            seeds.append(sd)
            names.append(name)
    print(f"Phase B, K={K}: {len(CELLS)} cells x {len(SEEDS)} seeds = {len(cfgs)} "
          f"rows of {K} boxes ({cfgs[0].n_particles} particles), "
          f"T={cfgs[0].T_total:.0f}, plain SHUS only")
    t0 = time.time()
    recs = wca.simulate_batch(cfgs, seeds, [SHUS], batch_seed=20260825 + K,
                              device=device, progress=25_000)
    wall = time.time() - t0
    print(f"wall {wall:.0f}s "
          f"({COMMON['n_steps']*len(cfgs)*K/wall/1e6:.1f}M box-steps/s)")
    os.makedirs(OUT, exist_ok=True)
    for i, rec in enumerate(recs):
        arrays = {k: rec[k] for k in
                  ("time", "pmf_t", "marginal_t", "x_grid", "F_ref", "l2_f_t",
                   "kl_u_t", "tv_u_t", "ess_anc_t", "wmax_t", "n_anc_t",
                   "dep_ref_l2_t", "dep_self_l2_t", "P_regions")}
        meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
                "config": rec["config"], "method": rec["method"], "seed": rec["seed"],
                "cell": names[i], "K": K, "stage": "appmap_phaseB_kscaling"}
        save_run(os.path.join(OUT, f"{names[i]}_K{K}_seed{rec['seed']}"),
                 arrays, meta)
    print(f"records -> {OUT}/*_K{K}_seed*.npz")


def load_rows(cell, K, floors):
    """Gate metrics for one (cell, K) from stored records (Stage-4 files at 1024)."""
    tol, noise95 = d_tol(cell, K, floors)
    rows = []
    for sd in SEEDS:
        path = (f"{STAGE4}/{cell}_seed{sd}.npz" if K == 1024
                else f"{OUT}/{cell}_K{K}_seed{sd}.npz")
        if not os.path.exists(path):
            return None, tol, noise95
        with np.load(path) as z:
            t, P, kl = z["time"], z["P_regions"], z["kl_u_t"]
            eT = float(z["l2_f_t"][-1])
        T = float(t[-1])
        th = hit_time(P[:, 2], t, hold_frac=0.05)
        te = establishment_time_median(kl, t, tol, hold_frac=0.10)
        # gap convention mirrors Stage-4 classify: censored T_est -> inf gap,
        # missing discovery -> nan gap (dropped by nanmedian; 'late' rule catches it)
        gap = (np.nan if not np.isfinite(th)
               else (np.inf if not np.isfinite(te) else (te - th) / T))
        rows.append(dict(seed=sd, T_hit=th, T_est=te,
                         hit_frac=th / T if np.isfinite(th) else np.nan,
                         est_frac=te / T if np.isfinite(te) else np.nan,
                         gap_frac=gap, D_T=float(kl[-1]), eT=eT))
    return rows, tol, noise95


def analyze():
    floors = kl_floors()
    ladder = list(K_LADDER) + [1024]
    summary = {"common": COMMON, "cells": {}, "floors": floors, "ladder": ladder}
    print(f"\n{'cell':<6s} {'K':>5s} {'D_tol':>8s} {'T_hit/T':>9s} {'T_est/T':>10s} "
          f"{'gap/T':>7s} {'eligible':>9s}  classification")
    for name in CELLS:
        summary["cells"][name] = {}
        for K in ladder:
            rows, tol, noise95 = load_rows(name, K, floors)
            if rows is None:
                print(f"{name:<6s} {K:>5d} {tol:>8.4f}   (no records yet)")
                continue
            cls = classify(rows)
            med = lambda k: float(np.nanmedian([r[k] for r in rows]))
            gaps = np.array([r["gap_frac"] for r in rows])
            med_gap = float(np.nanmedian(gaps))
            eligible = bool(med("hit_frac") <= 0.2 and med_gap >= 0.25)
            n_c = sum(not np.isfinite(r["est_frac"]) for r in rows)
            print(f"{name:<6s} {K:>5d} {tol:>8.4f} {med('hit_frac'):>9.3f} "
                  f"{med('est_frac'):>9.3f}({n_c}c) {med_gap:>7.3f} "
                  f"{str(eligible):>9s}  {cls}")
            summary["cells"][name][str(K)] = {
                "rows": rows, "classification": cls, "eligible": eligible,
                "D_tol": tol, "noise95": noise95,
                "median": {"hit_frac": med("hit_frac"),
                           "est_frac": med("est_frac"), "gap_frac": med_gap,
                           "D_T": med("D_T"), "eT": med("eT")}}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nsummary -> {OUT}/summary.json")
    make_figure(summary)


def make_figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, color in (("b1h2", "C0"), ("b2h6", "C3")):
        cells = summary["cells"][name]
        Ks = sorted(int(k) for k in cells)
        hit = [cells[str(K)]["median"]["hit_frac"] for K in Ks]
        est = [cells[str(K)]["median"]["est_frac"] for K in Ks]
        gap = [min(cells[str(K)]["median"]["gap_frac"], 1.2) for K in Ks]
        axes[0].plot(Ks, hit, "o-", color=color, label=name)
        axes[1].plot(Ks, est, "o-", color=color)
        axes[2].plot(Ks, gap, "o-", color=color)
    for ax, ylab in zip(axes, (r"median $T_{\rm hit}/T$",
                               r"median $T_{\rm est}/T$",
                               r"median gap $(T_{\rm est}-T_{\rm hit})/T$")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K (replicas)")
        ax.set_ylabel(ylab)
    axes[0].axhline(0.2, ls="--", c="gray", lw=0.8)
    axes[2].axhline(0.25, ls="--", c="gray", lw=0.8)
    axes[0].legend()
    fig.suptitle("Phase B: WCA discovery/establishment vs population size "
                 "(8 seeds; plain SHUS; K=1024 = frozen Stage-4 anchor)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "phaseB_kscaling_map.png"), dpi=130)
    print(f"figure -> {OUT}/phaseB_kscaling_map.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, choices=list(K_LADDER))
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    elif a.K:
        run_K(a.K)
    else:
        ap.error("pass --K VALUE or --analyze")
