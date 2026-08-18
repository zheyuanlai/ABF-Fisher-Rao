"""Stage 4: WCA dimer plain-SHUS screen (NO FR) — 3 cells x 8 seeds, one batch.

Gates preregistered in docs/PREREGISTRATION_WCA.md. e_F only for b1h2 (hp_v3);
b2h6/b4h1 gate on reference-free diagnostics with proxy-derived D_tol.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from abpfr.diagnostics import (establishment_time_median, hit_time,
                               kde_noise_floor)
from abpfr.io import save_run
from abpfr.systems import wca
from abpfr.systems.gateway import SHUS

SEEDS = list(range(8))
COMMON = dict(K=1024, dt=2e-3, n_steps=250_000, block=20, n_saves=400,
              ess_window_steps=4000)
CELLS = {
    "b1h2": dict(beta=1.0, h=2.0),
    "b2h6": dict(beta=2.0, h=6.0),
    "b4h1": dict(beta=4.0, h=1.0),
}
OUT = "results/stage4_wca_screen"


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


def tolerances(verbose=True):
    """Preregistered per-cell D_tol (printed before any interpretation)."""
    noise95 = float(np.quantile(
        kde_noise_floor(COMMON["K"], WCAB().eta_bw, wca.GRID, n_rep=512, seed=777),
        0.95))
    D_tols, floors = {}, {}
    for name, cell in CELLS.items():
        if name == "b1h2":
            F_ref, _ = wca.load_reference(wca.WCAConfig(**cell, **COMMON),
                                          device="cpu")
        else:
            F_ref = wca.load_gate_proxy(name, device="cpu")
        e_star, kl_star = wca.mollified_marginal_floor(
            F_ref, cell["beta"], WCAB().eps_bw, device="cpu")
        D_tols[name] = 1.5 * (kl_star + noise95)
        floors[name] = dict(e_star=e_star, kl_star=kl_star)
        if verbose:
            print(f"  {name}: floors e*={e_star:.4f} "
                  f"(beta*e*={cell['beta']*e_star:.3f} kT), KL*={kl_star:.5f} "
                  f"-> D_tol={D_tols[name]:.5f}"
                  + ("  [proxy-derived, gate only]" if name != "b1h2" else ""))
    return noise95, D_tols, floors


def WCAB():
    return wca.WCAConfig(**CELLS["b1h2"], **COMMON)


def run_cell(cell_name):
    device = wca.DEVICE
    cell = CELLS[cell_name]
    cfgs = [wca.WCAConfig(**cell, **COMMON) for _ in SEEDS]
    print(f"cell {cell_name}: {len(SEEDS)} seeds x K={COMMON['K']} boxes of "
          f"{cfgs[0].n_particles} particles, T={cfgs[0].T_total:.0f}")
    tolerances()
    t0 = time.time()
    recs = wca.simulate_batch(cfgs, SEEDS, [SHUS], batch_seed=20260821,
                              device=device, progress=10_000)
    wall = time.time() - t0
    print(f"wall {wall:.0f}s "
          f"({COMMON['n_steps']*len(cfgs)*COMMON['K']/wall/1e6:.1f}M box-steps/s)")
    os.makedirs(OUT, exist_ok=True)
    for rec in recs:
        arrays = {k: rec[k] for k in
                  ("time", "pmf_t", "marginal_t", "x_grid", "F_ref", "l2_f_t",
                   "kl_u_t", "tv_u_t", "ess_anc_t", "wmax_t", "n_anc_t",
                   "dep_ref_l2_t", "dep_self_l2_t", "P_regions")}
        meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
                "config": rec["config"], "method": rec["method"], "seed": rec["seed"],
                "cell": cell_name, "stage": "stage4_wca_screen"}
        save_run(os.path.join(OUT, f"{cell_name}_seed{rec['seed']}"), arrays, meta)
    print(f"records -> {OUT}/{cell_name}_seed*.npz")


def analyze():
    import glob
    noise95, D_tols, floors = tolerances()
    recs, names = [], []
    for f in sorted(glob.glob(f"{OUT}/*_seed*.npz")):
        b = os.path.basename(f)[:-4]
        name, sd = b.rsplit("_seed", 1)
        with np.load(f) as z:
            recs.append({k: z[k] for k in z.files})
        recs[-1]["seed"] = int(sd)
        names.append(name)
    T = float(recs[0]["time"][-1])
    summary = {"noise95": noise95, "common": COMMON, "floors": floors,
               "D_tols": D_tols, "cells": {}}
    print(f"\n{'cell':<7s} {'T_hit/T':>9s} {'T_est/T':>9s} {'gap/T':>7s} "
          f"{'D_T':>8s} {'e_F(T)':>8s}  classification")
    for name in CELLS:
        rows = []
        for i, rec in enumerate(recs):
            if names[i] != name:
                continue
            t = rec["time"]
            th = hit_time(rec["P_regions"][:, 2], t, hold_frac=0.05)
            te = establishment_time_median(rec["kl_u_t"], t, D_tols[name],
                                           hold_frac=0.10)
            rows.append(dict(
                seed=int(rec["seed"]), T_hit=th, T_est=te,
                hit_frac=th / T if np.isfinite(th) else np.nan,
                est_frac=te / T if np.isfinite(te) else np.nan,
                D_T=float(rec["kl_u_t"][-1]), eT=float(rec["l2_f_t"][-1])))
        cls = classify(rows)
        med = lambda k: float(np.nanmedian([r[k] for r in rows]))
        gap = med("est_frac") - med("hit_frac")
        n_c = sum(not np.isfinite(r["est_frac"]) for r in rows)
        print(f"{name:<7s} {med('hit_frac'):>9.3f} {med('est_frac'):>8.3f}({n_c}c) "
              f"{gap:>7.3f} {med('D_T'):>8.4f} {med('eT'):>8.4f}  {cls}")
        summary["cells"][name] = {"config": CELLS[name], "rows": rows,
                                  "classification": cls,
                                  "median": {k: med(k) for k in
                                             ("hit_frac", "est_frac", "D_T", "eT")}}
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
        kl = np.stack([r["kl_u_t"] for r in rs])
        axes[0, 0].plot(t, np.median(kl, 0), color=cmap[c], label=c)
        lo, hi = np.quantile(kl, [0.25, 0.75], axis=0)
        axes[0, 0].fill_between(t, lo, hi, color=cmap[c], alpha=0.15, lw=0)
        axes[0, 0].axhline(D_tols[c], color=cmap[c], ls="--", lw=0.8, alpha=0.7)
        occ = np.stack([r["P_regions"][:, 2] for r in rs])
        axes[0, 1].plot(t, np.median(occ, 0), color=cmap[c])
        if c == "b1h2":
            eF = np.stack([r["l2_f_t"] for r in rs])
            axes[1, 0].plot(t, np.median(eF, 0), color=cmap[c])
            lo, hi = np.quantile(eF, [0.25, 0.75], axis=0)
            axes[1, 0].fill_between(t, lo, hi, color=cmap[c], alpha=0.15, lw=0)
        dep = np.stack([r["dep_self_l2_t"] for r in rs])
        axes[1, 1].plot(t, np.median(dep, 0), color=cmap[c])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"KL$(\hat p_t\|u)$ (dashed: per-cell $D_{\rm tol}$)")
    axes[0, 0].legend()
    axes[0, 1].set_ylabel(r"stretched occupancy ($\xi > 0.75$)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel(r"$e_F(t)$ vs hp_v3 (b1h2 only)")
    axes[1, 1].set_ylabel(r"$\|d_n - r_n\|$ (deposition self-feed)")
    for ax in axes[1]:
        ax.set_xlabel("t")
    fig.suptitle("Stage 4: WCA plain-SHUS screen (8 seeds per cell; no FR)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stage4_overview.png"), dpi=130)
    print(f"figure -> {OUT}/stage4_overview.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELLS), help="run one cell's 8 seeds")
    ap.add_argument("--analyze", action="store_true", help="gate analysis from records")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    elif a.cell:
        run_cell(a.cell)
    else:
        ap.error("pass --cell NAME (per-GPU) or --analyze")
