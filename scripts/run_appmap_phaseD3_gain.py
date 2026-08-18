"""Application-map Phase D3a: t_mid adaptation-gain curve (NO FR).

Design frozen in docs/PREREGISTRATION_APPLICATION_MAP.md (commit fde0c9c): new
gains {2, 3, 4} on seeds 400-415 with batch_seed 20260828 — exactly noise-paired
with the stored D2 rows (the engine noise stream is method-independent), so
g = 1.0 ("shus") and g = 1.5 ("shus_g1.5") are read from the D2 records.

Frozen g* rule: qualifying = median paired e_F(T) ratio vs g = 1 <= 1.05;
g* = lowest median paired I_F among qualifying; within 2 points -> smaller g.
Extension {6, 8} only if g = 4 is argmin and beats g = 3 by > 2 points.

Usage: python scripts/run_appmap_phaseD3_gain.py [--gains "6.0,8.0"]
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from abpfr.diagnostics import (establishment_time_median, first_persistent,
                               kde_noise_floor2)
from abpfr.io import save_run
from abpfr.metrics import paired_bootstrap_ci
from abpfr.systems import torus2d as t2
from abpfr.systems.gateway import Method

CELL = dict(beta=4.0, H1=1.5, H2=1.5, Hc=0.5)          # t_mid, frozen
COMMON = dict(K=1024, dt=1e-3, n_steps=200_000, block=20, eps_bw=0.06,
              eta_bw=0.25, n_saves=400, profile_every=8, ess_window_steps=4000)
SEEDS = list(range(400, 416))
BATCH_SEED = 20260828                                  # = D2 (noise pairing)
D2_DIR = "results/appmap_phaseD2_torus_fr"
OUT = "results/appmap_phaseD3_gain"


def d_tol():
    noise95 = float(np.quantile(
        kde_noise_floor2(COMMON["K"], COMMON["eta_bw"], t2.GRID2, n_rep=256,
                         seed=777), 0.95))
    fp = t2.analytic_floors(t2.Torus2DConfig(**CELL, **COMMON))
    return 1.5 * (fp["kl_star"] + noise95)


def row_from_arrays(z, seed, tol):
    t, T = z["time"], float(z["time"][-1])
    l2 = z["l2_f_t"]
    occupied_all = (z["P_regions"] > 0).all(axis=1)
    th = first_persistent(occupied_all, t, hold_frac=0.05)
    te = establishment_time_median(z["kl_u_t"], t, tol, hold_frac=0.10)
    return dict(seed=int(seed), T_hit=th, T_est=te,
                I_F=float(np.trapezoid(l2, t)), eT=float(l2[-1]))


def load_stored(dirname, arm_name, tol):
    rows = {}
    for f in sorted(glob.glob(f"{dirname}/{arm_name}_seed*.npz")):
        sd = int(os.path.basename(f)[:-4].rsplit("seed", 1)[1])
        if sd not in SEEDS:
            continue
        with np.load(f) as z:
            rows[sd] = row_from_arrays(z, sd, tol)
    return rows if len(rows) == len(SEEDS) else None


def main(gains):
    device = t2.DEVICE
    tol = d_tol()
    print(f"D_tol = {tol:.4f}")
    cfgs = [t2.Torus2DConfig(**CELL, **COMMON) for _ in SEEDS]
    arms = [Method(f"shus_g{g:g}", g_shus=g) for g in gains]
    print(f"Phase D3a: t_mid, {len(SEEDS)} seeds x gains {list(gains)} "
          f"(noise-paired with D2 via batch_seed {BATCH_SEED})")
    t0 = time.time()
    recs = t2.simulate_batch(cfgs, SEEDS, arms, batch_seed=BATCH_SEED,
                             device=device, progress=50_000)
    print(f"wall {time.time()-t0:.0f}s")

    os.makedirs(OUT, exist_ok=True)
    for rec in recs:
        arrays = {k: rec[k] for k in
                  ("time", "profile_time", "pmf_t", "marginal_t", "x1_grid",
                   "x2_grid", "F_ref", "l2_f_t", "kl_u_t", "tv_u_t", "ess_anc_t",
                   "wmax_t", "n_anc_t", "dep_ref_l2_t", "dep_self_l2_t",
                   "P_regions")}
        meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
                "config": rec["config"], "method": rec["method"], "seed": rec["seed"],
                "cell": "t_mid", "stage": "appmap_phaseD3_gain"}
        save_run(os.path.join(OUT, f"{rec['method']['name']}_seed{rec['seed']}"),
                 arrays, meta)

    # assemble ALL gains: D2 stored baselines + any prior D3 arms + fresh
    by = {1.0: load_stored(D2_DIR, "shus", tol),
          1.5: load_stored(D2_DIR, "shus_g1.5", tol)}
    for f in glob.glob(f"{OUT}/shus_g*_seed{SEEDS[0]}.npz"):
        g = float(os.path.basename(f).split("_seed")[0].replace("shus_g", ""))
        by[g] = load_stored(OUT, f"shus_g{g:g}", tol)
    assert all(v is not None for v in by.values()), "missing stored baseline rows"
    base = by[1.0]

    summary = {"cell": CELL, "seeds": SEEDS, "batch_seed": BATCH_SEED,
               "D_tol": tol, "gains": sorted(by), "rows": {}, "paired": {}}
    print(f"\n{'gain':>6s} {'T_hit':>7s} {'T_est':>7s} {'I_F':>7s} {'e_F(T)':>8s} "
          f"{'dI_F%':>20s} {'eT ratio':>9s}")
    qualified = {}
    for g in sorted(by):
        rows = by[g]
        dI = np.array([(rows[sd]["I_F"] - base[sd]["I_F"]) / base[sd]["I_F"]
                       for sd in SEEDS])
        rT = np.array([rows[sd]["eT"] / base[sd]["eT"] for sd in SEEDS])
        m, lo, hi = paired_bootstrap_ci(dI)
        m_rT = float(np.median(rT))
        med = lambda k: float(np.nanmedian([rows[sd][k] for sd in SEEDS]))
        if m_rT <= 1.05:
            qualified[g] = m
        print(f"{g:>6g} {med('T_hit'):>7.1f} {med('T_est'):>7.1f} "
              f"{med('I_F'):>7.2f} {med('eT'):>8.4f} "
              f"{100*m:>7.1f} [{100*lo:>5.1f},{100*hi:>5.1f}] {m_rT:>9.3f}")
        summary["rows"][f"{g:g}"] = {"per_seed": list(rows.values()),
                                     "median": {k: med(k) for k in
                                                ("T_hit", "T_est", "I_F", "eT")}}
        summary["paired"][f"{g:g}"] = dict(dI_F=m, dI_F_ci=[lo, hi], eT_ratio=m_rT)

    best = min(qualified.values())
    cands = [g for g, v in qualified.items() if v <= best + 0.02]
    g_star = min(cands)                                  # ties -> smaller g (frozen)
    summary["g_star"] = g_star
    rows = by[g_star]
    th = np.array([rows[sd]["T_hit"] for sd in SEEDS])
    te = np.array([rows[sd]["T_est"] for sd in SEEDS])
    t_on = float(np.ceil(np.nanquantile(th, 0.9)))
    t_off = float(round(t_on + 0.25 * (np.nanquantile(te, 0.5) - t_on)))
    summary["gstar_quantiles"] = dict(Q90_T_hit=float(np.nanquantile(th, 0.9)),
                                      Q50_T_est=float(np.nanquantile(te, 0.5)))
    summary["d3b_window"] = [t_on, t_off]
    print(f"\nqualified: {sorted(qualified)}")
    print(f"g* (frozen rule) = {g_star:g}  (median paired dI_F = "
          f"{100*qualified[g_star]:.1f}%)")
    print(f"D3b window from g* rows: t_on={t_on:g}, t_off={t_off:g} "
          f"(Q90 T_hit={summary['gstar_quantiles']['Q90_T_hit']:.1f}, "
          f"Q50 T_est={summary['gstar_quantiles']['Q50_T_est']:.1f})")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"summary -> {OUT}/summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", type=str, default="2.0,3.0,4.0")
    a = ap.parse_args()
    main(tuple(float(x) for x in a.gains.split(",")))
