"""Application-map Phase D4: honest classification of t_cold at T = 800 (NO FR).

Design frozen in docs/PREREGISTRATION_APPLICATION_MAP.md (commit fde0c9c): seeds
0-7, arms shus + shus_g1.5 (secondary), n_steps = 800_000. Question: with T long
enough to resolve T_est, is t_cold's establishment deficit Type A
(population-oscillatory -> a reallocation candidate) or Type B
(adaptation-rate-limited -> the g arm removes it)?

Usage: python scripts/run_appmap_phaseD4_tcold.py
"""
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

CELL = dict(beta=8.0, H1=1.0, H2=1.0, Hc=0.25)         # t_cold, frozen
COMMON = dict(K=1024, dt=1e-3, n_steps=800_000, block=20, eps_bw=0.06,
              eta_bw=0.25, n_saves=400, profile_every=8, ess_window_steps=4000)
SEEDS = list(range(8))
BATCH_SEED = 20260829
OUT = "results/appmap_phaseD4_tcold"


def ring_metrics(rec, t_hit):
    """Descriptive Type-A indicators: basin-occupancy overshoot/crossings of the
    0.25 uniform target, and local maxima of a median-smoothed KL after T_hit."""
    t = rec["time"]
    P = rec["P_regions"]
    over, crossings = [], []
    for k in range(4):
        p = P[:, k]
        over.append(float(p.max()))
        m = np.isfinite(t_hit) & (t > (t_hit if np.isfinite(t_hit) else 0.0))
        s = np.sign(p[m] - 0.25)
        s = s[s != 0]
        crossings.append(int((np.diff(s) != 0).sum()) if s.size > 1 else 0)
    D = rec["kl_u_t"]
    w = 9
    Ds = np.array([np.median(D[max(0, i - w // 2): i + w // 2 + 1])
                   for i in range(len(D))])
    m = np.isfinite(t_hit) & (t > (t_hit if np.isfinite(t_hit) else 0.0))
    d = np.diff(Ds[m])
    d = d[np.abs(d) > 1e-3]
    s = np.sign(d)
    kl_flips = int((np.diff(s) != 0).sum()) if s.size > 1 else 0
    return dict(overshoot=over, crossings=crossings, kl_flips=kl_flips)


def main():
    device = t2.DEVICE
    noise95 = float(np.quantile(
        kde_noise_floor2(COMMON["K"], COMMON["eta_bw"], t2.GRID2, n_rep=256,
                         seed=777), 0.95))
    fp = t2.analytic_floors(t2.Torus2DConfig(**CELL, **COMMON))
    tol = 1.5 * (fp["kl_star"] + noise95)
    print(f"D_tol = {tol:.4f}")
    cfgs = [t2.Torus2DConfig(**CELL, **COMMON) for _ in SEEDS]
    T = cfgs[0].T_total
    arms = [Method("shus"), Method("shus_g1.5", g_shus=1.5)]
    print(f"Phase D4: t_cold, {len(SEEDS)} seeds x {len(arms)} arms, T={T:.0f}")
    t0 = time.time()
    recs = t2.simulate_batch(cfgs, SEEDS, arms, batch_seed=BATCH_SEED,
                             device=device, progress=100_000)
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
                "cell": "t_cold", "stage": "appmap_phaseD4_tcold"}
        save_run(os.path.join(OUT, f"{rec['method']['name']}_seed{rec['seed']}"),
                 arrays, meta)

    by = {}
    for rec in recs:
        by.setdefault(rec["method"]["name"], {})[rec["seed"]] = rec

    summary = {"cell": CELL, "common": COMMON, "batch_seed": BATCH_SEED,
               "D_tol": tol, "arms": {}}
    print(f"\n{'arm':<11s} {'T_hit/T':>9s} {'T_est/T':>10s} {'gap/T':>7s} "
          f"{'I_F':>8s} {'e_F(T)':>8s} {'klflips':>8s}")
    for name, rows_by_sd in by.items():
        rows = []
        for sd in SEEDS:
            rec = rows_by_sd[sd]
            t = rec["time"]
            occupied_all = (rec["P_regions"] > 0).all(axis=1)
            th = first_persistent(occupied_all, t, hold_frac=0.05)
            te = establishment_time_median(rec["kl_u_t"], t, tol, hold_frac=0.10)
            gap = (np.nan if not np.isfinite(th)
                   else (np.inf if not np.isfinite(te) else (te - th) / T))
            rows.append(dict(seed=sd, T_hit=th, T_est=te,
                             hit_frac=th / T if np.isfinite(th) else np.nan,
                             est_frac=te / T if np.isfinite(te) else np.nan,
                             gap_frac=gap, I_F=float(rec["int_l2_f"]),
                             eT=float(rec["l2_f_t"][-1]),
                             **ring_metrics(rec, th)))
        med = lambda k: float(np.nanmedian([r[k] for r in rows]))
        n_c = sum(not np.isfinite(r["est_frac"]) for r in rows)
        print(f"{name:<11s} {med('hit_frac'):>9.3f} {med('est_frac'):>9.3f}({n_c}c) "
              f"{float(np.nanmedian([r['gap_frac'] for r in rows])):>7.3f} "
              f"{med('I_F'):>8.2f} {med('eT'):>8.4f} {med('kl_flips'):>8.1f}")
        summary["arms"][name] = {"rows": rows,
                                 "median": {k: med(k) for k in
                                            ("hit_frac", "est_frac", "I_F", "eT",
                                             "kl_flips")}}
    dI = np.array([(by["shus_g1.5"][sd]["int_l2_f"] - by["shus"][sd]["int_l2_f"])
                   / by["shus"][sd]["int_l2_f"] for sd in SEEDS])
    m, lo, hi = paired_bootstrap_ci(dI)
    summary["gain_dI_F"] = [m, lo, hi]
    print(f"\nshus_g1.5 paired dI_F: {100*m:.1f}% [{100*lo:.1f},{100*hi:.1f}]")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"summary -> {OUT}/summary.json")


if __name__ == "__main__":
    main()
