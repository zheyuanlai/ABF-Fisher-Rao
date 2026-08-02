#!/usr/bin/env python
"""Sizing scan for the entropic-gateway screen: find (H, T) that can express every regime.

Run BEFORE the production phase diagram.  The plane is only informative if the run length
and barrier height are such that at least one cell is establishment-limited and at least
one is not; a design in which every cell lands in the same regime measures nothing about
(s, r).  This script sweeps the energetic barrier ``beta*H`` at the corners of the (s, r)
plane and prints the discovery / establishment split, so the production values are chosen
by measurement instead of by hope.

Sizing is not the experiment.  Nothing here feeds a verdict, no Fisher-Rao arm runs, and
the classification thresholds are the frozen ones from ``gateway_core``.

    CUDA_VISIBLE_DEVICES=2 python -u scripts/calibrate_gateway.py --steps 100000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import gateway_core as gw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--dt", type=float, default=4e-4)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--n-walkers", type=int, default=2048)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--H", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0])
    ap.add_argument("--s", type=float, nargs="+", default=[0.10, 0.30])
    ap.add_argument("--r", type=float, nargs="+", default=[4.0, 32.0])
    a = ap.parse_args()

    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU"

    cfgs, seeds = [], []
    for H in a.H:
        for s in a.s:
            for r in a.r:
                for sd in range(a.seeds):
                    cfgs.append(gw.GatewayConfig(beta=1.0, H=H, s=s, r=r, N=a.n_walkers,
                                                 dt=a.dt, n_steps=a.steps,
                                                 save_every=a.save_every, init="left"))
                    seeds.append(sd)
    print(f"{len(cfgs)} rows, T = {a.steps * a.dt:g}, N = {a.n_walkers}", flush=True)
    t0 = time.time()
    recs = gw.simulate_batch(gw.BatchSpec(configs=cfgs, seeds=seeds, methods=[gw.ABF],
                                          batch_seed=777))
    print(f"done in {time.time() - t0:.0f}s\n")

    print(f"{'betaH':>6s} {'s':>5s} {'r':>4s} {'barrier':>8s} {'T_hit/T':>8s} "
          f"{'T_est/T':>8s} {'gap':>7s} {'below½':>7s} {'P+':>7s} {'Q+':>7s} "
          f"{'L2(F)':>7s}  regime")
    by_cell = {}
    for rec in recs:
        by_cell.setdefault((rec["config"]["H"], rec["s"], rec["r_ratio"]), []).append(rec)
    for key in sorted(by_cell):
        rows = by_cell[key]
        H, s, r = key
        reg = gw.classify(rows)
        def med(k):
            v = np.array([x[k] for x in rows], dtype=float)
            return float(np.nanmedian(np.where(np.isfinite(v), v, np.nan)))
        print(f"{H:6.1f} {s:5.2f} {r:4.0f} {rows[0]['barrier_kT']:8.2f} "
              f"{med('T_hit_frac'):8.3f} {med('T_est_frac'):8.3f} {med('est_gap_frac'):7.3f} "
              f"{med('below_half_frac'):7.3f} {med('final_occupancy'):7.4f} "
              f"{med('final_target'):7.4f} {med('final_l2_f'):7.3f}  {reg}")


if __name__ == "__main__":
    main()
