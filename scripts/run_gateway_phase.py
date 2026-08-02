#!/usr/bin/env python
"""ABF-ONLY screen of the entropic-gateway (s, r) plane.

This runs **no Fisher-Rao arm of any kind**.  Its whole purpose is to produce the regime
map -- which (s, r) cells are ABF-sufficient, which are establishment-limited, which are
discovery-limited -- *before* any mFR arm is run anywhere, so that the cell the mFR
comparison is anchored at is chosen by a rule fixed in advance rather than by looking at
which cell mFR happens to win.

The entire 4x4 plane is reported, including the cells where mFR could not possibly help.
Reporting only the favourable corner is what makes a phase diagram indistinguishable from
parameter cherry-picking.

Design (frozen before the first production run; sized by scripts/calibrate_gateway.py)
--------------------------------------------------------------------------------------
    s    in {0.10, 0.15, 0.20, 0.30}   gateway width
    r    in {4, 8, 16, 32}             gateway severity omega_in / omega_out
    beta in {2, 4, 8, 16}              transport speed, at FIXED landscape (see below)
    beta*H = 8 kT held fixed           so H = 8/beta and the total barrier is 8 + log r kT
    N = 2048 walkers, 16 seeds, T = 40
    reference: ANALYTIC (no reference simulation, so no reference error to confound)
    init: 'left' (all walkers in the left basin) -- the headline arm
          'one_right' (one walker seeded in the right basin) -- secondary control in which
          discovery is free, so any acceleration measures establishment, not first passage

Why beta is on the map, and what it is not
-------------------------------------------
The sizing scan found that ``(s, r)`` modulates the regime only weakly, while the ratio of
transport time to run time moves it across every regime.  Reporting a single ``(s, r)``
slice would therefore have hidden the axis that actually does the work, so the whole
``(s, r, beta)`` map is run and reported.

Holding ``beta*H`` fixed makes the *dimensionless* landscape identical in every cell:
``beta F(x) = beta H (x^2-1)^2 + log omega(x)`` has no residual beta dependence.  What beta
changes is only how fast the walkers move through that landscape, at an unchanged sampling
budget of ``N * n_steps`` force evaluations.  This is deliberately *not* the forbidden
"shorten the run until a deficit appears" move, and the distinction is worth stating
precisely: shortening the run cuts exploration **and** compute together, whereas this axis
cuts exploration at **fixed** compute.  A regime that is defined by a ratio of timescales
can only be reached by moving that ratio; what would make it cherry-picking is choosing the
cell after seeing where mFR wins, which is what freezing the classification prevents.

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python -u scripts/run_gateway_phase.py
    CUDA_VISIBLE_DEVICES=2 python -u scripts/run_gateway_phase.py --pilot   # corners only
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import gateway_core as gw  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_ROOT = os.path.join(ROOT, "results", "gateway_phase")

S_VALUES = (0.10, 0.15, 0.20, 0.30)
R_VALUES = (4.0, 8.0, 16.0, 32.0)
BETA_VALUES = (2.0, 4.0, 8.0, 16.0)
BETA_H_KT = 8.0          # energetic barrier in kT, held fixed across the whole map
INITS = ("left", "one_right")


def build_rows(s_values, r_values, beta_values, seeds, inits, **cfg_kw):
    rows = []
    for init in inits:
        for beta in beta_values:
            for s in s_values:
                for r in r_values:
                    cfg = gw.GatewayConfig(beta=beta, H=BETA_H_KT / beta, s=s, r=r,
                                           init=init, **cfg_kw)
                    for sd in seeds:
                        rows.append((cfg, sd))
    return rows


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--dt", type=float, default=4e-4)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--n-walkers", type=int, default=2048)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512, help="max (config,seed) rows per batch")
    ap.add_argument("--pilot", action="store_true",
                    help="corners of the plane, few seeds, short: for sizing only")
    ap.add_argument("--tag", default="production")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # The policy is one GPU at a time, pinned explicitly; assert it rather than trust it.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, (
            f"exactly one GPU must be visible, saw {torch.cuda.device_count()} "
            f"(CUDA_VISIBLE_DEVICES={cvd!r})")

    s_vals, r_vals, b_vals = S_VALUES, R_VALUES, BETA_VALUES
    seeds = list(range(a.seeds))
    inits = list(INITS)
    if a.pilot:
        s_vals, r_vals, b_vals = (0.10, 0.30), (4.0, 32.0), (8.0,)
        seeds = list(range(4))
        inits = ["left"]

    rows = build_rows(s_vals, r_vals, b_vals, seeds, inits,
                      omega_out=1.0, N=a.n_walkers,
                      dt=a.dt, n_steps=a.steps, save_every=a.save_every)
    out_dir = a.out or os.path.join(OUT_ROOT, "pilot" if a.pilot else a.tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"gateway ABF-only screen: {len(s_vals)}x{len(r_vals)}x{len(b_vals)} cells "
          f"x {len(seeds)} seeds x {len(inits)} init arms = {len(rows)} runs")
    print(f"  T = {a.steps * a.dt:g}   dt = {a.dt:g}   N = {a.n_walkers}   "
          f"device = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
          f"(CUDA_VISIBLE_DEVICES={cvd!r})")
    # Explicit-Euler stability of the transverse OU channel: y <- y(1 - omega^2 dt) + noise
    # needs |1 - omega^2 dt| < 1.  A single dt across the whole plane keeps the comparison
    # free of an integrator confound, so the worst cell is what sets it.
    w2dt = (max(r_vals) ** 2) * a.dt
    print(f"  worst-cell omega_in^2 dt = {w2dt:.3f}  (must be < 2 for stability, "
          f"<= 1 preferred)")
    assert w2dt < 2.0, f"dt={a.dt} is unstable at r={max(r_vals)}"

    t_start = time.time()
    recs_all = []
    for i in range(0, len(rows), a.chunk):
        chunk = rows[i:i + a.chunk]
        spec = gw.BatchSpec(configs=[c for c, _ in chunk], seeds=[s for _, s in chunk],
                            methods=[gw.ABF], batch_seed=10_000 + i)
        t0 = time.time()
        recs = gw.simulate_batch(spec)
        dtb = time.time() - t0
        recs_all.extend(recs)
        print(f"  chunk {i // a.chunk + 1}/{-(-len(rows) // a.chunk)}: "
              f"{len(chunk)} rows in {dtb:.1f}s ({1e3 * dtb / a.steps:.2f} ms/step)",
              flush=True)

    # ------------------------------------------------------------------ save
    keys = ("t", "P_regions", "Q_regions", "l2_f_t", "l2_fp_t", "ess_t", "wmax_t",
            "x_grid", "F_hat", "Fp_hat", "F_ref", "Fp_ref")
    scalars = [k for k in recs_all[0] if k not in keys and k != "config"]
    npz = {}
    for k in keys:
        npz[k] = np.stack([r[k] for r in recs_all])
    for k in scalars:
        npz[k] = np.array([r[k] for r in recs_all])
    npz["config_json"] = np.array([json.dumps(r["config"], sort_keys=True)
                                   for r in recs_all])
    path = os.path.join(out_dir, "raw.npz")
    np.savez_compressed(path, **npz)

    prov = dict(
        script=os.path.basename(__file__), git_rev=git_rev(), host=socket.gethostname(),
        cuda_visible_devices=cvd,
        device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        torch=torch.__version__, python=sys.version.split()[0],
        s_values=list(s_vals), r_values=list(r_vals), beta_values=list(b_vals),
        beta_H_kT=BETA_H_KT, seeds=seeds, inits=inits,
        n_walkers=a.n_walkers, dt=a.dt, n_steps=a.steps, save_every=a.save_every,
        T_total=a.steps * a.dt, methods=["abf"], reference="analytic",
        thresholds=dict(discovery_frac=gw.DISCOVERY_FRAC,
                        est_gap_sufficient=gw.EST_GAP_SUFFICIENT,
                        est_gap_limited=gw.EST_GAP_LIMITED,
                        below_half_frac=gw.BELOW_HALF_FRAC,
                        est_band=list(gw.EST_BAND), hold_frac=gw.HOLD_FRAC,
                        discovery_seed_frac=gw.DISCOVERY_SEED_FRAC,
                        x_basin=gw.X_BASIN),
        wall_seconds=time.time() - t_start,
        n_runs=len(recs_all))
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"\nwrote {path}  ({len(recs_all)} runs, {time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
