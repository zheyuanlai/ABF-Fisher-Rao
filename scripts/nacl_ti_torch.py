"""NaCl constrained-TI reference on the batched torch engine (SPEC_nacl_water.md §5).

Adapted from ``methane_ti_torch.py`` (same stopping rule, same batching, same engine-order
constraint).  NaCl-specific: 4 hydration families per r-point --

    f0 CIP-derived   f1 SSIP-derived   f2 dissociated-derived
    f3 locally-equilibrated = f2's starts + an extra 100 ps constrained pre-equilibration

-- and the family-spread retirement criterion uses the MAX over family-pair differences, since
there are four families rather than two.  Family disagreement is the Gate 0 signal, reported
per point, never averaged away.

Requires: results/nacl/box (frozen L), results/nacl/baths (cached starts, torch-free process),
results/nacl/stage1/dynamics_gate.json (the chosen dt).

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/nacl_ti_torch.py --out results/nacl/ti_torch
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from methane.cv import PeriodicDistanceCV                        # noqa: E402
from methane.dynamics import (BAOAB, CompositeConstraints, PairConstraint,   # noqa: E402
                              RigidWaterConstraints)
from nacl import system as nsys                                  # noqa: E402
from nacl.nonbonded import NaClNonbonded                         # noqa: E402
from nacl.observables import HydrationDescriptors                # noqa: E402

CHECKPOINTS_PS = (50.0, 100.0, 250.0)
N_FAMILIES = 4
EXTRA_EQUIL_F3_PS = 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/nacl/ti_torch")
    ap.add_argument("--baths", default="results/nacl/baths")
    ap.add_argument("--builds", type=int, default=3)
    ap.add_argument("--replicas-per-family", type=int, default=3)
    ap.add_argument("--equil-ps", type=float, default=50.0)
    ap.add_argument("--sample-ps", type=float, default=0.1)
    ap.add_argument("--tol-frac", type=float, default=0.08)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--max-batch", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    box = json.load(open(nsys.REPO / "results/nacl/box/box_manifest.json"))
    L = float(box["L_nm"])
    r_hi = float(box["finite_size_gate"]["R_hi_nm"])
    r_grid = np.round(np.arange(nsys.R_LO_NM, r_hi + 1e-9, 0.02), 4)
    gate = json.load(open(nsys.REPO / "results/nacl/stage1/dynamics_gate.json"))
    dt = float(gate["dt_chosen_ps"])
    beta = nsys.beta_per_kJ()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bz = np.load(os.path.join(args.baths, "baths.npz"))
    bman = json.load(open(os.path.join(args.baths, "manifest.json")))
    if abs(float(bman["box_L_nm"]) - L) > 1e-9:
        raise SystemExit(f"baths built at L = {bman['box_L_nm']}, box is {L}")

    ff = NaClNonbonded(L, device=dev, dtype=torch.float32)
    ff.pair.energy_forces = torch.compile(ff.pair.energy_forces, dynamic=False)
    ff.recip.energy = torch.compile(ff.recip.energy, dynamic=False)
    hyd = HydrationDescriptors(ff.params["waters"], L, device=dev)
    cv = PeriodicDistanceCV(0, 1, L)
    print(f"[engine] torch on {dev}, float32, compiled, dt = {dt} ps", flush=True)

    # ---- replica set: (r, build, family, replica) ------------------------------------------
    recs, starts = [], []
    for r_nm in r_grid:
        for b in range(args.builds):
            for fam in range(N_FAMILIES):
                src_fam = 2 if fam == 3 else fam
                key = f"start_b{b}_f{src_fam}_r{r_nm:.4f}"
                if key not in bz:
                    raise SystemExit(f"cached starts lack {key}; rerun nacl_baths.py --per-r")
                for k in range(args.replicas_per_family):
                    recs.append((float(r_nm), b, fam, k))
                    starts.append(bz[key].astype(np.float64))
    recs = np.asarray(recs)
    starts = np.asarray(starts, dtype=np.float32)
    n_traj = len(recs)
    print(f"[plan] {len(r_grid)} r x {args.builds} builds x {N_FAMILIES} fam x "
          f"{args.replicas_per_family} rep = {n_traj} trajectories", flush=True)

    active = np.ones(n_traj, dtype=bool)
    fsum = np.zeros(n_traj); fcnt = np.zeros(n_traj)
    ysum = np.zeros((n_traj, 3)); ycnt = np.zeros(n_traj)
    retired_at = np.full(len(r_grid), np.nan)
    x_state = starts.copy()
    t_start = time.time()

    def run_block(idx, ps, equilibrate, seed_salt):
        nonlocal x_state
        n_steps = int(round(ps / dt))
        sample_every = max(1, int(round(args.sample_ps / dt)))
        y_every = 10 * sample_every                 # Y is cheap but not free; 1 ps cadence
        # even split: 2196 into 2048 + 148 would run the tail chunk at ~7 % occupancy
        n_chunks = max(1, int(np.ceil(len(idx) / args.max_batch)))
        step_size = int(np.ceil(len(idx) / n_chunks))
        for lo in range(0, len(idx), step_size):
            sub = idx[lo:lo + step_size]
            x = torch.tensor(x_state[sub], device=dev, dtype=torch.float32)
            cons = CompositeConstraints(
                [RigidWaterConstraints(ff.params["waters"], nsys.rigid_water_lengths(),
                                       ff.params["mass"], device=dev, dtype=torch.float32),
                 PairConstraint(0, 1, recs[sub, 0], ff.params["mass"],
                                device=dev, dtype=torch.float32)],
                atom_sets=[ff.params["waters"], [0, 1]])
            integ = BAOAB(lambda q: ff.energy_forces(q, chunk=args.chunk), ff.params["mass"],
                          cons, dt, nsys.TEMPERATURE_K, nsys.GAMMA_PS,
                          device=dev, dtype=torch.float32)
            # replica index enters the seed so same-start replicas carry independent noise
            gen = torch.Generator(device=dev).manual_seed(
                int(7e6 + seed_salt * 10007 + lo))
            cons.apply_positions(x, x.clone())
            v = integ.maxwell_velocities(x, generator=gen)
            _, f = ff.energy_forces(x, chunk=args.chunk)
            for s in range(n_steps):
                _, f = integ.step(x, v, f, generator=gen)
                if not equilibrate and (s + 1) % sample_every == 0:
                    fl, _, _ = cv.local_mean_force(x, f, beta)
                    fsum[sub] += fl.double().cpu().numpy()
                    fcnt[sub] += 1
                if not equilibrate and (s + 1) % y_every == 0:
                    ysum[sub] += hyd.Y(x).double().cpu().numpy()
                    ycnt[sub] += 1
            x_state[sub] = x.cpu().numpy()
            del x, v, f, integ, cons
            torch.cuda.empty_cache()

    # ---- equilibration: everyone 50 ps; f3 gets its extra 100 ps first ---------------------
    idx_f3 = np.flatnonzero(recs[:, 2] == 3)
    print(f"[equil] f3 extra {EXTRA_EQUIL_F3_PS} ps on {idx_f3.size} trajectories", flush=True)
    run_block(idx_f3, EXTRA_EQUIL_F3_PS, equilibrate=True, seed_salt=1)
    idx_all = np.flatnonzero(active)
    print(f"[equil] {args.equil_ps} ps on {idx_all.size} trajectories", flush=True)
    t0 = time.time()
    run_block(idx_all, args.equil_ps, equilibrate=True, seed_salt=2)
    print(f"[equil] done in {(time.time()-t0)/60:.1f} min", flush=True)

    done_ps = 0.0
    for cp in CHECKPOINTS_PS:
        block = cp - done_ps
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        print(f"\n[prod] -> {cp:.0f} ps  ({block:.0f} ps on {idx.size}) ...", flush=True)
        t0 = time.time()
        run_block(idx, block, equilibrate=False, seed_salt=int(cp))
        done_ps = cp
        fbar = np.where(fcnt > 0, fsum / np.maximum(fcnt, 1), np.nan)
        denom = np.nanmean(np.abs(fbar))

        print(f"{'r':>7} {'fbar':>9} {'build sp':>9} {'fam sp':>9} {'state':>10}")
        for ri, r_nm in enumerate(r_grid):
            m = recs[:, 0] == r_nm
            if not np.isnan(retired_at[ri]):
                continue
            bmeans = [fbar[m & (recs[:, 1] == b)].mean() for b in range(args.builds)]
            bsp = float(np.max(bmeans) - np.min(bmeans))
            fmeans = [fbar[m & (recs[:, 2] == f_)].mean() for f_ in range(N_FAMILIES)]
            fsp = float(np.max(fmeans) - np.min(fmeans))
            ok = (bsp <= args.tol_frac * denom) and (fsp <= args.tol_frac * denom)
            if ok:
                retired_at[ri] = cp
                active[m] = False
            print(f"{r_nm:7.3f} {fbar[m].mean():9.2f} {bsp:9.2f} {fsp:9.2f} "
                  f"{'RETIRED' if ok else 'extend':>10}", flush=True)
        np.savez_compressed(os.path.join(args.out, f"checkpoint_{cp:.0f}ps.npz"),
                            recs=recs, fbar=fbar, fcnt=fcnt, ysum=ysum, ycnt=ycnt,
                            retired_at=retired_at, active=active)
        print(f"[prod] checkpoint {cp:.0f} ps in {(time.time()-t0)/60:.1f} min; "
              f"{int(active.sum())} active", flush=True)

    fbar = np.where(fcnt > 0, fsum / np.maximum(fcnt, 1), np.nan)
    np.savez_compressed(os.path.join(args.out, "ti_final.npz"),
                        recs=recs, fbar=fbar, fcnt=fcnt, ysum=ysum, ycnt=ycnt,
                        retired_at=retired_at)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(dict(stage="nacl_ti_torch", engine="batched torch float32 compiled",
                       dt_ps=dt, r_grid_nm=r_grid.tolist(), builds=args.builds,
                       replicas_per_family=args.replicas_per_family, families=N_FAMILIES,
                       equil_ps=args.equil_ps, extra_equil_f3_ps=EXTRA_EQUIL_F3_PS,
                       checkpoints_ps=list(CHECKPOINTS_PS), tol_frac=args.tol_frac,
                       retired_at_ps=retired_at.tolist(), box_L_nm=L,
                       gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                       wall_hours=(time.time() - t_start) / 3600.0,
                       git_commit=subprocess.run(["git", "rev-parse", "HEAD"],
                                                 capture_output=True, text=True).stdout.strip()),
                  fh, indent=2)
    print(f"\n[done] {(time.time()-t_start)/3600:.2f} h -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
