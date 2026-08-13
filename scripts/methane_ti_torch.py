"""Constrained-TI reference on the **batched** engine, with the preregistered stopping rule.

Replaces the OpenMM driver (``methane_ti_reference.py``) for the production reference, on a
measurement rather than a preference.  On an idle H200 with the frozen 1538-site system:

    OpenMM, one context at a time     115 W,  ~70 % util,  465 ns/day
    batched torch, B = 512-2048       370 W,   100 % util,  ~750 ns/day

A single 1538-atom simulation cannot fill an H200; it is launch-bound, and process-level packing
does not fix it (CUDA contexts from separate processes time-slice rather than share SMs).  The
batched engine runs all 1392 TI replicas in one step, which is what the hardware wants.
OpenMM remains the **parity oracle** and is retained for the cross-validation of §"independence"
below, so Amendment 12.2's engine-independence argument is preserved where it does work.

Stopping rule (Amendment 12.2, preregistered and previously unimplemented)
--------------------------------------------------------------------------
The reference is *not* run to a fixed 200 ps everywhere.  Production advances in checkpoint
blocks at **50, 100, 200 ps**, and at each checkpoint an ``r``-point is **retired** when both

    build-to-build spread of fbar(r)   <=  tol_frac * mean|fbar|
    wet-dry family spread of fbar(r)   <=  tol_frac * mean|fbar|

Retired points stop consuming compute; only the unconverged ones are extended.  This is what the
amendment says ("extend only the points whose build spread or family disagreement remains
materially above the target tolerance") and it is strictly cheaper than the flat schedule.

**A retired point and a converged point are not the same claim.**  Points that never converge by
200 ps are reported as unconverged with their spreads, and a large wet-dry spread is not a failure
of the reference -- it is the Gate 0 signal.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/methane_ti_torch.py --out results/methane/ti_torch
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

import openmm as mm                                              # noqa: E402
import openmm.unit as u                                          # noqa: E402

from methane import system as msys                               # noqa: E402
from methane.cv import PeriodicDistanceCV                        # noqa: E402
from methane.dynamics import (BAOAB, CompositeConstraints, PairConstraint,   # noqa: E402
                              RigidWaterConstraints, water_molecules)
from methane.nonbonded import MethaneNonbonded                   # noqa: E402
from methane.observables import n_gap_batch                      # noqa: E402

R_GRID_NM = np.round(np.arange(0.34, 0.9001, 0.02), 4)
R_DRY_PREP_NM = 0.38
R_WET_PREP_NM = 0.80
CHECKPOINTS_PS = (50.0, 100.0, 200.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/methane/ti_torch")
    ap.add_argument("--box", default="results/methane/box")
    ap.add_argument("--baths", default="results/methane/baths")
    ap.add_argument("--builds", type=int, default=3)
    ap.add_argument("--replicas", type=int, default=16)
    ap.add_argument("--equil-ps", type=float, default=50.0)
    ap.add_argument("--sample-ps", type=float, default=0.1)
    ap.add_argument("--tol-frac", type=float, default=0.08,
                    help="retire an r-point when both spreads fall below this fraction of mean|f|")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--max-batch", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # The retirement rule is a JOINT criterion: build-to-build spread AND wet-dry family spread
    # must both fall below tolerance.  With a single build the build spread is 0 by construction,
    # so the criterion degenerates and EVERY r-point retires at the first checkpoint -- producing
    # a reference that looks converged because it was never given the chance to disagree with
    # itself.  This also forbids the obvious parallelisation (one build per process/GPU): the
    # stopping rule couples the builds, so the efficiency device is a correctness constraint on
    # how the work may be split.  Identified by the NaCl session in its own TI; latent here.
    if args.builds < 2:
        raise SystemExit(
            f"--builds {args.builds} degenerates the retirement rule: build-to-build spread is "
            "0 by construction with one build, so every r-point would retire at the first "
            "checkpoint. Use >= 2 builds, or split by r-point rather than by build.")

    man = json.load(open(os.path.join(args.box, "manifest.json")))
    L = float(man["box_L_nm"])
    pos_box = np.load(os.path.join(args.box, "box.npz"))["positions_nm"]
    beta = msys.beta_per_kJ()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mod = msys.build_modeller(r0_nm=0.55, seed=man["seed"])
    system = msys.build_system(mod.topology)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0) * u.nanometer,
                                        mm.Vec3(0, L, 0) * u.nanometer,
                                        mm.Vec3(0, 0, L) * u.nanometer)
    # Indices and parameters come out of the OpenMM System and need no CUDA; the torch engine is
    # constructed further down, only after every OpenMM context has been destroyed.
    params = msys.site_parameters(system, mod.topology)
    mi = params["methane_index"]
    oxy = np.flatnonzero((~params["is_methane"]) & (params["epsilon"] > 0))
    mols = water_molecules(mod.topology)

    print(f"[plan] {len(R_GRID_NM)} r x {args.replicas} replicas x {args.builds} builds "
          f"= {len(R_GRID_NM)*args.replicas*args.builds} trajectories", flush=True)
    print(f"[plan] checkpoints at {CHECKPOINTS_PS} ps, retire tol = {args.tol_frac}", flush=True)

    t_start = time.time()
    # Baths are built by scripts/methane_baths.py in a torch-free process and cached, because
    # importing torch disables OpenMM's CUDA platform in the same process (see that file).
    bz = np.load(os.path.join(args.baths, "baths.npz"))
    bman = json.load(open(os.path.join(args.baths, "manifest.json")))
    if abs(float(bman["box_L_nm"]) - L) > 1e-9:
        raise SystemExit(f"baths were built at L = {bman['box_L_nm']}, box is {L}")
    baths = {}
    for b in range(args.builds):
        for fam in (0, 1):
            key = f"b{b}_f{fam}"
            if key not in bz:
                raise SystemExit(f"cached baths lack {key}; rerun scripts/methane_baths.py "
                                 f"--builds {args.builds}")
            baths[(b, fam)] = bz[key].astype(np.float64)
    print(f"[baths] loaded {len(baths)} cached baths from {args.baths}", flush=True)
    for b in range(args.builds):
        print(f"  build {b}: dry n_gap = {n_gap_batch(baths[(b,0)], mi, oxy, L)[0]:.3f}   "
              f"wet n_gap = {n_gap_batch(baths[(b,1)], mi, oxy, L)[0]:.3f}", flush=True)

    # ---- torch CUDA is initialised only AFTER every OpenMM context is gone ------------------
    # Creating an OpenMM CUDA context on a device where torch has already initialised CUDA hangs
    # indefinitely: measured at >20 min pinned at 0 % utilisation on GPU 3, while the identical
    # OpenMM-only path finishes the same baths in seconds.  The two runtimes must not interleave
    # context creation on one device, so all OpenMM work is completed and released first.
    ff = MethaneNonbonded(system, mod.topology, L, device=dev, dtype=torch.float32)
    ff.pair.energy_forces = torch.compile(ff.pair.energy_forces, dynamic=False)
    ff.recip.energy = torch.compile(ff.recip.energy, dynamic=False)
    print(f"[engine] torch on {dev}, float32, compiled", flush=True)

    # ---- assemble the full replica set -----------------------------------------------------
    n_half = args.replicas // 2
    recs = []          # (r, build, family)
    starts = []
    for r_nm in R_GRID_NM:
        for b in range(args.builds):
            for k in range(args.replicas):
                fam = 1 if k >= n_half else 0
                key = f"start_b{b}_f{fam}_r{r_nm:.4f}"
                if key not in bz:
                    raise SystemExit(
                        f"cached starts lack {key}. Placing a methane at a new separation can "
                        "drop it on a water (forces ~1e10 kJ/mol/nm), which destroys the walker "
                        "on the first step and surfaces as a singular M-SHAKE matrix. Rerun "
                        "scripts/methane_baths.py --per-r to generate minimised starts.")
                recs.append((float(r_nm), b, fam))
                starts.append(bz[key].astype(np.float64))
    recs = np.asarray(recs)
    starts = np.asarray(starts, dtype=np.float32)
    print(f"[plan] {len(recs)} trajectories assembled", flush=True)

    active = np.ones(len(recs), dtype=bool)
    fsum = np.zeros(len(recs)); fcnt = np.zeros(len(recs))
    ngsum = np.zeros(len(recs)); ngcnt = np.zeros(len(recs))
    retired_at = np.full(len(R_GRID_NM), np.nan)
    x_state = starts.copy()

    def run_block(idx, ps, equilibrate):
        """Advance the trajectories ``idx`` by ``ps``; accumulate f_loc unless equilibrating."""
        nonlocal x_state
        n_steps = int(round(ps / msys.DT_PS))
        sample_every = int(round(args.sample_ps / msys.DT_PS))
        for lo in range(0, len(idx), args.max_batch):
            sub = idx[lo:lo + args.max_batch]
            x = torch.tensor(x_state[sub], device=dev, dtype=torch.float32)
            cons = CompositeConstraints(
                [RigidWaterConstraints(mols, [msys.R_OH_NM, msys.R_OH_NM, msys.r_HH_nm()],
                                       ff.params["mass"], device=dev, dtype=torch.float32),
                 PairConstraint(int(mi[0]), int(mi[1]), recs[sub, 0], ff.params["mass"],
                                device=dev, dtype=torch.float32)],
                atom_sets=[mols, mi])
            integ = BAOAB(lambda q: ff.energy_forces(q, chunk=args.chunk), ff.params["mass"],
                          cons, msys.DT_PS, msys.TEMPERATURE_K, msys.GAMMA_PS,
                          device=dev, dtype=torch.float32)
            cv = PeriodicDistanceCV(int(mi[0]), int(mi[1]), L)
            gen = torch.Generator(device=dev).manual_seed(int(1e6 + lo + int(ps * 10)))
            cons.apply_positions(x, x.clone())
            v = integ.maxwell_velocities(x, generator=gen)
            _, f = ff.energy_forces(x, chunk=args.chunk)
            for s in range(n_steps):
                _, f = integ.step(x, v, f, generator=gen)
                if not equilibrate and (s + 1) % sample_every == 0:
                    # cv.local_mean_force already returns (1/2)(F1-F2).e - 2/(beta r)
                    fl, _, _ = cv.local_mean_force(x, f, beta)
                    fsum[sub] += fl.double().cpu().numpy()
                    fcnt[sub] += 1
            if not equilibrate:
                ngsum[sub] += n_gap_batch(x.double().cpu().numpy(), mi, oxy, L)
                ngcnt[sub] += 1
            x_state[sub] = x.cpu().numpy()
            del x, v, f, integ, cons
            torch.cuda.empty_cache()

    idx_all = np.flatnonzero(active)
    print(f"\n[equil] {args.equil_ps} ps on {len(idx_all)} trajectories ...", flush=True)
    t0 = time.time()
    run_block(idx_all, args.equil_ps, equilibrate=True)
    print(f"[equil] done in {(time.time()-t0)/60:.1f} min", flush=True)

    done_ps = 0.0
    for cp in CHECKPOINTS_PS:
        block = cp - done_ps
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        print(f"\n[prod] -> {cp:.0f} ps  ({block:.0f} ps on {idx.size} trajectories) ...",
              flush=True)
        t0 = time.time()
        run_block(idx, block, equilibrate=False)
        done_ps = cp
        fbar = np.where(fcnt > 0, fsum / np.maximum(fcnt, 1), np.nan)
        denom = np.nanmean(np.abs(fbar))

        print(f"{'r':>7} {'fbar':>9} {'build sp':>9} {'wet-dry':>9} {'n_gap':>6} {'state':>10}")
        for ri, r_nm in enumerate(R_GRID_NM):
            m = recs[:, 0] == r_nm
            if not np.isnan(retired_at[ri]):
                continue
            bmeans = [fbar[m & (recs[:, 1] == b)].mean() for b in range(args.builds)]
            bsp = float(np.max(bmeans) - np.min(bmeans))
            wsp = abs(float(fbar[m & (recs[:, 2] == 1)].mean()
                            - fbar[m & (recs[:, 2] == 0)].mean()))
            ok = (bsp <= args.tol_frac * denom) and (wsp <= args.tol_frac * denom)
            if ok:
                retired_at[ri] = cp
                active[m] = False
            print(f"{r_nm:7.3f} {fbar[m].mean():9.2f} {bsp:9.2f} {wsp:9.2f} "
                  f"{np.divide(ngsum[m], np.maximum(ngcnt[m],1)).mean():6.2f} "
                  f"{'RETIRED' if ok else 'extend':>10}")
        np.savez_compressed(os.path.join(args.out, f"checkpoint_{cp:.0f}ps.npz"),
                            recs=recs, fbar=fbar, fcnt=fcnt, ngsum=ngsum,
                            retired_at=retired_at, active=active)
        print(f"[prod] checkpoint {cp:.0f} ps in {(time.time()-t0)/60:.1f} min; "
              f"{int(active.sum())} trajectories still active", flush=True)

    fbar = np.where(fcnt > 0, fsum / np.maximum(fcnt, 1), np.nan)
    np.savez_compressed(os.path.join(args.out, "ti_final.npz"),
                        recs=recs, fbar=fbar, fcnt=fcnt, ngsum=ngsum, retired_at=retired_at)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(dict(stage="ti_torch", engine="batched torch (float32, compiled)",
                       r_grid_nm=R_GRID_NM.tolist(), builds=args.builds,
                       replicas=args.replicas, equil_ps=args.equil_ps,
                       checkpoints_ps=list(CHECKPOINTS_PS), tol_frac=args.tol_frac,
                       retired_at_ps=retired_at.tolist(), box_L_nm=L,
                       wall_hours=(time.time() - t_start) / 3600.0,
                       git_commit=subprocess.run(["git", "rev-parse", "HEAD"],
                                                 capture_output=True, text=True).stdout.strip()),
                  fh, indent=2)
    print(f"\n[done] {(time.time()-t_start)/3600:.2f} h -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
