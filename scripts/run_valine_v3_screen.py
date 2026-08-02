#!/usr/bin/env python
"""Gate V3: does plain ABF DISCOVER every state and then leave one UNDER-ESTABLISHED?

This is the decisive experiment of the whole Val study, and it is deliberately run before any
publication-quality reference or any mFR arm.  It is also the gate that killed alanine.

**ABF only.**  No Fisher-Rao arm, no sham arm.  The question is a property of ABF, and running
mFR alongside would invite reading the comparison before the precondition for it is established.

Two initialisations, and they answer different questions
-------------------------------------------------------
``concentrated``  every walker starts in the dominant state.  This is the real workflow --
                  single initial basin -> discovery -> establishment -- and it is the headline.
``stratified``    walkers are spread over every state found by the S1 map.  Diagnostic only: it
                  separates "ABF cannot FIND the state" from "ABF cannot HOLD its population
                  once it exists".  A deficit that survives stratified initialisation is an
                  establishment failure; one that disappears was a discovery failure.

The establishment target is BIAS-AWARE
--------------------------------------
ABF changes the biased equilibrium as it learns, so the relevant target is not the unbiased
population of a state.  With the current estimate ``F_hat_t``, the ideal biased marginal is

    q*_t(z) ~ exp(-beta (F_pilot(z) - F_hat_t(z)))            normalised on T^2
    Q*_k(t) = integral over C_k of q*_t

and the deficit is ``D_k(t) = [Q*_k(t) - P_k(t)]_+``.  Comparing against the *unbiased*
population instead would flag a state as underpopulated even when it holds exactly the
population the current bias implies -- inventing a deficit for mFR to repair.

This script runs the sampler and saves everything needed to compute those quantities; the
metrics themselves are in ``analyze_valine_v3.py``, so the run does not have to be repeated if
a definition is refined.

Usage
-----
    CUDA_VISIBLE_DEVICES=7 python -u scripts/run_valine_v3_screen.py --smoke
    CUDA_VISIBLE_DEVICES=7 python -u scripts/run_valine_v3_screen.py \
        --init concentrated --seeds 0 1 2 3 4 5 6 7 --n-steps 1000000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alanine.basins import BasinMap                                           # noqa: E402
from alanine.core2d_ala import AlaSimConfig, run_sampler_ala                  # noqa: E402
from alanine.cv2d import BackboneCV2D, FastBackboneCV2D                       # noqa: E402
from alanine.dynamics import BAOAB, KB, SeedFailure                           # noqa: E402
from alanine.forcefield import TorchFF, extract_parameters, parameter_hash    # noqa: E402
from valine import accepted                                                   # noqa: E402
from valine.system import (CHI1_ATOMS, N_ATOMS, PHI_ATOMS, PSI_ATOMS,        # noqa: E402
                           make_seed, make_system, restrained_minimise, seed_lattice,
                           validate_seed)

ALLOWED_GPUS = {"4", "5", "6", "7"}
TWO_PI = 2.0 * math.pi

#: psi is not in the CV, so an initial structure still needs a psi.  Two values, alternated
#: across walkers, so the omitted coordinate is not initialised from a single conformation.
PSI_INIT_DEG = (120.0, -40.0)


def enforce_gpu_policy(est_peak_gib):
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        raise SystemExit("CUDA_VISIBLE_DEVICES must be set explicitly (allowed: 4,5,6,7)")
    cvd = cvd.strip()
    if cvd not in ALLOWED_GPUS:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={cvd!r} is not an absolute index in "
                         f"{sorted(ALLOWED_GPUS)}; GPUs 0-3 belong to another user")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"expected exactly 1 visible device, saw {torch.cuda.device_count()}")
    free = torch.cuda.mem_get_info()[0] / 2 ** 30
    if free < 1.5 * est_peak_gib:
        raise SystemExit(f"only {free:.1f} GiB free, need 1.5 x {est_peak_gib:.1f} GiB")
    return cvd


def git_info():
    def sh(*a):
        try:
            return subprocess.check_output(a, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:                                            # noqa: BLE001
            return "unknown"
    return {"commit": sh("git", "rev-parse", "HEAD"),
            "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(sh("git", "status", "--porcelain"))}


def build_initial_ensemble(system, tff, targets_deg, counts, equil_ps, dt, gamma, temperature,
                           device, dtype, seed, verbose=True):
    """Independently thermalised walkers placed at the requested ``(phi, psi, chi1)`` targets.

    ``targets_deg`` ``(S,3)`` and ``counts`` ``(S,)`` -- how many walkers start at each target.
    Every walker gets its own Maxwell velocities and its own Langevin noise for ``equil_ps``, so
    the ensemble is not ``B`` copies of one microscopic configuration; without that, the seed-to-
    seed spread of the V3 metrics would be an artifact of a single shared structure.

    Unbiased dynamics keep walkers in their starting state on this timescale by construction:
    Stage 0 measured 11.3-17.9 kT chi1 barriers, so nothing crosses during equilibration.
    """
    X, e = make_seed((-80.0, 80.0, 180.0), system=system)
    validate_seed(system, X[None], np.radians([[-80.0, 80.0, 180.0]]), energy=[e])
    reps, kept, dropped = [], [], []
    for s, tgt in enumerate(np.asarray(targets_deg, dtype=float)):
        rot = seed_lattice(X, np.radians([tgt]))[0]
        rel, _ = restrained_minimise(system, rot * 10.0, tgt)
        try:
            validate_seed(system, rel[None] * 0.1, np.radians([tgt])[None], cv_tol_deg=5.0)
        except ValueError as exc:
            # A stratified start can land on a (phi, psi, chi1) combination that is sterically
            # impossible even though its (phi, chi1) region is real -- psi is not in the CV, so
            # the region does not pick psi for us.  Record it and redistribute those walkers
            # rather than aborting; silently dropping them would shrink the ensemble instead.
            dropped.append({"target_deg": tgt.tolist(), "walkers": int(counts[s]),
                            "reason": str(exc)})
            continue
        reps.append(rel * 0.1)
        kept.append(int(counts[s]))
    if not reps:
        raise SystemExit("every requested start structure failed validation")
    total = int(np.sum(counts))
    kept = np.asarray(kept, dtype=np.int64)
    kept = kept + (total - kept.sum()) // len(kept)
    kept[-1] += total - kept.sum()
    x0 = np.concatenate([np.repeat(reps[s][None], int(c), axis=0)
                         for s, c in enumerate(kept) if int(c) > 0])
    if verbose:
        print(f"  {len(reps)} distinct start structures -> {x0.shape[0]} walkers; "
              f"thermalising {equil_ps} ps", flush=True)
        for dd in dropped:
            print(f"    dropped start {np.round(dd['target_deg'], 1).tolist()} deg: "
                  f"{dd['reason']}", flush=True)
    x = torch.as_tensor(x0, device=device, dtype=dtype).contiguous()
    integ = BAOAB(tff.masses.cpu().numpy(), dt, gamma, temperature, lambda z: tff.forces(z),
                  device=device, dtype=dtype)
    g = torch.Generator(device=device).manual_seed(int(seed))
    v = integ.maxwell(x.shape, g, device, dtype)
    f = tff.forces(x)
    for _ in range(int(equil_ps / dt)):
        x, v, f = integ.step(x, v, f, g)
    if not torch.isfinite(x).all():
        raise RuntimeError("non-finite positions after initial thermalisation")
    return x, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/valine/v3_screen")
    ap.add_argument("--pilot", default="results/valine/pilot_reference")
    ap.add_argument("--init", default="concentrated",
                    choices=("concentrated", "stratified"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--n-replicas", type=int, default=2048)
    ap.add_argument("--n-steps", type=int, default=1_000_000)
    ap.add_argument("--save-every", type=int, default=2_000)
    ap.add_argument("--init-equil-ps", type=float, default=20.0)
    ap.add_argument("--ceiling-kT", type=float, default=8.0)
    ap.add_argument("--min-prominence-kT", type=float, default=1.0)
    ap.add_argument("--max-basins", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="wiring checks only; never production")
    ap.add_argument("--dense-cv", action="store_true",
                    help="use the dense-Hessian CV instead of the union-block one (cross-check)")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.seeds, a.n_replicas, a.n_steps, a.save_every = [0, 1], 256, 20_000, 2_000
        a.out = a.out.rstrip("/") + "_smoke"

    dtype = torch.float64
    R, N = len(a.seeds), a.n_replicas
    est_peak = 1.35e-3 * R * N
    if a.cpu:
        device, cvd = "cpu", None
    else:
        cvd = enforce_gpu_policy(est_peak)
        device = "cuda"

    # ---------------------------------------------------------------- pilot reference + basins
    pilot_path = os.path.join(a.pilot, "pilot_reference.npz")
    if not os.path.exists(pilot_path):
        raise SystemExit(f"{pilot_path} not found -- build the pilot reference first; the V3 "
                         "establishment metric is undefined without a target free energy")
    pf = np.load(pilot_path, allow_pickle=True)
    pmeta = json.load(open(os.path.join(a.pilot, "meta.json")))
    F_pilot = pf["F"]
    kT = KB * 300.0
    n_grid = F_pilot.shape[0]
    accepted.assert_accepted(n_grid=n_grid, dt_ps=accepted.DT_UNRESTRAINED_PS, restrained=False)
    mask = np.isfinite(F_pilot) & (F_pilot < a.ceiling_kT * kT)
    # name_hints=() is not cosmetic: alanine's Ramachandran boxes are written for (phi, psi) and
    # would attach a backbone name to a chi1 rotamer here.  Neutral B0, B1, ... until the states
    # have earned a name from the S1 map.
    bm = BasinMap(F_pilot, mask, kT, ceiling_kT=a.ceiling_kT,
                  min_prominence_kT=a.min_prominence_kT, max_basins=a.max_basins,
                  name_hints=())
    pops = bm.population(F_pilot)
    print(f"pilot reference {pilot_path}  ({pmeta['n_windows']} windows, "
          f"{pmeta['grid_cells_filled']}/{pmeta['grid_cells']} cells filled)")
    print(f"selected-CV regions from the pilot watershed ({bm.n_states if hasattr(bm, 'n_states') else len(bm.names)}):")
    for k, nm in enumerate(bm.names):
        c = bm.centres_deg[k]
        print(f"  {nm:>4s}  centre (phi {c[0]:+7.1f}, chi1 {c[1]:+7.1f}) deg   "
              f"depth {bm.depths_kT[k]:5.2f} kT   pilot population {pops[nm]:.4f}   "
              f"cells {int((bm.label == k).sum())}")
    rare = int(np.argmin([pops[nm] for nm in bm.names]))
    print(f"  rarest region: {bm.names[rare]} at {pops[bm.names[rare]]:.4f}")

    # ---------------------------------------------------------------- system
    _, _, system = make_system()
    P = extract_parameters(system)
    phash = parameter_hash(P)
    if phash != accepted.PARAM_HASH:
        raise SystemExit(f"param_hash {phash} != accepted {accepted.PARAM_HASH}")
    if phash != pmeta["param_hash"]:
        raise SystemExit(f"param_hash {phash} != pilot's {pmeta['param_hash']}")
    tff = TorchFF(P, device=device, dtype=dtype)
    # The union-block CV restricts the den Otter machinery to the 6 atoms {4,6,8,10,12,20} the
    # two dihedrals actually touch -- 18 of 84 coordinates, so the Hessian contraction shrinks
    # ~22x.  It is an optimisation of an exactly known quantity: G, div_v, the local mean force
    # and the scattered bias force are bitwise or 1e-12 identical to the dense path
    # (tests/test_valine_sampler.py).  ``--dense-cv`` runs the original path for cross-checking.
    cv = (BackboneCV2D if a.dense_cv else FastBackboneCV2D)(
        PHI_ATOMS, CHI1_ATOMS, n_atoms=N_ATOMS)

    sim = AlaSimConfig(dt=accepted.DT_UNRESTRAINED_PS, n_steps=a.n_steps, n_replicas=N,
                       save_every=a.save_every, n_grid=n_grid, **accepted.ESTIMATOR)
    print(f"\nABF only: init={a.init} R={R} N={N} steps={sim.n_steps} "
          f"({sim.n_steps * sim.dt:.0f} ps)  grid {n_grid}  config_hash {sim.config_hash()}")

    # ---------------------------------------------------------------- initial ensemble
    B = R * N
    if a.init == "concentrated":
        dom = int(np.argmax([pops[nm] for nm in bm.names]))
        c = bm.centres_deg[dom]
        tgt = [[c[0], p, c[1]] for p in PSI_INIT_DEG]
        cnt = [B // 2, B - B // 2]
        print(f"  concentrated in {bm.names[dom]} at (phi {c[0]:+.1f}, chi1 {c[1]:+.1f}) deg, "
              f"psi starts {PSI_INIT_DEG}")
    else:
        tgt, cnt = [], []
        per = B // (len(bm.names) * len(PSI_INIT_DEG))
        for k, nm in enumerate(bm.names):
            c = bm.centres_deg[k]
            for p in PSI_INIT_DEG:
                tgt.append([c[0], p, c[1]])
                cnt.append(per)
        cnt[-1] += B - sum(cnt)
        print(f"  stratified over {len(bm.names)} regions x {len(PSI_INIT_DEG)} psi starts, "
              f"{per} walkers each")

    t0 = time.perf_counter()
    init, init_dropped = build_initial_ensemble(
        system, tff, tgt, cnt, a.init_equil_ps, sim.dt, sim.gamma, sim.temperature,
        device, dtype, seed=4242 + (0 if a.init == "concentrated" else 131))
    init = init.reshape(R, N, N_ATOMS, 3)
    print(f"  initial ensemble built in {time.perf_counter() - t0:.0f}s", flush=True)

    if a.benchmark:
        bsim = AlaSimConfig(**{**{f.name: getattr(sim, f.name)
                                  for f in AlaSimConfig.__dataclass_fields__.values()},
                               "n_steps": 300, "save_every": 300})
        t0 = time.perf_counter()
        run_sampler_ala("abf", tff, cv, bsim, a.seeds, init, bm.label_tensor(device=device),
                        device, dtype=dtype, reference_F=None, rare_basin=0, verbose=False)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 300 * 1e3
        print(f"\n{'dense' if a.dense_cv else 'union-block'} CV: {ms:.2f} ms/step at "
              f"B={R * N}  ->  {a.n_steps} steps = {a.n_steps * ms / 1e3 / 3600:.2f} h;  "
              f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
        return

    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    rid = f"abf__{a.init}__N{N}__T{sim.n_steps}__ns{R}__{sim.config_hash()}"
    path = os.path.join(a.out, "raw", rid + ".npz")
    if os.path.exists(path) and not a.overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")

    labels = bm.label_tensor(device=device)
    try:
        out = run_sampler_ala("abf", tff, cv, sim, a.seeds, init, labels, device, dtype=dtype,
                              reference_F=None, rare_basin=rare,
                              extra_angle_atoms=PSI_ATOMS,   # the OMITTED coordinate
                              dump_dir=os.path.join(a.out, "raw", "_failures"))
    except SeedFailure as e:
        raise SystemExit(f"sampler failed: {e}")

    payload = {k: v for k, v in out.items() if isinstance(v, (np.ndarray, np.generic))}
    payload["F_pilot"] = F_pilot
    payload["basin_label"] = bm.label
    payload["basin_centres_deg"] = np.array(bm.centres_deg)
    payload["meta"] = json.dumps(dict(
        stage="V3 ABF-only screen", method="abf", init=a.init, n_replicas=N,
        n_steps=sim.n_steps, seeds=list(a.seeds), config_hash=sim.config_hash(),
        cv_class=type(cv).__name__, cv_atoms=[list(PHI_ATOMS), list(CHI1_ATOMS)],
        param_hash=phash, pilot=pilot_path, pilot_config_hash=pmeta["config_hash"],
        pilot_is_screening_only=True,
        basin_names=bm.names, basin_centres_deg=bm.centres_deg,
        basin_depths_kT=bm.depths_kT, pilot_populations=pops, rare_basin=rare,
        ceiling_kT=a.ceiling_kT, min_prominence_kT=a.min_prominence_kT,
        psi_init_deg=list(PSI_INIT_DEG), init_equil_ps=a.init_equil_ps,
        init_targets_deg=tgt, init_dropped=init_dropped,
        cuda_visible_devices=cvd, device=device, dtype="float64", kT_kJ=kT,
        git=git_info(), wall_seconds=out["wall_seconds"], ms_per_step=out["ms_per_step"],
        clip_fraction=out["clip_fraction"], force_evaluations=out["force_evaluations"],
        peak_cuda_gib=out["peak_cuda_gib"]), default=float)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)

    frac = out["basin_frac"]                       # (T, R, n_basins)
    print(f"\nhealth: clip_fraction {out['clip_fraction']:.2e} "
          f"(guard 1e-4), non-finite {int(out['n_nonfinite'].sum())}, "
          f"mean T {np.nanmean(out['temperature']):.2f} K")
    print(f"final occupancy vs pilot population:")
    for k, nm in enumerate(bm.names):
        fh = out["first_hit"][:, k]
        hit = fh[fh >= 0] * sim.dt
        print(f"  {nm:>4s} occ {frac[-1, :, k].mean():.4f}  pilot {pops[nm]:.4f}  "
              f"first hit {'never' if hit.size == 0 else f'{hit.mean():.1f} ps'} "
              f"in {int((fh >= 0).sum())}/{R} seeds")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
