#!/usr/bin/env python
"""Stage 1: the global metastable-state map of Ace-Val-Nme in (phi, psi, chi1).

Gate sec.32 checked the omitted coordinate at six anchors.  That is strong evidence but it is
not a global decomposition, and gate V3 -- "does ABF discover every relevant state and then
leave one persistently under-established?" -- cannot even be *posed* until the states are named.
This script names them.

Why a lattice and not a long unbiased run
-----------------------------------------
Stage 0 measured chi1 barriers of 11.3-17.9 kT, and the backbone carries comparable ones.  An
unbiased trajectory crosses neither: e^-11 ~ 2e-5 per attempt.  So a multi-start run seeded only
at a handful of known structures returns exactly the states it was seeded with, and calls that a
discovery.  The alternative usually proposed is an exploratory bias, which buys coverage at the
cost of a second set of parameters to defend.

This script instead seeds a **dense regular lattice over the whole torus** and lets each walker
relax into whichever state contains it.  Coverage is then a property of the construction, not of
the dynamics: no state can hide from a lattice that covers T^3, up to the lattice spacing.  What
the dynamics has to supply is only *local* relaxation, which is fast and which unbiased dynamics
does correctly.  The cost is that the resulting density is NOT Boltzmann -- a state's sampled
weight is its basin-of-attraction volume.  That is the right measure for locating boundaries and
for state-CONDITIONED densities, and the wrong one for populations, which come from the pilot
free energy instead.  See `valine.states` for the same warning next to the code that enforces it.

Lattice points that are structurally impossible (steric contact, twisted peptide bond, an sp2
centre driven non-planar) are recorded and dropped, exactly as the Stage-2 windows were: they are
inaccessible regions, and reporting them is part of the map.

Usage
-----
    CUDA_VISIBLE_DEVICES=7 python -u scripts/run_valine_state_map.py --benchmark
    CUDA_VISIBLE_DEVICES=7 python -u scripts/run_valine_state_map.py \
        --out results/valine/state_map
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alanine.dynamics import BAOAB                                            # noqa: E402
from alanine.forcefield import TorchFF, extract_parameters, parameter_hash    # noqa: E402
from valine import accepted                                                   # noqa: E402
from valine.states import StateMap, transition_counts                         # noqa: E402
from valine.system import (CHI1_ATOMS, N_ATOMS, PHI_ATOMS, PSI_ATOMS,         # noqa: E402
                           make_seed, make_system, restrained_minimise, seed_lattice,
                           validate_seed)
from valine.umbrella import dihedrals_iupac                                    # noqa: E402

ALLOWED_GPUS = {"4", "5", "6", "7"}
KB = 0.008314462618
QUADS = (PHI_ATOMS, PSI_ATOMS, CHI1_ATOMS)


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


def build_lattice(system, n_phi, n_psi, n_chi, verbose=True):
    """Rigid-rotate a validated parent onto a regular ``T^3`` lattice, then relieve the strain.

    Returns ``(q0 (M,28,3) nm, targets (M,3) rad, kept_mask, dropped)``.

    Rigid rotation preserves bonds, angles, planarity and chirality exactly but does create
    steric clashes -- the Stage-2 lattice reached 0.138 nm contacts.  Restrained minimisation at
    the point's own targets relieves them while holding all three dihedrals, and it is applied
    UNIFORMLY rather than only to the failing points, so no selection effect is introduced.
    """
    X, e = make_seed((-80.0, 80.0, 180.0), system=system)
    validate_seed(system, X[None], np.radians([[-80.0, 80.0, 180.0]]), energy=[e])
    if verbose:
        print(f"  parent E = {e:.2f} kJ/mol", flush=True)

    grid = [np.linspace(-180.0, 180.0, k, endpoint=False) for k in (n_phi, n_psi, n_chi)]
    tgt_deg = np.stack(np.meshgrid(*grid, indexing="ij"), -1).reshape(-1, 3)
    M = tgt_deg.shape[0]
    rot = seed_lattice(X, np.radians(tgt_deg))                       # (M,28,3) nm, exact

    q0 = np.empty_like(rot)
    ok = np.zeros(M, dtype=bool)
    dropped = []
    t0 = time.time()
    for k in range(M):
        rel, _ = restrained_minimise(system, rot[k] * 10.0, tgt_deg[k])       # angstrom in/out
        q0[k] = rel * 0.1
        try:
            # cv_tol_deg matches the Stage-2 umbrella seeding: a 1-5 deg placement miss is a
            # numerical artifact, not a structural defect, and rejecting on it deletes real
            # regions.  Every STRUCTURAL check stays at full strictness.
            validate_seed(system, q0[k][None], np.radians(tgt_deg[k])[None], cv_tol_deg=5.0)
            ok[k] = True
        except ValueError as exc:
            why = str(exc)
            kind = ("cv_placement" if "off target" in why else
                    "steric" if "contact" in why else
                    "planarity" if "non-planar" in why else
                    "omega" if "omega" in why else
                    "chirality" if "chirality" in why else "other")
            dropped.append({"target_deg": tgt_deg[k].tolist(), "kind": kind, "reason": why})
        if verbose and (k + 1) % 500 == 0:
            print(f"  seeded {k + 1}/{M}  ({time.time() - t0:.0f}s)", flush=True)
    if verbose:
        kinds = {}
        for d in dropped:
            kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
        print(f"  {int(ok.sum())}/{M} lattice points validated in {time.time() - t0:.0f}s; "
              f"dropped {len(dropped)} as physically inaccessible: {kinds or 'none'}", flush=True)
    return q0, np.radians(tgt_deg), ok, dropped


def run_unbiased(tff, q0, n_rand, n_equil, n_prod, dt, gamma, temperature, seed, save_every,
                 device, dtype, label=""):
    """Unbiased BAOAB with a high-friction randomisation phase, saving (phi, psi, chi1).

    The randomisation phase at gamma = 20 ps^-1 decorrelates the walkers that share a lattice
    point from their common minimised structure; without it the ``walkers_per_cell`` copies are
    not independent and the split-half stability check below would be self-fulfilling.
    """
    def force(x):
        return tff.forces(x)

    integ = BAOAB(tff.masses.cpu().numpy(), dt, gamma, temperature, force,
                  device=device, dtype=dtype)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    x = q0.clone()
    v = integ.maxwell(x.shape, gen, device, dtype)
    f = force(x)

    def phase(n, g, keep):
        nonlocal x, v, f, integ
        if g is not None:
            integ = BAOAB(tff.masses.cpu().numpy(), dt, g, temperature, force,
                          device=device, dtype=dtype)
            f = force(x)
        out, temps = [], []
        t0 = time.perf_counter()
        for s in range(n):
            x, v, f = integ.step(x, v, f, gen)
            if keep and (s + 1) % save_every == 0:
                out.append(dihedrals_iupac(x, QUADS).to(torch.float32).cpu())
                temps.append(float(integ.kinetic_temperature(v)))
            if n >= 5 and (s + 1) % max(n // 5, 1) == 0:
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"non-finite positions at step {s + 1}")
                el = time.perf_counter() - t0
                print(f"    {label}{'prod' if keep else 'equil'} {s + 1}/{n}  "
                      f"{(s + 1) / el:.0f} steps/s  T={float(integ.kinetic_temperature(v)):.1f} K"
                      f"  eta {(n - s - 1) / ((s + 1) / el) / 60:.1f} min", flush=True)
        return (torch.stack(out, 1).numpy() if out else None,
                np.array(temps) if temps else np.array([]))

    phase(n_rand, 20.0, False)                    # decorrelate copies of a shared seed
    phase(n_equil, gamma, False)                  # discard: relaxation into the local state
    traj, temps = phase(n_prod, gamma, True)      # (B, frames, 3)
    return traj, temps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/valine/state_map")
    ap.add_argument("--n-phi", type=int, default=18)
    ap.add_argument("--n-psi", type=int, default=18)
    ap.add_argument("--n-chi", type=int, default=9)
    ap.add_argument("--walkers-per-cell", type=int, default=4)
    ap.add_argument("--randomize-ps", type=float, default=5.0)
    ap.add_argument("--equil-ps", type=float, default=20.0)
    ap.add_argument("--prod-ps", type=float, default=200.0)
    ap.add_argument("--save-every", type=int, default=200)      # 0.2 ps
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--cluster-cells", type=int, default=36)
    ap.add_argument("--min-prominence", type=float, default=1.5)
    ap.add_argument("--ceiling", type=float, default=6.0)
    ap.add_argument("--smooth-cells", type=float, default=1.0)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    dt = accepted.DT_UNRESTRAINED_PS            # 1 fs; unrestrained, so the 0.5 fs rule is moot
    accepted.assert_accepted(dt_ps=dt, restrained=False)

    M = a.n_phi * a.n_psi * a.n_chi
    B = M * a.walkers_per_cell
    est_peak = 6.0e-5 * B
    if a.cpu:
        device, cvd = "cpu", None
    else:
        cvd = enforce_gpu_policy(est_peak)
        device = "cuda"
    dtype = torch.float64

    print(f"lattice {a.n_phi} x {a.n_psi} x {a.n_chi} = {M} points, "
          f"{a.walkers_per_cell} walkers each -> batch {B}")
    print(f"device {device}  CUDA_VISIBLE_DEVICES={cvd}  dt={dt * 1000:.1f} fs (unrestrained)")

    _, _, system = make_system()
    P = extract_parameters(system)
    phash = parameter_hash(P)
    if phash != accepted.PARAM_HASH:
        raise SystemExit(f"param_hash {phash} != accepted {accepted.PARAM_HASH}")
    print(f"param_hash {phash}  atoms {system.getNumParticles()}  "
          f"constraints {system.getNumConstraints()}")
    tff = TorchFF(P, device=device, dtype=dtype)

    print("building lattice seeds ...", flush=True)
    q0_np, tgt, ok, dropped = build_lattice(system, a.n_phi, a.n_psi, a.n_chi)
    q0_np, tgt = q0_np[ok], tgt[ok]
    origin = np.repeat(np.arange(q0_np.shape[0]), a.walkers_per_cell)
    q0 = torch.as_tensor(np.repeat(q0_np, a.walkers_per_cell, axis=0),
                         device=device, dtype=dtype).contiguous()
    B = q0.shape[0]
    print(f"batch after dropping inaccessible points: {B}", flush=True)

    if a.benchmark:
        t0 = time.perf_counter()
        run_unbiased(tff, q0, 0, 0, 200, dt, a.gamma, a.temperature, a.seed, 1000,
                     device, dtype)
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 200 * 1e3
        total = (a.randomize_ps + a.equil_ps + a.prod_ps) / dt * ms / 1e3 / 60
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else 0.0
        print(f"\n{ms:.2f} ms/step at B={B} -> projected {total:.1f} min;  peak {peak:.2f} GiB")
        return

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    traj, temps = run_unbiased(
        tff, q0, int(a.randomize_ps / dt), int(a.equil_ps / dt), int(a.prod_ps / dt),
        dt, a.gamma, a.temperature, a.seed, a.save_every, device, dtype)
    wall = time.time() - t0
    print(f"exploration done in {wall / 60:.1f} min;  trajectory {traj.shape};  "
          f"mean T = {temps.mean():.1f} K", flush=True)

    np.savez_compressed(os.path.join(a.out, "explore.npz"),
                        theta=traj.astype(np.float32), origin=origin,
                        lattice_targets=tgt, temperature=temps)
    print(f"raw exploration written to {a.out}/explore.npz", flush=True)

    # ------------------------------------------------------------------ clustering
    # Each walker contributes equal total weight, so the density is a mixture over walkers
    # rather than being dominated by whichever state happens to have the most lattice points
    # inside it.  It is still a basin-of-attraction measure, NOT a Boltzmann one.
    pts = traj.reshape(-1, 3).astype(np.float64)
    w = np.full(pts.shape[0], 1.0 / traj.shape[1])
    print(f"clustering {pts.shape[0]:,} samples on a {a.cluster_cells}^3 torus grid ...",
          flush=True)
    sm = StateMap(pts, n=a.cluster_cells, weights=w, smooth_cells=a.smooth_cells,
                  min_prominence_kT=a.min_prominence, ceiling_kT=a.ceiling, kT=1.0)
    print(json.dumps(sm.summary(), indent=2))

    lab = sm.assign(traj)                                   # (B, frames)
    T = transition_counts(lab, sm.n_states)
    barriers = sm.barrier_matrix()

    # split-half over WALKERS: does the map survive halving the sample?
    half = StateMap(traj[::2].reshape(-1, 3).astype(np.float64), n=a.cluster_cells,
                    weights=np.full(traj[::2].size // 3, 1.0 / traj.shape[1]),
                    smooth_cells=a.smooth_cells, min_prominence_kT=a.min_prominence,
                    ceiling_kT=a.ceiling, kT=1.0)

    frac = np.array([float((lab == k).mean()) for k in range(sm.n_states)])
    unassigned = float((lab < 0).mean())
    print(f"\n{'state':>6s} {'phi':>8s} {'psi':>8s} {'chi1':>8s} {'cells':>7s} "
          f"{'attract':>8s} {'exits':>7s}")
    for k in range(sm.n_states):
        c = np.degrees(sm.centres[k])
        print(f"{sm.names[k]:>6s} {c[0]:8.1f} {c[1]:8.1f} {c[2]:8.1f} "
              f"{sm.cells_per_state()[k]:7d} {frac[k]:8.4f} {int(T[k].sum() - T[k, k]):7d}")
    print(f"\nunassigned frames (above the flood ceiling): {unassigned:.4f}")
    print(f"split-half state count: {half.n_states} vs {sm.n_states} on the full sample")
    finite_b = barriers[np.isfinite(barriers) & (barriers > 0)]
    print(f"pairwise min-max barriers (nats of relaxation density): "
          f"min {finite_b.min():.2f}  median {np.median(finite_b):.2f}  "
          f"n_infinite {int((~np.isfinite(barriers)).sum())}")

    np.savez_compressed(
        os.path.join(a.out, "states.npz"),
        label_grid=sm.label, level_grid=sm.level, G=sm.G, counts=sm.counts,
        centres=sm.centres, seeds=np.array(sm.seeds), depths=np.array(sm.depths),
        transitions=T, barriers=barriers, frame_labels=lab.astype(np.int16),
        origin=origin, lattice_targets=tgt, attract_fraction=frac)

    meta = {
        "stage": "S1 state map (phi, psi, chi1)",
        "param_hash": phash, "cuda_visible_devices": cvd, "device": device,
        "physical_model": accepted.PHYSICAL_MODEL, "dt_ps": dt,
        "lattice": {"n_phi": a.n_phi, "n_psi": a.n_psi, "n_chi": a.n_chi,
                    "points": M, "validated": int(ok.sum()), "dropped": len(dropped),
                    "walkers_per_cell": a.walkers_per_cell, "batch": B},
        "dynamics": {"randomize_ps": a.randomize_ps, "equil_ps": a.equil_ps,
                     "prod_ps": a.prod_ps, "save_every": a.save_every,
                     "gamma": a.gamma, "temperature": a.temperature, "seed": a.seed,
                     "frames": int(traj.shape[1]),
                     "mean_temperature_K": float(temps.mean())},
        "clustering": {"cells": a.cluster_cells, "min_prominence": a.min_prominence,
                       "ceiling": a.ceiling, "smooth_cells": a.smooth_cells,
                       "density_is_boltzmann": False,
                       "density_meaning": "basin-of-attraction volume from a uniform lattice; "
                                          "populations must come from the pilot free energy"},
        "states": sm.summary(),
        "attract_fraction": frac.tolist(),
        "unassigned_frame_fraction": unassigned,
        "split_half_n_states": int(half.n_states),
        "transitions": T.tolist(),
        "dropped_lattice_points": dropped[:50],
        "n_dropped_lattice_points": len(dropped),
        "wall_seconds": wall, "git": git_info(),
    }
    meta["config_hash"] = hashlib.md5(
        json.dumps({k: v for k, v in meta.items() if k not in ("git", "wall_seconds")},
                   sort_keys=True, default=str).encode()).hexdigest()[:12]
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwrote {a.out}/meta.json, explore.npz and states.npz")


if __name__ == "__main__":
    main()
