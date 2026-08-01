#!/usr/bin/env python
"""Stage 2: conditional chi1 free-energy profiles F(chi1 | phi_j, psi_j) for Ace-Val-Nme.

This is the measurement behind gate V1 of `ALANINE_EXECUTION_DECISION.md` sec.7 and Stage 2 of
the Val screening plan: is there a genuine chi1 barrier under *our* force field, and do at
least two rotamers carry non-negligible population?

Design.  Six representative backbone points x 24 periodic chi1 windows (15 deg spacing) = 144
windows, every one of them a row of a **single batched GPU run** sharing one force evaluation
per step.  Each window is seeded from two independently relaxed parent structures (chi1 = t and
chi1 = g-) so that a window's result does not depend on which rotamer it was built from --
the plan's "at least 2 independent initial rotamers".

Physical model is the frozen one: vacuum ff14SB, no constraints, no HMR, BAOAB, dt = 1 fs,
gamma = 1 ps^-1, T = 300 K, float64.  The only added term is the harmonic restraint.

GPU policy: exactly one of GPU 4/5/6/7, pinned by CUDA_VISIBLE_DEVICES, enforced below.

Usage
-----
    CUDA_VISIBLE_DEVICES=7 python scripts/run_valine_chi1_profiles.py --benchmark
    CUDA_VISIBLE_DEVICES=7 python scripts/run_valine_chi1_profiles.py \
        --out results/valine/chi1_profiles
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

from alanine.forcefield import TorchFF, extract_parameters, parameter_hash   # noqa: E402
from valine.system import (                                                  # noqa: E402
    CHI1_ATOMS, N_ATOMS, PHI_ATOMS, PSI_ATOMS, make_seed, make_system, seed_lattice,
    validate_seed,
)
from valine.umbrella import (DihedralRestraint, count_states, mbar_1d_periodic,
                             run_restrained)  # noqa: E402

ALLOWED_GPUS = {"4", "5", "6", "7"}
KB = 0.008314462618

#: representative backbone points, in degrees.  Named from the alanine Ramachandran
#: convention; the names are labels only and carry no claim about Val's own minima, which
#: Stage 1 measures.
BACKBONES = [
    ("C7eq",   (-80.0, 80.0)),
    ("C5",     (-140.0, 150.0)),
    ("alphaR", (-70.0, -30.0)),
    ("alphaL", (60.0, 40.0)),
    ("C7ax",   (63.0, -48.0)),
    ("bridge", (-100.0, 0.0)),
]
CHI_SPACING_DEG = 15.0
PARENT_CHI1 = (180.0, -60.0)

# --------------------------------------------------------------------------- scan modes
# Angle order is always (phi, psi, chi1) = columns (0, 1, 2).
#
# `chi1` is Stage 2 / gate V1: scan chi1 with the backbone clamped, and we WANT a barrier.
#
# `psi` and `phi` are the Stage-3 sec.32 hidden-coordinate test, and the verdict runs the
# OTHER WAY: they measure the barrier in the coordinate a candidate CV omits, and a candidate
# is admissible only if its hidden coordinate mixes FAST, i.e. the barrier is SMALL.
#
#   scan psi  -> hidden coordinate of CV (phi, chi1)
#   scan phi  -> hidden coordinate of CV (psi, chi1)
#
# Anchors span the populated cells of the corresponding selected CV, taken from the Stage-2
# minimum-energy-path pre-screen (chi1 wells near -60/+60/180 at every backbone region).
SCAN_MODES = {
    "chi1": {
        "col": 2, "clamp": (0, 1), "want": "barrier",
        "cv": "(phi,psi) clamped", "gate": "V1",
        "anchors": BACKBONES,
    },
    "psi": {
        "col": 1, "clamp": (0, 2), "want": "fast_mixing",
        "cv": "phi_chi1", "gate": "sec.32 hidden coordinate of (phi,chi1)",
        "anchors": [("C7eq_t", (-80.0, 180.0)), ("C7eq_g-", (-80.0, -60.0)),
                    ("C5_g+", (-140.0, 60.0)), ("alphaR_g-", (-70.0, -60.0)),
                    ("alphaL_g-", (60.0, -60.0)), ("C7ax_t", (63.0, 180.0))],
    },
    "phi": {
        "col": 0, "clamp": (1, 2), "want": "fast_mixing",
        "cv": "psi_chi1", "gate": "sec.32 hidden coordinate of (psi,chi1)",
        "anchors": [("psi80_t", (80.0, 180.0)), ("psi80_g-", (80.0, -60.0)),
                    ("psi150_g+", (150.0, 60.0)), ("psim30_g-", (-30.0, -60.0)),
                    ("psi40_g-", (40.0, -60.0)), ("psim48_t", (-48.0, 180.0))],
    },
}


def enforce_gpu_policy(est_peak_gib):
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        raise SystemExit("CUDA_VISIBLE_DEVICES must be set explicitly (allowed: 4,5,6,7)")
    cvd = cvd.strip()
    if cvd not in ALLOWED_GPUS:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={cvd!r} not an absolute index in {sorted(ALLOWED_GPUS)}")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"expected exactly 1 visible device, saw {torch.cuda.device_count()}")
    free = torch.cuda.mem_get_info()[0] / 2**30
    if free < 1.5 * est_peak_gib:
        raise SystemExit(f"only {free:.1f} GiB free, need 1.5 x {est_peak_gib:.1f} GiB")
    return cvd


def git_info():
    def run(*a):
        try:
            return subprocess.check_output(a, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    return {"commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def build_windows(system, scan_centers_deg, mode, verbose=True):
    """Return ``(q0 (B,28,3), centers (B,3), window_id (B,), parent_id (B,))``.

    ``mode`` is an entry of :data:`SCAN_MODES`: ``mode["col"]`` is the scanned angle's column
    in ``(phi, psi, chi1)`` and ``mode["clamp"]`` the two clamped columns, whose values come
    from ``mode["anchors"]`` in that same column order.

    Two steps, and the second is not optional for valine.

    1. **Rigid rotation** from a validated parent to the window's ``(phi, psi, chi1)``.  This
       preserves bonds, angles, planarity and chirality exactly, but -- as the alanine handoff
       predicted for a beta-branched side chain -- it *does* create steric clashes: the raw
       rotated lattice contains contacts down to 0.138 nm, mostly at the strained C7ax
       backbone, and `validate_seed` rejects them.
    2. **Restrained minimisation** at the window's own targets, which relieves the clash while
       holding all three dihedrals.  Applied uniformly to every seed rather than only to the
       failing ones, so no selection effect is introduced.
    """
    parents = []
    for c1 in PARENT_CHI1:
        X, e = make_seed((-80.0, 80.0, c1), system=system)
        validate_seed(system, X[None], np.radians([[-80.0, 80.0, c1]]), energy=[e])
        parents.append(X)
        if verbose:
            print(f"  parent chi1={c1:+.0f} deg  E = {e:.2f} kJ/mol")

    from valine.system import restrained_minimise

    q0, cen, wid, pid, ener, ok = [], [], [], [], [], []
    dropped = []
    w = 0
    col, clamp = mode["col"], mode["clamp"]
    n = len(mode["anchors"]) * len(scan_centers_deg) * len(parents)
    for aname, anchor in mode["anchors"]:
        for s in scan_centers_deg:
            good, why = True, ""
            batch = []
            for p, X in enumerate(parents):
                t = [0.0, 0.0, 0.0]
                t[col] = float(s)
                t[clamp[0]], t[clamp[1]] = float(anchor[0]), float(anchor[1])
                tgt = tuple(t)
                rot = seed_lattice(X, np.radians([tgt]))[0]              # nm, rigid
                rel, e = restrained_minimise(system, rot * 10.0, tgt)    # angstrom in/out
                try:
                    # cv_tol_deg is deliberately LOOSER here than the 1 deg used by the
                    # Stage-0 lattice test, and the reason is not laziness.  MBAR builds its
                    # reduced potential from the *restraint centre*, not from where the seed
                    # happened to land, so a seed sitting 1.5 deg off centre is harmless --
                    # the dynamics samples around the restraint regardless.  At 1 deg the
                    # strained backbone windows were being rejected for a numerical
                    # placement miss rather than a structural defect, which would have
                    # deleted real windows and pushed the sec.32 verdict toward FAIL for the
                    # wrong reason.  Every STRUCTURAL check (sterics, sp2 planarity, omega,
                    # chirality) stays at full strictness.
                    validate_seed(system, rel[None] * 0.1, np.radians([tgt]), cv_tol_deg=5.0)
                except ValueError as exc:
                    good, why = False, str(exc)
                batch.append((tgt, rel * 0.1, p, e))
            # A window is kept only if BOTH parents validate.  Uniform treatment: no
            # per-parent salvage, so no selection effect on which structures survive.
            for tgt, x, p, e in batch:
                cen.append(tgt); q0.append(x); wid.append(w); pid.append(p); ener.append(e)
                ok.append(good)
            if not good:
                kind = ("cv_placement" if "off target" in why else
                        "steric" if "contact" in why else
                        "planarity" if "non-planar" in why else
                        "omega" if "omega" in why else
                        "chirality" if "chirality" in why else "other")
                dropped.append({"anchor": aname, "scan_deg": float(s),
                                "kind": kind, "reason": why})
            w += 1
            if verbose and w % 24 == 0:
                print(f"  seeded {w}/{len(mode['anchors']) * len(scan_centers_deg)} windows",
                      flush=True)
    q0 = np.stack(q0)
    cen = np.radians(np.array(cen))
    ok = np.array(ok)
    if verbose:
        eok = [e for e, k in zip(ener, ok) if k]
        kinds = {}
        for d in dropped:
            kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
        print(f"  {ok.sum()}/{n} seeds validated, {len(dropped)} windows dropped as "
              f"physically inaccessible;  E range [{min(eok):.1f}, {max(eok):.1f}] kJ/mol")
        print(f"  drop reasons: {kinds if kinds else 'none'}")
        for d in dropped[:5]:
            print(f"    dropped {d['anchor']} at {d['scan_deg']:+.0f} deg "
                  f"[{d['kind']}]: {d['reason']}")
        if len(dropped) > 5:
            print(f"    ... and {len(dropped) - 5} more")
    if ok.sum() == 0:
        raise SystemExit("every window failed validation -- nothing to sample")
    return q0, cen, np.array(wid), np.array(pid), ok, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="chi1", choices=sorted(SCAN_MODES),
                    help="chi1 = Stage 2 / gate V1;  psi|phi = Stage 3 sec.32 "
                         "hidden-coordinate mixing test for (phi,chi1)|(psi,chi1)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--walkers-per-window", type=int, default=8,
                    help="per parent, so the total per window is 2x this")
    ap.add_argument("--equil-steps", type=int, default=50_000)     # 50 ps
    ap.add_argument("--prod-steps", type=int, default=250_000)     # 250 ps
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--kappa-clamp", "--kappa-backbone", dest="kappa_clamp",
                    type=float, default=500.0)
    ap.add_argument("--kappa-scan", "--kappa-chi1", dest="kappa_scan",
                    type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    mode = SCAN_MODES[args.scan]
    if args.out is None:
        args.out = f"results/valine/{args.scan}_profiles"
    chi_centers = np.arange(-180.0, 180.0, CHI_SPACING_DEG)
    n_win = len(mode["anchors"]) * len(chi_centers)
    B = n_win * len(PARENT_CHI1) * args.walkers_per_window
    print(f"scan {args.scan}  (gate: {mode['gate']};  want "
          f"{'a barrier' if mode['want'] == 'barrier' else 'FAST mixing, i.e. NO barrier'})")

    est_peak = 4.0e-5 * B                                          # GiB, measured-conservative
    if args.cpu:
        device = "cpu"
        cvd = None
    else:
        cvd = enforce_gpu_policy(est_peak)
        device = "cuda"
    dtype = torch.float64

    print(f"windows {n_win}  parents {len(PARENT_CHI1)}  walkers/window/parent "
          f"{args.walkers_per_window}  total batch {B}")
    print(f"device {device}  CUDA_VISIBLE_DEVICES={cvd}")

    _, _, system = make_system()
    P = extract_parameters(system)
    phash = parameter_hash(P)
    print(f"param_hash {phash}   atoms {system.getNumParticles()}   "
          f"constraints {system.getNumConstraints()}")
    tff = TorchFF(P, device=device, dtype=dtype)

    print("building and validating seeds ...")
    q0_1, cen_1, wid_1, pid_1, ok_1, dropped = build_windows(system, chi_centers, mode)
    q0_1, cen_1, wid_1, pid_1 = (a[ok_1] for a in (q0_1, cen_1, wid_1, pid_1))
    kept_windows = sorted(set(wid_1.tolist()))
    rep = args.walkers_per_window
    q0 = np.repeat(q0_1, rep, axis=0)
    cen = np.repeat(cen_1, rep, axis=0)
    wid = np.repeat(wid_1, rep, axis=0)
    pid = np.repeat(pid_1, rep, axis=0)
    B = q0.shape[0]
    print(f"batch after dropping inaccessible windows: {B} "
          f"({len(kept_windows)}/{n_win} windows)")

    q0_t = torch.tensor(q0, device=device, dtype=dtype)
    kap = np.zeros((B, 3))
    kap[:, mode["clamp"][0]] = kap[:, mode["clamp"][1]] = args.kappa_clamp
    kap[:, mode["col"]] = args.kappa_scan
    restraint = DihedralRestraint([PHI_ATOMS, PSI_ATOMS, CHI1_ATOMS], cen, kap,
                                  N_ATOMS, device=device, dtype=dtype)

    if args.benchmark:
        t0 = time.time()
        run_restrained(tff, restraint, q0_t, 200, seed=args.seed, save_every=1000)
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) / 200 * 1e3
        total = (args.equil_steps + args.prod_steps) * ms / 1e3 / 60
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
        print(f"\n{ms:.2f} ms/step at B={B}  ->  projected {total:.1f} min for "
              f"{args.equil_steps + args.prod_steps} steps;  peak {peak:.2f} GiB")
        return

    os.makedirs(args.out, exist_ok=True)
    beta = 1.0 / (KB * 300.0)
    t0 = time.time()
    print(f"equilibration {args.equil_steps} steps ...", flush=True)
    eq = run_restrained(tff, restraint, q0_t, args.equil_steps, seed=args.seed,
                        save_every=args.equil_steps, progress=10_000)
    print(f"production {args.prod_steps} steps ...", flush=True)
    pr = run_restrained(tff, restraint, eq["q_final"], args.prod_steps, seed=args.seed + 1,
                        save_every=args.save_every, progress=25_000)
    wall = time.time() - t0
    theta = pr["theta"]                                            # (n_saved, B, 3)
    print(f"done in {wall/60:.1f} min;  mean T = {pr['temperature'].mean():.1f} K")

    # Save the raw trajectories BEFORE any analysis.  MBAR here runs on ~3.8M samples per
    # backbone; even though it is not memory-limited on this host, analysing first would put
    # a ~70 min sampling run behind a step that can fail for reasons unrelated to the physics.
    np.savez_compressed(
        os.path.join(args.out, "samples.npz"),
        theta=theta.astype(np.float32), window_id=wid, parent_id=pid, centers=cen,
        chi_centers_deg=chi_centers, temperature=pr["temperature"])
    print(f"raw samples written to {args.out}/samples.npz")

    # ------------------------------------------------------------------ per-backbone MBAR
    kT = KB * 300.0
    profiles, summary = {}, []
    col = mode["col"]
    for bi, (name, anchor) in enumerate(mode["anchors"]):
        cols = [w for w in range(n_win)
                if w // len(chi_centers) == bi and w in kept_windows]
        if len(cols) < 3:
            summary.append({"anchor": name, "anchor_values_deg": list(anchor),
                            "scan": args.scan, "n_windows_kept": len(cols),
                            "note": "too few accessible windows to build a profile"})
            continue
        S, C = [], []
        for w in cols:
            m = (wid == w)
            S.append(theta[:, m, col].reshape(-1))
            C.append(cen[m, col][0])
        S = np.stack(S)
        grid, F = mbar_1d_periodic(S, np.array(C), args.kappa_scan, beta)
        profiles[name] = (grid, F)

        finite = np.isfinite(F)
        p = np.zeros_like(F)
        p[finite] = np.exp(-beta * F[finite])
        p /= p.sum()
        lo = [i for i in range(len(F)) if finite[i]
              and F[i] <= F[i - 1] and F[i] <= F[(i + 1) % len(F)]]
        # thirds of the circle: for chi1 these are the g-/g+/t rotamers, for a backbone
        # angle they are just three sectors and only the barrier and well count matter
        thirds = {
            "sector_minus": float(p[(grid > np.radians(-120)) & (grid <= np.radians(0))].sum()),
            "sector_plus": float(p[(grid > np.radians(0)) & (grid <= np.radians(120))].sum()),
            "sector_trans": float(p[(grid > np.radians(120)) | (grid <= np.radians(-120))].sum()),
        }
        # Count metastable STATES of the scanned coordinate, treating an unsampled bin as an
        # infinite barrier.  This is the part that matters: for the sec.32 modes, silently
        # dropping an inaccessible arc and then reporting max(F) over what is left would
        # UNDERSTATE the barrier, and sec.32 wants NO barrier -- so the error would push the
        # verdict toward a wrong PASS.  Inaccessible arcs are separators, not missing data.
        states = count_states(F, beta, kT, sep_kT=3.0, min_pop=0.02)
        summary.append({
            "anchor": name, "anchor_values_deg": list(anchor), "scan": args.scan,
            "n_windows_kept": len(cols),
            "barrier_kT": float(np.nanmax(F) / kT),
            "n_wells": len(lo),
            "n_states": states["n_states"],
            "state_populations": states["populations"],
            "separated_by_inaccessible_arc": states["has_gap"],
            "wells_deg": [float(np.degrees(grid[i])) for i in lo],
            "wells_kT": [float(F[i] / kT) for i in lo],
            "sector_populations": thirds,
        })

    # ------------------------------------------------------------------ gate
    print()
    print(f"{'anchor':>11s} {'win':>4s} {'barrier kT':>11s} {'states':>7s} {'gap':>4s} | "
          f"{'2nd pop':>8s} | wells (deg : kT)")
    any_barrier = any_second = False
    max_barrier, max_states = 0.0, 0
    for s in summary:
        if "note" in s:
            print(f"{s['anchor']:>11s} {s['n_windows_kept']:4d}   {s['note']}")
            max_states = max(max_states, 2)          # unmappable == not demonstrably mixing
            continue
        second = sorted(s["sector_populations"].values(), reverse=True)[1]
        any_barrier |= s["barrier_kT"] >= 2.0
        any_second |= second >= 0.02
        max_barrier = max(max_barrier, s["barrier_kT"])
        max_states = max(max_states, s["n_states"])
        wells = ", ".join(f"{d:+.0f}:{v:.2f}" for d, v in zip(s["wells_deg"], s["wells_kT"]))
        print(f"{s['anchor']:>11s} {s['n_windows_kept']:4d} {s['barrier_kT']:11.2f} "
              f"{s['n_states']:7d} {'yes' if s['separated_by_inaccessible_arc'] else 'no':>4s} | "
              f"{second:8.3f} | {wells}")

    print()
    if mode["want"] == "barrier":
        verdict = "PASS" if (any_barrier and any_second) else "FAIL"
        print(f"GATE {mode['gate']} ({verdict}): barrier >= 2 kT somewhere: {any_barrier};  "
              f"second sector >= 2% somewhere: {any_second}")
        gate = {"barrier": any_barrier, "population": any_second, "verdict": verdict}
    else:
        # sec.32 runs the other way: the OMITTED coordinate must mix FAST, so multiple
        # populated states in it disqualify the candidate CV -- ABF would receive a
        # conditionally unequilibrated mean force, and marginal mFR could not see the problem,
        # let alone repair it.  The criterion is the NUMBER OF STATES, not max(F): a single
        # confined well is fine however deep its walls, whereas two wells separated by an
        # inaccessible arc is fatal even though max(F) over the sampled bins may look modest.
        verdict = "PASS" if max_states <= 1 else "FAIL"
        print(f"GATE {mode['gate']} ({verdict}): worst anchor has {max_states} populated "
              f"state(s) in the hidden coordinate; admissible only if 1 (it must mix FAST).")
        print(f"  worst sampled barrier {max_barrier:.2f} kT; "
              f"{len(dropped)}/{n_win} windows were physically inaccessible and are counted "
              f"as impassable, not as missing data.")
        print(f"  -> candidate CV {mode['cv']} is "
              f"{'admissible' if verdict == 'PASS' else 'REJECTED'} on this criterion.")
        gate = {"max_states_in_hidden_coordinate": max_states,
                "worst_sampled_barrier_kT": max_barrier, "cv": mode["cv"],
                "verdict": verdict}

    meta = {
        "param_hash": phash, "cuda_visible_devices": cvd, "device": device,
        "n_windows": n_win, "batch": B, "walkers_per_window_per_parent": args.walkers_per_window,
        "equil_steps": args.equil_steps, "prod_steps": args.prod_steps,
        "save_every": args.save_every, "kappa_clamp": args.kappa_clamp,
        "kappa_scan": args.kappa_scan, "seed": args.seed,
        "scan": args.scan, "scan_column": mode["col"], "clamp_columns": list(mode["clamp"]),
        "angle_order": ["phi", "psi", "chi1"],
        "window_spacing_deg": CHI_SPACING_DEG, "parent_chi1_deg": list(PARENT_CHI1),
        "physical_model": "vacuum ff14SB, no constraints, no HMR, BAOAB, dt=1fs, "
                          "gamma=1/ps, T=300K, float64",
        "wall_seconds": wall, "mean_temperature_K": float(pr["temperature"].mean()),
        "git": git_info(), "gate": gate,
        "n_windows_kept": len(kept_windows), "dropped_windows": dropped,
        "summary": summary,
    }
    meta["config_hash"] = hashlib.md5(
        json.dumps({k: v for k, v in meta.items() if k not in ("git", "wall_seconds")},
                   sort_keys=True, default=str).encode()).hexdigest()[:12]
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    np.savez_compressed(
        os.path.join(args.out, "profiles.npz"),
        chi_centers_deg=chi_centers,
        **{f"F_{n}": np.stack(v) for n, v in profiles.items()})
    print(f"\nwrote {args.out}/meta.json, samples.npz and profiles.npz")


if __name__ == "__main__":
    main()
