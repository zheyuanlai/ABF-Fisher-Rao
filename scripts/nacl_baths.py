"""Prepare and cache the NaCl hydration-family baths.  **This file must never import torch.**

Separate process for the same measured reason as ``methane_baths.py``: importing torch disables
OpenMM's CUDA platform in-process (hangs at context creation).  Baths are built here with
OpenMM CUDA, cached, and consumed by the torch TI/Gate-0 drivers.

Families (SPEC_nacl_water.md §5/§6):
    f0  CIP-derived         solvent equilibrated with the pair in contact       (r_prep 0.28 nm)
    f1  SSIP-derived        solvent equilibrated at the solvent-separated range (r_prep 0.50 nm)
    f2  dissociated-derived solvent equilibrated with free ions                 (r_prep 1.20 nm)
    f3  locally-equilibrated: shares f2's per-r starts; the torch driver gives it an extended
        (+100 ps) constrained pre-equilibration at its target r, making it an independently
        prepared local-conditional family without 61 extra baths.

Per-r starts are minimised here (placing an ion at a new separation can drop it onto a water;
unminimised that surfaces later as a singular M-SHAKE matrix -- the methane lesson, kept).

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/nacl_baths.py --out results/nacl/baths --per-r
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import openmm as mm
import openmm.unit as u

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nacl import system as nsys                                  # noqa: E402

R_PREP_NM = {0: 0.28, 1: 0.50, 2: 1.20}
FAMILY_NAMES = {0: "CIP", 1: "SSIP", 2: "dissoc"}


def _platform():
    for cand in ("CUDA", "OpenCL", "CPU"):
        try:
            p = mm.Platform.getPlatformByName(cand)
            return p, ({"Precision": "mixed"} if cand in ("CUDA", "OpenCL") else {})
        except Exception:
            continue
    raise RuntimeError("no OpenMM platform")


def place(pos, r_nm, L):
    p = pos.copy()
    d = p[1] - p[0]
    d -= L * np.round(d / L)
    e = d / np.linalg.norm(d)
    mid = p[0] + 0.5 * d
    p[0], p[1] = mid - 0.5 * r_nm * e, mid + 0.5 * r_nm * e
    return p


def _held_system(L, r_nm):
    system, topology, _ = nsys.build_openmm_system(L)
    system.addConstraint(0, 1, float(r_nm) * u.nanometer)
    return system


def make_bath(L, pos, r_prep, ps, seed):
    system = _held_system(L, r_prep)
    integ = mm.LangevinMiddleIntegrator(nsys.TEMPERATURE_K * u.kelvin,
                                        nsys.GAMMA_PS / u.picosecond,
                                        nsys.DT_PS * u.picoseconds)
    integ.setRandomNumberSeed(seed)
    plat, props = _platform()
    ctx = mm.Context(system, integ, plat, props)
    ctx.setPositions(place(pos, r_prep, L) * u.nanometer)
    mm.LocalEnergyMinimizer.minimize(ctx, 1.0, 3000)
    ctx.setVelocitiesToTemperature(nsys.TEMPERATURE_K * u.kelvin, seed)
    integ.step(int(round(ps / nsys.DT_PS)))
    out = np.asarray(ctx.getState(getPositions=True).getPositions().value_in_unit(u.nanometer))
    del ctx
    return out


def r_grid_from_box(L):
    """The frozen TI grid on the (possibly truncated) evaluation domain."""
    box = json.load(open(nsys.REPO / "results/nacl/box/box_manifest.json"))
    r_hi = float(box["finite_size_gate"]["R_hi_nm"])
    return np.round(np.arange(nsys.R_LO_NM, r_hi + 1e-9, 0.02), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/nacl/baths")
    ap.add_argument("--builds", type=int, default=3)
    ap.add_argument("--ps", type=float, default=200.0)
    ap.add_argument("--per-r", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    box = json.load(open(nsys.REPO / "results/nacl/box/box_manifest.json"))
    L = float(box["L_nm"])
    pos = dict(np.load(nsys.REPO / "results/nacl/box/npt_final_state.npz"))["positions_nm"]
    print(f"[platform] {_platform()[0].getName()}   L = {L:.6f} nm", flush=True)

    # ---- solvent baths, checkpointed as soon as they exist ---------------------------------
    # Written in one shot at the very end, ~26 min of bath MD could only be preserved by
    # finishing the whole job -- so stopping early to free a shared device meant redoing all of
    # it.  The cost of an all-or-nothing write only appears on the day you need to stop, which
    # is the day you can least afford it.
    ckpt = os.path.join(args.out, "baths_only.npz")
    out, meta, baths = {}, [], {}
    if os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        meta = list(z["meta"])
        for b in range(args.builds):
            for fam in R_PREP_NM:
                key = f"b{b}_f{fam}"
                if key not in z:
                    raise SystemExit(f"bath checkpoint lacks {key}; delete {ckpt} and rerun")
                out[key] = z[key]
                baths[(b, fam)] = z[key].astype(np.float64)
        print(f"[baths] reloaded {len(baths)} cached baths from {ckpt}", flush=True)
    else:
        for b in range(args.builds):
            for fam, r_prep in R_PREP_NM.items():
                seed = 9000 + 100 * fam + b
                x = make_bath(L, pos, r_prep, args.ps, seed)
                out[f"b{b}_f{fam}"] = x.astype(np.float32)
                baths[(b, fam)] = x
                meta.append(dict(build=b, family=fam, name=FAMILY_NAMES[fam],
                                 r_prep_nm=r_prep, seed=seed))
                print(f"  build {b} {FAMILY_NAMES[fam]:6s} (r_prep = {r_prep}) done", flush=True)
        np.savez_compressed(ckpt, meta=np.array(meta, dtype=object), **out)
        print(f"[baths] checkpointed {len(baths)} baths -> {ckpt}", flush=True)

    if args.per_r:
        # ONE context per (build, family), reused across every r.  The ions are frozen by
        # setting their masses to zero -- OpenMM's minimiser then holds them exactly at the
        # target separation while the waters relax around them, so no constraint has to be
        # baked into the System and 549 context creations collapse to 9.  (Context creation,
        # not minimisation, dominates the naive loop.)
        r_grid = r_grid_from_box(L)
        print(f"[per-r] minimising {len(r_grid)} x {args.builds} x 3 families "
              f"({len(r_grid) * args.builds * 3} configurations, 9 contexts) ...", flush=True)
        worst = 0.0
        water_idx = None
        part = os.path.join(args.out, "starts_partial.npz")
        if os.path.exists(part):
            z = np.load(part)
            for k in z.files:
                out[k] = z[k]
            print(f"[per-r] resuming: {len(z.files)} starts already on disk", flush=True)
        for b in range(args.builds):
            for fam in (0, 1, 2):
                if all(f"start_b{b}_f{fam}_r{r:.4f}" in out for r in r_grid):
                    print(f"  build {b} family {fam} already complete, skipping", flush=True)
                    continue
                system, topology, _ = nsys.build_openmm_system(L)
                for i in (0, 1):
                    system.setParticleMass(i, 0.0 * u.dalton)
                plat, props = _platform()
                ctx = mm.Context(system, mm.VerletIntegrator(1e-6), plat, props)
                for r_nm in r_grid:
                    target = place(baths[(b, fam)], float(r_nm), L)
                    ctx.setPositions(target * u.nanometer)
                    mm.LocalEnergyMinimizer.minimize(ctx, 5.0, 2000)
                    st = ctx.getState(getPositions=True, getForces=True)
                    xr = np.asarray(st.getPositions().value_in_unit(u.nanometer))
                    d = xr[1] - xr[0]
                    d -= L * np.round(d / L)
                    if abs(float(np.linalg.norm(d)) - float(r_nm)) > 1e-6:
                        raise RuntimeError(f"frozen ions moved: r = {np.linalg.norm(d):.6f} "
                                           f"vs {r_nm}")
                    if water_idx is None:
                        water_idx = np.setdiff1d(np.arange(xr.shape[0]), [0, 1])
                    fmax = np.abs(np.asarray(st.getForces().value_in_unit(
                        u.kilojoule_per_mole / u.nanometer))[water_idx]).max()
                    worst = max(worst, float(fmax))
                    out[f"start_b{b}_f{fam}_r{r_nm:.4f}"] = xr.astype(np.float32)
                del ctx
                np.savez_compressed(part, **{k: v for k, v in out.items()
                                             if k.startswith("start_")})
            print(f"  build {b} done (worst water |F| so far {worst:.3e}); "
                  f"{sum(1 for k in out if k.startswith('start_'))} starts checkpointed",
                  flush=True)
        if worst > 1e5:
            raise RuntimeError(f"minimisation left max|F| = {worst:.3e}; dynamics would explode")
        print(f"[per-r] worst residual water max|F| = {worst:.3e} kJ/mol/nm", flush=True)

    np.savez_compressed(os.path.join(args.out, "baths.npz"), **out)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(dict(stage="nacl_baths", box_L_nm=L, ps=args.ps, builds=args.builds,
                       r_prep_nm=R_PREP_NM, families=FAMILY_NAMES, baths=meta,
                       gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "unset")), fh, indent=2)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
