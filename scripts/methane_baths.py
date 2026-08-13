"""Prepare and cache the wet/dry solvent baths.  **This file must never import torch.**

Why it is a separate process
----------------------------
Importing ``torch`` in the same process disables OpenMM's CUDA platform: creating an OpenMM CUDA
context afterwards hangs indefinitely -- measured at >20 min pinned at 0 % GPU utilisation on an
otherwise idle device, while the identical OpenMM-only path finishes the same baths in seconds.
Reordering does not help; the mere import is enough.  The two CUDA runtimes are therefore kept in
separate processes: baths are built here, cached to disk, and loaded by the torch TI driver.

The two baths are the Gate 0 instrument (SPEC §5.2): one solvent equilibrated with the methanes
in **contact** (gap empty, ``n_gap ~ 0.2``) and one at the **solvent-separated** distance (gap
filled, ``n_gap ~ 2.3``).  Any ``r`` sampled from both must give the same conditional mean force
if the conditional ensemble mixes; the difference is the signal.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/methane_baths.py --out results/methane/baths
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

from methane import system as msys                               # noqa: E402
from methane.observables import n_gap                            # noqa: E402

R_DRY_PREP_NM = 0.38
R_WET_PREP_NM = 0.80


def _platform():
    for cand in ("CUDA", "OpenCL", "CPU"):
        try:
            p = mm.Platform.getPlatformByName(cand)
            return p, ({"Precision": "mixed"} if cand in ("CUDA", "OpenCL") else {})
        except Exception:
            continue
    raise RuntimeError("no OpenMM platform")


def place(pos, mi, r_nm, L):
    """Move the methanes to separation ``r_nm``, symmetric about their current midpoint."""
    p = pos.copy()
    i, j = int(mi[0]), int(mi[1])
    d = p[j] - p[i]
    d -= L * np.round(d / L)
    e = d / np.linalg.norm(d)
    mid = p[i] + 0.5 * d
    p[i], p[j] = mid - 0.5 * r_nm * e, mid + 0.5 * r_nm * e
    return p


def make_bath(topology, L, pos, mi, r_prep, ps, seed):
    system = msys.build_system(topology, pin_pme=True)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0) * u.nanometer,
                                        mm.Vec3(0, L, 0) * u.nanometer,
                                        mm.Vec3(0, 0, L) * u.nanometer)
    system.addConstraint(int(mi[0]), int(mi[1]), float(r_prep) * u.nanometer)
    integ = mm.LangevinMiddleIntegrator(msys.TEMPERATURE_K * u.kelvin,
                                        msys.GAMMA_PS / u.picosecond,
                                        msys.DT_PS * u.picoseconds)
    integ.setRandomNumberSeed(seed)
    plat, props = _platform()
    ctx = mm.Context(system, integ, plat, props)
    ctx.setPositions(place(pos, mi, r_prep, L) * u.nanometer)
    mm.LocalEnergyMinimizer.minimize(ctx, 1.0, 3000)
    ctx.setVelocitiesToTemperature(msys.TEMPERATURE_K * u.kelvin, seed)
    integ.step(int(round(ps / msys.DT_PS)))
    out = np.asarray(ctx.getState(getPositions=True).getPositions().value_in_unit(u.nanometer))
    del ctx
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/methane/baths")
    ap.add_argument("--box", default="results/methane/box")
    ap.add_argument("--builds", type=int, default=3)
    ap.add_argument("--ps", type=float, default=200.0)
    ap.add_argument("--per-r", action="store_true",
                    help="also emit a minimised starting configuration per r")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    man = json.load(open(os.path.join(args.box, "manifest.json")))
    L = float(man["box_L_nm"])
    pos = np.load(os.path.join(args.box, "box.npz"))["positions_nm"]
    mod = msys.build_modeller(r0_nm=0.55, seed=man["seed"])
    p = msys.site_parameters(msys.build_system(mod.topology), mod.topology)
    mi = p["methane_index"]
    oxy = np.flatnonzero((~p["is_methane"]) & (p["epsilon"] > 0))
    print(f"[platform] {_platform()[0].getName()}   L = {L:.6f} nm", flush=True)

    out, meta = {}, []
    baths = {}
    for b in range(args.builds):
        for fam, r_prep, seed in ((0, R_DRY_PREP_NM, 9000 + b), (1, R_WET_PREP_NM, 9500 + b)):
            x = make_bath(mod.topology, L, pos, mi, r_prep, args.ps, seed)
            ng = n_gap(x, mi, oxy, L)
            out[f"b{b}_f{fam}"] = x.astype(np.float32)
            baths[(b, fam)] = x
            meta.append(dict(build=b, family=int(fam), r_prep_nm=r_prep, n_gap=float(ng),
                             seed=seed))
            print(f"  build {b} {'wet' if fam else 'dry'} (r = {r_prep}): "
                  f"n_gap = {ng:.3f}", flush=True)

    # ---- per-r starting configurations, minimised ------------------------------------------
    # Moving a methane to a new separation can drop it on top of a water: the raw configuration
    # then carries forces to ~1e10 kJ/mol/nm and the first 0.5 fs step destroys the walker.  In
    # the torch TI driver that surfaced as `linalg.solve: matrix is singular` inside M-SHAKE --
    # a cryptic failure whose real cause is a missing minimisation.  Minimising here, where an
    # OpenMM minimiser is available and CUDA works, keeps that failure mode out of the sampler.
    if args.per_r:
        r_grid = np.round(np.arange(0.34, 0.9001, 0.02), 4)
        print(f"[per-r] minimising {len(r_grid)} x {args.builds} x 2 starting configurations ...",
              flush=True)
        worst = 0.0
        for b in range(args.builds):
            for fam in (0, 1):
                for r_nm in r_grid:
                    system = msys.build_system(mod.topology, pin_pme=True)
                    system.setDefaultPeriodicBoxVectors(
                        mm.Vec3(L, 0, 0) * u.nanometer, mm.Vec3(0, L, 0) * u.nanometer,
                        mm.Vec3(0, 0, L) * u.nanometer)
                    system.addConstraint(int(mi[0]), int(mi[1]), float(r_nm) * u.nanometer)
                    plat, props = _platform()
                    ctx = mm.Context(system, mm.VerletIntegrator(1e-6), plat, props)
                    ctx.setPositions(place(baths[(b, fam)], mi, r_nm, L) * u.nanometer)
                    mm.LocalEnergyMinimizer.minimize(ctx, 5.0, 2000)
                    st = ctx.getState(getPositions=True, getForces=True)
                    xr = np.asarray(st.getPositions().value_in_unit(u.nanometer))
                    fmax = np.abs(np.asarray(st.getForces().value_in_unit(
                        u.kilojoule_per_mole / u.nanometer))).max()
                    worst = max(worst, float(fmax))
                    out[f"start_b{b}_f{fam}_r{r_nm:.4f}"] = xr.astype(np.float32)
                    del ctx
            print(f"  build {b} done", flush=True)
        print(f"[per-r] worst residual max|F| after minimisation = {worst:.3e} kJ/mol/nm",
              flush=True)
        if worst > 1e5:
            raise RuntimeError(f"minimisation left max|F| = {worst:.3e}; dynamics would explode")
    np.savez_compressed(os.path.join(args.out, "baths.npz"), **out)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(dict(stage="baths", box_L_nm=L, ps=args.ps, builds=args.builds,
                       r_dry_prep_nm=R_DRY_PREP_NM, r_wet_prep_nm=R_WET_PREP_NM,
                       baths=meta), fh, indent=2)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
