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
    for b in range(args.builds):
        for fam, r_prep, seed in ((0, R_DRY_PREP_NM, 9000 + b), (1, R_WET_PREP_NM, 9500 + b)):
            x = make_bath(mod.topology, L, pos, mi, r_prep, args.ps, seed)
            ng = n_gap(x, mi, oxy, L)
            out[f"b{b}_f{fam}"] = x.astype(np.float32)
            meta.append(dict(build=b, family=int(fam), r_prep_nm=r_prep, n_gap=float(ng),
                             seed=seed))
            print(f"  build {b} {'wet' if fam else 'dry'} (r = {r_prep}): "
                  f"n_gap = {ng:.3f}", flush=True)
    np.savez_compressed(os.path.join(args.out, "baths.npz"), **out)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(dict(stage="baths", box_L_nm=L, ps=args.ps, builds=args.builds,
                       r_dry_prep_nm=R_DRY_PREP_NM, r_wet_prep_nm=R_WET_PREP_NM,
                       baths=meta), fh, indent=2)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
