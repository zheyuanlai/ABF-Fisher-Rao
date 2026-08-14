"""OpenMM CUDA spot-check of the torch reference -- SPEC §5 acceptance clause.

>= 6 windows x 2 replicas, fixed cages, LangevinMiddleIntegrator, 50 ps equilibration +
250 ps production, <f> = mean of (1/2)(F_A,z - F_B,z) sampled every 0.25 ps with 5 ps block
SEMs.  Torch and OpenMM must agree within combined block error per spot.

No torch import anywhere in this process (the measured CUDA-runtime deadlock).

Usage:  CUDA_VISIBLE_DEVICES=3 python scripts/c60_reference_spotcheck.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from c60 import geometry, system as csys  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "c60", "reference")
BOX = os.path.join(os.path.dirname(__file__), "..", "results", "c60", "box", "frozen_box.npz")

SPOT_WINDOWS = (0.908, 0.968, 1.10, 1.30, 1.70, 2.428)
N_REP = 2
EQUIL_PS = 50.0
PROD_PS = 250.0
SAMPLE_PS = 0.25
BLOCK_PS = 5.0


def main():
    import openmm as mm
    import openmm.unit as u

    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if dev != "3":
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={dev!r}; SPEC §11 pins this study to GPU 3")
    os.makedirs(OUT, exist_ok=True)

    with open(os.path.join(os.path.dirname(OUT), "parity", "dt_gate.json")) as fh:
        dt = float(json.load(fh)["decision_dt_ps"])

    fz = np.load(BOX)
    lx, lz = float(fz["lx_nm"]), float(fz["lz_nm"])
    base = np.asarray(fz["positions"], dtype=np.float64)
    mod = csys.build_modeller()
    box = [mm.Vec3(lx, 0, 0), mm.Vec3(0, lx, 0), mm.Vec3(0, 0, lz)] * u.nanometer
    system = csys.build_system(mod.topology, box_vectors=box, pme_params=csys.pme_params())
    p = csys.site_parameters(system, mod.topology)
    center = np.array([0.5 * lx, 0.5 * lx, 0.5 * lz])

    results = {}
    t0 = time.perf_counter()
    for d in SPOT_WINDOWS:
        for rep in range(N_REP):
            integ = mm.LangevinMiddleIntegrator(csys.TEMPERATURE_K * u.kelvin,
                                                csys.GAMMA_PS / u.picosecond,
                                                dt * u.picoseconds)
            ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                             dict(Precision="mixed"))
            pos = base.copy()
            pos[p["carbon_index"]] = geometry.pair_positions(d, center)
            ctx.setPositions(pos * u.nanometer)
            ctx.applyConstraints(1e-10)
            ctx.computeVirtualSites()
            mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 200)
            ctx.setVelocitiesToTemperature(csys.TEMPERATURE_K * u.kelvin,
                                           3000 + rep + int(d * 1000))
            integ.step(int(round(EQUIL_PS / dt)))
            per = int(round(SAMPLE_PS / dt))
            n_samples = int(round(PROD_PS / SAMPLE_PS))
            fs = np.empty(n_samples)
            for k in range(n_samples):
                integ.step(per)
                st = ctx.getState(getForces=True)
                fvec = np.asarray(st.getForces()
                                  .value_in_unit(u.kilojoule_per_mole / u.nanometer))
                fs[k] = csys.local_mean_force(fvec, p["cage_a"], p["cage_b"])
            del ctx
            spb = int(round(BLOCK_PS / SAMPLE_PS))
            blocks = fs[: (len(fs) // spb) * spb].reshape(-1, spb).mean(axis=1)
            results[f"d{d}_rep{rep}"] = dict(
                d_nm=d, rep=rep, mean=float(fs.mean()),
                block_sem=float(blocks.std(ddof=1) / np.sqrt(len(blocks))),
                n_samples=n_samples)
            print(f"d={d} rep={rep}: <f> = {fs.mean():+.3f} "
                  f"+- {results[f'd{d}_rep{rep}']['block_sem']:.3f} kJ/mol/nm "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    with open(os.path.join(OUT, "spotcheck_openmm.json"), "w") as fh:
        json.dump(dict(results=results, dt_ps=dt, equil_ps=EQUIL_PS, prod_ps=PROD_PS,
                       device=dev), fh, indent=1)


if __name__ == "__main__":
    main()
