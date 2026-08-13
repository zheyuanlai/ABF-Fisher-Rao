"""Is the torch-vs-OpenMM temperature gap physics, or a kinetic-energy convention?

The dt gate compares our BAOAB kinetic temperature against OpenMM's, and OpenMM's number comes
from ``State.getKineticEnergy()``.  For Langevin-family integrators that quantity is not
necessarily ``sum m v^2 / 2`` at the on-step velocities -- OpenMM may report a half-step-shifted
or otherwise conventionalised kinetic energy -- while the torch side is the plain on-step sum.
If the two conventions differ, the gate is measuring bookkeeping rather than dynamics, and a
FAIL would cost the study a 2x compute increase for nothing.

So: run OpenMM once and compute the temperature BOTH ways on the SAME trajectory --
``getKineticEnergy()`` and the explicit sum over ``getVelocities()`` with the identical degrees
of freedom the torch side uses.  Any gap between them is convention, by construction.

Torch-free (OpenMM CUDA hangs after ``import torch``).  Run under the methane-cuda interpreter.

Usage:
    CUDA_VISIBLE_DEVICES=3 ~/miniconda3/envs/methane-cuda/bin/python \
        scripts/nacl_ke_convention_check.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import openmm as mm
import openmm.unit as u

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nacl import system as nsys                                  # noqa: E402

KB = 8.31446261815324e-3
WARM_PS, MEASURE_PS = 5.0, 15.0


def main():
    box = json.loads((nsys.REPO / "results/nacl/box/box_manifest.json").read_text())
    L = float(box["L_nm"])
    x0 = dict(np.load(nsys.REPO / "results/nacl/box/npt_final_state.npz"))["positions_nm"]

    out = {}
    for dt_fs in (2.0, 1.0):
        system, topology, _ = nsys.build_openmm_system(L)
        masses = np.array([system.getParticleMass(i).value_in_unit(u.dalton)
                           for i in range(system.getNumParticles())])
        n_cons = system.getNumConstraints()
        has_cmm = any(isinstance(f, mm.CMMotionRemover) for f in system.getForces())
        ndof_with_com = 3 * nsys.N_SITES - n_cons - 3
        ndof_no_com = 3 * nsys.N_SITES - n_cons

        integ = mm.LangevinMiddleIntegrator(nsys.TEMPERATURE_K * u.kelvin,
                                            nsys.GAMMA_PS / u.picosecond,
                                            dt_fs * 1e-3 * u.picosecond)
        integ.setRandomNumberSeed(int(10 * dt_fs))
        try:
            ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                             dict(Precision="double"))
        except Exception:
            ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CPU"))
        ctx.setPositions(x0 * u.nanometer)
        ctx.applyConstraints(1e-10)
        ctx.setVelocitiesToTemperature(nsys.TEMPERATURE_K * u.kelvin, int(10 * dt_fs))
        integ.step(int(WARM_PS / (dt_fs * 1e-3)))

        t_api, t_explicit, t_explicit_nocom = [], [], []
        n_chunk = 50
        for _ in range(int(MEASURE_PS / (dt_fs * 1e-3)) // n_chunk):
            integ.step(n_chunk)
            st = ctx.getState(getEnergy=True, getVelocities=True)
            ke_api = st.getKineticEnergy().value_in_unit(u.kilojoule_per_mole)
            v = np.asarray(st.getVelocities().value_in_unit(u.nanometer / u.picosecond))
            ke_exp = 0.5 * float((masses[:, None] * v * v).sum())
            t_api.append(2.0 * ke_api / (ndof_with_com * KB))
            t_explicit.append(2.0 * ke_exp / (ndof_with_com * KB))
            t_explicit_nocom.append(2.0 * ke_exp / (ndof_no_com * KB))
        del ctx

        rec = dict(
            T_from_getKineticEnergy=float(np.mean(t_api)),
            T_from_velocities_same_dof=float(np.mean(t_explicit)),
            T_from_velocities_no_com_subtraction=float(np.mean(t_explicit_nocom)),
            sem=float(np.std(t_explicit) / np.sqrt(len(t_explicit))),
            convention_gap_K=float(np.mean(t_api) - np.mean(t_explicit)),
            n_constraints=int(n_cons), cm_motion_remover=bool(has_cmm),
            ndof_with_com=int(ndof_with_com), ndof_no_com=int(ndof_no_com))
        out[f"{dt_fs:g}fs"] = rec
        print(f"dt = {dt_fs} fs: getKineticEnergy -> {rec['T_from_getKineticEnergy']:.3f} K | "
              f"explicit sum m v^2 (same dof) -> {rec['T_from_velocities_same_dof']:.3f} K | "
              f"gap = {rec['convention_gap_K']:+.3f} K | CMMotionRemover={has_cmm}", flush=True)

    path = nsys.REPO / "results/nacl/stage1/ke_convention_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
