"""Trajectory gate for the Triton pair kernel: 8 ps of constrained BAOAB against OpenMM.

The kernel already passed 5/5 static correctness gates (energies, forces, exclusions, batching,
finite differences) at 5.2e-6 against float64 -- closer than the compiled float32 tensor path
itself. **That is not sufficient to put it in a production screen.** A kernel can be right on
forces at fixed configurations and still wrong in a trajectory: the constraint-velocity defect
earlier in this campaign left every force, parity and constraint test passing at machine
precision while the ensemble sat at 156 K instead of 300 K. Temperature is the thing that
catches it, so temperature is what this gate asserts.

Exit code 0 = PASS (the caller may deploy the kernel), non-zero = FAIL (fall back to the tensor
path). Numbers are printed either way; a failure is reported as a failure, not as "pending".

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/methane_triton_gate.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import openmm as mm                                              # noqa: E402
import openmm.unit as u                                          # noqa: E402
import torch                                                     # noqa: E402

from methane import system as msys                               # noqa: E402
from methane.dynamics import (BAOAB, KB_KJ_PER_MOL_K,            # noqa: E402
                              RigidWaterConstraints, water_molecules)
from methane.nonbonded import MethaneNonbonded                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="results/methane/box")
    ap.add_argument("--out", default="results/methane/triton")
    ap.add_argument("--ps", type=float, default=8.0)
    ap.add_argument("--walkers", type=int, default=4)
    ap.add_argument("--tol-K", type=float, default=8.0, help="|T_triton - T_openmm|")
    ap.add_argument("--tol-setpoint-K", type=float, default=15.0)
    # SPEC §3.2's 1e-8 nm is a FLOAT64 statement, verified in float64 (measured 5.3e-16).
    # Production runs float32, whose round-off floor on ~1 nm coordinates is ~1e-7 nm, so 1e-8
    # is unreachable by arithmetic rather than by any solver defect -- a float32 run measures
    # ~6.6e-7 nm, about 6 ULP. Gating float32 at a float64 tolerance would reject a correct
    # kernel. 1e-5 nm is used here: still 1e-4 of a 0.1 nm bond, far below thermal fluctuation,
    # and ~10x above the round-off floor so a genuine solver failure is still caught.
    ap.add_argument("--tol-constraint-nm", type=float, default=1e-5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    man = json.load(open(os.path.join(args.box, "manifest.json")))
    L = float(man["box_L_nm"])
    pos = np.load(os.path.join(args.box, "box.npz"))["positions_nm"]
    mod = msys.build_modeller(r0_nm=0.55, seed=man["seed"])
    system = msys.build_system(mod.topology)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0) * u.nanometer,
                                        mm.Vec3(0, L, 0) * u.nanometer,
                                        mm.Vec3(0, 0, L) * u.nanometer)
    dof = 3 * msys.N_SITES - 3 * msys.N_WATERS - 3
    n_step = int(round(args.ps / msys.DT_PS))

    # ---- OpenMM reference thermostat trace (CPU: no CUDA context, so torch is unaffected) ----
    integ_mm = mm.LangevinMiddleIntegrator(msys.TEMPERATURE_K * u.kelvin,
                                           msys.GAMMA_PS / u.picosecond,
                                           msys.DT_PS * u.picoseconds)
    integ_mm.setRandomNumberSeed(7)
    ctx = mm.Context(system, integ_mm, mm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(pos * u.nanometer)
    ctx.setVelocitiesToTemperature(msys.TEMPERATURE_K * u.kelvin, 7)
    t_mm = []
    for _ in range(20):
        integ_mm.step(n_step // 20)
        ke = ctx.getState(getEnergy=True).getKineticEnergy().value_in_unit(u.kilojoule_per_mole)
        t_mm.append(2 * ke / (dof * KB_KJ_PER_MOL_K))
    del ctx
    ref = float(np.mean(t_mm[10:]))
    print(f"[openmm] mean T over the second half = {ref:.2f} K", flush=True)

    # ---- Triton path -------------------------------------------------------------------------
    dev = torch.device("cuda")
    ff = MethaneNonbonded(system, mod.topology, L, device=dev, dtype=torch.float32).enable_triton()
    cons = RigidWaterConstraints(water_molecules(mod.topology),
                                 [msys.R_OH_NM, msys.R_OH_NM, msys.r_HH_nm()],
                                 ff.params["mass"], device=dev, dtype=torch.float32)
    integ = BAOAB(lambda q: ff.energy_forces(q), ff.params["mass"], cons,
                  msys.DT_PS, msys.TEMPERATURE_K, msys.GAMMA_PS, device=dev,
                  dtype=torch.float32)
    x = torch.tensor(pos, device=dev, dtype=torch.float32).unsqueeze(0).repeat(args.walkers, 1, 1)
    gen = torch.Generator(device=dev).manual_seed(7)
    v = integ.maxwell_velocities(x, generator=gen)
    _, f = ff.energy_forces(x)
    t_tr, viol = [], []
    for i in range(n_step):
        _, f = integ.step(x, v, f, generator=gen)
        if (i + 1) % (n_step // 20) == 0:
            t_tr.append(float(integ.temperature(v).mean()))
            viol.append(cons.max_violation(x))
    ours = float(np.mean(t_tr[10:]))
    worst = float(max(viol))
    print(f"[triton] mean T over the second half = {ours:.2f} K", flush=True)
    print(f"[triton] max constraint violation    = {worst:.2e} nm", flush=True)

    d_ref = abs(ours - ref)
    d_set = abs(ours - msys.TEMPERATURE_K)
    ok = (d_ref < args.tol_K and d_set < args.tol_setpoint_K
          and worst < args.tol_constraint_nm and np.isfinite(ours))
    print(f"\n|T_triton - T_openmm| = {d_ref:.2f} K   (tol {args.tol_K})")
    print(f"|T_triton - 298 K|    = {d_set:.2f} K   (tol {args.tol_setpoint_K})")
    print(f"constraints           = {worst:.2e} nm (tol {args.tol_constraint_nm:.0e})")
    print(f"\nTRAJECTORY GATE: {'PASS' if ok else 'FAIL'}")

    with open(os.path.join(args.out, "trajectory_gate.json"), "w") as fh:
        json.dump(dict(T_triton=ours, T_openmm=ref, setpoint=msys.TEMPERATURE_K,
                       d_openmm_K=d_ref, d_setpoint_K=d_set, max_violation_nm=worst,
                       ps=args.ps, walkers=args.walkers, passed=bool(ok)), fh, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
