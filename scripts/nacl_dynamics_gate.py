"""SPEC_nacl_water.md §1.2 + §3.1 — the dynamics gate: constraints and equipartition, 1 vs 2 fs.

Torch BAOAB + M-SHAKE (the engine every arm runs) against OpenMM LangevinMiddle (the parity
oracle), same model, same dt, same gamma.  Gates:

  constraint violation (float64 dynamics)   <= 1e-8 nm     over a production-length stretch
  |T_torch - T_openmm|                      <= 2 K         at each dt, warmed and averaged

float32 is the production dtype for the screen; its constraint residual is *recorded* here
(expected ~1e-6 nm, float32 rounding at box scale) and audited at diagnostic cadence in every
run -- the 1e-8 gate is a float64 statement about the solver, not about float32 rounding.

Decides dt: 2 fs (published) if it passes both clauses, else 1 fs.  Runs BEFORE any free-energy
data; writes results/nacl/stage1/dynamics_gate.json.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/nacl_dynamics_gate.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from methane.dynamics import BAOAB, RigidWaterConstraints, KB_KJ_PER_MOL_K  # noqa: E402
from nacl import system as nsys                                  # noqa: E402
from nacl.nonbonded import NaClNonbonded                         # noqa: E402

OUT = nsys.REPO / "results/nacl/stage1"

WARM_PS = 5.0
MEASURE_PS = 20.0
B64 = 4                     #: float64 walkers (expensive, the gate)
B32 = 16                    #: float32 walkers (production dtype, recorded)


def run_torch(L, x0, dt_ps, dtype, n_walkers, device="cuda", seed=1):
    ff = NaClNonbonded(L, device=device, dtype=dtype)
    cons = RigidWaterConstraints(ff.params["waters"], nsys.rigid_water_lengths(),
                                 ff.params["mass"], device=device, dtype=dtype)
    integ = BAOAB(lambda q: ff.energy_forces(q, chunk=256), ff.params["mass"], cons,
                  dt_ps, nsys.TEMPERATURE_K, nsys.GAMMA_PS, device=device, dtype=dtype)
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.tensor(np.repeat(x0[None], n_walkers, 0), device=device, dtype=dtype)
    v = integ.maxwell_velocities(q, generator=gen)
    _, f = ff.energy_forces(q)

    n_warm = int(WARM_PS / dt_ps)
    n_meas = int(MEASURE_PS / dt_ps)
    temps, viol = [], 0.0
    t0 = time.perf_counter()
    for step in range(n_warm + n_meas):
        _, f = integ.step(q, v, f, generator=gen)
        if step >= n_warm and step % 50 == 0:
            temps.append(float(integ.temperature(v).mean()))
            viol = max(viol, cons.max_violation(q))
    wall = time.perf_counter() - t0
    ns_day = n_walkers * (n_warm + n_meas) * dt_ps * 1e-3 / wall * 86400.0
    return dict(T_mean=float(np.mean(temps)), T_sem=float(np.std(temps) / np.sqrt(len(temps))),
                max_violation_nm=viol, ns_per_day_aggregate=ns_day, wall_s=wall)


def run_openmm(L, x0, dt_ps, seed=1):
    import openmm as mm
    import openmm.unit as u

    system, topology, _ = nsys.build_openmm_system(L)
    integ = mm.LangevinMiddleIntegrator(nsys.TEMPERATURE_K * u.kelvin,
                                        nsys.GAMMA_PS / u.picosecond, dt_ps * u.picosecond)
    integ.setRandomNumberSeed(seed)
    try:
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                         dict(Precision="double"))
    except Exception:
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(x0 * u.nanometer)
    ctx.applyConstraints(1e-10)
    ctx.setVelocitiesToTemperature(nsys.TEMPERATURE_K * u.kelvin, seed)
    ndof = 3 * nsys.N_SITES - system.getNumConstraints() - 3
    integ.step(int(WARM_PS / dt_ps))
    temps = []
    n_chunk = 50
    for _ in range(int(MEASURE_PS / dt_ps) // n_chunk):
        integ.step(n_chunk)
        ke = ctx.getState(getEnergy=True).getKineticEnergy() \
            .value_in_unit(u.kilojoule_per_mole)
        temps.append(2.0 * ke / (ndof * KB_KJ_PER_MOL_K))
    return dict(T_mean=float(np.mean(temps)), T_sem=float(np.std(temps) / np.sqrt(len(temps))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    box = json.loads((nsys.REPO / "results/nacl/box/box_manifest.json").read_text())
    L = float(box["L_nm"])
    x0 = dict(np.load(nsys.REPO / "results/nacl/box/npt_final_state.npz"))["positions_nm"]

    report = dict(L_nm=L, gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                  warm_ps=WARM_PS, measure_ps=MEASURE_PS)
    # ALL OpenMM contexts run and die before torch touches CUDA -- interleaving the two
    # runtimes' context creation on one device hangs indefinitely (methane_ti_torch.py, measured)
    omm_by_dt = {dt_fs: run_openmm(L, x0, dt_fs * 1e-3, seed=int(10 * dt_fs))
                 for dt_fs in (2.0, 1.0)}
    verdicts = {}
    for dt_fs in (2.0, 1.0):
        dt = dt_fs * 1e-3
        t64 = run_torch(L, x0, dt, torch.float64, B64, seed=int(10 * dt_fs))
        t32 = run_torch(L, x0, dt, torch.float32, B32, seed=int(10 * dt_fs) + 1)
        omm = omm_by_dt[dt_fs]
        dT = abs(t64["T_mean"] - omm["T_mean"])
        ok = (t64["max_violation_nm"] <= 1e-8) and (dT <= 2.0)
        verdicts[f"{dt_fs:.0f}fs"] = dict(
            torch_float64=t64, torch_float32=t32, openmm=omm,
            dT_vs_openmm_K=dT, constraint_gate=t64["max_violation_nm"] <= 1e-8,
            equipartition_gate=dT <= 2.0, PASS=ok)
        print(f"dt={dt_fs} fs: torch64 T={t64['T_mean']:.2f}+-{t64['T_sem']:.2f} K "
              f"viol={t64['max_violation_nm']:.2e} nm | torch32 T={t32['T_mean']:.2f} "
              f"viol={t32['max_violation_nm']:.2e} | openmm T={omm['T_mean']:.2f} "
              f"| dT={dT:.2f} K -> {'PASS' if ok else 'FAIL'}", flush=True)

    chosen = 0.002 if verdicts["2fs"]["PASS"] else (0.001 if verdicts["1fs"]["PASS"] else None)
    report["verdicts"] = verdicts
    report["dt_chosen_ps"] = chosen
    if chosen is None:
        report["verdict"] = "BOTH TIMESTEPS FAIL -- engine defect, NaCl does not run"
    (OUT / "dynamics_gate.json").write_text(json.dumps(report, indent=2))
    print(f"dt chosen: {chosen} ps; wrote {OUT}/dynamics_gate.json")


if __name__ == "__main__":
    main()
