"""Constrained-dynamics gates for the methane engine (SPEC_methane_water.md §3.2).

The parity gate of ``test_methane_engine.py`` compares energies and forces at *fixed*
configurations, so it is blind to the integrator.  These are the two clauses of §3.2 that are
not: **constraint satisfaction** and **equipartition**.

They matter more than usual here because the constrained-TI reference runs in OpenMM while the
population arms run in this sampler (Amendment 12.4), so the two engines must agree on the
dynamics, not only on the forces.  Every wrong variant of the constraint velocity update tried
during development left the force tests, the parity gate and the constraint gate all passing and
only moved the temperature -- 156 K and 349 K against OpenMM's ~300 K.  Temperature is therefore
asserted, not reported.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_methane_dynamics.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

pytest.importorskip("openmm")
torch = pytest.importorskip("torch")

import openmm as mm                                              # noqa: E402
import openmm.unit as u                                          # noqa: E402

from methane import system as msys                               # noqa: E402
from methane.nonbonded import MethaneNonbonded                    # noqa: E402
from methane.dynamics import (BAOAB, KB_KJ_PER_MOL_K,             # noqa: E402
                              RigidWaterConstraints, water_molecules)

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def minimised():
    """A minimised box.  Dynamics may not start from raw ``addSolvent`` output.

    The freshly solvated configuration carries forces up to 1e6 kJ/mol/nm; a single 0.5 fs kick
    against those injects ~500 nm/ps and the run is destroyed on step one.  Minimisation is part
    of the protocol, not a convenience.
    """
    mod = msys.build_modeller(r0_nm=0.55, seed=20260812)
    system = msys.build_system(mod.topology)
    pos = msys.apply_constraints(
        system, mod.topology, np.asarray(mod.positions.value_in_unit(u.nanometer)))
    ctx = mm.Context(system, mm.VerletIntegrator(1e-6), mm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(pos * u.nanometer)
    f_before = np.abs(np.asarray(ctx.getState(getForces=True).getForces()
                                 .value_in_unit(u.kilojoule_per_mole / u.nanometer))).max()
    mm.LocalEnergyMinimizer.minimize(ctx, 1.0, 3000)
    st = ctx.getState(getPositions=True, getForces=True)
    out = np.asarray(st.getPositions().value_in_unit(u.nanometer))
    f_after = np.abs(np.asarray(st.getForces()
                                .value_in_unit(u.kilojoule_per_mole / u.nanometer))).max()
    del ctx
    L = float(mod.topology.getUnitCellDimensions().x)
    return mod, system, out, L, f_before, f_after


def _integrator(mod, system, L, device=None):
    ff = MethaneNonbonded(system, mod.topology, L, device=device)
    cons = RigidWaterConstraints(
        water_molecules(mod.topology),
        [msys.R_OH_NM, msys.R_OH_NM, msys.r_HH_nm()], ff.params["mass"], device=device)
    integ = BAOAB(lambda q: ff.energy_forces(q, chunk=512), ff.params["mass"], cons,
                  msys.DT_PS, msys.TEMPERATURE_K, msys.GAMMA_PS, device=device)
    return ff, cons, integ


def test_minimisation_is_required_before_dynamics(minimised):
    _, _, _, _, f_before, f_after = minimised
    assert f_before > 1e5, "raw solvated box should carry huge forces"
    assert f_after < 1e4, "minimisation should remove them"


def test_dof_count_excludes_constraints(minimised):
    mod, system, pos, L, _, _ = minimised
    _, cons, integ = _integrator(mod, system, L)
    assert cons.n_constraints == 3 * msys.N_WATERS
    # 3N - 3*N_w - 3 (centre of mass)
    assert integ.n_dof(msys.N_SITES) == 3 * msys.N_SITES - 3 * msys.N_WATERS - 3


def test_maxwell_draw_is_at_the_setpoint(minimised):
    mod, system, pos, L, _, _ = minimised
    _, _, integ = _integrator(mod, system, L)
    x = torch.tensor(pos).unsqueeze(0).repeat(2, 1, 1).contiguous()
    g = torch.Generator().manual_seed(3)
    v = integ.maxwell_velocities(x, generator=g)
    assert abs(float(integ.temperature(v).mean()) - msys.TEMPERATURE_K) < 25


def test_constraints_hold_over_a_trajectory(minimised):
    mod, system, pos, L, _, _ = minimised
    ff, cons, integ = _integrator(mod, system, L)
    x = torch.tensor(pos).unsqueeze(0).repeat(2, 1, 1).contiguous()
    g = torch.Generator().manual_seed(5)
    v = integ.maxwell_velocities(x, generator=g)
    e, f = ff.energy_forces(x, chunk=512)
    worst = 0.0
    for _ in range(60):
        e, f = integ.step(x, v, f, generator=g)
        worst = max(worst, cons.max_violation(x))
    assert worst < 1.0e-8, f"constraint violation {worst:.2e} nm exceeds the SPEC §3.2 gate"


@pytest.mark.slow
def test_equipartition_matches_openmm(minimised):
    """The gate that every wrong constraint-velocity variant failed. ~8 ps, minutes on CPU."""
    mod, system, pos, L, _, _ = minimised
    dof = 3 * msys.N_SITES - 3 * msys.N_WATERS - 3
    n_step = 16000

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

    ff, cons, integ = _integrator(mod, system, L)
    x = torch.tensor(pos).unsqueeze(0).repeat(2, 1, 1).contiguous()
    g = torch.Generator().manual_seed(7)
    v = integ.maxwell_velocities(x, generator=g)
    e, f = ff.energy_forces(x, chunk=512)
    t_ours = []
    for i in range(n_step):
        e, f = integ.step(x, v, f, generator=g)
        if (i + 1) % (n_step // 20) == 0:
            t_ours.append(float(integ.temperature(v).mean()))

    ours, ref = float(np.mean(t_ours[10:])), float(np.mean(t_mm[10:]))
    assert abs(ours - msys.TEMPERATURE_K) < 15, f"ours settled at {ours:.1f} K"
    assert abs(ours - ref) < 8, f"ours {ours:.1f} K vs OpenMM {ref:.1f} K"
