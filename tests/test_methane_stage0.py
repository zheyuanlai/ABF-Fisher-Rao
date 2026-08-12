"""Stage-0 gates for the methane/SPC-E system builder (SPEC_methane_water.md §1, §3.2).

These run before any engine exists.  They gate the things that would make the later
engine-equivalence gate *meaningless* rather than merely failing:

  * the built system is the specified one (512 waters, 1538 sites, rigid water, no flexible
    internal terms) -- a flexible water model would silently invalidate every constraint and
    equipartition statement downstream;
  * every force-field number matches the frozen constants, read back **out of OpenMM** rather
    than out of the module that wrote them;
  * PME ``alpha`` and grid are **pinned**, because OpenMM otherwise derives them at Context
    creation from the box, and a torch engine matched on one box would mismatch another;
  * the analytic dispersion correction really is **force-free**, which is the justification
    SPEC §1.1 and ``methane.system`` give for switching it off outside NPT.  It is asserted here
    rather than assumed.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_methane_stage0.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

pytest.importorskip("openmm")

import openmm as mm                                              # noqa: E402
import openmm.unit as u                                          # noqa: E402

from methane import system as msys                               # noqa: E402


@pytest.fixture(scope="module")
def built():
    mod = msys.build_modeller(r0_nm=0.55, seed=20260812)
    system = msys.build_system(mod.topology)
    return mod, system


def test_composition_is_the_specified_one(built):
    mod, system = built
    n_w = sum(1 for r in mod.topology.residues() if r.name in ("HOH", "WAT"))
    assert n_w == msys.N_WATERS == 512
    assert mod.topology.getNumAtoms() == msys.N_SITES == 1538
    assert system.getNumParticles() == 1538


def test_water_is_rigid_with_no_flexible_terms(built):
    _, system = built
    # 3 constraints per water (O-H, O-H, H-H); build_system raises if either count is wrong,
    # so this asserts the gate itself is live rather than trivially satisfied.
    assert system.getNumConstraints() == 3 * msys.N_WATERS == 1536
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            assert force.getNumBonds() == 0
        if isinstance(force, mm.HarmonicAngleForce):
            assert force.getNumAngles() == 0


def test_parameters_read_back_out_of_openmm_match_the_frozen_constants(built):
    mod, system = built
    p = msys.site_parameters(system, mod.topology)
    m = p["methane_index"]
    assert m.size == 2

    # methane: neutral single LJ site
    assert np.allclose(p["charge"][m], 0.0)
    assert np.allclose(p["sigma"][m], msys.SIGMA_M_NM, atol=1e-12)
    assert np.allclose(p["epsilon"][m], msys.EPSILON_M_KJ, rtol=1e-12)
    assert np.allclose(p["mass"][m], msys.MASS_METHANE_AMU, rtol=1e-9)

    # water: LJ on oxygen only, SPC/E charges, hydrogens carry no radius
    water = ~p["is_methane"]
    oxy = water & (p["epsilon"] > 0.0)
    hyd = water & (p["epsilon"] == 0.0)
    assert oxy.sum() == msys.N_WATERS and hyd.sum() == 2 * msys.N_WATERS
    assert np.allclose(p["sigma"][oxy], msys.SIGMA_O_NM, atol=1e-12)
    assert np.allclose(p["epsilon"][oxy], msys.EPSILON_O_KJ, rtol=1e-9)
    assert np.allclose(p["charge"][oxy], msys.Q_O_E, atol=1e-12)
    assert np.allclose(p["charge"][hyd], msys.Q_H_E, atol=1e-12)
    assert np.allclose(p["sigma"][hyd], 0.0)          # normalised, cannot contribute a radius

    # the whole box is neutral
    assert abs(float(p["charge"].sum())) < 1e-9


def test_simulated_sigma_O_is_ambers_not_the_nominal_literature_value():
    """Pinned so the 9e-5 gap cannot silently change, and cannot be silently "corrected".

    The published SPC/E sigma_O is 3.166 A; ``amber14/spce.xml`` installs 0.31657195050398826 nm.
    A torch engine written from the nominal value fails the 1e-6 parity gate of SPEC §3.2 by
    ~90x, for a reason that is not a bug.
    """
    rel = abs(msys.SIGMA_O_NM - msys.SIGMA_O_NOMINAL_NM) / msys.SIGMA_O_NOMINAL_NM
    assert rel == pytest.approx(9.0e-5, rel=0.05)
    assert rel > 1e-6, "gap is below the parity tolerance; this test is no longer needed"


def test_unlike_pair_is_the_declared_lorentz_berthelot_choice():
    sig, eps = msys.unlike_pair(msys.SIGMA_M_NM, msys.EPSILON_M_KJ,
                                msys.SIGMA_O_NM, msys.EPSILON_O_KJ)
    assert sig == pytest.approx(0.344786, abs=1e-6)
    assert eps == pytest.approx(0.894022, rel=1e-5)
    assert eps / msys.KCAL_PER_KJ == pytest.approx(0.21368, rel=1e-4)


def test_rigid_geometry_constants(built):
    mod, system = built
    assert msys.r_HH_nm() == pytest.approx(0.163299, abs=1e-6)
    # the constraints OpenMM actually installed carry the same two lengths
    lengths = set()
    for k in range(system.getNumConstraints()):
        _, _, d = system.getConstraintParameters(k)
        lengths.add(round(d.value_in_unit(u.nanometer), 9))
    assert len(lengths) == 2
    assert min(lengths) == pytest.approx(msys.R_OH_NM, abs=1e-6)
    assert max(lengths) == pytest.approx(msys.r_HH_nm(), abs=1e-6)
    # raw addSolvent output carries only ~1e-4 nm of internal precision; the rigid geometry is
    # exact only after constraints are applied, which is what apply_constraints exists for.
    raw = np.asarray(mod.positions.value_in_unit(u.nanometer))
    with pytest.raises(RuntimeError, match="rigid-water violation"):
        msys.validate_geometry(raw, mod.topology, tol_nm=1e-8)
    msys.apply_constraints(system, mod.topology, raw)      # validates at 1e-8 internally


def test_validate_geometry_is_a_live_gate(built):
    mod, _ = built
    pos = np.asarray(mod.positions.value_in_unit(u.nanometer)).copy()
    first_h = next(a.index for r in mod.topology.residues() if r.name in ("HOH", "WAT")
                   for a in r.atoms() if a.name == "H1")
    pos[first_h] += 0.01                       # 10 pm displacement, far above tolerance
    with pytest.raises(RuntimeError, match="rigid-water violation"):
        msys.validate_geometry(pos, mod.topology, tol_nm=1e-8)


def test_exclusions_are_pure_and_complete(built):
    _, system = built
    pairs = msys.exclusions(system)
    assert pairs.shape == (3 * msys.N_WATERS, 2) == (1536, 2)
    # every exclusion is intramolecular within one water
    assert np.all(pairs.max(axis=1) - pairs.min(axis=1) <= 2)


def test_pme_parameters_are_pinned_not_derived(built):
    mod, system = built
    nbf = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    alpha, nx, ny, nz = nbf.getPMEParameters()
    assert alpha.value_in_unit(u.nanometer**-1) == pytest.approx(msys.PME_ALPHA_PER_NM, rel=1e-12)
    assert (nx, ny, nz) == msys.PME_GRID

    # and they survive into the Context -- an unpinned system reports (0, 0, 0) here and lets
    # OpenMM re-derive per box, which is exactly what would break torch parity.
    ctx = mm.Context(system, mm.VerletIntegrator(1e-6),
                     mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(mod.positions)
    a2, gx, gy, gz = nbf.getPMEParametersInContext(ctx)
    a2 = a2.value_in_unit(u.nanometer**-1) if u.is_quantity(a2) else float(a2)
    assert a2 == pytest.approx(msys.PME_ALPHA_PER_NM, rel=1e-9)
    assert (gx, gy, gz) == msys.PME_GRID
    del ctx


def test_dispersion_correction_is_force_free(built):
    """SPEC §1.1 switches it off outside NPT on the grounds that it exerts no force.

    That claim is load-bearing -- it is why the torch engine need not implement it and why the
    parity gate is still a statement about the physical model.  Assert it.
    """
    mod, _ = built
    forces = {}
    energies = {}
    for flag in (False, True):
        system = msys.build_system(mod.topology, dispersion_correction=flag)
        ctx = mm.Context(system, mm.VerletIntegrator(1e-6),
                         mm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(mod.positions)
        state = ctx.getState(getForces=True, getEnergy=True)
        forces[flag] = np.asarray(
            state.getForces().value_in_unit(u.kilojoule_per_mole / u.nanometer))
        energies[flag] = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        del ctx

    scale = np.abs(forces[False]).max()
    assert np.abs(forces[True] - forces[False]).max() / scale < 1e-12, "correction exerts force"
    assert energies[True] != energies[False], "correction should shift the energy"
