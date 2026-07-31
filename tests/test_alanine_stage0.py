"""Stage-0 gates for the atomistic Ace-Ala-Nme system.

Covers V1/V2 (OpenMM parity), V7 (chirality), V14 (builder round-trip) and V15 (umbrella seed
strain) from ALANINE_SPEC.md.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alanine_stage0.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

openmm = pytest.importorskip("openmm")
import openmm.unit as u                                                       # noqa: E402
import torch                                                                  # noqa: E402

from alanine.forcefield import TorchFF, extract_parameters, parameter_hash    # noqa: E402
from alanine.system import (PHI_ATOMS, PSI_ATOMS, build_positions, chirality,  # noqa: E402
                            make_system, reference_minimum, seed_umbrella_lattice,
                            signed_dihedral_np, validate_seed, window_centers)

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def refmin():
    return reference_minimum()


def test_system_composition(refmin):
    system, _ = refmin
    P = extract_parameters(system)
    assert system.getNumParticles() == 22
    assert system.getNumConstraints() == 0
    assert len(P["bonds"][0]) == 21
    assert len(P["angles"][0]) == 36
    assert len(P["torsions"][0]) == 42
    assert len(P["exceptions"][0]) == 98
    assert abs(P["nb"][0].sum()) < 1e-10              # neutral
    assert abs(P["masses"].sum() - 144.176) < 1e-3    # C6H12N2O2
    assert len(parameter_hash(P)) == 12


def test_openmm_parity_energy_and_forces(refmin):
    """V1 / V2: the torch force field must reproduce OpenMM to 1e-8 relative."""
    system, X0 = refmin
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001 * u.picoseconds),
                         openmm.Platform.getPlatformByName("Reference"))
    tff = TorchFF(extract_parameters(system))
    rng = np.random.default_rng(20260731)
    cfg = X0[None] + 0.02 * rng.standard_normal((8, 22, 3))
    E_t = tff.energy(torch.as_tensor(cfg)).numpy()
    F_t = tff.forces(torch.as_tensor(cfg)).numpy()
    for k in range(len(cfg)):
        ctx.setPositions(cfg[k])
        s = ctx.getState(getEnergy=True, getForces=True)
        E_o = s.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        F_o = s.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer)
        assert abs(E_t[k] - E_o) / max(abs(E_o), 1.0) < 1e-8
        assert np.linalg.norm(F_t[k] - F_o) / np.linalg.norm(F_o) < 1e-8


def test_molecule_is_L_alanine(refmin):
    """V7: positive chirality volume == L (S at CA), verified by CIP construction."""
    _, X0 = refmin
    assert chirality(X0[None])[0] > 0
    mirrored = X0.copy()
    mirrored[:, 0] *= -1
    assert chirality(mirrored[None])[0] < 0


def test_force_field_is_reflection_invariant(refmin):
    """A classical FF cannot distinguish enantiomers: F_D(phi,psi) = F_L(-phi,-psi)."""
    system, X0 = refmin
    tff = TorchFF(extract_parameters(system))
    mirrored = X0.copy()
    mirrored[:, 0] *= -1
    e = tff.energy(torch.as_tensor(np.stack([X0, mirrored]))).numpy()
    assert abs(e[0] - e[1]) < 1e-9


@pytest.mark.parametrize("n", [6, 12])
def test_rigid_rotation_seeding_passes_V15(refmin, n):
    """V15: every umbrella seed must be strain-free and land on its requested centre."""
    system, X0 = refmin
    centers = window_centers(n)
    x = seed_umbrella_lattice(X0, centers)
    ok, rep = validate_seed(system, x, centers)
    assert ok.all(), (f"{(~ok).sum()}/{len(centers)} seeds failed V15: "
                      f"chirality {rep['n_fail_chirality']}, energy {rep['n_fail_energy']}, "
                      f"angle-dev {rep['n_fail_angle_dev']}, cv {rep['n_fail_cv']}")
    assert rep["angle_energy"].max() < 50.0
    assert rep["max_angle_dev_deg"].max() < 15.0


def test_rigid_rotation_preserves_internal_geometry(refmin):
    """A rigid dihedral rotation must change nothing except the dihedral."""
    system, X0 = refmin
    P = extract_parameters(system)
    bonds = P["bonds"][0]
    centers = window_centers(6)
    x = seed_umbrella_lattice(X0, centers)
    r0 = np.linalg.norm(X0[bonds[:, 0]] - X0[bonds[:, 1]], axis=-1)
    r = np.linalg.norm(x[:, bonds[:, 0]] - x[:, bonds[:, 1]], axis=-1)
    assert np.abs(r - r0[None]).max() < 1e-12
    assert (chirality(x) > 0).all()


def test_nerf_builder_is_unfit_for_seeding(refmin):
    """The rejected path, pinned so it cannot be silently reintroduced.

    ``build_positions`` rebuilds the whole molecule at each (phi,psi) and places the ACE
    carbonyl O from the wrong reference frame over much of the torus, inverting an sp2 centre.
    Measured here: 0% of windows pass V15, against 100% for rigid rotation.
    """
    system, _ = refmin
    centers = window_centers(6)
    x = np.stack([build_positions(np.degrees(a), np.degrees(b)) for a, b in centers])
    ok, rep = validate_seed(system, x, centers)
    assert ok.sum() == 0
    assert rep["angle_energy"].max() > 200.0


def test_builder_round_trip(refmin):
    """V14: the seeder must place (phi,psi) where it was asked to."""
    _, X0 = refmin
    centers = window_centers(8)
    x = seed_umbrella_lattice(X0, centers)
    phi = signed_dihedral_np(x, PHI_ATOMS)
    psi = signed_dihedral_np(x, PSI_ATOMS)

    def wrap(a):
        return np.abs((a + np.pi) % (2 * np.pi) - np.pi)

    assert np.degrees(wrap(phi - centers[:, 0])).max() < 1e-8
    assert np.degrees(wrap(psi - centers[:, 1])).max() < 1e-8


def test_hmr_conserves_total_mass():
    """HMR moves mass between atoms; it must not create or destroy any."""
    _, _, plain = make_system()
    _, _, hmr = make_system(hydrogen_mass=3.0)
    m0 = sum(plain.getParticleMass(i).value_in_unit(u.dalton) for i in range(22))
    m1 = sum(hmr.getParticleMass(i).value_in_unit(u.dalton) for i in range(22))
    assert abs(m0 - m1) < 1e-9
