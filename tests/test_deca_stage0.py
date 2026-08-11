"""Stage-0 gates for deca-alanine (Ace-(Ala)10-Nme, vacuum ff14SB).

Covers the three things that would silently corrupt every downstream number if wrong:

  * the **builder** -- an inverted sp2 centre or a flipped chirality yields a smooth,
    plausible-looking FES made of pure artifact (the failure mode documented in
    :mod:`alanine.system`), so the structural gate is asserted, not merely reported;
  * the **energy path** -- torch vs OpenMM parity on thermally displaced configurations;
  * the **compiled** energy path -- ``torch.compile`` is a performance change and must be
    numerically indistinguishable from eager.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_deca_stage0.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

pytest.importorskip("openmm")

from alanine.forcefield import TorchFF, extract_parameters       # noqa: E402
from alkanes.distance_cv import DistanceCV                       # noqa: E402
from deca import system as dsys                                  # noqa: E402

N_RES = 10
N_ATOMS = 112


@pytest.fixture(scope="module")
def built():
    _, top, system = dsys.make_system(N_RES)
    return system, top, extract_parameters(system)


def _configs(n=12, scale=(0.005, 0.01, 0.02)):
    """A helix, an extended chain, and thermally displaced copies of both."""
    rng = np.random.default_rng(20260811)
    base = [dsys.build_helix(-57.0, -47.0), dsys.build_helix(-150.0, 150.0)]
    out = list(base)
    while len(out) < n:
        b = base[len(out) % 2]
        out.append(b + scale[len(out) % len(scale)] * rng.standard_normal(b.shape))
    return np.stack(out)


# --------------------------------------------------------------------------- topology
def test_system_shape_and_supported_forces(built):
    system, top, P = built
    assert system.getNumParticles() == N_ATOMS
    assert top.getNumResidues() == N_RES + 2                    # ACE + 10 ALA + NME
    # extract_parameters raises on anything it cannot re-implement; reaching here means the
    # system contains only bond/angle/torsion/nonbonded terms.
    assert set(P) >= {"bonds", "angles", "torsions", "nb", "exceptions", "masses"}
    assert system.getNumConstraints() == 0, "the batched BAOAB integrator implements no SHAKE"
    assert abs(P["nb"][0].sum()) < 1e-10, "capped peptide must be neutral"


def test_cv_atoms_are_the_terminal_carbonyl_carbons(built):
    _, _, _ = built
    names, _ = dsys.names_and_bonds(N_RES)
    i, j = dsys.terminal_carbonyls(N_RES)
    assert names[i][:1] + names[i][2:3] == ("ACE", "C")
    assert names[j][0] == "ALA" and names[j][1] == N_RES and names[j][2] == "C"


# --------------------------------------------------------------------------- builder gate
@pytest.mark.parametrize("phi,psi", [(-57.0, -47.0), (-150.0, 150.0), (-80.0, 80.0),
                                     (-60.0, 130.0), (60.0, 60.0)])
def test_builder_passes_the_structural_gate(built, phi, psi):
    system, _, _ = built
    x = dsys.build_helix(phi, psi, n_res=N_RES)
    assert x.shape == (N_ATOMS, 3)
    ok, rep = dsys.validate_structure(system, x, N_RES)
    assert ok, f"structural gate failed at (phi,psi)=({phi},{psi}): {rep}"


def test_builder_realises_the_requested_torsions(built):
    x = dsys.build_helix(-57.0, -47.0, n_res=N_RES)[None]
    phi, psi, omg = dsys.backbone_dihedrals(x, N_RES)
    assert np.allclose(phi[0], -57.0, atol=1e-6)
    assert np.allclose(psi[0], -47.0, atol=1e-6)
    assert np.allclose(np.abs(omg[0]), 180.0, atol=1e-6)


def test_every_residue_is_L(built):
    for phi, psi in [(-57.0, -47.0), (-150.0, 150.0), (60.0, 60.0)]:
        chir = dsys.per_residue_chirality(dsys.build_helix(phi, psi, n_res=N_RES)[None], N_RES)
        assert (chir > 0).all(), f"D-residue produced at ({phi},{psi}): {chir}"


def test_carbonyl_oxygen_is_anti_to_its_own_alpha_carbon(built):
    """The convention the vendored dipeptide builder inverted.

    For a trans amide ``dihedral(CA_next, N_next, C, O) = 0``; the vendored
    ``alanine._ala22_src.build_positions`` used 180 and inverted the sp2 centre.
    """
    I = dsys.atom_index(N_RES)
    x = dsys.build_helix(-57.0, -47.0, n_res=N_RES)[None]
    for r in range(1, N_RES):
        d = float(dsys.dihedral_np(x, (I[(r + 1, "CA")], I[(r + 1, "N")],
                                       I[(r, "C")], I[(r, "O")]))[0])
        assert abs(d) < 5.0, f"residue {r} carbonyl O is not anti to CA: {d:.2f} deg"


def test_extended_chain_is_longer_than_the_helix(built):
    i, j = dsys.terminal_carbonyls(N_RES)
    h = dsys.build_helix(-57.0, -47.0, n_res=N_RES)
    e = dsys.build_helix(-150.0, 150.0, n_res=N_RES)
    r_h = float(np.linalg.norm(h[j] - h[i]))
    r_e = float(np.linalg.norm(e[j] - e[i]))
    assert 1.4 < r_h < 2.0 and 3.2 < r_e < 3.9 and r_e > r_h + 1.5


def test_gate_rejects_a_deliberately_broken_structure(built):
    """A gate that never fails is not a gate."""
    system, _, _ = built
    x = dsys.build_helix(-57.0, -47.0, n_res=N_RES).copy()
    I = dsys.atom_index(N_RES)
    x[I[(5, "CB")]], x[I[(5, "HA")]] = x[I[(5, "HA")]].copy(), x[I[(5, "CB")]].copy()
    ok, rep = dsys.validate_structure(system, x, N_RES)
    assert not ok and rep["n_fail_chirality"] == 1


# --------------------------------------------------------------------------- thermal gate
def test_thermal_gate_accepts_warm_structures_the_builder_gate_rejects():
    """The two gates ask different questions and must not be interchanged.

    A configuration perturbed at the scale of 300 K motion fails the builder tolerances on
    bonds and angles while remaining a perfectly valid L-peptide with all-trans amides.
    """
    rng = np.random.default_rng(5)
    x = dsys.build_helix(-57.0, -47.0, n_res=N_RES) + 0.012 * rng.standard_normal((N_ATOMS, 3))
    ok_t, rep_t = dsys.validate_thermal(x, N_RES)
    assert ok_t, rep_t
    assert rep_t["n_cis_bonds"] == 0


def test_thermal_gate_catches_a_chirality_flip_and_a_cis_bond():
    I = dsys.atom_index(N_RES)
    x = dsys.build_helix(-57.0, -47.0, n_res=N_RES).copy()
    x[I[(5, "CB")]], x[I[(5, "HA")]] = x[I[(5, "HA")]].copy(), x[I[(5, "CB")]].copy()
    ok, rep = dsys.validate_thermal(x, N_RES)
    assert not ok and rep["n_fail_chirality"] == 1

    y = dsys.build_helix(-57.0, -47.0, omega_deg=0.0, n_res=N_RES)
    ok2, rep2 = dsys.validate_thermal(y, N_RES)
    assert not ok2 and rep2["n_cis_bonds"] > 0


def test_thermal_gate_catches_non_finite():
    x = dsys.build_helix(-57.0, -47.0, n_res=N_RES).copy()
    x[17, 1] = np.nan
    ok, rep = dsys.validate_thermal(x, N_RES)
    assert not ok and rep["n_fail_finite"] == 1


# --------------------------------------------------------------------------- energy parity
def test_torch_matches_openmm(built):
    import openmm as mm
    import openmm.unit as u
    system, _, P = built
    confs = _configs()
    ctx = mm.Context(system, mm.VerletIntegrator(0.001 * u.picoseconds),
                     mm.Platform.getPlatformByName("Reference"))
    e_mm, f_mm = [], []
    for c in confs:
        ctx.setPositions(c)
        st = ctx.getState(getEnergy=True, getForces=True)
        e_mm.append(st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
        f_mm.append(st.getForces(asNumpy=True).value_in_unit(
            u.kilojoule_per_mole / u.nanometer))
    e_mm, f_mm = np.array(e_mm), np.array(f_mm)

    tff = TorchFF(P, device="cpu", dtype=torch.float64)
    x = torch.as_tensor(confs)
    e_t = tff.energy(x).numpy()
    f_t = tff.forces(x).numpy()

    rel_e = np.abs(e_t - e_mm) / np.maximum(np.abs(e_mm), 1.0)
    rel_f = np.abs(f_t - f_mm).max((1, 2)) / np.abs(f_mm).max((1, 2))
    assert rel_e.max() < 1e-6, f"energy parity {rel_e.max():.3e}"
    assert rel_f.max() < 1e-6, f"force parity {rel_f.max():.3e}"


def test_forces_are_minus_grad_energy(built):
    _, _, P = built
    tff = TorchFF(P, device="cpu", dtype=torch.float64)
    x = torch.as_tensor(_configs(4))
    f = tff.forces(x)
    eps = 1e-6
    for b, a, k in [(0, 3, 1), (1, 57, 0), (2, 104, 2), (3, 88, 1)]:
        xp, xm = x.clone(), x.clone()
        xp[b, a, k] += eps
        xm[b, a, k] -= eps
        fd = -(tff.energy(xp)[b] - tff.energy(xm)[b]) / (2 * eps)
        assert abs(float(fd - f[b, a, k])) < 1e-3 * max(1.0, abs(float(f[b, a, k])))


# --------------------------------------------------------------------------- CV geometry
def test_distance_cv_analytic_geometry_matches_autodiff(built):
    i, j = dsys.terminal_carbonyls(N_RES)
    cv = DistanceCV(i, j)
    x = torch.as_tensor(_configs(6))
    R_a, g_a, d_a = cv.geometry(x)
    R_b, g_b, d_b = cv.geometry_autodiff(x)
    assert torch.allclose(R_a, R_b, atol=1e-10)
    assert torch.allclose(g_a, g_b, atol=1e-10)
    assert torch.allclose(d_a, d_b, atol=1e-8)


# --------------------------------------------------------------------------- compiled path
@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled path is a GPU gate")
def test_compiled_forces_are_numerically_indistinguishable_from_eager(built):
    from deca.engine import DecaEngine
    system, _, _ = built
    eng = DecaEngine(system, device="cuda", dtype=torch.float64, compiled=True)
    x = torch.as_tensor(_configs(16), device="cuda", dtype=torch.float64).contiguous()
    f_c, f_e = eng.forces(x), eng.eager_forces(x)
    e_c, e_e = eng.energy(x), eng.eager_energy(x)
    rel_f = (f_c - f_e).abs().max() / f_e.abs().max()
    rel_e = (e_c - e_e).abs().max() / e_e.abs().max()
    assert float(rel_f) < 1e-12, f"compiled forces differ from eager: {float(rel_f):.3e}"
    assert float(rel_e) < 1e-12, f"compiled energy differs from eager: {float(rel_e):.3e}"
