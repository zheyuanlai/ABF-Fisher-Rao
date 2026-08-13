"""NaCl charged-solute engine-equivalence gate (SPEC_nacl_water.md §3.1; preregistration §8.1).

Mirrors ``test_methane_engine.py`` deliberately: same tolerance, same structure, same OpenMM
Reference-platform oracle -- extended for the charged solute: LJ-active hydrogens, ion charges,
and the ion-distance CV / local-mean-force wiring.

If any test here fails, NaCl does not run.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

mm = pytest.importorskip("openmm")
import openmm.unit as u                                          # noqa: E402

from methane.cv import PeriodicDistanceCV, W_from_F, Wprime_from_Fprime  # noqa: E402
from methane.nonbonded import PairTerms                          # noqa: E402
from methane.pme import PMEReciprocal                            # noqa: E402
from nacl import system as nsys                                  # noqa: E402
from nacl.nonbonded import NaClNonbonded                         # noqa: E402

torch.set_default_dtype(torch.float64)

TOL = 1.0e-6            # SPEC §3.1 == methane SPEC §3.2


@pytest.fixture(scope="module")
def built():
    state = dict(np.load(nsys.STAGE0 / "equilibrate_state.npz"))
    L = float(state["box_nm"][0])
    system, topology, _ = nsys.build_openmm_system(L)
    nsys.assert_openmm_matches_frozen(system)
    pos = nsys.apply_constraints(system, state["positions_constrained_nm"])
    return system, topology, pos, L


def _openmm_eval(system, pos):
    ctx = mm.Context(system, mm.VerletIntegrator(1e-6),
                     mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos * u.nanometer)
    st = ctx.getState(getEnergy=True, getForces=True)
    e = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    f = np.asarray(st.getForces().value_in_unit(u.kilojoule_per_mole / u.nanometer))
    del ctx
    return e, f


def _zeroed(system, kill_charges=False, kill_lj=False):
    import copy
    s = copy.deepcopy(system)
    nbf = next(f for f in s.getForces() if isinstance(f, mm.NonbondedForce))
    for i in range(s.getNumParticles()):
        q, sig, eps = nbf.getParticleParameters(i)
        nbf.setParticleParameters(i, 0.0 if kill_charges else q, sig, 0.0 if kill_lj else eps)
    for k in range(nbf.getNumExceptions()):
        i, j, qq, sig, eps = nbf.getExceptionParameters(k)
        nbf.setExceptionParameters(k, i, j, 0.0 if kill_charges else qq, sig,
                                   0.0 if kill_lj else eps)
    return s


def _configs(system, base, n_pert=2):
    """Thermal perturbations plus a span of ion separations (CIP -> dissociated)."""
    rng = np.random.default_rng(20260813)
    out = [base]
    for scale in (0.002, 0.005, 0.01):
        for _ in range(n_pert):
            out.append(nsys.apply_constraints(system, base + rng.normal(0.0, scale, base.shape)))
    for r_nm in (0.24, 0.30, 0.36, 0.50, 0.70, 1.00, 1.30):
        c = base.copy()
        mid = 0.5 * (c[0] + c[1])
        e = c[1] - c[0]
        e = e / np.linalg.norm(e)
        c[0], c[1] = mid - 0.5 * r_nm * e, mid + 0.5 * r_nm * e
        out.append(c)
    return out


# ------------------------------------------------------------------ frozen-parameter identity
def test_frozen_params_match_openmm(built):
    system, topology, pos, L = built
    assert nsys.assert_openmm_matches_frozen(system)


def test_every_site_is_charged_and_lj_active():
    p = nsys.load_site_params()
    assert (p["epsilon"] > 0).all(), "CHARMM TIP3P: every site carries LJ"
    assert (p["charge"] != 0).all()


def test_split_path_correctly_invalidated():
    """LJ-bearing hydrogens put intramolecular pairs in the LJ set; the split path must be
    flagged invalid (not raise at construction), and must refuse to run."""
    p = nsys.load_site_params()
    L = float(p["box_nm"][0])
    pt = PairTerms(p["sigma"], p["epsilon"], p["charge"], p["exclusions"],
                   L, nsys.CUTOFF_NM, nsys.SWITCH_NM, nsys.PME_ALPHA_PER_NM)
    assert not pt.split_path_valid
    with pytest.raises(RuntimeError, match="split path"):
        pt.energy_forces_split(torch.zeros(1, len(p["sigma"]), 3))


# ------------------------------------------------------------------ the gate itself
def test_lj_only_parity(built):
    system, topology, pos, L = built
    s = _zeroed(system, kill_charges=True)
    e_mm, f_mm = _openmm_eval(s, pos)
    p = nsys.load_site_params()
    pt = PairTerms(p["sigma"], p["epsilon"], np.zeros_like(p["charge"]),
                   p["exclusions"], L, nsys.CUTOFF_NM, nsys.SWITCH_NM,
                   nsys.PME_ALPHA_PER_NM)
    e_t, f_t = pt.energy_forces(torch.tensor(pos).unsqueeze(0))
    assert abs(float(e_t[0]) - e_mm) / abs(e_mm) < TOL
    assert float((f_t[0] - torch.tensor(f_mm)).abs().max()) / np.abs(f_mm).max() < TOL


def test_electrostatics_only_parity(built):
    system, topology, pos, L = built
    s = _zeroed(system, kill_lj=True)
    e_mm, f_mm = _openmm_eval(s, pos)
    p = nsys.load_site_params()
    x = torch.tensor(pos).unsqueeze(0)
    pt = PairTerms(np.zeros_like(p["sigma"]), np.zeros_like(p["epsilon"]), p["charge"],
                   p["exclusions"], L, nsys.CUTOFF_NM, nsys.SWITCH_NM,
                   nsys.PME_ALPHA_PER_NM)
    rec = PMEReciprocal(p["charge"], L, nsys.PME_GRID, nsys.PME_ALPHA_PER_NM,
                        order=nsys.PME_SPLINE_ORDER)
    e_r, f_r = pt.energy_forces(x)
    e_x, f_x = pt.exclusion_correction(x)
    e_k, f_k = rec.energy_forces(x)
    e_t = float(e_r[0] + e_x[0] + e_k[0]) + pt.self_energy()
    f_t = (f_r + f_x + f_k)[0]
    assert abs(e_t - e_mm) / abs(e_mm) < TOL
    assert float((f_t - torch.tensor(f_mm)).abs().max()) / np.abs(f_mm).max() < TOL
    assert abs(float(e_x[0])) > 0.1 * abs(e_mm)   # the exclusion term is load-bearing


def test_full_model_parity_over_an_energy_range(built):
    system, topology, pos, L = built
    ff = NaClNonbonded(L)
    configs = _configs(system, pos)
    assert len(configs) >= 12

    energies, rel_e, rel_f = [], [], []
    for c in configs:
        e_mm, f_mm = _openmm_eval(system, c)
        e_t, f_t = ff.energy_forces(torch.tensor(c).unsqueeze(0))
        energies.append(e_mm)
        rel_e.append(abs(float(e_t[0]) - e_mm) / abs(e_mm))
        rel_f.append(float((f_t[0] - torch.tensor(f_mm)).abs().max()) / np.abs(f_mm).max())

    assert max(energies) - min(energies) > 1e4, "configurations do not span an energy range"
    assert max(rel_e) < TOL
    assert max(rel_f) < TOL


def test_batched_equals_single(built):
    system, topology, pos, L = built
    ff = NaClNonbonded(L)
    configs = _configs(system, pos)[:6]
    xb = torch.tensor(np.stack(configs))
    eb, fb = ff.energy_forces(xb)
    for k in range(len(configs)):
        e1, f1 = ff.energy_forces(xb[k:k + 1])
        assert abs(float(eb[k] - e1[0])) <= 1e-9 * abs(float(e1[0]))
        assert float((fb[k] - f1[0]).abs().max()) < 1e-9 * float(f1[0].abs().max())


def test_forces_are_the_gradient_of_the_energy(built):
    system, topology, pos, L = built
    ff = NaClNonbonded(L)
    x = torch.tensor(pos).unsqueeze(0)
    _, f = ff.energy_forces(x)
    rng = np.random.default_rng(7)
    h = 1e-6
    for idx in [(0, 0), (1, 2), (2, 1), (100, 0), (2464, 2)]:
        i, d = idx
        xp = x.clone(); xp[0, i, d] += h
        xm = x.clone(); xm[0, i, d] -= h
        ep, _ = ff.energy_forces(xp)
        em, _ = ff.energy_forces(xm)
        fd = -(float(ep[0]) - float(em[0])) / (2 * h)
        assert abs(fd - float(f[0, i, d])) < 5e-3 * max(1.0, abs(float(f[0, i, d])))


# ------------------------------------------------------------------ CV / local mean force
def test_xi_and_local_mean_force_parity(built):
    """xi to float64 round-off; f_loc computed from OpenMM forces equals f_loc from torch
    forces to the same 1e-6 (SPEC §3.1)."""
    system, topology, pos, L = built
    ff = NaClNonbonded(L)
    cv = PeriodicDistanceCV(0, 1, L)
    beta = nsys.beta_per_kJ()

    for c in _configs(system, pos)[:8]:
        x = torch.tensor(c).unsqueeze(0)
        r_torch = float(cv.value(x)[0])
        d = c[1] - c[0]
        d = d - L * np.round(d / L)
        assert abs(r_torch - float(np.linalg.norm(d))) < 1e-12

        e_mm, f_mm = _openmm_eval(system, c)
        _, f_t = ff.energy_forces(x)
        f_loc_t, r, grad_full = cv.local_mean_force(x, f_t, beta)
        f_loc_mm, _, _ = cv.local_mean_force(x, torch.tensor(f_mm).unsqueeze(0), beta)
        denom = max(abs(float(f_loc_mm[0])), 1.0)
        assert abs(float(f_loc_t[0]) - float(f_loc_mm[0])) / denom < TOL

        # ABF bias force: equal and opposite on the ions, zero elsewhere
        bias = cv.bias_force(grad_full, torch.tensor([1.7]))
        assert float((bias[0, 0] + bias[0, 1]).abs().max()) < 1e-12
        assert float(bias[0, 2:].abs().max()) == 0.0


def test_W_F_identity():
    """W'(r) = F'(r) + 2/(beta r) to machine precision (SPEC §2)."""
    beta = nsys.beta_per_kJ()
    grid = torch.linspace(nsys.R_LO_NM, nsys.R_HI_NM, nsys.N_GRID)
    Fp = torch.sin(grid * 3.0) * 40.0
    Wp = Wprime_from_Fprime(Fp, grid, beta)
    assert float((Wp - (Fp + 2.0 / (beta * grid))).abs().max()) < 1e-10

    F = torch.cos(grid * 2.0) * 10.0
    W = W_from_F(F, grid, beta)
    assert float((W - F - (2.0 / beta) * torch.log(grid)).abs().max()) < 1e-12


def test_published_pmf_convention_is_F():
    """The shipped abf.pmf spans the published domain in the F convention (121 bins, kcal/mol,
    no hideJacobian); recorded so the external check in Stage II compares like with like."""
    pmf = np.loadtxt(nsys.SRC_TUTORIAL / "output/abf.pmf")
    assert pmf.shape[0] == 121
    assert pmf[0, 0] == pytest.approx(2.0)
    assert pmf[-1, 0] == pytest.approx(14.0)
    assert pmf[-1, 1] == pytest.approx(0.0, abs=1e-12)   # zeroed at the dissociated edge
