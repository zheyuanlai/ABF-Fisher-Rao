"""Engine-equivalence gate for the batched methane engine (SPEC_methane_water.md §3.2).

This is the gate that blocks everything downstream: "an approximate reimplementation is not a
literature reproduction and will not be called one. If the gate fails, methane does not run."

The comparison is decomposed rather than run only in aggregate, because a full-model agreement
can hide two compensating errors, and because each piece has its own characteristic failure:

  * **LJ alone** (charges zeroed) -- minimum image, the OpenMM switching function, the declared
    Lorentz-Berthelot rule, and LJ exclusions;
  * **electrostatics alone** (epsilons zeroed) -- real-space ``erfc``, the smooth-PME reciprocal
    sum, the Ewald self term, and the intramolecular exclusion correction.  The last of these is
    the term most often omitted in a hand-written PME, and omitting it produces a large, smooth,
    entirely plausible energy rather than an error;
  * **the full model** over configurations spanning a wide energy range;
  * **batching** -- ``B`` walkers evaluated together must equal ``B`` evaluated singly, or every
    population result is quietly wrong in a way no single-walker test can see.

Reference platform, double precision, PME parameters pinned by ``methane.system``.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_methane_engine.py -q
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
from methane.nonbonded import MethaneNonbonded, PairTerms        # noqa: E402
from methane.pme import PMEReciprocal, bspline_weights, influence_function  # noqa: E402

torch.set_default_dtype(torch.float64)

TOL = 1.0e-6            # SPEC §3.2


@pytest.fixture(scope="module")
def built():
    mod = msys.build_modeller(r0_nm=0.55, seed=20260812)
    system = msys.build_system(mod.topology)
    pos = msys.apply_constraints(
        system, mod.topology, np.asarray(mod.positions.value_in_unit(u.nanometer)))
    L = float(mod.topology.getUnitCellDimensions().x)
    return mod, system, pos, L


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
    """A copy of the NonbondedForce with one interaction switched off."""
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


def _configs(system, topology, base, n_pert=3):
    """Thermally perturbed configurations plus a span of methane separations."""
    rng = np.random.default_rng(20260812)
    out = [base]
    for scale in (0.002, 0.005, 0.01):
        for _ in range(n_pert):
            out.append(msys.apply_constraints(
                system, topology, base + rng.normal(0.0, scale, base.shape)))
    for r_nm in (0.34, 0.50, 0.72, 0.89):
        c = base.copy()
        mid = 0.5 * (c[0] + c[1])
        e = c[1] - c[0]
        e = e / np.linalg.norm(e)
        c[0], c[1] = mid - 0.5 * r_nm * e, mid + 0.5 * r_nm * e
        out.append(c)
    return out


# ------------------------------------------------------------------ B-spline / influence fn
@pytest.mark.parametrize("order", [4, 5, 6])
def test_bsplines_are_a_partition_of_unity(order):
    w = torch.rand(2000)
    s = bspline_weights(w, order).sum(-1)
    assert float((s - 1.0).abs().max()) < 1e-13


def test_euler_spline_repair_is_live_at_odd_order():
    """Order 5 on an even grid makes ``|b(K/2)|^-2`` vanish exactly; unrepaired it is +inf.

    OpenMM uses order 5, so this point is always hit and the repair is not optional.
    """
    ker = influence_function((20, 20, 20), 2.6148, msys.PME_ALPHA_PER_NM, 5)
    assert torch.isfinite(ker).all()
    assert float(ker.max()) < 1e6

    # the raw modulus really does vanish there -- i.e. the repair is doing work, not decoration
    from methane.pme import _bspline_at_integers
    Mn = _bspline_at_integers(5)
    k = torch.arange(4, dtype=torch.float64)
    denom = (Mn * torch.exp(1j * 2.0 * np.pi * 10.0 * k / 20.0)).sum()
    assert float(denom.abs()) < 1e-12


# ------------------------------------------------------------------ the gate itself
def test_lj_only_parity(built):
    mod, system, pos, L = built
    s = _zeroed(system, kill_charges=True)
    e_mm, f_mm = _openmm_eval(s, pos)
    p = msys.site_parameters(system, mod.topology)
    pt = PairTerms(p["sigma"], p["epsilon"], np.zeros_like(p["charge"]),
                   msys.exclusions(system), L, msys.CUTOFF_NM, msys.SWITCH_NM,
                   msys.PME_ALPHA_PER_NM)
    e_t, f_t = pt.energy_forces(torch.tensor(pos).unsqueeze(0))
    assert abs(float(e_t[0]) - e_mm) / abs(e_mm) < TOL
    assert float((f_t[0] - torch.tensor(f_mm)).abs().max()) / np.abs(f_mm).max() < TOL


def test_electrostatics_only_parity(built):
    mod, system, pos, L = built
    s = _zeroed(system, kill_lj=True)
    e_mm, f_mm = _openmm_eval(s, pos)
    p = msys.site_parameters(system, mod.topology)
    x = torch.tensor(pos).unsqueeze(0)
    pt = PairTerms(np.zeros_like(p["sigma"]), np.zeros_like(p["epsilon"]), p["charge"],
                   msys.exclusions(system), L, msys.CUTOFF_NM, msys.SWITCH_NM,
                   msys.PME_ALPHA_PER_NM)
    rec = PMEReciprocal(p["charge"], L, msys.PME_GRID, msys.PME_ALPHA_PER_NM,
                        order=msys.PME_SPLINE_ORDER)
    e_r, f_r = pt.energy_forces(x)
    e_x, f_x = pt.exclusion_correction(x)
    e_k, f_k = rec.energy_forces(x)
    e_t = float(e_r[0] + e_x[0] + e_k[0]) + pt.self_energy()
    f_t = (f_r + f_x + f_k)[0]
    assert abs(e_t - e_mm) / abs(e_mm) < TOL
    assert float((f_t - torch.tensor(f_mm)).abs().max()) / np.abs(f_mm).max() < TOL

    # and the exclusion correction is not negligible -- if it were, this test would not be
    # protecting against the defect it exists for
    assert abs(float(e_x[0])) > 0.1 * abs(e_mm)


def test_full_model_parity_over_an_energy_range(built):
    mod, system, pos, L = built
    ff = MethaneNonbonded(system, mod.topology, L)
    configs = _configs(system, mod.topology, pos)
    assert len(configs) >= 12, "SPEC §3.2 asks for at least 12 configurations"

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
    """B walkers together must equal B walkers singly, or every population result is wrong."""
    mod, system, pos, L = built
    ff = MethaneNonbonded(system, mod.topology, L)
    configs = _configs(system, mod.topology, pos)[:6]
    xb = torch.tensor(np.stack(configs))
    e_b, f_b = ff.energy_forces(xb)
    for k in range(len(configs)):
        e_1, f_1 = ff.energy_forces(xb[k:k + 1])
        assert abs(float(e_b[k] - e_1[0])) / abs(float(e_1[0])) < 1e-12
        assert float((f_b[k] - f_1[0]).abs().max() / f_1[0].abs().max()) < 1e-12


def test_forces_are_the_gradient_of_the_energy(built):
    """Finite-difference check -- independent of OpenMM, catches a consistent-but-wrong pair."""
    mod, system, pos, L = built
    ff = MethaneNonbonded(system, mod.topology, L)
    x = torch.tensor(pos).unsqueeze(0)
    _, f = ff.energy_forces(x)
    rng = np.random.default_rng(7)
    h = 1e-6
    for _ in range(5):
        i = int(rng.integers(0, x.shape[1]))
        d = int(rng.integers(0, 3))
        xp = x.clone(); xp[0, i, d] += h
        xm = x.clone(); xm[0, i, d] -= h
        ep, _ = ff.energy_forces(xp)
        em, _ = ff.energy_forces(xm)
        fd = -float(ep[0] - em[0]) / (2 * h)
        assert abs(fd - float(f[0, i, d])) <= 1e-4 * max(1.0, abs(float(f[0, i, d])))


def test_alternative_pair_paths_agree_with_the_parity_validated_one(built):
    """The neighbour-list and split-site paths are measured *slower* (see their docstrings).

    They are retained as recorded negatives, so they must stay correct: a future change that
    breaks them silently would waste someone's time re-deriving why they were rejected.
    """
    from methane.nonbonded import VerletList
    mod, system, pos, L = built
    ff = MethaneNonbonded(system, mod.topology, L)
    rng = np.random.default_rng(11)
    x = torch.tensor(np.stack([
        msys.apply_constraints(system, mod.topology, pos + rng.normal(0, 0.002, pos.shape))
        for _ in range(3)]))

    e_ap, f_ap = ff.pair.energy_forces(x, chunk=128)
    scale = float(f_ap.abs().max())

    e_sp, f_sp = ff.pair.energy_forces_split(x, chunk=128)
    assert float((e_sp - e_ap).abs().max()) / abs(float(e_ap[0])) < 1e-12
    assert float((f_sp - f_ap).abs().max()) / scale < 1e-12

    nl = VerletList(ff.n, L, msys.CUTOFF_NM)
    nl.rebuild(x, ff.pair.excluded, chunk=128)
    e_nl, f_nl = ff.pair.energy_forces_nl(x, nl, chunk=128)
    assert float((e_nl - e_ap).abs().max()) / abs(float(e_ap[0])) < 1e-12
    assert float((f_nl - f_ap).abs().max()) / scale < 1e-12
