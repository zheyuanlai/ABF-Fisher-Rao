"""Category-A correctness regression tests for src/alanine/.

Covers: IUPAC convention, full-state birth--death cloning, per-seed RNG isolation, non-finite
containment and fail-fast, online/frozen field equality, random-field projection consistency,
and structural no-reference-leakage.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alanine_categoryA.py -q
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

pytest.importorskip("openmm")

from alanine import projection as pj                                       # noqa: E402
from alanine.cv2d import BackboneCV2D, iupac_to_rb, rb_to_iupac            # noqa: E402
from alanine.dynamics import (BAOAB, SeedFailure, assert_no_reference_leakage,  # noqa: E402
                              birth_death_full_state, check_finite, make_seed_streams)
from alanine.forcefield import TorchFF, extract_parameters                 # noqa: E402
from alanine.system import (PHI_ATOMS, PSI_ATOMS, reference_minimum,       # noqa: E402
                            signed_dihedral_np)
from alkanes import density2d as d2                                        # noqa: E402
from alkanes import poisson2d as ps                                        # noqa: E402
from alkanes.cv2d import JointDihedralCV2D                                 # noqa: E402

PI = math.pi


@pytest.fixture(scope="module")
def sysmin():
    system, X0 = reference_minimum()
    return system, X0, TorchFF(extract_parameters(system))


# ---------------------------------------------------------------- IUPAC convention
def test_iupac_values_match_numpy_at_C7eq(sysmin):
    _, X0, _ = sysmin
    q = torch.as_tensor(X0)[None]
    cv = BackboneCV2D(PHI_ATOMS, PSI_ATOMS, n_atoms=22)
    want = np.array([signed_dihedral_np(X0[None], a)[0] for a in (PHI_ATOMS, PSI_ATOMS)])
    got = np.array([v.item() for v in cv.values(q)])
    assert np.abs(np.degrees(got - want)).max() < 1e-9
    # C7eq for L-alanine sits at negative phi, positive psi
    assert math.degrees(got[0]) < 0 and math.degrees(got[1]) > 0


def test_iupac_conversion_reaches_every_consumer_exactly_once(sysmin):
    """grad_only / geometry / local_mean_force route through self.values -- one shift only."""
    _, X0, _ = sysmin
    q = torch.as_tensor(X0)[None]
    cv = BackboneCV2D(PHI_ATOMS, PSI_ATOMS, n_atoms=22)
    want = torch.stack(cv.values(q), -1)
    assert (cv.grad_only(q)[0] - want).abs().max() < 1e-12
    assert (cv.geometry(q)["phi"] - want).abs().max() < 1e-12
    _, phi, _, _ = cv.local_mean_force(q, torch.zeros_like(q), 0.4)
    assert (phi - want).abs().max() < 1e-12


def test_convention_shift_does_not_change_any_geometry(sysmin):
    """A constant shift has zero derivative: mean force and Gram must be bit-identical."""
    _, X0, tff = sysmin
    rng = np.random.default_rng(0)
    q = torch.as_tensor(X0[None] + 0.01 * rng.standard_normal((4, 22, 3)))
    F = tff.forces(q)
    a = BackboneCV2D(PHI_ATOMS, PSI_ATOMS, n_atoms=22)
    b = JointDihedralCV2D(PHI_ATOMS, PSI_ATOMS, n_atoms=22)
    fa, _, ga, geoa = a.local_mean_force(q, F, 0.4)
    fb, _, gb, geob = b.local_mean_force(q, F, 0.4)
    assert (fa - fb).abs().max() == 0.0
    assert (ga - gb).abs().max() == 0.0
    assert (geoa["G"] - geob["G"]).abs().max() == 0.0


def test_convention_round_trip():
    a = torch.linspace(-PI, PI - 1e-6, 101)
    assert (iupac_to_rb(rb_to_iupac(a)) - a).abs().max() < 1e-12


# ---------------------------------------------------------------- full-state cloning
def _toy_state(R=3, N=64, A=22, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(R, N, A, 3, generator=g)
    v = torch.randn(R, N, A, 3, generator=g)
    f = torch.randn(R, N, A, 3, generator=g)
    anc = torch.arange(N).expand(R, N).clone()
    return q, v, f, anc


def _integrator():
    return BAOAB(masses=np.full(22, 12.0), dt=0.001, gamma=1.0, temperature=300.0,
                 force_fn=lambda x: torch.zeros_like(x))


def test_clone_copies_position_force_and_genealogy_but_not_velocity():
    q, v, f, anc = _toy_state()
    R, N = anc.shape
    score = torch.linspace(-2, 2, N).expand(R, N).contiguous()
    gens = make_seed_streams(11, R, q.device)
    q2, v2, f2, anc2, n_ev, deaths, births = birth_death_full_state(
        q, v, f, score, anc, gens, fr_rate=50.0, dt_eff=0.05,
        max_event_fraction=0.5, integrator=_integrator())
    assert n_ev.sum() > 0, "test needs at least one event"
    for r in range(R):
        if deaths[r] is None:
            continue
        di, src = deaths[r], births[r]
        assert torch.equal(q2[r, di], q[r, src])          # position cloned
        assert torch.equal(f2[r, di], f[r, src])          # cached force cloned
        assert torch.equal(anc2[r, di], anc[r, src])      # genealogy inherited
        # velocity is FRESH, not the parent's and not the dead replica's
        assert not torch.allclose(v2[r, di], v[r, src])
        assert not torch.allclose(v2[r, di], v[r, di])


def test_clone_leaves_the_parent_untouched():
    q, v, f, anc = _toy_state()
    R, N = anc.shape
    score = torch.linspace(-2, 2, N).expand(R, N).contiguous()
    gens = make_seed_streams(11, R, q.device)
    q2, v2, f2, anc2, n_ev, deaths, births = birth_death_full_state(
        q, v, f, score, anc, gens, fr_rate=50.0, dt_eff=0.05,
        max_event_fraction=0.5, integrator=_integrator())
    for r in range(R):
        if deaths[r] is None:
            continue
        survivors = torch.ones(N, dtype=torch.bool)
        survivors[deaths[r]] = False
        assert torch.equal(q2[r, survivors], q[r, survivors])
        assert torch.equal(v2[r, survivors], v[r, survivors])
        assert torch.equal(f2[r, survivors], f[r, survivors])


def test_population_is_fixed_and_no_self_aliasing():
    q, v, f, anc = _toy_state()
    R, N = anc.shape
    score = torch.linspace(-2, 2, N).expand(R, N).contiguous()
    gens = make_seed_streams(11, R, q.device)
    q2, *_ , n_ev, deaths, births = birth_death_full_state(
        q, v, f, score, anc, gens, fr_rate=50.0, dt_eff=0.05,
        max_event_fraction=0.5, integrator=_integrator())
    assert q2.shape == q.shape
    for r in range(R):
        if deaths[r] is None:
            continue
        # a birth source must never itself be one of the dead slots (it would alias)
        assert not bool(torch.isin(births[r], deaths[r]).any())


def test_fresh_momenta_have_the_maxwell_variance():
    A = 22
    masses = np.linspace(1.008, 16.0, A)
    integ = BAOAB(masses, dt=0.001, gamma=1.0, temperature=300.0,
                  force_fn=lambda x: torch.zeros_like(x))
    g = torch.Generator().manual_seed(3)
    v = integ.maxwell((200000, A, 3), g, torch.device("cpu"), torch.float64)
    kT = 0.008314462618 * 300.0
    assert tuple(v.shape) == (200000, A, 3)     # (A,1) sigma must not add a leading axis
    var = (v ** 2).mean(dim=(0, 2)).numpy()
    assert np.abs(var - kT / masses).max() / (kT / masses).max() < 0.02


# ---------------------------------------------------------------- per-seed RNG isolation
def test_seed_streams_are_independent_of_batch_composition():
    """Changing seed 0's data must not alter any other seed's realisation."""
    R, N = 4, 64
    q, v, f, anc = _toy_state(R=R, N=N)
    score = torch.linspace(-2, 2, N).expand(R, N).contiguous()
    integ = _integrator()

    def run(sc):
        gens = make_seed_streams(2024, R, q.device)
        return birth_death_full_state(q, v, f, sc, anc, gens, fr_rate=50.0, dt_eff=0.05,
                                      max_event_fraction=0.5, integrator=integ)

    base = run(score)
    perturbed_score = score.clone()
    perturbed_score[0] = -5.0                       # seed 0 now fires zero deaths
    pert = run(perturbed_score)
    assert int(pert[4][0]) == 0                     # seed 0 really did change
    for r in range(1, R):
        assert torch.equal(base[3][r], pert[3][r]), f"seed {r} ancestors changed"
        assert int(base[4][r]) == int(pert[4][r])


def test_a_seed_is_reproducible_in_isolation():
    """Seed r run alone must match seed r run inside a batch."""
    R, N = 4, 48
    q, v, f, anc = _toy_state(R=R, N=N, seed=5)
    score = torch.stack([torch.linspace(-2, 2, N) * (1 + 0.3 * r) for r in range(R)])
    integ = _integrator()
    full = birth_death_full_state(q, v, f, score, anc, make_seed_streams(7, R, q.device),
                                  fr_rate=50.0, dt_eff=0.05, max_event_fraction=0.5,
                                  integrator=integ)
    r = 2
    gens_alone = [make_seed_streams(7, R, q.device)[r]]
    alone = birth_death_full_state(q[r:r + 1], v[r:r + 1], f[r:r + 1], score[r:r + 1],
                                   anc[r:r + 1], gens_alone, fr_rate=50.0, dt_eff=0.05,
                                   max_event_fraction=0.5, integrator=integ)
    assert torch.equal(full[3][r], alone[3][0])
    assert int(full[4][r]) == int(alone[4][0])


# ---------------------------------------------------------------- non-finite containment
def test_check_finite_fails_fast_and_saves_state(tmp_path):
    x = torch.randn(3, 8, 22, 3)
    x[1, 4, 0, 0] = float("nan")
    with pytest.raises(SeedFailure) as e:
        check_finite(1234, ("q", x), dump_dir=str(tmp_path), tag="unit")
    assert e.value.seed_index == 1
    assert e.value.step == 1234
    assert os.path.exists(e.value.dump_path)
    d = np.load(e.value.dump_path)
    assert int(d["seed_index"]) == 1 and str(d["failing_tensor"]) == "q"


def test_check_finite_catches_inf_which_clamp_would_hide():
    x = torch.randn(2, 4, 22, 3)
    x[0, 0, 0, 0] = float("inf")
    with pytest.raises(SeedFailure):
        check_finite(0, ("f", x))
    # the old guard would have silently sanitised this
    assert torch.isfinite(torch.clamp(x, -1e3, 1e3)).all()


def test_check_finite_passes_clean_state():
    check_finite(0, ("q", torch.randn(2, 4, 22, 3)), ("v", torch.randn(2, 4, 22, 3)))


# ---------------------------------------------------------------- projection / frozen bias
@pytest.mark.parametrize("n", [35, 48, 97])
def test_project_bias_guarantees_gB_equals_grad_B(n):
    torch.manual_seed(0)
    g1c, g2c, dz1, dz2 = d2.torus_grid(n, n)
    K1, K2 = d2.kernels(g1c, g2c, 0.15, 0.15)
    count = torch.rand(2, n, n) * 50
    f1 = torch.randn(2, n, n) * count
    f2 = torch.randn(2, n, n) * count
    B, gB1, gB2, _, _ = pj.project_bias(f1, f2, count, K1, K2, dz1, dz2, min_count=5.0)
    s1, s2 = ps.spectral_gradient(B, dz1, dz2)
    assert (gB1 - s1).abs().max() < 1e-10
    assert (gB2 - s2).abs().max() < 1e-10
    assert ps.curl_norm(gB1, gB2, dz1, dz2).max() < 1e-10


def test_online_and_frozen_apply_the_same_field():
    """The frozen run re-differentiates the SAVED B; it must match the online gB exactly."""
    n = 97
    torch.manual_seed(1)
    g1c, g2c, dz1, dz2 = d2.torus_grid(n, n)
    K1, K2 = d2.kernels(g1c, g2c, 0.15, 0.15)
    count = torch.rand(1, n, n) * 50
    B, gB1, gB2, _, _ = pj.project_bias(torch.randn(1, n, n) * count,
                                        torch.randn(1, n, n) * count,
                                        count, K1, K2, dz1, dz2, min_count=5.0)
    fz1, fz2 = ps.spectral_gradient(B, dz1, dz2)          # what run_frozen_bias would apply
    phi = torch.rand(1, 500) * 2 * PI - PI
    psi = torch.rand(1, 500) * 2 * PI - PI
    on = pj.bias_at_particles(gB1, gB2, g1c, g2c, dz1, dz2, phi, psi, clip=200.0)
    fr = pj.bias_at_particles(fz1, fz2, g1c, g2c, dz1, dz2, phi, psi, clip=200.0)
    assert (on[0] - fr[0]).abs().max() < 1e-10
    assert (on[1] - fr[1]).abs().max() < 1e-10


def test_magnitude_clipping_preserves_direction():
    b1 = torch.tensor([[300.0, 10.0]])
    b2 = torch.tensor([[400.0, 0.0]])
    c1, c2 = pj.clip_magnitude(b1, b2, 200.0)
    assert abs(math.hypot(c1[0, 0].item(), c2[0, 0].item()) - 200.0) < 1e-9
    assert abs(c1[0, 0].item() / c2[0, 0].item() - 300.0 / 400.0) < 1e-9   # direction kept
    assert abs(c1[0, 1].item() - 10.0) < 1e-12                            # below clip untouched


def test_even_grid_is_rejected_for_this_study():
    with pytest.raises(ValueError, match="Nyquist"):
        pj.require_odd_grid(96)
    assert pj.require_odd_grid(97) == 97


# ---------------------------------------------------------------- leakage
def test_no_reference_leakage_is_structural():
    ref = np.zeros((97, 97))
    for m in ("abf", "fr_estimated", "fr_uniform"):
        assert_no_reference_leakage(m, None)
        with pytest.raises(AssertionError, match="LEAKAGE"):
            assert_no_reference_leakage(m, ref)
    assert_no_reference_leakage("fr_oracle", ref)
    with pytest.raises(ValueError):
        assert_no_reference_leakage("fr_oracle", None)
    with pytest.raises(ValueError):
        assert_no_reference_leakage("not_a_method", None)


# ---------------------------------------------------------------- BAOAB sanity
def test_baoab_samples_the_target_temperature(sysmin):
    """Thermostat gate: <T> within 2% of 300 K for a free particle set."""
    _, _, tff = sysmin
    masses = tff.masses.numpy()
    integ = BAOAB(masses, dt=0.001, gamma=10.0, temperature=300.0,
                  force_fn=lambda x: torch.zeros_like(x))
    g = torch.Generator().manual_seed(0)
    q = torch.zeros(1, 400, 22, 3)
    v = integ.maxwell((1, 400, 22, 3), g, q.device, q.dtype)
    f = torch.zeros_like(q)
    Ts = []
    for s in range(400):
        q, v, f = integ.step(q, v, f, g)
        if s > 100:
            Ts.append(integ.kinetic_temperature(v).item())
    assert abs(np.mean(Ts) - 300.0) / 300.0 < 0.02
