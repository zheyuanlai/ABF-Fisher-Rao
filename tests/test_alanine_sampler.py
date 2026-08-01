"""Sampler-level correctness tests for the corrected 2-D alanine ABF + oracle-mFR arms.

Covers the requirements that need a running sampler (the rest live in
``test_alanine_stage0.py`` and ``test_alanine_categoryA.py``):

  * ``abf`` == ``fr_oracle`` when ``fr_rate = 0``          (the arms differ ONLY by birth-death)
  * ``abf`` == ``fr_oracle`` before ``fr_start_steps``
  * common random numbers match across arms
  * a non-finite replica cannot contaminate another seed
  * basin assignment is periodic and matches the accepted reference minima
  * union-block and dense den Otter mean forces agree to float64 tolerance
  * no-reference-leakage is structural

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alanine_sampler.py -q
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

from alanine.basins import BasinMap, from_reference                      # noqa: E402
from alanine.core2d_ala import (AlaSimConfig, assert_no_reference_leakage,  # noqa: E402
                                run_sampler_ala, sanitize_reference)
from alanine.cv2d import BackboneCV2D, FastBackboneCV2D                  # noqa: E402
from alanine.dynamics import SeedFailure                                 # noqa: E402
from alanine.forcefield import TorchFF, extract_parameters               # noqa: E402
from alanine.system import PHI_ATOMS, PSI_ATOMS, reference_minimum       # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "results", "alanine", "reference", "reference.npz")
HAVE_REF = os.path.exists(REF)
needs_ref = pytest.mark.skipif(not HAVE_REF, reason="accepted reference artifact not present")


@pytest.fixture(scope="module")
def rig():
    system, X0 = reference_minimum()
    P = extract_parameters(system)
    tff = TorchFF(P, device="cpu")
    cv = BackboneCV2D(PHI_ATOMS, PSI_ATOMS, n_atoms=22)
    if HAVE_REF:
        bm, meta = from_reference(REF)
    else:
        n = 97
        F = np.zeros((n, n))
        bm = BasinMap(F, np.ones((n, n), bool), 2.4943)
        meta = {"kT_kJ": 2.4943}
    return tff, cv, X0, bm, meta


def _init(X0, R, N, seed=3):
    """Tightly clustered start (all walkers at C7eq) -- the primary study initialisation."""
    g = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(np.repeat(X0[None], R * N, 0)).reshape(R, N, 22, 3)
    return x + 0.002 * torch.randn(R, N, 22, 3, generator=g)


def _init_dispersed(X0, R, N, seed=5):
    """Walkers spread over the torus by rigid phi/psi rotation.

    Needed to exercise birth-death at all: the centred FR score is *identically zero* when every
    walker sits at the same (phi, psi), because no particle is then over- or under-represented
    relative to any other.  That is correct fixed-N mFR behaviour, not an inert code path.
    """
    from alanine.system import seed_umbrella_lattice, window_centers
    rng = np.random.default_rng(seed)
    cen = window_centers(8)
    seeds = seed_umbrella_lattice(X0, cen)
    idx = rng.integers(0, len(seeds), size=R * N)
    x = torch.as_tensor(seeds[idx]).reshape(R, N, 22, 3)
    return x + 0.002 * torch.as_tensor(rng.standard_normal((R, N, 22, 3)))


def _cfg(**kw):
    base = dict(n_steps=60, n_replicas=16, save_every=30, abf_warmup_steps=20,
                project_every=20, fr_start_steps=20, fr_every=10, fr_rate=5.0,
                rng_seed=1234, max_event_fraction=0.5, lineage_reset_steps=0)
    base.update(kw)
    return AlaSimConfig(**base)


# ------------------------------------------------------------------ arm equivalence
@needs_ref
def test_arms_identical_when_fr_rate_is_zero(rig):
    """The two arms must differ ONLY by birth-death; with rate 0 they are the same run."""
    tff, cv, X0, bm, _ = rig
    F = np.load(REF)["F"]
    lab = bm.label_tensor()
    init = _init(X0, 2, 16)
    sim = _cfg(fr_rate=0.0)
    a = run_sampler_ala("abf", tff, cv, sim, [0, 1], init, lab, "cpu", verbose=False)
    b = run_sampler_ala("fr_oracle", tff, cv, sim, [0, 1], init, lab, "cpu",
                        reference_F=F, verbose=False)
    assert int(b["total_events"].sum()) == 0
    assert np.abs(a["final_pmf"] - b["final_pmf"]).max() < 1e-12
    np.testing.assert_array_equal(a["first_hit"], b["first_hit"])


@needs_ref
def test_arms_identical_before_fr_start(rig):
    """Before fr_start_steps no birth-death can have occurred, so the states must match."""
    tff, cv, X0, bm, _ = rig
    F = np.load(REF)["F"]
    lab = bm.label_tensor()
    init = _init(X0, 2, 16)
    sim = _cfg(n_steps=30, fr_start_steps=10 ** 9, save_every=30)
    a = run_sampler_ala("abf", tff, cv, sim, [0, 1], init, lab, "cpu", verbose=False)
    b = run_sampler_ala("fr_oracle", tff, cv, sim, [0, 1], init, lab, "cpu",
                        reference_F=F, verbose=False)
    assert np.abs(a["final_pmf"] - b["final_pmf"]).max() < 1e-12
    assert int(b["total_events"].sum()) == 0


@needs_ref
def test_common_random_numbers_across_arms(rig):
    """The dynamical noise stream is method-independent: identical trajectories pre-FR."""
    tff, cv, X0, bm, _ = rig
    F = np.load(REF)["F"]
    lab = bm.label_tensor()
    init = _init(X0, 1, 12)
    sim = _cfg(n_steps=25, n_replicas=12, save_every=25, fr_start_steps=10 ** 9)
    a = run_sampler_ala("abf", tff, cv, sim, [0], init, lab, "cpu", verbose=False)
    b = run_sampler_ala("fr_oracle", tff, cv, sim, [0], init, lab, "cpu",
                        reference_F=F, verbose=False)
    # identical noise => identical basin occupancy history, not merely similar
    np.testing.assert_allclose(a["basin_frac"], b["basin_frac"], atol=0, rtol=0)


# ------------------------------------------------------------------ containment
@needs_ref
def test_nonfinite_replica_cannot_contaminate_other_seeds(rig, tmp_path):
    """A NaN injected into seed 1 must abort with seed 1 identified, and never silently pass."""
    tff, cv, X0, bm, _ = rig
    lab = bm.label_tensor()
    init = _init(X0, 3, 8)
    init[1, 0, 0, 0] = float("nan")
    sim = _cfg(n_steps=20, n_replicas=8, save_every=10)
    with pytest.raises(SeedFailure) as e:
        run_sampler_ala("abf", tff, cv, sim, [0, 1, 2], init, lab, "cpu",
                        dump_dir=str(tmp_path), verbose=False)
    assert e.value.seed_index == 1
    assert e.value.dump_path is not None and os.path.exists(e.value.dump_path)
    d = np.load(e.value.dump_path)
    assert int(d["seed_index"]) == 1
    # the other seeds' accumulator entries must be finite in the dump
    assert np.isfinite(d["csum"]).all()


# ------------------------------------------------------------------ basins
@needs_ref
def test_basin_assignment_is_periodic_and_matches_reference(rig):
    tff, cv, X0, bm, meta = rig
    assert "C7eq" in bm.index and "C7ax" in bm.index, bm.names
    # the accepted reference minima must land in their own basins
    for name, (pd, sd) in zip(bm.names, bm.centres_deg):
        lab = bm.assign_np(np.radians(pd), np.radians(sd))
        assert int(lab) == bm.index[name]
    # C7eq near (-74,+56), C7ax near (+63,-48)
    assert bm.assign_np(np.radians(-74.2), np.radians(55.7)) == bm.index["C7eq"]
    assert bm.assign_np(np.radians(63.1), np.radians(-48.2)) == bm.index["C7ax"]
    # periodic seam: +180 and -180 are the same column
    a = bm.assign_np(np.radians(179.9), np.radians(10.0))
    b = bm.assign_np(np.radians(-180.0), np.radians(10.0))
    assert a == b
    # wrapping by a full turn changes nothing
    assert bm.assign_np(np.radians(-74.2) + 2 * math.pi,
                        np.radians(55.7) - 2 * math.pi) == bm.index["C7eq"]


@needs_ref
def test_basin_populations_match_the_reference(rig):
    tff, cv, X0, bm, meta = rig
    F = np.load(REF)["F"]
    pops = bm.population(F)
    assert abs(pops["C7ax"] - 0.0311) < 0.005, pops     # reference C7ax box population
    assert pops["C7eq"] > 0.5


# ------------------------------------------------------------------ union-block CV
def test_union_block_mean_force_matches_dense(rig):
    """FastBackboneCV2D restricts to the 5-atom union; the maths must be unchanged."""
    tff, cv, X0, _, _ = rig
    fast = FastBackboneCV2D(PHI_ATOMS, PSI_ATOMS, n_atoms=22)
    assert fast.union == [4, 6, 8, 14, 16] and fast.nc == 15
    rng = np.random.default_rng(0)
    x = torch.as_tensor(X0[None] + 0.03 * rng.standard_normal((64, 22, 3)))
    F = tff.forces(x)
    beta = 1.0 / (0.008314462618 * 300.0)
    fs, ps_, gs, geos = cv.local_mean_force(x, F, beta)
    ff, pf, gf, geof = fast.local_mean_force(x, F, beta)
    assert (fs - ff).abs().max() < 1e-9
    assert (geos["G"] - geof["G"]).abs().max() == 0.0
    assert (geos["div_v"] - geof["div_v"]).abs().max() < 1e-12
    c1, c2 = torch.randn(64), torch.randn(64)
    dense = (c1[:, None, None] * gs[:, 0] + c2[:, None, None] * gs[:, 1])
    assert (dense - fast.scatter_bias(gf, c1, c2, 22)).abs().max() == 0.0


# ------------------------------------------------------------------ leakage
def test_no_reference_leakage_structural():
    ref = np.zeros((97, 97))
    assert_no_reference_leakage("abf", None)
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_reference_leakage("abf", ref)
    assert_no_reference_leakage("fr_oracle", ref)
    with pytest.raises(ValueError):
        assert_no_reference_leakage("fr_oracle", None)
    with pytest.raises(ValueError):
        assert_no_reference_leakage("fr_estimated", ref)     # not an arm of this study


# ------------------------------------------------------------------ birth-death fires
@needs_ref
def test_birth_death_actually_fires_and_stays_bounded(rig):
    """Guard against a silently inert FR path, and against runaway replacement."""
    tff, cv, X0, bm, _ = rig
    F = np.load(REF)["F"]
    lab = bm.label_tensor()
    init = _init_dispersed(X0, 2, 32)
    sim = _cfg(n_steps=80, n_replicas=32, fr_start_steps=10, fr_every=5,
               fr_rate=200.0, max_event_fraction=0.05, save_every=40)
    out = run_sampler_ala("fr_oracle", tff, cv, sim, [0, 1], init, lab, "cpu",
                          reference_F=F, verbose=False)
    assert int(out["total_events"].sum()) > 0, "FR path never fired"
    n_opp = (sim.n_steps - sim.fr_start_steps) // sim.fr_every + 1
    assert out["total_events"].max() <= sim.max_event_fraction * sim.n_replicas * n_opp


@needs_ref
def test_degenerate_ensemble_gives_exactly_zero_score():
    """Fixed-N mFR cannot reallocate an ensemble that occupies a single CV point.

    Tested directly on the score rather than through the sampler: the centred log-ratio is
    identically zero when every walker sits at the same (phi, psi), because no particle is then
    over- or under-represented relative to any other.  (Through the sampler this only holds at
    step 0 -- independent Langevin noise separates the walkers within a few steps, which is why
    an earlier version of this test appeared to pass for the wrong reason, on a NaN score.)
    """
    from alkanes import density2d as d2
    from alanine.core2d_ala import _oracle_target
    kT = 2.4943387854
    n = 97
    g1, g2, dz1, dz2 = d2.torus_grid(n, n)
    Kk1, Kk2 = d2.kernels(g1, g2, 0.15, 0.15)
    F = sanitize_reference(torch.as_tensor(np.load(REF)["F"]), kT)
    z1 = torch.full((1, 64), -1.295)                 # every walker at the same point
    z2 = torch.full((1, 64), 0.972)
    p_hat = d2.kde2(z1, z2, Kk1, Kk2, n, n, dz1, dz2)
    q_t = _oracle_target(F, torch.zeros(1, n, n), 1.0 / kT, dz1, dz2)
    score, _ = d2.fr_score_2d(z1, z2, p_hat, q_t, g1, g2, dz1, dz2, 2.0)
    assert torch.isfinite(score).all()
    assert float(score.abs().max()) < 1e-12
    # and a genuinely spread ensemble must produce a non-degenerate score
    z1b = torch.linspace(-3.0, 3.0, 64)[None]
    z2b = torch.linspace(-3.0, 3.0, 64)[None].flip(-1)
    p_b = d2.kde2(z1b, z2b, Kk1, Kk2, n, n, dz1, dz2)
    sb, _ = d2.fr_score_2d(z1b, z2b, p_b, q_t, g1, g2, dz1, dz2, 2.0)
    assert torch.isfinite(sb).all() and float(sb.abs().max()) > 1e-3
    assert float(sb.max()) > 0 and float(sb.min()) < 0      # centred: both signs present


# ------------------------------------------------------------------ oracle target sanity
@needs_ref
def test_reference_has_infinite_cells_and_must_be_sanitised():
    """Regression for a silent, study-voiding bug.

    The accepted reference stores +inf in the 226 of 9409 bins no umbrella window visited.
    ``F - F.mean()`` is then NaN everywhere, so the oracle target is NaN, the FR score is NaN,
    no birth or death weight is ever positive, and ``fr_oracle`` silently degenerates into
    ``abf`` reporting zero events -- which would have been read as "mFR is EQUIVALENT to ABF"
    when the mechanism had never been switched on at all.
    """
    F = np.load(REF)["F"]
    assert (~np.isfinite(F)).sum() > 0, "this regression only bites when empty bins exist"
    assert not np.isfinite(np.asarray(F).mean())          # the trap
    San = sanitize_reference(torch.as_tensor(F), 2.4943387854)
    assert torch.isfinite(San).all()
    assert abs(float(San.mean())) < 1e-9                   # zero-mean on the sanitised grid


@needs_ref
def test_oracle_target_is_finite_and_normalised(rig):
    from alkanes import density2d as d2
    from alanine.core2d_ala import _oracle_target
    kT = 2.4943387854
    F = sanitize_reference(torch.as_tensor(np.load(REF)["F"]), kT)
    n = 97
    g1, g2, dz1, dz2 = d2.torus_grid(n, n)
    q = _oracle_target(F, torch.zeros(2, n, n), 1.0 / kT, dz1, dz2)
    assert torch.isfinite(q).all() and (q >= 0).all()
    assert abs(float((q.sum(dim=(-2, -1)) * dz1 * dz2)[0]) - 1.0) < 1e-10
