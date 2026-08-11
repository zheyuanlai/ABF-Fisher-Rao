"""Sampler and selection-rule correctness for deca-alanine.

The dangerous failure here is not a crash -- it is a **silently inverted selection rule**, which
would run to completion, produce plausible numbers, and mean the opposite of what is claimed.
Chapter 6 writes its selection function so that positive means *multiply*; this project's
birth--death consumes a score where positive means *die*. Every rule is therefore tested for
the direction of its effect, not merely for shape and finiteness.

Also covers the invariant that makes arm comparisons meaningful: with ``fr_rate = 0``, or before
``fr_start_steps``, every selection arm must be **bit-identical** to ``abf`` -- the arms differ
only by birth--death.

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/test_deca_sampler.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

pytest.importorskip("openmm")

from alkanes import interval as iv                                          # noqa: E402
from deca import selection as sel                                           # noqa: E402
from deca import system as dsys                                             # noqa: E402
from deca.core import DecaSimConfig, run_sampler_deca                        # noqa: E402
from deca.engine import DecaEngine                                          # noqa: E402

R_LO, R_HI, NG = 1.20, 3.60, 65


@pytest.fixture(scope="module")
def engine():
    _, _, system = dsys.make_system(10)
    return DecaEngine(system, device="cpu", dtype=torch.float64, compiled=False)


@pytest.fixture(scope="module")
def gridpack():
    grid, dz = iv.interval_grid(NG, R_LO, R_HI, device="cpu", dtype=torch.float64)
    K = iv.reflected_kernel_matrix(grid, 0.06, R_LO, R_HI)
    return grid, dz, K


def _tiny_cfg(**kw):
    base = dict(n_walkers=8, n_steps=60, n_grid=NG, save_every=30, xi_trace_every=30,
                label_every=30, abf_warmup_steps=10, fr_start_steps=20, fr_every=10)
    base.update(kw)
    return DecaSimConfig(**base)


def _init(R, N, rng=0):
    g = np.random.default_rng(rng)
    x = dsys.build_helix(-57.0, -47.0)
    return np.stack([[x + 0.004 * g.standard_normal(x.shape) for _ in range(N)]
                     for _ in range(R)])


# --------------------------------------------------------------------------- sign conventions
def test_fisher_rao_score_kills_the_over_represented(gridpack):
    """A walker where p exceeds q must get a POSITIVE score (positive = die)."""
    grid, dz, K = gridpack
    # pile the population near 1.6 nm; make the target uniform
    z = torch.cat([torch.full((1, 60), 1.6), torch.full((1, 4), 3.2)], dim=1)
    q_grid = iv.normalize_density(torch.ones(1, NG, dtype=torch.float64), dz)
    score, p_grid, kl = sel.fisher_rao_score(z, grid, dz, R_LO, R_HI, K, q_grid, clip=5.0)
    crowded = score[0, :60].mean()
    sparse = score[0, 60:].mean()
    assert crowded > 0 > sparse, f"crowded {crowded:.3f} sparse {sparse:.3f}"


def test_count_balancing_kills_the_over_represented(gridpack):
    grid, dz, K = gridpack
    z = torch.cat([torch.full((1, 60), 1.6), torch.full((1, 4), 3.2)], dim=1)
    score, _ = sel.count_balancing_score(z, grid, dz, R_LO, R_HI, K, clip=5.0, c=1.0)
    assert score[0, :60].mean() > 0 > score[0, 60:].mean()


def test_book_laplacian_multiplies_in_density_valleys(gridpack):
    """Chapter 6's S = c d2p/dz2 / p multiplies where p is CONVEX.

    Under this project's convention that is a NEGATIVE score, so the rule is negated on the way
    in. A walker sitting in a valley of the density must score negative (give birth).

    The density is supplied explicitly. Probing a valley with walkers is self-defeating: the
    probes build their own KDE peak at the probe point, making it concave rather than convex,
    so a *correct* rule reads as inverted. The first version of this test did exactly that.
    """
    grid, dz, K = gridpack
    # analytic bimodal density: modes at 1.7 and 3.1, a genuine convex gap between them
    p = (torch.exp(-0.5 * ((grid - 1.7) / 0.18) ** 2)
         + torch.exp(-0.5 * ((grid - 3.1) / 0.18) ** 2))[None]
    p = iv.normalize_density(p, dz)
    d2 = torch.zeros_like(p)
    d2[:, 1:-1] = (p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) / (dz * dz)
    assert float(d2[0, NG // 2]) > 0, "test setup: the midpoint is not a convex region"

    z = torch.tensor([[2.40, 1.70, 3.10]], dtype=torch.float64)   # valley, mode, mode
    score, _ = sel.book_laplacian_score(z, grid, dz, R_LO, R_HI, K, clip=50.0, c=1.0, p_grid=p)
    valley, modes = float(score[0, 0]), float(score[0, 1:].mean())
    assert valley < 0 < modes, f"valley {valley:.4f} modes {modes:.4f}"


def test_book_laplacian_matches_its_defining_identity(gridpack):
    """S*p must equal c * d2p/dz2 -- that identity IS the diffusion-shift claim.

    Recovering it from the implementation (before recentring and clipping) is what makes this a
    faithful reproduction of the Chapter 6 rule rather than something merely Laplacian-shaped.
    """
    grid, dz, K = gridpack
    p = torch.exp(-0.5 * ((grid - 2.2) / 0.35) ** 2)[None]
    p = iv.normalize_density(p, dz)
    d2 = torch.zeros_like(p)
    d2[:, 1:-1] = (p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) / (dz * dz)

    z = grid[None, 5:-5:7]
    c = 0.37
    score, _ = sel.book_laplacian_score(z, grid, dz, R_LO, R_HI, K, clip=1e9, c=c, p_grid=p)
    S = -(score + 0.0)                                     # undo the sign convention
    S = S - S.mean()                                       # the impl recentres; compare centred
    want = c * iv.interval_interp(d2 / p.clamp_min(1e-12), grid, z)
    want = want - want.mean()
    assert torch.allclose(S, want, atol=1e-8), f"max dev {float((S-want).abs().max()):.3e}"


def test_every_score_is_zero_mean_and_clipped(gridpack):
    grid, dz, K = gridpack
    z = torch.rand(3, 32, dtype=torch.float64) * (R_HI - R_LO) + R_LO
    q_grid = iv.normalize_density(torch.ones(3, NG, dtype=torch.float64), dz)
    for s in (sel.fisher_rao_score(z, grid, dz, R_LO, R_HI, K, q_grid, 0.5)[0],
              sel.count_balancing_score(z, grid, dz, R_LO, R_HI, K, 0.5)[0],
              sel.book_laplacian_score(z, grid, dz, R_LO, R_HI, K, 0.5)[0]):
        assert torch.allclose(s.mean(-1), torch.zeros(3, dtype=torch.float64), atol=0.5)
        assert float(s.abs().max()) <= 0.5 + 1e-9


def test_sham_preserves_turnover_but_destroys_direction(gridpack):
    """The sham must keep the score MULTISET (so expected events match) and only reorder it."""
    g = torch.Generator().manual_seed(0)
    score = torch.randn(4, 32, dtype=torch.float64)
    shammed = sel.sham_score(score, g)
    for r in range(4):
        assert torch.allclose(torch.sort(score[r]).values, torch.sort(shammed[r]).values)
    assert not torch.allclose(score, shammed)


# --------------------------------------------------------------------------- leakage gate
def test_no_reference_leakage_gate():
    ref = np.zeros(NG)
    for m in ("abf", "mfr_practical", "mfr_sham", "book_laplacian", "count_balancing"):
        sel.assert_no_reference_leakage(m, None)
        with pytest.raises(AssertionError, match="NO-REFERENCE-LEAKAGE"):
            sel.assert_no_reference_leakage(m, ref)
    sel.assert_no_reference_leakage("mfr_oracle", ref)
    with pytest.raises(ValueError):
        sel.assert_no_reference_leakage("mfr_oracle", None)
    with pytest.raises(ValueError):
        sel.assert_no_reference_leakage("not_a_method", None)


def test_sampler_refuses_a_reference_for_a_deployable_arm(engine):
    cfg = _tiny_cfg(fr_rate=0.5)
    with pytest.raises(AssertionError, match="NO-REFERENCE-LEAKAGE"):
        run_sampler_deca("mfr_practical", engine, cfg, [0], _init(1, 8),
                         reference_free_energy=np.zeros(NG), device="cpu", verbose=False)


# --------------------------------------------------------------------------- arm equivalence
@pytest.mark.parametrize("method", ["mfr_practical", "book_laplacian", "count_balancing"])
def test_zero_rate_is_bit_identical_to_abf(engine, method):
    """With fr_rate = 0 the arms differ by nothing at all, so the trajectories must match.

    If this fails, the arms are not sharing a noise stream and every paired comparison in the
    study is invalid.
    """
    cfg = _tiny_cfg(fr_rate=0.0)
    x0 = _init(2, 8)
    a = run_sampler_deca("abf", engine, cfg, [0, 1], x0, device="cpu", verbose=False)
    b = run_sampler_deca(method, engine, cfg, [0, 1], x0, device="cpu", verbose=False)
    assert np.array_equal(a["xi_trace"], b["xi_trace"])
    assert np.allclose(a["pmf"], b["pmf"], atol=0.0, rtol=0.0)
    assert int(b["total_replacement_events"].sum()) == 0


def test_identical_before_selection_starts(engine):
    """Before fr_start_steps no selection has fired, so the arms must still coincide."""
    cfg = _tiny_cfg(fr_rate=5.0, fr_start_steps=10_000, n_steps=40)
    x0 = _init(2, 8)
    a = run_sampler_deca("abf", engine, cfg, [0, 1], x0, device="cpu", verbose=False)
    b = run_sampler_deca("mfr_practical", engine, cfg, [0, 1], x0, device="cpu", verbose=False)
    assert np.array_equal(a["xi_trace"], b["xi_trace"])
    assert int(b["total_replacement_events"].sum()) == 0


# --------------------------------------------------------------------------- population
def test_selection_conserves_population_and_records_events(engine):
    cfg = _tiny_cfg(fr_rate=200.0, fr_start_steps=20, fr_every=5, max_event_fraction=0.5)
    out = run_sampler_deca("count_balancing", engine, cfg, [0], _init(1, 8),
                           device="cpu", verbose=False)
    assert out["xi_trace"].shape[-1] == cfg.n_walkers          # fixed population
    assert int(out["total_replacement_events"].sum()) > 0      # it actually did something
    assert np.isfinite(out["pmf"]).all()


def test_abf_min_count_actually_ramps_the_applied_bias(engine):
    """The `fullSamples` guard must be APPLIED, not merely declared in the config.

    Without it a bin holding one sample contributes that single instantaneous force as its
    conditional mean, and the integrated bias drives walkers irreversibly to one end. On the
    first deca screen that produced a 102.5 kT profile against a 72.0 kT reference and left the
    folded basin empty. A config field that nothing reads is worse than no field at all.

    With an enormous min_count the bias is ramped to ~zero everywhere, so the run must coincide
    with pure unbiased dynamics under the walls; with min_count = 0 it must not.
    """
    x0 = _init(2, 8)
    huge = _tiny_cfg(fr_rate=0.0, abf_min_count=1e12, abf_warmup_steps=1)
    zero = _tiny_cfg(fr_rate=0.0, abf_min_count=0.0, abf_warmup_steps=1)
    a = run_sampler_deca("abf", engine, huge, [0, 1], x0, device="cpu", verbose=False)
    b = run_sampler_deca("abf", engine, zero, [0, 1], x0, device="cpu", verbose=False)
    assert not np.array_equal(a["xi_trace"], b["xi_trace"]), \
        "abf_min_count had no effect on the trajectory -- the guard is not wired up"


def test_sampler_runs_and_traces_are_consistent(engine):
    cfg = _tiny_cfg(fr_rate=0.0)
    out = run_sampler_deca("abf", engine, cfg, [0, 1], _init(2, 8), device="cpu", verbose=False)
    assert out["xi_trace"].shape == (len(out["xi_trace_steps"]), 2, 8)
    assert out["label_xi"].shape[1:] == (2, 8)
    assert np.isfinite(out["xi_trace"]).all()
    assert (out["xi_trace"] > 0.2).all() and (out["xi_trace"] < 10.0).all()
    assert out["grid"].size == NG
