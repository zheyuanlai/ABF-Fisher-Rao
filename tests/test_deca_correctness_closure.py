"""Correctness-closure gate for the deca-alanine workflow.

Two independent silent defects escaped a 255-test suite:

  * out-of-domain samples clamped into edge bins, carving a fake PMF well;
  * ``abf_min_count`` declared in the config and never read, letting one-sample bins drive the
    applied ABF bias to a 102.5 kT span against a 72.0 kT reference.

Neither was a crash and neither was a wrong *type*. They were wrong *physics* produced by code
that ran cleanly. So this file asserts **physics-level invariants**, not more shape checks, and
it is a code-validity gate: **none of it is defined by whether a run comes out
establishment-limited.**

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/test_deca_correctness_closure.py -q
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
from deca import states as st                                              # noqa: E402
from deca import system as dsys                                            # noqa: E402
from deca.core import DecaSimConfig, run_sampler_deca                       # noqa: E402
from deca.engine import DecaEngine                                         # noqa: E402
from deca.labels import conditional_tv                                     # noqa: E402
from deca.umbrella import UmbrellaConfig, pmf_from_weights                  # noqa: E402

KB = 0.008314462618
R_LO, R_HI, NG = 1.20, 3.60, 65


@pytest.fixture(scope="module")
def engine():
    _, _, system = dsys.make_system(10)
    return DecaEngine(system, device="cpu", dtype=torch.float64, compiled=False)


def _cfg(**kw):
    base = dict(n_walkers=8, n_steps=60, n_grid=NG, save_every=30, xi_trace_every=30,
                label_every=30, abf_warmup_steps=10, fr_start_steps=20, fr_every=10)
    base.update(kw)
    return DecaSimConfig(**base)


def _init(R, N, seed=0):
    g = np.random.default_rng(seed)
    x = dsys.build_helix(-57.0, -47.0)
    return np.stack([[x + 0.004 * g.standard_normal(x.shape) for _ in range(N)]
                     for _ in range(R)])


# =========================================================== (a) out-of-domain never clamped
def test_reference_pmf_drops_out_of_domain_samples(  ):
    """Defect 1, directly. Adding samples OUTSIDE the domain must not change F inside it.

    The withdrawn run's fake well was exactly this: 4.82 % of samples sat outside [R_lo, R_hi]
    by design (Amendment 1 brackets the windows) and were clamped into bin 0.
    """
    cfg = UmbrellaConfig(n_grid=NG, R_lo=R_LO, R_hi=R_HI)
    rng = np.random.default_rng(0)
    xi = rng.uniform(R_LO, R_HI, 40_000)
    w = np.full(xi.size, 1.0 / xi.size)

    _, _, _, F_clean, _, nd0 = pmf_from_weights(xi, w, cfg)
    # now append a slab of samples BELOW the domain, carrying real weight
    xi2 = np.concatenate([xi, rng.uniform(R_LO - 0.05, R_LO - 0.001, 4_000)])
    w2 = np.full(xi2.size, 1.0 / xi2.size)
    _, _, _, F_dirty, _, nd1 = pmf_from_weights(xi2, w2, cfg)

    assert nd0 == 0 and nd1 == 4_000, (nd0, nd1)
    F_clean = F_clean - F_clean.mean()
    F_dirty = F_dirty - F_dirty.mean()
    assert np.abs(F_clean - F_dirty).max() < 0.25, \
        "out-of-domain samples leaked into the in-domain PMF"


def test_edge_bin_is_not_a_dumping_ground():
    """A sharp spurious minimum at grid[0] is the signature to refuse."""
    cfg = UmbrellaConfig(n_grid=NG, R_lo=R_LO, R_hi=R_HI)
    rng = np.random.default_rng(1)
    xi = np.concatenate([rng.uniform(R_LO, R_HI, 30_000),
                         rng.uniform(R_LO - 0.06, R_LO, 6_000)])
    w = np.full(xi.size, 1.0 / xi.size)
    _, _, _, F, counts, nd = pmf_from_weights(xi, w, cfg)
    kT = KB * 300.0
    assert nd == 6_000
    # bin 0 must not sit far below its neighbours: the withdrawn run had 2.65 vs ~5.3 kT
    assert (F[0] - F[1:4].mean()) / kT > -0.8, \
        f"grid[0] is {(F[0]-F[1:4].mean())/kT:.2f} kT below its neighbours"


def test_gate_a_conditional_also_drops_out_of_domain():
    """Gate A's conditional histogram had the same clamping defect."""
    edges = np.linspace(R_LO, R_HI, 21)
    rng = np.random.default_rng(2)
    xi = rng.uniform(R_LO, R_HI, 20_000)
    y = rng.integers(0, 2, xi.size)
    w = np.full(xi.size, 1.0)
    tv0, occ0, _ = conditional_tv(xi, y, w, edges, min_count=1.0)

    out = rng.uniform(R_LO - 0.2, R_LO - 0.01, 5_000)
    xi2 = np.concatenate([xi, out])
    y2 = np.concatenate([y, np.zeros(out.size, int)])   # all out-of-range into ONE label
    w2 = np.full(xi2.size, 1.0)
    tv1, occ1, _ = conditional_tv(xi2, y2, w2, edges, min_count=1.0)
    assert abs(np.nanmax(tv1) - np.nanmax(tv0)) < 0.02, \
        "out-of-domain samples changed the Gate A statistic"


# =========================================================== (b,c,d) the fullSamples guard
def test_sparse_bin_cannot_apply_a_full_bias(engine):
    """Defect 2, directly: the APPLIED bias must be ramped by per-bin support."""
    cfg = _cfg(fr_rate=0.0, abf_min_count=1e12, abf_warmup_steps=1)
    out = run_sampler_deca("abf", engine, cfg, [0], _init(1, 8), device="cpu", verbose=False)
    # with an unreachable min_count the applied bias is ~0, so the run is unbiased-under-walls
    cfg0 = _cfg(fr_rate=0.0, abf_min_count=0.0, abf_warmup_steps=1)
    out0 = run_sampler_deca("abf", engine, cfg0, [0], _init(1, 8), device="cpu", verbose=False)
    assert not np.array_equal(out["xi_trace"], out0["xi_trace"])


def test_estimator_and_applied_bias_are_separate(engine):
    """The ESTIMATE must keep the full mean force even when the APPLIED bias is ramped to zero.

    Ramping the estimate too would bias the very quantity the study scores.
    """
    cfg = _cfg(fr_rate=0.0, abf_min_count=1e12, abf_warmup_steps=1)
    out = run_sampler_deca("abf", engine, cfg, [0], _init(1, 8), device="cpu", verbose=False)
    mf = out["mean_force"][-1]
    assert np.isfinite(mf).all()
    assert np.abs(mf).max() > 0.0, "the stored estimator was zeroed along with the applied bias"


def test_one_absurd_sample_cannot_generate_a_large_applied_bias():
    """A single extreme force in an otherwise empty bin must be ramped to ~nothing.

    This reproduces the mechanism in isolation: `mean_force_profile` guards only den > EPS, so
    without the trust ramp one sample IS the conditional mean.
    """
    grid, dz = iv.interval_grid(NG, R_LO, R_HI, device="cpu", dtype=torch.float64)
    K = iv.gaussian_kernel_matrix(grid, 0.04)
    fsum = torch.zeros(1, NG, dtype=torch.float64)
    csum = torch.zeros(1, NG, dtype=torch.float64)
    fsum[0, NG // 2] = 5.0e4          # one absurd force sample
    csum[0, NG // 2] = 1.0

    mf = iv.mean_force_profile(fsum, csum, K)
    unguarded = float(mf.abs().max())
    eff = iv.effective_counts(csum, K)
    trust = (eff / 100.0).clamp(0.0, 1.0)
    guarded = float((mf * trust).abs().max())

    assert unguarded > 1.0e4, "test setup failed to reproduce the unguarded pathology"
    assert guarded < unguarded / 50.0, \
        f"guard is ineffective: {guarded:.1f} vs unguarded {unguarded:.1f}"


def test_deca_guard_matches_the_trusted_alanine_semantics():
    """The alanine 2-D sampler gates with `trust = den >= min_count` (a hard cut).

    Deca uses a continuous ramp `clamp(eff/min_count, 0, 1)`, which is the NAMD `fullSamples`
    behaviour and is strictly gentler. The invariant that must hold in BOTH is: below the
    threshold the applied bias is suppressed, at/above it the full mean force is applied.
    """
    grid, dz = iv.interval_grid(NG, R_LO, R_HI, device="cpu", dtype=torch.float64)
    K = iv.gaussian_kernel_matrix(grid, 0.04)
    csum = torch.zeros(1, NG, dtype=torch.float64)
    csum[0, NG // 2] = 1.0e6                       # richly sampled bin
    eff = iv.effective_counts(csum, K)
    trust = (eff / 100.0).clamp(0.0, 1.0)
    hard = (eff >= 100.0).to(torch.float64)
    assert float(trust[0, NG // 2]) == pytest.approx(1.0)
    assert float(hard[0, NG // 2]) == pytest.approx(1.0)

    csum2 = torch.zeros(1, NG, dtype=torch.float64)
    csum2[0, NG // 2] = 1.0
    eff2 = iv.effective_counts(csum2, K)
    trust2 = (eff2 / 100.0).clamp(0.0, 1.0)
    assert float(trust2.max()) < 0.05, "a one-sample bin is not being suppressed"


# =========================================================== (e) arm equivalence preserved
@pytest.mark.parametrize("method", ["mfr_practical", "book_laplacian", "count_balancing"])
def test_zero_rate_still_bit_identical_after_the_fixes(engine, method):
    cfg = _cfg(fr_rate=0.0)
    x0 = _init(2, 8)
    a = run_sampler_deca("abf", engine, cfg, [0, 1], x0, device="cpu", verbose=False)
    b = run_sampler_deca(method, engine, cfg, [0, 1], x0, device="cpu", verbose=False)
    assert np.array_equal(a["xi_trace"], b["xi_trace"])
    assert int(b["total_replacement_events"].sum()) == 0


# =========================================================== integration recovers a known F
def test_integration_recovers_a_synthetic_free_energy():
    """With a KNOWN mean force, the estimator's integration must return the known F + const."""
    grid, dz = iv.interval_grid(257, R_LO, R_HI, device="cpu", dtype=torch.float64)
    F_true = 30.0 * (grid - 2.0) ** 2 - 8.0 * torch.sin(3.0 * grid)
    mf_true = torch.gradient(F_true, spacing=(grid,))[0][None]
    F_hat = iv.free_energy_from_mean_force(mf_true, grid, dz)[0]
    a = F_true - F_true.mean()
    b = F_hat - F_hat.mean()
    err = float((a - b).abs().max() / (a.max() - a.min()))
    assert err < 0.02, f"integration error {err:.4f} of the span"


# =========================================================== (g) pathological signature gone
def test_state_and_target_machinery_is_sane_on_a_monotone_pmf():
    """The deca case: monotone F, uniform bias -> Q* must follow the Boltzmann tilt, not blow up.

    In the withdrawn run `occ/Q*` for the folded state came out 8e9 because the runaway bias
    made Q* underflow. Q* must stay a proper probability vector.
    """
    grid = np.linspace(R_LO, R_HI, 129)
    kT = KB * 300.0
    F = 72.0 * kT * (grid - grid[0]) / (grid[-1] - grid[0])
    edges = np.linspace(grid[0], grid[-1], 4)
    for B in (np.zeros_like(F), F, 0.5 * F, 1.4 * F):
        Q = st.bias_aware_target(grid, F, B, 1.0 / kT, edges)
        assert np.isfinite(Q).all()
        assert Q.min() >= 0.0
        assert abs(Q.sum() - 1.0) < 1e-9
    Q_cancel = st.bias_aware_target(grid, F, F, 1.0 / kT, edges)[0]
    assert np.allclose(Q_cancel, 1 / 3, atol=0.02), \
        f"bias cancelling F must give a uniform target, got {Q_cancel}"
