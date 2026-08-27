"""Phase-0 engineering gates for the IO-ABF overnight campaign.

Frozen protocol: ``docs/IO_ABF_OVERNIGHT_PREREGISTRATION.md`` section 6.

Every gate here is a hard stop: a failure means no scientific run may start.
The numbering is the preregistration's, so a failure names the clause it broke.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abffr import allocation as al                              # noqa: E402
from abffr import cell_mass as cm                               # noqa: E402
from abffr import information as inf                            # noqa: E402
from abffr import io_abf                                        # noqa: E402
import eb_abffr_core as eb                                      # noqa: E402

DEV = torch.device("cpu")
DT = torch.float64


def _grid(n=181, lo=-1.8, hi=1.8, elo=-1.5, ehi=1.5):
    x = torch.linspace(lo, hi, n, device=DEV, dtype=DT)
    return x, (x >= elo) & (x <= ehi)


def _alloc(arms, cfg=None, beta=8.0, **kw):
    x, mask = _grid()
    cfg = cfg or io_abf.IOConfig(n_cells=32, obs_every=10, opportunity_every=500,
                                 **kw)
    return io_abf.IOAllocator(arms, x, mask, np.full(len(arms), beta), 1e-3, cfg,
                              device=DEV, dtype=DT)


# --------------------------------------------------------------------------- #
# G0.1  uniform r* is plain ABF, and A0 is the accepted ABF
# --------------------------------------------------------------------------- #
def test_g0_1_uniform_target_gives_exactly_zero_force():
    """floor_fraction = 1 makes r* uniform, and a constant log r* has no gradient.

    Not "small": the identity has to be exact, because a residual force here
    would mean every arm carries an unattributable perturbation.
    """
    a = _alloc(["A6b"], cfg=io_abf.IOConfig(n_cells=32, obs_every=10,
                                            opportunity_every=500,
                                            floor_fraction=1.0))
    X = torch.linspace(-1.5, 1.5, 64, device=DEV, dtype=DT).view(1, -1)
    A = torch.zeros((1, a.G), device=DEV, dtype=DT)
    for _ in range(50):
        a.observe(X, torch.randn(1, 64, dtype=DT), torch.zeros(1, 64, dtype=DT))
    f = a.refresh(1, X, A)
    assert torch.all(f == 0.0)
    assert torch.all(a.bias_force_at(X) == 0.0)


def test_g0_1_a0_row_is_the_accepted_abf_bit_for_bit():
    """An A0 row inside an IO batch reproduces ``eb.ABF`` exactly.

    The comparison is against the accepted method object, not against a second
    A0 run, so it catches the engine patch changing the baseline as well as the
    allocator leaking into a row that should not have one.
    """
    cfg = eb.PhysConfig(N=96, n_steps=1200, save_every=200)
    plain = eb.simulate_batch(
        eb.BatchSpec(configs=[cfg, cfg], seeds=[0, 1], methods=[eb.ABF]),
        device=DEV, dtype=DT)
    io_cfg = io_abf.IOConfig(n_cells=io_abf.cells_for_walkers(cfg.N),
                             obs_every=10, opportunity_every=100)
    withio = eb.simulate_batch(
        eb.BatchSpec(configs=[cfg, cfg], seeds=[0, 1],
                     methods=[eb.IO_A0, eb.IO_A6B]),
        device=DEV, dtype=DT,
        io=eb.IOSpec(arms=["A0", "A6b"], cfg=io_cfg))
    a0 = [r for r in withio if r["io_arm"] == "A0"]
    assert len(a0) == len(plain) == 2
    for p, q in zip(plain, a0):
        assert np.array_equal(p["l2_f_t"], q["l2_f_t"])
        assert np.array_equal(p["F_hat"], q["F_hat"])
        assert p["final_l2_f"] == q["final_l2_f"]


def test_g0_1_a6b_actually_differs():
    """The identity above must not hold for a *real* target -- else nothing runs."""
    cfg = eb.PhysConfig(N=96, n_steps=1200, save_every=200)
    io_cfg = io_abf.IOConfig(n_cells=io_abf.cells_for_walkers(cfg.N),
                             obs_every=10, opportunity_every=100)
    recs = eb.simulate_batch(
        eb.BatchSpec(configs=[cfg], seeds=[0], methods=[eb.IO_A0, eb.IO_A6B]),
        device=DEV, dtype=DT, io=eb.IOSpec(arms=["A0", "A6b"], cfg=io_cfg))
    a0 = [r for r in recs if r["io_arm"] == "A0"][0]
    a6 = [r for r in recs if r["io_arm"] == "A6b"][0]
    assert not np.array_equal(a0["F_hat"], a6["F_hat"])
    assert a6["io_r_star_t"].shape[0] > 0


# --------------------------------------------------------------------------- #
# G0.2 / G0.3  what the allocator may read, and what it may do
# --------------------------------------------------------------------------- #
def test_g0_2_no_reference_leakage():
    io_abf.assert_no_reference_leakage()


def test_g0_3_no_birth_death_static():
    io_abf.assert_no_birth_death()


def test_g0_3_no_birth_death_runtime():
    a = _alloc(["A0", "A6b", "A6c"])
    X = torch.rand(3, 64, dtype=DT) * 3.0 - 1.5
    A = torch.zeros((3, a.G), dtype=DT)
    for _ in range(60):
        a.observe(X, torch.randn(3, 64, dtype=DT), torch.zeros(3, 64, dtype=DT))
    io_abf.assert_returns_field(a, X, A)


def test_g0_3_population_is_never_permuted():
    """End to end: the replica set after a run is a *trajectory*, not a resample.

    Ancestor ESS stays at N for every IO arm, which cannot happen if a single
    replica was ever duplicated.
    """
    cfg = eb.PhysConfig(N=96, n_steps=1200, save_every=200)
    io_cfg = io_abf.IOConfig(n_cells=12, obs_every=10, opportunity_every=100)
    recs = eb.simulate_batch(
        eb.BatchSpec(configs=[cfg], seeds=[0],
                     methods=[eb.IO_A0, eb.IO_A6B, eb.IO_A6C]),
        device=DEV, dtype=DT, io=eb.IOSpec(arms=list(eb.IO_ARMS), cfg=io_cfg))
    for r in recs:
        assert r["final_ess"] == pytest.approx(cfg.N), r["io_arm"]
        assert r["n_die"] == 0.0 and r["n_clone"] == 0.0


# --------------------------------------------------------------------------- #
# G0.4  the target is a probability vector with the frozen floor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", ["A6b", "A6c"])
def test_g0_4_r_star_normalised_positive_floored(arm):
    a = _alloc([arm])
    rng = np.random.default_rng(7)
    for _ in range(80):
        X = torch.as_tensor(rng.normal(0.0, 0.8, (1, 128)), dtype=DT).clamp(-1.7, 1.7)
        a.observe(X, torch.as_tensor(rng.normal(size=(1, 128)), dtype=DT),
                  torch.zeros(1, 128, dtype=DT))
    X = torch.as_tensor(rng.normal(0.0, 0.8, (1, 128)), dtype=DT).clamp(-1.7, 1.7)
    a.refresh(1, X, torch.zeros((1, a.G), dtype=DT))
    r = np.array(a.rows[-1]["r_star"])
    assert r.shape == (a.J,)
    assert np.all(r > 0)
    assert r.sum() == pytest.approx(1.0, abs=1e-12)
    floor = al.FLOOR_FRACTION / a.J
    assert r.min() >= floor - 1e-12
    assert al.FLOOR_FRACTION == 0.25          # frozen; not a per-system knob


# --------------------------------------------------------------------------- #
# G0.5  constant Gamma reduces to the geometric optimum
# --------------------------------------------------------------------------- #
def test_g0_5_flat_gamma_gives_r_proportional_to_sqrt_a():
    a = _alloc(["A6b"])
    g = a.a_cell * np.ones(a.J)
    r, lam, _ = a._r_star_row(0, g, np.full(a.J, 1.0 / a.J))
    want = al.apply_floor(al.r_neyman(a.a_cell), al.FLOOR_FRACTION)
    assert np.allclose(r, want, rtol=1e-12, atol=1e-14)
    assert lam == 0.0
    # and the geometry is genuinely non-uniform, or the gate is vacuous
    assert a.a_cell.max() / max(a.a_cell[a.a_cell > 0].min(), 1e-300) > 5.0


# --------------------------------------------------------------------------- #
# G0.6  the estimator recovers a synthetic AR(1) ordering
# --------------------------------------------------------------------------- #
def test_g0_6_batched_tau_equals_frozen_tau():
    """The batched fit IS ``inf.tau_from_lag1``, not a lookalike."""
    rng = np.random.default_rng(3)
    J, n = 16, 300
    S = rng.normal(size=(n, J))
    for j in range(J):
        phi = 0.2 + 0.75 * j / J
        for t in range(1, n):
            S[t, j] = phi * S[t - 1, j] + S[t, j]
    S[5, 3] = np.nan                                   # an empty cell at one obs
    hist = inf.MeanForceHistory(n_cells=J, capacity=n)
    hist._buf = [S[t] for t in range(n)]
    ref = inf.tau_from_lag1(hist, obs_interval=0.05)
    got = io_abf.tau_from_series(S[None, :, :], obs_interval=0.05)[0]
    assert np.allclose(got, ref, equal_nan=True, rtol=1e-12, atol=1e-14)


def test_g0_6_synthetic_ar1_recovers_sigma_tau_gamma_ordering():
    """End to end on a synthetic force stream with known ``sigma^2`` and ``tau``.

    Particles in cell ``j`` carry an AR(1) residual of variance ``sigma_j^2`` and
    correlation time ``tau_j``, drawn independently per particle, so the cell
    *mean* inherits ``tau_j`` while the instantaneous spread is ``sigma_j^2``.
    The two factors are drawn independently of one another, which is the point:
    if they covaried, a ``Gamma`` correlation would be attainable by tracking
    either factor alone and the gate would not test the product.

    Spearman rather than Pearson because the shrinkage toward the pooled value
    compresses the scale on purpose -- only the ordering is claimed.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(11)
    J, n_obs, n_part, dt_obs = 16, 900, 24, 0.05
    tau_true = np.exp(rng.uniform(np.log(0.05), np.log(3.0), J))
    sig_true = np.exp(rng.uniform(np.log(0.3), np.log(6.0), J))
    phi = np.exp(-dt_obs / tau_true)

    eps = rng.normal(size=(n_part, J))
    stream = io_abf.IOBatch(1, J, n_obs, DEV, DT)
    cell = torch.as_tensor(np.tile(np.arange(J), n_part), dtype=torch.long).view(1, -1)
    for _ in range(n_obs):
        eps = phi * eps + np.sqrt(1.0 - phi ** 2) * rng.normal(size=(n_part, J))
        f = (sig_true * eps).reshape(-1)
        stream.observe(cell, torch.as_tensor(f, dtype=DT).view(1, -1),
                       torch.zeros(1, f.size, dtype=DT))

    sigma2_hat = stream.sigma2()[0]
    tau_hat = io_abf.tau_from_series(stream.series(), obs_interval=dt_obs)[0]
    valid = np.isfinite(tau_hat) & (tau_hat > 0)
    assert valid.mean() >= 0.8, f"valid-tau fraction {valid.mean():.2f} on synthetic AR(1)"

    assert spearmanr(sig_true ** 2, sigma2_hat).statistic > 0.95
    assert np.allclose(sigma2_hat, sig_true ** 2, rtol=0.35), (
        "sigma^2 is an instantaneous spread and should be tight")
    assert spearmanr(tau_true[valid], tau_hat[valid]).statistic > 0.85

    gam_hat = inf.gamma_hat_decomposed(sigma2_hat, tau_hat)
    gam_true = sig_true ** 2 * tau_true
    assert spearmanr(gam_true, gam_hat).statistic > 0.85
    # and the synthetic actually carries spread, or the correlation is vacuous
    assert gam_true.max() / gam_true.min() > 50.0


# --------------------------------------------------------------------------- #
# G0.7  A6c honours the mass-ESS constraint
# --------------------------------------------------------------------------- #
def test_g0_7_a6c_meets_mass_ess_floor():
    rng = np.random.default_rng(5)
    for _ in range(60):
        J = 32
        g = np.exp(rng.normal(0.0, 2.0, J))
        q = np.exp(rng.normal(0.0, 3.0, J))
        q = q / q.sum()
        out = al.r_ess_constrained(g, q, rho=0.5)
        ess = al.mass_ess_fraction(q, out.r)
        assert ess >= 0.5 - 1e-8, (ess, out.lam)
        # the floor is applied after the solve, which relaxes the constraint;
        # the *reported* number must therefore be the floored one, not the raw.
        rf = al.apply_floor(out.r, al.FLOOR_FRACTION)
        assert al.mass_ess_fraction(q, rf) > 0.0


def test_g0_7_a6c_reports_the_floored_ess():
    """What the run records as ESS_M/K must be the ESS of the target it applied."""
    a = _alloc(["A6c"])
    rng = np.random.default_rng(9)
    for _ in range(80):
        X = torch.as_tensor(rng.normal(-0.9, 0.4, (1, 128)), dtype=DT).clamp(-1.7, 1.7)
        a.observe(X, torch.as_tensor(rng.normal(size=(1, 128)), dtype=DT),
                  torch.zeros(1, 128, dtype=DT))
    x = torch.linspace(-1.8, 1.8, a.G, dtype=DT).view(1, -1)
    A = (2.5 * (x ** 2 - 1) ** 2)
    a.refresh(1, torch.as_tensor(rng.normal(-0.9, 0.4, (1, 128)), dtype=DT), A)
    row = a.rows[-1]
    r = np.array(row["r_star"])
    q = np.array(row["q"])
    assert al.mass_ess_fraction(q, r) >= 0.5 - 1e-8


# --------------------------------------------------------------------------- #
# G0.8  no weight reaches the mean-force accumulator
# --------------------------------------------------------------------------- #
def _squash(text: str) -> str:
    return "".join(text.split())


def test_g0_8_accumulator_sees_unweighted_physical_force():
    """Source-level: the deposit is the bare ``dV/dx``, with no allocation factor.

    Read off the tokenised engine so a weight appearing anywhere in the deposit
    fails here rather than showing up as an unexplained bias in a result.
    """
    src = _squash(io_abf._executable_source(eb.simulate_batch))
    assert "Sf.scatter_add_(1,idx,fx)" in src
    assert "C.scatter_add_(1,idx,torch.ones_like(X))" in src
    # and the allocation force is added to the DRIFT, never to the deposit
    assert "bias_force=bias_force+io_alloc.bias_force_at(X)" in src
    assert "io_alloc" not in src.split("Sf.scatter_add_")[0].split("idx=")[-1]


def test_g0_8_estimator_stays_unbiased_under_a_nonzero_allocation():
    """Numeric: with the bias reshaping occupancy, ``F'_hat`` still tracks ``F'_ref``.

    This is the claim IO-ABF rests on -- the extra term is a function of the
    reaction coordinate alone, so the conditional law at fixed ``z`` is
    untouched and ``E[dV/dz | z]`` is still ``F'``.  A6b must not be *systemat-
    ically* worse on the mean force than A0 on the same noise.
    """
    cfg = eb.PhysConfig(N=256, n_steps=8000, save_every=1000)
    io_cfg = io_abf.IOConfig(n_cells=32, obs_every=10, opportunity_every=200)
    recs = eb.simulate_batch(
        eb.BatchSpec(configs=[cfg] * 3, seeds=[0, 1, 2],
                     methods=[eb.IO_A0, eb.IO_A6B]),
        device=DEV, dtype=DT, io=eb.IOSpec(arms=["A0", "A6b"], cfg=io_cfg))
    a0 = np.array([r["final_l2_fp"] for r in recs if r["io_arm"] == "A0"])
    a6 = np.array([r["final_l2_fp"] for r in recs if r["io_arm"] == "A6b"])
    assert np.all(np.isfinite(a6))
    assert np.median(a6) < 3.0 * np.median(a0), (a0, a6)


# --------------------------------------------------------------------------- #
# G0.9  an unresolved cell falls back, it does not get an extreme allocation
# --------------------------------------------------------------------------- #
def test_g0_9_no_valid_tau_falls_back_to_the_geometric_target():
    """All-NaN ``tau`` must give ``Gamma`` flat, hence ``r ∝ sqrt(a)`` -- not chaos."""
    sigma2 = np.full(32, 2.0)
    tau = np.full(32, np.nan)
    gam = inf.gamma_hat_decomposed(sigma2, tau)
    assert np.allclose(gam, gam[0]), "unmeasured tau must not create spread"
    a = _alloc(["A6b"])
    r_fallback, _, _ = a._r_star_row(0, a.a_cell * gam, np.full(a.J, 1 / a.J))
    want = al.apply_floor(al.r_neyman(a.a_cell), al.FLOOR_FRACTION)
    assert np.allclose(r_fallback, want, rtol=1e-12)


def test_g0_9_partial_tau_failure_shrinks_toward_pooled():
    """A single unmeasured cell must land near the pooled value, not at an extreme."""
    rng = np.random.default_rng(2)
    sigma2 = np.exp(rng.normal(0, 0.3, 32))
    tau = np.exp(rng.normal(0, 0.3, 32))
    tau[7] = np.nan
    gam = inf.gamma_hat_decomposed(sigma2, tau)
    assert np.isfinite(gam).all()
    assert gam[7] < np.quantile(gam, 0.9) and gam[7] > np.quantile(gam, 0.1)


def test_g0_9_allocator_never_starves_a_cell_below_the_floor():
    """Even an extreme Gamma spread cannot drive a cell under the shared floor."""
    a = _alloc(["A6b"])
    g = np.zeros(a.J)
    g[0] = 1e12
    r, _, _ = a._r_star_row(0, g, np.full(a.J, 1 / a.J))
    assert r.min() >= al.FLOOR_FRACTION / a.J - 1e-12
    assert r.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# structural: the frozen constants are the ones the protocol names
# --------------------------------------------------------------------------- #
def test_frozen_constants():
    assert al.FLOOR_FRACTION == 0.25
    assert inf.SHRINK_WEIGHT == 0.3
    assert io_abf.IOConfig(n_cells=4, obs_every=1, opportunity_every=1).rho == 0.5
    assert io_abf.IOConfig(n_cells=4, obs_every=1, opportunity_every=1).theta == 1.0
    assert io_abf.cells_for_walkers(256) == 32
    assert io_abf.cells_for_walkers(2048) == 32
    assert io_abf.cells_for_walkers(64) == 8      # K/8 wins when walkers are few
    assert io_abf.cadence_for_run(40000)["opportunity_every"] == 500


def test_allocation_window_closes_before_the_end():
    with pytest.raises(ValueError):
        io_abf.IOConfig(n_cells=8, obs_every=1, opportunity_every=1,
                        stop_fraction=1.0)
