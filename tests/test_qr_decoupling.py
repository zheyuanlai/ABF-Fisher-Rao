"""Stage 0 engineering gates for the q-r decoupling campaign.

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md``.

These gates must all pass before any scientific run.  They test the properties
the campaign's conclusions rest on -- not that the code runs, but that the
objects mean what the preregistration says they mean:

  0A  no-reference architecture: the allocation modules cannot read the truth
  0B  mass projection is exact and fibre-constant
  0C  the ESS constraint interpolates A4b -> r = q, and rho = 1 recovers r = q
  0E  balanced offspring minimises the genealogy cost exactly
  0F  cell resampling is conditionally unbiased and never moves a configuration
  0G  empty cells stay empty: resampling cannot invent undiscovered support
  0H  leverage is pure geometry, vanishes off-mask, and is strongly non-uniform

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_qr_decoupling.py -q
"""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abffr import allocation as al                                    # noqa: E402
from abffr import balanced_representation as br                       # noqa: E402
from abffr import cell_mass as cm                                     # noqa: E402
from abffr import information as inf                                  # noqa: E402

CLEAN_V2_GRID = np.linspace(-3.0, 3.0, 401)
CLEAN_V2_MASK = (CLEAN_V2_GRID >= -2.5) & (CLEAN_V2_GRID <= 2.5)


# --------------------------------------------------------------------------- #
# Gate 0A -- no-reference architecture
# --------------------------------------------------------------------------- #
ALLOCATION_MODULES = ("abffr.allocation", "abffr.balanced_representation",
                      "abffr.information", "abffr.cell_mass")


def test_gate_0A_allocation_modules_cannot_reach_the_reference():
    """The allocation path must not import the truth, transitively.

    Checked on the import graph, not on the source text: a docstring that
    *discusses* why the reference is excluded is not a leak, and a gate that
    cannot tell the two apart would train us to stop writing the explanation.
    """
    import ast
    import importlib

    def imports_of(mod_name):
        mod = importlib.import_module(mod_name)
        tree = ast.parse(open(mod.__file__).read())
        out = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                out.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = "abffr" if node.level and node.module is None else node.module
                if node.level and node.module:
                    base = f"abffr.{node.module}"
                out.add(base or "")
                out.update(f"{base}.{a.name}" for a in node.names)
        return out

    seen, frontier = set(), list(ALLOCATION_MODULES)
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        for imp in imports_of(name):
            assert "reference" not in imp, f"{name} imports {imp}"
            if imp.startswith("abffr.") and imp.count(".") == 1:
                try:
                    importlib.import_module(imp)
                except ImportError:
                    continue                 # a symbol, not a submodule
                frontier.append(imp)


def test_gate_0A_evaluation_scope_is_geometry_not_free_energy():
    """The leverage is derived from the mask, so the mask may not be thermal.

    Under a scope such as R12 the mask is a function of ``F_ref``, which would
    make ``a_j`` -- and therefore the whole allocation -- an oracle quantity.
    Fixing the evaluation window by geometry is what keeps A4/A5 deployable, so
    it is a structural requirement of the campaign and not a matter of taste.
    """
    import inspect
    sig = inspect.signature(al.leverage)
    assert list(sig.parameters) == ["x_grid", "mask"]
    a = al.leverage(CLEAN_V2_GRID, CLEAN_V2_MASK)
    shifted = al.leverage(CLEAN_V2_GRID, (CLEAN_V2_GRID >= -2.0)
                          & (CLEAN_V2_GRID <= 2.0))
    assert not np.allclose(a, shifted), "the mask must actually determine a_j"


# --------------------------------------------------------------------------- #
# Gate 0H -- leverage is geometry
# --------------------------------------------------------------------------- #
def test_gate_0H_leverage_vanishes_outside_the_mask():
    a = al.leverage(CLEAN_V2_GRID, CLEAN_V2_MASK)
    assert np.all(a >= 0.0)
    assert np.allclose(a[~CLEAN_V2_MASK], 0.0, atol=1e-24), (
        "bins downstream of the mask never enter F there, and bins upstream "
        "shift it by a constant the centring removes")


def test_gate_0H_leverage_is_strongly_nonuniform():
    """The tie prediction must not be stated against uniform.

    ``a`` alone spreads the optimal allocation over an order of magnitude, so
    ``r ∝ sqrt(a Gamma)`` differs from count balancing even at flat ``Gamma``.
    This is why the campaign separates A4a (leverage only) from A4b.
    """
    a = al.leverage(CLEAN_V2_GRID, CLEAN_V2_MASK)
    J = 32
    edges = np.linspace(-3.0, 3.0, J + 1)
    cell = np.clip(np.digitize(CLEAN_V2_GRID, edges) - 1, 0, J - 1)
    g = al.cell_reduce(a, cell, J)
    r = al.r_neyman(g)
    live = r > 0
    assert r[live].max() / r[live].min() > 5.0
    assert (~live).sum() >= 2, "cells fully outside the mask should ask for none"


def test_gate_0H_leverage_is_translation_invariant():
    """Shifting the grid and mask together may not change the leverage."""
    a0 = al.leverage(CLEAN_V2_GRID, CLEAN_V2_MASK)
    a1 = al.leverage(CLEAN_V2_GRID + 7.5, CLEAN_V2_MASK)
    assert np.allclose(a0, a1, rtol=1e-10, atol=1e-30)


def test_leverage_matches_direct_monte_carlo_risk():
    """``sum_j a_j v_j`` must reproduce the metric applied to noisy profiles.

    The leverage is only meaningful if it is the same number the endpoint would
    report, so it is checked against the endpoint rather than against itself.
    """
    x = np.linspace(-3.0, 3.0, 121)
    mask = (x >= -2.0) & (x <= 2.0)
    dx = x[1] - x[0]
    rng = np.random.default_rng(0)
    v = rng.uniform(0.5, 2.0, size=x.size)          # per-bin mean-force variance

    a = al.leverage(x, mask)
    predicted = float(np.sum(a * v))

    xa, idx = x[mask], np.flatnonzero(mask)
    L = xa[-1] - xa[0]
    errs = []
    for _ in range(4000):
        d = rng.normal(0.0, np.sqrt(v))
        e = np.concatenate([[0.0], np.cumsum(0.5 * dx * (d[1:] + d[:-1]))])
        e = e[idx] - e[idx].mean()
        errs.append(np.trapezoid(e ** 2, xa) / L)
    empirical = float(np.mean(errs))
    assert abs(predicted - empirical) / predicted < 0.05, (predicted, empirical)


# --------------------------------------------------------------------------- #
# Gate 0B -- mass projection
# --------------------------------------------------------------------------- #
def test_gate_0B_projection_reproduces_cell_mass_exactly():
    J = 16
    mass = cm.CellMass(n_cells=J)
    rng = np.random.default_rng(1)
    mass.fr_step(np.log(rng.dirichlet(np.ones(J))))
    cell = rng.integers(0, J, size=256)
    w = mass.project(cell)

    live = np.bincount(cell, minlength=J) > 0
    got = np.array([w[cell == j].sum() for j in range(J)])
    want = np.where(live, mass.mass, 0.0)
    want = want / want.sum()
    assert np.allclose(got, want, rtol=1e-14, atol=1e-16)
    assert abs(w.sum() - 1.0) < 1e-14


def test_gate_0C_weights_are_constant_on_a_cell():
    """Fibre constancy: no path-dependent weight may reach the estimator."""
    J = 8
    mass = cm.CellMass(n_cells=J)
    rng = np.random.default_rng(2)
    for _ in range(20):                              # a long FR history
        mass.fr_step(np.log(rng.dirichlet(np.ones(J))))
    cell = rng.integers(0, J, size=200)
    w = mass.project(cell)
    for j in range(J):
        wj = w[cell == j]
        if wj.size > 1:
            assert np.allclose(wj, wj[0], rtol=0, atol=1e-18)


def test_fr_step_with_theta_one_is_projection_and_is_idempotent():
    J = 12
    rng = np.random.default_rng(3)
    q = rng.dirichlet(np.ones(J))
    mass = cm.CellMass(n_cells=J, theta=1.0)
    mass.fr_step(np.log(q))
    assert np.allclose(mass.mass, q, rtol=1e-12)
    mass.fr_step(np.log(q))
    assert np.allclose(mass.mass, q, rtol=1e-12)


def test_fr_step_with_theta_below_one_moves_partway_in_log_space():
    J = 6
    rng = np.random.default_rng(4)
    q = rng.dirichlet(np.ones(J))
    mass = cm.CellMass(n_cells=J, theta=0.4)
    p0 = mass.mass.copy()
    mass.fr_step(np.log(q))
    got = np.log(mass.mass)
    want = 0.6 * np.log(p0) + 0.4 * np.log(q)
    assert np.allclose(got - got.mean(), want - want.mean(), rtol=1e-12)


# --------------------------------------------------------------------------- #
# Gate 0C -- the ESS-constrained family
# --------------------------------------------------------------------------- #
def test_gate_0C_rho_one_recovers_r_equals_q():
    """clean-v2's equal-weight birth--death is this family's rho = 1 corner."""
    rng = np.random.default_rng(5)
    J = 24
    q = rng.dirichlet(np.ones(J) * 0.7)
    g = rng.uniform(0.1, 4.0, size=J)
    out = al.r_ess_constrained(g, q, rho=1.0)
    assert np.allclose(out.r, q / q.sum(), rtol=2e-3, atol=2e-4)


def test_gate_0C_constraint_is_met_and_risk_is_weakly_worse_than_A4b():
    """A5 is A4b under an extra constraint, so it cannot win the endpoint."""
    rng = np.random.default_rng(6)
    J = 32
    q = rng.dirichlet(np.ones(J) * 0.5)
    g = rng.uniform(0.05, 5.0, size=J)
    free = al.r_neyman(g)
    for rho in (0.3, 0.5, 0.7):
        out = al.r_ess_constrained(g, q, rho=rho)
        assert abs(out.r.sum() - 1.0) < 1e-12
        assert out.ess_fraction >= rho - 1e-6
        assert al.predicted_risk(g, out.r) >= al.predicted_risk(g, free) - 1e-12


def test_gate_0C_inactive_constraint_returns_the_neyman_optimum():
    J = 10
    q = np.full(J, 1.0 / J)
    g = np.full(J, 2.0)
    out = al.r_ess_constrained(g, q, rho=0.5)
    assert not out.constraint_active and out.lam == 0.0
    assert np.allclose(out.r, al.r_uniform(J))


def test_mass_ess_matches_its_particle_definition():
    rng = np.random.default_rng(7)
    J, K = 12, 480
    q = rng.dirichlet(np.ones(J))
    r = rng.dirichlet(np.ones(J) * 3.0)
    counts = al.desired_counts(r, K)
    r_real = counts / K
    w = np.repeat(q / np.maximum(counts, 1), counts)     # w_i = q_j / n_j
    w = w / w.sum()
    assert abs(al.mass_ess_fraction(q, r_real) - 1.0 / (K * np.sum(w ** 2))) < 1e-9


# --------------------------------------------------------------------------- #
# Gate 0E -- minimum-genealogy offspring
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_parents", range(1, 8))
@pytest.mark.parametrize("n_children", range(1, 13))
def test_gate_0E_balanced_offspring_is_the_exact_minimiser(n_parents, n_children):
    """Enumerate every integer multiplicity vector and check we hit the floor."""
    m = br.balanced_offspring(n_parents, n_children)
    assert m.sum() == n_children and m.size == n_parents
    best = min(
        br.duplicate_pairs(np.array(c))
        for c in itertools.product(range(n_children + 1), repeat=n_parents)
        if sum(c) == n_children)
    assert br.duplicate_pairs(m) == best


def test_rejuvenation_time_scales_with_measured_mixing():
    """A slow cell must hold its clones longer than a fast one."""
    fast = br.rejuvenation_steps(D=40, n_children=8, tau=0.1, dt=0.002)
    slow = br.rejuvenation_steps(D=40, n_children=8, tau=1.6, dt=0.002)
    assert slow > fast > 0
    assert abs(slow / fast - 16.0) < 0.1
    assert br.rejuvenation_steps(D=0, n_children=8, tau=1.0, dt=0.002) == 0


# --------------------------------------------------------------------------- #
# Gates 0F / 0G -- resampling
# --------------------------------------------------------------------------- #
def test_gate_0G_empty_cells_are_never_populated():
    """Reallocation establishes; it does not discover."""
    J = 6
    r = np.full(J, 1.0 / J)
    occupied = np.array([True, True, False, False, True, True])
    counts = al.desired_counts(r, 100, occupied=occupied)
    assert counts.sum() == 100
    assert counts[~occupied].sum() == 0

    cell = np.repeat([0, 1, 4, 5], 25)
    out = br.resample_cells(cell, counts, np.random.default_rng(8))
    assert np.all(cell[out.src] == np.repeat(np.arange(J), counts))


def test_gate_0G_assigning_an_empty_cell_is_an_error_not_a_silent_fix():
    cell = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="undiscovered support"):
        br.resample_cells(cell, np.array([2, 0, 2]), np.random.default_rng(9))


def test_gate_0F_resampling_never_moves_a_configuration_between_cells():
    rng = np.random.default_rng(10)
    J, K = 8, 128
    cell = rng.integers(0, J, size=K)
    occupied = np.bincount(cell, minlength=J) > 0
    r = al.apply_floor(al.r_neyman(rng.uniform(0.1, 3.0, size=J)))
    counts = al.desired_counts(r, K, occupied=occupied)
    out = br.resample_cells(cell, counts, rng)
    assert np.array_equal(np.bincount(cell[out.src], minlength=J), counts)


def test_gate_0F_cell_conditional_law_is_preserved_in_expectation():
    """Repeated resampling must not shift the within-cell empirical mean.

    This is the property the mean force is an expectation over: if selection
    inside a cell correlated with the force, the estimator would be biased and
    no amount of downstream care would recover it.
    """
    rng = np.random.default_rng(11)
    J, K = 4, 64
    cell = np.repeat(np.arange(J), K // J)
    f = rng.normal(size=K)
    counts = np.array([24, 8, 24, 8])

    before = np.array([f[cell == j].mean() for j in range(J)])
    acc = np.zeros(J)
    n_rep = 3000
    for _ in range(n_rep):
        out = br.resample_cells(cell, counts, rng)
        fr_ = f[out.src]
        cr = cell[out.src]
        acc += np.array([fr_[cr == j].mean() for j in range(J)])
    after = acc / n_rep
    se = np.array([f[cell == j].std(ddof=1) / np.sqrt(n_rep) for j in range(J)])
    assert np.all(np.abs(after - before) < 5.0 * se + 1e-12)


def test_resample_benefit_is_zero_when_already_optimal():
    g = np.array([1.0, 4.0, 9.0, 16.0])
    r_star = al.r_neyman(g)
    assert abs(br.resample_benefit(g, r_star, r_star)) < 1e-12
    assert br.resample_benefit(g, al.r_uniform(4), r_star) > 0.0


# --------------------------------------------------------------------------- #
# The difficulty estimator
# --------------------------------------------------------------------------- #
def _ar1_gamma(phi: float) -> float:
    """Asymptotic variance of AR(1) with unit innovations: sigma_x^2 * tau_int.

    ``sigma_x^2 = 1/(1-phi^2)`` and ``tau_int = (1+phi)/(1-phi)``, so the product
    is ``1/(1-phi)^2`` -- not ``(1+phi)/(1-phi)``, which omits the stationary
    variance and is the version that is easy to write down from memory.
    """
    return 1.0 / (1.0 - phi) ** 2


def _run_ar1(phis, n_blocks, block_len, seed):
    rng = np.random.default_rng(seed)
    phis = np.asarray(phis, dtype=float)
    acc = inf.BlockAccumulator(n_cells=phis.size, n_blocks=n_blocks)
    state = np.zeros(phis.size)
    for _ in range(500):                                  # burn in to stationarity
        state = phis * state + rng.normal(size=phis.size)
    for _ in range(n_blocks):
        for _ in range(block_len):
            state = phis * state + rng.normal(size=phis.size)
            acc.observe(np.arange(phis.size), state)
        acc.close_block()
    return acc


def test_gamma_hat_recovers_a_known_asymptotic_variance():
    """AR(1) with Gamma = 1/(1-phi)^2, given enough blocks to resolve it."""
    phi = 0.8
    acc = _run_ar1(np.full(3, phi), n_blocks=200, block_len=2000, seed=12)
    got = inf.gamma_hat(acc, shrink=0.0)
    want = _ar1_gamma(phi)
    assert np.all(np.abs(got / want - 1.0) < 0.20), (got, want)


def test_gamma_hat_at_ten_blocks_is_noisy_enough_to_need_shrinkage():
    """B = 10 gives ~47% relative error on a variance; record it, do not hide it.

    The campaign's block budget is a real limitation of the allocator, and the
    number belongs in a gate rather than in a caveat nobody re-derives.  It is
    also what the shrinkage exists for: an unshrunk Gamma_hat at this budget
    hands the allocator dispersion that looks exactly like heterogeneity.
    """
    phi = 0.8
    want = _ar1_gamma(phi)
    spread = []
    for seed in range(40):
        acc = _run_ar1(np.full(1, phi), n_blocks=10, block_len=2000, seed=100 + seed)
        spread.append(inf.gamma_hat(acc, shrink=0.0)[0] / want)
    spread = np.array(spread)
    assert abs(np.median(spread) - 1.0) < 0.25, np.median(spread)
    assert spread.std() > 0.25, "if this is small the shrinkage is unnecessary"

    flat = _run_ar1(np.full(24, phi), n_blocks=10, block_len=2000, seed=99)
    raw = inf.gamma_hat(flat, shrink=0.0)
    shrunk = inf.gamma_hat(flat, shrink=inf.SHRINK_WEIGHT)
    disp = lambda v: float(np.max(v) / np.min(v))          # noqa: E731
    assert disp(shrunk) < disp(raw), (disp(raw), disp(shrunk))


def test_gamma_hat_is_biased_low_when_blocks_are_shorter_than_tau():
    """The anti-detection failure: under-measuring difficulty where it is worst.

    tau_int for AR(1) is (1+phi)/(1-phi) steps; at phi = 0.99 that is 199, so a
    20-step block cannot see it.  Batch means then report the slow cell as
    *easier* than it is -- and an allocator following that estimate would move
    replicas away from exactly the cells that need them.  Stage 1B measures the
    block-length ratio before any allocation arm is allowed to run.
    """
    phi = 0.99
    want = _ar1_gamma(phi)
    short = inf.gamma_hat(_run_ar1(np.full(4, phi), 10, 20, seed=21), shrink=0.0)
    long = inf.gamma_hat(_run_ar1(np.full(4, phi), 10, 20000, seed=21), shrink=0.0)
    assert np.median(short) < 0.2 * want
    assert 0.4 * want < np.median(long) < 2.0 * want


def test_gamma_hat_ranks_cells_by_true_difficulty():
    """The allocator needs the ordering more than the absolute scale."""
    rng = np.random.default_rng(13)
    phis = np.array([0.0, 0.5, 0.9])
    acc = inf.BlockAccumulator(n_cells=3, n_blocks=10)
    state = np.zeros(3)
    for _ in range(10):
        for _ in range(4000):
            state = phis * state + rng.normal(0.0, 1.0, size=3)
            acc.observe(np.arange(3), state)
        acc.close_block()
    got = inf.gamma_hat(acc, shrink=0.0)
    assert got[0] < got[1] < got[2]


def test_gamma_hat_falls_back_to_pooled_not_zero_for_unmeasured_cells():
    """An unmeasured cell must not read as 'costless to ignore'."""
    acc = inf.BlockAccumulator(n_cells=4, n_blocks=10)
    rng = np.random.default_rng(14)
    for _ in range(6):
        acc.observe(np.array([0, 1, 2]), rng.normal(size=3))
        acc.close_block()
    got = inf.gamma_hat(acc)
    assert np.all(np.isfinite(got)) and np.all(got > 0)


def test_block_length_adequacy_flags_an_under_resolved_slow_cell():
    """The 16x kappa spread is exactly what a fixed block length can miss."""
    rep = inf.block_length_adequacy(500, 0.002, np.array([0.05, 0.1, 1.6]))
    assert rep["block_time"] == 1.0
    assert rep["ratio_min"] < 1.0 and rep["n_cells_below_10"] >= 1


# --------------------------------------------------------------------------- #
# The shared floor
# --------------------------------------------------------------------------- #
def test_floor_is_a_mixture_and_binds_on_every_arm_alike():
    J = 32
    r = al.apply_floor(al.r_neyman(np.concatenate([np.zeros(4), np.ones(J - 4)])))
    assert abs(r.sum() - 1.0) < 1e-14
    assert r.min() >= al.FLOOR_FRACTION / J - 1e-15
    assert np.allclose(al.apply_floor(al.r_uniform(J)), al.r_uniform(J))


# --------------------------------------------------------------------------- #
# The decomposed estimator, and why the campaign needs it
# --------------------------------------------------------------------------- #
def _ou_cells(taus_steps, n_steps, n_part, seed, sigma=1.0):
    """Independent OU walkers per cell: rho_1 = exp(-1/tau), known Gamma."""
    rng = np.random.default_rng(seed)
    taus_steps = np.asarray(taus_steps, dtype=float)
    J = taus_steps.size
    phi = np.exp(-1.0 / taus_steps)
    state = rng.normal(size=(J, n_part)) * sigma
    cell = np.repeat(np.arange(J), n_part)
    for _ in range(n_steps):
        state = phi[:, None] * state + np.sqrt(1.0 - phi[:, None] ** 2) * \
            rng.normal(size=(J, n_part)) * sigma
        yield cell, state.reshape(-1)


def _estimate_both(taus, n_steps, n_part, seed, n_blocks=10):
    """Run one realisation through both estimators on the same observations."""
    acc = inf.BlockAccumulator(n_cells=taus.size, n_blocks=n_blocks)
    hist = inf.MeanForceHistory(n_cells=taus.size, capacity=400)
    s2 = np.zeros(taus.size)
    n_obs = 0
    block = n_steps // n_blocks
    for t, (cell, f) in enumerate(_ou_cells(taus, n_steps, n_part, seed)):
        acc.observe(cell, f)
        if t % 10 == 0:                             # ABF update_every = 10
            hist.push(cell, f)
            s2 += inf.conditional_force_variance(cell, f, np.zeros_like(f),
                                                 taus.size)
            n_obs += 1
        if (t + 1) % block == 0:
            acc.close_block()
    dec = inf.gamma_hat_decomposed(
        s2 / n_obs, inf.tau_from_lag1(hist, obs_interval=10.0), shrink=0.0)
    return inf.gamma_hat(acc, shrink=0.0), dec


def test_decomposed_estimator_beats_batch_means_at_the_campaign_budget():
    """The measurement that forced this: tau up to ~480 steps, window ~3000.

    Batch means need a block long against tau *and* B of them; the
    decomposition needs a few tau to fit a decay and no window at all for the
    variance.  Judged on the median over realisations, because a single
    realisation at ``n/tau ~ 6`` is noisy for either estimator and it is the
    systematic recovery of the spread, not one draw, that decides whether the
    allocator can see heterogeneity at all.
    """
    taus = np.array([30.0, 120.0, 480.0])           # 16x spread, as K2/K3 build
    truth = 16.0
    bms, decs = [], []
    for seed in range(12):
        bm, dec = _estimate_both(taus, 3000, 8, seed=17 + seed)
        bms.append(bm[-1] / bm[0])
        decs.append(dec[-1] / dec[0])
    bm_med, dec_med = float(np.median(bms)), float(np.median(decs))

    assert dec_med > bm_med, (
        f"batch means recovered {bm_med:.1f}x of the true {truth:.0f}x, "
        f"decomposition {dec_med:.1f}x -- the decomposition must be the better "
        f"estimator at this budget or there is no reason to prefer it")
    assert dec_med > 0.5 * truth, (
        f"decomposition recovered only {dec_med:.1f}x of {truth:.0f}x; the "
        f"allocator cannot follow a signal it cannot see")


def test_decomposed_estimator_is_unbiased_given_enough_samples():
    """Separate the estimator's bias from its variance, and gate the bias.

    At the campaign budget a single cell estimate is noisy -- which is what the
    frozen shrinkage exists to damp, and which errs toward uniform allocation
    rather than toward an actively wrong one.  What must not be true is a
    systematic error, so the median is checked where samples are plentiful.
    """
    taus = np.array([30.0, 120.0])
    got = np.median([_estimate_both(taus, 30_000, 8, seed=40 + s)[1]
                     for s in range(9)], axis=0)
    assert 0.75 < (got[1] / got[0]) / 4.0 < 1.3, got


def test_batch_means_understates_the_hard_cell_at_this_budget():
    """Name the failure direction: it hides difficulty, it does not invent it."""
    taus = np.array([30.0, 480.0])
    acc = inf.BlockAccumulator(n_cells=2, n_blocks=10)
    for t, (cell, f) in enumerate(_ou_cells(taus, 3000, 8, seed=18)):
        acc.observe(cell, f)
        if (t + 1) % 300 == 0:
            acc.close_block()
    bm = inf.gamma_hat(acc, shrink=0.0)
    assert bm[1] / bm[0] < 16.0 * 0.6, (
        f"recovered {bm[1]/bm[0]:.1f}x of a true 16x -- if this ever passes "
        f"comfortably, the block budget is adequate and this gate is stale")


def test_tau_fit_reports_nothing_rather_than_something_small():
    """'Unmeasured' and 'easy' must not look alike to the allocator."""
    hist = inf.MeanForceHistory(n_cells=2, capacity=400)
    rng = np.random.default_rng(19)
    for _ in range(10):                            # far too few samples
        hist.push(np.array([0, 1]), rng.normal(size=2))
    assert np.all(np.isnan(inf.tau_from_lag1(hist, obs_interval=10.0)))
    filled = inf.gamma_hat_decomposed(np.array([1.0, 1.0]),
                                      np.array([np.nan, np.nan]))
    assert np.all(np.isfinite(filled)) and np.all(filled > 0)


def test_conditional_variance_does_not_charge_a_steep_mean_force_to_noise():
    """Residuals are taken against F' at each replica's own position."""
    rng = np.random.default_rng(20)
    cell = np.zeros(400, dtype=int)
    trend = np.linspace(-5.0, 5.0, 400)            # F' varies across the cell
    noise = rng.normal(0.0, 0.5, 400)
    naive = inf.conditional_force_variance(cell, trend + noise,
                                           np.zeros(400), 1)[0]
    corrected = inf.conditional_force_variance(cell, trend + noise, trend, 1)[0]
    assert corrected < 0.5 * naive
    assert abs(corrected - 0.25) < 0.05


# --------------------------------------------------------------------------- #
# The occupancy gate: why the benefit statistic cannot gate on its own
# --------------------------------------------------------------------------- #
def test_benefit_statistic_fires_on_pure_sampling_noise():
    """The defect the chi-square gate exists to fix, stated as a measurement.

    ``sum_j g_j / r_j`` is convex, so a population drawn *from the target
    itself* still shows an apparent gain from being equalised.  At this
    campaign's geometry that apparent gain clears the 0.10 benefit gate most of
    the time -- and the genealogy it would spend is real even though the
    misallocation it corrects is not.
    """
    rng = np.random.default_rng(0)
    J, K = 32, 256
    r = al.apply_floor(al.r_uniform(J))
    g = np.ones(J)
    fired = np.mean([br.resample_benefit(g, rng.multinomial(K, r) / K, r) > 0.10
                     for _ in range(400)])
    assert fired > 0.5, (
        f"only {fired:.2f} of on-target populations cleared the benefit gate; "
        f"if this is ever small the chi-square gate is no longer needed")


def test_occupancy_chi2_separates_noise_from_misallocation():
    """Checked on the arms' real profiles, not on an invented skew.

    The floor compresses an aggressive allocation a long way, so a
    plausible-looking synthetic target can sit inside the noise while the
    campaign's actual ones do not.  The realistic comparison is a population
    that ABF has already flattened against each arm's target.

    A3's target *is* roughly what ABF produces, so count balancing stands down
    once the marginal is flat -- which is the correct adaptive behaviour and
    not a defect: it intervenes early, when the population is not yet flat, and
    stops paying genealogy when there is nothing left to fix.
    """
    J, K = 32, 256
    x = np.linspace(-3.0, 3.0, 401)
    mask = (x >= -2.5) & (x <= 2.5)
    cell_of_grid = np.clip(np.digitize(x, np.linspace(-3.0, 3.0, J + 1)) - 1,
                           0, J - 1)
    a = al.cell_reduce(al.leverage(x, mask), cell_of_grid, J)
    rng = np.random.default_rng(2)
    flat = al.apply_floor(al.r_uniform(J))

    def chi2_of(target):
        want = al.desired_counts(al.apply_floor(target), K)
        return float(np.median([br.occupancy_chi2(rng.multinomial(K, flat), want)
                                for _ in range(300)]))

    assert chi2_of(al.r_uniform(J)) < 2.0, "A3 must stand down on a flat marginal"
    assert chi2_of(al.r_neyman(a)) > 3.0, "A4a must clear the gate"
    assert chi2_of(al.r_neyman(a * np.linspace(1.0, 16.0, J))) > 3.0, (
        "A4b must clear the gate")


def test_occupancy_chi2_ignores_cells_the_target_leaves_empty():
    counts = np.array([10, 0, 10])
    want = np.array([10, 0, 10])
    assert br.occupancy_chi2(counts, want) == 0.0


# --------------------------------------------------------------------------- #
# The dead band: deciding whether to move and how far are different questions
# --------------------------------------------------------------------------- #
def _leverage_target(J=32, K=256):
    x = np.linspace(-3.0, 3.0, 401)
    mask = (x >= -2.5) & (x <= 2.5)
    cg = np.clip(np.digitize(x, np.linspace(-3.0, 3.0, J + 1)) - 1, 0, J - 1)
    a = al.cell_reduce(al.leverage(x, mask), cg, J)
    return al.desired_counts(al.apply_floor(al.r_neyman(a)), K)


def test_deadband_absorbs_noise_but_still_moves_a_real_deviation():
    """Most of an exact snap's cost is fluctuation, not misallocation."""
    rng = np.random.default_rng(3)
    K = 256
    target = _leverage_target(K=K)

    on_target = [np.abs(al.deadband_counts(n, target) - n).sum() / 2
                 for n in (rng.multinomial(K, target / K) for _ in range(300))]
    exact = [np.abs(target - n).sum() / 2
             for n in (rng.multinomial(K, target / K) for _ in range(300))]
    assert np.mean(on_target) < 0.4 * np.mean(exact), (
        f"dead band moved {np.mean(on_target):.1f} against an exact snap's "
        f"{np.mean(exact):.1f}; it is not absorbing the noise")

    flat = al.desired_counts(al.apply_floor(al.r_uniform(target.size)), K)
    moved = np.abs(al.deadband_counts(flat, target) - flat).sum() / 2
    assert moved > 3 * np.mean(on_target), (
        "a genuinely misallocated population must still be moved")


def test_deadband_preserves_the_population_exactly():
    rng = np.random.default_rng(4)
    K = 256
    target = _leverage_target(K=K)
    for _ in range(50):
        n = rng.multinomial(K, np.full(target.size, 1.0 / target.size))
        out = al.deadband_counts(n, target)
        assert out.sum() == K, out.sum()
        assert np.all(out >= 0)


def test_deadband_never_populates_a_cell_the_target_excludes():
    """The empty-support rule must survive the shrinkage."""
    counts = np.array([10, 10, 10, 10])
    target = np.array([20, 20, 0, 0])
    out = al.deadband_counts(counts, target)
    assert out.sum() == 40 and out[2] == 0 and out[3] == 0


def test_leverage_has_the_brownian_bridge_closed_form():
    """``a(s) ∝ (s-L)(R-s)/(R-L)^2`` -- not a numerical accident.

    The centred cumulative integral of an uncorrelated mean-force error is a
    Brownian bridge on the evaluation window, and a bridge's pointwise variance
    is exactly ``s(1-s)``.  So the leverage vanishing at the window edges and
    peaking in the interior is a property of the endpoint's definition, not an
    artefact of the grid -- which is what licenses quoting it as theory.
    """
    x = np.linspace(-3.0, 3.0, 401)
    mask = (x >= -2.5) & (x <= 2.5)
    a = al.leverage(x, mask)
    L, R = x[mask][0], x[mask][-1]
    bridge = np.where(mask, (x - L) * (R - x) / (R - L) ** 2, 0.0)
    live = mask & (a > 0)
    scale = np.sum(a[live] * bridge[live]) / np.sum(bridge[live] ** 2)
    rel = np.abs(a[live] - scale * bridge[live]) / a[live].max()
    assert rel.max() < 0.01, rel.max()
    assert np.corrcoef(a[live], bridge[live])[0, 1] > 0.9999
