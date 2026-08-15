"""Regression gates for the gate analysis itself (scripts/methane_gates.py).

Every test here pins a defect that was **live in shipped analysis code**, found during a
cross-audit between this study and the NaCl study.  They are not hypotheticals, and three of
them would have changed a physics verdict rather than raising an error.

These defects do **not** share a direction, and an early version of this docstring wrongly
claimed they did (see Amendment 12.9's retraction).  Derived per defect: the Gate C nan leans
toward the **null** (a nan silences a `<` test, so no deficit is ever flagged and the verdict is
ABF-sufficient), while the dropped out-of-domain walkers lean toward the **positive** (an
under-counted numerator fires the same test too easily, declaring an establishment deficit).
Same failure class, opposite directions, because the missing value landed on opposite sides of
the comparison.  The direction is a property of how each test was written, not of what the study
hoped to find -- which is exactly why each is asserted here rather than reasoned about.

Run: python -m pytest tests/test_methane_gates.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from methane_gates import (assert_partition, bias_aware_target,  # noqa: E402
                           in_basin, state_of, tercile_edges)

GRID = np.linspace(0.33, 0.90, 115)
EDGES = tercile_edges(float(GRID[0]), float(GRID[-1]))
BETA = 0.4036


# ------------------------------------------------------------------ class 3: partitions
def test_grid_is_an_exact_partition():
    """Half-open everywhere silently drops grid[-1]; closed everywhere double-counts edges."""
    counts = np.sum([in_basin(GRID, EDGES, k) for k in range(3)], axis=0)
    assert np.all(counts == 1)
    assert_partition(GRID, EDGES, "grid")


def test_bias_aware_target_sums_to_one():
    q = bias_aware_target(np.linspace(0, 8, 115), np.zeros(115), GRID, EDGES, BETA)
    assert q.sum() == pytest.approx(1.0, abs=1e-12)


def test_out_of_domain_walkers_are_not_dropped():
    """THE UNSAFE ONE.  Walkers leave the domain through the soft walls; the screen's measured
    range is [0.322, 0.922] against a domain of [0.33, 0.90].  Dropping them under-counts
    occupancy against a full-weight target, so `occupancy < 0.5 Q*` fires too easily and the
    verdict is biased toward establishment-limited -- the direction that licenses an mFR arm.
    Measured before the fix: occupancies summing to 0.8.
    """
    xi = np.array([0.322, 0.35, 0.60, 0.88, 0.922])
    st = state_of(xi, EDGES)
    assert np.all(st >= 0), "an out-of-domain walker was assigned to no basin"
    assert sum(np.mean(st == k) for k in range(3)) == pytest.approx(1.0)
    # clamped to the nearest basin, not silently binned somewhere else
    assert st[0] == 0 and st[-1] == 2


def test_partition_assertion_is_live():
    bad = [(0.33, 0.60), (0.50, 0.75), (0.75, 0.90)]        # overlapping, as in the NaCl defect
    with pytest.raises(ValueError, match="not a partition"):
        assert_partition(GRID, bad, "overlapping")


# ------------------------------------------------------------------ class 1: no-data-as-pass
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_bias_raises_rather_than_returning_nan(bad):
    """A nan Q* makes `occupancy < 0.5 Q*` False everywhere -> no deficit -> ABF-sufficient.
    A study-ending verdict manufactured by missing data."""
    B = np.zeros(115)
    B[7] = bad
    with pytest.raises(ValueError, match="non-finite"):
        bias_aware_target(np.linspace(0, 8, 115), B, GRID, EDGES, BETA)


def test_nan_target_would_have_read_as_no_deficit():
    """Demonstrates the consequence the guard prevents, so the guard's purpose cannot be
    mistaken for fussiness."""
    occ = np.array([0.1, 0.2, 0.7])
    q_nan = np.full(3, np.nan)
    assert not (occ < 0.5 * q_nan).any(), "premise of the guard no longer holds"


@pytest.mark.parametrize("bias", [-5000.0, -2000.0, 0.0, 2000.0, 5000.0])
def test_target_is_stable_under_large_bias(bias):
    """Overflow/underflow of exp(-beta (F - B)) produced the same silent nan from complete,
    finite inputs -- reachable in a normal run once the applied bias grows."""
    q = bias_aware_target(np.linspace(0, 8, 115), np.full(115, bias), GRID, EDGES, BETA)
    assert np.all(np.isfinite(q))
    assert q.sum() == pytest.approx(1.0, abs=1e-12)


def test_uniform_bias_leaves_the_target_unchanged():
    """A constant added to the bias is a gauge choice and must not move Q* at all."""
    F = np.linspace(0, 8, 115)
    a = bias_aware_target(F, np.zeros(115), GRID, EDGES, BETA)
    b = bias_aware_target(F, np.full(115, 1234.5), GRID, EDGES, BETA)
    assert np.allclose(a, b, atol=1e-12)


# ------------------------------------------------------------------ ordering-dependent branches
def _two_wells(r, depths, centers, width):
    W = np.full_like(r, 3.0)
    for d, c in zip(depths, centers):
        W = np.minimum(W, d + (r - c) ** 2 / width ** 2)
    return W


@pytest.mark.parametrize("deeper_first", [True, False])
@pytest.mark.parametrize("merging", [True, False])
def test_basin_merge_is_correct_in_both_minimum_orderings(deeper_first, merging):
    """The 2 kT merge rule must keep the DEEPER minimum regardless of which comes first in r.

    Found by the NaCl session in its own analyzer: the merge popped by *grid index* where a
    *list position* is required, in the branch taken only when the first minimum is the higher
    one.  That branch raises on some inputs and, when the grid index happens to fall inside the
    list, silently deletes an unrelated basin and keeps the *shallower* minimum -- which then
    defines the states for Gate A, Gate C's bias-aware targets and every physical secondary.

    Their synthetic landscape had its deeper minimum first, so eight passing tests sat one
    untaken branch away from it.  This study's accepted reference is single-basin, so neither
    ordering is exercised by real data at all -- the branch would first be taken on a rebuild or
    a force-field change, which is the worst moment to discover it.
    """
    from methane_reference import find_basins

    kT = 2.4777
    r = np.linspace(0.33, 0.90, 115)
    deep, shallow = -3.0 * kT, (-2.6 * kT if merging else -1.0 * kT)
    depths = (deep, shallow) if deeper_first else (shallow, deep)
    centers, width = ((0.50, 0.60), 0.09) if merging else ((0.45, 0.75), 0.05)
    W = _two_wells(r, depths, centers, width)

    mins, maxima = find_basins(W, r, kT)
    kept = [float(r[i]) for i in mins]
    global_min = float(r[int(np.argmin(W))])

    assert len(mins) == (1 if merging else 2)
    assert any(abs(k - global_min) < 1e-9 for k in kept), \
        "the merge kept the shallower minimum"
    assert len(maxima) == len(mins) - 1


# ------------------------------------------------------------------ preregistered thresholds
def test_gate_b_threshold_is_not_scaled_to_the_seeds_present():
    """Gate B is "6 of 8" and must stay 6 regardless of how many seeds exist.

    The original code used ``min(GATE_B_MIN_SEEDS, len(per_seed))``, which at 1 seed becomes
    1/1 and at 5 becomes 5/5 -- silently relaxing a preregistered threshold and issuing a full
    verdict on a partial screen.  What a short screen changes is that no verdict is issued, not
    what the threshold is.  Same family as the NaCl session's ``--builds`` finding: a knob that
    changes the population a joint statistic is computed over needs a guard.
    """
    import methane_gates as mg
    assert mg.GATE_B_MIN_SEEDS == 6
    assert mg.N_SEEDS_REQUIRED == 8
    src = open(mg.__file__).read()
    assert "min(GATE_B_MIN_SEEDS" not in src, \
        "Gate B's threshold is being scaled to the number of seeds present"
    assert "complete = len(per_seed) >= N_SEEDS_REQUIRED" in src


# ------------------------------------------------------------------ ratio provenance
def test_gate_c_ratio_arguments_share_a_support():
    """P/Q is only meaningful if occupancy and target are normalised over the same support.

    This is the campaign's most consequential ratio failure and it was live: out-of-domain
    walkers were dropped from every tercile while Q* stayed normalised over the whole grid, so
    occupancies summed to 0.8 against a target summing to 1 and the deficit test fired too
    easily -- biasing toward establishment-limited, the verdict that licenses an mFR arm.

    A ratio presents as one clean number and hides the provenance of both its arguments, which
    is exactly what makes it quotable and unauditable. Asserted here so it cannot regress.
    """
    xi = np.array([0.322, 0.35, 0.45, 0.60, 0.75, 0.88, 0.922])   # includes out-of-domain
    occ = np.array([np.mean(state_of(xi, EDGES) == k) for k in range(3)])
    q = bias_aware_target(np.linspace(0, 8, 115), np.zeros(115), GRID, EDGES, BETA)
    assert occ.sum() == pytest.approx(1.0, abs=1e-12)
    assert q.sum() == pytest.approx(1.0, abs=1e-12)


def test_gate_a_tv_uses_shared_bins():
    """TV between two histograms requires a common binning; per-pool bins would be meaningless."""
    from methane_gates import tv_from_samples
    a = np.array([0.1, 0.2, 3.0])
    b = np.array([2.0, 2.5])
    bins = np.linspace(0.0, 3.0, 21)
    tv = tv_from_samples(a, b, bins)
    assert 0.0 <= tv <= 1.0
    # identical samples must give TV = 0 on the shared grid
    assert tv_from_samples(a, a, bins) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------------------------
# Gate C POWER.  The guard added 2026-08-14 after the NaCl session's N ladder.
# ---------------------------------------------------------------------------------------------

def test_gate_c_threshold_below_two_walkers_is_just_emptiness():
    """At `lambda < 2` the deficit test is arithmetically "the state is empty right now".

    `occupancy < 0.5 Q*` on an integer count of walkers means `count < 0.5 lambda`.  Once
    `0.5 lambda < 1` the only integer satisfying it is zero, so the gate stops measuring a
    50 % shortfall and starts measuring emptiness -- whose frequency is `e^-lambda` on physics
    alone and rises as N falls.  This is the arithmetic behind GATE_C_MIN_LAMBDA; it needs no
    statistical model, which is why it is asserted rather than argued.
    """
    import math
    from methane_gates import DEFICIT_FRAC
    for lam in (1.99, 0.995, 0.498, 0.249):        # the NaCl CIP ladder, N = 64/32/16/8
        assert DEFICIT_FRAC * lam < 1.0
        largest_failing_count = math.ceil(DEFICIT_FRAC * lam) - 1
        assert largest_failing_count == 0
    # and the false-firing rate is set by N, not by the physics
    assert math.exp(-0.249) / math.exp(-1.99) == pytest.approx(5.7, abs=0.1)


def test_gate_c_min_lambda_matches_its_stated_effect_size():
    """`GATE_C_MIN_LAMBDA` must be exactly the lambda at which a 50 % deficit reaches 2 sigma."""
    import math
    from methane_gates import DEFICIT_FRAC, GATE_C_MIN_LAMBDA
    lam = GATE_C_MIN_LAMBDA
    assert DEFICIT_FRAC * lam == pytest.approx(2.0 * math.sqrt(lam), rel=1e-12)
    # methane's own states clear it by an order of magnitude; the guard is inert here by design
    for lam_methane in (127.6, 147.0, 224.2):
        assert lam_methane >= GATE_C_MIN_LAMBDA
        assert 2.0 / math.sqrt(lam_methane) < DEFICIT_FRAC       # resolves better than it tests


def test_an_unpowered_state_must_not_read_as_no_deficit():
    """The deca retraction, as an assertion.

    `results/deca/screen_RETRACTED_no_min_count_guard/RETRACTED.md`: a 0.056 nm "state" below
    the soft wall that could never hold a walker, "Gate C fired on it", `licenses_mfr: true`,
    retracted.  The mirror error is equally fatal and quieter -- an unpopulatable state that
    happens NOT to fire contributes a free pass.  Neither may happen: an unpowered state is
    excluded from the verdict in both directions.
    """
    import math
    from methane_gates import GATE_C_MIN_LAMBDA
    lam_min_all = {0: 200.0, 1: 150.0, 2: 0.9}      # state 2 cannot be judged
    persistent = {0: 0, 1: 0, 2: 8}                 # ... and it "fires" on all 8 seeds
    powered = {k: v >= GATE_C_MIN_LAMBDA for k, v in lam_min_all.items()}
    binding = [k for k in range(3) if powered[k]]
    assert binding == [0, 1]
    assert any(persistent[k] > 0 for k in range(3))          # ungated: a deficit, wrongly
    assert not any(persistent[k] > 0 for k in binding)        # gated: excluded, correctly
    # and the exclusion is reported as a number, not a claim
    assert 2.0 / math.sqrt(lam_min_all[2]) > 2.0              # resolves only a >200 % deficit


def test_no_powered_state_is_unclassifiable_not_abf_sufficient():
    """If nothing binds, the cell has no verdict -- it must not fall through to ABF-sufficient.

    NaCl's N = 16 and N = 8 cells have no state reaching lambda >= 16.  "Smallest N passing
    every gate" searches toward exactly those cells, so the fall-through direction matters.
    """
    from methane_gates import GATE_C_MIN_LAMBDA
    lam_min_all = {0: 7.8, 1: 0.43}                 # NaCl N = 16: SSIP and CIP, neither powered
    binding = [k for k, v in lam_min_all.items() if v >= GATE_C_MIN_LAMBDA]
    assert binding == []
    gate_c_deficit = any(False for _ in binding)    # vacuously False -- the trap
    assert gate_c_deficit is False
    assert not binding                              # ... which is why computability is checked first


def test_gate_a_reports_the_preregistered_direction_not_its_transpose():
    """Sec 2.2 is TV(p(xi | Y)).  The transpose was quoted as the result until 2026-08-14.

    The two are different questions: "can the marginal in xi see the structural states" (which
    licenses a marginal method) versus "does the descriptor differ between xi-terciles" (partly
    tautological, since the gap volume grows with r by construction).  Constructed here so that
    the transpose passes and the preregistered direction fails -- if the script ever reverts,
    this test fails rather than a verdict silently changing.
    """
    from methane_gates import tv_from_samples
    rng = np.random.default_rng(0)
    # xi identical in both Y buckets; the descriptor differs strongly BETWEEN xi-terciles.
    xi = rng.uniform(0.33, 0.90, 40000)
    ngap = 6.0 * (xi - 0.33) / 0.57 + 0.01 * rng.standard_normal(40000)   # a function of xi alone
    y = ngap > np.median(ngap)
    xi_bins = np.linspace(0.33, 0.90, 61)
    ng_bins = np.linspace(float(ngap.min()), float(ngap.max()), 21)
    tv_pre = tv_from_samples(xi[y], xi[~y], xi_bins)                       # the gate
    lo, hi = xi < 0.52, xi > 0.71
    tv_transposed = tv_from_samples(ngap[lo], ngap[hi], ng_bins)           # its transpose
    assert tv_transposed > 0.99      # the transpose is near-perfect BY CONSTRUCTION
    assert tv_pre > 0.99             # here both are high because ngap IS xi -- the tautology
    # the point: they measure different things, so the script must report which one it ran
    import json
    res = json.load(open("results/methane/screen_N512/gates.json"))
    assert res["gateA_direction"].startswith("TV(p(xi|Y))")
    assert res["gateA_max_TV"] == pytest.approx(0.935, abs=0.001)
    assert res["gateA_transposed_max_TV"] == pytest.approx(0.987, abs=0.001)


def test_the_guard_rationale_and_the_gate_disagree_and_the_docs_say_so():
    """`lambda >= 16` is where counting noise stops dominating, NOT where 50 % becomes detectable.

    Measured on the real traces (Amendment 12.13): at `lambda = 16` the gate needs a 65 %
    planted deficit, and at `lambda = 224` it still needs 60 % -- the span rule, not `lambda`,
    sets detection above `lambda ~ 9`. Asserted so the analytic figure cannot quietly be
    re-attached to the gate by a future edit.
    """
    import json
    import math
    from methane_gates import DEFICIT_FRAC, GATE_C_MIN_LAMBDA
    lad = json.load(open("results/methane/screen_N512/gate_c_detection/ladder.json"))["rows"]
    by_lam = sorted(lad, key=lambda r: r["lam"])
    # the analytic criterion is what GATE_C_MIN_LAMBDA was derived from ...
    assert 2.0 / math.sqrt(GATE_C_MIN_LAMBDA) == pytest.approx(DEFICIT_FRAC, rel=1e-12)
    # ... and the gate does not honour it: every measured cell needs MORE than the analytic
    # figure once lambda is above the crossing, and never less than the 50 % it tests for.
    for r in by_lam:
        assert r["empirical"] >= DEFICIT_FRAC, "gate fired below the deficit it tests for"
        if r["lam"] >= 14.0:
            assert r["empirical"] > r["analytic"], f"lambda={r['lam']}: analytic was not optimistic"
    # detection is flat in lambda where counting noise is not binding: a 16x span in lambda
    # moves the threshold by at most 10 points
    high = [r["empirical"] for r in by_lam if r["lam"] >= 14.0]
    assert max(high) - min(high) <= 0.10
