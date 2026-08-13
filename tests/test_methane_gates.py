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
