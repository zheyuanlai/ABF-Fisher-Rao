"""Gates B and C for deca-alanine: basin finding, bias-aware targets, and classification.

These gates decide whether any mFR arm is licensed at all, so a bug here would not produce a
wrong number -- it would produce a wrong *study*. Each gate is therefore tested against
constructed cases whose correct verdict is known by construction, including the two failure
verdicts, because a gate that only ever passes is not a gate.

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/test_deca_states.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from deca import states as st                                              # noqa: E402

BETA = 1.0 / (0.008314462618 * 300.0)
KT = 1.0 / BETA


@pytest.fixture
def grid():
    return np.linspace(1.2, 3.6, 129)


# --------------------------------------------------------------------------- basins
def test_double_well_gives_two_states(grid):
    """A clean 10 kT double well must be found as two basins, split at the barrier."""
    F = 10 * KT * ((grid - 1.7) ** 2) * ((grid - 3.1) ** 2) / 0.5
    edges, mins, fallback = st.find_basins(grid, F, BETA)
    assert not fallback
    assert len(edges) - 1 == 2, edges
    assert 1.7 < edges[1] < 3.1


def test_shallow_ripples_are_merged_not_counted(grid):
    """Sub-2 kT wiggles on a single well are noise, not states."""
    F = 40 * KT * (grid - 2.4) ** 2 + 0.4 * KT * np.sin(30 * grid)
    edges, mins, fallback = st.find_basins(grid, F, BETA)
    assert fallback, "shallow ripples were counted as basins"
    assert len(edges) - 1 == 3, "fallback must be the frozen tercile partition"


def test_single_well_triggers_the_frozen_tercile_fallback(grid):
    F = 30 * KT * (grid - 1.7) ** 2
    edges, mins, fallback = st.find_basins(grid, F, BETA)
    assert fallback
    assert np.allclose(edges, np.linspace(grid[0], grid[-1], 4))


def test_monotonic_pmf_also_falls_back(grid):
    """The deca-alanine-shaped case: one minimum, then a long climb."""
    F = 60 * KT * (grid - grid[0]) / (grid[-1] - grid[0])
    edges, mins, fallback = st.find_basins(grid, F, BETA)
    assert fallback and len(edges) - 1 == 3


# --------------------------------------------------------------------------- Q*
def test_bias_aware_target_is_flat_when_the_bias_cancels_the_pmf(grid):
    """B_t = F_ref means the biased ensemble is uniform, so Q* is proportional to state width."""
    F = 20 * KT * (grid - 2.0) ** 2
    edges = np.linspace(grid[0], grid[-1], 4)
    Q = st.bias_aware_target(grid, F, F, BETA, edges)
    assert Q.shape == (1, 3)
    assert np.allclose(Q[0], 1.0 / 3.0, atol=0.02), Q


def test_bias_aware_target_differs_from_the_unbiased_mass(grid):
    """The whole point of Q*: a state rare at equilibrium can be well populated under the bias."""
    F = np.where(grid < 2.4, 0.0, 8.0 * KT)
    edges = np.array([grid[0], 2.4, grid[-1]])
    Q_unbiased = st.bias_aware_target(grid, F, np.zeros_like(F), BETA, edges)[0]
    Q_biased = st.bias_aware_target(grid, F, F, BETA, edges)[0]
    assert Q_unbiased[1] < 0.01, Q_unbiased
    assert Q_biased[1] > 0.4, Q_biased


# --------------------------------------------------------------------------- classification
def _trace(occ_per_state, T=100, R=8, N=16, edges=None):
    """Build a xi trace whose per-state occupancy matches ``occ_per_state`` (T, K)."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    K = len(centers)
    xi = np.zeros((T, R, N))
    for t in range(T):
        counts = np.round(np.asarray(occ_per_state[t]) * N).astype(int)
        counts[0] += N - counts.sum()
        vals = np.concatenate([np.full(c, centers[k]) for k, c in enumerate(counts)])
        xi[t] = np.tile(vals[:N], (R, 1))
    return xi


def test_abf_sufficient_when_everything_is_found_and_populated():
    edges = np.linspace(1.2, 3.6, 4)
    T, K = 100, 3
    occ = np.tile(np.array([[0.34, 0.33, 0.33]]), (T, 1))
    xi = _trace(occ, T=T, edges=edges)
    steps = np.arange(T) * 1000
    Q = np.tile(np.array([[1 / 3, 1 / 3, 1 / 3]]), (T, 1))
    v = st.classify(xi, steps, edges, Q, n_steps=steps[-1])
    assert v["regime"] == "ABF-sufficient" and not v["licenses_mfr"]


def test_discovery_limited_when_a_state_is_reached_too_late():
    edges = np.linspace(1.2, 3.6, 4)
    T = 100
    occ = np.tile(np.array([[0.5, 0.5, 0.0]]), (T, 1))
    occ[95:] = [0.34, 0.33, 0.33]                     # state 2 only appears at 95 % of the run
    xi = _trace(occ, T=T, edges=edges)
    steps = np.arange(T) * 1000
    Q = np.tile(np.array([[1 / 3, 1 / 3, 1 / 3]]), (T, 1))
    v = st.classify(xi, steps, edges, Q, n_steps=steps[-1])
    assert v["regime"] == "discovery-limited" and not v["licenses_mfr"]


def test_establishment_limited_when_found_early_but_starved_persistently():
    edges = np.linspace(1.2, 3.6, 4)
    T = 100
    occ = np.tile(np.array([[0.55, 0.39, 0.06]]), (T, 1))
    occ[0] = [0.34, 0.33, 0.33]                       # discovered immediately ...
    xi = _trace(occ, T=T, edges=edges)                # ... then starved for the whole run
    steps = np.arange(T) * 1000
    Q = np.tile(np.array([[1 / 3, 1 / 3, 1 / 3]]), (T, 1))
    v = st.classify(xi, steps, edges, Q, n_steps=steps[-1])
    assert v["regime"] == "establishment-limited" and v["licenses_mfr"]


def test_a_brief_dip_is_not_establishment_limited():
    """The 0.20 T contiguity requirement must actually bite."""
    edges = np.linspace(1.2, 3.6, 4)
    T = 100
    occ = np.tile(np.array([[0.34, 0.33, 0.33]]), (T, 1))
    occ[60:65] = [0.6, 0.36, 0.04]                    # 5 % of the run, well under 20 %
    xi = _trace(occ, T=T, edges=edges)
    steps = np.arange(T) * 1000
    Q = np.tile(np.array([[1 / 3, 1 / 3, 1 / 3]]), (T, 1))
    v = st.classify(xi, steps, edges, Q, n_steps=steps[-1])
    assert v["regime"] == "ABF-sufficient"


def test_hitting_times_and_assignment_are_consistent():
    edges = np.array([1.2, 2.0, 2.8, 3.6])
    assert list(st.assign_states(np.array([1.5, 2.4, 3.2]), edges)) == [0, 1, 2]
    assert list(st.assign_states(np.array([0.1, 9.9]), edges)) == [0, 2]
    xi = np.full((10, 2, 4), 1.5)
    xi[7, :, 0] = 3.2
    steps = np.arange(10) * 100
    h = st.hitting_times(xi, steps, edges)
    assert h[0, 0] == 0 and h[0, 2] == 700 and h[0, 1] == -1
