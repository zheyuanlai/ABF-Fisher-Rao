"""Tests for the T^3 metastable-state decomposition.

Each test pins one of the failure modes the module docstring names.  The seam test in
particular is the 3-D version of a bug that was real in 1-D: `count_states` reported a well
straddling +/-pi as two states until the traversal was forced to start at a separator.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from valine.states import (StateMap, histogram_nd, minmax_barrier, torus_distance,
                           transition_counts, wrap)

RNG = np.random.default_rng(20260801)


def blob(centre, n, sigma=0.18):
    """``n`` samples of a wrapped Gaussian on ``T^3`` about ``centre`` (radians)."""
    return wrap(np.asarray(centre)[None, :] + sigma * RNG.standard_normal((n, 3)))


# --------------------------------------------------------------------------- geometry
def test_torus_distance_uses_the_short_arc():
    a = np.array([[math.pi - 0.1, 0.0, 0.0]])
    b = np.array([[-math.pi + 0.1, 0.0, 0.0]])
    assert torus_distance(a, b) == pytest.approx(0.2, abs=1e-12)
    # the naive linear distance would be 2*pi - 0.2, i.e. 31x larger
    assert abs(a[0, 0] - b[0, 0]) == pytest.approx(2 * math.pi - 0.2)


def test_histogram_wraps_rather_than_clipping():
    h = histogram_nd(np.array([[math.pi + 0.01, 0.0, 0.0]]), 8)
    assert h.sum() == 1
    # pi + 0.01 wraps to just above -pi, i.e. the FIRST cell, not the last
    assert h[0, 4, 4] == 1


# --------------------------------------------------------------------------- clustering
def test_two_separated_blobs_give_two_states():
    x = np.concatenate([blob([-1.4, 1.4, math.pi], 40_000),
                        blob([1.1, -0.8, -1.05], 40_000)])
    sm = StateMap(x, n=24, min_prominence_kT=1.0, ceiling_kT=8.0)
    assert sm.n_states == 2
    lab = sm.assign(x)
    # every sample of a blob lands in one state, and the two blobs disagree
    assert len(set(lab[:40_000][lab[:40_000] >= 0])) == 1
    assert len(set(lab[40_000:][lab[40_000:] >= 0])) == 1
    assert lab[0] != lab[-1]


def test_a_state_straddling_the_seam_is_not_split():
    """THE periodicity test.  A blob centred at chi1 = pi has samples at both +pi and -pi.

    Clustering after linearly cutting the torus at -pi reports this as two states, which would
    invent a spurious barrier exactly where the trans rotamer of valine lives (chi1 ~ 180 deg).
    """
    x = blob([0.0, 0.0, math.pi], 60_000, sigma=0.22)
    assert (x[:, 2] > 3.0).any() and (x[:, 2] < -3.0).any(), "blob must actually cross the seam"
    sm = StateMap(x, n=24, min_prominence_kT=1.0, ceiling_kT=8.0)
    assert sm.n_states == 1
    lab = sm.assign(x)
    assert set(lab[lab >= 0]) == {0}


def test_unvisited_cells_are_walls_not_missing_data():
    """Two blobs a single empty cell-layer apart are two states, not one.

    The flood must refuse to cross a cell nothing sampled.  If empty cells were treated as
    absent rather than impassable the two blobs would merge, and a merged state cannot be
    reported as under-established -- the deficit would be averaged away.
    """
    x = np.concatenate([blob([-0.70, 0.0, 0.0], 40_000, sigma=0.06),
                        blob([+0.70, 0.0, 0.0], 40_000, sigma=0.06)])
    sm = StateMap(x, n=24, smooth_cells=0.0, min_prominence_kT=0.05, ceiling_kT=50.0)
    # The precondition is a WALL, not merely "some empty cell exists somewhere" -- almost every
    # cell of a 24^3 grid is empty here, so that weaker check passes vacuously.  What matters is
    # that a whole phi-slab between the blobs is unsampled, leaving no path around it.
    occupied_phi = np.unique(np.argwhere(sm.counts > 0)[:, 0])
    assert set(range(11, 13)).isdisjoint(occupied_phi), \
        f"phi cells 11-12 must be an empty slab; occupied phi cells were {occupied_phi}"
    assert sm.n_states == 2
    assert minmax_barrier(sm.G, sm.seeds[0], [sm.seeds[1]]) == float("inf")


def test_prominence_merging_absorbs_a_shallow_sub_minimum():
    """A weak dimple inside one blob must not become its own state."""
    x = np.concatenate([blob([0.0, 0.0, 0.0], 60_000, sigma=0.30),
                        blob([0.25, 0.0, 0.0], 6_000, sigma=0.12)])
    loose = StateMap(x, n=24, min_prominence_kT=0.02, ceiling_kT=12.0)
    strict = StateMap(x, n=24, min_prominence_kT=1.5, ceiling_kT=12.0)
    assert strict.n_states == 1
    assert strict.n_states <= loose.n_states


# --------------------------------------------------------------------------- transitions
def test_transitions_ignore_unassigned_frames():
    """A walker brushing a ridge (label -1) between two visits to the SAME state is not a move."""
    L = np.array([[0, 0, -1, -1, 0, 0]])
    assert transition_counts(L, 2).sum() == 0
    L2 = np.array([[0, 0, -1, 1, 1, 0]])
    T = transition_counts(L2, 2)
    assert T[0, 1] == 1 and T[1, 0] == 1 and T.sum() == 2


def test_transition_counts_are_directional():
    L = np.array([[0, 1, 1, 1]])
    T = transition_counts(L, 2)
    assert T[0, 1] == 1 and T[1, 0] == 0
