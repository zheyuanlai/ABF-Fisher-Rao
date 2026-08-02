"""Tests for the three corrected screening diagnostics.

Each test plants the exact failure the corresponding correction exists to remove, and
asserts both that the old statistic fails on it and that the new one does not.  A fix that
is only shown to work on clean data has not been shown to be a fix.
"""
from __future__ import annotations

import numpy as np
import pytest

from mfr_diagnostics import (assert_matched_conditioning, assert_supported_target,
                             bias_aware_region_target, corridor_aware_entries,
                             matched_cell_conditional, per_cell_conditional)

RNG = np.random.default_rng(20260802)
EDGES = np.linspace(-np.pi, np.pi, 37)


# --------------------------------------------------------------------- 1. conditional
def _build_two_cell_region(psi_a, psi_b, w_ref_a, n_ref=40_000):
    """A region of two CV cells whose omitted-coordinate conditionals differ.

    The reference weights the two cells ``w_ref_a : 1 - w_ref_a``.  Whatever the run does
    with them, the *conditional in each cell* is identical to the reference's, so a correct
    check must return ~0 no matter how the run weights the interior.
    """
    n_a = int(n_ref * w_ref_a)
    cells = np.concatenate([np.zeros(n_a, int), np.ones(n_ref - n_a, int)])
    vals = np.concatenate([RNG.normal(psi_a, 0.35, n_a),
                           RNG.normal(psi_b, 0.35, n_ref - n_a)])
    vals = (vals + np.pi) % (2 * np.pi) - np.pi
    w = np.ones_like(vals)
    hist, counts, ess = per_cell_conditional(cells, vals, w, 2, EDGES)
    return hist, ess


def test_matched_cell_conditional_removes_interior_weight_confound():
    """The confound: identical per-cell conditionals, different interior weighting.

    The reference sits mostly in cell A; the run (ABF flattens within a region) sits mostly
    in cell B.  Every conditional agrees exactly, so the honest answer is "no disagreement".
    The region-aggregated statistic nevertheless reads large, because it is comparing the
    reference's interior weighting against the run's.
    """
    cell_hist, cell_ess = _build_two_cell_region(-1.6, +1.6, w_ref_a=0.85)
    # the run's own draw, from the SAME per-cell conditionals, weighted the other way
    n_run = 40_000
    n_a = int(0.15 * n_run)
    run_vals = np.concatenate([RNG.normal(-1.6, 0.35, n_a),
                               RNG.normal(+1.6, 0.35, n_run - n_a)])
    run_vals = (run_vals + np.pi) % (2 * np.pi) - np.pi
    obs = np.histogram(run_vals, bins=EDGES)[0][None, :]

    res = matched_cell_conditional(
        cell_hist, cell_ess, cell_region=np.zeros(2, int),
        cell_weight=np.array([0.15, 0.85]),           # the run's actual interior weights
        obs_hist_by_region=obs, n_regions=1, min_cell_ess=20.0)
    r = res["per_region"][0]
    assert r["tv_matched"] < 0.03, r
    assert r["tv_unmatched"] > 0.5, r          # the old check calls this a gross failure
    assert r["dropped_weight"] == pytest.approx(0.0)


def test_matched_cell_conditional_still_detects_a_real_error():
    """A genuine omitted-coordinate error must survive the correction.

    Same construction, but the run's conditional is *shifted* inside every cell.  Matching
    the weights removes the artifact, not the signal.
    """
    cell_hist, cell_ess = _build_two_cell_region(-1.6, +1.6, w_ref_a=0.85)
    n_run = 40_000
    n_a = int(0.15 * n_run)
    run_vals = np.concatenate([RNG.normal(-1.6 + 2.0, 0.35, n_a),
                               RNG.normal(+1.6 + 2.0, 0.35, n_run - n_a)])
    run_vals = (run_vals + np.pi) % (2 * np.pi) - np.pi
    obs = np.histogram(run_vals, bins=EDGES)[0][None, :]
    res = matched_cell_conditional(cell_hist, cell_ess, np.zeros(2, int),
                                   np.array([0.15, 0.85]), obs, 1)
    assert res["per_region"][0]["tv_matched"] > 0.5


def test_matched_cell_conditional_reports_dropped_weight():
    """Cells with too little reference information are dropped, and *reported*."""
    cell_hist, cell_ess = _build_two_cell_region(-1.6, +1.6, w_ref_a=0.999)
    obs = np.histogram(RNG.normal(0.0, 1.0, 5000), bins=EDGES)[0][None, :]
    res = matched_cell_conditional(cell_hist, cell_ess, np.zeros(2, int),
                                   np.array([0.5, 0.5]), obs, 1, min_cell_ess=1e5)
    r = res["per_region"][0]
    assert r["dropped_weight"] == pytest.approx(1.0)
    assert r["tv_matched"] is None      # refuses to answer rather than answering wrongly


# ------------------------------------------------------------------------ 2. entries
def test_corridor_aware_entries_credits_a_crossing_through_an_unlabelled_corridor():
    """The exact valine failure: A -> corridor -> B reads zero entries into B."""
    # one walker: 6 frames in state 0, 5 unlabelled, 6 frames in state 1
    traj = np.array([0] * 6 + [-1] * 5 + [1] * 6, dtype=np.int64)
    labels = traj[:, None, None]                       # (T, R=1, N=1)
    entries, trans, first = corridor_aware_entries(labels, n_states=2, min_dwell=2)
    assert entries[0, 1] == 1, "entry into state 1 was not credited"
    assert trans[0, 0, 1] == 1, "the 0 -> 1 transition was not credited"
    assert first[0, 1] == 11

    # the naive consecutive-frame counter, for contrast
    naive = ((traj[:-1] == 0) & (traj[1:] == 1)).sum()
    assert naive == 0


def test_corridor_aware_entries_ignores_a_single_frame_brush():
    """One frame touching a state is not an entry; ``min_dwell`` is what enforces it."""
    traj = np.array([0] * 6 + [1] + [0] * 6, dtype=np.int64)
    labels = traj[:, None, None]
    entries, _, _ = corridor_aware_entries(labels, n_states=2, min_dwell=3)
    assert entries[0, 1] == 0
    # with no dwell requirement the same brush does count -- the knob is doing the work
    entries1, _, _ = corridor_aware_entries(labels, n_states=2, min_dwell=1)
    assert entries1[0, 1] == 1


def test_corridor_aware_entries_counts_repeat_visits_and_splits_runs():
    traj = np.array([0] * 4 + [-1] * 3 + [1] * 4 + [-1] * 3 + [0] * 4, dtype=np.int64)
    labels = np.stack([traj, traj[::-1]], axis=0).T[:, :, None]     # (T, R=2, N=1)
    entries, trans, _ = corridor_aware_entries(labels, n_states=2, min_dwell=2)
    assert entries[0, 0] == 2 and entries[0, 1] == 1
    assert trans[0, 0, 1] == 1 and trans[0, 1, 0] == 1


# ------------------------------------------------------------------------- 3. target
def _toy_grid():
    """A 1-D 'grid' with two supported regions and one unsupported strip."""
    F = np.array([0.0, 0.5, np.inf, 2.0, 1.0])
    label = np.array([0, 0, -1, 1, 1])
    return F, label


def test_bias_aware_target_sums_to_one_on_the_support():
    F, label = _toy_grid()
    B = np.zeros((3, 5))
    Q = bias_aware_region_target(F, B, label, beta=1.0)
    assert Q.shape == (3, 2)
    np.testing.assert_allclose(Q.sum(axis=-1), 1.0, atol=1e-12)


def test_bias_aware_target_tracks_the_bias():
    """As the bias floods region 1, that region's ideal biased population must rise."""
    F, label = _toy_grid()
    flat = np.zeros(5)
    flooded = np.array([0.0, 0.0, 0.0, 2.0, 1.0])       # bias == F on region 1
    Q0 = bias_aware_region_target(F, flat[None], label, beta=1.0)[0]
    Q1 = bias_aware_region_target(F, flooded[None], label, beta=1.0)[0]
    assert Q1[1] > Q0[1]


def test_assert_supported_target_raises_on_leaked_mass():
    """The guard must FAIL, not warn, when the decomposition misses mass."""
    with pytest.raises(AssertionError, match="reference-supported"):
        assert_supported_target(np.array([[0.4, 0.3]]))


def test_assert_matched_conditioning_catches_mismatched_support():
    P = np.array([[0.3, 0.3]])       # conditioned on the whole domain: sums to 0.6
    Q = np.array([[0.5, 0.5]])       # conditioned on the labelled support
    with pytest.raises(AssertionError, match="same support"):
        assert_matched_conditioning(P, Q)
    assert_matched_conditioning(P / P.sum(axis=-1, keepdims=True), Q)
