"""Amendment 6: the bias-aware structural target and the corroboration gate.

This gate exists to stop a tercile of a monotone 72 kT climb being reported as an establishment
deficit. It must therefore be able to say NO on exactly that case, and YES only when a
structural population is genuinely depleted relative to what the applied bias implies.

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/test_deca_structural_gate.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from deca import states as st                                              # noqa: E402

KB = 0.008314462618
BETA = 1.0 / (KB * 300.0)


@pytest.fixture
def grid():
    return np.linspace(1.20, 3.60, 129)


def _joint(grid, centers, widths, weights, n_labels=9):
    """Synthetic p_ref(xi, y): each label a Gaussian in xi with a prescribed total weight."""
    j = np.zeros((n_labels, grid.size))
    for a, (c, s, w) in enumerate(zip(centers, widths, weights)):
        if w > 0:
            g = np.exp(-0.5 * ((grid - c) / s) ** 2)
            j[a] = w * g / g.sum()
    return j / j.sum()


def test_frozen_eligible_set_matches_the_amendment(grid):
    """Amendment 6 froze {0,1,2,3,4,5,6,8}; label 7 alone is below the 1e-3 floor."""
    assert st.ELIGIBLE_LABELS == (0, 1, 2, 3, 4, 5, 6, 8)
    assert 7 not in st.ELIGIBLE_LABELS
    assert st.STRUCTURAL_WEIGHT_FLOOR == pytest.approx(1e-3)


def test_reference_joint_drops_out_of_domain_and_normalises(grid):
    rng = np.random.default_rng(0)
    xi = np.concatenate([rng.uniform(1.2, 3.6, 20_000),
                         rng.uniform(1.0, 1.19, 5_000)])     # outside the grid
    y = rng.integers(0, 9, xi.size)
    w = np.ones(xi.size)
    joint, lw = st.reference_joint(xi, y, w, grid)
    assert joint.shape == (9, grid.size)
    assert abs(joint.sum() - 1.0) < 1e-12
    assert np.all(joint >= 0)
    # the out-of-domain slab must not pile onto bin 0
    assert joint[:, 0].sum() < 3.0 * joint[:, 5].sum()


def test_zero_bias_target_equals_reference_label_weights(grid):
    """With no bias the structural target is just the reference label composition."""
    j = _joint(grid, [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2],
               [0.1] * 9, [0.2, 0.05, 0.1, 0.1, 0.05, 0.2, 0.1, 0.0, 0.2])
    Q = st.bias_aware_structural_target(j, np.zeros(grid.size), BETA)[0]
    e = np.array(st.ELIGIBLE_LABELS)
    want = j[e].sum(axis=1)
    want = want / want.sum()
    assert np.allclose(Q, want, atol=1e-10)
    assert abs(Q.sum() - 1.0) < 1e-12


def test_bias_reweights_the_structural_target_toward_the_biased_region(grid):
    """A bias favouring large xi must raise the target for labels that live at large xi."""
    j = _joint(grid, [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 3.3],
               [0.08] * 9, [0.2, 0.05, 0.1, 0.1, 0.05, 0.2, 0.1, 0.0, 0.2])
    e = list(st.ELIGIBLE_LABELS)
    i8 = e.index(8)                                     # the label sitting at xi = 3.3
    Q0 = st.bias_aware_structural_target(j, np.zeros(grid.size), BETA)[0]
    B = 10.0 / BETA * (grid - grid[0]) / (grid[-1] - grid[0])     # rises to 10 kT at large xi
    Q1 = st.bias_aware_structural_target(j, B, BETA)[0]
    assert Q1[i8] > Q0[i8] * 2.0, (Q0[i8], Q1[i8])
    assert abs(Q1.sum() - 1.0) < 1e-12


def test_no_deficit_when_occupancy_matches_the_target(grid):
    j = _joint(grid, [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2],
               [0.1] * 9, [0.2, 0.05, 0.1, 0.1, 0.05, 0.2, 0.1, 0.0, 0.2])
    B = np.zeros(grid.size)
    Q = st.bias_aware_structural_target(j, B, BETA)[0]
    T, R, N = 100, 4, 64
    counts = np.round(Q * N).astype(int)
    counts[0] += N - counts.sum()
    y = np.concatenate([np.full(c, st.ELIGIBLE_LABELS[i]) for i, c in enumerate(counts)])
    label_y = np.tile(y[:N], (T, R, 1))
    steps = np.arange(T) * 1000
    v = st.structural_establishment(label_y, steps, j, B, BETA, n_steps=steps[-1])
    assert not v["any_deficit"], v


def test_deficit_detected_when_a_structural_state_is_starved(grid):
    """A label the reference says should hold ~20 % but which is never occupied."""
    j = _joint(grid, [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2],
               [0.1] * 9, [0.2, 0.05, 0.1, 0.1, 0.05, 0.2, 0.1, 0.0, 0.2])
    B = np.zeros(grid.size)
    T, R, N = 100, 4, 64
    # every walker in label 5; labels 0 and 8 (each ~20 % of the target) are empty
    label_y = np.full((T, R, N), 5)
    steps = np.arange(T) * 1000
    v = st.structural_establishment(label_y, steps, j, B, BETA, n_steps=steps[-1])
    assert v["any_deficit"]
    assert 0 in v["labels_with_persistent_deficit"]
    assert 8 in v["labels_with_persistent_deficit"]


def test_a_brief_structural_dip_is_not_a_deficit(grid):
    """The 0.20 T contiguity requirement must bite here too."""
    j = _joint(grid, [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2],
               [0.1] * 9, [0.2, 0.05, 0.1, 0.1, 0.05, 0.2, 0.1, 0.0, 0.2])
    B = np.zeros(grid.size)
    Q = st.bias_aware_structural_target(j, B, BETA)[0]
    T, R, N = 100, 4, 64
    counts = np.round(Q * N).astype(int)
    counts[0] += N - counts.sum()
    good = np.concatenate([np.full(c, st.ELIGIBLE_LABELS[i]) for i, c in enumerate(counts)])[:N]
    label_y = np.tile(good, (T, R, 1))
    label_y[60:64] = 5                                  # 4 % of the run, well under 20 %
    steps = np.arange(T) * 1000
    v = st.structural_establishment(label_y, steps, j, B, BETA, n_steps=steps[-1])
    assert not v["any_deficit"]


def test_excluded_label_cannot_absorb_probability(grid):
    """Label 7 is excluded, so it must not appear in the target or the occupancy."""
    j = _joint(grid, [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2],
               [0.1] * 9, [0.2, 0.05, 0.1, 0.1, 0.05, 0.2, 0.1, 0.5, 0.2])
    Q = st.bias_aware_structural_target(j, np.zeros(grid.size), BETA)[0]
    assert Q.size == len(st.ELIGIBLE_LABELS) == 8
    assert abs(Q.sum() - 1.0) < 1e-12
    label_y = np.full((10, 2, 8), 7)                    # everyone in the excluded label
    occ = st.structural_occupancy(label_y)
    assert occ.sum() == 0.0
