"""Regression tests for the alanine Category-B endpoints (kernel matching, gradient error).

Both endpoints were arm-insensitive by construction until 2026-09-02: the kernel-matched
reference used an unnormalised kernel (x9.6), and the gradient error used a spectral derivative
of a reference whose unvisited cells were filled with a constant (global ringing).
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from alanine.metrics_ala import aligned_l2, build_masks, grad_errors, smooth_reference  # noqa: E402

N = 97
DZ = 2 * math.pi / N


def torus():
    phi = -math.pi + DZ * (np.arange(N) + 0.5)
    return np.meshgrid(phi, phi, indexing="ij")


def test_smoothing_a_constant_returns_the_constant():
    F = np.full((N, N), 3.7)
    assert np.allclose(smooth_reference(F, 0.08, N), 3.7, atol=1e-10)


def test_smoothing_is_a_normalised_average_not_a_scaled_sum():
    P1, P2 = torus()
    F = 5.0 * np.cos(P1) + 3.0 * np.sin(P2)
    Fs = smooth_reference(F, 0.08, N)
    # a normalised Gaussian of width h shrinks a unit-frequency mode by exp(-h^2/2); the old
    # unnormalised kernel multiplied it by ~9.6 instead
    shrink = math.exp(-0.5 * 0.08 ** 2)
    assert np.allclose(Fs, shrink * F, atol=2e-3)
    assert abs(Fs).max() < 1.05 * abs(F).max()


def test_smoothing_bias_on_a_reference_like_field_is_order_kT_not_order_25():
    P1, P2 = torus()
    kT = 2.494
    F = 10.0 * (1 - np.cos(P1 + 1.3)) + 6.0 * (1 - np.cos(2 * P2)) + 4.0 * np.sin(P1) * np.cos(P2)
    F[(np.abs(P1) < 0.3) & (P2 > 2.5)] = np.inf          # an unvisited corner, as in the real reference
    pack = build_masks(F, kT)
    w = pack["weights"]["equilibrium"]
    bias = aligned_l2(smooth_reference(F, 0.08, N), F, w)
    assert 0.0 < bias < 0.5 * kT


def test_gradient_error_is_zero_for_identical_and_for_shifted_fields():
    P1, P2 = torus()
    F = np.cos(P1) + 2 * np.sin(2 * P2)
    w = np.ones((N, N)) / N ** 2
    assert grad_errors(F, F, w, N) == 0.0
    assert grad_errors(F + 11.0, F, w, N) < 1e-12


def test_gradient_error_does_not_ring_from_an_unvisited_block():
    """A block of +inf in the REFERENCE must not create gradient error far away from it."""
    P1, P2 = torus()
    F = np.cos(P1) + 2 * np.sin(2 * P2)
    R = F.copy()
    R[40:50, 40:50] = np.inf
    w = np.zeros((N, N))
    w[:20, :20] = 1.0                       # weighted region far from the block
    w /= w.sum()
    assert grad_errors(F, R, w, N) < 1e-12   # the FFT version gave O(jump / dz) here


def test_gradient_error_excludes_stencils_touching_non_finite_cells():
    P1, P2 = torus()
    F = np.cos(P1)
    R = F.copy()
    R[10, 10] = np.inf
    w = np.zeros((N, N))
    w[9:12, 9:12] = 1.0                      # only the 3x3 patch around the hole
    # the hole's stencil neighbours are dropped; the corners of the patch survive with 0 error
    assert grad_errors(F, R, w, N) < 1e-12


@pytest.mark.skipif(not os.path.exists("results/alanine/reference/reference.npz"),
                    reason="real reference not present")
def test_real_reference_kernel_bias_is_sub_kT():
    import json
    r = np.load("results/alanine/reference/reference.npz", allow_pickle=True)
    F = r["F"].astype(float)
    kT = json.loads(str(r["meta"]))["kT_kJ"]
    w = build_masks(F, kT)["weights"]["equilibrium"]
    bias = aligned_l2(smooth_reference(F, 0.08, N), F, w)
    assert 0.3 < bias < 0.7          # 0.478 kJ/mol; the unnormalised kernel gave 25.6
