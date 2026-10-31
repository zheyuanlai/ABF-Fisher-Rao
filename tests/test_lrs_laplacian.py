"""LRS Laplacian selection: book Eq. (6.26), sign, boundaries, and zero-rate parity.

The published score S=c p''/p uses positive for birth, whereas the existing
resampling engines use positive for death.  These tests are intended to fail if
that convention is accidentally reversed.  No other selection baseline added.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gateway_ladder_numba as G  # noqa: E402
import lta_ladder_numba as L  # noqa: E402


def test_gateway_laplacian_is_positive_for_overpopulated_mode():
    x = np.linspace(G.XMIN, G.XMAX, G.N_GRID)
    dx = float(x[1] - x[0])
    # Neumann boundaries and one maximum at the left, minimum at the right.
    mode = np.cos(math.pi * (x - G.XMIN) / (G.XMAX - G.XMIN))
    p = (1.0 + 0.1 * mode) / (G.XMAX - G.XMIN)
    sc = np.empty(x.size)
    G.laplacian_scores(x, len(x), p, dx, 0.01, 10.0, sc)
    assert np.isfinite(sc).all()
    assert abs(sc.mean()) < 1e-12
    assert sc[0] > 0.0 and sc[-1] < 0.0
    assert np.corrcoef(sc, mode)[0, 1] > 0.998


def test_gateway_uniform_is_stationary_under_laplacian():
    x = np.linspace(G.XMIN, G.XMAX, G.N_GRID)
    p = np.full(G.N_GRID, 1.0 / (G.XMAX - G.XMIN))
    s = np.empty(x.size)
    G.laplacian_scores(x, len(x), p, x[1]-x[0], 0.01, 2.0, s)
    np.testing.assert_allclose(s, 0.0, atol=1e-12)


def test_lta_periodic_laplacian_sign_and_fourier_frequency():
    ng = 180
    dphi = 2 * math.pi / ng
    grid0 = -math.pi + 0.5 * dphi
    phi = grid0 + np.arange(ng) * dphi
    N = 2 * ng
    phin = np.tile(phi, 2)
    mode = np.cos(2.0 * phi)
    pk = (1.0 + 0.1 * mode) / (2.0 * math.pi)
    sc, raw, p2 = np.empty(N), np.empty(N), np.empty(ng)
    std, amax = L._laplacian_score(phin, N, ng, dphi, pk, grid0, 0.01, 2.0, raw, sc, p2)
    assert std > 0.0 and amax > 0.0
    assert np.isfinite(sc).all()
    assert abs(sc.mean()) < 1e-12
    assert np.corrcoef(sc[:ng], mode)[0, 1] > 0.999
    # mode maximum => p''<0 => project score>0 => higher death rate.
    assert sc[np.argmax(mode)] > 0.0
    assert sc[np.argmin(mode)] < 0.0


def test_lta_uniform_is_stationary_under_laplacian():
    ng = 180
    dphi = 2 * math.pi / ng
    grid0 = -math.pi + 0.5 * dphi
    phi = grid0 + np.arange(ng) * dphi
    pk = np.full(ng, 1.0 / (2 * math.pi))
    sc, raw, p2 = np.empty(ng), np.empty(ng), np.empty(ng)
    std, amax = L._laplacian_score(phi, ng, ng, dphi, pk, grid0, 0.01, 2.0, raw, sc, p2)
    np.testing.assert_allclose(sc, 0.0, atol=1e-12)


def test_gateway_zero_rate_equals_abf():
    cfg = G.full_cfg({"gamma": 0.0, "ramp_t": 0.0, "fr_interval_t": 0.00025})
    save = np.array([0, 20, 40], dtype=np.int64)
    a = G.run_arm(cfg, "abf", 16, 190002, 40, save, diagnostics=False, chunk_steps=20)
    b = G.run_arm(cfg, "laplacian", 16, 190002, 40, save, diagnostics=False, chunk_steps=20)
    for key in ("X_final", "Y_final", "M_all", "C_all"):
        np.testing.assert_array_equal(a[key], b[key])
    assert int(b["n_force_evals"]) == 16 * 40


@pytest.mark.skipif(not (ROOT / "cache/lta/framework.npz").exists(),
                    reason="LTA framework artifact is required")
def test_lta_zero_rate_equals_abf():
    cfg = L.make_cfg(150.0, 8, 190002, n_steps=40, fr_start_steps=10, fr_rate=0.0)
    save = np.array([0, 20, 40], dtype=np.int64)
    a = L.run_arm(cfg, "abf", save, chunk_steps=20)
    b = L.run_arm(cfg, "laplacian", save, chunk_steps=20)
    for key in ("q_final", "M_all", "C_all"):
        np.testing.assert_array_equal(a[key], b[key])
    assert int(b["n_force_evals"]) == 8 * 41
