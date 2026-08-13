"""NaCl sampler and descriptor unit gates (SPEC_nacl_water.md §2, §4, §7).

The `fullSamples` test exists because of Amendment 5 Defect 2: a min-count guard that was
declared and never read cost a retracted deca screen.  It fails if the configured value stops
being applied.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nacl import system as nsys                                  # noqa: E402
from nacl.core import NaClSimConfig, colvars_trust, masked_bin_sum, wall_force  # noqa: E402
from nacl.observables import HydrationDescriptors, rational_switch  # noqa: E402

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- the fullSamples ramp
def test_colvars_ramp_is_applied_not_merely_declared():
    fs = float(nsys.FULL_SAMPLES)
    eff = torch.tensor([0.0, 0.25 * fs, 0.5 * fs, 0.75 * fs, fs, 2 * fs])
    t = colvars_trust(eff, fs)
    assert float(t[0]) == 0.0
    assert float(t[1]) == 0.0                    # below fullSamples/2: no bias at all
    assert float(t[2]) == 0.0                    # exactly at the ramp start
    assert float(t[3]) == pytest.approx(0.5)     # linear through the ramp
    assert float(t[4]) == pytest.approx(1.0)
    assert float(t[5]) == 1.0                    # clamped above


def test_ramp_tracks_the_configured_value():
    """Changing fullSamples must change the applied bias -- the guard is read, not decorative."""
    eff = torch.tensor([300.0])
    assert float(colvars_trust(eff, 500.0)) == pytest.approx(0.2)
    assert float(colvars_trust(eff, 200.0)) == 1.0
    assert float(colvars_trust(eff, 5000.0)) == 0.0


def test_published_full_samples_value():
    assert nsys.FULL_SAMPLES == 500              # abf.in, verbatim


# ---------------------------------------------------------------- estimator masking
def test_out_of_domain_samples_are_dropped_not_clamped():
    lo, hi, n = 0.2, 1.4, 121
    r = torch.tensor([[0.15, 0.25, 1.35, 1.60]])          # two inside, two outside
    mask = ((r >= lo) & (r <= hi)).to(r.dtype)
    counts = masked_bin_sum(r, torch.ones_like(r), mask, n, lo, hi)
    assert float(counts.sum()) == 2.0
    # the edge bins carry only the genuinely-inside samples
    assert float(counts[0, 0]) == 0.0
    assert float(counts[0, -1]) == 0.0

    forces = torch.tensor([[1e6, 3.0, 5.0, -1e6]])
    fsum = masked_bin_sum(r, forces, mask, n, lo, hi)
    assert float(fsum.sum()) == pytest.approx(8.0)         # the 1e6 excursions contribute zero


# ---------------------------------------------------------------- walls
def test_walls_match_the_published_colvars_convention():
    """1 kcal/mol per (0.1 A colvar width)^2 -> 41840 kJ/mol/nm^2, at the published boundaries."""
    assert nsys.K_WALL_KJ_NM2 == pytest.approx(1.0 * 4.184 / (0.01 ** 2))
    assert nsys.K_WALL_KJ_NM2 == pytest.approx(41840.0)
    assert (nsys.WALL_LO_NM, nsys.WALL_HI_NM) == (0.20, 1.40)

    sim = NaClSimConfig()
    r = torch.tensor([0.15, 0.5, 1.45])
    g = wall_force(r, sim)
    assert float(g[0]) > 0                        # inside the lower wall: pushed outward
    assert float(g[1]) == 0.0                     # in the domain: no wall force
    assert float(g[2]) < 0                        # above the upper wall: pushed inward
    assert float(g[0]) == pytest.approx(nsys.K_WALL_KJ_NM2 * 0.05)


def test_grid_spacing_is_the_published_bin_width():
    dz = (nsys.R_HI_NM - nsys.R_LO_NM) / (nsys.N_GRID - 1)
    assert dz == pytest.approx(0.01)              # published width 0.1 A
    assert nsys.N_GRID % 2 == 1                   # odd: no Nyquist row


# ---------------------------------------------------------------- hydration descriptors
def test_rational_switch_shape():
    r0 = 0.315
    assert float(rational_switch(torch.tensor(0.0), r0)) == 1.0
    assert float(rational_switch(torch.tensor(0.1 * r0), r0)) == pytest.approx(1.0, abs=1e-5)
    assert float(rational_switch(torch.tensor(2.0 * r0), r0)) == pytest.approx(1 / 65, rel=1e-9)
    assert float(rational_switch(torch.tensor(3.0 * r0), r0)) < 2e-3
    s = rational_switch(torch.linspace(0.05, 1.0, 200) * r0, r0)
    assert bool((s.diff() <= 1e-9).all())         # monotone decreasing


def test_switch_is_one_half_at_r0_not_zero():
    """The literal (1-t^6)/(1-t^12) is 0/0 exactly at r0; the stable form gives 1/2.

    A naive implementation returns ~0 there, so waters at the shell edge -- precisely the ones
    the coordination number is meant to resolve -- would be silently uncounted.
    """
    r0 = 0.315
    assert float(rational_switch(torch.tensor(r0), r0)) == pytest.approx(0.5)
    near = rational_switch(torch.tensor([r0 - 1e-9, r0, r0 + 1e-9]), r0)
    assert float((near - 0.5).abs().max()) < 1e-6         # continuous through r0


def _toy_system(n_waters=4):
    """(waters, positions) with waters placed by hand around the two ions."""
    waters = np.array([[2 + 3 * k, 3 + 3 * k, 4 + 3 * k] for k in range(n_waters)])
    x = np.zeros((1, 2 + 3 * n_waters, 3))
    return waters, x


def test_n_NaO_counts_first_shell_waters():
    waters, x = _toy_system(4)
    L = 3.0
    x[0, 0] = [0.0, 0.0, 0.0]          # Na
    x[0, 1] = [1.0, 0.0, 0.0]          # Cl, far away
    for k, d in enumerate((0.10, 0.20, 0.80, 1.20)):     # 2 inside R0=0.315, 2 far outside
        x[0, waters[k, 0]] = [0.0, d, 0.0]
        x[0, waters[k, 1]] = [0.01, d, 0.0]
        x[0, waters[k, 2]] = [-0.01, d, 0.0]
    h = HydrationDescriptors(waters, L, r0_nao=0.315, r0_clh=0.28, r0_clo=0.38)
    out = h.compute(torch.tensor(x))
    # a SMOOTH count, so the analytic sum -- not the hard count 2 -- is the expectation: the
    # water at 0.20 nm sits inside the shell but on the switch's shoulder (s = 0.938)
    expected = float(sum(1.0 / (1.0 + (d / 0.315) ** 6) for d in (0.10, 0.20, 0.80, 1.20)))
    assert float(out["n_NaO"]) == pytest.approx(expected, rel=1e-9)
    assert 1.9 < expected < 2.0


def test_n_bridge_is_zero_without_a_bridging_water():
    waters, x = _toy_system(2)
    L = 3.0
    x[0, 0] = [0.0, 0.0, 0.0]
    x[0, 1] = [1.20, 0.0, 0.0]                     # dissociated
    x[0, waters[0, 0]] = [0.0, 0.25, 0.0]          # coordinates Na only
    x[0, waters[0, 1]] = [0.01, 0.26, 0.0]
    x[0, waters[0, 2]] = [-0.01, 0.26, 0.0]
    x[0, waters[1, 0]] = [1.20, 0.30, 0.0]         # near Cl only
    x[0, waters[1, 1]] = [1.21, 0.31, 0.0]
    x[0, waters[1, 2]] = [1.19, 0.31, 0.0]
    h = HydrationDescriptors(waters, L, r0_nao=0.315, r0_clh=0.28, r0_clo=0.38)
    out = h.compute(torch.tensor(x))
    assert float(out["n_bridge"]) < 0.05
    assert float(out["n_bridge_hard"]) == 0.0


def test_descriptors_respect_minimum_image():
    waters, x = _toy_system(1)
    L = 3.0
    x[0, 0] = [0.05, 0.0, 0.0]                     # Na near the low face
    x[0, 1] = [1.0, 1.0, 1.0]
    x[0, waters[0, 0]] = [L - 0.05, 0.0, 0.0]      # water across the boundary: 0.10 nm away
    x[0, waters[0, 1]] = [L - 0.04, 0.01, 0.0]
    x[0, waters[0, 2]] = [L - 0.06, 0.01, 0.0]
    h = HydrationDescriptors(waters, L, r0_nao=0.315, r0_clh=0.28, r0_clo=0.38)
    assert float(h.compute(torch.tensor(x))["n_NaO"]) == pytest.approx(1.0, abs=0.05)


def test_Y_is_the_frozen_triple():
    waters, x = _toy_system(2)
    h = HydrationDescriptors(waters, 3.0, r0_nao=0.315, r0_clh=0.28, r0_clo=0.38)
    Y = h.Y(torch.tensor(x + 0.5))
    assert Y.shape == (1, 3)


# ---------------------------------------------------------------- frozen protocol constants
def test_frozen_constants_match_the_published_inputs():
    assert nsys.TEMPERATURE_K == 300.0            # abf.conf langevinTemp
    assert nsys.GAMMA_PS == 1.0                   # abf.conf langevinDamping
    assert nsys.DT_PS == 0.002                    # abf.conf timestep 2.0 fs
    assert (nsys.SWITCH_NM, nsys.CUTOFF_NM) == (1.00, 1.20)   # switchdist 10 / cutoff 12
    assert nsys.PME_TOLERANCE == 1.0e-5           # PMETolerance 10e-6
    assert nsys.N_SITES == 2465 and nsys.N_WATERS == 821
