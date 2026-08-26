"""Gate 0I: the kappa-family instrument, and the engine wiring that applies it.

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md`` (Stage 2).

Stage 2 compares allocation arms across cells that are supposed to differ in
conditional *difficulty* and in nothing else.  If kappa moved the free energy,
K0/K2/K3 would be three different landscapes and the mirror test would compare
nothing; if it failed to move the mixing time, there would be no heterogeneity
to detect and a tie would prove nothing.  The instrument has to do both, so both
are gated.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_kappa_instrument.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from abffr import kappa_family as kf                                   # noqa: E402
from abffr import potentials                                           # noqa: E402

BETA = 4.0


def test_kappa_cells_are_the_preregistered_four():
    assert set(kf.KAPPA_CELLS) == {"K0", "K1", "K2", "K3"}
    assert kf.KAPPA_CELLS["K0"] == (0.0, 0.0)
    a2, s2 = kf.KAPPA_CELLS["K2"]
    a3, s3 = kf.KAPPA_CELLS["K3"]
    assert a2 == a3, "K3 must relocate the difficulty, not rescale it"
    assert s3 - s2 == pytest.approx(0.5 * kf.H_PERIOD)


@pytest.mark.parametrize("cell", ["K0", "K1", "K2", "K3"])
def test_gate_0I_kappa_never_exceeds_one(cell):
    """Mobility is an effective timestep: kappa > 1 integrates y less accurately.

    Gate 0I measured the consequence -- at kappa = 16 the sampled conditional
    misses exp(-beta V) by 3.3% total variation against 2% at kappa <= 1 -- so a
    parameterization that sped any region up would move F through discretisation
    error, which is the instrument becoming the confound it exists to remove.
    """
    a, shift = kf.KAPPA_CELLS[cell]
    k = kf.kappa_at(np.linspace(-3.0, 3.0, 401), a, shift)
    assert k.max() <= 1.0 + 1e-12
    assert k.min() >= np.exp(-a) - 1e-12


def test_K3_relocates_the_difficulty_rather_than_rescaling_it():
    """``kappa_{-a} = 1 / kappa_a``: same profile, half a period along.

    The mirror test needs K3 to put the *hard* region where K2 put the easy one.
    A cell that merely made everything slower would confirm nothing about
    whether the allocator follows difficulty or follows density.
    """
    x = np.linspace(-3.0, 3.0, 61)
    k2 = kf.kappa_at(x, *kf.KAPPA_CELLS["K2"])
    k3 = kf.kappa_at(x, *kf.KAPPA_CELLS["K3"])
    assert k2.max() / k2.min() == pytest.approx(16.0)
    assert np.allclose(k3, k2[::-1], atol=1e-12), "K3 is K2 reflected in x"
    assert kf.tau_spread(kf.KAPPA_CELLS["K2"][0]) == pytest.approx(16.0)


def test_difficulty_profile_is_asymmetric_where_the_free_energy_is_not():
    """Density and difficulty must not covary, or a win is unattributable.

    ``h`` is odd about the barrier at x = 0 while the potential is very nearly
    even, so an allocation tracking ``q_phys`` and one tracking ``Gamma`` cannot
    produce the same profile.  This is the whole reason the instrument exists.
    """
    x = np.linspace(-2.5, 2.5, 101)
    k2 = kf.kappa_at(x, *kf.KAPPA_CELLS["K2"])
    assert not np.allclose(k2, k2[::-1]), "difficulty must not be x-symmetric"

    y = np.linspace(-2.0, 3.0, 400)
    F = np.array([-np.log(np.trapezoid(
        np.exp(-BETA * potentials.potential_xy(xi, y)), y)) / BETA for xi in x])
    asym_F = np.abs(F - F[::-1]).max() / (F.max() - F.min())
    assert asym_F < 0.15, "the potential should be nearly symmetric"


def test_config_selects_a_cell_and_refuses_an_ambiguous_one():
    assert kf.cell_from_config({}) == (0.0, 0.0)
    assert kf.cell_from_config({"kappa": {"cell": "K2"}}) == kf.KAPPA_CELLS["K2"]
    assert kf.cell_from_config({"kappa": {"a": 0.5}}) == (0.5, 0.0)
    with pytest.raises(ValueError, match="not both"):
        kf.cell_from_config({"kappa": {"cell": "K2", "a": 0.5}})
    with pytest.raises(ValueError, match="unknown kappa cell"):
        kf.cell_from_config({"kappa": {"cell": "K9"}})


def test_K0_takes_the_unmodified_arithmetic_path():
    """A K0 run must be the clean-v2 run, not one multiplied by a float 1.0."""
    X = torch.linspace(-2.0, 2.0, 7)
    assert kf.kappa_at_torch(X, 0.0) is None
    a, shift = kf.KAPPA_CELLS["K2"]
    kap = kf.kappa_at_torch(X, a, shift)
    assert kap is not None and torch.all(kap > 0)
    assert torch.allclose(kap, torch.tensor(kf.kappa_at(X.numpy(), a, shift)))


def test_engine_applies_the_mobility_form_to_both_drift_and_noise():
    """Scaling only one of drift or noise would change the hidden temperature.

    Read out of the engine source: the y-update must carry ``kap *`` on the
    drift and ``sqrt(kap) *`` on the noise, and must be evaluated at X.
    """
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "src", "abffr", "simulation_torch.py")).read()
    assert "kap = kfam.kappa_at_torch(X, kappa_a, kappa_shift)" in src, \
        "kappa must read X, not Y"
    assert "Y + (-kap * dvdy) * dt" in src
    assert "torch.sqrt(kap) * noise_scale * noise_y" in src


@pytest.mark.parametrize("kap", [1.0, 0.25, 0.0625])
def test_gate_0I_integrator_preserves_the_conditional_at_fixed_x(kap):
    """The implemented y-integrator must sample ``exp(-beta V)`` for any kappa.

    Continuum invariance is exact -- the flux of ``exp(-beta V)`` vanishes
    pointwise, so any positive ``kappa(x)`` leaves it zero -- but Euler stepping
    and wall reflection are ``O(dt)`` approximations, and it is the *implemented*
    sampler the campaign runs.  x is held fixed so this isolates the y-channel
    and needs no barrier crossing to converge.
    """
    x0, dt, n_steps, n_part = 1.5, 0.002, 40_000, 4096
    ymin, ymax = -2.5, 3.5

    rng = np.random.default_rng(7)
    y = rng.uniform(ymin, ymax, n_part)
    x = np.full(n_part, x0)
    scale = np.sqrt(2.0 * kap * dt / BETA)
    hist = np.zeros(120)
    for s in range(n_steps):
        y = y - kap * potentials.dVdy_xy(x, y) * dt + scale * rng.normal(size=n_part)
        for _ in range(3):
            y = np.where(y < ymin, 2 * ymin - y, y)
            y = np.where(y > ymax, 2 * ymax - y, y)
        if s > n_steps // 4 and s % 20 == 0:
            hist += np.histogram(y, bins=120, range=(ymin, ymax))[0]

    edges = np.linspace(ymin, ymax, 121)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    got = hist / np.trapezoid(hist, ctr)
    want = np.exp(-BETA * potentials.potential_xy(x0, ctr))
    want = want / np.trapezoid(want, ctr)
    tv = 0.5 * np.trapezoid(np.abs(got - want), ctr)
    assert tv < 0.02, f"kappa={kap:.4g} total-variation error {tv:.4f}"
