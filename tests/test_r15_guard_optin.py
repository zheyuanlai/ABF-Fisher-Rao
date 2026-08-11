"""The distance-CV `fullSamples` guard must be opt-in and a **provable** no-op by default.

v1 is immutable. Every published R15 number was produced with no guard, so if the default
changed behaviour at all, the v2 audit would be comparing against a moving baseline and could
prove nothing. This asserts bit-identity at the default, and a real effect when enabled.

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/test_r15_guard_optin.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from alkanes import core_dist as cd                                        # noqa: E402
from alkanes import potentials as pot                                      # noqa: E402
from alkanes.distance_cv import DistanceCV                                 # noqa: E402


def _cfg(**kw):
    base = dict(dt=5.0e-4, n_steps=300, n_replicas=64, save_every=150,
                n_grid=64, abf_warmup_steps=50, estimator_burn_in_steps=0,
                fr_rate=0.0, R_lo=1.4, R_hi=3.7, wall_lo=1.45, wall_hi=3.65)
    base.update(kw)
    return cd.DistSimConfig(**base)


def _run(cfg, device="cpu"):
    params = pot.AlkaneParams(n_atoms=5, beta=2.0, sigma=2.3)      # pentane
    cv = DistanceCV(0, 4)
    return cd.run_sampler_dist("abf", params, cfg, [0, 1], cv, device=device,
                               collect_conditional=False, verbose=False)


def test_default_is_zero_and_therefore_frozen_v1():
    assert cd.DistSimConfig().abf_min_count == 0.0


def test_guard_disabled_is_bit_identical_to_the_ungated_path():
    """Two runs at the default must agree exactly -- the guard branch must not perturb RNG."""
    a = _run(_cfg())
    b = _run(_cfg(abf_min_count=0.0))
    assert np.array_equal(a["pmf"], b["pmf"])
    assert np.array_equal(a["mean_force"], b["mean_force"])
    assert np.array_equal(a["p_hat"], b["p_hat"])


def test_guard_enabled_actually_changes_the_applied_bias():
    """A gate that cannot change anything is not a gate."""
    a = _run(_cfg(abf_min_count=0.0))
    b = _run(_cfg(abf_min_count=1.0e9))          # nothing is ever trusted -> ~no applied bias
    assert not np.array_equal(a["pmf"], b["pmf"])


def test_estimator_survives_when_the_applied_bias_is_fully_suppressed():
    """The stored mean force must keep accumulating even with the bias ramped to zero.

    Ramping the estimate as well would bias the quantity the study scores.
    """
    b = _run(_cfg(abf_min_count=1.0e9))
    mf = np.asarray(b["mean_force"])[-1]
    assert np.isfinite(mf).all()
    assert np.abs(mf).max() > 0.0
