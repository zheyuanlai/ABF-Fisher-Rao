"""Sampler-level validation for the distance (1-D) and joint-torsion (2-D) CV samplers:
no-reference-leakage, whole-configuration cloning, genealogy, matched seeds, determinism.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alkanes_cv_samplers.py -q
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from alkanes import potentials as pot, core_dist as cd, core2d as c2  # noqa: E402
from alkanes.distance_cv import DistanceCV  # noqa: E402
from alkanes.cv2d import JointDihedralCV2D  # noqa: E402


# ------------------------------- distance CV (1-D) -------------------------------
def _dsim(**kw):
    base = dict(dt=5e-4, n_steps=300, n_replicas=32, save_every=150, rng_seed=7,
                R_lo=1.35, R_hi=2.82, wall_lo=1.4, wall_hi=2.78, n_grid=96,
                abf_warmup_steps=50, estimator_burn_in_steps=50, fr_start_steps=100,
                fr_every=5, fr_rate=0.6, max_event_fraction=0.1)
    base.update(kw)
    return cd.DistSimConfig(**base)


def _drun(method, seeds=(0, 1), oracle=None, **kw):
    p = pot.AlkaneParams(n_atoms=4, beta=1.0, decouple=True, force_clip=200.0)
    cv = DistanceCV(0, 3)
    return cd.run_sampler_dist(method, p, _dsim(**kw), list(seeds), cv, "cpu",
                               oracle_free_energy=oracle, verbose=False)


def test_dist_no_leakage():
    ref = np.zeros(96)
    for m in ("abf", "fr_estimated", "fr_uniform"):
        with pytest.raises(AssertionError):
            _drun(m, oracle=ref)
    with pytest.raises(ValueError):
        _drun("fr_oracle", oracle=None)


def test_dist_deterministic_and_genealogy():
    a = _drun("fr_estimated")
    b = _drun("fr_estimated")
    assert np.allclose(a["pmf"][-1], b["pmf"][-1])
    assert np.array_equal(a["total_replacement_events"], b["total_replacement_events"])
    out = _drun("fr_uniform", fr_rate=2.0, max_event_fraction=0.2, n_steps=500)
    assert out["total_replacement_events"].sum() > 0
    ess = out["ancestor_ess"][-1]
    assert np.all(ess <= 32 + 1e-6) and np.all(ess > 0)


def test_dist_matched_seeds_without_events():
    abf = _drun("abf", fr_start_steps=10 ** 9)
    fr = _drun("fr_uniform", fr_start_steps=10 ** 9)
    assert np.allclose(abf["pmf"][-1], fr["pmf"][-1], atol=1e-10)


# ------------------------------- joint torsion CV (2-D) -------------------------------
def _s2(**kw):
    base = dict(dt=5e-4, n_steps=200, n_replicas=48, save_every=100, rng_seed=7, n_grid=24,
                abf_warmup_steps=40, estimator_burn_in_steps=40, fr_start_steps=80,
                fr_every=5, fr_rate=0.6, max_event_fraction=0.1)
    base.update(kw)
    return c2.Sim2DConfig(**base)


def _run2(method, seeds=(0, 1), oracle=None, **kw):
    p = pot.AlkaneParams(n_atoms=5, beta=1.0, sigma=2.3, decouple=True, force_clip=200.0)
    cv = JointDihedralCV2D()
    return c2.run_sampler_2d(method, p, _s2(**kw), list(seeds), cv, "cpu",
                             oracle_free_energy=oracle, verbose=False)


def test_2d_no_leakage():
    ref = np.zeros((24, 24))
    for m in ("abf", "fr_estimated", "fr_uniform"):
        with pytest.raises(AssertionError):
            _run2(m, oracle=ref)
    with pytest.raises(ValueError):
        _run2("fr_oracle", oracle=None)
    out = _run2("fr_oracle", oracle=ref)
    assert np.isfinite(out["final_pmf"]).all()


def test_2d_whole_config_cloning_and_genealogy():
    out = _run2("fr_uniform", fr_rate=2.0, max_event_fraction=0.2, n_steps=400)
    assert out["total_replacement_events"].sum() > 0
    ess = out["ancestor_ess"][-1]
    nuq = out["n_unique_ancestor"][-1]
    assert np.all(ess <= 48 + 1e-6) and np.all(ess > 0)
    assert np.all(nuq <= 48)
    assert np.any(out["ancestor_ess"][-1] < 48)          # diversity dropped from cloning


def test_2d_deterministic_and_fixed_population():
    a = _run2("fr_estimated")
    b = _run2("fr_estimated")
    assert np.allclose(a["final_pmf"], b["final_pmf"])
    assert a["joint_hist"].shape == (2, 24, 24)
    assert np.array_equal(a["total_replacement_events"], b["total_replacement_events"])


def test_2d_matched_seeds_without_events():
    abf = _run2("abf", fr_start_steps=10 ** 9)
    fr = _run2("fr_uniform", fr_start_steps=10 ** 9)
    assert np.allclose(abf["final_pmf"], fr["final_pmf"], atol=1e-9)


def test_2d_gram_never_singular_on_physical_configs():
    out = _run2("abf", n_steps=300)
    assert out["gram_reg_activations"] == 0           # dihedral Gram is well-conditioned
    assert np.all(out["gram_lam_min_min"] > 1e-3)


def test_2d_estimator_stride_runs_and_matches_shape():
    # strided estimator (Hessian every k steps) runs and preserves outputs/shapes
    a = _run2("abf", n_steps=200, estimator_stride=1)
    b = _run2("abf", n_steps=200, estimator_stride=5)
    assert np.isfinite(a["final_pmf"]).all() and np.isfinite(b["final_pmf"]).all()
    assert a["final_pmf"].shape == b["final_pmf"].shape


def test_2d_frozen_bias_reconstructs():
    p = pot.AlkaneParams(n_atoms=5, beta=1.0, sigma=2.3, decouple=True, force_clip=200.0)
    cv = JointDihedralCV2D()
    sim = _s2(n_steps=200, estimator_burn_in_steps=40)
    learned = np.zeros((24, 24))                       # trivial (flat) frozen bias
    fb = c2.run_frozen_bias_2d(p, sim, learned, [0, 1], cv, "cpu", verbose=False)
    assert np.isfinite(fb["F_recon"]).all()
    assert fb["F_recon"].shape == (2, 24, 24)
    assert np.isfinite(fb["p_B"]).all() and np.all(fb["p_B"] >= 0)


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
