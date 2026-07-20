"""Cloning / RNG / genealogy / no-leakage / matched-seed validation (CPU, tiny).

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alkanes_cloning.py -q
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from alkanes import core, potentials as pot  # noqa: E402
from alkanes.cv import DihedralCV  # noqa: E402


def _tiny_sim(**kw):
    base = dict(dt=5e-4, n_steps=400, n_replicas=32, save_every=200, rng_seed=7,
                abf_warmup_steps=50, fr_start_steps=100, estimator_burn_in_steps=50,
                fr_every=5, fr_rate=0.5, max_event_fraction=0.05, n_grid=90)
    base.update(kw)
    return core.AlkaneSimConfig(**base)


# --------------------------- birth-death unit tests ---------------------------
def test_birth_death_clones_full_config_and_fixed_N():
    torch.manual_seed(0)
    R, N, A = 2, 16, 4
    q = torch.randn(R, N, A, 3)
    anc = torch.arange(N).expand(R, N).clone()
    # craft a score with clear deaths (positive) and births (negative)
    score = torch.zeros(R, N)
    score[:, :4] = 1.5      # will die
    score[:, 4:8] = -1.5    # birth pool
    sim = _tiny_sim(max_event_fraction=0.5, fr_rate=5.0)
    gen = torch.Generator().manual_seed(3)
    q2, anc2, n_repl, deaths, births = core._birth_death(q, score, anc, sim, gen)
    assert q2.shape == q.shape                         # fixed population
    for r in range(R):
        if deaths[r] is not None:
            for di, sr in zip(deaths[r].tolist(), births[r].tolist()):
                # the FULL molecular configuration was copied
                assert torch.allclose(q2[r, di], q[r, sr])
                assert anc2[r, di] == anc[r, sr]
                assert sr in range(4, 8)               # cloned from the birth pool


def test_no_source_target_aliasing():
    # a death slot that is ALSO a clone source must copy the ORIGINAL source config
    torch.manual_seed(1)
    R, N, A = 1, 8, 4
    q = torch.randn(R, N, A, 3)
    anc = torch.arange(N).expand(R, N).clone()
    score = torch.tensor([[2.0, 2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0]])
    sim = _tiny_sim(max_event_fraction=0.5, fr_rate=5.0)
    q2, anc2, *_ = core._birth_death(q, score, anc, sim, torch.Generator().manual_seed(9))
    # survivors (indices 2..7) keep their exact original configs (no aliasing corruption)
    for i in range(2, 8):
        assert torch.allclose(q2[0, i], q[0, i])


# --------------------------- integration: determinism / genealogy / RNG ---------------------------
def _run(method, seeds=(0, 1), **kw):
    p = pot.AlkaneParams(n_atoms=4, beta=1.0, decouple=True, force_clip=200.0)
    cv = DihedralCV((0, 1, 2, 3))
    return core.run_sampler(method, p, _tiny_sim(**kw), list(seeds), cv, "cpu", verbose=False)


def test_deterministic_repeatability():
    a = _run("fr_estimated")
    b = _run("fr_estimated")
    assert np.allclose(a["pmf"][-1], b["pmf"][-1])
    assert np.array_equal(a["total_replacement_events"], b["total_replacement_events"])


def test_fixed_population_and_genealogy():
    out = _run("fr_estimated", max_event_fraction=0.2, fr_rate=1.0)
    # ancestor ESS is <= N and unique-ancestor count <= N under resampling
    ess = out["ancestor_ess"][-1]
    nuq = out["n_unique_ancestor"][-1]
    assert np.all(ess <= 32 + 1e-6)
    assert np.all(nuq <= 32)
    assert out["total_replacement_events"].sum() > 0               # FR actually fired
    assert np.all(np.isfinite(ess)) and np.all(ess > 0)


def test_matched_seeds_abf_vs_fr_without_events():
    # with FR disabled (fr never fires), fr_uniform dynamics == abf dynamics exactly,
    # because Langevin noise is on a SEPARATE stream from FR resampling.
    abf = _run("abf", fr_start_steps=10 ** 9)
    fr = _run("fr_uniform", fr_start_steps=10 ** 9)
    assert np.allclose(abf["pmf"][-1], fr["pmf"][-1], atol=1e-10)
    assert np.allclose(abf["mean_force"][-1], fr["mean_force"][-1], atol=1e-10)


def test_cloning_reduces_ancestor_diversity():
    # once cloning has happened, ancestor ESS drops below N (diversity cost of FR).
    out = _run("fr_uniform", n_steps=600, max_event_fraction=0.2, fr_rate=2.0)
    assert out["total_replacement_events"].sum() > 0
    assert np.any(out["ancestor_ess"][-1] < 32)


def test_score_is_zero_mean_and_bounded():
    # zero mean (=> balanced birth/death masses in the fixed-population scheme) is the
    # property _birth_death relies on; the score stays bounded near the clip.
    torch.manual_seed(0)
    raw = torch.randn(3, 64) * 5
    s = core._recentered_clipped_score(raw, 2.0)
    assert s.mean(-1).abs().max() < 1e-9
    assert s.abs().max() <= 2.0 + 0.5     # recenter-after-clip => small overshoot only


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
