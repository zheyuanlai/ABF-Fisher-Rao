"""Tests for the pentane R15 timestep / FR campaign (docs/PENTANE_R15_DT_FR.md).

Physical-time scaling and the FR event clock; birth--death invariants; inertness of the additive engine
instrumentation (and byte-identity with the frozen sampler); the numerical identities the timestep audit
relies on (forces, mean-force formula, EM harmonic variance); the exact-sample generator.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
torch.set_default_dtype(torch.float64)

from alkanes import core as core_mod, core_dist as cd, potentials as pot, dt_validation as dv   # noqa: E402
from alkanes.distance_cv import DistanceCV                                                       # noqa: E402
from alkanes.dt_schedule import physical_schedule                                                 # noqa: E402

DTS = [5.0e-4, 2.5e-4, 1.25e-4, 6.25e-5]
FROZEN_COMMIT = "e82b576d"            # main before this campaign touched core_dist


def _params(clip=200.0):
    return pot.AlkaneParams(n_atoms=5, beta=2.0, sigma=2.3, epsilon=1.0, decouple=False, force_clip=clip)


# ---------------------------------------------------------------------------
# physical-time scaling and the FR clock
# ---------------------------------------------------------------------------
def test_physical_schedule_reproduces_frozen_production():
    s = physical_schedule(5.0e-4)
    assert (s["n_steps"], s["abf_warmup_steps"], s["estimator_burn_in_steps"], s["fr_start_steps"], s["fr_every"]) == (80000, 5000, 6000, 12000, 5)


@pytest.mark.parametrize("dt", DTS)
def test_physical_schedule_scales_every_knob(dt):
    s = physical_schedule(dt)
    k = 5.0e-4 / dt
    assert s["n_steps"] == round(80000 * k) and s["abf_warmup_steps"] == round(5000 * k)
    assert s["estimator_burn_in_steps"] == round(6000 * k) and s["fr_start_steps"] == round(12000 * k)
    assert s["fr_every"] == round(5 * k)
    assert s["fr_dt_eff"] == pytest.approx(0.0025, rel=0, abs=1e-15)
    assert s["fr_start_steps"] * dt == pytest.approx(6.0, abs=1e-12)


def test_schedule_rejects_non_integer_times():
    with pytest.raises(ValueError):
        physical_schedule(3.0e-4)          # 0.0025 / 3e-4 is not an integer


def _event_times(dt):
    s = physical_schedule(dt, T=6.1)
    ev = [nxt * dt for nxt in range(1, s["n_steps"] + 1)
          if nxt >= s["fr_start_steps"] and (nxt - s["fr_start_steps"]) % s["fr_every"] == 0]
    return np.array(ev)


def test_fr_event_times_identical_in_physical_time():
    ref = _event_times(5.0e-4)
    assert ref[0] == pytest.approx(6.0) and len(ref) == 41
    for dt in DTS[1:]:
        ev = _event_times(dt)
        assert len(ev) == len(ref)
        np.testing.assert_allclose(ev, ref, atol=1e-12)


class _Sim:
    def __init__(self, dt, fr_every, rate=0.02, mef=0.01):
        self.dt, self.fr_every, self.fr_rate, self.max_event_fraction = dt, fr_every, rate, mef


def test_birth_death_probability_identical_across_timesteps():
    """Identical score + identical FR stream => identical events at every dt with dt*fr_every fixed."""
    R, N = 3, 4096
    g = torch.Generator().manual_seed(1)
    score = core_mod._recentered_clipped_score(torch.randn(R, N, generator=g) * 3, 2.0)
    q = torch.randn(R, N, 5, 3, generator=g)
    anc = torch.arange(N).expand(R, N).clone()
    outs = []
    for dt in DTS:
        s = physical_schedule(dt)
        sim = _Sim(dt, s["fr_every"], rate=50.0, mef=0.05)         # high rate so events actually fire
        gen_fr = torch.Generator().manual_seed(99)
        outs.append(core_mod._birth_death(q, score, anc, sim, gen_fr))
    for o in outs[1:]:
        assert torch.equal(o[0], outs[0][0]) and torch.equal(o[1], outs[0][1]) and torch.equal(o[2], outs[0][2])
    assert int(outs[0][2].sum()) > 0
    # and the per-event death probability is what the code claims
    sim = _Sim(5e-4, 5, rate=0.02)
    dp = 1.0 - torch.exp(-sim.fr_rate * score.clamp_min(0) * sim.dt * sim.fr_every)
    smax = float(score.max())                               # recentring after the last clamp lets it exceed 2 slightly
    assert 1.99 < smax < 2.01
    assert float(dp.max()) == pytest.approx(1.0 - math.exp(-0.02 * smax * 0.0025), rel=1e-9)


def test_birth_death_preserves_population_and_clones_whole_configurations():
    R, N = 2, 512
    g = torch.Generator().manual_seed(3)
    score = core_mod._recentered_clipped_score(torch.randn(R, N, generator=g) * 3, 2.0)
    q = torch.randn(R, N, 5, 3, generator=g)
    anc = torch.arange(N).expand(R, N).clone()
    qn, an, nrep, deaths, births = core_mod._birth_death(q, score, anc, _Sim(5e-4, 5, rate=200.0, mef=0.05), torch.Generator().manual_seed(5))
    assert qn.shape == q.shape and an.shape == anc.shape
    for r in range(R):
        assert deaths[r] is not None and deaths[r].numel() == int(nrep[r]) <= int(0.05 * N)
        assert torch.equal(qn[r, deaths[r]], q[r, births[r]])                   # all 5 atoms x 3 coordinates copied
        assert torch.equal(an[r, deaths[r]], anc[r, births[r]])
        keep = torch.ones(N, dtype=torch.bool); keep[deaths[r]] = False
        assert torch.equal(qn[r, keep], q[r, keep])
        assert bool((score[r, deaths[r]] > 0).all()) and bool((score[r, births[r]] < 0).all())


# ---------------------------------------------------------------------------
# engine instrumentation: inert, and byte-identical to the frozen sampler
# ---------------------------------------------------------------------------
def _tiny_sim(**kw):
    base = dict(dt=5e-4, n_steps=300, n_replicas=48, save_every=100, rng_seed=7, abf_warmup_steps=50, estimator_burn_in_steps=60,
                fr_rate=400.0, fr_start_steps=120, fr_every=5, max_event_fraction=0.05)
    base.update(kw)
    return cd.DistSimConfig(**base)


def _run(method, sim, force_fn=None):
    return cd.run_sampler_dist(method, _params(), sim, [0, 1], DistanceCV(0, 4), "cpu", initial_dihedrals=[0.0, 0.0],
                               collect_conditional=True, verbose=False, force_fn=force_fn)


KEYS = ("pmf", "mean_force", "p_hat", "eff_counts", "total_replacement_events", "ancestor_ess", "cond_hist", "fsum_prod",
        "csum_prod", "birth_hist", "n_transitions")


@pytest.mark.parametrize("method", ["abf", "fr_uniform"])
def test_instrumentation_is_inert(method):
    torch.set_num_threads(1)
    a = _run(method, _tiny_sim())
    b = _run(method, _tiny_sim(numerics_diag=True, track_families=True, deposit_stride=1))
    for k in KEYS:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)
    assert b["fam_count"].shape[1:] == (2, 4) and np.all(b["fam_count"].sum(-1) == 48)
    assert np.all(b["n_nonfinite_walker_steps"] == 0) and np.all(b["max_step_displacement"] > 0)


def test_clip_stats_force_wrapper_is_bitwise_production():
    sys.path.insert(0, os.path.join(ROOT, "scripts", "pentane_r15_dt_fr"))
    from run_arm import ClipStatsForces
    q, _ = dv.exact_gibbs_pool(256, _params(), torch.Generator().manual_seed(0))
    q = q + 0.05 * torch.randn(q.shape, generator=torch.Generator().manual_seed(1))     # strained -> some clips fire
    ff = ClipStatsForces(2, 128)
    assert torch.equal(ff(q, _params()), pot.forces(q, _params()))
    assert ff.n_clip.sum() > 0


def test_abf_and_fr_arms_identical_before_fr_starts():
    torch.set_num_threads(1)
    sim = _tiny_sim(save_every=20, fr_start_steps=120)
    a = _run("abf", sim); f = _run("fr_uniform", sim)
    pre = np.asarray(a["steps"]) < 120
    np.testing.assert_array_equal(a["pmf"][pre], f["pmf"][pre])
    np.testing.assert_array_equal(a["p_hat"][pre], f["p_hat"][pre])
    assert int(f["total_replacement_events"].sum()) > 0
    assert not np.array_equal(a["pmf"][-1], f["pmf"][-1])


def test_deposit_stride_counts():
    torch.set_num_threads(1)
    out = _run("abf", _tiny_sim(deposit_stride=4, n_steps=200, estimator_burn_in_steps=60))
    n_dep_prod = len([s for s in range(201) if s % 4 == 0 and s >= 60])
    assert np.allclose(np.asarray(out["csum_prod"]).sum(-1), n_dep_prod * 48)


def test_byte_identical_to_frozen_sampler(tmp_path):
    """The default-config sampler equals the frozen main-branch core_dist bit for bit (abf and fr_uniform)."""
    try:
        frozen = subprocess.check_output(["git", "-C", ROOT, "show", f"{FROZEN_COMMIT}:src/alkanes/core_dist.py"], text=True)
    except Exception:
        pytest.skip("git history unavailable")
    pkg = tmp_path / "src"
    shutil.copytree(os.path.join(ROOT, "src", "alkanes"), pkg / "alkanes")
    (pkg / "alkanes" / "core_dist.py").write_text(frozen)
    code = r"""
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[1]); torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
from alkanes import core_dist as cd, potentials as pot
from alkanes.distance_cv import DistanceCV
p = pot.AlkaneParams(n_atoms=5, beta=2.0, sigma=2.3, epsilon=1.0, decouple=False, force_clip=200.0)
out = {}
for m in ("abf", "fr_uniform"):
    sim = cd.DistSimConfig(dt=5e-4, n_steps=300, n_replicas=48, save_every=100, rng_seed=7, abf_warmup_steps=50, estimator_burn_in_steps=60,
                           fr_rate=400.0, fr_start_steps=120, fr_every=5, max_event_fraction=0.05)
    o = cd.run_sampler_dist(m, p, sim, [0, 1], DistanceCV(0, 4), "cpu", initial_dihedrals=[0.0, 0.0], collect_conditional=True, verbose=False)
    for k in ("pmf", "mean_force", "total_replacement_events", "cond_hist", "ancestor_ess", "fsum_prod"):
        out[f"{m}_{k}"] = np.asarray(o[k])
np.savez(sys.argv[2], **out)
"""
    script = tmp_path / "run.py"; script.write_text(code)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    subprocess.check_call([sys.executable, str(script), str(pkg), str(tmp_path / "frozen.npz")], env=env)
    subprocess.check_call([sys.executable, str(script), os.path.join(ROOT, "src"), str(tmp_path / "new.npz")], env=env)
    a = np.load(tmp_path / "frozen.npz"); b = np.load(tmp_path / "new.npz")
    for k in a.files:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


# ---------------------------------------------------------------------------
# numerical identities used by the timestep audit
# ---------------------------------------------------------------------------
def test_forces_match_finite_differences():
    p0 = _params(0.0)
    q, _ = dv.exact_gibbs_pool(16, p0, torch.Generator().manual_seed(11))
    f = pot.forces(q, p0); h = 1e-6; fd = torch.zeros_like(q)
    for a in range(5):
        for c in range(3):
            e = torch.zeros_like(q); e[:, a, c] = h
            fd[:, a, c] = -(pot.total_energy(q + e, p0) - pot.total_energy(q - e, p0)) / (2 * h)
    assert float((f - fd).norm() / f.norm()) < 1e-8


def test_local_mean_force_closed_form_and_free_particle_limit():
    cv = DistanceCV(0, 4)
    q, _ = dv.exact_gibbs_pool(64, _params(), torch.Generator().manual_seed(12))
    F = pot.forces(q, _params(0.0))
    f, R, _ = cv.local_mean_force(q, F, 2.0)
    e = (q[:, 4] - q[:, 0]) / R[:, None]
    np.testing.assert_allclose(f.numpy(), (0.5 * (e * (F[:, 0] - F[:, 4])).sum(-1) - 2.0 / (2.0 * R)).numpy(), atol=1e-10)
    # V = 0: F(R) = -beta^-1 log R^2, F' = -2/(beta R)
    f0, R0, _ = cv.local_mean_force(q, torch.zeros_like(F), 2.0)
    np.testing.assert_allclose(f0.numpy(), (-2.0 / (2.0 * R0)).numpy(), atol=1e-14)


@pytest.mark.parametrize("lam_dt", [0.2, 0.7, 1.4])
def test_em_harmonic_variance_identity(lam_dt):
    lam, beta = 1000.0, 2.0; dt = lam_dt / lam
    g = torch.Generator().manual_seed(int(lam_dt * 100)); x = torch.zeros(400_000)
    for _ in range(int(40 / lam_dt) + 50):
        x = x - dt * lam * x + math.sqrt(2 * dt / beta) * torch.randn(x.shape, generator=g)
    assert float(x.var()) * beta * lam == pytest.approx(1.0 / (1.0 - lam_dt / 2.0), rel=0.02)


def test_exact_flat_sampler_is_flat_and_inside_window():
    p = _params()
    grid = np.linspace(1.4, 3.7, 257); grid = 0.5 * (grid[1:] + grid[:-1])
    mask = np.zeros(256, bool); mask[150:160] = True
    q, info = dv.sample_exact(200, p, 5, target="flat", n_pool=400_000, chunk=200_000, grid=grid, bias=np.zeros(256), window_mask=mask)
    R = DistanceCV(0, 4).value(q).numpy()
    dz = grid[1] - grid[0]; lo, hi = grid[150] - dz / 2, grid[159] + dz / 2
    assert q.shape[0] == 200 and np.all((R >= lo) & (R < hi))
    counts = np.histogram(R, bins=np.linspace(lo, hi, 11))[0]
    assert counts.min() >= 18 and counts.max() <= 20


def test_family_classifier():
    g = math.radians(116.57)
    phi1 = torch.tensor([0.0, 0.0, g, g, -g]); phi2 = torch.tensor([0.0, g, g, -g, -g])
    assert dv.family_of(phi1, phi2).tolist() == [0, 1, 2, 3, 2]
