"""Ethane/LTA engine additions of the histogram-estimator study (docs/LTA_HISTOGRAM_REPLICATION.md):

* the legacy kernel engine is BIT-IDENTICAL to the pre-change engine (fixture generated from the
  engine at 56ab774 before the edit) and every legacy ``config_hash`` is unchanged;
* the histogram (P0) branch: own-bin bias, exact piecewise-linear PMF, gamma = 0 identity;
* the matched sham replays its partner's realised per-opportunity counts exactly;
* the movie record is bit-inert and consistent.
All CPU, float64, small.
"""
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from alkanes import periodic as per                                          # noqa: E402
from lta.core_lta import (LTAParams, LTASimConfig, LTASystem, bin_index,     # noqa: E402
                          histogram_mean_force, histogram_pmf, run_sampler)

FIX = os.path.join(HERE, "fixtures", "lta_pre_histogram_fixture.npz")
CPU = torch.device("cpu")
PI = math.pi


@pytest.fixture(scope="module")
def system():
    return LTASystem(LTAParams(), CPU, root=ROOT)


def _sim(**kw):
    base = dict(n_steps=1500, n_replicas=64, save_every=500, abf_warmup_steps=200,
                estimator_burn_in_steps=200, fr_start_steps=300, fr_rate=20.0, rng_seed=11)
    base.update(kw)
    return LTASimConfig(**base)


def test_legacy_engine_bit_identical_to_fixture(system):
    z = np.load(FIX, allow_pickle=True)
    sim = LTASimConfig(**json.loads(str(z["sim_json"])))
    assert sim.abf_estimator == "kernel" and sim.config_hash() == str(z["config_hash"])
    for m in ("abf", "fr_uniform"):
        o = run_sampler(m, system, sim, seeds=[0, 1], verbose=False)
        for k in ("pmf", "mean_force", "p_hat", "eff_counts", "total_replacement_events",
                  "n_cage_crossings", "ancestor_ess", "kl_uniform", "u_of_z", "birth_hist", "death_hist"):
            assert np.array_equal(np.asarray(o[k]), z[f"{m}/{k}"], equal_nan=True), (m, k)
        if m == "fr_uniform":
            assert int(o["total_replacement_events"].sum()) > 0            # the fixture exercises FR
            assert o["event_counts"].shape[1] == 2 and o["event_counts"].sum() == o["total_replacement_events"].sum()


def test_sweep_config_hash_unchanged():
    s = json.load(open(os.path.join(ROOT, "configs/uniform_campaign/lta_sweep_prereg.json")))["sampler"]
    sim = LTASimConfig(n_steps=s["n_steps"], n_replicas=s["n_replicas"], dt=s["dt"], save_every=s["save_every"],
                       n_grid=s["n_grid"], abf_bandwidth=s["abf_bandwidth"], kde_bandwidth=s["kde_bandwidth"],
                       abf_warmup_steps=s["abf_warmup_steps"], abf_force_clip=s["abf_force_clip"],
                       estimator_burn_in_steps=s["estimator_burn_in_steps"], fr_start_steps=s["fr_start_steps"],
                       fr_every=s["fr_every"], score_clip=s["score_clip"], max_event_fraction=s["max_event_fraction"],
                       target_ema_rate=s["target_ema_rate"], fr_rate=0.2, rng_seed=20260904)
    assert sim.config_hash() == "afe2a9f9cf6d"           # meta of results/uniform_campaign/lta/production_T300/abf.npz
    # the record flags never enter the hash; the estimator does once it leaves its default
    assert LTASimConfig(**{**sim.__dict__, "store_snapshots": 7, "snapshot_seed_index": 1}).config_hash() == "afe2a9f9cf6d"
    assert LTASimConfig(**{**sim.__dict__, "abf_estimator": "histogram"}).config_hash() != "afe2a9f9cf6d"


def test_histogram_pmf_exact_and_bin_index():
    g = torch.Generator().manual_seed(3)
    n = 180
    grid, dphi = per.periodic_grid(n)
    gamma = torch.randn(2, n, generator=g, dtype=torch.float64)
    F = histogram_pmf(gamma, dphi)
    g0 = (gamma - gamma.mean(-1, keepdim=True)).numpy()
    # independent reconstruction: F at edges by cumulative sum, centre = edge + half step
    for r in range(2):
        Fe = np.concatenate([[0.0], np.cumsum(g0[r] * dphi)])
        Fc = Fe[:-1] + 0.5 * g0[r] * dphi
        Fc -= Fc.mean()
        np.testing.assert_allclose(F[r].numpy(), Fc, atol=1e-12)
        assert abs(Fe[-1]) < 1e-12                        # closed on the circle
        # the derivative of the exact PMF between consecutive centres is the bin-boundary mean of Gamma
        dF = np.diff(np.concatenate([Fc, Fc[:1]]))
        np.testing.assert_allclose(dF[:-1], 0.5 * (g0[r][:-1] + g0[r][1:]) * dphi, atol=1e-12)
    # own-bin rule: bin_index deposits exactly where bin_counts counts
    phi = (torch.rand(5000, generator=g, dtype=torch.float64) * 2 - 1) * PI
    idx = bin_index(phi, n)
    counts = torch.zeros(n, dtype=torch.float64).scatter_add_(0, idx, torch.ones_like(phi))
    assert torch.equal(counts, per.bin_counts(phi, n))
    # P0 mean force: M/C on populated bins, 0 elsewhere
    M = torch.tensor([[1.0, 0.0, -3.0]]); C = torch.tensor([[2.0, 0.0, 3.0]])
    assert torch.equal(histogram_mean_force(M, C), torch.tensor([[0.5, 0.0, -1.0]]))


def test_histogram_branch_runs_and_gamma_zero_identity(system):
    sim = _sim(abf_estimator="histogram")
    a = run_sampler("abf", system, sim, seeds=[0, 1], verbose=False)
    assert a["abf_estimator"] == "histogram" and np.isfinite(a["pmf"]).all()
    # saved mean_force is the raw P0 profile and eff_counts the raw counts
    assert np.array_equal(a["eff_counts"][-1], a["final_eff_counts"])
    assert np.all(a["final_eff_counts"] == np.round(a["final_eff_counts"]))
    assert abs(a["final_eff_counts"].sum() - 2 * 64 * (sim.n_steps + 1)) < 1e-6
    # the saved pmf is the exact integral of the saved mean force
    F = histogram_pmf(torch.as_tensor(a["mean_force"][-1]), 2 * PI / sim.n_grid).numpy()
    np.testing.assert_allclose(a["pmf"][-1], F, atol=1e-10)
    # gamma = 0 fr_uniform == abf bitwise (this engine's _birth_death draws no RNG at rate 0)
    u0 = run_sampler("fr_uniform", system, LTASimConfig(**{**sim.__dict__, "fr_rate": 0.0}),
                     seeds=[0, 1], verbose=False)
    for k in ("pmf", "mean_force", "p_hat", "n_cage_crossings"):
        assert np.array_equal(a[k], u0[k]), k
    assert u0["total_replacement_events"].sum() == 0
    # with a rate, FR fires and changes the state
    u = run_sampler("fr_uniform", system, sim, seeds=[0, 1], verbose=False)
    assert u["total_replacement_events"].sum() > 0 and not np.array_equal(a["pmf"][-1], u["pmf"][-1])


@pytest.mark.parametrize("estimator", ["kernel", "histogram"])
def test_sham_replays_partner_counts(system, estimator):
    sim = _sim(abf_estimator=estimator)
    u = run_sampler("fr_uniform", system, sim, seeds=[0, 1], verbose=False)
    counts = u["event_counts"]
    n_opps = (sim.n_steps - sim.fr_start_steps) // sim.fr_every + 1
    assert counts.shape == (n_opps, 2) and counts.sum() == u["total_replacement_events"].sum() > 0
    s = run_sampler("fr_sham", system, sim, seeds=[0, 1], shadow_counts=counts, verbose=False)
    assert np.array_equal(s["event_counts"], counts)                      # same schedule, exactly
    assert np.array_equal(s["total_replacement_events"], u["total_replacement_events"])
    assert np.isfinite(s["pmf"]).all() and np.all(s["fr_score_std"] == 0)   # no score computed
    assert np.isfinite(s["kl_pq"][-1]).all()                              # uniform diagnostics carried
    a = run_sampler("abf", system, sim, seeds=[0, 1], verbose=False)
    assert not np.array_equal(a["pmf"][-1], s["pmf"][-1])                 # it did resample
    # lineage bookkeeping is live for the sham: ancestors are copied with the configuration
    assert np.all(np.asarray(s["n_unique_ancestor"][-1]) <= 64 - 0) and np.any(np.asarray(s["ancestor_ess"][-1]) < 64)
    # refusals: sham without counts, wrong shape, counts handed to a non-sham
    with pytest.raises(ValueError):
        run_sampler("fr_sham", system, sim, seeds=[0, 1], verbose=False)
    with pytest.raises(ValueError):
        run_sampler("fr_sham", system, sim, seeds=[0], shadow_counts=counts, verbose=False)
    with pytest.raises(ValueError):
        run_sampler("fr_uniform", system, sim, seeds=[0, 1], shadow_counts=counts, verbose=False)


@pytest.mark.parametrize("estimator", ["kernel", "histogram"])
def test_snapshots_bit_inert_and_consistent(system, estimator):
    sim = _sim(abf_estimator=estimator)
    k, N = 100, 64
    for m in ("abf", "fr_uniform"):
        a = run_sampler(m, system, sim, seeds=[0, 1], verbose=False)
        b = run_sampler(m, system, LTASimConfig(**{**sim.__dict__, "store_snapshots": k, "snapshot_seed_index": 1}),
                        seeds=[0, 1], verbose=False)
        for key in ("pmf", "mean_force", "p_hat", "total_replacement_events", "n_cage_crossings", "event_counts"):
            assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), (m, key)
        assert "snap_q" not in a
        n_snap = sim.n_steps // k + 1
        assert b["snap_steps"].shape == (n_snap,) and b["snap_steps"][-1] == sim.n_steps and b["snap_seed"] == 1
        assert b["snap_q"].shape == (n_snap, N, 2, 3) and b["snap_q"].dtype == np.float32
        assert b["snap_phi"].shape == (n_snap, N) and b["snap_wid"].shape == (n_snap, N) and b["snap_wid"].dtype == np.int32
        assert b["snap_pmf"].shape == (n_snap, sim.n_grid) and b["snap_counts"].shape == (n_snap, sim.n_grid)
        # recorded CV == CV of the recorded configuration (float32 round-off)
        phi_q = system.cv_value(torch.as_tensor(b["snap_q"][-1], dtype=torch.float64)).numpy()
        d = np.abs(((phi_q - b["snap_phi"][-1]) + PI) % (2 * PI) - PI)
        assert d.max() < 1e-4
        # the recorded PMF is the reported estimator of seed index 1 at the final save
        np.testing.assert_allclose(b["snap_pmf"][-1], a["pmf"][-1][1].astype(np.float32), rtol=0, atol=1e-5)
        assert b["snap_crossings"][-1] == a["n_cage_crossings"][1]
        die, birth = b["snap_ev_die"], b["snap_ev_birth"]
        assert die.shape == birth.shape and die.shape[1] == 5 and np.isfinite(die).all()
        assert len(die) == int(a["total_replacement_events"][1])        # every event of that seed, once
        if len(die):
            assert (die[:, 0] >= 1).all() and (die[:, 0] <= n_snap).all()
            assert np.abs(die[:, 1]).max() <= PI and np.abs(birth[:, 1]).max() <= PI
            # births mint fresh ids: distinct ids seen grow by exactly the event count
            assert b["snap_wid"][-1].max() == N - 1 + len(die) and len(np.unique(b["snap_wid"][0])) == N
            # genealogy: (dying slot, source slot) and (child id, parent id) per event
            slots, ids = b["snap_ev_slots"], b["snap_ev_ids"]
            assert slots.shape == (len(die), 2) and ids.shape == (len(die), 2)
            assert (slots >= 0).all() and (slots < N).all() and (slots[:, 0] != slots[:, 1]).all()
            assert (ids[:, 0] >= N).all() and np.array_equal(ids[:, 0], np.arange(N, N + len(die)))  # minted in order
            assert (ids[:, 1] < ids[:, 0]).all()                                 # a parent is older than its child
            # the child id occupies the dying slot at the next snapshot unless that slot was replaced again
            for (s_, phi_, *_), (dslot, _), (cid, _) in zip(die, slots, ids):
                k_ = int(s_)
                later = ids[(die[:, 0] == s_) & (ids[:, 0] > cid) & (slots[:, 0] == dslot)]
                if len(later) == 0:
                    assert b["snap_wid"][k_][dslot] == cid
            # ancestor labels: a walker's ancestor is its parent's ancestor; the final labels reproduce the
            # engine's own n_unique_ancestor at the final save
            assert b["snap_anc"].shape == (n_snap, N) and len(np.unique(b["snap_anc"][-1])) == int(a["n_unique_ancestor"][-1][1])
            assert np.array_equal(b["snap_anc"][0], np.arange(N))
        else:
            assert m == "abf"
            assert b["snap_ev_slots"].shape == (0, 2) and np.array_equal(b["snap_anc"][-1], np.arange(N))
    assert sim.config_hash() == LTASimConfig(**{**sim.__dict__, "store_snapshots": k}).config_hash()
