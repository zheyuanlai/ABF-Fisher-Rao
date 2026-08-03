"""Tests for the WCA matched-sham arms.

The WCA sampler runs one method per process, so a sham cannot watch its partner online: it
replays a per-event count sequence recorded from the partner's own run. These pin the
properties that makes that replay a valid control.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import wca_abffr_core as wca


def _tiny():
    """A deliberately small dimer: enough to exercise the code path, not the physics."""
    params = wca.DimerWCAParams(n_dim=4, beta=1.0, h=1.0, w=2.0, a=1.5)
    sim = wca.SimConfig(n_replicas=64, n_steps=3000, save_every=1000, dt=2e-3,
                        n_grid=48, fr_start_steps=500, fr_every=25,
                        abf_warmup_steps=200, estimator_burn_in_steps=200,
                        fr_rate=0.5, max_event_fraction=0.1, seed=7)
    engine = wca.WCADimerEngine(params, device=torch.device("cpu"), dtype=torch.float64)
    return params, sim, engine


def _run(method, **kw):
    params, sim, engine = _tiny()
    return wca.run_sampler_gpu(method, params, sim, engine, collect_diagnostics=True,
                               verbose=False, **kw)


# ------------------------------------------------------------------ registry
def test_sham_arms_are_registered_and_are_fr_arms():
    for m in wca.SHAM_METHODS:
        assert m in wca.ALL_METHODS
    assert wca.SHAM_PARTNER["sham_practical"] == "fr_estimated"
    assert wca.SHAM_PARTNER["sham_oracle"] == "fr_oracle"


def test_sham_refuses_to_run_without_a_schedule():
    """A sham with nothing to replay would silently become plain ABF and still be scored."""
    with pytest.raises(ValueError, match="requires replay_counts"):
        _run("sham_practical")


def test_replay_counts_rejected_for_non_sham_methods():
    with pytest.raises(ValueError, match="only meaningful"):
        _run("fr_estimated", replay_counts=[1, 2, 3])


def test_sham_cannot_receive_an_oracle_reference():
    params, sim, engine = _tiny()
    with pytest.raises(AssertionError, match="NO-ORACLE-LEAKAGE"):
        wca.run_sampler_gpu("sham_oracle", params, sim, engine,
                            oracle_free_energy=np.zeros(sim.n_grid),
                            replay_counts=[0], verbose=False)


# ------------------------------------------------------------- replay fidelity
def test_sham_reproduces_its_partners_event_schedule_exactly():
    fr = _run("fr_estimated")
    counts = fr["fr_event_counts"]
    assert counts.sum() > 0, "the partner fired nothing; the test would be vacuous"
    sh = _run("sham_practical", replay_counts=counts)
    np.testing.assert_array_equal(sh["fr_event_counts"], counts)
    assert sh["total_replacement_events"] == fr["total_replacement_events"]
    assert sh["sham_replayed_events"] == len(counts)


def test_event_counts_have_one_slot_per_opportunity():
    """Zero-event opportunities must still occupy a slot.

    If they were dropped, every later count would be replayed at the wrong opportunity and
    the sham's schedule would be silently shifted relative to its partner's.
    """
    _, sim, _ = _tiny()
    fr = _run("fr_estimated")
    # the sampler fires at next_step = step+1 for step in [0, n_steps)
    n_opp = sum(1 for step in range(sim.n_steps)
                if (step + 1) >= sim.fr_start_steps
                and ((step + 1) - sim.fr_start_steps) % sim.fr_every == 0)
    assert len(fr["fr_event_counts"]) == n_opp, (len(fr["fr_event_counts"]), n_opp)


def test_sham_and_partner_differ_in_outcome():
    """Matched intensity, different direction => the trajectories must not coincide."""
    fr = _run("fr_estimated")
    sh = _run("sham_practical", replay_counts=fr["fr_event_counts"])
    assert not np.allclose(np.asarray(fr["pmf"][-1]), np.asarray(sh["pmf"][-1]))


def test_the_two_partners_fire_different_schedules():
    """Why two shams are needed rather than one.

    fr_estimated and fr_oracle build different targets, so a single sham shadowing the
    oracle is not an intensity-matched control for the practical arm.
    """
    params, sim, engine = _tiny()
    est = wca.run_sampler_gpu("fr_estimated", params, sim, engine, verbose=False)
    orc = wca.run_sampler_gpu("fr_oracle", params, sim, engine, verbose=False,
                              oracle_free_energy=np.linspace(0.0, 2.0, sim.n_grid))
    assert est["total_replacement_events"] != orc["total_replacement_events"]


# ------------------------------------------------- the replacement step itself
def test_uniform_birth_death_kills_and_refills_exactly_n():
    q = torch.arange(40, dtype=torch.float64).reshape(20, 2)
    anc = torch.arange(20)
    q2, anc2, stats = wca.uniform_birth_death_torch(q, 5, ancestors=anc)
    assert stats["replacement"] == 5
    assert q2.shape == q.shape
    assert stats["death_idx"].numel() == 5 and stats["birth_src"].numel() == 5
    # clone sources must be survivors, never one of the dead
    assert len(set(stats["birth_src"].tolist()) & set(stats["death_idx"].tolist())) == 0
    # the killed slots now hold copies of their sources, whole-replica
    for d, s in zip(stats["death_idx"].tolist(), stats["birth_src"].tolist()):
        torch.testing.assert_close(q2[d], q[s])
        assert anc2[d] == anc[s]
    # lineage count falls by exactly the number of replacements
    assert len(set(anc2.tolist())) == 20 - 5


def test_uniform_birth_death_is_a_noop_at_zero():
    q = torch.randn(16, 3, dtype=torch.float64)
    q2, _, stats = wca.uniform_birth_death_torch(q, 0)
    assert stats["replacement"] == 0
    torch.testing.assert_close(q2, q)


def test_uniform_birth_death_leaves_a_survivor():
    """Asking to kill everything must still leave something to clone from."""
    q = torch.randn(8, 2, dtype=torch.float64)
    _, _, stats = wca.uniform_birth_death_torch(q, 99)
    assert stats["replacement"] == 7
