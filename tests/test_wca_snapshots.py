"""The WCA movie record (``store_snapshots``) is bit-inert, and its ids and events are consistent."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import wca_abffr_core as wca  # noqa: E402

CPU, F64 = torch.device("cpu"), torch.float64


def _tiny(**kw):
    params = wca.DimerWCAParams(n_dim=4, beta=1.0, h=1.0, w=2.0, a=1.5)
    sim = wca.SimConfig(n_replicas=64, n_steps=3000, save_every=1000, dt=2e-3, n_grid=48, fr_start_steps=500,
                        fr_every=25, abf_warmup_steps=200, estimator_burn_in_steps=200, fr_rate=0.5,
                        max_event_fraction=0.1, seed=7, abf_estimator="histogram", abf_n_bins=24, **kw)
    engine = wca.WCADimerEngine(params, device=CPU, dtype=F64)
    return params, sim, engine


def _run(method, store_snapshots, slots=(0, 5)):
    p, s, e = _tiny()
    return wca.run_sampler_gpu(method, p, s, e, collect_diagnostics=True, verbose=False,
                               store_snapshots=store_snapshots, snapshot_replicas=slots)


def test_recording_snapshots_is_bit_inert():
    for m in ("abf", "fr_uniform"):
        a, b = _run(m, 0), _run(m, 50)
        for k in ("mean_force", "pmf", "hist_mf_bins", "hist_counts", "repl_cumulative", "ancestor_ess",
                  "frac_compact", "frac_stretched", "birth_hist", "death_hist", "fr_event_counts"):
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True), (m, k)
        assert a["total_replacement_events"] == b["total_replacement_events"]
        assert "snap_z" not in a and "snap_z" in b


def test_snapshot_ids_events_and_profiles_are_consistent():
    abf, fr = _run("abf", 50), _run("fr_uniform", 50)
    n_snap = len(abf["snap_steps"])
    assert n_snap == 3000 // 50 + 1 and abf["snap_steps"][-1] == 3000 and abf["snap_steps"][0] == 0
    assert abf["snap_z"].shape == (n_snap, 64) and abf["snap_wid"].shape == (n_snap, 64)
    assert abf["snap_q"].shape == (n_snap, 2, 16, 2) and np.array_equal(abf["snap_replicas"], [0, 5])
    assert abf["snap_pmf"].shape == (n_snap, 48) and abf["snap_mf_bins"].shape == (n_snap, 24)
    # an ABF run never resamples: ids are the identity at every snapshot, no events
    assert np.array_equal(abf["snap_wid"], np.tile(np.arange(64), (n_snap, 1)))
    assert abf["snap_ev_die"].shape == (0, 2) and abf["snap_ev_birth"].shape == (0, 2)
    # FR run: ids unique within every snapshot, never resurrected, population always N
    for f in range(n_snap):
        assert len(np.unique(fr["snap_wid"][f])) == 64, f
    gone, prev = set(), set(fr["snap_wid"][0].tolist())
    for f in range(1, n_snap):
        cur = set(fr["snap_wid"][f].tolist())
        assert not (cur & gone), f
        gone |= prev - cur
        prev = cur
    # every realised replacement is recorded once as a death and once as a birth, in time order
    n_ev = int(fr["total_replacement_events"])
    assert n_ev > 0 and len(fr["snap_ev_die"]) == n_ev and len(fr["snap_ev_birth"]) == n_ev
    for k in ("snap_ev_die", "snap_ev_birth"):
        ev = fr[k]
        assert ev[:, 0].min() >= 1 and ev[:, 0].max() <= n_snap - 1      # the first event follows fr_start
        assert np.all(np.diff(ev[:, 0]) >= 0)
    # the snapshot at the final step is the reported final profile, and the featured slot's
    # configuration reproduces the reaction coordinate that was recorded for it
    for d in (abf, fr):
        assert np.allclose(d["snap_pmf"][-1], d["pmf"][-1], atol=1e-5)
        assert np.allclose(d["snap_mf_bins"][-1], d["hist_mf_bins"][-1], atol=1e-5)
        assert np.allclose(d["snap_counts"][-1], d["hist_counts"][-1])
        params = wca.DimerWCAParams(n_dim=4, beta=1.0, h=1.0, w=2.0, a=1.5)
        z_feat = wca.reaction_coordinate(torch.as_tensor(d["snap_q"][:, 0], dtype=F64).reshape(-1, 16, 2), params).numpy()
        assert np.allclose(z_feat, d["snap_z"][:, 0], atol=1e-4)
        # the saved snapshots at save steps agree with the save-cadence profiles
        for j, st in enumerate(d["steps"]):
            f = int(np.where(d["snap_steps"] == st)[0][0])
            assert np.allclose(d["snap_pmf"][f], d["pmf"][j], atol=1e-5), st
