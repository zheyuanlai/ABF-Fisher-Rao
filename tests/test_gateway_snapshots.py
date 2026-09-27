"""The gateway movie record (``store_snapshots``) is bit-inert, and its ids and events are consistent."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import gateway_core as gw  # noqa: E402


def _run(store_snapshots, seed=7):
    cfg = gw.GatewayConfig(beta=16.0, H=0.5, r=32.0, s=0.1, N=256, dt=4e-4, n_steps=1500,
                           save_every=500, init="left", h=0.07, gamma=15.0)
    spec = gw.BatchSpec(configs=[cfg, cfg], seeds=[seed, seed + 1],
                        methods=[gw.ABF, gw.FR_UNIFORM], batch_seed=5)
    return gw.simulate_batch(spec, device="cpu", dtype=torch.float64, store_profiles=True,
                             store_snapshots=store_snapshots)


def test_recording_snapshots_is_bit_inert():
    a, b = _run(0), _run(50)
    for r in range(4):
        for k in ("F_hat", "Fp_hat", "l2_f_t", "l2_fp_t", "F_prof_t", "Fp_prof_t", "ess_t", "n_die", "n_clone"):
            assert np.array_equal(a[r][k], b[r][k]), (r, k)
    assert "snap_X" not in a[0] and "snap_X" in b[0]


def test_snapshot_ids_and_events_are_consistent():
    recs = _run(50)
    abf, fr = recs[0], recs[1]
    n_snap = len(abf["snap_t"])
    assert abf["snap_X"].shape == (n_snap, 256) and abf["snap_id"].shape == (n_snap, 256)
    # an ABF row never resamples: ids are the identity at every snapshot, no events
    assert np.array_equal(abf["snap_id"], np.tile(np.arange(256), (n_snap, 1)))
    assert abf["snap_ev_die"].shape == (0, 3) and abf["snap_ev_clone"].shape == (0, 3)
    # FR row: ids are unique within every snapshot (a copy always gets a fresh id) ...
    for f in range(n_snap):
        assert len(np.unique(fr["snap_id"][f])) == 256, f
    # ... ids never come back once gone, and the population is always N
    seen_gone = set()
    prev = set(fr["snap_id"][0].tolist())
    for f in range(1, n_snap):
        cur = set(fr["snap_id"][f].tolist())
        assert not (cur & seen_gone), f
        seen_gone |= prev - cur
        prev = cur
    # the engine's totals equal the recorded events, and events sit inside the domain
    assert fr["n_die"] == len(fr["snap_ev_die"]) and fr["n_clone"] == len(fr["snap_ev_clone"])
    assert fr["n_die"] + fr["n_clone"] > 0
    for k in ("snap_ev_die", "snap_ev_clone"):
        ev = fr[k]
        assert ev[:, 0].min() >= 0 and ev[:, 0].max() <= n_snap - 1
        assert np.all(np.diff(ev[:, 0]) >= 0)            # recorded in time order
        assert np.all(np.abs(ev[:, 1]) <= gw.XMAX)
    # the snapshot at the final step is the final state, and the profile matches the engine's
    assert np.allclose(fr["snap_F"][-1], fr["F_hat"], atol=1e-5)
    assert np.allclose(fr["snap_Fp"][-1], fr["Fp_hat"], atol=1e-4)
    # raw accumulators: the last snapshot holds every deposit, and reproduces the engine's read-out
    assert np.allclose(fr["snap_C"][-1].sum(), 256 * 1500)
    x = fr["x_grid"]; dx = float(x[1] - x[0])
    from analyze_gateway_bandwidth_audit import mean_force_at
    own = mean_force_at(fr["snap_Sf"][-1], fr["snap_C"][-1], 0.07, dx, 1.0)
    assert np.abs(own - fr["Fp_hat"]).max() < 1e-12
