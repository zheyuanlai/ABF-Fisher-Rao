"""The gateway raw-accumulator record is inert, and the offline read-out reproduces the engine exactly."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import gateway_core as gw                                                    # noqa: E402
from analyze_gateway_bandwidth_audit import mean_force_at, e_f               # noqa: E402
from eb_abffr_core import EVAL_LO, EVAL_HI                                   # noqa: E402


def _run(store_accumulators, h=0.07, seed=7):
    cfg = gw.GatewayConfig(beta=16.0, H=0.5, r=32.0, s=0.1, N=256, dt=4e-4, n_steps=1500,
                           save_every=500, init="left", h=h)
    spec = gw.BatchSpec(configs=[cfg, cfg], seeds=[seed, seed + 1], methods=[gw.ABF], batch_seed=5)
    return gw.simulate_batch(spec, device="cpu", dtype=torch.float64, store_profiles=True,
                             store_accumulators=store_accumulators)


def test_recording_accumulators_is_bit_inert():
    a, b = _run(False), _run(True)
    for k in ("F_hat", "Fp_hat", "l2_f_t", "l2_fp_t", "F_prof_t", "Fp_prof_t"):
        assert np.array_equal(a[0][k], b[0][k]), k
    assert "Sf_t" not in a[0] and "Sf_t" in b[0] and "C_t" in b[0]


def test_offline_readout_reproduces_the_engine_exactly():
    recs = _run(True)
    x = recs[0]["x_grid"]
    dx = float(x[1] - x[0])
    mask = (x >= EVAL_LO) & (x <= EVAL_HI)
    for r in recs:
        Sf, C = np.asarray(r["Sf_t"]), np.asarray(r["C_t"])
        own = mean_force_at(Sf, C, 0.07, dx, 1.0)
        assert np.abs(own - r["Fp_prof_t"]).max() < 1e-12
        e = e_f(own, np.asarray(r["F_ref"]), dx, mask)
        assert np.abs(e - r["l2_f_t"]).max() < 1e-12
        # raw bins: counts are integers and the last save holds every sample
        assert np.allclose(C[-1].sum(), 256 * 1500)
        assert np.array_equal(C, np.round(C))


def test_a_sharper_offline_readout_is_a_different_profile():
    recs = _run(True)
    x = recs[0]["x_grid"]
    dx = float(x[1] - x[0])
    Sf, C = np.asarray(recs[0]["Sf_t"]), np.asarray(recs[0]["C_t"])
    a, b = mean_force_at(Sf, C, 0.07, dx, 1.0), mean_force_at(Sf, C, 0.035, dx, 1.0)
    assert not np.allclose(a, b)
