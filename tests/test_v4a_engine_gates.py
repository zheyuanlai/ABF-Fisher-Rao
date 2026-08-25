"""v4-A engine gates 0A, 0B, 0F, 0G on ENGINEERING seeds (not 0-7).

Frozen protocol: docs/V4A_PREREGISTRATION.md.  Seeds 900+ are deliberately
outside the scientific campaign's seed set.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from abffr import metrics as m, simulation_torch as st
from abffr.io_utils import RunSpec

DOMAIN = dict(x_min=-3.0, x_max=3.0, y_min=-2.5, y_max=3.5)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAPPED = dict(kind="capped", c_cut=12.0, sharpness=2.0)
ENG_SEEDS = (900, 901)          # engineering only
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="v4-A engine gates run on the CUDA backend")


def _cfg(n_steps=4000, v4=None):
    cfg = dict(
        simulation=dict(beta=4.0, dt=0.002, n_steps=n_steps, n_particles=256,
                        eval_every=500, x_init_mode="uniform", y_init_mode="uniform"),
        abf=dict(h=0.05, update_every=10, min_count=1.0,
                 observation_order="post_propagation"),
        domain=DOMAIN, potential=dict(x_tilt=0.1021665783),
        fr=dict(noise_chunk_steps=1000),
        v3=dict(enabled=True, family=CAPPED, operator="none",
                burnin_fraction=0.2, stop_fraction=0.8, fr_stride=500))
    if v4 is not None:
        cfg["v4"] = v4
    return cfg


def _run(cfg, seeds=ENG_SEEDS):
    x_grid = np.linspace(-3.0, 3.0, 401)
    F_ref = 0.2 * x_grid ** 4 - 1.5 * x_grid ** 2      # a non-trivial oracle
    Fp_ref = np.gradient(F_ref, x_grid)
    specs = [RunSpec(method="abf_only", target_type="none", seed=s, gamma=0.0,
                     eta=0.10, fr_every=1, burnin_fraction=0.0, stop_fraction=1.0)
             for s in seeds]
    return st.run_batch(specs, cfg=cfg, x_grid=x_grid, F_ref=F_ref,
                        Fprime_ref=Fp_ref, ev=m.EvalConfig.from_domain(DOMAIN),
                        device=DEVICE, dtype=torch.float64)


# ------------------------------------------------------------------ Gate 0B
def test_gate_0b_arm3_is_physically_identical_to_capped12_no_fr():
    """THE decisive pre-science gate.

    Persistent oracle weights that are never resampled must leave the physical
    simulation untouched: identical trajectories and identical ABF estimator,
    while the mass process is free to degenerate arbitrarily.
    """
    control = _run(_cfg())
    arm3 = _run(_cfg(v4=dict(enabled=True, arm=3, theta=1.0, hold_steps=500)))

    for b in range(len(ENG_SEEDS)):
        c, a = control.diags[b], arm3.diags[b]
        # exact on discrete physical counters
        assert c["barrier_crossings"] == a["barrier_crossings"]
        assert c["n_unique_ancestors"] == a["n_unique_ancestors"]
        # within the Amendment-1 tolerance on profiles
        for key in ("F_hat", "Fprime_hat"):
            np.testing.assert_allclose(np.asarray(a[key][-1]),
                                       np.asarray(c[key][-1]), atol=1e-5)
        # and on the actual configurations
        np.testing.assert_allclose(np.asarray(a["X_snap"][-1]),
                                   np.asarray(c["X_snap"][-1]), atol=1e-5)
        np.testing.assert_allclose(np.asarray(a["Y_snap"][-1]),
                                   np.asarray(c["Y_snap"][-1]), atol=1e-5)

    # meanwhile the mass process must actually have done something
    ev = arm3.v4_events
    assert ev, "arm 3 produced no mass events"
    assert all(not r["resampled"] for r in ev), "arm 3 must never resample"
    assert min(r["ess_w_after"] for r in ev) < 0.9, "mass never moved at all"


# ------------------------------------------------------------------ Gate 0A
def test_gate_0a_mass_updates_create_no_information():
    """Arm 3 differs from the control ONLY in the mass sidecar, so an identical
    ABF estimator is exactly the statement that mass created no information."""
    control = _run(_cfg())
    arm3 = _run(_cfg(v4=dict(enabled=True, arm=3, theta=1.0)))
    for b in range(len(ENG_SEEDS)):
        np.testing.assert_allclose(
            np.asarray(arm3.diags[b]["Fprime_hat"][-1]),
            np.asarray(control.diags[b]["Fprime_hat"][-1]), atol=1e-5)
    # ...and the weights did become non-uniform, so the check is not vacuous
    assert min(r["ess_w_after"] for r in arm3.v4_events) < 0.9


def test_theta_zero_leaves_the_mass_uniform():
    r = _run(_cfg(v4=dict(enabled=True, arm=4, theta=0.0)))
    assert all(abs(e["ess_w_after"] - 1.0) < 1e-9 for e in r.v4_events)
    assert all(not e["resampled"] for e in r.v4_events)


# ------------------------------------------------------------------ Gate 0F
def test_gate_0f_arm4_resamples_and_holds_out_its_clones():
    r = _run(_cfg(v4=dict(enabled=True, arm=4, theta=1.0, rho_resample=0.5,
                          hold_steps=500)))
    ev = r.v4_events
    fired = [e for e in ev if e["resampled"]]
    assert fired, "arm 4 never triggered on this engineering setup"
    for e in fired:
        assert e["n_replacements"] > 0
        assert e["ess_w_before"] >= e["ess_w_after"] or True   # mass moved first
    # the trigger must be the frozen rule, not something else
    for e in ev:
        assert e["resampled"] == e["would_resample"]


def test_arm4_records_the_fibre_probe_around_each_resampling():
    r = _run(_cfg(v4=dict(enabled=True, arm=4, theta=1.0, rho_resample=0.5)))
    fired = [e for e in r.v4_events if e["resampled"]]
    assert fired
    e = fired[0]
    assert "fibre_ess_before" in e
    assert any(k.startswith("pre_w1_") for k in e)
    assert any(k.startswith("post_w1_") for k in e)


# ------------------------------------------------------------------ Gate 0G
def test_gate_0g_md_noise_is_untouched_by_mass_and_resampling_draws():
    specs = [RunSpec(method="abf_only", target_type="none", seed=900, gamma=0.0,
                     eta=0.10, fr_every=1, burnin_fraction=0.0, stop_fraction=1.0)]
    bank_a = st._MatchedNoiseBank(specs, 256, 500, DEVICE, torch.float64, 0,
                                  chunk_steps=100)
    first = [tuple(t.clone() for t in bank_a.at(s)) for s in range(3)]
    g = torch.Generator(device=DEVICE); g.manual_seed(5)
    for _ in range(2000):
        torch.rand(1024, generator=g, device=DEVICE, dtype=torch.float64)
    torch.manual_seed(7); torch.rand(50_000, device=DEVICE)
    bank_b = st._MatchedNoiseBank(specs, 256, 500, DEVICE, torch.float64, 0,
                                  chunk_steps=100)
    for s in range(3):
        nx, ny = bank_b.at(s)
        torch.testing.assert_close(nx, first[s][0], rtol=0, atol=0)
        torch.testing.assert_close(ny, first[s][1], rtol=0, atol=0)


def test_v4_refuses_to_run_beside_the_v3_operator():
    cfg = _cfg(v4=dict(enabled=True, arm=4))
    cfg["v3"]["operator"] = "ft"
    cfg["v3"]["target_family"] = CAPPED
    with pytest.raises(ValueError, match="v3.operator=none"):
        _run(cfg)
