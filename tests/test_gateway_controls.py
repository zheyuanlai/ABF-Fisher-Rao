"""Tests for the gateway engine's controls: matched shams and frozen-bias validation.

These pin the two properties the confirmatory experiment's conclusions rest on. If either
silently broke, the comparison would still run and still print a verdict.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import gateway_core as gw

CPU = torch.device("cpu")


def _cfg(**kw):
    base = dict(beta=16.0, H=0.5, s=0.10, r=32.0, N=256, n_steps=2500, save_every=250,
                dt=4e-4, gamma=1.5)
    base.update(kw)
    return gw.GatewayConfig(**base)


def _run(methods, cfg=None, seed=0):
    cfg = cfg or _cfg()
    recs = gw.simulate_batch(gw.BatchSpec(configs=[cfg], seeds=[seed], methods=methods),
                             device=CPU)
    return {r["method"]: r for r in recs}


# --------------------------------------------------------------- matched shams
def test_each_sham_matches_its_own_partner_exactly():
    """Timing and intensity must be matched by construction, not approximately."""
    out = _run([gw.ABF, gw.FR_ORACLE, gw.FR_ESTIMATED, gw.SHAM_ORACLE, gw.SHAM_PRACTICAL])
    for sham, partner in (("sham_oracle", "fr_oracle"),
                          ("sham_practical", "fr_estimated")):
        assert out[sham]["n_die"] == out[partner]["n_die"], sham
        assert out[sham]["n_clone"] == out[partner]["n_clone"], sham
        assert out[sham]["n_fr_apply"] == out[partner]["n_fr_apply"], sham


def test_the_two_shams_are_not_interchangeable():
    """The reason two shams are needed rather than one.

    The oracle and practical arms build different targets, so they fire different numbers of
    events. A single sham shadowing the oracle is therefore NOT an intensity-matched control
    for the practical arm, and using it as one would compare the practical method against a
    control that resamples a different amount.
    """
    out = _run([gw.ABF, gw.FR_ORACLE, gw.FR_ESTIMATED, gw.SHAM_ORACLE, gw.SHAM_PRACTICAL])
    assert (out["fr_oracle"]["n_die"], out["fr_oracle"]["n_clone"]) != \
           (out["fr_estimated"]["n_die"], out["fr_estimated"]["n_clone"])


def test_sham_actually_resamples_but_does_not_steer():
    """A sham that fired nothing would trivially equal ABF and prove nothing."""
    out = _run([gw.ABF, gw.FR_ORACLE, gw.SHAM_ORACLE])
    assert out["sham_oracle"]["n_die"] > 0 and out["sham_oracle"]["n_clone"] > 0
    assert out["abf"]["n_die"] == 0
    # and it must genuinely disturb the ensemble: lineage diversity has to fall
    assert out["sham_oracle"]["final_ess"] < out["abf"]["final_ess"]


def test_sham_carries_no_target_and_cannot_leak_a_reference():
    for m in (gw.SHAM_ORACLE, gw.SHAM_PRACTICAL):
        assert m.target_mode == "none"
    gw.assert_no_oracle_leakage([gw.ABF, gw.FR_ORACLE, gw.FR_ESTIMATED,
                                 gw.SHAM_ORACLE, gw.SHAM_PRACTICAL])


def test_sham_without_its_partner_is_refused():
    """Falling back to its own counts would be a different, weaker control."""
    with pytest.raises(AssertionError, match="matched intensity is unobtainable"):
        gw.assert_no_oracle_leakage([gw.ABF, gw.SHAM_PRACTICAL])


# ---------------------------------------------------------- frozen-bias validation
def test_frozen_bias_recovers_the_reference_from_a_good_bias():
    """Sanity: hand it the EXACT mean force and the reconstruction must be near-perfect.

    This is the calibration the endpoint needs. Without it, a large frozen-bias error could
    equally mean 'the bias is bad' or 'the reconstruction is broken'.
    """
    cfg = _cfg(N=4096)
    x_grid, dx, eval_mask, _ = gw.build_grid(CPU, gw.DTYPE)
    b = torch.tensor([[cfg.beta]], dtype=gw.DTYPE)
    _, Fp_ref = gw.eb.reference_profiles(
        x_grid, eval_mask, b, torch.tensor([[cfg.H]], dtype=gw.DTYPE),
        torch.tensor([[cfg.omega_out]], dtype=gw.DTYPE),
        torch.tensor([[cfg.omega_in]], dtype=gw.DTYPE),
        torch.tensor([[cfg.s]], dtype=gw.DTYPE))
    out = gw.run_frozen_bias(Fp_ref, [cfg], n_steps=20_000, device=CPU)
    assert out["l2_f_kT"][0] < 0.5, out["l2_f_kT"]


def test_frozen_bias_penalises_a_bias_that_is_simply_wrong():
    cfg = _cfg(N=4096)
    x_grid, dx, eval_mask, _ = gw.build_grid(CPU, gw.DTYPE)
    zero = torch.zeros((1, gw.N_GRID), dtype=gw.DTYPE)      # no bias at all
    out = gw.run_frozen_bias(zero, [cfg], n_steps=20_000, device=CPU)
    assert out["l2_f_kT"][0] > 1.0, out["l2_f_kT"]


def test_frozen_bias_rows_in_a_group_share_their_realisation():
    """Arms of one seed must be scored on the same fresh population AND the same noise.

    Two things are being pinned. First, the frozen-bias walkers must not inherit each arm's
    final ensemble -- an arm that happened to end with more walkers in the right basin would
    otherwise score better for a reason unrelated to the bias it learned. Second, the arms
    must be *paired*: with independent noise per row, the arm-to-arm difference picks up a
    sampling variance that has nothing to do with the bias, which is exactly the pairing the
    rest of the study is careful to preserve.
    """
    cfg = _cfg(N=512)
    Fp = torch.zeros((3, gw.N_GRID), dtype=gw.DTYPE)     # identical bias in all three rows
    grouped = gw.run_frozen_bias(Fp, [cfg] * 3, group=[0, 0, 0], n_steps=1500, device=CPU)
    np.testing.assert_allclose(grouped["l2_f_kT"][0], grouped["l2_f_kT"][1], rtol=0, atol=0)
    np.testing.assert_allclose(grouped["l2_f_kT"][0], grouped["l2_f_kT"][2], rtol=0, atol=0)
    assert grouped["n_groups"] == 1

    # ungrouped rows are independent replicas, so they must NOT coincide -- otherwise the
    # group argument would be doing nothing and the test above would be vacuous
    indep = gw.run_frozen_bias(Fp, [cfg] * 3, n_steps=1500, device=CPU)
    assert indep["l2_f_kT"][0] != indep["l2_f_kT"][1]
    assert indep["n_groups"] == 3


def test_frozen_bias_does_not_adapt():
    """Doubling the run must not improve a bad bias -- nothing is being learned."""
    cfg = _cfg(N=2048)
    zero = torch.zeros((1, gw.N_GRID), dtype=gw.DTYPE)
    a = gw.run_frozen_bias(zero, [cfg], n_steps=8_000, seed=1, device=CPU)["l2_f_kT"][0]
    b = gw.run_frozen_bias(zero, [cfg], n_steps=16_000, seed=1, device=CPU)["l2_f_kT"][0]
    assert abs(a - b) / a < 0.5, (a, b)
