"""Validation of the bi-channel Type-C system (Phase F).

The system exists to make ONE thing true that no earlier cell in this campaign had:
an error in the free energy the ABP is estimating that the bias cannot repair at any
adaptation gain, because it lives in a coordinate the bias does not touch.  These
tests pin the physics, the exact reference, and the size of that error.
"""
import math

import numpy as np
import pytest
import torch

from abpfr.io import _check_arrays
from abpfr.systems import bichannel as bc
from abpfr.systems.gateway import Method

PI = math.pi
DT = torch.float64
DEV = "cpu"


def _cfg(**kw):
    base = dict(K=64, dt=1e-3, n_steps=2_000, block=20, n_saves=20, profile_every=4,
                joint_every=8, ess_window_steps=1000, n_strata=16)
    base.update(kw)
    return bc.BiChannelConfig(**base)


def test_gradient_matches_autograd():
    torch.manual_seed(0)
    phi = (torch.rand(200, dtype=DT) * 2 - 1) * PI
    psi = (torch.rand(200, dtype=DT) * 2 - 1) * PI
    phi.requires_grad_(True)
    psi.requires_grad_(True)
    V = bc.V_of(phi, psi, 2.0, 0.7, 1.0, 1.3).sum()
    gphi, gpsi = torch.autograd.grad(V, [phi, psi])
    aphi, apsi = bc.gradV_of(phi.detach(), psi.detach(), 2.0, 0.7, 1.0, 1.3)
    assert float((aphi - gphi).abs().max()) < 1e-12
    assert float((apsi - gpsi).abs().max()) < 1e-12


def test_equal_barriers_give_equal_channel_partition_functions():
    """Ha = Hb makes the two channels images of each other under phi -> phi + pi/2,
    so the channel ratio is exactly e^{-beta Delta} and the target population is
    known analytically rather than fitted."""
    cfg = _cfg(beta=4.0, Hperp=2.0, Delta=0.5, Ha=1.0, Hb=1.0)
    P1, P2 = bc.GRID2.mesh(DEV, DT)
    rho = torch.exp(-cfg.beta * bc.V_of(P1, P2, cfg.Hperp, cfg.Delta, cfg.Ha, cfg.Hb))
    wB = bc.channel_weight_B(P2).to(DT)
    ZA, ZB = float((rho * (1 - wB)).sum()), float((rho * wB).sum())
    assert abs(ZB / ZA - math.exp(-cfg.beta * cfg.Delta)) < 0.15 * math.exp(
        -cfg.beta * cfg.Delta)
    ref = bc.reference_objects(cfg.beta, cfg.Hperp, cfg.Delta, cfg.Ha, cfg.Hb,
                               DEV, DT)
    assert abs(ref["p_B_ref"] - ZB / (ZA + ZB)) < 1e-9


def test_symmetric_cell_has_half_its_mass_in_the_hidden_channel():
    cfg = _cfg(beta=4.0, Hperp=1.5, Delta=0.0)
    ref = bc.reference_objects(cfg.beta, cfg.Hperp, cfg.Delta, cfg.Ha, cfg.Hb,
                               DEV, DT)
    assert abs(ref["p_B_ref"] - 0.5) < 1e-12
    # per-phi the channels are NOT balanced -- that is the point: at channel A's
    # well the hidden channel holds ~2% of the fiber, and the two profiles are exact
    # images of each other under the phi -> phi + pi/2 symmetry that makes Z_A = Z_B.
    pB = ref["pB_phi_ref"]
    assert float(pB.min()) < 0.05 and float(pB.max()) > 0.95
    assert float((torch.roll(pB, bc.NG // 4) - (1.0 - pB)).abs().max()) < 1e-12


@pytest.mark.parametrize("Delta,expect_big", [(0.0, True), (0.5, True), (4.0, False)])
def test_type_c_amplitude_stands_far_above_the_mollifier_floor(Delta, expect_big):
    """The pre-run design gate: the error from never populating channel B must be
    large compared with the estimator's own floor, or the cell cannot show Type C."""
    cfg = _cfg(beta=4.0, Hperp=1.5, Delta=Delta)
    amp = bc.type_c_amplitude(cfg, DEV, DT)
    e_star = bc.analytic_floors(cfg, DEV, DT)["e_star"]
    assert (amp > 30 * e_star) == expect_big, (amp, e_star)


def test_conditional_metric_floors_resolve_the_deficit():
    cfg = _cfg(beta=4.0, Hperp=1.5, Delta=0.0)
    fl = bc.conditional_floors(cfg, 1024, n_rep=16, device=DEV, dtype=DT)
    # E_chan and the channel-population error resolve a fully missing channel;
    # E_cond does NOT on this system, which is why it is a secondary readout.
    assert np.quantile(fl["e_chan"], 0.95) < 0.1
    assert np.quantile(fl["p_B_err"], 0.95) < 0.05


def test_engine_smoke_records_validate():
    cfg = _cfg(beta=4.0, Hperp=1.0, Delta=0.0)
    recs = bc.simulate_batch([cfg, cfg], [0, 1], [Method("shus"),
                                                 Method("shus_g2", g_shus=2.0)],
                             batch_seed=1, device=torch.device(DEV), dtype=DT)
    assert len(recs) == 4
    for r in recs:
        _check_arrays(r, "bichannel record")
        for k in ("l2_f_t", "e_cond_t", "e_chan_t", "P_regions", "pB_phi_t"):
            assert np.all(np.isfinite(r[k])), k
        assert abs(r["P_regions"].sum(axis=1) - 1.0).max() < 1e-12
        assert r["pmf_t"].shape[1] == bc.NG
        assert r["joint_t"].shape[1:] == (bc.NG, bc.NG)


def test_a_backed_off_conditional_arm_is_bitwise_the_plain_run():
    """Estimator protection, end to end: an event that cannot meet its ESS floor
    backs theta off to zero and must leave the trajectory bit-identical to plain
    SHUS -- nothing in the conditional path may touch accumulator state or the
    physics noise stream."""
    cfg = _cfg(beta=4.0, Hperp=1.0, Delta=0.0)
    arms = [Method("shus"),
            Method("cond_noop", use_fr=True, cond_fr=True, theta=0.01,
                   t_on_frac=0.0, t_off_frac=1.0, fr_every_blocks=2,
                   alpha_ess=1.0)]
    recs = bc.simulate_batch([cfg], [0], arms, batch_seed=2,
                             device=torch.device(DEV), dtype=DT)
    a, b = recs[0], recs[1]
    assert np.array_equal(a["l2_f_t"], b["l2_f_t"])
    assert np.array_equal(a["pmf_t"], b["pmf_t"])
    assert b["total_turnover"] == 0.0                     # no walker was replaced
    assert float(np.max(np.abs(b["event_theta"]))) < 1e-6  # theta backed off ~5 decades


def test_marginal_and_conditional_arms_coexist_in_one_batch():
    cfg = _cfg(beta=4.0, Hperp=1.0, Delta=0.0)
    arms = [
        Method("shus"),
        Method("fr_marg", use_fr=True, theta=0.05, t_on_frac=0.1, t_off_frac=0.9,
               fr_every_blocks=2),
        Method("fr_cond", use_fr=True, cond_fr=True, theta=0.05, t_on_frac=0.1,
               t_off_frac=0.9, fr_every_blocks=2),
        Method("cnt_cond", use_fr=True, cond_fr=True, cond_bins1=8, cond_bins2=8,
               theta=0.05, t_on_frac=0.1, t_off_frac=0.9, fr_every_blocks=2),
        Method("sham_cond", sham=True, shadows="fr_cond", theta=0.05,
               t_on_frac=0.1, t_off_frac=0.9, fr_every_blocks=2),
    ]
    recs = bc.simulate_batch([cfg], [0], arms, batch_seed=3,
                             device=torch.device(DEV), dtype=DT)
    by = {r["method"]["name"]: r for r in recs}
    assert by["shus"]["total_turnover"] == 0.0
    for n in ("fr_marg", "fr_cond", "cnt_cond", "sham_cond"):
        assert by[n]["total_turnover"] > 0.0, n
        assert np.all(np.isfinite(by[n]["l2_f_t"])), n
    # the sham is intensity-matched to the conditional arm it shadows
    assert abs(by["sham_cond"]["total_turnover"]
               - by["fr_cond"]["total_turnover"]) < 1e-9
