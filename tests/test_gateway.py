"""Integration tests for the gateway SHUS(+FR) engine: analytic reference, pairing,
determinism, checkpoint/resume, sham matching, and a short end-to-end convergence."""
import numpy as np
import pytest
import torch

from conftest import DEVICE, DTYPE
from abpfr.systems import gateway as gw


def small_cfg(**kw):
    # eps_bw pinned: these tests validate mechanics, not the calibrated bandwidth
    base = dict(beta=1.0, H=1.0, omega_out=1.0, r=2.0, s=0.3, K=256, dt=1e-3,
                n_steps=20_000, block=20, n_saves=50, ess_window_steps=2000,
                eps_bw=0.07)
    base.update(kw)
    return gw.GatewayConfig(**base)


def test_mollified_fixed_point_limits():
    cfg = small_cfg()
    fp_fine = gw.mollified_fixed_point(cfg, eps_bw=0.005)
    fp_coarse = gw.mollified_fixed_point(cfg, eps_bw=0.10)
    # bias floor vanishes with the bandwidth and grows with it
    assert fp_fine["e_star"] < 0.005 < fp_coarse["e_star"]
    assert fp_fine["kl_star"] < 1e-3 < fp_coarse["kl_star"]
    # the coarse fixed point is a real profile: centered, finite
    assert np.isfinite(fp_coarse["F_star"].numpy()).all()


def test_reference_matches_y_quadrature():
    cfg = small_cfg()
    xg = gw.GRID.x(DEVICE, DTYPE)
    mask = gw.GRID.eval_mask(DEVICE, DTYPE)
    beta = torch.tensor([[cfg.beta]], dtype=DTYPE)
    F_ref, Fp_ref = gw.reference_profiles(
        xg, mask, beta, torch.tensor([[cfg.H]], dtype=DTYPE),
        torch.tensor([[cfg.omega_out]], dtype=DTYPE),
        torch.tensor([[cfg.omega_in]], dtype=DTYPE),
        torch.tensor([[cfg.s]], dtype=DTYPE))
    # brute-force marginal: F(x) = -log integral exp(-beta V(x,y)) dy / beta
    y = torch.linspace(-8, 8, 4001, dtype=DTYPE)
    om = gw.omega_of(xg.unsqueeze(1), cfg.omega_out, cfg.omega_in, cfg.s)
    V = gw.U_of(xg.unsqueeze(1), cfg.H) + 0.5 * om ** 2 * y.unsqueeze(0) ** 2
    Fq = -torch.log(torch.exp(-cfg.beta * V).sum(dim=1) * (y[1] - y[0])) / cfg.beta
    Fq = Fq - Fq[mask].mean()
    assert float((F_ref[0] - Fq)[mask].abs().max()) < 1e-6
    # F'_ref consistent with a numerical derivative of F_ref
    num = np.gradient(F_ref[0].numpy(), gw.GRID.dx)
    assert np.abs(Fp_ref[0].numpy()[10:-10] - num[10:-10]).max() < 5e-3


def test_transverse_stability_assertion():
    with pytest.raises(AssertionError):
        gw.simulate_batch([small_cfg(r=100.0)], [0], [gw.SHUS], batch_seed=1,
                          device=DEVICE, dtype=DTYPE)


def test_paired_noise_identical_arms_identical_trajectories():
    cfg = small_cfg(n_steps=2000, n_saves=10)
    m1, m2 = gw.Method("shus_a"), gw.Method("shus_b")
    recs = gw.simulate_batch([cfg], [0], [m1, m2], batch_seed=7,
                             device=DEVICE, dtype=DTYPE)
    assert np.array_equal(recs[0]["pmf_t"], recs[1]["pmf_t"])
    assert np.array_equal(recs[0]["l2_f_t"], recs[1]["l2_f_t"])


def test_deterministic_given_seed():
    cfg = small_cfg(n_steps=2000, n_saves=10)
    fr = gw.Method("shus_fr", use_fr=True, theta=0.1, t_on_frac=0.1, t_off_frac=0.9,
                   fr_every_blocks=5)
    a = gw.simulate_batch([cfg], [3], [gw.SHUS, fr], batch_seed=42,
                          device=DEVICE, dtype=DTYPE)
    b = gw.simulate_batch([cfg], [3], [gw.SHUS, fr], batch_seed=42,
                          device=DEVICE, dtype=DTYPE)
    for r in range(2):
        assert np.array_equal(a[r]["pmf_t"], b[r]["pmf_t"])
        assert np.array_equal(a[r]["event_turnover"], b[r]["event_turnover"])


def test_checkpoint_resume_bitwise_equal():
    cfg = small_cfg(n_steps=4000, n_saves=20)
    fr = gw.Method("shus_fr", use_fr=True, theta=0.15, t_on_frac=0.0, t_off_frac=1.0,
                   fr_every_blocks=10)
    full = gw.simulate_batch([cfg], [5], [gw.SHUS, fr], batch_seed=9,
                             device=DEVICE, dtype=DTYPE)
    state = gw.simulate_batch([cfg], [5], [gw.SHUS, fr], batch_seed=9,
                              device=DEVICE, dtype=DTYPE, stop_at=2000)
    resumed = gw.simulate_batch([cfg], [5], [gw.SHUS, fr], batch_seed=9,
                                device=DEVICE, dtype=DTYPE, start_state=state)
    for r in range(2):
        assert np.array_equal(full[r]["pmf_t"], resumed[r]["pmf_t"])
        assert np.array_equal(full[r]["l2_f_t"], resumed[r]["l2_f_t"])
        assert np.array_equal(full[r]["event_theta"], resumed[r]["event_theta"])
        assert np.array_equal(full[r]["dep_self_l2_t"], resumed[r]["dep_self_l2_t"],
                              equal_nan=True)
        assert np.array_equal(full[r]["n_anc_t"], resumed[r]["n_anc_t"])


def test_sham_copies_partner_turnover_and_timing():
    cfg = small_cfg(n_steps=4000, n_saves=10)
    fr = gw.Method("shus_fr", use_fr=True, theta=0.2, t_on_frac=0.2, t_off_frac=0.8,
                   fr_every_blocks=5)
    sham = gw.Method("sham", use_fr=True, sham=True, shadows="shus_fr")
    recs = gw.simulate_batch([cfg], [1], [gw.SHUS, fr, sham], batch_seed=11,
                             device=DEVICE, dtype=DTYPE)
    r_fr = next(r for r in recs if r["method"]["name"] == "shus_fr")
    r_sh = next(r for r in recs if r["method"]["name"] == "sham")
    r_base = next(r for r in recs if r["method"]["name"] == "shus")
    assert np.array_equal(r_sh["event_turnover"], r_fr["event_turnover"])
    assert np.all(r_base["event_turnover"] == 0)
    # events confined to the window
    T = cfg.T_total
    active = r_fr["event_turnover"] > 0
    if active.any():
        tt = r_fr["event_time"][active]
        assert tt.min() >= 0.2 * T - 1e-9 and tt.max() <= 0.8 * T + 1e-9


def test_sham_requires_partner_in_batch():
    cfg = small_cfg(n_steps=200, n_saves=5)
    sham = gw.Method("sham", use_fr=True, sham=True, shadows="missing")
    with pytest.raises(AssertionError):
        gw.simulate_batch([cfg], [0], [gw.SHUS, sham], batch_seed=1,
                          device=DEVICE, dtype=DTYPE)


def test_one_right_init_places_discoverer():
    cfg = small_cfg(init="one_right", n_steps=20, block=20, n_saves=2)
    recs = gw.simulate_batch([cfg], [0], [gw.SHUS], batch_seed=2,
                             device=DEVICE, dtype=DTYPE)
    # right-basin occupancy nonzero from the very first save
    assert recs[0]["P_regions"][0, 2] > 0


def test_end_to_end_plain_shus_converges():
    """The engine's reason to exist: F_t must move decisively toward F_ref."""
    cfg = small_cfg()
    recs = gw.simulate_batch([cfg], [0], [gw.SHUS], batch_seed=123,
                             device=DEVICE, dtype=DTYPE)
    l2 = recs[0]["l2_f_t"]
    assert l2[-1] < 0.4 * l2[0], f"no convergence: e0={l2[0]:.3f}, eT={l2[-1]:.3f}"
    # and the biased marginal must flatten
    kl = recs[0]["kl_u_t"]
    assert kl[-1] < 0.3 * kl[0]


def test_global_vs_windowed_ancestry():
    cfg = small_cfg(n_steps=4000, n_saves=20, ess_window_steps=500)
    fr = gw.Method("shus_fr", use_fr=True, theta=0.3, t_on_frac=0.0, t_off_frac=1.0,
                   fr_every_blocks=5)
    recs = gw.simulate_batch([cfg], [2], [gw.SHUS, fr], batch_seed=21,
                             device=DEVICE, dtype=DTYPE)
    r_base = next(r for r in recs if r["method"]["name"] == "shus")
    r_fr = next(r for r in recs if r["method"]["name"] == "shus_fr")
    # no resampling: every original ancestor survives
    assert np.all(r_base["n_anc_t"] == cfg.K)
    # under persistent FR, global lineage loss is monotone and irreversible
    d = np.diff(r_fr["n_anc_t"])
    assert np.all(d <= 0) and r_fr["n_anc_t"][-1] < cfg.K
    # windowed ESS can recover; global ESS never exceeds the windowed one
    assert np.all(r_fr["ess_anc_glob_t"] <= r_fr["ess_anc_t"] + 1e-9)


def test_record_schema_roundtrip(tmp_path):
    from abpfr.io import load_run, save_run
    cfg = small_cfg(n_steps=400, n_saves=5)
    rec = gw.simulate_batch([cfg], [0], [gw.SHUS], batch_seed=4,
                            device=DEVICE, dtype=DTYPE)[0]
    arrays = {k: rec[k] for k in ("time", "pmf_t", "marginal_t", "x_grid", "F_ref",
                                  "l2_f_t", "kl_u_t")}
    meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
            "config": rec["config"], "method": rec["method"], "seed": rec["seed"]}
    p = str(tmp_path / "run0")
    save_run(p, arrays, meta)
    arr2, meta2 = load_run(p)
    assert np.array_equal(arr2["pmf_t"], rec["pmf_t"])
    assert meta2["reference_id"] == gw.REFERENCE_ID
    # schema is enforced
    with pytest.raises(AssertionError):
        save_run(str(tmp_path / "bad"), {"time": rec["time"]}, meta)
