"""Stage-0 engineering validation.  Fast: the whole file runs in ~60 s on one GPU."""
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import torch

from rcwfr.estimators import MeanForceAccumulator, gauge_l2
from rcwfr.fisher_rao import (kde_marginal, kl_to_uniform, selection_indices,
                              log_ratio_counts, log_ratio_kde)
from rcwfr.grid import (DEVICE, DTYPE, Grid1D, central_diff, gaussian_kernel,
                        smooth, trapz, wrap_into, reflect_into)
from rcwfr.registry import build
from rcwfr.resampling import systematic_resample, turnover_counts
from rcwfr.engines import RunConfig, run_wfr, run_abf, run_reti, run_shus, run_unbiased
from rcwfr.rowspec import expand_grid, row_column
from rcwfr.systems.base import SepSystem, SysParams
from rcwfr.wasserstein import w_step_sde


def gen(seed=0):
    g = torch.Generator(device=DEVICE); g.manual_seed(seed); return g


# --------------------------------------------------------------- grid ------
def test_binned_density_is_normalized():
    g = Grid1D(-2.0, 2.0, 401, -2.0, 2.0)
    X = torch.randn((5, 4096), device=DEVICE, dtype=DTYPE, generator=gen()) * 0.4
    X = g.enforce(X)
    p = kde_marginal(X, g, 0.08)
    assert torch.allclose(trapz(p, g.dx), torch.ones(5, device=DEVICE, dtype=DTYPE),
                          atol=1e-10)


def test_periodic_smoothing_wraps():
    g = Grid1D(-1.0, 1.0, 201, -1.0, 1.0, bc="periodic")
    k, r = gaussian_kernel(0.05, g.dx, DEVICE, DTYPE)
    v = torch.zeros((1, g.n), device=DEVICE, dtype=DTYPE)
    v[0, 0] = 1.0                       # mass sitting exactly on the seam
    s = smooth(v, k, r, g.dx, "periodic")
    assert s[0, 1] > 0 and s[0, -2] > 0          # spreads both ways across the seam
    assert abs(float(s[0, 1] - s[0, -2])) < 1e-12
    assert abs(float(s[0, 0] - s[0, -1])) < 1e-12  # duplicated end node stays consistent


def test_periodic_central_diff_wraps():
    g = Grid1D(-math.pi, math.pi, 401, -math.pi, math.pi, bc="periodic")
    x = g.x().unsqueeze(0)
    f = torch.sin(x)
    d = central_diff(f, g.dx, "periodic")
    assert float((d - torch.cos(x)).abs().max()) < 1e-4


def test_boundary_maps():
    x = torch.tensor([-3.7, -1.2, 0.3, 2.9], device=DEVICE, dtype=DTYPE)
    assert float(reflect_into(x, -1.0, 1.0).abs().max()) <= 1.0 + 1e-12
    w = wrap_into(x, -1.0, 1.0)
    assert float(w.min()) >= -1.0 - 1e-12 and float(w.max()) <= 1.0 + 1e-12


# ---------------------------------------------------------- resampling -----
def test_equal_weight_resample_is_identity():
    w = torch.full((4, 128), 1 / 128, device=DEVICE, dtype=DTYPE)
    sel = systematic_resample(w, gen(1))
    ar = torch.arange(128, device=DEVICE).unsqueeze(0).expand(4, 128)
    assert torch.equal(sel, ar)
    assert int(turnover_counts(sel, 128).sum()) == 0


def test_systematic_resample_expected_counts():
    R, N = 3, 4096
    w = torch.rand((R, N), device=DEVICE, dtype=DTYPE, generator=gen(2))
    w = w / w.sum(1, keepdim=True)
    sel = systematic_resample(w, gen(3))
    cnt = torch.zeros((R, N), device=DEVICE, dtype=DTYPE)
    cnt.scatter_add_(1, sel, torch.ones_like(sel, dtype=DTYPE))
    # systematic resampling gives floor/ceil of N*w exactly
    assert float((cnt - N * w).abs().max()) < 1.0 + 1e-9


def test_theta_zero_selection_is_noop():
    g = Grid1D(-1.0, 1.0, 201, -1.0, 1.0)
    X = g.enforce(torch.randn((4, 256), device=DEVICE, dtype=DTYPE, generator=gen(4)) * .3)
    th = torch.zeros(4, device=DEVICE, dtype=DTYPE)
    sel, info = selection_indices(X, g, "fr", th, gen(5), bw=0.05)
    assert torch.equal(sel, torch.arange(256, device=DEVICE).unsqueeze(0).expand(4, 256))
    assert int(info["turnover"].sum()) == 0


def test_uniform_target_fr_equals_count_balancing_in_the_histogram_limit():
    """The prior from the ABF/ABP campaign, made explicit and testable."""
    g = Grid1D(-1.0, 1.0, 2001, -1.0, 1.0)
    X = g.enforce(torch.randn((1, 200_000), device=DEVICE, dtype=DTYPE,
                              generator=gen(6)) * 0.35)
    n_bins = 40
    lr_c = log_ratio_counts(X, g, n_bins)
    lr_k = log_ratio_kde(X, g, bw=(g.volume / n_bins) / 2.0)
    r = torch.corrcoef(torch.stack([lr_c[0], lr_k[0]]))[0, 1]
    assert float(r) > 0.99, f"FR score and count score decorrelated: r = {float(r)}"


# ------------------------------------------------------------- system ------
def test_reference_matches_analytic_harmonic():
    g = Grid1D(-1.8, 1.8, 361, -1.5, 1.5)
    p = SysParams(m_spec=7, oms_in=3.0, oms_out=1.0)
    S = SepSystem(p, g)
    x = g.x(); m = g.eval_mask()
    F = p.H * (x ** 2 - 1) ** 2 + torch.log(S.omega(x)) / p.beta \
        + p.m_spec * torch.log(S.omega_s(x)) / p.beta
    F = F - F[m].mean()
    assert float((S.F_ref[0] - F).abs().max()) < 1e-12


def test_oracle_conditional_reproduces_the_mean_force():
    S = build("EB")
    g = S.grid
    G = gen(7)
    X = torch.full((1, 400_000), -0.7, device=DEVICE, dtype=DTYPE)
    Y = S.sample_conditional(X, G)
    f = S.mean_force(X, Y)
    i = int(round((-0.7 - g.xmin) / g.dx))
    se = float(f.std() / f.numel() ** 0.5)
    assert abs(float(f.mean()) - float(S.dF_ref[0, i])) < 5 * se + 1e-3


def test_channel_system_has_an_x_dependent_correct_channel():
    S = build("CHANNEL")
    g = S.grid
    pc = S.p_channel_ref
    lo = int((-1.4 - g.xmin) / g.dx); hi = int((1.4 - g.xmin) / g.dx)
    assert float(pc[lo]) > 0.95 and float(pc[hi]) < 0.05


# -------------------------------------------------------------- engines ----
@pytest.mark.parametrize("arm", ["wfr", "abf", "shus", "unbiased", "reti"])
def test_force_budget_is_matched_across_arms(arm):
    S = build("EB")
    cfg = RunConfig(N=64, n_seed=2, n_steps=2000, save_every=500, n_cond=5,
                    bw_mf=0.02, init="grid_cold" if arm == "reti" else "point")
    fn = {"wfr": run_wfr, "abf": run_abf, "shus": run_shus,
          "unbiased": run_unbiased, "reti": run_reti}[arm]
    r = fn(S, cfg, rows=2, seed=11)
    exact = cfg.N * cfg.n_steps
    assert abs(float(r["fe"][-1]) - exact) / exact < 0.11, \
        f"{arm}: fe={float(r['fe'][-1])} vs {exact}"


def test_fr_selection_cannot_expand_support():
    g = Grid1D(-3.0, 3.0, 601, -3.0, 3.0)
    X = g.enforce(torch.randn((3, 2048), device=DEVICE, dtype=DTYPE,
                              generator=gen(8)) * 0.15)
    G = gen(9)
    th = torch.full((3,), 0.6, device=DEVICE, dtype=DTYPE)
    w0 = (X.max(1).values - X.min(1).values).clone()
    for _ in range(200):
        sel, _ = selection_indices(X, g, "fr", th, G, bw=0.05, alpha_ess=0.0)
        X = torch.gather(X, 1, sel)
    assert float((X.max(1).values - X.min(1).values - w0).max()) <= 0.0


def test_oracle_lift_wfr_reaches_the_estimator_floor():
    from rcwfr.campaign import estimator_floor, score
    S = build("EB")
    cfg = RunConfig(N=256, n_seed=4, n_steps=20_000, save_every=2000, n_cond=5,
                    bw_mf=0.02, kappa=0.5, theta=0.6, lift="oracle")
    fl = float(estimator_floor(S, cfg, [2 ** 21], rows=2)[2 ** 21].mean())
    r = run_wfr(S, cfg, rows=4, seed=12)
    e = float(np.median(score(r, S)["e_F_final"]))
    assert e < 2.5 * fl, f"oracle-lift WFR e_F={e:.5f} vs floor {fl:.5f}"


def test_identity_lift_bias_grows_with_kappa():
    from rcwfr.campaign import score
    S = build("EB")
    ks = [0.03, 2.0]
    kap = row_column(ks, 4, DEVICE, DTYPE)
    cfg = RunConfig(N=256, n_seed=4, n_steps=20_000, save_every=2000, n_cond=5,
                    bw_mf=0.02, kappa=kap, theta=0.6, lift="identity")
    r = run_wfr(S, cfg, rows=8, seed=13)
    e = score(r, S)["e_F_final"].reshape(2, 4)
    assert np.median(e[1]) > 3 * np.median(e[0])


def test_paired_initial_conditions_across_arms():
    S = build("EB")
    cfg = RunConfig(N=32, n_seed=4, n_steps=10, save_every=10, n_cond=5, bw_mf=0.02)
    from rcwfr.engines import _init_state
    g1 = gen(21); g2 = gen(21)
    X1, Y1 = _init_state(S, cfg, 8, g1)
    X2, Y2 = _init_state(S, cfg, 8, g2)
    assert torch.equal(X1, X2) and torch.equal(Y1, Y2)
    # replicate s is shared by every configuration c
    assert torch.equal(X1[0], X1[4]) and torch.equal(Y1[0], Y1[4])
