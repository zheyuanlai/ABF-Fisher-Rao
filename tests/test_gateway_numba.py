"""Equivalence of the CPU (numba) gateway port with the accepted torch engine.

V0  constants and the initial condition equal gateway_core's;
V1  ABF dynamics + P0 histogram estimator are BITWISE-equal (to float64 round-off) to
    gateway_core.simulate_batch when both are driven by the same Langevin noise stream;
V2  the FR marginal (binned_density) and the uniform score equal the torch code to round-off;
V3  the FR resampling law (events, cap, which walkers die / are copied, pool law) matches
    gateway_core.resample_indices in distribution, including the small-N cap regime.

Run on CPU:  CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_gateway_numba.py -q
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import eb_abffr_core as eb          # noqa: E402
import gateway_core as gw           # noqa: E402
import gateway_numba as gn          # noqa: E402

CPU = torch.device("cpu")
DT = torch.float64
CELL = dict(beta=16.0, H=0.5, omega_out=1.0, omega_in=32.0, s=0.1, dt=4e-4)


def test_v0_constants_and_init():
    assert (gn.XMIN, gn.XMAX, gn.N_GRID, gn.EPS) == (eb.XMIN, eb.XMAX, eb.N_GRID, eb.EPS)
    assert gn.X_BASIN == gw.X_BASIN
    x_grid, dx, _, _ = eb.build_grid(CPU, DT)
    assert abs(gn.grid_dx() - dx) < 1e-15
    k_t, r_t = eb.gaussian_kernel(0.10, dx, CPU, DT)
    k_n, r_n = gn.gaussian_kernel_np(0.10, dx)
    assert r_t == r_n and np.allclose(k_t.numpy(), k_n, rtol=1e-13, atol=0)
    t = lambda v: torch.tensor([v], dtype=DT)
    X0, Y0 = gw.init_conditions([5300, 7], 50, t(16.0).repeat(2), t(1.0).repeat(2), t(32.0).repeat(2),
                                t(0.1).repeat(2), ["left", "left"], CPU, DT)
    for b, sd in enumerate([5300, 7]):
        x, y = gn.init_left(sd, 50, 16.0, 1.0, 32.0, 0.1)
        assert np.allclose(X0[b].numpy(), x, rtol=0, atol=1e-14)
        assert np.allclose(Y0[b].numpy(), y, rtol=0, atol=1e-14)


@pytest.mark.parametrize("n_bins", [45, 180])
def test_v1_abf_bitwise_with_shared_noise(n_bins):
    N, n_steps, seed, batch_seed = 64, 3000, 11, 5
    cfg = gw.GatewayConfig(beta=16.0, H=0.5, omega_out=1.0, r=32.0, s=0.1, N=N, dt=4e-4, n_steps=n_steps,
                           save_every=1000, init="left", estimator="histogram", n_bins=n_bins, min_count=1.0)
    spec = gw.BatchSpec(configs=[cfg], seeds=[seed], methods=[gw.ABF], batch_seed=batch_seed)
    rec = gw.simulate_batch(spec, device=CPU, dtype=DT, store_accumulators=True, store_final_state=True)[0]
    # reproduce the torch Langevin stream: per step zx (1,N) then zy (1,N) from gen_n
    g = torch.Generator(device=CPU); g.manual_seed(2000 + batch_seed)
    noise = np.empty((2 * n_steps, N))
    for st in range(n_steps):
        noise[2 * st] = torch.randn((1, N), device=CPU, dtype=DT, generator=g).numpy()[0]
        noise[2 * st + 1] = torch.randn((1, N), device=CPU, dtype=DT, generator=g).numpy()[0]
    dx = gn.grid_dx()
    kern, r = gn.gaussian_kernel_np(0.10, dx)
    x0, y0 = gn.init_left(seed, N, 16.0, 1.0, 32.0, 0.1)
    save_at = np.array([n_steps], dtype=np.int64)
    out = gn.simulate(1, N, n_steps, 4e-4, 16.0, 0.5, 1.0, 32.0, 0.1,
                      np.array([0], dtype=np.int64), np.array([n_bins], dtype=np.int64), np.array([0], dtype=np.int64), n_bins, 1.0,
                      1.5, 10000.0, 10, 3.0, 0, kern, r, dx, x0, y0, save_at, 4000, noise, True)
    M, C, X = out[0][0, -1], out[1][0, -1], out[8][0]
    assert np.array_equal(C, rec["C_bins"]), "bin counts differ"
    assert np.allclose(M, rec["M_bins"], rtol=1e-9, atol=1e-9 * np.abs(rec["M_bins"]).max())
    assert np.allclose(X, rec["X_final"], rtol=0, atol=1e-9)
    assert np.allclose(out[9][0], rec["Y_final"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("N", [1, 3, 37, 2048])
def test_v2_kde_and_uniform_score(N):
    rng = np.random.default_rng(N)
    x_grid, dx, _, _ = eb.build_grid(CPU, DT)
    k_eta, r_eta = eb.gaussian_kernel(0.10, dx, CPU, DT)
    # include the domain edges and points near them (reflection images)
    X = np.concatenate([rng.uniform(eb.XMIN, eb.XMAX, max(N - 3, 0)), [eb.XMIN, eb.XMAX, -1.79]])[:N]
    Xt = torch.as_tensor(X, dtype=DT).unsqueeze(0)
    p_t = eb.binned_density(Xt, k_eta, r_eta, dx)
    kern, r = gn.gaussian_kernel_np(0.10, dx)
    p = np.zeros(eb.N_GRID); h = np.zeros(eb.N_GRID)
    gn.kde_density(X, N, kern, r, dx, p, h)
    assert np.allclose(p, p_t.numpy()[0], rtol=1e-11, atol=1e-300)
    # uniform score exactly as gateway_core.simulate_batch builds it
    qu = torch.ones((1, eb.N_GRID), dtype=DT)
    q = qu / torch.clamp(eb.trapz(qu, dx), min=eb.EPS)
    kl = eb.trapz(p_t * (torch.log(torch.clamp(p_t, min=eb.EPS)) - torch.log(torch.clamp(q, min=eb.EPS))), dx).unsqueeze(1)
    S_t = (torch.log(torch.clamp(eb.interp1d(Xt, p_t, dx), min=eb.EPS))
           - torch.log(torch.clamp(eb.interp1d(Xt, q, dx), min=eb.EPS)) - kl)
    S_t = torch.clamp(S_t, -3.0, 3.0).numpy()[0]
    S = np.empty(N)
    gn.uniform_scores(X, N, p, dx, 3.0, S)
    assert np.allclose(S, S_t, rtol=0, atol=1e-10)


def _law_torch(S, cap, g, dt_fr, reps, seed):
    R, N = reps, S.shape[0]
    St = torch.as_tensor(np.tile(S, (R, 1)), dtype=DT)
    gen = torch.Generator(device=CPU); gen.manual_seed(seed)
    sel, die, clone = gw.resample_indices(St, torch.ones(R, dtype=torch.bool), torch.zeros(R, dtype=torch.bool),
                                          torch.arange(R), g, dt_fr, torch.full((R, 1), cap, dtype=torch.long), gen)
    kd = die.sum(1).numpy(); kc = clone.sum(1).numpy()
    copies = np.stack([np.bincount(sel[i].numpy(), minlength=N) for i in range(R)])
    return kd, kc, copies


def _law_numba(S, cap, g, dt_fr, reps, seed):
    N = S.shape[0]
    gn._seed_for_tests(seed)
    sel = np.empty(N, dtype=np.int64); dc = np.empty(N, dtype=np.int64); cc = np.empty(N, dtype=np.int64)
    pool = np.empty(2 * N, dtype=np.int64)
    kd = np.zeros(reps, dtype=np.int64); kc = np.zeros(reps, dtype=np.int64); copies = np.zeros((reps, N))
    for i in range(reps):
        a, b = gn.fr_resample(S, N, g, dt_fr, cap, sel, dc, cc, pool)
        kd[i], kc[i] = a, b
        copies[i] = np.bincount(sel, minlength=N) if a + b > 0 else 1.0
    return kd, kc, copies


@pytest.mark.parametrize("N,cap", [(8, 0), (8, 1), (16, 1), (32, 2), (64, 5)])
def test_v3_resampling_law_matches_torch(N, cap):
    rng = np.random.default_rng(100 + N)
    S = np.clip(rng.normal(0.0, 2.0, N), -3.0, 3.0)
    g, dt_fr, reps = 40.0, 0.004, 40000          # large g: many candidates, so the cap binds
    kd_t, kc_t, cp_t = _law_torch(S, cap, g, dt_fr, reps, 1)
    kd_n, kc_n, cp_n = _law_numba(S, cap, g, dt_fr, reps, 2)
    se = lambda a: a.std() / math.sqrt(len(a)) + 1e-12
    for a_t, a_n in ((kd_t, kd_n), (kc_t, kc_n)):
        assert abs(a_t.mean() - a_n.mean()) < 5 * math.hypot(se(a_t), se(a_n))
    # per-walker expected number of copies after the event (1 = untouched)
    d = np.abs(cp_t.mean(0) - cp_n.mean(0))
    tol = 5 * np.hypot(cp_t.std(0), cp_n.std(0)) / math.sqrt(reps) + 1e-9
    assert np.all(d < tol), (d, tol)
    if cap == 0:
        assert kd_n.max() == 0 and kc_n.max() == 0 and kd_t.max() == 0 and kc_t.max() == 0


@pytest.mark.parametrize("N", [2, 6, 64])
def test_v4_loo_self_term(N):
    """Exploratory LOO score: p_loo(x_i) = KDE of the OTHER N-1 walkers at x_i in the full sample's mass
    convention, = p_others(x_i) * mass_others / mass_full (interior and edge points)."""
    rng = np.random.default_rng(7 + N)
    X = (np.concatenate([[eb.XMIN + 0.003, eb.XMIN + 0.06, 1.79, 1.73], rng.uniform(-0.15, 0.15, N - 4)]) if N > 2 else np.array([-0.05, 0.08]))
    dx = gn.grid_dx()
    kern, r = gn.gaussian_kernel_np(0.10, dx)
    p = np.zeros(eb.N_GRID); h = np.zeros(eb.N_GRID)
    mass = gn.kde_density(X, N, kern, r, dx, p, h)
    S = np.empty(N)
    gn.centred_scores(X, N, p, dx, 1e9, S, 2, kern, r, mass)
    lv = np.empty(N)
    for i in range(N):
        Xo = np.delete(X, i)
        po = np.zeros(eb.N_GRID); ho = np.zeros(eb.N_GRID)
        m_o = gn.kde_density(Xo, N - 1, kern, r, dx, po, ho)
        pos = np.clip((X[i] - eb.XMIN) / dx, 0, eb.N_GRID - 1); i0 = min(int(np.floor(pos)), eb.N_GRID - 2); fr_ = pos - i0
        val = (po[i0] + fr_ * (po[i0 + 1] - po[i0])) * m_o / mass
        assert val > 1e-8, "test geometry must keep every walker inside another's kernel"
        lv[i] = np.log(val)
    assert np.allclose(S, lv - lv.mean(), atol=1e-8)
    # and mode 1 is the plain empirical centring
    gn.centred_scores(X, N, p, dx, 1e9, S, 1, kern, r, mass)
    pos = np.clip((X - eb.XMIN) / dx, 0, eb.N_GRID - 1); i0 = np.minimum(np.floor(pos).astype(int), eb.N_GRID - 2); fr_ = pos - i0
    l1 = np.log(p[i0] + fr_ * (p[i0 + 1] - p[i0]))
    assert np.allclose(S, l1 - l1.mean(), atol=1e-12)
