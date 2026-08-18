"""2D engine validation: SHUS2 conventions (sign, gauge, reweighting consistency,
block-frozen deposits, estimator protection, gain), the 2D FR finite-step law,
2D count balancing, and the torus system end-to-end (pairing, determinism,
convergence, fixed-point floors)."""
import math

import numpy as np
import pytest
import torch

from conftest import DEVICE, DTYPE
from abpfr import grid2d as g2
from abpfr.events2d import fr_event2
from abpfr.fisher_rao import fr_weights
from abpfr.grid2d import GridT2
from abpfr.resampling import systematic_resample
from abpfr.shus2d import ShusAccumulator2, mollified_fixed_point2
from abpfr.systems import torus2d as t2
from abpfr.systems.gateway import Method

PI = math.pi
G = t2.GRID2


def make_shus2(rows=1, beta=1.0, eps_bw=0.15, gain=None):
    g = None if gain is None else torch.full((rows,), gain, dtype=DTYPE)
    return ShusAccumulator2(rows, G, torch.full((rows, 1), beta, dtype=DTYPE),
                            eps_bw, DEVICE, DTYPE, gain=g)


def torus_gauss_cloud(n, c1, c2, s, seed):
    gen = torch.Generator().manual_seed(seed)
    X1 = g2.wrap_periodic(c1 + s * torch.randn((1, n), generator=gen, dtype=DTYPE),
                          -PI, 2 * PI)
    X2 = g2.wrap_periodic(c2 + s * torch.randn((1, n), generator=gen, dtype=DTYPE),
                          -PI, 2 * PI)
    return X1, X2


# -----------------------------------------------------------------------------
# SHUS2 conventions
# -----------------------------------------------------------------------------
def test_shus2_sign_convention():
    s = make_shus2()
    X1, X2 = torus_gauss_cloud(8000, 0.0, 0.0, 0.2, seed=0)
    for _ in range(5):
        s.deposit(X1, X2)
    s.update(dt=1e-3, K=8000)
    F = s.F[0]
    # visited origin must lose free energy relative to the unvisited (pi, pi) point
    i0 = int(torch.argmin(g2.torus_distance(G.x1(DEVICE, DTYPE),
                                            torch.tensor(0.0, dtype=DTYPE),
                                            2 * PI)))
    ipi = int(torch.argmin(g2.torus_distance(G.x1(DEVICE, DTYPE),
                                             torch.tensor(PI, dtype=DTYPE),
                                             2 * PI)))
    assert float(F[i0, i0]) < float(F[ipi, ipi])


def test_shus2_gauge_invariance():
    s1, s2 = make_shus2(), make_shus2()
    s2.R = s2.R * 37.5
    s2._refresh_bias()
    X1, X2 = torus_gauss_cloud(4000, 1.0, -0.5, 0.6, seed=1)
    for s in (s1, s2):
        s.deposit(X1, X2)
        s.update(dt=1e-3, K=4000)
    assert torch.allclose(s1.Fp1, s2.Fp1, atol=1e-10)
    assert torch.allclose(s1.Fp2, s2.Fp2, atol=1e-10)
    assert torch.allclose(s1.R, s2.R, atol=1e-12)


def test_shus2_reweighting_consistency():
    """Samples from the biased equilibrium deposited with weight R_n increment R
    proportionally to exp(-beta F_true): the 2D fixed-point identity."""
    torch.manual_seed(3)
    beta = 1.0
    P1, P2 = G.mesh(DEVICE, DTYPE)
    F_true = 0.5 * (1 - torch.cos(P1)) + 0.3 * (1 - torch.cos(P2))
    s = make_shus2(beta=beta)
    s.R = torch.exp(-0.7 * (1 - torch.cos(P1 - 0.4))).unsqueeze(0)  # nontrivial R_n
    s._refresh_bias()
    dens = torch.exp(-beta * (F_true - s.F[0]))
    dens = (dens / dens.sum()).reshape(-1)
    idx = torch.multinomial(dens, 800_000, replacement=True)
    i1, i2 = idx // G.n2, idx % G.n2
    jit = lambda n: (torch.rand(n, dtype=DTYPE) - 0.5)
    X1 = g2.wrap_periodic(G.x1(DEVICE, DTYPE)[i1] + jit(len(i1)) * G.dx1,
                          -PI, 2 * PI).unsqueeze(0)
    X2 = g2.wrap_periodic(G.x2(DEVICE, DTYPE)[i2] + jit(len(i2)) * G.dx2,
                          -PI, 2 * PI).unsqueeze(0)
    s.deposit(X1, X2)
    inc = g2.smooth2(s.buf.reshape(1, G.n1, G.n2), s.k1, s.r1, s.k2, s.r2)
    target = torch.exp(-beta * F_true)
    ratio = (inc[0] / target).reshape(-1)
    # robust spread: with ~150 samples/node the max-min over 5184 nodes is
    # Poisson-tail-dominated; the bulk must be flat
    q = torch.quantile(ratio, torch.tensor([0.005, 0.995], dtype=DTYPE))
    rel_spread = float((q[1] - q[0]) / ratio.mean())
    assert rel_spread < 0.10, f"2D increment deviates from exp(-beta F): {rel_spread:.3f}"


def test_shus2_deposit_weight_is_block_frozen():
    s = make_shus2()
    X1 = torch.full((1, 100), 0.5, dtype=DTYPE)
    X2 = torch.full((1, 100), -0.5, dtype=DTYPE)
    s.deposit(X1, X2)
    buf1 = s.buf.clone()
    s.deposit(X1, X2)
    assert torch.allclose(s.buf, 2.0 * buf1, atol=1e-12)


def test_shus2_estimator_protection():
    s = make_shus2()
    X1, X2 = torus_gauss_cloud(1024, -1.0, 1.0, 0.3, seed=4)
    s.deposit(X1, X2)
    s.update(dt=1e-3, K=1024)
    R_snap, buf_snap = s.R.clone(), s.buf.clone()
    k1, r1 = g2.periodic_gaussian_kernel(0.25, G.dx1, G.n1, DEVICE, DTYPE)
    k2, r2 = g2.periodic_gaussian_kernel(0.25, G.dx2, G.n2, DEVICE, DTYPE)
    gen = torch.Generator().manual_seed(5)
    act = torch.tensor([True])
    sel, turn, th, ef = fr_event2(
        X1, X2, act, torch.tensor([False]), torch.tensor([False]), 0,
        torch.tensor([0], dtype=torch.long), torch.tensor([0.2], dtype=DTYPE),
        torch.tensor([0.5], dtype=DTYPE), k1, r1, k2, r2, G, gen)
    _ = torch.gather(X1, 1, sel)
    assert torch.equal(s.R, R_snap)
    assert torch.equal(s.buf, buf_snap)


def test_shus2_gain_scaling_and_identity():
    s0, s1, sg = make_shus2(), make_shus2(gain=1.0), make_shus2(gain=0.5)
    X1, X2 = torus_gauss_cloud(2000, 0.5, 0.5, 0.4, seed=6)
    incs = []
    for s in (s0, s1, sg):
        s.deposit(X1, X2)
        incs.append(s.update(dt=1e-3, K=2000))
    assert torch.equal(s0.R, s1.R)                       # g=1 bitwise
    assert torch.allclose(incs[2], 0.5 * incs[0], atol=1e-15)


# -----------------------------------------------------------------------------
# 2D FR finite-step law and count balancing
# -----------------------------------------------------------------------------
def test_fr2_resampled_population_matches_power_target():
    """Resampling with a_k = (u/p)^theta realizes p^+ ~ p^{1-theta} u^theta on T^2."""
    P1, P2 = G.mesh(DEVICE, DTYPE)
    p_grid = torch.exp(-2.0 * (1 - torch.cos(P1)) - 1.0 * (1 - torch.cos(P2)))
    p_grid = (p_grid / (p_grid.sum() * G.dA)).unsqueeze(0)
    dens = p_grid[0].reshape(-1) / p_grid[0].sum()
    gen0 = torch.Generator().manual_seed(7)
    idx = torch.multinomial(dens, 200_000, replacement=True, generator=gen0)
    i1, i2 = idx // G.n2, idx % G.n2
    jit = lambda n: (torch.rand(n, dtype=DTYPE, generator=gen0) - 0.5)
    X1 = g2.wrap_periodic(G.x1(DEVICE, DTYPE)[i1] + jit(len(i1)) * G.dx1,
                          -PI, 2 * PI).unsqueeze(0)
    X2 = g2.wrap_periodic(G.x2(DEVICE, DTYPE)[i2] + jit(len(i2)) * G.dx2,
                          -PI, 2 * PI).unsqueeze(0)
    theta = 0.5
    logr = g2.uniform_log_ratio2(X1, X2, p_grid, G)      # exact-density score
    w, _ = fr_weights(logr, torch.tensor([theta], dtype=DTYPE))
    gen = torch.Generator().manual_seed(8)
    sel = systematic_resample(w, gen)
    X1r = torch.gather(X1, 1, sel)
    X2r = torch.gather(X2, 1, sel)
    k1, r1 = g2.periodic_gaussian_kernel(0.25, G.dx1, G.n1, DEVICE, DTYPE)
    k2, r2 = g2.periodic_gaussian_kernel(0.25, G.dx2, G.n2, DEVICE, DTYPE)
    p_after = g2.binned_density2(X1r, X2r, k1, r1, k2, r2, G)
    q = p_grid[0] ** (1 - theta) * (1.0 / G.volume) ** theta
    q = (q / (q.sum() * G.dA)).unsqueeze(0)
    tv = 0.5 * float(g2.integral2((p_after - q).abs(), G))
    assert tv < 0.06, f"TV(resampled, p^(1-th) u^th) = {tv:.3f}"


def test_count_balancing2_equalizes_occupied_cells():
    """The coarse arm's weights are per-cell counts: resampling equalizes occupied
    coarse cells in expectation."""
    nb = 6
    gen = torch.Generator().manual_seed(9)
    # 3/4 of walkers crowded in one cell region, 1/4 spread
    X1a, X2a = torus_gauss_cloud(6000, 0.0, 0.0, 0.15, seed=10)
    X1b = g2.wrap_periodic((2 * PI) * torch.rand((1, 2000), generator=gen,
                                                 dtype=DTYPE) - PI, -PI, 2 * PI)
    X2b = g2.wrap_periodic((2 * PI) * torch.rand((1, 2000), generator=gen,
                                                 dtype=DTYPE) - PI, -PI, 2 * PI)
    X1 = torch.cat([X1a, X1b], dim=1)
    X2 = torch.cat([X2a, X2b], dim=1)
    k1, r1 = g2.periodic_gaussian_kernel(0.25, G.dx1, G.n1, DEVICE, DTYPE)
    k2, r2 = g2.periodic_gaussian_kernel(0.25, G.dx2, G.n2, DEVICE, DTYPE)
    sel, turn, th, ef = fr_event2(
        X1, X2, torch.tensor([True]), torch.tensor([False]), torch.tensor([True]),
        nb, torch.tensor([0], dtype=torch.long), torch.tensor([1.0], dtype=DTYPE),
        torch.tensor([0.0], dtype=DTYPE), k1, r1, k2, r2, G,
        torch.Generator().manual_seed(11))
    X1r = torch.gather(X1, 1, sel)
    X2r = torch.gather(X2, 1, sel)
    bw = 2 * PI / nb
    b = (torch.remainder(((X1r + PI) / bw).long(), nb) * nb
         + torch.remainder(((X2r + PI) / bw).long(), nb))
    cnt = torch.bincount(b[0], minlength=nb * nb).to(torch.float64)
    occ = cnt[cnt > 0]
    # occupied-cell counts should be far more even than before (crowded cell had 6000+)
    assert float(occ.max()) < 2.5 * float(occ.mean())


# -----------------------------------------------------------------------------
# torus system end-to-end
# -----------------------------------------------------------------------------
def small_cfg(**kw):
    base = dict(beta=1.0, H1=0.8, H2=0.8, Hc=0.2, K=256, dt=2e-3, n_steps=4000,
                block=20, n_saves=20, profile_every=4, ess_window_steps=2000,
                eps_bw=0.15, eta_bw=0.30)
    base.update(kw)
    return t2.Torus2DConfig(**base)


def test_reference_surface_and_floors():
    F = t2.reference_surface(2.0, 2.0, 0.5, DEVICE, DTYPE)
    assert abs(float(F.mean())) < 1e-10
    fp_fine = mollified_fixed_point2(F, 4.0, 0.03, G)
    fp_coarse = mollified_fixed_point2(F, 4.0, 0.30, G)
    assert fp_fine["e_star"] < 0.01 < fp_coarse["e_star"]
    assert fp_fine["kl_star"] < fp_coarse["kl_star"]


def test_paired_noise_identical_arms_and_gain_wiring():
    cfg = small_cfg(n_steps=2000, n_saves=10)
    recs = t2.simulate_batch([cfg], [0],
                             [Method("a"), Method("b", g_shus=1.0),
                              Method("c", g_shus=0.5)],
                             batch_seed=7, device=DEVICE, dtype=DTYPE)
    assert np.array_equal(recs[0]["pmf_t"], recs[1]["pmf_t"])       # paired + g=1
    assert not np.array_equal(recs[0]["pmf_t"], recs[2]["pmf_t"])   # gain differs


def test_deterministic_given_seed_with_fr():
    cfg = small_cfg(n_steps=2000, n_saves=10)
    fr = Method("fr", use_fr=True, theta=0.1, t_on_frac=0.1, t_off_frac=0.9,
                fr_every_blocks=5)
    a = t2.simulate_batch([cfg], [3], [Method("shus"), fr], batch_seed=42,
                          device=DEVICE, dtype=DTYPE)
    b = t2.simulate_batch([cfg], [3], [Method("shus"), fr], batch_seed=42,
                          device=DEVICE, dtype=DTYPE)
    for r in range(2):
        assert np.array_equal(a[r]["pmf_t"], b[r]["pmf_t"])
        assert np.array_equal(a[r]["event_turnover"], b[r]["event_turnover"])
    r_fr = a[1]
    assert r_fr["event_turnover"].sum() > 0                 # FR actually fired


def test_end_to_end_plain_shus_converges_and_floods():
    cfg = small_cfg(n_steps=30_000, n_saves=60)
    recs = t2.simulate_batch([cfg], [0], [Method("shus")], batch_seed=123,
                             device=DEVICE, dtype=DTYPE)
    r = recs[0]
    # error decreases substantially from the flat-bias start
    assert r["l2_f_t"][-1] < 0.35 * r["l2_f_t"][0]
    # marginal flattens: KL drops by an order of magnitude
    assert r["kl_u_t"][-1] < 0.15 * r["kl_u_t"][0]
    # all four basins get discovered (every region occupied at the end)
    assert (r["P_regions"][-1] > 0.02).all()
    # sham/region bookkeeping intact
    assert abs(r["P_regions"][-1].sum() - 1.0) < 1e-9


def test_sham_copies_partner_turnover_2d():
    cfg = small_cfg(n_steps=4000, n_saves=10)
    fr = Method("fr", use_fr=True, theta=0.2, t_on_frac=0.2, t_off_frac=0.8,
                fr_every_blocks=5)
    sham = Method("sham", use_fr=True, sham=True, shadows="fr")
    recs = t2.simulate_batch([cfg], [1], [Method("shus"), fr, sham], batch_seed=11,
                             device=DEVICE, dtype=DTYPE)
    r_fr = next(r for r in recs if r["method"]["name"] == "fr")
    r_sh = next(r for r in recs if r["method"]["name"] == "sham")
    assert np.array_equal(r_fr["event_turnover"], r_sh["event_turnover"])
    assert r_fr["event_turnover"].sum() > 0


def test_2d_record_schema_roundtrip(tmp_path):
    from abpfr.io import load_run, save_run
    cfg = small_cfg(n_steps=400, n_saves=5, profile_every=2)
    rec = t2.simulate_batch([cfg], [0], [Method("shus")], batch_seed=4,
                            device=DEVICE, dtype=DTYPE)[0]
    arrays = {k: rec[k] for k in ("time", "profile_time", "pmf_t", "marginal_t",
                                  "x1_grid", "x2_grid", "F_ref", "l2_f_t",
                                  "kl_u_t", "P_regions")}
    meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
            "config": rec["config"], "method": rec["method"], "seed": rec["seed"]}
    p = str(tmp_path / "rec2d")
    save_run(p, arrays, meta)
    back, meta2 = load_run(p)
    assert np.array_equal(back["pmf_t"], np.asarray(rec["pmf_t"]))
    assert meta2["reference_id"] == t2.REFERENCE_ID
    # a record with NO grid at all must still be refused
    bad = {k: v for k, v in arrays.items() if k not in ("x1_grid", "x2_grid")}
    with pytest.raises(AssertionError):
        save_run(str(tmp_path / "bad"), bad, meta)
