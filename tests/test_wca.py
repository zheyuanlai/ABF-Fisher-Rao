"""WCA system validation: forces vs energy, Newton's third law, RC gradient,
reference guard, determinism, and a short integration smoke."""
import numpy as np
import pytest
import torch

from conftest import DEVICE, DTYPE
from abpfr.systems import wca
from abpfr.systems.gateway import Method, SHUS


def tiny_cfg(**kw):
    base = dict(beta=1.0, h=2.0, n_dim=4, K=8, dt=1e-3, n_steps=400, block=20,
                n_saves=5, ess_window_steps=200, force_clip=1e9)
    base.update(kw)
    return wca.WCAConfig(**base)


def random_box(cfg, n_boxes=3, seed=0, dtype=torch.float64):
    q = wca.lattice_init(cfg, n_boxes, seed, DEVICE, dtype=dtype, jitter=0.05)
    return q


def test_force_is_minus_grad_energy():
    cfg = tiny_cfg()
    eng = wca.WCAEngine(cfg, DEVICE)
    q = random_box(cfg)
    h = torch.full((q.shape[0], 1), cfg.h, dtype=torch.float64)
    f = eng.force(q, h)
    eps = 1e-6
    for (b, i, d) in [(0, 0, 0), (0, 1, 1), (1, 5, 0), (2, 11, 1)]:
        qp, qm = q.clone(), q.clone()
        qp[b, i, d] += eps
        qm[b, i, d] -= eps
        num = -(eng.energy(qp, h)[b] - eng.energy(qm, h)[b]) / (2 * eps)
        assert abs(float(f[b, i, d]) - float(num)) < 1e-5, (b, i, d)


def test_newtons_third_law():
    cfg = tiny_cfg()
    eng = wca.WCAEngine(cfg, DEVICE)
    q = random_box(cfg, seed=1)
    h = torch.full((q.shape[0], 1), cfg.h, dtype=torch.float64)
    f = eng.force(q, h)
    assert float(f.sum(dim=1).abs().max()) < 1e-10


def test_rc_gradient_matches_finite_difference():
    cfg = tiny_cfg()
    q = random_box(cfg, seed=2)
    forces = torch.zeros_like(q)
    one = torch.ones(q.shape[0], dtype=q.dtype)
    g = wca.add_rc_force(q, forces.clone(), one, cfg)
    eps = 1e-6
    for (b, i, d) in [(0, 0, 0), (1, 0, 1), (2, 1, 0)]:
        qp, qm = q.clone(), q.clone()
        qp[b, i, d] += eps
        qm[b, i, d] -= eps
        num = (wca.reaction_coordinate(qp, cfg)[b]
               - wca.reaction_coordinate(qm, cfg)[b]) / (2 * eps)
        assert abs(float(g[b, i, d]) - float(num)) < 1e-5


def test_minimum_image_and_wrap():
    L = 6.0
    d = wca.minimum_image(torch.tensor([5.5, -5.5, 2.0]), L)
    assert torch.allclose(d, torch.tensor([-0.5, 0.5, 2.0]))
    q = wca.wrap(torch.tensor([6.5, -0.5]), L)
    assert torch.allclose(q, torch.tensor([0.5, 5.5]))


def test_reference_guard():
    F_ref, meta = wca.load_reference(wca.WCAConfig(beta=1.0, h=2.0), device=DEVICE)
    assert F_ref.shape == (wca.GRID.n,)
    assert meta["reference_version"].startswith("hp")
    assert abs(float(F_ref[wca.GRID.eval_mask(DEVICE, DTYPE)].mean())) < 1e-9
    # wrong cell must be refused, loudly
    with pytest.raises(AssertionError):
        wca.load_reference(wca.WCAConfig(beta=2.0, h=6.0), device=DEVICE)


def test_lattice_init_no_overlap_and_compact_dimer():
    cfg = tiny_cfg()
    q = wca.lattice_init(cfg, 16, seed=3, device=DEVICE)
    z = wca.reaction_coordinate(q, cfg)
    assert float(z.abs().max()) < 1e-5          # dimer starts at xi = 0
    eng = wca.WCAEngine(cfg, DEVICE)
    qi = q.index_select(1, eng.pair_i)
    qj = q.index_select(1, eng.pair_j)
    r = torch.linalg.norm(wca.minimum_image(qi - qj, cfg.box_length), dim=-1)
    assert float(r.min()) > cfg.min_r


def test_integration_smoke_and_determinism():
    # n_dim=4 is NOT the reference cell, so this smoke runs unscored (the guard
    # refuses to score a mismatched cell -- covered below with the true cell)
    cfg = tiny_cfg(force_clip=250.0)
    fr = Method("fr", use_fr=True, theta=0.1, t_on_frac=0.1, t_off_frac=0.9,
                fr_every_blocks=2)
    a = wca.simulate_batch([cfg], [0], [SHUS, fr], batch_seed=5, device=DEVICE,
                           score_b1h2=False)
    b = wca.simulate_batch([cfg], [0], [SHUS, fr], batch_seed=5, device=DEVICE,
                           score_b1h2=False)
    for r in range(2):
        assert np.isfinite(a[r]["pmf_t"]).all()
        assert np.array_equal(a[r]["pmf_t"], b[r]["pmf_t"])
        assert abs(a[r]["P_regions"][-1].sum() - 1.0) < 1e-9
    r_fr = next(r for r in a if r["method"]["name"] == "fr")
    assert r_fr["event_turnover"].sum() >= 0


def test_true_cell_is_scored_against_reference():
    cfg = wca.WCAConfig(beta=1.0, h=2.0, K=4, n_steps=100, block=20, n_saves=3,
                        ess_window_steps=100)
    recs = wca.simulate_batch([cfg], [0], [SHUS], batch_seed=7, device=DEVICE)
    assert recs[0]["reference_id"] == wca.REFERENCE_ID
    assert np.isfinite(recs[0]["l2_f_t"]).all()
    # a mislabeled "b1h2" with the wrong box must be refused loudly
    bad = wca.WCAConfig(beta=1.0, h=2.0, n_dim=4, K=4, n_steps=100, block=20,
                        n_saves=3, ess_window_steps=100)
    with pytest.raises(AssertionError):
        wca.simulate_batch([bad], [0], [SHUS], batch_seed=8, device=DEVICE)


def test_non_b1h2_rows_carry_no_reference():
    cfg = tiny_cfg(beta=2.0, h=6.0, force_clip=250.0)
    recs = wca.simulate_batch([cfg], [0], [SHUS], batch_seed=6, device=DEVICE)
    assert recs[0]["reference_id"] == "none"
    assert np.isnan(recs[0]["l2_f_t"]).all()
    assert np.isfinite(recs[0]["kl_u_t"]).all()   # gates stay reference-free


def test_g_shus_wired_into_wca_engine():
    # g_shus=1.0 arm bitwise-matches the frozen default; g_shus=0.5 arm differs
    cfg = tiny_cfg(force_clip=250.0)
    recs = wca.simulate_batch([cfg], [0],
                              [SHUS, Method("g1", g_shus=1.0),
                               Method("g05", g_shus=0.5)],
                              batch_seed=23, device=DEVICE, score_b1h2=False)
    assert np.array_equal(recs[0]["pmf_t"], recs[1]["pmf_t"])
    assert not np.array_equal(recs[0]["pmf_t"], recs[2]["pmf_t"])
