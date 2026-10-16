"""Gate 4 (FR mechanism) and estimator tests for the ERI engine (src/eri/engine.py).  CPU only.

    NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" /home/zheyuanlai/miniconda3/envs/abffr/bin/python -m pytest -q tests/test_eri_engine.py
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from eri import engine as E  # noqa: E402
from eri.model import Model, cv_phi, local_mean_force  # noqa: E402
from lta_ladder_numba import _fr_score, fr_select_native, kde_matrix  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return Model("gard_tait")


def _cfg(**k):
    base = dict(N=64, n_steps=3000, save_every=500, warmup_steps=500, fr_start_steps=500, fr_every=10, seed=5,
                fr_rate=5.0, trust_count=200.0)
    base.update(k)
    return E.make_cfg(**base)


def test_arms_bitwise_identical_before_fr_start_and_rate0_equals_abf(model):
    c = _cfg()
    a = E.Run(c, "abf", model=model)
    f = E.Run(c, "fr", model=model)
    a.advance(c["fr_start_steps"])
    f.advance(c["fr_start_steps"])
    for k in ("q", "C", "Msum", "row", "lastcage"):
        assert np.array_equal(getattr(a, k), getattr(f, k)), k
    c0 = dict(c, fr_rate=0.0)
    z = E.Run(c0, "fr", model=model)
    a.advance(c["n_steps"] + 1)
    z.advance(c["n_steps"] + 1)
    assert np.array_equal(a.q, z.q) and np.array_equal(a.sv_C, z.sv_C) and np.array_equal(a.sv_M, z.sv_M)
    assert z.ist[E.S_REPL] == 0


def test_chunking_and_checkpoint_resume_are_bitwise(model, tmp_path):
    c = _cfg()
    r1 = E.Run(c, "fr", model=model)
    r1.advance(c["n_steps"] + 1)
    r2 = E.Run(c, "fr", model=model)
    for k in (137, 1000, 1001, 2222, c["n_steps"] + 1):
        r2.advance(k)
    assert np.array_equal(r1.q, r2.q) and np.array_equal(r1.sv_u, r2.sv_u) and np.array_equal(r1.ist, r2.ist)
    r3 = E.Run(c, "fr", model=model)
    r3.advance(1500)
    p = str(tmp_path / "ck.npz")
    r3.checkpoint(p)
    r4 = E.Run(c, "fr", model=model)
    r4.restore(p)
    r4.advance(c["n_steps"] + 1)
    assert np.array_equal(r1.q, r4.q) and np.array_equal(r1.sv_C, r4.sv_C) and np.array_equal(r1.anc, r4.anc)
    bad = dict(c, fr_rate=1.0)
    with pytest.raises(RuntimeError):
        E.Run(bad, "fr", model=model).restore(p)


def test_fr_bookkeeping_and_genealogy(model):
    c = _cfg(n_steps=4000, fr_rate=50.0, N=128)
    r = E.Run(c, "fr", model=model)
    r.advance(c["n_steps"] + 1)
    n = int(r.ist[E.S_REPL])
    assert n > 0
    assert r.BH.sum() == n and r.DH.sum() == n
    assert r.sv_i[-1, E.V_REPL] == n
    assert (r.anc >= 0).all() and (r.anc < c["N"]).all()
    assert r.sv_i[-1, E.V_NU] == len(np.unique(r.anc))
    assert 0 < r.sv_u[-1, E.U_ESS] <= 1.0 and r.sv_u[-1, E.U_MAXF] >= 1.0 / c["N"]
    # per opportunity at most cap events
    assert n <= r.ist[E.S_NOPP] * c["cap"]
    # windowed ancestors are reset at window boundaries: at least as diverse as the whole-run labels
    assert (r.sv_i[:, E.V_NUW] >= r.sv_i[:, E.V_NU]).all()


def test_fr_score_and_selection_direction():
    """Over-populated bins lose walkers, under-populated bins are the sources (uniform target)."""
    ng = 180
    Kk, grid, dphi = kde_matrix(ng, 0.10)
    N = 400
    rng = np.random.default_rng(0)
    phin = np.concatenate([rng.normal(2.5, 0.05, 380), rng.normal(-0.2, 0.3, 20)])
    phin = (phin + np.pi) % (2 * np.pi) - np.pi
    jn = (np.floor((phin + np.pi) / dphi).astype(np.int64)) % ng
    cnt = np.zeros(ng); pk = np.zeros(ng); raw = np.zeros(N); sc = np.zeros(N); touched = np.zeros(ng, np.int64)
    qv = 1.0 / (ng * dphi)
    _fr_score(phin, jn, N, ng, dphi, Kk, grid[0], qv, math.log(qv), 2.0, cnt, pk, raw, sc, touched)
    assert sc[:380].mean() > 0 > sc[380:].mean()
    cap = 50
    ublk = rng.random(N + 2 * cap)
    cand = np.zeros(N, np.int64); dsel = np.zeros(cap, np.int64); ssel = np.zeros(cap, np.int64)
    dp = np.zeros(N); cdf = np.zeros(N)
    n = fr_select_native(sc, N, cap, 1000.0, 1e-3, ublk, cand, dsel, ssel, dp, cdf)
    assert n > 0
    assert (dsel[:n] < 380).all()                 # deaths from the crowded cluster only
    # sources ~ S-: mostly the sparse cluster (its walkers all have S < 0); the crowded cluster's far tails can
    # also have S < 0 (density below the population's KL-weighted mean), so only a majority is asserted
    assert (ssel[:n] >= 380).mean() > 0.8
    assert (sc[ssel[:n]] < 0).all()
    assert len(set(dsel[:n]) & set(ssel[:n])) == 0


def test_clone_copies_the_whole_replica_with_guests(model):
    c = _cfg(M=3, N=48, n_steps=2500, fr_rate=200.0)
    r = E.Run(c, "fr", model=model)
    q_before = None
    hit = False
    for k in range(c["fr_start_steps"] - 1, c["n_steps"], c["fr_every"]):
        r.advance(k)                               # next step k: forces at k, move, FR at k + 1 (scheduled)
        q_before = r.q.copy()
        n0 = int(r.ist[E.S_REPL])
        r.advance(k + 1)
        if r.ist[E.S_REPL] > n0:
            hit = True
            break
    assert hit
    # after the event every replica is (up to the one EM move of this step) a configuration of the pre-step set;
    # exactness: the dying slots now hold an EXACT copy of another slot's post-move replica, guests included
    dup = 0
    for i in range(c["N"]):
        for j in range(i + 1, c["N"]):
            if np.array_equal(r.q[i], r.q[j]):
                dup += 1
                assert np.array_equal(r.q[i, 1:], r.q[j, 1:])
    assert dup >= 1


def test_deposit_equals_model_mean_force_and_bias_displacement(model):
    c = _cfg(N=8, n_steps=10, warmup_steps=0, fr_start_steps=10 ** 9, trust_count=0.0)
    r = E.Run(c, "abf", model=model)
    q0 = r.q.copy()
    F0 = np.zeros_like(q0)
    f = np.zeros(8)
    jb = np.zeros(8, int)
    for i in range(8):
        _, F = model.energy_force(q0[i])
        F0[i] = F
        f[i] = local_mean_force(F, model.fw.P)
        jb[i] = int(np.floor((cv_phi(q0[i], model.fw.P, model.fw.x_window) + np.pi) / (2 * np.pi / 180))) % 180
    r.advance(1)
    C = np.zeros(180); Mv = np.zeros(180)
    for i in range(8):
        C[jb[i]] += 1; Mv[jb[i]] += f[i]
    assert np.allclose(r.C, C) and np.allclose(r.Msum, Mv, rtol=1e-12, atol=1e-12)
    # the move: q1 = q0 + dt (F + Gamma grad phi) + noise; replay the noise stream
    g = E.seed_streams(c)[1]
    noise = g.standard_normal(q0.size).reshape(q0.shape)
    gam = np.where(C > 0, Mv / np.maximum(C, 1), 0)[jb]
    bias = np.zeros_like(q0)
    bias[:, 0, :, 0] = (gam * np.pi / model.fw.P)[:, None]
    q1 = q0 + c["dt"] * (F0 + bias) + math.sqrt(2 * c["dt"] * model.kT) * noise
    assert np.allclose(r.q, q1, rtol=0, atol=1e-12)


def test_histogram_pmf_reconstructs_known_profile():
    ng = 180
    dphi = 2 * np.pi / ng
    e = -np.pi + np.arange(ng + 1) * dphi
    Ftrue = lambda x: 3.0 * np.cos(x) + 0.5 * np.sin(2 * x)
    gamma = (Ftrue(e[1:]) - Ftrue(e[:-1])) / dphi           # exact bin-average force
    F = E.histogram_pmf(gamma)
    c = 0.5 * (e[1:] + e[:-1])
    ref = Ftrue(c) - Ftrue(c).mean()
    assert np.abs(F - ref).max() < 5e-3                     # O(dphi^2) cell-centre discretisation
