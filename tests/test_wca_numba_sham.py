"""Matched-sham arm of src/wca_numba.py (2026-10-09).

S1  adding a sham arm does not change the ABF and FR arms (bitwise), so earlier results stay reproducible;
S2  the sham replaces exactly as many replicas as its partner at every opportunity (cumulative counts and
    event-opportunity counts equal at every save), with its own genealogy bookkeeping;
S3  the sham's deaths are uniform over replicas (statistical, through the run-long ancestor counts of a
    long N = 8 run with frequent events: every original lineage has the same expected fate).
"""
import os
import sys

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_numba as wn  # noqa: E402

CFG = dict(wn.ACCEPTED_CFG, fr_start_steps=200, fr_every=5, fr_rate=2.0)


def test_S1_sham_does_not_perturb_other_arms():
    save = np.array([500, 1000, 1500])
    a = wn.run_ladder_point(3100, 16, 1500, [("abf", False), ("fr", True)], cfg=CFG, save_at=save, cap_min=1)
    b = wn.run_ladder_point(3100, 16, 1500, [("abf", False), ("fr", True), ("sham", "sham:fr")], cfg=CFG, save_at=save, cap_min=1)
    for k in ("M_bias", "C_bias", "repl_cumulative", "ancestor_ess", "q_final"):
        assert np.array_equal(np.asarray(a[k])[:2], np.asarray(b[k])[:2], equal_nan=True), k
    assert int(np.asarray(a["repl_cumulative"])[1, -1]) > 0


def test_S2_sham_replays_partner_counts():
    save = np.arange(250, 3001, 250)
    r = wn.run_ladder_point(3101, 32, 3000, [("abf", False), ("fr", True), ("sham", "sham:fr")], cfg=CFG, save_at=save, cap_min=1)
    rep = np.asarray(r["repl_cumulative"])
    nev = np.asarray(r["n_event_opportunities"])
    assert rep[1, -1] > 20
    assert np.array_equal(rep[1], rep[2]) and np.array_equal(nev[1], nev[2])
    assert np.all(np.isfinite(np.asarray(r["ancestor_ess"])[2]))           # sham genealogy reported
    assert not np.array_equal(np.asarray(r["q_final"])[1], np.asarray(r["q_final"])[2])
    assert np.isnan(np.asarray(r["min_ancestor_ess_window"])[0]) and np.isfinite(np.asarray(r["min_ancestor_ess_window"])[2])


def test_S3_sham_law_exact():
    """sham_select against the exact law: P(i dies) = k/N for every i; the k deaths are distinct; every
    source is a survivor; P(source = j | j survives) = 1/(N - k); sources iid (pairs uniform)."""
    rng = np.random.Generator(np.random.PCG64(7))
    N, k, n = 10, 3, 200_000
    perm = np.empty(N, np.int64); de = np.empty(N, np.int64); so = np.empty(N, np.int64)
    dcount = np.zeros(N); scount = np.zeros(N); same_pair = 0
    for _ in range(n):
        wn.sham_select(N, k, rng, perm, de, so)
        d = de[:k]; s_ = so[:k]
        assert len(set(d.tolist())) == k and not set(s_.tolist()) & set(d.tolist())
        dcount[d] += 1
        np.add.at(scount, s_, 1)
        same_pair += int(s_[0] == s_[1])
    pd = dcount / n
    se = np.sqrt((k / N) * (1 - k / N) / n)
    assert np.all(np.abs(pd - k / N) < 5 * se), pd
    ps = scount / (n * k)                       # marginal source frequency: uniform 1/N by symmetry
    assert np.all(np.abs(ps - 1 / N) < 5 * np.sqrt((1 / N) / (n * k))), ps
    p_same = same_pair / n                      # P(s1 == s2) = 1/(N - k) for iid uniform survivors
    assert abs(p_same - 1 / (N - k)) < 5 * np.sqrt((1 / (N - k)) / n), p_same
