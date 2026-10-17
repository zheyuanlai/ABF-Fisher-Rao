"""Arrival / provenance diagnostic (engine cfg diag_arr): a READ-ONLY record of where each deposit's walker came from
(last visited region, time since leaving it; last coarse-bin entry side and residence time) with fibre, transverse-COM
and sub-bin-offset sums.  It never changes a trajectory, an estimator or an RNG stream; the default path still
reproduces the trajectories frozen before any diagnostic existed (tests/test_diag_sel.py).

    NUMBA_NUM_THREADS=1 /home/zheyuanlai/miniconda3/envs/abffr/bin/python -m pytest -q tests/test_diag_arr.py
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from eri import engine as E  # noqa: E402
from eri.model import Model  # noqa: E402

KEYS = ("sv_C", "sv_M", "sv_h", "sv_i", "sv_u", "sv_cond", "q", "anc", "ancw", "BH", "DH", "C", "Msum", "Seta", "Sux2",
        "Sbond", "Suz2", "Scy", "Ssy", "Scz", "Ssz", "born", "ist", "fst", "row", "lastcage", "evwin", "dg")
GRID = (np.arange(180) + 0.5) / 180 * 2 * math.pi - math.pi


@pytest.fixture(scope="module")
def model():
    return Model("gard_tait")


def _cfg(**k):
    base = dict(N=64, n_steps=3000, save_every=500, warmup_steps=500, fr_start_steps=500, fr_every=10, seed=23,
                fr_rate=20.0, trust_count=200.0)
    base.update(k)
    return E.make_cfg(**base)


def _done(c, method, model):
    r = E.Run(c, method, model=model)
    r.advance(c["n_steps"] + 1)
    return r


def test_diagnostic_is_inert_in_both_arms_and_with_diag_sel(model):
    for method in ("abf", "fr"):
        a = _done(_cfg(), method, model)
        b = _done(_cfg(diag_arr=1), method, model)
        d = _done(_cfg(diag_arr=1, diag_sel=1, diag_arr_band_steps=[20, 200]), method, model)
        for k in KEYS:
            assert np.array_equal(getattr(a, k), getattr(b, k)), (method, k)
            assert np.array_equal(getattr(a, k), getattr(d, k)), (method, k)
        assert a.g_noise.bit_generator.state == b.g_noise.bit_generator.state
        assert a.g_fr.bit_generator.state == b.g_fr.bit_generator.state
        # the selection record is unchanged by the arrival record
        if method == "fr":
            e = _done(_cfg(diag_sel=1), method, model).selection_result()
            f = d.selection_result()
            for k in ("sel_ev_i", "sel_ev_f", "sel_evC", "sel_evM", "sel_evB", "sel_fh"):
                assert np.array_equal(e[k], f[k]), k


def test_record_partitions_the_deposits_and_labels_regions(model):
    c = _cfg(diag_arr=1)
    for method in ("abf", "fr"):
        r = _done(c, method, model)
        s = r.arrival_result()
        P, Ee = s["arr_P"], s["arr_E"]
        nb = len(E.ARR_BANDS_TU) + 1
        assert P.shape == (2, 3, nb, 180, E.NAQ) and Ee.shape == P.shape
        assert np.array_equal(s["arr_band_steps"], [int(round(b / c["dt"])) for b in E.ARR_BANDS_TU])
        # (1) exact partition: both records sum to the estimator's accumulators, per half
        ih = int(np.searchsorted(r.save_steps, c["n_steps"] // 2))
        assert np.array_equal(P[..., 0].sum((1, 2)), np.stack([r.sv_C[ih], r.C - r.sv_C[ih]]))
        assert np.array_equal(Ee[..., 0].sum((1, 2)), np.stack([r.sv_C[ih], r.C - r.sv_C[ih]]))
        assert np.allclose(P[..., 1].sum((0, 1, 2)), r.Msum, rtol=0, atol=1e-9 * np.abs(r.Msum).max())
        assert np.allclose(Ee[..., 1].sum((0, 1, 2)), r.Msum, rtol=0, atol=1e-9 * np.abs(r.Msum).max())
        for qi, acc in ((2, r.Seta), (3, r.Sux2), (4, r.Sbond), (5, r.Suz2), (6, r.Scy), (7, r.Ssy), (8, r.Scz), (9, r.Ssz)):
            assert np.allclose(P[..., qi].sum((0, 1, 2)), acc, rtol=0, atol=1e-9 * max(np.abs(acc).max(), 1))
        # (2) deposits in the cage region are 'cage provenance, time since leaving = 0'; window likewise
        # (bins WHOLLY inside a region: the region edges fall inside fine bins, where deposits split by their own phi)
        lo, hi = GRID - math.pi / 180, GRID + math.pi / 180
        cage = (np.abs(lo) >= c["cage_phi"]) & (np.abs(hi) >= c["cage_phi"])
        win = (np.abs(lo) < c["window_phi"]) & (np.abs(hi) < c["window_phi"])
        assert cage.sum() > 50 and win.sum() > 20
        assert np.array_equal(P[:, 0, 0][:, cage, 0], P[..., 0].sum((1, 2))[:, cage])
        assert np.array_equal(P[:, 1, 0][:, win, 0], P[..., 0].sum((1, 2))[:, win])
        # the flank has deposits of both provenances (and 'none' only before the first region visit)
        flank = (np.abs(GRID) >= c["window_phi"]) & (np.abs(GRID) < c["cage_phi"])
        assert P[:, 0, :, flank, 0].sum() > 0
        # window provenance outside the window needs a walker to have left the window within this short run; if none
        # did, every window-provenance deposit is inside the window itself
        assert P[:, 1, :, flank, 0].sum() > 0 or P[:, 1, ..., 0].sum() == P[:, 1, 0][:, win, 0].sum() + P[:, 1, 0][:, ~win & ~cage, 0].sum()
        assert P[1, 2].sum() <= P[0, 2].sum()
        # (3) sub-bin offsets lie within half a bin; transverse phases are bounded
        cnt = P[..., 0]
        off = np.where(cnt > 0, P[..., 10] / np.maximum(cnt, 1), 0.0)
        assert np.abs(off).max() <= math.pi / 180 + 1e-12
        for qi in (6, 7, 8, 9):
            assert (np.abs(P[..., qi]) <= cnt + 1e-9).all()
        # (4) entry side: 'unknown' only at the start (k = 0 labels every walker) or after a multi-bin jump
        assert Ee[0, 2, :, :, 0].sum() >= c["N"]
        assert Ee[:, :2, :, :, 0].sum() > 0.9 * Ee[..., 0].sum()
        # (5) per-walker state is consistent with the final positions
        ph = np.array([E._wrap(2 * math.pi * (0.5 * (r.q[i, 0, 0, 0] + r.q[i, 0, 1, 0]) - model.fw.x_window) / model.fw.P) for i in range(c["N"])])
        cb_final = (np.floor((ph + math.pi) / (2 * math.pi / 180)).astype(int) % 180) * 18 // 180
        assert np.array_equal(s["arr_ent_cb"], cb_final)
        assert (s["arr_last_cage_step"] <= c["n_steps"]).all() and (s["arr_ent_step"] <= c["n_steps"]).all()


def test_clones_inherit_and_chunking_checkpoint_are_bitwise(model, tmp_path):
    c = _cfg(diag_arr=1, N=128, n_steps=4000, fr_rate=50.0)
    r1 = _done(c, "fr", model)
    assert int(r1.ist[E.S_REPL]) > 60
    r2 = E.Run(c, "fr", model=model)
    for k in (137, 1000, 1001, 2222, 3333, c["n_steps"] + 1):
        r2.advance(k)
    r3 = E.Run(c, "fr", model=model)
    r3.advance(1500)
    p = str(tmp_path / "ck.npz")
    r3.checkpoint(p)
    r4 = E.Run(c, "fr", model=model)
    r4.restore(p)
    r4.advance(c["n_steps"] + 1)
    for other in (r2, r4):
        for k in ("arrP", "arrE", "ent_step", "ent_side", "ent_cb", "lcage_k", "lwin_k") + KEYS:
            assert np.array_equal(getattr(r1, k), getattr(other, k)), k
    # inheritance: a clone's provenance stamps equal its parent's at the event, so no slot's 'last cage step' can
    # exceed the current step and every slot that sits in the cage now has last cage step == final step
    ph = np.array([E._wrap(2 * math.pi * (0.5 * (r1.q[i, 0, 0, 0] + r1.q[i, 0, 1, 0]) - model.fw.x_window) / model.fw.P) for i in range(c["N"])])
    incage = np.abs(ph) >= c["cage_phi"]
    assert (r1.lcage_k[incage] == c["n_steps"]).all()


def test_cfg_keys(model):
    with pytest.raises(KeyError):
        E.make_cfg(diag_arrival=1)
    r = E.Run(_cfg(diag_arr=1, diag_arr_band_steps=[500, 100]), "abf", model=model)
    assert np.array_equal(r.abands, [100, 500]) and r.arrP.shape[2] == 3
    res = _done(_cfg(), "abf", model).result()
    assert "arr_P" not in res and "Scy" in res
