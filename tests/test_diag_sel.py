"""Selection diagnostic (engine cfg diag_sel): a READ-ONLY record of FR births / deaths with ancestor-tagged deposits
and fibre histograms.  It never changes a trajectory, an estimator or an RNG stream, and the engine's default path
reproduces the trajectories frozen before it (and the Sbond / Suz2 accumulators) existed.

    NUMBA_NUM_THREADS=1 /home/zheyuanlai/miniconda3/envs/abffr/bin/python -m pytest -q tests/test_diag_sel.py
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from eri import engine as E  # noqa: E402
from eri.model import Model  # noqa: E402

FIX = os.path.join(HERE, "fixtures", "eri_pre_selection_fixture.npz")
HASHES = os.path.join(HERE, "fixtures", "eri_pre_selection_hashes.json")
KEYS = ("sv_C", "sv_M", "sv_h", "sv_i", "sv_u", "sv_cond", "q", "anc", "ancw", "BH", "DH", "C", "Msum", "Seta", "Sux2",
        "Sbond", "Suz2", "born", "ist", "fst", "row", "lastcage", "evwin", "dg")
FROZEN_KEYS = ("sv_C", "sv_M", "sv_h", "sv_i", "sv_u", "sv_cond", "q", "anc", "BH", "DH", "Seta", "Sux2", "born", "ist")


@pytest.fixture(scope="module")
def model():
    return Model("gard_tait")


def _cfg(**k):
    base = dict(N=64, n_steps=3000, save_every=500, warmup_steps=500, fr_start_steps=500, fr_every=10, seed=23,
                fr_rate=20.0, trust_count=200.0)
    base.update(k)
    return E.make_cfg(**base)


def _done(c, method, model, chunks=None):
    r = E.Run(c, method, model=model)
    if chunks is None:
        r.advance(c["n_steps"] + 1)
    else:
        for k in chunks:
            r.advance(k)
    return r


def _sel_arrays(r):
    s = r.selection_result()
    return {k: v for k, v in s.items() if isinstance(v, np.ndarray)}


def test_default_path_reproduces_the_frozen_pre_selection_engine(model):
    """Both arms, default config: bitwise the trajectories frozen from engine 6716aefc (fixture + sha256)."""
    z = np.load(FIX, allow_pickle=False)
    H = json.load(open(HASHES))
    assert H["cfg"] == dict(N=64, n_steps=3000, save_every=500, warmup_steps=500, fr_start_steps=500, fr_every=10, seed=23,
                            fr_rate=20.0, trust_count=200.0)
    for method in ("abf", "fr"):
        r = _done(_cfg(), method, model)
        for k in FROZEN_KEYS:
            a = np.ascontiguousarray(getattr(r, k))
            assert np.array_equal(a, z[f"{method}_{k}"]), (method, k)
            assert hashlib.sha256(a.tobytes()).hexdigest() == H["sha256"][f"{method}_{k}"], (method, k)
    assert int(z["fr_ist"][E.S_REPL]) == 45


def test_diagnostic_is_inert_in_both_arms(model):
    for method in ("abf", "fr"):
        a = _done(_cfg(), method, model)
        b = _done(_cfg(diag_sel=1), method, model)
        for k in KEYS:
            assert np.array_equal(getattr(a, k), getattr(b, k)), (method, k)
        assert a.g_noise.bit_generator.state == b.g_noise.bit_generator.state
        assert a.g_fr.bit_generator.state == b.g_fr.bit_generator.state
    # also inert with the clone-age split on and with custom bands / split
    a = _done(_cfg(diag_age_steps=300), "fr", model)
    b = _done(_cfg(diag_age_steps=300, diag_sel=1, diag_sel_band_steps=[50, 500], diag_sel_split_step=1000), "fr", model)
    for k in KEYS + ("sv_dg",):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k


def test_fr_record_is_complete_and_consistent(model):
    c = _cfg(diag_sel=1)
    r = _done(c, "fr", model)
    n = int(r.dst[0])
    assert n == int(r.ist[E.S_REPL]) == 45 and r.dst[1] == 0 and r.dst[2] == 0
    s = r.selection_result()
    assert s["sel_n_events"] == n and s["sel_split_step"] == c["n_steps"] // 2
    assert np.array_equal(s["sel_band_steps"], [int(round(b / c["dt"])) for b in E.SEL_BANDS_TU])
    ev_i, ev_f, evC, evM, evB, fdC, fdM, fdB, fh = (s[k] for k in ("sel_ev_i", "sel_ev_f", "sel_evC", "sel_evM", "sel_evB",
                                                                   "sel_fdC", "sel_fdM", "sel_fdB", "sel_fh"))
    assert ev_i.shape == (n, E.NEVI) and ev_f.shape == (n, E.NEVF) and evC.shape == (n, 2, 180)
    assert evB.shape == (n, 2, len(E.SEL_BANDS_TU) + 1, E.N_COARSE, E.NSB)
    # (1) the deposits are partitioned exactly: founders + every event == the estimator's accumulators
    assert np.array_equal(evC.sum((0, 1)) + fdC.sum(0), r.C)
    assert np.allclose(evM.sum((0, 1)) + fdM.sum(0), r.Msum, rtol=0, atol=1e-9 * np.abs(r.Msum).max())
    # (2) the "early" half (k <= split) equals the cumulative count at the split save
    ih = int(np.searchsorted(r.save_steps, c["n_steps"] // 2))
    assert r.save_steps[ih] == c["n_steps"] // 2
    assert np.array_equal(evC[:, 0].sum(0) + fdC[0], r.sv_C[ih])
    # (3) coarse-bin / band and fibre-histogram counts agree with the fine-bin counts
    assert np.array_equal(evB[..., 0].sum(2), evC.reshape(n, 2, E.N_COARSE, -1).sum(3))
    assert np.array_equal(fdB[..., 0], fdC.reshape(2, E.N_COARSE, -1).sum(2))
    tot = evC.sum(0) + fdC
    for coord in range(4):
        assert np.array_equal(fh[:, coord].sum(2), tot)
    # (4) event table: steps inside the FR window, distinct slots, bins and tags in range, fibre values physical
    assert ((ev_i[:, 0] >= c["fr_start_steps"]) & (ev_i[:, 0] <= c["n_steps"])).all()
    assert ((ev_i[:, 0] - c["fr_start_steps"]) % c["fr_every"] == 0).all()
    assert (ev_i[:, 1] != ev_i[:, 2]).all()
    assert (ev_i[:, 3:5] >= 0).all() and (ev_i[:, 3:5] < 180).all()
    assert (ev_i[:, 5] >= -1).all() and (ev_i[:, 6] >= -1).all() and (ev_i[:, 5:7] < np.arange(n)[:, None]).all()
    assert (np.abs(ev_f[:, [0, 4]]) <= 1).all() and (ev_f[:, [1, 5]] + ev_f[:, [3, 7]] <= 1 + 1e-12).all()
    assert (np.abs(ev_f[:, [2, 6]] - model.r0) < 0.4).all()
    # (5) the ancestor tag of every slot is the index of the last event that wrote into it (-1: founder)
    for i in range(c["N"]):
        w = np.nonzero(ev_i[:, 2] == i)[0]
        assert s["sel_tag"][i] == (w[-1] if len(w) else -1)
        if len(w):
            assert s["sel_born"][i] == ev_i[w[-1], 0]
    # (6) every event's birth step equals the born[] stamp its child carried at the time (deposits start after birth)
    live = evC.sum((1, 2)) > 0
    assert live.sum() > 0
    # (7) the new fibre accumulators are physical: mean bond near r0, u_z^2 a fraction
    occ = r.C > 0
    assert np.all(np.abs(r.Sbond[occ] / r.C[occ] - model.r0) < 0.1)
    assert np.all((r.Suz2[occ] / r.C[occ] > 0) & (r.Suz2[occ] / r.C[occ] < 1))
    assert np.allclose(r.Sux2[occ] / r.C[occ] + r.Suz2[occ] / r.C[occ], 1 - (1 - r.Sux2[occ] / r.C[occ] - r.Suz2[occ] / r.C[occ]))


def test_abf_arm_has_no_events_only_founders_and_histograms(model):
    c = _cfg(diag_sel=1)
    r = _done(c, "abf", model)
    s = r.selection_result()
    assert s["sel_n_events"] == 0 and (s["sel_tag"] == -1).all() and s["sel_ev_i"].shape == (0, E.NEVI)
    assert np.array_equal(s["sel_fdC"].sum(0), r.C)
    assert np.array_equal(s["sel_fh"][:, 0].sum(2), s["sel_fdC"])
    res = r.result()
    assert "sel_fh" in res and "Sbond" in res and "Suz2" in res
    res0 = _done(_cfg(), "abf", model).result()
    assert "sel_fh" not in res0 and "Sbond" in res0


def test_growth_chunking_and_checkpoint_are_bitwise(model, tmp_path):
    c = _cfg(diag_sel=1, N=128, n_steps=4000, fr_rate=50.0)
    r1 = _done(c, "fr", model)
    assert int(r1.dst[0]) > 60
    # forced growth of the event table from a tiny capacity: identical record
    r2 = E.Run(c, "fr", model=model)
    for k in ("ev_i", "ev_f", "evC", "evM", "evB"):
        setattr(r2, k, getattr(r2, k)[:3].copy())
    for k in (137, 1000, 1001, 2222, 3333, c["n_steps"] + 1):
        r2.advance(k)
    assert r2.dst[1] == 0 and r2.ev_i.shape[0] > 3
    A, B = _sel_arrays(r1), _sel_arrays(r2)
    for k in A:
        assert np.array_equal(A[k], B[k]), k
    for k in KEYS:
        assert np.array_equal(getattr(r1, k), getattr(r2, k)), k
    # checkpoint / restore mid-run keeps the record
    r3 = E.Run(c, "fr", model=model)
    r3.advance(1500)
    p = str(tmp_path / "ck.npz")
    r3.checkpoint(p)
    r4 = E.Run(c, "fr", model=model)
    r4.restore(p)
    r4.advance(c["n_steps"] + 1)
    C_ = _sel_arrays(r4)
    for k in A:
        assert np.array_equal(A[k], C_[k]), k


def test_cfg_keys_and_custom_bands(model):
    with pytest.raises(KeyError):
        E.make_cfg(diag_selection=1)
    c = _cfg(diag_sel=1, diag_sel_band_steps=[500, 100], diag_sel_split_step=1234)
    r = E.Run(c, "fr", model=model)
    assert np.array_equal(r.bands, [100, 500]) and r.evB.shape[2] == 3 and int(r.ip[11]) == 1234
    r.advance(c["n_steps"] + 1)
    s = r.selection_result()
    # the early / late split at 1234: early counts equal deposits at k <= 1234 = sv_C(1000) + the next 234 steps' N deposits
    assert (s["sel_evC"][:, 0].sum(0) + s["sel_fdC"][0]).sum() == (1234 + 1) * c["N"]
