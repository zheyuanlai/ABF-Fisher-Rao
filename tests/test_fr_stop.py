"""Early-stopping FR schedule (engine cfg ``fr_stop_steps``): FR events only at steps fr_start <= k < fr_stop, after
which the run is plain histogram ABF on the SAME accumulators (no reset).  CPU only.

    NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" /home/zheyuanlai/miniconda3/envs/abffr/bin/python -m pytest -q tests/test_fr_stop.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from eri import engine as E  # noqa: E402
from eri.model import Model  # noqa: E402

KEYS = ("sv_C", "sv_M", "sv_h", "sv_i", "sv_u", "sv_cond")


@pytest.fixture(scope="module")
def model():
    return Model("gard_tait")


def _cfg(**k):
    base = dict(N=64, n_steps=3000, save_every=100, warmup_steps=500, fr_start_steps=500, fr_every=10, seed=11,
                fr_rate=20.0, trust_count=200.0)
    base.update(k)
    return E.make_cfg(**base)


def _done(c, method, model):
    r = E.Run(c, method, model=model)
    r.advance(c["n_steps"] + 1)
    return r


def test_stop_at_start_is_the_zero_fr_limit(model):
    c = _cfg()
    a = _done(c, "abf", model)
    z = _done(dict(c, fr_stop_steps=c["fr_start_steps"]), "fr", model)
    assert z.ist[E.S_REPL] == 0 and z.ist[E.S_NOPP] == 0
    for k in KEYS + ("q", "C", "Msum"):
        assert np.array_equal(getattr(a, k), getattr(z, k)), k


def test_stop_beyond_horizon_is_continuous_fr(model):
    c = _cfg()
    f = _done(c, "fr", model)
    g = _done(dict(c, fr_stop_steps=c["n_steps"] + 10), "fr", model)
    assert f.ist[E.S_REPL] > 0
    for k in KEYS + ("q", "anc"):
        assert np.array_equal(getattr(f, k), getattr(g, k)), k


def test_mid_run_stop(model):
    c = _cfg()
    stop = 1500
    f = _done(c, "fr", model)
    s = _done(dict(c, fr_stop_steps=stop), "fr", model)
    a = _done(c, "abf", model)
    st = s.save_steps
    pre = st < stop
    # identical to continuous FR at every save strictly before the stop
    for k in KEYS:
        assert np.array_equal(getattr(f, k)[pre], getattr(s, k)[pre]), k
    # FR really ran before the stop and never after it
    assert s.sv_i[pre, E.V_REPL][-1] > 0
    post = st >= stop
    assert (s.sv_i[post, E.V_REPL] == s.sv_i[post, E.V_REPL][0]).all()
    assert (s.sv_i[post, E.V_NOPP] == s.sv_i[post, E.V_NOPP][0]).all()
    assert s.ist[E.S_NOPP] == (stop - 1 - c["fr_start_steps"]) // c["fr_every"] + 1
    # the continuous arm keeps resampling after the stop time
    assert f.ist[E.S_REPL] > s.ist[E.S_REPL]
    # ABF keeps updating the same cumulative accumulators: every walker deposits at every step, in all arms
    for r in (a, f, s):
        assert np.array_equal(r.sv_C.sum(1), c["N"] * (st + 1.0))
        assert r.ist[E.S_NFE] == a.ist[E.S_NFE] == c["N"] * (c["n_steps"] + 1)
    # no estimator reset at the stop: the accumulators are monotone through it
    assert (np.diff(s.sv_C, axis=0) >= 0).all()


def test_fr_stop_is_inert_for_abf(model):
    c = _cfg()
    a = _done(c, "abf", model)
    b = _done(dict(c, fr_stop_steps=900), "abf", model)
    for k in KEYS + ("q",):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k


def test_checkpoint_resume_across_the_stop(model, tmp_path):
    c = _cfg(fr_stop_steps=1500)
    r1 = _done(c, "fr", model)
    r2 = E.Run(c, "fr", model=model)
    r2.advance(1203)
    p = str(tmp_path / "ck.npz")
    r2.checkpoint(p)
    r3 = E.Run(c, "fr", model=model)
    r3.restore(p)
    r3.advance(1777)
    r3.advance(c["n_steps"] + 1)
    for k in KEYS + ("q", "anc", "ist"):
        assert np.array_equal(getattr(r1, k), getattr(r3, k)), k
    with pytest.raises(RuntimeError):                 # a different stop time is a different config
        E.Run(dict(c, fr_stop_steps=1600), "fr", model=model).restore(p)
