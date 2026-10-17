"""Clone-age diagnostic (engine cfg diag_age_steps): a read-only split of the deposits; it never changes a trajectory.

    NUMBA_NUM_THREADS=1 /home/zheyuanlai/miniconda3/envs/abffr/bin/python -m pytest -q tests/test_diag_age.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from eri import engine as E  # noqa: E402
from eri.model import Model  # noqa: E402

KEYS = ("sv_C", "sv_M", "sv_h", "sv_i", "sv_u", "sv_cond", "q", "anc", "BH", "DH", "C", "Msum")


@pytest.fixture(scope="module")
def model():
    return Model("gard_tait")


def _cfg(**k):
    base = dict(N=64, n_steps=3000, save_every=100, warmup_steps=500, fr_start_steps=500, fr_every=10, seed=23,
                fr_rate=20.0, trust_count=200.0)
    base.update(k)
    return E.make_cfg(**base)


def _done(c, method, model):
    r = E.Run(c, method, model=model)
    r.advance(c["n_steps"] + 1)
    return r


def test_diagnostic_is_inert_and_consistent(model):
    a = _done(_cfg(), "fr", model)
    b = _done(_cfg(diag_age_steps=300), "fr", model)
    for k in KEYS:
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert b.ist[E.S_REPL] > 0 and b.dg[0].sum() > 0
    assert (b.dg[0] <= b.C).all()                                   # young deposits are a subset of all deposits
    assert np.allclose(b.sv_dg[-1, 0], b.dg[0]) and (np.diff(b.sv_dg[:, 0].sum(1)) >= 0).all()
    # a walker counts as young for exactly diag_age_steps steps after each clone: young count <= events x age window
    assert b.dg[0].sum() <= b.ist[E.S_REPL] * 300


def test_no_young_deposits_without_fr(model):
    r = _done(_cfg(diag_age_steps=300), "abf", model)
    assert r.dg[0].sum() == 0 and r.born.max() < 0
