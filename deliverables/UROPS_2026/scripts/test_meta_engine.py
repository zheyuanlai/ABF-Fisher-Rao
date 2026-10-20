"""Tests of meta_engine.py (run: python -m pytest -q test_meta_engine.py, with ABFFR_DATA_ROOT set)."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

C.setup_paths()
import meta_engine as E  # noqa: E402
from abffr import potentials as P  # noqa: E402

RNG = np.random.default_rng(20261020)
PTS = np.column_stack([RNG.uniform(E.XMIN, E.XMAX, 400), RNG.uniform(E.YMIN, E.YMAX, 400)])
PTS = np.vstack([PTS, [[1.048, -0.042], [-1.048, -0.042], [0.0, 1.537], [0.0, 0.0], [0.0, 1 / 3], [3.0, 3.5], [-3.0, -2.5]]])


def test_potential_matches_archived_module():
    for x, y in PTS:
        assert abs(E.V(x, y) - float(P.potential_xy(x, y))) <= 1e-12 * max(1.0, abs(E.V(x, y)))
        gx, gy = E.grad(x, y)
        assert abs(gx - float(P.dVdx_xy(x, y))) <= 1e-12 * max(1.0, abs(gx))
        assert abs(gy - float(P.dVdy_xy(x, y))) <= 1e-12 * max(1.0, abs(gy))


def test_gradient_vs_central_differences():
    e = 1e-6
    worst = 0.0
    for x, y in PTS:
        gx, gy = E.grad(x, y)
        fx = (E.V(x + e, y) - E.V(x - e, y)) / (2 * e)
        fy = (E.V(x, y + e) - E.V(x, y - e)) / (2 * e)
        worst = max(worst, abs(gx - fx) / max(1.0, abs(gx)), abs(gy - fy) / max(1.0, abs(gy)))
    assert worst < 1e-7, worst


def test_reflect():
    assert E.reflect(-3.1, -3.0, 3.0) == -2.9 or abs(E.reflect(-3.1, -3.0, 3.0) + 2.9) < 1e-12
    assert abs(E.reflect(3.25, -3.0, 3.0) - 2.75) < 1e-12
    assert abs(E.reflect(0.3, -3.0, 3.0) - 0.3) < 1e-15
    assert abs(E.reflect(3.6, -2.5, 3.5) - 3.4) < 1e-12


def small_cfg(**kw):
    c = dict(N=64, T=2.0, h=1e-3, nb=60, save_dt=0.5, fr_interval_t=0.01, genealogy_window_t=1.0, gamma=5.0, ramp_t=0.5,
             max_event_fraction=0.05)
    c.update(kw)
    return c


def test_predeposit_convention_and_determinism():
    a = E.run_arm(small_cfg(), "abf", 11)
    b = E.run_arm(small_cfg(), "abf", 11)
    assert np.array_equal(a["M_all"], b["M_all"]) and np.array_equal(a["X_final"], b["X_final"])
    assert np.allclose(a["C_all"].sum(1), 64 * a["save_step"])
    assert np.all(a["hist_inst"].sum(1) == 64)


def test_arms_share_noise_until_first_fr_event():
    a = E.run_arm(small_cfg(gamma=0.0), "abf", 5)
    f0 = E.run_arm(small_cfg(gamma=0.0), "fr", 5)            # FR at zero rate = ABF bitwise
    assert np.array_equal(a["M_all"], f0["M_all"]) and np.array_equal(a["X_final"], f0["X_final"])
    f = E.run_arm(small_cfg(), "fr", 5)
    assert f["deaths_cum"][-1] > 0
    first = int(np.argmax(f["deaths_cum"] > 0))
    assert np.array_equal(a["M_all"][:first], f["M_all"][:first])


def test_fr_cap_and_population():
    f = E.run_arm(small_cfg(gamma=50.0), "fr", 3)
    assert f["max_event_fraction_realised"] <= f["cap"] / 64 + 1e-12
    assert np.all(f["hist_inst"].sum(1) == 64)


def test_kde_normalisation():
    g, d, K = E.kde_matrix(0.1)
    cnt = np.zeros(g.size)
    cnt[[0, 300, 600]] = 1.0
    p = K @ cnt / 3.0
    m = d * (p.sum() - 0.5 * (p[0] + p[-1]))
    assert abs(m - 1.0) < 2e-3          # reflected images keep the mass inside the box (trapezoid error only)


def test_scorer_floor_and_exact_gamma():
    sc = E.Scorer(60)
    eF, eFp = sc.errors(sc.fbar[None, :] * (1e12 + 1.0), np.full((1, 60), 1e12))
    assert abs(eFp[0] - sc.floor_eFp) < 1e-9 and eF[0] < 0.01
