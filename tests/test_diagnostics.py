import numpy as np

from abpfr.diagnostics import (establishment_time, first_persistent, hit_time,
                               kde_noise_floor)
from abpfr.grid import Grid1D

G = Grid1D(xmin=-1.8, xmax=1.8, n=181, eval_lo=-1.5, eval_hi=1.5)


def test_first_persistent():
    t = np.linspace(0, 10, 101)
    cond = t >= 4.0
    assert abs(first_persistent(cond, t) - 4.0) < 0.11
    # a single blip must not count
    blip = np.zeros_like(t, dtype=bool)
    blip[20] = True
    assert np.isnan(first_persistent(blip, t, hold_frac=0.05))


def test_hit_and_establishment_wrappers():
    t = np.linspace(0, 10, 101)
    occ = np.where(t >= 2.0, 0.1, 0.0)
    assert abs(hit_time(occ, t) - 2.0) < 0.11
    D = np.where(t >= 6.0, 0.001, 1.0)
    assert abs(establishment_time(D, t, D_tol=0.01) - 6.0) < 0.11


def test_kde_noise_floor_scales_down_with_K():
    d_small = kde_noise_floor(256, 0.10, G, n_rep=64, seed=1)
    d_big = kde_noise_floor(4096, 0.10, G, n_rep=64, seed=2)
    assert np.median(d_big) < np.median(d_small)
    assert (d_small > 0).all()
