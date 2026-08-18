import numpy as np

from abpfr.metrics import (cosine_modes, integrated_error, l2_error_gauge,
                           paired_bootstrap_ci, relative_error_curve,
                           time_to_accuracy)


def test_l2_gauge_invariance():
    G = 101
    x = np.linspace(-1, 1, G)
    mask = np.abs(x) <= 0.8
    F_ref = x ** 2
    assert l2_error_gauge(F_ref + 3.7, F_ref, mask) < 1e-12
    e1 = l2_error_gauge(F_ref + 0.1 * np.sin(3 * x), F_ref, mask)
    e2 = l2_error_gauge(F_ref + 0.1 * np.sin(3 * x) - 42.0, F_ref, mask)
    assert abs(e1 - e2) < 1e-12
    # batched input
    batch = np.stack([F_ref + 1.0, F_ref + 0.5 * x])
    errs = l2_error_gauge(batch, F_ref, mask)
    assert errs.shape == (2,) and errs[0] < 1e-12 < errs[1]


def test_time_to_accuracy_persistence():
    t = np.linspace(0.0, 100.0, 401)
    # dips below 1 at t=10 but bounces back up at t=15; settles below from t=30
    err = np.where(t < 10, 2.0, np.where(t < 15, 0.5, np.where(t < 30, 1.5, 0.5)))
    tau = time_to_accuracy(t, err, eps=1.0, persist_frac=0.2)
    assert abs(tau - 30.0) < 0.5
    # censored: never below
    assert np.isnan(time_to_accuracy(t, np.full_like(t, 2.0), eps=1.0))


def test_integrated_error():
    t = np.linspace(0, 10, 11)
    err = np.ones_like(t)
    assert abs(integrated_error(t, err) - 10.0) < 1e-12


def test_relative_error_curve():
    r = relative_error_curve(np.array([1.0, 0.5]), np.array([2.0, 1.0]))
    assert np.allclose(r, [0.5, 0.5])


def test_cosine_modes_recover_pure_mode():
    x = np.linspace(-1.8, 1.8, 181)
    lo, hi = -1.5, 1.5
    m = (x >= lo) & (x <= hi)
    L = x[m][-1] - x[m][0]
    prof = 0.7 * np.cos(2 * np.pi * (x - x[m][0]) / L)
    a = cosine_modes(prof, x, lo, hi, k_max=3)
    assert abs(a[1] - 0.7) < 0.02          # k=2 coefficient
    assert abs(a[0]) < 0.02 and abs(a[2]) < 0.02


def test_paired_bootstrap_ci():
    rng = np.random.default_rng(0)
    v = rng.normal(-0.2, 0.05, 32)          # consistently negative differences
    med, lo, hi = paired_bootstrap_ci(v, n_boot=2000)
    assert lo < med < hi < 0.0
