"""Positive / negative controls for the ERI classifier statistics (Amendment 2; audit finding F11)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from eri import metrics as mt  # noqa: E402


def _occupancy(N, Q, T=40.0, ds=0.1, deficit=0.0, span=(0, 0), seed=0, rho=0.8):
    """Window fraction time series: AR(1)-correlated binomial-like noise around (1 - deficit in span) * Q."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, T + 1e-9, ds)
    mean = np.full_like(t, Q)
    mean[(t >= span[0]) & (t < span[1])] *= (1 - deficit)
    sd = np.sqrt(Q * (1 - Q) / N)
    e = np.zeros_like(t)
    for i in range(1, len(t)):
        e[i] = rho * e[i - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
    return t, np.clip(mean + sd * e, 0, 1), np.full_like(t, Q)


def test_establishment_blocks_null_has_no_false_positives():
    for N in (64, 256, 1024):
        hits = 0
        for s in range(200):
            t, f, Q = _occupancy(N, 0.157, seed=s)
            _, _, span, _ = mt.establishment_blocks(t, f, Q, N)
            hits += span >= 8.0
        assert hits == 0, (N, hits)


def test_establishment_blocks_detects_a_real_deficit():
    # measured power (Amendment 2): a 12 t.u. deficit of 40 % (N 1024), 50 % (N 256), 80 % (N 64) is detected
    # (>= 8 t.u. of consecutive deficient 2 t.u. blocks) in >= 90 % of synthetic runs
    for N, d in ((1024, 0.4), (256, 0.5), (64, 0.8)):
        det = 0
        for s in range(50):
            t, f, Q = _occupancy(N, 0.157, deficit=d, span=(4.0, 16.0), seed=100 + s)
            _, _, span, _ = mt.establishment_blocks(t, f, Q, N)
            det += span >= 8.0
        assert det >= 45, (N, d, det)


def test_old_per_save_rule_is_vacuous_at_N64():
    t, f, Q = _occupancy(64, 0.157, deficit=0.9, span=(2.0, 40.0), seed=1)
    _, _, span = mt.establishment(t, f, Q, 64, t_from=2.0)
    # the A1.4 rule's threshold 0.5 N Q - 2 sd is negative at N = 64, Q = 0.157: even a 90 % deficit is invisible
    assert span == 0.0


def test_conditional_lag_timeavg_null_and_positive():
    rng = np.random.default_rng(3)
    ns, nb, N = 401, 18, 256
    ref = {"eta": rng.uniform(-0.5, 0.9, nb), "ux2": rng.uniform(0.3, 0.6, nb)}
    var = {"eta": np.full(nb, 0.4), "ux2": np.full(nb, 0.05)}

    def run(lag):
        sv = np.zeros((ns, nb, 3))
        n = rng.multinomial(N, np.ones(nb) / nb, size=ns).astype(float)
        sv[:, :, 0] = n
        sv[:, :, 1] = n * (ref["eta"] + lag) + rng.standard_normal((ns, nb)) * np.sqrt(n * 0.4)
        sv[:, :, 2] = n * ref["ux2"] + rng.standard_normal((ns, nb)) * np.sqrt(n * 0.05)
        return {"sv_cond": sv, "times": np.arange(ns) * 0.1}
    s0, f0 = mt.conditional_lag_timeavg(run(0.0), ref, var)["eta"]
    s1, f1 = mt.conditional_lag_timeavg(run(0.15), ref, var)["eta"]
    assert s0 < 3 * f0
    assert s1 > 3 * f1


def test_tv_floor_matches_simulation():
    rng = np.random.default_rng(4)
    Q = np.array([0.5, 0.3, 0.2])
    sims = [0.5 * np.abs(rng.multinomial(50, Q) / 50 - Q).sum() for _ in range(20000)]
    assert abs(np.mean(sims) - mt.tv_floor(50, Q)) < 0.003
