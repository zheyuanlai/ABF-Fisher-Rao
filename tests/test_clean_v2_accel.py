"""Adversarial cases for the clean-v2 time-to-accuracy statistics.

Every number this campaign reports is a ratio of hitting times, and this
project's defect log says ratios have the worst record: check both arguments'
populations, and pin the censoring rule *before* it produces a number.  So these
cases are written against the rule, not against a run.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from abffr import accel

T = np.arange(0.0, 11.0)          # 11 saved frames, horizon 10


def _hit(e, eps=1.0, horizon=10.0, k=3):
    return accel.restricted_hitting_time(T, np.asarray(e, float), eps,
                                         horizon, consecutive=k)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_a_single_dip_is_not_convergence():
    e = [9, 8, 0.1, 8, 8, 0.1, 0.1, 8, 0.1, 0.1, 0.1]
    assert accel.hitting_time(T, e, 1.0, consecutive=3) == 8.0
    assert accel.hitting_time(T, e, 1.0, consecutive=1) == 2.0


def test_tau_is_the_first_frame_of_the_qualifying_run():
    e = [9, 9, 9, 0.5, 0.5, 0.5, 9, 9, 9, 9, 9]
    assert accel.hitting_time(T, e, 1.0, consecutive=3) == 3.0


def test_threshold_reached_only_at_the_very_end_is_censored():
    """A run of 3 cannot start in the last two frames.  Stating this is the
    point: it is a property of the rule, not an accident of a dataset."""
    e = [9] * 9 + [0.1, 0.1]
    assert accel.hitting_time(T, e, 1.0, consecutive=3) == accel.INF
    assert _hit(e).censored and _hit(e).restricted == 10.0


def test_non_finite_frames_break_a_run_rather_than_extend_it():
    e = [9, 0.1, np.nan, 0.1, 0.1, 0.1, 9, 9, 9, 9, 9]
    assert accel.hitting_time(T, e, 1.0, consecutive=3) == 3.0


def test_exactly_at_the_threshold_counts_as_reached():
    e = [9, 9, 1.0, 1.0, 1.0, 9, 9, 9, 9, 9, 9]
    assert accel.hitting_time(T, e, 1.0, consecutive=3) == 2.0
    assert accel.hitting_time(T, e, 0.999999, consecutive=3) == accel.INF


# --------------------------------------------------------------------------- #
# Censoring: not conservative, and the direction depends on which side is cut
# --------------------------------------------------------------------------- #
def test_censored_seeds_are_restricted_not_dropped():
    """Dropping them would compare converged-ABF against converged-FR."""
    fast = [_hit([0.1] * 11) for _ in range(4)]
    never = [_hit([9] * 11) for _ in range(4)]
    s = accel.speedup(never, fast)
    assert s.n_base == 4 and s.n_arm == 4
    assert s.n_censored_base == 4 and s.n_censored_arm == 0
    assert s.hit_fraction_base == 0.0 and s.hit_fraction_arm == 1.0


def test_restriction_INFLATES_the_ratio_when_only_the_arm_is_censored():
    """The direction that matters, and the one easiest to state backwards.

    Restriction replaces a censored ``tau`` by ``T``, the *smallest* value it
    could have had.  Doing that to the arm shrinks the denominator, so S^(T)
    comes out **larger** than the unrestricted speedup -- censoring flatters the
    arm.  Getting this backwards would license reporting an inflated headline
    number as a conservative one.
    """
    base = [_hit([9] * 6 + [0.1] * 5) for _ in range(4)]     # tau = 6, uncensored
    arm = [_hit([9] * 11) for _ in range(4)]                 # censored at T = 10
    s = accel.speedup(base, arm)
    assert s.n_censored_arm == 4 and s.n_censored_base == 0
    assert s.s == pytest.approx(6.0 / 10.0)
    unrestricted_if_arm_hit_at_50 = 6.0 / 50.0
    assert s.s > unrestricted_if_arm_hit_at_50      # inflated, not conservative
    assert s.censoring_inflates


def test_restriction_DEFLATES_the_ratio_when_only_the_baseline_is_censored():
    base = [_hit([9] * 11) for _ in range(4)]                # censored at T = 10
    arm = [_hit([9] * 4 + [0.1] * 7) for _ in range(4)]      # tau = 4, uncensored
    s = accel.speedup(base, arm)
    assert s.n_censored_base == 4 and s.n_censored_arm == 0
    assert s.s == pytest.approx(10.0 / 4.0)
    unrestricted_if_base_hit_at_50 = 50.0 / 4.0
    assert s.s < unrestricted_if_base_hit_at_50     # deflated
    assert not s.censoring_inflates


def test_no_arm_censoring_makes_the_restricted_speedup_a_lower_bound():
    """The one safe case, and the only one a headline claim may rest on."""
    base = [_hit([9] * 11)] * 2 + [_hit([9] * 6 + [0.1] * 5)] * 2
    arm = [_hit([9] * 3 + [0.1] * 8) for _ in range(4)]      # tau = 3, all hit
    s = accel.speedup(base, arm)
    assert s.n_censored_arm == 0 and s.n_censored_base == 2
    assert not s.censoring_inflates
    assert s.s == pytest.approx(((10 + 10 + 6 + 6) / 4) / 3.0)


def test_speedup_carries_both_counts_and_both_hit_fractions():
    row = accel.speedup([_hit([0.1] * 11)] * 3,
                        [_hit([9] * 11)] * 5).to_row()
    assert row["n_base"] == 3 and row["n_arm"] == 5
    assert row["hit_fraction_base"] == 1.0 and row["hit_fraction_arm"] == 0.0
    assert row["censoring_inflates"] is True
    assert set(("n_censored_base", "n_censored_arm")) <= set(row)


def test_confirms_refuses_a_threshold_where_the_arm_is_more_censored():
    """S^(T) is inflated exactly there, so it cannot carry a verdict."""
    def _sp(s_val, cens_arm, cens_base=0):
        return accel.Speedup(s=s_val, mean_base=1.0, mean_arm=1.0, n_base=32,
                             n_arm=32, n_censored_base=cens_base,
                             n_censored_arm=cens_arm, ci_lo=1.2, ci_hi=1.9,
                             n_boot=10)
    clean = [_sp(1.5, 0), _sp(1.4, 0)]
    good_fp = [_sp(1.1, 0), _sp(1.2, 0)]
    assert accel.confirms(clean, good_fp)
    tainted = [_sp(1.5, 0), _sp(1.4, cens_arm=5, cens_base=2)]
    assert not accel.confirms(tainted, good_fp)
    # ... but more *baseline* censoring is fine: that direction deflates S.
    ok = [_sp(1.5, 0), _sp(1.4, cens_arm=2, cens_base=5)]
    assert accel.confirms(ok, good_fp)


# --------------------------------------------------------------------------- #
# The bootstrap is paired, and says so
# --------------------------------------------------------------------------- #
def test_paired_bootstrap_refuses_unmatched_seed_counts():
    with pytest.raises(ValueError, match="matched seeds"):
        accel.paired_bootstrap_speedup([_hit([0.1] * 11)] * 3,
                                       [_hit([0.1] * 11)] * 4)


def test_paired_bootstrap_ci_brackets_a_real_speedup():
    rng = np.random.default_rng(0)
    base, arm = [], []
    for _ in range(32):
        tb = rng.uniform(6.0, 8.0)
        ta = tb / 1.5
        base.append(accel.Hit(tb, tb, False))
        arm.append(accel.Hit(ta, ta, False))
    s = accel.paired_bootstrap_speedup(base, arm, n_boot=2000, seed=1)
    assert s.s == pytest.approx(1.5, rel=1e-6)
    assert s.ci_lo < 1.5 < s.ci_hi and s.excludes_one

    # Null control: identical arms must NOT exclude one.
    null = accel.paired_bootstrap_speedup(base, base, n_boot=2000, seed=1)
    assert null.s == pytest.approx(1.0) and not null.excludes_one


# --------------------------------------------------------------------------- #
# Threshold freezing
# --------------------------------------------------------------------------- #
def test_thresholds_are_the_median_of_the_saved_frame_nearest_the_fraction():
    curves = [(T, np.linspace(10, 0, 11) + shift) for shift in (0.0, 1.0, 2.0)]
    eps = accel.freeze_thresholds(curves, [0.4, 0.6], horizon=10.0)
    # frame nearest 4.0 is index 4 (value 6 + shift); median shift is 1.0
    assert eps[0] == pytest.approx(7.0)
    assert eps[1] == pytest.approx(5.0)
    assert eps[1] < eps[0]          # a later fraction is the stringent one


def test_freezing_refuses_to_invent_a_threshold_from_no_data():
    with pytest.raises(ValueError, match="non-finite"):
        accel.freeze_thresholds([(T, np.full(11, np.nan))], [0.4], 10.0)
    with pytest.raises(ValueError, match="at least one"):
        accel.freeze_thresholds([], [0.4], 10.0)


def test_accel_cost_is_undefined_without_turnover():
    assert np.isnan(accel.accel_cost(1.5, 0.0))
    assert accel.accel_cost(1.5, 0.25) == pytest.approx(2.0)


def _spd(s_val, cens_arm=0, cens_base=0, n=8):
    return accel.Speedup(s=s_val, mean_base=1.0, mean_arm=1.0, n_base=n,
                         n_arm=n, n_censored_base=cens_base,
                         n_censored_arm=cens_arm)


# --------------------------------------------------------------------------- #
# The Stage-2 screen must see censoring too -- selection filters on it
# --------------------------------------------------------------------------- #
def test_pilot_screen_refuses_a_cell_inflated_by_arm_censoring():
    """Selection filters on ``promising``, so a screen blind to censoring sends
    an inflated cell to 96 fresh runs and only finds out at the verdict."""
    fp = [_spd(1.2), _spd(1.2)]
    assert accel.pilot_promising([_spd(1.6), _spd(1.5)], fp)
    # Same ratios, but the arm failed to converge more often than the baseline.
    tainted = [_spd(1.6), _spd(1.5, cens_arm=3, cens_base=1)]
    assert not accel.pilot_promising(tainted, fp)
    # More BASELINE censoring deflates S, so it is not disqualifying.
    assert accel.pilot_promising([_spd(1.6), _spd(1.5, cens_arm=1, cens_base=3)],
                                 fp)


def test_pilot_screen_and_verdict_apply_the_same_censoring_rule():
    """Two predicates that disagree about censoring is a latent selection bug."""
    inflated = [_spd(1.6, cens_arm=4, cens_base=0),
                _spd(1.5, cens_arm=4, cens_base=0)]
    fp = [_spd(1.2), _spd(1.2)]
    assert not accel.pilot_promising(inflated, fp)
    for v in inflated:
        v.ci_lo, v.ci_hi, v.n_boot = 1.3, 1.9, 10
    assert not accel.confirms(inflated, fp)


def test_pilot_screen_still_enforces_the_numeric_thresholds():
    fp = [_spd(1.2), _spd(1.2)]
    assert not accel.pilot_promising([_spd(1.6), _spd(1.1)], fp)     # S_F,2 low
    assert not accel.pilot_promising([_spd(1.6), _spd(1.5)],
                                     [_spd(1.05), _spd(1.05)])        # F' weak
    assert not accel.pilot_promising([_spd(1.6), _spd(1.5)],
                                     [_spd(1.2), _spd(0.9)])          # slowdown
