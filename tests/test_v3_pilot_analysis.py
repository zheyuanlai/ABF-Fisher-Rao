"""Gates for the v3 pilot analysis, especially the Amendment 6a censoring rule.

tau_eps is a primary endpoint.  A silent off-by-one in the persistence window,
or a censoring convention that quietly drops pairs, would produce a wrong
headline that no downstream check could catch -- so the convention is pinned
here with adversarial cases before it produces any number.
"""
import importlib.util
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "analyze_v3_pilot",
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "analyze_v3_pilot.py")
ap = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ap)

N_FRAMES = 101          # indices 0..100


def _series(values):
    return np.asarray(values, dtype=float)


def test_never_reaching_threshold_is_censored_not_dropped():
    err = _series(np.full(N_FRAMES, 10.0))
    tau, hit = ap.tau_eps(err, eps=1.0)
    assert tau == ap.TAU_CENSOR == 96
    assert hit is False


def test_immediately_accurate_gives_tau_zero():
    tau, hit = ap.tau_eps(_series(np.zeros(N_FRAMES)), eps=1.0)
    assert (tau, hit) == (0, True)


def test_a_dip_without_persistence_does_not_count():
    """One frame under the threshold is not sustained accuracy."""
    err = np.full(N_FRAMES, 10.0)
    err[40] = 0.1                       # single dip
    tau, hit = ap.tau_eps(_series(err), eps=1.0)
    assert (tau, hit) == (ap.TAU_CENSOR, False)

    err[40:45] = 0.1                    # 5 frames: still one short of [n, n+5]
    assert ap.tau_eps(_series(err), 1.0) == (ap.TAU_CENSOR, False)

    err[40:46] = 0.1                    # 6 frames = [n, n+5] inclusive
    assert ap.tau_eps(_series(err), 1.0) == (40, True)


def test_latest_valid_start_is_index_95():
    """A run that only becomes accurate at the very end still counts."""
    err = np.full(N_FRAMES, 10.0)
    err[95:] = 0.1                      # frames 95..100 = 6 frames
    assert ap.tau_eps(_series(err), 1.0) == (95, True)

    err = np.full(N_FRAMES, 10.0)
    err[96:] = 0.1                      # only 5 frames; window cannot close
    assert ap.tau_eps(_series(err), 1.0) == (ap.TAU_CENSOR, False)
    assert ap.TAU_CENSOR > 95


def _taus(pairs):
    return pd.DataFrame([dict(seed=i, tau=t, hit=h)
                         for i, (t, h) in enumerate(pairs)])


def test_speedup_handles_all_four_censoring_cases():
    hit_early = (10, True)
    hit_late = (20, True)
    cens = (ap.TAU_CENSOR, False)

    # both hit: ordinary comparison
    s = ap.speedup(_taus([hit_late] * 8), _taus([hit_early] * 8))
    assert s["median_S"] == pytest.approx(2.0) and s["favorable"] == 8

    # reference censored, method hits -> S > 1
    s = ap.speedup(_taus([cens] * 8), _taus([hit_early] * 8))
    assert s["median_S"] > 1.0
    assert s["censored_ref"] == 8 and s["censored_arm"] == 0

    # reference hits, method censored -> S < 1
    s = ap.speedup(_taus([hit_early] * 8), _taus([cens] * 8))
    assert s["median_S"] < 1.0 and s["censored_arm"] == 8

    # both censored -> S == 1, explicitly flagged
    s = ap.speedup(_taus([cens] * 8), _taus([cens] * 8))
    assert s["median_S"] == pytest.approx(1.0)
    assert s["both_censored"] == 8


def test_speedup_refuses_an_unmatched_seed_set():
    """Never silently compare on whichever seeds happen to be present."""
    with pytest.raises(ValueError):
        ap.speedup(_taus([(10, True)] * 8), _taus([(10, True)] * 6))


def test_censored_pairs_are_never_dropped():
    """A censored arm must still contribute a pair, not vanish from the median."""
    ref = _taus([(10, True)] * 8)
    arm = _taus([(10, True)] * 4 + [(ap.TAU_CENSOR, False)] * 4)
    s = ap.speedup(ref, arm)
    assert s["censored_arm"] == 4
    # 4 pairs at 1.0 and 4 at 10/96; dropping the censored half would give 1.0
    assert s["median_S"] < 1.0


def test_dose_decay_needs_a_real_window_and_returns_nan_otherwise():
    steps = np.arange(0, 50500, 500)
    rows = []
    for seed in range(2):
        cum = np.zeros(len(steps))
        rows += [dict(seed=seed, step=int(s), cumulative_replacements=int(c))
                 for s, c in zip(steps, cum)]
    assert np.isnan(ap.dose_decay(pd.DataFrame(rows)))


def test_dose_decay_measures_first_quarter_against_last():
    steps = np.arange(0, 50500, 500)
    rows = []
    for seed in range(2):
        per = np.zeros(len(steps) - 1)
        per[20:81] = np.concatenate([np.full(31, 20.0), np.full(30, 2.0)])
        cum = np.concatenate([[0], np.cumsum(per)])
        rows += [dict(seed=seed, step=int(s), cumulative_replacements=int(c))
                 for s, c in zip(steps, cum)]
    ratio = ap.dose_decay(pd.DataFrame(rows))
    assert ratio == pytest.approx(10.0, rel=0.2)      # 20 -> 2
