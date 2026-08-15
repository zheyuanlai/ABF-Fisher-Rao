"""The windowed power audit that decides whether CIP is CLEAR or merely UNKNOWN.

The gate's own statistic is unpowered at CIP (lambda 1.57), so the N=64 verdict clears CIP with
a windowed average instead. These tests pin that the window scan is complete and that it can
detect the thing it claims to exclude -- a guard exercised only on data that passes is
indistinguishable from a guard that always passes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import nacl_audit_cip_power as ap
import nacl_gates as ng

N_GRID, S, N_CP, T_PS = 10, 8, 101, 1000.0
TOTAL = 100                      # walker-count normalisation per checkpoint


def _inputs(mask_counts, seed=0):
    """mask_counts(c) -> counts placed on the 2 masked grid points at checkpoint c."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.2, 1.4, N_GRID)
    times = np.linspace(0.0, T_PS, N_CP)
    F_ref = np.zeros(N_GRID)
    diag_pmf = np.zeros((N_CP, S, N_GRID))            # B_t = 0 -> Q = masked width fraction
    msk = np.zeros(N_GRID, bool)
    msk[:2] = True
    occ = np.zeros((N_CP, S, N_GRID))
    for c in range(N_CP):
        for s in range(S):
            m = max(mask_counts(times[c]) + rng.normal(0, 0.3), 0.0)
            occ[c, s, msk] = m / 2.0
            occ[c, s, ~msk] = (TOTAL - m) / (N_GRID - 2)
    return occ, diag_pmf, times, grid, F_ref, msk


def _run(mask_counts, seed=0):
    occ, pmf, times, grid, F_ref, msk = _inputs(mask_counts, seed)
    wins, _, _ = ap.windowed_ratios(occ, pmf, times, grid, F_ref, msk,
                                    beta=1.0, T_ps=T_PS)
    return wins


def test_every_window_is_scanned_not_just_the_one_that_lands_exactly():
    """Regression. Requiring the span to reach `need` using only checkpoints INSIDE the window
    is short by up to one checkpoint spacing, which silently discarded 46 of 47 windows on the
    real data and still reported a pass -- on the single most favourable window."""
    wins = _run(lambda t: 20.0)
    # need = 0.20 * 1000 = 200 ps, spacing 10 ps, second half starts at 500 ps.
    # windows start at every checkpoint t0 in [500, 800] -> 31 of them.
    assert len(wins) == 31, f"scanned {len(wins)} windows, expected 31"
    assert all(w["t1"] - w["t0"] >= ng.DEFICIT_FRACTION * T_PS - 1e-9 for w in wins)


def test_positive_control_a_planted_sustained_deficit_is_detected():
    """Q = 0.2; plant P = 0.04 (ratio 0.2) for 300 ps > the 200 ps required duration."""
    wins = _run(lambda t: 4.0 if 600.0 <= t <= 900.0 else 20.0)
    v = ap.verdict_from_windows(wins, ng.DEFICIT_RATIO)
    assert v["SUSTAINED_DEFICIT"], f"planted deficit missed: worst hi = {v['hi']:.3f}"
    assert not v["ESTABLISHED_WITH_POWER"]


def test_negative_control_a_tracking_state_is_cleared():
    wins = _run(lambda t: 20.0)
    v = ap.verdict_from_windows(wins, ng.DEFICIT_RATIO)
    assert v["ESTABLISHED_WITH_POWER"], f"tracking state not cleared: worst lo = {v['lo']:.3f}"
    assert not v["SUSTAINED_DEFICIT"]
    assert abs(v["ratio"] - 1.0) < 0.05


def test_a_deficit_shorter_than_the_required_duration_does_not_clear_or_condemn():
    """A 50 ps dropout inside a 200 ps window is diluted -- which is the POINT of the gate's
    contiguity requirement, and the reason the windowed statistic is the right replacement."""
    wins = _run(lambda t: 4.0 if 700.0 <= t <= 750.0 else 20.0)
    v = ap.verdict_from_windows(wins, ng.DEFICIT_RATIO)
    assert not v["SUSTAINED_DEFICIT"], "a sub-duration dropout must not read as a deficit"


def test_verdict_is_driven_by_the_worst_window_not_the_last():
    """The real bug shipped a number from the final window (1.597) while the worst was 1.111."""
    wins = _run(lambda t: 8.0 if 600.0 <= t <= 900.0 else 40.0)
    v = ap.verdict_from_windows(wins, ng.DEFICIT_RATIO)
    all_ratios = [float(w["ratio_per_seed"].mean()) for w in wins]
    assert v["ratio"] <= min(all_ratios) + 1e-9 or v["hi"] <= min(
        m + 2 * float(w["ratio_per_seed"].std(ddof=1) / np.sqrt(S))
        for m, w in zip(all_ratios, wins)) + 1e-9
