"""Gate C is basin-INTEGRATED, so a within-basin redistribution is invisible to it.

These pin the fine-scale check: it must DETECT a jam (walkers piled at a wall while the basin
integral stays ~1), and it must not manufacture a finding out of a region with no target mass --
the failure the first version of this script actually produced (P/Q = 1274 from 0.0175 % of
walkers against a target 1000x smaller, while the basin ratio was unchanged to four decimals).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import nacl_audit_within_basin as awb

GRID = np.linspace(0.2, 1.4, 121)
MSK = (GRID >= 0.34)


def _q(shape):
    q = np.zeros_like(GRID); q[MSK] = shape; return q / q.sum()


def test_positive_control_a_jam_at_the_wall_is_detected():
    """Integral preserved, shape destroyed: all walkers in the outer 10 % of the basin."""
    target = _q(np.ones(MSK.sum()))
    occ = np.zeros_like(GRID); idx = np.flatnonzero(MSK)
    occ[idx[-len(idx)//10:]] = 1.0; occ /= occ.sum()
    a = awb.audit_basin(GRID, occ, target, MSK)
    assert abs(a["integrated_ratio"] - 1.0) < 1e-6, "the integral must be preserved by construction"
    assert a["shape_TV"] > 0.5, f"a jam must show a large shape TV, got {a['shape_TV']:.3f}"
    assert a["outer_edge_excess"] > 3.0, "walkers piled at the wall must show an edge excess"


def test_negative_control_a_tracking_basin_is_clean():
    target = _q(np.linspace(1.0, 2.0, MSK.sum()))
    a = awb.audit_basin(GRID, target.copy(), target, MSK)
    assert a["shape_TV"] < 1e-9 and abs(a["outer_edge_excess"] - 1.0) < 1e-9


def test_ratio_is_suppressed_where_the_target_carries_no_mass():
    """A ratio needs BOTH arguments' populations. Give the inner quarter ~zero target and a
    tiny occupancy: the ratio explodes and means nothing."""
    # Plant the empty region using the SAME split the code uses. Slicing by len//4 instead
    # leaves one full-target point inside q1 (array_split gives 27/27/27/26, not 26/26/26/26),
    # which legitimately clears the threshold -- a fixture that disagrees with the code it tests.
    idx = np.flatnonzero(MSK)
    q1_idx = np.array_split(idx, 4)[0]
    shape = np.ones(MSK.sum())
    shape[np.searchsorted(idx, q1_idx)] = 1e-9
    target = _q(shape)
    occ = target.copy(); occ[q1_idx[:5]] += 1e-4; occ /= occ.sum()
    a = awb.audit_basin(GRID, occ, target, MSK)
    q1 = a["quarters"][0]
    assert not q1["ratio_meaningful"] and q1["ratio"] is None, "must not report a ratio here"
    assert q1["note"] and "division by ~0" in q1["note"]
    assert a["quarters"][-1]["ratio_meaningful"], "quarters WITH target mass must still report"


# --- the sensitivity instrument: the guard is STRUCTURAL, not the scheme-agreement ----------
import nacl_gate_c_sensitivity as sens


def test_planting_scales_the_tested_state_by_exactly_f():
    """The load-bearing property, and the one the broken version violated.

    Gate C reads state k's SHARE, counts_in_k / total. If planting preserves the per-checkpoint
    total then P_k -> f * P_k exactly and NO redistribution scheme can move the tested state --
    which is why agreement between schemes is not independent evidence and must not be quoted as
    corroboration. The original bug dropped mass where no walkers sat outside the basin, so the
    total shrank, and the basin's share rose to 1.0: the opposite of a deficit.
    """
    rng = np.random.default_rng(0)
    occ = rng.random((12, 8, 121)) * 10.0
    occ[:, :, ~MSK] *= 0.05                      # most walkers inside, the reachable case
    occ[3, 2, ~MSK] = 0.0                        # a checkpoint with NOTHING outside at all
    for f in (0.5, 0.45, 0.1):
        for wts in (None, np.broadcast_to(np.ones(121) / 121.0, occ.shape)):
            out = sens.plant(occ, MSK, f, out_weights=wts)
            p_new = out[:, :, MSK].sum(2) / out.sum(2)
            p_old = occ[:, :, MSK].sum(2) / occ.sum(2)
            assert np.abs(p_new - f * p_old).max() < 1e-12, "the tested state must scale by f"
            assert np.abs(out.sum(2) - occ.sum(2)).max() < 1e-9, "totals must be preserved"


def test_planting_refuses_if_it_would_break_the_total():
    """The assertion that was missing. Without it the broken redistribution shipped a table."""
    occ = np.ones((4, 8, 121))
    bad = np.zeros_like(occ)                     # weights that carry no mass anywhere
    try:
        sens.plant(occ, MSK, 0.5, out_weights=bad)
    except RuntimeError as e:
        assert "total" in str(e)
    else:
        raise AssertionError("plant must refuse when the per-checkpoint total is not preserved")
