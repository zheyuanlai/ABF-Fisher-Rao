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
