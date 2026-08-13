"""Missing data must never read as a passed gate (the `isfinite`-as-skip defect).

The methane session found this shape in its own Gate A: a state with no samples produced a nan,
an `isfinite` guard *skipped* the gate, and an uncomputable gate read as a passed one.  Three
instances existed here.  Each is pinned below, because every one of them converts absent data
into a physics verdict that ENDS the study:

  Gate C   nan reference -> `P < 0.5 Q` is False -> "no deficit" -> **ABF-sufficient, STOP**
  Gate A   empty basin   -> zero histograms -> TV = 0 -> **CV-visibility failure, STOP**
  tau_perp censored point dropped -> tau_perp too small -> Gate D ceiling too PERMISSIVE
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- Gate C refuses a nan reference
def test_gate_c_refuses_a_non_finite_reference():
    g = _load("nacl_gates")
    n_cp, S, n_grid = 6, 8, 41
    grid = np.linspace(0.2, 1.4, n_grid)
    basins = [dict(label="CIP", r_lo_nm=0.2, r_hi_nm=0.4),
              dict(label="SSIP", r_lo_nm=0.4, r_hi_nm=1.4)]
    occ = np.ones((n_cp, S, n_grid))
    pmf = np.zeros((n_cp, S, n_grid))
    times = np.linspace(0, 1000.0, n_cp)

    F_ref = np.linspace(0.0, 10.0, n_grid)
    ok = g.gate_c(occ, pmf, times, grid, F_ref, basins, 0.4, 1000.0)
    assert set(ok) == {"CIP", "SSIP"}                    # finite input works

    F_bad = F_ref.copy()
    F_bad[7] = np.nan
    with pytest.raises(RuntimeError, match="NOT COMPUTABLE"):
        g.gate_c(occ, pmf, times, grid, F_bad, basins, 0.4, 1000.0)

    # and the failure mode it protects against: with the nan swallowed, no deficit is ever
    # flagged, which is exactly the "established -> ABF-sufficient" verdict
    with np.errstate(invalid="ignore"):
        w = np.exp(-0.4 * (F_bad - np.nanmin(F_bad)))
    assert not np.isfinite(w).all()
    assert not (0.1 < 0.5 * float(np.nan))               # the silent-pass comparison


def test_gate_c_refuses_a_non_finite_bias_trace():
    g = _load("nacl_gates")
    n_cp, S, n_grid = 4, 8, 21
    grid = np.linspace(0.2, 1.4, n_grid)
    basins = [dict(label="CIP", r_lo_nm=0.2, r_hi_nm=0.6)]
    pmf = np.zeros((n_cp, S, n_grid))
    pmf[2, 3, 5] = np.inf
    with pytest.raises(RuntimeError, match="NOT COMPUTABLE"):
        g.gate_c(np.ones((n_cp, S, n_grid)), pmf, np.linspace(0, 100, n_cp), grid,
                 np.linspace(0, 5, n_grid), basins, 0.4, 100.0)


# ---------------------------------------------------------------- Gate B counts a miss as a miss
def test_gate_b_treats_never_hit_as_not_discovered():
    g = _load("nacl_gates")
    n_frames, S, N = 40, 8, 4
    xi = np.full((n_frames, S, N), 0.25)                 # everyone parked in CIP forever
    steps = np.arange(n_frames) * 250
    basins = [dict(label="CIP", r_lo_nm=0.2, r_hi_nm=0.3),
              dict(label="SSIP", r_lo_nm=0.45, r_hi_nm=0.6)]
    out = g.gate_b(xi, steps, 0.002, basins, T_ps=1000.0)
    assert out["CIP"]["PASS"] is True
    assert out["SSIP"]["PASS"] is False                   # nan T_hit must not count as a hit
    assert all(np.isnan(t) for t in out["SSIP"]["T_hit_ps"])


# ---------------------------------------------------------------- tau_perp censoring
def test_censored_tau_enters_the_max_at_its_lower_bound():
    """A point that never decorrelates has tau > track; dropping it biases tau_perp DOWN and
    the Gate D ceiling 0.1/tau_perp UP, licensing a faster rate than the physics allows."""
    TRACK = 200.0
    measured = {"r0.30": {"n_NaO": 40.0, "n_ClH": 55.0},
                "r0.50": {"n_NaO": None, "n_ClH": 30.0}}     # one censored

    dropped = max(v for r in measured.values() for v in r.values() if v is not None)
    censored_aware = max((TRACK if v is None else v)
                         for r in measured.values() for v in r.values())
    assert dropped == 55.0
    assert censored_aware == TRACK
    assert 0.1 / censored_aware < 0.1 / dropped              # conservative, not permissive


def test_tau_perp_script_collects_censored_points():
    src = open(os.path.join(ROOT, "scripts", "nacl_tau_perp.py")).read()
    assert "TRACK_PS" in src and "censored" in src
    assert "vals.append(TRACK_PS)" in src, "censored points must enter the max at their bound"
    assert "gate_D_ceiling_is_conservative_bound" in src


# ---------------------------------------------------------------- Gate A computability
def test_gate_a_not_computable_is_not_a_failure():
    """An empty basin gives TV = 0 against everything, which reads as a CV-visibility STOP."""
    empty = np.zeros(20)
    other = np.zeros(20); other[3] = 1.0
    tv = 0.5 * float(np.abs(empty / max(empty.sum(), 1) - other / other.sum()).sum())
    assert tv == pytest.approx(0.5)          # with the max(...,1) guard the empty side is zeros
    # the analysis must therefore gate on sample counts, not on the TV value
    src = open(os.path.join(ROOT, "scripts", "nacl_ti_analyze.py")).read()
    assert "MIN_SAMPLES_PER_BASIN" in src
    assert "COMPUTABLE=bool(gateA_computable)" in src
    consumer = open(os.path.join(ROOT, "scripts", "nacl_gates.py")).read()
    assert 'gA.get("COMPUTABLE") is False' in consumer, "the consumer must branch on it"


# ---------------------------------------------------------------- numerical robustness of Q*
@pytest.mark.parametrize("bias_kind", ["zero", "plus2000", "minus2000", "ramp5000", "equal_F"])
def test_bias_aware_target_survives_extreme_biases(bias_kind):
    """The nan can also arrive through arithmetic from perfectly finite inputs: an unstabilised
    exp(-beta (F_ref - B_t)) overflows or underflows to all-zeros and the 0/0 reproduces exactly
    the silent 'no deficit' this suite exists to prevent.  The exponent is stabilised by
    subtracting its minimum, and this pins that it stays so."""
    g = _load("nacl_gates")
    n_grid, n_cp, S = 41, 4, 2
    grid = np.linspace(0.2, 1.4, n_grid)
    F_ref = 50.0 * np.sin(np.linspace(0, 3, n_grid))
    bias = {"zero": np.zeros(n_grid),
            "plus2000": np.full(n_grid, 2000.0),
            "minus2000": np.full(n_grid, -2000.0),
            "ramp5000": np.linspace(-5000.0, 5000.0, n_grid),
            "equal_F": F_ref.copy()}[bias_kind]
    basins = [dict(label="CIP", r_lo_nm=0.2, r_hi_nm=0.5),
              dict(label="SSIP", r_lo_nm=0.5, r_hi_nm=1.4)]
    pmf = np.broadcast_to(bias, (n_cp, S, n_grid)).copy()
    out = g.gate_c(np.ones((n_cp, S, n_grid)), pmf, np.linspace(0, 1000, n_cp), grid,
                   F_ref, basins, 0.40091, 1000.0)
    assert set(out) == {"CIP", "SSIP"}
    for v in out.values():
        assert all(np.isfinite(v["longest_deficit_ps"]))

    x = F_ref - bias
    w = np.exp(-0.40091 * (x - x.min()))
    assert np.isfinite(w).all() and w.sum() >= 1.0        # at least the argmin contributes 1


def test_basin_masks_partition_the_grid():
    """Closed-on-both-ends masks double-count the shared boundary bin; the targets then sum to
    more than one (measured 1.024 on a 41-point grid before the fix)."""
    g = _load("nacl_gates")
    grid = np.linspace(0.2, 1.4, 41)
    basins = [dict(label="CIP", r_lo_nm=0.2, r_hi_nm=0.5),
              dict(label="SSIP", r_lo_nm=0.5, r_hi_nm=0.9),
              dict(label="outer", r_lo_nm=0.9, r_hi_nm=1.4)]
    masks = g.basin_masks(grid, basins)
    stacked = np.stack([masks[b["label"]] for b in basins])
    assert (stacked.sum(axis=0) == 1).all(), "masks must partition: no bin in two basins, none lost"

    w = np.ones(len(grid))
    Q = [float(w[masks[b["label"]]].sum() / w.sum()) for b in basins]
    assert sum(Q) == pytest.approx(1.0, abs=1e-12)


def test_incomplete_reference_cannot_be_accepted():
    src = open(os.path.join(ROOT, "scripts", "nacl_ti_analyze.py")).read()
    assert "ACCEPTED=bool(accepted and complete)" in src
