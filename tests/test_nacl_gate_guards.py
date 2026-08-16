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
    n_grid, n_cp, S = 41, 4, 8      # full preregistered block: gate_c now refuses fewer
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


def test_partition_holds_on_sampled_walker_values_not_only_on_the_grid():
    """The grid and the walkers are different populations.

    Grid points sit inside the domain by construction; walkers reach past the soft walls.  A
    partition assertion that only ever sees the grid can pass while the same masks are wrong on
    real trajectories one line away -- so the assertion runs on the sampled values.
    """
    g = _load("nacl_gates")
    basins = [dict(label="CIP", r_lo_nm=0.20, r_hi_nm=0.35),
              dict(label="SSIP", r_lo_nm=0.35, r_hi_nm=0.60),
              dict(label="outer", r_lo_nm=0.60, r_hi_nm=1.40)]

    # exactly-on-boundary values are the ones a closed-everywhere mask double counts
    on_edges = np.array([0.20, 0.35, 0.60, 1.40])
    assert g.assert_partition(on_edges, basins, "edges") == 0.0

    # walkers past the soft walls belong to no basin, and that fraction is REPORTED
    with_excursions = np.array([0.18, 0.25, 0.42, 0.95, 1.44])
    outside = g.assert_partition(with_excursions, basins, "excursions")
    assert outside == pytest.approx(2 / 5)

    # a genuinely overlapping definition must raise rather than double count
    overlapping = [dict(label="A", r_lo_nm=0.20, r_hi_nm=0.60),
                   dict(label="B", r_lo_nm=0.40, r_hi_nm=1.40)]
    with pytest.raises(RuntimeError, match="not a partition"):
        g.assert_partition(np.array([0.5]), overlapping, "overlap")


def test_gate_b_does_not_credit_a_boundary_walker_to_two_states():
    g = _load("nacl_gates")
    n_frames, S, N = 30, 8, 2
    basins = [dict(label="CIP", r_lo_nm=0.20, r_hi_nm=0.40),
              dict(label="SSIP", r_lo_nm=0.40, r_hi_nm=1.40)]
    xi = np.full((n_frames, S, N), 0.40)          # every walker exactly on the shared boundary
    steps = np.arange(n_frames) * 250
    out = g.gate_b(xi, steps, 0.002, basins, T_ps=1000.0)
    assert out["CIP"]["PASS"] is False            # 0.40 belongs to SSIP alone, half-open [lo,hi)
    assert out["SSIP"]["PASS"] is True
    assert out["_diagnostics"]["fraction_outside_all_basins"] == 0.0


def test_occupancy_and_target_share_a_support_when_walkers_leave_the_domain():
    """The methane session's Gate C bug: walkers outside the domain were dropped from every
    basin while Q* stayed normalised over the whole grid, so occupancy was compared against a
    full-weight target, `P < 0.5 Q` fired too easily, and the verdict was biased toward
    **establishment-limited** -- the one direction that licenses an mFR arm.

    Here the screen's occupancy histogram already excludes out-of-domain samples (they are
    masked, not clamped), and `P` renormalises by the in-domain total, so both sides are
    conditional on the same support.  Pinned, because the safe behaviour is an accident of two
    separate choices agreeing.
    """
    g = _load("nacl_gates")
    n_grid, n_cp, S = 61, 3, 2
    grid = np.linspace(0.2, 1.4, n_grid)
    basins = [dict(label="CIP", r_lo_nm=0.20, r_hi_nm=0.50),
              dict(label="SSIP", r_lo_nm=0.50, r_hi_nm=1.40)]
    masks = g.basin_masks(grid, basins)

    # 100 walkers, of which 20 sat outside the domain and were masked out of the histogram
    counts = np.zeros(n_grid)
    counts[5] = 50.0
    counts[40] = 30.0                                   # 80 in-domain, 20 dropped
    P = [float(counts[masks[b["label"]]].sum() / counts.sum()) for b in basins]
    assert sum(P) == pytest.approx(1.0), "occupancy must be conditional on the domain"

    F_ref = np.linspace(0.0, 8.0, n_grid)
    w = np.exp(-0.40091 * (F_ref - F_ref.min()))
    Q = [float(w[masks[b["label"]]].sum() / w.sum()) for b in basins]
    assert sum(Q) == pytest.approx(1.0), "the bias-aware target must be normalised on that same support"

    # and the screen must be the thing that supplies a masked histogram.
    # NOTE: this originally read src/nacl/core.py, which held a second, UNCALLED copy of the
    # sampler -- so the assertion was verifying dead code while the live driver went unchecked.
    # That copy is gone; the live loop is the script, and this points at it.
    driver = open(os.path.join(ROOT, "scripts", "nacl_screen.py")).read()
    assert "masked_bin_sum(r, torch.ones_like(r), in_dom" in driver
    assert "out_of_domain" in driver, "the excursion fraction must be recorded, not silently absorbed"


def test_incomplete_reference_cannot_be_accepted():
    src = open(os.path.join(ROOT, "scripts", "nacl_ti_analyze.py")).read()
    assert "ACCEPTED=bool(accepted and complete)" in src


def test_single_build_reference_is_refused():
    """The retirement criterion is joint over builds; at --builds 1 its build-spread clause is
    vacuous and every point retires early, producing output shaped like a good reference.  The
    driver must refuse, except on the --smoke path (which never claims to be a reference)."""
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "nacl_ti_torch.py"),
                        "--builds", "1", "--out", "/tmp/nacl_builds1_refusal_test"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "vacuous" in (r.stdout + r.stderr)


# --- the CELL-axis guard: a study verdict must not come from a partial ladder -------------
import nacl_gates as _ng


def test_study_verdict_withheld_when_a_classifiable_cell_is_missing():
    """"no cell is eligible" reads the same whether the ladder was complete or whether a cell
    simply had no data -- and the frozen rule is the SMALLEST N passing, so a missing smaller
    cell can change the answer."""
    m = _ng.map_completeness([8, 16, 32, 64], {64}, n_basins=2)
    assert m["unexplained_missing"] == [32]
    assert not m["COMPLETE"]


def test_cells_below_the_power_floor_are_struck_a_priori_not_flagged_as_holes():
    m = _ng.map_completeness([8, 16, 32, 64], {32, 64}, n_basins=2)
    assert m["structurally_unclassifiable"] == [8, 16]
    assert m["unexplained_missing"] == []
    assert m["COMPLETE"], "8 and 16 need no data: lambda = N Q* < N <= 16"


def test_boundary_cell_is_excluded_only_because_the_inequality_is_strict():
    """N = 16 needs Q* = 1 -- one basin holding the ENTIRE target. With two basins that is
    impossible, so N=16 is unclassifiable; with a single basin it is attainable."""
    assert 16 in _ng.map_completeness([16], set(), n_basins=2)["structurally_unclassifiable"]
    assert 16 in _ng.map_completeness([16], set(), n_basins=1)["unexplained_missing"]


def test_complete_ladder_is_complete():
    m = _ng.map_completeness([8, 16, 32, 64], {8, 16, 32, 64}, n_basins=2)
    assert m["COMPLETE"] and not m["structurally_unclassifiable"]


def test_report_carries_analysis_provenance_so_a_superseded_tree_is_detectable():
    """The SAMPLER is pinned to a worktree that predates tonight's gate fixes. A report built by
    that tree's nacl_gates.py would look identical to a correct one; absence of this block is
    what makes it detectable."""
    p = _ng.analysis_provenance()
    assert p["analysis_commit"] and len(p["analysis_commit"]) >= 7
    for g in ("gate_c_power_guard_lambda_min_16", "cell_map_completeness_guard",
              "gate_a_preregistered_direction"):
        assert g in p["guards"], f"{g} missing from the guards manifest"


def test_gate_a_reads_the_preregistered_direction_not_the_transpose():
    """The check that can HALT the study was reading `max_TV` -- the superseded spec-transpose --
    because reference_report.json names the transpose generically and the verdict
    `preregistered_*`. Both cleared 0.30 so nothing moved, but the wrong quantity was gating."""
    import subprocess, sys, os, json, tempfile, shutil
    src = "results/nacl/reference/reference_report.json"
    rep = json.load(open(src))
    assert rep["gateA"]["preregistered_max_TV"] != rep["gateA"]["max_TV"], \
        "fixture assumption: the two directions must differ, else this test proves nothing"
    with tempfile.TemporaryDirectory() as td:
        shutil.copy("results/nacl/reference/reference.npz", td)
        # a report that PASSES on the transpose and FAILS on the preregistered direction:
        # reading the wrong key would let the study proceed.
        rep["gateA"]["preregistered_PASS"] = False
        rep["gateA"]["preregistered_max_TV"] = 0.11
        json.dump(rep, open(os.path.join(td, "reference_report.json"), "w"))
        r = subprocess.run([sys.executable, "scripts/nacl_gates.py",
                            "--screen", "results/nacl/screen_all", "--ref", td,
                            "--out", td], capture_output=True, text=True)
    assert r.returncode != 0, "Gate A must FAIL on the preregistered direction"
    assert "0.110" in r.stdout + r.stderr, "must report the preregistered value, not the transpose"


def test_missing_preregistered_field_raises_rather_than_falling_back():
    """A silent fallback to `max_TV` would restore the exact defect being fixed."""
    import subprocess, sys, os, json, tempfile, shutil
    rep = json.load(open("results/nacl/reference/reference_report.json"))
    with tempfile.TemporaryDirectory() as td:
        shutil.copy("results/nacl/reference/reference.npz", td)
        rep["gateA"].pop("preregistered_PASS", None)
        json.dump(rep, open(os.path.join(td, "reference_report.json"), "w"))
        r = subprocess.run([sys.executable, "scripts/nacl_gates.py",
                            "--screen", "results/nacl/screen_all", "--ref", td,
                            "--out", td], capture_output=True, text=True)
    assert r.returncode != 0 and "predates the transpose correction" in r.stdout + r.stderr


def test_merge_refuses_to_average_an_unclassifiable_field():
    """The merge infers each field's layout from its SHAPE. A field it cannot classify used to
    fall through to np.mean across halves -- silently producing a merged cell in which that
    field means something neither half meant, with nothing raised and the merge reporting
    success. Averaging is now allowed only for explicitly declared per-checkpoint scalars."""
    import importlib.util, numpy as _np
    spec = importlib.util.spec_from_file_location(
        "nsm", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "scripts", "nacl_screen_merge.py"))
    nsm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nsm)
    assert "diag_out_of_domain" in nsm.MEANABLE, "the known per-checkpoint scalar must stay meanable"
    # a field whose shape matches no seed/walker/shared pattern and is not declared meanable
    src = open(nsm.__file__).read()
    assert "raise SystemExit" in src and "silently" in src, \
        "the unclassifiable branch must refuse, not average"
    assert "np.mean(np.stack(arrays)" in src.split("elif key in MEANABLE")[1].split("else:")[0], \
        "averaging must sit behind the MEANABLE allowlist, not in the fallthrough"
