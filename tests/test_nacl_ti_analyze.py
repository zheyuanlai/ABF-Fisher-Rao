"""Run the reference analysis on synthetic TI data, before it decides anything real.

``nacl_ti_analyze.py`` turns the constrained-TI output into ``F_ref``, the frozen basins, the
endpoint window, the acceptance ratio, **Gate 0 and Gate A** -- i.e. it can end the study.  It
is ~250 lines of numpy that had never been executed.  Both sessions found defects the moment
they ran never-run analysis code rather than reading it, so this builds inputs with known
answers and checks the outputs against them.

The synthetic reference is a double well with a CIP minimum, a barrier and an SSIP minimum, so
the basin finder, the window and the physical secondaries all have checkable truth.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from nacl import system as nsys                                  # noqa: E402

R_CIP, R_BARRIER, R_SSIP = 0.28, 0.36, 0.50


def true_F(r, kT):
    """Double well: deep CIP, barrier, shallower SSIP, flat plateau beyond ~0.7 nm."""
    cip = -6.0 * kT * np.exp(-((r - R_CIP) / 0.035) ** 2)
    ssip = -2.5 * kT * np.exp(-((r - R_SSIP) / 0.05) ** 2)
    wall = 40.0 * kT * np.exp(-(r - 0.20) / 0.02)          # the steep repulsive edge
    return cip + ssip + wall


def build_synthetic(path, family_spread=0.0, y_separation=1.0, builds=3, reps=3,
                    drop_family_at=None):
    kT = nsys.kT_kJ()
    r_grid = np.round(np.arange(nsys.R_LO_NM, 1.4001, 0.02), 4)
    F = true_F(r_grid, kT)
    dF = np.gradient(F, r_grid)                              # the mean force the TI would find

    rng = np.random.default_rng(4)
    recs, fbar, ysum, ycnt = [], [], [], []
    for i, r in enumerate(r_grid):
        for b in range(builds):
            for fam in range(4):
                for k in range(reps):
                    if drop_family_at is not None and abs(r - drop_family_at) < 1e-9 and fam == 1:
                        recs.append((r, b, fam, k)); fbar.append(np.nan)
                        ysum.append([0, 0, 0]); ycnt.append(0)
                        continue
                    # a controlled cross-family offset: this is the Gate 0 signal
                    off = family_spread * (fam - 1.5) * abs(dF[i])
                    recs.append((r, b, fam, k))
                    fbar.append(dF[i] + off + rng.normal(0, 0.02 * max(abs(dF[i]), 1.0)))
                    # Y separates CIP-side from SSIP-side by `y_separation`
                    base = 9.0 + (y_separation if r > R_BARRIER else 0.0)
                    ysum.append([base + rng.normal(0, 0.05), 6.0, 0.5]); ycnt.append(1)
    os.makedirs(path, exist_ok=True)
    np.savez(os.path.join(path, "ti_final.npz"),
             recs=np.asarray(recs), fbar=np.asarray(fbar), fcnt=np.ones(len(recs)),
             ysum=np.asarray(ysum), ycnt=np.asarray(ycnt),
             retired_at=np.full(len(r_grid), 50.0))
    with open(os.path.join(path, "manifest.json"), "w") as fh:
        json.dump(dict(stage="synthetic", dt_ps=0.002), fh)
    return r_grid, F, kT


def run_analyzer(ti_dir, out_dir):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "nacl_ti_analyze.py"),
                        "--ti", ti_dir, "--out", out_dir],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise AssertionError(f"analyzer failed:\n{r.stdout}\n{r.stderr}")
    return json.load(open(os.path.join(out_dir, "reference_report.json"))), r.stdout


@pytest.fixture(scope="module")
def clean_run():
    with tempfile.TemporaryDirectory() as tmp:
        ti, out = os.path.join(tmp, "ti"), os.path.join(tmp, "ref")
        r_grid, F, kT = build_synthetic(ti, family_spread=0.0, y_separation=1.0)
        rep, stdout = run_analyzer(ti, out)
        yield rep, stdout, r_grid, F, kT, out


def test_it_runs_at_all(clean_run):
    rep, *_ = clean_run
    assert set(rep) >= {"acceptance", "basins", "gate0", "gateA", "endpoint_window",
                        "external_check", "physical", "completeness"}


def test_recovers_the_planted_basins(clean_run):
    rep, _, r_grid, F, kT, _ = clean_run
    labels = [b["label"] for b in rep["basins"]]
    assert "CIP" in labels and len(rep["basins"]) >= 2, rep["basins"]
    cip = rep["basins"][0]
    assert cip["r_min_nm"] == pytest.approx(R_CIP, abs=0.03), "CIP minimum misplaced"
    ssip = rep["basins"][1]
    assert ssip["r_lo_nm"] == pytest.approx(R_BARRIER, abs=0.04), "boundary is not the barrier"


def test_endpoint_window_excludes_the_repulsive_wall(clean_run):
    """The published domain starts ~113 kT up a wall; the frozen window must cut it out."""
    rep, _, r_grid, F, kT, _ = clean_run
    w = rep["endpoint_window"]
    assert w["r_lo_nm"] > nsys.R_LO_NM, "the window kept the repulsive edge"
    F_rel = (F - F.min()) / kT
    inside = (r_grid >= w["r_lo_nm"]) & (r_grid <= w["r_hi_nm"])
    assert F_rel[inside].max() <= 15.0 + 1e-6, "window admits a point above 15 kT"


def test_reference_is_accepted_when_builds_agree(clean_run):
    rep, *_ = clean_run
    assert rep["completeness"]["COMPLETE"] is True
    assert rep["acceptance"]["ACCEPTED"] is True
    assert rep["acceptance"]["ratio"] < 0.5


def test_gate0_is_small_when_families_agree(clean_run):
    rep, *_ = clean_run
    assert rep["gate0"]["COMPUTABLE"] is True
    assert rep["gate0"]["global_spread_ratio"] < 0.05, "no planted spread, yet Gate 0 is large"


def test_gate0_grows_with_a_planted_family_offset():
    """The controlled experiment: a deliberate cross-family disagreement must show up."""
    with tempfile.TemporaryDirectory() as tmp:
        ti, out = os.path.join(tmp, "ti"), os.path.join(tmp, "ref")
        build_synthetic(ti, family_spread=0.40, y_separation=1.0)
        rep, _ = run_analyzer(ti, out)
        assert rep["gate0"]["global_spread_ratio"] > 0.3, \
            "a 0.4x planted family offset did not register in Gate 0"


def test_gateA_passes_when_Y_separates_and_fails_when_it_does_not():
    for sep, expect in ((3.0, True), (0.0, False)):
        with tempfile.TemporaryDirectory() as tmp:
            ti, out = os.path.join(tmp, "ti"), os.path.join(tmp, "ref")
            build_synthetic(ti, y_separation=sep)
            rep, _ = run_analyzer(ti, out)
            assert rep["gateA"]["COMPUTABLE"] is True
            assert rep["gateA"]["PASS"] is expect, (sep, rep["gateA"])


def test_shallow_minima_merge_and_keep_the_DEEPER_one():
    """The `< 2 kT` merge rule, exercised in both orderings.

    The rule pops by list position; the original popped by *grid index* in the branch taken
    when the first minimum is the higher one, which either raises IndexError or -- when the
    grid index happens to be a valid position -- silently deletes an unrelated basin and keeps
    the shallower minimum. The synthetic double well above never reached that branch, because
    its first minimum is the deeper one.
    """
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "an", os.path.join(ROOT, "scripts", "nacl_ti_analyze.py"))
    an = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(an)
    kT = nsys.kT_kJ()
    r = np.linspace(0.2, 1.4, 61)

    # shallow first minimum, deep second, separated by a sub-2kT bump: must merge to the DEEP one
    F = (-1.0 * kT * np.exp(-((r - 0.30) / 0.03) ** 2)
         - 6.0 * kT * np.exp(-((r - 0.50) / 0.04) ** 2))
    mins, bounds = an.find_basins(r, F, kT)
    assert len(mins) == 1, f"a <2kT barrier did not merge: {[round(r[m], 3) for m in mins]}"
    assert r[mins[0]] == pytest.approx(0.50, abs=0.03), "merge kept the shallower minimum"

    # mirrored: deep first, shallow second -- same merge, same survivor rule
    F2 = (-6.0 * kT * np.exp(-((r - 0.30) / 0.04) ** 2)
          - 1.0 * kT * np.exp(-((r - 0.50) / 0.03) ** 2))
    mins2, _ = an.find_basins(r, F2, kT)
    assert len(mins2) == 1
    assert r[mins2[0]] == pytest.approx(0.30, abs=0.03)

    # and a genuine >2kT barrier must NOT merge
    F3 = (-6.0 * kT * np.exp(-((r - 0.28) / 0.025) ** 2)
          - 5.0 * kT * np.exp(-((r - 0.55) / 0.025) ** 2))
    mins3, bounds3 = an.find_basins(r, F3, kT)
    assert len(mins3) == 2 and len(bounds3) == 1


def test_a_missing_family_makes_the_reference_incomplete():
    """Class-1 guard, end to end: absent data must not be accepted as a reference."""
    with tempfile.TemporaryDirectory() as tmp:
        ti, out = os.path.join(tmp, "ti"), os.path.join(tmp, "ref")
        build_synthetic(ti, drop_family_at=0.40)
        rep, stdout = run_analyzer(ti, out)
        assert rep["completeness"]["COMPLETE"] is False
        assert rep["acceptance"]["ACCEPTED"] is False, "an incomplete reference was accepted"
        assert "INCOMPLETE" in stdout


def test_gate0_numerator_and_denominator_share_one_population():
    """Gate 0 is a RATIO, and ratios have this campaign's worst record.

    The failure mode is a numerator over one population and a denominator over another: here,
    a spread averaged only over points where every family reported, divided by a |f| averaged
    over ALL points including ones the numerator skipped. The wall points make that difference
    enormous (|f| ~ 24000 there against ~50 in the physical region), so an incommensurable
    Gate 0 would read ~500x too small and pass anything.

    Constructed so the two answers differ by orders of magnitude, and the commensurable one
    is required.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ti, out = os.path.join(tmp, "ti"), os.path.join(tmp, "ref")
        # family 1 missing at the two wall points, where |f| is ~1000x the physical region
        build_synthetic(ti, family_spread=0.30, drop_family_at=0.20)
        rep, _ = run_analyzer(ti, out)
        g0 = rep["gate0"]
        assert g0["COMPUTABLE"] is True
        # the wall points are excluded from BOTH arguments -> coverage says so
        assert g0["coverage"]["points_used"] < g0["coverage"]["points_total"]
        # and the reported ratio must reflect the planted 0.30 spread, not be diluted by the
        # wall's huge |f| sitting only in the denominator
        assert g0["global_spread_ratio"] > 0.10, (
            f"Gate 0 read {g0['global_spread_ratio']:.4f} for a planted 0.30 spread -- the "
            "denominator is averaging over points the numerator excluded")


def test_acceptance_ratio_uses_the_window_for_both_arguments():
    """max pairwise L2 / (0.10 * span): both must be the FROZEN WINDOW, not the full grid.

    The full grid includes the repulsive wall, whose span is ~25x the window's. An acceptance
    ratio with a window-restricted L2 over a full-grid span would be ~25x too small and would
    accept a reference that disagrees badly with itself.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ti, out = os.path.join(tmp, "ti"), os.path.join(tmp, "ref")
        r_grid, F, kT = build_synthetic(ti, family_spread=0.0)
        rep, _ = run_analyzer(ti, out)
        w, acc = rep["endpoint_window"], rep["acceptance"]
        # Compare against the analyzer's OWN F_ref, not against the true F: the difference
        # between those two is reconstruction noise, which is a different question. What is
        # under test is whether the span and the L2 come from the same masked array.
        d = np.load(os.path.join(out, "reference.npz"))
        F_ref, mask = d["F_ref"], d["endpoint_window"].astype(bool)
        full_span = float(F_ref.max() - F_ref.min())
        assert acc["F_span_kJ"] < 0.5 * full_span, (
            "the reported span is not the window's -- window "
            f"{acc['F_span_kJ']:.1f} vs full grid {full_span:.1f} kJ")
        assert acc["F_span_kJ"] == pytest.approx(
            float(F_ref[mask].max() - F_ref[mask].min()), rel=1e-9), \
            "span and window disagree: the ratio's two arguments are not commensurable"
