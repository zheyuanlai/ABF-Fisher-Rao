"""Gates on the authoritative v1 results inventory.

These tests protect the one property the inventory exists to provide: that every headline
number in the closure documentation is the number in the artifact, on a single sign
convention. They read the checked-in artifacts, so they fail loudly if a summary is
regenerated with a different value or an artifact goes missing.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INVENTORY = os.path.join(ROOT, "results/closure/v1_results_inventory.csv")


def _load_builder():
    path = os.path.join(ROOT, "scripts/build_closure_inventory.py")
    spec = importlib.util.spec_from_file_location("build_closure_inventory", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


@pytest.fixture(scope="module")
def rows(builder):
    out = []
    gw, _, _ = builder.gateway_rows()
    out += gw
    sham, _ = builder.wca_sham_rows()
    out += sham
    out += builder.wca_earlier_rows()
    out += builder.toy_rows()
    out += builder.alkane_rows()
    dip, _ = builder.dipeptide_rows()
    out += dip
    return out


def _get(rows, system, arm, comparator, endpoint_prefix):
    hits = [r for r in rows if r["system"] == system and r["arm"] == arm
            and r["comparator"] == comparator
            and r["endpoint"].startswith(endpoint_prefix)]
    assert len(hits) == 1, f"expected exactly one row for {system}/{arm}, got {len(hits)}"
    return hits[0]


def test_builder_consistency_checks_pass(builder, rows):
    errs, _ = builder.check(rows)
    assert errs == [], "inventory consistency checks failed:\n" + "\n".join(errs)


def test_checked_in_inventory_matches_a_fresh_build(builder, rows):
    """The committed CSV must be what the builder produces now, not a stale copy."""
    assert os.path.exists(INVENTORY), "run scripts/build_closure_inventory.py"
    with open(INVENTORY, newline="") as fh:
        stored = list(csv.DictReader(fh))
    assert len(stored) == len(rows)
    for s, f in zip(stored, rows):
        assert s == {k: str(v) for k, v in f.items()}, (
            f"stored row differs from a fresh build: {s['system']} / {s['arm']}")


def test_sign_convention_is_uniform(rows):
    """rel_pct must always be (arm-comparator)/comparator, so negative means better.

    The source artifacts disagree about this -- the WCA production family publishes gains
    (positive = better) while the sham and gateway families publish relative changes. A row
    that leaked the wrong polarity would invert a conclusion, so the direction string is
    pinned and the two anchors with known signs are asserted directly.
    """
    assert {r["direction"] for r in rows} == {
        "lower is better; rel_pct<0 = arm better than comparator"}

    helped = _get(rows, "WCA dimer, matched sham (Case IX)", "practical mFR (fr_estimated)",
                  "ABF", "integrated_l2_f")
    assert float(helped["rel_pct"]) < 0, "the WCA positive must read negative"

    hurt = _get(rows, "WCA dimer, matched sham (Case IX)",
                "matched-turnover sham (sham_practical)", "ABF", "integrated_l2_f")
    assert float(hurt["rel_pct"]) > 0, "the WCA sham is adverse and must read positive"


@pytest.mark.parametrize("tag,expected_pct,expected_wins", [
    ("gateway confirmatory v1", -12.124675075563442, 31),
    ("gateway confirmatory v2 (replicate; quoted)", -12.477509411559378, 31),
])
def test_gateway_headline_matches_artifact(rows, tag, expected_pct, expected_wins):
    r = _get(rows, tag, "practical mFR (fr_estimated)", "ABF", "int_l2_f")
    assert float(r["rel_pct"]) == pytest.approx(expected_pct, abs=5e-4)
    assert r["favorable_seeds"] == f"{expected_wins}/32"


def test_gateway_tost_split_is_recorded():
    """v1 passes the sham equivalence test and v2 does not. Both must stay visible."""
    v1 = json.load(open(os.path.join(
        ROOT, "results/gateway_anchor/confirmatory/confirmatory_summary.json")))
    v2 = json.load(open(os.path.join(
        ROOT, "results/gateway_anchor/confirmatory_v2/confirmatory_summary.json")))
    assert v1["primary_pass"] and v2["primary_pass"], "both replicates pass the primary rule"
    assert v1["sham_tost"]["sham_practical"]["equivalent"] is True
    assert v2["sham_tost"]["sham_practical"]["equivalent"] is False
    margin = v2["preregistration"]["sham_equivalence"]["margin_pct"][1]
    miss = v2["sham_tost"]["sham_practical"]["ci90"][1] - margin
    assert 0 < miss < 1.0, f"v2 should miss the margin by well under a point, got {miss}"


def test_v2_is_flagged_noise_matched():
    """Amendment 2 put all five arms in one batch; the provenance flag must say so.

    The flag was inferred from the batch *count*, which reads the single shared batch as the
    discarded-baseline defect it was introduced to remove.
    """
    v2 = json.load(open(os.path.join(
        ROOT, "results/gateway_anchor/confirmatory_v2/confirmatory_summary.json")))
    assert v2["abf_batches"] == ["all"]
    assert v2["baseline_noise_matched"] is True
    v1 = json.load(open(os.path.join(
        ROOT, "results/gateway_anchor/confirmatory/confirmatory_summary.json")))
    assert v1["baseline_noise_matched"] is False, "v1 genuinely has the pairing defect"


def test_wca_sham_headline_and_seed_set(rows):
    d = json.load(open(os.path.join(ROOT, "results/wca_sham/sham/sham_summary.json")))
    assert d["complete"] and d["sham_mismatches"] == 0 and d["nan_runs"] == 0
    assert d["seeds"] == list(range(400, 416))
    r = _get(rows, "WCA dimer, matched sham (Case IX)", "practical mFR (fr_estimated)",
             "ABF", "integrated_l2_f")
    assert float(r["rel_pct"]) == pytest.approx(-22.831995672030104, abs=5e-4)
    assert r["favorable_seeds"] == "16/16"
    direct = _get(rows, "WCA dimer, matched sham (Case IX)", "practical mFR (fr_estimated)",
                  "its own matched-turnover sham (sham_practical)", "integrated_l2_f")
    assert float(direct["rel_pct"]) == pytest.approx(-26.38036017720518, abs=5e-4)


def test_wca_round_trips_are_a_separate_endpoint(rows):
    """The transport diagnostic must never be filed under an accuracy endpoint."""
    rt = [r for r in rows if r["endpoint"].startswith("n_round_trips")]
    assert len(rt) == 1
    assert 0.5 < float(rt[0]["rel_pct"]) < 2.0, (
        "round trips move by about a percent; if this moves the establishment reading "
        "needs restating")
    acc = _get(rows, "WCA dimer, matched sham (Case IX)", "practical mFR (fr_estimated)",
               "ABF", "integrated_l2_f")
    assert abs(float(acc["rel_pct"])) > 20.0
    assert abs(float(acc["rel_pct"])) > 10 * abs(float(rt[0]["rel_pct"]))


def test_negative_and_neutral_systems_are_present(rows):
    """Closure must not quietly become a positive-results-only table."""
    regimes = {r["regime"] for r in rows}
    assert "ABF-sufficient" in regimes
    assert "discovery-limited" in regimes
    assert "establishment-limited" in regimes

    systems = " | ".join(r["system"] for r in rows)
    for required in ("butane", "pentane", "R15", "torsion torus", "alanine", "valine"):
        assert required in systems, f"{required} missing from the inventory"

    adverse = [r for r in rows if r["rel_pct"] and float(r["rel_pct"]) > 0]
    assert len(adverse) >= 10, "adverse rows appear to have been dropped"


def test_r15_is_discovery_limited_not_merely_an_mfr_null(rows):
    """The discovery-limited label must rest on ABF-only support evidence."""
    r15 = [r for r in rows if "R15" in r["system"]]
    assert r15, "no R15 rows"
    assert {x["regime"] for x in r15} == {"discovery-limited"}
    basis = r15[0]["regime_basis"]
    assert "starved" in basis and "support" in basis
    assert "mFR" not in basis, "a regime label defined by the mFR outcome would be circular"

    screen = os.path.join(ROOT,
                          "results/alkanes_cv_extension/r15/summaries/cv_starvation.csv")
    with open(screen, newline="") as fh:
        cells = list(csv.DictReader(fh))
    starved = [c for c in cells if c["verdict"] == "starved"]
    assert len(starved) == 2, "the ABF-only screen should mark exactly the two b2 R15 cells"
    assert all("b2" in c["cell"] for c in starved)


def test_valine_has_no_mfr_arm_and_says_so(rows):
    v = [r for r in rows if "valine" in r["system"]]
    assert len(v) == 1
    assert "no mFR arm" in v[0]["arm"] or "NO mFR ARM" in v[0]["notes"]
    assert v[0]["regime"] == "ABF-sufficient"
    assert "5.4 ps" in v[0]["regime_basis"] and "52 ps" in v[0]["regime_basis"]


def test_alanine_ladder_covers_the_full_rate_range(rows):
    """The higher-rate alanine arms must not be selectively omitted."""
    ladder = [r for r in rows if "rate ladder" in r["system"]]
    rates = sorted(float(r["system"].split("rate=")[1].split(" ")[0]) for r in ladder)
    assert rates == [0.02, 0.15, 0.45]
    for r in ladder:
        assert abs(float(r["rel_pct"])) < 1.0, (
            "every ladder rate is practically neutral; a large value here would change the "
            "alanine conclusion")


def test_every_artifact_path_exists(rows):
    for r in rows:
        p = r["artifact"].split(" ")[0]
        assert os.path.exists(os.path.join(ROOT, p)), f"missing artifact: {p}"


def test_no_smoke_or_tuning_rows(rows):
    for r in rows:
        assert "/smoke/" not in r["artifact"] and "/tuning/" not in r["artifact"]
        assert not r["setting"].startswith("tuning_"), r["system"]
