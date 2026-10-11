"""Tests of the Experiment A analysis tools (docs/final_campaign/PREREGISTRATION.md 1.2, 1.5-1.8):
scripts/final_campaign/expA_{analysis_common,cross_cell,mech,outcome,tables,figures,analyze,make_fixtures}.py.

  * contrast / Holm / Holm-consistent CI / decomposition math on synthetic per-seed vectors with known answers;
  * the frozen outcome classifier (every branch and boundary) and the combination rule across N;
  * guard g3, the genealogy time average, cost per success (incl. 0 successes -> inf), the inf-aware ratio of medians,
    the exact additivity of the gate-excess decomposition;
  * the output policy (frozen trees refused; same inputs skipped; a COMPLETE output of different inputs refused);
  * the frozen files (engine, harness, old analysis code) are byte-identical to their recorded sha256;
  * an end-to-end run on a tiny REAL fixture (scripts/final_campaign/expA_make_fixtures.py: real engine runs, N 16 / 4,
    10 seeds, under pytest's temporary directory) through analyze_ladder, cross_cell, mech, tables, figures, outcome.
    EXPA_FULL_PIPELINE=1 also runs the whole driver expA_analyze.py (incl. plot_config / plot_synthesis / the
    completeness audit, ~3 min on 4 cores).
Run pinned, e.g.  taskset -c 80-87 python -m pytest -q tests/test_final_expA_analysis.py --basetemp=<scratch dir>.
"""
import copy
import csv
import hashlib
import json
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FC = os.path.join(ROOT, "scripts", "final_campaign")
sys.path.insert(0, FC)
import expA_analysis_common as C  # noqa: E402
import expA_cross_cell as XC  # noqa: E402
import expA_mech as XM  # noqa: E402
import expA_outcome as XO  # noqa: E402
import expA_tables as XT  # noqa: E402
import cross_cell as X  # noqa: E402
import mech_analysis as MA  # noqa: E402

SEEDS = list(range(8300, 8332))
NB = 2000


@pytest.fixture(autouse=True, scope="session")
def _disk_pause_log_in_tmp(tmp_path_factory):
    """Disk-pause lines (PREREGISTRATION section 3) of in-process tool calls go to the temporary directory, never to
    results/final_campaign/expA/logs (the tools honour $EXPA_DISK_PAUSE_LOG)."""
    p = tmp_path_factory.mktemp("disk") / "disk_pauses.log"
    old = os.environ.get(C.DISK_PAUSE_LOG_ENV)
    os.environ[C.DISK_PAUSE_LOG_ENV] = str(p)
    yield p
    if old is None:
        os.environ.pop(C.DISK_PAUSE_LOG_ENV, None)
    else:
        os.environ[C.DISK_PAUSE_LOG_ENV] = old


# ============================================================================================ frozen files
FROZEN = {
    "src/gateway_family_numba.py": "69087f2c0c7edbe5e076615b9a4425ba7580a6c4537a61854ae6c633df9da4ed",
    "src/gateway_ladder_numba.py": "bdeaa40a42f82d509fcdbe1c209559a93c0d0fae9391633c8858488bc25074ce",
    "src/lta_ladder_numba.py": "d2d3c4313fc52bc37e047e30cb607e958b168fcff2dee8b24d9aa12b925a804b",
    "scripts/equal_budget/eqb_metrics.py": "f0c03971e561b16c755f6b3c4bdf41fdcd2747cc4afdf551cb9b3630404dc684",
    "scripts/equal_budget/eqb_family.py": "75c6b9dd539166f9f1ae9025bf6da43bc80a2cc92be306440ce3713a71b25f75",
    "scripts/analyze_gateway_replica_ladder.py": "f4bf0a0f0e99cfac971a6d53d060ce991dff9d97a241a80a448392493304c29d",
    "src/eb_abffr_core.py": "a99c7713ff194a9110b30e5057e6ed016bade0d0116a5d5e70eb5aef886d2a7c",
    "scripts/mechanism/cross_cell.py": "3b5da8683bda3f02915e0f00242e3870c7f19b169352ec9381ee801e5e6f4b37",
    "scripts/mechanism/mech_analysis.py": "d59dec74ca9b89fed70d6b2a0bf964eac1296fdeeb39fd63516a05c5290d085e",
    "scripts/equal_budget/analyze_ladder.py": "edc6a490c67e5eca1272819124e9a9ce8de4e2ca6c29c69848f90281b6423284",
    "scripts/equal_budget/plot_config.py": "7de9aab9fed5b34030a83330da89c6f15f36379477dd84e7e89b57ebbd14521e",
    "scripts/equal_budget/plot_synthesis.py": "45a86ab1e9f92f88c1f9972cba16d65f6fd6aa373050aa94bd12327d5b1a2e51",
    "scripts/equal_budget/audit_completeness.py": "a75462575dfde04beaf399a1881aa31ef5345b13d670efe14f38de800db4ddd0",
    "scripts/mechanism/build_references.py": "63af49c7be64d23f63fae49639dde4a476cbfbd7c80420006313ed59ea27ad05",
}


def test_frozen_files_are_unchanged():
    for rel, sha in FROZEN.items():
        h = hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()
        assert h == sha, f"{rel} changed (sha256 {h[:12]} != frozen {sha[:12]})"


# ============================================================================================ contrast math
def Lof(vals):
    return {s: float(v) for s, v in zip(SEEDS, vals)}


def test_contrast_row_constant_shift_exact_ratio_degenerate_cis_and_minimal_p():
    rng = np.random.default_rng(3)
    Lb = Lof(rng.normal(0.1, 0.2, 32))
    c = math.log(0.8)
    La = {s: v + c for s, v in Lb.items()}
    r = XC.contrast_row(La, Lb, NB, step1_ci=True)
    assert r["n"] == 32 and r["claim_allowed"]
    assert r["ratio"] == pytest.approx(0.8, abs=1e-12)
    assert r["ratio_ci95"] == pytest.approx([0.8, 0.8], abs=1e-12)
    assert r["ratio_ci975"] == pytest.approx([0.8, 0.8], abs=1e-12)
    assert r["p"] == pytest.approx(2.0 / (NB + 1))
    assert not r["equivalent"] and r["wins_a"] == 32


def test_holm_consistent_ci_uses_the_same_draws_and_contains_the_95_ci():
    rng = np.random.default_rng(11)
    La, Lb = Lof(rng.normal(-0.2, 0.3, 32)), Lof(rng.normal(0.0, 0.3, 32))
    r = XC.contrast_row(La, Lb, NB, step1_ci=True)
    d = np.array([La[s] - Lb[s] for s in SEEDS])
    g = np.random.default_rng(C.BOOT_SEED)
    boot = np.median(d[g.integers(0, 32, size=(NB, 32))], axis=1)
    assert r["Delta_ci95"] == pytest.approx([np.percentile(boot, 2.5), np.percentile(boot, 97.5)], abs=1e-15)
    assert r["Delta_ci975"] == pytest.approx([np.percentile(boot, 1.25), np.percentile(boot, 98.75)], abs=1e-15)
    assert r["Delta_ci975"][0] <= r["Delta_ci95"][0] and r["Delta_ci975"][1] >= r["Delta_ci95"][1]
    assert r["ratio_ci975"] == pytest.approx([math.exp(x) for x in r["Delta_ci975"]])
    assert XC.HOLM_STEP1_PCT == pytest.approx((1.25, 98.75))
    # cross_cell.contrast agrees (same construction, imported)
    c = X.contrast(La, Lb, NB, C.BOOT_SEED)
    assert c["Delta"] == r["Delta"] and c["p"] == r["p"]


def test_decomposition_known_answer():
    rng = np.random.default_rng(5)
    abf_b = {s: float(v) for s, v in zip(SEEDS, rng.uniform(0.01, 0.05, 32))}
    fr_b = {s: v * float(rng.uniform(0.7, 1.3)) for s, v in abf_b.items()}
    abf_a = {s: 0.5 * v for s, v in abf_b.items()}           # ABF error halves at lambda_a
    fr_a = {s: 0.4 * v for s, v in fr_b.items()}             # FR error falls to 40 %
    La = {s: math.log(fr_a[s] / abf_a[s]) for s in SEEDS}
    Lb = {s: math.log(fr_b[s] / abf_b[s]) for s in SEEDS}
    r = XC.contrast_row(La, Lb, NB, abf=(abf_a, abf_b), fr=(fr_a, fr_b))
    assert r["ratio"] == pytest.approx(0.8, abs=1e-12)        # 0.4 / 0.5
    assert r["decomposition"]["abf"]["ratio"] == pytest.approx(0.5, abs=1e-12)
    assert r["decomposition"]["fr"]["ratio"] == pytest.approx(0.4, abs=1e-12)
    assert r["decomposition"]["abf"]["ratio_ci95"] == pytest.approx([0.5, 0.5], abs=1e-12)


def test_holm_over_two_N():
    assert C.holm([0.01, 0.04], m=2) == pytest.approx([0.02, 0.04])
    assert C.holm([0.03, 0.04], m=2) == pytest.approx([0.06, 0.06])
    assert C.holm([0.01, math.nan], m=2) == pytest.approx([0.02, 1.0])


def _synthetic_cells(tmp_path, Lmap, abf_scale=None, G_hi=None, n_seeds=32):
    """Cells with synthetic summaries (no run files): Lmap {lam: {N: {seed: L}}}; ABF Ibar = 0.02 x abf_scale."""
    cells = {}
    seeds = SEEDS[:n_seeds]
    for lam, byN in Lmap.items():
        P = dict(cell=f"shift_lam{lam:g}", results_dir=str(tmp_path / f"r{lam:g}"), analysis_dir=str(tmp_path / f"a{lam:g}"),
                 fig_dir=str(tmp_path / f"f{lam:g}"), out_dir=f"r{lam:g}", system="gateway_family",
                 model=dict(variant="shift", alpha=None, kappa=1.0, lam=float(lam)),
                 engine_cfg=dict(variant="shift", alpha=None, kappa=1.0, lam=float(lam), h=2.5e-5),
                 B=3276800000, N_ladder=[512, 128], seeds=seeds, _fixture_of="synthetic")
        S = dict(per_N={}, contrasts={}, status={}, reference=dict(primary_reference=dict(digest="d0")))
        for N, L in byN.items():
            sc = (abf_scale or {}).get((lam, N), 1.0)
            abf = {s: 0.02 * sc for s in seeds}
            fr = {s: abf[s] * math.exp(L[s]) for s in seeds}
            S["per_N"][str(N)] = {m: {k: dict(per_seed={str(s): v[s] for s in seeds})
                                      for k in ("Ibar_F", "final_e_F", "Ibar_Fp_stat", "Ibar_TV_half")}
                                  for m, v in (("abf", abf), ("fr", fr))}
            g = (G_hi or {}).get((lam, N), 0.05)
            blk = dict(G_median=0.0, G_ci95=[-0.05, g], wins=16, losses=16, ties=0, n=len(seeds))
            S["contrasts"][str(N)] = dict(primary=dict(Ibar_F=blk, final_e_F=blk), secondary=dict(Ibar_Fp_stat=blk),
                                          marginal=dict(Ibar_TV_half=blk))
            S["status"][str(N)] = dict(methods={m: dict(complete=seeds) for m in ("abf", "fr")})
        cp, sp = tmp_path / f"c{lam:g}.json", tmp_path / f"s{lam:g}.json"
        cp.write_text(json.dumps(P))
        sp.write_text(json.dumps(S))
        cells[float(lam)] = dict(config=str(cp), P=P, lam=float(lam), name=P["cell"], summary_path=str(sp), S=S,
                                 why=None)
    return cells


def test_cross_cell_analyse_known_answers(tmp_path, monkeypatch):
    """E1 = log(0.8) shift at both N -> R 0.8, p = 2/(nb+1), Holm 2p; E2 compared with an independent computation;
    the decomposition; G_hi; the secondary metrics unadjusted."""
    monkeypatch.setattr(XC.X, "check_summary", lambda rec: [])
    rng = np.random.default_rng(21)
    base = {N: Lof(rng.normal(0.05, 0.15, 32)) for N in (512, 128)}
    noise = {N: Lof(rng.normal(0.0, 0.2, 32)) for N in (512, 128)}
    Lmap = {1: base,
            4: {N: {s: base[N][s] + 0.01 for s in SEEDS} for N in base},
            16: {N: {s: base[N][s] + math.log(0.8) for s in SEEDS} for N in base},
            128: {N: {s: base[N][s] + noise[N][s] for s in SEEDS} for N in base}}
    cells = _synthetic_cells(tmp_path, Lmap, abf_scale={(16, 512): 0.5, (16, 128): 0.5},
                             G_hi={(16, 512): -0.01, (16, 128): 0.02})
    doc = XC.analyse(cells, NB, verify_runs=False)
    rows = {(r["id"], r["N"]): r for r in doc["contrasts"]["Ibar_F"]["rows"]}
    p0 = 2.0 / (NB + 1)
    for N in (512, 128):
        e1 = rows[("E1", N)]
        assert e1["ratio"] == pytest.approx(0.8, abs=1e-9) and e1["p"] == pytest.approx(p0)
        assert e1["p_holm"] == pytest.approx(2 * p0)
        assert e1["decomposition"]["abf"]["ratio"] == pytest.approx(0.5, abs=1e-9)
        assert e1["decomposition"]["fr"]["ratio"] == pytest.approx(0.4, abs=1e-9)
        e2 = rows[("E2", N)]
        c = X.contrast(Lmap[128][N], Lmap[1][N], NB, C.BOOT_SEED)
        assert e2["ratio"] == pytest.approx(c["ratio"], rel=1e-9) and e2["p"] == pytest.approx(c["p"])
        assert rows[("T1", N)]["p_holm"] is None and rows[("T1", N)]["kind"] == "trend"
        assert rows[("T1", N)]["ratio"] == pytest.approx(math.exp(0.01), rel=1e-9)
    p2 = [rows[("E2", N)]["p"] for N in (512, 128)]
    assert [rows[("E2", N)]["p_holm"] for N in (512, 128)] == pytest.approx(X.holm(p2, 2))
    assert doc["G_hi"]["16"] == {"512": -0.01, "128": 0.02}
    assert all(r["p_holm"] is None for r in doc["contrasts"]["final_e_F"]["rows"])
    assert doc["contrasts"]["Ibar_F"]["families"]["E1"]["complete"]
    assert doc["per_cell"]["16"]["512"]["Ibar_F"]["ratio"] == pytest.approx(
        math.exp(float(np.median([Lmap[16][512][s] for s in SEEDS]))), rel=1e-9)


def test_cross_cell_small_n_and_missing_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(XC.X, "check_summary", lambda rec: [])
    rng = np.random.default_rng(2)
    seeds = SEEDS[:9]
    b = {N: {s: float(rng.normal()) for s in seeds} for N in (512, 128)}
    cells = _synthetic_cells(tmp_path, {1: b, 16: b}, n_seeds=9)
    doc = XC.analyse(cells, NB, verify_runs=False)
    rows = {(r["id"], r["N"]): r for r in doc["contrasts"]["Ibar_F"]["rows"]}
    assert rows[("E1", 512)]["n"] == 9 and not rows[("E1", 512)]["claim_allowed"]
    assert "no claim" in rows[("E1", 512)]["verdict"]
    assert rows[("E2", 512)]["n"] == 0 and rows[("E2", 512)]["missing"] == ["128"]
    assert math.isnan(rows[("E2", 512)]["p_holm"])
    assert not doc["contrasts"]["Ibar_F"]["families"]["E2"]["complete"]
    assert sorted(doc["cells_missing"]) == ["128", "4"]


# ============================================================================================ outcome classifier
G_ALL = dict(g1=True, g2=True, g3=True, g4=True)


@pytest.mark.parametrize("args,label", [
    ((0.8, 0.01, [0.7, 0.9], -0.05, G_ALL), "A"),
    ((0.8, 0.01, [0.7, 0.9], -0.05, dict(G_ALL, g3=False)), "A-qualified"),
    ((0.8, 0.01, [0.7, 0.9], -0.05, dict(G_ALL, g2=None)), "A-qualified"),
    ((0.8, 0.01, [0.7, 0.9], 0.0, G_ALL), "B"),                     # G_hi exactly 0 -> 'G_hi >= 0'
    ((0.8, 0.01, [0.7, 0.9], 0.10, dict(G_ALL, g1=False)), "B"),    # guards do not matter for B
    ((0.95, 0.04, [0.90, 1.00], 0.02, G_ALL), "B"),                 # CI touching 1: the rule uses p_Holm
    ((0.8, 0.05, [0.7, 0.95], -0.01, G_ALL), "I"),                  # p exactly 0.05 is NOT significant
    ((0.8, 0.20, [0.6, 1.1], -0.01, G_ALL), "I"),
    ((1.0, 0.01, [0.95, 1.05], -0.01, G_ALL), "I"),                 # R exactly 1: neither R < 1 nor R > 1
    ((1.0, 0.01, [0.95, 1.05], 0.01, G_ALL), "C-inconclusive"),     # significant but R = 1: not C-equivalent
    ((1.2, 0.01, [1.1, 1.3], 0.1, G_ALL), "C-reverse"),
    ((1.2, 0.01, [1.1, 1.3], -0.1, G_ALL), "C-reverse"),            # resolution r1 (also satisfies I)
    ((0.98, 0.05, [0.92, 1.05], 0.03, G_ALL), "C-equivalent"),      # p exactly 0.05 -> p >= 0.05
    ((1.00, 0.60, [0.90, 1.11], 0.03, G_ALL), "C-equivalent"),      # band inclusive at both ends
    ((1.00, 0.60, [0.90, 1.1100001], 0.03, G_ALL), "C-inconclusive"),
    ((1.00, 0.60, [0.8999999, 1.05], 0.03, G_ALL), "C-inconclusive"),
    ((0.9, 0.30, [0.8, 1.3], 0.05, G_ALL), "C-inconclusive"),
    ((0.9, 0.30, [0.8, 1.3], 0.0, G_ALL), "C-inconclusive"),        # G_hi = 0 -> not I
])
def test_classify_branches_and_boundaries(args, label):
    r = XO.classify(*args, n=32)
    assert r["label"] == label, (args, r)
    if label == "A-qualified":
        assert r["failed_guards"] and all(g.split()[0] in XO.GUARDS for g in r["failed_guards"])
        assert r["failed_guards"][0].split()[0] in r["wording"]
    if label == "C-reverse" and args[3] < 0:
        assert r["also_satisfies"] == ["I"] and r["resolution"] == "r1" and "nonetheless" in r["wording"]
    assert r["wording"].startswith(XO.WORDING[label].split("{")[0][:20])


def test_classify_guard_not_evaluable_is_named():
    r = XO.classify(0.8, 0.01, [0.7, 0.9], -0.05, dict(G_ALL, g4=None), n=32)
    assert r["label"] == "A-qualified" and r["failed_guards"] == ["g4 (not evaluable)"] and r["resolution"] == "r2"


def test_classify_no_claim_and_not_available():
    assert XO.classify(0.8, 0.01, [0.7, 0.9], -0.05, G_ALL, n=9)["label"] == "no-claim"
    assert XO.classify(0.8, 0.01, [0.7, 0.9], -0.05, G_ALL, n=10)["label"] == "A"
    assert XO.classify(0.8, 0.01, [0.7, 0.9], -0.05, G_ALL, n=0)["label"] == "not-available"
    assert XO.classify(math.nan, 0.01, [0.7, 0.9], -0.05, G_ALL, n=32)["label"] == "not-available"
    assert XO.classify(0.8, 0.01, [0.7, math.inf], -0.05, G_ALL, n=32)["label"] == "not-available"
    assert XO.classify(0.8, None, [0.7, 0.9], -0.05, G_ALL)["label"] == "not-available"


def _rule(label, R, p, lo, hi, G, guards):
    sig, eq = p < 0.05, (0.90 <= lo and hi <= 1.11)
    hold = all(guards.get(g) is True for g in XO.GUARDS)
    return dict(A=sig and R < 1 and G < 0 and hold, **{"A-qualified": sig and R < 1 and G < 0 and not hold},
                B=sig and R < 1 and G >= 0, I=(not (sig and R < 1)) and G < 0,
                **{"C-equivalent": (not sig) and eq and G >= 0, "C-reverse": sig and R > 1})[label] if label in (
        "A", "A-qualified", "B", "I", "C-equivalent", "C-reverse") else None


def test_classify_exhaustive_grid_matches_the_frozen_table():
    """Every combination on a grid of boundary values: the returned label's own rule holds; no other of the six
    explicit rules holds except in the documented I / C-reverse overlap (r1); C-inconclusive only when none does."""
    labs6 = ("A", "A-qualified", "B", "I", "C-equivalent", "C-reverse")
    n = 0
    for R in (0.7, 0.999, 1.0, 1.001, 1.3):
        for p in (0.001, 0.0499, 0.05, 0.0501, 0.5):
            for lo, hi in ((0.95, 1.05), (0.90, 1.11), (0.85, 1.0), (1.0, 1.2), (0.899, 1.111)):
                for G in (-0.1, -1e-12, 0.0, 0.1):
                    for guards in (G_ALL, dict(G_ALL, g3=False), dict(G_ALL, g1=None)):
                        r = XO.classify(R, p, [lo, hi], G, guards, n=32)
                        lab = r["label"]
                        hits = [L for L in labs6 if _rule(L, R, p, lo, hi, G, guards)]
                        if lab == "C-inconclusive":
                            assert hits == [], (R, p, lo, hi, G, hits)
                        elif set(hits) == {"I", "C-reverse"}:
                            assert lab == "C-reverse" and r["also_satisfies"] == ["I"]
                        else:
                            assert hits == [lab], (R, p, lo, hi, G, guards, hits, lab)
                        n += 1
    assert n == 5 * 5 * 5 * 4 * 3


# ============================================================================================ combination across N
def test_combine_across_N_every_pair():
    labs = XO.LABELS
    for a in labs:
        for b in labs:
            c = XO.combine_across_N({512: a, 128: b})
            g = c["global_label"]
            if a in XO.RESCUE and b in XO.RESCUE:
                weaker = min((a, b), key=lambda x: XO.STRENGTH[x])
                assert g == f"rescue ({weaker} governs)" and c["governing_label"] == weaker
                # finding 7: ONLY the governing label's wording; "rescues FR" only when both N are A
                assert c["wording"] == XO.WORDING[weaker].format(guards="see the per-N guards")
                assert ("rescues FR" in c["wording"]) is (weaker == "A")
            elif (a in XO.RESCUE and b in ("C-equivalent", "C-inconclusive", "I")):
                assert g == "rescue at N = 512 only"
            elif (b in XO.RESCUE and a in ("C-equivalent", "C-inconclusive", "I")):
                assert g == "rescue at N = 128 only"
            elif (a in XO.RESCUE and b == "C-reverse") or (b in XO.RESCUE and a == "C-reverse"):
                assert g == "N-dependent"
            elif a in ("C-equivalent", "C-reverse") and b in ("C-equivalent", "C-reverse"):
                assert g == "no rescue"
            else:
                assert g == "no global claim" and "no rescue" not in c["wording"].replace("no detectable rescue", "")


def test_combine_weaker_label_governs_and_incomplete():
    c = XO.combine_across_N({512: "A", 128: "B"})
    assert c["governing_label"] == "B" and c["wording"] == XO.WORDING["B"] and c["global_label"] == "rescue (B governs)"
    assert "rescues" not in c["wording"] and "A" in c["wording_context"]
    c = XO.combine_across_N({512: "A-qualified", 128: "A"}, failed={512: ["g3"], 128: []})
    assert c["governing_label"] == "A-qualified" and c["wording"] == XO.WORDING["A-qualified"].format(guards="g3 (N = 512)")
    assert XO.combine_across_N({512: "A", 128: "A"})["wording"] == "faster mobility rescues FR"
    c = XO.combine_across_N({512: "B", 128: "C-inconclusive"})
    assert c["global_label"] == "rescue at N = 512 only" and XO.WORDING["B"] in c["wording"]
    assert XO.combine_across_N({512: "A", 128: "A-qualified"})["governing_label"] == "A-qualified"
    assert XO.combine_across_N({512: "A", 128: "A"})["governing_label"] == "A"
    c = XO.combine_across_N({512: "A", 128: "no-claim"})
    assert c["global_label"] == "no global claim (incomplete)" and c["claimable_N"] == ["512"]
    assert XO.combine_across_N({512: "A"})["global_label"] == "no global claim (incomplete)"


# ============================================================================================ g3, genealogy, cost
def test_g3_rule_boundaries():
    ref = {s: 1.0 for s in SEEDS}
    mf_ref = {s: 0.5 for s in SEEDS}
    r = XM.g3_rule(dict(ref), ref, dict(mf_ref), mf_ref, NB)
    assert r["holds"] is True and r["ess_ratio"]["median"] == 1.0 and r["calibrated"]
    r = XM.g3_rule({s: 0.5 for s in SEEDS}, ref, mf_ref, mf_ref, NB)
    assert r["holds"] is False and r["fails_ess"] and not r["fails_maxfam"]
    r = XM.g3_rule({s: 0.8 for s in SEEDS}, ref, mf_ref, mf_ref, NB)      # upper CI exactly 0.8: not < 0.8
    assert r["ess_ratio"]["ci95"][1] == 0.8 and r["holds"] is True
    r = XM.g3_rule(ref, ref, {s: 0.625 for s in SEEDS}, mf_ref, NB)       # lower CI exactly 1.25: not > 1.25
    assert r["maxfam_ratio"]["ci95"][0] == 1.25 and r["holds"] is True
    r = XM.g3_rule(ref, ref, {s: 0.65 for s in SEEDS}, mf_ref, NB)
    assert r["holds"] is False and r["fails_maxfam"]
    r = XM.g3_rule({}, ref, mf_ref, mf_ref, NB)
    assert r["holds"] is None


def test_g3_rule_median_of_per_seed_ratios_with_bootstrap():
    rng = np.random.default_rng(4)
    ref = {s: float(v) for s, v in zip(SEEDS, rng.uniform(0.3, 0.9, 32))}
    hi = {s: ref[s] * float(rng.uniform(0.6, 0.9)) for s in SEEDS}
    r = XM.g3_rule(hi, ref, ref, ref, NB)
    rat = np.array([hi[s] / ref[s] for s in SEEDS])
    g = np.random.default_rng(C.BOOT_SEED)
    b = np.median(rat[g.integers(0, 32, size=(NB, 32))], axis=1)
    assert r["ess_ratio"]["median"] == pytest.approx(float(np.median(rat)))
    assert r["ess_ratio"]["ci95"] == pytest.approx([np.percentile(b, 2.5), np.percentile(b, 97.5)])
    assert r["holds"] is (not np.percentile(b, 97.5) < 0.8)


def test_ess_time_average_uses_uniform_saves_after_the_window():
    n_steps = 2000
    uni = np.arange(1, 201) * 10
    extra = np.array([3, 7, 15, 155])                           # log / physical saves (not uniform) are ignored
    step = np.unique(np.concatenate([uni, extra]))
    ess = np.where(np.isin(step, uni), step / 1000.0, 99.0)
    W = 150
    want = float(np.mean(uni[uni >= W] / 1000.0))
    assert XM.ess_time_average(step, ess, n_steps, W) == pytest.approx(want)
    assert math.isnan(XM.ess_time_average(step, ess, n_steps, 10**6))


def test_cost_per_success_known_values_and_zero_successes():
    r = XM.cost_per_success([0.5, math.inf, 0.25], [10.0, 10.0, 10.0])
    assert r["n_success"] == 2 and r["a"] == pytest.approx(15.0) and r["b"] == pytest.approx((5 + 10 + 2.5) / 2)
    r = XM.cost_per_success([math.inf, math.inf], [3.0, 4.0])
    assert r["n_success"] == 0 and r["a"] == math.inf and r["b"] == math.inf and "0 successes" in r["note"]
    r = XM.cost_per_success([0.5, math.nan], [2.0, 9.0])
    assert r["n_runs"] == 1 and r["a"] == 2.0 and r["b"] == 1.0


def test_walker_step_cost_is_exact():
    r = XM.walker_step_cost([0.5, math.inf, 0.25], 200, 4)
    assert r["numerator_a"] == 3 * 800 and r["numerator_b"] == 400 + 800 + 200
    assert isinstance(r["numerator_a"], int) and isinstance(r["numerator_b"], int)
    assert r["a"] == 1200.0 and r["b"] == 700.0
    r = XM.walker_step_cost([math.inf], 200, 4)
    assert r["a"] == math.inf and r["n_success"] == 0
    with pytest.raises(C.ExpAError):
        XM.walker_step_cost([0.5025], 200, 4)                   # 100.5 steps: not a save fraction


def test_ratio_of_medians_with_censoring_conventions():
    a = {s: math.inf for s in SEEDS[:11]}
    b = {s: 1.0 for s in SEEDS[:11]}
    assert C.ratio_of_medians(a, b, NB)["ratio"] == math.inf
    assert C.ratio_of_medians(b, a, NB)["ratio"] == 0.0
    r = C.ratio_of_medians(a, a, NB)
    assert math.isnan(r["ratio"]) and r["undefined_frac"] == 1.0 and r["unresolved"]
    x = {s: float(i + 1) for i, s in enumerate(SEEDS[:11])}
    y = {s: 2.0 * v for s, v in x.items()}
    r = C.ratio_of_medians(x, y, NB)
    assert r["ratio"] == pytest.approx(0.5) and r["undefined_frac"] == 0.0
    assert C.ratio_inf(np.array([0.0]), np.array([0.0]))[0] != C.ratio_inf(np.array([0.0]), np.array([0.0]))[0]


def test_row_median_inf_matches_quantile_inf():
    rng = np.random.default_rng(9)
    for n in (5, 6, 11, 12):
        X_ = rng.uniform(0, 1, size=(50, n))
        X_[rng.uniform(size=X_.shape) < 0.3] = np.inf
        got = C.row_median_inf(X_)
        want = [C.quantile_inf(row, 50) for row in X_]
        assert np.array_equal(got, np.array(want))


def _d2_arrays(rng, S=12, bias=None):
    n = rng.poisson(30, size=(S, MA.N_AGE, MA.NB)).astype(float)
    ref = np.linspace(-1, 1, MA.NB)
    b = np.zeros(MA.N_AGE) if bias is None else np.asarray(bias)
    m = ref[None, None, :] + b[None, :, None] + rng.normal(0, 0.05, size=n.shape)
    s1 = n * m
    s2 = n * (m * m + 0.2)
    cat = lambda a: np.concatenate([a, a.sum(1, keepdims=True)], axis=1)  # noqa: E731
    return cat(n), cat(s1), cat(s2), ref


def test_gate_excess_decomposition_is_exactly_additive():
    rng = np.random.default_rng(1)
    nF, s1F, s2F, ref = _d2_arrays(rng, bias=[0.3, 0.1, 0.0, 0.0, -0.05])
    nA, s1A, s2A, _ = _d2_arrays(rng)
    et = np.full(MA.NB, 0.2)
    cf, ca = MA.D2Calc(nF, s1F, s2F, ref, et), MA.D2Calc(nA, s1A, s2A, ref, et)
    d = XM.gate_excess_decomposition(cf, ca, 300)
    K = np.asarray(d["K"])
    assert abs(d["sum_classes_minus_all"]) < 1e-10 and abs(K[:MA.N_AGE].sum() - K[MA.ALL]) < 1e-10
    assert np.asarray(d["phi"])[:MA.N_AGE].sum() == pytest.approx(1.0) and d["phi"][MA.ALL] == pytest.approx(1.0)
    assert K[0] > K[2]                        # the young class carries its +0.3 bias
    lo, hi = np.asarray(d["K_ci95"][0]), np.asarray(d["K_ci95"][1])
    assert np.all(lo <= K + 1e-12) and np.all(K <= hi + 1e-12)
    assert d["coverage"] == pytest.approx(1.0)


def test_ci_guard():
    assert XM.ci_guard([0.01, 0.2]) is False          # entirely above 0: g1 / g2 fail
    assert XM.ci_guard([0.0, 0.2]) is True            # touching 0 is not 'entirely above'
    assert XM.ci_guard([-0.1, 0.2]) is True
    assert XM.ci_guard([None, 0.2]) is None


def test_mean_companions_known_answer():
    """PREREGISTRATION 1.6 mean-based companions: mean G and mean L with percentile seed-bootstrap CIs."""
    abf = {s: 0.02 for s in SEEDS[:10]}
    fr = {s: 0.02 * (1.0 + 0.1 * (i - 4.5)) for i, s in enumerate(SEEDS[:10])}
    fr[SEEDS[10]] = 1.0                                        # no ABF partner: dropped
    m = C.mean_companions(abf, fr, NB)
    G = np.array([(fr[s] - abf[s]) / abf[s] for s in SEEDS[:10]])
    L = np.log(np.array([fr[s] / abf[s] for s in SEEDS[:10]]))
    assert m["G"]["n"] == 10 and m["G"]["mean"] == pytest.approx(G.mean()) and m["L"]["mean"] == pytest.approx(L.mean())
    g = np.random.default_rng(C.BOOT_SEED)
    b = G[g.integers(0, 10, size=(NB, 10))].mean(axis=1)
    assert m["G"]["ci95"] == pytest.approx([np.percentile(b, 2.5), np.percentile(b, 97.5)])
    m = C.mean_companions({s: 0.02 for s in SEEDS[:3]}, {s: 0.03 for s in SEEDS[:3]}, NB)
    assert m["G"]["mean"] == pytest.approx(0.5) and m["G"]["ci95"] == pytest.approx([0.5, 0.5])
    assert C.mean_companions({}, {}, NB)["L"]["n"] == 0


def test_per_cell_block_carries_the_mean_companions(tmp_path, monkeypatch):
    monkeypatch.setattr(XC.X, "check_summary", lambda rec: [])
    rng = np.random.default_rng(8)
    L = {N: Lof(rng.normal(0.05, 0.1, 32)) for N in (512, 128)}
    cells = _synthetic_cells(tmp_path, {1: L, 16: L})
    doc = XC.analyse(cells, NB, verify_runs=False)
    blk = doc["per_cell"]["16"]["512"]["Ibar_F"]
    assert blk["mean"]["L"]["mean"] == pytest.approx(float(np.mean([L[512][s] for s in SEEDS])))
    assert blk["mean"]["G"]["mean"] == pytest.approx(float(np.mean([math.exp(L[512][s]) - 1 for s in SEEDS])))
    assert "mean G" in XC.md(C.reload_json(C.M.json_safe(doc)))


def test_gate_decompositions_per_flank_and_quadratic_are_additive():
    """Finding 10: opposite-sign class bias on the two flanks cancels in the pooled signed decomposition but not per
    flank, and the quadratic decomposition is sign-insensitive and exactly additive over the five classes."""
    rng = np.random.default_rng(13)
    S = 12
    x = MA.CENTRES
    sign = np.where(x < 0, 1.0, -1.0)
    n = rng.poisson(40, size=(S, MA.N_AGE, MA.NB)).astype(float)
    ref = np.zeros(MA.NB)
    bias = np.zeros((MA.N_AGE, MA.NB))
    bias[1] = 0.4 * sign                                       # young clones: +0.4 left flank, -0.4 right flank
    m = ref[None, None, :] + bias[None] + rng.normal(0, 0.02, size=n.shape)
    cat = lambda a: np.concatenate([a, a.sum(1, keepdims=True)], axis=1)  # noqa: E731
    nF, s1F, s2F = cat(n), cat(n * m), cat(n * (m * m + 0.1))
    nA = rng.poisson(40, size=(S, MA.N_AGE, MA.NB)).astype(float)
    mA = rng.normal(0, 0.02, size=nA.shape)
    et = np.full(MA.NB, 0.1)
    cf = MA.D2Calc(nF, s1F, s2F, ref, et)
    ca = MA.D2Calc(cat(nA), cat(nA * mA), cat(nA * (mA * mA + 0.1)), ref, et)
    d = XM.gate_decompositions(cf, ca, 200)
    Kg, Kn, Kp = (np.asarray(d["signed"][r]["K"]) for r in ("gate", "gate_neg", "gate_pos"))
    for r in XM.DECOMP_REGIONS:
        assert abs(d["signed"][r]["sum_classes_minus_all"]) < 1e-10
        assert abs(d["quadratic"][r]["sum_classes_minus_Q_sum"]) < 1e-10
    assert Kn[1] > 0.02 and Kp[1] < -0.02                      # the flanks carry the young-clone bias ...
    assert abs(Kg[1]) < 0.25 * min(Kn[1], -Kp[1])              # ... which (nearly) cancels in the pooled gate
    Qg = np.asarray(d["quadratic"]["gate"]["Q"])
    assert Qg[1] > 0.01 and Qg[1] > 10 * np.nanmax(np.abs(np.delete(Qg[:MA.N_AGE], 1)))
    assert d["quadratic"]["gate"]["Q_sum"] == pytest.approx(float(np.nansum(Qg[:MA.N_AGE])))
    lo, hi = d["quadratic"]["gate"]["Q_sum_ci95"]
    assert lo <= d["quadratic"]["gate"]["Q_sum"] <= hi
    flat = XM.gate_excess_decomposition(cf, ca, 200)
    assert np.allclose(np.asarray(flat["K"], float), Kg, equal_nan=True) and "CANCELS" in flat["definition"]


def test_disk_pause_waits_logs_resumes_and_stops_after_24h(tmp_path, monkeypatch):
    """PREREGISTRATION section 3: pause below 2 GB, resume only above 3 GB, log the pause; > 24 h stops."""
    GB = 1024 ** 3
    seq = iter([1.5 * GB, 2.5 * GB, 2.9 * GB, 3.5 * GB])
    clock = iter([0.0, 60.0, 120.0, 180.0, 240.0])
    lp = tmp_path / "pauses.log"
    slept = []
    assert C.wait_for_disk("x", free_fn=lambda: 5 * GB, sleep_fn=slept.append, log_path=str(lp)) == 0.0
    assert not lp.exists() and slept == []
    dt = C.wait_for_disk("x.json", free_fn=lambda: next(seq), sleep_fn=slept.append, log_path=str(lp),
                         now_fn=lambda: next(clock))
    assert len(slept) == 3 and dt > 0                          # 2.5 and 2.9 GB are not above 3 GB: still paused
    txt = lp.read_text()
    assert "DISK PAUSE before x.json" in txt and "DISK RESUME" in txt
    t = iter(float(i) * 3600 for i in range(100))
    with pytest.raises(C.ExpAError, match="24 h"):
        C.wait_for_disk("y", free_fn=lambda: 1 * GB, sleep_fn=lambda s: None, log_path=str(lp), now_fn=lambda: next(t))
    # without log_path the pause line goes to $EXPA_DISK_PAUSE_LOG (never into results/ during the tests)
    ep = tmp_path / "env.log"
    monkeypatch.setenv(C.DISK_PAUSE_LOG_ENV, str(ep))
    seq2 = iter([1.0 * GB, 4.0 * GB])
    C.wait_for_disk("z", free_fn=lambda: next(seq2), sleep_fn=lambda s: None)
    assert "DISK PAUSE before z" in ep.read_text()


def test_atomic_write_removes_its_temporary_file_on_failure(tmp_path):
    p = tmp_path / "out.json"
    def boom(tmp):
        open(tmp, "w").write("partial")
        raise OSError(28, "No space left on device")
    with pytest.raises(OSError):
        C._atomic(str(p), boom)
    assert not p.exists() and list(tmp_path.iterdir()) == []
    C.write_json(str(p), dict(a=1))
    with pytest.raises(OSError):
        C._atomic(str(p), boom)
    assert json.load(open(p)) == dict(a=1) and sorted(x.name for x in tmp_path.iterdir()) == ["out.json"]


def test_placeholder_is_neither_signed_nor_complete(tmp_path):
    p = str(tmp_path / "o.json")
    C.write_json(p, dict(input_signature="s1", complete=True))
    C.begin_output(p, "x/1", previous="s1")
    d = json.load(open(p))
    assert d["state"] == "writing" and d["input_signature"] is None and d["complete"] is False
    assert C.decide_write(p, "s2") == "write" and C.previous_signature(p) == "s1"
    with pytest.raises(C.ExpAError, match="interrupted write"):
        C.load_signed(p)


def _write_audit(rec, verdict="PASS", summary_sha=None, config_sha=None, engine=C.ENGINE_SHA256):
    p = C.audit_path(rec)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    sysd = dict(verdict=verdict, n_failures=0 if verdict == "PASS" else 1,
                failures=[] if verdict == "PASS" else [dict(code="ENGINE_MIXED")],
                summary=dict(sha256=summary_sha or C.sha256_file(rec["summary_path"])),
                config_sha256=config_sha or C.sha256_file(rec["config"]),
                run_files=dict(engines={f"gateway_family_numba {engine}": 64}))
    json.dump(dict(verdict=verdict, generated_utc="t", systems={C.MF.FAMILY: sysd}), open(p, "w"))


def test_completeness_requires_current_passing_audits_nboot_and_no_provisional(tmp_path, monkeypatch):
    """Findings 1, 3, 5: complete needs every cell's audit PASS on the CURRENT summary / config with only frozen-engine
    runs, n_boot 10 000 in the tool and every summary, and no --provisional."""
    rng = np.random.default_rng(1)
    L = {N: Lof(rng.normal(0, 0.1, 32)) for N in (512, 128)}
    cells = _synthetic_cells(tmp_path, {lam: L for lam in (1, 4, 16, 128)})
    for rec in cells.values():
        rec["S"]["n_boot"] = C.N_BOOT
        json.dump(rec["S"], open(rec["summary_path"], "w"))
    monkeypatch.setattr(C, "design_status", lambda cells: (True, False, []))
    reasons = C.completeness(cells, C.N_BOOT)["reasons"]
    assert any("no completeness audit" in r for r in reasons) and not C.completeness(cells, C.N_BOOT)["complete"]
    for rec in cells.values():
        _write_audit(rec)
    c = C.completeness(cells, C.N_BOOT)
    assert c["complete"] and c["reasons"] == [] and all(a["passes"] for a in c["audits"].values())
    assert not C.completeness(cells, C.N_BOOT, provisional=True)["complete"]
    assert not C.completeness(cells, C.N_BOOT, verify_runs=False)["complete"]
    assert any("--n-boot 2000" in r for r in C.completeness(cells, 2000)["reasons"])
    r16 = cells[16.0]
    r16["S"]["n_boot"] = 2000
    json.dump(r16["S"], open(r16["summary_path"], "w"))          # summary changed after the audit, and n_boot 2000
    rs = C.completeness(cells, C.N_BOOT)["reasons"]
    assert any("summary n_boot 2000" in r for r in rs) and any("re-run the audit" in r for r in rs)
    r16["S"]["n_boot"] = C.N_BOOT
    json.dump(r16["S"], open(r16["summary_path"], "w"))
    _write_audit(r16, verdict="FAIL")
    assert any("verdict 'FAIL'" in r for r in C.completeness(cells, C.N_BOOT)["reasons"])
    _write_audit(r16, engine="0" * 64)                          # a PASSing audit over a non-frozen engine build
    a = C.audit_status(r16)
    assert not a["passes"] and any("engine other than the frozen" in p for p in a["problems"])
    assert C.audit_failures(cells) == {16.0: a["problems"]}
    _write_audit(r16)
    assert C.audit_failures(cells) == {} and C.completeness(cells, C.N_BOOT)["complete"]
    sig0 = C.audit_signature(C.completeness(cells, C.N_BOOT)["audits"])
    _write_audit(r16, verdict="FAIL")
    assert C.audit_signature(C.completeness(cells, C.N_BOOT)["audits"]) != sig0   # the audit enters the signatures


def test_production_problems_catch_a_changed_engine_knob_or_generator_hash():
    """Finding 5: a cell whose engine_cfg, thresholds or _generated_from_sha256 differ from make_cells.py's output of
    the frozen configs/final_campaign/expA.json is not the production design."""
    P = json.load(open(os.path.join(C.CELLS_DIR, "shift_lam16.json")))
    assert C.production_problems(P) == []
    Q = copy.deepcopy(P)
    Q["engine_cfg"]["gamma"] = 1.25
    pr = C.production_problems(Q)
    assert pr and "engine_cfg keys ['gamma']" in pr[0]
    Q = copy.deepcopy(P)
    Q["_generated_from_sha256"] = "0" * 64
    assert any("_generated_from_sha256" in x for x in C.production_problems(Q))
    Q = copy.deepcopy(P)
    Q["thresholds"]["e_F"] = [0.002, 0.0055, 0.03]
    assert any("thresholds" in x for x in C.production_problems(Q))


def test_prereg_amendment_detection():
    """Finding 6: the outcome resolutions count only as a dated amendment BELOW the frozen amendments line."""
    base = open(C.PREREG).read()
    a = XO.prereg_amendment(base)
    assert not a["found"] and XO.AMENDMENT_MARKER not in base
    above = base.replace("## 1. Experiment A", f"[{XO.AMENDMENT_MARKER}]\n\n## 1. Experiment A")
    assert not XO.prereg_amendment(above)["found"]              # above the line: not an amendment
    txt = (base + f"\n### Amendment 1 (2026-10-11): outcome resolutions [{XO.AMENDMENT_MARKER}]\n\nr1 ... r3 ...\n"
           "\n### Amendment 2 (later): something else\n")
    b = XO.prereg_amendment(txt)
    assert b["found"] and "Amendment 1" in b["line"] and b["block_sha256"]
    txt2 = txt.replace("r1 ... r3 ...", "r1 ... r3 ... changed")
    assert XO.prereg_amendment(txt2)["block_sha256"] != b["block_sha256"]
    txt3 = txt.replace("something else", "something else entirely")       # a later amendment: same block hash
    assert XO.prereg_amendment(txt3)["block_sha256"] == b["block_sha256"]


# ============================================================================================ output policy, cells
def test_output_policy(tmp_path):
    p = str(tmp_path / "x.json")
    assert C.decide_write(p, "s1") == "write"
    C.write_json(p, dict(input_signature="s1", complete=False))
    assert C.decide_write(p, "s1") == "skip"
    assert C.decide_write(p, "s1", recompute=True) == "write"
    assert C.decide_write(p, "s2") == "write"               # an incomplete output may be replaced
    C.write_json(p, dict(input_signature="s1", complete=True))
    with pytest.raises(C.ExpAError):
        C.decide_write(p, "s2")
    assert C.decide_write(p, "s2", force=True) == "write"
    for bad in (("results", "mechanism", "x.json"), ("results", "equal_budget_v2", "y.json"),
                ("configs", "final_campaign", "z.json"), ("figures", "mechanism", "a.png")):
        with pytest.raises(C.ExpAError):
            C.guard_output(os.path.join(ROOT, *bad))
    C.guard_output(os.path.join(ROOT, "results", "final_campaign", "expA", "synthesis", "cross_cell.json"))


def test_load_cells_refuses_wrong_models(tmp_path):
    P = json.load(open(os.path.join(ROOT, "configs", "final_campaign", "cells", "expA", "shift_lam16.json")))
    d = tmp_path / "cells"
    d.mkdir()
    (d / "a.json").write_text(json.dumps(P))
    cells = C.load_cells(str(d))
    assert list(cells) == [16.0] and cells[16.0]["name"] == "shift_lam16"
    (d / "b.json").write_text(json.dumps(P))
    with pytest.raises(C.ExpAError, match="two cell configs"):
        C.load_cells(str(d))
    os.remove(d / "b.json")
    Q = copy.deepcopy(P)
    Q["model"]["lam"] = Q["engine_cfg"]["lam"] = 8.0
    (d / "b.json").write_text(json.dumps(Q))
    with pytest.raises(C.ExpAError, match="not one of the planned"):
        C.load_cells(str(d))
    Q = copy.deepcopy(P)
    Q["model"].update(variant="alpha", alpha=1.0, kappa=None)
    (d / "b.json").write_text(json.dumps(Q))
    with pytest.raises(C.ExpAError, match="not the shifted fibre"):
        C.load_cells(str(d))


def test_production_cells_are_the_frozen_design():
    cells = C.load_cells(C.CELLS_DIR)
    if len(cells) != 4:
        pytest.skip("production cell configs not all present")
    prod, fixture, probs = C.design_status(cells)
    assert prod and not fixture and probs == []
    assert C.common_N(cells) == [512, 128]


def test_replication_block_reads_the_exp1_shift_summary_read_only():
    if not os.path.exists(C.EXP1_SHIFT_SUMMARY):
        pytest.skip("Experiment I summary absent")
    sha0 = C.sha256_file(C.EXP1_SHIFT_SUMMARY)
    S = json.load(open(C.EXP1_SHIFT_SUMMARY))
    cells = {1.0: dict(S=S, P=dict(N_ladder=[512, 128], seeds=SEEDS))}
    r = XM.replication_block(cells, C.EXP1_SHIFT_SUMMARY, NB)
    assert r["N"] == [512, 128]
    row = r["per_N"]["512"]["Ibar_F"]
    g = S["contrasts"]["512"]["primary"]["Ibar_F"]
    assert row["exp1_shift"]["G_median"] == pytest.approx(g["G_median"]) == pytest.approx(row["expA_lam1"]["G_median"])
    assert C.sha256_file(C.EXP1_SHIFT_SUMMARY) == sha0


# ============================================================================================ end to end (real fixture)
@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    import expA_make_fixtures as MK
    import analyze_ladder as AL
    root = tmp_path_factory.mktemp("expA_fixture")
    rc = MK.main(["--root", str(root), "--seeds", "10", "--workers", "4"])
    assert rc == 0
    cdir = root / "configs" / "cells" / "expA"
    for p in sorted(cdir.glob("*.json")):
        AL.analyze("gateway_family", None, str(p), None, n_boot=C.N_BOOT, verbose=False)   # the audit needs 10 000
    return root


def test_end_to_end_on_tiny_real_cells(fixture_root):
    root = fixture_root
    cdir, syn, tab, fig = (str(root / "configs" / "cells" / "expA"), str(root / "syn"), str(root / "tables"),
                           str(root / "figs"))
    assert XC.main(["--cells-dir", cdir, "--out-dir", syn, "--n-boot", str(NB)]) == 0
    assert XM.main(["--cells-dir", cdir, "--out-dir", syn, "--n-boot", str(NB)]) == 0
    assert XT.main(["--cells-dir", cdir, "--out-dir", tab]) == 0
    import expA_figures as XF
    assert XF.main(["--cells-dir", cdir, "--synthesis-dir", syn, "--fig-dir", fig, "--strict-legibility"]) == 0
    assert XO.main(["--cells-dir", cdir, "--synthesis-dir", syn]) == 0
    cells = C.load_cells(cdir)
    cc = C.reload_json(json.load(open(os.path.join(syn, "cross_cell.json"))))
    assert cc["fixture"] and not cc["complete"] and cc["N"] == [16, 4] and cc["freshness_verified"]
    # E rows equal an independent recomputation from the summaries' per-seed values
    for r in cc["contrasts"]["Ibar_F"]["rows"]:
        Sa, Sb = cells[float(r["a"])]["S"], cells[float(r["b"])]["S"]
        La, _ = X.log_ratios(C.per_seed(Sa, r["N"], "abf", "Ibar_F"), C.per_seed(Sa, r["N"], "fr", "Ibar_F"))
        Lb, _ = X.log_ratios(C.per_seed(Sb, r["N"], "abf", "Ibar_F"), C.per_seed(Sb, r["N"], "fr", "Ibar_F"))
        d = [La[s] - Lb[s] for s in sorted(set(La) & set(Lb))]
        assert r["n"] == 10 and r["ratio"] == pytest.approx(math.exp(float(np.median(d))), rel=1e-12)
    for fid in ("E1", "E2"):
        fr = [r for r in cc["contrasts"]["Ibar_F"]["rows"] if r["id"] == fid]
        assert [r["p_holm"] for r in fr] == pytest.approx(X.holm([r["p"] for r in fr], 2))
    for lk in ("1", "4", "16", "128"):
        for N in ("16", "4"):
            g = C.summary_contrast(cells[float(lk)]["S"], int(N), "Ibar_F")
            assert C.fnum(cc["G_hi"][lk][N]) == pytest.approx(C.fnum(g["G_ci95"][1]))
    mech = C.reload_json(json.load(open(os.path.join(syn, "mech.json"))))
    assert set(mech["d2"]) == {"1", "4", "16", "128"}
    for lk, byN in mech["d2"].items():
        for N, b in byN.items():
            dec = b["gate_excess_decomposition"]
            assert abs(C.fnum(dec["sum_classes_minus_all"])) < 1e-9
            assert b["n_seeds"] == 10 and b["young_clone_class"] == "[0.01,0.1)"
            gd = mech["guards"][lk][N]
            assert gd["g4"]["holds"] is True, gd["g4"]["problems"][:3]
            assert gd["g1"]["holds"] in (True, False) and gd["g2"]["holds"] in (True, False)
            if lk != "1":
                assert gd["g3"]["holds"] in (True, False) and gd["g3"]["n"] == 10
            assert mech["genealogy"][lk][N]["maxfam_matches_summary"]
            assert mech["genealogy"][lk][N]["ess_win_tavg"]["n"] == 10
    for lk in ("4", "16", "128"):
        rows = mech["mc3"][lk]["per_N"]
        ph = X.holm([rows[N]["B_excess_did_p"] for N in ("16", "4")], 2)
        assert [rows[N]["p_holm"] for N in ("16", "4")] == pytest.approx(ph)
        for N in ("16", "4"):
            want = bool(rows[N]["p_holm"] < 0.05 and rows[N]["B_excess_did"] < 0)
            assert rows[N]["passes"] is want
    c = mech["cost"]["16"]["4"]["fr"]["tau_e_F_mid_u"]
    assert c["walker_steps"]["n_runs"] == 10 and c["core_seconds"]["n_runs"] == 10
    if c["walker_steps"]["n_success"] == 0:
        assert c["walker_steps"]["a"] == math.inf
    assert len(mech["run_files"]) == 4 * 2 * 2 * 10
    # tables: one row per (lambda, N, seed, arm); G equals the summary's paired G
    rows = list(csv.DictReader(open(os.path.join(tab, "seed_level.csv"))))
    assert len(rows) == 4 * 2 * 10 * 2
    S16 = cells[16.0]["S"]
    g16 = {int(k): float(v) for k, v in S16["contrasts"]["16"]["primary"]["Ibar_F"]["per_seed"].items()}
    for r in rows:
        if r["lambda"] == "16" and r["N"] == "16":
            assert float(r["G_Ibar_F"]) == pytest.approx(g16[int(r["seed"])], rel=1e-12)
    man = json.load(open(os.path.join(fig, "MANIFEST_expA_synthesis.json")))
    assert sorted(e["id"] for e in man["figures"]) == ["A1", "A2", "A3", "A5", "A6", "A7", "A8"]
    assert man["legibility_summary"]["n_fail"] == 0
    for e in man["figures"]:
        for f in e["files"]:
            assert os.path.getsize(os.path.join(fig, f)) > 1000
    out = C.reload_json(json.load(open(os.path.join(syn, "outcome.json"))))
    for fid in ("E1", "E2"):
        for N, r in out["families"][fid]["per_N"].items():
            assert r["label"] in XO.LABELS + ("no-claim", "not-available")
        assert out["families"][fid]["combined"]["global_label"]
    # finding 6: no amendment yet -> outcome cites its absence and is never complete
    assert out["prereg_amendment"]["found"] is False and not out["complete"]
    assert any("resolutions r1-r3" in r for r in out["completeness"]["reasons"])
    assert "NOT YET recorded as a dated amendment" in open(os.path.join(syn, "outcome.md")).read()
    # finding 1: no completeness audit on the fixture -> named in every output's completeness reasons
    for d in (cc, mech, out, man, json.load(open(os.path.join(tab, "tables_meta.json")))):
        assert any("no completeness audit" in r or "audit verdict" in r for r in d["completeness"]["reasons"])
    # finding 9: mean-based companions per (lambda, N, metric)
    blk = cc["per_cell"]["16"]["16"]["Ibar_F"]["mean"]
    assert blk["G"]["n"] == 10 and math.isfinite(C.fnum(blk["G"]["mean"])) and len(blk["L"]["ci95"]) == 2
    # finding 10: per-flank signed and quadratic decompositions, exactly additive
    for lk, byN in mech["d2"].items():
        for N, b in byN.items():
            dec = b["gate_decomposition"]
            for r in ("gate", "gate_neg", "gate_pos"):
                sg, qd = dec["signed"][r], dec["quadratic"][r]
                if math.isfinite(C.fnum(sg["K"][MA.ALL])):         # a region without deposits is NaN (tiny fixture)
                    assert abs(C.fnum(sg["sum_classes_minus_all"])) < 1e-9
                else:
                    assert all(math.isnan(C.fnum(k)) for k in sg["K"])
                if math.isfinite(C.fnum(qd["Q_sum"])):
                    assert abs(C.fnum(qd["sum_classes_minus_Q_sum"])) < 1e-9
            assert math.isfinite(C.fnum(dec["signed"]["gate"]["K"][MA.ALL]))
    # idempotency: same inputs -> skipped (file untouched)
    jp = os.path.join(syn, "cross_cell.json")
    m0 = os.path.getmtime(jp)
    assert XC.main(["--cells-dir", cdir, "--out-dir", syn, "--n-boot", str(NB)]) == 0
    assert os.path.getmtime(jp) == m0
    # a COMPLETE output of different inputs is refused, --force overrides
    doc = json.load(open(jp))
    doc["complete"] = True
    json.dump(doc, open(jp, "w"))
    with pytest.raises(C.ExpAError):
        XC.main(["--cells-dir", cdir, "--out-dir", syn, "--n-boot", str(NB + 1)])
    assert XC.main(["--cells-dir", cdir, "--out-dir", syn, "--n-boot", str(NB + 1), "--force"]) == 0
    assert json.load(open(jp))["replaced_signature"] == doc["input_signature"]


def test_cross_cell_refuses_a_stale_summary(fixture_root, tmp_path):
    """A run file rewritten after the analysis makes the summary stale: cross_cell refuses (check_freshness)."""
    root = fixture_root
    cdir = root / "configs" / "cells" / "expA"
    cells = C.load_cells(str(cdir))
    P = cells[4.0]["P"]
    p = os.path.join(P["results_dir"], "N16", f"s{P['seeds'][0]}_abf.npz")
    data = open(p, "rb").read()
    try:
        with open(p, "ab") as fh:
            fh.write(b"\0")                       # content change -> per-run cache key no longer valid
        with pytest.raises(C.ExpAError, match="STALE"):
            XC.analyse(C.load_cells(str(cdir)), 200, verify_runs=True)
    finally:
        open(p, "wb").write(data)


@pytest.mark.skipif(os.environ.get("EXPA_FULL_PIPELINE") != "1", reason="set EXPA_FULL_PIPELINE=1 (about 3 min)")
def test_full_driver_on_the_fixture(fixture_root):
    import expA_analyze as XA
    root = fixture_root
    args = ["--cells-dir", str(root / "configs" / "cells" / "expA"), "--log-dir", str(root / "logs"),
            "--synthesis-dir", str(root / "syn_full"), "--tables-dir", str(root / "tab_full"),
            "--fig-dir", str(root / "fig_full"), "--workers", "4"]
    assert XA.main(args) == 0
    rec = sorted((root / "logs").glob("expA_analyze_*.json"))[-1]
    d = json.load(open(rec))
    assert d["audit_failed"] == [] and d["synthesis_ok"] is True
    assert XA.main(args) == 0                           # second run: everything but the audit is skipped
    d2 = json.load(open(sorted((root / "logs").glob("expA_analyze_*.json"))[-1]))
    assert all(s["decision"] == "skip" for s in d2["steps"] if s.get("stage") in ("an", "pc", "ps"))


def test_outcome_refuses_inputs_from_different_cell_states():
    cc = dict(cells={"1": dict(config_sha256="a", summary_sha256="b"), "16": dict(config_sha256="c", summary_sha256="d")})
    mech = dict(cells={"1": dict(config_sha256="a", summary_sha256="b"), "16": dict(config_sha256="c", summary_sha256="d")})
    assert XO.consistency_problems(cc, mech) == []
    mech["cells"]["16"]["summary_sha256"] = "x"
    assert XO.consistency_problems(cc, mech) and "summary_sha256" in XO.consistency_problems(cc, mech)[0]
    del mech["cells"]["16"]
    assert "only one input" in XO.consistency_problems(cc, mech)[0]


def test_driver_decisions_on_the_fixture(fixture_root, tmp_path):
    """expA_analyze: an up-to-date summary is skipped; a COMPLETE summary whose cell config changed is refused unless
    --force-cell; the reference pre-check passes for built references and names the build command otherwise."""
    import expA_analyze as XA
    cells = C.load_cells(str(fixture_root / "configs" / "cells" / "expA"))
    rec = cells[16.0]
    assert XA.an_decision(rec)[0] == "skip"
    assert XA.reference_problems(rec) == []
    P = copy.deepcopy(rec["P"])
    P["thresholds"]["e_F"] = [0.002, 0.0055, 0.03]
    cp = tmp_path / "shift_lam16.json"
    cp.write_text(json.dumps(P))
    rec2 = dict(rec, config=str(cp), P=P)
    d, why = XA.an_decision(rec2)
    assert d == "refuse" and "COMPLETE" in why
    assert XA.an_decision(rec2, force=True)[0] == "run"
    P3 = dict(copy.deepcopy(rec["P"]), reference_file=str(tmp_path / "nope.npz"))
    probs = XA.reference_problems(dict(rec, P=P3))
    assert probs and "build_references.py" in probs[0]


def test_driver_refuses_to_degrade_a_complete_summary_when_run_files_disappear(fixture_root, tmp_path):
    """Finding 2: a COMPLETE summary whose run files were removed after the analysis is NOT re-analysed (which would
    replace it with an incomplete one) unless --force-cell; the driver exits 1 and leaves the summary untouched."""
    import expA_analyze as XA
    cdir = fixture_root / "configs" / "cells" / "expA"
    rec = C.load_cells(str(cdir))[4.0]
    P = rec["P"]
    moved = []
    for s_ in P["seeds"][:2]:
        src = os.path.join(P["results_dir"], "N16", f"s{s_}_fr.npz")
        dst = str(tmp_path / os.path.basename(src))
        os.replace(src, dst)
        moved.append((src, dst))
    sha0 = C.sha256_file(rec["summary_path"])
    try:
        d, why = XA.an_decision(rec)
        assert d == "refuse" and "stale against the run tree" in why and "--force-cell" in why
        assert XA.an_decision(rec, force=True)[0] == "run"
        rc = XA.main(["--cells-dir", str(cdir), "--stages", "an", "--only-lams", "4", "--log-dir", str(tmp_path / "logs"),
                      "--synthesis-dir", str(tmp_path / "syn"), "--tables-dir", str(tmp_path / "tab"),
                      "--fig-dir", str(tmp_path / "fig")])
        assert rc == 1 and C.sha256_file(rec["summary_path"]) == sha0
        recd = json.load(open(sorted((tmp_path / "logs").glob("expA_analyze_*.json"))[-1]))
        assert [s_["decision"] for s_ in recd["steps"] if s_.get("stage") == "an"] == ["refuse"]
    finally:
        for src, dst in moved:
            os.replace(dst, src)
    assert XA.an_decision(rec)[0] == "skip"


def test_driver_audit_gate_forwards_provisional(fixture_root, tmp_path, monkeypatch):
    """Finding 1: the synthesis is gated on every cell's audit JSON even when the 'audit' stage is not run; with
    --allow-audit-fail every synthesis tool gets --provisional and writes complete = false."""
    import argparse
    import expA_analyze as XA
    a = argparse.Namespace(cells_dir="c", synthesis_dir="s", tables_dir="t", fig_dir="f", n_boot=10000,
                           force_synthesis=False)
    cmds = XA.synthesis_cmds(a, provisional=True)
    assert all("--provisional" in c for c in cmds.values()) and "--cells-dir" in cmds["outcome"]
    assert not any("--provisional" in c for c in XA.synthesis_cmds(a, provisional=False).values())
    cdir = fixture_root / "configs" / "cells" / "expA"
    cells = C.load_cells(str(cdir))
    backups = {}
    for lam, rec in cells.items():                  # a FAILing audit for lambda 4, a PASSing one elsewhere
        p = C.audit_path(rec)
        if os.path.exists(p):
            backups[p] = open(p, "rb").read()
        _write_audit(rec, verdict="FAIL" if lam == 4.0 else "PASS")
    base = ["--cells-dir", str(cdir), "--log-dir", str(tmp_path / "logs"), "--synthesis-dir", str(tmp_path / "syn"),
            "--tables-dir", str(tmp_path / "tab"), "--fig-dir", str(tmp_path / "fig"), "--n-boot", "200",
            "--stages", "cross"]
    try:
        assert XA.main(base) == 1
        recd = json.load(open(sorted((tmp_path / "logs").glob("expA_analyze_*.json"))[-1]))
        assert recd["audit_not_passing"] == ["shift_lam4"] and recd["synthesis_ok"] is False
        assert not (tmp_path / "syn" / "cross_cell.json").exists()
        assert XA.main(base + ["--allow-audit-fail"]) == 1          # provisional: never exit 0
        cc = json.load(open(tmp_path / "syn" / "cross_cell.json"))
        assert cc["provisional"] is True and cc["complete"] is False
        assert any("--provisional" in r for r in cc["completeness"]["reasons"])
        assert any("lambda 4: completeness audit verdict 'FAIL'" in r for r in cc["completeness"]["reasons"])
    finally:
        for lam, rec in cells.items():
            p = C.audit_path(rec)
            if p in backups:
                open(p, "wb").write(backups[p])
            else:
                os.remove(p)


def test_driver_waits_for_disk_before_every_child(tmp_path, monkeypatch):
    """Finding 8: the unchanged equal-budget tools do not check free space, so the driver pauses BEFORE each child."""
    import expA_analyze as XA
    calls = []
    monkeypatch.setattr(XA.C, "wait_for_disk", lambda what="", **kw: calls.append((what, kw.get("log_path"))) or 0.0)
    rc, _ = XA.run_cmd([sys.executable, "-c", "import os; print(os.environ['EXPA_DISK_PAUSE_LOG'])"],
                       str(tmp_path / "l" / "x.log"))
    assert rc == 0 and len(calls) == 1 and calls[0][1] == str(tmp_path / "l" / "disk_pauses.log")
    assert str(tmp_path / "l" / "disk_pauses.log") in open(tmp_path / "l" / "x.log").read()


def test_mech_reports_runs_with_diagnostic_or_engine_problems(fixture_root, tmp_path, monkeypatch):
    """Findings 4 and 5: a run whose D1-D3 identities fail, or that was made by an engine other than the frozen
    69087f2c..., no longer aborts expA_mech ('run files changed'); g4 fails naming it and mech.json is incomplete."""
    cdir = str(fixture_root / "configs" / "cells" / "expA")
    cells = C.load_cells(cdir)
    s0, s1 = cells[16.0]["P"]["seeds"][:2]
    orig = XM.MA.load_run

    def fake(path, P, N, seed, method):
        r = orig(path, P, N, seed, method)
        if r.get("status") == "complete" and P["cell"] == "shift_lam16" and int(N) == 4 and method == "fr":
            if int(seed) == s0:
                r["diag_problems"] = ["D1: injected inconsistency"]
            elif int(seed) == s1:
                r["meta"] = dict(r["meta"], engine_sha256="f" * 64)
        return r
    monkeypatch.setattr(XM.MA, "load_run", fake)
    out = str(tmp_path / "syn")
    assert XM.main(["--cells-dir", cdir, "--out-dir", out, "--n-boot", "200"]) == 0
    m = C.reload_json(json.load(open(os.path.join(out, "mech.json"))))
    g4 = m["guards"]["16"]["4"]["g4"]
    assert g4["holds"] is False
    assert any(f"fr seed {s0}: D1-D3 diagnostics" in p for p in g4["problems"])
    assert any(f"fr seed {s1}: engine sha256 ffffffff" in p for p in g4["problems"])
    assert m["guards"]["16"]["16"]["g4"]["holds"] is True
    assert len(m["run_files"]) == 160 and not m["runs_complete"] and not m["complete"]
    assert m["engine_problems"] and any("engine other than the frozen" in r for r in m["completeness"]["reasons"])
    assert m["cells"]["16"]["per_N"]["4"]["fr"]["diag_problems"] == {s0: ["D1: injected inconsistency"]} or \
        m["cells"]["16"]["per_N"]["4"]["fr"]["diag_problems"] == {str(s0): ["D1: injected inconsistency"]}


def test_tables_figures_outcome_refuse_stale_or_foreign_summaries(fixture_root, tmp_path):
    """Finding 11: expA_tables refuses a summary that is stale against the run tree; expA_outcome and expA_figures
    refuse cross_cell / mech documents that agree with each other but not with the summaries on disk."""
    import expA_figures as XF
    cdir = str(fixture_root / "configs" / "cells" / "expA")
    cells = C.load_cells(cdir)
    P = cells[1.0]["P"]
    p = os.path.join(P["results_dir"], "N4", f"s{P['seeds'][0]}_abf.npz")
    data = open(p, "rb").read()
    try:
        with open(p, "ab") as fh:
            fh.write(b"\0")
        with pytest.raises(C.ExpAError, match="STALE"):
            XT.main(["--cells-dir", cdir, "--out-dir", str(tmp_path / "tab")])
        assert not (tmp_path / "tab" / "tables_meta.json").exists()
    finally:
        open(p, "wb").write(data)
    syn = str(tmp_path / "syn")
    assert XC.main(["--cells-dir", cdir, "--out-dir", syn, "--n-boot", "200"]) == 0
    assert XM.main(["--cells-dir", cdir, "--out-dir", syn, "--n-boot", "200"]) == 0
    assert XO.main(["--cells-dir", cdir, "--synthesis-dir", syn]) == 0
    for nm in ("cross_cell.json", "mech.json"):          # both claim the same, non-current lambda 16 summary
        q = os.path.join(syn, nm)
        d = json.load(open(q))
        d["cells"]["16"]["summary_sha256"] = "f" * 64
        json.dump(d, open(q, "w"))
    with pytest.raises(C.ExpAError, match="not made from the same, current"):
        XO.main(["--cells-dir", cdir, "--synthesis-dir", syn])
    with pytest.raises(C.ExpAError, match="not made from the current"):
        XF.main(["--cells-dir", cdir, "--synthesis-dir", syn, "--fig-dir", str(tmp_path / "fig")])
