#!/usr/bin/env python
"""Standalone validation of the UROPS 2026 figure-data package (run after every export).

Structural checks of the 21 CSVs (schema, numbers, time grids, quartile order, final time, seed records), consistency
of each CSV with the per-seed files (quartiles recomputed independently), the time grid against the raw saved
checkpoints (when ABFFR_DATA_ROOT holds the raw data), pre-FR identity of the paired arms, and the paired statistics
recomputed from the per-seed files and compared with (a) the exporter's meta and (b) the published audit references.
Writes VALIDATION_REPORT.md and validation_results.json in the package root.  Exit code 1 on any FAIL.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

HERE_ = os.path.dirname(os.path.abspath(__file__))

T_FINAL = dict(gateway=40.0, lta150=60.0, lta300=60.0, ite300=50.0, meta=200.0, alanine=100.0, wca=240.0)
N_SEEDS = dict(gateway=32, lta150=16, lta300=16, ite300=16, meta=32, alanine=16, wca=16)
RAW_TIMES = dict(
    gateway=("results/equal_budget_v2/gateway/N2048/s8100_abf.npz", "save_t"),
    lta150=("results/lta_histogram/production_T150/abf.npz", "times"),
    lta300=("results/lta_histogram/production_T300/abf.npz", "times"),
    ite300=("results/convergence_fr_campaign/E1_ite300_stop10/abf_N1024_s1100000.npz", "times"),
    meta=("results/urops_2026/metastability/confirm/s910000_abf.npz", "save_t"),
    alanine=("results/alanine_histogram/hA_c800_w5/raw/hA_c800_w5__abf__c7eq__N2048__T100000__ns16__e852e5d318e3.npz", "times"),
    wca=("results/numerical_validation/wca_confirmatory/dt0.000125_N256_T240/raw/s5100_abf.npz", "times"))
# experiments whose paired arms are documented as statistically (same init + same noise stream), not bitwise, paired:
# value = last save time up to which they must still be identical
STATISTICAL_PAIRING = dict(alanine=2.0)
# audit references (published repository findings; NOT imposed on the data): (statistic, expected, tolerance)
AUDIT = dict(
    gateway=[("Ibar_F frozen v2 anchor (eqb summary.json)", -0.3139, 0.002), ("Ibar_F dt-study h 2.5e-5 (-29.7 %)", -0.297, 0.005)],
    lta150=[("I_F full (LTA_VALIDATION -28.5 %)", -0.2854, 0.002), ("final e_F (-54.4 %)", -0.5438, 0.002)],
    lta300=[("I_F full (LTA_VALIDATION -13.7 %)", -0.1366, 0.002), ("final e_F (-7.9 %, n.s.)", -0.0793, 0.002)],
    ite300=[("post-ramp Ibar_F (-38 %)", -0.3768, 0.005), ("final e_F (-32 %)", -0.3232, 0.005), ("S_geo (1.59)", 1.586, 0.005)],
    alanine=[("dI_F own window (+0.04 %, neutral)", 0.0004, 0.002), ("final e_F (-0.6 %)", -0.0060, 0.002)],
    wca=[("I_F (+9.9 %)", 0.099, 0.003), ("final e_F (+131 %)", 1.31, 0.02)],
    meta=[])


def check(rows, ok, name, detail=""):
    rows.append(dict(check=name, status="PASS" if ok else "FAIL", detail=detail))
    return ok


def note(rows, name, detail):
    rows.append(dict(check=name, status="INFO", detail=detail))


def per_seed(exp, m, arm):
    return C.read_per_seed(os.path.join(C.SEED_DIR, f"{exp}_{m}_{arm}.csv"))


def recompute(exp, meta, rows):
    """The experiment's own paired statistic from the per-seed CSVs (independent of the exporter's arrays)."""
    t, A, _ = per_seed(exp, "eF", "abf")
    _, B, _ = per_seed(exp, "eF", "fr")
    rel = lambda a, b: float(np.median((b - a) / a))
    out = {}
    if exp == "gateway":
        n_steps = meta["config"]["n_steps"]
        k = np.arange(1, 201) * (n_steps // 200)
        idx = np.searchsorted(np.rint(t / meta["config"]["h"]).astype(np.int64), k)
        assert np.allclose(t[idx], k * meta["config"]["h"])
        out["Ibar_F frozen v2 anchor (eqb summary.json)"] = rel(A[idx].mean(0), B[idx].mean(0))
        out["Ibar_F dt-study h 2.5e-5 (-29.7 %)"] = meta["stats"]["dt_study_h2.5e-5"]["dIbar_pct_median"] / 100.0
        out["final e_F"] = rel(A[-1], B[-1])
        mine = dict(Ibar=meta["stats"]["Ibar_F"]["median"])
        out["_meta_consistency"] = abs(out["Ibar_F frozen v2 anchor (eqb summary.json)"] - mine["Ibar"])
    elif exp in ("lta150", "lta300"):
        out[f"I_F full (LTA_VALIDATION {'-28.5' if exp == 'lta150' else '-13.7'} %)"] = rel(np.trapezoid(A, t, axis=0), np.trapezoid(B, t, axis=0))
        out[[k for k, _, _ in AUDIT[exp]][1]] = rel(A[-1], B[-1])
        out["_meta_consistency"] = abs(out[AUDIT[exp][0][0]] - meta["stats"]["I_F_full_trapezoid"]["median"])
    elif exp == "ite300":
        tr = meta["config"]["fr_start_t"]
        sel = t >= tr - 1e-9
        out["post-ramp Ibar_F (-38 %)"] = rel(np.trapezoid(A[sel], t[sel], axis=0), np.trapezoid(B[sel], t[sel], axis=0))
        out["final e_F (-32 %)"] = rel(A[-1], B[-1])
        out["S_geo (1.59)"] = meta["stats"]["S_geo"]["reproduced"]["median"]
        out["_meta_consistency"] = abs(out["post-ramp Ibar_F (-38 %)"] - meta["stats"]["Ibar_F_post_ramp"]["reproduced"]["median_rel"])
    elif exp == "alanine":
        sel = (t >= 20.0 - 1e-9) & (t <= 100.0 + 1e-9)
        out["dI_F own window (+0.04 %, neutral)"] = rel(np.trapezoid(A[sel], t[sel], axis=0), np.trapezoid(B[sel], t[sel], axis=0))
        out["final e_F (-0.6 %)"] = rel(A[-1], B[-1])
        out["_meta_consistency"] = abs(out["dI_F own window (+0.04 %, neutral)"] - meta["stats"]["dI_F_own_window_20_100ps"]["reproduced"]["median"])
    elif exp == "wca":
        out["I_F (+9.9 %)"] = rel(np.trapezoid(A, t, axis=0), np.trapezoid(B, t, axis=0))
        out["final e_F (+131 %)"] = rel(A[-1], B[-1])
        out["_meta_consistency"] = abs(100 * out["I_F (+9.9 %)"] - meta["stats"]["I_F"]["reproduced"]["median"])
    elif exp == "meta":
        out["I_F"] = rel(np.trapezoid(A, t, axis=0), np.trapezoid(B, t, axis=0))
        out["final e_F"] = rel(A[-1], B[-1])
        out["_meta_consistency"] = abs(out["I_F"] - meta["stats"]["I_F"]["median"])
    return out


def validate(exp, rows):
    mp = os.path.join(C.META_DIR, f"{exp}.json")
    if not check(rows, os.path.exists(mp), f"{exp}: meta file present"):
        return None
    meta = json.load(open(mp))
    grids, seeds_ref = [], None
    for m in C.METRICS:
        p = os.path.join(C.DATA_DIR, f"{exp}_{m}.csv")
        if not check(rows, os.path.exists(p), f"{exp}_{m}.csv present"):
            continue
        with open(p) as fh:
            head = fh.readline().strip().split(",")
        check(rows, tuple(head) == C.COLUMNS, f"{exp}_{m}.csv header", ",".join(head))
        _, a = C.read_csv(p)
        check(rows, a.shape[1] == 7, f"{exp}_{m}.csv seven numeric columns", str(a.shape))
        nonfin = int((~np.isfinite(a)).sum())
        check(rows, nonfin == 0, f"{exp}_{m}.csv all values finite", f"{nonfin} non-finite")
        nonpos = int((a[:, 1:] <= 0).sum())
        check(rows, nonpos == 0, f"{exp}_{m}.csv strictly positive (log axis)", f"{nonpos} non-positive")
        t = a[:, 0]
        check(rows, bool(np.all(np.diff(t) > 0)), f"{exp}_{m}.csv t strictly increasing")
        check(rows, abs(t[-1] - T_FINAL[exp]) < 1e-9, f"{exp}_{m}.csv final t", f"{t[-1]:g} (expected {T_FINAL[exp]:g})")
        check(rows, 100 <= t.size <= 500 or exp == "wca", f"{exp}_{m}.csv row count", f"{t.size} rows"
              + (" (WCA saved only 49 checkpoints; documented)" if exp == "wca" else ""))
        q_ok = bool(np.all(a[:, 2] <= a[:, 1] + 1e-15) and np.all(a[:, 1] <= a[:, 3] + 1e-15)
                    and np.all(a[:, 5] <= a[:, 4] + 1e-15) and np.all(a[:, 4] <= a[:, 6] + 1e-15))
        check(rows, q_ok, f"{exp}_{m}.csv q25 <= median <= q75 (both arms)")
        grids.append(t)
        # per-seed consistency
        ta, A, sa = per_seed(exp, m, "abf")
        tf, B, sf = per_seed(exp, m, "fr")
        check(rows, sa == sf and len(sa) == len(set(sa)) == N_SEEDS[exp] == meta["n_seeds"], f"{exp}_{m} seeds matched, unique, n = {N_SEEDS[exp]}",
              f"{sa[0]}..{sa[-1]}")
        if seeds_ref is None:
            seeds_ref = sa
        check(rows, sa == seeds_ref == meta["seeds"], f"{exp}_{m} same seed set as the other panels and the meta")
        check(rows, np.array_equal(ta, t) and np.array_equal(tf, t), f"{exp}_{m} per-seed time grid = CSV grid")
        qa, qf = C.quartiles(A), C.quartiles(B)
        rec = np.column_stack([t, qa[1], qa[0], qa[2], qf[1], qf[0], qf[2]])
        dev = float(np.max(np.abs(rec - a) / np.maximum(np.abs(a), 1e-300)))
        check(rows, dev < 1e-8, f"{exp}_{m}.csv = quartiles of the per-seed file", f"max rel dev {dev:.1e}")
        check(rows, bool(np.all(np.isfinite(A)) and np.all(np.isfinite(B))), f"{exp}_{m} per-seed values finite (no missing seed record)")
        # pre-FR identity of the paired arms
        t_fr = meta["config"].get("fr_start_t")
        if exp in STATISTICAL_PAIRING and t_fr:
            ident = t <= STATISTICAL_PAIRING[exp] + 1e-9
            pre = t < t_fr - 1e-9
            d0 = float(np.max(np.abs(A[ident] - B[ident])))
            check(rows, d0 == 0.0, f"{exp}_{m} ABF == FR through t = {STATISTICAL_PAIRING[exp]:g} (before GPU-reduction divergence)",
                  f"max |diff| {d0:.1e}")
            rel_pre = float(np.median(np.abs(A[pre] - B[pre]) / np.abs(A[pre])))
            note(rows, f"{exp}_{m} pre-FR pairing is statistical, not bitwise (documented: docs/FR_START_TIMING.md, float64 "
                       f"scatter_add order-nondeterminism on CUDA)",
                 f"before t = {t_fr:g}: max |diff| {float(np.max(np.abs(A[pre] - B[pre]))):.2e}, median relative diff {rel_pre:.2e}")
        elif t_fr is not None and t_fr > 0:
            pre = t < t_fr - 1e-9
            d = float(np.max(np.abs(A[pre] - B[pre]))) if pre.any() else 0.0
            check(rows, d == 0.0, f"{exp}_{m} ABF == FR before the FR start t = {t_fr:g}", f"{int(pre.sum())} saves, max |diff| {d:.1e}")
    if grids:
        check(rows, all(np.array_equal(g, grids[0]) for g in grids), f"{exp}: the three panels share one time grid")
        rp = C.dpath(RAW_TIMES[exp][0])
        if os.path.exists(rp):
            rt = np.asarray(np.load(rp, allow_pickle=True)[RAW_TIMES[exp][1]], float)
            ok = rt.size == grids[0].size and np.allclose(rt, grids[0], rtol=0, atol=1e-9)
            check(rows, ok, f"{exp}: every CSV row is a raw saved checkpoint (no fabricated / interpolated times)",
                  f"{rt.size} raw saves vs {grids[0].size} rows")
        else:
            note(rows, f"{exp}: raw checkpoint comparison", f"raw file not available here ({RAW_TIMES[exp][0]})")
    u = meta.get("units", {})
    check(rows, all(k in u for k in ("t", "eF", "eFp", "exi")), f"{exp}: units recorded for t and the three errors")
    check(rows, "marginal_type" in meta and "reference" in meta and "definitions" in meta, f"{exp}: marginal type, reference and definitions recorded")
    prov = meta["provenance"]
    check(rows, not prov["scoring_code_tracked_files_modified"], f"{exp}: scoring code = committed code",
          f"commit {prov['scoring_code_commit'][:10]}")
    # statistics
    st = recompute(exp, meta, rows)
    check(rows, st.pop("_meta_consistency") < 1e-9, f"{exp}: paired statistic recomputed from the per-seed files = exporter's")
    for name, exp_v, tol in AUDIT[exp]:
        v = st[name]
        check(rows, abs(v - exp_v) <= tol, f"{exp}: audit {name}", f"reproduced {v:+.4f}, reference {exp_v:+.4f} (tol {tol})")
    for k, v in st.items():
        if all(k != n for n, _, _ in AUDIT[exp]):
            note(rows, f"{exp}: {k}", f"{v:+.4f}")
    return meta


def main():
    rows, metas = [], {}
    for exp in C.EXPERIMENTS:
        r = []
        metas[exp] = validate(exp, r)
        rows.append((exp, r))
    n_fail = sum(1 for _, r in rows for x in r if x["status"] == "FAIL")
    n_pass = sum(1 for _, r in rows for x in r if x["status"] == "PASS")
    json.dump(C.json_safe(dict(n_pass=n_pass, n_fail=n_fail, checks={e: r for e, r in rows})),
              open(os.path.join(C.PKG, "validation_results.json"), "w"), indent=1)
    L = ["# Validation report (generated by `scripts/validate_exports.py`)", "",
         f"**{n_pass} checks passed, {n_fail} failed.** INFO lines are reported values, not checks.", ""]
    for exp, r in rows:
        nf = sum(1 for x in r if x["status"] == "FAIL")
        L += [f"## {exp} — {'PASS' if nf == 0 and r else 'FAIL' if r else 'MISSING'}", "", "| check | status | detail |", "|---|---|---|"]
        L += [f"| {x['check']} | {x['status']} | {x['detail']} |" for x in r]
        L.append("")
    extra = os.path.join(HERE_, "validation_notes.md")
    if os.path.exists(extra):
        L += ["", open(extra).read()]
    open(os.path.join(C.PKG, "VALIDATION_REPORT.md"), "w").write("\n".join(L) + "\n")
    print(f"{n_pass} PASS, {n_fail} FAIL")
    for exp, r in rows:
        for x in r:
            if x["status"] == "FAIL":
                print(f"  FAIL {x['check']}: {x['detail']}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
