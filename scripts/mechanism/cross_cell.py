#!/usr/bin/env python
"""Preregistered cross-cell contrasts of the mechanism campaign (docs/mechanism/SCIENTIFIC_PLAN.md section 6).

    python scripts/mechanism/cross_cell.py [--cells-root configs/mechanism/cells] [--analysis-root DIR]
                                           [--out-dir results/mechanism/synthesis] [--n-boot 10000]

Reads ONLY each cell's analysis summary (scripts/equal_budget/analyze_ladder.py --system gateway_family): the
per-seed values per_N[N][abf|fr][metric].per_seed.  Nothing is re-scored.  For cells a and b of one experiment at the
same N, with seeds coupled across cells (same seeds, initial conditions and Langevin noise):

    L_c(s)   = log(I^FR_c(s) / I^ABF_c(s))                   (seed s, cell c; I = Ibar of the metric)
    Delta_ab = median over the seeds common to a and b of [L_a(s) - L_b(s)]
    ratio    = exp(Delta_ab): < 1 means a has the LARGER FR gain (FR/ABF lower in a than in b)
    CI       = 2.5 / 97.5 % percentiles of the median over n_boot seed-bootstrap resamples (paired: one index draw
               resamples the seed pairs; numpy default_rng(eqb_metrics.BOOT_SEED), one fresh generator per contrast)
    p        = two-sided bootstrap p = min(1, 2 min((#{D* <= 0} + 1)/(n_boot + 1), (#{D* >= 0} + 1)/(n_boot + 1)))
               (D* = the resampled medians; the +1 keeps p >= 2/(n_boot + 1) instead of an impossible 0).  Not
               calibrated for very few seeds (n = 3 all of one sign gives p = 2/(n_boot + 1)): with n < 10 paired seeds
               (fixtures only; production has 32) the verdict makes no claim and the exact sign test is shown
    Holm     within each frozen family: Experiment I = C1 alpha0 vs alpha1, C2 shift vs alpha1, C3 shift vs alpha0,
             C4 alpha0.5 vs alpha1 at every N of the experiment (3 N: 12 tests); Experiment II = D1 lam0.1 vs lam1 at
             every N (2 N: 2 tests).  A test that cannot be computed (cell or seeds missing) enters Holm with p = 1 and
             the family is marked incomplete.  'differs' = Holm-adjusted p < 0.05.
    equivalent iff the 95 % CI of exp(Delta) lies inside [0.90, 1.11] (inclusive).
Primary metric Ibar_F; the same contrasts on Ibar_TV_half and Ibar_Fp_stat as secondary results (Holm within each
metric's family).  Experiment II trend: lam0.25 vs lam1 and lam0.5 vs lam1 at each N, unadjusted, labelled 'trend'.
Descriptive companions per contrast: mean of the differences, wins (seeds with L_a < L_b), exact sign-test p,
Wilcoxon signed-rank p; per cell and N: median L, median G = exp(L) - 1, n seeds.

Checks before computing (an error stops the script; check_summary):
  * identity: every summary is a gateway_family summary of exactly its cell's CURRENT config -- experiment / cell,
    B, h, seeds, N ladder, thresholds, metrics_version, and the cell block's engine_version, results_dir,
    reference_file, model and reuse equal the config's (a summary left over from the reuse mode after
    make_cell_configs.py --rerun-reuse-cells, or from another engine / model, is refused); the summary's reference
    path and sha256 are the config's frozen reference file's; its scoring-code digest and config digest are the
    current ones (eqb_family.code_provenance / config_digest);
  * freshness (as plot_synthesis / the audit): the complete run files on disk are exactly the summary's complete
    seeds, and every one of them has a valid per-run cache entry (analyze_ladder.cache_entry_check: run-file
    content, metrics version, config, code, reference) -- skipped only with --no-verify-runs (synthetic inputs;
    recorded as freshness_verified false);
  * reuse cells: the plan's bitwise-reuse gate record licenses the reuse (eqb_family.reuse_gate_problems;
    scripts/mechanism/reuse_gate.py);
  * the primary reference digest is identical in every cell.
Writes <out-dir>/cross_cell.json (strict JSON) and cross_cell.md.  Missing cells are reported, never dropped.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "equal_budget"))
import eqb_metrics as M  # noqa: E402
import eqb_family as MF  # noqa: E402
import analyze_ladder as AL  # noqa: E402  (run paths, file status, per-run cache freshness)

SCHEMA = "mechanism_cross_cell/1"
CELLS_ROOT = os.path.join(ROOT, "configs", "mechanism", "cells")
OUT_DIR = os.path.join(ROOT, "results", "mechanism", "synthesis")
EQUIV = (0.90, 1.11)
ALPHA = 0.05
METRICS = (("Ibar_F", "primary"), ("Ibar_TV_half", "secondary"), ("Ibar_Fp_stat", "secondary"))
# frozen families (SCIENTIFIC_PLAN section 6; configs/mechanism/*.json primary_contrasts / multiplicity)
FAMILIES = {
    "matched_free_energy": dict(
        label="Experiment I (matched free energy)", frozen_size=12,
        primary=(("C1", "alpha0", "alpha1"), ("C2", "shift", "alpha1"), ("C3", "shift", "alpha0"),
                 ("C4", "alpha0.5", "alpha1")),
        trend=()),
    "conditional_relaxation": dict(
        label="Experiment II (transverse mobility)", frozen_size=2,
        primary=(("D1", "lam0.1", "lam1"),),
        trend=(("T1", "lam0.25", "lam1"), ("T2", "lam0.5", "lam1"))),
}


class CrossCellError(RuntimeError):
    pass


# ============================================================================================ statistics
def fnum(v):
    if v is None:
        return math.nan
    if isinstance(v, str):
        return float(v)
    return float(v)


def log_ratios(abf, fr):
    """abf, fr: {seed: Ibar} of one cell and N -> ({seed: log(FR/ABF)} for seeds with both values finite and > 0,
    {seed: reason} excluded)."""
    out, excl = {}, {}
    for s in sorted(set(abf) | set(fr), key=int):
        a, f = fnum(abf.get(s)), fnum(fr.get(s))
        if s not in abf or s not in fr:
            excl[int(s)] = "one arm missing"
        elif not (math.isfinite(a) and math.isfinite(f)) or a <= 0 or f <= 0:
            excl[int(s)] = f"non-finite or non-positive (ABF {a}, FR {f})"
        else:
            out[int(s)] = math.log(f / a)
    return out, excl


def bootstrap_p(boot, n_boot):
    """Two-sided percentile-bootstrap p of H0: Delta = 0 (with the +1 correction)."""
    boot = np.asarray(boot, float)
    lo = (np.sum(boot <= 0) + 1.0) / (n_boot + 1.0)
    hi = (np.sum(boot >= 0) + 1.0) / (n_boot + 1.0)
    return float(min(1.0, 2.0 * min(lo, hi)))


def contrast(La, Lb, n_boot=M.N_BOOT, seed=M.BOOT_SEED):
    """Paired cross-cell contrast of two {seed: L} dicts (module docstring)."""
    common = sorted(set(La) & set(Lb))
    only = dict(a_only=sorted(set(La) - set(Lb)), b_only=sorted(set(Lb) - set(La)))
    if not common:
        return dict(n=0, seeds=[], **only, Delta=math.nan, ratio=math.nan, ratio_ci95=[math.nan, math.nan],
                    Delta_ci95=[math.nan, math.nan], p=math.nan, mean_diff=math.nan, wins_a=0, losses_a=0, ties=0,
                    sign_test_p=math.nan, wilcoxon_p=math.nan, equivalent=False, per_seed={})
    d = np.array([La[s] - Lb[s] for s in common], float)
    n = d.size
    rng = np.random.default_rng(seed)
    boot = np.median(d[rng.integers(0, n, size=(int(n_boot), n))], axis=1)
    D = float(np.median(d))
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    wins, losses = int(np.sum(d < 0)), int(np.sum(d > 0))
    try:
        from scipy.stats import wilcoxon
        wp = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except Exception:  # noqa: BLE001
        wp = math.nan
    rlo, rhi = math.exp(lo), math.exp(hi)
    return dict(n=n, seeds=[int(s) for s in common], **only, Delta=D, Delta_ci95=[lo, hi], ratio=math.exp(D),
                ratio_ci95=[rlo, rhi], p=bootstrap_p(boot, int(n_boot)), mean_diff=float(np.mean(d)),
                wins_a=wins, losses_a=losses, ties=int(n - wins - losses),
                sign_test_p=M.sign_test_p(wins, losses), wilcoxon_p=wp,
                equivalent=bool(EQUIV[0] <= rlo and rhi <= EQUIV[1]),
                per_seed={int(s): float(x) for s, x in zip(common, d)})


def holm(pvals, m=None):
    """Holm step-down adjusted p of a list (NaN = not computed -> treated as 1); m = family size (default len)."""
    p = np.array([1.0 if (v is None or not math.isfinite(v)) else float(v) for v in pvals], float)
    m = int(m or p.size)
    if m < p.size:
        raise CrossCellError(f"Holm family size {m} < {p.size} tests")
    order = np.argsort(p, kind="stable")
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return [float(x) for x in adj]


SMALL_N = 10        # below this the percentile-bootstrap p of a median is not calibrated (n = 3, all seeds one sign:
                    # p = 2/(n_boot + 1) while the exact sign test gives 0.25): flagged, sign test shown


def trend_verdict(c):
    """Verdict of a trend row (outside every Holm family; no 'differs' claim): equivalence band, whether the 95 % CI
    of Delta excludes 0, and the unadjusted p."""
    if c["n"] == 0:
        return "NOT AVAILABLE"
    if c["n"] < SMALL_N:
        return f"trend: n = {c['n']} < {SMALL_N}: bootstrap p uncalibrated, no claim (sign-test p {c['sign_test_p']:.3g})"
    lo, hi = c["Delta_ci95"]
    side = ("CI excludes 0, a has the larger FR gain" if hi < 0 else
            "CI excludes 0, b has the larger FR gain" if lo > 0 else "CI includes 0")
    return (f"trend (unadjusted, no Holm): {'equivalent; ' if c['equivalent'] else ''}{side}; "
            f"unadjusted p = {c['p']:.3g}")


def verdict(c, p_holm):
    if c["n"] == 0:
        return "NOT AVAILABLE"
    if c["n"] < SMALL_N:
        return f"n = {c['n']} < {SMALL_N}: bootstrap p uncalibrated, no claim (sign-test p {c['sign_test_p']:.3g})"
    diff = p_holm is not None and math.isfinite(p_holm) and p_holm < ALPHA
    if c["equivalent"] and diff:
        return "equivalent (within [0.90, 1.11]) but differs (Holm)"
    if c["equivalent"]:
        return "equivalent"
    if diff:
        return "differs (Holm): " + ("a has the larger FR gain" if c["Delta"] < 0 else "b has the larger FR gain")
    return "inconclusive"


# ============================================================================================ inputs
def load_cells(cells_root, analysis_root=None):
    """{experiment: {cell: dict(config, P, summary_path, S or None, why)}} from the cell configs."""
    out = {}
    for p in sorted(glob.glob(os.path.join(cells_root, "*", "*.json"))):
        P = json.load(open(p))
        if not MF.is_cell(P):
            continue
        if analysis_root:          # <root>/<experiment>/<cell>/analysis, else <root>/<cell>/analysis
            a = os.path.join(analysis_root, P["experiment"], P["cell"], "analysis")
            if not os.path.isdir(a):
                a = os.path.join(analysis_root, P["cell"], "analysis")
        else:
            a = MF.cell_analysis_dir(P)
        sp = os.path.join(a, "summary.json")
        rec = dict(config=os.path.abspath(p), P=P, summary_path=sp, S=None, why=None)
        if os.path.exists(sp):
            rec["S"] = json.load(open(sp))
        else:
            rec["why"] = f"no summary at {sp}"
        out.setdefault(P["experiment"], {})[P["cell"]] = rec
    return out


CELL_IDENTITY = ("engine_version", "results_dir", "reference_file", "model", "reuse")


def _jsonable(v):
    return json.loads(json.dumps(v, default=str))


def check_summary(rec):
    """Identity problems that make a summary unusable for its cell's CURRENT config (empty = OK)."""
    S, P = rec["S"], rec["P"]
    probs = []
    if S.get("system") != MF.FAMILY:
        probs.append(f"system {S.get('system')!r}")
    cl = S.get("cell") or {}
    if cl.get("experiment") != P["experiment"] or cl.get("cell") != P["cell"]:
        probs.append(f"summary cell {cl.get('experiment')}/{cl.get('cell')} != config {P['experiment']}/{P['cell']}")
    for k in CELL_IDENTITY:          # the run tree, engine, reference and model the summary was made from
        if _jsonable(cl.get(k)) != _jsonable(P.get(k)):
            probs.append(f"summary cell {k} {cl.get(k)!r} != config {P.get(k)!r}")
    pl = S.get("plan") or {}
    want = dict(B=int(P["B"]), h=float(P["engine_cfg"]["h"]), N_ladder=[int(n) for n in P["N_ladder"]],
                seeds=[int(s) for s in P["seeds"]], thresholds=P["thresholds"])
    for k, v in want.items():
        if pl.get(k) != v:
            probs.append(f"plan {k} {pl.get(k)!r} != config {v!r}")
    if S.get("metrics_version") != M.METRICS_VERSION:
        probs.append(f"metrics_version {S.get('metrics_version')!r}")
    ref = S.get("reference") or {}
    rf = MF._abs(P["reference_file"])
    if not ref.get("path") or os.path.normpath(os.path.join(ROOT, ref["path"])) != os.path.normpath(rf):
        probs.append(f"summary reference {ref.get('path')!r} is not the config's {P['reference_file']!r}")
    if not os.path.exists(rf):
        probs.append(f"frozen reference {rf} does not exist")
    elif ref.get("sha256") != M.sha256_file(rf):
        probs.append(f"summary was scored against reference sha256 {str(ref.get('sha256'))[:12]}, the frozen file is "
                     f"{M.sha256_file(rf)[:12]}")
    prov = S.get("provenance") or {}
    code = MF.code_provenance(MF.FAMILY)["digest"]
    if (prov.get("code") or {}).get("digest") != code:
        probs.append(f"scoring code changed since the analysis ({(prov.get('code') or {}).get('digest')} -> {code})")
    cd = MF.config_digest(P)
    if prov.get("config_digest") != cd:
        probs.append(f"config digest {prov.get('config_digest')} != current {cd} (the cell config changed)")
    return probs


def check_freshness(rec):
    """Run-tree problems (empty = fresh): complete files on disk == the summary's complete seeds, and every complete
    run's per-run cache entry is valid for the current file, metrics version, config, code and reference (the same
    test as plot_synthesis.SysData.check_freshness)."""
    S, P = rec["S"], rec["P"]
    probs = []
    root = MF.cell_results_root(P)
    cache_dir = os.path.join(os.path.dirname(rec["summary_path"]), "run_metrics")
    prov = AL.provenance(MF.FAMILY, P)
    for r in AL.plan(P):
        N = r["N"]
        for m in r["methods"]:
            st = ((((S.get("status") or {}).get(str(N)) or {}).get("methods") or {}).get(m) or {})
            comp_sum = {int(s) for s in st.get("complete") or []}
            comp_now = set()
            for s in P["seeds"]:
                p = AL.run_path(root, P, N, int(s), m)
                if AL.file_status(p)[0] == "complete":
                    comp_now.add(int(s))
            if comp_now - comp_sum:
                probs.append(f"N {N} {m}: seeds {sorted(comp_now - comp_sum)} completed after the analysis")
            if comp_sum - comp_now:
                probs.append(f"N {N} {m}: seeds {sorted(comp_sum - comp_now)} in the summary but not complete on disk")
            for s in sorted(comp_sum & comp_now):
                p = AL.run_path(root, P, N, s, m)
                _, _, pr = AL.cache_entry_check(p, MF.FAMILY, P, cache_dir, dict(N=N, seed=s, method=m), prov)
                if pr:
                    probs.append(f"N {N} seed {s} {m}: " + "; ".join(pr))
    return probs


def per_seed(S, N, method, metric):
    b = (((S.get("per_N") or {}).get(str(N)) or {}).get(method) or {}).get(metric)
    return {int(k): v for k, v in ((b or {}).get("per_seed") or {}).items()}


# ============================================================================================ main computation
def analyse(cells, n_boot=M.N_BOOT, production=True, verify_runs=True):
    """cells: load_cells() output -> the cross_cell.json document (without generated fields).  verify_runs=False
    skips the run-tree freshness check (synthetic summaries without run files only; recorded in the document)."""
    doc = dict(schema=SCHEMA, n_boot=int(n_boot), boot_seed=M.BOOT_SEED, equivalence_band=list(EQUIV), alpha=ALPHA,
               metrics={m: g for m, g in METRICS}, cells={}, experiments={}, warnings=[],
               freshness_verified=bool(verify_runs))
    if not verify_runs:
        doc["warnings"].append("run-tree freshness NOT verified (--no-verify-runs): the summaries were not checked "
                               "against the run files and per-run caches")
    digests = {}
    for exp, cs in cells.items():
        for name, rec in cs.items():
            info = dict(config=os.path.relpath(rec["config"], ROOT) if rec["config"].startswith(ROOT) else rec["config"],
                        summary=rec["summary_path"], available=rec["S"] is not None, why=rec["why"])
            if rec["S"] is not None:
                probs = check_summary(rec)
                if probs:
                    raise CrossCellError(f"{exp}/{name}: summary {rec['summary_path']} does not belong to its config: "
                                         + "; ".join(probs))
                if verify_runs:
                    stale = check_freshness(rec)
                    if stale:
                        raise CrossCellError(f"{exp}/{name}: summary {rec['summary_path']} is STALE ({len(stale)} "
                                             f"issues, e.g. {stale[:3]}): re-run analyze_ladder.py on the cell")
                gate = MF.reuse_gate_problems(rec["P"])
                if gate:
                    raise CrossCellError(f"{exp}/{name}: reuse cell without a licensing bitwise-reuse gate (plan "
                                         f"section 5): " + "; ".join(gate))
                info.update(summary_sha256=M.sha256_file(rec["summary_path"]),
                            generated_utc=rec["S"].get("generated_utc"), n_boot=rec["S"].get("n_boot"),
                            engine_version=(rec["S"].get("cell") or {}).get("engine_version"),
                            model=(rec["S"].get("cell") or {}).get("model"),
                            results_dir=(rec["S"].get("cell") or {}).get("results_dir"),
                            code_digest=((rec["S"].get("provenance") or {}).get("code") or {}).get("digest"),
                            reuse_gate=(MF.reuse_gate_status(rec["P"]) if rec["P"].get("reuse") else None),
                            fixture=bool("_fixture_of" in rec["P"]))
                dg = ((rec["S"].get("reference") or {}).get("primary_reference") or {}).get("digest")
                digests[f"{exp}/{name}"] = dg
                if rec["S"].get("n_boot") != M.N_BOOT:
                    doc["warnings"].append(f"{exp}/{name}: summary n_boot {rec['S'].get('n_boot')} != {M.N_BOOT}")
                for w in rec["S"].get("warnings") or []:
                    if "complete (missing" in str(w):
                        doc["warnings"].append(f"{exp}/{name}: {w}")
            doc["cells"][f"{exp}/{name}"] = info
    if len(set(digests.values())) > 1 or None in digests.values():
        raise CrossCellError(f"the primary reference is not identical in every cell: {digests}")
    doc["primary_reference_digest"] = next(iter(digests.values())) if digests else None

    for exp, spec in FAMILIES.items():
        cs = cells.get(exp, {})
        Ns = None
        for rec in cs.values():
            nl = [int(n) for n in rec["P"]["N_ladder"]]
            if Ns is None:
                Ns = nl
            elif nl != Ns:
                raise CrossCellError(f"{exp}: cells have different N ladders ({Ns} vs {nl})")
        Ns = Ns or []
        E = dict(label=spec["label"], N=Ns, cells_present=sorted(n for n, r in cs.items() if r["S"] is not None),
                 cells_missing=sorted(set(n for c in spec["primary"] + spec["trend"] for n in c[1:])
                                      - {n for n, r in cs.items() if r["S"] is not None}),
                 per_cell={}, metrics={})
        size = len(spec["primary"]) * len(Ns)
        if production and Ns and size != spec["frozen_size"]:
            raise CrossCellError(f"{exp}: family of {size} tests, the plan freezes {spec['frozen_size']}")
        E["family_size"] = size
        # per-cell log ratios
        L = {}
        for metric, _ in METRICS:
            for name, rec in cs.items():
                if rec["S"] is None:
                    continue
                for N in Ns:
                    lr, excl = log_ratios(per_seed(rec["S"], N, "abf", metric), per_seed(rec["S"], N, "fr", metric))
                    L[(metric, name, N)] = lr
                    v = np.array(list(lr.values()), float)
                    E["per_cell"].setdefault(name, {}).setdefault(str(N), {})[metric] = dict(
                        n=int(v.size), median_L=float(np.median(v)) if v.size else math.nan,
                        median_G=float(math.expm1(np.median(v))) if v.size else math.nan, excluded=excl)
        for metric, grp in METRICS:
            rows = []
            for kind, lst in (("primary", spec["primary"]), ("trend", spec["trend"])):
                for cid, a, b in lst:
                    for N in Ns:
                        La, Lb = L.get((metric, a, N)), L.get((metric, b, N))
                        if La is None or Lb is None:
                            c = contrast({}, {}, n_boot)
                            c["missing"] = [x for x, y in ((a, La), (b, Lb)) if y is None]
                        else:
                            c = contrast(La, Lb, n_boot)
                        c.update(id=cid, a=a, b=b, N=int(N), kind=kind,
                                 label=f"{cid} {a} vs {b} at N {N}")
                        rows.append(c)
            prim = [r for r in rows if r["kind"] == "primary"]
            adj = holm([r["p"] for r in prim], m=max(size, len(prim))) if prim else []
            for r, ph in zip(prim, adj):
                r["p_holm"] = ph
                r["verdict"] = verdict(r, ph)
            for r in rows:
                if r["kind"] == "trend":
                    r["p_holm"] = None
                    r["verdict"] = trend_verdict(r)
            E["metrics"][metric] = dict(group=grp, family_complete=all(r["n"] > 0 for r in prim),
                                        n_tests=len(prim), contrasts=rows)
        doc["experiments"][exp] = E
    return doc


# ============================================================================================ markdown
def f(v, nd=3):
    v = fnum(v)
    if math.isnan(v):
        return "--"
    return f"{v:.{nd}g}"


def md(doc):
    L = ["# Mechanism campaign: preregistered cross-cell contrasts", "",
         f"Generated {doc.get('generated_utc')} by scripts/mechanism/cross_cell.py ({SCHEMA}); {doc['n_boot']} seed-"
         f"bootstrap resamples (seed {doc['boot_seed']}).", "",
         "Delta_ab = median over seeds of [log(I^FR/I^ABF)_a - log(I^FR/I^ABF)_b] (seeds paired across cells); ratio = "
         "exp(Delta): **< 1 means a has the larger FR gain**.  CI: seed bootstrap of the median; p: two-sided "
         "percentile-bootstrap p (+1 correction); Holm within the frozen family (Exp I: 12 tests, Exp II: 2); "
         f"equivalent iff the 95 % CI of exp(Delta) lies in [{EQUIV[0]}, {EQUIV[1]}].", ""]
    if doc.get("fixture"):
        L += ["**FIXTURE INPUT: not production data.**", ""]
    L += ["## Cells", "", "| cell | available | engine | model | summary generated | n_boot |", "|---|---|---|---|---|---|"]
    for k, c in doc["cells"].items():
        L.append(f"| {k} | {'yes' if c['available'] else 'NO: ' + str(c['why'])} | {c.get('engine_version') or '--'} | "
                 f"{c.get('model') or '--'} | {c.get('generated_utc') or '--'} | {c.get('n_boot') or '--'} |")
    L += ["", f"Primary reference digest (identical in every cell): `{doc.get('primary_reference_digest')}`", ""]
    for exp, E in doc["experiments"].items():
        L += [f"## {E['label']}", "", f"N = {E['N']}; family size {E['family_size']}; cells present "
              f"{E['cells_present']}; missing {E['cells_missing'] or 'none'}.", ""]
        for metric, blk in E["metrics"].items():
            L += [f"### {metric} ({blk['group']}){'' if blk['family_complete'] else ' -- FAMILY INCOMPLETE'}", "",
                  "| contrast | N | n | exp(Delta) [95 % CI] | p | p Holm | wins a / n | sign p | Wilcoxon p | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for r in blk["contrasts"]:
                L.append(f"| {r['id']} {r['a']} vs {r['b']}{' (trend)' if r['kind'] == 'trend' else ''} | {r['N']} | "
                         f"{r['n']} | {f(r['ratio'], 4)} [{f(r['ratio_ci95'][0], 4)}, {f(r['ratio_ci95'][1], 4)}] | "
                         f"{f(r['p'])} | {f(r['p_holm']) if r['p_holm'] is not None else 'n/a'} | "
                         f"{r['wins_a']}/{r['n']} | {f(r['sign_test_p'])} | {f(r['wilcoxon_p'])} | {r['verdict']} |")
            L.append("")
        L += ["### Per cell: median log(I^FR/I^ABF) and median G = exp(L) - 1", "",
              "| cell | N | " + " | ".join(f"{m} median G (n)" for m, _ in METRICS) + " |",
              "|---|---|" + "---|" * len(METRICS)]
        for name, byN in sorted(E["per_cell"].items()):
            for N, d in byN.items():
                L.append(f"| {name} | {N} | " + " | ".join(
                    f"{100 * fnum(d[m]['median_G']):+.1f} % ({d[m]['n']})" if m in d and d[m]["n"] else "--"
                    for m, _ in METRICS) + " |")
        L.append("")
    if doc["warnings"]:
        L += ["## Warnings", ""] + [f"* {w}" for w in doc["warnings"]]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-root", default=CELLS_ROOT)
    ap.add_argument("--analysis-root", default=None,
                    help="read <root>/<experiment>/<cell>/analysis or <root>/<cell>/analysis instead of each analysis_dir")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=M.N_BOOT)
    ap.add_argument("--no-verify-runs", action="store_true",
                    help="skip the run-tree freshness check (synthetic summaries without run files only)")
    a = ap.parse_args(argv)
    cells = load_cells(a.cells_root, a.analysis_root)
    if not cells:
        raise CrossCellError(f"no cell config under {a.cells_root}")
    fixture = any("_fixture_of" in r["P"] for cs in cells.values() for r in cs.values())
    doc = analyse(cells, a.n_boot, production=not fixture, verify_runs=not a.no_verify_runs)

    doc.update(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), fixture=fixture,
               cells_root=os.path.abspath(a.cells_root), analysis_root=a.analysis_root)
    out = os.path.abspath(a.out_dir)
    for d in MF.PROTECTED_OUTPUT_DIRS:
        if os.path.realpath(out) == os.path.realpath(d) or os.path.realpath(out).startswith(os.path.realpath(d) + os.sep):
            raise CrossCellError(f"refusing to write into the equal-budget tree {d}")
    os.makedirs(out, exist_ok=True)
    jp = os.path.join(out, "cross_cell.json")
    with open(jp + ".tmp", "w") as fh:
        json.dump(M.json_safe(doc), fh, indent=1, allow_nan=False)
    os.replace(jp + ".tmp", jp)
    with open(os.path.join(out, "cross_cell.md"), "w") as fh:
        fh.write(md(doc))
    for exp, E in doc["experiments"].items():
        for r in E["metrics"]["Ibar_F"]["contrasts"]:
            if r["kind"] == "primary":
                print(f"{exp} {r['label']}: exp(Delta) {f(r['ratio'], 4)} [{f(r['ratio_ci95'][0], 4)}, "
                      f"{f(r['ratio_ci95'][1], 4)}] p {f(r['p'])} Holm {f(r['p_holm'])} -> {r['verdict']}", flush=True)
    print(f"wrote {jp} and cross_cell.md", flush=True)
    return doc


if __name__ == "__main__":
    main()
