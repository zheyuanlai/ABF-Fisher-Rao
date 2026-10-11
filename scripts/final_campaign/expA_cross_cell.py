#!/usr/bin/env python
"""Experiment A cross-cell contrasts (docs/final_campaign/PREREGISTRATION.md sections 1.2, 1.6).

    python scripts/final_campaign/expA_cross_cell.py [--cells-dir configs/final_campaign/cells/expA]
        [--out-dir results/final_campaign/expA/synthesis] [--n-boot 10000] [--no-verify-runs] [--provisional]
        [--force] [--recompute] [--dry-run]

Reads ONLY each cell's analysis summary (scripts/equal_budget/analyze_ladder.py --system gateway_family --config
<cell>): per_N[N][abf|fr][metric].per_seed and the paired G blocks contrasts[N][group][metric].  Nothing is re-scored.
The statistics are those of scripts/mechanism/cross_cell.py, called by import (log_ratios, contrast, holm,
bootstrap_p, check_summary, check_freshness; that file is not edited and its hard-coded FAMILIES are not used):

    L_lam(s)  = log(Ibar^FR_lam(s) / Ibar^ABF_lam(s))                    (seed s; Ibar of the metric)
    Delta     = median over the seeds common to both cells of [L_a(s) - L_b(s)]   (seeds coupled across lambda)
    R         = exp(Delta): R < 1 means FR's performance relative to ABF is BETTER at lambda_a
    95 % CI   = 2.5 / 97.5 % percentiles of the median over n_boot paired seed-bootstrap resamples
                (numpy default_rng(20261010), one fresh generator per contrast: cross_cell.contrast)
    p         = two-sided bootstrap p (cross_cell.bootstrap_p, +1 correction)
    Holm      E1 (lambda 16 vs 1) at N 512 and N 128: one family of size 2; E2 (lambda 128 vs 1): a separate family
              of size 2.  A contrast that cannot be computed enters Holm with p = 1 (family marked incomplete).
    97.5 % CI the Holm-consistent CI of the first Holm step (m = 2: level 1 - 0.05/2), the 1.25 / 98.75 %
              percentiles of the SAME bootstrap draws, reported next to the 95 % CI of every E1 / E2 row of Ibar_F
    equivalent iff the 95 % CI of R lies in [0.90, 1.11] (inclusive)
Trend contrasts T1 (lambda 4 vs 1), T2 (16 vs 4), T3 (128 vs 16): unadjusted.  Secondary metrics final_e_F,
Ibar_Fp_stat and Ibar_TV_half: the same construction for E1, E2, T1-T3, all UNADJUSTED (labelled secondary).
Decomposition (with every contrast row): median over the same seeds of log(Ibar^ABF_a / Ibar^ABF_b) and of
log(Ibar^FR_a / Ibar^FR_b) (cross_cell.contrast on the log values: ratio, 95 % CI, p).  A contrast with fewer than
10 paired seeds makes no claim (PREREGISTRATION 1.6).
Per cell and N, for every metric: n pairs, median L with its bootstrap CI (exp: the FR/ABF ratio of figure A3), the
paired G block of the summary (median, 95 % CI, wins), the mean-based companions (PREREGISTRATION 1.6: mean G and
mean L with percentile seed-bootstrap 95 % CIs) and the inf-aware seed medians of each arm with bootstrap CIs.
G_hi(lambda, N) = upper end of the 95 % CI of G(Ibar_F) (outcome rule input, PREREGISTRATION 1.7).

Checks before computing AND before skipping an up-to-date output (an error stops the script): every summary belongs
to its cell's CURRENT config and the current scoring code / reference (cross_cell.check_summary); freshness
(cross_cell.check_freshness: the complete run files on disk are exactly the summary's, every per-run cache entry
valid) unless --no-verify-runs; the primary reference digest is identical in every cell; the cells share one N
ladder, h and B.

complete = expA_analysis_common.completeness(): production design, no fixture, freshness verified, every planned run
in every summary, n_boot = 10 000 here and in every summary, every cell's completeness audit PASS on the current
summary / config with only frozen-engine runs, and no --provisional (expA_analyze.py passes --provisional when it
runs the synthesis over a failed audit with --allow-audit-fail).  The reasons are written to 'completeness'.

Writes <out-dir>/cross_cell.json (strict JSON; input_signature, complete, completeness, provenance) and cross_cell.md
(placeholder JSON first, then the markdown, then the final JSON).  Output policy: expA_analysis_common.decide_write
(same signature -> skipped; a COMPLETE output of different inputs is never overwritten without --force).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import expA_analysis_common as C  # noqa: E402
import numpy as np  # noqa: E402
import cross_cell as X  # noqa: E402  (scripts/mechanism/cross_cell.py: statistics + summary checks, by import)
import eqb_metrics as M  # noqa: E402

SCHEMA = "final_expA_cross_cell/2"
METRICS = (("Ibar_F", "primary"), ("final_e_F", "secondary"), ("Ibar_Fp_stat", "secondary"),
           ("Ibar_TV_half", "secondary"))
PRIMARY_METRIC = "Ibar_F"
FAMILIES = (("E1", 16.0, 1.0, "primary (task-specified)"), ("E2", 128.0, 1.0, "extension (separate family)"))
TRENDS = (("T1", 4.0, 1.0), ("T2", 16.0, 4.0), ("T3", 128.0, 16.0))
FAMILY_SIZE = 2
HOLM_STEP1_PCT = (100 * (C.ALPHA / FAMILY_SIZE) / 2, 100 * (1 - (C.ALPHA / FAMILY_SIZE) / 2))   # (1.25, 98.75)
CODE_FILES = [os.path.abspath(__file__), os.path.join(HERE, "expA_analysis_common.py"),
              os.path.join(C.ROOT, "scripts", "mechanism", "cross_cell.py"),
              os.path.join(C.ROOT, "scripts", "equal_budget", "eqb_metrics.py"),
              os.path.join(C.ROOT, "scripts", "equal_budget", "eqb_family.py"),
              os.path.join(C.ROOT, "scripts", "equal_budget", "analyze_ladder.py")]


# ============================================================================================ statistics
def boot_draws(d, n_boot=C.N_BOOT, seed=C.BOOT_SEED):
    """The resampled medians of cross_cell.contrast (identical draws: fresh default_rng(seed), one index matrix)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    return np.median(d[rng.integers(0, d.size, size=(int(n_boot), d.size))], axis=1)


def contrast_row(La, Lb, n_boot=C.N_BOOT, step1_ci=False, abf=None, fr=None):
    """cross_cell.contrast of two {seed: L} dicts plus (step1_ci) the Holm-consistent 97.5 % CI from the same draws
    and (abf, fr = ({seed: Ibar_a}, {seed: Ibar_b}) pairs) the decomposition over the contrast's seeds."""
    c = X.contrast(La, Lb, n_boot, C.BOOT_SEED)
    c["claim_allowed"] = bool(c["n"] >= C.SMALL_N)
    if c["n"] > 0:
        d = np.array([La[s] - Lb[s] for s in c["seeds"]], float)
        b = boot_draws(d, n_boot)
        lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
        if [lo, hi] != list(c["Delta_ci95"]):                       # the draws must be cross_cell's own
            raise C.ExpAError(f"bootstrap draws differ from cross_cell.contrast ({[lo, hi]} vs {c['Delta_ci95']})")
        if step1_ci:
            l2, h2 = (float(np.percentile(b, q)) for q in HOLM_STEP1_PCT)
            c["Delta_ci975"] = [l2, h2]
            c["ratio_ci975"] = [math.exp(l2), math.exp(h2)]
    elif step1_ci:
        c["Delta_ci975"] = c["ratio_ci975"] = [math.nan, math.nan]
    if abf is not None and fr is not None:
        c["decomposition"] = decomposition(c["seeds"], abf, fr, n_boot)
    return c


def _logs(vals, seeds):
    out = {}
    for s in seeds:
        v = C.fnum(vals.get(s))
        if math.isfinite(v) and v > 0:
            out[s] = math.log(v)
    return out


def decomposition(seeds, abf, fr, n_boot=C.N_BOOT):
    """Median over ``seeds`` of log(I^ABF_a / I^ABF_b) and of log(I^FR_a / I^FR_b) with CIs (cross_cell.contrast on
    the log values; fresh generator per quantity)."""
    out = {}
    for tag, (va, vb) in (("abf", abf), ("fr", fr)):
        c = X.contrast(_logs(va, seeds), _logs(vb, seeds), n_boot, C.BOOT_SEED)
        out[tag] = dict(n=c["n"], median_log_ratio=c["Delta"], median_log_ratio_ci95=c["Delta_ci95"],
                        ratio=c["ratio"], ratio_ci95=c["ratio_ci95"], p=c["p"], wins_a=c["wins_a"],
                        losses_a=c["losses_a"])
    out["definition"] = ("median over the contrast's seeds of log(Ibar_a / Ibar_b) per arm; ratio = exp; ratio < 1: "
                         "the arm's error is lower at lambda_a")
    return out


def unadjusted_verdict(c, kind):
    if c["n"] == 0:
        return "NOT AVAILABLE"
    if c["n"] < C.SMALL_N:
        return f"{kind}: n = {c['n']} < {C.SMALL_N}: no claim (sign-test p {c['sign_test_p']:.3g})"
    lo, hi = c["Delta_ci95"]
    side = ("CI excludes 1, FR relatively better at lambda_a" if hi < 0 else
            "CI excludes 1, FR relatively better at lambda_b" if lo > 0 else "CI includes 1")
    return f"{kind} (unadjusted): {'equivalent; ' if c['equivalent'] else ''}{side}; p = {c['p']:.3g}"


def family_verdict(c, p_holm):
    if c["n"] == 0:
        return "NOT AVAILABLE"
    if c["n"] < C.SMALL_N:
        return f"n = {c['n']} < {C.SMALL_N}: no claim (sign-test p {c['sign_test_p']:.3g})"
    sig = p_holm is not None and math.isfinite(p_holm) and p_holm < C.ALPHA
    txt = ("Holm p < 0.05: " + ("R < 1 (FR relatively better at the higher lambda)" if c["Delta"] < 0 else
                                 "R > 1 (FR relatively worse at the higher lambda)" if c["Delta"] > 0 else "R = 1")
           if sig else "Holm p >= 0.05")
    if c["equivalent"]:
        txt += "; 95 % CI of R inside [0.90, 1.11]"
    return txt


# ============================================================================================ inputs
def check_inputs(cells, verify_runs=True):
    """cross_cell.check_summary / check_freshness on every cell with a summary; the primary reference digest must be
    identical.  Returns {lam: info}."""
    info, digests = {}, {}
    for lam, rec in sorted(cells.items()):
        d = dict(config=C.rel(rec["config"]), config_sha256=C.sha256_file(rec["config"]), cell=rec["name"],
                 summary=C.rel(rec["summary_path"]), available=rec["S"] is not None, why=rec["why"],
                 fixture=C.is_fixture(rec["P"]), h=float(rec["P"]["engine_cfg"]["h"]), B=int(rec["P"]["B"]))
        if rec["S"] is not None:
            C.require_current_summaries({lam: rec}, verify_runs)
            d.update(summary_sha256=C.sha256_file(rec["summary_path"]), generated_utc=rec["S"].get("generated_utc"),
                     n_boot=rec["S"].get("n_boot"),
                     code_digest=((rec["S"].get("provenance") or {}).get("code") or {}).get("digest"),
                     complete={str(N): {m: len(C.complete_seeds(rec["S"], N, m)) for m in ("abf", "fr")}
                               for N in rec["P"]["N_ladder"]},
                     planned_seeds=len(rec["P"]["seeds"]))
            digests[lam] = ((rec["S"].get("reference") or {}).get("primary_reference") or {}).get("digest")
        info[C.lam_key(lam)] = d
    if len(set(digests.values())) > 1 or None in digests.values():
        raise C.ExpAError(f"the primary reference is not identical in every cell: {digests}")
    return info, (next(iter(digests.values())) if digests else None)


# ============================================================================================ main computation
def per_cell_block(S, N, metric, n_boot):
    A, F = C.per_seed(S, N, "abf", metric), C.per_seed(S, N, "fr", metric)
    lr, excl = X.log_ratios(A, F)
    v = np.array([lr[s] for s in sorted(lr)], float)
    med, lo, hi = M.boot_median_ci(v, n_boot, C.BOOT_SEED) if v.size else (math.nan, math.nan, math.nan)
    g = C.summary_contrast(S, N, metric) or {}
    return dict(n=int(v.size), median_L=med, median_L_ci95=[lo, hi], mean=C.mean_companions(A, F, n_boot),
                ratio=math.exp(med) if v.size else math.nan,
                ratio_ci95=[math.exp(lo), math.exp(hi)] if v.size else [math.nan, math.nan],
                excluded={int(k): w for k, w in excl.items()},
                G=dict(median=C.fnum(g.get("G_median")), ci95=[C.fnum(x) for x in (g.get("G_ci95") or [None, None])],
                       wins=g.get("wins"), losses=g.get("losses"), ties=g.get("ties"), n=g.get("n")),
                abf=C.boot_median_inf(A, n_boot), fr=C.boot_median_inf(F, n_boot),
                claim_allowed=bool(v.size >= C.SMALL_N)), lr, A, F


def analyse(cells, n_boot=C.N_BOOT, verify_runs=True):
    """The cross_cell.json document (without provenance / signature fields)."""
    production, fixture, design_probs = C.design_status(cells)
    Ns = C.common_N(cells)
    if production and Ns != list(C.N_PROD):
        raise C.ExpAError(f"production cells with N ladder {Ns}")
    info, ref_digest = check_inputs(cells, verify_runs)
    doc = dict(schema=SCHEMA, n_boot=int(n_boot), boot_seed=C.BOOT_SEED, alpha=C.ALPHA,
               equivalence_band=list(C.EQUIV), small_n=C.SMALL_N, N=Ns, lams=[C.lam_key(l) for l in C.LAMS],
               cells=info, cells_missing=[C.lam_key(l) for l in C.LAMS if l not in cells or cells[l]["S"] is None],
               primary_reference_digest=ref_digest, production_design=production, fixture=fixture,
               design_problems=design_probs, freshness_verified=bool(verify_runs),
               metrics={m: g for m, g in METRICS}, per_cell={}, G_hi={}, contrasts={}, warnings=[],
               definitions=dict(
                   L="log(Ibar^FR / Ibar^ABF) per seed (cross_cell.log_ratios: seeds with both arms finite and > 0)",
                   R="exp(median over common seeds of [L_a - L_b]); R < 1: FR relatively better at lambda_a",
                   ci95="percentile 95 % CI of the seed-bootstrap median (default_rng(20261010), cross_cell.contrast)",
                   ci975="Holm-consistent CI of the first Holm step (m = 2), percentiles 1.25 / 98.75 of the same draws",
                   p="two-sided percentile-bootstrap p with the +1 correction (cross_cell.bootstrap_p)",
                   holm="E1 and E2 are separate families of size 2 (N 512, N 128), primary metric only",
                   secondary="final_e_F, Ibar_Fp_stat, Ibar_TV_half: same construction, every row unadjusted",
                   G_hi="upper end of the 95 % CI of the summary's paired G(Ibar_F) at (lambda, N)",
                   decomposition="median log(Ibar_a / Ibar_b) per arm over the contrast's seeds"))
    if not verify_runs:
        doc["warnings"].append("run-tree freshness NOT verified (--no-verify-runs)")
    if fixture:
        doc["warnings"].append("FIXTURE cells: synthetic tiny-budget data, never production evidence")
    for p in design_probs:
        doc["warnings"].append(f"design: {p}")
    L, A, F = {}, {}, {}
    for lam, rec in sorted(cells.items()):
        if rec["S"] is None:
            continue
        for N in Ns:
            for metric, _ in METRICS:
                blk, lr, a, f = per_cell_block(rec["S"], N, metric, n_boot)
                doc["per_cell"].setdefault(C.lam_key(lam), {}).setdefault(str(N), {})[metric] = blk
                L[(metric, lam, N)], A[(metric, lam, N)], F[(metric, lam, N)] = lr, a, f
            g = doc["per_cell"][C.lam_key(lam)][str(N)][PRIMARY_METRIC]["G"]
            doc["G_hi"].setdefault(C.lam_key(lam), {})[str(N)] = g["ci95"][1]
    for metric, grp in METRICS:
        rows = []
        for kind, specs in (("family", [(f, a, b) for f, a, b, _ in FAMILIES]), ("trend", list(TRENDS))):
            for cid, a, b in specs:
                for N in Ns:
                    ka, kb = (metric, a, N), (metric, b, N)
                    step1 = kind == "family" and metric == PRIMARY_METRIC
                    if ka in L and kb in L:
                        c = contrast_row(L[ka], L[kb], n_boot, step1, abf=(A[ka], A[kb]), fr=(F[ka], F[kb]))
                    else:
                        c = contrast_row({}, {}, n_boot, step1)
                        c["missing"] = [C.lam_key(x) for x, k in ((a, ka), (b, kb)) if k not in L]
                    c.update(id=cid, a=C.lam_key(a), b=C.lam_key(b), N=int(N), kind=kind,
                             label=f"{cid} lambda {a:g} vs {b:g} at N {N}")
                    rows.append(c)
        fams = {}
        if metric == PRIMARY_METRIC:
            for fid, a, b, flab in FAMILIES:
                fr_rows = [r for r in rows if r["kind"] == "family" and r["id"] == fid]
                adj = C.holm([r["p"] for r in fr_rows], m=FAMILY_SIZE)
                for r, ph in zip(fr_rows, adj):
                    r["p_holm"] = ph if r["n"] > 0 else math.nan
                    r["verdict"] = family_verdict(r, r["p_holm"])
                fams[fid] = dict(label=flab, a=C.lam_key(a), b=C.lam_key(b), size=FAMILY_SIZE,
                                 complete=all(r["n"] > 0 for r in fr_rows),
                                 claim_allowed=all(r["claim_allowed"] for r in fr_rows))
            for r in rows:
                if r["kind"] == "trend":
                    r["p_holm"] = None
                    r["verdict"] = unadjusted_verdict(r, "trend")
        else:
            for r in rows:
                r["p_holm"] = None
                r["verdict"] = unadjusted_verdict(r, "secondary")
        doc["contrasts"][metric] = dict(group=grp, families=fams, rows=rows)
    return doc


def input_signature(cells, n_boot, verify_runs, audits=None, provisional=False):
    ins = {}
    for lam, rec in sorted(cells.items()):
        ins[C.lam_key(lam)] = dict(config=C.sha256_file(rec["config"]),
                                   summary=(C.sha256_file(rec["summary_path"]) if rec["S"] is not None else None))
    return C.digest(dict(schema=SCHEMA, inputs=ins, code=C.code_digests(CODE_FILES), n_boot=int(n_boot),
                         verify_runs=bool(verify_runs), audits=C.audit_signature(audits or {}),
                         provisional=bool(provisional)))


# ============================================================================================ markdown
def _r(c, key="ratio_ci95"):
    return f"{C.fmt(c['ratio'], 4)} {C.fmt_ci(c.get(key), 4)}"


def md(doc):
    f = C.fmt
    Lines = ["# Experiment A: cross-cell contrasts (PREREGISTRATION.md 1.2, 1.6)", "",
             f"Generated {doc.get('provenance', {}).get('generated_utc')} by scripts/final_campaign/expA_cross_cell.py "
             f"({SCHEMA}); {doc['n_boot']} paired seed-bootstrap resamples (default_rng({doc['boot_seed']})).", "",
             "R = exp(median over coupled seeds of [L_a - L_b]), L = log(Ibar^FR / Ibar^ABF): **R < 1 means FR's "
             "performance relative to ABF is better at lambda_a**.  E1 (16 vs 1) and E2 (128 vs 1) are separate Holm "
             "families over the two N; the 97.5 % CI is the Holm-consistent CI of the first step.  Equivalent iff the "
             f"95 % CI of R lies in [{doc['equivalence_band'][0]}, {doc['equivalence_band'][1]}].  Contrasts with "
             f"fewer than {doc['small_n']} seed pairs make no claim.", ""]
    if doc.get("fixture"):
        Lines += ["**FIXTURE INPUT: synthetic tiny-budget data, never production evidence.**", ""]
    Lines += [f"Complete: {doc.get('complete')}; production design: {doc['production_design']}; freshness verified: "
              f"{doc['freshness_verified']}; input signature `{str(doc.get('input_signature'))[:16]}`.", ""]
    comp = doc.get("completeness") or {}
    if comp.get("reasons"):
        Lines += ["Not complete because:", ""] + [f"* {r}" for r in comp["reasons"][:20]] + [""]
    Lines += ["Completeness audits: " + "; ".join(f"lambda {k}: {a.get('verdict')} (current {a.get('current')}, "
                                                     f"passes {a.get('passes')})"
                                                     for k, a in (comp.get("audits") or {}).items()), "",
              "## Cells", "", "| lambda | cell | summary | complete ABF / FR per N | h | B |", "|---|---|---|---|---|---|"]
    for k, c in doc["cells"].items():
        comp = "; ".join(f"N {N}: {v['abf']}/{v['fr']}" for N, v in (c.get("complete") or {}).items())
        Lines.append(f"| {k} | {c['cell']} | {'yes' if c['available'] else 'NO: ' + str(c['why'])} | "
                     f"{comp or '--'} (of {c.get('planned_seeds', '--')}) | {c['h']:g} | {c['B']:,} |")
    Lines.append("")
    prim = doc["contrasts"].get(PRIMARY_METRIC, {})
    Lines += ["## Primary: Ibar_F, families E1 and E2", "",
              "| contrast | N | n | R [95 % CI] | R [97.5 % CI] | p | p Holm | wins a / n | equivalent | ABF ratio "
              "a/b [95 % CI] | FR ratio a/b [95 % CI] | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in prim.get("rows", []):
        if r["kind"] != "family":
            continue
        dc = r.get("decomposition") or {}
        ab, fr = dc.get("abf") or {}, dc.get("fr") or {}
        Lines.append(f"| {r['id']} {r['a']} vs {r['b']} | {r['N']} | {r['n']} | {_r(r)} | {_r(r, 'ratio_ci975')} | "
                     f"{f(r['p'])} | {f(r['p_holm'])} | {r['wins_a']}/{r['n']} | {'yes' if r['equivalent'] else 'no'} | "
                     f"{f(ab.get('ratio'), 4)} {C.fmt_ci(ab.get('ratio_ci95'), 4)} | {f(fr.get('ratio'), 4)} "
                     f"{C.fmt_ci(fr.get('ratio_ci95'), 4)} | {r['verdict']} |")
    for fid, fm in prim.get("families", {}).items():
        Lines.append(f"\n{fid} ({fm['label']}): complete {fm['complete']}, claim allowed {fm['claim_allowed']}.")
    Lines += ["", "## Trend contrasts (Ibar_F, unadjusted)", "",
              "| contrast | N | n | R [95 % CI] | p | wins a / n | ABF ratio a/b | FR ratio a/b | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in prim.get("rows", []):
        if r["kind"] != "trend":
            continue
        dc = r.get("decomposition") or {}
        Lines.append(f"| {r['id']} {r['a']} vs {r['b']} | {r['N']} | {r['n']} | {_r(r)} | {f(r['p'])} | "
                     f"{r['wins_a']}/{r['n']} | {f((dc.get('abf') or {}).get('ratio'), 4)} | "
                     f"{f((dc.get('fr') or {}).get('ratio'), 4)} | {r['verdict']} |")
    for metric, grp in METRICS:
        if metric == PRIMARY_METRIC:
            continue
        Lines += ["", f"## Secondary: {metric} (every row unadjusted)", "",
                  "| contrast | N | n | R [95 % CI] | p | wins a / n | verdict |", "|---|---|---|---|---|---|---|"]
        for r in doc["contrasts"][metric]["rows"]:
            Lines.append(f"| {r['id']} {r['a']} vs {r['b']} | {r['N']} | {r['n']} | {_r(r)} | {f(r['p'])} | "
                         f"{r['wins_a']}/{r['n']} | {r['verdict']} |")
    Lines += ["", "## Per cell: FR/ABF ratio exp(median L) and paired G", "",
              "| lambda | N | metric | n | exp(median L) [95 % CI] | G median [95 % CI] (wins/n) | mean G [95 % CI] | "
              "mean L [95 % CI] | ABF median [CI] | FR median [CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lam, byN in doc["per_cell"].items():
        for N, blk in byN.items():
            for metric, b in blk.items():
                g = b["G"]
                mg, ml = (b.get("mean") or {}).get("G") or {}, (b.get("mean") or {}).get("L") or {}
                mgs = (f"{100 * C.fnum(mg.get('mean')):+.1f} % [{100 * C.fnum((mg.get('ci95') or [None])[0]):+.1f}, "
                       f"{100 * C.fnum((mg.get('ci95') or [None, None])[1]):+.1f}]" if mg else "--")
                mls = f"{f(ml.get('mean'), 3)} {C.fmt_ci(ml.get('ci95'), 3)}" if ml else "--"
                Lines.append(f"| {lam} | {N} | {metric} | {b['n']} | {f(b['ratio'], 4)} {C.fmt_ci(b['ratio_ci95'], 4)} | "
                             f"{100 * C.fnum(g['median']):+.1f} % [{100 * C.fnum(g['ci95'][0]):+.1f}, "
                             f"{100 * C.fnum(g['ci95'][1]):+.1f}] ({g['wins']}/{g['n']}) | {mgs} | {mls} | "
                             f"{f(b['abf']['median'], 4)} "
                             f"{C.fmt_ci(b['abf']['ci95'], 4)} | {f(b['fr']['median'], 4)} {C.fmt_ci(b['fr']['ci95'], 4)} |")
    if doc["warnings"]:
        Lines += ["", "## Warnings", ""] + [f"* {w}" for w in doc["warnings"]]
    return "\n".join(Lines) + "\n"


# ============================================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", default=C.CELLS_DIR)
    ap.add_argument("--out-dir", default=C.SYN_DIR)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--no-verify-runs", action="store_true",
                    help="skip the run-tree freshness check (synthetic summaries without run files only)")
    ap.add_argument("--provisional", action="store_true",
                    help="force complete = false (the synthesis runs over a failed completeness audit)")
    ap.add_argument("--force", action="store_true", help="overwrite a COMPLETE output of different inputs")
    ap.add_argument("--recompute", action="store_true", help="rewrite an up-to-date output (same inputs)")
    ap.add_argument("--dry-run", action="store_true", help="report inputs and the write decision; write nothing")
    a = ap.parse_args(argv)
    cells = C.load_cells(a.cells_dir)
    if not cells:
        raise C.ExpAError(f"no cell config under {a.cells_dir}")
    out = os.path.abspath(a.out_dir)
    jp = os.path.join(out, "cross_cell.json")
    verify = not a.no_verify_runs
    comp = C.completeness(cells, a.n_boot, verify, a.provisional)
    sig = input_signature(cells, a.n_boot, verify, comp["audits"], a.provisional)
    if a.dry_run:
        for lam, rec in sorted(cells.items()):
            print(f"lambda {lam:g}: {C.rel(rec['config'])} -> summary {'present' if rec['S'] else rec['why']}; audit "
                  f"{comp['audits'][C.lam_key(lam)]['verdict']} (passes {comp['audits'][C.lam_key(lam)]['passes']})")
        sp = C.summary_problems(cells, verify)
        for lam, (kind, p) in sorted(sp.items()):
            print(f"lambda {lam:g}: summary {kind.upper()} ({len(p)}): {p[:2]}")
        try:
            decision = C.decide_write(jp, sig, a.force, a.recompute)
        except C.ExpAError as e:
            decision = f"REFUSE ({e})"
        print(f"complete-eligible {comp['complete']} ({len(comp['reasons'])} reason(s)); would {decision} {jp} "
              f"(signature {sig[:16]})" + ("; summaries not current: would stop" if sp else ""))
        return 0
    C.require_current_summaries(cells, verify)          # also before skipping: a skipped output must still be current
    decision = C.decide_write(jp, sig, a.force, a.recompute)
    if decision == "skip":
        print(f"{jp} is up to date (signature {sig[:16]}; summaries verified current); nothing to do (--recompute "
              "rewrites it)")
        return 0
    prev = C.previous_signature(jp)
    doc = analyse(cells, a.n_boot, verify_runs=verify)
    doc.update(input_signature=sig, provenance=C.provenance(CODE_FILES), complete=comp["complete"],
               completeness=comp, provisional=bool(a.provisional), cells_dir=C.rel(a.cells_dir))
    if prev and prev != sig:
        doc["replaced_signature"] = prev
    C.begin_output(jp, SCHEMA, prev)
    C.write_text(os.path.join(out, "cross_cell.md"), md(C.reload_json(M.json_safe(doc))))
    C.write_json(jp, doc)
    for r in doc["contrasts"][PRIMARY_METRIC]["rows"]:
        if r["kind"] == "family":
            print(f"{r['label']}: R {C.fmt(r['ratio'], 4)} {C.fmt_ci(r['ratio_ci95'], 4)} p {C.fmt(r['p'])} "
                  f"Holm {C.fmt(r['p_holm'])} n {r['n']} -> {r['verdict']}", flush=True)
    print(f"wrote {jp} and cross_cell.md (complete {doc['complete']})", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except C.ExpAError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
