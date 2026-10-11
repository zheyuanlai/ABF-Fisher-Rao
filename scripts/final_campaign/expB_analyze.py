#!/usr/bin/env python
"""Experiment B analysis: endpoints of prereg section 2.6 and the decision of section 2.7, per system.

    python scripts/final_campaign/expB_analyze.py --system gateway
    python scripts/final_campaign/expB_analyze.py --system lta150

Inputs: results/final_campaign/expB/bench/<system>/{decision.json, summary.json} (arms, t_arm, implementations,
serial throughput) and the production records results/final_campaign/expB/prod/<system>/<arm>/s<seed>[.rep<k>].*.

Replicate selection (prereg 2.5).  For every WALL-CLOCK endpoint the PRIMARY value of (arm, seed) comes from the first
replicate that is uncontended for that endpoint's wall window (W at a threshold: [0, W], whole run when censored;
Ibar_F^wall: the whole run); seeds with no such replicate are flagged, excluded from the primary analysis and included
(first replicate) in the SENSITIVITY analysis.  Science-only endpoints (force evaluations to accuracy, fraction
reaching each threshold, Ibar_F, final e_F) do not depend on timing (counter mode and the serial engine are
deterministic; repeats are sha256-checked against the first replicate): they use the first replicate of EVERY seed
with a complete record, contended or not.

Completeness.  Every planned (arm, seed) -- all decision.json arms, including an arm without an implementation, x the
configured seeds -- without a complete record is listed under ``missing_runs``, and every (arm, seed) with no
uncontended replicate whose repeats are not exhausted (expB_run.needs_repeat) under ``pending_repeats``; with either,
the stage status is INCOMPLETE and ``decision_primary`` reads "INCOMPLETE" (the decision computed on the available seeds is kept beside it as
``decision_on_available_seeds``).  An arm without any usable run also makes the IUT 'unresolved' (it must cover
every arm).

Endpoints.  Primary: W_mid (decision: expB_stats.compare_arms -> IUT, rho_best, Gehan-Wilcoxon / paired sign test,
budget-artefact labels).  Sensitivities: W_mid interpolated (same replicates), W_mid including the flagged seeds, and
W_mid with the CCD-wide contention verdict (Amendment 2, A2.4).  The A3.3 (a) stage label (prod/<system>/STAGE_LABEL.json)
is carried into the summary (``exploratory``, ``decision_role``).
Secondary: (1) W_strict, W_loose (same machinery, budget-artefact labels); (2) Ibar_F^wall(W*) with W* = the smallest
of the arms' median completion wall times over runs uncontended for the whole run, fixed before the bootstrap;
(3) force evaluations to accuracy (N x completed steps at the threshold save); (4) used (t_arm x W) and allocated
(16 x W) core-hours to accuracy; (5) effective parallelism (serial-equivalent speed-up = bench serial wall per
walker-step x B / wall, efficiency = speed-up / t_arm); (6) per-step ABF, synchronisation and FR overheads from the
in-engine timers, and FR's paired added wall time at N = 512 (own thread counts; at FR 512's t_arm -- measured when
ABF 512 runs at the same t_arm, otherwise a labelled benchmark projection); (7) fraction of runs reaching each
threshold; (8) seed-bootstrap 95 % CIs (10 000, default_rng(20261010)) for every quantity.

Outputs: results/final_campaign/expB/analysis/<system>/{summary.json, curves.npz, boot.npz} and seed-level tables
results/final_campaign/expB/tables/<system>_seeds.{csv,md}.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import expB_common as C  # noqa: E402
import expB_run as R  # noqa: E402
import expB_stats as S  # noqa: E402

ANALYZE_VERSION = "expB_analyze/1"
W_ENDPOINTS = ("W_strict", "W_mid", "W_loose")
N_CURVE = 240


def arm_sets(decision):
    """(arms with an implementation, FR arm names, ABF arm names) -- F and A over ALL decision arms, so an arm
    without an implementation (hence without runs) still enters the IUT, which then cannot declare an advantage."""
    arms = R.canonical_arms(decision)
    F = [a["arm"] for a in decision["arms"] if a["method"] == "fr"]
    A_ = [a["arm"] for a in decision["arms"] if a["method"] == "abf"]
    return arms, F, A_


def select(recs, window):
    """(record, primary: bool) -- the first replicate uncontended for ``window``, else the first replicate."""
    for r in recs:
        if r.get("contention_status", {}).get(window) == "uncontended":
            return r, True
    return (recs[0], False) if recs else (None, False)


def select_ccd(recs, window):
    """Amendment 2 (A2.4) sensitivity: the first replicate whose CCD-WIDE verdict for ``window`` is uncontended
    (else the first replicate, flagged)."""
    for r in recs:
        if r.get("contention", {}).get(window, {}).get("ccd_scope", {}).get("status") == "uncontended":
            return r, True
    return (recs[0], False) if recs else (None, False)


def bench_serial_c(bench_summary, method, N):
    s = (bench_summary or {}).get(f"{method}{N}_serial")
    return float(s["wall_per_walker_step"]) if s and s.get("n") else math.nan


def bench_thr(bench_summary, method, N, t):
    s = (bench_summary or {}).get(f"{method}{N}_parallel_t{t}")
    return float(s["throughput"]) if s and s.get("n") else math.nan


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "_boot"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def analyze(system, out_root=C.OUT_ROOT, spec=None, decision=None, bench_summary=None, write=True, n_boot=S.N_BOOT,
            gehan_perm=S.N_PERM, tables_dir=None):
    spec = spec or C.system_spec(system)
    if decision is None:
        decision = R.load_decision(system, out_root)
    if bench_summary is None:
        p = os.path.join(out_root, "bench", system, "summary.json")
        bench_summary = C.read_json(p)["summary"] if os.path.exists(p) else {}
    arms, F, A_ = arm_sets(decision)
    by_arm = {a["arm"]: a for a in arms}
    scorer = C.Scorer(spec)
    pairs = [("abf512", "fr512")] if ("abf512" in by_arm and "fr512" in by_arm) else []

    # ---- gather records
    recs = {}
    for a in arms:
        recs[a["arm"]] = {}
        for s in spec["seeds"]:
            j0 = R.prod_job(spec, a, s, 0)
            rr = R.replicate_records(j0, out_root)
            if rr:
                recs[a["arm"]][int(s)] = rr
    n_runs = {a: len(v) for a, v in recs.items()}
    missing_runs = [dict(arm=a["arm"], seed=int(s), reason=("arm has no implementation" if not a.get("impl") else
                                                            "no complete record"))
                    for a in decision["arms"] for s in spec["seeds"]
                    if int(s) not in recs.get(a["arm"], {})]
    # a seed with no uncontended replicate whose repeats are not exhausted is not final (prereg 2.5: excluded only
    # after up to 3 repeats); the stage is then INCOMPLETE like one with missing runs
    pending_repeats = [dict(arm=a, seed=int(s), n_complete_records=len(rr))
                       for a, per in recs.items() for s, rr in sorted(per.items()) if R.needs_repeat(rr)]
    stage_status = "INCOMPLETE" if (missing_runs or pending_repeats) else "COMPLETE"

    # ---- W endpoints (primary / sensitivity), completion
    vals = {"primary": {}, "sensitivity": {}}
    interp, completion, flags, chosen = {}, {}, {}, {}
    for a, per in recs.items():
        for key in ("primary", "sensitivity"):
            vals[key][a] = {ep: {} for ep in W_ENDPOINTS}
        interp[a], completion[a], flags[a], chosen[a] = {}, {ep: {} for ep in W_ENDPOINTS}, {}, {}
        for s, rr in per.items():
            for ep in W_ENDPOINTS:
                r, prim = select(rr, ep)
                if r is None:
                    continue
                W = C.num(r["endpoints"][ep])
                vals["sensitivity"][a][ep][s] = W
                completion[a][ep][s] = float(r["wall_total_s"])
                if prim:
                    vals["primary"][a][ep][s] = W
                    if ep == "W_mid":
                        interp[a][s] = C.num(r["endpoints"]["W_mid_interp"])
                else:
                    flags[a].setdefault(ep, []).append(int(s))
                if ep == "W_mid":
                    chosen[a][s] = dict(replicate=int(r["job"]["replicate"]), primary=bool(prim))

    def comp(key, ep, values=None):
        v = values if values is not None else {a: vals[key][a][ep] for a in vals[key]}
        v = {a: x for a, x in v.items() if x}
        c = {a: completion[a][ep] for a in v} if ep is not None else None
        if not any(a in v for a in F) or not any(a in v for a in A_):
            return dict(status="no data")
        return S.compare_arms(v, F, A_, pairs=pairs, completion=c, n_boot=n_boot, gehan_perm=gehan_perm)

    results = {}
    results["W_mid"] = comp("primary", "W_mid")
    results["W_mid_sensitivity_incl_flagged"] = comp("sensitivity", "W_mid")
    results["W_mid_interp"] = comp("primary", "W_mid", values={a: interp[a] for a in interp})
    results["W_strict"] = comp("primary", "W_strict")
    results["W_loose"] = comp("primary", "W_loose")

    # ---- Amendment 2 (A2.4): the W_mid decision repeated with the CCD-wide contention verdict
    ccd_vals, ccd_flags, ccd_comp = {}, {}, {}
    for a, per in recs.items():
        ccd_vals[a], ccd_flags[a], ccd_comp[a] = {}, [], {}
        for s, rr in per.items():
            r, prim = select_ccd(rr, "W_mid")
            if r is None:
                continue
            if prim:
                ccd_vals[a][s] = C.num(r["endpoints"]["W_mid"])
                ccd_comp[a][s] = float(r["wall_total_s"])     # completion of the SAME (CCD-selected) replicate
            else:
                ccd_flags[a].append(int(s))
    v_ccd = {a: x for a, x in ccd_vals.items() if x}
    if any(a in v_ccd for a in F) and any(a in v_ccd for a in A_):
        ccd_res = S.compare_arms(v_ccd, F, A_, pairs=pairs, completion={a: ccd_comp[a] for a in v_ccd},
                                 n_boot=n_boot, gehan_perm=gehan_perm)
    else:
        ccd_res = dict(status="no data")
    results["W_mid_sensitivity_ccd_scope"] = dict(ccd_res, flagged_seeds=ccd_flags,
                                                  n_seeds={a: len(x) for a, x in ccd_vals.items()})

    # ---- Ibar_F^wall(W*) on whole-run uncontended replicates; e_F cached per (arm, seed)
    whole, ecache, wcomp = {}, {}, {}
    for a, per in recs.items():
        whole[a] = {}
        wcomp[a] = []
        for s, rr in per.items():
            r, prim = select(rr, "whole_run")
            whole[a][s] = (r, prim)
            if prim:
                wcomp[a].append(float(r["wall_total_s"]))
    ibw, Wstar = {}, None
    if any(wcomp.values()):
        Wstar, Wstar_arm, med_comp = S.w_star({a: v for a, v in wcomp.items() if v})
        for a, per in whole.items():
            ibw[a] = {}
            for s, (r, prim) in per.items():
                if not prim:
                    continue
                job = r["job"]
                res = R.load_science(job, out_root)
                key = (a, s)
                if key not in ecache:
                    ecache[key] = scorer.e_F(res)
                ibw[a][s] = C.ibar_wall(ecache[key], res["wall_s_at_save"], Wstar, scorer.e_flat)
        results["Ibar_F_wall"] = dict(W_star=Wstar, W_star_arm=Wstar_arm, median_completion_uncontended=med_comp,
                                      compare=comp(None, None, values=ibw) if ibw else dict(status="no data"))
    else:
        results["Ibar_F_wall"] = dict(status="no run uncontended for the whole run")

    # ---- costs, parallelism, overheads, fractions (primary replicate of W_mid; whole-run replicate for walls)
    per_arm = {}
    for a, per in recs.items():
        m, N = by_arm[a]["method"], int(by_arm[a]["N"])
        t = int(by_arm[a]["threads"])
        cs = bench_serial_c(bench_summary, m, N)
        rec_whole = {s: select(rr, "whole_run") for s, rr in per.items()}
        d = dict(method=m, N=N, threads=t, impl=by_arm[a]["impl"], n_seeds=len(per), n_records=sum(map(len, per.values())),
                 flagged_seeds=flags[a])
        sci = {s: rr[0] for s, rr in per.items()}                       # science: first replicate, every seed
        d["n_seeds_science"] = len(sci)
        d["science_nonidentical_repeats"] = sorted(int(s) for s, rr in per.items()
                                                   if any(r.get("science_identical_to_first") is False for r in rr[1:]))
        for ep in W_ENDPOINTS:
            thr = ep[2:]
            rec_ep = {s: select(rr, ep) for s, rr in per.items()}      # the endpoint's own replicate selection
            fe = {s: C.num(r["endpoints"][f"force_evals_{thr}"]) for s, r in sci.items()}
            uc = {s: C.num(r["endpoints"][f"used_core_h_{thr}"]) for s, (r, p) in rec_ep.items() if p}
            ac = {s: C.num(r["endpoints"][f"alloc_core_h_{thr}"]) for s, (r, p) in rec_ep.items() if p}
            d[f"force_evals_{thr}"] = S.boot_ci_median(fe, n_boot) if fe else None
            d[f"used_core_h_{thr}"] = S.boot_ci_median(uc, n_boot) if uc else None
            d[f"alloc_core_h_{thr}"] = S.boot_ci_median(ac, n_boot) if ac else None
            d[f"frac_reached_{thr}"] = (S.boot_ci_fraction([float(not r["endpoints"][f"censored_{thr}"])
                                                            for r in sci.values()], n_boot) if sci else None)
        d["Ibar_F"] = S.boot_ci_median({s: C.num(r["endpoints"]["Ibar_F"]) for s, r in sci.items()}, n_boot) \
            if sci else None
        d["final_e_F"] = S.boot_ci_median({s: C.num(r["endpoints"]["final_e_F"]) for s, r in sci.items()}, n_boot) \
            if sci else None
        walls = {s: float(r["wall_total_s"]) for s, (r, p) in rec_whole.items() if p}
        d["completion_wall_s"] = S.boot_ci_median(walls, n_boot) if walls else None
        sp = {s: S.effective_parallelism(cs, spec["B"], w, t) for s, w in walls.items()}
        d["speedup"] = S.boot_ci_median({s: v["speedup"] for s, v in sp.items()}, n_boot) if sp else None
        d["efficiency"] = S.boot_ci_median({s: v["efficiency"] for s, v in sp.items()}, n_boot) if sp else None
        d["serial_wall_per_walker_step"] = cs
        ov = {}
        for k in C.TIMER_KEYS:
            fr_ = {s: float(r["timers"][k]) / float(r["wall_total_s"]) for s, (r, p) in rec_whole.items()
                   if p and r.get("timers", {}).get(k) is not None and math.isfinite(float(r["timers"][k]))}
            us = {s: float(r["timers"][k]) * 1e6 / int(r["job"]["n_steps"]) for s, (r, p) in rec_whole.items()
                  if p and r.get("timers", {}).get(k) is not None and math.isfinite(float(r["timers"][k]))}
            ov[k] = dict(fraction=(S.boot_ci_median(fr_, n_boot) if fr_ else None),
                         us_per_step=(S.boot_ci_median(us, n_boot) if us else None))
        d["overheads"] = ov
        per_arm[a] = d
    results["per_arm"] = per_arm

    # ---- FR's paired added wall at N 512
    if pairs:
        ca = {s: float(r["wall_total_s"]) for s, (r, p) in {s: select(rr, "whole_run")
                                                             for s, rr in recs["abf512"].items()}.items() if p}
        cf = {s: float(r["wall_total_s"]) for s, (r, p) in {s: select(rr, "whole_run")
                                                             for s, rr in recs["fr512"].items()}.items() if p}
        own = S.paired_diff_ci(ca, cf, n_boot)
        ta, tf = int(by_arm["abf512"]["threads"]), int(by_arm["fr512"]["threads"])
        if ta == tf:
            common = dict(own, source="measured (ABF 512 and FR 512 share t_arm)")
        else:
            thr_a, thr_f = bench_thr(bench_summary, "abf", 512, tf), bench_thr(bench_summary, "fr", 512, tf)
            common = dict(median=(spec["B"] / thr_f - spec["B"] / thr_a) if (thr_a and thr_f) else math.nan,
                          source=f"benchmark projection at t = {tf} (ABF 512 runs at t_arm = {ta}): B (1/thr_FR - "
                                 f"1/thr_ABF)")
        results["fr_added_wall_N512"] = dict(own_thread_counts=own, common_thread_count=common,
                                             t_arm=dict(abf512=ta, fr512=tf),
                                             paired_noise=by_arm["abf512"]["impl"] == by_arm["fr512"]["impl"])

    # ---- drift reference
    ddir = os.path.join(out_root, "drift")
    drift = []
    if os.path.isdir(ddir):
        for f in sorted(os.listdir(ddir)):
            if f.endswith(".json") and not f.startswith("."):
                try:
                    r = C.read_json(os.path.join(ddir, f))
                except (OSError, ValueError):
                    continue
                if r.get("status") == "complete":
                    drift.append(dict(file=f, finished_utc=r.get("finished_utc"), wall_total_s=r.get("wall_total_s"),
                                      throughput=r.get("throughput"),
                                      contended=r.get("contended_any")))
    results["drift_reference"] = drift

    # ---- curves (median e_F vs wall) and tables
    curves = {}
    allw = [float(r["wall_total_s"]) for per in whole.values() for (r, p) in per.values() if r is not None and p]
    if allw:
        lo = 1e-3 * min(allw)
        grid = np.logspace(math.log10(max(lo, 1e-6)), math.log10(max(allw)), N_CURVE)
        curves["w_grid"] = grid
        for a, per in whole.items():
            rows = []
            for s, (r, p) in per.items():
                if r is None or not p:                 # curves use the Ibar_F^wall set: whole-run uncontended only
                    continue
                res = R.load_science(r["job"], out_root)
                key = (a, s)
                if key not in ecache:
                    ecache[key] = scorer.e_F(res)
                rows.append(C.e_at_wall(ecache[key], res["wall_s_at_save"], grid, scorer.e_flat))
            if rows:
                Y = np.array(rows)
                curves[f"{a}_median"] = np.median(Y, axis=0)
                curves[f"{a}_q25"] = np.percentile(Y, 25, axis=0)
                curves[f"{a}_q75"] = np.percentile(Y, 75, axis=0)
                curves[f"{a}_completion_median"] = np.array([np.median([float(r["wall_total_s"])
                                                                        for (r, p) in per.values() if r and p])])
    table = []
    for a, per in recs.items():
        for s, rr in sorted(per.items()):
            rm, pm = select(rr, "W_mid")
            rw, pw = select(rr, "whole_run")
            ep = rm["endpoints"]
            table.append(dict(system=system, arm=a, seed=s, n_replicates=len(rr), replicate_W_mid=rm["job"]["replicate"],
                              W_mid_primary=pm, whole_run_uncontended=pw,
                              W_strict=C.num(ep["W_strict"]), W_mid=C.num(ep["W_mid"]),
                              W_mid_interp=C.num(ep["W_mid_interp"]), W_loose=C.num(ep["W_loose"]),
                              completion_wall_s=float(rm["wall_total_s"]), Ibar_F=C.num(ep["Ibar_F"]),
                              final_e_F=C.num(ep["final_e_F"]),
                              Ibar_F_wall=(ibw.get(a, {}).get(s, math.nan) if ibw else math.nan),
                              force_evals_mid=C.num(ep["force_evals_mid"]), used_core_h_mid=C.num(ep["used_core_h_mid"]),
                              threads=int(rm["job"]["threads"]), impl=rm["job"]["impl"],
                              contention_W_mid=rm.get("contention_status", {}).get("W_mid")))
    dec_avail = results["W_mid"].get("decision", {}).get("decision") if isinstance(results["W_mid"], dict) else None
    lp = os.path.join(out_root, "prod", system, "STAGE_LABEL.json")
    stage = C.read_json(lp) if os.path.exists(lp) else {}
    summary = dict(version=ANALYZE_VERSION, system=system, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                         time.gmtime()),
                   arms=[a["arm"] for a in arms], planned_arms=[a["arm"] for a in decision["arms"]], fr_arms=F,
                   abf_arms=A_, paired=pairs, n_runs=n_runs, planned_seeds=[int(s) for s in spec["seeds"]],
                   stage_status=stage_status, missing_runs=missing_runs, n_missing_runs=len(missing_runs),
                   thresholds=spec["thresholds"], B=spec["B"], e_flat=scorer.e_flat, results=_strip(results),
                   pending_repeats=pending_repeats, n_pending_repeats=len(pending_repeats),
                   decision_primary=("INCOMPLETE" if stage_status == "INCOMPLETE" else dec_avail),
                   decision_on_available_seeds=dec_avail,
                   stage_label=stage.get("label"), exploratory=bool(stage.get("exploratory")),
                   decision_role=("exploratory decision (after a frozen V-B4 FAIL; Amendment 3)"
                                  if stage.get("exploratory") else "confirmatory decision (section 2.7)"),
                   notes=["W in seconds from the engine's timed start; inf = censored",
                          "rho_best is descriptive; frac_below_1 is not a probability",
                          "a contrast labelled 'not comparable (budget artefact)' is reported but not interpreted"])
    if write:
        adir = os.path.join(out_root, "analysis", system)
        C.write_json(os.path.join(adir, "summary.json"), summary)
        if curves:
            np.savez_compressed(os.path.join(adir, "curves.npz"), **curves)
        boots = {}
        for name in ("W_mid", "W_strict", "W_loose"):
            r = results.get(name, {})
            for a, b in (r.get("_boot") or {}).items():
                boots[f"{name}__{a}"] = b
        if boots:
            np.savez_compressed(os.path.join(adir, "boot.npz"), **boots)
        tdir = tables_dir or os.path.join(out_root, "tables")
        write_tables(table, tdir, system)
    summary["_table"] = table
    return summary


def write_tables(rows, tdir, system):
    os.makedirs(tdir, exist_ok=True)
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(os.path.join(tdir, f"{system}_seeds.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("inf" if isinstance(v, float) and math.isinf(v) else v) for k, v in r.items()})

    def f(v):
        if isinstance(v, float):
            if math.isinf(v):
                return "inf"
            if math.isnan(v):
                return "-"
            return f"{v:.4g}"
        return str(v)
    with open(os.path.join(tdir, f"{system}_seeds.md"), "w") as fh:
        fh.write(f"# Experiment B seed-level table: {system}\n\n")
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(f(r[k]) for k in cols) + " |\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Experiment B analysis")
    ap.add_argument("--system", required=True, choices=sorted(C.SYSTEMS))
    ap.add_argument("--out-root", default=C.OUT_ROOT)
    a = ap.parse_args(argv)
    C.limit_driver_threads()
    s = analyze(a.system, a.out_root)
    print(json.dumps(dict(system=s["system"], decision_primary=s["decision_primary"],
                          decision_on_available_seeds=s["decision_on_available_seeds"], stage_status=s["stage_status"],
                          n_missing_runs=s["n_missing_runs"], n_runs=s["n_runs"]), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
