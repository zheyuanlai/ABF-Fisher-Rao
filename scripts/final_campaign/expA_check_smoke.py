#!/usr/bin/env python
"""Check 9 of Experiment A (docs/final_campaign/PREREGISTRATION.md section 1.3): validation of the cell smoke runs.

    python scripts/final_campaign/expA_check_smoke.py [--smoke-T 4] [--dry-run]     # results/final_campaign/expA/smoke/T4
    python scripts/final_campaign/expA_check_smoke.py --root DIR                     # any root, no plan / pipeline checks

The smoke runs are made by  scripts/final_campaign/run_cells.py --smoke  (seeds 8390, 8391; all 8 cells = lambda
{1, 4, 16, 128} x N {512, 128}; both arms; B = 2048 x smoke_T / h).  NEVER interpreted: no performance number is read,
printed or stored.  Per file:

  plan       (smoke mode) the file exists for every planned job and is a complete result of THAT run (run signature,
             incl. engine sha256 = the frozen 69087f2c...); eqb_family.load_run (the analysis contract keys) and
             eqb_family.check_plan against the cell config with B = the smoke budget;
  contract   every key of the engine's result contract with diagnostics on (RESULT_KEYS: the 53 keys of a stored
             mechanism production file) is present; meta.status complete; meta.engine_sha256 the frozen engine;
  finite     every float array finite, except the NaN written by design (ABF: gen_ess_* / gen_maxfam_*; first_arrival_t
             when there was no arrival); max_abs_bias, X_final, Y_final, M_all finite;
  accounting n_force_evals = N n_steps; C_all[s].sum() = N save_step[s] (pre-deposit convention);
  D2 / D3    sum over the clone-age classes of dep_age_C[s] == C_all at snap_step[s], exactly; the same for
             dep_cross_C; sum over classes of dep_age_M[s] == M_all at snap_step[s] to 1e-9 relative (float sums);
  D1         fr_death_pos_hist_cum[s].sum() == fr_birth_pos_hist_cum[s].sum() == fr_deaths_cum[s] at every save;
  FR / cap   the ABF arm's FR counters are all zero; fr_deaths_hist / fr_events_hist have cap + 1 bins, cap =
             max(cap_min, floor(max_event_fraction N)) (the engine raises if realised deaths exceed the cap);
  pipeline   (smoke mode, unless --no-pipeline) eqb_family.run_metrics with the cell's frozen reference scorer runs and
             yields no non-finite curve value (only the counts are recorded, never a metric value).
Then check 6 (scripts/final_campaign/expA_check_pairing.check_root) on the same root and plan; its record is also
written to results/final_campaign/expA/checks/check6_pairing_smoke.json (or --out6).  Verdict PASS iff every planned file passes
every clause and check 6 is PASS; AMENDMENT_REQUIRED (exit 4) iff every file passes and check 6 is AMENDMENT_REQUIRED
(pairs that pass only the non-preregistered event rule; --event-rule-amendment REF as in expA_check_pairing.py); FAIL
(exit 1) otherwise.  Output JSON default results/final_campaign/expA/checks/check9_smoke.json; idempotent (same input
signature: kept; different: refused without --force -- exit 2 if the verdict is PASS, else the verdict's exit code).  The input signature is expA_check_pairing.record_signature
(result files, the frozen config / engine / cell configs / references / analysis modules, the check tools, and the
options mode, smoke_T, replay, pipeline, event-rule amendment); the production gate of run_cells.py recomputes it.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import expA_run_common as C  # noqa: E402
import expA_check_pairing as CP  # noqa: E402

_EQB = os.path.join(C.ROOT, "scripts", "equal_budget")
if _EQB not in sys.path:
    sys.path.insert(0, _EQB)

ROOT = C.ROOT
CELLS_DIR = os.path.join(ROOT, "configs", "final_campaign", "cells", "expA")
SYSTEM = "gateway_family"
RESULT_KEYS = ("C_all", "M_all", "X_final", "Y_final", "cfg_json", "dep_age_C", "dep_age_M", "dep_age_M2",
               "dep_cross_C", "events_cum", "first_arrival_step", "first_arrival_t", "fr_birth_pos_hist_cum",
               "fr_death_pos_hist_cum", "fr_deaths_cum", "fr_deaths_hist", "fr_deaths_hist_cum", "fr_events_hist",
               "fr_events_hist_cum", "fr_kc_cum", "fr_kd_cum", "fr_opp_cum", "fr_opp_death_cum", "fr_opp_event_cum",
               "gen_ess_run", "gen_ess_win", "gen_maxfam_run", "gen_maxfam_win", "gen_nuniq_run", "gen_nuniq_win",
               "gen_win_age_steps", "hist_inst", "lineage_visited", "max_abs_bias", "meta_json",
               "n_checkpoints_written", "n_force_evals", "n_fr_steps", "peak_rss_mb", "region_frac", "resumed_from",
               "save_step", "save_t", "save_u", "snap_step", "snap_t", "snap_u", "traces", "traces_anc",
               "traces_rebirth", "traces_step", "traces_t", "traces_y", "wall_s")
NAN_BY_DESIGN_ABF = ("gen_ess_run", "gen_maxfam_run", "gen_ess_win", "gen_maxfam_win")
MUST_BE_FINITE = ("max_abs_bias", "X_final", "Y_final", "M_all", "C_all")


def check_file(path, job=None, cellP=None, scorer=None):
    """All per-file clauses; returns (record, list of problems).  No metric value is ever returned."""
    import gateway_family_numba as E
    probs = []
    d = CP.load(path)
    meta, cfg = d["meta"], d["cfg"]
    method, N, n_steps = meta.get("method"), int(meta.get("N", -1)), int(meta.get("n_steps", -1))
    rec = dict(path=os.path.relpath(path, ROOT) if path.startswith(ROOT) else path, method=method, N=N,
               n_steps=n_steps, sha256=C.sha256_file(path))
    miss = [k for k in RESULT_KEYS if k not in d]
    if miss:
        probs.append(f"result contract keys missing: {miss}")
    if meta.get("status") != "complete":
        probs.append(f"meta status {meta.get('status')!r}")
    if meta.get("engine_sha256") != C.EXPECTED_ENGINE_SHA256:
        probs.append(f"engine sha256 {meta.get('engine_sha256')} != frozen {C.EXPECTED_ENGINE_SHA256}")
    if job is not None:
        mm = E.signature_mismatch(meta.get("run_signature"), C.job_signature(job))
        if mm:
            probs.append(f"not the planned run (signature fields {mm})")
    if miss:
        return rec, probs
    # finite
    allow = set(NAN_BY_DESIGN_ABF) if method == "abf" else set()
    if int(np.asarray(d["first_arrival_step"]).ravel()[0]) < 0:
        allow.add("first_arrival_t")
    nonfinite = []
    for k, v in d.items():
        if k in ("meta", "cfg") or not isinstance(v, np.ndarray) or v.dtype.kind not in "fc" or k in allow:
            continue
        if not np.all(np.isfinite(v)):
            nonfinite.append(k)
    for k in MUST_BE_FINITE:
        if not np.all(np.isfinite(np.asarray(d[k]))):
            nonfinite.append(k)
    if nonfinite:
        probs.append(f"non-finite values in {sorted(set(nonfinite))}")
    if method == "fr":
        for k in NAN_BY_DESIGN_ABF:
            if not np.all(np.isfinite(d[k])):
                probs.append(f"FR arm: {k} not finite")
    # accounting
    step = np.asarray(d["save_step"], dtype=np.int64)
    Ca = np.asarray(d["C_all"])
    if int(d["n_force_evals"]) != N * n_steps:
        probs.append(f"n_force_evals {int(d['n_force_evals'])} != N n_steps {N * n_steps}")
    if not np.array_equal(Ca.sum(axis=1), (N * step).astype(np.float64)):
        probs.append("C_all[s].sum() != N save_step[s]")
    # D2 / D3
    snap = np.asarray(d["snap_step"], dtype=np.int64)
    idx = {int(s): i for i, s in enumerate(step)}
    miss_snap = [int(s) for s in snap if int(s) not in idx]
    if miss_snap:
        probs.append(f"snapshot steps {miss_snap[:4]} are not save steps")
    else:
        rows = [idx[int(s)] for s in snap]
        dC, dX, dM = np.asarray(d["dep_age_C"]), np.asarray(d["dep_cross_C"]), np.asarray(d["dep_age_M"])
        if dC.shape[1] != E.N_AGE_CLASSES or dX.shape[1] != E.N_AGE_CLASSES:
            probs.append(f"D2/D3 class axis {dC.shape[1]}/{dX.shape[1]} != {E.N_AGE_CLASSES}")
        if not np.array_equal(dC.sum(axis=1), Ca[rows]):
            probs.append("D2: sum over clone-age classes of dep_age_C != C_all at the snapshots")
        if not np.array_equal(dX.sum(axis=1), Ca[rows]):
            probs.append("D3: sum over crossing-age classes of dep_cross_C != C_all at the snapshots")
        Ma = np.asarray(d["M_all"])[rows]
        scale = max(1.0, float(np.max(np.abs(Ma))))
        if not np.allclose(dM.sum(axis=1), Ma, rtol=1e-9, atol=1e-9 * scale):
            probs.append("D2: sum over clone-age classes of dep_age_M != M_all at the snapshots (1e-9)")
        rec["n_snapshots"] = int(snap.size)
    # D1
    deaths = np.asarray(d["fr_deaths_cum"])
    dp, bp = np.asarray(d["fr_death_pos_hist_cum"]), np.asarray(d["fr_birth_pos_hist_cum"])
    if not np.array_equal(dp.sum(axis=1), deaths):
        probs.append("D1: death-position histogram totals != fr_deaths_cum")
    if not np.array_equal(bp.sum(axis=1), deaths):
        probs.append("D1: birth-position histogram totals != fr_deaths_cum")
    rec["fr_deaths_total"] = int(deaths[-1])
    # FR / cap
    c = E.full_cfg(cfg)
    cap = max(int(c["cap_min"]), int(math.floor(c["max_event_fraction"] * N)))
    for k in ("fr_deaths_hist", "fr_events_hist"):
        if np.asarray(d[k]).shape[0] != cap + 1:
            probs.append(f"{k} has {np.asarray(d[k]).shape[0]} bins != cap + 1 = {cap + 1}")
    if method == "abf":
        nz = [k for k in CP.ABF_ZERO_KEYS if np.any(np.asarray(d[k]) != 0)]
        if nz:
            probs.append(f"ABF arm with non-zero FR counters {nz}")
    elif method == "fr":
        if int(np.asarray(d["fr_opp_cum"])[-1]) <= 0:
            probs.append("FR arm with no FR opportunity")
    # analysis contract and pipeline (smoke mode)
    if cellP is not None:
        import eqb_family as MF
        try:
            res = MF.load_run(path, SYSTEM)
            MF.check_plan(res, cellP, path, dict(N=N, seed=int(meta["seed"]), method=method))
            rec["analysis_contract"] = "ok"
            if scorer is not None:
                _, sc = MF.run_metrics(res, SYSTEM, cellP, scorer, path)
                nn = {k: int(v) for k, v in sc.items() if k.startswith("n_nan_")}
                bad = {k: v for k, v in nn.items() if v}
                rec["pipeline"] = dict(ran=True, n_error_curves=len(nn), n_nan_keys=bad)
                if not nn or bad:
                    probs.append(f"pipeline: non-finite error-curve values {bad}" if bad else
                                 "pipeline: no error curves")
        except Exception as e:  # noqa: BLE001
            probs.append(f"analysis contract: {type(e).__name__}: {e}")
    return rec, probs


def check_smoke(root, expected=None, cells=None, pipeline=True, replay=True, verbose=True, event_rule_amendment=None):
    pairs = CP.discover(root, cells)
    files = []
    if expected is not None:
        for (cell, N, seed, m), job in sorted(expected.items()):
            files.append((cell, N, seed, m, job, os.path.join(root, cell, f"N{N}", f"s{seed}_{m}.npz")))
    else:
        for (cell, N, seed), have in sorted(pairs.items()):
            for m, p in sorted(have.items()):
                files.append((cell, N, seed, m, None, p))
    import eqb_family as MF
    cellPs, scorers = {}, {}
    recs, n_pass = [], 0
    for cell, N, seed, m, job, p in files:
        r = dict(cell=cell, N=N, seed=seed, method=m)
        if not os.path.exists(p):
            r.update(status="MISSING", problems=[f"{p} missing"])
            recs.append(r)
            continue
        cellP = scorer = None
        if job is not None:
            if cell not in cellPs:
                with open(os.path.join(CELLS_DIR, f"{cell}.json")) as fh:
                    cellPs[cell] = json.load(fh)
            cellP = copy.deepcopy(cellPs[cell])
            cellP["B"] = int(job["N"]) * int(job["n_steps"])
            if pipeline:
                if cell not in scorers:
                    scorers[cell] = MF.make_scorer(SYSTEM, cellP)
                scorer = scorers[cell]
        fr, probs = check_file(p, job, cellP, scorer)
        r.update(fr, problems=probs, status="PASS" if not probs else "FAIL")
        n_pass += not probs
        recs.append(r)
        if verbose:
            print(f"{cell} N{N} s{seed} {m}: {r['status']}" + ("" if not probs else f"  {probs}"), flush=True)
    c6 = CP.check_root(root, cells, expected, replay=replay, verbose=verbose,
                       event_rule_amendment=event_rule_amendment)
    files_ok = bool(files) and n_pass == len(files)
    if files_ok and c6["verdict"] in ("PASS", "AMENDMENT_REQUIRED"):
        verdict = c6["verdict"]
    else:
        verdict = "FAIL"
    return dict(check="9", root=root, n_files_expected=len(files), n_files_pass=n_pass,
                n_files_missing=sum(r["status"] == "MISSING" for r in recs), check6=c6, verdict=verdict,
                pipeline=bool(pipeline), replay=bool(replay), event_rule_amendment=event_rule_amendment,
                interpreted=False, files=recs)


def c9_options(mode, smoke_T, replay, pipeline, amendment):
    return dict(CP.c6_options(mode, smoke_T, replay, amendment), check="9", pipeline=bool(pipeline))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke-T", type=float, default=4.0)
    ap.add_argument("--root", default=None, help="generic root (no plan, no pipeline): <root>/<cell>/N<N>/*.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out6", default=None, help="check-6 record (smoke mode; default "
                                                 "results/final_campaign/expA/checks/check6_pairing_smoke.json)")
    ap.add_argument("--no-pipeline", action="store_true")
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--event-rule-amendment", default=None, metavar="REF",
                    help="license the event rule of check 6 (see expA_check_pairing.py); REF must occur in the "
                         "dated-amendment section of docs/final_campaign/PREREGISTRATION.md")
    ap.add_argument("--out-root", default=None, help=argparse.SUPPRESS)      # tests only (driver's --out-root)
    ap.add_argument("--prereg", default=None, help=argparse.SUPPRESS)        # tests only (amendment lookup)
    a = ap.parse_args(argv)
    C.check_engine()
    if a.event_rule_amendment is not None and not C.amendment_ref_found(a.event_rule_amendment, a.prereg):
        print(f"REFUSED: --event-rule-amendment {a.event_rule_amendment!r} does not occur in the dated-amendment section "
              f"of {a.prereg or C.PREREG}; write the amendment first", flush=True)
        return 2
    P = C.load_config()
    cells = [v["name"] for v in P["variants"]]
    if a.root:
        root, expected, cells = os.path.abspath(a.root), None, None
    else:
        root = C.smoke_root(a.out_root, a.smoke_T)
        expected = CP.expected_jobs("smoke", a.smoke_T)
    checks = os.path.join(C.prod_root(a.out_root), "checks")
    out = a.out or os.path.join(checks, "check9_smoke.json" if not a.root else "check9_custom.json")
    if a.dry_run:
        pairs = CP.discover(root, cells)
        n_f = sum(len(v) for v in pairs.values())
        print(f"check 9 dry run: root {root}; {n_f} files found" +
              (f", {len(expected)} planned" if expected else "") + f"; output {out}")
        return 0
    pipeline = expected is not None and not a.no_pipeline
    rec = check_smoke(root, expected, cells, pipeline=pipeline, replay=not a.no_replay,
                      event_rule_amendment=a.event_rule_amendment)
    inputs = [p for v in CP.discover(root, cells).values() for p in v.values()]
    mode = None if a.root else "smoke"
    sT = None if a.root else a.smoke_T
    st, _ = CP.write_record(rec, out, inputs, force=a.force, tool=os.path.relpath(os.path.abspath(__file__), ROOT),
                            options=c9_options(mode, sT, not a.no_replay, pipeline, a.event_rule_amendment))
    st6 = None
    if not a.root:
        out6 = a.out6 or os.path.join(checks, "check6_pairing_smoke.json")
        st6, _ = CP.write_record(rec["check6"], out6, inputs,
                                 force=a.force, tool=os.path.relpath(os.path.abspath(__file__), ROOT),
                                 options=CP.c6_options(mode, sT, not a.no_replay, a.event_rule_amendment))
        print(f"check 6 record {st6}: {out6}")
    c6 = rec["check6"]
    print(f"check 9: {rec['verdict']} ({rec['n_files_pass']}/{rec['n_files_expected']} files pass; check 6 "
          f"{c6['verdict']}: {c6['n_pass']}/{c6['n_expected_pairs']} pairs PASS the frozen rule, "
          f"{c6['n_pass_event_rule_only']} only the event rule); record {st}: {out}")
    print(f"  check 6: {c6['verdict_note']}")
    if st == "REFUSED" or st6 == "REFUSED":
        print("  an existing record with a DIFFERENT input signature is kept (use --force to replace it)")
    return CP.record_exit(rec["verdict"], st, st6)


if __name__ == "__main__":
    sys.exit(main())
