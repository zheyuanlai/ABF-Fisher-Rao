#!/usr/bin/env python
"""Check 8 of Experiment A (docs/final_campaign/PREREGISTRATION.md section 1.3): old configurations unchanged.

    python scripts/final_campaign/expA_check_old_bitwise.py --dry-run           # reconstruct + verify signatures only
    taskset -c <4 cores> python scripts/final_campaign/expA_check_old_bitwise.py --workers 4    # ~5 min per job

(a) engine identity: sha256 of src/gateway_family_numba.py == the frozen 69087f2c... == the imported module's ==
    meta.engine_sha256 of every stored file re-run below;
(b) re-runs, at HEAD, the four stored mechanism production jobs
        results/mechanism/matched_free_energy/shift/N128/s8100_{abf,fr}.npz     (Exp I shifted fibre, N 128)
        results/mechanism/matched_free_energy/alpha1/N512/s8100_{abf,fr}.npz    (Exp I alpha = 1, N 512)
    with the EXACT stored signature: cfg = the stored cfg_json, method / N / seed / n_steps / diagnostics from the
    stored meta, the call run_job made (save grid prereg_save_grid(n_steps, h), default trace / snapshot grids, default
    noise seed and initial state; chunk_steps = the stored meta.chunk_steps, which is not part of the signature).  The
    reconstructed run signature must equal the stored meta.run_signature field for field (engine sha256 included)
    BEFORE anything runs -- if the default call does not reproduce it, the stored grids / noise seed are passed
    explicitly and that is recorded; if neither does, the job FAILS without running.  The re-run is in memory (nothing
    is written next to the stored files) and every array of the stored file except the session bookkeeping (cfg_json,
    meta_json, wall_s, peak_rss_mb, n_checkpoints_written, resumed_from) must be bitwise equal (dtype, shape, values,
    NaN = NaN); the re-run's cfg_json and run signature are compared too (recorded).
(c) the harness-inertness part of check 8 (prereg 1.3 G-A1, "plus the harness inertness test above") is decided by
    its own test record results/final_campaign/expA/checks/check8c_harness_inertness.json, written by
    scripts/final_campaign/expA_record_tests.py --suite check8c; this tool READS it (verdict, and that its input
    signature is current: expA_record_tests.record_is_current) and never runs it.

Writes results/final_campaign/expA/checks/check8_old_bitwise.json with part_8ab_verdict (PASS iff (a) holds and every
job is EQUAL), part_8c (the inertness record's verdict and currency) and the check-8 verdict:
    PASS     8a/8b PASS and 8c PASS (record present and current)
    PARTIAL  8a/8b PASS but the 8c record is missing or stale (re-run this tool once it exists: job records are reused)
    FAIL     otherwise (8a/8b not PASS, or 8c FAIL).
Exit 0 only for PASS (PARTIAL exits 3).  One record per job under results/final_campaign/expA/checks/check8_jobs/
(resumable: a job record whose input signature -- stored-file sha256, engine sha256, this tool's sha256 -- is unchanged
is not re-run unless --recompute).  An existing final record with a different input signature is not overwritten
without --force (a record without a signature counts as different), EXCEPT one whose 8a/8b part has the same signature
(only the 8c record changed, e.g. a PARTIAL record completed once the 8c record exists, or an 8c record re-run after a
test edit; the superseded verdict is kept in the new record).  The final input signature covers the 8a/8b job signatures and the sha256 of the 8c
record.  Each re-run appends one row to results/final_campaign/compute_ledger.csv (stage expA-check8, threads 1).
--files replaces the four default jobs (tests); --inertness-record replaces the 8c record path (tests).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import expA_run_common as C  # noqa: E402

ROOT = C.ROOT
DEFAULT_FILES = [os.path.join(ROOT, "results", "mechanism", "matched_free_energy", f"{c}", f"N{N}", f"s8100_{m}.npz")
                 for c, N in (("shift", 128), ("alpha1", 512)) for m in ("abf", "fr")]
OUT = os.path.join(C.PROD_ROOT, "checks", "check8_old_bitwise.json")
INERTNESS_RECORD = os.path.join(C.PROD_ROOT, "checks", "check8c_harness_inertness.json")
EXCLUDED = ("cfg_json", "meta_json", "wall_s", "peak_rss_mb", "n_checkpoints_written", "resumed_from")
TOOL_FILES = [os.path.abspath(__file__), os.path.join(HERE, "expA_run_common.py")]


def tool_sha():
    h = hashlib.sha256()
    for p in TOOL_FILES:
        h.update(C.sha256_file(p).encode())
    return h.hexdigest()


def label(path):
    rel = os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path
    return rel.replace(os.sep, "__").replace(".npz", "")


def load_stored(path):
    with np.load(path, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    return arr, json.loads(str(arr["meta_json"])), json.loads(str(arr["cfg_json"]))


def reconstruct(path):
    """(call kwargs, record) reproducing the stored run signature exactly, or (None, record) if impossible."""
    import gateway_family_numba as E
    arr, meta, cfg = load_stored(path)
    stored_sig = meta.get("run_signature")
    method, N, seed, n_steps = meta["method"], int(meta["N"]), int(meta["seed"]), int(meta["n_steps"])
    diag = bool(meta.get("diagnostics", True))
    h = float(E.full_cfg(cfg)["h"])
    base = dict(cfg=cfg, method=method, N=N, seed=seed, n_steps=n_steps, diagnostics=diag,
                chunk_steps=int(meta.get("chunk_steps") or max(1, (1 << 23) // N)))
    rec = dict(file=os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path,
               file_sha256=C.sha256_file(path), method=method, N=N, seed=seed, n_steps=n_steps, h=h,
               diagnostics=diag, chunk_steps=base["chunk_steps"], stored_engine_sha256=meta.get("engine_sha256"),
               stored_git_commit=meta.get("git_commit"), stored_engine=meta.get("engine"))
    attempts = []
    default_save = E.prereg_save_grid(n_steps, h)
    cands = [("run_job default call (prereg_save_grid, default trace / snapshot grids, default noise seed)",
              dict(save_steps=default_save))]
    cands.append(("explicit stored grids and noise seed",
                  dict(save_steps=arr["save_step"], trace_steps=arr["traces_step"], snapshot_steps=arr["snap_step"],
                       noise_seed=int(meta["noise_seed"]))))
    for name, kw in cands:
        sig = E.run_signature(cfg, method, N, seed, n_steps, kw["save_steps"], diagnostics=diag,
                              **{k: v for k, v in kw.items() if k != "save_steps"})
        mm = E.signature_mismatch(stored_sig, sig)
        attempts.append(dict(call=name, signature_mismatch=mm))
        if not mm:
            rec.update(call=name, signature_attempts=attempts, signature_reproduced=True)
            return dict(base, **kw), rec
    rec.update(call=None, signature_attempts=attempts, signature_reproduced=False)
    return None, rec


def compare_arrays(stored, res):
    """Bitwise comparison of every stored array (bookkeeping excluded) with the re-run result."""
    new = {k: np.asarray(v) for k, v in res.items() if k not in ("cfg", "meta")}
    mism, missing, compared = [], [], []
    for k in sorted(stored):
        if k in EXCLUDED:
            continue
        if k not in new:
            missing.append(k)
            continue
        a, b = stored[k], new[k]
        compared.append(k)
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"):
            mism.append(k)
    extra = sorted(k for k in new if k not in stored and k not in EXCLUDED)
    return dict(n_compared=len(compared), compared=compared, mismatched=mism, missing_in_rerun=missing,
                extra_in_rerun=extra)


def run_job(path):
    """Re-run one stored job at HEAD and compare (pool worker; never raises)."""
    import gateway_family_numba as E
    t0 = time.time()
    rec = dict(started_utc=C.utc(t0))
    try:
        C.check_engine()
        kw, r = reconstruct(path)
        rec.update(r)
        if kw is None:
            rec.update(status="FAIL", reason="stored run signature not reproducible at HEAD; not run")
        else:
            stored, meta, cfg = load_stored(path)
            res = E.run_arm(kw["cfg"], kw["method"], kw["N"], kw["seed"], kw["n_steps"], kw["save_steps"],
                            trace_steps=kw.get("trace_steps"), snapshot_steps=kw.get("snapshot_steps"),
                            noise_seed=kw.get("noise_seed"), diagnostics=kw["diagnostics"],
                            chunk_steps=kw["chunk_steps"])
            cmp_ = compare_arrays(stored, res)
            rec.update(cmp_)
            rec["cfg_json_equal"] = json.dumps(res["cfg"], sort_keys=True) == str(stored["cfg_json"])
            rec["run_signature_equal"] = res["meta"]["run_signature"] == meta.get("run_signature")
            rec["rerun_engine_sha256"] = res["meta"]["engine_sha256"]
            rec["rerun_git_commit"] = res["meta"]["git_commit"]
            ok = (not cmp_["mismatched"] and not cmp_["missing_in_rerun"] and rec["cfg_json_equal"]
                  and rec["run_signature_equal"])
            rec["status"] = "EQUAL" if ok else "MISMATCH"
    except Exception as e:  # noqa: BLE001
        rec.update(status="FAIL", reason=f"{type(e).__name__}: {e}")
    rec["wall_s"] = time.time() - t0
    rec["start_epoch"] = t0
    return rec


def job_input_signature(path):
    return hashlib.sha256(f"{C.sha256_file(path)}|{C.EXPECTED_ENGINE_SHA256}|{tool_sha()}".encode()).hexdigest()


def part_8c(path=INERTNESS_RECORD):
    """The harness-inertness record (check 8c): dict(record, present, verdict, current, reason, sha256)."""
    import expA_record_tests as RT
    out = dict(record=os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path, present=False,
               verdict=None, current=False, reason="record missing", sha256=None)
    if not os.path.exists(path):
        return out
    try:
        rec = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        out.update(present=True, reason=f"unreadable: {type(e).__name__}")
        return out
    cur, why = RT.record_is_current(rec, RT.SUITES["check8c"])
    if rec.get("check") != "check8c":
        cur, why = False, f"not a check8c record (check {rec.get('check')!r})"
    out.update(present=True, verdict=rec.get("verdict"), current=bool(cur), reason=why, sha256=C.sha256_file(path),
               generated_utc=rec.get("generated_utc"), counts=rec.get("counts"))
    return out


def check8_verdict(part_ab, p8c):
    if part_ab != "PASS" or (p8c["present"] and p8c["current"] and p8c["verdict"] not in (None, "PASS")):
        return "FAIL"
    if p8c["present"] and p8c["current"] and p8c["verdict"] == "PASS":
        return "PASS"
    return "PARTIAL"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="*", default=None, help="stored result files (default: the four prereg jobs)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="engine check + signature reconstruction only")
    ap.add_argument("--recompute", action="store_true", help="re-run jobs whose record is up to date")
    ap.add_argument("--force", action="store_true", help="overwrite a final record with a different input signature")
    ap.add_argument("--compute-ledger", default=C.COMPUTE_LEDGER, help=argparse.SUPPRESS)   # tests only
    ap.add_argument("--inertness-record", default=INERTNESS_RECORD, help=argparse.SUPPRESS)  # tests only
    a = ap.parse_args(argv)
    files = [os.path.abspath(f) for f in (a.files or DEFAULT_FILES)]
    eng = dict(expected=C.EXPECTED_ENGINE_SHA256)
    try:
        eng["file"] = C.check_engine()
        eng["file_and_import_equal_expected"] = True
    except C.EngineMismatch as e:
        eng.update(file_and_import_equal_expected=False, error=str(e))
    missing = [f for f in files if not os.path.exists(f)]
    stored_shas = {}
    for f in files:
        if os.path.exists(f):
            stored_shas[os.path.relpath(f, ROOT) if f.startswith(ROOT + os.sep) else f] = load_stored(f)[1].get(
                "engine_sha256")
    eng["stored_meta_engine_sha256"] = stored_shas
    eng["stored_equal_expected"] = all(v == C.EXPECTED_ENGINE_SHA256 for v in stored_shas.values()) and bool(stored_shas)
    print(f"engine: file/import {'==' if eng['file_and_import_equal_expected'] else '!='} frozen "
          f"{C.EXPECTED_ENGINE_SHA256[:16]}...; stored metas {'all equal' if eng['stored_equal_expected'] else 'DIFFER'}",
          flush=True)
    if missing:
        print(f"MISSING stored files: {missing}", flush=True)
    if not eng["file_and_import_equal_expected"]:
        print(f"REFUSED: {eng.get('error')}", flush=True)
        return 2
    ab_sig = hashlib.sha256("|".join(job_input_signature(f) for f in files if os.path.exists(f)).encode()).hexdigest()
    p8c = part_8c(os.path.abspath(a.inertness_record))
    in_sig = hashlib.sha256(f"{ab_sig}|8c:{p8c['sha256'] or 'absent'}".encode()).hexdigest()
    print(f"part 8c (harness inertness record {p8c['record']}): "
          + (f"verdict {p8c['verdict']}, {p8c['reason']}" if p8c["present"] else "MISSING (write it with "
             "scripts/final_campaign/expA_record_tests.py --suite check8c)"), flush=True)
    supersedes = None
    if os.path.exists(a.out) and not a.dry_run:
        try:
            old = json.load(open(a.out))
        except Exception:  # noqa: BLE001
            old = {}
        same_ab = old.get("part_8ab_input_signature") == ab_sig     # only the 8c record changed (e.g. PARTIAL -> PASS)
        if old.get("input_signature") != in_sig and not a.force and not same_ab:
            print(f"REFUSED to overwrite {a.out} (a record with a different input signature exists; --force)",
                  flush=True)
            return 2
        if old.get("input_signature") != in_sig:
            supersedes = dict(generated_utc=old.get("generated_utc"), verdict=old.get("verdict"),
                              input_signature=old.get("input_signature"))
    jobs_dir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "check8_jobs")
    todo, recs = [], {}
    for f in files:
        if f in missing:
            recs[f] = dict(file=f, status="FAIL", reason="stored file missing")
            continue
        jp = os.path.join(jobs_dir, label(f) + ".json")
        sig = job_input_signature(f)
        if os.path.exists(jp) and not a.recompute:
            old = json.load(open(jp))
            if old.get("input_signature") == sig and old.get("status") in ("EQUAL", "MISMATCH"):
                recs[f] = old
                print(f"{label(f)}: up to date ({old['status']}), not re-run", flush=True)
                continue
        todo.append(f)
    if a.dry_run:
        for f in todo:
            kw, r = reconstruct(f)
            print(f"{label(f)}: signature {'REPRODUCED' if kw else 'NOT reproduced'} by '{r['call']}'; "
                  f"{r['method']} N {r['N']} seed {r['seed']} n_steps {r['n_steps']} h {r['h']} chunk "
                  f"{r['chunk_steps']} -> {r['N'] * r['n_steps']:,} walker-steps"
                  + ("" if kw else f"  attempts {r['signature_attempts']}"), flush=True)
        print(f"dry run: {len(todo)} job(s) to run, records -> {jobs_dir}, final -> {a.out}", flush=True)
        return 0
    cpu_probs = C.cpu_allocation_problems(a.workers) if todo else []
    if cpu_probs:
        print("REFUSED (CPU allocation, prereg section 3): " + "; ".join(cpu_probs), flush=True)
        return 2

    def finish(f, r):
        r["input_signature"] = job_input_signature(f) if os.path.exists(f) else None
        os.makedirs(jobs_dir, exist_ok=True)
        jp = os.path.join(jobs_dir, label(f) + ".json")
        with open(jp + ".tmp", "w") as fh:
            json.dump(r, fh, indent=1)
        os.replace(jp + ".tmp", jp)
        C.append_csv(a.compute_ledger, C.COMPUTE_FIELDS,
                     [C.compute_row("expA-check8", f"re-run {r.get('file', f)}", r["wall_s"], r["start_epoch"],
                                    note=r["status"])])
        recs[f] = r
        print(f"{label(f)}: {r['status']} ({r.get('n_compared', 0)} arrays compared"
              + (f", mismatched {r['mismatched']}" if r.get("mismatched") else "")
              + (f", {r['reason']}" if r.get("reason") else "") + f") {r['wall_s'] / 60:.1f} min", flush=True)

    if todo:
        if a.workers == 1 or len(todo) == 1:
            for f in todo:
                finish(f, run_job(f))
        else:
            import expA_check_old_bitwise as me   # importable worker (pickling under fork and spawn)
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=min(a.workers, len(todo))) as ex:
                futs = {ex.submit(me.run_job, f): f for f in todo}
                for fu in as_completed(futs):
                    finish(futs[fu], fu.result())
    jobs = [recs[f] for f in files]
    part_ab = ("PASS" if eng["file_and_import_equal_expected"] and eng["stored_equal_expected"] and not missing
               and all(j.get("status") == "EQUAL" for j in jobs) else "FAIL")
    verdict = check8_verdict(part_ab, p8c)
    out = dict(check="8", rule=("PREREGISTRATION.md 1.3 check 8: (8a) engine sha256 = 69087f2c...; (8b) the four "
                                "stored production jobs re-run at HEAD bitwise equal on all shared arrays (session "
                                "bookkeeping excluded); (8c) the harness inertness test (record check8c, "
                                "expA_record_tests.py)"),
               generated_utc=C.utc(), tool=os.path.relpath(__file__, ROOT), input_signature=in_sig,
               part_8ab_input_signature=ab_sig, provenance=C.provenance(),
               engine=eng, excluded_keys=list(EXCLUDED), n_jobs=len(jobs),
               n_equal=sum(j.get("status") == "EQUAL" for j in jobs), part_8ab_verdict=part_ab, part_8c=p8c,
               verdict=verdict,
               verdict_note={"PASS": "8a, 8b and 8c pass",
                             "PARTIAL": "8a/8b PASS; the 8c harness-inertness record is missing or stale -- NOT a "
                                        "check-8 pass",
                             "FAIL": "a part of check 8 fails"}[verdict],
               jobs=[{k: v for k, v in j.items() if k != "compared"} for j in jobs])
    if supersedes:
        out["supersedes"] = supersedes
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(f"check 8: {verdict} (8a/8b {part_ab}: {out['n_equal']}/{out['n_jobs']} jobs bitwise equal; 8c "
          f"{p8c['verdict'] if p8c['present'] else 'missing'}{'' if p8c['current'] or not p8c['present'] else ' (stale)'})"
          f" -> {a.out}", flush=True)
    return 0 if verdict == "PASS" else (3 if verdict == "PARTIAL" else 1)


if __name__ == "__main__":
    sys.exit(main())
