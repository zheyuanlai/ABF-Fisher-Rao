#!/usr/bin/env python
"""Production driver of Experiment A (docs/final_campaign/PREREGISTRATION.md sections 1.3 check 9 and 1.4).

    python scripts/final_campaign/run_cells.py --dry-run
    taskset -c <cores> python scripts/final_campaign/run_cells.py --smoke --workers 8       # check 9 smoke runs
    taskset -c <cores> python scripts/final_campaign/run_cells.py --workers 8                # production (512 runs)

Loads ONLY the frozen configs/final_campaign/expA.json (sha256 checked) and runs, through the mechanism driver's own
functions (scripts/mechanism/run_cells.py: job_list, job_state, run_one, out_path, warm_cache, _Ledgers; imported,
unmodified) and the unmodified engine src/gateway_family_numba.py (run_job), one single-threaded process per
(cell, N, seed, method):

  production  results/final_campaign/expA/<cell>/N<N>/s<seed>_<method>.npz, seeds 8300-8331, B = 2048 x 40 / h
              (n_steps = B / N; T_N 160 / 640 t.u. at N 512 / 128), ledger results/final_campaign/expA/ledger.csv
  --smoke     results/final_campaign/expA/smoke/T<smoke_T>/<cell>/N<N>/s<seed>_<method>.npz, the smoke seeds 8390,
              8391, B = 2048 x smoke_T / h -- the mechanism campaign's smoke convention: smoke_T (default 4 t.u.) is the
              horizon at the anchor N0 = 2048, so T_N = 16 / 64 t.u. at N 512 / 128 (FR's 4 t.u. ramp is completed);
              ledger <smoke root>/ledger.csv.  Never a production path.

Order: SEED-MAJOR (every cell x N x arm of a seed before the next seed; expA_run_common.seed_major), dispatched in that
order with at most --workers jobs in flight, so a truncated production leaves complete coupled seed blocks.

Before each launch: free space on / (shutil.disk_usage); below 2 GB no new job starts, the driver polls every 60 s
(logged to stdout and <root>/disk_pause.log) and resumes at >= 3 GB; a pause longer than 24 h stops the driver (exit 3,
prereg stopping rule (iv)).  Running jobs are not interrupted (their own final save is ~1 MB).

Refusals (exit 2, nothing run):
  * the engine file or the imported engine does not hash to EXPECTED_ENGINE_SHA256 (the sha256 recorded in the
    mechanism production files' meta); the config is not the frozen file;
  * a seed outside the plan: production seeds must be the config's seeds (8300-8331), --smoke seeds its smoke_seeds
    (8390, 8391);
  * a STALE output (a complete result of a DIFFERENT run at a job's path -- run signature, not status; never
    overwritten);
  * the CPU allocation breaks prereg section 3 (expA_run_common.cpu_allocation_problems): --workers > 64, the affinity
    set contains two SMT siblings of one physical core, or more workers than physical cores in the affinity set;
  * another driver holds <root>/.run_cells.lock;
  * the root's ledger records a NONFINITE run (a complete file with a non-finite state: prereg stopping rule (i)),
    unless --continue-on-failure;
  * production only, the PROJECTION GATE (prereg section 3): Experiment A compute already used + the projected cost of
    the jobs still to run > --stage-cap-core-h (default 110 = the prereg Experiment A ceiling, validation included).
    Used = the compute ledger's expA* rows + (unless the compute ledger already holds expA-validation rows) the
    validation tool's own ledger results/final_campaign/expA/validation/ledger.csv (G-A1 chains, lam = 1 replays, V5
    ABF smoke; one core per row) + (unless it holds expA-check4b rows) the check-4b record's wall time.  The projection
    uses the median measured us per walker-step of the expA smoke ledger (slowest cell) when it exists, else --est-us;
    jobs being resumed are projected at their full cost (their earlier sessions are not yet in the ledger);
  * production only, the PRODUCTION GATE (prereg 1.3 "Production starts only after the gate G-A1/G-A2 and all nine
    checks below pass"; scripts/final_campaign/expA_gate.py): the validation summary's selection is PASS at the config
    h, and the records of checks 1/7, 2, 4b, 8 (8a/8b/8c), 9 and 6 (smoke) exist, PASS and are current.  Every missing or
    failed item is printed.  --override-gate REF bypasses it only if REF occurs in the dated-amendment section of
    docs/final_campaign/PREREGISTRATION.md; the override and the failed items are written into every provenance row
    (column gate).  --dry-run prints the plan and the gate and exits 2 if production would be refused.

Failures (prereg stopping rule (i)): the worker checks every complete result for non-finite X_final, Y_final, M_all,
C_all, max_abs_bias (the engine does not raise on NaN) and labels such a run NONFINITE.  On the first FAILED or
NONFINITE run the driver starts no new job, lets the running ones finish (ledgered) and exits 1, unless
--continue-on-failure.

Idempotent / resumable: a complete result of THIS run is skipped (signature check); a checkpoint <out>.ckpt.npz is
resumed by run_job.  Every finished job appends one row to the ledger (mechanism format: experiment expA, variant =
cell, N, seed, method, n_steps, status, wall_s, peak_rss_mb, n_force_evals, finished_utc, note), one row to
results/final_campaign/compute_ledger.csv (stage expA-production / expA-smoke, threads 1, measured wall, used core-h;
a resumed run adds a second row for its earlier interrupted sessions: their checkpointed work plus <= one checkpoint
interval each, from the engine's session record) and one row to <root>/run_provenance.csv (start / end UTC, elapsed,
engine sha256, git commit, config sha256, h, lam, float64, cpu, threads 1, CPU set, pid, host, python / numpy / numba
versions, output sha256, prior_sessions_s, gate).  Each run file's own meta already records engine sha256, git commit,
seed, h, model and versions.

Pin the CPU set with taskset (the pool workers inherit it); each worker is single-threaded.  The worker function lives
in the importable module scripts/final_campaign/expA_run_common.py (run_timed).
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import expA_run_common as C  # noqa: E402

STAGE_CAP_DEFAULT = 110.0      # prereg section 3: Experiment A <= 110 core-h


def smoke_us_per_walker_step(smoke_ledger):
    """Slowest cell's median measured us per walker-step in a smoke ledger (None if none)."""
    if not os.path.exists(smoke_ledger):
        return None
    per = {}
    with open(smoke_ledger, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                if not r["status"].startswith("complete"):
                    continue
                us = float(r["wall_s"]) * 1e6 / (int(r["N"]) * int(r["n_steps"]))
            except (KeyError, ValueError, ZeroDivisionError):
                continue
            per.setdefault((r["variant"], r["N"]), []).append(us)
    if not per:
        return None
    return max(statistics.median(v) for v in per.values())


def used_core_h(ledger, prefix="expA"):
    if not os.path.exists(ledger):
        return 0.0
    tot = 0.0
    with open(ledger, newline="") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("stage", "")).startswith(prefix):
                try:
                    tot += float(r["used_core_h"])
                except (KeyError, ValueError):
                    pass
    return tot


def _stages(ledger):
    if not os.path.exists(ledger):
        return set()
    with open(ledger, newline="") as fh:
        return {str(r.get("stage", "")) for r in csv.DictReader(fh)}


def expA_used_core_h(compute_ledger, root):
    """(total, breakdown) of the Experiment A compute already used: the compute ledger's expA* rows, plus the validation
    tool's own ledger (<root>/validation/ledger.csv, one core per row) unless the compute ledger already holds
    expA-validation rows, plus the check-4b record wall time(s) unless it holds expA-check4b rows."""
    br = {"compute_ledger_expA_rows": used_core_h(compute_ledger)}
    stages = _stages(compute_ledger)
    if not any(st.startswith("expA-validation") for st in stages):
        v, path = 0.0, os.path.join(root, "validation", "ledger.csv")
        if os.path.exists(path):
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    try:
                        v += float(r["wall_s"]) / 3600.0
                    except (KeyError, ValueError, TypeError):
                        pass
        br["validation_ledger"] = v
    if not any(st.startswith("expA-check4b") for st in stages):
        v = 0.0
        cdir = os.path.join(root, "checks")
        if os.path.isdir(cdir):
            import json
            for fn in sorted(os.listdir(cdir)):
                if fn.startswith("check4b_frozen_x") and fn.endswith(".json"):
                    try:
                        v += float(json.load(open(os.path.join(cdir, fn)))["provenance"]["wall_s"]) / 3600.0
                    except Exception:  # noqa: BLE001
                        pass
        br["check4b_records"] = v
    return sum(br.values()), br


def nonfinite_ledger_rows(root):
    """Ledger rows of the root whose status is NONFINITE (complete file with a non-finite state)."""
    path = os.path.join(root, "ledger.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if str(r.get("status", "")).startswith("NONFINITE")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="smoke seeds 8390/8391, B = 2048 smoke_T / h, smoke root")
    ap.add_argument("--smoke-T", type=float, default=4.0, help="smoke horizon at the anchor N0 = 2048 (t.u.)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only-cell", nargs="*", default=None, help="cell names, e.g. shift_lam16")
    ap.add_argument("--only-N", type=int, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--est-us", type=float, default=0.09, help="us per walker-step if no smoke measurement exists")
    ap.add_argument("--stage-cap-core-h", type=float, default=STAGE_CAP_DEFAULT,
                    help="production projection gate: refuse if used(expA) + projected > this")
    ap.add_argument("--continue-on-failure", action="store_true",
                    help="keep dispatching after a FAILED / NONFINITE run (default: stop, prereg stopping rule (i))")
    ap.add_argument("--override-gate", default=None, metavar="REF",
                    help="production despite a failed production gate; REF must occur in the dated-amendment section "
                         "of docs/final_campaign/PREREGISTRATION.md; logged in every provenance row")
    ap.add_argument("--gate-smoke-T", type=float, default=4.0, help=argparse.SUPPRESS)  # tests only
    ap.add_argument("--prereg", default=None, help=argparse.SUPPRESS)                  # tests only
    ap.add_argument("--out-root", default=None, help=argparse.SUPPRESS)          # tests only
    ap.add_argument("--compute-ledger", default=C.COMPUTE_LEDGER, help=argparse.SUPPRESS)  # tests only
    ap.add_argument("--disk-path", default="/", help=argparse.SUPPRESS)          # tests only
    a = ap.parse_args(argv)

    try:
        P = C.load_config()
        C.check_engine()
    except (C.ConfigMismatch, C.EngineMismatch) as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    wit = C.witness_engine_sha()
    print(f"engine sha256 {C.EXPECTED_ENGINE_SHA256} (file and import); stored mechanism production meta: "
          f"{wit if wit is not None else 'witness file absent'}"
          + ("" if wit in (None, C.EXPECTED_ENGINE_SHA256) else "  WARNING: differs"), flush=True)
    cpu_probs = C.cpu_allocation_problems(a.workers)
    if cpu_probs:
        print("REFUSED (CPU allocation, prereg section 3; checked against this process's affinity set): "
              + "; ".join(cpu_probs), flush=True)
        return 2
    allowed = [int(x) for x in (P["smoke_seeds"] if a.smoke else P["seeds"])]
    if a.seeds:
        bad = [int(x) for x in a.seeds if int(x) not in allowed]
        if bad:
            print(f"REFUSED: seed(s) {bad} are not {'smoke' if a.smoke else 'production'} seeds of the frozen plan "
                  f"({allowed[0]}-{allowed[-1]}" + (f", {len(allowed)} labels" if len(allowed) > 2 else "") + ")",
                  flush=True)
            return 2
    if a.override_gate is not None and not C.amendment_ref_found(a.override_gate, a.prereg):
        print(f"REFUSED: --override-gate {a.override_gate!r} does not occur in the dated-amendment section of "
              f"{a.prereg or C.PREREG}; write the amendment first", flush=True)
        return 2

    R = C.rc()
    import gateway_family_numba as E
    root = C.smoke_root(a.out_root, a.smoke_T) if a.smoke else C.prod_root(a.out_root)
    stage = "expA-smoke" if a.smoke else "expA-production"
    try:
        jobs = C.expA_jobs(P, smoke=a.smoke, smoke_T=a.smoke_T, only_cell=a.only_cell, only_N=a.only_N,
                           seeds=a.seeds)
    except ValueError as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    states = [R.job_state(E, R.out_path(root, j), j) for j in jobs]
    n = {s: states.count(s) for s in ("complete", "stale", "resume", "new")}
    todo = [j for j, s in zip(jobs, states) if s in ("new", "resume")]
    walker_steps = sum(int(j["N"]) * int(j["n_steps"]) for j in todo)
    us_meas = smoke_us_per_walker_step(os.path.join(C.smoke_root(a.out_root, 4.0), "ledger.csv"))
    us = us_meas if us_meas is not None else a.est_us
    est = walker_steps * us * 1e-6 / 3600.0
    used, used_br = expA_used_core_h(a.compute_ledger, C.prod_root(a.out_root))
    print(f"{stage}: {len(jobs)} jobs ({n['complete']} complete, {n['stale']} STALE, {n['resume']} resume, "
          f"{n['new']} new) -> {root}", flush=True)
    print(f"  to run: {len(todo)} jobs, {walker_steps:,} walker-steps, projected {est:.2f} core-h at {us:.4f} us per "
          f"walker-step ({'measured, expA smoke ledger' if us_meas is not None else '--est-us'}); Experiment A used "
          f"so far {used:.2f} core-h (" + ", ".join(f"{k} {v:.2f}" for k, v in used_br.items()) + ")", flush=True)
    cells = {}
    for j in jobs:
        cells.setdefault((j["variant"], j["N"], j["n_steps"]), []).append(j)
    for (v, N, ns), js in cells.items():
        print(f"    {v:<13s} N {N:<4d} n_steps {ns:<10d} T_N {ns * js[0]['cfg']['h']:.6g}  lam {js[0]['cfg']['lam']:g}  "
              f"{len(js)} jobs", flush=True)
    print("  order (seed-major), first jobs: " + "; ".join(C.job_item(j) for j in jobs[:6])
          + (" ..." if len(jobs) > 6 else ""), flush=True)
    import shutil
    print(f"  disk: {shutil.disk_usage(a.disk_path).free / 1e9:.2f} GB free on {a.disk_path} (pause < "
          f"{C.DISK_LOW / 1e9:.0f} GB, resume >= {C.DISK_RESUME / 1e9:.0f} GB)", flush=True)
    if n["stale"]:
        for j, s in zip(jobs, states):
            if s == "stale":
                print(f"  STALE (complete result of a DIFFERENT run, never overwritten): {R.out_path(root, j)}",
                      flush=True)
        print("REFUSED: stale outputs present; move them away (they are never overwritten)", flush=True)
        return 2
    if not a.smoke and used + est > a.stage_cap_core_h:
        print(f"REFUSED (projection gate): used {used:.2f} + projected {est:.2f} core-h > stage cap "
              f"{a.stage_cap_core_h:.1f} core-h", flush=True)
        return 2
    bad_rows = nonfinite_ledger_rows(root)
    if bad_rows:
        for r in bad_rows[:8]:
            print(f"  NONFINITE run in the ledger: {r.get('variant')} N {r.get('N')} seed {r.get('seed')} "
                  f"{r.get('method')}: {r.get('status')}", flush=True)
        if not a.continue_on_failure:
            print(f"REFUSED: {len(bad_rows)} NONFINITE run(s) recorded in {os.path.join(root, 'ledger.csv')} (prereg "
                  f"stopping rule (i)); investigate, or pass --continue-on-failure", flush=True)
            return 2
    if a.smoke:
        gate_str = "smoke (no production gate)"
    else:
        import expA_gate as G
        gate = G.production_gate(P, a.out_root, smoke_T=a.gate_smoke_T, prereg=a.prereg)
        print(G.format_gate(gate), flush=True)
        if gate["ok"]:
            gate_str = "production gate satisfied (records PASS)"
        elif a.override_gate is not None:
            failed = [i["item"] for i in gate["items"] if not i["ok"]]
            gate_str = f"OVERRIDE {a.override_gate}: failed items {failed}"
            print(f"WARNING: production gate NOT satisfied; overridden by the amendment {a.override_gate!r} (logged in "
                  f"every provenance row)", flush=True)
        else:
            print("REFUSED (production gate): production starts only after G-A1/G-A2 and all nine checks pass "
                  "(prereg 1.3)" + ("; dry run" if a.dry_run else ""), flush=True)
            return 2
    if a.dry_run or not todo:
        print("dry run" if a.dry_run else "nothing to run", flush=True)
        return 0

    os.makedirs(root, exist_ok=True)
    lock = open(os.path.join(root, ".run_cells.lock"), "w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"REFUSED: another driver holds {lock.name}", flush=True)
        return 2
    pause_log = os.path.join(root, "disk_pause.log")

    def log_pause(msg):
        print(msg, flush=True)
        with open(pause_log, "a") as fh:
            fh.write(msg + "\n")

    ledgers = R._Ledgers({C.EXPERIMENT: root})
    n_fail = 0
    first_fail = None
    t0 = time.time()
    try:
        R.warm_cache()
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            pending, queue, done_n, broken = {}, list(todo), 0, None

            def collect(done):
                nonlocal n_fail, done_n, broken, first_fail
                for f in done:
                    job = pending.pop(f)
                    t_sub = job.get("_submitted", time.time())
                    try:
                        out = f.result()
                    except Exception as e:  # noqa: BLE001  (a dead worker / broken pool: FAILED, never complete)
                        broken = broken or e
                        r = R._row(job, f"FAILED: worker {type(e).__name__}: {e}", time.time() - t_sub,
                                   float("nan"), -1)
                        out = dict(row=r, prov=dict(experiment=job["experiment"], variant=job["variant"],
                                                    N=job["N"], seed=job["seed"], method=job["method"],
                                                    n_steps=job["n_steps"], status=r["status"]),
                                   start_epoch=t_sub, elapsed_s=time.time() - t_sub)
                    r = out["row"]
                    ledgers.write(r)
                    rows = [C.compute_row(stage, C.job_item(job), out["elapsed_s"], out["start_epoch"],
                                          note=r["status"][:120])]
                    if out.get("prior_sessions_s", 0.0) > 0:
                        rows.append(C.compute_row(stage, C.job_item(job) + " (earlier interrupted sessions)",
                                                  out["prior_sessions_s"], out["start_epoch"],
                                                  note="checkpointed work + <= 1 checkpoint interval per interrupted "
                                                       "session (upper bound), from the engine session record"))
                    C.append_csv(a.compute_ledger, C.COMPUTE_FIELDS, rows)
                    C.append_csv(os.path.join(root, "run_provenance.csv"), C.PROV_FIELDS,
                                 [dict(out["prov"], gate=gate_str)])
                    ok = r["status"].startswith("complete")
                    n_fail += not ok
                    if not ok and first_fail is None:
                        first_fail = f"{r['variant']} N {r['N']} seed {r['seed']} {r['method']}: {r['status']}"
                    done_n += 1
                    print(f"[{done_n}/{len(todo)}] {r['variant']} N {r['N']} seed {r['seed']} {r['method']}: "
                          f"{r['status']} {out['elapsed_s'] / 60:.1f} min (elapsed {(time.time() - t0) / 3600:.2f} h)",
                          flush=True)

            def pause_sleep(s):
                """During a disk pause keep collecting finished jobs (ledgers stay current); start nothing."""
                if pending:
                    done, _ = wait(list(pending), timeout=s, return_when=FIRST_COMPLETED)
                    collect(done)
                else:
                    time.sleep(s)

            stopped = None

            def halt():
                return (stopped is not None or broken is not None
                        or (first_fail is not None and not a.continue_on_failure))

            while queue or pending:
                while queue and len(pending) < a.workers and not halt():
                    try:
                        C.wait_for_disk(a.disk_path, log=log_pause, sleep=pause_sleep)
                    except C.DiskPauseTimeout as e:
                        stopped = e
                        break
                    if len(pending) >= a.workers:
                        continue
                    job = queue.pop(0)
                    job["_submitted"] = time.time()
                    pending[ex.submit(C.run_timed, {k: v for k, v in job.items() if k != "_submitted"}, root)] = job
                if halt():
                    queue = []
                if pending:
                    done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                    collect(done)
        if stopped is not None:
            print(f"STOPPED: {stopped} (prereg stopping rule iv); running jobs were allowed to finish and are "
                  f"ledgered; {len(todo) - done_n} job(s) not started", flush=True)
            return 3
        if broken is not None:
            print(f"STOPPED: the worker pool broke ({type(broken).__name__}: {broken}); affected jobs are FAILED in "
                  f"the ledger; {len(todo) - done_n} job(s) not started; re-run the driver to resume", flush=True)
            return 1
        if first_fail is not None and not a.continue_on_failure:
            print(f"STOPPED on the first failed run ({first_fail}; prereg stopping rule (i)): running jobs were allowed "
                  f"to finish and are ledgered; {len(todo) - done_n} job(s) not started; investigate (or "
                  f"--continue-on-failure)", flush=True)
            return 1
    finally:
        ledgers.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    print(f"done ({n_fail} failed)", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
