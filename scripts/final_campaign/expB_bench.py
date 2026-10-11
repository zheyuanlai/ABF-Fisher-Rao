#!/usr/bin/env python
"""Experiment B feasibility benchmark, gate, competitor completion and implementation rule (prereg section 2.4-2.5).

    # the whole benchmark of one system (every run on its own exclusive CCD set; resumable; logs every attempt)
    python scripts/final_campaign/expB_bench.py bench --system gateway
    python scripts/final_campaign/expB_bench.py bench --system lta150 --dry-run         # plan + projected cost only
    # summary, F1 / F1b gate, competitor completion and the per-arm implementation -> decision.json
    python scripts/final_campaign/expB_bench.py decide --system gateway

Protocol (frozen).  Production configurations truncated to a fixed walker-step count (gateway 1.6384e8 = 5 % of B,
LTA 150 K 3.072e7 = 10 % of B; n_steps = walker-steps / N; FR active), the Experiment B save grid on the truncated
n_steps.  Candidates ABF N in {1, ..., 512}, FR N in {64, 128, 256, 512}; implementations: the parallel engine in
counter mode at t in {1, 2, 4, 8, 16} (t <= N) and the validated serial engine (diagnostics off, one core, its replay
noise), timed like scripts/mechanism/parallel_benchmark.py (expB_common.run_serial); descriptive t = 32 for N >= 64
(four CCDs of one node; never enters the gate, c* or the implementation rule).  3 repeats in rotated order (repeat r
runs the plan rotated by r L / 3 and uses seed bench_seeds[r]); median walker-steps per second over the repeats.
Each run is a fresh process pinned (taskset) to its first t cores of an exclusive CCD set (expB_alloc: idle check,
30 s poll up to 30 min, then least loaded + flag) with the telemetry monitor running; its JSON records throughput,
peak memory, the timer split (force/move, ABF deposit + bias read, synchronisation, FR, saves), the allocation and the
whole-run contention verdict.  Every attempt (start, completion, failure) is appended to attempts.jsonl; a complete
run is never re-run or overwritten (the first complete attempt decides).

Gate per system (both required): F1 for both N = 512 arms best parallel throughput over t <= 16 >= 6.0 x the serial
engine at N = 512; F1b for FR 512 best parallel throughput over t <= 16 >= 4.0 x the parallel engine's own t = 1.
Competitor completion: W_proj(N, method) = Q50_s[tau_e_F_mid_u(N, method, s)] x B x c*(N, method), tau from the
equal-budget production summary (censoring-aware median, eqb_metrics.quantile_inf), c* the best median wall per
walker-step over the candidate implementations; the ABF and the FR candidate with the smallest W_proj are added as
arms if they are not core arms.  Implementation per arm: the fastest median throughput among {serial, parallel t in
{1, 2, 4, 8, 16}, t <= N}; within 3 % of the best (throughput >= 0.97 x best) counts as tied; ties go to fewer cores
(the serial engine = 1 core, winning a tie with parallel t = 1).

Attempts.  Each attempt has its own id: the child writes runs/<config>_r<k>.a<id>.partial (+ .a<id>.telemetry.jsonl,
.a<id>.log) and the driver moves the result to runs/<config>_r<k>.json after the contention check, so two attempts
can never write the same file.  The CCD lock carries the job key; a driver never launches an item whose key is held by
a live lock (an attempt orphaned by a driver that died: it waits for it), and a complete orphaned attempt found
later (its .partial) is ingested as that item's first complete attempt.  On exit by an exception or signal the driver
terminates its running child (logged as a failed attempt).  A disk pause longer than 24 h stops the benchmark.

Decision.  decision.json records the engine source hashes (identical over every benchmark run of an implementation,
else production is not licensed; they must also equal the V-B4 counter runs' hashes when those exist), and licenses
production only if the gate passes AND every core and completion arm has an implementation.  Competitor completion
uses tau_e_F_mid_u (the prereg's tau_mid,u; default) and records the tau_bgrid_e_F_mid_u variant (the equal-budget
summary's cross-N tau) beside it; if the two select different completion arms, production is not licensed until the
choice is fixed explicitly (``decide --tau-key``, with an amendment).

Outputs results/final_campaign/expB/bench/<system>/: runs/<config>_r<k>.json (+ .telemetry.npz), attempts.jsonl,
summary.json, decision.json; ledger rows in results/final_campaign/compute_ledger.csv (stage expB-bench-<system>).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import secrets
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import expB_common as C  # noqa: E402
import expB_alloc as A  # noqa: E402

BENCH_VERSION = "expB_bench/1"
F1_THRESHOLD = 6.0
F1B_THRESHOLD = 4.0
TIE_FRAC = 0.03
TAU_KEY = "tau_bgrid_e_F_mid_u"           # Amendment 2 (A2.3): the equal-budget cross-N tau (common budget grid)
TAU_KEY_ALT = "tau_e_F_mid_u"             # recorded beside it: full-grid persistence (physical checkpoints differ by N)
DRIFT = dict(system="gateway", method="fr", N=512, threads=16, budget_fraction=0.02, seed=8600, noise="counter",
             every_s=1800.0)


# ================================================================================================ plan
def config_id(item):
    base = f"{item['method']}{item['N']}_{item['impl']}"
    return base if item["impl"] == "serial" else f"{base}_t{item['threads']}"


def bench_plan(spec, include_descriptive=True):
    """Every (method, N, implementation, threads) of the frozen benchmark (descriptive t = 32 flagged)."""
    items = []
    for method, Ns in (("abf", spec["bench_abf_N"]), ("fr", spec["bench_fr_N"])):
        for N in Ns:
            items.append(dict(method=method, N=int(N), impl="serial", threads=1, descriptive=False))
            for t in spec["thread_counts"]:
                if t <= N:
                    items.append(dict(method=method, N=int(N), impl="parallel", threads=int(t), descriptive=False))
            if include_descriptive and N >= 64:
                for t in spec["descriptive_threads"]:
                    items.append(dict(method=method, N=int(N), impl="parallel", threads=int(t), descriptive=True))
    for it in items:
        it["config"] = config_id(it)
    return items


def rotated_order(items, repeats=3):
    """[(repeat, item)]: repeat r runs the plan rotated left by floor(r L / repeats)."""
    L = len(items)
    out = []
    for r in range(int(repeats)):
        s = (r * L) // int(repeats)
        out.extend((r, it) for it in items[s:] + items[:s])
    return out


def run_paths(out_dir, item, r):
    stem = os.path.join(out_dir, "runs", f"{item['config']}_r{r}")
    return stem + ".json", stem + ".telemetry.npz", stem + ".telemetry.jsonl", stem + ".log"


def attempt_paths(out_dir, item, r, attempt):
    """Per-attempt files (the child's result, its telemetry, its log, the marker of an attempt already in the
    ledger)."""
    stem = os.path.join(out_dir, "runs", f"{item['config']}_r{r}.a{attempt}")
    return dict(partial=stem + ".partial", tel_jsonl=stem + ".telemetry.jsonl", log=stem + ".log",
                ledger_mark=stem + ".ledger_logged", attempt=str(attempt))


def bench_job_key(system, item, r, seed):
    return f"bench|{system}|{item['config']}|s{int(seed)}|r{int(r)}"


def bench_signature(spec, item, r, seed, engine_module=None):
    n_steps = C.n_steps_for(spec["bench_walker_steps"], item["N"])
    src = C.engine_sources(spec, item["impl"], engine_module)
    sig = dict(version=BENCH_VERSION, system=spec["system"], config=item["config"], repeat=int(r), seed=int(seed),
               walker_steps=int(spec["bench_walker_steps"]), n_steps=n_steps,
               grid_sha=C.sha256_array(C.expb_save_grid(n_steps)[0]),
               engine_cfg=hashlib.sha256(json.dumps(spec["engine_cfg"], sort_keys=True).encode()).hexdigest()[:16],
               engine_sources=C.files_sha(src))
    return sig


def is_complete(json_path, sig):
    if not os.path.exists(json_path):
        return False
    try:
        d = C.read_json(json_path)
    except (OSError, ValueError):
        return False
    return d.get("status") == "complete" and d.get("signature") == json.loads(json.dumps(sig))


# ================================================================================================ one run (child)
def run_one(system, method, N, impl, threads, seed, walker_steps, out_json, engine_module=None, spec=None,
            telemetry=None, hold=False, hold_timeout=A.HOLD_TIMEOUT_S):
    """The benchmark run itself (inside the pinned child): writes the run JSON (no arrays).  hold (with the
    attempt's telemetry path): keep the monitor's boundary intervals outside the timed run
    (expB_alloc.hold_for_monitor)."""
    host = C.host_info()            # affinity at start: libgomp later binds the main thread to its first place
    spec = spec or C.system_spec(system)
    proc_start = A.proc_start_time(os.getpid())
    n_steps = C.n_steps_for(walker_steps, N)
    saves, _ = C.expb_save_grid(n_steps)
    holds = {}
    if hold and telemetry:
        if impl == "serial":
            import parallel_benchmark  # noqa: F401
            __import__(spec["validated_module"])
        else:
            mod = C.parallel_engine(system, engine_module)
            if callable(getattr(mod, "load_kernel", None)):  # gateway: build / load the C kernel (runs gcc --version)
                mod.load_kernel()
        holds["start"] = A.hold_for_monitor(telemetry, after=time.time(), timeout=hold_timeout, boundary="start")
    if impl == "serial":
        res = C.run_serial(spec, method, int(N), int(seed), n_steps, saves)
    else:
        res = C.run_parallel(spec, method, int(N), int(seed), n_steps, saves, int(threads), "counter",
                             module=engine_module)
    wall = float(res["wall_total_s"])
    ws = int(N) * int(n_steps)
    timers = {k: float(res[k]) for k in C.TIMER_KEYS}
    rec = dict(status="complete", version=BENCH_VERSION, system=system, method=method, N=int(N), impl=impl,
               threads=int(threads), seed=int(seed), walker_steps=ws, n_steps=int(n_steps), n_saves=int(saves.size),
               wall_total_s=wall, warmup_s=float(res["warmup_s"]), throughput=ws / wall,
               us_per_walker_step=wall * 1e6 / ws, wall_per_walker_step=wall / ws, timers=timers,
               timer_fractions={k: (v / wall if math.isfinite(v) else None) for k, v in timers.items()},
               peak_rss_mb=float(res.get("peak_rss_mb", math.nan)), engine=str(res["engine"]), pid=os.getpid(),
               noise_mode=str(res["noise_mode"]), engine_threads=int(res["threads"]),
               t0_epoch=float(res["t0_epoch"]), t0_epoch_source=res.get("t0_epoch_source"),
               fr_deaths=int(np.asarray(res["fr_deaths_cum"] if spec["kind"] == "gateway" else res["cum_deaths"])[-1]),
               science_sha256={k: C.sha256_array(res[k]) for k in C.SCORE_KEYS[spec["kind"]]},
               meta=C.parse_meta(res["meta"]), host=host, cpus_after_run=C.host_info()["cpus"],
               proc_start_ticks=proc_start,
               env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OMP_PLACES",
                                                   "OMP_PROC_BIND", "OMP_WAIT_POLICY", "NUMBA_THREADING_LAYER")},
               finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    rec["process_age_s"] = A.process_age_s()
    C.write_json(out_json, rec)
    if hold and telemetry:
        holds["end"] = A.hold_for_monitor(telemetry, after=time.time(), timeout=hold_timeout)
    if holds:
        rec["holds"] = holds
        C.write_json(out_json, rec)
    return rec


# ================================================================================================ driver
def log_attempt(path, **rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rec = dict(rec, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def projected_cost(spec, order, us_guess=None):
    """Rough core-hours of a benchmark plan from per-walker-step guesses (serial: the equal-budget config's estimate;
    parallel: the same / t; + 10 s overhead per run).  A projection, never a measurement."""
    us = float(us_guess if us_guess is not None else spec["P"].get("est_us_per_walker_step", 1.0))
    tot, wall = 0.0, 0.0
    for _r, it in order:
        w = spec["bench_walker_steps"] * us * 1e-6 / (1.0 if it["impl"] == "serial" else max(1, it["threads"]) * 0.7)
        w += 10.0
        wall += w
        tot += it["threads"] * w
    return dict(core_h=tot / 3600.0, serial_wall_h=wall / 3600.0, n_runs=len(order), us_per_walker_step_guess=us)


def _finish_attempt(out_dir, it, r, seed, sig, ap, tel, alloc_rec, proc_wall, attempts, log, recovered=False):
    """Move one complete attempt's result (its .partial) to the item's run JSON with the whole-run contention."""
    jp, tnpz, _tj, _lg = run_paths(out_dir, it, r)
    rec = C.read_json(ap["partial"])
    ct = A.contention(tel, float(rec["t0_epoch"]), float(rec["wall_total_s"]), it["threads"], pid=rec.get("pid"))
    rec.update(signature=sig, descriptive=bool(it["descriptive"]), config=it["config"], repeat=int(r),
               allocation=alloc_rec, proc_wall_s=proc_wall, attempt=ap["attempt"],
               telemetry=os.path.basename(tnpz) if tel is not None else None, contention=ct, recovered=bool(recovered))
    C.write_json(jp, rec)
    os.remove(ap["partial"])
    log_attempt(attempts, event="complete", config=it["config"], repeat=r, seed=seed, throughput=rec["throughput"],
                contended=ct["status"], proc_wall_s=proc_wall, recovered=bool(recovered))
    log(f"  {it['config']} r{r}: {rec['throughput']:.4g} walker-steps/s ({ct['status']})"
        + (" [recovered orphaned attempt]" if recovered else ""))
    return rec


def _ingest_leftovers(out_dir, it, r, seed, sig, attempts, ledger, system, log):
    """A complete attempt orphaned by an earlier driver (its .partial, no live lock): ingest the earliest one as the
    item's first complete attempt (whole-run contention from whatever telemetry it has; a ledger row from the
    child's process age).  Returns True if one was ingested."""
    stem = os.path.join(out_dir, "runs", f"{it['config']}_r{r}.a")
    parts = sorted(glob.glob(stem + "*.partial"), key=os.path.getmtime)
    done = False
    for pp in parts:
        attempt = pp[len(stem):-len(".partial")]
        ap = attempt_paths(out_dir, it, r, attempt)
        try:
            rec = C.read_json(pp)
        except (OSError, ValueError):
            rec = {}
        if done or rec.get("status") != "complete":
            os.replace(pp, pp + ".unused")
            continue
        _jp, tnpz, _tj, _lg = run_paths(out_dir, it, r)
        tel = A.finalize_telemetry(ap["tel_jsonl"], tnpz)
        wall = rec.get("process_age_s")
        wall = float(wall) if wall is not None and math.isfinite(float(wall)) else float(rec["wall_total_s"])
        mark = ap["ledger_mark"]
        if os.path.exists(mark):                  # counted when the driver terminated it
            os.remove(mark)
        else:
            C.ledger_append(C.ledger_row(f"expB-bench-{system}", f"{it['config']} r{r} s{seed}", it["threads"], wall,
                                         note=f"recovered orphaned attempt {attempt} (wall = the child's process "
                                              f"age)"), ledger)
        _finish_attempt(out_dir, it, r, seed, sig, ap, tel, None, wall, attempts, log, recovered=True)
        done = True
    return done


def run_bench(system, out_root=C.OUT_ROOT, registry_root=A.REGISTRY_DIR, allocator=None, engine_module=None,
              dry_run=False, repeats=None, only=None, ledger=C.LEDGER, monitor=True, monitor_allowed=None,
              spec=None, log=print, max_runs=None, child_python=sys.executable, pin_self=True,
              run_timeout_s=6 * 3600.0, guard=None, orphan_poll_s=30.0, sleep=time.sleep):
    spec = spec or C.system_spec(system)
    out_dir = os.path.join(out_root, "bench", system)
    plan = bench_plan(spec)
    if only:
        plan = [it for it in plan if it["config"] in only]
    order = rotated_order(plan, repeats or spec["bench_repeats"])
    proj = projected_cost(spec, order)
    log(f"{system}: {len(plan)} configurations x {repeats or spec['bench_repeats']} repeats = {len(order)} runs; "
        f"projected ~{proj['core_h']:.1f} core-h (rough, from the config's us/walker-step estimate)")
    if dry_run:
        return dict(plan=plan, order=[(r, it["config"]) for r, it in order], projection=proj)
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    attempts = os.path.join(out_dir, "attempts.jsonl")
    reg = A.Registry(registry_root)
    alloc = allocator or A.Allocator(reg, log=log, reserved=A.ccds_of_cpus(monitor_allowed) if monitor_allowed
                                     else ())
    guard = guard or C.DiskGuard(log=log)
    mon = A.start_monitor(registry_root, allowed=monitor_allowed) if monitor else None
    done = 0
    core = None
    stopped = None
    try:
        for r, it in order:
            if max_runs is not None and done >= max_runs:
                break
            if pin_self:                              # the driver never sits on an allocation
                core = A.pin_outside(reg, monitor_allowed, current=core, log=log)
            if mon is not None and mon.poll() is not None:
                log("  monitor process exited; restarting it")
                mon = A.start_monitor(registry_root, allowed=monitor_allowed)
            seed = spec["bench_seeds"][r]
            jp, tnpz, _tj, _lg = run_paths(out_dir, it, r)
            sig = bench_signature(spec, it, r, seed, engine_module)
            if is_complete(jp, sig):
                continue
            if os.path.exists(jp):
                old = C.read_json(jp)
                if old.get("status") == "complete":
                    raise C.ExpBError(f"{jp} holds a complete run with a DIFFERENT signature; refusing to overwrite")
            key = bench_job_key(system, it, r, seed)
            waited_orphan = False
            while key in reg.live_job_keys():        # an attempt of an earlier driver is still running: never twice
                if not waited_orphan:
                    log(f"  {it['config']} r{r}: an orphaned attempt is still running; waiting for it")
                    log_attempt(attempts, event="wait_orphan", config=it["config"], repeat=r, seed=seed)
                    waited_orphan = True
                sleep(orphan_poll_s)
            if _ingest_leftovers(out_dir, it, r, seed, sig, attempts, ledger, system, log):
                done += 1
                continue
            try:
                guard.wait()
            except C.DiskPauseTimeout as e:
                stopped = f"disk pause timeout (stopping rule iv): {e}"
                log_attempt(attempts, event="stop", reason=stopped)
                log(f"  STOP: {stopped}")
                break
            attempt = secrets.token_hex(4)
            ap = attempt_paths(out_dir, it, r, attempt)
            run_id = f"bench-{system}-{it['config']}-r{r}-{os.getpid()}"
            a = alloc.allocate(it["threads"], run_id, info=dict(
                kind="bench", telemetry=ap["tel_jsonl"], job_key=key, attempt=attempt,
                driver=dict(pid=os.getpid(), start=A.proc_start_time(os.getpid()))))
            if pin_self:                              # move off the new allocation before the run starts
                core = A.pin_outside(reg, monitor_allowed, current=core, log=log)
            log_attempt(attempts, event="start", config=it["config"], repeat=r, seed=seed, ccds=a["ccds"],
                        cores=a["cores"], flagged=a["flagged"], wait_s=a["wait_s"], attempt=attempt)
            cmd = [child_python, os.path.abspath(__file__), "run-one", "--system", system, "--method", it["method"],
                   "--N", str(it["N"]), "--impl", it["impl"], "--threads", str(it["threads"]), "--seed", str(seed),
                   "--walker-steps", str(spec["bench_walker_steps"]), "--out", ap["partial"],
                   "--telemetry", ap["tel_jsonl"]]
            if monitor:
                cmd += ["--hold"]
            if engine_module and it["impl"] == "parallel":
                cmd += ["--engine-module", engine_module]
            t_launch = time.time()
            p = None
            rc = None
            try:
                p = A.launch_pinned(cmd, a["cores"], log_path=ap["log"])
                try:
                    reg.add_pid(a["ccds"], a["token"], p.pid)
                except RuntimeError:
                    pass
                try:
                    rc = p.wait(timeout=run_timeout_s)
                except Exception:  # noqa: BLE001  (subprocess.TimeoutExpired: our own hung child)
                    rc = A.kill_process_group(p)
                    log(f"  TIMEOUT {it['config']} r{r}: terminated after {run_timeout_s:.0f} s")
            except BaseException:                     # driver exit (exception / signal): terminate our child
                if p is not None and p.poll() is None:
                    rc = A.kill_process_group(p)
                proc_wall = time.time() - t_launch
                alloc.release(a)
                C.ledger_append(C.ledger_row(f"expB-bench-{system}", f"{it['config']} r{r} s{seed}", it["threads"],
                                             proc_wall, note=f"terminated at driver exit (rc {rc}); ccds {a['ccds']}"),
                                ledger)
                log_attempt(attempts, event="failed", config=it["config"], repeat=r, seed=seed, rc=rc,
                            proc_wall_s=proc_wall, attempt=attempt, reason="terminated at driver exit")
                if os.path.exists(ap["partial"]):     # it had finished: kept for ingestion on the next start,
                    with open(ap["ledger_mark"], "w") as fh:                                    # already counted
                        fh.write(f"{proc_wall}\n")
                elif os.path.exists(ap["tel_jsonl"]):
                    os.remove(ap["tel_jsonl"])
                raise
            proc_wall = time.time() - t_launch
            alloc.release(a)
            C.ledger_append(C.ledger_row(f"expB-bench-{system}", f"{it['config']} r{r} s{seed}", it["threads"],
                                         proc_wall, note=f"rc {rc}; ccds {a['ccds']}; flagged {a['flagged']}"),
                            ledger)
            if rc != 0 or not os.path.exists(ap["partial"]):
                if os.path.exists(ap["tel_jsonl"]):
                    os.remove(ap["tel_jsonl"])
                log_attempt(attempts, event="failed", config=it["config"], repeat=r, seed=seed, rc=rc,
                            proc_wall_s=proc_wall, log=ap["log"], attempt=attempt)
                log(f"  FAILED {it['config']} r{r} (rc {rc}); see {ap['log']}")
                continue
            tel = A.finalize_telemetry(ap["tel_jsonl"], tnpz)
            _finish_attempt(out_dir, it, r, seed, sig, ap, tel, {k: v for k, v in a.items() if k != "token"},
                            proc_wall, attempts, log)
            done += 1
    finally:
        A.stop_monitor(mon)
    return dict(runs_done=done, stopped=stopped)


# ================================================================================================ summary / decision
def load_bench_runs(out_dir):
    runs = []
    d = os.path.join(out_dir, "runs")
    if not os.path.isdir(d):
        return runs
    for f in sorted(os.listdir(d)):
        if f.endswith(".json") and not f.endswith(".partial"):
            try:
                rec = C.read_json(os.path.join(d, f))
            except (OSError, ValueError):
                continue
            if rec.get("status") == "complete":
                runs.append(rec)
    return runs


def summarize(runs, repeats=3, uncontended_only=False):
    """{config: dict(method, N, impl, threads, descriptive, n, median throughput, median wall per walker-step,
    median timer fractions, peak memory, contention counts)}."""
    by = {}
    for r in runs:
        by.setdefault(r["config"], []).append(r)
    out = {}
    for cfg, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda x: x["repeat"])
        allr = rs
        if uncontended_only:
            rs = [x for x in rs if x.get("contention", {}).get("status") == "uncontended"]
        r0 = allr[0]
        thr = [float(x["throughput"]) for x in rs]
        med_thr = float(np.median(thr)) if thr else math.nan
        tf = {}
        for k in C.TIMER_KEYS:
            v = [x["timer_fractions"].get(k) for x in rs if x["timer_fractions"].get(k) is not None]
            tf[k] = float(np.median(v)) if v else None
        out[cfg] = dict(method=r0["method"], N=int(r0["N"]), impl=r0["impl"], threads=int(r0["threads"]),
                        descriptive=bool(r0.get("descriptive", False)), n=len(rs), n_planned=int(repeats),
                        throughput=med_thr, throughput_all=thr,
                        wall_per_walker_step=(1.0 / med_thr if med_thr and math.isfinite(med_thr) else math.nan),
                        us_per_walker_step=(1e6 / med_thr if med_thr and math.isfinite(med_thr) else math.nan),
                        timer_fractions=tf, peak_rss_mb=max([float(x.get("peak_rss_mb", math.nan)) for x in rs]
                                                            or [math.nan]),
                        n_contended=sum(1 for x in allr if x.get("contention", {}).get("status") != "uncontended"),
                        n_flagged=sum(1 for x in allr if x.get("allocation", {}).get("flagged")),
                        repeats=[int(x["repeat"]) for x in rs])
    return out


def _thr(summary, method, N, impl, t=1):
    cfg = config_id(dict(method=method, N=N, impl=impl, threads=t))
    s = summary.get(cfg)
    return (s["throughput"] if s and s["n"] > 0 else math.nan), s


def evaluate_gate(summary, spec):
    """F1 (both N = 512 arms, best parallel over t <= 16 >= 6 x serial) and F1b (FR 512 best parallel >= 4 x own
    t = 1); INCOMPLETE when a needed median is missing."""
    ts = [t for t in spec["thread_counts"] if t <= 16]
    res = dict(F1={}, F1b={}, rng_only_speedup={})
    complete = True
    for method in ("abf", "fr"):
        par = {t: _thr(summary, method, 512, "parallel", t)[0] for t in ts}
        ser = _thr(summary, method, 512, "serial")[0]
        fin = {t: v for t, v in par.items() if math.isfinite(v)}
        best_t = max(fin, key=lambda t: fin[t]) if fin else None
        best = fin[best_t] if fin else math.nan
        ratio = best / ser if (fin and math.isfinite(ser) and ser > 0) else math.nan
        ok = bool(math.isfinite(ratio) and ratio >= F1_THRESHOLD)
        complete &= len(fin) == len(ts) and math.isfinite(ser)
        res["F1"][method] = dict(best_parallel_throughput=best, best_threads=best_t, serial_throughput=ser,
                                 ratio=ratio, threshold=F1_THRESHOLD, passed=ok, parallel_by_threads=par)
    par = {t: _thr(summary, "fr", 512, "parallel", t)[0] for t in ts}
    fin = {t: v for t, v in par.items() if math.isfinite(v)}
    best_t = max(fin, key=lambda t: fin[t]) if fin else None
    t1 = par.get(1, math.nan)
    ratio = fin[best_t] / t1 if (fin and math.isfinite(t1) and t1 > 0) else math.nan
    res["F1b"] = dict(best_parallel_throughput=fin.get(best_t, math.nan), best_threads=best_t,
                      parallel_t1_throughput=t1, ratio=ratio, threshold=F1B_THRESHOLD,
                      passed=bool(math.isfinite(ratio) and ratio >= F1B_THRESHOLD))
    for cfg, s in summary.items():
        if s["impl"] == "parallel" and s["threads"] == 1:
            ser = _thr(summary, s["method"], s["N"], "serial")[0]
            res["rng_only_speedup"][f"{s['method']}{s['N']}"] = (s["throughput"] / ser if math.isfinite(ser) and ser
                                                                  else math.nan)
    f1 = all(res["F1"][m]["passed"] for m in ("abf", "fr"))
    if not complete or not math.isfinite(res["F1b"]["ratio"]):
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if (f1 and res["F1b"]["passed"]) else "FAIL"
    res.update(F1_pass=f1, F1b_pass=res["F1b"]["passed"], verdict=verdict,
               rule="F1: for abf512 and fr512, best parallel throughput over t <= 16 >= 6.0 x serial engine at N 512; "
                    "F1b: fr512 best parallel throughput over t <= 16 >= 4.0 x parallel t = 1")
    return res


def implementation_for(summary, method, N, thread_counts, tie_frac=TIE_FRAC):
    """Fastest median throughput among {serial, parallel t <= N, t in thread_counts}; tied within tie_frac of the best
    (throughput >= (1 - tie_frac) x best) -> fewest cores (serial = 1 core, before parallel t = 1)."""
    cands = [("serial", 1)] + [("parallel", int(t)) for t in thread_counts if t <= N]
    table = []
    for impl, t in cands:
        v, s = _thr(summary, method, N, impl, t)
        table.append(dict(impl=impl, threads=t, cores=1 if impl == "serial" else t, throughput=v,
                          n=(s["n"] if s else 0)))
    fin = [x for x in table if math.isfinite(x["throughput"])]
    if not fin:
        return None, table
    best = max(x["throughput"] for x in fin)
    tied = [x for x in fin if x["throughput"] >= (1.0 - tie_frac) * best]
    choice = min(tied, key=lambda x: (x["cores"], 0 if x["impl"] == "serial" else 1))
    for x in table:
        x["tied_with_best"] = bool(math.isfinite(x["throughput"]) and x["throughput"] >= (1.0 - tie_frac) * best)
    return dict(choice, wall_per_walker_step=1.0 / choice["throughput"], best_throughput=best), table


_EQB_SUMMARY = {}


def eqb_tau_mid(spec, method, N, key="tau_e_F_mid_u"):
    """Per-seed equal-budget persistent mid-threshold budget fractions {seed: u} ("inf" -> inf)."""
    if spec["eqb_summary"] not in _EQB_SUMMARY:
        _EQB_SUMMARY[spec["eqb_summary"]] = C.read_json(spec["eqb_summary"])
    d = _EQB_SUMMARY[spec["eqb_summary"]]
    pn = d["per_N"].get(str(int(N)), {})
    if method not in pn or key not in pn[method]:
        return {}
    return {int(s): C.num(v) for s, v in pn[method][key]["per_seed"].items()}


def competitor_completion(spec, summary, tau_source=None, key=TAU_KEY):
    """W_proj(N, method) = Q50_s[tau_mid_u] x B x c*(N, method) for every benchmark candidate; the smallest per
    method is added as an arm if it is not a core arm.  tau_source(method, N) -> {seed: u} (default eqb_tau_mid with
    the summary key ``key``)."""
    import eqb_metrics as M
    tau_source = tau_source or (lambda m, n: eqb_tau_mid(spec, m, n, key=key))
    core = {C.arm_name(m, n) for m, n in spec["core_arms"]}
    rows = {}
    for method, Ns in (("abf", spec["bench_abf_N"]), ("fr", spec["bench_fr_N"])):
        for N in Ns:
            tau = tau_source(method, N)
            q50 = M.quantile_inf(np.array(list(tau.values()), dtype=float), 50) if tau else math.nan
            _choice, table = implementation_for(summary, method, N, [t for t in spec["thread_counts"] if t <= 16])
            cstar = min([1.0 / x["throughput"] for x in table if math.isfinite(x["throughput"])], default=math.nan)
            w = q50 * spec["B"] * cstar if (math.isfinite(cstar)) else math.nan
            rows[C.arm_name(method, N)] = dict(method=method, N=int(N), tau_mid_u_median=q50, n_tau=len(tau),
                                               n_tau_censored=int(sum(1 for v in tau.values() if math.isinf(v))),
                                               c_star=cstar, W_proj_s=w, core=C.arm_name(method, N) in core)
    added = {}
    for method in ("abf", "fr"):
        cand = [(k, v) for k, v in rows.items() if v["method"] == method and not math.isnan(v["W_proj_s"])]
        if not cand:
            added[method] = None
            continue
        k, v = min(cand, key=lambda kv: (kv[1]["W_proj_s"], kv[1]["N"]))
        added[method] = None if v["core"] else k
        rows[k]["best_of_method"] = True
    return dict(rows=rows, added=added, tau_key=key,
                rule=f"W_proj = Q50_s[{key}] x B x c*, c* = best median wall per walker-step over serial and "
                     "parallel t in {1,2,4,8,16} (t <= N); smallest W_proj per method added if not a core arm")


def engine_provenance(runs):
    """{impl: engine source hashes} over the benchmark runs (from their signatures) and the problems: a run without
    recorded hashes, or more than one distinct hash set for an implementation (the engine changed mid-benchmark)."""
    sets, missing = {}, 0
    for r in runs:
        es = (r.get("signature") or {}).get("engine_sources")
        if not es:
            missing += 1
            continue
        sets.setdefault(r.get("impl"), {})[json.dumps(es, sort_keys=True)] = es
    probs = []
    if missing:
        probs.append(f"{missing} benchmark run(s) without recorded engine source hashes")
    out = {}
    for impl, d in sorted(sets.items()):
        if len(d) > 1:
            probs.append(f"{impl}: {len(d)} distinct engine source hash sets across the benchmark runs (the engine "
                         f"changed during the benchmark)")
        out[impl] = next(iter(d.values())) if len(d) == 1 else None
    return out, probs


def vb4_engine_sources(out_root, system):
    """Distinct engine source hash sets of the V-B4 counter-mode records (equiv/<system>/counter/*/s*.json)."""
    sets = {}
    for f in glob.glob(os.path.join(out_root, "equiv", system, "counter", "*", "s*.json")):
        try:
            es = (C.read_json(f).get("signature") or {}).get("engine_sources")
        except (OSError, ValueError):
            continue
        if es:
            sets[json.dumps(es, sort_keys=True)] = es
    return list(sets.values())


def decide(system, out_root=C.OUT_ROOT, spec=None, tau_source=None, write=True, tau_key=None, tau_source_alt=None):
    """Summary, gate, competitor completion, implementation per arm and the production licence -> decision.json.
    production_licensed iff the gate passes, every core and completion arm has an implementation, the engine source
    hashes are one set per implementation (and equal the V-B4 counter runs' when those exist), and the completion
    arms do not depend on the tau reading (unless ``tau_key`` fixes it explicitly)."""
    spec = spec or C.system_spec(system)
    out_dir = os.path.join(out_root, "bench", system)
    runs = load_bench_runs(out_dir)
    summ = summarize(runs, spec["bench_repeats"])
    summ_u = summarize(runs, spec["bench_repeats"], uncontended_only=True)
    gate = evaluate_gate(summ, spec)
    gate_u = evaluate_gate(summ_u, spec)
    key = tau_key or TAU_KEY
    comp = competitor_completion(spec, summ, tau_source, key=key)
    alt_key = TAU_KEY_ALT if key == TAU_KEY else TAU_KEY
    alt_src = tau_source_alt if tau_source_alt is not None else (None if tau_source is not None else
                                                                   (lambda m, n: eqb_tau_mid(spec, m, n, key=alt_key)))
    comp_alt = competitor_completion(spec, summ, alt_src, key=alt_key) if alt_src is not None else None
    problems = []
    if gate["verdict"] != "PASS":
        problems.append(f"feasibility gate verdict {gate['verdict']}")
    if comp_alt is not None:
        differs = comp_alt["added"] != comp["added"]
        comp["alternative"] = dict(tau_key=alt_key, added=comp_alt["added"],
                                   W_proj_s={k: v["W_proj_s"] for k, v in comp_alt["rows"].items()},
                                   selection_differs=differs)
        if differs and tau_key is None:
            problems.append(f"competitor completion depends on the tau reading ({key}: {comp['added']} vs {alt_key}: "
                            f"{comp_alt['added']}): fix it by an amendment and re-run decide --tau-key")
    ts = [t for t in spec["thread_counts"] if t <= 16]
    arms = [(m, n, "core") for m, n in spec["core_arms"]]
    for m in ("abf", "fr"):
        if comp["added"][m]:
            mm, nn = C.parse_arm(comp["added"][m])
            arms.append((mm, nn, "completion"))
    arm_recs = []
    for m, n, src in arms:
        choice, table = implementation_for(summ, m, n, ts)
        rec = dict(arm=C.arm_name(m, n), method=m, N=int(n), source=src, implementation_table=table)
        if choice is None:
            rec.update(impl=None, threads=None, status="no benchmark data")
        else:
            rec.update(impl=choice["impl"], threads=int(choice["threads"]),
                       noise="replay" if choice["impl"] == "serial" else "counter",
                       throughput=choice["throughput"], wall_per_walker_step=choice["wall_per_walker_step"],
                       projected_run_wall_s=spec["B"] * choice["wall_per_walker_step"])
        arm_recs.append(rec)
    no_impl = [r["arm"] for r in arm_recs if r.get("impl") is None]
    if no_impl:
        problems.append(f"arms without an implementation (no finite benchmark throughput): {no_impl}")
    engine, eprobs = engine_provenance(runs)
    problems.extend(eprobs)
    for impl in sorted({r["impl"] for r in arm_recs if r.get("impl")}):
        if not engine.get(impl):
            problems.append(f"no engine source hashes for the {impl} implementation used by an arm")
    vb4 = vb4_engine_sources(out_root, system)
    vb4_match = None
    if vb4:
        vb4_match = len(vb4) == 1 and engine.get("parallel") == vb4[0]
        if not vb4_match:
            problems.append(f"the benchmarked parallel engine differs from the V-B4 counter runs' engine "
                            f"({len(vb4)} distinct V-B4 hash set(s))")
    dec = dict(version=BENCH_VERSION, system=system, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               n_runs=len(runs), gate=gate, gate_uncontended_only_sensitivity=gate_u,
               production_licensed=not problems, license_problems=problems, competitor_completion=comp,
               arms=arm_recs, engine_sources=engine, vb4_engine_match=vb4_match,
               implementation_rule="fastest median throughput among serial and parallel t in {1,2,4,8,16} (t <= N); "
                                   "within 3 % of the best = tied; ties to fewer cores (serial = 1 core, wins a tie "
                                   "with parallel t = 1); serial-engine arms run the validated replay noise",
               note="validation V-B1..V-B5 must also have passed (prereg 2.3); this file records the benchmark side "
                    "only; expB_run.py prod refuses to start if the current engine sources differ from "
                    "engine_sources")
    if write:
        C.write_json(os.path.join(out_dir, "summary.json"), dict(summary=summ, summary_uncontended_only=summ_u,
                                                                 n_runs=len(runs)))
        C.write_json(os.path.join(out_dir, "decision.json"), dec)
    return dec


# ================================================================================================ drift reference
def drift_job(spec_gw=None, k=0):
    """The fixed drift reference (gateway FR 512, t = 16, 2 % of B, counter mode, seed 8600) as a scheduler job."""
    spec_gw = spec_gw or C.system_spec("gateway")
    ws = int(round(DRIFT["budget_fraction"] * spec_gw["B"]))
    n_steps = C.n_steps_for(ws, DRIFT["N"])
    return dict(kind="drift", system="gateway", arm="drift_fr512", method=DRIFT["method"], N=DRIFT["N"],
                seed=DRIFT["seed"], n_steps=n_steps, impl="parallel", threads=DRIFT["threads"], noise="counter",
                replicate=int(k))


# ================================================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Experiment B feasibility benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--system", required=True, choices=sorted(C.SYSTEMS))
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--repeats", type=int, default=None)
    b.add_argument("--only", nargs="*", default=None, help="config ids (e.g. fr512_parallel_t16)")
    b.add_argument("--engine-module", default=None)
    b.add_argument("--out-root", default=C.OUT_ROOT)
    b.add_argument("--registry", default=A.REGISTRY_DIR)
    b.add_argument("--max-runs", type=int, default=None)
    b.add_argument("--monitor-cpus", default=None, help="cores the monitor may sit on (default: any outside holds)")
    o = sub.add_parser("run-one")
    o.add_argument("--system", required=True)
    o.add_argument("--method", required=True)
    o.add_argument("--N", type=int, required=True)
    o.add_argument("--impl", required=True, choices=("serial", "parallel"))
    o.add_argument("--threads", type=int, required=True)
    o.add_argument("--seed", type=int, required=True)
    o.add_argument("--walker-steps", type=int, required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--engine-module", default=None)
    o.add_argument("--telemetry", default=None, help="this attempt's monitor JSONL (for --hold)")
    o.add_argument("--hold", action="store_true", help="hold the monitor's boundary intervals outside the timed run")
    d = sub.add_parser("decide")
    d.add_argument("--system", required=True, choices=sorted(C.SYSTEMS))
    d.add_argument("--out-root", default=C.OUT_ROOT)
    d.add_argument("--tau-key", default=None, choices=(TAU_KEY, TAU_KEY_ALT),
                   help="fix the competitor-completion tau explicitly (only needed, with an amendment, when the two "
                        "readings select different completion arms)")
    a = ap.parse_args(argv)
    if a.cmd == "run-one":
        run_one(a.system, a.method, a.N, a.impl, a.threads, a.seed, a.walker_steps, a.out, a.engine_module,
                telemetry=a.telemetry, hold=a.hold)
        return 0
    C.limit_driver_threads()
    C.install_exit_signals()
    if a.cmd == "bench":
        r = run_bench(a.system, out_root=a.out_root, registry_root=a.registry, engine_module=a.engine_module,
                      dry_run=a.dry_run, repeats=a.repeats, only=a.only, max_runs=a.max_runs,
                      monitor_allowed=A.parse_cpu_list(a.monitor_cpus) if a.monitor_cpus else None)
        print(json.dumps(r.get("projection", r), default=str, indent=1))
        return 0
    dec = decide(a.system, out_root=a.out_root, tau_key=a.tau_key)
    print(json.dumps(dict(verdict=dec["gate"]["verdict"], production_licensed=dec["production_licensed"],
                          license_problems=dec["license_problems"], added=dec["competitor_completion"]["added"],
                          arms=[(x["arm"], x.get("impl"), x.get("threads")) for x in dec["arms"]]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
