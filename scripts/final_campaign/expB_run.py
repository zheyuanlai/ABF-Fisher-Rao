#!/usr/bin/env python
"""Experiment B production scheduler and V-B4 equivalence runner (prereg sections 2.3, 2.5, 3).

    # production of one system (arms, implementations and threads from bench/<system>/decision.json)
    python scripts/final_campaign/expB_run.py prod --system gateway --cap 45 --validation-passed [--dry-run]
    # V-B4 equivalence runs (counter mode; NOT timing runs: any thread count, single-thread jobs side by side)
    python scripts/final_campaign/expB_run.py equiv --system lta150 --cpus 64-79 [--which counter|validated|all]
                                                    [--extension-arms abf512 fr512]
    python scripts/final_campaign/expB_run.py project --system gateway        # projection only

Production (frozen rules).  Arms = decision.json arms (core arms in config order, then the completion arms ABF, FR),
each with its implementation (parallel counter mode at t_arm threads, or the validated serial engine with its replay
noise).  Seeds = configs/final_campaign/expB.json.  Seed-major blocks; within a seed the arm order is
default_rng(20261011 + seed).permutation(n_arms) of that canonical list; jobs are dispatched strictly in that order
(head of line), each to a free exclusive CCD set (expB_alloc: idle check, poll 30 s, after 30 min least-loaded +
flag), several runs at a time on disjoint sets with at most --max-cores (64) campaign cores in use; every run is a
fresh process pinned with taskset to its t_arm cores (omp_env), with the telemetry monitor running.  A drift
reference (gateway FR 512, t = 16, 2 % of B, counter, seed 8600) is dispatched first and then every 30 min.

Records.  <out>/prod/<system>/<arm>/s<seed>.npz (compressed result arrays) + s<seed>.json (job, signature, timers,
allocation, endpoints, per-endpoint contention, telemetry file) + s<seed>.telemetry.npz.  A repeat k >= 1 is
s<seed>.rep<k>.*; counter mode (and the serial engine) is deterministic, so a repeat stores only its timing arrays
(wall_s_at_save, n_force_evals_at_save) after verifying that its science arrays are bitwise the first replicate's
(sha256), and every record is kept.

Contention and repeats.  After each run the scheduler scores e_F (expB_common.Scorer), takes the wall windows of
every endpoint ([0, W] at the strict / mid / loose thresholds; the whole run for Ibar_F^wall and for censored W) and
applies expB_alloc.contention; a run contended (or with unknown telemetry) for ANY endpoint is repeated (queued
next), until a replicate is uncontended for every endpoint, at most 3 repeats.  The analysis uses, per endpoint, the
first replicate uncontended for that endpoint; seeds with none are flagged (sensitivity analysis only).

Record life cycle.  The run child writes its record with status 'ran'; only the driver's finalisation (telemetry
sidecar, endpoints, contention, ledger row) sets 'complete'.  A driver that exits by an exception, SIGINT, SIGTERM or
SIGHUP terminates its own active run processes (their process groups) and logs them; a record already 'ran' is
finalised first.  After a hard driver death (SIGKILL, host) a restarted driver (i) finalises every 'ran' record
(endpoints from the npz; contention from whatever telemetry exists, 'unknown' -> a repeat), and (ii) never launches
a job whose key is held by a live CCD lock (an orphaned run still executing): it waits for that lock to go stale,
then finalises the orphan's record or, if the orphan failed, runs the job again.  An exception while finalising one
run is logged and never stops the scheduler (the record stays 'ran' and is recovered on the next start).

Resumability, disk, cost.  Complete runs are skipped by signature (a complete file with a different signature is
never overwritten: the scheduler stops); needed repeats are re-derived from the records.  New runs and saves pause
while the root filesystem has < 2 GB free and resume at >= 3 GB (expB_common.DiskGuard); a scheduler-level pause
longer than 24 h stops the stage (logged; active runs drain).  A child that pauses before saving records the paused
seconds (and a marker file the driver reads while it waits): paused time is excluded from the run timeout, from the
ledger and from the stage's used core-hours.  Every process writes a ledger row (results/final_campaign/
compute_ledger.csv; used core-h = threads x (process wall - disk pause)).  Projection gate: the stage is refused if
the projected used core-hours (bench wall per walker-step x B x t_arm + 15 s per run, + drift references) exceed
--cap; during the stage no new run is launched once used + projected remaining would exceed it.  The engine source
hashes must equal those recorded in decision.json (the benchmarked engine), else the stage is refused.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
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
import expB_alloc as A  # noqa: E402

RUN_VERSION = "expB_run/1"
MAX_REPEATS = 3
RUN_OVERHEAD_S = 15.0
DRIFT_EVERY_S = 1800.0
TIMING_KEYS = ("wall_s_at_save", "n_force_evals_at_save")


# ================================================================================================ jobs
def canonical_arms(decision):
    """Arms in the canonical order of decision.json (core arms in config order, then completion arms)."""
    return [a for a in decision["arms"] if a.get("impl")]


def production_order(arm_names, seeds):
    """Seed-major [(arm, seed)]: within a seed the arms permuted by default_rng(20261011 + seed)."""
    out = []
    for s in seeds:
        perm = np.random.default_rng(C.ORDER_SEED_BASE + int(s)).permutation(len(arm_names))
        out.extend((arm_names[int(i)], int(s)) for i in perm)
    return out


def prod_job(spec, arm_rec, seed, replicate=0):
    N = int(arm_rec["N"])
    return dict(kind="prod", system=spec["system"], arm=arm_rec["arm"], method=arm_rec["method"], N=N,
                seed=int(seed), n_steps=C.n_steps_for(spec["B"], N), impl=arm_rec["impl"],
                threads=int(arm_rec["threads"]), noise="replay" if arm_rec["impl"] == "serial" else "counter",
                replicate=int(replicate), grid="expb")


def job_paths(job, out_root):
    k = job["kind"]
    if k == "prod":
        base = os.path.join(out_root, "prod", job["system"], job["arm"], f"s{job['seed']}" +
                            ("" if job["replicate"] == 0 else f".rep{job['replicate']}"))
    elif k == "drift":
        base = os.path.join(out_root, "drift", f"d{job['replicate']:03d}")
    elif k in ("equiv_counter", "equiv_validated"):
        base = os.path.join(out_root, "equiv", job["system"], "counter" if k == "equiv_counter" else "validated",
                            job["arm"], f"s{job['seed']}")
    else:
        raise C.ExpBError(f"unknown job kind {k}")
    return dict(npz=base + ".npz", json=base + ".json", tel_npz=base + ".telemetry.npz",
                tel_jsonl=base + ".telemetry.jsonl", log=base + ".log", base=base, pause=base + ".diskpause.json",
                ledger_mark=base + ".ledger_logged")


def job_grid(job, spec):
    if job.get("grid", "expb") == "eqb":
        return C.eqb_production_save_grid(spec, job["n_steps"])
    return C.expb_save_grid(job["n_steps"])[0]


def job_signature(job, spec, engine_module=None):
    """Everything that defines the science and the timing configuration of a job (not the replicate index)."""
    src = C.engine_sources(spec, job["impl"], engine_module)
    sig = dict(version=RUN_VERSION, kind=job["kind"], system=job["system"], arm=job["arm"], method=job["method"],
               N=int(job["N"]), seed=int(job["seed"]), n_steps=int(job["n_steps"]), impl=job["impl"],
               threads=int(job["threads"]), noise=job["noise"], grid_sha=C.sha256_array(job_grid(job, spec)),
               engine_cfg=hashlib.sha256(json.dumps(spec["engine_cfg"], sort_keys=True).encode()).hexdigest()[:16],
               B=int(spec["B"]), engine_sources=C.files_sha(src))
    return json.loads(json.dumps(sig))


def finalized(rec):
    """True iff a record carries the driver-side results of its kind (endpoints + contention, contention, Ibar_F)."""
    k = (rec.get("job") or {}).get("kind")
    if k == "prod":
        return "endpoints" in rec and "contention_status" in rec
    if k == "drift":
        return "contention" in rec
    if k in ("equiv_counter", "equiv_validated"):
        return rec.get("Ibar_F") is not None
    return True


def record_status(paths, sig):
    """'complete' (finalised, signature matches), 'ran' (the child finished; not finalised yet), 'mismatch' (a
    'ran' / complete record of a DIFFERENT job), 'failed' or None."""
    if not os.path.exists(paths["json"]):
        return None
    try:
        rec = C.read_json(paths["json"])
    except (OSError, ValueError):
        return "failed"
    st = rec.get("status")
    if st not in ("complete", "ran"):
        return st or "failed"
    if rec.get("signature") != sig:
        return "mismatch"
    return "complete" if (st == "complete" and finalized(rec)) else "ran"


# ================================================================================================ the run (child)
def _validated_run(spec, job, saves, ckpt):
    """The validated engine at full budget (V-B4 reference runs): resumable via the engine's checkpoint."""
    if spec["kind"] == "gateway":
        import gateway_ladder_numba as E
        res = E.run_arm(dict(spec["engine_cfg"]), job["method"], job["N"], job["seed"], job["n_steps"], saves,
                        diagnostics=False, checkpoint_path=ckpt, checkpoint_every_s=600.0)
        meta = res.pop("meta")
        res.pop("cfg", None)
    else:
        import lta_ladder_numba as E
        c = E.make_cfg(float(spec["T_K"]), job["N"], job["seed"],
                       **dict(spec["engine_cfg"], diagnostics=False, n_steps=job["n_steps"]))
        res = E.run_arm(c, job["method"], saves, checkpoint_path=ckpt, checkpoint_every_s=900.0)
        meta = json.loads(str(res.pop("meta_json")))
        res.pop("cfg_json", None)
    for k in ("sessions_json", "resumed_from", "n_checkpoints_written"):
        res.pop(k, None)
    res["meta"] = json.dumps(dict(engine=meta.get("engine"), engine_sha256=meta.get("engine_sha256"),
                                  git_commit=meta.get("git_commit"), implementation="validated (V-B4 reference)"))
    res["engine"] = f"{meta.get('engine')} (validated)"
    res["noise_mode"] = "replay"
    res["threads"] = 1
    res["wall_total_s"] = float(res.pop("wall_s"))
    return res


def _preimport(spec, job, engine_module):
    """Import (and build) the engine before the start boundary, so the timed region is preceded only by the
    engine's own warm-up."""
    if job["impl"] == "parallel":
        mod = C.parallel_engine(spec["system"], engine_module)
        if callable(getattr(mod, "load_kernel", None)):      # gateway: build / load the C kernel (runs gcc --version)
            mod.load_kernel()
    elif job["impl"] == "serial":
        import parallel_benchmark  # noqa: F401
        importlib.import_module(spec["validated_module"])


def run_job_child(job, out_root=C.OUT_ROOT, engine_module=None, guard=None, hold=False,
                  hold_timeout=A.HOLD_TIMEOUT_S):
    """Execute one job inside its pinned process and write its npz + JSON record (status 'ran'; the driver's
    finalisation sets 'complete').  hold: keep the monitor's boundary intervals outside the timed region
    (expB_alloc.hold_for_monitor: before the engine call and after the records are written)."""
    host = C.host_info()            # affinity at start: libgomp later binds the main thread to its first place
    spec = C.system_spec(job["system"], overrides=job.get("spec_overrides"))
    paths = job_paths(job, out_root)
    sig = job_signature(job, spec, engine_module)
    saves = job_grid(job, spec)
    holds = {}
    if hold:
        _preimport(spec, job, engine_module)
        holds["start"] = A.hold_for_monitor(paths["tel_jsonl"], after=time.time(), timeout=hold_timeout,
                                            boundary="start")
    t_start = time.time()
    if job["impl"] == "validated":
        res = _validated_run(spec, job, saves, paths["base"] + ".ckpt.npz")
        res["t0_epoch"] = t_start
    elif job["impl"] == "serial":
        res = C.run_serial(spec, job["method"], job["N"], job["seed"], job["n_steps"], saves)
    else:
        res = C.run_parallel(spec, job["method"], job["N"], job["seed"], job["n_steps"], saves, job["threads"],
                             job["noise"], module=engine_module)
    science = {k: C.sha256_array(res[k]) for k in C.SCORE_KEYS[spec["kind"]] + ("save_step",)}
    rec = dict(status="ran", version=RUN_VERSION, job=job, signature=sig, science_sha256=science, pid=os.getpid(),
               wall_total_s=float(res["wall_total_s"]), warmup_s=float(res.get("warmup_s", 0.0)),
               timers={k: float(res.get(k, math.nan)) for k in C.TIMER_KEYS}, threads=int(res.get("threads", 1)),
               noise_mode=str(res.get("noise_mode")), engine=str(res.get("engine")),
               meta=C.parse_meta(res.get("meta")), t0_epoch=float(res.get("t0_epoch", t_start)),
               t0_epoch_source=res.get("t0_epoch_source"), peak_rss_mb=float(res.get("peak_rss_mb", math.nan)),
               host=host, cpus_after_run=C.host_info()["cpus"], child_wall_s=time.time() - t_start,
               n_saves=int(len(saves)),
               finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    arrays = True
    if job["kind"] == "prod" and job["replicate"] > 0:
        first = job_paths(dict(job, replicate=0), out_root)
        f = C.read_json(first["json"])
        rec["science_identical_to_first"] = bool(f.get("science_sha256") == science)
        arrays = not rec["science_identical_to_first"]   # a non-deterministic repeat keeps its arrays (and is flagged)
    if job["kind"] == "drift":
        arrays = False
        rec["throughput"] = job["N"] * job["n_steps"] / rec["wall_total_s"]
    rec["disk_pause_s"] = float((guard or C.DiskGuard()).wait(marker=paths["pause"]))
    rec["process_age_s"] = A.process_age_s()
    if arrays:
        keep = {k: v for k, v in res.items() if k not in ("t0_epoch", "t0_epoch_source", "call_epoch",
                                                           "return_epoch", "implementation")}
        rec = C.save_run(paths["npz"], keep, rec)
    else:
        if job["kind"] == "prod":
            rec = C.save_run(paths["npz"], {k: res[k] for k in TIMING_KEYS}, rec)
        else:
            C.write_json(paths["json"], rec)
    if os.path.exists(paths["base"] + ".ckpt.npz"):
        os.remove(paths["base"] + ".ckpt.npz")
    if hold:
        holds["end"] = A.hold_for_monitor(paths["tel_jsonl"], after=time.time(), timeout=hold_timeout)
    if holds:
        rec["holds"] = holds
        C.write_json(paths["json"], rec)
    return rec


# ================================================================================================ finalize
ENDPOINT_WINDOWS = ("W_strict", "W_mid", "W_loose", "whole_run")


def load_science(job, out_root):
    """The run's arrays: replicate 0's science + this replicate's timing arrays."""
    paths = job_paths(job, out_root)
    res = C.load_run(paths["npz"])
    if job["kind"] == "prod" and job["replicate"] > 0 and "M_all" not in res:
        first = C.load_run(job_paths(dict(job, replicate=0), out_root)["npz"])
        first.update({k: res[k] for k in TIMING_KEYS})
        res = first
    return res


def score_and_contention(job, rec, res, spec, scorer, tel):
    """Endpoints (expB_common.run_endpoints) and the per-endpoint contention verdicts."""
    ep, e = C.run_endpoints(res, spec, scorer, job["N"], job["n_steps"], job["threads"])
    whole = float(rec["wall_total_s"])
    cont = {}
    for name in ENDPOINT_WINDOWS:
        if name == "whole_run":
            w = whole
        else:
            W = ep[name]
            w = whole if math.isinf(W) else W
        cont[name] = A.contention(tel, float(rec["t0_epoch"]), w, job["threads"], pid=rec.get("pid"))
        # Amendment 2: the verdict above uses the run's own cores + SMT siblings; the pre-amendment CCD-wide verdict is
        # kept for the sensitivity analysis (it never decides repeats or the primary set).
        cont[name]["ccd_scope"] = {k: v for k, v in A.contention(tel, float(rec["t0_epoch"]), w, job["threads"],
                                                                 pid=rec.get("pid"), scope="ccd").items()
                                   if k in ("status", "max_foreign_busy", "max_foreign_busy_lower", "worst_cpu")}
    status = {k: v["status"] for k, v in cont.items()}
    return ep, e, cont, status


def _log_ledger_once(paths, stage, item, threads, used_wall_s, note, ledger):
    """One ledger row per run attempt: a marker file makes a repeated finalisation (driver restart) not count the
    same attempt twice."""
    if os.path.exists(paths["ledger_mark"]):
        return False
    C.ledger_append(C.ledger_row(stage, item, threads, used_wall_s, note=note), ledger)
    with open(paths["ledger_mark"], "w") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    return True


def clear_attempt_markers(paths):
    """Before a (re)launch: remove the previous attempt's ledger / disk-pause markers."""
    for k in ("ledger_mark", "pause"):
        if os.path.exists(paths[k]):
            os.remove(paths[k])


def finalize_job(job, paths, alloc, rc, proc_wall, spec, scorer, out_root, ledger, stage, log=print, note_extra="",
                 expected_sig=None):
    """Driver-side finalisation of one finished run process: telemetry sidecar, ledger row (used wall = process wall
    - disk pause, logged once per attempt), endpoints + per-endpoint contention (prod), contention (drift) or Ibar_F
    (equivalence), then status 'complete'.  Returns the record (dict(status='failed', ...) if the child left no
    record).  A record whose signature differs from ``expected_sig`` gets status 'signature_mismatch'."""
    tel = A.finalize_telemetry(paths["tel_jsonl"], paths["tel_npz"]) if job["kind"] in ("prod", "drift") else None
    rec = None
    if os.path.exists(paths["json"]):          # the child's record (it may still exit non-zero after writing it, e.g.
        try:                                   # terminated during its end hold): its science is complete
            rec = C.read_json(paths["json"])
        except (OSError, ValueError):
            rec = None
        if rec is not None and rec.get("status") not in ("ran", "complete"):
            rec = None
    pause = float(rec.get("disk_pause_s", 0.0)) if rec is not None else C.read_pause_marker(paths["pause"])
    used_wall = max(0.0, float(proc_wall) - pause)
    item = f"{job['arm']} s{job['seed']} rep{job['replicate']}"
    note = f"rc {rc}" + (f"; ccds {alloc['ccds']}; flagged {alloc['flagged']}" if alloc else "") + (
        f"; disk pause {pause:.0f} s excluded" if pause > 0 else "") + (f"; {note_extra}" if note_extra else "")
    if rec is None:
        _log_ledger_once(paths, stage, item, job["threads"], used_wall, note, ledger)
        log(f"  FAILED {job['kind']} {job['arm']} s{job['seed']} rep{job['replicate']} (rc {rc}); see {paths['log']}")
        return dict(status="failed", rc=rc, used_wall_s=used_wall)
    _log_ledger_once(paths, stage, item, job["threads"], used_wall, note, ledger)
    rec.update(allocation=({k: v for k, v in alloc.items() if k != "token"} if alloc else rec.get("allocation")),
               proc_wall_s=float(proc_wall), used_wall_s=used_wall, disk_pause_s=pause,
               telemetry=(os.path.basename(paths["tel_npz"]) if tel else None),
               used_core_h_run=job["threads"] * used_wall / 3600.0,
               alloc_core_h_run=(C.K_ALLOC * used_wall / 3600.0 if job["kind"] in ("prod", "drift") else None))
    if note_extra:
        rec["finalize_note"] = note_extra
    if expected_sig is not None and rec.get("signature") != json.loads(json.dumps(expected_sig)):
        rec["status"] = "signature_mismatch"
        C.write_json(paths["json"], rec)
        log(f"  SIGNATURE MISMATCH {item}: the child ran a different job definition (engine sources changed?)")
        return rec
    if job["kind"] == "prod":
        res = load_science(job, out_root)
        ep, _e, cont, status = score_and_contention(job, rec, res, spec, scorer, tel)
        rec.update(endpoints=ep, contention=cont, contention_status=status,
                   contended_any=any(v["contended"] for v in cont.values()),
                   uncontended_all=all(not v["contended"] for v in cont.values()))
    elif job["kind"] == "drift":
        cont = A.contention(tel, float(rec["t0_epoch"]), float(rec["wall_total_s"]), job["threads"],
                            pid=rec.get("pid"))
        rec.update(contention=dict(whole_run=cont), contended_any=cont["contended"])
    else:
        res = C.load_run(paths["npz"])
        e = scorer.e_F(res)
        rec.update(Ibar_F=C.ibar_uniform(e, res["save_step"], job["n_steps"]), final_e_F=float(e[-1]))
    rec["status"] = "complete"
    rec["finalized_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    C.write_json(paths["json"], rec)
    for k in ("pause",):
        if os.path.exists(paths[k]):
            os.remove(paths[k])
    return rec


def recover_record(job, out_root, spec, scorer, ledger, stage, log=print):
    """Finalise a record left 'ran' by an earlier driver (an orphan that finished, or an interrupted finalisation).
    The process wall is the child's recorded process age (the driver's own measurement is lost)."""
    paths = job_paths(job, out_root)
    rec0 = C.read_json(paths["json"])
    wall = rec0.get("process_age_s")
    if wall is None or not math.isfinite(float(wall)):
        wall = float(rec0.get("child_wall_s", 0.0))
    return finalize_job(job, paths, rec0.get("allocation"), 0, float(wall), spec, scorer, out_root, ledger, stage,
                        log, note_extra="recovered after the driver exited (wall = the child's process age)")


def replicate_records(job0, out_root):
    """All complete records of (arm, seed), replicate 0 first."""
    recs = []
    k = 0
    while True:
        p = job_paths(dict(job0, replicate=k), out_root)
        if not os.path.exists(p["json"]):
            if k == 0:
                break
            k += 1
            if k > MAX_REPEATS:
                break
            continue
        try:
            r = C.read_json(p["json"])
        except (OSError, ValueError):
            r = {"status": "failed"}
        if r.get("status") == "complete" and finalized(r):
            recs.append(r)
        k += 1
        if k > MAX_REPEATS:
            break
    return recs


def needs_repeat(recs):
    """True iff no complete replicate is uncontended for every endpoint and fewer than 1 + MAX_REPEATS exist."""
    if not recs:
        return False
    if any(r.get("uncontended_all") for r in recs):
        return False
    return len(recs) < 1 + MAX_REPEATS


def next_replicate(job0, out_root):
    k = 0
    while os.path.exists(job_paths(dict(job0, replicate=k), out_root)["json"]):
        k += 1
    return k


# ================================================================================================ launchers
class ProcessLauncher:
    """Each job in a fresh process pinned to its allocation's cores.  hold: the child keeps the monitor's boundary
    intervals outside its timed region (set when a telemetry monitor runs)."""

    def __init__(self, out_root, engine_module=None, python=sys.executable, hold=False):
        self.out_root, self.engine_module, self.python, self.hold = out_root, engine_module, python, bool(hold)

    def launch(self, job, paths, cores):
        os.makedirs(os.path.dirname(paths["json"]), exist_ok=True)
        cmd = [self.python, os.path.abspath(__file__), "run-one", "--job-json", json.dumps(job), "--out-root",
               self.out_root]
        if self.engine_module and job["kind"] != "drift":   # the drift reference always uses the gateway engine
            cmd += ["--engine-module", self.engine_module]
        if self.hold and job["kind"] in ("prod", "drift"):
            cmd += ["--hold"]
        return A.launch_pinned(cmd, cores, log_path=paths["log"])


class _Done:
    def __init__(self, rc):
        self.rc, self.pid = rc, os.getpid()
        self.returncode = rc

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        return self.rc


class InProcessLauncher:
    """Tests: run the child function synchronously in this process."""

    def __init__(self, out_root, engine_module=None):
        self.out_root, self.engine_module = out_root, engine_module
        self.hold = False

    def launch(self, job, paths, cores):
        try:
            run_job_child(job, self.out_root, None if job["kind"] == "drift" else self.engine_module)
            return _Done(0)
        except Exception as e:  # noqa: BLE001
            with open(paths["log"], "a") as fh:
                fh.write(f"{type(e).__name__}: {e}\n")
            return _Done(1)


def is_real_process(h):
    """True only for a real child process of this driver (never signal anything else)."""
    import subprocess
    return isinstance(h, subprocess.Popen)


# ================================================================================================ projection
def arm_cost(arm_rec, spec):
    """Projected (wall_s, used core-h) of one production run of an arm (bench median wall per walker-step)."""
    w = float(spec["B"]) * float(arm_rec["wall_per_walker_step"]) + RUN_OVERHEAD_S
    return w, int(arm_rec["threads"]) * w / 3600.0


def project_stage(spec, decision, pending_jobs, max_cores=64, drift_wall_s=None, drift_every_s=DRIFT_EVERY_S):
    by_arm = {a["arm"]: a for a in canonical_arms(decision)}
    core_h, serial_wall = 0.0, 0.0
    for j in pending_jobs:
        w, ch = arm_cost(by_arm[j["arm"]], spec)
        core_h += ch
        serial_wall += w
    wall_est = core_h * 3600.0 / max(1, max_cores)          # perfectly packed lower bound
    n_drift = int(math.ceil(max(wall_est, 1.0) / drift_every_s)) + 1 if drift_wall_s else 0
    drift_ch = n_drift * 16 * ((drift_wall_s or 0.0) + RUN_OVERHEAD_S) / 3600.0
    return dict(n_jobs=len(pending_jobs), run_core_h=core_h, drift_core_h=drift_ch, total_core_h=core_h + drift_ch,
                wall_h_packed_lower_bound=wall_est / 3600.0, n_drift=n_drift)


# ================================================================================================ scheduler
class Scheduler:
    def __init__(self, system, decision, out_root=C.OUT_ROOT, spec=None, allocator=None, launcher=None,
                 registry_root=A.REGISTRY_DIR, ledger=C.LEDGER, cap_core_h=math.inf, max_cores=64, max_ccds=8,
                 engine_module=None, drift=True, drift_every_s=DRIFT_EVERY_S, guard=None, log=print, poll_s=30.0,
                 max_wait_s=1800.0, clock=time.monotonic, sleep=time.sleep, monitor=True, monitor_allowed=None,
                 seeds=None, spec_overrides=None, finalize_sleep_s=0.0, pin_self=True, run_timeout_s=12 * 3600.0,
                 check_engine=True):
        self.system = system
        self.spec_overrides = spec_overrides
        self.spec = spec or C.system_spec(system, overrides=spec_overrides)
        self.decision = decision
        self.out_root = out_root
        self.registry = A.Registry(registry_root)
        self.monitor, self.monitor_allowed = monitor, monitor_allowed
        self.reserved_ccds = A.ccds_of_cpus(monitor_allowed) if monitor_allowed else []
        self.allocator = allocator or A.Allocator(self.registry, log=log, reserved=self.reserved_ccds)
        self.launcher = launcher or ProcessLauncher(out_root, engine_module, hold=monitor)
        self.ledger, self.cap = ledger, float(cap_core_h)
        self.max_cores, self.max_ccds = int(max_cores), int(max_ccds)
        self.engine_module = engine_module
        self.drift, self.drift_every_s = bool(drift), float(drift_every_s)
        self.guard = guard or C.DiskGuard(log=log)
        self.log, self.poll_s, self.max_wait_s = log, float(poll_s), float(max_wait_s)
        self.clock, self.sleep = clock, sleep
        self.seeds = seeds or self.spec["seeds"]
        self.finalize_sleep_s = float(finalize_sleep_s)
        self.pin_self = bool(pin_self)
        self.run_timeout_s = float(run_timeout_s)
        self.check_engine = bool(check_engine)
        self._core = None
        self.arms = canonical_arms(decision)
        self.by_arm = {a["arm"]: a for a in self.arms}
        self.stage = f"expB-prod-{system}"
        self.events = os.path.join(out_root, "prod", system, "scheduler_events.jsonl")
        self._scorer = None
        self._gw_spec = None
        self._drift_k = None
        self.used_core_h = 0.0
        self.orphans = {}
        self.stop = False

    # ---- helpers
    @property
    def scorer(self):
        if self._scorer is None:
            self._scorer = C.Scorer(self.spec)
        return self._scorer

    def event(self, **rec):
        os.makedirs(os.path.dirname(self.events), exist_ok=True)
        rec = dict(rec, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(self.events, "a") as fh:
            fh.write(json.dumps(C.json_safe(rec), default=str) + "\n")

    def _job(self, arm, seed, rep=0):
        j = prod_job(self.spec, self.by_arm[arm], seed, rep)
        if self.spec_overrides:
            j["spec_overrides"] = self.spec_overrides
        return j

    def _sig(self, job):
        return job_signature(job, self.spec if job["kind"] != "drift" else (self._gw_spec or C.system_spec(
            "gateway")), None if job["kind"] == "drift" else self.engine_module)

    def engine_check(self):
        """Problems (list) of the current engine sources vs decision.json's (the benchmarked engine)."""
        want = self.decision.get("engine_sources")
        if not want:
            return ["decision.json records no engine_sources (re-run expB_bench.py decide)"]
        probs = []
        for impl in sorted({a["impl"] for a in self.arms}):
            cur = C.engine_sources_sha(self.spec, impl, self.engine_module)
            if impl not in want:
                probs.append(f"decision.json has no engine_sources for impl {impl!r}")
            elif cur != want[impl]:
                diff = sorted(k for k in set(cur) | set(want[impl]) if cur.get(k) != want[impl].get(k))
                probs.append(f"{impl} engine sources differ from the benchmarked engine: {diff}")
        return probs

    def plan_full(self):
        """dict(pending (dispatch order), n_complete, repeats, mismatch, ran: records awaiting finalisation)."""
        order = production_order([a["arm"] for a in self.arms], self.seeds)
        pending, repeats, mismatch, ran, n_complete = [], [], [], [], 0
        for arm, seed in order:
            j0 = self._job(arm, seed, 0)
            sig = job_signature(j0, self.spec, self.engine_module)        # the replicate index is not in it
            st = record_status(job_paths(j0, self.out_root), sig)
            if st == "mismatch":
                mismatch.append(job_paths(j0, self.out_root)["json"])
            elif st in ("complete", "ran"):
                if st == "complete":
                    n_complete += 1
                else:
                    ran.append(j0)
                rr = []
                for k in range(1, 1 + MAX_REPEATS):
                    jk = self._job(arm, seed, k)
                    stk = record_status(job_paths(jk, self.out_root), sig)
                    if stk == "ran":
                        rr.append(jk)
                    elif stk == "mismatch":
                        mismatch.append(job_paths(jk, self.out_root)["json"])
                ran.extend(rr)
                if st == "complete" and not rr and needs_repeat(replicate_records(j0, self.out_root)):
                    repeats.append(self._job(arm, seed, next_replicate(j0, self.out_root)))
            else:
                pending.append(j0)
        return dict(pending=pending, n_complete=n_complete, repeats=repeats, mismatch=mismatch, ran=ran)

    def plan(self):
        """(pending jobs in dispatch order, complete count, repeat jobs needed, mismatches)."""
        p = self.plan_full()
        return p["pending"], p["n_complete"], p["repeats"], p["mismatch"]

    def projection(self, pending):
        dw = None
        if self.drift:
            dw = self._drift_wall()
        return project_stage(self.spec, self.decision, pending, self.max_cores, dw, self.drift_every_s)

    def _drift_wall(self):
        if getattr(self, "_drift_wall_cache", None) is not None:
            return self._drift_wall_cache
        self._drift_wall_cache = self._drift_wall_uncached()
        return self._drift_wall_cache

    def _drift_wall_uncached(self):
        try:
            bench = C.read_json(os.path.join(self.out_root, "bench", "gateway", "summary.json"))["summary"]
            s = bench.get("fr512_parallel_t16")
            if s and math.isfinite(s["wall_per_walker_step"]):
                gw = self._gw_spec or C.system_spec("gateway")
                return 0.02 * gw["B"] * s["wall_per_walker_step"]
        except (OSError, KeyError, ValueError):
            pass
        return 60.0

    def _drift_job(self):
        """The next drift reference; its replicate index is reserved here (never reused by a queued or running one)."""
        import expB_bench as BB
        if self._drift_k is None:
            ddir = os.path.join(self.out_root, "drift")
            used = [-1]
            if os.path.isdir(ddir):
                for f in os.listdir(ddir):
                    if f.startswith("d") and f[1:4].isdigit():
                        used.append(int(f[1:4]))
            for key in self.registry.live_job_keys():
                if key.startswith("drift|"):
                    used.append(int(key.rsplit("|r", 1)[1]))
            self._drift_k = max(used) + 1
        k = self._drift_k
        self._drift_k += 1
        return BB.drift_job(self._gw_spec, k)

    # ---- orphans (runs of a driver that died) and recovery
    def _foreign_drivers(self):
        """Live locks of this system's production runs whose driver process is another live scheduler."""
        out = []
        for info in self.registry.active_runs():
            key = info.get("job_key") or ""
            if not key.startswith(f"prod|{self.system}|"):
                continue
            d = info.get("driver") or {}
            if d.get("pid") and int(d["pid"]) != os.getpid() and A.pid_alive(d["pid"], d.get("start")):
                out.append(dict(job_key=key, driver=d))
        return out

    def _recover(self, job):
        try:
            rec = recover_record(job, self.out_root, self.spec, self.scorer, self.ledger, self.stage, self.log)
        except Exception as e:  # noqa: BLE001
            self.log(f"  could not finalise the recovered record of {A.job_key(job)}: {type(e).__name__}: {e}")
            self.event(event="recover_error", job_key=A.job_key(job), error=f"{type(e).__name__}: {e}")
            return None
        self.used_core_h += job["threads"] * float(rec.get("used_wall_s", 0.0)) / 3600.0
        self.event(event="recovered", job_key=A.job_key(job), status=rec.get("status"),
                   contended=rec.get("contended_any"))
        return rec

    def _after_complete(self, job, rec, queue):
        """Queue the repeat a complete production record needs (contended for some endpoint)."""
        if job["kind"] != "prod" or rec is None or rec.get("status") != "complete":
            return
        j0 = dict(job, replicate=0)
        recs = replicate_records(j0, self.out_root)
        if needs_repeat(recs):
            queue.appendleft(dict(job, replicate=next_replicate(j0, self.out_root)))
            self.log(f"  repeat queued: {job['arm']} s{job['seed']} (contended: "
                     f"{ {k: v for k, v in rec['contention_status'].items() if v != 'uncontended'} })")
        elif not any(r.get("uncontended_all") for r in recs):
            self.log(f"  FLAG: {job['arm']} s{job['seed']} has no uncontended replicate after "
                     f"{MAX_REPEATS} repeats (excluded from the primary analysis)")

    def _terminate_overdue_orphan(self, key, info):
        """An orphaned run (launched by this campaign, its driver dead) still alive after the run timeout counted from
        its lock's creation is hung: terminate it (identity checked by pid + start time; only run pids of the lock)."""
        import calendar
        import signal
        if info.get("_terminated") or key in getattr(self, "_orphans_terminated", set()):
            return
        try:
            created = calendar.timegm(time.strptime(info.get("created_utc", ""), "%Y-%m-%dT%H:%M:%SZ"))
        except (TypeError, ValueError):
            return
        if time.time() - created <= self.run_timeout_s:
            return
        for x in info.get("pids", []):
            if x.get("role") != "run" or not A.pid_alive(x.get("pid"), x.get("start")):
                continue
            try:
                os.killpg(int(x["pid"]), signal.SIGTERM)    # its own session (launch_pinned: start_new_session)
            except OSError:
                continue
            self.log(f"  TIMEOUT orphan {key}: terminated its run process {x['pid']} (alive "
                     f"{(time.time() - created) / 3600:.1f} h)")
            self.event(event="orphan_timeout", job_key=key, pid=int(x["pid"]))
        self._orphans_terminated = getattr(self, "_orphans_terminated", set()) | {key}

    def _watch_orphans(self, queue):
        if not self.orphans:
            return
        live = self.registry.live_job_keys()
        for key, job in list(self.orphans.items()):
            if key in live:
                self._terminate_overdue_orphan(key, live[key])
                continue
            del self.orphans[key]
            st = record_status(job_paths(job, self.out_root), job_signature(job, self.spec, self.engine_module))
            self.event(event="orphan_ended", job_key=key, record=st)
            if st == "ran":
                self._after_complete(job, self._recover(job), queue)
            elif st == "complete":
                self._after_complete(job, C.read_json(job_paths(job, self.out_root)["json"]), queue)
            elif st == "mismatch":
                self.log(f"  STOP: the orphaned run {key} wrote a record of a different job definition")
                self.event(event="stop", reason="orphan signature mismatch", job_key=key)
                self.stop = True
            else:
                queue.appendleft(job)            # the orphan failed: run the job again
                self.log(f"  orphaned run {key} left no record: queued again")

    # ---- main loop
    def run(self, dry_run=False):
        if not self.decision.get("production_licensed"):
            raise C.ExpBError(f"{self.system}: decision.json does not license production (gate verdict "
                              f"{self.decision.get('gate', {}).get('verdict')}; reasons "
                              f"{self.decision.get('license_problems')})")
        if self.check_engine:
            probs = self.engine_check()
            if probs:
                self.event(event="refused", reason="engine sources", problems=probs)
                raise C.ExpBError(f"{self.system}: refusing: " + "; ".join(probs))
        others = self._foreign_drivers()
        if others:
            raise C.ExpBError(f"another live scheduler is running {self.system} production: {others[:2]}")
        live = self.registry.live_job_keys()
        P = self.plan_full()
        if P["mismatch"]:
            raise C.ExpBError(f"complete records of a DIFFERENT job (signature mismatch), refusing: {P['mismatch'][:3]}")
        if not dry_run and any(A.job_key(j) not in live for j in P["ran"]):
            for j in P["ran"]:
                if A.job_key(j) not in live:
                    self._recover(j)
            P = self.plan_full()
        held_keys = {A.job_key(j): j for j in P["pending"] + P["repeats"] + P["ran"] if A.job_key(j) in live}
        pending = [j for j in P["pending"] if A.job_key(j) not in held_keys]
        repeats = [j for j in P["repeats"] if A.job_key(j) not in held_keys]
        proj = self.projection(repeats + pending)
        self.log(f"{self.system}: {P['n_complete']} complete, {len(repeats)} repeats needed, {len(pending)} pending, "
                 f"{len(held_keys)} still running from an earlier driver; projected {proj['total_core_h']:.2f} used "
                 f"core-h (cap {self.cap})")
        if proj["total_core_h"] > self.cap:
            self.event(event="refused", reason="projection exceeds cap", projection=proj, cap=self.cap)
            raise C.ExpBError(f"projection gate: {proj['total_core_h']:.2f} core-h > cap {self.cap}; not launched")
        if dry_run:
            return dict(projection=proj, pending=len(pending), repeats=len(repeats), complete=P["n_complete"],
                        orphans=sorted(held_keys), to_recover=len(P["ran"]))
        self.orphans = dict(held_keys)
        queue = collections.deque(repeats + pending)
        active = {}
        mon = A.start_monitor(self.registry.root, allowed=self.monitor_allowed) if self.monitor else None
        last_drift = -math.inf
        head_key, head_wait = None, None
        pause_t0 = None
        self.stop = False
        self.event(event="start", projection=proj, n_queue=len(queue), orphans=sorted(self.orphans))
        try:
            while queue or active or self.orphans:
                if self.pin_self:                         # this process scores runs: never on an allocation
                    self._core = A.pin_outside(self.registry, self.monitor_allowed, current=self._core, log=self.log)
                if mon is not None and mon.poll() is not None:
                    self.log("  monitor process exited; restarting it")
                    self.event(event="monitor_restart", rc=mon.returncode)
                    mon = A.start_monitor(self.registry.root, allowed=self.monitor_allowed)
                self._reap(active, queue)
                self._watch_orphans(queue)
                if self.stop:
                    if not active and not self.orphans:
                        break
                    self.sleep(1.0)
                    continue
                if not self.guard.ok():
                    if pause_t0 is None:
                        pause_t0 = self.clock()
                        self.event(event="disk_pause", free_bytes=self.guard.free())
                    elif self.clock() - pause_t0 >= self.guard.max_pause_s:
                        self.log(f"  STOP: disk pause did not clear within {self.guard.max_pause_s / 3600:.1f} h")
                        self.event(event="stop", reason="disk pause timeout (stopping rule iv)",
                                   paused_s=self.clock() - pause_t0)
                        self.stop = True
                        continue
                    self._wait(active, queue, self.guard.poll_s)
                    continue
                if pause_t0 is not None:
                    self.event(event="disk_resume", paused_s=self.clock() - pause_t0)
                    pause_t0 = None
                now = self.clock()
                if (self.drift and now - last_drift >= self.drift_every_s
                        and not any(v[0]["kind"] == "drift" for v in active.values())
                        and not any(j["kind"] == "drift" for j in queue)):
                    if C.engine_available("gateway"):
                        queue.appendleft(self._drift_job())
                    else:
                        self.log("  drift reference skipped: the gateway parallel engine is not available")
                    last_drift = now
                if not queue:
                    self._wait(active, queue, 1.0)
                    continue
                job = queue[0]
                key = A.job_key(job)
                if key != head_key:                       # a new head of line: its own wait starts now
                    head_key, head_wait = key, None
                held = self.registry.held()
                # campaign cores in use: every registered run of this campaign (incl. other campaign processes)
                used = sum(int(i.get("threads", 0)) for i in self.registry.active_runs())
                used = max(used, sum(v[0]["threads"] for v in active.values()))
                if used + job["threads"] > self.max_cores or (
                        len(held) + A.n_ccds_for(job["threads"]) > self.max_ccds):
                    self._wait(active, queue, 1.0)
                    continue
                remaining = self.projection([j for j in queue if j["kind"] == "prod"] +
                                            [v[0] for v in active.values() if v[0]["kind"] == "prod"])["run_core_h"]
                if self.used_core_h + remaining > self.cap:
                    self.log(f"  STOP: used {self.used_core_h:.2f} + projected {remaining:.2f} > cap {self.cap}")
                    self.event(event="cap_stop", used=self.used_core_h, remaining=remaining)
                    self.stop = True
                    continue
                if key in self.registry.live_job_keys():  # a run of this job is still alive (orphan): never twice
                    queue.popleft()
                    self.orphans[key] = job
                    continue
                paths = job_paths(job, self.out_root)
                run_id = f"{job['kind']}-{job['system']}-{job['arm']}-s{job['seed']}-r{job['replicate']}-{os.getpid()}"
                info = dict(kind=job["kind"], telemetry=paths["tel_jsonl"], job_key=key,
                            driver=dict(pid=os.getpid(), start=A.proc_start_time(os.getpid())))
                waited = 0.0 if head_wait is None else now - head_wait
                alloc = self.allocator.try_allocate(job["threads"], run_id, info, waited)
                if alloc is None and head_wait is not None and waited >= self.max_wait_s:
                    alloc = self.allocator.fallback(job["threads"], run_id, info, waited)
                if alloc is None:
                    if head_wait is None:
                        head_wait = now
                    self._wait(active, queue, self.poll_s)
                    continue
                head_key, head_wait = None, None
                if self.pin_self:                         # move off the new allocation before the run starts
                    self._core = A.pin_outside(self.registry, self.monitor_allowed, current=self._core, log=self.log)
                queue.popleft()
                os.makedirs(os.path.dirname(paths["json"]), exist_ok=True)
                if os.path.exists(paths["json"]):        # a failed earlier attempt: keep it for the record
                    os.replace(paths["json"], paths["base"] + f".failed{int(time.time())}.json")
                clear_attempt_markers(paths)
                sig = self._sig(job)
                t_launch = time.time()
                h = self.launcher.launch(job, paths, alloc["cores"])
                try:
                    self.registry.add_pid(alloc["ccds"], alloc["token"], h.pid)
                except RuntimeError as e:                 # the monitor cannot follow this run: its contention
                    self.log(f"  could not register the run pid of {run_id}: {e}")      # will read 'unknown'
                    self.event(event="add_pid_failed", run_id=run_id, error=str(e))
                active[run_id] = (job, paths, alloc, h, t_launch, sig)
                self.event(event="launch", run_id=run_id, job=job, ccds=alloc["ccds"], flagged=alloc["flagged"],
                           wait_s=alloc["wait_s"])
        finally:
            self._shutdown_active(active)
            A.stop_monitor(mon)
        self.event(event="end", used_core_h=self.used_core_h, stopped=self.stop)
        return dict(used_core_h=self.used_core_h, stopped=self.stop)

    def _wait(self, active, queue, seconds):
        t_end = self.clock() + float(seconds)
        while True:
            self._reap(active, queue)
            if self.clock() >= t_end:
                return
            self.sleep(min(1.0, max(0.0, t_end - self.clock())))

    def _finalize(self, job, paths, alloc, rc, proc_wall, sig, note=""):
        """finalize_job of one run; an exception is logged (the record stays 'ran', recovered on the next start)."""
        stage = "expB-drift" if job["kind"] == "drift" else self.stage
        spec = self.spec if job["kind"] != "drift" else (self._gw_spec or C.system_spec("gateway"))
        try:
            rec = finalize_job(job, paths, alloc, rc, proc_wall, spec, self.scorer if job["kind"] == "prod" else None,
                               self.out_root, self.ledger, stage, self.log, note_extra=note, expected_sig=sig)
        except Exception as e:  # noqa: BLE001
            self.log(f"  FINALIZE ERROR {A.job_key(job)}: {type(e).__name__}: {e} (record kept for recovery)")
            self.event(event="finalize_error", job_key=A.job_key(job), error=f"{type(e).__name__}: {e}")
            rec = dict(status="finalize_error", used_wall_s=max(0.0, proc_wall - C.read_pause_marker(paths["pause"])))
        if rec.get("status") == "signature_mismatch":
            self.event(event="stop", reason="signature mismatch", job_key=A.job_key(job))
            self.stop = True
        return rec

    def _reap(self, active, queue):
        for rid in list(active):
            job, paths, alloc, h, t_launch, sig = active[rid]
            rc = h.poll()
            if rc is None and is_real_process(h):
                run_elapsed = time.time() - t_launch - C.read_pause_marker(paths["pause"])
                if run_elapsed > self.run_timeout_s:
                    self.log(f"  TIMEOUT {rid}: terminating our own run process after {run_elapsed:.0f} s of run "
                             f"time (disk pauses excluded)")
                    rc = A.kill_process_group(h)
                    rc = -9 if rc is None else rc
            if rc is None:
                continue
            proc_wall = time.time() - t_launch
            if self.finalize_sleep_s and self.monitor:
                time.sleep(self.finalize_sleep_s)
            self.allocator.release(alloc)
            del active[rid]
            rec = self._finalize(job, paths, alloc, rc, proc_wall, sig)
            self.used_core_h += job["threads"] * float(rec.get("used_wall_s", proc_wall)) / 3600.0
            self.event(event="finish", run_id=rid, status=rec.get("status"), contended=rec.get("contended_any"),
                       proc_wall_s=proc_wall)
            try:
                self._after_complete(job, rec, queue)
            except Exception as e:  # noqa: BLE001
                self.log(f"  repeat check failed for {A.job_key(job)}: {type(e).__name__}: {e}")
                self.event(event="repeat_check_error", job_key=A.job_key(job), error=f"{type(e).__name__}: {e}")

    def _shutdown_active(self, active):
        """Driver exit with runs still active (exception / signal): terminate OUR run processes, finalise records
        they already wrote, log the rest to the ledger as killed, release their CCDs."""
        for rid in list(active):
            job, paths, alloc, h, t_launch, sig = active.pop(rid)
            rc = A.kill_process_group(h) if is_real_process(h) else h.poll()
            proc_wall = time.time() - t_launch
            st = record_status(paths, sig) if os.path.exists(paths["json"]) else None
            try:
                if st == "ran":
                    rec = self._finalize(job, paths, alloc, 0, proc_wall, sig,
                                         note="process terminated at driver exit after writing its record")
                else:
                    rec = self._finalize(job, paths, alloc, rc if rc not in (None, 0) else -15, proc_wall, sig,
                                         note="terminated at driver exit")
                self.used_core_h += job["threads"] * float(rec.get("used_wall_s", proc_wall)) / 3600.0
            finally:
                self.allocator.release(alloc)
            self.event(event="terminated_at_exit", run_id=rid, record=st, rc=rc)


# ================================================================================================ equivalence runs
def equiv_arms(spec):
    small = min((n for m, n in spec["core_arms"] if m == "abf"))
    return [("abf", small), ("abf", 512), ("fr", 512)]


def equiv_jobs(spec, which="all", extension_arms=None, threads=1):
    """V-B4 jobs: counter-mode runs on the counter seeds (+ extension counter seeds for extension_arms) and
    validated-engine runs (LTA: the 16 new validated seeds; + extension validated seeds for extension_arms; gateway:
    extension validated seeds only).  Equal-budget production save grid (identical to the reference runs)."""
    es = spec["equivalence_seeds"]
    ext = set(extension_arms or [])
    jobs = []
    for m, n in equiv_arms(spec):
        arm = C.arm_name(m, n)
        base = dict(system=spec["system"], arm=arm, method=m, N=int(n), n_steps=C.n_steps_for(spec["B"], n),
                    replicate=0, grid="eqb")
        if which in ("all", "counter"):
            seeds = list(es["counter"]) + (list(es["extension_counter"]) if arm in ext else [])
            jobs += [dict(base, kind="equiv_counter", seed=s, impl="parallel", threads=int(threads), noise="counter")
                     for s in seeds]
        if which in ("all", "validated"):
            seeds = list(es.get("new_validated", [])) + (list(es["extension_validated"]) if arm in ext else [])
            jobs += [dict(base, kind="equiv_validated", seed=s, impl="validated", threads=1, noise="replay")
                     for s in seeds]
    return sorted(jobs, key=lambda j: (j["seed"], j["kind"], j["arm"]))


def equiv_projection(spec, jobs, out_root=C.OUT_ROOT):
    """Rough used core-hours of V-B4 jobs: bench wall per walker-step at (method, N) (parallel t = 1 for counter
    jobs, serial for validated ones) when the benchmark exists, else the equal-budget config's estimate."""
    p = os.path.join(out_root, "bench", spec["system"], "summary.json")
    summ = C.read_json(p)["summary"] if os.path.exists(p) else {}
    est = float(spec["P"].get("est_us_per_walker_step", 1.0)) * 1e-6
    tot, src = 0.0, set()
    for j in jobs:
        key = f"{j['method']}{j['N']}_" + ("parallel_t1" if j["kind"] == "equiv_counter" else "serial")
        s = summ.get(key)
        c = float(s["wall_per_walker_step"]) if s and s.get("n") else est
        src.add("benchmark" if s and s.get("n") else "config estimate")
        tot += j["threads"] * (spec["B"] * c + RUN_OVERHEAD_S) / 3600.0
    return dict(core_h=tot, n_jobs=len(jobs), source="+".join(sorted(src)) or "none")


def run_equiv(system, cpus, which="all", extension_arms=None, threads=1, out_root=C.OUT_ROOT,
              registry_root=A.REGISTRY_DIR, ledger=C.LEDGER, engine_module=None, launcher=None, spec=None, log=print,
              dry_run=False, sleep=time.sleep, lock=True, spec_overrides=None, guard=None, clock=time.monotonic):
    """Run the V-B4 jobs on the given CPU list (one job per `threads` cores).  The CCDs of those CPUs are locked in
    the campaign registry for the duration, with every child's pid added to the lock (so the lock stays live while
    any child runs, even if this runner dies).  Records left 'ran' by an earlier runner are finalised first.  On exit
    (exception / signal) this runner terminates its own children.  A disk pause longer than 24 h stops the runner."""
    spec = spec or C.system_spec(system, overrides=spec_overrides)
    jobs = equiv_jobs(spec, which, extension_arms, threads)
    if spec_overrides:                                    # tests only: the child rebuilds the same spec
        for j in jobs:
            j["spec_overrides"] = spec_overrides
    todo, ran = [], []
    for j in jobs:
        st = record_status(job_paths(j, out_root), job_signature(j, spec, engine_module))
        if st == "mismatch":
            raise C.ExpBError(f"{job_paths(j, out_root)['json']} holds a different job; refusing")
        if st == "ran":
            ran.append(j)
        elif st != "complete":
            todo.append(j)
    proj = equiv_projection(spec, todo, out_root)
    log(f"{system} V-B4: {len(jobs)} jobs, {len(todo)} to run, {len(ran)} to finalise, on {len(cpus)} cpus "
        f"({threads} thread(s) each); projected ~{proj['core_h']:.1f} used core-h ({proj['source']})")
    if dry_run:
        return dict(n_jobs=len(jobs), todo=len(todo), to_finalise=len(ran), projection=proj)
    reg = A.Registry(registry_root)
    ccds = sorted({A.ccd_of_cpu(c) for c in cpus})
    token = None
    stage = f"expB-equiv-{system}"
    if lock:
        token = reg.acquire(ccds, dict(run_id=f"equiv-{system}-{os.getpid()}", kind="nontiming", cpus=sorted(cpus),
                                       cores=sorted(cpus), threads=len(cpus)))
        if token is None:
            raise C.ExpBError(f"CCDs {ccds} are held by another campaign process (or by children of an earlier "
                              f"equivalence runner that are still running)")
        try:                                              # the runner itself (scoring) stays on its own cpus
            os.sched_setaffinity(0, set(int(c) for c in cpus))
        except (OSError, AttributeError):
            pass
    slots = [list(cpus[i:i + threads]) for i in range(0, len(cpus) - threads + 1, threads)]
    free = list(slots)
    launcher = launcher or ProcessLauncher(out_root, engine_module)
    guard = guard or C.DiskGuard(log=log)
    scorer = C.Scorer(spec)
    queue = collections.deque(todo)
    active = []
    pause_t0, stopped = None, None

    def _fin(j, paths, rc, wall, note=""):
        try:
            rec = finalize_job(j, paths, None, rc, wall, spec, scorer, out_root, ledger, stage, log, note_extra=note)
        except Exception as e:  # noqa: BLE001
            log(f"  FINALIZE ERROR {j['kind']} {j['arm']} s{j['seed']}: {type(e).__name__}: {e} (kept for recovery)")
            return dict(status="finalize_error")
        log(f"  {j['kind']} {j['arm']} s{j['seed']}: {rec.get('status')} Ibar_F {rec.get('Ibar_F')}")
        return rec
    try:
        for j in ran:
            r0 = C.read_json(job_paths(j, out_root)["json"])
            w = r0.get("process_age_s")
            _fin(j, job_paths(j, out_root), 0, float(w if w is not None and math.isfinite(float(w)) else
                                                     r0.get("child_wall_s", 0.0)),
                 note="recovered after the runner exited (wall = the child's process age)")
        while queue or active:
            for item in list(active):
                j, paths, slot, h, t0 = item
                rc = h.poll()
                if rc is None:
                    continue
                active.remove(item)
                free.append(slot)
                _fin(j, paths, rc, time.time() - t0)
            if queue and free and stopped is None:
                if not guard.ok():
                    if pause_t0 is None:
                        pause_t0 = clock()
                    elif clock() - pause_t0 >= guard.max_pause_s:
                        stopped = "disk pause timeout (stopping rule iv)"
                        log(f"  STOP: {stopped}; {len(queue)} jobs not run")
                    sleep(1.0)
                    continue
                pause_t0 = None
                j = queue.popleft()
                slot = free.pop(0)
                paths = job_paths(j, out_root)
                os.makedirs(os.path.dirname(paths["json"]), exist_ok=True)
                clear_attempt_markers(paths)
                h = launcher.launch(j, paths, slot)
                if token and is_real_process(h):
                    try:
                        reg.add_pid(ccds, token, h.pid, role="equiv")
                    except RuntimeError:
                        pass
                active.append((j, paths, slot, h, time.time()))
                continue
            if stopped is not None and not active:
                break
            sleep(1.0)
    finally:
        for item in list(active):
            j, paths, slot, h, t0 = item
            rc = A.kill_process_group(h) if is_real_process(h) else h.poll()
            st = record_status(paths, job_signature(j, spec, engine_module)) if os.path.exists(paths["json"]) else None
            _fin(j, paths, 0 if st == "ran" else (rc if rc not in (None, 0) else -15), time.time() - t0,
                 note="terminated at runner exit")
        if token:
            reg.release(ccds, token)
    return dict(n_jobs=len(jobs), ran=len(todo) - len(queue), stopped=stopped)


# ================================================================================================ CLI
def load_decision(system, out_root, path=None):
    p = path or os.path.join(out_root, "bench", system, "decision.json")
    if not os.path.exists(p):
        raise C.ExpBError(f"no decision file {p}: run expB_bench.py decide first")
    return C.read_json(p)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Experiment B production scheduler / V-B4 runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prod")
    p.add_argument("--system", required=True, choices=sorted(C.SYSTEMS))
    p.add_argument("--cap", type=float, required=True, help="remaining stage cap in used core-hours")
    p.add_argument("--validation-passed", action="store_true",
                   help="operator assertion that V-B1..V-B5 passed for this system (recorded)")
    p.add_argument("--exploratory", default=None, metavar="LABEL",
                   help="run WITHOUT the validation assertion as a labelled exploratory stage (e.g. the gateway after "
                        "its frozen V-B4 FAIL, PREREGISTRATION.md Amendment 3); the label is recorded in the event "
                        "log and in prod/<system>/STAGE_LABEL.json")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--decision", default=None)
    p.add_argument("--out-root", default=C.OUT_ROOT)
    p.add_argument("--registry", default=A.REGISTRY_DIR)
    p.add_argument("--max-cores", type=int, default=64)
    p.add_argument("--max-ccds", type=int, default=8)
    p.add_argument("--no-drift", action="store_true")
    p.add_argument("--engine-module", default=None)
    p.add_argument("--monitor-cpus", default=None,
                   help="home cores of this driver and its monitor; their CCDs are never allocated to a run")
    q = sub.add_parser("project")
    q.add_argument("--system", required=True, choices=sorted(C.SYSTEMS))
    q.add_argument("--out-root", default=C.OUT_ROOT)
    q.add_argument("--decision", default=None)
    e = sub.add_parser("equiv")
    e.add_argument("--system", required=True, choices=sorted(C.SYSTEMS))
    e.add_argument("--cpus", required=True)
    e.add_argument("--which", default="all", choices=("all", "counter", "validated"))
    e.add_argument("--extension-arms", nargs="*", default=None)
    e.add_argument("--threads", type=int, default=1)
    e.add_argument("--out-root", default=C.OUT_ROOT)
    e.add_argument("--registry", default=A.REGISTRY_DIR)
    e.add_argument("--engine-module", default=None)
    e.add_argument("--dry-run", action="store_true")
    o = sub.add_parser("run-one")
    o.add_argument("--job-json", required=True)
    o.add_argument("--out-root", default=C.OUT_ROOT)
    o.add_argument("--engine-module", default=None)
    o.add_argument("--hold", action="store_true", help="hold the monitor's boundary intervals outside the timed run")
    a = ap.parse_args(argv)
    if a.cmd == "run-one":
        run_job_child(json.loads(a.job_json), a.out_root, a.engine_module, hold=a.hold)
        return 0
    C.limit_driver_threads()
    C.install_exit_signals()
    if a.cmd == "equiv":
        r = run_equiv(a.system, A.parse_cpu_list(a.cpus), a.which, a.extension_arms, a.threads, a.out_root,
                      a.registry, engine_module=a.engine_module, dry_run=a.dry_run)
        print(json.dumps(r, indent=1))
        return 0
    dec = load_decision(a.system, a.out_root, a.decision)
    if a.cmd == "project":
        s = Scheduler(a.system, dec, a.out_root, monitor=False)
        pending, n_complete, repeats, mismatch = s.plan()
        print(json.dumps(dict(s.projection(repeats + pending), complete=n_complete, mismatch=mismatch), indent=1))
        return 0
    label = (a.exploratory or "").strip()
    if a.validation_passed and label:
        print("refusing: --validation-passed and --exploratory are exclusive")
        return 2
    if not a.validation_passed and not label and not a.dry_run:
        print("refusing: production needs --validation-passed (V-B1..V-B5 passed for this system) or an explicit "
              "--exploratory LABEL")
        return 2
    s = Scheduler(a.system, dec, a.out_root, registry_root=a.registry, cap_core_h=a.cap, max_cores=a.max_cores,
                  max_ccds=a.max_ccds, engine_module=a.engine_module, drift=not a.no_drift,
                  monitor_allowed=A.parse_cpu_list(a.monitor_cpus) if a.monitor_cpus else None)
    if not a.dry_run:
        stage = dict(system=a.system, validation_passed_assertion=bool(a.validation_passed),
                     exploratory=bool(label), label=(label or "confirmatory (V-B1..V-B5 passed)"),
                     utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        C.write_json(os.path.join(a.out_root, "prod", a.system, "STAGE_LABEL.json"), stage)
        s.event(event="stage_label", **stage)
    r = s.run(dry_run=a.dry_run)
    print(json.dumps(r, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
