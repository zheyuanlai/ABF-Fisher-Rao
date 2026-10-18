#!/usr/bin/env python
"""Hybrid short-T study: CCD-exclusive job scheduler (benchmark, production, confirmation) and the run child.

    python scripts/hybrid_short_T/hst_sched.py run-one --job-json '{...}' [--hold]     (the pinned child; internal)

The scheduler is used by benchmark.py and run_campaign.py.  It reuses the Experiment B machinery unchanged
(scripts/final_campaign/expB_alloc.py: CCD lock registry, 1 Hz telemetry monitor, contention rule, pinned launcher)
and differs from expB_run.Scheduler only where configs/hybrid_short_T/campaign.json says so:

* allocation choice: least-loaded CCD set judged on the cores the run will USE and their SMT siblings (not every cpu
  of the CCD: other users' unpinned processes roam over every CCD of this host); within a single CCD the t least
  busy cores are used; qualifying = all of them <= 10 % busy over a 2 s sample; otherwise re-sample every 5 s up to
  60 s, then the least-loaded set (FLAGGED);
* repeats: a run contended over its WHOLE timed window (expB_alloc.contention, scope 'own') is repeated, at most 3
  times (fresh process, new allocation); the child of a repeat checks that its science arrays are bitwise the first
  replicate's (sha256) and stores only its timing then;
* projection / cap: used core-hours of this study (ledger results/hybrid_short_T/compute_ledger.csv, stage prefix
  'hst-') + the projected cost of the queued and active jobs must stay <= the cap, else no new launch;
* dispatch: strict head-of-line order of the job list given (the caller orders it).

Record life cycle (as Experiment B): the child writes its record with status 'ran'; only the driver's finalisation
(telemetry sidecar, whole-run contention, ledger row) sets 'complete'.  A driver exiting by an exception or by
SIGTERM / SIGHUP terminates its own active run processes and logs them; on restart, records left 'ran' are
finalised (contention from whatever telemetry exists; 'unknown' -> a repeat) and a job whose key is held by a live
lock (an orphan still running) is never launched twice.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import hst_common as H  # noqa: E402
import expB_common as C  # noqa: E402
import expB_alloc as A  # noqa: E402

SCHED_VERSION = "hst_sched/1"
MAX_REPEATS = 3
QUALIFY_BUSY = 0.10
ALLOC_POLL_S = 5.0
ALLOC_MAX_WAIT_S = 10.0                   # Amendment A1 (2026-10-18): was 60 s
RESERVED_CCDS = (15,)                    # driver, monitor and operator commands live on CCD 15 (cores 120-127)


# ================================================================================================ allocation
class HSTAllocator(A.Allocator):
    """Allocator of configs/hybrid_short_T/campaign.json hardware.allocation_choice."""

    def choose(self, t, busy):
        held = set(self.registry.held()) | set(self.reserved)
        best, best_key, least, least_key = None, None, None, None
        for s in A.candidate_sets(t):
            if set(s) & held:
                continue
            if len(s) == 1 and t < H.CCD_CORES:
                cores = sorted(A.ccd_cores(s[0]), key=lambda c: (busy.get(c, 1.0) + busy.get(A.sibling(c), 1.0), c))
                cores = sorted(cores[:t])
            else:
                cores = A.used_cores(s, t)
            used = cores + [A.sibling(c) for c in cores]
            mx = max(float(busy.get(c, 1.0)) for c in used)
            tot = sum(float(busy.get(c, 1.0)) for c in A.set_cpus(s))
            key = (mx, tot, s)
            if mx <= self.threshold and (best_key is None or key < best_key):
                best, best_key = (s, cores), key
            if least_key is None or key < least_key:
                least, least_key = (s, cores), key
        return best, least, (best_key or least_key)

    def _lock_cores(self, s, cores, t, run_id, info, busy, flagged, wait_s):
        rec = dict(info or {}, run_id=run_id, threads=int(t), cores=list(cores), cpus=A.set_cpus(s))
        token = self.registry.acquire(s, rec)
        if token is None:
            return None
        cpus = A.set_cpus(s)
        b = {int(c): float(busy.get(c, math.nan)) for c in cpus}
        used = list(cores) + [A.sibling(c) for c in cores]
        return dict(run_id=run_id, token=token, ccds=list(s), cores=list(cores), cpus=cpus, threads=int(t),
                    flagged=bool(flagged), wait_s=float(wait_s), busy_at_launch=b,
                    max_busy_used_at_launch=max(float(busy.get(c, math.nan)) for c in used),
                    max_busy_at_launch=max(v for v in b.values() if math.isfinite(v)) if b else math.nan)

    def try_allocate(self, t, run_id, info=None, wait_s=0.0):
        busy = self._sample()
        best, _, _ = self.choose(t, busy)
        if best is None:
            return None
        return self._lock_cores(best[0], best[1], t, run_id, info, busy, False, wait_s)

    def fallback(self, t, run_id, info=None, wait_s=0.0):
        busy = self._sample()
        best, least, key = self.choose(t, busy)
        pick = best or least
        if pick is None:
            return None
        if best is None:
            self.log(f"  FLAG: no qualifying CCD set for t = {t} after {wait_s:.0f} s; least loaded {pick[0]} "
                     f"(max busy of used cpus {key[0]:.2f})")
        return self._lock_cores(pick[0], pick[1], t, run_id, info, busy, best is None, wait_s)


# ================================================================================================ launcher
class ProcessLauncher:
    def __init__(self, out_root, hold=True, python=sys.executable, extra_env=None):
        self.out_root, self.hold, self.python = out_root, bool(hold), python
        self.extra_env = dict(extra_env or {})

    def launch(self, job, paths, cores):
        os.makedirs(os.path.dirname(paths["json"]), exist_ok=True)
        cmd = [self.python, os.path.abspath(__file__), "run-one", "--job-json", json.dumps(job), "--out-root",
               self.out_root]
        if self.hold:
            cmd += ["--hold"]
        env = dict(self.extra_env)
        env.setdefault("NUMBA_CACHE_DIR", os.environ.get("NUMBA_CACHE_DIR",
                                                         os.path.expanduser("~/.cache/numba_hst")))
        return A.launch_pinned(cmd, cores, log_path=paths["log"], extra_env=env)


def is_real_process(h):
    return isinstance(h, subprocess.Popen)


# ================================================================================================ finalisation
def _log_ledger_once(paths, stage, item, threads, wall_s, note, ledger):
    if os.path.exists(paths["ledger_mark"]):
        return False
    H.ledger_append(stage, item, threads, wall_s, note=note, path=ledger)
    with open(paths["ledger_mark"], "w") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    return True


def finalize_job(job, paths, alloc, rc, proc_wall, ledger, stage, log=print, note_extra="", expected_sig=None):
    """Telemetry sidecar, ledger row (once per attempt), whole-run contention, status 'complete'.  Returns the record
    (dict(status='failed', ...) if the child left none)."""
    tel = A.finalize_telemetry(paths["tel_jsonl"], paths["tel_npz"])
    rec = None
    if os.path.exists(paths["json"]):
        try:
            rec = C.read_json(paths["json"])
        except (OSError, ValueError):
            rec = None
        if rec is not None and rec.get("status") not in ("ran", "complete"):
            rec = None
    pause = float(rec.get("disk_pause_s", 0.0)) if rec is not None else C.read_pause_marker(paths["pause"])
    used_wall = max(0.0, float(proc_wall) - pause)
    item = f"{job['system']} {job['config']} s{job['seed']}" + (f" r{job['r']}" if "r" in job else "") + \
        f" rep{job['replicate']}"
    note = f"rc {rc}" + (f"; ccds {alloc['ccds']} cores {A.cpu_list_str(alloc['cores'])}; flagged {alloc['flagged']}"
                         if alloc else "") + (f"; {note_extra}" if note_extra else "")
    _log_ledger_once(paths, stage, item, job["threads"], used_wall, note, ledger)
    if rec is None:
        log(f"  FAILED {H.job_key(job)} (rc {rc}); see {paths['log']}")
        return dict(status="failed", rc=rc, used_wall_s=used_wall)
    nccd = len(alloc["ccds"]) if alloc else (1 if job["threads"] <= H.CCD_CORES else 2)
    rec.update(allocation=({k: v for k, v in alloc.items() if k != "token"} if alloc else rec.get("allocation")),
               proc_wall_s=float(proc_wall), used_wall_s=used_wall, disk_pause_s=pause,
               telemetry=(os.path.basename(paths["tel_npz"]) if tel else None),
               used_core_h_run=job["threads"] * used_wall / 3600.0,
               alloc_core_h_run=nccd * H.CCD_CORES * used_wall / 3600.0)
    if expected_sig is not None and rec.get("signature") != json.loads(json.dumps(expected_sig)):
        rec["status"] = "signature_mismatch"
        C.write_json(paths["json"], rec)
        log(f"  SIGNATURE MISMATCH {item}")
        return rec
    ct = A.contention(tel, float(rec["t0_epoch"]), float(rec["wall_total_s"]), job["threads"], pid=rec.get("pid"))
    ct_ccd = A.contention(tel, float(rec["t0_epoch"]), float(rec["wall_total_s"]), job["threads"],
                          pid=rec.get("pid"), scope="ccd")
    rec.update(contention_whole=ct, contended=bool(ct["contended"]),
               contention_whole_ccd_scope={k: v for k, v in ct_ccd.items()
                                           if k in ("status", "max_foreign_busy", "max_foreign_busy_lower",
                                                    "worst_cpu", "contended")})
    if job["replicate"] > 0 and rec.get("science_identical_to_first") is False:
        rec["nondeterministic_repeat"] = True
        log(f"  WARNING {item}: a repeat's science arrays differ from replicate 0 (engine nondeterminism)")
    rec["status"] = "complete"
    rec["finalized_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if note_extra:
        rec["finalize_note"] = note_extra
    C.write_json(paths["json"], rec)
    if os.path.exists(paths["pause"]):
        os.remove(paths["pause"])
    return rec


def replicate_records(job0, out_root):
    recs = []
    for k in range(0, 1 + MAX_REPEATS):
        p = H.job_paths(dict(job0, replicate=k), out_root)
        if not os.path.exists(p["json"]):
            continue
        try:
            r = C.read_json(p["json"])
        except (OSError, ValueError):
            continue
        if r.get("status") == "complete":
            recs.append(r)
    return recs


def needs_repeat(recs):
    if not recs:
        return False
    if any(not r.get("contended", True) for r in recs):
        return False
    return len(recs) < 1 + MAX_REPEATS


def next_replicate(job0, out_root):
    k = 0
    while os.path.exists(H.job_paths(dict(job0, replicate=k), out_root)["json"]):
        k += 1
    return k


# ================================================================================================ scheduler
class Scheduler:
    """Run ``jobs`` (list, dispatch order) of one stage on exclusive CCD sets, at most ``max_ccds`` CCDs / ``max_cores``
    cores of this campaign at a time.  ``cost_s(job)`` = projected process wall of a job (cap projection)."""

    def __init__(self, jobs, stage, cost_s, out_root=H.OUT_ROOT, cfg=None, cap_core_h=math.inf, max_ccds=12,
                 max_cores=96, registry_root=H.REGISTRY_DIR, ledger=H.LEDGER, log=print, monitor=True,
                 monitor_allowed=None, run_timeout_s=7200.0, poll_s=ALLOC_POLL_S, max_wait_s=ALLOC_MAX_WAIT_S,
                 repeats=True, launcher=None, events=None):
        self.jobs = list(jobs)
        self.stage = stage
        self.cost_s = cost_s
        self.out_root = out_root
        self.cfg = cfg
        self.specs = {}
        self.cap = float(cap_core_h)
        self.max_ccds, self.max_cores = int(max_ccds), int(max_cores)
        self.registry = A.Registry(registry_root)
        self.monitor = bool(monitor)
        self.monitor_allowed = monitor_allowed if monitor_allowed is not None else [
            c for i in RESERVED_CCDS for c in A.ccd_cores(i)]
        self.allocator = HSTAllocator(self.registry, log=log, reserved=RESERVED_CCDS, threshold=QUALIFY_BUSY,
                                      sample_s=1.0)
        self.launcher = launcher or ProcessLauncher(out_root, hold=monitor)
        self.ledger = ledger
        self.log = log
        self.run_timeout_s = float(run_timeout_s)
        self.poll_s, self.max_wait_s = float(poll_s), float(max_wait_s)
        self.repeats = bool(repeats)
        self.events = events or os.path.join(out_root, "logs", f"{stage}_events.jsonl")
        self.guard = C.DiskGuard(log=log)
        self.stop = False
        self._core = None
        self.used_core_h = 0.0

    def spec(self, system):
        if system not in self.specs:
            self.specs[system] = H.system_spec(system, self.cfg)
        return self.specs[system]

    def event(self, **rec):
        os.makedirs(os.path.dirname(self.events), exist_ok=True)
        rec = dict(rec, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(self.events, "a") as fh:
            fh.write(json.dumps(C.json_safe(rec), default=str) + "\n")

    def sig(self, job):
        return H.job_signature(job, self.spec(job["system"]))

    # ---- planning
    def plan(self):
        """(queue of jobs to launch, records to recover, mismatches, n complete)."""
        queue, ran, mismatch, n_complete = [], [], [], 0
        live = self.registry.live_job_keys()
        for j0 in self.jobs:
            sig = self.sig(j0)
            st = H.record_status(H.job_paths(j0, self.out_root), sig)
            if st == "mismatch":
                mismatch.append(H.job_paths(j0, self.out_root)["json"])
                continue
            if st in (None, "failed"):
                queue.append(j0)
                continue
            if st == "ran":
                ran.append(j0)
            else:
                n_complete += 1
            rr = []
            for k in range(1, 1 + MAX_REPEATS):
                jk = dict(j0, replicate=k)
                stk = H.record_status(H.job_paths(jk, self.out_root), sig)
                if stk == "ran":
                    rr.append(jk)
                elif stk == "mismatch":
                    mismatch.append(H.job_paths(jk, self.out_root)["json"])
            ran.extend(rr)
            if st == "complete" and not rr and self.repeats and needs_repeat(replicate_records(j0, self.out_root)):
                queue.append(dict(j0, replicate=next_replicate(j0, self.out_root)))
        ran = [j for j in ran if H.job_key(j) not in live]
        return queue, ran, mismatch, n_complete

    def projected_core_h(self, jobs):
        return sum(j["threads"] * float(self.cost_s(j)) for j in jobs) / 3600.0

    # ---- main loop
    def run(self, dry_run=False):
        C.install_exit_signals()
        queue, ran, mismatch, n_complete = self.plan()
        if mismatch:
            raise H.HSTError(f"{self.stage}: complete records of a DIFFERENT job definition: {mismatch[:3]}")
        used0 = H.ledger_used_core_h(self.ledger)
        proj = self.projected_core_h(queue)
        self.log(f"{self.stage}: {n_complete} complete, {len(ran)} to recover, {len(queue)} to run; projected "
                 f"{proj:.2f} used core-h (study total so far {used0:.2f}, cap {self.cap})")
        if dry_run:
            return dict(n_complete=n_complete, to_run=len(queue), to_recover=len(ran), projected_core_h=proj,
                        used_core_h=used0)
        for j in ran:
            self._recover(j, queue)
        queue = collections.deque(queue)
        active = {}
        os.makedirs(os.path.join(self.out_root, "logs"), exist_ok=True)
        mon = A.start_monitor(self.registry.root, allowed=self.monitor_allowed,
                              log_path=os.path.join(self.out_root, "logs", f"{self.stage}_monitor.log")) \
            if self.monitor else None
        head_key, head_wait = None, None
        self.event(event="start", n_queue=len(queue), projected_core_h=proj, used_core_h=used0)
        try:
            while queue or active:
                self._core = A.pin_outside(self.registry, self.monitor_allowed, current=self._core, log=self.log)
                if mon is not None and mon.poll() is not None:
                    self.log("  monitor process exited; restarting it")
                    self.event(event="monitor_restart", rc=mon.returncode)
                    mon = A.start_monitor(self.registry.root, allowed=self.monitor_allowed)
                self._reap(active, queue)
                if self.stop:
                    if not active:
                        break
                    time.sleep(1.0)
                    continue
                if not self.guard.ok():
                    self._wait(active, queue, self.guard.poll_s)
                    continue
                if not queue:
                    self._wait(active, queue, 1.0)
                    continue
                job = queue[0]
                key = H.job_key(job)
                if key != head_key:
                    head_key, head_wait = key, None
                if key in self.registry.live_job_keys():          # an orphan of an earlier driver still runs
                    queue.rotate(-1)
                    self._wait(active, queue, 2.0)
                    continue
                n_ccd = A.n_ccds_for(job["threads"])
                held = self.registry.held()
                used_cores = sum(v[0]["threads"] for v in active.values())
                if used_cores + job["threads"] > self.max_cores or (
                        len([i for i in held if i not in RESERVED_CCDS]) + n_ccd > self.max_ccds):
                    self._wait(active, queue, 0.5)
                    continue
                used = H.ledger_used_core_h(self.ledger)
                remaining = self.projected_core_h(list(queue) + [v[0] for v in active.values()])
                if used + remaining > self.cap:
                    self.log(f"  CAP: used {used:.2f} + projected {remaining:.2f} > cap {self.cap}; no new launches")
                    self.event(event="cap_stop", used=used, remaining=remaining, cap=self.cap)
                    self.stop = True
                    continue
                now = time.monotonic()
                waited = 0.0 if head_wait is None else now - head_wait
                paths = H.job_paths(job, self.out_root)
                run_id = f"hst-{job['kind']}-{job['system']}-{job['config']}-s{job['seed']}-rep{job['replicate']}-" \
                         f"{os.getpid()}"
                info = dict(kind=job["kind"], telemetry=paths["tel_jsonl"], job_key=key,
                            driver=dict(pid=os.getpid(), start=A.proc_start_time(os.getpid())))
                alloc = self.allocator.try_allocate(job["threads"], run_id, info, waited)
                if alloc is None and head_wait is not None and waited >= self.max_wait_s:
                    alloc = self.allocator.fallback(job["threads"], run_id, info, waited)
                if alloc is None:
                    if head_wait is None:
                        head_wait = now
                    self._wait(active, queue, self.poll_s)
                    continue
                head_key, head_wait = None, None
                self._core = A.pin_outside(self.registry, self.monitor_allowed, current=self._core, log=self.log)
                queue.popleft()
                os.makedirs(os.path.dirname(paths["json"]), exist_ok=True)
                if os.path.exists(paths["json"]):                 # a failed earlier attempt: kept for the record
                    os.replace(paths["json"], paths["base"] + f".failed{int(time.time())}.json")
                for k in ("ledger_mark", "pause", "tel_jsonl"):
                    if os.path.exists(paths[k]):
                        os.remove(paths[k])
                sig = self.sig(job)
                t_launch = time.time()
                h = self.launcher.launch(job, paths, alloc["cores"])
                try:
                    self.registry.add_pid(alloc["ccds"], alloc["token"], h.pid)
                except RuntimeError as e:
                    self.log(f"  could not register the run pid of {run_id}: {e}")
                    self.event(event="add_pid_failed", run_id=run_id, error=str(e))
                active[run_id] = (job, paths, alloc, h, t_launch, sig)
                self.event(event="launch", run_id=run_id, job_key=key, ccds=alloc["ccds"], cores=alloc["cores"],
                           flagged=alloc["flagged"], wait_s=alloc["wait_s"])
        finally:
            self._shutdown_active(active)
            A.stop_monitor(mon)
        self.event(event="end", used_core_h=self.used_core_h, stopped=self.stop)
        return dict(used_core_h=self.used_core_h, stopped=self.stop)

    def _wait(self, active, queue, seconds):
        t_end = time.monotonic() + float(seconds)
        while True:
            self._reap(active, queue)
            if time.monotonic() >= t_end:
                return
            time.sleep(min(0.5, max(0.0, t_end - time.monotonic())))

    def _finalize(self, job, paths, alloc, rc, proc_wall, sig, note=""):
        try:
            rec = finalize_job(job, paths, alloc, rc, proc_wall, self.ledger, self.stage, self.log, note_extra=note,
                               expected_sig=sig)
        except Exception as e:  # noqa: BLE001
            self.log(f"  FINALIZE ERROR {H.job_key(job)}: {type(e).__name__}: {e} (record kept for recovery)")
            self.event(event="finalize_error", job_key=H.job_key(job), error=f"{type(e).__name__}: {e}")
            rec = dict(status="finalize_error", used_wall_s=proc_wall)
        if rec.get("status") == "signature_mismatch":
            self.event(event="stop", reason="signature mismatch", job_key=H.job_key(job))
            self.stop = True
        return rec

    def _after(self, job, rec, queue):
        if rec is None or rec.get("status") != "complete":
            if rec is not None and rec.get("status") == "failed":
                self.event(event="failed", job_key=H.job_key(job), rc=rec.get("rc"))
            return
        if not self.repeats:
            return
        j0 = dict(job, replicate=0)
        recs = replicate_records(j0, self.out_root)
        if needs_repeat(recs):
            queue.appendleft(dict(job, replicate=next_replicate(j0, self.out_root)))
            self.log(f"  repeat queued: {H.job_key(j0)} (contended: {rec['contention_whole'].get('status')}, "
                     f"max foreign busy {rec['contention_whole'].get('max_foreign_busy')})")
        elif not any(not r.get("contended", True) for r in recs):
            self.log(f"  FLAG: {H.job_key(j0)} has no uncontended replicate after {MAX_REPEATS} repeats")

    def _reap(self, active, queue):
        for rid in list(active):
            job, paths, alloc, h, t_launch, sig = active[rid]
            rc = h.poll()
            if rc is None and is_real_process(h) and time.time() - t_launch > self.run_timeout_s:
                self.log(f"  TIMEOUT {rid}: terminating after {time.time() - t_launch:.0f} s")
                rc = A.kill_process_group(h)
                rc = -9 if rc is None else rc
            if rc is None:
                continue
            proc_wall = time.time() - t_launch
            self.allocator.release(alloc)
            del active[rid]
            rec = self._finalize(job, paths, alloc, rc, proc_wall, sig)
            self.used_core_h += job["threads"] * float(rec.get("used_wall_s", proc_wall)) / 3600.0
            self.event(event="finish", run_id=rid, status=rec.get("status"), contended=rec.get("contended"),
                       proc_wall_s=proc_wall, wall_total_s=rec.get("wall_total_s"))
            if rec.get("status") == "complete" and job["kind"] == "bench":
                b = rec.get("bench", {})
                self.log(f"  {job['system']} {job['config']} r{job.get('r')} rep{job['replicate']}: "
                         f"c_ss {b.get('c_ss', math.nan) * 1e9:.2f} ns/ws "
                         f"({rec['contention_whole'].get('status')})")
            elif rec.get("status") == "complete":
                self.log(f"  {job['system']} {job['config']} s{job['seed']} rep{job['replicate']}: "
                         f"{rec['wall_total_s']:.1f} s ({rec['contention_whole'].get('status')})")
            try:
                self._after(job, rec, queue)
            except Exception as e:  # noqa: BLE001
                self.log(f"  repeat check failed for {H.job_key(job)}: {type(e).__name__}: {e}")

    def _recover(self, job, queue):
        paths = H.job_paths(job, self.out_root)
        rec0 = C.read_json(paths["json"])
        wall = rec0.get("process_age_s") or rec0.get("wall_total_s") or 0.0
        rec = self._finalize(job, paths, rec0.get("allocation"), 0, float(wall), self.sig(job),
                             note="recovered after the driver exited (wall = the child's process age)")
        self.event(event="recovered", job_key=H.job_key(job), status=rec.get("status"))
        self._after(job, rec, queue)

    def _shutdown_active(self, active):
        for rid in list(active):
            job, paths, alloc, h, t_launch, sig = active.pop(rid)
            rc = A.kill_process_group(h) if is_real_process(h) else h.poll()
            proc_wall = time.time() - t_launch
            try:
                st = H.record_status(paths, sig)
                if st == "ran":
                    self._finalize(job, paths, alloc, 0, proc_wall, sig,
                                   note="process terminated at driver exit after writing its record")
                else:
                    self._finalize(job, paths, alloc, rc if rc not in (None, 0) else -15, proc_wall, sig,
                                   note="terminated at driver exit")
            finally:
                self.allocator.release(alloc)
            self.event(event="terminated_at_exit", run_id=rid, rc=rc)


# ================================================================================================ CLI (child)
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run-one")
    r.add_argument("--job-json", required=True)
    r.add_argument("--out-root", default=H.OUT_ROOT)
    r.add_argument("--hold", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run-one":
        job = json.loads(a.job_json)
        H.run_job_child(job, a.out_root, hold=a.hold)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
