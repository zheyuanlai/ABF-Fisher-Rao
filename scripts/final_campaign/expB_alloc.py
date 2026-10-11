#!/usr/bin/env python
"""Experiment B: CCD-aware CPU allocation, campaign lock registry, telemetry monitor and the contention rule.

Prereg section 2.4 ("Execution environment for every timed run") and 2.5 ("Contention").  Host topology (measured
at the freeze, 2 x EPYC 9554): 16 L3 groups (CCDs) of 8 physical cores, CCD i = cpus 8i .. 8i+7; the SMT sibling of
core c is c + 128; NUMA node 0 = cores 0-63 (CCDs 0-7), node 1 = cores 64-127 (CCDs 8-15).

Allocation.  A run with t <= 8 threads gets ONE CCD to itself, t = 16 two CCDs of one node (t = 32, descriptive
benchmark only: four CCDs of one node).  Qualifying sets: every core AND every SMT sibling of the set is <= 10 % busy
over a 2 s /proc/stat sample and no CCD of the set is held by this campaign (lock registry below).  Among qualifying
sets the least loaded (max busy, then summed busy) is taken.  If none qualifies the launcher polls every 30 s for up
to 30 min, then takes the least-loaded non-held set ("qualifying as far as possible") and FLAGS the run.  The run is
pinned (taskset) to its first t cores in CCD order; the other cores of its CCDs and every sibling stay idle.

Registry.  /tmp/claude-1008/final_campaign_alloc/ccdNN.lock, one JSON file per held CCD, created with O_EXCL (atomic
across processes): run_id, token, the holder pids with their /proc start times (the allocating process and, once
launched, the run process), cores, cpus, threads, telemetry path.  A lock is STALE (removed) when none of its pids
is alive with the recorded start time (pid reuse is not mistaken for life).

Monitor (``python expB_alloc.py monitor``).  Pinned to ONE core outside every held CCD (re-pinned whenever the
registry changes so that it never sits on an allocation), it samples at 1 Hz: /proc/stat for every core + sibling of
each active run's CCDs, the run process's threads (/proc/<pid>/task/*/stat utime + stime and last processor, and
/proc/<pid>/task/*/status nonvoluntary_ctxt_switches) and scaling_cur_freq of the run's cores.  Per run one compact
JSON-lines file (header + one line per sample with the per-interval deltas); exits when its parent exits.  Every
row's busy deltas and own-thread deltas cover the SAME interval (per-run previous snapshot).  Own-tick accounting at
the two ends of a run:
  * first sight of the run process (``i``): if the process started after the run's previous sample (its /proc
    starttime vs /proc/uptime at that sample) every tick of its threads belongs to the interval (i = 2, exact);
    otherwise its history is unknown (i = 1, own = 0) -- e.g. a monitor restarted mid-run;
  * a row whose run process is absent (exited, or not registered yet) has x = 1 (own = 0);
  * a ``seen`` marker line is written at the first sample that sees the run process.
The run child holds the boundaries outside its timed region (``hold_for_monitor``): after importing and building the
engine, before the engine call, it waits for a sample of itself taken after that moment (so the first-sight interval
and the import phase end before the timed start), and after its records are written it waits for a row sampled after
that moment with its threads alive (so the exit interval starts after the timed end).

Contention rule (``contention``): over an endpoint's wall window [t0, t0 + W] (the whole run for Ibar_F^wall) a run is
contended iff any core or sibling of its CCDs shows a FOREIGN busy fraction > 0.10 (busy jiffies minus the run's own
thread ticks on that cpu, summed over the 1 s sample intervals overlapping the window, divided by the total jiffies;
own ticks a thread accrued on another run core before it was bound -- attributed to its last processor -- are moved
back among the run's cores within the interval), or its threads incur > 5 involuntary context switches per
thread-second.  Rows with incomplete own accounting (i = 1
or x = 1) give bounds: upper = own taken as known (0), lower = all busy on the run's cores taken as own (their
switch counts unknown); contended iff the lower bound exceeds a threshold, uncontended iff the upper bound does not,
else 'unknown'.  Missing / insufficient telemetry (no row, window not covered, a gap or overlap between rows, a row
of another process) -> 'unknown', which the scheduler treats as contended (conservative).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import secrets
import sys
import time

import numpy as np

N_CCD = 16
CCD_SIZE = 8
N_PHYS = 128
SMT_OFFSET = 128
REGISTRY_DIR = "/tmp/claude-1008/final_campaign_alloc"
BUSY_THRESHOLD = 0.10
SAMPLE_S = 2.0
POLL_S = 30.0
MAX_WAIT_S = 1800.0
CTX_THRESHOLD = 5.0          # involuntary context switches per thread-second
MONITOR_INTERVAL_S = 1.0
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
GAP_TOL_S = 0.05             # consecutive rows of one run abut exactly (tp_{k+1} == t_k); more is a gap
HOLD_TIMEOUT_S = 6.0         # a run child waits at most this long for the monitor at each boundary


# ================================================================================================ topology
def ccd_cores(i):
    i = int(i)
    if not 0 <= i < N_CCD:
        raise ValueError(f"CCD index {i} out of range")
    return list(range(CCD_SIZE * i, CCD_SIZE * i + CCD_SIZE))


def sibling(c):
    return int(c) + SMT_OFFSET if int(c) < SMT_OFFSET else int(c) - SMT_OFFSET


def ccd_cpus(i):
    """Physical cores of CCD i and their SMT siblings."""
    cs = ccd_cores(i)
    return cs + [sibling(c) for c in cs]


def node_of_ccd(i):
    return 0 if int(i) < N_CCD // 2 else 1


def ccd_of_cpu(c):
    c = int(c)
    return (c % SMT_OFFSET) // CCD_SIZE


def n_ccds_for(t):
    t = int(t)
    if t < 1:
        raise ValueError("threads must be >= 1")
    if t <= CCD_SIZE:
        return 1
    n = -(-t // CCD_SIZE)
    if n not in (2, 4):
        raise ValueError(f"t = {t}: only t <= 8 (1 CCD), t <= 16 (2 CCDs) and t <= 32 (4 CCDs) are allocated")
    return n


def candidate_sets(t):
    """All CCD sets a t-thread run may get: single CCDs (t <= 8) or n CCDs of ONE NUMA node."""
    n = n_ccds_for(t)
    if n == 1:
        return [(i,) for i in range(N_CCD)]
    out = []
    for node in (0, 1):
        ccds = [i for i in range(N_CCD) if node_of_ccd(i) == node]
        out.extend(itertools.combinations(ccds, n))
    return out


def used_cores(ccds, t):
    cores = [c for i in ccds for c in ccd_cores(i)]
    if t > len(cores):
        raise ValueError(f"{t} threads do not fit on CCDs {ccds}")
    return cores[:int(t)]


def set_cpus(ccds):
    return sorted(c for i in ccds for c in ccd_cpus(i))


def cpu_list_str(cpus):
    """[120, 121, 122, 125] -> '120-122,125' (taskset / OMP-friendly)."""
    cpus = sorted(set(int(c) for c in cpus))
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(f"{cpus[i]}" if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(out)


def parse_cpu_list(s):
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return sorted(set(out))


# ================================================================================================ /proc/stat
def read_proc_stat():
    with open("/proc/stat") as fh:
        return fh.read()


def parse_proc_stat(text):
    """{cpu: (busy_jiffies, total_jiffies)}; total = user + nice + system + idle + iowait + irq + softirq + steal
    (guest time is already inside user), busy = total - idle - iowait."""
    out = {}
    for line in text.splitlines():
        if line.startswith("cpu") and len(line) > 3 and line[3].isdigit():
            f = line.split()
            v = [int(x) for x in f[1:9]] + [0] * max(0, 8 - len(f[1:9]))
            tot = sum(v[:8])
            idle = v[3] + v[4]
            out[int(f[0][3:])] = (tot - idle, tot)
    return out


def busy_fractions(a, b):
    """Busy fraction per cpu between two parsed snapshots (cpus missing from either are absent)."""
    out = {}
    for c in a:
        if c in b:
            dt = b[c][1] - a[c][1]
            out[c] = (b[c][0] - a[c][0]) / dt if dt > 0 else 0.0
    return out


def sample_busy(cpus=None, sample_s=SAMPLE_S, reader=read_proc_stat, sleep=time.sleep):
    a = parse_proc_stat(reader())
    sleep(float(sample_s))
    b = parse_proc_stat(reader())
    bf = busy_fractions(a, b)
    if cpus is not None:
        bf = {c: bf.get(c, 1.0) for c in cpus}
    return bf


# ================================================================================================ registry
def proc_start_time(pid):
    """/proc/<pid>/stat starttime (clock ticks since boot) or None if the process does not exist."""
    try:
        with open(f"/proc/{int(pid)}/stat") as fh:
            s = fh.read()
        return int(s[s.rindex(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def pid_alive(pid, start=None):
    st = proc_start_time(pid)
    if st is None:
        return False
    return start is None or int(start) == st


def read_uptime():
    """Seconds since boot (/proc/uptime, the clock of /proc/<pid>/stat starttime); None if unavailable."""
    try:
        with open("/proc/uptime") as fh:
            return float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def process_age_s(pid=None):
    """Seconds since the process started (uptime - starttime / CLK_TCK); NaN if unavailable."""
    st = proc_start_time(os.getpid() if pid is None else pid)
    up = read_uptime()
    return (up - st / CLK_TCK) if (st is not None and up is not None) else math.nan


def job_key(job):
    """Identity of a scheduled job (recorded in its CCD lock so that no second process ever runs it concurrently)."""
    return f"{job['kind']}|{job['system']}|{job['arm']}|s{int(job['seed'])}|r{int(job.get('replicate', 0))}"


class Registry:
    """CCD lock files of this campaign (atomic O_EXCL creation; stale cleanup by pid + start time)."""

    def __init__(self, root=REGISTRY_DIR, alive=pid_alive, partial_grace_s=30.0):
        self.root = root
        self.alive = alive
        self.partial_grace_s = float(partial_grace_s)
        os.makedirs(self.root, exist_ok=True)

    def path(self, ccd):
        return os.path.join(self.root, f"ccd{int(ccd):02d}.lock")

    def _read(self, p):
        try:
            with open(p) as fh:
                txt = fh.read()
            return json.loads(txt) if txt.strip() else None
        except (OSError, ValueError):
            return None

    def _stale(self, p, info):
        if info is None:          # being written, or garbage: stale only after a grace period
            try:
                return time.time() - os.path.getmtime(p) > self.partial_grace_s
            except OSError:
                return False
        pids = info.get("pids") or []
        return not any(self.alive(x.get("pid"), x.get("start")) for x in pids)

    def held(self, cleanup=True):
        """{ccd: lock info} of the live locks (stale ones removed when cleanup)."""
        out = {}
        for f in sorted(os.listdir(self.root)):
            if not (f.startswith("ccd") and f.endswith(".lock")):
                continue
            p = os.path.join(self.root, f)
            info = self._read(p)
            if self._stale(p, info):
                if cleanup:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                continue
            try:
                out[int(f[3:5])] = info if info is not None else {"partial": True}
            except ValueError:
                continue
        return out

    def held_cpus(self):
        return sorted(c for i in self.held() for c in ccd_cpus(i))

    def acquire(self, ccds, info):
        """Lock every CCD of ``ccds`` or none; returns the token or None."""
        token = secrets.token_hex(8)
        rec = dict(info, token=token, ccds=[int(i) for i in ccds],
                   created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        if "pids" not in rec:
            rec["pids"] = [dict(pid=os.getpid(), start=proc_start_time(os.getpid()))]
        got = []
        for i in sorted(int(c) for c in ccds):
            p = self.path(i)
            for attempt in (0, 1):
                try:
                    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                except FileExistsError:
                    if attempt == 0 and self._stale(p, self._read(p)):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                        continue
                    fd = None
                break
            if fd is None:
                for j in got:
                    self._remove_own(j, token)
                return None
            with os.fdopen(fd, "w") as fh:
                json.dump(rec, fh)
                fh.flush()
                os.fsync(fh.fileno())
            got.append(i)
        return token

    def _remove_own(self, ccd, token):
        p = self.path(ccd)
        info = self._read(p)
        if info is not None and info.get("token") == token:
            try:
                os.remove(p)
            except OSError:
                pass

    def update(self, ccds, token, **fields):
        for i in ccds:
            p = self.path(i)
            info = self._read(p)
            if info is None or info.get("token") != token:
                raise RuntimeError(f"CCD {i} lock is not held with token {token}")
            info.update(fields)
            tmp = p + f".tmp{os.getpid()}"
            with open(tmp, "w") as fh:
                json.dump(info, fh)
            os.replace(tmp, p)

    def add_pid(self, ccds, token, pid, role="run"):
        info = self._read(self.path(sorted(ccds)[0])) or {}
        pids = list(info.get("pids", [])) + [dict(pid=int(pid), start=proc_start_time(pid), role=role)]
        self.update(ccds, token, pids=pids, run_pid=int(pid))

    def release(self, ccds, token):
        for i in ccds:
            self._remove_own(i, token)

    def active_runs(self):
        """Lock infos grouped by run_id (one entry per run)."""
        runs = {}
        for i, info in self.held().items():
            if info.get("partial"):
                continue
            rid = info.get("run_id", f"ccd{i}")
            runs.setdefault(rid, info)
        return list(runs.values())

    def live_job_keys(self):
        """{job_key: lock info} of the live locks that carry a job key (a run whose process is still alive, possibly
        orphaned by a driver that exited)."""
        return {info["job_key"]: info for info in self.active_runs() if info.get("job_key")}


# ================================================================================================ allocation
def ccd_load(busy, i):
    vals = [float(busy.get(c, 1.0)) for c in ccd_cpus(i)]
    return max(vals), float(sum(vals))


def choose_set(t, busy, held=(), threshold=BUSY_THRESHOLD):
    """(best qualifying CCD set or None, least-loaded non-held set or None, {set: (max busy, summed busy)}).
    A cpu absent from ``busy`` counts as fully busy."""
    held = set(int(i) for i in held)
    sets = [s for s in candidate_sets(t) if not set(s) & held]
    loads = {}
    for s in sets:
        per = [ccd_load(busy, i) for i in s]
        loads[s] = (max(p[0] for p in per), sum(p[1] for p in per))
    qual = [s for s in sets if loads[s][0] <= threshold]
    key = lambda s: (loads[s][0], loads[s][1], s)          # noqa: E731
    best = min(qual, key=key) if qual else None
    least = min(sets, key=key) if sets else None
    return best, least, loads


def ccds_of_cpus(cpus):
    return sorted({ccd_of_cpu(c) for c in (cpus or ())})


class Allocator:
    """try_allocate (one 2 s sample) / fallback (least loaded + flag) / allocate (blocking: poll 30 s, up to 30 min,
    then fallback).  reader / sleep / clock are injectable (tests mock /proc/stat).  ``reserved``: CCDs never
    allocated (the home of the driver and the monitor, e.g. the CCDs of --monitor-cpus)."""

    def __init__(self, registry=None, reader=read_proc_stat, sleep=time.sleep, clock=time.monotonic,
                 threshold=BUSY_THRESHOLD, sample_s=SAMPLE_S, poll_s=POLL_S, max_wait_s=MAX_WAIT_S, log=print,
                 reserved=()):
        self.registry = registry if registry is not None else Registry()
        self.reader, self.sleep, self.clock = reader, sleep, clock
        self.threshold, self.sample_s, self.poll_s, self.max_wait_s = threshold, sample_s, poll_s, max_wait_s
        self.log = log
        self.reserved = sorted(int(i) for i in (reserved or ()))

    def _excluded(self):
        return set(self.registry.held()) | set(self.reserved)

    def _sample(self):
        return sample_busy(None, self.sample_s, self.reader, self.sleep)

    def _lock(self, s, t, run_id, info, busy, flagged, wait_s):
        cores = used_cores(s, t)
        rec = dict(info or {}, run_id=run_id, threads=int(t), cores=cores, cpus=set_cpus(s))
        token = self.registry.acquire(s, rec)
        if token is None:
            return None
        cpus = set_cpus(s)
        b = {int(c): float(busy.get(c, math.nan)) for c in cpus}
        return dict(run_id=run_id, token=token, ccds=list(s), cores=cores, cpus=cpus, threads=int(t),
                    flagged=bool(flagged), wait_s=float(wait_s), busy_at_launch=b,
                    max_busy_at_launch=max(v for v in b.values() if math.isfinite(v)) if b else math.nan)

    def try_allocate(self, t, run_id, info=None, wait_s=0.0):
        busy = self._sample()
        best, _, _ = choose_set(t, busy, self._excluded(), self.threshold)
        if best is None:
            return None
        return self._lock(best, t, run_id, info, busy, False, wait_s)

    def fallback(self, t, run_id, info=None, wait_s=0.0):
        busy = self._sample()
        best, least, loads = choose_set(t, busy, self._excluded(), self.threshold)
        s = best or least
        if s is None:
            return None
        if best is None:
            self.log(f"  FLAG: no qualifying CCD set for t = {t} after {wait_s:.0f} s; least loaded {s} "
                     f"(max busy {loads[s][0]:.2f})")
        return self._lock(s, t, run_id, info, busy, best is None, wait_s)

    def allocate(self, t, run_id, info=None):
        t0 = self.clock()
        while True:
            waited = self.clock() - t0
            a = self.try_allocate(t, run_id, info, waited)
            if a is not None:
                return a
            if waited >= self.max_wait_s:
                a = self.fallback(t, run_id, info, waited)
                if a is not None:
                    return a
                self.log(f"  no non-held CCD set for t = {t}; waiting")
            self.sleep(self.poll_s)

    def release(self, alloc):
        self.registry.release(alloc["ccds"], alloc["token"])


def omp_env(cores, base=None):
    """Thread / affinity environment of a run pinned to ``cores`` (t = len(cores))."""
    t = len(cores)
    env = dict(base if base is not None else os.environ)
    env.update(OMP_NUM_THREADS=str(t), NUMBA_NUM_THREADS=str(t), OMP_PROC_BIND="close",
               OMP_PLACES=",".join(f"{{{c}}}" for c in cores), OMP_WAIT_POLICY="active", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMBA_THREADING_LAYER="omp", OMP_DYNAMIC="false", CUDA_VISIBLE_DEVICES="")
    return env


def pin_outside(registry, allowed=None, busy=None, current=None, log=None):
    """Pin this process to one core outside every held CCD (and its siblings): the least busy allowed core; if every
    allowed core is held, the least busy non-held core anywhere (logged).  Returns the core (unchanged if the current
    one is still outside); None if no core is available."""
    held = set(registry.held_cpus())
    pool = sorted(set(allowed if allowed is not None else range(N_PHYS)) - held)
    if not pool and allowed is not None:
        pool = sorted(set(range(N_PHYS)) - held)
        if pool and current not in pool and log is not None:
            log(f"  every allowed core of this process is on a held CCD; moving to a non-held core outside the "
                f"allowed set")
    if not pool:
        return None
    allowed = pool
    if current is not None and current in allowed:
        return current
    c = min(allowed, key=lambda x: (float((busy or {}).get(x, 0.0)) + float((busy or {}).get(sibling(x), 0.0)), x))
    try:
        os.sched_setaffinity(0, {c})
    except (OSError, AttributeError):
        return None
    try:                                  # every thread of this process (library thread pools included)
        for tid in os.listdir("/proc/self/task"):
            try:
                os.sched_setaffinity(int(tid), {c})
            except OSError:
                pass
    except OSError:
        pass
    return c


# ================================================================================================ telemetry
def _task_stats(pid):
    """{tid: (utime + stime ticks, last processor, nonvoluntary ctx switches)} of the process's threads."""
    out = {}
    base = f"/proc/{int(pid)}/task"
    try:
        tids = os.listdir(base)
    except OSError:
        return out
    for tid in tids:
        try:
            with open(f"{base}/{tid}/stat") as fh:
                s = fh.read()
            f = s[s.rindex(")") + 2:].split()
            ticks = int(f[11]) + int(f[12])
            proc = int(f[36])
            nv = 0
            with open(f"{base}/{tid}/status") as fh:
                for line in fh:
                    if line.startswith("nonvoluntary_ctxt_switches"):
                        nv = int(line.split()[1])
                        break
            out[int(tid)] = (ticks, proc, nv)
        except (OSError, ValueError, IndexError):
            continue
    return out


def read_freq_khz(cpus):
    out = []
    for c in cpus:
        try:
            with open(f"/sys/devices/system/cpu/cpu{int(c)}/cpufreq/scaling_cur_freq") as fh:
                out.append(int(fh.read().strip()))
        except (OSError, ValueError):
            out.append(-1)
    return out


class Monitor:
    """1 Hz telemetry of every active run of the registry (see module docstring).  Per run, the busy deltas and the
    own-thread deltas of a row cover the same interval (tp, t]: the run's previous sample."""

    def __init__(self, registry, interval=MONITOR_INTERVAL_S, reader=read_proc_stat, task_stats=_task_stats,
                 freq=read_freq_khz, allowed=None, parent=None, clock=time.time, pin=True, uptime=read_uptime,
                 proc_start=proc_start_time, log=None):
        self.registry = registry
        self.interval = float(interval)
        self.reader, self.task_stats, self.freq, self.clock = reader, task_stats, freq, clock
        self.uptime, self.proc_start = uptime, proc_start
        self.allowed = allowed
        self.parent = parent
        self.pin = pin
        self.log = log
        self.core = None
        self.prev_stat = None
        self.prev_t = None
        self.runs = {}          # run_id -> dict(path, prev_t, prev_up, prev_stat, prev_tasks, prev_pid)

    @staticmethod
    def _snap(stat, cpus):
        return {c: stat.get(c, (0, 0)) for c in cpus}

    def sample_once(self):
        t = self.clock()
        up = self.uptime()
        stat = parse_proc_stat(self.reader())
        busy_all = busy_fractions(self.prev_stat, stat) if self.prev_stat else {}
        if self.pin:
            self.core = pin_outside(self.registry, self.allowed, busy_all, self.core, log=self.log)
        live = set()
        for info in self.registry.active_runs():
            rid = info.get("run_id")
            path = info.get("telemetry")
            if not rid or not path:
                continue
            drv = (info.get("driver") or {}).get("pid")
            if self.parent is not None and drv is not None and int(drv) != int(self.parent):
                continue                      # another driver's run: its own monitor samples it (no double rows)
            live.add(rid)
            cpus = [int(c) for c in info.get("cpus", [])]
            cores = [int(c) for c in info.get("cores", [])]
            pid = info.get("run_pid")
            tasks = self.task_stats(pid) if pid else {}
            alive = bool(pid and tasks)
            st = self.runs.get(rid)
            if st is None:                    # first sample of this run: header (+ seen marker) and the baseline
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "a") as fh:
                    fh.write(json.dumps(dict(hdr=1, run_id=rid, cpus=cpus, cores=cores, clk_tck=CLK_TCK,
                                             interval=self.interval, monitor_pid=os.getpid())) + "\n")
                    if alive:
                        fh.write(json.dumps(dict(seen=int(pid), t=round(t, 3), baseline=1)) + "\n")
                self.runs[rid] = dict(path=path, prev_t=t, prev_up=up, prev_stat=self._snap(stat, cpus),
                                      prev_tasks=tasks if alive else None, prev_pid=pid if alive else None)
                continue
            db = [stat.get(c, (0, 0))[0] - st["prev_stat"].get(c, (0, 0))[0] for c in cpus]
            dn = [stat.get(c, (0, 0))[1] - st["prev_stat"].get(c, (0, 0))[1] for c in cpus]
            own = [0] * len(cpus)
            pos = {c: k for k, c in enumerate(cpus)}
            dv, init, x, seen = 0, 0, 0, False
            if not alive:
                x = 1                         # run process absent (not registered yet, or exited): own unknown
            elif st["prev_tasks"] is None or st["prev_pid"] != pid:
                seen = True                   # first sight of this process
                start = self.proc_start(pid)
                if start is not None and st["prev_up"] is not None and start / CLK_TCK >= st["prev_up"]:
                    init = 2                  # started after the previous sample: all its ticks are in (tp, t]
                    for _tid, (ticks, proc, nv) in tasks.items():
                        dv += nv
                        if proc in pos and ticks > 0:
                            own[pos[proc]] += ticks
                else:
                    init = 1                  # history unknown (e.g. a monitor restarted mid-run): own = 0
            else:
                for tid, (ticks, proc, nv) in tasks.items():
                    p0 = st["prev_tasks"].get(tid)     # a thread unseen before started after the last sample
                    d_ticks = ticks - (p0[0] if p0 else 0)
                    dv += nv - (p0[2] if p0 else 0)
                    if proc in pos and d_ticks > 0:
                        own[pos[proc]] += d_ticks
            fq = self.freq(cores)
            with open(path, "a") as fh:
                if seen:
                    fh.write(json.dumps(dict(seen=int(pid), t=round(t, 3))) + "\n")
                fh.write(json.dumps(dict(t=round(t, 3), tp=round(st["prev_t"], 3), b=db, n=dn, o=own, dv=int(dv),
                                         h=len(tasks), pid=pid, i=init, x=x,
                                         f=[round(v / 1000) for v in fq])) + "\n")
            st.update(prev_t=t, prev_up=up, prev_stat=self._snap(stat, cpus), prev_tasks=tasks if alive else None,
                      prev_pid=pid if alive else None)
        for rid in list(self.runs):
            if rid not in live:
                del self.runs[rid]
        self.prev_stat, self.prev_t = stat, t

    def loop(self, max_iter=None):
        k = 0
        while max_iter is None or k < max_iter:
            t0 = time.monotonic()
            if self.parent is not None and not pid_alive(self.parent):
                return
            self.sample_once()
            k += 1
            time.sleep(max(0.0, self.interval - (time.monotonic() - t0)))


def _tail_lines(path, max_bytes=262144):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - int(max_bytes)))
            data = fh.read()
    except OSError:
        return []
    out = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def hold_for_monitor(tel_path, after=None, pid=None, timeout=HOLD_TIMEOUT_S, poll=0.05, sleep=time.sleep,
                     clock=time.monotonic, boundary=None):
    """Called by a run child to keep the monitor's boundary intervals outside its timed region: wait until the
    telemetry file shows a monitor sample of this process (a 'seen' marker, or a data row with its threads alive)
    taken after the epoch ``after`` (None: any).  Start boundary (after the engine is imported and built, before the
    engine call, after = that moment): the first-sight interval and the import phase then end before the timed start.
    End boundary (after the records are written, after = that moment): the exit interval then starts after the timed
    end.  Returns dict(boundary, seen, waited_s, row_t); seen False after ``timeout`` (no monitor)."""
    pid = os.getpid() if pid is None else int(pid)
    lo = -math.inf if after is None else float(after)
    c0 = clock()
    label = boundary or ("start" if after is None else "end")
    while True:
        for r in reversed(_tail_lines(tel_path)):
            data = "b" in r and r.get("pid") == pid and int(r.get("h", 0) or 0) > 0
            if (r.get("seen") == pid or data) and float(r.get("t", -math.inf)) > lo:
                return dict(boundary=label, seen=True, waited_s=clock() - c0, row_t=r.get("t"))
        if clock() - c0 >= float(timeout):
            return dict(boundary=label, seen=False, waited_s=clock() - c0, row_t=None)
        sleep(poll)


def load_telemetry(path):
    """Parse a monitor JSONL file -> dict(cpus, cores, t, tp (S,), busy, tot, own (S, C) jiffies per interval,
    dv (S,) involuntary switches per interval, nthr (S,), init (S,) first-sight flag (1 unknown / 2 exact), x (S,)
    run process absent, pid (S,) (-1 = none recorded), freq_mhz (S, n_cores)); None if absent / empty.  Marker lines
    ('seen') are skipped."""
    if not path or not os.path.exists(path):
        return None
    hdr, rows = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("hdr"):
                hdr = hdr or d
            elif "b" in d:
                rows.append(d)
    if hdr is None:
        return None
    C = len(hdr["cpus"])
    rows = [r for r in rows if len(r.get("b", [])) == C]
    S = len(rows)

    def arr(key, dtype=np.int64, width=None):
        if width is None:
            return np.array([r[key] for r in rows], dtype=dtype) if S else np.zeros(0, dtype=dtype)
        return np.array([r[key] for r in rows], dtype=dtype).reshape(S, width) if S else np.zeros((0, width), dtype)
    return dict(cpus=list(hdr["cpus"]), cores=list(hdr.get("cores", [])), clk_tck=hdr.get("clk_tck", CLK_TCK),
                t=arr("t", float), tp=arr("tp", float), busy=arr("b", np.int64, C), tot=arr("n", np.int64, C),
                own=arr("o", np.int64, C), dv=arr("dv"), nthr=arr("h"),
                init=np.array([int(r.get("i", 0)) for r in rows], dtype=np.int64),
                x=np.array([int(r.get("x", 0)) for r in rows], dtype=np.int64),
                pid=np.array([int(r["pid"]) if r.get("pid") is not None else -1 for r in rows], dtype=np.int64),
                freq_mhz=(np.array([r.get("f", []) for r in rows], dtype=float) if S and all(
                    len(r.get("f", [])) == len(hdr.get("cores", [])) for r in rows) else np.zeros((S, 0))))


def telemetry_arrays(tel):
    """Flat arrays for a compressed npz sidecar."""
    if tel is None:
        return {}
    return {f"tel_{k}": np.asarray(v) for k, v in tel.items() if k not in ("clk_tck",)}


def telemetry_from_arrays(z):
    keys = ("cpus", "cores", "t", "tp", "busy", "tot", "own", "dv", "nthr", "freq_mhz", "init", "x", "pid")
    if not all(f"tel_{k}" in z for k in keys[:9]):
        return None
    out = {k: np.asarray(z[f"tel_{k}"]) for k in keys if f"tel_{k}" in z}
    out["cpus"] = [int(c) for c in out["cpus"]]
    out["cores"] = [int(c) for c in out.get("cores", [])]
    return out


def contention(tel, t0_epoch, window_s, threads, busy_thr=BUSY_THRESHOLD, ctx_thr=CTX_THRESHOLD, pid=None, scope="own"):
    """Contention of one run over the wall window [t0, t0 + window_s] (prereg 2.5).

    Uses every sample interval (tp_i, t_i] overlapping the window.  Foreign busy per cpu = sum(max(busy - own, 0)) /
    sum(total) over those intervals; ctx rate = sum(involuntary switches) / (threads x covered seconds).  Rows with
    incomplete own accounting (init == 1: first sight with unknown history; x == 1: run process absent) bound the
    verdict: the upper bound takes their own ticks as recorded (0), the lower bound takes all their busy on the run's
    cores as own (their switch counts are unknown: rate upper bound inf).  contended iff a lower bound exceeds its
    threshold; uncontended iff no upper bound does and the telemetry is sound; else unknown.  Unsound telemetry: no
    overlapping row, the window not covered, a gap or overlap between consecutive rows, a row of a process other than
    ``pid`` (when given), a cpu without jiffies.  Returns dict(status 'contended' | 'uncontended' | 'unknown',
    contended (bool; unknown -> True), reasons, max_foreign_busy (upper), max_foreign_busy_lower, worst_cpu,
    ctx_per_thread_s, n_intervals, covered, ...).

    ``scope`` (Amendment 2 of the preregistration): "own" (default) decides on the run's own cores and their SMT
    siblings only -- where foreign load directly takes cycles or core resources from the run; the other cpus of the
    run's CCDs (L3 co-tenancy) are still measured and returned as the covariate ``ccd_max_foreign_busy`` (upper /
    ``_lower``).  "ccd" decides on every monitored cpu of the run's CCDs (the pre-amendment rule, kept for the
    sensitivity analysis)."""
    out = dict(window=[float(t0_epoch), float(t0_epoch) + float(window_s)], threads=int(threads),
               busy_threshold=busy_thr, ctx_threshold=ctx_thr)
    if tel is None or len(tel.get("t", [])) == 0:
        return dict(out, status="unknown", contended=True, reasons=["no telemetry"], max_foreign_busy=math.nan,
                    max_foreign_busy_lower=math.nan, worst_cpu=None, ctx_per_thread_s=math.nan, n_intervals=0,
                    covered=False)
    if not (math.isfinite(float(window_s)) and float(window_s) >= 0):
        raise ValueError(f"window must be finite and >= 0 (got {window_s})")
    a, b = float(t0_epoch), float(t0_epoch) + float(window_s)
    t, tp = np.asarray(tel["t"], float), np.asarray(tel["tp"], float)
    sel = (t > a) & (tp < b) if b > a else (t >= a) & (tp <= a)
    if not np.any(sel):
        return dict(out, status="unknown", contended=True, reasons=["no telemetry interval overlaps the window"],
                    max_foreign_busy=math.nan, max_foreign_busy_lower=math.nan, worst_cpu=None,
                    ctx_per_thread_s=math.nan, n_intervals=0, covered=False)
    idx = np.nonzero(sel)[0]
    idx = idx[np.argsort(tp[idx], kind="stable")]
    S = t.size
    busy = np.asarray(tel["busy"], float)[idx]
    tot = np.asarray(tel["tot"], float)[idx]
    own = np.asarray(tel["own"], float)[idx]
    dv = np.asarray(tel["dv"], float)[idx]
    init = np.asarray(tel.get("init", np.zeros(S)), dtype=np.int64)[idx]
    xx = np.asarray(tel.get("x", np.zeros(S)), dtype=np.int64)[idx]
    rpid = np.asarray(tel.get("pid", np.full(S, -1)), dtype=np.int64)[idx]
    ts, tps = t[idx], tp[idx]
    cpus = [int(c) for c in tel["cpus"]]
    cores = set(int(c) for c in tel.get("cores", []))
    is_core = np.array([c in cores for c in cpus], dtype=bool)
    inc = (init == 1) | (xx != 0)
    fr_rows = np.maximum(busy - own, 0.0)
    # own ticks are attributed to each thread's LAST processor: a thread that moved between the run's cores within an
    # interval (the main thread before the OpenMP pool binds it, during the warm-up) leaves a surplus (own > busy) on
    # one run core and a deficit on another.  The run's threads can only run on its own cores (taskset), so the
    # surplus is moved to the run cores' deficits (proportionally); the run cores' total foreign busy per interval
    # (sum busy - sum own) is unchanged.  Siblings and idle cores are never own.
    n_moved = 0
    if is_core.any():
        bc, oc = busy[:, is_core], own[:, is_core]
        surplus = np.maximum(oc - bc, 0.0).sum(axis=1)
        deficit = np.maximum(bc - oc, 0.0)
        dsum = deficit.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            keep = np.where(dsum > 0, 1.0 - np.minimum(1.0, surplus / np.where(dsum > 0, dsum, 1.0)), 1.0)
        fr_rows[:, is_core] = deficit * keep[:, None]
        n_moved = int(np.sum((surplus > 0) & (dsum > 0)))
    fu = fr_rows.sum(axis=0)
    fl = np.where(inc[:, None] & is_core[None, :], 0.0, fr_rows).sum(axis=0)
    total = tot.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, fu / np.where(total > 0, total, 1.0), np.nan)
        frac_l = np.where(total > 0, fl / np.where(total > 0, total, 1.0), np.nan)
    covered = bool(tps.min() <= a + 1e-9 and ts.max() >= b - 1e-9)
    steps = tps[1:] - ts[:-1]
    gap = bool(np.any(steps > GAP_TOL_S))
    overlap = bool(np.any(steps < -1e-6))
    mixed = bool(pid is not None and np.any((rpid >= 0) & (rpid != int(pid))))
    span = float(np.sum(ts - tps))
    ctx = float(np.sum(dv))
    rate = ctx / (max(int(threads), 1) * span) if span > 0 else math.nan
    rate_u = rate if not inc.any() else math.inf
    if scope not in ("own", "ccd"):
        raise ValueError(f"scope must be 'own' or 'ccd' (got {scope!r})")
    sib_of_core = set(sibling(c) for c in cores)
    in_scope = np.array([(c in cores) or (c in sib_of_core) for c in cpus], dtype=bool) if scope == "own" \
        else np.ones(len(cpus), dtype=bool)
    if scope == "own" and not in_scope.any():
        in_scope = np.ones(len(cpus), dtype=bool)        # no run cores recorded: fall back to every monitored cpu
    finite_all = np.isfinite(frac)
    ccd_mf = float(np.max(frac[finite_all])) if finite_all.any() else math.nan
    ccd_mfl = float(np.max(frac_l[finite_all])) if finite_all.any() else math.nan
    finite = finite_all & in_scope
    mf = float(np.max(frac[finite])) if finite.any() else math.nan
    mfl = float(np.max(frac_l[finite])) if finite.any() else math.nan
    worst = int(cpus[int(np.nanargmax(np.where(finite, frac, -1.0)))]) if finite.any() else None
    contended_l = (finite.any() and mfl > busy_thr) or (math.isfinite(rate) and rate > ctx_thr)
    clean_u = finite.any() and mf <= busy_thr and math.isfinite(rate) and rate_u <= ctx_thr
    reasons = []
    if finite.any() and mfl > busy_thr:
        reasons.append(f"foreign busy {mfl:.3f} > {busy_thr} on cpu {int(cpus[int(np.nanargmax(np.where(finite, frac_l, -1.0)))])}")
    elif finite.any() and mf > busy_thr:
        reasons.append(f"foreign busy upper bound {mf:.3f} > {busy_thr} on cpu {worst} (incomplete own accounting)")
    if math.isfinite(rate) and rate > ctx_thr:
        reasons.append(f"{rate:.2f} involuntary switches per thread-second > {ctx_thr}")
    if not finite_all[in_scope].all():
        reasons.append("a cpu has no jiffies in the window")
    if not covered:
        reasons.append("telemetry does not cover the whole window")
    if gap:
        reasons.append(f"telemetry gap between rows (max {float(steps.max()):.2f} s)")
    if overlap:
        reasons.append("overlapping telemetry rows (two writers)")
    if mixed:
        reasons.append(f"telemetry rows of another process (pids {sorted(set(int(p) for p in rpid if p >= 0))}, "
                       f"run pid {int(pid)})")
    if inc.any():
        reasons.append(f"{int(inc.sum())} interval(s) with incomplete own accounting (first sight with unknown history "
                       f"or run process absent): bounds used")
    if mixed or overlap:
        status = "unknown"
    elif contended_l:
        status = "contended"
    elif not covered or gap or not finite_all[in_scope].all() or not math.isfinite(rate) or not clean_u:
        status = "unknown"
    else:
        status = "uncontended"
    fq = np.asarray(tel.get("freq_mhz", np.zeros((S, 0))), float)
    fq = fq[idx] if fq.ndim == 2 and fq.shape[0] == S else np.zeros((0, 0))
    return dict(out, status=status, contended=status != "uncontended", reasons=reasons, scope=scope,
                ccd_max_foreign_busy=ccd_mf, ccd_max_foreign_busy_lower=ccd_mfl, max_foreign_busy=mf,
                max_foreign_busy_lower=mfl, worst_cpu=worst, ctx_per_thread_s=rate, n_intervals=int(idx.size),
                n_incomplete=int(inc.sum()), n_rows_own_moved=n_moved, gap=gap, overlap=overlap, mixed_pid=mixed,
                covered=covered,
                foreign_busy={int(c): (float(v) if np.isfinite(v) else None) for c, v in zip(cpus, frac)},
                mean_freq_mhz=(float(np.mean(fq[fq > 0])) if fq.size and np.any(fq > 0) else math.nan))


# ================================================================================================ processes
def launch_pinned(cmd, cores, log_path=None, env=None, extra_env=None):
    """Start ``cmd`` (argv list) in a fresh process pinned with taskset to ``cores`` and with the thread / affinity
    environment of omp_env(cores).  Returns the Popen (its pid is the run's pid: taskset execs the command)."""
    import subprocess
    e = omp_env(cores, base=env)
    if extra_env:
        e.update({k: str(v) for k, v in extra_env.items()})
    full = ["taskset", "-c", cpu_list_str(cores)] + [str(c) for c in cmd]
    out = open(log_path, "ab") if log_path else subprocess.DEVNULL
    try:
        return subprocess.Popen(full, env=e, stdout=out, stderr=subprocess.STDOUT if log_path else subprocess.DEVNULL,
                                start_new_session=True)
    finally:
        if log_path:
            out.close()


def start_monitor(registry_root=REGISTRY_DIR, allowed=None, interval=MONITOR_INTERVAL_S, log_path=None):
    """Start the telemetry monitor process for this campaign process (it exits when this process exits)."""
    import subprocess
    cmd = [sys.executable, os.path.abspath(__file__), "monitor", "--registry", registry_root, "--interval",
           str(interval), "--parent", str(os.getpid())]
    if allowed is not None:
        cmd += ["--allowed-cpus", cpu_list_str(allowed)]
    out = open(log_path, "ab") if log_path else subprocess.DEVNULL
    try:
        return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT if log_path else subprocess.DEVNULL)
    finally:
        if log_path:
            out.close()


def kill_process_group(proc, grace_s=10.0):
    """Terminate one of OUR run processes (launched with start_new_session, so its process group id is its pid):
    SIGTERM to the group, SIGKILL after ``grace_s``.  Returns the exit code (None if ``proc`` has no process)."""
    import signal
    import subprocess
    if not isinstance(proc, subprocess.Popen):          # only our own child processes are ever signalled
        return None
    if proc.poll() is not None:
        return proc.returncode
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, None)):
        try:
            os.killpg(int(proc.pid), sig)
        except (OSError, ProcessLookupError):
            try:
                proc.send_signal(sig)
            except Exception:  # noqa: BLE001
                pass
        try:
            return proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            continue
    return proc.poll()


def stop_monitor(proc, timeout=5.0):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def finalize_telemetry(jsonl_path, npz_path=None, delete=True):
    """Load a monitor JSONL file, write it as a compressed npz sidecar (when npz_path) and delete the JSONL.
    Returns the telemetry dict (None if absent).  If the JSONL is gone but the sidecar exists (an interrupted earlier
    finalisation), the sidecar is loaded."""
    if not (jsonl_path and os.path.exists(jsonl_path)) and npz_path and os.path.exists(npz_path):
        return load_telemetry_npz(npz_path)
    tel = load_telemetry(jsonl_path)
    if tel is not None and npz_path:
        d = os.path.dirname(os.path.abspath(npz_path))
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, f".{os.path.basename(npz_path)}.tmp{os.getpid()}")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **telemetry_arrays(tel))
        os.replace(tmp, npz_path)
    if delete and jsonl_path and os.path.exists(jsonl_path) and (tel is None or npz_path):
        os.remove(jsonl_path)
    return tel


def load_telemetry_npz(path):
    if not path or not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        return telemetry_from_arrays({k: z[k] for k in z.files})


# ================================================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("monitor", help="run the 1 Hz telemetry monitor (exits with its parent)")
    m.add_argument("--registry", default=REGISTRY_DIR)
    m.add_argument("--interval", type=float, default=MONITOR_INTERVAL_S)
    m.add_argument("--parent", type=int, default=None)
    m.add_argument("--allowed-cpus", default=None, help="cores the monitor may pin itself to (default 0-127)")
    m.add_argument("--max-iter", type=int, default=None)
    s = sub.add_parser("status", help="held CCDs and a 2 s busy sample per CCD")
    s.add_argument("--registry", default=REGISTRY_DIR)
    s.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    reg = Registry(a.registry)
    if a.cmd == "monitor":
        allowed = parse_cpu_list(a.allowed_cpus) if a.allowed_cpus else None
        Monitor(reg, a.interval, allowed=allowed, parent=a.parent,
                log=lambda m: print(m, file=sys.stderr, flush=True)).loop(a.max_iter)
        return 0
    held = reg.held()
    busy = sample_busy()
    best, least, loads = choose_set(a.threads, busy, held)
    print(json.dumps(dict(held={str(k): v.get("run_id") for k, v in held.items()},
                          per_ccd={i: round(ccd_load(busy, i)[0], 3) for i in range(N_CCD)},
                          best=best, least_loaded=least), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
