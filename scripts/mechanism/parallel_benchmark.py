#!/usr/bin/env python
"""Experiment III: wall-clock time-to-accuracy at fixed N = 512 (docs/mechanism/SCIENTIFIC_PLAN.md section 8,
configs/mechanism/parallel_benchmark.json).  BUILT AND SMOKED ONLY: the campaign starts after Experiments I and II
are analysed, with the seed count and backend details fixed in an amendment.

One process = one (system, backend, method, seed) run of the FULL frozen budget at N = 512 (B / N steps):

    # one run (pinned to ONE logical CPU -- refused otherwise; GPU: CUDA_VISIBLE_DEVICES names ONE allowed device)
    CUDA_VISIBLE_DEVICES=2 taskset -c 64 python scripts/mechanism/parallel_benchmark.py run \
        --system gateway --backend gpu_torch --method fr --seed 8100
    taskset -c 65 python scripts/mechanism/parallel_benchmark.py run --system lta300 --backend cpu_numba_1core \
        --method abf --seed 30000
    # the whole campaign (sequential GPU jobs; CPU jobs one per listed core), or its job list and projected cost
    python scripts/mechanism/parallel_benchmark.py campaign --backend cpu_numba_1core --cpus 64-71 --dry-run
    CUDA_VISIBLE_DEVICES=2 python scripts/mechanism/parallel_benchmark.py campaign --backend gpu_torch --cpus 64 \
        --systems lta300 lta150 --max-device-hours 6

Systems (frozen configs): gateway = configs/equal_budget_v2/gateway_production.json (h 2.5e-5, n_steps 6.4e6 at
N 512), lta300 / lta150 = configs/equal_budget_v2/lta_{300,150}K_production.json (h 2e-4, n_steps 6e5).

Backends
  gpu_torch        the existing torch production engines (src/gateway_core.py, src/lta/core_lta.py) driven by the
                   lean single-run loops of scripts/mechanism/bench_torch_engines.py on ONE CUDA device (float64,
                   eager mode).  The step bodies are the engines' own functions in their own order; the bitwise
                   equality with the unmodified engines (CPU float64) is tests/test_parallel_benchmark.py.
  cpu_numba_1core  the validated production numba engines (src/gateway_ladder_numba.py, src/lta_ladder_numba.py)
                   via their own run_arm, single-threaded.  With the production seeds and the full budget a run
                   reproduces the equal-budget production file bitwise (checked here: meta.bitwise_vs_production).
  cpu_numba_threads  NOT AVAILABLE: no walker-parallel numba build exists (plan section 8 option 2) and none is
                   written here; the backend is reported NOT TESTED.
  cpu_torch_test   the gpu_torch drivers on the CPU (torch single thread): harness tests and smoke ONLY, never an
                   Experiment III backend.

Timing (all time.perf_counter; CUDA work is synchronised before every stamp)
  import_s       process start (/proc) to the end of the imports (torch / numba / engines)
  cuda_init_s    gpu_torch: CUDA context creation + first allocation on the device
  warmup_s       one short run of the SAME code path (JIT compile or numba cache load, CUDA kernel / cuBLAS loading,
                 allocator growth): gateway 2 FR intervals, LTA 60 steps with the FR start moved to step 10 (its
                 runtime knobs only); it uses its own generators and touches nothing of the timed run
  wall_s_at_save the timed run's elapsed wall clock at every save of the production save grid (budget grid: 200
                 uniform + 24 log-spaced u, union the frozen physical-time checkpoints <= T, the rule of
                 scripts/equal_budget/run_ladder.save_grid).  numba: the engine's chunk kernel is called through a
                 timing hook that splits each chunk at the save steps (chunking does not change a bit: tested); the
                 gateway stamp follows the save's step exactly; the LTA stamp follows the sub-chunk that evaluates
                 step k, i.e. it also includes step k's move (<= 1 step late).  Chunk-level random draws (numba noise /
                 FR blocks) are made before the chunk, so the stamps also include them.
  wall_total_s   start of the timed run to its end (excludes import, CUDA init and warm-up)
  fr_time_s_at_save  device time inside the FR blocks (gpu_torch: CUDA events per opportunity; cpu_torch_test:
                 perf_counter); NaN for numba (no clock inside the nopython kernel: FR's added time is the paired
                 wall difference FR - ABF, computed by analyze_parallel_benchmark.py)
  monitor_s      gpu_torch: time spent sampling nvidia-smi at saves (at most every GPU_POLL_S s); the device is
                 idle there (synchronised) and this time is EXCLUDED from wall_s_at_save and wall_total_s
Memory
  peak_gpu_device_mb  gpu_torch, THE device footprint: this process's nvidia-smi used_memory (CUDA context, library
                 workspaces, torch's caching allocator), the maximum over the samples during the timed run and one
                 right after it.  torch's caching allocator returns no block to the driver without empty_cache(), so
                 the end-of-run sample includes the run's maximum reservation (gpu_reserved_end_mb ==
                 peak_gpu_reserved_mb is recorded as the check)
  peak_gpu_reserved_mb / peak_gpu_alloc_mb  torch.cuda.max_memory_reserved / max_memory_allocated of the timed run
                 (stats reset after the warm-up): TENSOR memory only, a secondary figure
  peak_rss_mb    host VmHWM.  gpu_torch / cpu_torch_test: reset (/proc/self/clear_refs) right before the timed run
                 (after the torch import, CUDA context creation and warm-up), read right after it; numba: the
                 engine's own per-run peak_rss_mb (VmHWM reset at run_arm entry, i.e. right before the timed run)
Contention telemetry (window = the timed run: begin right before t0, end right after the final device sync):
  affinity, load average, busy fraction of the pinned logical CPU and of its SMT siblings (/proc/stat); a timed run
  (gpu_torch, cpu_numba_1core) must be pinned to EXACTLY ONE logical CPU (refused otherwise unless --allow-unpinned,
  which the analysis then reports as contention 'unknown'); on the GPU, the other compute processes on the device
  before, during (every monitor sample) and after the run.  GPU guard: the physical index must be one of the allowed
  benchmark GPUs (benchmark config 'allowed_gpus' if an amendment sets it, else DEFAULT_ALLOWED_GPUS = [2] from plan
  section 8; never overridable), and a run refuses a device that holds another process OR whose occupancy cannot be
  read (nvidia-smi failing) unless --allow-busy-gpu, in which case 'unknown' / the processes are recorded.
  A busy SMT sibling makes a run 1.4-1.7x slower (measured): the campaign driver uses one logical CPU per physical
  core, launches a job only on a listed CPU that is idle together with its SMT sibling (1 s /proc/stat sample), waits
  up to --max-wait-s for one, and then STOPS (no contended launch) unless --allow-contended.  The analysis classifies
  every run as SMT-contended (sibling busy > 10 % of the timed run), uncontended, or unknown, and computes W1 on the
  uncontended pairs (analyze_parallel_benchmark.W1_RULE).
  NOTE (2026-10-10): on this host CPUs c and c + 128 are SMT siblings, so any job on 128-255 contends with 0-127.
Resource guard: the campaign stops before the allocated device-hours of this backend under out_root (finished runs'
process_wall_s, + the projection of running jobs) plus the projection of the next job and its not-yet-run partner
arm would exceed --max-device-hours (default: the plan section 9 GPU ceiling 8 GPU-h for gpu_torch; the plan's
Experiment III CPU estimate 20 core-h otherwise -- the 400 core-h ceiling is shared with Experiments I / II, whose use
this harness cannot see).  Projection: measured us/step of finished runs of the same (system, method), else
FALLBACK_US_PER_STEP (2026-10-10 measurements, host sibling busy: conservative).

Outputs  <out_root>/<system>/<backend>/s<seed>_<method>.npz (+ .json, the same meta and scalars without arrays):
  save_step, save_t, save_u, wall_s_at_save, fr_time_s_at_save, n_force_evals_at_save (gateway N k, LTA N (k + 1)),
  M_all, C_all (S, 180) [+ M_prod, C_prod for LTA] -- exactly what scripts/equal_budget/eqb_metrics' scorers read,
  FR counters (gateway: fr_kd_cum, fr_kc_cum candidates, + fr_deaths_cum realised for numba; LTA cum_deaths),
  scalars wall_total_s, warmup_s, import_s, cuda_init_s, peak_*, monitor_s; meta_json (signature, knobs, provenance,
  telemetry, process_wall_s = process start to the end of the run, the allocated device time).
A complete result with the same signature is skipped; one with a different signature is never overwritten.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_mech"))

import numpy as np  # noqa: E402

HARNESS_VERSION = "parallel_benchmark/2"
BENCH_CONFIG = os.path.join(ROOT, "configs", "mechanism", "parallel_benchmark.json")
SYSTEMS = {
    "gateway": dict(kind="gateway", out_dir="gateway"),
    "lta300": dict(kind="lta", out_dir="lta_T300", T_K=300.0),
    "lta150": dict(kind="lta", out_dir="lta_T150", T_K=150.0),
}
BACKENDS = ("gpu_torch", "cpu_numba_1core")            # Experiment III backends implemented here
UNAVAILABLE_BACKENDS = {"cpu_numba_threads": "no walker-parallel numba build exists (none written): NOT TESTED"}
TEST_BACKENDS = ("cpu_torch_test",)                    # harness tests / smoke only
METHODS = ("abf", "fr")
DEFAULT_SEEDS_PER_ARM = 8                              # plan: '>= 8, TBD in the amendment'
EXP_ROOT = os.path.join(ROOT, "results", "mechanism", "parallel_benchmark")
TORCH_WARM = dict(gateway_intervals=2, lta_steps=60, lta_fr_start=10)
DEFAULT_ALLOWED_GPUS = (2,)          # plan section 8 ("H200 #2"); GPUs 0, 1, 3 carry other users' jobs
GPU_POLL_S = 60.0                    # nvidia-smi sample interval during a gpu_torch run (excluded from the clock)
PLAN_CEILING_H = {"gpu_torch": 8.0, "cpu": 20.0}     # plan section 9: 8 GPU-h ceiling; <= 20 core-h Exp III estimate
# per-step wall at N = 512 measured 2026-10-10 with the host SMT sibling busy (the conservative state):
# bench_step_costs (lean torch, GPU 2) and full-budget numba runs; max over the two arms
FALLBACK_US_PER_STEP = {
    ("gpu_torch", "gateway"): 590.0, ("gpu_torch", "lta300"): 1150.0, ("gpu_torch", "lta150"): 1150.0,
    ("cpu_numba_1core", "gateway"): 66.0, ("cpu_numba_1core", "lta300"): 630.0, ("cpu_numba_1core", "lta150"): 630.0,
    ("cpu_torch_test", "gateway"): 3000.0, ("cpu_torch_test", "lta300"): 45000.0, ("cpu_torch_test", "lta150"): 45000.0,
}
RUN_OVERHEAD_FALLBACK_S = 20.0       # process start + imports + CUDA init + warm-up + output (measured 3-10 s)
SMT_BUSY_FLAG = 0.10                 # sibling busy fraction above which a run is SMT-contended


class BenchError(RuntimeError):
    pass


# ================================================================================================ configuration
def bench_config(path=None):
    with open(path or BENCH_CONFIG) as fh:
        return json.load(fh)


def system_spec(system, bench=None):
    """(production config dict P, N, mid threshold) of a system, cross-checked against the frozen configs."""
    if system not in SYSTEMS:
        raise BenchError(f"unknown system {system!r}; expected one of {sorted(SYSTEMS)}")
    bench = bench or bench_config()
    ent = [s for s in bench["systems"] if s["name"] == system]
    if len(ent) != 1:
        raise BenchError(f"{system} is not listed once in {BENCH_CONFIG}")
    ent = ent[0]
    with open(os.path.join(ROOT, ent["config"])) as fh:
        P = json.load(fh)
    N = int(ent["N"])
    if int(P["B"]) % N:
        raise BenchError(f"B {P['B']} not divisible by N {N}")
    thr = float(ent["threshold_e_F"])
    if not math.isclose(thr, float(P["thresholds"]["e_F"][1]), rel_tol=0, abs_tol=1e-15):
        raise BenchError(f"{system}: benchmark mid threshold {thr} != production e_F mid {P['thresholds']['e_F'][1]}")
    if SYSTEMS[system]["kind"] == "lta" and float(P["T_K"]) != SYSTEMS[system]["T_K"]:
        raise BenchError(f"{system}: config T_K {P['T_K']}")
    P = dict(P)
    P["config_path"] = ent["config"]
    return P, N, thr


def full_n_steps(P, N):
    return int(P["B"]) // int(N)


def save_grid(P, n_steps, kind):
    """scripts/equal_budget/run_ladder.save_grid: budget grid (200 uniform + 24 log-spaced u) union the frozen
    physical-time checkpoints t <= T at the nearest integration step (re-implemented here because run_ladder blanks
    CUDA_VISIBLE_DEVICES at import; equality with it is tested)."""
    u_lin = np.arange(1, 201) / 200.0
    u_log = np.logspace(np.log10(1e-4), np.log10(1.0 / 200), 24, endpoint=False)
    u = np.unique(np.concatenate([u_log, u_lin]))
    steps = np.unique(np.clip(np.round(u * n_steps).astype(np.int64), 1, n_steps))
    h = float(P["engine_cfg"]["h"])
    extra = [int(round(t / h)) for t in P["physical_checkpoints_t"] if 0 < round(t / h) <= n_steps]
    return np.unique(np.concatenate([steps, np.asarray(extra, dtype=np.int64)]))


def production_path(system, seed, method):
    return os.path.join(ROOT, "results", "equal_budget_v2", SYSTEMS[system]["out_dir"], "N512", f"s{seed}_{method}.npz")


def out_paths(out_root, system, backend, seed, method):
    d = os.path.join(out_root, system, backend)
    return os.path.join(d, f"s{seed}_{method}.npz"), os.path.join(d, f"s{seed}_{method}.json")


# ================================================================================================ telemetry
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def provenance(files):
    out = {}
    for f in files:
        p = os.path.join(ROOT, f)
        out[f] = dict(sha256=sha256_file(p) if os.path.exists(p) else None,
                      git_dirty=bool(_git("status", "--porcelain", "--", f)))
    return dict(git_commit=_git("rev-parse", "HEAD") or "unknown", files=out)


def proc_status_mb(field):
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def reset_peak_rss():
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


def proc_start_epoch():
    try:
        with open("/proc/self/stat") as fh:
            s = fh.read()
        ticks = int(s[s.rindex(")") + 2:].split()[19])
        with open("/proc/stat") as fh:
            btime = next(int(line.split()[1]) for line in fh if line.startswith("btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001
        return None


def smt_siblings(cpus):
    out = set()
    for c in cpus:
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as fh:
                txt = fh.read().strip()
            for part in txt.split(","):
                a, _, b = part.partition("-")
                out.update(range(int(a), int(b or a) + 1))
        except OSError:
            pass
    return sorted(out - set(cpus))


def cpu_times(cpus):
    """{cpu: (busy, total)} jiffies from /proc/stat."""
    out = {}
    try:
        with open("/proc/stat") as fh:
            for line in fh:
                if line.startswith("cpu") and line[3].isdigit():
                    f = line.split()
                    c = int(f[0][3:])
                    if c in cpus:
                        v = [int(x) for x in f[1:]]
                        idle = v[3] + (v[4] if len(v) > 4 else 0)
                        out[c] = (sum(v) - idle, sum(v))
    except OSError:
        pass
    return out


def busy_fraction(a, b):
    return {str(c): ((b[c][0] - a[c][0]) / max(b[c][1] - a[c][1], 1)) for c in a if c in b}


def idle_cpus(cpus, sample_s=1.0, threshold=0.10):
    """The logical CPUs of ``cpus`` that were idle (busy < threshold) TOGETHER WITH their SMT siblings over a
    ``sample_s`` window of /proc/stat, and the measured busy fractions {cpu: fraction}."""
    cpus = [int(c) for c in cpus]
    allc = set(cpus) | set(smt_siblings(cpus))
    a = cpu_times(allc)
    time.sleep(float(sample_s))
    b = cpu_times(allc)
    bf = {int(k): float(v) for k, v in busy_fraction(a, b).items()}
    ok = [c for c in cpus if bf.get(c, 1.0) < threshold and all(bf.get(x, 1.0) < threshold for x in smt_siblings([c]))]
    return ok, bf


def least_contended(free, bf):
    return min(free, key=lambda x: bf.get(x, 1.0) + max([bf.get(y, 1.0) for y in smt_siblings([x])] or [0.0]))


def choose_cpu(free, idle_check=True, max_wait_s=1800.0, poll_s=30.0, threshold=0.10, log=print,
               allow_contended=False):
    """A CPU of ``free`` that is idle with its SMT sibling (idle_cpus), waiting up to max_wait_s for one.  After
    that: with allow_contended, the free CPU with the least-busy sibling and a warning (the run's telemetry flags it,
    the analysis excludes it from W1); otherwise (None, False) = do not launch.  Returns (cpu, idle: bool)."""
    if not idle_check:
        return free[0], True
    t0 = time.time()
    while True:
        ok, bf = idle_cpus(free, threshold=threshold)
        if ok:
            return ok[0], True
        if time.time() - t0 >= max_wait_s:
            c = least_contended(free, bf)
            sib = {y: round(bf.get(y, float('nan')), 2) for y in smt_siblings([c])}
            if not allow_contended:
                log(f"  STOP: no idle physical core among {free} after {max_wait_s:.0f} s (least busy: cpu {c}, "
                    f"siblings {sib}); not launching a contended run (pass --allow-contended to accept it)")
                return None, False
            log(f"  WARNING: no idle physical core among {free} after {max_wait_s:.0f} s; launching on cpu {c} "
                f"(busy {bf.get(c, float('nan')):.2f}, siblings {sib}) with --allow-contended: the run will be "
                f"SMT-contended and excluded from W1")
            return c, False
        time.sleep(poll_s)


def gpu_processes():
    """Compute processes per physical GPU index (nvidia-smi), or None if the occupancy cannot be read (nvidia-smi
    missing, failing, timing out or listing no GPU): None means UNKNOWN, never 'free'."""
    try:
        r1 = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=30)
        r2 = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    if r1.returncode != 0 or r2.returncode != 0:
        return None
    idx = r1.stdout.strip().splitlines()
    apps = r2.stdout.strip().splitlines()
    uu = {}
    names = {}
    for line in idx:
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3:
            uu[p[1]] = int(p[0])
            names[int(p[0])] = p[2]
    if not names:
        return None
    out = {i: [] for i in names}
    for line in apps:
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3 and p[1] in uu:
            try:
                out[uu[p[1]]].append(dict(pid=int(p[0]), used_memory=p[2]))
            except ValueError:
                return None
    return dict(processes=out, names=names)


def parse_mib(v):
    """'624 MiB' -> 624.0 (nvidia-smi used_memory); NaN if unparseable."""
    try:
        num, _, unit = str(v).strip().partition(" ")
        f = float(num)
        return f * {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}.get(unit.strip() or "MiB", math.nan)
    except ValueError:
        return math.nan


def allowed_gpus(bench=None):
    """Physical GPU indices a benchmark run may use: the benchmark config's 'allowed_gpus' (an amendable field; absent
    in the frozen config) else DEFAULT_ALLOWED_GPUS."""
    bench = bench or bench_config()
    return [int(x) for x in bench.get("allowed_gpus", DEFAULT_ALLOWED_GPUS)]


def gpu_guard(allow_busy=False, pid=None):
    """The physical GPU of this process (CUDA_VISIBLE_DEVICES = exactly one index) after the occupancy checks:
    the index must be allowed (never overridable); a device that holds another compute process, or whose
    occupancy cannot be read, is refused unless allow_busy (the processes / None = unknown are returned and recorded).
    Returns (index, processes on it before the run or None)."""
    idx = physical_gpu()
    if idx is None:
        raise BenchError("gpu_torch needs CUDA_VISIBLE_DEVICES set to exactly ONE physical GPU index")
    allowed = allowed_gpus()
    if idx not in allowed:
        raise BenchError(f"GPU {idx} is not an allowed benchmark device {allowed} (plan section 8; other users' jobs "
                         f"run on the others); refusing")
    gp = gpu_processes()
    pid = os.getpid() if pid is None else pid
    if gp is None or idx not in gp["processes"]:
        if not allow_busy:
            raise BenchError(f"occupancy of GPU {idx} cannot be read (nvidia-smi failed or lists no such GPU); "
                             f"refusing (--allow-busy-gpu runs anyway and records it as unknown)")
        return idx, None
    procs = gp["processes"][idx]
    others = [p for p in procs if p["pid"] != pid]
    if others and not allow_busy:
        raise BenchError(f"GPU {idx} already runs {others}; refusing (pass --allow-busy-gpu to override)")
    return idx, procs


class GPUMonitor:
    """nvidia-smi samples of the benchmark GPU during a timed run: this process's device memory and the other
    compute processes.  sample() is called at the saves (device synchronised, so idle) and samples at most every
    poll_s seconds of run wall clock; the caller excludes its duration from the clock."""

    def __init__(self, gpu_idx, poll_s=GPU_POLL_S, pid=None):
        self.idx, self.poll_s = int(gpu_idx), float(poll_s)
        self.pid = os.getpid() if pid is None else int(pid)
        self.samples = []
        self._last = -math.inf

    def sample(self, wall_s, force=False):
        if not force and wall_s - self._last < self.poll_s:
            return
        self._last = wall_s
        w = float(wall_s) if math.isfinite(wall_s) else None          # None = the sample right after the run
        gp = gpu_processes()
        if gp is None or self.idx not in gp["processes"]:
            self.samples.append(dict(wall_s=w, ok=False))
            return
        procs = gp["processes"][self.idx]
        own = [parse_mib(p["used_memory"]) for p in procs if p["pid"] == self.pid]
        self.samples.append(dict(wall_s=w, ok=True, own_mb=(max(own) if own else None),
                                 others=[p for p in procs if p["pid"] != self.pid], processes=procs))

    def summary(self):
        ok = [x for x in self.samples if x["ok"]]
        own = [x["own_mb"] for x in ok if x["own_mb"] is not None]
        others = {}
        for x in ok:
            for p in x["others"]:
                others.setdefault(p["pid"], p)
        return dict(poll_s=self.poll_s, n_samples=len(self.samples), n_failed=len(self.samples) - len(ok),
                    peak_device_mb=(max(own) if own else math.nan),
                    other_processes=list(others.values()),
                    occupancy=("unknown" if not ok or len(ok) < len(self.samples) else
                               ("shared" if others else "exclusive")),
                    last_processes=(ok[-1]["processes"] if ok and self.samples[-1]["ok"] else None),
                    samples=[{k: v for k, v in x.items() if k != "processes"} for x in self.samples])


class RunTelemetry:
    """CPU busy fractions of the pinned CPU(s) and their SMT siblings, and (torch backends) the host VmHWM reset,
    bracketing exactly the timed run: begin() right before t0, end() right after the final device sync."""

    def __init__(self, cpus, reset_rss):
        self.cpus = set(int(c) for c in cpus)
        self.reset_rss = bool(reset_rss)
        self.c0 = self.c1 = None
        self.rss_reset_ok = None

    def begin(self):
        if self.reset_rss:
            self.rss_reset_ok = reset_peak_rss()
        self.c0 = cpu_times(self.cpus)

    def end(self):
        self.c1 = cpu_times(self.cpus)

    def busy(self):
        return busy_fraction(self.c0, self.c1) if (self.c0 and self.c1) else {}


def smt_state(affinity, siblings, busy, flag=SMT_BUSY_FLAG):
    """'contended' | 'uncontended' | 'unknown' from a run's telemetry: unknown when the run was not pinned to exactly
    one logical CPU or the sibling busy fractions are missing."""
    if not affinity or len(affinity) != 1:
        return "unknown", math.nan
    sib = [str(c) for c in (siblings or [])]
    if not sib:
        return "uncontended", 0.0            # pinned, and the core has no SMT sibling
    vals = [float(busy[c]) for c in sib if busy and c in busy]
    if len(vals) < len(sib):
        return "unknown", (max(vals) if vals else math.nan)
    m = max(vals)
    return ("contended" if m > flag else "uncontended"), m


def physical_gpu():
    v = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if v is None or v.strip() == "" or "," in v:
        return None
    try:
        return int(v.strip())
    except ValueError:
        return None


# ================================================================================================ numba timing hooks
class SaveClock:
    """Wall-clock stamps at the save steps, relative to t0 (set right before the timed run)."""

    def __init__(self, save_steps):
        self.steps = np.asarray(save_steps, dtype=np.int64)
        self.index = {int(k): i for i, k in enumerate(self.steps)}
        self.wall = np.full(self.steps.size, np.nan)
        self.t0 = None

    def start(self):
        self.t0 = time.perf_counter()

    def stamp(self, k):
        i = self.index.get(int(k))
        if i is not None and self.t0 is not None:
            self.wall[i] = time.perf_counter() - self.t0


@contextlib.contextmanager
def gateway_numba_clock(E, clock):
    """Run gateway_ladder_numba.run_arm with its chunk kernel ``_advance`` split at the save steps and stamped.
    A save at completed-step count k fires at the end of step k - 1, i.e. at the end of a sub-call [., k).  FR
    uniforms: opportunity j of a call reads block j of ubuf, so a sub-call starting at s reads ubuf from block
    (#FR steps in [step0, s)).  The kernel state lives in the arrays passed (mutated in place) and in numba's MT
    stream, which continues across calls in this thread; tests prove bitwise equality with the plain run."""
    orig = E._advance
    saves = clock.steps

    def hooked(step0, step1, N, h, beta, H, oout, oin, s, nb, min_count, is_fr, gamma, ramp_steps, fr_every,
               score_clip, cap, kern, r_eta, dx, ubuf, ublk, fr_internal, *rest):
        ks = saves[(saves > step0) & (saves <= step1)]
        cur = int(step0)
        uses_blocks = bool(is_fr) and int(N) >= 2 and not fr_internal
        fe = int(fr_every)
        off = 0
        for k in ks.tolist():
            if k > cur:
                orig(cur, k, N, h, beta, H, oout, oin, s, nb, min_count, is_fr, gamma, ramp_steps, fr_every,
                     score_clip, cap, kern, r_eta, dx, ubuf[off * ublk:] if uses_blocks else ubuf, ublk, fr_internal,
                     *rest)
                if uses_blocks:
                    off += (-(-k // fe)) - (-(-cur // fe))
                cur = k
            clock.stamp(k)
        if cur < step1:
            orig(cur, step1, N, h, beta, H, oout, oin, s, nb, min_count, is_fr, gamma, ramp_steps, fr_every,
                 score_clip, cap, kern, r_eta, dx, ubuf[off * ublk:] if uses_blocks else ubuf, ublk, fr_internal, *rest)

    E._advance = hooked
    try:
        yield
    finally:
        E._advance = orig


@contextlib.contextmanager
def lta_numba_clock(E, clock):
    """Run lta_ladder_numba.run_arm with its chunk kernel ``_run_chunk`` split at the save steps and stamped.  A
    sub-chunk ending at evaluated step k (which writes save k) makes k's move unless k = n_steps; noise rows are
    consumed one per move and FR blocks one per opportunity (returned by the kernel), so the remainder continues
    from the consumed counts."""
    orig = E._run_chunk
    saves = clock.steps

    def hooked(n_eval, fp, ip, Ox, Oy, Oz, Kk, save_steps, trace_steps, noise, frb, rp_u, rp_perm, rp_src, q, anc,
               ancw, wst, acc, ist, *rest):
        step0 = int(ist[E.S_STEP])
        last = step0 + int(n_eval) - 1
        ks = saves[(saves >= step0) & (saves <= last)]
        cur, mi, oi = step0, 0, 0
        for k in ks.tolist():
            im, io = orig(k - cur + 1, fp, ip, Ox, Oy, Oz, Kk, save_steps, trace_steps, noise[mi:], frb[oi:], rp_u,
                          rp_perm, rp_src, q, anc, ancw, wst, acc, ist, *rest)
            mi += int(im)
            oi += int(io)
            clock.stamp(k)
            cur = k + 1
        if cur <= last:
            im, io = orig(last - cur + 1, fp, ip, Ox, Oy, Oz, Kk, save_steps, trace_steps, noise[mi:], frb[oi:], rp_u,
                          rp_perm, rp_src, q, anc, ancw, wst, acc, ist, *rest)
            mi += int(im)
            oi += int(io)
        return mi, oi

    E._run_chunk = hooked
    try:
        yield
    finally:
        E._run_chunk = orig


# ================================================================================================ backends
def _numba_gateway(P, method, N, seed, n_steps, saves, diagnostics, warm, tel=None):
    t = time.perf_counter()
    import gateway_ladder_numba as E
    out = dict(engine_import_s=time.perf_counter() - t)
    cfg = dict(P["engine_cfg"])
    if warm:
        kn = E.derived_knobs(E.full_cfg(cfg), N)
        nw = 2 * int(kn["fr_every"])
        wclock = SaveClock(np.array([nw // 2, nw]))
        t = time.perf_counter()
        with gateway_numba_clock(E, wclock):
            wclock.start()
            E.run_arm(cfg, method, N, seed, nw, np.array([nw // 2, nw]), diagnostics=diagnostics)
        out["warmup_s"] = time.perf_counter() - t
        out["warmup_steps"] = nw
    clock = SaveClock(saves)
    with gateway_numba_clock(E, clock):
        if tel is not None:
            tel.begin()
        clock.start()
        res = E.run_arm(cfg, method, N, seed, n_steps, saves, diagnostics=diagnostics)
        wall_total = time.perf_counter() - clock.t0
        if tel is not None:
            tel.end()
    k = np.asarray(res["save_step"], dtype=np.int64)
    out.update(save_step=k, save_t=np.asarray(res["save_t"], float), save_u=np.asarray(res["save_u"], float),
               wall_s_at_save=clock.wall, fr_time_s_at_save=np.full(k.size, np.nan), n_force_evals_at_save=N * k,
               M_all=np.asarray(res["M_all"], float), C_all=np.asarray(res["C_all"], float),
               fr_kd_cum=np.asarray(res["fr_kd_cum"], np.int64), fr_kc_cum=np.asarray(res["fr_kc_cum"], np.int64),
               fr_deaths_cum=np.asarray(res["fr_deaths_cum"], np.int64),
               X_final=np.asarray(res["X_final"], float), Y_final=np.asarray(res["Y_final"], float),
               n_fr_opps=int(res["n_fr_steps"]), wall_total_s=wall_total, engine_wall_s=float(res["wall_s"]),
               n_force_evals=int(res["n_force_evals"]), peak_rss_mb=float(res["peak_rss_mb"]))
    out["engine_meta"] = dict(engine=res["meta"]["engine"], engine_sha256=res["meta"]["engine_sha256"],
                              cap=res["meta"]["cap"], fr_every=res["meta"]["fr_every"],
                              ramp_steps=res["meta"]["ramp_steps"], noise_seed=res["meta"]["noise_seed"],
                              chunk_steps=res["meta"]["chunk_steps"], diagnostics=res["meta"]["diagnostics"],
                              peak_rss_source=res["meta"]["peak_rss_source"])
    return out


def _numba_lta(P, T_K, method, N, seed, n_steps, saves, diagnostics, warm, tel=None):
    t = time.perf_counter()
    import lta_ladder_numba as E
    out = dict(engine_import_s=time.perf_counter() - t)
    ecfg = dict(P["engine_cfg"])
    ecfg["diagnostics"] = bool(diagnostics)
    if warm:
        wc = dict(ecfg, n_steps=TORCH_WARM["lta_steps"], fr_start_steps=TORCH_WARM["lta_fr_start"],
                  warmup_steps=TORCH_WARM["lta_fr_start"], burn_in_steps=TORCH_WARM["lta_fr_start"])
        cw = E.make_cfg(T_K, N, seed, **wc)
        ws = np.array([TORCH_WARM["lta_steps"] // 2, TORCH_WARM["lta_steps"]])
        wclock = SaveClock(ws)
        t = time.perf_counter()
        with lta_numba_clock(E, wclock):
            wclock.start()
            E.run_arm(cw, method, ws)
        out["warmup_s"] = time.perf_counter() - t
        out["warmup_steps"] = TORCH_WARM["lta_steps"]
    c = E.make_cfg(T_K, N, seed, **dict(ecfg, n_steps=int(n_steps)))
    clock = SaveClock(saves)
    with lta_numba_clock(E, clock):
        if tel is not None:
            tel.begin()
        clock.start()
        res = E.run_arm(c, method, saves)
        wall_total = time.perf_counter() - clock.t0
        if tel is not None:
            tel.end()
    meta = json.loads(str(res["meta_json"]))
    k = np.asarray(res["save_step"], dtype=np.int64)
    out.update(save_step=k, save_t=np.asarray(res["save_t"], float), save_u=np.asarray(res["save_u"], float),
               wall_s_at_save=clock.wall, fr_time_s_at_save=np.full(k.size, np.nan),
               n_force_evals_at_save=N * (k + 1),
               M_all=np.asarray(res["M_all"], float), C_all=np.asarray(res["C_all"], float),
               M_prod=np.asarray(res["M_prod"], float), C_prod=np.asarray(res["C_prod"], float),
               cum_deaths=np.asarray(res["cum_deaths"], np.int64), q_final=np.asarray(res["q_final"], float),
               n_fr_opps=int(np.asarray(res["cum_opps"])[-1]), wall_total_s=wall_total,
               engine_wall_s=float(res["wall_s"]), n_force_evals=int(res["n_force_evals"]),
               peak_rss_mb=float(res["peak_rss_mb"]))
    out["engine_meta"] = dict(engine=meta.get("engine"), engine_sha256=meta.get("engine_sha256"), cap=meta.get("cap"),
                              compile_target=meta.get("compile_target"), diagnostics=bool(diagnostics),
                              peak_rss_method=meta.get("peak_rss_method"))
    return out


def _torch_run(kind, P, T_K, method, N, seed, n_steps, saves, device, fr_timing, warm, tel=None, monitor=None):
    t_imp = time.perf_counter()
    import torch
    import bench_torch_engines as BT
    out = dict(engine_import_s=time.perf_counter() - t_imp)
    torch.set_num_threads(1)
    dev = torch.device(device)
    if dev.type == "cuda":
        t = time.perf_counter()
        torch.cuda.init()
        torch.zeros(1, device=dev)
        torch.cuda.synchronize(dev)
        out["cuda_init_s"] = time.perf_counter() - t
        out["device_name"] = torch.cuda.get_device_name(dev)
    cfg = dict(P["engine_cfg"])

    def make(nst, sv, warm_run=False):
        if kind == "gateway":
            return BT.GatewayTorchRun(cfg, method, N, seed, nst, sv, dev, fr_timing=fr_timing)
        c = dict(cfg)
        if warm_run:
            c.update(fr_start_steps=TORCH_WARM["lta_fr_start"], warmup_steps=TORCH_WARM["lta_fr_start"],
                     burn_in_steps=TORCH_WARM["lta_fr_start"])
        return BT.LTATorchRun(c, T_K, method, N, seed, nst, sv, dev, fr_timing=fr_timing)

    if warm:
        if kind == "gateway":
            nw = TORCH_WARM["gateway_intervals"] * BT.gateway_knobs(cfg, N)["fr_every"]
        else:
            nw = TORCH_WARM["lta_steps"]
        t = time.perf_counter()
        make(nw, np.array([nw // 2, nw]), warm_run=True).run()
        out["warmup_s"] = time.perf_counter() - t
        out["warmup_steps"] = nw
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    drv = make(n_steps, saves)
    res = drv.run(on_start=(tel.begin if tel is not None else None),
                  on_save=(monitor.sample if monitor is not None else None),
                  on_end=(tel.end if tel is not None else None))
    if tel is not None and tel.reset_rss:
        out["peak_rss_mb"] = proc_status_mb("VmHWM") if tel.rss_reset_ok else math.nan
    if dev.type == "cuda":
        out["peak_gpu_alloc_mb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20
        out["peak_gpu_reserved_mb"] = torch.cuda.max_memory_reserved(dev) / 2 ** 20
        out["gpu_reserved_end_mb"] = torch.cuda.memory_reserved(dev) / 2 ** 20
        if monitor is not None:
            monitor.sample(math.inf, force=True)        # after the run: the caching allocator still holds its peak
    out.update(res)
    out["engine_meta"] = drv.meta()
    out["torch_version"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    out["torch_deterministic"] = bool(torch.are_deterministic_algorithms_enabled())
    return out


# ================================================================================================ one run
def run_signature(system, backend, method, seed, N, n_steps, P, diagnostics, device):
    sig = dict(harness=HARNESS_VERSION, system=system, backend=backend, method=method, seed=int(seed), N=int(N),
               n_steps=int(n_steps), config=P["config_path"],
               engine_cfg_sha=hashlib.sha256(json.dumps(P["engine_cfg"], sort_keys=True).encode()).hexdigest()[:16],
               diagnostics=bool(diagnostics) if backend == "cpu_numba_1core" else False,
               device_type=("cuda" if backend == "gpu_torch" else "cpu"))
    return sig


def existing_status(npz, sig):
    """'absent' | 'complete' (same signature) | 'different' (a complete result of another run)."""
    js = os.path.splitext(npz)[0] + ".json"
    if not (os.path.exists(npz) and os.path.exists(js)):
        return "absent"
    try:
        with open(js) as fh:
            m = json.load(fh)
    except Exception:  # noqa: BLE001
        return "different"
    if m.get("status") != "complete":
        return "absent"
    return "complete" if m.get("signature") == sig else "different"


def atomic_savez(path, **arrays):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "." + os.path.basename(path) + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_json(path, obj):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_json_default)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def bitwise_vs_production(system, seed, method, out):
    """Compare the accumulators of a full-budget numba run with the equal-budget production file of the same seed."""
    p = production_path(system, seed, method)
    if not os.path.exists(p):
        return dict(status="no production file", path=os.path.relpath(p, ROOT))
    with np.load(p, allow_pickle=False) as z:
        keys = ["save_step", "M_all", "C_all"] + (["M_prod", "C_prod"] if "M_prod" in out else [])
        same = {k: bool(k in z.files and np.array_equal(np.asarray(z[k]), np.asarray(out[k]))) for k in keys}
    return dict(status="equal" if all(same.values()) else "DIFFERENT", per_key=same, path=os.path.relpath(p, ROOT))


def run_one(system, backend, method, seed, out_root=EXP_ROOT, n_steps=None, device=None, diagnostics=False,
            fr_timing=True, warm=True, allow_busy_gpu=False, force=False, allow_unpinned=False, gpu_poll_s=GPU_POLL_S):
    """Run one (system, backend, method, seed) and write its npz + json.  Returns the json dict."""
    entry_epoch = time.time()
    if backend in UNAVAILABLE_BACKENDS:
        raise BenchError(f"{backend}: {UNAVAILABLE_BACKENDS[backend]}")
    if backend not in BACKENDS + TEST_BACKENDS:
        raise BenchError(f"unknown backend {backend!r}")
    if method not in METHODS:
        raise BenchError(f"unknown method {method!r}")
    P, N, thr = system_spec(system)
    kind = SYSTEMS[system]["kind"]
    T_K = float(P.get("T_K", 0.0)) if kind == "lta" else None
    n_full = full_n_steps(P, N)
    n_steps = n_full if n_steps is None else int(n_steps)
    if not (0 < n_steps <= n_full):
        raise BenchError(f"n_steps {n_steps} outside (0, {n_full}]")
    saves = save_grid(P, n_steps, kind)
    npz, js = out_paths(out_root, system, backend, seed, method)
    sig = run_signature(system, backend, method, seed, N, n_steps, P, diagnostics, device)
    st = existing_status(npz, sig)
    if st == "complete" and not force:
        with open(js) as fh:
            return json.load(fh)
    if st == "different":
        raise BenchError(f"{npz} holds a result of a DIFFERENT run; refusing to overwrite it")

    affinity = sorted(os.sched_getaffinity(0))
    pinned = len(affinity) == 1
    if backend in BACKENDS and not pinned and not allow_unpinned:
        raise BenchError(f"a timed {backend} run must be pinned to exactly ONE logical CPU (taskset -c <cpu>); "
                         f"affinity is {len(affinity)} CPUs (pass --allow-unpinned to run anyway: contention 'unknown')")
    gpu_idx, gpu_before, monitor = None, None, None
    if backend == "gpu_torch":
        gpu_idx, gpu_before = gpu_guard(allow_busy=allow_busy_gpu)
        device = device or "cuda:0"
        monitor = GPUMonitor(gpu_idx, poll_s=gpu_poll_s)
    elif backend == "cpu_torch_test":
        device = "cpu"
    sib = smt_siblings(affinity)
    load0 = os.getloadavg()
    start_epoch = proc_start_epoch()
    import_s = (entry_epoch - start_epoch) if start_epoch else math.nan

    tel = RunTelemetry(set(affinity) | set(sib), reset_rss=(backend != "cpu_numba_1core"))
    if backend == "cpu_numba_1core":
        if kind == "gateway":
            out = _numba_gateway(P, method, N, seed, n_steps, saves, diagnostics, warm, tel=tel)
        else:
            out = _numba_lta(P, T_K, method, N, seed, n_steps, saves, diagnostics, warm, tel=tel)
    else:
        out = _torch_run(kind, P, T_K, method, N, seed, n_steps, saves, device, fr_timing, warm, tel=tel,
                         monitor=monitor)
    load1 = os.getloadavg()
    gmon = monitor.summary() if monitor is not None else None
    gpu_after = gmon["last_processes"] if gmon is not None else None
    busy = tel.busy()
    smt, smt_busy = smt_state(affinity, sib, busy)

    # ---- consistency of the run with the request
    if not np.array_equal(out["save_step"], saves):
        raise BenchError("engine save grid differs from the requested one")
    if not np.all(np.isfinite(out["wall_s_at_save"])) or np.any(np.diff(out["wall_s_at_save"]) < 0):
        raise BenchError("save timestamps missing or not monotone")
    for k in ("M_all", "C_all"):
        if not np.all(np.isfinite(out[k])):
            raise BenchError(f"non-finite {k}")
    n_dep = out["C_all"].sum(axis=1)
    exp_dep = (N * saves) if kind == "gateway" else (N * (saves + 1))
    if not np.allclose(n_dep, exp_dep, rtol=0, atol=0.5):
        raise BenchError("deposit counts at the saves differ from the force-evaluation accounting")

    engine_files = (["src/gateway_ladder_numba.py"] if kind == "gateway" else ["src/lta_ladder_numba.py"]) \
        if backend == "cpu_numba_1core" else (["src/gateway_core.py", "src/eb_abffr_core.py"] if kind == "gateway"
                                             else ["src/lta/core_lta.py", "src/alkanes/core.py",
                                                   "src/alkanes/periodic.py"])
    files = engine_files + ["scripts/mechanism/parallel_benchmark.py"] + \
        (["scripts/mechanism/bench_torch_engines.py"] if backend != "cpu_numba_1core" else [])
    meta = dict(
        status="complete", signature=sig, harness=HARNESS_VERSION, system=system, kind=kind, backend=backend,
        method=method, seed=int(seed), N=int(N), n_steps=int(n_steps), n_steps_full=int(n_full),
        budget_fraction=n_steps / n_full, h=float(P["engine_cfg"]["h"]), T=n_steps * float(P["engine_cfg"]["h"]),
        T_K=T_K, threshold_e_F_mid=thr, thresholds=P["thresholds"], config=P["config_path"],
        engine_cfg=P["engine_cfg"], save_grid_rule="run_ladder.save_grid (budget grid + physical checkpoints <= T)",
        n_saves=int(saves.size), device=device, physical_gpu=gpu_idx, allowed_gpus=allowed_gpus(),
        gpu_processes_before=gpu_before, gpu_processes_after=gpu_after, gpu_monitor=gmon,
        gpu_occupancy=(None if gmon is None else ("unknown" if gpu_before is None else
                                                  ("shared" if [p for p in gpu_before if p["pid"] != os.getpid()]
                                                   else gmon["occupancy"]))),
        pid=os.getpid(), affinity=affinity, pinned=pinned, smt_siblings=sib, loadavg_before=load0,
        loadavg_after=load1, cpu_busy_fraction=busy, smt_state=smt, smt_sibling_busy_max=smt_busy,
        telemetry_window="timed run: begin right before t0, end right after the final device sync",
        host=platform.node(), python=platform.python_version(),
        numpy=np.__version__, import_s=import_s, diagnostics=bool(diagnostics), fr_timing=bool(fr_timing),
        provenance=provenance(files), finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        timing_definitions=dict(
            wall_total_s="perf_counter from the start of the timed run to its end (after a device sync)",
            wall_s_at_save="elapsed wall at each save (gateway: right after step k - 1; LTA: after the sub-chunk "
                           "that evaluates step k, which includes k's move); device synchronised before each stamp",
            warmup_s="one short run of the same code path before the timed run (JIT compile / numba cache load, "
                     "CUDA kernel and library loading)",
            import_s="process start (/proc) to run_one entry (interpreter start-up + numpy + this module)",
            engine_import_s="import of the engine modules (torch + engines, or numba + engine)",
            fr_time_s_at_save="cumulative device time inside the FR blocks (CUDA events / perf_counter); NaN = not "
                              "measurable (numba)",
            monitor_s="gpu_torch: nvidia-smi sampling at saves (device idle), excluded from every stamp and the total",
            process_wall_s="process start (/proc) to the end of the run (before writing): the allocated device time",
            peak_gpu_device_mb="nvidia-smi used_memory of this process (context + workspaces + allocator cache), max "
                               "over the samples during and right after the timed run",
            peak_rss_mb="host VmHWM; torch backends: reset right before the timed run, read right after it"))
    for k in ("warmup_s", "warmup_steps", "cuda_init_s", "engine_import_s", "device_name", "torch_version", "torch_cuda",
              "torch_deterministic", "engine_wall_s", "engine_meta"):
        if k in out:
            meta[k] = out.pop(k)
    scal = dict(wall_total_s=float(out.pop("wall_total_s")), n_fr_opps=int(out.pop("n_fr_opps")),
                n_force_evals=int(out.pop("n_force_evals")), peak_rss_mb=float(out.pop("peak_rss_mb", math.nan)),
                peak_gpu_alloc_mb=float(out.pop("peak_gpu_alloc_mb", math.nan)),
                peak_gpu_reserved_mb=float(out.pop("peak_gpu_reserved_mb", math.nan)),
                gpu_reserved_end_mb=float(out.pop("gpu_reserved_end_mb", math.nan)),
                peak_gpu_device_mb=float(gmon["peak_device_mb"]) if gmon is not None else math.nan,
                monitor_s=float(out.pop("monitor_s", 0.0)))
    scal["us_per_step"] = scal["wall_total_s"] * 1e6 / n_steps
    scal["ns_per_walker_step"] = scal["wall_total_s"] * 1e9 / (n_steps * N)
    meta["scalars"] = scal
    if backend == "cpu_numba_1core" and n_steps == n_full:
        meta["bitwise_vs_production"] = bitwise_vs_production(system, seed, method, out)
    else:
        meta["bitwise_vs_production"] = dict(status="not applicable (" + (
            "torch backend: different random streams" if backend != "cpu_numba_1core" else "budget fraction < 1") + ")")
    meta["process_wall_s"] = (time.time() - start_epoch) if start_epoch else math.nan
    arrays = {k: np.asarray(v) for k, v in out.items()}
    arrays.update({k: np.asarray(v) for k, v in scal.items()})
    arrays["meta_json"] = np.array(json.dumps(meta, sort_keys=True, default=_json_default))
    atomic_savez(npz, **arrays)
    atomic_json(js, meta)
    return meta


def load_run(npz):
    """The arrays and the parsed meta of one benchmark result."""
    with np.load(npz, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta_json"}
        out["meta"] = json.loads(str(z["meta_json"]))
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray) and v.ndim == 0:
            out[k] = v[()]
    return out


# ================================================================================================ campaign driver
def parse_cpus(s):
    out = []
    for part in str(s).split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def job_list(backend, systems, seeds_per_arm, seeds=None):
    J = []
    for system in systems:
        P, N, _ = system_spec(system)
        ss = list(seeds) if seeds else list(P["seeds"])[:int(seeds_per_arm)]
        for i, seed in enumerate(ss):
            arms = METHODS if i % 2 == 0 else METHODS[::-1]        # alternate the arm order (drift cancels)
            for m in arms:
                J.append((system, backend, m, int(seed)))
    return J


def _complete_metas(out_root, backend):
    out = []
    for system in SYSTEMS:
        d = os.path.join(out_root, system, backend)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".json") and f.startswith("s"):
                try:
                    with open(os.path.join(d, f)) as fh:
                        m = json.load(fh)
                except Exception:  # noqa: BLE001
                    continue
                if m.get("status") == "complete":
                    out.append(m)
    return out


def run_cost_s(meta):
    """Allocated device-seconds of one finished run: process start to the end of the run (process_wall_s), or for
    an older file the sum of its measured phases."""
    v = meta.get("process_wall_s")
    if v is not None and math.isfinite(float(v)):
        return float(v)
    sc = meta.get("scalars", {})
    parts = [sc.get("wall_total_s"), meta.get("warmup_s"), meta.get("cuda_init_s"), meta.get("import_s"),
             meta.get("engine_import_s")]
    return float(sum(float(x) for x in parts if x is not None and math.isfinite(float(x))))


def used_device_hours(out_root, backend):
    return sum(run_cost_s(m) for m in _complete_metas(out_root, backend)) / 3600.0


def projected_run_s(out_root, system, backend, method, n_steps):
    """(seconds, source) of one run: the measured us/step and per-run overhead of finished runs of the same
    (system, backend, method) under out_root, else FALLBACK_US_PER_STEP + RUN_OVERHEAD_FALLBACK_S."""
    ms = [m for m in _complete_metas(out_root, backend) if m.get("system") == system and m.get("method") == method]
    if ms:
        us = float(np.median([m["scalars"]["us_per_step"] for m in ms]))
        ov = float(np.median([max(run_cost_s(m) - m["scalars"]["wall_total_s"], 0.0) for m in ms]))
        return us * 1e-6 * n_steps + ov, f"measured ({len(ms)} runs)"
    us = FALLBACK_US_PER_STEP.get((backend, system))
    if us is None:
        raise BenchError(f"no cost model for {backend} / {system}")
    return us * 1e-6 * n_steps + RUN_OVERHEAD_FALLBACK_S, "fallback (FALLBACK_US_PER_STEP)"


def default_max_hours(backend):
    return PLAN_CEILING_H["gpu_torch" if backend == "gpu_torch" else "cpu"]


def budget_allows(out_root, job, partner_pending, committed_s, max_hours, n_steps_of):
    """Whether launching ``job`` keeps the backend's allocated device-hours within max_hours: finished runs under
    out_root + committed (projected, running) + this job + its partner arm if that has not run or started yet."""
    system, backend, method, seed = job[:4]
    used_s = used_device_hours(out_root, backend) * 3600.0
    need, src = projected_run_s(out_root, system, backend, method, n_steps_of(system))
    if partner_pending:
        other = "fr" if method == "abf" else "abf"
        need += projected_run_s(out_root, system, backend, other, n_steps_of(system))[0]
    tot_h = (used_s + committed_s + need) / 3600.0
    return tot_h <= max_hours, dict(used_h=used_s / 3600.0, committed_h=committed_s / 3600.0, next_h=need / 3600.0,
                                    total_h=tot_h, max_h=float(max_hours), source=src)


def campaign(backend, systems, seeds_per_arm, cpus, out_root=EXP_ROOT, n_steps=None, seeds=None, dry_run=False,
             diagnostics=False, extra=(), idle_check=True, max_wait_s=1800.0, allow_contended=False,
             max_device_hours=None, log=print):
    if backend in UNAVAILABLE_BACKENDS:
        raise BenchError(f"{backend}: {UNAVAILABLE_BACKENDS[backend]}")
    if backend == "gpu_torch" and not dry_run:
        gpu_guard(allow_busy=False, pid=-1)          # refuse up front: disallowed index / busy / unknown occupancy
    max_h = default_max_hours(backend) if max_device_hours is None else float(max_device_hours)
    J = job_list(backend, systems, seeds_per_arm, seeds)

    def n_steps_of(system):
        P, N, _ = system_spec(system)
        return full_n_steps(P, N) if n_steps is None else int(n_steps)
    todo = []
    for system, b, m, seed in J:
        P, N, _ = system_spec(system)
        sig = run_signature(system, b, m, seed, N, n_steps_of(system), P, diagnostics, None)
        st = existing_status(out_paths(out_root, system, b, seed, m)[0], sig)
        todo.append((system, b, m, seed, st))
    absent = [t for t in todo if t[4] == "absent"]
    proj = sum(projected_run_s(out_root, t[0], t[1], t[2], n_steps_of(t[0]))[0] for t in absent) / 3600.0
    used = used_device_hours(out_root, backend)
    unit = "GPU-h" if backend == "gpu_torch" else "core-h"
    log(f"{backend}: {len(J)} jobs, {len(absent)} to run; projected {proj:.2f} {unit} + {used:.2f} used under "
        f"{out_root} vs max {max_h:.2f}" + ("  -> WILL STOP EARLY (resource guard)" if proj + used > max_h else ""))
    for t in todo:
        log(f"   {t}")
    phys, seen = [], set()
    for c in cpus:                          # one logical CPU per physical core: never two jobs on SMT siblings
        key = tuple(sorted([c] + smt_siblings([c])))
        if key not in seen:
            seen.add(key)
            phys.append(c)
    if phys != list(cpus):
        log(f"  cpus {list(cpus)} -> one per physical core: {phys} (SMT siblings make a run 1.4-1.7x slower)")
    cpus = phys
    if dry_run:
        return todo
    base = [sys.executable, os.path.abspath(__file__), "run", "--out-root", out_root]
    if n_steps is not None:
        base += ["--n-steps", str(int(n_steps))]
    if diagnostics:
        base += ["--diagnostics"]
    base += list(extra)

    def cmd(t, cpu):
        system, b, m, seed, _ = t
        return ["taskset", "-c", str(cpu)] + base + ["--system", system, "--backend", b, "--method", m,
                                                     "--seed", str(seed)]
    started = set()

    def partner_pending(t):
        other = "fr" if t[2] == "abf" else "abf"
        return any(u[0] == t[0] and u[3] == t[3] and u[2] == other and (u[0], u[3], u[2]) not in started
                   for u in absent)
    if backend == "gpu_torch":
        for t in absent:                    # ONE simulation on the device at a time; host thread on an idle core
            ok, info = budget_allows(out_root, t, partner_pending(t), 0.0, max_h, n_steps_of)
            if not ok:
                log(f"  STOP (resource guard): {t[:4]} would bring {backend} to {info['total_h']:.2f} > "
                    f"{max_h:.2f} {unit} ({info})")
                break
            cpu, idle = choose_cpu(list(cpus), idle_check, max_wait_s, allow_contended=allow_contended, log=log)
            if cpu is None:
                break
            started.add((t[0], t[3], t[2]))
            r = subprocess.run(cmd(t, cpu))
            log(f"  {t[:4]} host cpu {cpu}{'' if idle else ' (CONTENDED)'} -> exit {r.returncode}")
        return todo
    procs = {}
    queue = list(absent)
    free = list(cpus)
    wait_since = None                       # start of the current wait for an idle physical core
    stop = False
    while (queue and not stop) or procs:
        while queue and free and not stop:
            t = queue[0]
            committed = sum(c for _, _, c in procs.values())
            ok, info = budget_allows(out_root, t, partner_pending(t), committed, max_h, n_steps_of)
            if not ok:
                log(f"  STOP (resource guard): {t[:4]} would bring {backend} to {info['total_h']:.2f} > "
                    f"{max_h:.2f} {unit} ({info}); waiting for the running jobs")
                stop = True
                break
            if idle_check:
                okc, bf = idle_cpus(free)
                if okc:
                    cpu, idle, wait_since = okc[0], True, None
                else:
                    wait_since = wait_since or time.time()
                    if time.time() - wait_since < max_wait_s:
                        break               # wait for a core (and its sibling) to become idle
                    if not allow_contended:
                        log(f"  STOP: no idle physical core among {free} after {max_wait_s:.0f} s; not launching "
                            f"contended runs (pass --allow-contended); waiting for the running jobs")
                        stop = True
                        break
                    cpu, idle = least_contended(free, bf), False
            else:
                cpu, idle = free[0], True
            free.remove(cpu)
            queue.pop(0)
            started.add((t[0], t[3], t[2]))
            cost = projected_run_s(out_root, t[0], t[1], t[2], n_steps_of(t[0]))[0]
            procs[cpu] = (t, subprocess.Popen(cmd(t, cpu)), cost)
            if not idle:
                log(f"  WARNING: {t[:4]} launched on cpu {cpu} with a busy SMT sibling (--allow-contended): "
                    f"excluded from W1 by the analysis")
        time.sleep(2.0)
        for cpu in list(procs):
            t, p, _ = procs[cpu]
            if p.poll() is not None:
                log(f"  {t[:4]} on cpu {cpu} -> exit {p.returncode}")
                del procs[cpu]
                free.append(cpu)
    return todo


# ================================================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one (system, backend, method, seed) run")
    r.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    r.add_argument("--backend", required=True, choices=list(BACKENDS + TEST_BACKENDS) + list(UNAVAILABLE_BACKENDS))
    r.add_argument("--method", required=True, choices=METHODS)
    r.add_argument("--seed", required=True, type=int)
    r.add_argument("--out-root", default=EXP_ROOT)
    r.add_argument("--n-steps", type=int, default=None, help="shortened budget (smoke / tests); default B / N")
    r.add_argument("--device", default=None)
    r.add_argument("--diagnostics", action="store_true", help="numba: production diagnostics on (default off)")
    r.add_argument("--no-fr-timing", action="store_true")
    r.add_argument("--no-warmup", action="store_true")
    r.add_argument("--allow-busy-gpu", action="store_true",
                   help="run on an allowed GPU that holds another process or whose occupancy is unknown (recorded)")
    r.add_argument("--allow-unpinned", action="store_true",
                   help="run a timed backend without a one-CPU affinity (recorded; contention 'unknown')")
    r.add_argument("--gpu-poll-s", type=float, default=GPU_POLL_S)
    r.add_argument("--force", action="store_true", help="re-run even if a complete result with this signature exists")
    c = sub.add_parser("campaign", help="all jobs of one backend")
    c.add_argument("--backend", required=True, choices=list(BACKENDS + TEST_BACKENDS) + list(UNAVAILABLE_BACKENDS))
    c.add_argument("--systems", nargs="*", default=sorted(SYSTEMS))
    c.add_argument("--seeds-per-arm", type=int, default=DEFAULT_SEEDS_PER_ARM)
    c.add_argument("--seeds", type=int, nargs="*", default=None)
    c.add_argument("--cpus", default="64")
    c.add_argument("--out-root", default=EXP_ROOT)
    c.add_argument("--n-steps", type=int, default=None)
    c.add_argument("--diagnostics", action="store_true")
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--no-idle-check", action="store_true",
                   help="launch without checking that the CPU and its SMT sibling are idle (default: check)")
    c.add_argument("--max-wait-s", type=float, default=1800.0,
                   help="longest wait for an idle physical core; then STOP unless --allow-contended")
    c.add_argument("--allow-contended", action="store_true",
                   help="after --max-wait-s launch on the least-contended core (the run is excluded from W1)")
    c.add_argument("--max-device-hours", type=float, default=None,
                   help="resource guard: allocated device-hours of this backend under out-root (default: 8 GPU-h for "
                        "gpu_torch = the plan ceiling; 20 core-h otherwise = the plan's Experiment III estimate)")
    a = ap.parse_args(argv)
    if a.backend in UNAVAILABLE_BACKENDS:
        print(f"{a.backend}: {UNAVAILABLE_BACKENDS[a.backend]}", file=sys.stderr)
        return 2
    if a.cmd == "run":
        if a.backend != "gpu_torch":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        m = run_one(a.system, a.backend, a.method, a.seed, out_root=a.out_root, n_steps=a.n_steps, device=a.device,
                    diagnostics=a.diagnostics, fr_timing=not a.no_fr_timing, warm=not a.no_warmup,
                    allow_busy_gpu=a.allow_busy_gpu, force=a.force, allow_unpinned=a.allow_unpinned,
                    gpu_poll_s=a.gpu_poll_s)
        s = m["scalars"]
        print(f"{a.system} {a.backend} {a.method} s{a.seed}: n_steps {m['n_steps']} wall {s['wall_total_s']:.2f} s "
              f"({s['us_per_step']:.1f} us/step, {s['ns_per_walker_step']:.1f} ns/walker-step), warm-up "
              f"{m.get('warmup_s', float('nan')):.2f} s, bitwise vs production: {m['bitwise_vs_production']['status']}",
              flush=True)
        return 0
    try:
        campaign(a.backend, a.systems, a.seeds_per_arm, parse_cpus(a.cpus), out_root=a.out_root, n_steps=a.n_steps,
                 seeds=a.seeds, dry_run=a.dry_run, diagnostics=a.diagnostics, idle_check=not a.no_idle_check,
                 max_wait_s=a.max_wait_s, allow_contended=a.allow_contended, max_device_hours=a.max_device_hours,
                 log=lambda *x: print(*x, flush=True))
    except BenchError as e:
        print(f"campaign refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
