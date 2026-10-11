"""Walker-parallel CPU engine for the entropic gateway (Experiment B): C/OpenMP step kernel + validated numba FR.

Final campaign, Experiment B (docs/final_campaign/PREREGISTRATION.md section 2.2-2.3; engineering brief
docs/final_campaign/EXPB_BACKEND_DESIGN.md).  Same mathematical algorithm as the validated equal-budget engine
``src/gateway_ladder_numba.py`` (= gateway_family_numba at alpha = 1, lam = 1), which is imported and never edited.

Structure
---------
* ``gw_kernel.c`` (built here on first use; see BUILD below) advances the loop steps [step0, step1) between two FR
  opportunities / saves: per step the force/bin of every walker (static walker partition), barrier, BIN-OWNER
  deposit (thread t owns a contiguous bin range and scans all N walkers in slot order: per bin the serial
  slot-order sum, bitwise, at any thread count), barrier, bias read + Euler-Maruyama move (+ the next step's force
  of the same walker).  One OpenMP parallel region per kernel call, two barriers per step.
* The driver ``_drive_chunk`` is an @njit function that loops over the segments of a chunk and calls the kernel
  through its ctypes function pointer (no Python per segment); after a segment it runs FR (validated numba
  ``kde_density``, ``uniform_scores``, ``fr_resample`` with the opportunity's fixed block of
  L = N + cap + max(N, cap) uniforms of PCG64(SeedSequence([seed, N, 1])), drawn per chunk in Python exactly as
  ``run_arm`` draws them), the gather and counters of ``_advance``'s diagnostics-off path, then the save.
  A segment ends at the next FR loop step + 1, the next save, or (replay) the noise-buffer length.
* Kernel variants (``variant=`` / env GW_PARALLEL_VARIANT; all bitwise the same result, all covered by the tests):
  'owner_omp' (DEFAULT: the design brief's bin-owner deposit, two ``#pragma omp barrier`` per step),
  'replicated_omp' (every thread adds all N deposits in slot order into a private copy of the table: one barrier
  per step), 'owner_dissem' / 'replicated_dissem' (the same with a C11 dissemination barrier inside the OpenMP
  region instead of ``omp barrier``; it yields the cpu after ~2^14 spins, so an oversubscribed core costs a yield,
  not a time slice per barrier).  The variant is recorded in meta and in the run signature.
* Noise.  ``replay``: the validated engine's stream -- numba's MT19937 seeded with
  ``default_noise_seed(seed, N)`` through ``gateway_ladder_numba._mt_seed``, 2N ``np.random.standard_normal()`` per
  step in the order zx[0..N-1], zy[0..N-1], drawn serially by the driver into a buffer before each kernel call
  (bitwise the validated engine at every thread count; serial, so it is the validation mode, not the timing
  mode).  ``counter``: Philox4x64-10 (Random123 constants) keyed by SeedSequence([seed, N, 2]).generate_state(2,
  np.uint64), counter (loop step, slot, 0, 0), u = (o >> 11) 2^-53, Box-Muller (zx, zy) = sqrt(-2 log(1 - u0))
  (cos, sin)(2 pi u1), generated inside the parallel region; results are bitwise independent of the thread count.
  A clone continues with its slot's stream.  The loop step is the completed-step count before the move.

Result contract: ``run_arm_parallel`` returns every key ``gateway_ladder_numba.run_arm(..., diagnostics=False)``
returns except the wall/provenance keys (wall_s, peak_rss_mb, n_checkpoints_written, resumed_from, meta), with the
same names, dtypes and conventions (pre-deposit accumulators, N force evaluations per completed step), plus
wall_s_at_save, n_force_evals_at_save (N k), wall_total_s, warmup_s, t_force_move_s, t_abf_s (deposit + bias
read), t_sync_s (region launch + barrier + join waits), t_fr_s (incl. drawing the FR uniform blocks), t_save_s,
threads, noise_mode, engine ('gateway_parallel/1') and meta (a JSON string: sources' sha256, git commit, compiler,
flags, library versions, host, cpus, OpenMP binding, TSC frequency, driver overhead, extra timers, run signature).
meta['cpus'] is the process allocation, os.sched_getaffinity(0) read at IMPORT (before this module loads libgomp:
with OMP_PROC_BIND=close libgomp binds the initial thread to place 0 when it loads, after which the process-wide
call returns that one cpu); the master's post-binding affinity is meta['cpus_master_bound'] and the team's cpus
meta['omp']['binding']['team_cpus'].  allocated_cpus() returns the same allocation for callers (harness).
The in-kernel timers are per-thread rdtsc tick counts averaged over the team.  The per-call stamps (region launch,
entry force/bin, step-loop start / end, join) are exact; the phase reads inside the step loop are taken on a
deterministic sample of ~1 loop step in 2^TIMER_SAMPLE_LOG2 (= 256): s_0 = 0, s_{i+1} = s_i + 1 + (mix64(s_i) >>
(63 - TIMER_SAMPLE_LOG2)) with mix64 the splitmix64 finaliser, a function of the global step only.  The exact
step-loop ticks are split over force/move, deposit, bias read and barrier wait in proportion to the sampled ticks,
so t_force_move + t_abf + t_sync = the exact parallel-region wall time and only the split inside the loop is a
(ratio) estimate.  Between sampled steps the kernel runs the timers-off code.  Reason: reading rdtsc 3-6 times
every step cost +27 % / +8 % / +4 % at N = 1 / 2 / 4 (t = 1), an overhead the validated serial engine does not pay;
sampled, the measured overhead is <= 0.33 % at N = 1-16 (re-measure with --overhead).  With timers=False they are
NaN (driver timers stay on).
The timed start t0 is after the build, JIT compilation, a warm-up run of the same code path (same N, threads,
method, noise; it also starts the OpenMP thread pool) and the state set-up; warmup_s is everything before t0.

OpenMP environment (set at IMPORT, before the kernel library and libgomp load; libgomp reads it once):
OMP_WAIT_POLICY=active, OMP_PROC_BIND=close, OMP_PLACES = one place per cpu of os.sched_getaffinity(0) (sorted),
OMP_DYNAMIC=false, so thread t runs on the t-th allocated cpu; GOMP_SPINCOUNT defaults to 1e7 (~0.1 s; libgomp's
default under the active policy is 3e10, ~5 min, which leaves the idle pool of a finished run spinning on the
allocated cores and starves a later process there; every gap inside a run is <= a few ms, so in-run behaviour is
the active policy's).  Launch every run with its cpu set already applied (taskset) and import this module in the
process that runs the engine, before anything else that loads libgomp; meta records whether libgomp was already
loaded at import, the environment and the measured team binding (team_cpus).  GW_PARALLEL_KEEP_OMP_ENV=1 leaves
the caller's OMP_* / GOMP_* values in place.  Side effect of the binding: once the kernel is loaded the main thread
is pinned to the first allocated cpu, and a child process started from it inherits that single cpu; start children
from a process that never loaded the engine, or restore allocated_cpus() in the child (preexec_fn).

BUILD: gcc -O3 -march=native -fopenmp -fno-fast-math -ffp-contract=off -fPIC -shared, output
~/.cache/final_campaign_build/gw_kernel_<tag>.so with tag = sha256(source sha256, compiler, flags, gcc version);
a changed source or compiler gives a new tag (rebuild).  Override the directory with GW_PARALLEL_BUILD_DIR, the
compiler with GW_PARALLEL_CC.

Counter-mode helpers for the RNG checks: counter_key(seed, N), philox_raw(ctr, key) (the C Philox),
counter_normals(key, step0, n_steps, N, nthreads) (the kernel's exact noise, layout [step][zx|zy][slot]).  The
counter map (step, slot, 0, 0) is the coordinate-wise embedding (injective for step, slot < 2^64; the LTA
backend's parallel.philox.check_counter_injective(N, n_steps, words=1) states it for one configuration).

CLI (PYTHONPATH=src):  python -m parallel.gateway_parallel --vb1 --N 512 --method fr --threads 1,4,16 --out v.json
                       python -m parallel.gateway_parallel --vb2 --N 512 --method fr --threads 1,2,4,8,16 --out v.json
                       python -m parallel.gateway_parallel --vb5 --N 512 --threads 16 --out v.json
                       python -m parallel.gateway_parallel --probe 512 16 20000   (development only)
                       python -m parallel.gateway_parallel --overhead 1,2,4,8,16 --out o.json   (timer overhead,
                                                                                   development only)
Exit status: 0 = PASS; 1 = FAIL (incl. V-B1 IDENTICAL_BUT_SHORT); 2 = usage error (argparse); 3 (V-B5 only) = the
preregistered criterion ('bitwise identical before FR's first realised death') fails while the weaker
before-the-first-gather criterion passes (verdict 'FAIL_PREREG_DEATH_CRITERION_PASS_EVENT_ONLY').  The V-B5 gate
is the preregistered death criterion unless --vb5-criterion event --amendment '<dated amendment reference>' is
given (recorded in the output JSON).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

# ---------------------------------------------------------------------------------------------------------------
# OpenMP environment: must be in place before libgomp is loaded (it reads the environment once, at load)
# ---------------------------------------------------------------------------------------------------------------


def _libgomp_loaded():
    try:
        with open("/proc/self/maps") as f:
            return any("libgomp" in line for line in f)
    except OSError:
        return None


_LIBGOMP_PRELOADED = _libgomp_loaded()
_AFFINITY_AT_IMPORT = sorted(os.sched_getaffinity(0))
_OMP_ENV_BEFORE = {k: os.environ.get(k) for k in ("OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES",
                                                  "OMP_NUM_THREADS", "OMP_DYNAMIC", "GOMP_SPINCOUNT")}
if os.environ.get("GW_PARALLEL_KEEP_OMP_ENV") != "1":
    os.environ["OMP_WAIT_POLICY"] = "active"
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = ",".join("{%d}" % c for c in _AFFINITY_AT_IMPORT)
    os.environ["OMP_DYNAMIC"] = "false"
    # libgomp's spin before futex-sleeping: with OMP_WAIT_POLICY=active its default is 3e10 (~5 min), so idle pool
    # threads of a finished run keep spinning on the allocated cores and starve any later process there.  1e7 spins
    # (~0.1 s) covers every gap inside a run (FR, saves, chunk set-up: <= a few ms), so the in-run behaviour is the
    # active policy's.  A caller-set GOMP_SPINCOUNT is kept.
    os.environ.setdefault("GOMP_SPINCOUNT", "10000000")

import numpy as np  # noqa: E402
import numba  # noqa: E402
from numba import njit  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
ROOT = os.path.dirname(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import gateway_ladder_numba as gl  # noqa: E402

ENGINE = "gateway_parallel/1"
KERNEL_SRC = os.path.join(_HERE, "gw_kernel.c")
SOURCE_FILES = (os.path.join(_HERE, "gateway_parallel.py"), KERNEL_SRC, os.path.join(_HERE, "__init__.py"))
VALIDATED_FILES = (os.path.join(_SRC, "gateway_ladder_numba.py"),)
CC = os.environ.get("GW_PARALLEL_CC", "/usr/bin/gcc")
CFLAGS = ("-O3", "-march=native", "-fopenmp", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-shared")
BUILD_DIR = os.environ.get("GW_PARALLEL_BUILD_DIR", os.path.expanduser("~/.cache/final_campaign_build"))
REPLAY_BUF_DOUBLES = 1 << 20          # replay noise buffer (8 MB): segment length <= REPLAY_BUF_DOUBLES // (2N) steps
CHUNK_OPPS = 256                      # FR opportunities per Python chunk (one PCG64 draw of CHUNK_OPPS * L doubles)
WALL_KEYS = ("wall_s", "peak_rss_mb", "n_checkpoints_written", "resumed_from", "meta")   # not part of the contract
NOISE_MODES = {"replay": 0, "counter": 1}
# kernel variants (gw_kernel.c ip[7]); every variant is bitwise the same algorithm; 'owner_omp' is the design brief's
# kernel (bin-owner deposit, two `omp barrier` per step) and the default
VARIANTS = {"owner_omp": 0, "replicated_omp": 1, "owner_dissem": 2, "replicated_dissem": 3}
DEFAULT_VARIANT = os.environ.get("GW_PARALLEL_VARIANT", "replicated_dissem")   # Amendment 2 (A2.1)
# in-kernel timers: phase reads on ~1 loop step in 2^TIMER_SAMPLE_LOG2 (gw_kernel.c: sampling gaps uniform on
# [1, 2^(TIMER_SAMPLE_LOG2 + 1)]; 0 = every step)
TIMER_SAMPLE_LOG2 = 8
N_ACC = 16                            # gw_kernel.c acc[] length

# integer-counter slots shared with the validated engine
I_SPTR, I_NOPP, I_NOPP_EV, I_KD, I_KC, I_DEATHS = gl.I_SPTR, gl.I_NOPP, gl.I_NOPP_EV, gl.I_KD, gl.I_KC, gl.I_DEATHS
I_NFE, I_STEP, I_NOPP_D, N_ICTR = gl.I_NFE, gl.I_STEP, gl.I_NOPP_D, gl.N_ICTR
# driver timer slots (seconds, CLOCK_MONOTONIC)
TI_KERNEL, TI_FR, TI_SAVE, TI_REPLAY, TI_NSEG, TI_NFR = range(6)
N_TI = 8


def _sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # pragma: no cover
        return ""


def _provenance_at_import():
    rels = [os.path.relpath(p, ROOT) for p in SOURCE_FILES + VALIDATED_FILES]
    status = _git("status", "--porcelain", "--", *rels)
    return dict(git_commit=_git("rev-parse", "HEAD") or "unknown", git_dirty=bool(status), git_status=status,
                sources_sha256={os.path.relpath(p, ROOT): _sha256_file(p) for p in SOURCE_FILES if os.path.exists(p)},
                validated_sha256={os.path.relpath(p, ROOT): _sha256_file(p) for p in VALIDATED_FILES},
                imported_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


_PROVENANCE = _provenance_at_import()

# ---------------------------------------------------------------------------------------------------------------
# build / load the C kernel
# ---------------------------------------------------------------------------------------------------------------
_LIB = None
_BUILD_INFO = None
_c_advance = None          # ctypes function objects, read by the numba driver at its (first-call) compilation
_c_now = None


def _gcc_version():
    out = subprocess.run([CC, "--version"], capture_output=True, text=True, timeout=30)
    return out.stdout.splitlines()[0].strip() if out.stdout else "unknown"


def build_kernel(force=False):
    """Compile gw_kernel.c (if the tagged .so is absent) and return the build record."""
    src_sha = _sha256_file(KERNEL_SRC)
    gver = _gcc_version()
    tag = hashlib.sha256("\n".join([src_sha, CC, " ".join(CFLAGS), gver]).encode()).hexdigest()[:16]
    os.makedirs(BUILD_DIR, exist_ok=True)
    so = os.path.join(BUILD_DIR, f"gw_kernel_{tag}.so")
    built_now = False
    if force or not os.path.exists(so):
        tmp = so + f".tmp{os.getpid()}"
        cmd = [CC, *CFLAGS, "-o", tmp, KERNEL_SRC, "-lm"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"kernel build failed ({' '.join(cmd)}):\n{r.stderr}")
        os.replace(tmp, so)
        built_now = True
    return dict(so_path=so, so_sha256=_sha256_file(so), source_sha256=src_sha, compiler=CC, compiler_version=gver,
                cflags=list(CFLAGS), command=" ".join([CC, *CFLAGS, "-o", so, KERNEL_SRC, "-lm"]), tag=tag,
                built_in_this_process=built_now)


def load_kernel():
    """Build (if needed) and load the kernel library; set the ctypes signatures and the driver's globals."""
    global _LIB, _BUILD_INFO, _c_advance, _c_now
    if _LIB is not None:
        return _LIB
    info = build_kernel()
    lib = ctypes.CDLL(info["so_path"])
    vp, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    lib.gw_advance.argtypes = [vp] * 14
    lib.gw_advance.restype = ctypes.c_int
    lib.gw_now.argtypes = []
    lib.gw_now.restype = ctypes.c_double
    lib.gw_rdtsc.argtypes = []
    lib.gw_rdtsc.restype = ctypes.c_uint64
    lib.gw_max_threads.restype = ctypes.c_int
    lib.gw_omp_proc_bind.restype = ctypes.c_int
    lib.gw_omp_num_places.restype = ctypes.c_int
    lib.gw_omp_place_ids.argtypes = [i32, vp, i32]
    lib.gw_omp_place_ids.restype = ctypes.c_int
    lib.gw_team_places.argtypes = [i32, vp]
    lib.gw_team_places.restype = ctypes.c_int
    lib.gw_philox_many.argtypes = [i64, vp, vp, vp]
    lib.gw_philox_many.restype = None
    lib.gw_counter_normals.argtypes = [vp, i64, i64, i64, i32, vp]
    lib.gw_counter_normals.restype = None
    _LIB, _BUILD_INFO = lib, info
    _c_advance, _c_now = lib.gw_advance, lib.gw_now
    return lib


def _ptr(a):
    return ctypes.c_void_p(a.ctypes.data)


def allocated_cpus():
    """The process cpu allocation: os.sched_getaffinity(0) read at import, before this module loads libgomp (once
    loaded with OMP_PROC_BIND=close, libgomp binds the initial thread to place 0 and sched_getaffinity(0) then
    returns that single cpu).  If libgomp was already loaded at import (meta omp.libgomp_loaded_before_import),
    the import-time value may itself be the bound one; the union of the OpenMP places is recorded alongside."""
    return list(_AFFINITY_AT_IMPORT)


def _aligned_zeros(n, dtype=np.float64, align=128):
    """Zero array of n elements whose data starts on an `align`-byte boundary (the kernel partitions walkers in
    blocks of 8, i.e. 64 bytes of doubles, so no cache line is shared by two threads)."""
    dt = np.dtype(dtype)
    buf = np.zeros(int(n) * dt.itemsize + align, dtype=np.uint8)
    off = (-buf.ctypes.data) % align
    a = buf[off:off + int(n) * dt.itemsize].view(dt)
    assert a.ctypes.data % align == 0
    return a


def now():
    """CLOCK_MONOTONIC seconds (the clock of every wall stamp of this engine)."""
    return load_kernel().gw_now()


# ---------------------------------------------------------------------------------------------------------------
# counter-mode RNG helpers (exact kernel noise; used by tests and the RNG battery)
# ---------------------------------------------------------------------------------------------------------------
def counter_key(seed, N):
    """Philox key of the gateway counter mode: SeedSequence([seed, N, 2]).generate_state(2, np.uint64)."""
    return np.random.SeedSequence([int(seed), int(N), 2]).generate_state(2, np.uint64)


def philox_raw(ctr, key):
    """C Philox4x64-10 blocks: ctr (n, 4) uint64, key (2,) uint64 -> (n, 4) uint64."""
    lib = load_kernel()
    ctr = np.ascontiguousarray(ctr, dtype=np.uint64).reshape(-1, 4)
    key = np.ascontiguousarray(key, dtype=np.uint64).reshape(2)
    out = np.empty_like(ctr)
    lib.gw_philox_many(ctr.shape[0], _ptr(ctr), _ptr(key), _ptr(out))
    return out


def counter_normals(key, step0, n_steps, N, nthreads=1):
    """The kernel's counter-mode noise of moves step0 .. step0 + n_steps - 1: (n_steps, 2, N) = [step][zx|zy][slot]."""
    lib = load_kernel()
    key = np.ascontiguousarray(key, dtype=np.uint64).reshape(2)
    out = np.empty((int(n_steps), 2, int(N)))
    lib.gw_counter_normals(_ptr(key), int(step0), int(n_steps), int(N), int(nthreads), _ptr(out))
    return out


def omp_binding(threads):
    """Measured OpenMP binding: proc_bind policy, places (cpu ids) and the cpu of each team thread."""
    lib = load_kernel()
    npl = lib.gw_omp_num_places()
    ids = np.zeros(1024, dtype=np.int32)
    places = []
    for p in range(npl):
        n = lib.gw_omp_place_ids(p, _ptr(ids), 1024)
        places.append([int(v) for v in ids[:n]])
    team = np.full(int(threads), -9, dtype=np.int32)
    got = lib.gw_team_places(int(threads), _ptr(team))
    team_cpus = [places[t] if 0 <= t < len(places) else None for t in team[:got].tolist()]
    return dict(proc_bind=int(lib.gw_omp_proc_bind()), proc_bind_close=(lib.gw_omp_proc_bind() == 3),
                num_places=npl, places=places, team_size=int(got), team_place=team[:got].tolist(),
                team_cpus=team_cpus)


# ---------------------------------------------------------------------------------------------------------------
# numba driver
# ---------------------------------------------------------------------------------------------------------------
@njit
def _record_save(sptr, Mh, Ch, ictr, ev_hist, dh_hist, o_M, o_C, o_fr, o_evh, o_dth, o_nfe, o_wall, wall):
    """gateway_ladder_numba._rec_save with diag off, plus the wall stamp and the force-evaluation count."""
    nb = Mh.shape[0]
    for j in range(nb):
        o_M[sptr, j] = Mh[j]
        o_C[sptr, j] = Ch[j]
    o_fr[sptr, 0] = ictr[I_DEATHS]
    o_fr[sptr, 1] = ictr[I_KD]
    o_fr[sptr, 2] = ictr[I_KC]
    o_fr[sptr, 3] = ictr[I_NOPP]
    o_fr[sptr, 4] = ictr[I_NOPP_EV]
    o_fr[sptr, 5] = ictr[I_NOPP_D]
    for k in range(ev_hist.shape[0]):
        o_evh[sptr, k] = ev_hist[k]
        o_dth[sptr, k] = dh_hist[k]
    o_nfe[sptr] = ictr[I_NFE]
    o_wall[sptr] = wall


@njit
def _drive_chunk(c0, c1, N, fe, do_fr, gamma, ramp_steps, score_clip, cap, kern, r_eta, dx, dt_fr, ubuf, ublk,
                 save_at, max_seg, fill, zext, t0,
                 ip, dp, key, X, Y, Mh, Ch, fx, fy, jb, bias, zbuf, fctr, acc,
                 ictr, ev_hist, dh_hist, o_M, o_C, o_fr, o_evh, o_dth, o_nfe, o_wall,
                 p, hist, Sc, sel, die_c, clone_c, pool, tmpx, tmpy, ocnt, tim):
    """Advance loop steps [c0, c1) segment by segment: kernel, then FR (if the segment's last loop step is an FR
    step), then the save (if one is due at the segment end).  fill: 0 = draw the replay noise from numba's MT
    (validated order), 1 = counter mode (noise inside the kernel), 2 = test hook: replay noise read from zext
    (layout [step][zx|zy][slot] over the whole run).  Returns the number of FR opportunities used (ubuf blocks)."""
    n_saves = save_at.shape[0]
    sptr = ictr[I_SPTR]
    kopp = 0
    step = c0
    while step < c1:
        end = c1
        if do_fr:
            nf = ((step + fe - 1) // fe) * fe          # first FR loop step >= step
            if nf + 1 < end:
                end = nf + 1
        if sptr < n_saves and save_at[sptr] < end:
            end = save_at[sptr]
        if end - step > max_seg:
            end = step + max_seg
        ip[3] = step
        ip[4] = end
        if fill == 0:
            ta = _c_now()
            k = 0
            for _s in range(step, end):
                for i in range(N):
                    zbuf[k + i] = np.random.standard_normal()
                for i in range(N):
                    zbuf[k + N + i] = np.random.standard_normal()
                k += 2 * N
            tim[TI_REPLAY] += _c_now() - ta
            ta = _c_now()
            nth = _c_advance(ip.ctypes, dp.ctypes, key.ctypes, X.ctypes, Y.ctypes, Mh.ctypes, Ch.ctypes, fx.ctypes,
                             fy.ctypes, jb.ctypes, bias.ctypes, zbuf.ctypes, fctr.ctypes, acc.ctypes)
        elif fill == 2:
            zv = zext[step * 2 * N:]
            ta = _c_now()
            nth = _c_advance(ip.ctypes, dp.ctypes, key.ctypes, X.ctypes, Y.ctypes, Mh.ctypes, Ch.ctypes, fx.ctypes,
                             fy.ctypes, jb.ctypes, bias.ctypes, zv.ctypes, fctr.ctypes, acc.ctypes)
        else:
            ta = _c_now()
            nth = _c_advance(ip.ctypes, dp.ctypes, key.ctypes, X.ctypes, Y.ctypes, Mh.ctypes, Ch.ctypes, fx.ctypes,
                             fy.ctypes, jb.ctypes, bias.ctypes, zbuf.ctypes, fctr.ctypes, acc.ctypes)
        tim[TI_KERNEL] += _c_now() - ta
        if nth != ip[2]:
            raise RuntimeError("the OpenMP team size differs from the requested thread count (or a bad request)")
        ictr[I_NFE] += N * (end - step)
        tim[TI_NSEG] += 1.0
        sl = end - 1
        # --- uniform FR on the post-move positions (gateway_ladder_numba._advance, diagnostics off) ---
        if do_fr and (sl % fe) == 0:
            ta = _c_now()
            if ramp_steps > 0:
                g = gamma * (1.0 - math.exp(-sl / ramp_steps))
            else:
                g = gamma
            gl.kde_density(X, N, kern, r_eta, dx, p, hist)
            gl.uniform_scores(X, N, p, dx, score_clip, Sc)
            kd, kc, nu = gl.fr_resample(Sc, N, g, dt_fr, cap, sel, die_c, clone_c, pool, ubuf, kopp * ublk, False)
            kopp += 1
            ictr[I_NOPP] += 1
            ev_hist[kd + kc] += 1
            nde = 0
            if kd + kc > 0:
                ictr[I_NOPP_EV] += 1
                ictr[I_KD] += kd
                ictr[I_KC] += kc
                for i in range(N):
                    tmpx[i] = X[sel[i]]
                    tmpy[i] = Y[sel[i]]
                for i in range(N):
                    X[i] = tmpx[i]
                    Y[i] = tmpy[i]
                for i in range(N):
                    ocnt[i] = 0
                for i in range(N):
                    ocnt[sel[i]] += 1
                for i in range(N):
                    if ocnt[i] == 0:
                        nde += 1
                ictr[I_DEATHS] += nde
                if nde > 0:
                    ictr[I_NOPP_D] += 1
            if nde > cap:
                raise RuntimeError("realised deaths exceed the cap at one FR opportunity")
            dh_hist[nde] += 1
            tim[TI_FR] += _c_now() - ta
            tim[TI_NFR] += 1.0
        # --- save after `end` completed steps ---
        if sptr < n_saves and save_at[sptr] == end:
            ta = _c_now()
            _record_save(sptr, Mh, Ch, ictr, ev_hist, dh_hist, o_M, o_C, o_fr, o_evh, o_dth, o_nfe, o_wall, ta - t0)
            sptr += 1
            tim[TI_SAVE] += _c_now() - ta
        step = end
    ictr[I_SPTR] = sptr
    ictr[I_STEP] = c1
    return kopp


# ---------------------------------------------------------------------------------------------------------------
# one run of the code path (used for the warm-up and the timed run)
# ---------------------------------------------------------------------------------------------------------------
def _core(c, kn, N, seed, n_steps, save_at, x0, y0, is_fr, noise_seed, key, threads, fill, timers, zext=None,
          chunk_opps=CHUNK_OPPS, variant=0, sample_log2=TIMER_SAMPLE_LOG2):
    lib = load_kernel()
    nb = int(c["nb"])
    h = float(c["h"])
    dx = gl.grid_dx()
    kern, r_eta = gl.gaussian_kernel_np(float(c["eta"]), dx)
    cap, L, fe = int(kn["cap"]), int(kn["fr_block"]), int(kn["fr_every"])
    do_fr = bool(is_fr and N >= 2)
    S = len(save_at)
    X = _aligned_zeros(N)
    X[:] = x0
    Y = _aligned_zeros(N)
    Y[:] = y0
    Mh = _aligned_zeros(nb)
    Ch = _aligned_zeros(nb)
    fx = _aligned_zeros(2 * N)                 # step-parity double buffers (kernel)
    fy = _aligned_zeros(N)
    jb = _aligned_zeros(2 * N, np.int32)
    bias = _aligned_zeros(N)
    ictr = np.zeros(N_ICTR, dtype=np.int64)
    fctr = np.zeros(1)
    ev_hist = np.zeros(cap + 1, dtype=np.int64)
    dh_hist = np.zeros(cap + 1, dtype=np.int64)
    o_M = np.zeros((S, nb))
    o_C = np.zeros((S, nb))
    o_fr = np.zeros((S, 6), dtype=np.int64)
    o_evh = np.zeros((S, cap + 1), dtype=np.int64)
    o_dth = np.zeros((S, cap + 1), dtype=np.int64)
    o_nfe = np.zeros(S, dtype=np.int64)
    o_wall = np.full(S, np.nan)
    p = np.zeros(gl.N_GRID)
    hist = np.zeros(gl.N_GRID)
    Sc = np.zeros(N)
    sel = np.zeros(N, dtype=np.int64)
    die_c = np.zeros(N, dtype=np.int64)
    clone_c = np.zeros(N, dtype=np.int64)
    pool = np.zeros(2 * N, dtype=np.int64)
    tmpx = np.zeros(N)
    tmpy = np.zeros(N)
    ocnt = np.zeros(N, dtype=np.int64)
    mode = 0 if fill in (0, 2) else 1
    # kernel timer mode (gw_kernel.c ip[6]) m = 1 + sample_log2: m = 1 every step; m >= 2 sampling gaps uniform on
    # [1, 2^m] (mean ~2^(m-1) = 2^sample_log2)
    tmode = (1 + int(sample_log2)) if timers else 0
    ip = np.array([N, nb, int(threads), 0, 0, mode, tmode, int(variant)], dtype=np.int64)
    dp = np.array([h, float(c["beta"]), float(c["H"]), float(c["omega_out"]), float(c["omega_in"]), float(c["s"]),
                   float(c["min_count"]), 0.0])
    keyarr = np.ascontiguousarray(key, dtype=np.uint64).reshape(2)
    acc = np.zeros(N_ACC)
    tim = np.zeros(N_TI)
    if fill == 0:
        max_seg = max(1, REPLAY_BUF_DOUBLES // (2 * N))
        zbuf = np.zeros(max_seg * 2 * N)
    else:
        max_seg = int(n_steps)
        zbuf = np.zeros(1)
    if zext is None:
        zext = np.zeros(1)
    else:
        zext = np.ascontiguousarray(zext, dtype=np.float64).ravel()
        if zext.size != 2 * N * int(n_steps):
            raise ValueError("zext must hold 2 N n_steps normals")
    empty_u = np.zeros(0)
    fr_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(gl.fr_seed_entropy(seed, N))))
    if fill == 0:
        gl.check_mt_layout()
        gl._mt_seed(int(noise_seed))
    t_fr_py = 0.0
    t_save_py = 0.0
    n_chunks = 0
    # ------------------------------------------------------------- timed start
    t0 = lib.gw_now()
    tsc0 = lib.gw_rdtsc()
    ta = lib.gw_now()
    while ictr[I_SPTR] < S and save_at[ictr[I_SPTR]] == 0:          # saves of the initial state
        _record_save(ictr[I_SPTR], Mh, Ch, ictr, ev_hist, dh_hist, o_M, o_C, o_fr, o_evh, o_dth, o_nfe, o_wall,
                     lib.gw_now() - t0)
        ictr[I_SPTR] += 1
    t_save_py += lib.gw_now() - ta
    step = 0
    while step < n_steps:
        if do_fr:
            c1 = min(int(n_steps), step + chunk_opps * fe)
            n_opp = -(-c1 // fe) - (-(-step // fe))                    # FR steps in [step, c1)
            ta = lib.gw_now()
            ubuf = fr_rng.random(n_opp * L) if n_opp > 0 else empty_u
            t_fr_py += lib.gw_now() - ta
        else:
            c1 = int(n_steps)
            n_opp = 0
            ubuf = empty_u
        used = _drive_chunk(step, c1, N, fe, do_fr, float(c["gamma"]), float(kn["ramp_steps"]),
                            float(c["score_clip"]), cap, kern, int(r_eta), float(dx), float(kn["dt_fr"]), ubuf, L,
                            save_at, max_seg, int(fill), zext, t0,
                            ip, dp, keyarr, X, Y, Mh, Ch, fx, fy, jb, bias, zbuf, fctr, acc,
                            ictr, ev_hist, dh_hist, o_M, o_C, o_fr, o_evh, o_dth, o_nfe, o_wall,
                            p, hist, Sc, sel, die_c, clone_c, pool, tmpx, tmpy, ocnt, tim)
        if used != n_opp:
            raise RuntimeError(f"FR opportunities used {used} != drawn blocks {n_opp} in chunk [{step}, {c1})")
        if fill == 0 and gl.mt_has_gauss() != 0:
            raise RuntimeError("numba MT holds a cached gaussian at a chunk boundary (odd normal count)")
        step = c1
        n_chunks += 1
    t_end = lib.gw_now()
    tsc_end = lib.gw_rdtsc()
    # ------------------------------------------------------------- timed end
    if int(ictr[I_SPTR]) != S or int(ictr[I_NFE]) != N * int(n_steps) or int(ictr[I_STEP]) != int(n_steps):
        raise RuntimeError("run incomplete: not every save fired or the force count is wrong")
    return dict(X=X.copy(), Y=Y.copy(), ictr=ictr, fctr=fctr, ev_hist=ev_hist, dh_hist=dh_hist, o_M=o_M, o_C=o_C,
                o_fr=o_fr, o_evh=o_evh, o_dth=o_dth, o_nfe=o_nfe, o_wall=o_wall, acc=acc, tim=tim, t0=t0, t_end=t_end,
                tsc0=tsc0, tsc_end=tsc_end, t_fr_py=t_fr_py, t_save_py=t_save_py, n_chunks=n_chunks,
                max_seg=max_seg)


def _check_model(cfg):
    for k in ("alpha", "lam"):
        if cfg is not None and k in cfg and float(cfg[k]) != 1.0:
            raise ValueError(f"the gateway parallel engine implements MODEL_ORIG only ({k} = {cfg[k]} != 1)")
    if cfg is not None and cfg.get("model") not in (None, "orig", "MODEL_ORIG"):
        raise ValueError(f"the gateway parallel engine implements MODEL_ORIG only (model = {cfg.get('model')})")


def _peak_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def run_arm_parallel(cfg, method, N, seed, n_steps, save_steps, *, threads, noise, timers=True, warmup=True,
                     variant=None, _noise_ext=None, _timer_sample_log2=None):
    """Run ONE arm (method 'abf' | 'fr') of the gateway (N, seed) job for n_steps loop steps on `threads` OpenMP
    threads with `noise` 'replay' | 'counter'; returns the result dict (module docstring).  warmup=False skips the
    warm-up run (tests only: the timers then include JIT compilation).  variant: kernel variant name (VARIANTS;
    default DEFAULT_VARIANT = env GW_PARALLEL_VARIANT or 'replicated_dissem', Amendment 2); every variant computes bitwise the same
    result.  _noise_ext (test hook, replay layout (n_steps, 2, N)) replaces the replay stream by the given
    normals.  _timer_sample_log2 (development / tests): timer sampling with mean gap ~2^k steps (default
    TIMER_SAMPLE_LOG2; 0 = phase reads on every step, the full-cost instrumentation)."""
    t_entry = time.monotonic()            # CLOCK_MONOTONIC, the clock of gw_now()
    lib = load_kernel()
    if noise not in NOISE_MODES:
        raise ValueError("noise must be 'replay' or 'counter'")
    threads = int(threads)
    variant = DEFAULT_VARIANT if variant is None else variant
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
    if threads < 1 or threads > lib.gw_max_threads():
        raise ValueError(f"threads must be in [1, {lib.gw_max_threads()}]")
    slog2 = TIMER_SAMPLE_LOG2 if _timer_sample_log2 is None else int(_timer_sample_log2)
    if not 0 <= slog2 <= 30:
        raise ValueError("_timer_sample_log2 must be in [0, 30]")
    _check_model(cfg)
    clean_cfg = {k: v for k, v in (cfg or {}).items() if k not in ("alpha", "lam", "model")}
    P = gl._prepare(clean_cfg, method, N, seed, n_steps, save_steps, None, None, False, None, None, False)
    c, kn, h, N, seed, n_steps, is_fr = P["c"], P["kn"], P["h"], P["N"], P["seed"], P["n_steps"], P["is_fr"]
    save_at, x0, y0, noise_seed = P["save_at"], P["x0"], P["y0"], P["noise_seed"]
    key = counter_key(seed, N)
    fill = NOISE_MODES[noise]
    if _noise_ext is not None:
        if noise != "replay":
            raise ValueError("_noise_ext needs noise='replay'")
        fill = 2
    binding = omp_binding(threads)
    if warmup:
        fe = int(kn["fr_every"])
        n_w = int(min(n_steps, (2 * fe + 2) if is_fr else 512))
        zw = None if _noise_ext is None else np.ascontiguousarray(_noise_ext).reshape(-1)[: 2 * N * n_w]
        _core(c, kn, N, seed, n_w, np.array([n_w], dtype=np.int64), x0, y0, is_fr, noise_seed, key, threads, fill,
              timers, zext=zw, variant=VARIANTS[variant], sample_log2=slog2)
    R = _core(c, kn, N, seed, n_steps, save_at, x0, y0, is_fr, noise_seed, key, threads, fill, timers,
              zext=_noise_ext, variant=VARIANTS[variant], sample_log2=slog2)
    wall_total = R["t_end"] - R["t0"]
    freq = (R["tsc_end"] - R["tsc0"]) / wall_total if wall_total > 0 else float("nan")
    acc, tim = R["acc"], R["tim"]
    nan = float("nan")
    t_fm = t_dep = t_bias = t_sync = t_region = t_loop = t_exact_sum = nan
    n_sampled = int(acc[10])
    if timers:
        # exact parts: entry force/bin acc[0], launch + join wait acc[1], step loop acc[2], table copies acc[11];
        # the step loop is split over the phases in proportion to the sampled phase ticks acc[3..6]
        s_tot = acc[3] + acc[4] + acc[5] + acc[6]
        if s_tot > 0 and n_sampled > 0:
            sh_fm, sh_dep, sh_bias, sh_sync = acc[3] / s_tot, acc[4] / s_tot, acc[5] / s_tot, acc[6] / s_tot
            t_fm = (acc[0] + acc[2] * sh_fm) / freq
            t_dep = (acc[11] + acc[2] * sh_dep) / freq
            t_bias = (acc[2] * sh_bias) / freq
            t_sync = (acc[1] + acc[2] * sh_sync) / freq
        t_region = acc[7] / freq
        t_loop = acc[2] / freq
        t_exact_sum = (acc[0] + acc[1] + acc[2] + acc[11]) / freq
    t_fr = float(tim[TI_FR] + R["t_fr_py"])
    t_save = float(tim[TI_SAVE] + R["t_save_py"])
    t_kernel = float(tim[TI_KERNEL])
    t_replay = float(tim[TI_REPLAY])
    n_seg = int(tim[TI_NSEG])
    driver_resid = wall_total - t_kernel - t_fr - t_save - t_replay
    o_fr = R["o_fr"]
    ic = R["ictr"]
    sig = dict(engine=ENGINE, cfg=c, method=method, N=N, seed=seed, n_steps=n_steps, noise=noise, threads=threads,
               kernel_variant=variant,
               noise_seed=noise_seed, counter_key=[int(v) for v in key], fr_entropy=gl.fr_seed_entropy(seed, N),
               save_sha=hashlib.sha256(np.ascontiguousarray(save_at).tobytes()).hexdigest(),
               x0_sha=hashlib.sha256(x0.tobytes()).hexdigest(), y0_sha=hashlib.sha256(y0.tobytes()).hexdigest(),
               sources_sha256=_PROVENANCE["sources_sha256"], validated_sha256=_PROVENANCE["validated_sha256"],
               timers=bool(timers), timer_sample_log2=int(slog2))
    sig = json.loads(json.dumps(sig, sort_keys=True))
    meta = dict(
        engine=ENGINE, system="gateway", model="MODEL_ORIG (alpha = 1, lam = 1)", method=method, N=N, seed=seed,
        n_steps=n_steps, h=h, B=N * n_steps, threads=threads, noise_mode=noise, test_noise_ext=_noise_ext is not None,
        kernel_variant=variant, kernel_variant_code=VARIANTS[variant],
        noise_replay=("numba MT19937 seeded with default_noise_seed(seed, N) = %d via gateway_ladder_numba._mt_seed; "
                      "2N standard normals per step, zx[0..N-1] then zy[0..N-1], drawn serially by the driver"
                      % noise_seed),
        noise_counter=("Philox4x64-10 (Random123), key SeedSequence([seed, N, 2]).generate_state(2, uint64) = "
                       "[%d, %d]; counter (loop step, slot, 0, 0); u = (o >> 11) 2^-53; "
                       "(zx, zy) = sqrt(-2 log(1 - u0)) (cos, sin)(2 pi u1); loop step = completed steps before the "
                       "move" % (int(key[0]), int(key[1]))),
        fr_stream="numpy Generator(PCG64(SeedSequence([seed, N, 1]))); opportunity k owns doubles [kL, (k+1)L)",
        fr_block_L=int(kn["fr_block"]), cap=int(kn["cap"]), fr_every=int(kn["fr_every"]),
        ramp_steps=float(kn["ramp_steps"]), dt_fr=float(kn["dt_fr"]),
        accumulator_convention=("pre-deposit (gateway_ladder_numba): the save at completed-step count k holds the "
                                "deposits of the states 0..k-1 (C_all[k].sum() = N k)"),
        force_eval_convention="N per completed step: n_force_evals_at_save = N k, n_force_evals = N n_steps",
        reduction_order=("per bin, walkers added in slot order (bin-owner scan, or every thread into its private "
                         "copy for the replicated variant): thread-count invariant"),
        wall_definition=("CLOCK_MONOTONIC seconds from the timed start t0 (after build, JIT, warm-up run of the same "
                         "code path incl. thread-pool start, and state set-up)"),
        timers=bool(timers),
        timer_definitions=dict(
            t_force_move_s=("in-kernel force/bin + move (+ counter-mode noise), team mean of per-thread rdtsc ticks: "
                            "exact entry force/bin + the step loop's sampled share"),
            t_abf_s=("in-kernel deposit + bias read (incl. running max |bias|) + replicated-table copies, team "
                     "mean: exact copies + the step loop's sampled share"),
            t_sync_s=("exact region launch wait + join wait + the step loop's sampled share of barrier waits "
                      "(2 per step owner, 1 per step replicated), team mean"),
            t_fr_s="driver FR (KDE, scores, resampling, gather, counters) + Python PCG64 block draws",
            t_save_s="driver saves (copy of M, C, counters)",
            sampling=("phase reads inside the step loop only on the sampled steps s_0 = 0, s_(i+1) = s_i + 1 + "
                      "(mix64(s_i) >> (63 - k)), mix64 = splitmix64 finaliser of (s + 1) * 0x9E3779B97F4A7C15, "
                      "k = timer_sample_log2 (gaps ~uniform on [1, 2^(k+1)], mean ~2^k; every step if k = 0); "
                      "untimed code between them; the exact "
                      "step-loop ticks are split in proportion to the sampled phase ticks (ratio estimator); "
                      "t_force_move + t_abf + t_sync = the exact region wall (team mean)")),
        timer_sample_log2=int(slog2), timer_sample_mean_gap_steps=(1.0 if slog2 == 0 else (2 ** (slog2 + 1) + 1) / 2),
        timers_extra=dict(t_deposit_s=t_dep, t_bias_read_s=t_bias, t_region_wall_s=t_region,
                          t_step_loop_s=t_loop, t_exact_parts_sum_s=t_exact_sum,
                          n_sampled_steps=n_sampled, n_loop_steps=int(acc[9]),
                          t_kernel_calls_wall_s=t_kernel, t_replay_fill_s=t_replay, t_driver_residual_s=driver_resid,
                          driver_overhead_per_segment_s=(driver_resid / n_seg if n_seg else float("nan")),
                          n_segments=n_seg, n_kernel_calls=int(acc[8]), n_fr_calls=int(tim[TI_NFR]),
                          n_chunks=int(R["n_chunks"]), replay_max_segment_steps=int(R["max_seg"]),
                          tsc_hz=freq),
        warmup=bool(warmup), peak_rss_mb=_peak_rss_mb(),
        omp=dict(env_set_by_module=os.environ.get("GW_PARALLEL_KEEP_OMP_ENV") != "1",
                 env_before_import=_OMP_ENV_BEFORE,
                 env={k: os.environ.get(k) for k in ("OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES", "OMP_DYNAMIC",
                                                     "GOMP_SPINCOUNT")},
                 libgomp_loaded_before_import=_LIBGOMP_PRELOADED, binding=binding),
        build=_BUILD_INFO, git_commit=_PROVENANCE["git_commit"], git_dirty=_PROVENANCE["git_dirty"],
        git_status=_PROVENANCE["git_status"], sources_sha256=_PROVENANCE["sources_sha256"],
        validated_engine_sha256=_PROVENANCE["validated_sha256"], validated_engine=gl.ENGINE_VERSION,
        numba=numba.__version__, numpy=np.__version__, python=platform.python_version(), host=platform.node(),
        cpus=allocated_cpus(), cpus_source=("os.sched_getaffinity(0) at import, before libgomp load (the process "
                                            "allocation)"),
        cpus_master_bound=sorted(os.sched_getaffinity(0)),
        cpus_omp_places_union=sorted({c for pl in binding["places"] for c in pl}),
        run_signature=sig, run_signature_sha256=hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest(),
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), status="complete",
    )
    res = dict(
        save_step=save_at.copy(), save_t=save_at * h, save_u=save_at / float(n_steps),
        M_all=R["o_M"], C_all=R["o_C"],
        fr_deaths_cum=o_fr[:, 0], fr_kd_cum=o_fr[:, 1], fr_kc_cum=o_fr[:, 2], fr_opp_cum=o_fr[:, 3],
        fr_opp_event_cum=o_fr[:, 4], fr_opp_death_cum=o_fr[:, 5],
        fr_events_hist=R["ev_hist"], fr_events_hist_cum=R["o_evh"],
        fr_deaths_hist=R["dh_hist"], fr_deaths_hist_cum=R["o_dth"],
        X_final=R["X"], Y_final=R["Y"], max_abs_bias=np.float64(R["fctr"][0]), n_fr_steps=np.int64(ic[I_NOPP]),
        n_force_evals=np.int64(ic[I_NFE]),
        cfg=c,
        wall_s_at_save=R["o_wall"], n_force_evals_at_save=R["o_nfe"],
        wall_total_s=float(wall_total), warmup_s=float(R["t0"] - t_entry),
        t_force_move_s=float(t_fm), t_abf_s=float(t_dep + t_bias), t_sync_s=float(t_sync), t_fr_s=t_fr,
        t_save_s=t_save, threads=threads, noise_mode=noise, engine=ENGINE,
        meta=json.dumps(meta, sort_keys=True, default=_json_default),
    )
    return res


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ---------------------------------------------------------------------------------------------------------------
# comparison helpers (V-B1 / V-B2 / V-B5)
# ---------------------------------------------------------------------------------------------------------------
TIMING_KEYS = ("wall_s_at_save", "wall_total_s", "warmup_s", "t_force_move_s", "t_abf_s", "t_sync_s", "t_fr_s",
               "t_save_s", "threads", "meta")


def shared_keys(ref):
    """Keys of a validated diagnostics-off result that the parallel result must reproduce bitwise."""
    return sorted(k for k in ref if k not in WALL_KEYS)


def science_keys(res):
    """Keys of a parallel result that are not timing / labels (noise_mode, engine): what two runs of the same
    dynamics must share bitwise."""
    return sorted(k for k in res if k not in TIMING_KEYS and k not in ("noise_mode", "engine"))


def _diff_record(k, a, b):
    if isinstance(a, dict) or isinstance(b, dict):
        return dict(key=k, equal=json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True))
    a = np.asarray(a)
    b = np.asarray(b)
    rec = dict(key=k, dtype_a=str(a.dtype), dtype_b=str(b.dtype), shape_a=list(a.shape), shape_b=list(b.shape))
    same_kind = a.dtype == b.dtype and a.shape == b.shape
    eq = bool(same_kind and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())   # bitwise
    rec["equal"] = eq
    if a.dtype.kind in "USOb" or b.dtype.kind in "USOb":
        return rec
    if not eq and a.shape == b.shape and a.size:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        rec["max_abs_diff"] = float(np.nanmax(d)) if np.isfinite(d).any() else None
        rec["n_diff"] = int(np.count_nonzero(a != b))
        if a.ndim >= 1:
            rows = np.nonzero((a != b).reshape(a.shape[0], -1).any(axis=1))[0]
            rec["first_diff_index"] = int(rows[0]) if rows.size else None
    return rec


def compare(ref, par, keys):
    """Per-key bitwise comparison records; returns (all_equal, records)."""
    recs = []
    for k in keys:
        if k not in par:
            recs.append(dict(key=k, equal=False, missing_in_parallel=True))
            continue
        recs.append(_diff_record(k, ref[k], par[k]))
    return all(r["equal"] for r in recs), recs


def production_cfg():
    with open(os.path.join(ROOT, "configs", "equal_budget_v2", "gateway_production.json")) as f:
        return json.load(f)["engine_cfg"]


def default_vb1_steps(N, h):
    """Prereg V-B1 run length: >= 12 t.u. at N 16, >= 6 t.u. at N 64 and 512."""
    t = 12.0 if int(N) <= 16 else 6.0
    return int(round(t / h))


def _arm_summary(res):
    return dict(n_saves=int(len(res["save_step"])), realised_deaths=int(res["fr_deaths_cum"][-1]),
                fr_opportunities=int(res["fr_opp_cum"][-1]), fr_candidate_events=int(res["fr_opp_event_cum"][-1]),
                opportunities_with_death=int(res["fr_opp_death_cum"][-1]), n_force_evals=int(res["n_force_evals"]),
                max_abs_bias=float(res["max_abs_bias"]))


def vb1(N, method, threads_list, seed=8600, n_steps=None, cfg=None, save_steps=None, warmup=True, variant=None):
    """V-B1: replay mode bitwise vs gateway_ladder_numba.run_arm(..., diagnostics=False) at each thread count."""
    cfg = production_cfg() if cfg is None else cfg
    h = float(gl.full_cfg(cfg)["h"])
    n_steps = default_vb1_steps(N, h) if n_steps is None else int(n_steps)
    save_at = gl.budget_save_grid(n_steps)[0] if save_steps is None else np.asarray(save_steps, dtype=np.int64)
    t = time.monotonic()
    ref = gl.run_arm(cfg, method, N, seed, n_steps, save_at, diagnostics=False)
    wall_ref = time.monotonic() - t
    keys = shared_keys(ref)
    out = dict(test="V-B1 replay identity", engine=ENGINE, validated_engine=gl.ENGINE_VERSION, N=int(N),
               kernel_variant=variant or DEFAULT_VARIANT,
               method=method, seed=int(seed), n_steps=n_steps, T_tu=n_steps * h, h=h, keys_compared=keys,
               validated=dict(_arm_summary(ref), wall_s=wall_ref), runs=[])
    ok_all = True
    for T in threads_list:
        par = run_arm_parallel(cfg, method, N, seed, n_steps, save_at, threads=int(T), noise="replay",
                               warmup=warmup, variant=variant)
        eq, recs = compare(ref, par, keys)
        ok_all &= eq
        out["runs"].append(dict(threads=int(T), identical=eq, mismatches=[r for r in recs if not r["equal"]],
                                wall_total_s=par["wall_total_s"],
                                omp_binding=json.loads(par["meta"])["omp"]["binding"]))
    s = out["validated"]
    length_ok = s["n_saves"] >= 20 and (method == "abf" or s["realised_deaths"] >= 50)
    out.update(identical_all=bool(ok_all), length_requirements_met=bool(length_ok),
               prereg_lengths=("gateway >= 12 t.u. at N 16, >= 6 t.u. at N 64/512; >= 20 saves; FR >= 50 realised "
                               "deaths"),
               verdict="PASS" if (ok_all and length_ok) else ("FAIL" if not ok_all else "IDENTICAL_BUT_SHORT"))
    return out


def vb2(N, method, threads_list, seed=8600, n_steps=None, cfg=None, save_steps=None, warmup=True, variant=None):
    """V-B2: counter mode bitwise identical over thread counts (every result array except timing/meta)."""
    cfg = production_cfg() if cfg is None else cfg
    h = float(gl.full_cfg(cfg)["h"])
    n_steps = default_vb1_steps(N, h) if n_steps is None else int(n_steps)
    save_at = gl.budget_save_grid(n_steps)[0] if save_steps is None else np.asarray(save_steps, dtype=np.int64)
    runs = {}
    for T in threads_list:
        runs[int(T)] = run_arm_parallel(cfg, method, N, seed, n_steps, save_at, threads=int(T), noise="counter",
                                        warmup=warmup, variant=variant)
    T0 = int(threads_list[0])
    keys = science_keys(runs[T0])
    out = dict(test="V-B2 counter-mode thread-count invariance", engine=ENGINE, N=int(N), method=method,
               kernel_variant=variant or DEFAULT_VARIANT,
               seed=int(seed), n_steps=n_steps, T_tu=n_steps * h, reference_threads=T0, keys_compared=keys,
               arm=_arm_summary(runs[T0]), runs=[])
    ok_all = True
    for T in threads_list[1:]:
        eq, recs = compare(runs[T0], runs[int(T)], keys)
        ok_all &= eq
        out["runs"].append(dict(threads=int(T), identical=eq, mismatches=[r for r in recs if not r["equal"]],
                                wall_total_s=runs[int(T)]["wall_total_s"]))
    out.update(identical_all=bool(ok_all), verdict="PASS" if ok_all else "FAIL")
    return out


VB5_PREREG_CRITERION = ("death: ABF and FR arms bitwise identical (M_all, C_all) at every save before FR's first "
                        "realised death (PREREGISTRATION.md section 2.3, V-B5, frozen wording)")
VB5_EVENT_CRITERION = ("event: bitwise identical at every save before FR's first opportunity with a candidate (first "
                       "gather); NOT preregistered -- gates only with a recorded amendment")
VB5_EXIT = {"PASS": 0, "FAIL": 1, "FAIL_PREREG_DEATH_CRITERION_PASS_EVENT_ONLY": 3}   # (2 = argparse usage error)


def vb5_verdict(out, criterion="death", amendment=None):
    """Gate verdict of a vb5() record.  criterion 'death' (default) = the preregistered wording: PASS iff the death
    criterion passes; if it fails while the event criterion passes the verdict is
    'FAIL_PREREG_DEATH_CRITERION_PASS_EVENT_ONLY' (exit status 3), never PASS.  criterion 'event' gates on the
    before-the-first-gather criterion and requires `amendment` (the dated preregistration amendment that adopts it).
    Returns dict(verdict, gating_criterion, amendment)."""
    if criterion not in ("death", "event"):
        raise ValueError("criterion must be 'death' or 'event'")
    if criterion == "event" and not (amendment and str(amendment).strip()):
        raise ValueError("gating V-B5 on the event criterion needs the amendment reference that adopts it")
    d_ok = out["verdict_death_criterion"] == "PASS"
    e_ok = out["verdict_event_criterion"] == "PASS"
    if criterion == "death":
        v = "PASS" if d_ok else ("FAIL_PREREG_DEATH_CRITERION_PASS_EVENT_ONLY" if e_ok else "FAIL")
        crit = VB5_PREREG_CRITERION
    else:
        v = "PASS" if e_ok else "FAIL"
        crit = VB5_EVENT_CRITERION
    return dict(verdict=v, gating_criterion=crit, gating_criterion_name=criterion,
                amendment=(str(amendment) if amendment else None))


def vb5(N, threads, seed=8600, n_steps=None, cfg=None, save_steps=None, warmup=True, variant=None, criterion="death",
        amendment=None):
    """V-B5: counter-mode ABF and FR arms of one (seed, N): bitwise identity of M_all, C_all at the early saves.

    Reports both criteria: (a) identity at every save with fr_deaths_cum == 0 -- the preregistration's frozen
    wording, 'before FR's first realised death', which is the GATING criterion ('verdict', see vb5_verdict) unless
    criterion='event' with an amendment reference; and (b) identity at every save with fr_opp_event_cum == 0
    (before the first opportunity with a candidate, i.e. the first gather) -- the validated engine's own pairing
    property (its test Q4), reported, not gating.  (a) can fail by construction when an opportunity with
    kd = 0 < kc gathers a pure slot permutation (no realised death) before the first death: the noise is per slot
    and the deposit order is the slot order; such a run reports 'FAIL_PREREG_DEATH_CRITERION_PASS_EVENT_ONLY'."""
    cfg = production_cfg() if cfg is None else cfg
    h = float(gl.full_cfg(cfg)["h"])
    n_steps = default_vb1_steps(N, h) if n_steps is None else int(n_steps)
    save_at = gl.budget_save_grid(n_steps)[0] if save_steps is None else np.asarray(save_steps, dtype=np.int64)
    a = run_arm_parallel(cfg, "abf", N, seed, n_steps, save_at, threads=threads, noise="counter", warmup=warmup,
                         variant=variant)
    f = run_arm_parallel(cfg, "fr", N, seed, n_steps, save_at, threads=threads, noise="counter", warmup=warmup,
                         variant=variant)

    def same(rows):
        return bool(np.array_equal(a["M_all"][rows], f["M_all"][rows]) and
                    np.array_equal(a["C_all"][rows], f["C_all"][rows]))

    pre_ev = f["fr_opp_event_cum"] == 0
    pre_d = f["fr_deaths_cum"] == 0
    first_ev = int(save_at[np.argmin(pre_ev)]) if (~pre_ev).any() else None
    first_d = int(save_at[np.argmin(pre_d)]) if (~pre_d).any() else None
    ok_ev = same(pre_ev)
    ok_d = same(pre_d)
    post_differs = bool((~pre_ev).any() and not same(~pre_ev))
    out = dict(test="V-B5 ABF/FR pairing (counter mode)", engine=ENGINE, kernel_variant=variant or DEFAULT_VARIANT,
               N=int(N), seed=int(seed), threads=int(threads),
               n_steps=n_steps, n_saves_before_first_event=int(pre_ev.sum()),
               n_saves_before_first_death=int(pre_d.sum()), first_save_after_first_event=first_ev,
               first_save_after_first_death=first_d, identical_before_first_event=ok_ev,
               identical_before_first_death=ok_d, differs_after_first_event=post_differs,
               fr_arm=_arm_summary(f),
               verdict_event_criterion="PASS" if (ok_ev and pre_ev.sum() >= 1) else "FAIL",
               verdict_death_criterion="PASS" if (ok_d and pre_d.sum() >= 1) else "FAIL")
    out.update(vb5_verdict(out, criterion, amendment))
    return out


def timer_overhead(Ns=(1, 2, 4, 8, 16), reps=5, walker_steps=8e6, threads=1, method="abf", seed=8605, cfg=None,
                   variant=None, max_overhead_pct=1.0):
    """Development check (NOT the feasibility benchmark): wall time per walker-step with timers on vs off, counter
    mode, `reps` interleaved repeats per N (on/off order alternating), medians; verdict PASS iff every median
    overhead <= max_overhead_pct.  Run on an otherwise idle, pinned core set."""
    cfg = production_cfg() if cfg is None else cfg
    rows = []
    for N in Ns:
        N = int(N)
        n = max(1, int(float(walker_steps) // N))
        sv = np.array([n], dtype=np.int64)
        w = {True: [], False: []}
        for r in range(int(reps)):
            for tm in ((True, False) if r % 2 == 0 else (False, True)):
                res = run_arm_parallel(cfg, method, N, seed, n, sv, threads=int(threads), noise="counter",
                                       timers=tm, variant=variant)
                w[tm].append(1e9 * res["wall_total_s"] / (N * n))
        on, off = float(np.median(w[True])), float(np.median(w[False]))
        rows.append(dict(N=N, n_steps=n, ns_per_walker_step_timers_on=on, ns_per_walker_step_timers_off=off,
                         overhead_pct=100.0 * (on / off - 1.0), runs_on=w[True], runs_off=w[False]))
    worst = max(r["overhead_pct"] for r in rows)
    return dict(label="development timer-overhead check (not a benchmark)", method=method, threads=int(threads),
                kernel_variant=variant or DEFAULT_VARIANT, timer_sample_log2=TIMER_SAMPLE_LOG2, reps=int(reps),
                walker_steps=float(walker_steps), rows=rows, worst_overhead_pct=worst,
                max_overhead_pct=float(max_overhead_pct),
                verdict="PASS" if worst <= float(max_overhead_pct) else "FAIL")


def probe(N, threads, n_steps, method="fr", noise="counter", seed=8600, timers=True, cfg=None, variant=None):
    """Development throughput probe (NOT the feasibility benchmark)."""
    cfg = production_cfg() if cfg is None else cfg
    res = run_arm_parallel(cfg, method, N, seed, n_steps, [n_steps], threads=threads, noise=noise, timers=timers,
                           variant=variant)
    m = json.loads(res["meta"])
    w = res["wall_total_s"]
    return dict(label="development probe only (not a benchmark)", kernel_variant=m["kernel_variant"], N=int(N),
                threads=int(threads),
                n_steps=int(n_steps), method=method, noise=noise, wall_total_s=w, us_per_step=1e6 * w / n_steps,
                walker_steps_per_s=N * n_steps / w, t_force_move_s=res["t_force_move_s"], t_abf_s=res["t_abf_s"],
                t_sync_s=res["t_sync_s"], t_fr_s=res["t_fr_s"], t_save_s=res["t_save_s"], warmup_s=res["warmup_s"],
                timers_extra=m["timers_extra"], team_cpus=m["omp"]["binding"]["team_cpus"],
                proc_bind_close=m["omp"]["binding"]["proc_bind_close"])


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="gateway walker-parallel engine: validation and probe CLI")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--vb1", action="store_true", help="V-B1 replay identity vs the validated engine")
    g.add_argument("--vb2", action="store_true", help="V-B2 counter-mode thread-count invariance")
    g.add_argument("--vb5", action="store_true", help="V-B5 counter-mode ABF/FR pairing")
    g.add_argument("--probe", nargs=3, type=int, metavar=("N", "THREADS", "STEPS"),
                   help="development throughput probe (not the feasibility benchmark)")
    g.add_argument("--overhead", metavar="N_LIST",
                   help="development timer-overhead check at these N (comma list; counter mode, timers on vs off)")
    ap.add_argument("--N", type=int)
    ap.add_argument("--method", choices=("abf", "fr"), default="fr")
    ap.add_argument("--threads", default=None,
                    help="comma list (V-B1 / V-B2; default 1,4,16); one value for V-B5 (default: the first of "
                         "1,4,16) and --overhead (default 1)")
    ap.add_argument("--seed", type=int, default=8600)
    ap.add_argument("--steps", type=int, default=None, help="loop steps (default: prereg V-B1 length)")
    ap.add_argument("--noise", choices=("counter", "replay"), default="counter", help="probe only")
    ap.add_argument("--no-timers", action="store_true", help="probe only")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default=None,
                    help="kernel variant (default: env GW_PARALLEL_VARIANT or owner_omp)")
    ap.add_argument("--vb5-criterion", choices=("death", "event"), default="death",
                    help="V-B5 gating criterion: 'death' = the preregistered wording (default); 'event' needs "
                         "--amendment")
    ap.add_argument("--amendment", default=None,
                    help="dated preregistration amendment adopting --vb5-criterion event (recorded in the JSON)")
    ap.add_argument("--reps", type=int, default=5, help="--overhead: interleaved repeats per N")
    ap.add_argument("--walker-steps", type=float, default=8e6, help="--overhead: walker-steps per run")
    ap.add_argument("--max-overhead-pct", type=float, default=1.0, help="--overhead: PASS threshold")
    ap.add_argument("--out", default=None, help="JSON verdict path (default: stdout)")
    a = ap.parse_args(argv)
    if a.vb5 and a.vb5_criterion == "event" and not (a.amendment and a.amendment.strip()):
        ap.error("--vb5-criterion event needs --amendment '<dated amendment reference>'")
    thr = [int(v) for v in str(a.threads or ("1" if a.overhead else "1,4,16")).split(",") if v.strip()]
    t0 = time.monotonic()
    if a.vb1:
        out = vb1(a.N, a.method, thr, seed=a.seed, n_steps=a.steps, variant=a.variant)
    elif a.vb2:
        out = vb2(a.N, a.method, thr, seed=a.seed, n_steps=a.steps, variant=a.variant)
    elif a.vb5:
        out = vb5(a.N, thr[0], seed=a.seed, n_steps=a.steps, variant=a.variant, criterion=a.vb5_criterion,
                  amendment=a.amendment)
    elif a.overhead:
        out = timer_overhead([int(v) for v in a.overhead.split(",") if v.strip()], reps=a.reps,
                             walker_steps=a.walker_steps, threads=thr[0],
                             variant=a.variant, max_overhead_pct=a.max_overhead_pct)
    else:
        N, T, n = a.probe
        out = probe(N, T, n, method=a.method, noise=a.noise, seed=a.seed, timers=not a.no_timers,
                    variant=a.variant)
    out["cli_wall_s"] = time.monotonic() - t0
    out["cpus"] = allocated_cpus()                       # the process allocation (import time, before libgomp)
    out["cpus_master_bound"] = sorted(os.sched_getaffinity(0))
    out["host"] = platform.node()
    out["git_commit"] = _PROVENANCE["git_commit"]
    out["sources_sha256"] = _PROVENANCE["sources_sha256"]
    out["kernel_build"] = _BUILD_INFO
    out["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    s = json.dumps(out, indent=1, sort_keys=True, default=_json_default)
    if a.out:
        d = os.path.dirname(os.path.abspath(a.out))
        os.makedirs(d, exist_ok=True)
        tmp = a.out + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(s)
        os.replace(tmp, a.out)
    print(s if not a.out else json.dumps({k: out[k] for k in out if k in ("verdict", "identical_all",
                                                                           "gating_criterion_name",
                                                                           "verdict_death_criterion",
                                                                           "verdict_event_criterion",
                                                                           "worst_overhead_pct",
                                                                           "us_per_step", "cli_wall_s")}))
    if a.probe:
        return 0
    return VB5_EXIT.get(out.get("verdict"), 1)        # every validation path sets 'verdict'; missing = FAIL


if __name__ == "__main__":
    sys.exit(_main())
