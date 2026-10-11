"""Walker-parallel CPU engine of the ethane/LTA 150 K histogram-ABF (+ uniform FR) sampler (Experiment B).

Final campaign, Experiment B (docs/final_campaign/PREREGISTRATION.md section 2.2-2.3; engineering brief
docs/final_campaign/EXPB_BACKEND_DESIGN.md).  The SAME algorithm as the validated serial engine
``src/lta_ladder_numba.py`` (``_run_chunk`` with diagnostics off, native FR mode 0), whose numba functions
``_mol_force``, ``_wrap``, ``_bin``, ``_fr_score`` and ``fr_select_native`` are imported and called unchanged;
that file is never modified.

Per evaluated step k (identical arithmetic and order to the validated kernel)
-----------------------------------------------------------------------------
 (1) forces / CV / local mean force / bin of every walker at the pre-move positions   [PARALLEL over walkers]
 (2) per-step staged bin sums in slot order ('touched' order), added to the all-steps and (k >= burn-in)
     production accumulators                                                            [serial, slot order]
 (3) bias = ramp clip(M_j / max(C_j, 1), +-60) pi / a from the all-steps accumulators incl. step k  [serial]
 (5) save at save_steps[sp] == k (state at the START of step k, accumulators incl. step k)        [serial]
 (6) Euler-Maruyama move with the [step][slot][bead][xyz] noise                         [PARALLEL over walkers]
 (7) FR at nxt = k + 1 >= fr_start, (nxt - fr_start) % fr_every == 0, on the post-move CV: the validated
     _fr_score and fr_select_native with the opportunity's block of N + 2 cap uniforms (child 2 of
     SeedSequence([seed, N, 150])), whole-molecule copies                                          [serial]

Parallel decomposition (one numba ``parallel=True`` kernel, threading layer omp = libgomp)
-----------------------------------------------------------------------------------------
* A parallel phase is ``prange`` over the T threads, thread t owning the static contiguous walker block
  [t N // T, (t + 1) N // T) (so per-thread busy time can be timed); every walker's arithmetic is the serial
  code, hence the result is bitwise independent of T and of the partition.
* FUSION: the move of step k and the force evaluation of step k + 1 of the same walker run in ONE parallel phase
  (one launch per step).  At an FR step the fused phase evaluates the forces at the post-move positions, the CV /
  bin of that evaluation are exactly the validated FR's post-move CV, and after the FR copies a clone's force,
  CV, local force and bin are copied from its source -- bitwise what a re-evaluation at the copied position gives,
  because the evaluation is a pure function of the walker's coordinates and sources never die.  The last step of
  a kernel call is unfused (the next call starts with an evaluation phase).
* Deposits are staged serially in slot order exactly as the validated kernel; the bias read, the save and FR
  are serial (they are O(N) / O(n_grid^2) per step against O(N x 768 pair terms) of force work).

Noise
-----
* ``noise='replay'``: child 1 of SeedSequence([seed, N, 150]).spawn(3) (PCG64), ``standard_normal((n_moves, N, 2,
  3))`` per chunk in Python exactly as lta_ladder_numba.run_arm draws it (the stream is chunking invariant), so
  every shared result is BITWISE the validated engine run with diagnostics=False, at every thread count (V-B1).
* ``noise='counter'``: in-kernel Philox4x64-10 (``parallel.philox``), key SeedSequence([seed, N, 150, 2]),
  counter (global step of the move, slot, word in {0, 1}, 0); Box-Muller; first 6 of 8 normals in [bead][xyz]
  order.  Results are bitwise independent of the thread count (V-B2) and of the chunking.
* Initial conditions (child 0) and the FR uniforms (child 2, fixed N + 2 cap block per opportunity) are the
  validated engine's in both modes, so the ABF and FR arms of one (seed, N) share IC and noise slot by slot.

Timing (warm-up excluded)
-------------------------
``warmup_s`` = JIT / cache load + OpenMP thread-pool start (a 5-step dummy run of the kernel with the requested
thread count).  The timed start is after it; ``wall_s_at_save`` are seconds since the timed start (clock_gettime
CLOCK_MONOTONIC from inside the kernel, the clock of time.perf_counter), ``wall_total_s`` the timed loop.  With
``timers=True`` the kernel also stamps its phases:
  t_force_move_s  mean over threads of the time spent in walker work inside the parallel phases
  t_sync_s        parallel-phase wall minus that mean (launch + join + load-imbalance wait; best effort)
  t_abf_s         deposit staging + accumulator update + bias read (serial)
  t_fr_s          FR score + selection + copies (kernel) + FR uniform draws (Python)
  t_save_s        saves (kernel)
  t_noise_s       replay-noise draws in Python (0 in counter mode)
  t_other_s       wall_total_s minus all of the above (driver, dispatch)
Each stamp costs ~30 ns (a handful per step, two per thread per parallel phase); with timers=False the phase timers
are NaN and only the per-save wall stamps are taken.

Compiled-code freshness
-----------------------
numba keys a cached function only by the file that defines it, but this module's machine code also contains
philox.py's and lta_ladder_numba.py's functions.  Every function here is therefore cached with a source stamp that
includes the sha256 of every file in _SOURCES (``_jit``), so an edit to any of them recompiles this module; and the
first run_arm_parallel call of a process runs ``kernel_selfcheck`` (replay == lta_ladder_numba.run_arm and counter ==
replay fed with parallel.philox.lta_noise_block on a 16-walker, 200-step FR job), raising StaleKernelError on a
mismatch.  meta["numba_cache"] and meta["selfcheck"] record both.

Public API
----------
  res = run_arm_parallel(cfg, method, N, seed, n_steps, save_steps, threads=t, noise='counter'|'replay')
  SHARED_KEYS (bitwise-identical to lta_ladder_numba.run_arm with diagnostics=False in replay mode), COST_KEYS
  kernel_selfcheck(), cache_info(), first_step_forces() (V-B1 fallback evidence from the kernel's own forces),
  counter_map_check() (kernel counter noise == parallel.philox.lta_noise_block over a whole run)
CLI (python -m parallel.lta_parallel with PYTHONPATH=src): --vb1, --vb2, --vb3, --probe; see ``main``.
Tests: tests/test_parallel_lta.py.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import numba
from numba import njit, prange, types
from numba.core import cgutils
from numba.extending import intrinsic
from llvmlite import ir

numba.config.THREADING_LAYER = "omp"

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
ROOT = os.path.dirname(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import lta_ladder_numba as E  # noqa: E402
from lta_ladder_numba import (_mol_force, _wrap, _bin, _fr_score, fr_select_native,  # noqa: E402
                              F_DT, F_NS, F_CPHI, F_CF, F_GPX, F_DPHI, F_FCLIP, F_DCLIP, F_RATE, F_DTEFF, F_SCLIP,
                              F_BSCALE, F_GRID0, F_QV, F_LQ,
                              I_N, I_NG, I_NSTEPS, I_WARM, I_BURN, I_FRSTART, I_FREVERY, I_CAP, I_ISFR, I_FRON,
                              S_STEP, S_SP, S_REPL, S_NOPP, S_NOPPEV, S_NSCORE, S_NFE, NIST,
                              A_FS, A_CS, A_FSP, A_CSP, A_BH, A_DH, NA, G_SSTD, G_SMAX, NFST)

try:  # package import (python -m parallel.lta_parallel, or `import parallel.lta_parallel`)
    from . import philox as PX  # noqa: E402
except ImportError:  # executed as a script
    from parallel import philox as PX  # noqa: E402
from parallel.philox import lta_normals6  # noqa: E402

ENGINE = "lta_parallel/1"
NOISE_MODES = ("counter", "replay")
PAD = 8                                     # busy-time slots one cache line apart
T_PAR, T_ABF, T_SAVE, T_FR, T_NPAR = range(5)
NT = 5

# keys of the validated engine's diagnostics=False result that this engine reproduces (bitwise in replay mode)
SHARED_KEYS = ("save_step", "save_t", "save_u", "M_all", "C_all", "M_prod", "C_prod", "cum_deaths", "cum_opps",
               "cum_opps_with_event", "events_per_opp_hist", "q_final", "anc_final", "M_all_final", "C_all_final",
               "M_prod_final", "C_prod_final", "birth_hist", "death_hist", "total_replacement_events", "n_fr_opps",
               "fr_score_std", "fr_score_absmax", "n_force_evals", "cfg_json")
# keys that legitimately differ between two executions of one job
COST_KEYS = ("wall_s_at_save", "wall_total_s", "warmup_s", "t_force_move_s", "t_abf_s", "t_sync_s", "t_fr_s",
             "t_save_s", "t_noise_s", "t_other_s", "peak_rss_mb", "meta", "threads", "n_parallel_regions",
             "busy_per_thread_s")
# integer outputs (the V-B1 fallback compares these exactly)
INTEGER_KEYS = ("save_step", "C_all", "C_prod", "cum_deaths", "cum_opps", "cum_opps_with_event",
                "events_per_opp_hist", "anc_final", "C_all_final", "C_prod_final", "birth_hist", "death_hist",
                "total_replacement_events", "n_fr_opps", "n_force_evals")

# every source file whose numba code is compiled into this module's functions (the dependency set of the cache key
# below) and whose sha256 is the provenance; lta_ladder_numba imports no other project module
_SOURCES = (os.path.join(_HERE, "lta_parallel.py"), os.path.join(_HERE, "philox.py"),
            os.path.join(_HERE, "__init__.py"), os.path.join(_SRC, "lta_ladder_numba.py"))


def _sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _git(args):
    try:
        r = subprocess.run(["git", "-C", ROOT] + args, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # pragma: no cover
        return None


_SRC_REL = tuple(os.path.relpath(p, ROOT) for p in _SOURCES)
_SRC_SHA256 = {r: _sha256(p) for r, p in zip(_SRC_REL, _SOURCES)}     # read ONCE: provenance and cache key agree


def _provenance():
    st = _git(["status", "--porcelain", "--"] + list(_SRC_REL))
    return dict(sources_sha256=dict(_SRC_SHA256),
                git_commit=_git(["rev-parse", "HEAD"]) or "unknown",
                git_dirty=(None if st is None else bool(st)), git_status=st or "",
                imported_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


_PROVENANCE = _provenance()


# =====================================================================================
# numba on-disk cache keyed by EVERY dependency source (not only this file)
# =====================================================================================
# numba validates a cached function only against the content of the file that DEFINES it, but the machine code of
# this module's functions also contains the code they call in philox.py (lta_normals6) and lta_ladder_numba.py
# (_mol_force, _fr_score, fr_select_native, ...).  With plain cache=True an edit to either file would leave every new
# process running the OLD compiled code while meta.sources_sha256 names the new files.  Every function of this
# module is therefore compiled with ``_jit``: numba's own FunctionCache and locators, with the source stamp
# extended by _DEP_FINGERPRINT (sha256 over the sha256 of every file in _SOURCES), so a change to any of them
# invalidates the index of every function here (stale data files are overwritten in place, no directory growth).
# If numba's caching internals are unavailable, or NUMBA_CACHE_LOCATOR_CLASSES overrides the locator list, the
# on-disk cache is DISABLED for this module (fresh compile per process; correct, only slower).  A once-per-process
# run-time self-check (kernel_selfcheck) independently verifies the compiled kernel against the current philox.py
# and lta_ladder_numba.py before any job runs.
def _dep_fingerprint(shas):
    h = hashlib.sha256(b"lta_parallel numba cache dependencies/1\n")
    for rel in sorted(shas):
        h.update(f"{rel}\0{shas[rel]}\n".encode())
    return h.hexdigest()


_DEP_FINGERPRINT = _dep_fingerprint(_SRC_SHA256)


def _make_dep_cache():
    if numba.config.CACHE_LOCATOR_CLASSES:
        return None, None, "disabled: NUMBA_CACHE_LOCATOR_CLASSES overrides the locator list"
    try:
        from numba.core import caching as nc
        bases = [getattr(nc, n, None) or getattr(nc, "_" + n)
                 for n in ("UserProvidedCacheLocator", "InTreeCacheLocator", "UserWideCacheLocator")]
        fcache, impl = nc.FunctionCache, nc.CompileResultCacheImpl
    except (ImportError, AttributeError) as e:  # pragma: no cover
        return None, None, f"disabled: numba caching internals unavailable ({e!r})"

    class DepStampLocator:
        """Mixin: numba's source stamp of the defining file + the dependency fingerprint."""

        def get_source_stamp(self):
            return (super().get_source_stamp(), "deps", _DEP_FINGERPRINT)

    locs = [type("Dep" + b.__name__.lstrip("_"), (DepStampLocator, b), {}) for b in bases]
    dimpl = type("DepCompileResultCacheImpl", (impl,), {"_locator_classes": locs})
    dcache = type("DepFunctionCache", (fcache,), {"_impl_class": dimpl})
    return dcache, DepStampLocator, "dependency-keyed"


_DEP_CACHE, _DEP_STAMP, _CACHE_MODE = _make_dep_cache()
_JITTED = []                                # every dispatcher made by _jit (tests / cache_info)


def _jit(**kw):
    """``njit(**kw, cache=True)`` with the dependency-keyed on-disk cache described above."""
    def deco(fn):
        d = njit(**kw)(fn)
        if _DEP_CACHE is not None:
            d._cache = _DEP_CACHE(fn)       # what Dispatcher.enable_caching does, with the dependency-aware class
            if not isinstance(d._cache._impl.locator, _DEP_STAMP):  # pragma: no cover
                raise RuntimeError(f"numba cache locator for {fn.__name__} is not dependency-keyed")
        _JITTED.append(d)
        return d
    return deco


# =====================================================================================
# clock (cacheable: clock_gettime is resolved by symbol name at link time, no embedded address)
# =====================================================================================
@intrinsic
def _clock_monotonic(typingctx):
    sig = types.float64()

    def codegen(context, builder, signature, args):
        i32 = ir.IntType(32)
        i64 = ir.IntType(64)
        ts_t = ir.LiteralStructType([i64, i64])
        fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(i32, [i32, ts_t.as_pointer()]),
                                            "clock_gettime")
        ts = cgutils.alloca_once(builder, ts_t)
        builder.call(fn, [ir.Constant(i32, 1), ts])          # CLOCK_MONOTONIC = 1 (Linux)
        zero = ir.Constant(i32, 0)
        sec = builder.load(builder.gep(ts, [zero, zero]))
        nsec = builder.load(builder.gep(ts, [zero, ir.Constant(i32, 1)]))
        dt = ir.DoubleType()
        return builder.fadd(builder.sitofp(sec, dt), builder.fmul(builder.sitofp(nsec, dt), ir.Constant(dt, 1e-9)))

    return sig, codegen


@_jit()
def now():
    """CLOCK_MONOTONIC seconds (the clock behind time.perf_counter on Linux)."""
    return _clock_monotonic()


# =====================================================================================
# per-walker work (the validated arithmetic, op for op)
# =====================================================================================
@_jit(error_model="numpy")
def _eval_walker(q, i, Ox, Oy, Oz, fp, F, phi, fl, jb, cphi, cf, dclip, dphi, ng):
    """Phase (1) of lta_ladder_numba._run_chunk for walker i (xc is diagnostics-only and not kept)."""
    _mol_force(q, i, Ox, Oy, Oz, fp, F)
    x = (q[i, 0, 0] + q[i, 1, 0]) * 0.5
    ph = _wrap(cphi * x)
    phi[i] = ph
    f = cf * (F[i, 0, 0] + F[i, 1, 0])
    if f < -dclip:
        f = -dclip
    elif f > dclip:
        f = dclip
    fl[i] = f
    jb[i] = _bin(ph, dphi, ng)


@_jit(error_model="numpy")
def _post_cv(q, i, cphi, dphi, ng, phin, jn):
    """The validated FR's post-move CV and bin of walker i."""
    x = (q[i, 0, 0] + q[i, 1, 0]) * 0.5
    ph = _wrap(cphi * x)
    phin[i] = ph
    jn[i] = _bin(ph, dphi, ng)


@_jit(error_model="numpy")
def _move_replay(q, i, F, bxi, dt, ns, noise, imove):
    for b in range(2):
        q[i, b, 0] = (q[i, b, 0] + dt * (F[i, b, 0] + bxi)) + ns * noise[imove, i, b, 0]
        q[i, b, 1] = (q[i, b, 1] + dt * F[i, b, 1]) + ns * noise[imove, i, b, 1]
        q[i, b, 2] = (q[i, b, 2] + dt * F[i, b, 2]) + ns * noise[imove, i, b, 2]


@_jit(error_model="numpy")
def _move_counter(q, i, F, bxi, dt, ns, k0, k1, step):
    z0, z1, z2, z3, z4, z5 = lta_normals6(k0, k1, step, i)
    q[i, 0, 0] = (q[i, 0, 0] + dt * (F[i, 0, 0] + bxi)) + ns * z0
    q[i, 0, 1] = (q[i, 0, 1] + dt * F[i, 0, 1]) + ns * z1
    q[i, 0, 2] = (q[i, 0, 2] + dt * F[i, 0, 2]) + ns * z2
    q[i, 1, 0] = (q[i, 1, 0] + dt * (F[i, 1, 0] + bxi)) + ns * z3
    q[i, 1, 1] = (q[i, 1, 1] + dt * F[i, 1, 1]) + ns * z4
    q[i, 1, 2] = (q[i, 1, 2] + dt * F[i, 1, 2]) + ns * z5


# =====================================================================================
# serial phases
# =====================================================================================
@_jit(error_model="numpy")
def _deposit_bias(step, N, burn, warm, bscale, fclip, gpx, jb, fl, acc, tf, tc, touched, bxs):
    """Phases (2) and (3) of the validated kernel (accumulate_u off)."""
    prod = step >= burn
    nt = 0
    for i in range(N):
        j = jb[i]
        if tc[j] == 0.0:
            touched[nt] = j
            nt += 1
        tf[j] += fl[i]
        tc[j] += 1.0
    for t in range(nt):
        j = touched[t]
        acc[A_FS, j] += tf[j]
        acc[A_CS, j] += tc[j]
        if prod:
            acc[A_FSP, j] += tf[j]
            acc[A_CSP, j] += tc[j]
        tf[j] = 0.0
        tc[j] = 0.0
    ramp = step / (warm if warm > 1 else 1)
    if ramp > 1.0:
        ramp = 1.0
    scl = bscale * ramp
    for i in range(N):
        j = jb[i]
        c = acc[A_CS, j]
        if c > 0.0:
            G = acc[A_FS, j] / (c if c > 1.0 else 1.0)
        else:
            G = 0.0
        if G < -fclip:
            G = -fclip
        elif G > fclip:
            G = fclip
        bxs[i] = (scl * G) * gpx


@_jit(error_model="numpy")
def _save(sp, ng, acc, ist, evh, sv_acc, sv_i, sv_evh):
    for k in range(ng):
        sv_acc[sp, 0, k] = acc[A_FS, k]
        sv_acc[sp, 1, k] = acc[A_CS, k]
        sv_acc[sp, 2, k] = acc[A_FSP, k]
        sv_acc[sp, 3, k] = acc[A_CSP, k]
    sv_i[sp, 0] = ist[S_REPL]
    sv_i[sp, 1] = ist[S_NOPP]
    sv_i[sp, 2] = ist[S_NOPPEV]
    for k in range(evh.shape[0]):
        sv_evh[sp, k] = evh[k]


@_jit(error_model="numpy")
def _fr_step(N, ng, cap, fr_on, fp, Kk, php, jp, frb, iopp, q, anc, acc, ist, fst, evh, cnt, pk, raw, sc, touched,
             dp, cdf, cand, dsel, ssel, F, phi, fl, jb, fused):
    """Phase (7) of the validated kernel (mode 0, diagnostics off) on the post-move CV (php, jp); with ``fused`` the
    post-move evaluation (F, phi, fl, jb) of every clone is copied from its source.  Returns the number of FR
    uniform blocks consumed (0 or 1)."""
    ist[S_NOPP] += 1
    sstd, sabs = _fr_score(php, jp, N, ng, fp[F_DPHI], Kk, fp[F_GRID0], fp[F_QV], fp[F_LQ], fp[F_SCLIP], cnt, pk,
                           raw, sc, touched)
    fst[G_SSTD] += sstd
    if sabs > fst[G_SMAX]:
        fst[G_SMAX] = sabs
    ist[S_NSCORE] += 1
    nev = 0
    used = 0
    if fr_on:
        nev = fr_select_native(sc, N, cap, fp[F_RATE], fp[F_DTEFF], frb[iopp], cand, dsel, ssel, dp, cdf)
        used = 1
    for t in range(nev):
        d = dsel[t]
        s_ = ssel[t]
        acc[A_BH, jp[s_]] += 1.0
        acc[A_DH, jp[d]] += 1.0
        for b in range(2):
            for c3 in range(3):
                q[d, b, c3] = q[s_, b, c3]
        anc[d] = anc[s_]
    if fused:
        for t in range(nev):
            d = dsel[t]
            s_ = ssel[t]
            for b in range(2):
                for c3 in range(3):
                    F[d, b, c3] = F[s_, b, c3]
            phi[d] = phi[s_]
            fl[d] = fl[s_]
            jb[d] = jb[s_]
    evh[nev] += 1
    if nev > 0:
        ist[S_NOPPEV] += 1
        ist[S_REPL] += nev
    return used


# =====================================================================================
# the parallel kernel
# =====================================================================================
@_jit(parallel=True, error_model="numpy")
def _kernel(n_eval, fp, ip, Ox, Oy, Oz, Kk, save_steps, noise, frb, k0, k1, counter, timers, t_start, nth,
            q, anc, acc, ist, fst, evh, sv_acc, sv_i, sv_evh, sv_wall, sv_nfe,
            F, phi, fl, jb, bxs, phin, jn, tf, tc, touched, cnt, pk, raw, sc, dp, cdf, cand, dsel, ssel, tim, busy):
    """Advance ``n_eval`` evaluated steps in place (state between calls: after the FR opportunity of step k-1 -> k,
    before the evaluation of step k = ist[S_STEP], as the validated kernel).  ``nth`` = numba.get_num_threads() of
    the caller (the number of static walker blocks; passed in because calling numba.get_num_threads() inside the
    kernel embeds a ctypes pointer and disables caching).  Returns (moves, FR blocks used)."""
    N = ip[I_N]
    ng = ip[I_NG]
    n_steps = ip[I_NSTEPS]
    warm = ip[I_WARM]
    burn = ip[I_BURN]
    fr_start = ip[I_FRSTART]
    fr_every = ip[I_FREVERY]
    cap = ip[I_CAP]
    is_fr = ip[I_ISFR] != 0
    fr_on = ip[I_FRON] != 0
    dt = fp[F_DT]
    ns = fp[F_NS]
    cphi = fp[F_CPHI]
    cf = fp[F_CF]
    gpx = fp[F_GPX]
    dphi = fp[F_DPHI]
    fclip = fp[F_FCLIP]
    dclip = fp[F_DCLIP]
    bscale = fp[F_BSCALE]
    S = save_steps.shape[0]
    fvalid = False
    imove = 0
    iopp = 0
    for _it in range(n_eval):
        step = ist[S_STEP]
        # (1) forces / CV / bin at the pre-move positions (unless the fused phase of the last step did it)
        if not fvalid:
            ta = now() if timers else 0.0
            for t in prange(nth):
                lo = (t * N) // nth
                hi = ((t + 1) * N) // nth
                tb = now() if timers else 0.0
                for i in range(lo, hi):
                    _eval_walker(q, i, Ox, Oy, Oz, fp, F, phi, fl, jb, cphi, cf, dclip, dphi, ng)
                if timers:
                    busy[t * PAD] += now() - tb
            if timers:
                tim[T_PAR] += now() - ta
                tim[T_NPAR] += 1.0
        ist[S_NFE] += N
        # (2) + (3) deposits of all walkers in slot order, then the bias read
        ta = now() if timers else 0.0
        _deposit_bias(step, N, burn, warm, bscale, fclip, gpx, jb, fl, acc, tf, tc, touched, bxs)
        if timers:
            tim[T_ABF] += now() - ta
        # (5) save
        sp = ist[S_SP]
        if sp < S and save_steps[sp] == step:
            ta = now()
            _save(sp, ng, acc, ist, evh, sv_acc, sv_i, sv_evh)
            sv_nfe[sp] = ist[S_NFE]
            sv_wall[sp] = ta - t_start
            ist[S_SP] = sp + 1
            if timers:
                tim[T_SAVE] += now() - ta
        if step >= n_steps:
            ist[S_STEP] = step + 1
            fvalid = False
            break
        # (6) move [+ the evaluation of step k + 1 at the new position]
        nxt = step + 1
        fr_now = is_fr and nxt >= fr_start and (nxt - fr_start) % fr_every == 0
        fuse = _it + 1 < n_eval
        ta = now() if timers else 0.0
        for t in prange(nth):
            lo = (t * N) // nth
            hi = ((t + 1) * N) // nth
            tb = now() if timers else 0.0
            for i in range(lo, hi):
                if counter:
                    _move_counter(q, i, F, bxs[i], dt, ns, k0, k1, step)
                else:
                    _move_replay(q, i, F, bxs[i], dt, ns, noise, imove)
                if fuse:
                    _eval_walker(q, i, Ox, Oy, Oz, fp, F, phi, fl, jb, cphi, cf, dclip, dphi, ng)
                elif fr_now:
                    _post_cv(q, i, cphi, dphi, ng, phin, jn)
            if timers:
                busy[t * PAD] += now() - tb
        if timers:
            tim[T_PAR] += now() - ta
            tim[T_NPAR] += 1.0
        imove += 1
        fvalid = fuse
        # (7) Fisher-Rao at the post-move positions
        if fr_now:
            ta = now() if timers else 0.0
            if fuse:
                iopp += _fr_step(N, ng, cap, fr_on, fp, Kk, phi, jb, frb, iopp, q, anc, acc, ist, fst, evh, cnt, pk,
                                 raw, sc, touched, dp, cdf, cand, dsel, ssel, F, phi, fl, jb, True)
            else:
                iopp += _fr_step(N, ng, cap, fr_on, fp, Kk, phin, jn, frb, iopp, q, anc, acc, ist, fst, evh, cnt, pk,
                                 raw, sc, touched, dp, cdf, cand, dsel, ssel, F, phi, fl, jb, False)
            if timers:
                tim[T_FR] += now() - ta
        ist[S_STEP] = nxt
    return imove, iopp


# =====================================================================================
# Python driver
# =====================================================================================
def resolve(cfg, N, seed, n_steps):
    """The lta_ladder_numba cfg of this job: make_cfg(T_K (default 150), N, seed, **cfg, n_steps, diagnostics
    False).  ``cfg`` is the equal-budget engine_cfg (overrides) or an already resolved cfg; N / seed / n_steps /
    T_K given in it must agree with the arguments; derived keys are recomputed."""
    cc = dict(cfg)
    T_K = float(cc.pop("T_K", 150.0))
    for k, v in (("N", N), ("seed", seed), ("n_steps", n_steps)):
        if k in cc and cc[k] is not None and int(cc[k]) != int(v):
            raise ValueError(f"cfg[{k!r}] = {cc[k]} disagrees with the argument {v}")
        cc.pop(k, None)
    for k in ("cap", "genealogy_window_steps", "n_trace"):
        cc.pop(k, None)
    cc.pop("diagnostics", None)
    if cc.get("accumulate_u"):
        raise ValueError("accumulate_u is not implemented in the parallel engine")
    if cc.get("record_events"):
        raise ValueError("record_events is not implemented in the parallel engine")
    return E.make_cfg(T_K, int(N), int(seed), n_steps=int(n_steps), diagnostics=False, **cc)


def _alloc(N, ng, cap, S, nth):
    return dict(
        anc=np.arange(N, dtype=np.int64), acc=np.zeros((NA, ng)), ist=np.zeros(NIST, dtype=np.int64),
        fst=np.zeros(NFST), evh=np.zeros(cap + 1, dtype=np.int64), sv_acc=np.zeros((S, 4, ng)),
        sv_i=np.full((S, 3), -1, dtype=np.int64), sv_evh=np.zeros((S, cap + 1), dtype=np.int64),
        sv_wall=np.full(S, np.nan), sv_nfe=np.full(S, -1, dtype=np.int64),
        F=np.zeros((N, 2, 3)), phi=np.zeros(N), fl=np.zeros(N), jb=np.zeros(N, dtype=np.int64), bxs=np.zeros(N),
        phin=np.zeros(N), jn=np.zeros(N, dtype=np.int64), tf=np.zeros(ng), tc=np.zeros(ng),
        touched=np.zeros(ng, dtype=np.int64), cnt=np.zeros(ng), pk=np.zeros(ng), raw=np.zeros(N), sc=np.zeros(N),
        dp=np.zeros(N), cdf=np.zeros(N), cand=np.zeros(N, dtype=np.int64), dsel=np.zeros(N, dtype=np.int64),
        ssel=np.zeros(N, dtype=np.int64), tim=np.zeros(NT), busy=np.zeros(max(nth, 1) * PAD))


def _call(n_eval, fp, ip, Ox, Oy, Oz, Kk, save_steps, noise, frb, k0, k1, counter, timers, t_start, q, st):
    nth = (st["busy"].shape[0]) // PAD
    return _kernel(n_eval, fp, ip, Ox, Oy, Oz, Kk, save_steps, noise, frb, k0, k1, counter, timers, t_start,
                   np.int64(nth), q, st["anc"], st["acc"], st["ist"], st["fst"], st["evh"], st["sv_acc"], st["sv_i"],
                   st["sv_evh"], st["sv_wall"], st["sv_nfe"], st["F"], st["phi"], st["fl"], st["jb"], st["bxs"],
                   st["phin"], st["jn"], st["tf"], st["tc"], st["touched"], st["cnt"], st["pk"], st["raw"], st["sc"],
                   st["dp"], st["cdf"], st["cand"], st["dsel"], st["ssel"], st["tim"], st["busy"])


def _warmup(fp, ip, Ox, Oy, Oz, Kk, q0, k0, k1, counter, timers, nth):
    """JIT / cache load and OpenMP pool start: 5 evaluated steps of a dummy job with max(2, nth) walkers, FR at
    every step (same argument types as the real call, so the real call does not compile)."""
    Nw = max(2, nth)
    ng = int(ip[I_NG])
    ipw = ip.copy()
    ipw[I_N] = Nw
    ipw[I_NSTEPS] = 4
    ipw[I_FRSTART] = 1
    ipw[I_FREVERY] = 1
    ipw[I_CAP] = 1
    ipw[I_ISFR] = 1
    ipw[I_FRON] = 1
    qw = np.ascontiguousarray(q0[np.arange(Nw) % q0.shape[0]]).copy()
    stw = _alloc(Nw, ng, 1, 1, nth)
    noise = np.zeros((4 if not counter else 0, Nw, 2, 3))
    frb = np.random.default_rng(0).random((4, Nw + 2))
    _call(5, fp, ipw, Ox, Oy, Oz, Kk, np.array([2], dtype=np.int64), noise, frb, k0, k1, counter, timers, now(),
          qw, stw)


def _meta(c, method, threads, noise, timers, chunk_steps, layer, fr_on, k0, k1, rng_info, extra):
    import llvmlite
    import llvmlite.binding as ll
    tgt = E.compile_target()
    env = {k: os.environ.get(k) for k in ("OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT",
                                          "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMBA_CACHE_DIR",
                                          "NUMBA_CPU_NAME", "NUMBA_CPU_FEATURES", "CUDA_VISIBLE_DEVICES")}
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        cpus = None
    m = dict(
        engine=ENGINE, validated_engine=E.ENGINE, system="lta", method=method, T_K=c["T_K"], N=c["N"],
        seed=c["seed"], n_steps=c["n_steps"], h=c["h"], cap=c["cap"], fr_active=bool(fr_on), threads=int(threads),
        noise_mode=noise, timers=bool(timers), chunk_steps=int(chunk_steps), threading_layer=layer,
        numba_num_threads_config=int(numba.config.NUMBA_NUM_THREADS),
        counter_key=[str(int(k0)), str(int(k1))],
        counter_key_entropy=[c["seed"], c["N"], int(round(c["T_K"])), 2],
        counter_layout="Philox4x64-10 counter (move step, slot, word 0|1, 0); Box-Muller (u0,u1),(u2,u3) per word; "
                       "first 6 of 8 normals in [bead][xyz] order",
        rng=dict(rng_info, noise_replay="child 1: standard_normal((n_moves, N, 2, 3)) per chunk (validated layout)",
                 fr="child 2: random((n_opportunities, N + 2 cap)) per chunk (validated layout)", ic="child 0"),
        conventions=dict(save="state at the start of the saved step; accumulators include that step's deposit "
                              "(validated engine)",
                         n_force_evals_at_save="N (k + 1) at the save of step k",
                         wall_s_at_save="seconds since the timed start (after warm-up), CLOCK_MONOTONIC in the kernel",
                         t_sync_s="parallel-phase wall minus the mean per-thread walker-work time (launch, join, "
                                  "imbalance); best effort",
                         timers_off="phase timers NaN"),
        decomposition="prange over threads, static contiguous walker blocks; move(k)+force(k+1) fused; deposits, "
                      "bias, saves and FR serial in slot order",
        compiler=dict(numba=numba.__version__, llvmlite=llvmlite.__version__,
                      llvm=".".join(map(str, ll.llvm_version_info)),
                      flags="njit(parallel=True, error_model='numpy'); per-walker code njit(error_model='numpy'); "
                            "on-disk cache keyed by every dependency source (see numba_cache); validated _lj_bead "
                            "fastmath={'reassoc'} (sum only); no FMA contraction",
                      compile_target=dict(triple=tgt["triple"], cpu_name=tgt["cpu_name"],
                                          cpu_features_sha256=tgt["cpu_features_sha256"])),
        numpy=np.__version__, python=platform.python_version(), host=platform.node(),
        host_cpu=E._host_cpu(), cpus=cpus, n_cpus=(len(cpus) if cpus else None), env=env,
        numba_cache=cache_info(), **_PROVENANCE)
    m.update(extra)
    return m


def _setup(c, method):
    """Arrays of one resolved job (the validated engine's functions): framework, parameters, KDE matrix, the three
    PCG64 streams, the initial conditions and the counter key."""
    N = c["N"]
    O, a, L = E.framework(c["framework_npz"])
    Ox, Oy, Oz = (np.ascontiguousarray(O[:, k]) for k in range(3))
    fp, ip, beta, fr_on = E._params(c, method, O, a, L, 0)
    Kk, _, _ = E.kde_matrix(int(c["n_grid"]), float(c["kde_bandwidth"]))
    gens, rng_info = E.seed_streams(c["seed"], N, c["T_K"])
    q = np.ascontiguousarray(E.initial_conditions(gens[0], N, a, L, float(c["r0_bond"])), dtype=np.float64).copy()
    k0, k1 = PX.lta_key(c["seed"], N, c["T_K"])
    return dict(Ox=Ox, Oy=Oy, Oz=Oz, fp=fp, ip=ip, fr_on=fr_on, Kk=Kk, gen_noise=gens[1], gen_fr=gens[2],
                rng_info=rng_info, q=q, k0=k0, k1=k1)


def _validate(cfg, method, N, seed, n_steps, save_steps, threads, noise, chunk_steps):
    if noise not in NOISE_MODES:
        raise ValueError(f"noise must be one of {NOISE_MODES}, not {noise!r}")
    if method not in E.METHODS:
        raise ValueError(f"method must be one of {E.METHODS}, not {method!r}")
    threads = int(threads)
    if threads < 1:
        raise ValueError("threads must be >= 1")
    c = resolve(cfg, N, seed, n_steps)
    if method == "fr" and c["N"] < 2:
        raise ValueError("no FR arm at N = 1 (the uniform score is identically 0): run ABF only")
    save_steps = np.ascontiguousarray(np.asarray(save_steps, dtype=np.int64).ravel())
    if save_steps.size and (np.any(np.diff(save_steps) <= 0) or save_steps[0] < 0
                            or save_steps[-1] > c["n_steps"]):
        raise ValueError("save_steps must be sorted, unique and inside [0, n_steps]")
    if chunk_steps is None:
        chunk_steps = max(1, (4_000_000 if noise == "counter" else 1_000_000) // c["N"])
    chunk_steps = int(chunk_steps)
    if chunk_steps < 1:
        raise ValueError("chunk_steps must be >= 1")
    return c, save_steps, threads, chunk_steps


def run_arm_parallel(cfg, method, N, seed, n_steps, save_steps, *, threads, noise, timers=True, chunk_steps=None,
                     _noise_override=None):
    """Run ONE arm ('abf' or 'fr') of one LTA (N, seed) job on the walker-parallel engine (frozen interface).

    cfg         equal-budget engine_cfg (``lta_ladder_numba.make_cfg(150.0, N, seed, **cfg)``; diagnostics forced off)
    save_steps  sorted unique completed-step counts in [0, n_steps]
    threads     numba threads (numba.set_num_threads; restored on return; must be <= NUMBA_NUM_THREADS, numba's
                pool size, default the host CPU count)
    noise       'replay' (validated PCG64 stream, bitwise = validated engine) | 'counter' (in-kernel Philox)
    timers      in-kernel phase timers (see module docstring)
    chunk_steps evaluated steps per kernel call (default: replay max(1, 1e6 // N) as the validated engine,
                counter max(1, 4e6 // N)); results do not depend on it
    _noise_override  test hook: callable(step0, n_moves) -> (n_moves, N, 2, 3) noise used in replay mode

    The first call in a process runs ``kernel_selfcheck`` (milliseconds once compiled; it also absorbs the JIT /
    cache load, so that call's warmup_s is the thread-pool start only) and raises RuntimeError if the compiled
    kernel does not match the current philox.py / lta_ladder_numba.py.  Its record is meta["selfcheck"].
    """
    c, save_steps, threads, chunk_steps = _validate(cfg, method, N, seed, n_steps, save_steps, threads, noise,
                                                    chunk_steps)
    ran_now = _SELFCHECK is None
    sc = kernel_selfcheck(c, threads)
    extra = dict(selfcheck=dict(passed=sc["passed"], wall_s=sc["wall_s"], ran_in_this_call=ran_now,
                                fr_events=sc["fr_events"], N=sc["N"], n_steps=sc["n_steps"], seed=sc["seed"],
                                pid=sc["pid"]))
    return _run(c, method, save_steps, threads, noise, bool(timers), chunk_steps, _noise_override, extra)


def _run(c, method, save_steps, threads, noise, timers, chunk_steps, noise_override=None, meta_extra=None):
    """The engine on an already validated job (``_validate``); no self-check."""
    reset_ok = E._reset_peak_rss()
    N = c["N"]
    n_steps = c["n_steps"]
    h = float(c["h"])
    counter = noise == "counter"
    s = _setup(c, method)
    Ox, Oy, Oz, fp, ip, Kk, q, k0, k1 = (s[k] for k in ("Ox", "Oy", "Oz", "fp", "ip", "Kk", "q", "k0", "k1"))
    fr_on = s["fr_on"]
    gen_noise, gen_fr = s["gen_noise"], s["gen_fr"]
    ng = int(c["n_grid"])
    cap = int(c["cap"])
    fr_start, fr_every = int(c["fr_start_steps"]), int(c["fr_every"])

    prev_threads = numba.get_num_threads()
    numba.set_num_threads(threads)
    try:
        nth = numba.get_num_threads()
        st = _alloc(N, ng, cap, save_steps.size, nth)
        tw = time.perf_counter()
        _warmup(fp, ip, Ox, Oy, Oz, Kk, q, k0, k1, counter, timers, nth)
        warmup_s = time.perf_counter() - tw
        layer = numba.threading_layer()
        if layer != "omp":
            raise RuntimeError(f"numba threading layer is {layer!r}, the backend requires 'omp'")
        empty_noise = np.zeros((0, N, 2, 3))
        empty_frb = np.zeros((0, N + 2 * cap))
        ist = st["ist"]
        t_noise = 0.0
        t_frdraw = 0.0
        n_chunks = 0
        t_start = now()
        while ist[S_STEP] <= n_steps:
            step0 = int(ist[S_STEP])
            n_eval = min(chunk_steps, n_steps + 1 - step0)
            last = step0 + n_eval - 1
            n_moves = n_eval if last < n_steps else n_eval - 1
            if counter:
                nz = empty_noise
            else:
                t0 = now()
                if noise_override is not None:
                    nz = np.ascontiguousarray(noise_override(step0, n_moves), dtype=np.float64)
                    nz = nz.reshape(n_moves, N, 2, 3)
                else:
                    nz = gen_noise.standard_normal((n_moves, N, 2, 3))
                t_noise += now() - t0
            if method == "fr" and fr_on:
                t0 = now()
                n_opp = E._n_opps_in(step0 + 1, step0 + n_moves, fr_start, fr_every)
                frb = gen_fr.random((n_opp, N + 2 * cap))
                t_frdraw += now() - t0
            else:
                n_opp = 0
                frb = empty_frb
            imove, iopp = _call(n_eval, fp, ip, Ox, Oy, Oz, Kk, save_steps, nz, frb, k0, k1, counter, timers,
                                t_start, q, st)
            if int(ist[S_STEP]) != step0 + n_eval or imove != n_moves or iopp != n_opp:
                raise RuntimeError(f"chunk accounting error: step {ist[S_STEP]} vs {step0 + n_eval}, "
                                   f"moves {imove}/{n_moves}, opps {iopp}/{n_opp}")
            n_chunks += 1
        wall_total = now() - t_start
    finally:
        numba.set_num_threads(prev_threads)

    S = save_steps.size
    if int(ist[S_SP]) != S:
        raise RuntimeError(f"only {int(ist[S_SP])} of {S} saves written")
    nfe = int(ist[S_NFE])
    if nfe != N * (n_steps + 1):
        raise RuntimeError(f"force-evaluation count {nfe} != N (n_steps + 1) = {N * (n_steps + 1)}")
    tim = st["tim"]
    if timers:
        busy_mean = float(st["busy"][::PAD][:nth].sum()) / nth
        t_fm = busy_mean
        t_sync = float(tim[T_PAR]) - busy_mean
        t_abf = float(tim[T_ABF])
        t_fr = float(tim[T_FR]) + t_frdraw
        t_save = float(tim[T_SAVE])
        t_other = wall_total - (t_fm + t_sync + t_abf + t_fr + t_save + t_noise)
        npar = int(tim[T_NPAR])
    else:
        t_fm = t_sync = t_abf = t_fr = t_save = t_other = float("nan")
        npar = -1
    sv_acc, sv_i = st["sv_acc"], st["sv_i"]
    acc = st["acc"]
    peak, peak_method = E._peak_rss_mb(reset_ok)
    extra = dict(n_chunks=n_chunks, n_saves=int(S), peak_rss_method=peak_method,
                 run_end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 keys_shared_with_validated=list(SHARED_KEYS), cost_keys=list(COST_KEYS))
    extra.update(meta_extra or {})
    meta = _meta(c, method, threads, noise, timers, chunk_steps, layer, fr_on, k0, k1, s["rng_info"], extra)
    res = dict(
        save_step=save_steps.copy(), save_t=save_steps * h, save_u=save_steps / float(n_steps),
        M_all=sv_acc[:, 0].copy(), C_all=sv_acc[:, 1].copy(), M_prod=sv_acc[:, 2].copy(), C_prod=sv_acc[:, 3].copy(),
        cum_deaths=sv_i[:, 0].copy(), cum_opps=sv_i[:, 1].copy(), cum_opps_with_event=sv_i[:, 2].copy(),
        events_per_opp_hist=st["sv_evh"].copy(),
        q_final=q.copy(), anc_final=st["anc"].copy(),
        M_all_final=acc[A_FS].copy(), C_all_final=acc[A_CS].copy(),
        M_prod_final=acc[A_FSP].copy(), C_prod_final=acc[A_CSP].copy(),
        birth_hist=acc[A_BH].copy(), death_hist=acc[A_DH].copy(),
        total_replacement_events=np.int64(ist[S_REPL]), n_fr_opps=np.int64(ist[S_NOPP]),
        fr_score_std=np.float64(st["fst"][G_SSTD] / max(int(ist[S_NSCORE]), 1)),
        fr_score_absmax=np.float64(st["fst"][G_SMAX]),
        n_force_evals=np.int64(nfe),
        cfg_json=json.dumps(c, sort_keys=True, default=str),
        wall_s_at_save=st["sv_wall"].copy(), n_force_evals_at_save=st["sv_nfe"].copy(),
        wall_total_s=np.float64(wall_total), warmup_s=np.float64(warmup_s),
        t_force_move_s=np.float64(t_fm), t_abf_s=np.float64(t_abf), t_sync_s=np.float64(t_sync),
        t_fr_s=np.float64(t_fr), t_save_s=np.float64(t_save), t_noise_s=np.float64(t_noise),
        t_other_s=np.float64(t_other), n_parallel_regions=np.int64(npar),
        busy_per_thread_s=(st["busy"][::PAD][:nth].copy() if timers else np.full(nth, np.nan)),
        threads=np.int64(nth), noise_mode=noise, engine=ENGINE, peak_rss_mb=np.float64(peak),
        meta=json.dumps(meta, sort_keys=True, default=str),
    )
    return res


# =====================================================================================
# once-per-process self-check of the compiled kernel (stale numba cache guard)
# =====================================================================================
SELFCHECK_SEED = 15409                       # the last never-scored bitwise-test seed (prereg section 4)
SELFCHECK_N = 16
SELFCHECK_STEPS = 200
SELFCHECK_KNOBS = dict(warmup_steps=2, burn_in_steps=2, fr_start_steps=2, fr_every=1, fr_rate=400.0)
_SELFCHECK = None                            # the passing record, once per process


class StaleKernelError(RuntimeError):
    """The compiled kernel does not reproduce the current philox.py / lta_ladder_numba.py."""


def selfcheck_cfg(c=None):
    """The resolved cfg of the self-check job: the caller's physics (framework, T, h, grid, clips) with N = 16,
    seed 15409, 200 steps and FR from step 2 at every step with fr_rate 400 (about 20 events at production
    physics, so the FR score / selection code runs)."""
    base = dict(c) if c is not None else resolve(production_engine_cfg(), SELFCHECK_N, SELFCHECK_SEED, 1)
    for k in ("cap", "genealogy_window_steps", "n_trace"):
        base.pop(k, None)
    base.update(N=SELFCHECK_N, seed=SELFCHECK_SEED, n_steps=SELFCHECK_STEPS, diagnostics=False, **SELFCHECK_KNOBS)
    return E.resolve_cfg(base)


def kernel_selfcheck(c=None, threads=None, *, force=False):
    """Verify, once per process, that the compiled kernel runs the CURRENT dependency code.

    (a) replay mode == lta_ladder_numba.run_arm (diagnostics off) on every shared key: the kernel's linked copies of
        _mol_force / _wrap / _bin / _fr_score / fr_select_native are the validated module's current code;
    (b) counter mode == replay mode fed with parallel.philox.lta_noise_block: the kernel's linked Philox / Box-Muller
        is philox.py's current code (the generator the V-B3 oracle and battery test).
    The references are compiled from their own files (numba keys their caches by those files), so a stale kernel
    is caught even if the dependency-keyed cache were bypassed.  Raises StaleKernelError on a mismatch; returns the
    passing record (cached for the process unless ``force``)."""
    global _SELFCHECK
    if _SELFCHECK is not None and not force:
        return _SELFCHECK
    t0 = time.perf_counter()
    cs = selfcheck_cfg(c)
    if threads is None:
        threads = numba.get_num_threads()
    threads = int(threads)
    N, n = cs["N"], cs["n_steps"]
    grid = np.arange(0, n + 1, dtype=np.int64)
    k0, k1 = PX.lta_key(cs["seed"], N, cs["T_K"])
    chunk = max(1, n // 3)                   # several kernel calls: the unfused chunk boundary is exercised too
    ref = E.run_arm(cs, "fr", grid)
    rep = _run(cs, "fr", grid, threads, "replay", True, chunk)
    cnt = _run(cs, "fr", grid, threads, "counter", True, chunk)
    inj = _run(cs, "fr", grid, threads, "replay", True, chunk,
               noise_override=lambda s0, m: PX.lta_noise_block(k0, k1, s0, m, N))
    bad_val = compare_shared(ref, rep)
    bad_ctr = compare_shared(cnt, inj)
    rec = dict(passed=not bad_val and not bad_ctr, kernel_equals_validated=not bad_val,
               keys_differing_validated=bad_val, kernel_counter_equals_philox_module=not bad_ctr,
               keys_differing_philox=bad_ctr, fr_events=int(ref["total_replacement_events"]), N=int(N),
               n_steps=int(n), seed=int(cs["seed"]), threads=threads, wall_s=time.perf_counter() - t0,
               pid=os.getpid(), dependency_fingerprint=_DEP_FINGERPRINT)
    if not rec["passed"]:
        raise StaleKernelError(
            "stale compiled kernel: " + ("replay != lta_ladder_numba.run_arm on %s; " % bad_val if bad_val else "")
            + ("counter != parallel.philox.lta_noise_block on %s; " % bad_ctr if bad_ctr else "")
            + f"clear the numba cache (NUMBA_CACHE_DIR={numba.config.CACHE_DIR or '<in-tree __pycache__>'}) and "
              f"re-run; record: {json.dumps(rec, default=str)}")
    _SELFCHECK = rec
    return rec


def cache_info():
    """How this module's numba code is cached (meta["numba_cache"])."""
    loc = getattr(getattr(_kernel._cache, "_impl", None), "locator", None)
    st = _kernel.stats
    return dict(mode=_CACHE_MODE, dependency_fingerprint=_DEP_FINGERPRINT, dependency_files=list(_SRC_REL),
                locator=(type(loc).__name__ if loc is not None else None),
                kernel_cache_path=getattr(_kernel._cache, "cache_path", None),
                numba_cache_dir=(numba.config.CACHE_DIR or None),
                kernel_cache_hits=int(sum(st.cache_hits.values())),
                kernel_cache_misses=int(sum(st.cache_misses.values())))


def save_result(path, res):
    """Atomic compressed .npz (no pickle), the validated engine's writer."""
    E.save_result(path, res)


def load_result(path):
    return E.load_result(path)


# =====================================================================================
# validation helpers (tests/test_parallel_lta.py and the CLI)
# =====================================================================================
PROD_CFG_PATH = os.path.join(ROOT, "configs", "equal_budget_v2", "lta_150K_production.json")
LTA_B = 307_200_000
BITWISE_SEEDS = tuple(range(15400, 15410))      # prereg section 4: benchmarks / bitwise tests, never scored
VB1_CASES = ((2, "abf", 40_000), (16, "abf", 200_000), (16, "fr", 200_000), (64, "abf", 200_000),
             (64, "fr", 200_000), (512, "abf", 60_000), (512, "fr", 60_000))   # prereg section 2.3 minimum lengths
VB1_THREADS = (1, 4, 16)
VB2_THREADS = (1, 2, 4, 8, 16)
VB3_TESTS = ("mean", "variance", "skewness", "excess_kurtosis", "lag1_steps", "lag1_slots", "lag1_words",
             "box_muller_partners", "ks_normal")


def production_engine_cfg():
    """The frozen equal-budget LTA 150 K engine_cfg (diagnostics is forced off by run_arm_parallel / resolve)."""
    with open(PROD_CFG_PATH) as f:
        return dict(json.load(f)["engine_cfg"])


def expb_save_grid(n_steps):
    """Experiment B save grid (prereg section 2.5): u = k/200 (k = 1..200) union u = 1e-4 1.02^j <= 1;
    steps = unique(clip(round(u n_steps), 1, n_steps)).  (The harness's grid is authoritative for production.)"""
    u_lin = np.arange(1, 201, dtype=np.float64) / 200.0
    u_log = 1e-4 * 1.02 ** np.arange(0, 1000, dtype=np.float64)
    u_log = u_log[u_log <= 1.0]
    u = np.unique(np.concatenate([u_lin, u_log]))
    return np.unique(np.clip(np.round(u * int(n_steps)).astype(np.int64), 1, int(n_steps)))


def same_bits(x, y):
    if isinstance(x, str) or isinstance(y, str):
        return x == y
    x = np.asarray(x)
    y = np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def compare_shared(ref, res, keys=SHARED_KEYS):
    """Keys of ``keys`` on which ``res`` is not bitwise ``ref`` (dtype, shape and bytes); missing keys included."""
    return [k for k in keys if k not in ref or k not in res or not same_bits(ref[k], res[k])]


def fallback_report(ref, res):
    """The preregistered LTA-only V-B1 fallback quantities: identical integer outputs (bins / FR events) and the
    max |dq| of the final coordinates and max relative difference of the float accumulators."""
    ints_bad = [k for k in INTEGER_KEYS if not same_bits(ref[k], res[k])]
    dq = float(np.max(np.abs(np.asarray(ref["q_final"]) - np.asarray(res["q_final"]))))
    rel = {}
    for k in ("M_all", "M_prod", "M_all_final", "fr_score_std"):
        a = np.asarray(ref[k], dtype=np.float64)
        b = np.asarray(res[k], dtype=np.float64)
        rel[k] = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300))) if a.size else 0.0
    return dict(integer_keys_identical=not ints_bad, integer_keys_differing=ints_bad, max_abs_dq_A=dq,
                max_rel_float=rel, fallback_pass=(not ints_bad) and dq <= 1e-8)


@_jit(parallel=True, error_model="numpy")
def _forces_par(q, Ox, Oy, Oz, fp, F, phi, fl, jb, cphi, cf, dclip, dphi, ng):
    """DIAGNOSTIC ONLY: a separate prange helper over walkers.  It is NOT the engine kernel (another compilation
    context); first_step_forces takes its evidence from _kernel and reports this only as a labelled extra."""
    for i in prange(q.shape[0]):
        _eval_walker(q, i, Ox, Oy, Oz, fp, F, phi, fl, jb, cphi, cf, dclip, dphi, ng)


def kernel_forces(c, threads):
    """The engine kernel's OWN per-walker forces (ABF arm, replay noise = the validated child-1 stream):

    F0 at the initial positions q0 from _kernel's evaluation phase (one _call with n_eval 1, n_steps 1: step 0 is
    evaluated, then moved unfused), and F1 at the positions q1 after the first move from _kernel's FUSED
    move + evaluation phase (a fresh _call with n_eval 2: the move of step 0 and the evaluation of step 1 run in one
    parallel phase; step 1 is the last, so it is not moved).  Every production force evaluation is one of these two
    code paths.  Returns (q0, F0, q1, F1)."""
    s = _setup(c, "abf")
    N = c["N"]
    ng = int(c["n_grid"])
    cap = int(c["cap"])
    ip = s["ip"].copy()
    ip[I_NSTEPS] = 1
    noise = np.ascontiguousarray(s["gen_noise"].standard_normal((1, N, 2, 3)))
    frb = np.zeros((0, N + 2 * cap))
    no_saves = np.zeros(0, dtype=np.int64)
    out = []
    prev = numba.get_num_threads()
    numba.set_num_threads(int(threads))
    try:
        nth = numba.get_num_threads()
        for n_eval in (1, 2):
            q = s["q"].copy()
            st = _alloc(N, ng, cap, 0, nth)
            imove, _ = _call(n_eval, s["fp"], ip, s["Ox"], s["Oy"], s["Oz"], s["Kk"], no_saves, noise, frb, s["k0"],
                             s["k1"], False, True, now(), q, st)
            if imove != 1 or int(st["ist"][S_STEP]) != n_eval:
                raise RuntimeError("kernel_forces: unexpected kernel accounting")
            out.append((q, st["F"].copy()))
    finally:
        numba.set_num_threads(prev)
    (q1a, F0), (q1, F1) = out
    if not same_bits(q1a, q1):
        raise RuntimeError("kernel_forces: the two calls moved to different positions")
    return s["q"], F0, q1, F1


def validated_forces(c, q):
    """Per-walker forces at q from the VALIDATED module's own compiled entry point lta_ladder_numba._mol_force,
    called per walker from Python (no code of this module involved).  _run_chunk's forces live in a local array
    and cannot be read without modifying the validated engine; V-B1's bitwise q_final is the end-to-end proof
    that the kernel's forces equal _run_chunk's."""
    O, a, L = E.framework(c["framework_npz"])
    Ox, Oy, Oz = (np.ascontiguousarray(O[:, k]) for k in range(3))
    fp, _, _, _ = E._params(c, "abf", O, a, L, 0)
    q = np.ascontiguousarray(q, dtype=np.float64)
    F = np.zeros(q.shape)
    for i in range(q.shape[0]):
        E._mol_force(q, i, Ox, Oy, Oz, fp, F)
    return F


def _force_cmp(Fk, Fr):
    """Per-walker relative difference |F_k,i - F_r,i| / |F_r,i| (norm over the walker's 2 x 3 components; the
    prereg's 'per-walker forces ... relative'), plus the elementwise maximum and bitwise identity."""
    n = Fr.shape[0]
    d = (Fk - Fr).reshape(n, -1)
    nr = np.sqrt(np.sum(Fr.reshape(n, -1) ** 2, axis=1))
    per = np.sqrt(np.sum(d ** 2, axis=1)) / np.maximum(nr, 1e-300)
    elem = np.abs(Fk - Fr) / np.maximum(np.abs(Fr), 1e-300)
    return dict(max_rel_diff=float(per.max()), max_rel_diff_elementwise=float(elem.max()),
                max_abs_diff=float(np.max(np.abs(d))), bitwise=same_bits(Fk, Fr))


FALLBACK_FORCE_RTOL = 1e-13                  # prereg section 2.3 (LTA-only V-B1 fallback condition)


def first_step_forces(N, seed, threads, T_K=150.0, cfg=None):
    """V-B1 fallback evidence: the engine kernel's own per-walker forces (kernel_forces: step 0 from the evaluation
    phase, step 1 from the fused move + evaluation phase) vs the validated lta_ladder_numba._mol_force at the same
    positions (validated_forces).  ``force_condition_met`` is max per-walker relative difference <= 1e-13 over both
    steps.  ``helper_prange_diagnostic`` (a separate prange helper, not the kernel) is a labelled extra only."""
    base = dict(cfg if cfg is not None else production_engine_cfg())
    base.setdefault("T_K", T_K)
    c = resolve(base, N, seed, 1)
    q0, F0, q1, F1 = kernel_forces(c, threads)
    s0 = _force_cmp(F0, validated_forces(c, q0))
    s1 = _force_cmp(F1, validated_forces(c, q1))
    # labelled diagnostic: the old helper kernel (NOT evidence about the engine kernel)
    s = _setup(c, "abf")
    Fh = np.zeros((int(N), 2, 3))
    prev = numba.get_num_threads()
    numba.set_num_threads(int(threads))
    try:
        _forces_par(q0, s["Ox"], s["Oy"], s["Oz"], s["fp"], Fh, np.zeros(N), np.zeros(N), np.zeros(N, np.int64),
                    s["fp"][F_CPHI], s["fp"][F_CF], s["fp"][F_DCLIP], s["fp"][F_DPHI], int(c["n_grid"]))
    finally:
        numba.set_num_threads(prev)
    helper = _force_cmp(Fh, validated_forces(c, q0))
    mx = max(s0["max_rel_diff"], s1["max_rel_diff"])
    return dict(N=int(N), threads=int(threads), seed=int(seed),
                source="parallel.lta_parallel._kernel (st['F'] after _call): step 0 evaluation phase, step 1 fused "
                       "move + evaluation phase; ABF arm, replay noise",
                reference="lta_ladder_numba._mol_force (validated module's compiled entry point) per walker",
                step0_eval_phase=s0, step1_fused_phase=s1, max_rel_diff=mx,
                bitwise=bool(s0["bitwise"] and s1["bitwise"]),
                force_condition_met=bool(mx <= FALLBACK_FORCE_RTOL), force_rtol=FALLBACK_FORCE_RTOL,
                helper_prange_diagnostic=dict(helper, note="separate prange helper (_forces_par), NOT the engine "
                                                           "kernel; diagnostic only, not V-B1 fallback evidence"))


# ---- RNG battery (V-B3) -------------------------------------------------------------------------------------
@_jit()
def _cacc(cs, r, a, b):
    cs[r, 0] += 1.0
    cs[r, 1] += a
    cs[r, 2] += b
    cs[r, 3] += a * a
    cs[r, 4] += b * b
    cs[r, 5] += a * b


@_jit()
def _battery_sums(z):
    """z (S, N, 6) counter-mode normals in kernel order.  Raw power sums (blockwise per step) and the bivariate
    sums [n, sum a, sum b, sum a^2, sum b^2, sum ab] of four pair families: 0 lag-1 across steps (same slot and
    component), 1 adjacent slots (same step and component), 2 across words ((z0, z4), (z1, z5)), 3 Box-Muller
    partners ((z0, z1), (z2, z3), (z4, z5))."""
    S, N, K = z.shape
    m = np.zeros(4)
    cs = np.zeros((4, 6))
    lm = np.zeros(4)
    lc = np.zeros((4, 6))
    for s in range(S):
        lm[:] = 0.0
        lc[:, :] = 0.0
        for i in range(N):
            for k in range(K):
                v = z[s, i, k]
                v2 = v * v
                lm[0] += v
                lm[1] += v2
                lm[2] += v2 * v
                lm[3] += v2 * v2
                if s + 1 < S:
                    _cacc(lc, 0, v, z[s + 1, i, k])
                if i + 1 < N:
                    _cacc(lc, 1, v, z[s, i + 1, k])
            _cacc(lc, 2, z[s, i, 0], z[s, i, 4])
            _cacc(lc, 2, z[s, i, 1], z[s, i, 5])
            _cacc(lc, 3, z[s, i, 0], z[s, i, 1])
            _cacc(lc, 3, z[s, i, 2], z[s, i, 3])
            _cacc(lc, 3, z[s, i, 4], z[s, i, 5])
        m += lm
        cs += lc
    return m, cs


def _p_two_sided(zs):
    return float(math.erfc(abs(zs) / math.sqrt(2.0)))


def holm(pvals, alpha):
    """Holm step-down over the dict name -> p at FWER alpha.  Returns {name: dict(p, threshold, rejected)}."""
    names = sorted(pvals, key=lambda k: (pvals[k], k))
    m = len(names)
    out = {}
    stop = False
    for i, k in enumerate(names):
        thr = alpha / (m - i)
        rej = (not stop) and pvals[k] <= thr
        if not rej:
            stop = True
        out[k] = dict(p=float(pvals[k]), threshold=float(thr), rejected=bool(rej))
    return out


def battery_from_normals(z, alpha=0.01):
    """The 9 V-B3 (iii) statistics of z (S, N, 6) normals in kernel order ([step][slot][bead 0 xyz, bead 1 xyz];
    z0..z3 from word 0, z4, z5 from word 1), two-sided p-values, one Holm family at FWER alpha."""
    from scipy import stats
    z = np.ascontiguousarray(z, dtype=np.float64)
    m, cs = _battery_sums(z)
    n = float(z.size)
    mu = m[0] / n
    r1, r2, r3 = m[1] / n, m[2] / n, m[3] / n
    m2 = r1 - mu * mu
    m3 = r2 - 3.0 * mu * r1 + 2.0 * mu ** 3
    m4 = r3 - 4.0 * mu * r2 + 6.0 * mu * mu * r1 - 3.0 * mu ** 4
    g1 = m3 / m2 ** 1.5
    g2 = m4 / (m2 * m2) - 3.0
    stat = dict(mean=mu * math.sqrt(n), variance=(m2 - 1.0) / math.sqrt(2.0 / n), skewness=g1 / math.sqrt(6.0 / n),
                excess_kurtosis=g2 / math.sqrt(24.0 / n))
    corr = {}
    for name, r in (("lag1_steps", 0), ("lag1_slots", 1), ("lag1_words", 2), ("box_muller_partners", 3)):
        npair, sa, sb, saa, sbb, sab = cs[r]
        ma, mb = sa / npair, sb / npair
        rho = (sab / npair - ma * mb) / math.sqrt((saa / npair - ma * ma) * (sbb / npair - mb * mb))
        corr[name] = dict(r=float(rho), n_pairs=int(npair))
        stat[name] = math.atanh(rho) * math.sqrt(npair - 3.0)
    pv = {k: _p_two_sided(v) for k, v in stat.items()}
    ks = stats.kstest(z.ravel(), "norm")
    pv["ks_normal"] = float(ks.pvalue)
    stat["ks_normal"] = float(ks.statistic)
    h = holm(pv, alpha)
    return dict(n_normals=int(n), alpha_fwer=alpha, family=list(VB3_TESTS),
                tests="z-tests: mean sqrt(n); (var - 1) / sqrt(2/n); skew / sqrt(6/n); exkurt / sqrt(24/n); "
                      "Fisher atanh(r) sqrt(n_pairs - 3) for the four pair families; KS vs N(0, 1) (scipy kstest)",
                moments=dict(mean=float(mu), variance=float(m2), skewness=float(g1), excess_kurtosis=float(g2)),
                correlations=corr, statistics={k: float(v) for k, v in stat.items()}, pvalues=pv, holm=h,
                n_rejected=int(sum(v["rejected"] for v in h.values())),
                passed=not any(v["rejected"] for v in h.values()))


def rng_battery(n_normals=10 ** 8, seed=BITWISE_SEEDS[0], N=512, T_K=150.0, alpha=0.01, step0=0):
    """V-B3 (iii) on >= n_normals counter-mode LTA normals: the kernel's noise (parallel.philox.lta_noise_block, the
    same lta_normals6 the kernel calls) of moves step0 .. step0 + S - 1, S = ceil(n / (6 N)), key
    SeedSequence([seed, N, 150, 2]).  Uses numba's current thread count for the generation (no effect on values)."""
    k0, k1 = PX.lta_key(seed, N, T_K)
    S = -(-int(n_normals) // (6 * int(N)))
    z = PX.lta_noise_block(k0, k1, int(step0), S, int(N)).reshape(S, int(N), 6)
    out = battery_from_normals(z, alpha)
    out.update(S_steps=S, N=int(N), seed=int(seed), key=[str(int(k0)), str(int(k1))], step0=int(step0),
               source="parallel.philox.lta_noise_block (kernel noise layout)")
    return out


def philox_oracle_check(n_counters=10_000, keys=None, rng_seed=20261011):
    """V-B3 (i): numba Philox4x64-10 raw outputs vs numpy's Philox (exact) for n_counters per key: half
    production-structured counters (step, slot, word, 0) and half random 256-bit counters incl. carry / edge
    cases, plus a sequential run checked against one numpy stream.  Returns a dict with the mismatch counts."""
    if keys is None:
        keys = [PX.lta_key(15200, 512), PX.lta_key(15400, 16), PX.gateway_key(8400, 512),
                (np.uint64(0), np.uint64(2 ** 64 - 1))]
    rng = np.random.default_rng(rng_seed)
    out = []
    for (k0, k1) in keys:
        nh = n_counters // 2
        st = rng.integers(0, LTA_B, size=nh, dtype=np.int64)
        sl = rng.integers(0, 1024, size=nh, dtype=np.int64)
        wd = rng.integers(0, 2, size=nh, dtype=np.int64)
        c_prod = np.stack([st, sl, wd, np.zeros(nh, np.int64)], axis=1).astype(np.uint64)
        c_rand = rng.integers(0, 2 ** 64, size=(n_counters - nh, 4), dtype=np.uint64)
        edge = np.array([[0, 0, 0, 0], [2 ** 64 - 1] * 4, [2 ** 64 - 1, 0, 0, 0], [2 ** 64 - 1, 2 ** 64 - 1, 0, 0],
                         [0, 2 ** 64 - 1, 1, 0], [1, 0, 0, 0]], dtype=np.uint64)
        c_rand[:edge.shape[0]] = edge
        ctr = np.ascontiguousarray(np.concatenate([c_prod, c_rand]))
        got = PX.raw_block(np.uint64(k0), np.uint64(k1), ctr)
        bad = 0
        for i in range(ctr.shape[0]):
            if not np.array_equal(got[i], PX.numpy_raw(k0, k1, ctr[i])):
                bad += 1
        # sequential: one numpy stream from counter c - 1 gives blocks c, c+1, ... (c0 increments with carry)
        nseq = 4096
        c0 = np.array([2 ** 64 - 1000, 7, 1, 0], dtype=np.uint64)
        base = PX.ctr_to_int(c0)
        seq_ctr = np.ascontiguousarray(np.stack([PX.int_to_ctr(base + j) for j in range(nseq)]))
        bg = np.random.Philox(key=np.array([int(k0), int(k1)], dtype=np.uint64), counter=PX.int_to_ctr(base - 1))
        seq_np = bg.random_raw(4 * nseq).reshape(nseq, 4)
        seq_bad = int(np.sum(np.any(PX.raw_block(np.uint64(k0), np.uint64(k1), seq_ctr) != seq_np, axis=1)))
        py_bad = sum(1 for i in range(0, ctr.shape[0], 97)
                     if [int(x) for x in got[i]] != PX.python_philox(ctr[i], (int(k0), int(k1))))
        out.append(dict(key=[str(int(k0)), str(int(k1))], n_counters=int(ctr.shape[0]), mismatches=int(bad),
                        sequential_counters=nseq, sequential_mismatches=seq_bad,
                        python_crosscheck_counters=len(range(0, ctr.shape[0], 97)), python_mismatches=py_bad))
    return dict(keys=out, numpy_counter_convention="numpy Philox(counter=c - 1) -> first block = Philox(c)",
                passed=all(o["mismatches"] == 0 and o["sequential_mismatches"] == 0 and o["python_mismatches"] == 0
                           for o in out))


def production_counter_configs():
    """Every Experiment B / equal-budget production (system, N, n_steps, words): LTA 150 K B = 3.072e8 at
    N = 1..1024 (powers of two), 2 words; gateway B = 3.2768e9 at N = 1..512, 1 word."""
    cfgs = []
    for N in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        cfgs.append(("lta150", N, LTA_B // N, PX.LTA_WORDS))
    for N in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        cfgs.append(("gateway", N, 3_276_800_000 // N, PX.GATEWAY_WORDS))
    return cfgs


def counter_injectivity_check(exhaustive_N=512, exhaustive_steps=256):
    """V-B3 (ii): the analytic check for every production configuration, plus an exhaustive enumeration of a small
    block (every (step, slot, word) of exhaustive_steps x exhaustive_N x 2): distinct counters and distinct Philox
    output blocks under the production key."""
    rows = [dict(system=s, **PX.check_counter_injective(N, n, w)) for s, N, n, w in production_counter_configs()]
    st, sl, wd = np.meshgrid(np.arange(exhaustive_steps), np.arange(exhaustive_N), np.arange(2), indexing="ij")
    ctr = np.ascontiguousarray(np.stack([st.ravel(), sl.ravel(), wd.ravel(), np.zeros(st.size, np.int64)],
                                        axis=1).astype(np.uint64))
    k0, k1 = PX.lta_key(15200, exhaustive_N)
    outs = PX.raw_block(k0, k1, ctr)
    n_ctr_unique = np.unique(ctr, axis=0).shape[0]
    n_out_unique = np.unique(outs, axis=0).shape[0]
    return dict(configs=rows, exhaustive=dict(n=int(ctr.shape[0]), unique_counters=int(n_ctr_unique),
                                              unique_output_blocks=int(n_out_unique)),
                passed=all(r["injective"] for r in rows) and n_ctr_unique == ctr.shape[0]
                and n_out_unique == ctr.shape[0])


# =====================================================================================
# CLI
# =====================================================================================
def _json_dump(path, obj):
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def _run_validated(cfg, method, N, seed, n_steps, save_steps):
    c = resolve(cfg, N, seed, n_steps)
    t = time.perf_counter()
    ref = E.run_arm(c, method, save_steps)
    return ref, time.perf_counter() - t


def counter_map_check(cfg, method, N, seed, n_steps, grid, threads, base=None):
    """Counter mode == replay mode fed with parallel.philox.lta_noise_block over a whole run (every shared key):
    the noise the compiled kernel used IS the generator that the V-B3 oracle and battery test (lta_noise_block is
    compiled from philox.py's own cache).  ``base``: an existing counter-mode result of this job (any thread count,
    V-B2 makes them identical); otherwise it is run at ``threads``."""
    if base is None:
        base = run_arm_parallel(cfg, method, N, seed, n_steps, grid, threads=threads, noise="counter")
    c = resolve(cfg, N, seed, n_steps)
    k0, k1 = PX.lta_key(c["seed"], c["N"], c["T_K"])
    Nc = c["N"]
    inj = run_arm_parallel(cfg, method, N, seed, n_steps, grid, threads=threads, noise="replay",
                           _noise_override=lambda s0, m: PX.lta_noise_block(k0, k1, s0, m, Nc))
    bad = compare_shared(base, inj)
    return dict(N=int(N), method=method, n_steps=int(n_steps), threads=int(threads), seed=int(seed),
                kernel_equals_philox_module=not bad, keys_differing=bad,
                deaths=int(inj["total_replacement_events"]),
                reference="replay mode fed with parallel.philox.lta_noise_block(key, step0, n_moves, N)")


def cli_vb1(cases, threads, seed, out, min_deaths=50, max_doublings=4):
    """V-B1: replay mode bitwise vs lta_ladder_numba.run_arm (diagnostics off), production cfg, Exp B save grid.

    Lengths are the preregistered minima; an FR case whose validated run realises fewer than ``min_deaths`` deaths
    is lengthened mechanically (n doubled, decided on the validated engine alone, at most ``max_doublings``
    times) so that it meets the '>= 50 realised deaths' requirement; every attempt is recorded."""
    cfg = production_engine_cfg()
    rows = []
    for N, method, n in cases:
        attempts = []
        while True:
            grid = expb_save_grid(n)
            ref, wref = _run_validated(cfg, method, N, seed, n, grid)
            deaths = int(ref["total_replacement_events"])
            attempts.append(dict(n_steps=n, validated_deaths=deaths, validated_wall_s=wref))
            if method != "fr" or deaths >= min_deaths or len(attempts) > max_doublings:
                break
            print(json.dumps(dict(N=N, method=method, n_steps=n, deaths=deaths, action="double n_steps")),
                  flush=True)
            n *= 2
        for t in threads:
            res = run_arm_parallel(cfg, method, N, seed, n, grid, threads=t, noise="replay")
            bad = compare_shared(ref, res)
            row = dict(N=N, method=method, n_steps=n, threads=t, seed=seed, n_saves=int(grid.size),
                       deaths=int(res["total_replacement_events"]), fr_opps=int(res["n_fr_opps"]),
                       bitwise=not bad, keys_differing=bad, validated_wall_s=wref, length_attempts=attempts,
                       parallel_wall_total_s=float(res["wall_total_s"]), warmup_s=float(res["warmup_s"]),
                       requirements=dict(saves_ge_20=bool(grid.size >= 20),
                                         deaths_ge_50=(bool(res["total_replacement_events"] >= min_deaths)
                                                       if method == "fr" else None)))
            if bad:
                row["fallback"] = fallback_report(ref, res)
                row["first_step_forces"] = first_step_forces(N, seed, t)       # the kernel's own forces
                row["fallback"]["force_condition_met"] = row["first_step_forces"]["force_condition_met"]
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("N", "method", "n_steps", "threads", "bitwise", "deaths",
                                                  "n_saves")}), flush=True)
    fsf = [first_step_forces(N, seed, t) for N in sorted({c[0] for c in cases}) for t in threads]
    ok = all(r["bitwise"] and r["requirements"]["saves_ge_20"] and r["requirements"]["deaths_ge_50"] is not False
             for r in rows)
    summary = dict(test="V-B1 replay identity (LTA 150 K)", engine=ENGINE, rows=rows, first_step_forces=fsf,
                   first_step_forces_source="the engine kernel's own st['F'] (evaluation phase at step 0, fused "
                                            "move + evaluation phase at step 1) vs lta_ladder_numba._mol_force",
                   passed_bitwise=all(r["bitwise"] for r in rows), passed=ok,
                   length_rule="preregistered minimum; FR case doubled on the validated engine until >= 50 deaths",
                   meta=json.loads(_meta_static()), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if out:
        _json_dump(os.path.join(out, "lta_vb1.json"), summary)
    return summary


def _first_death_step(cfg, N, seed, n_steps, grid, res_fr, threads):
    """Exact first realised-death step d of the counter-mode FR arm (dense re-run inside the bracketing saves)."""
    cd = np.asarray(res_fr["cum_deaths"])
    idx = np.flatnonzero(cd > 0)
    if idx.size == 0:
        return None
    hi = int(grid[idx[0]])
    lo = int(grid[idx[0] - 1]) if idx[0] > 0 else 0
    dense = np.arange(lo, hi + 1, dtype=np.int64)
    r = run_arm_parallel(cfg, "fr", N, seed, hi, dense, threads=threads, noise="counter", timers=False)
    return int(dense[np.flatnonzero(np.asarray(r["cum_deaths"]) > 0)[0]])


def cli_vb2(cases, threads, seed, out, vb5_threads=16, min_deaths=50, max_doublings=4):
    """V-B2 counter-mode thread invariance and V-B5 ABF/FR pairing before FR's first realised death.  The same
    length rule as V-B1 (an FR case is doubled, decided on the first thread count's run, until >= 50 deaths)."""
    cfg = production_engine_cfg()
    rows = []
    runs = {}
    cmap = []
    for N, method, n in cases:
        attempts = []
        while True:
            grid = expb_save_grid(n)
            base = run_arm_parallel(cfg, method, N, seed, n, grid, threads=threads[0], noise="counter")
            deaths = int(base["total_replacement_events"])
            attempts.append(dict(n_steps=n, deaths=deaths))
            if method != "fr" or deaths >= min_deaths or len(attempts) > max_doublings:
                break
            n *= 2
        runs[(N, method)] = (base, n, grid)
        for t in threads:
            res = base if t == threads[0] else run_arm_parallel(cfg, method, N, seed, n, grid, threads=t,
                                                                 noise="counter")
            bad = compare_shared(base, res)
            rows.append(dict(N=N, method=method, n_steps=n, threads=t, vs_threads=threads[0], invariant=not bad,
                             keys_differing=bad, deaths=int(res["total_replacement_events"]),
                             length_attempts=attempts, wall_total_s=float(res["wall_total_s"])))
            print(json.dumps({k: rows[-1][k] for k in ("N", "method", "threads", "invariant", "deaths")}), flush=True)
        cmap.append(counter_map_check(cfg, method, N, seed, n, grid, threads[-1], base=base))
        print(json.dumps({k: cmap[-1][k] for k in ("N", "method", "n_steps", "threads", "kernel_equals_philox_module",
                                                   "deaths")}), flush=True)
    vb5 = []
    for (N, method), (rfr, n, grid) in sorted(runs.items()):
        if method != "fr" or (N, "abf") not in runs:
            continue
        rab, n_ab, _ = runs[(N, "abf")]
        if n_ab != n:       # the FR case was lengthened: pair it with an ABF run of the same length and grid
            rab = run_arm_parallel(cfg, "abf", N, seed, n, grid, threads=vb5_threads, noise="counter", timers=False)
        d = _first_death_step(cfg, N, seed, n, grid, rfr, vb5_threads)
        row = dict(N=N, n_steps=n, first_death_step=d)
        if d is None:
            row.update(saves_identical=compare_shared(rab, rfr) == [], note="no realised death in the run")
        else:
            pre = grid < d
            keys = ("M_all", "C_all", "M_prod", "C_prod")
            row["n_saves_before_death"] = int(pre.sum())
            row["saves_before_death_identical"] = all(same_bits(np.asarray(rab[k])[pre], np.asarray(rfr[k])[pre])
                                                      for k in keys)
            a = run_arm_parallel(cfg, "abf", N, seed, d - 1, np.array([d - 1]), threads=vb5_threads, noise="counter",
                                 timers=False)
            f = run_arm_parallel(cfg, "fr", N, seed, d - 1, np.array([d - 1]), threads=vb5_threads, noise="counter",
                                 timers=False)
            row["state_at_d_minus_1_identical"] = (same_bits(a["q_final"], f["q_final"])
                                                   and same_bits(a["M_all_final"], f["M_all_final"])
                                                   and same_bits(a["C_all_final"], f["C_all_final"]))
            a2 = run_arm_parallel(cfg, "abf", N, seed, d, np.array([d]), threads=vb5_threads, noise="counter",
                                  timers=False)
            f2 = run_arm_parallel(cfg, "fr", N, seed, d, np.array([d]), threads=vb5_threads, noise="counter",
                                  timers=False)
            row["state_at_d_differs"] = not same_bits(a2["q_final"], f2["q_final"])
            row["passed"] = bool(row["saves_before_death_identical"] and row["state_at_d_minus_1_identical"]
                                 and row["state_at_d_differs"])
        vb5.append(row)
        print(json.dumps(row), flush=True)
    summary = dict(test="V-B2 counter thread invariance + V-B5 pairing (LTA 150 K)", engine=ENGINE, rows=rows,
                   vb2_passed=all(r["invariant"] for r in rows), vb5=vb5,
                   vb5_passed=all(r.get("passed", False) for r in vb5),
                   counter_map=cmap, counter_map_passed=bool(cmap) and all(r["kernel_equals_philox_module"]
                                                                          for r in cmap),
                   meta=json.loads(_meta_static()), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if out:
        _json_dump(os.path.join(out, "lta_vb2_vb5.json"), summary)
    return summary


VB3_KERNEL_CASES = ((16, 5_000), (512, 2_000))   # (N, n_steps) of the kernel-noise identity in --vb3


def cli_vb3(n_normals, out, seed=BITWISE_SEEDS[0], n_oracle=10_000):
    """V-B3 (i)-(iii) on parallel.philox, plus ``kernel_noise_identity``: the compiled kernel's counter-mode runs
    (production cfg, ABF, seed ``seed``, N 16 and 512 -- the battery's key at N 512 and its first 2000 steps) equal
    replay mode fed with parallel.philox.lta_noise_block, i.e. the oracle / battery describe the kernel's noise."""
    t = time.perf_counter()
    sc = kernel_selfcheck()
    cfg = production_engine_cfg()
    kn = [counter_map_check(cfg, "abf", N, seed, n, expb_save_grid(n), numba.get_num_threads())
          for N, n in VB3_KERNEL_CASES]
    i = philox_oracle_check(n_oracle)
    ii = counter_injectivity_check()
    iii = rng_battery(n_normals, seed=seed)
    kpass = all(r["kernel_equals_philox_module"] for r in kn)
    summary = dict(test="V-B3 RNG (LTA 150 K counter mode)", engine=ENGINE, oracle=i, injectivity=ii, battery=iii,
                   kernel_noise_identity=dict(rows=kn, passed=kpass), selfcheck=sc,
                   passed=bool(i["passed"] and ii["passed"] and iii["passed"] and kpass),
                   wall_s=time.perf_counter() - t,
                   meta=json.loads(_meta_static()), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if out:
        _json_dump(os.path.join(out, "lta_vb3.json"), summary)
    return summary


def cli_probe(N, threads, n_steps, seed, serial=True):
    """DEV ONLY (not the feasibility benchmark): counter-mode throughput at production knobs and the timer split."""
    cfg = production_engine_cfg()
    grid = expb_save_grid(n_steps)
    rows = []
    if serial:
        ref, w = _run_validated(cfg, "fr", N, seed, n_steps, grid)
        rows.append(dict(impl="validated serial", method="fr", threads=1, wall_s=w,
                         walker_steps_per_s=N * n_steps / w))
        print(json.dumps(rows[-1]), flush=True)
    for method in ("abf", "fr"):
        for t in threads:
            r = run_arm_parallel(cfg, method, N, seed, n_steps, grid, threads=t, noise="counter")
            w = float(r["wall_total_s"])
            rows.append(dict(impl="parallel counter", method=method, threads=t, wall_s=w,
                             walker_steps_per_s=N * n_steps / w, warmup_s=float(r["warmup_s"]),
                             split={k: round(float(r[k]), 4) for k in ("t_force_move_s", "t_sync_s", "t_abf_s",
                                                                       "t_fr_s", "t_save_s", "t_other_s")},
                             deaths=int(r["total_replacement_events"])))
            print(json.dumps(rows[-1]), flush=True)
    return rows


def _meta_static():
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        cpus = None
    tgt = E.compile_target()
    return json.dumps(dict(engine=ENGINE, numba=numba.__version__, numpy=np.__version__, host=platform.node(),
                           cpus=cpus, compile_target=dict(cpu_name=tgt["cpu_name"],
                                                          cpu_features_sha256=tgt["cpu_features_sha256"]),
                           numba_cache=cache_info(), selfcheck=_SELFCHECK, **_PROVENANCE), default=str)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="LTA 150 K walker-parallel engine: validation CLIs (Experiment B)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--vb1", action="store_true", help="V-B1 replay bitwise identity vs the validated engine")
    g.add_argument("--vb2", action="store_true", help="V-B2 counter thread invariance + V-B5 pairing")
    g.add_argument("--vb3", action="store_true", help="V-B3 RNG: Philox oracle, counter injectivity, battery")
    g.add_argument("--probe", action="store_true", help="dev-only throughput probe (NOT the feasibility benchmark)")
    ap.add_argument("--out", default=None, help="output directory for the JSON summary")
    ap.add_argument("--seed", type=int, default=BITWISE_SEEDS[0])
    ap.add_argument("--threads", type=str, default=None, help="comma-separated thread counts")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the preregistered V-B1/V-B2 run lengths (< 1 = smoke only, not the formal test)")
    ap.add_argument("--n-normals", type=int, default=10 ** 8)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--n-steps", type=int, default=22_000)
    ap.add_argument("--no-serial", action="store_true")
    a = ap.parse_args(argv)
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        ncpu = os.cpu_count()
    numba.set_num_threads(min(ncpu, numba.config.NUMBA_NUM_THREADS, 16))
    cases = tuple((N, m, max(1, int(round(n * a.scale)))) for N, m, n in VB1_CASES)
    if a.vb1:
        thr = tuple(int(x) for x in a.threads.split(",")) if a.threads else VB1_THREADS
        s = cli_vb1(cases, thr, a.seed, a.out)
        print(json.dumps(dict(passed=s["passed"], passed_bitwise=s["passed_bitwise"], scale=a.scale)))
        return 0 if s["passed"] else 1
    if a.vb2:
        thr = tuple(int(x) for x in a.threads.split(",")) if a.threads else VB2_THREADS
        s = cli_vb2(cases, thr, a.seed, a.out)
        print(json.dumps(dict(vb2_passed=s["vb2_passed"], vb5_passed=s["vb5_passed"],
                              counter_map_passed=s["counter_map_passed"], scale=a.scale)))
        return 0 if (s["vb2_passed"] and s["vb5_passed"] and s["counter_map_passed"]) else 1
    if a.vb3:
        s = cli_vb3(a.n_normals, a.out, seed=a.seed)
        print(json.dumps(dict(passed=s["passed"], oracle=s["oracle"]["passed"], injectivity=s["injectivity"]["passed"],
                              battery=s["battery"]["passed"], kernel_noise=s["kernel_noise_identity"]["passed"],
                              holm=s["battery"]["holm"])))
        return 0 if s["passed"] else 1
    if a.probe:
        thr = tuple(int(x) for x in a.threads.split(",")) if a.threads else (1, 4, 16)
        cli_probe(a.N, thr, a.n_steps, a.seed, serial=not a.no_serial)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
