"""A100 backend of the entropic-gateway histogram-ABF (+ uniform FR) engine (NSCC A100 study, backend 'G1').

Same algorithm as src/gateway_ladder_numba.py (diagnostics off) and its walker-parallel CPU form
src/parallel/gateway_parallel.py in counter mode; the kernel and its design are documented in src/gpu/gw_cuda.cu.
This module prepares a run exactly as gateway_parallel does (gateway_ladder_numba._prepare: cfg, knobs, save grid,
init_left initial conditions; Philox key SeedSequence([seed, N, 2]); FR stream PCG64(SeedSequence([seed, N, 1])),
handed to the library as its numpy state), runs a warm-up of the same code path (CUDA context, module load, first
launch), then the timed run, and returns the gateway_parallel result contract (scripts/final_campaign/expB_common
check_contract) with engine 'gateway_gpu/1'.

Launch configuration (blocks, threads, wpt) is an engineering knob: it changes the timing only -- every walker's
arithmetic is the same and the histogram reductions are exact integer sums, so results are bitwise identical for
every configuration (tested).  Default: LAUNCH_TABLE (chosen by timing on the A100, configs/nscc_a100/gpu_launch.json
if present), else the heuristic in default_launch.

Timing: the timed start t0 is after the warm-up and the device set-up (allocation, uploads); wall_s_at_save are
CLOCK_MONOTONIC seconds since t0, stamped after the save's device-to-host copy completed (the estimate is on the
host); wall_total_s ends after the final state copy.  Device phase split (block 0, thread 0, clock64): deposit +
accumulator update (t_abf_s), grid-barrier waits (t_sync_s), bias read + noise + move + force (t_force_move_s), FR
including its barrier (t_fr_s), scaled from cycles to seconds by the device's reported clock; t_save_s is the host
time spent recording saves.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gpu_build as GB  # noqa: E402
import gateway_ladder_numba as gl  # noqa: E402

ENGINE = "gateway_gpu/1"
CU_SRC = os.path.join(HERE, "gw_cuda.cu")
CU_DEPS = (os.path.join(HERE, "gpu_common.cuh"),)
SOURCE_FILES = (os.path.abspath(__file__), CU_SRC, *CU_DEPS, os.path.join(HERE, "gpu_build.py"))
FSCALE_LOG2 = 34            # deposit fixed-point grid 2^-34 (|f| <= 2^62 / (2^34 N) guarded on the device)
CHUNK_OPPS = 256            # FR opportunities per uniform chunk (as gateway_parallel)
LAUNCH_JSON = os.path.join(ROOT, "configs", "nscc_a100", "gpu_launch.json")

_LIB = None
_BUILD = None


def load_lib():
    global _LIB, _BUILD
    if _LIB is not None:
        return _LIB
    lib, info = GB.load("gw_cuda", CU_SRC, CU_DEPS)
    vp, i32, i64, dbl = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_double
    lib.gw_gpu_run.argtypes = [vp, vp, vp, vp, i32, vp, vp, vp, i64] + [vp] * 12 + [ctypes.c_char_p, i32]
    lib.gw_gpu_run.restype = i32
    lib.gw_gpu_fr_test.argtypes = [vp, vp, vp, vp, i32, vp, vp, dbl, i32, vp, vp, ctypes.c_char_p, i32]
    lib.gw_gpu_fr_test.restype = i32
    lib.gw_gpu_normals.argtypes = [vp, i64, i64, i32, vp, ctypes.c_char_p, i32]
    lib.gw_gpu_normals.restype = i32
    lib.gw_pcg64_doubles.argtypes = [vp, i64, vp]
    lib.gw_pcg64_doubles.restype = None
    lib.gw_gpu_now.restype = dbl
    _LIB, _BUILD = lib, info
    return lib


def _sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sources_sha256():
    return {os.path.relpath(p, ROOT): _sha256_file(p) for p in SOURCE_FILES}


# ----------------------------------------------------------------------------------------------- RNG helpers
def counter_key(seed, N):
    """Philox key of the gateway counter mode (= gateway_parallel.counter_key)."""
    return np.random.SeedSequence([int(seed), int(N), 2]).generate_state(2, np.uint64)


def pcg64_state_words(seed, N):
    """numpy PCG64(SeedSequence([seed, N, 1])) state as [state_hi, state_lo, inc_hi, inc_lo] (the FR stream)."""
    bg = np.random.PCG64(np.random.SeedSequence(gl.fr_seed_entropy(seed, N)))
    st = bg.state["state"]
    m = (1 << 64) - 1
    return [st["state"] >> 64, st["state"] & m, st["inc"] >> 64, st["inc"] & m]


def pcg64_doubles(words, n):
    lib = load_lib()
    w = np.asarray(words, dtype=np.uint64)
    out = np.empty(int(n))
    lib.gw_pcg64_doubles(GB.ptr(w), int(n), GB.ptr(out))
    return out


def counter_normals(key, step0, n_steps, N):
    """The kernel's counter-mode noise of moves step0 .. step0 + n_steps - 1 on the GPU: (n_steps, 2, N)."""
    lib = load_lib()
    key = np.ascontiguousarray(key, dtype=np.uint64).reshape(2)
    out = np.empty((int(n_steps), 2, int(N)))
    err = ctypes.create_string_buffer(512)
    if lib.gw_gpu_normals(GB.ptr(key), int(step0), int(n_steps), int(N), GB.ptr(out), err, 512) != 0:
        raise RuntimeError(err.value.decode())
    return out


# ----------------------------------------------------------------------------------------------- launch config
def default_launch(N):
    """Heuristic (blocks, threads, wpt, helpers) used when no tuned entry exists."""
    N = int(N)
    if N <= 1024:
        return 1, max(32, -(-N // 32) * 32), 1, 0
    blocks = -(-N // 256)
    return blocks, 256, 1, 0


def launch_for(N, method):
    if os.path.exists(LAUNCH_JSON):
        with open(LAUNCH_JSON) as f:
            tab = json.load(f).get("gateway", {})
        e = tab.get(f"{method}{int(N)}") or tab.get(str(int(N)))
        if e:
            return (int(e["blocks"]), int(e["threads"]), int(e["wpt"]), int(e.get("helpers", 0)),
                    "configs/nscc_a100/gpu_launch.json")
    b, t, w, hp = default_launch(N)
    return b, t, w, hp, "default_launch heuristic"


def _arrays(c, kn, N, n_steps, is_fr, blocks, threads, wpt, timers, chunk_opps, device, helpers=0):
    dx = gl.grid_dx()
    kern, r_eta = gl.gaussian_kernel_np(float(c["eta"]), dx)
    do_fr = 1 if (is_fr and N >= 2) else 0
    ip = np.array([N, int(c["nb"]), int(n_steps), int(blocks), int(threads), int(wpt), int(kn["fr_every"]),
                   int(kn["cap"]), int(kn["fr_block"]), do_fr, 1 if timers else 0, int(chunk_opps), int(device),
                   FSCALE_LOG2, int(helpers), 0], dtype=np.int64)
    dp = np.zeros(16)
    dp[:12] = [float(c["h"]), float(c["beta"]), float(c["H"]), float(c["omega_out"]), float(c["omega_in"]),
               float(c["s"]), float(c["min_count"]), float(c["gamma"]), float(kn["ramp_steps"]),
               float(c["score_clip"]), float(kn["dt_fr"]), float(dx)]
    kern = np.ascontiguousarray(kern, dtype=np.float64)
    if kern.size != 2 * int(r_eta) + 1:
        raise RuntimeError("KDE kernel length != 2 r + 1")
    return ip, dp, kern, do_fr


def _core(lib, c, kn, N, seed, n_steps, save_at, x0, y0, is_fr, launch, timers, chunk_opps, device):
    blocks, threads, wpt, helpers = launch
    ip, dp, kern, do_fr = _arrays(c, kn, N, n_steps, is_fr, blocks, threads, wpt, timers, chunk_opps, device,
                                  helpers)
    key = counter_key(seed, N)
    u64 = np.array([int(key[0]), int(key[1]), *pcg64_state_words(seed, N), 0, 0], dtype=np.uint64)
    nb = int(c["nb"])
    cap1 = int(kn["cap"]) + 1
    S = int(save_at.size)
    save_at = np.ascontiguousarray(save_at, dtype=np.int64)
    o_M = np.zeros((S, nb))
    o_C = np.zeros((S, nb))
    o_fr = np.zeros((S, 6), dtype=np.int64)
    o_evh = np.zeros((S, cap1), dtype=np.int64)
    o_dth = np.zeros((S, cap1), dtype=np.int64)
    o_nfe = np.zeros(S, dtype=np.int64)
    o_wall = np.full(S, np.nan)
    xo = np.zeros(N)
    yo = np.zeros(N)
    evh = np.zeros(cap1, dtype=np.int64)
    dth = np.zeros(cap1, dtype=np.int64)
    tim = np.zeros(24)
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    y0 = np.ascontiguousarray(y0, dtype=np.float64)
    err = ctypes.create_string_buffer(512)
    P = GB.ptr
    rc = lib.gw_gpu_run(P(ip), P(dp), P(u64), P(kern), int(kern.size), P(x0), P(y0), P(save_at), S,
                        P(o_M), P(o_C), P(o_fr), P(o_evh), P(o_dth), P(o_nfe), P(o_wall), P(xo), P(yo), P(evh),
                        P(dth), P(tim), err, 512)
    if rc != 0:
        raise RuntimeError(f"gateway GPU run failed: {err.value.decode()}")
    return dict(o_M=o_M, o_C=o_C, o_fr=o_fr, o_evh=o_evh, o_dth=o_dth, o_nfe=o_nfe, o_wall=o_wall, X=xo, Y=yo,
                evh=evh, dth=dth, tim=tim, key=key, do_fr=do_fr)


def run_arm_gpu(cfg, method, N, seed, n_steps, save_steps, *, launch=None, timers=True, warmup=True, device=0,
                chunk_opps=CHUNK_OPPS):
    """Run ONE arm ('abf' | 'fr') of the gateway (N, seed) job for n_steps loop steps on one GPU; returns the
    gateway_parallel result contract (module docstring).  launch = (blocks, threads, wpt) or None (table /
    heuristic)."""
    t_entry = time.monotonic()
    lib = load_lib()
    clean = {k: v for k, v in (cfg or {}).items() if k not in ("alpha", "lam", "model")}
    for k in ("alpha", "lam"):
        if cfg is not None and k in cfg and float(cfg[k]) != 1.0:
            raise ValueError("the gateway GPU engine implements MODEL_ORIG only")
    Pp = gl._prepare(clean, method, N, seed, n_steps, save_steps, None, None, False, None, None, False)
    c, kn, h, N, seed, n_steps, is_fr = Pp["c"], Pp["kn"], Pp["h"], Pp["N"], Pp["seed"], Pp["n_steps"], Pp["is_fr"]
    save_at, x0, y0 = Pp["save_at"], Pp["x0"], Pp["y0"]
    if launch is None:
        b, t, w, hp, lsrc = launch_for(N, method)
        launch = (b, t, w, hp)
    else:
        launch = tuple(int(v) for v in launch) + ((0,) if len(launch) == 3 else ())
        lsrc = "explicit"
    if warmup:
        fe = int(kn["fr_every"])
        n_w = int(min(n_steps, (2 * fe + 2) if is_fr else 512))
        _core(lib, c, kn, N, seed, n_w, np.array([n_w], dtype=np.int64), x0, y0, is_fr, launch, timers,
              chunk_opps, device)
    R = _core(lib, c, kn, N, seed, n_steps, save_at, x0, y0, is_fr, launch, timers, chunk_opps, device)
    tim = R["tim"]
    t0, t_end = float(tim[0]), float(tim[1])
    wall_total = t_end - t0
    t0_epoch = time.time() - (lib.gw_gpu_now() - t0)
    khz = float(tim[8])
    cyc = tim[9:16]
    nan = float("nan")
    if timers and cyc[0] > 0:
        sec = 1.0 / (khz * 1e3)
        t_dep, t_sync, t_move, t_fr = (float(cyc[1] * sec), float(cyc[2] * sec), float(cyc[3] * sec),
                                       float(cyc[4] * sec))
        t_dev = float(cyc[0] * sec)
    else:
        t_dep = t_sync = t_move = t_fr = t_dev = nan
    o_fr = R["o_fr"]
    dev = GB.device_info(lib, "gw", device)
    sig = dict(engine=ENGINE, cfg=c, method=method, N=N, seed=seed, n_steps=n_steps, noise="counter",
               counter_key=[int(v) for v in R["key"]], fr_entropy=gl.fr_seed_entropy(seed, N),
               save_sha=hashlib.sha256(np.ascontiguousarray(save_at).tobytes()).hexdigest(),
               x0_sha=hashlib.sha256(x0.tobytes()).hexdigest(), y0_sha=hashlib.sha256(y0.tobytes()).hexdigest(),
               sources_sha256=sources_sha256(), fscale_log2=FSCALE_LOG2)
    sig = json.loads(json.dumps(sig, sort_keys=True, default=str))
    meta = dict(
        engine=ENGINE, system="gateway", method=method, N=N, seed=seed, n_steps=n_steps, h=h, B=N * n_steps,
        noise_mode="counter", backend="G1",
        launch=dict(blocks=int(launch[0]), threads=int(launch[1]), wpt=int(launch[2]), helpers=int(launch[3]),
                    source=lsrc,
                    max_coresident_blocks=int(tim[18]), dyn_smem_bytes=int(tim[19])),
        noise_counter=("Philox4x64-10 (Random123), key SeedSequence([seed, N, 2]) = [%d, %d]; counter (loop step, "
                       "slot, 0, 0); u = (o >> 11) 2^-53; (zx, zy) = sqrt(-2 log(1 - u0)) (cos, sin)(2 pi u1) with "
                       "CUDA libm" % (int(R["key"][0]), int(R["key"][1]))),
        fr_stream="numpy PCG64(SeedSequence([seed, N, 1])) reproduced in C++ (tested bitwise); opportunity k owns "
                  "doubles [kL, (k+1)L)",
        fr_block_L=int(kn["fr_block"]), cap=int(kn["cap"]), fr_every=int(kn["fr_every"]),
        ramp_steps=float(kn["ramp_steps"]), dt_fr=float(kn["dt_fr"]),
        deposit=("f rounded to the grid 2^-%d, exact 64-bit integer bin sums per step, added once per step to the "
                 "float64 running accumulators (deterministic; launch-configuration invariant)" % FSCALE_LOG2),
        accumulator_convention="pre-deposit (gateway_ladder_numba): the save at k holds deposits of states 0..k-1",
        force_eval_convention="N per completed step",
        wall_definition=("CLOCK_MONOTONIC seconds from the timed start t0 (after the warm-up run, device set-up and "
                         "uploads); a save is stamped after its device-to-host copy completed"),
        timers=bool(timers), device_seconds_of_block0=t_dev, device_clock_khz=khz,
        timers_extra=dict(t_kernel_calls=int(tim[6]), n_chunks=int(tim[7]), t_fr_uniform_wait_s=float(tim[4]),
                          setup_s=float(tim[2]), cycles=[float(v) for v in cyc]),
        device=dev, build=_BUILD, sources_sha256=sources_sha256(), validated_engine=gl.ENGINE_VERSION,
        numpy=np.__version__, python=platform.python_version(), host=platform.node(), t0_epoch=t0_epoch,
        warmup=bool(warmup), run_signature=sig,
        run_signature_sha256=hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest(),
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), status="complete", **GB.git_state())
    return dict(
        save_step=save_at.copy(), save_t=save_at * h, save_u=save_at / float(n_steps),
        M_all=R["o_M"], C_all=R["o_C"],
        fr_deaths_cum=o_fr[:, 0], fr_kd_cum=o_fr[:, 1], fr_kc_cum=o_fr[:, 2], fr_opp_cum=o_fr[:, 3],
        fr_opp_event_cum=o_fr[:, 4], fr_opp_death_cum=o_fr[:, 5],
        fr_events_hist=R["evh"], fr_events_hist_cum=R["o_evh"], fr_deaths_hist=R["dth"],
        fr_deaths_hist_cum=R["o_dth"],
        X_final=R["X"], Y_final=R["Y"], max_abs_bias=np.float64(tim[5]),
        n_fr_steps=np.int64(o_fr[-1, 3] if o_fr.shape[0] else 0), n_force_evals=np.int64(N * n_steps),
        cfg=c, wall_s_at_save=R["o_wall"], n_force_evals_at_save=R["o_nfe"],
        wall_total_s=float(wall_total), warmup_s=float(t0 - t_entry),      # both CLOCK_MONOTONIC
        t_force_move_s=t_move, t_abf_s=t_dep, t_sync_s=t_sync, t_fr_s=t_fr, t_save_s=float(tim[3]),
        threads=int(launch[0]) * int(launch[1]), noise_mode="counter", engine=ENGINE,
        meta=json.dumps(meta, sort_keys=True, default=str))


def fr_test(N, X, ublk, g, cfg=None, threads=256):
    """gw_fr_block on given post-move X and one uniform block: returns (sel, kd, kc, nd, nc, deaths)."""
    lib = load_lib()
    c = gl.full_cfg(cfg)
    kn = gl.derived_knobs(c, int(N))
    ip, dp, kern, _ = _arrays(c, kn, int(N), 1, True, 1, threads, 1, False, 1, 0)
    u64 = np.zeros(8, dtype=np.uint64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    ublk = np.ascontiguousarray(ublk, dtype=np.float64)
    if ublk.size != int(kn["fr_block"]):
        raise ValueError("ublk must hold L doubles")
    sel = np.zeros(int(N), dtype=np.int32)
    info = np.zeros(8, dtype=np.int32)
    err = ctypes.create_string_buffer(512)
    P = GB.ptr
    if lib.gw_gpu_fr_test(P(ip), P(dp), P(u64), P(kern), int(kern.size), P(X), P(ublk), float(g), int(threads),
                          P(sel), P(info), err, 512) != 0:
        raise RuntimeError(err.value.decode())
    return sel.astype(np.int64), int(info[0]), int(info[1]), int(info[2]), int(info[3]), int(info[4])
