"""A100 backend of the ethane/LTA histogram-ABF (+ uniform FR) engine (NSCC A100 study, backend 'G1').

Same algorithm as src/lta_ladder_numba.py (diagnostics off, native FR) and its walker-parallel CPU form
src/parallel/lta_parallel.py in counter mode; the kernel and its design are documented in src/gpu/lta_cuda.cu.  A job
is prepared exactly as lta_parallel does (lta_ladder_numba.make_cfg with diagnostics off; framework, _params, the
wrapped-Gaussian KDE matrix, the three PCG64 children of SeedSequence([seed, N, round(T)]): child 0 the initial
conditions, child 2 the FR uniform blocks (handed to the library as its numpy state); Philox key SeedSequence([seed,
N, round(T), 2])), a warm-up run of the same code path (CUDA context, module load, FR path) precedes the timed run,
and the result follows the lta_parallel contract (expB_common.check_contract) with engine 'lta_gpu/1'.

Launch configuration (G threads per walker, WB walkers per block) changes the timing only: every walker's arithmetic
is fixed by G (the LJ sum is split over the G threads), so results are bitwise identical across WB and differ at the
ulp level across G (the force sum order).  Default: configs/nscc_a100/gpu_launch.json (chosen by timing), else
default_launch.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
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
import lta_ladder_numba as E  # noqa: E402

ENGINE = "lta_gpu/1"
CU_SRC = os.path.join(HERE, "lta_cuda.cu")
CU_DEPS = (os.path.join(HERE, "gpu_common.cuh"),)
SOURCE_FILES = (os.path.abspath(__file__), CU_SRC, *CU_DEPS, os.path.join(HERE, "gpu_build.py"))
FSCALE_LOG2 = 43            # deposit fixed-point grid 2^-43: N x 480 x 2^43 < 2^62 for N <= 1024 (checked)
CHUNK_OPPS = 4096           # FR opportunities per uniform chunk
LAUNCH_JSON = os.path.join(ROOT, "configs", "nscc_a100", "gpu_launch.json")
NG = 180

_LIB = None
_BUILD = None


def load_lib():
    global _LIB, _BUILD
    if _LIB is not None:
        return _LIB
    lib, info = GB.load("lta_cuda", CU_SRC, CU_DEPS)
    vp, i32, i64, dbl = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_double
    lib.lt_gpu_run.argtypes = [vp] * 7 + [i64] + [vp] * 12 + [ctypes.c_char_p, i32]
    lib.lt_gpu_run.restype = i32
    lib.lt_gpu_fr_test.argtypes = [vp] * 6 + [i32, vp, vp, ctypes.c_char_p, i32]
    lib.lt_gpu_fr_test.restype = i32
    lib.lt_gpu_force_test.argtypes = [vp] * 6 + [ctypes.c_char_p, i32]
    lib.lt_gpu_force_test.restype = i32
    lib.lt_gpu_normals.argtypes = [vp, i64, i64, i32, vp, ctypes.c_char_p, i32]
    lib.lt_gpu_normals.restype = i32
    lib.lt_gpu_now.restype = dbl
    _LIB, _BUILD = lib, info
    return lib


def _sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sources_sha256():
    return {os.path.relpath(p, ROOT): _sha256_file(p) for p in SOURCE_FILES}


def resolve(cfg, N, seed, n_steps):
    """lta_parallel.resolve: the lta_ladder_numba cfg of this job (diagnostics off)."""
    cc = dict(cfg)
    T_K = float(cc.pop("T_K", 150.0))
    for k, v in (("N", N), ("seed", seed), ("n_steps", n_steps)):
        if k in cc and cc[k] is not None and int(cc[k]) != int(v):
            raise ValueError(f"cfg[{k!r}] = {cc[k]} disagrees with the argument {v}")
        cc.pop(k, None)
    for k in ("cap", "genealogy_window_steps", "n_trace", "diagnostics"):
        cc.pop(k, None)
    if cc.get("accumulate_u") or cc.get("record_events"):
        raise ValueError("accumulate_u / record_events are not implemented in the GPU engine")
    if int(cc.get("fr_stop_steps", -1)) >= 0 or cc.get("fr_controller") is not None:
        raise ValueError("fr_stop_steps / fr_controller are not implemented in the GPU engine")
    return E.make_cfg(T_K, int(N), int(seed), n_steps=int(n_steps), diagnostics=False, **cc)


def counter_key(seed, N, T_K):
    return np.random.SeedSequence([int(seed), int(N), int(round(float(T_K))), 2]).generate_state(2, np.uint64)


def setup(c, method):
    N = c["N"]
    O, a, L = E.framework(c["framework_npz"])
    fp, ip, beta, fr_on = E._params(c, method, O, a, L, 0)
    Kk, _, _ = E.kde_matrix(int(c["n_grid"]), float(c["kde_bandwidth"]))
    gens, rng_info = E.seed_streams(c["seed"], N, c["T_K"])
    q = np.ascontiguousarray(E.initial_conditions(gens[0], N, a, L, float(c["r0_bond"])), dtype=np.float64).copy()
    st = gens[2].bit_generator.state["state"]
    m = (1 << 64) - 1
    pcg = [st["state"] >> 64, st["state"] & m, st["inc"] >> 64, st["inc"] & m]
    # framework O atoms in spatial (Morton) order of a 4 x 4 x 4 cell grid: the 32 atoms a warp handles in one
    # iteration are close together, so whole warps skip the LJ branch beyond the cutoff.  Only the ORDER of the
    # (exact-zero outside the cutoff) pair terms in the force sum changes: ulp-level, like any other sum order.
    cell = np.floor(np.mod(O, L) / (L / 4.0)).astype(np.int64).clip(0, 3)
    morton = np.zeros(O.shape[0], dtype=np.int64)
    for bit in range(2):
        for ax in range(3):
            morton |= ((cell[:, ax] >> bit) & 1) << (3 * bit + ax)
    order = np.lexsort((np.arange(O.shape[0]), morton))
    Os = O[order]
    Oc = np.ascontiguousarray(np.stack([Os[:, 0], Os[:, 1], Os[:, 2]]), dtype=np.float64)   # (3, nO)
    return dict(O=Oc, O_order=order, a=a, L=L, fp=np.ascontiguousarray(fp, dtype=np.float64), ip_numba=ip, beta=beta, fr_on=fr_on,
                Kk=np.ascontiguousarray(Kk, dtype=np.float64), q0=q, pcg=pcg, rng_info=rng_info,
                key=counter_key(c["seed"], N, c["T_K"]))


def default_launch(N):
    """(G, WB) used when no tuned entry exists (block = G WB threads + one noise warp, <= 512)."""
    N = int(N)
    if N <= 16:
        return 256, 1
    if N <= 128:
        return 64, 2
    return 32, max(4, -(-N // 108))


def launch_for(N, method):
    if os.path.exists(LAUNCH_JSON):
        with open(LAUNCH_JSON) as f:
            tab = json.load(f).get("lta", {})
        e = tab.get(f"{method}{int(N)}") or tab.get(str(int(N)))
        if e:
            return int(e["G"]), int(e["WB"]), "configs/nscc_a100/gpu_launch.json"
    G, WB = default_launch(N)
    return G, WB, "default_launch heuristic"


def _ip(c, method, n_steps, launch, timers, chunk_opps, device, nO, **over):
    G, WB = launch
    v = dict(N=int(c["N"]), ng=int(c["n_grid"]), n_steps=int(n_steps), warm=int(c["warmup_steps"]),
             burn=int(c["burn_in_steps"]), fr_start=int(c["fr_start_steps"]), fr_every=int(c["fr_every"]),
             cap=int(c["cap"]), is_fr=1 if method == "fr" else 0)
    v.update(over)
    return np.array([v["N"], v["ng"], v["n_steps"], v["warm"], v["burn"], v["fr_start"], v["fr_every"], v["cap"],
                     v["is_fr"], int(G), int(WB), 1 if timers else 0, int(chunk_opps), int(device), FSCALE_LOG2,
                     int(nO)], dtype=np.int64)


def _core(lib, ip, s, save_steps):
    N, ng, cap = int(ip[0]), int(ip[1]), int(ip[7])
    S = int(save_steps.size)
    save_steps = np.ascontiguousarray(save_steps, dtype=np.int64)
    u64 = np.array([int(s["key"][0]), int(s["key"][1]), *s["pcg"], 0, 0], dtype=np.uint64)
    sv_acc = np.zeros((S, 4, ng))
    sv_i = np.full((S, 3), -1, dtype=np.int64)
    sv_evh = np.zeros((S, cap + 1), dtype=np.int64)
    sv_nfe = np.full(S, -1, dtype=np.int64)
    sv_wall = np.full(S, np.nan)
    q_out = np.zeros((N, 2, 3))
    anc = np.zeros(N, dtype=np.int64)
    acc = np.zeros((4, ng))
    bdh = np.zeros((2, ng))
    evh = np.zeros(cap + 1, dtype=np.int64)
    misc = np.zeros(8)
    tim = np.zeros(24)
    q0 = np.ascontiguousarray(s["q0"][:N], dtype=np.float64)
    err = ctypes.create_string_buffer(512)
    P = GB.ptr
    rc = lib.lt_gpu_run(P(ip), P(s["fp"]), P(u64), P(s["O"]), P(s["Kk"]), P(q0), P(save_steps), S, P(sv_acc),
                        P(sv_i), P(sv_evh), P(sv_nfe), P(sv_wall), P(q_out), P(anc), P(acc), P(bdh), P(evh), P(misc),
                        P(tim), err, 512)
    if rc != 0:
        raise RuntimeError(f"LTA GPU run failed: {err.value.decode()}")
    return dict(sv_acc=sv_acc, sv_i=sv_i, sv_evh=sv_evh, sv_nfe=sv_nfe, sv_wall=sv_wall, q=q_out, anc=anc, acc=acc,
                bdh=bdh, evh=evh, misc=misc, tim=tim)


def run_arm_gpu(cfg, method, N, seed, n_steps, save_steps, *, launch=None, timers=True, warmup=True, device=0,
                chunk_opps=CHUNK_OPPS):
    """Run ONE arm ('abf' | 'fr') of one LTA (N, seed) job on one GPU; returns the lta_parallel result contract."""
    t_entry = time.monotonic()
    lib = load_lib()
    if method not in E.METHODS:
        raise ValueError(f"method must be one of {E.METHODS}")
    c = resolve(cfg, N, seed, n_steps)
    N, n_steps, h = c["N"], c["n_steps"], float(c["h"])
    if method == "fr" and N < 2:
        raise ValueError("no FR arm at N = 1")
    if N * float(c["deposit_clip"]) * 2.0 ** FSCALE_LOG2 >= 2.0 ** 62:
        raise ValueError("N x deposit_clip exceeds the fixed-point range")
    save_steps = np.ascontiguousarray(np.asarray(save_steps, dtype=np.int64).ravel())
    if save_steps.size and (np.any(np.diff(save_steps) <= 0) or save_steps[0] < 0 or save_steps[-1] > n_steps):
        raise ValueError("save_steps must be sorted, unique and inside [0, n_steps]")
    s = setup(c, method)
    if method == "fr" and not s["fr_on"]:
        raise ValueError("FR arm with fr_on False (cap 0 or rate 0) is not implemented in the GPU engine")
    nO = int(s["O"].shape[1])
    if launch is None:
        G, WB, lsrc = launch_for(N, method)
        launch = (G, WB)
    else:
        launch = (int(launch[0]), int(launch[1]))
        lsrc = "explicit"
    if warmup:
        # the same code path on a short job: FR from step 2 at every step (as lta_parallel's warm-up)
        nw = int(min(n_steps, 8))
        ipw = _ip(c, method, nw, launch, timers, chunk_opps, device, nO, warm=2, burn=2, fr_start=2, fr_every=1)
        _core(lib, ipw, s, np.array([nw], dtype=np.int64))
    ip = _ip(c, method, n_steps, launch, timers, chunk_opps, device, nO)
    R = _core(lib, ip, s, save_steps)
    tim = R["tim"]
    t0, t_end = float(tim[0]), float(tim[1])
    wall_total = t_end - t0
    t0_epoch = time.time() - (lib.lt_gpu_now() - t0)
    khz = float(tim[8])
    cyc = tim[9:16]
    nan = float("nan")
    if timers and cyc[0] > 0:
        sec = 1.0 / (khz * 1e3)
        t_force, t_dep, t_sync, t_move, t_fr = (float(cyc[i] * sec) for i in (1, 2, 3, 4, 5))
        t_dev = float(cyc[0] * sec)
    else:
        t_force = t_dep = t_sync = t_move = t_fr = t_dev = nan
    sv_acc, sv_i, misc = R["sv_acc"], R["sv_i"], R["misc"]
    nscore = int(misc[2])
    dev = GB.device_info(lib, "lt", device)
    meta = dict(
        engine=ENGINE, validated_engine=E.ENGINE, system="lta", method=method, T_K=c["T_K"], N=N, seed=c["seed"],
        n_steps=n_steps, h=h, cap=c["cap"], fr_active=bool(s["fr_on"] and method == "fr"), noise_mode="counter",
        backend="G1",
        launch=dict(G=int(launch[0]), WB=int(launch[1]), blocks=int(tim[16]), threads=int(tim[17]), source=lsrc,
                    max_coresident_blocks=int(tim[18]), dyn_smem_bytes=int(tim[19])),
        counter_key=[str(int(v)) for v in s["key"]], counter_key_entropy=[c["seed"], N, int(round(c["T_K"])), 2],
        counter_layout="Philox4x64-10 counter (move step, slot, word 0|1, 0); Box-Muller (u0,u1),(u2,u3) per word; "
                       "first 6 of 8 normals in [bead][xyz] order; CUDA libm",
        rng=dict(s["rng_info"], fr="child 2: N + 2 cap uniforms per opportunity, numpy PCG64 reproduced in C++",
                 ic="child 0"),
        deposit=("clipped local force rounded to 2^-%d, exact 64-bit integer bin sums per step added once per step to "
                 "the float64 all-steps / production accumulators" % FSCALE_LOG2),
        conventions=dict(save="state at the start of the saved step; accumulators include that step's deposit",
                         n_force_evals_at_save="N (k + 1) at the save of step k",
                         wall_s_at_save="seconds since the timed start (after warm-up and device set-up), stamped "
                                        "after the save's device-to-host copy"),
        timers=bool(timers), device_seconds_of_block0=t_dev, device_clock_khz=khz,
        timers_extra=dict(t_kernel_calls=int(tim[6]), n_chunks=int(tim[7]), t_fr_uniform_wait_s=float(tim[4]),
                          setup_s=float(tim[2]), t_force_s=t_force, t_move_s=t_move,
                          cycles=[float(v) for v in cyc],
                          force_split_cycles=dict(pairs=float(tim[21]), reduce=float(tim[22]),
                                                  noise_divergence=float(tim[23]))),
        device=dev, build=_BUILD, sources_sha256=sources_sha256(), numpy=np.__version__,
        python=platform.python_version(), host=platform.node(), t0_epoch=t0_epoch, warmup=bool(warmup),
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), status="complete", **GB.git_state())
    acc = R["acc"]
    return dict(
        save_step=save_steps.copy(), save_t=save_steps * h, save_u=save_steps / float(n_steps),
        M_all=sv_acc[:, 0].copy(), C_all=sv_acc[:, 1].copy(), M_prod=sv_acc[:, 2].copy(), C_prod=sv_acc[:, 3].copy(),
        cum_deaths=sv_i[:, 0].copy(), cum_opps=sv_i[:, 1].copy(), cum_opps_with_event=sv_i[:, 2].copy(),
        events_per_opp_hist=R["sv_evh"].copy(), q_final=R["q"].copy(), anc_final=R["anc"].copy(),
        M_all_final=acc[0].copy(), C_all_final=acc[1].copy(), M_prod_final=acc[2].copy(), C_prod_final=acc[3].copy(),
        birth_hist=R["bdh"][0].copy(), death_hist=R["bdh"][1].copy(),
        total_replacement_events=np.int64(misc[3]), n_fr_opps=np.int64(misc[4]),
        fr_score_std=np.float64(misc[0] / max(nscore, 1)), fr_score_absmax=np.float64(misc[1]),
        n_force_evals=np.int64(N * (n_steps + 1)), cfg_json=json.dumps(c, sort_keys=True, default=str),
        wall_s_at_save=R["sv_wall"].copy(), n_force_evals_at_save=R["sv_nfe"].copy(),
        wall_total_s=np.float64(wall_total), warmup_s=np.float64(t0 - t_entry),
        t_force_move_s=np.float64(t_force + t_move) if timers else np.float64(nan), t_abf_s=np.float64(t_dep),
        t_sync_s=np.float64(t_sync), t_fr_s=np.float64(t_fr), t_save_s=np.float64(tim[3]),
        threads=np.int64(int(tim[16]) * int(tim[17])), noise_mode="counter", engine=ENGINE,
        meta=json.dumps(meta, sort_keys=True, default=str))


# ------------------------------------------------------------------------------------------------ test hooks
def counter_normals(key, step0, n_steps, N):
    """GPU counter-mode noise of moves step0 .. step0 + n_steps - 1: (n_steps, N, 2, 3)."""
    lib = load_lib()
    key = np.ascontiguousarray(key, dtype=np.uint64).reshape(2)
    out = np.empty((int(n_steps), int(N), 2, 3))
    err = ctypes.create_string_buffer(512)
    if lib.lt_gpu_normals(GB.ptr(key), int(step0), int(n_steps), int(N), GB.ptr(out), err, 512) != 0:
        raise RuntimeError(err.value.decode())
    return out


def force_test(c, q, G=32):
    """GPU forces (N, 2, 3), CV, local force and bin of positions q (N, 2, 3)."""
    lib = load_lib()
    s = setup(c, "abf")
    q = np.ascontiguousarray(q, dtype=np.float64)
    N = q.shape[0]
    ip = _ip(dict(c, N=N), "abf", 1, (G, 1), False, 1, 0, s["O"].shape[1])
    F = np.zeros((N, 2, 3))
    aux = np.zeros((N, 3))
    err = ctypes.create_string_buffer(512)
    if lib.lt_gpu_force_test(GB.ptr(ip), GB.ptr(s["fp"]), GB.ptr(s["O"]), GB.ptr(q), GB.ptr(F), GB.ptr(aux), err,
                             512) != 0:
        raise RuntimeError(err.value.decode())
    return F, aux[:, 0], aux[:, 1], aux[:, 2].astype(np.int64)


def fr_test(c, phin, jn, ublk, threads=256):
    """GPU FR selection on given post-move CVs / bins and one block of N + 2 cap uniforms: (n, dsel, ssel, sc,
    score std, max |s|)."""
    lib = load_lib()
    s = setup(c, "fr")
    N = int(phin.size)
    cap = int(c["cap"])
    ip = _ip(dict(c, N=N), "fr", 1, (1, 1), False, 1, 0, s["O"].shape[1])
    out = np.zeros(1 + 2 * cap, dtype=np.int32)
    dout = np.zeros(N + 2)
    err = ctypes.create_string_buffer(512)
    phin = np.ascontiguousarray(phin, dtype=np.float64)
    jn32 = np.ascontiguousarray(jn, dtype=np.int32)
    ublk = np.ascontiguousarray(ublk, dtype=np.float64)
    if lib.lt_gpu_fr_test(GB.ptr(ip), GB.ptr(s["fp"]), GB.ptr(s["Kk"]), GB.ptr(phin), GB.ptr(jn32), GB.ptr(ublk),
                          int(threads), GB.ptr(out), GB.ptr(dout), err, 512) != 0:
        raise RuntimeError(err.value.decode())
    n = int(out[0])
    return n, out[1:1 + n].astype(np.int64), out[1 + cap:1 + cap + n].astype(np.int64), dout[:N], dout[N], dout[N + 1]
