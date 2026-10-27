"""A100 throughput benchmark (Stage II) on the frozen benchmark windows of the hybrid short-T study.

    python scripts/nscc_a100/bench_gpu.py --system gateway --N 64,512,2048 --methods abf,fr --sweep \
        --out results/nscc_a100/bench/tune_gateway.jsonl

Each configuration runs the production configuration truncated to the window n_b(N) = max(bench_min_steps,
ceil(bench_walker_steps / N)) with its 20 uniform saves (+ fr_start for LTA) (hst_common.bench_window), in a fresh
engine call with the backend's own warm-up; c_ss = (wall at the last save - wall at s0) / (N (n_b - s0)) with s0 the
first save >= max(n_b / 2, fr_start) (hst_common.bench_metrics), so FR is active over the whole timed window.  One
JSON line per run: config, c_ss (ns / walker-step), us / step, the device phase split of block 0 and the launch.
--sweep tries the candidate launch configurations (timing only: results are bitwise independent of the launch).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "gpu"), os.path.join(ROOT, "scripts"),
           os.path.join(ROOT, "scripts", "final_campaign"), os.path.join(ROOT, "scripts", "hybrid_short_T"),
           os.path.join(ROOT, "scripts", "equal_budget")):
    if _p not in sys.path:
        sys.path.append(_p)

import hst_common as H  # noqa: E402


def gateway_candidates(N):
    """(blocks, threads, wpt, helpers) candidates: one block (if it fits) and multi-block grids of 32-256-thread
    walker blocks, each without and (if the block can double) with helper threads computing the noise."""
    out = []
    for wpt in (1, 2, 4):
        t = -(-N // wpt)
        t = max(32, -(-t // 32) * 32)
        if t <= 512 // wpt:
            out.append((1, t, wpt, 0))
            if 2 * t <= 512 // wpt:
                out.append((1, 2 * t, wpt, 1))
    for B in (2, 4, 8, 16, 32, 64):
        wpb = -(-N // B)
        if wpb < 32:
            break
        wpt = 1
        t = max(32, -(-wpb // 32) * 32)
        if t <= 512:
            out.append((B, t, wpt, 0))
            if 2 * t <= 512:
                out.append((B, 2 * t, wpt, 1))
    return sorted(set(out))


def lta_candidates(N):
    """(G, WB) candidates: threads per walker G and walkers per block WB (G WB a multiple of 32, G WB + 32 <= 512
    with the block's noise warp, WB <= 32, at most 216 blocks)."""
    out = []
    for G in (8, 16, 32, 64, 128, 256, 448):
        if G & (G - 1):
            continue
        for WB in (1, 2, 4, 6, 8, 10, 12, 14, 15, 16, 24, 32):
            T = G * WB
            if T + 32 > 512 or T % 32 or WB > max(1, N):
                continue
            B = -(-N // WB)
            if B > 108:            # one 128-register block per SM at most
                continue
            out.append((G, WB))
    return out


def scaled_window(spec, N, scale):
    """The benchmark window with bench_walker_steps scaled (launch tuning only; the official benchmark uses 1)."""
    N = int(N)
    n_b = max(int(spec["bench_min_steps"]), int(math.ceil(scale * spec["bench_walker_steps"] / N)))
    n_b = min(n_b, H.n_steps_for(spec, N))
    st = np.unique(np.clip(np.round(np.arange(1, H.BENCH_N_SAVES + 1) / H.BENCH_N_SAVES * n_b).astype(np.int64), 1,
                           n_b))
    if spec["kind"] == "lta" and 0 < spec["fr_start"] < n_b:
        st = np.unique(np.concatenate([st, [spec["fr_start"]]]))
    lo = max(n_b / 2.0, float(spec["fr_start"]))
    cand = st[st >= lo]
    s0 = int(cand[0]) if cand.size else int(st[0])
    return int(n_b), st, s0


def metrics(res, N, st, s0, n_b):
    k = np.asarray(res["save_step"], dtype=np.int64)
    w = np.asarray(res["wall_s_at_save"], dtype=float)
    if not np.array_equal(k, st):
        raise RuntimeError("bench save grid mismatch")
    i0 = int(np.flatnonzero(k == s0)[0])
    c_ss = (float(w[-1]) - float(w[i0])) / (N * (int(k[-1]) - int(k[i0])))
    return dict(c_ss=c_ss, c_total=float(res["wall_total_s"]) / (N * n_b))


def run_one(system, spec, method, N, launch, seed, long=False, scale=1.0):
    n_b, st, s0 = H.bench_window(spec, N, long) if scale == 1.0 else scaled_window(spec, N, scale)
    job = dict(N=int(N), long=long)
    if spec["kind"] == "gateway":
        import gateway_gpu as E
        res = E.run_arm_gpu(dict(spec["engine_cfg"]), method, int(N), int(seed), int(n_b), st, launch=launch)
    else:
        import lta_gpu as E
        import expB_common as C
        cfg = C.engine_cfg_for(spec, N, seed, n_b)
        res = E.run_arm_gpu(cfg, method, int(N), int(seed), int(n_b), st, launch=launch)
    bm = H.bench_metrics(res, spec, job) if scale == 1.0 else metrics(res, int(N), st, s0, n_b)
    meta = json.loads(res["meta"])
    return dict(system=system, method=method, N=int(N), seed=int(seed), launch=(list(launch) if launch else None),
                n_b=n_b, s0=s0, scale=scale,
                c_ss_ns=bm["c_ss"] * 1e9, us_per_step=bm["c_ss"] * N * 1e6, c_total_ns=bm["c_total"] * 1e9,
                wall_total_s=float(res["wall_total_s"]), warmup_s=float(res["warmup_s"]),
                t_force_move_s=float(res["t_force_move_s"]), t_abf_s=float(res["t_abf_s"]),
                t_sync_s=float(res["t_sync_s"]), t_fr_s=float(res["t_fr_s"]), t_save_s=float(res["t_save_s"]),
                device_s=meta.get("device_seconds_of_block0"), launch_meta=meta.get("launch"),
                calls=meta.get("timers_extra", {}).get("t_kernel_calls"), host=os.uname().nodename,
                force_split_cycles=meta.get("timers_extra", {}).get("force_split_cycles"),
                cycles=meta.get("timers_extra", {}).get("cycles"), clock_khz=meta.get("device_clock_khz"),
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=H.SYSTEMS)
    ap.add_argument("--N", default=None, help="comma list (default: the ladder)")
    ap.add_argument("--methods", default="abf,fr")
    ap.add_argument("--sweep", action="store_true", help="try every candidate launch (default: the table entry)")
    ap.add_argument("--launch", default=None, help="explicit launch, e.g. 8,256,1 (gateway) or 32,4 (LTA)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed-index", type=int, default=0, help="index into the system's bench_seeds")
    ap.add_argument("--scale", type=float, default=1.0, help="window scale (launch tuning only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = H.system_spec(a.system)
    Ns = [int(v) for v in a.N.split(",")] if a.N else sorted(spec["N_ladder"])
    methods = a.methods.split(",")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cand = gateway_candidates if spec["kind"] == "gateway" else lta_candidates
    with open(a.out, "a") as fh:
        for r in range(a.repeats):
            seed = spec["bench_seeds"][(a.seed_index + r) % len(spec["bench_seeds"])]
            for N in Ns:
                if a.launch:
                    launches = [tuple(int(v) for v in a.launch.split(","))]
                elif a.sweep:
                    launches = cand(N)
                else:
                    launches = [None]
                for method in methods:
                    if method == "fr" and N < 2:
                        continue
                    for L in launches:
                        try:
                            row = run_one(a.system, spec, method, N, L, seed, scale=a.scale)
                        except Exception as e:  # noqa: BLE001 (a launch may not fit: record and go on)
                            row = dict(system=a.system, method=method, N=N, launch=L, error=repr(e)[:300])
                        row["repeat"] = r
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        if "error" in row:
                            print(f"ERR {a.system} {method}{N} {L}: {row['error']}", flush=True)
                        else:
                            print(f"{a.system} {method}{N:<5d} launch {str(row['launch_meta'] and [row['launch_meta'].get(k) for k in ('blocks', 'threads', 'wpt', 'helpers', 'G', 'WB')]):<32s} "
                                  f"c_ss {row['c_ss_ns']:9.2f} ns  {row['us_per_step']:9.3f} us/step  "
                                  f"[fm {row['t_force_move_s']:.3f} abf {row['t_abf_s']:.3f} sync {row['t_sync_s']:.3f} "
                                  f"fr {row['t_fr_s']:.3f} save {row['t_save_s']:.3f} / wall {row['wall_total_s']:.3f} s]",
                                  flush=True)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
