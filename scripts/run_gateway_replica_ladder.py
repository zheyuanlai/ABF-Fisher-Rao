#!/usr/bin/env python
"""Gateway replica ladder at equal force-evaluation budget (docs/GATEWAY_REPLICA_LADDER.md).

Design frozen in configs/gateway_replica_ladder/design.json.  CPU only (src/gateway_numba.py): one
(N, seed) job per process, all arms of a job paired on shared noise.

  python scripts/run_gateway_replica_ladder.py --stage validation --workers 16
  python scripts/run_gateway_replica_ladder.py --stage production --workers 96
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_numba as gn  # noqa: E402

DESIGN = os.path.join(ROOT, "configs", "gateway_replica_ladder", "design.json")
OUT = os.path.join(ROOT, "results", "gateway_replica_ladder")


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def job(stage, N, seed, n_steps, arms, cell, fr, save_at, path, meta):
    t0 = time.perf_counter()
    if save_at is None:
        res = gn.run_ladder_point(seed, N, n_steps, arms, cell, fr)
    else:
        res = run_with_save_at(seed, N, n_steps, arms, cell, fr, np.asarray(save_at, dtype=np.int64))
    wall = time.perf_counter() - t0
    meta = dict(meta, N=N, seed=seed, n_steps=n_steps, T=n_steps * cell["dt"], budget=N * n_steps,
                arms=[a[0] for a in arms], arm_use_fr=[bool(a[1]) for a in arms], arm_nbins=[int(a[2]) for a in arms],
                arm_score_mode=[int(a[3]) if len(a) > 3 else 0 for a in arms],
                cap=int(res["cap"]), noise_seed=int(res["noise_seed"]), n_fr_opportunities=int(res["n_fr"]),
                wall_seconds=wall, host=socket.gethostname(), pid=os.getpid())
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta_json=np.array(json.dumps(meta)), save_at=res["save_at"], u=res["u"],
                        M=res["M"], C=res["C"], ess=res["ess"], wmax=res["wmax"], P=res["P"],
                        die=res["die"], clone=res["clone"], maxbias=res["maxbias"],
                        X_final=res["X_final"], Y_final=res["Y_final"])
    os.replace(tmp, path)
    return stage, N, seed, wall


def run_with_save_at(seed, N, n_steps, arms, cell, fr, save_at):
    """run_ladder_point with an explicit save grid (validation: the GPU engine's save steps)."""
    import math
    dx = gn.grid_dx()
    kern, r = gn.gaussian_kernel_np(fr["eta"], dx)
    x0, y0 = gn.init_left(seed, N, cell["beta"], cell["omega_out"], cell["omega_in"], cell["s"])
    cap = max(int(fr["cap_min"]), int(math.floor(fr["max_event_fraction"] * N)))
    arm_fr = np.array([1 if a[1] else 0 for a in arms], dtype=np.int64)
    arm_nb = np.array([int(a[2]) for a in arms], dtype=np.int64)
    arm_sc = np.array([int(a[3]) if len(a) > 3 else 0 for a in arms], dtype=np.int64)
    noise_seed = int(1_000_003 * int(seed) + 7919 * int(N) + 17) % (2 ** 31 - 1)
    out = gn.simulate(noise_seed, int(N), int(n_steps), float(cell["dt"]), float(cell["beta"]), float(cell["H"]),
                      float(cell["omega_out"]), float(cell["omega_in"]), float(cell["s"]),
                      arm_fr, arm_nb, arm_sc, int(arm_nb.max()), float(fr["min_count"]),
                      float(fr["gamma"]), float(fr["ramp_steps"]), int(fr["fr_every"]), float(fr["score_clip"]), int(cap),
                      kern, int(r), float(dx), x0, y0, save_at, 4000, np.zeros((0, N)), False)
    M, C, ess, wmax, P, die, clone, maxbias, X, Y, n_fr = out
    return dict(save_at=save_at, u=save_at / float(n_steps), M=M, C=C, ess=ess, wmax=wmax, P=P, die=die, clone=clone,
                maxbias=maxbias, X_final=X, Y_final=Y, n_fr=n_fr, cap=cap, noise_seed=noise_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="validation")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    d = json.load(open(DESIGN))
    c = d["cell"]
    cell = dict(beta=c["beta"], H=c["H"], omega_out=c["omega_out"], omega_in=c["omega_in"], s=c["s"], dt=c["dt"])
    fru = d["fr_uniform"]
    B = int(d["budget"]["B_walker_steps"])
    design_sha = subprocess.check_output(["sha256sum", DESIGN], text=True).split()[0]
    meta0 = dict(stage=a.stage, git_rev=git_rev(), design_sha=design_sha, cell=cell,
                 timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), engine="src/gateway_numba.py")
    jobs = []
    if a.stage == "validation":
        fr = dict(fru, min_count=1.0, cap_min=0)          # the accepted law exactly (cap floor(0.08 N))
        arms = [("abf_h45", False, 45), ("fr_h45", True, 45)]
        n_steps = 100000
        save_at = [st + 1 for st in range(n_steps) if st % 500 == 0 or st == n_steps - 1]
        out = os.path.join(OUT, "validation", "raw"); os.makedirs(out, exist_ok=True)
        for sd in range(5300, 5316):
            jobs.append(("validation", 2048, sd, n_steps, arms, cell, fr, save_at, os.path.join(out, f"N2048_s{sd}.npz")))
    elif a.stage == "production":
        fr = dict(fru, min_count=1.0)
        out = os.path.join(OUT, "production", "raw"); os.makedirs(out, exist_ok=True)
        lo, hi = 7100, 7131
        # longest-first is irrelevant here (every job costs ~ the same B); interleave N for early coverage
        for sd in range(lo, hi + 1):
            for N in d["ladder_N"]:
                arms = [tuple(x) for x in d["arms"] if (N >= 2 or not x[1])]
                assert B % N == 0
                jobs.append(("production", int(N), sd, B // N, arms, cell, fr, None, os.path.join(out, f"N{N}_s{sd}.npz")))
    elif a.stage == "score_fix":
        # EXPLORATORY (configs/gateway_replica_ladder/exploratory_score_fix.json)
        fr = dict(fru, min_count=1.0)
        out = os.path.join(OUT, "score_fix", "raw"); os.makedirs(out, exist_ok=True)
        arms = [("abf_h180", False, 180, 0), ("fr_h180", True, 180, 0), ("fr_emp_h180", True, 180, 1), ("fr_loo_h180", True, 180, 2)]
        for sd in range(7100, 7132):
            for N in (1024, 256, 64, 16, 4):
                jobs.append(("score_fix", N, sd, B // N, arms, cell, fr, None, os.path.join(out, f"N{N}_s{sd}.npz")))
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    todo = [j for j in jobs if a.overwrite or not os.path.exists(j[-1])]
    print(f"[{a.stage}] {len(todo)}/{len(jobs)} jobs, {a.workers} workers, git {meta0['git_rev'][:10]} design {design_sha[:12]}", flush=True)
    gn.run_ladder_point(1, 4, 100, [("abf", False, 45), ("fr", True, 45)], cell, dict(fru, min_count=1.0, cap_min=1))  # compile cache
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, *j, meta0) for j in todo]
        for k, f in enumerate(as_completed(futs)):
            st, N, sd, w = f.result()
            print(f"  [{k + 1}/{len(todo)}] {st} N={N} seed={sd} {w:.1f}s  (elapsed {time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
