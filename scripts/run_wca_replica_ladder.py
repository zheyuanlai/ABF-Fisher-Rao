#!/usr/bin/env python
"""WCA dimer replica ladder at equal force-evaluation budget (configs/wca_replica_ladder/design.json).

CPU only (src/wca_numba.py), one (N, seed) job per process.
  python scripts/run_wca_replica_ladder.py --stage validation_f32,validation_f64,production --workers 128
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_numba as wn  # noqa: E402

DESIGN = os.path.join(ROOT, "configs", "wca_replica_ladder", "design.json")
OUT = os.path.join(ROOT, "results", "wca_replica_ladder")
ARMS = [("hist_abf", False), ("hist_fr_uniform", True)]


CFG_DT0005 = dict(wn.ACCEPTED_CFG, dt=0.0005, abf_warmup_steps=40000, estimator_burn_in_steps=40000,
                  fr_start_steps=80000, fr_every=20, ess_window_steps=16000)


def job(stage, N, seed, n_steps, arms, save_at, cap_min, f32, path, meta):
    t0 = time.perf_counter()
    cfg = CFG_DT0005 if stage.endswith("_dt0.0005") else wn.ACCEPTED_CFG
    res = wn.run_ladder_point(seed, N, n_steps, arms, cfg=cfg, save_at=save_at, cap_min=cap_min, accum_float32=f32)
    wall = time.perf_counter() - t0
    res["meta_json"] = json.dumps(dict(meta, stage=stage, N=N, seed=seed, n_steps=n_steps, budget=N * n_steps,
                                       cap_min=cap_min, accum_float32=f32, wall_seconds=wall, pid=os.getpid()))
    tmp = path[:-4] + ".tmp.npz"
    wn.save_result(res, tmp, keep_state=False)
    os.replace(tmp, path)
    return stage, N, seed, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="validation_f32")
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    d = json.load(open(DESIGN))
    B = int(d["budget"]["B_replica_steps"])
    rev = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    sha = subprocess.check_output(["sha256sum", DESIGN], text=True).split()[0]
    meta = dict(git_rev=rev, design_sha=sha, engine="src/wca_numba.py", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    jobs = []
    for st in a.stage.split(","):
        if st.startswith("validation"):
            f32 = st == "validation_f32"
            out = os.path.join(OUT, st, "raw"); os.makedirs(out, exist_ok=True)
            for sd in range(3100, 3116):
                jobs.append((st, 1024, sd, 120000, ARMS, np.arange(0, 120001, 2500), 0, f32, os.path.join(out, f"N1024_s{sd}.npz")))
        elif st == "production":
            out = os.path.join(OUT, "production", "raw"); os.makedirs(out, exist_ok=True)
            for N in d["ladder_N"]:            # longest jobs (2 arms) first, N = 1 (1 arm) last
                for sd in range(3300, 3316):
                    arms = ARMS if N >= 2 else ARMS[:1]
                    jobs.append(("production", N, sd, B // N, arms, None, 1, False, os.path.join(out, f"N{N}_s{sd}.npz")))
        elif st == "reference":
            # POST-HOC (2026-10-04 ~09:45 UTC): independent dynamics-consistent reference. The pooled serial ABF
            # runs converge to a stationary profile 0.076 (RMS) away from the v2 TI reference, concentrated in a
            # step across the trough; these seeds are used by NO compared arm. Two ABF arms per job (160 and
            # 640 bins) test that the step is not a histogram-resolution artefact.
            out = os.path.join(OUT, "reference", "raw"); os.makedirs(out, exist_ok=True)
            for sd in range(3400, 3432):
                jobs.append(("reference", 1, sd, B, [("hist_abf", False, 160), ("hist_abf_640", False, 640)], None, 1, False,
                             os.path.join(out, f"N1_s{sd}.npz")))
        elif st in ("production_dt0.0005", "reference_dt0.0005"):
            # configs/wca_replica_ladder/design_dt0.0005.json: same physical times, every step knob x4, budget x4
            B4 = 4 * B
            out = os.path.join(OUT, st, "raw"); os.makedirs(out, exist_ok=True)
            if st == "production_dt0.0005":
                for N in d["ladder_N"]:
                    for sd in range(3700, 3716):
                        jobs.append((st, N, sd, B4 // N, ARMS if N >= 2 else ARMS[:1], None, 1, False, os.path.join(out, f"N{N}_s{sd}.npz")))
            else:
                for sd in range(3800, 3832):
                    jobs.append((st, 1, sd, B4, ARMS[:1], None, 1, False, os.path.join(out, f"N1_s{sd}.npz")))
        else:
            raise SystemExit(st)
    todo = [j for j in jobs if not os.path.exists(j[-1])]
    print(f"{len(todo)}/{len(jobs)} jobs, {a.workers} workers, git {rev[:10]}, design {sha[:12]}", flush=True)
    wn.run_ladder_point(3300, 2, 50, ARMS, save_at=np.array([50]))          # compile in the parent before forking
    wn.run_ladder_point(3300, 2, 50, ARMS, cfg=CFG_DT0005, save_at=np.array([50]))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, *j, meta) for j in todo]
        for k, f in enumerate(as_completed(futs)):
            st, N, sd, w = f.result()
            print(f"  [{k + 1}/{len(todo)}] {st} N={N} seed={sd} {w / 60:.1f} min (elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
