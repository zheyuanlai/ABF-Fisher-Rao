#!/usr/bin/env python
"""Run the frozen WCA Fisher-Rao timing confirmation (deliverables/UROPS_2026_WCA_timing/frozen_config.json).

One process per (seed, arm), single-threaded numba, every arm from the seed's lattice initial condition and the
seed's Langevin noise stream (default_noise_seed(seed, 256)); each FR arm draws from its own frozen FR stream
(fr_stream_ids).  Raw results: $ABFFR_DATA_ROOT/results/urops_2026_wca_timing/raw/s<seed>_<arm>.npz (git-ignored),
written atomically; an existing file is never overwritten (resumable).  Refuses to run if the engine, the frozen
config or this script differ from the committed versions.

    python run_campaign.py --workers 112 [--arms abf,fr_start0,...] [--benchmark]
"""
import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_wca as CW  # noqa: E402


def job(args):
    seed, arm, commit = args
    wn = CW.engine()
    path = CW.raw_path(seed, arm)
    if os.path.exists(path):
        return seed, arm, "kept", 0.0
    arms, cfg, kw = CW.arm_job(arm)
    save = CW.save_grid()
    n = CW.s(CW.T)
    t0 = time.perf_counter()
    res = wn.run_ladder_point(seed, CW.N, n, arms, cfg=cfg, save_at=save, cap_min=0, **kw)
    wall = time.perf_counter() - t0
    A = {a["name"]: a for a in CW.frozen()["arms"]}[arm]
    res["meta_json"] = json.dumps(dict(seed=seed, arm=arm, arm_spec=A, N=CW.N, T=CW.T, dt=CW.DT, n_steps=n, commit=commit,
                                       engine_sha256=CW.sha256_file(CW.engine_path()), wall_s=wall, precision="float64",
                                       device="cpu", frozen_config_sha256=CW.sha256_file(CW.FROZEN), kwargs=kw,
                                       finite_state=bool(np.all(np.isfinite(res["q_final"])))))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    wn.save_result(res, tmp, keep_state=False)
    os.replace(tmp, path)
    return seed, arm, "ran", wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--arms", default=None, help="comma list (default: the frozen arm list, in priority order)")
    ap.add_argument("--allow-dirty", action="store_true", help="debug only; never for the confirmation")
    a = ap.parse_args()
    fz = CW.frozen()
    tracked = [os.path.relpath(CW.engine_path(), CW.REPO), os.path.relpath(CW.FROZEN, CW.REPO), os.path.relpath(__file__, CW.REPO),
               os.path.relpath(os.path.join(HERE, "common_wca.py"), CW.REPO)]
    if CW.git_dirty(tracked) and not a.allow_dirty:
        raise SystemExit(f"uncommitted changes in {tracked}: commit the frozen design first")
    commit = CW.git_head()
    arms = a.arms.split(",") if a.arms else [x["name"] for x in fz["arms"] if x["run"]]
    seeds = [int(x) for x in fz["seeds"]]
    jobs = [(sd, arm, commit) for arm in arms for sd in seeds if not os.path.exists(CW.raw_path(sd, arm))]
    print(f"commit {commit}; {len(jobs)} jobs ({len(arms)} arms x {len(seeds)} seeds, minus existing)", flush=True)
    wn = CW.engine()
    wn.run_ladder_point(seeds[0], 4, 200, [("w", True)], cfg=dict(CW.validated_cfg(0.0)), save_at=np.array([0, 200]),
                        cap_min=0, fr_stream_ids=[0], fr_stop_steps=100)          # compile once (numba cache)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(job, j) for j in jobs]):
            sd, arm, st, w = f.result()
            print(f"{time.strftime('%H:%M:%S')} seed {sd} {arm}: {st} {w / 3600:.2f} h", flush=True)
    print(f"done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
