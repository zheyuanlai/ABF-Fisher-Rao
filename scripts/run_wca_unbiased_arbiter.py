#!/usr/bin/env python
"""POST-HOC unbiased-MD arbiter (configs/wca_replica_ladder/unbiased_arbiter.json)."""
import os, sys, time, json
from concurrent.futures import ProcessPoolExecutor, as_completed
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_numba as wn
OUT = os.path.join(ROOT, "results", "wca_replica_ladder", "unbiased", "raw")


def job(seed):
    t0 = time.perf_counter()
    cfg = dict(wn.ACCEPTED_CFG, abf_bias_scale=0.0)
    res = wn.run_ladder_point(seed, 1, 122_880_000, [("unbiased", False)], cfg=cfg, cap_min=1)
    res["meta_json"] = json.dumps(dict(seed=seed, wall=time.perf_counter() - t0, abf_bias_scale=0.0))
    p = os.path.join(OUT, f"N1_s{seed}.npz"); tmp = p[:-4] + ".tmp.npz"
    wn.save_result(res, tmp, keep_state=False); os.replace(tmp, p)
    return seed, time.perf_counter() - t0


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [s for s in range(3500, 3532) if not os.path.exists(os.path.join(OUT, f"N1_s{s}.npz"))]
    wn.run_ladder_point(3500, 1, 50, [("unbiased", False)], cfg=dict(wn.ACCEPTED_CFG, abf_bias_scale=0.0), save_at=np.array([50]))
    print(len(jobs), "jobs", flush=True)
    with ProcessPoolExecutor(max_workers=32) as ex:
        for k, f in enumerate(as_completed([ex.submit(job, s) for s in jobs])):
            s, w = f.result(); print(f"[{k + 1}/{len(jobs)}] seed {s} {w / 60:.1f} min", flush=True)
    print("done", flush=True)
