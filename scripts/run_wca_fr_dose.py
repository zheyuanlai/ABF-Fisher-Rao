#!/usr/bin/env python
"""POST-HOC FR dose-response diagnostic (configs/wca_replica_ladder/fr_dose_test.json)."""
import os, sys, time, json
from concurrent.futures import ProcessPoolExecutor, as_completed
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_numba as wn
OUT = os.path.join(ROOT, "results", "wca_replica_ladder", "fr_dose", "raw")


def job(rate, seed):
    t0 = time.perf_counter()
    cfg = dict(wn.ACCEPTED_CFG, fr_rate=float(rate))
    res = wn.run_ladder_point(seed, 16, 1_920_000, [("hist_abf", False), ("hist_fr_uniform", True)], cfg=cfg, cap_min=1)
    res["meta_json"] = json.dumps(dict(rate=rate, seed=seed, wall=time.perf_counter() - t0))
    p = os.path.join(OUT, f"r{rate}_s{seed}.npz"); tmp = p[:-4] + ".tmp.npz"
    wn.save_result(res, tmp, keep_state=False); os.replace(tmp, p)
    return rate, seed, time.perf_counter() - t0


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(r, s) for r in (0.1, 0.03, 0.01) for s in range(3300, 3316) if not os.path.exists(os.path.join(OUT, f"r{r}_s{s}.npz"))]
    wn.run_ladder_point(3300, 2, 50, [("hist_abf", False), ("hist_fr_uniform", True)], save_at=np.array([50]))
    print(len(jobs), "jobs", flush=True)
    with ProcessPoolExecutor(max_workers=48) as ex:
        for k, f in enumerate(as_completed([ex.submit(job, *j) for j in jobs])):
            r, s, w = f.result(); print(f"[{k + 1}/{len(jobs)}] rate {r} seed {s} {w / 60:.1f} min", flush=True)
    print("done", flush=True)
