#!/usr/bin/env python
"""POST-HOC estimator/density consistency test vs dt and clips (configs/wca_replica_ladder/consistency_test.json)."""
import os, sys, time, json
from concurrent.futures import ProcessPoolExecutor, as_completed
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_numba as wn
OUT = os.path.join(ROOT, "results", "wca_replica_ladder", "consistency", "raw")
TPHYS = 30720.0
CONFIGS = {
    "A_dt0.002": dict(dt=0.002),
    "B_dt0.001": dict(dt=0.001),
    "C_dt0.0005": dict(dt=0.0005),
    "D_dt0.0005_noclip": dict(dt=0.0005, force_clip=1e5, min_r=0.5, mean_force_sample_clip=1e7),
}


def job(name, seed):
    t0 = time.perf_counter()
    cfg = dict(wn.ACCEPTED_CFG, abf_bias_scale=0.0, **CONFIGS[name])
    n = int(round(TPHYS / cfg["dt"]))
    res = wn.run_ladder_point(seed, 1, n, [("unbiased", False)], cfg=cfg, cap_min=1, save_at=np.array([n // 2, n]))
    res["meta_json"] = json.dumps(dict(config=name, seed=seed, wall=time.perf_counter() - t0, cfg=CONFIGS[name]))
    p = os.path.join(OUT, f"{name}_s{seed}.npz"); tmp = p[:-4] + ".tmp.npz"
    wn.save_result(res, tmp, keep_state=False); os.replace(tmp, p)
    return name, seed, time.perf_counter() - t0


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(c, s) for c in CONFIGS for s in range(3600, 3632) if not os.path.exists(os.path.join(OUT, f"{c}_s{s}.npz"))]
    for c in CONFIGS:
        wn.run_ladder_point(3600, 1, 50, [("unbiased", False)], cfg=dict(wn.ACCEPTED_CFG, abf_bias_scale=0.0, **CONFIGS[c]), save_at=np.array([50]))
    print(len(jobs), "jobs", flush=True)
    with ProcessPoolExecutor(max_workers=128) as ex:
        for k, f in enumerate(as_completed([ex.submit(job, *j) for j in jobs])):
            c, s, w = f.result(); print(f"[{k + 1}/{len(jobs)}] {c} seed {s} {w / 60:.1f} min", flush=True)
    print("done", flush=True)
