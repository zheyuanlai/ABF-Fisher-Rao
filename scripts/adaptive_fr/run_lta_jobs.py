"""Run a list of LTA numba-engine jobs (one single-threaded process each), resumable and checkpointed.

    NUMBA_NUM_THREADS=1 python scripts/adaptive_fr/run_lta_jobs.py configs/adaptive_fr/jobs_<block>.json --procs 32

JOBS.json: {"stage": ..., "jobs": [{"T_K":..., "N":..., "seed":..., "cfg": {overrides}, "method": "abf"|"fr",
            "save_every": steps, "out": "relative/path.npz"}]}
core-seconds = wall x 1 thread.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


def job(j):
    out = os.path.join(ROOT, j["out"])
    if os.path.exists(out):
        return j["out"], 0.0, "skip"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import numpy as np
    import lta_ladder_numba as E
    c = E.make_cfg(float(j["T_K"]), int(j["N"]), int(j["seed"]), **j["cfg"])
    n = int(E.resolve_cfg(c)["n_steps"])
    ss = np.arange(0, n + 1, int(j["save_every"]), dtype=np.int64)
    if ss[-1] != n:
        ss = np.append(ss, n)
    t0 = time.time()
    res = E.run_arm(c, j["method"], ss, checkpoint_path=out + ".ckpt.npz", checkpoint_every_s=600.0)
    E.save_result(out, res)
    if os.path.exists(out + ".ckpt.npz"):
        os.remove(out + ".ckpt.npz")
    return j["out"], time.time() - t0, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    spec = json.load(open(a.jobs))
    jobs = spec["jobs"]
    print(f"[{time.strftime('%H:%M:%S')}] stage {spec.get('stage')}: {len(jobs)} jobs on {a.procs} processes", flush=True)
    t0 = time.time()
    tot = 0.0
    n = 0
    with Pool(a.procs, maxtasksperchild=4) as pool:
        for out, el, st in pool.imap_unordered(job, jobs):
            n += 1
            tot += el
            print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(jobs)} {st} {out} {el:.0f}s (cum {tot / 3600:.2f} core-h, "
                  f"wall {time.time() - t0:.0f}s)", flush=True)
    print(f"ALLDONE core-h {tot / 3600:.3f} wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
