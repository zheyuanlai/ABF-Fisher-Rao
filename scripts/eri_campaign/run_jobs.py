"""Run a list of ERI engine jobs (one process each, single-threaded numba), resumable and checkpointed.

    NUMBA_NUM_THREADS=1 python scripts/eri_campaign/run_jobs.py JOBS.json --procs 32

JOBS.json: {"stage": ..., "jobs": [{"cfg": {...make_cfg overrides...}, "method": "abf"|"fr", "out": "relative/path.npz"}]}
A job whose output exists is skipped; checkpoints (out + ".ckpt.npz") are written every 10 min and resumed.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
_MODEL = {}


def job(j):
    out = os.path.join(ROOT, j["out"])
    if os.path.exists(out):
        return j["out"], 0.0, "skip"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    from eri import engine as E
    from eri.model import Model
    c = E.make_cfg(**j["cfg"])
    key = (c["variant"], c["T_K"], json.dumps(c["ff"], sort_keys=True))
    if key not in _MODEL:
        _MODEL[key] = Model(c["variant"], T_K=c["T_K"], params=c["ff"])
    t0 = time.time()
    E.run(c, j["method"], out_path=out, checkpoint_path=out + ".ckpt.npz", model=_MODEL[key])
    return j["out"], time.time() - t0, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    spec = json.load(open(a.jobs))
    jobs = spec["jobs"]
    print(f"[{time.strftime('%H:%M:%S')}] stage {spec.get('stage')}: {len(jobs)} jobs on {a.procs} processes "
          f"(NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS')})", flush=True)
    t0 = time.time()
    tot = 0.0
    n = 0
    with Pool(a.procs, maxtasksperchild=8) as pool:
        for out, el, st in pool.imap_unordered(job, jobs):
            n += 1
            tot += el
            print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(jobs)} {st} {out} {el:.0f}s "
                  f"(cum {tot / 3600:.2f} core-h, wall {time.time() - t0:.0f}s)", flush=True)
    print(f"ALLDONE core-h {tot / 3600:.3f} wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
