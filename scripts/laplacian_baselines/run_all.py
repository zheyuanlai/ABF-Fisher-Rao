#!/usr/bin/env python3
"""Run LRS-baseline job files (any mix of gateway / LTA / ERI entries) in ONE pool of single-threaded processes.

    NUMBA_NUM_THREADS=1 python scripts/laplacian_baselines/run_all.py configs/laplacian_baselines/rerun_*.json --procs 128

Each job dict carries "engine" in {"gateway", "lta", "eri"} (scripts/laplacian_baselines/make_jobs.py).  LTA and ERI
entries run through the UNCHANGED job functions of scripts/adaptive_fr/run_lta_jobs.py and
scripts/eri_campaign/run_jobs.py (the archived campaigns' code paths); gateway entries through
scripts/laplacian_baselines/run_gateway.gateway_job.  Jobs whose output exists are skipped, checkpoints resume, so a
re-submission continues where the previous job stopped.  Longest jobs are started first.  A ledger row per finished
job is appended to results/laplacian_baselines/runs.jsonl.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
import time
from multiprocessing import Pool

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
_MODS = {}
_FILES = {"lta": "scripts/adaptive_fr/run_lta_jobs.py", "eri": "scripts/eri_campaign/run_jobs.py",
          "gateway": "scripts/laplacian_baselines/run_gateway.py"}


def _mod(engine):
    if engine not in _MODS:
        path = os.path.join(ROOT, _FILES[engine])
        spec = importlib.util.spec_from_file_location(f"lb_{engine}", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MODS[engine] = m
    return _MODS[engine]


def job(j):
    t0 = time.time()
    try:
        eng = j["engine"]
        if eng == "gateway":
            out, el, st = _mod(eng).gateway_job(j)
        else:
            out, el, st = _mod(eng).job(j)
        return dict(out=out, wall_s=el, status=st, engine=eng, host=socket.gethostname())
    except Exception as e:  # keep the pool alive; the ledger records the failure
        return dict(out=j.get("out"), wall_s=time.time() - t0, status=f"FAILED {type(e).__name__}: {e}", engine=j.get("engine"),
                    host=socket.gethostname())


def cost(j):
    """Rough expected wall (s) for longest-first scheduling (archived per-run walls)."""
    eng = j["engine"]
    if eng == "eri":
        N = int(j["cfg"]["N"])
        return 850.0 if N >= 1024 else 430.0
    if eng == "lta":
        return 490.0
    return 260.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobfiles", nargs="+")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("NPROCS", "16")))
    ap.add_argument("--ledger", default="results/laplacian_baselines/runs.jsonl")
    a = ap.parse_args(argv)
    jobs = []
    for jf in a.jobfiles:
        spec = json.load(open(os.path.join(ROOT, jf)))
        for j in spec["jobs"]:
            if "engine" not in j:
                raise ValueError(f"{jf}: job without 'engine': {j.get('out')}")
            jobs.append(j)
    jobs.sort(key=cost, reverse=True)
    stage = ",".join(os.path.basename(f) for f in a.jobfiles)
    print(f"[{time.strftime('%H:%M:%S')}] {len(jobs)} jobs from {len(a.jobfiles)} files on {a.procs} processes "
          f"(est. {sum(map(cost, jobs)) / 3600:.1f} core-h)", flush=True)
    led = os.path.join(ROOT, a.ledger)
    os.makedirs(os.path.dirname(led), exist_ok=True)
    t0 = time.time()
    tot = 0.0
    n = 0
    nfail = 0
    with Pool(a.procs, maxtasksperchild=4) as pool, open(led, "a") as f:
        for rec in pool.imap_unordered(job, jobs):
            n += 1
            tot += rec["wall_s"]
            nfail += rec["status"].startswith("FAILED")
            rec.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), pbs_job=os.environ.get("PBS_JOBID"),
                       stage=stage)
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(jobs)} {rec['status']} {rec['out']} {rec['wall_s']:.0f}s "
                  f"(cum {tot / 3600:.2f} core-h, wall {time.time() - t0:.0f}s)", flush=True)
    print(f"ALLDONE core-h {tot / 3600:.3f} wall {time.time() - t0:.0f}s failed {nfail}", flush=True)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
