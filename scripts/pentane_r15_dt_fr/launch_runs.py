#!/usr/bin/env python
"""Job queue for the pentane R15 timestep campaign (frozen plans: configs/pentane_r15_dt_fr/prereg_stages_bcd.json).

    python scripts/pentane_r15_dt_fr/launch_runs.py pilot  --max-procs 9
    python scripts/pentane_r15_dt_fr/launch_runs.py calib  --dt 1.25e-4
    python scripts/pentane_r15_dt_fr/launch_runs.py confirm --dt 1.25e-4 --secondary-rate 0.05

Each job = one ``run_arm.py`` process (single thread, 4 seeds, one .npz); finished jobs are skipped (resume-safe).
Longest jobs start first.  Failures are logged and re-queued once.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(ROOT, "results", "pentane_r15_dt_fr")
RUN = os.path.join(ROOT, "scripts", "pentane_r15_dt_fr", "run_arm.py")
sys.path.insert(0, os.path.join(ROOT, "scripts", "pentane_r15_dt_fr"))


def job(arm, dt, T=40.0, rate=0.02, rng=20261101, dep=None, n_seeds=4):
    return dict(arm=arm, dt=dt, T=T, rate=rate, rng=rng, dep=dep, n_seeds=n_seeds)


def plan(a):
    if a.plan == "pilot":
        out = os.path.join(RES, "pilot", "raw"); r = 20261101
        js = [job("abf", 5e-4, 160.0, rng=r), job("abf", 5e-4, 40.0, rng=r), job("abf", 2.5e-4, 80.0, rng=r), job("abf", 1.25e-4, 40.0, rng=r),
              job("abf", 1.25e-4, 40.0, rng=r, dep=5e-4), job("abf", 2.5e-4, 40.0, rng=r, dep=5e-4)]
        js += [job("fr", d, 40.0, rate=0.02, rng=r) for d in (5e-4, 2.5e-4, 1.25e-4)]
        if a.dt is not None and a.dt not in (5e-4, 2.5e-4, 1.25e-4):
            js += [job("abf", a.dt, 40.0, rng=r), job("fr", a.dt, 40.0, rate=0.02, rng=r)]
    elif a.plan == "calib":
        out = os.path.join(RES, "calib", "raw")
        js = [job("fr", a.dt, 40.0, rate=x, rng=20261102) for x in (0.02, 0.05, 0.10)]
    elif a.plan == "confirm":
        out = os.path.join(RES, "confirm", "raw")
        rates = [0.02] + ([a.secondary_rate] if a.secondary_rate else [])
        js = []
        for dt in [a.dt] + ([5e-4] if a.dt != 5e-4 else []):
            for rng in (20261201, 20261202, 20261203, 20261204):
                js.append(job("abf", dt, 40.0, rng=rng))
                js += [job("fr", dt, 40.0, rate=x, rng=rng) for x in rates]
    elif a.plan == "explore":
        # exploratory active-FR arms (configs/pentane_r15_dt_fr/addendum_exploratory_active_fr.json)
        out = os.path.join(RES, "explore", "raw")
        js = [job("fr", 2.5e-4, 40.0, rate=x, rng=rng) for rng in (20261201, 20261202, 20261203, 20261204) for x in (0.05, 0.10)]
    else:
        raise SystemExit(a.plan)
    return out, js


def name_of(j):
    from run_arm import run_name
    ns = argparse.Namespace(arm=j["arm"], fr_rate=j["rate"], dt=j["dt"], T=j["T"], deposit_interval=j["dep"], n_seeds=j["n_seeds"], rng_seed=j["rng"])
    return run_name(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", choices=["pilot", "calib", "confirm", "explore"])
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--secondary-rate", type=float, default=None)
    ap.add_argument("--max-procs", type=int, default=12)
    a = ap.parse_args()
    out, js = plan(a)
    os.makedirs(os.path.join(out, "..", "logs"), exist_ok=True)
    js = [j for j in js if not os.path.exists(os.path.join(out, name_of(j) + ".npz"))]
    js.sort(key=lambda j: -(j["T"] / j["dt"]))
    print(f"[{time.strftime('%H:%M:%S')}] plan {a.plan}: {len(js)} jobs to run, max {a.max_procs} concurrent -> {out}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    running = []; tries = {}
    while js or running:
        while js and len(running) < a.max_procs:
            j = js.pop(0); nm = name_of(j)
            log = open(os.path.join(out, "..", "logs", nm + ".log"), "a")
            cmd = [sys.executable, "-u", RUN, "--arm", j["arm"], "--fr-rate", repr(j["rate"]), "--dt", repr(j["dt"]), "--T", repr(j["T"]),
                   "--n-seeds", str(j["n_seeds"]), "--rng-seed", str(j["rng"]), "--out", out]
            if j["dep"] is not None:
                cmd += ["--deposit-interval", repr(j["dep"])]
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT), log, j, nm))
            print(f"[{time.strftime('%H:%M:%S')}] start {nm}", flush=True)
        time.sleep(10)
        still = []
        for pr, log, j, nm in running:
            rc = pr.poll()
            if rc is None:
                still.append((pr, log, j, nm)); continue
            log.close()
            print(f"[{time.strftime('%H:%M:%S')}] end {nm} rc={rc}", flush=True)
            if rc != 0:
                tries[nm] = tries.get(nm, 0) + 1
                if tries[nm] <= 1:
                    print(f"  FAILED -> re-queued once", flush=True); js.append(j)
                else:
                    print(f"  FAILED twice -> left for diagnosis (see log)", flush=True)
        running = still
    print(f"[{time.strftime('%H:%M:%S')}] plan {a.plan} done", flush=True)


if __name__ == "__main__":
    main()
