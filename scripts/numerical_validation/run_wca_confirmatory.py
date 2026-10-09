#!/usr/bin/env python
"""CONDITIONAL Phase-V confirmatory run (configs/numerical_validation/wca_confirmatory_prereg.json):
ABF vs ABF + uniform FR vs matched-turnover sham on the WCA dimer at the VALIDATED time step.

    python scripts/numerical_validation/run_wca_confirmatory.py --dt DT --N N --T T --seeds 4100-4115 [--workers 16]

Every physical-time knob of the accepted histogram confirmation (dt 0.002) is kept in physical time:
warm-up 20 t.u., estimator burn-in 20 t.u., FR start 40 t.u., FR every 0.01 t.u. (dt_eff = 0.01),
ESS window 8 t.u., saves every 5 t.u.; accepted event-cap law (cap_min 0, cap = floor(0.02 N)).
Each seed runs as TWO processes on the SAME Langevin noise stream (default_noise_seed(seed, N), lattice init of
the seed): [hist_abf] and [hist_fr_uniform, hist_fr_sham].  An arm's trajectory does not depend on the other
arms of its process (tests/test_wca_numba.py V1d), so this is the 3-arm paired design at 2/3 the wall time;
the sham must share a process with its FR partner (it replays its per-opportunity counts).
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "results", "numerical_validation", "wca_confirmatory")
ARMS = [("hist_abf", False), ("hist_fr_uniform", True), ("hist_fr_sham", "sham:hist_fr_uniform")]
PARTS = {"abf": ARMS[:1], "frsham": ARMS[1:]}


def cfg_for(dt):
    import wca_numba as wn
    s = lambda t: int(round(t / dt))
    return dict(wn.ACCEPTED_CFG, dt=dt, abf_warmup_steps=s(20.0), estimator_burn_in_steps=s(20.0), fr_start_steps=s(40.0),
                fr_every=s(0.01), ess_window_steps=s(8.0))


def job(args):
    seed, N, T, dt, part = args
    import wca_numba as wn
    t0 = time.perf_counter()
    cfg = cfg_for(dt)
    n = int(round(T / dt))
    every = int(round(5.0 / dt))
    save = np.arange(0, n + 1, every)
    res = wn.run_ladder_point(seed, N, n, PARTS[part], cfg=cfg, save_at=save, cap_min=0)
    try:
        commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    res["meta_json"] = json.dumps(dict(seed=seed, N=N, T=T, dt=dt, n_steps=n, commit=commit, wall_s=time.perf_counter() - t0,
                                       precision="float64", device="cpu", arms=[a[0] for a in PARTS[part]]))
    p = os.path.join(OUT, f"dt{dt:g}_N{N}_T{T:g}", "raw", f"s{seed}_{part}.npz")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p[:-4] + ".tmp.npz"
    wn.save_result(res, tmp, keep_state=False)
    os.replace(tmp, p)
    return seed, part, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    J = [(s, a.N, a.T, a.dt, part) for part in ("frsham", "abf") for s in range(lo, hi + 1)
         if not os.path.exists(os.path.join(OUT, f"dt{a.dt:g}_N{a.N}_T{a.T:g}", "raw", f"s{s}_{part}.npz"))]
    print(f"{len(J)} jobs", flush=True)
    import wca_numba as wn
    wn.run_ladder_point(lo, 4, 50, ARMS, cfg=dict(cfg_for(a.dt), fr_start_steps=0), save_at=np.array([50]), cap_min=1)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(job, j) for j in J]):
            s, part, w = f.result()
            print(f"seed {s} {part}: {w / 3600:.2f} h", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
