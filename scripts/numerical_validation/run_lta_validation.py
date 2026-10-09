#!/usr/bin/env python
"""Runs of configs/numerical_validation/lta_validation_prereg.json (CPU, numba).

    python scripts/numerical_validation/run_lta_validation.py --T 300 [--workers 64] [--only MC EM2e-4 ...]

One job = (scheme, dt, group, window): 16 or 8 molecules.  Output
results/numerical_validation/lta/T{T}/<tag>/g{group}_w{window}.npz (skipped if present).
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
KAPPA = {300.0: 300.0, 150.0: 254.0}
CENTRES = np.linspace(-np.pi, np.pi, 40, endpoint=False)
TAGS = {  # tag: (scheme, dt, molecules, steps, burn-in steps, sample_every)
    "MC": ("MC", 2e-4, 16, 200_000, 5_000, 5),
    "EM2e-4": ("EM", 2e-4, 16, 240_000, 5_000, 5),
    "EM1e-4": ("EM", 1e-4, 16, 480_000, 10_000, 10),
    "EM5e-5": ("EM", 5e-5, 8, 960_000, 20_000, 20),
    "LM2e-4": ("LM", 2e-4, 16, 240_000, 5_000, 5),
}
COST_US = dict(MC=10.0, EM=6.0, LM=6.0)


def out_path(T, tag, g, w):
    return os.path.join(ROOT, "results", "numerical_validation", "lta", f"T{int(T)}", tag, f"g{g:02d}_w{w:02d}.npz")


def run(job):
    T, tag, g, w = job
    import numba
    import lta_validation as LV
    sch, dt, nmol, n, burn, se = TAGS[tag]
    t0 = time.perf_counter()
    seed = 7000 + 1000 * list(TAGS).index(tag) + 50 * g + w      # distinct per (tag, group, window)
    o = LV.run(sch, T, nmol, n, burn, seed, dt=dt, kappa=KAPPA[T], centre=float(CENTRES[w]), sample_every=se)
    try:
        commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    meta = dict(T=T, tag=tag, scheme=sch, dt=dt, n_mol=nmol, n_steps=n, burn_in=burn, n_pre_mc=5000, sample_every=se,
                group=g, window=w, centre=float(CENTRES[w]), kappa=KAPPA[T], seed=seed, commit=commit,
                numba=numba.__version__, numpy=np.__version__, precision="float64", device="cpu",
                wall_s=time.perf_counter() - t0)
    p = out_path(T, tag, g, w)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, meta_json=json.dumps(meta), **{k: np.asarray(v) for k, v in o.items()})
    os.replace(tmp, p)
    return tag, g, w, meta["wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    tags = a.only or list(TAGS)
    J = [(a.T, t, g, w) for t in tags for g in range(16) for w in range(40) if not os.path.exists(out_path(a.T, t, g, w))]
    cost = lambda j: TAGS[j[1]][2] * (TAGS[j[1]][3] + TAGS[j[1]][4]) * COST_US[TAGS[j[1]][0]] * 1e-6
    J.sort(key=cost, reverse=True)
    print(f"{len(J)} jobs, est. {sum(map(cost, J)) / 3600:.1f} core-h", flush=True)
    import lta_validation as LV
    for sch in ("MC", "EM", "LM"):
        LV.run(sch, a.T, 1, 10, 0, 1, kappa=KAPPA[a.T], centre=0.0, n_pre_mc=10)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run, j) for j in J]
        for k, f in enumerate(as_completed(futs)):
            tag, g, w, wall = f.result()
            if (k + 1) % 40 == 0 or k + 1 == len(J):
                print(f"[{k + 1}/{len(J)}] {tag} g{g} w{w} {wall:.0f} s (elapsed {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
