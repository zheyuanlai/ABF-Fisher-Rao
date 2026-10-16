"""Gate 1 / Gate 2 umbrella jobs (configs/eri_campaign/reference.json): one (group, window) per job, resumable.

    NUMBA_NUM_THREADS=1 python scripts/eri_campaign/run_reference.py --variant schlenker --scheme MC --groups 0-3 --procs 32
    NUMBA_NUM_THREADS=1 python scripts/eri_campaign/run_reference.py --variant schlenker --scheme EM --dt 1e-4 --groups 0-15

Outputs results/eri_campaign/reference/<variant>/<tag>/g<gg>_w<ww>.npz (tag MC or EM_1e-04 ...).  A job whose file
exists is skipped; files are written atomically (tmp + rename).
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
CFG_PATH = os.environ.get("ERI_REFERENCE_CFG", os.path.join(ROOT, "configs", "eri_campaign", "reference.json"))
CFG = json.load(open(CFG_PATH))
_MODEL = {}


def tag_of(scheme, dt):
    return "MC" if scheme == "MC" else f"{scheme}_{dt:.0e}"


def job(args):
    variant, scheme, dt, g, w, out = args
    if os.path.exists(out):
        return out, 0.0, "skip"
    from eri.model import Model
    from eri import reference as ref
    if variant not in _MODEL:
        _MODEL[variant] = Model(variant, T_K=CFG["T_K"], params=CFG.get("params"))
    m = _MODEL[variant]
    K = CFG["windows"]["n"]
    centres = np.linspace(-np.pi, np.pi, K, endpoint=False)
    t0 = time.time()
    if scheme == "MC":
        c = CFG["MC"]
        seed = c["seed_base"] + 1000 * g + w
        r = ref.run(m, "MC", centres[w], CFG["windows"]["kappa"], c["n_mol"], c["n_steps"], c["burn_in"], seed,
                    n_pre_mc=c["n_pre_mc"], sample_every=c["sample_every"], d_bead=c["d_bead"],
                    d_trans=c["d_trans"], d_rot=c["d_rot"], n_ts=CFG["n_ts"])
    else:
        c = CFG["LANGEVIN"]
        seed = c["seed_base"] + 1000 * g + w
        se = int(round(c["sample_every_tu"] / dt))
        r = ref.run(m, scheme, centres[w], CFG["windows"]["kappa"], c["n_mol"], int(round(c["run_tu"] / dt)),
                    int(round(c["burn_in_tu"] / dt)), seed, dt=dt, n_pre_mc=c["n_pre_mc"], sample_every=se,
                    d_bead=CFG["MC"]["d_bead"], d_trans=CFG["MC"]["d_trans"], d_rot=CFG["MC"]["d_rot"],
                    n_ts=CFG["n_ts"])
    el = time.time() - t0
    tmp = out + ".tmp.npz"
    np.savez(tmp, **r, centre=centres[w], kappa=CFG["windows"]["kappa"], group=g, window=w, seed=seed,
             scheme=scheme, dt=dt, variant=variant, elapsed=el)
    os.replace(tmp, out)
    return out, el, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--scheme", default="MC")
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--groups", default="0-15")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.groups.split("-"))
    tag = tag_of(a.scheme, a.dt)
    od = os.path.join(ROOT, CFG.get("results_dir", os.path.join("results", "eri_campaign", "reference")), a.variant,
                      tag)
    os.makedirs(od, exist_ok=True)
    jobs = [(a.variant, a.scheme, a.dt, g, w, os.path.join(od, f"g{g:02d}_w{w:02d}.npz"))
            for g in range(lo, hi + 1) for w in range(CFG["windows"]["n"])]
    print(f"{len(jobs)} jobs -> {od} on {a.procs} processes (NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS')})",
          flush=True)
    t0 = time.time()
    tot = 0.0
    n = 0
    with Pool(a.procs) as pool:
        for out, el, st in pool.imap_unordered(job, jobs):
            n += 1
            tot += el
            if n % 20 == 0 or n == len(jobs):
                print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(jobs)} wall {time.time() - t0:.0f}s, "
                      f"job-seconds {tot:.0f} ({tot / 3600:.2f} core-h)", flush=True)
    json.dump({"tag": tag, "variant": a.variant, "groups": [lo, hi], "core_seconds": tot,
               "wall_seconds": time.time() - t0, "procs": a.procs},
              open(os.path.join(od, f"_run_g{lo}-{hi}_{int(time.time())}.json"), "w"))


if __name__ == "__main__":
    main()
