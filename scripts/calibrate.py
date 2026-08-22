"""Stage 1: per-arm hyperparameter calibration on a CALIBRATION system + seeds.

Every arm gets a screen of comparable size.  The winner per arm is frozen and
carried unchanged into the confirmation stage on FRESH seeds (docs/PREREGISTRATION.md).
This exists because the previous campaign's main methodological failure was
tuning the new arm while the baseline received a single configuration.
"""
from __future__ import annotations

import argparse, itertools, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from rcwfr.campaign import ARM_LIBRARY, run_arm, score, save_json
from rcwfr.engines import RunConfig
from rcwfr.registry import build

GRIDS = {
    "abf":      {"bias_n_min": [1.0, 10.0, 100.0, 1000.0]},
    "shus":     {"shus_gain": [1.0, 10.0, 100.0, 1000.0, 10000.0],
                 "shus_block": [100]},
    "unbiased": {},
    "ti_warm":  {},
    "ti_cold":  {},
    "reti_warm": {"n_ex": [5, 20, 100]},
    "reti_cold": {"n_ex": [5, 20, 100]},
    "w_only":   {"kappa": [0.03, 0.125, 0.5, 2.0], "n_cond": [5, 20, 100]},
    "fr_only":  {"theta": [0.1, 0.3, 0.6], "n_cond": [5, 20, 100]},
    "wfr":      {"kappa": [0.03, 0.125, 0.5, 2.0], "theta": [0.1, 0.3, 0.6],
                 "n_cond": [5, 20, 100]},
    "w_count":  {"kappa": [0.03, 0.125, 0.5, 2.0], "theta": [0.1, 0.3, 0.6],
                 "n_cond": [5, 20, 100]},
}


def combos(gridspec):
    if not gridspec:
        return [{}]
    keys = list(gridspec)
    return [dict(zip(keys, v)) for v in itertools.product(*[gridspec[k] for k in keys])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="EB")
    ap.add_argument("--arms", nargs="*", default=list(GRIDS))
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="results/calibration")
    a = ap.parse_args()

    sysm = build(a.system)
    base = RunConfig(N=a.N, n_steps=a.steps, dt=1e-3, save_every=500,
                     bw_mf=0.02, n_min=1.0, bw_kde=0.10, n_bins_count=45,
                     x0=-1.0, ess_window=40)
    rec = {}
    for arm in a.arms:
        best, rows = None, []
        for ov in combos(GRIDS[arm]):
            t0 = time.time()
            run, cfg = run_arm(sysm, arm, base, a.rows, a.seed, overrides=ov)
            sc = score(run, sysm)
            IF = float(np.median(sc["I_F"]))
            eF = float(np.median(sc["e_F_final"]))
            rows.append({"ov": ov, "I_F": IF, "e_F_final": eF,
                         "cov": float(sc["cov"][-1].mean()),
                         "wall": round(time.time() - t0, 1)})
            print(f"  {arm:10s} {str(ov):58s} I_F={IF:.5f} e_F={eF:.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if best is None or IF < best["I_F"]:
                best = rows[-1]
            del run
            torch.cuda.empty_cache()
        rec[arm] = {"best": best, "all": rows}
        print(f"{arm:10s} BEST {best['ov']}  I_F={best['I_F']:.5f}", flush=True)
    save_json(os.path.join(a.out, f"{a.system}_calib.json"), rec)
    print("\nwrote", os.path.join(a.out, f"{a.system}_calib.json"))


if __name__ == "__main__":
    main()
