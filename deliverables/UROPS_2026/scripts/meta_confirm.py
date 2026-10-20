#!/usr/bin/env python
"""Stage 4 of deliverables/UROPS_2026/metastability/PREREGISTRATION.md: the confirmation runs.

32 fresh seeds 910000-910031, arms abf and fr, every knob read from FROZEN.json (nothing else is accepted).  One
process per (seed, arm); raw results results/urops_2026/metastability/confirm/s<seed>_<arm>.npz under
ABFFR_DATA_ROOT.  Existing complete files are kept (never overwritten).

    python meta_confirm.py --out $ABFFR_DATA_ROOT/results/urops_2026/metastability/confirm --procs 64
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

C.setup_paths()
import meta_engine as E  # noqa: E402

SEEDS = list(range(910000, 910032))
FROZEN = os.path.join(C.PKG, "metastability", "FROZEN.json")


def frozen_cfg():
    fz = json.load(open(FROZEN))
    for k in ("timestep", "resolution", "fr_rate"):
        if k not in fz or fz[k].get("h" if k == "timestep" else "nb" if k == "resolution" else "gamma") is None:
            raise SystemExit(f"FROZEN.json has no usable '{k}' entry: confirmation refused")
    return dict(h=float(fz["timestep"]["h"]), nb=int(fz["resolution"]["nb"]), gamma=float(fz["fr_rate"]["gamma"]))


def job(j):
    cfg, seed, arm, out = j
    path = os.path.join(out, f"s{seed}_{arm}.npz")
    if os.path.exists(path):
        return path, "kept"
    res = E.run_arm(cfg, arm, seed)
    E.save_npz(path, res)
    return path, f"{res['wall_s']:.0f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args()
    cfg = frozen_cfg()
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(cfg=dict(E.DEFAULT, **cfg), seeds=SEEDS, frozen=FROZEN), open(os.path.join(a.out, "run_config.json"), "w"), indent=1)
    jobs = [(cfg, s, arm, a.out) for s in SEEDS for arm in ("abf", "fr")]
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for path, st in pool.imap_unordered(job, jobs):
            print(os.path.basename(path), st, flush=True)


if __name__ == "__main__":
    main()
