"""Long frozen-bias validation of the Stage-3 arms.

The in-run frozen-bias pass used 60k steps (12 time units): shorter than the
~6.5-unit global left-right relaxation under a flattened landscape, so its ratios
may be statistics-limited. This rerun scores the SAME stored final biases with a
200k-step run (40 units, burn 50% -> 20 units of averaging).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from abpfr.systems import gateway as gw

ARMS = ["shus", "fr_temp", "fr_persistent", "sham", "count"]
OUT = "results/stage3_confirmatory"


def main():
    by = {}
    for f in sorted(glob.glob(f"{OUT}/*_seed*.npz")):
        b = os.path.basename(f)[:-4]
        name, sd = b.rsplit("_seed", 1)
        with np.load(f) as z:
            by.setdefault(name, {})[int(sd)] = z["pmf_t"][-1]
    seeds = sorted(by["shus"])
    F = np.stack([by[a][sd] for sd in seeds for a in ARMS])
    group = [sd for sd in seeds for a in ARMS]
    cfg = gw.GatewayConfig(beta=16.0, H=0.5, s=0.10, r=32.0, K=1024, dt=2e-4,
                           n_steps=500_000, block=20, n_saves=400)
    fb = gw.run_frozen_bias(F, [cfg] * len(F), group=group, n_steps=200_000,
                            seed=13579)
    l2 = fb["l2_f"].reshape(len(seeds), len(ARMS))
    res = {}
    print("long frozen-bias (200k steps, 40 t, burn 50%): median l2, paired ratio")
    for j, a in enumerate(ARMS):
        r = float(np.median(l2[:, j] / l2[:, 0]))
        res[a] = dict(median_l2=float(np.median(l2[:, j])), ratio_vs_shus=r,
                      l2=[float(x) for x in l2[:, j]])
        print(f"  {a:<14s} l2={np.median(l2[:, j]):.4f}  ratio={r:.3f}")
    with open(f"{OUT}/frozen_bias_long.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
