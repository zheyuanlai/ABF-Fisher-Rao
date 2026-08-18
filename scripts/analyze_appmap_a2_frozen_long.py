"""Long frozen-bias validation of the Phase-A2 arms (same 200k protocol as Stage 3).

Scores the stored A2 final biases with a 200k-step equilibrated rescoring run and
reports paired ratios vs BOTH baselines (shus and shus_gbest) with bootstrap CIs.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from abpfr.metrics import paired_bootstrap_ci
from abpfr.systems import gateway as gw

ARMS = ["shus", "shus_gbest", "fr_temp", "count", "sham"]
OUT = "results/appmap_phaseA2_compare"


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
                            seed=24680)
    l2 = fb["l2_f"].reshape(len(seeds), len(ARMS))
    res = {}
    print("long frozen-bias (200k steps): paired ratios [bootstrap 95% CI]")
    for j, a in enumerate(ARMS):
        row = dict(median_l2=float(np.median(l2[:, j])),
                   l2=[float(x) for x in l2[:, j]])
        for base, bj in (("shus", 0), ("shus_gbest", 1)):
            m, lo, hi = paired_bootstrap_ci(l2[:, j] / l2[:, bj])
            row[f"ratio_vs_{base}"] = [m, lo, hi]
        res[a] = row
        rs, rg = row["ratio_vs_shus"], row["ratio_vs_shus_gbest"]
        print(f"  {a:<12s} l2={row['median_l2']:.4f}  vs shus "
              f"{rs[0]:.3f} [{rs[1]:.3f},{rs[2]:.3f}]  vs gbest "
              f"{rg[0]:.3f} [{rg[1]:.3f},{rg[2]:.3f}]")
    with open(f"{OUT}/frozen_bias_long.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
