#!/usr/bin/env python3
"""EXPLORATORY (not part of the predeclared protocol): Laplacian selection with a wider KDE bandwidth.

Motivation (scripts/laplacian_baselines/score_noise_diagnostic.py, 2026-11-01): at the KDE bandwidth shared with FR
(0.1 rad) the instantaneous p''/p field of the K = 1024 LTA marginal is ~99 % finite-sample noise (std 11.8 rad^-2 vs
0.75 for the time-averaged marginal, correlation 0.04), so the Laplacian arm kills and clones at random with respect to
the marginal.  This job file runs ONE cell (LTA 150 K, K1024) on 8 FRESH seeds with ABF, FR (unchanged) and the
Laplacian arm with kde_bandwidth in {0.3, 0.5} rad x scale in {0.1, 0.3, 1, 3} (the wider kernel shrinks |p''/p|, so
the scale ladder is shifted up).  Outputs under results/laplacian_baselines/explore/; nothing here feeds the primary
analysis (analyze.py / SUMMARY.md sections 1-6).

  python scripts/laplacian_baselines/make_explore.py [R2C3|R2H80]  -> configs/laplacian_baselines/explore_<cell>_bw.json
"""
import copy
import json
import os

import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
# cell -> (pilot job file used as template, fresh seed base, output tag). 80 K added after the score-noise diagnostic
# showed it is the one LTA cell whose time-averaged marginal is NOT already flat under ABF (TV 0.20), i.e. the only
# cell where a marginal-flattening rule has something to act on.
CELLS = {"R2C3": ("pilot_R2C3.json", 1963000, "R2C3_lta150_K1024_bw"),
         "R2H80": ("pilot_R2H80.json", 1965000, "R2H80_lta80_K1024_bw")}
BW = [0.3, 0.5]
SCALES = [0.1, 0.3, 1.0, 3.0]


def main():
    cell = sys.argv[1] if len(sys.argv) > 1 else "R2C3"
    src_name, seed0, tagname = CELLS[cell]
    SRC = os.path.join(ROOT, "configs/laplacian_baselines", src_name)
    OUT = os.path.join(ROOT, f"configs/laplacian_baselines/explore_{cell}_bw.json")
    SEEDS = list(range(seed0, seed0 + 8))
    TAG = f"results/laplacian_baselines/explore/{tagname}"
    src = json.load(open(SRC))
    template = next(j for j in src["jobs"] if j["method"] == "fr")
    jobs = []
    for seed in SEEDS:
        for method in ("abf", "fr"):
            j = copy.deepcopy(template)
            j["seed"] = seed
            j["method"] = method
            j["out"] = f"{TAG}/{method}_N{j['N']}_s{seed}.npz"
            jobs.append(j)
        for bw in BW:
            for sc in SCALES:
                j = copy.deepcopy(template)
                j["seed"] = seed
                j["method"] = "laplacian"
                j["cfg"]["laplacian_scale"] = sc
                j["cfg"]["kde_bandwidth"] = bw
                j["out"] = f"{TAG}/laplacian_bw{bw:g}_sc{sc:g}_N{j['N']}_s{seed}.npz"
                jobs.append(j)
    spec = {"stage": f"LB_EXPLORE_{cell}_BW", "source_config": f"configs/laplacian_baselines/{src_name} (FR job template)",
            "seeds": SEEDS, "kde_bandwidths": BW, "scales": SCALES,
            "purpose": __doc__.strip().splitlines()[0] + " See the module docstring of scripts/laplacian_baselines/make_explore.py.",
            "jobs": jobs}
    json.dump(spec, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(jobs)} jobs")


if __name__ == "__main__":
    main()
