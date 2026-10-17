"""Prereg section 5: temperature contrast at fixed K = difference (300 K - 150 K) of the per-cell median per-seed
relative change, with a 95 % CI from INDEPENDENT bootstrap resampling of seeds within each cell (10 000).

    python scripts/convergence_fr/temperature_contrast.py M300_256 M150_256
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
AN = os.path.join(ROOT, "results", "convergence_fr_campaign", "analysis")


def contrast(a, b, n=10000, seed=2026):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ma = np.median(a[rng.integers(0, len(a), (n, len(a)))], 1)
    mb = np.median(b[rng.integers(0, len(b), (n, len(b)))], 1)
    d = ma - mb
    return dict(diff=float(np.median(a) - np.median(b)), ci95=[float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                hot=float(np.median(a)), cold=float(np.median(b)))


def main():
    hot, cold = sys.argv[1], sys.argv[2]
    H = json.load(open(os.path.join(AN, f"{hot}.json")))
    C = json.load(open(os.path.join(AN, f"{cold}.json")))
    out = dict(hot=hot, cold=cold, definition="median_hot - median_cold of per-seed relative changes (FR/ABF - 1)")
    for k in ("Ibar_F_post_ramp", "Ibar_F_full", "Ibar_F_post_ramp_to_10", "eF_T", "Ibar_F_late_window"):
        out[k] = contrast(H["secondary"][k]["per_seed"], C["secondary"][k]["per_seed"])
    out["primary_log_S_geo"] = contrast(np.log(H["primary"]["per_seed"]), np.log(C["primary"]["per_seed"]))
    out["primary_log_S_geo"]["note"] = "difference of median log S_geo; exp(diff) = ratio of median speedups"
    out["verdict_post_ramp"] = ("difference PRESENT" if out["Ibar_F_post_ramp"]["ci95"][0] > 0 or out["Ibar_F_post_ramp"]["ci95"][1] < 0
                                else "no resolved difference")
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"{k:24s} hot {v['hot']:+.3f} cold {v['cold']:+.3f} diff {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    print("post-ramp verdict:", out["verdict_post_ramp"])
    json.dump(out, open(os.path.join(AN, f"contrast_{hot}_vs_{cold}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
