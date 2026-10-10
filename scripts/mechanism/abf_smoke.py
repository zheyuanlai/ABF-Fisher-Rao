#!/usr/bin/env python
"""V5 ABF-bias stability record from the 2-seed smoke runs (docs/mechanism/SCIENTIFIC_PLAN.md section 3, Amendment 1 A1).

    python scripts/mechanism/abf_smoke.py [--smoke-root results/mechanism/smoke/T4] [--out results/mechanism/validation/abf_smoke.json]

For every dynamics (Experiment I variants + Experiment II lambdas) and both arms at N = 2048 (smoke horizon 4 t.u.):
  finite                 every saved accumulator, the final state and the final bias are finite
  max_abs_Gamma_visited  max |Gamma_j| of the FINAL bias profile Gamma = M/(C + min_count) over bins with >= 100
                         deposits (Amendment 1 A1: the gated reading)
  max_abs_Fp_visited     max |F*'| at the centres of the same bins (the gate is Gamma <= 2 x this)
  running_max_abs_bias   the engine's running max over all bias reads (reported, not gated)
Writes the hook file read by scripts/mechanism/analyze_validation.py: {dynamics: {"2.5e-05": [records]}}.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
XMIN, XMAX, NB, MIN_VISIT = -1.8, 1.8, 180, 100
DYN = {"matched_free_energy": ("alpha1", "alpha0.5", "alpha0", "shift"),
       "conditional_relaxation": ("lam0.5", "lam0.25", "lam0.1")}


def fp_star(x, beta=16.0, H=0.5, s=0.1, oo=1.0, oi=32.0):
    e = np.exp(-x * x / (2 * s * s))
    om = oo + (oi - oo) * e
    dom = -(oi - oo) * (x / (s * s)) * e
    return 4 * H * x * (x * x - 1) + dom / (beta * om)


def record(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    cfg = json.loads(str(z["cfg_json"]))
    h = float(cfg["h"])
    M, C = z["M_all"][-1], z["C_all"][-1]
    mc = float(cfg.get("min_count", 1.0))
    G = np.where(C + mc > 0, M / (C + mc), 0.0)
    vis = C >= MIN_VISIT
    xc = XMIN + (np.arange(NB) + 0.5) * (XMAX - XMIN) / NB
    finite = bool(np.isfinite(z["M_all"]).all() and np.isfinite(z["C_all"]).all() and np.isfinite(z["X_final"]).all()
                  and np.isfinite(z["Y_final"]).all() and np.isfinite(G).all())
    seed = int(meta.get("seed", -1))
    return h, dict(seed=seed, method=os.path.basename(path).split("_")[1].split(".")[0], finite=finite,
                   max_abs_Gamma_visited=float(np.abs(G[vis]).max()) if vis.any() else float("nan"),
                   max_abs_Fp_visited=float(np.abs(fp_star(xc[vis])).max()) if vis.any() else float("nan"),
                   n_visited_bins=int(vis.sum()), running_max_abs_bias=float(z["max_abs_bias"]),
                   gate_pass=bool(finite and vis.any() and np.abs(G[vis]).max() <= 2 * np.abs(fp_star(xc[vis])).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-root", default=os.path.join(ROOT, "results", "mechanism", "smoke", "T4"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "mechanism", "validation", "abf_smoke.json"))
    a = ap.parse_args()
    out = {}
    for exp, names in DYN.items():
        for d in names:
            recs = []
            for p in sorted(glob.glob(os.path.join(a.smoke_root, exp, d, "N2048", "s*_*.npz"))):
                if p.endswith(".ckpt.npz"):
                    continue
                h, r = record(p)
                recs.append(r)
            if recs:
                out[d] = {f"{h:g}": recs}
                print(f"{d:9s} h {h:g}: " + "; ".join(
                    f"s{r['seed']} {r['method']}: finite {r['finite']} |G|max {r['max_abs_Gamma_visited']:.2f} vs 2x|F'| "
                    f"{2 * r['max_abs_Fp_visited']:.2f} (running {r['running_max_abs_bias']:.1f}) {'PASS' if r['gate_pass'] else 'FAIL'}"
                    for r in recs))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
