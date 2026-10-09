#!/usr/bin/env python
"""Hamiltonian (gamma = 0) part of the BAOAB check: velocity-Verlet energy conservation from exact-Gibbs
(MC) configurations with Maxwell momenta, dt 0.005 / 0.0025 / 0.00125, 16 starts x 100 t.u. each.
Shadow-Hamiltonian theory: RMS fluctuation of H ~ dt^2, no secular drift.  -> results/numerical_validation/wca/nve_check.json"""
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_validation as V

def main():
    ph = V.phys_array(V.DEFAULT_PHYS)
    starts = [np.load(os.path.join(ROOT, "results", "numerical_validation", "wca", "mc", f"mc_s{s}.npz"))["q_final"] for s in range(9100, 9116)]
    out = {}
    for dt in (0.005, 0.0025, 0.00125):
        n = int(round(100.0 / dt)); every = max(1, int(round(0.05 / dt)))
        fl, dr = [], []
        for i, q in enumerate(starts):
            p = np.random.default_rng(100 + i).standard_normal(q.shape)
            H = V.nve_run(np.ascontiguousarray(q), p, ph, dt, n, every)
            t = np.arange(len(H)) * every * dt
            fl.append(float(np.std(H - np.polyval(np.polyfit(t, H, 1), t))))
            dr.append(float(np.polyfit(t, H, 1)[0]))
        out[f"{dt:g}"] = dict(rms_fluct_H=float(np.mean(fl)), rms_fluct_H_per_dof=float(np.mean(fl)) / 200, drift_per_tu_mean=float(np.mean(dr)),
                              drift_per_tu_sd=float(np.std(dr) / np.sqrt(len(dr))), n_starts=len(starts), T=100.0)
        print(dt, json.dumps(out[f"{dt:g}"]), flush=True)
    ks = sorted(out, key=float)
    out["fluct_ratio_per_halving"] = [out[ks[i + 1]]["rms_fluct_H"] / out[ks[i]]["rms_fluct_H"] for i in range(len(ks) - 1)]
    json.dump(out, open(os.path.join(ROOT, "results", "numerical_validation", "wca", "nve_check.json"), "w"), indent=1)
    print(out["fluct_ratio_per_halving"])

if __name__ == "__main__":
    main()
