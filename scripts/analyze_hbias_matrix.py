#!/usr/bin/env python
"""The (h_bias, h_read) matrix, scored at the FROZEN primary readout.

All arms share the init-pool draw and seed labels, so the per-seed differences
are paired and the comparison does not have to carry between-run nuisance
variance.  The primary readout h_read = 0.05 A was frozen in
configs/information_campaign/bandwidth_matrix_prereg.md BEFORE these runs; other
readouts are sensitivity, not selection.

    python scripts/analyze_hbias_matrix.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from alkanes import periodic as per                                    # noqa: E402
from zif8.core_zif8 import mean_force_regularized                      # noqa: E402

H_READ_PRIMARY = 0.05
K_PHI = 0.42701


def eF_per_seed(fsum, csum, h_read, F_ref, min_count=20.0):
    G = fsum.shape[-1]
    grid, dphi = per.periodic_grid(G, dtype=torch.float64)
    K = per.wrapped_gaussian_kernel_matrix(grid, h_read * K_PHI)
    mf = mean_force_regularized(torch.as_tensor(fsum, dtype=torch.float64),
                                torch.as_tensor(csum, dtype=torch.float64),
                                K, min_count)
    F = per.free_energy_from_mean_force(mf, grid, dphi).numpy()
    d = F - F_ref[None, :]; d = d - d.mean(-1, keepdims=True)
    return np.sqrt((d * d).mean(-1)), F


def main():
    ref = np.load(os.path.join(ROOT, "results/uniform_campaign/zif8/reference/"
                                     "reference_T300.npz"), allow_pickle=True)
    F_ref = np.asarray(ref["F"], float)
    acc = json.loads(str(ref["acceptance"]))["values"]
    ref_noise = acc["split_half_rms_kT"] * float(ref["kT"])
    bref = F_ref.max() - F_ref.min()

    arms = {}
    for f in sorted(glob.glob(os.path.join(
            ROOT, "results/information_campaign/zif8_raw_accumulators_T300*.npz"))):
        hb = 0.20 if "hb" not in f else float(f.split("hb")[1].split(".npz")[0])
        z = np.load(f); arms[hb] = (z["raw_fsum"], z["raw_csum"])
    if not arms:
        print("no raw accumulators found"); return 1
    base = 0.20
    print(f"reference own noise {ref_noise:.4f} kJ/mol; barrier {bref:.3f} kJ/mol")
    print(f"PRIMARY readout h_read = {H_READ_PRIMARY} A (frozen before the runs)\n")
    print(f"{'h_bias':>7} {'e_F med':>9} {'e_F sem':>8} {'vs 0.20 paired':>16} "
          f"{'barrier err':>12} {'refnoise share':>15}")
    out = {}
    e0 = None
    for hb in sorted(arms, reverse=True):
        fs, cs = arms[hb]
        e, F = eF_per_seed(fs, cs, H_READ_PRIMARY, F_ref)
        bar = 100 * (np.median(F.max(-1) - F.min(-1)) / bref - 1)
        if e0 is None and abs(hb - base) < 1e-9:
            e0 = e
        pair = ""
        if e0 is not None and not (abs(hb - base) < 1e-9):
            d = 100 * (e - e0) / e0
            pair = f"{np.median(d):+7.1f}% ({int((d<0).sum())}/{len(d)})"
        share = 100 * (ref_noise / np.median(e)) ** 2
        out[hb] = dict(eF=float(np.median(e)), sem=float(e.std(ddof=1)/np.sqrt(len(e))),
                       barrier_err_pct=float(bar), ref_noise_share_pct=float(share))
        print(f"{hb:7.3f} {np.median(e):9.4f} {out[hb]['sem']:8.4f} {pair:>16} "
              f"{bar:+11.2f}% {share:14.0f}%")

    print("\n  'refnoise share' = how much of the SQUARED error is the reference's own")
    print("  uncertainty.  Once this approaches 100% the experiment is reference-limited")
    print("  and a better estimator can no longer be resolved with this reference.")
    with open(os.path.join(ROOT, "results/information_campaign/hbias_matrix.json"),
              "w") as fh:
        json.dump(dict(h_read_primary=H_READ_PRIMARY, ref_noise=ref_noise,
                       arms={str(k): v for k, v in out.items()}), fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
