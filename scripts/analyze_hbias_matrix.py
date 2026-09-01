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
          f"{'barrier err':>12} {'refnoise share':>15} {'e_F refcorr*':>13}")
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
        # The reference enters every arm identically, so its error is
        # common-mode: E[e^2] = MSE_true + sigma_ref^2 when the two are
        # independent.  Subtracting it recovers an unbiased estimate of the
        # arm's OWN error -- but the correction grows as a share of the signal,
        # so it is reported beside the raw number, never instead of it.
        share = 100 * (ref_noise / np.median(e)) ** 2
        corr2 = np.median(e) ** 2 - ref_noise ** 2
        e_corr = float(np.sqrt(corr2)) if corr2 > 0 else float("nan")
        out[hb] = dict(eF=float(np.median(e)), sem=float(e.std(ddof=1)/np.sqrt(len(e))),
                       eF_ref_floor_corrected_diagnostic=e_corr, barrier_err_pct=float(bar),
                       ref_noise_share_pct=float(share))
        print(f"{hb:7.3f} {np.median(e):9.4f} {out[hb]['sem']:8.4f} {pair:>16} "
              f"{bar:+11.2f}% {share:14.0f}% {e_corr:12.4f}")

    # --- the failure mode a small online bandwidth is SUPPOSED to have -----
    # Smaller h_bias should make the adaptive force noisy.  Measure it directly
    # on the force the DYNAMICS actually felt (each arm smoothed at its OWN
    # h_bias) rather than assuming it.
    G2 = F_ref.size
    grid2, dphi2 = per.periodic_grid(G2, dtype=torch.float64)
    Fp_ref = np.gradient(F_ref, dphi2, edge_order=2)
    R_ref = float(np.sqrt(np.mean(np.gradient(Fp_ref, dphi2, edge_order=2) ** 2)))
    print(f"\n  online bias-force roughness vs the reference ({R_ref:.1f}); "
          f"~1 = as smooth as the truth, >>1 = dynamics pushed by estimator noise")
    for hb in sorted(arms, reverse=True):
        fs, cs = arms[hb]
        K = per.wrapped_gaussian_kernel_matrix(grid2, hb * K_PHI)
        mf = mean_force_regularized(torch.as_tensor(fs, dtype=torch.float64),
                                    torch.as_tensor(cs, dtype=torch.float64),
                                    K, 20.0).numpy()
        rough = np.sqrt(np.mean(np.gradient(mf, dphi2, axis=-1, edge_order=2) ** 2, -1))
        clip = 100 * np.mean(np.abs(mf) >= 30.0 / K_PHI)
        out[hb]["roughness_ratio"] = float(np.median(rough) / R_ref)
        out[hb]["clip_pct"] = float(clip)
        print(f"    h_bias {hb:6.3f}: roughness ratio {np.median(rough)/R_ref:5.2f}  "
              f"clipping {clip:.2f}%")

    print("\n  * e_F refcorr is a REFERENCE-NOISE-FLOOR CORRECTED DIAGNOSTIC, not the")
    print("    true error. sqrt(e^2 - sigma_ref^2) is unbiased only in expectation: for")
    print("    the ONE realized reference the cross term -2<F_hat - F, F_ref - F> does")
    print("    not vanish. The conclusion does not rest on it -- the raw column already")
    print("    carries it.")
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
