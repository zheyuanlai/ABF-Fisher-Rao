#!/usr/bin/env python
"""One-dimensional references for the incomplete-CV alanine study, with propagated uncertainty.

Free energy (as in docs/ALANINE_1D_CV.md and scripts/analyze_alanine_1d.py):
    F1(xi) = -kT log sum_hidden exp(-F2/kT)   (alanine.core1d_ala.marginal_reference_1d)
on the 97 cells of the project's 2-D MBAR reference results/alanine/reference/reference.npz, for xi = phi, xi = psi,
and the phi < 0 conditional marginal of psi (the "visited-side" reference).

Mean force: there is no independent 1-D mean-force reference (the umbrella data hold angles, not forces).  The
reference mean force is the periodic central difference of the cell free energies,
    F1'(xi_i) ~ (F1_{i+1} - F1_{i-1}) / (2 dz),
which equals the CELL-AVERAGED mean force [F1(edge_{i+1/2}) - F1(edge_{i-1/2})] / dz to O(dz^2) (edge values =
midpoints of neighbouring cell values) -- the same object the histogram ABF reports for each bin.  It is NOT exact.
Its uncertainty is estimated two ways, and both are reported:
  * statistical: the repository's own copy-level block bootstrap (scripts/bootstrap_alanine_reference.py: whole
    copies resampled within every umbrella window, MBAR re-solved, 300 frames per window, 200 replicates, seed
    20260805), propagated replicate by replicate through the marginalisation and the derivative (neighbour
    correlations and window free-energy uncertainty included);
  * discretisation: the difference between the 3-point and the 5-point periodic stencils.
Reliability rule (fixed here before any ABF mean-force error is computed): the mean-force reference of a CV is
RELIABLE if the equilibrium-weighted reference uncertainty sqrt(noise^2 + discretisation^2) is < 0.5 x the median
final ABF e_F' of that CV (the reference then contributes < 25 % of the squared error); otherwise it is FLAGGED.

    CUDA_VISIBLE_DEVICES=3 python reference_1d.py --out ../data/reference
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_ala as CA  # noqa: E402

CA.setup_paths()
import torch  # noqa: E402
from alanine import reference as ref  # noqa: E402
from alanine.core1d_ala import marginal_reference_1d  # noqa: E402
from alanine.metrics_ala import aligned_l2, build_masks  # noqa: E402


def deriv3(F, dz):
    return (np.roll(F, -1, axis=-1) - np.roll(F, 1, axis=-1)) / (2.0 * dz)


def deriv5(F, dz):
    return (-np.roll(F, -2, axis=-1) + 8 * np.roll(F, -1, axis=-1) - 8 * np.roll(F, 1, axis=-1) + np.roll(F, 2, axis=-1)) / (12.0 * dz)


def profiles(F2, kT, n):
    grid = -np.pi + (np.arange(n) + 0.5) * 2 * np.pi / n
    phi_neg = (grid < 0)[:, None] & np.ones((1, n), bool)
    return dict(phi=marginal_reference_1d(F2, kT, 0), psi=marginal_reference_1d(F2, kT, 1),
                psi_cond=marginal_reference_1d(F2, kT, 1, mask=phi_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--per-window", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    d = np.load(CA.dpath(CA.REF), allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    kT, beta, kappa, ng, ncop = meta["kT_kJ"], 1.0 / meta["kT_kJ"], meta["kappa"], meta["n_grid"], meta["copies"]
    traj = d["traj"]
    centers = torch.as_tensor(d["centers"], device=dev, dtype=dtype)
    K = centers.shape[0]
    frames = traj.shape[1]
    tw = traj.reshape(K, ncop, frames, 2)

    def solve(sel):                                  # scripts/bootstrap_alanine_reference.py::solve (F only)
        flat = sel.reshape(K, ncop * frames, 2)
        stride = max(1, (ncop * frames) // a.per_window)
        sub = flat[:, ::stride][:, :a.per_window]
        phi = torch.as_tensor(sub[..., 0].reshape(-1), device=dev, dtype=dtype)
        psi = torch.as_tensor(sub[..., 1].reshape(-1), device=dev, dtype=dtype)
        N_k = torch.full((K,), sub.shape[1], device=dev, dtype=torch.long)
        f, _, _ = ref.mbar_solve(phi, psi, centers, kappa, beta, N_k, tol=1e-8, max_iter=5000)
        logw = ref.mbar_log_weights(phi, psi, centers, kappa, beta, N_k, f)
        F, _, _ = ref.fes_from_weights(phi, psi, logw, ng, beta)
        return F.cpu().numpy()

    t0 = time.perf_counter()
    F_point = solve(tw)
    F_stored = np.asarray(d["F"], float)
    fin = np.isfinite(F_stored) & np.isfinite(F_point)
    same_support = bool(np.array_equal(np.isfinite(F_stored), np.isfinite(F_point)))
    point_dev = float(np.abs((F_point - np.nanmin(F_point[fin])) - (F_stored - np.nanmin(F_stored[fin])))[fin].max())
    rng = np.random.default_rng(20260805)
    reps = []
    for b in range(a.n_boot):
        idx = rng.integers(0, ncop, size=(K, ncop))
        sel = np.take_along_axis(tw, idx[:, :, None, None], axis=1)
        try:
            reps.append(profiles(solve(sel), kT, ng))
        except RuntimeError as e:
            print(f"replicate {b}: MBAR failed ({e}); skipped", flush=True)
        if (b + 1) % 50 == 0:
            print(f"{b + 1}/{a.n_boot} ({time.perf_counter() - t0:.0f} s)", flush=True)
    dz = 2 * np.pi / ng
    P0 = profiles(F_stored, kT, ng)                  # the reference used by every 1-D analysis (stored 2-D F)
    out = dict(source=CA.REF, kT_kJ=kT, n_grid=ng, n_boot=len(reps), per_window=a.per_window, bootstrap_seed=20260805,
               device=dev, wall_s=time.perf_counter() - t0, point_estimate_reproduces_stored=dict(
                   same_support=same_support, max_abs_dev_kJ=point_dev),
               grid_rad=(-np.pi + (np.arange(ng) + 0.5) * dz).tolist(), profiles={})
    for key in ("phi", "psi", "psi_cond"):
        F = P0[key]
        pack = build_masks(F, kT)
        w = pack["weights"]["equilibrium"]
        R = np.array([r[key] for r in reps])
        Rf = np.where(np.isfinite(R), R, np.nan)
        # free-energy reference noise floor: aligned L2 (equilibrium weight, 8 kT mask) of each replicate vs the point
        fl_F = np.array([aligned_l2(Rf[i], pack["F"], w) for i in range(len(reps))])
        G = deriv3(F, dz)
        G5 = deriv5(F, dz)
        GR = deriv3(Rf, dz)
        ok = np.isfinite(F) & np.isfinite(np.roll(F, 1)) & np.isfinite(np.roll(F, -1)) & (w > 0)
        ww = np.where(ok, w, 0.0) / np.where(ok, w, 0.0).sum()
        se_G = np.nanstd(GR, axis=0)
        noise_Fp = float(np.sqrt(np.nansum(ww * se_G ** 2)))
        disc_Fp = float(np.sqrt(np.sum(ww * np.where(ok, (G - G5) ** 2, 0.0))))
        fl_Fp_reps = np.array([math.sqrt(np.nansum(ww * (GR[i] - G) ** 2)) for i in range(len(reps))])
        out["profiles"][key] = dict(
            F=np.where(np.isfinite(F), F, None).tolist(), F_boot_se=np.nanstd(Rf, axis=0).tolist(),
            Fp_central=np.where(np.isfinite(G), G, None).tolist(), Fp_5point=np.where(np.isfinite(G5), G5, None).tolist(),
            Fp_boot_se=se_G.tolist(), mask8=pack["mask8"].tolist(), eq_weight=w.tolist(),
            eF_reference_noise_floor_median=float(np.nanmedian(fl_F)),
            eFp_reference_noise_rms=noise_Fp, eFp_reference_noise_replicate_median=float(np.median(fl_Fp_reps)),
            eFp_discretisation_rms=disc_Fp, eFp_reference_uncertainty_total=float(math.sqrt(noise_Fp ** 2 + disc_Fp ** 2)),
            n_mask8=int(pack["mask8"].sum()))
        print(f"{key}: e_F ref-noise floor {np.nanmedian(fl_F):.4f} kJ/mol | F' noise {noise_Fp:.3f}, discretisation "
              f"{disc_Fp:.3f}, total {math.sqrt(noise_Fp ** 2 + disc_Fp ** 2):.3f} kJ/mol/rad", flush=True)
    json.dump(out, open(os.path.join(a.out, "reference_1d.json"), "w"))
    print(f"point estimate reproduces the stored reference: support {same_support}, max dev {point_dev:.2e} kJ/mol; "
          f"{len(reps)} replicates in {out['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
