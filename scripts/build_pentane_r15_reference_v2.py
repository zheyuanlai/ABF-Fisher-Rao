#!/usr/bin/env python
"""Corrected pentane R15 (beta 2) reference, mean-force route (docs/PENTANE_R15_OT_REPAIR.md, §Reference).

The legacy reference (`ref_pentane_b2_..._g256_ns800000_g248.npz`) is F = -beta^-1 log of a
reflected KDE (h = 0.04) of the importance-sampled R histogram.  Two defects, measured on
2026-09-06 with 40M exact samples: (i) the log-of-smoothed-density derivative is damped by
~beta F' F'' h^2 where F is steep and curved (F' -33.6 vs -64.1 at R 2.05; 29 vs 42 at 3.56),
(ii) the legacy window (F - F_min <= 10 = 20 kT at beta 2) extends to R 3.70 where the
importance sampler has NO effective samples (n_eff < 100 above R 3.55).  Both routes agree to
~0.3 in F' over the interior.

This builder uses the same exact importance sampler (internal-coordinate v4 proposal, weights
exp(-beta V_nb)) but estimates F'(R) DIRECTLY as the weighted conditional mean of the ABF
estimator's own local mean force f_R = grad V . v - beta^-1 div v per fine grid bin (no
smoothing; the quantity ABF estimates, as the WCA TI reference does), with a per-bin standard
error from the effective sample count, and integrates it.  The evaluation window is the largest
contiguous run of bins with n_eff >= NEFF_MIN whose F lies within `thermal_delta` of the run's
minimum.  A fine-KDE (h = 0.01) log-density route is stored as an independent cross-check, and
the conditional torsion reference p(phi1, phi2 | R bin) is rebuilt from the same samples.

    CUDA_VISIBLE_DEVICES=1 python scripts/build_pentane_r15_reference_v2.py
-> cache/alkanes_cv/ref_pentane_b2_R15_v2_meanforce.npz
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
torch.set_default_dtype(torch.float64)
from alkanes import potentials as pot, geometry as geom, interval as iv, density2d as d2   # noqa: E402
from alkanes.reference import sample_bond_lengths, sample_bond_angles                      # noqa: E402
from alkanes.reference_cv import sample_dihedral_v4                                        # noqa: E402
from alkanes.distance_cv import DistanceCV                                                 # noqa: E402
from alkanes.ot_repair_dist import compiled_forces, eager_forces                           # noqa: E402

LEGACY = os.path.join(ROOT, "cache", "alkanes_cv", "ref_pentane_b2_s2.3_full_R04_lo1.4_hi3.7_g256_ns800000_g248.npz")
OUT = os.path.join(ROOT, "cache", "alkanes_cv", "ref_pentane_b2_R15_v2_meanforce.npz")
PI = math.pi
NEFF_MIN = 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=200_000_000)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = pot.AlkaneParams(n_atoms=5, beta=2.0, sigma=2.3, epsilon=1.0, decouple=False, force_clip=200.0)
    cv = DistanceCV(0, 4)
    force_fn = compiled_forces() if dev == "cuda" else eager_forces
    leg = np.load(LEGACY, allow_pickle=True)
    grid = np.asarray(leg["grid"]); dz = float(leg["dz"]); n_grid = grid.size; lo, hi = float(leg["R_lo"]), float(leg["R_hi"])
    nb = int(leg["cond_hist"].shape[0]); n2 = int(leg["cond_hist"].shape[1])
    g1c, g2c, dphi1c, dphi2c = d2.torus_grid(n2, n2, device=dev)
    edges = torch.linspace(lo, hi, nb + 1, device=dev)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    csum = torch.zeros(n_grid, device=dev); fsum = torch.zeros_like(csum); f2sum = torch.zeros_like(csum); w2sum = torch.zeros_like(csum)
    cond = torch.zeros(nb * n2 * n2, device=dev); cond_w = torch.zeros(nb, device=dev)
    wtot = torch.zeros((), device=dev); w2tot = torch.zeros((), device=dev)
    t0 = time.time(); done = 0
    with torch.no_grad():
        while done < a.n_samples:
            m = min(a.chunk, a.n_samples - done)
            bonds = torch.stack([sample_bond_lengths(m, p, gen, dev) for _ in range(4)], 1)
            angles = torch.stack([sample_bond_angles(m, p, gen, dev) for _ in range(3)], 1)
            dih = torch.stack([sample_dihedral_v4(m, p, gen, dev) for _ in range(2)], 1)
            q = geom.place_chain_internal(bonds, angles, dih, 5, device=dev)
            R = cv.value(q); w = torch.exp(-p.beta * pot.nonbonded_energy(q, p))
            f, _, _ = cv.local_mean_force(q, force_fn(q, p), p.beta)
            f = torch.clamp(f, -480.0, 480.0)                        # the sampler's own sample clip (8 x abf_force_clip)
            idx = iv.bin_index(R, n_grid, lo, hi)
            csum.scatter_add_(0, idx, w); fsum.scatter_add_(0, idx, w * f); f2sum.scatter_add_(0, idx, w * f * f); w2sum.scatter_add_(0, idx, w * w)
            bin_id = (torch.bucketize(R, edges) - 1).clamp(0, nb - 1)
            i1 = torch.floor((dih[:, 0] + PI) / dphi1c).long().clamp(0, n2 - 1); i2 = torch.floor((dih[:, 1] + PI) / dphi2c).long().clamp(0, n2 - 1)
            cond.scatter_add_(0, bin_id * (n2 * n2) + i1 * n2 + i2, w); cond_w.scatter_add_(0, bin_id, w)
            wtot += w.sum(); w2tot += (w * w).sum()
            done += m
    print(f"{a.n_samples} exact samples in {time.time() - t0:.0f}s on {dev}; global ESS frac {float(wtot ** 2 / w2tot) / a.n_samples:.3f}", flush=True)
    csum, fsum, f2sum, w2sum = (x.cpu().numpy() for x in (csum, fsum, f2sum, w2sum))
    neff = csum ** 2 / np.clip(w2sum, 1e-300, None)
    Fp = fsum / np.clip(csum, 1e-300, None)
    var = np.clip(f2sum / np.clip(csum, 1e-300, None) - Fp ** 2, 0, None)
    Fp_se = np.sqrt(var / np.clip(neff, 1.0, None))
    ok = neff >= NEFF_MIN
    # largest contiguous run of well-determined bins
    best, cur = (0, 0), None
    for i, o in enumerate(list(ok) + [False]):
        if o and cur is None:
            cur = i
        if not o and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    run = np.zeros(n_grid, bool); run[best[0]:best[1]] = True
    Fp_run = np.where(run, Fp, 0.0)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (Fp_run[1:] + Fp_run[:-1]) * dz)])
    Fmin = F[run].min()
    window = run & ((F - Fmin) <= 10.0)
    # keep the window contiguous
    idxs = np.nonzero(window)[0]; window = np.zeros(n_grid, bool); window[idxs.min():idxs.max() + 1] = True
    F = F - F[window].mean()
    # fine-KDE cross-check route
    gt = torch.as_tensor(grid)
    K = iv.reflected_kernel_matrix(gt, 0.01, lo, hi)
    p01 = iv.normalize_density(iv.smooth(torch.as_tensor(csum)[None, :], K), dz)[0].numpy()
    F01 = -(1.0 / p.beta) * np.log(np.clip(p01, 1e-300, None)); F01 = F01 - F01[window].mean(); Fp01 = np.gradient(F01, dz)
    # conditional reference
    ch = cond.reshape(nb, n2, n2).cpu().numpy()
    cond_dens = ch / np.clip(ch.sum(axis=(1, 2), keepdims=True) * float(dphi1c) * float(dphi2c), 1e-300, None)
    gnp = g1c.cpu().numpy(); bar = math.radians(61.6)
    masks = {"T": np.abs(gnp) < bar, "Gp": gnp >= bar, "Gm": gnp <= -bar}
    basin = np.zeros((nb, 9)); names = []
    for ai, (n1, m1) in enumerate(masks.items()):
        for bi, (nm2, m2) in enumerate(masks.items()):
            names.append(f"{n1}_{nm2}")
            basin[:, ai * 3 + bi] = ch[:, m1][:, :, m2].sum(axis=(1, 2)) / np.clip(ch.sum(axis=(1, 2)), 1e-300, None)
    cond_wn = cond_w.cpu().numpy(); cond_wn = cond_wn / cond_wn.sum()
    # legacy comparison on the new window
    Fleg = np.asarray(leg["F"]); Fleg = Fleg - Fleg[window].mean(); Fpleg = np.asarray(leg["Fprime"])
    l2_leg = math.sqrt(np.mean((Fleg - F)[window] ** 2)); l2_01 = math.sqrt(np.mean((F01 - F)[window] ** 2))
    wlo, whi = float(grid[window].min() - dz / 2), float(grid[window].max() + dz / 2)
    print(f"window: R in [{wlo:.3f}, {whi:.3f}] ({window.sum()} bins; legacy window had {(np.asarray(leg['F']) - np.asarray(leg['F']).min() <= 10).sum()}); "
          f"F range in window {F[window].max() - F[window].min():.2f}; median SE(F') {np.median(Fp_se[window]):.3f}, max {Fp_se[window].max():.3f}; min n_eff in window {neff[window].min():.0f}")
    print(f"cross-checks on the window: KDE h=0.01 route vs mean-force route L2(F) {l2_01:.3f}, RMS dF' {math.sqrt(np.mean((Fp01 - Fp)[window] ** 2)):.3f}; "
          f"LEGACY reference vs mean-force route L2(F) {l2_leg:.3f}, RMS dF' {math.sqrt(np.mean((Fpleg - Fp)[window] ** 2)):.3f} (RMS F' {math.sqrt(np.mean(Fp[window] ** 2)):.2f})")
    print("conditional samples per R bin (effective, weighted):", np.round(cond_w.cpu().numpy() ** 2 / np.clip((cond ** 2).reshape(nb, -1).sum(1).cpu().numpy(), 1e-300, None)).astype(int).tolist() if False else np.round(ch.sum(axis=(1, 2))).astype(int).tolist())
    np.savez(a.out, grid=grid, dz=dz, R_lo=lo, R_hi=hi, beta=2.0, sigma=2.3, force_clip=200.0, n_samples=a.n_samples, seed=a.seed,
             F=F, Fprime=Fp, Fprime_se=Fp_se, neff=neff, cv_hist_weighted=csum, window_mask=window, window_lo=wlo, window_hi=whi,
             F_kde01=F01, Fprime_kde01=Fp01, F_legacy=Fleg, Fprime_legacy=Fpleg,
             cond_grid1=gnp, cond_grid2=g2c.cpu().numpy(), cond_dphi=float(dphi1c), cond_edges=edges.cpu().numpy(),
             cond_hist=ch, cond_dens=cond_dens, cond_weight=cond_wn, cond_basin_probs=basin, cond_basin_names=np.array(names),
             route="mean_force_conditional_mean_per_bin", neff_min=NEFF_MIN, thermal_delta=10.0)
    json.dump(dict(window=[wlo, whi], n_window_bins=int(window.sum()), l2_legacy_vs_v2=l2_leg, l2_kde01_vs_v2=l2_01,
                   median_se_Fp=float(np.median(Fp_se[window])), n_samples=a.n_samples, wall_s=time.time() - t0),
              open(a.out.replace(".npz", ".json"), "w"), indent=1)
    print(f"wrote {os.path.relpath(a.out, ROOT)}")


if __name__ == "__main__":
    main()
