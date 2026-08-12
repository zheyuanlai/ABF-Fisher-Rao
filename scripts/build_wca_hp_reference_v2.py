"""HP reference v2 — nonuniform acquisition, unsmoothed, uniform evaluation grid.

    python scripts/build_wca_hp_reference_v2.py --out results/v2_validity_audits/wca_hp_v2

Supersedes the 41-point build. Three defects it fixes, all recorded in
`results/v2_validity_audits/wca_trough_discovery/`:

1. **Acquisition resolution.** The cached grid (`dz = 0.028`) cannot represent a feature with
   structure at 0.010. v2 acquires at `dz = 0.010` through `[0.18, 0.34]` and `dz = 0.035`
   elsewhere -- effort where the curvature is.

2. **No smoothing.** `smooth_profile_torch(sigma=1.0)` takes sigma in **grid cells**, so two
   references on different grids are not processed identically; on the 41-point build it turned
   a raw 0.601 into 2.166, erasing the very feature this exists to capture. With `se_prep <=
   0.073` against a trough amplitude of ~3.7 there is no statistical case for blurring at all.
   `core.smooth_profile_torch` is **left untouched** so v1 remains reproducible; a physical-width
   replacement is provided here for any future caller that genuinely needs one.

3. **Quadrature weighting.** A nonuniform grid handed to a scorer computing an unweighted
   `sum_j (F_hat_j - F_ref_j)^2` would over-weight the densely sampled trough -- the dense region
   would silently dominate the score. **The acquisition grid and the evaluation grid are
   therefore kept strictly separate**: acquisition is nonuniform, and the emitted reference is
   interpolated onto the *standard uniform 160-point* grid the Case IX scorer already assumes,
   so every arm is scored with equal weights and no scorer change is required.

Integration is variable-spacing trapezoid on the raw acquisition points; interpolation is
**PCHIP**, which is shape-preserving. An unconstrained cubic spline is refused here: it
overshoots around a trough this steep, and an overshoot in the reference is indistinguishable
from a real feature downstream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import wca_abffr_core as core                                              # noqa: E402

CELL = dict(n_dim=10, a=1.5, sigma=1.0, epsilon=1.0, h=2.0, w=2.0, beta=1.0)
PREPS = ("lattice", "from_lo", "from_hi", "hot")
CACHED = "cache/phase/wca_ti_b1_h2_w2_n10_a1.5_g160.npz"


def smooth_profile_physical(z, y, sigma_z):
    """Gaussian smoothing with a bandwidth in the **physical CV**, valid on a nonuniform grid.

    The replacement for `core.smooth_profile_torch(..., sigma=<grid cells>)`, whose bandwidth
    changes physical meaning with the grid. Weights are built from actual `z` separations and
    the local interval widths, so the operator is grid-invariant.

    Not used by this build (`sigma_z = 0` -> identity); provided so a future caller cannot
    reintroduce the grid-unit defect.
    """
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    if not sigma_z:
        return y.copy()
    w_int = np.gradient(z)
    K = np.exp(-0.5 * ((z[:, None] - z[None, :]) / sigma_z) ** 2) * w_int[None, :]
    return (K @ y) / K.sum(1)


def acquisition_grid(z_min, z_max, fine_lo, fine_hi, dz_fine, dz_coarse):
    """Nonuniform: fine through the trough, coarse where the curve is smooth."""
    a = np.arange(z_min, fine_lo, dz_coarse)
    b = np.arange(fine_lo, fine_hi + 0.5 * dz_fine, dz_fine)
    c = np.arange(fine_hi + dz_coarse, z_max + 0.5 * dz_coarse, dz_coarse)
    z = np.unique(np.concatenate([a, b, c, [z_min, z_max]]))
    return z[(z >= z_min - 1e-12) & (z <= z_max + 1e-12)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v2_validity_audits/wca_hp_v2")
    ap.add_argument("--z-min", type=float, default=-0.2)
    ap.add_argument("--z-max", type=float, default=1.2)
    ap.add_argument("--fine-lo", type=float, default=0.18)
    ap.add_argument("--fine-hi", type=float, default=0.34)
    ap.add_argument("--dz-fine", type=float, default=0.010)
    ap.add_argument("--dz-coarse", type=float, default=0.035)
    ap.add_argument("--n-rep", type=int, default=128)
    ap.add_argument("--dt", type=float, default=2.0e-3)
    ap.add_argument("--prep-steps", type=int, default=20_000)
    ap.add_argument("--equil-steps", type=int, default=20_000)
    ap.add_argument("--prod-steps", type=int, default=80_000)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--n-grid", type=int, default=160, help="uniform EVAL grid; scorer contract")
    ap.add_argument("--acquire-on-eval-grid", action="store_true",
                    help="acquire directly on the uniform eval grid: interpolation becomes the\n"
                         "identity, removing the scheme ambiguity instead of bounding it")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    params = core.DimerWCAParams(**CELL)
    engine = core.WCADimerEngine(params, core.DEVICE, core.DTYPE)
    dev, dtp = engine.device, engine.dtype
    noise = math.sqrt(2.0 * args.dt / params.beta)

    if args.acquire_on_eval_grid:
        # The eval grid (dz = 1.4/159 = 0.0088) is FINER than the trough's 0.010 structure, so
        # acquiring on it resolves the feature AND makes interpolation the identity. The
        # linear-vs-PCHIP ambiguity was 0.0418 -- 2x the Case IX effect size -- and this
        # eliminates that term rather than bounding it.
        z_acq = np.linspace(args.z_min, args.z_max, args.n_grid)
    else:
        z_acq = acquisition_grid(args.z_min, args.z_max, args.fine_lo, args.fine_hi,
                                 args.dz_fine, args.dz_coarse)
    nZ, nP, nR = z_acq.size, len(PREPS), args.n_rep
    B = nZ * nP * nR
    print(f"WCA HP reference v2")
    print(f"  acquisition: {nZ} z-values, dz={args.dz_fine} on [{args.fine_lo},{args.fine_hi}], "
          f"dz={args.dz_coarse} elsewhere")
    print(f"  {nZ} x {nP} preparations x {nR} replicas = {B} states")
    print(f"  smoothing: NONE.  evaluation grid: uniform {args.n_grid} points (scorer contract)")
    print(f"  ~{B*(args.prep_steps+args.equil_steps+args.prod_steps)/1.30e6/60:.0f} min\n",
          flush=True)

    zt_np = np.repeat(z_acq, nP * nR)
    z_target = torch.as_tensor(zt_np, device=dev, dtype=dtp)
    prep_idx = torch.as_tensor(np.tile(np.repeat(np.arange(nP), nR), nZ), device=dev)

    q = core.lattice_initial_conditions(params, B, dev, dtp, seed=20260812)
    g = torch.Generator(device=dev); g.manual_seed(31415)
    hot = prep_idx == PREPS.index("hot")
    q = q.clone()
    q[hot, 2:, :] = torch.rand(q[hot, 2:, :].shape, generator=g, device=dev,
                               dtype=dtp) * params.box_length
    z_prep = z_target.clone()
    z_prep[prep_idx == PREPS.index("from_lo")] = args.z_min
    z_prep[prep_idx == PREPS.index("from_hi")] = args.z_max

    def march(q, zt, n, tag):
        t0 = time.perf_counter()
        for s in range(n):
            f = core.clip_forces(engine.force(q, compute_energy=False), params.force_clip)
            q = core.wrap_positions(q + args.dt * f + noise * torch.randn_like(q),
                                    params.box_length)
            q = core.project_dimer_to_z(q, zt, params)
            if (s + 1) % 20_000 == 0:
                print(f"    {tag} {s+1}/{n} ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
        return q

    q = core.project_dimer_to_z(q, z_prep, params)
    q = march(q, z_prep, args.prep_steps, "prep ")
    q = core.project_dimer_to_z(q, z_target, params)
    q = march(q, z_target, args.equil_steps, "equil")

    fsum = torch.zeros(B, device=dev, dtype=dtp)
    n_acc = 0
    t0 = time.perf_counter()
    for s in range(args.prod_steps):
        f = core.clip_forces(engine.force(q, compute_energy=False), params.force_clip)
        q = core.wrap_positions(q + args.dt * f + noise * torch.randn_like(q), params.box_length)
        q = core.project_dimer_to_z(q, z_target, params)
        if s % args.sample_every == 0:
            fs = core.clip_forces(engine.force(q, compute_energy=False), params.force_clip)
            fsum += core.local_mean_force(q, fs, params)
            n_acc += 1
        if (s + 1) % 20_000 == 0:
            print(f"    prod  {s+1}/{args.prod_steps} ({(time.perf_counter()-t0)/60:.1f} min)",
                  flush=True)
    if not torch.isfinite(q).all():
        raise RuntimeError("non-finite configuration at end of production")

    per_rep = (fsum / max(n_acc, 1)).view(nZ, nP, nR)
    mf_prep = core.to_numpy(per_rep.mean(-1))
    mf_raw = mf_prep.mean(-1)
    se_replica = core.to_numpy(per_rep.reshape(nZ, -1).std(-1)) / math.sqrt(nP * nR)
    se_prep = mf_prep.std(-1) / math.sqrt(nP)

    # ---- integrate RAW on the nonuniform acquisition grid (variable-spacing trapezoid) ----
    F_acq = np.concatenate([[0.0], np.cumsum(0.5 * (mf_raw[1:] + mf_raw[:-1]) * np.diff(z_acq))])
    mid = 0.5 * (args.z_min + args.z_max)
    F_acq = F_acq - np.interp(mid, z_acq, F_acq)

    # ---- PCHIP onto the UNIFORM evaluation grid the scorer expects ----
    from scipy.interpolate import PchipInterpolator
    z_eval = np.linspace(args.z_min, args.z_max, args.n_grid)
    mf_eval = PchipInterpolator(z_acq, mf_raw)(z_eval)
    fe_eval = PchipInterpolator(z_acq, F_acq)(z_eval)
    fe_eval = fe_eval - np.interp(mid, z_eval, fe_eval)

    # cross-check: integrating the interpolated mean force must reproduce the interpolated F
    dz_e = float(z_eval[1] - z_eval[0])
    F_from_mf = np.concatenate([[0.0], np.cumsum(0.5 * (mf_eval[1:] + mf_eval[:-1]) * dz_e)])
    F_from_mf -= np.interp(mid, z_eval, F_from_mf)
    consistency = float(np.abs(F_from_mf - fe_eval).max())
    # linear interpolation as an independent scheme, per the "verify both agree" requirement
    fe_lin = np.interp(z_eval, z_acq, F_acq)
    fe_lin -= np.interp(mid, z_eval, fe_lin)
    lin_vs_pchip = float(np.abs(fe_lin - fe_eval).max())

    cz = np.load(CACHED, allow_pickle=True)
    mf_c = np.interp(z_eval, cz["grid"], cz["mean_force"])
    fe_c = np.interp(z_eval, cz["grid"], cz["free_energy"])
    l2_F = float(np.sqrt(((fe_eval - fe_c) ** 2).sum() * dz_e))

    print(f"\n  interpolation cross-checks")
    print(f"    max |integrate(PCHIP mf) - PCHIP(F)| : {consistency:.5f}")
    print(f"    max |linear(F) - PCHIP(F)|           : {lin_vs_pchip:.5f}")
    # the trough, not the domain edge: search inside the fine window
    win = (z_acq >= args.fine_lo) & (z_acq <= args.fine_hi)
    i = int(np.flatnonzero(win)[np.argmin(mf_raw[win])])
    print(f"\n  trough: min F' = {mf_raw[i]:.3f} +- {se_prep[i]:.3f} at z = {z_acq[i]:.3f}"
          f"   (cached there: {np.interp(z_acq[i], cz['grid'], cz['mean_force']):.3f})")
    print(f"  max se_prep over all z: {se_prep.max():.4f}")
    print(f"  L2(F_v2 - F_cached) on the {args.n_grid}-point eval grid: {l2_F:.4f}")

    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    hsh = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]  # noqa: E731
    meta = dict(reference_version="hp_v2", source_commit=rev, cell=CELL,
                dtype=str(core.DTYPE), preps=list(PREPS), n_rep=nR,
                z_acquisition=z_acq.tolist(), n_z_acquisition=int(nZ),
                dz_fine=args.dz_fine, dz_coarse=args.dz_coarse,
                fine_window=[args.fine_lo, args.fine_hi],
                smoothing_applied=False, sigma_z=None,
                interpolation_method="PCHIP (shape-preserving)",
                integration_method="variable-spacing trapezoid on the raw acquisition grid",
                evaluation_grid="uniform, n_grid points, equal quadrature weights",
                n_grid_eval=int(args.n_grid),
                prep_steps=args.prep_steps, equil_steps=args.equil_steps,
                prod_steps=args.prod_steps, samples_per_replica=int(n_acc),
                raw_profile_hash=hsh(mf_raw), eval_profile_hash=hsh(mf_eval),
                max_se_prep=float(se_prep.max()), max_se_replica=float(se_replica.max()),
                trough_z=float(z_acq[i]), trough_value=float(mf_raw[i]),
                pchip_vs_linear_maxdiff=lin_vs_pchip,
                integrate_interp_consistency=consistency,
                l2_F_vs_cached=l2_F)
    np.savez(os.path.join(args.out, "wca_hp_v2.npz"),
             label="HP reference v2: 4 preparations, nonuniform acquisition, unsmoothed",
             grid=z_eval, mean_force=mf_eval, free_energy=fe_eval, z_ti=z_acq,
             mf_raw=mf_raw, mf_per_prep=mf_prep, se_prep=se_prep, se_replica=se_replica,
             F_acq=F_acq, cached_mean_force=mf_c, cached_free_energy=fe_c,
             meta_json=json.dumps(meta))
    with open(os.path.join(args.out, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    os.makedirs("cache/phase_hp_v3", exist_ok=True)
    np.savez("cache/phase_hp_v3/wca_ti_b1_h2_w2_n10_a1.5_g160.npz",
             label="HP reference v2 (v2 campaign; unsmoothed, nonuniform acquisition)",
             grid=z_eval, mean_force=mf_eval, free_energy=fe_eval, z_ti=z_acq)
    print(f"\nwrote {args.out} and cache/phase_hp_v3/ (drop-in, {args.n_grid}-point uniform grid)")


if __name__ == "__main__":
    main()
