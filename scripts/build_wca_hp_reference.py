"""Stage A — high-precision WCA TI reference, from MULTIPLE independent preparations.

    python scripts/build_wca_hp_reference.py --out results/v2_validity_audits/wca_hp_reference

**Why the cached reference is not simply "more samples away" from being right.**
`wca_abffr_core.constrained_ti_reference_gpu` seeds every replica from the same lattice
(`lattice_initial_conditions`) plus a small jitter. More samples of one preparation converge
accurately to whatever that preparation gives. The Gate 0 audit
(`results/v2_validity_audits/wca_gate0/`) found four independently prepared solvent cages
agreeing with **each other** to 0.179 at `z = 0.25` while sitting **1.163 (56 %)** from the
cached value — agreement that tight among independent preparations cannot be explained by their
own sampling error, so the defect is in the reference, not in the conditional ensemble.

This build therefore changes the *preparation*, not just the sample count. Every `z` carries four
preparations:

  ``lattice``  the standard TI preparation (control)
  ``from_lo``  equilibrated at ``z = -0.2`` (compact dimer, tight cage), then projected
  ``from_hi``  equilibrated at ``z = +1.2`` (stretched dimer, open cage), then projected
  ``hot``      solvent randomised

**The falsifiable target.** The Gate 0 audit predicts `F'(0.25) ~ 0.93`, not the cached 2.09.
This build either reproduces that or it does not.

**Uncertainty is reported two ways**, because they answer different questions:
  * `se_replica`  -- spread across the 512 replica time-averages / sqrt(512): sampling error;
  * `se_prep`     -- spread across the 4 preparation means: **preparation bias**, which is the
                     quantity the cached reference had no way to see.

v1 is untouched: the cached reference is read for comparison and written nowhere.
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import wca_abffr_core as core                                              # noqa: E402

CELL = dict(n_dim=10, a=1.5, sigma=1.0, epsilon=1.0, h=2.0, w=2.0, beta=1.0)
PREPS = ("lattice", "from_lo", "from_hi", "hot")
CACHED = "cache/phase/wca_ti_b1_h2_w2_n10_a1.5_g160.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v2_validity_audits/wca_hp_reference")
    ap.add_argument("--n-z", type=int, default=71)
    ap.add_argument("--z-min", type=float, default=-0.2)
    ap.add_argument("--z-max", type=float, default=1.2)
    ap.add_argument("--n-rep", type=int, default=128, help="replicas per (z, preparation)")
    ap.add_argument("--dt", type=float, default=2.0e-3)
    ap.add_argument("--prep-steps", type=int, default=20_000)
    ap.add_argument("--equil-steps", type=int, default=20_000)
    ap.add_argument("--prod-steps", type=int, default=120_000)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--n-grid", type=int, default=160)
    ap.add_argument("--smooth-sigma", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    params = core.DimerWCAParams(**CELL)
    engine = core.WCADimerEngine(params, core.DEVICE, core.DTYPE)
    dev, dtp = engine.device, engine.dtype
    noise = math.sqrt(2.0 * args.dt / params.beta)

    nZ, nP, nR = args.n_z, len(PREPS), args.n_rep
    B = nZ * nP * nR
    z_all = torch.linspace(args.z_min, args.z_max, nZ, device=dev, dtype=dtp)
    # layout (z, prep, rep)
    z_target = z_all[:, None, None].expand(nZ, nP, nR).reshape(B).contiguous()
    prep_idx = torch.arange(nP, device=dev)[None, :, None].expand(nZ, nP, nR).reshape(B)

    print(f"WCA high-precision TI reference")
    print(f"  cell {CELL}")
    print(f"  {nZ} z-values x {nP} preparations x {nR} replicas = {B} states")
    print(f"  {args.prep_steps} prep + {args.equil_steps} equil + {args.prod_steps} prod")
    print(f"  ~{B*(args.prep_steps+args.equil_steps+args.prod_steps)/1.30e6/60:.0f} min at "
          f"1.30 M state-steps/s\n", flush=True)

    q = core.lattice_initial_conditions(params, B, dev, dtp, seed=987654)
    g = torch.Generator(device=dev); g.manual_seed(4242)
    hot = prep_idx == PREPS.index("hot")
    if bool(hot.any()):
        q = q.clone()
        q[hot, 2:, :] = torch.rand(q[hot, 2:, :].shape, generator=g, device=dev,
                                   dtype=dtp) * params.box_length

    # every state equilibrates at its own preparation z, then is projected to its target
    z_prep = z_target.clone()
    z_prep[prep_idx == PREPS.index("from_lo")] = args.z_min
    z_prep[prep_idx == PREPS.index("from_hi")] = args.z_max

    def march(q, zt, n_steps, tag):
        t0 = time.perf_counter()
        for s in range(n_steps):
            f = core.clip_forces(engine.force(q, compute_energy=False), params.force_clip)
            q = core.wrap_positions(q + args.dt * f + noise * torch.randn_like(q),
                                    params.box_length)
            q = core.project_dimer_to_z(q, zt, params)
            if (s + 1) % 20_000 == 0:
                print(f"    {tag} {s+1}/{n_steps}  ({(time.perf_counter()-t0)/60:.1f} min)",
                      flush=True)
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
        q = core.wrap_positions(q + args.dt * f + noise * torch.randn_like(q),
                                params.box_length)
        q = core.project_dimer_to_z(q, z_target, params)
        if s % args.sample_every == 0:
            fs = core.clip_forces(engine.force(q, compute_energy=False), params.force_clip)
            fsum += core.local_mean_force(q, fs, params)
            n_acc += 1
        if (s + 1) % 20_000 == 0:
            print(f"    prod  {s+1}/{args.prod_steps}  ({(time.perf_counter()-t0)/60:.1f} min)",
                  flush=True)
    if not torch.isfinite(q).all():
        raise RuntimeError("non-finite configuration at the end of production")

    per_rep = (fsum / max(n_acc, 1)).view(nZ, nP, nR)
    mf_prep = per_rep.mean(-1)                                   # (nZ, nP)
    mf = mf_prep.mean(-1)                                        # (nZ,)
    se_replica = per_rep.reshape(nZ, -1).std(-1) / math.sqrt(nP * nR)
    se_prep = mf_prep.std(-1) / math.sqrt(nP)

    # same convention as the cached reference: smooth, integrate, zero at midpoint
    mf_s = core.smooth_profile_torch(mf, args.smooth_sigma)
    fe = core.normalize_profile_zero_at_midpoint_torch(
        core.cumulative_trapezoid_torch(mf_s, z_all), z_all)
    eval_grid = torch.linspace(-0.2, 1.2, args.n_grid, device=dev, dtype=dtp)
    mf_eval = core.interp_uniform_grid(mf_s, z_all, eval_grid, outside_value=0.0)
    fe_eval = core.normalize_profile_zero_at_midpoint_torch(
        core.interp_uniform_grid(fe, z_all, eval_grid,
                                 outside_value=float(fe[0].item())), eval_grid)

    # ---- comparison with the cached reference (read only) ----
    cz = np.load(CACHED, allow_pickle=True)
    mf_c = np.interp(core.to_numpy(eval_grid), cz["grid"], cz["mean_force"])
    fe_c = np.interp(core.to_numpy(eval_grid), cz["grid"], cz["free_energy"])
    gnp, mnp, fnp = core.to_numpy(eval_grid), core.to_numpy(mf_eval), core.to_numpy(fe_eval)
    dmf, dfe = mnp - mf_c, fnp - fe_c
    dz_e = float(gnp[1] - gnp[0])
    l2_F = float(np.sqrt((dfe ** 2).sum() * dz_e))

    znp = core.to_numpy(z_all)
    print(f"\n{'z':>7} {'F_new':>9} {'F_cached':>9} {'delta':>8} {'se_rep':>8} {'se_prep':>8}")
    for i in range(0, nZ, max(nZ // 24, 1)):
        zc = float(znp[i])
        c = float(np.interp(zc, cz["grid"], cz["mean_force"]))
        flag = "  <-- Gate 0 probe" if abs(zc - 0.25) < 0.011 else ""
        print(f"{zc:7.3f} {float(mf[i]):9.3f} {c:9.3f} {float(mf[i])-c:8.3f} "
              f"{float(se_replica[i]):8.4f} {float(se_prep[i]):8.4f}{flag}")

    i25 = int(np.argmin(np.abs(znp - 0.25)))
    print(f"\n  FALSIFIABLE TARGET at z = {float(znp[i25]):.3f}")
    print(f"    Gate 0 pools predicted : 0.931")
    print(f"    cached reference       : {float(np.interp(0.25, cz['grid'], cz['mean_force'])):.3f}")
    print(f"    high-precision build   : {float(mf[i25]):.3f} +- {float(se_prep[i25]):.3f} (prep)")

    tr = (znp >= 0.2) & (znp <= 0.8)
    print(f"\n  transition region 0.2-0.8:")
    print(f"    mean |F'_new - F'_cached| : {np.abs(core.to_numpy(mf)[tr] - np.interp(znp[tr], cz['grid'], cz['mean_force'])).mean():.4f}")
    print(f"    max  se_prep              : {float(se_prep[torch.as_tensor(tr, device=dev)].max()):.4f}")
    print(f"\n  L2(F_new - F_cached) on the 160-point eval grid : {l2_F:.4f}")

    np.savez(os.path.join(args.out, "wca_hp_reference.npz"),
             label="high-precision constrained TI, 4 independent preparations",
             grid=gnp, mean_force=mnp, free_energy=fnp, z_ti=znp,
             mf_raw=core.to_numpy(mf), mf_per_prep=core.to_numpy(mf_prep),
             se_replica=core.to_numpy(se_replica), se_prep=core.to_numpy(se_prep),
             preps=np.array(PREPS), cached_mean_force=mf_c, cached_free_energy=fe_c,
             delta_mean_force=dmf, delta_free_energy=dfe)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(dict(cell=CELL, n_z=nZ, preps=list(PREPS), n_rep=nR, batch=B,
                       prep_steps=args.prep_steps, equil_steps=args.equil_steps,
                       prod_steps=args.prod_steps, n_samples_per_replica=n_acc,
                       Fp_at_0p25_new=float(mf[i25]),
                       Fp_at_0p25_cached=float(np.interp(0.25, cz["grid"], cz["mean_force"])),
                       Fp_at_0p25_gate0_pools=0.931,
                       se_prep_at_0p25=float(se_prep[i25]),
                       l2_F_new_minus_cached=l2_F,
                       max_abs_delta_mean_force=float(np.abs(dmf).max()),
                       cached_path=CACHED), fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
