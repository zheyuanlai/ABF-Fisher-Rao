"""Gate 0 backfill for the WCA dimer — is its solvent conditional ensemble equilibrated?

    python scripts/audit_wca_gate0.py --out results/v2_validity_audits/wca_gate0

The WCA dimer carries the project's strongest *physical* positive: Case IX, practical mFR
**-22.83 %** against ABF on 16/16 seeds and **-26.38 %** against its own matched sham. It was
classified establishment-limited **before Gate 0 existed**, so under Amendment 10 that reading
is provisional. This is the last provisional system.

**What is hidden here.** The CV is the dimer bond coordinate `z`; the hidden coordinate is the
**solvent cage** around the dimer. Deca's hidden coordinate was peptide conformation, R15's was
torsional state, the gateway's was the transverse channel `y`. If the solvent cage does not
relax at fixed `z` within the time a walker spends there, ABF's conditional mean force is biased
exactly as deca's was, and the establishment reading collapses.

**Why the existing TI reference does not already answer this.** `constrained_ti_reference_gpu`
seeds *every* replica from the same lattice (`lattice_initial_conditions`) plus a small jitter.
It therefore measures the conditional mean force under one solvent preparation and never tests
whether that preparation matters. This audit supplies genuinely different cages:

  * ``lattice``   the standard TI preparation (control -- should match the reference)
  * ``from_lo``   equilibrated at ``z_min`` (compact dimer, tight cage) then projected to z
  * ``from_hi``   equilibrated at ``z_max`` (stretched dimer, open cage) then projected to z
  * ``hot``       solvent randomised, then quenched -- a deliberately wrong cage

`from_lo` and `from_hi` are the physically meaningful pair: the solvent structure around a
compact dimer differs from that around a stretched one, and a walker driven quickly along `z`
by the ABF bias carries the wrong cage with it. That is the failure mode, constructed on purpose.

**Statistic (Amendment 9).** Not span ratio, and not the spread of any hidden-state distribution
-- non-mixing is harmless where `f_loc` does not depend on the hidden state. The Gate 0 quantity
is the **spread of `<f_loc>` across pools relative to `|F'_ref|`**, plus each pool's deviation
from the reference. Benchmarks for the same statistic: deca 0.61, R15 beta=2 0.564/0.593,
gateway 0.036 global / 0.189 in the constriction.
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

# ---- accepted Case IX cell (results/wca_sham/sham/provenance.json) ----
CELL = dict(n_dim=10, a=1.5, sigma=1.0, epsilon=1.0, h=2.0, w=2.0, beta=1.0)
POOLS = ("lattice", "from_lo", "from_hi", "hot")


def solvent_descriptor(q, params, r_cut=1.6):
    """Cage occupancy: solvent particles within ``r_cut`` of the dimer midpoint.

    The orthogonal observable whose relaxation defines ``tau_perp`` for this system.
    """
    L = params.box_length
    mid = q[:, 0, :] + 0.5 * core.minimum_image(q[:, 1, :] - q[:, 0, :], L)
    d = core.minimum_image(q[:, 2:, :] - mid[:, None, :], L)
    return (torch.linalg.norm(d, dim=-1) < r_cut).sum(dim=1).to(q.dtype)


def make_pool(kind, params, engine, n, z_target, dt, seed, equil_steps):
    """Build ``n`` replicas at ``z_target`` with a pool-specific solvent cage."""
    dev, dtp = engine.device, engine.dtype
    noise = math.sqrt(2.0 * dt / params.beta)
    q = core.lattice_initial_conditions(params, n, dev, dtp, seed=seed)
    if kind == "hot":
        g = torch.Generator(device=dev); g.manual_seed(seed + 7)
        q = q.clone()
        q[:, 2:, :] = torch.rand(q[:, 2:, :].shape, generator=g, device=dev,
                                 dtype=dtp) * params.box_length
    if kind in ("from_lo", "from_hi"):
        z_prep = -0.2 if kind == "from_lo" else 1.2
        q = core.project_dimer_to_z(q, torch.full((n,), z_prep, device=dev, dtype=dtp), params)
        for _ in range(equil_steps):                       # equilibrate the cage AT z_prep
            f = core.clip_forces(engine.force(q, compute_energy=False), params.force_clip)
            q = core.wrap_positions(q + dt * f + noise * torch.randn_like(q), params.box_length)
            q = core.project_dimer_to_z(q, torch.full((n,), z_prep, device=dev, dtype=dtp),
                                        params)
    return core.project_dimer_to_z(q, torch.full((n,), float(z_target), device=dev, dtype=dtp),
                                   params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v2_validity_audits/wca_gate0")
    ap.add_argument("--z-values", type=float, nargs="*",
                    default=[0.0, 0.25, 0.40, 0.55, 0.75, 1.0])
    ap.add_argument("--n-rep", type=int, default=256)
    ap.add_argument("--dt", type=float, default=2.0e-3)
    ap.add_argument("--prep-steps", type=int, default=20_000)
    ap.add_argument("--equil-steps", type=int, default=10_000)
    ap.add_argument("--prod-steps", type=int, default=60_000)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-pool", default=None,
                    help="run one pool and save it; lets the pools run as parallel processes")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    params = core.DimerWCAParams(**CELL)
    engine = core.WCADimerEngine(params, core.DEVICE, core.DTYPE)
    dev, dtp = engine.device, engine.dtype
    noise = math.sqrt(2.0 * args.dt / params.beta)
    zs = np.asarray(args.z_values, float)

    # v1's cached TI reference, read only
    ref = np.load(f"cache/phase/wca_ti_b1_h2_w2_n10_a1.5_g160.npz", allow_pickle=True)
    ref_mf = np.interp(zs, ref["grid"], ref["mean_force"])

    print(f"WCA Gate 0: cell {CELL}")
    print(f"  {len(zs)} z-values x {len(POOLS)} pools x {args.n_rep} replicas")
    print(f"  pools: {POOLS}\n")

    res = {}
    t0 = time.perf_counter()
    todo = [args.only_pool] if args.only_pool else list(POOLS)
    for kind in todo:
        part = os.path.join(args.out, f"pool_{kind}.npy")
        if not args.only_pool and os.path.exists(part):
            res[kind] = dict(mf=np.load(part), cage_relax=[])
            print(f"  {kind:8s} loaded from {part}", flush=True)
            continue
        mfs, cages, cage_t = [], [], []
        for zi, zv in enumerate(zs):
            q = make_pool(kind, params, engine, args.n_rep, zv, args.dt,
                          seed=1000 + 97 * zi, equil_steps=args.prep_steps)
            zt = torch.full((args.n_rep,), float(zv), device=dev, dtype=dtp)
            traj = []
            for s in range(args.equil_steps + args.prod_steps):
                f = core.clip_forces(engine.force(q, compute_energy=False), params.force_clip)
                q = core.wrap_positions(q + args.dt * f + noise * torch.randn_like(q),
                                        params.box_length)
                q = core.project_dimer_to_z(q, zt, params)
                if s < args.equil_steps and s % 200 == 0:
                    traj.append(float(solvent_descriptor(q, params).mean()))
                if s >= args.equil_steps and (s - args.equil_steps) % args.sample_every == 0:
                    fs = core.clip_forces(engine.force(q, compute_energy=False),
                                          params.force_clip)
                    mfs.append(float(core.local_mean_force(q, fs, params).mean()))
                    if len(mfs) % 50 == 0:
                        cages.append(float(solvent_descriptor(q, params).mean()))
            n_s = (args.prod_steps + args.sample_every - 1) // args.sample_every
            cage_t.append(traj)
            print(f"  {kind:8s} z={zv:5.2f}  <f_loc>={np.mean(mfs[-n_s:]):9.3f}  "
                  f"cage={float(solvent_descriptor(q, params).mean()):6.2f}  "
                  f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
        n_s = (args.prod_steps + args.sample_every - 1) // args.sample_every
        res[kind] = dict(mf=np.array([np.mean(mfs[i*n_s:(i+1)*n_s]) for i in range(len(zs))]),
                         cage_relax=cage_t)
        np.save(os.path.join(args.out, f"pool_{kind}.npy"), res[kind]["mf"])
    if args.only_pool:
        print(f"  saved pool_{args.only_pool}.npy"); return

    # ---------------- Gate 0 statistic ----------------
    M = np.stack([res[k]["mf"] for k in POOLS])            # (n_pool, n_z)
    spread = M.max(0) - M.min(0)
    rel_spread = spread / np.abs(ref_mf)
    print(f"\n{'z':>6} {'F_ref':>9} " + " ".join(f"{k:>9}" for k in POOLS)
          + f" {'spread':>8} {'/|ref|':>8}")
    for j, zv in enumerate(zs):
        print(f"{zv:6.2f} {ref_mf[j]:9.3f} " + " ".join(f"{M[i,j]:9.3f}" for i in range(len(POOLS)))
              + f" {spread[j]:8.3f} {rel_spread[j]:8.3f}")

    overall = float(spread.mean() / np.abs(ref_mf).mean())
    trans = (zs >= 0.25) & (zs <= 0.75)
    trans_rel = float(spread[trans].mean() / np.abs(ref_mf[trans]).mean()) if trans.any() else np.nan
    dev_ref = float(np.abs(M.mean(0) - ref_mf).mean() / np.abs(ref_mf).mean())
    print(f"\n  GATE 0 STATISTIC (spread of <f_loc> across solvent pools / |F'_ref|)")
    print(f"    all z                       : {overall:.3f}")
    print(f"    transition region 0.25-0.75 : {trans_rel:.3f}   <- where mFR acts")
    print(f"    pool-mean vs TI reference   : {dev_ref:.3f}")
    print(f"    benchmarks: deca 0.61 | R15 b2 0.564/0.593 | gateway 0.036 global, 0.189 gate")

    passed = bool(overall < 0.25 and trans_rel < 0.25)
    verdict = ("GATE 0 PASSES: the solvent cage equilibrates at fixed z -- pools prepared from "
               "compact, stretched, lattice and randomised cages agree. The WCA "
               "establishment-limited classification, and the -22.83 % Case IX positive resting "
               "on it, survive the backfill.") if passed else (
               "GATE 0 FAILS: <f_loc> depends on how the solvent cage was prepared. The WCA "
               "establishment-limited classification must be revisited, and the Case IX positive "
               "with it.")
    print(f"\n  VERDICT: {verdict}")

    np.savez_compressed(os.path.join(args.out, "wca_gate0.npz"),
                        z=zs, ref_mf=ref_mf, pools=np.array(POOLS), mf=M,
                        spread=spread, rel_spread=rel_spread,
                        **{f"cage_relax_{k}": np.array(res[k]["cage_relax"], dtype=object)
                           for k in POOLS})
    with open(os.path.join(args.out, "verdict.json"), "w") as fh:
        json.dump(dict(cell=CELL, z_values=zs.tolist(), pools=list(POOLS),
                       mf_per_pool={k: res[k]["mf"].tolist() for k in POOLS},
                       ref_mean_force=ref_mf.tolist(),
                       rel_spread_all=overall, rel_spread_transition=trans_rel,
                       pool_mean_vs_reference=dev_ref,
                       benchmarks=dict(deca=0.61, r15_b2=[0.564, 0.593],
                                       gateway_global=0.036, gateway_constriction=0.189),
                       gate0_pass=passed, verdict=verdict), fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
