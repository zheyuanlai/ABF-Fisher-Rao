"""Gate 0 for pentane R15: does the torsional conditional ensemble equilibrate at fixed R?

    python scripts/audit_r15_conditional_mixing.py --out results/v2_validity_audits/r15_conditional

Resolves the question the earlier restrained mean-force test could not: is R15 at beta=2
*discovery-limited* (v1's classification) or *conditional-equilibration-limited*?

**Why the previous test failed and this one does not.** That test ran two long trajectories from
different torsional starts and asked whether they agreed. They did not, which established only
that mixing is slow -- it could not measure `p(Y | R)` itself, so it could not separate a slow
conditional from a wrong reference. Here every torsional basin is initialised as its **own
pool** at each fixed R, and `p_t(Y | R, Y_0)` is tracked directly. Convergence of the pools to a
common law IS the measurement, not a hoped-for side effect.

Design, per Amendment 10 (Gate 0 leads; classification by first failing gate):

  * K fixed R values spanning the dominant region, the under-supported region v1 flagged, and
    the tail where the earlier test already showed agreement (a built-in control);
  * all 9 torsional pools (phi1, phi2) in {trans, gauche+, gauche-}^2, propagated independently;
  * R restrained tightly; the previous test used k=400 and let the low-R windows slide 0.39 nm,
    leaving R < 1.9 uncovered. Stiffness is raised and dt lowered together, because overdamped
    Euler needs `dt * 2k < 2` (the CV Hessian contributes `k |grad R|^2 = 2k`).

Outcomes, fixed in advance:

  I    pools converge quickly to a common p(Y|R) and a common <f_loc>  -> Gate 0 PASSES;
       R15 is genuinely discovery-limited and stands as v1's negative control
  II   pools retain memory of Y_0 through the run                      -> Gate 0 FAILS;
       R15 is conditional-equilibration-limited
  III  some R converge and others do not                               -> classify by the region
       responsible for the free-energy error
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alkanes import geometry as geom                                       # noqa: E402
from alkanes import potentials as pot                                      # noqa: E402
from alkanes.distance_cv import DistanceCV, dist_bias_force                # noqa: E402

GAUCHE = math.radians(116.57)
BARRIER = math.radians(61.6)          # v1's basin boundary (alkanes.reference_cv._basin_idx_np)
STATES = (0.0, GAUCHE, -GAUCHE)       # trans, gauche+, gauche-
NAMES = ("t", "g+", "g-")


def basin(phi):
    """v1's convention: |phi| < 61.6 deg = trans(0); phi >= +61.6 = g+(1); <= -61.6 = g-(2)."""
    idx = torch.zeros_like(phi, dtype=torch.long)
    idx = torch.where(phi >= BARRIER, torch.ones_like(idx), idx)
    idx = torch.where(phi <= -BARRIER, 2 * torch.ones_like(idx), idx)
    return idx


def label(q):
    """Joint torsional label in [0, 9): 3 * basin(phi1) + basin(phi2)."""
    p1 = geom.signed_dihedral(q, 0, 1, 2, 3)
    p2 = geom.signed_dihedral(q, 1, 2, 3, 4)
    return 3 * basin(p1) + basin(p2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v2_validity_audits/r15_conditional")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=2.3)
    ap.add_argument("--r-values", type=float, nargs="*",
                    default=[1.70, 2.00, 2.30, 2.60, 2.90, 3.20])
    ap.add_argument("--n-rep", type=int, default=48)
    ap.add_argument("--k-umbrella", type=float, default=4000.0)
    ap.add_argument("--dt", type=float, default=2.0e-4)     # dt*2k = 1.6 < 2 (stability)
    ap.add_argument("--pull-steps", type=int, default=50_000)
    ap.add_argument("--prod-steps", type=int, default=600_000)
    ap.add_argument("--sample-every", type=int, default=2_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    assert args.dt * 2 * args.k_umbrella < 2.0, "overdamped Euler unstable: need dt*2k < 2"
    os.makedirs(args.out, exist_ok=True)
    dev, dtype = args.device, torch.float64
    params = pot.AlkaneParams(n_atoms=5, beta=args.beta, sigma=args.sigma)
    cv = DistanceCV(0, 4)
    beta, inv_beta = args.beta, 1.0 / args.beta

    pools = list(itertools.product(range(3), repeat=2))      # 9 torsional pools
    Rs = np.asarray(args.r_values, float)
    nP, nR, nRep = len(pools), len(Rs), args.n_rep
    B = nP * nR * nRep
    print(f"R15 conditional-mixing audit: {nP} pools x {nR} R-values x {nRep} replicas "
          f"= {B} states; dt={args.dt}, k={args.k_umbrella}", flush=True)

    # layout: (pool, R, rep) flattened
    d0 = torch.tensor([[STATES[a], STATES[b]] for a, b in pools], device=dev, dtype=dtype)
    d0 = d0[:, None, None, :].expand(nP, nR, nRep, 2).reshape(B, 2).contiguous()
    gen = torch.Generator(device=dev).manual_seed(20260811)
    d0 = d0 + 0.05 * torch.randn(d0.shape, generator=gen, device=dev, dtype=dtype)
    q = geom.remove_com(geom.place_chain(d0, 5, d0=params.d0, theta0=params.theta0,
                                         device=dev, dtype=dtype))
    centers = torch.as_tensor(np.tile(np.repeat(Rs, nRep), nP), device=dev, dtype=dtype)
    noise_scale = math.sqrt(2.0 * args.dt / beta)

    def _drift_obs(q, k_eff):
        F = pot.forces(q, params)
        R, gf, div_v = cv.geometry(q)
        fr = dist_bias_force(gf, -k_eff * (R - centers))
        gg = (gf * gf).sum(dim=(-2, -1)).clamp_min(1e-12)
        f_loc = -(F * gf).sum(dim=(-2, -1)) / gg - inv_beta * div_v
        return F + fr, f_loc, R

    drift_obs = torch.compile(_drift_obs, dynamic=False)
    k_t = torch.zeros((), device=dev, dtype=dtype)

    def step(q, ks=1.0):
        k_t.fill_(args.k_umbrella * ks)
        d, f_loc, R = drift_obs(q, k_t)
        n = torch.randn(q.shape, generator=gen, device=dev, dtype=dtype)
        return geom.remove_com(q + args.dt * d + noise_scale * n), f_loc, R

    t0 = time.perf_counter()
    for s in range(args.pull_steps):
        q, _, _ = step(q, ks=min(1.0, (s + 1) / max(args.pull_steps * 0.5, 1)))
    Rpull = cv.value(q).reshape(nP, nR, nRep).mean(-1)
    print(f"  pull done ({time.perf_counter()-t0:.0f}s): worst |<R> - R_c| = "
          f"{float((Rpull - torch.as_tensor(Rs, device=dev)).abs().max()):.4f} nm", flush=True)

    # ---- production: track p_t(Y | R, Y0) and running <f_loc> per pool ----
    hist, fsum, xsum, nacc = [], torch.zeros(B, device=dev, dtype=dtype), \
        torch.zeros(B, device=dev, dtype=dtype), 0
    tsteps = []
    for s in range(args.prod_steps):
        q, f_loc, R = step(q)
        fsum += f_loc
        xsum += R
        nacc += 1
        if (s + 1) % args.sample_every == 0:
            lab = label(q).reshape(nP, nR, nRep)
            oh = torch.zeros(nP, nR, 9, device=dev, dtype=dtype)
            oh.scatter_add_(2, lab, torch.ones_like(lab, dtype=dtype))
            hist.append((oh / nRep).cpu().numpy())
            tsteps.append(s + 1)
        if (s + 1) % 200_000 == 0:
            print(f"    prod {100*(s+1)/args.prod_steps:5.1f}%  "
                  f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)

    hist = np.stack(hist)                                   # (T, nP, nR, 9)
    mf = (fsum / nacc).reshape(nP, nR, nRep).mean(-1).cpu().numpy()
    xb = (xsum / nacc).reshape(nP, nR, nRep).mean(-1).cpu().numpy()

    # reference derivative at the achieved <R>
    z = np.load(glob.glob("results/alkanes_cv_extension/r15/raw/"
                          f"screen__dist__pentane__abf__trans__b{beta:g}__*.npz")[0],
                allow_pickle=True)
    refFp = np.interp(xb.mean(0), z["grid"], z["ref_Fprime"])

    # ---- diagnosis ----
    half = hist.shape[0] // 2
    pbar = hist[half:].mean(0)                              # (nP, nR, 9) second-half law
    print(f"\n{'R':>6} {'maxTV(pool,pool)':>17} {'mf spread':>10} {'<mf>':>9} {'dF_ref':>9} "
          f"{'|<mf>-ref|/|ref|':>17}")
    rows = []
    for j, r in enumerate(Rs):
        tv = 0.0
        for a in range(nP):
            for b in range(a + 1, nP):
                tv = max(tv, 0.5 * np.abs(pbar[a, j] - pbar[b, j]).sum())
        spread = float(mf[:, j].max() - mf[:, j].min())
        mmf = float(mf[:, j].mean())
        rel = abs(mmf - refFp[j]) / max(abs(refFp[j]), 1e-9)
        rows.append(dict(R=float(r), max_pool_tv=float(tv), mf_spread=spread,
                         mf_mean=mmf, ref=float(refFp[j]), rel_err=float(rel),
                         xbar=float(xb[:, j].mean())))
        print(f"{r:6.2f} {tv:17.3f} {spread:10.2f} {mmf:9.2f} {refFp[j]:9.2f} {rel:17.3f}")

    MIX_TV = 0.15
    mixed = np.array([row["max_pool_tv"] <= MIX_TV for row in rows])
    print(f"\n  pools mixed (max pairwise TV <= {MIX_TV}) at: "
          f"{[f'{r:.2f}' for r, m in zip(Rs, mixed) if m] or 'NO R VALUE'}")
    print(f"  pools NOT mixed at: {[f'{r:.2f}' for r, m in zip(Rs, mixed) if not m] or 'none'}")

    if mixed.all():
        verdict = ("CASE I -- GATE 0 PASSES: every torsional pool converges to a common "
                   "p(Y|R) at every R. R15 is genuinely DISCOVERY-LIMITED and stands as v1's "
                   "negative control.")
    elif not mixed.any():
        verdict = ("CASE II -- GATE 0 FAILS: no R value mixes. R15 is "
                   "CONDITIONAL-EQUILIBRATION-LIMITED; the v1 'mFR cannot create undiscovered "
                   "R15 configurations' reading must be softened to 'the scalar distance "
                   "coordinate leaves slowly mixing torsional structure unresolved'.")
    else:
        verdict = ("CASE III -- MIXED: classify by the region carrying the free-energy error. "
                   "Gate 0 fails where the pools do not mix; do NOT call those regions "
                   "discovery-limited merely because rare states are also poorly visited.")
    print(f"\n  VERDICT: {verdict}")

    np.savez_compressed(os.path.join(args.out, "r15_conditional.npz"),
                        R_values=Rs, pools=np.array(pools), hist=hist,
                        sample_steps=np.array(tsteps), mf=mf, xbar=xb, ref_Fprime=refFp,
                        p_second_half=pbar)
    with open(os.path.join(args.out, "verdict.json"), "w") as fh:
        json.dump(dict(rows=rows, mix_tv_threshold=MIX_TV,
                       mixed_at=[float(r) for r, m in zip(Rs, mixed) if m],
                       not_mixed_at=[float(r) for r, m in zip(Rs, mixed) if not m],
                       verdict=verdict, dt=args.dt, k_umbrella=args.k_umbrella,
                       n_rep=nRep, prod_steps=args.prod_steps, beta=beta), fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
