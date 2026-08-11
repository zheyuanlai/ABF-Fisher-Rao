"""Controlled restrained-sampling test for pentane R15 — the instrument that settled deca.

    python scripts/audit_r15_mean_force.py --out results/v2_validity_audits/r15_mean_force

The corrected R15 audit left one question open: the beta=2 cells show relative mean-force error
**0.564 / 0.593** against the reference, essentially deca-alanine's 0.61. Amendment 8 forbids
forcing that into the three-way box, so the question is whether R15 beta=2 is
*discovery-limited* (v1's classification) or *conditional-equilibration-limited*.

A screen statistic cannot answer it. This runs the controlled experiment instead: accumulate the
**same** ``f_loc = grad V . v - beta^-1 div v`` estimator inside umbrella-restrained windows,
where the CV is pinned and only the orthogonal coordinates have to relax.

**The confound this design handles.** The R15 reference is importance-sampling based
(:mod:`alkanes.reference_cv`), a *different object* from deca's umbrella+MBAR reference, so a
disagreement could be reference error rather than sampling error. Two independent builds are
therefore run from **deliberately different torsional states** — all-trans and all-gauche+ —
which separates three outcomes that a single build cannot:

===============================  ==============================================================
builds agree with each other
AND with the reference           restrained sampling equilibrates and the reference is sound
                                 -> ABF's own conditional sampling is implicated;
                                    R15 beta=2 is conditional-equilibration-limited
builds agree with each other
but NOT with the reference       sampling is fine; the REFERENCE is implicated
builds disagree with each other  even restrained sampling does not equilibrate at this budget;
                                 the hidden torsional coordinate is slower than the test, and
                                 the comparison is inconclusive
===============================  ==============================================================

Physical model is v1's exactly: overdamped Brownian dynamics with unit mobility,
``dt = 5e-4``, ``noise = sqrt(2 dt / beta)``, pentane at ``beta = 2``, ``sigma = 2.3``, CV
``R15 = |q4 - q0|``. Nothing is retuned.
"""
from __future__ import annotations

import argparse
import glob
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


def build(n_win, n_rep, dihedrals, params, device, dtype, gen):
    """Place ``n_win * n_rep`` chains at a fixed torsional state, with a small jitter."""
    B = n_win * n_rep
    d = torch.as_tensor(dihedrals, device=device, dtype=dtype).reshape(1, -1).expand(B, -1)
    d = d + 0.05 * torch.randn(d.shape, generator=gen, device=device, dtype=dtype)
    q = geom.place_chain(d, params.n_atoms, d0=params.d0, theta0=params.theta0,
                         device=device, dtype=dtype)
    return geom.remove_com(q + 1e-3 * torch.randn(q.shape, generator=gen,
                                                  device=device, dtype=dtype))


def run_build(tag, dihedrals, centers_np, args, params, cv, device, dtype, seed):
    n_win, n_rep = len(centers_np), args.n_rep
    B = n_win * n_rep
    beta = params.beta
    gen = torch.Generator(device=device).manual_seed(seed)
    q = build(n_win, n_rep, dihedrals, params, device, dtype, gen)
    centers = torch.as_tensor(np.repeat(centers_np, n_rep), device=device, dtype=dtype)
    noise_scale = math.sqrt(2.0 * args.dt / beta)

    # Pentane is 5 atoms, so the step is entirely launch-bound: measured 13.75 ms/step eager,
    # the same cost v1's sampler paid. `pot.forces` goes through autograd, which is exactly the
    # pattern torch.compile fused 6.6x on deca. Physics is unchanged; a mismatch is asserted
    # below rather than assumed.
    # Returns the drift AND the observables, from ONE force evaluation. The production loop
    # previously called `pot.forces` a second time, uncompiled, to accumulate f_loc -- that
    # single uncompiled autograd call was ~9.6 ms of an ~11 ms step, i.e. most of the runtime,
    # and it is pure duplication. f_loc is now evaluated at the current q BEFORE the move
    # instead of at the new q after it; same estimator, same trajectory average.
    inv_beta = 1.0 / beta

    def _drift_obs(q, k_eff):
        F = pot.forces(q, params)
        R, gf, div_v = cv.geometry(q)
        fr = dist_bias_force(gf, -k_eff * (R - centers))
        gg = (gf * gf).sum(dim=(-2, -1)).clamp_min(1e-12)
        f_loc = -(F * gf).sum(dim=(-2, -1)) / gg - inv_beta * div_v
        return F + fr, f_loc, R

    drift_obs = torch.compile(_drift_obs, dynamic=False) if args.compile else _drift_obs

    # `k_eff` must be a TENSOR, not a Python float. Dynamo specialises on float values, so the
    # ramped pull triggered a recompile on every step, blew the cache limit, and fell back to
    # eager -- the 10.7x win silently evaporating into a warning storm.
    k_t = torch.zeros((), device=device, dtype=dtype)

    def step(q, k_scale=1.0):
        k_t.fill_(args.k_umbrella * k_scale)
        d, _, _ = drift_obs(q, k_t)
        noise = torch.randn(q.shape, generator=gen, device=device, dtype=dtype)
        return geom.remove_com(q + args.dt * d + noise_scale * noise)

    t0 = time.perf_counter()
    for s in range(args.pull_steps):
        q = step(q, k_scale=min(1.0, (s + 1) / max(args.pull_steps * 0.5, 1)))
    for _ in range(args.equil_steps):
        q = step(q)
    if not torch.isfinite(q).all():
        raise RuntimeError(f"{tag}: non-finite state after equilibration")

    fsum = torch.zeros(B, device=device, dtype=dtype)
    xsum = torch.zeros(B, device=device, dtype=dtype)
    d1sum = torch.zeros(B, device=device, dtype=dtype)
    n = 0
    k_t.fill_(args.k_umbrella)
    for s in range(args.prod_steps):
        d, f_loc, R = drift_obs(q, k_t)
        fsum += f_loc
        xsum += R
        d1sum += torch.cos(geom.signed_dihedral(q, 0, 1, 2, 3))
        n += 1
        noise = torch.randn(q.shape, generator=gen, device=device, dtype=dtype)
        q = geom.remove_com(q + args.dt * d + noise_scale * noise)
        if args.verbose and (s + 1) % 100_000 == 0:
            print(f"    {tag} prod {100*(s+1)/args.prod_steps:5.1f}%  "
                  f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)

    mf = (fsum / n).reshape(n_win, n_rep)
    return dict(mf=mf.mean(-1).cpu().numpy(),
                mf_sem=(mf.std(-1) / math.sqrt(n_rep)).cpu().numpy(),
                xbar=(xsum / n).reshape(n_win, n_rep).mean(-1).cpu().numpy(),
                cos_phi1=(d1sum / n).reshape(n_win, n_rep).mean(-1).cpu().numpy(),
                seconds=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v2_validity_audits/r15_mean_force")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=2.3)
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--n-rep", type=int, default=64)
    ap.add_argument("--k-umbrella", type=float, default=400.0)
    ap.add_argument("--dt", type=float, default=5.0e-4)
    ap.add_argument("--pull-steps", type=int, default=40_000)
    ap.add_argument("--equil-steps", type=int, default=200_000)
    ap.add_argument("--prod-steps", type=int, default=400_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--only-build", default=None,
                    help="run one build and save it; lets the two run as parallel processes")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype = torch.float64
    params = pot.AlkaneParams(n_atoms=5, beta=args.beta, sigma=args.sigma)
    cv = DistanceCV(0, 4)
    centers_np = np.linspace(1.55, 3.55, args.n_windows)      # inside the v1 walls [1.45, 3.65]

    # v1's own reference, read from the frozen screen artifact (not recomputed)
    z = np.load(glob.glob("results/alkanes_cv_extension/r15/raw/"
                          f"screen__dist__pentane__abf__trans__b{args.beta:g}__*.npz")[0],
                allow_pickle=True)
    grid, refFp, refF = z["grid"], z["ref_Fprime"], z["ref_F"]
    thermal = float(z["thermal_delta"])

    print(f"pentane R15 restrained mean-force test: {args.n_windows} windows x {args.n_rep} "
          f"replicas = {args.n_windows*args.n_rep} states, beta={args.beta}", flush=True)

    SPECS = {"all-trans": ([0.0, 0.0], 20260811),
             "all-gauche+": ([GAUCHE, GAUCHE], 20260812)}
    if args.only_build:
        dih, sd = SPECS[args.only_build]
        print(f"  build {args.only_build} (isolated process)", flush=True)
        r = run_build(args.only_build, dih, centers_np, args, params, cv, args.device, dtype, sd)
        np.savez_compressed(os.path.join(args.out, f"build_{args.only_build}.npz"), **r)
        print(f"    done in {r['seconds']/60:.1f} min -> build_{args.only_build}.npz", flush=True)
        return
    builds = {}
    for tag, (dih, sd) in SPECS.items():
        part = os.path.join(args.out, f"build_{tag}.npz")
        if os.path.exists(part):
            z = np.load(part)
            builds[tag] = {k: z[k] for k in z.files}
            print(f"  build {tag}: loaded from {part}", flush=True)
            continue
        print(f"  build {tag}: dihedrals {np.round(np.degrees(dih),1).tolist()} deg", flush=True)
        builds[tag] = run_build(tag, dih, centers_np, args, params, cv, args.device, dtype, sd)
        print(f"    done in {builds[tag]['seconds']/60:.1f} min", flush=True)

    A, Bb = builds["all-trans"], builds["all-gauche+"]
    refA = np.interp(A["xbar"], grid, refFp)
    refB = np.interp(Bb["xbar"], grid, refFp)
    m = np.interp(A["xbar"], grid, refF - refF.min()) <= thermal   # v1's thermal mask

    print(f"\n{'R_c':>7} {'<xi>':>7} {'trans':>9} {'gauche':>9} {'ref':>9} "
          f"{'|A-B|':>8} {'|A-ref|':>8} {'cosF1 A':>8} {'cosF1 B':>8}")
    for w in range(len(centers_np)):
        flag = "" if m[w] else "  (out of thermal mask)"
        print(f"{centers_np[w]:7.3f} {A['xbar'][w]:7.3f} {A['mf'][w]:9.2f} {Bb['mf'][w]:9.2f} "
              f"{refA[w]:9.2f} {abs(A['mf'][w]-Bb['mf'][w]):8.2f} "
              f"{abs(A['mf'][w]-refA[w]):8.2f} {A['cos_phi1'][w]:8.3f} "
              f"{Bb['cos_phi1'][w]:8.3f}{flag}")

    def rel(x, y):
        return float(np.abs(x[m] - y[m]).mean() / np.abs(y[m]).mean())

    ab = float(np.abs(A["mf"][m] - Bb["mf"][m]).mean()
               / np.abs(0.5 * (A["mf"][m] + Bb["mf"][m])).mean())
    a_ref, b_ref = rel(A["mf"], refA), rel(Bb["mf"], refB)
    mean_mf = 0.5 * (A["mf"] + Bb["mf"])
    pooled_ref = rel(mean_mf, refA)

    print(f"\n  build-vs-build relative difference : {ab:.3f}")
    print(f"  all-trans   vs reference           : {a_ref:.3f}")
    print(f"  all-gauche+ vs reference           : {b_ref:.3f}")
    print(f"  pooled      vs reference           : {pooled_ref:.3f}")
    print(f"  (ABF screen at beta=2 gave 0.564 / 0.593; deca gave 0.61)")

    AGREE = 0.15
    if ab > AGREE:
        verdict = ("INCONCLUSIVE: the two torsional starts do not agree, so even restrained "
                   "sampling has not equilibrated the hidden coordinate at this budget")
    elif pooled_ref <= AGREE:
        verdict = ("ABF CONDITIONAL SAMPLING IMPLICATED: restrained sampling reproduces the "
                   "reference, so the estimator and reference are sound and R15 beta=2 carries "
                   "the deca signature -- conditional-equilibration-limited")
    else:
        verdict = ("REFERENCE IMPLICATED: the two independent builds agree with each other but "
                   "not with the importance-sampling reference")
    print(f"\n  VERDICT: {verdict}")

    np.savez_compressed(os.path.join(args.out, "r15_mean_force.npz"),
                        centers=centers_np, thermal_mask=m,
                        mf_trans=A["mf"], mf_gauche=Bb["mf"],
                        sem_trans=A["mf_sem"], sem_gauche=Bb["mf_sem"],
                        xbar_trans=A["xbar"], xbar_gauche=Bb["xbar"],
                        cos_phi1_trans=A["cos_phi1"], cos_phi1_gauche=Bb["cos_phi1"],
                        ref_at_trans=refA, ref_at_gauche=refB)
    with open(os.path.join(args.out, "verdict.json"), "w") as fh:
        json.dump(dict(build_vs_build=ab, trans_vs_ref=a_ref, gauche_vs_ref=b_ref,
                       pooled_vs_ref=pooled_ref, agreement_threshold=AGREE,
                       abf_screen_b2_relerr=[0.564, 0.593], deca_relerr=0.61,
                       verdict=verdict, n_windows=args.n_windows, n_rep=args.n_rep,
                       k_umbrella=args.k_umbrella, equil_steps=args.equil_steps,
                       prod_steps=args.prod_steps, beta=args.beta), fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
