"""Is the ABF local-mean-force ESTIMATOR wrong, or is ABF's conditional SAMPLING wrong?

The corrected deca screen still produced an `A_hat` spanning 87-110 kT against a 72 kT
reference, with walkers pinned above 2.80 nm, and its learned mean force disagreed with
`dF_ref/dR` by ~61 % **in bins holding up to 2e6 effective counts**. Sampling volume is
therefore not the explanation. Two hypotheses remain and they demand different responses:

  H1  the den Otter estimator `f_loc = grad V . v - beta^-1 div v` is implemented wrongly
      -> a code bug; everything built on ABF in this project is suspect

  H2  the estimator is right, but ABF's *conditional* distribution at fixed xi is not
      equilibrated -- the peptide's hidden conformational degrees of freedom do not relax at
      fixed end-to-end distance within the budget
      -> not a bug; deca-alanine is simply not ABF-tractable at 16 x 0.5 ns, which is a
         scientific finding about the benchmark

This script separates them. It runs **umbrella-restrained** dynamics -- whose conditional
sampling at fixed xi we have already validated (three builds from different conformational
pools agreeing to 0.6 kJ/mol, time drift 0.18 of a resolvable effect) -- and accumulates the
*same* `f_loc` estimator inside each window. If `<f_loc>` reproduces `dF_ref/dR`, the estimator
is correct and H2 holds. If it does not, H1 holds and the bug is in the estimator.

    python scripts/audit_deca_mean_force.py --out results/v2_validity_audits/deca_mean_force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alanine.dynamics import BAOAB                                          # noqa: E402
from alkanes.distance_cv import DistanceCV, dist_bias_force                 # noqa: E402
from deca import system as dsys                                            # noqa: E402
from deca.engine import make_engine                                        # noqa: E402
from deca.umbrella import UmbrellaConfig, diverse_pool, relax_pool          # noqa: E402

KB = 0.008314462618


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v2_validity_audits/deca_mean_force")
    ap.add_argument("--n-windows", type=int, default=25)
    ap.add_argument("--n-rep", type=int, default=48)
    ap.add_argument("--equil-steps", type=int, default=100_000)   # 100 ps
    ap.add_argument("--prod-steps", type=int, default=400_000)    # 400 ps
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype = torch.float64
    cfg = UmbrellaConfig()
    beta, dt = cfg.beta, cfg.dt

    centers_np = np.linspace(1.30, 3.50, args.n_windows)
    n_w, n_r = args.n_windows, args.n_rep
    B = n_w * n_r

    engine, system, top = make_engine(10, device=args.device, dtype=dtype, compiled=True)
    i_cv, j_cv = dsys.terminal_carbonyls(dsys.N_RES)
    cv = DistanceCV(i_cv, j_cv)
    centers = torch.as_tensor(np.repeat(centers_np, n_r), device=args.device, dtype=dtype)

    rng = np.random.default_rng(4242)
    q = torch.as_tensor(diverse_pool(B, cfg, rng), device=args.device, dtype=dtype).contiguous()
    q, ok, rep = relax_pool(engine, q)
    bad = int((~ok).sum())
    if bad:
        good = np.flatnonzero(ok)
        q[torch.as_tensor(np.flatnonzero(~ok), device=args.device)] = \
            q[torch.as_tensor(good[rng.integers(0, good.size, bad)], device=args.device)].clone()
    print(f"windows {n_w} x {n_r} replicas = {B} states; {bad} seeds replaced", flush=True)

    gen = torch.Generator(device=args.device).manual_seed(31337)
    integ = BAOAB(engine.masses, dt=dt, gamma=cfg.gamma, temperature=cfg.temperature,
                  force_fn=engine.forces, device=args.device, dtype=dtype)
    v = integ.maxwell((B, engine.n_atoms, 3), gen, args.device, dtype)
    f = engine.forces(q)
    k_u = 8000.0                      # stiff: we want tight windows, not broad ones

    def step(qq, vv, ff, ks=1.0):
        R, gf, _ = cv.geometry(qq)
        fr = dist_bias_force(gf, -k_u * ks * (R - centers))
        vv = vv + (0.5 * dt) * (ff + fr) / integ.m
        qq = qq + (0.5 * dt) * vv
        vv = integ.c1 * vv + integ.c2 * torch.randn(vv.shape, generator=gen, device=args.device,
                                                    dtype=dtype) * integ.sigma
        qq = qq + (0.5 * dt) * vv
        ff = engine.forces(qq)
        R2, gf2, _ = cv.geometry(qq)
        fr2 = dist_bias_force(gf2, -k_u * ks * (R2 - centers))
        vv = vv + (0.5 * dt) * (ff + fr2) / integ.m
        return qq, vv, ff

    t0 = time.perf_counter()
    for s in range(20_000):
        q, v, f = step(q, v, f, ks=min(1.0, (s + 1) / 10_000))
    for _ in range(args.equil_steps):
        q, v, f = step(q, v, f)
    print(f"  pull+equil done ({time.perf_counter()-t0:.0f}s)", flush=True)

    # accumulate <f_loc> and <xi> per replica
    fsum = torch.zeros(B, device=args.device, dtype=dtype)
    xsum = torch.zeros(B, device=args.device, dtype=dtype)
    n = 0
    for s in range(args.prod_steps):
        q, v, f = step(q, v, f)
        f_loc, R, _ = cv.local_mean_force(q, f, beta)
        fsum += f_loc
        xsum += R
        n += 1
        if (s + 1) % 100_000 == 0:
            print(f"    prod {100*(s+1)/args.prod_steps:5.1f}%  "
                  f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)

    mf = (fsum / n).reshape(n_w, n_r).mean(-1).cpu().numpy()
    mf_sd = (fsum / n).reshape(n_w, n_r).std(-1).cpu().numpy() / np.sqrt(n_r)
    xbar = (xsum / n).reshape(n_w, n_r).mean(-1).cpu().numpy()

    # reference mean force at the ACTUAL mean xi of each window
    import glob
    rz = np.load(sorted(glob.glob("results/deca/reference/raw/deca_umbrella__*.npz"))[-1],
                 allow_pickle=True)
    grid, F = rz["grid"], rz["F_consensus"]
    dF = np.gradient(F, grid)
    dF_at = np.interp(xbar, grid, dF)

    print("\n=== ABF local mean force from EQUILIBRATED umbrella sampling ===")
    print(f"{'R_c':>7} {'<xi>':>7} {'<f_loc>':>10} {'+-':>7} {'dF_ref/dR':>10} {'diff':>9}")
    for w in range(n_w):
        print(f"{centers_np[w]:7.3f} {xbar[w]:7.3f} {mf[w]:10.2f} {mf_sd[w]:7.2f} "
              f"{dF_at[w]:10.2f} {mf[w]-dF_at[w]:9.2f}")

    err = np.abs(mf - dF_at)
    rel = err.mean() / np.abs(dF_at).mean()
    print(f"\n  mean |<f_loc> - dF_ref/dR| : {err.mean():.2f} kJ/mol/nm")
    print(f"  mean |dF_ref/dR|            : {np.abs(dF_at).mean():.2f}")
    print(f"  RELATIVE error              : {rel:.3f}")
    print(f"  mean signed bias            : {(mf-dF_at).mean():+.2f}")
    # integrate <f_loc> and compare spans
    Fi = np.concatenate([[0.0], np.cumsum(0.5 * (mf[1:] + mf[:-1]) * np.diff(xbar))])
    Fr = np.interp(xbar, grid, F)
    Fi -= Fi.mean(); Fr -= Fr.mean()
    kT = KB * cfg.temperature
    print(f"  span from integrating <f_loc>: {(Fi.max()-Fi.min())/kT:.1f} kT")
    print(f"  span of F_ref on same range  : {(Fr.max()-Fr.min())/kT:.1f} kT")
    verdict = "ESTIMATOR OK -> H2 (ABF conditional sampling)" if rel < 0.15 else \
              "ESTIMATOR DISAGREES -> H1 (code bug in the mean force)"
    print(f"\n  VERDICT: {verdict}")

    np.savez_compressed(os.path.join(args.out, "mean_force_audit.npz"),
                        centers=centers_np, xbar=xbar, mf=mf, mf_sd=mf_sd, dF_at=dF_at,
                        F_int=Fi, F_ref_at=Fr)
    with open(os.path.join(args.out, "verdict.json"), "w") as fh:
        json.dump(dict(relative_error=float(rel), mean_abs_error=float(err.mean()),
                       mean_signed_bias=float((mf - dF_at).mean()),
                       span_integrated_kT=float((Fi.max() - Fi.min()) / kT),
                       span_reference_kT=float((Fr.max() - Fr.min()) / kT),
                       estimator_ok=bool(rel < 0.15), verdict=verdict,
                       n_windows=n_w, n_rep=n_r, k_umbrella=k_u,
                       equil_steps=args.equil_steps, prod_steps=args.prod_steps), fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
