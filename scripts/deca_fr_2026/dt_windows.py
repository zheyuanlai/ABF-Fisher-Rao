"""Stage I-B, direct form: the conditional mean force <f_loc | R> in stiff umbrella windows at two
timesteps (the 2026-08 `audit_deca_mean_force.py` protocol, parametrised by dt).

    python scripts/deca_fr_2026/dt_windows.py --dt 0.001  --out results/deca_fr_2026/validation/dt_windows_1fs
    python scripts/deca_fr_2026/dt_windows.py --dt 0.0005 --out results/deca_fr_2026/validation/dt_windows_05fs

Same windows, same seeds, same physical pull / equilibration / production times.  What the
ABF-based comparison cannot separate (integration error vs. where the walkers happen to be at
200 ps), this can: the restrained conditional ensemble is near equilibrium in every window, so a
timestep effect shows up directly as a systematic shift of <f_loc> relative to its standard error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                                          # noqa: E402
from alanine.dynamics import BAOAB                                          # noqa: E402
from alkanes.distance_cv import DistanceCV, dist_bias_force                 # noqa: E402
from deca import system as dsys                                             # noqa: E402
from deca.umbrella import UmbrellaConfig, diverse_pool, relax_pool          # noqa: E402

KB = 0.008314462618


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-windows", type=int, default=25)
    ap.add_argument("--n-rep", type=int, default=48)
    ap.add_argument("--pull-ps", type=float, default=20.0)
    ap.add_argument("--equil-ps", type=float, default=100.0)
    ap.add_argument("--prod-ps", type=float, default=400.0)
    ap.add_argument("--k-umbrella", type=float, default=8000.0)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev, dtype = args.device, torch.float64
    cfg = UmbrellaConfig()
    beta, dt = cfg.beta, args.dt
    centers_np = np.linspace(1.30, 3.50, args.n_windows)
    n_w, n_r = args.n_windows, args.n_rep
    B = n_w * n_r
    engine = C.engine_for(dev)
    i_cv, j_cv = dsys.terminal_carbonyls(dsys.N_RES)
    cv = DistanceCV(i_cv, j_cv)
    centers = torch.as_tensor(np.repeat(centers_np, n_r), device=dev, dtype=dtype)
    rng = np.random.default_rng(args.seed)
    q = torch.as_tensor(diverse_pool(B, cfg, rng), device=dev, dtype=dtype).contiguous()
    q, ok, _ = relax_pool(engine, q)
    bad = int((~ok).sum())
    if bad:
        good = np.flatnonzero(ok)
        q[torch.as_tensor(np.flatnonzero(~ok), device=dev)] = \
            q[torch.as_tensor(good[rng.integers(0, good.size, bad)], device=dev)].clone()
    gen = torch.Generator(device=dev).manual_seed(args.seed + 31337)
    integ = BAOAB(engine.masses, dt=dt, gamma=cfg.gamma, temperature=cfg.temperature,
                  force_fn=engine.forces, device=dev, dtype=dtype)
    v = integ.maxwell((B, engine.n_atoms, 3), gen, dev, dtype)
    f = engine.forces(q)
    k_u = args.k_umbrella

    def step(qq, vv, ff, ks=1.0):
        R, gf, _ = cv.geometry(qq)
        fr = dist_bias_force(gf, -k_u * ks * (R - centers))
        vv = vv + (0.5 * dt) * (ff + fr) / integ.m
        qq = qq + (0.5 * dt) * vv
        vv = integ.c1 * vv + integ.c2 * torch.randn(vv.shape, generator=gen, device=dev, dtype=dtype) * integ.sigma
        qq = qq + (0.5 * dt) * vv
        ff = engine.forces(qq)
        R2, gf2, _ = cv.geometry(qq)
        fr2 = dist_bias_force(gf2, -k_u * ks * (R2 - centers))
        vv = vv + (0.5 * dt) * (ff + fr2) / integ.m
        return qq, vv, ff

    n_pull, n_eq, n_prod = (int(round(x / dt)) for x in (args.pull_ps, args.equil_ps, args.prod_ps))
    print(f"dt={dt} ps: {n_w} windows x {n_r} = {B} states; pull {n_pull} equil {n_eq} prod {n_prod} steps; "
          f"{bad} seeds replaced", flush=True)
    t0 = time.perf_counter()
    for s in range(n_pull):
        q, v, f = step(q, v, f, ks=min(1.0, (s + 1) / max(n_pull // 2, 1)))
    for _ in range(n_eq):
        q, v, f = step(q, v, f)
    ok_t, rep_t = dsys.validate_thermal(q.cpu().numpy(), dsys.N_RES)
    print(f"  pull+equil done ({time.perf_counter()-t0:.0f}s); structural {int(ok_t.sum())}/{B} pass", flush=True)
    fsum = torch.zeros(B, device=dev, dtype=dtype); f2sum = torch.zeros(B, device=dev, dtype=dtype)
    xsum = torch.zeros(B, device=dev, dtype=dtype); Tsum = 0.0
    n = 0
    for s in range(n_prod):
        q, v, f = step(q, v, f)
        f_loc, R, _ = cv.local_mean_force(q, f, beta)
        fsum += f_loc; f2sum += f_loc * f_loc; xsum += R; n += 1
        if (s + 1) % 1000 == 0:
            Tsum += float(integ.kinetic_temperature(v))
        if (s + 1) % (n_prod // 4) == 0:
            print(f"    prod {100*(s+1)/n_prod:5.1f}%  {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    ok_e, rep_e = dsys.validate_thermal(q.cpu().numpy(), dsys.N_RES)
    keep = torch.as_tensor(ok_t & ok_e, device=dev)
    mf_rep = (fsum / n).reshape(n_w, n_r).cpu().numpy()
    x_rep = (xsum / n).reshape(n_w, n_r).cpu().numpy()
    keep_np = keep.reshape(n_w, n_r).cpu().numpy()
    mf = np.array([mf_rep[w][keep_np[w]].mean() for w in range(n_w)])
    se = np.array([mf_rep[w][keep_np[w]].std(ddof=1) / np.sqrt(keep_np[w].sum()) for w in range(n_w)])
    xbar = np.array([x_rep[w][keep_np[w]].mean() for w in range(n_w)])
    T_kin = Tsum / max(n_prod // 1000, 1)
    np.savez_compressed(os.path.join(args.out, "windows.npz"), centers=centers_np, xbar=xbar, mf=mf, mf_se=se,
                        mf_rep=mf_rep, x_rep=x_rep, keep=keep_np, dt=dt, n_prod=n_prod)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(dict(dt=dt, n_windows=n_w, n_rep=n_r, pull_ps=args.pull_ps, equil_ps=args.equil_ps,
                       prod_ps=args.prod_ps, k_umbrella=k_u, seed=args.seed, n_kept=int(keep_np.sum()),
                       kinetic_temperature_K=T_kin, n_fail_cis=int(rep_e["n_fail_cis"]),
                       n_fail_chirality=int(rep_e["n_fail_chirality"]),
                       runtime_min=(time.perf_counter() - t0) / 60, git=C.git_commit()), fh, indent=2)
    print(f"  done {(time.perf_counter()-t0)/60:.1f} min; kept {int(keep_np.sum())}/{B}; T_kin {T_kin:.1f} K")
    print("   R_c     <xi>    <f_loc>   +-se")
    for w in range(n_w):
        print(f"  {centers_np[w]:5.2f}  {xbar[w]:6.3f}  {mf[w]:8.2f}  {se[w]:5.2f}")


if __name__ == "__main__":
    main()
