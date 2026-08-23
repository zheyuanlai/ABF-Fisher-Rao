"""A stratified constrained-TI reference, for a CV whose basins unbiased MD cannot connect.

Alanine's C7ax basin is separated from the rest by ~14 k_B T in phi.  Unbiased
Brownian dynamics does not cross that in any affordable run -- the first attempt
reported P(C7ax) = 0.3548 in ALL EIGHT independent blocks, i.e. exactly its share
of the uniform initial condition, unchanged to four decimals.  No amount of
re-seeding repairs that: the estimate is of the initial condition, not of the
Boltzmann measure.

Constrained TI has no such problem.  Each window is PINNED at its own z, so the
barrier in z is never crossed and never needs to be; the only slow coordinate
left is the fiber, and every window starts with its fiber torsions spread
uniformly rather than at a point.

This is the engine Gate I validated against unbiased MD on butane (e_F = 0.053
against a 0.049 estimator floor, Fixman ESS 0.98) and on pentane, so it is a
checked instrument rather than an assumption -- and it is cross-checked against
the unbiased alanine run on the sub-arc where that run IS ergodic.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.mol import systems as S
from rcwfr.mol.engines import MolCfg, run_constrained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ALA")
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-eq", type=int, default=400_000)
    ap.add_argument("--bw-mf", type=float, default=0.05)
    ap.add_argument("--dep-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--joint-nb", type=int, default=0,
                    help=">0: also accumulate a stratified (z, y) conditional table")
    ap.add_argument("--out", default="results/mol/ref")
    a = ap.parse_args()
    dev, dt = torch.device("cuda"), torch.float64
    sy = S.REGISTRY[a.system](dev, dt)
    cfg = MolCfg(N=a.N, n_steps=a.steps, n_cond=50, dep_every=a.dep_every,
                 save_every=a.save_every, n_eq=a.n_eq, init="grid_spread",
                 n_windows=a.windows, w_mode="none", fr_rule="none", bw_mf=a.bw_mf,
                 joint_nb=a.joint_nb)
    t0 = time.time()
    out = run_constrained(sy, cfg, a.rows, seed=a.seed, ref=None)
    torch.cuda.synchronize()
    F = out["F"].cpu().numpy()              # (n_saves, rows, G)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"{a.system}_tiref.npz")
    extra = ({"Hjoint": out["Hjoint"].cpu().numpy()} if "Hjoint" in out else {})
    np.savez_compressed(p, F=F, fe=out["fe"].cpu().numpy(), **extra,
                        ess_fix=out["ess_fix"].cpu().numpy(),
                        resid=out["resid"].cpu().numpy(),
                        beta=sy.beta, N=a.N, rows=a.rows, windows=a.windows,
                        steps=a.steps, bw_mf=a.bw_mf, wall=time.time() - t0)
    g = lambda X: X - X.mean(-1, keepdims=True)
    sd = g(F[-1]).std(0, ddof=1) / np.sqrt(a.rows)
    drift = np.sqrt(((g(F[-1]) - g(F[-2])) ** 2).mean())
    print(json.dumps({"system": a.system, "wall_s": time.time() - t0,
                      "row_sd_rms": float(np.sqrt((sd ** 2).mean())),
                      "last_save_drift_rms": float(drift),
                      "ess_fixman": float(np.median(out["ess_fix"][-1].cpu().numpy())),
                      "F_span": float(g(F[-1]).mean(0).max() - g(F[-1]).mean(0).min()),
                      "path": p}, indent=1))


if __name__ == "__main__":
    main()
