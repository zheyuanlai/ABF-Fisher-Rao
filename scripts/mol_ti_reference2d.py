"""Stratified 2-CV constrained TI reference for alanine's (phi, psi) surface."""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.mol import systems as S
from rcwfr.mol.engines import MolCfg
from rcwfr.mol.engines2d import run2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4096)          # windows per row
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--n-eq", type=int, default=60_000)
    ap.add_argument("--bw-mf", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--out", default="results/mol/ref/ALA2D_tiref.npz")
    a = ap.parse_args()
    dev, dt = torch.device("cuda"), torch.float64
    sy, g2 = S.alanine2d(dev, dt)
    cfg = MolCfg(N=a.N, n_steps=a.steps, n_cond=50, dep_every=20,
                 save_every=max(a.steps // 8, 50), n_eq=a.n_eq, bw_mf=a.bw_mf,
                 z0=-0.5236)
    t0 = time.time()
    out = run2d(sy, sy.cv, g2, cfg, a.rows, seed=a.seed,
                w_mode="none", fr_rule="none", init="grid", n_win=a.N)
    torch.cuda.synchronize()
    F = out["F"].cpu().numpy()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, F=F[-1], F_traj=F, fe=out["fe"].cpu().numpy(),
                        curl=out["curl"].cpu().numpy(),
                        ess_fix=out["ess_fix"].cpu().numpy(),
                        beta=sy.beta, N=a.N, rows=a.rows, steps=a.steps,
                        bw_mf=a.bw_mf, wall=time.time() - t0)
    g = lambda X: X - X.reshape(X.shape[0], -1).mean(1)[:, None, None]
    sd = g(F[-1]).std(0, ddof=1) / np.sqrt(a.rows)
    drift = np.sqrt(((g(F[-1]) - g(F[-2])) ** 2).mean())
    print(json.dumps({"wall_s": time.time() - t0,
                      "row_sd_rms": float(np.sqrt((sd ** 2).mean())),
                      "last_save_drift_rms": float(drift),
                      "curl_frac": float(np.median(out["curl"][-1].cpu().numpy())),
                      "F_span": float(g(F[-1]).mean(0).max() - g(F[-1]).mean(0).min()),
                      "path": a.out}, indent=1))


if __name__ == "__main__":
    main()
