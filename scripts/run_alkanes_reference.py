#!/usr/bin/env python3
"""Build + validate the independent alkane references (evaluation only).

- B0/P0 exact gates: decoupled butane/pentane FEP == V4(+V4)+C.
- Full-model references: butane (== V4, pipeline cross-check) and pentane joint
  F(phi1,phi2) via internal-coordinate FEP; 1-D F(phi1) by marginalisation.
- Convergence ladder over FEP sample count + delta-method bootstrap SEM (cross-check).

Writes references to cache/alkanes/, a JSON validation report and figures to
results/alkanes/references/.

GPU: uses the single visible CUDA device (from {4,5,6,7}); CPU fallback works but is
slow. Run with CUDA_VISIBLE_DEVICES=<one of 4-7>.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)
from alkanes import potentials as pot, periodic as per, reference as refmod  # noqa: E402

OUT = "results/alkanes/references"
GAUCHE = math.radians(116.57)


def _circ_l2(a, b, dphi):
    a = a - np.mean(a - b)
    return math.sqrt(np.sum((a - b) ** 2) * dphi / (2 * math.pi))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-grid", type=int, default=180)
    ap.add_argument("--n-grid2", type=int, default=48)
    ap.add_argument("--sigma", type=float, default=2.3)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    args = ap.parse_args(argv)
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("cache/alkanes", exist_ok=True)
    report = {"device": dev, "checks": []}
    grid, dphi = per.periodic_grid(args.n_grid, device=dev)
    g2, dphi2 = per.periodic_grid(args.n_grid2, device=dev)

    # ---------------- B0/P0 exact decoupled gates ----------------
    for beta in args.betas:
        pb = pot.AlkaneParams(n_atoms=4, beta=beta, decouple=True)
        Rb = refmod.qmc_reference_butane(grid, pb, n_samples=8000, seed=11, device=dev)
        V4c = (pot.V4(grid, pb) - pot.V4(grid, pb).mean()).cpu().numpy()
        err = float(np.max(np.abs(Rb["F"] - V4c)))
        report["checks"].append({"gate": "B0_decoupled_butane", "beta": beta,
                                 "max_abs_F_minus_V4": err, "pass": err < 1e-9})
        pp = pot.AlkaneParams(n_atoms=5, beta=beta, sigma=args.sigma, decouple=True)
        Rp = refmod.qmc_reference_pentane(g2, g2, pp, n_samples=8000, seed=12, device=dev)
        V4j = (pot.V4(g2[:, None], pp) + pot.V4(g2[None, :], pp)).cpu().numpy()
        V4j = V4j - V4j.mean()
        errp = float(np.max(np.abs(Rp["F"] - V4j)))
        report["checks"].append({"gate": "P0_decoupled_pentane_joint", "beta": beta,
                                 "max_abs_F_minus_V4V4": errp, "pass": errp < 1e-9})

    # ---------------- Full references + convergence ladder ----------------
    for beta in args.betas:
        # butane full == V4 (no LJ) -- pipeline cross-check
        pbf = pot.AlkaneParams(n_atoms=4, beta=beta, decouple=False)
        Rbf = refmod.qmc_reference_butane(grid, pbf, n_samples=20000, seed=21, device=dev)
        np.savez(f"cache/alkanes/ref_butane_b{beta:g}_full_g{args.n_grid}.npz",
                 grid=grid.cpu().numpy(), F=Rbf["F"], Fprime=Rbf["Fprime"])
        # pentane full joint + convergence ladder
        ppf = pot.AlkaneParams(n_atoms=5, beta=beta, sigma=args.sigma, decouple=False)
        ladder = {}
        prev = None
        for ns in (5000, 10000, 20000, 40000):
            R = refmod.qmc_reference_pentane(g2, g2, ppf, n_samples=ns, seed=31, device=dev)
            F1 = refmod.marginalize_joint_to_phi1(R["F"], g2.cpu(), g2.cpu(), beta)
            ladder[ns] = F1 - F1.mean()
            if prev is not None:
                d = _circ_l2(ladder[ns], prev, dphi2)
                report["checks"].append({"gate": "pentane_convergence", "beta": beta,
                                         "n_samples": ns, "delta_F1_from_prev": d})
            prev = ladder[ns]
        Rj = refmod.qmc_reference_pentane(grid, g2, ppf, n_samples=40000, seed=41, device=dev)
        F1 = refmod.marginalize_joint_to_phi1(Rj["F"], grid.cpu(), g2.cpu(), beta)
        report["checks"].append({"gate": "pentane_ref_SEM_max", "beta": beta,
                                 "max_nb_correction_sem": float(np.max(Rj["nb_correction_sem"]))})
        np.savez(f"cache/alkanes/ref_pentane_b{beta:g}_s{args.sigma:g}_full_g{args.n_grid}.npz",
                 grid=grid.cpu().numpy(), F=(F1 - F1.mean()), joint_F=Rj["F"],
                 grid2=g2.cpu().numpy(), nb_sem=Rj["nb_correction_sem"])

    with open(os.path.join(OUT, "reference_validation.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    n_fail = sum(1 for c in report["checks"] if "pass" in c and not c["pass"])
    print(json.dumps(report, indent=2))
    print(f"\n[reference] gates: {sum(1 for c in report['checks'] if c.get('pass'))} pass, {n_fail} fail")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
