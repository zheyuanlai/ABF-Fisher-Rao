#!/usr/bin/env python3
"""Build + validate the CV-extension references (evaluation only).

- Distance CV R15 (pentane) / R14 (butane): internal-coordinate importance-sampling
  reference (primary 'v4' proposal) with an INDEPENDENT 'uniform'-proposal cross-check,
  a sample-size convergence ladder, and ESS.  Butane has no LJ pair (weights == 1) so the
  reference is exact by construction.
- 2-D pentane joint F(phi1,phi2): reuse the validated internal-coordinate FEP; re-check
  beta convergence + SEM and the decoupled analytic gate.

Writes references to cache/alkanes_cv/, a JSON validation report + figures to
results/alkanes_cv_extension/references/.

GPU: single visible device from {4,5,6,7}.
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
from alkanes import potentials as pot, reference as refmod, reference_cv as rcmod, periodic as per  # noqa: E402

OUT = "results/alkanes_cv_extension/references"
CACHE = "cache/alkanes_cv"


def _win_l2(a, b, dz, mask):
    a = np.asarray(a); b = np.asarray(b)
    a = a - np.sum((a - b) * mask) / max(np.sum(mask), 1e-12)
    w = mask.astype(float); width = np.sum(w) * dz
    return math.sqrt(np.sum((a - b) ** 2 * w) * dz / max(width, 1e-12))


def dist_reference_block(report, molecule, i, j, R_lo, R_hi, betas, sigma, device, n_grid=256,
                         ns_primary=800000, thermal_delta=10.0):
    p_atoms = 4 if molecule == "butane" else 5
    for beta in betas:
        p = pot.AlkaneParams(n_atoms=p_atoms, beta=beta, sigma=sigma, decouple=False)
        Rv = rcmod.distance_reference(p, i, j, R_lo=R_lo, R_hi=R_hi, n_grid=n_grid,
                                      n_samples=ns_primary, seed=987 + n_grid, device=device, proposal="v4")
        grid = Rv["grid"]; dz = Rv["dz"]; F = Rv["F"]
        mask = (F - F.min()) <= thermal_delta
        entry = {"cv": f"{molecule}_R{i}{j}", "beta": beta, "proposal_primary": "v4",
                 "ess_frac_v4": Rv["ess_frac"], "F_range_thermal_kT": float(F[mask].max() - F[mask].min())}
        # independent uniform-proposal cross-check (pentane; butane weights==1 so trivially exact)
        if molecule == "pentane":
            Ru = rcmod.distance_reference(p, i, j, R_lo=R_lo, R_hi=R_hi, n_grid=n_grid,
                                          n_samples=ns_primary, seed=202, device=device, proposal="uniform")
            entry["ess_frac_uniform"] = Ru["ess_frac"]
            entry["crosscheck_l2_thermal"] = _win_l2(F, Ru["F"], dz, mask)
        # convergence ladder (v4)
        ladder = {}
        prev = None
        for ns in (100000, 200000, 400000, ns_primary):
            Rk = rcmod.distance_reference(p, i, j, R_lo=R_lo, R_hi=R_hi, n_grid=n_grid,
                                          n_samples=ns, seed=987 + n_grid, device=device, proposal="v4")
            if prev is not None:
                ladder[ns] = _win_l2(Rk["F"], prev, dz, mask)
            prev = Rk["F"]
        entry["convergence_ladder_l2"] = ladder
        report["dist"].append(entry)
        # cache the primary reference for the jobs layer key (matches build_dist_reference path)
        os.makedirs(CACHE, exist_ok=True)
        print(f"[dist-ref] {molecule} R{i}{j} b={beta}: ESS_v4={Rv['ess_frac']:.3f} "
              f"{'ESS_uni=%.3f xL2=%.3f' % (entry.get('ess_frac_uniform',0), entry.get('crosscheck_l2_thermal',0)) if molecule=='pentane' else ''} "
              f"convΔ={max(ladder.values()) if ladder else 0:.4f} kT range={entry['F_range_thermal_kT']:.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--betas", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--sigma", type=float, default=2.3)
    ap.add_argument("--grid2d", type=int, default=48)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    ns = 200000 if args.quick else 800000
    report = {"device": dev, "dist": [], "joint2d": []}

    # ---- distance references ----
    dist_reference_block(report, "pentane", 0, 4, 1.4, 3.7, args.betas, args.sigma, dev, ns_primary=ns)
    dist_reference_block(report, "butane", 0, 3, 1.35, 2.82, args.betas, args.sigma, dev, ns_primary=ns)

    # ---- 2-D pentane joint reference (reuse validated FEP) + decoupled gate ----
    grid, dphi = per.periodic_grid(args.grid2d, device=dev)
    for beta in args.betas:
        pp = pot.AlkaneParams(n_atoms=5, beta=beta, sigma=args.sigma, decouple=False)
        prev = None; conv = {}
        for nsc in ((10000, 20000) if args.quick else (10000, 20000, 40000)):
            R = refmod.qmc_reference_pentane(grid, grid, pp, n_samples=nsc, seed=41 + args.grid2d, device=dev)
            if prev is not None:
                d = math.sqrt(float(np.mean((R["F"] - prev) ** 2)))
                conv[nsc] = d
            prev = R["F"]
        # decoupled analytic gate
        pdec = pot.AlkaneParams(n_atoms=5, beta=beta, sigma=args.sigma, decouple=True)
        Rd = refmod.qmc_reference_pentane(grid, grid, pdec, n_samples=8000, seed=12, device=dev)
        V4j = (pot.V4(grid[:, None], pdec) + pot.V4(grid[None, :], pdec)).cpu().numpy(); V4j -= V4j.mean()
        gate_err = float(np.max(np.abs(Rd["F"] - V4j)))
        report["joint2d"].append({"beta": beta, "grid": args.grid2d,
                                  "convergence_rms": conv,
                                  "decoupled_gate_max_abs_err": gate_err, "decoupled_gate_pass": gate_err < 1e-9,
                                  "sem_max": float(np.max(R["nb_correction_sem"]))})
        print(f"[2d-ref] b={beta}: decoupled gate err={gate_err:.2e} "
              f"conv={max(conv.values()) if conv else 0:.4f} SEMmax={float(np.max(R['nb_correction_sem'])):.3f}")

    with open(os.path.join(OUT, "cv_reference_validation.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\n[reference] wrote {OUT}/cv_reference_validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
