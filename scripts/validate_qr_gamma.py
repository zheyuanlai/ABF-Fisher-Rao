#!/usr/bin/env python
"""Stage 1B: is ``Gamma_hat`` good enough to allocate on?

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md``.

The allocation arms follow measured difficulty.  If the online estimator cannot
see the difficulty the kappa-family builds, then A4b and A5 are following noise
and a null result would say nothing about the Neyman hypothesis -- it would say
the estimator was not ready.  So this gate runs **before** A4b and A5 do, and it
can stop the campaign.

What it compares
----------------
``Gamma_ref``: from long plain-ABF runs (4 seeds x 4T), batch means with blocks
set from the measured ``tau`` rather than guessed.  Offline, evaluation-only,
and it never reaches a candidate.

``Gamma_hat``: exactly what an arm would see mid-run -- the decomposed
``sigma^2 tau`` estimator reading the same eligible stream the accumulator does.
Measured on **A2**, which runs the estimator and changes nothing else, so the
validation is not comparing an estimator against dynamics it has itself
perturbed.

Reported
--------
rank correlation, multiplicative error, top-difficulty-cell overlap, and the
**K2/K3 inversion**: the campaign's mirror test needs the estimator to place the
hard region on opposite sides in those two cells.  If it cannot, do not run
A4b/A5 -- that is an estimator failure, not a method failure.

Usage::

    python scripts/validate_qr_gamma.py --out results/qr_decoupling/stage1b
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abffr import (allocation as al, information as inf, kappa_family as kf,
                   metrics, qr_arms as qra, reference, simulation_torch)
from abffr.io_utils import RunSpec

DOMAIN = {"x_min": -3.0, "x_max": 3.0, "y_min": -2.5, "y_max": 3.5}
TILT, BETA = 0.1021665783, 4.0


def base_cfg(cell, n_steps, n_particles, arm=None, n_cells=32):
    cfg = {
        "simulation": {"beta": BETA, "dt": 0.002, "n_steps": n_steps,
                       "n_particles": n_particles, "eval_every": 500,
                       "x_init_mode": "uniform", "y_init_mode": "uniform"},
        "domain": dict(DOMAIN), "potential": {"x_tilt": TILT},
        "kappa": {"cell": cell},
        "abf": {"estimator": "binned_smooth",
                "observation_order": "post_propagation",
                "h": 0.05, "update_every": 10, "min_count": 1.0},
        "fr": {"enabled": False, "noise_chunk_steps": 256},
    }
    if arm is not None:
        cfg["qr"] = dict(enabled=True, arm=arm, n_cells=n_cells,
                         opportunity_every=500, burnin_fraction=0.20,
                         stop_fraction=0.80, history_capacity=400)
    return cfg


def run(cfg, seeds, x, ref, device):
    specs = [RunSpec(method="abf_only", target_type="none", seed=int(s),
                     gamma=0.0, eta=0.10, burnin_fraction=0.0, fr_every=1,
                     stop_fraction=1.0) for s in seeds]
    return simulation_torch.run_batch(
        specs, cfg=cfg, x_grid=x, F_ref=ref["F_ref"],
        Fprime_ref=ref["Fprime_ref"],
        ev=metrics.EvalConfig.from_domain(DOMAIN), device=device,
        dtype=torch.float64, estimator="binned_smooth", base_seed=0)


def gamma_reference(cell, x, ref, device, n_steps, n_particles, n_cells, seeds):
    """Long-run difficulty, offline.  Never reaches a candidate."""
    cfg = base_cfg(cell, n_steps, n_particles, arm="A2", n_cells=n_cells)
    cfg["qr"]["history_capacity"] = 100_000        # the whole run, for the fit
    res = run(cfg, seeds, x, ref, device)
    return res


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/qr_decoupling/stage1b")
    ap.add_argument("--n-cells", type=int, default=32)
    ap.add_argument("--short-steps", type=int, default=50_000)
    ap.add_argument("--long-steps", type=int, default=200_000)
    ap.add_argument("--particles", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[5100, 5101, 5102, 5103])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    x = np.linspace(DOMAIN["x_min"], DOMAIN["x_max"], 401)
    y = np.linspace(DOMAIN["y_min"], DOMAIN["y_max"], 801)
    ref = reference.compute_reference(x, y, beta=BETA, x_tilt=TILT)

    report = {"n_cells": args.n_cells, "cells": {}}
    prof = {}
    for cell in ("K0", "K2", "K3"):
        short = run(base_cfg(cell, args.short_steps, args.particles,
                             arm="A2", n_cells=args.n_cells),
                    args.seeds, x, ref, device)
        long = gamma_reference(cell, x, ref, device, args.long_steps,
                               args.particles, args.n_cells, args.seeds)
        g_hat = np.median([d["qr_gamma_final"] for d in short.diags], axis=0)
        g_ref = np.median([d["qr_gamma_final"] for d in long.diags], axis=0)
        live = np.isfinite(g_hat) & np.isfinite(g_ref) & (g_ref > 0)
        k = max(3, args.n_cells // 8)
        top_hat = set(np.argsort(-np.where(live, g_hat, -np.inf))[:k])
        top_ref = set(np.argsort(-np.where(live, g_ref, -np.inf))[:k])
        prof[cell] = g_hat
        report["cells"][cell] = {
            "spearman": spearman(g_hat[live], g_ref[live]),
            "median_mult_error": float(np.median(
                np.abs(np.log(g_hat[live] / g_ref[live])))),
            "top_cell_overlap": len(top_hat & top_ref) / k,
            "spread_hat": float(np.nanmax(g_hat[live]) / np.nanmin(g_hat[live])),
            "spread_ref": float(np.nanmax(g_ref[live]) / np.nanmin(g_ref[live])),
        }
        print(f"{cell}: " + "  ".join(f"{k2}={v:.3f}" for k2, v in
                                      report["cells"][cell].items()), flush=True)

    # The mirror test the campaign's H3 rests on.
    inv = spearman(prof["K2"], prof["K3"][::-1])
    report["k2_k3_mirror_spearman"] = inv
    report["verdict"] = (
        "GO" if (inv > 0.5
                 and all(report["cells"][c]["spearman"] > 0.5
                         for c in ("K2", "K3"))) else "STOP")
    print(f"\nK2 vs mirrored K3 rank correlation: {inv:.3f}")
    print(f"VERDICT: {report['verdict']}"
          + ("" if report["verdict"] == "GO" else
             "  -- do NOT run A4b/A5; fix the estimator first"))
    with open(os.path.join(args.out, "gamma_validation.json"), "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
