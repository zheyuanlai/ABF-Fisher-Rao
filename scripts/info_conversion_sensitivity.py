#!/usr/bin/env python
"""Amendment-1 sensitivity sidecar: G_ideal as a function of the horizon.

Frozen protocol: ``docs/INFORMATION_CONVERSION_AUDIT_PREREGISTRATION.md``
(Amendment 1).  REPORTED-ONLY: nothing here is a gate input, licenses an FR
run, or can change the campaign verdict.  It exists so a Stage-0D stop is
attributable — "no exploitable heterogeneity" versus "no heterogeneity within
a horizon that is short by construction".

Two computations, both pure oracle arithmetic / oracle dynamics:

1. Fixed-x fibre tau at every allocation-cell centre, with the FROZEN AR(1)
   estimator (``information.tau_from_lag1``) applied per-walker at fixed x —
   the Gate-0I sampling design (y-channel isolated), the frozen estimator.
2. ``G_ideal(H')`` from the saved Stage-0 checkpoint counts, on the horizon
   grid H' in {H_frozen, 250, 600, 1200, 2350, 6000} plus H_fib.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from abffr import info_conversion as ic          # noqa: E402
from abffr import information as inf             # noqa: E402
from abffr import kappa_family as kfam           # noqa: E402
from abffr import potentials                     # noqa: E402

H_GRID = [250, 600, 1200, 2350, 6000]
YMIN, YMAX = -2.5, 3.5
TILT = 0.1021665783
BETA, DT = 4.0, 0.002
OBS_EVERY = 10                     # 0.02 time units, as the reference machinery


def reflect(v, lo, hi):
    r = np.mod(v - lo, 2.0 * (hi - lo))
    return lo + np.where(r > (hi - lo), 2.0 * (hi - lo) - r, r)


def fibre_tau(cell: str, centres: np.ndarray, n_walkers=512, n_steps=100_000,
              warmup=20_000, seed=1234):
    """Per-cell fixed-x fibre tau of the force observable, frozen estimator."""
    a, shift = kfam.KAPPA_CELLS[cell]
    kap = kfam.kappa_at(centres, a, shift)                 # (J,)
    rng = np.random.default_rng(seed)
    J = centres.size
    y = rng.uniform(YMIN, YMAX, (J, n_walkers))
    x = np.broadcast_to(centres[:, None], (J, n_walkers))
    noise_scale = np.sqrt(2.0 * DT / BETA) * np.sqrt(kap)[:, None]
    drift = -kap[:, None] * DT
    taus = np.full(J, np.nan)
    for j0 in range(0, J, J):                               # single block
        pass
    hists = [inf.MeanForceHistory(n_cells=n_walkers,
                                  capacity=(n_steps - warmup) // OBS_EVERY + 2)
             for _ in range(J)]
    idx = np.arange(n_walkers)
    for step in range(n_steps):
        dvdy = potentials.dVdy_xy(x, y)
        y = reflect(y + drift * dvdy
                    + noise_scale * rng.standard_normal((J, n_walkers)),
                    YMIN, YMAX)
        if step >= warmup and step % OBS_EVERY == 0:
            f = potentials.dVdx_xy(x, y) + TILT
            for j in range(J):
                hists[j].push(idx, f[j])
    for j in range(J):
        per_walker = inf.tau_from_lag1(hists[j],
                                       obs_interval=OBS_EVERY * DT)
        taus[j] = float(np.nanmedian(per_walker))
    return taus


def g_ideal_curve(cells_csv: str, av: np.ndarray, K: int, H_list, J: int):
    df = pd.read_csv(cells_csv)
    out = []
    for seed, grp in df.groupby("seed"):
        grp = grp.sort_values("j")
        C = grp.C.values
        for H in H_list:
            M = float(K * H)
            sol = ic.solve_finite_horizon_target(av, C, M, K)
            R_unif = ic.predicted_finite_risk(av, C, M, np.full(J, 1.0 / J))
            out.append(dict(seed=int(seed), H=int(H),
                            G_ideal=1.0 - sol["risk"] / R_unif))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/information_conversion")
    ap.add_argument("--stage", default="pilot")
    ap.add_argument("--cells", nargs="+", default=["K2", "K3"])
    ap.add_argument("--walkers", type=int, default=512)
    ap.add_argument("--steps", type=int, default=100_000)
    args = ap.parse_args()

    ref = json.load(open(os.path.join(
        args.root, "reference", "reference_difficulty.json")))["cells"]
    K = 256
    J = 32
    edges = np.linspace(-3.0, 3.0, J + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])

    report = {}
    for cell in args.cells:
        a_cell = np.asarray(ref[cell]["a_cell"], float)
        V = np.asarray(ref[cell]["V"], float)
        av = a_cell * V
        H_frozen = int(ref[cell]["H"])

        tau_fib = fibre_tau(cell, centres, n_walkers=args.walkers,
                            n_steps=args.steps)
        ev = a_cell > 0
        tau_fib_max = float(np.nanmax(tau_fib[ev]))
        H_fib = int(np.ceil(tau_fib_max / DT))
        H_list = sorted(set([H_frozen] + H_GRID + [H_fib]))

        curve = g_ideal_curve(
            os.path.join(args.root, args.stage, f"{cell}_stage0_cells.csv"),
            av, K, H_list, J)
        med = curve.groupby("H").G_ideal.median()
        report[cell] = dict(
            H_frozen=H_frozen, H_fib=H_fib, tau_fib_max=tau_fib_max,
            tau_fib=tau_fib.tolist(),
            tau_ref=ref[cell]["tau"],
            median_G_ideal_by_H={str(int(h)): float(g)
                                 for h, g in med.items()})
        curve.to_csv(os.path.join(args.root, args.stage,
                                  f"{cell}_sensitivity_curve.csv"), index=False)
        print(f"{cell}: tau_fib_max={tau_fib_max:.2f} (H_fib={H_fib}); "
              "median G_ideal by H: "
              + "  ".join(f"{int(h)}:{g:.3f}" for h, g in med.items()),
              flush=True)

    with open(os.path.join(args.root, args.stage, "sensitivity.json"),
              "w") as fh:
        json.dump(dict(note="Amendment 1: reported-only, never a gate input",
                       cells=report), fh, indent=1)


if __name__ == "__main__":
    main()
