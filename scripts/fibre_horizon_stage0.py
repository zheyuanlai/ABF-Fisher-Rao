"""Stage 0 of the fibre-horizon audit: re-solve pi* at H_fib with V_fib.

Frozen protocol: ``docs/FIBRE_HORIZON_AUDIT_PREREGISTRATION.md``.

Offline arithmetic on the checkpoint the previous audit already saved -- no new
dynamics, no simulation.  The counts C_j(t=20) and the leverage a_j come from
``results/information_conversion/pilot/*_stage0_cells.csv``; the difficulty and
horizon come from the fibre reference built alongside this campaign's config.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from abffr import info_conversion as ic                           # noqa: E402

CFG = os.path.join(ROOT, "configs", "fibre_horizon", "frozen.yaml")
PREV = os.path.join(ROOT, "results", "information_conversion", "pilot")


def main():
    cfg = yaml.safe_load(open(CFG))
    K = int(cfg["simulation"]["n_particles"])
    ref = json.load(open(os.path.join(
        ROOT, cfg["output_root"], "reference", "reference_difficulty.json")))
    outdir = os.path.join(ROOT, cfg["output_root"], "stage0")
    os.makedirs(outdir, exist_ok=True)

    rows, cellrows, medians = [], [], {}
    for cell in cfg["cells"]:
        rc = ref["cells"][cell]
        V = np.array(rc["V"], float)                 # V_fib
        Vprev = np.array(rc["V_cellmean_prev"], float)
        H, M = int(rc["H"]), float(rc["M"])
        df = pd.read_csv(os.path.join(PREV, f"{cell}_stage0_cells.csv"))
        a = df.groupby("j")["a"].first().to_numpy()
        J = a.size
        av, av_prev = a * V, a * Vprev
        live = av > 0

        # asymptotic comparator, for the same contrast the previous audit drew
        r_as = np.sqrt(np.maximum(av, 0.0)); r_as = r_as / r_as.sum()
        G_asym = 1.0 - float(np.sum(av[live] / r_as[live])) / float(np.sum(av[live] * J))

        gs = []
        for seed, g in df.groupby("seed"):
            C = g.sort_values("j")["C"].to_numpy(float)
            sol = ic.solve_finite_horizon_target(av, C, M, K)
            R_opt = sol["risk"]
            R_unif = ic.predicted_finite_risk(av, C, M, np.full(J, 1.0 / J))
            G = 1.0 - R_opt / R_unif
            gs.append(G)
            # what the previous (short-horizon, cell-mean-V) solution looked like
            pi_prev = g.sort_values("j")["pi_star"].to_numpy(float)
            rows.append(dict(cell=cell, seed=int(seed), H=H, M=M,
                             G_ideal=G, G_asym=G_asym, lam=sol["lam"],
                             n_floor_bound=int(sol["floor_bound"].sum()),
                             pi_max=float(sol["pi"].max()),
                             pi_top5=float(np.sort(sol["pi"])[-5:].sum()),
                             pi_max_prev=float(pi_prev.max()),
                             pi_top5_prev=float(np.sort(pi_prev)[-5:].sum()),
                             tv_to_prev=float(0.5*np.abs(sol["pi"]-pi_prev).sum()),
                             C_total=float(C.sum())))
            for j in range(J):
                cellrows.append(dict(cell=cell, seed=int(seed), j=j, C=C[j],
                                     a=a[j], V_fib=V[j], V_prev=Vprev[j],
                                     pi_star=sol["pi"][j], pi_prev=pi_prev[j]))
        medians[cell] = float(np.median(gs))
        r = [x for x in rows if x["cell"] == cell]
        print(f"{cell}: H={H} M={M:.0f}  median G_ideal = {medians[cell]:.4f} "
              f"(range {min(gs):.3f}-{max(gs):.3f}); asymptotic comparator "
              f"{G_asym:.3f}")
        print(f"     pi* concentration: max {np.median([x['pi_max'] for x in r]):.3f} "
              f"(was {np.median([x['pi_max_prev'] for x in r]):.3f}), "
              f"top-5 {np.median([x['pi_top5'] for x in r]):.3f} "
              f"(was {np.median([x['pi_top5_prev'] for x in r]):.3f}), "
              f"cells on floor {np.median([x['n_floor_bound'] for x in r]):.0f}/{J}, "
              f"TV to old target {np.median([x['tv_to_prev'] for x in r]):.3f}")

    gmin = float(cfg["gate_0d"]["g_ideal_min"])
    passed = any(m >= gmin for m in medians.values())
    print(f"\nGATE 0  median G_ideal >= {gmin:g} in at least one cell -> "
          f"{'PASS' if passed else 'FAIL'}   {medians}")
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "stage0.csv"), index=False)
    pd.DataFrame(cellrows).to_csv(os.path.join(outdir, "stage0_cells.csv"),
                                  index=False)
    with open(os.path.join(outdir, "gate0.json"), "w") as fh:
        json.dump(dict(median_G_ideal=medians, g_ideal_min=gmin,
                       gate_pass=bool(passed)), fh, indent=1)
    print("wrote", os.path.relpath(outdir, ROOT))


if __name__ == "__main__":
    main()
