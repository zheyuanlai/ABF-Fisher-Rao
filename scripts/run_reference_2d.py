#!/usr/bin/env python3
"""Compute and plot the reference free energy / mean force for xi(x, y) = x.

Outputs (under <output_root>/reference/):
    reference_profile.csv      columns: x, F_ref, Fprime_ref, p_ref
    reference_grid.npz         arrays:  x_grid, y_grid, V_grid, rho_grid
    fig_potential_contour.png
    fig_target_density.png
    fig_reference_free_energy.png
    fig_reference_mean_force.png

Example:
    python scripts/run_reference_2d.py --config configs/two_dim_xi_x_smoke.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abffr import io_utils, plotting, potentials, reference  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--output-root", default=None,
                   help="Override output_root from the config.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = io_utils.load_config(args.config)
    if args.output_root:
        cfg["output_root"] = args.output_root

    beta = float(cfg["simulation"]["beta"])
    domain = cfg["domain"]
    out_dir = io_utils.reference_dir(cfg)
    plotting.set_style()

    # --- 1-D reference profiles on the reaction-coordinate grid ------------- #
    x_profile = reference.profile_grid(cfg)
    y_quad = np.linspace(domain["y_min"], domain["y_max"], int(domain["ny_ref"]))
    ref = reference.compute_reference(x_profile, y_quad, beta)

    df = pd.DataFrame(dict(
        x=x_profile,
        F_ref=ref["F_ref"],
        Fprime_ref=ref["Fprime_ref"],
        p_ref=ref["p_ref"],
    ))
    csv_path = os.path.join(out_dir, "reference_profile.csv")
    df.to_csv(csv_path, index=False)

    # --- 2-D reference grid (potential + Boltzmann density) ----------------- #
    x_grid, y_grid, V_grid, rho_grid = reference.build_reference_grid(cfg, beta)
    npz_path = os.path.join(out_dir, "reference_grid.npz")
    np.savez_compressed(npz_path, x_grid=x_grid, y_grid=y_grid,
                        V_grid=V_grid, rho_grid=rho_grid)

    # --- Figures ------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    cf = plotting.potential_contour(ax, x_grid, y_grid, V_grid,
                                    title=r"Potential $V(x,y)$")
    fig.colorbar(cf, ax=ax)
    f1 = plotting.save_fig(fig, os.path.join(out_dir, "fig_potential_contour.png"))

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    cf = plotting.density_contour(ax, x_grid, y_grid, rho_grid,
                                  title=r"Target density $\propto e^{-\beta V}$"
                                        f"  ($\\beta={beta:g}$)")
    fig.colorbar(cf, ax=ax)
    f2 = plotting.save_fig(fig, os.path.join(out_dir, "fig_target_density.png"))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    plotting.reference_profile(ax, x_profile, ref["F_ref"], r"$F_{\rm ref}(x)$",
                               r"Reference free energy $F_{\rm ref}(x)$")
    f3 = plotting.save_fig(fig, os.path.join(out_dir, "fig_reference_free_energy.png"))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    plotting.reference_profile(ax, x_profile, ref["Fprime_ref"],
                               r"$F'_{\rm ref}(x)$",
                               r"Reference mean force $F'_{\rm ref}(x)$")
    f4 = plotting.save_fig(fig, os.path.join(out_dir, "fig_reference_mean_force.png"))

    # --- Report ------------------------------------------------------------- #
    print("[run_reference_2d] wrote:")
    for path in (csv_path, npz_path, f1, f2, f3, f4):
        print("   ", os.path.relpath(path))
    print(f"[run_reference_2d] beta={beta}, profile grid={len(x_profile)} pts, "
          f"y-quadrature={len(y_quad)} pts")
    print(f"[run_reference_2d] F_ref range=[{ref['F_ref'].min():.3f}, "
          f"{ref['F_ref'].max():.3f}], all finite="
          f"{bool(np.all(np.isfinite(ref['F_ref'])) and np.all(np.isfinite(ref['Fprime_ref'])))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
