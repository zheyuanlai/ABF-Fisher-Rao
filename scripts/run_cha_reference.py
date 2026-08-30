#!/usr/bin/env python
"""Independent umbrella/WHAM reference for one olefin/CHA cell, plus the
U(xi) / U_nonbond(xi) conditionals for the entropy decomposition.

    CUDA_VISIBLE_DEVICES=2 python -u scripts/run_cha_reference.py --guest ethene --temperature 450
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from publication_style import PALETTE, apply_publication_style, save_figure  # noqa: E402
from cha.core_cha import (KB, CHASimConfig, CHASystem, conditional_u_line,  # noqa: E402
                          run_umbrella, wham_1d_line)

PREREG = os.path.join(ROOT, "configs/uniform_campaign/cha_prereg.json")
OUT = os.path.join(ROOT, "results/uniform_campaign/cha/reference")


def barrier(mids, F, xi_A, xi_B):
    win = np.abs(mids) < 0.5
    cage = ((np.abs(mids - xi_A) < 1.0) | (np.abs(mids - xi_B) < 1.0))
    return float(np.nanmean(F[win]) - np.nanmin([np.nanmean(F[np.abs(mids - x) < 1.0])
                                                 for x in (xi_A, xi_B)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guest", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    rr = pre["reference_rule"]
    tag = f"{a.guest}_{a.temperature:g}"
    seed = rr["seeds"][tag]
    os.makedirs(OUT, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = CHASystem(a.guest, a.temperature, device, root=ROOT)
    s = pre["sampler"]
    sim = CHASimConfig(**{k: v for k, v in s.items() if not k.startswith("_")})
    kT = KB * a.temperature
    kappa = kT / 0.20 ** 2
    centers = np.linspace(sim.xi_lo + 0.3, sim.xi_hi - 0.3, rr["windows"])
    print(f"CHA reference {tag}: {rr['windows']} windows, kappa={kappa:.1f}, "
          f"{rr['n_rep']} reps, {rr['n_steps']} steps", flush=True)

    phis, us, unbs = run_umbrella(system, sim, centers, kappa, rr["n_steps"],
                                  rr["n_rep"], rr["burn_in"], rr["sample_every"], seed)
    beta = system.beta
    mids, F, p, hist = wham_1d_line(phis, centers, kappa, beta, sim.xi_lo, sim.xi_hi,
                                    rr["n_bins"])
    T2 = phis.shape[0] // 2
    _, Fa, _, _ = wham_1d_line(phis[:T2], centers, kappa, beta, sim.xi_lo, sim.xi_hi,
                               rr["n_bins"])
    _, Fb, _, _ = wham_1d_line(phis[T2:], centers, kappa, beta, sim.xi_lo, sim.xi_hi,
                               rr["n_bins"])
    dF = barrier(mids, F, system.xi_A, system.xi_B)
    dFa, dFb = barrier(mids, Fa, system.xi_A, system.xi_B), \
        barrier(mids, Fb, system.xi_A, system.xi_B)

    _, U, ucnt = conditional_u_line(phis, us, sim.xi_lo, sim.xi_hi, rr["n_bins"])
    _, Unb, _ = conditional_u_line(phis, unbs, sim.xi_lo, sim.xi_hi, rr["n_bins"])
    ok = ucnt > 100
    U = np.where(ok, U, np.nan); Unb = np.where(ok, Unb, np.nan)
    ref0 = np.nanmin([np.nanmean(U[np.abs(mids - x) < 1.0])
                      for x in (system.xi_A, system.xi_B)])
    U = U - ref0
    dU = barrier(mids, U, system.xi_A, system.xi_B)
    mTdS = dF - dU
    # adjacent-window overlap check
    ov = []
    for w in range(hist.shape[0] - 1):
        both = (hist[w] > 0) & (hist[w + 1] > 0)
        ov.append(int(both.sum()))
    print(f"  dF = {dF:.2f} kJ/mol = {dF*beta:.2f} kT "
          f"(split {dFa*beta:.2f}/{dFb*beta:.2f} kT)")
    print(f"  dU = {dU:.2f} kJ/mol = {dU*beta:.2f} kT; -T dS = {mTdS:.2f} kJ/mol "
          f"= {mTdS*beta:.2f} kT ({100*mTdS/dF if dF else float('nan'):.0f}%)")
    print(f"  min adjacent-window overlap: {min(ov)} bins")
    assert abs(dFa - dFb) * beta < 0.3, "split-half acceptance failed"
    assert min(ov) >= 2, "umbrella chain disconnected"

    np.savez_compressed(os.path.join(OUT, f"reference_{tag}.npz"),
                        grid=mids, F=F, U=U, U_nonbond=Unb, p=p, u_counts=ucnt,
                        F_split=[Fa, Fb], window_hist=hist,
                        xi_A=system.xi_A, xi_B=system.xi_B,
                        guest=a.guest, temperature=a.temperature, kT=kT,
                        dF_barrier=dF, dU_barrier=dU, mTdS_barrier=mTdS,
                        protocol=json.dumps(dict(rr, kappa=kappa, tag=tag)))
    print(f"  wrote reference_{tag}.npz")

    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
    ax = axes[0]
    ax.plot(mids, F * beta, color=PALETTE["black"], lw=1.5, label=r"$F(\xi)$")
    ax.plot(mids, U * beta, color=PALETTE["blue"], lw=1.3, ls="--", label=r"$U(\xi)$")
    ax.plot(mids, (F - U) * beta, color=PALETTE["vermillion"], lw=1.3, ls="-.",
            label=r"$-TS(\xi)$")
    for x in (system.xi_A, system.xi_B, 0.0):
        ax.axvline(x, color=PALETTE["gray"], lw=0.6, ls=":")
    ax.set_xlabel(r"$\xi$ (A)  [window at 0]")
    ax.set_ylabel("kT")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"{a.guest} {a.temperature:g} K: "
                 f"$\\Delta F^\\ddagger$={dF*beta:.1f} kT, "
                 f"$\\Delta U^\\ddagger$={dU*beta:.1f}, "
                 f"$-T\\Delta S^\\ddagger$={mTdS*beta:.1f}", fontsize=9)
    ax = axes[1]
    ax.plot(mids, p, color=PALETTE["black"], lw=1.3)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\xi$ (A)")
    ax.set_ylabel(r"$p(\xi)$ (WHAM)")
    save_figure(fig, os.path.join(OUT, f"fig_cha_reference_{tag}"))
    print(f"  wrote fig_cha_reference_{tag}")


if __name__ == "__main__":
    main()
