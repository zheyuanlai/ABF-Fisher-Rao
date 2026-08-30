#!/usr/bin/env python
"""Independent reference for ethane/LTA: umbrella + WHAM F(z), conditional U(z),
and the entropic-barrier decomposition -T dS = dF - dU.

This is NOT a method arm: it exists so the two-arm production can be scored
against a reference that never touches the ABF estimator, and so the claim
"molecular ENTROPIC barrier" is measured, not asserted.  A long unbiased run
cross-checks the WHAM profile wherever unbiased sampling reaches.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_lta_reference.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from publication_style import PALETTE, apply_publication_style, save_figure  # noqa: E402
from lta.core_lta import (KB, LTAParams, LTASimConfig, LTASystem,  # noqa: E402
                          conditional_u, run_umbrella, wham_1d)

PI = math.pi
OUT = os.path.join(ROOT, "results/uniform_campaign/lta/reference")

# ---- frozen reference protocol (recorded into the artifact) ----
# Defaults are the closed 300 K stage's values; the temperature-sweep prereg
# (configs/uniform_campaign/lta_sweep_prereg.json) overrides kappa/steps/burn-in
# per T on the command line, and those values are recorded into the artifact.
N_WINDOWS = 40
KAPPA = 300.0            # kJ/mol/rad^2  -> window sd ~ sqrt(kT/kappa) ~ 0.09 rad
N_REP = 256
N_STEPS = 150_000
BURN_IN = 30_000
SAMPLE_EVERY = 25
N_BINS = 180
SEED = 20260829
UNBIASED_N = 16_384
UNBIASED_STEPS = 300_000


def unbiased_check(system, sim, n, n_steps, seed):
    """Long unbiased BD; histogram free energy where sampling reaches."""
    device, dtype = system.device, system.dtype
    gen = torch.Generator(device=device).manual_seed(seed)
    q = system.initial_conditions(1, n, gen).reshape(n, 2, 3)
    noise_scale = math.sqrt(2.0 * sim.dt / system.p.beta)
    edges = np.linspace(-PI, PI, N_BINS + 1)
    hist = np.zeros(N_BINS)
    t0 = time.time()
    for step in range(n_steps):
        F = system.forces(q)
        q = q + sim.dt * F + noise_scale * torch.randn(
            q.shape, generator=gen, device=device, dtype=dtype)
        if step % 20 == 0 and step > n_steps // 5:
            phi = system.cv_value(q).cpu().numpy()
            h, _ = np.histogram(phi, bins=edges)
            hist += h
    print(f"  unbiased check: {n} molecules x {n_steps} steps in {time.time()-t0:.0f}s, "
          f"{int(hist.sum())} samples", flush=True)
    return hist


def barrier_stats(mids, F, a):
    """dF between the window (phi=0) and the cage (phi=+-pi), by local means."""
    z = mids * a / (2 * PI)
    win = np.abs(z) < 0.4
    cage = np.abs(np.abs(mids) - PI) < 0.4 * 2 * PI / a
    return float(np.nanmean(F[win]) - np.nanmean(F[cage]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    ap.add_argument("--burn-in", type=int, default=BURN_IN)
    ap.add_argument("--n-rep", type=int, default=N_REP)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--unbiased-steps", type=int, default=UNBIASED_STEPS)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = LTAParams(temperature=a.temperature)
    system = LTASystem(params, device, root=ROOT)
    sim = LTASimConfig()
    beta = params.beta
    kT = 1.0 / beta
    print(f"LTA reference at T={a.temperature:g} K (kT={kT:.4f} kJ/mol), device={device}")

    centers = np.linspace(-PI, PI, N_WINDOWS, endpoint=False)
    phis, us = run_umbrella(system, sim, centers, a.kappa, a.n_steps, a.n_rep,
                            a.burn_in, SAMPLE_EVERY, a.seed)

    mids, F_phi, p_phi, hist = wham_1d(phis, centers, a.kappa, beta, N_BINS)
    # split-half consistency of the barrier
    T2 = phis.shape[0] // 2
    _, F_a, _, _ = wham_1d(phis[:T2], centers, a.kappa, beta, N_BINS)
    _, F_b, _, _ = wham_1d(phis[T2:], centers, a.kappa, beta, N_BINS)
    dF = barrier_stats(mids, F_phi, system.a)
    dF_a = barrier_stats(mids, F_a, system.a)
    dF_b = barrier_stats(mids, F_b, system.a)

    mids_u, U_phi, u_counts = conditional_u(phis, us, N_BINS)
    assert np.allclose(mids, mids_u)
    # align U like F: zero mean over bins with data
    U_phi = U_phi - np.nanmean(U_phi)
    dU = barrier_stats(mids, U_phi, system.a)
    TS_phi = U_phi - F_phi                     # T*S(z) up to a constant
    mTdS = dF - dU                             # -T dS(barrier) = dF - dU

    # unbiased cross-check
    hist_unb = unbiased_check(system, sim, UNBIASED_N, a.unbiased_steps, a.seed + 7)
    with np.errstate(divide="ignore"):
        F_unb = -kT * np.log(hist_unb / max(hist_unb.sum(), 1))
    F_unb[hist_unb < 50] = np.nan
    if np.isfinite(F_unb).any():
        F_unb = F_unb - np.nanmean(F_unb - F_phi)     # align to WHAM where defined
    overlap = np.isfinite(F_unb)
    rms = (float(np.sqrt(np.nanmean((F_unb[overlap] - F_phi[overlap]) ** 2)))
           if overlap.any() else float("nan"))

    z = mids * system.a / (2 * PI)
    print(f"  barrier dF = {dF:.3f} kJ/mol = {dF*beta:.2f} kT "
          f"(split halves {dF_a*beta:.2f} / {dF_b*beta:.2f} kT)")
    print(f"  dU = {dU:.3f} kJ/mol = {dU*beta:.2f} kT")
    print(f"  -T dS = dF - dU = {mTdS:.3f} kJ/mol = {mTdS*beta:.2f} kT "
          f"({100*mTdS/dF if dF else float('nan'):.0f}% of the barrier)")
    print(f"  unbiased cross-check overlap: {int(overlap.sum())}/{N_BINS} bins, "
          f"RMS diff {rms:.3f} kJ/mol")

    tag = f"T{a.temperature:g}"
    np.savez_compressed(
        os.path.join(OUT, f"reference_{tag}.npz"),
        grid_phi=mids, z=z, F=F_phi, U=U_phi, TS=TS_phi, p=p_phi,
        F_unbiased=F_unb, unbiased_hist=hist_unb, window_hist=hist,
        a_pseudo=system.a, box=system.L, temperature=a.temperature, kT=kT,
        dF_barrier=dF, dU_barrier=dU, mTdS_barrier=mTdS,
        dF_split=[dF_a, dF_b],
        protocol=json.dumps(dict(n_windows=N_WINDOWS, kappa=a.kappa, n_rep=a.n_rep,
                                 n_steps=a.n_steps, burn_in=a.burn_in,
                                 sample_every=SAMPLE_EVERY, n_bins=N_BINS,
                                 seed=a.seed, unbiased_n=UNBIASED_N,
                                 unbiased_steps=a.unbiased_steps,
                                 params=dict(eps_go=params.eps_go,
                                             sigma_go=params.sigma_go,
                                             rc=params.rc, r0=params.r0_bond,
                                             k_bond=params.k_bond,
                                             dt=sim.dt))))
    print(f"  wrote {OUT}/reference_{tag}.npz")

    # ---- figure ----
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
    ax = axes[0]
    ax.plot(z, F_phi * beta, color=PALETTE["black"], lw=1.5, label=r"$F(z)$")
    ax.plot(z, U_phi * beta, color=PALETTE["blue"], lw=1.3, ls="--", label=r"$U(z)$")
    ax.plot(z, -TS_phi * beta, color=PALETTE["vermillion"], lw=1.3, ls="-.",
            label=r"$-TS(z)$")
    ax.plot(z, F_unb * beta, color=PALETTE["gray"], lw=1.0, ls=":",
            label="unbiased check")
    ax.set_xlabel("z (A)  [window at 0, cage centers at $\\pm$a/2]")
    ax.set_ylabel("kT")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"T = {a.temperature:g} K:  "
                 f"$\\Delta F^\\ddagger$={dF*beta:.1f} kT, "
                 f"$\\Delta U^\\ddagger$={dU*beta:.1f} kT, "
                 f"$-T\\Delta S^\\ddagger$={mTdS*beta:.1f} kT", fontsize=9)
    ax = axes[1]
    ax.plot(z, p_phi, color=PALETTE["black"], lw=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("z (A)")
    ax.set_ylabel(r"$p(z)$ (WHAM)")
    ax.set_title("equilibrium marginal", fontsize=9)
    save_figure(fig, os.path.join(OUT, f"fig_lta_reference_{tag}"))
    print(f"  wrote fig_lta_reference_{tag}")


if __name__ == "__main__":
    main()
