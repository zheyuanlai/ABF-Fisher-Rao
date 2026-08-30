#!/usr/bin/env python
"""Independent umbrella/WHAM reference for the ethane/ZIF-8 cell, plus the
U(xi) decomposition and -- the piece this stage exists for -- the reference
CONDITIONAL distribution of the hidden gate coordinate at the window.

No ABF and no Fisher-Rao anywhere in this file.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/run_zif8_reference.py --temperature 300
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
from zif8.core_zif8 import (KB, TWO_PI, ZIF8SimConfig, ZIF8System,   # noqa: E402
                            conditional_mean_periodic, engine_kwargs,
                            run_umbrella, wham_periodic)

PREREG = os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")
OUT = os.path.join(ROOT, "results/uniform_campaign/zif8/reference")


def barrier(xi, F, half_window=0.6, cage_half=1.5):
    """Window minus cage: the cage sits at |xi| = L/2 (the circle's far side)."""
    win = np.abs(xi) < half_window
    cage = np.abs(np.abs(xi) - np.abs(xi).max()) < cage_half
    return float(np.nanmean(F[win]) - np.nanmean(F[cage]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--windows", type=int, default=None)
    ap.add_argument("--n-rep", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None)
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    rr = pre["reference_rule"]
    tag = f"T{a.temperature:g}"
    W = a.windows or rr["windows"]
    n_rep = a.n_rep or rr["n_rep"]
    n_steps = a.steps or rr["n_steps"]
    os.makedirs(OUT, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ek = engine_kwargs(pre)
    if a.chunk:
        ek["chunk"] = a.chunk
    system = ZIF8System(a.temperature, device, root=ROOT, **ek)
    print(f"  engine: dtype {ek['dtype']}, force kernel "
          f"{ek['force_dtype'] or ek['dtype']}, chunk {ek['chunk']}")
    s = {k: v for k, v in pre["sampler"].items() if not k.startswith("_")}
    sim = ZIF8SimConfig(**s)
    pool = os.path.join(ROOT, f"cache/zif8/init_pool_{tag}.npz")
    kT = KB * a.temperature
    # kappa is set by a width in ANGSTROM, converted to the circular CV
    kappa = kT / (rr["kappa_width_A"] * system.k_phi) ** 2
    centers = -math.pi + (np.arange(W) + 0.5) * (TWO_PI / W)
    print(f"ZIF-8 reference {tag}: {W} windows (spacing "
          f"{TWO_PI/W/system.k_phi:.3f} A), kappa={kappa:.1f} kJ/mol/rad^2, "
          f"{n_rep} reps, {n_steps} steps, dt={sim.dt} ps "
          f"({n_steps*sim.dt:.0f} ps/window)", flush=True)

    t0 = time.time()
    phis, us, uhgs, gates, thetas = run_umbrella(
        system, sim, centers, kappa, n_steps, n_rep, rr["burn_in"],
        rr["sample_every"], rr["seed"], pool)
    beta = system.beta
    nb = sim.n_grid
    mids, F, p, hist = wham_periodic(phis, centers, kappa, beta, n_bins=nb)
    T2 = phis.shape[0] // 2
    _, Fa, _, _ = wham_periodic(phis[:T2], centers, kappa, beta, n_bins=nb)
    _, Fb, _, _ = wham_periodic(phis[T2:], centers, kappa, beta, n_bins=nb)
    xi = mids / system.k_phi
    dF = barrier(xi, F)
    dFa, dFb = barrier(xi, Fa), barrier(xi, Fb)

    _, U, ucnt = conditional_mean_periodic(phis, us, nb)
    _, Uhg, _ = conditional_mean_periodic(phis, uhgs, nb)
    ok = ucnt > rr["min_u_counts"]
    U = np.where(ok, U, np.nan); Uhg = np.where(ok, Uhg, np.nan)
    U = U - np.nanmean(U[np.abs(np.abs(xi) - np.abs(xi).max()) < 1.5])
    dU = barrier(xi, U)
    mTdS = dF - dU

    # --- the hidden-gate reference: p_ref(A_gate | |xi| < gate_band) --------
    band = np.abs(phis / system.k_phi) < sim.gate_band_A
    edges = np.linspace(sim.gate_lo, sim.gate_hi, sim.n_gate_bins + 1)
    gh, _ = np.histogram(gates[band], bins=edges)
    th_band = thetas[band]
    print(f"  gate samples inside |xi| < {sim.gate_band_A} A: {int(band.sum())}")
    assert int(band.sum()) > rr["min_gate_samples"], \
        f"too few gate samples at the window: {int(band.sum())}"

    # split-half of the gate conditional, so the reference's OWN convergence
    # is testable before it is used to judge either arm
    gh_a, _ = np.histogram(gates[:T2][band[:T2]], bins=edges)
    gh_b, _ = np.histogram(gates[T2:][band[T2:]], bins=edges)
    from zif8.core_zif8 import js_divergence
    js_half = float(js_divergence(gh_a.astype(float), gh_b.astype(float)))

    ov = [int(((hist[w] > 0) & (hist[w + 1] > 0)).sum()) for w in range(W - 1)]
    ov.append(int(((hist[-1] > 0) & (hist[0] > 0)).sum()))     # circular closure
    print(f"  dF = {dF:.2f} kJ/mol = {dF*beta:.2f} kT "
          f"(split-half {dFa*beta:.2f}/{dFb*beta:.2f} kT); "
          f"anchor paper (Krokidas FF, NPT 300 K): 24.2 +- 2.6 kJ/mol")
    print(f"  dU = {dU:.2f} kJ/mol = {dU*beta:.2f} kT;  -T dS = {mTdS:.2f} kJ/mol "
          f"= {mTdS*beta:.2f} kT  ({100*mTdS/dF if dF else float('nan'):.0f}% entropic)")
    print(f"  min adjacent-window overlap: {min(ov)} bins (circular)")
    print(f"  gate conditional split-half JS: {js_half:.5f}")
    acc = dict(
        split_half_dF_kT=float(abs(dFa - dFb) * beta),
        split_half_rms_kT=float(np.sqrt(np.nanmean(
            ((Fa - np.nanmean(Fa)) - (Fb - np.nanmean(Fb))) ** 2)) * beta),
        min_overlap_bins=int(min(ov)),
        gate_split_half_js=js_half)
    gates_pass = dict(
        split_half_barrier=acc["split_half_dF_kT"] < rr["accept_dF_kT"],
        split_half_profile=acc["split_half_rms_kT"] < rr["accept_rms_kT"],
        window_overlap=acc["min_overlap_bins"] >= rr["accept_overlap_bins"],
        gate_conditional=js_half < rr["accept_gate_js"])
    for k, v in gates_pass.items():
        print(f"  GATE {k:22s}: {'PASS' if v else 'FAIL'} "
              f"({list(acc.values())[list(gates_pass).index(k)]:.4f})")

    np.savez_compressed(
        os.path.join(OUT, f"reference_{tag}.npz"),
        grid=mids, xi_grid=xi, F=F, U=U, U_hostguest=Uhg, p=p, u_counts=ucnt,
        F_split=np.stack([Fa, Fb]), window_hist=hist, centers=centers,
        kappa=kappa, period=system.period, k_phi=system.k_phi,
        xi_A=system.xi_A, xi_B=system.xi_B, temperature=a.temperature, kT=kT,
        dF_barrier=dF, dU_barrier=dU, mTdS_barrier=mTdS,
        gate_hist_window=gh, gate_edges=edges,
        gate_hist_split=np.stack([gh_a, gh_b]),
        gate_theta_window=np.histogram(th_band, bins=60, range=(0, 90))[0],
        acceptance=json.dumps(dict(values=acc, gates=gates_pass)),
        protocol=json.dumps(dict(rr, windows=W, n_rep=n_rep, n_steps=n_steps,
                                 kappa=kappa, tag=tag, dt=sim.dt)))
    print(f"  wrote reference_{tag}.npz  [{(time.time()-t0)/60:.1f} min]")
    assert all(gates_pass.values()), "reference acceptance gates failed"


if __name__ == "__main__":
    main()
