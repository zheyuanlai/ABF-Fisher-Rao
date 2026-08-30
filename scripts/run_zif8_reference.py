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


def barrier(xi, F, half_window, cage_half):
    """Window minus cage: the cage sits at |xi| = L/2 (the circle's far side)."""
    win = np.abs(xi) < half_window
    cage = np.abs(np.abs(xi) - np.abs(xi).max()) < cage_half
    if not (np.isfinite(F[win]).any() and np.isfinite(F[cage]).any()):
        return float("nan")
    return float(np.nanmean(F[win]) - np.nanmean(F[cage]))


def barrier_peak(xi, F):
    """Peak minus minimum -- the definition the anchor paper's 24.2 kJ/mol uses.
    The window-mean-minus-cage-mean above is systematically SMALLER (5% on a
    smooth test barrier, more as the barrier sharpens), so quoting one against
    the other would be comparing two different quantities."""
    if not np.isfinite(F).any():
        return float("nan")
    return float(np.nanmax(F) - np.nanmin(F))


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
    # run_umbrella sorts its centres internally; sort here too so the two
    # never disagree about which bias belongs to which window
    centers = np.sort(-math.pi + (np.arange(W) + 0.5) * (TWO_PI / W))
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
    empty = hist.sum(axis=0) == 0
    if empty.any():
        F = np.where(empty, np.nan, F)      # -log(1e-300)/beta = +1718 kJ/mol
    print(f"  unsampled WHAM bins: {int(empty.sum())} of {nb}")
    T2 = phis.shape[0] // 2
    _, Fa, _, _ = wham_periodic(phis[:T2], centers, kappa, beta, n_bins=nb)
    _, Fb, _, _ = wham_periodic(phis[T2:], centers, kappa, beta, n_bins=nb)
    xi = mids / system.k_phi
    hw, ch = rr["barrier_half_window_A"], rr["barrier_cage_half_A"]
    dF = barrier(xi, F, hw, ch)
    dFa, dFb = barrier(xi, Fa, hw, ch), barrier(xi, Fb, hw, ch)
    dF_peak = barrier_peak(xi, F)

    _, U, ucnt = conditional_mean_periodic(phis, us, nb)
    _, Uhg, _ = conditional_mean_periodic(phis, uhgs, nb)
    ok = ucnt > rr["min_u_counts"]
    U = np.where(ok, U, np.nan); Uhg = np.where(ok, Uhg, np.nan)
    U = U - np.nanmean(U[np.abs(np.abs(xi) - np.abs(xi).max()) < ch])
    dU = barrier(xi, U, hw, ch)
    mTdS = dF - dU

    # --- the hidden-gate reference: p_ref(A_gate | xi sub-bin of the band) --
    # RESOLVED IN xi.  An unresolved p(A_gate | |xi| < band) is a mixture whose
    # weights are each ensemble's own p(xi | band) -- and FR deliberately
    # changes that marginal, so an unresolved comparison cannot tell a real
    # conditional displacement from a marginal reshuffle.
    xi_s = phis / system.k_phi
    band = np.abs(xi_s) < sim.gate_band_A
    edges = np.linspace(sim.gate_lo, sim.gate_hi, sim.n_gate_bins + 1)
    xedges = np.linspace(-sim.gate_band_A, sim.gate_band_A, sim.n_gate_xi + 1)

    def gate_2d(mask):
        h, _, _ = np.histogram2d(xi_s[mask], gates[mask], bins=[xedges, edges])
        return h
    gh_xa = gate_2d(band)
    gh = gh_xa.sum(axis=0)
    th_band = thetas[band]
    per_cell = gh_xa.sum(axis=1)
    print(f"  gate samples inside |xi| < {sim.gate_band_A} A: {int(band.sum())} "
          f"({int(per_cell.min())}-{int(per_cell.max())} per xi sub-bin); "
          f"A_gate out of [{sim.gate_lo}, {sim.gate_hi}]: "
          f"{int(band.sum() - gh_xa.sum())}")
    assert int(band.sum()) > rr["min_gate_samples"], \
        f"too few gate samples at the window: {int(band.sum())}"
    assert per_cell.min() > rr["min_gate_cell_samples"], \
        f"a xi sub-bin of the gate band is under-sampled: {per_cell.astype(int)}"
    assert band.sum() - gh_xa.sum() < 0.01 * band.sum(), \
        "more than 1% of A_gate samples fall outside the histogram range"

    # split-half of the gate conditional, so the reference's OWN convergence
    # is testable before it is used to judge either arm -- and split-half on
    # the SAME conditional statistic the screen and analysis will use
    from zif8.core_zif8 import js_divergence
    ha, hb = gate_2d(band & (np.arange(len(phis))[:, None, None] < T2)), \
        gate_2d(band & (np.arange(len(phis))[:, None, None] >= T2))
    okc = (ha.sum(1) >= rr["min_gate_cell_samples"]) & \
          (hb.sum(1) >= rr["min_gate_cell_samples"])
    js_half = float(np.mean(js_divergence(ha[okc], hb[okc])))
    gh_a, gh_b = ha.sum(0), hb.sum(0)

    ov = [int(((hist[w] > 0) & (hist[w + 1] > 0)).sum()) for w in range(W - 1)]
    ov.append(int(((hist[-1] > 0) & (hist[0] > 0)).sum()))     # circular closure
    print(f"  dF(peak-min) = {dF_peak:.2f} kJ/mol = {dF_peak*beta:.2f} kT "
          f"<- the definition comparable to the anchor paper's 24.2 +- 2.6")
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

    accepted = all(gates_pass.values())
    assert accepted or os.environ.get("ZIF8_KEEP_REJECTED_REFERENCE"), \
        f"reference acceptance gates failed: {gates_pass}"
    np.savez_compressed(
        os.path.join(OUT, f"reference_{tag}.npz"),
        accepted=accepted, empty_bins=int(empty.sum()), dF_peak=dF_peak,
        gate_hist_window_xi=gh_xa, gate_xi_edges=xedges,
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


if __name__ == "__main__":
    main()
