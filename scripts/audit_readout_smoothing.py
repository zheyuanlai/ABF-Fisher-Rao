#!/usr/bin/env python
"""Deterministic read-out over-smoothing audit, one number per system (no runs).

Apply each engine's legacy read-out kernel to that system's OWN reference mean
force and report: roughness ratio RMS(dF'/dz)_smoothed / RMS(dF'/dz)_true (the
metric analyze_hbias_matrix.py used on ZIF-8's live bias force), the barrier
error of the re-integrated smoothed profile, and the aligned-RMS smoothing bias
of the free energy in ENDPOINT units next to the measured ABF e_F(T).  The last
ratio (deterministic bias / measured error) says how much of the baseline's
endpoint error a sharper READ-OUT could remove at most.

Calibration: ZIF-8's legacy 0.20 A gives roughness 0.912 here vs 0.93 measured on
the live force, and the bias share tracks the published offline sweep.

    python scripts/audit_readout_smoothing.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results/information_campaign/readout_smoothing_audit.json")
TWO_PI = 2.0 * math.pi
# ZIF-8 offline sweep at h_bias = 0.20 A, 8 seeds x 150 ps (docs/INFORMATION_CLOCK_AUDIT.md)
ZIF8_MEASURED = {0.40: 0.8373, 0.30: 0.5369, 0.20: 0.3018, 0.15: 0.2186, 0.10: 0.1620,
                 0.07: 0.1374, 0.05: 0.1266, 0.03: 0.1251}


def rough(mf, dx):
    return float(np.sqrt(np.mean(np.gradient(mf, dx, edge_order=2) ** 2)))


def periodic_smooth(Fp, grid, bw):
    d = np.abs(grid[:, None] - grid[None, :])
    d = np.minimum(d, TWO_PI - d)
    K = np.exp(-0.5 * (d / bw) ** 2)
    return (K @ Fp) / K.sum(1)


def periodic_integrate(mf, dphi):
    mf0 = mf - mf.mean()
    F = np.cumsum(0.5 * (mf0 + np.roll(mf0, -1)) * dphi)
    F = np.roll(F, 1)
    F[0] = 0.0
    return F


def periodic_rows(F, grid, bws, legacy, measured=None):
    dphi = grid[1] - grid[0]
    Fp = np.gradient(F, dphi, edge_order=2)
    R0, bar0 = rough(Fp, dphi), F.max() - F.min()
    rows = []
    for bw in bws:
        mfs = periodic_smooth(Fp, grid, bw)
        Fs = periodic_integrate(mfs, dphi)
        dd = Fs - F
        dd -= dd.mean()
        rows.append(dict(h=bw, h_per_bin=bw / dphi, roughness=rough(mfs, dphi) / R0,
                         barrier_err_pct=100 * ((Fs.max() - Fs.min()) / bar0 - 1),
                         det_bias=float(np.sqrt((dd ** 2).mean())), legacy=bool(abs(bw - legacy) < 1e-12)))
    return rows


def line_smooth(y, bw_bins, sigma_bins, pad=40):
    yp = np.pad(y, pad, mode="edge")
    x = np.arange(len(yp))
    if bw_bins > 0:
        K = np.exp(-0.5 * ((x[:, None] - x[None, :]) / bw_bins) ** 2)
        yp = (K @ yp) / K.sum(1)
    if sigma_bins > 0:                     # wca_abffr_core.smooth_profile_torch (replicate pad)
        rad = max(1, int(math.ceil(4.0 * sigma_bins)))
        k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma_bins) ** 2)
        k /= k.sum()
        yp = np.convolve(np.pad(yp, rad, mode="edge"), k, mode="valid")
    return yp[pad:-pad]


def median_csv(path, col):
    return float(np.median([float(r[col]) for r in csv.DictReader(open(path))]))


def show(name, rows, e_meas, unit):
    print(f"\n{name}   (measured ABF e_F(T) = {e_meas:.4f} {unit})")
    print(f"  {'h':>8} {'h/bin':>6} {'rough':>6} {'barrier':>8} {'det.bias':>9} {'share':>6}")
    for r in rows:
        tag = "  <- legacy" if r["legacy"] else ""
        print(f"  {r['h']:8.4f} {r['h_per_bin']:6.2f} {r['roughness']:6.3f} {r['barrier_err_pct']:+7.2f}%"
              f" {r['det_bias']:9.4f} {r['det_bias']/e_meas:6.2f}{tag}")


def main():
    out = {}
    # ---- ZIF-8 300 K (calibration) ----
    z = np.load(os.path.join(ROOT, "results/uniform_campaign/zif8/reference/reference_T300.npz"))
    kphi = float(z["k_phi"])
    rows = periodic_rows(z["F"], z["grid"], [a * kphi for a in ZIF8_MEASURED], 0.20 * kphi)
    for r, (hA, e) in zip(rows, ZIF8_MEASURED.items()):
        r["h_A"], r["measured_eF"], r["share"] = hA, e, r["det_bias"] / e
    print("ZIF-8 300 K: h in rad (A->rad x%.4f); measured e_F per h from the offline sweep" % kphi)
    print(f"  {'h(A)':>6} {'h/bin':>6} {'rough':>6} {'barrier':>8} {'det.bias':>9} {'measured':>9} {'share':>6}")
    for r in rows:
        print(f"  {r['h_A']:6.2f} {r['h_per_bin']:6.2f} {r['roughness']:6.3f} {r['barrier_err_pct']:+7.2f}%"
              f" {r['det_bias']:9.4f} {r['measured_eF']:9.4f} {r['share']:6.2f}{'  <- legacy' if r['legacy'] else ''}")
    out["zif8_T300"] = rows

    # ---- LTA per T ----
    for T in (80, 150, 225, 300):
        r = np.load(os.path.join(ROOT, f"results/uniform_campaign/lta/reference/reference_T{T}.npz"))
        rows = periodic_rows(r["F"], r["grid_phi"], (0.10, 0.075, 0.05, 0.035, 0.025, 0.0175, 0.0125), 0.05)
        e = median_csv(os.path.join(ROOT, f"results/uniform_campaign/lta/comparison_T{T}.csv"), "final_abf")
        for row in rows:
            row["share"] = row["det_bias"] / e
        show(f"LTA {T} K (h in rad)", rows, e, "kJ/mol")
        out[f"lta_T{T}"] = dict(rows=rows, measured_eF_T=e)

    # ---- WCA Case IX ----
    files = sorted(glob.glob(os.path.join(
        ROOT, "results/uniform_campaign/wca/uniform/raw/uniform__abf__*.npz")))
    e = float(np.median([float(np.load(f, allow_pickle=True)["l2_f"]) for f in files]))
    d = np.load(files[0], allow_pickle=True)
    g, mf = d["grid"], d["ref_mean_force"].astype(float)
    dz = g[1] - g[0]
    mask = (g >= -0.1) & (g <= 1.1)
    F0 = np.concatenate([[0], np.cumsum(0.5 * (mf[1:] + mf[:-1]) * dz)])
    R0, bar0 = rough(mf[mask], dz), F0[mask].max() - F0[mask].min()
    rows = []
    for bw, sig in ((0.05, 0.5), (0.025, 0.5), (0.025, 0.0), (0.0125, 0.5), (0.0125, 0.0),
                    (0.00625, 0.0), (0.0, 0.5), (0.0, 0.0)):
        ms = line_smooth(mf, bw / dz, sig)
        Fs = np.concatenate([[0], np.cumsum(0.5 * (ms[1:] + ms[:-1]) * dz)])
        dd = (Fs - F0)[mask]
        dd -= dd.mean()
        rows.append(dict(h=bw, smooth_sigma_bins=sig, h_per_bin=bw / dz, roughness=rough(ms[mask], dz) / R0,
                         barrier_err_pct=100 * ((Fs[mask].max() - Fs[mask].min()) / bar0 - 1),
                         det_bias=float(np.sqrt((dd ** 2).mean())), legacy=bool(bw == 0.025 and sig == 0.5)))
        rows[-1]["share"] = rows[-1]["det_bias"] / e
    print(f"\nWCA Case IX (h in CV units, dz={dz:.5f}; eval mask [-0.1, 1.1])   (measured ABF final l2_f = {e:.4f})")
    print(f"  {'h':>8} {'sig':>4} {'h/bin':>6} {'rough':>6} {'barrier':>8} {'det.bias':>9} {'share':>6}")
    for r in rows:
        print(f"  {r['h']:8.5f} {r['smooth_sigma_bins']:4.1f} {r['h_per_bin']:6.2f} {r['roughness']:6.3f}"
              f" {r['barrier_err_pct']:+7.2f}% {r['det_bias']:9.4f} {r['share']:6.2f}{'  <- legacy' if r['legacy'] else ''}")
    out["wca_caseIX"] = dict(rows=rows, measured_l2_f=e)

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
