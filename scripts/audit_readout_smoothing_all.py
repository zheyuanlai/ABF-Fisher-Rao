#!/usr/bin/env python
"""Bandwidth-defect screen for EVERY headline system of the uniform-FR campaign (no runs).

Deterministic smoothing-bias upper-bound diagnostic: apply each engine's OWN legacy
read-out kernel (same kernel form, boundary treatment and normalisation as the engine's
Nadaraya--Watson estimator) to that system's reference mean force, re-integrate, and
report the aligned-RMS bias in the system's endpoint units next to the MEASURED ABF
e_F(T).  share = det_bias / e_F(T); the bias-only upper bound on what a sharper read-out
could buy is  predicted MSE gain = 1 / (1 - share^2).

Calibration (three systems where the expensive sweep was actually run):
    ZIF-8 300 K   predicted 4.61x   measured 5.82x   (docs/INFORMATION_CLOCK_AUDIT.md)
    WCA Case IX   predicted 1.03x   measured 1.04x   (docs/WCA_BASELINE_AUDIT.md)
    LTA 80 K      predicted 1.04x   measured 0.96x   (docs/LTA_READOUT_SWEEP.md; variance
                                                      rises when the kernel is removed)
It is an UPPER BOUND, not an optimal-bandwidth predictor.

New systems here: entropic gateway, CHA (3 cells), R15 mid-beta (2 cells), alanine.
The three calibrated systems are re-read from audit_readout_smoothing.py's JSON so the
validated code is not touched.

    python scripts/audit_readout_smoothing_all.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
ROOT = os.path.join(SCRIPTS, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from audit_readout_smoothing import rough, OUT as CALIB_JSON   # noqa: E402

OUT = os.path.join(ROOT, "results/information_campaign/readout_smoothing_screen_all.json")
MEASURED_SWEEP_GAIN = {"zif8_T300": 5.82, "wca_caseIX": 1.04, "lta_T80": 0.96}


def gain(share):
    return float("inf") if share >= 1 else 1.0 / (1.0 - share ** 2)


def cumtrapz(mf, dz):
    return np.concatenate([[0.0], np.cumsum(0.5 * (mf[1:] + mf[:-1]) * dz)])


def aligned_rms(d, w=None):
    w = np.ones_like(d) if w is None else np.asarray(w, float)
    c = (d * w).sum() / w.sum()
    return float(math.sqrt((((d - c) ** 2) * w).sum() / w.sum()))


def nw_line_unpadded(mf, grid, bw):
    """(K @ F') / (K @ 1), unnormalised Gaussian K, NO padding -- cha.core_cha.mean_force_profile
    and alkanes.interval.mean_force_profile at infinite uniform counts."""
    K = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / bw) ** 2)
    return (K @ mf) / K.sum(1)


def conv_reflect_normalised(mf, bw, dx):
    """eb_abffr_core.gaussian_kernel + smooth: radius round(4 bw/dx), kernel / (sum*dx), reflect
    pad, valid conv; divided by smooth(1) = 1/dx  ->  a normalised reflect-padded convolution."""
    r = max(1, int(round(4.0 * bw / dx)))
    t = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (t * dx / bw) ** 2)
    k = k / k.sum()
    return np.convolve(np.pad(mf, r, mode="reflect"), k[::-1], mode="valid")


def row(h, h_per_bin, mf_true, mf_s, F_true, F_s, dz, mask, w=None):
    d = (F_s - F_true)[mask]
    R0 = rough(mf_true[mask], dz)
    return dict(h=float(h), h_per_bin=float(h_per_bin), roughness=rough(mf_s[mask], dz) / R0,
                barrier_err_pct=100 * ((F_s[mask].max() - F_s[mask].min())
                                       / (F_true[mask].max() - F_true[mask].min()) - 1),
                det_bias=aligned_rms(d, None if w is None else w[mask]))


def finish(name, rows, e_meas, unit, legacy_h, note=""):
    for r in rows:
        r["legacy"] = bool(abs(r["h"] - legacy_h) < 1e-12)
        r["share"] = r["det_bias"] / e_meas
        r["pred_mse_gain"] = gain(r["share"])
    print(f"\n{name}   (measured ABF e_F(T) = {e_meas:.4f} {unit}){('   ' + note) if note else ''}")
    print(f"  {'h':>8} {'h/bin':>6} {'rough':>6} {'barrier':>8} {'det.bias':>9} {'share':>6} {'gain':>6}")
    for r in rows:
        g = f"{r['pred_mse_gain']:6.2f}" if np.isfinite(r["pred_mse_gain"]) else "   inf"
        print(f"  {r['h']:8.4f} {r['h_per_bin']:6.2f} {r['roughness']:6.3f} {r['barrier_err_pct']:+7.2f}%"
              f" {r['det_bias']:9.4f} {r['share']:6.2f} {g}{'  <- legacy' if r['legacy'] else ''}")
    return dict(rows=rows, measured_eF_T=e_meas, unit=unit, note=note)


def main():
    out = {}
    calib = json.load(open(CALIB_JSON))

    # ---- entropic gateway (eb_abffr_core kernel; h = 0.07, dx = 0.02, eval [-1.5, 1.5]) ----
    z = np.load(os.path.join(ROOT, "results/uniform_campaign/gateway/raw.npz"), allow_pickle=True)
    abf = np.asarray(z["method"]) == "abf"
    cells = sorted({(float(s), float(r)) for s, r in zip(z["s"][abf], z["r_ratio"][abf])})
    for s, rr in cells:
        sel = abf & (z["s"] == s) & (z["r_ratio"] == rr)
        i = int(np.flatnonzero(sel)[0])
        x, Fp, F = (np.asarray(z[k][i], float) for k in ("x_grid", "Fp_ref", "F_ref"))
        dx = x[1] - x[0]
        mask = (x >= -1.5) & (x <= 1.5)
        e = float(np.median(z["final_l2_f"][sel]))
        cfg = json.loads(str(z["config_json"][i]))
        h_leg = float(cfg["h"])
        rows = []
        for h in (2 * h_leg, h_leg, h_leg / 2, h_leg / 4):
            mfs = conv_reflect_normalised(Fp, h, dx)
            Fs = cumtrapz(mfs, dx)
            rows.append(row(h, h / dx, Fp, mfs, cumtrapz(Fp, dx), Fs, dx, mask))
        out[f"gateway_s{s:g}_r{rr:g}"] = finish(
            f"Gateway s={s:g} r={rr:g} (h in x units, dx={dx:.3f}; eval [-1.5, 1.5]; {int(sel.sum())} abf rows)",
            rows, e, "energy", h_leg, "reflect-padded normalised conv (engine kernel)")

    # ---- CHA 8-ring, three cells (non-periodic NW, bw 0.15 A, min_count 20; mask [xi_A-1, xi_B+1]) ----
    for tag in ("ethene_450", "propene_450", "propene_600"):
        r = np.load(os.path.join(ROOT, f"results/uniform_campaign/cha/reference/reference_{tag}.npz"), allow_pickle=True)
        g, F = np.asarray(r["grid"], float), np.asarray(r["F"], float)
        dz = g[1] - g[0]
        mask = (g >= float(r["xi_A"]) - 1.0) & (g <= float(r["xi_B"]) + 1.0)
        Fp = np.gradient(F, g)
        e = float(np.median([float(x["final_abf"]) for x in
                             csv.DictReader(open(os.path.join(ROOT, f"results/uniform_campaign/cha/comparison_{tag}.csv")))]))
        rows = []
        for h in (0.30, 0.15, 0.075, 0.0375):
            mfs = nw_line_unpadded(Fp, g, h)
            rows.append(row(h, h / dz, Fp, mfs, cumtrapz(Fp, dz), cumtrapz(mfs, dz), dz, mask))
        out[f"cha_{tag}"] = finish(f"CHA {tag} (h in A, dz={dz:.4f}; kT={float(r['kT']):.3f} kJ/mol)",
                                   rows, e, "kJ/mol", 0.15)

    # ---- R15 mid-beta, pentane distance CV, two cells (non-periodic NW, bw 0.04, thermal mask <= 10) ----
    for f in sorted(glob.glob(os.path.join(ROOT, "results/uniform_campaign/r15_midbeta_methods/raw/production__dist__pentane__abf__*.npz"))):
        d = np.load(f, allow_pickle=True)
        spec = json.loads(str(d["spec_json"]))
        g, F, Fp = (np.asarray(d[k], float) for k in ("grid", "ref_F", "ref_Fprime"))
        dz = float(d["dz"])
        mask = (F - F.min()) <= float(spec["thermal_delta"])
        e = float(np.median([p["final_l2_F"] for p in json.loads(str(d["per_seed"]))]))
        h_leg = float(spec["dist_abf_bandwidth"])
        rows = []
        for h in (2 * h_leg, h_leg, h_leg / 2, h_leg / 4):
            mfs = nw_line_unpadded(Fp, g, h)
            rows.append(row(h, h / dz, Fp, mfs, cumtrapz(Fp, dz), cumtrapz(mfs, dz), dz, mask))
        out[f"r15_pentane_b{spec['beta']:g}"] = finish(
            f"R15 pentane beta={spec['beta']:g} (h in CV units, dz={dz:.5f}; thermal mask <= {spec['thermal_delta']:g})",
            rows, e, "energy", h_leg)

    # ---- alanine (phi,psi) torus, normalised separable wrapped-Gaussian NW, bw 0.08 rad ----
    # On the torus smoothing commutes with the gradient and the Poisson reconstruction of a smoothed
    # gradient field is the smoothed F, so det_bias = aligned RMS(K*F - F) exactly.  Equilibrium
    # weighting on mask8 (metrics_ala.build_masks).  NOTE: metrics_ala.smooth_reference uses the
    # UNNORMALISED kernel (row sum 3.10 per axis) -- that is a separate finding, reported below.
    from alanine import metrics_ala as M       # noqa: E402
    r = np.load(os.path.join(ROOT, "results/alanine/reference/reference.npz"), allow_pickle=True)
    F = np.asarray(r["F"], float)
    kT = float(json.loads(str(r["meta"]))["kT_kJ"])
    n = F.shape[0]
    dphi = 2 * math.pi / n
    masks = M.build_masks(F, kT)
    w = masks["weights"]["equilibrium"]
    fill = np.nanmax(np.where(np.isfinite(F), F, np.nan))
    Ff = np.where(np.isfinite(F), F, fill)
    phi = -math.pi + dphi * (np.arange(n) + 0.5)
    dcirc = np.abs(phi[:, None] - phi[None, :])
    dcirc = np.minimum(dcirc, 2 * math.pi - dcirc)
    rows_al = []
    for h in (0.16, 0.08, 0.04, 0.02):
        K = np.exp(-0.5 * (dcirc / h) ** 2)
        K = K / K.sum(1, keepdims=True)
        Fs = K @ Ff @ K.T
        # roughness on the torus: RMS |grad F| over mask8 (periodic gradient)
        def grad_rms(A):
            gx = (np.roll(A, -1, 0) - np.roll(A, 1, 0)) / (2 * dphi)
            gy = (np.roll(A, -1, 1) - np.roll(A, 1, 1)) / (2 * dphi)
            return float(np.sqrt(np.mean((gx ** 2 + gy ** 2)[masks["mask8"]])))
        rows_al.append(dict(h=h, h_per_bin=h / dphi, roughness=grad_rms(Fs) / grad_rms(Ff),
                            barrier_err_pct=100 * ((Fs[masks["mask8"]].max() - Fs[masks["mask8"]].min())
                                                   / (Ff[masks["mask8"]].max() - Ff[masks["mask8"]].min()) - 1),
                            det_bias=M.aligned_l2(Fs, F, w)))
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results/uniform_campaign/alanine/analysis/paired_seed_metrics_N2048_uniform.csv"))))
    e = float(np.median([float(x["final_eF_equilibrium"]) for x in rows if x["method"] == "abf"]))
    e_km = float(np.median([float(x["final_eF_km_equilibrium"]) for x in rows if x["method"] == "abf"]))
    Fkm = M.smooth_reference(F, 0.08, n)
    out["alanine_N2048"] = finish("Alanine (phi,psi) 300 K (h in rad, 97x97; equilibrium-weighted mask8)",
                                  rows_al, e, "kJ/mol", 0.08,
                                  "measured = UN-matched final_eF_equilibrium; see km note")
    out["alanine_N2048"]["km_artifact"] = dict(
        measured_final_eF_km_equilibrium_abf=e_km,
        deterministic_aligned_l2_unnormalised_KF_vs_F=M.aligned_l2(Fkm, F, w),
        kernel_row_sum_per_axis=float(np.exp(-0.5 * (dcirc / 0.08) ** 2).sum(1)[0]),
        note="metrics_ala.smooth_reference smooths with the UNNORMALISED wrapped Gaussian, so the "
             "'kernel-matched' reference is ~9.6x a smoothed F_ref; the closed study's primary km "
             "endpoint (25.7 kJ/mol) is 99.7% this arm-independent constant")
    print(f"  km artifact: measured km e_F {e_km:.3f} vs deterministic |K_unnorm F - F| "
          f"{out['alanine_N2048']['km_artifact']['deterministic_aligned_l2_unnormalised_KF_vs_F']:.3f} kJ/mol "
          f"(kernel row sum {out['alanine_N2048']['km_artifact']['kernel_row_sum_per_axis']:.3f} per axis)")

    # ---- unified legacy table incl. the three calibrated systems ----
    print("\n==== legacy read-out, every headline system: share and bias-only predicted MSE gain ====")
    print(f"  {'system':>26} {'h/bin':>6} {'rough':>6} {'share':>6} {'pred':>6} {'measured':>9}")
    summary = {}
    def add(name, r, measured=None):
        summary[name] = dict(h_per_bin=r["h_per_bin"], roughness=r["roughness"], share=r["share"],
                             pred_mse_gain=gain(r["share"]), measured_sweep_gain=measured)
        m = f"{measured:9.2f}" if measured else f"{'--':>9}"
        print(f"  {name:>26} {r['h_per_bin']:6.2f} {r['roughness']:6.3f} {r['share']:6.2f} {gain(r['share']):6.2f} {m}")
    zr = [r for r in calib["zif8_T300"] if r["legacy"]][0]
    add("zif8_T300 (0.20 A)", zr, MEASURED_SWEEP_GAIN["zif8_T300"])
    add("wca_caseIX (0.025)", [r for r in calib["wca_caseIX"]["rows"] if r["legacy"]][0], MEASURED_SWEEP_GAIN["wca_caseIX"])
    for T in (80, 150, 225, 300):
        add(f"lta_T{T} (0.05 rad)", [r for r in calib[f"lta_T{T}"]["rows"] if r["legacy"]][0],
            MEASURED_SWEEP_GAIN.get(f"lta_T{T}"))
    for k, v in out.items():
        add(k, [r for r in v["rows"] if r["legacy"]][0])
    out["_summary_legacy"] = summary
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
