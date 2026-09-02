#!/usr/bin/env python
"""Analyzer for the gateway ABF-only online-bandwidth audit -- written BEFORE the data.

Prereg: configs/information_campaign/gateway_baseline_audit_prereg.json.

Read-outs are recomputed EXACTLY from the saved raw accumulators with the engine's own kernel
(eb_abffr_core.gaussian_kernel + smooth, mirrored in numpy): F'_h = s_h(Sf) / (s_h(C) + min_count),
raw = Sf / C.  Stage 1 (legacy arm only): read-out ladder, 2% plateau, h_read* = legacy if on the
plateau else the largest plateau point.  Stage 2: every arm at h_read*, per-row paired relative
change vs the legacy arm, median + 10k bootstrap (seed 20260829), failure-mode instrumentation,
outcome A/B/C/D per the prereg.

    python scripts/analyze_gateway_bandwidth_audit.py [--raw-dir DIR]
"""
from __future__ import annotations

import argparse
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
from analyze_uniform_lta import BOOT_SEED, boot_median           # noqa: E402  (10k resamples)
from eb_abffr_core import EVAL_LO, EVAL_HI, EPS                  # noqa: E402

PREREG = os.path.join(ROOT, "configs/information_campaign/gateway_baseline_audit_prereg.json")
DEFAULT_RAW = os.path.join(ROOT, "results/information_campaign/gateway_baseline_audit")
PLATEAU_TOL = 0.02
LEGACY_H = 0.07
LADDER = (0.07, 0.035, 0.0175, 0.00875, 0.0)      # 0.0 = raw bins


def eb_kernel(h, dx):
    """numpy mirror of eb_abffr_core.gaussian_kernel: radius round(4h/dx), normalised by sum*dx."""
    r = max(1, int(round(4.0 * h / dx)))
    t = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (t * dx / h) ** 2)
    return k / (k.sum() * dx), r


def eb_smooth(v, h, dx):
    """numpy mirror of eb_abffr_core.smooth: reflect pad, valid conv, NO dx scaling. v (..., G)."""
    G = v.shape[-1]
    k, r = eb_kernel(h, dx)
    pad = min(r, G - 1)
    ks = k[r - pad: len(k) - (r - pad)]
    vp = np.pad(v, [(0, 0)] * (v.ndim - 1) + [(pad, pad)], mode="reflect")
    flat = vp.reshape(-1, vp.shape[-1])
    out = np.stack([np.convolve(row, ks, mode="valid") for row in flat])
    return out.reshape(v.shape[:-1] + (G,))


def mean_force_at(Sf, C, h, dx, min_count):
    if h <= 0:
        return np.where(C > 0, Sf / np.maximum(C, 1.0), 0.0)
    return eb_smooth(Sf, h, dx) / (eb_smooth(C, h, dx) + min_count + EPS)


def cumtrapz(y, dx):
    seg = 0.5 * (y[..., 1:] + y[..., :-1]) * dx
    out = np.zeros_like(y)
    out[..., 1:] = np.cumsum(seg, axis=-1)
    return out


def e_f(Fp_t, F_ref, dx, mask):
    """engine convention: F = cumtrapz(F'), both F and F_ref centred on the eval window, RMS over it."""
    F = cumtrapz(Fp_t, dx)
    F = F - F[..., mask].mean(-1, keepdims=True)
    R = F_ref - F_ref[..., mask].mean(-1, keepdims=True)
    d = (F - R[..., None, :] if F.ndim == R.ndim + 1 else F - R)[..., mask]
    return np.sqrt((d * d).mean(-1))


def load_arm(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def label(h):
    return "raw" if h <= 0 else f"{h:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=DEFAULT_RAW)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    arms = {}
    for f in sorted(glob.glob(os.path.join(a.raw_dir, "raw_hb*.npz"))):
        d = load_arm(f)
        arms[float(d["h_bias"])] = d
    hs = sorted(arms, reverse=True)
    assert hs and hs[0] == LEGACY_H, f"legacy arm missing: {hs}"
    leg = arms[LEGACY_H]
    x = np.asarray(leg["x_grid"][0], float)
    dx = float(x[1] - x[0])
    mask = (x >= EVAL_LO) & (x <= EVAL_HI)
    t = np.asarray(leg["t"][0], float)
    cfg0 = json.loads(str(leg["config_json"][0]))
    min_count = float(cfg0["min_count"])
    keys = [(str(i), int(s)) for i, s in zip(leg["init"], leg["seed"])]
    for h in hs[1:]:
        assert [(str(i), int(s)) for i, s in zip(arms[h]["init"], arms[h]["seed"])] == keys, "rows differ across arms"
    print(f"arms h_bias {hs}; {len(keys)} paired rows per arm ({sorted(set(k[0] for k in keys))})")

    # ---- offline read-outs for every arm (exact) + self-check against the engine ----
    ro = {}
    for h in hs:
        d = arms[h]
        Sf, C, F_ref = np.asarray(d["Sf_t"], float), np.asarray(d["C_t"], float), np.asarray(d["F_ref"], float)
        ro[h] = {label(hr): e_f(mean_force_at(Sf, C, hr, dx, min_count), F_ref, dx, mask) for hr in LADDER}
        # the production read-out recomputed from Sf, C must equal the engine's own l2_f_t and Fp_prof_t
        own = mean_force_at(Sf, C, h, dx, min_count)
        dev_fp = float(np.abs(own - np.asarray(d["Fp_prof_t"], float)).max())
        dev_e = float(np.abs(e_f(own, F_ref, dx, mask) - np.asarray(d["l2_f_t"], float)).max())
        print(f"  self-check h_bias={h:g}: offline F' vs saved Fp_prof_t max|dev| {dev_fp:.2e}; e_F(t) vs saved l2_f_t {dev_e:.2e}")
        assert dev_fp < 1e-9 and dev_e < 1e-9

    # ---- Stage 1: legacy arm only ----
    print("\nStage 1 (legacy arm only): read-out ladder at fixed legacy dynamics")
    s1 = {}
    print(f"  {'read-out':>8} {'e_F(T) med':>11} {'row sd':>8} {'I_F med':>8}   per-init medians")
    inits = sorted(set(k[0] for k in keys))
    for hr in LADDER:
        lab = label(hr)
        fin = ro[LEGACY_H][lab][:, -1]
        I = np.trapezoid(ro[LEGACY_H][lab], t, axis=1)
        per_init = {ini: float(np.median([fin[j] for j, k in enumerate(keys) if k[0] == ini])) for ini in inits}
        s1[lab] = dict(eF_T_median=float(np.median(fin)), eF_T_row_sd=float(np.std(fin)), IF_median=float(np.median(I)),
                       per_init=per_init)
        print(f"  {lab:>8} {s1[lab]['eF_T_median']:11.5f} {s1[lab]['eF_T_row_sd']:8.5f} {s1[lab]['IF_median']:8.4f}   "
              + "  ".join(f"{k} {v:.5f}" for k, v in per_init.items()) + ("  <- legacy" if hr == LEGACY_H else ""))
    emin = min(v["eF_T_median"] for v in s1.values())
    plateau = [label(hr) for hr in LADDER if s1[label(hr)]["eF_T_median"] <= (1 + PLATEAU_TOL) * emin]
    h_star_lab = label(LEGACY_H) if label(LEGACY_H) in plateau else plateau[0]
    h_star = 0.0 if h_star_lab == "raw" else float(h_star_lab)
    print(f"  plateau (within {100*PLATEAU_TOL:.0f}% of {emin:.5f}): {plateau} -> h_read* = {h_star_lab}"
          f" ({'legacy ON plateau, kept' if h_star == LEGACY_H else 'legacy OFF plateau'})")

    # ---- Stage 2: all arms at h_read* ----
    print(f"\nStage 2: all arms scored at h_read* = {h_star_lab}; paired vs legacy h_bias {LEGACY_H:g}")
    fin = {h: ro[h][h_star_lab][:, -1] for h in hs}
    I = {h: np.trapezoid(ro[h][h_star_lab], t, axis=1) for h in hs}
    Fp_ref = np.asarray(leg["Fp_ref"], float)
    r_ref = np.median(np.sqrt(np.mean(np.gradient(Fp_ref[:, mask], dx, axis=1) ** 2, axis=1)))
    s2 = {}
    print(f"  {'h_bias':>7} {'e_F(T) med':>11} {'d e_F(T)':>9} {'CI95':>20} {'wins':>6} {'d I_F':>8} {'CI95':>20} {'rough':>6} {'max|F|':>7}")
    for k, h in enumerate(hs):
        d_fin = 100 * (fin[h] - fin[LEGACY_H]) / fin[LEGACY_H]
        d_int = 100 * (I[h] - I[LEGACY_H]) / I[LEGACY_H]
        ci_f = list(boot_median(d_fin, BOOT_SEED + 10 * k)) if h != LEGACY_H else [0.0, 0.0]
        ci_i = list(boot_median(d_int, BOOT_SEED + 10 * k + 1)) if h != LEGACY_H else [0.0, 0.0]
        online = np.asarray(arms[h]["Fp_prof_t"], float)[:, -1, :]
        rough = float(np.median(np.sqrt(np.mean(np.gradient(online[:, mask], dx, axis=1) ** 2, axis=1))) / r_ref)
        per_init = {ini: float(np.median([d_fin[j] for j, kk in enumerate(keys) if kk[0] == ini])) for ini in inits}
        s2[f"{h:g}"] = dict(h_bias=h, eF_T_median=float(np.median(fin[h])), d_fin=float(np.median(d_fin)), d_fin_ci=ci_f,
                            wins=int((d_fin < 0).sum()), n=len(keys), d_int=float(np.median(d_int)), d_int_ci=ci_i,
                            roughness=rough, max_abs_F=float(np.abs(online[:, mask]).max()), d_fin_per_init=per_init)
        print(f"  {h:7g} {np.median(fin[h]):11.5f} {np.median(d_fin):+9.2f} [{ci_f[0]:+8.2f},{ci_f[1]:+8.2f}]"
              f" {int((d_fin < 0).sum()):3d}/{len(keys)} {np.median(d_int):+8.2f} [{ci_i[0]:+8.2f},{ci_i[1]:+8.2f}]"
              f" {rough:6.3f} {np.abs(online[:, mask]).max():7.2f}   per-init " + "  ".join(f"{a_}: {b_:+.2f}" for a_, b_ in per_init.items()))

    # ---- outcome ----
    others = [f"{h:g}" for h in hs[1:]]
    resolved_better = [m for m in others if s2[m]["d_fin_ci"][1] < 0]
    resolved_worse = [m for m in others if s2[m]["d_fin_ci"][0] > 0]
    first = f"{hs[1]:g}" if len(hs) > 1 else None
    if resolved_worse:
        outcome = "C_online_hurts"
        h_bias_corr = LEGACY_H
    elif first in resolved_better:
        outcome = "B_online_helps"
        # smallest RESOLVED step: 1/4 only if it is resolved against 1/2 as well
        h_bias_corr = hs[1]
        if len(hs) > 2:
            d = 100 * (fin[hs[2]] - fin[hs[1]]) / fin[hs[1]]
            lo, hi = boot_median(d, BOOT_SEED + 99)
            print(f"  1/4 vs 1/2 at h_read*: {np.median(d):+.2f}% [{lo:+.2f}, {hi:+.2f}] -> {'resolved' if hi < 0 else 'not resolved'}")
            if hi < 0:
                h_bias_corr = hs[2]
    elif h_star != LEGACY_H:
        outcome = "A_readout_only"
        h_bias_corr = LEGACY_H
    else:
        outcome = "D_nothing_moves"
        h_bias_corr = LEGACY_H
    print(f"\n  OUTCOME: {outcome}   (resolved better: {resolved_better}; resolved worse: {resolved_worse})")
    print(f"  corrected baseline for step 2: h_bias = {h_bias_corr:g}, h_read* = {h_star_lab}"
          f"; rate rule: {'gamma 1.5 inherited' if h_bias_corr == LEGACY_H else 'RE-EARN by the pre-specified safety ladder'}")
    if len(keys) != 2 * pre["n_seeds"]:
        print(f"  NOTE: {len(keys)} rows of {2 * pre['n_seeds']} preregistered -- not confirmatory")

    out_json = a.out_json or os.path.join(a.raw_dir, "analysis.json")
    with open(out_json, "w") as fh:
        json.dump(dict(prereg=os.path.relpath(PREREG, ROOT), raw_dir=os.path.relpath(a.raw_dir, ROOT), n_rows=len(keys),
                       stage1=s1, plateau=plateau, h_read_star=h_star, h_read_star_label=h_star_lab, stage2=s2,
                       outcome=outcome, h_bias_corrected=h_bias_corr), fh, indent=2)
    print(f"  wrote {os.path.relpath(out_json, ROOT)}")


if __name__ == "__main__":
    main()
