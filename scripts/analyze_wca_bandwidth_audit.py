#!/usr/bin/env python
"""Analyzer for the WCA Case IX ABF-only online-bandwidth audit -- written BEFORE the data.

Prereg: configs/information_campaign/wca_baseline_audit_prereg.json.

Stage 1 (legacy arm ONLY): read-out ladder {0.025 = production profile, bank 0.0125,
bank 0.00625, raw bins + 0.5-bin smoothing, raw bins}; plateau = points within 2 % of
the ladder-minimum median final e_F; h_read* = legacy if on the plateau else the
largest plateau point.  Stage 2: every arm scored at h_read*; per-seed relative change
vs the legacy arm; median + 10k bootstrap (seed 20260829).  Failure-mode instrumentation
on each arm's own online profile.  Outcome A/B/C/D per the prereg.

    python scripts/analyze_wca_bandwidth_audit.py [--raw-dir DIR] [--stage NAME]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from analyze_uniform_lta import BOOT_SEED, boot_median   # noqa: E402  (10k resamples)

ROOT = os.path.join(SCRIPTS, "..")
PREREG = os.path.join(ROOT, "configs/information_campaign/wca_baseline_audit_prereg.json")
DEFAULT_RAW = os.path.join(ROOT, "results/information_campaign/wca_baseline_audit/bandwidth_audit/raw")
Z_LO, Z_HI = -0.1, 1.1
CLIP = 40.0
PLATEAU_TOL = 0.02
LEGACY_KEY = "0.025"


def smooth_line(y, sigma_bins):
    """numpy mirror of wca_abffr_core.smooth_profile_torch (replicate padding)."""
    if sigma_bins <= 0:
        return np.asarray(y, dtype=float).copy()
    rad = max(1, int(math.ceil(4.0 * sigma_bins)))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(y, rad, mode="edge"), k, mode="valid")


def pmf_from_mf(mf, grid):
    """cumulative trapezoid along the last axis (engine convention: pmf = int F')."""
    mf = np.asarray(mf, dtype=float)
    inc = 0.5 * (mf[..., 1:] + mf[..., :-1]) * np.diff(grid)
    return np.concatenate([np.zeros(mf.shape[:-1] + (1,)), np.cumsum(inc, axis=-1)], axis=-1)


def e_f(pmf, ref_F, grid, mask):
    """aligned RMS over the eval window, per leading index (== core l2_f)."""
    d = pmf[..., mask] - ref_F[mask]
    d = d - d.mean(axis=-1, keepdims=True)
    g = grid[mask]
    return np.sqrt(np.trapezoid(d * d, g, axis=-1) / (g[-1] - g[0]))


def readouts(run, grid, mask, ref_F, sigma):
    """dict label -> e_F(t) series for every read-out available in this run."""
    out = {}
    out[LEGACY_KEY] = e_f(pmf_from_mf(run["mean_force_t"], grid), ref_F, grid, mask)
    for k in run:
        m = re.match(r"readout_mean_force_t__h(.+)$", k)
        if m:
            out[m.group(1)] = e_f(pmf_from_mf(run[k], grid), ref_F, grid, mask)
    if "raw_fsum_t" in run:
        fs, cs = run["raw_fsum_t"], run["raw_csum_t"]
        mf_raw = np.where(cs > 0, fs / np.maximum(cs, 1.0), 0.0)
        out["raw+sigma"] = e_f(pmf_from_mf(np.stack([smooth_line(r, sigma) for r in mf_raw]), grid), ref_F, grid, mask)
        out["raw"] = e_f(pmf_from_mf(mf_raw, grid), ref_F, grid, mask)
    return out


def load_runs(raw_dir, stage):
    runs = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, f"{stage}__abf_hb*__*.npz"))):
        d = np.load(f, allow_pickle=True)
        r = {k: d[k] for k in d.files}
        arm = str(r["name"])
        runs.setdefault(arm, {})[int(r["seed"])] = r
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=DEFAULT_RAW)
    ap.add_argument("--stage", default="bandwidth_audit")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    runs = load_runs(a.raw_dir, a.stage)
    if not runs:
        sys.exit(f"no runs under {a.raw_dir}")
    arms = sorted(runs, key=lambda s: -float(s[len("abf_hb"):]))
    legacy_arm = arms[0]
    assert legacy_arm == "abf_hb0.025", arms
    seeds = sorted(set.intersection(*[set(runs[m]) for m in arms]))
    print(f"arms {arms}; seeds with all arms: {len(seeds)} {seeds[:4]}{'...' if len(seeds) > 4 else ''}")

    any_run = runs[legacy_arm][seeds[0]]
    grid = np.asarray(any_run["grid"], dtype=float)
    ref_F = np.asarray(any_run["reference_free_energy"], dtype=float)
    ref_mf = np.asarray(any_run["reference_mean_force"], dtype=float)
    t = np.asarray(any_run["profile_times"], dtype=float)
    mask = (grid >= Z_LO) & (grid <= Z_HI)
    sigma = float(any_run.get("abf_smooth_sigma", 0.5))

    # self-check: the legacy read-out recomputed here must reproduce the saved l2_f
    dev = max(abs(float(readouts(runs[legacy_arm][s], grid, mask, ref_F, sigma)[LEGACY_KEY][-1])
                  - float(runs[legacy_arm][s]["l2_f"])) for s in seeds)
    print(f"self-check: recomputed legacy e_F(T) vs saved l2_f, max |dev| {dev:.2e} (float32 profile)")
    assert dev < 1e-5

    # ---- Stage 1: legacy arm only ----
    print("\nStage 1 (legacy arm only): read-out ladder at fixed legacy dynamics")
    ro_legacy = {s: readouts(runs[legacy_arm][s], grid, mask, ref_F, sigma) for s in seeds}
    labels = list(ro_legacy[seeds[0]].keys())
    s1 = {}
    print(f"  {'read-out':>10} {'e_F(T) med':>11} {'seed sd':>8} {'I_F med':>8}")
    for lab in labels:
        fin = np.array([ro_legacy[s][lab][-1] for s in seeds])
        I = np.array([np.trapezoid(ro_legacy[s][lab], t) for s in seeds])
        s1[lab] = dict(eF_T_median=float(np.median(fin)), eF_T_seed_sd=float(np.std(fin)),
                       IF_median=float(np.median(I)))
        print(f"  {lab:>10} {s1[lab]['eF_T_median']:11.5f} {s1[lab]['eF_T_seed_sd']:8.5f} {s1[lab]['IF_median']:8.3f}"
              f"{'  <- legacy' if lab == LEGACY_KEY else ''}")
    emin = min(v["eF_T_median"] for v in s1.values())
    plateau = [lab for lab in labels if s1[lab]["eF_T_median"] <= (1 + PLATEAU_TOL) * emin]
    if LEGACY_KEY in plateau:
        h_star = LEGACY_KEY
    else:
        # largest plateau point in the ladder order (labels are ordered coarse -> fine)
        h_star = plateau[0]
    print(f"  plateau (within {100*PLATEAU_TOL:.0f}% of {emin:.5f}): {plateau} -> h_read* = {h_star}"
          f" ({'legacy ON plateau, kept' if h_star == LEGACY_KEY else 'legacy OFF plateau'})")

    # ---- Stage 2: every arm at h_read* ----
    print(f"\nStage 2: all arms scored at h_read* = {h_star}; paired vs {legacy_arm}")
    ro = {m: {s: readouts(runs[m][s], grid, mask, ref_F, sigma) for s in seeds} for m in arms}
    fin = {m: np.array([ro[m][s][h_star][-1] for s in seeds]) for m in arms}
    I = {m: np.array([np.trapezoid(ro[m][s][h_star], t) for s in seeds]) for m in arms}
    s2 = {}
    print(f"  {'arm':>14} {'e_F(T) med':>11} {'d e_F(T)':>9} {'CI95':>20} {'wins':>5} {'d I_F':>8} {'CI95':>20}"
          f" {'rough':>6} {'clip%':>6} {'max|F|':>7}")
    for m in arms:
        d_fin = 100 * (fin[m] - fin[legacy_arm]) / fin[legacy_arm]
        d_int = 100 * (I[m] - I[legacy_arm]) / I[legacy_arm]
        n = len(seeds)
        ci_f = list(boot_median(d_fin, BOOT_SEED)) if n >= 2 else [float("nan")] * 2
        ci_i = list(boot_median(d_int, BOOT_SEED + 1)) if n >= 2 else [float("nan")] * 2
        # failure-mode instrumentation on this arm's own online profile (production estimator,
        # same bandwidth as the bias estimator; differs from it only by the burn-in samples)
        mf_T = np.stack([runs[m][s]["mean_force_t"][-1] for s in seeds]).astype(float)
        dz = grid[1] - grid[0]
        r_ref = np.sqrt(np.mean(np.gradient(ref_mf[mask], dz) ** 2))
        rough = np.median([np.sqrt(np.mean(np.gradient(x[mask], dz) ** 2)) / r_ref for x in mf_T])
        clip = 100 * np.mean(np.abs(mf_T[:, mask]) >= CLIP)
        s2[m] = dict(eF_T_median=float(np.median(fin[m])), d_fin=float(np.median(d_fin)), d_fin_ci=ci_f,
                     wins=int((d_fin < 0).sum()), n=n, d_int=float(np.median(d_int)), d_int_ci=ci_i,
                     roughness=float(rough), clip_pct=float(clip), max_abs_F=float(np.abs(mf_T[:, mask]).max()))
        print(f"  {m:>14} {s2[m]['eF_T_median']:11.5f} {s2[m]['d_fin']:+9.2f} [{ci_f[0]:+8.2f},{ci_f[1]:+8.2f}]"
              f" {s2[m]['wins']:3d}/{n} {s2[m]['d_int']:+8.2f} [{ci_i[0]:+8.2f},{ci_i[1]:+8.2f}]"
              f" {rough:6.3f} {clip:6.2f} {s2[m]['max_abs_F']:7.2f}")

    # ---- Outcome (prereg outcomes_frozen) ----
    resolved_better = [m for m in arms[1:] if s2[m]["d_fin_ci"][1] < 0]
    resolved_worse = [m for m in arms[1:] if s2[m]["d_fin_ci"][0] > 0]
    if resolved_worse:
        outcome = "C_online_hurts"
    elif resolved_better:
        outcome = "B_online_helps"
    elif h_star != LEGACY_KEY:
        outcome = "A_readout_only"
    else:
        outcome = "D_nothing_moves"
    print(f"\n  OUTCOME: {outcome}   (resolved better: {resolved_better}; resolved worse: {resolved_worse})")
    if len(seeds) < pre["n_seeds"]:
        print(f"  NOTE: {len(seeds)} of {pre['n_seeds']} preregistered seeds present -- not confirmatory")

    out_json = a.out_json or os.path.join(os.path.dirname(os.path.dirname(a.raw_dir)), "analysis.json")
    with open(out_json, "w") as fh:
        json.dump(dict(prereg=os.path.relpath(PREREG, ROOT), raw_dir=os.path.relpath(a.raw_dir, ROOT),
                       seeds=seeds, stage1=s1, plateau=plateau, h_read_star=h_star, stage2=s2,
                       outcome=outcome), fh, indent=2)
    print(f"  wrote {os.path.relpath(out_json, ROOT)}")


if __name__ == "__main__":
    main()
