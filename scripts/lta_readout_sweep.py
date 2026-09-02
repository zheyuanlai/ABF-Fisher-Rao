#!/usr/bin/env python
"""Offline READ-OUT bandwidth sweep for the LTA temperature sweep (re-analysis only).

The LTA engine (src/lta/core_lta.py) keeps raw binned accumulators and smooths
them at read time with a FIXED wrapped-Gaussian matrix K (abf_bandwidth 0.05 rad):
    mean_force = (K fsum)/(K csum),   eff_counts = K csum.
Both saved arrays are float64 and K is well conditioned (cond ~1.25e4), so the
raw binned accumulators are recovered EXACTLY by solving the linear system.
That lets the read-out bandwidth be swept offline at FIXED dynamics -- the same
instrument used for ZIF-8 (commit cbcbf3c) -- without a single new run.

Stage 1 (ABF arm only): e_F(T) and I_F vs h_read; the plateau rule (largest h
whose median e_F(T) is within PLATEAU_TOL of the ladder minimum, the rule that
reproduces the frozen ZIF-8 choice) selects h_read*.
Stage 2: both arms re-scored at every ladder point; the campaign endpoints
(paired per-seed dI_F, median, 10k bootstrap CI seed 20260829) are reported at
the legacy read-out (must reproduce sweep_summary.json) and at h_read*.

This does NOT change the ONLINE bandwidth (h_bias): the dynamics are the legacy
dynamics.  It answers only: is the LTA benefit a read-out artefact?

    python scripts/lta_readout_sweep.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from analyze_uniform_lta import (BOOT_SEED, boot_median, circular_interp_ref,  # noqa: E402
                                 error_series)

ROOT = os.path.join(SCRIPTS, "..")
LTA = os.path.join(ROOT, "results/uniform_campaign/lta")
PREREG = os.path.join(ROOT, "configs/uniform_campaign/lta_sweep_prereg.json")
OUT_JSON = os.path.join(ROOT, "results/information_campaign/lta_readout_sweep.json")
TEMPS = (80, 150, 225, 300)
H_LEGACY = 0.05
LADDER = (0.10, 0.075, 0.05, 0.035, 0.025, 0.0175, 0.0125, 0.00875, 0.0)
PLATEAU_TOL = 0.02
EPS = 1.0e-12          # alkanes.periodic.EPS
TWO_PI = 2.0 * math.pi


def kmat(grid, bw):
    """alkanes.periodic.wrapped_gaussian_kernel_matrix; bw <= 0 -> identity (raw bins)."""
    if bw <= 0:
        return np.eye(len(grid))
    d = np.abs(grid[:, None] - grid[None, :])
    d = np.minimum(d, TWO_PI - d)
    return np.exp(-0.5 * (d / bw) ** 2)


def free_energy_from_mean_force(mf, dphi):
    """numpy mirror of alkanes.periodic.free_energy_from_mean_force (checked to 1e-14)."""
    mf0 = mf - mf.mean(-1, keepdims=True)
    inc = 0.5 * (mf0 + np.roll(mf0, -1, axis=-1)) * dphi
    F = np.cumsum(inc, axis=-1)
    F = np.roll(F, 1, axis=-1)
    F[..., 0] = 0.0
    return F - F.mean(-1, keepdims=True)


def recover_raw(run, K):
    """Exact inversion of the saved smoothed arrays -> (fsum_used, csum_used), (S,R,G).

    eff_counts(t) = K csum(t) (FULL-run counts); u_counts = csum_prod(T); the burn-in
    block csum(T) - csum_prod(T) is time-independent, so csum_prod(t) follows.  The
    engine reports fsum_prod/csum_prod once csum_prod.sum() > 0 and fsum/csum before.
    """
    ec = np.asarray(run["eff_counts"], dtype=np.float64)
    S, R, G = ec.shape
    c_full = np.linalg.solve(K, ec.reshape(-1, G).T).T.reshape(S, R, G)
    int_dev = float(np.abs(c_full - np.round(c_full)).max())
    c_full = np.round(c_full)
    burn = np.round(c_full[-1] - np.asarray(run["u_counts"], dtype=np.float64))
    assert (burn >= 0).all(), "burn-in counts must be non-negative"
    c_prod = c_full - burn[None]
    use_prod = c_prod.sum(-1) > 0                                # (S,R)
    c_used = np.where(use_prod[..., None], c_prod, c_full)
    mf = np.asarray(run["mean_force"], dtype=np.float64)
    Kc = (c_used.reshape(-1, G) @ K.T).reshape(S, R, G)
    f_used = np.linalg.solve(K, (mf * Kc).reshape(-1, G).T).T.reshape(S, R, G)
    f_used = np.where(c_used > 0, f_used, 0.0)                   # no support -> no force
    return f_used, c_used, int_dev


def score(f, c, K_h, dphi, F_ref):
    """Engine convention: mf = (K f)/(K c) where K c > EPS else 0; PMF; aligned RMS."""
    S, R, G = f.shape
    num = f.reshape(-1, G) @ K_h.T
    den = c.reshape(-1, G) @ K_h.T
    mf = np.where(den > EPS, num / np.maximum(den, EPS), 0.0).reshape(S, R, G)
    pmf = free_energy_from_mean_force(mf, dphi)
    return error_series(pmf, F_ref), mf


def paired(e_abf, e_fr, t, seed=BOOT_SEED):
    I_a, I_f = np.trapezoid(e_abf, t, axis=0), np.trapezoid(e_fr, t, axis=0)
    d_int = 100.0 * (I_f - I_a) / I_a
    d_fin = 100.0 * (e_fr[-1] - e_abf[-1]) / e_abf[-1]
    lo, hi = boot_median(d_int, seed)
    flo, fhi = boot_median(d_fin, seed + 1)
    return dict(d_int=float(np.median(d_int)), d_int_ci=[lo, hi],
                wins=int((d_int < 0).sum()), d_fin=float(np.median(d_fin)),
                d_fin_ci=[flo, fhi], I_abf_median=float(np.median(I_a)),
                I_fr_median=float(np.median(I_f)),
                eF_T_abf_median=float(np.median(e_abf[-1])),
                eF_T_fr_median=float(np.median(e_fr[-1])))


def main():
    pre = json.load(open(PREREG))
    rule = pre["success_rule"]
    out = dict(ladder=list(LADDER), h_legacy=H_LEGACY, plateau_tol=PLATEAU_TOL,
               rule=("plateau = ladder h with median ABF e_F(T) <= (1+tol) * ladder min; "
                     "h_read* = legacy if legacy is on the plateau, else the largest plateau h"),
               per_T={})
    for T in TEMPS:
        ref = np.load(os.path.join(LTA, "reference", f"reference_T{T}.npz"), allow_pickle=True)
        runs = {m: np.load(os.path.join(LTA, f"production_T{T}", f"{m}.npz"), allow_pickle=True)
                for m in ("abf", "fr_uniform")}
        grid = np.asarray(runs["abf"]["grid"], dtype=np.float64)
        dphi = float(runs["abf"]["dphi"])
        t = np.asarray(runs["abf"]["times"], dtype=float)
        F_ref = circular_interp_ref(ref["F"], ref["grid_phi"], grid)
        K_leg = kmat(grid, H_LEGACY)

        raw, chk = {}, {}
        for m, run in runs.items():
            f, c, int_dev = recover_raw(run, K_leg)
            raw[m] = (f, c)
            e_re, mf_re = score(f, c, K_leg, dphi, F_ref)
            e_saved = error_series(np.asarray(run["pmf"], dtype=float), F_ref)
            chk[m] = dict(count_int_dev=int_dev,
                          mf_dev=float(np.abs(mf_re - run["mean_force"]).max()),
                          eF_dev=float(np.abs(e_re - e_saved).max()))
        print(f"\n=== T = {T} K ===  exactness of the inversion (re-smoothed at legacy h):")
        for m, c in chk.items():
            print(f"  {m:10s} counts->int {c['count_int_dev']:.1e}   |mf - saved| {c['mf_dev']:.1e}"
                  f"   |e_F - saved| {c['eF_dev']:.1e}")
            assert c["mf_dev"] < 1e-7 and c["eF_dev"] < 1e-9, "inversion not exact"

        # ---- Stage 1: ABF arm only ----
        print(f"  Stage 1 (ABF only): read-out sweep at fixed legacy dynamics")
        print(f"  {'h_read':>8} {'h/bin':>6} {'e_F(T) med':>11} {'seed sd':>8} {'I_F med':>9}")
        s1 = {}
        e_abf_at = {}
        for h in LADDER:
            e, _ = score(*raw["abf"], kmat(grid, h), dphi, F_ref)
            e_abf_at[h] = e
            s1[h] = dict(eF_T_median=float(np.median(e[-1])), eF_T_seed_sd=float(np.std(e[-1])),
                         IF_median=float(np.median(np.trapezoid(e, t, axis=0))))
            tag = "  <- legacy" if h == H_LEGACY else ""
            print(f"  {h:8.5f} {h/dphi:6.2f} {s1[h]['eF_T_median']:11.4f} {s1[h]['eF_T_seed_sd']:8.4f}"
                  f" {s1[h]['IF_median']:9.3f}{tag}")
        emin = min(v["eF_T_median"] for v in s1.values())
        plateau = [h for h in LADDER if s1[h]["eF_T_median"] <= (1 + PLATEAU_TOL) * emin]
        # ZIF-8 rule generalised: the legacy read-out is kept if it is already on the
        # plateau (no change is licensed); otherwise the largest plateau h is taken.
        h_star = H_LEGACY if H_LEGACY in plateau else max(plateau)
        mse_gain = (s1[H_LEGACY]["eF_T_median"] / s1[h_star]["eF_T_median"]) ** 2
        note = "legacy ON plateau, kept" if H_LEGACY in plateau else "legacy OFF plateau"
        print(f"  plateau (within {100*PLATEAU_TOL:.0f}% of min {emin:.4f}): {plateau}  ->  h_read* = {h_star} ({note})"
              f"   ABF e_F(T) {s1[H_LEGACY]['eF_T_median']:.4f} -> {s1[h_star]['eF_T_median']:.4f}"
              f"  ({mse_gain:.2f}x MSE)")

        # ---- Stage 2: both arms ----
        print(f"  Stage 2 (both arms): FR vs ABF at every read-out")
        print(f"  {'h_read':>8} {'dI_F med':>9} {'CI95':>20} {'wins':>5} {'d e_F(T)':>9} {'CI95':>20}")
        s2 = {}
        for h in LADDER:
            e_fr, _ = score(*raw["fr_uniform"], kmat(grid, h), dphi, F_ref)
            p = paired(e_abf_at[h], e_fr, t)
            accel = p["d_int"] <= rule["median_rel_change_pct_max"] and p["d_int_ci"][1] < rule["ci95_upper_pct_max"]
            p["accel"] = bool(accel)
            s2[h] = p
            tag = "  <- legacy" if h == H_LEGACY else ("  <- h_read*" if h == h_star else "")
            print(f"  {h:8.5f} {p['d_int']:+9.2f} [{p['d_int_ci'][0]:+8.2f},{p['d_int_ci'][1]:+8.2f}]"
                  f" {p['wins']:3d}/16 {p['d_fin']:+9.2f} [{p['d_fin_ci'][0]:+8.2f},{p['d_fin_ci'][1]:+8.2f}]{tag}")
        summ = json.load(open(os.path.join(LTA, f"summary_T{T}.json")))
        rep = summ["d_int_pct"]["median"]
        print(f"  reproduction of the published legacy dI_F: {s2[H_LEGACY]['d_int']:+.4f} vs {rep:+.4f}"
              f"  (diff {s2[H_LEGACY]['d_int']-rep:+.1e})")
        assert abs(s2[H_LEGACY]["d_int"] - rep) < 1e-6
        out["per_T"][str(T)] = dict(exactness=chk, stage1={str(h): v for h, v in s1.items()},
                                    plateau=plateau, h_read_star=h_star, abf_mse_gain=mse_gain,
                                    stage2={str(h): v for h, v in s2.items()},
                                    published_legacy_d_int=rep)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.relpath(OUT_JSON, ROOT)}")


if __name__ == "__main__":
    main()
