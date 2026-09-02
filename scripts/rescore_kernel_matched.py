#!/usr/bin/env python
"""EXPLORATORY (not preregistered): re-score closed two-arm comparisons against the estimator's
own fixed point -- the reference mean force smoothed with the engine's legacy read-out kernel
(kernel matching) -- next to the published raw-reference numbers.

Why: the bandwidth-defect screen (audit_readout_smoothing_all.py) found the gateway's ABF
mean-force error equals the deterministic kernel bias to 1.3%.  When the baseline is read-out
limited, an arm contrast scored against the UNSMOOTHED reference is contaminated by an
occupancy-dependent kernel bias -- and occupancy is exactly what marginal FR changes.  Kernel
matching removes the deterministic part of that contamination (it is NOT a sharper read-out;
raw accumulators would be needed for that, and these runs saved none).

Systems with stored profile time series: gateway (F_prof_t), CHA x3 (pmf), WCA Case IX
(mean_force_t).  Paired per-seed relative change, median, 10k bootstrap (seed 20260829).

    python scripts/rescore_kernel_matched.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
ROOT = os.path.join(SCRIPTS, "..")
from audit_readout_smoothing import line_smooth                                   # noqa: E402
from audit_readout_smoothing_all import conv_reflect_normalised, nw_line_unpadded, cumtrapz  # noqa: E402
from analyze_uniform_lta import boot_median, BOOT_SEED                             # noqa: E402

OUT = os.path.join(ROOT, "results/information_campaign/kernel_matched_rescore.json")


def stat(d, k=0):
    d = np.asarray(d, float)
    lo, hi = boot_median(d, BOOT_SEED + k)
    return dict(median=float(np.median(d)), ci95=[lo, hi], wins=int((d < 0).sum()), n=int(len(d)))


def fmt(s):
    return f"{s['median']:+7.2f}% [{s['ci95'][0]:+7.2f},{s['ci95'][1]:+7.2f}] {s['wins']:2d}/{s['n']}"


def aligned_series(prof_t, ref, mask, w=None):
    """(T, G) profiles -> aligned RMS over mask per time."""
    d = (prof_t - ref[None, :])[:, mask]
    d = d - d.mean(1, keepdims=True)
    return np.sqrt((d * d).mean(1))


def contrast(e_abf, e_fr, t, k):
    Ia, Iu = np.array([np.trapezoid(e, t) for e in e_abf]), np.array([np.trapezoid(e, t) for e in e_fr])
    fa, fu = np.array([e[-1] for e in e_abf]), np.array([e[-1] for e in e_fr])
    return dict(d_int=stat(100 * (Iu - Ia) / Ia, k), d_fin=stat(100 * (fu - fa) / fa, k + 1),
                abf_eF_T_median=float(np.median(fa)))


def show(name, res):
    print(f"\n{name}")
    for lab in ("raw", "km"):
        print(f"  [{lab:>3}] dI_F {fmt(res[lab]['d_int'])} | d e_F(T) {fmt(res[lab]['d_fin'])} | ABF e_F(T) {res[lab]['abf_eF_T_median']:.4f}")


def main():
    out = {}

    # ---- gateway: 64 pairs (seed, init); engine kernel h from the row config ----
    z = np.load(os.path.join(ROOT, "results/uniform_campaign/gateway/raw.npz"), allow_pickle=True)
    meth, seed, init = (np.asarray(z[k]) for k in ("method", "seed", "init"))
    x = np.asarray(z["x_grid"][0], float)
    dx = x[1] - x[0]
    mask = (x >= -1.5) & (x <= 1.5)
    pairs = []
    for s in sorted(set(seed.tolist())):
        for ini in sorted(set(init.tolist())):
            ia = np.flatnonzero((meth == "abf") & (seed == s) & (init == ini))
            iu = np.flatnonzero((meth == "fr_uniform") & (seed == s) & (init == ini))
            if len(ia) == 1 and len(iu) == 1:
                pairs.append((int(ia[0]), int(iu[0])))
    h = float(json.loads(str(z["config_json"][pairs[0][0]]))["h"])
    t = np.asarray(z["t"][pairs[0][0]], float)
    res, ratio = {}, {}
    for lab in ("raw", "km"):
        ea, eu, ratios, chk = [], [], [], []
        for ia, iu in pairs:
            F_ref, Fp_ref = np.asarray(z["F_ref"][ia], float), np.asarray(z["Fp_ref"][ia], float)
            R = F_ref if lab == "raw" else cumtrapz(conv_reflect_normalised(Fp_ref, h, dx), dx)
            a = aligned_series(np.asarray(z["F_prof_t"][ia], float), R, mask)
            u = aligned_series(np.asarray(z["F_prof_t"][iu], float), R, mask)
            ea.append(a); eu.append(u); ratios.append(u / a)
            if lab == "raw":
                chk.append(abs(a[-1] - float(z["final_l2_f"][ia])))
        if lab == "raw":
            assert max(chk) < 1e-6, max(chk)
        res[lab] = contrast(ea, eu, t, 0 if lab == "raw" else 10)
        r = np.median(np.array(ratios), 0)
        idx = [int(np.argmin(abs(t - v))) for v in (2, 5, 10, 17, 20, 30, 40)]
        ratio[lab] = dict(t=[float(t[i]) for i in idx], uni_over_abf=[float(r[i]) for i in idx], final=float(r[-1]))
    # F' error vs the deterministic kernel bias (the screen's saturation diagnosis)
    ia0 = pairs[0][0]
    Fp_ref = np.asarray(z["Fp_ref"][ia0], float)
    det = float(np.sqrt(np.mean((conv_reflect_normalised(Fp_ref, h, dx) - Fp_ref)[mask] ** 2)))
    meas = float(np.median(z["final_l2_fp"][meth == "abf"]))
    out["gateway"] = dict(n_pairs=len(pairs), h=h, results=res, error_ratio_time_course=ratio,
                          abf_final_l2_fp_median=meas, deterministic_kernel_bias_Fp=det,
                          note="F' error equals the deterministic kernel bias to %.1f%%" % (100 * abs(meas / det - 1)))
    show(f"Gateway ({len(pairs)} pairs, h={h})", res)
    for lab in ("raw", "km"):
        print(f"  ratio uni/abf [{lab}] at t={ratio[lab]['t']}: {np.round(ratio[lab]['uni_over_abf'], 3).tolist()} final {ratio[lab]['final']:.3f}")
    print(f"  ABF F' error {meas:.4f} vs deterministic kernel bias {det:.4f}")

    # ---- CHA x3 (pmf series (T, R, G); unnormalised non-periodic NW, bw 0.15) ----
    for tag in ("ethene_450", "propene_450", "propene_600"):
        ref = np.load(os.path.join(ROOT, f"results/uniform_campaign/cha/reference/reference_{tag}.npz"), allow_pickle=True)
        runs = {m: np.load(os.path.join(ROOT, f"results/uniform_campaign/cha/production_{tag}/{m}.npz"), allow_pickle=True)
                for m in ("abf", "fr_uniform")}
        g = np.asarray(runs["abf"]["grid"], float)
        dz = g[1] - g[0]
        F = np.asarray(ref["F"], float)
        mask = (g >= float(ref["xi_A"]) - 1.0) & (g <= float(ref["xi_B"]) + 1.0)
        t = np.asarray(runs["abf"]["times"], float)
        Fkm = cumtrapz(nw_line_unpadded(np.gradient(F, g), g, 0.15), dz)
        res = {}
        for lab, R in (("raw", F), ("km", Fkm)):
            e = {}
            for m in runs:
                P = np.asarray(runs[m]["pmf"], float)                 # (T, R, G)
                e[m] = [aligned_series(P[:, r, :], R, mask) for r in range(P.shape[1])]
            res[lab] = contrast(e["abf"], e["fr_uniform"], t, 20 if lab == "raw" else 30)
        out[f"cha_{tag}"] = dict(n_pairs=len(e["abf"]), results=res)
        show(f"CHA {tag} ({len(e['abf'])} pairs)", res)

    # ---- WCA Case IX (mean_force_t; NW 0.025 + 0.5-bin smoothing mirrored by line_smooth) ----
    runs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results/uniform_campaign/wca/uniform/raw/uniform__*__*.npz"))):
        d = np.load(f, allow_pickle=True)
        sp = json.loads(str(d["spec_json"]))
        runs[(sp["method"], sp["seed"])] = {k: d[k] for k in ("grid", "reference_free_energy", "ref_mean_force",
                                                              "mean_force_t", "profile_times", "l2_f")}
    seeds = sorted({s for (m, s) in runs if m == "abf" and ("fr_uniform", s) in runs})
    a0 = runs[("abf", seeds[0])]
    g = np.asarray(a0["grid"], float)
    dz = g[1] - g[0]
    mask = (g >= -0.1) & (g <= 1.1)
    t = np.asarray(a0["profile_times"], float)
    F, mf = np.asarray(a0["reference_free_energy"], float), np.asarray(a0["ref_mean_force"], float)

    def pmf(m):
        m = np.asarray(m, float)
        inc = 0.5 * (m[..., 1:] + m[..., :-1]) * dz
        return np.concatenate([np.zeros(m.shape[:-1] + (1,)), np.cumsum(inc, -1)], -1)

    def eF(P, R):   # engine convention: trapezoid-weighted aligned RMS over the window
        d = (P - R)[..., mask]
        d = d - d.mean(-1, keepdims=True)
        gg = g[mask]
        return np.sqrt(np.trapezoid(d * d, gg, axis=-1) / (gg[-1] - gg[0]))
    Fkm = pmf(line_smooth(mf, 0.025 / dz, 0.5))
    chk = max(abs(eF(pmf(runs[("abf", s)]["mean_force_t"])[-1], F) - float(runs[("abf", s)]["l2_f"])) for s in seeds)
    assert chk < 1e-5, chk
    res = {}
    for lab, R in (("raw", F), ("km", Fkm)):
        ea = [eF(pmf(runs[("abf", s)]["mean_force_t"]), R) for s in seeds]
        eu = [eF(pmf(runs[("fr_uniform", s)]["mean_force_t"]), R) for s in seeds]
        res[lab] = contrast(ea, eu, t, 40 if lab == "raw" else 50)
    out["wca_caseIX"] = dict(n_pairs=len(seeds), results=res)
    show(f"WCA Case IX ({len(seeds)} pairs)", res)

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
