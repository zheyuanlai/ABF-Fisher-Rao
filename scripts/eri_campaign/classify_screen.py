"""Regime classifier of the ABF-only screen, exactly as frozen in PREREGISTRATION.md Amendment 1 (A1.3-A1.4).

    python scripts/eri_campaign/classify_screen.py results/eri_campaign/screen_dilute [--out ...json]

Per K (median over seeds unless stated):
 1 discovery-limited    : median first true translocation > 0.1 T, or any seed without one by 0.5 T
 2 cond.-equil.-limited : (a) tau_int(eta) of the cage axial coordinate (Gate 2 EM 2e-4 umbrella time series) >= 0.1 T
                          (b) median over saves in [T/2, T] of d_eta / floor > 3 (median seed)
 3 establishment-limited: (E) longest span in [2, T] with n_W < 0.5 N Q*_W - 2 sd >= 0.2 T  AND
                          (I) window deposit share ratio < 0.5 OR window bins carry >= 50 % of final squared MF error
 4 otherwise ABF-sufficient
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import metrics as mt  # noqa: E402
from eri.model import KB  # noqa: E402

TWO_PI = 2 * math.pi


def tau_int(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    v = x.var()
    if v <= 0 or n < 20:
        return float("nan")
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / (v * np.arange(n, 0, -1))
    t = 0.5
    for k in range(1, n // 2):
        t += ac[k]
        if k > 5 * t:
            break
    return float(t)


def hidden_stats(variant="gard_tait", tag="EM_2e-04", sample_tu=1e-3, n_coarse=18, with_means=False, base=None):
    """tau_int(eta) (t.u.) per cage-region window from the Gate 2 umbrella time series of the first molecule, and the
    conditional variance Var(eta | coarse bin) pooled over all windows' time series."""
    d = os.path.join(base or os.path.join(ROOT, "results", "eri_campaign", "reference"), variant, tag)
    taus = []
    phis, etas, uxs = [], [], []
    for p in sorted(glob.glob(os.path.join(d, "g*_w*.npz"))):
        z = np.load(p)
        ts = z["ts"]
        c = float(z["centre"])
        if abs(c) >= 2.0 and len(ts) > 100:
            taus.append(tau_int(ts[:, 1]) * sample_tu)
        phis.append(ts[:, 0])
        etas.append(ts[:, 1])
        uxs.append(ts[:, 2])
    phis = np.concatenate(phis)
    etas = np.concatenate(etas)
    uxs = np.concatenate(uxs)
    b = (np.floor((phis + math.pi) / TWO_PI * n_coarse).astype(int)) % n_coarse
    var = np.array([etas[b == k].var() if (b == k).sum() > 10 else np.nan for k in range(n_coarse)])
    if not with_means:
        return np.array(taus), var
    # NOTE: umbrella time series pool all windows unweighted; within a 20-degree coarse bin the umbrella bias varies,
    # so these are approximate conditional moments (used for the lag floor and the reference means of A2.2).
    means = {"eta": np.array([etas[b == k].mean() for k in range(n_coarse)]),
             "ux2": np.array([uxs[b == k].mean() for k in range(n_coarse)])}
    vars_ = {"eta": var, "ux2": np.array([uxs[b == k].var() for k in range(n_coarse)])}
    return np.array(taus), var, means, vars_


def classify(dirpath, T_frac_disc=0.1, span_frac=0.2, variant="gard_tait", hidden_tag="EM_2e-04"):
    files = sorted(f for f in glob.glob(os.path.join(dirpath, "abf_N*_s*.npz")) if not f.endswith((".ckpt.npz", ".tmp.npz")))
    byK = {}
    for p in files:
        K = int(re.search(r"_N(\d+)_", os.path.basename(p)).group(1))
        byK.setdefault(K, []).append(p)
    ref = np.load(os.path.join(mt.REF_BASE, variant, "reference_profile.npz"))
    Fr, Gr, _ = mt.reference_on_grid(180, variant)
    taus, var_eta, ref_means, ref_vars = hidden_stats(variant=variant, tag=hidden_tag, with_means=True,
                                                      base=mt.REF_BASE)
    # reference conditional means from the MC accumulators (exact, count-weighted per coarse bin)
    cc = ref["counts"].reshape(18, -1)
    ref_means = {"eta": (ref["eta"].reshape(18, -1) * cc).sum(1) / cc.sum(1),
                 "ux2": (ref["ux2"].reshape(18, -1) * cc).sum(1) / cc.sum(1)}
    out = {"tau_int_eta_cage_windows_tu": dict(median=float(np.nanmedian(taus)), max=float(np.nanmax(taus)),
                                               n=int(len(taus))),
           "var_eta_coarse": var_eta.tolist(), "cells": {}}
    for K in sorted(byK):
        seeds = []
        for p in byK[K]:
            r = mt.load_run(p)
            c = r["cfg"]
            dt = c["dt"]
            t = r["times"]
            T = float(t[-1])
            kT = KB * c["T_K"]
            N = int(c["N"])
            # 1 discovery
            ft = r["ist"][6]
            t_trans = ft * dt if ft >= 0 else float("inf")
            i_half = np.searchsorted(t, 0.5 * T)
            trans_by_half = int(r["sv_i"][i_half, 1])
            fw = r["ist"][5]
            t_win = fw * dt if fw >= 0 else float("inf")
            # 2 conditional lag
            lag = mt.conditional_lag(r, ref)
            cnt = r["sv_cond"][:, :, 0]
            occ = cnt > 0
            floor = np.sqrt(np.nansum(np.where(occ, var_eta[None, :], 0.0), 1) / N)
            sel = t >= 0.5 * T
            ratio = float(np.median(lag["d_eta"][sel] / floor[sel]))
            lag2 = mt.conditional_lag_timeavg(r, ref_means, ref_vars)
            # 3 establishment + information
            m = mt.marginal(r, Fr, kT)
            T_est, f_under, span = mt.establishment(t, m["frac_win"], m["Q_win"], N, t_from=2.0)
            blk = 2.0 if T <= 50 else 0.04 * T              # ERI A2 / overnight prereg section 5
            ratios_b, flags_b, span_b, _ = mt.establishment_blocks(t, m["frac_win"], m["Q_win"], N, block=blk)
            Wfrac = float((np.abs((np.arange(r["sv_C"].shape[1]) + 0.5) / r["sv_C"].shape[1] * TWO_PI - math.pi)
                           < c["window_phi"]).mean())
            ratios_u, flags_u, span_u, _ = mt.establishment_blocks(t, m["frac_win"], np.full_like(t, Wfrac), N,
                                                                   block=blk)
            ng = r["sv_C"].shape[1]
            grid = (np.arange(ng) + 0.5) / ng * TWO_PI - math.pi
            W = np.abs(grid) < c["window_phi"]
            C = r["sv_C"][-1]
            share = (C[W].sum() / C.sum()) / (W.sum() / ng)
            G = mt.mean_force(r["sv_M"][-1], C)
            se2 = (G - Gr) ** 2
            err_share_W = float(se2[W].sum() / se2.sum())
            e = mt.errors(r, Fr, Gr)
            i2 = np.searchsorted(t, 2.0)
            Gl = mt.mean_force(r["sv_M"][-1] - r["sv_M"][i2], C - r["sv_C"][i2])
            late = float(mt.centred_rms(mt.histogram_pmf(Gl) - Fr))
            excess_tv = float(np.median((m["tv_Q"] - m["tv_floor_Q"])[sel]))
            seeds.append(dict(file=os.path.relpath(p, ROOT), t_trans=t_trans, trans_by_half=trans_by_half, t_win=t_win,
                              d_eta_ratio=ratio, T_est=T_est, f_under=f_under, span=span, deposit_share_W=share,
                              err_share_W=err_share_W, eF_T=e["eF_T"], Ibar_F=e["Ibar_F"], late_eF=late,
                              tv_excess_second_half=excess_tv, n_transloc=int(r["sv_i"][-1, 1]),
                              span_blocks=span_b, min_block_ratio=float(ratios_b.min()),
                              median_block_ratio=float(np.median(ratios_b)), span_blocks_uniform=span_u,
                              min_block_ratio_uniform=float(ratios_u.min()),
                              lag_eta=lag2["eta"][0] / lag2["eta"][1], lag_ux2=lag2["ux2"][0] / lag2["ux2"][1],
                              transient_ratio=e["eF_T"] / max(late, 1e-12),
                              n_wcross=int(r["sv_i"][-1, 0]), T=T))
        med = lambda k: float(np.median([s[k] for s in seeds]))
        T = seeds[0]["T"]
        disc_lim = (med("t_trans") > T_frac_disc * T) or any(s["trans_by_half"] == 0 for s in seeds)
        ce_a = out["tau_int_eta_cage_windows_tu"]["median"] >= 0.1 * T
        ce_b = (med("lag_eta") > 3.0) or (med("lag_ux2") > 3.0)          # Amendment 2 (2b')
        E_A1 = med("span") >= span_frac * T                                # frozen A1 rule (vacuous), reported
        E = med("span_blocks") >= span_frac * T                            # Amendment 2 (E')
        I = (med("deposit_share_W") < 0.5) or (med("err_share_W") >= 0.5)
        if disc_lim:
            label = "DISCOVERY-LIMITED"
        elif ce_a or ce_b:
            label = "CONDITIONAL-EQUILIBRATION-LIMITED"
        elif E and I:
            label = "ESTABLISHMENT-LIMITED (eligible)"
        else:
            label = "ABF-SUFFICIENT"
        out["cells"][str(K)] = dict(label=label, n_seeds=len(seeds), tests=dict(
            discovery=dict(median_t_trans=med("t_trans"), threshold=T_frac_disc * T,
                           seeds_without_translocation_by_half_T=int(sum(s["trans_by_half"] == 0 for s in seeds)),
                           limited=bool(disc_lim)),
            conditional=dict(tau_eta=out["tau_int_eta_cage_windows_tu"]["median"], tau_threshold=0.1 * T,
                             median_lag_eta_over_floor=med("lag_eta"), median_lag_ux2_over_floor=med("lag_ux2"),
                             A1_median_d_eta_ratio=med("d_eta_ratio"), ratio_threshold=3.0,
                             limited=bool(ce_a or ce_b)),
            establishment=dict(median_longest_deficient_block_run=med("span_blocks"),
                               median_min_block_ratio=med("min_block_ratio"),
                               median_median_block_ratio=med("median_block_ratio"),
                               uniform_target_median_run=med("span_blocks_uniform"),
                               uniform_target_median_min_ratio=med("min_block_ratio_uniform"),
                               A1_median_longest_span=med("span"), A1_E=bool(E_A1), span_threshold=span_frac * T,
                               median_T_est=med("T_est"), median_f_under=med("f_under"), E=bool(E),
                               median_deposit_share_W=med("deposit_share_W"), median_err_share_W=med("err_share_W"),
                               I=bool(I))),
            secondary=dict(median_transient_ratio=med("transient_ratio"), median_eF_T=med("eF_T"), median_Ibar_F=med("Ibar_F"), median_late_eF=med("late_eF"),
                           median_tv_excess_second_half=med("tv_excess_second_half"), median_t_win=med("t_win"),
                           median_n_transloc=med("n_transloc")), seeds=seeds)
        print(f"K={K:5d}: {label}")
        print("   ", json.dumps(out["cells"][str(K)]["tests"], default=float))
        print("   ", json.dumps(out["cells"][str(K)]["secondary"], default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--refbase", default=None)
    ap.add_argument("--variant", default="gard_tait")
    ap.add_argument("--hidden-tag", default="EM_2e-04")
    a = ap.parse_args()
    if a.refbase:
        mt.set_reference_base(a.refbase)
    res = classify(os.path.join(ROOT, a.dir) if not os.path.isabs(a.dir) else a.dir, variant=a.variant,
                   hidden_tag=a.hidden_tag)
    out = a.out or os.path.join(a.dir, "classification.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print("->", out)
