#!/usr/bin/env python
"""Analyze Stage 1 of the uniform-FR campaign (entropic gateway, abf vs fr_uniform).

Reads results/uniform_campaign/gateway/raw.npz, pairs the two arms per (init, seed),
and applies the endpoints frozen in docs/UNIFORM_FR_CAMPAIGN.md:

  primary   Delta I_F   = paired per-seed median relative change of int_l2_f, 10k bootstrap
  secondary Delta e_F(T), time-to-accuracy speedups (atlas convention), frozen-bias endpoint
  health    min ancestor ESS/N >= 0.30, max lineage share <= 0.05
  verdict   acceleration-positive / safe / neutral / negative per the frozen rules

    python scripts/analyze_uniform_gateway.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "results/uniform_campaign/gateway/raw.npz")
PREREG = os.path.join(ROOT, "configs/uniform_campaign/gateway_prereg.json")
OUT = os.path.join(ROOT, "results/uniform_campaign/gateway")

PERSIST = 0.2          # atlas convention: threshold must hold for 0.2 * T
FRACTIONS = (0.5, 0.25, 0.125)


def boot_median(x, n_boot, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    med = np.median(np.asarray(x)[idx], axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def tau(times, curve, eps, persist):
    """First t where curve <= eps sustained for persist*T; inf if never."""
    T = times[-1]
    hold = persist * T
    below = curve <= eps
    for i in range(len(times)):
        if not below[i]:
            continue
        j = np.searchsorted(times, times[i] + hold, side="left")
        j = min(j, len(times) - 1)
        if below[i:j + 1].all():
            return float(times[i])
    return float("inf")


def main():
    pre = json.load(open(PREREG))
    z = np.load(RAW, allow_pickle=True)
    method = np.array([str(m) for m in z["method"]])
    init = np.array([str(i) for i in z["init"]])
    seed = z["seed"].astype(int)
    n_boot = pre["bootstrap"]["n_resamples"]
    bseed = pre["bootstrap"]["seed"]

    rows = {}
    for i in range(len(method)):
        rows.setdefault((init[i], seed[i]), {})[method[i]] = i
    pairs = sorted(rows)
    assert all(set(v) == {"abf", "fr_uniform"} for v in rows.values()), "unpaired rows"

    t = z["t"][0]
    summary = {"prereg": os.path.relpath(PREREG, ROOT), "n_pairs": len(pairs),
               "per_init": {}, "pooled": {}}
    csv_lines = ["init,seed,int_lf_abf,int_lf_uni,d_int_pct,final_abf,final_uni,"
                 "d_final_pct,frozen_abf_kT,frozen_uni_kT,min_ess_frac_uni,max_wmax_uni"]

    def contrast(sel_pairs, label):
        d_int, d_fin, d_froz = [], [], []
        ess_min, wmax_max = [], []
        for key in sel_pairs:
            ia, iu = rows[key]["abf"], rows[key]["fr_uniform"]
            a, u = float(z["int_l2_f"][ia]), float(z["int_l2_f"][iu])
            fa, fu = float(z["final_l2_f"][ia]), float(z["final_l2_f"][iu])
            za, zu = float(z["frozen_l2_f_kT"][ia]), float(z["frozen_l2_f_kT"][iu])
            d_int.append(100.0 * (u - a) / a)
            d_fin.append(100.0 * (fu - fa) / fa)
            d_froz.append(100.0 * (zu - za) / za)
            ess_min.append(float(z["min_ess_frac"][iu]))
            wmax_max.append(float(z["max_wmax"][iu]))
            csv_lines.append(f"{key[0]},{key[1]},{a:.5f},{u:.5f},{d_int[-1]:.3f},"
                             f"{fa:.5f},{fu:.5f},{d_fin[-1]:.3f},{za:.5f},{zu:.5f},"
                             f"{ess_min[-1]:.4f},{wmax_max[-1]:.4f}")
        lo, hi = boot_median(d_int, n_boot, bseed)
        flo, fhi = boot_median(d_fin, n_boot, bseed + 1)
        zlo, zhi = boot_median(d_froz, n_boot, bseed + 2)

        # time-to-accuracy on the per-arm median curves (atlas convention)
        curves = {}
        for m in ("abf", "fr_uniform"):
            idxs = [rows[k][m] for k in sel_pairs]
            curves[m] = np.median(z["l2_f_t"][idxs], axis=0)
        e0 = float(curves["abf"][0])
        eps_list = {f"e0/{int(1 / f)}": e0 * f for f in FRACTIONS}
        eps_list["abf_final"] = float(curves["abf"][-1])
        speedups = {}
        for name, eps in eps_list.items():
            ta = tau(t, curves["abf"], eps, PERSIST)
            tu = tau(t, curves["fr_uniform"], eps, PERSIST)
            speedups[name] = dict(eps=eps, tau_abf=ta, tau_uni=tu,
                                  speedup=(ta / tu if np.isfinite(ta) and np.isfinite(tu)
                                           and tu > 0 else None),
                                  status=("ok" if np.isfinite(ta) and np.isfinite(tu) else
                                          "abf_never" if not np.isfinite(ta) and np.isfinite(tu) else
                                          "arm_never" if np.isfinite(ta) else "neither"))

        rule = pre["success_rule"]
        med = float(np.median(d_int))
        med_fin = float(np.median(d_fin))
        # The frozen confirmatory convention gates on the MEDIAN across seeds of the
        # per-run min ESS fraction / max lineage share (analyze_gateway_confirm.py:149).
        # The worst single seed is reported alongside, not silently substituted.
        ess_med, wmax_med = float(np.median(ess_min)), float(np.median(wmax_max))
        health_ok = (ess_med >= rule["ess_anc_over_N_min"]
                     and wmax_med <= rule["wmax_max"])
        accel = med <= rule["median_rel_change_pct_max"] and hi < rule["ci95_upper_pct_max"]
        safe = accel and med_fin <= rule["final_noninferiority_margin_pct"] and health_ok
        neutral = (abs(med) < abs(rule["median_rel_change_pct_max"])
                   and med_fin <= rule["final_noninferiority_margin_pct"])
        verdict = ("SAFE_ACCELERATOR" if safe else
                   "ACCELERATION_POSITIVE" if accel else
                   "NEUTRAL" if neutral else "NEGATIVE_OR_UNSAFE")
        return dict(
            n=len(sel_pairs),
            d_int_pct=dict(median=med, ci95=[lo, hi],
                           wins=int(np.sum(np.array(d_int) < 0))),
            d_final_pct=dict(median=med_fin, ci95=[flo, fhi],
                             wins=int(np.sum(np.array(d_fin) < 0))),
            d_frozen_pct=dict(median=float(np.median(d_froz)), ci95=[zlo, zhi],
                              wins=int(np.sum(np.array(d_froz) < 0))),
            health=dict(median_min_ess_frac=ess_med, median_max_wmax=wmax_med,
                        worst_min_ess_frac=min(ess_min), worst_max_wmax=max(wmax_max),
                        ok=bool(health_ok)),
            time_to_accuracy=speedups, verdict=verdict)

    for ini in sorted(set(init)):
        sel = [k for k in pairs if k[0] == ini]
        summary["per_init"][ini] = contrast(sel, ini)
    summary["pooled"] = contrast(pairs, "pooled")

    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(os.path.join(OUT, "comparison.csv"), "w") as fh:
        fh.write("\n".join(csv_lines) + "\n")

    p = summary["pooled"]
    print(f"gateway uniform-FR: n={p['n']} pairs")
    print(f"  Delta I_F   median {p['d_int_pct']['median']:+.2f}%  "
          f"CI95 [{p['d_int_pct']['ci95'][0]:+.2f}, {p['d_int_pct']['ci95'][1]:+.2f}]  "
          f"wins {p['d_int_pct']['wins']}/{p['n']}")
    print(f"  Delta e_F(T) median {p['d_final_pct']['median']:+.2f}%  "
          f"CI95 [{p['d_final_pct']['ci95'][0]:+.2f}, {p['d_final_pct']['ci95'][1]:+.2f}]  "
          f"wins {p['d_final_pct']['wins']}/{p['n']}")
    print(f"  frozen-bias  median {p['d_frozen_pct']['median']:+.2f}%  "
          f"CI95 [{p['d_frozen_pct']['ci95'][0]:+.2f}, {p['d_frozen_pct']['ci95'][1]:+.2f}]")
    print(f"  health (frozen convention, median across seeds): "
          f"min ESS/N {p['health']['median_min_ess_frac']:.3f}, "
          f"max wmax {p['health']['median_max_wmax']:.4f}, ok={p['health']['ok']} "
          f"(worst seed: {p['health']['worst_min_ess_frac']:.3f}/"
          f"{p['health']['worst_max_wmax']:.4f})")
    for name, s in p["time_to_accuracy"].items():
        sp = f"{s['speedup']:.2f}x" if s["speedup"] else s["status"]
        print(f"  tau[{name}]: abf {s['tau_abf']:.1f} vs uni {s['tau_uni']:.1f} -> {sp}")
    print(f"  VERDICT: {p['verdict']}")
    for ini, s in summary["per_init"].items():
        print(f"  [{ini}] dI_F {s['d_int_pct']['median']:+.2f}% "
              f"CI [{s['d_int_pct']['ci95'][0]:+.2f}, {s['d_int_pct']['ci95'][1]:+.2f}] "
              f"wins {s['d_int_pct']['wins']}/{s['n']}; final {s['d_final_pct']['median']:+.2f}%; "
              f"verdict {s['verdict']}")


if __name__ == "__main__":
    main()
