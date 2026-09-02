#!/usr/bin/env python
"""Analyzer for the gateway corrected-baseline CONFIRMATION -- written BEFORE its data.

Prereg: configs/information_campaign/gateway_corrected_confirmation_prereg.json.
Primary: paired Delta I_F (fr_uniform vs abf) at h_read* (from step 1), the campaign's frozen
rule (median <= -10%, CI95 upper < 0; SAFE if final <= +5% and the genealogy floors hold under
the frozen median-across-seeds convention).  Alongside: the legacy 0.07 read-out (the closed
study's convention, i.e. the replication of -11.8% / +9.8%), raw bins, the frozen-bias endpoint,
time-to-accuracy, and the error-ratio time course (the reversal signature).

    python scripts/analyze_gateway_corrected_confirmation.py [--raw PATH]
    python scripts/analyze_gateway_corrected_confirmation.py --raw <synthetic two-method npz>   # machinery smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
ROOT = os.path.join(SCRIPTS, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from analyze_uniform_lta import BOOT_SEED, boot_median                       # noqa: E402
from analyze_uniform_gateway import tau, PERSIST, FRACTIONS                  # noqa: E402
from analyze_gateway_bandwidth_audit import mean_force_at, e_f, label        # noqa: E402
from eb_abffr_core import EVAL_LO, EVAL_HI                                   # noqa: E402

PREREG = os.path.join(ROOT, "configs/information_campaign/gateway_corrected_confirmation_prereg.json")
STEP1 = os.path.join(ROOT, "results/information_campaign/gateway_baseline_audit/analysis.json")
DEFAULT_RAW = os.path.join(ROOT, "results/information_campaign/gateway_corrected_confirmation/raw.npz")
CLOSED = dict(d_int=-11.81, d_int_ci=(-14.04, -9.02), d_fin=+9.82, d_fin_ci=(+8.20, +11.74))   # closed campaign, legacy read-out
RATIO_TIMES = (2, 5, 10, 17, 20, 30, 40)


def paired(arm, ref, k):
    d = 100.0 * (arm - ref) / ref
    lo, hi = boot_median(d, BOOT_SEED + k)
    return dict(median=float(np.median(d)), ci95=[lo, hi], wins=int((d < 0).sum()), n=int(len(d)),
                per_row=[float(v) for v in d])


def fmt(s):
    return f"{s['median']:+7.2f}% [{s['ci95'][0]:+7.2f},{s['ci95'][1]:+7.2f}] {s['wins']:2d}/{s['n']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    pre = json.load(open(PREREG)) if os.path.exists(PREREG) else None
    step1 = json.load(open(STEP1))
    h_star = float(step1["h_read_star"])
    rule = (pre or {}).get("success_rule", dict(median_rel_change_pct_max=-10.0, ci95_upper_pct_max=0.0,
                                                 final_noninferiority_margin_pct=5.0))
    floors = (pre or {}).get("health_floors", dict(ess_anc_over_N_min=0.30, wmax_max=0.05))
    smoke = os.path.abspath(a.raw) != os.path.abspath(DEFAULT_RAW)
    if smoke:
        print("*** MACHINERY SMOKE on a non-confirmation file -- the numbers below are NOT a result ***")

    z = np.load(a.raw, allow_pickle=True)
    method = np.array([str(m) for m in z["method"]])
    init = np.array([str(i) for i in z["init"]])
    seed = z["seed"].astype(int)
    rows = {}
    for i in range(len(method)):
        rows.setdefault((init[i], seed[i]), {})[method[i]] = i
    pairs = sorted(k for k, v in rows.items() if set(v) >= {"abf", "fr_uniform"})
    assert pairs, "no complete pairs"
    ia = np.array([rows[k]["abf"] for k in pairs]); iu = np.array([rows[k]["fr_uniform"] for k in pairs])
    x = np.asarray(z["x_grid"][0], float); dx = float(x[1] - x[0]); mask = (x >= EVAL_LO) & (x <= EVAL_HI)
    t = np.asarray(z["t"][0], float)
    cfg0 = json.loads(str(z["config_json"][0]))
    h_bias, min_count = float(cfg0["h"]), float(cfg0["min_count"])
    inits = sorted(set(init))
    print(f"pairs: {len(pairs)} ({', '.join(f'{i}: {sum(1 for k in pairs if k[0] == i)}' for i in inits)}); "
          f"h_bias {h_bias:g}; h_read* {label(h_star)}; gamma {float(z['gamma'][iu[0]]):g}")

    # ---- offline read-outs from the raw accumulators; self-check against the engine at h_bias ----
    Sf, C, F_ref = np.asarray(z["Sf_t"], float), np.asarray(z["C_t"], float), np.asarray(z["F_ref"], float)
    ladder = sorted({h_bias, h_star, 0.07, 0.0}, reverse=True)
    ro = {label(h): e_f(mean_force_at(Sf, C, h, dx, min_count), F_ref, dx, mask) for h in ladder}
    dev = float(np.abs(ro[label(h_bias)] - np.asarray(z["l2_f_t"], float)).max())
    print(f"self-check: offline read-out at h_bias vs engine l2_f_t, max|dev| {dev:.2e}")
    assert dev < 1e-9

    # ---- contrasts at every read-out ----
    res = {}
    print(f"\n  {'read-out':>8} {'ABF e_F(T)':>10} {'d I_F':>8} {'CI95':>20} {'wins':>6} {'d e_F(T)':>9} {'CI95':>20}")
    for k, h in enumerate(ladder):
        lab = label(h)
        I = np.trapezoid(ro[lab], t, axis=1); fin = ro[lab][:, -1]
        res[lab] = dict(h=h, d_int=paired(I[iu], I[ia], 10 * k), d_fin=paired(fin[iu], fin[ia], 10 * k + 1),
                        abf_eF_T_median=float(np.median(fin[ia])), fr_eF_T_median=float(np.median(fin[iu])),
                        per_init={ini: dict(d_int=float(np.median([100 * (I[iu][j] - I[ia][j]) / I[ia][j] for j, kk in enumerate(pairs) if kk[0] == ini])),
                                            d_fin=float(np.median([100 * (fin[iu][j] - fin[ia][j]) / fin[ia][j] for j, kk in enumerate(pairs) if kk[0] == ini])))
                                  for ini in inits})
        tag = ("  <- PRIMARY (h_read*)" if abs(h - h_star) < 1e-12 else "") + ("  <- legacy (closed-study convention)" if abs(h - 0.07) < 1e-12 else "")
        print(f"  {lab:>8} {res[lab]['abf_eF_T_median']:10.5f} {res[lab]['d_int']['median']:+8.2f} [{res[lab]['d_int']['ci95'][0]:+8.2f},{res[lab]['d_int']['ci95'][1]:+8.2f}]"
              f" {res[lab]['d_int']['wins']:3d}/{len(pairs)} {res[lab]['d_fin']['median']:+9.2f} [{res[lab]['d_fin']['ci95'][0]:+8.2f},{res[lab]['d_fin']['ci95'][1]:+8.2f}]{tag}")
        print("           per-init: " + "  ".join(f"{i}: dI_F {v['d_int']:+.2f}%, final {v['d_fin']:+.2f}%" for i, v in res[lab]["per_init"].items()))

    # ---- reversal signature: error-ratio time course (median over pairs) ----
    ratio = {}
    for lab in (label(h_star), "0.07"):
        r = np.median(ro[lab][iu] / ro[lab][ia], axis=0)
        idx = [int(np.argmin(abs(t - v))) for v in RATIO_TIMES]
        ratio[lab] = dict(t=[float(t[i]) for i in idx], uni_over_abf=[float(r[i]) for i in idx], final=float(r[-1]),
                          crosses_one_after=bool((r[idx[1]:] > 1.0).any()))
        print(f"  ratio uni/abf [{lab}] at t={RATIO_TIMES}: {np.round(ratio[lab]['uni_over_abf'], 3).tolist()} final {r[-1]:.3f}"
              f" {'(crosses 1 -> reversal)' if ratio[lab]['crosses_one_after'] else '(never crosses 1)'}")

    # ---- frozen bias, genealogy, time-to-accuracy ----
    froz = None
    if "frozen_l2_f_kT" in z.files:
        fz = np.asarray(z["frozen_l2_f_kT"], float)
        froz = paired(fz[iu], fz[ia], 77)
        print(f"  frozen-bias endpoint: {fmt(froz)}")
    ess = np.asarray(z["min_ess_frac"], float)[iu]; wmax = np.asarray(z["max_wmax"], float)[iu]
    ess_med, wmax_med = float(np.median(ess)), float(np.median(wmax))
    health_ok = ess_med >= floors["ess_anc_over_N_min"] and wmax_med <= floors["wmax_max"]
    print(f"  genealogy (median convention): min ESS/N {ess_med:.3f} (floor {floors['ess_anc_over_N_min']}), max wmax {wmax_med:.4f} "
          f"(cap {floors['wmax_max']}) -> ok={health_ok}; worst row {ess.min():.3f}/{wmax.max():.4f}")
    curves = {m: np.median(ro[label(h_star)][idx], axis=0) for m, idx in (("abf", ia), ("fr_uniform", iu))}
    e0 = float(curves["abf"][0])
    eps_list = {f"e0/{int(1 / f)}": e0 * f for f in FRACTIONS}; eps_list["abf_final"] = float(curves["abf"][-1])
    speed = {}
    for nm, eps in eps_list.items():
        ta, tu = tau(t, curves["abf"], eps, PERSIST), tau(t, curves["fr_uniform"], eps, PERSIST)
        speed[nm] = dict(eps=eps, tau_abf=ta, tau_fr=tu, speedup=(ta / tu if np.isfinite(ta) and np.isfinite(tu) and tu > 0 else None))
        print(f"  tau[{nm}] @ h_read*: abf {ta:.1f} vs fr {tu:.1f} -> {(f'{ta / tu:.2f}x' if speed[nm]['speedup'] else 'censored')}")

    # ---- verdicts and outcome ----
    def verdict(lab):
        di, df = res[lab]["d_int"], res[lab]["d_fin"]
        accel = di["median"] <= rule["median_rel_change_pct_max"] and di["ci95"][1] < rule["ci95_upper_pct_max"]
        safe = accel and df["median"] <= rule["final_noninferiority_margin_pct"] and health_ok
        neutral = abs(di["median"]) < abs(rule["median_rel_change_pct_max"]) and df["median"] <= rule["final_noninferiority_margin_pct"]
        return "SAFE_ACCELERATOR" if safe else "ACCELERATION_POSITIVE" if accel else "NEUTRAL" if neutral else "NEGATIVE_OR_UNSAFE"
    v_star, v_leg = verdict(label(h_star)), verdict("0.07")
    P = res[label(h_star)]
    if not health_ok:
        outcome = "G4_unsafe"
    elif v_star == "SAFE_ACCELERATOR" and P["d_fin"]["ci95"][1] < 0:
        outcome = "G1_persistent_positive"
    elif P["d_int"]["ci95"][1] < 0 and (P["d_fin"]["median"] > rule["final_noninferiority_margin_pct"] or P["d_fin"]["ci95"][0] > 0):
        outcome = "G2_transient_with_reversal"
    elif P["d_int"]["ci95"][1] < 0:
        outcome = "G1b_positive_final_unresolved"
    else:
        outcome = "G3_null"
    print(f"\n  verdict at h_read* {label(h_star)}: {v_star};  at legacy 0.07: {v_leg}")
    print(f"  closed study at legacy read-out: dI_F {CLOSED['d_int']:+.2f} {CLOSED['d_int_ci']}, final {CLOSED['d_fin']:+.2f} {CLOSED['d_fin_ci']}")
    print(f"  OUTCOME: {outcome}{'   [SMOKE -- not a result]' if smoke else ''}")
    if smoke:
        return
    out_dir = a.out_dir or os.path.dirname(a.raw)
    summary = dict(prereg=os.path.relpath(PREREG, ROOT), step1=os.path.relpath(STEP1, ROOT), raw=os.path.relpath(a.raw, ROOT),
                   n_pairs=len(pairs), h_bias=h_bias, h_read_star=h_star, gamma=float(z["gamma"][iu[0]]),
                   per_readout=res, error_ratio=ratio, frozen_bias=froz,
                   health=dict(median_min_ess_frac=ess_med, median_max_wmax=wmax_med, worst_min_ess_frac=float(ess.min()),
                               worst_max_wmax=float(wmax.max()), ok=bool(health_ok), floors=floors),
                   time_to_accuracy=speed, verdict_h_read_star=v_star, verdict_legacy=v_leg, closed_study_legacy=CLOSED, outcome=outcome)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(os.path.join(out_dir, "comparison.csv"), "w") as fh:
        labs = [label(h) for h in ladder]
        fh.write("init,seed," + ",".join(f"d_int_{l},d_fin_{l}" for l in labs) + ",min_ess_frac_fr,wmax_fr" + (",d_frozen" if froz else "") + "\n")
        for j, k in enumerate(pairs):
            fh.write(f"{k[0]},{k[1]}," + ",".join(f"{res[l]['d_int']['per_row'][j]:.3f},{res[l]['d_fin']['per_row'][j]:.3f}" for l in labs)
                     + f",{ess[j]:.4f},{wmax[j]:.4f}" + (f",{froz['per_row'][j]:.3f}" if froz else "") + "\n")
    print(f"  wrote {os.path.relpath(out_dir, ROOT)}/summary.json, comparison.csv")


if __name__ == "__main__":
    main()
