#!/usr/bin/env python
"""Analyzer for the transport campaign (gateway horizontal OT) -- written BEFORE its data.

Prereg: configs/transport_campaign/gateway_horizontal_transport_prereg.json.

PRIMARY: Delta_{OT|FR} = per-row relative change of I_F (ot_matched vs fr_uniform) at h_read*,
cluster bootstrap by seed (both inits of a drawn seed travel together), 10000 resamples;
decision OT_better / FR_better / equivalent (TOST, 90% CI inside +-10%) / INCONCLUSIVE, with a
heterogeneity check per init.  SECONDARY: every arm vs abf (accelerator / SAFE / NEGATIVE /
NEUTRAL), ot_full vs ot_matched and vs fr_uniform, the legacy 0.07 and raw-bin read-outs,
I_F' , the frozen-bias endpoint, time-to-accuracy, and the mechanism records (J_KL, int D_cond,
D_move, genealogy).  Outcomes H1-H5 / INCONCLUSIVE exactly as frozen.

Precedence for the vs-abf verdict (fixed here, before data): SAFE_ACCELERATOR if the accelerator
rule holds and the final median is non-inferior (<= +5%; FR also needs the genealogy floors);
else ACCELERATION_POSITIVE (with a reversal flag if the final CI95 lower > 0); else NEGATIVE if
the integrated or final CI95 lower > 0; else NEUTRAL.

    python scripts/analyze_gateway_horizontal_transport.py [--raw PATH] [--no-figures]
    python scripts/analyze_gateway_horizontal_transport.py --raw <synthetic npz>   # machinery smoke
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
from analyze_uniform_gateway import tau, PERSIST, FRACTIONS                  # noqa: E402
from analyze_gateway_bandwidth_audit import mean_force_at, e_f, label        # noqa: E402
from eb_abffr_core import EVAL_LO, EVAL_HI                                   # noqa: E402

PREREG = os.path.join(ROOT, "configs/transport_campaign/gateway_horizontal_transport_prereg.json")
STEP1 = os.path.join(ROOT, "results/information_campaign/gateway_baseline_audit/analysis.json")
CAMPAIGN = os.path.join(ROOT, "results/transport_campaign/gateway_horizontal")
DEFAULT_RAW = os.path.join(CAMPAIGN, "production", "raw.npz")
CAL_RAW = os.path.join(CAMPAIGN, "calibration", "raw.npz")
CAL_SEL = os.path.join(CAMPAIGN, "calibration", "alpha_selection.json")
N_BOOT, BOOT_SEED = 10000, 20260902
MARGIN, FINAL_MARGIN = 10.0, 5.0
FLOORS = dict(ess_anc_over_N_min=0.30, wmax_max=0.05)
ARMS = ("abf", "fr_uniform", "ot_matched", "ot_full")
RATIO_TIMES = (2, 5, 10, 17, 20, 30, 40)
T_DOSE_START = 4.0


# ----------------------------------------------------------------------------- statistics
def boot_cluster(d, clusters, seed, n_boot=N_BOOT):
    """Percentile bootstrap of the median, resampling CLUSTERS (seed labels) with replacement.

    Every cluster must have the same size (2 rows per seed pooled; 1 row per seed per init)."""
    d, clusters = np.asarray(d, float), np.asarray(clusters)
    uniq = np.unique(clusters)
    members = [np.nonzero(clusters == c)[0] for c in uniq]
    m = len(members[0])
    assert all(len(x) == m for x in members), "unbalanced clusters"
    idx = np.array(members)                                   # (K, m)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    med = np.median(d[idx[draws].reshape(n_boot, -1)], axis=1)
    return ([float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))],
            [float(np.percentile(med, 5.0)), float(np.percentile(med, 95.0))])


def contrast(arm, ref, clusters, k):
    d = 100.0 * (arm - ref) / ref
    ci95, ci90 = boot_cluster(d, clusters, BOOT_SEED + k)
    return dict(median=float(np.median(d)), ci95=ci95, ci90=ci90, wins=int((d < 0).sum()),
                n=int(len(d)), per_row=[float(v) for v in d])


def decide_primary(c):
    m, (lo, hi), (lo90, hi90) = c["median"], c["ci95"], c["ci90"]
    if m <= -MARGIN and hi < 0:
        return "OT_better"
    if m >= MARGIN and lo > 0:
        return "FR_better"
    if lo90 > -MARGIN and hi90 < MARGIN:
        return "equivalent"
    return "INCONCLUSIVE"


def verdict_vs_abf(d_int, d_fin, health_ok=True):
    accel = d_int["median"] <= -MARGIN and d_int["ci95"][1] < 0
    reversal = d_fin["ci95"][0] > 0
    if accel and d_fin["median"] <= FINAL_MARGIN and health_ok:
        return "SAFE_ACCELERATOR"
    if accel:
        return "ACCELERATION_POSITIVE" + ("_WITH_REVERSAL" if reversal else "")
    if d_int["ci95"][0] > 0 or reversal:
        return "NEGATIVE"
    return "NEUTRAL"


def fmt(c):
    return f"{c['median']:+7.2f}% [{c['ci95'][0]:+7.2f},{c['ci95'][1]:+7.2f}] {c['wins']:2d}/{c['n']}"


# ----------------------------------------------------------------------------- figures
def make_figures(out_dir, t, curves, curves_fp, kl_med, dcond_med, cal, h_star_lab):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  (figures skipped: {e})")
        return []
    os.makedirs(out_dir, exist_ok=True)
    col = dict(abf="#4d4d4d", fr_uniform="#d95f02", ot_matched="#1b9e77", ot_full="#7570b3")
    lab = dict(abf="ABF", fr_uniform="ABF + uniform FR", ot_matched="ABF + matched horizontal OT",
               ot_full="ABF + full horizontal OT (alpha = 1)")
    made = []

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    # B: free-energy convergence
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in ARMS:
        ax.plot(t, curves[m], color=col[m], lw=1.6, label=lab[m])
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel(f"median $e_F(t)$ at $h_{{read}}$ = {h_star_lab}")
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25, which="both")
    save(fig, "fig_B_convergence_eF")
    # C: marginal vs conditional
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 5.6), sharex=True)
    for m in ARMS:
        axes[0].plot(t, kl_med[m], color=col[m], lw=1.6, label=lab[m])
        axes[1].plot(t, dcond_med[m], color=col[m], lw=1.6)
    axes[0].set_yscale("log"); axes[0].set_ylabel(r"median KL$(\hat p_t^x \| U)$")
    axes[1].set_yscale("log"); axes[1].set_ylabel(r"median $D_{\rm cond}(t)$"); axes[1].set_xlabel("t")
    axes[0].legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    save(fig, "fig_C_marginal_vs_conditional")
    # D: mean-force error
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in ARMS:
        ax.plot(t, curves_fp[m], color=col[m], lw=1.6, label=lab[m])
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel(f"median $e_{{F'}}(t)$ at $h_{{read}}$ = {h_star_lab}")
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25, which="both")
    save(fig, "fig_D_mean_force_error")
    # E: calibration Pareto (marginal benefit vs conditional damage)
    if cal is not None:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for name, d in cal.items():
            c = col.get(name, "#1b9e77" if name.startswith("ot_") else "#4d4d4d")
            ax.scatter(d["J_KL"], d["int_Dcond"], color=c, s=28, zorder=3)
            ax.annotate(name.replace("ot_", r"$\alpha$="), (d["J_KL"], d["int_Dcond"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$J_{\rm KL} = \int_4^{40} {\rm KL}(\hat p_t^x\|U)\,dt$ (marginal action)")
        ax.set_ylabel(r"$\int_0^{40} D_{\rm cond}(t)\,dt$ (conditional damage)")
        ax.grid(alpha=0.25, which="both")
        save(fig, "fig_E_calibration_pareto")
    return made


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    step1 = json.load(open(STEP1))
    h_star = float(step1["h_read_star"])
    smoke = os.path.abspath(a.raw) != os.path.abspath(DEFAULT_RAW)
    if smoke:
        print("*** MACHINERY SMOKE on a non-production file -- the numbers below are NOT a result ***")

    z = np.load(a.raw, allow_pickle=True)
    method = np.array([str(m) for m in z["method"]])
    init = np.array([str(i) for i in z["init"]])
    seed = z["seed"].astype(int)
    rows = {}
    for i in range(len(method)):
        rows.setdefault((init[i], seed[i]), {})[method[i]] = i
    keys = sorted(k for k, v in rows.items() if set(v) >= set(ARMS))
    assert keys, "no complete rows"
    idx = {m: np.array([rows[k][m] for k in keys]) for m in ARMS}
    clusters = np.array([k[1] for k in keys])
    init_of = np.array([k[0] for k in keys])
    inits = sorted(str(i) for i in set(init_of))
    x = np.asarray(z["x_grid"][0], float); dx = float(x[1] - x[0]); mask = (x >= EVAL_LO) & (x <= EVAL_HI)
    t = np.asarray(z["t"][0], float)
    cfg0 = json.loads(str(z["config_json"][0]))
    h_bias, min_count = float(cfg0["h"]), float(cfg0["min_count"])
    alpha_star = float(z["alpha_star"]) if "alpha_star" in z.files else float(z["alpha"][idx["ot_matched"][0]])
    gamma = float(z["gamma"][idx["fr_uniform"][0]])
    n_seeds = len(set(clusters))
    print(f"rows: {len(keys)} ({', '.join(f'{i}: {int((init_of == i).sum())}' for i in inits)}), seeds {n_seeds}; "
          f"h_bias {h_bias:g}; h_read* {label(h_star)}; gamma {gamma:g}; alpha* {alpha_star:g}")

    # ---- offline read-outs, self-check ----
    Sf, C, F_ref, Fp_ref = (np.asarray(z[k], float) for k in ("Sf_t", "C_t", "F_ref", "Fp_ref"))
    ladder = sorted({h_star, 0.07, 0.0, h_bias}, reverse=True)
    ro = {label(h): e_f(mean_force_at(Sf, C, h, dx, min_count), F_ref, dx, mask) for h in ladder}
    dev = float(np.abs(ro[label(h_bias)] - np.asarray(z["l2_f_t"], float)).max())
    print(f"self-check: offline read-out at h_bias vs engine l2_f_t, max|dev| {dev:.2e}")
    assert dev < 1e-9
    Fp_star = mean_force_at(Sf, C, h_star, dx, min_count)
    efp = np.sqrt(np.mean(((Fp_star - Fp_ref[:, None, :])[..., mask]) ** 2, axis=-1))       # e_F'(t) at h*
    I_Fp = np.trapezoid(efp, t, axis=1)
    hs = label(h_star)

    # ---- per-read-out metrics ----
    I = {lab: np.trapezoid(ro[lab], t, axis=1) for lab in ro}
    fin = {lab: ro[lab][:, -1] for lab in ro}

    def per_init(fn):
        return {ini: fn(np.nonzero(init_of == ini)[0]) for ini in inits}

    res = {}
    k = 0
    print(f"\n{'read-out':>8} {'contrast':>22} {'d I_F':>8} {'CI95':>20} {'wins':>6} {'d e_F(T)':>9} {'CI95':>20}")
    report_labels = [label(h) for h in (h_star, 0.07, 0.0)]
    pairs = [("ot_matched", "fr_uniform"), ("fr_uniform", "abf"), ("ot_matched", "abf"), ("ot_full", "abf"),
             ("ot_full", "ot_matched"), ("ot_full", "fr_uniform")]
    for lab in report_labels:
        res[lab] = dict(h=(0.0 if lab == "raw" else float(lab)),
                        abf_eF_T_median=float(np.median(fin[lab][idx["abf"]])),
                        eF_T_median={m: float(np.median(fin[lab][idx[m]])) for m in ARMS},
                        I_F_median={m: float(np.median(I[lab][idx[m]])) for m in ARMS}, contrasts={})
        for arm, ref in pairs:
            ia, ir = idx[arm], idx[ref]
            c_int = contrast(I[lab][ia], I[lab][ir], clusters, k); k += 1
            c_fin = contrast(fin[lab][ia], fin[lab][ir], clusters, k); k += 1
            pi = {}
            for ini in inits:
                sel = np.nonzero(init_of == ini)[0]
                pi[ini] = dict(d_int=contrast(I[lab][ia][sel], I[lab][ir][sel], clusters[sel], k),
                               d_fin=contrast(fin[lab][ia][sel], fin[lab][ir][sel], clusters[sel], k + 1))
                k += 2
            res[lab]["contrasts"][f"{arm}|{ref}"] = dict(d_int=c_int, d_fin=c_fin, per_init=pi)
            tag = "  <- PRIMARY" if (lab == hs and (arm, ref) == pairs[0]) else ""
            print(f"{lab:>8} {arm + ' vs ' + ref:>22} {c_int['median']:+8.2f} [{c_int['ci95'][0]:+8.2f},{c_int['ci95'][1]:+8.2f}]"
                  f" {c_int['wins']:3d}/{c_int['n']} {c_fin['median']:+9.2f} [{c_fin['ci95'][0]:+8.2f},{c_fin['ci95'][1]:+8.2f}]{tag}")
            if lab == hs:
                print("         per-init: " + "  ".join(f"{i}: dI_F {v['d_int']['median']:+.2f}% [{v['d_int']['ci95'][0]:+.1f},{v['d_int']['ci95'][1]:+.1f}], "
                                                        f"final {v['d_fin']['median']:+.2f}%" for i, v in pi.items()))

    # ---- secondary endpoint I_F' at h*, frozen bias ----
    sec = {}
    for arm, ref in pairs[:4]:
        sec[f"{arm}|{ref}"] = contrast(I_Fp[idx[arm]], I_Fp[idx[ref]], clusters, k); k += 1
    print("\nI_F' at h_read*: " + "; ".join(f"{p}: {fmt(c)}" for p, c in sec.items()))
    froz = None
    if "frozen_l2_f_kT" in z.files:
        fz = np.asarray(z["frozen_l2_f_kT"], float)
        froz = {f"{arm}|{ref}": contrast(fz[idx[arm]], fz[idx[ref]], clusters, k + j) for j, (arm, ref) in enumerate(pairs)}
        k += len(pairs)
        print("frozen-bias endpoint: " + "; ".join(f"{p}: {fmt(c)}" for p, c in froz.items()))

    # ---- mechanism records ----
    kl = np.asarray(z["kl_uniform_t"], float); dc = np.nan_to_num(np.asarray(z["dcond_t"], float))
    md = t >= T_DOSE_START - 1e-12
    J = np.trapezoid(kl[:, md], t[md], axis=1); Dc = np.trapezoid(dc, t, axis=1)
    mech = {}
    for m in ARMS:
        r = idx[m]
        d = dict(median_J_KL=float(np.median(J[r])), median_int_Dcond=float(np.median(Dc[r])),
                 median_final_P_plus=float(np.median(np.asarray(z["P_regions"], float)[r, -1, 2])),
                 median_final_P_gate=float(np.median(np.asarray(z["P_regions"], float)[r, -1, 1])))
        if m.startswith("ot_"):
            d.update(median_mean_absdx=float(np.median(np.mean(np.asarray(z["ot_absdx_t"], float)[r][:, 1:], axis=1))),
                     median_mean_Dmove=float(np.median(np.mean(np.asarray(z["dmove_mean_t"], float)[r][:, 1:], axis=1))),
                     median_p95_Dmove=float(np.median(np.mean(np.asarray(z["dmove_p95_t"], float)[r][:, 1:], axis=1))),
                     median_max_Dmove=float(np.median(np.max(np.asarray(z["dmove_max_t"], float)[r], axis=1))))
        if m == "fr_uniform":
            ess = np.asarray(z["min_ess_frac"], float)[r]; wmax = np.asarray(z["max_wmax"], float)[r]
            d.update(median_min_ess_frac=float(np.median(ess)), median_max_wmax=float(np.median(wmax)),
                     worst_min_ess_frac=float(ess.min()), worst_max_wmax=float(wmax.max()),
                     median_repl_fraction=float(np.median(np.asarray(z["repl_fraction"], float)[r])))
        mech[m] = d
        rel = d['median_J_KL'] / mech['abf']['median_J_KL'] if mech['abf']['median_J_KL'] > 0 else float('nan')
        print(f"  {m:>10}: J_KL {d['median_J_KL']:.4f} (x{rel:.3f} abf)  "
              f"int D_cond {d['median_int_Dcond']:.4f}  P+ final {d['median_final_P_plus']:.3f} gate {d['median_final_P_gate']:.3f}"
              + (f"  |dx|/event {d['median_mean_absdx']:.2e}  D_move mean {d['median_mean_Dmove']:.2e} p95 {d['median_p95_Dmove']:.2e} max {d['median_max_Dmove']:.2e}" if m.startswith("ot_") else "")
              + (f"  ESS/N {d['median_min_ess_frac']:.3f} wmax {d['median_max_wmax']:.4f} repl {d['median_repl_fraction']:.4f}" if m == "fr_uniform" else ""))
    health_ok = (mech["fr_uniform"]["median_min_ess_frac"] >= FLOORS["ess_anc_over_N_min"]
                 and mech["fr_uniform"]["median_max_wmax"] <= FLOORS["wmax_max"])
    ratio_J = (mech["ot_matched"]["median_J_KL"] / mech["fr_uniform"]["median_J_KL"]
               if mech["fr_uniform"]["median_J_KL"] > 0 else float("nan"))
    print(f"  dose check on production rows: J_KL(ot_matched)/J_KL(fr) = {ratio_J:.3f}  (calibration matched to [0.9, 1.1])")

    # ---- error-ratio time courses, time-to-accuracy ----
    curves = {m: np.median(ro[hs][idx[m]], axis=0) for m in ARMS}
    curves_fp = {m: np.median(efp[idx[m]], axis=0) for m in ARMS}
    ratio = {}
    ti = [int(np.argmin(abs(t - v))) for v in RATIO_TIMES]
    for m in ARMS[1:]:
        r = np.median(ro[hs][idx[m]] / ro[hs][idx["abf"]], axis=0)
        ratio[m] = dict(t=[float(t[i]) for i in ti], over_abf=[float(r[i]) for i in ti], final=float(r[-1]),
                        crosses_one_after_t5=bool((r[ti[1]:] > 1.0).any()))
        print(f"  ratio {m}/abf [{hs}] at t={RATIO_TIMES}: {np.round(ratio[m]['over_abf'], 3).tolist()} final {r[-1]:.3f}"
              f" {'(crosses 1 after t=5)' if ratio[m]['crosses_one_after_t5'] else '(never crosses 1 after t=5)'}")
    e0 = float(curves["abf"][0])
    eps_list = {f"e0/{int(1 / f)}": e0 * f for f in FRACTIONS}; eps_list["abf_final"] = float(curves["abf"][-1])
    speed = {}
    for nm, eps in eps_list.items():
        ta = tau(t, curves["abf"], eps, PERSIST)
        speed[nm] = dict(eps=eps, tau_abf=ta)
        for m in ARMS[1:]:
            tm = tau(t, curves[m], eps, PERSIST)
            speed[nm][f"tau_{m}"] = tm
            speed[nm][f"speedup_{m}"] = (ta / tm if np.isfinite(ta) and np.isfinite(tm) and tm > 0 else None)
        print(f"  tau[{nm}] @ {hs}: abf {ta:.1f}; " + "; ".join(
            f"{m} {speed[nm][f'tau_{m}']:.1f} ({speed[nm][f'speedup_{m}']:.2f}x)" if speed[nm][f"speedup_{m}"] else f"{m} censored"
            for m in ARMS[1:]))

    # ---- decisions and outcome ----
    P = res[hs]["contrasts"]
    prim = P["ot_matched|fr_uniform"]
    decision = decide_primary(prim["d_int"])
    per_init_dec = {ini: decide_primary(prim["per_init"][ini]["d_int"]) for ini in inits}
    kinds = {decision} | set(per_init_dec.values())
    heterogeneous = any(d != decision and d != "INCONCLUSIVE" and decision != "INCONCLUSIVE" for d in per_init_dec.values())
    verd = {m: verdict_vs_abf(P[f"{m}|abf"]["d_int"], P[f"{m}|abf"]["d_fin"], health_ok if m == "fr_uniform" else True)
            for m in ARMS[1:]}
    replicated = P["fr_uniform|abf"]["d_int"]["ci95"][1] < 0
    if not replicated:
        outcome = "FAILED_REPLICATION_OF_POSITIVE_CONTROL"
    elif decision == "FR_better":
        outcome = "H1_FR_wins"
    elif decision == "OT_better" and verd["ot_matched"] == "SAFE_ACCELERATOR":
        outcome = "H2_OT_wins"
    elif decision == "OT_better":
        outcome = "H2b_OT_better_than_FR_but_not_SAFE_vs_abf"
    elif decision == "equivalent":
        outcome = "H3_equivalent"
    else:
        outcome = "INCONCLUSIVE"
    h4 = verd["ot_matched"].startswith("SAFE") or verd["ot_matched"].startswith("ACCELERATION")
    h4 = bool(h4 and verd["ot_full"] == "NEGATIVE")
    h5 = bool(verd["ot_full"] == "SAFE_ACCELERATOR" and P["ot_full|fr_uniform"]["d_fin"]["median"] <= FINAL_MARGIN)
    print(f"\n  PRIMARY (ot_matched vs fr_uniform, I_F at {hs}): {fmt(prim['d_int'])}  90% CI [{prim['d_int']['ci90'][0]:+.2f}, {prim['d_int']['ci90'][1]:+.2f}]"
          f" -> {decision}; per-init {per_init_dec}{'  ** HETEROGENEOUS **' if heterogeneous else ''}")
    print(f"  vs abf at {hs}: " + ", ".join(f"{m} {v}" for m, v in verd.items()) + f"; FR genealogy ok={health_ok}")
    print(f"  positive control replicated (fr_uniform vs abf CI95 upper < 0): {replicated}")
    print(f"  OUTCOME: {outcome}   H4 strength trade-off: {h4}   H5 full-OT strong: {h5}"
          f"{'   [SMOKE -- not a result]' if smoke else ''}")
    if smoke:
        return

    out_dir = a.out_dir or os.path.dirname(a.raw)
    cal = None
    cal_summary = None
    if os.path.exists(CAL_RAW):
        cz = np.load(CAL_RAW, allow_pickle=True)
        cm = np.array([str(m) for m in cz["method"]]); ct = np.asarray(cz["t"][0], float)
        ckl = np.asarray(cz["kl_uniform_t"], float); cdc = np.nan_to_num(np.asarray(cz["dcond_t"], float))
        cmd = ct >= T_DOSE_START - 1e-12
        cal = {}
        for m in sorted(set(cm), key=lambda s: (not s.startswith("ot_"), s)):
            r = cm == m
            cal[m] = dict(J_KL=float(np.median(np.trapezoid(ckl[r][:, cmd], ct[cmd], axis=1))),
                          int_Dcond=float(np.median(np.trapezoid(cdc[r], ct, axis=1))))
        cal_summary = dict(file=os.path.relpath(CAL_RAW, ROOT), per_arm=cal,
                           selection=(json.load(open(CAL_SEL)) if os.path.exists(CAL_SEL) else None))
        if cal_summary["selection"]:
            cal_summary["selection"].pop("descriptive", None)
    figs = [] if a.no_figures else make_figures(os.path.join(out_dir, "figures"), t, curves, curves_fp,
                                                {m: np.median(kl[idx[m]], axis=0) for m in ARMS},
                                                {m: np.median(dc[idx[m]], axis=0) for m in ARMS}, cal, hs)
    summary = dict(prereg=os.path.relpath(PREREG, ROOT), step1=os.path.relpath(STEP1, ROOT), raw=os.path.relpath(a.raw, ROOT),
                   n_rows=len(keys), n_seeds=n_seeds, h_bias=h_bias, h_read_star=h_star, gamma=gamma, alpha_star=alpha_star,
                   bootstrap=dict(n_resamples=N_BOOT, seed=BOOT_SEED, cluster="seed"),
                   per_readout=res, secondary_I_Fp=sec, frozen_bias=froz, mechanism=mech, dose_ratio_production=ratio_J,
                   health=dict(ok=bool(health_ok), floors=FLOORS), error_ratio=ratio, time_to_accuracy=speed,
                   primary=dict(decision=decision, per_init=per_init_dec, heterogeneous=bool(heterogeneous)),
                   verdict_vs_abf=verd, positive_control_replicated=bool(replicated),
                   outcome=outcome, H4_strength_tradeoff=h4, H5_full_OT_strong=h5, calibration=cal_summary, figures=figs)
    with open(os.path.join(out_dir, "analysis.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(os.path.join(out_dir, "comparison.csv"), "w") as fh:
        cols = [f"{p.replace('|', '_vs_')}_{q}_{lab}" for lab in report_labels for p in res[lab]["contrasts"] for q in ("dI", "dF")]
        fh.write("init,seed," + ",".join(cols) + ",dIFp_ot_matched_vs_fr,min_ess_frac_fr,wmax_fr" + (",dfrozen_ot_matched_vs_fr" if froz else "") + "\n")
        for j, kk in enumerate(keys):
            vals = [f"{res[lab]['contrasts'][p][q]['per_row'][j]:.3f}" for lab in report_labels for p in res[lab]["contrasts"] for q in ("d_int", "d_fin")]
            ess = float(np.asarray(z["min_ess_frac"], float)[idx["fr_uniform"][j]]); wm = float(np.asarray(z["max_wmax"], float)[idx["fr_uniform"][j]])
            fh.write(f"{kk[0]},{kk[1]}," + ",".join(vals) + f",{sec['ot_matched|fr_uniform']['per_row'][j]:.3f},{ess:.4f},{wm:.4f}"
                     + (f",{froz['ot_matched|fr_uniform']['per_row'][j]:.3f}" if froz else "") + "\n")
    print(f"  wrote {os.path.relpath(out_dir, ROOT)}/analysis.json, comparison.csv" + (f", figures/ ({', '.join(figs)})" if figs else ""))


if __name__ == "__main__":
    main()
