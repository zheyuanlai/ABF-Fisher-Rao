#!/usr/bin/env python
"""Analyzer for the transport-refresh campaign -- written BEFORE its data.

Prereg: configs/transport_campaign/gateway_transport_refresh_prereg.json.  Reuses the frozen
statistics of the first campaign's analyzer (cluster bootstrap by seed, decision and verdict
rules) and adds the mechanism descriptives the closure identified: left-flank signed F' error
and barrier error at T.

    python scripts/analyze_gateway_transport_refresh.py [--raw PATH] [--no-figures]
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
import analyze_gateway_horizontal_transport as A1                             # noqa: E402
from analyze_gateway_horizontal_transport import (                            # noqa: E402
    contrast, decide_primary, verdict_vs_abf, fmt, MARGIN, FLOORS, T_DOSE_START)
from analyze_uniform_gateway import tau, PERSIST, FRACTIONS                  # noqa: E402
from analyze_gateway_bandwidth_audit import mean_force_at, e_f, label, cumtrapz   # noqa: E402
from eb_abffr_core import EVAL_LO, EVAL_HI                                   # noqa: E402

PREREG = os.path.join(ROOT, "configs/transport_campaign/gateway_transport_refresh_prereg.json")
STEP1 = os.path.join(ROOT, "results/information_campaign/gateway_baseline_audit/analysis.json")
DEFAULT_RAW = os.path.join(ROOT, "results/transport_campaign/gateway_horizontal/production_refresh/raw.npz")
ARMS = ("abf", "fr_uniform", "ot_exact", "abf_refresh", "fr_uniform_refresh", "ot_exact_refresh", "ot_full_refresh")
NAMED = [("P1", "ot_exact", "fr_uniform"), ("P2", "ot_exact_refresh", "ot_exact"),
         ("C1", "abf_refresh", "abf"), ("C2", "fr_uniform_refresh", "fr_uniform"),
         ("A1", "ot_exact_refresh", "abf_refresh"), ("A2", "ot_exact_refresh", "fr_uniform_refresh"),
         ("A3", "ot_full_refresh", "abf_refresh"), ("A4", "ot_full_refresh", "fr_uniform_refresh"),
         ("V_fr", "fr_uniform", "abf"), ("V_ot", "ot_exact", "abf"), ("V_otr", "ot_exact_refresh", "abf"),
         ("V_fullr", "ot_full_refresh", "abf"), ("V_frr", "fr_uniform_refresh", "abf")]
A1.BOOT_SEED = 20260903          # this campaign's frozen bootstrap seed (used by contrast())


def make_figures(out_dir, x, t, curves, kl_med, dc_med, dFp_med, hs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  (figures skipped: {e})"); return []
    os.makedirs(out_dir, exist_ok=True)
    col = dict(abf="#4d4d4d", fr_uniform="#d95f02", ot_exact="#1b9e77", abf_refresh="#4d4d4d",
               fr_uniform_refresh="#d95f02", ot_exact_refresh="#1b9e77", ot_full_refresh="#7570b3")
    ls = {m: ("--" if m.endswith("_refresh") else "-") for m in ARMS}
    made = []

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig); made.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for m in ("abf", "fr_uniform", "ot_exact"):
        axes[0].plot(t, curves[m], color=col[m], ls=ls[m], lw=1.6, label=m)
    for m in ("abf", "abf_refresh", "fr_uniform_refresh", "ot_exact_refresh", "ot_full_refresh"):
        axes[1].plot(t, curves[m], color=col[m], ls=ls[m], lw=1.6, label=m)
    for ax, title in zip(axes, ("no refresh", "oracle fibre refresh (dashed)")):
        ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_title(title, fontsize=9); ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel(f"median $e_F(t)$ at $h_{{read}}$ = {hs}")
    save(fig, "fig_B2_convergence_eF")
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.8), sharex=True)
    for m in ARMS:
        axes[0].plot(t, kl_med[m], color=col[m], ls=ls[m], lw=1.4, label=m)
        axes[1].plot(t, dc_med[m], color=col[m], ls=ls[m], lw=1.4)
    axes[0].set_yscale("log"); axes[0].set_ylabel(r"median KL$(\hat p_t^x\|U)$"); axes[0].legend(fontsize=6, frameon=False, ncol=2)
    axes[1].set_yscale("log"); axes[1].set_ylabel(r"median $D_{\rm cond}(t)$"); axes[1].set_xlabel("t")
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    save(fig, "fig_C2_marginal_vs_conditional")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for m in ARMS:
        ax.plot(x, dFp_med[m], color=col[m], ls=ls[m], lw=1.4, label=m)
    ax.axvspan(-0.3, -0.05, color="#7570b3", alpha=0.10, lw=0); ax.axvspan(0.05, 0.3, color="#7570b3", alpha=0.10, lw=0)
    ax.axhline(0, color="k", lw=0.6); ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel("x"); ax.set_ylabel(f"median $\\hat F'_T - F'_{{ref}}$ at $h_{{read}}$ = {hs}"); ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    save(fig, "fig_F_signed_mean_force_error")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    step1 = json.load(open(STEP1)); h_star = float(step1["h_read_star"]); hs = label(h_star)
    smoke = os.path.abspath(a.raw) != os.path.abspath(DEFAULT_RAW)
    if smoke:
        print("*** MACHINERY SMOKE on a non-production file -- the numbers below are NOT a result ***")
    z = np.load(a.raw, allow_pickle=True)
    method = np.array([str(m) for m in z["method"]]); init = np.array([str(i) for i in z["init"]]); seed = z["seed"].astype(int)
    rows = {}
    for i in range(len(method)):
        rows.setdefault((init[i], seed[i]), {})[method[i]] = i
    keys = sorted(k for k, v in rows.items() if set(v) >= set(ARMS))
    assert keys, "no complete rows"
    idx = {m: np.array([rows[k][m] for k in keys]) for m in ARMS}
    clusters = np.array([k[1] for k in keys]); init_of = np.array([k[0] for k in keys]); inits = sorted(str(i) for i in set(init_of))
    x = np.asarray(z["x_grid"][0], float); dx = float(x[1] - x[0]); mask = (x >= EVAL_LO) & (x <= EVAL_HI); t = np.asarray(z["t"][0], float)
    cfg0 = json.loads(str(z["config_json"][0])); h_bias, min_count = float(cfg0["h"]), float(cfg0["min_count"])
    alpha_star = float(z["alpha_star"]); gamma = float(z["gamma"][idx["fr_uniform"][0]])
    print(f"rows: {len(keys)} ({', '.join(f'{i}: {int((init_of == i).sum())}' for i in inits)}), seeds {len(set(clusters))}; "
          f"h_bias {h_bias:g}; h_read* {hs}; gamma {gamma:g}; alpha** {alpha_star:g}")
    Sf, C, F_ref, Fp_ref = (np.asarray(z[k], float) for k in ("Sf_t", "C_t", "F_ref", "Fp_ref"))
    ladder = sorted({h_star, 0.07, 0.0, h_bias}, reverse=True)
    ro = {label(h): e_f(mean_force_at(Sf, C, h, dx, min_count), F_ref, dx, mask) for h in ladder}
    dev = float(np.abs(ro[label(h_bias)] - np.asarray(z["l2_f_t"], float)).max())
    print(f"self-check: offline read-out at h_bias vs engine l2_f_t, max|dev| {dev:.2e}"); assert dev < 1e-9
    Fp_star = mean_force_at(Sf, C, h_star, dx, min_count)
    efp = np.sqrt(np.mean(((Fp_star - Fp_ref[:, None, :])[..., mask]) ** 2, axis=-1)); I_Fp = np.trapezoid(efp, t, axis=1)
    I = {lab: np.trapezoid(ro[lab], t, axis=1) for lab in ro}; fin = {lab: ro[lab][:, -1] for lab in ro}

    res, k = {}, 0
    print(f"\n{'read-out':>8} {'tag':>6} {'contrast':>36} {'d I_F':>8} {'CI95':>20} {'wins':>6} {'d e_F(T)':>9} {'CI95':>20}")
    for lab in [hs, "0.07", "raw"]:
        res[lab] = dict(eF_T_median={m: float(np.median(fin[lab][idx[m]])) for m in ARMS},
                        I_F_median={m: float(np.median(I[lab][idx[m]])) for m in ARMS}, contrasts={})
        for tag, arm, ref in NAMED:
            ia, ir = idx[arm], idx[ref]
            c_int = contrast(I[lab][ia], I[lab][ir], clusters, k); c_fin = contrast(fin[lab][ia], fin[lab][ir], clusters, k + 1); k += 2
            pi = {}
            if lab == hs:
                for ini in inits:
                    sel = np.nonzero(init_of == ini)[0]
                    pi[ini] = dict(d_int=contrast(I[lab][ia][sel], I[lab][ir][sel], clusters[sel], k),
                                   d_fin=contrast(fin[lab][ia][sel], fin[lab][ir][sel], clusters[sel], k + 1)); k += 2
            res[lab]["contrasts"][tag] = dict(arm=arm, ref=ref, d_int=c_int, d_fin=c_fin, per_init=pi)
            print(f"{lab:>8} {tag:>6} {arm + ' vs ' + ref:>36} {c_int['median']:+8.2f} [{c_int['ci95'][0]:+8.2f},{c_int['ci95'][1]:+8.2f}]"
                  f" {c_int['wins']:3d}/{c_int['n']} {c_fin['median']:+9.2f} [{c_fin['ci95'][0]:+8.2f},{c_fin['ci95'][1]:+8.2f}]"
                  + ("" if not pi else "   per-init dI_F " + ", ".join(f"{i}: {v['d_int']['median']:+.1f}%" for i, v in pi.items())))

    sec = {tag: contrast(I_Fp[idx[arm]], I_Fp[idx[ref]], clusters, k + j) for j, (tag, arm, ref) in enumerate(NAMED)}; k += len(NAMED)
    print("\nI_F' at h_read*: " + "; ".join(f"{tag} {fmt(c)}" for tag, c in sec.items()))
    froz = None
    if "frozen_l2_f_kT" in z.files:
        fz = np.asarray(z["frozen_l2_f_kT"], float)
        froz = {tag: contrast(fz[idx[arm]], fz[idx[ref]], clusters, k + j) for j, (tag, arm, ref) in enumerate(NAMED)}; k += len(NAMED)
        print("frozen-bias endpoint: " + "; ".join(f"{tag} {fmt(c)}" for tag, c in froz.items()))

    # ---- mechanism ----
    kl = np.asarray(z["kl_uniform_t"], float); dc = np.nan_to_num(np.asarray(z["dcond_t"], float)); md = t >= T_DOSE_START - 1e-12
    J = np.trapezoid(kl[:, md], t[md], axis=1); Dc = np.trapezoid(dc, t, axis=1)
    dFp_T = Fp_star[:, -1, :] - Fp_ref
    Lf = (x > -0.3) & (x < -0.05)
    F_T = cumtrapz(Fp_star[:, -1, :], dx); i0, im = int(np.argmin(abs(x))), int(np.argmin(abs(x + 1)))
    beta = float(cfg0["beta"])
    bar_err = ((F_T[:, i0] - F_T[:, im]) - (F_ref[:, i0] - F_ref[:, im])) * beta
    mech = {}
    print("\nmechanism (medians): J_KL | int D_cond | left-flank signed F' err at T | barrier err (kT) | extras")
    for m in ARMS:
        r = idx[m]
        d = dict(median_J_KL=float(np.median(J[r])), median_int_Dcond=float(np.median(Dc[r])),
                 median_left_flank_signed_Fp_err=float(np.median(dFp_T[r][:, Lf].mean(1))), median_barrier_err_kT=float(np.median(bar_err[r])),
                 median_final_P_plus=float(np.median(np.asarray(z["P_regions"], float)[r, -1, 2])))
        extra = ""
        if m.startswith("ot_"):
            d.update(median_mean_absdx=float(np.median(np.mean(np.asarray(z["ot_absdx_t"], float)[r][:, 1:], axis=1))),
                     median_max_Dmove=float(np.median(np.max(np.asarray(z["dmove_max_t"], float)[r], axis=1))))
            extra += f"|dx| {d['median_mean_absdx']:.2e} maxDmove {d['median_max_Dmove']:.2e} "
        if m.startswith("fr_"):
            ess = np.asarray(z["min_ess_frac"], float)[r]; wm = np.asarray(z["max_wmax"], float)[r]
            d.update(median_min_ess_frac=float(np.median(ess)), median_max_wmax=float(np.median(wm)), worst_min_ess_frac=float(ess.min()))
            extra += f"ESS/N {d['median_min_ess_frac']:.3f} wmax {d['median_max_wmax']:.4f}"
        mech[m] = d
        print(f"  {m:>20}: {d['median_J_KL']:.4f} | {d['median_int_Dcond']:.4f} | {d['median_left_flank_signed_Fp_err']:+.4f} | {d['median_barrier_err_kT']:+.3f} | {extra}")
    health = {m: (mech[m]["median_min_ess_frac"] >= FLOORS["ess_anc_over_N_min"] and mech[m]["median_max_wmax"] <= FLOORS["wmax_max"]) for m in ("fr_uniform", "fr_uniform_refresh")}
    ratio_J = mech["ot_exact"]["median_J_KL"] / mech["fr_uniform"]["median_J_KL"] if mech["fr_uniform"]["median_J_KL"] > 0 else float("nan")
    print(f"  dose check on production rows: J_KL(ot_exact)/J_KL(fr) = {ratio_J:.3f}")

    # ---- time-to-accuracy ----
    curves = {m: np.median(ro[hs][idx[m]], axis=0) for m in ARMS}
    e0 = float(curves["abf"][0]); eps_list = {f"e0/{int(1 / f)}": e0 * f for f in FRACTIONS}; eps_list["abf_final"] = float(curves["abf"][-1])
    speed = {}
    for nm, eps in eps_list.items():
        ta = tau(t, curves["abf"], eps, PERSIST); speed[nm] = dict(eps=eps, tau_abf=ta)
        for m in ARMS[1:]:
            tm = tau(t, curves[m], eps, PERSIST); speed[nm][f"tau_{m}"] = tm
            speed[nm][f"speedup_{m}"] = (ta / tm if np.isfinite(ta) and np.isfinite(tm) and tm > 0 else None)
        print(f"  tau[{nm}]: abf {ta:.1f}; " + "; ".join(f"{m} {speed[nm][f'tau_{m}']:.1f}" for m in ARMS[1:]))

    # ---- decisions ----
    P = res[hs]["contrasts"]
    p1 = decide_primary(P["P1"]["d_int"])
    repaired = P["P2"]["d_int"]["median"] <= -MARGIN and P["P2"]["d_int"]["ci95"][1] < 0
    a1 = verdict_vs_abf(P["A1"]["d_int"], P["A1"]["d_fin"]); a2 = decide_primary(P["A2"]["d_int"])
    a3 = verdict_vs_abf(P["A3"]["d_int"], P["A3"]["d_fin"]); a4 = decide_primary(P["A4"]["d_int"])
    c1 = verdict_vs_abf(P["C1"]["d_int"], P["C1"]["d_fin"]); c2 = verdict_vs_abf(P["C2"]["d_int"], P["C2"]["d_fin"], health["fr_uniform_refresh"])
    vs_abf = {tag: verdict_vs_abf(P[tag]["d_int"], P[tag]["d_fin"], health.get(P[tag]["arm"], True)) for tag in ("V_fr", "V_ot", "V_otr", "V_fullr", "V_frr")}
    replicated = P["V_fr"]["d_int"]["ci95"][1] < 0
    if not replicated:
        outcome = "FAILED_REPLICATION_OF_POSITIVE_CONTROL"
    elif not repaired:
        outcome = "INCONCLUSIVE" if (abs(P["P2"]["d_int"]["median"]) < MARGIN and P["P2"]["d_int"]["ci95"][0] < 0 < P["P2"]["d_int"]["ci95"][1]) else "R2_not_repaired"
    elif a1 == "NEGATIVE":
        outcome = "R3_partial"
    else:
        outcome = "R1a_repaired_competitive" if a2 in ("OT_better", "equivalent") else "R1b_repaired_still_inferior"
    h5 = a3.startswith("SAFE") or a3.startswith("ACCELERATION")
    print(f"\n  P1 exact-dose replication (ot_exact vs fr_uniform): {fmt(P['P1']['d_int'])} -> {p1}")
    print(f"  P2 repair (ot_exact_refresh vs ot_exact): {fmt(P['P2']['d_int'])} -> {'REPAIRED' if repaired else 'not repaired'}")
    print(f"  C1 refresh control (abf_refresh vs abf): {fmt(P['C1']['d_int'])} -> {c1};  C2 (fr+refresh vs fr): {fmt(P['C2']['d_int'])} -> {c2}")
    print(f"  A1 (ot_exact_refresh vs abf_refresh): {fmt(P['A1']['d_int'])} -> {a1};  A2 (vs fr_uniform_refresh): {fmt(P['A2']['d_int'])} -> {a2}")
    print(f"  A3 (ot_full_refresh vs abf_refresh): {fmt(P['A3']['d_int'])} -> {a3};  A4 (vs fr_uniform_refresh): {fmt(P['A4']['d_int'])} -> {a4}")
    print(f"  vs abf: " + ", ".join(f"{P[tag]['arm']} {v}" for tag, v in vs_abf.items()) + f"; positive control replicated: {replicated}")
    print(f"  OUTCOME: {outcome}   H5_refresh (pinned sampler works at exact fibres): {h5}{'   [SMOKE -- not a result]' if smoke else ''}")
    if smoke:
        return
    out_dir = a.out_dir or os.path.dirname(a.raw)
    figs = [] if a.no_figures else make_figures(os.path.join(out_dir, "figures"), x, t, curves,
                                                {m: np.median(kl[idx[m]], axis=0) for m in ARMS}, {m: np.median(dc[idx[m]], axis=0) for m in ARMS},
                                                {m: np.median(dFp_T[idx[m]], axis=0) for m in ARMS}, hs)
    summary = dict(prereg=os.path.relpath(PREREG, ROOT), raw=os.path.relpath(a.raw, ROOT), n_rows=len(keys), n_seeds=len(set(clusters)),
                   h_bias=h_bias, h_read_star=h_star, gamma=gamma, alpha_star=alpha_star, bootstrap=dict(n_resamples=A1.N_BOOT, seed=A1.BOOT_SEED, cluster="seed"),
                   per_readout=res, secondary_I_Fp=sec, frozen_bias=froz, mechanism=mech, dose_ratio_production=ratio_J, health=health,
                   time_to_accuracy=speed, decisions=dict(P1=p1, P2_repaired=bool(repaired), C1=c1, C2=c2, A1=a1, A2=a2, A3=a3, A4=a4, vs_abf=vs_abf),
                   positive_control_replicated=bool(replicated), outcome=outcome, H5_refresh=bool(h5), figures=figs)
    with open(os.path.join(out_dir, "analysis.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(os.path.join(out_dir, "comparison.csv"), "w") as fh:
        fh.write("init,seed," + ",".join(f"{tag}_{q}_{lab}" for lab in (hs, "0.07", "raw") for tag, _, _ in NAMED for q in ("dI", "dF")) + "\n")
        for j, kk in enumerate(keys):
            fh.write(f"{kk[0]},{kk[1]}," + ",".join(f"{res[lab]['contrasts'][tag][q]['per_row'][j]:.3f}" for lab in (hs, "0.07", "raw") for tag, _, _ in NAMED for q in ("d_int", "d_fin")) + "\n")
    print(f"  wrote {os.path.relpath(out_dir, ROOT)}/analysis.json, comparison.csv" + (f", figures/ ({', '.join(figs)})" if figs else ""))


if __name__ == "__main__":
    main()
