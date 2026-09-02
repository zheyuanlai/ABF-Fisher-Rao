#!/usr/bin/env python
"""Analyzer for the targeted-relaxation campaign -- written BEFORE its data.  Stages D0, D1, D2.

Prereg: configs/transport_campaign/gateway_targeted_relax_prereg.json.

    python scripts/analyze_gateway_targeted_relax.py --stage D0     # estimator validity (pass/fail)
    python scripts/analyze_gateway_targeted_relax.py --stage D1     # cost ladder -> rho_selection.json
    python scripts/analyze_gateway_targeted_relax.py --stage D2     # confirmatory
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
import analyze_gateway_horizontal_transport as A1                                   # noqa: E402
from analyze_gateway_horizontal_transport import contrast, decide_primary, verdict_vs_abf, fmt, MARGIN   # noqa: E402
from analyze_gateway_fibre_relax import load, endpoints, med                        # noqa: E402
from analyze_uniform_gateway import tau, PERSIST, FRACTIONS                        # noqa: E402
from analyze_gateway_bandwidth_audit import eb_smooth, label                       # noqa: E402
from eb_abffr_core import EVAL_LO, EVAL_HI, EPS                                    # noqa: E402
import gateway_core as gw                                                          # noqa: E402
import torch                                                                       # noqa: E402

PREREG = os.path.join(ROOT, "configs/transport_campaign/gateway_targeted_relax_prereg.json")
CAMPAIGN = os.path.join(ROOT, "results/transport_campaign/gateway_horizontal")
DIRS = {s: os.path.join(CAMPAIGN, f"targeted_{s}") for s in ("D0", "D1", "D2")}
D0_CORR_MIN = 0.90
FLANK = 0.35
COMPUTE_GATE = 0.8


def vhat_profiles(z, idx, h, min_count, save=-1):
    x = np.asarray(z["x_grid"][0], float); dx = float(x[1] - x[0])
    Sf2, Sf, C = (np.asarray(z[k], float)[idx, save, :] for k in ("Sf2_t", "Sf_t", "C_t"))
    den = eb_smooth(C, h, dx) + min_count + EPS
    return np.clip(eb_smooth(Sf2, h, dx) / den - (eb_smooth(Sf, h, dx) / den) ** 2, 0, None), C


def refs(z, cfg0, x, C, h, min_count):
    """Analytic Var(f|x) and its resolution-matched version s(v_ref C)/(s(C)+min_count)."""
    dx = float(x[1] - x[0])
    tt = lambda v: torch.tensor(float(v), dtype=torch.float64)  # noqa: E731
    v_ref = gw.sensitivity_ref(torch.tensor(x), tt(cfg0["beta"]), tt(cfg0["omega_out"]), tt(cfg0["omega_out"] * cfg0["r"]), tt(cfg0["s"])).numpy()
    v_match = eb_smooth(v_ref[None, :] * C, h, dx) / (eb_smooth(C, h, dx) + min_count + EPS)
    return v_ref, v_match


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------------------- D0
def stage_d0(raw, out_dir, no_figures):
    z, arms, keys, idx, clusters, init_of, x, dx, mask, t, cfg0 = load(raw)
    h, mc = float(cfg0["h"]), float(cfg0["min_count"])
    r = idx["abf"]
    res = {}
    print(f"D0: {len(keys)} ABF rows; online kernel h_bias {h:g}")
    for tv in (4.0, 10.0, float(t[-1])):
        s = int(np.argmin(abs(t - tv)))
        vh, C = vhat_profiles(z, r, h, mc, save=s)
        v_ref, v_match = refs(z, cfg0, x, C, h, mc)
        vh_med, vm_med = np.median(vh, axis=0), np.median(v_match, axis=0)
        loc_h = float(np.trapezoid(np.where(abs(x) < FLANK, vh_med, 0)[mask], x[mask]) / max(np.trapezoid(vh_med[mask], x[mask]), EPS))
        loc_r = float(np.trapezoid(np.where(abs(x) < FLANK, v_ref, 0)[mask], x[mask]) / np.trapezoid(v_ref[mask], x[mask]))
        rel_l2 = float(np.sqrt(np.mean((vh_med - vm_med)[mask] ** 2)) / np.sqrt(np.mean(vm_med[mask] ** 2)))
        res[f"t={t[s]:g}"] = dict(corr_raw=corr(vh_med[mask], v_ref[mask]), corr_matched=corr(vh_med[mask], vm_med[mask]),
                                  rel_l2_matched=rel_l2, localisation_vhat=loc_h, localisation_ref=loc_r,
                                  peak_x_vhat=float(x[mask][np.argmax(vh_med[mask])]), peak_x_ref=float(x[mask][np.argmax(v_ref[mask])]))
        d = res[f"t={t[s]:g}"]
        print(f"  t={t[s]:5.1f}: corr(v_hat, analytic) {d['corr_raw']:.3f}; corr(v_hat, resolution-matched) {d['corr_matched']:.3f}; rel L2 {rel_l2:.3f}; "
              f"share within |x|<{FLANK}: v_hat {loc_h:.3f} vs analytic {loc_r:.3f}; peak at x = {d['peak_x_vhat']:+.2f} (analytic {d['peak_x_ref']:+.2f})")
    final = res[f"t={t[-1]:g}"]
    passed = final["corr_matched"] >= D0_CORR_MIN
    print(f"  D0 {'PASS' if passed else 'FAIL'}: corr(resolution-matched) at T = {final['corr_matched']:.3f} (requirement >= {D0_CORR_MIN})")
    with open(os.path.join(out_dir, "analysis.json"), "w") as fh:
        json.dump(dict(prereg=os.path.relpath(PREREG, ROOT), raw=os.path.relpath(raw, ROOT), n_rows=len(keys), h_bias=h, per_time=res,
                       requirement=D0_CORR_MIN, passed=bool(passed)), fh, indent=2, default=float)
    if not no_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fd = os.path.join(out_dir, "figures"); os.makedirs(fd, exist_ok=True)
            vh, C = vhat_profiles(z, r, h, mc, save=-1); v_ref, v_match = refs(z, cfg0, x, C, h, mc)
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.plot(x, v_ref, "k-", lw=1.2, label=r"analytic $\mathrm{Var}(f|x)=2\beta^{-2}(\omega'/\omega)^2$")
            ax.plot(x, np.median(v_match, axis=0), "k--", lw=1.2, label="resolution-matched reference")
            ax.plot(x, np.median(vh, axis=0), color="#1b9e77", lw=1.6, label=r"online $\hat v_T(x)$ (median of 16 ABF rows)")
            ax.set_xlim(-1.5, 1.5); ax.set_xlabel("x"); ax.set_ylabel("conditional variance of the local force"); ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25)
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(fd, f"fig_I_online_sensitivity.{ext}"), dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"  (figure skipped: {e})")
    print(f"  wrote {os.path.relpath(out_dir, ROOT)}/analysis.json")


# ----------------------------------------------------------------------------- D1
def select_rho(ladder, dI_T_vs_A, compute_ratio_T):
    """Smallest rho <= 1 with median Delta I_F(T_rho vs A_rho) <= -10% and C_T(eps_ABF)/C_A0(eps_ABF) <= 0.8; else (1.0, False)."""
    for rho in sorted(float(r) for r in ladder):
        if rho > 1.0:
            continue
        if dI_T_vs_A[f"{rho:g}"] <= -MARGIN and compute_ratio_T[f"{rho:g}"] <= COMPUTE_GATE:
            return rho, True
    return 1.0, False


def name_t(al, rho, move=False):
    base = {"A": "abf", "F": "fr_uniform", "T": "ot_exact"}[al]
    return base if rho == 0 else f"{base}_targ{rho:g}" + ("move" if move else "")


def per_arm_table(z, arms, idx, P, Praw, t, fz):
    rows = {}
    for m in arms:
        r = idx[m]
        d = dict(I_F=med(P["I"][r]), eF_T=med(P["fin"][r]), b_flank=med(P["b_flank"][r]), barrier_kT=med(P["barrier_kT"][r]),
                 frozen=(med(fz[r]) if fz is not None else None),
                 cost_ratio=(med(np.asarray(z["fibre_cost_ratio"], float)[r]) if "fibre_cost_ratio" in z.files else 0.0))
        if "targ_flank_frac_t" in z.files and str(z["refresh"][r[0]]) == "targeted":
            d.update(flank_budget_frac=med(np.mean(np.asarray(z["targ_flank_frac_t"], float)[r][:, 1:], axis=1)),
                     active_frac=med(np.mean(np.asarray(z["targ_active_frac_t"], float)[r][:, 1:], axis=1)),
                     mean_c=med(np.mean(np.asarray(z["targ_cmean_t"], float)[r][:, 1:], axis=1)))
        if "ot_flank_dx_frac_t" in z.files and str(z["transport"][r[0]]) == "horizontal_ot":
            d.update(flank_dx_frac=med(np.mean(np.asarray(z["ot_flank_dx_frac_t"], float)[r][:, 1:], axis=1)),
                     tau_move=med(np.mean(np.asarray(z["ot_tau_move_t"], float)[r][:, 1:], axis=1)))
        rows[m] = d
    return rows


def compute_to_accuracy(P, idx, arms, t, table):
    e_abf = np.median(P["e"][idx["abf"]], axis=0); e0 = float(e_abf[0])
    eps_list = {f"e0/{int(1 / f)}": e0 * f for f in FRACTIONS}; eps_list["abf_final"] = float(e_abf[-1])
    out = {}
    for nm, eps in eps_list.items():
        out[nm] = dict(eps=eps)
        for m in arms:
            tm = tau(t, np.median(P["e"][idx[m]], axis=0), eps, PERSIST)
            out[nm][m] = dict(tau=tm, compute=(tm * (1 + table[m]["cost_ratio"]) if np.isfinite(tm) else float("inf")))
        ref = out[nm]["abf"]["compute"]
        for m in arms:
            out[nm][m]["ratio_to_abf"] = (out[nm][m]["compute"] / ref if np.isfinite(ref) and ref > 0 else float("nan"))
    return out


def stage_d1(raw, out_dir, no_figures):
    pre = json.load(open(PREREG))
    z, arms, keys, idx, clusters, init_of, x, dx, mask, t, cfg0 = load(raw)
    ladder = [float(r) for r in pre["stage_D1_cost_ladder"]["ladder_rho"]]
    P = endpoints(z, x, dx, mask, t, cfg0, 0.0); P2 = endpoints(z, x, dx, mask, t, cfg0, 0.0175)
    fz = np.asarray(z["frozen_l2_f_kT"], float) if "frozen_l2_f_kT" in z.files else None
    table = per_arm_table(z, arms, idx, P, P2, t, fz)
    print(f"D1: rows {len(keys)}, seeds {len(set(clusters))}, arms {len(arms)}; read-out raw bins")
    print(f"{'arm':>26} {'I_F':>7} {'e_F(T)':>8} {'b_flank':>8} {'bar kT':>7} {'frozen':>7} {'cost':>6} {'flank$':>7} {'active':>7} {'mean c':>7} {'flank dx':>8} {'tau_mv':>7}")
    order = ["abf", "fr_uniform", "ot_exact"] + [name_t(al, rho, mv) for rho in ladder for al, mv in (("A", False), ("F", False), ("T", False), ("T", True))] + ["abf_relax0.5", "ot_exact_relax0.5"]
    for m in order:
        d = table[m]
        print(f"{m:>26} {d['I_F']:7.4f} {d['eF_T']:8.5f} {d['b_flank']:+8.4f} {d['barrier_kT']:+7.3f} {(d['frozen'] if d['frozen'] is not None else float('nan')):7.4f} {d['cost_ratio']:6.2f} "
              f"{d.get('flank_budget_frac', float('nan')):7.3f} {d.get('active_frac', float('nan')):7.3f} {d.get('mean_c', float('nan')):7.2f} {d.get('flank_dx_frac', float('nan')):8.3f} {d.get('tau_move', float('nan')):7.3f}")
    # retention vs the all-walker c = 0.5 reference, matched contrasts, compute-to-accuracy
    k = 0; ret, matched = {}, {}
    print("\nretention of the all-walker c=0.5 benefit, ret_X(rho) = (I_F(X_0) - I_F(X_rho)) / (I_F(X_0) - I_F(X_all0.5)):")
    for al, base, refm in (("A", "abf", "abf_relax0.5"), ("T", "ot_exact", "ot_exact_relax0.5")):
        ret[al] = {}
        for rho in ladder:
            R = (P["I"][idx[base]] - P["I"][idx[name_t(al, rho)]]) / (P["I"][idx[base]] - P["I"][idx[refm]])
            ci95, _ = A1.boot_cluster(R, clusters, 20260905 + k); k += 1
            ret[al][f"{rho:g}"] = dict(median=med(R), ci95=ci95)
        print(f"  {al}: " + "  ".join(f"rho={rho:g}: {ret[al][f'{rho:g}']['median']:.3f} [{ret[al][f'{rho:g}']['ci95'][0]:.2f},{ret[al][f'{rho:g}']['ci95'][1]:.2f}]" for rho in ladder))
    print("matched-treatment contrasts (Delta I_F / Delta e_F(T), raw bins, 8 seeds -- descriptive):")
    for rho in ladder:
        T, Am, F, Tm = name_t("T", rho), name_t("A", rho), name_t("F", rho), name_t("T", rho, True)
        mt = {}
        for tag, arm, ref in (("T_vs_A", T, Am), ("T_vs_F", T, F), ("Tmove_vs_T", Tm, T), ("F_vs_A", F, Am)):
            mt[tag] = dict(d_int=contrast(P["I"][idx[arm]], P["I"][idx[ref]], clusters, 300 + k), d_fin=contrast(P["fin"][idx[arm]], P["fin"][idx[ref]], clusters, 301 + k)); k += 2
        matched[f"{rho:g}"] = mt
        print(f"  rho={rho:g}: T vs A {fmt(mt['T_vs_A']['d_int'])} / final {mt['T_vs_A']['d_fin']['median']:+.1f}%;  T vs F {fmt(mt['T_vs_F']['d_int'])};  "
              f"Tmove vs T {fmt(mt['Tmove_vs_T']['d_int'])};  F vs A {fmt(mt['F_vs_A']['d_int'])}")
    cta = compute_to_accuracy(P, idx, arms, t, table)
    print("compute to accuracy, C = tau x (1 + cost ratio), relative to plain ABF:")
    for nm in cta:
        print(f"  eps={nm:>9}: " + "  ".join(f"{m} {cta[nm][m]['ratio_to_abf']:.2f}" for m in ["fr_uniform"] + [name_t("T", rho) for rho in ladder] + [name_t("A", rho) for rho in ladder] + ["ot_exact_relax0.5"]))
    dI = {f"{rho:g}": matched[f"{rho:g}"]["T_vs_A"]["d_int"]["median"] for rho in ladder}
    cr = {f"{rho:g}": cta["abf_final"][name_t("T", rho)]["ratio_to_abf"] for rho in ladder}
    rho_star, gate = select_rho(ladder, dI, cr)
    mech = all(table[name_t(al, rho)].get("flank_budget_frac", 0) > 0.8 for rho in ladder for al in ("A", "F", "T")) and \
        all(table[name_t("T", rho)].get("flank_dx_frac", 1) < 0.3 for rho in ladder)
    print(f"\n  rho* = {rho_star:g} (rule: smallest rho <= 1 with dI_F(T vs A) <= -10% and C_T(eps_ABF)/C_A0 <= 0.8); gate_D1 = {gate}")
    print(f"  mechanism check (budget fraction at |x|<{FLANK} > 0.8 for all targeted arms AND displacement fraction there < 0.3): {mech}")
    sel = dict(prereg=os.path.relpath(PREREG, ROOT), raw=os.path.relpath(raw, ROOT), ladder_rho=ladder, dI_T_vs_A_median=dI, compute_ratio_T=cr,
               rho_star=rho_star, gate_D1=bool(gate), mechanism_check=bool(mech), rule="smallest rho <= 1 with dI_F(T vs A) <= -10% and C_T/C_A0 <= 0.8 at eps = ABF final; else 1 with gate False")
    with open(os.path.join(out_dir, "rho_selection.json"), "w") as fh:
        json.dump(sel, fh, indent=2, default=float)
    with open(os.path.join(out_dir, "analysis.json"), "w") as fh:
        json.dump(dict(sel, per_arm=table, retention=ret, matched=matched, compute_to_accuracy=cta), fh, indent=2, default=float)
    if not no_figures:
        d1_figs(out_dir, z, idx, x, t, P, table, ladder, cfg0)
    print(f"  wrote {os.path.relpath(out_dir, ROOT)}/rho_selection.json, analysis.json")


def d1_figs(out_dir, z, idx, x, t, P, table, ladder, cfg0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  (figures skipped: {e})"); return
    fd = os.path.join(out_dir, "figures"); os.makedirs(fd, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    # (a) error vs compute for T_rho, A_rho, anchors and the all-walker reference
    N_over_dt = 1.0
    for m, col, ls in (("abf", "#4d4d4d", "-"), ("fr_uniform", "#d95f02", "-"), ("ot_exact", "#1b9e77", "-"), ("ot_exact_relax0.5", "#7570b3", "-")):
        axes[0].plot(t * (1 + table[m]["cost_ratio"]) * N_over_dt, np.median(P["e"][idx[m]], axis=0), color=col, ls=ls, lw=1.3, label=m)
    for rho, ls in zip(ladder, (":", "-.", "--", (0, (1, 1)))):
        for al, col in (("T", "#1b9e77"), ("A", "#4d4d4d")):
            m = name_t(al, rho)
            axes[0].plot(t * (1 + table[m]["cost_ratio"]), np.median(P["e"][idx[m]], axis=0), color=col, ls=ls, lw=1.1, label=m)
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("compute (outer-time-equivalent, incl. relaxation)"); axes[0].set_ylabel("median $e_F$ (raw bins)")
    axes[0].legend(fontsize=5, frameon=False, ncol=2); axes[0].grid(alpha=0.25, which="both")
    # (b) where the budget goes vs where mass moves
    rhos = ladder
    axes[1].plot(rhos, [table[name_t("T", r)]["flank_budget_frac"] for r in rhos], "o-", color="#1b9e77", label=f"T: budget fraction at |x|<{FLANK}")
    axes[1].plot(rhos, [table[name_t("A", r)]["flank_budget_frac"] for r in rhos], "s-", color="#4d4d4d", label="A: budget fraction")
    axes[1].plot(rhos, [table[name_t("T", r)]["flank_dx_frac"] for r in rhos], "o--", color="#7570b3", label="T: OT displacement fraction there")
    axes[1].set_xscale("log"); axes[1].set_ylim(0, 1.05); axes[1].set_xlabel(r"cost ratio $\rho$"); axes[1].legend(fontsize=7, frameon=False); axes[1].grid(alpha=0.25, which="both")
    # (c) online sensitivity vs analytic, from the T_1 arm at T
    h, mc = float(cfg0["h"]), float(cfg0["min_count"])
    m = name_t("T", 1.0)
    vh, C = vhat_profiles(z, idx[m], h, mc, save=-1); v_ref, v_match = refs(z, cfg0, x, C, h, mc)
    axes[2].plot(x, v_ref, "k-", lw=1.1, label="analytic Var(f|x)"); axes[2].plot(x, np.median(vh, axis=0), color="#1b9e77", lw=1.5, label=f"online $\\hat v_T$ ({m})")
    axes[2].axvspan(-FLANK, FLANK, color="#7570b3", alpha=0.08, lw=0); axes[2].set_xlim(-1.5, 1.5); axes[2].set_xlabel("x"); axes[2].legend(fontsize=7, frameon=False); axes[2].grid(alpha=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fd, f"fig_J_cost_ladder.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- D2
def stage_d2(raw, out_dir, no_figures):
    A1.BOOT_SEED = 20260905
    z, arms, keys, idx, clusters, init_of, x, dx, mask, t, cfg0 = load(raw)
    inits = sorted(str(i) for i in set(init_of))
    rho = float(z["rho_star"]); gate_d1 = bool(z["gate_D1"])
    A0, F0, T0 = "abf", "fr_uniform", "ot_exact"; Ar, Fr, Tr, Tm = name_t("A", rho), name_t("F", rho), name_t("T", rho), name_t("T", rho, True)
    assert set(arms) == {A0, F0, T0, Ar, Fr, Tr, Tm}, arms
    print(f"D2: rows {len(keys)}, seeds {len(set(clusters))}; rho* {rho:g} (gate_D1 {gate_d1}); read-out raw bins")
    E = {lab: endpoints(z, x, dx, mask, t, cfg0, h) for lab, h in (("raw", 0.0), ("0.0175", 0.0175))}
    fz = np.asarray(z["frozen_l2_f_kT"], float) if "frozen_l2_f_kT" in z.files else None
    table = per_arm_table(z, arms, idx, E["raw"], E["0.0175"], t, fz)
    NAMED = [("PRIMARY", Tr, Ar), ("S1", Tr, Fr), ("S2", Tm, Tr), ("S3", Fr, Ar), ("S4", Ar, A0), ("S5", Tr, T0),
             ("V_F0", F0, A0), ("V_T0", T0, A0), ("V_Ar", Ar, A0), ("V_Fr", Fr, A0), ("V_Tr", Tr, A0), ("V_Tm", Tm, A0)]
    res, k = {}, 0
    print(f"\n{'read-out':>8} {'tag':>7} {'contrast':>44} {'d I_F':>8} {'CI95':>20} {'wins':>6} {'d e_F(T)':>9} {'CI95':>20}")
    for lab in ("raw", "0.0175"):
        P = E[lab]; res[lab] = {}
        for tag, arm, ref in NAMED:
            ci = contrast(P["I"][idx[arm]], P["I"][idx[ref]], clusters, k); cf = contrast(P["fin"][idx[arm]], P["fin"][idx[ref]], clusters, k + 1); k += 2
            pi = {}
            if lab == "raw":
                for ini in inits:
                    sel = np.nonzero(init_of == ini)[0]
                    pi[ini] = dict(d_int=contrast(P["I"][idx[arm]][sel], P["I"][idx[ref]][sel], clusters[sel], k)); k += 1
            res[lab][tag] = dict(arm=arm, ref=ref, d_int=ci, d_fin=cf, per_init=pi)
            print(f"{lab:>8} {tag:>7} {arm + ' vs ' + ref:>44} {ci['median']:+8.2f} [{ci['ci95'][0]:+8.2f},{ci['ci95'][1]:+8.2f}] {ci['wins']:3d}/{ci['n']} "
                  f"{cf['median']:+9.2f} [{cf['ci95'][0]:+8.2f},{cf['ci95'][1]:+8.2f}]" + ("" if not pi else "  per-init " + ", ".join(f"{i}: {v['d_int']['median']:+.1f}%" for i, v in pi.items())))
    froz = None
    if fz is not None:
        froz = {tag: contrast(fz[idx[arm]], fz[idx[ref]], clusters, 800 + j) for j, (tag, arm, ref) in enumerate(NAMED)}
        print("frozen-bias: " + "; ".join(f"{tag} {fmt(c)}" for tag, c in froz.items()))
    print("per arm: " + "; ".join(f"{m}: e_F(T) {table[m]['eF_T']:.5f}, b_flank {table[m]['b_flank']:+.4f}, cost {table[m]['cost_ratio']:.2f}"
                                  + (f", flank budget {table[m]['flank_budget_frac']:.2f}" if "flank_budget_frac" in table[m] else "") for m in arms))
    cta = compute_to_accuracy(E["raw"], idx, arms, t, table)
    print("compute to accuracy relative to plain ABF: " + "; ".join(f"{nm}: T {cta[nm][Tr]['ratio_to_abf']:.2f}, A {cta[nm][Ar]['ratio_to_abf']:.2f}, F {cta[nm][Fr]['ratio_to_abf']:.2f}, fr_0 {cta[nm][F0]['ratio_to_abf']:.2f}" for nm in cta))
    Pr = res["raw"]
    prim = verdict_vs_abf(Pr["PRIMARY"]["d_int"], Pr["PRIMARY"]["d_fin"])
    science = Pr["PRIMARY"]["d_int"]["median"] <= -MARGIN and Pr["PRIMARY"]["d_int"]["ci95"][1] < 0
    deploy = cta["abf_final"][Tr]["ratio_to_abf"] < COMPUTE_GATE and rho <= 1.0
    s1 = decide_primary(Pr["S1"]["d_int"]); s3 = verdict_vs_abf(Pr["S3"]["d_int"], Pr["S3"]["d_fin"]); s4 = verdict_vs_abf(Pr["S4"]["d_int"], Pr["S4"]["d_fin"])
    replicated = Pr["V_F0"]["d_int"]["ci95"][1] < 0
    outcome = ("FAILED_REPLICATION_OF_POSITIVE_CONTROL" if not replicated else "GATE_MET" if (science and deploy) else "SCIENCE_ONLY" if science else "NO_VALUE")
    mech = table[Tr].get("flank_budget_frac", 0) > 0.8 and table[Tr].get("flank_dx_frac", 1) < 0.3
    print(f"\n  PRIMARY T_rho* vs A_rho*: {fmt(Pr['PRIMARY']['d_int'])} final {Pr['PRIMARY']['d_fin']['median']:+.2f}% -> {prim}; scientific success {science}")
    print(f"  deployability: C_T(eps_ABF_final)/C_A0 = {cta['abf_final'][Tr]['ratio_to_abf']:.3f} (< {COMPUTE_GATE}), rho* {rho:g} <= 1 -> {deploy}")
    print(f"  S1 T vs F: {fmt(Pr['S1']['d_int'])} -> {s1};  S2 Tmove vs T: {fmt(Pr['S2']['d_int'])};  S3 F vs A: {s3};  S4 A_rho vs A_0: {fmt(Pr['S4']['d_int'])} -> {s4};  S5 T_rho vs T_0: {fmt(Pr['S5']['d_int'])}")
    print(f"  mechanism (budget on flank > 0.8, displacement there < 0.3): {mech};  positive control replicated: {replicated}")
    print(f"  OUTCOME: {outcome}")
    summary = dict(prereg=os.path.relpath(PREREG, ROOT), raw=os.path.relpath(raw, ROOT), n_rows=len(keys), n_seeds=len(set(clusters)), rho_star=rho, gate_D1=gate_d1,
                   per_readout=res, frozen_bias=froz, per_arm=table, compute_to_accuracy=cta,
                   decisions=dict(PRIMARY=prim, scientific_success=bool(science), deployability=bool(deploy), S1=s1, S3=s3, S4=s4),
                   mechanism_check=bool(mech), positive_control_replicated=bool(replicated), outcome=outcome)
    with open(os.path.join(out_dir, "analysis.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(os.path.join(out_dir, "comparison.csv"), "w") as fh:
        fh.write("init,seed," + ",".join(f"{tag}_{q}_{lab}" for lab in ("raw", "0.0175") for tag, _, _ in NAMED for q in ("dI", "dF")) + "\n")
        for j, kk in enumerate(keys):
            fh.write(f"{kk[0]},{kk[1]}," + ",".join(f"{res[lab][tag][q]['per_row'][j]:.3f}" for lab in ("raw", "0.0175") for tag, _, _ in NAMED for q in ("d_int", "d_fin")) + "\n")
    if not no_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fd = os.path.join(out_dir, "figures"); os.makedirs(fd, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
            col = {A0: "#4d4d4d", F0: "#d95f02", T0: "#1b9e77", Ar: "#4d4d4d", Fr: "#d95f02", Tr: "#1b9e77", Tm: "#7570b3"}
            for m in arms:
                e = np.median(E["raw"]["e"][idx[m]], axis=0); ls = "--" if "targ" in m else "-"
                axes[0].plot(t, e, color=col[m], ls=ls, lw=1.3, label=m)
                axes[1].plot(t * (1 + table[m]["cost_ratio"]), e, color=col[m], ls=ls, lw=1.3, label=m)
            axes[0].set_yscale("log"); axes[0].set_xlabel("physical t"); axes[0].set_ylabel("median $e_F(t)$ (raw bins)"); axes[0].legend(fontsize=6, frameon=False, ncol=2)
            axes[1].set_xscale("log"); axes[1].set_xlabel("compute (outer-time-equivalent, incl. relaxation)")
            for ax in axes:
                ax.grid(alpha=0.25, which="both")
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(fd, f"fig_K_confirmatory_two_axes.{ext}"), dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"  (figures skipped: {e})")
    print(f"  wrote {os.path.relpath(out_dir, ROOT)}/analysis.json, comparison.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["D0", "D1", "D2"], required=True)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    raw = a.raw or os.path.join(DIRS[a.stage], "raw.npz")
    out_dir = a.out_dir or os.path.dirname(raw)
    {"D0": stage_d0, "D1": stage_d1, "D2": stage_d2}[a.stage](raw, out_dir, a.no_figures)


if __name__ == "__main__":
    main()
