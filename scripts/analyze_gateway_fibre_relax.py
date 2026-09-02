#!/usr/bin/env python
"""Analyzer for the fibre-relaxation campaign -- written BEFORE its data.  Stage C1 and C2.

Prereg: configs/transport_campaign/gateway_fibre_relax_prereg.json.  C1: the read-out plateau
INTERSECTION rule first (h_read**), then the recovery curves R_X(c), the flank excess E(c), the
D_cond peaks, the matched-fibre contrasts T_c vs A_c / F_c, costs and tau_move, and the frozen
c* rule -> c_selection.json.  C2: the confirmatory contrasts and verdicts at h_read**.

    python scripts/analyze_gateway_fibre_relax.py --stage C1
    python scripts/analyze_gateway_fibre_relax.py --stage C2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
ROOT = os.path.join(SCRIPTS, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import analyze_gateway_horizontal_transport as A1                               # noqa: E402
from analyze_gateway_horizontal_transport import contrast, decide_primary, verdict_vs_abf, fmt, MARGIN   # noqa: E402
from analyze_uniform_gateway import tau, PERSIST                                # noqa: E402
from analyze_gateway_bandwidth_audit import mean_force_at, e_f, label, cumtrapz    # noqa: E402
from eb_abffr_core import EVAL_LO, EVAL_HI                                     # noqa: E402

PREREG = os.path.join(ROOT, "configs/transport_campaign/gateway_fibre_relax_prereg.json")
CAMPAIGN = os.path.join(ROOT, "results/transport_campaign/gateway_horizontal")
C1_DIR, C2_DIR = os.path.join(CAMPAIGN, "relax_C1"), os.path.join(CAMPAIGN, "relax_C2")
READOUT_LADDER = (0.0175, 0.00875, 0.004375, 0.0)
PLATEAU_TOL = 0.02
R_TARGET = 0.90
ALLOC = dict(A="abf", F="fr_uniform", T="ot_exact", P="ot_full")
INF = float("inf")
T_DOSE_START = 4.0


def name_of(alloc, c):
    base = ALLOC[alloc]
    return base if c == 0 else (f"{base}_refresh" if math.isinf(c) else f"{base}_relax{c:g}")


def select_h(ladder, e_per_arm):
    """Plateau intersection: largest h such that EVERY arm's median e_F(T; h) <= 1.02 x its own min over the ladder."""
    ok = []
    for h in sorted(ladder, reverse=True):
        good = all(e[label(h)] <= (1 + PLATEAU_TOL) * min(e.values()) for e in e_per_arm.values())
        ok.append((h, good))
    for h, good in ok:
        if good:
            return h
    return 0.0


def select_c(ladder, R_T, dI_T_vs_A):
    """Smallest c with median R_T(c) >= 0.9 AND median Delta I_F(T_c vs A_c) <= 0; else 5 (bucket C4)."""
    for c in sorted(float(x) for x in ladder):
        if R_T[f"{c:g}"] >= R_TARGET and dI_T_vs_A[f"{c:g}"] <= 0.0:
            return c
    return max(float(x) for x in ladder)


def load(raw):
    z = np.load(raw, allow_pickle=True)
    method = np.array([str(m) for m in z["method"]]); init = np.array([str(i) for i in z["init"]]); seed = z["seed"].astype(int)
    rows = {}
    for i in range(len(method)):
        rows.setdefault((init[i], seed[i]), {})[method[i]] = i
    arms = sorted(set(method))
    keys = sorted(k for k, v in rows.items() if set(v) >= set(arms))
    idx = {m: np.array([rows[k][m] for k in keys]) for m in arms}
    clusters = np.array([k[1] for k in keys]); init_of = np.array([k[0] for k in keys])
    x = np.asarray(z["x_grid"][0], float); dx = float(x[1] - x[0]); mask = (x >= EVAL_LO) & (x <= EVAL_HI)
    t = np.asarray(z["t"][0], float); cfg0 = json.loads(str(z["config_json"][0]))
    return z, arms, keys, idx, clusters, init_of, x, dx, mask, t, cfg0


def endpoints(z, x, dx, mask, t, cfg0, h):
    Sf, C, F_ref, Fp_ref = (np.asarray(z[k], float) for k in ("Sf_t", "C_t", "F_ref", "Fp_ref"))
    Fp = mean_force_at(Sf, C, h, dx, float(cfg0["min_count"]))
    e = e_f(Fp, F_ref, dx, mask); I = np.trapezoid(e, t, axis=1); fin = e[:, -1]
    dFp_T = Fp[:, -1, :] - Fp_ref; Lf = (x > -0.3) & (x < -0.05); b_flank = dFp_T[:, Lf].mean(1)
    F_T = cumtrapz(Fp[:, -1, :], dx); i0, im = int(np.argmin(abs(x))), int(np.argmin(abs(x + 1)))
    bar = ((F_T[:, i0] - F_T[:, im]) - (F_ref[:, i0] - F_ref[:, im])) * float(cfg0["beta"])
    return dict(e=e, I=I, fin=fin, b_flank=b_flank, barrier_kT=bar)


def med(v):
    return float(np.median(v))


# ----------------------------------------------------------------------------- Stage C1
def stage_c1(raw, out_dir, no_figures):
    pre = json.load(open(PREREG))
    z, arms, keys, idx, clusters, init_of, x, dx, mask, t, cfg0 = load(raw)
    ladder = [float(c) for c in pre["ladder_c"]]; cs = [0.0] + ladder + [INF]
    print(f"C1: rows {len(keys)}, seeds {len(set(clusters))}, arms {len(arms)}")
    Sf, C, F_ref = (np.asarray(z[k], float) for k in ("Sf_t", "C_t", "F_ref"))
    # ---- 1. read-out rule FIRST (no contrast yet) ----
    e_per_arm = {}
    for m in arms:
        e_per_arm[m] = {label(h): med(e_f(mean_force_at(Sf[idx[m], -1:, :], C[idx[m], -1:, :], h, dx, 1.0), F_ref[idx[m]], dx, mask)[:, 0])
                        for h in READOUT_LADDER}
    h2 = select_h(READOUT_LADDER, e_per_arm)
    print(f"read-out plateau intersection over {len(arms)} arms (tol {PLATEAU_TOL:.0%}): h_read** = {label(h2)}")
    worst = sorted(((m, e[label(0.0175)] / min(e.values()) - 1) for m, e in e_per_arm.items()), key=lambda kv: -kv[1])[:4]
    print("  most off-plateau at 0.0175: " + ", ".join(f"{m} +{v:.0%}" for m, v in worst))
    # ---- 2. endpoints at h** (and raw) ----
    E = {label(h): endpoints(z, x, dx, mask, t, cfg0, h) for h in (h2, 0.0)}
    hs = label(h2); P = E[hs]
    kl = np.asarray(z["kl_uniform_t"], float); dc = np.nan_to_num(np.asarray(z["dcond_t"], float))
    early = t <= 5.0 + 1e-12
    fz = np.asarray(z["frozen_l2_f_kT"], float) if "frozen_l2_f_kT" in z.files else None
    table = {}
    print(f"\n{'arm':>22} {'I_F':>8} {'e_F(T)':>8} {'e_F(T) raw':>10} {'b_flank':>8} {'bar kT':>7} {'Dcond pk':>8} {'frozen':>7} {'cost':>7} {'tau_mv':>7}")
    for al in ("A", "F", "T", "P"):
        for c in cs:
            m = name_of(al, c); r = idx[m]
            d = dict(alloc=al, c=c, I_F=med(P["I"][r]), eF_T=med(P["fin"][r]), eF_T_raw=med(E["raw"]["fin"][r]),
                     b_flank=med(P["b_flank"][r]), barrier_kT=med(P["barrier_kT"][r]),
                     dcond_peak=med(dc[r][:, early].max(1)), int_dcond=med(np.trapezoid(dc[r], t, axis=1)),
                     J_KL=med(np.trapezoid(kl[r][:, t >= T_DOSE_START - 1e-12], t[t >= T_DOSE_START - 1e-12], axis=1)),
                     frozen=(med(fz[r]) if fz is not None else None),
                     fibre_cost_ratio=(med(np.asarray(z["fibre_cost_ratio"], float)[r]) if "fibre_cost_ratio" in z.files and 0 < c < INF else 0.0),
                     tau_move=(med(np.mean(np.asarray(z["ot_tau_move_t"], float)[r][:, 1:], axis=1)) if al in ("T", "P") else None))
            table[m] = d
            print(f"{m:>22} {d['I_F']:8.4f} {d['eF_T']:8.5f} {d['eF_T_raw']:10.5f} {d['b_flank']:+8.4f} {d['barrier_kT']:+7.3f} {d['dcond_peak']:8.4f} "
                  f"{(d['frozen'] if d['frozen'] is not None else float('nan')):7.4f} {d['fibre_cost_ratio']:7.1f} {(d['tau_move'] if d['tau_move'] is not None else float('nan')):7.4f}")
    # ---- 3. recovery curves, flank excess, matched-fibre contrasts ----
    k = 0
    rec = {al: {} for al in ALLOC}
    print("\nrecovery R_X(c) = (I_F(X_0) - I_F(X_c)) / (I_F(X_0) - I_F(X_inf)), median [CI95]; analytic 1 - e^{-2c}")
    for al in ("A", "F", "T", "P"):
        I0, Iinf = P["I"][idx[name_of(al, 0)]], P["I"][idx[name_of(al, INF)]]
        for c in ladder:
            R = (I0 - P["I"][idx[name_of(al, c)]]) / (I0 - Iinf)
            ci95, _ = A1.boot_cluster(R, clusters, 20260904 + k); k += 1
            rec[al][f"{c:g}"] = dict(median=med(R), ci95=ci95, analytic=1 - math.exp(-2 * c))
        print(f"  {al}: " + "  ".join(f"c={c:g}: {rec[al][f'{c:g}']['median']:.3f} [{rec[al][f'{c:g}']['ci95'][0]:.2f},{rec[al][f'{c:g}']['ci95'][1]:.2f}] (an. {1 - math.exp(-2 * c):.3f})" for c in ladder))
    flank = {}
    E0 = med(P["b_flank"][idx["ot_exact"]] - P["b_flank"][idx["abf"]])
    for c in cs:
        Ec = med(P["b_flank"][idx[name_of("T", c)]] - P["b_flank"][idx[name_of("A", c)]])
        flank[f"{c:g}"] = dict(excess=Ec, normalised=(Ec / E0 if E0 != 0 else float("nan")), analytic=(math.exp(-2 * c) if not math.isinf(c) else 0.0))
    print("flank excess E(c) = b_flank(T_c) - b_flank(A_c), normalised by E(0); analytic e^{-2c}: "
          + "  ".join(f"c={c}: {v['normalised']:.3f} ({v['analytic']:.3f})" for c, v in flank.items()))
    mono = all(flank[f"{ladder[i]:g}"]["normalised"] >= flank[f"{ladder[i + 1]:g}"]["normalised"] for i in range(len(ladder) - 1))
    print(f"  E monotone decreasing in c: {mono}; E(2)/E(0) = {flank['2']['normalised']:.3f} (< 0.2 predicted)")
    matched = {}
    print("matched-fibre contrasts at h_read** (Delta I_F / Delta e_F(T), cluster bootstrap over 8 seeds -- descriptive):")
    for c in cs:
        mt = {}
        for ref_al in ("A", "F"):
            ci = contrast(P["I"][idx[name_of("T", c)]], P["I"][idx[name_of(ref_al, c)]], clusters, 500 + k); cf = contrast(P["fin"][idx[name_of("T", c)]], P["fin"][idx[name_of(ref_al, c)]], clusters, 501 + k); k += 2
            mt[f"T_vs_{ref_al}"] = dict(d_int=ci, d_fin=cf)
        ci = contrast(P["I"][idx[name_of("P", c)]], P["I"][idx[name_of("A", c)]], clusters, 502 + k); k += 1
        mt["P_vs_A"] = dict(d_int=ci)
        matched[f"{c:g}"] = mt
        print(f"  c={c:g}: T vs A {fmt(mt['T_vs_A']['d_int'])} / final {mt['T_vs_A']['d_fin']['median']:+.1f}%;  T vs F {fmt(mt['T_vs_F']['d_int'])};  P vs A {fmt(mt['P_vs_A']['d_int'])}")
    # ---- 4. c* rule ----
    R_T = {c: rec["T"][c]["median"] for c in rec["T"]}
    dI = {f"{c:g}": matched[f"{c:g}"]["T_vs_A"]["d_int"]["median"] for c in ladder}
    c_star = select_c(ladder, R_T, dI)
    qualifies = any(R_T[f"{c:g}"] >= R_TARGET and dI[f"{c:g}"] <= 0 for c in ladder)
    bucket = ("C4_not_recovered" if not qualifies else "C1_gentle" if c_star <= 0.5 else "C2_moderate" if c_star <= 2 else "C3_full_equilibration")
    # time-to-accuracy on both axes for the T and P arms
    e_abf = np.median(P["e"][idx["abf"]], axis=0); eps = float(e_abf[-1])
    tta = {}
    for m in arms:
        tm = tau(t, np.median(P["e"][idx[m]], axis=0), eps, PERSIST)
        rho = table[m]["fibre_cost_ratio"]
        tta[m] = dict(tau=tm, tau_eff=(tm * (1 + rho) if np.isfinite(tm) else tm), cost_ratio=rho)
    print("\ntime to ABF-final accuracy (physical t | t_eff = t (1 + fibre cost ratio)):")
    for al in ("A", "F", "T", "P"):
        print(f"  {al}: " + "  ".join(f"c={c:g}: {tta[name_of(al, c)]['tau']:.1f} | {tta[name_of(al, c)]['tau_eff']:.1f}" for c in cs))
    print(f"\n  c* = {c_star:g} (rule: smallest c with R_T >= {R_TARGET} and Delta I_F(T_c vs A_c) <= 0; qualifies: {qualifies}) -> bucket {bucket}")
    print(f"  cost at c*: fibre_cost_ratio {table[name_of('T', c_star)]['fibre_cost_ratio']:.1f} (all-walker relaxation); tau_move {table[name_of('T', c_star)]['tau_move']:.4f}")
    sel = dict(prereg=os.path.relpath(PREREG, ROOT), raw=os.path.relpath(raw, ROOT), ladder_c=ladder,
               readout_ladder=list(READOUT_LADDER), readout_eF_T_median_per_arm=e_per_arm, h_read_star2=h2,
               R_T_median=R_T, dI_T_vs_A_median=dI, c_star=c_star, qualifies=bool(qualifies), bucket=bucket,
               rule=f"smallest c with R_T >= {R_TARGET} and median dI_F(T_c vs A_c) <= 0; else max ladder c (C4)")
    with open(os.path.join(out_dir, "c_selection.json"), "w") as fh:
        json.dump(sel, fh, indent=2, default=float)
    summary = dict(sel, per_arm=table, recovery=rec, flank_excess=flank, flank_monotone=bool(mono), matched_fibre=matched, time_to_accuracy=tta)
    with open(os.path.join(out_dir, "analysis.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    if not no_figures:
        figs(out_dir, t, ladder, cs, rec, flank, P, idx, dc, table, hs)
    print(f"  wrote {os.path.relpath(out_dir, ROOT)}/c_selection.json, analysis.json")


def figs(out_dir, t, ladder, cs, rec, flank, P, idx, dc, table, hs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  (figures skipped: {e})"); return
    fd = os.path.join(out_dir, "figures"); os.makedirs(fd, exist_ok=True)
    col = dict(A="#4d4d4d", F="#d95f02", T="#1b9e77", P="#7570b3")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    cgrid = np.linspace(0, 5, 200)
    axes[0].plot(cgrid, 1 - np.exp(-2 * cgrid), "k--", lw=1, label=r"$1-e^{-2c}$")
    for al in ("A", "F", "T", "P"):
        v = [rec[al][f"{c:g}"]["median"] for c in ladder]; lo = [rec[al][f"{c:g}"]["ci95"][0] for c in ladder]; hi = [rec[al][f"{c:g}"]["ci95"][1] for c in ladder]
        axes[0].errorbar(ladder, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="o-", color=col[al], ms=4, capsize=2, label=f"{al}: {ALLOC[al]}")
    axes[0].set_xlabel("c (local relaxation times)"); axes[0].set_ylabel(r"recovery $R_X(c)$"); axes[0].set_ylim(-0.1, 1.15); axes[0].legend(fontsize=7, frameon=False)
    axes[1].plot(cgrid, np.exp(-2 * cgrid), "k--", lw=1, label=r"$e^{-2c}$")
    axes[1].plot([c for c in cs if not math.isinf(c)], [flank[f"{c:g}"]["normalised"] for c in cs if not math.isinf(c)], "o-", color=col["T"], ms=4, label="E(c)/E(0)")
    axes[1].set_yscale("symlog", linthresh=0.01); axes[1].set_xlabel("c"); axes[1].set_ylabel("normalised flank excess"); axes[1].legend(fontsize=7, frameon=False)
    for al in ("A", "T"):
        for c, ls in zip(cs, ("-", ":", "-.", "--", (0, (1, 1)), (0, (5, 1)))):
            m = name_of(al, c); axes[2].plot(t, np.median(P["e"][idx[m]], axis=0), color=col[al], ls=ls, lw=1.2, label=f"{m}")
    axes[2].set_yscale("log"); axes[2].set_xlabel("t"); axes[2].set_ylabel(f"median $e_F(t)$ at $h_{{read}}$ = {hs}"); axes[2].legend(fontsize=5, frameon=False, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fd, f"fig_G_recovery_curves.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- Stage C2
def stage_c2(raw, out_dir, no_figures):
    A1.BOOT_SEED = 20260904
    z, arms, keys, idx, clusters, init_of, x, dx, mask, t, cfg0 = load(raw)
    inits = sorted(str(i) for i in set(init_of))
    c_star = float(z["c_star"]); h2 = float(z["h_read_star2"])
    A0, F0, T0 = "abf", "fr_uniform", "ot_exact"
    Ac, Fc, Tc, Pc = (name_of(al, c_star) for al in ("A", "F", "T", "P"))
    assert set(arms) == {A0, F0, T0, Ac, Fc, Tc, Pc}, arms
    print(f"C2: rows {len(keys)}, seeds {len(set(clusters))}; c* {c_star:g}; h_read** {label(h2)}; arms {arms}")
    E = {label(h): endpoints(z, x, dx, mask, t, cfg0, h) for h in (h2, 0.0175, 0.0)}
    dev = float(np.abs(e_f(mean_force_at(np.asarray(z["Sf_t"], float), np.asarray(z["C_t"], float), float(cfg0["h"]), dx, 1.0), np.asarray(z["F_ref"], float), dx, mask) - np.asarray(z["l2_f_t"], float)).max())
    print(f"self-check: offline read-out at h_bias vs engine, max|dev| {dev:.2e}"); assert dev < 1e-9
    NAMED = [("PRIMARY", Tc, Ac), ("S1", Tc, Fc), ("S2", Fc, Ac), ("S3", Ac, A0), ("S4", Tc, T0), ("S5", Pc, Ac), ("S6", Pc, Fc),
             ("V_F0", F0, A0), ("V_T0", T0, A0), ("V_Fc", Fc, A0), ("V_Tc", Tc, A0), ("V_Pc", Pc, A0), ("S7", Fc, F0)]
    res, k = {}, 0
    print(f"\n{'read-out':>8} {'tag':>7} {'contrast':>40} {'d I_F':>8} {'CI95':>20} {'wins':>6} {'d e_F(T)':>9} {'CI95':>20}")
    for lab in (label(h2), "0.0175", "raw"):
        if lab in res:
            continue
        P = E[lab]; res[lab] = {}
        for tag, arm, ref in NAMED:
            ci = contrast(P["I"][idx[arm]], P["I"][idx[ref]], clusters, k); cf = contrast(P["fin"][idx[arm]], P["fin"][idx[ref]], clusters, k + 1); k += 2
            pi = {}
            if lab == label(h2):
                for ini in inits:
                    sel = np.nonzero(init_of == ini)[0]
                    pi[ini] = dict(d_int=contrast(P["I"][idx[arm]][sel], P["I"][idx[ref]][sel], clusters[sel], k)); k += 1
            res[lab][tag] = dict(arm=arm, ref=ref, d_int=ci, d_fin=cf, per_init=pi)
            print(f"{lab:>8} {tag:>7} {arm + ' vs ' + ref:>40} {ci['median']:+8.2f} [{ci['ci95'][0]:+8.2f},{ci['ci95'][1]:+8.2f}] {ci['wins']:3d}/{ci['n']} "
                  f"{cf['median']:+9.2f} [{cf['ci95'][0]:+8.2f},{cf['ci95'][1]:+8.2f}]" + ("" if not pi else "  per-init " + ", ".join(f"{i}: {v['d_int']['median']:+.1f}%" for i, v in pi.items())))
    fz = np.asarray(z["frozen_l2_f_kT"], float) if "frozen_l2_f_kT" in z.files else None
    froz = None
    if fz is not None:
        froz = {tag: contrast(fz[idx[arm]], fz[idx[ref]], clusters, 700 + j) for j, (tag, arm, ref) in enumerate(NAMED)}
        print("frozen-bias: " + "; ".join(f"{tag} {fmt(c)}" for tag, c in froz.items()))
    P = E[label(h2)]
    mech = {m: dict(b_flank=med(P["b_flank"][idx[m]]), barrier_kT=med(P["barrier_kT"][idx[m]]), eF_T=med(P["fin"][idx[m]]), eF_T_raw=med(E["raw"]["fin"][idx[m]]),
                    fibre_cost_ratio=(med(np.asarray(z["fibre_cost_ratio"], float)[idx[m]]) if "fibre_cost_ratio" in z.files else 0.0)) for m in arms}
    for m in (F0, Fc):
        ess = np.asarray(z["min_ess_frac"], float)[idx[m]]; wm = np.asarray(z["max_wmax"], float)[idx[m]]
        mech[m].update(median_min_ess_frac=med(ess), median_max_wmax=med(wm))
    print("mechanism: " + "; ".join(f"{m}: b_flank {v['b_flank']:+.4f}, barrier {v['barrier_kT']:+.3f} kT, e_F(T) {v['eF_T']:.5f} (raw {v['eF_T_raw']:.5f}), cost {v['fibre_cost_ratio']:.1f}" for m, v in mech.items()))
    e_abf = np.median(P["e"][idx[A0]], axis=0); eps = float(e_abf[-1])
    tta = {m: dict(tau=tau(t, np.median(P["e"][idx[m]], axis=0), eps, PERSIST), cost_ratio=mech[m]["fibre_cost_ratio"]) for m in arms}
    for m in arms:
        tta[m]["tau_eff"] = tta[m]["tau"] * (1 + tta[m]["cost_ratio"]) if np.isfinite(tta[m]["tau"]) else tta[m]["tau"]
    print("time to ABF-final accuracy (t | t_eff): " + "; ".join(f"{m} {v['tau']:.1f} | {v['tau_eff']:.1f}" for m, v in tta.items()))
    Pr = res[label(h2)]
    health = {m: mech[m]["median_min_ess_frac"] >= 0.30 and mech[m]["median_max_wmax"] <= 0.05 for m in (F0, Fc)}
    prim = verdict_vs_abf(Pr["PRIMARY"]["d_int"], Pr["PRIMARY"]["d_fin"])
    s1 = decide_primary(Pr["S1"]["d_int"]); s2 = verdict_vs_abf(Pr["S2"]["d_int"], Pr["S2"]["d_fin"], health[Fc]); s3 = verdict_vs_abf(Pr["S3"]["d_int"], Pr["S3"]["d_fin"])
    s5 = verdict_vs_abf(Pr["S5"]["d_int"], Pr["S5"]["d_fin"]); s6 = decide_primary(Pr["S6"]["d_int"])
    replicated = Pr["V_F0"]["d_int"]["ci95"][1] < 0
    deployable = (prim.startswith("SAFE") or prim.startswith("ACCELERATION")) and s1 in ("OT_better", "equivalent")
    prim_raw = verdict_vs_abf(res["raw"]["PRIMARY"]["d_int"], res["raw"]["PRIMARY"]["d_fin"])
    outcome = ("FAILED_REPLICATION_OF_POSITIVE_CONTROL" if not replicated else
               "DEPLOYABLE_ALLOCATOR" if deployable else "TRANSPORT_NOT_BETTER_AT_FINITE_C")
    print(f"\n  PRIMARY T_c* vs A_c*: {fmt(Pr['PRIMARY']['d_int'])} final {Pr['PRIMARY']['d_fin']['median']:+.2f}% -> {prim} (raw bins: {prim_raw})")
    print(f"  S1 T_c* vs F_c*: {fmt(Pr['S1']['d_int'])} -> {s1};  S2 F_c* vs A_c*: {s2};  S3 A_c* vs A_0: {fmt(Pr['S3']['d_int'])} -> {s3}")
    print(f"  S4 T_c* vs T_0 (finite-c repair): {fmt(Pr['S4']['d_int'])};  S5 P_c* vs A_c*: {fmt(Pr['S5']['d_int'])} -> {s5};  S6 P_c* vs F_c*: {s6}")
    print(f"  positive control replicated: {replicated};  OUTCOME: {outcome}")
    summary = dict(prereg=os.path.relpath(PREREG, ROOT), raw=os.path.relpath(raw, ROOT), n_rows=len(keys), n_seeds=len(set(clusters)), c_star=c_star, h_read_star2=h2,
                   per_readout=res, frozen_bias=froz, mechanism=mech, time_to_accuracy=tta, health=health,
                   decisions=dict(PRIMARY=prim, PRIMARY_raw=prim_raw, S1=s1, S2=s2, S3=s3, S5=s5, S6=s6), positive_control_replicated=bool(replicated), outcome=outcome)
    with open(os.path.join(out_dir, "analysis.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(os.path.join(out_dir, "comparison.csv"), "w") as fh:
        labs = [label(h2), "0.0175", "raw"]
        fh.write("init,seed," + ",".join(f"{tag}_{q}_{lab}" for lab in labs for tag, _, _ in NAMED for q in ("dI", "dF")) + "\n")
        for j, kk in enumerate(keys):
            fh.write(f"{kk[0]},{kk[1]}," + ",".join(f"{res[lab][tag][q]['per_row'][j]:.3f}" for lab in labs for tag, _, _ in NAMED for q in ("d_int", "d_fin")) + "\n")
    if not no_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fd = os.path.join(out_dir, "figures"); os.makedirs(fd, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 3.8))
            col = {A0: "#4d4d4d", F0: "#d95f02", T0: "#1b9e77", Ac: "#4d4d4d", Fc: "#d95f02", Tc: "#1b9e77", Pc: "#7570b3"}
            for m in arms:
                ax.plot(t, np.median(P["e"][idx[m]], axis=0), color=col[m], ls=("--" if "relax" in m else "-"), lw=1.4, label=m)
            ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel(f"median $e_F(t)$ at $h_{{read}}$ = {label(h2)}"); ax.legend(fontsize=6, frameon=False, ncol=2); ax.grid(alpha=0.25, which="both")
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(fd, f"fig_H_confirmatory_eF.{ext}"), dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"  (figures skipped: {e})")
    print(f"  wrote {os.path.relpath(out_dir, ROOT)}/analysis.json, comparison.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["C1", "C2"], required=True)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    d = C1_DIR if a.stage == "C1" else C2_DIR
    raw = a.raw or os.path.join(d, "raw.npz")
    out_dir = a.out_dir or os.path.dirname(raw)
    (stage_c1 if a.stage == "C1" else stage_c2)(raw, out_dir, a.no_figures)


if __name__ == "__main__":
    main()
