#!/usr/bin/env python
"""Analyzer for the histogram-ABF campaign, entropic gateway (Experiment C; docs/HISTOGRAM_ABF_REPLICATION.md).

  --select      C1: ABF-only bin-width calibration on the frozen rule (configs/histogram_abf/campaign.json),
                writes calibration/summary.json + figure, FREEZES the width in selected_bins.json (gateway entry).
  (default)     C2: four-arm confirmation.  Kernel arms scored by the accepted corrected convention: the
                fine-grid accumulators read out offline at h_read* = 0.0175 (analyze_gateway_bandwidth_audit.
                mean_force_at / e_f, bit for bit the accepted machinery), legacy 0.07 (= the engine's own
                profile) alongside; histogram arms by their own bins.  FR effect within each estimator
                (paired by (init, seed) row), replication verdict, absolute accuracy, safety (gateway floors),
                per-init breakdown, figures.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPTS, "..")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
from analyze_uniform_lta import BOOT_SEED, boot_median                    # noqa: E402
from analyze_gateway_bandwidth_audit import mean_force_at, e_f            # noqa: E402  (accepted read-out machinery)
from analyze_histogram_abf_entropic import load_batch, paired, med_iqr, _mpl, _save, COL, LS   # noqa: E402
from eb_abffr_core import EVAL_LO, EVAL_HI, XMIN, XMAX                    # noqa: E402

CAMPAIGN = os.path.join(ROOT, "configs", "histogram_abf", "campaign.json")
SELECTED = os.path.join(ROOT, "configs", "histogram_abf", "selected_bins.json")
RES = os.path.join(ROOT, "results", "histogram_abf", "gateway")


def by_arm(rows):
    """rows keyed by (method) -> {(init, seed): row}; the pairing unit is the (init, seed) row."""
    out = {}
    for r in rows:
        out.setdefault(str(r["method"]), {})[(str(r["init"]), int(r["seed"]))] = r
    return out


def kernel_readout(row, h):
    """e_F(t), e_F'(t) of a kernel row at read-out bandwidth h from its saved fine-grid accumulators."""
    x = np.asarray(row["x_grid"], float); dx = float(x[1] - x[0]); mask = (x >= EVAL_LO) & (x <= EVAL_HI)
    cfg = row["config"]
    Fp = mean_force_at(np.asarray(row["Sf_t"], float), np.asarray(row["C_t"], float), h, dx, float(cfg["min_count"]))
    eF = e_f(Fp, np.asarray(row["F_ref"], float), dx, mask)
    d = (Fp - np.asarray(row["Fp_ref"], float)[None, :])[:, mask]
    return eF, np.sqrt((d * d).mean(-1)), Fp[-1]


# ------------------------------------------------------------------------------------------ C1
def select(a):
    cj = json.load(open(CAMPAIGN)); c, rule = cj["gateway"], cj["bin_selection_rule_frozen"]
    floors = {int(r["n_bins"]): r for r in json.load(open(os.path.join(RES, "floor", "p0_floor.json")))["rows"]}
    ladder = [int(n) for n in c["ladder_n_bins"]]
    cal = {}
    for nb in ladder:
        rows, meta = load_batch(os.path.join(RES, "calibration", "raw", f"hist_nbins{nb}.npz"))
        ba = by_arm(rows); assert set(ba) == {"abf"}, set(ba)
        cal[nb] = dict(rows=ba["abf"], meta=meta)
    keys = sorted(cal[ladder[0]]["rows"])
    assert sorted({s for _, s in keys}) == sorted(c["calibration"]["seeds"])
    kp = os.path.join(RES, "calibration", "raw", "kernel.npz")
    kctx = None
    if os.path.exists(kp):
        krows, kmeta = load_batch(kp); kctx = dict(rows=by_arm(krows)["abf"], meta=kmeta)

    def vec(nb, key):
        return np.array([float(cal[nb]["rows"][k][key]) for k in keys])
    table = []
    for i, nb in enumerate(ladder):
        eF, IF, eFp = vec(nb, "final_l2_f"), vec(nb, "int_l2_f"), vec(nb, "final_l2_fp")
        row = dict(n_bins=nb, delta=float(3.6 / nb), final_l2_f=med_iqr(eF), int_l2_f=med_iqr(IF), final_l2_fp=med_iqr(eFp),
                   final_l2_fp_nodes=med_iqr(vec(nb, "final_l2_fp_nodes")), floor_l2_f=floors[nb]["floor_l2_f"], floor_l2_fp=floors[nb]["floor_l2_fp"],
                   share_f=floors[nb]["floor_l2_f"] / float(np.median(eF)), share_fp=floors[nb]["floor_l2_fp"] / float(np.median(eFp)),
                   min_count_window=med_iqr(vec(nb, "min_count_window")), frac_untrusted_window=med_iqr(vec(nb, "frac_untrusted_window")),
                   max_abs_bias_force=med_iqr(vec(nb, "max_abs_bias_force")), ms_per_step=1e3 * cal[nb]["meta"]["wall_seconds_per_step"],
                   wall_seconds=cal[nb]["meta"]["wall_seconds"])
        row["pass_floor"] = bool(row["share_f"] <= 0.20 and row["share_fp"] <= 0.20)
        if i + 1 < len(ladder):
            nf = ladder[i + 1]; row["next_n_bins"] = nf
            row["to_finer"] = dict(final_l2_f=paired(vec(nf, "final_l2_f"), eF, 3 * i), int_l2_f=paired(vec(nf, "int_l2_f"), IF, 3 * i + 1),
                                   final_l2_fp=paired(vec(nf, "final_l2_fp"), eFp, 3 * i + 2))
            dF, dI, dP = (abs(row["to_finer"][k]["median"]) for k in ("final_l2_f", "int_l2_f", "final_l2_fp"))
            row["pass_plateau"] = bool(dF <= 5.0 and dI <= 5.0 and dP <= 10.0); row["plateau_violation"] = float(max(dF / 5.0, dI / 5.0, dP / 10.0))
        else:
            row["next_n_bins"], row["pass_plateau"], row["plateau_violation"] = None, None, None
        table.append(row)
    passing = [r for r in table if r["pass_floor"] and r["pass_plateau"]]
    if passing:
        chosen, how = passing[0], "coarsest width passing (a) floor and (b) plateau"
    else:
        cands = [r for r in table if r["pass_floor"] and r["plateau_violation"] is not None] or [r for r in table if r["plateau_violation"] is not None]
        chosen = min(cands, key=lambda r: r["plateau_violation"]); how = "NO width passed the frozen rule; predeclared fallback: smallest normalised plateau violation among widths passing (a)"
    print(f"\nGateway C1 calibration (ABF only, {len(keys)} rows = {len(c['calibration']['seeds'])} seeds x {c['inits']})")
    print(f"  {'n_bins':>6} {'Delta':>6} {'e_F(T)':>8} {'I_F':>7} {'e_Fp(T)':>8} {'floorF':>7} {'floorFp':>7} {'shareF':>6} {'shareFp':>7} {'dF%':>6} {'dI%':>6} {'dFp%':>6} {'floor':>5} {'plat':>5} {'mincnt':>7} {'ms/step':>7}")
    for r in table:
        tf = r.get("to_finer"); d = (f"{tf['final_l2_f']['median']:+6.1f} {tf['int_l2_f']['median']:+6.1f} {tf['final_l2_fp']['median']:+6.1f}" if tf else " " * 20)
        print(f"  {r['n_bins']:6d} {r['delta']:6.3f} {r['final_l2_f']['median']:8.5f} {r['int_l2_f']['median']:7.4f} {r['final_l2_fp']['median']:8.4f} {r['floor_l2_f']:7.5f} {r['floor_l2_fp']:7.4f} "
              f"{r['share_f']:6.3f} {r['share_fp']:7.3f} {d} {str(r['pass_floor']):>5} {str(r['pass_plateau']):>5} {r['min_count_window']['median']:7.0f} {r['ms_per_step']:7.3f}")
    if kctx:
        kr = kctx["rows"]
        # context: the kernel's own h = 0.07 numbers and the accepted h_read* read-out
        eFs = np.array([kernel_readout(kr[k], 0.0175)[0][-1] for k in keys])
        print(f"  kernel h=0.07 (context only): own e_F(T) {np.median([kr[k]['final_l2_f'] for k in keys]):.5f}, at h_read* 0.0175 {np.median(eFs):.5f}; "
              f"own I_F {np.median([kr[k]['int_l2_f'] for k in keys]):.4f}; own e_F'(T) {np.median([kr[k]['final_l2_fp'] for k in keys]):.4f}; {1e3 * kctx['meta']['wall_seconds_per_step']:.3f} ms/step")
    print(f"  SELECTED n_bins = {chosen['n_bins']} (Delta {chosen['delta']:.3f}): {how}")
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1)); deltas = [r["delta"] for r in table]
    for ax, key, fkey, lab in zip(axes, ("final_l2_f", "int_l2_f", "final_l2_fp"), ("floor_l2_f", None, "floor_l2_fp"), ("final $e_F$", "$I_F$", "final $e_{F'}$")):
        med = [r[key]["median"] for r in table]; lo = [r[key]["iqr"][0] for r in table]; hi = [r[key]["iqr"][1] for r in table]
        ax.errorbar(deltas, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="o-", color=COL["histogram"], label="histogram ABF (own)")
        if fkey:
            ax.plot(deltas, [r[fkey] for r in table], "s:", color="0.3", label="deterministic P0 floor"); ax.set_yscale("log")
        if kctx:
            ax.axhline(np.median([kctx["rows"][k][key] for k in keys]), color=COL["kernel"], ls="-.", label="kernel ABF h=0.07 own (context)")
        ax.axvline(chosen["delta"], color="k", lw=0.8, alpha=0.5); ax.set_xscale("log"); ax.set_xlabel(r"bin width $\Delta$"); ax.set_title(lab)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Entropic gateway, ABF-only bin-width calibration ({len(keys)} rows); selected Delta = {chosen['delta']:.3f}", fontsize=9)
    fig_path = _save(fig, os.path.join(RES, "figures"), "gateway_calibration")
    summary = dict(system="gateway", rows=[list(k) for k in keys], ladder_n_bins=ladder, rule=rule, table=table, selected_n_bins=chosen["n_bins"],
                   selected_delta=chosen["delta"], selection_how=how, all_passed_rule=bool(passing), figure=os.path.relpath(fig_path, ROOT),
                   timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if kctx:
        summary["kernel_context"] = dict(final_l2_f_own=med_iqr([kctx["rows"][k]["final_l2_f"] for k in keys]), final_l2_f_hstar=med_iqr(eFs),
                                         int_l2_f_own=med_iqr([kctx["rows"][k]["int_l2_f"] for k in keys]), ms_per_step=1e3 * kctx["meta"]["wall_seconds_per_step"])
    os.makedirs(os.path.join(RES, "calibration"), exist_ok=True)
    json.dump(summary, open(os.path.join(RES, "calibration", "summary.json"), "w"), indent=2, default=float)
    sel = json.load(open(SELECTED)) if os.path.exists(SELECTED) else {}
    if "gateway" in sel and not a.reselect:
        print(f"  selected_bins.json already carries gateway = {sel['gateway']['selected_n_bins']} (frozen); not rewritten")
    else:
        sel["gateway"] = dict(selected_n_bins=chosen["n_bins"], selected_delta=chosen["delta"], how=how, all_passed_rule=bool(passing),
                              selected_at=summary["timestamp"], calibration_seeds=c["calibration"]["seeds"], from_summary=os.path.relpath(os.path.join(RES, "calibration", "summary.json"), ROOT))
        json.dump(sel, open(SELECTED, "w"), indent=2); print(f"  FROZEN in {os.path.relpath(SELECTED, ROOT)}")


# ------------------------------------------------------------------------------------------ C2
def confirm(a):
    cj = json.load(open(CAMPAIGN)); c, rules = cj["gateway"], cj["confirmation_verdict_rules_frozen"]
    sel = json.load(open(SELECTED))["gateway"]; nb = int(sel["selected_n_bins"])
    h_star = float(c["kernel_baseline"]["h_read_star"])
    raw = os.path.join(RES, "confirmation", "raw")
    K, kmeta = load_batch(os.path.join(raw, "kernel.npz")); H, hmeta = load_batch(os.path.join(raw, f"hist_nbins{nb}.npz"))
    K, H = by_arm(K), by_arm(H)
    keys = sorted(set(K["abf"]) & set(K["fr_uniform"]) & set(H["abf"]) & set(H["fr_uniform"]))
    n_exp = int(c["confirmation"]["n_seeds"]) * len(c["inits"])
    print(f"\nGateway C2 confirmation: {len(keys)} (init, seed) rows (expected {n_exp}); histogram n_bins={nb} (Delta {3.6 / nb:.3f}); kernel scored at h_read* {h_star}")
    assert all(H["abf"][k]["abf_n_bins"] == nb for k in keys)
    arms = {"kernel_abf": K["abf"], "kernel_fr_uniform": K["fr_uniform"], "hist_abf": H["abf"], "hist_fr_uniform": H["fr_uniform"]}
    t = np.asarray(K["abf"][keys[0]]["t"], dtype=float)
    ser, final_prof = {}, {}
    for m in arms:
        ser[m], final_prof[m] = {}, {}
        for k in keys:
            r = arms[m][k]
            if m.startswith("kernel"):
                eF, eFp, Fp_last = kernel_readout(r, h_star)
                eF_leg, eFp_leg = np.asarray(r["l2_f_t"], float), np.asarray(r["l2_fp_t"], float)
                ser[m][k] = dict(eF=eF, eFp=eFp, eF_legacy=eF_leg, eFp_legacy=eFp_leg); final_prof[m][k] = Fp_last
            else:
                ser[m][k] = dict(eF=np.asarray(r["l2_f_t"], float), eFp=np.asarray(r["l2_fp_t"], float), eFp_nodes=np.asarray(r["l2_fp_nodes_t"], float))
    # self-check: the legacy read-out recomputed offline equals the engine's own l2_f (accepted machinery)
    dev = max(abs(float(kernel_readout(K["abf"][k], 0.07)[0][-1]) - float(K["abf"][k]["final_l2_f"])) for k in keys[:4])
    assert dev < 1e-9, dev
    metrics = {}
    for m in arms:
        d = dict(int_l2_f=[float(np.trapezoid(ser[m][k]["eF"], t)) for k in keys], final_l2_f=[float(ser[m][k]["eF"][-1]) for k in keys],
                 final_l2_fp=[float(ser[m][k]["eFp"][-1]) for k in keys], repl_fraction=[float(arms[m][k]["repl_fraction"]) for k in keys],
                 min_ess_frac=[float(arms[m][k]["min_ess_frac"]) for k in keys], max_wmax=[float(arms[m][k]["max_wmax"]) for k in keys],
                 max_abs_bias_force=[float(arms[m][k]["max_abs_bias_force"]) for k in keys])
        if m.startswith("kernel"):
            d["int_l2_f_legacy"] = [float(np.trapezoid(ser[m][k]["eF_legacy"], t)) for k in keys]; d["final_l2_f_legacy"] = [float(ser[m][k]["eF_legacy"][-1]) for k in keys]
            d["final_l2_fp_legacy"] = [float(ser[m][k]["eFp_legacy"][-1]) for k in keys]
        else:
            d["final_l2_fp_nodes"] = [float(ser[m][k]["eFp_nodes"][-1]) for k in keys]
            d["min_count_window"] = [float(arms[m][k]["min_count_window"]) for k in keys]; d["frac_untrusted_window"] = [float(arms[m][k]["frac_untrusted_window"]) for k in keys]
        metrics[m] = d
    summ = {m: {k: med_iqr(v) for k, v in d.items()} for m, d in metrics.items()}
    effect = {}
    for i, est in enumerate(("kernel", "hist")):
        effect[est] = {k: paired(metrics[f"{est}_fr_uniform"][k], metrics[f"{est}_abf"][k], 10 * i + j) for j, k in enumerate(("int_l2_f", "final_l2_f", "final_l2_fp"))}
    effect["kernel_legacy"] = {k: paired(metrics["kernel_fr_uniform"][k + "_legacy"], metrics["kernel_abf"][k + "_legacy"], 50 + j) for j, k in enumerate(("int_l2_f", "final_l2_f", "final_l2_fp"))}
    cross = {arm: {k: paired(metrics[f"hist_{arm}"][k], metrics[f"kernel_{arm}"][k], 100 + 10 * j + jj) for jj, k in enumerate(("int_l2_f", "final_l2_f", "final_l2_fp"))} for j, arm in enumerate(("abf", "fr_uniform"))}
    # per-init breakdown (descriptive)
    per_init = {}
    for init in c["inits"]:
        idx = [i for i, k in enumerate(keys) if k[0] == init]
        per_init[init] = {est: paired(np.asarray(metrics[f"{est}_fr_uniform"]["int_l2_f"])[idx], np.asarray(metrics[f"{est}_abf"]["int_l2_f"])[idx], 400) for est in ("kernel", "hist")}
    ek, eh = effect["kernel"], effect["hist"]
    same_sign = all(np.sign(ek[k]["median"]) == np.sign(eh[k]["median"]) for k in ("int_l2_f", "final_l2_f"))
    ci_ok = (eh["int_l2_f"]["ci95"][1] < 0) if ek["int_l2_f"]["ci95"][1] < 0 else True
    overlap = lambda a_, b_: not (a_[1] < b_[0] or b_[1] < a_[0])
    mag_int = abs(eh["int_l2_f"]["median"] - ek["int_l2_f"]["median"]) <= 10.0 or overlap(eh["int_l2_f"]["ci95"], ek["int_l2_f"]["ci95"])
    mag_fin = abs(eh["final_l2_f"]["median"] - ek["final_l2_f"]["median"]) <= 15.0 or overlap(eh["final_l2_f"]["ci95"], ek["final_l2_f"]["ci95"])
    replicates = bool(same_sign and ci_ok and mag_int and mag_fin)
    acc_ok = bool(cross["abf"]["final_l2_f"]["median"] <= 25.0 and cross["abf"]["int_l2_f"]["median"] <= 25.0)
    fl = c["safety_floors"]
    health = {est: dict(min_ess_frac=float(np.min(metrics[f"{est}_fr_uniform"]["min_ess_frac"])), max_wmax=float(np.max(metrics[f"{est}_fr_uniform"]["max_wmax"])),
                        ok=bool(np.min(metrics[f"{est}_fr_uniform"]["min_ess_frac"]) >= fl["min_ess_frac"] and np.max(metrics[f"{est}_fr_uniform"]["max_wmax"]) <= fl["wmax"]))
              for est in ("kernel", "hist")}
    print(f"  {'arm':>18} {'I_F':>16} {'e_F(T)':>20} {'e_Fp(T)':>16}   [median (IQR)]  (kernel: h_read* {h_star}; hist: own)")
    for m in arms:
        s = summ[m]; f = lambda k, p=4: f"{s[k]['median']:.{p}f} ({s[k]['iqr'][0]:.{p}f}-{s[k]['iqr'][1]:.{p}f})"
        print(f"  {m:>18} {f('int_l2_f', 3):>16} {f('final_l2_f', 5):>20} {f('final_l2_fp', 3):>16}")
    print(f"  kernel arms at the legacy 0.07 read-out (engine's own): ABF e_F(T) {summ['kernel_abf']['final_l2_f_legacy']['median']:.5f}, I_F {summ['kernel_abf']['int_l2_f_legacy']['median']:.3f}")
    print("  FR effect (fr_uniform vs abf, paired by row, %):")
    for est in ("kernel", "kernel_legacy", "hist"):
        for k in ("int_l2_f", "final_l2_f", "final_l2_fp"):
            e = effect[est][k]; print(f"    {est:>13} {k:>12}: {e['median']:+7.2f} [{e['ci95'][0]:+7.2f}, {e['ci95'][1]:+7.2f}]  wins {e['wins']}/{e['n']}")
    print("  per init, Delta I_F: " + "; ".join(f"{init}: kernel {per_init[init]['kernel']['median']:+.1f} % ({per_init[init]['kernel']['wins']}/{per_init[init]['kernel']['n']}), hist {per_init[init]['hist']['median']:+.1f} % ({per_init[init]['hist']['wins']}/{per_init[init]['hist']['n']})" for init in per_init))
    print("  histogram vs kernel (same arm, identical noise, %):")
    for arm in ("abf", "fr_uniform"):
        for k in ("int_l2_f", "final_l2_f", "final_l2_fp"):
            e = cross[arm][k]; print(f"    {arm:>10} {k:>12}: {e['median']:+7.2f} [{e['ci95'][0]:+7.2f}, {e['ci95'][1]:+7.2f}]  wins {e['wins']}/{e['n']}")
    for est in ("kernel", "hist"):
        h = health[est]; print(f"  safety {est:>6}: min windowed ESS/N {h['min_ess_frac']:.3f} (floor {fl['min_ess_frac']}), max lineage share {h['max_wmax']:.4f} (cap {fl['wmax']}) -> ok={h['ok']}; repl fraction {summ[f'{est}_fr_uniform']['repl_fraction']['median']:.4f}")
    print(f"  hist support: min window count {summ['hist_abf']['min_count_window']['median']:.0f} / {summ['hist_fr_uniform']['min_count_window']['median']:.0f}; untrusted {summ['hist_abf']['frac_untrusted_window']['median']:.3f}; max|bias| {summ['hist_abf']['max_abs_bias_force']['median']:.2f} (kernel {summ['kernel_abf']['max_abs_bias_force']['median']:.2f})")
    print(f"  efficiency: kernel batch {kmeta['wall_seconds']:.1f}s ({1e3 * kmeta['wall_seconds_per_step']:.3f} ms/step, R={kmeta['R']}), histogram {hmeta['wall_seconds']:.1f}s ({1e3 * hmeta['wall_seconds_per_step']:.3f} ms/step) -> {kmeta['wall_seconds'] / hmeta['wall_seconds']:.2f}x")
    print(f"  VERDICT: replicates={replicates} (same sign {same_sign}, CI rule {ci_ok}, magnitude I_F {mag_int}, e_F(T) {mag_fin}); absolute accuracy acceptable={acc_ok} "
          f"(hist ABF vs kernel ABF @h_read*: I_F {cross['abf']['int_l2_f']['median']:+.1f}%, e_F(T) {cross['abf']['final_l2_f']['median']:+.1f}%)")
    if len(keys) < n_exp:
        print(f"  NOTE: {len(keys)} of {n_exp} rows present -- not the full block")

    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, key, lab in zip(axes, ("eF", "eFp"), ("$e_F(t)$", "$e_{F'}(t)$")):
        for m in arms:
            est, meth = ("kernel" if m.startswith("kernel") else "histogram"), ("abf" if m.endswith("_abf") else "fr_uniform")
            S = np.stack([ser[m][k][key] for k in keys]); med, lo, hi = np.median(S, 0), np.percentile(S, 25, 0), np.percentile(S, 75, 0)
            ax.plot(t, med, color=COL[est], ls=LS[meth], label=f"{est} {'ABF' if meth == 'abf' else 'ABF+FR (uniform)'}"); ax.fill_between(t, lo, hi, color=COL[est], alpha=0.12, lw=0)
        ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_title(lab)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Entropic gateway confirmation ({len(keys)} rows); kernel at h_read*={h_star}, histogram Delta={3.6 / nb:.3f} (own); median + IQR", fontsize=9)
    f_conv = _save(fig, os.path.join(RES, "figures"), "gateway_convergence")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.asarray(K["abf"][keys[0]]["x_grid"], float); dx = float(x[1] - x[0]); mask = (x >= EVAL_LO) & (x <= EVAL_HI)
    F_ref, Fp_ref = np.asarray(K["abf"][keys[0]]["F_ref"]), np.asarray(K["abf"][keys[0]]["Fp_ref"])
    axes[0].plot(x, F_ref, "k", lw=1.6, label="reference"); axes[1].plot(x, Fp_ref, "k", lw=1.6)
    for m in arms:
        est, meth = ("kernel" if m.startswith("kernel") else "histogram"), ("abf" if m.endswith("_abf") else "fr_uniform")
        if est == "kernel":
            Fp = np.median(np.stack([final_prof[m][k] for k in keys]), 0)
            F = np.concatenate([[0.0], np.cumsum(0.5 * (Fp[1:] + Fp[:-1]) * dx)]); F -= F[mask].mean(); F += F_ref[mask].mean()
            axes[0].plot(x, F, color=COL[est], ls=LS[meth], lw=1, label=f"{est} {'ABF' if meth == 'abf' else 'ABF+FR'} (h_read*)"); axes[1].plot(x, Fp, color=COL[est], ls=LS[meth], lw=1)
        else:
            edges = np.asarray(arms[m][keys[0]]["hist_edges"]); Gm = np.median(np.stack([np.asarray(arms[m][k]["Fp_bins"]) for k in keys]), 0)
            Fm = np.median(np.stack([np.asarray(arms[m][k]["F_hat"]) for k in keys]), 0)
            axes[0].plot(x, Fm, color=COL[est], ls=LS[meth], lw=1, label=f"{est} {'ABF' if meth == 'abf' else 'ABF+FR'} (own)"); axes[1].stairs(Gm, edges, color=COL[est], ls=LS[meth], lw=1, baseline=None)
    for ax in axes:
        ax.axvspan(-1.8, -1.5, color="0.9"); ax.axvspan(1.5, 1.8, color="0.9"); ax.set_xlabel("x")
    axes[0].set_title("final $F$ (row-median profile)"); axes[1].set_title("final $F'$"); axes[0].legend(fontsize=7)
    f_prof = _save(fig, os.path.join(RES, "figures"), "gateway_final_profiles")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i, k in enumerate(("int_l2_f", "final_l2_f", "final_l2_fp")):
        for est in ("kernel", "hist"):
            e = effect[est][k]; xx = i + (-0.17 if est == "kernel" else 0.17)
            ax.errorbar([xx], [e["median"]], yerr=[[e["median"] - e["ci95"][0]], [e["ci95"][1] - e["median"]]], fmt="o", color=COL["kernel" if est == "kernel" else "histogram"], capsize=3, label=(est if i == 0 else None))
            ax.annotate(f"{e['wins']}/{e['n']}", (xx, e["ci95"][1]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["$I_F$", "$e_F(T)$", "$e_{F'}(T)$"]); ax.set_ylabel("FR effect vs ABF (%)"); ax.legend(fontsize=8)
    ax.set_title("Entropic gateway: FR effect by estimator", fontsize=9)
    f_eff = _save(fig, os.path.join(RES, "figures"), "gateway_effect_size")
    out = dict(system="gateway", rows=[list(k) for k in keys], n_expected=n_exp, selected_n_bins=nb, delta=3.6 / nb, h_read_star=h_star, arms=list(arms),
               metrics_summary=summ, fr_effect=effect, per_init_int_l2_f=per_init, hist_vs_kernel=cross, health=health,
               verdict=dict(replicates=replicates, same_sign=same_sign, ci_rule=ci_ok, magnitude_int=mag_int, magnitude_final=mag_fin, absolute_accuracy_acceptable=acc_ok, rules=rules),
               efficiency=dict(kernel_wall_s=kmeta["wall_seconds"], hist_wall_s=hmeta["wall_seconds"], kernel_ms_per_step=1e3 * kmeta["wall_seconds_per_step"], hist_ms_per_step=1e3 * hmeta["wall_seconds_per_step"],
                               speedup=kmeta["wall_seconds"] / hmeta["wall_seconds"], R=kmeta["R"], kernel_peak_gpu_mem_bytes=kmeta.get("peak_gpu_mem_bytes"), hist_peak_gpu_mem_bytes=hmeta.get("peak_gpu_mem_bytes")),
               figures=[os.path.relpath(p, ROOT) for p in (f_conv, f_prof, f_eff)], git_rev=kmeta.get("git_rev"), timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(out, open(os.path.join(RES, "confirmation", "summary.json"), "w"), indent=2, default=float)
    with open(os.path.join(RES, "confirmation", "comparison.csv"), "w") as fh:
        fh.write("init,seed," + ",".join(f"{m}_{k}" for m in arms for k in ("int_l2_f", "final_l2_f", "final_l2_fp")) + "\n")
        for i, k in enumerate(keys):
            fh.write(f"{k[0]},{k[1]}," + ",".join(f"{metrics[m][kk][i]:.6f}" for m in arms for kk in ("int_l2_f", "final_l2_f", "final_l2_fp")) + "\n")
    print(f"  wrote {os.path.relpath(RES, ROOT)}/confirmation/summary.json, comparison.csv; figures {[os.path.basename(p) for p in (f_conv, f_prof, f_eff)]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true"); ap.add_argument("--reselect", action="store_true")
    a = ap.parse_args()
    select(a) if a.select else confirm(a)


if __name__ == "__main__":
    main()
