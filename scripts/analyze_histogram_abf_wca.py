#!/usr/bin/env python
"""Analyzer for the histogram-ABF campaign, WCA dimer (docs/HISTOGRAM_ABF_REPLICATION.md).

  --select      B1: histogram-ABF-only calibration on the frozen rule (configs/histogram_abf/
                campaign.json): floor share <= 0.20 for e_F and e_F'; paired median change to the
                next finer n_bins <= 5 % (final e_F, I_F), <= 10 % (final e_F'); coarsest passing.
                Writes calibration/summary.json + figure, FREEZES configs/histogram_abf/selected_bins.json
                (wca entry).  Reads NO FR result.
  (default)     B2: four-arm confirmation.  Kernel arms scored by the accepted corrected convention
                (bank read-out h_read* = 0.0125, legacy 0.025 alongside; machinery imported from
                analyze_wca_bandwidth_audit so it is bit-for-bit the accepted one); histogram arms by
                their OWN estimator (e_F: exact integral of the bins at the nodes; e_F': function-space
                RMS of the P0 profile).  FR effect within each estimator (paired, median, bootstrap CI,
                wins), replication verdict, absolute accuracy, safety (Case IX floors), round trips,
                genealogy, support diagnostics, wall time, figures.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPTS, "..")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
from analyze_uniform_lta import BOOT_SEED, boot_median                      # noqa: E402
from analyze_uniform_wca import tau, PERSIST, FRACTIONS, ESS_FLOOR, WMAX_CAP  # noqa: E402
from analyze_wca_bandwidth_audit import readouts, Z_LO, Z_HI, LEGACY_KEY     # noqa: E402
import wca_abffr_core as core                                                # noqa: E402

CAMPAIGN = os.path.join(ROOT, "configs", "histogram_abf", "campaign.json")
SELECTED = os.path.join(ROOT, "configs", "histogram_abf", "selected_bins.json")
RES = os.path.join(ROOT, "results", "histogram_abf", "wca")
COL = {"kernel": "#1f77b4", "histogram": "#d62728"}
LS = {"abf": "-", "fr_uniform": "--"}


def load_runs(raw_dir, stage):
    runs = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, f"{stage}__*.npz"))):
        d = np.load(f, allow_pickle=True)
        r = {k: d[k] for k in d.files}
        for k, v in list(r.items()):
            if isinstance(v, np.ndarray) and v.ndim == 0:
                r[k] = v.item()
        runs.setdefault(str(r["name"]), {})[int(r["seed"])] = r
    return runs


def paired(x_arm, x_ref, seed_offset=0):
    d = 100.0 * (np.asarray(x_arm, dtype=float) - np.asarray(x_ref, dtype=float)) / np.asarray(x_ref, dtype=float)
    lo, hi = boot_median(d, BOOT_SEED + seed_offset)
    return dict(median=float(np.median(d)), ci95=[lo, hi], wins=int((d < 0).sum()), n=int(len(d)), per_seed=[float(v) for v in d])


def med_iqr(x):
    q1, q2, q3 = np.percentile(np.asarray(x, dtype=float), [25, 50, 75])
    return dict(median=float(q2), iqr=[float(q1), float(q3)])


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
    return plt


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    return os.path.join(out_dir, f"{name}.png")


# ------------------------------------------------------------------------------------------ B1
def select(a):
    cj = json.load(open(CAMPAIGN))
    c, rule = cj["wca"], cj["bin_selection_rule_frozen"]
    floors = {int(r["n_bins"]): r for r in json.load(open(os.path.join(RES, "floor", "p0_floor.json")))["rows"]}
    ladder = [int(n) for n in c["ladder_n_bins"]]
    runs = load_runs(os.path.join(RES, "calibration", "raw"), "calibration")
    names = {nb: f"hist{nb}_abf" for nb in ladder}
    assert all(names[nb] in runs for nb in ladder), f"missing arms: {[names[nb] for nb in ladder if names[nb] not in runs]}"
    assert all(str(runs[n][s]["method"]) == "abf" for n in runs for s in runs[n]), "calibration must be ABF only"
    seeds = sorted(set.intersection(*[set(runs[names[nb]]) for nb in ladder]))
    assert seeds, "no complete ladder for any seed"
    t = np.asarray(runs[names[ladder[0]]][seeds[0]]["times"], dtype=float)

    def vec(nb, key):
        return np.array([float(runs[names[nb]][s][key]) for s in seeds])
    table = []
    for i, nb in enumerate(ladder):
        eF, IF, eFp = vec(nb, "l2_f"), vec(nb, "integrated_l2_f"), vec(nb, "l2_fp")
        assert all(int(runs[names[nb]][s]["abf_n_bins"]) == nb and str(runs[names[nb]][s]["abf_estimator"]) == "histogram" for s in seeds)
        row = dict(n_bins=nb, delta=float(1.4 / nb), final_l2_f=med_iqr(eF), int_l2_f=med_iqr(IF), final_l2_fp=med_iqr(eFp),
                   final_l2_fp_nodes=med_iqr(vec(nb, "l2_fp_nodes")),
                   floor_l2_f=floors[nb]["floor_l2_f"], floor_l2_fp=floors[nb]["floor_l2_fp"],
                   share_f=floors[nb]["floor_l2_f"] / float(np.median(eF)), share_fp=floors[nb]["floor_l2_fp"] / float(np.median(eFp)),
                   min_count_window=med_iqr(vec(nb, "hist_min_count_window")), frac_untrusted_window=med_iqr(vec(nb, "hist_frac_untrusted_window")),
                   bias_absmax=med_iqr(vec(nb, "bias_absmax")), bias_clip_fraction=med_iqr(vec(nb, "bias_clip_fraction")),
                   round_trips=med_iqr(vec(nb, "n_round_trips")),
                   runtime_seconds=med_iqr(vec(nb, "runtime_seconds")), wall_seconds=med_iqr(vec(nb, "wall_seconds")))
        row["pass_floor"] = bool(row["share_f"] <= 0.20 and row["share_fp"] <= 0.20)
        if i + 1 < len(ladder):
            nf = ladder[i + 1]
            row["next_n_bins"] = nf
            row["to_finer"] = dict(final_l2_f=paired(vec(nf, "l2_f"), eF, 3 * i), int_l2_f=paired(vec(nf, "integrated_l2_f"), IF, 3 * i + 1),
                                   final_l2_fp=paired(vec(nf, "l2_fp"), eFp, 3 * i + 2))
            dF, dI, dP = (abs(row["to_finer"][k]["median"]) for k in ("final_l2_f", "int_l2_f", "final_l2_fp"))
            row["pass_plateau"] = bool(dF <= 5.0 and dI <= 5.0 and dP <= 10.0)
            row["plateau_violation"] = float(max(dF / 5.0, dI / 5.0, dP / 10.0))
        else:
            row["next_n_bins"], row["pass_plateau"], row["plateau_violation"] = None, None, None
        table.append(row)
    passing = [r for r in table if r["pass_floor"] and r["pass_plateau"]]
    if passing:
        chosen, how = passing[0], "coarsest n_bins passing (a) floor and (b) plateau"
    else:
        cands = [r for r in table if r["pass_floor"] and r["plateau_violation"] is not None] or [r for r in table if r["plateau_violation"] is not None]
        chosen = min(cands, key=lambda r: r["plateau_violation"])
        how = "NO n_bins passed the frozen rule; predeclared fallback: smallest normalised plateau violation among widths passing (a)"
    print(f"\nWCA B1 calibration (histogram ABF only, {len(seeds)} seeds {seeds[0]}-{seeds[-1]})")
    print(f"  {'n_bins':>6} {'Delta':>8} {'e_F(T)':>8} {'I_F':>7} {'e_Fp(T)':>8} {'floorF':>7} {'floorFp':>7} {'shareF':>6} {'shareFp':>7} "
          f"{'dF%':>6} {'dI%':>6} {'dFp%':>6} {'floor':>5} {'plat':>5} {'mincnt':>6} {'untr':>5} {'clip':>7} {'run s':>6}")
    for r in table:
        tf = r.get("to_finer")
        d = (f"{tf['final_l2_f']['median']:+6.1f} {tf['int_l2_f']['median']:+6.1f} {tf['final_l2_fp']['median']:+6.1f}" if tf else " " * 20)
        print(f"  {r['n_bins']:6d} {r['delta']:8.5f} {r['final_l2_f']['median']:8.4f} {r['int_l2_f']['median']:7.3f} {r['final_l2_fp']['median']:8.4f} "
              f"{r['floor_l2_f']:7.4f} {r['floor_l2_fp']:7.4f} {r['share_f']:6.3f} {r['share_fp']:7.3f} {d} {str(r['pass_floor']):>5} {str(r['pass_plateau']):>5} "
              f"{r['min_count_window']['median']:6.0f} {r['frac_untrusted_window']['median']:5.2f} {r['bias_clip_fraction']['median']:7.5f} {r['runtime_seconds']['median']:6.0f}")
    print(f"  SELECTED n_bins = {chosen['n_bins']} (Delta {chosen['delta']:.5f}): {how}")

    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    deltas = [r["delta"] for r in table]
    for ax, key, fkey, lab in zip(axes, ("final_l2_f", "int_l2_f", "final_l2_fp"), ("floor_l2_f", None, "floor_l2_fp"), ("final $e_F$", "$I_F$", "final $e_{F'}$")):
        med = [r[key]["median"] for r in table]; lo = [r[key]["iqr"][0] for r in table]; hi = [r[key]["iqr"][1] for r in table]
        ax.errorbar(deltas, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="o-", color=COL["histogram"], label="histogram ABF (own)")
        if fkey:
            ax.plot(deltas, [r[fkey] for r in table], "s:", color="0.3", label="deterministic P0 floor"); ax.set_yscale("log")
        ax.axvline(chosen["delta"], color="k", lw=0.8, alpha=0.5); ax.set_xscale("log"); ax.set_xlabel(r"bin width $\Delta$"); ax.set_title(lab)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"WCA dimer, histogram-ABF-only bin-width calibration ({len(seeds)} seeds); selected n_bins = {chosen['n_bins']}", fontsize=9)
    fig_path = _save(fig, os.path.join(RES, "figures"), "wca_calibration")
    summary = dict(system="wca", seeds=seeds, ladder_n_bins=ladder, rule=rule, table=table, selected_n_bins=chosen["n_bins"],
                   selected_delta=chosen["delta"], selection_how=how, all_passed_rule=bool(passing), figure=os.path.relpath(fig_path, ROOT),
                   timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(summary, open(os.path.join(RES, "calibration", "summary.json"), "w"), indent=2, default=float)
    with open(os.path.join(RES, "calibration", "table.csv"), "w") as fh:
        fh.write("n_bins,delta,final_l2_f_med,int_l2_f_med,final_l2_fp_med,floor_l2_f,floor_l2_fp,share_f,share_fp,dF_pct,dI_pct,dFp_pct,pass_floor,pass_plateau,runtime_s_med\n")
        for r in table:
            tf = r.get("to_finer") or {}
            g = lambda k: (f"{tf[k]['median']:.3f}" if tf else "")
            fh.write(f"{r['n_bins']},{r['delta']:.6f},{r['final_l2_f']['median']:.5f},{r['int_l2_f']['median']:.4f},{r['final_l2_fp']['median']:.5f},"
                     f"{r['floor_l2_f']:.5f},{r['floor_l2_fp']:.5f},{r['share_f']:.4f},{r['share_fp']:.4f},{g('final_l2_f')},{g('int_l2_f')},{g('final_l2_fp')},"
                     f"{r['pass_floor']},{r['pass_plateau']},{r['runtime_seconds']['median']:.1f}\n")
    sel = json.load(open(SELECTED)) if os.path.exists(SELECTED) else {}
    if "wca" in sel and not a.reselect:
        print(f"  selected_bins.json already carries wca = {sel['wca']['selected_n_bins']} (frozen); not rewritten")
    else:
        sel["wca"] = dict(selected_n_bins=chosen["n_bins"], selected_delta=chosen["delta"], how=how, all_passed_rule=bool(passing),
                          selected_at=summary["timestamp"], calibration_seeds=seeds,
                          from_summary=os.path.relpath(os.path.join(RES, "calibration", "summary.json"), ROOT))
        json.dump(sel, open(SELECTED, "w"), indent=2)
        print(f"  FROZEN in {os.path.relpath(SELECTED, ROOT)}")


# ------------------------------------------------------------------------------------------ B2
def confirm(a):
    cj = json.load(open(CAMPAIGN))
    c, rules = cj["wca"], cj["confirmation_verdict_rules_frozen"]
    sel = json.load(open(SELECTED))["wca"]
    nb = int(sel["selected_n_bins"])
    h_star = f"{c['kernel_baseline']['primary_readout_h_read_star']:g}"
    runs = load_runs(os.path.join(RES, "confirmation", "raw"), "confirmation")
    arms = ["kernel_abf", "kernel_fr_uniform", "hist_abf", "hist_fr_uniform"]
    assert all(m in runs for m in arms), f"missing arms {[m for m in arms if m not in runs]}"
    seeds = sorted(set.intersection(*[set(runs[m]) for m in arms]))
    n_exp = int(c["confirmation"]["n_seeds"])
    print(f"\nWCA B2 confirmation: {len(seeds)} seeds {seeds[0]}-{seeds[-1]} (expected {n_exp}); histogram n_bins={nb} (Delta {1.4 / nb:.5f})")
    labels = {str(runs[m][s]["reference_label"]) for m in arms for s in seeds}
    assert len(labels) == 1 and "v2" in next(iter(labels)), labels
    assert all(int(runs[m][s]["abf_n_bins"]) == nb and str(runs[m][s]["abf_estimator"]) == "histogram" for m in arms[2:] for s in seeds)
    assert all(str(runs[m][s]["abf_estimator"]) == "kernel" and float(runs[m][s]["abf_bandwidth_online"]) == 0.025 for m in arms[:2] for s in seeds)
    any_run = runs["kernel_abf"][seeds[0]]
    grid = np.asarray(any_run["grid"], dtype=float)
    ref_F, ref_mf = np.asarray(any_run["reference_free_energy"], dtype=float), np.asarray(any_run["reference_mean_force"], dtype=float)
    t = np.asarray(any_run["profile_times"], dtype=float)
    mask = (grid >= Z_LO) & (grid <= Z_HI)
    sigma = float(any_run.get("abf_smooth_sigma", 0.5))

    # kernel arms: accepted read-out ladder (h_read* primary); own F' error at h_read* from the bank profile
    ro = {m: {s: readouts(runs[m][s], grid, mask, ref_F, sigma) for s in seeds} for m in arms[:2]}
    dev = max(abs(float(ro[m][s][LEGACY_KEY][-1]) - float(runs[m][s]["l2_f"])) for m in ro for s in seeds)
    assert dev < 1e-5, dev
    ladder_keys = list(ro["kernel_abf"][seeds[0]].keys())

    def e_fp_readout(run, key):
        mf = np.asarray(run[f"readout_mean_force_t__h{key}"], dtype=float)
        return np.array([core.profile_l2_error_np(m_, ref_mf, grid, mask=mask) for m_ in mf])

    # e_F(t), e_F'(t) series per arm under its PRIMARY convention
    ser = {}
    for m in arms[:2]:
        ser[m] = {s: dict(eF=np.asarray(ro[m][s][h_star]), eFp=e_fp_readout(runs[m][s], h_star),
                          eF_legacy=np.asarray(ro[m][s][LEGACY_KEY]), eFp_legacy=np.asarray(runs[m][s]["l2_fp_t"], dtype=float)) for s in seeds}
    for m in arms[2:]:
        ser[m] = {s: dict(eF=np.asarray(runs[m][s]["l2_f_t"], dtype=float), eFp=np.asarray(runs[m][s]["l2_fp_t"], dtype=float),
                          eFp_nodes=np.asarray(runs[m][s]["l2_fp_nodes_t"], dtype=float)) for s in seeds}
    metrics = {}
    for m in arms:
        d = dict(int_l2_f=[float(np.trapezoid(ser[m][s]["eF"], t)) for s in seeds], final_l2_f=[float(ser[m][s]["eF"][-1]) for s in seeds],
                 final_l2_fp=[float(ser[m][s]["eFp"][-1]) for s in seeds],
                 round_trips=[float(runs[m][s]["n_round_trips"]) for s in seeds], crossings=[float(runs[m][s]["n_barrier_crossings"]) for s in seeds],
                 bias_absmax=[float(runs[m][s]["bias_absmax"]) for s in seeds], bias_clip_fraction=[float(runs[m][s]["bias_clip_fraction"]) for s in seeds],
                 runtime_seconds=[float(runs[m][s]["runtime_seconds"]) for s in seeds], wall_seconds=[float(runs[m][s]["wall_seconds"]) for s in seeds])
        if m.startswith("kernel"):
            d["int_l2_f_legacy"] = [float(np.trapezoid(ser[m][s]["eF_legacy"], t)) for s in seeds]
            d["final_l2_f_legacy"] = [float(ser[m][s]["eF_legacy"][-1]) for s in seeds]
            d["final_l2_fp_legacy"] = [float(ser[m][s]["eFp_legacy"][-1]) for s in seeds]
        else:
            d["final_l2_fp_nodes"] = [float(ser[m][s]["eFp_nodes"][-1]) for s in seeds]
            d["min_count_window"] = [float(runs[m][s]["hist_min_count_window"]) for s in seeds]
            d["frac_untrusted_window"] = [float(runs[m][s]["hist_frac_untrusted_window"]) for s in seeds]
        if m.endswith("fr_uniform"):
            d["repl_events"] = [float(runs[m][s]["total_replacement_events"]) for s in seeds]
            d["fr_event_fraction"] = [float(runs[m][s]["fr_event_fraction"]) for s in seeds]
            d["min_ess_window_frac"] = [float(runs[m][s]["min_ancestor_ess_window"]) / float(runs[m][s]["n_replicas"]) for s in seeds]
            d["min_ess_frac"] = [float(runs[m][s]["min_ancestor_ess"]) / float(runs[m][s]["n_replicas"]) for s in seeds]
            d["wmax"] = [float(runs[m][s]["max_ancestor_frac_over_time"]) for s in seeds]
        metrics[m] = d
    summ = {m: {k: med_iqr(v) for k, v in d.items()} for m, d in metrics.items()}

    effect = {}
    for i, est in enumerate(("kernel", "hist")):
        effect[est] = {k: paired(metrics[f"{est}_fr_uniform"][k], metrics[f"{est}_abf"][k], 10 * i + j) for j, k in enumerate(("int_l2_f", "final_l2_f", "final_l2_fp"))}
    effect["kernel_legacy"] = {k: paired(metrics["kernel_fr_uniform"][k + "_legacy"], metrics["kernel_abf"][k + "_legacy"], 50 + j) for j, k in enumerate(("int_l2_f", "final_l2_f", "final_l2_fp"))}
    # every kernel read-out (the accepted ladder) for the record
    ladder_eff = {}
    for k_i, lab in enumerate(ladder_keys):
        I = {m: np.array([np.trapezoid(ro[m][s][lab], t) for s in seeds]) for m in ro}
        fin = {m: np.array([ro[m][s][lab][-1] for s in seeds]) for m in ro}
        ladder_eff[lab] = dict(d_int=paired(I["kernel_fr_uniform"], I["kernel_abf"], 200 + 2 * k_i), d_fin=paired(fin["kernel_fr_uniform"], fin["kernel_abf"], 201 + 2 * k_i),
                               abf_eF_T_median=float(np.median(fin["kernel_abf"])))
    cross = {arm: {k: paired(metrics[f"hist_{arm}"][k], metrics[f"kernel_{arm}"][k], 100 + 10 * j + jj) for jj, k in enumerate(("int_l2_f", "final_l2_f", "final_l2_fp"))}
             for j, arm in enumerate(("abf", "fr_uniform"))}

    ek, eh = effect["kernel"], effect["hist"]
    same_sign = all(np.sign(ek[k]["median"]) == np.sign(eh[k]["median"]) for k in ("int_l2_f", "final_l2_f"))
    ci_ok = (eh["int_l2_f"]["ci95"][1] < 0) if ek["int_l2_f"]["ci95"][1] < 0 else True
    overlap = lambda a_, b_: not (a_[1] < b_[0] or b_[1] < a_[0])
    mag_int = abs(eh["int_l2_f"]["median"] - ek["int_l2_f"]["median"]) <= 10.0 or overlap(eh["int_l2_f"]["ci95"], ek["int_l2_f"]["ci95"])
    mag_fin = abs(eh["final_l2_f"]["median"] - ek["final_l2_f"]["median"]) <= 15.0 or overlap(eh["final_l2_f"]["ci95"], ek["final_l2_f"]["ci95"])
    replicates = bool(same_sign and ci_ok and mag_int and mag_fin)
    acc_ok = bool(cross["abf"]["final_l2_f"]["median"] <= 25.0 and cross["abf"]["int_l2_f"]["median"] <= 25.0)
    health = {est: dict(min_ess_window_frac=float(np.min(metrics[f"{est}_fr_uniform"]["min_ess_window_frac"])),
                        min_ess_frac=float(np.min(metrics[f"{est}_fr_uniform"]["min_ess_frac"])),
                        max_wmax=float(np.max(metrics[f"{est}_fr_uniform"]["wmax"])),
                        ok=bool(np.min(metrics[f"{est}_fr_uniform"]["min_ess_frac"]) >= ESS_FLOOR and np.max(metrics[f"{est}_fr_uniform"]["wmax"]) <= WMAX_CAP))
              for est in ("kernel", "hist")}
    # time-to-accuracy on the median curves (each estimator's own primary series)
    speed = {}
    for est in ("kernel", "hist"):
        ca = np.median([ser[f"{est}_abf"][s]["eF"] for s in seeds], axis=0); cf = np.median([ser[f"{est}_fr_uniform"][s]["eF"] for s in seeds], axis=0)
        e0 = float(ca[0]); eps_list = {f"e0/{int(1 / f)}": e0 * f for f in FRACTIONS}; eps_list["abf_final"] = float(ca[-1])
        speed[est] = {}
        for nm, eps in eps_list.items():
            ta, tu = tau(t, ca, eps, PERSIST), tau(t, cf, eps, PERSIST)
            speed[est][nm] = dict(eps=eps, tau_abf=ta, tau_fr=tu, speedup=(ta / tu if np.isfinite(ta) and np.isfinite(tu) and tu > 0 else None))

    print(f"  {'arm':>18} {'I_F':>16} {'e_F(T)':>18} {'e_Fp(T)':>16}  {'round trips':>11} {'sampler s':>9}   [median (IQR)]  (kernel: h_read*={h_star}; hist: own)")
    for m in arms:
        s = summ[m]
        f = lambda k, p=4: f"{s[k]['median']:.{p}f} ({s[k]['iqr'][0]:.{p - 1}f}-{s[k]['iqr'][1]:.{p - 1}f})"
        print(f"  {m:>18} {f('int_l2_f', 3):>16} {f('final_l2_f'):>18} {f('final_l2_fp', 3):>16}  {s['round_trips']['median']:11.0f} {s['runtime_seconds']['median']:9.0f}")
    print(f"  kernel arms at the legacy read-out {LEGACY_KEY}: ABF e_F(T) {summ['kernel_abf']['final_l2_f_legacy']['median']:.5f}, I_F {summ['kernel_abf']['int_l2_f_legacy']['median']:.3f}")
    print("  FR effect (fr_uniform vs abf, paired, %):")
    for est in ("kernel", "kernel_legacy", "hist"):
        for k in ("int_l2_f", "final_l2_f", "final_l2_fp"):
            e = effect[est][k]
            print(f"    {est:>13} {k:>12}: {e['median']:+7.2f} [{e['ci95'][0]:+7.2f}, {e['ci95'][1]:+7.2f}]  wins {e['wins']}/{e['n']}")
    print("  kernel read-out ladder (accepted instrument): " + "; ".join(f"{lab}: dI {ladder_eff[lab]['d_int']['median']:+.1f}%, dF(T) {ladder_eff[lab]['d_fin']['median']:+.1f}%" for lab in ladder_keys))
    print("  histogram vs kernel (same arm, shared init, %):")
    for arm in ("abf", "fr_uniform"):
        for k in ("int_l2_f", "final_l2_f", "final_l2_fp"):
            e = cross[arm][k]
            print(f"    {arm:>10} {k:>12}: {e['median']:+7.2f} [{e['ci95'][0]:+7.2f}, {e['ci95'][1]:+7.2f}]  wins {e['wins']}/{e['n']}")
    for est in ("kernel", "hist"):
        h = health[est]; fr = summ[f"{est}_fr_uniform"]
        print(f"  safety {est:>6}: min ESS/N {h['min_ess_frac']:.3f} (floor {ESS_FLOOR}), windowed {h['min_ess_window_frac']:.3f}, max wmax {h['max_wmax']:.4f} (cap {WMAX_CAP}) -> ok={h['ok']}; "
              f"repl events {fr['repl_events']['median']:.0f}, event fraction {fr['fr_event_fraction']['median']:.4f}")
        sp = speed[est]
        print(f"    time-to-accuracy: " + ", ".join(f"{nm} {('%.2fx' % v['speedup']) if v['speedup'] else 'censored'}" for nm, v in sp.items()))
    print(f"  hist support: min count in window {summ['hist_abf']['min_count_window']['median']:.0f} / {summ['hist_fr_uniform']['min_count_window']['median']:.0f}; untrusted fraction "
          f"{summ['hist_abf']['frac_untrusted_window']['median']:.3f} / {summ['hist_fr_uniform']['frac_untrusted_window']['median']:.3f}; "
          f"|bias|max {summ['hist_abf']['bias_absmax']['median']:.1f} (kernel {summ['kernel_abf']['bias_absmax']['median']:.1f}); clip fraction {summ['hist_abf']['bias_clip_fraction']['median']:.2e} (kernel {summ['kernel_abf']['bias_clip_fraction']['median']:.2e})")
    print(f"  wall (sampler s, median): kernel abf {summ['kernel_abf']['runtime_seconds']['median']:.0f} (with bank), hist abf {summ['hist_abf']['runtime_seconds']['median']:.0f}; "
          f"kernel fr {summ['kernel_fr_uniform']['runtime_seconds']['median']:.0f}, hist fr {summ['hist_fr_uniform']['runtime_seconds']['median']:.0f}")
    print(f"  VERDICT: replicates={replicates} (same sign {same_sign}, CI rule {ci_ok}, magnitude I_F {mag_int}, e_F(T) {mag_fin}); "
          f"absolute accuracy acceptable={acc_ok} (hist ABF vs kernel ABF @h_read*: I_F {cross['abf']['int_l2_f']['median']:+.1f}%, e_F(T) {cross['abf']['final_l2_f']['median']:+.1f}%)")
    if len(seeds) < n_exp:
        print(f"  NOTE: {len(seeds)} of {n_exp} seeds present -- not the full block")

    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, key, lab in zip(axes, ("eF", "eFp"), ("$e_F(t)$", "$e_{F'}(t)$")):
        for m in arms:
            est, meth = ("kernel" if m.startswith("kernel") else "histogram"), ("abf" if m.endswith("_abf") else "fr_uniform")
            S = np.stack([ser[m][s][key] for s in seeds]); med, lo, hi = np.median(S, 0), np.percentile(S, 25, 0), np.percentile(S, 75, 0)
            ax.plot(t, med, color=COL[est], ls=LS[meth], label=f"{est} {'ABF' if meth == 'abf' else 'ABF+FR (uniform)'}"); ax.fill_between(t, lo, hi, color=COL[est], alpha=0.12, lw=0)
        ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_title(lab)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"WCA dimer confirmation ({len(seeds)} seeds); kernel at h_read*={h_star}, histogram n_bins={nb} (own); median + IQR", fontsize=9)
    f_conv = _save(fig, os.path.join(RES, "figures"), "wca_convergence")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(grid, ref_F - ref_F[mask].mean(), "k", lw=1.6, label="TI reference"); axes[1].plot(grid, ref_mf, "k", lw=1.6, label="TI reference")
    for m in arms:
        est, meth = ("kernel" if m.startswith("kernel") else "histogram"), ("abf" if m.endswith("_abf") else "fr_uniform")
        if est == "kernel":
            mf = np.median(np.stack([np.asarray(runs[m][s][f"readout_mean_force_t__h{h_star}"])[-1] for s in seeds]), 0)
            F = np.concatenate([[0.0], np.cumsum(0.5 * (mf[1:] + mf[:-1]) * np.diff(grid))])
            axes[0].plot(grid, F - F[mask].mean(), color=COL[est], ls=LS[meth], lw=1, label=f"{est} {'ABF' if meth == 'abf' else 'ABF+FR'} (h_read*)"); axes[1].plot(grid, mf, color=COL[est], ls=LS[meth], lw=1)
        else:
            edges = np.asarray(runs[m][seeds[0]]["hist_edges_abf"]); Gm = np.median(np.stack([np.asarray(runs[m][s]["final_hist_mf_bins"]) for s in seeds]), 0)
            F = core.histogram_pmf_nodes_np(Gm, edges, grid)
            axes[0].plot(grid, F - F[mask].mean(), color=COL[est], ls=LS[meth], lw=1, label=f"{est} {'ABF' if meth == 'abf' else 'ABF+FR'} (own bins)"); axes[1].stairs(Gm, edges, color=COL[est], ls=LS[meth], lw=1, baseline=None)
    for ax in axes:
        ax.axvspan(-0.2, -0.1, color="0.9"); ax.axvspan(1.1, 1.2, color="0.9"); ax.set_xlabel("z")
    axes[0].set_title("final $F$ (seed-median profile)"); axes[1].set_title("final $F'$"); axes[1].set_ylim(-30, 15); axes[0].legend(fontsize=7)
    f_prof = _save(fig, os.path.join(RES, "figures"), "wca_final_profiles")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i, k in enumerate(("int_l2_f", "final_l2_f", "final_l2_fp")):
        for est in ("kernel", "hist"):
            e = effect[est][k]; xx = i + (-0.17 if est == "kernel" else 0.17)
            ax.errorbar([xx], [e["median"]], yerr=[[e["median"] - e["ci95"][0]], [e["ci95"][1] - e["median"]]], fmt="o", color=COL["kernel" if est == "kernel" else "histogram"], capsize=3, label=(est if i == 0 else None))
            ax.annotate(f"{e['wins']}/{e['n']}", (xx, e["ci95"][1]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["$I_F$", "$e_F(T)$", "$e_{F'}(T)$"]); ax.set_ylabel("FR effect vs ABF (%)"); ax.legend(fontsize=8)
    ax.set_title("WCA dimer: FR effect by estimator", fontsize=9)
    f_eff = _save(fig, os.path.join(RES, "figures"), "wca_effect_size")

    out = dict(system="wca", seeds=seeds, n_expected=n_exp, selected_n_bins=nb, delta=1.4 / nb, h_read_star=h_star, arms=arms, metrics_summary=summ,
               fr_effect=effect, kernel_readout_ladder=ladder_eff, hist_vs_kernel=cross, health=health, time_to_accuracy=speed,
               verdict=dict(replicates=replicates, same_sign=same_sign, ci_rule=ci_ok, magnitude_int=mag_int, magnitude_final=mag_fin, absolute_accuracy_acceptable=acc_ok, rules=rules),
               efficiency={m: dict(runtime_s=summ[m]["runtime_seconds"], wall_s=summ[m]["wall_seconds"]) for m in arms},
               figures=[os.path.relpath(p, ROOT) for p in (f_conv, f_prof, f_eff)], reference_label=next(iter(labels)),
               git_rev=str(any_run.get("git_rev", "")), timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(out, open(os.path.join(RES, "confirmation", "summary.json"), "w"), indent=2, default=float)
    with open(os.path.join(RES, "confirmation", "comparison.csv"), "w") as fh:
        fh.write("seed," + ",".join(f"{m}_{k}" for m in arms for k in ("int_l2_f", "final_l2_f", "final_l2_fp")) + ",kernel_fr_min_ess_frac,kernel_fr_wmax,hist_fr_min_ess_frac,hist_fr_wmax\n")
        for i, s in enumerate(seeds):
            fh.write(f"{s}," + ",".join(f"{metrics[m][k][i]:.6f}" for m in arms for k in ("int_l2_f", "final_l2_f", "final_l2_fp")) +
                     f",{metrics['kernel_fr_uniform']['min_ess_frac'][i]:.4f},{metrics['kernel_fr_uniform']['wmax'][i]:.4f},{metrics['hist_fr_uniform']['min_ess_frac'][i]:.4f},{metrics['hist_fr_uniform']['wmax'][i]:.4f}\n")
    print(f"  wrote {os.path.relpath(RES, ROOT)}/confirmation/summary.json, comparison.csv; figures {[os.path.basename(p) for p in (f_conv, f_prof, f_eff)]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--reselect", action="store_true")
    a = ap.parse_args()
    select(a) if a.select else confirm(a)


if __name__ == "__main__":
    main()
