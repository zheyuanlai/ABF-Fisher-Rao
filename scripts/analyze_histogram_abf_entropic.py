#!/usr/bin/env python
"""Analyzer for the histogram-ABF campaign, entropic bottleneck (docs/HISTOGRAM_ABF_REPLICATION.md).

Two modes, in the order the campaign runs them:

  --select      A1: ABF-only bin-width calibration.  Reads the deterministic P0 floors and the
                calibration batches, applies the FROZEN rule of configs/histogram_abf/campaign.json
                (floor share <= 0.20 for e_F and e_F'; refinement plateau: paired median change to
                the next finer width <= 5 % for final e_F and I_F, <= 10 % for final e_F'; coarsest
                passing width), writes calibration/summary.json + the calibration figure, and FREEZES
                the choice in configs/histogram_abf/selected_bins.json.  Refuses to re-freeze unless
                --reselect.  Reads NO FR result.
  (default)     A2: four-arm confirmation.  Own-estimator metrics per arm (kernel arms: the accepted
                h = 0.07 profile; histogram arms: their own bins), the FR effect within each estimator
                (paired by seed, median, bootstrap 95 % CI, wins), the replication verdict, absolute
                accuracy, safety, efficiency, and the figures.

Statistics follow the repository convention (analyze_uniform_lta.boot_median: 10 000 resamples,
seed 20260829).
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
from analyze_uniform_lta import BOOT_SEED, boot_median   # noqa: E402  (10k resamples, seed 20260829)

CAMPAIGN = os.path.join(ROOT, "configs", "histogram_abf", "campaign.json")
SELECTED = os.path.join(ROOT, "configs", "histogram_abf", "selected_bins.json")
RES = os.path.join(ROOT, "results", "histogram_abf", "entropic_bottleneck")

COL = {"kernel": "#1f77b4", "histogram": "#d62728"}
LS = {"abf": "-", "fr_uniform": "--"}


# ------------------------------------------------------------------------------------------ io
def load_batch(path):
    with np.load(path, allow_pickle=True) as d:
        meta = json.loads(str(d["meta_json"]))
        n = int(d["n_rows"])
        rows = []
        for i in range(n):
            pre = f"r{i}/"
            r = {k[len(pre):]: d[k] for k in d.files if k.startswith(pre)}
            r["config"] = json.loads(str(r.pop("config_json")))
            for k, v in list(r.items()):
                if isinstance(v, np.ndarray) and v.ndim == 0:
                    r[k] = v.item()
            rows.append(r)
    return rows, meta


def by_method(rows):
    out = {}
    for r in rows:
        out.setdefault(str(r["method"]), {})[int(r["seed"])] = r
    return out


def paired(x_arm, x_ref, seed_offset=0):
    d = 100.0 * (np.asarray(x_arm) - np.asarray(x_ref)) / np.asarray(x_ref)
    lo, hi = boot_median(d, BOOT_SEED + seed_offset)
    return dict(median=float(np.median(d)), ci95=[lo, hi], wins=int((d < 0).sum()), n=int(len(d)),
                per_seed=[float(v) for v in d])


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


# ------------------------------------------------------------------------------------------ A1
def select(a):
    c = json.load(open(CAMPAIGN))["entropic_bottleneck"]
    rule = json.load(open(CAMPAIGN))["bin_selection_rule_frozen"]
    floors = {int(r["n_bins"]): r for r in json.load(open(os.path.join(RES, "floor", "p0_floor.json")))["rows"]}
    ladder = [int(n) for n in c["ladder_n_bins"]]
    cal = {}
    for nb in ladder:
        rows, meta = load_batch(os.path.join(RES, "calibration", "raw", f"hist_nbins{nb}.npz"))
        bm = by_method(rows)
        assert set(bm) == {"abf"}, f"calibration batch {nb} is not ABF-only: {set(bm)}"
        seeds = sorted(bm["abf"])
        assert seeds == sorted(c["calibration"]["seeds"]), (seeds, c["calibration"]["seeds"])
        cal[nb] = dict(rows=bm["abf"], meta=meta, seeds=seeds)
    kernel_ctx = None
    kp = os.path.join(RES, "calibration", "raw", "kernel.npz")
    if os.path.exists(kp):
        krows, kmeta = load_batch(kp)
        kernel_ctx = dict(rows=by_method(krows)["abf"], meta=kmeta)
    seeds = cal[ladder[0]]["seeds"]

    def vec(nb, key):
        return np.array([float(cal[nb]["rows"][s][key]) for s in seeds])

    table = []
    for i, nb in enumerate(ladder):
        eF, IF, eFp = vec(nb, "final_l2_f"), vec(nb, "int_l2_f"), vec(nb, "final_l2_fp")
        row = dict(n_bins=nb, delta=float(3.6 / nb), final_l2_f=med_iqr(eF), int_l2_f=med_iqr(IF), final_l2_fp=med_iqr(eFp),
                   final_l2_fp_nodes=med_iqr(vec(nb, "final_l2_fp_nodes")),
                   floor_l2_f=floors[nb]["floor_l2_f"], floor_l2_fp=floors[nb]["floor_l2_fp"],
                   share_f=floors[nb]["floor_l2_f"] / float(np.median(eF)), share_fp=floors[nb]["floor_l2_fp"] / float(np.median(eFp)),
                   min_count_window=med_iqr(vec(nb, "min_count_window")), frac_untrusted_window=med_iqr(vec(nb, "frac_untrusted_window")),
                   max_abs_bias_force=med_iqr(vec(nb, "max_abs_bias_force")), final_ess=med_iqr(vec(nb, "final_ess")),
                   wall_seconds=cal[nb]["meta"]["wall_seconds"], ms_per_step=1e3 * cal[nb]["meta"]["wall_seconds_per_step"],
                   peak_gpu_mem_bytes=cal[nb]["meta"].get("peak_gpu_mem_bytes"))
        row["pass_floor"] = bool(row["share_f"] <= 0.20 and row["share_fp"] <= 0.20)
        if i + 1 < len(ladder):
            nf = ladder[i + 1]
            row["next_n_bins"] = nf
            row["to_finer"] = dict(final_l2_f=paired(vec(nf, "final_l2_f"), eF, 3 * i),
                                   int_l2_f=paired(vec(nf, "int_l2_f"), IF, 3 * i + 1),
                                   final_l2_fp=paired(vec(nf, "final_l2_fp"), eFp, 3 * i + 2))
            dF, dI, dP = (abs(row["to_finer"][k]["median"]) for k in ("final_l2_f", "int_l2_f", "final_l2_fp"))
            row["pass_plateau"] = bool(dF <= 5.0 and dI <= 5.0 and dP <= 10.0)
            row["plateau_violation"] = float(max(dF / 5.0, dI / 5.0, dP / 10.0))
        else:
            row["next_n_bins"] = None
            row["pass_plateau"] = None
            row["plateau_violation"] = None
        table.append(row)

    passing = [r for r in table if r["pass_floor"] and r["pass_plateau"]]
    if passing:
        chosen, how = passing[0], "coarsest width passing (a) floor and (b) plateau"
    else:
        cands = [r for r in table if r["pass_floor"] and r["plateau_violation"] is not None]
        if not cands:
            cands = [r for r in table if r["plateau_violation"] is not None]
        chosen = min(cands, key=lambda r: r["plateau_violation"])
        how = "NO width passed the frozen rule; predeclared fallback: smallest normalised plateau violation among widths passing (a)"
    print(f"\nEntropic bottleneck A1 calibration (ABF only, {len(seeds)} seeds {seeds[0]}-{seeds[-1]})")
    print(f"  {'n_bins':>6} {'Delta':>6} {'e_F(T)':>8} {'I_F':>7} {'e_Fp(T)':>8} {'floorF':>7} {'floorFp':>7} {'shareF':>6} {'shareFp':>7} "
          f"{'dF%':>6} {'dI%':>6} {'dFp%':>6} {'floor':>5} {'plat':>5} {'ms/step':>7}")
    for r in table:
        tf = r.get("to_finer")
        d = (f"{tf['final_l2_f']['median']:+6.1f} {tf['int_l2_f']['median']:+6.1f} {tf['final_l2_fp']['median']:+6.1f}" if tf else " " * 20)
        print(f"  {r['n_bins']:6d} {r['delta']:6.3f} {r['final_l2_f']['median']:8.4f} {r['int_l2_f']['median']:7.3f} {r['final_l2_fp']['median']:8.4f} "
              f"{r['floor_l2_f']:7.4f} {r['floor_l2_fp']:7.4f} {r['share_f']:6.3f} {r['share_fp']:7.3f} {d} "
              f"{str(r['pass_floor']):>5} {str(r['pass_plateau']):>5} {r['ms_per_step']:7.3f}")
    if kernel_ctx:
        kr = kernel_ctx["rows"]
        print(f"  kernel h=0.07 (context only): e_F(T) {np.median([kr[s]['final_l2_f'] for s in seeds]):.4f}  "
              f"I_F {np.median([kr[s]['int_l2_f'] for s in seeds]):.3f}  e_F'(T) {np.median([kr[s]['final_l2_fp'] for s in seeds]):.4f}  "
              f"{1e3 * kernel_ctx['meta']['wall_seconds_per_step']:.3f} ms/step")
    print(f"  SELECTED n_bins = {chosen['n_bins']} (Delta {chosen['delta']:.3f}): {how}")

    # calibration figure
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    deltas = [r["delta"] for r in table]
    for ax, key, fkey, lab in zip(axes, ("final_l2_f", "int_l2_f", "final_l2_fp"), ("floor_l2_f", None, "floor_l2_fp"),
                                  ("final $e_F$", "$I_F$", "final $e_{F'}$")):
        med = [r[key]["median"] for r in table]
        lo = [r[key]["iqr"][0] for r in table]; hi = [r[key]["iqr"][1] for r in table]
        ax.errorbar(deltas, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="o-", color=COL["histogram"], label="histogram ABF (own)")
        if fkey:
            ax.plot(deltas, [r[fkey] for r in table], "s:", color="0.3", label="deterministic P0 floor")
        if kernel_ctx:
            kv = np.median([kernel_ctx["rows"][s][key] for s in seeds])
            ax.axhline(kv, color=COL["kernel"], ls="-.", label="kernel ABF h=0.07 (context)")
        ax.axvline(chosen["delta"], color="k", lw=0.8, alpha=0.5)
        ax.set_xscale("log"); ax.set_xlabel(r"bin width $\Delta$"); ax.set_title(lab)
        if fkey:
            ax.set_yscale("log")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Entropic bottleneck, ABF-only bin-width calibration ({len(seeds)} seeds); selected Delta = {chosen['delta']:.3f}", fontsize=9)
    fig_path = _save(fig, os.path.join(RES, "figures"), "eb_calibration")

    summary = dict(system="entropic_bottleneck", seeds=seeds, ladder_n_bins=ladder, rule=rule, table=table,
                   selected_n_bins=chosen["n_bins"], selected_delta=chosen["delta"], selection_how=how,
                   all_passed_rule=bool(passing), kernel_context=(None if not kernel_ctx else dict(
                       final_l2_f=med_iqr([kernel_ctx["rows"][s]["final_l2_f"] for s in seeds]),
                       int_l2_f=med_iqr([kernel_ctx["rows"][s]["int_l2_f"] for s in seeds]),
                       final_l2_fp=med_iqr([kernel_ctx["rows"][s]["final_l2_fp"] for s in seeds]),
                       ms_per_step=1e3 * kernel_ctx["meta"]["wall_seconds_per_step"], wall_seconds=kernel_ctx["meta"]["wall_seconds"])),
                   figure=os.path.relpath(fig_path, ROOT), timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    os.makedirs(os.path.join(RES, "calibration"), exist_ok=True)
    json.dump(summary, open(os.path.join(RES, "calibration", "summary.json"), "w"), indent=2, default=float)
    with open(os.path.join(RES, "calibration", "table.csv"), "w") as fh:
        fh.write("n_bins,delta,final_l2_f_med,int_l2_f_med,final_l2_fp_med,floor_l2_f,floor_l2_fp,share_f,share_fp,dF_pct,dI_pct,dFp_pct,pass_floor,pass_plateau,ms_per_step\n")
        for r in table:
            tf = r.get("to_finer") or {}
            g = lambda k: (f"{tf[k]['median']:.3f}" if tf else "")
            fh.write(f"{r['n_bins']},{r['delta']:.4f},{r['final_l2_f']['median']:.5f},{r['int_l2_f']['median']:.4f},{r['final_l2_fp']['median']:.5f},"
                     f"{r['floor_l2_f']:.5f},{r['floor_l2_fp']:.5f},{r['share_f']:.4f},{r['share_fp']:.4f},{g('final_l2_f')},{g('int_l2_f')},{g('final_l2_fp')},"
                     f"{r['pass_floor']},{r['pass_plateau']},{r['ms_per_step']:.4f}\n")
    # freeze
    sel = json.load(open(SELECTED)) if os.path.exists(SELECTED) else {}
    if "entropic_bottleneck" in sel and not a.reselect:
        print(f"  selected_bins.json already carries entropic_bottleneck = {sel['entropic_bottleneck']['selected_n_bins']} (frozen); not rewritten")
    else:
        sel["entropic_bottleneck"] = dict(selected_n_bins=chosen["n_bins"], selected_delta=chosen["delta"], how=how,
                                          all_passed_rule=bool(passing), selected_at=summary["timestamp"],
                                          calibration_seeds=seeds, from_summary=os.path.relpath(os.path.join(RES, "calibration", "summary.json"), ROOT))
        json.dump(sel, open(SELECTED, "w"), indent=2)
        print(f"  FROZEN in {os.path.relpath(SELECTED, ROOT)}")


# ------------------------------------------------------------------------------------------ A2
def confirm(a):
    c = json.load(open(CAMPAIGN))
    rules = c["confirmation_verdict_rules_frozen"]
    sel = json.load(open(SELECTED))["entropic_bottleneck"]
    nb = int(sel["selected_n_bins"])
    raw = os.path.join(RES, "confirmation", "raw")
    K, kmeta = load_batch(os.path.join(raw, "kernel.npz"))
    H, hmeta = load_batch(os.path.join(raw, f"hist_nbins{nb}.npz"))
    K, H = by_method(K), by_method(H)
    seeds = sorted(set(K["abf"]) & set(K["fr_uniform"]) & set(H["abf"]) & set(H["fr_uniform"]))
    n_exp = int(c["entropic_bottleneck"]["confirmation"]["n_seeds"])
    print(f"\nEntropic bottleneck A2 confirmation: {len(seeds)} seeds {seeds[0]}-{seeds[-1]} (expected {n_exp}); histogram n_bins={nb} (Delta {3.6 / nb:.3f})")
    assert all(H["abf"][s]["abf_n_bins"] == nb for s in seeds)
    arms = {"kernel_abf": K["abf"], "kernel_fr_uniform": K["fr_uniform"], "hist_abf": H["abf"], "hist_fr_uniform": H["fr_uniform"]}
    t = np.asarray(K["abf"][seeds[0]]["t"], dtype=float)

    def series(arm, key):
        return np.stack([np.asarray(arms[arm][s][key], dtype=float) for s in seeds])
    metrics = {}
    for arm in arms:
        eF_t, eFp_t = series(arm, "l2_f_t"), series(arm, "l2_fp_t")
        m = dict(int_l2_f=[float(np.trapezoid(e, t)) for e in eF_t], final_l2_f=[float(e[-1]) for e in eF_t],
                 final_l2_fp=[float(e[-1]) for e in eFp_t], int_l2_fp=[float(np.trapezoid(e, t)) for e in eFp_t])
        # engine-recorded int_l2_f uses the same trapezoid; cross-check
        assert np.allclose(m["int_l2_f"], [float(arms[arm][s]["int_l2_f"]) for s in seeds], rtol=1e-8)
        if arm.startswith("hist"):
            m["final_l2_fp_nodes"] = [float(arms[arm][s]["final_l2_fp_nodes"]) for s in seeds]
        # post-hoc DIAGNOSTIC (not an endpoint): the same final F error restricted to the interior
        # [-1.3, 1.3], i.e. without the outer 0.2 of the window where the kernel ABF stalls at the
        # steep walls (seen in the ABF-only calibration before any FR result was read)
        xg = np.asarray(arms[arm][seeds[0]]["x_grid"], dtype=float)
        mi = (xg >= -1.3) & (xg <= 1.3)
        m["final_l2_f_interior"] = []
        for s in seeds:
            d = np.asarray(arms[arm][s]["F_hat"], dtype=float) - np.asarray(arms[arm][s]["F_ref"], dtype=float)
            d = d[mi] - d[mi].mean()
            m["final_l2_f_interior"].append(float(np.sqrt(np.mean(d * d))))
        m["repl_fraction"] = [float(arms[arm][s]["repl_fraction"]) for s in seeds]
        m["min_ess_window_frac"] = [float(np.min(arms[arm][s]["ess_t"]) / 256.0) for s in seeds]
        m["cond_abserr_mean"] = [float(np.nanmean(arms[arm][s]["cond_abs_err"])) for s in seeds]
        m["max_abs_bias_force"] = [float(arms[arm][s]["max_abs_bias_force"]) for s in seeds]
        if arm.startswith("hist"):
            m["min_count_window"] = [float(arms[arm][s]["min_count_window"]) for s in seeds]
            m["frac_untrusted_window"] = [float(arms[arm][s]["frac_untrusted_window"]) for s in seeds]
        metrics[arm] = m
    summ = {arm: {k: med_iqr(v) for k, v in m.items()} for arm, m in metrics.items()}

    # FR effect within each estimator
    effect = {}
    for i, est in enumerate(("kernel", "hist")):
        effect[est] = {k: paired(metrics[f"{est}_fr_uniform"][k], metrics[f"{est}_abf"][k], 10 * i + j)
                       for j, k in enumerate(("int_l2_f", "final_l2_f", "final_l2_fp"))}
    # cross-estimator absolute accuracy (ABF vs ABF, FR vs FR), paired on identical noise
    cross = {}
    for j, arm in enumerate(("abf", "fr_uniform")):
        cross[arm] = {k: paired(metrics[f"hist_{arm}"][k], metrics[f"kernel_{arm}"][k], 100 + 10 * j + jj)
                      for jj, k in enumerate(("int_l2_f", "final_l2_f", "final_l2_fp"))}

    # post-hoc diagnostic: the same contrasts on the interior-window final F error
    for i, est in enumerate(("kernel", "hist")):
        effect[est]["final_l2_f_interior"] = paired(metrics[f"{est}_fr_uniform"]["final_l2_f_interior"], metrics[f"{est}_abf"]["final_l2_f_interior"], 300 + i)
    for j, arm in enumerate(("abf", "fr_uniform")):
        cross[arm]["final_l2_f_interior"] = paired(metrics[f"hist_{arm}"]["final_l2_f_interior"], metrics[f"kernel_{arm}"]["final_l2_f_interior"], 310 + j)

    ek, eh = effect["kernel"], effect["hist"]
    same_sign = all(np.sign(ek[k]["median"]) == np.sign(eh[k]["median"]) for k in ("int_l2_f", "final_l2_f"))
    ci_ok = (eh["int_l2_f"]["ci95"][1] < 0) if ek["int_l2_f"]["ci95"][1] < 0 else True
    overlap = lambda a_, b_: not (a_[1] < b_[0] or b_[1] < a_[0])
    mag_int = abs(eh["int_l2_f"]["median"] - ek["int_l2_f"]["median"]) <= 10.0 or overlap(eh["int_l2_f"]["ci95"], ek["int_l2_f"]["ci95"])
    mag_fin = abs(eh["final_l2_f"]["median"] - ek["final_l2_f"]["median"]) <= 15.0 or overlap(eh["final_l2_f"]["ci95"], ek["final_l2_f"]["ci95"])
    replicates = bool(same_sign and ci_ok and mag_int and mag_fin)
    acc_ok = bool(cross["abf"]["final_l2_f"]["median"] <= 25.0 and cross["abf"]["int_l2_f"]["median"] <= 25.0)
    health = {est: dict(min_ess_frac=float(np.min(metrics[f"{est}_fr_uniform"]["min_ess_window_frac"])),
                        repl_fraction=med_iqr(metrics[f"{est}_fr_uniform"]["repl_fraction"])) for est in ("kernel", "hist")}

    print(f"  {'arm':>18} {'I_F':>14} {'e_F(T)':>16} {'e_Fp(T)':>16}   [median (IQR)]")
    for arm in arms:
        s = summ[arm]
        f = lambda k: f"{s[k]['median']:.4f} ({s[k]['iqr'][0]:.3f}-{s[k]['iqr'][1]:.3f})"
        print(f"  {arm:>18} {f('int_l2_f'):>14} {f('final_l2_f'):>16} {f('final_l2_fp'):>16}")
    print(f"  diagnostic, interior [-1.3, 1.3] final e_F (median): " + ", ".join(f"{arm} {summ[arm]['final_l2_f_interior']['median']:.4f}" for arm in arms))
    print("  FR effect (fr_uniform vs abf, paired, %):")
    for est in ("kernel", "hist"):
        for k in ("int_l2_f", "final_l2_f", "final_l2_fp", "final_l2_f_interior"):
            e = effect[est][k]
            print(f"    {est:>6} {k:>20}: {e['median']:+7.2f} [{e['ci95'][0]:+7.2f}, {e['ci95'][1]:+7.2f}]  wins {e['wins']}/{e['n']}")
    print("  histogram vs kernel (same arm, paired on identical noise, %):")
    for arm in ("abf", "fr_uniform"):
        for k in ("int_l2_f", "final_l2_f", "final_l2_fp", "final_l2_f_interior"):
            e = cross[arm][k]
            print(f"    {arm:>10} {k:>20}: {e['median']:+7.2f} [{e['ci95'][0]:+7.2f}, {e['ci95'][1]:+7.2f}]  wins {e['wins']}/{e['n']}")
    print(f"  safety: kernel FR min windowed ESS/N {health['kernel']['min_ess_frac']:.3f}, hist FR {health['hist']['min_ess_frac']:.3f}; "
          f"repl fraction kernel {health['kernel']['repl_fraction']['median']:.4f}, hist {health['hist']['repl_fraction']['median']:.4f}")
    print(f"  hist support: min count in window {summ['hist_abf']['min_count_window']['median']:.0f} (ABF) / {summ['hist_fr_uniform']['min_count_window']['median']:.0f} (FR); "
          f"untrusted fraction {summ['hist_abf']['frac_untrusted_window']['median']:.3f} / {summ['hist_fr_uniform']['frac_untrusted_window']['median']:.3f}; "
          f"max |bias| {summ['hist_abf']['max_abs_bias_force']['median']:.2f} vs kernel {summ['kernel_abf']['max_abs_bias_force']['median']:.2f}")
    print(f"  efficiency: kernel batch {kmeta['wall_seconds']:.1f}s ({1e3 * kmeta['wall_seconds_per_step']:.3f} ms/step, R={kmeta['R']}), "
          f"histogram batch {hmeta['wall_seconds']:.1f}s ({1e3 * hmeta['wall_seconds_per_step']:.3f} ms/step) -> {kmeta['wall_seconds'] / hmeta['wall_seconds']:.2f}x")
    print(f"  VERDICT: replicates={replicates} (same sign {same_sign}, CI rule {ci_ok}, magnitude I_F {mag_int}, e_F(T) {mag_fin}); "
          f"absolute accuracy acceptable={acc_ok} (hist ABF vs kernel ABF: I_F {cross['abf']['int_l2_f']['median']:+.1f}%, e_F(T) {cross['abf']['final_l2_f']['median']:+.1f}%)")
    if len(seeds) < n_exp:
        print(f"  NOTE: {len(seeds)} of {n_exp} seeds present -- not the full block")

    # ---- figures
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, key, lab in zip(axes, ("l2_f_t", "l2_fp_t"), ("$e_F(t)$", "$e_{F'}(t)$ (own estimator)")):
        for arm in arms:
            est, meth = ("kernel", "abf") if arm == "kernel_abf" else ("kernel", "fr_uniform") if arm == "kernel_fr_uniform" else ("histogram", "abf") if arm == "hist_abf" else ("histogram", "fr_uniform")
            S = series(arm, key)
            med, lo, hi = np.median(S, 0), np.percentile(S, 25, 0), np.percentile(S, 75, 0)
            ax.plot(t, med, color=COL[est], ls=LS[meth], label=f"{est} {'ABF' if meth == 'abf' else 'ABF+FR (uniform)'}")
            ax.fill_between(t, lo, hi, color=COL[est], alpha=0.12, lw=0)
        ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_title(lab)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Entropic bottleneck confirmation ({len(seeds)} seeds; kernel h=0.07 vs histogram Delta={3.6 / nb:.3f}); median + IQR", fontsize=9)
    f_conv = _save(fig, os.path.join(RES, "figures"), "eb_convergence")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.asarray(K["abf"][seeds[0]]["x_grid"], dtype=float)
    F_ref, Fp_ref = np.asarray(K["abf"][seeds[0]]["F_ref"]), np.asarray(K["abf"][seeds[0]]["Fp_ref"])
    axes[0].plot(x, F_ref, "k", lw=1.6, label="reference"); axes[1].plot(x, Fp_ref, "k", lw=1.6, label="reference")
    for arm in arms:
        est, meth = ("kernel", "abf") if arm == "kernel_abf" else ("kernel", "fr_uniform") if arm == "kernel_fr_uniform" else ("histogram", "abf") if arm == "hist_abf" else ("histogram", "fr_uniform")
        Fm = np.median(np.stack([np.asarray(arms[arm][s]["F_hat"]) for s in seeds]), 0)
        axes[0].plot(x, Fm, color=COL[est], ls=LS[meth], lw=1, label=f"{est} {'ABF' if meth == 'abf' else 'ABF+FR'}")
        if est == "kernel":
            axes[1].plot(x, np.median(np.stack([np.asarray(arms[arm][s]["Fp_hat"]) for s in seeds]), 0), color=COL[est], ls=LS[meth], lw=1)
        else:
            edges = np.asarray(arms[arm][seeds[0]]["hist_edges"])
            Gm = np.median(np.stack([np.asarray(arms[arm][s]["Fp_bins"]) for s in seeds]), 0)
            axes[1].stairs(Gm, edges, color=COL[est], ls=LS[meth], lw=1, baseline=None)
    for ax in axes:
        ax.axvspan(-1.8, -1.5, color="0.9"); ax.axvspan(1.5, 1.8, color="0.9"); ax.set_xlabel("x")
    axes[0].set_title("final $F$ (seed-median profile)"); axes[1].set_title("final $F'$ (kernel: grid; histogram: bins)")
    axes[0].legend(fontsize=7)
    f_prof = _save(fig, os.path.join(RES, "figures"), "eb_final_profiles")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i, (k, lab) in enumerate((("int_l2_f", "$\\Delta I_F$"), ("final_l2_f", "$\\Delta e_F(T)$"), ("final_l2_fp", "$\\Delta e_{F'}(T)$"))):
        for j, est in enumerate(("kernel", "hist")):
            e = effect[est][k]
            xx = i + (-0.17 if est == "kernel" else 0.17)
            ax.errorbar([xx], [e["median"]], yerr=[[e["median"] - e["ci95"][0]], [e["ci95"][1] - e["median"]]], fmt="o", color=COL["kernel" if est == "kernel" else "histogram"],
                        capsize=3, label=(est if i == 0 else None))
            ax.annotate(f"{e['wins']}/{e['n']}", (xx, e["ci95"][1]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["$I_F$", "$e_F(T)$", "$e_{F'}(T)$"])
    ax.set_ylabel("FR effect vs ABF (%)"); ax.legend(fontsize=8); ax.set_title("Entropic bottleneck: FR effect by estimator", fontsize=9)
    f_eff = _save(fig, os.path.join(RES, "figures"), "eb_effect_size")

    out = dict(system="entropic_bottleneck", seeds=seeds, n_expected=n_exp, selected_n_bins=nb, delta=3.6 / nb,
               arms=list(arms), metrics_summary=summ, fr_effect=effect, hist_vs_kernel=cross, health=health,
               verdict=dict(replicates=replicates, same_sign=same_sign, ci_rule=ci_ok, magnitude_int=mag_int, magnitude_final=mag_fin,
                            absolute_accuracy_acceptable=acc_ok, rules=rules),
               efficiency=dict(kernel_wall_s=kmeta["wall_seconds"], hist_wall_s=hmeta["wall_seconds"],
                               kernel_ms_per_step=1e3 * kmeta["wall_seconds_per_step"], hist_ms_per_step=1e3 * hmeta["wall_seconds_per_step"],
                               speedup=kmeta["wall_seconds"] / hmeta["wall_seconds"], R=kmeta["R"],
                               kernel_peak_gpu_mem_bytes=kmeta.get("peak_gpu_mem_bytes"), hist_peak_gpu_mem_bytes=hmeta.get("peak_gpu_mem_bytes")),
               figures=[os.path.relpath(p, ROOT) for p in (f_conv, f_prof, f_eff)],
               git_rev=kmeta.get("git_rev"), timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(out, open(os.path.join(RES, "confirmation", "summary.json"), "w"), indent=2, default=float)
    with open(os.path.join(RES, "confirmation", "comparison.csv"), "w") as fh:
        fh.write("seed," + ",".join(f"{arm}_{k}" for arm in arms for k in ("int_l2_f", "final_l2_f", "final_l2_fp")) + "\n")
        for i, s in enumerate(seeds):
            fh.write(f"{s}," + ",".join(f"{metrics[arm][k][i]:.6f}" for arm in arms for k in ("int_l2_f", "final_l2_f", "final_l2_fp")) + "\n")
    print(f"  wrote {os.path.relpath(RES, ROOT)}/confirmation/summary.json, comparison.csv; figures {[os.path.basename(p) for p in (f_conv, f_prof, f_eff)]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true", help="A1: apply the frozen bin-selection rule and freeze the width")
    ap.add_argument("--reselect", action="store_true", help="allow rewriting an existing frozen entry (audit use only)")
    a = ap.parse_args()
    if a.select:
        select(a)
    else:
        confirm(a)


if __name__ == "__main__":
    main()
