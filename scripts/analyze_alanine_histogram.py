#!/usr/bin/env python
"""Analysis for the alanine adaptive-box histogram study (docs/ALANINE_HISTOGRAM_ABF.md).

Stage A (ABF only): every histogram arm vs the closed kernel ABF baseline
(results/fr_start_timing/alanine/abf, paired by construction), on the RAW read-out (the
histogram's own) and on the kernel-matched read-out (the kernel's accepted one); the frozen
selection rule writes configs/alanine_histogram/selected.json once all three candidates exist.
Stage B (FR): every histogram FR arm vs its MATCHED histogram ABF arm (same c_min, warm-up,
levels) with the FR-start-timing statistics and verdict rules, plus the kernel FR arm at the
same (start, rate) for the cross-estimator table.  Safe on partial data.

    python scripts/analyze_alanine_histogram.py [--no-figs]
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
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "..", "src"))

from analyze_fr_start_timing import (FLOORS, _colors, _style, flat_stats,  # noqa: E402
                                     integrate, paired_stats, speedups, verdict, write_csv)
from alanine.metrics_ala import aligned_l2, build_masks, smooth_reference  # noqa: E402

ROOT = os.path.join(SCRIPTS, "..")
REF = os.path.join(ROOT, "results/alanine/reference/reference.npz")
KERNEL_ROOT = os.path.join(ROOT, "results/fr_start_timing/alanine")
HIST_ROOT = os.path.join(ROOT, "results/alanine_histogram")
OUT = os.path.join(HIST_ROOT, "analysis")
SELECTED = os.path.join(ROOT, "configs/alanine_histogram/selected.json")
W0, W1 = (1.0, 100.0), (5.0, 100.0)
CANDIDATES = (50.0, 200.0, 800.0)
READOUTS = ("raw", "km", "raw_u8", "sm")
BOOT = os.path.join(os.path.dirname(REF), "bootstrap.npz")


# --------------------------------------------------------------------------- loading
def load_runs(root, prefix):
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "raw", "*.npz"))):
        stage = os.path.basename(os.path.dirname(os.path.dirname(f)))
        d = np.load(f, allow_pickle=True)
        if "final_pmf" not in d.files:
            continue
        meta = json.loads(str(d["meta"]))
        keys = [k for k in d.files if k not in ("meta", "pmf", "marg_hist", "final_f1s", "final_f2s")]
        data = {k: np.asarray(d[k]) for k in keys}
        data["pmf"] = np.asarray(d["pmf"])
        runs[prefix + stage] = dict(stage=stage, data=data, meta=meta, path=f,
                                    t=np.asarray(d["times"], float))
    return runs


def arm_info(run):
    m = run["meta"]
    return dict(method=m["method"], estimator=m.get("abf_estimator", "kernel"),
                c_min=float(m.get("abf_min_count", 200.0)),
                warmup_ps=float(m.get("abf_warmup_steps", 5000)) * 1e-3,
                levels=int(m.get("abf_hist_levels", 4)), fixed=bool(m.get("abf_hist_fixed", False)),
                fr_start_ps=float(m.get("fr_start_steps", 20000)) * 1e-3,
                fr_rate=float(m.get("fr_rate", 0.02)), fr_every=int(m.get("fr_every", 500)))


def error_series(runs):
    refd = np.load(REF, allow_pickle=True)
    F_ref = refd["F"]
    rmeta = json.load(open(os.path.join(os.path.dirname(REF), "meta.json")))
    kT, n_grid = float(rmeta["kT_kJ"]), int(rmeta["n_grid"])
    pack = build_masks(F_ref, kT)
    F_sm = smooth_reference(F_ref, 0.08, n_grid)
    w_eq, w_u8 = pack["weights"]["equilibrium"], pack["weights"]["uniform8"]
    import torch
    from alkanes import density2d as d2
    g1, g2, _, _ = d2.torus_grid(n_grid, n_grid, dtype=torch.float64)
    K1, K2 = d2.kernels(g1, g2, 0.08, 0.08)
    K1, K2 = K1 / K1.sum(1, keepdim=True), K2 / K2.sum(1, keepdim=True)
    floor = float("nan")
    if os.path.exists(BOOT):
        se = np.load(BOOT, allow_pickle=True)["F_se"]
        ok = (w_eq > 0) & np.isfinite(se)
        floor = float(np.sqrt((se[ok] ** 2 * w_eq[ok]).sum() / w_eq[ok].sum()))
    for run in runs.values():
        pmf = run["data"]["pmf"]
        T, R = pmf.shape[:2]
        e = {k: np.zeros((T, R)) for k in READOUTS}
        for ti in range(T):
            psm = d2.smooth2(torch.as_tensor(pmf[ti]), K1, K2).numpy()
            for r in range(R):
                e["raw"][ti, r] = aligned_l2(pmf[ti, r], pack["F"], w_eq)
                e["km"][ti, r] = aligned_l2(pmf[ti, r], F_sm, w_eq)
                e["raw_u8"][ti, r] = aligned_l2(pmf[ti, r], pack["F"], w_u8)
                e["sm"][ti, r] = aligned_l2(psm[r], F_sm, w_eq)
        run["e"] = e
        del run["data"]["pmf"]
    return dict(reference=REF, kT_kJ=kT, n_grid=n_grid, reference_noise_floor_raw=floor,
                raw_vs_km_reference_constant=float(aligned_l2(pack["F"], F_sm, w_eq)),
                raw="equilibrium-weighted aligned L2 vs the RAW reference on the 97x97 cells (8 kT mask); its endpoint floor is the reference's own bootstrap noise",
                km="the same vs the row-normalised K_h * F_ref, h = 0.08 (kernel-matched; the kernel arm's accepted read-out)",
                raw_u8="uniform weight on the 8 kT mask vs the raw reference (sign check)",
                sm="estimate AND reference both smoothed at h = 0.08 (common read-out for cross-estimator endpoints)")


def med_int(t, e, w):
    return float(np.median(integrate(t, e, *w)))


def at_times(t, series, times=(1, 2, 5, 10, 20, 50, 100)):
    out = {}
    for tt in times:
        i = int(np.argmin(np.abs(t - tt)))
        out[f"@{tt:g}"] = float(np.nanmedian(series[i]))
    return out


# --------------------------------------------------------------------------- stage A
def stage_a(runs, kabf, t):
    rows = []
    for name, run in sorted(runs.items()):
        info = arm_info(run)
        if info["method"] != "abf":
            continue
        rec = dict(arm=name, **info, n_seeds=int(run["e"]["raw"].shape[1]),
                   ms_per_step=run["meta"].get("ms_per_step"), wall_s=run["meta"].get("wall_seconds"),
                   clip_fraction=run["meta"].get("clip_fraction"))
        for ro in READOUTS:
            e = run["e"][ro]
            rec[f"IF_W0_{ro}"] = med_int(t, e, W0)
            rec[f"IF_W1_{ro}"] = med_int(t, e, W1)
            rec[f"final_{ro}"] = float(np.median(e[-1]))
            if kabf is not None and name != "kernel_abf":
                ek = kabf["e"][ro]
                rec.update(flat_stats(f"vsK_IF_W1_{ro}", paired_stats(integrate(t, ek, *W1), integrate(t, e, *W1))))
                rec.update(flat_stats(f"vsK_IF_W0_{ro}", paired_stats(integrate(t, ek, *W0), integrate(t, e, *W0))))
                rec.update(flat_stats(f"vsK_final_{ro}", paired_stats(ek[-1], e[-1])))
        d = run["data"]
        rec.update({f"trust{k}": v for k, v in at_times(t, d["trust_frac"][:, None] if d["trust_frac"].ndim == 1 else d["trust_frac"]).items()})
        if "hist_level_mean" in d:
            rec.update({f"level{k}": v for k, v in at_times(t, d["hist_level_mean"]).items()})
            rec.update({f"untrusted{k}": v for k, v in at_times(t, d["hist_untrusted_frac"]).items()})
        rec.update({f"kl{k}": v for k, v in at_times(t, d["kl_uniform"], (1, 5, 20, 100)).items()})
        rec.update({f"c7ax{k}": v for k, v in at_times(t, d["basin_frac"][:, :, 2], (5, 20, 100)).items()})
        rec["c7ax_first_hit_ps_med"] = float(np.median(d["first_hit"][:, 2]) * 1e-3)
        outside = 1.0 - d["basin_frac"].sum(-1)
        rec.update({f"outside{k}": v for k, v in at_times(t, outside, (1, 5, 20, 100)).items()})
        rec.update({f"T{k}": v for k, v in at_times(t, np.asarray(d["temperature"], float)[:, None], (1, 2, 5, 100)).items()})
        rows.append(rec)
    return rows


def select(rows_a, runs, t):
    """Frozen rule, docs/ALANINE_HISTOGRAM_ABF.md section 5.  Returns (selection dict | None)."""
    cands = {r["c_min"]: r for r in rows_a
             if r["estimator"] == "histogram" and not r["fixed"] and r["warmup_ps"] == 0.0
             and r["c_min"] in CANDIDATES and r["levels"] == 4}
    if set(cands) != set(CANDIDATES):
        return None
    by_if = sorted(CANDIDATES, key=lambda c: cands[c]["IF_W0_raw"])
    best = by_if[0]
    # tie clause: within 3 % -> nearer 200
    ties = [c for c in CANDIDATES if cands[c]["IF_W0_raw"] <= 1.03 * cands[best]["IF_W0_raw"]]
    chosen = min(ties, key=lambda c: (abs(np.log(c / 200.0)), cands[c]["IF_W0_raw"]))
    best_final = min(cands[c]["final_raw"] for c in CANDIDATES)
    order = [chosen] + [c for c in by_if if c != chosen]
    for c in order:
        if cands[c]["final_raw"] <= 1.10 * best_final:
            chosen = c
            break
    log = dict(IF_W0_raw={str(c): cands[c]["IF_W0_raw"] for c in CANDIDATES},
               final_raw={str(c): cands[c]["final_raw"] for c in CANDIDATES},
               ties_within_3pct=[str(c) for c in ties], endpoint_clause_best_final=best_final)
    warmup_ps = 0.0
    ctrl = [n for n, r in runs.items() if arm_info(r)["method"] == "abf" and arm_info(r)["estimator"] == "histogram"
            and not arm_info(r)["fixed"] and arm_info(r)["c_min"] == 200.0 and arm_info(r)["warmup_ps"] == 5.0]
    chosen_name = [n for n, r in runs.items() if arm_info(r)["method"] == "abf" and arm_info(r)["estimator"] == "histogram"
                   and not arm_info(r)["fixed"] and arm_info(r)["c_min"] == chosen and arm_info(r)["warmup_ps"] == 0.0][0]
    if ctrl:
        s = paired_stats(integrate(t, runs[ctrl[0]]["e"]["raw"], *W1), integrate(t, runs[chosen_name]["e"]["raw"], *W1))
        log["warmup_clause_vs_w5_IF_W1_raw"] = s
        if s["median"] > 0.05 and s["lo"] > 0:
            warmup_ps = 5.0
    else:
        log["warmup_clause_vs_w5_IF_W1_raw"] = "control arm hA_c200_w5 absent; warm-up 0 by default"
    return dict(c_min=chosen, warmup_ps=warmup_ps, levels=4, fixed=False, log=log,
                selected_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                rule="docs/ALANINE_HISTOGRAM_ABF.md section 5")


# --------------------------------------------------------------------------- stage B
def matched_abf(runs, info):
    for n, r in runs.items():
        i = arm_info(r)
        if (i["method"] == "abf" and i["estimator"] == info["estimator"] and i["c_min"] == info["c_min"]
                and i["warmup_ps"] == info["warmup_ps"] and i["levels"] == info["levels"] and i["fixed"] == info["fixed"]):
            return n
    return None


def fr_row(name, run, abf, t, ro_primary="raw"):
    info = arm_info(run)
    e, ea = run["e"][ro_primary], abf["e"][ro_primary]
    T = float(t[-1])
    own = (max(info["fr_start_ps"], 1.0), T)
    rec = dict(arm=name, **info, matched_abf=abf.get("name"), n_seeds=int(e.shape[1]),
               own_window=f"[{own[0]:g},{own[1]:g}]", readout=ro_primary)
    prim = paired_stats(integrate(t, ea, *own), integrate(t, e, *own))
    rec.update(flat_stats("dIF_own", prim))
    for ro in READOUTS:
        for wn, w in (("W0", W0), ("W1", W1), ("own", own)):
            rec.update(flat_stats(f"dIF_{wn}_{ro}", paired_stats(integrate(t, abf["e"][ro], *w), integrate(t, run["e"][ro], *w))))
        rec.update(flat_stats(f"final_{ro}", paired_stats(abf["e"][ro][-1], run["e"][ro][-1])))
    fin = paired_stats(ea[-1], e[-1])
    rec.update(flat_stats("final", fin))
    rec["abf_final_med"], rec["arm_final_med"] = float(np.median(ea[-1])), float(np.median(e[-1]))
    sp, e0 = speedups(t, ea, e, T, own[0])
    rec["e0_own"] = e0
    for k, v in sp.items():
        rec[f"tau_{k}_abf"], rec[f"tau_{k}_arm"], rec[f"S_{k}"] = v["tau_abf"], v["tau_arm"], v["speedup"]
        rec[f"cens_{k}"] = f"{v['censored_abf']}/{v['censored_arm']}"
    ratio = np.median(e / ea, 1)
    rec["ratio_min"], rec["ratio_min_t"] = float(ratio.min()), float(t[int(np.argmin(ratio))])
    rec["ratio_final"] = float(ratio[-1])
    d = run["data"]
    N = int(run["meta"]["n_replicas"])
    sel = t >= own[0] - 1e-9
    n_opp = max(int((int(run["meta"]["n_steps"]) - int(round(info["fr_start_ps"] * 1000))) // info["fr_every"]) + 1, 1)
    ev = d["total_events"].astype(float)
    rec["ess_age_min"] = float(np.nanmin(d["ess_age"][sel]))
    rec["ess_perm_min"] = float(np.nanmin(d["ess_perm"][sel]))
    rec["wmax_max"] = float(np.nanmax(d["wmax"][sel]))
    rec["events_per_opp_med"] = float(np.median(ev)) / n_opp
    rec["event_frac_per_opp_max"] = float(ev.max()) / N / n_opp
    rec["event_frac_cum_med"] = float(np.median(ev)) / N
    rec["n_opportunities"] = n_opp
    floors_ok = (rec["ess_age_min"] >= FLOORS["ess"] and rec["wmax_max"] <= FLOORS["wmax"]
                 and rec["event_frac_per_opp_max"] < FLOORS["event_frac"])
    rec["floors_ok"] = bool(floors_ok)
    rec["kl_final_arm"], rec["kl_final_abf"] = float(np.median(d["kl_uniform"][-1])), float(np.median(abf["data"]["kl_uniform"][-1]))
    rec["kl_min_arm"] = float(np.median(d["kl_uniform"], 1).min())
    rec["c7ax_final_arm"] = float(np.median(d["basin_frac"][-1, :, 2]))
    rec["c7ax_final_abf"] = float(np.median(abf["data"]["basin_frac"][-1, :, 2]))
    rec["clip_fraction"] = float(run["meta"].get("clip_fraction", np.nan))
    rec["verdict"] = verdict(prim, fin, floors_ok)
    return rec


def stage_b(runs, t):
    rows = []
    for name, run in sorted(runs.items()):
        info = arm_info(run)
        if info["method"] == "abf":
            continue
        mname = matched_abf(runs, info)
        if mname is None:
            print(f"[stage B] {name}: no matched ABF arm (estimator {info['estimator']}, c_min {info['c_min']}, warm-up {info['warmup_ps']}) -- skipped")
            continue
        abf = dict(runs[mname], name=mname)
        ro = "raw" if info["estimator"] == "histogram" else "km"
        rows.append(fr_row(name, run, abf, t, ro_primary=ro))
    return rows


# --------------------------------------------------------------------------- figures
def figures(runs, rows_a, rows_b, t, out):
    plt = _style()
    fd = os.path.join(out, "figures")
    os.makedirs(fd, exist_ok=True)
    kabf = runs.get("kernel_abf")
    hist_abf = [n for n, r in runs.items() if arm_info(r)["method"] == "abf" and arm_info(r)["estimator"] == "histogram"]
    hist_fr = [n for n, r in runs.items() if arm_info(r)["method"] != "abf" and arm_info(r)["estimator"] == "histogram"]
    cols_a = dict(zip(sorted(hist_abf), _colors(max(len(hist_abf), 1))))
    # --- stage A
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), layout="constrained")
    ax = axes[0, 0]
    if kabf is not None:
        ax.plot(t, np.median(kabf["e"]["raw"], 1), "k-", lw=1.8, label="kernel ABF, raw read-out")
        ax.plot(t, np.median(kabf["e"]["km"], 1), "k--", lw=1.4, label="kernel ABF, kernel-matched")
    for n in sorted(hist_abf):
        ax.plot(t, np.median(runs[n]["e"]["raw"], 1), color=cols_a[n], lw=1.2, label=n.replace("hist_", ""))
    ax.set_yscale("log"); ax.set_xlabel("t (ps)"); ax.set_ylabel("aligned L2 FES error (kJ/mol)")
    ax.set_title("ABF only: median over 16 paired seeds", fontsize=9); ax.legend(fontsize=6, frameon=False)
    ax = axes[0, 1]
    if kabf is not None:
        for n in sorted(hist_abf):
            ax.plot(t, np.median(runs[n]["e"]["raw"] / kabf["e"]["raw"], 1), color=cols_a[n], lw=1.2, label=n.replace("hist_", "") + " / kernel (raw)")
            ax.plot(t, np.median(runs[n]["e"]["raw"] / kabf["e"]["km"], 1), color=cols_a[n], lw=0.9, ls="--")
    ax.axhline(1, color="k", lw=0.7, ls=":"); ax.set_yscale("log")
    ax.set_xlabel("t (ps)"); ax.set_ylabel("e_hist(raw) / e_kernel  (solid: raw, dashed: km)")
    ax.set_title("histogram ABF relative to kernel ABF", fontsize=9); ax.legend(fontsize=6, frameon=False)
    ax = axes[1, 0]
    for n in sorted(hist_abf):
        d = runs[n]["data"]
        if "hist_level_mean" in d:
            ax.plot(t, np.nanmedian(d["hist_level_mean"], 1), color=cols_a[n], lw=1.2, label=n.replace("hist_", "") + " level")
            ax.plot(t, np.nanmedian(d["hist_untrusted_frac"], 1) * 4, color=cols_a[n], lw=0.9, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("t (ps)"); ax.set_ylabel("mean box level (solid);  4 x untrusted cell fraction (dotted)")
    ax.set_title("adaptive bin size over time", fontsize=9); ax.legend(fontsize=6, frameon=False)
    ax = axes[1, 1]
    if kabf is not None:
        ax.plot(t, np.median(kabf["data"]["kl_uniform"], 1), "k-", lw=1.6, label="kernel ABF")
    for n in sorted(hist_abf):
        ax.plot(t, np.median(runs[n]["data"]["kl_uniform"], 1), color=cols_a[n], lw=1.0, label=n.replace("hist_", ""))
    ax.set_xscale("log"); ax.set_xlabel("t (ps)"); ax.set_ylabel("KL(p_t || uniform torus)"); ax.legend(fontsize=6, frameon=False)
    ax.set_title("walker marginal vs the FR target", fontsize=9)
    fig.savefig(os.path.join(fd, "stageA_curves.png"), dpi=160); fig.savefig(os.path.join(fd, "stageA_curves.pdf")); plt.close(fig)
    # --- stage B
    if hist_fr:
        cols_b = dict(zip(sorted(hist_fr), _colors(len(hist_fr))))
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), layout="constrained")
        ax = axes[0, 0]
        drawn = set()
        for n in sorted(hist_fr):
            m = matched_abf(runs, arm_info(runs[n]))
            if m and m not in drawn:
                ax.plot(t, np.median(runs[m]["e"]["raw"], 1), "k-", lw=1.8, label=f"{m.replace('hist_', '')} (ABF)"); drawn.add(m)
            info = arm_info(runs[n])
            ax.plot(t, np.median(runs[n]["e"]["raw"], 1), color=cols_b[n], lw=1.1, label=f"{n.replace('hist_', '')} (FR @{info['fr_start_ps']:g} ps, r={info['fr_rate']:g})")
            ax.axvline(max(info["fr_start_ps"], 0.5), color=cols_b[n], lw=0.5, ls=":")
        ax.set_yscale("log"); ax.set_xlabel("t (ps)"); ax.set_ylabel("aligned L2 FES error, raw read-out (kJ/mol)")
        ax.set_title("histogram estimator: ABF vs ABF + uniform FR", fontsize=9); ax.legend(fontsize=6, frameon=False)
        ax = axes[0, 1]
        for n in sorted(hist_fr):
            m = matched_abf(runs, arm_info(runs[n]))
            if m:
                ax.plot(t, np.median(runs[n]["e"]["raw"] / runs[m]["e"]["raw"], 1), color=cols_b[n], lw=1.1, label=n.replace("hist_", ""))
            info = arm_info(runs[n])
            kn = f"kernel_{'u' if info['method'] == 'fr_uniform' else 'o'}{int(round(info['fr_rate'] * 100)):02d}_t{int(info['fr_start_ps']):d}"
            if kn in runs and kabf is not None:
                ax.plot(t, np.median(runs[kn]["e"]["km"] / kabf["e"]["km"], 1), color=cols_b[n], lw=0.9, ls="--")
        ax.axhline(1, color="k", lw=0.7, ls=":"); ax.set_ylim(0.6, 1.6)
        ax.set_xlabel("t (ps)"); ax.set_ylabel("e_FR / e_ABF (median per-seed ratio)")
        ax.set_title("solid: histogram (raw);  dashed: kernel arm at the same start/rate (km)", fontsize=8); ax.legend(fontsize=6, frameon=False)
        ax = axes[1, 0]
        for n in sorted(hist_fr):
            ax.plot(t, np.median(runs[n]["data"]["kl_uniform"], 1), color=cols_b[n], lw=1.0, label=n.replace("hist_", ""))
            m = matched_abf(runs, arm_info(runs[n]))
            if m:
                ax.plot(t, np.median(runs[m]["data"]["kl_uniform"], 1), "k-", lw=1.4)
        ax.set_xlabel("t (ps)"); ax.set_ylabel("KL(p_t || uniform torus)"); ax.set_title("marginal vs target (black: matched ABF)", fontsize=9)
        ax = axes[1, 1]
        for n in sorted(hist_fr):
            ax.plot(t, np.median(runs[n]["data"]["ess_age"], 1), color=cols_b[n], lw=1.0, label=n.replace("hist_", ""))
        ax.axhline(0.30, color="k", ls=":", lw=0.7); ax.set_xlabel("t (ps)"); ax.set_ylabel("age-aware ESS / N"); ax.legend(fontsize=6, frameon=False)
        fig.savefig(os.path.join(fd, "stageB_curves.png"), dpi=160); fig.savefig(os.path.join(fd, "stageB_curves.pdf")); plt.close(fig)
    if rows_b:
        fig, ax = plt.subplots(figsize=(6.5, 0.6 + 0.45 * len(rows_b)), layout="constrained")
        for i, r in enumerate(rows_b):
            for j, (key, mk, lab) in enumerate((("dIF_own", "o", "own window"), ("dIF_W1_" + r["readout"], "s", "W1 [5,100] ps"), ("final", "^", "final"))):
                y = i + (j - 1) * 0.22
                ax.errorbar(100 * r[f"{key}_median"], y, xerr=[[100 * (r[f"{key}_median"] - r[f"{key}_lo"])], [100 * (r[f"{key}_hi"] - r[f"{key}_median"])]],
                            fmt=mk, ms=4, color=["C0", "C1", "C2"][j], capsize=2, label=lab if i == 0 else None)
        ax.axvline(0, color="k", lw=0.7); ax.axvline(-10, color="gray", lw=0.6, ls="--")
        ax.set_yticks(range(len(rows_b))); ax.set_yticklabels([r["arm"] for r in rows_b], fontsize=8)
        ax.set_xlabel("paired relative change vs matched ABF (%)  [negative = better]"); ax.legend(fontsize=7, frameon=False)
        ax.set_title("FR arms vs matched ABF: histogram arms on the raw read-out, kernel arms on the kernel-matched one", fontsize=8)
        fig.savefig(os.path.join(fd, "stageB_forest.png"), dpi=160); fig.savefig(os.path.join(fd, "stageB_forest.pdf")); plt.close(fig)


# --------------------------------------------------------------------------- report
def pct(r, k):
    return f"{100 * r[k + '_median']:+6.2f} % [{100 * r[k + '_lo']:+6.2f}, {100 * r[k + '_hi']:+6.2f}] {int(round(r[k + '_win'] * r[k + '_n']))}/{r[k + '_n']}"


def scoreboard(rows_a, rows_b, sel, prov, out):
    L = ["# Alanine adaptive-box histogram study: scoreboard", "",
         f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. Read-outs: raw = {prov['raw']}; km = {prov['km']}.", ""]
    L += ["## Stage A: ABF only (medians over 16 paired seeds; vs kernel ABF = paired relative change)", "",
          "| arm | estimator | c_min | levels | fixed | warm-up | I_F W0 raw | I_F W1 raw | final raw | I_F W1 km | final km | vs kernel I_F W1 (raw) | vs kernel final (raw) | vs kernel I_F W1 (km) | vs kernel final (km) | level @1/5/20/100 ps | untrusted @1/5/100 | trust @1/100 | ms/step |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows_a:
        lv = "/".join(f"{r.get(f'level@{k}', float('nan')):.2f}" for k in (1, 5, 20, 100)) if "level@1" in r else "-"
        un = "/".join(f"{r.get(f'untrusted@{k}', float('nan')):.3f}" for k in (1, 5, 100)) if "untrusted@1" in r else "-"
        tr = f"{r.get('trust@1', float('nan')):.3f}/{r.get('trust@100', float('nan')):.3f}"
        vk = [pct(r, k) if f"{k}_median" in r else "-" for k in ("vsK_IF_W1_raw", "vsK_final_raw", "vsK_IF_W1_km", "vsK_final_km")]
        L.append(f"| {r['arm']} | {r['estimator']} | {r['c_min']:g} | {r['levels']} | {r['fixed']} | {r['warmup_ps']:g} ps | {r['IF_W0_raw']:.3f} | {r['IF_W1_raw']:.3f} | {r['final_raw']:.4f} | {r['IF_W1_km']:.3f} | {r['final_km']:.4f} | {vk[0]} | {vk[1]} | {vk[2]} | {vk[3]} | {lv} | {un} | {tr} | {r['ms_per_step'] if r['ms_per_step'] is None else f'{r['ms_per_step']:.2f}'} |")
    L += ["", f"Raw read-out endpoint floor = reference bootstrap noise {prov['reference_noise_floor_raw']:.3f} kJ/mol (raw-vs-smoothed reference constant {prov['raw_vs_km_reference_constant']:.3f}); an endpoint at that level is read-out-limited for every arm.", "",
          "Early dynamics (medians): fraction of walkers outside the three basins, kinetic temperature, C7ax first hit, KL to uniform.", "",
          "| arm | warm-up | outside @1/5/20/100 ps | T @1/2/5/100 ps (K) | C7ax first hit (ps) | KL(p||U) @1/5/20/100 | final sm | vs kernel final (sm) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows_a:
        vk = pct(r, "vsK_final_sm") if "vsK_final_sm_median" in r else "-"
        L.append(f"| {r['arm']} | {r['warmup_ps']:g} ps | " + "/".join(f"{r[f'outside@{k}']:.3f}" for k in (1, 5, 20, 100)) + " | "
                 + "/".join(f"{r[f'T@{k}']:.1f}" for k in (1, 2, 5, 100)) + f" | {r['c7ax_first_hit_ps_med']:.2f} | "
                 + "/".join(f"{r[f'kl@{k}']:.2f}" for k in (1, 5, 20, 100)) + f" | {r['final_sm']:.3f} | {vk} |")
    L += ["", f"Selection (frozen rule): {json.dumps({k: v for k, v in (sel or {}).items() if k != 'log'})}", ""]
    if rows_b:
        L += ["## Stage B: FR arms vs matched ABF (histogram arms scored on the raw read-out)", "",
              "| arm | estimator | start | rate | matched ABF | dI_F own | dI_F W1 | final | S(e0/2) | S(e0/4) | S(e0/8) | ratio min (t) | ratio final | events/opp | ESS_age min | wmax | KL final arm/abf | C7ax arm/abf | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows_b:
            S = "/".join("-" if r[f"S_{k}"] is None else f"{r[f'S_{k}']:.2f}" for k in ("e0/2", "e0/4", "e0/8"))
            L.append(f"| {r['arm']} | {r['estimator']} | {r['fr_start_ps']:g} ps | {r['fr_rate']:g} | {r['matched_abf']} | {pct(r, 'dIF_own')} | {pct(r, 'dIF_W1_' + r['readout'])} | {pct(r, 'final')} | {S.split('/')[0]} | {S.split('/')[1]} | {S.split('/')[2]} | {r['ratio_min']:.3f} ({r['ratio_min_t']:g}) | {r['ratio_final']:.3f} | {r['events_per_opp_med']:.1f} | {r['ess_age_min']:.3f} | {r['wmax_max']:.3f} | {r['kl_final_arm']:.3f}/{r['kl_final_abf']:.3f} | {r['c7ax_final_arm']:.3f}/{r['c7ax_final_abf']:.3f} | {r['verdict']} |")
    open(os.path.join(out, "scoreboard.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    runs = load_runs(KERNEL_ROOT, "kernel_")
    runs.update(load_runs(HIST_ROOT, "hist_"))
    runs = {k: v for k, v in runs.items() if not k.endswith("analysis")}
    if not runs:
        print("no runs"); return
    prov = error_series(runs)
    t = runs[next(iter(runs))]["t"]
    for n, r in list(runs.items()):
        if not np.allclose(r["t"], t):
            print(f"{n}: time axis differs, dropped"); runs.pop(n)
    kabf = runs.get("kernel_abf")
    rows_a = stage_a(runs, kabf, t)
    sel = None
    if os.path.exists(SELECTED):
        sel = json.load(open(SELECTED))
        print(f"selection already frozen at {sel.get('selected_at')}: c_min {sel['c_min']}, warm-up {sel['warmup_ps']} ps")
    else:
        sel = select(rows_a, runs, t)
        if sel is not None:
            json.dump(sel, open(SELECTED, "w"), indent=2, default=float)
            print(f"SELECTED (frozen now): c_min {sel['c_min']}, warm-up {sel['warmup_ps']} ps -> {SELECTED}")
        else:
            print("selection: candidates incomplete, nothing frozen")
    rows_b = stage_b(runs, t)
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, "stageA.csv"), rows_a)
    write_csv(os.path.join(a.out, "stageB.csv"), rows_b)
    json.dump(dict(provenance=prov, windows=dict(W0=W0, W1=W1), selection=sel,
                   stage_a={r["arm"]: r for r in rows_a}, stage_b={r["arm"]: r for r in rows_b},
                   runs={n: dict(path=r["path"], **arm_info(r)) for n, r in runs.items()}),
              open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=float)
    scoreboard(rows_a, rows_b, sel, prov, a.out)
    if not a.no_figs:
        figures(runs, rows_a, rows_b, t, a.out)
    print(open(os.path.join(a.out, "scoreboard.md")).read())


if __name__ == "__main__":
    main()
