#!/usr/bin/env python
"""Analysis for the FR-start timing experiment (docs/FR_START_TIMING.md).

Two systems, one statistic: for every FR arm, the per-seed paired relative change of the
integrated error against the fresh ABF baseline on the PRIMARY window, plus the secondary
windows, the final error, time-to-accuracy, mechanism series and the genealogy floors --
all frozen in the preregistration.  Partial data are analysed as they arrive (arms still
running are simply absent), so the script is safe to re-run at any time.

    python scripts/analyze_fr_start_timing.py                 # both systems, default roots
    python scripts/analyze_fr_start_timing.py --system alanine --alanine-root <dir>

Outputs under ``<out>/`` (default results/fr_start_timing/analysis):
  alanine_arms.csv / r15_arms.csv        one row per (arm[, cell]) with every endpoint
  alanine_series.csv / r15_series.csv    median-over-seeds curves for plotting
  alanine_summary.json / r15_summary.json  the same plus verdicts and provenance
  figures/                               convergence, ratio, mechanism, forest, genealogy
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "..", "src"))

from alanine.metrics_ala import (aligned_l2, build_masks, paired_bootstrap,  # noqa: E402
                                 smooth_reference)

ROOT = os.path.join(SCRIPTS, "..")
REF_ALA = os.path.join(ROOT, "results/alanine/reference/reference.npz")
CAMPAIGN_ALA = os.path.join(ROOT, "results/uniform_campaign/alanine/N2048_uniform/raw")
CAMPAIGN_R15 = os.path.join(ROOT, "results/uniform_campaign/r15_midbeta_methods/raw")

# frozen thresholds (docs/FR_START_TIMING.md, inherited from the uniform-FR campaign)
ACCEL_MEDIAN = -0.10
FINAL_MARGIN = 0.05
NEUTRAL_BAND = 0.10
PERSIST_FRAC = 0.2
EPS_LEVELS = ("e0/2", "e0/4", "e0/8", "abf_final")
FLOORS = dict(ess=0.30, wmax=0.05, event_frac=0.05)


# --------------------------------------------------------------------------- helpers
def integrate(t, e, t0, t1):
    sel = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
    if sel.sum() < 2:
        return np.full(e.shape[1], np.nan)
    return np.trapezoid(e[sel], t[sel], axis=0)


def paired_stats(a, b):
    """(b - a) / a per seed -> median, BCa CI, win rate (alanine.metrics_ala)."""
    s = paired_bootstrap(np.asarray(a, float), np.asarray(b, float))
    return dict(median=s["median"], lo=s["lo"], hi=s["hi"], win_rate=s["win_rate"], n=s["n"])


def tau_eps(t, e, eps, T):
    """First t with e <= eps sustained for PERSIST_FRAC*T (per seed); NaN = censored."""
    R = e.shape[1]
    hold = PERSIST_FRAC * T
    out = np.full(R, np.nan)
    for r in range(R):
        for i, ti in enumerate(t):
            if ti + hold > T + 1e-9:
                break
            sel = (t >= ti - 1e-9) & (t <= ti + hold + 1e-9)
            if np.all(e[sel, r] <= eps[r] if np.ndim(eps) else e[sel, r] <= eps):
                out[r] = ti
                break
    return out


def speedups(t, e_abf, e_arm, T, t0):
    """tau_eps for both arms at the four frozen levels; e0 = median ABF error at t0."""
    i0 = int(np.argmin(np.abs(t - t0)))
    e0 = float(np.median(e_abf[i0]))
    levels = {"e0/2": e0 / 2, "e0/4": e0 / 4, "e0/8": e0 / 8,
              "abf_final": np.median(e_abf[-1])}
    res = {}
    for name in EPS_LEVELS:
        eps = levels[name]
        ta = tau_eps(t, e_abf, eps, T)
        tb = tau_eps(t, e_arm, eps, T)
        ok = np.isfinite(ta) & np.isfinite(tb)
        res[name] = dict(eps=float(eps), tau_abf=float(np.nanmedian(ta)) if np.isfinite(ta).any() else None,
                         tau_arm=float(np.nanmedian(tb)) if np.isfinite(tb).any() else None,
                         speedup=float(np.median(ta[ok] / tb[ok])) if ok.any() else None,
                         censored_abf=int((~np.isfinite(ta)).sum()),
                         censored_arm=int((~np.isfinite(tb)).sum()), n=int(len(ta)))
    return res, e0


def verdict(primary, final, floors_ok):
    """Frozen rules: acceleration-positive / safe / neutral / harmful / inconclusive."""
    m, hi, lo = primary["median"], primary["hi"], primary["lo"]
    fm, flo, fhi = final["median"], final["lo"], final["hi"]
    if m <= ACCEL_MEDIAN and hi < 0:
        v = "ACCELERATION_POSITIVE"
        if fm <= FINAL_MARGIN:
            v = "SAFE_ACCELERATOR" if floors_ok else "ACCELERATION_POSITIVE_FLOOR_VIOLATION"
        elif not floors_ok:
            v += "_FLOOR_VIOLATION"
        return v
    # Amendment A1 (docs/FR_START_TIMING.md): precedence positive > neutral > harmful.
    # A CI-excluding-zero change inside the +-10% band is neutral in the campaign's sense
    # (small); the suffix _SIG keeps the significance visible instead of hiding it.
    if abs(m) < NEUTRAL_BAND and abs(fm) <= FINAL_MARGIN:
        sig = "_SIG" if (lo > 0 or hi < 0) else ""
        return ("NEUTRAL" + sig) if floors_ok else ("NEUTRAL" + sig + "_FLOOR_VIOLATION")
    if lo > 0 or (fm > FINAL_MARGIN and flo > 0):
        return "HARMFUL" if floors_ok else "HARMFUL_FLOOR_VIOLATION"
    return "INCONCLUSIVE" if floors_ok else "INCONCLUSIVE_FLOOR_VIOLATION"


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")


def flat_stats(prefix, s):
    return {f"{prefix}_median": s["median"], f"{prefix}_lo": s["lo"], f"{prefix}_hi": s["hi"],
            f"{prefix}_win": s["win_rate"], f"{prefix}_n": s["n"]}


# --------------------------------------------------------------------------- alanine
def _ala_arm_meta(stage, meta):
    """(method, fr_start_ps, fr_rate) from the artifact meta, else from the stage name."""
    method = meta["method"]
    if "fr_start_steps" in meta:
        return method, float(meta["fr_start_steps"]) * 0.001, float(meta["fr_rate"])
    m = re.match(r"([uo])(\d\d)_t(\d+)$", stage)
    if m:
        return method, float(m.group(3)), int(m.group(2)) / 100.0
    return method, 20.0, 0.02


def load_alanine(root, campaign_dir):
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "raw", "*.npz"))):
        stage = os.path.basename(os.path.dirname(os.path.dirname(f)))
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        runs[stage] = dict(data={k: np.asarray(d[k]) for k in d.files if k != "meta"}, meta=meta,
                           arm_meta=_ala_arm_meta(stage, meta), path=f)
    if campaign_dir and os.path.isdir(campaign_dir):
        for f in sorted(glob.glob(os.path.join(campaign_dir, "*.npz"))):
            d = np.load(f, allow_pickle=True)
            meta = json.loads(str(d["meta"]))
            stage = "campaign_" + meta["method"]
            runs[stage] = dict(data={k: np.asarray(d[k]) for k in d.files if k != "meta"},
                               meta=meta, arm_meta=(meta["method"], 20.0, 0.02), path=f)
    return runs


def alanine_error_series(runs, ref_path):
    refd = np.load(ref_path, allow_pickle=True)
    F_ref = refd["F"]
    rmeta = json.load(open(os.path.join(os.path.dirname(ref_path), "meta.json")))
    kT, n_grid = float(rmeta["kT_kJ"]), int(rmeta["n_grid"])
    pack = build_masks(F_ref, kT)
    F_sm = smooth_reference(F_ref, 0.08, n_grid)
    w = pack["weights"]["equilibrium"]
    for st, run in runs.items():
        pmf = run["data"]["pmf"]
        T, R = pmf.shape[:2]
        e = np.zeros((T, R))
        for ti in range(T):
            for r in range(R):
                e[ti, r] = aligned_l2(pmf[ti, r], F_sm, w)
        run["e"] = e
        run["t"] = np.asarray(run["data"]["times"], float)
    return dict(reference=ref_path, param_hash=rmeta.get("param_hash"), kT_kJ=kT, n_grid=n_grid,
                error="kernel-matched (row-normalised K_h, h=0.08) equilibrium-weighted aligned L2")


def analyse_alanine(root, out, campaign_dir=CAMPAIGN_ALA, ref_path=REF_ALA):
    runs = load_alanine(root, campaign_dir)
    if "abf" not in runs:
        print("[alanine] no abf stage yet -- nothing to pair against")
        return None
    prov = alanine_error_series(runs, ref_path)
    abf = runs["abf"]
    t, T = abf["t"], float(abf["t"][-1])
    W1, W2 = (5.0, T), (20.0, T)
    N = int(abf["meta"]["n_replicas"])
    rows, series, summary = [], [], {}
    # baseline series
    series.append(dict(arm="abf", **{f"e_med@{tt:g}": float(np.median(abf["e"][i])) for i, tt in enumerate(t)}))
    for stage, run in runs.items():
        if stage == "abf":
            continue
        method, t_fr, rate = run["arm_meta"]
        if not np.allclose(run["t"], t):
            print(f"[alanine] {stage}: time axis differs from abf, skipped")
            continue
        seeds_ok = np.array_equal(run["data"]["seeds"], abf["data"]["seeds"])
        e = run["e"]
        rec = dict(arm=stage, method=method, fr_start_ps=t_fr, fr_rate=rate, seeds_match=seeds_ok,
                   n_seeds=int(e.shape[1]))
        own = (t_fr, T) if method != "abf" else W1
        for wname, (a0, a1) in (("W1", W1), ("W2", W2), ("own", own)):
            Ia, Ib = integrate(t, abf["e"], a0, a1), integrate(t, e, a0, a1)
            s = paired_stats(Ia, Ib)
            rec.update(flat_stats(f"dIF_{wname}", s))
            rec[f"{wname}_window"] = f"[{a0:g},{a1:g}]"
            if wname == "W1":
                prim = s
        fin = paired_stats(abf["e"][-1], e[-1])
        rec.update(flat_stats("final", fin))
        rec["abf_final_med"] = float(np.median(abf["e"][-1]))
        rec["arm_final_med"] = float(np.median(e[-1]))
        sp, e0 = speedups(t, abf["e"], e, T, W1[0])
        rec["e0_W1"] = e0
        for k, v in sp.items():
            rec[f"tau_{k}_abf"] = v["tau_abf"]
            rec[f"tau_{k}_arm"] = v["tau_arm"]
            rec[f"S_{k}"] = v["speedup"]
            rec[f"cens_{k}"] = f"{v['censored_abf']}/{v['censored_arm']}"
        # genealogy / dose on [t_fr, T]
        d = run["data"]
        sel = t >= t_fr - 1e-9
        if method in ("fr_uniform", "fr_oracle"):
            n_opp = max(int((int(run["meta"]["n_steps"]) - int(round(t_fr * 1000))) // int(run["meta"].get("fr_every", 500))) + 1, 1)
            ev = d["total_events"].astype(float)
            rec["ess_age_min"] = float(np.nanmin(d["ess_age"][sel]))
            rec["ess_perm_min"] = float(np.nanmin(d["ess_perm"][sel]))
            rec["wmax_max"] = float(np.nanmax(d["wmax"][sel]))
            rec["wmax_rare_max"] = float(np.nanmax(d["wmax_rare"][sel])) if "wmax_rare" in d else None
            rec["events_per_opp_med"] = float(np.median(ev)) / n_opp
            rec["event_frac_per_opp_max"] = float(ev.max()) / N / n_opp
            rec["event_frac_cum_med"] = float(np.median(ev)) / N
            rec["n_opportunities"] = n_opp
            floors_ok = (rec["ess_age_min"] >= FLOORS["ess"] and rec["wmax_max"] <= FLOORS["wmax"]
                         and rec["event_frac_per_opp_max"] < FLOORS["event_frac"])
        else:
            floors_ok = True
        rec["floors_ok"] = bool(floors_ok)
        # mechanism
        kl_a, kl_b = d["kl_uniform"], abf["data"]["kl_uniform"]
        rec["kl_uniform_final_arm"] = float(np.median(kl_a[-1]))
        rec["kl_uniform_final_abf"] = float(np.median(kl_b[-1]))
        rec["kl_uniform_min_arm"] = float(np.median(kl_a, 1).min())
        bf = d["basin_frac"]
        rec["c7ax_final_arm"] = float(np.median(bf[-1, :, 2]))
        rec["c7ax_final_abf"] = float(np.median(abf["data"]["basin_frac"][-1, :, 2]))
        rec["clip_fraction"] = float(run["meta"].get("clip_fraction", np.nan))
        rec["verdict"] = verdict(prim, fin, floors_ok) if method != "abf" else "baseline-replication"
        rows.append(rec)
        series.append(dict(arm=stage, **{f"e_med@{tt:g}": float(np.median(e[i])) for i, tt in enumerate(t)}))
        series.append(dict(arm=stage + ":ratio", **{f"e_med@{tt:g}": float(np.median(e[i] / abf["e"][i])) for i, tt in enumerate(t)}))
        series.append(dict(arm=stage + ":kl", **{f"e_med@{tt:g}": float(np.median(kl_a[i])) for i, tt in enumerate(t)}))
        series.append(dict(arm=stage + ":c7ax", **{f"e_med@{tt:g}": float(np.median(bf[i, :, 2])) for i, tt in enumerate(t)}))
    series.append(dict(arm="abf:kl", **{f"e_med@{tt:g}": float(np.median(abf["data"]["kl_uniform"][i])) for i, tt in enumerate(t)}))
    series.append(dict(arm="abf:c7ax", **{f"e_med@{tt:g}": float(np.median(abf["data"]["basin_frac"][i, :, 2])) for i, tt in enumerate(t)}))
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, "alanine_arms.csv"), rows)
    write_csv(os.path.join(out, "alanine_series.csv"), series)
    replication = {}
    for a_name, b_name, label in (("campaign_abf", "abf", "fresh abf vs campaign abf"),
                                  ("campaign_fr_uniform", "u02_t20", "fresh u02_t20 vs campaign fr_uniform"),
                                  ("campaign_abf", "campaign_fr_uniform", "closed verdict reproduced (campaign pair)")):
        if a_name in runs and b_name in runs and np.allclose(runs[a_name]["t"], t):
            ea, eb = runs[a_name]["e"], runs[b_name]["e"]
            replication[label] = dict(
                dIF_W2=paired_stats(integrate(t, ea, *W2), integrate(t, eb, *W2)),
                dIF_W1=paired_stats(integrate(t, ea, *W1), integrate(t, eb, *W1)),
                final=paired_stats(ea[-1], eb[-1]))
    summary = dict(system="alanine", root=root, provenance=prov, windows=dict(W1=W1, W2=W2),
                   replication=replication,
                   thresholds=dict(accel_median=ACCEL_MEDIAN, final_margin=FINAL_MARGIN,
                                   neutral_band=NEUTRAL_BAND, floors=FLOORS, persist=PERSIST_FRAC),
                   abf=dict(path=abf["path"], ms_per_step=abf["meta"].get("ms_per_step"),
                            wall_seconds=abf["meta"].get("wall_seconds"),
                            cuda_graph=abf["meta"].get("cuda_graph"), final_med=float(np.median(abf["e"][-1]))),
                   arms={r["arm"]: r for r in rows})
    json.dump(summary, open(os.path.join(out, "alanine_summary.json"), "w"), indent=2, default=float)
    plot_alanine(runs, abf, t, out)
    print_table("alanine", rows)
    for label, r in replication.items():
        print(f"  replication: {label:45s} W1 {100*r['dIF_W1']['median']:+6.2f}% [{100*r['dIF_W1']['lo']:+6.2f},{100*r['dIF_W1']['hi']:+6.2f}]  "
              f"W2 {100*r['dIF_W2']['median']:+6.2f}%  final {100*r['final']['median']:+6.2f}%")
    return summary


# --------------------------------------------------------------------------- R15
def load_r15(root, campaign_dir):
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, "raw", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        if "l2_F_t" not in d.files:
            continue
        spec = json.loads(str(d["spec_json"]))
        runs[(float(d["beta"]), str(d["name"]))] = dict(
            data={k: np.asarray(d[k]) for k in d.files if k not in ("per_seed", "spec_json")},
            per_seed=json.loads(str(d["per_seed"])), spec=spec, path=f,
            t=np.asarray(d["times"], float), e=np.asarray(d["l2_F_t"], float).T)   # (T, R)
    if campaign_dir and os.path.isdir(campaign_dir):
        for f in sorted(glob.glob(os.path.join(campaign_dir, "*.npz"))):
            d = np.load(f, allow_pickle=True)
            spec = json.loads(str(d["spec_json"]))
            runs[(float(d["beta"]), "campaign_" + str(d["name"]))] = dict(
                data={k: np.asarray(d[k]) for k in d.files if k not in ("per_seed", "spec_json")},
                per_seed=json.loads(str(d["per_seed"])), spec=spec, path=f,
                t=np.asarray(d["times"], float), e=np.asarray(d["l2_F_t"], float).T)
    return runs


def analyse_r15(root, out, campaign_dir=CAMPAIGN_R15):
    runs = load_r15(root, campaign_dir)
    betas = sorted({b for b, _ in runs if (b, "abf") in runs})
    if not betas:
        print("[r15] no abf baseline yet")
        return None
    rows, series, summary = [], [], {}
    for beta in betas:
        abf = runs[(beta, "abf")]
        t, T = abf["t"], float(abf["t"][-1])
        dt = float(abf["spec"]["dt"])
        W1, W2 = (5000 * dt, T), (0.0, T)
        N = int(abf["spec"]["n_replicas"])
        series.append(dict(cell=f"b{beta:g}", arm="abf", **{f"e_med@{tt:g}": float(np.median(abf["e"][i])) for i, tt in enumerate(t)}))
        for (b, name), run in sorted(runs.items()):
            if b != beta or name == "abf":
                continue
            method = run["spec"]["method"]
            t_fr = float(run["spec"]["fr_start_steps"]) * dt
            rate = float(run["spec"]["fr_rate"])
            # the closed arms were saved every 5000 steps: pair on the common times only
            common = np.isin(abf["t"], run["t"])
            if not common.all():
                ta = abf["t"][common]
                ea = abf["e"][common]
                if not np.allclose(ta, run["t"]):
                    print(f"[r15] b{beta:g} {name}: time axes incompatible, skipped")
                    continue
            else:
                ta, ea = abf["t"], abf["e"]
            e = run["e"]
            rec = dict(cell=f"b{beta:g}", arm=name, method=method, fr_start_steps=int(run["spec"]["fr_start_steps"]),
                       fr_start_tu=t_fr, fr_rate=rate, n_seeds=int(e.shape[1]),
                       seeds_match=bool(np.array_equal(run["data"]["seeds"], abf["data"]["seeds"])),
                       save_every=int(run["spec"]["save_every"]))
            own = (t_fr, T) if method != "abf" else W1
            for wname, (a0, a1) in (("W1", W1), ("W2", W2), ("own", own)):
                Ia, Ib = integrate(ta, ea, a0, a1), integrate(ta, e, a0, a1)
                s = paired_stats(Ia, Ib)
                rec.update(flat_stats(f"dIF_{wname}", s))
                rec[f"{wname}_window"] = f"[{a0:g},{a1:g}]"
                if wname == "W1":
                    prim = s
            fin = paired_stats(ea[-1], e[-1])
            rec.update(flat_stats("final", fin))
            rec["abf_final_med"] = float(np.median(ea[-1]))
            rec["arm_final_med"] = float(np.median(e[-1]))
            sp, e0 = speedups(ta, ea, e, T, W1[0])
            rec["e0_W1"] = e0
            for k, v in sp.items():
                rec[f"tau_{k}_abf"] = v["tau_abf"]
                rec[f"tau_{k}_arm"] = v["tau_arm"]
                rec[f"S_{k}"] = v["speedup"]
                rec[f"cens_{k}"] = f"{v['censored_abf']}/{v['censored_arm']}"
            ps = run["per_seed"]
            if method != "abf":
                ess = np.array([p["final_ancestor_ess"] for p in ps]) / N
                mess = np.array([p["min_ancestor_ess"] for p in ps]) / N
                wmx = np.array([p["final_max_ancestor_frac"] for p in ps])
                evf = np.array([p.get("fr_event_fraction", 0.0) for p in ps])
                rep = np.array([p["total_replacements"] for p in ps], float)
                rec["ess_final_med"] = float(np.median(ess))
                rec["ess_min_med"] = float(np.median(mess))
                rec["wmax_med"] = float(np.median(wmx))
                rec["event_frac_med"] = float(np.median(evf))
                rec["replacements_med"] = float(np.median(rep))
                rec["replacements_cum_frac_med"] = float(np.median(rep)) / N
                floors_ok = (rec["ess_final_med"] >= FLOORS["ess"] and rec["wmax_med"] <= FLOORS["wmax"]
                             and rec["event_frac_med"] < FLOORS["event_frac"])
            else:
                floors_ok = True
            rec["floors_ok"] = bool(floors_ok)
            # mechanism
            d = run["data"]
            if "series_kl_pq" in d and np.isfinite(d["series_kl_pq"]).any():
                rec["kl_pq_final_med"] = float(np.nanmedian(d["series_kl_pq"][-1]))
                rec["kl_pq_min_med"] = float(np.nanmin(np.nanmedian(d["series_kl_pq"], 1)))
            if "series_frac_compact" in d:
                rec["frac_compact_final_arm"] = float(np.median(d["series_frac_compact"][-1]))
            if "series_frac_compact" in abf["data"]:
                rec["frac_compact_final_abf"] = float(np.median(abf["data"]["series_frac_compact"][-1]))
            if "series_ancestor_ess" in d and method != "abf":
                rec["ess_series_min_med"] = float(np.nanmin(np.nanmedian(d["series_ancestor_ess"], 1))) / N
            rec["cond_tv_med_arm"] = float(np.median([p.get("dist_cond_tv_weighted", np.nan) for p in ps]))
            rec["cond_tv_med_abf"] = float(np.median([p.get("dist_cond_tv_weighted", np.nan) for p in abf["per_seed"]]))
            rec["low_support_med_arm"] = float(np.median([p.get("low_support_fraction", np.nan) for p in ps]))
            rec["wall_seconds"] = float(run["data"]["wall_seconds"])
            rec["verdict"] = verdict(prim, fin, floors_ok) if method != "abf" else "baseline-replication"
            rows.append(rec)
            series.append(dict(cell=f"b{beta:g}", arm=name, **{f"e_med@{tt:g}": float(np.median(e[i])) for i, tt in enumerate(run["t"])}))
            series.append(dict(cell=f"b{beta:g}", arm=name + ":ratio", **{f"e_med@{tt:g}": float(np.median(e[i] / ea[i])) for i, tt in enumerate(run["t"])}))
        summary[f"b{beta:g}"] = dict(abf=dict(path=abf["path"], wall_seconds=float(abf["data"]["wall_seconds"]),
                                             final_med=float(np.median(abf["e"][-1]))),
                                    windows=dict(W1=W1, W2=W2))
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, "r15_arms.csv"), rows)
    write_csv(os.path.join(out, "r15_series.csv"), series)
    summary = dict(system="r15", root=root, cells=summary,
                   thresholds=dict(accel_median=ACCEL_MEDIAN, final_margin=FINAL_MARGIN,
                                   neutral_band=NEUTRAL_BAND, floors=FLOORS, persist=PERSIST_FRAC),
                   error="thermal-window aligned interval L2 (jobs_cv.execute_dist l2_F_t)",
                   arms={f"{r['cell']}:{r['arm']}": r for r in rows})
    json.dump(summary, open(os.path.join(out, "r15_summary.json"), "w"), indent=2, default=float)
    plot_r15(runs, betas, out)
    print_table("r15", rows)
    return summary


# --------------------------------------------------------------------------- reporting
def print_table(system, rows):
    print("=" * 100)
    print(f"{system}: paired vs fresh ABF (median, BCa CI95, wins)   W1 = primary window")
    print("=" * 100)
    for r in rows:
        cell = r.get("cell", "")
        print(f"  {cell:5s} {r['arm']:22s} start={r.get('fr_start_ps', r.get('fr_start_tu', 0)):>5g} rate={r.get('fr_rate', 0):<5g} "
              f"dI_F(W1) {100*r['dIF_W1_median']:+6.2f}% [{100*r['dIF_W1_lo']:+6.2f},{100*r['dIF_W1_hi']:+6.2f}] "
              f"{int(round(r['dIF_W1_win']*r['dIF_W1_n']))}/{r['dIF_W1_n']}  "
              f"W2 {100*r['dIF_W2_median']:+6.2f}%  final {100*r['final_median']:+6.2f}% "
              f"[{100*r['final_lo']:+6.2f},{100*r['final_hi']:+6.2f}]  floors={'ok' if r['floors_ok'] else 'VIOLATED'}  "
              f"-> {r['verdict']}")


def _style():
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    try:
        from publication_style import apply_publication_style
        apply_publication_style()
    except Exception:                                   # noqa: BLE001
        pass
    return plt


def _colors(n):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    return [cmap(x) for x in np.linspace(0.05, 0.9, max(n, 2))]


def plot_alanine(runs, abf, t, out):
    plt = _style()
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    arms = [s for s in runs if s != "abf"]
    order = sorted(arms, key=lambda s: (runs[s]["arm_meta"][2], runs[s]["arm_meta"][1], s))
    cols = dict(zip(order, _colors(len(order))))
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), layout="constrained")
    ax = axes[0, 0]
    ax.plot(t, np.median(abf["e"], 1), color="k", lw=1.8, label="ABF (fresh)")
    for s in order:
        m, tf, rate = runs[s]["arm_meta"]
        ax.plot(t, np.median(runs[s]["e"], 1), color=cols[s], lw=1.1,
                label=f"{s} ({m.replace('fr_', '')} @{tf:g} ps, r={rate:g})")
        if not s.startswith("campaign"):
            ax.axvline(tf, color=cols[s], lw=0.6, ls=":")
    ax.set_yscale("log"); ax.set_xlabel("t (ps)"); ax.set_ylabel("kernel-matched aligned L2 (kJ/mol)")
    ax.set_title("alanine: FES error, median over 16 paired seeds", fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    ax = axes[0, 1]
    for s in order:
        ax.plot(t, np.median(runs[s]["e"] / abf["e"], 1), color=cols[s], lw=1.1, label=s)
    ax.axhline(1, color="k", lw=0.7, ls=":"); ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("t (ps)"); ax.set_ylabel("e_arm / e_abf (median of per-seed ratio)")
    ax.set_title("below 1 = arm ahead", fontsize=9); ax.legend(fontsize=6, frameon=False)
    ax = axes[1, 0]
    ax.plot(t, np.median(abf["data"]["kl_uniform"], 1), color="k", lw=1.6)
    for s in order:
        ax.plot(t, np.median(runs[s]["data"]["kl_uniform"], 1), color=cols[s], lw=1.0)
    ax.set_xlabel("t (ps)"); ax.set_ylabel("KL(p_t || uniform)"); ax.set_title("marginal vs the uniform target", fontsize=9)
    ax = axes[1, 1]
    ax.plot(t, np.median(abf["data"]["basin_frac"][:, :, 2], 1), color="k", lw=1.6)
    for s in order:
        ax.plot(t, np.median(runs[s]["data"]["basin_frac"][:, :, 2], 1), color=cols[s], lw=1.0)
    ax.set_yscale("log"); ax.set_xlabel("t (ps)"); ax.set_ylabel("C7ax fraction"); ax.set_title("rare-basin occupancy", fontsize=9)
    fig.savefig(os.path.join(fig_dir, "alanine_curves.png"), dpi=160); fig.savefig(os.path.join(fig_dir, "alanine_curves.pdf"))
    plt.close(fig)
    # forest plot of the primary endpoint
    rows = json.load(open(os.path.join(out, "alanine_summary.json")))["arms"] if os.path.exists(os.path.join(out, "alanine_summary.json")) else None
    if rows:
        fig, ax = plt.subplots(figsize=(6.0, 0.5 + 0.42 * len(rows)), layout="constrained")
        names = list(rows)
        for i, n in enumerate(names):
            r = rows[n]
            for j, (key, mk, lab) in enumerate((("dIF_W1", "o", "W1 [5,100] ps"), ("dIF_W2", "s", "W2 [20,100] ps"), ("final", "^", "final"))):
                y = i + (j - 1) * 0.22
                ax.errorbar(100 * r[f"{key}_median"], y, xerr=[[100 * (r[f"{key}_median"] - r[f"{key}_lo"])], [100 * (r[f"{key}_hi"] - r[f"{key}_median"])]],
                            fmt=mk, ms=4, color=["C0", "C1", "C2"][j], capsize=2, label=lab if i == 0 else None)
        ax.axvline(0, color="k", lw=0.7); ax.axvline(-10, color="gray", lw=0.6, ls="--")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("paired relative change vs fresh ABF (%)  [negative = better]")
        ax.legend(fontsize=7, frameon=False); ax.set_title("alanine: integrated / final error", fontsize=9)
        fig.savefig(os.path.join(fig_dir, "alanine_forest.png"), dpi=160); fig.savefig(os.path.join(fig_dir, "alanine_forest.pdf"))
        plt.close(fig)
    # genealogy
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9), layout="constrained")
    for s in order:
        d = runs[s]["data"]
        if runs[s]["arm_meta"][0] == "abf":
            continue
        axes[0].plot(t, np.median(d["ess_age"], 1), color=cols[s], lw=1.0, label=s)
        axes[1].plot(t, np.median(d["wmax"], 1), color=cols[s], lw=1.0)
        axes[2].plot(t, np.median(d["events_cum"], 1) / float(runs[s]["meta"]["n_replicas"]), color=cols[s], lw=1.0)
    axes[0].axhline(0.30, color="k", ls=":", lw=0.7); axes[0].set_ylabel("age-aware ESS / N"); axes[0].legend(fontsize=6, frameon=False)
    axes[1].axhline(0.05, color="k", ls=":", lw=0.7); axes[1].set_ylabel("max lineage share")
    axes[2].set_ylabel("cumulative events / N")
    for ax in axes:
        ax.set_xlabel("t (ps)")
    fig.savefig(os.path.join(fig_dir, "alanine_genealogy.png"), dpi=160)
    plt.close(fig)


def plot_r15(runs, betas, out):
    plt = _style()
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for beta in betas:
        abf = runs[(beta, "abf")]
        arms = [n for (b, n) in runs if b == beta and n != "abf"]
        order = sorted(arms, key=lambda n: (runs[(beta, n)]["spec"]["fr_rate"], runs[(beta, n)]["spec"]["fr_start_steps"], n))
        cols = dict(zip(order, _colors(len(order))))
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
        ax = axes[0]
        ax.plot(abf["t"], np.median(abf["e"], 1), color="k", lw=1.8, label="ABF (fresh)")
        for n in order:
            r = runs[(beta, n)]
            ax.plot(r["t"], np.median(r["e"], 1), color=cols[n], lw=1.0,
                    label=f"{n} (@{r['spec']['fr_start_steps']} r={r['spec']['fr_rate']:g})")
        ax.set_yscale("log"); ax.set_xlabel("t (time units)"); ax.set_ylabel("thermal-window L2(F)")
        ax.set_title(f"pentane R15 beta={beta:g}", fontsize=9); ax.legend(fontsize=6, frameon=False)
        ax = axes[1]
        for n in order:
            r = runs[(beta, n)]
            common = np.isin(abf["t"], r["t"])
            ax.plot(r["t"], np.median(r["e"] / abf["e"][common], 1), color=cols[n], lw=1.0, label=n)
        ax.axhline(1, color="k", lw=0.7, ls=":"); ax.set_ylim(0.8, 1.2)
        ax.set_xlabel("t"); ax.set_ylabel("e_arm / e_abf"); ax.set_title("below 1 = arm ahead", fontsize=9)
        ax = axes[2]
        if "series_frac_compact" in abf["data"]:
            ax.plot(abf["t"], np.median(abf["data"]["series_frac_compact"], 1), color="k", lw=1.6)
        for n in order:
            r = runs[(beta, n)]
            if "series_frac_compact" in r["data"]:
                ax.plot(r["t"], np.median(r["data"]["series_frac_compact"], 1), color=cols[n], lw=1.0)
        ax.set_xlabel("t"); ax.set_ylabel("compact-R fraction"); ax.set_title("occupancy of the compact region", fontsize=9)
        fig.savefig(os.path.join(fig_dir, f"r15_b{beta:g}_curves.png"), dpi=160); fig.savefig(os.path.join(fig_dir, f"r15_b{beta:g}_curves.pdf"))
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=("alanine", "r15", "both"), default="both")
    ap.add_argument("--alanine-root", default=os.path.join(ROOT, "results/fr_start_timing/alanine"))
    ap.add_argument("--r15-root", default=os.path.join(ROOT, "results/fr_start_timing/r15"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/fr_start_timing/analysis"))
    ap.add_argument("--no-campaign", action="store_true", help="skip the closed campaign arms")
    a = ap.parse_args()
    if a.system in ("alanine", "both"):
        analyse_alanine(a.alanine_root, a.out, campaign_dir=None if a.no_campaign else CAMPAIGN_ALA)
    if a.system in ("r15", "both"):
        analyse_r15(a.r15_root, a.out, campaign_dir=None if a.no_campaign else CAMPAIGN_R15)


if __name__ == "__main__":
    main()
