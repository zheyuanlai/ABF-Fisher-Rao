#!/usr/bin/env python
"""Score the ethane/LTA histogram-estimator study (docs/LTA_HISTOGRAM_REPLICATION.md).

Three questions, one script:

  1. REPLICATION  does the closed kernel temperature sweep's uniform-FR gain
                  (results/uniform_campaign/lta/sweep_summary.json) survive the swap
                  of the kernel ABF estimator for the textbook histogram (P0)
                  estimator at 80/150/225/300 K?
  2. EXTENSION    the new 350 K point (no kernel counterpart).
  3. ATTRIBUTION  the matched sham (same realised replacement counts per FR
                  opportunity, uniformly random deaths/births): FR vs sham is the
                  DIRECT contrast.

Endpoint conventions are the kernel sweep's, IMPORTED from analyze_uniform_lta.py:
e_F = additive-constant-aligned full-circle RMS per (save, seed) against the
umbrella/WHAM reference interpolated to the engine grid; I_F = trapezoid over saves;
paired per-seed relative change (%), median, 10 000-resample bootstrap CI of the
median (seed 20260829, +1 for the final endpoint), wins; tau to accuracy with
persistence 0.2 T; genealogy health = median across seeds of each seed's min
ancestor ESS/N over FR-active saves and of its max lineage share (worst seed
alongside, never substituted); the sweep's success_rule verdicts.  The kernel
sweep's d_int is recomputed from the kernel production files with this code and
asserted against sweep_summary.json (1e-6) so the two analyses are provably the
same arithmetic.

Inputs (any subset may exist; what is missing is reported, never imputed):
  <results-dir>/production_T{T}/{abf,fr_uniform,fr_sham}.npz
  <reference-dir>/reference_T{T}.npz
  <kernel-dir>/production_T{T}/{abf,fr_uniform}.npz, sweep_summary.json
  <results-dir>/calibration/width_ladder_T300/n{90,180,360}.npz
  <results-dir>/calibration/fr_rate_selection_T350.json
  <campaign>  configs/lta_histogram/campaign.json (per-key fallbacks if absent)

Outputs under <out-dir> (= <results-dir> by default): summary.json, scoreboard.md,
comparison_T{T}.csv, figures/fig_lta_hist_*.{pdf,png}.

    python scripts/analyze_lta_histogram.py                  # the real study
    python scripts/analyze_lta_histogram.py --kernel-only    # only reproduce the sweep's d_int
    python scripts/analyze_lta_histogram.py --results-dir <fixture> --campaign <fixture>/campaign.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPTS, ".."))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
from publication_style import PALETTE, apply_publication_style, save_figure  # noqa: E402
from analyze_uniform_lta import (BOOT_SEED, FRACTIONS, N_BOOT, PERSIST,  # noqa: E402,F401
                                 boot_median, circular_interp_ref, error_series, tau)

try:  # the engine's exact piecewise-linear reconstruction (torch); numpy mirror as fallback
    import torch
    from lta.core_lta import histogram_pmf as _histogram_pmf_torch

    def histogram_pmf_np(gamma, dphi):
        g = torch.as_tensor(np.asarray(gamma, dtype=np.float64))
        return _histogram_pmf_torch(g, float(dphi)).numpy()
    HIST_PMF_SOURCE = "src/lta/core_lta.py::histogram_pmf"
except Exception as _exc:  # pragma: no cover - only when torch/the engine cannot be imported
    def histogram_pmf_np(gamma, dphi):
        g = np.asarray(gamma, dtype=np.float64)
        g0 = g - g.mean(-1, keepdims=True)
        F_right = np.cumsum(g0 * dphi, axis=-1)
        F_c = F_right - 0.5 * g0 * dphi
        return F_c - F_c.mean(-1, keepdims=True)
    HIST_PMF_SOURCE = f"numpy mirror of histogram_pmf (engine import failed: {_exc})"

PI = math.pi
TWO_PI = 2.0 * PI
ARMS = ("abf", "fr_uniform", "fr_sham")
FR_ARMS = ("fr_uniform", "fr_sham")
KERNEL_ARMS = ("abf", "fr_uniform")
COLORS = {"abf": PALETTE["blue"], "fr_uniform": PALETTE["vermillion"], "fr_sham": PALETTE["gray"]}
LABELS = {"abf": "ABF", "fr_uniform": "ABF + uniform FR", "fr_sham": "ABF + matched sham"}
DEFAULT_TEMPS = [300, 150, 350, 225, 80]
DEFAULT_NGRIDS = [90, 180, 360]
DEFAULT_SUCCESS_RULE = dict(median_rel_change_pct_max=-10.0, ci95_upper_pct_max=0.0,
                            final_noninferiority_margin_pct=5.0, ess_anc_over_N_min=0.30,
                            wmax_max=0.05)
DEFAULT_REPL = dict(magnitude_int_points=10.0, magnitude_final_points=15.0,
                    absolute_accuracy_pct_max=25.0)
DEFAULT_WIDTH_LADDER = dict(n_grids=DEFAULT_NGRIDS, floor_share_max=0.20, plateau_pct=5.0,
                            temperature_K=300)
KL_PROBE_TIMES = (4.0, 12.0)
SWEEP_PREREG = os.path.join(ROOT, "configs/uniform_campaign/lta_sweep_prereg.json")
NEW_T = 350.0


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def rel(p):
    p = os.path.abspath(p)
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p


def tkey_of(T):
    return f"{float(T):g}"


def sanitize(o):
    """numpy -> python; non-finite floats -> None so summary.json is STRICT JSON."""
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.str_):
        return str(o)
    return o


def paired(arm, base, seed):
    """Paired per-seed relative change in % (median, bootstrap CI of the median, wins)."""
    arm = np.asarray(arm, dtype=float)
    base = np.asarray(base, dtype=float)
    d = 100.0 * (arm - base) / base
    lo, hi = boot_median(d, seed)
    return (dict(median=float(np.median(d)), ci95=[lo, hi], wins=int((d < 0).sum()),
                 n=int(len(d))), d)


def fmt(s):
    if s is None:
        return "n/a"
    return f"{s['median']:+.2f}% [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}] {s['wins']}/{s['n']}"


def ftime(x):
    if x is None:
        return "n/a"
    return "never" if not np.isfinite(x) else f"{x:.1f}"


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["_path"] = path
    try:
        d["_meta"] = json.loads(str(z["meta"])) if "meta" in z.files else {}
    except Exception:
        d["_meta"] = {}
    est = None
    if "abf_estimator" in z.files:
        est = str(z["abf_estimator"])
    elif "abf_estimator" in d["_meta"]:
        est = str(d["_meta"]["abf_estimator"])
    d["_estimator"] = est
    return d


# ---------------------------------------------------------------------------
# campaign configuration with graceful fallbacks
# ---------------------------------------------------------------------------
def load_campaign(path, notes):
    camp = {}
    if path and os.path.exists(path):
        camp = json.load(open(path))
        notes.append(f"campaign loaded from {rel(path)}")
    else:
        notes.append(f"campaign json NOT found at {path}: every key falls back "
                     f"(sweep prereg {rel(SWEEP_PREREG)} + built-in defaults)")
    sweep = json.load(open(SWEEP_PREREG)) if os.path.exists(SWEEP_PREREG) else {}

    def get(key, default, src="built-in default"):
        if key in camp:
            return camp[key]
        notes.append(f"campaign key '{key}' missing -> {src}")
        return default

    cfg = {}
    cfg["temperatures_K"] = [float(x) for x in get("temperatures_K", DEFAULT_TEMPS)]
    cfg["arms"] = [str(a) for a in get("arms", list(ARMS))]
    sampler = dict(sweep.get("sampler", {}))
    sampler.update(get("sampler", {"abf_estimator": "histogram", "n_grid": 180},
                       "sweep prereg sampler + abf_estimator=histogram"))
    sampler.setdefault("n_replicas", 1024)
    sampler.setdefault("fr_start_steps", 20000)
    sampler.setdefault("abf_estimator", "histogram")
    cfg["sampler"] = sampler
    per_T = {}
    for tk, v in sweep.get("per_T", {}).items():
        per_T[tk] = dict(rng_seed=v.get("rng_seed"), seeds_first=v.get("seeds_first"),
                         fr_rate=v.get("fr_rate"),
                         reference=f"results/uniform_campaign/lta/reference/reference_T{tk}.npz",
                         kernel_production=f"results/uniform_campaign/lta/production_T{tk}")
    for tk, v in get("per_T", {}, "sweep prereg per_T (kernel sweep temperatures only)").items():
        per_T.setdefault(str(tk), {}).update(v if isinstance(v, dict) else {})
    cfg["per_T"] = per_T
    cfg["seeds_count"] = int(get("seeds_count", 16))
    cfg["replication_rules_raw"] = get("replication_rules", {})
    cfg["replication_rules"] = resolve_replication_rules(cfg["replication_rules_raw"], notes)
    sr = dict(DEFAULT_SUCCESS_RULE)
    sr.update(sweep.get("success_rule", {}))
    sr.update(get("success_rule", {}, "sweep prereg success_rule"))
    cfg["success_rule"] = sr
    wl = dict(DEFAULT_WIDTH_LADDER)
    if "width_ladder" in camp:
        wl.update(camp["width_ladder"])
    elif "width_rule" in camp:
        wl.update(width_ladder_from_width_rule(camp["width_rule"], notes))
    else:
        notes.append("campaign keys 'width_ladder' / 'width_rule' missing -> built-in width-ladder defaults")
    cfg["width_ladder"] = wl
    if "predictions" in camp:
        cfg["predictions"] = camp["predictions"]
    elif "predictions_before_any_run" in camp:
        cfg["predictions"] = camp["predictions_before_any_run"]
    else:
        cfg["predictions"] = None
        notes.append("campaign key 'predictions' / 'predictions_before_any_run' missing -> none recorded")
    cfg["kernel_sampler"] = sweep.get("sampler", {})
    return cfg, camp


def width_ladder_from_width_rule(wr, notes):
    """The campaign writes the width rule as two prose clauses; parse the numbers out of them."""
    out = {}
    a = str(wr.get("clause_a_floor", ""))
    b = str(wr.get("clause_b_plateau", ""))
    m = re.search(r"<=\s*(\d+(?:\.\d+)?)", a)
    if m:
        out["floor_share_max"] = float(m.group(1))
    m = re.search(r"\{([\d,\s]+)\}", b)
    if m:
        out["n_grids"] = [int(x) for x in m.group(1).split(",") if x.strip()]
    m = re.search(r"<=\s*(\d+(?:\.\d+)?)\s*%", b)
    if m:
        out["plateau_pct"] = float(m.group(1))
    m = re.search(r"rng_seed\s*(\d+)", b)
    if m:
        out["rng_seed"] = int(m.group(1))
    m = re.search(r"(\d+)\s*K\b", b)
    if m:
        out["temperature_K"] = float(m.group(1))
    if "n_grid" in wr:
        out["selected_n_grid"] = int(wr["n_grid"])
    notes.append(f"width_ladder parsed from campaign 'width_rule' clauses: {out}")
    return out


def resolve_replication_rules(raw, notes):
    """Find the three numbers of the replication rule whatever the campaign calls them."""
    hits = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + [str(k)])
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            hits.append((".".join(path), float(o)))
    walk(raw if isinstance(raw, dict) else {}, [])

    def classify(key):
        k = key.lower().replace("i_f", "if").replace("e_f", "ef").replace("delta", "")
        parts = set(re.split(r"[^a-z0-9]+", k)) - {""}
        is_abs = bool(parts & {"abs", "absolute", "accuracy", "accur", "acc"})
        is_fin = bool(parts & {"fin", "final", "ef", "endpoint"})
        is_int = bool(parts & {"int", "integrated", "if", "dint"})
        return is_abs, is_fin, is_int

    # prose clauses ("... <= 10 points OR overlapping CIs; same for Delta e_F(T) with 15 points",
    # "... within +25 % of kernel ABF's ...")
    texts = {}

    def walk_text(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk_text(v, path + [str(k)])
        elif isinstance(o, str):
            texts[".".join(path)] = o
    walk_text(raw if isinstance(raw, dict) else {}, [])
    parsed = {}
    for key, s in texts.items():
        kl = key.lower()
        pts = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*points", s)]
        if ("replicat" in kl or "magnitude" in kl) and pts:
            parsed.setdefault("magnitude_int_points", (pts[0], f"text:{key}"))
            if len(pts) >= 2:
                parsed.setdefault("magnitude_final_points", (pts[1], f"text:{key}"))
        if "abs" in kl or "accur" in kl:
            m = re.search(r"\+\s*(\d+(?:\.\d+)?)\s*%", s)
            if m:
                parsed.setdefault("absolute_accuracy_pct_max", (float(m.group(1)), f"text:{key}"))

    out = {}
    for name, want in (("magnitude_int_points", "int"), ("magnitude_final_points", "fin"),
                       ("absolute_accuracy_pct_max", "abs")):
        found = parsed.get(name)
        for key, val in ([] if found else hits):
            is_abs, is_fin, is_int = classify(key)
            if want == "abs" and is_abs:
                found = (val, key)
            elif want == "fin" and is_fin and not is_abs:
                found = (val, key)
            elif want == "int" and is_int and not is_abs and not is_fin:
                found = (val, key)
            if found:
                break
        if found is None:
            out[name] = DEFAULT_REPL[name]
            out[name + "_source"] = "default"
            notes.append(f"replication rule '{name}' not found in campaign -> default {DEFAULT_REPL[name]}")
        else:
            out[name], out[name + "_source"] = found
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def validate_arms(arms, T, expect_estimator, notes):
    """times equal, seeds equal, meta.temperature_K == T, estimator as expected."""
    v = dict(ok=True, checks=[])
    names = list(arms)
    if not names:
        return v
    a0 = arms[names[0]]
    for a in names[1:]:
        z = arms[a]
        same_t = np.allclose(np.asarray(a0["times"], float), np.asarray(z["times"], float))
        same_s = np.array_equal(np.asarray(a0["seeds"]), np.asarray(z["seeds"]))
        v["checks"].append(dict(check=f"times {names[0]} == {a}", ok=bool(same_t)))
        v["checks"].append(dict(check=f"seeds {names[0]} == {a}", ok=bool(same_s)))
        assert same_t, f"T={T}: times differ between {names[0]} and {a}"
        assert same_s, f"T={T}: seeds differ between {names[0]} and {a}"
    for a in names:
        z = arms[a]
        mT = z["_meta"].get("temperature_K")
        ok_T = mT is not None and abs(float(mT) - float(T)) < 1e-9
        v["checks"].append(dict(check=f"{a}.meta.temperature_K == {T:g}", ok=bool(ok_T), value=mT))
        assert ok_T, f"{a}: meta.temperature_K {mT} != {T}"
        if expect_estimator is not None:
            est = z["_estimator"]
            if est is None:
                notes.append(f"T={T:g} {a}: no abf_estimator key in the file or its meta (cannot verify)")
                v["checks"].append(dict(check=f"{a}.abf_estimator == {expect_estimator}", ok=None))
            else:
                ok_e = est == expect_estimator
                v["checks"].append(dict(check=f"{a}.abf_estimator == {expect_estimator}", ok=bool(ok_e), value=est))
                assert ok_e, f"{a}: abf_estimator {est!r} != {expect_estimator!r}"
    v["ok"] = all(c["ok"] is not False for c in v["checks"])
    return v


def score_group(arms, ref, fr_start_steps, N, notes, tag):
    """Per-arm error series and endpoints on ONE reference (the kernel sweep's arithmetic)."""
    F_ref_cache = {}
    g = {}
    for a, z in arms.items():
        grid = np.asarray(z["grid"], float)
        key = (len(grid), float(grid[0]))
        if key not in F_ref_cache:
            F_ref_cache[key] = circular_interp_ref(np.asarray(ref["F"], float),
                                                   np.asarray(ref["grid_phi"], float), grid)
        F_ref = F_ref_cache[key]
        t = np.asarray(z["times"], dtype=float)
        err = error_series(np.asarray(z["pmf"], dtype=float), F_ref)      # (n_save, R)
        steps = np.asarray(z["steps"])
        R = err.shape[1]
        d = dict(t=t, steps=steps, err=err, I=np.trapezoid(err, t, axis=0), fin=err[-1],
                 curve=np.median(err, axis=1), R=R, F_ref=F_ref, grid=grid,
                 seeds=[int(s) for s in np.asarray(z["seeds"])],
                 rng_seed=z["_meta"].get("rng_seed"), fr_rate=z["_meta"].get("fr_rate"),
                 estimator=z["_estimator"], path=rel(z["_path"]))
        if "kl_uniform" in z:
            d["kl_curve"] = np.median(np.asarray(z["kl_uniform"], float), axis=1)
        if "total_replacement_events" in z:
            ev = np.asarray(z["total_replacement_events"])
            d["events_total"] = int(ev.sum())
            d["events_per_replica"] = float(ev.sum() / (R * N))
        if "n_cage_crossings" in z:
            cr = np.asarray(z["n_cage_crossings"])
            d["crossings_total"] = int(cr.sum())
            d["crossings_per_replica"] = float(cr.sum() / (R * N))
            d["crossings_per_seed"] = [int(c) for c in cr]
        if "u_counts" in z:                              # raw post-burn-in support (histogram arms)
            uc = np.asarray(z["u_counts"], float)
            d["support"] = dict(min_bin_count=float(uc.min()),
                                n_empty_bins_total=int((uc <= 0).sum()),
                                median_bin_count=float(np.median(uc)))
            if "fsum_prod" in z and d["estimator"] == "histogram":
                fs = np.asarray(z["fsum_prod"], float)
                gamma = np.where(uc > 0, fs / np.maximum(uc, 1.0), 0.0)
                dphi = TWO_PI / len(grid)
                F_raw = histogram_pmf_np(gamma, dphi)
                pmf_last = np.asarray(z["pmf"], float)[-1]
                d["raw_accumulators_reproduce_final_pmf_maxabs"] = float(np.abs(F_raw - pmf_last).max())
        if a in FR_ARMS and "ancestor_ess" in z:
            active = steps >= fr_start_steps
            if active.any():
                ess = np.asarray(z["ancestor_ess"], float)[active] / N
                wmax = np.asarray(z["max_ancestor_frac"], float)[active]
                ess_min = np.nanmin(ess, axis=0)
                wmax_max = np.nanmax(wmax, axis=0)
                d["health"] = dict(median_min_ess_frac=float(np.median(ess_min)),
                                   median_max_wmax=float(np.median(wmax_max)),
                                   worst_min_ess_frac=float(np.nanmin(ess_min)),
                                   worst_max_wmax=float(np.nanmax(wmax_max)),
                                   n_active_saves=int(active.sum()),
                                   convention="median across seeds (gateway rule); worst seed alongside")
                d["per_seed_min_ess"] = ess_min
                d["per_seed_max_wmax"] = wmax_max
            else:
                notes.append(f"{tag} {a}: no save at/after fr_start_steps={fr_start_steps}; no health")
        g[a] = d
    return g


def arm_public(d):
    """The JSON-facing view of a scored arm (no per-save arrays beyond the median curve)."""
    keys = ("R", "seeds", "rng_seed", "fr_rate", "estimator", "path", "events_total",
            "events_per_replica", "crossings_total", "crossings_per_replica", "crossings_per_seed",
            "support", "raw_accumulators_reproduce_final_pmf_maxabs", "health")
    out = {k: d[k] for k in keys if k in d}
    out["median_eF_T"] = float(np.median(d["fin"]))
    out["median_I_F"] = float(np.median(d["I"]))
    out["iqr_eF_T"] = [float(np.percentile(d["fin"], 25)), float(np.percentile(d["fin"], 75))]
    out["iqr_I_F"] = [float(np.percentile(d["I"], 25)), float(np.percentile(d["I"], 75))]
    out["curve_median_eF_t"] = [float(v) for v in d["curve"]]
    return out


def contrast(g, arm, base):
    c_int, d_int = paired(g[arm]["I"], g[base]["I"], BOOT_SEED)
    c_fin, d_fin = paired(g[arm]["fin"], g[base]["fin"], BOOT_SEED + 1)
    return dict(d_int=c_int, d_fin=c_fin), dict(d_int=d_int, d_fin=d_fin)


def tau_table(g, arms, t):
    """Time-to-accuracy on the median curves at the sweep's four thresholds (ABF-defined)."""
    curve_abf = g["abf"]["curve"]
    e0 = float(curve_abf[0])
    eps_list = {f"e0/{int(1 / f)}": e0 * f for f in FRACTIONS}
    eps_list["abf_final"] = float(curve_abf[-1])
    table = {}
    for name, eps in eps_list.items():
        row = dict(eps=eps, tau_abf=tau(t, curve_abf, eps, PERSIST))
        for a, suffix in (("fr_uniform", "uni"), ("fr_sham", "sham")):
            if a in arms:
                ta = row["tau_abf"]
                tu = tau(t, g[a]["curve"], eps, PERSIST)
                row[f"tau_{suffix}"] = tu
                row[f"speedup_{suffix}"] = (ta / tu if np.isfinite(ta) and np.isfinite(tu) and tu > 0
                                            else None)
                row[f"status_{suffix}"] = ("ok" if np.isfinite(ta) and np.isfinite(tu) else
                                           "abf_never" if np.isfinite(tu) else
                                           "arm_never" if np.isfinite(ta) else "neither")
        if "speedup_uni" in row:          # the sweep's names, for the scoreboard
            row["speedup"] = row["speedup_uni"]
            row["status"] = row["status_uni"]
        table[name] = row
    return table


def verdict_from_rule(c_int, c_fin, health, rule):
    med, hi = c_int["median"], c_int["ci95"][1]
    med_fin = c_fin["median"]
    health_ok = bool(health is not None
                     and health["median_min_ess_frac"] >= rule["ess_anc_over_N_min"]
                     and health["median_max_wmax"] <= rule["wmax_max"])
    accel = med <= rule["median_rel_change_pct_max"] and hi < rule["ci95_upper_pct_max"]
    safe = accel and med_fin <= rule["final_noninferiority_margin_pct"] and health_ok
    neutral = (abs(med) < abs(rule["median_rel_change_pct_max"])
               and med_fin <= rule["final_noninferiority_margin_pct"])
    verdict = ("SAFE_ACCELERATOR" if safe else "ACCELERATION_POSITIVE" if accel
               else "NEUTRAL" if neutral else "NEGATIVE_OR_UNSAFE")
    return verdict, health_ok


def replication_clauses(eff_h, eff_k, rules):
    """hist FR effect vs kernel FR effect: sign, CI-upper, magnitude (points or overlapping CIs)."""
    def overlap(a, b):
        return not (a[1] < b[0] or b[1] < a[0])
    same_sign = all(np.sign(eff_k[k]["median"]) == np.sign(eff_h[k]["median"])
                    for k in ("d_int", "d_fin"))
    ci_rule = (eff_h["d_int"]["ci95"][1] < 0) if eff_k["d_int"]["ci95"][1] < 0 else True
    gap_int = abs(eff_h["d_int"]["median"] - eff_k["d_int"]["median"])
    gap_fin = abs(eff_h["d_fin"]["median"] - eff_k["d_fin"]["median"])
    mag_int = gap_int <= rules["magnitude_int_points"] or overlap(eff_h["d_int"]["ci95"], eff_k["d_int"]["ci95"])
    mag_fin = gap_fin <= rules["magnitude_final_points"] or overlap(eff_h["d_fin"]["ci95"], eff_k["d_fin"]["ci95"])
    return dict(same_sign=bool(same_sign), ci_rule=bool(ci_rule),
                magnitude_int=bool(mag_int), magnitude_final=bool(mag_fin),
                gap_int_points=float(gap_int), gap_final_points=float(gap_fin),
                replicates=bool(same_sign and ci_rule and mag_int and mag_fin),
                rule=dict(ci_rule="kernel d_int CI upper < 0 implies histogram d_int CI upper < 0",
                          magnitude_int_points=rules["magnitude_int_points"],
                          magnitude_final_points=rules["magnitude_final_points"],
                          magnitude="|median gap| <= points OR overlapping 95% CIs"))


def absolute_accuracy(gh, gk, rules):
    """hist ABF vs kernel ABF: medians (+X % margin) and, when the seeds pair, per-seed paired ratio."""
    out = {}
    for key, label in (("fin", "eF_T"), ("I", "I_F")):
        mh, mk = float(np.median(gh["abf"][key])), float(np.median(gk["abf"][key]))
        out[label] = dict(hist_median=mh, kernel_median=mk, pct_of_medians=100.0 * (mh - mk) / mk)
    margin = rules["absolute_accuracy_pct_max"]
    out["margin_pct"] = margin
    out["ok"] = bool(out["eF_T"]["pct_of_medians"] <= margin and out["I_F"]["pct_of_medians"] <= margin)
    same_seeds = gh["abf"]["seeds"] == gk["abf"]["seeds"]
    same_rng = (gh["abf"]["rng_seed"] is not None and gh["abf"]["rng_seed"] == gk["abf"]["rng_seed"])
    same_R = len(gh["abf"]["I"]) == len(gk["abf"]["I"])
    out["paired_by_seed_labels"] = bool(same_seeds)
    out["same_rng_seed"] = bool(same_rng)
    out["same_R"] = bool(same_R)
    # the campaign pairs ACROSS estimators by rng_seed + replica index (same initial conditions and
    # Langevin noise stream); seed LABELS are fresh, so label equality is not required
    pairable = same_R and (same_rng or same_seeds)
    out["paired_by"] = ("seed labels" if same_seeds and same_R else
                        "rng_seed + replica index (same initial conditions and noise stream; labels differ)"
                        if pairable else None)
    if pairable:
        for arm in ("abf", "fr_uniform"):
            if arm in gh and arm in gk:
                c_fin, d_fin = paired(gh[arm]["fin"], gk[arm]["fin"], BOOT_SEED + 1)
                c_int, d_int = paired(gh[arm]["I"], gk[arm]["I"], BOOT_SEED)
                rat_fin = gh[arm]["fin"] / gk[arm]["fin"]
                rat_int = gh[arm]["I"] / gk[arm]["I"]
                out[f"paired_{arm}"] = dict(
                    eF_T_pct=c_fin, I_F_pct=c_int,
                    ratio_eF_T_median=float(np.median(rat_fin)),
                    ratio_eF_T_ci95=list(boot_median(rat_fin, BOOT_SEED + 1)),
                    ratio_I_F_median=float(np.median(rat_int)),
                    ratio_I_F_ci95=list(boot_median(rat_int, BOOT_SEED)))
    else:
        why = ("R differs" if not same_R else "neither rng_seed nor seed labels are shared")
        out["note"] = f"no per-seed pairing of histogram vs kernel ABF ({why}): medians only"
    return out


def establishment_diag(z, d, N):
    """ABF-only discovery / establishment diagnostics (histogram and kernel ABF alike)."""
    t, err, curve = d["t"], d["err"], d["curve"]
    R = err.shape[1]
    T_run = float(t[-1])
    out = dict(T_run=T_run)
    if "frac_window" in z:
        fw = np.asarray(z["frac_window"], float)
        first = [float(t[int(np.argmax(fw[:, r] > 0))]) if (fw[:, r] > 0).any() else float("inf")
                 for r in range(R)]
        first1 = [float(t[int(np.argmax(fw[:, r] >= 0.01))]) if (fw[:, r] >= 0.01).any() else float("inf")
                  for r in range(R)]
        out["t_first_window"] = float(np.median(first))
        out["t_first_window_n_never"] = int(sum(not np.isfinite(x) for x in first))
        out["t_first_frac_window_1pct"] = float(np.median(first1))
        out["t_first_frac_window_1pct_n_never"] = int(sum(not np.isfinite(x) for x in first1))
    e_fin = float(curve[-1])
    out["eF_final_median"] = e_fin
    out["t_to_1p5x_final"] = tau(t, curve, 1.5 * e_fin, PERSIST)
    out["t_to_2x_final"] = tau(t, curve, 2.0 * e_fin, PERSIST)
    out["tau_e0_8"] = tau(t, curve, float(curve[0]) / 8.0, PERSIST)

    def first_hit(eps):  # the campaign's wording "first comes within": no persistence
        below = np.nonzero(curve <= eps)[0]
        return float(t[below[0]]) if len(below) else float("inf")
    out["t_to_1p5x_final_first_hit"] = first_hit(1.5 * e_fin)
    out["t_to_2x_final_first_hit"] = first_hit(2.0 * e_fin)
    if "kl_uniform" in z:
        kl = np.asarray(z["kl_uniform"], float)
        out["kl_uniform_median_at"] = {}
        for tt in tuple(KL_PROBE_TIMES) + (T_run,):
            i = min(int(np.searchsorted(t, tt - 1e-9)), len(t) - 1)
            out["kl_uniform_median_at"][f"t={tt:g}"] = float(np.median(kl[i]))
    if "crossings_per_replica" in d:
        out["crossings_per_replica"] = d["crossings_per_replica"]
    T_hit = out.get("t_first_window", float("inf"))
    T_est = out["t_to_1p5x_final"]
    out["ratios_to_T_run"] = dict(T_hit=T_hit / T_run, T_est=T_est / T_run,
                                  t_to_2x_final=out["t_to_2x_final"] / T_run,
                                  tau_e0_8=out["tau_e0_8"] / T_run)
    out["screening_rule"] = dict(
        rule="T_hit/T_run < 0.1 and 0.25 < T_est/T_run < 0.75; T_hit = median t_first_window, "
             "T_est = time (median curve, persistence 0.2 T) to 1.5x the final error",
        T_hit=T_hit, T_est=T_est,
        hit_ok=bool(T_hit / T_run < 0.1),
        est_ok=bool(0.25 < T_est / T_run < 0.75),
        satisfied=bool(T_hit / T_run < 0.1 and 0.25 < T_est / T_run < 0.75),
        T_est_first_hit=out["t_to_1p5x_final_first_hit"],
        satisfied_first_hit=bool(T_hit / T_run < 0.1
                                 and 0.25 < out["t_to_1p5x_final_first_hit"] / T_run < 0.75))
    return out


def p0_floor(ref, n_grid):
    """Deterministic floor of the exact-integration P0 estimator at this bin width.

    Periodic cubic spline through the reference F on its 180 centres (extended by one
    period) -> exact per-bin average force (F(e_{j+1}) - F(e_j))/Delta -> histogram_pmf
    -> aligned RMS against the spline at that width's cell centres.
    """
    from scipy.interpolate import CubicSpline
    x = np.asarray(ref["grid_phi"], float)
    F = np.asarray(ref["F"], float)
    order = np.argsort(x)
    x, F = x[order], F[order]
    x_ext = np.concatenate([x, [x[0] + TWO_PI]])
    F_ext = np.concatenate([F, [F[0]]])
    S = CubicSpline(x_ext, F_ext, bc_type="periodic")

    def ev(u):
        return S(((np.asarray(u, float) - x[0]) % TWO_PI) + x[0])
    dphi = TWO_PI / n_grid
    edges = -PI + dphi * np.arange(n_grid + 1)
    centres = -PI + dphi * (np.arange(n_grid) + 0.5)
    Fe = ev(edges)
    gamma = (Fe[1:] - Fe[:-1]) / dphi
    F_hat = histogram_pmf_np(gamma[None, :], dphi)[0]
    dlt = F_hat - ev(centres)
    dlt = dlt - dlt.mean()
    return float(np.sqrt((dlt * dlt).mean()))


# ---------------------------------------------------------------------------
# per-temperature analysis
# ---------------------------------------------------------------------------
def analyze_T(T, cfg, args, notes, missing, sweep_rows):
    tk = tkey_of(T)
    pt = cfg["per_T"].get(tk, {})
    res = dict(temperature_K=float(T), per_T_config=pt)
    ref_path = pt.get("reference") or os.path.join(args.reference_dir, f"reference_T{tk}.npz")
    if not os.path.isabs(ref_path):
        cand = os.path.join(ROOT, ref_path)
        ref_path = cand if os.path.exists(cand) else os.path.join(args.reference_dir, os.path.basename(ref_path))
    if not os.path.exists(ref_path):
        missing.append(f"T={tk}: reference {rel(ref_path)}")
        res["status"] = "no reference: nothing scored"
        return res
    ref = np.load(ref_path, allow_pickle=True)
    beta = 1.0 / float(ref["kT"])
    res["reference"] = dict(path=rel(ref_path), kT=float(ref["kT"]),
                            temperature_K=float(ref["temperature"]),
                            dF_barrier_kT=float(ref["dF_barrier"]) * beta,
                            dU_barrier_kT=float(ref["dU_barrier"]) * beta,
                            mTdS_barrier_kT=float(ref["mTdS_barrier"]) * beta,
                            entropic_fraction=float(ref["mTdS_barrier"] / ref["dF_barrier"]))
    res["p0_floor_kJmol"] = {str(n): p0_floor(ref, int(n)) for n in cfg["width_ladder"]["n_grids"]}

    N = int(cfg["sampler"].get("n_replicas", 1024))
    fr_start = int(cfg["sampler"].get("fr_start_steps", 20000))
    N_k = int(cfg["kernel_sampler"].get("n_replicas", N))
    fr_start_k = int(cfg["kernel_sampler"].get("fr_start_steps", fr_start))

    # ---- histogram arms ----
    arms = {}
    if not args.kernel_only:
        for a in cfg["arms"]:
            p = os.path.join(args.results_dir, f"production_T{tk}", f"{a}.npz")
            if os.path.exists(p):
                arms[a] = load_npz(p)
            else:
                missing.append(f"T={tk}: histogram arm {a} ({rel(p)})")
    # ---- kernel arms ----
    karms = {}
    kdir = pt.get("kernel_production")
    if kdir is None and "kernel_production" not in pt:
        kdir = os.path.join(args.kernel_dir, f"production_T{tk}")
    if kdir:
        kdir_abs = kdir if os.path.isabs(kdir) else os.path.join(ROOT, kdir)
        if not os.path.isdir(kdir_abs):
            kdir_abs = os.path.join(args.kernel_dir, f"production_T{tk}")
        for a in KERNEL_ARMS:
            p = os.path.join(kdir_abs, f"{a}.npz")
            if os.path.exists(p):
                karms[a] = load_npz(p)
            else:
                missing.append(f"T={tk}: kernel arm {a} ({rel(p)})")
    else:
        notes.append(f"T={tk}: no kernel production declared (new temperature)")

    res["validation"] = dict(histogram=validate_arms(arms, T, cfg["sampler"].get("abf_estimator", "histogram"), notes),
                             kernel=validate_arms(karms, T, None, notes))
    gh = score_group(arms, ref, fr_start, N, notes, f"T={tk} hist") if arms else {}
    gk = score_group(karms, ref, fr_start_k, N_k, notes, f"T={tk} kernel") if karms else {}
    res["_hist"], res["_kernel"] = gh, gk
    res["arms"] = {a: arm_public(gh[a]) for a in gh}
    res["kernel_arms"] = {a: arm_public(gk[a]) for a in gk}
    if gh:
        t = gh[next(iter(gh))]["t"]
        res["times"] = [float(v) for v in t]
        steps = gh[next(iter(gh))]["steps"]
        res["fr_start_t"] = float(t[min(int(np.searchsorted(steps, fr_start)), len(t) - 1)])
        res["n_pairs"] = int(gh[next(iter(gh))]["R"])
    elif gk:
        t = gk[next(iter(gk))]["t"]
        steps = gk[next(iter(gk))]["steps"]
        res["fr_start_t"] = float(t[min(int(np.searchsorted(steps, fr_start_k)), len(t) - 1)])

    # ---- sham replay check ----
    if "fr_sham" in arms and "fr_uniform" in arms:
        es, eu = np.asarray(arms["fr_sham"]["event_counts"]), np.asarray(arms["fr_uniform"]["event_counts"])
        same = es.shape == eu.shape and np.array_equal(es, eu)
        res["sham_replay"] = dict(event_counts_equal=bool(same), shape_fr=list(eu.shape),
                                  shape_sham=list(es.shape), total_fr=int(eu.sum()), total_sham=int(es.sum()),
                                  total_replacement_events_fr=gh["fr_uniform"].get("events_total"),
                                  total_replacement_events_sham=gh["fr_sham"].get("events_total"))
        assert same, (f"T={tk}: fr_sham.event_counts != fr_uniform.event_counts "
                      f"(shapes {es.shape} vs {eu.shape}): not the matched sham")
    elif "fr_sham" not in arms and arms:
        notes.append(f"T={tk}: no fr_sham file -> sham contrasts and attribution skipped")

    # ---- paired contrasts ----
    res["contrasts"], res["_per_seed"] = {}, {}
    for arm, base in (("fr_uniform", "abf"), ("fr_sham", "abf"), ("fr_uniform", "fr_sham")):
        if arm in gh and base in gh:
            c, d = contrast(gh, arm, base)
            res["contrasts"][f"{arm}_vs_{base}"] = c
            res["_per_seed"][f"{arm}_vs_{base}"] = d
    if "abf" in gk and "fr_uniform" in gk:
        c, d = contrast(gk, "fr_uniform", "abf")
        res["contrasts"]["kernel_fr_uniform_vs_kernel_abf"] = c
        res["_per_seed"]["kernel_fr_uniform_vs_kernel_abf"] = d
        row = sweep_rows.get(float(T))
        rep = dict(recomputed_d_int=c["d_int"]["median"], recomputed_d_fin=c["d_fin"]["median"],
                   recomputed_d_int_ci=c["d_int"]["ci95"], recomputed_d_fin_ci=c["d_fin"]["ci95"],
                   recomputed_wins=c["d_int"]["wins"])
        if row is None:
            rep["status"] = "no sweep_summary row for this T"
        else:
            rep.update(sweep_d_int=row["d_int"], sweep_d_fin=row["d_fin"], sweep_d_int_ci=row["d_int_ci"],
                       sweep_d_fin_ci=row["d_fin_ci"], sweep_wins=row["wins"],
                       abs_diff_d_int=abs(c["d_int"]["median"] - row["d_int"]),
                       abs_diff_d_fin=abs(c["d_fin"]["median"] - row["d_fin"]),
                       abs_diff_ci=float(max(abs(a - b) for a, b in
                                             zip(c["d_int"]["ci95"] + c["d_fin"]["ci95"],
                                                 row["d_int_ci"] + row["d_fin_ci"]))))
            rep["ok_d_int"] = bool(rep["abs_diff_d_int"] <= args.kernel_tol)
            rep["ok_d_fin"] = bool(rep["abs_diff_d_fin"] <= args.kernel_tol)
            rep["ok_ci"] = bool(rep["abs_diff_ci"] <= args.kernel_tol)
            rep["status"] = "ok" if rep["ok_d_int"] and rep["ok_d_fin"] and rep["ok_ci"] else "MISMATCH"
            assert rep["ok_d_int"], (f"T={tk}: recomputed kernel d_int {c['d_int']['median']:.9f} != "
                                     f"sweep_summary {row['d_int']:.9f} (tol {args.kernel_tol})")
        res["kernel_reproduction"] = rep

    # ---- health / verdict / attribution ----
    rule = cfg["success_rule"]
    if "fr_uniform_vs_abf" in res["contrasts"]:
        c = res["contrasts"]["fr_uniform_vs_abf"]
        v, hok = verdict_from_rule(c["d_int"], c["d_fin"], gh["fr_uniform"].get("health"), rule)
        res["verdict_fr_uniform"] = v
        res["health_ok_fr_uniform"] = hok
    if "fr_sham_vs_abf" in res["contrasts"]:
        c = res["contrasts"]["fr_sham_vs_abf"]
        v, hok = verdict_from_rule(c["d_int"], c["d_fin"], gh["fr_sham"].get("health"), rule)
        res["verdict_fr_sham"] = v
        res["health_ok_fr_sham"] = hok
    if "fr_uniform_vs_fr_sham" in res["contrasts"]:
        c = res["contrasts"]["fr_uniform_vs_fr_sham"]
        beats = lambda s: (s["median"] < 0 and s["ci95"][1] < 0)  # noqa: E731
        res["attribution"] = dict(
            rule="FR beats sham if the DIRECT fr_uniform-vs-fr_sham contrast has median < 0 and 95% CI upper < 0 "
                 "(primary: Delta I_F; final reported alongside)",
            d_int_median=c["d_int"]["median"], d_int_ci_upper=c["d_int"]["ci95"][1],
            d_fin_median=c["d_fin"]["median"], d_fin_ci_upper=c["d_fin"]["ci95"][1],
            integrated="FR_BEATS_SHAM" if beats(c["d_int"]) else "NOT_DISTINGUISHED_FROM_SHAM",
            final="FR_BEATS_SHAM" if beats(c["d_fin"]) else "NOT_DISTINGUISHED_FROM_SHAM")
        if "fr_uniform_vs_abf" in res["contrasts"]:
            m_abf = res["contrasts"]["fr_uniform_vs_abf"]["d_int"]["median"]
            res["attribution"]["direct_over_vs_abf_margin"] = (c["d_int"]["median"] / m_abf if m_abf != 0 else None)

    # ---- replication / absolute accuracy against the kernel ----
    if "fr_uniform_vs_abf" in res["contrasts"] and "kernel_fr_uniform_vs_kernel_abf" in res["contrasts"]:
        res["replication"] = replication_clauses(res["contrasts"]["fr_uniform_vs_abf"],
                                                 res["contrasts"]["kernel_fr_uniform_vs_kernel_abf"],
                                                 cfg["replication_rules"])
    if "abf" in gh and "abf" in gk:
        res["absolute_accuracy"] = absolute_accuracy(gh, gk, cfg["replication_rules"])

    # ---- tau table ----
    if "abf" in gh:
        res["time_to_accuracy"] = tau_table(gh, list(gh), gh["abf"]["t"])

    # ---- establishment diagnostics (ABF arms) ----
    res["establishment"] = {}
    if "abf" in gh:
        res["establishment"]["histogram_abf"] = establishment_diag(arms["abf"], gh["abf"], N)
    if "abf" in gk:
        res["establishment"]["kernel_abf"] = establishment_diag(karms["abf"], gk["abf"], N_k)
    if "abf" in gh:
        n_sel = int(cfg["width_ladder"].get("selected_n_grid", cfg["sampler"].get("n_grid", 180)))
        f_sel = res["p0_floor_kJmol"].get(str(n_sel))
        if f_sel is None:
            f_sel = p0_floor(ref, n_sel)
            res["p0_floor_kJmol"][str(n_sel)] = f_sel
        share = f_sel / float(np.median(gh["abf"]["fin"]))
        share_max = float(cfg["width_ladder"].get("floor_share_max", 0.20))
        res["p0_floor_share_at_180"] = share if n_sel == 180 else None
        res["floor_clause"] = dict(
            n_grid=n_sel, floor_kJmol=f_sel, share=share, share_max=share_max, ok=bool(share <= share_max),
            rule="campaign width_rule clause a: deterministic P0 floor share <= share_max of the histogram ABF's "
                 "own median e_F(T); if it fails, the ladder's finer width is the reported width")
        if share > share_max:
            notes.append(f"T={tk}: FLOOR CLAUSE FAILS at n_grid {n_sel}: share {share:.3f} > {share_max}")
    res["status"] = ("scored" if gh else "kernel only" if gk else "no arms on disk")
    return res


# ---------------------------------------------------------------------------
# width ladder
# ---------------------------------------------------------------------------
def analyze_width_ladder(cfg, args, notes, missing):
    wl = cfg["width_ladder"]
    T = float(wl.get("temperature_K", 300))
    tk = tkey_of(T)
    n_grids = [int(n) for n in wl.get("n_grids", DEFAULT_NGRIDS)]
    ladder_dir = os.path.join(args.results_dir, "calibration", f"width_ladder_T{tk}")
    out = dict(temperature_K=T, dir=rel(ladder_dir), n_grids_requested=n_grids,
               floor_share_max=wl.get("floor_share_max", 0.20), plateau_pct=wl.get("plateau_pct", 5.0),
               plateau_interpretation="campaign width_rule clause b: paired |median % change| of BOTH e_F(T) and "
                                      "I_F from n to the next finer width <= plateau_pct (medians-only step "
                                      "change reported alongside)",
               rows={})
    ref_path = os.path.join(args.reference_dir, f"reference_T{tk}.npz")
    if not os.path.exists(ref_path):
        missing.append(f"width ladder: reference {rel(ref_path)}")
        return out
    ref = np.load(ref_path, allow_pickle=True)
    for n in n_grids:
        p = os.path.join(ladder_dir, f"n{n}.npz")
        if not os.path.exists(p):
            missing.append(f"width ladder n_grid={n} ({rel(p)})")
            continue
        z = load_npz(p)
        grid = np.asarray(z["grid"], float)
        if len(grid) != n:
            notes.append(f"width ladder: file n{n}.npz has {len(grid)} grid points (expected {n})")
        F_ref = circular_interp_ref(np.asarray(ref["F"], float), np.asarray(ref["grid_phi"], float), grid)
        t = np.asarray(z["times"], float)
        err = error_series(np.asarray(z["pmf"], float), F_ref)
        I = np.trapezoid(err, t, axis=0)
        fin = err[-1]
        floor = p0_floor(ref, len(grid))
        out["rows"][str(n)] = dict(
            n_grid=int(len(grid)), R=int(err.shape[1]), rng_seed=z["_meta"].get("rng_seed"),
            estimator=z["_estimator"], seeds=[int(s) for s in np.asarray(z["seeds"])],
            eF_T_median=float(np.median(fin)), I_F_median=float(np.median(I)),
            eF_T_per_seed=[float(v) for v in fin], I_F_per_seed=[float(v) for v in I],
            floor_kJmol=floor, floor_share=float(floor / np.median(fin)),
            floor_share_ok=bool(floor / np.median(fin) <= out["floor_share_max"]),
            curve_median_eF_t=[float(v) for v in np.median(err, axis=1)], times=[float(v) for v in t],
            _I=I, _fin=fin)
        if z["_estimator"] not in (None, "histogram"):
            notes.append(f"width ladder n{n}: abf_estimator={z['_estimator']!r} (expected histogram)")
    rows = out["rows"]
    if not rows:
        return out
    ns = sorted(int(k) for k in rows)
    base = 180 if "180" in rows else ns[0]
    out["baseline_n"] = base
    seeds_sets = {tuple(rows[str(n)]["seeds"]) for n in ns}
    rngs = {rows[str(n)]["rng_seed"] for n in ns}
    out["noise_paired"] = bool(len(seeds_sets) == 1 and len(rngs) == 1 and None not in rngs)
    for i, n in enumerate(ns):
        r = rows[str(n)]
        b = rows[str(base)]
        if n != base and len(r["_I"]) == len(b["_I"]):
            r["vs_baseline"] = dict(d_int=paired(r["_I"], b["_I"], BOOT_SEED)[0],
                                    d_fin=paired(r["_fin"], b["_fin"], BOOT_SEED + 1)[0])
        if i + 1 < len(ns):
            finer = rows[str(ns[i + 1])]
            step = 100.0 * (finer["eF_T_median"] - r["eF_T_median"]) / r["eF_T_median"]
            r["step_change_pct_to_next_finer_medians"] = float(step)
            if len(r["_I"]) == len(finer["_I"]):
                # campaign clause b: PAIRED |median change| of e_F(T) AND I_F from n to the finer width
                vf = dict(d_int=paired(finer["_I"], r["_I"], BOOT_SEED)[0],
                          d_fin=paired(finer["_fin"], r["_fin"], BOOT_SEED + 1)[0])
                r["vs_next_finer"] = vf
                r["plateau_ok"] = bool(abs(vf["d_int"]["median"]) <= out["plateau_pct"]
                                       and abs(vf["d_fin"]["median"]) <= out["plateau_pct"])
            else:
                r["plateau_ok"] = None
        else:
            r["step_change_pct_to_next_finer_medians"] = None
            r["plateau_ok"] = None
        finest = rows[str(ns[-1])]
        r["rel_to_finest_pct"] = float(100.0 * (r["eF_T_median"] - finest["eF_T_median"]) / finest["eF_T_median"])
    out["passing_both"] = [n for n in ns if rows[str(n)]["floor_share_ok"] and rows[str(n)]["plateau_ok"]]
    for n in ns:
        rows[str(n)].pop("_I", None)
        rows[str(n)].pop("_fin", None)
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def guarded(fn, name, errors, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as exc:  # a figure must never take the scoring down
        errors.append(dict(figure=name, error=f"{type(exc).__name__}: {exc}",
                           trace=traceback.format_exc().splitlines()[-3:]))
        print(f"  WARNING figure {name} skipped: {type(exc).__name__}: {exc}")
        plt.close("all")
        return None


def fig_convergence(T, res, fig_dir):
    gh, gk = res["_hist"], res["_kernel"]
    if not gh and not gk:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
    for a in ARMS:
        if a in gh:
            ax.plot(gh[a]["t"], gh[a]["curve"], color=COLORS[a], lw=1.4, label=f"{LABELS[a]} (histogram)")
    for a in KERNEL_ARMS:
        if a in gk:
            ax.plot(gk[a]["t"], gk[a]["curve"], color=COLORS[a], lw=1.1, ls=":", label=f"{LABELS[a]} (kernel)")
    if "fr_start_t" in res:
        ax.axvline(res["fr_start_t"], color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("t (BD units)")
    ax.set_ylabel(r"median $e_F(t)$ (kJ/mol)")
    n = res.get("n_pairs", "")
    ax.set_title(f"Ethane/LTA {T:g} K: histogram (solid) vs kernel (dotted), {n} seeds", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)
    save_figure(fig, os.path.join(fig_dir, f"fig_lta_hist_convergence_T{tkey_of(T)}"))


def fig_benefit_vs_T(per_T, sweep_rows, fig_dir):
    Ts = sorted(T for T, r in per_T.items() if r.get("contrasts"))
    if not Ts:
        return
    series = [("fr_uniform_vs_abf", "histogram FR vs ABF", PALETTE["vermillion"], "o", True, "-"),
              ("kernel", "kernel FR vs ABF (sweep)", PALETTE["vermillion"], "o", False, "-"),
              ("fr_sham_vs_abf", "matched sham vs ABF", PALETTE["gray"], "s", True, "-"),
              ("fr_uniform_vs_fr_sham", "FR vs sham (direct)", PALETTE["black"], "D", True, "--")]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), layout="constrained")
    for ax, key, ylabel in ((axes[0], "d_int", r"$\Delta I_F$ (%)"),
                            (axes[1], "d_fin", r"$\Delta e_F(T)$ (%)")):
        for i, (name, lab, c, m, filled, ls) in enumerate(series):
            xs, ys, los, his = [], [], [], []
            for T in Ts:
                if name == "kernel":
                    row = sweep_rows.get(float(T))
                    if row is None:
                        continue
                    val, ci = row[key], row[key + "_ci"]
                else:
                    cc = per_T[T]["contrasts"].get(name)
                    if cc is None:
                        continue
                    val, ci = cc[key]["median"], cc[key]["ci95"]
                xs.append(T + (i - 1.5) * 3.0)
                ys.append(val)
                los.append(val - ci[0])
                his.append(ci[1] - val)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=np.array([los, his]), color=c, marker=m, ms=5, lw=1.2, ls=ls,
                        capsize=2.5, mfc=(c if filled else "white"), mec=c, label=lab)
        ax.axhline(0, color=PALETTE["black"], lw=0.8, ls=":")
        if NEW_T in [float(T) for T in Ts]:
            ax.axvspan(NEW_T - 12, NEW_T + 12, color=PALETTE["yellow"], alpha=0.35, lw=0)
            ax.annotate("new\n350 K", (NEW_T, ax.get_ylim()[1]), xytext=(0, -2), textcoords="offset points",
                        ha="center", va="top", fontsize=7)
        ax.set_xlabel("T (K)")
        ax.set_ylabel(ylabel + "  [negative = arm better]")
    axes[0].legend(frameon=False, fontsize=6.5, loc="best")
    fig.suptitle("Ethane/LTA uniform-FR benefit vs T: histogram estimator, kernel sweep, matched sham",
                 fontsize=9)
    save_figure(fig, os.path.join(fig_dir, "fig_lta_hist_benefit_vs_T"))


def fig_profiles(T, res, arms_raw, ref, fig_dir):
    gh = res["_hist"]
    if not gh:
        return
    a0 = gh[next(iter(gh))]
    grid, F_ref = a0["grid"], a0["F_ref"]
    kT = float(ref["kT"])
    a_ps = float(arms_raw[next(iter(arms_raw))]["a_pseudo"]) if "a_pseudo" in arms_raw[next(iter(arms_raw))] else 11.919
    z = grid * a_ps / TWO_PI
    Fp_ref = np.gradient(F_ref, grid)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6), layout="constrained")
    for a in ARMS:
        if a not in gh:
            continue
        raw = arms_raw[a]
        F = np.median(np.asarray(raw["pmf"], float)[-1], axis=0)
        F = F - (F - F_ref).mean()
        axes[0].plot(z, F / kT, color=COLORS[a], lw=1.2, label=LABELS[a])
        if "mean_force" in raw:
            mf = np.median(np.asarray(raw["mean_force"], float)[-1], axis=0)
            axes[1].plot(z, mf, color=COLORS[a], lw=1.2)
        if "p_hat" in raw:
            p = np.median(np.asarray(raw["p_hat"], float)[-1], axis=0)
            axes[2].plot(z, p, color=COLORS[a], lw=1.2)
    axes[0].plot(z, F_ref / kT, color=PALETTE["black"], lw=0.9, ls=":", label="umbrella/WHAM ref")
    axes[1].plot(z, Fp_ref, color=PALETTE["black"], lw=0.9, ls=":")
    axes[2].axhline(1.0 / TWO_PI, color=PALETTE["black"], lw=0.9, ls=":", label="uniform")
    axes[0].set_ylabel(r"$\hat F_T(z)/k_BT$")
    axes[1].set_ylabel(r"$d\hat F_T/d\varphi$ (kJ/mol/rad)")
    axes[2].set_ylabel(r"final marginal $\hat p_T(\varphi)$")
    for ax in axes:
        ax.set_xlabel("z (A)")
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[2].legend(frameon=False, fontsize=6.5)
    fig.suptitle(f"Ethane/LTA {T:g} K, histogram estimator: final median profiles "
                 f"(window at z=0, cages at $\\pm$a/2)", fontsize=9)
    save_figure(fig, os.path.join(fig_dir, f"fig_lta_hist_profiles_T{tkey_of(T)}"))


def fig_establishment(per_T, fig_dir):
    Ts = sorted(T for T, r in per_T.items() if r.get("_hist") or r.get("_kernel"))
    if not Ts:
        return
    n = len(Ts)
    fig = plt.figure(figsize=(6.9, 6.0), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.3])
    top = outer[0].subgridspec(1, n)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.3, 1.0])
    first_ax = None
    for j, T in enumerate(Ts):
        ax = fig.add_subplot(top[0, j], sharey=first_ax)
        if first_ax is None:
            first_ax = ax
        res = per_T[T]
        for a in ARMS:
            d = res["_hist"].get(a)
            if d is not None and "kl_curve" in d:
                ax.plot(d["t"], d["kl_curve"], color=COLORS[a], lw=1.2, label=f"{LABELS[a]} (hist)")
        for a in KERNEL_ARMS:
            d = res["_kernel"].get(a)
            if d is not None and "kl_curve" in d:
                ax.plot(d["t"], d["kl_curve"], color=COLORS[a], lw=0.9, ls=":", label=f"{LABELS[a]} (kernel)")
        if "fr_start_t" in res:
            ax.axvline(res["fr_start_t"], color=PALETTE["gray"], lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"{T:g} K", fontsize=9)
        ax.set_xlabel("t")
        if j == 0:
            ax.set_ylabel(r"median $D_{KL}(\hat p_t\|{\rm uniform})$")
            ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    axL = fig.add_subplot(bot[0, 0])
    axR = fig.add_subplot(bot[0, 1])
    diag_keys = (("t_first_window", "o", r"$t_{\rm hit}$ (first window visit)"),
                 ("t_first_frac_window_1pct", "v", "t: 1% in window"),
                 ("t_to_1p5x_final", "s", r"$T_{\rm est}$ (1.5x final error)"),
                 ("t_to_2x_final", "D", "t: 2x final error"),
                 ("tau_e0_8", "^", r"$\tau(e_0/8)$"))
    cols = [PALETTE["blue"], PALETTE["sky"], PALETTE["vermillion"], PALETTE["orange"], PALETTE["green"]]
    T_run = None
    for est, filled in (("histogram_abf", True), ("kernel_abf", False)):
        for (key, m, lab), c in zip(diag_keys, cols):
            xs, ys = [], []
            for T in Ts:
                d = per_T[T].get("establishment", {}).get(est)
                if d is None or key not in d:
                    continue
                T_run = d["T_run"]
                v = d[key]
                xs.append(T)
                ys.append(v if np.isfinite(v) else np.nan)
            if xs:
                axL.plot(xs, ys, color=c, marker=m, ms=5, lw=0.8, ls=("-" if filled else ":"),
                         mfc=(c if filled else "white"), mec=c,
                         label=(lab if filled else None))
    if T_run:
        for frac, lab in ((0.1, "0.1"), (0.25, "0.25"), (0.75, "0.75")):
            axL.axhline(frac * T_run, color=PALETTE["gray"], lw=0.7, ls=":")
            axL.annotate(f"{lab} T_run", (1.0, frac * T_run), xycoords=("axes fraction", "data"),
                         ha="right", va="bottom", fontsize=6, color=PALETTE["gray"])
    axL.set_yscale("log")
    axL.set_xlabel("T (K)")
    axL.set_ylabel("time (BD units)")
    axL.set_title("ABF-only discovery / establishment (filled hist, open kernel)", fontsize=8)
    axL.legend(frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    width = 0.35
    xs = np.arange(len(Ts))
    for k, (est, c, lab) in enumerate((("histogram_abf", PALETTE["blue"], "hist ABF"),
                                       ("kernel_abf", PALETTE["sky"], "kernel ABF"))):
        vals = [per_T[T].get("establishment", {}).get(est, {}).get("crossings_per_replica", np.nan) for T in Ts]
        axR.bar(xs + (k - 0.5) * width, vals, width, color=c, label=lab)
    axR.set_xticks(xs, [f"{T:g}" for T in Ts])
    axR.set_xlabel("T (K)")
    axR.set_ylabel("ABF window crossings per replica")
    axR.set_yscale("log")
    axR.legend(frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    # the reference's entropic share as a label per temperature (no second y-axis)
    ef = [per_T[T].get("reference", {}).get("entropic_fraction", np.nan) for T in Ts]
    for x, e in zip(xs, ef):
        if np.isfinite(e):
            axR.text(x, axR.get_ylim()[1], f"{100 * e:.0f} %", ha="center", va="bottom", fontsize=6,
                     color=PALETTE["gray"])
    axR.set_title("entropic share of the barrier above each bar", fontsize=7, color=PALETTE["gray"], pad=10)
    fig.suptitle("Ethane/LTA establishment: marginal KL (top) and ABF-only timing diagnostics (bottom)",
                 fontsize=9)
    save_figure(fig, os.path.join(fig_dir, "fig_lta_hist_establishment"))


def fig_width_ladder(wl, fig_dir):
    rows = wl.get("rows", {})
    if not rows:
        return
    ns = sorted(int(k) for k in rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), layout="constrained")
    ax = axes[0]
    for n in ns:
        r = rows[str(n)]
        ax.plot([n] * len(r["eF_T_per_seed"]), r["eF_T_per_seed"], "o", color=PALETTE["blue"], ms=3, alpha=0.4)
    ax.plot(ns, [rows[str(n)]["eF_T_median"] for n in ns], color=PALETTE["blue"], marker="o", ms=5, lw=1.4,
            label="histogram ABF median $e_F(T)$")
    ax.plot(ns, [rows[str(n)]["floor_kJmol"] for n in ns], color=PALETTE["black"], marker="x", ms=5, lw=1.0,
            ls=":", label="deterministic P0 floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"$n_{\rm grid}$")
    ax.set_ylabel(r"$e_F(T)$ (kJ/mol)")
    ax.legend(frameon=False, fontsize=6.5)
    ax = axes[1]
    for n in ns:
        r = rows[str(n)]
        ax.plot([n] * len(r["I_F_per_seed"]), r["I_F_per_seed"], "o", color=PALETTE["blue"], ms=3, alpha=0.4)
    ax.plot(ns, [rows[str(n)]["I_F_median"] for n in ns], color=PALETTE["blue"], marker="o", ms=5, lw=1.4)
    ax.set_xscale("log")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"$n_{\rm grid}$")
    ax.set_ylabel(r"$I_F$ (kJ/mol $\cdot$ t)")
    fig.suptitle(f"Ethane/LTA {wl['temperature_K']:g} K width ladder (ABF only, histogram estimator, "
                 f"R={rows[str(ns[0])]['R']} noise-paired seeds)", fontsize=9)
    save_figure(fig, os.path.join(fig_dir, "fig_lta_hist_width_ladder"))


# ---------------------------------------------------------------------------
# the campaign's predictions, read numerically (None = not evaluable from what is on disk)
# ---------------------------------------------------------------------------
def check_predictions(per_T, cfg):
    def c(T, k):
        return per_T.get(T, {}).get("contrasts", {}).get(k)

    def all_or_none(vals):
        vals = [v for v in vals if v is not None]
        return bool(all(vals)) if vals else None

    out = {}
    # P1: histogram FR effect replicates the kernel sweep at the four kernel temperatures
    p1 = {}
    for T in (300.0, 150.0, 225.0, 80.0):
        h, k = c(T, "fr_uniform_vs_abf"), c(T, "kernel_fr_uniform_vs_kernel_abf")
        if h is None or k is None:
            continue
        row = dict(same_sign=bool(np.sign(h["d_int"]["median"]) == np.sign(k["d_int"]["median"])
                                  and np.sign(h["d_fin"]["median"]) == np.sign(k["d_fin"]["median"])),
                   gap_int_points=abs(h["d_int"]["median"] - k["d_int"]["median"]),
                   gap_fin_points=abs(h["d_fin"]["median"] - k["d_fin"]["median"]),
                   wins_int=h["d_int"]["wins"], n=h["d_int"]["n"],
                   cis_exclude_zero=bool(h["d_int"]["ci95"][1] < 0 and h["d_fin"]["ci95"][1] < 0))
        row["pass_"] = bool(row["same_sign"] and row["gap_int_points"] <= 10
                            and row["wins_int"] >= row["n"] - 1 and row["cis_exclude_zero"])
        p1[tkey_of(T)] = row
    out["P1_replication"] = dict(per_T=p1, pass_=all_or_none([v["pass_"] for v in p1.values()]),
                                 reading="same sign (I_F and e_F(T)); |gap Delta I_F| <= 10 points (final gap "
                                         "reported); wins >= n-1; both CIs exclude zero")
    # P2: histogram ABF baseline close to the kernel ABF
    p2 = {}
    for T, r in per_T.items():
        a = r.get("absolute_accuracy")
        if a:
            p2[tkey_of(T)] = dict(eF_T_pct=a["eF_T"]["pct_of_medians"], I_F_pct=a["I_F"]["pct_of_medians"],
                                  pass_=bool(abs(a["eF_T"]["pct_of_medians"]) <= 10
                                             and a["I_F"]["pct_of_medians"] <= 15))
    out["P2_baseline"] = dict(per_T=p2, pass_=all_or_none([v["pass_"] for v in p2.values()]),
                              reading="hist ABF median e_F(T) within +-10 % of kernel ABF's; I_F within +15 %")
    # P3: 350 K keeps shrinking toward neutral
    h = c(350.0, "fr_uniform_vs_abf")
    if h:
        v = per_T[350.0].get("verdict_fr_uniform")
        out["P3_350K"] = dict(d_int=h["d_int"]["median"], d_fin=h["d_fin"]["median"], verdict=v,
                              pass_=bool(-12 <= h["d_int"]["median"] <= 0 and abs(h["d_fin"]["median"]) <= 10
                                         and v in ("NEUTRAL", "ACCELERATION_POSITIVE")),
                              alternative_entropy_share_supported=bool(h["d_int"]["median"] < -15),
                              reading="Delta I_F in [-12, 0], |Delta e_F(T)| <= 10, verdict NEUTRAL or "
                                      "ACCELERATION_POSITIVE; the alternative reading if Delta I_F < -15")
    else:
        out["P3_350K"] = dict(pass_=None, reading="350 K not scored")
    # P4: the matched sham is neutral and FR beats it directly
    p4 = {}
    for T, r in per_T.items():
        s, d, f = c(T, "fr_sham_vs_abf"), c(T, "fr_uniform_vs_fr_sham"), c(T, "fr_uniform_vs_abf")
        if s is None or d is None or f is None:
            continue
        sham_neutral = abs(s["d_int"]["median"]) < 5 and 0 <= s["d_fin"]["median"] <= 10
        direct_ok = d["d_int"]["median"] <= (2.0 / 3.0) * f["d_int"]["median"] and d["d_int"]["ci95"][1] < 0
        p4[tkey_of(T)] = dict(sham_d_int=s["d_int"]["median"], sham_d_fin=s["d_fin"]["median"],
                              direct_d_int=d["d_int"]["median"], direct_ci_upper=d["d_int"]["ci95"][1],
                              fr_vs_abf_d_int=f["d_int"]["median"], sham_neutral=bool(sham_neutral),
                              direct_ok=bool(direct_ok), pass_=bool(sham_neutral and direct_ok))
    out["P4_sham"] = dict(per_T=p4, pass_=all_or_none([v["pass_"] for v in p4.values()]),
                          reading="sham |Delta I_F| < 5 and final in [0, +10]; direct FR-vs-sham Delta I_F <= 2/3 "
                                  "of FR-vs-ABF with CI upper < 0")
    # P5: the screening rule classifies 150-300 K on both estimators; the e0/8 convention does not
    p5 = {}
    for est in ("histogram_abf", "kernel_abf"):
        rows = {}
        for T, r in per_T.items():
            d = r.get("establishment", {}).get(est)
            if d:
                q = d["ratios_to_T_run"]["tau_e0_8"]
                rows[tkey_of(T)] = dict(satisfied=d["screening_rule"]["satisfied"], tau_e0_8_over_T_run=q,
                                        e0_8_in_window=bool(0.25 < q < 0.75))
        if rows:
            mid = [rows[k] for k in rows if float(k) in (150.0, 225.0, 300.0)]
            p5[est] = dict(per_T=rows,
                           pass_=(bool(all(x["satisfied"] and not x["e0_8_in_window"] for x in mid)) if mid else None))
    out["P5_screening_rule"] = dict(per_est=p5, pass_=all_or_none([v["pass_"] for v in p5.values()]),
                                    reading="rule satisfied at 150/225/300 K; tau(e0/8)/T_run outside (0.25, 0.75)")
    # P6: histogram FR fires +30-70 % more events than kernel FR and stays healthy at T >= 150 K
    p6 = {}
    for T, r in per_T.items():
        hf = r.get("arms", {}).get("fr_uniform")
        kf = r.get("kernel_arms", {}).get("fr_uniform")
        if hf is None:
            continue
        row = dict(hist_events=hf.get("events_total"), kernel_events=(kf or {}).get("events_total"),
                   median_min_ess=(hf.get("health") or {}).get("median_min_ess_frac"))
        if row["kernel_events"]:
            row["events_ratio_minus_1"] = row["hist_events"] / row["kernel_events"] - 1
            row["events_in_30_70"] = bool(0.30 <= row["events_ratio_minus_1"] <= 0.70)
        row["ess_ok"] = bool(row["median_min_ess"] >= 0.30) if row["median_min_ess"] is not None else None
        parts = [row.get("events_in_30_70"), (row["ess_ok"] if float(T) >= 150 else None)]
        row["pass_"] = all_or_none(parts)
        p6[tkey_of(T)] = row
    out["P6_health"] = dict(per_T=p6, pass_=all_or_none([v["pass_"] for v in p6.values()]),
                            reading="hist FR events +30-70 % vs kernel FR; median min ESS/N >= 0.30 at T >= 150 K "
                                    "(80 K reported, not scored)")
    return out


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def write_csv(T, res, path):
    gh = res["_hist"]
    if not gh:
        return False
    seeds = gh[next(iter(gh))]["seeds"]
    R = len(seeds)
    cols = ["seed"]
    for a in ARMS:
        cols += [f"int_{a}", f"final_{a}"]
    for k in ("fr_uniform_vs_abf", "fr_sham_vs_abf", "fr_uniform_vs_fr_sham"):
        cols += [f"d_int_pct_{k}", f"d_final_pct_{k}"]
    for a in FR_ARMS:
        cols += [f"min_ess_{a}", f"wmax_{a}"]
    lines = [",".join(cols)]
    for r in range(R):
        row = [str(seeds[r])]
        for a in ARMS:
            row += ([f"{gh[a]['I'][r]:.5f}", f"{gh[a]['fin'][r]:.6f}"] if a in gh else ["", ""])
        for k in ("fr_uniform_vs_abf", "fr_sham_vs_abf", "fr_uniform_vs_fr_sham"):
            d = res["_per_seed"].get(k)
            row += ([f"{d['d_int'][r]:.4f}", f"{d['d_fin'][r]:.4f}"] if d is not None else ["", ""])
        for a in FR_ARMS:
            if a in gh and "per_seed_min_ess" in gh[a]:
                row += [f"{gh[a]['per_seed_min_ess'][r]:.4f}", f"{gh[a]['per_seed_max_wmax'][r]:.5f}"]
            else:
                row += ["", ""]
        lines.append(",".join(row))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return True


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def write_scoreboard(summary, path):
    L = ["# Ethane/LTA histogram-estimator study: scoreboard", ""]
    L.append(f"Generated by scripts/analyze_lta_histogram.py; histogram_pmf from {summary['histogram_pmf_source']}.")
    L.append("Endpoints: kernel-sweep conventions (aligned full-circle RMS vs umbrella/WHAM reference; "
             "paired per-seed % change, median, 10k-bootstrap CI seed 20260829/+1, wins; tau persistence 0.2 T).")
    L.append("")
    if summary.get("missing_inputs"):
        L.append("## Missing inputs")
        L += [f"- {m}" for m in summary["missing_inputs"]]
        L.append("")
    if summary.get("notes"):
        L.append("## Notes / fallbacks")
        L += [f"- {m}" for m in summary["notes"]]
        L.append("")
    # headline table
    L.append("## Headline per temperature")
    L.append("| T (K) | entropic frac | hist FR vs ABF dI_F | hist FR vs ABF de_F(T) | kernel FR vs ABF dI_F | "
             "sham vs ABF dI_F | FR vs sham dI_F (direct) | verdict (hist FR) | attribution | replicates | abs. acc. ok |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for tk in sorted(summary["per_T"], key=float):
        r = summary["per_T"][tk]
        c = r.get("contrasts", {})
        rep = r.get("replication", {})
        acc = r.get("absolute_accuracy", {})
        L.append(f"| {tk} | {_f(r.get('reference', {}).get('entropic_fraction'), 3)} | "
                 f"{fmt(c.get('fr_uniform_vs_abf', {}).get('d_int'))} | {fmt(c.get('fr_uniform_vs_abf', {}).get('d_fin'))} | "
                 f"{fmt(c.get('kernel_fr_uniform_vs_kernel_abf', {}).get('d_int'))} | "
                 f"{fmt(c.get('fr_sham_vs_abf', {}).get('d_int'))} | {fmt(c.get('fr_uniform_vs_fr_sham', {}).get('d_int'))} | "
                 f"{r.get('verdict_fr_uniform', 'n/a')} | {r.get('attribution', {}).get('integrated', 'n/a')} | "
                 f"{rep.get('replicates', 'n/a')} | {acc.get('ok', 'n/a')} |")
    L.append("")
    for tk in sorted(summary["per_T"], key=float):
        r = summary["per_T"][tk]
        L.append(f"## T = {tk} K  ({r.get('status')})")
        ref = r.get("reference")
        if ref:
            L.append(f"Reference: dF = {ref['dF_barrier_kT']:.2f} kT (dU {ref['dU_barrier_kT']:.2f}, "
                     f"-TdS {ref['mTdS_barrier_kT']:.2f}; entropic fraction {100 * ref['entropic_fraction']:.0f}%); "
                     f"P0 floor (kJ/mol) at 90/180/360: "
                     + ", ".join(f"{k}: {v:.4f}" for k, v in r.get('p0_floor_kJmol', {}).items())
                     + (f"; floor share at 180 = {r['p0_floor_share_at_180']:.3f}" if 'p0_floor_share_at_180' in r else ""))
        if r.get("arms") or r.get("kernel_arms"):
            L.append("")
            L.append("| arm | median e_F(T) | median I_F | ESS/N (median min / worst) | wmax (median max / worst) | "
                     "events | crossings (total / per replica) | fr_rate |")
            L.append("|---|---|---|---|---|---|---|---|")
            for grp, pre in (("arms", "hist "), ("kernel_arms", "kernel ")):
                for a, d in r.get(grp, {}).items():
                    h = d.get("health")
                    L.append(f"| {pre}{a} | {d['median_eF_T']:.4f} | {d['median_I_F']:.3f} | "
                             f"{(_f(h['median_min_ess_frac']) + ' / ' + _f(h['worst_min_ess_frac'])) if h else 'n/a'} | "
                             f"{(_f(h['median_max_wmax'], 4) + ' / ' + _f(h['worst_max_wmax'], 4)) if h else 'n/a'} | "
                             f"{d.get('events_total', 'n/a')} | {d.get('crossings_total', 'n/a')} / "
                             f"{_f(d.get('crossings_per_replica'), 3)} | {d.get('fr_rate', 'n/a')} |")
        c = r.get("contrasts", {})
        if c:
            L.append("")
            L.append("| contrast | dI_F | de_F(T) |")
            L.append("|---|---|---|")
            for k, v in c.items():
                L.append(f"| {k} | {fmt(v['d_int'])} | {fmt(v['d_fin'])} |")
        if "sham_replay" in r:
            s = r["sham_replay"]
            L.append(f"\nSham replay: event_counts equal = {s['event_counts_equal']} "
                     f"(shape {s['shape_fr']}, total {s['total_fr']} vs {s['total_sham']}).")
        if "kernel_reproduction" in r:
            k = r["kernel_reproduction"]
            L.append(f"\nKernel sweep reproduction: d_int recomputed {k['recomputed_d_int']:+.6f} vs sweep "
                     f"{k.get('sweep_d_int', float('nan')):+.6f} -> {k['status']}"
                     + (f" (|diff| d_int {k['abs_diff_d_int']:.2e}, d_fin {k['abs_diff_d_fin']:.2e}, CI {k['abs_diff_ci']:.2e})"
                        if 'abs_diff_d_int' in k else ""))
        if "replication" in r:
            p = r["replication"]
            L.append(f"\nReplication clauses: same sign {p['same_sign']}, CI rule {p['ci_rule']}, magnitude I_F "
                     f"{p['magnitude_int']} (gap {p['gap_int_points']:.1f} pts), magnitude e_F(T) {p['magnitude_final']} "
                     f"(gap {p['gap_final_points']:.1f} pts) -> **replicates = {p['replicates']}**")
        if "absolute_accuracy" in r:
            a = r["absolute_accuracy"]
            L.append(f"\nAbsolute accuracy (hist ABF vs kernel ABF medians): e_F(T) {a['eF_T']['pct_of_medians']:+.1f}%, "
                     f"I_F {a['I_F']['pct_of_medians']:+.1f}% (margin +{a['margin_pct']:.0f}%) -> ok = {a['ok']}; "
                     f"paired by seed labels: {a['paired_by_seed_labels']}, same rng_seed: {a['same_rng_seed']}"
                     + (f"; paired ABF ratio hist/kernel e_F(T) {a['paired_abf']['ratio_eF_T_median']:.3f} "
                        f"[{a['paired_abf']['ratio_eF_T_ci95'][0]:.3f}, {a['paired_abf']['ratio_eF_T_ci95'][1]:.3f}], "
                        f"I_F {a['paired_abf']['ratio_I_F_median']:.3f} "
                        f"[{a['paired_abf']['ratio_I_F_ci95'][0]:.3f}, {a['paired_abf']['ratio_I_F_ci95'][1]:.3f}]"
                        if 'paired_abf' in a else ""))
        if "time_to_accuracy" in r:
            L.append("")
            L.append("| threshold | eps | tau ABF | tau FR | speedup FR | tau sham | speedup sham |")
            L.append("|---|---|---|---|---|---|---|")
            for name, sp in r["time_to_accuracy"].items():
                su = sp.get("speedup_uni")
                ss = sp.get("speedup_sham")
                L.append(f"| {name} | {sp['eps']:.4f} | {ftime(sp['tau_abf'])} | {ftime(sp.get('tau_uni'))} | "
                         f"{(f'{su:.2f}x' if su else sp.get('status_uni', 'n/a'))} | {ftime(sp.get('tau_sham'))} | "
                         f"{(f'{ss:.2f}x' if ss else sp.get('status_sham', 'n/a'))} |")
        est = r.get("establishment", {})
        if est:
            L.append("")
            L.append("| ABF arm | t_first_window | t 1% window | T_est (1.5x final) | t 2x final | tau(e0/8) | "
                     "KL_u(4) | KL_u(12) | KL_u(T) | crossings/replica | screen |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for name, d in est.items():
                kl = d.get("kl_uniform_median_at", {})
                klv = list(kl.values())
                L.append(f"| {name} | {ftime(d.get('t_first_window'))} | {ftime(d.get('t_first_frac_window_1pct'))} | "
                         f"{ftime(d['t_to_1p5x_final'])} | {ftime(d['t_to_2x_final'])} | {ftime(d['tau_e0_8'])} | "
                         f"{_f(klv[0] if len(klv) > 0 else None, 3)} | {_f(klv[1] if len(klv) > 1 else None, 3)} | "
                         f"{_f(klv[2] if len(klv) > 2 else None, 3)} | {_f(d.get('crossings_per_replica'), 3)} | "
                         f"{d['screening_rule']['satisfied']} |")
        if "verdict_fr_uniform" in r:
            L.append(f"\n**Verdict (hist fr_uniform): {r['verdict_fr_uniform']}** (health ok {r['health_ok_fr_uniform']})"
                     + (f"; sham: {r['verdict_fr_sham']}" if 'verdict_fr_sham' in r else "")
                     + (f"; attribution: integrated {r['attribution']['integrated']}, final {r['attribution']['final']}"
                        if 'attribution' in r else ""))
        L.append("")
    scr = summary.get("screening_rule_summary")
    if scr:
        L.append("## Screening rule (T_hit/T_run < 0.1 and 0.25 < T_est/T_run < 0.75)")
        for est, lst in scr.items():
            L.append(f"- {est}: satisfied at T = {lst['satisfied'] or 'none'}; not at T = {lst['not_satisfied'] or 'none'}")
        L.append("")
    wl = summary.get("width_ladder")
    if wl and wl.get("rows"):
        L.append(f"## Width ladder ({wl['temperature_K']:g} K, ABF only; noise-paired = {wl.get('noise_paired')})")
        L.append("| n_grid | R | median e_F(T) | median I_F | floor | floor share (<= "
                 f"{wl['floor_share_max']}) | to next finer: paired dI_F | paired de_F(T) | plateau (<= "
                 f"{wl['plateau_pct']} %) | vs n={wl.get('baseline_n')}: dI_F | de_F(T) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for n in sorted(wl["rows"], key=int):
            r = wl["rows"][n]
            vb = r.get("vs_baseline")
            vf = r.get("vs_next_finer")
            L.append(f"| {n} | {r['R']} | {r['eF_T_median']:.4f} | {r['I_F_median']:.3f} | {r['floor_kJmol']:.4f} | "
                     f"{r['floor_share']:.3f} ({r['floor_share_ok']}) | "
                     f"{fmt(vf['d_int']) if vf else 'finest'} | {fmt(vf['d_fin']) if vf else 'finest'} | "
                     f"{r['plateau_ok']} | "
                     f"{fmt(vb['d_int']) if vb else 'baseline'} | {fmt(vb['d_fin']) if vb else 'baseline'} |")
        L.append(f"\nWidths passing both clauses: {wl.get('passing_both')}. "
                 f"Plateau interpretation: {wl['plateau_interpretation']}.")
        L.append("")
    pc = summary.get("predictions_check")
    if pc:
        L.append("## Campaign predictions (numerical reading of predictions_before_any_run; None = not evaluable)")
        for k, v in pc.items():
            L.append(f"- {k}: pass = {v.get('pass_')}  ({v.get('reading', '')})")
        L.append("")
    if summary.get("fr_rate_selection_T350"):
        s = summary["fr_rate_selection_T350"]
        L.append(f"## 350 K FR-rate calibration (copied): selected {s.get('selected')}; ladder "
                 + ", ".join(f"{e.get('rate')}: ESS {e.get('ess_min', float('nan')):.3f}/wmax {e.get('wmax_max', float('nan')):.4f} ok={e.get('ok')}"
                             for e in s.get("ladder", []) if isinstance(e, dict)))
        L.append("")
    if summary.get("figure_errors"):
        L.append("## Figure errors")
        L += [f"- {e['figure']}: {e['error']}" for e in summary["figure_errors"]]
        L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L))


def console_report(summary):
    print("\n=== ethane/LTA histogram-estimator study: analysis ===")
    if summary["notes"]:
        print("notes:")
        for n_ in summary["notes"]:
            print(f"  - {n_}")
    if summary["missing_inputs"]:
        print(f"missing inputs ({len(summary['missing_inputs'])}):")
        for m in summary["missing_inputs"]:
            print(f"  - {m}")
    for tk in sorted(summary["per_T"], key=float):
        r = summary["per_T"][tk]
        print(f"\nT = {tk} K  [{r.get('status')}]")
        ref = r.get("reference")
        if ref:
            print(f"  reference dF {ref['dF_barrier_kT']:.2f} kT, entropic fraction {100 * ref['entropic_fraction']:.0f}%; "
                  f"P0 floor@180 {r.get('p0_floor_kJmol', {}).get('180', float('nan')):.4f} kJ/mol"
                  + (f" (share {r['p0_floor_share_at_180']:.3f})" if 'p0_floor_share_at_180' in r else ""))
        for grp, pre in (("arms", "hist"), ("kernel_arms", "kernel")):
            for a, d in r.get(grp, {}).items():
                h = d.get("health")
                print(f"  {pre:6s} {a:10s} e_F(T) {d['median_eF_T']:.4f}  I_F {d['median_I_F']:.3f}"
                      + (f"  ESS/N {h['median_min_ess_frac']:.3f} (worst {h['worst_min_ess_frac']:.3f})"
                         f" wmax {h['median_max_wmax']:.4f}" if h else "")
                      + (f"  events {d['events_total']}" if 'events_total' in d else "")
                      + (f"  crossings/rep {d['crossings_per_replica']:.3f}" if 'crossings_per_replica' in d else ""))
        for k, v in r.get("contrasts", {}).items():
            print(f"  {k:36s} dI_F {fmt(v['d_int'])}   de_F(T) {fmt(v['d_fin'])}")
        if "sham_replay" in r:
            print(f"  sham replay: event_counts equal = {r['sham_replay']['event_counts_equal']}")
        if "kernel_reproduction" in r:
            k = r["kernel_reproduction"]
            print(f"  kernel reproduction: {k['status']}"
                  + (f" (|diff| d_int {k['abs_diff_d_int']:.2e}, d_fin {k['abs_diff_d_fin']:.2e}, CI {k['abs_diff_ci']:.2e})"
                     if 'abs_diff_d_int' in k else ""))
        if "replication" in r:
            p = r["replication"]
            print(f"  replication: sign {p['same_sign']} CI {p['ci_rule']} magI {p['magnitude_int']} "
                  f"magF {p['magnitude_final']} -> replicates = {p['replicates']}")
        if "absolute_accuracy" in r:
            a = r["absolute_accuracy"]
            print(f"  absolute accuracy: e_F(T) {a['eF_T']['pct_of_medians']:+.1f}%  I_F {a['I_F']['pct_of_medians']:+.1f}% "
                  f"-> ok = {a['ok']}"
                  + (f"; paired ratio e_F(T) {a['paired_abf']['ratio_eF_T_median']:.3f}, I_F {a['paired_abf']['ratio_I_F_median']:.3f}"
                     if 'paired_abf' in a else ""))
        for name, sp in r.get("time_to_accuracy", {}).items():
            s = f"{sp['speedup_uni']:.2f}x" if sp.get("speedup_uni") else sp.get("status_uni", "n/a")
            ss = (f"{sp['speedup_sham']:.2f}x" if sp.get("speedup_sham") else sp.get("status_sham", "")) if "tau_sham" in sp else ""
            print(f"  tau[{name}]: abf {ftime(sp['tau_abf'])} fr {ftime(sp.get('tau_uni'))} -> {s}"
                  + (f"; sham {ftime(sp.get('tau_sham'))} -> {ss}" if ss else ""))
        for name, d in r.get("establishment", {}).items():
            sr = d["screening_rule"]
            print(f"  {name}: T_hit {ftime(d.get('t_first_window'))} T_est {ftime(d['t_to_1p5x_final'])} "
                  f"(2x {ftime(d['t_to_2x_final'])}, tau e0/8 {ftime(d['tau_e0_8'])}) of T_run {d['T_run']:g}; "
                  f"screen {sr['satisfied']}")
        if "verdict_fr_uniform" in r:
            print(f"  VERDICT hist fr_uniform: {r['verdict_fr_uniform']}"
                  + (f"; sham {r['verdict_fr_sham']}" if 'verdict_fr_sham' in r else "")
                  + (f"; attribution {r['attribution']['integrated']} (final {r['attribution']['final']})"
                     if 'attribution' in r else ""))
    scr = summary.get("screening_rule_summary")
    if scr:
        for est, lst in scr.items():
            print(f"screening rule [{est}]: satisfied at T = {lst['satisfied'] or 'none'}")
    wl = summary.get("width_ladder")
    if wl and wl.get("rows"):
        print(f"width ladder {wl['temperature_K']:g} K: "
              + "; ".join(f"n{n}: e_F(T) {wl['rows'][n]['eF_T_median']:.4f} floor {wl['rows'][n]['floor_kJmol']:.4f} "
                          f"share {wl['rows'][n]['floor_share']:.3f}" for n in sorted(wl['rows'], key=int))
              + f"; passing both: {wl.get('passing_both')}")
    for k, v in (summary.get("predictions_check") or {}).items():
        print(f"prediction {k}: pass = {v.get('pass_')}")
    if summary.get("figure_errors"):
        print(f"figure errors: {len(summary['figure_errors'])}")
    print(f"wrote {summary['outputs']['summary']} , {summary['outputs']['scoreboard']}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results/lta_histogram"))
    ap.add_argument("--reference-dir", default=os.path.join(ROOT, "results/uniform_campaign/lta/reference"))
    ap.add_argument("--kernel-dir", default=os.path.join(ROOT, "results/uniform_campaign/lta"),
                    help="holds production_T{T}/ of the kernel sweep and sweep_summary.json")
    ap.add_argument("--campaign", default=os.path.join(ROOT, "configs/lta_histogram/campaign.json"))
    ap.add_argument("--out-dir", default=None, help="default: --results-dir")
    ap.add_argument("--temperatures", type=float, nargs="*", default=None,
                    help="override the campaign's temperature list")
    ap.add_argument("--kernel-only", action="store_true",
                    help="only recompute the kernel sweep's contrasts and assert against sweep_summary.json")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--kernel-tol", type=float, default=1e-6)
    args = ap.parse_args()
    args.results_dir = os.path.abspath(args.results_dir)
    args.reference_dir = os.path.abspath(args.reference_dir)
    args.kernel_dir = os.path.abspath(args.kernel_dir)
    out_dir = os.path.abspath(args.out_dir or args.results_dir)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    notes, missing = [], []
    cfg, camp_raw = load_campaign(args.campaign, notes)
    temps = [float(T) for T in (args.temperatures if args.temperatures else cfg["temperatures_K"])]

    # auto kernel-only when no histogram production file exists at all
    any_hist = any(os.path.exists(os.path.join(args.results_dir, f"production_T{tkey_of(T)}", f"{a}.npz"))
                   for T in temps for a in cfg["arms"])
    if not any_hist and not args.kernel_only:
        notes.append("no histogram production file found for any temperature -> kernel-only mode")
        args.kernel_only = True

    sweep_rows = {}
    sp = os.path.join(args.kernel_dir, "sweep_summary.json")
    if os.path.exists(sp):
        sweep_rows = {float(r["T"]): r for r in json.load(open(sp))["rows"]}
    else:
        missing.append(f"kernel sweep summary {rel(sp)} (no kernel reproduction assert possible)")

    per_T, raw_by_T, refs = {}, {}, {}
    for T in temps:
        res = analyze_T(T, cfg, args, notes, missing, sweep_rows)
        per_T[T] = res
        if "reference" in res:
            refs[T] = np.load(os.path.join(ROOT, res["reference"]["path"]) if not os.path.isabs(res["reference"]["path"])
                              else res["reference"]["path"], allow_pickle=True)
        # keep the raw arrays of the histogram arms for the figures
        raw = {}
        for a in cfg["arms"]:
            p = os.path.join(args.results_dir, f"production_T{tkey_of(T)}", f"{a}.npz")
            if os.path.exists(p) and not args.kernel_only:
                raw[a] = load_npz(p)
        raw_by_T[T] = raw

    # screening-rule roll-up
    scr = {}
    for est in ("histogram_abf", "kernel_abf"):
        sat, nsat = [], []
        for T in temps:
            d = per_T[T].get("establishment", {}).get(est)
            if d is None:
                continue
            (sat if d["screening_rule"]["satisfied"] else nsat).append(f"{T:g}")
        if sat or nsat:
            scr[est] = dict(satisfied=sat, not_satisfied=nsat)

    wl = {} if args.kernel_only else analyze_width_ladder(cfg, args, notes, missing)
    pred = {} if args.kernel_only else check_predictions(per_T, cfg)

    sel350 = None
    p350 = os.path.join(args.results_dir, "calibration", "fr_rate_selection_T350.json")
    if os.path.exists(p350):
        sel350 = json.load(open(p350))
        declared = cfg["per_T"].get("350", {}).get("fr_rate")
        if declared is not None and sel350.get("selected") is not None and declared != sel350["selected"]:
            notes.append(f"350 K: campaign per_T fr_rate {declared} != calibration selection {sel350['selected']}")
    elif not args.kernel_only:
        missing.append(f"350 K FR-rate calibration {rel(p350)}")

    # ---- CSV + figures ----
    figure_errors = []
    for T in temps:
        if per_T[T].get("_hist"):
            write_csv(T, per_T[T], os.path.join(out_dir, f"comparison_T{tkey_of(T)}.csv"))
    if not args.no_figures and not args.kernel_only:
        apply_publication_style()
        os.makedirs(fig_dir, exist_ok=True)
        for T in temps:
            if per_T[T].get("_hist") or per_T[T].get("_kernel"):
                guarded(fig_convergence, f"convergence_T{tkey_of(T)}", figure_errors, T, per_T[T], fig_dir)
            if per_T[T].get("_hist") and T in refs:
                guarded(fig_profiles, f"profiles_T{tkey_of(T)}", figure_errors, T, per_T[T], raw_by_T[T], refs[T], fig_dir)
        guarded(fig_benefit_vs_T, "benefit_vs_T", figure_errors, per_T, sweep_rows, fig_dir)
        guarded(fig_establishment, "establishment", figure_errors, per_T, fig_dir)
        if wl.get("rows"):
            guarded(fig_width_ladder, "width_ladder", figure_errors, wl, fig_dir)

    # ---- summary ----
    public_per_T = {}
    for T, r in per_T.items():
        public_per_T[tkey_of(T)] = {k: v for k, v in r.items() if not k.startswith("_")}
    summary = dict(
        study="ethane/LTA histogram-estimator replication + 350 K extension + matched sham",
        mode="kernel-only" if args.kernel_only else "full",
        campaign=dict(path=rel(args.campaign), loaded=bool(camp_raw), temperatures_K=temps, arms=cfg["arms"],
                      sampler=cfg["sampler"], success_rule=cfg["success_rule"],
                      replication_rules=cfg["replication_rules"], replication_rules_raw=cfg["replication_rules_raw"],
                      width_ladder=cfg["width_ladder"], seeds_count=cfg["seeds_count"],
                      predictions=cfg["predictions"]),
        inputs=dict(results_dir=rel(args.results_dir), reference_dir=rel(args.reference_dir),
                    kernel_dir=rel(args.kernel_dir), out_dir=rel(out_dir)),
        conventions=dict(error="additive-constant-aligned full-circle RMS per (save, seed) vs the umbrella/WHAM "
                               "reference interpolated to the engine grid (kJ/mol)",
                         integrated="trapezoid over saves", bootstrap=dict(n=N_BOOT, seed_int=BOOT_SEED, seed_final=BOOT_SEED + 1),
                         tau_persistence=PERSIST, health="median across seeds of per-seed min ESS/N and max lineage "
                                                           "share over FR-active saves; worst seed alongside",
                         p0_floor="periodic CubicSpline of the reference F -> exact bin-average force -> "
                                  "histogram_pmf -> aligned RMS at the cell centres"),
        histogram_pmf_source=HIST_PMF_SOURCE,
        per_T=public_per_T,
        kernel_reproduction={tk: r.get("kernel_reproduction") for tk, r in public_per_T.items() if r.get("kernel_reproduction")},
        screening_rule_summary=scr,
        width_ladder=wl,
        predictions_check=pred,
        fr_rate_selection_T350=sel350,
        notes=notes, missing_inputs=missing, figure_errors=figure_errors,
        outputs=dict(summary=rel(os.path.join(out_dir, "summary.json")),
                     scoreboard=rel(os.path.join(out_dir, "scoreboard.md")), figures=rel(fig_dir)))
    summary = sanitize(summary)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)
    write_scoreboard(summary, os.path.join(out_dir, "scoreboard.md"))
    console_report(summary)


if __name__ == "__main__":
    main()
