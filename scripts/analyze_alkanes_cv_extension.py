#!/usr/bin/env python3
"""Aggregate CV-extension raw runs -> summary CSVs, matched-seed paired stats, ABF
starvation classification, and the pre-registered mFR success verdict.

Outputs (under <output_root>/summaries/):
  cv_runs_long.csv        one row per (job, seed): all per-seed metrics + physics
  cv_config_summary.csv   per (cell, method): median / IQR over seeds
  cv_paired.csv           matched-seed deltas of each method vs ABF + bootstrap CI + win rate
  cv_starvation.csv       ABF starvation classification per cell (decision gate)
  cv_success.csv          pre-registered mFR success criteria verdict per (cell, method)
  cv_main.csv             headline comparison table

No GPU. Reads <output_root>/raw/*.npz.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from alkanes import jobs_cv as J  # noqa: E402

RNG = np.random.default_rng(20260719)


def load_long(raw_dir):
    rows = []
    meta_by_cell = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            continue
        if "per_seed" not in d.files:
            continue
        per_seed = json.loads(str(d["per_seed"]))
        kind = str(d["kind"])
        meta = dict(kind=kind, molecule=str(d["molecule"]), method=str(d["method"]),
                    name=str(d["name"]), init_mode=str(d["init_mode"]), beta=float(d["beta"]),
                    sigma=float(d["sigma"]), decouple=bool(d["decouple"]), stage=str(d["stage"]),
                    n_steps=int(d["n_steps"]), n_replicas=int(d["n_replicas"]), run_id=str(d["run_id"]),
                    wall_seconds=float(d["wall_seconds"]))
        # cell key INCLUDES stage + grid so different stages (screen/resgate/production/
        # confirm/tuning/control) and grid resolutions never merge or cross-pair on seeds.
        spec = json.loads(str(d["spec_json"])) if "spec_json" in d.files else {}
        grid_tag = int(spec.get("grid2d", 0)) if kind == "joint2d" else int(spec.get("dist_n_grid", 0))
        cell = f"{meta['stage']}_{kind}_{meta['molecule']}_b{meta['beta']:g}_{meta['init_mode']}_g{grid_tag}"
        F_range = float(d["F_range_thermal"]) if "F_range_thermal" in d.files else np.nan
        # final-quarter L2 decrease per seed (plateau diagnostic)
        l2t = d["l2_F_t"] if "l2_F_t" in d.files else None
        for si, rec in enumerate(per_seed):
            row = dict(meta); row.update(rec); row["cell"] = cell; row["F_range_thermal"] = F_range
            if l2t is not None and l2t.ndim == 2 and l2t.shape[1] >= 4:
                T = l2t.shape[1]; q3 = l2t[si, (3 * T) // 4]; fin = l2t[si, -1]
                row["final_quarter_decrease"] = float((q3 - fin) / max(abs(q3), 1e-9))
                row["norm_final_l2_F"] = float(fin / F_range) if F_range > 0 else np.nan
            rows.append(row)
        meta_by_cell.setdefault(cell, kind)
    return pd.DataFrame(rows), meta_by_cell


def _boot_ci(x, n=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    bs = x[idx].mean(1)
    return float(np.median(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def config_summary(df):
    skip = {"beta", "sigma", "n_steps", "seed", "wall_seconds", "F_range_thermal"}
    metrics = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in skip]
    g = df.groupby(["cell", "kind", "molecule", "beta", "init_mode", "stage", "name", "method"])
    out = g[metrics].median().add_suffix("_med")
    out["n_seeds"] = g.size()
    return out.reset_index()


def paired(df, key="final_l2_F"):
    rows = []
    for cell, sub in df.groupby("cell"):
        abf = sub[sub.method == "abf"].set_index("seed")[key] if key in sub else None
        if abf is None or abf.empty:
            continue
        for name, m in sub.groupby("name"):
            if name == "abf" or key not in m:
                continue
            mm = m.set_index("seed")[key]
            common = abf.index.intersection(mm.index)
            if len(common) == 0:
                continue
            a = abf.loc[common].values.astype(float); b = mm.loc[common].values.astype(float)
            delta = b - a
            rel = (b - a) / np.where(np.abs(a) > 1e-12, np.abs(a), np.nan)
            med, lo, hi = _boot_ci(delta); rmed, rlo, rhi = _boot_ci(rel)
            rows.append(dict(cell=cell, method=name, metric=key, n_pairs=len(common),
                             abf_med=float(np.median(a)), method_med=float(np.median(b)),
                             delta_med=med, delta_lo=lo, delta_hi=hi,
                             rel_med=rmed, rel_lo=rlo, rel_hi=rhi, win_rate=float(np.mean(b < a))))
    return pd.DataFrame(rows)


def starvation(df):
    """ABF-only starvation classification per SCREEN cell (>=2 criteria => starved).

    Restricted to stage=='screen' so the resolution gate (stage 'resgate', same cell) and
    methods stages do not pollute the headline screen verdict.
    """
    rows = []
    scr = df[(df.method == "abf") & (df.stage == "screen")]
    for cell, sub in scr.groupby("cell"):
        kind = sub["kind"].iloc[0]
        norm_l2 = np.nanmedian(sub.get("norm_final_l2_F", pd.Series([np.nan])))
        dec = np.nanmedian(sub.get("final_quarter_decrease", pd.Series([np.nan])))
        # The mean-force ESTIMATOR floor (measured exactly on the decoupled 2-D torsion gate)
        # is ~5.8% of the free-energy range, shared by every method. The 1-D distance pipeline
        # has its own (KDE-reference) floor that is NOT separately calibrated; 10% is used as a
        # CONSERVATIVE bound well above any plausible 1-D or 2-D floor (the starved R15 cells sit
        # at 14-16%, ~2.5x the 2-D floor). "Starved" = error well above the floor, not >5%.
        FLOOR = 0.10
        crit = {}
        crit["c1_normL2_above_floor"] = bool(np.isfinite(norm_l2) and norm_l2 > FLOOR)
        # plateau ONLY counts as starvation if the error is still above the floor (else it is
        # a converged run that has stopped improving at the estimator floor).
        crit["c2_plateau_and_high"] = bool(np.isfinite(dec) and dec < 0.10
                                           and np.isfinite(norm_l2) and norm_l2 > FLOOR)
        Nrep = float(sub["n_replicas"].iloc[0]) if "n_replicas" in sub else 1024.0
        if kind == "dist":
            low = np.nanmedian(sub.get("low_support_fraction", pd.Series([np.nan])))
            rt = np.nanmedian(sub.get("n_round_trips", pd.Series([np.nan])))
            rt_per = rt / Nrep if np.isfinite(rt) else np.nan
            condtv = np.nanmedian(sub.get("dist_cond_tv_weighted", pd.Series([np.nan])))
            crit["c3_low_support_20pct"] = bool(np.isfinite(low) and low >= 0.20)
            crit["c4_poor_mixing"] = bool(np.isfinite(rt_per) and rt_per < 1.0)
            crit["c5_conditional_biased"] = bool(np.isfinite(condtv) and condtv > 0.15)
        else:
            nb = np.nanmedian(sub.get("n_basins_visited", pd.Series([np.nan])))
            rt = np.nanmedian(sub.get("n_round_trips", pd.Series([np.nan])))
            Nrep = float(sub["n_replicas"].iloc[0]) if "n_replicas" in sub else 2048.0
            rt_per = rt / Nrep if np.isfinite(rt) else np.nan
            # basin discovery + mixing are the valid 2-D signals (occupancy of the BIASED
            # samples is ~uniform by design, so it is not a starvation signal). Fidelity of
            # the reconstructed landscape is captured by c1 (thermal L2) + eq-weighted L2.
            crit["c3_missing_basins"] = bool(np.isfinite(nb) and nb < 8.5)
            crit["c4_poor_mixing"] = bool(np.isfinite(rt_per) and rt_per < 1.0)
        # Count DISTINCT evidence FAMILIES, not raw criteria: c1 (magnitude) and c2 (plateau)
        # are not independent -- c2 implies c1 by construction -- so they form ONE
        # convergence-failure family; support/mixing/conditional are independent mechanism
        # families. "Starved" requires >=2 distinct families (magnitude alone => intermediate).
        fam_convergence = bool(crit["c1_normL2_above_floor"] or crit["c2_plateau_and_high"])
        fam_support = bool(crit.get("c3_low_support_20pct", crit.get("c3_missing_basins", False)))
        fam_mixing = bool(crit.get("c4_poor_mixing", False))
        fam_conditional = bool(crit.get("c5_conditional_biased", False))
        n_families = sum([fam_convergence, fam_support, fam_mixing, fam_conditional])
        verdict = "starved" if n_families >= 2 else ("intermediate" if n_families == 1 else "easy")
        rows.append(dict(cell=cell, kind=kind, n_families=n_families,
                         n_criteria_fired=sum(crit.values()), verdict=verdict,
                         norm_final_l2_F=norm_l2, final_quarter_decrease=dec,
                         fam_convergence=fam_convergence, fam_support=fam_support,
                         fam_mixing=fam_mixing, fam_conditional=fam_conditional, **crit))
    return pd.DataFrame(rows)


def success(df):
    """Pre-registered mFR success verdict per (cell, method). A POSITIVE result requires ALL
    per-seed-computable pre-registered criteria (not just the L2 ones): (1,2) median final AND
    integrated L2 improve >=15%; (3) paired 95% CI excludes 0; (4) win rate >=0.75; (6) basin-
    occupancy fidelity worsens <=10% vs ABF; (7) ancestor ESS >=0.30 N; (8) max ancestor frac
    <=0.05; (9) event fraction not saturating (<5%). Criteria (5) eq/thermal agreement, (10)
    frozen-bias, (11) grid-resolution, (12) equal-compute are checked SEPARATELY (frozen/confirm
    stages) and reported in finalization -- a POSITIVE here is necessary, not sufficient, for a
    headline positive. A run improving L2 but violating genealogy/basin/event is 'false' (not
    positive). ESS/frac/event checks apply to FR (birth-death) methods only.
    """
    rows = []
    pf = paired(df, "final_l2_F"); pi = paired(df, "integrated_l2_F")
    med = df.groupby(["cell", "name"]).median(numeric_only=True)
    for cell in df.cell.unique():
        abf_basin = (med.loc[(cell, "abf")].get("basin_occupancy_tv", np.nan)
                     if (cell, "abf") in med.index else np.nan)
        for name in df[df.cell == cell]["name"].unique():
            if name == "abf":
                continue
            rf = pf[(pf.cell == cell) & (pf.method == name)]
            ri = pi[(pi.cell == cell) & (pi.method == name)]
            if rf.empty or ri.empty:
                continue
            rf = rf.iloc[0]; ri = ri.iloc[0]
            is_fr = ("fr" in str(name)) and ("opes" not in str(name))
            improve_both = (rf.rel_med <= -0.15) and (ri.rel_med <= -0.15)
            ci_excl = (rf.rel_hi < 0) and (ri.rel_hi < 0)
            win_ok = rf.win_rate >= 0.75
            geneal_ok = event_ok = basin_ok = True
            if (cell, name) in med.index:
                m = med.loc[(cell, name)]
                nrep = float(m.get("n_replicas", 1024) or 1024)
                if is_fr:
                    ess = m.get("final_ancestor_ess", np.nan)
                    geneal_ok = bool(np.isfinite(ess) and ess / nrep >= 0.30
                                     and (m.get("final_max_ancestor_frac", 1.0) <= 0.05))
                    event_ok = bool(m.get("fr_event_fraction", 0.0) < 0.05)
                mb = m.get("basin_occupancy_tv", np.nan)
                if np.isfinite(mb) and np.isfinite(abf_basin):
                    basin_ok = bool(mb <= 1.10 * abf_basin + 1e-9)
            positive = improve_both and ci_excl and win_ok and geneal_ok and event_ok and basin_ok
            equiv = (rf.rel_lo >= -0.10) and (rf.rel_hi <= 0.10)
            harmful = rf.rel_lo > 0.10
            # 'false improvement': L2 better but a guard (genealogy/event/basin) violated
            false_improve = improve_both and ci_excl and not (geneal_ok and event_ok and basin_ok)
            verdict = ("POSITIVE" if positive else "false-improvement" if false_improve
                       else "harmful" if harmful else "equivalent" if equiv else "inconclusive")
            rows.append(dict(cell=cell, method=name, verdict=verdict,
                             rel_final=rf.rel_med, rel_final_lo=rf.rel_lo, rel_final_hi=rf.rel_hi,
                             rel_integrated=ri.rel_med, win_rate_final=rf.win_rate,
                             geneal_ok=geneal_ok, event_ok=event_ok, basin_ok=basin_ok))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = J.load_yaml(args.config)
    root = cfg["output_root"]; raw_dir = os.path.join(root, "raw"); out_dir = os.path.join(root, "summaries")
    os.makedirs(out_dir, exist_ok=True)
    df, _ = load_long(raw_dir)
    if df.empty:
        print("no runs found in", raw_dir); return 1
    df.to_csv(os.path.join(out_dir, "cv_runs_long.csv"), index=False)
    config_summary(df).to_csv(os.path.join(out_dir, "cv_config_summary.csv"), index=False)
    pd.concat([paired(df, k) for k in ("final_l2_F", "integrated_l2_F")], ignore_index=True).to_csv(
        os.path.join(out_dir, "cv_paired.csv"), index=False)
    stv = starvation(df); stv.to_csv(os.path.join(out_dir, "cv_starvation.csv"), index=False)
    success(df).to_csv(os.path.join(out_dir, "cv_success.csv"), index=False)
    main_cols = ["final_l2_F", "integrated_l2_F", "n_transitions", "n_round_trips",
                 "final_ancestor_ess", "fr_event_fraction", "n_basins_visited",
                 "basin_occupancy_tv", "cond_tv_weighted", "dist_cond_tv_weighted",
                 "low_support_fraction", "final_neff_frac"]
    have = [c for c in main_cols if c in df.columns]
    df.groupby(["cell", "name"])[have].median().reset_index().to_csv(
        os.path.join(out_dir, "cv_main.csv"), index=False)
    print(f"[analyze-cv] {len(df)} run-seeds -> {out_dir}")
    if not stv.empty:
        print("\n=== ABF starvation classification ===")
        print(stv[["cell", "verdict", "n_criteria_fired"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
