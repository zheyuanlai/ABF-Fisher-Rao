#!/usr/bin/env python3
"""Aggregate WCA phase-diagram raw runs into the analysis CSVs used by the
plotting script and the report.

Reads ``<output_root>/raw/*.npz`` and writes, under ``<output_root>/summaries/``
(override with --out):

  phase_final_summary.csv    one row per run: all final scalar metrics + physics
  phase_runs_long.csv        long-format per-(run, saved time) time series
  phase_config_summary.csv   per (cell, method) median/IQR over seeds
  phase_profiles.csv         seed-mean final F/F'/p/q/Neff/birth/death per cell+method
  phase_fr_events.csv        per-run FR birth-death event statistics
  phase_genealogy.csv        per-run + matched genealogy diagnostics
  phase_main_table.csv       per cell: ABF vs FR-estimated headline (gain, R, wins)
  phase_improvement_ratios.csv  per (cell, FR method): improvement ratios + gains

A "cell" is one physical parameter setting, identified by ``physics_tag``
(b{beta}_h{h}_w{w}_n{n_dim}_a{a}). Matched-seed pairing is within
(stage, physics_tag, n_steps, n_replicas, seed).

Usage:
  python scripts/analyze_wca_phase_diagram.py --config configs/wca_phase_diagram_pilot.yaml --stages pilot
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import wca_phase_jobs as pj  # noqa: E402

SCALAR_COLS = [
    "run_id", "stage", "name", "method", "seed", "n_steps", "n_replicas",
    "beta", "h", "w", "n_dim", "M", "a", "beta_h", "physics_tag",
    "fr_rate", "target_ema_rate", "max_event_fraction", "fr_every", "fr_start_steps",
    "score_clip", "config_hash", "core_version", "had_nan", "device", "cuda_visible_devices",
    "total_replacement_events",
    "l2_f", "l2_fp", "integrated_l2_f",
    "l2_f_compact", "l2_f_transition", "l2_f_stretched",
    "l2_fp_compact", "l2_fp_transition", "l2_fp_stretched",
    "marginal_l2_uniform", "marginal_l2_target", "marginal_l2_ref",
    "n_compact_to_stretched", "n_stretched_to_compact", "n_barrier_crossings", "n_round_trips",
    "fr_event_fraction", "max_fr_event_fraction", "n_fr_applications",
    "deaths_per_fr_application", "clones_per_fr_application",
    "final_ancestor_ess", "min_ancestor_ess", "final_n_unique_ancestor",
    "final_max_ancestor_frac", "max_ancestor_frac_over_time",
    "runtime_seconds", "wall_seconds",
]


def _val(d, k):
    v = d[k]
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode()
    return v


def _physics_tag(d):
    return (f"b{_val(d,'beta'):g}_h{_val(d,'h'):g}_w{_val(d,'w'):g}"
            f"_n{int(_val(d,'n_dim'))}_a{_val(d,'a'):g}")


def load_runs(raw_dir, stages=None):
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        try:
            d = pj.load_run(path)
        except Exception as exc:
            print(f"  skip unreadable {path}: {exc!r}")
            continue
        if "l2_f" not in d:
            continue
        if stages and str(_val(d, "stage")) not in stages:
            continue
        rows.append(d)
    return rows


def _scalar_row(d):
    row = {}
    for k in SCALAR_COLS:
        if k == "physics_tag":
            row[k] = _physics_tag(d)
        elif k in d:
            row[k] = _val(d, k)
    return row


def write_csv(rows, out_path, cols=None):
    if not rows:
        print(f"  no rows -> skip {out_path}")
        return
    cols = cols or list(rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out_path} ({len(rows)} rows)")


def _iqr(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


# ---------------------------------------------------------------------------
def write_final_summary(runs, out_path):
    rows = [_scalar_row(d) for d in runs]
    write_csv(rows, out_path, cols=SCALAR_COLS)
    return rows


def write_runs_long(runs, out_path):
    rows = []
    for d in runs:
        t = np.asarray(d["times"], float)
        l2f = np.asarray(d["l2_f_t"], float)
        l2fp = np.asarray(d["l2_fp_t"], float)
        repl = np.asarray(d["repl_cumulative"], float)
        fc = np.asarray(d["frac_compact"], float)
        ft = np.asarray(d["frac_transition"], float)
        fs = np.asarray(d["frac_stretched"], float)
        ess = np.asarray(d["ancestor_ess_t"], float)
        maf = np.asarray(d["max_ancestor_frac_t"], float)
        tag = _physics_tag(d)
        for j in range(len(t)):
            rows.append(dict(
                stage=str(_val(d, "stage")), method=str(_val(d, "method")),
                physics_tag=tag, beta=_val(d, "beta"), h=_val(d, "h"), M=int(_val(d, "M")),
                seed=int(_val(d, "seed")), n_steps=int(_val(d, "n_steps")),
                t=float(t[j]), l2_f=float(l2f[j]), l2_fp=float(l2fp[j]),
                repl_cumulative=float(repl[j]),
                frac_compact=float(fc[j]), frac_transition=float(ft[j]), frac_stretched=float(fs[j]),
                ancestor_ess=float(ess[j]) if j < len(ess) else float("nan"),
                max_ancestor_frac=float(maf[j]) if j < len(maf) else float("nan")))
    write_csv(rows, out_path)


def _cell_key(d):
    return (str(_val(d, "stage")), _physics_tag(d), str(_val(d, "method")),
            int(_val(d, "n_steps")), int(_val(d, "n_replicas")))


def write_config_summary(runs, out_path):
    groups = {}
    for d in runs:
        groups.setdefault(_cell_key(d), []).append(d)
    out = []
    metrics = ["l2_f", "l2_fp", "integrated_l2_f", "l2_f_transition", "l2_fp_transition",
               "marginal_l2_ref", "marginal_l2_uniform",
               "n_barrier_crossings", "n_round_trips",
               "fr_event_fraction", "max_fr_event_fraction",
               "final_ancestor_ess", "min_ancestor_ess", "final_max_ancestor_frac",
               "total_replacement_events"]
    for key, ds in sorted(groups.items()):
        d0 = ds[0]
        rec = dict(stage=key[0], physics_tag=key[1], method=key[2], n_steps=key[3], n_replicas=key[4],
                   beta=_val(d0, "beta"), h=_val(d0, "h"), w=_val(d0, "w"),
                   n_dim=int(_val(d0, "n_dim")), M=int(_val(d0, "M")), a=_val(d0, "a"),
                   beta_h=_val(d0, "beta_h"), n_seeds=len(ds))
        for m in metrics:
            med, lo, hi = _iqr([_val(x, m) for x in ds if m in x])
            rec[f"{m}_median"] = med
            rec[f"{m}_q25"] = lo
            rec[f"{m}_q75"] = hi
        out.append(rec)
    write_csv(out, out_path)
    return out


def write_fr_events(runs, out_path):
    rows = []
    for d in runs:
        if str(_val(d, "method")) == "abf":
            continue
        rows.append(dict(
            stage=str(_val(d, "stage")), method=str(_val(d, "method")),
            physics_tag=_physics_tag(d), beta=_val(d, "beta"), h=_val(d, "h"),
            M=int(_val(d, "M")), seed=int(_val(d, "seed")),
            total_replacement_events=int(_val(d, "total_replacement_events")),
            n_fr_applications=int(_val(d, "n_fr_applications")),
            fr_event_fraction=_val(d, "fr_event_fraction"),
            max_fr_event_fraction=_val(d, "max_fr_event_fraction"),
            deaths_per_fr_application=_val(d, "deaths_per_fr_application"),
            clones_per_fr_application=_val(d, "clones_per_fr_application")))
    write_csv(rows, out_path)


def write_genealogy(runs, out_path):
    rows = []
    for d in runs:
        if str(_val(d, "method")) == "abf":
            continue
        rows.append(dict(
            stage=str(_val(d, "stage")), method=str(_val(d, "method")),
            physics_tag=_physics_tag(d), beta=_val(d, "beta"), h=_val(d, "h"),
            M=int(_val(d, "M")), seed=int(_val(d, "seed")), n_replicas=int(_val(d, "n_replicas")),
            final_ancestor_ess=_val(d, "final_ancestor_ess"),
            min_ancestor_ess=_val(d, "min_ancestor_ess"),
            final_n_unique_ancestor=int(_val(d, "final_n_unique_ancestor")),
            final_max_ancestor_frac=_val(d, "final_max_ancestor_frac"),
            max_ancestor_frac_over_time=_val(d, "max_ancestor_frac_over_time"),
            ess_fraction=float(_val(d, "final_ancestor_ess")) / float(_val(d, "n_replicas"))))
    write_csv(rows, out_path)


def write_profiles(runs, out_path):
    """Seed-mean final profiles per (cell, method) in long format."""
    groups = {}
    for d in runs:
        groups.setdefault(_cell_key(d), []).append(d)
    rows = []
    for key, ds in sorted(groups.items()):
        d0 = ds[0]
        grid = np.asarray(d0["grid"], float)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            F = np.nanmean(np.stack([np.asarray(x["final_pmf"], float) for x in ds]), axis=0)
            Fp = np.nanmean(np.stack([np.asarray(x["final_mean_force"], float) for x in ds]), axis=0)
            p = np.nanmean(np.stack([np.asarray(x["final_p_hat"], float) for x in ds]), axis=0)
            q = np.nanmean(np.stack([np.asarray(x["final_q_target"], float) for x in ds]), axis=0)
            neff = np.nanmean(np.stack([np.asarray(x["final_eff_counts"], float) for x in ds]), axis=0)
        refF = np.asarray(d0["ref_free_energy"], float)
        refFp = np.asarray(d0["ref_mean_force"], float)
        refp = np.asarray(d0["ref_p_boltzmann"], float)
        for j in range(len(grid)):
            rows.append(dict(stage=key[0], physics_tag=key[1], method=key[2],
                             beta=_val(d0, "beta"), h=_val(d0, "h"), M=int(_val(d0, "M")),
                             z=float(grid[j]), F=float(F[j]), Fp=float(Fp[j]),
                             p=float(p[j]), q=float(q[j]), neff=float(neff[j]),
                             ref_F=float(refF[j]), ref_Fp=float(refFp[j]), ref_p=float(refp[j])))
    write_csv(rows, out_path)


def _matched_pairs(runs):
    """Return abf baseline indexed by matched key, plus FR runs grouped by (cell, method)."""
    def mkey(d):
        return (str(_val(d, "stage")), _physics_tag(d), int(_val(d, "n_steps")),
                int(_val(d, "n_replicas")), int(_val(d, "seed")))
    abf = {}
    for d in runs:
        if str(_val(d, "method")) == "abf":
            abf[mkey(d)] = d
    return abf, mkey


def write_improvement_ratios(runs, out_path):
    abf, mkey = _matched_pairs(runs)
    groups = {}
    for d in runs:
        if str(_val(d, "method")) == "abf":
            continue
        base = abf.get(mkey(d))
        if base is None:
            continue
        gk = (str(_val(d, "stage")), _physics_tag(d), str(_val(d, "method")),
              int(_val(d, "n_steps")), int(_val(d, "n_replicas")))
        groups.setdefault(gk, []).append((d, base))
    out = []
    for gk, pairs in sorted(groups.items()):
        d0 = pairs[0][0]
        R_f, R_int, gains_f, gains_int, wins = [], [], [], [], 0
        for fr, base in pairs:
            bf, ff = float(_val(base, "l2_f")), float(_val(fr, "l2_f"))
            bi, fi = float(_val(base, "integrated_l2_f")), float(_val(fr, "integrated_l2_f"))
            if ff > 0:
                R_f.append(bf / ff)
            if fi > 0:
                R_int.append(bi / fi)
            if bf > 0:
                gains_f.append(100.0 * (bf - ff) / bf)
            if bi > 0:
                gains_int.append(100.0 * (bi - fi) / bi)
            wins += int(ff < bf)
        n = len(pairs)
        out.append(dict(
            stage=gk[0], physics_tag=gk[1], method=gk[2], n_steps=gk[3], n_replicas=gk[4],
            beta=_val(d0, "beta"), h=_val(d0, "h"), M=int(_val(d0, "M")), beta_h=_val(d0, "beta_h"),
            n_pairs=n, n_wins=wins, win_rate=(wins / n if n else float("nan")),
            R_final_median=float(np.median(R_f)) if R_f else float("nan"),
            R_integrated_median=float(np.median(R_int)) if R_int else float("nan"),
            median_gain_pct_F=float(np.median(gains_f)) if gains_f else float("nan"),
            mean_gain_pct_F=float(np.mean(gains_f)) if gains_f else float("nan"),
            median_gain_pct_intF=float(np.median(gains_int)) if gains_int else float("nan")))
    write_csv(out, out_path)
    return out


def write_main_table(runs, out_path):
    """Per cell: ABF vs FR-estimated headline (median L2(F), gain%, R, wins)."""
    abf, mkey = _matched_pairs(runs)
    # group abf and fr_estimated medians per cell
    cells = {}
    for d in runs:
        ck = (str(_val(d, "stage")), _physics_tag(d), int(_val(d, "n_steps")), int(_val(d, "n_replicas")))
        cells.setdefault(ck, {"abf": [], "fr_estimated": [], "fr_uniform": [], "fr_oracle": [], "d0": d})
        m = str(_val(d, "method"))
        if m in cells[ck]:
            cells[ck][m].append(d)
    out = []
    for ck, grp in sorted(cells.items()):
        d0 = grp["d0"]
        def med(ms, metric="l2_f"):
            v = [float(_val(x, metric)) for x in grp.get(ms, [])]
            v = [x for x in v if np.isfinite(x)]
            return float(np.median(v)) if v else float("nan")
        abf_f = med("abf")
        est_f = med("fr_estimated")
        # matched gain / winrate for estimated
        gains, wins, n = [], 0, 0
        for fr in grp.get("fr_estimated", []):
            base = abf.get(mkey(fr))
            if base is None:
                continue
            bf, ff = float(_val(base, "l2_f")), float(_val(fr, "l2_f"))
            n += 1
            wins += int(ff < bf)
            if bf > 0:
                gains.append(100.0 * (bf - ff) / bf)
        out.append(dict(
            stage=ck[0], physics_tag=ck[1], n_steps=ck[2], n_replicas=ck[3],
            beta=_val(d0, "beta"), h=_val(d0, "h"), M=int(_val(d0, "M")), beta_h=_val(d0, "beta_h"),
            abf_l2_f=abf_f, fr_est_l2_f=est_f,
            fr_uniform_l2_f=med("fr_uniform"), fr_oracle_l2_f=med("fr_oracle"),
            R_est=(abf_f / est_f if est_f and np.isfinite(est_f) and est_f > 0 else float("nan")),
            median_gain_pct=float(np.median(gains)) if gains else float("nan"),
            n_seeds=n, n_wins=wins, win_rate=(wins / n if n else float("nan")),
            abf_barrier_crossings=med("abf", "n_barrier_crossings"),
            fr_est_barrier_crossings=med("fr_estimated", "n_barrier_crossings"),
            fr_est_final_ess=med("fr_estimated", "final_ancestor_ess"),
            fr_est_max_anc_frac=med("fr_estimated", "final_max_ancestor_frac")))
    write_csv(out, out_path)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stages", nargs="*", default=None)
    args = ap.parse_args(argv)
    cfg = pj.load_yaml(args.config)
    raw_dir = args.raw or os.path.join(cfg["output_root"], "raw")
    out_dir = args.out or os.path.join(cfg["output_root"], "summaries")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[analyze] raw={raw_dir} out={out_dir} stages={args.stages or 'ALL'}")

    runs = load_runs(raw_dir, args.stages)
    if not runs:
        print("[analyze] no runs found")
        return 1
    print(f"[analyze] loaded {len(runs)} runs")
    write_final_summary(runs, os.path.join(out_dir, "phase_final_summary.csv"))
    write_runs_long(runs, os.path.join(out_dir, "phase_runs_long.csv"))
    write_config_summary(runs, os.path.join(out_dir, "phase_config_summary.csv"))
    write_profiles(runs, os.path.join(out_dir, "phase_profiles.csv"))
    write_fr_events(runs, os.path.join(out_dir, "phase_fr_events.csv"))
    write_genealogy(runs, os.path.join(out_dir, "phase_genealogy.csv"))
    write_improvement_ratios(runs, os.path.join(out_dir, "phase_improvement_ratios.csv"))
    write_main_table(runs, os.path.join(out_dir, "phase_main_table.csv"))
    print("[analyze] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
