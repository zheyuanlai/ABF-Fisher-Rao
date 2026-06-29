#!/usr/bin/env python3
"""Aggregate a WCA follow-up study's raw runs into analysis CSVs + LaTeX tables.

Handles all three follow-up studies (selected by the run npz ``mode``/``study``):

  representative (sample) ->
      representative_cells_summary.csv   per (cell, method) median/IQR over seeds
      matched_seed_table.csv             abf vs each FR method, matched-seed gain
      adaptive_fr_event_log.csv          combined per-FR-event adaptive log
      LaTeX: report/tables/wca_representative_main.tex,
             report/tables/wca_adaptive_compare.tex
  equal_compute (sample) ->
      equal_compute_summary.csv          per (cell, method, budget) median/IQR
      LaTeX: report/tables/wca_equal_compute.tex
  frozen_bias (frozen) ->
      frozen_bias_summary.csv            per (cell, source method) recon vs learned L2
      LaTeX: report/tables/wca_frozen_bias.tex

Always also writes ``<study>_run_summary.csv`` (one row per run). The online
source-study L2(F) is folded into the frozen summary for the online-vs-frozen
contrast when --online-summary is given.

Usage:
  python scripts/analyze_wca_followup.py --config configs/wca_representative.yaml --stages representative
  python scripts/analyze_wca_followup.py --config configs/wca_equal_compute.yaml --stages equal_compute equal_compute_plus
  python scripts/analyze_wca_followup.py --config configs/wca_frozen_bias.yaml --stages frozen_bias \
      --online-summary results/wca_representative/summaries/representative_cells_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import wca_followup_jobs as fj  # noqa: E402

REPORT_TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "tables")


def _val(d, k, default=None):
    if k not in (d.files if hasattr(d, "files") else d):
        return default
    v = d[k]
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode()
    return v


def _iqr(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


def write_csv(rows, out_path, cols=None):
    if not rows:
        print(f"  no rows -> skip {out_path}")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = cols or list(rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out_path} ({len(rows)} rows)")


def load_runs(raw_dir, stages):
    runs = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        try:
            d = np.load(path, allow_pickle=True)
        except Exception as exc:
            print(f"  skip unreadable {path}: {exc!r}")
            continue
        if "l2_f" not in d.files:
            continue
        runs.append({k: d[k] for k in d.files})
    return runs


def _tag(d):
    return (f"b{_val(d,'beta'):g}_h{_val(d,'h'):g}_w{_val(d,'w'):g}"
            f"_n{int(_val(d,'n_dim'))}_a{_val(d,'a'):g}")


# --------------------------------------------------------------------------- #
# sample studies (representative / equal_compute)
# --------------------------------------------------------------------------- #
SAMPLE_SCALARS = [
    "run_id", "study", "stage", "mode", "name", "method", "seed", "n_steps", "n_replicas",
    "budget", "beta", "h", "w", "n_dim", "M", "a", "beta_h",
    "fr_rate", "adaptive_support_mode", "adaptive_base_fr_rate",
    "l2_f", "l2_fp", "integrated_l2_f", "l2_f_transition", "l2_fp_transition",
    "marginal_l2_uniform", "marginal_l2_target", "marginal_l2_ref",
    "n_barrier_crossings", "n_round_trips",
    "fr_event_fraction", "max_fr_event_fraction",
    "fr_score_std", "fr_score_absmax", "fr_score_clip_fraction",
    "final_ancestor_ess", "min_ancestor_ess", "final_n_unique_ancestor",
    "final_max_ancestor_frac", "had_nan", "runtime_seconds", "wall_seconds",
]


def sample_run_rows(runs):
    rows = []
    for d in runs:
        if str(_val(d, "mode", "sample")) == "frozen":
            continue
        row = {k: _val(d, k) for k in SAMPLE_SCALARS if k in d}
        row["physics_tag"] = _tag(d)
        rows.append(row)
    return rows


def _cellkey(d):
    return (_tag(d), int(_val(d, "n_replicas")), int(_val(d, "n_steps")))


def _matched(runs):
    """abf baseline indexed by (cell-physics, budget, seed)."""
    def mk(d):
        return (_tag(d), int(_val(d, "n_replicas")), int(_val(d, "n_steps")), int(_val(d, "seed")))
    abf = {mk(d): d for d in runs if str(_val(d, "method")) == "abf"}
    return abf, mk


def config_summary(runs, metrics):
    groups = {}
    for d in runs:
        if str(_val(d, "mode", "sample")) == "frozen":
            continue
        groups.setdefault((_cellkey(d), str(_val(d, "method")), str(_val(d, "name"))), []).append(d)
    out = []
    for (ck, method, name), ds in sorted(groups.items()):
        d0 = ds[0]
        rec = dict(physics_tag=ck[0], n_replicas=ck[1], n_steps=ck[2],
                   budget=ck[1] * ck[2], method=method, name=name,
                   beta=_val(d0, "beta"), h=_val(d0, "h"), M=int(_val(d0, "M")),
                   beta_h=_val(d0, "beta_h"), n_seeds=len(ds))
        for m in metrics:
            med, lo, hi = _iqr([_val(x, m) for x in ds if m in x])
            rec[f"{m}_median"] = med
            rec[f"{m}_q25"] = lo
            rec[f"{m}_q75"] = hi
        out.append(rec)
    return out


def matched_seed_table(runs):
    abf, mk = _matched(runs)
    groups = {}
    for d in runs:
        if str(_val(d, "method")) == "abf" or str(_val(d, "mode", "sample")) == "frozen":
            continue
        groups.setdefault((_cellkey(d), str(_val(d, "name"))), []).append(d)
    out = []
    for (ck, name), ds in sorted(groups.items()):
        d0 = ds[0]
        gains, R, wins, n = [], [], 0, 0
        for fr in ds:
            base = abf.get(mk(fr))
            if base is None:
                continue
            bf, ff = float(_val(base, "l2_f")), float(_val(fr, "l2_f"))
            if not (np.isfinite(bf) and np.isfinite(ff)):
                continue
            n += 1
            wins += int(ff < bf)
            if bf > 0:
                gains.append(100.0 * (bf - ff) / bf)
            if ff > 0:
                R.append(bf / ff)
        abf_rows = [abf[k] for k in abf if k[:3] == (ck[0], ck[1], ck[2])]
        out.append(dict(
            physics_tag=ck[0], n_replicas=ck[1], n_steps=ck[2], method=str(_val(d0, "method")),
            name=name, beta=_val(d0, "beta"), h=_val(d0, "h"), M=int(_val(d0, "M")),
            abf_l2_f_median=_iqr([float(_val(x, "l2_f")) for x in abf_rows])[0],
            fr_l2_f_median=_iqr([float(_val(x, "l2_f")) for x in ds])[0],
            n_seeds=n, n_wins=wins, win_rate=(wins / n if n else float("nan")),
            median_gain_pct_F=(float(np.median(gains)) if gains else float("nan")),
            q25_gain_pct_F=(float(np.percentile(gains, 25)) if gains else float("nan")),
            q75_gain_pct_F=(float(np.percentile(gains, 75)) if gains else float("nan")),
            R_final_median=(float(np.median(R)) if R else float("nan")),
            fr_event_fraction_median=_iqr([_val(x, "fr_event_fraction") for x in ds])[0],
            fr_final_ess_median=_iqr([_val(x, "final_ancestor_ess") for x in ds])[0],
            fr_max_anc_frac_median=_iqr([_val(x, "final_max_ancestor_frac") for x in ds])[0]))
    return out


def adaptive_event_log(runs):
    rows = []
    for d in runs:
        if str(_val(d, "method")) != "fr_estimated_adaptive":
            continue
        steps = np.asarray(_val(d, "adaptive_log_step", np.array([])), float)
        if steps.size == 0:
            continue
        keys = ["fr_rate_eff", "support_gate", "diversity_gate", "event_gate",
                "support_ema", "ess_frac", "event_fraction", "score_std", "score_clip_fraction"]
        arrs = {k: np.asarray(_val(d, f"adaptive_log_{k}", np.array([])), float) for k in keys}
        tag = _tag(d)
        for j in range(steps.size):
            row = dict(physics_tag=tag, beta=_val(d, "beta"), h=_val(d, "h"),
                       seed=int(_val(d, "seed")), step=int(steps[j]))
            for k in keys:
                a = arrs[k]
                row[k] = float(a[j]) if j < a.size else float("nan")
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# frozen study
# --------------------------------------------------------------------------- #
def frozen_summary(runs, online_lookup=None):
    groups = {}
    for d in runs:
        if str(_val(d, "mode", "sample")) != "frozen":
            continue
        groups.setdefault((_tag(d), str(_val(d, "frozen_source_method"))), []).append(d)
    out = []
    for (tag, src), ds in sorted(groups.items()):
        d0 = ds[0]
        rec = dict(physics_tag=tag, source_method=src, beta=_val(d0, "beta"), h=_val(d0, "h"),
                   M=int(_val(d0, "M")), n_frozen_seeds=len(ds),
                   n_bias_sources=int(_val(d0, "n_bias_sources", -1)),
                   frozen_recon_l2_f_median=_iqr([_val(x, "frozen_recon_l2_f") for x in ds])[0],
                   frozen_recon_l2_f_q25=_iqr([_val(x, "frozen_recon_l2_f") for x in ds])[1],
                   frozen_recon_l2_f_q75=_iqr([_val(x, "frozen_recon_l2_f") for x in ds])[2],
                   learned_bias_l2_f=_iqr([_val(x, "learned_bias_l2_f") for x in ds])[0])
        if online_lookup is not None:
            rec["online_l2_f_median"] = online_lookup.get((tag, src), float("nan"))
        out.append(rec)
    return out


def load_online_lookup(path):
    """(physics_tag, method) -> online median L2(F) from a representative summary."""
    if not path or not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        try:
            out[(r["physics_tag"], r["method"])] = float(r["l2_f_median"])
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def _f(x, nd=4):
    try:
        x = float(x)
    except Exception:
        return "--"
    return "--" if not np.isfinite(x) else f"{x:.{nd}f}"


def latex_representative(matched, out_path):
    by = {}
    for r in matched:
        by.setdefault(r["physics_tag"], {})[r["method"]] = r
    order = ["fr_estimated", "fr_uniform", "fr_oracle", "fr_estimated_adaptive"]
    lines = [r"% auto-generated by analyze_wca_followup.py",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"cell & method & ABF $\Ltwo(F)$ & mFR $\Ltwo(F)$ & gain\% & wins \\",
             r"\midrule"]
    for tag in sorted(by):
        first = True
        for m in order:
            r = by[tag].get(m)
            if r is None:
                continue
            cell = (f"$\\beta{{=}}{r['beta']:g},h{{=}}{r['h']:g}$" if first else "")
            first = False
            lines.append(
                f"{cell} & \\texttt{{{m.replace('_','-')}}} & {_f(r['abf_l2_f_median'])} & "
                f"{_f(r['fr_l2_f_median'])} & {_f(r['median_gain_pct_F'],1)} & "
                f"{int(r['n_wins'])}/{int(r['n_seeds'])} \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    _write(out_path, "\n".join(lines))


def latex_adaptive_compare(matched, out_path):
    by = {}
    for r in matched:
        by.setdefault(r["physics_tag"], {})[r["method"]] = r
    lines = [r"% auto-generated by analyze_wca_followup.py",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"cell & fixed mFR gain\% & adaptive gain\% & adaptive ESS frac & adaptive event frac \\",
             r"\midrule"]
    for tag in sorted(by):
        fixed = by[tag].get("fr_estimated")
        adapt = by[tag].get("fr_estimated_adaptive")
        if fixed is None and adapt is None:
            continue
        b = (fixed or adapt)
        ess = adapt["fr_final_ess_median"] if adapt else float("nan")
        N = adapt["n_replicas"] if adapt else 1024
        lines.append(
            f"$\\beta{{=}}{b['beta']:g},h{{=}}{b['h']:g}$ & "
            f"{_f(fixed['median_gain_pct_F'],1) if fixed else '--'} & "
            f"{_f(adapt['median_gain_pct_F'],1) if adapt else '--'} & "
            f"{_f((ess/N) if adapt else float('nan'),3)} & "
            f"{_f(adapt['fr_event_fraction_median'],4) if adapt else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out_path, "\n".join(lines))


def latex_equal_compute(summary, out_path):
    rows = [r for r in summary]
    by = {}
    for r in rows:
        by.setdefault(r["physics_tag"], []).append(r)
    lines = [r"% auto-generated by analyze_wca_followup.py",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"cell & method & $N$ & $n_{\rm steps}$ & budget & $\Ltwo(F)$ \\",
             r"\midrule"]
    for tag in sorted(by):
        block = sorted(by[tag], key=lambda r: (r["method"], r["budget"]))
        first = True
        for r in block:
            cell = (f"$\\beta{{=}}{r['beta']:g},h{{=}}{r['h']:g}$" if first else "")
            first = False
            lines.append(
                f"{cell} & \\texttt{{{r['method'].replace('_','-')}}} & {int(r['n_replicas'])} & "
                f"{int(r['n_steps'])} & {int(r['budget']):.0e} & {_f(r['l2_f_median'])} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    _write(out_path, "\n".join(lines))


def latex_frozen(summary, out_path):
    lines = [r"% auto-generated by analyze_wca_followup.py",
             r"\begin{tabular}{llrrr}", r"\toprule",
             r"cell & learned bias & online $\Ltwo(F)$ & frozen recon $\Ltwo(F)$ & learned-bias $\Ltwo(F)$ \\",
             r"\midrule"]
    by = {}
    for r in summary:
        by.setdefault(r["physics_tag"], []).append(r)
    for tag in sorted(by):
        first = True
        for r in sorted(by[tag], key=lambda r: r["source_method"]):
            cell = (f"$\\beta{{=}}{r['beta']:g},h{{=}}{r['h']:g}$" if first else "")
            first = False
            lines.append(
                f"{cell} & \\texttt{{{r['source_method'].replace('_','-')}}} & "
                f"{_f(r.get('online_l2_f_median', float('nan')))} & "
                f"{_f(r['frozen_recon_l2_f_median'])} & {_f(r['learned_bias_l2_f'])} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    _write(out_path, "\n".join(lines))


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text + "\n")
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--online-summary", default=None,
                    help="representative_cells_summary.csv for the frozen online-vs-frozen column")
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args(argv)
    cfg = fj.load_yaml(args.config)
    study = cfg.get("experiment_name", "followup")
    raw_dir = args.raw or os.path.join(cfg["output_root"], "raw")
    out_dir = args.out or os.path.join(cfg["output_root"], "summaries")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[analyze-followup] study={study} raw={raw_dir} out={out_dir}")

    runs = load_runs(raw_dir, args.stages)
    if args.stages:
        runs = [d for d in runs if str(_val(d, "stage")) in args.stages]
    if not runs:
        print("[analyze-followup] no runs found")
        return 1
    is_frozen = any(str(_val(d, "mode", "sample")) == "frozen" for d in runs)
    print(f"[analyze-followup] loaded {len(runs)} runs (frozen={is_frozen})")

    if is_frozen:
        lut = load_online_lookup(args.online_summary)
        fsum = frozen_summary(runs, online_lookup=lut)
        write_csv(fsum, os.path.join(out_dir, "frozen_bias_summary.csv"))
        if not args.no_latex and fsum:
            latex_frozen(fsum, os.path.join(REPORT_TABLES, "wca_frozen_bias.tex"))
        print("[analyze-followup] done")
        return 0

    # sample studies
    metrics = ["l2_f", "l2_fp", "integrated_l2_f", "l2_f_transition",
               "marginal_l2_ref", "n_barrier_crossings", "fr_event_fraction",
               "max_fr_event_fraction", "final_ancestor_ess", "final_max_ancestor_frac",
               "fr_score_std", "fr_score_clip_fraction"]
    write_csv(sample_run_rows(runs), os.path.join(out_dir, f"{study}_run_summary.csv"))
    cfg_sum = config_summary(runs, metrics)
    matched = matched_seed_table(runs)

    if "equal_compute" in study:
        write_csv(cfg_sum, os.path.join(out_dir, "equal_compute_summary.csv"))
        if not args.no_latex and cfg_sum:
            latex_equal_compute(cfg_sum, os.path.join(REPORT_TABLES, "wca_equal_compute.tex"))
    else:  # representative (default)
        write_csv(cfg_sum, os.path.join(out_dir, "representative_cells_summary.csv"))
        write_csv(matched, os.path.join(out_dir, "matched_seed_table.csv"))
        alog = adaptive_event_log(runs)
        write_csv(alog, os.path.join(out_dir, "adaptive_fr_event_log.csv"))
        if not args.no_latex and matched:
            latex_representative(matched, os.path.join(REPORT_TABLES, "wca_representative_main.tex"))
            latex_adaptive_compare(matched, os.path.join(REPORT_TABLES, "wca_adaptive_compare.tex"))
    print("[analyze-followup] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
