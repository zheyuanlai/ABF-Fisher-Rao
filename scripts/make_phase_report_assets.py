#!/usr/bin/env python3
"""Turn WCA phase-diagram summary CSVs into report assets.

Writes, under report/tables/ (override with --tabledir):
  wca_phase_main.tex      the per-cell main LaTeX table (table environment)
  wca_phase_numbers.tex   \newcommand macros for in-text numbers (prefix WCAPD)
  wca_phase_numbers.json  the same numbers, machine-readable

This script is intentionally separate from the existing checked-number pipeline,
and uses only csv/numpy so it runs in the sampler environment without pandas.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

import numpy as np


def _parse(v: str):
    if v == "":
        return np.nan
    low = v.lower()
    if low == "nan":
        return np.nan
    if low == "inf":
        return np.inf
    if low == "-inf":
        return -np.inf
    try:
        return float(v)
    except ValueError:
        return v


def read_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [{k: _parse(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def load_many(summary_dirs: list[str], name: str) -> list[dict]:
    rows, seen = [], set()
    for d in summary_dirs:
        for r in read_rows(os.path.join(d, name)):
            key = tuple(sorted(r.items()))
            if key not in seen:
                seen.add(key); rows.append(r)
    return rows


def finite(vals):
    arr = np.asarray([float(v) for v in vals if isinstance(v, (int, float, np.floating))], dtype=float)
    return arr[np.isfinite(arr)]


def mean(vals):
    arr = finite(vals)
    return float(np.mean(arr)) if arr.size else np.nan


def median(vals):
    arr = finite(vals)
    return float(np.median(arr)) if arr.size else np.nan


def mode(rows, key):
    vals = [r[key] for r in rows if key in r and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return Counter(vals).most_common(1)[0][0] if vals else np.nan


def row_is(row, **conds):
    for k, v in conds.items():
        if k not in row:
            return False
        if isinstance(v, (int, float, np.floating)):
            if not np.isclose(float(row[k]), float(v)):
                return False
        elif row[k] != v:
            return False
    return True


def _fmt(x, nd=4):
    if x is None or not isinstance(x, (int, float, np.floating)) or not np.isfinite(float(x)):
        return "--"
    return f"{float(x):.{nd}f}"


def _cell_label(beta, h):
    return f"$\\beta{{=}}{float(beta):g},\\,h{{=}}{float(h):g}$"


def build_numbers(main, impr, gen, cfg_summary):
    n = {}
    M0 = int(mode(main, "M"))
    plane = sorted([r for r in main if row_is(r, M=M0)], key=lambda r: float(r["beta"]) * float(r["h"]))
    n["nCells"] = int(len(main))
    n["nPlaneCells"] = int(len(plane))
    n["modalM"] = M0
    n["nSeeds"] = int(max(r["n_seeds"] for r in main))
    n["nSteps"] = int(max(r["n_steps"] for r in main))

    g = [r for r in plane if np.isfinite(float(r.get("median_gain_pct", np.nan)))]
    gains = [r["median_gain_pct"] for r in g]
    n["meanGain"] = mean(gains)
    n["medianGain"] = median(gains)
    n["nWinCells"] = int(sum(float(x) > 0 for x in gains))
    n["nLoseCells"] = int(sum(float(x) <= 0 for x in gains))
    if g:
        best = max(g, key=lambda r: float(r["median_gain_pct"]))
        worst = min(g, key=lambda r: float(r["median_gain_pct"]))
        n["maxGain"] = float(best["median_gain_pct"])
        n["maxGainCell"] = _cell_label(best["beta"], best["h"])
        n["maxGainR"] = float(best["R_est"])
        n["minGain"] = float(worst["median_gain_pct"])
        n["minGainCell"] = _cell_label(worst["beta"], worst["h"])

    anc = [r for r in plane if row_is(r, beta=1.0, h=2.0)]
    if anc:
        a = anc[0]
        n["anchorGain"] = float(a["median_gain_pct"])
        n["anchorR"] = float(a["R_est"])
        n["anchorAbfErr"] = float(a["abf_l2_f"])
        n["anchorFrErr"] = float(a["fr_est_l2_f"])

    if plane:
        easiest, hardest = plane[0], plane[-1]
        for prefix, r in [("easiest", easiest), ("hardest", hardest)]:
            n[f"{prefix}Cell"] = _cell_label(r["beta"], r["h"])
            n[f"{prefix}Bh"] = float(r["beta"]) * float(r["h"])
            n[f"{prefix}AbfErr"] = float(r["abf_l2_f"])
        n["easiestGain"] = float(easiest["median_gain_pct"])
        n["hardestFrErr"] = float(hardest["fr_est_l2_f"])
        n["hardestGain"] = float(hardest["median_gain_pct"])
    if len(g) >= 3:
        bh = np.asarray([float(r["beta"]) * float(r["h"]) for r in g], dtype=float)
        ga = np.asarray([float(r["median_gain_pct"]) for r in g], dtype=float)
        n["corrGainBh"] = float(np.corrcoef(bh, ga)[0, 1])

    if impr:
        by_tag = {}
        for r in impr:
            by_tag.setdefault(r["physics_tag"], {})[r["method"]] = r
        diffs_uni, diffs_ora = [], []
        for methods in by_tag.values():
            est = methods.get("fr_estimated")
            if est and methods.get("fr_uniform"):
                diffs_uni.append(abs(float(est["median_gain_pct_F"]) - float(methods["fr_uniform"]["median_gain_pct_F"])))
            if est and methods.get("fr_oracle"):
                diffs_ora.append(abs(float(est["median_gain_pct_F"]) - float(methods["fr_oracle"]["median_gain_pct_F"])))
        if diffs_uni:
            n["meanAbsUniMinusEst"] = mean(diffs_uni)
        if diffs_ora:
            n["meanAbsOraMinusEst"] = mean(diffs_ora)

    ge = [r for r in gen if r.get("method") == "fr_estimated"]
    if ge:
        n["minEss"] = float(min(r["min_ancestor_ess"] for r in ge))
        n["minFinalEss"] = float(min(r["final_ancestor_ess"] for r in ge))
        n["maxAncFrac"] = float(max(r["max_ancestor_frac_over_time"] for r in ge))
        n["meanFinalEss"] = mean([r["final_ancestor_ess"] for r in ge])

    masses = sorted({int(r["M"]) for r in main})
    if len(masses) > 1:
        n["hasMassAxis"] = 1
        n["massList"] = ",".join(str(m) for m in masses)
    return n


def write_macros(n, path):
    def esc(v):
        if isinstance(v, str):
            return v
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if abs(v) >= 100:
                return f"{v:.0f}"
            return f"{v:.3g}" if abs(v) < 1 else f"{v:.2f}"
        return str(v)
    lines = ["% Auto-generated by scripts/make_phase_report_assets.py. Do not edit by hand.",
             "% Macros for the WCA phase-diagram subsection (prefix WCAPD)."]
    for k, v in n.items():
        lines.append(f"\\newcommand{{\\WCAPD{k[0].upper()+k[1:]}}}{{{esc(v)}}}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {path} ({len(n)} macros)")


def write_main_table(main, path):
    M0 = int(mode(main, "M"))
    plane = sorted([r for r in main if row_is(r, M=M0)], key=lambda r: (float(r["beta"]), float(r["h"])))
    rows = []
    for r in plane:
        gain = float(r["median_gain_pct"])
        gain_s = ("$+%.1f$" % gain) if np.isfinite(gain) and gain >= 0 else ("$%.1f$" % gain)
        bh = float(r["beta"]) * float(r["h"])
        rows.append(
            f"{float(r['beta']):g} & {float(r['h']):g} & {bh:g} & "
            f"{_fmt(r['abf_l2_f'])} & {_fmt(r['fr_est_l2_f'])} & "
            f"{_fmt(r.get('fr_uniform_l2_f', np.nan))} & {_fmt(r.get('fr_oracle_l2_f', np.nan))} & "
            f"{_fmt(r['R_est'], 2)} & {gain_s} & {int(r['n_wins'])}/{int(r['n_seeds'])} & "
            f"{_fmt(r.get('fr_est_final_ess', np.nan), 0)} \\\\")
    body = "\n".join(rows)
    nsteps_str = "{:,}".format(int(plane[0]["n_steps"])).replace(",", "{,}") if plane else "--"
    tex = f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{WCA phase diagram over $(\\beta,h)$ at $M={M0}$ physical particles
($N=1024$ replicas, ${nsteps_str}$ steps).
$R=L^2(F)_{{\\rm ABF}}/L^2(F)_{{\\rm mFR\\,est}}$; gain is the matched-seed median
percentage reduction in final $L^2(F)$. The oracle column is a diagnostic that
uses the TI reference and is \\emph{{not}} deployable.}}
\\label{{tab:wca_phase_main}}
\\begin{{tabular}}{{rrr rrrr rr rr}}
\\toprule
$\\beta$ & $h$ & $\\beta h$ & ABF & mFR est & mFR uni & mFR ora$^{{*}}$ & $R$ & gain (\\%) & wins & ESS \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\par\\smallskip\\footnotesize $^{{*}}$\\,Diagnostic oracle target (uses $F_{{\\rm ref}}$); not deployable.
\\end{{table}}
"""
    with open(path, "w") as fh:
        fh.write(tex)
    print(f"  wrote {path} ({len(rows)} cells)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", required=True, nargs="+",
                    help="one or more summaries dirs (merged; e.g. production main + mass)")
    ap.add_argument("--tabledir", default="report/tables")
    args = ap.parse_args(argv)

    main_t = load_many(args.summaries, "phase_main_table.csv")
    impr = load_many(args.summaries, "phase_improvement_ratios.csv")
    gen = load_many(args.summaries, "phase_genealogy.csv")
    cfg_s = load_many(args.summaries, "phase_config_summary.csv")
    if not main_t:
        print("no phase_main_table.csv found")
        return 1
    os.makedirs(args.tabledir, exist_ok=True)
    n = build_numbers(main_t, impr, gen, cfg_s)
    write_macros(n, os.path.join(args.tabledir, "wca_phase_numbers.tex"))
    with open(os.path.join(args.tabledir, "wca_phase_numbers.json"), "w") as fh:
        json.dump(n, fh, indent=2)
    write_main_table(main_t, os.path.join(args.tabledir, "wca_phase_main.tex"))
    print("[assets] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
