#!/usr/bin/env python
"""Apply the frozen WCA sham-control rule.

The decision rule lives in ``results/wca_sham/PREREGISTRATION.md`` and is applied here
mechanically.  The statistic that decides attribution is the **direct** paired contrast of
each FR arm against its own sham: it holds the event schedule and the replacement count fixed
by construction, so the only difference is the selection direction.

    python scripts/analyze_wca_sham.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

# Frozen in the preregistration.
PRIMARY = "fr_estimated"
METRIC = "integrated_l2_f"
RULE = dict(median_pct_max=-10.0, ci95_upper_max=-5.0, min_seeds=12, n_seeds=16,
            direct_median_max=-5.0, ess_frac_min=0.10, wmax_max=0.05)
MARGIN = (-5.0, 5.0)
BOOT_SEED, N_BOOT = 20260803, 10_000
PARTNER = {"sham_practical": "fr_estimated", "sham_oracle": "fr_oracle"}
SCALARS = ("integrated_l2_f", "l2_f", "l2_fp", "n_round_trips", "n_barrier_crossings",
           "final_ancestor_ess", "min_ancestor_ess", "max_ancestor_frac_over_time",
           "total_replacement_events", "n_replicas", "frac_stretched_final")


def boot(x, rng, n=N_BOOT):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([np.nan])
    return np.median(x[rng.integers(0, x.size, size=(n, x.size))], axis=1)


def ci(dist, level):
    lo = 50.0 * (1.0 - level)
    return float(np.percentile(dist, lo)), float(np.percentile(dist, 100.0 - lo))


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    out = np.full(a.shape, np.nan)
    out[ok] = 100.0 * (a[ok] - b[ok]) / b[ok]
    return out


def load(raw_dir):
    runs = {}
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            def g(k, default=np.nan):
                if k not in z.files:
                    return default
                v = z[k]
                return v.item() if getattr(v, "ndim", 1) == 0 else v
            rec = {k: g(k) for k in SCALARS if k != "frac_stretched_final"}
            fs = g("frac_stretched", None)
            rec["frac_stretched_final"] = (float(np.asarray(fs)[-1])
                                           if fs is not None and np.size(fs) else np.nan)
            rec["method"] = str(g("method", ""))
            rec["name"] = str(g("name", ""))
            rec["seed"] = int(g("seed", -1))
            rec["had_nan"] = bool(g("had_nan", False))
            rec["fr_event_total"] = int(np.asarray(g("fr_event_counts", np.zeros(1))).sum())
            runs.setdefault(rec["name"], {})[rec["seed"]] = rec
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results/wca_sham/sham"))
    a = ap.parse_args()
    raw = os.path.join(a.dir, "raw")
    runs = load(raw)
    if not runs:
        raise SystemExit(f"no runs under {raw}")
    rng = np.random.default_rng(BOOT_SEED)
    arms = [x for x in ("abf", "fr_estimated", "sham_practical", "fr_oracle", "sham_oracle")
            if x in runs]
    seeds = sorted(set.intersection(*[set(runs[x]) for x in arms]))
    # A partial run must NOT produce a verdict. The seed-count criterion is absolute
    # (">= 12 of 16"), so on 5 complete seeds it fails mechanically and the decision block
    # would announce NOT REPLICATED for a run that is simply unfinished -- a confident wrong
    # answer of exactly the kind this project keeps having to catch. Interim numbers are
    # still printed, because watching a run is legitimate; the verdict is withheld.
    complete = len(seeds) >= RULE["n_seeds"]
    print(f"WCA matched-sham control -- {len(seeds)} complete seeds "
          f"{min(seeds)}-{max(seeds)}, arms {arms}")
    if not complete:
        print(f"\n*** INTERIM: {len(seeds)} of {RULE['n_seeds']} preregistered seeds are "
              f"complete. Numbers below are provisional and NO VERDICT IS ISSUED. ***")
    nan_runs = [(m, s) for m in arms for s in seeds if runs[m][s]["had_nan"]]
    print(f"  runs flagged had_nan: {len(nan_runs)}"
          + (f"  {nan_runs}" if nan_runs else ""))

    # ---------------------------------------------------- sham intensity match
    bad = [(sh, s) for sh, pa in PARTNER.items() if sh in runs and pa in runs
           for s in seeds
           if runs[sh][s]["total_replacement_events"] != runs[pa][s]["total_replacement_events"]]
    print(f"  sham intensity mismatches: {len(bad)}" + (f"  {bad}" if bad else "")
          + "  (each sham vs its own partner)\n")

    # ------------------------------------------------------- vs ABF, per arm
    rows = []
    for arm in arms:
        if arm == "abf":
            continue
        v = np.array([runs[arm][s][METRIC] for s in seeds], float)
        b = np.array([runs["abf"][s][METRIC] for s in seeds], float)
        r = rel(v, b)
        dist = boot(r, rng)
        lo95, hi95 = ci(dist, 0.95)
        lo90, hi90 = ci(dist, 0.90)
        ess = np.array([runs[arm][s]["min_ancestor_ess"] for s in seeds], float)
        maf = np.array([runs[arm][s]["max_ancestor_frac_over_time"] for s in seeds], float)
        n_rep = np.array([runs[arm][s]["n_replicas"] for s in seeds], float)
        rows.append(dict(arm=arm, n_seeds=len(seeds), pct=float(np.nanmedian(r)),
                         ci95=[lo95, hi95], ci90=[lo90, hi90],
                         wins=int(np.sum(v < b)),
                         median_metric=float(np.nanmedian(v)),
                         abf_metric=float(np.nanmedian(b)),
                         ess_frac=float(np.nanmedian(ess / n_rep)),
                         max_anc_frac=float(np.nanmedian(maf)),
                         round_trips=float(np.nanmedian(
                             [runs[arm][s]["n_round_trips"] for s in seeds])),
                         repl=float(np.nanmedian(
                             [runs[arm][s]["total_replacement_events"] for s in seeds]))))
    print(f"vs ABF on {METRIC} (paired, negative = better)")
    print(f"{'arm':>16s} {'pct':>8s} {'95% CI':>18s} {'90% CI':>18s} {'won':>6s} "
          f"{'ESS/N':>7s} {'wmaxA':>7s} {'rtrips':>7s} {'repl':>7s}")
    print(f"{'abf':>16s} {'--':>8s} {'--':>18s} {'--':>18s} {'--':>6s} "
          f"{'--':>7s} {'--':>7s} "
          f"{np.nanmedian([runs['abf'][s]['n_round_trips'] for s in seeds]):7.0f} {0:7.0f}")
    for r in rows:
        print(f"{r['arm']:>16s} {r['pct']:8.2f} "
              f"[{r['ci95'][0]:7.2f},{r['ci95'][1]:7.2f}] "
              f"[{r['ci90'][0]:7.2f},{r['ci90'][1]:7.2f}] "
              f"{r['wins']:3d}/{r['n_seeds']:<2d} {r['ess_frac']:7.3f} "
              f"{r['max_anc_frac']:7.3f} {r['round_trips']:7.0f} {r['repl']:7.0f}")

    # ------------------------------------------- direct arm vs its own sham
    print("\nDIRECT contrast -- each FR arm against ITS OWN sham (same seed, same event "
          "schedule)\nthis is the attribution test: only the selection direction differs")
    direct = {}
    for sh, pa in PARTNER.items():
        if sh not in runs or pa not in runs:
            continue
        va = np.array([runs[pa][s][METRIC] for s in seeds], float)
        vs = np.array([runs[sh][s][METRIC] for s in seeds], float)
        r = rel(va, vs)
        lo95, hi95 = ci(boot(r, rng), 0.95)
        direct[pa] = dict(vs=sh, pct=float(np.nanmedian(r)), ci95=[lo95, hi95],
                          wins=int(np.sum(va < vs)), n_seeds=len(seeds),
                          excludes_zero=bool(lo95 * hi95 > 0))
        print(f"  {pa:>14s} vs {sh:<16s} {direct[pa]['pct']:+7.2f}% "
              f"[{lo95:+6.2f},{hi95:+6.2f}]  {direct[pa]['wins']:2d}/{len(seeds)}  "
              f"{'CI EXCLUDES zero' if direct[pa]['excludes_zero'] else 'CI includes zero'}")

    # ------------------------------------------------------------ decision
    prim = next(r for r in rows if r["arm"] == PRIMARY)
    dpr = direct.get(PRIMARY, {})
    checks = {
        f"median <= {RULE['median_pct_max']:g}%": (prim["pct"] <= RULE["median_pct_max"],
                                                   prim["pct"]),
        f"CI95 upper < {RULE['ci95_upper_max']:g}%": (prim["ci95"][1] < RULE["ci95_upper_max"],
                                                      prim["ci95"][1]),
        f"seeds improved >= {RULE['min_seeds']} of {RULE['n_seeds']}":
            (prim["wins"] >= RULE["min_seeds"], prim["wins"]),
        f"min ESS/N >= {RULE['ess_frac_min']:g}": (prim["ess_frac"] >= RULE["ess_frac_min"],
                                                   prim["ess_frac"]),
        f"max anc frac <= {RULE['wmax_max']:g}": (prim["max_anc_frac"] <= RULE["wmax_max"],
                                                  prim["max_anc_frac"]),
    }
    print(f"\n{'=' * 96}\nCRITERION 1 -- {PRIMARY} beats ABF")
    for k, (ok, v) in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k:<26s} measured {v:.4g}")
    c1 = all(ok for ok, _ in checks.values())
    print(f"  => criterion 1: {'PASS' if c1 else 'FAIL'}")

    c2 = bool(dpr) and dpr["excludes_zero"] and dpr["pct"] <= RULE["direct_median_max"]
    print(f"\nCRITERION 2 -- attribution: {PRIMARY} vs sham_practical directly")
    if dpr:
        print(f"  [{'PASS' if dpr['excludes_zero'] else 'FAIL'}] 95% CI excludes zero"
              f"        measured [{dpr['ci95'][0]:.2f}, {dpr['ci95'][1]:.2f}]")
        print(f"  [{'PASS' if dpr['pct'] <= RULE['direct_median_max'] else 'FAIL'}] "
              f"median <= {RULE['direct_median_max']:g}%          measured {dpr['pct']:.2f}")
    print(f"  => criterion 2: {'PASS' if c2 else 'FAIL'}")

    print(f"\nCRITERION 3 (secondary) -- sham equivalence, TOST margin "
          f"[{MARGIN[0]:g}%, {MARGIN[1]:g}%], 90% CI inside")
    tost = {}
    for sh in PARTNER:
        r = next((x for x in rows if x["arm"] == sh), None)
        if r is None:
            continue
        lo, hi = r["ci90"]
        ok = (lo >= MARGIN[0]) and (hi <= MARGIN[1])
        tost[sh] = dict(equivalent=bool(ok), ci90=[lo, hi], median=r["pct"])
        print(f"  [{'PASS' if ok else 'FAIL'}] {sh:<16s} median {r['pct']:+6.2f}%  "
              f"90% CI [{lo:+6.2f},{hi:+6.2f}]")

    print(f"\n{'=' * 96}")
    if not complete:
        verdict = (f"NO VERDICT -- {len(seeds)}/{RULE['n_seeds']} seeds complete. The "
                   f"seed-count criterion cannot be met by an unfinished run, so the "
                   f"decision rule is not applied.")
        print(f"VERDICT: {verdict}")
        out = dict(seeds=seeds, arms=arms, metric=METRIC, rule=RULE, margin=MARGIN,
                   complete=False, vs_abf=rows, direct=direct, tost=tost,
                   criterion_1=None, criterion_2=bool(c2), verdict=verdict,
                   sham_mismatches=len(bad), nan_runs=len(nan_runs))
        with open(os.path.join(a.dir, "sham_summary.json"), "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"\nwrote {a.dir}/sham_summary.json (INTERIM -- complete=false)")
        return
    if c1 and c2:
        verdict = ("TRANSFERS -- the directional Fisher-Rao mechanism beats matched turnover "
                   "on a many-body molecular system")
    elif c1:
        verdict = ("DOES NOT TRANSFER -- mFR beats ABF but not its matched sham; the "
                   "molecular gain is not attributable to the Fisher-Rao direction")
    else:
        verdict = ("NOT REPLICATED -- the accepted WCA positive did not reproduce on fresh "
                   "seeds under this protocol")
    print(f"VERDICT: {verdict}")

    out = dict(seeds=seeds, arms=arms, metric=METRIC, rule=RULE, margin=MARGIN,
               complete=True, vs_abf=rows, direct=direct, tost=tost,
               criterion_1=bool(c1), criterion_2=bool(c2), verdict=verdict,
               sham_mismatches=len(bad), nan_runs=len(nan_runs))
    with open(os.path.join(a.dir, "sham_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    flat = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}
            for r in rows]
    with open(os.path.join(a.dir, "sham_comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader(); w.writerows(flat)
    print(f"\nwrote {a.dir}/sham_summary.json and sham_comparison.csv")


if __name__ == "__main__":
    main()
