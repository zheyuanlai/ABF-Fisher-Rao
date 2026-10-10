#!/usr/bin/env python
"""Resource ledger of the equal-budget campaign (docs/equal_budget/EXPERIMENT_LOG.md).

    python scripts/equal_budget/ledger_summary.py [--json results/equal_budget_v2/ledger_summary.json]

Sums the per-run production ledgers (results/equal_budget_v2/<dir>/ledger.csv: one row per finished arm, wall_s
of one single-threaded process = core-seconds) and the gateway timestep-validation log
(results/equal_budget_v2/gateway_validation/run.log, "<k>/<n>] <job>: <s> s").  Prints a markdown table.
Engine tests, smoke tests, reference building and the analysis/figure scripts are not in these ledgers; they are
listed separately in EXPERIMENT_LOG.md as measured or bounded estimates.
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BASE = os.path.join(ROOT, "results", "equal_budget_v2")
DIRS = {"lta300": "lta_T300", "lta150": "lta_T150", "gateway": "gateway"}


def production():
    rows = {}
    for sysk, d in DIRS.items():
        p = os.path.join(BASE, d, "ledger.csv")
        if not os.path.exists(p):
            continue
        last = {}
        for r in csv.DictReader(open(p)):
            last[(r["N"], r["seed"], r["method"])] = r  # a re-run of a job supersedes its earlier row
        agg = defaultdict(float)
        agg["runs"] = len(last)
        agg["failed"] = sum(not r["status"].startswith("complete") for r in last.values())
        agg["core_s"] = sum(float(r["wall_s"]) for r in last.values())
        agg["core_s_all_rows"] = sum(float(r["wall_s"]) for r in csv.DictReader(open(p)))
        agg["force_evals"] = sum(max(int(r["n_force_evals"]), 0) for r in last.values())
        agg["max_rss_mb"] = max((float(r["peak_rss_mb"]) for r in last.values() if r["peak_rss_mb"] != "nan"), default=float("nan"))
        ts = sorted(r["finished_utc"] for r in last.values())
        agg["first_finished"], agg["last_finished"] = ts[0], ts[-1]
        rows[sysk] = dict(agg)
    return rows


def validation():
    p = os.path.join(BASE, "gateway_validation", "run.log")
    if not os.path.exists(p):
        return None
    s = [float(m.group(1)) for m in re.finditer(r": ([0-9.]+) s$", open(p).read(), re.M)]
    return dict(runs=len(s), core_s=sum(s))


def inventory():
    """Per (system, N, method): seeds completed / planned, walker-steps, measured core-h (FINAL_RESULTS section 3)."""
    CFG = {"lta300": ("lta_300K_production.json", "ethane/LTA", "300 K"), "lta150": ("lta_150K_production.json", "ethane/LTA", "150 K"),
           "gateway": ("gateway_production.json", "entropic gateway", "--")}
    print("| system | T | N | T_N (t.u.) | h | walker-steps per run | method | seeds completed | status | core-h (sum) | wall per run (median s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for sysk, d in DIRS.items():
        P = json.load(open(os.path.join(ROOT, "configs", "equal_budget_v2", CFG[sysk][0])))
        p = os.path.join(BASE, d, "ledger.csv")
        last = {}
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                last[(int(r["N"]), r["seed"], r["method"])] = r
        B, h = int(P["B"]), float(P["engine_cfg"]["h"])
        for N in P["N_ladder"]:
            for m in (("abf",) if N == 1 else ("abf", "fr")):
                rr = [r for (n, s, mm), r in last.items() if n == N and mm == m]
                ok = [r for r in rr if r["status"].startswith("complete")
                      and os.path.exists(os.path.join(BASE, d, f"N{N}", f"s{r['seed']}_{m}.npz"))]
                ns = len(P["seeds"])
                st = "complete" if len(ok) == ns else ("NOT RUN" if not ok else f"partial ({len(ok)}/{ns})")
                w = sorted(float(r["wall_s"]) for r in ok)
                med = w[len(w) // 2] if w else float("nan")
                print(f"| {CFG[sysk][1]} | {CFG[sysk][2]} | {N} | {B * h / N:g} | {h:g} | {B:.4g} | {'ABF' if m == 'abf' else 'ABF+FR'} | "
                      f"{len(ok)}/{ns} | {st} | {sum(w) / 3600:.2f} | {med:.0f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--inventory", action="store_true", help="print the per-(system, N, method) inventory table instead")
    a = ap.parse_args()
    if a.inventory:
        inventory()
        return
    P, V = production(), validation()
    out = dict(production=P, gateway_validation=V)
    print("| item | runs | failed | core-h | force evaluations | max RSS (MB) | finished (UTC) |")
    print("|---|---|---|---|---|---|---|")
    tot = 0.0
    if V:
        print(f"| gateway timestep validation | {V['runs']} | 0 | {V['core_s'] / 3600:.2f} | -- | -- | -- |")
        tot += V["core_s"]
    for k, r in P.items():
        print(f"| {k} production | {r['runs']:.0f} | {r['failed']:.0f} | {r['core_s'] / 3600:.2f} | {r['force_evals']:.4g} | "
              f"{r['max_rss_mb']:.0f} | {r['first_finished']} .. {r['last_finished']} |")
        tot += r["core_s"]
    print(f"| **total (ledgered)** | | | **{tot / 3600:.1f}** | | | |")
    out["total_core_h"] = tot / 3600
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
