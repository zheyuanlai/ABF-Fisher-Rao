"""Analysis of the NSCC A100 study (campaign.json 'analysis', 'selection', 'confirmation').

    python scripts/nscc_a100/analyze.py law         # V4 law check (GPU vs CPU, validation seeds)
    python scripts/nscc_a100/analyze.py prod        # exploration: per-N tables, best allocation, selection proposal
    python scripts/nscc_a100/analyze.py select      # freeze results/nscc_a100/selection/selection.json (once)
    python scripts/nscc_a100/analyze.py confirm     # confirmatory ratios, CIs, Holm, decision
Outputs under results/nscc_a100/analysis/ (summary JSON + tables.md).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nscc_common as NC  # noqa: E402

H = NC.H
import eqb_metrics as M  # noqa: E402

OUT = os.path.join(NC.OUT_ROOT, "analysis")
SEL = os.path.join(NC.OUT_ROOT, "selection", "selection.json")
N_BOOT = 10_000
BOOT_SEED = 20261026
THR = ("strict", "mid", "loose")


def load(kind, system=None):
    """Complete records of a kind; endpoint values converted from the JSON convention ("inf" strings) to floats."""
    recs = []
    pat = os.path.join(NC.OUT_ROOT, kind, system or "*", "*", "s*.json")
    for p in sorted(glob.glob(pat)):
        r = json.load(open(p))
        if r.get("status") == "complete" and "endpoints" in r:
            r["endpoints"] = {k: (NC.C.num(v) if isinstance(v, (int, float, str)) and k != "backend" else v)
                              for k, v in r["endpoints"].items()}
            recs.append(r)
    return recs


def med_inf(x):
    x = np.asarray(x, float)
    return float(M.quantile_inf(x, 50)) if x.size else math.nan


def boot_median_ratio(a, b, n_boot=N_BOOT, seed=BOOT_SEED, paired=False):
    """median(a) / median(b) with a percentile bootstrap CI (inf-aware); p-values for ratio < 1 and > 1."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    pt = ratio(med_inf(a), med_inf(b))
    R = np.empty(n_boot)
    for i in range(n_boot):
        if paired:
            idx = rng.integers(0, a.size, a.size)
            ra, rb = a[idx], b[idx]
        else:
            ra = a[rng.integers(0, a.size, a.size)]
            rb = b[rng.integers(0, b.size, b.size)]
        R[i] = ratio(med_inf(ra), med_inf(rb))
    fin = R[~np.isnan(R)]
    lo, hi = (np.percentile(fin, [2.5, 97.5]) if fin.size else (math.nan, math.nan))
    p_below = float(np.mean(fin >= 1.0)) if fin.size else math.nan     # one-sided p for "ratio < 1"
    p_above = float(np.mean(fin <= 1.0)) if fin.size else math.nan
    return dict(point=pt, ci95=[float(lo), float(hi)], p_ratio_below_1=p_below, p_ratio_above_1=p_above,
                p_two_sided=min(1.0, 2 * min(p_below, p_above)) if fin.size else math.nan,
                n_a=int(a.size), n_b=int(b.size), censored_a=int(np.sum(np.isinf(a))),
                censored_b=int(np.sum(np.isinf(b))), frac_undefined=float(np.mean(np.isnan(R))))


def ratio(x, y):
    if math.isinf(x) and math.isinf(y):
        return math.nan
    if math.isinf(y):
        return 0.0
    if math.isinf(x):
        return math.inf
    return x / y if y > 0 else math.nan


# ------------------------------------------------------------------------------------------------ law check
def cmd_law(a):
    cfg = NC.load_config()
    rows = []
    for system in NC.SYSTEMS:
        recs = load("law", system)
        by = {}
        for r in recs:
            by.setdefault((r["N"], r["method"], r["backend"] == "G1"), {})[r["seed"]] = r["endpoints"]
        for N in cfg["systems"][system]["law_N"]:
            for m in ("abf", "fr"):
                g = by.get((N, m, True), {})
                c = by.get((N, m, False), {})
                seeds = sorted(set(g) & set(c))
                if not seeds:
                    continue
                row = dict(system=system, N=N, method=m, n_pairs=len(seeds))
                ok = True
                for key in ("final_e_F", "Ibar_F", "u_mid"):
                    xa = np.array([g[s][key] for s in seeds], float)
                    xb = np.array([c[s][key] for s in seeds], float)
                    st = boot_median_ratio(xa, xb, paired=True)
                    row[key] = st
                    row[key + "_identical_frac"] = float(np.mean(xa == xb))
                    row[key + "_max_rel_paired"] = float(np.nanmax(np.abs(xa - xb) / np.maximum(np.abs(xb), 1e-300))
                                                         if np.all(np.isfinite(xa) | np.isinf(xb)) else math.nan)
                    lo, hi = st["ci95"]
                    contains = (lo <= 1.0 <= hi) if math.isfinite(lo) else True
                    inside = (0.8 <= lo and hi <= 1.25) if math.isfinite(lo) else False
                    ok &= bool(contains or inside)
                row["passed"] = ok
                rows.append(row)
                print(f"LAW {system} {m}{N}: pairs {len(seeds)}  " + "  ".join(
                    f"{k} {row[k]['point']:.3f} [{row[k]['ci95'][0]:.3f}, {row[k]['ci95'][1]:.3f}] "
                    f"(identical {row[k + '_identical_frac']:.2f})" for k in ("final_e_F", "Ibar_F", "u_mid")) +
                    f"  -> {'PASS' if ok else 'FAIL'}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    out = dict(rows=rows, passed=all(r["passed"] for r in rows) if rows else None,
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(OUT, "law_check.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("LAW_VERDICT", "PASS" if out["passed"] else "FAIL")


# ------------------------------------------------------------------------------------------------ exploration
def table(recs, key):
    """{N: {method: {seed: value}}}"""
    t = {}
    for r in recs:
        t.setdefault(int(r["N"]), {}).setdefault(r["method"], {})[int(r["seed"])] = float(r["endpoints"][key])
    return t


def cmd_prod(a, write=True):
    S = {}
    for system in NC.SYSTEMS:
        recs = load("prod", system)
        if not recs:
            continue
        spec = H.system_spec(system)
        Ns = sorted({int(r["N"]) for r in recs})
        per = {}
        for N in Ns:
            for m in ("abf", "fr"):
                rs = [r for r in recs if r["N"] == N and r["method"] == m]
                if not rs:
                    continue
                e = {k: [r["endpoints"][k] for r in rs] for k in ("W_mid", "W_mid_e2e", "W_loose", "W_strict",
                                                                    "final_e_F", "Ibar_F", "start_overhead_s",
                                                                    "completion_wall_s")}
                per[f"{m}{N}"] = dict(N=N, method=m, backend=rs[0]["backend"], n=len(rs),
                                      **{f"median_{k}": med_inf(v) for k, v in e.items()},
                                      censored_mid=int(np.sum(np.isinf(e["W_mid"]))))
        best = {}
        for key in ("W_mid", "W_mid_e2e", "W_loose", "W_strict"):
            T = table(recs, key)
            Nab = [N for N in Ns if "abf" in T.get(N, {})]
            Nfr = [N for N in Ns if "fr" in T.get(N, {})]
            b = M.best_allocation(T, Nab, Nfr, n_boot=N_BOOT, seed=BOOT_SEED)
            if b.get("abf") and b.get("fr") and "fr_minus_abf" in b:
                d = b["fr_minus_abf"]
                b["R_wall"] = dict(point=ratio(b["fr"]["best_value"], b["abf"]["best_value"]),
                                   ci95=[1.0 + d["rel_ci95"][0], 1.0 + d["rel_ci95"][1]],
                                   frac_resamples_fr_better=d["frac_resamples_fr_better"])
            best[key] = b
        S[system] = dict(per=per, best=best, n_runs=len(recs), thresholds=spec["thresholds"])
        b = best["W_mid"]
        print(f"{system}: runs {len(recs)}; best ABF {b['abf'] and b['abf']['best_N']} "
              f"({b['abf'] and b['abf']['best_value']}); best FR {b['fr'] and b['fr']['best_N']} "
              f"({b['fr'] and b['fr']['best_value']})", flush=True)
    if write:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "prod_summary.json"), "w") as f:
            json.dump(dict(systems=S, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), f, indent=1,
                      default=str)
    return S


def cmd_select(a):
    if os.path.exists(SEL):
        raise SystemExit(f"{SEL} exists (frozen)")
    S = cmd_prod(a, write=True)
    sel = dict(rule=NC.load_config()["selection"], systems={},
               frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    for system, s in S.items():
        out = {}
        for m in ("abf", "fr"):
            cand = {k: v for k, v in s["per"].items() if v["method"] == m}
            if not cand:
                out[m] = None
                continue
            k, v = min(cand.items(), key=lambda kv: (kv[1]["median_W_mid"], kv[1]["N"]))
            out[m] = None if not math.isfinite(v["median_W_mid"]) else dict(
                N=v["N"], backend=v["backend"], median_W_mid=v["median_W_mid"], arm=k)
        sel["systems"][system] = out
    os.makedirs(os.path.dirname(SEL), exist_ok=True)
    with open(SEL, "w") as f:
        json.dump(sel, f, indent=1)
    print(json.dumps(sel["systems"], indent=1))


def cmd_confirm(a):
    sel = json.load(open(SEL))
    res = {}
    for system in NC.SYSTEMS:
        recs = load("confirm", system)
        s = sel["systems"][system]
        if not recs or s["abf"] is None or s["fr"] is None:
            continue
        arms = {m: [r for r in recs if r["method"] == m and r["N"] == s[m]["N"] and r["backend"] == s[m]["backend"]]
                for m in ("abf", "fr")}
        out = dict(selection=s, n={m: len(v) for m, v in arms.items()})
        for key in ("W_mid", "W_mid_e2e", "W_loose", "W_strict", "W_loose_e2e", "W_strict_e2e"):
            fa = [r["endpoints"][key] for r in arms["fr"]]
            ab = [r["endpoints"][key] for r in arms["abf"]]
            out[key] = dict(median_fr=med_inf(fa), median_abf=med_inf(ab), R=boot_median_ratio(fa, ab))
        for key in ("final_e_F", "Ibar_F"):
            out[key] = dict(median_fr=med_inf([r["endpoints"][key] for r in arms["fr"]]),
                            median_abf=med_inf([r["endpoints"][key] for r in arms["abf"]]))
        rw, re = out["W_mid"]["R"], out["W_mid_e2e"]["R"]
        below = rw["ci95"][1] < 1 and re["ci95"][1] < 1
        above = rw["ci95"][0] > 1 and re["ci95"][0] > 1
        out["decision_unadjusted"] = "FR faster" if below else ("ABF faster" if above else "unresolved")
        out["p_system"] = max(rw["p_two_sided"], re["p_two_sided"])
        res[system] = out
    # Holm over the two primary systems
    prim = [s for s in ("gateway", "lta150") if s in res]
    order = sorted(prim, key=lambda s: res[s]["p_system"])
    stop = False
    for i, s in enumerate(order):
        alpha = 0.05 / (len(order) - i)
        sig = (not stop) and res[s]["p_system"] <= alpha
        stop = stop or not sig
        res[s]["holm_alpha"] = alpha
        res[s]["holm_significant"] = bool(sig)
        res[s]["decision"] = res[s]["decision_unadjusted"] if sig else "unresolved"
    if "lta300" in res:
        res["lta300"]["decision"] = res["lta300"]["decision_unadjusted"] + " (secondary)"
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "confirm_summary.json"), "w") as f:
        json.dump(dict(systems=res, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), f, indent=1,
                  default=str)
    for s, o in res.items():
        print(f"{s}: R_W {o['W_mid']['R']['point']:.3f} {o['W_mid']['R']['ci95']}  R_e2e "
              f"{o['W_mid_e2e']['R']['point']:.3f} {o['W_mid_e2e']['R']['ci95']}  -> {o['decision']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("law", "prod", "select", "confirm"))
    a = ap.parse_args(argv)
    {"law": cmd_law, "prod": cmd_prod, "select": cmd_select, "confirm": cmd_confirm}[a.cmd](a)


if __name__ == "__main__":
    main()
