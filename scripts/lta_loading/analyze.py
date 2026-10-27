"""Scoring and decisions of the finite-loading LTA study (docs/lta_loading/PREREGISTRATION.md §6-8).

e_F per save: the equal-budget LTA scorer convention (eqb_metrics.lta_estimator / lta_errors_from) --
Gamma = M / max(C, 1) from the production accumulators once they hold counts (else all-steps), F =
histogram_pmf(Gamma), RMS over all 180 bins after removing the mean difference -- against this study's
reference (results/lta_loading/reference/reference.npz).

  b_thr   walker-steps N k at the first save after which e_F <= thr at every later save (inf if censored)
  I_F     time-normalised integral of e_F over the run (trapezoid in u = k / n_steps)

Usage:
  python scripts/lta_loading/analyze.py pilot                   # §5: the budget B from ABF-only pilot runs
  python scripts/lta_loading/analyze.py ladder                  # §6: exploration table + frozen selection
  python scripts/lta_loading/analyze.py confirm                 # §7: confirmation decision
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import lta_ladder_numba as E  # noqa: E402
from lta_loading import engine as LE  # noqa: E402

RES = os.path.join(ROOT, "results", "lta_loading")
THR = dict(strict=0.03, mid=0.13, loose=0.5)        # kJ/mol, inherited from the single-guest 150 K study
N_BOOT = 10_000
BOOT_SEED = 20261027


def reference():
    z = np.load(os.path.join(RES, "reference", "reference.npz"))
    return z["F_ref"].astype(float), z["gamma_ref"].astype(float)


def errors(out, F_ref):
    """e_F (S, R) of a batch npz."""
    acc = out["sv_acc"]                                   # (S, R, 4, ng)
    M_all, C_all, M_p, C_p = acc[:, :, 0], acc[:, :, 1], acc[:, :, 2], acc[:, :, 3]
    use_p = C_p.sum(-1, keepdims=True) > 0
    M = np.where(use_p, M_p, M_all)
    C = np.where(use_p, C_p, C_all)
    G = E.mean_force(M, C)
    F = E.histogram_pmf(G, 2 * np.pi / G.shape[-1])
    d = F - F_ref
    d = d - d.mean(-1, keepdims=True)
    return np.sqrt((d ** 2).mean(-1))


def persistent_first(ok):
    """First index i with ok[j] for every j >= i; None if ok[-1] is False."""
    ok = np.asarray(ok, bool)
    if not ok[-1]:
        return None
    bad = np.nonzero(~ok)[0]
    return 0 if bad.size == 0 else int(bad[-1]) + 1


def endpoints(out, meta, F_ref):
    e = errors(out, F_ref)
    steps = out["save_step"].astype(float)
    u = steps / float(meta["n_steps"])
    N = int(meta["N"])
    rows = []
    for r in range(e.shape[1]):
        row = dict(seed=int(out["seeds"][r]), N=N, method=meta["method"], e_final=float(e[-1, r]),
                   I_F=float(np.trapezoid(e[:, r], u)) if hasattr(np, "trapezoid") else float(np.trapz(e[:, r], u)),
                   repl=int(out["sv_i"][-1, r, 0]), cross_per_walker=float(out["sv_i"][-1, r, 1] / N),
                   ess_min=float(out["sv_f"][:, r, 0].min()), wmax=float(out["sv_f"][:, r, 1].max()))
        for name, thr in THR.items():
            i = persistent_first(e[:, r] <= thr)
            row[f"b_{name}"] = math.inf if i is None else float(steps[i] * N)
            row[f"k_{name}"] = math.inf if i is None else float(steps[i])
        rows.append(row)
    return rows, e


def median_inf(x):
    x = np.sort(np.asarray(x, float))
    return float(np.median(x)) if np.isfinite(np.median(x)) else math.inf


def boot_ratio(a, b, n=N_BOOT, seed=BOOT_SEED):
    """median(a)/median(b), unpaired bootstrap 95 % CI; inf-aware (censored = inf)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    est = np.median(a) / np.median(b)
    ia = rng.integers(0, len(a), (n, len(a)))
    ib = rng.integers(0, len(b), (n, len(b)))
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = np.median(a[ia], axis=1) / np.median(b[ib], axis=1)
    lo, hi = np.nanpercentile(rr, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def boot_paired_median(d, n=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = np.median(d[idx], axis=1)
    return float(np.median(d)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_dir(pattern, F_ref):
    rows = []
    for p in sorted(glob.glob(pattern)):
        out, meta = LE.load_npz(p)
        r, _ = endpoints(out, meta, F_ref)
        for x in r:
            x["file"] = os.path.relpath(p, ROOT)
            x["run_s"] = meta["timers"]["run_s"]
        rows.extend(r)
    return rows


def cmd_pilot():
    F_ref, _ = reference()
    rows = load_dir(os.path.join(RES, "pilot", "*.npz"), F_ref)
    b = [x["b_mid"] for x in rows]
    med = median_inf(b)
    B = int(math.ceil(2.0 * med / 1024.0 / 64.0) * 1024 * 64) if math.isfinite(med) else None
    summ = dict(rule="B = 2 x median b_mid of ABF at N = 256 (8 pilot seeds), rounded up to a multiple of 65536",
                b_mid=b, median_b_mid=med, B=B, rows=rows)
    with open(os.path.join(RES, "pilot", "budget.json"), "w") as fh:
        json.dump(summ, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1, default=float))


def cmd_ladder():
    F_ref, _ = reference()
    rows = load_dir(os.path.join(RES, "ladder", "N*", "*.npz"), F_ref)
    table = {}
    for x in rows:
        table.setdefault((x["method"], x["N"]), []).append(x)
    lines = ["| N | ABF median b_mid | cens. | FR median b_mid | cens. | ABF I_F | FR I_F | paired dI_F % [95% CI] | "
             "FR repl | FR ESS_min | ABF cross/walker |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    best = {}
    summ = []
    for N in sorted({k[1] for k in table}):
        a = table.get(("abf", N), [])
        f = table.get(("fr", N), [])
        ma = median_inf([x["b_mid"] for x in a]) if a else math.inf
        mf = median_inf([x["b_mid"] for x in f]) if f else math.inf
        for m, val in (("abf", ma), ("fr", mf)):
            if m not in best or val < best[m][1]:
                best[m] = (N, val)
        da = {x["seed"]: x["I_F"] for x in a}
        dfr = {x["seed"]: x["I_F"] for x in f}
        common = sorted(set(da) & set(dfr))
        d = [100 * (dfr[s] - da[s]) / da[s] for s in common]
        dm = boot_paired_median(d) if d else (math.nan,) * 3
        rec = dict(N=N, abf_median_b_mid=ma, fr_median_b_mid=mf,
                   abf_cens=sum(not math.isfinite(x["b_mid"]) for x in a),
                   fr_cens=sum(not math.isfinite(x["b_mid"]) for x in f),
                   abf_I_F=float(np.median([x["I_F"] for x in a])) if a else math.nan,
                   fr_I_F=float(np.median([x["I_F"] for x in f])) if f else math.nan,
                   dI_F_pct=dm, n_seeds=(len(a), len(f)),
                   fr_repl=float(np.median([x["repl"] for x in f])) if f else math.nan,
                   fr_ess_min=float(np.median([x["ess_min"] for x in f])) if f else math.nan,
                   abf_cross=float(np.median([x["cross_per_walker"] for x in a])) if a else math.nan)
        summ.append(rec)
        lines.append(f"| {N} | {ma:.4g} | {rec['abf_cens']}/{len(a)} | {mf:.4g} | {rec['fr_cens']}/{len(f)} | "
                     f"{rec['abf_I_F']:.4f} | {rec['fr_I_F']:.4f} | {dm[0]:+.1f} [{dm[1]:+.1f}, {dm[2]:+.1f}] | "
                     f"{rec['fr_repl']:.0f} | {rec['fr_ess_min']:.2f} | {rec['abf_cross']:.2f} |")
    sel = dict(best_abf_N=best.get("abf", (None,))[0], best_fr_N=best.get("fr", (None,))[0],
               rule="argmin over N of the median b_mid (censored = inf), per method")
    a = [x["b_mid"] for x in table.get(("abf", sel["best_abf_N"]), [])]
    f = [x["b_mid"] for x in table.get(("fr", sel["best_fr_N"]), [])]
    if a and f:
        sel["exploratory_R_B"] = boot_ratio(f, a)
    out = dict(rows=rows, summary=summ, selection=sel)
    with open(os.path.join(RES, "ladder", "ladder.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(os.path.join(RES, "ladder", "ladder_table.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n\n" + json.dumps(sel, default=float) + "\n")
    print("\n".join(lines))
    print(json.dumps(sel, default=float))


def cmd_confirm():
    F_ref, _ = reference()
    rows = load_dir(os.path.join(RES, "confirm", "*.npz"), F_ref)
    a = [x["b_mid"] for x in rows if x["method"] == "abf"]
    f = [x["b_mid"] for x in rows if x["method"] == "fr"]
    R = boot_ratio(f, a)
    decision = "FR better" if R[2] < 1 else ("ABF better" if R[1] > 1 else "unresolved")
    out = dict(R_B=R, decision=decision, n=(len(a), len(f)),
               abf_cens=sum(not math.isfinite(x) for x in a), fr_cens=sum(not math.isfinite(x) for x in f),
               abf_median=median_inf(a), fr_median=median_inf(f), rows=rows)
    with open(os.path.join(RES, "confirm", "confirm.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=float))


if __name__ == "__main__":
    {"pilot": cmd_pilot, "ladder": cmd_ladder, "confirm": cmd_confirm}[sys.argv[1]]()
