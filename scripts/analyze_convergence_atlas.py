"""Turn the convergence atlas into the two numbers a "faster convergence" claim needs.

    python scripts/analyze_convergence_atlas.py

Reads `results/convergence_atlas/atlas.npz` and writes

    results/convergence_atlas/speedup.json      per panel, per arm
    results/convergence_atlas/scoreboard.csv    one row per (panel, arm)

ENDPOINT 1 -- time-integrated error, the campaign's preregistered primary:

    I_F = int_0^T ||F_hat_t - F_ref||  dt ,   reported as the PAIRED median per-seed
                                              relative change vs ABF (the repo convention;
                                              reproduces every shipped verdict exactly).

ENDPOINT 2 -- time to a prescribed accuracy, which is what "converges faster" means
literally:

    tau_eps = inf { t : e_F(s) <= eps  for all s in [t, min(t+Delta, T)] } ,  Delta = 0.2 T
    S_eps   = tau_eps(ABF) / tau_eps(arm)          > 1 means the arm gets there sooner.

The persistence window Delta is what stops one lucky dip below the threshold from being
scored as convergence.

THRESHOLDS ARE FIXED BY A RULE DECLARED BEFORE LOOKING AT ANY ARM, never by picking the
flattering one. Two families:

  * scale-free      eps = f * e_0,  f in {1/2, 1/4, 1/8}, where e_0 is the error at t=0.
                    At t=0 the bias is identically zero in every arm, so e_0 is a property
                    of the SYSTEM, not of the method -- the rule cannot favour an arm.
                    Where a threshold is never met it is reported CENSORED, not dropped.

  * ABF-final       eps = median_seeds e_F^ABF(T), the accuracy ABF has when its run ends.
                    tau for that eps answers the question a practitioner actually asks:
                    how much simulation does the arm need to reach the answer ABF finishes
                    with? This one is defined for every panel.

CENSORING IS REPORTED, NEVER IMPUTED. An arm that never reaches a threshold has tau = inf,
and the JSON records how many seeds were censored. Silently dropping them would turn "this
method never got there" into "this method is not in the average".
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "convergence_atlas")
FRACTIONS = (0.5, 0.25, 0.125)
PERSIST = 0.2                      # Delta = 0.2 * T
N_BOOT, BOOT_SEED = 10000, 20260817


def tau(times, curve, eps, persist=PERSIST):
    """First time the curve drops below eps AND stays there for the persistence window."""
    t = np.asarray(times, dtype=float)
    c = np.asarray(curve, dtype=float)
    T = float(t[-1])
    delta = persist * T
    below = c <= eps
    for i in range(len(t)):
        if not below[i]:
            continue
        end = min(t[i] + delta, T)
        window = (t >= t[i]) & (t <= end)
        if np.all(below[window]):
            return float(t[i])
    return float("inf")


def boot_median(x, n=N_BOOT, seed=BOOT_SEED, lo=2.5, hi=97.5):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    m = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.percentile(m, lo)), float(np.percentile(m, hi))


def integrated(times, curves):
    return np.trapezoid(curves, times, axis=1)


def main():
    d = np.load(os.path.join(OUT, "atlas.npz"))
    meta = json.load(open(os.path.join(OUT, "atlas.json")))
    panels = meta["panels"]
    out, rows = {}, []

    for p in panels:
        name = p["panel"]
        t = d[f"{name}::times"]
        arms = [a for a in p["arms"] if "__alt" not in a]
        if "abf" not in arms:
            continue
        base = d[f"{name}::abf"]
        e0 = float(np.median([np.median(d[f"{name}::{a}"][:, 0]) for a in arms]))
        eps_abf_final = float(np.median(base[:, -1]))
        thresholds = {f"e0/{int(1/f)}": f * e0 for f in FRACTIONS}
        thresholds["abf_final"] = eps_abf_final

        med = {a: np.median(d[f"{name}::{a}"], axis=0) for a in arms}
        tau_med = {a: {k: tau(t, med[a], v) for k, v in thresholds.items()} for a in arms}
        tau_seed = {a: {k: np.array([tau(t, c, v) for c in d[f"{name}::{a}"]])
                        for k, v in thresholds.items()} for a in arms}

        rec = dict(label=p["label"], t_fr=p["t_fr"], t_max=p["t_max"], e0=e0,
                   eps_abf_final=eps_abf_final, thresholds=thresholds,
                   note=p["note"], ref_label=p["ref_label"], arms={})
        iF_base = integrated(t, base)
        for a in arms:
            cur = d[f"{name}::{a}"]
            iF = integrated(t, cur)
            n = min(len(iF), len(iF_base))
            rel = 100.0 * (iF[:n] - iF_base[:n]) / iF_base[:n]
            m, lo, hi = boot_median(rel)
            entry = dict(n_seeds=int(cur.shape[0]),
                         integrated_median=float(np.median(iF)),
                         final_median=float(np.median(cur[:, -1])),
                         rel_integrated_pct=m, rel_integrated_ci95=[lo, hi],
                         wins=int((rel < 0).sum()), n_paired=int(n),
                         tau={}, speedup={})
            # final-time endpoint, same paired convention
            fin, fin_b = cur[:, -1], base[:, -1]
            relf = 100.0 * (fin[:n] - fin_b[:n]) / fin_b[:n]
            fm, flo, fhi = boot_median(relf)
            entry["rel_final_pct"], entry["rel_final_ci95"] = fm, [flo, fhi]
            for k in thresholds:
                ta, tb = tau_seed[a][k], tau_seed["abf"][k]
                cens_a = int(np.sum(~np.isfinite(ta)))
                both = np.isfinite(ta[:n]) & np.isfinite(tb[:n])
                ratio = np.where(both, tb[:n] / np.maximum(ta[:n], 1e-12), np.nan)
                sm, slo, shi = boot_median(ratio[both]) if both.any() else (np.nan,)*3
                entry["tau"][k] = dict(median_curve=tau_med[a][k],
                                       per_seed_median=float(np.median(ta[np.isfinite(ta)]))
                                       if np.isfinite(ta).any() else float("inf"),
                                       censored_seeds=cens_a, n_seeds=int(len(ta)))
                ta_m, tb_m = tau_med[a][k], tau_med["abf"][k]
                if np.isfinite(ta_m) and np.isfinite(tb_m):
                    s_med, status = (tb_m / ta_m if ta_m > 0 else float("inf")), "ok"
                elif np.isfinite(ta_m):          # arm reaches it, ABF never does
                    s_med, status = float("inf"), "abf_never"
                elif np.isfinite(tb_m):          # ABF reaches it, arm never does
                    s_med, status = float("nan"), "arm_never"
                else:
                    s_med, status = float("nan"), "neither_reaches"
                entry["speedup"][k] = dict(
                    median_curve=s_med, status=status,
                    paired_median=sm, ci95=[slo, shi],
                    n_pairs_usable=int(both.sum()),
                    n_pairs_censored=int(n - both.sum()))
            rec["arms"][a] = entry
            def _s(k):
                sp = entry["speedup"][k]
                return round(sp["median_curve"], 3) if sp["status"] == "ok" else sp["status"]
            rows.append(dict(
                panel=name, label=p["label"], arm=a, n_seeds=entry["n_seeds"],
                rel_integrated_pct=round(m, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                wins=f"{entry['wins']}/{n}", rel_final_pct=round(fm, 2),
                tau_abf_final=entry["tau"]["abf_final"]["median_curve"],
                speedup_abf_final=_s("abf_final"), speedup_e0_over_8=_s("e0/8")))
        out[name] = rec

    json.dump(dict(method=dict(fractions=list(FRACTIONS), persistence_fraction=PERSIST,
                               n_boot=N_BOOT, boot_seed=BOOT_SEED),
                   panels=out, excluded=meta["excluded"]),
              open(os.path.join(OUT, "speedup.json"), "w"), indent=2, default=str)
    with open(os.path.join(OUT, "scoreboard.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- console summary: mFR vs ABF only, which is the claim under test ----
    print(f"{'panel':17s} {'arm':16s} {'int L2(F) vs ABF':>18s} {'wins':>7s} "
          f"{'final':>9s} {'S(ABF-final)':>13s}")
    print("-" * 88)
    for name, rec in out.items():
        for a, e in rec["arms"].items():
            if a == "abf":
                continue
            sp = e["speedup"]["abf_final"]
            ss = (f"{sp['median_curve']:.2f}x" if sp["status"] == "ok"
                  else {"arm_never": "never reaches", "abf_never": "ABF never",
                        "neither_reaches": "neither"}[sp["status"]])
            print(f"{name:17s} {a:16s} {e['rel_integrated_pct']:+9.2f} %"
                  f" [{e['rel_integrated_ci95'][0]:+6.1f},{e['rel_integrated_ci95'][1]:+6.1f}]"
                  f" {e['wins']:3d}/{e['n_paired']:<3d} {e['rel_final_pct']:+8.1f}% {ss:>13s}")
        print()
    print(f"wrote {os.path.join(OUT, 'speedup.json')} and scoreboard.csv")


if __name__ == "__main__":
    main()
