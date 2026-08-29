#!/usr/bin/env python
"""Analyze Stage 2 of the uniform-FR campaign (WCA Case IX cell, abf vs fr_uniform).

Pairs the two fresh arms per seed (they ran in the same process, same initial
conditions) and applies the endpoints frozen in docs/UNIFORM_FR_CAMPAIGN.md.
Context anchor: the closed Case IX v2 EMA arm scored Delta I_F = -17.97 % on the
same cell, reference and seeds.

    python scripts/analyze_uniform_wca.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "results/uniform_campaign/wca/uniform/raw")
OUT = os.path.join(ROOT, "results/uniform_campaign/wca")

PERSIST = 0.2
FRACTIONS = (0.5, 0.25, 0.125)
N_BOOT, BOOT_SEED = 10000, 20260829
N_REPLICAS = 1024
ESS_FLOOR, WMAX_CAP = 0.10, 0.05      # the Case IX health floors
RULE = dict(median_max=-10.0, ci_upper_max=0.0, final_margin=5.0)


def boot_median(x, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    return (float(np.percentile(np.median(np.asarray(x)[idx], axis=1), 2.5)),
            float(np.percentile(np.median(np.asarray(x)[idx], axis=1), 97.5)))


def tau(times, curve, eps, persist):
    T = times[-1]
    below = curve <= eps
    for i in range(len(times)):
        if not below[i]:
            continue
        j = min(np.searchsorted(times, times[i] + persist * T, side="left"),
                len(times) - 1)
        if below[i:j + 1].all():
            return float(times[i])
    return float("inf")


def load_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(RAW, "*.npz"))):
        with np.load(path, allow_pickle=True) as z:
            spec = json.loads(str(z["spec_json"]))
            runs[(spec["method"], spec["seed"])] = dict(
                int_lf=float(z["integrated_l2_f"]), l2_f=float(z["l2_f"]),
                l2_f_t=np.asarray(z["l2_f_t"]), times=np.asarray(z["times"]),
                ess_t=np.asarray(z["ancestor_ess_t"]),
                wmax_t=np.asarray(z["max_ancestor_frac_t"]),
                min_ess=float(z["min_ancestor_ess"]),
                wmax_max=float(z["max_ancestor_frac_over_time"]),
                round_trips=float(z["n_round_trips"]),
                ref_label=str(z["reference_label"]))
    return runs


def main():
    runs = load_runs()
    seeds = sorted({s for (m, s) in runs if m == "abf" and ("fr_uniform", s) in runs})
    print(f"{len(runs)} runs loaded, {len(seeds)} complete pairs: {seeds}")
    assert seeds, "no complete pairs yet"
    labels = {runs[(m, s)]["ref_label"] for m in ("abf", "fr_uniform") for s in seeds}
    assert len(labels) == 1 and "v2" in next(iter(labels)), f"mixed references: {labels}"

    d_int, d_fin, ess_mins, wmax_maxs, rt = [], [], [], [], {}
    lines = ["seed,int_lf_abf,int_lf_uni,d_int_pct,final_abf,final_uni,d_final_pct,"
             "min_ess_uni,wmax_uni,rt_abf,rt_uni"]
    for s in seeds:
        a, u = runs[("abf", s)], runs[("fr_uniform", s)]
        d_int.append(100.0 * (u["int_lf"] - a["int_lf"]) / a["int_lf"])
        d_fin.append(100.0 * (u["l2_f"] - a["l2_f"]) / a["l2_f"])
        ess_mins.append(u["min_ess"] / N_REPLICAS)
        wmax_maxs.append(u["wmax_max"])
        rt.setdefault("abf", []).append(a["round_trips"])
        rt.setdefault("uni", []).append(u["round_trips"])
        lines.append(f"{s},{a['int_lf']:.3f},{u['int_lf']:.3f},{d_int[-1]:.3f},"
                     f"{a['l2_f']:.5f},{u['l2_f']:.5f},{d_fin[-1]:.3f},"
                     f"{ess_mins[-1]:.4f},{wmax_maxs[-1]:.4f},"
                     f"{a['round_trips']:.0f},{u['round_trips']:.0f}")

    lo, hi = boot_median(d_int, BOOT_SEED)
    flo, fhi = boot_median(d_fin, BOOT_SEED + 1)
    med, med_fin = float(np.median(d_int)), float(np.median(d_fin))

    t = runs[("abf", seeds[0])]["times"]
    curves = {m: np.median([runs[(m, s)]["l2_f_t"] for s in seeds], axis=0)
              for m in ("abf", "fr_uniform")}
    e0 = float(curves["abf"][0])
    eps_list = {f"e0/{int(1 / f)}": e0 * f for f in FRACTIONS}
    eps_list["abf_final"] = float(curves["abf"][-1])
    speed = {}
    for name, eps in eps_list.items():
        ta, tu = tau(t, curves["abf"], eps, PERSIST), tau(t, curves["fr_uniform"], eps, PERSIST)
        speed[name] = dict(eps=eps, tau_abf=ta, tau_uni=tu,
                           speedup=(ta / tu if np.isfinite(ta) and np.isfinite(tu) and tu > 0
                                    else None),
                           status=("ok" if np.isfinite(ta) and np.isfinite(tu) else
                                   "abf_never" if np.isfinite(tu) else
                                   "arm_never" if np.isfinite(ta) else "neither"))

    health_ok = min(ess_mins) >= ESS_FLOOR and max(wmax_maxs) <= WMAX_CAP
    accel = med <= RULE["median_max"] and hi < RULE["ci_upper_max"]
    safe = accel and med_fin <= RULE["final_margin"] and health_ok
    neutral = abs(med) < abs(RULE["median_max"]) and med_fin <= RULE["final_margin"]
    verdict = ("SAFE_ACCELERATOR" if safe else "ACCELERATION_POSITIVE" if accel
               else "NEUTRAL" if neutral else "NEGATIVE_OR_UNSAFE")

    summary = dict(
        n_pairs=len(seeds), seeds=seeds,
        reference_label=next(iter(labels)),
        d_int_pct=dict(median=med, ci95=[lo, hi],
                       wins=int(np.sum(np.array(d_int) < 0))),
        d_final_pct=dict(median=med_fin, ci95=[flo, fhi],
                         wins=int(np.sum(np.array(d_fin) < 0))),
        health=dict(min_ess_frac=min(ess_mins), max_wmax=max(wmax_maxs), ok=bool(health_ok),
                    floors=dict(ess=ESS_FLOOR, wmax=WMAX_CAP)),
        round_trips=dict(abf_median=float(np.median(rt["abf"])),
                         uni_median=float(np.median(rt["uni"]))),
        time_to_accuracy=speed, verdict=verdict,
        context=dict(caseix_v2_ema_d_int_pct=-17.97))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(os.path.join(OUT, "comparison.csv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"WCA uniform-FR: n={len(seeds)} pairs  (reference: {next(iter(labels))[:40]}...)")
    print(f"  Delta I_F    median {med:+.2f}%  CI95 [{lo:+.2f}, {hi:+.2f}]  "
          f"wins {summary['d_int_pct']['wins']}/{len(seeds)}")
    print(f"  Delta e_F(T) median {med_fin:+.2f}%  CI95 [{flo:+.2f}, {fhi:+.2f}]  "
          f"wins {summary['d_final_pct']['wins']}/{len(seeds)}")
    print(f"  health: min ESS/N {min(ess_mins):.3f} (floor {ESS_FLOOR}), "
          f"max wmax {max(wmax_maxs):.4f} (cap {WMAX_CAP}), ok={health_ok}")
    print(f"  round trips: abf {np.median(rt['abf']):.0f} vs uni {np.median(rt['uni']):.0f}")
    for name, s in speed.items():
        sp = f"{s['speedup']:.2f}x" if s["speedup"] else s["status"]
        print(f"  tau[{name}]: abf {s['tau_abf']:.0f} vs uni {s['tau_uni']:.0f} -> {sp}")
    print(f"  VERDICT: {verdict}   (context: Case IX v2 EMA arm was -17.97%)")


if __name__ == "__main__":
    main()
