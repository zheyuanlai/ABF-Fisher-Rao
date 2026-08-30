#!/usr/bin/env python
"""Analyze Stage 4 of the uniform-FR campaign (ethane/LTA, abf vs fr_uniform).

Scores both arms' FES series against the independent umbrella/WHAM reference
(never seen by either arm), then applies the frozen campaign endpoints.
Genealogy is gated on the campaign's median-across-seeds convention (the
gateway rule); the worst single seed is reported alongside, never substituted.

    python scripts/analyze_uniform_lta.py
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROD = os.path.join(ROOT, "results/uniform_campaign/lta/production")
REF = os.path.join(ROOT, "results/uniform_campaign/lta/reference/reference_T300.npz")
PREREG = os.path.join(ROOT, "configs/uniform_campaign/lta_prereg.json")
OUT = os.path.join(ROOT, "results/uniform_campaign/lta")

PI = math.pi
PERSIST = 0.2
FRACTIONS = (0.5, 0.25, 0.125)
N_BOOT, BOOT_SEED = 10000, 20260829


def boot_median(x, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    med = np.median(np.asarray(x)[idx], axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


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


def circular_interp_ref(F_ref, grid_ref, grid_eng):
    """Interpolate the reference onto the engine grid (both on [-pi, pi))."""
    order = np.argsort(grid_ref)
    gr, fr = grid_ref[order], F_ref[order]
    gx = np.concatenate([gr - 2 * PI, gr, gr + 2 * PI])
    fx = np.concatenate([fr, fr, fr])
    return np.interp(grid_eng, gx, fx)


def error_series(pmf, F_ref_on_grid):
    """Aligned full-circle RMS per (save, seed)."""
    d = pmf - F_ref_on_grid[None, None, :]
    d = d - d.mean(axis=-1, keepdims=True)
    return np.sqrt((d * d).mean(axis=-1))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-prereg", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    a = ap.parse_args()
    if a.sweep_prereg is not None:
        assert a.temperature is not None
        pre = json.load(open(a.sweep_prereg))
        tkey = f"{a.temperature:g}"
        prod = os.path.join(ROOT, f"results/uniform_campaign/lta/production_T{tkey}")
        ref_path = os.path.join(ROOT,
                                f"results/uniform_campaign/lta/reference/reference_T{tkey}.npz")
        out_json = os.path.join(OUT, f"summary_T{tkey}.json")
        out_csv = os.path.join(OUT, f"comparison_T{tkey}.csv")
        seeds_first = pre["per_T"][tkey]["seeds_first"]
        rate_val = pre["per_T"][tkey]["fr_rate"]
    else:
        pre = json.load(open(PREREG))
        prod, ref_path = PROD, REF
        out_json = os.path.join(OUT, "summary.json")
        out_csv = os.path.join(OUT, "comparison.csv")
        seeds_first = 600
        rate_val = pre["fr_rate"]["value"]
    rule = pre["success_rule"]
    ref = np.load(ref_path, allow_pickle=True)
    runs = {m: np.load(os.path.join(prod, f"{m}.npz"), allow_pickle=True)
            for m in ("abf", "fr_uniform")}
    grid_eng = runs["abf"]["grid"]
    F_ref = circular_interp_ref(ref["F"], ref["grid_phi"], grid_eng)
    t = np.asarray(runs["abf"]["times"], dtype=float)
    assert np.allclose(t, np.asarray(runs["fr_uniform"]["times"], dtype=float))

    err = {m: error_series(np.asarray(runs[m]["pmf"], dtype=float), F_ref)
           for m in runs}                                   # (T, R)
    R = err["abf"].shape[1]
    I = {m: np.trapezoid(err[m], t, axis=0) for m in err}
    fin = {m: err[m][-1] for m in err}
    d_int = 100.0 * (I["fr_uniform"] - I["abf"]) / I["abf"]
    d_fin = 100.0 * (fin["fr_uniform"] - fin["abf"]) / fin["abf"]
    lo, hi = boot_median(d_int, BOOT_SEED)
    flo, fhi = boot_median(d_fin, BOOT_SEED + 1)
    med, med_fin = float(np.median(d_int)), float(np.median(d_fin))

    curves = {m: np.median(err[m], axis=1) for m in err}
    e0 = float(curves["abf"][0])
    eps_list = {f"e0/{int(1/f)}": e0 * f for f in FRACTIONS}
    eps_list["abf_final"] = float(curves["abf"][-1])
    speed = {}
    for name, eps in eps_list.items():
        ta = tau(t, curves["abf"], eps, PERSIST)
        tu = tau(t, curves["fr_uniform"], eps, PERSIST)
        speed[name] = dict(eps=eps, tau_abf=ta, tau_uni=tu,
                           speedup=(ta / tu if np.isfinite(ta) and np.isfinite(tu)
                                    and tu > 0 else None),
                           status=("ok" if np.isfinite(ta) and np.isfinite(tu) else
                                   "abf_never" if np.isfinite(tu) else
                                   "arm_never" if np.isfinite(ta) else "neither"))

    # genealogy of the uniform arm: per-seed min ESS/N over FR-active saves
    N = int(pre["sampler"]["n_replicas"])
    steps = np.asarray(runs["fr_uniform"]["steps"])
    active = steps >= pre["sampler"]["fr_start_steps"]
    # starvation measures for the sweep-level analysis: how established was the
    # ABF arm when FR came on?  (crossings are cumulative per save in repl... use
    # the abf arm's cage-crossing count series if present; fall back to totals)
    ess = np.asarray(runs["fr_uniform"]["ancestor_ess"], dtype=float)[active] / N
    wmax = np.asarray(runs["fr_uniform"]["max_ancestor_frac"], dtype=float)[active]
    ess_min_per_seed = np.nanmin(ess, axis=0)
    wmax_max_per_seed = np.nanmax(wmax, axis=0)
    ess_med = float(np.median(ess_min_per_seed))
    wmax_med = float(np.median(wmax_max_per_seed))
    health_ok = ess_med >= rule["ess_anc_over_N_min"] and wmax_med <= rule["wmax_max"]

    accel = med <= rule["median_rel_change_pct_max"] and hi < rule["ci95_upper_pct_max"]
    safe = accel and med_fin <= rule["final_noninferiority_margin_pct"] and health_ok
    neutral = (abs(med) < abs(rule["median_rel_change_pct_max"])
               and med_fin <= rule["final_noninferiority_margin_pct"])
    verdict = ("SAFE_ACCELERATOR" if safe else "ACCELERATION_POSITIVE" if accel
               else "NEUTRAL" if neutral else "NEGATIVE_OR_UNSAFE")

    beta = 1.0 / float(ref["kT"])
    summary = dict(
        n_pairs=R,
        temperature_K=float(ref["temperature"]),
        reference=dict(path=os.path.relpath(ref_path, ROOT),
                       dF_barrier_kT=float(ref["dF_barrier"]) * beta,
                       dU_barrier_kT=float(ref["dU_barrier"]) * beta,
                       mTdS_barrier_kT=float(ref["mTdS_barrier"]) * beta,
                       entropic_fraction=float(ref["mTdS_barrier"] / ref["dF_barrier"])),
        d_int_pct=dict(median=med, ci95=[lo, hi], wins=int((d_int < 0).sum())),
        d_final_pct=dict(median=med_fin, ci95=[flo, fhi], wins=int((d_fin < 0).sum())),
        health=dict(median_min_ess_frac=ess_med, median_max_wmax=wmax_med,
                    worst_min_ess_frac=float(np.nanmin(ess_min_per_seed)),
                    worst_max_wmax=float(np.nanmax(wmax_max_per_seed)),
                    ok=bool(health_ok), convention="median across seeds (gateway rule)"),
        crossings=dict(abf=int(np.asarray(runs["abf"]["n_cage_crossings"]).sum()),
                       uni=int(np.asarray(runs["fr_uniform"]["n_cage_crossings"]).sum())),
        time_to_accuracy=speed, verdict=verdict,
        fr_rate=rate_val,
        abf_tau_e0_8=speed["e0/8"]["tau_abf"])
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    lines = ["seed,int_abf,int_uni,d_int_pct,final_abf,final_uni,d_final_pct,"
             "min_ess_uni,wmax_uni"]
    for r in range(R):
        lines.append(f"{seeds_first + r},{I['abf'][r]:.4f},{I['fr_uniform'][r]:.4f},"
                     f"{d_int[r]:.3f},{fin['abf'][r]:.5f},{fin['fr_uniform'][r]:.5f},"
                     f"{d_fin[r]:.3f},{ess_min_per_seed[r]:.4f},{wmax_max_per_seed[r]:.4f}")
    with open(out_csv, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    rf = summary["reference"]
    print(f"LTA uniform-FR: n={R} paired seed labels; barrier dF={rf['dF_barrier_kT']:.2f} kT "
          f"(dU {rf['dU_barrier_kT']:.2f}, -TdS {rf['mTdS_barrier_kT']:.2f}; "
          f"entropic fraction {100*rf['entropic_fraction']:.0f}%)")
    print(f"  Delta I_F    median {med:+.2f}%  CI95 [{lo:+.2f}, {hi:+.2f}]  "
          f"wins {summary['d_int_pct']['wins']}/{R}")
    print(f"  Delta e_F(T) median {med_fin:+.2f}%  CI95 [{flo:+.2f}, {fhi:+.2f}]  "
          f"wins {summary['d_final_pct']['wins']}/{R}")
    print(f"  health (median conv.): ESS/N {ess_med:.3f}, wmax {wmax_med:.4f}, "
          f"ok={health_ok} (worst {summary['health']['worst_min_ess_frac']:.3f}/"
          f"{summary['health']['worst_max_wmax']:.4f})")
    print(f"  crossings: abf {summary['crossings']['abf']} vs uni {summary['crossings']['uni']}")
    for name, sp in speed.items():
        s = f"{sp['speedup']:.2f}x" if sp["speedup"] else sp["status"]
        print(f"  tau[{name}]: abf {sp['tau_abf']:.1f} vs uni {sp['tau_uni']:.1f} -> {s}")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
