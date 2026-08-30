#!/usr/bin/env python
"""Analyze one olefin/CHA cell: abf vs fr_uniform against the umbrella/WHAM
reference, with the frozen campaign endpoints.  Also computes e_F'(t) (mean
force) for the three-panel mechanism figure.

    python scripts/analyze_uniform_cha.py --guest ethene --temperature 450
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CHA = os.path.join(ROOT, "results/uniform_campaign/cha")
PREREG = os.path.join(ROOT, "configs/uniform_campaign/cha_prereg.json")

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


def error_series(prof, ref, mask):
    d = (prof - ref[None, None, :])[:, :, mask]
    d = d - d.mean(axis=-1, keepdims=True)
    return np.sqrt((d * d).mean(axis=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guest", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    rule = pre["success_rule"]
    tag = f"{a.guest}_{a.temperature:g}"
    ref = np.load(os.path.join(CHA, "reference", f"reference_{tag}.npz"),
                  allow_pickle=True)
    runs = {m: np.load(os.path.join(CHA, f"production_{tag}", f"{m}.npz"),
                       allow_pickle=True) for m in ("abf", "fr_uniform")}
    grid = runs["abf"]["grid"]
    assert np.allclose(grid, ref["grid"]), "reference and engine grids differ"
    xi_A, xi_B = float(ref["xi_A"]), float(ref["xi_B"])
    # scoring mask: the physically meaningful stretch, cage A to cage B inclusive
    mask = (grid >= xi_A - 1.0) & (grid <= xi_B + 1.0)
    F_ref = np.asarray(ref["F"], dtype=float)
    dFdxi_ref = np.gradient(F_ref, grid)
    t = np.asarray(runs["abf"]["times"], dtype=float)

    err = {m: error_series(np.asarray(runs[m]["pmf"], dtype=float), F_ref, mask)
           for m in runs}
    errp = {m: error_series(np.asarray(runs[m]["mean_force"], dtype=float),
                            dFdxi_ref, mask) for m in runs}
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

    N = int(pre["sampler"]["n_replicas"])
    steps = np.asarray(runs["fr_uniform"]["steps"])
    active = steps >= pre["sampler"]["fr_start_steps"]
    ess = np.asarray(runs["fr_uniform"]["ancestor_ess"], dtype=float)[active] / N
    wmax = np.asarray(runs["fr_uniform"]["max_ancestor_frac"], dtype=float)[active]
    ess_min = np.nanmin(ess, axis=0)
    wmax_mx = np.nanmax(wmax, axis=0)
    ess_med, wmax_med = float(np.median(ess_min)), float(np.median(wmax_mx))
    health_ok = ess_med >= rule["ess_anc_over_N_min"] and wmax_med <= rule["wmax_max"]

    accel = med <= rule["median_rel_change_pct_max"] and hi < rule["ci95_upper_pct_max"]
    safe = accel and med_fin <= rule["final_noninferiority_margin_pct"] and health_ok
    neutral = (abs(med) < abs(rule["median_rel_change_pct_max"])
               and med_fin <= rule["final_noninferiority_margin_pct"])
    verdict = ("SAFE_ACCELERATOR" if safe else "ACCELERATION_POSITIVE" if accel
               else "NEUTRAL" if neutral else "NEGATIVE_OR_UNSAFE")

    beta = 1.0 / float(ref["kT"])
    summary = dict(
        cell=tag, n_pairs=R,
        reference=dict(dF_kT=float(ref["dF_barrier"]) * beta,
                       dU_kT=float(ref["dU_barrier"]) * beta,
                       mTdS_kT=float(ref["mTdS_barrier"]) * beta,
                       entropic_fraction=float(ref["mTdS_barrier"] / ref["dF_barrier"])),
        d_int_pct=dict(median=med, ci95=[lo, hi], wins=int((d_int < 0).sum())),
        d_final_pct=dict(median=med_fin, ci95=[flo, fhi], wins=int((d_fin < 0).sum())),
        health=dict(median_min_ess_frac=ess_med, median_max_wmax=wmax_med,
                    worst_min_ess_frac=float(np.nanmin(ess_min)),
                    worst_max_wmax=float(np.nanmax(wmax_mx)), ok=bool(health_ok)),
        crossings=dict(abf=int(np.asarray(runs["abf"]["n_crossings"]).sum()),
                       uni=int(np.asarray(runs["fr_uniform"]["n_crossings"]).sum())),
        time_to_accuracy=speed, verdict=verdict,
        fr_rate=json.loads(str(runs["fr_uniform"]["meta"]))["fr_rate"])
    with open(os.path.join(CHA, f"summary_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    lines = ["seed,int_abf,int_uni,d_int_pct,final_abf,final_uni,d_final_pct"]
    s0 = pre["production"]["seeds_first"][tag]
    for r in range(R):
        lines.append(f"{s0 + r},{I['abf'][r]:.4f},{I['fr_uniform'][r]:.4f},{d_int[r]:.3f},"
                     f"{fin['abf'][r]:.5f},{fin['fr_uniform'][r]:.5f},{d_fin[r]:.3f}")
    with open(os.path.join(CHA, f"comparison_{tag}.csv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    rf = summary["reference"]
    print(f"CHA {tag}: barrier dF={rf['dF_kT']:.2f} kT (dU {rf['dU_kT']:.2f}, "
          f"-TdS {rf['mTdS_kT']:.2f}, {100*rf['entropic_fraction']:.0f}% entropic)")
    print(f"  Delta I_F    median {med:+.2f}%  CI95 [{lo:+.2f}, {hi:+.2f}]  "
          f"wins {summary['d_int_pct']['wins']}/{R}")
    print(f"  Delta e_F(T) median {med_fin:+.2f}%  CI95 [{flo:+.2f}, {fhi:+.2f}]  "
          f"wins {summary['d_final_pct']['wins']}/{R}")
    print(f"  health: ESS/N {ess_med:.3f} (worst {summary['health']['worst_min_ess_frac']:.3f}), "
          f"wmax {wmax_med:.4f}, ok={health_ok}")
    print(f"  crossings: abf {summary['crossings']['abf']} vs uni "
          f"{summary['crossings']['uni']}")
    for name, sp in speed.items():
        ss = f"{sp['speedup']:.2f}x" if sp["speedup"] else sp["status"]
        print(f"  tau[{name}]: abf {sp['tau_abf']:.1f} vs uni {sp['tau_uni']:.1f} -> {ss}")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
