#!/usr/bin/env python
"""Analyze the ethane/ZIF-8 cell: abf vs fr_uniform against the umbrella/WHAM
reference, with the frozen campaign endpoints, plus the hidden-gate
conditional analysis that is this stage's own contribution.

    python scripts/analyze_uniform_zif8.py --temperature 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from zif8.core_zif8 import js_divergence  # noqa: E402

ZIF = os.path.join(ROOT, "results/uniform_campaign/zif8")
PREREG = os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")

PERSIST = 0.2
FRACTIONS = (0.5, 0.25, 0.125)
N_BOOT, BOOT_SEED = 10000, 20260829


def boot_median(x, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    med = np.median(x[idx], axis=1)
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


def aligned_error(prof, ref):
    """Gauge-aligned RMS over the FULL circular domain (the whole channel is
    the scoring region here: the CV is periodic and there is no wall)."""
    d = prof - ref[None, None, :]
    d = d - d.mean(axis=-1, keepdims=True)
    return np.sqrt((d * d).mean(axis=-1))


def free_error(prof, ref):
    """No additive alignment: the mean force has no gauge freedom."""
    d = prof - ref[None, None, :]
    return np.sqrt((d * d).mean(axis=-1))


def circular_gradient(F, grid):
    d = np.gradient(F, grid)
    d[0] = (F[1] - F[-1]) / (grid[1] - grid[0] + (grid[-1] - grid[-2]))
    d[-1] = (F[0] - F[-2]) / (grid[1] - grid[0] + (grid[-1] - grid[-2]))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=300.0)
    a = ap.parse_args()
    pre = json.load(open(PREREG))
    rule = pre["success_rule"]
    tag = f"T{a.temperature:g}"
    ref = np.load(os.path.join(ZIF, "reference", f"reference_{tag}.npz"),
                  allow_pickle=True)
    runs = {m: np.load(os.path.join(ZIF, f"production_{tag}", f"{m}.npz"),
                       allow_pickle=True) for m in ("abf", "fr_uniform")}
    grid = runs["abf"]["grid"]
    assert np.allclose(grid, ref["grid"]), "reference and engine grids differ"
    xi = np.asarray(ref["xi_grid"], float)
    F_ref = np.asarray(ref["F"], float)
    Fp_ref = circular_gradient(F_ref, np.asarray(grid, float))
    t = np.asarray(runs["abf"]["times"], float)
    kT = float(ref["kT"])

    shapes = {m: np.asarray(runs[m]["pmf"]).shape for m in runs}
    assert shapes["abf"] == shapes["fr_uniform"], f"arm shape mismatch {shapes}"
    err = {m: aligned_error(np.asarray(runs[m]["pmf"], float), F_ref) for m in runs}
    errp = {m: free_error(np.asarray(runs[m]["mean_force"], float), Fp_ref)
            for m in runs}
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
        ta, tu = tau(t, curves["abf"], eps, PERSIST), tau(t, curves["fr_uniform"],
                                                          eps, PERSIST)
        speed[name] = dict(eps=eps, tau_abf=ta, tau_uni=tu,
                           speedup=(ta / tu if np.isfinite(ta) and np.isfinite(tu)
                                    and tu > 0 else None),
                           status=("ok" if np.isfinite(ta) and np.isfinite(tu) else
                                   "abf_never" if np.isfinite(tu) else
                                   "arm_never" if np.isfinite(ta) else "neither"))

    N = int(json.loads(str(runs["fr_uniform"]["meta"]))["n_replicas"])
    steps = np.asarray(runs["fr_uniform"]["steps"])
    active = steps >= pre["sampler"]["fr_start_steps"]
    ess = np.asarray(runs["fr_uniform"]["ancestor_ess"], float)[active] / N
    wmax = np.asarray(runs["fr_uniform"]["max_ancestor_frac"], float)[active]
    ess_min, wmax_mx = np.nanmin(ess, axis=0), np.nanmax(wmax, axis=0)
    ess_med, wmax_med = float(np.median(ess_min)), float(np.median(wmax_mx))
    health_ok = ess_med >= rule["ess_anc_over_N_min"] and wmax_med <= rule["wmax_max"]

    # ---- the hidden gate: did FR multiply correlated gate states? ----------
    gref = np.asarray(ref["gate_hist_window"], float)
    gate = {}
    for m in runs:
        cum = np.asarray(runs[m]["gate_hist_cumulative"], float)
        blocks = np.asarray(runs[m]["gate_hist_block"], float).sum(axis=1)
        tot = blocks.sum(axis=-1)
        js_t = np.where(tot > pre["screen"]["gate_min_samples"],
                        js_divergence(blocks, np.broadcast_to(gref, blocks.shape)),
                        np.nan)
        edges = np.asarray(runs[m]["gate_edges"], float)
        mids = 0.5 * (edges[1:] + edges[:-1])
        w = cum.sum(axis=0)
        # An EMPTY gate histogram must not read as a number: js_divergence of
        # an all-zero row against the reference returns 0.5*log 2 = 0.347, a
        # perfectly plausible-looking value, and the weighted mean returns
        # 0.0 A.  Both are "no data", and this project has been bitten before
        # by no-data reading as a result.
        n_gate = float(w.sum())
        enough = n_gate >= pre["screen"]["gate_min_samples"]
        finite_js = js_t[-5:][np.isfinite(js_t[-5:])]
        gate[m] = dict(
            n_gate_samples=n_gate,
            js_final=(float(finite_js.mean()) if finite_js.size else None),
            js_vs_reference_cumulative=(float(js_divergence(w, gref))
                                        if enough else None),
            mean_A_gate=(float((w * mids).sum() / n_gate) if enough else None),
            crossing_A_gate=(float(np.mean(runs[m]["cross_gate_samples"]))
                             if np.asarray(runs[m]["cross_gate_samples"]).size
                             else None),
            n_crossing_samples=int(np.asarray(runs[m]["cross_gate_samples"]).size))
    ref_mids = 0.5 * (np.asarray(ref["gate_edges"], float)[1:]
                      + np.asarray(ref["gate_edges"], float)[:-1])
    gate["reference"] = dict(mean_A_gate=float((gref * ref_mids).sum()
                                               / max(gref.sum(), 1e-9)))
    # the conditional-limitation read-out: is the FR arm's gate conditional
    # WORSE than plain ABF's?  That is the R15 failure mode, made visible.
    ja, ju = (gate["abf"]["js_vs_reference_cumulative"],
              gate["fr_uniform"]["js_vs_reference_cumulative"])
    gate["fr_gate_worse_than_abf"] = (None if (ja is None or ju is None)
                                      else bool(ju > ja))

    accel = med <= rule["median_rel_change_pct_max"] and hi < rule["ci95_upper_pct_max"]
    safe = accel and med_fin <= rule["final_noninferiority_margin_pct"] and health_ok
    neutral = (abs(med) < abs(rule["median_rel_change_pct_max"])
               and med_fin <= rule["final_noninferiority_margin_pct"])
    verdict = ("SAFE_ACCELERATOR" if safe else
               "ACCELERATION_POSITIVE_BUT_UNSAFE" if accel else
               "NEUTRAL" if neutral else "NEGATIVE_OR_UNSAFE")

    beta = 1.0 / kT
    summary = dict(
        cell=tag, n_pairs=R, n_replicas=N,
        horizon_ps=float(t[-1]),
        reference=dict(dF_kT=float(ref["dF_barrier"]) * beta,
                       dF_kJmol=float(ref["dF_barrier"]),
                       dU_kT=float(ref["dU_barrier"]) * beta,
                       mTdS_kT=float(ref["mTdS_barrier"]) * beta,
                       entropic_fraction=float(ref["mTdS_barrier"]
                                               / ref["dF_barrier"]),
                       anchor_paper_dF_kJmol="24.2 +- 2.6 (Krokidas FF, NPT 300 K)"),
        d_int_pct=dict(median=med, ci95=[lo, hi], wins=int((d_int < 0).sum())),
        d_final_pct=dict(median=med_fin, ci95=[flo, fhi],
                         wins=int((d_fin < 0).sum())),
        health=dict(median_min_ess_frac=ess_med, median_max_wmax=wmax_med,
                    worst_min_ess_frac=float(np.nanmin(ess_min)),
                    worst_max_wmax=float(np.nanmax(wmax_mx)), ok=bool(health_ok)),
        transits=dict(abf=int(np.asarray(runs["abf"]["n_crossings"]).sum()),
                      uni=int(np.asarray(runs["fr_uniform"]["n_crossings"]).sum())),
        gate=gate, time_to_accuracy=speed, verdict=verdict,
        fr_rate=json.loads(str(runs["fr_uniform"]["meta"]))["fr_rate"])
    with open(os.path.join(ZIF, f"summary_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    lines = ["seed,int_abf,int_uni,d_int_pct,final_abf,final_uni,d_final_pct"]
    s0 = pre["production"]["seed_first"]
    for r in range(R):
        lines.append(f"{s0 + r},{I['abf'][r]:.5f},{I['fr_uniform'][r]:.5f},"
                     f"{d_int[r]:.3f},{fin['abf'][r]:.6f},{fin['fr_uniform'][r]:.6f},"
                     f"{d_fin[r]:.3f}")
    with open(os.path.join(ZIF, f"comparison_{tag}.csv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    rf = summary["reference"]
    print(f"ZIF-8 {tag}: barrier dF={rf['dF_kT']:.2f} kT = {rf['dF_kJmol']:.1f} kJ/mol "
          f"(dU {rf['dU_kT']:.2f}, -TdS {rf['mTdS_kT']:.2f}, "
          f"{100*rf['entropic_fraction']:.0f}% entropic); "
          f"anchor paper {rf['anchor_paper_dF_kJmol']}")
    print(f"  Delta I_F    median {med:+.2f}%  CI95 [{lo:+.2f}, {hi:+.2f}]  "
          f"wins {summary['d_int_pct']['wins']}/{R}")
    print(f"  Delta e_F(T) median {med_fin:+.2f}%  CI95 [{flo:+.2f}, {fhi:+.2f}]  "
          f"wins {summary['d_final_pct']['wins']}/{R}")
    print(f"  health: ESS/N {ess_med:.3f} (worst {summary['health']['worst_min_ess_frac']:.3f}), "
          f"wmax {wmax_med:.4f}, ok={health_ok}")
    print(f"  transits: abf {summary['transits']['abf']} vs uni "
          f"{summary['transits']['uni']}")
    fmt = lambda x: "n/a" if x is None else f"{x:.5f}"
    print(f"  hidden gate JS vs umbrella reference: abf "
          f"{fmt(gate['abf']['js_vs_reference_cumulative'])} vs uni "
          f"{fmt(gate['fr_uniform']['js_vs_reference_cumulative'])}"
          f"{'  <-- FR WORSE (conditional damage)' if gate['fr_gate_worse_than_abf'] else ''}")
    print(f"  A_gate at the window: abf {fmt(gate['abf']['mean_A_gate'])}, uni "
          f"{fmt(gate['fr_uniform']['mean_A_gate'])}, reference "
          f"{gate['reference']['mean_A_gate']:.4f} A "
          f"(samples abf {gate['abf']['n_gate_samples']:.0f}, "
          f"uni {gate['fr_uniform']['n_gate_samples']:.0f})")
    for name, sp in speed.items():
        ss = f"{sp['speedup']:.2f}x" if sp["speedup"] else sp["status"]
        print(f"  tau[{name}]: abf {sp['tau_abf']:.1f} vs uni {sp['tau_uni']:.1f} "
              f"-> {ss}")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
