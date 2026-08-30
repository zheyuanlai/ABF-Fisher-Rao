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
    """First t with ``curve <= eps`` sustained for a FULL ``persist * T``.

    The persistence window must fit inside the data.  Letting it truncate at
    the tail makes the criterion easiest exactly where the evidence is
    weakest -- a single final save below eps would report a tau with zero
    persistence -- and the preregistration says censoring is reported, never
    imputed."""
    T = times[-1]
    below = np.asarray(curve) <= eps
    for i in range(len(times)):
        if not below[i]:
            continue
        if times[i] + persist * T > T + 1e-12:
            break
        j = np.searchsorted(times, times[i] + persist * T, side="left")
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
    assert bool(ref["accepted"]), \
        "the umbrella reference on disk FAILED its acceptance gates"
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
    assert R == pre["production"]["n_seeds"], \
        f"{R} seed pairs on disk, preregistration says {pre['production']['n_seeds']}"
    I = {m: np.trapezoid(err[m], t, axis=0) for m in err}
    # The arms are BIT-IDENTICAL before fr_start (regression-tested), so that
    # segment contributes exactly zero to the difference while contributing a
    # large share of I_F -- e_F is biggest early.  The frozen primary keeps the
    # campaign's full-horizon convention; this declared secondary restricts to
    # the window where the arms can actually differ, and is reported next to it
    # so the dilution is visible rather than silently absorbed by the -10%
    # threshold.
    post = t >= pre["sampler"]["fr_start_steps"] * pre["sampler"]["dt"]
    I_post = {m: np.trapezoid(err[m][post], t[post], axis=0) for m in err}
    d_int_post = 100.0 * (I_post["fr_uniform"] - I_post["abf"]) / I_post["abf"]
    fin = {m: err[m][-1] for m in err}
    d_int = 100.0 * (I["fr_uniform"] - I["abf"]) / I["abf"]
    d_fin = 100.0 * (fin["fr_uniform"] - fin["abf"]) / fin["abf"]
    lo, hi = boot_median(d_int, BOOT_SEED)
    plo, phi_ = boot_median(d_int_post, BOOT_SEED)
    med_post = float(np.median(d_int_post))
    pre_share = float(np.median(1.0 - I_post["abf"] / I["abf"]))
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
    gref = np.asarray(ref["gate_hist_window_xi"], float)      # (n_xi, n_gate)
    assert np.allclose(np.asarray(ref["gate_edges"], float),
                       np.asarray(runs["abf"]["gate_edges"], float)), \
        "reference and production gate histograms use different bins"
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from run_zif8_screen import gate_js_series
    min_cell = pre["screen"]["gate_min_samples"]
    gate = {}
    for m in runs:
        cum = np.asarray(runs[m]["gate_hist_cumulative"], float)   # (R,n_xi,n_g)
        blocks = np.asarray(runs[m]["gate_hist_block"], float).sum(axis=1)
        js_t = gate_js_series(blocks, gref, min_cell)
        edges = np.asarray(runs[m]["gate_edges"], float)
        mids = 0.5 * (edges[1:] + edges[:-1])
        cum_xa = cum.sum(axis=0)                                   # (n_xi,n_g)
        w = cum_xa.sum(axis=0)
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
            js_vs_reference_cumulative=(
                float(np.mean(js_divergence(
                    cum_xa[(cum_xa.sum(1) >= min_cell) & (gref.sum(1) >= min_cell)],
                    gref[(cum_xa.sum(1) >= min_cell) & (gref.sum(1) >= min_cell)])))
                if enough and ((cum_xa.sum(1) >= min_cell)
                               & (gref.sum(1) >= min_cell)).any() else None),
            mean_A_gate=(float((w * mids).sum() / n_gate) if enough else None),
            crossing_A_gate=(float(np.mean(runs[m]["cross_gate_samples"]))
                             if np.asarray(runs[m]["cross_gate_samples"]).size
                             else None),
            n_crossing_samples=int(np.asarray(runs[m]["cross_gate_samples"]).size))
    ref_mids = 0.5 * (np.asarray(ref["gate_edges"], float)[1:]
                      + np.asarray(ref["gate_edges"], float)[:-1])
    gref_1d = gref.sum(axis=0)
    gate["reference"] = dict(mean_A_gate=float((gref_1d * ref_mids).sum()
                                               / max(gref_1d.sum(), 1e-9)))
    # the conditional-limitation read-out: is the FR arm's gate conditional
    # WORSE than plain ABF's?  That is the R15 failure mode, made visible.
    ja, ju = (gate["abf"]["js_vs_reference_cumulative"],
              gate["fr_uniform"]["js_vs_reference_cumulative"])
    gate["fr_gate_worse_than_abf"] = (None if (ja is None or ju is None)
                                      else bool(ju > ja))

    # A genuine partition.  The previous tree let an arm 9.9% WORSE with a CI
    # excluding zero on the bad side report NEUTRAL, let an arm in genealogical
    # collapse report NEUTRAL (health gated only the "safe" label), and labelled
    # a 12%-better arm NEGATIVE_OR_UNSAFE when its CI merely straddled zero.
    # Equivalence is now TOST-style -- the whole CI inside the margin -- which
    # is the rule the gateway study already wrote into this project's methods.
    M = abs(rule["median_rel_change_pct_max"])
    accel = med <= -M and hi < rule["ci95_upper_pct_max"]
    harmful = med >= M or lo > 0.0
    equivalent = (lo > -M) and (hi < M)
    final_ok = med_fin <= rule["final_noninferiority_margin_pct"]
    if accel and health_ok and final_ok:
        verdict = "SAFE_ACCELERATOR"
    elif accel:
        verdict = "ACCELERATION_POSITIVE_BUT_UNSAFE"
    elif harmful:
        verdict = "HARMFUL"
    elif equivalent and health_ok and final_ok:
        verdict = "NEUTRAL"
    else:
        verdict = "INCONCLUSIVE"

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
        d_int_pct_post_fr=dict(median=med_post, ci95=[plo, phi_],
                               wins=int((d_int_post < 0).sum()),
                               pre_fr_share_of_I_F=pre_share,
                               _note=("declared secondary: the arms are "
                                      "bit-identical before fr_start, so the "
                                      "pre-FR segment can only dilute")),
        health_flags=dict(accel=bool(accel), harmful=bool(harmful),
                          equivalent_tost=bool(equivalent),
                          final_noninferior=bool(final_ok),
                          genealogy_ok=bool(health_ok)),
        d_final_pct=dict(median=med_fin, ci95=[flo, fhi],
                         wins=int((d_fin < 0).sum())),
        health=dict(median_min_ess_frac=ess_med, median_max_wmax=wmax_med,
                    worst_min_ess_frac=float(np.nanmin(ess_min)),
                    worst_max_wmax=float(np.nanmax(wmax_mx)), ok=bool(health_ok)),
        transits=dict(
            abf=int(np.asarray(runs["abf"]["cross_gate_samples"]).size),
            uni=int(np.asarray(runs["fr_uniform"]["cross_gate_samples"]).size),
            _note=("counted from the per-event list, not the per-walker "
                   "cumulative counter, which a clone must not inherit")),
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
          f"wins {summary['d_int_pct']['wins']}/{R}   [FROZEN PRIMARY]")
    print(f"  Delta I_F post-FR only: median {med_post:+.2f}%  "
          f"CI95 [{plo:+.2f}, {phi_:+.2f}]  wins "
          f"{summary['d_int_pct_post_fr']['wins']}/{R}   "
          f"(the pre-FR segment, where the arms are identical, is "
          f"{100*pre_share:.0f}% of I_F)")
    print(f"  Delta e_F(T) median {med_fin:+.2f}%  CI95 [{flo:+.2f}, {fhi:+.2f}]  "
          f"wins {summary['d_final_pct']['wins']}/{R}")
    print(f"  health: ESS/N {ess_med:.3f} (worst {summary['health']['worst_min_ess_frac']:.3f}), "
          f"wmax {wmax_med:.4f}, ok={health_ok}")
    print(f"  transits (events): abf {summary['transits']['abf']} vs uni "
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
