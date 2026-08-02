#!/usr/bin/env python
"""Matched-seed comparison of ABF / sham / oracle mFR / practical mFR at the anchor.

Every arm inside a ``(config, seed)`` row shares initial conditions and Langevin noise, so
the comparison is paired and the seed-to-seed spread cancels.  Differences are therefore
reported as **paired** medians with a bootstrap CI over seeds and a win rate, not as a
difference of independent means -- the mistake that made an earlier valine diagnostic read
as a failure.

    python scripts/analyze_gateway_anchor.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

ESS_GATE = 0.30          # ESS_anc / N floor, preregistered
WMAX_GATE = 0.05         # largest lineage share ceiling, preregistered
EQUIV_MARGIN = 0.10      # +/-10 %: the study's standing practical-equivalence margin
RNG = np.random.default_rng(20260802)


def boot_ci(x, n=10_000, lo=2.5, hi=97.5):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, x.size, size=(n, x.size))
    m = np.median(x[idx], axis=1)
    return float(np.percentile(m, lo)), float(np.percentile(m, hi))


def verdict(med_pct, ci):
    """Practical-equivalence call on the standing +/-10 % margin.

    Sign convention: a NEGATIVE percentage is an improvement (the error went down).
    """
    if not np.isfinite(med_pct):
        return "undetermined"
    if abs(med_pct) <= EQUIV_MARGIN * 100 and abs(ci[0]) <= EQUIV_MARGIN * 100 \
            and abs(ci[1]) <= EQUIV_MARGIN * 100:
        return "equivalent"
    return "improvement" if med_pct < 0 else "harmful"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results/gateway_anchor/production"))
    a = ap.parse_args()
    d = np.load(os.path.join(a.dir, "raw.npz"), allow_pickle=True)
    prov = json.load(open(os.path.join(a.dir, "provenance.json")))
    anc = prov["anchor"]
    N = prov["n_walkers"]
    T = prov["T_total"]

    method = d["method"].astype(str)
    init = d["init"].astype(str)
    gamma = d["gamma"].astype(float)
    seed = d["seed"].astype(int)
    arms = list(dict.fromkeys(method.tolist()))

    print(f"anchor beta={anc['beta']:g} s={anc['s']:g} r={anc['r']:g} "
          f"[{anc['regime']}], frozen {prov['frozen_at']}")
    print(f"{len(method)} runs; arms {arms}; gamma ladder {prov['gamma_ladder']} "
          f"(frozen {prov['gamma_frozen']:g}); N={N}, T={T:g}\n")

    METRICS = [("int_l2_f", "I_F", True), ("int_l2_fp", "I_F'", True),
               ("integrated_deficit", "int. deficit", True),
               ("T_est", "T_est", True), ("T_hit", "T_hit", True)]

    rows = []
    for g in sorted(set(gamma)):
        for ini in sorted(set(init)):
            base = {}
            for s in sorted(set(seed)):
                m = (gamma == g) & (init == ini) & (seed == s) & (method == "abf")
                if m.any():
                    base[s] = {k: float(d[k][m][0]) for k, _, _ in METRICS}
            for arm in arms:
                if arm == "abf":
                    continue
                sel = (gamma == g) & (init == ini) & (method == arm)
                seeds_here = seed[sel]
                rec = dict(gamma=g, init=ini, arm=arm, n_seeds=int(sel.sum()))
                for key, lab, _ in METRICS:
                    v = d[key][sel].astype(float)
                    b = np.array([base[s][key] for s in seeds_here], dtype=float)
                    ok = np.isfinite(v) & np.isfinite(b) & (np.abs(b) > 0)
                    rel = np.where(ok, 100.0 * (v - b) / np.where(ok, b, 1.0), np.nan)
                    med = float(np.nanmedian(rel)) if ok.any() else float("nan")
                    ci = boot_ci(rel)
                    rec[f"{key}_pct"] = med
                    rec[f"{key}_ci_lo"], rec[f"{key}_ci_hi"] = ci
                    rec[f"{key}_wins"] = int(np.sum(v[ok] < b[ok]))
                    rec[f"{key}_verdict"] = verdict(med, ci)
                    rec[f"{key}_abf"] = float(np.nanmedian(b))
                    rec[f"{key}_arm"] = float(np.nanmedian(v))
                rec["min_ess_frac"] = float(np.nanmedian(d["min_ess_frac"][sel]))
                rec["final_ess_frac"] = float(np.nanmedian(d["final_ess"][sel]) / N)
                rec["max_wmax"] = float(np.nanmedian(d["max_wmax"][sel]))
                rec["repl_fraction"] = float(np.nanmedian(d["repl_fraction"][sel]))
                rec["n_die"] = float(np.nanmedian(d["n_die"][sel]))
                rec["n_clone"] = float(np.nanmedian(d["n_clone"][sel]))
                rec["health_ok"] = bool(rec["min_ess_frac"] >= ESS_GATE
                                        and rec["max_wmax"] <= WMAX_GATE)
                rows.append(rec)

    # ------------------------------------------------------- sham/oracle match check
    print("Sham intensity match (must be exact -- the sham copies its partner's counts):")
    bad = 0
    for g in sorted(set(gamma)):
        for ini in sorted(set(init)):
            for s in sorted(set(seed)):
                m1 = (gamma == g) & (init == ini) & (seed == s) & (method == "sham")
                m2 = (gamma == g) & (init == ini) & (seed == s) & (method == "fr_oracle")
                if m1.any() and m2.any():
                    if not (d["n_die"][m1][0] == d["n_die"][m2][0]
                            and d["n_clone"][m1][0] == d["n_clone"][m2][0]):
                        bad += 1
    print(f"  {bad} mismatches out of {len(set(gamma)) * len(set(init)) * len(set(seed))} "
          f"(sham vs fr_oracle clone/delete counts)\n")

    # ------------------------------------------------------------------ report
    for ini in sorted(set(init)):
        tag = "HEADLINE" if ini == "left" else "MECHANISM CONTROL (discovery is free)"
        print(f"{'=' * 100}\ninit = {ini}   [{tag}]\n{'=' * 100}")
        print(f"{'gamma':>6s} {'arm':>14s} {'I_F %':>9s} {'95% CI':>17s} {'win':>6s} "
              f"{'verdict':>12s} {'I_Fp %':>9s} {'deficit %':>10s} {'T_est %':>9s} "
              f"{'ESSmin/N':>9s} {'wmax':>7s} {'health':>7s}")
        for r in [x for x in rows if x["init"] == ini]:
            print(f"{r['gamma']:6.1f} {r['arm']:>14s} {r['int_l2_f_pct']:9.2f} "
                  f"[{r['int_l2_f_ci_lo']:7.2f},{r['int_l2_f_ci_hi']:7.2f}] "
                  f"{r['int_l2_f_wins']:3d}/{r['n_seeds']:<2d} "
                  f"{r['int_l2_f_verdict']:>12s} {r['int_l2_fp_pct']:9.2f} "
                  f"{r['integrated_deficit_pct']:10.2f} {r['T_est_pct']:9.2f} "
                  f"{r['min_ess_frac']:9.3f} {r['max_wmax']:7.3f} "
                  f"{'PASS' if r['health_ok'] else 'FAIL':>7s}")
        print()

    csv_path = os.path.join(a.dir, "anchor_comparison.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    # headline: the frozen rate, headline init
    gf = prov["gamma_frozen"]
    head = {r["arm"]: r for r in rows if r["init"] == "left" and r["gamma"] == gf}
    # best rate that passes the health gates, per arm -- so a null cannot be blamed on rate
    best = {}
    for arm in arms:
        if arm == "abf":
            continue
        cand = [r for r in rows if r["init"] == "left" and r["arm"] == arm
                and r["health_ok"]]
        best[arm] = min(cand, key=lambda r: r["int_l2_f_pct"]) if cand else None

    summary = dict(anchor=anc, gamma_frozen=gf, headline=head, best_healthy=best,
                   sham_count_mismatches=bad, ess_gate=ESS_GATE, wmax_gate=WMAX_GATE,
                   equivalence_margin_pct=EQUIV_MARGIN * 100, rows=rows)
    with open(os.path.join(a.dir, "anchor_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"wrote {os.path.join(a.dir, 'anchor_summary.json')}")

    print("\n" + "=" * 100)
    print("HEADLINE (init=left, frozen gamma = %g):" % gf)
    for arm, r in head.items():
        print(f"  {arm:>14s}: I_F {r['int_l2_f_pct']:+.2f}% "
              f"[{r['int_l2_f_ci_lo']:+.2f},{r['int_l2_f_ci_hi']:+.2f}] -> "
              f"{r['int_l2_f_verdict']};  deficit {r['integrated_deficit_pct']:+.1f}%;  "
              f"health {'PASS' if r['health_ok'] else 'FAIL'} "
              f"(ESS/N {r['min_ess_frac']:.3f}, wmax {r['max_wmax']:.3f})")
    print("\nBest health-passing rate per arm (init=left):")
    for arm, r in best.items():
        print(f"  {arm:>14s}: " + ("no rate passes the health gates" if r is None else
              f"gamma {r['gamma']:g}, I_F {r['int_l2_f_pct']:+.2f}% "
              f"[{r['int_l2_f_ci_lo']:+.2f},{r['int_l2_f_ci_hi']:+.2f}] -> "
              f"{r['int_l2_f_verdict']}, deficit {r['integrated_deficit_pct']:+.1f}%"))
    make_figure(os.path.join(a.dir, "gateway_anchor.pdf"), d, prov, rows)


def make_figure(path, d, prov, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method = d["method"].astype(str)
    init = d["init"].astype(str)
    gamma = d["gamma"].astype(float)
    N = prov["n_walkers"]
    gf = prov["gamma_frozen"]
    arms = ["abf", "sham", "fr_oracle", "fr_estimated"]
    colors = {"abf": "0.25", "sham": "#8172B3", "fr_oracle": "#DD8452",
              "fr_estimated": "#55A868"}
    t = d["t"][0]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))

    ax = axes[0, 0]
    for arm in arms:
        m = (method == arm) & (init == "left") & (gamma == gf)
        if not m.any():
            continue
        P = d["P_regions"][m][:, :, 2]
        ax.plot(t, P.mean(0), color=colors[arm], lw=1.8, label=arm)
    m = (method == "abf") & (init == "left") & (gamma == gf)
    ax.plot(t, d["Q_regions"][m][:, :, 2].mean(0), color="crimson", lw=1.4, ls="--",
            label="bias-aware target $Q^*_+$")
    ax.set_xlabel("t"); ax.set_ylabel("population of $B_+$")
    ax.set_title(f"right-basin occupancy at the anchor ($\\gamma={gf:g}$)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for arm in arms:
        m = (method == arm) & (init == "left") & (gamma == gf)
        if m.any():
            ax.plot(t, d["l2_f_t"][m].mean(0), color=colors[arm], lw=1.8, label=arm)
    ax.set_xlabel("t"); ax.set_ylabel("$\\|\\widehat{F}_t - F\\|_{L^2}$")
    ax.set_yscale("log"); ax.set_title("free-energy error"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    gl = sorted(set(gamma))
    for arm in arms[1:]:
        y = [next((r["int_l2_f_pct"] for r in rows
                   if r["init"] == "left" and r["arm"] == arm and r["gamma"] == g),
                  np.nan) for g in gl]
        lo = [next((r["int_l2_f_ci_lo"] for r in rows if r["init"] == "left"
                    and r["arm"] == arm and r["gamma"] == g), np.nan) for g in gl]
        hi = [next((r["int_l2_f_ci_hi"] for r in rows if r["init"] == "left"
                    and r["arm"] == arm and r["gamma"] == g), np.nan) for g in gl]
        ax.errorbar(gl, y, yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)],
                    color=colors[arm], marker="o", lw=1.6, capsize=3, label=arm)
    ax.axhline(0, color="0.4", lw=1.0)
    ax.axhspan(-10, 10, color="0.9", zorder=0, label="$\\pm10\\%$ equivalence margin")
    ax.set_xscale("log"); ax.set_xlabel("FR rate $\\gamma$")
    ax.set_ylabel("$I_F$ vs ABF (%)  [negative = better]")
    ax.set_title("rate ladder, matched seeds"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    for arm in arms[1:]:
        y = [next((r["min_ess_frac"] for r in rows if r["init"] == "left"
                   and r["arm"] == arm and r["gamma"] == g), np.nan) for g in gl]
        ax.plot(gl, y, color=colors[arm], marker="o", lw=1.6, label=arm)
    ax.axhline(ESS_GATE, color="crimson", ls="--", lw=1.3,
               label=f"gate ESS/N $\\geq$ {ESS_GATE}")
    ax.set_xscale("log"); ax.set_xlabel("FR rate $\\gamma$")
    ax.set_ylabel("min ancestor ESS / N")
    ax.set_title("the cost side: lineage diversity"); ax.legend(fontsize=8)

    fig.suptitle("Entropic gateway, preregistered establishment-limited anchor "
                 f"($\\beta$={prov['anchor']['beta']:g}, $s$={prov['anchor']['s']:g}, "
                 f"$r$={prov['anchor']['r']:g})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, format="pdf", bbox_inches="tight")
    # PNG as well: every other figure in the report is a PNG, and mixing raster
    # and vector includes has bitten this build before.
    fig.savefig(path.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
