#!/usr/bin/env python
"""Apply the frozen confirmatory rule to the fresh-seed gateway run.

The rule is read from the preregistration and applied mechanically.  Nothing here chooses a
rate, a window, or a threshold: those were fixed before the run.

Two statistical points the calibration analysis got wrong and this one does not:

* **A directional call requires an interval that excludes zero.**  Otherwise a wide interval
  around a small median gets labelled an effect, which reports the imprecision of the
  estimate rather than its size.
* **"No interval excluded zero" is not equivalence.**  Failure to detect is not evidence of
  absence.  The shams are judged by two one-sided tests against a predeclared +/-5 % margin:
  equivalence is claimed only when the 90 % CI (the TOST interval at alpha = 0.05) lies
  entirely inside it.

    python scripts/analyze_gateway_confirm.py
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


def boot(x, rng, n=10_000):
    """Bootstrap distribution of the median of a paired difference."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([np.nan])
    idx = rng.integers(0, x.size, size=(n, x.size))
    return np.median(x[idx], axis=1)


def ci(dist, level):
    lo = 50.0 * (1.0 - level)
    return float(np.percentile(dist, lo)), float(np.percentile(dist, 100.0 - lo))


def paired_rel(v_arm, v_base):
    ok = np.isfinite(v_arm) & np.isfinite(v_base) & (np.abs(v_base) > 0)
    out = np.full(v_arm.shape, np.nan)
    out[ok] = 100.0 * (v_arm[ok] - v_base[ok]) / v_base[ok]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results/gateway_anchor/confirmatory"))
    a = ap.parse_args()
    d = np.load(os.path.join(a.dir, "raw.npz"), allow_pickle=True)
    prov = json.load(open(os.path.join(a.dir, "provenance.json")))
    pre = prov["preregistration"]
    rule = pre["success_rule"]
    eqv = pre["sham_equivalence"]
    rng = np.random.default_rng(pre["bootstrap"]["seed"])
    NB = pre["bootstrap"]["n_resamples"]
    N = pre["sampler"]["N"]

    method = d["method"].astype(str)
    init = d["init"].astype(str)
    seed = d["seed"].astype(int)
    has_frozen = "frozen_l2_f_kT" in d.files

    print(f"CONFIRMATORY analysis -- rule frozen in {prov['preregistration_path']}")
    print(f"  seeds {min(seed)}-{max(seed)} ({len(set(seed.tolist()))} fresh), "
          f"arms {sorted(set(method.tolist()))}")
    print(f"  primary: {pre['primary_arm']} on {pre['primary_metric']}; "
          f"median <= {rule['median_rel_change_pct_max']:g}%, "
          f"CI95 upper < {rule['ci95_upper_pct_max']:g}%, "
          f">= {rule['min_seeds_improved']}/{rule['n_seeds']} seeds, "
          f"ESS/N >= {rule['ess_anc_over_N_min']:g}, wmax <= {rule['wmax_max']:g}")
    print(f"  sham equivalence: TOST alpha={eqv['alpha']:g}, "
          f"{int(eqv['ci_level_for_tost'] * 100)}% CI inside "
          f"[{eqv['margin_pct'][0]:g}%, {eqv['margin_pct'][1]:g}%]\n")

    METRICS = ["int_l2_f", "int_l2_fp", "integrated_deficit", "T_est", "T_hit"]
    if has_frozen:
        METRICS.append("frozen_l2_f_kT")

    # Each arm is compared against the ABF baseline from ITS OWN batch. The two FR rates
    # force two batches and the noise stream is keyed to the batch, so a cross-batch baseline
    # shares initial conditions but not noise and its interval does not cancel. If the
    # artifact predates that fix it carries only one baseline; fall back to it and say so.
    group = (d["group"].astype(str) if "group" in d.files
             else np.full(method.shape, "practical"))
    abf_groups = sorted(set(group[method == "abf"].tolist()))
    per_group_baseline = len(abf_groups) > 1
    if not per_group_baseline:
        print(f"*** WARNING: only one ABF batch ({abf_groups}) is present, so arms from the "
              f"other batch are scored against a NOISE-UNMATCHED baseline. Their intervals "
              f"are inflated. See Amendment 1 of the preregistration. ***\n")

    def baseline_for(arm):
        g = group[method == arm]
        return g[0] if per_group_baseline and g.size else abf_groups[0]

    rows = []
    for ini in sorted(set(init.tolist())):
        base = {}
        for gname in abf_groups:
            for s_ in sorted(set(seed.tolist())):
                m = (init == ini) & (seed == s_) & (method == "abf") & (group == gname)
                if m.any():
                    base[(gname, s_)] = {k: float(d[k][m][0]) for k in METRICS}
        for arm in [x for x in pre["arms"] if x != "abf"]:
            sel = (init == ini) & (method == arm)
            if not sel.any():
                continue
            gb = baseline_for(arm)
            sd = seed[sel]
            rec = dict(init=ini, arm=arm, n_seeds=int(sel.sum()),
                       gamma=float(d["gamma"][sel][0]), baseline_batch=str(gb),
                       baseline_noise_matched=bool(per_group_baseline))
            for k in METRICS:
                v = d[k][sel].astype(float)
                b = np.array([base[(gb, x)][k] for x in sd], dtype=float)
                rel = paired_rel(v, b)
                dist = boot(rel, rng, NB)
                lo95, hi95 = ci(dist, 0.95)
                lo90, hi90 = ci(dist, eqv["ci_level_for_tost"])
                rec[f"{k}_pct"] = float(np.nanmedian(rel))
                rec[f"{k}_ci95"] = [lo95, hi95]
                rec[f"{k}_ci90"] = [lo90, hi90]
                rec[f"{k}_wins"] = int(np.sum(v[np.isfinite(rel)] < b[np.isfinite(rel)]))
                rec[f"{k}_abf"] = float(np.nanmedian(b))
                rec[f"{k}_arm"] = float(np.nanmedian(v))
            rec["min_ess_frac"] = float(np.nanmedian(d["min_ess_frac"][sel]))
            rec["max_wmax"] = float(np.nanmedian(d["max_wmax"][sel]))
            rec["n_die"] = float(np.nanmedian(d["n_die"][sel]))
            rec["n_clone"] = float(np.nanmedian(d["n_clone"][sel]))
            rec["health_ok"] = bool(rec["min_ess_frac"] >= rule["ess_anc_over_N_min"]
                                    and rec["max_wmax"] <= rule["wmax_max"])
            rows.append(rec)

    # ------------------------------------- direct FR-vs-its-own-sham contrast
    # Comparing each arm against ABF separately and then eyeballing the two contrasts is a
    # weaker attribution test than comparing the arm against its sham DIRECTLY on the same
    # seed: the direct contrast holds the event schedule and count fixed by construction, so
    # the only thing that differs is the selection direction.
    direct = []
    for sham, partner in pre["sham_partners"].items():
        for ini in sorted(set(init.tolist())):
            va, vs, sds = [], [], []
            for s_ in sorted(set(seed.tolist())):
                m1 = (init == ini) & (seed == s_) & (method == partner)
                m2 = (init == ini) & (seed == s_) & (method == sham)
                if m1.any() and m2.any():
                    va.append(float(d[pre["primary_metric"]][m1][0]))
                    vs.append(float(d[pre["primary_metric"]][m2][0]))
                    sds.append(s_)
            va, vs = np.asarray(va), np.asarray(vs)
            rel = paired_rel(va, vs)      # FR arm relative to its own sham
            dist = boot(rel, rng, NB)
            lo95, hi95 = ci(dist, 0.95)
            rec = dict(arm=partner, sham=sham, init=ini, n_seeds=len(va),
                       pct=float(np.nanmedian(rel)), ci95=[lo95, hi95],
                       wins=int(np.sum(va < vs)), excludes_zero=bool(lo95 * hi95 > 0))
            # Same contrast on the estimator-independent endpoint. This matters when a sham
            # is not itself equivalent to ABF: the arm-vs-sham difference is still the
            # quantity that isolates direction, on whichever endpoint it is measured.
            if has_frozen:
                fa = np.asarray([float(d["frozen_l2_f_kT"][(init == ini) & (seed == x)
                                                           & (method == partner)][0])
                                 for x in sds])
                fs = np.asarray([float(d["frozen_l2_f_kT"][(init == ini) & (seed == x)
                                                           & (method == sham)][0])
                                 for x in sds])
                frel = paired_rel(fa, fs)
                fd = boot(frel, rng, NB)
                flo, fhi = ci(fd, 0.95)
                rec.update(frozen_pct=float(np.nanmedian(frel)), frozen_ci95=[flo, fhi],
                           frozen_wins=int(np.sum(fa < fs)),
                           frozen_excludes_zero=bool(flo * fhi > 0))
            direct.append(rec)
    print("Direct contrast -- each FR arm against ITS OWN sham, same seed, same event "
          "schedule\n(this is the attribution test: only the selection direction differs)")
    for r in direct:
        tag = "" if r["init"] == "left" else "   [mechanism control]"
        line = (f"  {r['arm']:>14s} vs {r['sham']:<15s} {r['pct']:+7.2f}% "
                f"[{r['ci95'][0]:+6.2f},{r['ci95'][1]:+6.2f}]  {r['wins']:2d}/{r['n_seeds']} "
                f"{'CI excl. 0' if r['excludes_zero'] else 'CI incl. 0'}")
        if "frozen_pct" in r:
            line += (f" | frozen {r['frozen_pct']:+7.2f}% "
                     f"[{r['frozen_ci95'][0]:+6.2f},{r['frozen_ci95'][1]:+6.2f}] "
                     f"{r['frozen_wins']:2d}/{r['n_seeds']} "
                     f"{'excl. 0' if r['frozen_excludes_zero'] else 'incl. 0'}")
        print(line + tag)
    print()

    # -------------------------------------------------- sham intensity match
    bad = 0
    for sham, partner in pre["sham_partners"].items():
        for ini in sorted(set(init.tolist())):
            for s_ in sorted(set(seed.tolist())):
                m1 = (init == ini) & (seed == s_) & (method == sham)
                m2 = (init == ini) & (seed == s_) & (method == partner)
                if m1.any() and m2.any():
                    if not (d["n_die"][m1][0] == d["n_die"][m2][0]
                            and d["n_clone"][m1][0] == d["n_clone"][m2][0]):
                        bad += 1
    print(f"sham intensity match: {bad} mismatches "
          f"(each sham vs its own partner, all seeds and inits)\n")

    # ------------------------------------------------------------- report
    mk = pre["primary_metric"]
    for ini in sorted(set(init.tolist())):
        tag = "PRIMARY" if ini == "left" else "mechanism control (discovery is free)"
        print("=" * 108)
        print(f"init = {ini}   [{tag}]")
        print("=" * 108)
        hdr = (f"{'arm':>16s} {'gamma':>6s} {'I_F %':>8s} {'95% CI':>18s} "
               f"{'90% CI':>18s} {'won':>6s} {'ESS/N':>7s} {'wmax':>7s} {'health':>7s}")
        if has_frozen:
            hdr += f" {'frozen %':>9s}"
        print(hdr)
        for r in [x for x in rows if x["init"] == ini]:
            line = (f"{r['arm']:>16s} {r['gamma']:6.1f} {r[mk + '_pct']:8.2f} "
                    f"[{r[mk + '_ci95'][0]:7.2f},{r[mk + '_ci95'][1]:7.2f}] "
                    f"[{r[mk + '_ci90'][0]:7.2f},{r[mk + '_ci90'][1]:7.2f}] "
                    f"{r[mk + '_wins']:3d}/{r['n_seeds']:<2d} "
                    f"{r['min_ess_frac']:7.3f} {r['max_wmax']:7.3f} "
                    f"{'PASS' if r['health_ok'] else 'FAIL':>7s}")
            if has_frozen:
                line += f" {r['frozen_l2_f_kT_pct']:9.2f}"
            print(line)
        print()

    # ------------------------------------------------------ frozen decision
    prim = next(r for r in rows if r["init"] == "left" and r["arm"] == pre["primary_arm"])
    checks = {
        "median <= %g%%" % rule["median_rel_change_pct_max"]:
            (prim[mk + "_pct"] <= rule["median_rel_change_pct_max"], prim[mk + "_pct"]),
        "CI95 upper < %g%%" % rule["ci95_upper_pct_max"]:
            (prim[mk + "_ci95"][1] < rule["ci95_upper_pct_max"], prim[mk + "_ci95"][1]),
        "seeds improved >= %d" % rule["min_seeds_improved"]:
            (prim[mk + "_wins"] >= rule["min_seeds_improved"], prim[mk + "_wins"]),
        "ESS/N >= %g" % rule["ess_anc_over_N_min"]:
            (prim["min_ess_frac"] >= rule["ess_anc_over_N_min"], prim["min_ess_frac"]),
        "wmax <= %g" % rule["wmax_max"]:
            (prim["max_wmax"] <= rule["wmax_max"], prim["max_wmax"]),
    }
    print("=" * 108)
    print(f"PRIMARY CLAIM -- {pre['primary_arm']}, init=left, metric {mk}")
    for k, (ok, val) in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k:<28s} measured {val:.4g}")
    primary_pass = all(ok for ok, _ in checks.values())
    print(f"  => primary rule: {'PASS' if primary_pass else 'FAIL'}")

    # TOST on each sham
    lo_m, hi_m = eqv["margin_pct"]
    tost = {}
    print(f"\nSHAM EQUIVALENCE (TOST, margin [{lo_m:g}%, {hi_m:g}%], "
          f"{int(eqv['ci_level_for_tost'] * 100)}% CI must lie inside)")
    for sham in pre["sham_partners"]:
        r = next((x for x in rows if x["init"] == "left" and x["arm"] == sham), None)
        if r is None:
            continue
        lo, hi = r[mk + "_ci90"]
        ok = (lo >= lo_m) and (hi <= hi_m)
        lo95, hi95 = r[mk + "_ci95"]
        ok95 = (lo95 >= lo_m) and (hi95 <= hi_m)
        tost[sham] = dict(equivalent=bool(ok), ci90=[lo, hi], ci95=[lo95, hi95],
                          median=r[mk + "_pct"], also_at_95=bool(ok95))
        note = "" if ok95 == ok else ("  (the 95 % CI does NOT also fit"
                                      if ok else "  (nor does the 95 % CI)")
        print(f"  [{'PASS' if ok else 'FAIL'}] {sham:<16s} median {r[mk + '_pct']:+6.2f}%  "
              f"90% CI [{lo:+6.2f},{hi:+6.2f}]  95% CI [{lo95:+6.2f},{hi95:+6.2f}]{note}"
              + (")" if note else ""))

    partner_sham = next((s for s, p in pre["sham_partners"].items()
                         if p == pre["primary_arm"]), None)
    headline = primary_pass and tost.get(partner_sham, {}).get("equivalent", False)
    print(f"\n{'=' * 108}")
    print(f"HEADLINE -- 'the deployable gain is directional Fisher-Rao selection, not "
          f"generic turnover': {'SUPPORTED' if headline else 'NOT SUPPORTED'}")
    if not headline:
        if not primary_pass:
            print("  the primary arm did not meet the frozen accuracy/health rule")
        elif not tost.get(partner_sham, {}).get("equivalent", False):
            print(f"  {partner_sham} was not shown equivalent within the margin; the gain "
                  f"cannot be attributed to direction on this evidence")
    if has_frozen:
        fz = prim["frozen_l2_f_kT_pct"]
        fz95 = prim["frozen_l2_f_kT_ci95"]
        agree = (fz < 0) == (prim[mk + "_pct"] < 0)
        print(f"\nESTIMATOR-INDEPENDENT CONFIRMATION (frozen bias, no adaptation, no "
              f"birth-death)")
        print(f"  {pre['primary_arm']}: {fz:+.2f}% [{fz95[0]:+.2f},{fz95[1]:+.2f}] "
              f"vs ABF on the reconstructed free energy")
        print(f"  sign {'AGREES with' if agree else 'DISAGREES with'} the online endpoint"
              + ("" if agree else " -- the frozen-bias reading wins, per the prereg"))
        # The shams must be read on this endpoint too: equivalence on the online metric does
        # not imply equivalence on the reconstructed bias, and the difference is informative.
        for sham in pre["sham_partners"]:
            rs = next((x for x in rows if x["init"] == "left" and x["arm"] == sham), None)
            if rs is None:
                continue
            lo, hi = rs["frozen_l2_f_kT_ci90"]
            inside = (lo >= eqv["margin_pct"][0]) and (hi <= eqv["margin_pct"][1])
            print(f"  {sham:>16s}: {rs['frozen_l2_f_kT_pct']:+.2f}% "
                  f"90% CI [{lo:+.2f},{hi:+.2f}] -> "
                  f"{'equivalent' if inside else 'NOT equivalent on this endpoint'}")

    out = dict(preregistration=pre, primary=prim, direct_vs_sham=direct,
               per_group_baseline=bool(per_group_baseline), abf_batches=abf_groups,
               primary_checks=
               {k: dict(pass_=bool(v[0]), value=float(v[1])) for k, v in checks.items()},
               primary_pass=bool(primary_pass), sham_tost=tost,
               headline_supported=bool(headline),
               sham_count_mismatches=bad, rows=rows)
    with open(os.path.join(a.dir, "confirmatory_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    flat = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}
            for r in rows]
    with open(os.path.join(a.dir, "confirmatory_comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader(); w.writerows(flat)
    print(f"\nwrote {a.dir}/confirmatory_summary.json and confirmatory_comparison.csv")
    make_figure(os.path.join(a.dir, "gateway_confirmatory.pdf"), d, pre, rows, tost,
                has_frozen)


def make_figure(path, d, pre, rows, tost, has_frozen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method = d["method"].astype(str)
    init = d["init"].astype(str)
    t = d["t"][0]
    mk = pre["primary_metric"]
    colors = {"abf": "0.25", "fr_estimated": "#55A868", "sham_practical": "#8172B3",
              "fr_oracle": "#DD8452", "sham_oracle": "#C44E52"}
    order = pre["arms"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for arm in order:
        m = (method == arm) & (init == "left")
        if m.any():
            ax.plot(t, d["P_regions"][m][:, :, 2].mean(0), color=colors[arm], lw=1.7,
                    label=arm)
    m = (method == "abf") & (init == "left")
    ax.plot(t, d["Q_regions"][m][:, :, 2].mean(0), color="crimson", lw=1.3, ls="--",
            label="target $Q^*_+$")
    ax.set_xlabel("t"); ax.set_ylabel("population of $B_+$")
    ax.set_title("right-chamber occupancy, 32 fresh seeds"); ax.legend(fontsize=7)

    ax = axes[0, 1]
    for arm in order:
        m = (method == arm) & (init == "left")
        if m.any():
            ax.plot(t, d["l2_f_t"][m].mean(0), color=colors[arm], lw=1.7, label=arm)
    ax.set_yscale("log"); ax.set_xlabel("t")
    ax.set_ylabel(r"$\|\widehat F_t - F\|_{L^2}$")
    ax.set_title("free-energy error"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    left = [r for r in rows if r["init"] == "left"]
    y = np.arange(len(left))
    med = [r[mk + "_pct"] for r in left]
    lo = [r[mk + "_pct"] - r[mk + "_ci95"][0] for r in left]
    hi = [r[mk + "_ci95"][1] - r[mk + "_pct"] for r in left]
    ax.barh(y, med, color=[colors[r["arm"]] for r in left], height=0.6)
    ax.errorbar(med, y, xerr=[lo, hi], fmt="none", ecolor="k", capsize=4, lw=1.2)
    m0, m1 = pre["sham_equivalence"]["margin_pct"]
    ax.axvspan(m0, m1, color="0.88", zorder=0,
               label=f"equivalence margin [{m0:g}%, {m1:g}%]")
    ax.axvline(pre["success_rule"]["ci95_upper_pct_max"], color="crimson", ls="--", lw=1.3,
               label=f"success bar {pre['success_rule']['ci95_upper_pct_max']:g}%")
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([r["arm"] for r in left], fontsize=8)
    ax.set_xlabel(r"$I_F$ vs ABF (%)  [negative = better]")
    ax.set_title("primary endpoint, paired, 95% CI"); ax.legend(fontsize=7, loc="lower right")

    ax = axes[1, 1]
    if has_frozen:
        fz = [r["frozen_l2_f_kT_pct"] for r in left]
        flo = [r["frozen_l2_f_kT_pct"] - r["frozen_l2_f_kT_ci95"][0] for r in left]
        fhi = [r["frozen_l2_f_kT_ci95"][1] - r["frozen_l2_f_kT_pct"] for r in left]
        ax.barh(y, fz, color=[colors[r["arm"]] for r in left], height=0.6)
        ax.errorbar(fz, y, xerr=[flo, fhi], fmt="none", ecolor="k", capsize=4, lw=1.2)
        ax.axvline(0, color="k", lw=0.9)
        ax.set_yticks(y); ax.set_yticklabels([r["arm"] for r in left], fontsize=8)
        ax.set_xlabel(r"frozen-bias $\|\widehat F - F\|_{L^2}$ vs ABF (%)")
        ax.set_title("estimator-independent endpoint\n(no adaptation, no birth--death)")
    else:
        ax.axis("off")

    fig.suptitle("Gateway anchor, CONFIRMATORY run: frozen rates, 32 fresh seeds, "
                 "no tuning", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, format="pdf", bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
