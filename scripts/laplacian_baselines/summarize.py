#!/usr/bin/env python3
"""Summary tables of the LRS Laplacian baseline (results/laplacian_baselines/analysis/SUMMARY.md).

Reads analysis/<case>.json (analyze.py: paired eqb_metrics contrasts, three arms at identical seeds and force
evaluations), analysis/<cell>_sgeo_lap_vs_abf.json and rerun_<cell>_r2.json (the archive's R2 sustained time-to-accuracy
metric via the unchanged analyzer: Laplacian vs ABF, re-run FR vs ABF), results/r2_campaign/analysis/<cell>.json
(archived FR vs ABF) and pilot/decision.json.  No new statistics: it only tabulates.
"""
import json
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
AN = os.path.join(ROOT, "results", "laplacian_baselines", "analysis")
CASES = ["gateway512", "gateway2048", "R2C3", "R2C4", "R2H80", "R2H350", "R2C1", "R2C2"]
LABEL = {"gateway512": "gateway K512", "gateway2048": "gateway K2048", "R2C3": "LTA 150 K K1024", "R2C4": "LTA 300 K K1024",
         "R2H80": "LTA 80 K K1024", "R2H350": "LTA 350 K K1024", "R2C1": "ITE 300 K K1024", "R2C2": "ITE 150 K K256"}


def _load(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


def g(c):
    return f"{100 * c['G_median']:+.1f} % [{100 * c['G_ci95'][0]:+.1f}, {100 * c['G_ci95'][1]:+.1f}] ({c['wins']}/{c['n']})"


def main():
    dec = _load(os.path.join(ROOT, "results/laplacian_baselines/pilot/decision.json")) or {"cells": {}}
    L = ["# LRS Laplacian selection baseline: summary (NSCC re-runs, three arms at identical seeds)", "",
         "Arms: ABF, ABF + Fisher-Rao (archived configuration), ABF + LRS Laplacian selection (scale frozen per cell by the "
         "fresh-seed pilot rule, docs/LRS_LAPLACIAN_BASELINE.md).  Contrast G = (second - first) / first on paired seeds, "
         "median with seed-bootstrap 95 % CI, (wins / n) = seeds where the second arm is lower.  Negative favours the second arm.", "",
         "## 1. Time-integrated and final errors (eqb_metrics scorers, 200 uniform budget fractions)", "",
         "| cell | scale | I_F: FR vs ABF | I_F: Lap vs ABF | I_F: Lap vs FR | e_F(T): FR vs ABF | e_F(T): Lap vs ABF | e_F(T): Lap vs FR |",
         "|---|---:|---|---|---|---|---|---|"]
    for c in CASES:
        r = _load(os.path.join(AN, f"{c}.json"))
        if r is None:
            L.append(f"| {LABEL[c]} | | (no result) | | | | | |")
            continue
        m = r["metrics"]
        sc = ", ".join(f"{x:g}" for x in r.get("laplacian_scale", []))
        L.append(f"| {LABEL[c]} | {sc} | {g(m['I_F']['fr_vs_abf'])} | {g(m['I_F']['laplacian_vs_abf'])} | {g(m['I_F']['laplacian_vs_fr'])} | "
                 f"{g(m['e_F']['fr_vs_abf'])} | {g(m['e_F']['laplacian_vs_abf'])} | {g(m['e_F']['laplacian_vs_fr'])} |")
    L += ["", "## 2. Mean-force errors", "",
          "| cell | I_F': FR vs ABF | I_F': Lap vs ABF | I_F': Lap vs FR | e_F'(T): FR vs ABF | e_F'(T): Lap vs ABF | e_F'(T): Lap vs FR |",
          "|---|---|---|---|---|---|---|"]
    for c in CASES:
        r = _load(os.path.join(AN, f"{c}.json"))
        if r is None:
            continue
        m = r["metrics"]
        L.append(f"| {LABEL[c]} | {g(m['I_Fp']['fr_vs_abf'])} | {g(m['I_Fp']['laplacian_vs_abf'])} | {g(m['I_Fp']['laplacian_vs_fr'])} | "
                 f"{g(m['e_Fp']['fr_vs_abf'])} | {g(m['e_Fp']['laplacian_vs_abf'])} | {g(m['e_Fp']['laplacian_vs_fr'])} |")
    L += ["", "## 3. Selection activity and genealogy (medians over seeds)", "",
          "| cell | arm | replacements | min windowed ESS/K | max family share | safety stops | established | t_stop |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in CASES:
        r = _load(os.path.join(AN, f"{c}.json"))
        if r is None:
            continue
        for arm in ("fr", "laplacian"):
            a = r["activity"][arm]
            ts = "" if a.get("t_stop_median") is None else f"{a['t_stop_median']:.2f}"
            L.append(f"| {LABEL[c]} | {arm} | {a['repl_median']:.0f} | {a['essw_min_median']:.3f} | {a['maxfam_max_median']:.3f} | "
                     f"{a['n_safety_stops']} | {a['n_established']} | {ts} |")
    L += ["", "## 4. Archived R2 metric: sustained time-to-accuracy speedup S_geo (unchanged analyzer)", "",
          "| cell | FR vs ABF, archive (2026-10-17) | FR vs ABF, NSCC re-run | Laplacian vs ABF, NSCC |", "|---|---|---|---|"]
    for c in ("R2C3", "R2C4", "R2H80", "R2H350", "R2C1", "R2C2"):
        arch = _load(os.path.join(ROOT, "results/r2_campaign/analysis", f"{c}.json"))
        rer = _load(os.path.join(AN, f"rerun_{c}_r2.json"))
        lap = _load(os.path.join(AN, f"{c}_sgeo_lap_vs_abf.json"))

        def s(x):
            if x is None:
                return "(none)"
            p = x["primary"]
            return f"{p['median']:.2f} [{p['ci95'][0]:.2f}, {p['ci95'][1]:.2f}] {p.get('verdict', '')} ({p.get('fr_faster_seeds', '?')}/{x.get('n_seeds', '?')} faster)"
        L.append(f"| {LABEL[c]} | {s(arch)} | {s(rer)} | {s(lap)} |")
    L += ["", "## 5. Pilot decision (fresh seeds; reference-free rule)", ""]
    for c, d in dec.get("cells", {}).items():
        L.append(f"- {c}: scale **{d.get('scale')}** ({d.get('status')}); scales passing safety {d.get('scales_passing_safety')}; "
                 f"FR replacements median {d.get('fr_repl_median')}; log-ratio of Laplacian/FR replacements by scale {d.get('log_ratio_to_fr')}")
    L += ["", "## 6. Re-run vs archive (reproducibility of the ABF / FR arms on NSCC)", "",
          "See `reproducibility.md` in this directory." if os.path.exists(os.path.join(AN, "reproducibility.md")) else "(not run)"]
    L += ["", "## 7. Scale sensitivity on the fresh-seed pilot (secondary; Laplacian vs FR only, 8 seeds per cell)", "",
          "See `pilot_sensitivity.md` in this directory." if os.path.exists(os.path.join(AN, "pilot_sensitivity.md")) else "(not run)"]
    open(os.path.join(AN, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
