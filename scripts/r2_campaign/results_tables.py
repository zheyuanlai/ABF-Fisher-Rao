"""Markdown tables for docs/r2_campaign/RESULTS.md from the frozen analysis outputs (results/r2_campaign/analysis/*.json
and verdicts.json).  Formatting only: every number is read from those files.

    python scripts/r2_campaign/results_tables.py > results/r2_campaign/analysis/tables.md
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
AN = os.path.join(ROOT, "results", "r2_campaign", "analysis")
sys.path.insert(0, os.path.join(ROOT, "scripts", "r2_campaign"))
from verdicts import CELLS  # noqa: E402


def pct(x):
    return f"{100 * x['median_rel']:+.0f} % [{100 * x['ci95'][0]:+.0f}, {100 * x['ci95'][1]:+.0f}] ({x['wins_fr']}/{x['n']})"


def main():
    V = json.load(open(os.path.join(AN, "verdicts.json")))
    D = {b: json.load(open(os.path.join(AN, f"{b}.json"))) for b, _, _ in CELLS if os.path.exists(os.path.join(AN, f"{b}.json"))}
    print("### Primary outcome\n")
    print("| cell | system | role | test | R2 stop, t.u.: median [IQR] (range) | S_geo [95 % CI] | 90 % CI | Bonferroni-8 CI | FR faster | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for b, role, test in CELLS:
        if b not in D:
            print(f"| {b} | {role} | | {test} | not run | | | | | |")
            continue
        v, d = V[b], D[b]
        s = d["fr_schedule"]
        print(f"| {b} | {role.split(' (')[0]} | {role.split('(')[1].rstrip(')')} | {test} | {s['stop_t_median']:.2f} [{s['stop_t_iqr'][0]:.2f}, {s['stop_t_iqr'][1]:.2f}] "
              f"({s['stop_t_min']:.2f}–{s['stop_t_max']:.2f}) | **{v['median']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]** | "
              f"[{v['ci90'][0]:.2f}, {v['ci90'][1]:.2f}] | [{v['ci_bonf'][0]:.2f}, {v['ci_bonf'][1]:.2f}] | {v['fr_faster']}/16 | **{v['verdict']}** |")
    print("\n### Secondary outcomes (median per-seed FR/ABF − 1 [95 % CI], FR better; wins/16)\n")
    print("| cell | post-ramp Ī_F | early Ī_F [t_FR, 0.4 T] | late-window Ī_F [0.8 T, T] | e_F(T/2) | e_F(T) | e_F′(T) | post-stop-only estimate | endpoint flag |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b, role, test in CELLS:
        if b not in D:
            continue
        sec = D[b]["secondary"]
        print(f"| {b} | {pct(sec['Ibar_F_post_ramp'])} | {pct(sec['Ibar_F_early'])} | {pct(sec['Ibar_F_late_window'])} | {pct(sec['eF_at_0.5T'])} | "
              f"{pct(sec['eF_T'])} | {pct(sec['eG_T'])} | {pct(sec['post_stop_estimate_eF'])} | {V[b]['endpoint']} |")
    print("\n### The online schedule actually realised\n")
    print("| cell | never stopped | reasons | t_est median (range) | FR replacements per run | events after stop | min windowed ESS/K | max lineage | window establishment ABF → FR (t.u.) | paired diff. | translocations ABF / FR | wall s ABF / FR |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b, role, test in CELLS:
        if b not in D:
            continue
        d = D[b]
        s, g, e = d["fr_schedule"], d["genealogy_fr"], d["establishment"]
        t1 = np.array(s.get("t1", []), float)
        pd_ = e.get("paired_difference")
        print(f"| {b} | {s['never_stopped']} | {', '.join(sorted(set(s.get('reason', []))))} | {np.median(t1):.2f} ({t1.min():.2f}–{t1.max():.2f}) | "
              f"{g['replacements_median']:.0f} | {s['events_after_stop_total']} | {g['min_essw_median']:.3f} | {g['max_lineage_median']:.3f} | "
              f"{e['abf_median']:.2f} → {e['fr_median']:.2f} | {('%.2f [%.2f, %.2f]' % tuple(pd_)) if pd_ else '–'} | "
              f"{d['translocations']['abf_median']:.0f} / {d['translocations']['fr_median']:.0f} | {d['wall_seconds_median']['abf']:.0f} / {d['wall_seconds_median']['fr']:.0f} |")
    print("\n### Per-rung sustained speedups\n")
    for b, role, test in CELLS:
        if b not in D:
            continue
        d = D[b]
        rows = d["time_to_accuracy"]["sustained"]
        print(f"* **{b}** ({d['ladder_rule']} ladder): " + "; ".join(
            f"ε {r['eps_kJmol']:.3f}: S {r['S_median']:.2f} [{r['S_ci95'][0]:.2f}, {r['S_ci95'][1]:.2f}] reach A{r['reached_abf']}/B{r['reached_fr']}" for r in rows))
    print("\n### Integrity\n")
    for b, role, test in CELLS:
        if b not in D:
            continue
        d = D[b]
        print(f"* {b}: paired bitwise before FR start {sum(d['paired_bitwise_before_fr_start'])}/16; force evaluations per run "
              f"{d['force_evals_per_run']:,} (equal in both arms); arms differ only in {d['allowed_config_diff']}; seeds {d['cfg_seeds'][0]}–{d['cfg_seeds'][-1]}.")
    q = V["_questions"]
    print("\n### Rule-level questions\n")
    for k, val in q.items():
        print(f"* {k}: **{val}**")


if __name__ == "__main__":
    main()
