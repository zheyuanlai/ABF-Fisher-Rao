"""Markdown summary of analysed blocks (analysis/<block>.json), generated -- no hand transcription.

    python scripts/convergence_fr/block_tables.py E1 [E2 ...]
"""
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
AN = os.path.join(ROOT, "results", "convergence_fr_campaign", "analysis")


def pc(x):
    return f"{100 * x['median_rel']:+.1f} % [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}] ({x['wins_fr']}/{x['n']})"


for b in sys.argv[1:]:
    R = json.load(open(os.path.join(AN, f"{b}.json")))
    p = R["primary"]
    s = R["secondary"]
    print(f"\n#### {b}: K {R['K']}, T {R['T']:g} t.u., {R['T_K']:g} K, {R['bins']} bins, FR rate {R['rate']:g} from "
          f"{R['t_fr']:g} t.u." + (f", stopped at {R['t_stop']:g} t.u." if R.get("t_stop") else ", continuous")
          + f"; {R['n_seeds']} paired seeds {min(R['cfg_seeds'])}–{max(R['cfg_seeds'])}\n")
    print(f"* integrity: bitwise paired before the first FR opportunity {sum(R['paired_bitwise_before_fr_start'])}/{R['n_seeds']}; "
          f"force evaluations per run {R['force_evals_per_run']:,} (equal in both arms); arms differ only in "
          f"{R['config_keys_differing_between_arms'] or 'the method'}"
          + (f"; FR opportunities per run {R['fr_opportunities_per_run']}, FR events after the stop {R['fr_events_after_stop']}"
             if 'fr_events_after_stop' in R else ""))
    lad = R["ladder"]
    print(f"* ladder (kJ/mol): {', '.join(f'{e:.3f}' for e in lad['eps_kJmol'])}"
          + (f" (= {lad['multiples_of_median_abf_eF_T']} × median ABF e_F(T) {lad['median_abf_eF_T']:.4f})" if 'median_abf_eF_T' in lad else
             f" (= {lad['eps_kT']} k_BT)"))
    print(f"* **PRIMARY S_geo {p['median']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}]** (97.5 %: [{p['ci975'][0]:.3f}, "
          f"{p['ci975'][1]:.3f}]); FR faster in {p['fr_faster_seeds']}/{R['n_seeds']} seeds → **{p['verdict']}**\n")
    print("| statistic | median per-seed FR/ABF − 1 [95 % CI] (FR better) | ABF median | FR median |")
    print("|---|---|---|---|")
    names = [("Ibar_F_post_ramp", "post-ramp Ī_F (companion)"), ("Ibar_F_full", "full-horizon Ī_F"),
             ("Ibar_F_post_ramp_to_10", "post-ramp Ī_F over [t_FR, 10]"),
             ("Ibar_F_early_post_ramp_to_2tstop", "post-ramp Ī_F over [t_FR, 2 t_stop] (H1)"),
             ("eF_at_0.25T", "e_F(T/4)"), ("eF_at_0.5T", "e_F(T/2)"), ("eF_at_0.75T", "e_F(3T/4)"),
             ("Ibar_F_late_window", "late-window Ī_F [0.8T, T]"), ("eF_T", "e_F(T) (H2)"), ("eG_T", "e_F′(T)"),
             ("post_window_estimate_eF", "estimate from deposits after " + ("t_stop (H4)" if R.get("t_stop") else "t_FR"))]
    for k, lab in names:
        if k in s:
            x = s[k]
            print(f"| {lab} | {pc(x)} | {x['abf_median']:.4f} | {x['fr_median']:.4f} |")
    print("\n| ε (kJ/mol) | sustained S [95 % CI] | reached ABF / FR | FR faster / ABF faster | first-hit S |")
    print("|---|---|---|---|---|")
    for st, fh in zip(R["time_to_accuracy"]["sustained"], R["time_to_accuracy"]["first"]):
        print(f"| {st['eps_kJmol']:.3f} | {st['S_median']:.2f} [{st['S_ci95'][0]:.2f}, {st['S_ci95'][1]:.2f}] | "
              f"{st['reached_abf']} / {st['reached_fr']} | {st['fr_faster']} / {st['abf_faster']} | {fh['S_median']:.2f} |")
    c = R["crossing_eF"]
    es = R["establishment"]
    g = R["genealogy_fr"]
    print(f"\n* crossing (FR permanently worse after): {c['n_seeds_crossing']}/{c['n_seeds']} seeds; median curve "
          f"{'none' if c['median_curve_crossing'] == float('inf') else round(c['median_curve_crossing'], 1)}"
          + (f", bootstrap 2.5/50/97.5 % {'/'.join(f'{q:.1f}' for q in c['boot_quantiles'])} t.u. (fraction crossing {c['boot_fraction_crossing']:.2f})" if c['boot_quantiles'] else ""))
    pd = es["paired_difference"]
    print(f"* window establishment: ABF {es['abf_median']:.2f}, FR {es['fr_median']:.2f} t.u."
          + (f"; paired {pd[0]:+.2f} [{pd[1]:+.2f}, {pd[2]:+.2f}]" if pd else ""))
    print(f"* FR genealogy: min windowed ESS/K {g['min_essw_median']:.3f}, max lineage share {g['max_lineage_median']:.3f}, "
          f"replacements {g['replacements_median']:.0f} (median over seeds); true translocations ABF "
          f"{R['translocations']['abf_median']:.0f} vs FR {R['translocations']['fr_median']:.0f}")
    pl = s["pooled_post_window_rms_vs_ref"]
    pf = s["pooled_full_rms_vs_ref"]
    print(f"* pooled over seeds vs MC: whole run ABF {pf['abf']:.4f} / FR {pf['fr']:.4f}; post-window deposits ABF "
          f"{pl['abf']:.4f} / FR {pl['fr']:.4f}; FR vs ABF post-window {s['pooled_post_window_rms_fr_vs_abf']:.4f}")
    w = R["wall_seconds_median"]
    print(f"* wall-clock per run (single thread, production phase): ABF {w['abf']:.0f} s, FR {w['fr']:.0f} s "
          f"({100 * (w['fr'] / w['abf'] - 1):+.1f} %)")
