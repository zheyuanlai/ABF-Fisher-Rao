"""Markdown tables for CONVERGENCE_ATLAS.md, generated from atlas_summary.json (no hand transcription)."""
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
S = json.load(open(os.path.join(ROOT, "results", "convergence_fr_campaign", "atlas", "atlas_summary.json")))["systems"]
KEYS = sys.argv[1:] or ["LTA150", "LTA300", "ITE300", "ITE150", "ITE150c", "ITE150_T50", "ITE150c_T50"]


def pc(x):
    return f"{100 * x['median_rel']:+.1f} % [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}] ({x['wins_fr']}/{x['n']})"


print("| system | K | T | bins | rate | Ī_F full | Ī_F post-ramp | e_F(T) | e_F′(T) |")
print("|---|---|---|---|---|---|---|---|---|")
for k in KEYS:
    r = S[k]
    e = r["endpoints"]
    print(f"| {r['label']} | {r['K']} | {r['T']:g} | {r['bins']} | {r['rate']:g} | {pc(e['Ibar_F_full'])} | "
          f"{pc(e['Ibar_F_post_ramp'])} | {pc(e['eF_T'])} | {pc(e['eG_T'])} |")
print()
print("Absolute medians (ABF → FR), kJ/mol: ")
for k in KEYS:
    e = S[k]["endpoints"]
    print(f"* {S[k]['label']}: Ī_F {e['Ibar_F_full']['abf_median']:.3f} → {e['Ibar_F_full']['fr_median']:.3f}; "
          f"post-ramp {e['Ibar_F_post_ramp']['abf_median']:.3f} → {e['Ibar_F_post_ramp']['fr_median']:.3f}; "
          f"e_F(T) {e['eF_T']['abf_median']:.4f} → {e['eF_T']['fr_median']:.4f}")
print()
for kind in ("sustained", "first"):
    print(f"\n**Time to accuracy ({kind})** — S median [95 % CI]; reached A/B of n; FR faster / ABF faster; "
          f"median τ ABF → FR (t.u.; '–' if ≤ half reach)\n")
    print("| system | ε (kJ/mol) | ε/k_BT | S | reached ABF / FR | FR faster / ABF faster | τ ABF → FR | note |")
    print("|---|---|---|---|---|---|---|---|")
    for k in KEYS:
        r = S[k]
        for ek, st in r["time_to_accuracy"][kind].items():
            ta = "–" if st["tau_abf_median"] is None else f"{st['tau_abf_median']:.1f}"
            tb = "–" if st["tau_fr_median"] is None else f"{st['tau_fr_median']:.1f}"
            print(f"| {r['label']} | {st['eps_kJmol']:.3f} | {ek[:-2]} | {st['S_median']:.2f} [{st['S_ci95'][0]:.2f}, "
                  f"{st['S_ci95'][1]:.2f}] | {st['reached_abf']} / {st['reached_fr']} of {st['n_seeds']} | "
                  f"{st['fr_faster']} / {st['abf_faster']} | {ta} → {tb} | {st['bound']} |")
print("\n**Crossing (FR permanently worse than ABF after this save) and window establishment**\n")
print("| system | seeds crossing | per-seed median | median-curve crossing | bootstrap: fraction crossing; 2.5/50/97.5 % | t_est ABF | t_est FR | paired Δ t_est |")
print("|---|---|---|---|---|---|---|---|")
for k in KEYS:
    r = S[k]
    c = r["crossing_eF"]
    es = r["establishment"]
    q = c["boot_quantiles"]
    pd = es["paired_difference"]
    print(f"| {r['label']} | {c['n_seeds_crossing']}/{c['n_seeds']} | {'–' if c['per_seed_median'] is None else round(c['per_seed_median'], 1)} | "
          f"{'none' if c['median_curve_crossing'] == float('inf') else round(c['median_curve_crossing'], 1)} | "
          f"{c['boot_fraction_crossing']:.2f}; {'–' if q is None else '/'.join(f'{x:.1f}' for x in q)} | "
          f"{es['abf_median']:.1f} | {es['fr_median']:.1f} | {'–' if pd is None else f'{pd[0]:+.1f} [{pd[1]:+.1f}, {pd[2]:+.1f}]'} |")
