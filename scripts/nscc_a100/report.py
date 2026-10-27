"""Tables of the NSCC A100 study generated from the JSON records (no number is typed by hand).

    python scripts/nscc_a100/report.py        # writes results/nscc_a100/analysis/{validation,law,prod,confirm}_tables.md
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nscc_common as NC  # noqa: E402

A = os.path.join(NC.OUT_ROOT, "analysis")
V = os.path.join(NC.OUT_ROOT, "validation")


def f(x, fmt=".3g"):
    if x is None:
        return "--"
    if isinstance(x, str):
        return x
    if isinstance(x, float) and math.isinf(x):
        return "censored"
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    return format(x, fmt)


def validation_md():
    L = ["# Stage I gates of the A100 engine (generated)", ""]
    for system in NC.SYSTEMS:
        p = os.path.join(V, f"{system}_validation.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        L += [f"## {system}", "", f"host {r.get('host')}, {r.get('finished_utc')}, {r.get('wall_s', 0):.0f} s", "",
              "| gate | passed | key numbers |", "|---|---|---|"]
        for g in ("RNG1", "RNG2", "FORCE", "FR", "TRAJ", "INV", "DET", "CTR"):
            if g not in r:
                continue
            d = r[g]
            if g == "RNG1":
                k = f"{d['n']} doubles, bitwise {d['bitwise']}"
            elif g == "RNG2":
                k = f"{d['n']} normals, max abs diff {f(d['max_abs'])}, bitwise fraction {f(d['frac_bitwise'])}"
            elif g == "FORCE":
                k = "; ".join(f"G={x['G']}: max rel {f(x['max_rel_F'])}, bins identical {x['bins_identical']}"
                              for x in d["rows"])
            elif g == "FR":
                k = (f"{sum(c['n_opp'] for c in d['cases'])} opportunities (N {', '.join(str(c['N']) for c in d['cases'])}), "
                     f"{d['opportunities_with_events']} with events, mismatches {d['mismatches']}"
                     + (f", max score diff {f(d['max_abs_score_dev'])}" if "max_abs_score_dev" in d else ""))
            elif g == "TRAJ":
                k = "<br>".join(
                    f"{x['method']}{x['N']} {x['n_steps']} steps: counts identical over the whole run "
                    f"{x['C_identical_all']}, max rel dM (first quarter) {f(x.get('max_rel_dM_first_quarter'))}, "
                    f"deaths CPU/GPU {x['deaths_cpu']}/{x['deaths_gpu']}, "
                    f"final position diff {f(x.get('max_abs_dX_final', x.get('max_abs_dq_final')))}"
                    for x in d["rows"])
            elif g == "INV":
                k = f"{len(d['configs'])} launch configurations, identical {d['identical']}"
            else:
                k = d.get("note", "")
            L.append(f"| {g} | {'PASS' if d.get('passed') else 'FAIL'} | {k} |")
        L.append("")
    return "\n".join(L)


def law_md():
    p = os.path.join(A, "law_check.json")
    if not os.path.exists(p):
        return ""
    r = json.load(open(p))
    L = ["# V4 law check: A100 vs CPU on the validation seeds (generated)", "",
         "Ratio of medians GPU / CPU (paired bootstrap 95 % CI); 'identical' = fraction of seeds whose value is "
         "bitwise equal on both engines.", "",
         "| system | arm | pairs | final e_F | Ibar_F | u_mid | verdict |", "|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        cells = []
        for k in ("final_e_F", "Ibar_F", "u_mid"):
            s = x[k]
            cells.append(f"{f(s['point'])} [{f(s['ci95'][0])}, {f(s['ci95'][1])}] (identical {f(x[k + '_identical_frac'], '.2f')})")
        L.append(f"| {x['system']} | {x['method']}{x['N']} | {x['n_pairs']} | " + " | ".join(cells) +
                 f" | {'PASS' if x['passed'] else 'FAIL'} |")
    L += ["", f"Overall: {'PASS' if r['passed'] else 'FAIL'}"]
    return "\n".join(L)


def prod_md():
    p = os.path.join(A, "prod_summary.json")
    if not os.path.exists(p):
        return ""
    S = json.load(open(p))["systems"]
    L = ["# Exploration (production seeds): per-N medians and best allocation (generated)", ""]
    for system, s in S.items():
        L += [f"## {system} ({s['n_runs']} runs)", "",
              "| arm | backend | n | median W_mid (s) | censored | median W_mid_e2e (s) | median final e_F | "
              "median completion (s) | median start overhead (s) |", "|---|---|---|---|---|---|---|---|---|"]
        for k, v in sorted(s["per"].items(), key=lambda kv: (kv[1]["method"], kv[1]["N"])):
            L.append(f"| {k} | {v['backend']} | {v['n']} | {f(v['median_W_mid'])} | {v['censored_mid']} | "
                     f"{f(v['median_W_mid_e2e'])} | {f(v['median_final_e_F'])} | {f(v['median_completion_wall_s'])} | "
                     f"{f(v['median_start_overhead_s'])} |")
        for key in ("W_mid", "W_mid_e2e", "W_loose", "W_strict"):
            b = s["best"][key]
            if not (b and b.get("abf") and b.get("fr")):
                continue
            rr = b.get("R_wall") or {}
            L.append(f"\n* best allocation {key} (exploratory): ABF N = {b['abf']['best_N']} ({f(b['abf']['best_value'])} s), "
                     f"FR N = {b['fr']['best_N']} ({f(b['fr']['best_value'])} s); R = {f(rr.get('point'))} "
                     f"[{f((rr.get('ci95') or [None, None])[0])}, {f((rr.get('ci95') or [None, None])[1])}]")
        L.append("")
    return "\n".join(L)


def confirm_md():
    p = os.path.join(A, "confirm_summary.json")
    if not os.path.exists(p):
        return ""
    S = json.load(open(p))["systems"]
    L = ["# Confirmation (fresh seeds, selection frozen) (generated)", "",
         "| system | ABF arm | FR arm | n ABF / FR | median W_mid ABF / FR (s) | R_W [95 % CI] | R_e2e [95 % CI] | "
         "censored ABF / FR | Holm alpha | decision |", "|---|---|---|---|---|---|---|---|---|---|"]
    for system, o in S.items():
        s = o["selection"]
        rw, re = o["W_mid"]["R"], o["W_mid_e2e"]["R"]
        L.append(f"| {system} | ABF {s['abf']['N']} {s['abf']['backend']} | FR {s['fr']['N']} {s['fr']['backend']} | "
                 f"{o['n']['abf']} / {o['n']['fr']} | {f(o['W_mid']['median_abf'])} / {f(o['W_mid']['median_fr'])} | "
                 f"{f(rw['point'])} [{f(rw['ci95'][0])}, {f(rw['ci95'][1])}] | {f(re['point'])} [{f(re['ci95'][0])}, "
                 f"{f(re['ci95'][1])}] | {rw['censored_b']} / {rw['censored_a']} | {f(o.get('holm_alpha'))} | "
                 f"**{o['decision']}** |")
    L += ["", "R = median FR / median ABF; < 1 favours FR. Secondary thresholds:", "",
          "| system | R loose (W) | R strict (W) |", "|---|---|---|"]
    for system, o in S.items():
        L.append(f"| {system} | {f(o['W_loose']['R']['point'])} [{f(o['W_loose']['R']['ci95'][0])}, "
                 f"{f(o['W_loose']['R']['ci95'][1])}] | {f(o['W_strict']['R']['point'])} "
                 f"[{f(o['W_strict']['R']['ci95'][0])}, {f(o['W_strict']['R']['ci95'][1])}] |")
    return "\n".join(L)


def main():
    os.makedirs(A, exist_ok=True)
    for name, fn in (("validation", validation_md), ("law", law_md), ("prod", prod_md), ("confirm", confirm_md)):
        txt = fn()
        if txt:
            with open(os.path.join(A, f"{name}_tables.md"), "w") as fh:
                fh.write(txt + "\n")
            print("wrote", name)


if __name__ == "__main__":
    main()
