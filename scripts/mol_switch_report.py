"""Did switching transport off remove RC-WFR's residual error floor?

Reads the switch runs, which each carry two estimators -- `e_F` keeps every
deposit, `e_F_prod` uses only post-switch samples -- and puts them next to
persistent RC-WFR and the unbiased baselines at matched force evaluations.

The question is specifically about SLOPE, not just level: an arm that has
removed its bias should converge at the statistical rate again.
"""
from __future__ import annotations

import argparse, glob, os, re, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.campaign import paired_bootstrap, rel_change


def curve(path, key="e_F"):
    d = np.load(path)
    n = int(d["n_cfg"]) if "n_cfg" in d else 1
    ns = d[key].shape[1] // n
    return d["fe"], d[key].reshape(d[key].shape[0], n, ns)[:, 0], d


def at(fe, e, target):
    i = int(np.argmin(np.abs(fe - target)))
    return i, e[i]


def slope(fe, e, span=4.0):
    m = fe >= fe[-1] / span
    return float(np.polyfit(np.log(fe[m]), np.log(np.median(e[m], -1)), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="PEN")
    ap.add_argument("--dir", default="results/mol/campaign")
    ap.add_argument("--arm", default="wfr_lmh")
    ap.add_argument("--unit", default="kcal/mol")
    ap.add_argument("--budgets", default="1.07e8,4.3e8")
    a = ap.parse_args()
    budgets = [float(x) for x in a.budgets.split(",")]
    ref_runs = {}
    for nm, f in [("persistent RC-WFR", f"{a.system}_{a.arm}_long.npz"),
                  ("ABF", f"{a.system}_abf_long.npz"),
                  ("stratified TI, cold", f"{a.system}_ti_cold_long.npz"),
                  ("RC-WFR, naive lift", f"{a.system}_wfr_rot_long.npz"),
                  ("OPES / ABP", f"{a.system}_opes_long.npz")]:
        p = os.path.join(a.dir, f)
        if os.path.exists(p):
            ref_runs[nm] = curve(p)
    sw = {}
    for p in sorted(glob.glob(os.path.join(a.dir, f"{a.system}_{a.arm}_sw*.npz"))):
        m = re.search(r"_sw(snaponly|snap)?(\d+)\.npz$", p)
        if m is None:
            continue
        kind = {None: "frozen in place", "snap": "snapped + frozen proposal",
                "snaponly": "snapped only"}[m.group(1)]
        sw[(int(m.group(2)), kind)] = p

    hdr = " | ".join(f"{b:.2g}" for b in budgets)
    print(f"| arm | estimator | {hdr} | late slope |")
    print("|---|---|" + "---|" * (len(budgets) + 1))
    for nm, (fe, e, _d) in ref_runs.items():
        cells = [f"{np.median(at(fe, e, b)[1]):.4f}" for b in budgets]
        print(f"| {nm} | all deposits | " + " | ".join(cells)
              + f" | {slope(fe, e):+.3f} |")
    for (ts, kind), p in sorted(sw.items()):
        fe, e, d = curve(p, "e_F")
        _, ep, _ = curve(p, "e_F_prod")
        fsw = float(d["fe_switch"]) if "fe_switch" in d else 0.0
        for lab, v in (("all deposits", e), ("**post-switch only**", ep)):
            cells = [f"{np.median(at(fe, v, b)[1]):.4f}" for b in budgets]
            print(f"| WFR->TI, {kind}, @{ts:.0g} steps (fe {fsw:.2g}) | {lab} | "
                  + " | ".join(cells) + f" | {slope(fe, v):+.3f} |")
    print()
    if "persistent RC-WFR" in ref_runs and sw:
        print("### Paired change vs persistent RC-WFR (median, 95% bootstrap CI)\n")
        print("| switch | estimator | budget | change in e_F |")
        print("|---|---|---|---|")
        fe0, e0, _ = ref_runs["persistent RC-WFR"]
        for (ts, kind), p in sorted(sw.items()):
            fe, e, _ = curve(p, "e_F")
            _, ep, _ = curve(p, "e_F_prod")
            for lab, v in (("all", e), ("**post-switch**", ep)):
                for b in budgets:
                    i, vi = at(fe, v, b); j, vj = at(fe0, e0, b)
                    m, lo, hi = paired_bootstrap(rel_change(vi, vj))
                    star = "**" if lo * hi > 0 else ""
                    print(f"| {kind} @{ts:.0g} | {lab} | {b:.2g} | "
                          f"{star}{100*m:+.1f}%{star} [{100*lo:+.1f}, {100*hi:+.1f}] |")


if __name__ == "__main__":
    main()
