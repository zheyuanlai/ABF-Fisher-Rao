"""Confirmation tables: medians, IQRs and paired bootstrap CIs on fresh seeds."""
from __future__ import annotations

import argparse, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.campaign import paired_bootstrap, rel_change

ORDER = ["ti_cold", "ti_warm", "abf", "wfr_shake", "wfr_rot", "wfr_ymap",
         "wfr_yref", "wfr_lmap", "wfr_lref", "wfr_ymh", "wfr_lmh", "wfr_qref",
         "w_only", "fr_only", "w_count", "w_only_y", "wfr_flow", "wfr_flow_y"]
LBL = {"wfr_rot": "RC-WFR, naive rotation lift",
       "wfr_shake": "RC-WFR, min-norm SHAKE lift",
       "wfr_ymap": "RC-WFR + oracle y CDF-map",
       "wfr_yref": "RC-WFR + oracle y refresh",
       "wfr_lmap": "RC-WFR + learned y CDF-map",
       "wfr_lref": "RC-WFR + learned y refresh",
       "ti_cold": "stratified constrained TI (cold)",
       "ti_warm": "stratified constrained TI (warm start, oracle)",
       "abf": "ABF, multiple walkers",
       "wfr_ymh": "RC-WFR + Metropolis y-move, oracle proposal",
       "wfr_lmh": "RC-WFR + Metropolis y-move, LEARNED proposal",
       "wfr_qref": "RC-WFR + full conditional refresh (ceiling)",
       "w_only": "ablation: W only (no Fisher-Rao)",
       "fr_only": "ablation: Fisher-Rao only (no transport)",
       "w_count": "ablation: W + count balancing",
       "w_only_y": "ablation: W only, oracle y-refresh",
       "wfr_flow": "probability-flow W, naive lift",
       "wfr_flow_y": "probability-flow W, oracle y-refresh"}


def load(system, arm, tag, d="results/mol/campaign"):
    p = os.path.join(d, f"{system}_{arm}_{tag}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    n_cfg = int(z["n_cfg"]) if "n_cfg" in z else 1
    ns = z["e_F_final"].shape[0] // n_cfg
    return dict(e_F=z["e_F"].reshape(z["e_F"].shape[0], n_cfg, ns)[:, 0],
                fe=z["fe"], dcond=z["dcond"].reshape(z["dcond"].shape[0], n_cfg, ns)[:, 0],
                dall=(z["dcond_all"].reshape(z["dcond_all"].shape[0], n_cfg, ns, -1)[:, 0]
                      if "dcond_all" in z else None),
                cov=z["cov"].reshape(z["cov"].shape[0], n_cfg, ns)[:, 0],
                ess=z["ess_fix"].reshape(z["ess_fix"].shape[0], n_cfg, ns)[:, 0],
                wall=float(z["wall"]))


def at_budget(d, frac):
    """Index of the save closest to `frac` of the final force-evaluation count."""
    return int(np.argmin(np.abs(d["fe"] - frac * d["fe"][-1])))


def hexane_table(system="HEX", d="results/mol/campaign"):
    """Which fiber mode has to be promoted?"""
    rows = [("wfr_rot", "hex", "none"),
            ("wfr_ymh", "hex_p1", "phi2 (adjacent, strongly coupled)"),
            ("wfr_ymh", "hex_p2", "phi3 (distal, weakly coupled)"),
            ("wfr_ymh", "hex_p12", "both"),
            ("ti_cold", "hex", "-"), ("abf", "hex", "-")]
    print("| arm | promoted | e_F | I_F | D_cond(phi2) | D_cond(phi3) | accept |")
    print("|---|---|---|---|---|---|---|")
    for arm, tag, what in rows:
        p = os.path.join(d, f"{system}_{arm}_{tag}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        e = np.median(z["e_F_final"]); iF = np.median(z["I_F"])
        da = z["dcond_all"][-1] if "dcond_all" in z else None
        d2 = np.median(da[:, 0]) if da is not None and da.shape[-1] > 0 else float("nan")
        d3 = np.median(da[:, 1]) if da is not None and da.shape[-1] > 1 else float("nan")
        ac = np.median(z["lift_cov"][-1])
        print(f"| {LBL.get(arm, arm)} | {what} | {e:.4f} | {iF:.4f} | {d2:.4f} | "
              f"{d3:.4f} | {ac:.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="PEN")
    ap.add_argument("--tag", default="confirm")
    ap.add_argument("--floor", type=float, default=0.0127)
    ap.add_argument("--base", default="ti_cold")
    ap.add_argument("--hexane", action="store_true")
    ap.add_argument("--unit", default="kcal/mol")
    a = ap.parse_args()
    if a.hexane:
        hexane_table(a.system)
        return
    data = {arm: load(a.system, arm, a.tag) for arm in ORDER}
    data = {k: v for k, v in data.items() if v is not None}
    if not data:
        print("no confirmation runs yet"); return
    ref = data[list(data)[0]]
    budgets = [(0.125, "0.5x"), (0.25, "1x"), (1.0, "4x")]
    nseed = data[list(data)[0]]["e_F"].shape[1]
    print(f"### {a.system}: free-energy error at three budgets "
          f"({nseed} fresh seeds, median [IQR], {a.unit})\n")
    print("| arm | " + " | ".join(b[1] for b in budgets) + " | D_cond (4x) | ESS_Fix | wall (s) |")
    print("|---|" + "---|" * (len(budgets) + 3))
    for arm in ORDER:
        if arm not in data:
            continue
        d = data[arm]
        cells = []
        for f, _ in budgets:
            i = at_budget(d, f)
            v = d["e_F"][i]
            cells.append(f"{np.median(v):.4f} [{np.quantile(v,.25):.4f},{np.quantile(v,.75):.4f}]")
        dc = d["dcond"][-1]
        cells.append(f"{np.median(dc):.4f}")
        cells.append(f"{np.median(d['ess'][-1]):.3f}")
        cells.append(f"{d['wall']:.0f}")
        print(f"| {LBL.get(arm, arm)} | " + " | ".join(cells) + " |")
    print(f"\nEstimator floor at this bandwidth: **{a.floor:.4f}** {a.unit}.")
    print(f"Force evaluations at 4x: {ref['fe'][-1]:.3g}.\n")

    print("### Paired relative change vs each comparator (median, 95% bootstrap CI)\n")
    print("| arm | vs | budget | change in e_F | change in I_F |")
    print("|---|---|---|---|---|")
    pairs = [("wfr_rot", "wfr_shake"), ("wfr_ymap", "wfr_rot"), ("wfr_yref", "wfr_rot"),
             ("wfr_ymh", "wfr_rot"), ("wfr_lmh", "wfr_rot"), ("wfr_lmh", "wfr_ymh"),
             ("wfr_lmap", "wfr_rot"), ("wfr_lref", "wfr_rot"),
             ("wfr_qref", "wfr_yref"), ("wfr_lmh", "wfr_qref"),
             ("wfr_lmh", "ti_cold"), ("wfr_lmh", "abf"), ("wfr_lmh", "ti_warm"),
             ("wfr_yref", "ti_cold"), ("wfr_yref", "abf"),
             ("w_only", "wfr_rot"), ("w_count", "wfr_rot")]
    for arm, base in pairs:
        if arm not in data or base not in data:
            continue
        for f, lab in budgets[1:]:
            i, j = at_budget(data[arm], f), at_budget(data[base], f)
            r = rel_change(data[arm]["e_F"][i], data[base]["e_F"][j])
            m, lo, hi = paired_bootstrap(r)
            iF = lambda d, k: np.trapezoid(d["e_F"][:k + 1], d["fe"][:k + 1], axis=0) \
                / max(d["fe"][k] - d["fe"][0], 1.0)
            r2 = rel_change(iF(data[arm], i), iF(data[base], j))
            m2, lo2, hi2 = paired_bootstrap(r2)
            star = "**" if (lo * hi > 0) else ""
            print(f"| {arm} | {base} | {lab} | {star}{100*m:+.1f}%{star} "
                  f"[{100*lo:+.1f}, {100*hi:+.1f}] | {100*m2:+.1f}% [{100*lo2:+.1f}, {100*hi2:+.1f}] |")


if __name__ == "__main__":
    main()
