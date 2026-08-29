"""Fibre-horizon audit: the one hard gate, plus the Stage-3 causal chain.

Frozen protocol: ``docs/FIBRE_HORIZON_AUDIT_PREREGISTRATION.md``.

Separate from ``analyze_info_conversion.py`` on purpose: that script implements
the previous, closed campaign's gate set (which made ancestor ESS a hard gate),
and editing it would rewrite a closed campaign's analysis.  Here the only hard
gate is the empirical estimator risk; everything else is reported.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "configs", "fibre_horizon", "frozen.yaml")


def paired_ratio_ci(fr, ab, n=10000, seed=20260829):
    """Ratio of means with a paired-by-seed bootstrap CI."""
    fr, ab = np.asarray(fr, float), np.asarray(ab, float)
    rng = np.random.default_rng(seed)
    m = fr.size
    vals = np.empty(n)
    for i in range(n):
        k = rng.integers(0, m, m)
        vals[i] = fr[k].mean() / ab[k].mean()
    return (float(fr.mean() / ab.mean()),
            float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def rho_sibling(path):
    """Corr(f(q_t^1), f(q_t^2)) over FR-born pairs, by steps since the pulse."""
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    out = []
    for (arm, step), g in d.groupby(["arm", "step_since"]):
        n = g["n_pairs"].sum()
        if n < 20:
            continue
        sa, sb = g["sum_fa"].sum(), g["sum_fb"].sum()
        sab, sa2, sb2 = g["sum_fafb"].sum(), g["sum_fa2"].sum(), g["sum_fb2"].sum()
        cov = sab / n - (sa / n) * (sb / n)
        va, vb = sa2 / n - (sa / n) ** 2, sb2 / n - (sb / n) ** 2
        if va > 0 and vb > 0:
            out.append(dict(arm=arm, step_since=int(step), n_pairs=int(n),
                            rho=float(cov / np.sqrt(va * vb))))
    return pd.DataFrame(out)


def main():
    cfg = yaml.safe_load(open(CFG))
    root = os.path.join(ROOT, cfg["output_root"], "pilot")
    G = cfg["gates"]
    cells = cfg["cells"]
    dt = float(cfg["simulation"]["dt"])
    ref = json.load(open(os.path.join(ROOT, cfg["output_root"], "reference",
                                      "reference_difficulty.json")))

    tabs = {c: pd.read_csv(os.path.join(root, f"{c}_runs.csv")) for c in cells}
    doses = sorted(x for x in tabs[cells[0]]["p90"].dropna().unique())

    print("=" * 92)
    print("FIBRE-HORIZON AUDIT -- PILOT")
    print("=" * 92)
    print("\nSTAGE 2 (the only hard gate): empirical estimator risk R_FR / R_ABF")
    print("%-4s %-7s %10s %10s %9s %-18s %8s" %
          ("cell", "dose", "R_ABF", "R_FR", "ratio", "95% CI", "gate"))
    print("-" * 76)
    rows = []
    for c in cells:
        t = tabs[c]
        ab = t[t.arm == "abf"].sort_values("seed")["R_s"].to_numpy()
        for d in doses:
            fr = t[t.p90 == d].sort_values("seed")["R_s"].to_numpy()
            r, lo, hi = paired_ratio_ci(fr, ab, int(G["bootstrap_n"]),
                                        int(G["bootstrap_seed"]))
            ok = (r <= G["risk_ratio_max"]) and (hi < 1.0)
            rows.append(dict(cell=c, p90=d, R_abf=ab.mean(), R_fr=fr.mean(),
                             ratio=r, lo95=lo, hi95=hi, gate_risk=bool(ok)))
            print("%-4s %-7g %10.5g %10.5g %9.4f [%7.4f,%7.4f] %8s" %
                  (c, d, ab.mean(), fr.mean(), r, lo, hi,
                   "PASS" if ok else "fail"))

    df = pd.DataFrame(rows)
    passing = [d for d in doses
               if all(df[(df.cell == c) & (df.p90 == d)].gate_risk.iloc[0]
                      for c in cells)]
    selected = min(passing) if passing else None

    print("\nSTAGE 3 (reported whatever the endpoint does): the causal chain")
    print("%-4s %-7s %11s %11s %11s %10s %9s %9s" %
          ("cell", "dose", "KL_post/pre", "TV_fut FR", "TV_fut ABF",
           "events", "ess_anc", "wmax"))
    print("-" * 84)
    chain = []
    for c in cells:
        t = tabs[c]
        tv_ab = t[t.arm == "abf"]["tv_future"].median()
        for d in doses:
            g = t[t.p90 == d]
            klr = float((g["kl_post"] / g["kl_pre"]).median())
            rec = dict(cell=c, p90=d, kl_ratio=klr,
                       tv_future_fr=float(g["tv_future"].median()),
                       tv_future_abf=float(tv_ab),
                       n_events=float(g["n_events"].median()),
                       ess_anc=float(g["ess_anc_final"].median()),
                       wmax=float(g["wmax_family"].median()))
            chain.append(rec)
            print("%-4s %-7g %11.4f %11.4f %11.4f %10.1f %9.4f %9.4f" %
                  (c, d, klr, rec["tv_future_fr"], tv_ab, rec["n_events"],
                   rec["ess_anc"], rec["wmax"]))

    print("\nSTAGE 3c: sibling decorrelation rho(t) over FR-born pairs")
    for c in cells:
        sb = rho_sibling(os.path.join(root, f"{c}_siblings.csv"))
        tf = ref["cells"][c]["tau_max_eval"]
        if sb is None or sb.empty:
            print(f"  {c}: no sibling records")
            continue
        top = sb[sb.arm == f"p{max(doses):g}"].sort_values("step_since")
        if top.empty:
            top = sb.sort_values("step_since")
        pts = [top.iloc[(top.step_since - s).abs().argmin()]
               for s in (0, 50, 200, 600, 1500, top.step_since.max())]
        print(f"  {c} (tau_fib_max = {tf:.2f} = {tf/dt:.0f} steps), arm p{max(doses):g}:")
        print("     " + "  ".join("t=%d:rho=%+.3f" % (p.step_since, p.rho)
                                  for p in pts))

    verdict = dict(campaign="fibre_horizon_audit", stage="pilot",
                   gate0_G_ideal=json.load(open(os.path.join(
                       ROOT, cfg["output_root"], "stage0", "gate0.json")))["median_G_ideal"],
                   risk=df.to_dict("records"), chain=chain,
                   selected_p90=selected)
    kl_moved = all(x["kl_ratio"] < 1.0 for x in chain)
    if selected is not None:
        verdict["outcome"] = "D_or_C_pending_FEC"
    elif kl_moved:
        verdict["outcome"] = "B_representation_without_information"
    else:
        verdict["outcome"] = "A_operator_too_weak"
    print("\n" + "=" * 92)
    print("SELECTED DOSE:", selected if selected is not None else "NONE -- no dose passes")
    print("OUTCOME:", verdict["outcome"])
    print("=" * 92)
    df.to_csv(os.path.join(root, "risk_gate.csv"), index=False)
    pd.DataFrame(chain).to_csv(os.path.join(root, "causal_chain.csv"), index=False)
    with open(os.path.join(root, "fibre_verdict.json"), "w") as fh:
        json.dump(verdict, fh, indent=1, default=float)
    print("wrote", os.path.relpath(root, ROOT))


if __name__ == "__main__":
    main()
