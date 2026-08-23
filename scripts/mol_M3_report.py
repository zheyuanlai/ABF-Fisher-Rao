"""M3: do the three pentane arms separate once the numerical floor is lowered?

The old comparison bottomed out near 0.020 kcal/mol for every constrained arm,
and section 20 of `MOLECULAR_RESULTS.md` showed that number was the estimator's
kernel plus the constrained integrator's time step rather than anything any arm
was doing.  This reruns the three arms that matter at `h` = 1e-3, `b_mf` = 0.02
and a 257-node grid, where the same accounting puts the floor near 0.005.

Three arms, no more: warm stratified TI is the practical ceiling, cold
stratified TI is the same estimator with no transport at all, and persistent
RC-WFR with the learned Metropolis lift is the method.  If RC-WFR carries a
transport bias of its own it has ~0.015 of newly exposed room to show up in.
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.campaign import paired_bootstrap, rel_change

CAM = "results/mol/campaign"
LBL = {"ti_warm": "stratified TI, warm (ceiling)", "ti_cold": "stratified TI, cold",
       "wfr_lmh": "RC-WFR + Metropolis y-move (learned)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="M3")
    ap.add_argument("--system", default="PEN")
    ap.add_argument("--arms", default="ti_warm,wfr_lmh,ti_cold")
    ap.add_argument("--floor", type=float, default=0.0020,
                    help="analytic smoothing floor at this convention")
    ap.add_argument("--out", default="results/mol/M3_report.json")
    a = ap.parse_args()
    arms = a.arms.split(",")
    D = {}
    for arm in arms:
        p = os.path.join(CAM, f"{a.system}_{arm}_{a.tag}.npz")
        if os.path.exists(p):
            D[arm] = np.load(p)
    if not D:
        print("no M3 archives yet"); return
    fe = D[arms[0]]["fe"]
    # budgets to report: quarter, half and the full run
    idx = [len(fe) // 4 - 1, len(fe) // 2 - 1, len(fe) - 1]

    print(f"# M3: {a.system} at h=1e-3, b_mf=0.02, n=257 "
          f"(smoothing floor {a.floor:.4f})\n")
    print("| arm | " + " | ".join(f"{fe[i]:.2e} fe" for i in idx) + " |")
    print("|---|" + "---|" * len(idx))
    med = {}
    for arm in arms:
        if arm not in D:
            continue
        e = D[arm]["e_F"]                      # (n_saves, rows)
        med[arm] = np.median(e, axis=-1)
        print(f"| {LBL.get(arm, arm)} | "
              + " | ".join(f"{med[arm][i]:.4f}" for i in idx) + " |")

    print("\n## Paired change (negative = first arm better), 95% bootstrap CI\n")
    print("| contrast | " + " | ".join(f"{fe[i]:.2e} fe" for i in idx) + " |")
    print("|---|" + "---|" * len(idx))
    out = {"fe": fe.tolist(), "median": {k: v.tolist() for k, v in med.items()},
           "contrasts": {}}
    pairs = [("wfr_lmh", "ti_cold"), ("wfr_lmh", "ti_warm"), ("ti_warm", "ti_cold")]
    for x, y in pairs:
        if x not in D or y not in D:
            continue
        cells, rec = [], []
        for i in idx:
            m, lo, hi = paired_bootstrap(rel_change(D[x]["e_F"][i], D[y]["e_F"][i]))
            cells.append(f"{100*m:+.1f}% [{100*lo:+.1f}, {100*hi:+.1f}]")
            rec.append([float(m), float(lo), float(hi)])
        print(f"| {x} vs {y} | " + " | ".join(cells) + " |")
        out["contrasts"][f"{x}_vs_{y}"] = rec

    print("\n## Late-time rate  d log e_F / d log fe  (last half of the run)\n")
    print("| arm | slope |")
    print("|---|---|")
    for arm in arms:
        if arm not in D:
            continue
        h0 = len(fe) // 2
        m = med[arm][h0:] > 0
        sl = np.polyfit(np.log(fe[h0:][m]), np.log(med[arm][h0:][m]), 1)[0]
        print(f"| {LBL.get(arm, arm)} | {sl:+.3f} |")
        out.setdefault("slope", {})[arm] = float(sl)

    for arm in arms:
        if arm in D and "dcond" in D[arm]:
            out.setdefault("dcond_final", {})[arm] = float(
                np.median(D[arm]["dcond"][-1]))
    if "dcond_final" in out:
        print("\nfinal conditional error (nats): "
              + "  ".join(f"{k}={v:.4f}" for k, v in out["dcond_final"].items()))
    json.dump(out, open(a.out, "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
