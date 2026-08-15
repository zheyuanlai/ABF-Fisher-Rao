"""C2 — the recalibrated Case IX headline, scored against the high-precision reference.

    python scripts/analyze_caseix_hp.py --dir results/wca_caseix_hp/sham/raw

Recomputes the Case IX contrasts on the **time-integrated** endpoint

    I_F = int_0^T ||F_hat_t - F_ref||_L2 dt

which is the one `-22.83 %` refers to and the one C0 could not recover, because the original
artifacts kept only already-scored scalars. These runs were scored online against
`cache/phase_hp_v3/` and additionally store `pmf_t`, so this endpoint is now reproducible and a
future reference change will never again require re-running the dynamics.

Reports the preregistered pair -- mFR vs ABF, and mFR vs its **own matched sham** -- paired by
seed with bootstrap CIs and win counts, alongside the v1 numbers scored against the cached
reference.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

ARMS = ("abf", "fr_estimated", "fr_oracle", "sham_practical", "sham_oracle")
V1 = {"mFR vs ABF": -22.83, "mFR vs its own sham": -26.38, "sham vs ABF": +2.60}


def load(raw_dir, arm, key):
    out = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, f"*__{arm}__*.npz"))):
        m = re.search(r"seed(\d+)", os.path.basename(f))
        if not m:
            continue
        z = np.load(f, allow_pickle=True)
        if key in z.files:
            out[int(m.group(1))] = np.asarray(z[key], dtype=np.float64)
    return out


def boot(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = np.median(np.asarray(d)[idx], axis=1)
    return float(np.median(d)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/wca_caseix_hp/sham/raw")
    ap.add_argument("--out", default="results/wca_caseix_hp")
    args = ap.parse_args()

    intF = {a: load(args.dir, a, "integrated_l2_f") for a in ARMS}
    intF = {a: v for a, v in intF.items() if v}
    finF = {a: load(args.dir, a, "l2_f") for a in ARMS}
    if not intF:
        raise SystemExit(f"no runs with `integrated_l2_f` under {args.dir}")
    seeds = sorted(set.intersection(*[set(v) for v in intF.values()]))
    print(f"arms {list(intF)}   paired seeds {len(seeds)}\n")

    print(f"{'arm':>16} {'integrated L2(F)':>18} {'final L2(F)':>13}")
    for a in intF:
        fi = [float(finF[a][s]) for s in seeds] if a in finF and finF[a] else [np.nan]
        print(f"{a:>16} {np.median([intF[a][s] for s in seeds]):18.4f} {np.median(fi):13.4f}")

    def contrast(arm, base, label, table):
        x = np.array([float(table[arm][s]) for s in seeds])
        y = np.array([float(table[base][s]) for s in seeds])
        d = 100.0 * (x - y) / y
        med, lo, hi = boot(d)
        wins = int((d < 0).sum())
        v1 = V1.get(label)
        v1s = f"   v1(cached): {v1:+.2f} %" if v1 is not None else ""
        print(f"  {label:<24} {med:+8.2f} %   95% CI [{lo:+.2f}, {hi:+.2f}]   "
              f"{wins}/{len(d)}{v1s}")
        return dict(median_pct=med, ci95=[lo, hi], wins=wins, n=len(d), v1_cached=v1)

    pairs = (("fr_estimated", "abf", "mFR vs ABF"),
             ("fr_estimated", "sham_practical", "mFR vs its own sham"),
             ("sham_practical", "abf", "sham vs ABF"),
             ("fr_oracle", "abf", "mFR-oracle vs ABF"))
    res = {}
    print(f"\n--- TIME-INTEGRATED endpoint (the headline), HP v3 reference ---")
    for a, b, lab in pairs:
        if a in intF and b in intF:
            res[f"integrated::{lab}"] = contrast(a, b, lab, intF)
    if finF and all(finF.get(a) for a, b, _ in pairs if a in finF):
        print(f"\n--- FINAL-TIME endpoint (secondary) ---")
        for a, b, lab in pairs:
            if finF.get(a) and finF.get(b):
                res[f"final::{lab}"] = contrast(a, b, lab, finF)

    # Transport diagnostic on THESE runs. The v1 figure (+1.27 %) came from the v1 seed set,
    # so pairing it against the v2 accuracy number would be a ratio across two run trees.
    # Same statistic and same independent stream as analyze_wca_sham._round_trip_paired.
    rt_tab = {a: load(args.dir, a, "n_round_trips") for a in ARMS}
    rt = None
    if rt_tab.get("fr_estimated") and rt_tab.get("abf"):
        v = np.array([float(rt_tab["fr_estimated"][s]) for s in seeds])
        b = np.array([float(rt_tab["abf"][s]) for s in seeds])
        med, lo, hi = boot(100.0 * (v - b) / b, n=10_000, seed=20260804)
        rt = dict(median_pct=med, ci95=[lo, hi], wins_more_crossings=int((v > b).sum()),
                  n=len(seeds), abf_median=float(np.median(b)),
                  mfr_median=float(np.median(v)),
                  note="transport diagnostic, NOT an accuracy endpoint: more crossings is "
                       "not 'better', so the count is of seeds where mFR crossed MORE")
        print(f"\n--- transport diagnostic (same runs) ---")
        print(f"  round trips, mFR vs ABF  {med:+8.2f} %   95% CI [{lo:+.2f}, {hi:+.2f}]   "
              f"more on {rt['wins_more_crossings']}/{len(seeds)}")

    prim = res.get("integrated::mFR vs ABF")
    att = res.get("integrated::mFR vs its own sham")
    checks = {}
    if prim:
        checks["median <= -10%"] = bool(prim["median_pct"] <= -10.0)
        checks["CI95 upper < 0"] = bool(prim["ci95"][1] < 0.0)
        # preregistered rule is 12/16 = 75 %; expressed as a fraction so a partial-seed
        # run is not scored against a threshold it cannot reach
        need = int(np.ceil(0.75 * prim["n"]))
        checks[f"wins >= {need}/{prim['n']} (75 %)"] = bool(prim["wins"] >= need)
    if att:
        checks["attribution: mFR-vs-sham CI95 upper < 0"] = bool(att["ci95"][1] < 0.0)
    print(f"\n--- preregistered checks ---")
    for k, v in checks.items():
        print(f"  {k:<42} {v}")
    passed = all(checks.values()) if checks else False
    print(f"\n  CASE IX SURVIVES THE CORRECTED REFERENCE: {passed}")
    if prim:
        print(f"  headline: {V1['mFR vs ABF']:+.2f} % (cached)  ->  "
              f"{prim['median_pct']:+.2f} % (HP v3)")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "caseix_hp_verdict.json"), "w") as fh:
        json.dump(dict(n_seeds=len(seeds), arms=list(intF), contrasts=res,
                       round_trips=rt, checks=checks, passed=passed,
                       reference="cache/phase_hp_v3 (unsmoothed, 4 preparations, "
                                 "acquisition == eval grid)",
                       endpoint="time-integrated L2(F), the -22.83% headline endpoint"),
                  fh, indent=2)
    print(f"\nwrote {args.out}/caseix_hp_verdict.json")


if __name__ == "__main__":
    main()
