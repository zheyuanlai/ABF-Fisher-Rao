"""Q1 — is the Fisher-Rao direction better than the directed selection ABF already had?

    python scripts/analyze_wca_five_arm.py --dir results/wca_five_arm/confirm/raw

v1 measured mFR against ABF and against matched **random** turnover. It never measured it
against a **directed** alternative, and Chapter 6 of Lelievre-Rousset-Stoltz applies selection
to this very WCA dimer. This is that missing comparison.

Decision rules are §4.3 of the preregistration, quoted verbatim in the code so they cannot drift:

  Primary     median Delta <= -10 %, 95 % CI upper < -5 %, wins >= 12/16,
              ESS_anc/N >= 0.30, w_max <= 0.05
  Attribution 95 % CI of the DIRECT mFR-vs-sham contrast below zero
  Novelty Q1  median (I_F(mFR) - I_F(prior)) / I_F(prior) <= -5 % with 95 % CI < 0,
              against BOTH prior-art arms. Otherwise a **tie**, tested by TOST.

The TOST clause matters. §4.3 forbids concluding equivalence from "no CI excluded zero", which
is an absence-of-evidence argument; equivalence has to be demonstrated positively by showing the
90 % CI lies inside the +-5 % margin. Both directions are reported so a tie cannot be quietly
read as a win.

Turnover is re-checked **on the confirmatory runs themselves**, not just on the calibration
seeds where `c` was chosen -- a match on held-out seeds does not guarantee a match here, and an
arm that selects harder could win for that reason alone.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

ARMS = ("abf", "fr_estimated", "sham_practical", "book_laplacian", "count_balancing")
PRIOR = ("book_laplacian", "count_balancing")
N_REPLICAS = 1024
EQUIV_MARGIN = 5.0          # per cent, the same margin §4.3 uses for superiority


def load(raw_dir, arm, key):
    out = {}
    # the file stem is `five_<arm>`; match on that so `sham_practical` cannot also pick up
    # `fr_estimated` runs through a substring collision
    for f in sorted(glob.glob(os.path.join(raw_dir, f"*__five_{arm}__*.npz"))):
        m = re.search(r"seed(\d+)", os.path.basename(f))
        if not m:
            continue
        z = np.load(f, allow_pickle=True)
        if key in z.files:
            out[int(m.group(1))] = np.asarray(z[key], dtype=np.float64)
    return out


def boot_median(d, n=20000, seed=0, lo_pct=2.5, hi_pct=97.5):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=np.float64)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = np.median(d[idx], axis=1)
    return float(np.median(d)), float(np.percentile(m, lo_pct)), float(np.percentile(m, hi_pct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/wca_five_arm/confirm/raw")
    ap.add_argument("--out", default="results/wca_five_arm")
    args = ap.parse_args()

    intF = {a: load(args.dir, a, "integrated_l2_f") for a in ARMS}
    missing = [a for a, v in intF.items() if not v]
    if missing:
        raise SystemExit(f"no runs for arms {missing} under {args.dir}")
    seeds = sorted(set.intersection(*[set(v) for v in intF.values()]))
    if not seeds:
        raise SystemExit("no seed is present in all five arms")
    repl = {a: load(args.dir, a, "total_replacement_events") for a in ARMS}
    # `min_ancestor_ess_window`, NOT `min_ancestor_ess`. The latter traces ancestry from t=0, so
    # it decays monotonically with run length for any birth-death process and hits 0.18 here on
    # an arm whose genealogy is demonstrably healthy -- the matched-turnover RANDOM sham lands at
    # 0.19 too. The gateway confirmatory that set the 0.30 floor measured it over a 4000-step
    # window (`ess_window_steps: 4000`), which is what the gate means.
    ess = {a: load(args.dir, a, "min_ancestor_ess_window") for a in ARMS}
    ess_run = {a: load(args.dir, a, "min_ancestor_ess") for a in ARMS}
    wmax = {a: load(args.dir, a, "max_ancestor_frac_over_time") for a in ARMS}
    print(f"five arms, {len(seeds)} fully paired seeds ({seeds[0]}-{seeds[-1]})\n")

    # ---- turnover match, re-checked where it actually matters ----
    tgt = float(np.median([repl["fr_estimated"][s] for s in seeds]))
    print(f"{'arm':>16} {'integrated L2(F)':>18} {'replacements':>13} {'x mFR':>7}")
    for a in ARMS:
        r = float(np.median([repl[a][s] for s in seeds]))
        x = f"{r / tgt:6.2f}" if a != "abf" else "     -"
        print(f"{a:>16} {np.median([intF[a][s] for s in seeds]):18.4f} {r:13.0f} {x:>7}")
    turnover_ok = {m: bool(0.5 <= float(np.median([repl[m][s] for s in seeds])) / tgt <= 2.0)
                   for m in PRIOR}
    print(f"\n  turnover matched within 0.5-2.0x of mFR: {turnover_ok}")

    def contrast(arm, base, label, tag=""):
        x = np.array([float(intF[arm][s]) for s in seeds])
        y = np.array([float(intF[base][s]) for s in seeds])
        d = 100.0 * (x - y) / y
        med, lo, hi = boot_median(d)
        _, lo90, hi90 = boot_median(d, lo_pct=5.0, hi_pct=95.0)
        wins = int((d < 0).sum())
        print(f"  {label:<34} {med:+8.2f} %   95% CI [{lo:+7.2f}, {hi:+7.2f}]   "
              f"{wins:2d}/{len(d)}{tag}")
        return dict(median_pct=med, ci95=[lo, hi], ci90=[lo90, hi90],
                    wins=wins, n=len(d))

    res = {}
    print(f"\n--- PRIMARY and ATTRIBUTION (endpoint: integrated L2(F), HP v3 reference) ---")
    res["mFR vs ABF"] = contrast("fr_estimated", "abf", "mFR vs ABF")
    res["mFR vs its own sham"] = contrast("fr_estimated", "sham_practical",
                                          "mFR vs its own sham")
    res["sham vs ABF"] = contrast("sham_practical", "abf", "sham vs ABF")

    print(f"\n--- Q1 NOVELTY: mFR against directed prior art ---")
    for m in PRIOR:
        res[f"mFR vs {m}"] = contrast("fr_estimated", m, f"mFR vs {m}")
    print(f"\n--- context: does the prior art beat plain ABF at all? ---")
    for m in PRIOR:
        res[f"{m} vs ABF"] = contrast(m, "abf", f"{m} vs ABF")

    # ---- §4.3 primary ----
    p = res["mFR vs ABF"]
    need = int(np.ceil(0.75 * p["n"]))
    checks = {
        "primary: median <= -10 %": bool(p["median_pct"] <= -10.0),
        "primary: 95 % CI upper < -5 %": bool(p["ci95"][1] < -5.0),
        f"primary: wins >= {need}/{p['n']}": bool(p["wins"] >= need),
        "attribution: mFR-vs-sham 95 % CI upper < 0":
            bool(res["mFR vs its own sham"]["ci95"][1] < 0.0),
    }
    fr_ess = float(np.median([ess["fr_estimated"][s] for s in seeds]))
    fr_w = float(np.median([wmax["fr_estimated"][s] for s in seeds]))
    fr_ess_frac = fr_ess / N_REPLICAS if fr_ess > 1.0 else fr_ess
    checks["genealogy: windowed ESS_anc/N >= 0.30"] = bool(fr_ess_frac >= 0.30)
    checks["genealogy: w_max <= 0.05"] = bool(fr_w <= 0.05)
    run_ess = {a: float(np.median([ess_run[a][s] for s in seeds])) / N_REPLICAS
               for a in ARMS if ess_run.get(a)}
    print(f"\n--- preregistered checks (§4.3) ---")
    print(f"  [mFR genealogy: windowed ESS_anc/N = {fr_ess_frac:.3f} (gate), "
          f"max ancestor frac = {fr_w:.4f}]")
    print(f"  [run-long ESS_anc/N, NOT the gate statistic: "
          + ", ".join(f"{a}={run_ess[a]:.3f}" for a in ("fr_estimated", "sham_practical")
                      if a in run_ess)
          + " -- the random sham sits here too, which is why this one cannot be the gate]")
    for k, v in checks.items():
        print(f"  {k:<48} {v}")

    # ---- Q1 verdict, superiority then TOST equivalence ----
    print(f"\n--- Q1 VERDICT (novelty vs prior directed selection) ---")
    q1 = {}
    for m in PRIOR:
        r = res[f"mFR vs {m}"]
        superior = bool(r["median_pct"] <= -5.0 and r["ci95"][1] < 0.0)
        # TOST: equivalent iff the 90 % CI lies strictly inside +-margin. This is the positive
        # demonstration §4.3 demands, NOT "the 95 % CI happened to contain zero".
        equivalent = bool(r["ci90"][0] > -EQUIV_MARGIN and r["ci90"][1] < EQUIV_MARGIN)
        worse = bool(r["median_pct"] >= 5.0 and r["ci95"][0] > 0.0)
        verdict = ("mFR SUPERIOR" if superior else
                   "EQUIVALENT (TOST)" if equivalent else
                   "mFR WORSE" if worse else "INCONCLUSIVE")
        q1[m] = dict(verdict=verdict, superior=superior, equivalent_tost=equivalent,
                     worse=worse, median_pct=r["median_pct"], ci95=r["ci95"],
                     ci90=r["ci90"], tost_margin=EQUIV_MARGIN,
                     turnover_matched=turnover_ok[m])
        print(f"  vs {m:<18} {verdict:<20} median {r['median_pct']:+.2f} %  "
              f"90% CI [{r['ci90'][0]:+.2f}, {r['ci90'][1]:+.2f}]  "
              f"(TOST margin +-{EQUIV_MARGIN:.0f} %)")

    novelty = all(q1[m]["superior"] for m in PRIOR)
    all_tie = all(q1[m]["equivalent_tost"] for m in PRIOR)
    # "superior vs one arm, equivalent vs the other" is a DECIDED outcome, not an inconclusive
    # one: both per-arm verdicts are positive statements, and TOST equivalence is a positive
    # demonstration rather than a failure to reject. Only a genuinely undecided arm -- one that
    # is neither superior, nor worse, nor equivalent within the margin -- is inconclusive.
    decided = all(q1[m]["verdict"] != "INCONCLUSIVE" for m in PRIOR)
    if novelty:
        claim = ("FR direction BEATS prior directed selection on both arms "
                 "at matched turnover")
    elif all_tie:
        claim = ("TIE on both arms. The claim becomes: a principled selection rule consistent "
                 "with the establishment mechanism -- NOT a performance win over prior art")
    elif decided:
        beat = [m for m in PRIOR if q1[m]["superior"]]
        tied = [m for m in PRIOR if q1[m]["equivalent_tost"]]
        lost = [m for m in PRIOR if q1[m]["worse"]]
        claim = ("SPLIT, and decided on every arm: mFR beats " + ", ".join(beat or ["none"])
                 + "; ties (TOST) " + ", ".join(tied or ["none"])
                 + ("; loses to " + ", ".join(lost) if lost else "")
                 + ". §4.3 requires BOTH to license novelty, so the novelty claim FAILS and "
                   "the claim becomes: a principled selection rule consistent with the "
                   "establishment mechanism, matched but not beaten by a simpler rule")
    else:
        claim = ("INCONCLUSIVE on at least one arm -- no novelty claim is licensed by §4.3")
    print(f"\n  PRIMARY (mFR vs ABF) PASSES: {all(checks.values())}")
    print(f"  Q1 NOVELTY CLAIM LICENSED:   {novelty}")
    print(f"  -> {claim}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "five_arm_verdict.json")
    with open(path, "w") as fh:
        json.dump(dict(n_seeds=len(seeds), seeds=seeds, arms=list(ARMS),
                       endpoint="integrated_l2_f", reference="cache/phase_hp_v3",
                       target_turnover=tgt, turnover_matched=turnover_ok,
                       contrasts=res, primary_checks=checks,
                       primary_passed=all(checks.values()),
                       q1=q1, novelty_claim_licensed=novelty, claim=claim), fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
