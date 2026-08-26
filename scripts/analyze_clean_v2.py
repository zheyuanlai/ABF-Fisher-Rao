#!/usr/bin/env python3
"""Acceleration analysis for a clean-v2 stage.

The only question asked here is whether the ABF convergence curve moved to the
left.  For every FR schedule and every frozen threshold this computes

    tau_eps  per seed (three consecutive frames at or below eps, censored at T),
    S_eps    = E[tau~_baseline] / E[tau~_arm]  over matched seeds,

with a paired bootstrap CI, and it reports the pre-declared verdict.  Final and
integrated L2 error are computed and printed as *safety* diagnostics -- they
never enter a verdict and never rank a schedule.

Examples
--------
  # Stage 2, the 9-schedule map
  python scripts/analyze_clean_v2.py \
      --stage-root results/clean_v2/stage2_pilot/pilot \
      --thresholds results/clean_v2/thresholds.json --screen pilot

  # Stage 3, fresh seeds, three arms
  python scripts/analyze_clean_v2.py \
      --stage-root results/clean_v2/stage3_confirmation/confirmation \
      --thresholds results/clean_v2/thresholds.json --screen confirm
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abffr import accel  # noqa: E402

BASELINE_METHOD = "abf_only"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage-root", required=True)
    p.add_argument("--thresholds", required=True)
    p.add_argument("--scope", default=None,
                   help="Evaluation scope (default: the frozen primary scope).")
    p.add_argument("--screen", default="pilot", choices=["pilot", "confirm"])
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--boot-seed", type=int, default=20260826)
    p.add_argument("--out", default=None,
                   help="Destination CSV (default: <stage-root>/acceleration.csv).")
    return p.parse_args(argv)


def _load(stage_root, kind):
    hits = [f for f in os.listdir(stage_root)
            if f.endswith(f"_{kind}.csv") and "__" not in f]
    if not hits:
        return None
    return pd.read_csv(os.path.join(stage_root, sorted(hits)[0]))


def _hits(df, col, eps, horizon):
    """``{seed: Hit}`` for one config's error curve."""
    out = {}
    for seed, g in df.sort_values("t").groupby("seed"):
        out[int(seed)] = accel.restricted_hitting_time(
            g["t"].to_numpy(), g[col].to_numpy(), eps, horizon,
            consecutive=accel.CONSECUTIVE_FRAMES)
    return out


def main(argv=None):
    args = parse_args(argv)
    frozen = json.load(open(args.thresholds))
    scope = args.scope or frozen["primary_scope"]
    if scope not in frozen["thresholds"]:
        raise SystemExit(f"scope {scope!r} was not frozen in {args.thresholds}")
    eps_F = frozen["thresholds"][scope]["F"]
    eps_Fp = frozen["thresholds"][scope]["Fprime"]
    fractions = frozen["fractions"]

    long_df = _load(args.stage_root, "runs_long")
    if long_df is None:
        raise SystemExit(f"no merged *_runs_long.csv under {args.stage_root}")
    final_df = _load(args.stage_root, "final_summary")
    horizon = float(long_df["t"].max())
    if abs(horizon - float(frozen["horizon"])) > 1e-9:
        print(f"[analyze] NOTE: stage horizon T={horizon:g} differs from the "
              f"calibration horizon T={frozen['horizon']:g}; tau is censored at "
              f"this stage's T, which is the honest choice for a longer run.")

    col_F, col_Fp = f"l2_F_{scope}", f"l2_Fprime_{scope}"
    for col in (col_F, col_Fp):
        if col not in long_df:
            raise SystemExit(f"{col} missing from the stage CSV")

    base_df = long_df[long_df["method"] == BASELINE_METHOD]
    if base_df.empty:
        raise SystemExit(
            f"no {BASELINE_METHOD} rows: a speedup needs its own baseline, and "
            f"borrowing one from another stage would break the seed pairing")
    base_F = [_hits(base_df, col_F, e, horizon) for e in eps_F]
    base_Fp = [_hits(base_df, col_Fp, e, horizon) for e in eps_Fp]

    repl = {}
    if final_df is not None and "mean_fr_event_fraction" in final_df:
        repl = final_df.groupby("config_id")["mean_fr_event_fraction"].median()

    rows = []
    for config_id, g in long_df[long_df["method"] != BASELINE_METHOD].groupby(
            "config_id"):
        arm_F = [_hits(g, col_F, e, horizon) for e in eps_F]
        arm_Fp = [_hits(g, col_Fp, e, horizon) for e in eps_Fp]
        shared = sorted(set(arm_F[0]) & set(base_F[0]))
        dropped = sorted(set(arm_F[0]) - set(base_F[0]))
        if dropped:
            print(f"[analyze] WARNING {config_id}: {len(dropped)} arm seed(s) "
                  f"have no matched baseline and are excluded: {dropped}")
        if not shared:
            print(f"[analyze] WARNING {config_id}: no matched seeds; skipped")
            continue

        sp_F, sp_Fp = [], []
        for k in range(len(eps_F)):
            sp_F.append(accel.paired_bootstrap_speedup(
                [base_F[k][s] for s in shared], [arm_F[k][s] for s in shared],
                n_boot=args.n_boot, seed=args.boot_seed + k))
            sp_Fp.append(accel.paired_bootstrap_speedup(
                [base_Fp[k][s] for s in shared], [arm_Fp[k][s] for s in shared],
                n_boot=args.n_boot, seed=args.boot_seed + 100 + k))

        meta = g.iloc[0]
        rf = float(repl.get(config_id, np.nan))
        row = dict(
            config_id=config_id, method=meta["method"],
            target_type=meta["target_type"], gamma=float(meta["gamma"]),
            fr_every=int(meta["fr_every"]),
            burnin_fraction=float(meta["burnin_fraction"]),
            stop_fraction=float(meta["stop_fraction"]),
            scope=scope, n_matched_seeds=len(shared),
            replacement_fraction=rf,
        )
        for k, f in enumerate(fractions):
            row[f"eps_F_{k+1}"] = eps_F[k]
            row[f"eps_Fprime_{k+1}"] = eps_Fp[k]
            for label, sp in (("F", sp_F[k]), ("Fprime", sp_Fp[k])):
                d = sp.to_row()
                row[f"S_{label}_{k+1}"] = d["s"]
                row[f"S_{label}_{k+1}_lo"] = d["ci_lo"]
                row[f"S_{label}_{k+1}_hi"] = d["ci_hi"]
                row[f"tau_base_{label}_{k+1}"] = d["mean_base"]
                row[f"tau_arm_{label}_{k+1}"] = d["mean_arm"]
                row[f"cens_base_{label}_{k+1}"] = d["n_censored_base"]
                row[f"cens_arm_{label}_{k+1}"] = d["n_censored_arm"]
                # P(tau <= T) on each side.  S is a RESTRICTED ratio at horizon
                # T; which way restriction biases it is decided by these two
                # numbers, so they travel with every S rather than in an
                # appendix.
                row[f"hit_{label}_{k+1}_base"] = d["hit_fraction_base"]
                row[f"hit_{label}_{k+1}_arm"] = d["hit_fraction_arm"]
                row[f"inflated_{label}_{k+1}"] = d["censoring_inflates"]
        # Speedup objects, not bare ratios: the screen must see censoring, or an
        # inflated cell wins selection and reaches fresh seeds.
        row["promising"] = accel.pilot_promising(sp_F, sp_Fp)
        row["confirmed"] = accel.confirms(sp_F, sp_Fp)
        row["C_accel"] = accel.accel_cost(row["S_F_2"], rf)
        # Safety diagnostics only -- these rank nothing.
        if final_df is not None:
            fg = final_df[final_df["config_id"] == config_id]
            fb = final_df[final_df["method"] == BASELINE_METHOD]
            for col, name in (("final_l2_F", "final_l2_F"),
                              ("final_l2_Fprime", "final_l2_Fprime")):
                if col in fg and not fg.empty and not fb.empty:
                    row[f"safety_{name}_ratio"] = (
                        float(fg[col].median()) / float(fb[col].median()))
            if "final_ancestor_ess" in fg and not fg.empty:
                row["final_ancestor_ess"] = float(fg["final_ancestor_ess"].median())
        rows.append(row)

    if not rows:
        raise SystemExit("no FR configs found to analyse")
    out_df = pd.DataFrame(rows).sort_values("S_F_2", ascending=False)
    out_path = args.out or os.path.join(args.stage_root, "acceleration.csv")
    out_df.to_csv(out_path, index=False)

    key = "confirmed" if args.screen == "confirm" else "promising"
    print(f"\n  scope={scope}  T={horizon:g}  "
          f"eps_F=({eps_F[0]:.4g}, {eps_F[1]:.4g})  "
          f"eps_F'=({eps_Fp[0]:.4g}, {eps_Fp[1]:.4g})")
    print(f"  baseline: {BASELINE_METHOD}, {len(base_F[0])} seeds")
    print(f"  S is the RESTRICTED speedup at horizon T, "
          f"E[min(tau_base,T)] / E[min(tau_arm,T)] -- not an estimate of the "
          f"unrestricted ratio.")
    print(f"  hit = P(tau <= T) on each side; '!' marks a threshold where the "
          f"arm is censored MORE than the baseline, which inflates S there.\n")
    # The arm name is printed, not implied: an estimated-target and an
    # oracle-target row can carry identical (gamma, L_FR) and would otherwise be
    # indistinguishable in the table.
    short = {"abf_fr_physical": "physical", "abf_fr_physical_oracle": "ORACLE*"}
    head = (f"{'arm':>9} {'gamma':>6} {'L_FR':>5} {'n':>3} {'S_F,1':>19} "
            f"{'S_F,2':>19} {'hit1 b/a':>10} {'hit2 b/a':>10} "
            f"{'S_F′,1':>7} {'S_F′,2':>7} {'repl':>6} {key:>10}")
    print(head)
    print("-" * len(head))
    for _, r in out_df.iterrows():
        f1 = "!" if r["inflated_F_1"] else " "
        f2 = "!" if r["inflated_F_2"] else " "
        print(f"{short.get(r['method'], r['method'])[:9]:>9} "
              f"{r['gamma']:>6.3g} {r['fr_every']:>5d} "
              f"{r['n_matched_seeds']:>3d} "
              f"{r['S_F_1']:>6.3f}[{r['S_F_1_lo']:.2f},{r['S_F_1_hi']:.2f}]{f1} "
              f"{r['S_F_2']:>6.3f}[{r['S_F_2_lo']:.2f},{r['S_F_2_hi']:.2f}]{f2} "
              f"{r['hit_F_1_base']:>4.2f}/{r['hit_F_1_arm']:<5.2f} "
              f"{r['hit_F_2_base']:>4.2f}/{r['hit_F_2_arm']:<5.2f} "
              f"{r['S_Fprime_1']:>7.3f} {r['S_Fprime_2']:>7.3f} "
              f"{r['replacement_fraction']:>6.3f} {str(r[key]):>10}")
    if "abf_fr_physical_oracle" in set(out_df["method"]):
        print("\n  * ORACLE uses q propto exp(-beta F_ref).  It is a diagnostic "
              "answering whether A_t is accurate enough by t_burn; it is never "
              "a candidate method and never enters a verdict about the method.")
    ncens = int(out_df[[c for c in out_df if c.startswith("cens_")]].to_numpy().sum())
    print(f"\n  censored run-thresholds across all arms and baselines: {ncens} "
          f"(restricted at T, never dropped)")
    inflated = out_df[[c for c in out_df if c.startswith("inflated_F_")]].to_numpy()
    if inflated.any():
        print(f"  ! {int(inflated.sum())} free-energy threshold(s) have MORE arm "
              f"censoring than baseline censoring.  S is inflated there and "
              f"cannot carry a verdict; the horizon is the thing to fix.")
    worst = min(float(out_df[f"hit_F_{k}_base"].min()) for k in (1, 2))
    if worst < 0.5:
        print(f"  ! plain ABF reaches a frozen threshold in only "
              f"{worst:.0%} of seeds within T.  The horizon, not the method, is "
              f"the binding constraint -- say so in the result.")
    print(f"  wrote {os.path.relpath(out_path)}")
    if args.screen == "confirm":
        deployable = out_df[out_df["method"] == "abf_fr_physical"]
        won = deployable[deployable["confirmed"]]
        print(f"\n  VERDICT: {len(won)} of {len(deployable)} DEPLOYABLE arm(s) meet the "
              f"pre-declared confirmatory criterion "
              f"(S_F >= {accel.PILOT_MIN_S_F} at both thresholds, "
              f"paired-bootstrap 95% CI above 1, and no threshold at which the "
              f"arm is censored more than the baseline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
