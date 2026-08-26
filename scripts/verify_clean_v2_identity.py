#!/usr/bin/env python3
"""Production-scale identity checks for the clean-v2 baseline, as a data file.

Two claims in `docs/CLEAN_V2_PREREGISTRATION.md` are about the engine rather
than about the science, and both need numbers rather than prose:

**Gate A at production scale.** ``gamma = 0`` must reproduce plain ABF.  On GPU
the engine is not bitwise reproducible (reduction order varies), so "identical"
needs a scale.  This comparison supplies its own: the ``gamma = 0`` arm never
enters the FR branch, so it is identical *by construction* and its residual **is**
the measured non-determinism floor -- not a tolerance chosen to pass.

**The removal of ``abf.ema_alpha`` was a target change, not an estimator change.**
Clean-v2 forbids that key, so the plain-ABF baseline must be shown to be the same
plain ABF the legacy campaigns ran.  Running ``abf_only`` at matched seeds under a
clean config and under a legacy config that still carries ``abf.ema_alpha``,
``fr.score_clip`` and ``fr.max_event_fraction`` is the direct test, and it is
judged against the floor the first comparison measured.

Produce the two stage roots with::

    export CUDA_VISIBLE_DEVICES=2
    python scripts/run_reference_2d.py --config configs/clean_v2/identity_clean.yaml
    python scripts/run_clean_v2.py --config configs/clean_v2/identity_clean.yaml \
        --stage calibration --device cuda
    python scripts/run_reference_2d.py --config configs/clean_v2/identity_legacy.yaml
    python scripts/run_abf_fr_grid_torch.py --config configs/clean_v2/identity_legacy.yaml \
        --stage tuning_gpu --device cuda

then::

    python scripts/verify_clean_v2_identity.py \
        --clean-root results/clean_v2/identity/clean/calibration \
        --legacy-root results/clean_v2/identity/legacy/tuning_gpu \
        --out results/clean_v2/identity_checks.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

METRICS = ["l2_F_R12", "l2_Fprime_R12", "l2_F_full", "l2_Fprime_full",
           "deltaF_error"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean-root", required=True)
    p.add_argument("--legacy-root", required=True)
    p.add_argument("--out", default="results/clean_v2/identity_checks.json")
    return p.parse_args(argv)


def _load(stage_root, kind):
    hits = [f for f in os.listdir(stage_root)
            if f.endswith(f"_{kind}.csv") and "__" not in f]
    if not hits:
        raise SystemExit(f"no merged *_{kind}.csv under {stage_root}")
    return pd.read_csv(os.path.join(stage_root, sorted(hits)[0]))


def _compare(a_long, b_long, a_prof, b_prof, seed):
    def _ser(df, method):
        return df[(df.method == method[0]) & (df.seed == seed)].sort_values("step")
    out = {"seed": int(seed)}
    ga, gb = a_long.sort_values("step"), b_long.sort_values("step")
    if len(ga) != len(gb):
        raise SystemExit(f"seed {seed}: {len(ga)} vs {len(gb)} snapshots")
    for col in METRICS:
        out[f"max_abs_delta_{col}"] = float(
            np.max(np.abs(ga[col].to_numpy() - gb[col].to_numpy())))
    pa = a_prof.sort_values("x"); pb = b_prof.sort_values("x")
    out["max_abs_delta_Fprime_profile"] = float(
        np.max(np.abs(pa.Fprime_hat.to_numpy() - pb.Fprime_hat.to_numpy())))
    out["max_abs_delta_F_profile"] = float(
        np.max(np.abs(pa.F_hat.to_numpy() - pb.F_hat.to_numpy())))
    return out


def main(argv=None):
    args = parse_args(argv)
    cl, cp = _load(args.clean_root, "runs_long"), _load(args.clean_root, "profiles")
    ll, lp = _load(args.legacy_root, "runs_long"), _load(args.legacy_root, "profiles")
    seeds = sorted(set(cl[cl.method == "abf_only"].seed)
                   & set(ll[ll.method == "abf_only"].seed))
    if not seeds:
        raise SystemExit("no seed is present as abf_only in both stage roots")

    sim_rows = cl[cl.method == "abf_only"]
    payload = {
        "n_steps": int(sim_rows.step.max()),
        "horizon": float(sim_rows.t.max()),
        "seeds": [int(s) for s in seeds],
        "clean_root": os.path.relpath(args.clean_root),
        "legacy_root": os.path.relpath(args.legacy_root),
        "gate_A_gamma0_vs_plain_abf": [],
        "ema_alpha_clean_vs_legacy_plain_abf": [],
    }
    for seed in seeds:
        base = cl[(cl.method == "abf_only") & (cl.seed == seed)]
        zero = cl[(cl.method == "abf_fr_physical") & (cl.seed == seed)]
        if zero.empty:
            raise SystemExit(
                f"seed {seed} has no gamma=0 physical arm in the clean root; "
                f"without it there is no measured non-determinism floor and the "
                f"ema comparison has nothing to be judged against")
        row = _compare(base, zero,
                       cp[(cp.method == "abf_only") & (cp.seed == seed)],
                       cp[(cp.method == "abf_fr_physical") & (cp.seed == seed)],
                       seed)
        row["fr_events"] = int(zero.cumulative_fr_events.iloc[-1])
        row["fr_replacements"] = int(zero.cumulative_replacements.iloc[-1])
        row["counters_exactly_zero"] = bool(
            row["fr_events"] == 0 and row["fr_replacements"] == 0)
        payload["gate_A_gamma0_vs_plain_abf"].append(row)

        payload["ema_alpha_clean_vs_legacy_plain_abf"].append(_compare(
            base, ll[(ll.method == "abf_only") & (ll.seed == seed)],
            cp[(cp.method == "abf_only") & (cp.seed == seed)],
            lp[(lp.method == "abf_only") & (lp.seed == seed)], seed))

    floor = max(r["max_abs_delta_Fprime_profile"]
                for r in payload["gate_A_gamma0_vs_plain_abf"])
    ema = max(r["max_abs_delta_Fprime_profile"]
              for r in payload["ema_alpha_clean_vs_legacy_plain_abf"])
    payload["measured_nondeterminism_floor_Fprime"] = floor
    payload["ema_removal_delta_Fprime"] = ema
    payload["ema_removal_within_2x_floor"] = bool(ema <= 2.0 * floor)
    payload["gate_A_counters_all_zero"] = bool(
        all(r["counters_exactly_zero"] for r in payload["gate_A_gamma0_vs_plain_abf"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print(f"[identity] {len(seeds)} matched seed(s), {payload['n_steps']} steps, "
          f"T={payload['horizon']:g}")
    print(f"[identity] Gate A: FR counters exactly zero on every seed: "
          f"{payload['gate_A_counters_all_zero']}")
    print(f"[identity] measured non-determinism floor (max |dF'|): {floor:.3e}")
    print(f"[identity] removing abf.ema_alpha (max |dF'|):         {ema:.3e}  "
          f"-> within 2x the floor: {payload['ema_removal_within_2x_floor']}")
    print(f"[identity] wrote {os.path.relpath(args.out)}")
    return 0 if (payload["gate_A_counters_all_zero"]
                 and payload["ema_removal_within_2x_floor"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
