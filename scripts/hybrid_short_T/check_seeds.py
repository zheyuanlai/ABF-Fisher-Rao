#!/usr/bin/env python
"""Seed freshness and stream-collision check of the hybrid short-T study (prereg section 5: fresh seeds).

Read-only.  Reuses scripts/final_campaign/check_seeds.py's inventory of every earlier gateway / LTA seed label and
its recorded gateway MT19937 noise seeds, adds the blocks of the final campaign and of every results file found on
disk, and checks every block of configs/hybrid_short_T/campaign.json:

* label disjointness against every earlier label of the same system family;
* gateway S1 Langevin noise seeds f(s, N) = (1000003 s + 7919 N) mod (2^31 - 1) at every ladder N against every
  recorded MT seed and against each other;
* SeedSequence states (generate_state(4)) of the FR / IC / LTA streams and of the counter-mode keys of the new
  blocks against those of every earlier block: no duplicates.

Writes results/hybrid_short_T/seed_check.json and exits non-zero on any collision.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hst_common as H  # noqa: E402

sys.path.insert(0, os.path.join(H.ROOT, "scripts", "final_campaign"))
import check_seeds as CS  # noqa: E402


def earlier_labels():
    gw = set(CS.USED_GATEWAY_LABELS)
    lta = set(CS.USED_LTA_LABELS)
    for v in CS.NEW_GATEWAY.values():
        gw |= set(v[0])
    for v in CS.NEW_LTA.values():
        lta |= set(v[0])
    # every seed label in a stored gateway / LTA run file name of the equal-budget, mechanism and final campaigns
    pats = {"gw": ["results/equal_budget_v2/gateway*/N*/s*_*.npz", "results/equal_budget_v2/confirm_best_gateway/N*/s*",
                   "results/mechanism/**/gateway*/**/s*", "results/final_campaign/**/gateway/**/s*"],
            "lta": ["results/equal_budget_v2/lta_T*/N*/s*_*.npz", "results/equal_budget_v2/confirm_best_lta*/N*/s*",
                    "results/mechanism/**/lta*/**/s*", "results/final_campaign/**/lta*/**/s*",
                    "results/r2_campaign/**/s*", "results/adaptive_fr_campaign/**/s*",
                    "results/convergence_fr_campaign/**/s*"]}
    for fam, ps in pats.items():
        for p in ps:
            for f in glob.glob(os.path.join(H.ROOT, p), recursive=True):
                m = re.search(r"/s(\d+)[._]", f)
                if m:
                    (gw if fam == "gw" else lta).add(int(m.group(1)))
    return gw, lta


def ss_state(entropy):
    return tuple(int(x) for x in np.random.SeedSequence(entropy).generate_state(4))


def main():
    X = H.load_config()
    problems, rep = [], {}
    gw_old, lta_old = earlier_labels()
    rep["n_earlier_gateway_labels"], rep["n_earlier_lta_labels"] = len(gw_old), len(lta_old)
    new = {}
    for s, sx in X["systems"].items():
        for k in ("prod_seeds", "confirm_seeds", "bench_seeds", "validation_seeds"):
            new[(s, k)] = [int(x) for x in sx[k]]
    for (s, k), seeds in new.items():
        old = gw_old if s == "gateway" else lta_old
        hit = sorted(set(seeds) & old)
        rep[f"label_overlap_{s}_{k}"] = hit
        if hit:
            problems.append(f"{s} {k}: earlier labels reused {hit[:5]}")
    for s in X["systems"]:
        allb = [x for (ss, k), v in new.items() if ss == s for x in v]
        if len(allb) != len(set(allb)):
            problems.append(f"{s}: new blocks overlap each other")
    # gateway MT noise seeds at every ladder N
    rec, nrec = CS.recorded_mt_seeds()
    rep["recorded_mt_seed_files"], rep["recorded_distinct_mt_seeds"] = nrec, len(rec)
    forbidden = dict(rec)
    for v in CS.USED_VALIDATION_RAW + CS.NEW_VALIDATION_RAW:
        forbidden.setdefault(v, "validation chain (inventory)")
    mine = {}
    gw = X["systems"]["gateway"]
    for k in ("prod_seeds", "confirm_seeds", "bench_seeds", "validation_seeds"):
        for sd in gw[k]:
            for N in gw["N_ladder"]:
                f = CS.f_noise(sd, N)
                if f in forbidden:
                    problems.append(f"gateway noise seed f({sd}, {N}) = {f} collides with {forbidden[f]}")
                if f in mine:
                    problems.append(f"gateway noise seed f({sd}, {N}) = f{mine[f]}")
                mine[f] = (sd, N)
    rep["new_gateway_noise_seeds"] = len(mine)
    # SeedSequence states: gateway FR [seed, N, 1], counter [seed, N, 2]; LTA [seed, N, T] (+ counter [.., 2])
    states = {}
    for (s, k), seeds in new.items():
        sx = X["systems"][s]
        for sd in seeds:
            for N in sx["N_ladder"]:
                ents = ([[sd, N, 1], [sd, N, 2]] if s == "gateway" else
                        [[sd, N, int(round(sx["T_K"]))], [sd, N, int(round(sx["T_K"])), 2]])
                for e in ents:
                    st = ss_state(e)
                    if st in states and states[st] != tuple(e):
                        problems.append(f"SeedSequence collision {e} vs {states[st]}")
                    states[st] = tuple(e)
    old_states = 0
    for sd in sorted(gw_old):
        for N in gw["N_ladder"]:
            for e in ([sd, N, 1], [sd, N, 2]):
                st = ss_state(e)
                old_states += 1
                if st in states:
                    problems.append(f"SeedSequence state of new {states[st]} equals earlier {e}")
    for T in (300, 150):
        for sd in sorted(lta_old):
            for N in X["systems"]["lta300"]["N_ladder"]:
                for e in ([sd, N, T], [sd, N, T, 2]):
                    st = ss_state(e)
                    old_states += 1
                    if st in states:
                        problems.append(f"SeedSequence state of new {states[st]} equals earlier {e}")
    rep["new_seedsequence_states"], rep["earlier_states_checked"] = len(states), old_states
    rep["problems"] = problems
    rep["verdict"] = "PASS" if not problems else "FAIL"
    out = os.path.join(H.OUT_ROOT, "seed_check.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("label_overlap")}, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
