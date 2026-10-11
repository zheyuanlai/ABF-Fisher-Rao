"""Seed-disjointness and stream-collision check for the final campaign (docs/final_campaign/PREREGISTRATION.md §4).

Read-only. Builds the set of previously used gateway / LTA seed labels and random-stream seeds from
(a) the inventory of every config / ledger / result file (the 2026-10-11 code audit, reproduced in USED_* below) and
(b) the noise seeds actually recorded in the meta of every stored gateway numba run and validation chain,
then checks every new block of this campaign:

* label disjointness (gateway labels; LTA labels);
* gateway Langevin noise seeds f(s, N) = (1000003 s + 7919 N) mod (2^31 - 1) against every recorded MT seed;
* raw MT seeds of the new validation chains (np.random.seed(v) in the harness) against every recorded MT seed and
  every production noise seed;
* SeedSequence states (generate_state(4)) of every FR / LTA / IC / counter-mode stream, new and old: no duplicates.

Writes results/final_campaign/seed_check.json and exits non-zero on any collision.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
P = 2 ** 31 - 1


def f_noise(s, N, off=0):
    return (1_000_003 * int(s) + 7919 * int(N) + off) % P


def r(a, b):
    return list(range(a, b + 1))


# (a) previously used labels (code audit 2026-10-11; sources in docs/final_campaign/PREREGISTRATION.md §4)
USED_GATEWAY_LABELS = sorted(set(
    r(0, 15) + r(100, 131) + r(200, 207) + r(300, 331) + r(400, 415) + r(480, 487) + r(500, 571) + r(580, 587)
    + r(800, 831) + r(900, 915) + r(1200, 1207) + r(1300, 1331) + r(5200, 5207) + r(5300, 5315) + r(7100, 7131)
    + r(8100, 8131) + r(8200, 8263) + [1, 7, 8]))
USED_LTA_LABELS = sorted(set(
    r(600, 615) + r(700, 775) + [900, 901] + r(1000, 1095) + r(1100, 1103) + [1110, 1111, 1200]
    + r(15000, 15015) + r(15100, 15131) + r(30000, 30015) + r(30100, 30131)))
USED_VALIDATION_RAW = sorted(set(
    r(61000, 61015) + r(61100, 61115) + r(61200, 61215) + r(62000, 62015) + [63000, 63001, 63002]
    + [71000 + 1000 * d + g for d in range(7) for g in range(16)]
    + [78000 + 100 * d + g for d in range(7) for g in range(16)]
    + [79000 + 100 * d for d in range(7)]
    + [80000 + 100 * d + g for d in range(7) for g in range(16)]
    + [81000 + 100 * d + g for d in range(7) for g in range(16)]
    + [7000 + 1000 * t + 50 * g + w for t in range(5) for g in range(16) for w in range(40)]
    + r(9100, 9163) + r(9200, 9231) + r(3600, 3663) + r(9300, 9331) + r(9400, 9431) + r(9500, 9515) + r(5100, 5115)))

# new blocks of this campaign
EXPA = r(8300, 8331)
EXPA_SMOKE = [8390, 8391, 8392, 8393]
EXPA_FROZENX = r(8650, 8661)
EXPB_GW = r(8400, 8463)
EXPB_GW_EQ = r(8500, 8563)          # 8500-8531 counter, 8532-8563 extension counter
EXPB_GW_EQ_VAL = r(8700, 8731)      # extension, validated engine
EXPB_GW_BENCH = r(8600, 8609)
EXPB_LTA = r(15200, 15231)
EXPB_LTA_EQ = r(15300, 15331) + r(15360, 15391)   # counter (+ extension)
EXPB_LTA_EQ_VAL = r(15340, 15355) + r(15460, 15491)  # validated engine (+ extension)
EXPB_LTA_BENCH = r(15400, 15409)
LADDER_GW = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
LADDER_LTA = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
NEW_GATEWAY = {"expA": (EXPA, [512, 128]), "expA_smoke": (EXPA_SMOKE, [512, 128, 2048]),
               "expA_frozenx": (EXPA_FROZENX, [32, 64]),
               "expB_gw": (EXPB_GW, LADDER_GW), "expB_gw_eq": (EXPB_GW_EQ, [16, 512]),
               "expB_gw_eq_val": (EXPB_GW_EQ_VAL, [16, 512]), "expB_gw_bench": (EXPB_GW_BENCH, LADDER_GW)}
NEW_LTA = {"expB_lta": (EXPB_LTA, LADDER_LTA), "expB_lta_eq": (EXPB_LTA_EQ, [2, 512]),
           "expB_lta_eq_val": (EXPB_LTA_EQ_VAL, [2, 512]), "expB_lta_bench": (EXPB_LTA_BENCH, LADDER_LTA)}
# validation chains of Exp A (raw MT seeds; d' = DYNAMICS_ORDER index - 7 in {0, 1}; k = h index in {0, 1})
NEW_VALIDATION_RAW = sorted(set(
    [400000 + 1000 * dp + 100 * k + g for dp in range(3) for k in range(2) for g in range(16)]
    + [410000 + 100 * dp + g for dp in range(3) for g in range(16)]
    + [411000 + 100 * dp + 10 * k for dp in range(3) for k in range(2)]
    + [420000 + 100 * dp + g for dp in range(3) for g in range(16)]
    + [421000 + 100 * dp + g for dp in range(3) for g in range(16)]
    + [440000 + i for i in range(4)]))   # expA_scout_transport.py


def recorded_mt_seeds():
    """Noise seeds recorded in the meta of every stored gateway numba run / validation chain (read-only)."""
    out = {}
    pats = ["results/gateway_replica_ladder/*/raw/*.npz", "results/numerical_validation/gateway_dt/*/*.npz",
            "results/equal_budget_v2/gateway/N*/s*_*.npz", "results/equal_budget_v2/confirm_best_gateway/N*/s*_*.npz",
            "results/mechanism/*/*/N*/s*_*.npz", "results/mechanism/smoke/**/N*/s*_*.npz",
            "results/mechanism/parallel_benchmark/gateway/*/s*_*.npz",
            "results/equal_budget_v2/gateway_validation/*/g*.npz", "results/mechanism/validation/*/*/g*.npz"]
    n = 0
    for pat in pats:
        for fn in glob.glob(os.path.join(ROOT, pat), recursive=True):
            if "/exact" in fn:
                continue
            try:
                z = np.load(fn, allow_pickle=False)
                m = json.loads(str(z["meta_json"])) if "meta_json" in z.files else json.loads(str(z["meta"]))
            except Exception:
                continue
            ns = m.get("noise_seed", (m.get("engine_meta") or {}).get("noise_seed", m.get("seed")))
            if ns is None:
                continue
            out.setdefault(int(ns), os.path.relpath(fn, ROOT))
            n += 1
    return out, n


def main():
    problems = []
    rep = {}
    # labels
    for name, (seeds, _) in NEW_GATEWAY.items():
        hit = sorted(set(seeds) & set(USED_GATEWAY_LABELS))
        rep[f"label_overlap_{name}"] = hit
        if hit:
            problems.append(f"{name}: gateway labels reused {hit[:5]}")
    for name, (seeds, _) in NEW_LTA.items():
        hit = sorted(set(seeds) & set(USED_LTA_LABELS))
        rep[f"label_overlap_{name}"] = hit
        if hit:
            problems.append(f"{name}: LTA labels reused {hit[:5]}")
    allnew = [s for v in NEW_GATEWAY.values() for s in v[0]]
    if len(allnew) != len(set(allnew)):
        problems.append("new gateway blocks overlap each other")
    allnew_l = [s for v in NEW_LTA.values() for s in v[0]]
    if len(allnew_l) != len(set(allnew_l)):
        problems.append("new LTA blocks overlap each other")
    # MT noise seeds
    rec, nrec = recorded_mt_seeds()
    rep["recorded_mt_seed_files"] = nrec
    rep["recorded_distinct_mt_seeds"] = len(rec)
    forbidden = dict(rec)
    forbidden.setdefault(12345, "check_mt_layout")
    forbidden.setdefault(f_noise(1, 4), "compile warm-up run_arm(seed 1, N 4)")
    for v in USED_VALIDATION_RAW:
        forbidden.setdefault(v, "validation chain (inventory)")
    new_noise = {}
    for name, (seeds, Ns) in NEW_GATEWAY.items():
        for s, N in itertools.product(seeds, Ns):
            v = f_noise(s, N)
            if v in forbidden:
                problems.append(f"{name}: noise seed of (s {s}, N {N}) = {v} already used by {forbidden[v]}")
            if v in new_noise and new_noise[v] != (s, N):
                problems.append(f"{name}: internal noise-seed collision {(s, N)} vs {new_noise[v]}")
            new_noise[v] = (s, N)
    for v in NEW_VALIDATION_RAW:
        if v in forbidden or v in new_noise:
            problems.append(f"validation raw seed {v} collides")
    rep["min_new_noise_seed"] = min(new_noise)
    # SeedSequence states
    states = {}

    def add(tag, ss):
        k = tuple(ss.generate_state(4).tolist())
        states.setdefault(k, set()).add(tag)

    for s in USED_GATEWAY_LABELS + allnew:
        for N in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
            add(("gwFR", s, N), np.random.SeedSequence([s, N, 1]))
    for s in allnew:
        for N in sorted(set(LADDER_GW + [128, 2048])):
            add(("gwCounter", s, N), np.random.SeedSequence([s, N, 2]))
    for s in USED_LTA_LABELS + allnew_l:
        for N in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            for T in (150, 300):
                for i, ch in enumerate(np.random.SeedSequence([s, N, T]).spawn(3)):
                    add(("lta", s, N, T, i), ch)
    for s in allnew_l:
        for N in LADDER_LTA:
            add(("ltaCounter", s, N), np.random.SeedSequence([s, N, 150, 2]))
    for v in set([1000 + s for s in USED_GATEWAY_LABELS + allnew]) | set(NEW_VALIDATION_RAW):
        add(("int", v), np.random.SeedSequence(v))
    dups = [sorted(map(str, t)) for t in states.values() if len(t) > 1]
    rep["seedsequence_states_checked"] = len(states)
    rep["seedsequence_duplicates"] = dups[:10]
    if dups:
        problems.append(f"{len(dups)} duplicate SeedSequence states")
    rep["problems"] = problems
    rep["verdict"] = "PASS" if not problems else "FAIL"
    os.makedirs(os.path.join(ROOT, "results/final_campaign"), exist_ok=True)
    with open(os.path.join(ROOT, "results/final_campaign/seed_check.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("verdict", "recorded_mt_seed_files", "recorded_distinct_mt_seeds",
                                           "min_new_noise_seed", "seedsequence_states_checked", "problems")}, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
