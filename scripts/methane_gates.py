"""Gates A / B / C from ABF-only screen data (Gate 0 already passed at the reference stage).

Order is Amendment 10's, and classification is **by the first failing gate**:

    Gate 0  is the ABF conditional mean force trustworthy?   PASSED at the reference stage
                                                            (0.048 global / 0.081 core)
    Gate A  can the relevant states be told apart through xi?
    Gate B  were they discovered?          T_hit < 0.1 T on >= 6 of 8 seeds
    Gate C  were they established?         occupancy < 0.5 Q*_k(t) for >= 0.20 T, second half

States come from the accepted reference by Amendment 3.  That reference is **single-basin** --
`W` has contact/barrier/solvent-separated features at literature positions but the barrier is
0.67 kT from the higher minimum, below the 2 kT merge threshold -- so the preregistered
**tercile fallback** fires and the partition is declared as a partition of the coordinate, not
as a claim of metastability.

`Q*_k(t)` is the **bias-aware** target of §2.1: a state can be rare at equilibrium and perfectly
populated under the bias ABF has already applied, so the deficit is measured against
`exp(-beta[F_ref - B_t])`, not against the unbiased equilibrium mass.

This script reads only saved traces.  It never touches the sampler, and every threshold in it is
preregistered.

Usage:
    python scripts/methane_gates.py --screen results/methane/screen_N512 --ref results/methane/ref
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from methane import system as msys                               # noqa: E402

T_HIT_FRAC = 0.10          #: Gate B: T_hit < 0.1 T
GATE_B_MIN_SEEDS = 6       #: of 8
DEFICIT_FRAC = 0.50        #: Gate C: occupancy below 0.5 Q*
DEFICIT_SPAN = 0.20        #: for a contiguous 0.20 T, in the second half
GATE_A_TV = 0.30           #: Gate A: max pairwise TV(p(n_gap | state)) must reach this


def tercile_edges(lo, hi):
    w = (hi - lo) / 3.0
    return [(lo, lo + w), (lo + w, lo + 2 * w), (lo + 2 * w, hi)]


def state_of(xi, edges):
    out = np.full(xi.shape, -1, dtype=np.int64)
    for k, (a, b) in enumerate(edges):
        out[(xi >= a) & (xi < b)] = k
    out[xi >= edges[-1][1]] = len(edges) - 1
    return out


def bias_aware_target(F_ref_grid, B_t, grid, edges, beta):
    """`Q*_k(t)` from the reference and the bias ABF has applied at time `t`."""
    w = np.exp(-beta * (F_ref_grid - B_t))
    w = np.where(np.isfinite(w), w, 0.0)
    tot = w.sum()
    if tot <= 0:
        return np.full(len(edges), np.nan)
    return np.asarray([w[(grid >= a) & (grid < b)].sum() / tot for a, b in edges])


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def tv_from_samples(a, b, bins):
    ha, _ = np.histogram(a, bins=bins, density=False)
    hb, _ = np.histogram(b, bins=bins, density=False)
    if ha.sum() == 0 or hb.sum() == 0:
        return np.nan
    pa = ha / ha.sum()
    pb = hb / hb.sum()
    return 0.5 * np.abs(pa - pb).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="results/methane/screen_N512")
    ap.add_argument("--ref", default="results/methane/ref")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or args.screen

    ref = json.load(open(os.path.join(args.ref, "reference.json")))
    r_ref = np.asarray(ref["r_nm"])
    F_ref = np.asarray(ref["F_kJ"])
    beta = msys.beta_per_kJ()
    kT = msys.kT_kJ()

    files = sorted(glob.glob(os.path.join(args.screen, "seed*.npz")))
    if not files:
        raise SystemExit(f"no seed files in {args.screen}")
    print(f"[load] {len(files)} seeds: {[os.path.basename(f) for f in files]}")

    per_seed = []
    ngap_pool = {0: [], 1: [], 2: []}
    for path in files:
        d = np.load(path)
        grid = d["grid"].astype(np.float64)
        edges = tercile_edges(float(grid[0]), float(grid[-1]))
        xi = d["xi_trace"].astype(np.float64)            # (n_frames, n_walkers)
        steps = d["xi_steps"]
        dt = msys.DT_PS
        times = steps * dt
        T = float(times[-1])
        F_on_grid = np.interp(grid, r_ref, F_ref)

        # ---- Gate B: first persistent arrival in each state --------------------------------
        st = state_of(xi, edges)
        t_hit = {}
        for k in range(3):
            present = (st == k).any(axis=1)
            # persistence: state occupied at two consecutive trace frames
            idx = np.flatnonzero(present[:-1] & present[1:])
            t_hit[k] = float(times[idx[0]]) if idx.size else float("inf")

        # ---- Gate C: occupancy vs the bias-aware target ------------------------------------
        occ = np.asarray([[np.mean(state_of(xi[i], edges) == k) for k in range(3)]
                          for i in range(len(times))])
        pmf_t = d["diag_pmf"].astype(np.float64)          # (n_saves, n_grid), the applied bias
        pmf_steps = d["diag_steps"]
        qstar = np.zeros_like(occ)
        for i, t_step in enumerate(steps):
            j = int(np.argmin(np.abs(pmf_steps - t_step)))
            qstar[i] = bias_aware_target(F_on_grid, pmf_t[j], grid, edges, beta)

        half = times >= 0.5 * T
        deficit_span = {}
        for k in range(3):
            below = (occ[half, k] < DEFICIT_FRAC * qstar[half, k])
            frames = longest_run(below)
            span = frames * float(times[1] - times[0])
            deficit_span[k] = span

        # ---- Gate A material: n_gap by state ------------------------------------------------
        ng = d["ngap_trace"].astype(np.float64)
        ngx = d["ngap_xi"].astype(np.float64)
        ngs = state_of(ngx, edges)
        for k in range(3):
            ngap_pool[k].append(ng[ngs == k])

        per_seed.append(dict(seed=int(d["seed"]), T_ps=T, t_hit=t_hit,
                             deficit_span=deficit_span,
                             occ_final=occ[-1].tolist(), qstar_final=qstar[-1].tolist(),
                             max_pinned=float(occ[half].max())))
        print(f"  seed {int(d['seed'])}: T_hit = "
              f"{[f'{t_hit[k]:.1f}' if np.isfinite(t_hit[k]) else 'never' for k in range(3)]} ps"
              f"   worst deficit span = {max(deficit_span.values()):.1f} ps"
              f"   max tercile occupancy (2nd half) = {occ[half].max():.3f}")

    T = per_seed[0]["T_ps"]
    # ---- Gate 0 pinning clause (retained by Amendment 9) ----------------------------------
    pinned = max(s["max_pinned"] for s in per_seed)
    gate0_pin_ok = pinned <= 0.90

    # ---- Gate A ---------------------------------------------------------------------------
    pools = {k: (np.concatenate([v for v in ngap_pool[k] if v.size])
                 if any(v.size for v in ngap_pool[k]) else np.array([])) for k in range(3)}
    all_ng = np.concatenate([p for p in pools.values() if p.size]) if any(
        p.size for p in pools.values()) else np.array([0.0])
    bins = np.linspace(0.0, max(1.0, float(all_ng.max())), 21)
    tvs = {}
    for a in range(3):
        for b in range(a + 1, 3):
            if pools[a].size and pools[b].size:
                tvs[f"{a}-{b}"] = tv_from_samples(pools[a], pools[b], bins)
    finite = [v for v in tvs.values() if np.isfinite(v)]
    gate_a = max(finite) if finite else float("nan")
    # An UNCOMPUTABLE Gate A is not a passed Gate A.  It happens when a state is never visited
    # and so has no n_gap samples -- which is a Gate B statement, not a CV-visibility statement,
    # so the verdict must fall through to Gate B explicitly rather than by the accident of a
    # `np.isfinite` guard skipping the Gate A branch.
    empty_states = [k for k in range(3) if not pools[k].size]
    gate_a_computable = not empty_states

    # ---- Gate B ---------------------------------------------------------------------------
    thr = T_HIT_FRAC * T
    hits = {k: sum(1 for s in per_seed if s["t_hit"][k] < thr) for k in range(3)}
    gate_b_ok = all(h >= min(GATE_B_MIN_SEEDS, len(per_seed)) for h in hits.values())

    # ---- Gate C ---------------------------------------------------------------------------
    span_thr = DEFICIT_SPAN * T
    persistent = {k: sum(1 for s in per_seed if s["deficit_span"][k] >= span_thr)
                  for k in range(3)}
    gate_c_deficit = any(v > 0 for v in persistent.values())

    print(f"\n[gate 0] max tercile occupancy over any seed, second half = {pinned:.3f} "
          f"(pinning clause: <= 0.90)  {'OK' if gate0_pin_ok else 'FAIL'}")
    if gate_a_computable:
        print(f"[gate A] max pairwise TV(p(n_gap | tercile)) = {gate_a:.3f}  "
              f"(threshold {GATE_A_TV})  pairs: "
              + ", ".join(f"{k}={v:.3f}" for k, v in tvs.items()))
    else:
        print(f"[gate A] NOT COMPUTABLE -- states {empty_states} have no n_gap samples "
              f"(never visited). Deferring to Gate B; this is not a Gate A pass.")
    print(f"[gate B] T_hit < {thr:.1f} ps on: "
          + ", ".join(f"state {k}: {hits[k]}/{len(per_seed)}" for k in range(3))
          + f"   (need >= {min(GATE_B_MIN_SEEDS, len(per_seed))})  "
          + ("OK" if gate_b_ok else "FAIL -> discovery-limited"))
    print(f"[gate C] persistent deficit (>= {span_thr:.1f} ps below 0.5 Q*) on: "
          + ", ".join(f"state {k}: {persistent[k]}/{len(per_seed)}" for k in range(3)))

    if not gate0_pin_ok:
        verdict = "ABF-baseline-invalid (Gate 0 pinning clause) -- STOP"
    elif gate_a_computable and gate_a < GATE_A_TV:
        verdict = "CV-visibility negative (Gate A) -- STOP, a stop for the CV, not for mFR"
    elif not gate_b_ok:
        verdict = "discovery-limited (Gate B) -- STOP"
    elif not gate_c_deficit:
        verdict = "ABF-sufficient (Gate C: no persistent deficit) -- STOP"
    else:
        verdict = "establishment-limited -- CONTINUE to Gate D"
    print(f"\n[VERDICT] {verdict}")

    res = dict(verdict=verdict, n_seeds=len(per_seed), T_ps=T,
               gate0_max_pinned=pinned, gate0_pin_ok=bool(gate0_pin_ok),
               gateA_max_TV=float(gate_a), gateA_pairs=tvs, gateA_threshold=GATE_A_TV,
               gateA_computable=bool(gate_a_computable), gateA_empty_states=empty_states,
               gateB_hits=hits, gateB_threshold_ps=thr, gateB_ok=bool(gate_b_ok),
               gateC_persistent=persistent, gateC_span_threshold_ps=span_thr,
               gateC_deficit=bool(gate_c_deficit), per_seed=per_seed,
               reference=os.path.join(args.ref, "reference.json"),
               git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip())
    with open(os.path.join(out_dir, "gates.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"[done] -> {out_dir}/gates.json")


if __name__ == "__main__":
    main()
