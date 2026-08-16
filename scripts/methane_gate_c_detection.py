"""Does Gate C *fire*? Detection calibration on methane's real traces.

Methane's Gate C power claim is analytic: `lambda = Q* N` is large, so the smallest deficit
resolvable at 2 sigma on one checkpoint is `2/sqrt(lambda)` = 13-18 %. That characterises
**counting noise** and nothing else. The gate as implemented is not a single-checkpoint test --
it requires a **contiguous** `DEFICIT_SPAN * T` run below `DEFICIT_FRAC * Q*` -- and contiguity
interacts with the trace's own correlation time in a direction the analytic figure does not see:
it suppresses false firing (intended) and it also suppresses true firing (not accounted for).

So the analytic number calibrates **exclusion**. It says nothing about **detection**, and every
methane state cleared the gate, so the gate has never been observed to fire on this system at
all. That gap was named by the NaCl session (2026-08-15) about its own CIP statistic, where it
cannot be closed because no NaCl state had a real deficit either. Here it *can* be closed,
because the deficit can be planted in the real occupancy traces rather than in synthetic data --
preserving the actual correlation structure, which is the whole quantity in question.

Method: scale one state's second-half occupancy by `f` and redistribute the removed population
proportionally across the other states, so the partition still sums to 1 at every checkpoint.
This is a walker-conserving deficit -- walkers that leave state k are somewhere, which is what a
real establishment failure looks like. Then run the unmodified Gate C span test and find the
smallest planted deficit `1 - f` at which it fires.

Usage:
    python scripts/methane_gate_c_detection.py --screen results/methane/screen_N512 \
        --ref results/methane/ref
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from methane import system as msys                                        # noqa: E402
from methane_gates import (DEFICIT_FRAC, DEFICIT_SPAN, GATE_C_MIN_LAMBDA,  # noqa: E402
                           bias_aware_target, longest_run, state_of, tercile_edges)


def plant(occ, k, f, scheme="observed", qstar=None):
    """Scale state `k` by `f`, redistributing its lost mass over the other states.

    Walker-conserving: rows still sum to 1.  A deficit that did NOT conserve walkers would be
    testing the renormalisation, not the gate.

    `scheme` selects WHERE the displaced population goes, and it is a free choice that the
    result must not depend on:

      "observed" -- proportional to the other states' current occupancy.  The NaCl session
                    (2026-08-15) found this one has a failure mode: at checkpoints where every
                    walker is already inside the basin there is nothing to be proportional TO,
                    and an implementation that does not handle it drops the mass, raising the
                    basin's share to 1.0 -- the opposite of a deficit.  Handled here by the
                    uniform fallback and caught in any case by the partition assert below.
      "target"   -- proportional to the other states' bias-aware target `Q*`.
      "uniform"  -- equally over the other states.

    Agreement across the three is what licenses reading the threshold as a property of the gate
    rather than of the planting.
    """
    out = occ.copy()
    lost = (1.0 - f) * out[:, k]
    out[:, k] *= f
    others = [j for j in range(occ.shape[1]) if j != k]
    if scheme == "uniform":
        share = np.full((occ.shape[0], len(others)), 1.0 / len(others))
    elif scheme == "target":
        if qstar is None:
            raise ValueError("scheme='target' needs qstar")
        q = qstar[:, others]
        qs = q.sum(axis=1)
        share = np.where(qs[:, None] > 0, q / np.where(qs[:, None] > 0, qs[:, None], 1.0),
                         1.0 / len(others))
    else:
        rest = out[:, others].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(rest[:, None] > 0,
                             out[:, others] / np.where(rest[:, None] > 0, rest[:, None], 1.0),
                             1.0 / len(others))
    assert np.allclose(share.sum(axis=1), 1.0, atol=1e-10), "redistribution shares must sum to 1"
    out[:, others] += share * lost[:, None]
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-10), "planting broke the partition"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="results/methane/screen_N512")
    ap.add_argument("--ref", default="results/methane/ref")
    ap.add_argument("--out", default=None)
    ap.add_argument("--subsample", type=int, default=0,
                    help="emulate a smaller cell by using only this many walkers, so the "
                         "detection threshold can be read at a lambda the C60 branch will "
                         "actually see (0 = use all)")
    ap.add_argument("--scheme", default="observed", choices=("observed", "target", "uniform"),
                    help="where the displaced population goes; the threshold must not depend "
                         "on this choice")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.screen, "gate_c_detection")
    os.makedirs(out_dir, exist_ok=True)

    ref = json.load(open(os.path.join(args.ref, "reference.json")))
    r_ref = np.asarray(ref["r_nm"], dtype=np.float64)
    F_ref = np.asarray(ref["F_kJ"], dtype=np.float64)
    beta = msys.beta_per_kJ()

    deficits = np.round(np.arange(0.30, 0.86, 0.05), 2)
    fired = {float(d): {k: 0 for k in range(3)} for d in deficits}
    lam_min = {k: np.inf for k in range(3)}
    n_seeds = 0

    for path in sorted(glob.glob(os.path.join(args.screen, "seed*.npz"))):
        d = np.load(path)
        grid = d["grid"].astype(np.float64)
        edges = tercile_edges(float(grid[0]), float(grid[-1]))
        xi = d["xi_trace"].astype(np.float64)
        if args.subsample:
            # Deterministic per-seed draw: emulating a smaller cell must not also introduce a
            # random-seed axis, or the threshold moves for a reason that is not lambda.
            rng = np.random.default_rng(int(d["seed"]))
            xi = xi[:, rng.choice(xi.shape[1], args.subsample, replace=False)]
        times = d["xi_steps"] * msys.DT_PS
        T = float(times[-1])
        F_on_grid = np.interp(grid, r_ref, F_ref)

        occ = np.asarray([[np.mean(state_of(xi[i], edges) == k) for k in range(3)]
                          for i in range(len(times))])
        pmf_t = d["diag_pmf"].astype(np.float64)
        pmf_steps = d["diag_steps"]
        qstar = np.zeros_like(occ)
        for i, t_step in enumerate(d["xi_steps"]):
            j = int(np.argmin(np.abs(pmf_steps - t_step)))
            qstar[i] = bias_aware_target(F_on_grid, pmf_t[j], grid, edges, beta)

        half = times >= 0.5 * T
        span_thr = DEFICIT_SPAN * T
        dt_frame = float(times[1] - times[0])
        n_walk = int(xi.shape[1])
        for k in range(3):
            lam_min[k] = min(lam_min[k], float((qstar[half] * n_walk)[:, k].min()))

        for dd in deficits:
            for k in range(3):
                oc = plant(occ[half], k, 1.0 - float(dd), args.scheme, qstar[half])
                below = oc[:, k] < DEFICIT_FRAC * qstar[half, k]
                if longest_run(below) * dt_frame >= span_thr:
                    fired[float(dd)][k] += 1
        n_seeds += 1

    print(f"[detection] scheme={args.scheme}  {n_seeds} seeds, span {DEFICIT_SPAN:.2f} T, "
          f"deficit threshold {DEFICIT_FRAC:.2f} Q*\n")
    print("planted   seeds firing (of %d), per state" % n_seeds)
    print("deficit    state 0   state 1   state 2")
    first = {k: None for k in range(3)}
    for dd in deficits:
        row = fired[float(dd)]
        for k in range(3):
            if first[k] is None and row[k] == n_seeds:
                first[k] = float(dd)
        print(f"  {dd:.0%}       {row[0]:>3d}/{n_seeds}     {row[1]:>3d}/{n_seeds}"
              f"     {row[2]:>3d}/{n_seeds}")

    print()
    for k in range(3):
        an = 2.0 / np.sqrt(lam_min[k])
        emp = first[k]
        print(f"state {k}: lambda = {lam_min[k]:7.1f}   analytic 1-checkpoint 2-sigma "
              f"= {an:.0%}   empirical all-seed firing = "
              + (f"{emp:.0%}" if emp is not None else f">{deficits[-1]:.0%}"))

    res = dict(n_seeds=n_seeds, deficits=[float(x) for x in deficits],
               fired={str(k): v for k, v in fired.items()},
               lambda_min={str(k): float(v) for k, v in lam_min.items()},
               analytic_mde={str(k): float(2.0 / np.sqrt(v)) for k, v in lam_min.items()},
               empirical_all_seed_firing={str(k): first[k] for k in first},
               deficit_frac=DEFICIT_FRAC, deficit_span=DEFICIT_SPAN,
               gate_c_min_lambda=GATE_C_MIN_LAMBDA)
    with open(os.path.join(out_dir, "detection.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n[done] -> {out_dir}/detection.json")


if __name__ == "__main__":
    main()
