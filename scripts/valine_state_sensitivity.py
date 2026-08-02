#!/usr/bin/env python
"""How much of the state map is a property of the molecule, and how much of the knobs?

The screening plan's AMBIGUOUS branch is exactly this: "if the result depends strongly on
clustering boundaries or the coarse reference, improve the pilot reference locally and repeat
V3."  That branch cannot be evaluated without measuring the dependence, so this script re-runs
the clustering over a grid of settings on the ALREADY-SAMPLED exploration cloud.  No new
dynamics: the sampling is fixed, only the analysis moves.

The number that matters is not whether the state count is perfectly constant -- it will not be,
since prominence merging is a threshold on a continuum -- but whether the states that carry the
V3 verdict survive.  A state that appears only at one setting cannot support a claim about
under-establishment.

Usage
-----
    python scripts/valine_state_sensitivity.py --state-map results/valine/state_map
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from valine.states import StateMap, torus_distance                           # noqa: E402


def match_states(ref_centres, centres, tol_rad=0.7):
    """Greedy nearest-centre matching on the torus; returns how many reference states survive."""
    if len(centres) == 0 or len(ref_centres) == 0:
        return 0, []
    used, hits = set(), []
    for i, c in enumerate(ref_centres):
        d = torus_distance(np.asarray(centres), np.asarray(c)[None, :])
        order = np.argsort(d)
        j = next((int(k) for k in order if int(k) not in used and d[k] <= tol_rad), None)
        if j is not None:
            used.add(j)
            hits.append((i, j, float(d[j])))
    return len(hits), hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-map", default="results/valine/state_map")
    ap.add_argument("--cells", type=int, nargs="+", default=[30, 36, 44])
    ap.add_argument("--prominence", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--ceiling", type=float, nargs="+", default=[5.0, 6.0, 8.0])
    ap.add_argument("--subsample", type=int, default=2, help="frame stride, to keep this cheap")
    a = ap.parse_args()

    ex = np.load(os.path.join(a.state_map, "explore.npz"), allow_pickle=True)
    base = json.load(open(os.path.join(a.state_map, "meta.json")))["clustering"]
    theta = ex["theta"][:, ::a.subsample]
    pts = theta.reshape(-1, 3).astype(np.float64)
    w = np.full(pts.shape[0], 1.0 / theta.shape[1])
    print(f"{pts.shape[0]:,} samples (stride {a.subsample});  baseline "
          f"cells={base['cells']} prominence={base['min_prominence']} "
          f"ceiling={base['ceiling']}")

    ref = StateMap(pts, n=base["cells"], weights=w, smooth_cells=base["smooth_cells"],
                   min_prominence_kT=base["min_prominence"], ceiling_kT=base["ceiling"], kT=1.0)
    print(f"baseline at this stride: {ref.n_states} states\n")
    print(f"{'cells':>6s} {'promin':>7s} {'ceil':>6s} {'states':>7s} {'matched':>8s} {'sec':>6s}")
    rows = []
    for n, pr, ce in itertools.product(a.cells, a.prominence, a.ceiling):
        t0 = time.time()
        sm = StateMap(pts, n=n, weights=w, smooth_cells=base["smooth_cells"],
                      min_prominence_kT=pr, ceiling_kT=ce, kT=1.0)
        nm, _ = match_states(ref.centres, sm.centres)
        el = time.time() - t0
        print(f"{n:6d} {pr:7.2f} {ce:6.1f} {sm.n_states:7d} "
              f"{nm:4d}/{ref.n_states:<3d} {el:6.0f}", flush=True)
        rows.append(dict(cells=n, prominence=pr, ceiling=ce, n_states=sm.n_states,
                         matched_baseline=nm, centres_deg=np.degrees(sm.centres).round(1).tolist(),
                         seconds=el))

    ns = np.array([r["n_states"] for r in rows])
    frac = np.array([r["matched_baseline"] / max(ref.n_states, 1) for r in rows])
    print(f"\nstate count over {len(rows)} settings: min {ns.min()} median "
          f"{int(np.median(ns))} max {ns.max()}")
    print(f"fraction of baseline states recovered: min {frac.min():.2f} "
          f"median {np.median(frac):.2f}")
    stable = frac.min() >= 0.8
    print(f"-> the decomposition is {'STABLE' if stable else 'SENSITIVE'} to the clustering knobs"
          + ("" if stable else "; a V3 verdict resting on a state that appears at only one "
                              "setting is not supportable"))

    out = os.path.join(a.state_map, "state_sensitivity.json")
    with open(out, "w") as fh:
        json.dump(dict(baseline=base, baseline_n_states=ref.n_states,
                       baseline_centres_deg=np.degrees(ref.centres).round(1).tolist(),
                       subsample=a.subsample, rows=rows,
                       n_states_min=int(ns.min()), n_states_max=int(ns.max()),
                       recovered_fraction_min=float(frac.min()),
                       stable=bool(stable)), fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
