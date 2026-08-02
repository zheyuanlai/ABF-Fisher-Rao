#!/usr/bin/env python
"""Name the pilot's 2-D (phi, chi1) regions by the 3-D states that occupy them.

The V3 screen measures discovery and establishment for regions ``C_k`` of the SELECTED CV --
those are what ABF and a marginal FR score can see.  The physics, though, lives in the 3-D states
``B_j`` of the S1 map.  Without an explicit correspondence a V3 table reports deficits for
regions with no stated physical meaning, and "region B2 is starved" cannot be turned into a
sentence about valine.

The mapping is measured, not assumed: every S1 exploration frame is assigned to its 3-D state and
to its pilot region, and the joint table is reported.  A region receiving frames from two 3-D
states that are NOT the known B3/B5 pair would contradict the distinguishability gate, so this is
also a consistency check on that gate rather than only a labelling convenience.

Usage
-----
    python scripts/valine_map_regions_to_states.py \
        --state-map results/valine/state_map --pilot results/valine/pilot_reference
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alanine.basins import BasinMap                                          # noqa: E402
from valine.states import to_cell                                           # noqa: E402

KB = 0.008314462618


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-map", default="results/valine/state_map")
    ap.add_argument("--pilot", default="results/valine/pilot_reference")
    ap.add_argument("--ceiling-kT", type=float, default=8.0)
    ap.add_argument("--min-prominence-kT", type=float, default=1.0)
    ap.add_argument("--max-basins", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=300.0)
    a = ap.parse_args()
    kT = KB * a.temperature

    pf = np.load(os.path.join(a.pilot, "pilot_reference.npz"), allow_pickle=True)
    F = pf["F"]
    mask = np.isfinite(F) & (F < a.ceiling_kT * kT)
    bm = BasinMap(F, mask, kT, ceiling_kT=a.ceiling_kT,
                  min_prominence_kT=a.min_prominence_kT, max_basins=a.max_basins,
                  name_hints=())
    pops = bm.population(F)
    n = F.shape[0]

    st = np.load(os.path.join(a.state_map, "states.npz"), allow_pickle=True)
    ex = np.load(os.path.join(a.state_map, "explore.npz"), allow_pickle=True)
    th = ex["theta"]
    lab3 = st["frame_labels"]
    centres3 = np.degrees(st["centres"])
    K3 = centres3.shape[0]

    keep = lab3 >= 0
    ang = th.reshape(-1, 3)[keep.reshape(-1)].astype(np.float64)
    s3 = lab3.reshape(-1)[keep.reshape(-1)].astype(np.int64)
    reg = bm.label[to_cell(ang[:, 0], n), to_cell(ang[:, 2], n)]

    K2 = len(bm.names)
    J = np.zeros((K2, K3), dtype=np.int64)
    for k in range(K2):
        m = reg == k
        if m.any():
            J[k] = np.bincount(s3[m], minlength=K3)
    outside = int((reg < 0).sum())

    def rot(c):
        return "t" if abs(abs(c) - 180) < 60 else ("g+" if c > 0 else "g-")

    print(f"pilot regions {K2}, S1 states {K3}, "
          f"{len(reg) - outside:,} frames mapped, {outside:,} outside every region")
    print(f"\n{'region':>7s} {'phi':>7s} {'chi1':>7s} {'pilotP':>8s}  composition (3-D states)")
    rows = []
    for k in range(K2):
        c = bm.centres_deg[k]
        tot = J[k].sum()
        if tot == 0:
            print(f"{bm.names[k]:>7s} {c[0]:7.1f} {c[1]:7.1f} {pops[bm.names[k]]:8.4f}  "
                  f"(no exploration frames)")
            rows.append(dict(region=bm.names[k], centre_deg=list(c),
                             pilot_population=pops[bm.names[k]], composition={}, purity=None))
            continue
        order = np.argsort(-J[k])
        parts = [f"B{int(j)}({rot(centres3[j][2])},phi{'>' if centres3[j][0] > 0 else '<'}0)"
                 f" {J[k][j] / tot:.2f}" for j in order[:3] if J[k][j] > 0]
        print(f"{bm.names[k]:>7s} {c[0]:7.1f} {c[1]:7.1f} {pops[bm.names[k]]:8.4f}  "
              + "  ".join(parts))
        rows.append(dict(region=bm.names[k], centre_deg=list(c),
                         pilot_population=pops[bm.names[k]],
                         composition={f"B{int(j)}": float(J[k][j] / tot)
                                      for j in order if J[k][j] > 0},
                         dominant_state=f"B{int(order[0])}",
                         purity=float(J[k][order[0]] / tot),
                         rotamer=rot(centres3[order[0]][2]),
                         backbone="phi>0" if centres3[order[0]][0] > 0 else "phi<0"))

    pure = [r for r in rows if r["purity"] is not None]
    worst = min(pure, key=lambda r: r["purity"]) if pure else None
    if worst:
        print(f"\nleast pure region: {worst['region']} at {worst['purity']:.2f} "
              f"(dominant {worst['dominant_state']})")
    # split states: one 3-D state spread across several regions is fine and expected for the
    # B3/B5 pair, which is kinetically one state; anywhere else it means the CV cuts a state.
    print("\n3-D state -> regions:")
    for j in range(K3):
        tot = J[:, j].sum()
        if tot == 0:
            continue
        sh = {bm.names[k]: J[k, j] / tot for k in range(K2) if J[k, j] / max(tot, 1) > 0.05}
        c = centres3[j]
        print(f"  B{j} ({rot(c[2])}, phi{'>' if c[0] > 0 else '<'}0): "
              + ", ".join(f"{k} {v:.2f}" for k, v in sorted(sh.items(), key=lambda t: -t[1])))

    out = os.path.join(a.pilot, "region_state_map.json")
    with open(out, "w") as fh:
        json.dump(dict(regions=rows, joint_counts=J.tolist(),
                       state_centres_deg=centres3.round(1).tolist(),
                       frames_outside_regions=outside,
                       pilot=a.pilot, state_map=a.state_map), fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
