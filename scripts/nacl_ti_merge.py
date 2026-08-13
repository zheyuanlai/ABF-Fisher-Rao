"""Merge r-point-split TI outputs into one ti_final.npz for the analysis.

Splitting the reference by r-POINT keeps every point's full build x family x replica set in one
process, so each point's joint retirement criterion is evaluated over exactly the population it
is defined over.  The halves are therefore independent and concatenate without any statistic
being recomputed across the seam -- which is precisely why a build split would not merge.

Refuses to merge unless the union covers the full grid exactly once: a missing or duplicated
r-point would silently change the population every downstream number is defined over.

Usage:
    python scripts/nacl_ti_merge.py --parts results/nacl/ti_torch results/nacl/ti_torch_B \
                                    --out results/nacl/ti_merged
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nacl import system as nsys                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", default="results/nacl/ti_merged")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    box = json.load(open(nsys.REPO / "results/nacl/box/box_manifest.json"))
    r_hi = float(box["finite_size_gate"]["R_hi_nm"])
    r_full = np.round(np.arange(nsys.R_LO_NM, r_hi + 1e-9, 0.02), 4)

    recs, fbar, fcnt, ysum, ycnt = [], [], [], [], []
    retired = {}
    for part in args.parts:
        z = np.load(os.path.join(part, "ti_final.npz"))
        recs.append(z["recs"]); fbar.append(z["fbar"]); fcnt.append(z["fcnt"])
        ysum.append(z["ysum"]); ycnt.append(z["ycnt"])
        rs = np.unique(z["recs"][:, 0])
        for i, r in enumerate(rs):
            retired[float(r)] = float(z["retired_at"][i]) if i < len(z["retired_at"]) else np.nan
        print(f"  {part}: {len(z['recs'])} trajectories, {len(rs)} r-points "
              f"[{rs.min():.2f}, {rs.max():.2f}]")

    recs = np.concatenate(recs); fbar = np.concatenate(fbar); fcnt = np.concatenate(fcnt)
    ysum = np.concatenate(ysum); ycnt = np.concatenate(ycnt)

    got = np.unique(recs[:, 0])
    missing = sorted(set(np.round(r_full, 4)) - set(np.round(got, 4)))
    if missing:
        raise SystemExit(f"merge covers {len(got)}/{len(r_full)} r-points; missing {missing}")
    counts = {float(r): int((recs[:, 0] == r).sum()) for r in got}
    if len(set(counts.values())) != 1:
        raise SystemExit(f"uneven trajectory counts per r-point: "
                         f"{sorted(set(counts.values()))} -- a point was split or duplicated")

    order = np.lexsort((recs[:, 3], recs[:, 2], recs[:, 1], recs[:, 0]))
    retired_at = np.array([retired.get(float(r), np.nan) for r in r_full])
    np.savez_compressed(os.path.join(args.out, "ti_final.npz"),
                        recs=recs[order], fbar=fbar[order], fcnt=fcnt[order],
                        ysum=ysum[order], ycnt=ycnt[order], retired_at=retired_at)
    man = dict(stage="nacl_ti_merged", parts=list(args.parts),
               n_trajectories=int(len(recs)), n_r_points=int(len(got)),
               per_point=int(next(iter(set(counts.values())))),
               note="r-point split: each point's full build x family x replica set stayed in "
                    "one process, so no joint statistic crosses the seam")
    json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(f"merged {len(recs)} trajectories over {len(got)} r-points "
          f"({next(iter(set(counts.values())))} each) -> {args.out}")


if __name__ == "__main__":
    main()
