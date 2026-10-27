"""Which walkers of the reference pilot (seeds 30000-30003, N 1024, n 16) end pre-equilibration overlapped, and why.

Replays Batch.preequilibrate step by step (same counters, so the same trajectories) and reports, per bad walker:
when it went bad (after the sweeps or during the dynamics), the per-molecule energies, the closest bead pairs, and
the coordinates of the offending molecules.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from lta_loading import engine as LE  # noqa: E402


def fmax_per_walker(b):
    b.forces(deposit=False)
    return np.abs(b.F.get()).reshape(b.W, -1).max(1)


def describe(b, w):
    q = b.q.get()[w]
    p = b.p
    X = q.reshape(-1, 3)
    L = p["L"]
    d = X[:, None] - X[None]
    d -= L * np.rint(d / L)
    r = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    for m in range(b.n):
        r[2 * m, 2 * m + 1] = r[2 * m + 1, 2 * m] = np.inf
    i, j = np.unravel_index(np.argmin(r), r.shape)
    dO = X[:, None] - p["O"][None]
    dO -= L * np.rint(dO / L)
    rO = np.sqrt((dO ** 2).sum(-1)).min(1)
    bond = np.linalg.norm(q[:, 0] - q[:, 1], axis=1)
    com = 0.5 * (q[:, 0] + q[:, 1])
    corner = com - p["a"] * np.rint(com / p["a"])
    return dict(min_bead_bead=float(r[i, j]), pair=(int(i), int(j)), mols=(int(i // 2), int(j // 2)),
                min_bead_O=float(rO.min()), bead_min_O=int(rO.argmin()), bonds=bond.round(3).tolist(),
                com_dist_corner=np.sqrt((corner ** 2).sum(-1)).round(2).tolist(),
                max_abs_coord=float(np.abs(q).max()), q=q.round(3).tolist())


def main():
    cfg = LE.resolve_cfg()
    b = LE.Batch(list(range(30000, 30004)), "abf", 1024, 10, cfg=cfg, verbose=False)
    c = b.c
    rep = {}
    f0 = fmax_per_walker(b)
    rep["initial_bad"] = int((f0 > 1e4).sum())
    ctr = 0
    for s in range(int(c["preeq_sweeps"])):
        for m in range(1, b.n):
            b.mc(LE.PRE_MC_BASE + ctr, m_fixed=m)
            ctr += 1
        if s in (0, 4, 9, 19, 39):
            f = fmax_per_walker(b)
            rep[f"bad_after_sweep_{s + 1}"] = int((f > 1e4).sum())
    f = fmax_per_walker(b)
    bad_sweeps = np.nonzero(f > 1e4)[0]
    rep["bad_after_sweeps_walkers"] = bad_sweeps.tolist()
    rep["after_sweeps_detail"] = {int(w): describe(b, w) for w in bad_sweeps[:4]}
    went_bad = {}
    for k in range(int(c["preeq_steps"])):
        b.forces(deposit=False)
        fw = np.abs(b.F.get()).reshape(b.W, -1).max(1) if k % 50 == 0 else None
        if fw is not None:
            for w in np.nonzero(fw > 1e4)[0]:
                if int(w) not in went_bad:
                    went_bad[int(w)] = dict(step=k, fmax=float(fw[w]), detail=describe(b, w))
        b.move(LE.PRE_MD_BASE + k, 0.0, freeze_tagged=True)
        if (k + 1) % int(c["mc_every"]) == 0:
            b.mc(LE.PRE_MC_BASE + ctr)
            ctr += 1
    f = fmax_per_walker(b)
    rep["bad_after_dynamics"] = np.nonzero(f > 1e4)[0].tolist()
    rep["went_bad_during_dynamics"] = went_bad
    out = os.path.join(ROOT, "results/lta_loading/validation/debug_preeq.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1)
    slim = {k: v for k, v in rep.items() if k not in ("after_sweeps_detail", "went_bad_during_dynamics")}
    print(json.dumps(slim, indent=1))
    for w, v in list(went_bad.items())[:4]:
        d = dict(v["detail"]); d.pop("q")
        print("went bad", w, v["step"], v["fmax"], json.dumps(d))
    for w, v in list(rep["after_sweeps_detail"].items())[:4]:
        d = dict(v); d.pop("q")
        print("bad after sweeps", w, json.dumps(d))
    print("DEBUG_DONE")


if __name__ == "__main__":
    main()
