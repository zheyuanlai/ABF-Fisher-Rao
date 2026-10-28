"""Is the barrier drift of the frozen-bias run the relaxation of the tagged guest's own phi-marginal?  Frozen bias (the
superseded pilot table), 4 x 1024 walkers, 800k steps, reinsertion every step; tagged IC at cage centres (the design)
vs uniform along x (the marginal already spread).  Barrier of each 100k-step window's own mean-force estimate."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import lta_ladder_numba as E  # noqa: E402
from lta_loading import engine as LE  # noqa: E402

table = np.load(os.path.join(ROOT, "results/lta_loading/superseded_v1/reference/table_pilot.npy"))
steps = 800_000
saves = list(range(0, steps + 1, 20_000))
out = {}
for ic in ("uniform_x", "cage"):
    cfg = LE.resolve_cfg(tagged_ic=ic, warmup_steps=1, burn_in_steps=0)
    b = LE.Batch([30400, 30401, 30402, 30403], "frozen", 1024, steps, cfg=cfg, table=table, save_steps=saves,
                 verbose=False)
    b.preequilibrate()
    res, meta = b.run(label=f"relax2 {ic}")
    acc = res["sv_acc"]
    rows = []
    for i in range(0, len(saves) - 5, 5):
        j = i + 5
        bars = []
        for r in range(4):
            Fr = E.histogram_pmf(E.mean_force((acc[j, r, 2] - acc[i, r, 2])[None], (acc[j, r, 3] - acc[i, r, 3])[None]))[0]
            bars.append(Fr[89:91].mean() - Fr[[0, 1, 178, 179]].mean())
        dM = (acc[j, :, 2] - acc[i, :, 2]).sum(0); dC = (acc[j, :, 3] - acc[i, :, 3]).sum(0)
        F = E.histogram_pmf(E.mean_force(dM[None], dC[None]))[0]
        occ_win = float(dC[80:100].sum() / dC.sum())
        rows.append(dict(window=[saves[i], saves[j]], barrier=float(F[89:91].mean() - F[[0, 1, 178, 179]].mean()),
                         barrier_sem=float(np.std(bars, ddof=1) / 2.0), window_occupancy=occ_win))
        print(f"   {ic} {saves[i]:7d}-{saves[j]:7d}: barrier {rows[-1]['barrier']:.3f} +- {rows[-1]['barrier_sem']:.3f}"
              f"  window-bin occupancy {occ_win:.3f}", flush=True)
    out[ic] = dict(rows=rows, run_s=meta["timers"]["run_s"])
with open(os.path.join(ROOT, "results/lta_loading/validation/env_relax2.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("RELAX2_DONE")
