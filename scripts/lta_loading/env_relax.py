"""Environment relaxation diagnostic (before Amendment A4 freezes the burn-in): frozen bias (the pilot table), 4 x 1024
walkers, 400k steps, saves every 10k; reinsertion every step (A4) vs every 10 steps (the superseded design).  Reports,
per 50k-step window, the barrier of the window's own mean-force estimate and the mean potential energy."""
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
F_ref = np.load(os.path.join(ROOT, "results/lta_loading/superseded_v1/reference/reference.npz"))["F_ref"]
steps = 400_000
saves = list(range(0, steps + 1, 10_000))
out = {}
for mc_every in (1, 10):
    cfg = LE.resolve_cfg(mc_every=mc_every, warmup_steps=1, burn_in_steps=0)
    b = LE.Batch([30300, 30301, 30302, 30303], "frozen", 1024, steps, cfg=cfg, table=table, save_steps=saves,
                 verbose=False)
    b.preequilibrate()
    res, meta = b.run(label=f"relax mc{mc_every}")
    acc = res["sv_acc"]
    rows = []
    for i in range(0, len(saves) - 5, 5):
        j = i + 5
        dM = (acc[j, :, 2] - acc[i, :, 2]).sum(0); dC = (acc[j, :, 3] - acc[i, :, 3]).sum(0)
        F = E.histogram_pmf(E.mean_force(dM[None], dC[None]))[0]
        bars = []
        for r in range(4):
            Fr = E.histogram_pmf(E.mean_force((acc[j, r, 2] - acc[i, r, 2])[None], (acc[j, r, 3] - acc[i, r, 3])[None]))[0]
            bars.append(Fr[89:91].mean() - Fr[[0, 1, 178, 179]].mean())
        rows.append(dict(window=[saves[i], saves[j]], barrier=float(F[89:91].mean() - F[[0, 1, 178, 179]].mean()),
                         barrier_sem=float(np.std(bars, ddof=1) / 2.0), U=float(res["sv_U"][i:j + 1].mean())))
    out[f"mc_every_{mc_every}"] = dict(rows=rows, run_s=meta["timers"]["run_s"], preeq_s=meta["timers"]["preeq_s"],
                                       mc_acc=float(res["sv_i"][-1, :, 2].sum() / max(1, res["sv_i"][-1, :, 3].sum())),
                                       U_series=res["sv_U"].mean(1).tolist())
    print(f"mc_every={mc_every}: run {meta['timers']['run_s']:.0f}s preeq {meta['timers']['preeq_s']:.0f}s "
          f"acc {out[f'mc_every_{mc_every}']['mc_acc']:.4f}", flush=True)
    for r in rows:
        print(f"   {r['window']}: barrier {r['barrier']:.3f} +- {r['barrier_sem']:.3f}  U {r['U']:.2f}", flush=True)
os.makedirs(os.path.join(ROOT, "results/lta_loading/validation"), exist_ok=True)
with open(os.path.join(ROOT, "results/lta_loading/validation/env_relax.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("RELAX_DONE")
