"""Diagnostics of the V5 failure (2026-10-27): per-bin density ratio with walker-group errors, and the dU mismatch."""
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
from validate import make_batch, np_energy_forces  # noqa: E402


def part_b():
    out = {}
    for label, kw in (("short_preeq", dict(preeq_steps=500, preeq_sweeps=3)), ("default_preeq", {})):
        c2 = make_batch(4, 1, 256, seeds=[778], **kw)
        c2.preequilibrate()
        U = c2.energies().get()
        c2.forces(deposit=False)
        F = c2.F.get()
        rows = []
        worst = 0.0
        for t in range(20):
            U0 = c2.energies().get(); q0 = c2.q.get()
            c2.mc(10 ** 6 + t)
            U1 = c2.energies().get(); q1 = c2.q.get(); dU = c2.mc_dU.get()
            moved = np.nonzero(np.any(q0 != q1, axis=(1, 2, 3)))[0]
            for w in moved:
                rel = abs((U1[w] - U0[w]) - dU[w]) / max(1.0, abs(dU[w]))
                if rel > 1e-9 and len(rows) < 10:
                    m = int(np.nonzero(np.any(q0[w] != q1[w], axis=(1, 2)))[0][0])
                    Un0, _ = np_energy_forces(q0[w], c2.p)
                    Un1, _ = np_energy_forces(q1[w], c2.p)
                    rows.append(dict(t=t, w=int(w), m=m, U0=float(U0[w]), U1=float(U1[w]), dU_kernel=float(dU[w]),
                                     dU_total=float(U1[w] - U0[w]), U0_numpy=float(Un0), U1_numpy=float(Un1)))
                worst = max(worst, rel)
        out[label] = dict(U_max=float(U.max()), U_median=float(np.median(U)), n_U_gt_0=int((U > 0).sum()),
                          Fmax=float(np.abs(F).max()), worst_rel=worst, examples=rows)
    return out


def part_a(N=4096, steps=40_000, every=100, groups=16):
    ref = np.load(os.path.join(ROOT, "results/equal_budget_v2/references/lta_T150_reference.npz"))
    b = make_batch(2, 1, N, method="abf", eps_gg_K=0.0, preeq_steps=2000, preeq_sweeps=5, seeds=[777])
    b.preequilibrate()
    hs = np.zeros((groups, 180))
    gid = np.arange(N) % groups
    for k in range(steps):
        b.forces(deposit=False)
        if k % every == 0 and k >= 5000:
            q = b.q.get()
            x = 0.5 * (q[:, 1, 0, 0] + q[:, 1, 1, 0])
            ph = np.mod(2 * np.pi * x / b.p["a"] + np.pi, 2 * np.pi) - np.pi
            j = ((ph + np.pi) / (2 * np.pi / 180)).astype(int) % 180
            np.add.at(hs, (gid, j), 1)
        b.move(k, 0.0)
        if (k + 1) % 10 == 0:
            b.mc(k + 1)
    beta = b.p["beta"]
    pr = np.exp(-beta * (ref["F_ref"] - ref["F_ref"].min())); pr /= pr.sum()
    pe = hs.sum(0) / hs.sum()
    pg = hs / hs.sum(1, keepdims=True)
    lr = np.log(np.maximum(pe, 1e-300) / pr)
    lrg = np.log(np.maximum(pg, 1e-300) / pr)
    se = lrg.std(0, ddof=1) / np.sqrt(groups)
    expct = pr * hs.sum()
    sel = expct >= 400
    z = lr / np.maximum(se, 1e-12)
    worst = np.argsort(-np.abs(lr) * sel)[:8]
    return dict(n_judged=int(sel.sum()), max_abs_lr=float(np.abs(lr[sel]).max()),
                max_abs_z=float(np.abs(z[sel]).max()), rms_z=float(np.sqrt(np.mean(z[sel] ** 2))),
                worst_bins=[dict(j=int(j), phi=float(-np.pi + (j + 0.5) * 2 * np.pi / 180), lr=float(lr[j]),
                                 se=float(se[j]), expected=float(expct[j])) for j in worst],
                mean_lr_judged=float(lr[sel].mean()))


if __name__ == "__main__":
    res = dict(part_b=part_b())
    print(json.dumps(res["part_b"], indent=1), flush=True)
    res["part_a"] = part_a()
    print(json.dumps(res["part_a"], indent=1), flush=True)
    os.makedirs(os.path.join(ROOT, "results/lta_loading/validation"), exist_ok=True)
    with open(os.path.join(ROOT, "results/lta_loading/validation/debug_v5.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("DEBUG_DONE")
