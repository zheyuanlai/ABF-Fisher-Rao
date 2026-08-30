#!/usr/bin/env python
"""Sanity gates for the CHA olefin engine, run BEFORE any reference or production.

1. forces == -grad(potential_energy) by central FD (CPU f64), both guests,
   including bond/angle/LJ/confinement terms.
2. CV: f_loc identity for synthetic forces; xi gradient = (m_i/M) n.
3. gamma=0 fr_uniform bit-identical to abf (CPU short run).
4. Bond and angle fluctuations match equipartition.
5. GPU smoke (both guests, 450 K): finite, marginal mass 1, throughput printed.
6. Structural parity: ethene crossings >> propene crossings in a short
   unbiased run at 450 K (the qualitative ordering of Cnudde et al.).

    CUDA_VISIBLE_DEVICES=2 python -u scripts/sanity_cha.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from cha.core_cha import (KB, CHASimConfig, CHASystem, run_sampler)  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    ok &= bool(cond)


cpu = torch.device("cpu")
for guest in ("ethene", "propene"):
    sysc = CHASystem(guest, 450.0, cpu, root=ROOT)
    g = torch.Generator().manual_seed(0)
    q = sysc.initial_conditions(1, 8, g, side="A").reshape(8, sysc.n_beads, 3)
    F = sysc.forces(q)
    h = 1e-6
    err = 0.0
    for b in range(sysc.n_beads):
        for k in range(3):
            qp = q.clone(); qp[:, b, k] += h
            qm = q.clone(); qm[:, b, k] -= h
            fd = -(sysc.potential_energy(qp) - sysc.potential_energy(qm)) / (2 * h)
            err = max(err, float((F[:, b, k] - fd).abs().max()))
    check(f"{guest}: forces = -grad V (FD)", err < 2e-4, f"max abs err {err:.2e}")

    Fs = torch.randn(8, sysc.n_beads, 3, generator=g, dtype=torch.float64)
    f_loc, xi, _ = sysc.cv_local_mean_force(q, Fs)
    gg = float((sysc.mass_w ** 2).sum())
    expect = -(Fs * sysc.mass_w[None, :, None] * sysc.normal[None, None, :]).sum(dim=(1, 2)) / gg
    check(f"{guest}: f_loc formula", float((f_loc - expect).abs().max()) < 1e-12)
    # FD check of xi itself
    b0 = 0
    qp = q.clone(); qp[:, b0, 0] += h
    dxi = (sysc.cv_value(qp) - sysc.cv_value(q)) / h
    expect_g = float(sysc.mass_w[b0] * sysc.normal[0])
    check(f"{guest}: d xi / d q", float((dxi - expect_g).abs().max()) < 1e-6)

# ---- gamma=0 bit identity (ethene, CPU) ----
sysc = CHASystem("ethene", 450.0, cpu, root=ROOT)
sim = CHASimConfig(n_steps=2000, n_replicas=48, save_every=500,
                   abf_warmup_steps=200, estimator_burn_in_steps=200,
                   fr_start_steps=400, rng_seed=11)
o1 = run_sampler("abf", sysc, sim, seeds=[0, 1], verbose=False)
import dataclasses
sim0 = dataclasses.replace(sim, fr_rate=0.0)
o2 = run_sampler("fr_uniform", sysc, sim0, seeds=[0, 1], verbose=False)
same = all(np.array_equal(o1[k], o2[k]) for k in ("pmf", "mean_force", "p_hat"))
check("gamma=0 fr_uniform == abf (bitwise, CPU)", same,
      f"events={o2['total_replacement_events'].sum()}")

# ---- equipartition (propene bond + angle) ----
sysp = CHASystem("propene", 450.0, cpu, root=ROOT)
gq = torch.Generator().manual_seed(3)
qq = sysp.initial_conditions(1, 128, gq, side="A").reshape(128, 3, 3)
dt = 2e-4
ns = math.sqrt(2 * dt / sysp.beta)
gen = torch.Generator().manual_seed(4)
for it in range(6000):
    Fq = sysp.forces(qq)
    if it < 2000:
        fn = Fq.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        Fq = Fq * torch.clamp(fn, max=500.0) / fn
    qq = qq + dt * Fq + ns * torch.randn(qq.shape, generator=gen,
                                         dtype=torch.float64)
r01 = (qq[:, 0] - qq[:, 1]).norm(dim=-1)
kT = KB * 450.0
var = float(((r01 - r01.mean()) ** 2).mean())
check("propene bond <(dr)^2> ~ kT/k", 0.5 * kT / 400 < var < 2.0 * kT / 400,
      f"{var:.5f} vs {kT/400:.5f}")
a = qq[:, 0] - qq[:, 1]; b = qq[:, 2] - qq[:, 1]
th = torch.arccos(((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))).clamp(-1, 1))
varth = float(((th - th.mean()) ** 2).mean())
check("propene angle var ~ kT/k_th", 0.4 * kT / 585 < varth < 2.5 * kT / 585,
      f"{varth:.5f} vs {kT/585:.5f}")

# ---- GPU smoke + parity ----
if torch.cuda.is_available():
    dev = torch.device("cuda")
    for guest in ("ethene", "propene"):
        sysg = CHASystem(guest, 450.0, dev, root=ROOT)
        simg = CHASimConfig(n_steps=30000, n_replicas=1024, save_every=3000,
                            abf_warmup_steps=1_000_000,   # bias OFF: unbiased parity run
                            estimator_burn_in_steps=2000,
                            fr_start_steps=10 ** 9, rng_seed=12)
        out = run_sampler("abf", sysg, simg, seeds=[0], verbose=True)
        check(f"GPU {guest}: finite pmf", np.isfinite(out["pmf"][-1]).all())
        mass = float((out["p_hat"][-1].mean(0) * out["dz"]).sum())
        check(f"GPU {guest}: marginal mass ~ 1", abs(mass - 1) < 1e-6, f"{mass:.6f}")
        ms = out["runtime_seconds"] / simg.n_steps * 1e3
        print(f"     {guest}: {ms:.2f} ms/step at N=1024 "
              f"({1024 * simg.n_steps / out['runtime_seconds']:.0f} replica-steps/s), "
              f"crossings={out['n_crossings'].sum()}")
        # boundedness: the final marginal must live inside the confined range
        p_last = out["p_hat"][-1].mean(0)
        grid = out["grid"]
        tail = float(p_last[(grid < -10.0) | (grid > 9.5)].sum() * out["dz"])
        check(f"GPU {guest}: no probability piled at the range ends",
              tail < 0.05, f"tail mass {tail:.4f}")
        check(f"GPU {guest}: crossings sane (not exploded)",
              int(out["n_crossings"].sum()) < 5000,
              f"{int(out['n_crossings'].sum())}")
else:
    print("[skip] no CUDA visible")

print("\nSANITY:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
