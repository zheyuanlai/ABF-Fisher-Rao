#!/usr/bin/env python
"""Sanity gates for the LTA engine, run BEFORE the reference or any production.

1. forces == -grad(potential_energy) by central finite differences (CPU, f64).
2. CV formula: f_loc identity for a synthetic force field; grad(phi) layout.
3. gamma=0 fr_uniform is bit-identical to abf (same seed, CPU) -- this engine's
   _birth_death consumes no RNG when the rate is zero, so the property holds.
4. Bond-length fluctuation matches equipartition: <(r-r0)^2> ~ kT/k.
5. Short GPU smoke: no NaN, marginal develops, crossings counted.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/sanity_lta.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from lta.core_lta import (KB, LTAParams, LTASimConfig, LTASystem,  # noqa: E402
                          run_sampler)

PI = math.pi
ok = True


def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    ok &= bool(cond)


# ---------- 1. finite-difference forces (CPU) ----------
cpu = torch.device("cpu")
params = LTAParams()
sys_cpu = LTASystem(params, cpu, root=ROOT)
g = torch.Generator().manual_seed(0)
q = sys_cpu.initial_conditions(1, 16, g).reshape(16, 2, 3)
F = sys_cpu.forces(q)
h = 1e-6
err = 0.0
for b in range(2):
    for k in range(3):
        qp = q.clone(); qp[:, b, k] += h
        qm = q.clone(); qm[:, b, k] -= h
        fd = -(sys_cpu.potential_energy(qp) - sys_cpu.potential_energy(qm)) / (2 * h)
        err = max(err, float((F[:, b, k] - fd).abs().max()))
check("forces = -grad V (central FD)", err < 1e-5, f"max abs err {err:.2e}")

# ---------- 2. CV formula ----------
Fsyn = torch.randn(16, 2, 3, generator=g, dtype=torch.float64)
f_loc, phi, grad_full = sys_cpu.cv_local_mean_force(q, Fsyn)
expect = -(sys_cpu.a / (2 * PI)) * (Fsyn[:, 0, 0] + Fsyn[:, 1, 0])
check("f_loc = -(a/2pi)(F1x+F2x)", float((f_loc - expect).abs().max()) < 1e-12)
gp = torch.zeros_like(q); gp[:, :, 0] = PI / sys_cpu.a
check("grad(phi) layout", torch.equal(grad_full, gp))
x = q[..., 0].mean(-1)
check("phi wrap", float((phi - ((2 * PI / sys_cpu.a * x + PI) % (2 * PI) - PI))
                        .abs().max()) < 1e-12)

# ---------- 3. gamma=0 bit-identity (CPU, short) ----------
sim = LTASimConfig(n_steps=3000, n_replicas=64, save_every=500,
                   abf_warmup_steps=200, estimator_burn_in_steps=200,
                   fr_start_steps=400, rng_seed=11)
out_abf = run_sampler("abf", sys_cpu, sim, seeds=[0, 1], verbose=False)
sim0 = LTASimConfig(**{**sim.__dict__, "fr_rate": 0.0})
out_uni0 = run_sampler("fr_uniform", sys_cpu, sim0, seeds=[0, 1], verbose=False)
same = all(np.array_equal(out_abf[k], out_uni0[k]) for k in ("pmf", "mean_force", "p_hat"))
check("gamma=0 fr_uniform == abf (bitwise, CPU)", same,
      f"events={out_uni0['total_replacement_events'].sum()}")

# ---------- 4. bond equipartition (CPU, unbiased-ish short run) ----------
qq = sys_cpu.initial_conditions(1, 256, torch.Generator().manual_seed(3)).reshape(256, 2, 3)
noise_scale = math.sqrt(2 * sim.dt / params.beta)
gen = torch.Generator().manual_seed(4)
for _ in range(4000):
    Ff = sys_cpu.forces(qq)
    qq = qq + sim.dt * Ff + noise_scale * torch.randn(qq.shape, generator=gen,
                                                      dtype=torch.float64)
r = (qq[:, 0] - qq[:, 1]).norm(dim=-1)
var = float(((r - r.mean()) ** 2).mean())
kT_over_k = KB * params.temperature / params.k_bond
check("bond <(dr)^2> ~ kT/k", 0.5 * kT_over_k < var < 2.0 * kT_over_k,
      f"measured {var:.4f} vs kT/k {kT_over_k:.4f}")

# ---------- 5. GPU smoke ----------
if torch.cuda.is_available():
    dev = torch.device("cuda")
    sys_gpu = LTASystem(params, dev, root=ROOT)
    simg = LTASimConfig(n_steps=20000, n_replicas=512, save_every=2000,
                        abf_warmup_steps=2000, estimator_burn_in_steps=2000,
                        fr_start_steps=4000, rng_seed=12)
    out = run_sampler("fr_uniform", sys_gpu, simg, seeds=[0, 1], verbose=True)
    check("GPU: finite pmf", np.isfinite(out["pmf"][-1]).all())
    check("GPU: events fired", out["total_replacement_events"].sum() > 0)
    check("GPU: marginal mass ~ 1",
          abs(float((out["p_hat"][-1].mean(0) * out["dphi"]).sum()) - 1.0) < 1e-6)
    print(f"     crossings so far: {out['n_cage_crossings'].sum()}")
else:
    print("[skip] no CUDA visible")

print("\nSANITY:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
