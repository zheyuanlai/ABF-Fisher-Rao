"""Independent cross-check of the alanine reference FES using OPES.

This is a *genuinely different sampler*, not another estimator on the umbrella trajectories:
different bias (adaptive kernel-density reweighting vs static harmonic windows), different
trajectories, different initial conditions, different systematic errors.  Three estimators
applied to one set of umbrella trajectories share every systematic identically and cannot
validate anything.

Physics is byte-identical to the reference (same force field, masses, integrator, dt, gamma,
temperature, dtype, CV convention, grid parity), so the comparison isolates the sampler.

Acceptance: dG(phi>0 vs phi<=0) must agree with the umbrella/MBAR reference to ~0.3 kT.

Usage: CUDA_VISIBLE_DEVICES=7 python -u scripts/run_alanine_opes_crosscheck.py
       OMP_NUM_THREADS=64 python -u scripts/run_alanine_opes_crosscheck.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alanine.dynamics import BAOAB, KB, check_finite, make_seed_streams          # noqa: E402
from alanine.forcefield import TorchFF, extract_parameters, parameter_hash       # noqa: E402
from alanine.projection import require_odd_grid                                  # noqa: E402
from alanine.system import PHI_ATOMS, PSI_ATOMS, reference_minimum               # noqa: E402
from alkanes.opes_cv import BatchedTorusOPES, TorusOPESConfig                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_alanine_reference import dihedral_t                                     # noqa: E402

TWO_PI = 2.0 * math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/alanine/opes_crosscheck")
    ap.add_argument("--walkers", type=int, default=1024)
    ap.add_argument("--ps", type=float, default=400.0)
    ap.add_argument("--equil-ps", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--n-grid", type=int, default=97)
    ap.add_argument("--sigma", type=float, default=0.20)
    ap.add_argument("--barrier", type=float, default=40.0)      # kJ/mol
    ap.add_argument("--pace", type=int, default=500)
    ap.add_argument("--clip", type=float, default=400.0)        # kJ/mol/rad
    ap.add_argument("--rng-seed", type=int, default=20260802)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    require_odd_grid(a.n_grid)
    os.makedirs(a.out, exist_ok=True)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise SystemExit("CUDA_VISIBLE_DEVICES must be set explicitly (never GPUs 0-3)")
    dtype = torch.float64
    beta = 1.0 / (KB * a.temperature)
    kT = KB * a.temperature
    t0 = time.perf_counter()

    system, X0 = reference_minimum()
    P = extract_parameters(system)
    tff = TorchFF(P, device=dev, dtype=dtype)
    N = a.walkers
    x = torch.as_tensor(np.repeat(X0[None], N, 0), device=dev, dtype=dtype).contiguous()

    cfg = TorusOPESConfig(n_grid=a.n_grid, beta=beta, barrier=a.barrier, pace=a.pace,
                          sigma=a.sigma, bias_force_clip=a.clip, warmup_steps=5000)
    opes = BatchedTorusOPES(cfg, R=1, device=dev, dtype=dtype)

    def cv(q):
        return dihedral_t(q, PHI_ATOMS), dihedral_t(q, PSI_ATOMS)

    step_holder = {"s": 0}

    def total_force(q):
        with torch.enable_grad():
            qg = q.detach().requires_grad_(True)
            phi, psi = cv(qg)
            E = tff.energy(qg)
            g, = torch.autograd.grad(E.sum(), qg)
        phys = -g
        # OPES bias enters through the CV Jacobian: + f_a grad(phi_a)
        with torch.enable_grad():
            qg2 = q.detach().requires_grad_(True)
            p1, p2 = cv(qg2)
            b1, b2 = opes.bias_force_at(p1[None], p2[None], step=step_holder["s"])
            gb, = torch.autograd.grad((b1[0].detach() * p1 + b2[0].detach() * p2).sum(), qg2)
        return phys + gb

    integ = BAOAB(P["masses"], a.dt, a.gamma, a.temperature, total_force, device=dev, dtype=dtype)
    gen = make_seed_streams(a.rng_seed, 1, dev)[0]
    v = integ.maxwell((N, 22, 3), gen, dev, dtype)
    f = total_force(x)
    check_finite(0, ("q", x), ("f", f), dump_dir=a.out, tag="init")

    n_eq = int(a.equil_ps / a.dt)
    n_pr = int(a.ps / a.dt)
    print(f"OPES cross-check: device={dev} walkers={N} {a.ps} ps @dt={a.dt*1000:.1f} fs "
          f"grid={a.n_grid} sigma={a.sigma} barrier={a.barrier} kJ/mol | "
          f"param_hash={parameter_hash(P)}", flush=True)

    hist = torch.zeros(a.n_grid, a.n_grid, device=dev, dtype=dtype)
    logw_max = None
    n_acc = 0
    for s in range(n_eq + n_pr):
        step_holder["s"] = s
        x, v, f = integ.step(x, v, f, gen)
        p1, p2 = cv(x)
        if (s + 1) % cfg.pace == 0:
            opes.deposit(p1[None], p2[None])
        if s >= n_eq and (s + 1) % 100 == 0:
            # reweight to the unbiased ensemble: w = exp(+beta * A(z))
            from alkanes import density2d as d2
            A_at = d2.bilinear_interp2(opes._bias, opes.g1, opes.g2, opes.dz1, opes.dz2,
                                       p1[None], p2[None])[0]
            w = torch.exp(torch.clamp(beta * A_at, max=50.0))
            i = torch.floor((p1 + math.pi) / opes.dz1).long().clamp_(0, a.n_grid - 1)
            j = torch.floor((p2 + math.pi) / opes.dz2).long().clamp_(0, a.n_grid - 1)
            hist.view(-1).scatter_add_(0, i * a.n_grid + j, w)
            n_acc += 1
        if (s + 1) % max((n_eq + n_pr) // 10, 1) == 0:
            check_finite(s + 1, ("q", x), ("v", v), ("f", f), dump_dir=a.out, tag="opes")
            el = time.perf_counter() - t0
            frac_pos = float((p1 > 0).to(dtype).mean())
            print(f"  {s+1}/{n_eq+n_pr}  {(s+1)/el:.0f} steps/s  T={float(integ.kinetic_temperature(v)):.1f} K"
                  f"  frac(phi>0)_biased={frac_pos:.3f}  neff={opes.neff_frac()[0]:.3f}"
                  f"  eta {(n_eq+n_pr-s-1)/((s+1)/el)/60:.1f} min", flush=True)

    p = (hist / hist.sum()).cpu().numpy()
    g = np.degrees(-math.pi + (np.arange(a.n_grid) + 0.5) * (TWO_PI / a.n_grid))
    pos = g > 0
    P_pos = float(p[pos, :].sum())
    dG = -kT * math.log(max(P_pos, 1e-300) / max(1 - P_pos, 1e-300))
    F = np.where(p > 0, -kT * np.log(np.maximum(p, 1e-300)), np.inf)
    F = F - F[np.isfinite(F)].min()

    out = dict(sampler="OPES", walkers=N, ps=a.ps, n_grid=a.n_grid, sigma=a.sigma,
               barrier=a.barrier, pace=a.pace, dt_ps=a.dt, gamma=a.gamma,
               temperature=a.temperature, param_hash=parameter_hash(P),
               device=dev, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
               P_phi_pos=P_pos, dG_phi_pos_kJ=dG, dG_phi_pos_kT=dG / kT,
               neff_frac=float(opes.neff_frac()[0]), n_kernels=int(opes.n_kernels()[0]),
               wall_seconds=time.perf_counter() - t0)
    np.savez_compressed(os.path.join(a.out, "opes.npz"), F=F, p=p, meta=json.dumps(out))
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
