"""Fast OPES cross-check for the alanine reference.

Why the first version took hours for a 22-atom molecule: at B=1024 the workload is entirely
**kernel-launch bound**, and it wasted launches three ways.

  1. ``total_force`` ran **two separate autograd passes** per step -- one for the physical
     energy, one for the bias -- doubling both graph construction and kernel count.
  2. Nothing was compiled, so every one of the ~150 small ops per step paid full dispatch and
     launch overhead.
  3. The batch was far too small. At 22 atoms an H200 is idle at B=1024; the reference measured
     7.9 ms/step at **B=9216**, i.e. 9x the walkers for the same wall-clock. Walkers are
     essentially free, and with N walkers sharing one adaptive bias the bias converges ~N times
     faster **in simulated time**, which is what the run length is actually gated on.

Fixes here:

  * **One fused autograd pass.** The OPES bias is folded into the energy as
    ``E = V(q) + A(phi(q), psi(q))`` with a differentiable bilinear interpolation of the bias
    grid, so a single ``autograd.grad`` yields the total force. This is exactly equivalent to
    the previous ``phys + bias_force . grad(CV)`` construction (the applied potential is ``+A``
    and the applied force ``-grad(V+A)``), but at half the passes.
  * **``torch.compile`` with the bias grid as an explicit tensor argument**, so the 500-step
    deposit updates its *values* without triggering recompilation.
  * **Large batch** (default 8192) and a correspondingly shorter physical run.

Correctness is unchanged: same force field, integrator, dt, gamma, temperature, dtype, CV
convention and odd grid as the reference, and the same OPES reweighting ``w = exp(+beta A)``.

Usage: CUDA_VISIBLE_DEVICES=7 python -u scripts/run_alanine_opes_fast.py
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

from alanine.dynamics import KB, check_finite                                    # noqa: E402
from alanine.forcefield import TorchFF, extract_parameters, parameter_hash       # noqa: E402
from alanine.projection import require_odd_grid                                  # noqa: E402
from alanine.system import PHI_ATOMS, PSI_ATOMS, reference_minimum               # noqa: E402
from alkanes import density2d as d2                                              # noqa: E402
from alkanes.opes_cv import BatchedTorusOPES, TorusOPESConfig                    # noqa: E402

TWO_PI = 2.0 * math.pi


def dihedral_t(x, idx):
    p0, p1, p2, p3 = (x[:, i] for i in idx)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    return torch.atan2((torch.linalg.cross(b1n, v, dim=-1) * w).sum(-1), (v * w).sum(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/alanine/opes_fast")
    ap.add_argument("--walkers", type=int, default=8192)
    ap.add_argument("--ps", type=float, default=250.0)
    ap.add_argument("--bias-equil-ps", type=float, default=120.0)
    ap.add_argument("--block-ps", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--n-grid", type=int, default=97)
    ap.add_argument("--sigma", type=float, default=0.20)
    ap.add_argument("--barrier", type=float, default=40.0)
    ap.add_argument("--pace", type=int, default=500)
    ap.add_argument("--rng-seed", type=int, default=20260804)
    ap.add_argument("--compile", default="reduce-overhead")
    ap.add_argument("--init", default="dispersed", choices=("c7eq", "dispersed"),
                    help="dispersed = tile the validated 24x24 rigid-rotation umbrella seeds over "
                         "the torus. Starting every walker at C7eq leaves the BIASED distribution "
                         "filling outward for a long time, and the reweighting identity only holds "
                         "in stationarity -- measured as dG blocks drifting 2.34 -> 1.89 kT with "
                         "P(phi>0) rising 0.088 -> 0.131.")
    ap.add_argument("--bench-only", action="store_true")
    a = ap.parse_args()

    require_odd_grid(a.n_grid)
    os.makedirs(a.out, exist_ok=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise SystemExit("CUDA_VISIBLE_DEVICES must be set explicitly")
    dev, dtype = "cuda", torch.float64
    beta, kT = 1.0 / (KB * a.temperature), KB * a.temperature
    torch.manual_seed(a.rng_seed)
    torch.cuda.manual_seed_all(a.rng_seed)
    t0 = time.perf_counter()

    system, X0 = reference_minimum()
    P = extract_parameters(system)
    tff = TorchFF(P, device=dev, dtype=dtype)
    N = a.walkers
    if a.init == "dispersed":
        from alanine.system import relax_seeds, seed_umbrella_lattice, window_centers
        cen = window_centers(24)
        seeds = seed_umbrella_lattice(X0, cen)
        seeds = relax_seeds(TorchFF(P, device="cpu", dtype=dtype),
                            torch.as_tensor(seeds), cen, kappa=200.0, n_steps=400).numpy()
        reps = int(np.ceil(N / len(seeds)))
        x = torch.as_tensor(np.tile(seeds, (reps, 1, 1))[:N], device=dev, dtype=dtype).contiguous()
        print(f"init=dispersed: {len(seeds)} validated torus seeds tiled to {N} walkers", flush=True)
    else:
        x = torch.as_tensor(np.repeat(X0[None], N, 0), device=dev, dtype=dtype).contiguous()

    cfg = TorusOPESConfig(n_grid=a.n_grid, beta=beta, barrier=a.barrier, pace=a.pace,
                          sigma=a.sigma, bias_force_clip=1e30, warmup_steps=0)
    opes = BatchedTorusOPES(cfg, R=1, device=dev, dtype=dtype)
    g1, g2, dz1, dz2 = opes.g1, opes.g2, opes.dz1, opes.dz2

    m = tff.masses.reshape(-1, 1)
    dt, gamma = a.dt, a.gamma
    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt(1.0 - c1 * c1)
    sig = math.sqrt(kT) / m.sqrt()

    def total_force(q, A):
        """One fused pass: E = V(q) + A(phi(q), psi(q)); force = -grad E."""
        with torch.enable_grad():
            qg = q.detach().requires_grad_(True)
            phi = dihedral_t(qg, PHI_ATOMS)
            psi = dihedral_t(qg, PSI_ATOMS)
            Eb = d2.bilinear_interp2(A[None], g1, g2, dz1, dz2, phi[None], psi[None])[0]
            g, = torch.autograd.grad((tff.energy(qg) + Eb).sum(), qg)
        return -g

    def step(x, v, f, A):
        v = v + (0.5 * dt) * f / m
        x = x + (0.5 * dt) * v
        v = c1 * v + c2 * sig * torch.randn_like(v)
        x = x + (0.5 * dt) * v
        f = total_force(x, A)
        v = v + (0.5 * dt) * f / m
        return x, v, f

    stepf = torch.compile(step, mode=a.compile) if a.compile != "none" else step

    A = opes._bias[0].contiguous()
    v = torch.randn_like(x) * sig
    f = total_force(x, A)
    for _ in range(30):                      # warm up / compile
        x, v, f = stepf(x, v, f, A)
    torch.cuda.synchronize()
    tb = time.perf_counter()
    for _ in range(100):
        x, v, f = stepf(x, v, f, A)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - tb) / 100 * 1e3
    n_tot = int(a.ps / dt)
    print(f"walkers={N} compile={a.compile} -> {ms:.2f} ms/step ; {a.ps} ps = {n_tot} steps "
          f"= {ms*n_tot/1000/60:.1f} min ; peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
          flush=True)
    if a.bench_only:
        return

    gdeg = np.degrees(-math.pi + (np.arange(a.n_grid) + 0.5) * (TWO_PI / a.n_grid))
    pos = torch.as_tensor(gdeg > 0, device=dev)
    n_beq, n_blk = int(a.bias_equil_ps / dt), int(a.block_ps / dt)
    hist = torch.zeros(a.n_grid, a.n_grid, device=dev, dtype=dtype)
    blk = torch.zeros_like(hist)
    blocks = []

    def dG_of(h):
        p = h / h.sum().clamp_min(1e-30)
        Pp = float(p[pos, :].sum())
        return Pp, -math.log(max(Pp, 1e-300) / max(1 - Pp, 1e-300))

    for s in range(n_tot):
        x, v, f = stepf(x, v, f, A)
        if (s + 1) % a.pace == 0:
            p1, p2 = dihedral_t(x, PHI_ATOMS), dihedral_t(x, PSI_ATOMS)
            opes.deposit(p1[None], p2[None])
            A.copy_(opes._bias[0])           # in-place: values change, no recompile
        if s >= n_beq and (s + 1) % 50 == 0:
            p1, p2 = dihedral_t(x, PHI_ATOMS), dihedral_t(x, PSI_ATOMS)
            A_at = d2.bilinear_interp2(A[None], g1, g2, dz1, dz2, p1[None], p2[None])[0]
            w = torch.exp(torch.clamp(beta * A_at, max=50.0))
            i = torch.floor((p1 + math.pi) / dz1).long().clamp_(0, a.n_grid - 1)
            j = torch.floor((p2 + math.pi) / dz2).long().clamp_(0, a.n_grid - 1)
            lin = i * a.n_grid + j
            hist.view(-1).scatter_add_(0, lin, w)
            blk.view(-1).scatter_add_(0, lin, w)
        if s >= n_beq and (s + 1 - n_beq) % n_blk == 0 and float(blk.sum()) > 0:
            Pb, dGb = dG_of(blk)
            Pc, dGc = dG_of(hist)
            blocks.append(dict(t_ps=(s + 1) * dt, P_block=Pb, dG_block_kT=dGb,
                               P_cum=Pc, dG_cum_kT=dGc))
            print(f"    t={(s+1)*dt:7.1f} ps  dG_block={dGb:6.3f} kT  dG_cum={dGc:6.3f} kT  "
                  f"P_blk={Pb:.4f}  neff={opes.neff_frac()[0]:.3f}", flush=True)
            blk.zero_()
        if (s + 1) % max(n_tot // 10, 1) == 0:
            check_finite(s + 1, ("q", x), ("v", v), ("f", f), dump_dir=a.out, tag="opes")
            el = time.perf_counter() - t0
            print(f"  {s+1}/{n_tot}  {(s+1)/el:.0f} steps/s  "
                  f"frac(phi>0)={float((dihedral_t(x, PHI_ATOMS) > 0).to(dtype).mean()):.3f}  "
                  f"eta {(n_tot-s-1)/((s+1)/el)/60:.1f} min", flush=True)

    Pp, dG = dG_of(hist)
    p = (hist / hist.sum()).cpu().numpy()
    F = np.where(p > 0, -kT * np.log(np.maximum(p, 1e-300)), np.inf)
    F = F - F[np.isfinite(F)].min()
    tail = [b["dG_block_kT"] for b in blocks[len(blocks) // 2:]]
    out = dict(sampler="OPES-fast", walkers=N, ps=a.ps, bias_equil_ps=a.bias_equil_ps,
               n_grid=a.n_grid, sigma=a.sigma, barrier=a.barrier, pace=a.pace, dt_ps=dt,
               gamma=gamma, temperature=a.temperature, param_hash=parameter_hash(P),
               ms_per_step=ms, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
               P_phi_pos=Pp, dG_phi_pos_kT=dG, dG_phi_pos_kJ=dG * kT,
               dG_tail_mean_kT=float(np.mean(tail)) if tail else None,
               dG_tail_sd_kT=float(np.std(tail)) if tail else None,
               neff_frac=float(opes.neff_frac()[0]), n_kernels=int(opes.n_kernels()[0]),
               blocks=blocks, wall_seconds=time.perf_counter() - t0)
    np.savez_compressed(os.path.join(a.out, "opes.npz"), F=F, p=p, meta=json.dumps(out))
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "blocks"}, indent=2, default=float))


if __name__ == "__main__":
    main()
