"""Corrected 2-D periodic umbrella + MBAR reference FES for Ace-Ala-Nme (ff14SB, vacuum).

Frozen first-study physics (identical for the reference and every later method arm):
  no HMR, no constraints, BAOAB, dt = 1 fs, gamma = 1 ps^-1, T = 300 K, float64,
  IUPAC (phi, psi), FES grid n_grid = 97 (ODD).

Seeding uses ONLY rigid rotations of the verified minimised L-alanine C7eq structure; the
whole-molecule NeRF rebuild never appears in the seeding path.  Every window/copy must pass the
geometry, chirality and finiteness gates before any dynamics.

The 16 copies of each window are independently thermalised (independent Maxwell velocities plus
an independent high-friction randomisation phase), so the block bootstrap can see the
copy-to-copy component instead of being blind to a single shared microscopic structure.

Usage:
  CUDA_VISIBLE_DEVICES=7 python -u scripts/run_alanine_reference.py --out results/alanine/reference
  ... --smoke      # 6x6 windows, 2 copies, short -- for wiring checks only
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

from alanine import reference as ref                                          # noqa: E402
from alanine.dynamics import BAOAB, KB, check_finite, make_seed_streams       # noqa: E402
from alanine.forcefield import TorchFF, extract_parameters, parameter_hash    # noqa: E402
from alanine.projection import require_odd_grid                              # noqa: E402
from alanine.system import (PHI_ATOMS, PSI_ATOMS, reference_minimum, relax_seeds,  # noqa: E402
                            seed_umbrella_lattice, validate_seed, window_centers)

TWO_PI = 2.0 * math.pi


def dihedral_t(x, idx):
    """IUPAC signed dihedral of ``x (B,22,3)`` for atom tuple ``idx``."""
    p0, p1, p2, p3 = (x[:, i] for i in idx)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    return torch.atan2((torch.linalg.cross(b1n, v, dim=-1) * w).sum(-1), (v * w).sum(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/alanine/reference")
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--copies", type=int, default=16)
    ap.add_argument("--kappa", type=float, default=200.0)
    ap.add_argument("--dt", type=float, default=0.001)          # ps  (1 fs)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--randomize-ps", type=float, default=20.0)
    ap.add_argument("--equil-ps", type=float, default=100.0)
    ap.add_argument("--prod-ps", type=float, default=1000.0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--n-grid", type=int, default=97)
    ap.add_argument("--rng-seed", type=int, default=20260801)
    ap.add_argument("--relax-steps", type=int, default=800)
    ap.add_argument("--relaxed-cv-tol-deg", type=float, default=None)
    ap.add_argument("--relaxed-angle-energy", type=float, default=90.0)
    ap.add_argument("--relaxed-angle-dev-deg", type=float, default=25.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.windows, a.copies = 6, 2
        a.randomize_ps, a.equil_ps, a.prod_ps = 1.0, 2.0, 5.0
        a.save_every, a.n_grid = 50, 35
        # smoke uses 60 deg spacing, so kappa must be relaxed to keep neighbour overlap:
        # sigma = sqrt(kT/kappa) should be comparable to half the spacing.
        a.kappa = 10.0

    # The relaxed seed sits at the minimum of V + restraint, displaced from the centre by
    # ~|grad F| / kappa.  With max|grad F| ~ 150 kJ/mol/rad that is ~26 deg at kappa=200 and
    # ~5x more at the smoke's kappa=10, so the bound must scale with kappa.  This is a sanity
    # diagnostic only -- MBAR does not require centred samples, and neighbour OVERLAP is the
    # real acceptance criterion (gates 3 and 4).
    if a.relaxed_cv_tol_deg is None:
        a.relaxed_cv_tol_deg = max(30.0, math.degrees(150.0 / a.kappa))

    require_odd_grid(a.n_grid)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    beta = 1.0 / (KB * a.temperature)
    t_start = time.perf_counter()

    print(f"device={dev} visible={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')} "
          f"dtype=float64", flush=True)
    if dev == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise SystemExit("CUDA_VISIBLE_DEVICES must be set explicitly (never GPUs 0-3)")

    # ---------------------------------------------------------------- system + seeds
    system, X0 = reference_minimum()
    P = extract_parameters(system)
    phash = parameter_hash(P)
    tff = TorchFF(P, device=dev, dtype=dtype)
    centers_np = window_centers(a.windows)
    K = len(centers_np)
    seeds_np = seed_umbrella_lattice(X0, centers_np)

    # Stage A -- validate the RIGID ROTATION itself.  A rotation changes only the dihedral, so
    # bonds, angles, planarity and chirality must be untouched and the CV must land exactly on
    # the requested centre.
    okA, repA = validate_seed(system, seeds_np, centers_np, cv_tol_deg=1e-4)
    print(f"seed gate A (rigid): {int(okA.sum())}/{K} pass | angleE max {repA['angle_energy'].max():.2f} "
          f"kJ/mol | maxdev {repA['max_angle_dev_deg'].max():.2f} deg | "
          f"CV err max {max(repA['dphi_deg'].max(), repA['dpsi_deg'].max()):.2e} deg | "
          f"chirality min {repA['chirality'].min():+.5f}", flush=True)
    if not okA.all():
        raise SystemExit(f"SEED GATE A FAILED for {int((~okA).sum())}/{K} windows -- aborting")

    # Stage B -- relieve STERIC clashes.  A rigid rotation preserves bonded geometry but can
    # drive non-bonded atoms together: 18.6% of rigid seeds sit above E_min+200 kJ/mol (peak
    # 2.3e6, forces to 4.7e8), and BAOAB started from those reaches ~1e27 K WITHOUT any NaN.
    tff_cpu = TorchFF(P, device="cpu", dtype=dtype)
    relaxed = relax_seeds(tff_cpu, torch.as_tensor(seeds_np), centers_np, kappa=a.kappa,
                          n_steps=a.relax_steps).numpy()
    E_seed = tff_cpu.energy(torch.as_tensor(relaxed)).numpy()
    F_seed = tff_cpu.forces(torch.as_tensor(relaxed)).norm(dim=-1).amax(-1).numpy()
    # The relaxed structure sits at the minimum of V + restraint, which is displaced from the
    # centre by ~ -(dF/dphi)/kappa (measured median 8.9 deg, max 24.6 deg at kappa=200). That is
    # correct physics, not an error -- MBAR does not require samples to sit at the centre -- so
    # the CV tolerance here is a sanity bound, not a constraint.
    # Stage B gates on what makes a seed *usable*: chirality, finite and bounded total energy,
    # bounded force, and the CV sanity bound.  The 15 deg bond-angle-deviation bound belongs to
    # Stage A, where it verifies the ROTATION (max 3.57 deg, huge margin).  After relaxing into
    # the target region, some deviation is genuine physical strain: the distribution is med 5.6,
    # p99 13.2, max 18.6 deg, and every outlier sits in the sterically crowded bridge region
    # near (+-22..37 deg, -+22..37 deg) with a perfectly normal total energy.  Rejecting those
    # would delete real, high-free-energy parts of the torus from the reference.  The bound here
    # is kept only to catch pathology -- for scale, the rejected NeRF path reached 114 deg.
    okB, repB = validate_seed(system, relaxed, centers_np, cv_tol_deg=a.relaxed_cv_tol_deg,
                              max_angle_energy=a.relaxed_angle_energy,
                              max_angle_dev_deg=a.relaxed_angle_dev_deg,
                              energy=E_seed, force_max=F_seed)
    print(f"seed gate B (relaxed): {int(okB.sum())}/{K} pass | E med {np.median(E_seed):.1f} "
          f"max {E_seed.max():.1f} (thresh {repB['energy_threshold']:.1f}) | |F|max {F_seed.max():.0f} | "
          f"CV drift med {np.median(np.maximum(repB['dphi_deg'], repB['dpsi_deg'])):.2f} "
          f"max {np.maximum(repB['dphi_deg'], repB['dpsi_deg']).max():.2f} deg | "
          f"failE {repB.get('n_fail_total_energy',0)} failF {repB.get('n_fail_force',0)} "
          f"failAngE {repB['n_fail_angle_energy']} failAngDev {repB['n_fail_angle_dev']} "
          f"failCV {repB['n_fail_cv']} failChir {repB['n_fail_chirality']} | "
          f"angle_dev med {np.median(repB['max_angle_dev_deg']):.2f} "
          f"p99 {np.percentile(repB['max_angle_dev_deg'], 99):.2f} "
          f"max {repB['max_angle_dev_deg'].max():.2f} deg", flush=True)
    if not okB.all():
        raise SystemExit(f"SEED GATE B FAILED for {int((~okB).sum())}/{K} windows -- aborting")
    seeds_np = relaxed
    ok, rep = okB, repB

    B = K * a.copies
    x = torch.as_tensor(np.repeat(seeds_np, a.copies, axis=0), device=dev, dtype=dtype).contiguous()
    centers = torch.as_tensor(np.repeat(centers_np, a.copies, axis=0), device=dev, dtype=dtype)
    kappa = float(a.kappa)

    def total_force(q):
        with torch.enable_grad():
            qg = q.detach().requires_grad_(True)
            phi = dihedral_t(qg, PHI_ATOMS)
            psi = dihedral_t(qg, PSI_ATOMS)
            E = tff.energy(qg) + ref.restraint_energy(phi, psi, centers, kappa)
            g, = torch.autograd.grad(E.sum(), qg)
        return -g

    integ = BAOAB(P["masses"], a.dt, a.gamma, a.temperature, total_force, device=dev, dtype=dtype)
    gens = make_seed_streams(a.rng_seed, 1, dev)
    gen = gens[0]
    v = integ.maxwell((B, 22, 3), gen, dev, dtype)            # independent per copy
    f = total_force(x)
    check_finite(0, ("q", x), ("f", f), dump_dir=a.out, tag="init")
    print(f"windows={K} copies={a.copies} batch={B} kappa={kappa} "
          f"sigma_restrained={ref.restrained_sigma_deg(kappa, a.temperature):.2f} deg "
          f"| param_hash={phash}", flush=True)

    def run(n_steps, label, gamma=None, save_every=0):
        nonlocal x, v, f
        if gamma is not None:
            integ.__init__(P["masses"], a.dt, gamma, a.temperature, total_force,
                           device=dev, dtype=dtype)
            f = total_force(x)
        out = []
        t0 = time.perf_counter()
        for s in range(n_steps):
            x, v, f = integ.step(x, v, f, gen)
            if save_every and (s + 1) % save_every == 0:
                out.append(torch.stack([dihedral_t(x, PHI_ATOMS),
                                        dihedral_t(x, PSI_ATOMS)], -1).to(torch.float32).cpu())
            if (s + 1) % max(n_steps // 5, 1) == 0:
                el = time.perf_counter() - t0
                check_finite(s + 1, ("q", x), ("v", v), ("f", f),
                             dump_dir=a.out, tag=label)
                print(f"  {label} {s+1}/{n_steps}  {(s+1)/el:.0f} steps/s  "
                      f"T={float(integ.kinetic_temperature(v)):.1f} K  "
                      f"eta {(n_steps-s-1)/((s+1)/el)/60:.1f} min", flush=True)
        return torch.stack(out, 1).numpy() if out else None

    n_rand = int(a.randomize_ps / a.dt)
    n_eq = int(a.equil_ps / a.dt)
    n_pr = int(a.prod_ps / a.dt)
    print(f"randomize {a.randomize_ps} ps @gamma=20 | equil {a.equil_ps} ps | "
          f"prod {a.prod_ps} ps @dt={a.dt*1000:.1f} fs", flush=True)
    run(n_rand, "randomize", gamma=20.0)          # decorrelate copies from the shared seed
    run(n_eq, "equil", gamma=a.gamma)             # discarded
    traj = run(n_pr, "prod", save_every=a.save_every)   # (B, frames, 2)
    print(f"production trajectory {traj.shape}", flush=True)

    # ---------------------------------------------------------------- MBAR
    frames = traj.shape[1]
    per_window = min(300, a.copies * frames)      # uncorrelated-ish subsample for the solve
    stride = max(1, (a.copies * frames) // per_window)
    tw = traj.reshape(K, a.copies * frames, 2)[:, ::stride][:, :per_window]
    phi = torch.as_tensor(tw[..., 0].reshape(-1), device=dev, dtype=dtype)
    psi = torch.as_tensor(tw[..., 1].reshape(-1), device=dev, dtype=dtype)
    N_k = torch.full((K,), tw.shape[1], device=dev, dtype=torch.long)
    cen = torch.as_tensor(centers_np, device=dev, dtype=dtype)
    print(f"MBAR: K={K} N={phi.numel()} ({tw.shape[1]}/window)", flush=True)
    t0 = time.perf_counter()
    fk, iters, resid = ref.mbar_solve(phi, psi, cen, kappa, beta, N_k, verbose=True)
    print(f"MBAR converged in {iters} iters, resid {resid:.2e}, {time.perf_counter()-t0:.1f}s",
          flush=True)

    logw = ref.mbar_log_weights(phi, psi, cen, kappa, beta, N_k, fk)
    F, counts, p = ref.fes_from_weights(phi, psi, logw, a.n_grid, beta)
    O = ref.overlap_matrix(phi, psi, cen, kappa, beta, N_k, fk)

    kT = KB * a.temperature
    Fn = F.cpu().numpy()
    finite = np.isfinite(Fn)
    mask8 = finite & (Fn - np.nanmin(Fn[finite]) <= 8 * kT)
    # window-level mask: is the window centre inside the 8 kT region?
    gi = np.floor((centers_np[:, 0] + math.pi) / (TWO_PI / a.n_grid)).astype(int) % a.n_grid
    gj = np.floor((centers_np[:, 1] + math.pi) / (TWO_PI / a.n_grid)).astype(int) % a.n_grid
    wmask = mask8[gi, gj]
    nn_all, st_all = ref.nn_overlap_stats(O.cpu(), a.windows)
    nn_ev, st_ev = ref.nn_overlap_stats(O.cpu(), a.windows, mask=wmask)

    imin = np.unravel_index(np.nanargmin(np.where(finite, Fn, np.inf)), Fn.shape)
    dz = TWO_PI / a.n_grid
    gmin = (math.degrees(-math.pi + (imin[0] + 0.5) * dz),
            math.degrees(-math.pi + (imin[1] + 0.5) * dz))

    meta = dict(
        param_hash=phash, windows=a.windows, copies=a.copies, kappa=kappa,
        dt_ps=a.dt, gamma=a.gamma, temperature=a.temperature, n_grid=a.n_grid,
        randomize_ps=a.randomize_ps, equil_ps=a.equil_ps, prod_ps=a.prod_ps,
        save_every=a.save_every, rng_seed=a.rng_seed, hmr=None, constraints=None,
        dtype="float64", convention="IUPAC", device=dev,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        forcefield="amber14/protein.ff14SB.xml",
        mbar_iters=iters, mbar_resid=resid, mbar_N=int(phi.numel()),
        mbar_per_window=int(tw.shape[1]),
        seed_gate_pass=int(ok.sum()), seed_gate_total=K,
        nn_overlap_all=st_all, nn_overlap_eval=st_ev,
        global_min_deg=gmin, F_range_kJ=float(np.nanmax(Fn[finite]) - np.nanmin(Fn[finite])),
        kT_kJ=kT, wall_seconds=time.perf_counter() - t_start,
    )
    np.savez_compressed(os.path.join(a.out, "reference.npz"),
                        F=Fn, counts=counts.cpu().numpy(), p=p.cpu().numpy(),
                        f_k=fk.cpu().numpy(), overlap=O.cpu().numpy(),
                        centers=centers_np, traj=traj, mask8=mask8,
                        meta=json.dumps(meta))
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print(json.dumps({k: v for k, v in meta.items() if k != "nn_overlap_all"},
                     indent=2, default=float), flush=True)
    print(f"TOTAL {time.perf_counter()-t_start:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
