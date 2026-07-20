"""Independent reference free energies for the alkanes (evaluation only).

Two families, both independent of the tested ABF/mFR/OPES estimators:

1. **Decoupled analytic** (LJ off): ``F(phi)=V4(phi)+C`` (butane) and
   ``F(phi1,phi2)=V4(phi1)+V4(phi2)+C`` (pentane).  Hard B0/P0 gates.

2. **Full-model internal-coordinate free-energy perturbation (QMC)**.  In
   bond-angle-dihedral internal coordinates the configurational measure factorises
   with a phi-independent Jacobian ``prod r_i^2 prod sin(theta_j)``, so at fixed
   dihedral(s)

       exp(-beta F(phi)) proportional to exp(-beta sum V4(phi))
                                        * < exp(-beta V_nonbonded) >_{r,theta},

   where the bonds are sampled from ``r^2 exp(-beta (k0/2)(r-d0)^2)`` and the angles
   from ``sin(theta) exp(-beta (k_theta/2)(theta-theta0)^2)`` (exact 1-D inverse-CDF
   sampling, capturing both the harmonic Boltzmann weight and the r^2 / sin(theta)
   Jacobian).  Hence

       F(phi) = sum V4(phi) - beta^{-1} log < exp(-beta V_nb) >_{r,theta} + C.

   For **butane** there is no LJ pair, so this returns ``V4+C`` -- a pipeline
   cross-check that the full sampler + geometric ABF force reproduce ``V4+C`` with
   fluctuating bonds/angles.  For **pentane** the 1-5 LJ produces the joint coupling
   ``F(phi1,phi2)``; the 1-D reference is obtained by marginalising ``exp(-beta F)``
   over phi2.

Convergence-ladder + bootstrap uncertainty are reported (cross-check).  A separate
constrained-dihedral MD thermodynamic-integration cross-check lives in
``alkanes.core`` (shares no code with the QMC estimator except the potential).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import geometry as geom
from . import periodic as per
from . import potentials as pot

EPS = 1.0e-12


# ---------------------------------------------------------------------------
# 1-D inverse-CDF samplers for the internal-coordinate Boltzmann+Jacobian measure
# ---------------------------------------------------------------------------
def sample_bond_lengths(n, p: pot.AlkaneParams, gen, device, dtype=torch.float64, n_nodes=2000):
    """Sample r ~ r^2 exp(-beta (k0/2)(r-d0)^2) by inverse-CDF on a fine grid."""
    lo = max(1e-3, p.d0 - 8.0 / math.sqrt(p.beta * p.k0))
    hi = p.d0 + 8.0 / math.sqrt(p.beta * p.k0)
    r = torch.linspace(lo, hi, n_nodes, device=device, dtype=dtype)
    logw = 2.0 * torch.log(r) - p.beta * 0.5 * p.k0 * (r - p.d0) ** 2
    w = torch.exp(logw - logw.max())
    cdf = torch.cumsum(w, 0)
    cdf = cdf / cdf[-1]
    u = torch.rand(n, generator=gen, device=device, dtype=dtype)
    idx = torch.searchsorted(cdf, u.clamp(0, 1 - 1e-12))
    return r[idx.clamp(max=n_nodes - 1)]


def sample_bond_angles(n, p: pot.AlkaneParams, gen, device, dtype=torch.float64, n_nodes=2000):
    """Sample theta ~ sin(theta) exp(-beta (k_theta/2)(theta-theta0)^2) by inverse-CDF."""
    lo = max(1e-3, p.theta0 - 8.0 / math.sqrt(p.beta * p.k_theta))
    hi = min(math.pi - 1e-3, p.theta0 + 8.0 / math.sqrt(p.beta * p.k_theta))
    th = torch.linspace(lo, hi, n_nodes, device=device, dtype=dtype)
    logw = torch.log(torch.sin(th)) - p.beta * 0.5 * p.k_theta * (th - p.theta0) ** 2
    w = torch.exp(logw - logw.max())
    cdf = torch.cumsum(w, 0)
    cdf = cdf / cdf[-1]
    u = torch.rand(n, generator=gen, device=device, dtype=dtype)
    idx = torch.searchsorted(cdf, u.clamp(0, 1 - 1e-12))
    return th[idx.clamp(max=n_nodes - 1)]


# ---------------------------------------------------------------------------
# Decoupled analytic references
# ---------------------------------------------------------------------------
def decoupled_butane(grid, p: pot.AlkaneParams):
    """F(phi)=V4(phi)-mean over the circular grid; also F'=V4'."""
    F = pot.V4(grid, p)
    F = F - F.mean()
    return {"grid": geom.to_numpy(grid) if hasattr(geom, "to_numpy") else grid.cpu().numpy(),
            "F": F.cpu().numpy(), "Fprime": pot.V4_prime(grid, p).cpu().numpy()}


def decoupled_pentane(grid1, grid2, p: pot.AlkaneParams):
    """Joint F(phi1,phi2)=V4(phi1)+V4(phi2)-C on the torus grid."""
    F = pot.V4(grid1[:, None], p) + pot.V4(grid2[None, :], p)
    F = F - F.mean()
    return {"grid1": grid1.cpu().numpy(), "grid2": grid2.cpu().numpy(), "F": F.cpu().numpy()}


# ---------------------------------------------------------------------------
# Full-model internal-coordinate FEP reference
# ---------------------------------------------------------------------------
def _nb_free_energy_correction(dih_grid, p, n_samples, gen, device, dtype=torch.float64,
                               node_chunk=256):
    """-beta^{-1} log < exp(-beta V_nb) > at each dihedral node in ``dih_grid``.

    ``dih_grid`` is ``(G, n_dih)``.  For speed and a low-noise reference surface the
    bond/angle samples are drawn ONCE and reused across all dihedral nodes (a valid
    common-random-numbers FEP; the average is over the same measure).  Nodes are
    processed in GPU-batched chunks.  The SEM uses the delta method for a
    log-mean-exp: ``SEM(corr) = std(w)/mean(w)/sqrt(n_eff)/beta`` with ``w=exp(-beta
    V_nb)`` -- adequate for the reference uncertainty band and far cheaper than a
    resampling bootstrap.  Returns ``(corr (G,), sem (G,))``.
    """
    G = dih_grid.shape[0]
    n_atoms = p.n_atoms
    dih_grid = dih_grid.to(device=device, dtype=dtype)
    # draw shared bond/angle samples once
    bonds = torch.stack([sample_bond_lengths(n_samples, p, gen, device, dtype)
                         for _ in range(n_atoms - 1)], dim=1)         # (S, n_bonds)
    angles = torch.stack([sample_bond_angles(n_samples, p, gen, device, dtype)
                          for _ in range(n_atoms - 2)], dim=1)        # (S, n_angles)
    corr = torch.zeros(G, device=device, dtype=dtype)
    sem = torch.zeros(G, device=device, dtype=dtype)
    for lo in range(0, G, node_chunk):
        hi = min(lo + node_chunk, G)
        nc = hi - lo
        b = bonds[None, :, :].expand(nc, -1, -1).reshape(nc * n_samples, -1)
        a = angles[None, :, :].expand(nc, -1, -1).reshape(nc * n_samples, -1)
        dih = dih_grid[lo:hi, None, :].expand(-1, n_samples, -1).reshape(nc * n_samples, -1)
        q = geom.place_chain_internal(b, a, dih, n_atoms, device=device, dtype=dtype)
        vnb = pot.nonbonded_energy(q, p).reshape(nc, n_samples)       # (nc, S)
        x = -p.beta * vnb
        m = x.max(dim=1, keepdim=True).values
        w = torch.exp(x - m)
        wm = w.mean(dim=1)
        lme = m.squeeze(1) + torch.log(wm.clamp_min(EPS))
        corr[lo:hi] = -(1.0 / p.beta) * lme
        # delta-method SEM of -beta^-1 log mean(w): std(w)/mean(w)/sqrt(S)/beta
        rel = (w.std(dim=1) / wm.clamp_min(EPS)) / math.sqrt(n_samples)
        sem[lo:hi] = rel / p.beta
    return corr, sem


def qmc_reference_butane(grid, p: pot.AlkaneParams, n_samples=20000, seed=314159, device="cpu"):
    """Full butane reference (== V4+C since no LJ); validates the FEP pipeline."""
    gen = torch.Generator(device=device).manual_seed(seed)
    dih = grid[:, None].to(device)
    corr, sem = _nb_free_energy_correction(dih, p, n_samples, gen, device, grid.dtype)
    F = pot.V4(grid.to(device), p) + corr
    F = F - F.mean()
    return {"grid": grid.cpu().numpy(), "F": F.cpu().numpy(),
            "nb_correction": corr.cpu().numpy(), "nb_correction_sem": sem.cpu().numpy(),
            "Fprime": pot.V4_prime(grid.to(device), p).cpu().numpy(),
            "n_samples": n_samples, "seed": seed}


def qmc_reference_pentane(grid1, grid2, p: pot.AlkaneParams, n_samples=20000,
                          seed=314159, device="cpu"):
    """Full pentane joint reference F(phi1,phi2) via internal-coordinate FEP (1-5 LJ)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    G1, G2 = grid1.numel(), grid2.numel()
    P1, P2 = torch.meshgrid(grid1.to(device), grid2.to(device), indexing="ij")
    dih = torch.stack([P1.reshape(-1), P2.reshape(-1)], dim=1)
    corr, sem = _nb_free_energy_correction(dih, p, n_samples, gen, device, grid1.dtype)
    V = pot.V4(P1.reshape(-1), p) + pot.V4(P2.reshape(-1), p)
    F = (V + corr).reshape(G1, G2)
    F = F - F.mean()
    return {"grid1": grid1.cpu().numpy(), "grid2": grid2.cpu().numpy(),
            "F": F.cpu().numpy(), "nb_correction": corr.reshape(G1, G2).cpu().numpy(),
            "nb_correction_sem": sem.reshape(G1, G2).cpu().numpy(),
            "n_samples": n_samples, "seed": seed}


def marginalize_joint_to_phi1(F_joint, grid1, grid2, beta):
    """1-D reference F_ref(phi1) = -beta^-1 log integral exp(-beta F(phi1,phi2)) dphi2."""
    F = torch.as_tensor(F_joint)
    g2 = torch.as_tensor(grid2)
    dphi2 = float(g2[1] - g2[0])
    logsum = torch.logsumexp(-beta * F, dim=1) + math.log(dphi2)
    F1 = -(1.0 / beta) * logsum
    F1 = F1 - F1.mean()
    return F1.cpu().numpy()


def conditional_phi2_given_phi1(F_joint, grid2, beta):
    """Reference conditional p(phi2 | phi1) for each phi1 row -> ``(G1, G2)`` densities."""
    F = torch.as_tensor(F_joint, dtype=torch.float64)
    g2 = torch.as_tensor(grid2, dtype=torch.float64)
    dphi2 = float(g2[1] - g2[0])
    logp = -beta * F
    logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
    p = torch.exp(logp) / dphi2
    return p.cpu().numpy()
