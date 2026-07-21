"""Independent references for the distance CVs (R15/R14) via internal-coordinate
importance sampling (evaluation only).

R15/R14 depend on *all* internal coordinates (bonds, angles, both dihedrals), so unlike
the dihedral marginal they cannot be read off a fixed-dihedral FEP.  We importance-sample
the exact configurational Boltzmann measure and histogram the CV:

  proposal 'v4'      : bonds ~ r^2 e^{-beta V_bond}, angles ~ sin(theta) e^{-beta V_angle}
                       (exact inverse-CDF, existing samplers), dihedrals ~ e^{-beta V4}
                       (per-dihedral inverse-CDF).  Weight w = e^{-beta V_nb} (the 1-5 LJ;
                       butane has no LJ pair so w == 1 and 'v4' is exact by construction).
  proposal 'uniform' : dihedrals ~ Uniform[-pi,pi); weight w = e^{-beta (sum V4 + V_nb)}.
                       An INDEPENDENT second scheme (different variance) for cross-check.

Both are exact in the large-sample limit; the pentane 1-5 LJ is bounded (mildly attractive
minimum, repulsive core down-weighted to ~0), so the weights are well-conditioned (high
ESS).  Push samples through ``place_chain_internal`` -> Cartesian -> ``R = |q_j - q_i|``,
form the weighted reflected-KDE ``p_ref(R)``, ``F_ref = -beta^{-1} log p_ref + C``,
``F'_ref`` by finite difference, and the hidden-conditional ``p_ref(phi1,phi2 | R in I_k)``
on a coarse torus grid with 9-basin probabilities per R bin.  No reference ever enters the
deployable dynamics.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import geometry as geom
from . import interval as iv
from . import potentials as pot
from .reference import sample_bond_lengths, sample_bond_angles

EPS = 1.0e-12
PI = math.pi
TWO_PI = 2.0 * math.pi
GAUCHE = math.radians(116.57)


def sample_dihedral_v4(n, p: pot.AlkaneParams, gen, device, dtype=torch.float64, n_nodes=4000):
    """Sample phi ~ exp(-beta V4(phi)) on [-pi, pi) by inverse-CDF on a fine grid."""
    phi = torch.linspace(-PI, PI, n_nodes, device=device, dtype=dtype)
    logw = -p.beta * pot.V4(phi, p)
    w = torch.exp(logw - logw.max())
    cdf = torch.cumsum(w, 0); cdf = cdf / cdf[-1]
    u = torch.rand(n, generator=gen, device=device, dtype=dtype)
    idx = torch.searchsorted(cdf, u.clamp(0, 1 - 1e-12))
    return phi[idx.clamp(max=n_nodes - 1)]


def _basin_idx_np(phi, barrier):
    idx = np.zeros_like(phi, dtype=np.int64)
    idx[phi >= barrier] = 1
    idx[phi <= -barrier] = 2
    return idx


def distance_reference(params: pot.AlkaneParams, i, j, R_lo, R_hi, n_grid=256,
                       n_samples=400000, seed=314159, device="cpu", dtype=torch.float64,
                       bandwidth=0.04, proposal="v4", n_cond_bins=10, n_grid2=48,
                       chunk=200000):
    """Importance-sampling reference for ``R = |q_i - q_j|``.

    Returns a dict with grid/F/Fprime/p_ref, ESS, per-R-bin conditional torsion histograms
    and 9-basin probabilities, plus the raw weighted CV histogram for bootstrapping.
    """
    p = params
    A = p.n_atoms
    n_dih = A - 3
    dev = device
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    grid, dz = iv.interval_grid(n_grid, R_lo, R_hi, device=dev, dtype=dtype)
    Kr = iv.reflected_kernel_matrix(grid, bandwidth, R_lo, R_hi)
    g2, dz2 = None, None
    barrier = math.radians(61.6)

    # accumulators
    csum = torch.zeros(n_grid, device=dev, dtype=dtype)     # weighted CV histogram
    wsum = torch.zeros((), device=dev, dtype=torch.float64)
    w2sum = torch.zeros((), device=dev, dtype=torch.float64)
    # conditional: per R-bin joint (phi1,phi2) weighted histogram (coarse torus)
    from . import density2d as d2
    g1c, g2c, dphi1c, dphi2c = d2.torus_grid(n_grid2, n_grid2, device=dev, dtype=dtype)
    cond_edges = torch.linspace(R_lo, R_hi, n_cond_bins + 1, device=dev, dtype=dtype)
    cond_hist = torch.zeros(n_cond_bins, n_grid2, n_grid2, device=dev, dtype=dtype)
    cond_w = torch.zeros(n_cond_bins, device=dev, dtype=dtype)

    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        bonds = torch.stack([sample_bond_lengths(m, p, gen, dev, dtype) for _ in range(A - 1)], dim=1)
        angles = torch.stack([sample_bond_angles(m, p, gen, dev, dtype) for _ in range(A - 2)], dim=1)
        if proposal == "v4":
            dih = torch.stack([sample_dihedral_v4(m, p, gen, dev, dtype) for _ in range(n_dih)], dim=1)
        elif proposal == "uniform":
            dih = (torch.rand(m, n_dih, generator=gen, device=dev, dtype=dtype) * 2 - 1) * PI
        else:
            raise ValueError(proposal)
        q = geom.place_chain_internal(bonds, angles, dih, A, device=dev, dtype=dtype)
        R = torch.linalg.norm(q[:, j, :] - q[:, i, :], dim=-1)
        # importance weights
        logw = -p.beta * pot.nonbonded_energy(q, p)
        if proposal == "uniform":
            vtors = torch.zeros(m, device=dev, dtype=dtype)
            for d in range(n_dih):
                vtors = vtors + pot.V4(dih[:, d], p)
            logw = logw - p.beta * vtors
        w = torch.exp(logw - 0.0)                # keep absolute scale (bounded)
        csum += iv.bin_sum(R[None, :], w[None, :], n_grid, R_lo, R_hi)[0]
        wsum += w.sum().to(torch.float64)
        w2sum += (w.to(torch.float64) ** 2).sum()
        # conditional accumulation (pentane only meaningful; butane n_dih=1 -> phi2 col absent)
        if n_dih >= 2:
            bin_id = torch.bucketize(R, cond_edges) - 1
            bin_id = bin_id.clamp(0, n_cond_bins - 1)
            i1 = torch.floor((dih[:, 0] + PI) / dphi1c).long().clamp(0, n_grid2 - 1)
            i2 = torch.floor((dih[:, 1] + PI) / dphi2c).long().clamp(0, n_grid2 - 1)
            lin = bin_id * (n_grid2 * n_grid2) + i1 * n_grid2 + i2
            cond_hist.view(-1).scatter_add_(0, lin, w)
            cond_w.scatter_add_(0, bin_id, w)
        done += m

    p_ref = iv.normalize_density(iv.smooth(csum[None, :], Kr), dz)[0]
    F = -(1.0 / p.beta) * torch.log(p_ref.clamp_min(EPS))
    F = F - F.mean()
    # finite-difference derivative (non-periodic)
    Fp = torch.zeros_like(F)
    Fp[1:-1] = (F[2:] - F[:-2]) / (2 * dz)
    Fp[0] = (F[1] - F[0]) / dz; Fp[-1] = (F[-1] - F[-2]) / dz
    ess = float((wsum ** 2 / w2sum.clamp_min(EPS)).item())

    out = {
        "cv": f"R_{i}_{j}", "molecule": ("butane" if A == 4 else "pentane"),
        "beta": float(p.beta), "sigma": float(p.sigma), "proposal": proposal,
        "R_lo": float(R_lo), "R_hi": float(R_hi), "n_grid": n_grid, "bandwidth": bandwidth,
        "grid": grid.cpu().numpy(), "dz": float(dz),
        "p_ref": p_ref.cpu().numpy(), "F": F.cpu().numpy(), "Fprime": Fp.cpu().numpy(),
        "cv_hist_weighted": csum.cpu().numpy(),
        "ess": ess, "ess_frac": ess / n_samples, "n_samples": int(n_samples), "seed": int(seed),
    }
    if n_dih >= 2:
        # per-bin conditional densities + 9-basin probabilities
        ch = cond_hist  # (nb, g2, g2)
        cond_dens = ch / (ch.sum(dim=(-2, -1), keepdim=True).clamp_min(EPS) * dphi1c * dphi2c)
        out["cond_grid1"] = g1c.cpu().numpy(); out["cond_grid2"] = g2c.cpu().numpy()
        out["cond_dphi"] = float(dphi1c)
        out["cond_edges"] = cond_edges.cpu().numpy()
        out["cond_hist"] = ch.cpu().numpy()
        out["cond_dens"] = cond_dens.cpu().numpy()
        out["cond_weight"] = (cond_w / cond_w.sum().clamp_min(EPS)).cpu().numpy()
        # 9-basin probabilities per R bin
        gnp = g1c.cpu().numpy()
        T = np.abs(gnp) < barrier; Gp = gnp >= barrier; Gm = gnp <= -barrier
        masks = {"T": T, "Gp": Gp, "Gm": Gm}
        basin = np.zeros((n_cond_bins, 9))
        chn = ch.cpu().numpy()
        names = []
        for a, (n1, m1) in enumerate(masks.items()):
            for b, (n2, m2) in enumerate(masks.items()):
                names.append(f"{n1}_{n2}")
                tot = chn.sum(axis=(1, 2))
                sel = chn[:, m1][:, :, m2].sum(axis=(1, 2))
                basin[:, a * 3 + b] = sel / np.clip(tot, EPS, None)
        out["cond_basin_probs"] = basin
        out["cond_basin_names"] = np.array(names)
    return out


def determine_range(params: pot.AlkaneParams, i, j, n_samples=200000, seed=7,
                    device="cpu", dtype=torch.float64, q_lo=0.001, q_hi=0.999, margin=0.10):
    """Empirical [R_lo, R_hi] from unweighted-in-torsion support (for grid/wall setup).

    Uses the 'v4'-proposal geometry (bonds/angles/dihedrals) *without* the LJ weight, i.e.
    the widest physically reachable support, then pads by ``margin`` fraction of the range.
    """
    p = params; A = p.n_atoms; n_dih = A - 3
    gen = torch.Generator(device=device).manual_seed(int(seed))
    bonds = torch.stack([sample_bond_lengths(n_samples, p, gen, device, dtype) for _ in range(A - 1)], dim=1)
    angles = torch.stack([sample_bond_angles(n_samples, p, gen, device, dtype) for _ in range(A - 2)], dim=1)
    dih = (torch.rand(n_samples, n_dih, generator=gen, device=device, dtype=dtype) * 2 - 1) * PI
    q = geom.place_chain_internal(bonds, angles, dih, A, device=device, dtype=dtype)
    R = torch.linalg.norm(q[:, j, :] - q[:, i, :], dim=-1)
    lo = torch.quantile(R, q_lo).item(); hi = torch.quantile(R, q_hi).item()
    span = hi - lo
    return lo - margin * span, hi + margin * span, float(R.min()), float(R.max())
