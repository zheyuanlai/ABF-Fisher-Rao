"""Wasserstein (horizontal OT) reallocation along a DISTANCE collective variable
``R = |q_j - q_i|`` + constrained fibre repair -- the ``ot=`` option of
:func:`alkanes.core_dist.run_sampler_dist` (pentane R15 / butane R14).

ADDITIVE: nothing here runs unless the sampler is handed a :class:`DistOTConfig`; every
accepted path stays byte-identical.  The 1-D transport map is the WCA one
(:mod:`wca_ot_repair`); the LIFT and the REPAIR are the distance-CV versions:

  * LIFT (Euclidean, symmetric): with ``e = (q_j - q_i)/R`` and ``dR = R' - R``,
        q_i' = q_i - (dR/2) e,   q_j' = q_j + (dR/2) e,
    the other atoms untouched, so ``|q_j' - q_i'| = R'`` exactly and the centre of mass is
    preserved.  The bonded neighbours of ``i`` and ``j`` see a small instantaneous bond /
    angle strain (the fibre lag this system injects).
  * REPAIR (projected constrained Euler--Maruyama at fixed R'): every atom takes one
    Langevin step of the outer dynamics (same dt, beta, force, force clip), then the pair
    ``(i, j)`` is re-projected to ``R'`` by the same symmetric move.  Because
    ``|grad R|^2 = 2`` is constant, the constrained (surface) measure equals the exact
    conditional measure ``p(. | R = R')`` (co-area factor constant), so this operator samples
    the fibre law up to the O(dt) discretisation bias the outer dynamics share.  Own RNG,
    nothing deposited, every inner step charged.

One OT opportunity (on the FR schedule ``fr_start_steps + k fr_every``):
  1. ``R_i = R(q_i)``; sort; uniform quantiles ``u_i = lo + (i - 1/2)(hi - lo)/N`` on the
     target domain ``[lo, hi]``;
  2. rank-matched displacement interpolation with a per-event cap,
        ``R_i' = R_i + clip(alpha (u_(rank i) - R_i), -dR_max, dR_max)``;
  3. LIFT; 4. REPAIR ``m_repair`` steps for EVERY walker (M3 parity: R, F+R and T+R receive
     literally the same fibre treatment; ``alpha = 0`` with ``m_repair > 0`` is the R arm).

Target domain: the uniform target lives on ``[lo, hi]``.  :func:`lj_forbidden_radius` gives
the reference-free lower edge used by the pentane preregistration -- the smallest R at which
the 1-5 Lennard-Jones pair energy ALONE already exceeds the study's ``thermal_delta`` (a lower
bound on the true free-energy cost, derived from the potential, no reference consulted).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from . import geometry as geom

EPS = 1.0e-12


@dataclass(frozen=True)
class DistOTConfig:
    alpha: float = 0.0                 # displacement fraction toward the uniform quantiles; 0 => no transport
    dR_max: float = 0.018              # per-event displacement cap (2 grid bins of the R15 grid)
    m_repair: int = 0                  # projected constrained steps per opportunity for EVERY walker; 0 => none
    domain: Optional[Tuple[float, float]] = None   # uniform target domain; None => (R_lo, R_hi)
    inner_seed_offset: int = 8_000_000
    min_move: float = 1.0e-9           # |dR| below this is "not moved" for the deposit-free diagnostics
    cond_snapshot: bool = True         # store the cumulative conditional torsion histogram at every save


def lj_forbidden_radius(params, delta):
    """Smallest ``R`` with ``4 eps [(sigma/R)^12 - (sigma/R)^6] <= delta`` (pair energy alone)."""
    x = 0.5 * (1.0 + math.sqrt(1.0 + float(delta) / float(params.epsilon)))    # x = (sigma/R)^6
    return float(params.sigma) / x ** (1.0 / 6.0)


def uniform_quantiles(n, lo, hi, device, dtype):
    i = torch.arange(n, device=device, dtype=dtype)
    return lo + (i + 0.5) * (hi - lo) / float(n)


def ot_displacement_batched(R, alpha, dR_max, u):
    """``R' = R + clip(alpha (u_rank - R), +-dR_max)`` along the last dim of ``R`` (B, N);
    ranks preserved, identity for ``alpha = 0``, no RNG."""
    zs, order = torch.sort(R, dim=-1, stable=True)
    step = torch.clamp(alpha * (u.unsqueeze(0) - zs), -dR_max, dR_max)
    return torch.empty_like(R).scatter_(-1, order, zs + step)


def lift_to_R(q, R_new, i, j):
    """Symmetric Euclidean lift of atoms ``i, j`` along their axis to ``|q_j - q_i| = R_new``.
    ``q`` (..., n_atoms, 3), ``R_new`` (...).  COM preserved; other atoms untouched."""
    r = q[..., j, :] - q[..., i, :]
    Rc = torch.linalg.norm(r, dim=-1).clamp_min(EPS)
    e = r / Rc[..., None]
    d = (0.5 * (R_new - Rc))[..., None] * e
    out = q.clone()
    out[..., i, :] = q[..., i, :] - d
    out[..., j, :] = q[..., j, :] + d
    return out


def projected_relax(q, R_fixed, m, params, dt, beta, gen, force_fn, cv, record_first=False):
    """``m`` projected constrained Euler--Maruyama steps at fixed ``R_fixed`` for every walker.
    ``q`` (B, n_atoms, 3), ``R_fixed`` (B,).  If ``record_first``, returns the estimator's
    local-mean-force sample at the state handed in (the lifted state) -- computed from the first
    inner step's force, so it costs nothing extra.  Returns ``(q_new, f_first or None)``."""
    noise_scale = math.sqrt(2.0 * dt / beta)
    f_first = None
    for k in range(int(m)):
        F = force_fn(q, params)
        if k == 0 and record_first:
            f_first, _, _ = cv.local_mean_force(q, F, beta)
        noise = torch.randn(q.shape, generator=gen, device=q.device, dtype=q.dtype)
        q = lift_to_R(q + dt * F + noise_scale * noise, R_fixed, cv.i, cv.j)
        q = geom.remove_com(q)
    return q, f_first


def torsion_cond_indices(q, Rv, cond_edges, n_rbins, n_grid2, dphi):
    """Linear index into a flattened ``(n_rbins, n_grid2, n_grid2)`` (R-bin, phi1, phi2) histogram
    for pentane configurations ``q`` (B, 5, 3) with CV values ``Rv`` (B,)."""
    PI = math.pi
    phi1 = geom.signed_dihedral(q, 0, 1, 2, 3)
    phi2 = geom.signed_dihedral(q, 1, 2, 3, 4)
    bin_id = (torch.bucketize(Rv, cond_edges) - 1).clamp(0, n_rbins - 1)
    i1 = torch.floor((phi1 + PI) / dphi).long().clamp(0, n_grid2 - 1)
    i2 = torch.floor((phi2 + PI) / dphi).long().clamp(0, n_grid2 - 1)
    return bin_id * (n_grid2 * n_grid2) + i1 * n_grid2 + i2, phi1, phi2


# ---------------------------------------------------------------------------
# Compiled force (opt-in): torch.compile fuses the autograd force of the 5-atom chain into a
# handful of kernels (13.8 -> 1.3 ms per evaluation measured on 16k walkers, agreement 1e-10,
# run-to-run bitwise).  Callers may run under torch.no_grad(); the wrapper re-enables grad.
# ---------------------------------------------------------------------------
def compiled_forces(dynamic=True):
    from . import potentials as pot
    energy_c = torch.compile(pot.total_energy, dynamic=dynamic)

    def forces(q, p):
        with torch.enable_grad():
            qg = q.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(energy_c(qg, p).sum(), qg, create_graph=False)
        f = -g
        if p.force_clip and p.force_clip > 0:
            norm = torch.linalg.norm(f, dim=-1, keepdim=True)
            f = f * torch.clamp(p.force_clip / norm.clamp_min(EPS), max=1.0)
        return f.detach()
    return forces


def eager_forces(q, p):
    from . import potentials as pot
    with torch.enable_grad():
        return pot.forces(q, p)
