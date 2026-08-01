"""Corrected 2-D periodic umbrella reference FES on T^2, with MBAR.

Restraint (per CV, summed over the two dihedrals):

    u_k(z) = kappa * [ (1 - cos(phi - c_phi)) + (1 - cos(psi - c_psi)) ]

chosen over the wrapped harmonic ``0.5 k d_T^2``, which has a ``2 k pi`` force discontinuity at
the antipode.  Near the centre ``1 - cos(x) ~ x^2/2``, so the effective harmonic stiffness is
``k_eff = kappa`` and the restrained width is ``sigma = sqrt(1/(beta kappa))``.

**Rank-4 factorisation.**  Expanding the cosine,

    u_k(z_n) = 2 kappa - kappa * [ cos c_k cos phi_n + sin c_k sin phi_n
                                   + cos d_k cos psi_n + sin d_k sin psi_n ]
             = 2 kappa - A_k . B_n,     A_k, B_n in R^4,

so the ``K x N`` reduced-potential matrix is never materialised: it is a ``(K,4) @ (4,N)``
product evaluated in chunks.  With K = 576 this removes the memory wall entirely (a dense
``u_kn`` at production N would be tens of GiB).  The constant ``2 kappa`` and any ``beta U_0``
cancel in MBAR and are dropped.

**Anderson acceleration.**  The plain self-consistent MBAR iteration converges at the spectral
gap of the fixed-point map (rho ~ 0.98 here), needing ~1200 iterations; warm-starting does not
help because the rate, not the starting point, is the bottleneck.  Anderson/DIIS with history 8
reaches the same solution in ~80 iterations.
"""
from __future__ import annotations

import math

import numpy as np
import torch

KB = 0.008314462618          # kJ/mol/K
TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- restraint
def restraint_energy(phi, psi, centers, kappa):
    """``kappa[(1-cos(phi-c_phi)) + (1-cos(psi-c_psi))]``; ``centers`` is ``(B,2)``."""
    return kappa * ((1.0 - torch.cos(phi - centers[:, 0]))
                    + (1.0 - torch.cos(psi - centers[:, 1])))


def restrained_sigma_deg(kappa, temperature=300.0):
    """Harmonic-equivalent restrained standard deviation, in degrees."""
    return math.degrees(math.sqrt(KB * temperature / kappa))


def _basis(phi, psi):
    """``B_n = (cos phi, sin phi, cos psi, sin psi)`` -> ``(4, N)``."""
    return torch.stack([torch.cos(phi), torch.sin(phi), torch.cos(psi), torch.sin(psi)], 0)


def _coeff(centers, kappa, beta):
    """``A_k = beta kappa (cos c, sin c, cos d, sin d)`` -> ``(K, 4)``."""
    c = centers
    return beta * kappa * torch.stack([torch.cos(c[:, 0]), torch.sin(c[:, 0]),
                                       torch.cos(c[:, 1]), torch.sin(c[:, 1])], 1)


# --------------------------------------------------------------------------- MBAR
def mbar_solve(phi, psi, centers, kappa, beta, N_k, tol=1e-8, max_iter=5000,
               anderson_m=8, chunk=200_000, verbose=False):
    """Solve MBAR for the free energies ``f_k`` (in kT) of the umbrella windows.

    ``phi, psi`` are the pooled samples ``(N,)``; ``N_k`` the per-window sample counts.
    Returns ``(f_k, n_iter, final_residual)``.  Raises if it does not converge.
    """
    dev, dt = phi.device, phi.dtype
    A = _coeff(centers, kappa, beta)                       # (K,4)
    B = _basis(phi, psi)                                   # (4,N)
    K, N = A.shape[0], B.shape[1]
    logN = torch.log(N_k.to(dt).clamp_min(1e-30))
    f = torch.zeros(K, device=dev, dtype=dt)

    def sweep(f):
        """One self-consistent update, chunked over samples."""
        acc = torch.full((K,), -float("inf"), device=dev, dtype=dt)
        for s in range(0, N, chunk):
            b = B[:, s:s + chunk]
            u = -(A @ b)                                   # (K,n)  [const dropped]
            # log denominator per sample: logsumexp_l (logN_l + f_l - u_ln)
            ld = torch.logsumexp((logN + f)[:, None] - u, dim=0)      # (n,)
            acc = torch.logaddexp(acc, torch.logsumexp(-u - ld[None, :], dim=1))
        return -acc

    hist_g, hist_x = [], []
    for it in range(1, max_iter + 1):
        g = sweep(f)
        g = g - g[0]
        resid = (g - f).abs().max()
        if resid < tol:
            return f, it, float(resid)
        # Anderson/DIIS on the fixed-point residual
        hist_x.append(f.clone())
        hist_g.append(g - f)
        if len(hist_x) > anderson_m:
            hist_x.pop(0)
            hist_g.pop(0)
        m = len(hist_x)
        if m == 1:
            f = g
        else:
            G = torch.stack(hist_g, 1)                     # (K,m)
            try:
                # min ||G a|| s.t. sum a = 1
                GtG = G.T @ G
                GtG = GtG + (1e-10 * torch.diagonal(GtG).mean().clamp_min(1e-30)) * torch.eye(m, device=dev, dtype=dt)
                ones = torch.ones(m, 1, device=dev, dtype=dt)
                a = torch.linalg.solve(GtG, ones)
                a = a / a.sum()
                f = (torch.stack(hist_x, 1) + G) @ a.squeeze(1)
            except Exception:                              # noqa: BLE001
                f = g
        f = f - f[0]
        if verbose and it % 20 == 0:
            print(f"    mbar it {it:4d} resid {float(resid):.3e}", flush=True)
    raise RuntimeError(
        f"MBAR did not converge: residual {float(resid):.3e} > {tol:.1e} after {max_iter} "
        "iterations. The usual cause is insufficient neighbour overlap -- check that the "
        "restrained width sqrt(1/(beta kappa)) is comparable to the window spacing.")


def mbar_log_weights(phi, psi, centers, kappa, beta, N_k, f, chunk=200_000):
    """Unnormalised log importance weights of every pooled sample under the unbiased ensemble."""
    A = _coeff(centers, kappa, beta)
    B = _basis(phi, psi)
    logN = torch.log(N_k.to(phi.dtype).clamp_min(1e-30))
    out = torch.empty(B.shape[1], device=phi.device, dtype=phi.dtype)
    for s in range(0, B.shape[1], chunk):
        u = -(A @ B[:, s:s + chunk])
        out[s:s + chunk] = -torch.logsumexp((logN + f)[:, None] - u, dim=0)
    return out


def fes_from_weights(phi, psi, logw, n_grid, beta):
    """Bin the MBAR-weighted samples onto an ``n_grid x n_grid`` torus and return ``F`` in kJ/mol.

    Empty bins are returned as ``+inf`` and must be excluded by the evaluation mask.
    """
    n = int(n_grid)
    dz = TWO_PI / n
    i = torch.floor((phi + math.pi) / dz).long().clamp_(0, n - 1)
    j = torch.floor((psi + math.pi) / dz).long().clamp_(0, n - 1)
    lin = i * n + j
    w = torch.exp(logw - logw.max())
    hist = torch.zeros(n * n, device=phi.device, dtype=phi.dtype)
    hist.scatter_add_(0, lin, w)
    counts = torch.zeros(n * n, device=phi.device, dtype=phi.dtype)
    counts.scatter_add_(0, lin, torch.ones_like(w))
    p = hist / (hist.sum() * dz * dz)
    F = torch.full_like(p, float("inf"))
    ok = p > 0
    F[ok] = -(1.0 / beta) * torch.log(p[ok])
    F[ok] = F[ok] - F[ok].min()
    return F.reshape(n, n), counts.reshape(n, n), p.reshape(n, n)


# --------------------------------------------------------------------------- overlap
def overlap_matrix(phi, psi, centers, kappa, beta, N_k, f, chunk=200_000):
    """MBAR overlap matrix ``O_ij = N_i sum_n W_in W_jn`` (Klimovich--Shirts).  Returns ``(K,K)``.

    ``W_kn = exp(f_k - u_kn - ld_n)`` are the normalised MBAR weights, which satisfy
    ``sum_n W_kn = 1`` at the solution.  ``O_ii`` is then the inverse participation ratio of
    window ``i``'s own weights; off-diagonal ``O_ij`` measures phase-space sharing.
    """
    A = _coeff(centers, kappa, beta)
    B = _basis(phi, psi)
    K, N = A.shape[0], B.shape[1]
    Nf = N_k.to(phi.dtype)
    logN = torch.log(Nf.clamp_min(1e-30))
    S = torch.zeros(K, K, device=phi.device, dtype=phi.dtype)
    for s in range(0, N, chunk):
        u = -(A @ B[:, s:s + chunk])
        ld = torch.logsumexp((logN + f)[:, None] - u, dim=0)       # (n,)
        W = torch.exp((f[:, None] - u) - ld[None, :])              # (K,n)
        S += W @ W.T
    return Nf[:, None] * S


def nn_overlap_stats(O, n_per_axis, mask=None):
    """Nearest-neighbour overlaps on the periodic ``n x n`` window lattice.

    Returns ``(values, dict)`` where values are the 4-neighbour overlaps (each pair once).
    ``mask`` (K,) restricts to pairs whose BOTH centres are inside the evaluation region.
    """
    n = int(n_per_axis)
    K = O.shape[0]
    assert K == n * n
    vals = []
    for a in range(n):
        for b in range(n):
            k = a * n + b
            for da, db in ((1, 0), (0, 1)):
                l = ((a + da) % n) * n + (b + db) % n
                if mask is not None and not (bool(mask[k]) and bool(mask[l])):
                    continue
                vals.append(float(O[k, l]))
    v = np.asarray(vals)
    if v.size == 0:
        return v, {}
    return v, dict(min=float(v.min()), p1=float(np.percentile(v, 1)),
                   p5=float(np.percentile(v, 5)), median=float(np.median(v)),
                   n_below_0p03=int((v < 0.03).sum()), n_pairs=int(v.size))
