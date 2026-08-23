"""Chapter-3 geometry for a molecular reaction coordinate xi : R^{3A} -> R^m.

Everything here is the mass-metric version of `rcwfr/manifold.py`.  With
mobility M^{-1} (Brownian dynamics with per-atom friction proportional to mass)
the natural Gram matrix is

    G(q) = grad xi^T M^{-1} grad xi                                  (m x m)

and the two measures that matter on Sigma(z) = {xi = z} are

    nu_rgd(dq|z)  propto  e^{-beta V} sigma_Sigma^M (dq)          "rigid"
    nu^xi(dq|z)   propto  e^{-beta V} (det G)^{-1/2} sigma_Sigma^M(dq)

where sigma^M is the surface measure of the mass metric.  Constrained Brownian
dynamics on the BARE potential samples the first; thermodynamic integration
needs the second.  Two routes:

  RIGID ROUTE (used everywhere here).  Simulate the bare V, and deposit the
  local mean force with the importance weight (det G)^{-1/2}.  The shared
  MeanForceAccumulator is already a self-normalised weighted estimator, so
  F'(z) = E_rgd[w f]/E_rgd[w] falls out of it with no extra machinery, and the
  reweighting ESS is a directly reportable diagnostic.

  FIXMAN ROUTE (kept as a control).  Simulate V + (1/2 beta) log det G and
  deposit unweighted.  Needs second derivatives of xi every step.

Local mean force (Lelievre-Rousset-Stoltz eq. 3.32, mass-metric, vector CV):

    w_j    = (M^{-1} grad xi G^{-1})_{. j}      (dual basis: grad xi_k . w_j = delta)
    f_j    = (G^{-1} grad xi^T M^{-1} grad V)_j - beta^{-1} div w_j
    dF/dz_j(z) = E_{nu^xi(.|z)} [ f_j ].

div w_j is assembled from Hessian contractions of xi only -- and a torsion angle
depends on four atoms, so the Hessian is 12 x 12 (15 x 15 for two torsions that
share atoms) regardless of how big the molecule is.
"""
from __future__ import annotations

import torch

from .ff import dihedral, _wrap


def small_solve(A, b):
    """Solve A x = b for A: (..., m, m) with m in {1, 2}, b: (..., m).

    torch.linalg.solve dispatches to a batched LAPACK path that costs more in
    launch overhead than the whole rest of a SHAKE sweep; m is 1 or 2 here, so
    the closed form is both faster and exact.
    """
    m = A.shape[-1]
    if m == 1:
        return b / A[..., 0, 0].unsqueeze(-1)
    if m == 2:
        det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
        x0 = (A[..., 1, 1] * b[..., 0] - A[..., 0, 1] * b[..., 1]) / det
        x1 = (-A[..., 1, 0] * b[..., 0] + A[..., 0, 0] * b[..., 1]) / det
        return torch.stack([x0, x1], dim=-1)
    return torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)


def small_inv(A):
    m = A.shape[-1]
    if m == 1:
        return 1.0 / A
    if m == 2:
        det = (A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]).unsqueeze(-1).unsqueeze(-1)
        adj = torch.stack([torch.stack([A[..., 1, 1], -A[..., 0, 1]], -1),
                           torch.stack([-A[..., 1, 0], A[..., 0, 0]], -1)], -2)
        return adj / det
    return torch.linalg.inv(A)


class TorsionCV:
    """m dihedral angles as the reaction coordinate, trans-at-zero convention."""

    def __init__(self, idx, mass, periodic=True, shift=True):
        self.idx = idx                                   # (m, 4) long, global atom ids
        self.m = idx.shape[0]
        self.mass = mass
        self.minv = 1.0 / mass                           # (A,)
        self.A = mass.numel()
        self.periodic = periodic
        self.shift = shift
        sup = torch.unique(idx.reshape(-1))
        self.support = sup                               # (S,) global ids
        self.S = sup.numel()
        remap = torch.full((self.A,), -1, dtype=torch.long, device=idx.device)
        remap[sup] = torch.arange(self.S, device=idx.device)
        self.idx_local = remap[idx]                      # (m, 4) into the support
        self.minv_sup = self.minv[sup]                   # (S,)
        self._gradfns = [torch.func.grad(
            (lambda x, k=k: self._val_local(x)[..., k].sum())) for k in range(self.m)]


    # -- values -------------------------------------------------------------
    def value(self, q):
        """(..., A, 3) -> (..., m)"""
        return dihedral(q, self.idx, self.shift)

    def _val_local(self, qs):
        return dihedral(qs, self.idx_local, self.shift)

    def grad_local(self, q):
        """d xi / d q restricted to the support atoms: (..., m, S, 3)."""
        qs = q[..., self.support, :]
        return torch.stack([f(qs) for f in self._gradfns], dim=-3)

    def grad(self, q):
        """Full (..., m, A, 3) gradient, zero off the support."""
        gs = self.grad_local(q)
        out = torch.zeros(q.shape[:-2] + (self.m, self.A, 3),
                          device=q.device, dtype=q.dtype)
        out[..., :, self.support, :] = gs
        return out

    # -- metric -------------------------------------------------------------
    def grad_local_raw(self, qs):
        """Gradient when the caller already holds only the support slice."""
        return torch.stack([f(qs) for f in self._gradfns], dim=-3)

    def gram_from_grad(self, gs):
        """gs: (..., m, S, 3) -> G: (..., m, m).  Uses the support masses."""
        w = gs * self.minv_sup.view(-1, 1)               # M^{-1} grad xi
        return torch.einsum("...kij,...lij->...kl", gs, w)

    def gram(self, q):
        return self.gram_from_grad(self.grad_local(q))

    def log_det_G(self, q):
        G = self.gram(q)
        return torch.logdet(G) if self.m > 1 else torch.log(G[..., 0, 0])

    # -- projections --------------------------------------------------------
    def tangent_project_local(self, gs, G, v):
        """M-orthogonal projection of v (..., S, 3) onto T Sigma."""
        c = torch.einsum("...kij,...ij->...k", gs, v)             # grad xi . v
        lam = small_solve(G, c)                                   # G^{-1} c
        corr = torch.einsum("...k,...kij->...ij", lam, gs) * self.minv_sup.view(-1, 1)
        return v - corr

    def dz_residual(self, xi, z):
        return _wrap(xi - z) if self.periodic else xi - z

    def project(self, q, z, n_newton: int = 4, n_outer: int = 1, gs=None, G=None):
        """SHAKE: find lam with xi(q + M^{-1} grad xi(q)^T lam) = z.

        Quasi-Newton with the Jacobian FROZEN at q -- which is exactly what
        SHAKE does, and here it is also the whole speed story: the frozen
        Jacobian is G(q), already in hand, so the iteration costs only forward
        evaluations of xi and the step needs ONE backward pass instead of one
        per iteration.  Convergence is linear but the contraction factor is
        O(|dz|), so a dynamics-sized displacement reaches machine precision in
        3-4 sweeps.  `n_outer > 1` refreshes the frozen gradient, which is what
        makes a LARGE displacement (an initial placement, or a big lift)
        converge.  A fixed iteration count avoids a device sync.
        """
        out = q
        lam_tot = torch.zeros(q.shape[:-2] + (self.m,), device=q.device, dtype=q.dtype)
        for it in range(n_outer):
            if it == 0 and gs is not None:
                g, Gm = gs, (G if G is not None else self.gram_from_grad(gs))
            else:
                g = self.grad_local(out)
                Gm = self.gram_from_grad(g)
            qs = out[..., self.support, :]
            d = g * self.minv_sup.view(-1, 1)                     # (..., m, S, 3)
            lam = torch.zeros_like(lam_tot)
            for _i in range(n_newton):
                qn = qs + torch.einsum("...k,...kij->...ij", lam, d)
                r = self.dz_residual(self._val_local(qn), z)      # (..., m)
                lam = lam - small_solve(Gm, r)
            qn = qs + torch.einsum("...k,...kij->...ij", lam, d)
            out = _scatter(out, qn, self.support)
            lam_tot = lam_tot + lam
        return out, lam_tot

    # -- mean force ---------------------------------------------------------
    def _hessian(self, qs):
        """Full Hessian of every CV w.r.t. the support coordinates.

        qs: (..., S, 3) -> (..., m, 3S, 3S).  torch.func vectorises the 3S
        tangents into one batched forward pass, which is ~50x faster than 3S
        separate `autograd.grad` calls and is why the mean force can be
        deposited often enough to matter.
        """
        n = 3 * self.S
        flat = qs.reshape(-1, n)
        fn = lambda x: self._val_local(x.reshape(self.S, 3))
        H = torch.func.vmap(torch.func.jacfwd(torch.func.jacrev(fn)))(flat)
        return H.reshape(qs.shape[:-2] + (self.m, n, n))

    def mean_force(self, q, grad_V, beta, gs=None):
        """f: (..., m).  Includes the beta^{-1} divergence term.

        f_j = (G^{-1} grad xi^T M^{-1} grad V)_j - beta^{-1} div w_j,
        div w_j = sum_k Gi[k,j] tr(M^-1 H_k)
                  - sum_{k,a,b} (Q[a,k,b] + Q[b,k,a]) Gi[k,a] Gi[b,j],
        Q[k,a,b] = (M^-1 grad xi_a) . H_k (M^-1 grad xi_b).
        """
        qs = q[..., self.support, :]
        if gs is None:
            gs = self.grad_local(q)
        d = gs * self.minv_sup.view(-1, 1)                     # M^{-1} grad xi
        G = torch.einsum("...kij,...lij->...kl", gs, d)
        Gi = small_inv(G)
        H = self._hessian(qs)                                  # (..., m, 3S, 3S)
        n = 3 * self.S
        minv3 = self.minv_sup.repeat_interleave(3)             # (3S,)
        T = torch.einsum("...kii,i->...k", H, minv3)
        df = d.reshape(d.shape[:-2] + (n,))                    # (..., m, 3S)
        Q = torch.einsum("...ai,...kij,...bj->...kab", df, H, df)
        div = torch.einsum("...kj,...k->...j", Gi, T)
        div = div - torch.einsum("...akb,...ka,...bj->...j", Q, Gi, Gi) \
                  - torch.einsum("...bka,...ka,...bj->...j", Q, Gi, Gi)
        gvs = grad_V[..., self.support, :]
        proj = torch.einsum("...kij,...ij->...k", d, gvs)
        f = torch.einsum("...jk,...k->...j", Gi, proj) - div / beta
        return f, G


def _scatter(q, qs, support):
    out = q.clone()
    out[..., support, :] = qs
    return out


def _grad_at(cv: TorsionCV, qn):
    return cv.grad_local_raw(qn)
