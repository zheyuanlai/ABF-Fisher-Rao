"""Chapter-3 geometry for a NONLINEAR reaction coordinate, and the lifts.

The rest of this package uses  xi(q) = x, for which  grad xi = e_1,  G = 1  and
every object below degenerates: the tangent projector is "drop the first
component", the Fixman factor (det G)^{-1/2} is 1, the mean-force formula is
f = dV/dx, and the minimum-norm horizontal lift  grad xi G^{-1} u  is EXACTLY
the identity lift ("move the CV, drag the fiber unchanged").  None of the
manifold structure can be tested there.  This module supplies a family where
det G varies ALONG the fiber, so all of it becomes measurable.

Geometry (m = 1).  For  xi : R^n -> R,

    grad xi(q) in R^n,      G(q) = |grad xi(q)|^2,
    Sigma(z)   = {q : xi(q) = z},
    P(q)       = I - grad xi grad xi^T / G          (tangent projector),
    V^xi(q)    = V(q) + (1/2 beta) log G(q)         (Fixman-corrected potential).

The conditional measure that thermodynamic integration needs is

    nu^xi(dq | z)  propto  e^{-beta V} (det G)^{-1/2} sigma_Sigma(dq)
                   propto  e^{-beta V^xi}           sigma_Sigma(dq),

so a constrained sampler run on the BARE V samples the *rigid* measure
e^{-beta V} sigma_Sigma and returns F_rgd, not F.  `fixman=False` reproduces
that error on purpose.

Local mean force (Lelievre-Rousset-Stoltz eq. 3.32, m = 1):

    f(q) = (grad xi . grad V) / G  -  beta^{-1} div( grad xi / G ),
    F'(z) = E_{nu^xi(.|z)} [ f ].

LIFTS.  Any  dq  with  grad xi . dq = dz  moves a configuration from Sigma(z) to
Sigma(z + dz); they form an affine space over the tangent space T_q Sigma(z).
Three members are implemented here and they are genuinely different once xi is
nonlinear:

  'minnorm'   dq = grad xi G^{-1} dz   -- the minimum-EUCLIDEAN-norm choice, i.e.
              the horizontal lift of the ambient metric.  This is the one a
              moving hard constraint produces, because a d'Alembert constraint
              force acts along grad xi.
  'cartesian' dq = dz e_1 / (grad xi)_1 -- move one ambient coordinate.  Cheapest;
              has a nonzero tangential component; the naive lift.
  'adiabatic' the fiber velocity that solves the conditional continuity equation
              (see systems/graph.py).  Zero conditional lag by construction; the
              only one with a STATISTICAL justification rather than a metric one.

Minimum norm is a statement about the ambient metric, not about nu^xi.  Which of
the three is best is an empirical question, and it is the one this module exists
to ask.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .grid import DEVICE, DTYPE, EPS


# ---------------------------------------------------------------------------
# reaction coordinate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GraphCV:
    """xi(q) = x + a sin(k y_1)  on  q = (x, y_1, ..., y_d).

    Chosen because Sigma(z) is a GRAPH over the fiber coordinates,

        Sigma(z) = { (z - a sin(k y_1), y) : y in R^d },

    which makes every Chapter-3 object exactly computable:

        grad xi = (1, c, 0, ..., 0),   c := a k cos(k y_1),
        G       = 1 + c^2                      (varies ALONG the fiber),
        dsigma  = sqrt(G) dy,
        (det G)^{-1/2} dsigma = dy             (co-area, Jacobian 1).

    The last line is the whole point: the conditional measure in the graph
    parameterization is the plain  nu(y|z) propto e^{-beta V(z - a sin k y_1, y)},
    with NO extra factor, so an exact reference is one quadrature away, while an
    ambient (molecular-style) implementation has to get  (det G)^{-1/2}  right or
    it converges to the wrong measure.

    a = 0 recovers the linear coordinate xi = x used by the rest of the package.
    """
    a: float = 0.6
    k: float = 1.4
    d: int = 1

    # ---- basic scalars (all take y1 or q and broadcast) --------------------
    def s(self, y1):
        return self.a * torch.sin(self.k * y1)

    def c(self, y1):
        """ds/dy_1 = a k cos(k y_1)."""
        return self.a * self.k * torch.cos(self.k * y1)

    def dc(self, y1):
        """d^2 s/dy_1^2."""
        return -self.a * self.k * self.k * torch.sin(self.k * y1)

    def xi(self, q):
        return q[..., 0] + self.s(q[..., 1])

    def G(self, y1):
        c = self.c(y1)
        return 1.0 + c * c

    def log_det_G(self, y1):
        return torch.log(self.G(y1))

    def grad_xi(self, q):
        """(..., n) with n = 1 + d."""
        g = torch.zeros_like(q)
        g[..., 0] = 1.0
        g[..., 1] = self.c(q[..., 1])
        return g

    # ---- projectors --------------------------------------------------------
    def tangent_project(self, q, v):
        """P(q) v = v - grad xi (grad xi . v) / G."""
        gx = self.grad_xi(q)
        coef = (gx * v).sum(-1, keepdim=True) / self.G(q[..., 1]).unsqueeze(-1)
        return v - coef * gx

    # ---- Fixman ------------------------------------------------------------
    def fixman_grad(self, q, beta):
        """grad of (1/2 beta) log det G;  nonzero only in the y_1 slot."""
        y1 = q[..., 1]
        c, dc = self.c(y1), self.dc(y1)
        g = torch.zeros_like(q)
        g[..., 1] = c * dc / (beta * (1.0 + c * c))
        return g

    # ---- mean force --------------------------------------------------------
    def mean_force(self, q, grad_V, beta):
        """f = (grad xi . grad V)/G - beta^{-1} div(grad xi / G).

        div(grad xi / G) = d/dy_1 [ c / G ] = c' (1 - c^2) / G^2
        (the d/dx term vanishes because G does not depend on x).
        """
        y1 = q[..., 1]
        c, dc = self.c(y1), self.dc(y1)
        G = 1.0 + c * c
        dot = grad_V[..., 0] + c * grad_V[..., 1]
        div = dc * (1.0 - c * c) / (G * G)
        return dot / G - div / beta

    # ---- retraction / projection onto Sigma(z) -----------------------------
    def project(self, q, z, mode="minnorm", n_newton: int = 8):
        """Return q' with xi(q') = z.

        mode='minnorm'   Newton on  xi(q + lam * grad xi(q)) = z  -- the SHAKE
                         projection, moving along the constraint-force direction.
        mode='cartesian' exact one-liner  x <- z - s(y_1),  fiber untouched.

        A fixed iteration count, deliberately: a residual-based early exit needs
        `float(r.max())`, which synchronizes the device on every iteration and, in
        an inner dynamics loop, costs far more than the extra arithmetic.  Newton
        reaches machine precision here in about four steps; `tests/test_manifold.py`
        asserts the residual.
        """
        if mode == "cartesian":
            out = q.clone()
            out[..., 0] = z - self.s(q[..., 1])
            return out
        gx = self.grad_xi(q)              # frozen at q, as SHAKE does
        lam = torch.zeros(q.shape[:-1], device=q.device, dtype=q.dtype)
        c0 = gx[..., 1]
        for _ in range(n_newton):
            y1 = q[..., 1] + lam * c0
            r = q[..., 0] + lam + self.s(y1) - z
            dr = 1.0 + self.c(y1) * c0
            lam = lam - r / torch.clamp(dr.abs(), min=1e-9) * torch.sign(dr)
        return q + lam.unsqueeze(-1) * gx

    # ---- lifts -------------------------------------------------------------
    def lift(self, q, dz, mode="minnorm"):
        """One infinitesimal lift step; the caller re-projects afterwards."""
        if mode == "minnorm":
            gx = self.grad_xi(q)
            return q + (dz / self.G(q[..., 1])).unsqueeze(-1) * gx
        if mode == "cartesian":
            out = q.clone()
            out[..., 0] = out[..., 0] + dz
            return out
        raise ValueError(mode)


# ---------------------------------------------------------------------------
# constrained overdamped Langevin on Sigma(z)
# ---------------------------------------------------------------------------
def constrained_step(cv: GraphCV, q, z, grad_V, dt, beta, gen,
                     fixman: bool = True, proj: str = "minnorm"):
    """One projected Euler-Maruyama step of constrained overdamped Langevin.

        q~ = q + P(q) [ -grad W dt + sqrt(2 dt / beta) dB ]
        q' = project(q~ -> Sigma(z))                      (SHAKE, along grad xi)

    with W = V^xi if `fixman` else V.  The tangential projection of the noise
    plus the constraint projection is the standard scheme; it carries the usual
    O(dt) discretization bias, which the calling experiment measures separately
    so that the Fixman effect can be reported against it.
    """
    W = grad_V + (cv.fixman_grad(q, beta) if fixman else 0.0)
    noise = torch.randn(q.shape, device=q.device, dtype=q.dtype, generator=gen)
    step = -W * dt + ((2.0 * dt / beta) ** 0.5) * noise
    qt = q + cv.tangent_project(q, step)
    return cv.project(qt, z, mode=proj)
