"""Phase-5 benchmark: constant sigma^2, resolvable heterogeneous tau.

Frozen protocol: ``docs/MECHANISM_CAMPAIGN_PREREGISTRATION.md`` (Phase 5).

Model::

    V(x, y) = U(x) + (k/2)(y - c x)^2,      U(x) = H (x^2 - 1)^2,   xi = x

At fixed x the fibre is Gaussian with x-independent width, so analytically

    A(x)       = U(x) + const          A'(x) = U'(x)
    f_loc      = U'(x) - k c (y - c x)
    sigma_f^2  = k c^2 / beta          CONSTANT in x

Only the fibre *mobility* varies::

    dY = -kappa(x) k (y - c x) dt + sqrt(2 kappa(x) / beta) dW

**The trajectory relaxation is NOT 1/(k kappa).**  The physical x-force is
``-U' + k c u`` with ``u = y - c x``, so x responds to the fibre residual, and
``du = dy - c dx`` picks up ``-k c^2 u dt`` from that response::

    tau(x) = 1 / ( k (kappa(x) + c^2) )

Fluctuation--dissipation still gives ``Var(u) = 1/(beta k)`` exactly (measured:
matched to 7 percent while the first design's tau collapsed 16x -> 1.9x), which
is why the flaw showed only in the correlation time.  So ``c`` must satisfy
``c^2 << kappa_min`` for the kappa spread to survive -- but also
``k c^2 / beta >> Var_within-cell(U')`` or the cell-mean history reads the
composition noise of ``U'`` instead of the fibre signal.  With ``kappa_min =
1/16`` those bracket ``c``; the defaults use ``c = 0.1`` and a mild ``H`` so
both hold with room.

which leaves the conditional equilibrium untouched (same argument as the
kappa-family: the flux of exp(-beta V) in y vanishes pointwise, and any positive
x-dependent mobility multiplies zero).  ``kappa <= 1`` always -- difficulty is
created by slowing down, never by speeding up, so no region is integrated less
accurately than the baseline (the kappa-family's Gate 0I lesson).

Resolvability needs FOUR inequalities, not three.  The first validation run
failed its own gate and taught the missing one: the frozen estimator reads the
autocorrelation of the CELL-MEAN force, and a particle leaves an allocation
cell of width ``w`` in ``t_cross ~ w^2 beta / 2``; if ``tau`` is not small
against that, the series decorrelates by population turnover and every cell
reads fast (measured: tau_hat = 0.036 x truth, rank correlation 0.157, with
sigma^2 flat at 1.03 exactly as designed).  So::

    obs_dt  <<  tau_min  <  tau_max  <<  t_cross(w, beta)  <<  T

Default regime (v2, after that amendment)::

    k = 2000, dt = 1e-5      ->  tau_min = 5e-4 = 50 dt
    kappa in [1/16, 1]       ->  tau_max = 8e-3 = 800 dt
    J = 32 on [-1.8, 1.8], beta = 4  ->  t_cross ~ 2.5e-2 ~ 3 tau_max
    (use J = 16, w = 0.225, for t_cross ~ 0.10 ~ 12 tau_max where needed)

The Euler multiplier on y is ``1 - kappa k dt in [0.98, 0.99875]``: stable
everywhere, and the *hard* cells are the more accurately integrated ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

XMIN, XMAX = -1.8, 1.8
N_GRID = 181
EVAL_LO, EVAL_HI = -1.5, 1.5
EPS = 1e-30
KAPPA_PERIOD = XMAX - XMIN          # one full period across the domain


@dataclass(frozen=True)
class TauConfig:
    beta: float = 4.0
    H: float = 0.25
    k: float = 2000.0
    c: float = 0.1
    #: x-mobility.  Slowing x lengthens the cell-residence time t_res ~
    #: w^2 beta / (pi^2 mu_x) that caps what the cell-mean estimator can see of
    #: tau, and shrinks the c^2 feedback damping to mu_x k c^2.  The price is
    #: slower global x transport, which matters for arm comparisons, not for
    #: per-cell statistics.
    mu_x: float = 0.1
    a_kappa: float = math.log(16.0)   # fold-spread of tau; 0 = flat control
    kappa_shift: float = 0.0
    N: int = 1024
    dt: float = 1e-5
    n_steps: int = 200_000
    save_every: int = 2000
    h: float = 0.07
    min_count: float = 1.0

    @property
    def sigma_f2(self):
        return self.k * self.c ** 2 / self.beta

    def tau_of_kappa(self, kappa):
        """Trajectory relaxation time: 1 / (k (kappa + mu_x c^2))."""
        return 1.0 / (self.k * (kappa + self.mu_x * self.c ** 2))

    def tau_range(self):
        return (self.tau_of_kappa(1.0),
                self.tau_of_kappa(math.exp(-self.a_kappa)))


def kappa_of(x, cfg: TauConfig):
    """``exp(a (sin(2 pi (x+shift)/P) - 1)/2)`` in [e^-a, 1].  Never reads y."""
    if cfg.a_kappa == 0.0:
        return torch.ones_like(x)
    hshape = torch.sin(2.0 * math.pi * (x + cfg.kappa_shift) / KAPPA_PERIOD)
    return torch.exp(0.5 * cfg.a_kappa * (hshape - 1.0))


def U_of(x, cfg):    return cfg.H * (x * x - 1.0) ** 2
def dU_of(x, cfg):   return 4.0 * cfg.H * x * (x * x - 1.0)


def reference(xg, cfg: TauConfig, eval_mask):
    """Analytic A and A' -- the fibre's log-width is x-independent."""
    A = U_of(xg, cfg)
    A = A - A[eval_mask].mean()
    return A, dU_of(xg, cfg)


def local_force(X, Y, cfg):
    return dU_of(X, cfg) - cfg.k * cfg.c * (Y - cfg.c * X)


def step_xy(X, Y, fx_applied, cfg, gen, device, dtype):
    """One Euler--Maruyama step.  kappa is evaluated at the PRE-step X (explicit
    scheme, y-noise independent of the same step's x update -- the kappa-family
    convention)."""
    kap = kappa_of(X, cfg)
    zx = torch.randn(X.shape, generator=gen, device=device, dtype=dtype)
    zy = torch.randn(X.shape, generator=gen, device=device, dtype=dtype)
    fy = kap * cfg.k * (Y - cfg.c * X)
    Yn = Y - fy * cfg.dt + torch.sqrt(2.0 * kap / cfg.beta * cfg.dt) * zy
    Xn = X + cfg.mu_x * fx_applied * cfg.dt \
        + math.sqrt(2.0 * cfg.mu_x * cfg.dt / cfg.beta) * zx
    # reflect X into the domain
    span = XMAX - XMIN
    Xw = torch.remainder(Xn - XMIN, 2.0 * span)
    Xn = torch.where(Xw > span, 2.0 * span - Xw, Xw) + XMIN
    return Xn, Yn


def init_conditions(cfg: TauConfig, R, seed, device, dtype):
    rng = np.random.default_rng(1000 + seed)
    X = torch.as_tensor(rng.uniform(XMIN, XMAX, (R, cfg.N)), device=device,
                        dtype=dtype)
    Z = torch.as_tensor(rng.normal(0.0, 1.0, (R, cfg.N)), device=device,
                        dtype=dtype)
    Y = cfg.c * X + Z / math.sqrt(cfg.beta * cfg.k)
    return X, Y
