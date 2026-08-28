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
    tau(x) = 1 / (k kappa(x))

which leaves the conditional equilibrium untouched (same argument as the
kappa-family: the flux of exp(-beta V) in y vanishes pointwise, and any positive
x-dependent mobility multiplies zero).  ``kappa <= 1`` always -- difficulty is
created by slowing down, never by speeding up, so no region is integrated less
accurately than the baseline (the kappa-family's Gate 0I lesson).

Default regime, chosen against the preregistered resolvability window
``50 dt <= tau <= 800 dt << T``::

    k = 20, dt = 1e-3  ->  tau_min = 1/(k kappa_max) = 0.05 = 50 dt
    kappa in [1/16, 1] ->  tau_max = 0.8 = 800 dt;  T = 40 = 50 tau_max

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
    H: float = 2.5
    k: float = 20.0
    c: float = 1.0
    a_kappa: float = math.log(16.0)   # fold-spread of tau; 0 = flat control
    kappa_shift: float = 0.0
    N: int = 1024
    dt: float = 1e-3
    n_steps: int = 40_000
    save_every: int = 400
    h: float = 0.07
    min_count: float = 1.0

    @property
    def sigma_f2(self):
        return self.k * self.c ** 2 / self.beta

    def tau_range(self):
        return 1.0 / self.k, math.exp(self.a_kappa) / self.k


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
    Yn = Y - fy * cfg.dt + torch.sqrt(2.0 * kap / cfg.beta * cfg.dt) * zy \
        + cfg.c * 0.0
    Xn = X + fx_applied * cfg.dt + math.sqrt(2.0 * cfg.dt / cfg.beta) * zx
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
