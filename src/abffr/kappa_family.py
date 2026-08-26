"""The kappa-family instrument: vary conditional difficulty at fixed free energy.

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md`` (Stage 2, Gate 0I).

The campaign has to separate two things the previous four campaigns could not:
whether ``r`` should follow the physical density, and whether it should follow
statistical difficulty.  On any ordinary potential those covary -- barrier
regions are both rare and hard -- so a win is unattributable.  This module
breaks the tie by scaling the mobility of the *hidden* coordinate::

    dY = -kappa(X) dV/dY dt + sqrt(2 kappa(X) / beta) dW

Why ``F`` does not move
-----------------------
The Fokker--Planck flux of ``exp(-beta V)`` vanishes pointwise in each
coordinate -- ``dV/dy rho + beta^-1 drho/dy == 0`` -- so multiplying the
``y``-flux by any positive ``kappa`` leaves it zero.  The invariant density, and
hence ``F(x)`` and ``q_phys``, is exactly unchanged.  There is no cancellation to
verify and no dependence on the shape of ``h``.

Why kappa never exceeds 1
-------------------------
The continuum invariance holds for any positive ``kappa``, but the engine runs
Euler--Maruyama at a fixed ``dt``, and mobility ``kappa`` on the hidden
coordinate is an effective timestep ``kappa dt`` there.  Gate 0I measured this:
at ``kappa = 16`` the sampled conditional misses ``exp(-beta V)`` by 3.3% total
variation, against 2% at ``kappa <= 1`` -- so a "speed up half the domain"
parameterization would move ``F`` through discretisation error even though the
continuum statement says it cannot.  That is the instrument becoming the
confound.

So difficulty is created by *slowing down*, never by speeding up::

    kappa_a(x) = exp( a (h(x) - 1) / 2 )   in   [exp(-a), 1]

``a = log C`` gives a ``C``-fold spread in mixing time with ``kappa <= 1``
everywhere, so no cell is integrated less accurately than the baseline is.  K3
mirrors K2 by shifting ``h`` half a period rather than by negating ``a``, which
would put ``kappa`` above 1 again.

Why kappa may not depend on ``y``
---------------------------------
With ``kappa(x)`` the Ito term ``d2[(kappa/beta) rho]/dy2`` equals
``d[(kappa/beta) drho/dy]/dy`` and the flux form above holds.  With
``kappa(x, y)`` it does not: the residual drift ``beta^-1 dkappa/dy`` shifts
``F``, turning the instrument into a confound of exactly the kind it exists to
remove.  :func:`kappa_at` therefore takes ``x`` only, and the engine evaluates it
at the pre-step ``X`` so the scheme stays explicit and the ``y``-noise stays
independent of the same step's ``x`` update.

The algorithm never sees any of this
------------------------------------
``a`` and ``h`` are benchmark-constructor inputs.  No allocation module imports
this one (Gate 0A), and difficulty reaches ``r`` only through the online
``Gamma_hat`` measured from force observations.  We know where the difficulty is;
the method has to find out.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

#: The frozen difficulty shape: exactly one period across the [-3, 3] domain, so
#: the profile is *asymmetric* about the barrier at x = 0 while ``F`` is nearly
#: symmetric.  An allocation that tracked density would be symmetric; one that
#: tracks difficulty cannot be.  That asymmetry is what makes the K2/K3 mirror
#: test able to tell the two apart.
H_PERIOD = 6.0

#: Preregistered cells as ``(a, shift)``.  ``a = log C`` is the fold-spread in
#: mixing time; ``shift`` relocates the hard region.  K3 mirrors K2 by moving
#: ``h`` half a period, NOT by negating ``a`` -- negating would push kappa above
#: 1 and reintroduce the discretisation bias Gate 0I found.
KAPPA_CELLS: Dict[str, tuple] = {
    "K0": (0.0, 0.0),
    "K1": (math.log(4.0), 0.0),
    "K2": (math.log(16.0), 0.0),
    "K3": (math.log(16.0), 0.5 * H_PERIOD),
}


def h_shape(x, shift: float = 0.0):
    """``sin(2 pi (x + shift) / H_PERIOD)`` -- fixed, algorithm-invisible."""
    x = np.asarray(x, dtype=float) + float(shift)
    return np.sin(2.0 * np.pi * x / H_PERIOD)


def kappa_at(x, a: float, shift: float = 0.0):
    """``exp(a (h(x) - 1) / 2)`` in ``[exp(-a), 1]``.  Never reads ``y``."""
    return np.exp(0.5 * float(a) * (h_shape(x, shift) - 1.0))


def kappa_at_torch(X, a: float, shift: float = 0.0):
    """Torch version for the engine.  ``a == 0`` returns ``None``.

    ``None`` rather than a tensor of ones so the unmodified propagation is the
    literal same arithmetic as before this module existed -- a K0 run must not
    differ from a clean-v2 run by a multiply-by-one rounding path.
    """
    import torch
    if float(a) == 0.0:
        return None
    h = torch.sin(2.0 * math.pi * (X + float(shift)) / H_PERIOD)
    return torch.exp(0.5 * float(a) * (h - 1.0))


def tau_spread(a: float) -> float:
    """Fold-range of the conditional mixing time this cell creates."""
    return float(math.exp(float(a)))


def cell_from_config(cfg: Dict) -> tuple:
    """``(a, shift)`` for the configured kappa cell.  Absent block means K0."""
    block = (cfg.get("kappa", {}) or {})
    if not block:
        return (0.0, 0.0)
    if "a" in block and "cell" in block:
        raise ValueError("give kappa.cell or kappa.a, not both")
    if "a" in block:
        return float(block["a"]), float(block.get("shift", 0.0))
    name = str(block.get("cell", "K0"))
    if name not in KAPPA_CELLS:
        raise ValueError(f"unknown kappa cell {name!r}; have {sorted(KAPPA_CELLS)}")
    return KAPPA_CELLS[name]
