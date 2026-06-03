"""Reference free energy / mean force for ``xi(x, y) = x`` by quadrature.

Definitions (free energy defined up to an additive constant ``C``):

    F_ref(x)      = -(1/beta) * log integral_y exp(-beta V(x, y)) dy + C
    F'_ref(x)     =  integral_y dV/dx(x, y) exp(-beta V(x, y)) dy
                     -----------------------------------------------------
                            integral_y          exp(-beta V(x, y)) dy
    p_ref(x)      proportional to integral_y exp(-beta V(x, y)) dy   (unbiased x-marginal)

The y-integral is evaluated by the trapezoidal rule on a fine grid with a
log-sum-exp shift for numerical stability.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from . import potentials

EPS = 1e-300


def compute_reference(x_grid, y_grid, beta,
                      V_func=potentials.potential_xy,
                      dVdx_func=potentials.dVdx_xy):
    """Compute reference profiles on ``x_grid`` using y-quadrature on ``y_grid``.

    Returns a dict with keys ``log_Z``, ``F_ref``, ``Fprime_ref``, ``p_ref``,
    all 1-D arrays on ``x_grid``.  ``F_ref`` is centred to have zero mean over
    ``x_grid`` (the additive constant is fixed by the convention
    ``F <- F - mean(F)``); ``p_ref`` integrates to 1 over ``x_grid``.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)

    # Shape (n_x, n_y): each row is a fixed x swept over the y quadrature nodes.
    xx = x_grid[:, None]
    yy = y_grid[None, :]

    phi = beta * V_func(xx, yy)               # beta V(x, y)
    dvdx = dVdx_func(xx, yy)                   # dV/dx(x, y)

    m = phi.min(axis=1, keepdims=True)        # per-x shift for stability
    w = np.exp(-(phi - m))                     # exp(-(beta V - m)) in [0, 1]

    Z_stab = np.trapezoid(w, y_grid, axis=1)   # integral of shifted weights
    log_Z = -m[:, 0] + np.log(np.maximum(Z_stab, EPS))

    Fprime_ref = (np.trapezoid(dvdx * w, y_grid, axis=1)
                  / np.maximum(Z_stab, EPS))

    F_ref = -(1.0 / beta) * log_Z
    F_ref = F_ref - np.mean(F_ref)             # F <- F - mean(F)

    # Unbiased x-marginal p_ref(x) proportional to Z(x) = exp(log_Z).
    lz = log_Z - log_Z.max()
    p_unnorm = np.exp(lz)
    p_ref = p_unnorm / np.maximum(np.trapezoid(p_unnorm, x_grid), EPS)

    return dict(log_Z=log_Z, F_ref=F_ref, Fprime_ref=Fprime_ref, p_ref=p_ref)


def conditional_y_density(x0, y_grid, beta, V_func=potentials.potential_xy):
    """Reference conditional density ``p_ref(y | x = x0)`` on ``y_grid``.

    Normalised to integrate to 1 over ``y_grid``.
    """
    y_grid = np.asarray(y_grid, dtype=float)
    phi = beta * V_func(np.full_like(y_grid, float(x0)), y_grid)
    phi = phi - phi.min()
    w = np.exp(-phi)
    Z = np.maximum(np.trapezoid(w, y_grid), EPS)
    return w / Z


def build_reference_grid(cfg: Dict, beta: float):
    """Build the 2-D reference grid (potential and Boltzmann density).

    Returns ``(x_grid, y_grid, V_grid, rho_grid)`` where ``V_grid`` and
    ``rho_grid`` have shape ``(ny, nx)`` (``indexing="xy"``) and ``rho_grid``
    integrates to 1 over the 2-D domain.
    """
    d = cfg["domain"]
    x_grid, y_grid, XX, YY = potentials.make_grid(
        d["x_min"], d["x_max"], d["y_min"], d["y_max"],
        d["nx_ref"], d["ny_ref"],
    )
    V_grid = potentials.potential_xy(XX, YY)
    phi = beta * (V_grid - V_grid.min())
    rho = np.exp(-phi)
    norm = np.trapezoid(np.trapezoid(rho, x_grid, axis=1), y_grid)
    rho_grid = rho / np.maximum(norm, EPS)
    return x_grid, y_grid, V_grid, rho_grid


def profile_grid(cfg: Dict):
    """1-D reaction-coordinate grid used for ABF profiles and FR targets."""
    d = cfg["domain"]
    return np.linspace(d["x_min"], d["x_max"], int(d["nx_profile"]))
