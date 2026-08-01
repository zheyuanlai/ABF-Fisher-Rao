"""Bias projection with an enforced online/frozen-bias consistency guarantee.

The frozen-bias validation compares a run's online behaviour against fresh dynamics under the
*saved* potential.  That comparison is only meaningful if the field the dynamics actually felt is
the gradient of the field that was saved.  Two things can break it:

  1. the Nyquist defect in :func:`alkanes.poisson2d.poisson_projection` (fixed upstream in
     ``c6a6718``, but only for the returned pair -- an odd grid removes the failure mode entirely
     because there is no ``k = n/2`` index);
  2. per-component clipping of the interpolated bias, which destroys the curl-free property the
     projection exists to create.  Clipping ``gB1`` and ``gB2`` independently is not a tuning
     choice, it is a correctness bug.  Here the **magnitude** of the 2-vector is limited, which
     preserves direction and therefore keeps the applied field a positive rescaling of a gradient
     along each streamline.

:func:`project_bias` is the single supported entry point and asserts the guarantee it provides.
"""
from __future__ import annotations

import torch

from alkanes import density2d as d2
from alkanes import poisson2d as ps

EPS = 1.0e-12


def require_odd_grid(n):
    """Odd grids have no Nyquist index, so ``gB == grad B`` holds by construction."""
    if n % 2 == 0:
        raise ValueError(
            f"n_grid={n} is even. The Nyquist row k=n/2 has no representable derivative "
            "(fftfreq assigns k=-n/2 there and its conjugate partner shares the index), so the "
            "projection must discard it. Use an odd grid (e.g. 97) for the alanine study.")
    return int(n)


def project_bias(f1_sum, f2_sum, count, K1, K2, dz1, dz2, min_count, check=True, tol=1e-10):
    """Smoothed mean-force field -> conservative bias ``B`` and its gradient.

    Cells with less than ``min_count`` smoothed support contribute no force.  Returns
    ``(B, gB1, gB2, g1, g2)`` where ``(gB1, gB2)`` is guaranteed (and, with ``check``, asserted)
    to equal ``spectral_gradient(B)``.
    """
    g1, g2, den = d2.mean_force_fields(f1_sum, f2_sum, count, K1, K2)
    trust = den >= min_count
    g1 = torch.where(trust, g1, torch.zeros_like(g1))
    g2 = torch.where(trust, g2, torch.zeros_like(g2))
    B, gB1, gB2 = ps.poisson_projection(g1, g2, dz1, dz2)
    if check:
        s1, s2 = ps.spectral_gradient(B, dz1, dz2)
        err = max((gB1 - s1).abs().max().item(), (gB2 - s2).abs().max().item())
        if err > tol:
            raise AssertionError(
                f"projection returned gB != grad(B) (max abs {err:.3e} > {tol:.1e}); online and "
                "frozen-bias runs would apply different fields")
    return B, gB1, gB2, g1, g2


def clip_magnitude(b1, b2, clip):
    """Limit the magnitude of the 2-vector ``(b1, b2)``, preserving its direction.

    Per-component clipping would reintroduce a curl after the projection removed it.
    """
    if clip is None or clip <= 0:
        return b1, b2
    mag = torch.sqrt(b1 * b1 + b2 * b2).clamp_min(EPS)
    scale = torch.clamp(clip / mag, max=1.0)
    return b1 * scale, b2 * scale


def bias_at_particles(gB1, gB2, g1c, g2c, dz1, dz2, phi1, phi2, clip=None):
    """Interpolate the projected bias gradient to particles, with magnitude clipping."""
    b1 = d2.bilinear_interp2(gB1, g1c, g2c, dz1, dz2, phi1, phi2)
    b2 = d2.bilinear_interp2(gB2, g1c, g2c, dz1, dz2, phi1, phi2)
    return clip_magnitude(b1, b2, clip)
