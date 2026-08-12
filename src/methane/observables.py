"""Orthogonal descriptors for the methane pair -- above all ``n_gap`` (SPEC §5.1, frozen).

At fixed ``r`` the collective variable sees nothing about the water, and Amendment 8 proves that
no ``xi``-only selection score can act on ``p(y | xi)`` at the mean-field level.  The study
therefore needs an explicit orthogonal coordinate, and the physically motivated one is the water
occupancy of the inter-methane gap:

    n_gap ~ 0   dry, contact-like gap
    n_gap > 0   water inserted between the methanes

It is used for three separate jobs, which is why it is frozen once here rather than defined at
each use site: Gate A's CV-visibility test, ``tau_perp`` from the fixed-``r`` conditional-mixing
experiment (SPEC §5.2), and the conditional-fidelity endpoint ``TV(p(n_gap|r))``.

Definition, fixed in advance and **not** tuned
----------------------------------------------
With ``m`` the methane midpoint and ``e`` the unit vector along the pair, decompose each water
oxygen's offset ``d = O_j - m`` into an axial part ``u = d.e`` and a radial part ``w = |d - u e|``:

    n_gap = sum_j  s(|u_j| ; r/2)  *  s(w_j ; R_cyl)

with the SPC/E-standard rational switch ``s(x; x0) = (1 - (x/x0)^6) / (1 - (x/x0)^12)`` and
``R_cyl = 0.20 nm``.  A *smooth* count is used rather than a hard one so the descriptor is
differentiable and its time correlation is not dominated by boundary crossings -- the correlation
time is exactly what ``tau_perp`` measures.

The axial half-width is ``r/2``, i.e. the cylinder ends on the two methane centres, so the
descriptor means the same thing at every separation.
"""
from __future__ import annotations

import numpy as np

R_CYL_NM = 0.20           #: frozen; never tuned (SPEC §5.1)
EPS = 1.0e-12


def _switch(x, x0):
    """``(1 - (x/x0)^6) / (1 - (x/x0)^12)``, the standard rational coordination switch.

    Equals ``1`` at ``x = 0``, ``1/2`` at ``x = x0``, and decays smoothly beyond.  Evaluated as
    ``1 / (1 + (x/x0)^6)``, which is algebraically identical and free of the ``0/0`` at ``x = x0``.
    """
    t = (np.asarray(x, dtype=np.float64) / max(float(x0), EPS)) ** 6
    return 1.0 / (1.0 + t)


def gap_geometry(pos, methane_idx, oxygen_idx, box_nm):
    """``(u, w, r)`` -- axial offset, radial offset and pair separation, minimum image."""
    pos = np.asarray(pos, dtype=np.float64)
    i, j = int(methane_idx[0]), int(methane_idx[1])
    L = float(box_nm)

    d = pos[j] - pos[i]
    d -= L * np.round(d / L)
    r = float(np.linalg.norm(d))
    e = d / max(r, EPS)
    mid = pos[i] + 0.5 * d

    off = pos[oxygen_idx] - mid
    off -= L * np.round(off / L)
    u = off @ e
    w = np.linalg.norm(off - u[:, None] * e[None, :], axis=1)
    return u, w, r


def n_gap(pos, methane_idx, oxygen_idx, box_nm, r_cyl_nm=R_CYL_NM):
    """Smooth count of water oxygens in the inter-methane gap (SPEC §5.1)."""
    u, w, r = gap_geometry(pos, methane_idx, oxygen_idx, box_nm)
    return float((_switch(np.abs(u), 0.5 * r) * _switch(w, r_cyl_nm)).sum())


def n_gap_batch(pos, methane_idx, oxygen_idx, box_nm, r_cyl_nm=R_CYL_NM):
    """``n_gap`` for a batch of configurations ``(B, N, 3)`` -> ``(B,)``."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim == 2:
        return np.asarray([n_gap(pos, methane_idx, oxygen_idx, box_nm, r_cyl_nm)])
    return np.asarray([n_gap(p, methane_idx, oxygen_idx, box_nm, r_cyl_nm) for p in pos])


def pair_distance(pos, methane_idx, box_nm):
    """The collective variable ``xi = |Q_2 - Q_1|`` under minimum image."""
    pos = np.asarray(pos, dtype=np.float64)
    i, j = int(methane_idx[0]), int(methane_idx[1])
    d = pos[..., j, :] - pos[..., i, :]
    d -= float(box_nm) * np.round(d / float(box_nm))
    return np.linalg.norm(d, axis=-1)
