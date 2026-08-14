"""Orthogonal solvent descriptors for the C60 pair -- above all ``n_gap`` (SPEC §4, frozen).

``n_gap`` is the smooth count of water oxygens in the inter-fullerene gap, with the radial
extent frozen from the paper's own I2-water analysis region: a cylinder of radius **0.62 nm**
in the xy-plane centred on the line joining the two cage COMs (Zangi 2014, Figure 3).  The
axis is ``z`` by construction (SPEC §1.3), which makes the geometry simpler than methane's:

    u_j = z_Oj - z_mid          (axial offset, minimum image along z)
    w_j = |xy_Oj - xy_center|   (lateral offset, minimum image in x and y)

    n_gap = sum_j s(|u_j|; xi/2) * s(w_j; R_CYL)

with the standard rational switch ``s(x; x0) = 1/(1 + (x/x0)^6)`` of ``methane.observables``
(consumed unmodified).  The axial half-width is ``xi/2`` -- the cylinder ends at the cage
centres -- so the descriptor means the same thing at every separation, as for methane.

``n_shell`` (diagnostic only, never a gate input) counts oxygens within the paper's I1+I2
interfacial band: 1.082 nm of either cage COM (Figure 3's bulk/interfacial boundary).
"""
from __future__ import annotations

import numpy as np

from methane.observables import _switch

R_CYL_NM = 0.62            #: frozen: the paper's I2 cylinder radius (Zangi 2014 Fig. 3)
R_SHELL_NM = 1.082         #: the paper's interfacial boundary (two water layers)


def gap_geometry(pos, cage_a, cage_b, oxygen_idx, box_nm):
    """``(u, w, xi)`` for one configuration; axial/lateral offsets under minimum image."""
    pos = np.asarray(pos, dtype=np.float64)
    L = np.asarray(box_nm, dtype=np.float64)

    com_a = pos[cage_a].mean(axis=0)
    com_b = pos[cage_b].mean(axis=0)
    xi = float(com_b[2] - com_a[2])
    center = 0.5 * (com_a + com_b)

    off = pos[oxygen_idx] - center
    off -= L * np.round(off / L)
    u = off[:, 2]
    w = np.sqrt(off[:, 0] ** 2 + off[:, 1] ** 2)
    return u, w, xi, off


def n_gap(pos, cage_a, cage_b, oxygen_idx, box_nm, r_cyl_nm=R_CYL_NM):
    """Smooth count of water oxygens in the inter-cage gap (SPEC §4)."""
    u, w, xi, _ = gap_geometry(pos, cage_a, cage_b, oxygen_idx, box_nm)
    return float((_switch(np.abs(u), 0.5 * xi) * _switch(w, r_cyl_nm)).sum())


def n_shell(pos, cage_a, cage_b, oxygen_idx, box_nm, r_shell_nm=R_SHELL_NM):
    """Smooth count of oxygens within the interfacial band of either cage (diagnostic)."""
    pos = np.asarray(pos, dtype=np.float64)
    L = np.asarray(box_nm, dtype=np.float64)
    out = np.zeros(2)
    for k, com in enumerate((pos[cage_a].mean(axis=0), pos[cage_b].mean(axis=0))):
        off = pos[oxygen_idx] - com
        off -= L * np.round(off / L)
        d = np.linalg.norm(off, axis=1)
        out[k] = _switch(d, r_shell_nm).sum()
    return float(out.sum())


def n_gap_batch(pos, cage_a, cage_b, oxygen_idx, box_nm, r_cyl_nm=R_CYL_NM):
    """``n_gap`` for ``(B, N, 3)`` -> ``(B,)``."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim == 2:
        return np.asarray([n_gap(pos, cage_a, cage_b, oxygen_idx, box_nm, r_cyl_nm)])
    return np.asarray([n_gap(p, cage_a, cage_b, oxygen_idx, box_nm, r_cyl_nm) for p in pos])
