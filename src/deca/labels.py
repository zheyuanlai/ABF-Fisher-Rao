"""Structural labels ``Y`` for deca-alanine: the orthogonal descriptors Gate A and the
conditional-fidelity metric are computed from.

The whole deca-alanine question is whether the *hidden* conformational structure the literature
associates with slow convergence -- compact configurations in parallel valleys -- is visible in
the end-to-end distance.  That question is only as good as the labels, so they are defined here
once, frozen, and used identically by the reference build, the screen and every production arm.

All functions are batched torch on ``(B, 112, 3)`` in nm and return ``(B,)`` or ``(B, n_res)``.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .system import N_RES, atom_index, build_helix

#: i -> i+4 backbone hydrogen bond: O of residue i to amide H of residue i+4.
HBOND_CUTOFF_NM = 0.25

#: Backbone basin boundaries in degrees.  Deliberately coarse -- these separate the
#: right-handed helical region, the extended/beta region and the left-handed region, and
#: nothing finer is claimed.
ALPHA_PHI = (-160.0, -20.0)
ALPHA_PSI = (-120.0, 50.0)


def _dihedral_t(x, i, j, k, l):
    """Signed dihedral in radians for index arrays ``i,j,k,l``; ``x`` is ``(B, A, 3)``."""
    p0, p1, p2, p3 = x[:, i], x[:, j], x[:, k], x[:, l]
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    return torch.atan2((torch.linalg.cross(b1n, v, dim=-1) * w).sum(-1), (v * w).sum(-1))


class DecaLabels:
    """Frozen structural descriptors.  Build once per run; all index tensors are cached."""

    def __init__(self, n_res=N_RES, device="cuda", dtype=torch.float64):
        I = atom_index(n_res)
        self.n_res = n_res
        self.device, self.dtype = device, dtype
        L = lambda a: torch.as_tensor(a, device=device, dtype=torch.long)      # noqa: E731

        # --- backbone (phi, psi) index arrays, one entry per alanine ---
        c_prev, n_i, ca_i, c_i, n_next = [], [], [], [], []
        for r in range(1, n_res + 1):
            c_prev.append(I[(0, "C")] if r == 1 else I[(r - 1, "C")])
            n_i.append(I[(r, "N")])
            ca_i.append(I[(r, "CA")])
            c_i.append(I[(r, "C")])
            n_next.append(I[(r + 1, "N")] if r < n_res else I[(n_res + 1, "N")])
        self.c_prev, self.n_i, self.ca_i, self.c_i, self.n_next = map(
            L, (c_prev, n_i, ca_i, c_i, n_next))

        # --- i -> i+4 hydrogen bonds.  Donor H lives on residue i+4; acceptor O on residue i.
        #     Residue n_res+1 is NME, which has an amide H, so i+4 may reach the cap.
        don, acc = [], []
        for r in range(1, n_res + 1):
            s = r + 4
            if s <= n_res + 1:
                acc.append(I[(r, "O")])
                don.append(I[(s, "H")])
        self.hb_acc, self.hb_don = L(acc), L(don)
        self.n_hbond_pairs = len(acc)

        # --- masses for the radius of gyration, and the ideal-helix RMSD target ---
        from .system import make_system
        from alanine.forcefield import extract_parameters
        _, _, system = make_system(n_res)
        m = extract_parameters(system)["masses"]
        self.masses = torch.as_tensor(m, device=device, dtype=dtype)
        self.m_tot = self.masses.sum()
        ref = build_helix(-57.0, -47.0, n_res=n_res)
        self.helix_ref = torch.as_tensor(ref, device=device, dtype=dtype)
        self.ca_idx = L([I[(r, "CA")] for r in range(1, n_res + 1)])

    # ------------------------------------------------------------------ individual labels
    def phi_psi(self, x):
        """``(phi, psi)`` in radians, each ``(B, n_res)``."""
        phi = _dihedral_t(x, self.c_prev, self.n_i, self.ca_i, self.c_i)
        psi = _dihedral_t(x, self.n_i, self.ca_i, self.c_i, self.n_next)
        return phi, psi

    def n_helical_hbonds(self, x):
        """Count of ``i -> i+4`` backbone hydrogen bonds under a distance cutoff.  ``(B,)``."""
        d = (x[:, self.hb_acc] - x[:, self.hb_don]).norm(dim=-1)
        return (d < HBOND_CUTOFF_NM).sum(-1)

    def alpha_fraction(self, x):
        """Fraction of residues in the right-handed helical basin.  ``(B,)``."""
        phi, psi = self.phi_psi(x)
        phid, psid = torch.rad2deg(phi), torch.rad2deg(psi)
        inA = ((phid > ALPHA_PHI[0]) & (phid < ALPHA_PHI[1])
               & (psid > ALPHA_PSI[0]) & (psid < ALPHA_PSI[1]))
        return inA.to(x.dtype).mean(-1)

    def radius_of_gyration(self, x):
        """Mass-weighted radius of gyration in nm.  ``(B,)``."""
        w = self.masses[None, :, None]
        com = (w * x).sum(1, keepdim=True) / self.m_tot
        d2 = ((x - com) ** 2).sum(-1)
        return torch.sqrt((self.masses[None] * d2).sum(-1) / self.m_tot)

    def ca_rmsd_to_helix(self, x):
        """Optimally superposed CA RMSD to the ideal ``(-57, -47)`` helix, in nm.  ``(B,)``.

        Kabsch superposition -- without it the value would be dominated by rigid-body motion
        rather than by conformation, which is the opposite of what the label is for.
        """
        P = x[:, self.ca_idx]
        Q = self.helix_ref[self.ca_idx][None].expand_as(P)
        P = P - P.mean(1, keepdim=True)
        Q = Q - Q.mean(1, keepdim=True)
        H = P.transpose(1, 2) @ Q
        U, _, Vh = torch.linalg.svd(H)
        d = torch.sign(torch.linalg.det(U @ Vh))
        D = torch.diag_embed(torch.stack(
            [torch.ones_like(d), torch.ones_like(d), d], dim=-1))
        R = U @ D @ Vh
        return ((P @ R - Q) ** 2).sum(-1).mean(-1).sqrt()

    # ------------------------------------------------------------------ the Gate A label
    def gate_a_label(self, x, hb_edges=(0.5, 3.5), af_edges=(0.35, 0.75)):
        """The frozen composite label ``Y`` for Gate A.  ``(B,)`` integer in ``[0, 9)``.

        ``Y = 3 * hbond_bucket + alpha_bucket`` with three buckets each:
        few / some / many ``i -> i+4`` hydrogen bonds, crossed with low / mixed / high helical
        fraction.  Coarse on purpose: Gate A asks whether *structurally distinct* states are
        separated in ``xi``, and a fine partition would manufacture separation out of noise.
        """
        hb = self.n_helical_hbonds(x).to(self.dtype)
        af = self.alpha_fraction(x)
        hb_b = (hb > hb_edges[0]).long() + (hb > hb_edges[1]).long()
        af_b = (af > af_edges[0]).long() + (af > af_edges[1]).long()
        return 3 * hb_b + af_b

    def all_labels(self, x):
        """Every descriptor at once, as a dict of ``(B,)`` tensors."""
        return dict(n_hbonds=self.n_helical_hbonds(x).to(self.dtype),
                    alpha_frac=self.alpha_fraction(x),
                    rg=self.radius_of_gyration(x),
                    ca_rmsd_helix=self.ca_rmsd_to_helix(x),
                    y=self.gate_a_label(x).to(self.dtype))


N_GATE_A_STATES = 9


def conditional_tv(xi, y, weights, grid_edges, n_states=N_GATE_A_STATES, min_count=200.0):
    """Pairwise total-variation distances between ``p(xi | Y = a)`` on a fixed grid.

    This is the Gate A statistic.  ``xi`` ``(M,)``, ``y`` ``(M,)`` integer labels, ``weights``
    ``(M,)`` non-negative (MBAR weights for a reference ensemble, or ones for a raw one),
    ``grid_edges`` ``(n_bins+1,)``.

    Returns ``(tv, occupancy, p_cond)``: the ``(n_states, n_states)`` TV matrix with ``nan``
    where either state is too sparse to compare, the per-state weighted occupancy, and the
    ``(n_states, n_bins)`` conditional densities.  States with weighted occupancy below
    ``min_count`` are excluded rather than compared on noise.
    """
    xi = np.asarray(xi, float).ravel()
    y = np.asarray(y).ravel().astype(int)
    w = np.asarray(weights, float).ravel()
    edges = np.asarray(grid_edges, float)
    nb = len(edges) - 1

    # Drop out-of-range samples rather than clamping them into the edge bins.  The umbrella
    # windows deliberately bracket the evaluation domain (Amendment 1), so samples outside it
    # exist by design; clamping them piles that mass onto the two end bins and distorts every
    # conditional density there.  Same defect that carved a fake basin into the reference PMF.
    inside = (xi >= edges[0]) & (xi <= edges[-1])
    xi, y, w = xi[inside], y[inside], w[inside]
    idx = np.clip(np.digitize(xi, edges) - 1, 0, nb - 1)

    p = np.zeros((n_states, nb))
    occ = np.zeros(n_states)
    for a in range(n_states):
        m = y == a
        if not m.any():
            continue
        occ[a] = w[m].sum()
        np.add.at(p[a], idx[m], w[m])
    tot = p.sum(1, keepdims=True)
    live = (occ >= min_count) & (tot.ravel() > 0)
    p_cond = np.divide(p, tot, out=np.zeros_like(p), where=tot > 0)

    tv = np.full((n_states, n_states), np.nan)
    for a in range(n_states):
        for b in range(n_states):
            if a != b and live[a] and live[b]:
                tv[a, b] = 0.5 * np.abs(p_cond[a] - p_cond[b]).sum()
    return tv, occ, p_cond
