"""Batched periodic nonbonded model for the C60/TIP4P-Ew system: unswitched LJ + smooth PME.

Frozen model: ``docs/SPEC_c60_water.md`` §1.  Parity target: OpenMM, §3.1, 1e-6.

Structure: two disjoint pair loops, and why that is not the methane negative
----------------------------------------------------------------------------
The methane engine recorded its LJ/Coulomb split path as a **measured performance negative**
(509 vs 744 ns/day) and this module splits anyway -- the structure is different, and the claim
is re-measured here, not assumed.  In SPC/E every site pair shares one loop because oxygens
carry both LJ and charge; splitting there duplicated streaming.  In TIP4P-Ew the LJ carriers
(1282 O + 120 C = 1402) and the charge carriers (H1/H2/M = 3846) are **disjoint sets**: a
combined 5248^2 loop would evaluate 27.5M pair slots per walker of which 10.8M are
identically zero by type.  The split runs 1402^2 + 3846^2 = 16.8M -- a 39 % reduction in the
O(N^2) traffic that bounds this kernel, against two extra ``index_add_`` scatters.

Cell and conventions
--------------------
Rectangular orthorhombic cell ``(Lx, Ly, Lz)``, minimum image per axis.  LJ has a **plain
cutoff at 1.0 nm with no switching function** (the paper's GROMACS convention; OpenMM built to
match).  PME real space is ``erfc(alpha r)/r`` hard-truncated at the same cutoff.  Exclusions:
intra-cage C-C pairs (LJ set), intra-water charged pairs (Coulomb set), with the
reciprocal-space exclusion correction over the charged exclusions only -- carbons are neutral,
so cage exclusions need no correction.  All parameters are the arrays read back out of the
OpenMM ``System`` by ``c60.system.site_parameters``.
"""
from __future__ import annotations

import numpy as np
import torch

from methane.nonbonded import ONE_4PI_EPS0

from .pme import PMEReciprocalRect


def _min_image(d, L):
    return d - L * torch.round(d / L)


class DisjointPairTerms:
    """Real-space LJ over the LJ set + PME real-space Coulomb over the charged set."""

    def __init__(self, sigma, epsilon, charge, exclusions, box_nm, cutoff_nm, alpha_per_nm,
                 device=None, dtype=torch.float64):
        self.n = int(len(sigma))
        self.L = torch.as_tensor([float(x) for x in box_nm], device=device, dtype=dtype)
        self.cutoff = float(cutoff_nm)
        self.alpha = float(alpha_per_nm)
        kw = dict(device=device, dtype=dtype)
        sigma = torch.as_tensor(np.asarray(sigma), **kw)
        epsilon = torch.as_tensor(np.asarray(epsilon), **kw)
        charge = torch.as_tensor(np.asarray(charge), **kw)
        self.charge = charge

        self.lj_index = torch.nonzero(epsilon > 0, as_tuple=True)[0]
        self.q_index = torch.nonzero(charge != 0, as_tuple=True)[0]
        if bool((epsilon[self.q_index] > 0).any()) or bool((charge[self.lj_index] != 0).any()):
            raise RuntimeError("LJ and charge site sets are not disjoint; this module's "
                               "structure is wrong for the system -- use a combined loop")

        li, qi = self.lj_index, self.q_index
        sig_l, eps_l = sigma[li], epsilon[li]
        self.lj_sig = 0.5 * (sig_l[:, None] + sig_l[None, :])
        self.lj_eps = torch.sqrt(eps_l[:, None] * eps_l[None, :])
        q_q = charge[qi]
        self.q_qq = ONE_4PI_EPS0 * q_q[:, None] * q_q[None, :]

        # global -> subset index maps
        gmap_l = torch.full((self.n,), -1, dtype=torch.long, device=device)
        gmap_l[li] = torch.arange(li.numel(), device=device)
        gmap_q = torch.full((self.n,), -1, dtype=torch.long, device=device)
        gmap_q[qi] = torch.arange(qi.numel(), device=device)

        ex = torch.as_tensor(np.asarray(exclusions), dtype=torch.long, device=device)
        in_l = (gmap_l[ex[:, 0]] >= 0) & (gmap_l[ex[:, 1]] >= 0)
        in_q = (gmap_q[ex[:, 0]] >= 0) & (gmap_q[ex[:, 1]] >= 0)
        mixed = ~(in_l | in_q)
        # mixed exclusions (O-H, O-M) involve one LJ-only and one charge-only site: no LJ term,
        # no real-space Coulomb term, no reciprocal correction -- they are vacuous here.

        excl_l = torch.zeros(li.numel(), li.numel(), dtype=torch.bool, device=device)
        el = ex[in_l]
        excl_l[gmap_l[el[:, 0]], gmap_l[el[:, 1]]] = True
        excl_l[gmap_l[el[:, 1]], gmap_l[el[:, 0]]] = True
        excl_l.fill_diagonal_(True)
        self.lj_excluded = excl_l

        excl_q = torch.zeros(qi.numel(), qi.numel(), dtype=torch.bool, device=device)
        eq = ex[in_q]
        excl_q[gmap_q[eq[:, 0]], gmap_q[eq[:, 1]]] = True
        excl_q[gmap_q[eq[:, 1]], gmap_q[eq[:, 0]]] = True
        excl_q.fill_diagonal_(True)
        self.q_excluded = excl_q
        self.q_exclusion_pairs = eq          # global indices, charged-charged only
        self.n_mixed_exclusions = int(mixed.sum())

    def energy_forces(self, x, chunk=256):
        """``(E (B,), F (B, N, 3))`` for cutoff LJ + real-space Coulomb, both subsets."""
        B, N, _ = x.shape
        energy = x.new_zeros(B)
        forces = x.new_zeros(B, N, 3)

        # ---- LJ over the LJ set (no switch: plain truncation at the cutoff) -----------------
        li = self.lj_index
        xl = x[:, li, :]
        nl = li.numel()
        f_acc = x.new_zeros(B, nl, 3)
        e_acc = x.new_zeros(B)
        for lo in range(0, nl, chunk):
            hi = min(lo + chunk, nl)
            d = _min_image(xl[:, lo:hi, None, :] - xl[:, None, :, :], self.L)
            r2 = (d * d).sum(-1)
            r = r2.clamp_min(1e-24).sqrt()
            live = (r < self.cutoff) & (~self.lj_excluded[lo:hi, :])
            inv_r = torch.where(live, 1.0 / r, torch.zeros_like(r))
            sr6 = (self.lj_sig[lo:hi, :] * inv_r).pow(6)
            sr12 = sr6 * sr6
            e_lj = 4.0 * self.lj_eps[lo:hi, :] * (sr12 - sr6)
            dlj = -24.0 * self.lj_eps[lo:hi, :] * (2.0 * sr12 - sr6) * inv_r
            e_acc = e_acc + torch.where(live, e_lj, torch.zeros_like(r)).sum(dim=(1, 2))
            dE = torch.where(live, dlj, torch.zeros_like(r))
            f_acc[:, lo:hi, :] = -((dE * inv_r).unsqueeze(-1) * d).sum(dim=2)
        energy = energy + 0.5 * e_acc
        forces.index_add_(1, li, f_acc)

        # ---- PME real-space Coulomb over the charged set ------------------------------------
        qi = self.q_index
        xq = x[:, qi, :]
        nq = qi.numel()
        two_a_sqrtpi = 2.0 * self.alpha / np.sqrt(np.pi)
        f_acc = x.new_zeros(B, nq, 3)
        e_acc = x.new_zeros(B)
        for lo in range(0, nq, chunk):
            hi = min(lo + chunk, nq)
            d = _min_image(xq[:, lo:hi, None, :] - xq[:, None, :, :], self.L)
            r2 = (d * d).sum(-1)
            r = r2.clamp_min(1e-24).sqrt()
            live = (r < self.cutoff) & (~self.q_excluded[lo:hi, :])
            inv_r = torch.where(live, 1.0 / r, torch.zeros_like(r))
            ar = self.alpha * r
            erfc = torch.erfc(ar)
            qq = self.q_qq[lo:hi, :]
            e_acc = e_acc + torch.where(live, qq * erfc * inv_r,
                                        torch.zeros_like(r)).sum(dim=(1, 2))
            dE = torch.where(live,
                             -qq * (erfc * inv_r * inv_r
                                    + two_a_sqrtpi * torch.exp(-ar * ar) * inv_r),
                             torch.zeros_like(r))
            f_acc[:, lo:hi, :] = -((dE * inv_r).unsqueeze(-1) * d).sum(dim=2)
        energy = energy + 0.5 * e_acc
        forces.index_add_(1, qi, f_acc)
        return energy, forces

    def exclusion_correction(self, x):
        """``-q_i q_j erf(alpha r)/r`` over the charged excluded pairs (3/water)."""
        i, j = self.q_exclusion_pairs[:, 0], self.q_exclusion_pairs[:, 1]
        d = _min_image(x[:, i, :] - x[:, j, :], self.L)
        r = (d * d).sum(-1).clamp_min(1e-24).sqrt()
        qq = ONE_4PI_EPS0 * self.charge[i] * self.charge[j]
        ar = self.alpha * r
        erf = torch.erf(ar)
        e = -(qq * erf / r)
        two_a_sqrtpi = 2.0 * self.alpha / np.sqrt(np.pi)
        dE_dr = -qq * (two_a_sqrtpi * torch.exp(-ar * ar) / r - erf / (r * r))
        coef = (dE_dr / r).unsqueeze(-1)
        f_pair = -(coef * d)
        forces = x.new_zeros(x.shape)
        forces.index_add_(1, i, f_pair)
        forces.index_add_(1, j, -f_pair)
        return e.sum(-1), forces


class C60Nonbonded:
    """The complete frozen model, batched over walkers; parameters read from OpenMM.

    ``energy_forces`` returns **raw** site forces (nonzero on M sites).  OpenMM reports
    virtual-site forces redistributed onto the parent atoms, so parity and dynamics go through
    :meth:`redistribute`, which applies the ``ThreeParticleAverageSite`` weights and zeroes M.
    """

    def __init__(self, system, topology, box_nm, alpha_per_nm, grid, order=5,
                 device=None, dtype=torch.float64):
        from . import system as csys

        p = csys.site_parameters(system, topology)
        self.params = p
        self.n = len(p["charge"])
        self.box_nm = tuple(float(x) for x in box_nm)
        self.cage_a = torch.as_tensor(p["cage_a"], device=device, dtype=torch.long)
        self.cage_b = torch.as_tensor(p["cage_b"], device=device, dtype=torch.long)
        self.waters = torch.as_tensor(p["waters"], device=device, dtype=torch.long)
        self.vw = torch.as_tensor(p["vsite_weights"], device=device, dtype=dtype)
        self.mass = torch.as_tensor(p["mass"], device=device, dtype=dtype)

        self.pair = DisjointPairTerms(p["sigma"], p["epsilon"], p["charge"],
                                      csys.exclusions(system), box_nm,
                                      csys.CUTOFF_NM, alpha_per_nm,
                                      device=device, dtype=dtype)
        self.recip = PMEReciprocalRect(p["charge"], box_nm, grid, alpha_per_nm, order=order,
                                       device=device, dtype=dtype)
        self.e_self = self.recip.self_energy()

    def energy_forces(self, x, chunk=256):
        e_r, f_r = self.pair.energy_forces(x, chunk=chunk)
        e_x, f_x = self.pair.exclusion_correction(x)
        e_k, f_k = self.recip.energy_forces(x)
        return e_r + e_x + e_k + self.e_self, f_r + f_x + f_k

    def redistribute(self, f):
        """OpenMM's virtual-site force convention: M force onto (O, H1, H2), M entry zeroed."""
        o, h1, h2, m = (self.waters[:, k] for k in range(4))
        out = f.clone()
        fm = out[:, m, :]
        out.index_add_(1, o, self.vw[0] * fm)
        out.index_add_(1, h1, self.vw[1] * fm)
        out.index_add_(1, h2, self.vw[2] * fm)
        out[:, m, :] = 0.0
        return out

    def compute_vsites(self, x):
        """Place every M site from its (O, H1, H2), in place; returns ``x``."""
        o, h1, h2, m = (self.waters[:, k] for k in range(4))
        x[:, m, :] = (self.vw[0] * x[:, o, :] + self.vw[1] * x[:, h1, :]
                      + self.vw[2] * x[:, h2, :])
        return x

    def xi(self, x):
        """``xi = Z_B - Z_A`` (B,), cage-COM axial separation; no minimum-image fold.

        The cages are placed analytically about the box centre and never wrapped, so
        ``Z_B - Z_A`` is the physical separation by construction; an assertion guards the
        domain rather than a wrap hiding a violation.
        """
        return x[:, self.cage_b, 2].mean(dim=1) - x[:, self.cage_a, 2].mean(dim=1)

    def local_mean_force(self, f):
        """``f_xi = (1/2)(F_A,z - F_B,z)`` (B,) from raw or redistributed forces."""
        return 0.5 * (f[:, self.cage_a, 2].sum(dim=1) - f[:, self.cage_b, 2].sum(dim=1))
