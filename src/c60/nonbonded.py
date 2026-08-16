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


class SubsetNeighborList:
    """Fixed-capacity per-subset Verlet list, rectangular cell, int32 -- the H200 fix.

    Why this exists (and why the methane negative does not transfer): the all-pairs kernel is
    MEMORY-bound, streaming (B, chunk, N) intermediates of which ~90 % are beyond the cutoff
    at THIS box (cutoff sphere 4.19 nm^3 vs cell 39.9 nm^3 = 10.5 %; methane's box was 2.5
    cutoffs across and culled only 1.8x).  Measured before this change: 402 ms/step at
    B = 816 = ~3 % of the device's fp32 throughput.  The list removes the dead traffic at its
    source.  Capacity is fixed (int32, headroom asserted at build); a rebuild that overflows
    RAISES rather than silently dropping pairs, and a drift beyond half the skin RAISES
    rather than aging the list.
    """

    def __init__(self, index, box_nm, r_list_nm, m_cap, device=None):
        self.index = index                       # (n_sub,) long, global site ids
        # float64 ALWAYS, cast at use: a dtype-defaulted (float32) box truncates the cell at
        # ~1e-7 relative and shifts every wrapped pair by ~3e-7 nm -- invisible in energy,
        # ~1e-3 in close-pair forces (measured on all 3846 sites before this line existed)
        self.L = torch.as_tensor([float(x) for x in box_nm], dtype=torch.float64,
                                 device=device)
        self.r_list = float(r_list_nm)
        self.m_cap = int(m_cap)
        self.nbr = None                          # (B, n_sub, M) int32, subset-local ids
        self.count = None                        # (B, n_sub) int32
        self.x_built = None

    def rebuild(self, x_sub, excluded_sub, chunk=128):
        """``x_sub``: (B, n_sub, 3) positions of the subset's sites."""
        B, n, _ = x_sub.shape
        nbr = torch.zeros(B, n, self.m_cap, dtype=torch.int32, device=x_sub.device)
        count = torch.zeros(B, n, dtype=torch.int32, device=x_sub.device)
        L = self.L.to(x_sub.dtype)
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            d = x_sub[:, lo:hi, None, :] - x_sub[:, None, :, :]
            d = d - L * torch.round(d / L)
            m = ((d * d).sum(-1) < self.r_list ** 2) & (~excluded_sub[lo:hi, :])
            c = m.sum(-1, dtype=torch.int32)
            if int(c.max()) > self.m_cap:
                raise RuntimeError(f"neighbor overflow: {int(c.max())} > cap {self.m_cap}")
            # stable sort brings True entries first, in index order
            order = torch.argsort((~m).to(torch.uint8), dim=-1, stable=True)[..., :self.m_cap]
            nbr[:, lo:hi] = order.to(torch.int32)
            count[:, lo:hi] = c
        self.nbr, self.count = nbr, count
        self.x_built = x_sub.detach().clone()
        return int(count.max())

    def drift_ok(self, x_sub, half_skin):
        d = x_sub - self.x_built
        L = self.L.to(x_sub.dtype)
        d = d - L * torch.round(d / L)
        return float(d.norm(dim=-1).max()) < half_skin


def _nl_pair_terms(x_sub, nl, sig=None, eps=None, q=None, cutoff=1.0, alpha=None,
                   chunk=384):
    """Energy (B,) + forces (B, n_sub, 3) over a subset neighbor list, chunked over rows.

    Full (not half) lists: every ordered pair appears once on each side, so energies are
    halved and each row's force sum is already complete (the methane convention).
    Parameters are combined from per-site vectors -- the (N, N) tables would re-create the
    memory traffic the list removes.  Chunking bounds intermediates at (B, chunk, M).
    """
    B, n, M = nl.nbr.shape
    L = nl.L.to(x_sub.dtype)
    arangeM = torch.arange(M, device=x_sub.device)
    energy = x_sub.new_zeros(B)
    forces = x_sub.new_zeros(B, n, 3)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        j = nl.nbr[:, lo:hi].long()                                    # (B, c, M)
        valid = arangeM[None, None, :] < nl.count[:, lo:hi, None].long()
        xj = torch.gather(x_sub, 1, j.reshape(B, -1, 1).expand(-1, -1, 3)) \
            .view(B, hi - lo, M, 3)
        d = x_sub[:, lo:hi, None, :] - xj
        d = d - L * torch.round(d / L)
        r = (d * d).sum(-1).clamp_min(1e-24).sqrt()
        live = valid & (r < cutoff)
        inv_r = torch.where(live, 1.0 / r, torch.zeros_like(r))

        e = torch.zeros_like(r)
        dE = torch.zeros_like(r)
        if eps is not None:
            sig_j = torch.gather(sig.expand(B, -1), 1, j.reshape(B, -1)).view(B, hi - lo, M)
            eps_j = torch.gather(eps.expand(B, -1), 1, j.reshape(B, -1)).view(B, hi - lo, M)
            sig_ij = 0.5 * (sig[lo:hi][None, :, None] + sig_j)
            eps_ij = torch.sqrt(eps[lo:hi][None, :, None] * eps_j)
            sr6 = (sig_ij * inv_r).pow(6)
            sr12 = sr6 * sr6
            e = e + 4.0 * eps_ij * (sr12 - sr6)
            dE = dE - 24.0 * eps_ij * (2.0 * sr12 - sr6) * inv_r
        if q is not None:
            q_j = torch.gather(q.expand(B, -1), 1, j.reshape(B, -1)).view(B, hi - lo, M)
            qq = ONE_4PI_EPS0 * q[lo:hi][None, :, None] * q_j
            ar = alpha * r
            erfc = torch.erfc(ar)
            two_a_sqrtpi = 2.0 * alpha / np.sqrt(np.pi)
            e = e + qq * erfc * inv_r
            dE = dE - qq * (erfc * inv_r * inv_r
                            + two_a_sqrtpi * torch.exp(-ar * ar) * inv_r)

        e = torch.where(live, e, torch.zeros_like(e))
        dE = torch.where(live, dE, torch.zeros_like(dE))
        energy = energy + 0.5 * e.sum(dim=(1, 2))
        forces[:, lo:hi, :] = -((dE * inv_r).unsqueeze(-1) * d).sum(dim=2)
    return energy, forces


#: neighbor-list lifecycle constants -- skin sized so ~1-2 ps of water diffusion fits
NL_SKIN_NM = 0.15


class _FastPath:
    """Neighbor-list state and evaluation for C60Nonbonded (attached lazily)."""

    def __init__(self, eng, x, dtype):
        pt = eng.pair
        dev = x.device
        # per-site vectors (the diagonal of an LB table is the site's own value)
        self.sig_l = torch.diagonal(pt.lj_sig).to(dtype).contiguous()
        self.eps_l = torch.diagonal(pt.lj_eps).to(dtype).contiguous()
        self.q_q = pt.charge.to(dtype)[pt.q_index].contiguous()
        self.half_skin = 0.5 * NL_SKIN_NM
        r_list = pt.cutoff + NL_SKIN_NM
        self.nl_lj = SubsetNeighborList(pt.lj_index, eng.box_nm, r_list, 8, device=dev)
        self.nl_q = SubsetNeighborList(pt.q_index, eng.box_nm, r_list, 8, device=dev)
        # size caps from a SMALL-SLICE count pass: probing at full B with cap = n_sub
        # allocates a (B, n, n) argsort -- 90 GB at B = 816 (measured OOM).  Counts vary
        # little across walkers; 30 % headroom + the rebuild overflow-raise are the guard.
        for nl, idx, excl in ((self.nl_lj, pt.lj_index, pt.lj_excluded),
                              (self.nl_q, pt.q_index, pt.q_excluded)):
            xs = x[:, idx, :]
            probe = SubsetNeighborList(idx, eng.box_nm, r_list, xs.shape[1], device=dev)
            mx = probe.rebuild(xs[: min(4, xs.shape[0])].contiguous(), excl)
            del probe
            nl.m_cap = int(np.ceil(mx * 1.30 / 8.0) * 8)
            nl.rebuild(xs, excl)
        self.n_rebuilds = 2

    def energy_forces(self, eng, x):
        pt = eng.pair
        xl = x[:, pt.lj_index, :].contiguous()
        xq = x[:, pt.q_index, :].contiguous()
        if not (self.nl_lj.drift_ok(xl, self.half_skin)
                and self.nl_q.drift_ok(xq, self.half_skin)):
            self.nl_lj.rebuild(xl, pt.lj_excluded)
            self.nl_q.rebuild(xq, pt.q_excluded)
            self.n_rebuilds += 1
        e_l, f_l = _nl_pair_terms(xl, self.nl_lj, sig=self.sig_l, eps=self.eps_l,
                                  cutoff=pt.cutoff)
        e_q, f_q = _nl_pair_terms(xq, self.nl_q, q=self.q_q, cutoff=pt.cutoff,
                                  alpha=pt.alpha)
        forces = x.new_zeros(x.shape)
        forces.index_add_(1, pt.lj_index, f_l)
        forces.index_add_(1, pt.q_index, f_q)
        e_x, f_x = pt.exclusion_correction(x)
        e_k, f_k = eng.recip.energy_forces(x)
        return e_l + e_q + e_x + e_k + eng.e_self, forces + f_x + f_k


def _attach_fast(eng, x):
    if getattr(eng, "_fast", None) is None:
        eng._fast = _FastPath(eng, x, x.dtype)
    return eng._fast


def energy_forces_fast(eng, x):
    """Neighbor-list evaluation of the full model; identical physics to ``energy_forces``.

    Gated two ways before use: internally against the all-pairs path (< 1e-9 relative, the
    methane NL convention) by ``tests/test_c60_engine.py::test_neighbor_list_parity``, and
    externally by the same OpenMM parity suite the all-pairs path passes.  Rebuild-on-drift
    with a raise-on-overflow capacity; a stale list cannot silently drop pairs.
    """
    return _attach_fast(eng, x).energy_forces(eng, x)


C60Nonbonded.energy_forces_fast = energy_forces_fast
