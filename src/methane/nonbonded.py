"""Batched periodic nonbonded energy and forces: switched LJ + smooth PME electrostatics.

Frozen model: ``docs/SPEC_methane_water.md`` §1.  Parity target: OpenMM, §3.2, ``1e-6``.

Why this exists rather than a library
-------------------------------------
``torch-pme`` 0.5.0 was evaluated first, as Amendment 12.4 requires, and **rejected on a measured
fact**: its calculators take ``positions`` of shape ``(n_atoms, 3)`` and a single ``(3, 3)`` cell,
with no batch dimension (the ``node_mask``/``pair_mask`` arguments mask within one system).  The
ABF/mFR arms need ``B`` walkers stepped simultaneously -- a Python loop over ``B = 512`` PME calls
per timestep defeats the only reason the torch engine exists.  Everything here is therefore
written batched over a leading walker axis, and validated directly against OpenMM.

Layout and conventions
----------------------
Positions are ``(B, N, 3)`` in nm; every routine returns ``(energy (B,), forces (B, N, 3))`` in
kJ/mol and kJ/mol/nm.  The cell is a single cube of side ``L`` shared by every walker -- the
production ensemble is NVT at a frozen box (SPEC §1.3), so a per-walker cell would be dead weight.

Three OpenMM conventions are reproduced exactly, and each is the kind of detail that silently
costs a factor rather than an error message:

* **LJ switching.**  For ``r_switch < r < r_cut`` OpenMM multiplies the LJ energy by
  ``S(x) = 1 - 10x^3 + 15x^4 - 6x^5`` with ``x = (r - r_switch)/(r_cut - r_switch)``, and the
  force picks up the ``S'`` term.  The switch is applied to **LJ only**; the PME real-space term
  is truncated hard at the cutoff because ``erfc`` has already decayed.
* **Exclusions.**  The three intramolecular pairs of each water contribute no LJ and no
  real-space Coulomb, and their *reciprocal-space* contribution is removed by subtracting
  ``q_i q_j erf(alpha r)/r``.  Forgetting this correction is the single most common defect in a
  hand-written PME, so it is a separately gated term here.
* **Lorentz--Berthelot** unlike pairs, which OpenMM applies internally and which SPEC §1.1
  declares as our choice.

All-pairs, deliberately
-----------------------
No neighbour list.  At ``L ~ 2.6 nm`` with a ``1.05 nm`` cutoff the cutoff sphere is ~27 % of the
box, so a neighbour list saves under 4x while costing rebuilds, buffers and a class of
correctness bug that does not announce itself.  The pair loop is chunked over ``i`` to bound
memory instead.
"""
from __future__ import annotations

import numpy as np
import torch

#: OpenMM's ``ONE_4PI_EPS0`` in kJ nm / (mol e^2).  Matched to OpenMM's own constant, not to
#: CODATA, because the parity gate compares against OpenMM.
ONE_4PI_EPS0 = 138.93545764438198


def switch_and_derivative(r, r_switch, r_cut):
    """OpenMM's LJ switching function ``S`` and ``dS/dr``.

    ``S = 1 - 10x^3 + 15x^4 - 6x^5``, ``x = (r - r_switch)/(r_cut - r_switch)`` clamped to
    ``[0, 1]``.  ``S(0) = 1``, ``S(1) = 0``, and ``S'`` vanishes at both ends, so the force stays
    continuous.
    """
    width = r_cut - r_switch
    x = ((r - r_switch) / width).clamp(0.0, 1.0)
    x2 = x * x
    x3 = x2 * x
    s = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
    ds = (-30.0 * x2 + 60.0 * x3 - 30.0 * x3 * x) / width
    # outside the switching window the derivative is identically zero
    ds = torch.where((r > r_switch) & (r < r_cut), ds, torch.zeros_like(ds))
    return s, ds


class PairTerms:
    """Real-space LJ + PME real-space Coulomb over all pairs, minimum image, chunked."""

    def __init__(self, sigma, epsilon, charge, exclusions, box_nm,
                 cutoff_nm, switch_nm, alpha_per_nm, device=None, dtype=torch.float64):
        self.n = int(len(sigma))
        self.L = float(box_nm)
        self.cutoff = float(cutoff_nm)
        self.switch = float(switch_nm)
        self.alpha = float(alpha_per_nm)
        kw = dict(device=device, dtype=dtype)
        self.sigma = torch.as_tensor(sigma, **kw)
        self.epsilon = torch.as_tensor(epsilon, **kw)
        self.charge = torch.as_tensor(charge, **kw)

        # Lorentz-Berthelot, precomputed as full (N, N) tables.  N = 1538 so this is 19 MB in
        # float64 -- cheaper than recomputing per step, and it makes the mixing rule inspectable.
        self.sig_ij = 0.5 * (self.sigma[:, None] + self.sigma[None, :])
        self.eps_ij = torch.sqrt(self.epsilon[:, None] * self.epsilon[None, :])
        self.qq_ij = ONE_4PI_EPS0 * self.charge[:, None] * self.charge[None, :]

        # exclusion mask, and the self-pair, removed from the pair sum entirely
        excl = torch.zeros(self.n, self.n, dtype=torch.bool, device=device)
        ex = torch.as_tensor(np.asarray(exclusions), dtype=torch.long, device=device)
        excl[ex[:, 0], ex[:, 1]] = True
        excl[ex[:, 1], ex[:, 0]] = True
        excl.fill_diagonal_(True)
        self.excluded = excl
        self.exclusion_pairs = ex

        # --- site subsets for energy_forces_split -------------------------------------------
        self.lj_index = torch.nonzero(self.epsilon > 0, as_tuple=True)[0]
        self.q_index = torch.nonzero(self.charge != 0, as_tuple=True)[0]
        li, qi = self.lj_index, self.q_index
        self.lj_sig = self.sig_ij[li][:, li].contiguous()
        self.lj_eps = self.eps_ij[li][:, li].contiguous()
        self.lj_self = torch.eye(li.numel(), dtype=torch.bool, device=device)
        self.q_qq = self.qq_ij[qi][:, qi].contiguous()
        self.q_excluded = excl[qi][:, qi].contiguous()
        # every intramolecular exclusion is O-H or H-H, and for SPC/E hydrogens carry no LJ, so
        # the LJ set can only ever exclude the self-pair.  CHARMM TIP3P hydrogens DO carry LJ
        # (the NaCl system, Amendment 14.1), which puts intramolecular pairs inside the LJ set;
        # the main energy_forces path handles exclusions exactly, so only the split path -- a
        # recorded performance negative that nothing calls in production -- becomes invalid.
        self.split_path_valid = not bool(excl[li][:, li].fill_diagonal_(False).any())

    def _min_image(self, d):
        return d - self.L * torch.round(d / self.L)

    def energy_forces(self, x, chunk=256):
        """``(E (B,), F (B, N, 3))`` for switched LJ + real-space Coulomb.

        Chunked over ``i``; each chunk materialises ``(B, chunk, N, 3)``.

        The loop visits every *ordered* pair once, so the **energy** is halved.  The **force**
        is not: only the chunk-side term ``F_i = -(dE/dr)(d_ij/r)`` summed over all ``j`` is
        accumulated, and since each atom lies in exactly one chunk that is already its complete
        force.  Accumulating both sides and halving is equivalent but does twice the work.
        """
        B, N, _ = x.shape
        energy = x.new_zeros(B)
        forces = x.new_zeros(B, N, 3)
        for lo in range(0, N, chunk):
            hi = min(lo + chunk, N)
            d = self._min_image(x[:, lo:hi, None, :] - x[:, None, :, :])       # (B,c,N,3)
            r2 = (d * d).sum(-1)
            r = r2.clamp_min(1e-24).sqrt()

            live = (r < self.cutoff) & (~self.excluded[lo:hi, :])
            inv_r = torch.where(live, 1.0 / r, torch.zeros_like(r))

            # --- Lennard-Jones with the OpenMM switch --------------------------------------
            sr = self.sig_ij[lo:hi, :] * inv_r
            sr6 = sr.pow(6)
            sr12 = sr6 * sr6
            e_lj = 4.0 * self.eps_ij[lo:hi, :] * (sr12 - sr6)
            # dE/dr before switching
            dlj = -24.0 * self.eps_ij[lo:hi, :] * (2.0 * sr12 - sr6) * inv_r
            s, ds = switch_and_derivative(r, self.switch, self.cutoff)
            e_lj_s = e_lj * s
            dlj_s = dlj * s + e_lj * ds

            # --- PME real space: erfc(alpha r)/r, hard cutoff, no switch -------------------
            ar = self.alpha * r
            erfc = torch.erfc(ar)
            e_el = self.qq_ij[lo:hi, :] * erfc * inv_r
            # d/dr [ erfc(ar)/r ] = -( erfc(ar)/r^2 + 2a/sqrt(pi) exp(-a^2 r^2)/r )
            two_a_sqrtpi = 2.0 * self.alpha / np.sqrt(np.pi)
            del_ = -self.qq_ij[lo:hi, :] * (erfc * inv_r * inv_r
                                            + two_a_sqrtpi * torch.exp(-ar * ar) * inv_r)

            e_pair = torch.where(live, e_lj_s + e_el, torch.zeros_like(r))
            dE_dr = torch.where(live, dlj_s + del_, torch.zeros_like(r))
            energy = energy + e_pair.sum(dim=(1, 2))

            # F_i = -dE/dr * (d/r) summed over j;  d = x_i - x_j
            coef = (dE_dr * inv_r).unsqueeze(-1)
            forces[:, lo:hi, :] = -(coef * d).sum(dim=2)                       # (B,c,3)
        return 0.5 * energy, forces

    def energy_forces_split(self, x, chunk=128):
        """Same result as :meth:`energy_forces`, with LJ and Coulomb over their own site sets.

        .. warning::
           **MEASURED SLOWER -- kept as a recorded negative, do not re-try without a new idea.**
           B=512, float32, compiled, idle H200: **509 ns/day against 744** for the plain
           all-pairs path (chunk 256; chunk 128 is far worse at 109).  Correct to ``1e-11`` in
           energy against the parity-validated path, so this is a performance result, not a bug.
           The two ``index_add_`` scatters and the sub-set gathers cost more than the 9/10 of LJ
           arithmetic they remove, because the kernel is bound by memory traffic and this
           restructuring adds two extra passes over the positions.

        Only **514** of the 1538 sites carry Lennard-Jones (512 water oxygens + 2 methanes); the
        1024 hydrogens carry charge alone.  The combined loop therefore evaluates ``sr6``,
        ``sr12`` and the switching polynomial for ~2.1 M pairs per walker whose ``epsilon`` is
        identically zero.  Splitting the two interactions costs one extra gather and removes
        9/10 of the LJ arithmetic.

        A convenient consequence: **the LJ set needs no exclusion mask.**  Every intramolecular
        exclusion is O-H or H-H, and hydrogens are not in the LJ set, so only the self-pair has
        to be dropped.  The Coulomb set keeps the full mask.
        """
        if not self.split_path_valid:
            raise RuntimeError("LJ site set carries a non-self exclusion (LJ-bearing hydrogens); "
                               "the split path is invalid for this system -- use energy_forces")
        B, N, _ = x.shape
        energy = x.new_zeros(B)
        forces = x.new_zeros(B, N, 3)

        # ---- Lennard-Jones over LJ-bearing sites only ---------------------------------------
        li = self.lj_index
        xl = x[:, li, :]
        nl_ = li.numel()
        e_acc = x.new_zeros(B)
        f_acc = x.new_zeros(B, nl_, 3)
        for lo in range(0, nl_, chunk):
            hi = min(lo + chunk, nl_)
            d = self._min_image(xl[:, lo:hi, None, :] - xl[:, None, :, :])
            r = (d * d).sum(-1).clamp_min(1e-24).sqrt()
            live = (r < self.cutoff) & (~self.lj_self[lo:hi, :])
            inv_r = torch.where(live, 1.0 / r, torch.zeros_like(r))
            sr6 = (self.lj_sig[lo:hi, :] * inv_r).pow(6)
            sr12 = sr6 * sr6
            e_lj = 4.0 * self.lj_eps[lo:hi, :] * (sr12 - sr6)
            dlj = -24.0 * self.lj_eps[lo:hi, :] * (2.0 * sr12 - sr6) * inv_r
            s, ds = switch_and_derivative(r, self.switch, self.cutoff)
            e_acc = e_acc + torch.where(live, e_lj * s, torch.zeros_like(r)).sum(dim=(1, 2))
            dE = torch.where(live, dlj * s + e_lj * ds, torch.zeros_like(r))
            f_acc[:, lo:hi, :] = -((dE * inv_r).unsqueeze(-1) * d).sum(dim=2)
        energy = energy + 0.5 * e_acc
        forces.index_add_(1, li, f_acc)

        # ---- PME real-space Coulomb over charged sites only ---------------------------------
        qi = self.q_index
        xq = x[:, qi, :]
        nq = qi.numel()
        two_a_sqrtpi = 2.0 * self.alpha / np.sqrt(np.pi)
        e_acc = x.new_zeros(B)
        f_acc = x.new_zeros(B, nq, 3)
        for lo in range(0, nq, chunk):
            hi = min(lo + chunk, nq)
            d = self._min_image(xq[:, lo:hi, None, :] - xq[:, None, :, :])
            r = (d * d).sum(-1).clamp_min(1e-24).sqrt()
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

    def energy_forces_nl(self, x, nl, chunk=256):
        """Same quantities as :meth:`energy_forces`, over a :class:`VerletList`.

        Parameters are combined **from per-site vectors** rather than gathered out of ``(N, N)``
        tables: the tables would themselves stream ``(B, chunk, N)`` per step, which is precisely
        the traffic the neighbour list exists to remove.
        """
        B, N, _ = x.shape
        energy = x.new_zeros(B)
        forces = x.new_zeros(B, N, 3)
        two_a_sqrtpi = 2.0 * self.alpha / np.sqrt(np.pi)
        for lo in range(0, N, chunk):
            hi = min(lo + chunk, N)
            j = nl.idx[:, lo:hi, :]                                    # (B,c,M)
            ok = nl.valid[:, lo:hi, :]
            xj = torch.gather(x, 1, j.reshape(B, -1, 1).expand(-1, -1, 3)).view(*j.shape, 3)
            d = self._min_image(x[:, lo:hi, None, :] - xj)
            r = (d * d).sum(-1).clamp_min(1e-24).sqrt()

            live = ok & (r < self.cutoff)
            inv_r = torch.where(live, 1.0 / r, torch.zeros_like(r))

            sig = 0.5 * (self.sigma[lo:hi].view(1, -1, 1) + self.sigma[j])
            eps = torch.sqrt(self.epsilon[lo:hi].view(1, -1, 1) * self.epsilon[j])
            qq = ONE_4PI_EPS0 * self.charge[lo:hi].view(1, -1, 1) * self.charge[j]

            sr6 = (sig * inv_r).pow(6)
            sr12 = sr6 * sr6
            e_lj = 4.0 * eps * (sr12 - sr6)
            dlj = -24.0 * eps * (2.0 * sr12 - sr6) * inv_r
            s, ds = switch_and_derivative(r, self.switch, self.cutoff)

            ar = self.alpha * r
            erfc = torch.erfc(ar)
            e_el = qq * erfc * inv_r
            del_ = -qq * (erfc * inv_r * inv_r + two_a_sqrtpi * torch.exp(-ar * ar) * inv_r)

            e_pair = torch.where(live, e_lj * s + e_el, torch.zeros_like(r))
            dE_dr = torch.where(live, dlj * s + e_lj * ds + del_, torch.zeros_like(r))
            energy = energy + e_pair.sum(dim=(1, 2))
            forces[:, lo:hi, :] = -((dE_dr * inv_r).unsqueeze(-1) * d).sum(dim=2)
        return 0.5 * energy, forces

    def exclusion_correction(self, x):
        """``-q_i q_j erf(alpha r)/r`` over excluded pairs: removes their reciprocal share.

        Gated separately by the parity test because omitting it produces a large, smooth,
        entirely plausible-looking error rather than a failure.
        """
        i, j = self.exclusion_pairs[:, 0], self.exclusion_pairs[:, 1]
        d = self._min_image(x[:, i, :] - x[:, j, :])
        r = (d * d).sum(-1).clamp_min(1e-24).sqrt()
        qq = self.qq_ij[i, j]
        ar = self.alpha * r
        erf = torch.erf(ar)
        e = -(qq * erf / r)
        two_a_sqrtpi = 2.0 * self.alpha / np.sqrt(np.pi)
        # d/dr [ -qq erf(ar)/r ] = -qq ( 2a/sqrt(pi) exp(-a^2r^2)/r - erf(ar)/r^2 )
        dE_dr = -qq * (two_a_sqrtpi * torch.exp(-ar * ar) / r - erf / (r * r))
        coef = (dE_dr / r).unsqueeze(-1)
        f_pair = -(coef * d)
        forces = x.new_zeros(x.shape)
        forces.index_add_(1, i, f_pair)
        forces.index_add_(1, j, -f_pair)
        return e.sum(-1), forces

    def self_energy(self):
        """Ewald self term ``-alpha/sqrt(pi) * ONE_4PI_EPS0 * sum q^2``; no forces."""
        return -(self.alpha / np.sqrt(np.pi)) * ONE_4PI_EPS0 * float((self.charge ** 2).sum())


class VerletList:
    """Padded per-walker neighbour list within ``cutoff + skin``, rebuilt every few hundred steps.

    .. warning::
       **MEASURED SLOWER -- kept as a recorded negative, do not re-try at this box size.**
       B=512, float32, compiled, idle H200: **655 ns/day against 744** for plain all-pairs, and
       correct to ``5e-12`` in energy against the parity-validated path.  The reason is
       structural rather than fixable: at ``L = 2.61 nm`` a ``1.25 nm`` list radius captures
       **862 of 1538** sites -- 56 % of the box -- so the list culls only 1.8x while replacing
       contiguous streaming with random-access gathers.  A neighbour list needs a box several
       cutoffs wide to pay, and this one is 2.5 cutoffs across.  It would become worth
       re-measuring for a larger box (the 1024-water finite-size check of SPEC §1.3).

    Cell lists are useless at this box size -- ``L ~ 2.6 nm`` with a ``1.05 nm`` cutoff admits
    only 2 cells per axis, so every cell is a neighbour of every other and nothing is culled.
    A Verlet list is the only structure that helps, and it helps because the *memory traffic*, not
    the arithmetic, is what bounds this kernel: the all-pairs path streams ``(B, chunk, N)``
    intermediates when only ~40 % of those slots are inside the cutoff.

    The list is ``(B, N, M)`` with a validity mask, ``M`` fixed at build time from the observed
    maximum.  Rebuild cadence is set by the skin: SPC/E oxygens diffuse ~0.03 nm/ps, so a 0.2 nm
    skin survives thousands of 0.5 fs steps; ``rebuild_every`` is nonetheless checked against the
    **measured** maximum displacement, and a violation raises rather than silently dropping pairs.
    """

    def __init__(self, n_sites, box_nm, cutoff_nm, skin_nm=0.20):
        self.n = int(n_sites)
        self.L = float(box_nm)
        self.r_list = float(cutoff_nm) + float(skin_nm)
        self.skin = float(skin_nm)
        self.idx = None            # (B, N, M) long
        self.valid = None          # (B, N, M) bool
        self.x_built = None        # positions at last rebuild, for the drift check

    def _min_image(self, d):
        return d - self.L * torch.round(d / self.L)

    def rebuild(self, x, excluded, chunk=128, headroom=1.15):
        """Rebuild from positions ``(B, N, 3)``; excluded and self pairs are dropped here once."""
        B, N, _ = x.shape
        counts = []
        masks = []
        for lo in range(0, N, chunk):
            hi = min(lo + chunk, N)
            d = self._min_image(x[:, lo:hi, None, :] - x[:, None, :, :])
            r2 = (d * d).sum(-1)
            m = (r2 < self.r_list ** 2) & (~excluded[lo:hi, :])
            masks.append(m)
            counts.append(m.sum(-1))
        counts = torch.cat(counts, dim=1)
        m_max = int(counts.max().item())
        M = min(N, max(8, int(m_max * headroom)))

        idx = x.new_zeros(B, N, M, dtype=torch.long)
        valid = x.new_zeros(B, N, M, dtype=torch.bool)
        off = 0
        for m in masks:
            c = m.shape[1]
            # stable argsort of ~mask brings the True entries to the front, in index order
            order = torch.argsort((~m).to(torch.uint8), dim=-1, stable=True)[..., :M]
            idx[:, off:off + c] = order
            valid[:, off:off + c] = torch.gather(m, -1, order)
            off += c
        self.idx, self.valid = idx, valid
        self.x_built = x.detach().clone()
        return M

    def check_drift(self, x):
        """Max displacement since the rebuild; must stay under half the skin."""
        d = self._min_image(x - self.x_built)
        return float(d.norm(dim=-1).max())


class MethaneNonbonded:
    """The complete frozen model: switched LJ + PME, batched over walkers.

    Built from the parameters read back out of the OpenMM ``System`` (SPEC §1.1), so the engine
    and its parity target cannot disagree about a constant.  ``energy_forces`` returns the same
    quantities OpenMM's ``getState`` does, in the same units.
    """

    def __init__(self, system, topology, box_nm, device=None, dtype=torch.float64,
                 alpha_per_nm=None, grid=None, order=None):
        from . import system as msys

        p = msys.site_parameters(system, topology)
        self.params = p
        self.n = len(p["charge"])
        self.box_nm = float(box_nm)
        self.methane_index = torch.as_tensor(p["methane_index"], device=device, dtype=torch.long)
        self.mass = torch.as_tensor(p["mass"], device=device, dtype=dtype)

        alpha = msys.PME_ALPHA_PER_NM if alpha_per_nm is None else float(alpha_per_nm)
        grid = msys.PME_GRID if grid is None else tuple(grid)
        order = msys.PME_SPLINE_ORDER if order is None else int(order)

        self.pair = PairTerms(p["sigma"], p["epsilon"], p["charge"], msys.exclusions(system),
                              box_nm, msys.CUTOFF_NM, msys.SWITCH_NM, alpha,
                              device=device, dtype=dtype)
        from .pme import PMEReciprocal
        self.recip = PMEReciprocal(p["charge"], box_nm, grid, alpha, order=order,
                                   device=device, dtype=dtype)
        self.e_self = self.pair.self_energy()

    def enable_triton(self, block_i=64, block_j=64):
        """Route the pair term through the fused Triton kernel (performance-only, gated).

        Builds the molecule-id table and hard-asserts its equal-id mask against the
        OpenMM-derived exclusion mask before anything can run (see ``triton_pair``).
        """
        from .triton_pair import build_mol_id
        self._mol_id = build_mol_id(self.pair)
        self._triton_blocks = (int(block_i), int(block_j))
        return self

    def energy_forces(self, x, chunk=256):
        """``(E (B,), F (B, N, 3))`` in kJ/mol and kJ/mol/nm."""
        if getattr(self, "_mol_id", None) is not None and x.is_cuda:
            from .triton_pair import pair_energy_forces_triton
            bi, bj = self._triton_blocks
            e_r, f_r = pair_energy_forces_triton(self.pair, x, self._mol_id,
                                                 block_i=bi, block_j=bj)
        else:
            e_r, f_r = self.pair.energy_forces(x, chunk=chunk)
        e_x, f_x = self.pair.exclusion_correction(x)
        e_k, f_k = self.recip.energy_forces(x)
        return e_r + e_x + e_k + self.e_self, f_r + f_x + f_k
