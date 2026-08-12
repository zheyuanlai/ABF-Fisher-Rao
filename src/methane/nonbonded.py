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

    def energy_forces(self, x, chunk=256):
        """``(E (B,), F (B, N, 3))`` in kJ/mol and kJ/mol/nm."""
        e_r, f_r = self.pair.energy_forces(x, chunk=chunk)
        e_x, f_x = self.pair.exclusion_correction(x)
        e_k, f_k = self.recip.energy_forces(x)
        return e_r + e_x + e_k + self.e_self, f_r + f_x + f_k
