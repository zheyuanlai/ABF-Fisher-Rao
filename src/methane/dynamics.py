"""Constrained BAOAB Langevin dynamics for rigid SPC/E water, batched over walkers.

Frozen integrator: ``docs/SPEC_methane_water.md`` §1 -- BAOAB, ``T = 298 K``, ``gamma = 1 ps^-1``,
``dt = 0.5 fs``, rigid water by constraint.  This is the same splitting OpenMM's
``LangevinMiddleIntegrator`` uses, which matters because the constrained-TI reference runs in
OpenMM and the population arms run here (Amendment 12.4): the two engines must agree on the
*dynamics*, not only on the forces.

Constraints: matrix SHAKE / RATTLE, not SETTLE
----------------------------------------------
Each water carries three distance constraints (two O-H and one H-H), so the constraint Jacobian
per molecule is ``3 x 9`` and ``J M^-1 J^T`` is a ``3 x 3`` matrix.  Newton iteration on the
Lagrange multipliers is therefore a **batched 3x3 solve** -- branch-free, fully vectorised over
``(walker, molecule)``, and it converges to ``1e-13`` in two or three passes.

SETTLE is the analytic alternative and is what OpenMM uses.  It is not used here because its
closed form is a long sequence of frame constructions that is easy to get subtly wrong and hard
to test in isolation, whereas M-SHAKE's convergence is *self-checking*: the residual it drives to
zero is exactly the quantity SPEC §3.2 gates (``|constraint violation| <= 1e-8 nm``).  The
correctness argument here is a measurement, not a derivation.

Rigid-body degrees of freedom
-----------------------------
Constraints remove ``3 x N_w`` degrees of freedom, so the kinetic temperature must be computed
against the **constrained** count, not ``3N``.  Getting this wrong makes a correct integrator
look 1.5x too cold and is the classic way to "discover" a thermostat bug that is not there.
"""
from __future__ import annotations

import numpy as np
import torch

#: Boltzmann constant in kJ/(mol K) -- the project's internal unit system.
KB_KJ_PER_MOL_K = 8.31446261815324e-3


class RigidWaterConstraints:
    """Batched M-SHAKE / RATTLE for a set of 3-site rigid molecules.

    ``molecules`` is an ``(n_mol, 3)`` integer array of site indices ``(O, H1, H2)``; ``lengths``
    is the ``(3,)`` set of target distances for the pairs ``(O,H1), (O,H2), (H1,H2)``.
    """

    #: the three intramolecular pairs, as offsets into a molecule's ``(O, H1, H2)`` triple
    PAIRS = ((0, 1), (0, 2), (1, 2))

    def __init__(self, molecules, lengths, mass, device=None, dtype=torch.float64,
                 tol_nm=1.0e-10, max_iter=8):
        self.mol = torch.as_tensor(np.asarray(molecules), device=device, dtype=torch.long)
        self.d2 = torch.as_tensor(np.asarray(lengths), device=device, dtype=dtype) ** 2
        self.inv_mass = 1.0 / torch.as_tensor(np.asarray(mass), device=device, dtype=dtype)
        self.tol = float(tol_nm)
        self.max_iter = int(max_iter)
        self.n_mol = self.mol.shape[0]
        self.n_constraints = 3 * self.n_mol
        # per-molecule inverse masses of (O, H1, H2)
        self.w = self.inv_mass[self.mol]                                  # (n_mol, 3)

    def _gather(self, x):
        """``(B, n_mol, 3, 3)`` -- walker, molecule, site-in-molecule, xyz."""
        B = x.shape[0]
        return x[:, self.mol.reshape(-1), :].view(B, self.n_mol, 3, 3)

    def _scatter_add(self, x, delta):
        B = x.shape[0]
        return x.index_add_(1, self.mol.reshape(-1), delta.reshape(B, -1, 3))

    def apply_positions(self, x, x_ref):
        """Project ``x`` back onto the constraint manifold, in place.

        Newton on the multipliers: ``g_k = |r_k|^2 - d_k^2``, ``dg_k/dlambda_l`` built from the
        **reference** (pre-move) bond vectors, which is what makes this SHAKE rather than a
        naive projection and keeps it symplectic-compatible.
        Returns the final maximum absolute residual in nm.
        """
        ref = self._gather(x_ref)
        for _ in range(self.max_iter):
            cur = self._gather(x)
            rc = torch.stack([cur[:, :, b, :] - cur[:, :, a, :] for a, b in self.PAIRS], dim=2)
            rr = torch.stack([ref[:, :, b, :] - ref[:, :, a, :] for a, b in self.PAIRS], dim=2)
            g = (rc * rc).sum(-1) - self.d2                                # (B, n_mol, 3)
            if float(g.abs().max()) < self.tol:
                break
            # A[k,l] = 2 * rc_k . rr_l * (w_a[k] delta_a + w_b[k] delta_b ... )
            A = x.new_zeros(*g.shape, 3)
            for k, (ak, bk) in enumerate(self.PAIRS):
                for l, (al, bl) in enumerate(self.PAIRS):
                    coef = ((1.0 if bk == bl else 0.0) * self.w[:, bk]
                            - (1.0 if bk == al else 0.0) * self.w[:, bk]
                            - (1.0 if ak == bl else 0.0) * self.w[:, ak]
                            + (1.0 if ak == al else 0.0) * self.w[:, ak])
                    A[:, :, k, l] = 2.0 * (rc[:, :, k, :] * rr[:, :, l, :]).sum(-1) * coef
            lam = torch.linalg.solve(A, g.unsqueeze(-1)).squeeze(-1)       # (B, n_mol, 3)
            delta = x.new_zeros(x.shape[0], self.n_mol, 3, 3)
            for l, (al, bl) in enumerate(self.PAIRS):
                contrib = lam[:, :, l, None] * rr[:, :, l, :]
                delta[:, :, bl, :] -= self.w[:, bl, None] * contrib
                delta[:, :, al, :] += self.w[:, al, None] * contrib
            self._scatter_add(x, delta)
        cur = self._gather(x)
        rc = torch.stack([cur[:, :, b, :] - cur[:, :, a, :] for a, b in self.PAIRS], dim=2)
        r = (rc * rc).sum(-1).sqrt()
        return float((r - self.d2.sqrt()).abs().max())

    def apply_velocities(self, x, v):
        """RATTLE: remove the velocity component along every constraint, in place.

        Exact in one solve -- the constraint is linear in ``v`` -- so no iteration is needed.
        """
        pos = self._gather(x)
        vel = self._gather(v)
        rc = torch.stack([pos[:, :, b, :] - pos[:, :, a, :] for a, b in self.PAIRS], dim=2)
        dv = torch.stack([vel[:, :, b, :] - vel[:, :, a, :] for a, b in self.PAIRS], dim=2)
        rhs = (rc * dv).sum(-1)                                            # (B, n_mol, 3)
        A = v.new_zeros(*rhs.shape, 3)
        for k, (ak, bk) in enumerate(self.PAIRS):
            for l, (al, bl) in enumerate(self.PAIRS):
                coef = ((1.0 if bk == bl else 0.0) * self.w[:, bk]
                        - (1.0 if bk == al else 0.0) * self.w[:, bk]
                        - (1.0 if ak == bl else 0.0) * self.w[:, ak]
                        + (1.0 if ak == al else 0.0) * self.w[:, ak])
                A[:, :, k, l] = (rc[:, :, k, :] * rc[:, :, l, :]).sum(-1) * coef
        lam = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
        delta = v.new_zeros(v.shape[0], self.n_mol, 3, 3)
        for l, (al, bl) in enumerate(self.PAIRS):
            contrib = lam[:, :, l, None] * rc[:, :, l, :]
            delta[:, :, bl, :] -= self.w[:, bl, None] * contrib
            delta[:, :, al, :] += self.w[:, al, None] * contrib
        self._scatter_add(v, delta)
        return float(rhs.abs().max())

    def max_violation(self, x):
        cur = self._gather(x)
        rc = torch.stack([cur[:, :, b, :] - cur[:, :, a, :] for a, b in self.PAIRS], dim=2)
        return float(((rc * rc).sum(-1).sqrt() - self.d2.sqrt()).abs().max())


class PairConstraint:
    """A single rigid distance between two sites -- the constrained-TI holder for ``xi``.

    One constraint means SHAKE is a scalar, not a solve: with ``r = x_j - x_i`` and reference
    direction ``rr`` from the pre-move configuration,

        lambda = (|r|^2 - d^2) / (2 (r . rr) (w_i + w_j))

    applied as ``x_j -= w_j lambda rr``, ``x_i += w_i lambda rr``.  Iterated to convergence, which
    takes two passes.

    Used to hold the methane pair at fixed separation for the TI reference.  Because
    ``|grad xi|^2 = 2`` is constant for this CV there is no Fixman/metric correction, so the
    conditional average of the physical local mean force is ``F'(r)`` exactly -- and the
    constraint force never appears in the accumulated forces, exactly as in OpenMM.
    """

    def __init__(self, i, j, distance_nm, mass, device=None, dtype=torch.float64,
                 tol_nm=1.0e-10, max_iter=6):
        self.i, self.j = int(i), int(j)
        self.d = torch.as_tensor(np.asarray(distance_nm), device=device, dtype=dtype)
        inv = 1.0 / torch.as_tensor(np.asarray(mass), device=device, dtype=dtype)
        self.wi, self.wj = inv[self.i], inv[self.j]
        self.tol = float(tol_nm)
        self.max_iter = int(max_iter)
        self.n_constraints = 1

    def _d2(self):
        return self.d * self.d

    def apply_positions(self, x, x_ref):
        rr = x_ref[:, self.j, :] - x_ref[:, self.i, :]
        for _ in range(self.max_iter):
            rc = x[:, self.j, :] - x[:, self.i, :]
            g = (rc * rc).sum(-1) - self._d2()
            if float(g.abs().max()) < self.tol:
                break
            denom = 2.0 * (rc * rr).sum(-1) * (self.wi + self.wj)
            lam = g / torch.where(denom.abs() < 1e-30, torch.full_like(denom, 1e-30), denom)
            x[:, self.j, :] -= self.wj * lam[:, None] * rr
            x[:, self.i, :] += self.wi * lam[:, None] * rr
        rc = x[:, self.j, :] - x[:, self.i, :]
        return float((rc.norm(dim=-1) - self.d).abs().max())

    def apply_velocities(self, x, v):
        rc = x[:, self.j, :] - x[:, self.i, :]
        dv = v[:, self.j, :] - v[:, self.i, :]
        denom = (rc * rc).sum(-1) * (self.wi + self.wj)
        lam = (rc * dv).sum(-1) / denom.clamp_min(1e-30)
        v[:, self.j, :] -= self.wj * lam[:, None] * rc
        v[:, self.i, :] += self.wi * lam[:, None] * rc
        return float((rc * dv).sum(-1).abs().max())

    def max_violation(self, x):
        rc = x[:, self.j, :] - x[:, self.i, :]
        return float((rc.norm(dim=-1) - self.d).abs().max())


class CompositeConstraints:
    """Apply several independent constraint sets in sequence.

    Exact when the sets share no atoms -- the methane pair and the waters are disjoint, so the
    two projections commute and one pass of each suffices.  A shared atom would need the sets
    solved jointly, so the disjointness is asserted rather than assumed.
    """

    def __init__(self, parts, atom_sets=None):
        self.parts = list(parts)
        self.n_constraints = sum(p.n_constraints for p in self.parts)
        if atom_sets is not None:
            seen = set()
            for s in atom_sets:
                s = set(int(a) for a in np.asarray(s).reshape(-1))
                if seen & s:
                    raise ValueError("constraint sets share atoms; they must be solved jointly")
                seen |= s

    def apply_positions(self, x, x_ref):
        return max(p.apply_positions(x, x_ref) for p in self.parts)

    def apply_velocities(self, x, v):
        return max(p.apply_velocities(x, v) for p in self.parts)

    def max_violation(self, x):
        return max(p.max_violation(x) for p in self.parts)


class BAOAB:
    """Constrained BAOAB Langevin, batched over walkers.

    ``force_fn(x) -> (energy (B,), force (B, N, 3))``.  Optional ``bias_fn(x) -> (B, N, 3)`` adds
    the ABF/mFR Cartesian bias force; it is a separate argument rather than folded into
    ``force_fn`` so that the *physical* energy stays available unbiased for diagnostics.
    """

    def __init__(self, force_fn, mass, constraints, dt_ps, temperature_k, gamma_ps,
                 device=None, dtype=torch.float64):
        self.force_fn = force_fn
        self.cons = constraints
        self.dt = float(dt_ps)
        self.T = float(temperature_k)
        self.gamma = float(gamma_ps)
        self.m = torch.as_tensor(np.asarray(mass), device=device, dtype=dtype)
        self.inv_m = 1.0 / self.m
        self.kT = KB_KJ_PER_MOL_K * self.T
        # Ornstein-Uhlenbeck coefficients for the full-step O block
        self.c1 = float(np.exp(-self.gamma * self.dt))
        self.c2 = float(np.sqrt(1.0 - self.c1 ** 2))
        self.sigma = torch.sqrt(self.kT * self.inv_m)[None, :, None]

    def n_dof(self, n_sites):
        """Degrees of freedom after constraints and centre-of-mass removal."""
        return 3 * int(n_sites) - self.cons.n_constraints - 3

    def temperature(self, v):
        """Instantaneous kinetic temperature per walker, against the constrained DOF count."""
        ke = 0.5 * (self.m[None, :, None] * v * v).sum(dim=(1, 2))
        return 2.0 * ke / (self.n_dof(v.shape[1]) * KB_KJ_PER_MOL_K)

    def maxwell_velocities(self, x, generator=None):
        v = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        v = v * self.sigma
        self.cons.apply_velocities(x, v)
        return v

    def step(self, x, v, f, bias_fn=None, generator=None):
        """One BAOAB step, in place on ``x`` and ``v``.  Returns the new ``(energy, force)``.

        Splitting: ``B A O A B``, with RATTLE after every velocity update and SHAKE after every
        position update, which is the constrained form OpenMM's ``LangevinMiddleIntegrator`` uses.
        """
        half = 0.5 * self.dt
        # --- B (full kick), exactly OpenMM's LangevinMiddleIntegrator ordering
        v += self.dt * self.inv_m[None, :, None] * f
        self.cons.apply_velocities(x, v)
        # --- A O A, with the position constraint applied once at the end of the pair
        x_ref = x.clone()                     # bond directions for SHAKE
        x += half * v
        noise = torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=generator)
        v.mul_(self.c1).add_(self.c2 * self.sigma * noise)
        x += half * v
        x_unc = x.clone()                     # unconstrained target of the second A block
        self.cons.apply_positions(x, x_ref)
        # Only the *constraint* displacement enters the velocity, divided by the **full** dt
        # because SHAKE is applied once and absorbs the drift of both A blocks.
        #
        # Two neighbouring forms were measured against OpenMM over 8 ps, and both fail silently --
        # every force, parity and constraint test still passes while the ensemble is simply at the
        # wrong temperature:
        #   v <- (x - x_ref)/dt         (replace, not add)  ->  156 K   against OpenMM's 303 K
        #   v += (x - x_unc)/(dt/2)     (half-step divisor)  ->  349 K  against OpenMM's 299 K
        #   v += (x - x_unc)/dt         (this one)           ->  297 K  against OpenMM's 300 K
        # Replacing v by the step-average displacement is wrong because that average has
        # systematically smaller variance than the instantaneous velocity, so it bleeds kinetic
        # energy every step.
        v += (x - x_unc) / self.dt
        self.cons.apply_velocities(x, v)
        # --- force at the new positions, carried into the next step's B block
        e, f_new = self.force_fn(x)
        if bias_fn is not None:
            f_new = f_new + bias_fn(x)
        return e, f_new


def water_molecules(topology):
    """``(n_waters, 3)`` site indices ordered ``(O, H1, H2)``, from the OpenMM topology."""
    out = []
    for res in topology.residues():
        if res.name not in ("HOH", "WAT"):
            continue
        idx = {a.name: a.index for a in res.atoms()}
        out.append((idx["O"], idx["H1"], idx["H2"]))
    return np.asarray(out, dtype=np.int64)
