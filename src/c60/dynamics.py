"""Constrained BAOAB Langevin for TIP4P-Ew water around the C60 pair, batched over walkers.

Frozen integrator: ``docs/SPEC_c60_water.md`` §1 -- BAOAB, 300 K, ``gamma = 1 ps^-1``,
``dt`` per the §3.4 gate, rigid water by M-SHAKE/RATTLE (the methane machinery, consumed
unmodified), and the TIP4P-Ew M site as a massless virtual site.

Massless sites
--------------
The mass vector carries zeros at every M site and every carbon.  ``inv_mass`` is zeroed there
(not infinite), so the B kick, the O noise and the A drift all act on massive sites only --
exactly OpenMM's convention that massless particles do not respond to dynamics.  After every
position update the M sites are *recomputed* from their (O, H1, H2) with the virtual-site
weights read from OpenMM, and forces arriving on M have already been redistributed onto the
parents by ``C60Nonbonded.redistribute`` before they reach the integrator.

The single solute degree of freedom
-----------------------------------
SPEC §1.3: the only solute coordinate is ``xi = Z_B - Z_A``, effective mass
``mu = M_cage/2 = 360.33 amu``, propagated by the same BAOAB splitting as a scalar channel;
the 120 carbon positions are reconstructed from ``xi`` after every drift.  In fixed-cage mode
(reference windows, pools) the channel is off and carbons never move.

DOF counting: ``6`` per water (9 coordinates minus 3 constraints), ``+1`` for ``xi`` when
dynamic; **no** centre-of-mass subtraction -- the fixed cages are an external potential, water
momentum is not conserved, and the OpenMM comparison system is built with
``removeCMMotion=False`` to match.
"""
from __future__ import annotations

import numpy as np
import torch

from methane.dynamics import KB_KJ_PER_MOL_K, RigidWaterConstraints

from . import geometry
from . import system as csys


class C60Dynamics:
    """BAOAB for the C60/TIP4P-Ew system; fixed cages by default, optional xi channel."""

    def __init__(self, engine, dt_ps, temperature_k=csys.TEMPERATURE_K,
                 gamma_ps=csys.GAMMA_PS, xi_dynamic=False, force_fn=None,
                 device=None, dtype=torch.float64):
        self.engine = engine
        #: ``force_fn(x) -> (E (B,), F_raw (B, N, 3))``.  Injectable so drivers can pass a
        #: ``torch.compile``d wrapper; defaulting to the eager engine would silently discard
        #: the compilation the methane engine measured at 8.1x.
        self.force_fn = force_fn if force_fn is not None else engine.energy_forces
        self.dt = float(dt_ps)
        self.T = float(temperature_k)
        self.gamma = float(gamma_ps)
        self.kT = KB_KJ_PER_MOL_K * self.T
        self.xi_dynamic = bool(xi_dynamic)
        self.device, self.dtype = device, dtype

        mass = engine.mass.to(device=device, dtype=dtype)
        self.mass = mass
        self.inv_m = torch.where(mass > 0, 1.0 / mass.clamp_min(1e-30),
                                 torch.zeros_like(mass))
        self.massive = mass > 0

        waters = engine.waters
        self.cons = RigidWaterConstraints(
            waters[:, :3].cpu().numpy(),
            [csys.R_OH_NM, csys.R_OH_NM, csys.R_HH_NM],
            mass.cpu().numpy(), device=device, dtype=dtype)

        self.c1 = float(np.exp(-self.gamma * self.dt))
        self.c2 = float(np.sqrt(1.0 - self.c1 ** 2))
        sig = torch.sqrt(self.kT * self.inv_m)
        self.sigma = sig[None, :, None]

        # xi channel
        self.mu = csys.MU_XI_AMU
        self.sigma_xi = float(np.sqrt(self.kT / self.mu))
        cage = geometry.c60_cage()
        self.cage_template = torch.as_tensor(cage, device=device, dtype=dtype)
        self.n_waters = int(waters.shape[0])

    # ------------------------------------------------------------------ helpers
    def n_dof(self):
        return 6 * self.n_waters + (1 if self.xi_dynamic else 0)

    def temperature(self, v, v_xi=None):
        ke = 0.5 * (self.mass[None, :, None] * v * v).sum(dim=(1, 2))
        if self.xi_dynamic and v_xi is not None:
            ke = ke + 0.5 * self.mu * v_xi * v_xi
        return 2.0 * ke / (self.n_dof() * KB_KJ_PER_MOL_K)

    def place_cages(self, x, xi, center):
        """Rebuild the 120 carbon positions from ``xi`` (B,), in place."""
        e_a = self.engine.cage_a
        e_b = self.engine.cage_b
        c = torch.as_tensor(center, device=x.device, dtype=x.dtype)
        x[:, e_a, :] = self.cage_template[None] + c[None, None, :]
        x[:, e_a, 2] += -0.5 * xi[:, None]
        x[:, e_b, :] = self.cage_template[None] + c[None, None, :]
        x[:, e_b, 2] += +0.5 * xi[:, None]
        return x

    def maxwell_velocities(self, x, generator=None, xi=False):
        v = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        v = v * self.sigma
        self.cons.apply_velocities(x, v)
        if not xi:
            return v
        B = x.shape[0]
        v_xi = self.sigma_xi * torch.randn(B, device=x.device, dtype=x.dtype,
                                           generator=generator)
        return v, v_xi

    # ------------------------------------------------------------------ the step
    def step(self, x, v, f, xi=None, v_xi=None, f_xi=None, center=None,
             bias_fn=None, generator=None):
        """One BAOAB step in place.  ``f`` must be **redistributed** forces.

        Fixed-cage mode: ``step(x, v, f)`` -> ``(e, f_new)``.
        xi mode: pass ``xi, v_xi, f_xi`` (generalised force on xi, physical + bias + wall) and
        ``center``; ``bias_fn(xi_new) -> generalised force`` is evaluated at the new position
        inside the step (the stale-bias lesson).  Returns
        ``(e, f_new, xi, v_xi, f_xi_phys)`` where ``f_xi_phys`` is the *physical* local mean
        force at the new configuration -- exactly the ABF estimator sample.
        """
        dyn_xi = self.xi_dynamic and xi is not None

        # --- B
        v += self.dt * self.inv_m[None, :, None] * f
        self.cons.apply_velocities(x, v)
        if dyn_xi:
            v_xi += self.dt * f_xi / self.mu

        # --- A O A
        half = 0.5 * self.dt
        x_ref = x.clone()
        x += half * v
        noise = torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=generator)
        v.mul_(self.c1).add_(self.c2 * self.sigma * noise)
        x += half * v
        if dyn_xi:
            xi = xi + half * v_xi
            xi_noise = torch.randn_like(v_xi) if generator is None else torch.randn(
                v_xi.shape, device=v_xi.device, dtype=v_xi.dtype, generator=generator)
            v_xi = self.c1 * v_xi + self.c2 * self.sigma_xi * xi_noise
            xi = xi + half * v_xi

        x_unc = x.clone()
        self.cons.apply_positions(x, x_ref)
        v += (x - x_unc) / self.dt
        self.cons.apply_velocities(x, v)

        # --- rebuild dependent coordinates, then force at the new positions
        if dyn_xi:
            self.place_cages(x, xi, center)
        self.engine.compute_vsites(x)
        e, f_raw = self.force_fn(x)
        f_new = self.engine.redistribute(f_raw)

        if not dyn_xi:
            return e, f_new
        f_xi_phys = self.engine.local_mean_force(f_raw)
        f_xi_new = f_xi_phys + (bias_fn(xi) if bias_fn is not None else 0.0)
        return e, f_new, xi, v_xi, f_xi_new, f_xi_phys
