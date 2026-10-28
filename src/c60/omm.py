"""OpenMM-CUDA engine for the C60/TIP4P-Ew pair: fixed-cage windows and the dynamical xi channel.

The physical model is ``c60.system`` unchanged (SPEC §1).  This module only decides how the
cages move and what the integrator records; no force-field number is restated here.

Fixed-cage mode (reference windows, conditional pools)
-------------------------------------------------------
Carbons have zero mass (``build_system(fix_cages=True)``) and never move.  The integrator is
OpenMM's documented CustomIntegrator form of ``LangevinMiddleIntegrator`` (fused-kick BAOAB,
SETTLE rigid water) plus one reduction per step:

    floc = sum_DOF f * wf,      wf = +1/2 e_z on cage-A carbons, -1/2 e_z on cage-B carbons

so ``floc = (1/2)(F_A,z - F_B,z) = dV/dxi`` (SPEC §2), accumulated on the device and read once
per block.

xi mode (ABF / FR walkers)
--------------------------
The single solute DOF ``xi = Z_B - Z_A`` (midpoint, lateral offsets and orientations fixed,
SPEC §1.3) is carried by one massive particle ``P`` with mass ``2 M_cage``:

    Z_B = Z_P,   Z_A = 2 c_z - Z_P,   xi = 2 (Z_P - c_z)

Every carbon is a ``LocalCoordinatesSite`` of ``(P, G0, G1, G2)``: origin ``P`` for cage B and
``2 G0 - P`` for cage A, frame from three massless (immobile) ghosts at ``c``, ``c + e_x`` and
``c + e_y``.  OpenMM then hands ``P`` the virtual-site force ``F_P = sum_B F - sum_A F``, so
``F_P,z = -2 dV/dxi``; with ``m_P = 2 M`` the induced xi dynamics is exactly
``mu xi'' = -dV/dxi`` with ``mu = M/2`` (KE = (1/8) m_P xi'^2 = (1/2)(M/2) xi'^2).
P's lateral components are frozen by a per-DOF mask in the integrator.

The ABF bias enters as a generalised force ``+A'_hat(xi)`` on xi, i.e. ``2 A'_hat`` on Z_P,
read from a per-DOF table by linear interpolation; the soft walls likewise.  Neither enters
``floc``, which is built from the System's physical forces only.  Each step writes
``(xi, floc)`` of the configuration at the start of the step into a per-DOF ring buffer, so
the host receives the full 1 fs time series once per chunk and owns all estimator decisions.

Sign convention, asserted by the tests: ``floc = +dV/dxi`` (the mean force ``F'(xi)`` is its
conditional mean); the physical generalised force on xi is ``-floc``.
"""
from __future__ import annotations

import numpy as np

from . import geometry
from . import system as csys

KB = 8.31446261815324e-3            # kJ/mol/K

#: frozen PME at the frozen box (results/c60/box/pme_params.json, SPEC §1.4); restated in
#: docs/C60_EXECUTION_STATE.md, which is where the reconstruction after the host move reads it
FROZEN_LX_NM = 2.651139
FROZEN_LZ_NM = 5.673840
FROZEN_PME = (2.628260884878466, 24, 24, 48)


def frozen_box():
    return np.array([FROZEN_LX_NM, FROZEN_LX_NM, FROZEN_LZ_NM])


def box_vectors(L):
    import openmm as mm
    import openmm.unit as u
    return tuple(mm.Vec3(*row) for row in np.diag(np.asarray(L, float))) * u.nanometer


def platform_and_props(name="CUDA", precision="mixed", device=None):
    import openmm as mm
    plat = mm.Platform.getPlatformByName(name)
    props = {}
    if name in ("CUDA", "OpenCL"):
        props["Precision"] = precision
        if device is not None:
            props["DeviceIndex"] = str(device)
    return plat, props


# ---------------------------------------------------------------------------------- systems
class Layout:
    """Index structure of a built system (cages, waters, extra xi particles)."""

    def __init__(self, params, n_base, xi_mode):
        self.cage_a = np.asarray(params["cage_a"])
        self.cage_b = np.asarray(params["cage_b"])
        self.carbon = np.asarray(params["carbon_index"])
        self.waters = np.asarray(params["waters"])
        self.oxygen = self.waters[:, 0]
        self.n_base = int(n_base)
        self.xi_mode = bool(xi_mode)
        if xi_mode:
            self.P = n_base
            self.ghosts = (n_base + 1, n_base + 2, n_base + 3)
            self.n = n_base + 4
        else:
            self.P = None
            self.ghosts = ()
            self.n = n_base


def build(topology, mode="fixed", L=None, pme=FROZEN_PME):
    """``(system, layout)`` at the frozen box; ``mode`` is ``"fixed"`` or ``"xi"``."""
    import openmm as mm
    import openmm.unit as u

    L = frozen_box() if L is None else np.asarray(L, float)
    system = csys.build_system(topology, box_vectors=box_vectors(L), pme_params=pme,
                               fix_cages=True)
    params = csys.site_parameters(system, topology)
    n_base = system.getNumParticles()
    lay = Layout(params, n_base, mode == "xi")
    if mode == "fixed":
        return system, lay
    if mode != "xi":
        raise ValueError(mode)

    nbf = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    P = system.addParticle(2.0 * csys.MASS_CAGE_AMU * u.amu)
    ghosts = [system.addParticle(0.0) for _ in range(3)]
    for k in [P] + ghosts:
        nbf.addParticle(0.0, 1.0, 0.0)
    assert (P, *ghosts) == (lay.P, *lay.ghosts)
    cage = geometry.c60_cage()
    cage = cage - cage.mean(axis=0)
    g0, g1, g2 = ghosts
    for local, idx, origin in ((cage, lay.cage_b, [1.0, 0.0, 0.0, 0.0]),
                               (cage, lay.cage_a, [-1.0, 2.0, 0.0, 0.0])):
        for k, i in enumerate(idx):
            site = mm.LocalCoordinatesSite([P, g0, g1, g2], origin,
                                           [0.0, -1.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0],
                                           mm.Vec3(*local[k]))
            system.setVirtualSite(int(i), site)
    return system, lay


def xi_positions(base_pos, lay, xi, L=None):
    """Full position array for xi mode: cages placed at ``xi`` about the box centre."""
    L = frozen_box() if L is None else np.asarray(L, float)
    c = 0.5 * L
    pos = np.zeros((lay.n, 3))
    pos[:lay.n_base] = base_pos[:lay.n_base]
    pos[lay.carbon] = geometry.pair_positions(float(xi), c)
    pos[lay.P] = c + np.array([0.0, 0.0, 0.5 * float(xi)])
    pos[lay.ghosts[0]] = c
    pos[lay.ghosts[1]] = c + np.array([1.0, 0.0, 0.0])
    pos[lay.ghosts[2]] = c + np.array([0.0, 1.0, 0.0])
    return pos


# ------------------------------------------------------------------------------ integrators
def fixed_integrator(dt_ps, lay, T=csys.TEMPERATURE_K, gamma=csys.GAMMA_PS, u_every=0):
    """LangevinMiddle (OpenMM's documented CustomIntegrator form) + device-side floc sums.

    Globals ``fsum, f2sum, nf`` accumulate ``floc`` every step; with ``u_every > 0`` the
    potential energy is accumulated every ``u_every`` steps into ``usum, nu``.  The host
    resets them per block (:func:`read_reset`).
    """
    import openmm as mm

    ig = mm.CustomIntegrator(float(dt_ps))
    ig.addGlobalVariable("a", float(np.exp(-gamma * dt_ps)))
    ig.addGlobalVariable("b", float(np.sqrt(1.0 - np.exp(-2.0 * gamma * dt_ps))))
    ig.addGlobalVariable("kT", KB * float(T))
    for g in ("floc", "fsum", "f2sum", "nf", "usum", "nu", "kstep"):
        ig.addGlobalVariable(g, 0.0)
    ig.addPerDofVariable("wf", 0.0)
    ig.addPerDofVariable("x1", 0.0)
    ig.addUpdateContextState()
    ig.addComputeSum("floc", "f*wf")
    ig.addComputeGlobal("fsum", "fsum + floc")
    ig.addComputeGlobal("f2sum", "f2sum + floc*floc")
    ig.addComputeGlobal("nf", "nf + 1")
    if u_every and u_every > 0:
        ig.addComputeGlobal("kstep", "kstep + 1")
        ig.beginIfBlock(f"kstep >= {int(u_every)}")
        ig.addComputeGlobal("usum", "usum + energy")
        ig.addComputeGlobal("nu", "nu + 1")
        ig.addComputeGlobal("kstep", "0")
        ig.endBlock()
    ig.addComputePerDof("v", "v + dt*f/m")
    ig.addConstrainVelocities()
    ig.addComputePerDof("x", "x + 0.5*dt*v")
    ig.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
    ig.addComputePerDof("x", "x + 0.5*dt*v")
    ig.addComputePerDof("x1", "x")
    ig.addConstrainPositions()
    ig.addComputePerDof("v", "v + (x-x1)/dt")
    return ig


def set_fixed_masks(ig, lay):
    import openmm as mm
    wf = [mm.Vec3(0, 0, 0)] * lay.n
    for i in lay.cage_a:
        wf[int(i)] = mm.Vec3(0, 0, 0.5)
    for i in lay.cage_b:
        wf[int(i)] = mm.Vec3(0, 0, -0.5)
    ig.setPerDofVariableByName("wf", wf)


def xi_integrator(dt_ps, lay, n_grid, lo, hi, k_wall, ring, T=csys.TEMPERATURE_K,
                  gamma=csys.GAMMA_PS, L=None):
    """LangevinMiddle on waters + the xi particle, with the ABF table, walls and a ring buffer.

    Per-DOF variables set by the host (:func:`set_xi_masks`): ``mask`` (1 on water DOFs,
    ``(0,0,1)`` on P, 0 on massless particles), ``invm`` (= mask/m, 0 where massless),
    ``wP`` (``(0,0,1)`` on P), ``wf`` (the floc weights, as in fixed mode), ``slot`` (the
    particle index on all three components; ``cx``/``cy`` pick the component) and the bias table ``btab`` (x component of
    slots ``0..n_grid-1``: the generalised ABF force ``+A'_hat`` at the grid nodes).

    Ring buffer ``ring_buf``: at step ``k`` of a chunk (``kslot = k``), slot ``k`` receives
    ``(xi, floc, 0)`` of the configuration at the start of the step.  The chunk length must
    not exceed ``ring`` (asserted by the driver).
    """
    import openmm as mm

    L = frozen_box() if L is None else np.asarray(L, float)
    cz = 0.5 * float(L[2])
    dz = (hi - lo) / (n_grid - 1)
    ig = mm.CustomIntegrator(float(dt_ps))
    ig.addGlobalVariable("a", float(np.exp(-gamma * dt_ps)))
    ig.addGlobalVariable("b", float(np.sqrt(1.0 - np.exp(-2.0 * gamma * dt_ps))))
    ig.addGlobalVariable("kT", KB * float(T))
    for g in ("xi", "floc", "gb", "gw", "kslot", "kb", "wb", "zP"):
        ig.addGlobalVariable(g, 0.0)
    for name in ("mask", "invm", "wP", "wf", "slot", "btab", "x1", "ring_buf", "cx", "cy"):
        ig.addPerDofVariable(name, 0.0)
    ig.addUpdateContextState()
    # --- the sample at the start of the step: xi and floc of the current configuration
    ig.addComputeSum("zP", "x*wP")
    ig.addComputeGlobal("xi", f"2*(zP - {cz!r})")
    ig.addComputeSum("floc", "f*wf")
    ig.addComputePerDof("ring_buf", "ring_buf + delta(slot - kslot)*(cx*xi + cy*floc)")
    ig.addComputeGlobal("kslot", "kslot + 1")
    # --- ABF bias (linear interpolation on the node table, clamped to the domain)
    ig.addComputeGlobal("kb", f"min(max(floor((xi - {lo!r})/{dz!r}), 0), {n_grid - 2})")
    ig.addComputeGlobal("wb", f"min(max((xi - {lo!r})/{dz!r} - kb, 0), 1)")
    ig.addComputeSum("gb", "btab*((1-wb)*delta(slot - kb) + wb*delta(slot - kb - 1))")
    # --- soft walls (generalised force on xi)
    ig.addComputeGlobal("gw", f"-{k_wall!r}*(step(xi - {hi!r})*(xi - {hi!r}) "
                              f"- step({lo!r} - xi)*({lo!r} - xi))")
    # --- BAOAB (fused kick); P gets 2*(bias + wall) on z
    ig.addComputePerDof("v", "v + dt*(f + 2*(gb + gw)*wP)*invm")
    ig.addConstrainVelocities()
    ig.addComputePerDof("x", "x + 0.5*dt*v*mask")
    ig.addComputePerDof("v", "mask*(a*v + b*sqrt(kT*invm)*gaussian)")
    ig.addComputePerDof("x", "x + 0.5*dt*v*mask")
    ig.addComputePerDof("x1", "x")
    ig.addConstrainPositions()
    ig.addComputePerDof("v", "v + (x-x1)/dt")
    return ig


def set_xi_masks(ig, system, lay):
    import openmm as mm
    import openmm.unit as u
    n = lay.n
    mass = np.array([system.getParticleMass(i).value_in_unit(u.amu) for i in range(n)])
    mask = np.zeros((n, 3))
    mask[mass > 0] = 1.0
    mask[lay.P] = (0.0, 0.0, 1.0)
    invm = np.where(mass[:, None] > 0, mask / np.where(mass > 0, mass, 1.0)[:, None], 0.0)
    wP = np.zeros((n, 3)); wP[lay.P, 2] = 1.0
    wf = np.zeros((n, 3)); wf[lay.cage_a, 2] = 0.5; wf[lay.cage_b, 2] = -0.5
    slot = np.repeat(np.arange(n, dtype=float)[:, None], 3, axis=1)
    cx = np.zeros((n, 3)); cx[:, 0] = 1.0
    cy = np.zeros((n, 3)); cy[:, 1] = 1.0
    for name, arr in (("mask", mask), ("invm", invm), ("wP", wP), ("wf", wf),
                      ("slot", slot), ("cx", cx), ("cy", cy)):
        ig.setPerDofVariableByName(name, [mm.Vec3(*r) for r in arr])
    return mass


def set_bias_table(ig, table, n):
    import openmm as mm
    t = np.zeros(n)
    t[:len(table)] = np.asarray(table, float)
    ig.setPerDofVariableByName("btab", [mm.Vec3(float(v), 0.0, 0.0) for v in t])


def read_ring(ig, k):
    """``(xi, floc)`` arrays of the first ``k`` ring slots; clears the ring and ``kslot``."""
    import openmm as mm
    buf = np.asarray([tuple(v) for v in ig.getPerDofVariableByName("ring_buf")])
    ig.setPerDofVariableByName("ring_buf", [mm.Vec3(0, 0, 0)] * buf.shape[0])
    ig.setGlobalVariableByName("kslot", 0.0)
    return buf[:k, 0].copy(), buf[:k, 1].copy()


def read_reset(ig, names=("fsum", "f2sum", "nf", "usum", "nu")):
    out = {g: ig.getGlobalVariableByName(g) for g in names}
    for g in names:
        ig.setGlobalVariableByName(g, 0.0)
    return out
