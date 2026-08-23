"""Alanine dipeptide (Ace-Ala-Nme, 22 atoms, AMBER ff14SB, vacuum).

OpenMM builds the system and supplies the parameters; the hot loop is the same
batched torch code the alkanes use.  Parity against OpenMM is checked in
`scripts/mol_ala_gate.py` and is a gate, not an assumption.

MASSES ARE SET UNIFORM (12 amu).  Masses do not appear in e^{-beta V}, so F(z)
is EXACTLY unchanged; only the Brownian kinetics are.  Without this the X-H
bonds (k ~ 2.5e5 kJ/mol/nm^2 against m = 1) force h <~ 4e-7 and the torsional
diffusion per step drops by two orders of magnitude, for no gain in the object
being computed.  With uniform masses the mean force is mass-free outright
(w = grad xi/|grad xi|^2) and alanine's torsional diffusion per step matches
pentane's to within a few percent, so the two systems share a budget scale.

Topology tables and the NeRF reference-minimum builder are vendored from the
author's earlier alanine campaign (see docs/PROVENANCE.md).

Units: kJ/mol, nm, amu, radian.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

ONE_4PI_EPS0 = 138.935456          # kJ/mol nm e^-2
KB_KJ = 0.008314462618             # kJ/mol/K

# amber14 order: ACE(HH31 CH3 HH32 HH33 C O) ALA(N H CA HA CB HB1 HB2 HB3 C O)
#                NME(N H CH3 HH31 HH32 HH33)
NAMES = [('ACE', 'HH31', 'H'), ('ACE', 'CH3', 'C'), ('ACE', 'HH32', 'H'),
         ('ACE', 'HH33', 'H'), ('ACE', 'C', 'C'), ('ACE', 'O', 'O'),
         ('ALA', 'N', 'N'), ('ALA', 'H', 'H'), ('ALA', 'CA', 'C'), ('ALA', 'HA', 'H'),
         ('ALA', 'CB', 'C'), ('ALA', 'HB1', 'H'), ('ALA', 'HB2', 'H'),
         ('ALA', 'HB3', 'H'), ('ALA', 'C', 'C'), ('ALA', 'O', 'O'),
         ('NME', 'N', 'N'), ('NME', 'H', 'H'), ('NME', 'CH3', 'C'),
         ('NME', 'HH31', 'H'), ('NME', 'HH32', 'H'), ('NME', 'HH33', 'H')]
BONDS = [(0, 1), (1, 2), (1, 3), (1, 4), (4, 5), (4, 6), (6, 7), (6, 8), (8, 9),
         (8, 10), (10, 11), (10, 12), (10, 13), (8, 14), (14, 15), (14, 16),
         (16, 17), (16, 18), (18, 19), (18, 20), (18, 21)]
PHI_ATOMS = (4, 6, 8, 14)          # C(ACE) N CA C(ALA)
PSI_ATOMS = (6, 8, 14, 16)         # N CA C(ALA) N(NME)
PHI_BOND, PHI_MOVING = (6, 8), (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
PSI_BOND, PSI_MOVING = (8, 14), (15, 16, 17, 18, 19, 20, 21)


def _nerf(a, b, c, r, theta, phi):
    theta, phi = np.radians(theta), np.radians(phi)
    bc = c - b; bc = bc / np.linalg.norm(bc)
    n = np.cross(b - a, bc); n = n / np.linalg.norm(n)
    m = np.cross(n, bc)
    return c + (-r * np.cos(theta)) * bc + (r * np.sin(theta) * np.cos(phi)) * m \
        + (r * np.sin(theta) * np.sin(phi)) * n


def build_positions(phi_deg=-80.0, psi_deg=80.0, cb_offset=-120.0):
    """One structure only -- the reference minimum.  Every other (phi, psi) is
    reached from it by RIGID rotation, which preserves bonds, angles and
    chirality exactly; the builder itself places the ACE carbonyl O from the
    wrong frame over much of the torus and must not be used as a seeder."""
    X = np.zeros((22, 3))
    X[1] = [0., 0., 0.]
    X[4] = [1.522, 0., 0.]
    a = np.radians(116.6)
    X[6] = X[4] + 1.335 * np.array([-np.cos(a), np.sin(a), 0.])
    X[8] = _nerf(X[1], X[4], X[6], 1.449, 121.9, 180.0)
    X[14] = _nerf(X[4], X[6], X[8], 1.522, 110.4, phi_deg)
    X[16] = _nerf(X[6], X[8], X[14], 1.335, 116.6, psi_deg)
    X[18] = _nerf(X[8], X[14], X[16], 1.449, 121.9, 180.0)
    X[5] = _nerf(X[8], X[6], X[4], 1.229, 122.9, 180.0)
    X[15] = _nerf(X[18], X[16], X[14], 1.229, 122.9, 180.0)
    X[10] = _nerf(X[4], X[6], X[8], 1.526, 110.5, phi_deg + cb_offset)
    X[9] = _nerf(X[4], X[6], X[8], 1.090, 108.0, phi_deg - cb_offset)
    X[7] = _nerf(X[1], X[4], X[6], 1.010, 119.0, 0.0)
    X[17] = _nerf(X[8], X[14], X[16], 1.010, 119.0, 0.0)
    for j, idx in enumerate((0, 2, 3)):
        X[idx] = _nerf(X[6], X[4], X[1], 1.090, 109.5, 60.0 + 120.0 * j)
    for j, idx in enumerate((11, 12, 13)):
        X[idx] = _nerf(X[6], X[8], X[10], 1.090, 109.5, 60.0 + 120.0 * j)
    for j, idx in enumerate((19, 20, 21)):
        X[idx] = _nerf(X[14], X[16], X[18], 1.090, 109.5, 60.0 + 120.0 * j)
    return X * 0.1                    # Angstrom -> nm


def make_openmm_system(ff_files=("amber14/protein.ff14SB.xml",)):
    import openmm.app as app
    top = app.Topology(); ch = top.addChain(); E = app.element
    el = {"H": "hydrogen", "C": "carbon", "N": "nitrogen", "O": "oxygen"}
    res, atoms = {}, []
    for rn, an, e in NAMES:
        if rn not in res:
            res[rn] = top.addResidue(rn, ch)
        atoms.append(top.addAtom(an, getattr(E, el[e]), res[rn]))
    for i, j in BONDS:
        top.addBond(atoms[i], atoms[j])
    ff = app.ForceField(*ff_files)
    return ff, top, ff.createSystem(top, nonbondedMethod=app.NoCutoff,
                                    constraints=None, rigidWater=False,
                                    removeCMMotion=False)


def extract_parameters(system):
    import openmm.unit as u
    P = {}
    for f in system.getForces():
        n = f.__class__.__name__
        if n == "HarmonicBondForce":
            b = [f.getBondParameters(i) for i in range(f.getNumBonds())]
            P["bonds"] = (np.array([[x[0], x[1]] for x in b]),
                          np.array([x[2].value_in_unit(u.nanometer) for x in b]),
                          np.array([x[3].value_in_unit(
                              u.kilojoule_per_mole / u.nanometer ** 2) for x in b]))
        elif n == "HarmonicAngleForce":
            a = [f.getAngleParameters(i) for i in range(f.getNumAngles())]
            P["angles"] = (np.array([[x[0], x[1], x[2]] for x in a]),
                           np.array([x[3].value_in_unit(u.radian) for x in a]),
                           np.array([x[4].value_in_unit(
                               u.kilojoule_per_mole / u.radian ** 2) for x in a]))
        elif n == "PeriodicTorsionForce":
            t = [f.getTorsionParameters(i) for i in range(f.getNumTorsions())]
            P["torsions"] = (np.array([[x[0], x[1], x[2], x[3]] for x in t]),
                             np.array([x[4] for x in t], dtype=np.int64),
                             np.array([x[5].value_in_unit(u.radian) for x in t]),
                             np.array([x[6].value_in_unit(u.kilojoule_per_mole) for x in t]))
        elif n == "NonbondedForce":
            m = f.getNumParticles()
            P["nb"] = (np.array([f.getParticleParameters(i)[0].value_in_unit(
                                     u.elementary_charge) for i in range(m)]),
                       np.array([f.getParticleParameters(i)[1].value_in_unit(
                                     u.nanometer) for i in range(m)]),
                       np.array([f.getParticleParameters(i)[2].value_in_unit(
                                     u.kilojoule_per_mole) for i in range(m)]))
            exc = [f.getExceptionParameters(i) for i in range(f.getNumExceptions())]
            P["exceptions"] = (np.array([[x[0], x[1]] for x in exc]),
                               np.array([x[2].value_in_unit(u.elementary_charge ** 2) for x in exc]),
                               np.array([x[3].value_in_unit(u.nanometer) for x in exc]),
                               np.array([x[4].value_in_unit(u.kilojoule_per_mole) for x in exc]))
        elif n == "CMMotionRemover":
            continue
        else:
            raise ValueError(f"unhandled force {n!r}")
    if system.getNumConstraints():
        raise ValueError("constraints present")
    return P


class AlaTopology:
    """Duck-typed like `mol.ff.Topology`: mass, n_atoms, tor_idx, energy, grad."""

    def __init__(self, P, device, dtype, uniform_mass=12.0):
        t = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=dtype)
        L = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=torch.long)
        self.name = "ALA-ff14SB"
        self.bi, self.b0, self.bk = L(P["bonds"][0]), t(P["bonds"][1]), t(P["bonds"][2])
        self.ai, self.a0, self.ak = L(P["angles"][0]), t(P["angles"][1]), t(P["angles"][2])
        self.ti, self.tn = L(P["torsions"][0]), t(P["torsions"][1])
        self.tp, self.tk = t(P["torsions"][2]), t(P["torsions"][3])
        q, sig, eps = P["nb"]
        n = len(q)
        self.n_atoms = n
        iu, ju = np.triu_indices(n, 1)
        exmap = {(min(int(i), int(j)), max(int(i), int(j))): (qq, ss, ee)
                 for (i, j), qq, ss, ee in zip(*P["exceptions"])}
        qq, ss, ee = q[iu] * q[ju], 0.5 * (sig[iu] + sig[ju]), np.sqrt(eps[iu] * eps[ju])
        for k, (i, j) in enumerate(zip(iu, ju)):
            if (int(i), int(j)) in exmap:
                qq[k], ss[k], ee[k] = exmap[(int(i), int(j))]
        keep = ~((qq == 0) & (ee == 0))
        self.pi, self.pj = L(iu[keep]), L(ju[keep])
        self.pq, self.ps, self.pe = t(qq[keep] * ONE_4PI_EPS0), t(ss[keep]), t(ee[keep])
        self.mass = torch.full((n,), uniform_mass, device=device, dtype=dtype)
        self.tor_idx = L([list(PHI_ATOMS), list(PSI_ATOMS)])
        self._gradfn = None

    def energy(self, x):
        """(..., 22, 3) nm -> (...) kJ/mol."""
        d = x[..., self.bi[:, 0], :] - x[..., self.bi[:, 1], :]
        E = (0.5 * self.bk * (d.norm(dim=-1) - self.b0) ** 2).sum(-1)
        v1 = x[..., self.ai[:, 0], :] - x[..., self.ai[:, 1], :]
        v2 = x[..., self.ai[:, 2], :] - x[..., self.ai[:, 1], :]
        th = torch.atan2(torch.linalg.cross(v1, v2, dim=-1).norm(dim=-1), (v1 * v2).sum(-1))
        E = E + (0.5 * self.ak * (th - self.a0) ** 2).sum(-1)
        p0, p1, p2, p3 = (x[..., self.ti[:, k], :] for k in range(4))
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1n = b1 / b1.norm(dim=-1, keepdim=True)
        vv = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
        ww = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
        ang = torch.atan2((torch.linalg.cross(b1n, vv, dim=-1) * ww).sum(-1), (vv * ww).sum(-1))
        E = E + (self.tk * (1.0 + torch.cos(self.tn * ang - self.tp))).sum(-1)
        rp = (x[..., self.pi, :] - x[..., self.pj, :]).norm(dim=-1).clamp_min(1e-8)
        sr6 = (self.ps / rp) ** 6
        return E + (4.0 * self.pe * (sr6 * sr6 - sr6) + self.pq / rp).sum(-1)

    def grad(self, x):
        if self._gradfn is None:
            self._gradfn = torch.func.grad(lambda y: self.energy(y).sum())
        return self._gradfn(x)


def reference_minimum(max_iterations=20000):
    """Minimised L-alanine C7eq structure (22, 3) in nm, plus the OpenMM system."""
    import openmm as mm
    import openmm.unit as u
    ff, top, system = make_openmm_system()
    ctx = mm.Context(system, mm.VerletIntegrator(0.001 * u.picoseconds),
                     mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(build_positions(-80.0, 80.0))
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=max_iterations)
    X0 = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
    return system, np.asarray(X0)
