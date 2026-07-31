"""Batched torch re-implementation of the AMBER ff14SB vacuum energy for Ace-Ala-Nme.

OpenMM builds the system and supplies the parameters; the hot loop is pure torch on
``(B, n_atoms, 3)`` in nm, returning kJ/mol.  Parity against OpenMM is a Stage-0 gate
(V1/V2): measured max relative error **1.04e-9** on energy and **3.03e-10** on forces over 24
thermally displaced configurations spanning E in [1923, 4823] kJ/mol.

Term counts for this system: 21 bonds, 36 angles, 42 periodic torsions, 98 nonbonded exceptions
(57 fully excluded + 41 scaled 1-4), 174 surviving pair interactions of 231, total charge
8.3e-17 e, zero constraints.

Units: kJ/mol, nm, ps, amu, radians.  ``1 kJ/mol == 1 amu nm^2 / ps^2`` (no 418.4 anywhere).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

ONE_4PI_EPS0 = 138.935456          # kJ/mol nm e^-2 (OpenMM's value)


def extract_parameters(system):
    """OpenMM ``System`` -> plain numpy arrays.  Raises if an unexpected force is present."""
    import openmm.unit as u
    P, seen = {}, []
    for f in system.getForces():
        n = f.__class__.__name__
        seen.append(n)
        if n == "HarmonicBondForce":
            b = [f.getBondParameters(i) for i in range(f.getNumBonds())]
            P["bonds"] = (np.array([[x[0], x[1]] for x in b]),
                          np.array([x[2].value_in_unit(u.nanometer) for x in b]),
                          np.array([x[3].value_in_unit(u.kilojoule_per_mole / u.nanometer ** 2) for x in b]))
        elif n == "HarmonicAngleForce":
            a = [f.getAngleParameters(i) for i in range(f.getNumAngles())]
            P["angles"] = (np.array([[x[0], x[1], x[2]] for x in a]),
                           np.array([x[3].value_in_unit(u.radian) for x in a]),
                           np.array([x[4].value_in_unit(u.kilojoule_per_mole / u.radian ** 2) for x in a]))
        elif n == "PeriodicTorsionForce":
            t = [f.getTorsionParameters(i) for i in range(f.getNumTorsions())]
            P["torsions"] = (np.array([[x[0], x[1], x[2], x[3]] for x in t]),
                             np.array([x[4] for x in t], dtype=np.int64),
                             np.array([x[5].value_in_unit(u.radian) for x in t]),
                             np.array([x[6].value_in_unit(u.kilojoule_per_mole) for x in t]))
        elif n == "NonbondedForce":
            m = f.getNumParticles()
            P["nb"] = (np.array([f.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in range(m)]),
                       np.array([f.getParticleParameters(i)[1].value_in_unit(u.nanometer) for i in range(m)]),
                       np.array([f.getParticleParameters(i)[2].value_in_unit(u.kilojoule_per_mole) for i in range(m)]))
            exc = [f.getExceptionParameters(i) for i in range(f.getNumExceptions())]
            P["exceptions"] = (np.array([[x[0], x[1]] for x in exc]),
                               np.array([x[2].value_in_unit(u.elementary_charge ** 2) for x in exc]),
                               np.array([x[3].value_in_unit(u.nanometer) for x in exc]),
                               np.array([x[4].value_in_unit(u.kilojoule_per_mole) for x in exc]))
        elif n == "CMMotionRemover":
            continue
        else:
            raise ValueError(f"unhandled force {n!r}; the torch re-implementation would be wrong")
    if system.getNumConstraints() != 0:
        raise ValueError("constraints present; the torch integrator does not implement SHAKE")
    P["masses"] = np.array([system.getParticleMass(i).value_in_unit(u.dalton)
                            for i in range(system.getNumParticles())])
    P["forces_present"] = seen
    return P


def parameter_hash(P):
    """Content hash of the parameter set, for run provenance."""
    payload = {k: (np.asarray(v).tolist() if not isinstance(v, tuple)
                   else [np.asarray(a).tolist() for a in v])
               for k, v in P.items() if k != "forces_present"}
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


class TorchFF(torch.nn.Module):
    """Batched ff14SB energy/forces.  ``energy(x)``: ``(B,n,3)`` nm -> ``(B,)`` kJ/mol."""

    def __init__(self, P, device="cpu", dtype=torch.float64):
        super().__init__()
        t = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=dtype)      # noqa: E731
        L = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=torch.long)  # noqa: E731
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
        qq = q[iu] * q[ju]
        ss = 0.5 * (sig[iu] + sig[ju])
        ee = np.sqrt(eps[iu] * eps[ju])
        for k, (i, j) in enumerate(zip(iu, ju)):
            key = (int(i), int(j))
            if key in exmap:
                qq[k], ss[k], ee[k] = exmap[key]
        keep = ~((qq == 0) & (ee == 0))                 # drop fully excluded 1-2 / 1-3 pairs
        self.pi, self.pj = L(iu[keep]), L(ju[keep])
        self.pq, self.ps, self.pe = t(qq[keep] * ONE_4PI_EPS0), t(ss[keep]), t(ee[keep])
        self.masses = t(P["masses"])

    def energy(self, x):
        d = x[:, self.bi[:, 0]] - x[:, self.bi[:, 1]]
        E = (0.5 * self.bk * (d.norm(dim=-1) - self.b0) ** 2).sum(-1)

        v1 = x[:, self.ai[:, 0]] - x[:, self.ai[:, 1]]
        v2 = x[:, self.ai[:, 2]] - x[:, self.ai[:, 1]]
        th = torch.atan2(torch.linalg.cross(v1, v2, dim=-1).norm(dim=-1), (v1 * v2).sum(-1))
        E = E + (0.5 * self.ak * (th - self.a0) ** 2).sum(-1)

        p0, p1, p2, p3 = (x[:, self.ti[:, k]] for k in range(4))
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1n = b1 / b1.norm(dim=-1, keepdim=True)
        vv = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
        ww = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
        ang = torch.atan2((torch.linalg.cross(b1n, vv, dim=-1) * ww).sum(-1), (vv * ww).sum(-1))
        E = E + (self.tk * (1.0 + torch.cos(self.tn * ang - self.tp))).sum(-1)

        rp = (x[:, self.pi] - x[:, self.pj]).norm(dim=-1).clamp_min(1e-8)
        sr6 = (self.ps / rp) ** 6
        return E + (4.0 * self.pe * (sr6 * sr6 - sr6) + self.pq / rp).sum(-1)

    def forces(self, x):
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            g, = torch.autograd.grad(self.energy(xg).sum(), xg)
        return -g
