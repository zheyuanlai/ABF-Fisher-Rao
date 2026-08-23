"""Frozen molecular systems for the RC-WFR campaign.

Each entry fixes topology, temperature, integrator step and CV once and for all.
`h` is the Brownian step dt/gamma; it is chosen so that the stiffest harmonic
mode has h * k / m well below 1 (the projected-Euler stationary variance is
inflated by 1/(1 - h k / 2m), so 0.074 buys < 4% on bond lengths and far less
on anything the torsion cares about).  Reference and every arm use the SAME h,
so the comparison never mixes integrator biases.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

import math

from ..grid import Grid1D
from .ff import KB, Topology, ua_alkane, ideal_alkane
from .geom import TorsionCV


@dataclass
class MolSystem:
    tag: str
    top: Topology
    cv: TorsionCV
    beta: float
    h: float
    n_c: int
    grid: Grid1D                      # grid for the PRIMARY CV z
    y_grid: Grid1D = None             # grid for the secondary CV y (if any)
    T: float = 300.0
    # exact internal-coordinate lift: rotate `movers` about the bond (j, k).
    # For a linear alkane torsion (i, j, k, l) the distal fragment is every
    # atom past k, and rotating it changes THAT dihedral and nothing else.
    z_bond: tuple = (1, 2)
    z_movers: tuple = ()
    y_bond: tuple = (2, 3)
    y_movers: tuple = ()
    # every candidate secondary CV: (torsion index, rotation bond, moving atoms).
    # Rotating the distal fragment about torsion k's central bond changes ONLY
    # torsion k -- the other dihedrals' two defining planes are both carried by
    # the same rotation, so the angle between them is invariant.  The promoted
    # modes can therefore be set independently and in any order.
    y_specs: tuple = ()
    ideal_fn: object = None           # override the alkane NeRF builder
    drift_cap: float = None           # see dynamics.free_step
    y0: float = 0.0                   # cold-start value of every fiber torsion
    cv_shift: float = 0.0             # constant offset baked into the CV

    @property
    def device(self):
        return self.top.mass.device

    @property
    def dtype(self):
        return self.top.mass.dtype

    def ideal(self, phis):
        if self.ideal_fn is not None:
            return self.ideal_fn(phis)
        return ideal_alkane(self.top, self.n_c, phis, self.device, self.dtype)


def periodic_grid(n=129):
    import math
    return Grid1D(-math.pi, math.pi, n, -math.pi, math.pi, bc="periodic")


def arc_grid(half_width, n=129):
    """Reflecting grid on [-w, w]: a CV with an inaccessible arc."""
    return Grid1D(-half_width, half_width, n, -half_width, half_width, bc="reflect")


def butane(device, dtype=torch.float64, T=300.0, h=0.002, n_grid=129):
    top = ua_alkane(4, device, dtype)
    cv = TorsionCV(top.tor_idx, top.mass)          # the single central torsion
    return MolSystem("BUT", top, cv, 1.0 / (KB * T), h, 4, periodic_grid(n_grid), T=T,
                     z_bond=(1, 2), z_movers=(3,))


def pentane(device, dtype=torch.float64, T=300.0, h=0.002, n_grid=129, m=1):
    """m=1: z = phi1, fiber contains phi2 (the slow hidden mode).
       m=2: z = (phi1, phi2)  -- the complete-CV control."""
    top = ua_alkane(5, device, dtype)
    idx = top.tor_idx[:m]
    cv = TorsionCV(idx, top.mass)
    g = periodic_grid(n_grid)
    return MolSystem("PEN", top, cv, 1.0 / (KB * T), h, 5, g,
                     y_grid=periodic_grid(n_grid), T=T,
                     z_bond=(1, 2), z_movers=(3, 4), y_bond=(2, 3), y_movers=(4,),
                     y_specs=((1, (2, 3), (4,)),))


def pentane_full(device, dtype=torch.float64, T=300.0, h=0.002):
    """Both torsions as a 2-CV constraint: used for the y-oracle lift, which has
    to place a configuration at a prescribed (z, y) before releasing y."""
    top = ua_alkane(5, device, dtype)
    return TorsionCV(top.tor_idx, top.mass)


def hexane(device, dtype=torch.float64, T=300.0, h=0.002, n_grid=129, m=1):
    """z = phi1; the fiber now holds TWO slow torsions.

    phi2 is adjacent to phi1 and shares the 1-5 CH3...CH2 contact with it;
    phi3 is one bond further out and reaches phi1 only through the 1-6 pair.
    Both relax on the same torsional timescale, so promoting one and not the
    other separates COUPLING from TIMESCALE as the criterion for which fiber
    mode has to be transported -- the toy campaign only tested timescale.
    """
    top = ua_alkane(6, device, dtype)
    cv = TorsionCV(top.tor_idx[:m], top.mass)
    g = periodic_grid(n_grid)
    return MolSystem("HEX", top, cv, 1.0 / (KB * T), h, 6, g,
                     y_grid=periodic_grid(n_grid), T=T,
                     z_bond=(1, 2), z_movers=(3, 4, 5),
                     y_bond=(2, 3), y_movers=(4, 5),
                     y_specs=((1, (2, 3), (4, 5)), (2, (3, 4), (5,))))


_ALA_CACHE = {}


def alanine2d(device, dtype=torch.float64, T=300.0, n_grid=97):
    """z = (phi, psi): the COMPLETE-coordinate control.

    Same molecule, same shift and same restricted phi arc as the 1-CV version, so
    the two are directly comparable; psi keeps its full period.
    """
    from .grid2d import Grid2D
    sy = alanine(device, dtype, T=T, m=2)
    g2 = Grid2D(arc_grid(80.0 * math.pi / 180.0, n_grid), periodic_grid(n_grid))
    return sy, g2


def alanine(device, dtype=torch.float64, T=300.0, h=None, n_grid=129, m=1):
    """Ace-Ala-Nme in vacuum, ff14SB.  z = phi, the hidden slow mode is psi.

    Units switch to kJ/mol and nm here (OpenMM's), so `h` and `beta` are not
    comparable numbers with the alkanes'; the physics that IS comparable -- the
    torsional diffusion per step -- is matched by construction (see alanine.py).
    """
    from . import alanine as A
    from .ff import rotate_about_bond, _wrap
    key = (str(device), str(dtype))
    if key not in _ALA_CACHE:
        system, X0 = A.reference_minimum()
        P = A.extract_parameters(system)
        top = A.AlaTopology(P, device, dtype)
        _ALA_CACHE[key] = (top, torch.as_tensor(X0, device=device, dtype=dtype))
    top, X0 = _ALA_CACHE[key]
    # phi has a ~96-degree arc that carries no Boltzmann weight at all (F > 18 kT)
    # and, on the far side of it, the C7ax basin, which sits behind a ~14 kT
    # barrier.  Neither is reachable by unbiased dynamics, and a periodic
    # cumulative TI integral would have to cross both.  The CV is therefore
    # rotated by -105 degrees and the domain restricted to the ergodic arc
    # [-80, +80] around it, which holds C7eq, the beta/C5 region and the barrier
    # between them -- the standard negative-phi half of the Ramachandran map.
    SHIFT = -105.0 * math.pi / 180.0
    cv = TorsionCV(top.tor_idx[:m], top.mass, shift=SHIFT)
    full = TorsionCV(top.tor_idx, top.mass, shift=SHIFT)
    if h is None:
        # match the alkanes' stability margin: h * k_max / mu ~ 0.15
        h = 0.15 * float(top.mass[0]) / 2.0 / float(top.bk.max())
    g = arc_grid(80.0 * math.pi / 180.0, n_grid)

    def _ideal(phis):
        x = X0.expand(phis.shape[0], -1, -1).clone()
        cur = full.value(x)
        x = rotate_about_bond(x, A.PHI_BOND[0], A.PHI_BOND[1], list(A.PHI_MOVING),
                              -_wrap(phis[..., 0] - cur[..., 0]))
        if phis.shape[-1] > 1:
            cur = full.value(x)
            x = rotate_about_bond(x, A.PSI_BOND[0], A.PSI_BOND[1], list(A.PSI_MOVING),
                                  -_wrap(phis[..., 1] - cur[..., 1]))
        return x

    return MolSystem("ALA", top, cv, 1.0 / (A.KB_KJ * T), h, 22, g,
                     y_grid=periodic_grid(n_grid), T=T, cv_shift=SHIFT,
                     z_bond=A.PHI_BOND, z_movers=A.PHI_MOVING,
                     y_bond=A.PSI_BOND, y_movers=A.PSI_MOVING,
                     y_specs=((1, A.PSI_BOND, A.PSI_MOVING),),
                     ideal_fn=_ideal, drift_cap=20.0, y0=-0.9)


def heptane(device, dtype=torch.float64, T=300.0, h=0.002, n_grid=129, m=1):
    """z = phi1; THREE candidate hidden torsions, at increasing distance from z.

    Extends the hexane contrast into a family, so the S_k tau_k^2 diagnostic can
    be checked against measured promotion gains over more than one contrast.
    """
    top = ua_alkane(7, device, dtype)
    cv = TorsionCV(top.tor_idx[:m], top.mass)
    g = periodic_grid(n_grid)
    return MolSystem("HEP", top, cv, 1.0 / (KB * T), h, 7, g,
                     y_grid=periodic_grid(n_grid), T=T,
                     z_bond=(1, 2), z_movers=(3, 4, 5, 6),
                     y_bond=(2, 3), y_movers=(4, 5, 6),
                     y_specs=((1, (2, 3), (4, 5, 6)),
                              (2, (3, 4), (5, 6)),
                              (3, (4, 5), (6,))))


REGISTRY = {"BUT": butane, "PEN": pentane, "HEX": hexane, "HEP": heptane,
            "ALA": alanine}
