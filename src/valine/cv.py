"""Candidate collective variables for Ace-Val-Nme.

The selected CV stays **two-dimensional**.  That is the single most important cost fact in
this study: `BackboneCV2D` is already parameterised by its two atom quadruples, and every
piece of downstream machinery -- the FFT Poisson projection, the torus KDE, the birth-death
score, the watershed basins, the rank-4 MBAR factorisation -- is 2-D.  Choosing a 2-D CV
that happens to contain chi1 therefore reuses the validated alanine pipeline unchanged, and
an umbrella reference costs 24x24 = 576 windows exactly as alanine's did.

A genuinely 3-D CV would instead need `fftn`, a trilinear interpolator, a rank-6 MBAR basis,
a 97^3 = 912673-cell grid per seed, and 24^3 = 13824 umbrella windows.  Nothing in this plan
requires that, and it should not be built speculatively.

Three candidates, in the order the screening plan tries them:

  ``phi_chi1``  (phi, chi1) -- hidden coordinate psi
  ``psi_chi1``  (psi, chi1) -- hidden coordinate phi
  ``phi_psi``   (phi, psi)  -- hidden coordinate chi1; the alanine CV, retained as the
                direct control.  This is the pairing the *original* frozen design used, and
                it is the one the decision doc's gate V2 was written about.

**Standing prediction, recorded before the screen is run.**  ``psi_chi1`` is expected to fail
the hidden-coordinate mixing gate, because the coordinate it hides is phi, and phi carries the
dominant backbone barrier -- for alanine the min-max barrier between the phi<0 megabasin and
C7ax is 15.79 kT.  Under ``phi_chi1`` that barrier is biased by ABF and the hidden coordinate
is psi, which alanine's accepted reference showed carries at most 0.75 kT at every populated
phi.  If this prediction fails it is informative, which is why it is written down here rather
than after the fact.
"""
from __future__ import annotations

from alanine.cv2d import BackboneCV2D

from .system import CHI1_ATOMS, N_ATOMS, PHI_ATOMS, PSI_ATOMS

#: name -> (atoms_a, atoms_b, hidden-coordinate name)
CANDIDATES = {
    'phi_chi1': (PHI_ATOMS, CHI1_ATOMS, 'psi'),
    'psi_chi1': (PSI_ATOMS, CHI1_ATOMS, 'phi'),
    'phi_psi':  (PHI_ATOMS, PSI_ATOMS, 'chi1'),
}

#: column index of each angle in the ``(..., 3)`` arrays returned by ``system.angles_np``
ANGLE_COLUMN = {'phi': 0, 'psi': 1, 'chi1': 2}

AXIS_LABELS = {
    'phi_chi1': (r'$\phi$', r'$\chi_1$'),
    'psi_chi1': (r'$\psi$', r'$\chi_1$'),
    'phi_psi':  (r'$\phi$', r'$\psi$'),
}


def make_cv(name, ridge=0.0, reg_threshold=1e-8):
    """Return the `BackboneCV2D` for a named candidate.

    IUPAC convention throughout (trans at pi), inherited from `BackboneCV2D`, which overrides
    only `values()` -- gradients, Hessians, the metric G, the dual fields and the den Otter
    local mean force are numerically identical to the underlying RB convention because the
    shift is constant.
    """
    if name not in CANDIDATES:
        raise KeyError(f"unknown CV {name!r}; choose from {sorted(CANDIDATES)}")
    a, b, _ = CANDIDATES[name]
    return BackboneCV2D(a, b, n_atoms=N_ATOMS, ridge=ridge, reg_threshold=reg_threshold)


def hidden_coordinate(name):
    """Name of the angle a candidate CV does *not* resolve."""
    return CANDIDATES[name][2]


def cv_columns(name):
    """Columns of ``system.angles_np`` output corresponding to a candidate's two axes."""
    a, b, _ = CANDIDATES[name]
    inv = {PHI_ATOMS: 'phi', PSI_ATOMS: 'psi', CHI1_ATOMS: 'chi1'}
    return ANGLE_COLUMN[inv[a]], ANGLE_COLUMN[inv[b]]


def union_size(name):
    """Number of distinct atoms the two dihedrals touch.

    Relevant only to the `FastBackboneCV2D` union optimisation, which is tested but not wired
    into production.  For (phi, chi1) the union is {4,6,8,20} u {6,8,10,12} = 6 atoms / 18 of
    84 coordinates, so the union trick would pay off more here than it did for alanine (5/22).
    """
    a, b, _ = CANDIDATES[name]
    return len(set(a) | set(b))
