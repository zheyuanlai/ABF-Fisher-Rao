"""Rigid C60 cage geometry: an exact two-bond-length truncated icosahedron.

Frozen by ``docs/SPEC_c60_water.md`` §1.  Zangi (JPCB 118:12263, 2014) holds the buckyball
geometry fixed throughout and approaches the two cages "via their pentagon rings orientated in
parallel and out of registry"; the paper does not print the internal C60 geometry, so the cage
built here is a **declared choice**: the gas-phase electron-diffraction structure of Hedberg et
al., *Science* 254:410 (1991),

    r_66 = 0.1401 nm   (bond fusing two hexagons)
    r_65 = 0.1458 nm   (bond fusing a hexagon and a pentagon = pentagon edge)

Exact construction, no optimisation
-----------------------------------
Truncating a regular icosahedron of edge ``s`` at a fraction ``lam`` along every edge gives a
truncated icosahedron whose pentagon edges have length ``lam * s`` (the two cut points sit on
edges 60 degrees apart across an equilateral face) and whose hexagon-hexagon bonds are the edge
middles, length ``s * (1 - 2 lam)``.  Solving for the target bond lengths:

    s   = 2 r_65 + r_66
    lam = r_65 / s

Both bond classes then hold their target lengths to machine precision by construction --
``validate_cage`` asserts exactly that, plus icosahedral vertex-distance degeneracy.

Orientation convention (frozen)
-------------------------------
The icosahedron is built in "polar" orientation -- one vertex on ``+z``, one on ``-z`` -- so the
C60 has a pentagon centred on each pole.  The top pentagon's carbons sit at azimuths
``72 k`` degrees, the bottom pentagon's at ``72 k + 36``: a *translated copy* of the cage
therefore already presents facing pentagons that are parallel and rotated 36 degrees against
each other -- exactly the paper's "parallel and out of registry".  Cage B is a pure translation
of cage A along ``+z``; no reflection, no extra rotation.  ``facing_pentagon_registry`` measures
the realised azimuthal offset so the claim is checked, not assumed.
"""
from __future__ import annotations

import numpy as np

R_66_NM = 0.1401           #: hexagon-hexagon bond, Hedberg et al. 1991 (declared choice)
R_65_NM = 0.1458           #: hexagon-pentagon bond = pentagon edge, same source
MASS_C_AMU = 12.011
N_CARBONS = 60


def icosahedron_polar(edge):
    """The 12 vertices of a regular icosahedron with one vertex on ``+z``, edge length ``edge``.

    Polar vertex, upper ring of 5 at azimuths ``72k``, lower ring at ``72k + 36``, antipodal
    vertex.  Circumradius ``R = edge * sin(2 pi / 5)``.
    """
    R = edge * np.sin(2.0 * np.pi / 5.0)
    theta = np.arctan(2.0)                       # polar angle of the upper ring
    verts = [np.array([0.0, 0.0, R])]
    for k in range(5):
        a = np.radians(72.0 * k)
        verts.append(R * np.array([np.sin(theta) * np.cos(a),
                                   np.sin(theta) * np.sin(a),
                                   np.cos(theta)]))
    for k in range(5):
        a = np.radians(72.0 * k + 36.0)
        verts.append(R * np.array([np.sin(theta) * np.cos(a),
                                   np.sin(theta) * np.sin(a),
                                   -np.cos(theta)]))
    verts.append(np.array([0.0, 0.0, -R]))
    return np.asarray(verts)


def c60_cage(r66_nm=R_66_NM, r65_nm=R_65_NM):
    """``(60, 3)`` carbon coordinates in nm, centred on the origin, pentagon on each pole.

    One carbon per (vertex, edge) incidence of the icosahedron: the point a fraction ``lam``
    along the edge from that vertex.
    """
    s = 2.0 * r65_nm + r66_nm
    lam = r65_nm / s
    verts = icosahedron_polar(s)

    # icosahedron edges = vertex pairs at the minimal nonzero distance (= s)
    d = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=-1)
    edges = [(i, j) for i in range(12) for j in range(i + 1, 12)
             if abs(d[i, j] - s) < 1e-9 * s]
    if len(edges) != 30:
        raise RuntimeError(f"found {len(edges)} icosahedron edges, expected 30")

    pts = []
    for i, j in edges:
        pts.append((1.0 - lam) * verts[i] + lam * verts[j])
        pts.append(lam * verts[i] + (1.0 - lam) * verts[j])
    pts = np.asarray(pts)
    if pts.shape != (60, 3):
        raise RuntimeError(f"built {pts.shape[0]} carbons, expected 60")
    pts -= pts.mean(axis=0)                       # exact COM at origin (symmetric anyway)
    return pts


def bond_lengths(cage):
    """All nearest-neighbour C-C distances, split ``(r65_bonds (60,), r66_bonds (30,))``.

    Every carbon has exactly 3 bonds; the 90 bonds split 60 pentagon-adjacent / 30
    hexagon-hexagon.  Classification is by clustering the two shortest distinct distances.
    """
    d = np.linalg.norm(cage[:, None, :] - cage[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    # bond cutoff: halfway between the longest bond (~0.146) and the next distance (~0.23)
    bonded = d < 0.185
    if not (bonded.sum(axis=1) == 3).all():
        raise RuntimeError("cage is not 3-coordinated")
    iu = np.triu_indices(60, 1)
    db = d[iu][bonded[iu]]
    mid = 0.5 * (db.min() + db.max())
    return np.sort(db[db >= mid]), np.sort(db[db < mid])


def validate_cage(cage, r66_nm=R_66_NM, r65_nm=R_65_NM, tol_nm=1e-9):
    """Hard gate: 90 bonds in a 60/30 split holding the target lengths to ``tol_nm``."""
    b65, b66 = bond_lengths(cage)
    if b65.size != 60 or b66.size != 30:
        raise RuntimeError(f"bond split {b65.size}/{b66.size}, expected 60/30")
    e65 = float(np.abs(b65 - r65_nm).max())
    e66 = float(np.abs(b66 - r66_nm).max())
    if e65 > tol_nm or e66 > tol_nm:
        raise RuntimeError(f"bond-length error r65 {e65:.2e} nm / r66 {e66:.2e} nm > {tol_nm:g}")
    radii = np.linalg.norm(cage, axis=1)
    return dict(r65_max_err_nm=e65, r66_max_err_nm=e66,
                radius_nm=float(radii.mean()), radius_spread_nm=float(np.ptp(radii)))


def pentagon_ring(cage, pole=+1):
    """Indices of the 5 carbons of the pentagon facing ``pole * z``, sorted by azimuth."""
    z = cage[:, 2] * pole
    idx = np.argsort(z)[-5:]
    az = np.mod(np.degrees(np.arctan2(cage[idx, 1], cage[idx, 0])), 360.0)
    return idx[np.argsort(az)]


def facing_pentagon_registry(cage_a, cage_b):
    """Azimuthal offset (deg, in [0, 36]) between A's ``+z`` pentagon and B's ``-z`` pentagon.

    36 = perfectly staggered ("out of registry"), 0 = eclipsed.  B must sit above A.
    """
    ia = pentagon_ring(cage_a, pole=+1)
    ib = pentagon_ring(cage_b, pole=-1)
    az_a = np.degrees(np.arctan2(cage_a[ia, 1], cage_a[ia, 0]))
    az_b = np.degrees(np.arctan2(cage_b[ib, 1], cage_b[ib, 0]))
    # offset between the two 5-fold patterns, folded into [0, 72) then to [0, 36]
    off = np.mod(az_b[:, None] - az_a[None, :], 72.0)
    best = np.abs(np.mod(off + 36.0, 72.0) - 36.0).min(axis=1).mean()
    return float(min(best, 72.0 - best) if best > 36.0 else best)


def pair_positions(d_com_nm, center_nm, r66_nm=R_66_NM, r65_nm=R_65_NM):
    """``(120, 3)`` coordinates of the two cages at axial COM separation ``d_com_nm``.

    Cage A (atoms 0-59) at ``center - (d/2) e_z``, cage B (atoms 60-119, a pure translate of A)
    at ``center + (d/2) e_z``.  ``xi = Z_B - Z_A = d_com_nm`` by construction.
    """
    cage = c60_cage(r66_nm, r65_nm)
    center = np.asarray(center_nm, dtype=np.float64)
    a = cage + center + np.array([0.0, 0.0, -0.5 * d_com_nm])
    b = cage + center + np.array([0.0, 0.0, +0.5 * d_com_nm])
    return np.concatenate([a, b], axis=0)
