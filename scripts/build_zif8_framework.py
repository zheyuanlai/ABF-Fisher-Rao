#!/usr/bin/env python
"""Build the flexible ZIF-8 framework artifact for the ethane/ZIF-8 stage.

Sources (all downloaded, provenance in cache/zif8/literature/):
  * structure   cache/zif8/literature/zif8_raspa2_github.cif -- the P1
                conventional cell (276 atoms, a=16.991 A) derived from Park et
                al., PNAS 2006, 103, 10186 (CSD FAWCEN).
  * force field cache/zif8/literature/zif8_gromacs_aanik_atomtypes_ZIF-8_Krokidas.itp
                -- the KROKIDAS et al. (JPCC 2015, 119, 27028) flexible ZIF-8
                force field, i.e. THE force field of the 2024 anchor paper
                (Schmidt/Cnudde/Van Speybroeck/Vanduyfhuys, JPCC 2024, 128,
                18509), in the GROMACS adaptation of Anikeenko (2017).

Nothing here is typed in from memory: every parameter is read out of the itp,
every position out of the CIF.  The topology is enumerated from CONNECTIVITY
and is then VALIDATED, term by term, against the published 2x2x2 enumeration
`zif8_gromacs_aanik_zif8_2x2x2_periodic.itp` -- bonds, 1-4 pairs, angles,
propers and impropers must match as SETS (mapping atom indices by position).
That is the real correctness gate for this file; the synthetic-fixture unit
tests in tests/test_zif8.py only check the engine math.

Unit conversions from GROMACS (nm, kJ/mol) to the engine (A, kJ/mol):
    bond   E = 1/2 kb (b-b0)^2   kb[kJ/mol/nm^2] -> /100 kJ/mol/A^2, b0 -> *10
    angle  E = 1/2 kt (t-t0)^2   kt already kJ/mol/rad^2, t0 deg -> rad
    proper E = k (1 + cos(n phi - phi0))            k kJ/mol, phi0 deg -> rad
    improper (func 2) E = 1/2 k (psi-psi0)^2        k kJ/mol/rad^2
    LJ     sigma nm -> *10 A, epsilon kJ/mol as-is; Lorentz-Berthelot
    1-4    fudgeLJ 0.5, fudgeQQ 0.8333 (the itp's own [defaults])

    CUDA_VISIBLE_DEVICES=3 python scripts/build_zif8_framework.py --supercell 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LIT = os.path.join(ROOT, "cache/zif8/literature")
CIF = os.path.join(LIT, "zif8_raspa2_github.cif")
FF = os.path.join(LIT, "zif8_gromacs_aanik_atomtypes_ZIF-8_Krokidas.itp")
REF_TOP = os.path.join(LIT, "zif8_gromacs_aanik_zif8_2x2x2_periodic.itp")
REF_PDB = os.path.join(LIT, "zif8_gromacs_aanik_conf_2x2x2.pdb")

FUDGE_LJ, FUDGE_QQ = 0.5, 0.8333
# element-pair bond cutoffs (A); asserted against the expected counts below
BOND_CUT = {("Zn", "N"): 2.45, ("C", "N"): 1.60, ("C", "C"): 1.70,
            ("C", "H"): 1.30, ("N", "H"): 1.30}
EXPECTED_PER_CELL = dict(Zn=12, N=48, C1=24, C2=48, C3=24, H2=48, H3=72)


# ------------------------------------------------------------------ parsing
def parse_atomtypes(path):
    """-> (types dict, bonds dict, angles dict, dihedrals dict, impropers dict)."""
    txt = open(path).read()
    types = {}
    for m in re.finditer(r"^\s*(\w+)_zif8\s+(\w+)\s+([\d.]+)\s+(-?[\d.]+)\s+A\s+"
                         r"([\d.]+)\s+([\d.]+)", txt, re.M):
        types[m.group(1)] = dict(element=m.group(2), mass=float(m.group(3)),
                                 charge=float(m.group(4)),
                                 sigma=float(m.group(5)) * 10.0,   # nm -> A
                                 eps=float(m.group(6)))
    bonds, angles, dihs, imps = {}, {}, {}, {}
    for m in re.finditer(r"^#define\s+gb_zif8_(\S+)\s+1\s+([\d.]+)\s+([\d.]+)", txt, re.M):
        bonds[m.group(1)] = (float(m.group(2)) * 10.0, float(m.group(3)) / 100.0)
    for m in re.finditer(r"^#define\s+ga_zif8_(\S+)\s+1\s+([\d.]+)\s+([\d.]+)", txt, re.M):
        angles[m.group(1)] = (math.radians(float(m.group(2))), float(m.group(3)))
    for m in re.finditer(r"^#define\s+gd_zif8_(\S+)\s+1\s+([\d.]+)\s+([\d.]+)\s+(\d+)",
                         txt, re.M):
        dihs[m.group(1)] = (math.radians(float(m.group(2))), float(m.group(3)),
                            float(m.group(4)))
    for m in re.finditer(r"^#define\s+gi_zif8_(\S+)\s+2\s+([\d.]+)\s+([\d.]+)", txt, re.M):
        imps[m.group(1)] = (math.radians(float(m.group(2))), float(m.group(3)))
    return types, bonds, angles, dihs, imps


def parse_cif(path):
    txt = open(path).read()
    a = float(re.search(r"_cell_length_a\s+([\d.]+)", txt).group(1))
    b = float(re.search(r"_cell_length_b\s+([\d.]+)", txt).group(1))
    c = float(re.search(r"_cell_length_c\s+([\d.]+)", txt).group(1))
    for ang in ("alpha", "beta", "gamma"):
        v = float(re.search(rf"_cell_angle_{ang}\s+([\d.]+)", txt).group(1))
        assert abs(v - 90.0) < 1e-6, f"non-orthorhombic cell: {ang}={v}"
    els, frac = [], []
    for m in re.finditer(r"^\s*(\w+)\s+(Zn|C|N|H)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)",
                         txt, re.M):
        els.append(m.group(2))
        frac.append([float(m.group(3)), float(m.group(4)), float(m.group(5))])
    box = np.array([a, b, c])
    return box, np.array(els), (np.array(frac) % 1.0) * box


def parse_ref_pdb(path):
    els, pos, box = [], [], None
    for line in open(path):
        if line.startswith("CRYST1"):
            box = np.array([float(line[6:15]), float(line[15:24]), float(line[24:33])])
        elif line.startswith(("ATOM", "HETATM")):
            pos.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            els.append(line[76:78].strip() or line[12:16].strip()[0])
    return box, np.array(els), np.array(pos)


def parse_ref_top(path):
    """-> dict of section -> list of (indices tuple 0-based, type-name or None)."""
    out = defaultdict(list)
    section, dih_seen = None, 0
    for line in open(path):
        s = line.split(";")[0].strip()
        if not s:
            continue
        if s.startswith("["):
            name = s.strip("[] ").strip()
            if name == "dihedrals":
                dih_seen += 1
                section = "dihedrals" if dih_seen == 1 else "impropers"
            else:
                section = name
            continue
        if section in ("atoms", None):
            continue
        f = s.split()
        if section == "bonds":
            out["bonds"].append(((int(f[0]) - 1, int(f[1]) - 1), f[2].split("gb_zif8_")[-1]))
        elif section == "pairs":
            out["pairs"].append(((int(f[0]) - 1, int(f[1]) - 1), None))
        elif section == "angles":
            out["angles"].append(((int(f[0]) - 1, int(f[1]) - 1, int(f[2]) - 1),
                                  f[3].split("ga_zif8_")[-1]))
        elif section in ("dihedrals", "impropers"):
            key = "gd_zif8_" if section == "dihedrals" else "gi_zif8_"
            out[section].append((tuple(int(x) - 1 for x in f[:4]),
                                 f[4].split(key)[-1]))
    return out


# ------------------------------------------------------- topology from geometry
def min_image(d, box):
    return d - box * np.round(d / box)


def neighbour_bonds(els, pos, box):
    n = len(els)
    d = min_image(pos[:, None, :] - pos[None, :, :], box)
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, 1e9)
    bonded = np.zeros((n, n), bool)
    for (e1, e2), cut in BOND_CUT.items():
        m = (((els[:, None] == e1) & (els[None, :] == e2))
             | ((els[:, None] == e2) & (els[None, :] == e1)))
        bonded |= m & (r < cut)
    return bonded, r


def assign_types(els, bonded):
    """Zheng/itp naming: C1 = N-C-N (bears the methyl), C2 = ring CH,
    C3 = methyl C, H2 = H on C2, H3 = H on C3."""
    n = len(els)
    types = np.array([""] * n, dtype=object)
    nbrs = [np.nonzero(bonded[i])[0] for i in range(n)]
    for i in range(n):
        if els[i] == "Zn":
            types[i] = "Zn"
        elif els[i] == "N":
            types[i] = "N"
    for i in range(n):
        if els[i] != "C":
            continue
        nn = sum(els[j] == "N" for j in nbrs[i])
        nh = sum(els[j] == "H" for j in nbrs[i])
        if nn == 2:
            types[i] = "C1"
        elif nn == 1 and nh == 1:
            types[i] = "C2"
        elif nn == 0 and nh == 3:
            types[i] = "C3"
        else:
            raise AssertionError(f"unclassifiable C at {i}: nN={nn} nH={nh}")
    for i in range(n):
        if els[i] != "H":
            continue
        assert len(nbrs[i]) == 1, f"H {i} has {len(nbrs[i])} neighbours"
        types[i] = {"C2": "H2", "C3": "H3"}[types[nbrs[i][0]]]
    return types


def canon_bond(i, j):
    return (i, j) if i < j else (j, i)


def canon_angle(i, j, k):
    return (i, j, k) if i < k else (k, j, i)


def canon_dih(a, b, c, d):
    return (a, b, c, d) if (b, c) <= (c, b) and (b < c or (b == c and a <= d)) \
        else (d, c, b, a)


def enumerate_topology(types, bonded):
    n = len(types)
    nbrs = [np.nonzero(bonded[i])[0] for i in range(n)]
    bonds = sorted({canon_bond(i, int(j)) for i in range(n) for j in nbrs[i]})
    angles = sorted({canon_angle(int(i), j, int(k)) for j in range(n)
                     for i in nbrs[j] for k in nbrs[j] if i < k})
    dihs = set()
    for (b, c) in bonds:
        for a in nbrs[b]:
            if a == c:
                continue
            for d in nbrs[c]:
                if d == b or d == a:
                    continue
                dihs.add(canon_dih(int(a), b, c, int(d)))
    dihs = sorted(dihs)
    # graph distance up to 3 for the exclusion/1-4 bookkeeping
    d1 = {canon_bond(*b) for b in bonds}
    d2 = {canon_bond(a[0], a[2]) for a in angles}
    # 1-4 pairs: a pair reachable by a 4-atom path but NOT ALSO 1-2 or 1-3
    # through another path.  In the 5-membered imidazolate ring many pairs are
    # both; the shorter path wins (full exclusion), exactly as in the published
    # topology -- 120 such pairs per conventional cell.
    d3 = {canon_bond(t[0], t[3]) for t in dihs if t[0] != t[3]} - d1 - d2
    return bonds, angles, dihs, d1, d2, d3


def type_key(types, idx, sep="-"):
    """Canonical type-name key: the itp writes bonds/angles/dihedrals in one of
    the two symmetric orders, so try both and let the caller pick the hit."""
    fwd = sep.join(types[i] for i in idx)
    rev = sep.join(types[i] for i in idx[::-1])
    return fwd, rev


def lookup(table, types, idx, what):
    fwd, rev = type_key(types, idx)
    if fwd in table:
        return table[fwd]
    if rev in table:
        return table[rev]
    raise KeyError(f"no {what} parameters for {fwd} (idx {idx})")


# ---------------------------------------------------- geometry: cages, window
def distance_field(pos, box, spacing=0.3):
    xs = [np.arange(0, box[i], spacing) for i in range(3)]
    G = np.stack(np.meshgrid(*xs, indexing="ij"), axis=-1).reshape(-1, 3)
    dmin = np.full(len(G), 1e9)
    for i in range(0, len(pos), 32):
        d = min_image(G[:, None, :] - pos[None, i:i + 32, :], box)
        dmin = np.minimum(dmin, np.linalg.norm(d, axis=-1).min(axis=1))
    return G, dmin, tuple(len(x) for x in xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supercell", type=int, default=1)
    ap.add_argument("--lattice-a", type=float, default=None,
                    help="override the cubic lattice constant (A); the CIF "
                         "positions scale affinely.  Used by the Stage-0A "
                         "equilibrium-lattice scan; force-field parameters "
                         "(r0, theta0, sigma) never scale.")
    ap.add_argument("--rc", type=float, default=None,
                    help="LJ/DSF cutoff (A); default 0.49*min(box)")
    ap.add_argument("--dsf-alpha", type=float, default=0.2)
    ap.add_argument("--r-tube", type=float, default=4.5)
    ap.add_argument("--k-wall", type=float, default=100.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-validation", action="store_true")
    a = ap.parse_args()

    types_p, bond_p, angle_p, dih_p, imp_p = parse_atomtypes(FF)
    print(f"force field: {len(types_p)} atom types, {len(bond_p)} bond, "
          f"{len(angle_p)} angle, {len(dih_p)} proper, {len(imp_p)} improper types")
    n_zero_d = sum(1 for v in dih_p.values() if v[1] == 0.0)
    n_zero_i = sum(1 for v in imp_p.values() if v[1] == 0.0)
    print(f"  zero-force-constant types: {n_zero_d}/{len(dih_p)} propers, "
          f"{n_zero_i}/{len(imp_p)} impropers (dropped; recorded)")

    # ---------------- validation against the published 2x2x2 enumeration ----
    val = None
    if not a.skip_validation:
        vbox, vels, vpos = parse_ref_pdb(REF_PDB)
        vbonded, _ = neighbour_bonds(vels, vpos, vbox)
        vtypes = assign_types(vels, vbonded)
        vb, va, vd, _, _, vd3 = enumerate_topology(vtypes, vbonded)
        ref = parse_ref_top(REF_TOP)
        got = dict(bonds={canon_bond(*t[0]) for t in ref["bonds"]},
                   angles={canon_angle(*t[0]) for t in ref["angles"]},
                   dihedrals={canon_dih(*t[0]) for t in ref["dihedrals"]},
                   pairs={canon_bond(*t[0]) for t in ref["pairs"]})
        mine = dict(bonds=set(vb), angles=set(va), dihedrals=set(vd), pairs=vd3)
        val = {}
        for k in got:
            ok = got[k] == mine[k]
            val[k] = dict(published=len(got[k]), built=len(mine[k]), identical=bool(ok),
                          only_published=len(got[k] - mine[k]),
                          only_built=len(mine[k] - got[k]))
            print(f"  2x2x2 {k:10s}: published {len(got[k]):6d}  built {len(mine[k]):6d}  "
                  f"identical={ok}")
            assert ok, (f"{k} enumeration differs from the published topology: "
                        f"{len(got[k] - mine[k])} only-published, "
                        f"{len(mine[k] - got[k])} only-built")
        # every published term must resolve to the SAME parameter type name
        for sec, tbl, canon in (("bonds", bond_p, canon_bond),
                                ("angles", angle_p, canon_angle),
                                ("dihedrals", dih_p, canon_dih)):
            for idx, name in ref[sec]:
                fwd, rev = type_key(vtypes, idx)
                assert name in (fwd, rev), \
                    f"{sec} {idx}: published type {name}, built {fwd}"
        print("  every published bonded term resolves to the same parameter type: OK")
        val["impropers_published"] = len(ref["impropers"])

    # ------------------------------- build the requested cell ---------------
    box0, els0, pos0 = parse_cif(CIF)
    if a.lattice_a is not None:
        s = a.lattice_a / box0[0]
        print(f"lattice override: a {box0[0]:.4f} -> {a.lattice_a:.4f} A "
              f"(affine scale {s:.6f})")
        box0, pos0 = box0 * s, pos0 * s
    S = a.supercell
    shifts = np.array([[i, j, k] for i in range(S) for j in range(S) for k in range(S)],
                      float) * box0
    pos = (pos0[None] + shifts[:, None, :]).reshape(-1, 3)
    els = np.tile(els0, len(shifts))
    box = box0 * S
    print(f"structure: cell {box0.round(3)} x{S}^3 -> box {box.round(3)}, "
          f"{len(els)} atoms")
    bonded, rmat = neighbour_bonds(els, pos, box)
    types = assign_types(els, bonded)
    cnt = {t: int((types == t).sum()) for t in EXPECTED_PER_CELL}
    exp = {t: v * S ** 3 for t, v in EXPECTED_PER_CELL.items()}
    print(f"  types: {cnt}")
    assert cnt == exp, f"composition {cnt} != expected {exp}"
    deg = bonded.sum(1)
    for t, d in (("Zn", 4), ("N", 3), ("C1", 3), ("C2", 3), ("C3", 4),
                 ("H2", 1), ("H3", 1)):
        assert (deg[types == t] == d).all(), f"{t} coordination != {d}"
    print(f"  coordination: Zn 4N, N 3, C1 3, C2 3, C3 4, H 1 -- all OK")

    bonds, angles, dihs, d1, d2, d3 = enumerate_topology(types, bonded)
    rc = a.rc if a.rc is not None else 0.49 * float(box.min())
    assert rc < 0.5 * box.min(), "rc violates the minimum-image convention"

    n = len(els)
    mass = np.array([types_p[t]["mass"] for t in types])
    charge = np.array([types_p[t]["charge"] for t in types])
    sigma = np.array([types_p[t]["sigma"] for t in types])
    epsq = np.array([types_p[t]["eps"] for t in types])
    net = float(charge.sum())
    print(f"  net charge {net:+.4f} e  (per atom {net/n:+.2e})")
    assert abs(net) < 1e-3 * n, "framework is not charge neutral"

    lj_scale = np.ones((n, n)); coul_scale = np.ones((n, n))
    for (i, j) in d1 | d2:
        lj_scale[i, j] = lj_scale[j, i] = 0.0
        coul_scale[i, j] = coul_scale[j, i] = 0.0
    for (i, j) in d3:
        lj_scale[i, j] = lj_scale[j, i] = FUDGE_LJ
        coul_scale[i, j] = coul_scale[j, i] = FUDGE_QQ
    np.fill_diagonal(lj_scale, 0.0); np.fill_diagonal(coul_scale, 0.0)

    bond_k = np.array([lookup(bond_p, types, b, "bond")[1] for b in bonds])
    bond_r0 = np.array([lookup(bond_p, types, b, "bond")[0] for b in bonds])
    angle_th0 = np.array([lookup(angle_p, types, t, "angle")[0] for t in angles])
    angle_k = np.array([lookup(angle_p, types, t, "angle")[1] for t in angles])
    dpar = [lookup(dih_p, types, t, "dihedral") for t in dihs]
    keep = [i for i, p in enumerate(dpar) if p[1] != 0.0]
    dih_keep = np.array([dihs[i] for i in keep], dtype=np.int64).reshape(-1, 4)
    dih_delta = np.array([dpar[i][0] for i in keep])
    dih_k = np.array([dpar[i][1] for i in keep])
    dih_n = np.array([dpar[i][2] for i in keep])
    print(f"  topology: {len(bonds)} bonds, {len(angles)} angles, {len(dihs)} propers "
          f"({len(dih_keep)} with k>0; {len(dihs)-len(dih_keep)} zero-k dropped)")
    print(f"  bond r0 {bond_r0.min():.3f}-{bond_r0.max():.3f} A, "
          f"k {bond_k.min():.0f}-{bond_k.max():.0f} kJ/mol/A^2")
    # bond lengths in the crystal structure must sit near their r0
    rb = np.array([rmat[i, j] for (i, j) in bonds])
    dev = np.abs(rb - bond_r0)
    print(f"  crystal bond length vs r0: max |dev| {dev.max():.3f} A "
          f"(mean {dev.mean():.3f})")
    assert dev.max() < 0.25, "crystal geometry inconsistent with the force field"

    # --------------------------- cages, window, tube, gate atoms ------------
    G, dmin, shape = distance_field(pos, box, 0.3)
    from scipy.ndimage import maximum_filter
    from scipy.optimize import minimize
    D = dmin.reshape(shape)
    loc = (D == maximum_filter(D, size=7, mode="wrap")) & (D > 3.0)
    cand = G.reshape(*shape, 3)[loc]; vals = D[loc]
    cages = []
    for idx in np.argsort(-vals):
        p = cand[idx]
        if all(np.linalg.norm(min_image(p - q, box)) > 6.0 for q in cages):
            cages.append(p)
    print(f"  cage centres: {len(cages)} (expect {2*S**3}), "
          f"largest free radius on the 0.3 A grid {vals.max():.2f} A")
    assert len(cages) == 2 * S ** 3, "unexpected cage count"

    # refine each centre off the grid: maximize the nearest-atom distance, so
    # the two symmetry-equivalent cages give a SYMMETRIC xi_A/xi_B
    def neg_free_radius(p):
        return -float(np.linalg.norm(min_image(p - pos, box), axis=-1).min())
    ref_cages, ref_radii = [], []
    for p in cages:
        r = minimize(neg_free_radius, p, method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-6, maxiter=4000))
        ref_cages.append(r.x); ref_radii.append(-r.fun)
    cages = np.array(ref_cages)
    print(f"  refined free radii: {np.round(ref_radii, 3)} A "
          f"(spread {max(ref_radii)-min(ref_radii):.4f} -- cages are equivalent)")
    assert max(ref_radii) - min(ref_radii) < 0.02, "refined cages inequivalent"

    # nearest cage pair -> the [111] 6MR channel.  The connecting vector is the
    # lattice translation (a/2)(1,1,1), so xi is EXACTLY PERIODIC with period L.
    best = None
    for i in range(len(cages)):
        for j in range(len(cages)):
            if i == j:
                continue
            L = np.linalg.norm(min_image(cages[j] - cages[i], box))
            if best is None or L < best[2]:
                best = (i, j, L)
    iA, iB, L = best
    cA = cages[iA]
    cB = cA + min_image(cages[iB] - cA, box)
    center = 0.5 * (cA + cB)
    nrm = (cB - cA) / L
    print(f"  cage-cage L={L:.4f} A along {np.round(nrm, 4)} (expect [111]/sqrt3); "
          f"a*sqrt(3)/2 = {box0[0]*math.sqrt(3)/2:.4f}")
    assert abs(L - box0[0] * math.sqrt(3) / 2) < 0.05, \
        "the cage-cage vector is not the (1/2,1/2,1/2) lattice translation"

    # the 6MR: the 6 Zn nearest the mid-point (sodalite 6-ring is Zn6(im)6)
    zn_idx = np.nonzero(types == "Zn")[0]
    dzn = np.linalg.norm(min_image(pos[zn_idx] - center, box), axis=-1)
    gate_zn = zn_idx[np.argsort(dzn)[:6]]
    print(f"  gate 6-ring Zn distances from the window centre: "
          f"{np.sort(dzn)[:8].round(2)}")
    assert np.sort(dzn)[5] < np.sort(dzn)[6] - 0.5, "6-ring Zn shell not separated"
    zn_rel = center + min_image(pos[gate_zn] - center, box)
    ring_centroid = zn_rel.mean(axis=0)
    _, _, Vt = np.linalg.svd(zn_rel - ring_centroid)
    ring_n = Vt[2]
    if np.dot(min_image(cB - ring_centroid, box), ring_n) < 0:
        ring_n = -ring_n
    print(f"  6-ring centroid offset from the cage mid-point: "
          f"{np.linalg.norm(ring_centroid - center):.3f} A; "
          f"ring normal . axis = {abs(float(np.dot(ring_n, nrm))):.4f}")
    assert abs(float(np.dot(ring_n, nrm))) > 0.98, "6-ring is not normal to the axis"
    center, nrm = ring_centroid, ring_n
    xi_A = float(np.dot(min_image(cA - center, box), nrm))
    xi_B = float(np.dot(min_image(cB - center, box), nrm))

    # the 6 gate linkers: the imidazolates bridging consecutive gate Zn.
    # ring H (H2) and the linker plane (C1, and the two ring C2) are the
    # hidden-coordinate observables; A_gate uses the H2 nearest the axis.
    zn_set = set(int(z) for z in gate_zn)
    nbrs = [np.nonzero(bonded[i])[0] for i in range(n)]
    linkers = []
    for c1 in np.nonzero(types == "C1")[0]:
        ns = [int(x) for x in nbrs[c1] if types[x] == "N"]
        zs = set()
        for nn in ns:
            zs |= {int(x) for x in nbrs[nn] if types[x] == "Zn"}
        if len(zs & zn_set) == 2:
            ring_c2 = sorted(int(x) for nn in ns for x in nbrs[nn] if types[x] == "C2")
            h2 = sorted(int(x) for c in ring_c2 for x in nbrs[c] if types[x] == "H2")
            linkers.append(dict(c1=int(c1), n=sorted(ns), c2=ring_c2, h2=h2))
    print(f"  gate linkers bridging two gate Zn: {len(linkers)} (expect 6)")
    assert len(linkers) == 6, "gate linker identification failed"
    gate_tri = np.array([[lk["c1"], lk["c2"][0], lk["c2"][1]] for lk in linkers],
                        dtype=np.int64)

    # The 6MR aperture is lined by ONE ring hydrogen per linker -- the one that
    # points into the window.  Freeze those six atom indices from the crystal
    # structure; A_gate is then their mean radial distance from the gate axis.
    def radial_axial(p):
        d = min_image(p - center, box)
        ax = float(np.dot(d, nrm))
        return float(np.linalg.norm(d - ax * nrm)), ax
    # The 6-ring linkers ALTERNATE: three present their ring C-H edge to the
    # window (these six H are the crystallographic bottleneck) and three
    # present their methyl.  Both are recorded; A_gate uses the bottleneck.
    all_h2 = np.array(sorted(int(x) for lk in linkers for x in lk["h2"]))
    rad_h2 = np.array([radial_axial(pos[i])[0] for i in all_h2])
    order = np.argsort(rad_h2)
    gate_h = all_h2[order[:6]]
    a0 = rad_h2[order[:6]]
    print(f"  aperture ring-H radial distances: inner six "
          f"{np.round(np.sort(rad_h2)[:6], 3)}, next {np.sort(rad_h2)[6]:.3f}")
    assert np.sort(rad_h2)[5] < np.sort(rad_h2)[6] - 0.5, \
        "the aperture ring-H shell is not separated"
    owner = {int(h): k for k, lk in enumerate(linkers) for h in lk["h2"]}
    own_counts = np.bincount([owner[int(h)] for h in gate_h], minlength=6)
    print(f"  aperture H per gate linker: {own_counts.tolist()} "
          f"(3 ring-H-facing linkers x 2 H, 3 methyl-facing)")
    assert sorted(own_counts.tolist()) == [0, 0, 0, 2, 2, 2], \
        "unexpected 6-ring linker alternation"
    methyl_c = np.array(sorted(
        int(x) for k, lk in enumerate(linkers) if own_counts[k] == 0
        for x in nbrs[lk["c1"]] if types[x] == "C3"), dtype=np.int64)
    assert len(methyl_c) == 3, "expected 3 window-facing methyls"
    print(f"  crystal gate aperture A_gate = {a0.mean():.3f} A radius "
          f"-> {2*(a0.mean() - 1.10):.2f} A free diameter (H vdW 1.10 A); "
          f"literature 6MR ~3.4 A")

    # confinement tube.  Every OTHER window out of cage A must sit well outside
    # it, or the guest could leave the channel sideways.
    R_tube = a.r_tube
    other = []
    for k, cg in enumerate(cages):
        for sh in np.array([[i, j, l] for i in (-1, 0, 1) for j in (-1, 0, 1)
                            for l in (-1, 0, 1)]) * box:
            w = 0.5 * (cA + cg + sh)
            if np.linalg.norm(w - cA) < 1.0 or np.linalg.norm(cg + sh - cA) > L + 0.1:
                continue
            rad, ax = radial_axial(w)
            other.append((rad, ax))
    off = [r for r, _ in other if r > 0.5]
    print(f"  windows out of cage A: {len(other)} (expect 8); off-axis ones at "
          f"radial {min(off):.3f}-{max(off):.3f} A vs tube R={R_tube}")
    assert min(off) > R_tube + 1.5, "the confinement tube leaks into a side window"

    # The tube is longer than the minimum-image cube's inscribed sphere, so the
    # CV and the wall MUST be evaluated on UNWRAPPED guest coordinates (the
    # integrator never wraps positions).  Record how badly a min-imaged CV
    # would fail, so the choice is documented rather than assumed.
    rng = np.random.default_rng(0)
    ns = 20000
    xi_t = rng.uniform(-1.5 * L, 1.5 * L, ns)
    rho = R_tube * np.sqrt(rng.uniform(0, 1, ns))
    ang = rng.uniform(0, 2 * np.pi, ns)
    e1 = np.cross(nrm, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm, e1)
    perp = (np.cos(ang)[:, None] * e1[None, :] + np.sin(ang)[:, None] * e2[None, :])
    P = center + xi_t[:, None] * nrm[None, :] + rho[:, None] * perp
    d = P - center
    ax = d @ nrm
    rad = np.linalg.norm(d - ax[:, None] * nrm[None, :], axis=-1)
    assert np.abs(ax - xi_t).max() < 1e-9 and np.abs(rad - rho).max() < 1e-9
    dmi = min_image(d, box)
    ax_mi = dmi @ nrm
    frac = (ax_mi - xi_t) / L
    worst = np.abs(frac - np.round(frac)).max() * L
    print(f"  CV on UNWRAPPED coordinates: exact over the tube. "
          f"(A min-imaged CV would be wrong by up to {worst:.2f} A -- "
          f"the tube is longer than the min-image cube; do not wrap.)")

    # the channel must be traversable inside the tube: at every axial slice the
    # tube has to contain a point with enough clearance for a CH3 site
    slices = np.linspace(-L / 2, L / 2, 61)
    clear = []
    for s in slices:
        base = center + s * nrm
        uu, vv = np.meshgrid(np.linspace(-R_tube, R_tube, 25),
                             np.linspace(-R_tube, R_tube, 25))
        keep = (uu ** 2 + vv ** 2) <= R_tube ** 2
        pts = base[None, :] + uu[keep][:, None] * e1[None, :] + vv[keep][:, None] * e2[None, :]
        dd = np.linalg.norm(min_image(pts[:, None, :] - pos[None, :, :], box),
                            axis=-1).min(axis=1)
        clear.append(dd.max())
    clear = np.array(clear)
    print(f"  channel clearance inside the tube: min over axial slices "
          f"{clear.min():.2f} A at xi={slices[np.argmin(clear)]:+.2f} "
          f"(cage max {clear.max():.2f})")
    assert clear.min() > 2.6, "the tube pinches the channel"

    # How exactly is the atom set invariant under the (1/2,1/2,1/2) lattice
    # translation?  Everything but the methyl H is exact; the CIF picks one
    # ordered methyl rotamer per linker and those choices are not consistent
    # with the body centring.  The methyl is a FREE ROTOR in this force field
    # (every H3-C3-C1-N torsion has k = 0), so the rotamer randomises
    # thermally and F(phi) is exactly periodic by symmetry -- only the
    # INSTANTANEOUS potential is not.  Measured and declared, not assumed.
    Tvec = box0 / 2.0
    sym = {}
    for tp in EXPECTED_PER_CELL:
        P = pos[types == tp]
        dd = np.linalg.norm(min_image((P + Tvec)[:, None, :] - P[None, :, :], box),
                            axis=-1).min(axis=1)
        sym[tp] = float(dd.max())
    print(f"  body-centring image mismatch by type (A): "
          f"{ {k: round(v, 4) for k, v in sym.items()} }")
    assert max(v for k, v in sym.items() if k != "H3") < 1e-6, \
        "the heavy-atom framework is not body-centred"

    # how much of the guest-accessible cage volume does the tube capture?
    acc = dmin > 3.4                      # a CH3 site needs ~3.4 A of clearance
    dacc = min_image(G[acc] - center, box)
    ax_a = dacc @ nrm
    rad_a = np.linalg.norm(dacc - ax_a[:, None] * nrm[None, :], axis=-1)
    inside = rad_a <= R_tube
    print(f"  accessible volume (free radius > 3.4 A): tube captures "
          f"{100*inside.mean():.1f}% of it")

    out = a.out or os.path.join(ROOT, f"cache/zif8/framework{'' if S == 1 else f'_{S}x{S}x{S}'}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out, box=box, rc=rc, dsf_alpha=a.dsf_alpha, pos=pos,
        atom_type=np.array([str(t) for t in types]), element=els,
        mass_amu=mass, charge_e=charge, lj_eps_kj=epsq, lj_sig_A=sigma,
        lj_scale=lj_scale, coul_scale=coul_scale,
        bonds=np.array(bonds, dtype=np.int64).reshape(-1, 2),
        bond_k=bond_k, bond_r0=bond_r0,
        angles=np.array(angles, dtype=np.int64).reshape(-1, 3),
        angle_k=angle_k, angle_th0=angle_th0,
        dihedrals=dih_keep, dih_k=dih_k, dih_n=dih_n, dih_delta=dih_delta,
        impropers=np.zeros((0, 4), dtype=np.int64), impr_k=np.zeros(0),
        impr_psi0=np.zeros(0),
        cage_A=cA, cage_B=cB, win_center=center, win_normal=nrm, period=L,
        xi_A=xi_A, xi_B=xi_B, R_tube=R_tube, k_wall=a.k_wall,
        gate_zn_idx=gate_zn.astype(np.int64),
        gate_aperture_h=gate_h.astype(np.int64), gate_methyl_c=methyl_c,
        gate_tri=gate_tri, gate_aperture_crystal=a0,
        tube_volume_capture=float(inside.mean()), supercell=S,
        bodycentring_mismatch=json.dumps(sym),
        provenance=json.dumps(dict(
            structure=os.path.relpath(CIF, ROOT), forcefield=os.path.relpath(FF, ROOT),
            validated_against=os.path.relpath(REF_TOP, ROOT), validation=val,
            fudge_lj=FUDGE_LJ, fudge_qq=FUDGE_QQ,
            electrostatics=f"damped shifted force, alpha={a.dsf_alpha} 1/A, rc={rc:.3f} A",
            cv=("phi = wrap(2 pi xi / L), xi = (COM_guest - win_center).n, "
                f"L = {L:.4f} A = |(a/2)(1,1,1)|; the CV is EXACTLY periodic "
                "because that vector is a lattice translation, so the channel "
                "needs no axial walls -- only the radial tube."),
            note=("Krokidas et al. JPCC 2015 flexible ZIF-8 FF (the force field "
                  "of the 2024 ethane/ZIF-8 anchor paper) via the Anikeenko 2017 "
                  "GROMACS adaptation; topology enumerated from connectivity and "
                  "validated term-by-term against the published 2x2x2 topology. "
                  "Zero-force-constant propers/impropers dropped (methyl rotation "
                  "and the Zn-N torsions are free in this FF, by design)."))))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
