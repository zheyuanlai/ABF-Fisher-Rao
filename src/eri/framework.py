"""ERI framework builder (docs/eri_campaign/MODEL_REPORT.md).

Reads a CIF with a WHITELIST parser (the symmetry operators are parsed into integer/rational matrices; nothing in
the file is ever evaluated), expands the asymmetric unit with the 24 operators of P6_3/mmc, keeps the majority
framework sites (T and O positions whose summed occupancy at one position is >= 0.5; cations, water and the ~5 %
offretite-fault sites are dropped; every T is Si), and builds the orthorhombic supercell

    X along the hexagonal a1 axis, Y perpendicular to it in the ab-plane, Z along c,
    box = (3a, 2 sqrt(3) a, 3c)        (Schlenker 1977: 39.756 x 45.906 x 44.43 A = the PNAS 2005 box, axes relabelled)

so that the minimum image is valid for the 12 A cutoff (every width > 24 A).

Two variants (LIT_CROSSCHECK.md Sec. 8):
  * ``gard_tait``  -- RASPA2 ``ERI_SI.cif`` (= archived IZA record 55, Gard & Tait 1973): a 13.27, c 15.05 A, the cited
                      structure;
  * ``schlenker``  -- COD 2106432 (Schlenker, Pluth & Smith 1977): a 13.252, c 14.81 A, the cell the PNAS box fits.

Cartesian frame: a1 = (a, 0, 0), a2 = (-a/2, a sqrt(3)/2, 0), a3 = (0, 0, c).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from fractions import Fraction

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LIT = os.path.join(ROOT, "cache", "eri", "literature", "structure")
SOURCES = {
    "gard_tait": os.path.join(LIT, "raspa2_ERI_SI_cif", "ERI_SI.cif"),
    "schlenker": os.path.join(LIT, "cod_2106432", "2106432.cif"),
}
DEDUP_TOL = 2.0e-3            # fractional; 1e-4 silently duplicates the Gard & Tait O5 orbit (LIT_CROSSCHECK Sec. 5)
O_TYPE, SI_TYPE = 0, 1

_TERM = re.compile(r"([+-]?)(\d+/\d+|\d*\.\d+|\d+|x|y|z)")


def parse_symop(s):
    """'x-y,x,z+1/2' -> (R int 3x3, t Fraction 3).  Raises on anything that is not a signed sum of x, y, z and
    rational constants."""
    parts = s.replace(" ", "").replace("'", "").lower().split(",")
    if len(parts) != 3:
        raise ValueError(f"bad symop {s!r}")
    R = np.zeros((3, 3), dtype=np.int64)
    t = [Fraction(0)] * 3
    for i, p in enumerate(parts):
        pos = 0
        if not p:
            raise ValueError(f"bad symop {s!r}")
        while pos < len(p):
            m = _TERM.match(p, pos)
            if m is None or m.end() == pos:
                raise ValueError(f"bad symop term in {s!r} at {p[pos:]!r}")
            sign = -1 if m.group(1) == "-" else 1
            tok = m.group(2)
            if tok in ("x", "y", "z"):
                R[i, "xyz".index(tok)] += sign
            else:
                t[i] += sign * Fraction(tok)
            pos = m.end()
    return R, np.array([float(v) for v in t])


def read_cif(path):
    """Minimal whitelist CIF reader: cell, symmetry operators, atom sites (label, type, x, y, z, occupancy)."""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    cell = {}
    for ln in lines:
        for key in ("_cell_length_a", "_cell_length_b", "_cell_length_c", "_cell_angle_gamma"):
            if ln.strip().startswith(key + " ") or ln.strip().startswith(key + "\t"):
                cell[key] = float(re.sub(r"\(.*\)", "", ln.split()[1]))
    ops, sites = [], []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            heads = []
            while j < len(lines) and lines[j].strip().startswith("_"):
                heads.append(lines[j].strip().split()[0])
                j += 1
            rows = []
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(("_", "loop_", "#")):
                rows.append(lines[j].strip())
                j += 1
            if any(h in ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz") for h in heads):
                col = [k for k, h in enumerate(heads)
                       if h in ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz")][0]
                for r in rows:
                    tok = re.findall(r"'[^']*'|\S+", r)
                    ops.append(parse_symop(tok[col] if len(heads) > 1 else r))
            if "_atom_site_fract_x" in heads:
                ix = {h: k for k, h in enumerate(heads)}
                for r in rows:
                    tok = r.split()
                    if len(tok) < len(heads):
                        continue
                    occ = float(re.sub(r"\(.*\)", "", tok[ix["_atom_site_occupancy"]])) \
                        if "_atom_site_occupancy" in ix else 1.0
                    sites.append((tok[ix["_atom_site_label"]], tok[ix["_atom_site_type_symbol"]],
                                  *[float(re.sub(r"\(.*\)", "", tok[ix[k]]))
                                    for k in ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")],
                                  occ))
            i = j
        else:
            i += 1
    if abs(cell["_cell_angle_gamma"] - 120.0) > 1e-6 or abs(cell["_cell_length_a"] - cell["_cell_length_b"]) > 1e-9:
        raise ValueError("expected a hexagonal cell")
    if len(ops) != 24:
        raise ValueError(f"expected the 24 operators of P6_3/mmc, got {len(ops)}")
    return cell["_cell_length_a"], cell["_cell_length_c"], ops, sites


def read_cif_general(path):
    """Like read_cif but for any cell: returns ((a, b, c, alpha, beta, gamma), ops, sites).  Whitelist parser only."""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    keys = ("_cell_length_a", "_cell_length_b", "_cell_length_c", "_cell_angle_alpha", "_cell_angle_beta",
            "_cell_angle_gamma")
    cell = {}
    for ln in lines:
        tok = ln.strip().split()
        if tok and tok[0] in keys and len(tok) > 1:
            cell[tok[0]] = float(re.sub(r"\(.*\)", "", tok[1]))
    ops, sites = [], []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            heads = []
            while j < len(lines) and lines[j].strip().startswith("_"):
                heads.append(lines[j].strip().split()[0])
                j += 1
            rows = []
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(("_", "loop_", "#")):
                rows.append(lines[j].strip())
                j += 1
            symk = [k for k, h in enumerate(heads) if h in ("_symmetry_equiv_pos_as_xyz",
                                                             "_space_group_symop_operation_xyz")]
            if symk:
                for r in rows:
                    tok = re.findall(r"'[^']*'|\S+", r)
                    ops.append(parse_symop(tok[symk[0]] if len(heads) > 1 else r))
            if "_atom_site_fract_x" in heads:
                ix = {h: k for k, h in enumerate(heads)}
                for r in rows:
                    tok = r.split()
                    if len(tok) < len(heads):
                        continue
                    occ = float(re.sub(r"\(.*\)", "", tok[ix["_atom_site_occupancy"]])) \
                        if "_atom_site_occupancy" in ix else 1.0
                    typ = tok[ix["_atom_site_type_symbol"]] if "_atom_site_type_symbol" in ix else tok[0]
                    sites.append((tok[ix["_atom_site_label"]], typ,
                                  *[float(re.sub(r"\(.*\)", "", tok[ix[k]]))
                                    for k in ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")],
                                  occ))
            i = j
        else:
            i += 1
    return tuple(cell[k] for k in keys), ops, sites


def build_orthorhombic(path, reps, perm=(0, 1, 2), name="framework", n_t=None, n_o=None):
    """Orthorhombic supercell of an orthorhombic CIF.  ``perm`` maps (X, Y, Z) of the box to crystal axes (the CV
    axis is X).  Returns a dict like ``build`` with ``cell`` (the unit cell in box axes)."""
    (a, b, c, al, be, ga), ops, sites = read_cif_general(path)
    if max(abs(al - 90), abs(be - 90), abs(ga - 90)) > 1e-6:
        raise ValueError("not an orthorhombic cell")
    asym = majority_framework_sites(sites)
    fpos, ftyp = expand_cell(asym, ops)
    if n_t is not None and ((ftyp == SI_TYPE).sum() != n_t or (ftyp == O_TYPE).sum() != n_o):
        raise ValueError(f"expected {n_t} T + {n_o} O per cell, got {(ftyp == SI_TYPE).sum()} + {(ftyp == O_TYPE).sum()}")
    # an odd permutation of the axes is an improper map (it would build the MIRROR image of the framework)
    sign = np.linalg.det(np.eye(3)[list(perm)])
    if sign < 0:
        raise ValueError(f"perm {perm} is odd (improper): use an even permutation such as (1, 2, 0)")
    L = np.array([a, b, c])
    cellv = L[list(perm)]
    repsv = np.asarray(reps, dtype=np.int64)
    box = cellv * repsv
    cart = []
    types = []
    for i in range(repsv[0]):
        for j in range(repsv[1]):
            for k in range(repsv[2]):
                xyz = (fpos * L)[:, list(perm)] + cellv * np.array([i, j, k])
                cart.append(xyz)
                types.append(ftyp)
    cart = np.concatenate(cart) % box
    types = np.concatenate(types)
    ncell = int(np.prod(repsv))
    # duplicate check across orbits (audit F5): no two atoms closer than 1.0 A (min image)
    d = cart[:, None, :] - cart[None, :, :]
    d -= box * np.round(d / box)
    rr = np.sqrt((d * d).sum(-1))
    rr[np.diag_indices_from(rr)] = np.inf
    if rr.min() < 1.0:
        raise ValueError(f"duplicate / overlapping framework atoms (min distance {rr.min():.3f} A)")
    return {"variant": name, "source": os.path.relpath(path, ROOT), "source_sha256": sha256_file(path),
            "a": a, "b": b, "c": c, "box": box, "pos": cart, "types": types, "reps": repsv, "asym": asym,
            "n_cells": ncell, "cell": cellv, "perm": np.array(perm)}


def majority_framework_sites(sites):
    """Group T (Si/Al) and O sites by position; keep positions whose summed occupancy >= 0.5.  Returns a list of
    (element 'Si'|'O', x, y, z)."""
    out = []
    groups = {}
    for lab, typ, x, y, z, occ in sites:
        el = re.match(r"[A-Za-z]+", typ).group(0)
        if el in ("Si", "Al"):
            kind = "Si"
        elif el == "O" and not lab.lower().startswith(("ow", "w")):
            kind = "O"
        else:
            continue                                   # cations, water
        key = (kind, round(x, 4), round(y, 4), round(z, 4))
        groups[key] = groups.get(key, 0.0) + occ
    for (kind, x, y, z), occ in sorted(groups.items()):
        if occ >= 0.5:
            out.append((kind, x, y, z))
    return out


def expand_cell(asym, ops):
    """Fractional positions of the full hexagonal cell, de-duplicated with DEDUP_TOL."""
    pos, typ = [], []
    for kind, x, y, z in asym:
        v = np.array([x, y, z])
        orbit = []
        for R, t in ops:
            w = (R @ v + t) % 1.0
            dup = False
            for u in orbit:
                d = (w - u + 0.5) % 1.0 - 0.5
                if np.max(np.abs(d)) < DEDUP_TOL:
                    dup = True
                    break
            if not dup:
                orbit.append(w)
        pos.extend(orbit)
        typ.extend([O_TYPE if kind == "O" else SI_TYPE] * len(orbit))
    return np.array(pos), np.array(typ, dtype=np.int64)


def hex_to_cart(s, a, c):
    s = np.atleast_2d(s)
    return np.stack([a * s[:, 0] - 0.5 * a * s[:, 1], 0.5 * math.sqrt(3.0) * a * s[:, 1], c * s[:, 2]], axis=1)


def build(variant, reps=(3, 2, 3)):
    """Orthorhombic supercell (reps[0] a) x (reps[1] sqrt(3) a) x (reps[2] c).  Returns a dict."""
    path = SOURCES[variant]
    a, c, ops, sites = read_cif(path)
    asym = majority_framework_sites(sites)
    fpos, ftyp = expand_cell(asym, ops)
    if (ftyp == SI_TYPE).sum() != 36 or (ftyp == O_TYPE).sum() != 72:
        raise ValueError(f"{variant}: expected 36 Si + 72 O per cell, got {(ftyp == SI_TYPE).sum()} + "
                         f"{(ftyp == O_TYPE).sum()}")
    box = np.array([reps[0] * a, reps[1] * math.sqrt(3.0) * a, reps[2] * c])
    cart = []
    types = []
    nx, ny = reps[0] + reps[1] + 2, 2 * reps[1] + 2
    for i in range(-nx, nx + 1):
        for j in range(-1, ny + 1):
            for k in range(reps[2]):
                xyz = hex_to_cart(fpos + np.array([i, j, k]), a, c)
                inside = ((xyz[:, 0] >= -1e-9) & (xyz[:, 0] < box[0] - 1e-9) & (xyz[:, 1] >= -1e-9)
                          & (xyz[:, 1] < box[1] - 1e-9) & (xyz[:, 2] >= -1e-9) & (xyz[:, 2] < box[2] - 1e-9))
                cart.append(xyz[inside])
                types.append(ftyp[inside])
    cart = np.concatenate(cart)
    types = np.concatenate(types)
    cart = cart % box
    # de-duplicate (atoms on the box faces)
    order = np.lexsort((cart[:, 2], cart[:, 1], cart[:, 0]))
    cart, types = cart[order], types[order]
    keep = np.ones(len(cart), bool)
    for n in range(len(cart)):
        if not keep[n]:
            continue
        d = cart[n + 1:] - cart[n]
        d -= box * np.round(d / box)
        close = np.nonzero(np.sqrt((d * d).sum(1)) < 0.05)[0]
        keep[n + 1 + close] = False
    cart, types = cart[keep], types[keep]
    ncell = reps[0] * reps[1] * 2 * reps[2]
    if (types == SI_TYPE).sum() != 36 * ncell or (types == O_TYPE).sum() != 72 * ncell:
        raise ValueError(f"supercell count mismatch: {(types == SI_TYPE).sum()} Si, {(types == O_TYPE).sum()} O "
                         f"for {ncell} cells")
    return {"variant": variant, "source": os.path.relpath(path, ROOT), "source_sha256": sha256_file(path),
            "a": a, "c": c, "box": box, "pos": cart, "types": types, "reps": np.array(reps),
            "asym": asym, "n_cells": ncell}


def coordination_check(fw, cut=2.0):
    """Every Si has 4 O and every O has 2 Si within ``cut`` (minimum image).  Returns (ok, dmin, Si-O range)."""
    P, T, box = fw["pos"], fw["types"], fw["box"]
    si = P[T == SI_TYPE]
    o = P[T == O_TYPE]
    d = si[:, None, :] - o[None, :, :]
    d -= box * np.round(d / box)
    r = np.sqrt((d * d).sum(-1))
    nb = r < cut
    ok = bool((nb.sum(1) == 4).all() and (nb.sum(0) == 2).all())
    allp = P[:, None, :] - P[None, :, :]
    allp -= box * np.round(allp / box)
    rr = np.sqrt((allp * allp).sum(-1))
    rr[np.diag_indices_from(rr)] = np.inf
    return ok, float(rr.min()), (float(r[nb].min()), float(r[nb].max()))


def special_sites(fw):
    """Cartesian positions (in the box) of the ERI cage centres, 8-ring window centres and CAN / D6R centres."""
    a, c, box, reps = fw["a"], fw["c"], fw["box"], fw["reps"]
    frac = {
        "cage": [(1 / 3, 2 / 3, 3 / 4), (2 / 3, 1 / 3, 1 / 4)],
        "window": [(0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.5)],
        "can": [(0, 0, 0.25), (0, 0, 0.75)],
        "d6r": [(0, 0, 0), (0, 0, 0.5)],
    }
    out = {}
    nx, ny = reps[0] + reps[1] + 2, 2 * reps[1] + 2
    for name, lst in frac.items():
        pts = []
        for i in range(-nx, nx + 1):
            for j in range(-1, ny + 1):
                for k in range(reps[2]):
                    xyz = hex_to_cart(np.array(lst) + np.array([i, j, k]), a, c)
                    pts.append(xyz)
        pts = np.concatenate(pts) % box
        uniq = []
        for p in pts:
            if not any(np.max(np.abs(((p - q) + box / 2) % box - box / 2)) < 1e-6 for q in uniq):
                uniq.append(p)
        out[name] = np.array(uniq)
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def save_generic(fw, path, cages, windows, P, x_window):
    """Save a non-ERI framework with EXPLICIT cage centres and window centres (box coordinates), the CV period P and
    the X offset of the phi = 0 windows.  Refuses a framework that fails the coordination check (audit F5)."""
    ok, dmin, (lo, hi) = coordination_check(fw)
    if not ok:
        raise ValueError(f"coordination check failed (T-O range {lo:.3f}-{hi:.3f}, d_min {dmin:.3f})")
    np.savez(path, pos=fw["pos"], types=fw["types"], box=fw["box"], a=fw["a"], c=fw["c"], b=fw["b"], reps=fw["reps"],
             cell=fw["cell"], perm=fw["perm"], variant=fw["variant"], source=fw["source"],
             source_sha256=fw["source_sha256"], cages=np.asarray(cages, float), windows=np.asarray(windows, float),
             P=float(P), x_window=float(x_window))


def save(fw, path):
    s = special_sites(fw)
    np.savez(path, pos=fw["pos"], types=fw["types"], box=fw["box"], a=fw["a"], c=fw["c"], reps=fw["reps"],
             variant=fw["variant"], source=fw["source"], source_sha256=fw["source_sha256"],
             cages=s["cage"], windows=s["window"], can=s["can"], d6r=s["d6r"])


def load(variant):
    """The cached framework npz (built by scripts/eri_campaign/build_frameworks.py)."""
    z = np.load(os.path.join(ROOT, "cache", "eri", f"framework_{variant}.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}
