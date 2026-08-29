#!/usr/bin/env python
"""Build the rigid all-silica LTA framework from the IZA CIF (cache/lta/LTA.cif).

Expands the Pm-3m asymmetric unit (3 O + 1 Si, 48 ops) to the full unit cell,
validates the known LTA invariants (Si24O48 per cell, Si-O bond length, the
8-ring window at the face centre, the alpha-cage at the body centre), then
replicates to the requested supercell and writes cache/lta/framework.npz.

    python scripts/build_lta_framework.py --super 2
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CIF = os.path.join(ROOT, "cache/lta/LTA.cif")
OUT = os.path.join(ROOT, "cache/lta/framework.npz")


def parse_cif(path):
    txt = open(path).read()
    a = float(re.search(r"_cell_length_a\s+([\d.]+)", txt).group(1))
    ops = re.findall(r"'([+\-xyz,]+)'", txt)
    sites = []
    for m in re.finditer(r"^\s*(\w+)\s+(O|Si)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)",
                         txt, re.M):
        sites.append((m.group(2), np.array([float(m.group(3)), float(m.group(4)),
                                            float(m.group(5))])))
    return a, ops, sites


def apply_op(op, xyz):
    x, y, z = xyz
    env = {"x": x, "y": y, "z": z}
    return np.array([eval(c.replace("+", ""), {}, env) for c in op.split(",")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--super", type=int, default=2, help="supercell replication per axis")
    a_cif = ap.parse_args()

    a, ops, sites = parse_cif(CIF)
    assert len(ops) == 48, f"expected 48 ops, got {len(ops)}"
    kinds, frac = [], []
    for kind, xyz in sites:
        for op in ops:
            p = apply_op(op, xyz) % 1.0
            # dedupe with wrap-aware tolerance
            dup = False
            for q in frac:
                d = np.abs(p - q)
                d = np.minimum(d, 1.0 - d)
                if (d < 1e-4).all():
                    dup = True
                    break
            if not dup:
                frac.append(p)
                kinds.append(kind)
    frac = np.array(frac)
    kinds = np.array(kinds)
    n_si, n_o = int((kinds == "Si").sum()), int((kinds == "O").sum())
    print(f"unit cell: a = {a} A, {n_si} Si + {n_o} O")
    assert (n_si, n_o) == (24, 48), "LTA stoichiometry violated"

    cart = frac * a

    def min_image(d, L):
        return d - L * np.round(d / L)

    # Si-O bond sanity: every Si has 4 O at ~1.59-1.62 A
    si, ox = cart[kinds == "Si"], cart[kinds == "O"]
    d = np.linalg.norm(min_image(si[:, None, :] - ox[None, :, :], a), axis=-1)
    four = np.sort(d, axis=1)[:, :4]
    print(f"Si-O first shell: min {four.min():.3f} max {four.max():.3f} A")
    assert 1.55 < four.min() and four.max() < 1.70, "Si-O bond length wrong"

    # alpha-cage at the body centre: nearest O far away; window at face centre
    centre = np.array([a / 2, a / 2, a / 2])
    r_cage = np.linalg.norm(min_image(ox - centre, a), axis=-1).min()
    win = np.array([0.0, a / 2, a / 2])
    dw = np.linalg.norm(min_image(ox - win, a), axis=-1)
    ring = np.sort(dw)[:8]
    print(f"alpha-cage centre -> nearest O: {r_cage:.2f} A (expect ~5.4-5.8)")
    print(f"8-ring O radii around face centre: {ring.min():.2f}-{ring.max():.2f} A "
          f"(expect ~3.3-3.6)")
    assert r_cage > 5.0, "body centre is not the alpha-cage"
    assert ring.max() < 4.0, "face centre is not an 8-ring window"

    # supercell
    S = a_cif.super
    shifts = np.array([[i, j, k] for i in range(S) for j in range(S) for k in range(S)])
    pos = (cart[None, :, :] + shifts[:, None, :] * a).reshape(-1, 3)
    kind_all = np.tile(kinds, S ** 3)
    L = S * a
    print(f"supercell {S}x{S}x{S}: box L = {L:.3f} A, "
          f"{(kind_all == 'Si').sum()} Si + {(kind_all == 'O').sum()} O")

    np.savez_compressed(OUT, a_pseudo=a, box=L, supercell=S,
                        pos=pos, kind=kind_all,
                        o_pos=pos[kind_all == "O"],
                        cif="cache/lta/LTA.cif",
                        note=("IZA DLS76-optimized all-silica LTA, Pm-3m, expanded from the "
                              "asymmetric unit; alpha-cage at body centre (a/2,a/2,a/2), "
                              "8-ring windows at face centres; CV = ethane COM x mod a."))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
