"""Percolation of the single-CH3-bead energy landscape of rigid ITE (Dubbeldam 2004 O-CH3) along each box axis."""
import json, os, sys
import numpy as np
from scipy import ndimage
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
E = np.load(os.path.join(ROOT, "cache", "eri", "ite_itq3_bead_grid.npy"))
KB = 0.008314462618; kT = KB * 300
def perc(thr):
    acc = E < thr
    lab, n = ndimage.label(acc)
    out = []
    for ax in range(3):
        # periodic percolation along ax: a label present on both faces AND connected through periodic images in the other axes
        s0 = [slice(None)] * 3; s1 = [slice(None)] * 3; s0[ax] = 0; s1[ax] = -1
        a, b = lab[tuple(s0)], lab[tuple(s1)]
        both = set(np.unique(a[(a > 0) & (b > 0) & (a == b)]))
        # tile 2x along ax and check whether a component touching the start face reaches the end face of the tile
        T = np.concatenate([acc, acc], axis=ax)
        L2, n2 = ndimage.label(T)
        f0 = set(np.unique(L2[tuple(s0)])) - {0}
        s1b = [slice(None)] * 3; s1b[ax] = -1
        f1 = set(np.unique(L2[tuple(s1b)])) - {0}
        out.append(len(f0 & f1) > 0)
    return out
for thr in (0, 2, 5, 8, 10, 12, 15, 20, 30, 50, 100, 300):
    print(f"E < {thr:4d} kJ/mol ({thr / kT:6.1f} kT): percolates along X(b) / Y(c) / Z(a) = {perc(thr)}")
