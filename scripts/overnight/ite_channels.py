"""Channels of rigid ITE spanning X at a bead-energy threshold; cage/window centres; channel classes."""
import json, os, sys
import numpy as np
from scipy import ndimage
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
E = np.load(os.path.join(ROOT, "cache", "eri", "ite_itq3_bead_grid.npy"))
J = json.load(open(os.path.join(ROOT, "cache", "eri", "ite_itq3_channels.json")))
cell = np.array(J["cell"]); h = 0.25
gx = np.arange(0, cell[0], h); gy = np.arange(0, cell[1], h); gz = np.arange(0, cell[2], h)
KB = 0.008314462618; kT = KB * 300
thr = 8.0
acc = E < thr
lab, n = ndimage.label(acc, structure=np.ones((3, 3, 3)))
# periodic merge in X only (channels run along X); also merge across Y/Z faces for components split by the cell boundary
parent = list(range(n + 1))
def f(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for ax in range(3):
    s0 = [slice(None)] * 3; s1 = [slice(None)] * 3; s0[ax] = 0; s1[ax] = -1
    for u, v in zip(lab[tuple(s0)].ravel(), lab[tuple(s1)].ravel()):
        if u and v:
            parent[f(u)] = f(v)
roots = np.array([f(l) for l in range(n + 1)]); L = roots[lab]
out = []
for c in [c for c in np.unique(L) if c]:
    m = L == c
    if not m.any((1, 2)).all():
        continue
    Em = np.where(m, E, np.inf)
    prof = Em.reshape(len(gx), -1).min(1)
    iw = int(np.argmax(prof)); ic = int(np.argmin(prof))
    # circular mean of Y, Z positions of the channel (Y period cell[1], Z period cell[2])
    pts = np.argwhere(m)
    def cmean(idx, P):
        th = 2 * np.pi * idx * h / P
        return (np.arctan2(np.sin(th).mean(), np.cos(th).mean()) % (2 * np.pi)) * P / (2 * np.pi)
    yc, zc = cmean(pts[:, 1], cell[1]), cmean(pts[:, 2], cell[2])
    wpt = np.unravel_index(np.argmin(Em[iw]), Em[iw].shape)
    cpt = np.unravel_index(np.argmin(Em[ic]), Em[ic].shape)
    out.append(dict(label=int(c), y_axis=float(yc), z_axis=float(zc), x_window=float(gx[iw]), x_cagemin=float(gx[ic]),
                    window_point=[float(gx[iw]), float(gy[wpt[0]]), float(gz[wpt[1]])],
                    cage_point=[float(gx[ic]), float(gy[cpt[0]]), float(gz[cpt[1]])],
                    bead_barrier=float(prof[iw] - prof[ic]), vol=float(m.sum() * h ** 3),
                    prof=[float(p) for p in prof]))
for o in out:
    print("chan %d axis (Y,Z)=(%.2f, %.2f) vol %.0f  window X %.2f point %s  cage-min X %.2f point %s  bead barrier %.2f kJ/mol = %.2f kT" % (
        o["label"], o["y_axis"], o["z_axis"], o["vol"], o["x_window"], np.round(o["window_point"], 2), o["x_cagemin"],
        np.round(o["cage_point"], 2), o["bead_barrier"], o["bead_barrier"] / kT))
    p = np.array(o["prof"]); print("   bead profile (every 1 A along X):", np.round(p[::4], 1))
J["channels_thr8"] = out
json.dump(J, open(os.path.join(ROOT, "cache", "eri", "ite_itq3_channels.json"), "w"), indent=1)
