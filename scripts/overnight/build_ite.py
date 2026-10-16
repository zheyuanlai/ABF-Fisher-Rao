"""Build rigid Si-ITE (ITQ-3, Camblor et al. 1997; RASPA2 ITQ-3.cif) with crystal b along box X, map the ethane-
accessible channels with the Dubbeldam-2004 CH3-O potential, locate cage and window centres, save
cache/eri/framework_ite_itq3.npz.  CPU only, < 2 min."""
import json, math, os, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import framework as fwm
CIF = os.path.join(ROOT, "cache", "overnight", "literature", "candA", "raspa2_cif_ITQ_3_cif", "ITQ-3.cif")
FF = dict(T_K=300.0, eps_O_K=93.0, sig_O=3.48, eps_Si_K=0.0, sig_Si=1.0, eps_CC_K=108.0, sig_CC=3.76, rc=12.0,
          shift=True, k_bond_K=96500.0, r0=1.54)
fw = fwm.build_orthorhombic(CIF, reps=(3, 2, 2), perm=(1, 2, 0), name="ite_itq3", n_t=64, n_o=128)
ok, dmin, (lo, hi) = fwm.coordination_check(fw)
print("box", fw["box"], "cell", fw["cell"], "Si", (fw["types"] == 1).sum(), "O", (fw["types"] == 0).sum(),
      "coordination", ok, "Si-O", round(lo, 4), round(hi, 4), "dmin", round(dmin, 3))
# single CH3 bead energy on a grid over one unit cell (box axes: X = b, Y = c, Z = a), O only, shifted LJ, 12 A
KB = 0.008314462618
O = fw["pos"][fw["types"] == 0]
box = fw["box"]
cell = fw["cell"]
eps, sig, rc = 93.0 * KB, 3.48, 12.0
vrc = 4 * eps * ((sig / rc) ** 12 - (sig / rc) ** 6)
h = 0.25
gx = np.arange(0, cell[0], h); gy = np.arange(0, cell[1], h); gz = np.arange(0, cell[2], h)
E = np.zeros((len(gx), len(gy), len(gz)))
for i, x in enumerate(gx):
    P = np.stack(np.meshgrid([x], gy, gz, indexing="ij"), -1).reshape(-1, 3)
    d = P[:, None, :] - O[None, :, :]
    d -= box * np.round(d / box)
    r2 = (d * d).sum(-1)
    m = r2 < rc * rc
    s6 = (sig * sig / np.where(m, r2, 1.0)) ** 3
    e = np.where(m, 4 * eps * (s6 * s6 - s6) - vrc, 0.0).sum(1)
    E[i] = e.reshape(len(gy), len(gz))
kT = KB * 300.0
print("bead energy min %.2f kJ/mol; fraction of cell with E < 0: %.3f" % (E.min(), (E < 0).mean()))
# connected accessible region (E < 0) with periodic labelling (scipy)
from scipy import ndimage
acc = E < 0.0
lab, n = ndimage.label(acc)
# merge labels across periodic faces
parent = list(range(n + 1))
def f(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for ax in range(3):
    s0 = [slice(None)] * 3; s1 = [slice(None)] * 3
    s0[ax] = 0; s1[ax] = -1
    a, b = lab[tuple(s0)], lab[tuple(s1)]
    for u, v in zip(a.ravel(), b.ravel()):
        if u and v:
            parent[f(u)] = f(v)
roots = np.array([f(l) for l in range(n + 1)])
lab2 = roots[lab]
comps = [c for c in np.unique(lab2) if c != 0]
info = []
for c in comps:
    msk = lab2 == c
    vol = msk.sum() * h ** 3
    spanx = msk.any((1, 2)).all()
    info.append((c, vol, spanx))
info.sort(key=lambda t: -t[1])
print("components (label, volume A^3, spans X):", [(int(c), round(v, 1), bool(s)) for c, v, s in info[:8]])
# per channel (components spanning X): min energy over (Y,Z) at each X -> cage minimum and window maximum
chans = [c for c, v, s in info if s]
res = []
for c in chans:
    msk = lab2 == c
    Em = np.where(msk, E, np.inf)
    prof = Em.reshape(len(gx), -1).min(1)
    iw = int(np.argmax(prof)); ic = int(np.argmin(prof))
    jw = np.unravel_index(np.argmin(Em[iw]), Em[iw].shape)
    jc = np.unravel_index(np.argmin(Em[ic]), Em[ic].shape)
    # COM-like centre of the channel cross-section at the cage and the window (mean of accessible points)
    yz_w = np.argwhere(msk[iw]); yz_c = np.argwhere(msk[ic])
    res.append(dict(label=int(c), x_window=float(gx[iw]), x_cage_min=float(gx[ic]), bead_barrier_kJ=float(prof[iw] - prof[ic]),
                    window_point=[float(gx[iw]), float(gy[jw[0]]), float(gz[jw[1]])],
                    window_section_centre=[float(gx[iw]), float(gy[yz_w[:, 0]].mean()), float(gz[yz_w[:, 1]].mean())],
                    cage_section_centre=[float(gx[ic]), float(gy[yz_c[:, 0]].mean()), float(gz[yz_c[:, 1]].mean())],
                    profile=prof.tolist()))
for r in res:
    print("channel", r["label"], "window X %.2f cage-min X %.2f bead barrier %.2f kJ/mol (%.2f kT); window centre %s; cage centre %s" % (
        r["x_window"], r["x_cage_min"], r["bead_barrier_kJ"], r["bead_barrier_kJ"] / kT,
        np.round(r["window_section_centre"], 2), np.round(r["cage_section_centre"], 2)))
json.dump(dict(box=box.tolist(), cell=cell.tolist(), channels=res, ff=FF), open(os.path.join(ROOT, "cache", "eri", "ite_itq3_channels.json"), "w"), indent=1)
np.save(os.path.join(ROOT, "cache", "eri", "ite_itq3_bead_grid.npy"), E)
