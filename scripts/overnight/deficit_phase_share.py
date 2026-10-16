"""Descriptive, ABF-only: how much of ABF's integrated error accrues while the window population is still deficient
(block occupancy / Q* < 0.75, 2 t.u. blocks)?  This is the only phase in which marginal FR could act."""
import glob, json, os, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import metrics as mt
from eri.model import KB
cells = [("ERI 600 K", "results/eri_campaign/screen_dilute", "results/eri_campaign/reference", "gard_tait"),
         ("ITE 300 K", "results/overnight/ite/screen", "results/overnight/ite/reference", "ite_itq3")]
if len(sys.argv) > 1:
    cells.append(("ITE 150 K", sys.argv[1], "results/overnight/ite150/reference", "ite_itq3"))
out = {}
for name, d, rb, var in cells:
    mt.set_reference_base(rb)
    Fr, Gr, _ = mt.reference_on_grid(180, var)
    for K in (64, 256, 1024):
        rows = []
        for p in sorted(f for f in glob.glob(os.path.join(ROOT, d, f"abf_N{K}_s*.npz")) if not f.endswith("ckpt.npz")):
            r = mt.load_run(p); t = r["times"]
            Fr, Gr = mt.reference_for_grid(r["sv_C"].shape[1], var)
            e = mt.errors(r, Fr, Gr)
            m = mt.marginal(r, Fr, KB * r["cfg"]["T_K"])
            R = m["frac_win"] / np.maximum(m["Q_win"], 1e-12)
            edges = np.arange(0, t[-1] + 1e-9, 2.0)
            last = 0.0
            for a, b in zip(edges[:-1], edges[1:]):
                sel = (t >= a) & (t < b)
                if sel.any() and R[sel].mean() < 0.75:
                    last = b
            I = np.trapezoid(e["eF"], t)
            sel = t <= last
            Idef = np.trapezoid(e["eF"][sel], t[sel]) if sel.sum() > 1 else 0.0
            rows.append((last, last / t[-1], Idef / I))
        rows = np.array(rows)
        out[f"{name} K{K}"] = dict(deficit_end_tu=float(np.median(rows[:, 0])), deficit_frac_T=float(np.median(rows[:, 1])),
                                   IF_share_in_deficit=float(np.median(rows[:, 2])))
        print(f"{name:10s} K={K:5d}: window deficit (block ratio<0.75) ends at {np.median(rows[:,0]):5.1f} t.u. "
              f"({100*np.median(rows[:,1]):4.1f} % of T); share of I_F accrued by then {100*np.median(rows[:,2]):5.1f} %")
json.dump(out, open(os.path.join(ROOT, "results", "overnight", "deficit_phase_share.json"), "w"), indent=1)
