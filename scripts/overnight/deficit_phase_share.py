"""Descriptive, ABF-only (not a classifier input): when does the window population reach its bias-aware target, and
how much of ABF's integrated error has accrued by then?  T_reach = end of the first 2 t.u. block after which the
MEDIAN-over-seeds block ratio (window occupancy / Q*_W) stays >= 0.75 for every later block (the first 1 t.u., the
bias ramp, is excluded because Q* is ill-defined while the bias is ~0).  Share = median over seeds of
int_0^T_reach e_F dt / I_F.  This early phase is the only one in which marginal FR could act."""
import glob, json, os, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import metrics as mt
from eri.model import KB
cells = [("ERI 600 K", "results/eri_campaign/screen_dilute", "results/eri_campaign/reference", "gard_tait"),
         ("ITE 300 K", "results/overnight/ite/screen", "results/overnight/ite/reference", "ite_itq3"),
         ("ITE 150 K", "results/overnight/ite150/screen", "results/overnight/ite150/reference", "ite_itq3")]
out = {}
for name, d, rb, var in cells:
    mt.set_reference_base(rb)
    for K in (64, 256, 1024):
        R, E, T = [], [], None
        for p in sorted(f for f in glob.glob(os.path.join(ROOT, d, f"abf_N{K}_s*.npz")) if not f.endswith("ckpt.npz")):
            r = mt.load_run(p); t = r["times"]; T = t
            Fr, Gr = mt.reference_for_grid(r["sv_C"].shape[1], var)
            e = mt.errors(r, Fr, Gr); E.append(e["eF"])
            m = mt.marginal(r, Fr, KB * r["cfg"]["T_K"])
            ratio = m["frac_win"] / np.maximum(m["Q_win"], 1e-12)
            edges = np.arange(1.0, t[-1] + 1e-9, 2.0)
            R.append([ratio[(t >= a) & (t < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
        R = np.median(np.array(R), 0)
        bad = np.nonzero(R < 0.75)[0]
        Treach = float(edges[bad.max() + 1]) if bad.size else float(edges[0])
        sel = T <= Treach
        shares = [np.trapezoid(ee[sel], T[sel]) / np.trapezoid(ee, T) for ee in E]
        out[f"{name} K{K}"] = dict(T_reach=Treach, frac_T=Treach / float(T[-1]), IF_share=float(np.median(shares)),
                                   first_block_ratio=float(R[0]))
        print(f"{name:10s} K={K:5d}: median window occupancy reaches >=0.75 Q* (for good) at {Treach:5.1f} t.u. "
              f"({100 * Treach / T[-1]:4.1f} % of T); I_F share accrued by then {100 * np.median(shares):5.1f} %; "
              f"first block (1-3 t.u.) ratio {R[0]:.2f}")
json.dump(out, open(os.path.join(ROOT, "results", "overnight", "deficit_phase_share.json"), "w"), indent=1)
