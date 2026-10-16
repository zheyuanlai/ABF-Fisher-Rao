"""Summarize ABF-only calibration runs: final error, I_F, late-deposit error, establishment, conditional lag."""
import glob, os, sys, re, numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import metrics as mt
from eri.model import KB
pat = sys.argv[1]
rows = {}
for p in sorted(glob.glob(pat)):
    r = mt.load_run(p)
    ng = r["sv_C"].shape[1]
    Fr, Gr, _ = mt.reference_on_grid(ng)
    e = mt.errors(r, Fr, Gr)
    t = e["t"]
    i2 = np.searchsorted(t, 2.0)
    G = mt.mean_force(r["sv_M"][-1] - r["sv_M"][i2], r["sv_C"][-1] - r["sv_C"][i2])
    late = mt.centred_rms(mt.histogram_pmf(G) - Fr)
    key = re.sub(r"_s\d+\.npz$", "", os.path.basename(p))
    rows.setdefault(key, []).append((e["eF_T"], e["Ibar_F"], late, e["eF"][np.searchsorted(t, 10)], e["eF"][np.searchsorted(t, 20)]))
print(f"{'config':32s} n  eF(T)          Ibar_F        late(t>2)  eF(10)  eF(20)")
for k, v in rows.items():
    v = np.array(v)
    print(f"{k:32s} {len(v)}  {np.median(v[:,0]):.4f} [{v[:,0].min():.3f},{v[:,0].max():.3f}]  {np.median(v[:,1]):.4f}  {np.median(v[:,2]):.4f}    {np.median(v[:,3]):.3f}   {np.median(v[:,4]):.3f}")
