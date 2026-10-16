"""Quick look at ERI engine runs: e_F(t), window fraction vs bias-aware target, events, genealogy."""
import sys, os, numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import metrics as mt
from eri.model import KB
for p in sys.argv[1:]:
    r = mt.load_run(p)
    c = r["cfg"]
    ng = r["sv_C"].shape[1]
    Fr, Gr, _ = mt.reference_on_grid(ng)
    e = mt.errors(r, Fr, Gr)
    kT = KB * c["T_K"]
    m = mt.marginal(r, Fr, kT)
    t = e["t"]
    print(f"== {os.path.basename(p)} N={c['N']} T={t[-1]:.1f} I_F={e['I_F']:.3f} Ibar={e['Ibar_F']:.4f} eF(T)={e['eF_T']:.4f} eG(T)={e['eG_T']:.3f}")
    for tt in (0.5, 1, 2, 4, 8, 12, 16, 20, 30, 40, 60, 80):
        i = np.searchsorted(t, tt)
        if i < len(t):
            print(f"  t={t[i]:5.1f} eF={e['eF'][i]:.4f} eG={e['eG'][i]:.2f} win {m['frac_win'][i]:.3f} Q* {m['Q_win'][i]:.3f} tvU {m['tv_unif'][i]:.3f} tvQ {m['tv_Q'][i]:.3f} (floor {m['tv_floor_Q'][i]:.3f}) cross {r['sv_i'][i,0]} transloc {r['sv_i'][i,1]} repl {r['sv_i'][i,2]} ESSw {r['sv_u'][i,2]:.2f}")
    print("  first window step", r["ist"][5], "first translocation step", r["ist"][6], "max|bias|", r["fst"][1], "wall", r.get("wall_seconds_session"))
