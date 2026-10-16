"""FR mechanism / long-time correctness diagnostics for a paired block (ABF vs ABF + FR only).
1. pooled LATE-half estimators (deposits in [T/2, T]) per arm, seed-jackknife noise, debiased D(FR, ABF), D(each, MC);
2. where the FR-ABF late difference sits (window |phi|<0.5 / flank / cage);
3. deposit-weighted conditional means of eta and u_x^2 per coarse bin (whole run) vs the MC reference;
4. deposit counts per region."""
import argparse, glob, json, math, os, re, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import metrics as mt
ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--K", type=int, required=True); ap.add_argument("--refbase", required=True)
ap.add_argument("--variant", default="ite_itq3")
a = ap.parse_args()
mt.set_reference_base(a.refbase)
arms = {"abf": [], "fr": []}
for m in arms:
    for f in sorted(glob.glob(os.path.join(ROOT, a.dir, f"{m}_N{a.K}_s*.npz"))):
        if f.endswith((".ckpt.npz", ".tmp.npz")):
            continue
        z = np.load(f, allow_pickle=False)
        t = z["save_steps"] * json.loads(str(z["cfg"]))["dt"]
        ih = int(np.searchsorted(t, 0.5 * t[-1]))
        arms[m].append(dict(Ch=z["sv_C"][-1] - z["sv_C"][ih], Mh=z["sv_M"][-1] - z["sv_M"][ih], C=z["sv_C"][-1],
                            M=z["sv_M"][-1], Seta=z["Seta"], Sux2=z["Sux2"]))
ng = arms["abf"][0]["C"].shape[0]
Fr, Gr = mt.reference_for_grid(ng, a.variant)
grid = (np.arange(ng) + 0.5) / ng * 2 * np.pi - np.pi
W = np.abs(grid) < 0.5; Cg = np.abs(grid) >= 2.0; Fl = ~W & ~Cg
def prof(runs, idx, late=True):
    C = sum(runs[i]["Ch" if late else "C"] for i in idx); M = sum(runs[i]["Mh" if late else "M"] for i in idx)
    G = mt.mean_force(M, C); return mt.histogram_pmf(G), G
def jack(runs, late=True):
    n = len(runs); F, G = prof(runs, range(n), late)
    loo = np.array([prof(runs, [j for j in range(n) if j != i], late)[0] for i in range(n)])
    se = np.sqrt((n - 1) / n * ((loo - loo.mean(0)) ** 2).sum(0)); return F, G, se
def D(Fa, sa, Fb, sb):
    d = Fa - Fb; d = d - d.mean(); n2 = float(np.mean(sa ** 2 + sb ** 2)); r = float(np.sqrt(np.mean(d ** 2)))
    return dict(D=math.sqrt(max(0.0, r * r - n2)), rms=r, noise=math.sqrt(n2))
out = {}
for late in (True, False):
    tag = "late_half" if late else "whole_run"
    Fa, Ga, sa = jack(arms["abf"], late); Ff, Gf, sf = jack(arms["fr"], late)
    z0 = np.zeros_like(Fr)
    out[tag] = dict(fr_vs_abf=D(Ff, sf, Fa, sa), abf_vs_mc=D(Fa, sa, Fr, z0), fr_vs_mc=D(Ff, sf, Fr, z0))
    dd = (Ff - Fa) - (Ff - Fa).mean(); dG = Gf - Ga
    out[tag]["region_rms_dF"] = {k: float(np.sqrt(np.mean(dd[msk] ** 2))) for k, msk in (("window", W), ("flank", Fl), ("cage", Cg))}
    out[tag]["region_rms_dGamma"] = {k: float(np.sqrt(np.mean(dG[msk] ** 2))) for k, msk in (("window", W), ("flank", Fl), ("cage", Cg))}
    out[tag]["region_mean_dGamma_vs_ref"] = {m: {k: float(np.mean((G - Gr)[msk])) for k, msk in (("window", W), ("flank", Fl), ("cage", Cg))} for m, G in (("abf", Ga), ("fr", Gf))}
    print(tag, json.dumps(out[tag], indent=None, default=float))
# deposit-weighted conditional hidden variables (whole run), per coarse bin, vs the MC reference
ref = np.load(os.path.join(mt.REF_BASE, a.variant, "reference_profile.npz"))
cc = ref["counts"].reshape(18, -1)
re_eta = (ref["eta"].reshape(18, -1) * cc).sum(1) / cc.sum(1); re_ux = (ref["ux2"].reshape(18, -1) * cc).sum(1) / cc.sum(1)
for m in arms:
    C = sum(r["C"] for r in arms[m]).reshape(18, -1).sum(1)
    eta = sum(r["Seta"] for r in arms[m]).reshape(18, -1).sum(1) / C
    ux = sum(r["Sux2"] for r in arms[m]).reshape(18, -1).sum(1) / C
    out[f"cond_{m}"] = dict(eta_minus_ref=(eta - re_eta).tolist(), ux2_minus_ref=(ux - re_ux).tolist(),
                            rms_eta=float(np.sqrt(np.mean((eta - re_eta) ** 2))), rms_ux2=float(np.sqrt(np.mean((ux - re_ux) ** 2))),
                            deposit_share={k: float(sum(r["C"] for r in arms[m])[msk].sum() / sum(r["C"] for r in arms[m]).sum()) for k, msk in (("window", W), ("flank", Fl), ("cage", Cg))})
    print(m, "cond. RMS(eta-ref) %.4f RMS(ux2-ref) %.4f deposit share" % (out[f"cond_{m}"]["rms_eta"], out[f"cond_{m}"]["rms_ux2"]), out[f"cond_{m}"]["deposit_share"])
    print("   eta-ref per coarse bin:", np.round(eta - re_eta, 3))
    print("   ux2-ref per coarse bin:", np.round(ux - re_ux, 3))
json.dump(out, open(os.path.join(ROOT, a.dir, f"fr_mechanism_K{a.K}.json"), "w"), indent=1, default=float)
