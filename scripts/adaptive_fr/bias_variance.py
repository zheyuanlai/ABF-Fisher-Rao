"""Experiment D1: bias-variance decomposition of histogram-ABF estimates across independent seeds, per arm, over time.

For an arm with n seeds and a quantity X_t (the centred free-energy profile F_t, or the bin mean force Gamma_t) on a
fixed grid, at every save:
    mean-square error   MSE(t)  = mean_seeds ||X_t - X_ref||^2 / n_bins                     (= median-free RMS^2)
    variance            V(t)    = mean_bins  var_seeds(X_t)                                 (unbiased, ddof = 1)
    squared bias        B2(t)   = ||mean_seeds X_t - X_ref||^2 / n_bins  -  V(t) / n  -  s_ref^2
with the finite-seed correction V/n (E||xbar - mu||^2 = ||bias||^2 + tr Sigma / n) and the reference's own noise
s_ref^2 (mean squared standard error of the exact-MC reference) removed; B2 can come out slightly negative from noise
and is reported as is.  F is centred (additive constant removed) before every norm.  Late-only variant: the same
for the estimate built only from deposits after a split step (after the FR stop for stopped schedules, after T/2
otherwise), against the same reference.  For stopped schedules the ABF arm also gets the matched-window variant
(ABF deposits after the FR arm's stop step), so the two arms can be compared over the same window.

    ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao python scripts/adaptive_fr/bias_variance.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "convergence_fr"))
sys.path.insert(0, HERE)
import systems as S  # noqa: E402
from eri import metrics as mt  # noqa: E402

BLOCKS = {
    # name: (dir, K, refbase, variant, abf_dir, label)
    "X1": ("results/overnight/ite/prod_confirm", 1024, "results/overnight/ite/reference", None, "ITE 300 K K1024, continuous FR (exploratory)"),
    "E1": ("results/convergence_fr_campaign/E1_ite300_stop10", 1024, "results/overnight/ite/reference", None, "ITE 300 K K1024, FR stop 10"),
    "A": ("results/adaptive_fr_campaign/A_ite300_K256_T200_stop10", 256, "results/overnight/ite/reference",
          "results/convergence_fr_campaign/EB_ite300_abf_K256_T200", "ITE 300 K K256 T200, FR stop 10"),
    "M300_256": ("results/convergence_fr_campaign/M300_256_rate1", 256, "results/overnight/ite/reference", None, "ITE 300 K K256, continuous FR"),
    "M150_256": ("results/convergence_fr_campaign/M150_256_rate1", 256, "results/overnight/ite150/reference", None, "ITE 150 K K256, continuous FR"),
    "X2": ("results/overnight/ite150/prod_confirm", 256, "results/overnight/ite150/reference", None, "ITE 150 K K256 T200, continuous FR (exploratory; 360 bins)"),
    "E2": ("results/convergence_fr_campaign/E2_ite150_stop15", 256, "results/overnight/ite150/reference", None, "ITE 150 K K256 T100, FR stop 15"),
    "ERI256": ("results/convergence_fr_campaign/ERI_K256", 256, "results/eri_campaign/reference", None, "ERI 600 K K256, continuous FR"),
    "ERI1024": ("results/convergence_fr_campaign/ERI_K1024", 1024, "results/eri_campaign/reference", None, "ERI 600 K K1024, continuous FR"),
}


def ref_noise(refbase, variant, ng):
    """Mean squared standard error of the exact-MC reference F on the run grid (0 if not available)."""
    try:
        z = np.load(os.path.join(S.dpath(refbase), variant, "reference_profile.npz"))
        se = np.asarray(z["F_se"], float) if "F_se" in z.files else None
        if se is None or se.size != ng:
            return 0.0
        return float(np.mean(se ** 2))
    except Exception:
        return 0.0


def decompose(X, Xref, sref2=0.0):
    """X (n_seeds, n_save, n_bins), centred if F; returns MSE, V, B2 over saves."""
    n = X.shape[0]
    d = X - Xref[None, None, :]
    mse = (d ** 2).mean(-1).mean(0)
    V = X.var(axis=0, ddof=1).mean(-1)
    m = d.mean(0)
    B2 = (m ** 2).mean(-1) - V / n - sref2
    return mse, V, B2


def centre(F):
    return F - F.mean(-1, keepdims=True)


def block(name):
    d, K, refbase, abf_dir, label = BLOCKS[name]
    variant = "gard_tait" if "eri_campaign" in refbase else "ite_itq3"
    mt.set_reference_base(S.dpath(refbase))
    out = dict(block=name, label=label)
    arrays = {}
    # the FR arm's split (its stop, or T/2); used for the ABF arm's matched-window late-only variant
    c_fr = mt.load_run(S.glob_both(d, f"fr_N{K}_s*.npz")[0])["cfg"]
    stop_fr = int(c_fr.get("fr_stop_steps", -1))
    ks_fr = stop_fr if stop_fr >= 0 else None
    for arm, dd, pat in (("abf", abf_dir or d, f"abf_N{K}_s*.npz"), ("fr", d, f"fr_N{K}_s*.npz")):
        runs = [mt.load_run(f) for f in S.glob_both(dd, pat)]
        ng = runs[0]["sv_C"].shape[1]
        Fr, Gr = mt.reference_for_grid(ng, variant)
        sref2 = ref_noise(refbase, variant, ng)
        t = runs[0]["times"]
        G = np.array([mt.mean_force(r["sv_M"], r["sv_C"]) for r in runs])
        F = centre(mt.histogram_pmf(G))
        mF, VF, BF = decompose(F, centre(Fr), sref2)
        mG, VG, BG = decompose(G, Gr)
        # late-only: deposits after the FR arm's stop (paired seeds share it) or after T/2
        c = runs[0]["cfg"]
        stop = int(c.get("fr_stop_steps", -1))
        ks = stop if stop >= 0 else int(runs[0]["save_steps"][-1] // 2)
        i0 = int(np.searchsorted(runs[0]["save_steps"], ks))
        Gl = np.array([mt.mean_force(r["sv_M"][i0:] - r["sv_M"][i0], r["sv_C"][i0:] - r["sv_C"][i0]) for r in runs])
        Fl = centre(mt.histogram_pmf(Gl))
        lF = decompose(Fl[:, 1:], centre(Fr), sref2)
        arrays[arm] = dict(t=t, mse=mF, var=VF, bias2=BF, mseG=mG, varG=VG, bias2G=BG, t_late=t[i0 + 1:], late=lF)
        out[arm] = dict(n_seeds=len(runs), ref_noise_sq=sref2, split_t=float(ks * c["dt"]),
                        final=dict(rmse=float(np.sqrt(mF[-1])), sd=float(np.sqrt(VF[-1])), bias=float(np.sign(BF[-1]) * np.sqrt(abs(BF[-1]))),
                                   bias_share_of_mse=float(BF[-1] / mF[-1])),
                        final_meanforce=dict(rmse=float(np.sqrt(mG[-1])), sd=float(np.sqrt(VG[-1])),
                                             bias=float(np.sign(BG[-1]) * np.sqrt(abs(BG[-1]))), bias_share_of_mse=float(BG[-1] / mG[-1])),
                        late_only_final=dict(rmse=float(np.sqrt(lF[0][-1])), sd=float(np.sqrt(lF[1][-1])),
                                             bias=float(np.sign(lF[2][-1]) * np.sqrt(abs(lF[2][-1]))),
                                             bias_share_of_mse=float(lF[2][-1] / lF[0][-1])))
        if arm == "abf" and ks_fr is not None:
            # matched window: ABF deposits after the FR arm's stop step (same window as the FR late-only estimate)
            j0 = int(np.searchsorted(runs[0]["save_steps"], ks_fr))
            Gm = np.array([mt.mean_force(r["sv_M"][j0:] - r["sv_M"][j0], r["sv_C"][j0:] - r["sv_C"][j0]) for r in runs])
            mm = decompose(centre(mt.histogram_pmf(Gm))[:, 1:], centre(Fr), sref2)
            out[arm]["late_only_matched_to_fr_stop"] = dict(split_t=float(ks_fr * c["dt"]), rmse=float(np.sqrt(mm[0][-1])),
                                                            sd=float(np.sqrt(mm[1][-1])),
                                                            bias=float(np.sign(mm[2][-1]) * np.sqrt(abs(mm[2][-1]))),
                                                            bias_share_of_mse=float(mm[2][-1] / mm[0][-1]))
        for frac in (0.1, 0.25, 0.5):
            i = int(np.searchsorted(t, frac * t[-1] - 1e-9))
            out[arm][f"at_{frac:g}T"] = dict(rmse=float(np.sqrt(mF[i])), sd=float(np.sqrt(VF[i])),
                                              bias=float(np.sign(BF[i]) * np.sqrt(abs(BF[i]))))
    return out, arrays


def main():
    names = sys.argv[1:] or list(BLOCKS)
    res = {}
    store = {}
    for n in names:
        o, arr = block(n)
        res[n] = o
        for arm in ("abf", "fr"):
            for k, v in arr[arm].items():
                if k == "late":
                    for j, nm in enumerate(("mse", "var", "bias2")):
                        store[f"{n}_{arm}_late_{nm}"] = v[j]
                else:
                    store[f"{n}_{arm}_{k}"] = v
        f = lambda a, k: o[a][k]
        print(f"{n:9s} {o['label']}")
        for arm in ("abf", "fr"):
            fi, la = o[arm]["final"], o[arm]["late_only_final"]
            print(f"   {arm:3s} final F: rmse {fi['rmse']:.4f} = bias {fi['bias']:+.4f} (share {fi['bias_share_of_mse']:.2f}) & sd {fi['sd']:.4f} | "
                  f"late-only (after t={o[arm]['split_t']:g}): rmse {la['rmse']:.4f} bias {la['bias']:+.4f} sd {la['sd']:.4f} | "
                  f"T/4: rmse {o[arm]['at_0.25T']['rmse']:.3f} bias {o[arm]['at_0.25T']['bias']:+.3f} sd {o[arm]['at_0.25T']['sd']:.3f}")
    p = os.path.join(S.ROOT, "results", "adaptive_fr_campaign", "mechanism")
    os.makedirs(p, exist_ok=True)
    json.dump(res, open(os.path.join(p, "bias_variance.json"), "w"), indent=1)
    np.savez(os.path.join(p, "bias_variance_curves.npz"), **store)
    print("wrote", os.path.join(p, "bias_variance.json"))


if __name__ == "__main__":
    main()
