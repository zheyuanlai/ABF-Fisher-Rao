"""Experiment 5 (read-only): periodic mean-force circulation c(t) = (1/2pi) int Gamma_t dphi in paired ABF vs
ABF + FR blocks.  Nothing in either algorithm is changed; everything comes from saved accumulators.

    ABFFR_DATA_ROOT=... python scripts/convergence_fr/circulation.py X1 X2 [E1 ERI1024 ERI256 ...]

Per block:
* closure check of the reference: mean over bins of Gamma_ref (the exact mean force integrates to zero);
* c(t) (read-out) and c_app(t) (applied: ramp x trust x Gamma) per seed; late-half means; paired FR - ABF;
* late-window (last 20 %) estimate error per arm: F from M(T) - M(0.8T), C(T) - C(0.8T);
* across-seed Spearman correlations, within each arm: late-half c vs e_F(T), late-half c vs late-window error;
* FR event rate after the start (and after the stop, if any) and the pooled FR - ABF final profile difference.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atlas_lib as L  # noqa: E402
import systems as S  # noqa: E402
from eri import metrics as mt  # noqa: E402

BLOCKS = {
    "X1": ("results/overnight/ite/prod_confirm", 1024, "results/overnight/ite/reference", "ite_itq3", None),
    "X2": ("results/overnight/ite150/prod_confirm", 256, "results/overnight/ite150/reference", "ite_itq3", None),
    "E1": ("results/convergence_fr_campaign/E1_ite300_stop10", 1024, "results/overnight/ite/reference", "ite_itq3", None),
    "ERI1024": ("results/convergence_fr_campaign/ERI_K1024", 1024, "results/eri_campaign/reference", "gard_tait", None),
    "ERI256": ("results/convergence_fr_campaign/ERI_K256", 256, "results/eri_campaign/reference", "gard_tait", None),
}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def circ(r):
    c = r["cfg"]
    G = mt.mean_force(r["sv_M"], r["sv_C"])
    k = r["save_steps"].astype(float)
    ramp = np.ones_like(k) if c["warmup_steps"] <= 0 else np.minimum(1.0, k / c["warmup_steps"])
    w = np.ones_like(G) if c["trust_count"] <= 0 else np.minimum(1.0, r["sv_C"] / c["trust_count"])
    return G.mean(1), (ramp[:, None] * w * G).mean(1)


def block(name, spec):
    d = S.load_eri_cell(spec[0], spec[1], spec[2], spec[3], keep_runs=True)
    t, T = d["t"], d["T"]
    Fr, Gr = d["F_ref"], d["G_ref"]
    out = dict(block=name, K=d["K"], T=T, T_K=d["T_K"], rate=d["rate"], t_fr=d["t_fr"], t_stop=d["t_stop"],
               n_seeds=len(d["seeds"]), reference_closure_mean_Gamma=float(np.mean(Gr)))
    late = t >= 0.5 * T
    i8 = int(np.searchsorted(t, 0.8 * T - 1e-9))
    arms = {}
    for arm in ("abf", "fr"):
        rr = d["arms"][arm]["runs"]
        C = np.array([circ(r)[0] for r in rr]).T
        Ca = np.array([circ(r)[1] for r in rr]).T
        lw = []
        for r in rr:
            Mp = r["sv_M"][-1] - r["sv_M"][i8]
            Cp = r["sv_C"][-1] - r["sv_C"][i8]
            lw.append(float(mt.centred_rms(mt.histogram_pmf(mt.mean_force(Mp, Cp)) - Fr)))
        eT = d["arms"][arm]["eF"][-1]
        cl = C[late].mean(0)
        ih = int(np.searchsorted(t, 0.5 * T - 1e-9))
        cdep = np.array([mt.mean_force(r["sv_M"][-1] - r["sv_M"][ih], r["sv_C"][-1] - r["sv_C"][ih]).mean() for r in rr])
        arms[arm] = dict(c=C, capp=Ca, c_late=cl, lw=np.array(lw), eT=eT,
                         Ffin=mt.histogram_pmf(mt.mean_force(sum(r["sv_M"][-1] for r in rr), sum(r["sv_C"][-1] for r in rr))))
        arms[arm]["cdep"] = cdep
        out[arm] = dict(c_late_half_deposits_median=float(np.median(cdep)), c_late_half_median=float(np.median(cl)), c_late_half_iqr=[float(np.quantile(cl, 0.25)), float(np.quantile(cl, 0.75))],
                        capp_late_half_median=float(np.median(Ca[late].mean(0))),
                        c_final_median=float(np.median(C[-1])),
                        late_window_error_median=float(np.median(lw)),
                        spearman_c_late_vs_eF_T=spearman(cl, eT), spearman_c_late_vs_late_window_error=spearman(cl, np.array(lw)),
                        spearman_abs_c_late_vs_eF_T=spearman(np.abs(cl), eT))
    out["paired_c_late_difference_fr_minus_abf"] = L.boot_median(arms["fr"]["c_late"] - arms["abf"]["c_late"])
    out["paired_c_late_deposits_difference_fr_minus_abf"] = L.boot_median(arms["fr"]["cdep"] - arms["abf"]["cdep"])
    out["fr_gt_abf_c_late_deposits"] = int((arms["fr"]["cdep"] > arms["abf"]["cdep"]).sum())
    out["paired_late_window_error"] = L.paired_rel(arms["abf"]["lw"], arms["fr"]["lw"])
    # pooled across arms: does the paired FR-ABF circulation difference track the paired e_F(T) difference?
    out["spearman_paired_dc_vs_paired_deF_T"] = spearman(arms["fr"]["c_late"] - arms["abf"]["c_late"],
                                                         arms["fr"]["eT"] / arms["abf"]["eT"] - 1)
    dF = arms["fr"]["Ffin"] - arms["abf"]["Ffin"]
    out["pooled_final_profile_fr_minus_abf_rms"] = float(mt.centred_rms(dF))
    rb = d["arms"]["fr"]["runs"]
    ev = np.array([r["sv_i"][:, 2] for r in rb]).T                   # cumulative replacements per save
    rate_fr = np.gradient(np.median(ev, 1), t)
    out["fr_replacements_per_tu_median"] = dict(
        first_5_after_start=float(np.mean(rate_fr[(t >= d["t_fr"]) & (t < d["t_fr"] + 5)])),
        late_half=float(np.mean(rate_fr[late])))
    if d["t_stop"] is not None:
        pre = (t >= d["t_stop"] - 5) & (t < d["t_stop"])
        post = (t >= d["t_stop"]) & (t < d["t_stop"] + 5)
        out["c_around_stop"] = {arm: dict(pre5=float(np.median(arms[arm]["c"][pre].mean(0))),
                                          post5=float(np.median(arms[arm]["c"][post].mean(0))))
                                for arm in ("abf", "fr")}
    curves = dict(t=t, dF_pooled=dF, F_ref=Fr)
    for arm in ("abf", "fr"):
        curves[f"c_{arm}"] = arms[arm]["c"]
        curves[f"capp_{arm}"] = arms[arm]["capp"]
    return out, curves


def main():
    names = sys.argv[1:] or ["X1", "X2"]
    outdir = os.path.join(S.ROOT, "results", "convergence_fr_campaign", "mechanism")
    os.makedirs(outdir, exist_ok=True)
    for n in names:
        o, c = block(n, BLOCKS[n])
        json.dump(o, open(os.path.join(outdir, f"circulation_{n}.json"), "w"), indent=1, default=float)
        np.savez(os.path.join(outdir, f"circulation_{n}_curves.npz"), **c)
        print(json.dumps(o, indent=1, default=float))


if __name__ == "__main__":
    main()
