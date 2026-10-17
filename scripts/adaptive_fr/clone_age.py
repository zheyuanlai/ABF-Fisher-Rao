"""Experiment D2 (exploratory): do deposits from recently cloned walkers (cloned < 1 t.u. ago) carry the FR bias?

Per block (continuous FR, re-run bitwise with the read-only age split):
* young share of deposits per region (window |phi| < 0.5, flank, cage |phi| >= 2), whole run and late half;
* pooled-over-seeds late-half (t > T/2) mean force from young deposits, old deposits, all deposits, and the ABF arm,
  each vs the exact-MC reference: count-weighted RMS over bins of (Gamma - Gamma_ref), seed-jackknife s.e.;
* pooled late-half free-energy error (centred RMS) of the estimate from ALL FR deposits vs from OLD deposits only
  vs ABF -- does dropping the young deposits remove FR's late bias?
* eta conditional (ERI: the cage-axis coordinate) and u_x^2 conditional (relative) by age class vs the reference,
  per coarse bin.

    ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao python scripts/adaptive_fr/clone_age.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "convergence_fr"))
import systems as S  # noqa: E402
from eri import metrics as mt  # noqa: E402

BLOCKS = {
    "ERI256": ("results/adaptive_fr_campaign/D2_eri600_K256_age", "results/convergence_fr_campaign/ERI_K256", 256,
               "results/eri_campaign/reference", "gard_tait"),
    "ITE300_K256": ("results/adaptive_fr_campaign/D2_ite300_K256_age", "results/convergence_fr_campaign/M300_256_rate1", 256,
                    "results/overnight/ite/reference", "ite_itq3"),
}


def wrms(D, w):
    w = np.asarray(w, float)
    m = w > 0
    return float(np.sqrt((w[m] * D[m] ** 2).sum() / w[m].sum()))


def block(name):
    fr_dir, abf_dir, K, refbase, variant = BLOCKS[name]
    mt.set_reference_base(S.dpath(refbase))
    Fr, Gr = mt.reference_for_grid(180, variant)
    ref = np.load(os.path.join(S.dpath(refbase), variant, "reference_profile.npz"))
    fr = [mt.load_run(f) for f in S.glob_both(fr_dir, f"fr_N{K}_s*.npz")]
    ab = [mt.load_run(f) for f in S.glob_both(abf_dir, f"abf_N{K}_s*.npz")]
    assert len(fr) == len(ab) == 16
    ih = int(np.searchsorted(fr[0]["save_steps"], fr[0]["save_steps"][-1] // 2))
    grid = (np.arange(180) + 0.5) / 180 * 2 * np.pi - np.pi
    reg = {"window": np.abs(grid) < 0.5, "flank": (np.abs(grid) >= 0.5) & (np.abs(grid) < 2.0), "cage": np.abs(grid) >= 2.0}

    def late(r, young=False):
        if young:
            return r["sv_dg_young"][-1, 1] - r["sv_dg_young"][ih, 1], r["sv_dg_young"][-1, 0] - r["sv_dg_young"][ih, 0]
        return r["sv_M"][-1] - r["sv_M"][ih], r["sv_C"][-1] - r["sv_C"][ih]

    per = {"young": [], "old": [], "all": [], "abf": []}
    for r in fr:
        My, Cy = late(r, True)
        Ma, Ca = late(r)
        per["young"].append((My, Cy))
        per["old"].append((Ma - My, Ca - Cy))
        per["all"].append((Ma, Ca))
    for r in ab:
        per["abf"].append(late(r))
    out = dict(block=name, late_from_t=float(fr[0]["times"][ih]))
    # young share of deposits
    Cy_tot = sum(r["dg_young"][0] for r in fr)
    C_tot = sum(r["sv_C"][-1] for r in fr)
    out["young_share_whole_run"] = {k: float(Cy_tot[m].sum() / C_tot[m].sum()) for k, m in reg.items()}
    Cyl = sum(p[1] for p in per["young"])
    Cal = sum(p[1] for p in per["all"])
    out["young_share_late_half"] = {k: float(Cyl[m].sum() / Cal[m].sum()) for k, m in reg.items()}

    def pooled(lst, drop=None):
        idx = [i for i in range(len(lst)) if i != drop]
        M = sum(lst[i][0] for i in idx)
        C = sum(lst[i][1] for i in idx)
        return M, C

    def stats(key):
        M, C = pooled(per[key])
        G = mt.mean_force(M, C)
        e_g = wrms(G - Gr, C)
        e_f = float(mt.centred_rms(mt.histogram_pmf(G) - Fr)) if (C > 0).all() else float("nan")
        jk = []
        for d in range(16):
            Mj, Cj = pooled(per[key], d)
            jk.append(wrms(mt.mean_force(Mj, Cj) - Gr, Cj))
        se = float(np.sqrt(15 / 16 * ((np.array(jk) - np.mean(jk)) ** 2).sum()))
        reg_err = {k: float(np.average((G - Gr)[m], weights=np.maximum(C[m], 1e-12))) for k, m in reg.items()}
        return dict(meanforce_wrms_vs_ref=e_g, jackknife_se=se, free_energy_rms_vs_ref=e_f,
                    signed_mean_force_error_by_region=reg_err, deposits=float(C.sum()))
    out["late_half"] = {k: stats(k) for k in per}
    # eta conditional by age class (whole run, coarse 18 bins) vs reference
    cc = ref["counts"].reshape(18, -1)
    re_eta = (ref["eta"].reshape(18, -1) * cc).sum(1) / cc.sum(1)
    Cy = sum(r["dg_young"][0] for r in fr).reshape(18, -1).sum(1)
    Ey = sum(r["dg_young"][2] for r in fr).reshape(18, -1).sum(1)
    Ct = sum(r["sv_C"][-1] for r in fr).reshape(18, -1).sum(1)
    Et = sum(r["Seta"] for r in fr).reshape(18, -1).sum(1)
    Ca_ = sum(r["sv_C"][-1] for r in ab).reshape(18, -1).sum(1)
    Ea_ = sum(r["Seta"] for r in ab).reshape(18, -1).sum(1)
    eta_y = np.where(Cy > 0, Ey / np.maximum(Cy, 1), np.nan)
    eta_o = (Et - Ey) / np.maximum(Ct - Cy, 1)
    eta_a = Ea_ / np.maximum(Ca_, 1)
    out["eta_conditional_minus_ref"] = dict(young=(eta_y - re_eta).tolist(), old=(eta_o - re_eta).tolist(), abf=(eta_a - re_eta).tolist(),
                                            rms_young=float(np.sqrt(np.nanmean((eta_y - re_eta) ** 2))),
                                            rms_old=float(np.sqrt(np.mean((eta_o - re_eta) ** 2))),
                                            rms_abf=float(np.sqrt(np.mean((eta_a - re_eta) ** 2))))
    # u_x^2 conditional (transverse displacement, the other fibre coordinate the engine accumulates), same construction;
    # reported relative to the reference value per coarse bin
    re_u = (ref["ux2"].reshape(18, -1) * cc).sum(1) / cc.sum(1)
    Uy = sum(r["dg_young"][3] for r in fr).reshape(18, -1).sum(1)
    Ut = sum(r["Sux2"] for r in fr).reshape(18, -1).sum(1)
    Ua_ = sum(r["Sux2"] for r in ab).reshape(18, -1).sum(1)
    u_y = np.where(Cy > 0, Uy / np.maximum(Cy, 1), np.nan)
    u_o = (Ut - Uy) / np.maximum(Ct - Cy, 1)
    u_a = Ua_ / np.maximum(Ca_, 1)
    rel = lambda u: (u - re_u) / re_u
    out["ux2_conditional_relative_to_ref"] = dict(young=rel(u_y).tolist(), old=rel(u_o).tolist(), abf=rel(u_a).tolist(),
                                                  rms_young=float(np.sqrt(np.nanmean(rel(u_y) ** 2))),
                                                  rms_old=float(np.sqrt(np.mean(rel(u_o) ** 2))),
                                                  rms_abf=float(np.sqrt(np.mean(rel(u_a) ** 2))))
    return out


def main():
    res = {}
    for n in BLOCKS:
        o = block(n)
        res[n] = o
        print(f"== {n} (late half from t = {o['late_from_t']:g})")
        print("   young share of deposits (whole run):", {k: round(v, 3) for k, v in o["young_share_whole_run"].items()},
              " late half:", {k: round(v, 3) for k, v in o["young_share_late_half"].items()})
        for k, v in o["late_half"].items():
            print(f"   late-half {k:5s}: mean-force wRMS vs ref {v['meanforce_wrms_vs_ref']:.4f} +- {v['jackknife_se']:.4f} | F RMS vs ref "
                  f"{v['free_energy_rms_vs_ref']:.4f} | signed error W/flank/cage " +
                  " ".join(f"{x:+.3f}" for x in v["signed_mean_force_error_by_region"].values()))
        e = o["eta_conditional_minus_ref"]
        print(f"   eta conditional RMS vs ref: young {e['rms_young']:.4f}  old {e['rms_old']:.4f}  ABF {e['rms_abf']:.4f}")
        u = o["ux2_conditional_relative_to_ref"]
        print(f"   u_x^2 conditional relative RMS vs ref: young {u['rms_young']:.4f}  old {u['rms_old']:.4f}  ABF {u['rms_abf']:.4f}")
    p = os.path.join(S.ROOT, "results", "adaptive_fr_campaign", "mechanism", "clone_age.json")
    json.dump(res, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
