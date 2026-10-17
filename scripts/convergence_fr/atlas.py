"""Experiment 1: convergence and time-to-accuracy atlas of EXISTING paired ABF vs ABF+FR runs (DESCRIPTIVE).

    ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao python scripts/convergence_fr/atlas.py
        -> results/convergence_fr_campaign/atlas/atlas_summary.json, figures/convergence_fr/*.pdf|png

Definitions: scripts/convergence_fr/atlas_lib.py (frozen in PREREGISTRATION.md section 2 before this ran).
Tolerances are in units of kBT so that the four systems share one ladder: eps/kBT in EPS_LADDER
(0.048 / 0.040 / 0.032 kBT = 0.12 / 0.10 / 0.08 kJ/mol at 300 K).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "eri_campaign"))
import atlas_lib as L  # noqa: E402
import systems as S  # noqa: E402

EPS_LADDER = [0.08, 0.064, 0.048, 0.040, 0.032]
EPS_DENSE = np.geomspace(0.016, 0.32, 41)
ORDER = ["LTA150", "LTA300", "ITE300", "ITE150"]
OUT = os.path.join(S.ROOT, "results", "convergence_fr_campaign", "atlas")
FIG = os.path.join(S.ROOT, "figures", "convergence_fr")


def analyse(key, d):
    t, T = d["t"], d["T"]
    A, B = d["arms"]["abf"], d["arms"]["fr"]
    res = dict(key=key, label=S.SYSTEMS[key]["label"], model=S.SYSTEMS[key]["model"], status=S.SYSTEMS[key]["status"],
               K=d["K"], T=T, dt=d["dt"], bins=d["bins"], run_bins=d["run_bins"], T_K=d["T_K"], kT=d["kT"],
               t_fr=d["t_fr"], t_ramp=d["t_ramp"], rate=d["rate"], n_seeds=len(d["seeds"]), seeds=d["seeds"],
               save_spacing=float(np.median(np.diff(t))))
    tr = d["t_fr"]
    ib_a, ib_b = L.running_ibar(A["eF"], t), L.running_ibar(B["eF"], t)
    ip_a, ip_b = L.running_ibar(A["eF"], t, tr), L.running_ibar(B["eF"], t, tr)
    end = {}
    end["Ibar_F_full"] = L.paired_rel(ib_a[-1], ib_b[-1])
    end["Ibar_F_post_ramp"] = L.paired_rel(ip_a[-1], ip_b[-1])
    end["eF_T"] = L.paired_rel(A["eF"][-1], B["eF"][-1])
    end["eG_T"] = L.paired_rel(A["eG"][-1], B["eG"][-1])
    sel = t >= tr
    end["Ibar_G_post_ramp"] = L.paired_rel(L.running_ibar(A["eG"], t, tr)[-1], L.running_ibar(B["eG"], t, tr)[-1])
    res["endpoints"] = end
    # running relative advantage curves (per seed first, then the median over seeds)
    rel_full = ib_b / ib_a - 1
    rel_post = ip_b / ip_a - 1
    curves = dict(t=t.tolist())
    for name, X in (("eF_abf", A["eF"]), ("eF_fr", B["eF"]), ("eG_abf", A["eG"]), ("eG_fr", B["eG"]),
                    ("ibar_abf", ib_a), ("ibar_fr", ib_b), ("rel_ibar_full", rel_full), ("rel_ibar_post", rel_post),
                    ("rel_eF", B["eF"] / A["eF"] - 1), ("win_abf", A["win"]), ("win_fr", B["win"])):
        med, lo, hi = L.boot_median_curve(X)
        curves[name] = dict(median=med.tolist(), ci_lo=lo.tolist(), ci_hi=hi.tolist(),
                            q25=np.nanquantile(X, 0.25, axis=1).tolist(), q75=np.nanquantile(X, 0.75, axis=1).tolist())
    # time to accuracy
    tta = {}
    for kind in ("first", "sustained", "windowed"):
        tta[kind] = {}
        for ek in EPS_LADDER:
            eps = ek * d["kT"]
            ta, tb = L.tau(A["eF"], t, eps, kind), L.tau(B["eF"], t, eps, kind)
            st = L.speedup_stats(ta, tb, T)
            st["eps_kJmol"] = eps
            tta[kind][f"{ek:.3f}kT"] = st
    res["time_to_accuracy"] = tta
    dense = {}
    for kind in ("first", "sustained"):
        rows = []
        for ek in EPS_DENSE:
            eps = ek * d["kT"]
            st = L.speedup_stats(L.tau(A["eF"], t, eps, kind), L.tau(B["eF"], t, eps, kind), T, n=2000)
            rows.append(dict(eps_kT=float(ek), eps=float(eps), S=st["S_median"], lo=st["S_ci95"][0], hi=st["S_ci95"][1],
                             cens_abf=st["censored_abf"], cens_fr=st["censored_fr"], fr_faster=st["fr_faster"],
                             abf_faster=st["abf_faster"]))
        dense[kind] = rows
    res["speedup_dense"] = dense
    res["crossing_eF"] = L.crossing(A["eF"], B["eF"], t, t_min=tr)
    res["crossing_ibar_full"] = L.crossing(ib_a[1:], ib_b[1:], t[1:], t_min=tr)
    ea, eb = L.establishment_time(A["win"], t, d["win_share"]), L.establishment_time(B["win"], t, d["win_share"])
    res["establishment"] = dict(rule="first save at which the trailing 1-t.u. mean window occupancy >= 0.75 x the "
                                     "flat-bias window share", win_share=d["win_share"],
                                abf=[float(x) for x in ea], fr=[float(x) for x in eb],
                                abf_median=float(np.median(ea)), fr_median=float(np.median(eb)),
                                paired_difference=L.boot_median(eb - ea) if np.isfinite(ea).all() and np.isfinite(eb).all() else None)
    res["curves"] = curves
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    allres = {}
    for key in list(S.SYSTEMS):
        print("loading", key, flush=True)
        d = S.SYSTEMS[key]["loader"]()
        r = analyse(key, d)
        r["F_ref"] = d["F_ref"].tolist()
        allres[key] = r
        e = r["endpoints"]
        print(f"{key}: Ibar_F full {e['Ibar_F_full']['median_rel']:+.3f} {e['Ibar_F_full']['ci95']} "
              f"{e['Ibar_F_full']['wins_fr']}/{e['Ibar_F_full']['n']} | post {e['Ibar_F_post_ramp']['median_rel']:+.3f} "
              f"| eF_T {e['eF_T']['median_rel']:+.3f} {e['eF_T']['ci95']}", flush=True)
    # ITE 150 K at the common early horizon T = 50 (truncation is exact: the dynamics never depend on n_steps)
    for key, rb in (("ITE150_T50", None), ("ITE150c_T50", 180)):
        d = S.load_eri_cell("results/overnight/ite150/prod_confirm", 256, "results/overnight/ite150/reference",
                            "ite_itq3", read_bins=rb, t_max=50.0)
        S.SYSTEMS[key] = dict(label=f"ITE 150 K, first 50 t.u.{' (180-bin read-out)' if rb else ''}",
                              model=S.SYSTEMS["ITE150"]["model"], status="X2 truncated at the common horizon T = 50")
        allres[key] = analyse(key, d)
        e = allres[key]["endpoints"]
        print(f"{key}: Ibar_F full {e['Ibar_F_full']['median_rel']:+.3f} {e['Ibar_F_full']['ci95']} | eF_T "
              f"{e['eF_T']['median_rel']:+.3f} {e['eF_T']['ci95']}", flush=True)
    json.dump(dict(eps_ladder_kT=EPS_LADDER, systems=allres), open(os.path.join(OUT, "atlas_summary.json"), "w"),
              indent=1, default=float)
    print("wrote", os.path.join(OUT, "atlas_summary.json"))


if __name__ == "__main__":
    main()
