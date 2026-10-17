"""E1x (EXPLORATORY, Amendment A3): t_stop ladder {5, 20} t.u. on X1 seeds 980000-980003, each paired with the X1
ABF arm of the same seed; the X1 continuous-FR arm of the same seed is shown as the t_stop = 50 (never) end point.
Pairing is by cfg seed.  Checks: every arm bitwise equal to ABF before fr_start; every stopped arm bitwise equal to
the continuous arm at every save before its stop.

    ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao python scripts/convergence_fr/ladder_e1x.py
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

LADDER_KT = [0.080, 0.064, 0.048, 0.040, 0.032]


def main():
    mt.set_reference_base(S.dpath("results/overnight/ite/reference"))
    Fr, Gr = mt.reference_for_grid(180, "ite_itq3")
    seeds = list(range(980000, 980004))
    runs = {}
    for i, s in enumerate(seeds):
        runs[s] = dict(abf=mt.load_run(S.dpath(f"results/overnight/ite/prod_confirm/abf_N1024_s{i}.npz")),
                       cont=mt.load_run(S.dpath(f"results/overnight/ite/prod_confirm/fr_N1024_s{i}.npz")),
                       stop5=mt.load_run(S.dpath(f"results/convergence_fr_campaign/E1x_ite300_ladder/fr_stop5_N1024_s{s}.npz")),
                       stop20=mt.load_run(S.dpath(f"results/convergence_fr_campaign/E1x_ite300_ladder/fr_stop20_N1024_s{s}.npz")))
        for k, r in runs[s].items():
            assert int(r["cfg"]["seed"]) == s, (k, s)
        a = runs[s]["abf"]
        k0 = int(a["cfg"]["fr_start_steps"])
        for k in ("cont", "stop5", "stop20"):
            pre = a["save_steps"] < k0
            assert np.array_equal(a["sv_C"][pre], runs[s][k]["sv_C"][pre]), ("pre-FR pairing", k, s)
        for k, kst in (("stop5", 50000), ("stop20", 200000)):
            pre = runs[s]["cont"]["save_steps"] < kst
            assert np.array_equal(runs[s]["cont"]["sv_C"][pre], runs[s][k]["sv_C"][pre]), ("pre-stop identity", k, s)
            assert np.array_equal(runs[s]["cont"]["sv_M"][pre], runs[s][k]["sv_M"][pre]), ("pre-stop identity", k, s)
    t = runs[seeds[0]]["abf"]["times"]
    T = float(t[-1])
    kT = S.KB * 300.0
    E = {k: np.array([mt.errors(runs[s][k], Fr, Gr)["eF"] for s in seeds]).T for k in ("abf", "stop5", "cont", "stop20")}
    out = dict(status="EXPLORATORY (A3): 4 X1 seeds; never pooled with E1/E2", seeds=seeds,
               checks="pre-FR bitwise pairing with ABF and pre-stop bitwise identity with continuous FR: all pass",
               schedules={})
    for k, lab in (("stop5", "t_stop 5"), ("stop20", "t_stop 20"), ("cont", "continuous (X1)")):
        logS = np.zeros(len(seeds))
        for ek in LADDER_KT:
            st = L.speedup_stats(L.tau(E["abf"], t, ek * kT, "sustained"), L.tau(E[k], t, ek * kT, "sustained"), T, n=2000)
            logS += np.log(np.array(st["per_seed_S"])) / len(LADDER_KT)
        ipa, ipb = L.running_ibar(E["abf"], t, 1.0)[-1], L.running_ibar(E[k], t, 1.0)[-1]
        out["schedules"][lab] = dict(S_geo_per_seed=np.exp(logS).tolist(), S_geo_median=float(np.exp(np.median(logS))),
                                     dIbar_post_ramp_per_seed=(ipb / ipa - 1).tolist(),
                                     deF_T_per_seed=(E[k][-1] / E["abf"][-1] - 1).tolist())
        r = out["schedules"][lab]
        print(f"{lab:16s} S_geo per seed {np.round(r['S_geo_per_seed'], 2)} (median {r['S_geo_median']:.2f}) | "
              f"dIbar_post {np.round(r['dIbar_post_ramp_per_seed'], 3)} | de_F(T) {np.round(r['deF_T_per_seed'], 3)}")
    p = os.path.join(S.ROOT, "results", "convergence_fr_campaign", "analysis", "E1x_ladder.json")
    json.dump(out, open(p, "w"), indent=1)
    np.savez(p.replace(".json", "_curves.npz"), t=t, **{f"eF_{k}": v for k, v in E.items()})
    print("wrote", p)


if __name__ == "__main__":
    main()
