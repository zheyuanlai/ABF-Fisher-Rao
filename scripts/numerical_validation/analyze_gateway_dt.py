#!/usr/bin/env python
"""Analysis of configs/numerical_validation/gateway_dt_prereg.json: the accepted gateway FR effect vs time step.

    python scripts/numerical_validation/analyze_gateway_dt.py

Scorer, Ibar_F definition and bootstrap are the gateway replica ladder's (scripts/analyze_gateway_replica_ladder.py).
"""
import glob
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze_gateway_replica_ladder as G  # noqa: E402

OUT = os.path.join(ROOT, "results", "numerical_validation", "gateway_dt")


def em_floor(cell):
    """Closed-form RMS (eval window, centred) of F_EM - F_analytic on a fine grid."""
    import eb_abffr_core as eb
    x = np.linspace(eb.EVAL_LO, eb.EVAL_HI, 20001)
    b, H, oo, oi, s, dt = (cell[k] for k in ("beta", "H", "omega_out", "omega_in", "s", "dt"))
    e = np.exp(-x * x / (2 * s * s)); om = oo + (oi - oo) * e; dom = -(oi - oo) * (x / (s * s)) * e
    d = dom / (om * b) * (1.0 / (1.0 - om * om * dt / 2.0) - 1.0)          # mean-force difference
    D = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(x))])
    D -= D.mean()
    return float(np.sqrt(np.mean(D ** 2)))


def main():
    S = dict(prereg="configs/numerical_validation/gateway_dt_prereg.json", per_dt={})
    for ddir in sorted(glob.glob(os.path.join(OUT, "dt*")), key=lambda p: -float(os.path.basename(p)[2:])):
        files = sorted(glob.glob(os.path.join(ddir, "s*.npz")))
        files = [f for f in files if ".tmp." not in f]
        if not files:
            continue
        meta0 = json.loads(str(np.load(files[0])["meta_json"]))
        cell = meta0["cell"]
        res_dt = dict(n_seeds=len(files), em_floor_vs_analytic=em_floor(cell), dt=cell["dt"])
        for ref in ("analytic", "em"):
            sc = G.Scorer(cell, ref)
            recs = {}
            for f in files:
                z = np.load(f)
                meta = json.loads(str(z["meta_json"]))
                u = z["u"]
                lin = np.isclose(u * 200, np.round(u * 200)) & (u > 0)
                for a, name in enumerate(meta["arms"]):
                    nb = 180 if name.endswith("180") else 45
                    eF, eFp = sc.score(z["M"][a, :, :nb], z["C"][a, :, :nb])
                    recs[(meta["seed"], name)] = dict(Ibar=float(np.mean(eF[lin])), fin=float(eF[-1]), finp=float(eFp[-1]),
                                                       events=float(z["die"][a, -1] + z["clone"][a, -1]))
            seeds = sorted({s for s, _ in recs})
            out = {}
            for nb in (180, 45):
                ab, fr = f"abf_h{nb}", f"fr_h{nb}"
                dI = [100 * (recs[(s, fr)]["Ibar"] / recs[(s, ab)]["Ibar"] - 1) for s in seeds]
                dF = [100 * (recs[(s, fr)]["fin"] / recs[(s, ab)]["fin"] - 1) for s in seeds]
                out[str(nb)] = dict(abf_Ibar=float(np.median([recs[(s, ab)]["Ibar"] for s in seeds])),
                                    fr_Ibar=float(np.median([recs[(s, fr)]["Ibar"] for s in seeds])),
                                    abf_fin=float(np.median([recs[(s, ab)]["fin"] for s in seeds])),
                                    fr_fin=float(np.median([recs[(s, fr)]["fin"] for s in seeds])),
                                    dIbar=G.boot_median(dI), dfin=G.boot_median(dF),
                                    wins_Ibar=int(np.sum(np.array(dI) < 0)), wins_fin=int(np.sum(np.array(dF) < 0)),
                                    fr_events=float(np.median([recs[(s, fr)]["events"] for s in seeds])), n=len(seeds))
            res_dt[ref] = out
        S["per_dt"][f"{cell['dt']:g}"] = res_dt
        a = res_dt["analytic"]["180"]
        res_dt["admissible"] = bool(res_dt["em_floor_vs_analytic"] <= a["abf_fin"] / 3.0)
        for ref in ("analytic", "em"):
            for nb in ("180", "45"):
                r = res_dt[ref][nb]
                print(f"dt {cell['dt']:g} ref {ref:8s} {nb:>3s} bins: ABF Ibar {r['abf_Ibar']:.5f} fin {r['abf_fin']:.5f} | FR Ibar {r['fr_Ibar']:.5f} "
                      f"fin {r['fr_fin']:.5f} | dIbar {r['dIbar'][0]:+.1f}% [{r['dIbar'][1]:+.1f},{r['dIbar'][2]:+.1f}] ({r['wins_Ibar']}/{r['n']}) "
                      f"dfin {r['dfin'][0]:+.1f}% [{r['dfin'][1]:+.1f},{r['dfin'][2]:+.1f}] ({r['wins_fin']}/{r['n']}) events {r['fr_events']:.0f}", flush=True)
        print(f"dt {cell['dt']:g}: EM floor vs analytic {res_dt['em_floor_vs_analytic']:.5f}; admissible (<= ABF final/3): {res_dt['admissible']}", flush=True)
    val = [k for k, v in S["per_dt"].items() if v["admissible"]]
    cis = {k: S["per_dt"][k]["analytic"]["180"]["dIbar"] for k in val}
    if val and all(c[2] < 0 for c in cis.values()):
        verdict = "SURVIVES"
    elif any(c[1] > 0 for c in cis.values()):
        verdict = "REVERSED"
    elif val:
        verdict = "NOT_ESTABLISHED"
    else:
        verdict = "NO_VALIDATED_DT"
    S["verdict"] = verdict
    S["validated_dts"] = val
    print("verdict:", verdict, "validated dts:", val)
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
