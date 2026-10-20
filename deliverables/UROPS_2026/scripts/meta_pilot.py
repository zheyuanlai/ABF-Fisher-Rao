#!/usr/bin/env python
"""Stages 2 and 3 of deliverables/UROPS_2026/metastability/PREREGISTRATION.md on the 8 pilot seeds 900000-900007.

  --stage resolution : ABF only, nb in {60, 120, 240}; rule = smallest median I_F, ties within 3 % broken by the
                       smaller median final e_F  -> FROZEN.json['resolution']
  --stage rate       : FR at the frozen h and nb on the rate ladder; SAFETY ONLY (windowed ESS / family share,
                       events per opportunity, finiteness).  No error is computed and no accumulator is kept for the
                       FR pilot runs.  gamma = the rung below the largest all-seeds-safe rung -> FROZEN.json['fr_rate']
"""
import argparse
import datetime
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

C.setup_paths()
import meta_engine as E  # noqa: E402

PILOT_SEEDS = list(range(900000, 900008))
NB_CANDIDATES = [60, 120, 240]
RATES = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
FLOORS = dict(ess_win_min=0.30, fam_win_max=0.05, event_frac_max=0.05)       # analyze_fr_start_timing.FLOORS
FROZEN = os.path.join(C.PKG, "metastability", "FROZEN.json")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def abf_job(job):
    h, nb, seed, out = job
    res = E.run_arm(dict(h=h, nb=nb), "abf", seed)
    E.save_npz(os.path.join(out, f"abf_nb{nb}_s{seed}.npz"), res)
    sc = E.Scorer(nb, ref=_ref())
    eF, eFp = sc.errors(res["M_all"], res["C_all"])
    t = res["save_t"]
    return dict(nb=nb, seed=seed, I_F=float(np.trapezoid(eF, t)), final_eF=float(eF[-1]), final_eFp=float(eFp[-1]),
                floor_eFp=sc.floor_eFp, wall_s=float(res["wall_s"]))


def fr_job(job):
    h, nb, gamma, seed = job
    res = E.run_arm(dict(h=h, nb=nb, gamma=gamma), "fr", seed)
    ok = bool(np.all(np.isfinite(res["X_final"])) and np.all(np.isfinite(res["Y_final"])))
    return dict(gamma=gamma, seed=seed, min_ess_win=float(res["min_ess_win"]), max_fam_win=float(res["max_fam_win"]),
                max_event_fraction=float(res["max_event_fraction_realised"]), n_cap_binding=int(res["n_cap_binding"]),
                deaths=int(res["deaths_cum"][-1]), finite=ok, wall_s=float(res["wall_s"]))


_R = {}


def _ref():
    if "r" not in _R:
        _R["r"] = E.reference(4.0, nx=6001, ny=2001)
    return _R["r"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("resolution", "rate"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    fz = json.load(open(FROZEN))
    h = float(fz["timestep"]["h"])
    os.makedirs(a.out, exist_ok=True)
    if a.stage == "resolution":
        assert "resolution" not in fz, "resolution already frozen"
        jobs = [(h, nb, s, a.out) for nb in NB_CANDIDATES for s in PILOT_SEEDS]
        with mp.get_context("spawn").Pool(a.procs) as pool:
            rows = pool.map(abf_job, jobs, chunksize=1)
        summ = {}
        for nb in NB_CANDIDATES:
            r = [x for x in rows if x["nb"] == nb]
            summ[nb] = dict(I_F_median=float(np.median([x["I_F"] for x in r])),
                            final_eF_median=float(np.median([x["final_eF"] for x in r])),
                            final_eFp_median=float(np.median([x["final_eFp"] for x in r])), floor_eFp=r[0]["floor_eFp"])
            print(f"nb {nb}: median I_F {summ[nb]['I_F_median']:.4f}, final e_F {summ[nb]['final_eF_median']:.5f}, "
                  f"final e_F' {summ[nb]['final_eFp_median']:.4f} (floor {summ[nb]['floor_eFp']:.4f})", flush=True)
        best = min(NB_CANDIDATES, key=lambda nb: summ[nb]["I_F_median"])
        ties = [nb for nb in NB_CANDIDATES if summ[nb]["I_F_median"] <= 1.03 * summ[best]["I_F_median"]]
        chosen = min(ties, key=lambda nb: summ[nb]["final_eF_median"])
        fz = json.load(open(FROZEN))
        fz["resolution"] = dict(nb=chosen, ties=ties, best_I_F=best, summary={str(k): v for k, v in summ.items()},
                                rows=rows, pilot_seeds=PILOT_SEEDS, frozen_utc=now())
        json.dump(fz, open(FROZEN, "w"), indent=1)
        print("selected nb:", chosen, "ties:", ties)
    else:
        assert "resolution" in fz and "fr_rate" not in fz
        nb = int(fz["resolution"]["nb"])
        jobs = [(h, nb, g, s) for g in RATES for s in PILOT_SEEDS]
        with mp.get_context("spawn").Pool(a.procs) as pool:
            rows = pool.map(fr_job, jobs, chunksize=1)
        safe = {}
        for g in RATES:
            r = [x for x in rows if x["gamma"] == g]
            safe[g] = bool(all(x["finite"] and x["min_ess_win"] >= FLOORS["ess_win_min"]
                               and x["max_fam_win"] <= FLOORS["fam_win_max"]
                               and x["max_event_fraction"] < FLOORS["event_frac_max"] for x in r))
            print(f"gamma {g}: safe {safe[g]} | min ESS_win {min(x['min_ess_win'] for x in r):.3f} | max fam_win "
                  f"{max(x['max_fam_win'] for x in r):.4f} | max events/opp {max(x['max_event_fraction'] for x in r):.3f} | "
                  f"deaths median {np.median([x['deaths'] for x in r]):.0f}", flush=True)
        ok = [g for g in RATES if safe[g]]
        if not ok:
            gamma = None
        else:
            g_safe = max(ok)
            i = RATES.index(g_safe)
            gamma = RATES[i - 1] if i > 0 else RATES[0]
        json.dump(rows, open(os.path.join(a.out, "fr_safety_rows.json"), "w"), indent=1)
        fz = json.load(open(FROZEN))
        fz["fr_rate"] = dict(gamma=gamma, gamma_safe=(max(ok) if ok else None), safe={str(k): v for k, v in safe.items()},
                             floors=FLOORS, ladder=RATES, rows=rows, pilot_seeds=PILOT_SEEDS, frozen_utc=now(),
                             note="safety-only selection; no accuracy read-out of any FR pilot run")
        json.dump(fz, open(FROZEN, "w"), indent=1)
        print("gamma_safe:", max(ok) if ok else None, "-> production gamma:", gamma)


if __name__ == "__main__":
    main()
