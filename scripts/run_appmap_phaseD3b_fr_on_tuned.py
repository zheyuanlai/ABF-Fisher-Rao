"""Application-map Phase D3b: the decisive test — tuned SHUS vs tuned SHUS + FR.

Design frozen in docs/PREREGISTRATION_APPLICATION_MAP.md (commit fde0c9c): fresh
seeds 500-515 on t_mid; g* and the FR window are read from the completed D3a
summary (frozen derivation rules). Arms: shus_g1 anchor, shus_gstar baseline,
gstar+fr_temp (theta 0.01, stride 10, transferred), gstar+count9, gstar+sham.

Frozen decision rule on d = paired dI_F of gstar+FR vs gstar:
  d >= -2% or CI straddles 0  ->  FR adds nothing on top of a tuned base ABP
                                  on t_mid (question CLOSED for this cell);
  d <= -5% with CI < 0 AND beats its sham -> FR has independent value (Outcome 1);
  otherwise: point estimate, no headline.

Usage: python scripts/run_appmap_phaseD3b_fr_on_tuned.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from abpfr.io import save_run
from abpfr.metrics import paired_bootstrap_ci
from abpfr.systems import torus2d as t2
from abpfr.systems.gateway import Method

CELL = dict(beta=4.0, H1=1.5, H2=1.5, Hc=0.5)          # t_mid, frozen
COMMON = dict(K=1024, dt=1e-3, n_steps=200_000, block=20, eps_bw=0.06,
              eta_bw=0.25, n_saves=400, profile_every=8, ess_window_steps=4000)
SEEDS = list(range(500, 516))                          # 16 fresh seeds, frozen
THETA, STRIDE = 0.01, 10                               # transferred, unretuned
BATCH_SEED = 20260831
D3A_SUMMARY = "results/appmap_phaseD3_gain/summary.json"
OUT = "results/appmap_phaseD3b_fr_on_tuned"


def build_arms(T, g_star, t_on, t_off):
    fr_kw = dict(use_fr=True, theta=THETA, t_on_frac=t_on / T,
                 t_off_frac=t_off / T, fr_every_blocks=STRIDE, g_shus=g_star)
    return [
        Method("shus_g1"),
        Method("shus_gstar", g_shus=g_star),
        Method("gstar_fr", **fr_kw),
        Method("gstar_count9", **fr_kw, coarse_bins=9),
        Method("gstar_sham", use_fr=True, sham=True, shadows="gstar_fr",
               g_shus=g_star),
    ]


def main():
    with open(D3A_SUMMARY) as f:
        d3a = json.load(f)
    g_star = float(d3a["g_star"])
    t_on, t_off = (float(x) for x in d3a["d3b_window"])
    device = t2.DEVICE
    cfgs = [t2.Torus2DConfig(**CELL, **COMMON) for _ in SEEDS]
    T = cfgs[0].T_total
    arms = build_arms(T, g_star, t_on, t_off)
    print(f"Phase D3b: t_mid, {len(SEEDS)} seeds x {len(arms)} arms, "
          f"g*={g_star:g}, window [{t_on:g},{t_off:g}], theta={THETA}, "
          f"stride={STRIDE}")
    t0 = time.time()
    recs = t2.simulate_batch(cfgs, SEEDS, arms, batch_seed=BATCH_SEED,
                             device=device, progress=50_000)
    print(f"wall {time.time()-t0:.0f}s")

    os.makedirs(OUT, exist_ok=True)
    for rec in recs:
        arrays = {k: rec[k] for k in
                  ("time", "profile_time", "pmf_t", "marginal_t", "x1_grid",
                   "x2_grid", "F_ref", "l2_f_t", "kl_u_t", "tv_u_t", "ess_anc_t",
                   "wmax_t", "ess_anc_glob_t", "wmax_glob_t", "n_anc_t",
                   "dep_ref_l2_t", "dep_self_l2_t", "P_regions", "event_time",
                   "event_theta", "event_ess_fr", "event_turnover")}
        meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
                "config": rec["config"], "method": rec["method"], "seed": rec["seed"],
                "cell": "t_mid", "stage": "appmap_phaseD3b_fr_on_tuned"}
        save_run(os.path.join(OUT, f"{rec['method']['name']}_seed{rec['seed']}"),
                 arrays, meta)

    by = {}
    for rec in recs:
        by.setdefault(rec["method"]["name"], {})[rec["seed"]] = rec
    base = by["shus_gstar"]
    K = COMMON["K"]

    summary = {"cell": CELL, "seeds": SEEDS, "batch_seed": BATCH_SEED,
               "g_star": g_star, "window": [t_on, t_off], "theta": THETA,
               "stride": STRIDE, "arms": {}}
    print(f"\n{'arm':<13s} {'dI_F% vs gstar [CI]':>24s} {'eT ratio':>9s} "
          f"{'minESS':>7s} {'n_anc':>6s} {'turnover':>9s}")
    for name in ("shus_g1", "gstar_fr", "gstar_count9", "gstar_sham"):
        rows = by[name]
        dIF = np.array([(rows[sd]["int_l2_f"] - base[sd]["int_l2_f"])
                        / base[sd]["int_l2_f"] for sd in SEEDS])
        m, lo, hi = paired_bootstrap_ci(dIF)
        eT = float(np.median([rows[sd]["final_l2_f"] / base[sd]["final_l2_f"]
                              for sd in SEEDS]))
        ess = float(np.median([rows[sd]["ess_anc_t"].min() / K for sd in SEEDS]))
        nanc = float(np.median([rows[sd]["n_anc_t"][-1] / K for sd in SEEDS]))
        turn = float(np.median([rows[sd]["total_turnover"] for sd in SEEDS]))
        print(f"{name:<13s} {100*m:>8.1f} [{100*lo:>6.1f},{100*hi:>6.1f}] "
              f"{eT:>9.3f} {ess:>7.2f} {nanc:>6.2f} {turn:>9.0f}")
        summary["arms"][name] = dict(dIF=[float(x) for x in dIF],
                                     dIF_ci=[m, lo, hi], eT_ratio=eT,
                                     min_ess=ess, n_anc=nanc, turnover=turn)

    m, lo, hi = summary["arms"]["gstar_fr"]["dIF_ci"]
    m_sham = summary["arms"]["gstar_sham"]["dIF_ci"][0]
    if m >= -0.02 or (lo < 0 < hi):
        verdict = ("FR adds nothing on top of a tuned base ABP on t_mid; "
                   "the SHUS-FR question is CLOSED for this cell")
    elif m <= -0.05 and hi < 0 and m < m_sham:
        verdict = ("FR has independent value on top of tuning (Outcome 1)")
    else:
        verdict = "no headline claim (frozen rule): point estimate recorded"
    summary["verdict"] = verdict
    print(f"\ngstar+FR vs gstar: {100*m:.1f}% [{100*lo:.1f},{100*hi:.1f}]")
    print(f"VERDICT: {verdict}")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"summary -> {OUT}/summary.json")


if __name__ == "__main__":
    main()
