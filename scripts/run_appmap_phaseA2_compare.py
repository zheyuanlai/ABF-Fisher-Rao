"""Application-map Phase A2: best gain-tuned SHUS vs the frozen FR winner (fresh seeds).

Design frozen in docs/PREREGISTRATION_APPLICATION_MAP.md (commit ae1a998): seeds
300..315, arms shus(g=1) / shus_gbest / fr_temp / count / sham, with fr_temp the
UNRETUNED Stage-3 winner (theta=0.01, stride 10 blocks, window [6,14]) and g_best
read from the Phase-A1 summary. If g_best = 1.0 the phase is unnecessary (Q1 is
answered from A1 directly) and this script refuses to run.

Frozen interpretation of the direct contrast d = (I_F^fr - I_F^gbest)/I_F^gbest:
median d >= 0  -> "FR is not practically necessary on the gateway (adaptation-rate
compensation)"; median d <= -5% with CI < 0 -> "the population correction supplies
something adaptation-rate tuning does not reproduce"; otherwise record without a
headline claim.

Usage: python scripts/run_appmap_phaseA2_compare.py [--gbest G]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from abpfr.io import save_run
from abpfr.metrics import paired_bootstrap_ci, time_to_accuracy
from abpfr.systems import gateway as gw

CELL = dict(beta=16.0, H=0.5, s=0.10, r=32.0)          # anchor_D, frozen
COMMON = dict(K=1024, dt=2e-4, n_steps=500_000, block=20, n_saves=400,
              ess_window_steps=4000)
SEEDS = list(range(300, 316))                          # 16 fresh matched seeds
T_ON, T_OFF, THETA, STRIDE = 6.0, 14.0, 0.01, 10       # frozen Stage-3 winner
E0 = 0.236                                             # frozen ladder base
BATCH_SEED = 20260826
A1_SUMMARY = "results/appmap_phaseA_gain/summary.json"
OUT = "results/appmap_phaseA2_compare"


def build_arms(T, g_best):
    gbest = gw.Method(f"shus_gbest", g_shus=g_best)
    fr = gw.Method("fr_temp", use_fr=True, theta=THETA, t_on_frac=T_ON / T,
                   t_off_frac=T_OFF / T, fr_every_blocks=STRIDE)
    count = gw.Method("count", use_fr=True, theta=THETA, t_on_frac=T_ON / T,
                      t_off_frac=T_OFF / T, fr_every_blocks=STRIDE, coarse_bins=9)
    sham = gw.Method("sham", use_fr=True, sham=True, shadows="fr_temp")
    return [gw.SHUS, gbest, fr, count, sham]


def main(g_best):
    device = gw.DEVICE
    cfgs = [gw.GatewayConfig(**CELL, **COMMON) for _ in SEEDS]
    T = cfgs[0].T_total
    arms = build_arms(T, g_best)
    print(f"Phase A2: anchor_D, {len(SEEDS)} seeds x {len(arms)} arms = "
          f"{len(SEEDS)*len(arms)} rows, T={T:.0f}, g_best={g_best}")
    t0 = time.time()
    recs = gw.simulate_batch(cfgs, SEEDS, arms, batch_seed=BATCH_SEED,
                             device=device, progress=100_000)
    print(f"adaptive wall {time.time()-t0:.0f}s")

    # frozen-bias endpoint (secondary; interpretation only invoked on a near-tie)
    F_frozen = np.stack([r["pmf_t"][-1] for r in recs])
    t0 = time.time()
    fb = gw.run_frozen_bias(F_frozen, [gw.GatewayConfig(**CELL, **COMMON)] * len(recs),
                            group=[r["seed"] for r in recs], n_steps=60_000,
                            device=device)
    print(f"frozen-bias wall {time.time()-t0:.0f}s")

    os.makedirs(OUT, exist_ok=True)
    for i, rec in enumerate(recs):
        arrays = {k: rec[k] for k in
                  ("time", "pmf_t", "marginal_t", "x_grid", "F_ref", "l2_f_t",
                   "l2_fp_t", "kl_u_t", "tv_u_t", "ess_anc_t", "wmax_t",
                   "ess_anc_glob_t", "wmax_glob_t", "n_anc_t", "dep_ref_l2_t",
                   "dep_self_l2_t", "P_regions", "Q_regions", "event_time",
                   "event_theta", "event_ess_fr", "event_turnover")}
        arrays["frozen_bias_l2"] = np.array([fb["l2_f"][i]])
        meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
                "config": rec["config"], "method": rec["method"], "seed": rec["seed"],
                "cell": "anchor_D", "stage": "appmap_phaseA2_compare"}
        save_run(os.path.join(OUT, f"{rec['method']['name']}_seed{rec['seed']}"),
                 arrays, meta)

    by = {}
    for i, rec in enumerate(recs):
        rec["fb_l2"] = float(fb["l2_f"][i])
        by.setdefault(rec["method"]["name"], {})[rec["seed"]] = rec
    base = by["shus"]
    ladder = [E0 / 2, E0 / 4, E0 / 8]
    tau_b = {eps: np.array([time_to_accuracy(base[sd]["time"], base[sd]["l2_f_t"],
                                             eps) for sd in SEEDS])
             for eps in ladder}

    summary = {"g_best": g_best, "e0": E0, "seeds": SEEDS,
               "batch_seed": BATCH_SEED, "arms": {}}
    print(f"\n{'arm':<12s} {'dI_F% [CI]':>22s} {'eT ratio':>9s} {'fb ratio':>9s} "
          f"{'S(e0/2)':>8s} {'S(e0/4)':>8s} {'S(e0/8)':>8s}")
    for name in [a.name for a in build_arms(T, g_best)][1:]:
        rows = by[name]
        dIF = np.array([(rows[sd]["int_l2_f"] - base[sd]["int_l2_f"])
                        / base[sd]["int_l2_f"] for sd in SEEDS])
        m, lo, hi = paired_bootstrap_ci(dIF)
        eT = float(np.median([rows[sd]["final_l2_f"] / base[sd]["final_l2_f"]
                              for sd in SEEDS]))
        fbr = float(np.median([rows[sd]["fb_l2"] / base[sd]["fb_l2"]
                               for sd in SEEDS]))
        Ss = {}
        for eps in ladder:
            tau = np.array([time_to_accuracy(rows[sd]["time"], rows[sd]["l2_f_t"],
                                             eps) for sd in SEEDS])
            S = tau_b[eps] / tau
            Sv = S[np.isfinite(S)]
            Ss[eps] = float(np.median(Sv)) if len(Sv) else float("nan")
        print(f"{name:<12s} {100*m:>7.1f} [{100*lo:>5.1f},{100*hi:>5.1f}] "
              f"{eT:>9.3f} {fbr:>9.3f} "
              + " ".join(f"{Ss[e]:>8.2f}" for e in ladder))
        summary["arms"][name] = dict(dIF=[float(x) for x in dIF],
                                     dIF_ci=[m, lo, hi], eT_ratio=eT,
                                     fb_ratio=fbr,
                                     S_ladder={f"{e:.4f}": Ss[e] for e in ladder})

    # frozen direct contrast: fr_temp vs shus_gbest
    d_fg = np.array([(by["fr_temp"][sd]["int_l2_f"] - by["shus_gbest"][sd]["int_l2_f"])
                     / by["shus_gbest"][sd]["int_l2_f"] for sd in SEEDS])
    m, lo, hi = paired_bootstrap_ci(d_fg)
    if m >= 0.0:
        verdict = ("FR is not practically necessary on the gateway: "
                   "adaptation-rate tuning matches or beats the frozen FR winner")
    elif m <= -0.05 and hi < 0.0:
        verdict = ("the population correction supplies something adaptation-rate "
                   "tuning does not reproduce")
    else:
        verdict = "no headline claim (frozen rule): point estimate recorded"
    summary["contrast_fr_vs_gbest"] = dict(d=[float(x) for x in d_fg],
                                           ci=[m, lo, hi], verdict=verdict)
    print(f"\nfr_temp vs shus_gbest: {100*m:.1f}% [{100*lo:.1f},{100*hi:.1f}]")
    print(f"VERDICT: {verdict}")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"summary -> {OUT}/summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbest", type=float, default=None,
                    help="override g_best (default: read Phase-A1 summary)")
    a = ap.parse_args()
    g = a.gbest
    if g is None:
        with open(A1_SUMMARY) as f:
            g = float(json.load(f)["g_best"])
        print(f"g_best from A1 summary: {g}")
    if g == 1.0:
        sys.exit("g_best = 1.0: Phase A2 is unnecessary (prereg: Q1 answered from "
                 "A1 -- gain tuning does not beat the frozen baseline).")
    main(g)
