"""Application-map Phase D2: FR vs count-balancing resolution on t_mid (2D torus).

Design frozen in docs/PREREGISTRATION_APPLICATION_MAP.md before this run: seeds
400..415, seven paired arms (shus / fr_temp / count6 / count9 / count12 / sham /
shus_g1.5), theta = 0.01, stride 10 blocks, window [14, 50] derived from the
frozen rule on D1 SHUS-only quantiles. This is the Q3 experiment: does smooth FR
reallocation beat coarse histogram balancing as the histogram gets sparse?

Usage: python scripts/run_appmap_phaseD2_torus_fr.py
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
SEEDS = list(range(400, 416))                          # 16 fresh seeds, frozen
T_ON, T_OFF, THETA, STRIDE = 14.0, 50.0, 0.01, 10      # frozen derivation
BATCH_SEED = 20260828
OUT = "results/appmap_phaseD2_torus_fr"


def build_arms(T):
    fr_kw = dict(use_fr=True, theta=THETA, t_on_frac=T_ON / T,
                 t_off_frac=T_OFF / T, fr_every_blocks=STRIDE)
    return [
        Method("shus"),
        Method("fr_temp", **fr_kw),
        Method("count6", **fr_kw, coarse_bins=6),
        Method("count9", **fr_kw, coarse_bins=9),
        Method("count12", **fr_kw, coarse_bins=12),
        Method("sham", use_fr=True, sham=True, shadows="fr_temp"),
        Method("shus_g1.5", g_shus=1.5),
    ]


def main():
    device = t2.DEVICE
    cfgs = [t2.Torus2DConfig(**CELL, **COMMON) for _ in SEEDS]
    T = cfgs[0].T_total
    arms = build_arms(T)
    print(f"Phase D2: t_mid, {len(SEEDS)} seeds x {len(arms)} arms = "
          f"{len(SEEDS)*len(arms)} rows, T={T:.0f}, window [{T_ON},{T_OFF}], "
          f"theta={THETA}, stride={STRIDE}")
    t0 = time.time()
    recs = t2.simulate_batch(cfgs, SEEDS, arms, batch_seed=BATCH_SEED,
                             device=device, progress=25_000)
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
                "cell": "t_mid", "stage": "appmap_phaseD2_torus_fr"}
        save_run(os.path.join(OUT, f"{rec['method']['name']}_seed{rec['seed']}"),
                 arrays, meta)

    by = {}
    for rec in recs:
        by.setdefault(rec["method"]["name"], {})[rec["seed"]] = rec
    base = by["shus"]

    summary = {"cell": CELL, "seeds": SEEDS, "batch_seed": BATCH_SEED,
               "window": [T_ON, T_OFF], "theta": THETA, "stride": STRIDE,
               "arms": {}}
    K = COMMON["K"]
    print(f"\n{'arm':<11s} {'dI_F% [CI]':>22s} {'eT ratio':>9s} {'minESS':>7s} "
          f"{'n_anc':>6s} {'turnover':>9s}")
    for name in [a.name for a in build_arms(T)][1:]:
        rows = by[name]
        dIF = np.array([(rows[sd]["int_l2_f"] - base[sd]["int_l2_f"])
                        / base[sd]["int_l2_f"] for sd in SEEDS])
        m, lo, hi = paired_bootstrap_ci(dIF)
        eT = float(np.median([rows[sd]["final_l2_f"] / base[sd]["final_l2_f"]
                              for sd in SEEDS]))
        ess = float(np.median([rows[sd]["ess_anc_t"].min() / K for sd in SEEDS]))
        nanc = float(np.median([rows[sd]["n_anc_t"][-1] / K for sd in SEEDS]))
        turn = float(np.median([rows[sd]["total_turnover"] for sd in SEEDS]))
        print(f"{name:<11s} {100*m:>7.1f} [{100*lo:>5.1f},{100*hi:>5.1f}] "
              f"{eT:>9.3f} {ess:>7.2f} {nanc:>6.2f} {turn:>9.0f}")
        summary["arms"][name] = dict(dIF=[float(x) for x in dIF],
                                     dIF_ci=[m, lo, hi], eT_ratio=eT,
                                     min_ess=ess, n_anc=nanc, turnover=turn)

    # direct paired contrasts: fr_temp vs each count resolution
    print()
    summary["contrasts"] = {}
    for cn in ("count6", "count9", "count12"):
        d = np.array([(by["fr_temp"][sd]["int_l2_f"] - by[cn][sd]["int_l2_f"])
                      / by[cn][sd]["int_l2_f"] for sd in SEEDS])
        m, lo, hi = paired_bootstrap_ci(d)
        summary["contrasts"][f"fr_vs_{cn}"] = [m, lo, hi]
        print(f"fr_temp vs {cn}: {100*m:>6.1f}% [{100*lo:.1f},{100*hi:.1f}]")

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"summary -> {OUT}/summary.json")
    make_figure(by)


def make_figure(by):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    order = ("shus", "fr_temp", "count6", "count9", "count12", "sham", "shus_g1.5")
    cmap = {n: f"C{i}" for i, n in enumerate(order)}
    for name in order:
        rs = [by[name][sd] for sd in SEEDS]
        t = rs[0]["time"]
        for key, ax in (("l2_f_t", axes[0]), ("kl_u_t", axes[1])):
            curves = np.stack([r[key] for r in rs])
            ax.plot(t, np.median(curves, 0), color=cmap[name], label=name, lw=1.2)
            ax.set_yscale("log")
        occ = np.stack([r["P_regions"].min(axis=1) for r in rs])
        axes[2].plot(t, np.median(occ, 0), color=cmap[name], lw=1.2)
    for x in (14.0, 50.0):
        for ax in axes:
            ax.axvline(x, color="gray", ls=":", lw=0.8)
    axes[0].set_ylabel(r"$e_F(t)$")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel(r"KL$(\hat p_t\|u)$")
    axes[2].set_ylabel("min basin occupancy")
    for ax in axes:
        ax.set_xlabel("t")
    fig.suptitle("Phase D2: FR vs count-balancing resolution on t_mid "
                 "(16 seeds; dotted: FR window)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "phaseD2_overview.png"), dpi=130)
    print(f"figure -> {OUT}/phaseD2_overview.png")


if __name__ == "__main__":
    main()
