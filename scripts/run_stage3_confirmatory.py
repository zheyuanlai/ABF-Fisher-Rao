"""Stage 3: gateway confirmatory on anchor_D — 5 preregistered arms x 32 fresh seeds.

Arms (docs/PREREGISTRATION_GATEWAY.md; FR parameters frozen from the Stage-2 pilot
before this script ever ran):
    shus            plain baseline
    fr_temp         theta=0.01, stride 10 blocks, window [6, 14]   (the winner)
    fr_persistent   same theta/stride, window [6, T]               (overdamping arm)
    sham            matched-turnover control shadowing fr_temp
    count           coarse 9-bin count balancing, same theta/stride/window

Co-primary endpoints: paired dI_F (success: median <= -10%, bootstrap 95% CI < 0)
and S_eps* at eps* = 0.0110 (success: median >= 1.25, CI > 1). Secondary: e_F(T)
non-inferiority (<= 1.05x), ancestry floors (min windowed ESS >= 0.5, final
n_anc/K >= 0.5), frozen-bias validation (paired noise across arms of a seed).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from abpfr.io import save_run
from abpfr.metrics import paired_bootstrap_ci, time_to_accuracy
from abpfr.systems import gateway as gw

CELL = dict(beta=16.0, H=0.5, s=0.10, r=32.0)
COMMON = dict(K=1024, dt=2e-4, n_steps=500_000, block=20, n_saves=400,
              ess_window_steps=4000)
SEEDS = list(range(100, 132))              # 32 fresh production seeds, frozen
T_ON, T_OFF, THETA, STRIDE = 6.0, 14.0, 0.01, 10
EPS_STAR = 0.0110
E0 = 0.236
OUT = "results/stage3_confirmatory"

ARMS = None  # built in main() (needs T_total)


def build_arms(T):
    fr_temp = gw.Method("fr_temp", use_fr=True, theta=THETA,
                        t_on_frac=T_ON / T, t_off_frac=T_OFF / T,
                        fr_every_blocks=STRIDE)
    fr_pers = gw.Method("fr_persistent", use_fr=True, theta=THETA,
                        t_on_frac=T_ON / T, t_off_frac=1.0,
                        fr_every_blocks=STRIDE)
    sham = gw.Method("sham", use_fr=True, sham=True, shadows="fr_temp")
    count = gw.Method("count", use_fr=True, theta=THETA,
                      t_on_frac=T_ON / T, t_off_frac=T_OFF / T,
                      fr_every_blocks=STRIDE, coarse_bins=9)
    return [gw.SHUS, fr_temp, fr_pers, sham, count]


def main():
    device = gw.DEVICE
    cfgs = [gw.GatewayConfig(**CELL, **COMMON) for _ in SEEDS]
    T = cfgs[0].T_total
    arms = build_arms(T)
    print(f"confirmatory: anchor_D, {len(SEEDS)} seeds x {len(arms)} arms "
          f"= {len(SEEDS)*len(arms)} rows, T={T:.0f}")
    t0 = time.time()
    recs = gw.simulate_batch(cfgs, SEEDS, arms, batch_seed=20260820,
                             device=device, progress=100_000)
    print(f"adaptive wall {time.time()-t0:.0f}s")

    # frozen-bias validation: score every arm's final learned bias with a fresh
    # population; arms of one seed share initial conditions and noise (paired)
    F_frozen = np.stack([r["pmf_t"][-1] for r in recs])
    group = [r["seed"] for r in recs]
    t0 = time.time()
    fb = gw.run_frozen_bias(F_frozen, [gw.GatewayConfig(**CELL, **COMMON)] * len(recs),
                            group=group, n_steps=60_000, device=device)
    print(f"frozen-bias wall {time.time()-t0:.0f}s")

    os.makedirs(OUT, exist_ok=True)
    for i, rec in enumerate(recs):
        name = f"{rec['method']['name']}_seed{rec['seed']}"
        arrays = {k: rec[k] for k in
                  ("time", "pmf_t", "marginal_t", "x_grid", "F_ref", "l2_f_t",
                   "l2_fp_t", "kl_u_t", "tv_u_t", "ess_anc_t", "wmax_t",
                   "ess_anc_glob_t", "wmax_glob_t", "n_anc_t", "dep_ref_l2_t",
                   "dep_self_l2_t", "P_regions", "Q_regions", "event_time",
                   "event_theta", "event_ess_fr", "event_turnover")}
        arrays["frozen_bias_l2"] = np.array([fb["l2_f"][i]])
        meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
                "config": rec["config"], "method": rec["method"], "seed": rec["seed"],
                "cell": "anchor_D", "stage": "stage3_confirmatory"}
        save_run(os.path.join(OUT, name), arrays, meta)

    by = {}
    for i, rec in enumerate(recs):
        rec["fb_l2"] = float(fb["l2_f"][i])
        by.setdefault(rec["method"]["name"], {})[rec["seed"]] = rec
    base = by["shus"]
    tau_b = {sd: time_to_accuracy(base[sd]["time"], base[sd]["l2_f_t"], EPS_STAR)
             for sd in SEEDS}

    summary = {"eps_star": EPS_STAR, "e0": E0, "arms": {}}
    print(f"\n{'arm':<14s} {'dI_F% [CI]':>22s} {'S_eps* [CI]':>20s} {'cens':>5s} "
          f"{'eT ratio':>8s} {'fb ratio':>8s} {'minESS':>7s} {'n_anc':>6s}")
    for name in ("fr_temp", "fr_persistent", "sham", "count"):
        rows = by[name]
        dIF = np.array([(rows[sd]["int_l2_f"] - base[sd]["int_l2_f"])
                        / base[sd]["int_l2_f"] for sd in SEEDS])
        m_dIF, lo_dIF, hi_dIF = paired_bootstrap_ci(dIF)
        tau = np.array([time_to_accuracy(rows[sd]["time"], rows[sd]["l2_f_t"],
                                         EPS_STAR) for sd in SEEDS])
        S = np.array([tau_b[sd] / tau[i] for i, sd in enumerate(SEEDS)])
        cens = int(np.isnan(S).sum())
        Sv = S[~np.isnan(S)]
        m_S, lo_S, hi_S = paired_bootstrap_ci(Sv) if len(Sv) > 3 else (np.nan,) * 3
        eT = float(np.median([rows[sd]["final_l2_f"] / base[sd]["final_l2_f"]
                              for sd in SEEDS]))
        fbr = float(np.median([rows[sd]["fb_l2"] / base[sd]["fb_l2"]
                               for sd in SEEDS]))
        ess = float(np.median([rows[sd]["ess_anc_t"].min() / COMMON["K"]
                               for sd in SEEDS]))
        nanc = float(np.median([rows[sd]["n_anc_t"][-1] / COMMON["K"]
                                for sd in SEEDS]))
        print(f"{name:<14s} {100*m_dIF:>7.1f} [{100*lo_dIF:.1f},{100*hi_dIF:.1f}] "
              f"{m_S:>7.2f} [{lo_S:.2f},{hi_S:.2f}] {cens:>5d} {eT:>8.3f} "
              f"{fbr:>8.3f} {ess:>7.2f} {nanc:>6.2f}")
        summary["arms"][name] = dict(
            dIF=[float(x) for x in dIF], dIF_ci=[m_dIF, lo_dIF, hi_dIF],
            S=[float(x) for x in S], S_ci=[m_S, lo_S, hi_S], censored=cens,
            eT_ratio=eT, fb_ratio=fbr, min_ess=ess, n_anc=nanc)
    tb = np.array([tau_b[sd] for sd in SEEDS])
    summary["tau_base_censored"] = int(np.isnan(tb).sum())
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nsummary -> {OUT}/summary.json")
    make_figure(by)


def make_figure(by):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["shus", "fr_temp", "fr_persistent", "sham", "count"]
    colors = dict(zip(order, ["k", "C1", "C3", "C7", "C0"]))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), sharex=True)
    t = by["shus"][SEEDS[0]]["time"]
    meds = {}
    for name in order:
        cur = np.stack([by[name][sd]["l2_f_t"] for sd in SEEDS])
        meds[name] = np.median(cur, 0)
        axes[0, 0].plot(t, meds[name], color=colors[name], label=name, lw=1.6)
        lo, hi = np.quantile(cur, [0.25, 0.75], axis=0)
        axes[0, 0].fill_between(t, lo, hi, color=colors[name], alpha=0.12, lw=0)
        kl = np.median(np.stack([by[name][sd]["kl_u_t"] for sd in SEEDS]), 0)
        axes[1, 0].plot(t, kl, color=colors[name], lw=1.4)
        na = np.median(np.stack([by[name][sd]["n_anc_t"] for sd in SEEDS]), 0)
        axes[1, 1].plot(t, na / COMMON["K"], color=colors[name], lw=1.4)
    for name in order[1:]:
        axes[0, 1].plot(t, meds[name] / meds["shus"], color=colors[name], lw=1.4)
    axes[0, 1].axhline(1.0, color="k", ls=":", lw=0.8)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"$e_F(t)$ (median, IQR)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_ylabel(r"$R_F(t)$ vs plain SHUS")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel(r"KL$(\hat p_t\|u)$")
    axes[1, 1].set_ylabel(r"$N_{\rm anc}/K$ (global)")
    for ax in axes.flat:
        ax.axvspan(T_ON, T_OFF, color="C1", alpha=0.08)
    for ax in axes[1]:
        ax.set_xlabel("t")
    fig.suptitle("Stage 3 confirmatory: anchor_D, 32 fresh seeds, 5 preregistered "
                 "arms (temporary-FR window shaded)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confirmatory_overview.png"), dpi=130)
    print(f"figure -> {OUT}/confirmatory_overview.png")


if __name__ == "__main__":
    main()
