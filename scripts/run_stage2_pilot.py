"""Stage 2: weak-late-short FR pilot on the frozen anchor_D cell.

Design frozen in docs/PREREGISTRATION_GATEWAY.md (2026-08-18, from SHUS-only data):
theta in {0.01, 0.025, 0.05} x stride in {5, 10} blocks x t_off in {14, 22},
t_on = 6.0, pilot seeds 8..15.  Every FR config carries its own matched-turnover
sham; all 25 arms of a seed share initial conditions and Langevin noise.

Selection (frozen): reject if min windowed ESS_anc/K < 0.5 or median paired
e_F(T) > 1.05x SHUS; win = median paired dI_F <= -10% AND beats own sham; among
winners pick the smallest turnover budget gamma_eff * window.
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from abpfr.io import save_run
from abpfr.metrics import time_to_accuracy
from abpfr.systems import gateway as gw

CELL = dict(beta=16.0, H=0.5, s=0.10, r=32.0)          # anchor_D, frozen
COMMON = dict(K=1024, dt=2e-4, n_steps=500_000, block=20, n_saves=400,
              ess_window_steps=4000)
SEEDS = list(range(8, 16))                              # pilot seeds, frozen
T_ON, T_OFFS = 6.0, (14.0, 22.0)
THETAS = (0.01, 0.025, 0.05)
STRIDES = (5, 10)
EPS_STAR = 0.0110                                       # frozen from Stage 1
OUT = "results/stage2_pilot"


def build_methods(T_total):
    methods = [gw.SHUS]
    for th in THETAS:
        for st in STRIDES:
            for toff in T_OFFS:
                name = f"fr_t{th:g}_s{st}_w{int(toff)}"
                fr = gw.Method(name, use_fr=True, theta=th,
                               t_on_frac=T_ON / T_total, t_off_frac=toff / T_total,
                               fr_every_blocks=st, alpha_ess=0.5)
                sham = gw.Method("sham_" + name, use_fr=True, sham=True, shadows=name)
                methods += [fr, sham]
    return methods


def gamma_eff(th, st):
    dt_fr = st * COMMON["block"] * COMMON["dt"]
    return -math.log(1.0 - th) / dt_fr


def main():
    device = gw.DEVICE
    cfgs = [gw.GatewayConfig(**CELL, **COMMON) for _ in SEEDS]
    T = cfgs[0].T_total
    methods = build_methods(T)
    print(f"pilot: anchor_D, {len(SEEDS)} seeds x {len(methods)} arms "
          f"= {len(SEEDS)*len(methods)} rows, T={T:.0f}")
    t0 = time.time()
    recs = gw.simulate_batch(cfgs, SEEDS, methods, batch_seed=20260819,
                             device=device, progress=100_000)
    print(f"wall {time.time()-t0:.0f}s")

    os.makedirs(OUT, exist_ok=True)
    for rec in recs:
        name = f"{rec['method']['name']}_seed{rec['seed']}"
        arrays = {k: rec[k] for k in
                  ("time", "pmf_t", "marginal_t", "x_grid", "F_ref", "l2_f_t",
                   "l2_fp_t", "kl_u_t", "tv_u_t", "ess_anc_t", "wmax_t",
                   "ess_anc_glob_t", "wmax_glob_t", "n_anc_t", "dep_ref_l2_t",
                   "dep_self_l2_t", "P_regions", "Q_regions", "event_time",
                   "event_theta", "event_ess_fr", "event_turnover")}
        meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
                "config": rec["config"], "method": rec["method"], "seed": rec["seed"],
                "cell": "anchor_D", "stage": "stage2_pilot"}
        save_run(os.path.join(OUT, name), arrays, meta)

    by = {}
    for rec in recs:
        by.setdefault(rec["method"]["name"], {})[rec["seed"]] = rec

    base = by["shus"]
    tau_base = {sd: time_to_accuracy(base[sd]["time"], base[sd]["l2_f_t"], EPS_STAR)
                for sd in SEEDS}

    def stats(name):
        rows = by[name]
        dIF = [(rows[sd]["int_l2_f"] - base[sd]["int_l2_f"]) / base[sd]["int_l2_f"]
               for sd in SEEDS]
        eT_ratio = [rows[sd]["final_l2_f"] / base[sd]["final_l2_f"] for sd in SEEDS]
        min_ess = [rows[sd]["ess_anc_t"].min() / COMMON["K"] for sd in SEEDS]
        n_anc = [rows[sd]["n_anc_t"][-1] / COMMON["K"] for sd in SEEDS]
        tau = [time_to_accuracy(rows[sd]["time"], rows[sd]["l2_f_t"], EPS_STAR)
               for sd in SEEDS]
        S = [tau_base[sd] / tau[i] if np.isfinite(tau[i]) else np.nan
             for i, sd in enumerate(SEEDS)]
        IF = [rows[sd]["int_l2_f"] for sd in SEEDS]
        return dict(dIF=float(np.median(dIF)), eT_ratio=float(np.median(eT_ratio)),
                    min_ess=float(np.median(min_ess)), n_anc=float(np.median(n_anc)),
                    S=float(np.nanmedian(S)), IF=IF,
                    dIF_all=[float(x) for x in dIF])

    print(f"\n{'config':<22s} {'g_eff':>6s} {'budget':>7s} {'dI_F%':>7s} "
          f"{'vs sham%':>8s} {'eT ratio':>8s} {'S_eps*':>7s} {'minESS':>7s} "
          f"{'n_anc':>6s}  verdict")
    summary = {"eps_star": EPS_STAR, "configs": {}}
    winners = []
    for th in THETAS:
        for st in STRIDES:
            for toff in T_OFFS:
                name = f"fr_t{th:g}_s{st}_w{int(toff)}"
                fr, sh = stats(name), stats("sham_" + name)
                vs_sham = float(np.median(
                    [(f - s) / s for f, s in zip(fr["IF"], sh["IF"])]))
                ok_safe = fr["min_ess"] >= 0.5 and fr["eT_ratio"] <= 1.05
                win = ok_safe and fr["dIF"] <= -0.10 and vs_sham < 0.0
                g = gamma_eff(th, st)
                budget = g * (toff - T_ON)
                verdict = "WIN" if win else ("rejected" if not ok_safe else "no gain")
                print(f"{name:<22s} {g:>6.2f} {budget:>7.1f} {100*fr['dIF']:>6.1f}% "
                      f"{100*vs_sham:>7.1f}% {fr['eT_ratio']:>8.3f} {fr['S']:>7.2f} "
                      f"{fr['min_ess']:>7.2f} {fr['n_anc']:>6.2f}  {verdict}")
                summary["configs"][name] = dict(
                    theta=th, stride=st, t_on=T_ON, t_off=toff, gamma_eff=g,
                    budget=budget, fr=fr, sham=sh, vs_sham=vs_sham, win=bool(win))
                if win:
                    winners.append((budget, name))
    if winners:
        winners.sort()
        print(f"\nFROZEN WINNER (smallest budget): {winners[0][1]}")
        summary["winner"] = winners[0][1]
    else:
        print("\nNO CONFIG WINS under the frozen rule.")
        summary["winner"] = None
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    make_figure(by, base)


def make_figure(by, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    t = base[SEEDS[0]]["time"]
    bmed = np.median(np.stack([base[sd]["l2_f_t"] for sd in SEEDS]), 0)
    for ax in axes[:2]:
        ax.plot(t, bmed, "k-", lw=2, label="shus")
        ax.set_yscale("log")
        ax.set_xlabel("t")
    for i, (name, rows) in enumerate(sorted(by.items())):
        if name == "shus" or name.startswith("sham"):
            continue
        med = np.median(np.stack([rows[sd]["l2_f_t"] for sd in SEEDS]), 0)
        axes[0].plot(t, med, lw=1.0, alpha=0.8, label=name)
        rel = med / bmed
        axes[1].plot(t, rel, lw=1.0, alpha=0.8)
    axes[1].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[0].set_ylabel(r"$e_F(t)$ (median over pilot seeds)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].set_ylabel(r"$R_F(t) = e_F^{\rm FR}/e_F^{\rm SHUS}$")
    axes[1].set_yscale("linear")
    for name, rows in sorted(by.items()):
        if not name.startswith("fr_"):
            continue
        med = np.median(np.stack([rows[sd]["n_anc_t"] for sd in SEEDS]), 0)
        axes[2].plot(t, med / rows[SEEDS[0]]["config"]["K"], lw=1.0, alpha=0.8)
    axes[2].set_ylabel(r"$N_{\rm anc}/K$ (global)")
    axes[2].set_xlabel("t")
    fig.suptitle("Stage-2 pilot: weak-late-short FR on anchor_D (windows shaded "
                 f"[{T_ON}, {T_OFFS}]) ")
    for ax in axes:
        ax.axvspan(T_ON, T_OFFS[1], color="C1", alpha=0.06)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pilot_overview.png"), dpi=130)
    print(f"figure -> {OUT}/pilot_overview.png")


if __name__ == "__main__":
    main()
