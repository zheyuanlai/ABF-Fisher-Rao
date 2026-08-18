"""Application-map Phase A1: plain-SHUS adaptation-gain screen on anchor_D (NO FR).

Design frozen in docs/PREREGISTRATION_APPLICATION_MAP.md (commit ae1a998, before this
run): g_shus in {0.25, 0.5, 0.75, 1.0, 1.5} as five noise-paired arms, fresh seeds
200..215, all gateway conventions unchanged. Question: does simply slowing the SHUS
adaptation rate already critically damp the establishment ring that temporary FR
damps (dI_F = -11.4% frozen Stage-3 result)?

g_best selection rule (frozen): among gains with median paired e_F(T) ratio vs
g = 1.0 <= 1.05, the lowest median paired dI_F wins; qualifying gains within 2% of
the winner resolve toward g closest to 1.0.

A1b extension (amendment frozen 2026-08-18 in the prereg doc): --extend "2.0,3.0"
runs additional gains with the SAME seeds and batch_seed — the engine noise stream is
method-independent, so extension rows are exactly noise-paired with the stored A1
rows; paired stats are computed against the stored g=1 records and the g_best rule
re-applies over the union of screened gains.

Usage: python scripts/run_appmap_phaseA_gain.py [--extend "2.0,3.0"]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from abpfr.diagnostics import (establishment_time_median, hit_time, kde_noise_floor)
from abpfr.io import save_run
from abpfr.metrics import cosine_modes, paired_bootstrap_ci
from abpfr.systems import gateway as gw

CELL = dict(beta=16.0, H=0.5, s=0.10, r=32.0)          # anchor_D, frozen
COMMON = dict(K=1024, dt=2e-4, n_steps=500_000, block=20, n_saves=400,
              ess_window_steps=4000)
SEEDS = list(range(200, 216))                          # 16 fresh seeds, frozen
GAINS = (0.25, 0.5, 0.75, 1.0, 1.5)                    # frozen screen grid
BATCH_SEED = 20260824
OUT = "results/appmap_phaseA_gain"


def gname(g):
    return f"shus_g{g:g}"


def ring_metrics(rec, e_star, t_hit):
    """Supporting transient metrics: a_1 sign changes after T_hit, ring-out time
    (last t with e_F > 2 e*), occupancy overshoot max P_+."""
    t = rec["time"]
    a1 = cosine_modes(rec["pmf_t"] - rec["F_ref"], rec["x_grid"],
                      gw.GRID.eval_lo, gw.GRID.eval_hi, k_max=1)[:, 0]
    m = np.isfinite(t_hit) & (t > (t_hit if np.isfinite(t_hit) else 0.0))
    s = np.sign(a1[m][np.abs(a1[m]) > 1e-4])
    signflips = int((np.diff(s) != 0).sum()) if s.size > 1 else 0
    over = rec["l2_f_t"] > 2.0 * e_star
    ringout = float(t[over][-1]) if over.any() else 0.0
    return dict(a1_signflips=signflips, ringout=ringout,
                overshoot=float(rec["P_regions"][:, 2].max()))


def main(gains=GAINS, prior_summary=None):
    device = gw.DEVICE
    print(f"device: {device}")
    cfgs = [gw.GatewayConfig(**CELL, **COMMON) for _ in SEEDS]
    T = cfgs[0].T_total
    arms = [gw.Method(gname(g), g_shus=g) for g in gains]
    print(f"Phase A1{'b' if prior_summary else ''}: anchor_D, {len(SEEDS)} seeds x "
          f"{len(gains)} gains = {len(SEEDS)*len(gains)} rows, T={T:.0f}, "
          f"plain SHUS only")
    t0 = time.time()
    recs = gw.simulate_batch(cfgs, SEEDS, arms, batch_seed=BATCH_SEED,
                             device=device, progress=50_000)
    wall = time.time() - t0
    print(f"wall {wall:.0f}s")

    os.makedirs(OUT, exist_ok=True)
    for rec in recs:
        arrays = {k: rec[k] for k in
                  ("time", "pmf_t", "marginal_t", "x_grid", "F_ref", "Fp_ref",
                   "l2_f_t", "l2_fp_t", "kl_u_t", "tv_u_t", "ess_anc_t", "wmax_t",
                   "n_anc_t", "dep_ref_l2_t", "dep_self_l2_t", "P_regions",
                   "Q_regions")}
        meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
                "config": rec["config"], "method": rec["method"], "seed": rec["seed"],
                "cell": "anchor_D", "stage": "appmap_phaseA_gain"}
        save_run(os.path.join(OUT, f"{rec['method']['name']}_seed{rec['seed']}"),
                 arrays, meta)

    # ---- gate/transient metrics per row -----------------------------------------
    fp = gw.mollified_fixed_point(cfgs[0])
    noise95 = float(np.quantile(
        kde_noise_floor(COMMON["K"], 0.10, gw.GRID, n_rep=512, seed=777), 0.95))
    D_tol = 1.5 * (fp["kl_star"] + noise95)
    e_star = fp["e_star"]
    print(f"D_tol={D_tol:.5f}  e*={e_star:.5f}")

    by = {g: {} for g in gains}          # gain -> seed -> row dict
    for rec in recs:
        g = rec["method"]["g_shus"]
        t = rec["time"]
        th = hit_time(rec["P_regions"][:, 2], t, hold_frac=0.05)
        te = establishment_time_median(rec["kl_u_t"], t, D_tol, hold_frac=0.10)
        row = dict(seed=rec["seed"], T_hit=th, T_est=te,
                   I_F=float(rec["int_l2_f"]), eT=float(rec["l2_f_t"][-1]),
                   **ring_metrics(rec, e_star, th))
        by[g][rec["seed"]] = row
    if prior_summary is not None:        # A1b: merge the stored (noise-paired) rows
        for g in prior_summary["gains"]:
            by[float(g)] = {r["seed"]: r
                            for r in prior_summary["rows"][gname(g)]["per_seed"]}
    all_gains = sorted(by)

    # ---- paired comparison vs g = 1.0 and frozen g_best selection ----------------
    summary = {"cell": CELL, "common": COMMON, "seeds": SEEDS, "gains": all_gains,
               "batch_seed": BATCH_SEED, "D_tol": D_tol, "e_star": e_star,
               "noise95": noise95, "wall_s": wall, "rows": {}, "paired": {}}
    med = lambda rows, k: float(np.nanmedian([r[k] for r in rows]))
    base = by[1.0]
    print(f"\n{'gain':>6s} {'T_hit':>7s} {'T_est':>7s} {'I_F':>7s} {'e_F(T)':>8s} "
          f"{'dI_F%':>18s} {'eT ratio':>9s} {'flips':>6s} {'ringout':>8s} {'ovsh':>6s}")
    qualified = {}
    for g in all_gains:
        rows = list(by[g].values())
        dI = np.array([(by[g][sd]["I_F"] - base[sd]["I_F"]) / base[sd]["I_F"]
                       for sd in SEEDS])
        rT = np.array([by[g][sd]["eT"] / base[sd]["eT"] for sd in SEEDS])
        m_dI, lo, hi = paired_bootstrap_ci(dI)
        m_rT = float(np.median(rT))
        summary["rows"][gname(g)] = {k: med(rows, k) for k in
                                     ("T_hit", "T_est", "I_F", "eT",
                                      "a1_signflips", "ringout", "overshoot")}
        summary["rows"][gname(g)]["per_seed"] = rows
        summary["paired"][gname(g)] = dict(dI_F=m_dI, dI_F_ci=[lo, hi],
                                           eT_ratio=m_rT)
        if m_rT <= 1.05:
            qualified[g] = m_dI
        print(f"{g:>6g} {med(rows,'T_hit'):>7.2f} {med(rows,'T_est'):>7.2f} "
              f"{med(rows,'I_F'):>7.3f} {med(rows,'eT'):>8.4f} "
              f"{100*m_dI:>7.1f} [{100*lo:>5.1f},{100*hi:>5.1f}] {m_rT:>9.3f} "
              f"{med(rows,'a1_signflips'):>6.1f} {med(rows,'ringout'):>8.1f} "
              f"{med(rows,'overshoot'):>6.3f}")

    best_dI = min(qualified.values())
    cands = [g for g, v in qualified.items() if v <= best_dI + 0.02]
    g_best = min(cands, key=lambda g: (abs(g - 1.0), -g))
    summary["g_best"] = g_best
    summary["qualified"] = {gname(g): v for g, v in qualified.items()}
    print(f"\nqualified (eT ratio <= 1.05): {sorted(qualified)}")
    print(f"g_best (frozen rule) = {g_best}  "
          f"(median paired dI_F = {100*qualified[g_best]:.1f}%)")

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"summary -> {OUT}/summary.json")
    make_figure(recs, gains,
                "phaseA_gain_overview_ext.png" if prior_summary
                else "phaseA_gain_overview.png")


def make_figure(recs, gains=GAINS, fname="phaseA_gain_overview.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = {g: f"C{i}" for i, g in enumerate(gains)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for g in gains:
        rs = [r for r in recs if r["method"]["g_shus"] == g]
        t = rs[0]["time"]
        for key, ax in (("l2_f_t", axes[0, 0]), ("kl_u_t", axes[0, 1])):
            curves = np.stack([r[key] for r in rs])
            ax.plot(t, np.median(curves, 0), color=cmap[g], label=f"g={g:g}")
            ax.fill_between(t, *np.quantile(curves, [0.25, 0.75], axis=0),
                            color=cmap[g], alpha=0.15, lw=0)
            ax.set_yscale("log")
        occ = np.stack([r["P_regions"][:, 2] for r in rs])
        axes[1, 0].plot(t, np.median(occ, 0), color=cmap[g])
        a1 = np.stack([np.abs(cosine_modes(r["pmf_t"] - r["F_ref"], r["x_grid"],
                                           gw.GRID.eval_lo, gw.GRID.eval_hi,
                                           k_max=1))[:, 0] for r in rs])
        axes[1, 1].plot(t, np.median(a1, 0), color=cmap[g])
        axes[1, 1].set_yscale("log")
    axes[0, 0].set_ylabel(r"$e_F(t)$ (median, IQR)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_ylabel(r"KL$(\hat p_t\|u)$")
    axes[1, 0].set_ylabel(r"$P_+$")
    axes[1, 1].set_ylabel(r"$|a_1(t)|$")
    for ax in axes[1]:
        ax.set_xlabel("t")
    fig.suptitle("Phase A1: plain-SHUS adaptation-gain screen on anchor_D "
                 "(16 seeds; no FR)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    print(f"figure -> {OUT}/{fname}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extend", type=str, default=None,
                    help='comma-separated extra gains, e.g. "2.0,3.0" (A1b)')
    a = ap.parse_args()
    if a.extend:
        new_gains = tuple(float(x) for x in a.extend.split(","))
        with open(os.path.join(OUT, "summary.json")) as f:
            prior = json.load(f)
        assert not (set(new_gains) & set(float(g) for g in prior["gains"])), \
            "extension gains overlap the stored screen"
        main(gains=new_gains, prior_summary=prior)
    else:
        main()
