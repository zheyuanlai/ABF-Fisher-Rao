#!/usr/bin/env python
"""Movie: the entropic gateway under ABF and under ABF + Fisher-Rao, same seed, same noise.

Two columns, one per arm, driven by ONE run of the frozen corrected-baseline cell
(beta 16, s 0.10, r 32, N 2048, T 40, gamma 1.5; results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json
+ configs/information_campaign/gateway_corrected_confirmation_prereg.json).  Both arms sit in one
batch, so they share the initial condition and every Langevin increment: whatever differs between
the columns is the birth-death step.

Per column, top to bottom:
  * walkers on the beta V(x, y) landscape (the narrow y-channel at x = 0 IS the entropic gateway),
    with every Fisher-Rao death (red cross) and birth (green ring) flashed where it happened;
  * the walker histogram along xi against the uniform target;
  * the learned free energy at the corrected read-out h_read* against the analytic reference.
Bottom strip: e_F(t) for both arms with a moving cursor, and the fraction of walkers in B_+.

    CUDA_VISIBLE_DEVICES=3 python scripts/make_gateway_movie.py simulate            # ~1 min on one GPU
    python scripts/make_gateway_movie.py render --workers 24                          # frames + mp4

Data: results/gateway_movie/movie_data.npz; movie: results/gateway_movie/gateway_abf_vs_fr.mp4.
The engine record behind it is ``store_snapshots`` (tests/test_gateway_snapshots.py: bit-inert).

``--estimator histogram`` (both stages) runs the SAME cell, seed and noise with the textbook
histogram (P0) mean-force estimator at the campaign's frozen width
(configs/histogram_abf/selected_bins.json, 45 bins = 0.08; docs/HISTOGRAM_ABF_REPLICATION.md):
both arms feel the own-bin bias force M_j / (C_j + 1) and the drawn / scored free energy is the
estimator's OWN exact piecewise-linear PMF (no kernel read-out).  Output in
results/gateway_movie_histogram/ (gateway_abf_vs_fr_histogram.mp4).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

BASE_PREREG = os.path.join(ROOT, "results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json")
PREREG = os.path.join(ROOT, "configs/information_campaign/gateway_corrected_confirmation_prereg.json")
SELECTED = os.path.join(ROOT, "configs/histogram_abf/selected_bins.json")
OUT_DIR = {"kernel": os.path.join(ROOT, "results", "gateway_movie"),
           "histogram": os.path.join(ROOT, "results", "gateway_movie_histogram")}
MOVIE = {"kernel": "gateway_abf_vs_fr", "histogram": "gateway_abf_vs_fr_histogram"}
# confirmation medians (final e_F, integrated), quoted next to this seed's numbers
CONFIRM = {"kernel": (-59.4, -31.9, "32 pairs, docs/GATEWAY_CORRECTED_BASELINE.md"),
           "histogram": (-47.7, -30.0, "32 pairs, results/histogram_abf/gateway/confirmation/summary.json")}

C_ABF = "#2a78d6"     # blue   (presentation figures)
C_FR = "#eb6834"      # orange (presentation figures)
C_DIE = "#d62728"
C_BIRTH = "#1f9e4a"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e4e3df"
C_REF = "#0b0b0b"
XMIN, XMAX = -1.8, 1.8
YMIN, YMAX = -0.8, 0.8
X_BASIN = 0.5
V_MAX_KT = 20.0


# ----------------------------------------------------------------------------------------------
# stage 1: simulate
# ----------------------------------------------------------------------------------------------
def simulate(a):
    import torch
    import gateway_core as gw
    from run_gateway_bandwidth_audit import build_config
    from analyze_gateway_bandwidth_audit import mean_force_at, e_f
    from eb_abffr_core import EVAL_LO, EVAL_HI, XMIN as eb_XMIN, XMAX as eb_XMAX

    base, pre = json.load(open(BASE_PREREG)), json.load(open(PREREG))
    sampler, cell = base["sampler"], base["cell"]
    cb = pre["corrected_baseline"]
    h_bias, h_star, gamma = float(cb["h_bias"]), float(cb["h_read_star"]), float(pre["rate"]["gamma"])
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU (CUDA_VISIBLE_DEVICES)"
    cfg = build_config(sampler, cell, a.init, h_bias)
    n_bins = 0
    if a.estimator == "histogram":
        n_bins = int(a.n_bins or json.load(open(SELECTED))["gateway"]["selected_n_bins"])
        cfg = dataclasses.replace(cfg, estimator="histogram", n_bins=n_bins)
    arms = [gw.ABF, dataclasses.replace(gw.FR_UNIFORM, gamma=gamma)]
    spec = gw.BatchSpec(configs=[cfg], seeds=[a.seed], methods=arms, batch_seed=a.batch_seed)
    est_txt = (f"histogram estimator, {n_bins} bins (Delta {(eb_XMAX - eb_XMIN) / n_bins:g}), own read-out"
               if n_bins else f"kernel estimator h_bias {h_bias:g}, read-out h_read* {h_star:g}")
    print(f"cell beta {cfg.beta:g} s {cfg.s:g} r {cfg.r:g} H {cfg.H:g} N {cfg.N} T {cfg.T_total:g}; "
          f"{est_txt}; gamma {gamma:g}; init {a.init} seed {a.seed}; "
          f"snapshot every {a.snap_every} steps", flush=True)
    t0 = time.time()
    recs = gw.simulate_batch(spec, store_profiles=True, store_accumulators=True,
                             store_snapshots=a.snap_every, progress=20_000)
    print(f"simulated in {time.time() - t0:.0f}s", flush=True)
    abf, fr = recs[0], recs[1]
    assert abf["method"] == "abf" and fr["method"] == "fr_uniform"

    x = np.asarray(abf["x_grid"], float); dx = float(x[1] - x[0])
    mask = (x >= EVAL_LO) & (x <= EVAL_HI)
    F_ref = np.asarray(abf["F_ref"], float)
    out = dict(x_grid=x, F_ref=F_ref, Fp_ref=np.asarray(abf["Fp_ref"], float), eval_mask=mask,
               t_save=np.asarray(abf["t"], float), snap_t=np.asarray(abf["snap_t"], float),
               snap_step=np.asarray(abf["snap_step"]), config_json=json.dumps(abf["config"], sort_keys=True),
               h_bias=h_bias, h_read_star=h_star, gamma=gamma, seed=a.seed, init=a.init,
               batch_seed=a.batch_seed, methods=np.array(["abf", "fr_uniform"]),
               estimator=a.estimator, n_bins=n_bins,
               hist_edges=(np.linspace(eb_XMIN, eb_XMAX, n_bins + 1) if n_bins else np.zeros(0)))
    for j, r in enumerate((abf, fr)):
        m = r["method"]
        for k in ("X", "Y", "S", "id", "p", "q"):
            out[f"{m}/snap_{k}"] = r["snap_" + k]
        out[f"{m}/snap_ev_die"] = r["snap_ev_die"]
        out[f"{m}/snap_ev_clone"] = r["snap_ev_clone"]
        # corrected read-out, per snapshot (movie) and per save (self-check against the engine)
        Fp_star = mean_force_at(np.asarray(r["snap_Sf"], float), np.asarray(r["snap_C"], float), h_star, dx, 1.0)
        Fc = np.cumsum(np.concatenate([np.zeros((Fp_star.shape[0], 1)),
                                       0.5 * (Fp_star[:, 1:] + Fp_star[:, :-1]) * dx], axis=1), axis=1)
        Fc = Fc - Fc[:, mask].mean(axis=1, keepdims=True)
        out[f"{m}/snap_F_star"] = Fc.astype(np.float32)
        out[f"{m}/snap_eF_star"] = e_f(Fp_star, F_ref, dx, mask)
        out[f"{m}/snap_F_bias"] = r["snap_F"]                       # what the walkers feel (h_bias / own bins)
        if n_bins:
            # own read-out: the engine's exact piecewise-linear PMF of the bins, centred on the
            # window (snap_F), scored exactly as the engine scores it (eb.l2_error on the window)
            assert r["abf_estimator"] == "histogram" and int(r["abf_n_bins"]) == n_bins
            F_own = np.asarray(r["snap_F"], np.float64)
            F_ref_c = F_ref - F_ref[mask].mean()
            eF_own = np.sqrt(np.mean((F_own[:, mask] - F_ref_c[mask]) ** 2, axis=1))
            dev = abs(eF_own[-1] - float(r["final_l2_f"]))
            assert dev < 1e-6, f"own read-out does not reproduce the engine's final e_F ({dev:.2e})"
            # ... and at every save step that is also a snapshot step (save_every 500 vs snapshots every 40)
            steps_save = np.rint(np.asarray(r["t"], float) / float(cfg.dt)).astype(int)
            common = np.isin(steps_save, out["snap_step"])
            sv = np.searchsorted(out["snap_step"], steps_save[common])
            assert common.sum() >= 2 and np.array_equal(out["snap_step"][sv], steps_save[common])
            dev_t = np.abs(eF_own[sv] - np.asarray(r["l2_f_t"], float)[common]).max()
            assert dev_t < 1e-6, f"own read-out does not reproduce the engine's e_F(t) at the saves ({dev_t:.2e})"
            out[f"{m}/snap_F_show"] = F_own.astype(np.float32)
            out[f"{m}/snap_eF_show"] = eF_own
            out[f"{m}/snap_Fp_bins"] = r["snap_Fp"]                 # P0 value at the nodes (the bin's Gamma_j)
        else:
            Fp_leg = mean_force_at(np.asarray(r["snap_Sf"], float), np.asarray(r["snap_C"], float), h_bias, dx, 1.0)
            dev = np.abs(e_f(Fp_leg, F_ref, dx, mask)[-1] - r["final_l2_f"])
            assert dev < 1e-9, f"offline read-out at h_bias does not reproduce the engine ({dev:.2e})"
            out[f"{m}/snap_F_show"] = out[f"{m}/snap_F_star"]
            out[f"{m}/snap_eF_show"] = out[f"{m}/snap_eF_star"]
        out[f"{m}/save_eF_star"] = e_f(mean_force_at(np.asarray(r["Sf_t"], float), np.asarray(r["C_t"], float),
                                                     h_star, dx, 1.0), F_ref, dx, mask)
        out[f"{m}/P_regions"] = np.asarray(r["P_regions"], float)
        out[f"{m}/Q_regions"] = np.asarray(r["Q_regions"], float)
        out[f"{m}/ess_t"] = np.asarray(r["ess_t"], float)
        out[f"{m}/n_die"] = r["n_die"]; out[f"{m}/n_clone"] = r["n_clone"]
        out[f"{m}/final_eF_star"] = float(out[f"{m}/snap_eF_star"][-1])
        out[f"{m}/final_eF_show"] = float(out[f"{m}/snap_eF_show"][-1])
        own = f"own {out[f'{m}/final_eF_show']:.5f} (kernel read-out at h_read* {out[f'{m}/final_eF_star']:.5f})" if n_bins \
            else f"at h_read* {out[f'{m}/final_eF_star']:.5f}"
        print(f"  {m:11s} e_F(T) {own}   events: {r['n_die']:.0f} deaths, "
              f"{r['n_clone']:.0f} births   min ESS/N {r['min_ess_frac']:.3f}", flush=True)
    d = 100 * (out["fr_uniform/final_eF_show"] / out["abf/final_eF_show"] - 1)
    I = {m: np.trapezoid(out[f"{m}/snap_eF_show"], out["snap_t"]) for m in ("abf", "fr_uniform")}
    cf, ci, csrc = CONFIRM[a.estimator]
    print(f"  this seed: final e_F {d:+.1f} %, integrated {100 * (I['fr_uniform'] / I['abf'] - 1):+.1f} % "
          f"(confirmation medians, {csrc}: {cf:+.1f} % / {ci:+.1f} %)")
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "movie_data.npz"), **out)
    print("saved", os.path.relpath(os.path.join(a.out, "movie_data.npz"), ROOT))


# ----------------------------------------------------------------------------------------------
# stage 2: render
# ----------------------------------------------------------------------------------------------
G = {}   # shared, read-only, populated before the fork


def _landscape(cfg):
    H, beta, r, s, w_out = (float(cfg[k]) for k in ("H", "beta", "r", "s", "omega_out"))
    w_in = r * w_out
    x = np.linspace(XMIN, XMAX, 721); y = np.linspace(YMIN, YMAX, 321)
    X, Y = np.meshgrid(x, y)
    omega = w_out + (w_in - w_out) * np.exp(-X * X / (2 * s * s))
    return X, Y, beta * (H * (X * X - 1) ** 2 + 0.5 * omega ** 2 * Y * Y)


def _load(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["cfg"] = json.loads(str(d["config_json"]))
    d["LX"], d["LY"], d["LV"] = _landscape(d["cfg"])
    x = d["x_grid"]; d["dx"] = float(x[1] - x[0])
    m = d["eval_mask"].astype(bool)
    Fr = d["F_ref"] - d["F_ref"][m].mean()
    d["F_ref_c"] = Fr
    d["beta"] = float(d["cfg"]["beta"]); d["N"] = int(d["cfg"]["N"])
    d["T"] = float((int(d["snap_step"][-1]) + 1) * float(d["cfg"]["dt"]))   # run length n_steps * dt
    d["estimator"] = str(d["estimator"]) if "estimator" in d else "kernel"
    d["n_bins"] = int(d["n_bins"]) if "n_bins" in d else 0
    for mn in ("abf", "fr_uniform"):          # files written before the estimator option: kernel read-out
        if f"{mn}/snap_F_show" not in d:
            d[f"{mn}/snap_F_show"] = d[f"{mn}/snap_F_star"]; d[f"{mn}/snap_eF_show"] = d[f"{mn}/snap_eF_star"]
    return d


def draw_frame(args):
    i, path = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    d = G["d"]; stride_win = G["event_window"]
    beta, N, T = d["beta"], d["N"], d["T"]
    t = float(d["snap_t"][i]); step = int(d["snap_step"][i])
    m = d["eval_mask"].astype(bool); x = d["x_grid"]
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.edgecolor": C_INK2, "axes.linewidth": 0.8, "xtick.color": C_INK2, "ytick.color": C_INK2,
        "axes.labelcolor": C_INK, "text.color": C_INK, "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(16, 9), dpi=120)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[3.1, 1.05, 1.55, 1.5], hspace=0.42, wspace=0.16,
                  left=0.055, right=0.985, top=0.875, bottom=0.07)
    gsb = gs[3, :].subgridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.18)

    hist = d["estimator"] == "histogram"
    fig.text(0.055, 0.968, "Entropic gateway:  ABF  vs  ABF + Fisher–Rao birth–death"
             + ("   (histogram mean-force estimator)" if hist else ""), fontsize=17,
             fontweight="bold", ha="left", va="center")
    fig.text(0.055, 0.937, f"same initial condition, same Langevin noise in both columns;  N = {N} walkers,  "
             f"$\\beta$ = {beta:g},  barrier {beta * float(d['cfg']['H']) + np.log(float(d['cfg']['r'])):.1f} $k_BT$  "
             f"({beta * float(d['cfg']['H']):.0f} energetic + {np.log(float(d['cfg']['r'])):.1f} entropic)",
             fontsize=11.5, color=C_INK2, ha="left", va="center")
    fig.text(0.985, 0.968, f"t = {t:6.2f} / {T:g}", fontsize=17, ha="right", va="center", family="monospace")
    fig.text(0.985, 0.937, f"step {step:,} of {int(d['snap_step'][-1]) + 1:,}", fontsize=11.5, color=C_INK2,
             ha="right", va="center", family="monospace")

    gamma_t = float(d["gamma"]) * (1 - np.exp(-t / (0.1 * T)))   # the engine's ramp: 10 % of the run
    for col, (mname, colour, label) in enumerate((("abf", C_ABF, "ABF"),
                                                  ("fr_uniform", C_FR, "ABF + Fisher–Rao"))):
        X = d[f"{mname}/snap_X"][i]; Y = d[f"{mname}/snap_Y"][i]
        # ---- landscape + walkers ------------------------------------------------------------
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(np.clip(d["LV"], 0, V_MAX_KT), origin="lower", extent=(XMIN, XMAX, YMIN, YMAX),
                  cmap="Greys", vmin=0, vmax=V_MAX_KT * 1.15, aspect="auto", interpolation="bilinear")
        ax.contour(d["LX"], d["LY"], d["LV"], levels=[1, 2, 4, 8, 16], colors="white", linewidths=0.55, alpha=0.9)
        ax.axvline(-X_BASIN, color=C_INK2, lw=0.6, ls=(0, (4, 3)), alpha=0.6)
        ax.axvline(X_BASIN, color=C_INK2, lw=0.6, ls=(0, (4, 3)), alpha=0.6)
        ax.scatter(X, Y, s=5, c=colour, alpha=0.55, linewidths=0, rasterized=True)
        if mname == "fr_uniform":
            for key, c, marker, sz in (("snap_ev_die", C_DIE, "x", 34), ("snap_ev_clone", C_BIRTH, "o", 40)):
                ev = d[f"{mname}/{key}"]
                sel = (ev[:, 0] > i - stride_win) & (ev[:, 0] <= i)
                if sel.any():
                    from matplotlib.colors import to_rgba
                    age = (i - ev[sel, 0]) / max(stride_win, 1)
                    rgba = np.tile(np.asarray(to_rgba(c)), (int(sel.sum()), 1))
                    rgba[:, 3] = np.clip(1.0 - 0.75 * age, 0.2, 1.0)
                    if marker == "o":
                        ax.scatter(ev[sel, 1], ev[sel, 2], s=sz, marker="o", facecolors="none",
                                   edgecolors=rgba, linewidths=1.3)
                    else:
                        ax.scatter(ev[sel, 1], ev[sel, 2], s=sz, marker="x", c=rgba, linewidths=1.3)
            n_die = int((d[f"{mname}/snap_ev_die"][:, 0] <= i).sum())
            n_cl = int((d[f"{mname}/snap_ev_clone"][:, 0] <= i).sum())
            ax.text(0.015, 0.965, f"rate $\\gamma(t)$ = {gamma_t:.2f}     "
                    f"deaths ×  {n_die:,}     births ○  {n_cl:,}", transform=ax.transAxes, fontsize=10.5,
                    ha="left", va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        else:
            ax.text(-1.0, 0.66, "basin $B_-$", ha="center", va="center", fontsize=11.5, color=C_INK2)
            ax.text(1.0, 0.66, "basin $B_+$", ha="center", va="center", fontsize=11.5, color=C_INK2)
            ax.annotate("entropic gateway", xy=(0.0, 0.05), xytext=(0.0, 0.5), ha="center", va="bottom",
                        fontsize=11.5, color="white",
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.0, shrinkA=0, shrinkB=2))
        frac_plus = float((X > X_BASIN).mean())
        ax.text(0.985, 0.04, f"in $B_+$: {100 * frac_plus:4.1f} %", transform=ax.transAxes, fontsize=10.5,
                ha="right", va="bottom", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        ax.set_title(label, color=colour, fontsize=15, fontweight="bold", pad=6)
        ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
        ax.set_ylabel("fast coordinate  $y$")
        ax.set_xticks(np.arange(-1.5, 1.6, 0.5)); ax.tick_params(labelbottom=False)

        # ---- histogram along xi ---------------------------------------------------------------
        ax = fig.add_subplot(gs[1, col])
        nb = 72; edges = np.linspace(XMIN, XMAX, nb + 1)
        h, _ = np.histogram(X, bins=edges)
        uni = N / nb
        ax.stairs(h / uni, edges, fill=True, color=colour, alpha=0.55, lw=0)
        ax.stairs(h / uni, edges, color=colour, lw=1.0)
        ax.axhline(1.0, color=C_INK, lw=1.0, ls=(0, (5, 3)))
        ax.text(XMAX - 0.03, 1.08, "uniform target", ha="right", va="bottom", fontsize=9.5, color=C_INK2)
        ax.set_xlim(XMIN, XMAX); ax.set_ylim(0, 4.0)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_ylabel("walkers / uniform", fontsize=10.5)
        ax.tick_params(labelbottom=False)
        ax.grid(True, axis="y", color=C_GRID, lw=0.6)
        ax.axvline(-X_BASIN, color=C_INK2, lw=0.6, ls=(0, (4, 3)), alpha=0.6)
        ax.axvline(X_BASIN, color=C_INK2, lw=0.6, ls=(0, (4, 3)), alpha=0.6)
        over = h / uni > 4.0
        if over.any():
            xc = 0.5 * (edges[1:] + edges[:-1])
            ax.scatter(xc[over], np.full(over.sum(), 3.85), marker="^", s=18, color=colour, clip_on=False)

        # ---- learned free energy --------------------------------------------------------------
        ax = fig.add_subplot(gs[2, col])
        Fs = d[f"{mname}/snap_F_show"][i]
        ax.plot(x[m], beta * d["F_ref_c"][m], color=C_REF, lw=1.3, ls=(0, (5, 3)), label="analytic $F$")
        if hist:
            lab = f"learned $\\hat F_t$  (histogram, {d['n_bins']} bins, $\\Delta\\xi$ = {(XMAX - XMIN) / d['n_bins']:g})"
        else:
            lab = f"learned $\\hat F_t$  (read-out $h^*$ = {float(d['h_read_star']):g})"
        ax.plot(x[m], beta * Fs[m], color=colour, lw=2.2, label=lab)
        eF = float(d[f"{mname}/snap_eF_show"][i])
        ax.text(0.015, 0.94, f"$e_F(t)$ = {beta * eF:.3f} $k_BT$", transform=ax.transAxes, fontsize=10.5,
                ha="left", va="top")
        ax.set_xlim(XMIN, XMAX)
        lo, hi = beta * d["F_ref_c"][m].min(), beta * d["F_ref_c"][m].max()
        ax.set_ylim(lo - 2.5, hi + 3.0)
        if hist:      # the estimator's bins, as tick marks along the bottom edge
            ax.vlines(d["hist_edges"], lo - 2.5, lo - 2.5 + 0.09 * (hi - lo + 5.5), color=colour, lw=0.7, alpha=0.7)
        ax.set_xlabel(r"collective variable  $\xi = x$")
        ax.set_ylabel("$\\beta F(\\xi)$  [$k_BT$]", fontsize=10.5)
        ax.grid(True, color=C_GRID, lw=0.6)
        ax.legend(loc="upper right", fontsize=9.5, frameon=False, ncol=1)
        ax.set_xticks(np.arange(-1.5, 1.6, 0.5))

    # ---- bottom strip: e_F(t) and occupancy of B_+ --------------------------------------------
    ax = fig.add_subplot(gsb[0, 0])
    ts = d["snap_t"]
    for mname, colour, label in (("abf", C_ABF, "ABF"), ("fr_uniform", C_FR, "ABF + FR")):
        e = beta * d[f"{mname}/snap_eF_show"]
        keep = ts > 0
        ax.plot(ts[keep], e[keep], color=colour, lw=1.2, alpha=0.28)
        past = keep & (ts <= t)
        if past.any():
            ax.plot(ts[past], e[past], color=colour, lw=2.4, label=label)
            ax.plot([t], [e[i]], "o", color=colour, ms=7)
    ax.axvline(t, color=C_INK2, lw=0.8, alpha=0.6)
    ax.set_yscale("log"); ax.set_xlim(0, T)
    e_all = beta * np.concatenate([d["abf/snap_eF_show"][1:], d["fr_uniform/snap_eF_show"][1:]])
    ax.set_ylim(e_all.min() * 0.6, e_all.max() * 1.6)
    ax.set_xlabel("time  $t$"); ax.set_ylabel("free-energy error  $e_F(t)$  [$k_BT$]", fontsize=10.5)
    ax.grid(True, which="both", color=C_GRID, lw=0.6)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    if t >= T - 1e-9:
        d_fin = 100 * (d["fr_uniform/snap_eF_show"][-1] / d["abf/snap_eF_show"][-1] - 1)
        ax.text(0.985, 0.55, f"final: {d_fin:+.0f} %", transform=ax.transAxes, ha="right", va="top",
                fontsize=11, color=C_FR, fontweight="bold")
    ax.text(0.015, 0.06, f"FR rate ramps in over the first {0.1 * T:.0f} time units", transform=ax.transAxes,
            fontsize=9.5, color=C_INK2, ha="left", va="bottom")

    ax = fig.add_subplot(gsb[0, 1])
    frac = G["frac_plus"]
    for mname, colour, label in (("abf", C_ABF, "ABF"), ("fr_uniform", C_FR, "ABF + FR")):
        f = frac[mname]
        ax.plot(ts, f, color=colour, lw=1.2, alpha=0.28)
        past = ts <= t
        ax.plot(ts[past], f[past], color=colour, lw=2.4)
        ax.plot([t], [f[i]], "o", color=colour, ms=7)
    ax.axhline((XMAX - X_BASIN) / (XMAX - XMIN), color=C_INK, lw=1.0, ls=(0, (5, 3)))
    ax.text(T * 0.98, (XMAX - X_BASIN) / (XMAX - XMIN) + 0.02, "uniform target", ha="right", va="bottom",
            fontsize=9.5, color=C_INK2)
    ax.axvline(t, color=C_INK2, lw=0.8, alpha=0.6)
    ax.set_xlim(0, T); ax.set_ylim(0, 0.6)
    ax.set_xlabel("time  $t$"); ax.set_ylabel("fraction of walkers in $B_+$", fontsize=10.5)
    ax.grid(True, color=C_GRID, lw=0.6)

    fig.savefig(path, dpi=120)
    plt.close(fig)
    return i


def render(a):
    from movie_common import render_movie
    d = _load(os.path.join(a.out, "movie_data.npz"))
    G["d"] = d
    G["event_window"] = a.event_window
    G["frac_plus"] = {mn: (d[f"{mn}/snap_X"] > X_BASIN).mean(axis=1) for mn in ("abf", "fr_uniform")}
    name = MOVIE[d["estimator"]]
    render_movie(draw_frame, len(d["snap_t"]), stride=a.stride, frames_dir=a.frames_dir or os.path.join(a.out, "frames"),
                 workers=a.workers, fps=a.fps, mp4=os.path.join(a.out, name + ".mp4"), only=a.only, hold_seconds=2.0,
                 gif=(os.path.join(a.out, name + ".gif") if a.gif else None), gif_fps=a.gif_fps, gif_width=a.gif_width,
                 keep_frames=a.keep_frames, root=ROOT)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--estimator", default="kernel", choices=["kernel", "histogram"])
    s.add_argument("--n-bins", type=int, default=0, help="histogram only; default: the frozen width (selected_bins.json)")
    s.add_argument("--seed", type=int, default=400)
    s.add_argument("--init", default="left", choices=["left", "one_right"])
    s.add_argument("--batch-seed", type=int, default=41_000)
    s.add_argument("--snap-every", type=int, default=40)
    s.add_argument("--out", default=None, help="default: results/gateway_movie[_histogram]")
    r = sub.add_parser("render")
    r.add_argument("--estimator", default="kernel", choices=["kernel", "histogram"], help="selects the default --out")
    r.add_argument("--out", default=None)
    r.add_argument("--frames-dir", default=None)
    r.add_argument("--stride", type=int, default=2, help="snapshots per frame")
    r.add_argument("--fps", type=int, default=30)
    r.add_argument("--event-window", type=int, default=4, help="flash events from the last k snapshots")
    r.add_argument("--workers", type=int, default=24)
    r.add_argument("--only", type=int, default=None, help="render one frame index (layout check)")
    r.add_argument("--keep-frames", action="store_true")
    r.add_argument("--gif", action="store_true")
    r.add_argument("--gif-fps", type=int, default=12)
    r.add_argument("--gif-width", type=int, default=960)
    a = ap.parse_args()
    if a.out is None:
        a.out = OUT_DIR[a.estimator]
    simulate(a) if a.stage == "simulate" else render(a)


if __name__ == "__main__":
    main()
