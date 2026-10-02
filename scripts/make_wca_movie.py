#!/usr/bin/env python
"""Movie: the WCA dimer under ABF and under ABF + Fisher-Rao birth-death, histogram estimator.

Two columns, one per arm, from two runs of the accepted corrected Case IX cell (beta 1, h 2, w 2,
M = 100 particles, N 1024 replicas, 120 000 steps of dt 0.002; configs/histogram_abf/campaign.json)
with the textbook histogram (P0) mean-force estimator at the campaign's frozen width
(configs/histogram_abf/selected_bins.json: 160 bins, Delta xi 0.00875).  Both arms start from the
same lattice configuration with the same random seed; the FR arm draws its birth-death randomness
from the same stream, so the two trajectories share their noise until the first Fisher-Rao
opportunity (fr_start_steps) and are paired in distribution afterwards (the WCA force kernel's
atomics are not bitwise reproducible anyway, see memory / docs/WCA_CORRECTED_CONFIRMATION.md).

Per column, top to bottom:
  * one replica's physical configuration (the dimer in its WCA solvent, re-centred on the dimer)
    next to every replica drawn as a bead on the reference free-energy profile F_ref(xi), with
    every Fisher-Rao death (red cross) and birth (green ring) flashed where it happened;
  * the replica histogram along xi against the uniform target;
  * the learned free energy (the histogram estimator's OWN exact piecewise-linear PMF, the
    reported quantity of the campaign) against the constrained-TI reference.
Bottom strip: e_F(t) for both arms with a moving cursor, and the fraction of replicas in the
stretched state.

    CUDA_VISIBLE_DEVICES=3 python scripts/make_wca_movie.py simulate            # ~6 min on one H200
    python scripts/make_wca_movie.py render --workers 24                         # frames + mp4

Data: results/wca_movie_histogram/movie_data.npz; movie: results/wca_movie_histogram/wca_abf_vs_fr_histogram.mp4.
The engine record behind it is ``store_snapshots`` (tests/test_wca_snapshots.py: bit-inert).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OUT_DIR = os.path.join(ROOT, "results", "wca_movie_histogram")
MOVIE = "wca_abf_vs_fr_histogram"
# confirmation medians (final e_F, integrated) of the histogram arms, 16 seeds 3100-3115
CONFIRM = (-50.5, -23.1, "16 seeds, results/histogram_abf/wca/confirmation/summary.json")

C_ABF = "#2a78d6"     # blue   (presentation figures)
C_FR = "#eb6834"      # orange (presentation figures)
C_DIE = "#d62728"
C_BIRTH = "#1f9e4a"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e4e3df"
C_REF = "#0b0b0b"
C_SOLV = "#c9c7c1"
C_SOLV_EDGE = "#8d8b85"


# ----------------------------------------------------------------------------------------------
# stage 1: simulate
# ----------------------------------------------------------------------------------------------
def simulate(a):
    import torch
    import wca_abffr_core as core
    import wca_phase_jobs as jobs
    from run_histogram_abf_wca import load_frozen, make_spec, CACHE, SELECTED, REFERENCE_NPZ, git_rev

    c, base, fr = load_frozen()
    n_bins = int(a.n_bins or json.load(open(SELECTED))["wca"]["selected_n_bins"])
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "pin exactly one GPU (CUDA_VISIBLE_DEVICES)"
    engines, out = {}, {}
    print(f"cell {c['cell']} run {c['run']} fr {fr}; histogram {n_bins} bins (Delta {1.4 / n_bins:.5f}); "
          f"seed {a.seed}; featured replica {a.featured}; snapshot every {a.snap_every} steps", flush=True)
    for name, method in (("hist_abf", "abf"), ("hist_fr_uniform", "fr_uniform")):
        sp = make_spec("movie", name, method, a.seed, c, fr, "histogram", n_bins)
        if a.fr_start is not None and method == "fr_uniform":
            # earlier FR start (user request 2026-10-01): a different stage label so the cached run id
            # cannot collide with the frozen-start movie; everything else is the frozen cell
            import dataclasses
            sp = dataclasses.replace(sp, stage=f"movie_fr{int(a.fr_start)}", fr_start_steps=int(a.fr_start))
        if a.smoke:       # layout / pipeline check only: tiny run, early FR, same physics and reference
            import dataclasses
            sp = dataclasses.replace(sp, stage="movie_smoke", n_steps=3000, n_replicas=64, save_every=500, fr_start_steps=500)
        eng = jobs.get_engine(sp, engines)
        sim = jobs.build_sim(sp, base)
        t0 = time.time()
        r = jobs.execute_run(sp, base, eng, cache_dir=CACHE, verbose=True, store_profiles=True,
                             store_snapshots=a.snap_every, snapshot_replicas=(a.featured,))
        assert "v2" in str(r.get("reference_label", "")), r.get("reference_label")
        assert r["abf_estimator"] == "histogram" and int(r["abf_n_bins"]) == n_bins and not r["had_nan"]
        print(f"  {name}: {time.time() - t0:.0f}s (sampler {r['runtime_seconds']:.0f}s)", flush=True)
        grid = np.asarray(r["grid"], float)
        ref_F, ref_mf = np.asarray(r["ref_free_energy"], float), np.asarray(r["ref_mean_force"], float)
        mask = core.eval_window_mask_np(grid, sim)
        # own read-out at snapshot cadence, exactly the campaign's e_F (align on the window, RMS)
        pmf = np.asarray(r["snap_pmf"], np.float64)
        eF = np.array([core.profile_l2_error_np(core.align_additive_constant_np(p, ref_F, grid, mask=mask), ref_F, grid, mask=mask)
                       for p in pmf])
        sv = np.searchsorted(r["snap_steps"], np.asarray(r["profile_steps"]))
        assert np.array_equal(r["snap_steps"][sv], np.asarray(r["profile_steps"]))
        dev = np.abs(eF[sv] - np.asarray(r["l2_f_t"], float)).max()
        assert dev < 1e-6, f"snapshot read-out does not reproduce the engine's e_F(t) ({dev:.2e})"
        assert abs(eF[-1] - float(r["l2_f"])) < 1e-6
        if not out:
            out.update(grid=grid, F_ref=ref_F, Fp_ref=ref_mf, eval_mask=mask, spec_json=str(r["spec_json"]),
                       sim_json=json.dumps({k: getattr(sim, k) for k in sim.__dataclass_fields__}, sort_keys=True, default=float),
                       params_json=json.dumps({k: getattr(jobs.build_params(sp), k) for k in jobs.build_params(sp).__dataclass_fields__}, sort_keys=True),
                       snap_t=np.asarray(r["snap_times"], float), snap_step=np.asarray(r["snap_steps"]),
                       save_t=np.asarray(r["profile_times"], float), n_bins=n_bins, hist_edges=np.asarray(r["hist_edges_abf"], float),
                       seed=a.seed, featured=a.featured, reference_label=str(r["reference_label"]),
                       reference_path=os.path.relpath(REFERENCE_NPZ, ROOT), git_rev=git_rev(), methods=np.array(["abf", "fr_uniform"]))
        if method == "fr_uniform":   # the FR arm's sim carries the FR start the frames annotate (t_fr)
            out["sim_json"] = json.dumps({k: getattr(sim, k) for k in sim.__dataclass_fields__}, sort_keys=True, default=float)
        z = np.asarray(r["snap_z"], np.float32)
        out[f"{method}/snap_z"] = z
        out[f"{method}/snap_wid"] = np.asarray(r["snap_wid"], np.int32)
        out[f"{method}/snap_F"] = pmf.astype(np.float32)
        out[f"{method}/snap_Fp_bins"] = np.asarray(r["snap_mf_bins"], np.float32)
        out[f"{method}/snap_counts"] = np.asarray(r["snap_counts"], np.float32)
        out[f"{method}/snap_q"] = np.asarray(r["snap_q"], np.float32)[:, 0]        # (n_snap, M, 2) of the featured replica
        out[f"{method}/snap_ev_die"] = np.asarray(r["snap_ev_die"], np.float32)
        out[f"{method}/snap_ev_birth"] = np.asarray(r["snap_ev_birth"], np.float32)
        out[f"{method}/snap_eF"] = eF
        out[f"{method}/save_eF"] = np.asarray(r["l2_f_t"], float)
        out[f"{method}/save_eFp"] = np.asarray(r["l2_fp_t"], float)
        out[f"{method}/frac_stretched"] = (z > sim.transition_hi).mean(axis=1)
        out[f"{method}/frac_transition"] = ((z >= sim.transition_lo) & (z <= sim.transition_hi)).mean(axis=1)
        for k in ("total_replacement_events", "min_ancestor_ess_window", "min_ancestor_ess", "max_ancestor_frac_over_time",
                  "n_round_trips", "n_barrier_crossings", "runtime_seconds", "l2_f", "l2_fp", "integrated_l2_f",
                  "bias_absmax", "bias_clip_fraction", "fr_event_fraction"):
            out[f"{method}/{k}"] = float(r[k])
        print(f"  {method:11s} e_F(T) own {eF[-1]:.5f}  I_F {float(r['integrated_l2_f']):.3f}  e_F'(T) {float(r['l2_fp']):.4f}  "
              f"replacements {int(r['total_replacement_events'])}  windowed ESS/N {float(r['min_ancestor_ess_window']) / sim.n_replicas:.3f}  "
              f"round trips {int(r['n_round_trips'])}", flush=True)
    d = 100 * (out["fr_uniform/l2_f"] / out["abf/l2_f"] - 1)
    di = 100 * (out["fr_uniform/integrated_l2_f"] / out["abf/integrated_l2_f"] - 1)
    cf, ci, csrc = CONFIRM
    print(f"  this seed: final e_F {d:+.1f} %, integrated {di:+.1f} % (confirmation medians, {csrc}: {cf:+.1f} % / {ci:+.1f} %)")
    # how long the two arms stay on the same trajectory (same seed; the FR arm consumes extra RNG from fr_start on)
    same = np.abs(out["abf/snap_z"] - out["fr_uniform/snap_z"]).max(axis=1)
    k = int(np.argmax(same > 1e-6)) if (same > 1e-6).any() else len(same)
    print(f"  arms identical (max |dz| < 1e-6) up to snapshot {k} (t = {out['snap_t'][min(k, len(same) - 1)]:.1f}); "
          f"FR starts at t = {float(jobs.build_sim(sp, base).fr_start_steps) * float(jobs.build_sim(sp, base).dt):.0f}")
    out["arms_identical_until_t"] = float(out["snap_t"][min(k, len(same) - 1)])
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "movie_data.npz"), **out)
    print("saved", os.path.relpath(os.path.join(a.out, "movie_data.npz"), ROOT))


# ----------------------------------------------------------------------------------------------
# stage 2: render
# ----------------------------------------------------------------------------------------------
G = {}   # shared, read-only, populated before the fork


def _load(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["sim"] = json.loads(str(d["sim_json"])); d["params"] = json.loads(str(d["params_json"]))
    p, s = d["params"], d["sim"]
    d["beta"] = float(p["beta"]); d["N"] = int(s["n_replicas"]); d["L"] = float(p["n_dim"] * p["a"])
    d["r0"] = 2.0 ** (1.0 / 6.0) * float(p["sigma"]); d["w"] = float(p["w"]); d["h"] = float(p["h"])
    d["M"] = int(p["n_dim"]) ** 2
    d["T"] = float(s["n_steps"] * s["dt"]); d["dt"] = float(s["dt"])
    d["t_fr"] = float(s["fr_start_steps"] * s["dt"]); d["t_warm"] = float(s["abf_warmup_steps"] * s["dt"])
    d["t_burn"] = float(s["estimator_burn_in_steps"] * s["dt"])
    d["z_min"], d["z_max"] = float(s["z_min"]), float(s["z_max"])
    d["z_lo"], d["z_hi"] = float(s["eval_z_lo"]), float(s["eval_z_hi"])
    d["tr_lo"], d["tr_hi"] = float(s["transition_lo"]), float(s["transition_hi"])
    m = d["eval_mask"].astype(bool); d["eval_mask"] = m
    d["F_ref_c"] = d["F_ref"] - d["F_ref"][m].mean()
    for mn in ("abf", "fr_uniform"):
        F = d[f"{mn}/snap_F"].astype(np.float64)
        d[f"{mn}/snap_F_c"] = F - F[:, m].mean(axis=1, keepdims=True)
        wid = d[f"{mn}/snap_wid"][:, int(d["featured"])]
        d[f"{mn}/replaced_at"] = np.flatnonzero(np.diff(wid) != 0) + 1      # snapshot indices where the featured slot got a new walker
    rng = np.random.default_rng(12345)
    d["jitter"] = rng.uniform(-1.0, 1.0, size=d["N"])                       # fixed per slot: beads do not shimmer
    return d


def _recentre(q, L):
    """Shift the periodic box so the dimer midpoint sits at the centre; returns (q', d01)."""
    d01 = q[0] - q[1]; d01 -= L * np.round(d01 / L)
    mid = q[1] + 0.5 * d01
    qq = np.mod(q - mid + 0.5 * L, L)
    qq[0] = 0.5 * L + 0.5 * d01; qq[1] = 0.5 * L - 0.5 * d01
    return qq, d01


def draw_frame(args):
    i, path = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import to_rgba
    d = G["d"]; win = G["event_window"]
    beta, N, T, L = d["beta"], d["N"], d["T"], d["L"]
    t = float(d["snap_t"][i]); step = int(d["snap_step"][i])
    m = d["eval_mask"]; x = d["grid"]
    zmin, zmax = d["z_min"], d["z_max"]
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.edgecolor": C_INK2, "axes.linewidth": 0.8, "xtick.color": C_INK2, "ytick.color": C_INK2,
        "axes.labelcolor": C_INK, "text.color": C_INK, "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(16, 9), dpi=120)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[3.0, 1.0, 1.55, 1.5], hspace=0.45, wspace=0.16,
                  left=0.055, right=0.985, top=0.875, bottom=0.07)
    gsb = gs[3, :].subgridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.18)

    fig.text(0.055, 0.968, "WCA dimer:  ABF  vs  ABF + Fisher–Rao birth–death   (histogram mean-force estimator)",
             fontsize=17, fontweight="bold", ha="left", va="center")
    fig.text(0.055, 0.937, f"same lattice initial condition and seed in both columns;  N = {N} replicas of {d['M']} particles,  "
             f"$\\beta$ = {beta:g},  bare dimer barrier $\\beta h$ = {beta * d['h']:g} $k_BT$;  "
             f"{int(d['n_bins'])} bins, $\\Delta\\xi$ = {(zmax - zmin) / int(d['n_bins']):.5f}",
             fontsize=11.5, color=C_INK2, ha="left", va="center")
    fig.text(0.985, 0.968, f"t = {t:6.1f} / {T:g}", fontsize=17, ha="right", va="center", family="monospace")
    fig.text(0.985, 0.937, f"step {step:,} of {int(d['snap_step'][-1]):,}", fontsize=11.5, color=C_INK2,
             ha="right", va="center", family="monospace")

    F_land = beta * d["F_ref_c"]
    y_lo, y_hi = F_land.min() - 1.6, F_land.max() + 2.2
    V_bare = lambda z: 16.0 * beta * d["h"] * z ** 2 * (1 - z) ** 2
    for col, (mname, colour, label) in enumerate((("abf", C_ABF, "ABF"),
                                                  ("fr_uniform", C_FR, "ABF + Fisher–Rao"))):
        z = d[f"{mname}/snap_z"][i]
        gst = gs[0, col].subgridspec(1, 2, width_ratios=[1.0, 1.75], wspace=0.10)
        # ---- physical configuration of the featured replica ----------------------------------
        ax = fig.add_subplot(gst[0, 0])
        q, d01 = _recentre(d[f"{mname}/snap_q"][i].astype(np.float64), L)
        rad = 0.5 * d["r0"]
        ax.add_collection(PatchCollection([Circle((xx, yy), rad) for xx, yy in q[2:]], facecolor=C_SOLV,
                                          edgecolor=C_SOLV_EDGE, linewidths=0.5))
        ax.plot([q[0, 0], q[1, 0]], [q[0, 1], q[1, 1]], color=colour, lw=2.4, solid_capstyle="round", zorder=3)
        ax.add_collection(PatchCollection([Circle((xx, yy), rad) for xx, yy in q[:2]], facecolor=colour,
                                          edgecolor="white", linewidths=0.8, zorder=4))
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        zf = float(z[int(d["featured"])])
        state = "compact" if zf < d["tr_lo"] else ("stretched" if zf > d["tr_hi"] else "transition")
        ax.text(0.03, 0.97, f"replica {int(d['featured'])}:  $\\xi$ = {zf:5.2f}  ({state})", transform=ax.transAxes,
                fontsize=10.5, ha="left", va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        ax.set_xlabel(f"periodic box {L:g} × {L:g}, re-centred on the dimer", fontsize=8.5, color=C_INK2, labelpad=3)
        rep = d[f"{mname}/replaced_at"]
        recent = rep[(rep > i - win) & (rep <= i)]
        if recent.size:
            age = (i - recent.max()) / max(win, 1)
            ax.text(0.5, 0.5, "replaced by a copy\nof another replica", transform=ax.transAxes, fontsize=12, fontweight="bold",
                    color=C_BIRTH, alpha=float(np.clip(1.0 - 0.75 * age, 0.25, 1.0)), ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_BIRTH, alpha=0.9))
        ax.set_title(label, color=colour, fontsize=15, fontweight="bold", pad=6, loc="left")

        # ---- every replica as a bead on the reference free-energy profile ----------------------
        ax = fig.add_subplot(gst[0, 1])
        zz = np.linspace(zmin, zmax, 400)
        ax.plot(zz, np.interp(zz, x, F_land), color="#9a9892", lw=3.0, zorder=1, label="reference $F(\\xi)$ (constrained TI)")
        ax.plot(zz, V_bare(zz) + np.interp(0.0, x, F_land), color=C_INK2, lw=1.0, ls=(0, (2, 2)), zorder=1,
                label="bare dimer potential $V_S(\\xi)$ + const")
        yb = np.interp(z, x, F_land) + 0.22 * d["jitter"]
        ax.scatter(z, yb, s=7, c=colour, alpha=0.55, linewidths=0, zorder=3, rasterized=True)
        for zl in (d["tr_lo"], d["tr_hi"]):
            ax.axvline(zl, color=C_INK2, lw=0.6, ls=(0, (4, 3)), alpha=0.6)
        for xx, lab_ in ((0.5 * (zmin + d["tr_lo"]), "compact"), (0.5 * (d["tr_lo"] + d["tr_hi"]), "transition"),
                         (0.5 * (d["tr_hi"] + zmax), "stretched")):
            ax.text(xx, y_hi - 0.15, lab_, ha="center", va="top", fontsize=10.5, color=C_INK2)
        if mname == "fr_uniform":
            for key, c, marker, sz in (("snap_ev_die", C_DIE, "x", 34), ("snap_ev_birth", C_BIRTH, "o", 40)):
                ev = d[f"{mname}/{key}"]
                sel = (ev[:, 0] > i - win) & (ev[:, 0] <= i)
                if sel.any():
                    age = (i - ev[sel, 0]) / max(win, 1)
                    rgba = np.tile(np.asarray(to_rgba(c)), (int(sel.sum()), 1))
                    rgba[:, 3] = np.clip(1.0 - 0.75 * age, 0.2, 1.0)
                    ye = np.interp(ev[sel, 1], x, F_land)
                    if marker == "o":
                        ax.scatter(ev[sel, 1], ye, s=sz, marker="o", facecolors="none", edgecolors=rgba, linewidths=1.3, zorder=5)
                    else:
                        ax.scatter(ev[sel, 1], ye, s=sz, marker="x", c=rgba, linewidths=1.3, zorder=5)
            n_die = int((d[f"{mname}/snap_ev_die"][:, 0] <= i).sum())
            n_cl = int((d[f"{mname}/snap_ev_birth"][:, 0] <= i).sum())
            on = "on" if t >= d["t_fr"] else f"starts at t = {d['t_fr']:g}"
            ax.text(0.985, 0.04, f"Fisher–Rao {on}     deaths ×  {n_die:,}     births ○  {n_cl:,}", transform=ax.transAxes,
                    fontsize=10.5, ha="right", va="bottom", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        else:
            ax.legend(loc="lower center", fontsize=9, frameon=True, framealpha=0.85, edgecolor="none")
        ax.set_xlim(zmin, zmax); ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("$\\beta F(\\xi)$  [$k_BT$]", fontsize=10.5)
        ax.tick_params(labelbottom=False)
        ax.grid(True, color=C_GRID, lw=0.6)
        frac_s = float((z > d["tr_hi"]).mean())

        # ---- histogram along xi ---------------------------------------------------------------
        ax = fig.add_subplot(gs[1, col])
        nb = 56; edges = np.linspace(zmin, zmax, nb + 1)
        h, _ = np.histogram(np.clip(z, zmin, zmax), bins=edges)
        uni = N / nb
        ax.stairs(h / uni, edges, fill=True, color=colour, alpha=0.55, lw=0)
        ax.stairs(h / uni, edges, color=colour, lw=1.0)
        ax.axhline(1.0, color=C_INK, lw=1.0, ls=(0, (5, 3)))
        ax.text(zmax - 0.01, 1.08, "uniform target", ha="right", va="bottom", fontsize=9.5, color=C_INK2)
        ax.text(0.015, 0.93, f"stretched ($\\xi$ > {d['tr_hi']:g}): {100 * frac_s:4.1f} %", transform=ax.transAxes, fontsize=10,
                ha="left", va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax.set_xlim(zmin, zmax); ax.set_ylim(0, 4.0)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_ylabel("replicas / uniform", fontsize=10.5)
        ax.tick_params(labelbottom=False)
        ax.grid(True, axis="y", color=C_GRID, lw=0.6)
        for zl in (d["tr_lo"], d["tr_hi"]):
            ax.axvline(zl, color=C_INK2, lw=0.6, ls=(0, (4, 3)), alpha=0.6)
        over = h / uni > 4.0
        if over.any():
            xc = 0.5 * (edges[1:] + edges[:-1])
            ax.scatter(xc[over], np.full(over.sum(), 3.85), marker="^", s=18, color=colour, clip_on=False)

        # ---- learned free energy --------------------------------------------------------------
        ax = fig.add_subplot(gs[2, col])
        Fs = d[f"{mname}/snap_F_c"][i]
        ax.axvspan(zmin, d["z_lo"], color=C_GRID, alpha=0.5, lw=0); ax.axvspan(d["z_hi"], zmax, color=C_GRID, alpha=0.5, lw=0)
        ax.plot(x, beta * d["F_ref_c"], color=C_REF, lw=1.3, ls=(0, (5, 3)), label="reference $F$ (constrained TI)")
        ax.plot(x, beta * Fs, color=colour, lw=2.2,
                label=f"learned $\\hat F_t$  (histogram, {int(d['n_bins'])} bins, own read-out)")
        eF = float(d[f"{mname}/snap_eF"][i])
        ax.text(0.015, 0.94, f"$e_F(t)$ = {beta * eF:.3f} $k_BT$", transform=ax.transAxes, fontsize=10.5, ha="left", va="top")
        ax.set_xlim(zmin, zmax)
        lo, hi = beta * d["F_ref_c"][m].min(), beta * d["F_ref_c"][m].max()
        ax.set_ylim(lo - 1.2, hi + 1.6)
        ax.set_xlabel(r"normalised dimer bond length  $\xi$ = (|q$_1$ − q$_2$| − r$_0$) / 2w")
        ax.set_ylabel("$\\beta F(\\xi)$  [$k_BT$]", fontsize=10.5)
        ax.grid(True, color=C_GRID, lw=0.6)
        ax.legend(loc="upper right", fontsize=9.5, frameon=True, framealpha=0.85, edgecolor="none", ncol=1)

    # ---- bottom strip: e_F(t) and the stretched fraction --------------------------------------
    ax = fig.add_subplot(gsb[0, 0])
    ts = d["snap_t"]
    for mname, colour, label in (("abf", C_ABF, "ABF"), ("fr_uniform", C_FR, "ABF + FR")):
        e = beta * d[f"{mname}/snap_eF"]
        keep = ts > 0
        ax.plot(ts[keep], e[keep], color=colour, lw=1.2, alpha=0.28)
        past = keep & (ts <= t)
        if past.any():
            ax.plot(ts[past], e[past], color=colour, lw=2.4, label=label)
            ax.plot([t], [e[i]], "o", color=colour, ms=7)
    ax.axvline(t, color=C_INK2, lw=0.8, alpha=0.6)
    for tt, lab_ in ((d["t_warm"], "ABF bias fully on"), (d["t_fr"], "FR starts")):
        ax.axvline(tt, color=C_INK2, lw=0.7, ls=(0, (2, 2)), alpha=0.7)
    ax.set_yscale("log"); ax.set_xlim(0, T)
    e_all = beta * np.concatenate([d["abf/snap_eF"][1:], d["fr_uniform/snap_eF"][1:]])
    ax.set_ylim(e_all.min() * 0.6, e_all.max() * 1.6)
    ax.text(d["t_warm"], e_all.max() * 1.25, " ABF bias fully on; reported estimator restarts (burn-in)", fontsize=8.5, color=C_INK2, ha="left", va="top")
    ax.text(d["t_fr"], e_all.min() * 0.75, " FR starts", fontsize=8.5, color=C_INK2, ha="left", va="bottom")
    ax.set_xlabel("time  $t$"); ax.set_ylabel("free-energy error  $e_F(t)$  [$k_BT$]", fontsize=10.5)
    ax.grid(True, which="both", color=C_GRID, lw=0.6)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    if t >= T - 1e-9:
        d_fin = 100 * (d["fr_uniform/snap_eF"][-1] / d["abf/snap_eF"][-1] - 1)
        ax.text(0.985, 0.55, f"final: {d_fin:+.0f} %", transform=ax.transAxes, ha="right", va="top",
                fontsize=11, color=C_FR, fontweight="bold")

    ax = fig.add_subplot(gsb[0, 1])
    target = (zmax - d["tr_hi"]) / (zmax - zmin)
    for mname, colour, label in (("abf", C_ABF, "ABF"), ("fr_uniform", C_FR, "ABF + FR")):
        f = d[f"{mname}/frac_stretched"]
        ax.plot(ts, f, color=colour, lw=1.2, alpha=0.28)
        past = ts <= t
        ax.plot(ts[past], f[past], color=colour, lw=2.4)
        ax.plot([t], [f[i]], "o", color=colour, ms=7)
    ax.axhline(target, color=C_INK, lw=1.0, ls=(0, (5, 3)))
    ax.text(T * 0.98, target + 0.015, "uniform target", ha="right", va="bottom", fontsize=9.5, color=C_INK2)
    ax.axvline(t, color=C_INK2, lw=0.8, alpha=0.6)
    ax.axvline(d["t_fr"], color=C_INK2, lw=0.7, ls=(0, (2, 2)), alpha=0.7)
    ax.set_xlim(0, T); ax.set_ylim(0, 0.6)
    ax.set_xlabel("time  $t$"); ax.set_ylabel("stretched fraction", fontsize=10.5)
    ax.grid(True, color=C_GRID, lw=0.6)

    fig.savefig(path, dpi=120)
    plt.close(fig)
    return i


def render(a):
    from movie_common import render_movie
    d = _load(os.path.join(a.out, "movie_data.npz"))
    G["d"] = d
    G["event_window"] = a.event_window
    render_movie(draw_frame, len(d["snap_t"]), stride=a.stride, frames_dir=a.frames_dir or os.path.join(a.out, "frames"),
                 workers=a.workers, fps=a.fps, mp4=os.path.join(a.out, MOVIE + ".mp4"), only=a.only, hold_seconds=2.0,
                 gif=(os.path.join(a.out, MOVIE + ".gif") if a.gif else None), gif_fps=a.gif_fps, gif_width=a.gif_width,
                 keep_frames=a.keep_frames, root=ROOT)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--seed", type=int, default=3200, help="fresh label (campaign.json lists the taken ones)")
    s.add_argument("--n-bins", type=int, default=0, help="default: the frozen width (selected_bins.json)")
    s.add_argument("--featured", type=int, default=0, help="replica slot drawn in the configuration panel")
    s.add_argument("--snap-every", type=int, default=50)
    s.add_argument("--fr-start", type=int, default=None, help="FR start step for the FR arm (default: the frozen 20 000)")
    s.add_argument("--smoke", action="store_true", help="tiny run for a pipeline check (not a result)")
    s.add_argument("--out", default=OUT_DIR)
    r = sub.add_parser("render")
    r.add_argument("--out", default=OUT_DIR)
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
    simulate(a) if a.stage == "simulate" else render(a)


if __name__ == "__main__":
    main()
