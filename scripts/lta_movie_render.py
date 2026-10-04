#!/usr/bin/env python
"""Frame renderer: ethane in LTA zeolite, ABF vs ABF + Fisher-Rao birth-death, histogram estimator.

Reads ``<out>/movie_data.npz`` (one seed, both arms, written by the simulate stage from the LTA
engine's bit-inert ``store_snapshots`` record) and draws one 1920x1080 frame per ``stride``
snapshots with the fork pool of scripts/movie_common.py, then encodes the mp4.

    python scripts/lta_movie_render.py --out results/lta_movie/T300 --only 500      # one frame (layout check)
    python scripts/lta_movie_render.py --out results/lta_movie/T300 --workers 24    # frames + mp4
    python scripts/lta_movie_render.py --selftest                                   # synthetic data: 2 frames + tiny mp4

Per column (left ABF, right ABF + Fisher-Rao), top to bottom:
  * framework + replica overlay: the x-y projection of the slab through the cage and window
    centres (two alpha-cages joined by the 8-ring window at x = a, edge windows at x = 0 and 2a),
    every replica's COM folded into that cell pair, the featured ethane as two beads and a bond,
    and (right column) every Fisher-Rao death (red cross) and birth (green ring) flashed where it
    happened; a card beside the drawing carries the legend, the featured replica's z_CV / region
    and the running death / birth counters;
  * the replica histogram along z_CV against the uniform target (window / neck / cage marked);
  * the learned free energy (the histogram estimator's OWN piecewise-linear PMF) against the
    umbrella-sampling reference, with e_F(t).
Bottom strip: (a) the reference decomposition F = U - TS with cursors at the featured replicas
(why the barrier is entropic), (b) the transverse (y, z) configuration cloud in the cage and in the
window with the 8-ring drawn underneath and the mean |cos theta| of the molecular axis (where the
entropy goes), (c) e_F(t) for both arms on a log axis with the window fractions.

Every frame-independent quantity (folded COMs, |u_x|, z_CV, kT-scaled profiles, framework slab
and ring coordinates, replacement indices) is precomputed in :func:`load_movie_data`; a frame only
slices arrays.  The module-level ``G`` is populated before the pool is forked (movie_common).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

MOVIE = "lta_abf_vs_fr_histogram"
OUT_DIR = os.path.join(ROOT, "results", "lta_movie", "T300")
FRAMEWORK_NPZ = os.path.join(ROOT, "cache", "lta", "framework.npz")
REFERENCE_NPZ = os.path.join(ROOT, "results", "uniform_campaign", "lta", "reference", "reference_T300.npz")

C_ABF = "#2a78d6"     # blue   (presentation figures)
C_FR = "#eb6834"      # orange (presentation figures)
C_DIE = "#d62728"
C_BIRTH = "#1f9e4a"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e4e3df"
C_REF = "#0b0b0b"
C_O = "#dddbd5"       # framework oxygen
C_O_EDGE = "#b3b1ab"
C_SI = "#8d8b85"      # framework silicon
C_U = "#1f77b4"       # U(z) in the decomposition
C_TS = "#d55e00"      # -TS(z) in the decomposition (vermilion)
C_CLOUD = "#3b3a37"

WINDOW_HALF, CAGE_MIN = 1.5, 4.0   # A: |z_CV| < 1.5 window, > 4 cage (LTASimConfig.window_half / cage_min)
SLAB_HALF = 2.5                    # A: framework atoms with |z mod a - a/2| below this are drawn in the overlay
RING_TOL = 1.0                     # A: O atoms with |x mod a| below this are the 8-ring of the cross-section
R_O, R_SI, R_BEAD = 1.1, 0.5, 0.9   # A: disc radii (data units)
FIG_W, FIG_H, DPI = 16.0, 9.0, 120  # 1920 x 1080

G = {}   # shared, read-only, populated before the fork


# ----------------------------------------------------------------------------------------------
# loading / precomputation
# ----------------------------------------------------------------------------------------------
def _folded_images(xy, period_x, period_y, margin=3.0):
    """Append the periodic images that fall just beyond the high x / y edges of the drawn cell."""
    out = [xy]
    for dx, dy in ((period_x, 0.0), (0.0, period_y), (period_x, period_y)):
        sel = np.ones(len(xy), bool)
        if dx:
            sel &= xy[:, 0] < margin
        if dy:
            sel &= xy[:, 1] < margin
        out.append(xy[sel] + np.array([dx, dy]))
    return np.concatenate(out)


def load_movie_data(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    for k in list(d):
        if np.asarray(d[k]).ndim == 0:
            d[k] = np.asarray(d[k]).item()
    for k in ("temperature", "kT", "beta", "a_pseudo", "box", "t_fr", "t_warm", "t_burn", "fr_rate", "T", "dt",
              "dF_kT", "dU_kT", "mTdS_kT"):
        d[k] = float(d[k])
    for k in ("n_bins", "seed", "featured", "N"):
        d[k] = int(d[k])
    d["arms_identical_until_t"] = float(d["arms_identical_until_t"]) if "arms_identical_until_t" in d else None
    d["methods"] = [str(m) for m in np.atleast_1d(d["methods"])]
    try:
        d["sim"] = json.loads(str(d.get("sim_json", "{}")))
    except (TypeError, ValueError):
        d["sim"] = {}
    a, beta = d["a_pseudo"], d["beta"]
    d["win_target"] = 2.0 * WINDOW_HALF / a                 # uniform-marginal window fraction, 3.0 / 11.919
    # reference profiles in kT, mean-centred on the engine grid
    Fr, Ur, Sr = (np.asarray(d[k], np.float64) for k in ("F_ref", "U_ref", "mTS_ref"))
    d["Fk"] = beta * (Fr - Fr.mean()); d["Uk"] = beta * (Ur - Ur.mean()); d["mTSk"] = beta * (Sr - Sr.mean())
    d["zg"] = np.asarray(d["z_grid"], np.float64)
    # framework: the slab through the cage / window centres in the x-y projection (+ edge images)
    for key, src in (("slab_o", "o_pos"), ("slab_si", "si_pos")):
        p = np.asarray(d[src], np.float64)
        dz = np.abs(np.mod(p[:, 2], a) - 0.5 * a)
        order = np.argsort(-dz[dz < SLAB_HALF])                     # atoms nearest the mid-plane drawn last (on top)
        pm = p[dz < SLAB_HALF][order]
        xy = np.stack([np.mod(pm[:, 0], 2 * a), np.mod(pm[:, 1], a)], axis=1)
        d[key] = _folded_images(xy, 2 * a, a)
    # the 8-ring: O atoms on the window plane, in the transverse (y mod a, z mod a) cross-section
    o = np.asarray(d["o_pos"], np.float64)
    dx = np.mod(o[:, 0] + 0.5 * a, a) - 0.5 * a
    ring = np.stack([np.mod(o[:, 1], a), np.mod(o[:, 2], a)], axis=1)[np.abs(dx) < RING_TOL]
    d["ring_yz"] = np.unique(np.round(ring, 3), axis=0)
    # per arm
    for m in d["methods"]:
        q = np.asarray(d[f"{m}/snap_q"], np.float32)                      # (n_snap, N, 2, 3) unwrapped
        com = q.mean(axis=2)
        d[f"{m}/com_xy"] = np.stack([np.mod(com[..., 0], 2 * a), np.mod(com[..., 1], a)], axis=-1)
        d[f"{m}/com_yz"] = np.stack([np.mod(com[..., 1], a), np.mod(com[..., 2], a)], axis=-1)
        bond = q[:, :, 0, :] - q[:, :, 1, :]
        d[f"{m}/abs_ux"] = np.abs(bond[..., 0]) / np.maximum(np.linalg.norm(bond, axis=-1), 1e-9)
        d[f"{m}/zcv"] = (np.asarray(d[f"{m}/snap_phi"], np.float64) * a / (2 * np.pi)).astype(np.float32)
        F = np.asarray(d[f"{m}/snap_F"], np.float64)
        d[f"{m}/Fk"] = beta * (F - F.mean(axis=1, keepdims=True))
        d[f"{m}/eFk"] = beta * np.asarray(d[f"{m}/snap_eF"], np.float64)
        wid = np.asarray(d[f"{m}/snap_wid"])[:, d["featured"]]
        d[f"{m}/replaced_at"] = np.flatnonzero(np.diff(wid) != 0) + 1   # snapshots where the featured slot got a new walker
        for key in ("snap_ev_die", "snap_ev_birth"):
            ev = np.asarray(d[f"{m}/{key}"], np.float64).reshape(-1, 5)
            d[f"{m}/{key}"] = ev
            d[f"{m}/{key}_xy"] = np.stack([np.mod(ev[:, 2], 2 * a), np.mod(ev[:, 3], a)], axis=1)
        d[f"{m}/snap_crossings"] = np.asarray(d[f"{m}/snap_crossings"]).astype(np.int64)
        d[f"{m}/frac_window"] = np.asarray(d[f"{m}/frac_window"], np.float64)
    e_all = np.concatenate([d[f"{m}/eFk"][1:] for m in d["methods"]])
    e_all = e_all[np.isfinite(e_all) & (e_all > 0)]
    d["eF_lim"] = (float(e_all.min()) * 0.6, float(e_all.max()) * 1.6) if e_all.size else (1e-2, 10.0)
    return d


# ----------------------------------------------------------------------------------------------
# one frame
# ----------------------------------------------------------------------------------------------
def draw_frame(args):
    i, path = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    d = G["d"]; win = int(G.get("event_window", 4)); cw = int(G.get("cloud_window", 40))
    a, N, T = d["a_pseudo"], d["N"], d["T"]
    zh = 0.5 * a
    t = float(d["snap_t"][i]); step = int(d["snap_step"][i])
    zg, Fk = d["zg"], d["Fk"]
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.edgecolor": C_INK2, "axes.linewidth": 0.8, "xtick.color": C_INK2, "ytick.color": C_INK2,
        "axes.labelcolor": C_INK, "text.color": C_INK, "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)

    def ax_in(x, y, w, h, **kw):      # axes rectangle in inches from the bottom-left corner
        return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H], **kw)

    box = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88)

    # ---- header -------------------------------------------------------------------------------
    fig.text(0.047, 0.966, f"Ethane in LTA zeolite ({d['temperature']:g} K):  ABF  vs  ABF + Fisher–Rao birth–death   "
             "(histogram mean-force estimator)", fontsize=15, fontweight="bold", ha="left", va="center")
    same = (f"same seed in both columns (identical until t = {d['arms_identical_until_t']:.1f})"
            if d["arms_identical_until_t"] is not None else "same seed in both columns")
    fig.text(0.047, 0.938, f"N = {N} independent replicas (one ethane each) in rigid all-silica LTA (a = {a:.3f} Å, periodic "
             f"2×2×2 cells);  CV = ethane COM along the cage–window–cage axis",
             fontsize=10, color=C_INK2, ha="left", va="center")
    fig.text(0.047, 0.914, f"barrier $\\Delta F^\\ddagger$ = {d['dF_kT']:.1f} $k_BT$, of which $-T\\Delta S^\\ddagger$ = "
             f"{d['mTdS_kT']:.1f} $k_BT$ is entropic ($\\Delta U^\\ddagger$ = {d['dU_kT']:.1f} $k_BT$);  "
             f"histogram estimator, {d['n_bins']} bins;  {same}, Fisher–Rao from t = {d['t_fr']:g} at rate {d['fr_rate']:g}",
             fontsize=10, color=C_INK2, ha="left", va="center")
    fig.text(0.985, 0.966, f"t = {t:5.1f} / {T:g}", fontsize=17, ha="right", va="center", family="monospace")
    fig.text(0.985, 0.938, f"step {step:,} of {int(d['snap_step'][-1]):,}", fontsize=11, color=C_INK2,
             ha="right", va="center", family="monospace")

    # ---- geometry (inches) -----------------------------------------------------------------------
    COL_X, COL_W = (0.75, 8.35), 6.9
    Y_OV, H_OV = 5.22, 2.65
    Y_HI, H_HI = 3.95, 0.72
    Y_FE, H_FE = 2.68, 1.12
    Y_BS, H_BS = 0.55, 1.35
    xr = (-0.6, 2 * a + 0.6); yr = (-0.6, a + 1.35)
    W_OV = H_OV * (xr[1] - xr[0]) / (yr[1] - yr[0])

    feat_z = {}
    for col, (m, colour, label) in enumerate((("abf", C_ABF, "ABF"), ("fr_uniform", C_FR, "ABF + Fisher–Rao"))):
        x0 = COL_X[col]
        zcv = d[f"{m}/zcv"][i]
        zf = float(zcv[d["featured"]]); feat_z[m] = zf
        region = "window" if abs(zf) < WINDOW_HALF else ("cage" if abs(zf) > CAGE_MIN else "neck")

        # ---- framework + replica overlay ---------------------------------------------------------
        ax = ax_in(x0, Y_OV, W_OV, H_OV)
        ax.set_anchor("W")
        ax.add_collection(PatchCollection([Circle(tuple(p), R_O) for p in d["slab_o"]], facecolor=C_O,
                                          edgecolor=C_O_EDGE, linewidths=0.4, alpha=0.85, zorder=1))
        ax.add_collection(PatchCollection([Circle(tuple(p), R_SI) for p in d["slab_si"]], facecolor=C_SI,
                                          edgecolor="none", alpha=0.9, zorder=2))
        for xw in (0.0, a, 2 * a):
            ax.axvline(xw, color=C_INK2, lw=0.7, ls=(0, (1, 3)), alpha=0.8, zorder=0)
        for xx, lab_ in ((0.5 * a, "α-cage"), (a, "window"), (1.5 * a, "α-cage")):
            ax.text(xx, a + 0.4, lab_, ha="center", va="bottom", fontsize=9.5, color=C_INK2)
        xy = d[f"{m}/com_xy"][i]
        ax.scatter(xy[:, 0], xy[:, 1], s=6, c=colour, alpha=0.55, linewidths=0, zorder=3, rasterized=True)
        q = np.asarray(d[f"{m}/snap_q"][i, d["featured"]], np.float64)          # (2, 3) unwrapped beads
        com = q.mean(axis=0)
        qq = q - np.array([np.floor(com[0] / (2 * a)) * 2 * a, np.floor(com[1] / a) * a, 0.0])
        ax.plot(qq[:, 0], qq[:, 1], color=colour, lw=3.2, solid_capstyle="round", zorder=5)
        ax.add_collection(PatchCollection([Circle((p[0], p[1]), R_BEAD) for p in qq], facecolor=colour,
                                          edgecolor="white", linewidths=1.0, zorder=6))
        n_die = n_cl = 0
        if m == "fr_uniform":
            for key, c, marker, sz in (("snap_ev_die", C_DIE, "x", 40), ("snap_ev_birth", C_BIRTH, "o", 48)):
                ev = d[f"{m}/{key}"]; exy = d[f"{m}/{key}_xy"]
                sel = (ev[:, 0] > i - win) & (ev[:, 0] <= i)
                if sel.any():
                    age = (i - ev[sel, 0]) / max(win, 1)
                    rgba = np.tile(np.asarray(to_rgba(c)), (int(sel.sum()), 1))
                    rgba[:, 3] = np.clip(1.0 - 0.75 * age, 0.2, 1.0)
                    if marker == "o":
                        ax.scatter(exy[sel, 0], exy[sel, 1], s=sz, marker="o", facecolors="none", edgecolors=rgba,
                                   linewidths=1.4, zorder=7)
                    else:
                        ax.scatter(exy[sel, 0], exy[sel, 1], s=sz, marker="x", c=rgba, linewidths=1.4, zorder=7)
            n_die = int((d[f"{m}/snap_ev_die"][:, 0] <= i).sum())
            n_cl = int((d[f"{m}/snap_ev_birth"][:, 0] <= i).sum())
            rep = d[f"{m}/replaced_at"]
            recent = rep[(rep > i - win) & (rep <= i)]
            if recent.size:
                age = (i - recent.max()) / max(win, 1)
                ax.text(0.5, 0.5, "featured replica replaced\nby a copy of another replica", transform=ax.transAxes,
                        fontsize=12, fontweight="bold", color=C_BIRTH, alpha=float(np.clip(1.0 - 0.75 * age, 0.25, 1.0)),
                        ha="center", va="center", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_BIRTH, alpha=0.9))
        ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 5, 10, 15, 20]); ax.set_yticks([0, 5, 10])
        ax.tick_params(labelsize=9)
        ax.set_xlabel(f"x (Å) along the cage–window–cage axis\noverlay of {N} independent replicas; x folded into the two cages "
                      "of the periodic box, y into one cell", fontsize=8.8, color=C_INK2, loc="left", labelpad=3)
        ax.set_ylabel("y (Å)", fontsize=9.5, labelpad=2)
        ax.set_title(label, color=colour, fontsize=15, fontweight="bold", pad=5, loc="left")

        # ---- card: legend, featured replica, counters -------------------------------------------
        cx = x0 + W_OV + 0.12
        cax = ax_in(cx, Y_OV, COL_W - (W_OV + 0.12), H_OV); cax.axis("off")
        handles = [Line2D([], [], marker="o", ls="none", mfc=C_O, mec=C_O_EDGE, ms=9, label="framework O (slab)"),
                   Line2D([], [], marker="o", ls="none", mfc=C_SI, mec="none", ms=5, label="framework Si"),
                   Line2D([], [], marker="o", ls="none", mfc=colour, mec="none", ms=4, alpha=0.7, label="replica COM"),
                   Line2D([], [], marker="o", ls="-", color=colour, lw=2.5, mfc=colour, mec="white", ms=8,
                          label="featured ethane")]
        if m == "fr_uniform":
            handles += [Line2D([], [], marker="x", ls="none", color=C_DIE, ms=7, mew=1.4, label="death"),
                        Line2D([], [], marker="o", ls="none", mfc="none", mec=C_BIRTH, ms=7, mew=1.4, label="birth (copy)")]
        cax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False, handlelength=1.2, handletextpad=0.6,
                   borderaxespad=0.0, labelspacing=0.4)
        cax.text(0.0, 0.40, f"featured replica (slot {d['featured']})\n$z_{{CV}}$ = {zf:+5.2f} Å   ({region})",
                 transform=cax.transAxes, fontsize=10, ha="left", va="top", color=C_INK,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colour, lw=1.0))
        if m == "fr_uniform":
            on = "on" if t >= d["t_fr"] else f"starts at t = {d['t_fr']:g}"
            cax.text(0.0, 0.17, f"Fisher–Rao {on}\ndeaths ×  {n_die:,}      births ○  {n_cl:,}", transform=cax.transAxes,
                     fontsize=10, ha="left", va="top", color=C_INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_INK2, lw=0.6))

        # ---- replica histogram along z_CV -------------------------------------------------------
        ax = ax_in(x0, Y_HI, COL_W, H_HI)
        nb = 60; edges = np.linspace(-zh, zh, nb + 1)
        h, _ = np.histogram(np.clip(zcv, -zh, zh - 1e-6), bins=edges)
        uni = N / nb
        ax.stairs(h / uni, edges, fill=True, color=colour, alpha=0.55, lw=0)
        ax.stairs(h / uni, edges, color=colour, lw=1.0)
        ax.axhline(1.0, color=C_INK, lw=1.0, ls=(0, (5, 3)))
        ax.text(zh - 0.1, 1.1, "uniform target", ha="right", va="bottom", fontsize=9, color=C_INK2,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
        for zl in (-CAGE_MIN, -WINDOW_HALF, WINDOW_HALF, CAGE_MIN):
            ax.axvline(zl, color=C_INK2, lw=0.6, ls=(0, (4, 3)), alpha=0.6)
        for xx, lab_ in ((-5.0, "cage"), (0.0, "window"), (5.0, "cage")):
            ax.text(xx, 0.12, lab_, ha="center", va="bottom", fontsize=8.5, color=C_INK2, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
        frac_w = float((np.abs(zcv) < WINDOW_HALF).mean()); nx = int(d[f"{m}/snap_crossings"][i])
        ax.text(0.012, 0.92, f"in window (|z| < {WINDOW_HALF:g} Å): {100 * frac_w:4.1f} %", transform=ax.transAxes,
                fontsize=9.5, ha="left", va="top", bbox=box)
        ax.text(0.988, 0.92, f"cage→cage crossings so far: {nx:,}", transform=ax.transAxes, fontsize=9.5,
                ha="right", va="top", bbox=box)
        ax.set_xlim(-zh, zh); ax.set_ylim(0, 4.0); ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_ylabel("replicas / uniform", fontsize=9.5)
        ax.tick_params(labelbottom=False, labelsize=9)
        ax.grid(True, axis="y", color=C_GRID, lw=0.6)
        over = h / uni > 4.0
        if over.any():
            xc = 0.5 * (edges[1:] + edges[:-1])
            ax.scatter(xc[over], np.full(int(over.sum()), 3.85), marker="^", s=18, color=colour, clip_on=False)

        # ---- learned free energy ----------------------------------------------------------------
        ax = ax_in(x0, Y_FE, COL_W, H_FE)
        ax.axvspan(-WINDOW_HALF, WINDOW_HALF, color=C_GRID, alpha=0.55, lw=0)
        for zl in (-CAGE_MIN, CAGE_MIN):
            ax.axvline(zl, color=C_INK2, lw=0.6, ls=(0, (4, 3)), alpha=0.6)
        ax.plot(zg, Fk, color=C_REF, lw=1.3, ls=(0, (5, 3)), label="reference $\\beta F$ (umbrella sampling)")
        ax.plot(zg, d[f"{m}/Fk"][i], color=colour, lw=2.2,
                label=f"learned $\\beta\\hat F_t$ (histogram, {d['n_bins']} bins)")
        ax.text(0.012, 0.93, f"$e_F(t)$ = {float(d[f'{m}/eFk'][i]):.2f} $k_BT$", transform=ax.transAxes, fontsize=10.5,
                ha="left", va="top", bbox=box)
        ax.set_xlim(-zh, zh); ax.set_ylim(Fk.min() - 1.5, Fk.max() + 2.2)
        ax.set_xlabel("$z_{CV}$ (Å)   ethane COM along the axis   [window at 0, cage centres at ±a/2]", fontsize=10, labelpad=2)
        ax.set_ylabel("$\\beta F(z)$  [$k_BT$]", fontsize=9.5)
        ax.tick_params(labelsize=9)
        ax.grid(True, color=C_GRID, lw=0.6)
        ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.88, edgecolor="none", borderaxespad=0.3,
                  handlelength=1.8)

    # ---- (a) thermodynamic decomposition ---------------------------------------------------------
    ax = ax_in(0.75, Y_BS, 4.0, H_BS)
    Uk, mTSk = d["Uk"], d["mTSk"]
    ax.plot(zg, Fk, color=C_INK, lw=1.6, label="$\\beta F(z)$")
    ax.plot(zg, Uk, color=C_U, lw=1.3, ls="--", label="$\\beta U(z)$")
    ax.plot(zg, mTSk, color=C_TS, lw=1.3, ls="-.", label="$-TS(z)/k_BT$")
    ax.axvline(feat_z["abf"], ymax=0.70, color=C_ABF, lw=1.6, alpha=0.9, label="featured replica, ABF")
    ax.axvline(feat_z["fr_uniform"], ymax=0.70, color=C_FR, lw=1.1, alpha=0.9, label="featured replica, FR")
    y_lo = min(Fk.min(), Uk.min(), mTSk.min()) - 1.0; y_hi = max(Fk.max(), Uk.max(), mTSk.max()) + 3.6
    ax.annotate(f"window: few configurations\n→ $-T\\Delta S^\\ddagger$ = {d['mTdS_kT']:.1f} $k_BT$",
                xy=(zg[np.argmax(mTSk)], mTSk.max()), xytext=(zh * 0.97, y_hi - 0.3), ha="right", va="top",
                fontsize=8.5, color=C_TS, arrowprops=dict(arrowstyle="-", color=C_TS, lw=0.7, alpha=0.8))
    ax.text(1.8, Uk.max() + 0.25, f"$\\Delta U^\\ddagger$ = {d['dU_kT']:.1f} $k_BT$", color=C_U, fontsize=8.5,
            ha="left", va="bottom")
    ax.set_xlim(-zh, zh); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("$z_{CV}$ (Å)", fontsize=9.5, labelpad=1); ax.set_ylabel("$k_BT$", fontsize=9.5, labelpad=2)
    ax.tick_params(labelsize=9)
    ax.grid(True, color=C_GRID, lw=0.6)
    ax.legend(loc="upper left", fontsize=8, frameon=False, handlelength=1.6, borderaxespad=0.3, labelspacing=0.2)
    ax.set_title("(a) why the barrier is entropic:  F = U − TS  (reference)", fontsize=10, loc="left", pad=4)

    # ---- (b) configuration cloud: cage vs window -------------------------------------------------
    lo = max(0, i - cw + 1)
    yz = np.concatenate([d[f"{m}/com_yz"][lo:i + 1].reshape(-1, 2) for m in ("abf", "fr_uniform")])
    zc = np.concatenate([d[f"{m}/zcv"][lo:i + 1].ravel() for m in ("abf", "fr_uniform")])
    ux = np.concatenate([d[f"{m}/abs_ux"][lo:i + 1].ravel() for m in ("abf", "fr_uniform")])
    S = H_BS; bx0 = 5.45
    for k, (mask, lab_) in enumerate(((np.abs(zc) > CAGE_MIN, f"in cage (|$z_{{CV}}$| > {CAGE_MIN:g} Å)"),
                                      (np.abs(zc) < WINDOW_HALF, f"in window (|$z_{{CV}}$| < {WINDOW_HALF:g} Å)"))):
        ax = ax_in(bx0 + k * (S + 0.1), Y_BS, S, S)
        ax.add_collection(PatchCollection([Circle(tuple(p), R_O) for p in d["ring_yz"]], facecolor=C_O,
                                          edgecolor=C_O_EDGE, linewidths=0.5, zorder=1))
        pts = yz[mask]
        if len(pts):
            if len(pts) > 6000:
                pts = pts[::int(np.ceil(len(pts) / 6000))]
            ax.scatter(pts[:, 0], pts[:, 1], s=2.5, c=C_CLOUD, alpha=0.3, linewidths=0, zorder=3, rasterized=True)
            ax.text(0.04, 0.05, f"⟨|cos θ|⟩ = {float(ux[mask].mean()):.2f}", transform=ax.transAxes, fontsize=8.5,
                    ha="left", va="bottom", bbox=box, zorder=5)
        else:
            ax.text(0.5, 0.5, "none yet", transform=ax.transAxes, fontsize=10, ha="center", va="center", color=C_INK2)
        ax.text(0.04, 0.96, lab_, transform=ax.transAxes, fontsize=8.5, ha="left", va="top", bbox=box, zorder=5)
        ax.set_xlim(0, a); ax.set_ylim(0, a); ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("y (Å)", fontsize=8.5, labelpad=2)
        if k == 0:
            ax.set_ylabel("z (Å)", fontsize=8.5, labelpad=2)
            ax.set_title("(b) where the entropy goes: positions and orientations", fontsize=10, loc="left", pad=4)
    fig.text((bx0 + S + 0.05) / FIG_W, 0.1 / FIG_H, f"COM cross-sections pooled over both arms, last {min(cw, i + 1)} snapshots; "
             "8-ring O in grey", fontsize=8, color=C_INK2, ha="center", va="bottom")

    # ---- (c) e_F(t) for both arms ---------------------------------------------------------------
    ax = ax_in(9.45, Y_BS, 5.8, H_BS)
    ts = d["snap_t"]
    for m, colour, lab_ in (("abf", C_ABF, "ABF"), ("fr_uniform", C_FR, "ABF + FR")):
        e = d[f"{m}/eFk"]
        keep = (ts > 0) & np.isfinite(e) & (e > 0)
        ax.plot(ts[keep], e[keep], color=colour, lw=1.2, alpha=0.28)
        past = keep & (ts <= t)
        if past.any():
            ax.plot(ts[past], e[past], color=colour, lw=2.4, label=lab_)
            ax.plot([t], [e[i]], "o", color=colour, ms=7)
    ax.axvline(t, color=C_INK2, lw=0.8, alpha=0.6)
    for tt in (d["t_warm"], d["t_fr"]):
        ax.axvline(tt, color=C_INK2, lw=0.7, ls=(0, (2, 2)), alpha=0.7)
    ax.set_yscale("log"); ax.set_xlim(0, T); ax.set_ylim(*d["eF_lim"])
    ax.text(d["t_warm"], d["eF_lim"][1] * 0.93, " ABF bias fully on; reported estimator restarts", fontsize=8.5,
            color=C_INK2, ha="left", va="top")
    ax.text(d["t_fr"], d["eF_lim"][0] * 1.25, " FR starts", fontsize=8.5, color=C_INK2, ha="left", va="bottom")
    ax.set_xlabel("time  $t$", fontsize=9.5, labelpad=1)
    ax.set_ylabel("$e_F(t)$  [$k_BT$]", fontsize=9.5, labelpad=2)
    ax.tick_params(labelsize=9)
    ax.grid(True, which="both", color=C_GRID, lw=0.6)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=9.5, frameon=False, ncol=2, columnspacing=1.2, borderaxespad=0.3)
    fw = {m: float(d[f"{m}/frac_window"][i]) for m in ("abf", "fr_uniform")}
    ax.text(1.0, 1.03, f"window fraction:  ABF {fw['abf']:.2f}  |  FR {fw['fr_uniform']:.2f}   (uniform {d['win_target']:.2f})",
            transform=ax.transAxes, fontsize=9, color=C_INK2, ha="right", va="bottom")
    if t >= T - 1e-9:
        d_fin = 100 * (d["fr_uniform/eFk"][-1] / d["abf/eFk"][-1] - 1)
        ax.text(0.985, 0.70, f"final: {d_fin:+.0f} %", transform=ax.transAxes, ha="right", va="top",
                fontsize=11, color=C_FR, fontweight="bold")
    ax.set_title("(c) free-energy error $e_F(t)$", fontsize=10, loc="left", pad=4)

    if G.get("debug_bbox"):
        _report_clipping(fig, i)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return i


def _report_clipping(fig, i):
    """Layout check: list every text artist whose extent leaves the canvas (selftest only)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches() * fig.dpi
    arts = list(fig.texts)
    for ax in fig.axes:
        arts += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.get_legend() is not None:
            arts.append(ax.get_legend())
    for art in arts:
        try:
            bb = art.get_window_extent(renderer=r)
        except Exception:
            continue
        if not art.get_visible() or bb.width == 0:
            continue
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            txt = art.get_text() if hasattr(art, "get_text") else "legend"
            print(f"  frame {i}: OUTSIDE canvas: {txt[:60]!r}  bbox x[{bb.x0:.0f},{bb.x1:.0f}] y[{bb.y0:.0f},{bb.y1:.0f}]")


# ----------------------------------------------------------------------------------------------
# render
# ----------------------------------------------------------------------------------------------
def render(out_dir, *, stride=3, fps=30, workers=24, event_window=4, cloud_window=40, only=None, keep_frames=False,
           gif=False, movie_name=MOVIE):
    from movie_common import render_movie
    d = load_movie_data(os.path.join(out_dir, "movie_data.npz"))
    G["d"] = d
    G["event_window"] = int(event_window)
    G["cloud_window"] = int(cloud_window)
    return render_movie(draw_frame, len(d["snap_t"]), stride=stride, frames_dir=os.path.join(out_dir, "frames"),
                        workers=workers, fps=fps, mp4=os.path.join(out_dir, movie_name + ".mp4"), only=only,
                        hold_seconds=2.0, gif=(os.path.join(out_dir, movie_name + ".gif") if gif else None),
                        keep_frames=keep_frames, root=ROOT)


# ----------------------------------------------------------------------------------------------
# self-test: a consistent synthetic movie_data.npz (same keys / dtypes as the simulate stage)
# ----------------------------------------------------------------------------------------------
def synthetic_movie_data(path, *, N=64, n_snap=30, seed=1):
    rng = np.random.default_rng(seed)
    fw = np.load(FRAMEWORK_NPZ, allow_pickle=True); ref = np.load(REFERENCE_NPZ, allow_pickle=True)
    a = float(fw["a_pseudo"]); L = float(fw["box"]); kT = float(ref["kT"]); beta = 1.0 / kT
    o_pos = np.asarray(fw["o_pos"], np.float32); si_pos = np.asarray(fw["pos"])[np.asarray(fw["kind"]) == "Si"].astype(np.float32)
    n_bins = 180
    grid = -np.pi + (np.arange(n_bins) + 0.5) * 2 * np.pi / n_bins
    assert np.allclose(grid, ref["grid_phi"]), "engine grid != reference grid"
    F_ref = ref["F"] - ref["F"].mean(); U_ref = ref["U"] - ref["U"].mean(); mTS_ref = F_ref - U_ref
    T, dt = 60.0, 2.0e-4
    snap_t = np.linspace(0.0, T, n_snap); snap_step = np.round(snap_t / dt).astype(np.int64)
    t_warm, t_fr, t_burn = 4.0, 8.0, 4.0
    # COM random walks: start in the two alpha-cages, diffuse (x faster), soft confinement to the cage axis
    com = np.empty((n_snap, N, 3))
    com[0] = np.stack([rng.choice([0.5 * a, 1.5 * a], N) + rng.normal(0, 1.2, N),
                       0.5 * a + rng.normal(0, 1.5, N) + a * rng.integers(0, 2, N),
                       0.5 * a + rng.normal(0, 1.5, N) + a * rng.integers(0, 2, N)], axis=1)
    for k in range(1, n_snap):
        stp = rng.normal(0, 1.1, (N, 3)); stp[:, 0] *= 1.8
        com[k] = com[k - 1] + stp
        for ax_ in (1, 2):
            com[k, :, ax_] -= 0.25 * (np.mod(com[k, :, ax_], a) - 0.5 * a)
    u = rng.normal(size=(n_snap, N, 3)); u /= np.linalg.norm(u, axis=-1, keepdims=True)

    def arm(com, u, noise_scale):
        q = np.stack([com + 0.77 * u, com - 0.77 * u], axis=2).astype(np.float32)
        phi = np.mod(2 * np.pi * com[..., 0] / a + np.pi, 2 * np.pi) - np.pi
        # learned PMF: reference + decaying smooth noise (4 random Fourier modes)
        modes = np.stack([np.cos(k * grid + rng.uniform(0, 2 * np.pi)) for k in (1, 2, 3, 5)])
        snap_F = np.empty((n_snap, n_bins))
        for s in range(n_snap):
            amp = noise_scale * (6.0 * np.exp(-snap_t[s] / 15.0) + 0.3)
            snap_F[s] = F_ref + amp * (rng.normal(size=4) @ modes)
        snap_F -= snap_F.mean(axis=1, keepdims=True)
        eF = np.sqrt(((snap_F - F_ref) ** 2).mean(axis=1))
        z = phi * a / (2 * np.pi)
        return dict(q=q, phi=phi.astype(np.float32), F=snap_F.astype(np.float32), eF=eF.astype(np.float32),
                    frac_window=(np.abs(z) < WINDOW_HALF).mean(axis=1).astype(np.float32),
                    frac_cage=(np.abs(z) > CAGE_MIN).mean(axis=1).astype(np.float32))

    out = dict(temperature=np.float32(ref["temperature"]), kT=np.float32(kT), beta=np.float32(beta), a_pseudo=np.float32(a),
               box=np.float32(L), n_bins=np.int64(n_bins), seed=np.int64(seed), featured=np.int64(0), t_fr=np.float32(t_fr),
               t_warm=np.float32(t_warm), t_burn=np.float32(t_burn), fr_rate=np.float32(0.1), git_rev="selftest",
               reference_path=os.path.relpath(REFERENCE_NPZ, ROOT), sim_json=json.dumps({"n_replicas": N, "dt": dt}),
               params_json=json.dumps({"temperature": 300.0}), methods=np.array(["abf", "fr_uniform"]),
               arms_identical_until_t=np.float32(t_fr), N=np.int64(N), T=np.float32(T), dt=np.float32(dt),
               grid=grid.astype(np.float32), z_grid=(grid * a / (2 * np.pi)).astype(np.float32),
               dphi=np.float32(2 * np.pi / n_bins), F_ref=F_ref.astype(np.float32), U_ref=U_ref.astype(np.float32),
               mTS_ref=mTS_ref.astype(np.float32), dF_kT=np.float32(float(ref["dF_barrier"]) * beta),
               dU_kT=np.float32(float(ref["dU_barrier"]) * beta), mTdS_kT=np.float32(float(ref["mTdS_barrier"]) * beta),
               o_pos=o_pos, si_pos=si_pos, snap_t=snap_t, snap_step=snap_step, save_t=snap_t[::3])
    A = arm(com, u, 1.0)
    # FR arm: same walk until t_fr, then a few replacements (the featured slot 0 dies once)
    com_f = com.copy(); u_f = u.copy()
    wid = np.tile(np.arange(N, dtype=np.int32), (n_snap, 1))
    die_rows, birth_rows = [], []
    next_id = N
    for s, dying in ((12, [0, 5, 9]), (16, [20, 21]), (20, [3]), (24, [40, 41, 42, 43]), (27, [7])):
        for slot in dying:
            src = int(rng.choice([j for j in range(N) if j not in dying]))
            cp = com_f[s - 1, slot].copy()
            die_rows.append([s, float(np.mod(2 * np.pi * cp[0] / a + np.pi, 2 * np.pi) - np.pi), *cp])
            cs = com_f[s - 1, src].copy()
            birth_rows.append([s, float(np.mod(2 * np.pi * cs[0] / a + np.pi, 2 * np.pi) - np.pi), *cs])
            com_f[s:, slot] = com_f[s:, src] + rng.normal(0, 0.6, (n_snap - s, 3)) * np.arange(1, n_snap - s + 1)[:, None] ** 0.5
            u_f[s:, slot] = u_f[s:, src]
            wid[s:, slot] = next_id; next_id += 1
    B = arm(com_f, u_f, 0.7)
    for m, r, w, die, birth in (("abf", A, np.tile(np.arange(N, dtype=np.int32), (n_snap, 1)), np.zeros((0, 5), np.float32),
                                 np.zeros((0, 5), np.float32)),
                                ("fr_uniform", B, wid, np.asarray(die_rows, np.float32), np.asarray(birth_rows, np.float32))):
        counts = rng.poisson(3.0, (n_snap, n_bins)).astype(np.float32).cumsum(axis=0)
        cross = np.cumsum(rng.poisson(0.6, n_snap)).astype(np.int64)
        out.update({f"{m}/snap_q": r["q"], f"{m}/snap_phi": r["phi"], f"{m}/snap_wid": w, f"{m}/snap_F": r["F"],
                    f"{m}/snap_counts": counts, f"{m}/snap_crossings": cross, f"{m}/snap_eF": r["eF"],
                    f"{m}/snap_ev_die": die, f"{m}/snap_ev_birth": birth, f"{m}/save_eF": r["eF"][::3],
                    f"{m}/l2_f": np.float32(r["eF"][-1]), f"{m}/integrated_l2_f": np.float32(np.trapezoid(r["eF"], snap_t) / T),
                    f"{m}/total_replacement_events": np.float32(len(die)), f"{m}/min_ess_frac": np.float32(0.9),
                    f"{m}/max_wmax": np.float32(0.01), f"{m}/n_cage_crossings": np.float32(cross[-1]),
                    f"{m}/frac_window": r["frac_window"], f"{m}/frac_cage": r["frac_cage"]})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def selftest(out_dir, workers=4, frames=(3, 15)):
    """Fabricate a tiny consistent movie_data.npz, draw two check frames, then the full tiny movie."""
    os.makedirs(out_dir, exist_ok=True)
    npz = synthetic_movie_data(os.path.join(out_dir, "movie_data.npz"))
    print("synthetic data:", npz)
    d = load_movie_data(npz)
    G.update(d=d, event_window=4, cloud_window=40, debug_bbox=True)
    for k in frames:
        p = os.path.join(out_dir, f"check_{k:03d}.png")
        draw_frame((k, p)); print("wrote", p)
    G["debug_bbox"] = False
    render(out_dir, stride=1, fps=10, workers=workers, keep_frames=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=OUT_DIR, help="directory holding movie_data.npz; frames and the mp4 go there")
    ap.add_argument("--stride", type=int, default=3, help="snapshots per frame")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--event-window", type=int, default=4, help="flash events from the last k snapshots")
    ap.add_argument("--cloud-window", type=int, default=40, help="snapshots pooled in the configuration cloud")
    ap.add_argument("--only", type=int, default=None, help="render one frame index (layout check)")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--movie-name", default=MOVIE)
    ap.add_argument("--selftest", action="store_true", help="synthetic data in --out (default: a scratch directory)")
    a = ap.parse_args()
    if a.selftest:
        out = a.out if a.out != OUT_DIR else os.path.join(os.environ.get("TMPDIR", "/tmp"), "lta_movie_selftest")
        selftest(out, workers=min(a.workers, 8))
        return
    render(a.out, stride=a.stride, fps=a.fps, workers=a.workers, event_window=a.event_window, cloud_window=a.cloud_window,
           only=a.only, keep_frames=a.keep_frames, gif=a.gif, movie_name=a.movie_name)


if __name__ == "__main__":
    main()
