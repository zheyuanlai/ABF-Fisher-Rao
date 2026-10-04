#!/usr/bin/env python
"""Frame renderer: "Ethane in LTA: the simulation ENSEMBLE under ABF vs ABF + Fisher-Rao birth-death".

A three-act presentation movie (1920x1080, 30 fps, ~34 s) from the SAME ``movie_data.npz`` that
scripts/lta_movie_render.py draws (one seed, both arms, the LTA engine's bit-inert ``store_snapshots``
record).  Where that movie follows ONE ethane, this one follows the POPULATION of N replicas: where it
sits along the CV, how Fisher-Rao births / deaths move it, and which lineages the births come from.

    python scripts/lta_ensemble_movie_render.py --out results/lta_movie/T150 --workers 24      # frames + mp4
    python scripts/lta_ensemble_movie_render.py --out results/lta_movie/T150 --storyboard       # one PNG per act
    python scripts/lta_ensemble_movie_render.py --data results/lta_movie/T150/movie_data.npz --out /tmp/x --only 300
    python scripts/lta_ensemble_movie_render.py --selftest                                       # synthetic data

Act I   (6 s)  the physical problem.  The x-y slab of the framework with ONE replica of the ABF arm (the
               ensemble's first window discoverer) moving over t in [0, 6]; the reference decomposition
               F = U - TS with a cursor at that replica's z_CV.  In the last 1.5 s all N replica COMs fade in:
               "one dot = one replica, not one atom".
Act II  (16 s) the ensemble, ABF left / ABF + FR right, NON-UNIFORM clock (t 0->4 in 12.5 % of the frames,
               the lineage bloom t 4->16 in 68.75 %, t 16->60 in 18.75 %).  Per column: a beeswarm of all
               replicas along z_CV -- grey = original walker (id < N), warm orange = Fisher-Rao copy
               (id >= N), the K highlighted families in saturated colours with their sizes printed live --
               the density ratio log2 p_t(z)/q(z) as a heat strip, the window-occupancy meter
               R_W(t) = p_t(W)/q(W) with the copy fractions in the window / cages and the number of
               families alive, the learned PMF; below, e_F(t) and R_W(t) of both arms.
Act III (10 s) the memory.  Kymographs of log2 p_t(z)/q(z) revealed left to right, FR deaths / births
               overlaid, the highlighted families' worldlines (a single line under ABF, a branching tree
               under FR; the largest family in full colour, the others faint), the fraction of copies among
               the window replicas vs t, R_W(t) of both arms, and the closing statement on a 4 s hold.
               The mechanism is collective -- hundreds of small families (largest ~13 of 1024), the window
               populated by copies while originals trickle in -- so the closing line quotes the copy
               fraction in the window, not a lineage takeover.

Genealogy keys ``{m}/snap_anc`` (initial-ancestor label per slot), ``{m}/snap_ev_slots`` and
``{m}/snap_ev_ids`` are used when present; without them every walker is its own family (the highlighted
walkers are then tracked singly).  ``--family-rule`` picks the highlighted families: ``largest`` (default,
the largest final families among those that reach the window) or ``first`` (the first to reach it).  Everything frame-independent is precomputed in
:func:`load_ensemble_data`; a frame only slices arrays.  ``G`` is populated before the pool is forked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import lta_movie_render as base  # noqa: E402
from lta_movie_render import (C_ABF, C_BIRTH, C_DIE, C_GRID, C_INK, C_INK2, C_O, C_O_EDGE, C_REF, C_SI, C_TS,  # noqa: E402
                              C_U, CAGE_MIN, DPI, FIG_H, FIG_W, FRAMEWORK_NPZ, R_BEAD, R_O, R_SI, WINDOW_HALF,
                              load_movie_data)
from lta_movie_render import C_FR  # noqa: E402

MOVIE = "lta_ensemble_abf_vs_fr"
OUT_DIR = os.path.join(ROOT, "results", "lta_movie", "T150")
REFERENCE_T150 = os.path.join(ROOT, "results", "uniform_campaign", "lta", "reference", "reference_T150.npz")

C_BEAD = "#b8b6b0"                                   # an ORIGINAL walker (id < N) outside the highlighted families
C_COPY = "#f4b183"                                   # a Fisher-Rao COPY (id >= N) outside the highlighted families
C_COPY_TXT = "#b8651f"                               # the same, dark enough for text
C_FAM = ("#1f77b4", "#9467bd", "#2ca02c", "#e377c2", "#8c564b", "#17becf")   # highlighted families
C_BAND_CAGE, C_BAND_WIN = "#f3f1ec", "#e8edf5"
C_PAPER = "#f6f5f2"                                  # unrevealed kymograph
N_SWARM_BINS, SWARM_CAP = 96, 36                     # beeswarm: z-bins, rows before the overflow count
N_DENS_BINS, LOG2_CLIP = 60, 1.5                     # density ratio: z-bins, colour clip of log2(p/q)
ARMS = (("abf", C_ABF, "ABF"), ("fr_uniform", C_FR, "ABF + Fisher–Rao"))
ACT_NAMES = {1: "Act I — the physical problem: an entropic barrier, one replica at a time",
             2: "Act II — the ensemble: where the N replicas are, and how birth–death moves them",
             3: "Act III — the memory of the movie: density kymographs and lineage trees"}

G = {}   # shared, read-only, populated before the fork


# ----------------------------------------------------------------------------------------------
# precomputation
# ----------------------------------------------------------------------------------------------
def _nearest_snap(snap_t, t):
    k = int(np.clip(np.searchsorted(snap_t, t), 1, len(snap_t) - 1))
    return k - 1 if abs(snap_t[k - 1] - t) <= abs(snap_t[k] - t) else k


def _trailing_mean(x, w):
    """Mean of the last ``w`` rows (fewer at the start), along axis 0."""
    if w <= 1:
        return x
    cs = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x], axis=0), axis=0)
    n = len(x); hi = np.arange(1, n + 1); lo = np.maximum(hi - w, 0)
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None]


def _density_log2_ratio(zc, a, nb, N, smooth):
    """(n_snap, nb) log2 of the replica density over the uniform target, from an nb-bin histogram of z_CV,
    averaged over the trailing ``smooth`` snapshots, clipped to +-5 before the log."""
    n_snap = zc.shape[0]; zh = 0.5 * a
    b = np.clip(np.floor((zc.astype(np.float64) + zh) / a * nb).astype(np.int64), 0, nb - 1)
    counts = np.bincount((np.arange(n_snap)[:, None] * nb + b).ravel(), minlength=n_snap * nb)
    counts = counts.reshape(n_snap, nb).astype(np.float64)
    R = _trailing_mean(counts, smooth) / (N / nb)
    return np.log2(np.clip(R, 2.0 ** -5, 2.0 ** 5))


def _worldlines(snap_t, zc, wid, mask, a, N):
    """One series per walker id that ever belongs to the family (``mask``: (n_snap, N) bool).

    Each series: t, z (with NaN breaks where z wraps across +-a/2), t_fill (the NaN rows carry the
    time of the next point so a ``t_fill <= t`` slice keeps the breaks), t0 / z0 (first appearance) and
    ``born`` (minted by a birth, i.e. id >= N).  A walker occupies one slot from birth to death, so the
    snapshots where its id is present form a contiguous run."""
    ii, ss = np.nonzero(mask)
    if ii.size == 0:
        return []
    ids = wid[ii, ss]
    order = np.lexsort((ii, ids))
    ids, ii, ss = ids[order], ii[order], ss[order]
    cuts = np.flatnonzero(np.diff(ids)) + 1
    out = []
    for blk in np.split(np.arange(ids.size), cuts):
        tt = np.asarray(snap_t[ii[blk]], np.float64); zz = zc[ii[blk], ss[blk]].astype(np.float64)
        jump = np.flatnonzero(np.abs(np.diff(zz)) > 0.5 * a) + 1
        tb = np.insert(tt, jump, np.nan); zb = np.insert(zz, jump, np.nan)
        tf = np.insert(tt, jump, tt[jump])
        out.append(dict(id=int(ids[blk[0]]), t=tb, z=zb, t_fill=tf, t0=float(tt[0]), z0=float(zz[0]),
                        born=bool(ids[blk[0]] >= N), n=int(blk.size)))
    return out


def load_ensemble_data(npz_path, *, n_families=3, smooth_snaps=5, family_rule="largest", verbose=True):
    """Everything frame-independent.  Builds on :func:`lta_movie_render.load_movie_data` (framework slab,
    folded COMs, z_CV, kT-scaled profiles, event rows) and adds the ensemble quantities: the density ratio
    per snapshot and arm, R_W(t), the highlighted families (ids, first window entry, membership masks,
    sizes, worldlines), the copy fractions, the number of families alive and the act-I replica.
    ``family_rule``: "largest" (default) = among the initial ancestors that ever have a member in the window
    of the FR arm, the largest final families (tie-break: peak size, then first entry); "first" = the first
    ``n_families`` to reach the window."""
    d = load_movie_data(npz_path)
    raw = np.load(npz_path, allow_pickle=True)
    a, N, T = d["a_pseudo"], d["N"], d["T"]
    snap_t = np.asarray(d["snap_t"], np.float64); n_snap = len(snap_t)
    d["snap_t"] = snap_t; d["zh"] = 0.5 * a
    d["has_genealogy"] = all(f"{m}/snap_anc" in raw.files for m in d["methods"])
    d["smooth_snaps"] = int(smooth_snaps)
    d["smooth_t"] = float(smooth_snaps * (snap_t[1] - snap_t[0])) if n_snap > 1 else 0.0
    for m in d["methods"]:
        wid = np.asarray(d[f"{m}/snap_wid"]).astype(np.int64)
        d[f"{m}/snap_wid"] = wid
        d[f"{m}/anc"] = (np.asarray(raw[f"{m}/snap_anc"]).astype(np.int64) if f"{m}/snap_anc" in raw.files
                         else wid.copy())
        for key in ("snap_ev_slots", "snap_ev_ids"):
            d[f"{m}/{key}"] = (np.asarray(raw[f"{m}/{key}"]).reshape(-1, 2).astype(np.int64) if f"{m}/{key}" in raw.files
                               else np.zeros((0, 2), np.int64))
        zc = d[f"{m}/zcv"]
        d[f"{m}/log2R"] = _density_log2_ratio(zc, a, N_DENS_BINS, N, smooth_snaps)
        d[f"{m}/RW"] = d[f"{m}/frac_window"] / d["win_target"]
        die, birth = d[f"{m}/snap_ev_die"], d[f"{m}/snap_ev_birth"]
        d[f"{m}/ev_snap"] = die[:, 0].astype(np.int64)
        d[f"{m}/ev_t"] = snap_t[np.clip(d[f"{m}/ev_snap"], 0, n_snap - 1)]
        d[f"{m}/ev_zdie"] = die[:, 1] * a / (2 * np.pi)
        d[f"{m}/ev_zbirth"] = birth[:, 1] * a / (2 * np.pi)
        # copies (minted ids) among the replicas in the window / in the cages, and families alive
        is_copy = wid >= N
        win = np.abs(zc) < WINDOW_HALF; cage = np.abs(zc) > CAGE_MIN
        nw = win.sum(axis=1); nc = cage.sum(axis=1)
        d[f"{m}/copy_frac_window"] = np.where(nw > 0, (is_copy & win).sum(axis=1) / np.maximum(nw, 1), np.nan)
        d[f"{m}/copy_frac_cage"] = np.where(nc > 0, (is_copy & cage).sum(axis=1) / np.maximum(nc, 1), np.nan)
        anc_m = d[f"{m}/anc"]; n_lab = int(anc_m.max()) + 1
        d[f"{m}/n_families"] = np.array([np.unique(anc_m[k]).size for k in range(n_snap)], dtype=np.int64)
        d[f"{m}/max_family"] = int(max(int(np.bincount(anc_m[k], minlength=n_lab).max()) for k in range(n_snap)))
    # ---- highlighted families: the first K initial-ancestor labels with a member in the window (FR arm)
    fr = "fr_uniform" if "fr_uniform" in d["methods"] else d["methods"][-1]
    zc = d[f"{fr}/zcv"]; anc = d[f"{fr}/anc"]
    ii, ss = np.nonzero(np.abs(zc) < WINDOW_HALF)                   # time-major order
    labels = anc[ii, ss]
    uniq, first = np.unique(labels, return_index=True)
    if family_rule == "largest":
        n_lab = int(max(N, anc.max() + 1))
        final = np.bincount(anc[-1], minlength=n_lab)[uniq]
        peak = np.max(np.stack([np.bincount(anc[k], minlength=n_lab) for k in range(0, n_snap, max(n_snap // 300, 1))]), axis=0)[uniq]
        order = np.lexsort((first, -peak, -final))[:n_families]
    elif family_rule == "first":
        order = np.argsort(first, kind="stable")[:n_families]
    else:
        raise ValueError(f"family_rule {family_rule!r} not in ('first', 'largest')")
    fam_ids = [int(v) for v in uniq[order]]
    fam_first_snap = [int(ii[first[k]]) for k in order]
    fam_first_slot = [int(ss[first[k]]) for k in order]
    d["fam_ids"] = fam_ids; d["family_rule"] = family_rule
    d["fam_first_t"] = [float(snap_t[k]) for k in fam_first_snap]
    d["fam_first_slot"] = fam_first_slot
    d["fam_shared"] = [t_ < d["t_fr"] for t_ in d["fam_first_t"]]  # discovered while the arms still share the noise
    for m in d["methods"]:
        masks = [d[f"{m}/anc"] == r for r in fam_ids]
        d[f"{m}/fam_mask"] = masks
        d[f"{m}/fam_size"] = [mk.sum(axis=1) for mk in masks]
        d[f"{m}/fam_any"] = np.logical_or.reduce(masks) if masks else np.zeros((n_snap, N), bool)
        d[f"{m}/worldlines"] = [_worldlines(snap_t, d[f"{m}/zcv"], d[f"{m}/snap_wid"], mk, a, N) for mk in masks]
    # ---- act I follows the ABF arm's first window discoverer (slot = id under ABF)
    za = d["abf/zcv"] if "abf" in d["methods"] else zc
    ever = (np.abs(za) < WINDOW_HALF).any(axis=0)
    if ever.any():
        fe = np.where(ever, np.argmax(np.abs(za) < WINDOW_HALF, axis=0), n_snap)
        d["act1_slot"] = int(np.argmin(fe)); d["act1_first_t"] = float(snap_t[fe[d["act1_slot"]]])
    else:
        d["act1_slot"] = int(d["featured"]); d["act1_first_t"] = float("nan")
    d["RW_lim"] = float(max(1.6, 1.1 * max(float(d[f"{m}/RW"].max()) for m in d["methods"])))
    # the family drawn in full colour in act III (the largest final one under FR; index 0 under rule "largest")
    d["fam_main"] = int(np.argmax([int(d[f"{fr}/fam_size"][k][-1]) for k in range(len(fam_ids))])) if fam_ids else 0
    # the closing statement's reference time: copy fraction in the window at t = 8 (or T if shorter)
    k_ref = _nearest_snap(snap_t, min(8.0, T))
    d["copy_ref"] = (float(snap_t[k_ref]), {m: float(100.0 * np.nan_to_num(d[f"{m}/copy_frac_window"][k_ref])) for m in d["methods"]})
    if verbose:
        print(f"ensemble data: N = {N}, {n_snap} snapshots, T = {T:g}, t_fr = {d['t_fr']:g}, "
              f"genealogy keys {'present' if d['has_genealogy'] else 'ABSENT (every walker its own family)'}")
        print(f"  act I replica: slot {d['act1_slot']} (ABF arm's first window entry at t = {d['act1_first_t']:.2f})")
        print(f"  highlighted families: rule {family_rule!r}; largest family at any time: "
              f"{ {m: d[f'{m}/max_family'] for m in d['methods']} }; families alive at T: "
              f"{ {m: int(d[f'{m}/n_families'][-1]) for m in d['methods']} }")
        tr, pr = d["copy_ref"]
        print(f"  copies among window replicas at t = {tr:g}: " + ", ".join(f"{m} {pr[m]:.0f} %" for m in d["methods"]))
        for k, r in enumerate(fam_ids):
            sizes = {m: int(d[f'{m}/fam_size'][k][-1]) for m in d["methods"]}
            peak = {m: int(d[f'{m}/fam_size'][k].max()) for m in d["methods"]}
            nwl = {m: len(d[f'{m}/worldlines'][k]) for m in d["methods"]}
            print(f"  family {k + 1}: ancestor {r}, first in window at t = {d['fam_first_t'][k]:.2f} (slot {fam_first_slot[k]}, "
                  f"{'before' if d['fam_shared'][k] else 'AFTER'} t_fr -> {'same walker in both arms' if d['fam_shared'][k] else 'FR arm only'}); "
                  f"final size {sizes}, peak {peak}, walkers ever {nwl}")
    return d


def build_schedule(d, act_seconds, fps):
    """List of (act, snapshot index, progress within the act) per frame.  Act I: t 0 -> min(6, T) linearly.
    Act II: 12.5 % of its frames for t 0 -> t_fr, 68.75 % for t_fr -> t_fr + 12 (the bloom), 18.75 % for
    the rest.  Act III: the first 60 % sweep t 0 -> T, the rest hold at T."""
    snap_t, T, t_fr = d["snap_t"], d["T"], d["t_fr"]
    n1, n2, n3 = (max(int(round(s * fps)), 1) for s in act_seconds)
    t1_end = min(6.0, T); t_bloom = min(t_fr + 12.0, T)
    sched = []
    for k in range(n1):
        p = k / max(n1 - 1, 1)
        sched.append((1, _nearest_snap(snap_t, p * t1_end), p))
    for k in range(n2):
        p = k / max(n2 - 1, 1)
        if p < 0.125:
            t = t_fr * p / 0.125
        elif p < 0.8125:
            t = t_fr + (t_bloom - t_fr) * (p - 0.125) / 0.6875
        else:
            t = t_bloom + (T - t_bloom) * (p - 0.8125) / 0.1875
        sched.append((2, _nearest_snap(snap_t, t), p))
    for k in range(n3):
        p = k / max(n3 - 1, 1)
        sched.append((3, _nearest_snap(snap_t, T * min(p / 0.6, 1.0)), p))
    return sched


# ----------------------------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------------------------
def _ax_in(fig, x, y, w, h, **kw):
    """Axes rectangle in inches from the bottom-left corner."""
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H], **kw)


def _bbox(ec="none", alpha=0.88, pad=0.25):
    return dict(boxstyle=f"round,pad={pad}", fc="white", ec=ec, alpha=alpha)


def _header(fig, d, act, t):
    T = d["T"]
    fig.text(0.047, 0.966, f"Ethane in LTA ({d['temperature']:g} K): the simulation ensemble under ABF vs ABF + Fisher–Rao "
             "birth–death", fontsize=16, fontweight="bold", ha="left", va="center")
    fig.text(0.047, 0.934, ACT_NAMES[act], fontsize=12, color=C_INK2, ha="left", va="center")
    fig.text(0.985, 0.966, f"t = {t:5.1f} / {T:g}", fontsize=17, ha="right", va="center", family="monospace")
    if act >= 2:
        on = t >= d["t_fr"]
        fig.text(0.985, 0.934, f"Fisher–Rao {'ON since' if on else 'off, starts at'} t = {d['t_fr']:g}", fontsize=10.5,
                 color=C_FR if on else C_INK2, ha="right", va="center")


def _draw_slab(ax, d):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle
    a = d["a_pseudo"]
    ax.add_collection(PatchCollection([Circle(tuple(p), R_O) for p in d["slab_o"]], facecolor=C_O, edgecolor=C_O_EDGE,
                                      linewidths=0.4, alpha=0.85, zorder=1))
    ax.add_collection(PatchCollection([Circle(tuple(p), R_SI) for p in d["slab_si"]], facecolor=C_SI, edgecolor="none",
                                      alpha=0.9, zorder=2))
    for xw in (0.0, a, 2 * a):
        ax.axvline(xw, color=C_INK2, lw=0.7, ls=(0, (1, 3)), alpha=0.8, zorder=0)
    for xx, lab_ in ((0.5 * a, "α-cage"), (a, "window"), (1.5 * a, "α-cage")):
        ax.text(xx, a + 0.45, lab_, ha="center", va="bottom", fontsize=11, color=C_INK2)


def _region_bands(ax, zh, y0=None, y1=None, labels=True, label_y=0.03):
    kw = {} if y0 is None else dict(ymin=y0, ymax=y1)
    ax.axvspan(-zh, -CAGE_MIN, color=C_BAND_CAGE, lw=0, zorder=0, **kw)
    ax.axvspan(CAGE_MIN, zh, color=C_BAND_CAGE, lw=0, zorder=0, **kw)
    ax.axvspan(-WINDOW_HALF, WINDOW_HALF, color=C_BAND_WIN, lw=0, zorder=0, **kw)
    for zl in (-CAGE_MIN, -WINDOW_HALF, WINDOW_HALF, CAGE_MIN):
        ax.axvline(zl, color=C_INK2, lw=0.5, ls=(0, (4, 3)), alpha=0.5, zorder=1)
    if labels:
        for xx, lab_ in ((-(zh + CAGE_MIN) / 2, "cage"), (0.0, "window"), ((zh + CAGE_MIN) / 2, "cage")):
            ax.text(xx, label_y, lab_, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5,
                    color=C_INK2, zorder=6, bbox=_bbox(alpha=0.7, pad=0.12))


def _swarm_layout(z, hl, a, nb=N_SWARM_BINS):
    """Dot-histogram layout: bin index, row within the bin (highlighted replicas take the lowest rows, then
    slot order), bin centre x, and the stack height per bin."""
    N = z.size; zh = 0.5 * a
    b = np.clip(np.floor((z.astype(np.float64) + zh) / a * nb).astype(np.int64), 0, nb - 1)
    order = np.lexsort((np.arange(N), ~hl, b))
    bs = b[order]
    starts = np.r_[0, np.flatnonzero(np.diff(bs)) + 1]
    row_sorted = np.arange(N) - np.repeat(starts, np.diff(np.r_[starts, N]))
    row = np.empty(N, np.int64); row[order] = row_sorted
    height = np.bincount(b, minlength=nb)
    return b, row, -zh + (b + 0.5) * a / nb, height


def _series_past(ax, ts, y, t, colour, label=None, lw=2.2, log=False):
    keep = np.isfinite(y) & ((y > 0) if log else True) & (ts >= 0)
    ax.plot(ts[keep], y[keep], color=colour, lw=1.0, alpha=0.25)
    past = keep & (ts <= t + 1e-9)
    if past.any():
        ax.plot(ts[past], y[past], color=colour, lw=lw, label=label)
        ax.plot([ts[past][-1]], [y[past][-1]], "o", color=colour, ms=6.5)


# ----------------------------------------------------------------------------------------------
# act I
# ----------------------------------------------------------------------------------------------
def _act1(fig, d, i, p):
    import matplotlib.pyplot as plt  # noqa: F401
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    a, N, zh = d["a_pseudo"], d["N"], d["zh"]
    t = d["snap_t"][i]; slot = d["act1_slot"]
    m = "abf" if "abf" in d["methods"] else d["methods"][0]
    fade = float(np.clip((p - 0.75) / 0.14, 0.0, 1.0))            # all replicas fade in over frames >= 75 % of the act
    zcv = d[f"{m}/zcv"][i]; zf = float(zcv[slot])
    region = "window" if abs(zf) < WINDOW_HALF else ("cage" if abs(zf) > CAGE_MIN else "neck")

    # ---- framework slab with one replica ------------------------------------------------------
    xr = (-0.6, 2 * a + 0.6); yr = (-0.6, a + 1.6)
    H_OV = 4.75; W_OV = H_OV * (xr[1] - xr[0]) / (yr[1] - yr[0])
    ax = _ax_in(fig, 0.6, 2.75, W_OV, H_OV)
    _draw_slab(ax, d)
    lo = max(0, i - 60)
    q = np.asarray(d[f"{m}/snap_q"][lo:i + 1, slot], np.float64)   # (k, 2, 3) unwrapped
    com = q.mean(axis=1); cur = com[-1]
    off = np.array([np.floor(cur[0] / (2 * a)) * 2 * a, np.floor(cur[1] / a) * a, 0.0])
    trail = com - off
    if len(trail) > 1:
        jump = np.flatnonzero((np.abs(np.diff(trail[:, 0])) > a) | (np.abs(np.diff(trail[:, 1])) > 0.5 * a)) + 1
        tx = np.insert(trail[:, 0], jump, np.nan); ty = np.insert(trail[:, 1], jump, np.nan)
        ax.plot(tx, ty, color=C_ABF, lw=1.3, alpha=0.35, zorder=4)
    qq = q[-1] - off
    ax.plot(qq[:, 0], qq[:, 1], color=C_ABF, lw=3.6, solid_capstyle="round", zorder=5)
    ax.add_collection(PatchCollection([Circle((pp[0], pp[1]), R_BEAD) for pp in qq], facecolor=C_ABF, edgecolor="white",
                                      linewidths=1.0, zorder=6))
    if fade > 0:
        xy = d[f"{m}/com_xy"][i]
        ax.scatter(xy[:, 0], xy[:, 1], s=11, c=C_ABF, alpha=0.65 * fade, linewidths=0, zorder=3)
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 5, 10, 15, 20]); ax.set_yticks([0, 5, 10]); ax.tick_params(labelsize=9)
    ax.set_xlabel("x (Å) along the cage–window–cage axis  (the collective variable $z_{CV}$ = x folded into one cell, "
                  "window at 0)", fontsize=9.5, color=C_INK2, labelpad=4)
    ax.set_ylabel("y (Å)", fontsize=9.5, labelpad=2)
    ax.set_title(f"one replica of the ABF arm (slot {slot}): two beads, one bond, in the x–y slab through the cage and "
                 "window centres", fontsize=11.5, loc="left", pad=6, color=C_INK)
    handles = [Line2D([], [], marker="o", ls="none", mfc=C_O, mec=C_O_EDGE, ms=10, label="framework O"),
               Line2D([], [], marker="o", ls="none", mfc=C_SI, mec="none", ms=5, label="framework Si"),
               Line2D([], [], marker="o", ls="-", color=C_ABF, lw=3, mfc=C_ABF, mec="white", ms=9, label="this ethane"),
               Line2D([], [], color=C_ABF, lw=1.3, alpha=0.4, label="its path (last 60 snapshots)")]
    if fade > 0.05:
        handles.append(Line2D([], [], marker="o", ls="none", mfc=C_ABF, mec="none", ms=4, alpha=0.7,
                              label=f"COM of each of the {N} replicas"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.145), fontsize=9, frameon=False, ncol=5,
              handlelength=1.4, handletextpad=0.6, columnspacing=1.5, borderaxespad=0.0)

    # ---- the reference decomposition with a cursor ---------------------------------------------
    zg, Fk, Uk, mTSk = d["zg"], d["Fk"], d["Uk"], d["mTSk"]
    ax = _ax_in(fig, 10.35, 3.15, 5.0, 4.0)
    _region_bands(ax, zh, labels=True, label_y=0.02)
    ax.plot(zg, Fk, color=C_INK, lw=2.0, label="$\\beta F(z)$  free energy")
    ax.plot(zg, Uk, color=C_U, lw=1.5, ls="--", label="$\\beta U(z)$  mean energy")
    ax.plot(zg, mTSk, color=C_TS, lw=1.5, ls="-.", label="$-TS(z)/k_BT$  entropy term")
    ax.axvline(zf, color=C_ABF, lw=1.8, alpha=0.9, zorder=4)
    ax.plot([zf], [np.interp(zf, zg, Fk, period=a)], "o", color=C_ABF, ms=9, mec="white", mew=1.2, zorder=5)
    y_lo = min(Fk.min(), Uk.min(), mTSk.min()) - 1.0; y_hi = max(Fk.max(), Uk.max(), mTSk.max()) + 4.5
    ax.text(0.985, 0.97, f"this replica:  $z_{{CV}}$ = {zf:+.1f} Å  ({region})", transform=ax.transAxes, color=C_ABF,
            fontsize=10, ha="right", va="top", zorder=6, bbox=_bbox(ec=C_ABF, alpha=0.95))
    ax.set_xlim(-zh, zh); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("$z_{CV}$ (Å)   window at 0, cage centres at ±a/2 (periodic)", fontsize=9.5, labelpad=2)
    ax.set_ylabel("$k_BT$", fontsize=10, labelpad=2); ax.tick_params(labelsize=9)
    ax.grid(True, color=C_GRID, lw=0.6)
    ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.9, edgecolor="none", handlelength=1.8,
              borderaxespad=0.4, labelspacing=0.3)
    ax.set_title("the window is not high in energy — it has few configurations\n"
                 f"$-T\\Delta S^\\ddagger$ = {d['mTdS_kT']:.1f} $k_BT$ of $\\Delta F^\\ddagger$ = {d['dF_kT']:.1f} $k_BT$ "
                 f"($\\Delta U^\\ddagger$ = {d['dU_kT']:.1f} $k_BT$)", fontsize=11.5, loc="left", pad=8)

    # ---- captions -------------------------------------------------------------------------------
    cap_early = ("one ethane diffuses in the α-cages; the 8-ring window is a bottleneck of configurations, not of energy")
    sub_early = (f"first window visit of the whole ensemble at t = {d['act1_first_t']:.1f}" if np.isfinite(d["act1_first_t"])
                 else "no replica reaches the window in this act")
    cap_late = f"we actually simulate N = {N} independent replicas — from here on, one dot = one replica, not one atom"
    sub_late = "the question of the next act is where this population sits along $z_{CV}$, and how Fisher–Rao birth–death moves it"
    a_early = float(np.clip(1.0 - 2.0 * fade, 0.0, 1.0)); a_late = float(np.clip(2.0 * fade - 1.0, 0.0, 1.0))
    if a_early > 0.0:
        fig.text(0.5, 1.25 / FIG_H, cap_early, fontsize=15, ha="center", va="center", alpha=a_early, fontweight="bold")
        fig.text(0.5, 0.8 / FIG_H, sub_early, fontsize=11.5, ha="center", va="center", alpha=a_early, color=C_INK2)
    if a_late > 0.0:
        fig.text(0.5, 1.25 / FIG_H, cap_late, fontsize=15, ha="center", va="center", alpha=a_late, fontweight="bold", color=C_ABF)
        fig.text(0.5, 0.8 / FIG_H, sub_late, fontsize=11.5, ha="center", va="center", alpha=a_late, color=C_INK2)


# ----------------------------------------------------------------------------------------------
# act II
# ----------------------------------------------------------------------------------------------
def _act2(fig, d, i, p, i_frame):
    import matplotlib as mpl
    from matplotlib.colors import to_rgba
    a, N, T, zh = d["a_pseudo"], d["N"], d["T"], d["zh"]
    snap_t = d["snap_t"]; t = snap_t[i]
    K = len(d["fam_ids"]); ew = int(G.get("event_window", 3))
    sched = G["schedule"]
    i_prev = sched[max(i_frame - ew, 0)][1]
    if sched[max(i_frame - ew, 0)][0] != 2 or i_prev >= i:
        i_prev = i - 1
    COL_X, COL_W = (0.75, 8.35), 6.9
    Y_SW, H_SW = 5.45, 2.5
    Y_HT, H_HT = 5.08, 0.27
    Y_MD, H_MD = 3.0, 1.3
    Y_BS, H_BS = 0.55, 1.7
    W_METER = 3.0
    row_px = H_SW * DPI / (SWARM_CAP + 5.0)
    s_bead = (0.86 * row_px * 72.0 / DPI) ** 2
    cmap = mpl.colormaps["RdBu_r"]; norm = mpl.colors.Normalize(-LOG2_CLIP, LOG2_CLIP)
    edges = np.linspace(-zh, zh, N_DENS_BINS + 1)

    for col, (m, colour, label) in enumerate(ARMS):
        if m not in d["methods"]:
            continue
        x0 = COL_X[col]
        zcv = d[f"{m}/zcv"][i]
        hl = d[f"{m}/fam_any"][i]

        # ---- (a) beeswarm ------------------------------------------------------------------------
        ax = _ax_in(fig, x0, Y_SW, COL_W, H_SW)
        _region_bands(ax, zh, label_y=0.015)
        b, row, xb, height = _swarm_layout(zcv, hl, a)
        vis = row < SWARM_CAP
        is_copy = d[f"{m}/snap_wid"][i] >= N
        orig = vis & ~hl & ~is_copy; cop = vis & ~hl & is_copy
        ax.scatter(xb[orig], row[orig] + 0.5, s=s_bead, c=C_BEAD, linewidths=0, zorder=3)
        if cop.any():
            ax.scatter(xb[cop], row[cop] + 0.5, s=s_bead * 1.25, c=C_COPY, linewidths=0, zorder=4)
        for k in range(K):
            sel = d[f"{m}/fam_mask"][k][i] & vis
            if sel.any():
                ax.scatter(xb[sel], row[sel] + 0.5, s=s_bead * 1.8, c=C_FAM[k], edgecolors="white", linewidths=0.5, zorder=5)
        xc = -zh + (np.arange(N_SWARM_BINS) + 0.5) * a / N_SWARM_BINS
        for bb in np.flatnonzero(height > SWARM_CAP):
            ax.text(xc[bb], SWARM_CAP + 0.7, f"+{int(height[bb] - SWARM_CAP)}", rotation=90, fontsize=6.5, ha="center",
                    va="bottom", color=C_INK2, zorder=6)
        n_die = n_cl = 0
        if m == "fr_uniform":
            ev_s = d[f"{m}/ev_snap"]
            sel = (ev_s > i_prev) & (ev_s <= i)
            n_die = int((ev_s <= i).sum()); n_cl = n_die
            if sel.any():
                age = (i - ev_s[sel]) / max(i - i_prev, 1)
                alpha = np.clip(1.0 - 0.7 * age, 0.25, 1.0)
                for zz, c, marker, sz in ((d[f"{m}/ev_zdie"][sel], C_DIE, "x", 42), (d[f"{m}/ev_zbirth"][sel], C_BIRTH, "o", 50)):
                    bz = np.clip(np.floor((zz + zh) / a * N_SWARM_BINS).astype(np.int64), 0, N_SWARM_BINS - 1)
                    # stack several flashes in the same bin
                    order = np.argsort(bz, kind="stable"); bz_s = bz[order]
                    k_in = np.arange(bz.size) - np.repeat(np.r_[0, np.flatnonzero(np.diff(bz_s)) + 1],
                                                          np.diff(np.r_[np.r_[0, np.flatnonzero(np.diff(bz_s)) + 1], bz.size]))
                    yy = np.empty(bz.size); yy[order] = np.minimum(height[bz_s], SWARM_CAP) + 1.3 + 1.1 * k_in
                    rgba = np.tile(np.asarray(to_rgba(c)), (int(sel.sum()), 1)); rgba[:, 3] = alpha
                    if marker == "o":
                        ax.scatter(xc[bz], yy, s=sz, marker="o", facecolors="none", edgecolors=rgba, linewidths=1.5, zorder=7)
                    else:
                        ax.scatter(xc[bz], yy, s=sz, marker="x", c=rgba, linewidths=1.5, zorder=7)
        fam_lines = []
        for k, r in enumerate(d["fam_ids"]):
            n_mem = int(d[f"{m}/fam_size"][k][i])
            fam_lines.append((f"family of replica {r}: {n_mem} member{'s' if n_mem != 1 else ''}"
                              + ("  (extinct)" if n_mem == 0 else ""), C_FAM[k]))
        for k, (txt, c) in enumerate(fam_lines):
            ax.text(0.5, 0.975 - 0.075 * k, txt, transform=ax.transAxes, ha="center", va="top", fontsize=10,
                    color=c, fontweight="bold", zorder=8, bbox=_bbox(alpha=0.8, pad=0.15))
        ax.set_xlim(-zh, zh); ax.set_ylim(0, SWARM_CAP + 5.0)
        ax.set_yticks([0, 12, 24, 36]); ax.tick_params(labelbottom=False, labelsize=8.5)
        ax.set_ylabel("replicas per bin (stacked)", fontsize=9, labelpad=2)
        ax.set_title(label, color=colour, fontsize=15, fontweight="bold", pad=5, loc="left")
        ax.set_title(f"all {N} replicas are original walkers;  {N_SWARM_BINS} bins of $z_{{CV}}$" if col == 0 else
                     "deaths × and births ○ flash where they happen", fontsize=9.5, color=C_INK2, pad=5, loc="right")

        # ---- (b) density-ratio heat strip ------------------------------------------------------------
        ax = _ax_in(fig, x0, Y_HT, COL_W, H_HT)
        ax.imshow(d[f"{m}/log2R"][i][None, :], extent=(-zh, zh, 0, 1), aspect="auto", cmap=cmap, norm=norm,
                  interpolation="nearest")
        for zl in (-CAGE_MIN, -WINDOW_HALF, WINDOW_HALF, CAGE_MIN):
            ax.axvline(zl, color=C_INK2, lw=0.5, ls=(0, (4, 3)), alpha=0.5)
        ax.set_xlim(-zh, zh); ax.set_yticks([])
        ax.set_xticks([-zh, -CAGE_MIN, -WINDOW_HALF, 0, WINDOW_HALF, CAGE_MIN, zh])
        ax.set_xticklabels([f"{-zh:.1f}", f"{-CAGE_MIN:g}", f"{-WINDOW_HALF:g}", "0", f"{WINDOW_HALF:g}", f"{CAGE_MIN:g}", f"{zh:.1f}"])
        ax.tick_params(labelsize=8, length=2, pad=1.5)
        ax.set_ylabel("$\\log_2 \\hat p_t/q$", fontsize=8, labelpad=4, rotation=0, ha="right", va="center")
        if col == 0:
            fig.text((x0) / FIG_W, (Y_HT - 0.19) / FIG_H,
                     f"$\\hat p_t(z)$: {N_DENS_BINS} bins, last {d['smooth_t']:.2g} t.u.;  q = uniform.   "
                     "blue: too few replicas → births;  red: too many → deaths", fontsize=8.5, color=C_INK2, ha="left", va="top")
        else:
            fig.text((x0) / FIG_W, (Y_HT - 0.19) / FIG_H, "FR event: replica reallocation, not molecular teleportation",
                     fontsize=9.5, color=C_FR, ha="left", va="top", fontweight="bold")
            cax = _ax_in(fig, x0 + COL_W - 1.9, Y_HT - 0.37, 1.9, 0.08)
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
            cb.set_ticks([-LOG2_CLIP, 0, LOG2_CLIP]); cb.set_ticklabels([f"≤ −{LOG2_CLIP:g}", "0", f"≥ +{LOG2_CLIP:g}"])
            cax.xaxis.set_ticks_position("top")
            cax.tick_params(labelsize=7.5, length=2, pad=1)
            cax.text(-0.04, 0.5, "$\\log_2 \\hat p_t/q$", transform=cax.transAxes, ha="right", va="center", fontsize=8, color=C_INK2)
            from matplotlib.lines import Line2D
            lax = _ax_in(fig, x0, Y_HT - 0.52, COL_W - 2.0, 0.14); lax.axis("off")
            fam_word = "largest families" if d.get("family_rule") == "largest" else "first families to reach the window"
            lax.legend(handles=[Line2D([], [], marker="o", ls="none", mfc=C_BEAD, mec="none", ms=6, label="grey = original walker"),
                                Line2D([], [], marker="o", ls="none", mfc=C_COPY, mec="none", ms=7, label="orange = Fisher–Rao copy"),
                                Line2D([], [], marker="o", ls="none", mfc=C_FAM[0], mec="white", ms=7,
                                       label=f"coloured = the {K} {fam_word}")],
                       loc="center left", ncol=3, frameon=False, fontsize=8.5, handlelength=1.0, handletextpad=0.4,
                       columnspacing=1.3, borderaxespad=0.0)

        # ---- (c) window-occupancy meter ----------------------------------------------------------------
        ax = _ax_in(fig, x0, Y_MD, W_METER, H_MD); ax.axis("off")
        RW = float(d[f"{m}/RW"][i]); fw = float(d[f"{m}/frac_window"][i])
        ax.text(0.0, 0.87, f"{RW:.2f}×", transform=ax.transAxes, fontsize=28, fontweight="bold", color=colour, ha="left", va="center")
        ax.text(0.47, 0.87, "the uniform target\nin the window  ($R_W$)", transform=ax.transAxes, fontsize=8.5, color=C_INK2,
                ha="left", va="center", linespacing=1.1)
        rows = [(f"in the window:  {100 * fw:.1f} %    (uniform {100 * d['win_target']:.0f} %)", C_INK, 10)]

        def pct(v):
            return "—" if not np.isfinite(v) else f"{100 * v:.0f} %"

        if m == "fr_uniform":
            rows += [(f"so far:  {n_die:,} deaths ×,  {n_cl:,} births ○", C_INK, 9.5),
                     (f"copies among window replicas:  {pct(float(d[f'{m}/copy_frac_window'][i]))}", C_COPY_TXT, 9.5),
                     (f"copies among cage replicas:  {pct(float(d[f'{m}/copy_frac_cage'][i]))}", C_COPY_TXT, 9.5)]
            if t >= d["t_fr"]:
                rows.append((f"families alive:  {int(d[f'{m}/n_families'][i]):,} of {N:,}", C_INK, 9.5))
        else:
            rows += [("no births, no deaths", C_INK2, 9.5), (f"all {N:,} replicas are original walkers", C_INK2, 9.5)]
        for k, (txt, c, fs) in enumerate(rows):
            ax.text(0.0, 0.655 - 0.135 * k, txt, transform=ax.transAxes, fontsize=fs, color=c, ha="left", va="center")

        # ---- (d) learned free energy --------------------------------------------------------------------
        ax = _ax_in(fig, x0 + W_METER + 0.5, Y_MD, COL_W - W_METER - 0.5, H_MD)
        Fk = d["Fk"]
        ax.axvspan(-WINDOW_HALF, WINDOW_HALF, color=C_BAND_WIN, lw=0)
        for zl in (-CAGE_MIN, CAGE_MIN):
            ax.axvline(zl, color=C_INK2, lw=0.5, ls=(0, (4, 3)), alpha=0.5)
        ax.plot(d["zg"], Fk, color=C_REF, lw=1.2, ls=(0, (5, 3)))
        ax.plot(d["zg"], d[f"{m}/Fk"][i], color=colour, lw=2.0)
        ax.text(0.015, 0.92, f"$e_F(t)$ = {float(d[f'{m}/eFk'][i]):.2f} $k_BT$", transform=ax.transAxes, fontsize=10, ha="left",
                va="top", bbox=_bbox())
        ax.set_xlim(-zh, zh); ax.set_ylim(Fk.min() - 1.5, Fk.max() + 2.5)
        ax.set_xlabel("$z_{CV}$ (Å)", fontsize=9, labelpad=1); ax.set_ylabel("$\\beta F$  [$k_BT$]", fontsize=9, labelpad=2)
        ax.tick_params(labelsize=8.5); ax.grid(True, color=C_GRID, lw=0.6)
        ax.set_title("learned $\\beta\\hat F_t$ (colour) vs reference (dashed)", fontsize=9.5, loc="left", pad=3, color=C_INK2)

    # ---- bottom strip: e_F(t) and R_W(t) of both arms -------------------------------------------------
    for k, which in enumerate(("eF", "RW")):
        ax = _ax_in(fig, COL_X[k], Y_BS, COL_W, H_BS)
        for m, colour, label in ARMS:
            if m not in d["methods"]:
                continue
            y = d[f"{m}/eFk"] if which == "eF" else d[f"{m}/RW"]
            _series_past(ax, snap_t, y, t, colour, label=label, log=(which == "eF"))
        ax.axvline(t, color=C_INK2, lw=0.8, alpha=0.6)
        ax.axvline(d["t_fr"], color=C_FR, lw=0.8, ls=(0, (2, 2)), alpha=0.8)
        ax.set_xlim(0, T); ax.tick_params(labelsize=8.5); ax.grid(True, which="both", color=C_GRID, lw=0.6)
        ax.set_xlabel("time  $t$", fontsize=9.5, labelpad=1)
        if which == "eF":
            ax.set_yscale("log"); ax.set_ylim(*d["eF_lim"])
            ax.text(d["t_fr"], d["eF_lim"][0] * 1.2, " FR starts", fontsize=8.5, color=C_FR, ha="left", va="bottom")
            ax.set_ylabel("$e_F(t)$  [$k_BT$]", fontsize=9.5, labelpad=2)
            ax.set_title("(e) free-energy error $e_F(t)$, log scale", fontsize=10, loc="left", pad=4)
            ax.legend(loc="upper right", fontsize=9.5, frameon=False, ncol=2, columnspacing=1.2, borderaxespad=0.3)
        else:
            ax.axhline(1.0, color=C_INK, lw=1.0, ls=(0, (5, 3)))
            ax.text(T * 0.995, 1.03, "uniform target  $R_W$ = 1", fontsize=8.5, color=C_INK2, ha="right", va="bottom")
            ax.set_ylim(0, d["RW_lim"]); ax.set_ylabel("$R_W(t)$", fontsize=9.5, labelpad=2)
            ax.set_title("(f) window occupancy $R_W(t) = \\hat p_t(W)/q(W)$, both arms", fontsize=10, loc="left", pad=4)
            ax.legend(loc="lower right", fontsize=9.5, frameon=False, ncol=2, columnspacing=1.2, borderaxespad=0.3)


# ----------------------------------------------------------------------------------------------
# act III
# ----------------------------------------------------------------------------------------------
def _act3(fig, d, i, p):
    import matplotlib as mpl
    a, N, T, zh = d["a_pseudo"], d["N"], d["T"], d["zh"]
    snap_t = d["snap_t"]; t = snap_t[i]; hold = p >= 0.6 - 1e-9
    K = len(d["fam_ids"]); k_main = int(d.get("fam_main", 0))
    COL_X, COL_W = (0.75, 8.35), 6.9
    Y_K, H_K = 4.55, 3.2
    Y_C, H_C = 3.2, 0.55
    Y_R, H_R = 1.7, 1.2
    X_FULL, W_FULL = COL_X[0], COL_X[1] + COL_W - COL_X[0]
    cmap = mpl.colormaps["RdBu_r"].copy(); cmap.set_bad(C_PAPER); norm = mpl.colors.Normalize(-LOG2_CLIP, LOG2_CLIP)

    for col, (m, colour, label) in enumerate(ARMS):
        if m not in d["methods"]:
            continue
        x0 = COL_X[col]
        ax = _ax_in(fig, x0, Y_K, COL_W, H_K)
        img = np.where(snap_t[None, :] <= t + 1e-9, d[f"{m}/log2R"].T, np.nan)
        ax.imshow(np.ma.masked_invalid(img), extent=(snap_t[0], snap_t[-1], -zh, zh), origin="lower", aspect="auto",
                  cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
        for zl in (-CAGE_MIN, -WINDOW_HALF, WINDOW_HALF, CAGE_MIN):
            ax.axhline(zl, color=C_INK2, lw=0.5, ls=(0, (4, 3)), alpha=0.6, zorder=2)
        for yy, lab_ in ((-(zh + CAGE_MIN) / 2, "cage"), (0.0, "window"), ((zh + CAGE_MIN) / 2, "cage")):
            ax.text(0.995, yy, lab_, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5, color=C_INK2,
                    zorder=6, bbox=_bbox(alpha=0.7, pad=0.12))
        if m == "fr_uniform":
            sel = d[f"{m}/ev_t"] <= t + 1e-9
            if sel.any():
                ax.scatter(d[f"{m}/ev_t"][sel], d[f"{m}/ev_zdie"][sel], s=5, c=C_DIE, linewidths=0, alpha=0.75, zorder=3)
                ax.scatter(d[f"{m}/ev_t"][sel], d[f"{m}/ev_zbirth"][sel], s=5, c=C_BIRTH, linewidths=0, alpha=0.75, zorder=3)
        for k in sorted(range(K), key=lambda q: q == k_main):          # the main family last (on top)
            c = C_FAM[k]; main = k == k_main
            lw, al, ms = (0.8, 0.95, 3.6) if main else (0.55, 0.35, 2.4)
            for wl in d[f"{m}/worldlines"][k]:
                if wl["t0"] > t + 1e-9:
                    continue
                keep = wl["t_fill"] <= t + 1e-9
                ax.plot(wl["t"][keep], wl["z"][keep], color=c, lw=lw, alpha=al, zorder=4 + main, solid_capstyle="round")
                if wl["born"]:
                    ax.plot([wl["t0"]], [wl["z0"]], "o", color=c, ms=ms, mec="white", mew=0.5, alpha=max(al, 0.6), zorder=5 + main)
        if not hold:
            ax.axvline(t, color=C_INK, lw=1.0, alpha=0.8, zorder=6)
        ax.axvline(d["t_fr"], color=C_INK2, lw=0.7, ls=(0, (2, 2)), alpha=0.8, zorder=2)
        ax.text(d["t_fr"], zh * 0.97, " FR starts", fontsize=8, color=C_INK2, ha="left", va="top", zorder=6,
                bbox=_bbox(alpha=0.75, pad=0.12))
        ax.set_xlim(snap_t[0], snap_t[-1]); ax.set_ylim(-zh, zh)
        ax.tick_params(labelsize=8.5)
        ax.set_xlabel("time  $t$", fontsize=9.5, labelpad=1)
        if col == 0:
            ax.set_ylabel("$z_{CV}$ (Å)", fontsize=9.5, labelpad=2)
        ax.set_title(label, color=colour, fontsize=15, fontweight="bold", pad=5, loc="left")
        ax.set_title("density / uniform ($\\log_2$);  worldlines (largest family in full colour, others faint)"
                     if m != "fr_uniform" else "density / uniform ($\\log_2$);  worldlines;  deaths · births",
                     fontsize=9.5, color=C_INK2, pad=5, loc="right")
        # family sizes: one text row under the x-axis label, outside the image (largest family in bold)
        anchors = [("left", 0.0), ("center", 0.5), ("right", 1.0)] if K == 3 else [("left", k / max(K, 1)) for k in range(K)]
        for k, r in enumerate(d["fam_ids"]):
            n_mem = int(d[f"{m}/fam_size"][k][i]); n_ever = len(d[f"{m}/worldlines"][k])
            txt = f"replica {r}: {n_mem} member{'s' if n_mem != 1 else ''}" + (f" ({n_ever} ever)" if n_ever > 1 else "")
            ha, fx = anchors[k]
            fig.text((x0 + fx * COL_W) / FIG_W, (Y_K - 0.5) / FIG_H, txt, fontsize=8.5, color=C_FAM[k],
                     fontweight="bold" if k == k_main else "normal", ha=ha, va="center")
    cax = _ax_in(fig, COL_X[1] + COL_W + 0.1, Y_K + 0.5, 0.11, H_K - 1.0)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical")
    cb.set_ticks([-LOG2_CLIP, 0, LOG2_CLIP])
    cb.set_ticklabels([f"≤ −{LOG2_CLIP:g}\ndeficit", "0", f"≥ +{LOG2_CLIP:g}\nexcess"])
    cax.tick_params(labelsize=7.5, length=2, pad=1)
    cax.text(0.5, 1.03, "$\\log_2 \\hat p_t/q$", transform=cax.transAxes, ha="center", va="bottom", fontsize=8, color=C_INK2)

    # ---- fraction of copies among the window replicas (FR arm) -------------------------------------------
    fr = "fr_uniform" if "fr_uniform" in d["methods"] else d["methods"][-1]
    ax = _ax_in(fig, X_FULL, Y_C, W_FULL, H_C)
    _series_past(ax, snap_t, d[f"{fr}/copy_frac_window"], t, C_FR, label="among the replicas in the window", lw=2.0)
    y_c = d[f"{fr}/copy_frac_cage"]; keep = np.isfinite(y_c)
    ax.plot(snap_t[keep], y_c[keep], color=C_COPY_TXT, lw=0.9, ls=(0, (4, 2)), alpha=0.3)
    past = keep & (snap_t <= t + 1e-9)
    if past.any():
        ax.plot(snap_t[past], y_c[past], color=C_COPY_TXT, lw=1.2, ls=(0, (4, 2)), label="among the replicas in the cages")
    ax.axvline(d["t_fr"], color=C_INK2, lw=0.7, ls=(0, (2, 2)), alpha=0.8)
    if not hold:
        ax.axvline(t, color=C_INK, lw=1.0, alpha=0.8)
    ax.set_xlim(0, T); ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=8.5, labelbottom=False); ax.grid(True, color=C_GRID, lw=0.6)
    ax.set_ylabel("copies", fontsize=9, labelpad=2)
    ax.set_title("ABF + Fisher–Rao: fraction of replicas that are Fisher–Rao COPIES (id ≥ N) — the window fills with copies "
                 "while originals trickle in   (ABF: none)", fontsize=10, loc="left", pad=3)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.85, edgecolor="none", ncol=2, columnspacing=1.5,
              borderaxespad=0.2, handlelength=1.8)

    # ---- shared R_W(t) -----------------------------------------------------------------------------------
    ax = _ax_in(fig, X_FULL, Y_R, W_FULL, H_R)
    for m, colour, label in ARMS:
        if m in d["methods"]:
            _series_past(ax, snap_t, d[f"{m}/RW"], t, colour, label=label, lw=2.0)
    ax.axhline(1.0, color=C_INK, lw=1.0, ls=(0, (5, 3)))
    ax.axvline(d["t_fr"], color=C_INK2, lw=0.7, ls=(0, (2, 2)), alpha=0.8)
    if not hold:
        ax.axvline(t, color=C_INK, lw=1.0, alpha=0.8)
    ax.set_xlim(0, T); ax.set_ylim(0, d["RW_lim"]); ax.tick_params(labelsize=8.5); ax.grid(True, color=C_GRID, lw=0.6)
    ax.set_ylabel("$R_W(t)$", fontsize=9.5, labelpad=2); ax.set_xlabel("time  $t$", fontsize=9.5, labelpad=1)
    ax.set_title("window occupancy $R_W(t) = \\hat p_t(W)/q(W)$, both arms;  dashed: the uniform target", fontsize=10, loc="left", pad=3)
    ax.text(0.5, 0.9, "ABF approaches the target slowly; FR quickly", transform=ax.transAxes, ha="center", va="top", fontsize=11,
            color=C_INK, fontweight="bold", bbox=_bbox(alpha=0.85))
    ax.legend(loc="lower right", fontsize=9.5, frameon=False, ncol=2, columnspacing=1.2, borderaxespad=0.3)

    # ---- closing statement (hold) --------------------------------------------------------------------------
    if hold:
        fig.text(0.5, 1.1 / FIG_H, "physical dynamics discovers the window   →   Fisher–Rao establishes it",
                 fontsize=15, fontweight="bold", ha="center", va="center")
        t_ref, pr = d["copy_ref"]
        line2 = (f"at t = {t_ref:g}, {pr.get(fr, float('nan')):.0f} % of the replicas in the window are Fisher–Rao copies "
                 f"(ABF: {pr.get('abf', 0.0):.0f} %);   largest family {d[f'{fr}/max_family']} of {N:,}")
        fig.text(0.5, 0.72 / FIG_H, line2, fontsize=11.5, ha="center", va="center", color=C_INK)
        if all(m in d["methods"] for m, _, _ in ARMS):
            eA, eF_ = float(d["abf/eFk"][-1]), float(d["fr_uniform/eFk"][-1])
            d_fin = 100.0 * (eF_ / eA - 1.0) if eA > 0 else float("nan")
            fig.text(0.5, 0.36 / FIG_H, f"$e_F(T)$:  ABF {eA:.2f} $k_BT$,  ABF + FR {eF_:.2f} $k_BT$  ({d_fin:+.0f} %)",
                     fontsize=11.5, ha="center", va="center", color=C_INK2)
    else:
        fig.text(0.5, 0.7 / FIG_H, "the kymograph replays the whole run: each column is one snapshot of the replica density; "
                 "the coloured lines are the highlighted families", fontsize=11, ha="center", va="center", color=C_INK2)


# ----------------------------------------------------------------------------------------------
# one frame
# ----------------------------------------------------------------------------------------------
def draw_frame(args):
    i_frame, path = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = G["d"]
    act, i, p = G["schedule"][i_frame]
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.edgecolor": C_INK2, "axes.linewidth": 0.8, "xtick.color": C_INK2, "ytick.color": C_INK2,
        "axes.labelcolor": C_INK, "text.color": C_INK, "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    _header(fig, d, act, float(d["snap_t"][i]))
    if act == 1:
        _act1(fig, d, i, p)
    elif act == 2:
        _act2(fig, d, i, p, i_frame)
    else:
        _act3(fig, d, i, p)
    if G.get("debug_bbox"):
        base._report_clipping(fig, i_frame)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return i_frame


# ----------------------------------------------------------------------------------------------
# render / storyboard
# ----------------------------------------------------------------------------------------------
def _prepare(out_dir, data=None, *, n_families=3, event_window=3, act_seconds=(6, 16, 10), fps=30, family_rule="largest"):
    d = load_ensemble_data(data or os.path.join(out_dir, "movie_data.npz"), n_families=n_families, family_rule=family_rule)
    G["d"] = d
    G["event_window"] = int(event_window)
    G["schedule"] = build_schedule(d, act_seconds, fps)
    G["n_act"] = [int(round(s * fps)) for s in act_seconds]
    return d


def render(out_dir, *, fps=30, workers=24, event_window=3, n_families=3, only=None, keep_frames=False,
           movie_name=MOVIE, act_seconds=(6, 16, 10), data=None, family_rule="largest"):
    from movie_common import render_movie
    _prepare(out_dir, data, n_families=n_families, event_window=event_window, act_seconds=act_seconds, fps=fps,
             family_rule=family_rule)
    n_frames = len(G["schedule"])
    print(f"schedule: {n_frames} frames = {' + '.join(str(n) for n in G['n_act'])} at {fps} fps "
          f"({n_frames / fps:.1f} s + 2 s hold)")
    return render_movie(draw_frame, n_frames, stride=1, frames_dir=os.path.join(out_dir, "frames_ensemble"), workers=workers,
                        fps=fps, mp4=os.path.join(out_dir, movie_name + ".mp4"), only=only, hold_seconds=2.0,
                        keep_frames=keep_frames, root=ROOT)


def storyboard(out_dir, *, data=None, n_families=3, event_window=3, act_seconds=(6, 16, 10), fps=30, debug_bbox=False,
               progress=(0.86, 0.47, 0.8), family_rule="largest"):
    """One representative frame per act: act I with the replica dots fading in, act II in the middle of the
    lineage bloom (t = t_fr + 6), act III on the hold."""
    os.makedirs(out_dir, exist_ok=True)
    _prepare(out_dir, data, n_families=n_families, event_window=event_window, act_seconds=act_seconds, fps=fps,
             family_rule=family_rule)
    G["debug_bbox"] = bool(debug_bbox)
    sched = G["schedule"]; paths = []
    off = 0
    for act, (n, pr) in enumerate(zip(G["n_act"], progress), start=1):
        k = off + int(round(pr * (n - 1)))
        path = os.path.join(out_dir, f"storyboard_act{act}.png")
        draw_frame((k, path))
        a_, i_, p_ = sched[k]
        print(f"storyboard act {act}: frame {k} (snapshot {i_}, t = {G['d']['snap_t'][i_]:.2f}, progress {p_:.2f}) -> {path}")
        paths.append(path); off += n
    G["debug_bbox"] = False
    return paths


# ----------------------------------------------------------------------------------------------
# self-test: a consistent synthetic movie_data.npz WITH the genealogy keys
# ----------------------------------------------------------------------------------------------
def synthetic_ensemble_data(path, *, N=96, n_snap=240, seed=3, T=60.0):
    """Random walks along the CV that start in the cages and leak into the window as the landscape is
    flattened, both arms identical until t_fr, then FR events in the second arm with consistent
    snap_wid / snap_anc / snap_ev_slots / snap_ev_ids (deaths in the cages, births from window / neck
    replicas so the discoverer families bloom).  Framework from cache/lta/framework.npz, F / U / TS from the
    150 K reference."""
    rng = np.random.default_rng(seed)
    fw = np.load(FRAMEWORK_NPZ, allow_pickle=True); ref = np.load(REFERENCE_T150, allow_pickle=True)
    a = float(fw["a_pseudo"]); L = float(fw["box"]); kT = float(ref["kT"]); beta = 1.0 / kT; zh = 0.5 * a
    o_pos = np.asarray(fw["o_pos"], np.float32)
    si_pos = np.asarray(fw["pos"])[np.asarray(fw["kind"]) == "Si"].astype(np.float32)
    n_bins = 180
    grid = -np.pi + (np.arange(n_bins) + 0.5) * 2 * np.pi / n_bins
    assert np.allclose(grid, ref["grid_phi"]), "engine grid != reference grid"
    zg = grid * a / (2 * np.pi)
    F_ref = ref["F"] - ref["F"].mean(); U_ref = ref["U"] - ref["U"].mean(); mTS_ref = F_ref - U_ref
    dt = 2.0e-4
    snap_t = np.linspace(0.0, T, n_snap); snap_step = np.round(snap_t / dt).astype(np.int64)
    t_warm = t_fr = t_burn = 4.0
    k_fr = int(np.searchsorted(snap_t, t_fr))
    # overdamped walk on the periodic CV in scale(t) * beta F_ref(z); the scale decays as "ABF flattens the landscape"
    gF = np.gradient(beta * F_ref, zg)
    gF = 0.5 * (gF + np.roll(gF, 1))

    def dFdz(z):
        return np.interp(z, zg, gF, period=a)

    def scale(t):
        return 0.08 + 0.55 * np.exp(-t / 5.0)

    D, n_sub = 2.0, 8
    dts = (snap_t[1] - snap_t[0]) / n_sub

    def advance(z, t):
        sc = scale(t)
        for _ in range(n_sub):
            z = z - D * sc * dFdz(z) * dts + np.sqrt(2 * D * dts) * rng.normal(size=z.size)
            z = np.mod(z + zh, a) - zh
        return z

    zA = np.empty((n_snap, N))
    zA[0] = np.mod(rng.choice([-1.0, 1.0], N) * (zh - np.abs(rng.normal(0, 1.0, N))) + zh, a) - zh
    for s in range(1, n_snap):
        zA[s] = advance(zA[s - 1], snap_t[s])
    # FR arm: shares the walk until k_fr, then its own noise + events
    zF = zA.copy()
    wid = np.tile(np.arange(N, dtype=np.int64), (n_snap, 1)); anc = wid.copy()
    cell = rng.integers(0, 2, N)                       # which of the two cages along x each replica started in
    die_rows, birth_rows, ev_slots, ev_ids = [], [], [], []
    next_id = N
    com_prev = None
    for s in range(k_fr + 1, n_snap):
        zF[s] = advance(zF[s - 1], snap_t[s])
        wid[s] = wid[s - 1]; anc[s] = anc[s - 1]
        if (s - k_fr) % 2 == 1 and s < n_snap - 2:
            n_ev = int(rng.integers(1, 4))
            cage = np.flatnonzero(np.abs(zF[s - 1]) > CAGE_MIN)
            src_pool = np.flatnonzero(np.abs(zF[s - 1]) < WINDOW_HALF)
            if src_pool.size == 0:
                src_pool = np.flatnonzero(np.abs(zF[s - 1]) < 3.0)
            if cage.size < n_ev or src_pool.size == 0:
                continue
            dying = rng.choice(cage, n_ev, replace=False)
            for slot in dying:
                src = int(rng.choice(src_pool))
                zd, zs = float(zF[s - 1, slot]), float(zF[s - 1, src])
                cd = np.array([zd + a * cell[slot], zh + rng.normal(0, 2.0), zh + rng.normal(0, 2.0)])
                cs = np.array([zs + a * cell[src], zh + rng.normal(0, 0.6), zh + rng.normal(0, 0.6)])
                die_rows.append([s, 2 * np.pi * zd / a, *cd]); birth_rows.append([s, 2 * np.pi * zs / a, *cs])
                ev_slots.append([int(slot), src]); ev_ids.append([next_id, int(wid[s - 1, src])])
                zF[s, slot] = np.mod(zF[s, src] + rng.normal(0, 0.15) + zh, a) - zh
                wid[s, slot] = next_id; anc[s, slot] = anc[s - 1, src]; cell[slot] = cell[src]
                next_id += 1

    def arm(z, noise_scale):
        rho = np.where(np.abs(z) > CAGE_MIN, 2.2, np.where(np.abs(z) < WINDOW_HALF, 0.6, 1.3))
        yz = np.empty((n_snap, N, 2)); yz[0] = rng.normal(size=(N, 2))
        for s_ in range(1, n_snap):                                   # AR(1) in units of the local radius
            yz[s_] = 0.6 * yz[s_ - 1] + 0.8 * rng.normal(size=(N, 2))
        com = np.stack([z + a * rng.integers(0, 2, N)[None, :].repeat(n_snap, 0), zh + rho * yz[..., 0], zh + rho * yz[..., 1]], axis=-1)
        u = rng.normal(size=(n_snap, N, 3)); u /= np.linalg.norm(u, axis=-1, keepdims=True)
        q = np.stack([com + 0.77 * u, com - 0.77 * u], axis=2).astype(np.float32)
        phi = (2 * np.pi * z / a).astype(np.float32)
        modes = np.stack([np.cos(k * grid + rng.uniform(0, 2 * np.pi)) for k in (1, 2, 3, 5)])
        snap_F = np.empty((n_snap, n_bins))
        for s in range(n_snap):
            amp = noise_scale * (6.0 * np.exp(-snap_t[s] / 15.0) + 0.3)
            snap_F[s] = F_ref + amp * (rng.normal(size=4) @ modes)
        snap_F -= snap_F.mean(axis=1, keepdims=True)
        eF = np.sqrt(((snap_F - F_ref) ** 2).mean(axis=1))
        return dict(q=q, phi=phi, F=snap_F.astype(np.float32), eF=eF.astype(np.float32),
                    frac_window=(np.abs(z) < WINDOW_HALF).mean(axis=1).astype(np.float32),
                    frac_cage=(np.abs(z) > CAGE_MIN).mean(axis=1).astype(np.float32))

    out = dict(temperature=np.float64(ref["temperature"]), kT=np.float64(kT), beta=np.float64(beta), a_pseudo=np.float64(a),
               box=np.float64(L), n_bins=np.int64(n_bins), seed=np.int64(seed), featured=np.int64(0), t_fr=np.float64(t_fr),
               t_warm=np.float64(t_warm), t_burn=np.float64(t_burn), fr_rate=np.float64(0.2), git_rev="selftest",
               reference_path=os.path.relpath(REFERENCE_T150, ROOT), sim_json=json.dumps({"n_replicas": N, "dt": dt}),
               params_json=json.dumps({"temperature": float(ref["temperature"])}), methods=np.array(["abf", "fr_uniform"]),
               arms_identical_until_t=np.float64(snap_t[k_fr]), N=np.int64(N), T=np.float64(T), dt=np.float64(dt),
               grid=grid, z_grid=zg, dphi=np.float64(2 * np.pi / n_bins), F_ref=F_ref.astype(np.float32),
               U_ref=U_ref.astype(np.float32), mTS_ref=mTS_ref.astype(np.float32),
               dF_kT=np.float64(float(ref["dF_barrier"]) * beta), dU_kT=np.float64(float(ref["dU_barrier"]) * beta),
               mTdS_kT=np.float64(float(ref["mTdS_barrier"]) * beta), o_pos=o_pos, si_pos=si_pos, snap_t=snap_t,
               snap_step=snap_step, save_t=snap_t[::3])
    A = arm(zA, 1.0); B = arm(zF, 0.7)
    empty5 = np.zeros((0, 5), np.float32); empty2i = np.zeros((0, 2), np.int32); empty2l = np.zeros((0, 2), np.int64)
    for m, r, w, an, die, birth, sl, ids in (
            ("abf", A, np.tile(np.arange(N, dtype=np.int32), (n_snap, 1)), np.tile(np.arange(N, dtype=np.int32), (n_snap, 1)),
             empty5, empty5, empty2i, empty2l),
            ("fr_uniform", B, wid.astype(np.int32), anc.astype(np.int32), np.asarray(die_rows, np.float32).reshape(-1, 5),
             np.asarray(birth_rows, np.float32).reshape(-1, 5), np.asarray(ev_slots, np.int32).reshape(-1, 2),
             np.asarray(ev_ids, np.int64).reshape(-1, 2))):
        counts = rng.poisson(3.0, (n_snap, n_bins)).astype(np.float32).cumsum(axis=0)
        cross = np.cumsum(rng.poisson(0.6, n_snap)).astype(np.int64)
        out.update({f"{m}/snap_q": r["q"], f"{m}/snap_phi": r["phi"], f"{m}/snap_wid": w, f"{m}/snap_anc": an, f"{m}/snap_F": r["F"],
                    f"{m}/snap_counts": counts, f"{m}/snap_crossings": cross, f"{m}/snap_eF": r["eF"],
                    f"{m}/snap_ev_die": die, f"{m}/snap_ev_birth": birth, f"{m}/snap_ev_slots": sl, f"{m}/snap_ev_ids": ids,
                    f"{m}/save_eF": r["eF"][::3], f"{m}/l2_f": np.float64(r["eF"][-1]),
                    f"{m}/integrated_l2_f": np.float64(np.trapezoid(r["eF"], snap_t) / T),
                    f"{m}/total_replacement_events": np.float64(len(die)), f"{m}/min_ess_frac": np.float64(0.9),
                    f"{m}/max_wmax": np.float64(0.01), f"{m}/n_cage_crossings": np.float64(cross[-1]),
                    f"{m}/frac_window": r["frac_window"], f"{m}/frac_cage": r["frac_cage"]})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"synthetic data: {path}  (N = {N}, {n_snap} snapshots, {len(die_rows)} FR events, "
          f"window fraction at T: ABF {A['frac_window'][-1]:.2f}, FR {B['frac_window'][-1]:.2f})")
    return path


def selftest(out_dir, workers=4, act_seconds=(1.0, 2.0, 1.0), fps=12):
    """Fabricate the synthetic record, draw the storyboard with the clipping report, then the tiny movie."""
    os.makedirs(out_dir, exist_ok=True)
    synthetic_ensemble_data(os.path.join(out_dir, "movie_data.npz"))
    storyboard(out_dir, debug_bbox=True)
    render(out_dir, fps=fps, workers=workers, act_seconds=act_seconds, keep_frames=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=OUT_DIR, help="directory holding movie_data.npz (unless --data); frames, PNGs and the mp4 go there")
    ap.add_argument("--data", default=None, help="movie_data.npz to read (default: <out>/movie_data.npz)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--event-window", type=int, default=3, help="flash FR events from the last k frames")
    ap.add_argument("--n-families", type=int, default=3, help="highlighted families (first to reach the window)")
    ap.add_argument("--family-rule", choices=("first", "largest"), default="largest",
                    help="which families to highlight: the largest final ones (default) or the first to reach the window")
    ap.add_argument("--only", type=int, default=None, help="render one frame index (layout check)")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--movie-name", default=MOVIE)
    ap.add_argument("--act-seconds", type=float, nargs=3, default=(6.0, 16.0, 10.0), metavar=("ACT1", "ACT2", "ACT3"))
    ap.add_argument("--storyboard", action="store_true", help="one representative frame per act into <out>/storyboard_act{1,2,3}.png")
    ap.add_argument("--selftest", action="store_true", help="synthetic data in --out (default: a scratch directory)")
    a = ap.parse_args()
    if a.selftest:
        out = a.out if a.out != OUT_DIR else os.path.join(os.environ.get("TMPDIR", "/tmp"), "lta_ensemble_selftest")
        selftest(out, workers=min(a.workers, 8))
        return
    if a.storyboard:
        storyboard(a.out, data=a.data, n_families=a.n_families, event_window=a.event_window, act_seconds=tuple(a.act_seconds),
                   fps=a.fps, debug_bbox=True, family_rule=a.family_rule)
        return
    render(a.out, fps=a.fps, workers=a.workers, event_window=a.event_window, n_families=a.n_families, only=a.only,
           keep_frames=a.keep_frames, movie_name=a.movie_name, act_seconds=tuple(a.act_seconds), data=a.data,
           family_rule=a.family_rule)


if __name__ == "__main__":
    main()
