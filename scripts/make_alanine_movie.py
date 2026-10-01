#!/usr/bin/env python
"""Alanine dipeptide movies: ABF vs ABF + Fisher-Rao on the Ramachandran plot (docs/ALANINE_MOVIES.md).

    python scripts/make_alanine_movie.py --case ala2d [--only 300] [--stride 2] [--fps 30] [--workers 48]
    python scripts/make_alanine_movie.py --case phi
    python scripts/make_alanine_movie.py --case psi

Reads the paired single-seed runs under results/alanine_movie/<case>/ (store_snapshots records) and the
300 K reference; renders every ``stride``-th snapshot with the fork pool of scripts/movie_common.py and
encodes an mp4.  The error series drawn is the engine's own live PMF against the reference, per snapshot.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS); sys.path.insert(0, os.path.join(SCRIPTS, "..", "src"))
from movie_common import render_movie                                       # noqa: E402
from alanine.basins import from_reference                                   # noqa: E402
from alanine.core1d_ala import marginal_reference_1d                        # noqa: E402
from alanine.metrics_ala import aligned_l2, build_masks                     # noqa: E402

ROOT = os.path.join(SCRIPTS, "..")
REF = os.path.join(ROOT, "results/alanine/reference/reference.npz")
C_ABF, C_FR, C_DIE, C_BIRTH = "#1f4e79", "#b5451b", "#d62728", "#2ca02c"
C_INK, C_INK2 = "#222222", "#555555"
BASIN_C = {0: "#1f77b4", 1: "#2ca02c", 2: "#ff7f0e", -1: "#7b3294"}
BASIN_XY = ((-74, 56), (-152, 156), (63, -48))
F_MAX = 90.0          # full range of the reference (its finite maximum is ~93 kJ/mol at 300 K)
G = {}


def _load(case):
    root = os.path.join(ROOT, "results/alanine_movie", "ala1d" if case in ("phi", "psi") else "ala2d")
    stages = ("abf", "fr") if case == "ala2d" else (f"{case}_abf", f"{case}_fr")
    refd = np.load(REF, allow_pickle=True); F2 = refd["F"]; rmeta = json.loads(str(refd["meta"]))
    kT = float(rmeta["kT_kJ"]); n = int(rmeta["n_grid"])
    bm, _ = from_reference(REF)
    pack2 = build_masks(F2, kT); w2 = pack2["weights"]["equilibrium"]
    pops = bm.population(F2, kT)
    d = dict(case=case, kT=kT, n=n, F2=F2, names=bm.names, pops=[pops[nm] for nm in bm.names],
             grid_deg=np.degrees(-np.pi + (np.arange(n) + 0.5) * 2 * np.pi / n))
    if case != "ala2d":
        ax = 0 if case == "phi" else 1
        F1 = marginal_reference_1d(F2, kT, ax); pack1 = build_masks(F1, kT)
        d["F1"] = F1; d["w1"] = pack1["weights"]["equilibrium"]; d["F1_fin"] = pack1["F"]
        phi_neg = (np.degrees(-np.pi + (np.arange(n) + 0.5) * 2 * np.pi / n) < 0)[:, None] & np.ones((1, n), bool)
        d["F1_vis"] = marginal_reference_1d(F2, kT, ax, mask=phi_neg) if case == "psi" else None
    for key, st in zip(("abf", "fr"), stages):
        f = glob.glob(os.path.join(root, st, "raw", "*.npz"))
        if not f:
            raise SystemExit(f"missing run for stage {st} under {root}")
        z = np.load(f[0], allow_pickle=True); meta = json.loads(str(z["meta"]))
        r = dict(t=z["snap_times"], steps=z["snap_steps"], wid=z["snap_wid"], basin=z["snap_basin"],
                 pmf=z["snap_pmf"], die=z["snap_ev_die"], birth=z["snap_ev_birth"], meta=meta,
                 n_events=int(z["total_events"][0]), rate=float(meta.get("fr_rate", 0.0)),
                 save_t=np.asarray(z["times"], float), basin_frac=z["basin_frac"][:, 0, :], kl=z["kl_uniform"][:, 0])
        if case == "ala2d":
            r["phi"], r["psi"] = z["snap_phi"], z["snap_psi"]
            r["err"] = np.array([aligned_l2(r["pmf"][i].astype(float), pack2["F"], w2) for i in range(len(r["t"]))])
        else:
            cvv, hid = z["snap_cv"], z["snap_hidden"]
            r["phi"], r["psi"] = (cvv, hid) if case == "phi" else (hid, cvv)
            r["err"] = np.array([aligned_l2(r["pmf"][i].astype(float), d["F1_fin"], d["w1"]) for i in range(len(r["t"]))])
            if case == "psi":
                pv = build_masks(d["F1_vis"], kT)
                r["err_vis"] = np.array([aligned_l2(r["pmf"][i].astype(float), pv["F"], pv["weights"]["equilibrium"]) for i in range(len(r["t"]))])
        d[key] = r
    d["N"] = int(d["abf"]["meta"]["n_replicas"]); d["T"] = float(d["abf"]["t"][-1])
    assert np.array_equal(d["abf"]["steps"], d["fr"]["steps"])
    return d


def _aligned_pmf2(P, F2, w):
    ok = (w > 0) & np.isfinite(F2)
    c = ((P - F2)[ok] * w[ok]).sum() / w[ok].sum()
    return P - c + np.nanmin(F2[np.isfinite(F2)]) - np.nanmin(F2[np.isfinite(F2)])


def draw_frame(args):
    i, path = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.gridspec import GridSpec
    d = G["d"]; win = G["event_window"]; case = d["case"]
    t = float(d["abf"]["t"][i]); T = d["T"]; N = d["N"]; n = d["n"]; g = d["grid_deg"]
    F2 = d["F2"]; Fbg = np.where(np.isfinite(F2), F2 - np.nanmin(F2[np.isfinite(F2)]), F_MAX)
    ext = (-180, 180, -180, 180)
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.edgecolor": C_INK2, "axes.linewidth": 0.8,
                         "xtick.color": C_INK2, "ytick.color": C_INK2, "axes.labelcolor": C_INK, "text.color": C_INK,
                         "figure.facecolor": "white"})
    fig = plt.figure(figsize=(16, 9), dpi=120)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 0.92], height_ratios=[1.0, 0.92], hspace=0.30, wspace=0.22,
                  left=0.05, right=0.985, top=0.868, bottom=0.07)
    gsr = gs[:, 2].subgridspec(3, 1, hspace=0.55)
    title = {"ala2d": "Alanine dipeptide, CV = (φ, ψ):  ABF  vs  ABF + Fisher–Rao birth–death",
             "phi": "Alanine dipeptide, CV = φ alone (ψ hidden):  ABF  vs  ABF + Fisher–Rao",
             "psi": "Alanine dipeptide, CV = ψ alone (φ hidden):  ABF  vs  ABF + Fisher–Rao"}[case]
    fig.text(0.055, 0.968, title, fontsize=17, fontweight="bold", ha="left", va="center")
    fig.text(0.055, 0.937, f"same initial ensemble and Langevin noise in both columns;  N = {N} walkers, 300 K;  "
             f"histogram mean-force estimator (c_min 800, 5 ps ramp);  FR: uniform target, rate {d['fr']['rate']:g} from 0.5 ps",
             fontsize=11, color=C_INK2, ha="left", va="center")
    fig.text(0.985, 0.968, f"t = {t:6.2f} / {T:g} ps", fontsize=17, ha="right", va="center", family="monospace")
    fig.text(0.055, 0.912, "φ, ψ are periodic (±180° coincide).  ABF flattens F, so the biased walkers cover the whole torus, high-energy regions "
             "included.  Purple = outside the three reference wells;  hatched = never sampled by the reference.",
             fontsize=9.5, color=C_INK2, ha="left", va="center", style="italic")
    for col, (key, colour, label) in enumerate((("abf", C_ABF, "ABF"), ("fr", C_FR, "ABF + Fisher–Rao"))):
        r = d[key]
        phi = np.degrees(r["phi"][i]); psi = np.degrees(r["psi"][i]); b = r["basin"][i]
        # ---- Ramachandran -------------------------------------------------------------------
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(np.clip(Fbg.T, 0, F_MAX), origin="lower", extent=ext, cmap="Greys", vmin=-10, vmax=F_MAX * 1.35, aspect="equal", interpolation="bilinear")
        ax.contour(g, g, Fbg.T, levels=[5, 10, 20, 40, 60, 80], colors="white", linewidths=0.5, alpha=0.9)
        nanm = ~np.isfinite(F2)
        if nanm.any():
            ax.contourf(g, g, nanm.T.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"], alpha=0)
        if col == 0:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[0, 20, 40, 60, 80])
            cb.set_label("reference F (kJ/mol)", fontsize=9); cb.ax.tick_params(labelsize=8)
        for k in (0, 1, 2, -1):
            sel = b == k
            if sel.any():
                ax.scatter(phi[sel], psi[sel], s=4, c=BASIN_C[k], alpha=0.65, linewidths=0, rasterized=True)
        if key == "fr":
            for ev, c, marker, sz in ((r["die"], C_DIE, "x", 34), (r["birth"], C_BIRTH, "o", 40)):
                sel = (ev[:, 0] > i - win) & (ev[:, 0] <= i)
                if sel.any():
                    age = (i - ev[sel, 0]) / max(win, 1)
                    rgba = np.tile(np.asarray(to_rgba(c)), (int(sel.sum()), 1)); rgba[:, 3] = np.clip(1.0 - 0.75 * age, 0.2, 1.0)
                    x = np.degrees(ev[sel, 1]); y = np.degrees(ev[sel, 2])
                    if case == "psi":
                        x, y = y, x
                    if marker == "o":
                        ax.scatter(x, y, s=sz, marker="o", facecolors="none", edgecolors=rgba, linewidths=1.3)
                    else:
                        ax.scatter(x, y, s=sz, marker="x", c=rgba, linewidths=1.3)
            n_die = int((r["die"][:, 0] <= i).sum()); n_b = int((r["birth"][:, 0] <= i).sum())
            ax.text(0.985, 0.975, f"deaths ×  {n_die:,}     births ○  {n_b:,}", transform=ax.transAxes, fontsize=10.5, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        for k, nm, (cx, cy) in zip(range(3), d["names"], BASIN_XY):
            ax.text(cx, min(cy + 24, 150), nm, ha="center", va="bottom", fontsize=10.5, color=C_INK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        occ = " ".join(f"{nm} {100 * float((b == k).mean()):4.1f} %" for k, nm in enumerate(d["names"]))
        ax.text(0.985, 0.03, occ, transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        ax.set_title(label, color=colour, fontsize=15, fontweight="bold", pad=6)
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-180, 181, 60))
        ax.set_xlabel("φ (deg)"); ax.set_ylabel("ψ (deg)")
        # ---- learned free energy ----------------------------------------------------------
        ax = fig.add_subplot(gs[1, col])
        if case == "ala2d":
            P = r["pmf"][i].astype(float); w = build_masks(F2, d["kT"])["weights"]["equilibrium"]
            ok = (w > 0) & np.isfinite(F2); c = ((P - (F2 - np.nanmin(F2[np.isfinite(F2)])))[ok] * w[ok]).sum() / w[ok].sum()
            im2 = ax.imshow(np.clip((P - c).T, 0, F_MAX), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=F_MAX, aspect="equal", interpolation="bilinear")
            ax.contour(g, g, Fbg.T, levels=[5, 10, 20, 40, 60, 80], colors="white", linewidths=0.5, alpha=0.8)
            if col == 0:
                cb = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02, ticks=[0, 20, 40, 60, 80])
                cb.set_label("learned F (kJ/mol)", fontsize=9); cb.ax.tick_params(labelsize=8)
            ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-180, 181, 60))
            ax.set_xlabel("φ (deg)"); ax.set_ylabel("ψ (deg)")
            ax.set_title(f"learned F (0–90 kJ/mol) + reference contours;  error {r['err'][i]:.2f} kJ/mol", fontsize=10.5)
        else:
            Pv = r["pmf"][i].astype(float); w1 = d["w1"]; ok = w1 > 0
            c = ((Pv - d["F1_fin"])[ok] * w1[ok]).sum() / w1[ok].sum()
            ax.plot(g, d["F1"] - np.nanmin(d["F1"][np.isfinite(d["F1"])]), "k-", lw=1.6, label="reference (full marginal)")
            if d.get("F1_vis") is not None:
                Fv = d["F1_vis"]; ax.plot(g, Fv - np.nanmin(Fv[np.isfinite(Fv)]), "k--", lw=1.0, label="reference, φ<0 side only")
            ax.plot(g, Pv - c, color=colour, lw=1.6, label="learned F")
            cvdeg = phi if case == "phi" else psi
            h, e = np.histogram(cvdeg, bins=48, range=(-180, 180)); ax2 = ax.twinx()
            ax2.stairs(h / (N / 48), e, fill=True, color=colour, alpha=0.18, lw=0); ax2.set_ylim(0, 8); ax2.set_yticks([])
            ymax = float(np.nanmax(d["F1"][np.isfinite(d["F1"])] - np.nanmin(d["F1"][np.isfinite(d["F1"])]))) + 6
            ax.set_ylim(-2, ymax); ax.set_xlim(-180, 180); ax.set_xticks(range(-180, 181, 60))
            ax.set_xlabel(("φ" if case == "phi" else "ψ") + " (deg)"); ax.set_ylabel("F (kJ/mol)")
            ax.legend(fontsize=8.5, frameon=False, loc="upper right")
            extra = f" (vs φ<0 side {r['err_vis'][i]:.2f})" if case == "psi" else ""
            ax.set_title(f"learned F(" + ("φ" if case == "phi" else "ψ") + f") + walker histogram;  error {r['err'][i]:.2f}{extra} kJ/mol", fontsize=10.5)
    # ---- right column: error, distance to the uniform target, occupancies ---------------------
    ax = fig.add_subplot(gsr[0])
    for key, colour, label in (("abf", C_ABF, "ABF"), ("fr", C_FR, "ABF + FR")):
        r = d[key]; ax.plot(r["t"], r["err"], color=colour, lw=1.4, label=label)
        ax.plot([t], [r["err"][i]], "o", color=colour, ms=6)
    if case == "psi":
        ax.plot(d["abf"]["t"], d["abf"]["err_vis"], color=C_ABF, lw=0.9, ls="--", label="ABF vs φ<0-side reference")
    ax.set_yscale("log"); ax.set_xlim(0, T); ax.set_xlabel("t (ps)"); ax.set_ylabel("FES error (kJ/mol)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right"); ax.set_title("free-energy error vs the reference", fontsize=10.5)
    ax = fig.add_subplot(gsr[1])
    for key, colour, label in (("abf", C_ABF, "ABF"), ("fr", C_FR, "ABF + FR")):
        r = d[key]; ax.plot(r["save_t"], r["kl"], color=colour, lw=1.4, label=label)
    ax.axvline(t, color=C_INK2, lw=0.8); ax.set_xlim(0, T); ax.set_xlabel("t (ps)"); ax.set_ylabel("KL(p‖uniform)")
    ax.set_title("walker marginal's distance to the FR target", fontsize=10.5); ax.legend(fontsize=8.5, frameon=False)
    ax = fig.add_subplot(gsr[2])
    for key, ls, label in (("abf", "-", "ABF"), ("fr", "--", "FR")):
        r = d[key]
        for k, nm in enumerate(d["names"]):
            ax.plot(r["save_t"], r["basin_frac"][:, k], color=BASIN_C[k], lw=1.3, ls=ls, label=f"{nm} ({label})" if key == "abf" else None)
    for k in range(3):
        ax.axhline(d["pops"][k], color=BASIN_C[k], lw=0.7, ls=":")
    ax.axvline(t, color=C_INK2, lw=0.8)
    ax.set_xlim(0, T); ax.set_ylim(0, 1); ax.set_xlabel("t (ps)"); ax.set_ylabel("fraction of walkers")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
    ax.set_title("basin occupancy: solid ABF, dashed FR, dotted reference", fontsize=10.5)
    fig.savefig(path, dpi=120); plt.close(fig)
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=("ala2d", "phi", "psi"))
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=48); ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--event-window", type=int, default=6)
    a = ap.parse_args()
    d = _load(a.case); G["d"] = d; G["event_window"] = a.event_window
    out = os.path.join(ROOT, "results/alanine_movie", a.case); os.makedirs(out, exist_ok=True)
    name = f"alanine_{a.case}_abf_vs_fr"
    render_movie(draw_frame, len(d["abf"]["t"]), stride=a.stride, frames_dir=os.path.join(out, "frames"), workers=a.workers,
                 fps=a.fps, mp4=os.path.join(out, name + ".mp4"), only=a.only, hold_seconds=2.0, root=ROOT)
    if a.only is None:
        s = dict(case=a.case, seed=int(d["abf"]["meta"]["seeds"][0]), N=d["N"], T=d["T"], rate=d["fr"]["rate"],
                 err_final=dict(abf=float(d["abf"]["err"][-1]), fr=float(d["fr"]["err"][-1])),
                 err_integrated=dict(abf=float(np.trapezoid(d["abf"]["err"], d["abf"]["t"])), fr=float(np.trapezoid(d["fr"]["err"], d["fr"]["t"]))),
                 events=dict(deaths=int(len(d["fr"]["die"])), births=int(len(d["fr"]["birth"]))),
                 c7ax_final=dict(abf=float(d["abf"]["basin_frac"][-1, 2]), fr=float(d["fr"]["basin_frac"][-1, 2])))
        json.dump(s, open(os.path.join(out, "movie_summary.json"), "w"), indent=2); print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
