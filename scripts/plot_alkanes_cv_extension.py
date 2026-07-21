#!/usr/bin/env python3
"""Manuscript figures for the CV-extension study, generated from artifacts (no GPU).

Each figure is guarded by data availability so partial stages still plot. Reads raw
``*.npz`` under the given output roots + the validation/reference JSONs. Writes PNGs to
``--figdir`` (default results/alkanes_cv_extension/figures) and optionally copies the
report set into ``--report-figdir``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PI = np.pi
GAUCHE = np.radians(116.57)


def _load_runs(raw_dir, kind=None):
    runs = []
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        try:
            d = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if "per_seed" not in d.files:
            continue
        if kind and str(d["kind"]) != kind:
            continue
        runs.append(d)
    return runs


def fig_r15_reference_and_screen(r15_raw, figdir):
    runs = _load_runs(r15_raw, "dist")
    abf = [d for d in runs if str(d["method"]) == "abf" and str(d["init_mode"]) == "trans"]
    if not abf:
        return None
    # panels (b)/(c) show the screen cells only: the resolution-gate repeats are the same
    # physical cell at other grids/bandwidths and would plot as duplicate-labelled curves.
    screen = [d for d in runs if str(d["method"]) == "abf" and str(d["stage"]) == "screen"] or abf

    def _cell_lbl(d):
        init = "trans" if str(d["init_mode"]) == "trans" else "disp"
        return f"$\\beta={float(d['beta']):g}$ {init}"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # (a) reference F(R15) at available betas
    seen = set()
    for d in abf:
        b = float(d["beta"])
        if b in seen:
            continue
        seen.add(b)
        g = d["grid"]; F = d["ref_F"]
        ax[0].plot(g, F - F.min(), label=f"$\\beta={b:g}$")
    ax[0].set_xlabel("$R_{15}$"); ax[0].set_ylabel("$F_\\mathrm{ref}(R_{15})-F_\\min$ [kT]")
    ax[0].set_title("(a) R15 reference free energy"); ax[0].set_ylim(0, 15); ax[0].legend()
    # (b) ABF convergence: L2(F) vs time per beta
    for d in sorted(screen, key=lambda d: (float(d["beta"]), str(d["init_mode"]))):
        if "l2_F_t" in d.files:
            l2 = d["l2_F_t"].mean(0); t = d["times"]
            ax[1].plot(t, l2, label=_cell_lbl(d))
    ax[1].set_xlabel("time"); ax[1].set_ylabel("thermal-window $L_2(F)$ [kT]")
    ax[1].set_title("(b) R15 ABF convergence"); ax[1].legend()
    # (c) final biased marginal p(R15) vs reference Boltzmann: one trans cell per beta
    seen_c = set()
    for d in sorted(screen, key=lambda d: float(d["beta"])):
        b = float(d["beta"])
        if b in seen_c or str(d["init_mode"]) != "trans":
            continue
        seen_c.add(b)
        g = d["grid"]
        line, = ax[2].plot(g, d["final_p_hat"].mean(0), label=f"ABF $p(R)$ $\\beta={b:g}$")
        pb = np.exp(-b * (d["ref_F"] - d["ref_F"].min())); pb /= np.trapezoid(pb, g)
        ax[2].plot(g, pb, "--", color=line.get_color(), label=f"Boltzmann $\\beta={b:g}$")
    ax[2].set_xlabel("$R_{15}$"); ax[2].set_ylabel("density"); ax[2].set_title("(c) R15 marginal"); ax[2].legend()
    fig.tight_layout(); out = os.path.join(figdir, "fig_cv_01_r15_reference_screen.png")
    fig.savefig(out, dpi=130); plt.close(fig); return out


def fig_2d_reference_and_gate(two_d_raw, gate_json, figdir):
    outs = []
    runs = _load_runs(two_d_raw, "joint2d")
    ref = next((d for d in runs if "ref_joint_F" in d.files), None)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    if ref is not None:
        F = ref["ref_joint_F"]; g = ref["grid1"]
        im = ax[0].pcolormesh(g, g, (F - F.min()).T, cmap="viridis", vmax=12, shading="auto")
        ax[0].set_title("(a) 2-D reference $F(\\varphi_1,\\varphi_2)$")
        ax[0].set_xlabel("$\\varphi_1$"); ax[0].set_ylabel("$\\varphi_2$"); fig.colorbar(im, ax=ax[0])
        # a representative ABF reconstruction
        abf = next((d for d in runs if str(d["method"]) == "abf"), None)
        if abf is not None:
            B = abf["final_pmf"].mean(0); B = B - B.min()
            im2 = ax[1].pcolormesh(g, g, B.T, cmap="viridis", vmax=12, shading="auto")
            ax[1].set_title(f"(b) 2-D ABF $\\hat F$ ({str(abf['init_mode'])})")
            ax[1].set_xlabel("$\\varphi_1$"); ax[1].set_ylabel("$\\varphi_2$"); fig.colorbar(im2, ax=ax[1])
    # (c) decoupled gate: L2 vs grid
    if gate_json and os.path.exists(gate_json):
        d = json.load(open(gate_json))
        ng = [g["n_grid"] for g in d["gate"]]; l2 = [g["l2_thermal_median"] for g in d["gate"]]
        ax[2].plot(ng, l2, "o-")
        ax[2].set_xlabel("grid ($n\\times n$)"); ax[2].set_ylabel("decoupled-gate thermal $L_2$ [kT]")
        ax[2].set_title("(c) exact decoupled 2-D gate")
        ax[2].set_ylim(0, max(l2) * 1.35 if l2 else 1.0)
    fig.tight_layout(); out = os.path.join(figdir, "fig_cv_02_2d_reference_gate.png")
    fig.savefig(out, dpi=130); plt.close(fig); outs.append(out); return out


def fig_2d_screen_basins(two_d_raw, figdir):
    runs = [d for d in _load_runs(two_d_raw, "joint2d") if str(d["method"]) == "abf"]
    if not runs:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    labels = []
    for d in runs:
        cell = f"b{float(d['beta']):g} {str(d['init_mode'])[:4]}"
        jh = d["joint_hist"].sum(0); jh = jh / max(jh.sum(), 1)
        # basin occupancy 3x3
        g = d["grid1"]; T = np.abs(g) < np.radians(61.6); Gp = g >= np.radians(61.6); Gm = g <= -np.radians(61.6)
        masks = [T, Gp, Gm]
        occ = np.array([jh[np.ix_(m1, m2)].sum() for m1 in masks for m2 in masks])
        ax[0].plot(range(9), occ, "o-", label=cell); labels.append(cell)
    ax[0].set_xticks(range(9)); ax[0].set_xticklabels(["TT", "TG+", "TG-", "G+T", "G+G+", "G+G-", "G-T", "G-G+", "G-G-"], rotation=45)
    ax[0].set_ylabel("occupancy"); ax[0].set_title("(a) 2-D ABF joint basin occupancy"); ax[0].legend(fontsize=8)
    # (b) L2 convergence per cell
    for d in runs:
        if "l2_F_t" in d.files:
            ax[1].plot(d["times"], d["l2_F_t"].mean(0), label=f"b{float(d['beta']):g} {str(d['init_mode'])[:4]}")
    ax[1].set_xlabel("time"); ax[1].set_ylabel("thermal $L_2(F)$ [kT]"); ax[1].set_title("(b) 2-D ABF convergence"); ax[1].legend(fontsize=8)
    fig.tight_layout(); out = os.path.join(figdir, "fig_cv_03_2d_screen_basins.png")
    fig.savefig(out, dpi=130); plt.close(fig); return out


def fig_methods_comparison(two_d_raw, summaries_dir, figdir):
    """mFR/OPES vs ABF: matched-seed paired deltas + ancestor ESS (if a methods stage ran)."""
    pcsv = os.path.join(summaries_dir, "cv_paired.csv")
    if not os.path.exists(pcsv):
        return None
    import pandas as pd
    pf = pd.read_csv(pcsv)
    pf = pf[pf.metric == "final_l2_F"]
    if pf.empty:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for cell, sub in pf.groupby("cell"):
        ax[0].errorbar(sub.method, sub.rel_med * 100,
                       yerr=[(sub.rel_med - sub.rel_lo) * 100, (sub.rel_hi - sub.rel_med) * 100],
                       fmt="o", capsize=3, label=cell)
    ax[0].axhline(0, color="k", lw=0.6); ax[0].axhline(-15, color="g", ls="--", lw=0.8, label="-15% (positive)")
    ax[0].axhspan(-10, 10, color="gray", alpha=0.15)
    ax[0].set_ylabel("rel. change in final $L_2(F)$ vs ABF [%]")
    ax[0].set_title("(a) matched-seed mFR/OPES vs ABF"); ax[0].tick_params(axis="x", rotation=45); ax[0].legend(fontsize=7)
    # final ancestor ESS/N per FR method (per-seed finals; the time series is not stored)
    names, vals = [], []
    for d in _load_runs(two_d_raw, "joint2d"):
        if "fr" not in str(d["name"]):
            continue
        ps = json.loads(str(d["per_seed"]))
        e = np.nanmedian([s.get("final_ancestor_ess", np.nan) for s in ps]) / int(d["n_replicas"])
        if np.isfinite(e):
            names.append(f"{str(d['name'])}\nb{float(d['beta']):g}"); vals.append(e)
    if names:
        order = np.argsort(vals)[::-1]
        ax[1].bar([names[i] for i in order], [vals[i] for i in order], color="steelblue")
    ax[1].axhline(0.30, color="r", ls="--", lw=0.9, label="0.30 N genealogy floor")
    ax[1].set_ylabel("final ancestor ESS / N"); ax[1].set_title("(b) genealogy (mFR)")
    ax[1].tick_params(axis="x", rotation=45, labelsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); out = os.path.join(figdir, "fig_cv_04_methods_comparison.png")
    fig.savefig(out, dpi=130); plt.close(fig); return out


def fig_birth_death_map(two_d_raw, figdir):
    runs = [d for d in _load_runs(two_d_raw, "joint2d")
            if str(d["method"]) in ("fr_estimated", "fr_uniform") and d["birth_hist"].sum() > 0]
    if not runs:
        return None
    d = max(runs, key=lambda x: x["birth_hist"].sum())
    g = d["grid1"]; bh = d["birth_hist"].sum(0); dh = d["death_hist"].sum(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, (h, ttl) in zip(ax, [(bh, "births (clone sources)"), (dh, "deaths")]):
        im = a.pcolormesh(g, g, h.T, cmap="magma", shading="auto")
        a.set_title(ttl); a.set_xlabel("$\\varphi_1$"); a.set_ylabel("$\\varphi_2$"); fig.colorbar(im, ax=a)
    fig.suptitle(f"FR birth/death spatial map ({str(d['name'])}, b{float(d['beta']):g})")
    fig.tight_layout(); out = os.path.join(figdir, "fig_cv_05_fr_birth_death.png")
    fig.savefig(out, dpi=130); plt.close(fig); return out


def fig_r15_methods(meth_raw, meth_sum, figdir):
    """R15 mFR/OPES comparison on the starved cell: paired L2 vs ABF, rate ladder, conditional."""
    runs = _load_runs(meth_raw, "dist")
    if not runs:
        return None
    import pandas as pd
    pcsv = os.path.join(meth_sum, "cv_paired.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # (a) matched-seed paired rel change vs ABF (final L2)
    if os.path.exists(pcsv):
        pf = pd.read_csv(pcsv)
        pf = pf[pf.metric == "final_l2_F"]
        for cell, sub in pf.groupby("cell"):
            ax[0].errorbar(sub.method, sub.rel_med * 100,
                           yerr=[(sub.rel_med - sub.rel_lo) * 100, (sub.rel_hi - sub.rel_med) * 100],
                           fmt="o", capsize=3, label=cell.replace("dist_pentane_", ""))
        ax[0].axhline(0, color="k", lw=0.6); ax[0].axhline(-15, color="g", ls="--", lw=0.8)
        ax[0].axhspan(-10, 10, color="gray", alpha=0.15)
        ax[0].set_ylabel("rel. change in final $L_2(F)$ vs ABF [%]")
        ax[0].set_title("(a) R15 mFR/OPES vs ABF (starved)"); ax[0].tick_params(axis="x", rotation=45); ax[0].legend(fontsize=7)
    # (b) rate ladder: integrated L2 + ancestor ESS vs fr_rate (tuning runs)
    lad = {}
    for d in runs:
        if str(d["stage"]) != "tuning":
            continue
        spec = json.loads(str(d["spec_json"])); rate = float(spec["fr_rate"])
        ps = json.loads(str(d["per_seed"]))
        il2 = np.median([s["integrated_l2_F"] for s in ps])
        ess = np.median([s.get("final_ancestor_ess", np.nan) for s in ps])
        lad[rate] = (il2, ess / int(d["n_replicas"]) if np.isfinite(ess) else np.nan)
    if lad:
        rates = sorted(lad); ax2 = ax[1].twinx()
        ax[1].plot(rates, [lad[r][0] for r in rates], "o-b", label="integrated $L_2(F)$")
        ax2.plot(rates, [lad[r][1] for r in rates], "s--r", label="ancestor ESS/N")
        ax[1].set_xscale("log"); ax[1].set_xlabel("mFR rate"); ax[1].set_ylabel("integrated $L_2(F)$", color="b")
        ax2.set_ylabel("ancestor ESS / N", color="r"); ax2.axhline(0.30, color="r", ls=":", lw=0.7)
        ax[1].set_title("(b) R15 mFR rate ladder")
    # (c) conditional torsion fidelity per method (production)
    for d in runs:
        if str(d["stage"]) != "production":
            continue
        ps = json.loads(str(d["per_seed"]))
        ctv = np.median([s.get("dist_cond_tv_weighted", np.nan) for s in ps])
        ax[2].bar(str(d["name"]), ctv)
    ax[2].set_ylabel("cond. TV $p(\\varphi_1,\\varphi_2|R)$"); ax[2].set_title("(c) R15 hidden-conditional fidelity")
    ax[2].tick_params(axis="x", rotation=45)
    fig.tight_layout(); out = os.path.join(figdir, "fig_cv_06_r15_methods.png")
    fig.savefig(out, dpi=130); plt.close(fig); return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/alkanes_cv_extension")
    ap.add_argument("--figdir", default="results/alkanes_cv_extension/figures")
    ap.add_argument("--report-figdir", default=None)
    args = ap.parse_args(argv)
    os.makedirs(args.figdir, exist_ok=True)
    r15_raw = os.path.join(args.root, "r15", "raw")
    two_d_raw = os.path.join(args.root, "2d", "raw")           # ABF-only screen
    meth_raw = os.path.join(args.root, "2d_methods", "raw")    # 2-D production methods
    meth_sum = os.path.join(args.root, "2d_methods", "summaries")
    r15m_raw = os.path.join(args.root, "r15_methods", "raw")   # R15 production methods
    r15m_sum = os.path.join(args.root, "r15_methods", "summaries")
    gate_json = os.path.join(args.root, "validation", "decoupled_2d_gate.json")
    made = []
    for fn in [lambda: fig_r15_reference_and_screen(r15_raw, args.figdir),
               lambda: fig_2d_reference_and_gate(two_d_raw, gate_json, args.figdir),
               lambda: fig_2d_screen_basins(two_d_raw, args.figdir),
               lambda: fig_methods_comparison(meth_raw, meth_sum, args.figdir),
               lambda: fig_birth_death_map(meth_raw, args.figdir),
               lambda: fig_r15_methods(r15m_raw, r15m_sum, args.figdir)]:
        try:
            r = fn()
            if r:
                made.append(r); print("[plot]", os.path.relpath(r))
        except Exception as e:
            print("[plot] skipped:", repr(e))
    if args.report_figdir and made:
        os.makedirs(args.report_figdir, exist_ok=True)
        for m in made:
            shutil.copy(m, os.path.join(args.report_figdir, os.path.basename(m)))
        print(f"[plot] copied {len(made)} figures to {args.report_figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
