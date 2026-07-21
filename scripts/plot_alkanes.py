#!/usr/bin/env python3
"""Manuscript figures for the alkane study, generated from artifacts (no hand values).

Figures (written to <output_root>/figures_<stage>/ and copied to report/figures/):
  fig_alk_01_model            butane/pentane schematics + dihedral definitions
  fig_alk_02_rb_torsion       RB torsion V4 with T/G minima and barriers
  fig_alk_03_butane_reference full butane reference F and F'
  fig_alk_04_pentane_joint    pentane joint reference F(phi1,phi2)
  fig_alk_05_pentane_marginal pentane 1-D reference F(phi1) vs V4
  fig_alk_06_butane_convergence   convergence + final profiles (ABF/mFR/OPES)
  fig_alk_07_butane_equivalence   equivalence/gain + event/genealogy diagnostics
  fig_alk_08_pentane_convergence  pentane convergence + final profiles
  fig_alk_09_pentane_conditional  p(phi2|phi1) fidelity per method
  fig_alk_10_pentane_basins       basin occupancy / transitions
  fig_alk_11_fr_rate_ladder       FR rate vs accuracy + ancestor ESS
  fig_alk_12_cross_case           butane+pentane on the starvation regime map
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch  # noqa: E402
torch.set_default_dtype(torch.float64)
from alkanes import jobs as J, potentials as pot, periodic as per, geometry as geom  # noqa: E402

GAUCHE = math.radians(116.57)
METHOD_COLORS = {"abf": "#444444", "fr_estimated": "#d62728", "fr_uniform": "#1f77b4",
                 "fr_oracle": "#2ca02c", "opes": "#9467bd", "fr_active": "#ff7f0e"}
METHOD_LABEL = {"abf": "ABF", "fr_estimated": "ABF+mFR (est.)", "fr_uniform": "ABF+mFR (uniform)",
                "fr_oracle": "ABF+mFR (oracle)", "opes": "OPES", "fr_active": "ABF+mFR (active)"}


def _load_runs(raw_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        try:
            d = np.load(p, allow_pickle=True)
            if "per_seed" in d.files:
                runs.append(d)
        except Exception:
            pass
    return runs


def fig_model(figdir):
    fig = plt.figure(figsize=(9, 3.6))
    for k, (n_atoms, title) in enumerate([(4, "butane: $\\varphi_1(q_1q_2q_3q_4)$"),
                                          (5, "pentane: $\\varphi_1,\\varphi_2$")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        q = geom.place_chain(torch.tensor([[0.0]] if n_atoms == 4 else [[GAUCHE, -GAUCHE]]),
                             n_atoms=n_atoms)[0].numpy()
        ax.plot(q[:, 0], q[:, 1], q[:, 2], "-o", color="#333", ms=9, lw=2)
        for i, (x, y, z) in enumerate(q):
            ax.text(x, y, z + 0.1, f"$q_{i+1}$", fontsize=9)
        ax.set_title(title); ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_alk_01_model.png"), dpi=140)
    plt.close(fig)


def fig_rb_torsion(figdir):
    p = pot.AlkaneParams(n_atoms=4)
    phi = torch.linspace(-math.pi, math.pi, 1000)
    V = pot.V4(phi, p).numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.degrees(phi), V, "k-", lw=2)
    for a, name in [(0, "T"), (116.57, "G$^+$"), (-116.57, "G$^-$")]:
        ax.axvline(a, color="green", ls=":", alpha=.5)
        ax.text(a, -0.6, name, ha="center", color="green")
    for a in (61.6, -61.6, 180, -180):
        ax.axvline(a, color="red", ls=":", alpha=.3)
    ax.set_xlabel("$\\varphi$ (deg)"); ax.set_ylabel("$V_4(\\varphi)$  ($k_BT$ at $\\beta{=}1$)")
    ax.set_title("Ryckaert--Bellemans torsion")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_alk_02_rb_torsion.png"), dpi=140)
    plt.close(fig)


def fig_butane_reference(figdir):
    p = pot.AlkaneParams(n_atoms=4, beta=1.0)
    grid, dphi = per.periodic_grid(180)
    F = (pot.V4(grid, p) - pot.V4(grid, p).mean()).numpy()
    Fp = pot.V4_prime(grid, p).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(np.degrees(grid), F, "k-"); axes[0].set_title("butane $F(\\varphi_1)$")
    axes[1].plot(np.degrees(grid), Fp, "k-"); axes[1].set_title("butane $F'(\\varphi_1)$")
    for ax in axes:
        ax.set_xlabel("$\\varphi_1$ (deg)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_alk_03_butane_reference.png"), dpi=140)
    plt.close(fig)


def fig_pentane_reference(figdir, runs):
    # pull a pentane run to get the cached joint reference (any full pentane run)
    pen = [d for d in runs if str(d["molecule"]) == "pentane" and "ref_joint_F" in d.files]
    if not pen:
        return
    d = pen[0]
    J2 = d["ref_joint_F"]; g2 = np.degrees(d["grid2"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    Jc = np.clip(J2, None, np.percentile(J2, 98))
    im = ax.pcolormesh(g2, g2, Jc.T, shading="auto", cmap="viridis")
    ax.set_xlabel("$\\varphi_1$ (deg)"); ax.set_ylabel("$\\varphi_2$ (deg)")
    ax.set_title("pentane joint reference $F(\\varphi_1,\\varphi_2)$")
    fig.colorbar(im, ax=ax, label="$F$ ($k_BT$)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_alk_04_pentane_joint.png"), dpi=140)
    plt.close(fig)
    # 1D marginal vs V4
    grid = d["grid"]; F1 = d["ref_F"]
    p = pot.AlkaneParams(n_atoms=5, beta=float(d["beta"]))
    V4 = (pot.V4(torch.tensor(grid), p) - pot.V4(torch.tensor(grid), p).mean()).numpy()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(np.degrees(grid), F1, "b-", label="$F_{ref}(\\varphi_1)$ (marginalised)")
    ax.plot(np.degrees(grid), V4, "k--", label="$V_4(\\varphi_1)$ only")
    ax.legend(); ax.set_xlabel("$\\varphi_1$ (deg)"); ax.set_ylabel("$F$ ($k_BT$)")
    ax.set_title("pentane 1-D reference: coupling barely shifts the marginal")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_alk_05_pentane_marginal.png"), dpi=140)
    plt.close(fig)


def fig_convergence(figdir, runs, molecule, fname):
    sub = [d for d in runs if str(d["molecule"]) == molecule]
    if not sub:
        return
    # pick the primary cell (localized init, smallest beta present == 1 if available)
    cells = sorted({(float(d["beta"]), str(d["init_mode"])) for d in sub})
    beta, init = cells[0]
    cell = [d for d in sub if float(d["beta"]) == beta and str(d["init_mode"]) == init]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    grid = None
    for d in cell:
        m = str(d["name"])
        times = d["times"]; l2 = np.median(d["l2_F_t"], axis=0)
        axes[0].plot(times, l2, color=METHOD_COLORS.get(m, "k"), label=METHOD_LABEL.get(m, m))
        grid = d["grid"]
        Fmed = np.median(d["final_pmf"], axis=0)
        axes[1].plot(np.degrees(grid), Fmed - Fmed.mean(), color=METHOD_COLORS.get(m, "k"))
    if cell:
        axes[1].plot(np.degrees(cell[0]["grid"]), cell[0]["ref_F"], "k:", lw=2, label="reference")
    axes[0].set_xlabel("time"); axes[0].set_ylabel("median $L_2(F)$"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"{molecule} $\\beta$={beta:g} convergence")
    axes[1].set_xlabel("$\\varphi_1$ (deg)"); axes[1].set_ylabel("$F$"); axes[1].set_title("final profiles")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, fname), dpi=140); plt.close(fig)


def fig_equivalence(figdir, summ_dir):
    eq = os.path.join(summ_dir, "alkanes_equivalence.csv")
    pr = os.path.join(summ_dir, "alkanes_paired.csv")
    if not os.path.exists(pr):
        return
    import pandas as pd
    p = pd.read_csv(pr)
    p = p[p.metric == "final_l2_F"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ys = []
    for i, (_, r) in enumerate(p.iterrows()):
        lab = f"{r['cell']}\n{r['method']}"
        ax.errorbar(r["rel_med"], i, xerr=[[r["rel_med"] - r["rel_lo"]], [r["rel_hi"] - r["rel_med"]]],
                    fmt="o", color="#d62728")
        ys.append(lab)
    ax.axvline(0, color="k", lw=.8); ax.axvspan(-0.1, 0.1, color="green", alpha=.1)
    ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys, fontsize=7)
    ax.set_xlabel("relative change in final $L_2(F)$ vs ABF (matched seeds)")
    ax.set_title("equivalence: shaded = $\\pm$10% practical margin")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_alk_07_butane_equivalence.png"), dpi=140)
    plt.close(fig)


def fig_pentane_conditional(figdir, runs):
    sub = [d for d in runs if str(d["molecule"]) == "pentane" and "cond_tv_weighted" not in d.files]
    # use main summary instead; here show per-method weighted conditional TV bars
    import pandas as pd
    # gather from per_seed
    rows = []
    for d in runs:
        if str(d["molecule"]) != "pentane":
            continue
        for rec in json.loads(str(d["per_seed"])):
            if "cond_tv_weighted" in rec:
                rows.append(dict(method=str(d["name"]), beta=float(d["beta"]),
                                 cond_tv=rec["cond_tv_weighted"], l2F=rec["final_l2_F"]))
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    order = ["abf", "fr_estimated", "fr_active", "fr_uniform", "fr_oracle", "opes"]
    g = df.groupby("method")
    for ax, col, ttl in [(axes[0], "l2F", "final $L_2(F(\\varphi_1))$"),
                         (axes[1], "cond_tv", "weighted conditional TV $p(\\varphi_2|\\varphi_1)$")]:
        meds = [g.get_group(m)[col].median() if m in g.groups else np.nan for m in order]
        ax.bar(range(len(order)), meds, color=[METHOD_COLORS[m] for m in order])
        ax.set_xticks(range(len(order))); ax.set_xticklabels([METHOD_LABEL[m] for m in order], rotation=30, fontsize=7)
        ax.set_title(ttl)
    fig.suptitle("pentane: marginal accuracy vs hidden-conditional fidelity")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_alk_09_pentane_conditional.png"), dpi=140)
    plt.close(fig)


def fig_pentane_basins(figdir, runs):
    import pandas as pd
    rows = []
    for d in runs:
        if str(d["molecule"]) != "pentane":
            continue
        for rec in json.loads(str(d["per_seed"])):
            r = dict(method=str(d["name"]), beta=float(d["beta"]))
            r.update({k: rec.get(k) for k in rec if k.startswith("basin_") or k in
                      ("n_basins_visited", "n_transitions", "n_round_trips")})
            rows.append(r)
    if not rows:
        return
    df = pd.DataFrame(rows)
    order = ["abf", "fr_estimated", "fr_active", "fr_uniform", "fr_oracle", "opes"]
    basin_cols = [c for c in df.columns if c.startswith("basin_")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    g = df.groupby("method")
    if basin_cols:
        M = np.array([[g.get_group(m)[c].median() if m in g.groups else 0 for c in basin_cols] for m in order])
        im = axes[0].imshow(M, aspect="auto", cmap="magma")
        axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([METHOD_LABEL[m] for m in order], fontsize=7)
        axes[0].set_xticks(range(len(basin_cols)))
        axes[0].set_xticklabels([c.replace("basin_", "") for c in basin_cols], rotation=90, fontsize=6)
        axes[0].set_title("median basin occupancy (9 torsional basins)")
        fig.colorbar(im, ax=axes[0])
    for col, ttl in [("n_transitions", "transitions"), ("n_round_trips", "round trips")]:
        if col in df.columns:
            meds = [g.get_group(m)[col].median() if m in g.groups else np.nan for m in order]
            axes[1].bar(np.arange(len(order)) + (0.2 if col == "n_round_trips" else -0.2), meds,
                        width=0.35, label=ttl)
    axes[1].set_xticks(range(len(order))); axes[1].set_xticklabels([METHOD_LABEL[m] for m in order], rotation=30, fontsize=7)
    axes[1].legend(); axes[1].set_title("exploration")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_alk_10_pentane_basins.png"), dpi=140)
    plt.close(fig)


def fig_rate_ladder(figdir, tuning_root):
    import pandas as pd
    csv = os.path.join(tuning_root, "summaries", "alkanes_config_summary.csv")
    if not os.path.exists(csv):
        return
    df = pd.read_csv(csv)
    fr = df[df.name.str.startswith("fr_r")].copy()
    if fr.empty:
        return
    fr["rate"] = fr.name.str.replace("fr_r", "").astype(float) / 100.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mol, sub in fr.groupby("molecule"):
        sub = sub.sort_values("rate")
        axes[0].plot(sub.rate, sub.integrated_l2_F_med, "-o", label=mol)
        if "final_ancestor_ess_med" in sub:
            axes[1].plot(sub.rate, sub.final_ancestor_ess_med, "-o", label=mol)
    # abf baseline
    for mol, sub in df[df.name == "abf"].groupby("molecule"):
        axes[0].axhline(sub.integrated_l2_F_med.iloc[0], ls="--", alpha=.4)
    axes[0].set_xscale("log"); axes[0].set_xlabel("FR rate"); axes[0].set_ylabel("median int. $L_2(F)$")
    axes[0].legend(); axes[0].set_title("accuracy vs FR rate")
    axes[1].set_xscale("log"); axes[1].set_xlabel("FR rate"); axes[1].set_ylabel("ancestor ESS")
    axes[1].legend(); axes[1].set_title("diversity vs FR rate")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_alk_11_fr_rate_ladder.png"), dpi=140)
    plt.close(fig)


def fig_cross_case(figdir, summ_dir):
    """mFR gain (relative improvement vs ABF) vs measured ABF baseline error, per cell.

    Overlays butane and pentane cells; if the WCA starvation scatter is available it is
    drawn as grey background context.
    """
    import pandas as pd
    pr = os.path.join(summ_dir, "alkanes_paired.csv")
    cs = os.path.join(summ_dir, "alkanes_config_summary.csv")
    if not (os.path.exists(pr) and os.path.exists(cs)):
        return
    paired = pd.read_csv(pr); conf = pd.read_csv(cs)
    p = paired[(paired.metric == "final_l2_F") & (paired.method == "fr_estimated")]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    # WCA background (optional)
    wca = "results/wca_phase_diagram/production/starvation/starvation_gain_vs_error.csv"
    if os.path.exists(wca):
        try:
            w = pd.read_csv(wca)
            xc = [c for c in w.columns if "abf" in c.lower() and ("l2" in c.lower() or "err" in c.lower())]
            yc = [c for c in w.columns if "gain" in c.lower()]
            if xc and yc:
                ax.scatter(w[xc[0]], w[yc[0]], c="#cccccc", s=25, label="WCA cells", zorder=1)
        except Exception:
            pass
    for _, r in p.iterrows():
        gain = -r["rel_med"] * 100.0                      # positive = mFR better
        mol = "butane" if "butane" in r["cell"] else "pentane"
        col = "#d62728" if mol == "pentane" else "#1f77b4"
        ax.errorbar(r["abf_med"], gain, yerr=[[max(0, (r["rel_hi"] - r["rel_med"]) * 100)],
                    [max(0, (r["rel_med"] - r["rel_lo"]) * 100)]], fmt="o", color=col, zorder=3)
        ax.annotate(r["cell"].replace("_s2.3", "").replace("_full", ""), (r["abf_med"], gain),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")
    # Collective-variable extension overlay: the harder coordinates (R15, R14, 2-D torus).
    # These carry the decisive point -- a genuinely starved cell with NO mFR gain -- so they
    # are drawn with distinct markers rather than folded into the phi1 series.
    cv_specs = [("results/alkanes_cv_extension/r15_methods/summaries/cv_paired.csv",
                 "production", "*", 200, "#7b1fa2", "pentane $R_{15}$ (starved)"),
                ("results/alkanes_cv_extension/2d_methods/summaries/cv_paired.csv",
                 "production", "^", 70, "#00897b", "pentane $(\\varphi_1,\\varphi_2)$ 2-D")]
    for path, stage, marker, size, col, lab in cv_specs:
        if not os.path.exists(path):
            continue
        try:
            c = pd.read_csv(path)
            c = c[(c.metric == "final_l2_F") & (c.method == "fr_estimated")
                  & c.cell.astype(str).str.startswith(stage)]
            for _, r in c.iterrows():
                gain = -r["rel_med"] * 100.0
                ax.errorbar(r["abf_med"], gain,
                            yerr=[[max(0, (r["rel_hi"] - r["rel_med"]) * 100)],
                                  [max(0, (r["rel_med"] - r["rel_lo"]) * 100)]],
                            fmt=marker, ms=(12 if marker == "*" else 7), color=col,
                            zorder=4, label=lab)
                lab = None  # legend entry once per series
        except Exception:
            pass
    ax.axhline(0, color="k", lw=.7)
    ax.set_xlabel("measured ABF baseline error  (median final $L_2(F)$)")
    ax.set_ylabel("mFR gain vs ABF  (\\% reduction in $L_2(F)$)")
    ax.set_title("Cross-case: mFR gain vs ABF starvation")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_alk_12_cross_case.png"), dpi=140)
    plt.close(fig)


def _placeholder(figdir, name, msg):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, color="#888")
    ax.set_axis_off()
    fig.savefig(os.path.join(figdir, name), dpi=120); plt.close(fig)


def ensure_all(figdir):
    """Guarantee every referenced figure PNG exists (placeholder if not yet generated)."""
    needed = {"fig_alk_01_model": "molecular schematics",
              "fig_alk_02_rb_torsion": "RB torsion", "fig_alk_03_butane_reference": "butane reference",
              "fig_alk_04_pentane_joint": "pentane joint reference",
              "fig_alk_05_pentane_marginal": "pentane 1D reference",
              "fig_alk_06_butane_convergence": "butane convergence",
              "fig_alk_07_butane_equivalence": "butane equivalence",
              "fig_alk_08_pentane_convergence": "pentane convergence",
              "fig_alk_09_pentane_conditional": "pentane conditional fidelity",
              "fig_alk_10_pentane_basins": "pentane basins",
              "fig_alk_11_fr_rate_ladder": "FR rate ladder",
              "fig_alk_12_cross_case": "cross-case starvation map"}
    for name, msg in needed.items():
        p = os.path.join(figdir, name + ".png")
        if not os.path.exists(p):
            _placeholder(figdir, name + ".png", f"[{msg}]\n(pending production/tuning)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--report-figdir", default=None)
    ap.add_argument("--tuning-root", default="results/alkanes/tuning")
    args = ap.parse_args(argv)
    cfg = J.load_yaml(args.config)
    root = cfg["output_root"]
    raw_dir = os.path.join(root, "raw")
    summ_dir = os.path.join(root, "summaries")
    figdir = os.path.join(root, f"figures_{args.stage or 'all'}")
    os.makedirs(figdir, exist_ok=True)
    runs = _load_runs(raw_dir)
    fig_model(figdir)
    fig_rb_torsion(figdir)
    fig_butane_reference(figdir)
    fig_pentane_reference(figdir, runs)
    fig_convergence(figdir, runs, "butane", "fig_alk_06_butane_convergence.png")
    fig_convergence(figdir, runs, "pentane", "fig_alk_08_pentane_convergence.png")
    fig_equivalence(figdir, summ_dir)
    fig_pentane_conditional(figdir, runs)
    fig_pentane_basins(figdir, runs)
    fig_rate_ladder(figdir, args.tuning_root)
    fig_cross_case(figdir, summ_dir)
    ensure_all(figdir)
    print(f"[plot] wrote figures to {figdir}")
    if args.report_figdir:
        import shutil
        os.makedirs(args.report_figdir, exist_ok=True)
        for f in glob.glob(os.path.join(figdir, "*.png")):
            shutil.copy(f, args.report_figdir)
        print(f"[plot] copied to {args.report_figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
