#!/usr/bin/env python3
"""Starvation diagnostics for the WCA phase diagram (Part B).

This is a *pure re-analysis* of the existing WCA phase-diagram raw runs
(``results/wca_phase_diagram/<stage>/raw/*.npz``): it computes no new dynamics and
launches no GPU work. Every quantity below is either stored in each run's npz or
derived from the stored final profiles / time series, so the script is cheap and
idempotent.

It tests the central reviewer-facing claim of the WCA study:

    The *measured* ABF baseline error, not the nominal product beta*h, is the
    useful sample-starvation diagnostic for when marginal Fisher--Rao (mFR) helps.

Outputs (under --out, default <output_root>/starvation/):

  wca_starvation_summary.csv        one row per run: final/integrated L2(F),L2(F'),
                                    Neff support (transition/compact/stretched + a
                                    scale-free transition-vs-median ratio), per-bin
                                    transition mean-force error, RC-marginal
                                    distances (L2/TV/KL vs uniform and vs Boltzmann
                                    reference), transition/round-trip counts,
                                    birth-death safety (event fractions), and
                                    ancestor genealogy (ESS, ESS fraction, unique
                                    ancestors, max ancestor fraction).
  wca_phase_diagram_augmented.csv   one row per (cell, FR method): matched-seed
                                    median gain% and improvement ratio R over ABF,
                                    win rate, ABF starvation diagnostics, FR safety
                                    + genealogy medians, and a regime label
                                    (starved / intermediate / easy) derived from the
                                    measured ABF error.
  manifest.json                     git commit, argv, params, device, timestamp.

  plots/ :
    fig_starv_01_abferr_vs_gain.{png}     THE key plot: ABF final L2(F) (x) vs
                                          matched-seed median mFR gain% (y), annotated
                                          by (beta,h); the diagnostic-quality test.
    fig_starv_02_neff_vs_gain.png         ABF transition Neff (mean & min) vs gain.
    fig_starv_03_ess_vs_l2.png            FR ancestor ESS vs FR final L2(F).
    fig_starv_04_crossratio_vs_gain.png   transition-count ratio (FR/ABF) vs gain.
    fig_starv_05_error_vs_time.png        representative error-vs-time curves.
    fig_starv_06_genealogy_vs_time.png    representative genealogy-vs-time curves.

Usage:
  python scripts/analyze_wca_starvation.py \
      --config configs/wca_phase_diagram_production.yaml --stages production
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import wca_phase_jobs as pj  # noqa: E402

EPS = 1.0e-12


# --------------------------------------------------------------------------- #
# small helpers (mirror analyze_wca_phase_diagram conventions)
# --------------------------------------------------------------------------- #
def _val(d, k, default=None):
    if k not in d:
        return default
    v = d[k]
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode()
    return v


def _physics_tag(d):
    return (f"b{_val(d,'beta'):g}_h{_val(d,'h'):g}_w{_val(d,'w'):g}"
            f"_n{int(_val(d,'n_dim'))}_a{_val(d,'a'):g}")


def _iqr(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_csv(rows, out_path, cols=None):
    if not rows:
        print(f"  no rows -> skip {out_path}")
        return
    cols = cols or list(rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out_path} ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
# density distances on the interior evaluation window
# --------------------------------------------------------------------------- #
def _renorm(p, grid, mask):
    p = np.clip(np.asarray(p, float), 0.0, None)[mask]
    g = np.asarray(grid, float)[mask]
    z = np.trapezoid(p, g)
    if not np.isfinite(z) or z <= 0:
        return None, g
    return p / z, g


def tv_distance(p, q, grid, mask):
    pp, g = _renorm(p, grid, mask)
    qq, _ = _renorm(q, grid, mask)
    if pp is None or qq is None:
        return float("nan")
    return float(0.5 * np.trapezoid(np.abs(pp - qq), g))


def kl_divergence(p, q, grid, mask):
    pp, g = _renorm(p, grid, mask)
    qq, _ = _renorm(q, grid, mask)
    if pp is None or qq is None:
        return float("nan")
    pp = np.clip(pp, EPS, None)
    qq = np.clip(qq, EPS, None)
    return float(np.trapezoid(pp * np.log(pp / qq), g))


def l2_density(p, q, grid, mask):
    pp, g = _renorm(p, grid, mask)
    qq, _ = _renorm(q, grid, mask)
    if pp is None or qq is None:
        return float("nan")
    return float(np.sqrt(np.trapezoid((pp - qq) ** 2, g) / (g[-1] - g[0])))


# --------------------------------------------------------------------------- #
# per-run augmented metrics (all derived from the stored npz)
# --------------------------------------------------------------------------- #
def per_run_metrics(d, tlo, thi, elo, ehi):
    grid = np.asarray(d["grid"], float)
    eval_mask = (grid >= elo) & (grid <= ehi)
    tmask = (grid >= tlo) & (grid <= thi) & eval_mask
    cmask = (grid < tlo) & eval_mask
    smask = (grid > thi) & eval_mask

    method = str(_val(d, "method"))
    is_fr = method != "abf"

    # --- accuracy ---
    l2_f = float(_val(d, "l2_f", np.nan))
    l2_fp = float(_val(d, "l2_fp", np.nan))
    times = np.asarray(_val(d, "times", np.array([])), float)
    l2_f_t = np.asarray(_val(d, "l2_f_t", np.array([])), float)
    l2_fp_t = np.asarray(_val(d, "l2_fp_t", np.array([])), float)
    integ_f = float(_val(d, "integrated_l2_f", np.nan))
    integ_fp = (float(np.trapezoid(l2_fp_t, times))
                if times.size > 1 and l2_fp_t.size == times.size else float("nan"))

    # --- ABF support / denominator (kernel-weighted cumulative count per bin) ---
    eff = np.asarray(_val(d, "final_eff_counts", np.full_like(grid, np.nan)), float)
    neff_tr_mean = float(np.nanmean(eff[tmask])) if tmask.any() else float("nan")
    neff_tr_min = float(np.nanmin(eff[tmask])) if tmask.any() else float("nan")
    neff_compact = float(np.nanmean(eff[cmask])) if cmask.any() else float("nan")
    neff_stretched = float(np.nanmean(eff[smask])) if smask.any() else float("nan")
    eff_med = float(np.nanmedian(eff[eval_mask])) if eval_mask.any() else float("nan")
    # scale-free transition support: transition support / median support (=> robust
    # across cells/snapshots; <1 means the transition window is under-supported).
    neff_tr_ratio = (neff_tr_mean / eff_med) if (eff_med and np.isfinite(eff_med) and eff_med > 0) else float("nan")
    neff_tr_min_ratio = (neff_tr_min / eff_med) if (eff_med and np.isfinite(eff_med) and eff_med > 0) else float("nan")

    # --- per-bin mean-force error in the transition window (no alignment for F') ---
    mf = np.asarray(_val(d, "final_mean_force", np.full_like(grid, np.nan)), float)
    rmf = np.asarray(_val(d, "ref_mean_force", np.full_like(grid, np.nan)), float)
    if tmask.any():
        gt = grid[tmask]
        tw_mf_err = float(np.sqrt(np.trapezoid((mf[tmask] - rmf[tmask]) ** 2, gt) / (gt[-1] - gt[0]))) \
            if gt.size > 1 else float("nan")
    else:
        tw_mf_err = float("nan")

    # --- RC marginal distances vs uniform and vs Boltzmann reference ---
    p_hat = np.asarray(_val(d, "final_p_hat", np.full_like(grid, np.nan)), float)
    pref = np.asarray(_val(d, "ref_p_boltzmann", np.full_like(grid, np.nan)), float)
    uniform = np.ones_like(grid)
    marg_l2_uniform = float(_val(d, "marginal_l2_uniform", np.nan))
    marg_l2_ref = float(_val(d, "marginal_l2_ref", np.nan))
    tv_ref = tv_distance(p_hat, pref, grid, eval_mask)
    kl_ref = kl_divergence(p_hat, pref, grid, eval_mask)
    tv_uniform = tv_distance(p_hat, uniform, grid, eval_mask)

    row = dict(
        run_id=str(_val(d, "run_id")), stage=str(_val(d, "stage")),
        method=method, physics_tag=_physics_tag(d),
        beta=_val(d, "beta"), h=_val(d, "h"), w=_val(d, "w"),
        n_dim=int(_val(d, "n_dim")), M=int(_val(d, "M")), a=_val(d, "a"),
        beta_h=_val(d, "beta_h"), seed=int(_val(d, "seed")),
        n_steps=int(_val(d, "n_steps")), n_replicas=int(_val(d, "n_replicas")),
        # accuracy
        final_l2_f=l2_f, final_l2_fp=l2_fp,
        integrated_l2_f=integ_f, integrated_l2_fp=integ_fp,
        l2_f_transition=float(_val(d, "l2_f_transition", np.nan)),
        l2_fp_transition=float(_val(d, "l2_fp_transition", np.nan)),
        # ABF support
        neff_transition_mean=neff_tr_mean, neff_transition_min=neff_tr_min,
        neff_compact=neff_compact, neff_stretched=neff_stretched,
        neff_median_eval=eff_med, neff_transition_ratio=neff_tr_ratio,
        neff_transition_min_ratio=neff_tr_min_ratio,
        transition_window_mean_force_error=tw_mf_err,
        # marginal
        marginal_l2_uniform=marg_l2_uniform, marginal_l2_ref=marg_l2_ref,
        marginal_tv_ref=tv_ref, marginal_kl_ref=kl_ref, marginal_tv_uniform=tv_uniform,
        # transitions
        n_compact_to_stretched=int(_val(d, "n_compact_to_stretched", -1)),
        n_stretched_to_compact=int(_val(d, "n_stretched_to_compact", -1)),
        n_barrier_crossings=int(_val(d, "n_barrier_crossings", -1)),
        n_round_trips=int(_val(d, "n_round_trips", -1)),
        # birth-death safety
        fr_event_fraction=float(_val(d, "fr_event_fraction", np.nan)) if is_fr else float("nan"),
        max_fr_event_fraction=float(_val(d, "max_fr_event_fraction", np.nan)) if is_fr else float("nan"),
        deaths_per_fr_application=float(_val(d, "deaths_per_fr_application", np.nan)) if is_fr else float("nan"),
        total_replacement_events=int(_val(d, "total_replacement_events", 0)),
        # score stats are not stored by the original phase runs (re-runs add them);
        # left blank here so the schema is stable and honest.
        score_std=_val(d, "fr_score_std", ""),
        score_max=_val(d, "fr_score_absmax", ""),
        score_clip_fraction=_val(d, "fr_score_clip_fraction", ""),
        # genealogy
        final_ancestor_ess=float(_val(d, "final_ancestor_ess", np.nan)) if is_fr else float("nan"),
        min_ancestor_ess=float(_val(d, "min_ancestor_ess", np.nan)) if is_fr else float("nan"),
        final_n_unique_ancestor=int(_val(d, "final_n_unique_ancestor", -1)) if is_fr else -1,
        final_max_ancestor_frac=float(_val(d, "final_max_ancestor_frac", np.nan)) if is_fr else float("nan"),
        ess_fraction=((float(_val(d, "final_ancestor_ess", np.nan)) / float(_val(d, "n_replicas")))
                      if is_fr else float("nan")),
        had_nan=bool(_val(d, "had_nan", False)),
    )
    return row


# --------------------------------------------------------------------------- #
# load runs
# --------------------------------------------------------------------------- #
def load_runs(raw_dir, stages):
    runs = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        try:
            d = pj.load_run(path)
        except Exception as exc:
            print(f"  skip unreadable {path}: {exc!r}")
            continue
        if "l2_f" not in d:
            continue
        if stages and str(_val(d, "stage")) not in stages:
            continue
        runs.append(d)
    return runs


# --------------------------------------------------------------------------- #
# matched-seed pairing and the augmented per-cell table
# --------------------------------------------------------------------------- #
def _mkey(r):
    return (r["stage"], r["physics_tag"], r["n_steps"], r["n_replicas"], r["seed"])


def _cellkey(r):
    return (r["stage"], r["physics_tag"], r["n_steps"], r["n_replicas"])


def regime_label(abf_l2_f, starved_thr, easy_thr):
    if not np.isfinite(abf_l2_f):
        return "unknown"
    if abf_l2_f >= starved_thr:
        return "starved"
    if abf_l2_f <= easy_thr:
        return "easy"
    return "intermediate"


def build_augmented(summary_rows, starved_thr, easy_thr):
    abf = {_mkey(r): r for r in summary_rows if r["method"] == "abf"}
    # group fr rows by (cell, method)
    cells = {}
    for r in summary_rows:
        if r["method"] == "abf":
            continue
        cells.setdefault((_cellkey(r), r["method"]), []).append(r)

    # ABF medians per cell (for diagnostics + regime)
    abf_cell = {}
    for r in summary_rows:
        if r["method"] != "abf":
            continue
        abf_cell.setdefault(_cellkey(r), []).append(r)

    def cmed(rows, key):
        return _iqr([x[key] for x in rows])[0]

    out = []
    for (ck, method), frs in sorted(cells.items()):
        abf_rows = abf_cell.get(ck, [])
        d0 = frs[0]
        gains, R, wins, n = [], [], 0, 0
        for fr in frs:
            base = abf.get(_mkey(fr))
            if base is None:
                continue
            bf, ff = base["final_l2_f"], fr["final_l2_f"]
            if not (np.isfinite(bf) and np.isfinite(ff)):
                continue
            n += 1
            wins += int(ff < bf)
            if bf > 0:
                gains.append(100.0 * (bf - ff) / bf)
            if ff > 0:
                R.append(bf / ff)
        abf_l2_med = cmed(abf_rows, "final_l2_f")
        rec = dict(
            stage=ck[0], physics_tag=ck[1], n_steps=ck[2], n_replicas=ck[3],
            method=method, beta=d0["beta"], h=d0["h"], M=d0["M"], beta_h=d0["beta_h"],
            n_seeds=n, n_wins=wins, win_rate=(wins / n if n else float("nan")),
            median_gain_pct_F=(float(np.median(gains)) if gains else float("nan")),
            q25_gain_pct_F=(float(np.percentile(gains, 25)) if gains else float("nan")),
            q75_gain_pct_F=(float(np.percentile(gains, 75)) if gains else float("nan")),
            R_final_median=(float(np.median(R)) if R else float("nan")),
            # ABF starvation diagnostics (the proposed predictor)
            abf_final_l2_f_median=abf_l2_med,
            abf_final_l2_fp_median=cmed(abf_rows, "final_l2_fp"),
            abf_neff_transition_ratio_median=cmed(abf_rows, "neff_transition_ratio"),
            abf_neff_transition_min_ratio_median=cmed(abf_rows, "neff_transition_min_ratio"),
            abf_transition_mf_err_median=cmed(abf_rows, "transition_window_mean_force_error"),
            abf_marginal_tv_ref_median=cmed(abf_rows, "marginal_tv_ref"),
            abf_barrier_crossings_median=cmed(abf_rows, "n_barrier_crossings"),
            # FR diagnostics
            fr_final_l2_f_median=cmed(frs, "final_l2_f"),
            fr_event_fraction_median=cmed(frs, "fr_event_fraction"),
            fr_max_event_fraction_median=cmed(frs, "max_fr_event_fraction"),
            fr_final_ess_median=cmed(frs, "final_ancestor_ess"),
            fr_ess_fraction_median=cmed(frs, "ess_fraction"),
            fr_max_anc_frac_median=cmed(frs, "final_max_ancestor_frac"),
            fr_barrier_crossings_median=cmed(frs, "n_barrier_crossings"),
            fr_transition_mf_err_median=cmed(frs, "transition_window_mean_force_error"),
            # transition-count ratio FR/ABF (median of cell medians)
            crossing_ratio_fr_over_abf=(cmed(frs, "n_barrier_crossings")
                                        / cmed(abf_rows, "n_barrier_crossings")
                                        if cmed(abf_rows, "n_barrier_crossings") else float("nan")),
            regime=regime_label(abf_l2_med, starved_thr, easy_thr),
        )
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _annot(ax, x, y, label):
    ax.annotate(label, (x, y), fontsize=7, textcoords="offset points", xytext=(4, 3))


def make_plots(summary_rows, augmented, runs, plot_dir, focus_method="fr_estimated"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    aug = [r for r in augmented if r["method"] == focus_method]
    color = {"starved": "#2c7fb8", "intermediate": "#d95f0e", "easy": "#cc0033",
             "unknown": "#888888"}

    # 1) THE key plot: ABF final L2(F) vs matched-seed median gain
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    pts = [r for r in aug if np.isfinite(r["abf_final_l2_f_median"]) and np.isfinite(r["median_gain_pct_F"])]
    # the starved beta=1, M=100 cells form a tight top-right cluster; label as a group
    cluster = [r for r in pts if r["beta"] == 1 and r["M"] == 100]
    for r in pts:
        x, y = r["abf_final_l2_f_median"], r["median_gain_pct_F"]
        ax.scatter(x, y, s=48, color=color.get(r["regime"], "#888"),
                   edgecolor="k", linewidth=0.4, zorder=3)
        if r in cluster:
            continue
        lab = f"b{r['beta']:g}h{r['h']:g}" + (f"M{r['M']}" if r['M'] != 100 else "")
        ax.annotate(lab, (x, y), fontsize=7, textcoords="offset points", xytext=(5, 3))
    if cluster:
        cx = float(np.median([r["abf_final_l2_f_median"] for r in cluster]))
        cy = float(np.median([r["median_gain_pct_F"] for r in cluster]))
        ax.annotate(r"$\beta{=}1$ row" + "\n(h=1,2,4,6)", (cx, cy),
                    fontsize=8, ha="right", va="top", textcoords="offset points", xytext=(-12, -6),
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#555"))
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("ABF final $L_2(F)$  (measured baseline error)")
    ax.set_ylabel(f"matched-seed median gain of {focus_method} over ABF (%)")
    ax.set_title("Measured ABF error predicts where mFR helps")
    handles = [plt.Line2D([0], [0], marker='o', ls='', color=color[k],
               markeredgecolor='k', label=k) for k in ("starved", "intermediate", "easy")]
    ax.legend(handles=handles, title="regime (by ABF error)", loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "fig_starv_01_abferr_vs_gain.png"), dpi=150)
    plt.close(fig)

    # 2) ABF transition Neff ratio (mean & min) vs gain
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, lab in zip(axes,
                            ["abf_neff_transition_ratio_median", "abf_neff_transition_min_ratio_median"],
                            ["mean transition support / median", "min transition support / median"]):
        for r in aug:
            x, y = r[key], r["median_gain_pct_F"]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax.scatter(x, y, s=40, color=color.get(r["regime"], "#888"), edgecolor="k", linewidth=0.4)
            _annot(ax, x, y, f"b{r['beta']:g}h{r['h']:g}")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel(f"ABF {lab}")
        ax.set_ylabel("median gain (%)")
        ax.grid(alpha=0.3)
    axes[0].set_title("ABF transition support vs mFR gain")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "fig_starv_02_neff_vs_gain.png"), dpi=150)
    plt.close(fig)

    # 3) FR ancestor ESS vs FR final L2(F)
    fig, ax = plt.subplots(figsize=(7, 5))
    frrows = [r for r in summary_rows if r["method"] == focus_method]
    xs = [r["final_ancestor_ess"] for r in frrows]
    ys = [r["final_l2_f"] for r in frrows]
    cs = []
    auglook = {(_cellkey_from_row(r)): r for r in aug}
    for r in frrows:
        a = auglook.get(_cellkey_from_row(r))
        cs.append(color.get(a["regime"], "#888") if a else "#888")
    ax.scatter(xs, ys, s=34, c=cs, edgecolor="k", linewidth=0.3, alpha=0.85)
    ax.set_xlabel(f"{focus_method} final ancestor ESS  (out of N)")
    ax.set_ylabel(f"{focus_method} final $L_2(F)$")
    ax.set_title("Ancestor diversity vs accuracy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "fig_starv_03_ess_vs_l2.png"), dpi=150)
    plt.close(fig)

    # 4) transition-count ratio (FR/ABF) vs gain
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in aug:
        x, y = r["crossing_ratio_fr_over_abf"], r["median_gain_pct_F"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax.scatter(x, y, s=44, color=color.get(r["regime"], "#888"), edgecolor="k", linewidth=0.4)
        _annot(ax, x, y, f"b{r['beta']:g}h{r['h']:g}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(1, color="k", lw=0.8, ls=":")
    ax.set_xlabel("barrier-crossing ratio (mFR / ABF)")
    ax.set_ylabel("median gain (%)")
    ax.set_title("More crossings does not imply more accuracy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "fig_starv_04_crossratio_vs_gain.png"), dpi=150)
    plt.close(fig)

    # 5) representative error-vs-time (one starved, one easy, one intermediate)
    reps = _pick_representative_cells(aug)
    fig, ax = plt.subplots(figsize=(7.6, 5))
    for regime, tag in reps.items():
        for method, ls in [("abf", "--"), (focus_method, "-")]:
            curves = _stack_timeseries(runs, tag, method, "l2_f_t")
            if curves is None:
                continue
            t, med, lo, hi = curves
            line, = ax.plot(t, med, ls, lw=1.8, label=f"{regime}:{method}")
            ax.fill_between(t, lo, hi, alpha=0.15, color=line.get_color())
    ax.set_xlabel("time"); ax.set_ylabel("$L_2(F)$")
    ax.set_title("Error vs time (median, IQR) — representative cells")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "fig_starv_05_error_vs_time.png"), dpi=150)
    plt.close(fig)

    # 6) representative genealogy-vs-time
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for regime, tag in reps.items():
        ess = _stack_timeseries(runs, tag, focus_method, "ancestor_ess_t")
        maf = _stack_timeseries(runs, tag, focus_method, "max_ancestor_frac_t")
        if ess is not None:
            t, med, lo, hi = ess
            line, = axes[0].plot(t, med, lw=1.8, label=regime)
            axes[0].fill_between(t, lo, hi, alpha=0.15, color=line.get_color())
        if maf is not None:
            t, med, lo, hi = maf
            line, = axes[1].plot(t, med, lw=1.8, label=regime)
            axes[1].fill_between(t, lo, hi, alpha=0.15, color=line.get_color())
    axes[0].set_xlabel("time"); axes[0].set_ylabel("ancestor ESS"); axes[0].set_title(f"{focus_method} ancestor ESS")
    axes[1].set_xlabel("time"); axes[1].set_ylabel("max ancestor fraction"); axes[1].set_title("max ancestor fraction")
    for a in axes:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "fig_starv_06_genealogy_vs_time.png"), dpi=150)
    plt.close(fig)
    print(f"  wrote 6 figures -> {plot_dir}")


def _cellkey_from_row(r):
    return (r["stage"], r["physics_tag"], r["n_steps"], r["n_replicas"])


def _pick_representative_cells(aug):
    """Pick one starved / intermediate / easy M=100 cell by ABF error extremes."""
    m100 = [r for r in aug if r["M"] == 100 and np.isfinite(r["abf_final_l2_f_median"])]
    reps = {}
    if not m100:
        return reps
    starved = max(m100, key=lambda r: r["abf_final_l2_f_median"])
    easy = min(m100, key=lambda r: r["abf_final_l2_f_median"])
    reps["starved"] = starved["physics_tag"]
    reps["easy"] = easy["physics_tag"]
    inter = [r for r in m100 if r["regime"] == "intermediate"]
    if inter:
        reps["intermediate"] = sorted(inter, key=lambda r: r["abf_final_l2_f_median"])[len(inter) // 2]["physics_tag"]
    return reps


def _stack_timeseries(runs, physics_tag, method, key):
    sel = [d for d in runs if _physics_tag(d) == physics_tag and str(_val(d, "method")) == method]
    if not sel:
        return None
    t = np.asarray(_val(sel[0], "times"), float)
    mats = []
    for d in sel:
        v = np.asarray(_val(d, key), float)
        if v.size == t.size:
            mats.append(v)
    if not mats:
        return None
    A = np.vstack(mats)
    with np.errstate(all="ignore"):
        med = np.nanmedian(A, axis=0)
        lo = np.nanpercentile(A, 25, axis=0)
        hi = np.nanpercentile(A, 75, axis=0)
    return t, med, lo, hi


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw", default=None, help="raw dir (default <output_root>/raw)")
    ap.add_argument("--out", default=None, help="output dir (default <output_root>/starvation)")
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--focus-method", default="fr_estimated",
                    help="deployable FR method used for the gain scatter")
    ap.add_argument("--starved-threshold", type=float, default=0.05,
                    help="ABF final L2(F) >= this => starved regime")
    ap.add_argument("--easy-threshold", type=float, default=0.02,
                    help="ABF final L2(F) <= this => easy regime")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    cfg = pj.load_yaml(args.config)
    base = cfg.get("base", {})
    tlo = float(base.get("transition_lo", 0.25))
    thi = float(base.get("transition_hi", 0.75))
    elo = float(base.get("eval_z_lo", -0.1))
    ehi = float(base.get("eval_z_hi", 1.1))
    raw_dir = args.raw or os.path.join(cfg["output_root"], "raw")
    out_dir = args.out or os.path.join(cfg["output_root"], "starvation")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[starvation] raw={raw_dir} out={out_dir} stages={args.stages or 'ALL'} "
          f"transition=[{tlo},{thi}] eval=[{elo},{ehi}]")
    runs = load_runs(raw_dir, args.stages)
    if not runs:
        print("[starvation] no runs found")
        return 1
    print(f"[starvation] loaded {len(runs)} runs")

    summary_rows = [per_run_metrics(d, tlo, thi, elo, ehi) for d in runs]
    write_csv(summary_rows, os.path.join(out_dir, "wca_starvation_summary.csv"))

    augmented = build_augmented(summary_rows, args.starved_threshold, args.easy_threshold)
    write_csv(augmented, os.path.join(out_dir, "wca_phase_diagram_augmented.csv"))

    if not args.no_plots:
        make_plots(summary_rows, augmented, runs,
                   os.path.join(out_dir, "plots"), focus_method=args.focus_method)

    manifest = dict(
        script="analyze_wca_starvation.py", git_commit=_git_commit(),
        argv=sys.argv, config=os.path.abspath(args.config), raw_dir=os.path.abspath(raw_dir),
        stages=args.stages, n_runs=len(runs), focus_method=args.focus_method,
        transition_window=[tlo, thi], eval_window=[elo, ehi],
        starved_threshold=args.starved_threshold, easy_threshold=args.easy_threshold,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), code_version="wca_starvation_v1",
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  wrote {os.path.join(out_dir, 'manifest.json')}")
    print("[starvation] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
