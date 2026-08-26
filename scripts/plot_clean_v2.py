#!/usr/bin/env python3
"""The clean-v2 figures: convergence first, error last.

Four panels, in the order the claim is made:

Figure 1  e_F(t) against physical time (and force evaluations) for plain ABF and
          the FR arm, median with IQR band, with t_burn and t_off marked.
Figure 2  the time-to-accuracy speedup S_eps at both frozen thresholds -- the
          most important quantitative figure.
Figure 3  e_F'(t), the quantity ABF actually learns.
Figure 4  mechanism: phat_t, q_t and the reference physical marginal at selected
          times, plus the genealogy appendix.

Nothing here plots final error as a headline.  It appears once, as an annotation
on Figure 1, because its job in this campaign is to show the curves *converging
back together* after FR switches off -- which is what acceleration looks like,
not what failure looks like.

Example
-------
  python scripts/plot_clean_v2.py \
      --stage-root results/clean_v2/stage3_confirmation/confirmation \
      --thresholds results/clean_v2/thresholds.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abffr import accel  # noqa: E402

ARM_STYLE = {
    "abf_only": dict(color="#444444", label="plain ABF"),
    "abf_fr_physical": dict(color="#c1121f",
                            label="ABF + intermittent physical-target FR"),
    "abf_fr_physical_oracle": dict(color="#0b6e4f", ls="--",
                                   label="oracle physical target (diagnostic)"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage-root", required=True)
    p.add_argument("--thresholds", required=True)
    p.add_argument("--scope", default=None)
    p.add_argument("--acceleration", default=None,
                   help="acceleration.csv (default: <stage-root>/acceleration.csv)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--logy", action="store_true", default=True)
    return p.parse_args(argv)


def _load(stage_root, kind):
    hits = [f for f in os.listdir(stage_root)
            if f.endswith(f"_{kind}.csv") and "__" not in f]
    return (pd.read_csv(os.path.join(stage_root, sorted(hits)[0]))
            if hits else None)


def _band(ax, df, col, style, band=True, alpha=1.0):
    g = df.groupby("t")[col]
    t = np.asarray(sorted(df["t"].unique()))
    med = g.median().reindex(t).to_numpy()
    if band:
        lo = g.quantile(0.25).reindex(t).to_numpy()
        hi = g.quantile(0.75).reindex(t).to_numpy()
        ax.fill_between(t, lo, hi, color=style["color"], alpha=0.15, lw=0)
    kw = {k: v for k, v in style.items() if k != "label"}
    ax.plot(t, med, lw=1.8 if band else 1.1, alpha=alpha, label=style["label"],
            **kw)
    return t, med


def _series(long_df):
    """``[(method, config_id, sub_df, is_single)]`` -- never pool schedules.

    Averaging several FR schedules into one curve would draw a method that no
    run performed.  When a stage carries more than one schedule per method each
    is drawn separately and labelled with its own (gamma, L_FR).
    """
    out = []
    for m in [k for k in ARM_STYLE if k in set(long_df["method"])]:
        sub = long_df[long_df["method"] == m]
        cids = sorted(sub["config_id"].unique())
        for cid in cids:
            out.append((m, cid, sub[sub["config_id"] == cid], len(cids) == 1))
    return out


def _phase_marks(ax, t_burn, t_off, y=0.97):
    for t, lab in ((t_burn, r"$t_{\rm burn}$"), (t_off, r"$t_{\rm off}$")):
        ax.axvline(t, color="#888888", lw=0.8, ls=":", zorder=0)
        ax.annotate(lab, xy=(t, y), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=8, color="#555555")
    ax.axvspan(t_burn, t_off, color="#c1121f", alpha=0.04, lw=0, zorder=0)


def _force_axis(ax, long_df, n_particles=None):
    """Top axis in force evaluations: the cost currency, not wall clock."""
    if n_particles is None:
        return
    top = ax.secondary_xaxis(
        "top", functions=(lambda t: t, lambda t: t))
    top.set_xlabel("force evaluations")


def main(argv=None):
    args = parse_args(argv)
    frozen = json.load(open(args.thresholds))
    scope = args.scope or frozen["primary_scope"]
    eps_F = frozen["thresholds"][scope]["F"]
    eps_Fp = frozen["thresholds"][scope]["Fprime"]
    fracs = frozen["fractions"]

    long_df = _load(args.stage_root, "runs_long")
    if long_df is None:
        raise SystemExit(f"no merged *_runs_long.csv under {args.stage_root}")
    out_dir = args.out_dir or os.path.join(args.stage_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    series = _series(long_df)
    fr_rows = long_df[long_df["method"] != "abf_only"]
    t_burn = float(fr_rows["burnin_fraction"].iloc[0]) * float(long_df["t"].max()) \
        if not fr_rows.empty else np.nan
    t_off = float(fr_rows["stop_fraction"].iloc[0]) * float(long_df["t"].max()) \
        if not fr_rows.empty else np.nan

    # ---------------- Figure 1 / 3: convergence ---------------------------- #
    for fig_no, col, eps, label in (
            (1, f"l2_F_{scope}", eps_F, r"$e_F(t)$"),
            (3, f"l2_Fprime_{scope}", eps_Fp, r"$e_{F'}(t)$")):
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        multi = [g for g in series if not g[3]]
        for j, (m, cid, sub, single) in enumerate(series):
            style = dict(ARM_STYLE[m])
            if not single:
                r = sub.iloc[0]
                style["label"] = (f"{style['label']}  "
                                  f"($\\gamma$={r['gamma']:g}, "
                                  f"$L$={int(r['fr_every'])})")
                style["alpha"] = 0.85
            _band(ax, sub, col, style, band=single,
                  alpha=style.pop("alpha", 1.0))
        if multi:
            print(f"[plot] {len(multi)} schedules drawn separately; schedules "
                  f"are never averaged into a single curve.")
        for e, f in zip(eps, fracs):
            ax.axhline(e, color="#3a86ff", lw=0.8, ls="--", zorder=0)
            ax.annotate(rf"$\epsilon$ (={f:g}$T$ ABF)", xy=(0.995, e),
                        xycoords=("axes fraction", "data"), ha="right",
                        va="bottom", fontsize=7.5, color="#3a86ff")
        if np.isfinite(t_burn):
            _phase_marks(ax, t_burn, t_off)
        if args.logy:
            ax.set_yscale("log")
        ax.set_xlabel("physical time $t$")
        _force_axis(ax, long_df)
        ax.set_ylabel(label + f"   [scope {scope}]")
        ax.set_title("Free-energy convergence" if fig_no == 1
                     else "Mean-force convergence", loc="left", fontsize=11)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        ax.grid(alpha=0.15, lw=0.5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"fig{fig_no}_convergence_"
                                     f"{'F' if fig_no == 1 else 'Fprime'}.{ext}"),
                        dpi=200)
        plt.close(fig)

    # ---------------- Figure 2: the speedup -------------------------------- #
    acc_path = args.acceleration or os.path.join(args.stage_root,
                                                 "acceleration.csv")
    if os.path.exists(acc_path):
        acc = pd.read_csv(acc_path)
        acc = acc[acc["scope"] == scope]
        fig, ax = plt.subplots(figsize=(6.4, 0.6 + 0.55 * max(len(acc), 2)))
        y = np.arange(len(acc))[::-1]
        for k, (dx, c, lab) in enumerate(
                ((-0.16, "#3a86ff", rf"$S_{{F,1}}$ ({fracs[0]:g}$T$)"),
                 (0.16, "#c1121f", rf"$S_{{F,2}}$ ({fracs[1]:g}$T$)")), start=1):
            s = acc[f"S_F_{k}"].to_numpy()
            lo = acc[f"S_F_{k}_lo"].to_numpy()
            hi = acc[f"S_F_{k}_hi"].to_numpy()
            ax.errorbar(s, y + dx, xerr=[s - lo, hi - s], fmt="o", ms=4,
                        lw=1.2, capsize=2.5, color=c, label=lab)
        ax.axvline(1.0, color="#444444", lw=1.0)
        ax.axvline(accel.PILOT_MIN_S_F, color="#888888", lw=0.8, ls="--")
        ax.annotate(f"pre-declared {accel.PILOT_MIN_S_F}",
                    xy=(accel.PILOT_MIN_S_F, 1.0), xycoords=("data", "axes fraction"),
                    fontsize=7.5, color="#888888", ha="left", va="top", rotation=90)
        ax.set_yticks(y)
        short = {"abf_fr_physical": "physical",
                 "abf_fr_physical_oracle": "oracle (diag.)"}
        ax.set_yticklabels(
            [f"{short.get(r.method, r.method)}  "
             f"$\\gamma$={r.gamma:g}, $L_{{\\rm FR}}$={r.fr_every}"
             f"  (n={r.n_matched_seeds})" for r in acc.itertuples()], fontsize=8)
        ax.set_xlabel(r"time-to-accuracy speedup  $S_\epsilon = "
                      r"\mathbb{E}[\tilde\tau^{\rm ABF}]/"
                      r"\mathbb{E}[\tilde\tau^{\rm ABF+FR}]$")
        ax.set_title("Less simulation time to the same free-energy accuracy",
                     loc="left", fontsize=11)
        ax.legend(fontsize=8, frameon=False, loc="best")
        ax.grid(axis="x", alpha=0.15, lw=0.5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"fig2_speedup.{ext}"), dpi=200)
        plt.close(fig)
    else:
        print(f"[plot] no {os.path.relpath(acc_path)}; skipping Figure 2")

    # ---------------- Figure 5: threshold-reaching survival ----------------- #
    # S is a RESTRICTED ratio at horizon T, so which way censoring biases it is
    # decided by how many seeds on each side ever cross.  This panel shows that
    # directly: a genuine acceleration is a curve that steps DOWN EARLIER, not
    # one that merely ends lower because its seeds ran out of budget.
    fig, axes = plt.subplots(1, len(eps_F), figsize=(4.6 * len(eps_F), 3.6),
                             squeeze=False)
    for k, (eps, frac) in enumerate(zip(eps_F, fracs)):
        ax = axes[0][k]
        for m, cid, sub_df, single in series:
            style = ARM_STYLE[m]
            taus = []
            for _, g in sub_df.sort_values("t").groupby("seed"):
                taus.append(accel.hitting_time(
                    g["t"].to_numpy(), g[f"l2_F_{scope}"].to_numpy(), eps,
                    consecutive=accel.CONSECUTIVE_FRAMES))
            taus = np.asarray(taus, dtype=float)
            grid = np.asarray(sorted(long_df["t"].unique()))
            surv = [(taus > t).mean() for t in grid]
            lab = style["label"] if single else (
                f"{style['label']} ($\\gamma$={sub_df['gamma'].iloc[0]:g}, "
                f"$L$={int(sub_df['fr_every'].iloc[0])})")
            ax.step(grid, surv, where="post", lw=1.6 if single else 1.1,
                    color=style["color"], ls=style.get("ls", "-"), label=lab)
            # The hit fraction belongs next to the curve it qualifies: a curve
            # that ends at 0.2 because a fifth of its seeds never crossed reads
            # very differently from one that ends at 0.2 slowly.
            ax.plot([], [], " ",
                    label=f"      hits by $T$: {np.isfinite(taus).mean():.0%}")
        if np.isfinite(t_burn):
            _phase_marks(ax, t_burn, t_off)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("physical time $t$")
        ax.set_ylabel(r"$P(\tau_\epsilon > t)$")
        ax.set_title(rf"threshold $\epsilon$ = {eps:.4g}  ({frac:g}$T$ ABF)",
                     loc="left", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.grid(alpha=0.15, lw=0.5)
    fig.suptitle("Who crosses the threshold, and when  "
                 "(a flat tail at 1 is censoring, not slowness)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig5_threshold_survival.{ext}"),
                    dpi=200)
    plt.close(fig)

    # ---------------- Figure 4: mechanism + genealogy ---------------------- #
    prof = _load(args.stage_root, "profiles")
    pulses = _load(args.stage_root, "fr_pulses")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    if prof is not None:
        fr_name = "abf_fr_physical"
        sub = prof[prof["method"] == fr_name] if fr_name in set(prof["method"]) \
            else prof
        g = sub.groupby("x")
        x = np.asarray(sorted(sub["x"].unique()))
        axes[0].plot(x, g["p_hat"].median().reindex(x), color="#c1121f", lw=1.6,
                     label=r"$\hat p_T(z)$ (FR arm, final)")
        axes[0].plot(x, g["q_target"].median().reindex(x), color="#0b6e4f",
                     lw=1.4, ls="--", label=r"$q_T\propto e^{-\beta A_T}$")
        if "p_ref" in sub:
            axes[0].plot(x, g["p_ref"].median().reindex(x), color="#222222",
                         lw=1.0, ls=":", label=r"$p^{\rm phys}_{\rm ref}(z)$")
        axes[0].set_xlabel("$z$"); axes[0].set_ylabel("density")
        axes[0].set_title("What FR is pulling towards", loc="left", fontsize=10)
        axes[0].legend(fontsize=7.5, frameon=False)
        axes[0].grid(alpha=0.15, lw=0.5)
    if pulses is not None and not pulses.empty:
        for m, style in ARM_STYLE.items():
            s = pulses[pulses["method"] == m]
            if s.empty:
                continue
            g = s.groupby("t")["ess_anc_after"]
            t = np.asarray(sorted(s["t"].unique()))
            axes[1].plot(t, g.median().reindex(t), lw=1.5,
                         color=style["color"], label=style["label"])
        axes[1].set_xlabel("physical time $t$")
        axes[1].set_ylabel(r"ancestral ESS / $K$")
        axes[1].set_ylim(0, 1.02)
        axes[1].set_title("Genealogy spent (diagnostic, not an objective)",
                          loc="left", fontsize=10)
        axes[1].legend(fontsize=7.5, frameon=False)
        axes[1].grid(alpha=0.15, lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig4_mechanism.{ext}"), dpi=200)
    plt.close(fig)

    print(f"[plot] wrote figures under {os.path.relpath(out_dir)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
