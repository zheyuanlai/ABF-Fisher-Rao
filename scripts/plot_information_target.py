#!/usr/bin/env python3
"""Publication figure for the oracle information-target Stage-A campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from publication_style import (
    FigureStyle,
    PALETTE,
    add_panel_labels,
    apply_publication_style,
    save_figure,
)

BASE = "abf_only"
ARM = "abf_fr_information_oracle"
STYLES = {
    BASE: dict(color=PALETTE["black"], ls="-", label="Plain ABF"),
    ARM: dict(
        color=PALETTE["vermillion"],
        ls="--",
        label="Oracle information target, 3 FR-BD pulses",
    ),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--verdict", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def _load_one(root: Path, suffix: str) -> pd.DataFrame:
    hits = sorted(path for path in root.glob(f"*_{suffix}.csv") if "__" not in path.name)
    if len(hits) != 1:
        raise SystemExit(f"expected one merged *_{suffix}.csv; got {hits}")
    return pd.read_csv(hits[0])


def _validate(long_df: pd.DataFrame, pulses: pd.DataFrame) -> None:
    required = {
        "method",
        "seed",
        "t",
        "l2_F_R12",
        "l2_Fprime_R12",
    }
    missing = required - set(long_df.columns)
    if missing:
        raise SystemExit(f"runs_long is missing columns: {sorted(missing)}")
    if set(long_df.method.unique()) != {BASE, ARM}:
        raise SystemExit(f"unexpected methods: {sorted(long_df.method.unique())}")
    counts = long_df.groupby("method").seed.nunique()
    if counts.get(BASE, 0) != counts.get(ARM, 0):
        raise SystemExit(f"unmatched seed counts: {counts.to_dict()}")
    if not {"method", "t", "step"}.issubset(pulses.columns):
        raise SystemExit("fr_pulses lacks method/t/step columns")


def _curve(ax, df: pd.DataFrame, column: str, method: str) -> None:
    grouped = df[df.method == method].groupby("t")[column]
    times = np.asarray(sorted(df.t.unique()), dtype=float)
    median = grouped.median().reindex(times).to_numpy(float)
    lower = grouped.quantile(0.25).reindex(times).to_numpy(float)
    upper = grouped.quantile(0.75).reindex(times).to_numpy(float)
    if not np.isfinite(np.r_[median, lower, upper]).all() or np.any(lower <= 0):
        raise SystemExit(f"invalid positive finite convergence values in {column}/{method}")
    style = STYLES[method]
    ax.fill_between(times, lower, upper, color=style["color"], alpha=0.14, lw=0)
    ax.plot(times, median, color=style["color"], ls=style["ls"], label=style["label"])


def _pulse_marks(ax, pulse_times: list[float]) -> None:
    for index, pulse_time in enumerate(pulse_times, start=1):
        ax.axvline(pulse_time, color=PALETTE["vermillion"], lw=0.8, ls=":", alpha=0.8)
        ax.annotate(
            str(index),
            xy=(pulse_time, 0.98 - 0.075 * (index - 1)),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=6.8,
            color=PALETTE["vermillion"],
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.stage_root)
    frozen = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    verdict = json.loads(Path(args.verdict).read_text(encoding="utf-8"))
    long_df = _load_one(root, "runs_long")
    pulses = _load_one(root, "fr_pulses")
    _validate(long_df, pulses)

    scope = frozen["primary_scope"]
    if scope != "R12":
        raise SystemExit(f"this preregistered figure expects R12 scope, got {scope!r}")
    pulse_times = sorted(pulses.loc[pulses.method == ARM, "t"].unique().astype(float))
    if len(pulse_times) != 3:
        raise SystemExit(f"expected exactly three unique pulse times, got {pulse_times}")

    style = apply_publication_style(FigureStyle(width_in=6.9, height_in=5.2, font_size=8.2))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(style.width_in, style.height_in),
        constrained_layout=True,
    )

    ax = axes[0, 0]
    for method in (BASE, ARM):
        _curve(ax, long_df, "l2_F_R12", method)
    for epsilon in frozen["thresholds"][scope]["F"]:
        ax.axhline(epsilon, color=PALETTE["blue"], lw=0.7, ls=(0, (3, 2)))
    _pulse_marks(ax, pulse_times)
    ax.set_yscale("log")
    ax.set_xlabel("Physical time $t$")
    ax.set_ylabel(r"$e_F(t)$ (R12)")
    ax.set_title("Free-energy convergence", loc="left")
    ax.legend(loc="upper right", title="Median; band = IQR ($n=32$)")
    ax.grid(axis="y", alpha=0.16, lw=0.5)

    ax = axes[0, 1]
    for method in (BASE, ARM):
        _curve(ax, long_df, "l2_Fprime_R12", method)
    for epsilon in frozen["thresholds"][scope]["Fprime"]:
        ax.axhline(epsilon, color=PALETTE["blue"], lw=0.7, ls=(0, (3, 2)))
    _pulse_marks(ax, pulse_times)
    ax.set_yscale("log")
    ax.set_xlabel("Physical time $t$")
    ax.set_ylabel(r"$e_{F'}(t)$ (R12)")
    ax.set_title("Mean-force convergence", loc="left")
    ax.grid(axis="y", alpha=0.16, lw=0.5)

    ax = axes[1, 0]
    rows = [row for row in verdict["threshold_speedups"] if row["metric"] == "F"]
    positions = np.arange(len(rows))[::-1]
    speedup = np.asarray([row["speedup"] for row in rows], dtype=float)
    ci_lo = np.asarray([row["ci_lo"] for row in rows], dtype=float)
    ci_hi = np.asarray([row["ci_hi"] for row in rows], dtype=float)
    ax.errorbar(
        speedup,
        positions,
        xerr=[speedup - ci_lo, ci_hi - speedup],
        fmt="o",
        color=PALETTE["vermillion"],
        capsize=2.5,
        lw=1.2,
    )
    ax.axvline(1.0, color=PALETTE["black"], lw=0.9, label="No speedup")
    ax.axvline(1.15, color=PALETTE["gray"], lw=0.8, ls="--", label="Preregistered target")
    ax.set_yticks(positions, [rf"$\epsilon_{{F,{row['threshold_index']}}}$" for row in rows])
    ax.set_xlabel(r"Restricted speedup $S_\epsilon^{(T)}$")
    ax.set_title("F-threshold speedups (paired 95% CI)", loc="left")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.16, lw=0.5)

    ax = axes[1, 1]
    final_time = float(long_df.t.max())
    final_rows = long_df[np.isclose(long_df.t, final_time)]
    paired = final_rows.pivot(index="seed", columns="method", values="l2_F_R12").dropna()
    x = paired[BASE].to_numpy(float)
    y = paired[ARM].to_numpy(float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise SystemExit("endpoint errors must be positive for logarithmic paired plot")
    lower = 0.9 * min(x.min(), y.min())
    upper = 1.1 * max(x.max(), y.max())
    ax.scatter(x, y, s=16, alpha=0.62, color=PALETTE["vermillion"], linewidths=0)
    ax.plot([lower, upper], [lower, upper], color=PALETTE["black"], lw=0.9, label="Equal endpoint error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Plain ABF $e_F(T)$")
    ax.set_ylabel(r"Information target $e_F(T)$")
    ax.set_title("Matched endpoint errors", loc="left")
    ax.text(
        0.04,
        0.96,
        rf"median ratio = {verdict['final_error_ratio']:.3f}",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.16, lw=0.5)

    add_panel_labels(axes.ravel(), x=-0.14, y=1.03)
    fig.suptitle(
        f"Oracle information-target FR-BD diagnostic: {verdict['verdict']}",
        x=0.01,
        ha="left",
        fontsize=9.2,
    )
    png, pdf = save_figure(
        fig,
        Path(args.out_dir) / "fig_information_target_stage_a",
        dpi=400,
        tight=False,
    )
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
