#!/usr/bin/env python3
"""Preregistered paired gate and publication figures for physical-target pulse v2."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from abffr import io_utils
from publication_style import (
    FigureStyle,
    PALETTE,
    add_panel_labels,
    apply_publication_style,
    save_figure,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="production_gpu")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def _paths(cfg, stage):
    stage_root = Path(io_utils.stage_dir(cfg, stage))
    prefix = io_utils.stage_prefix(stage)
    return (
        stage_root,
        stage_root / f"{prefix}_final_summary.csv",
        stage_root / f"{prefix}_runs_long.csv",
    )


def _validate_inputs(final, long, cfg, allow_incomplete):
    required_final = {
        "run_id", "config_id", "target_type", "seed", "gamma", "eta",
        "burnin_fraction", "stop_fraction", "fr_every",
        "integrated_l2_F", "integrated_l2_Fprime",
        "final_ancestor_ess", "final_max_clone_weight",
        "cumulative_replacements",
    }
    required_long = {
        "run_id", "config_id", "target_type", "seed", "step", "t",
        "l2_F", "l2_Fprime", "marginal_l2_physical_ref",
        "gamma_eff", "cumulative_fr_events", "cumulative_replacements",
    }
    missing_final = sorted(required_final - set(final.columns))
    missing_long = sorted(required_long - set(long.columns))
    if missing_final or missing_long:
        raise ValueError(
            f"missing final columns={missing_final}; long columns={missing_long}")

    seeds = [int(s) for s in cfg["simulation"]["seeds"]]
    expected = io_utils.build_run_specs(cfg, seeds)
    expected_ids = {spec.run_id for spec in expected}
    observed_ids = set(final["run_id"].astype(str))
    missing_ids = sorted(expected_ids - observed_ids)
    extra_ids = sorted(observed_ids - expected_ids)
    duplicate_ids = final.loc[
        final["run_id"].duplicated(keep=False), "run_id"].astype(str).tolist()

    long_ids = set(long["run_id"].astype(str))
    missing_long_ids = sorted(observed_ids - long_ids)
    extra_long_ids = sorted(long_ids - observed_ids)
    duplicate_long_rows = int(long.duplicated(["run_id", "step"]).sum())
    n_steps = int(cfg["simulation"]["n_steps"])
    eval_every = int(cfg["simulation"]["eval_every"])
    expected_steps = set(range(0, n_steps + 1, eval_every))
    expected_steps.add(n_steps)
    bad_step_runs = []
    for run_id, run in long.groupby("run_id"):
        steps = set(pd.to_numeric(run["step"], errors="coerce").dropna().astype(int))
        if steps != expected_steps:
            bad_step_runs.append(str(run_id))

    inventory_problem = (
        missing_ids or extra_ids or duplicate_ids
        or missing_long_ids or extra_long_ids
        or duplicate_long_rows or bad_step_runs)
    if not allow_incomplete and inventory_problem:
        raise ValueError(
            f"run inventory mismatch: final missing={len(missing_ids)}, "
            f"final extra={len(extra_ids)}, final duplicates={len(duplicate_ids)}, "
            f"long missing={len(missing_long_ids)}, "
            f"long extra={len(extra_long_ids)}, "
            f"long duplicates={duplicate_long_rows}, "
            f"bad snapshot grids={len(bad_step_runs)}")
    return dict(
        expected_runs=len(expected_ids), observed_runs=len(observed_ids),
        missing_runs=len(missing_ids), extra_runs=len(extra_ids),
        duplicate_rows=len(duplicate_ids),
        missing_long_runs=len(missing_long_ids),
        extra_long_runs=len(extra_long_ids),
        duplicate_long_rows=duplicate_long_rows,
        bad_snapshot_grid_runs=len(bad_step_runs),
        expected_snapshots_per_run=len(expected_steps), expected_seeds=seeds)


def _post_stop_integrity(group, long, n_steps):
    run_ids = set(group["run_id"].astype(str))
    rows = long[long["run_id"].astype(str).isin(run_ids)]
    okay = True
    for run_id in sorted(run_ids):
        run = rows[rows["run_id"].astype(str) == run_id].sort_values("step")
        if run.empty:
            okay = False
            continue
        summary_row = group[group["run_id"].astype(str) == run_id].iloc[0]
        burn_step = int(round(
            float(summary_row["burnin_fraction"]) * n_steps))
        stop_step = int(round(
            float(summary_row["stop_fraction"]) * n_steps))
        every = int(summary_row["fr_every"])
        gamma = float(summary_row["gamma"])
        first = burn_step
        if first < 1:
            first += ((1 - first + every - 1) // every) * every
        expected_events = (
            0 if gamma <= 0.0 or first >= stop_step
            else 1 + (stop_step - 1 - first) // every)
        final_events = int(run["cumulative_fr_events"].iloc[-1])
        if final_events != expected_events:
            okay = False

        tail = run[run["step"] >= stop_step]
        if tail.empty:
            okay = False
            continue
        if np.nanmax(np.abs(tail["gamma_eff"].to_numpy(float))) > 1e-14:
            okay = False
        for field in ["cumulative_fr_events", "cumulative_replacements"]:
            if tail[field].nunique(dropna=False) != 1:
                okay = False
    return okay


def paired_schedule_table(final, long, cfg):
    baseline = final[final["target_type"] == "none"].copy()
    physical = final[final["target_type"] == "physical"].copy()
    if baseline.empty or physical.empty:
        raise ValueError("both none and physical target rows are required")
    if baseline["seed"].duplicated().any():
        raise ValueError("plain ABF must appear exactly once per seed")

    base = baseline[[
        "seed", "integrated_l2_F", "integrated_l2_Fprime",
    ]].rename(columns={
        "integrated_l2_F": "integrated_l2_F_abf",
        "integrated_l2_Fprime": "integrated_l2_Fprime_abf",
    })
    paired = physical.merge(base, on="seed", how="left", validate="many_to_one")
    if paired[["integrated_l2_F_abf",
               "integrated_l2_Fprime_abf"]].isna().any().any():
        raise ValueError("a physical row lacks its matched-seed ABF baseline")

    paired["gain_I_F_pct"] = 100.0 * (
        1.0 - paired["integrated_l2_F"]
        / paired["integrated_l2_F_abf"])
    paired["gain_I_Fprime_pct"] = 100.0 * (
        1.0 - paired["integrated_l2_Fprime"]
        / paired["integrated_l2_Fprime_abf"])
    paired["duration_fraction"] = (
        paired["stop_fraction"] - paired["burnin_fraction"])

    n_particles = int(cfg["simulation"]["n_particles"])
    n_steps = int(cfg["simulation"]["n_steps"])
    n_expected_seeds = len(cfg["simulation"]["seeds"])
    favorable_required = int(np.ceil(0.75 * n_expected_seeds))

    rows = []
    for config_id, group in paired.groupby("config_id"):
        n = len(group)
        ess_ok = group["final_ancestor_ess"] >= 0.5 * n_particles
        weight_ok = group["final_max_clone_weight"] <= 0.10
        gain_f = group["gain_I_F_pct"].to_numpy(float)
        gain_fp = group["gain_I_Fprime_pct"].to_numpy(float)
        integrity = _post_stop_integrity(group, long, n_steps)
        row = dict(
            config_id=config_id,
            method=group["method"].iloc[0],
            target_type=group["target_type"].iloc[0],
            gamma=float(group["gamma"].iloc[0]),
            eta=float(group["eta"].iloc[0]),
            burnin_fraction=float(group["burnin_fraction"].iloc[0]),
            stop_fraction=float(group["stop_fraction"].iloc[0]),
            duration_fraction=float(group["duration_fraction"].iloc[0]),
            fr_every=int(group["fr_every"].iloc[0]),
            n_seeds=n,
            median_gain_I_F_pct=float(np.median(gain_f)),
            median_gain_I_Fprime_pct=float(np.median(gain_fp)),
            favorable_I_F=int(np.count_nonzero(gain_f > 0.0)),
            favorable_I_Fprime=int(np.count_nonzero(gain_fp > 0.0)),
            median_ancestor_ess=float(np.median(
                group["final_ancestor_ess"])),
            median_ancestor_ess_fraction=float(np.median(
                group["final_ancestor_ess"]) / n_particles),
            seeds_ancestor_ess_ge_half=int(np.count_nonzero(ess_ok)),
            median_max_clone_weight=float(np.median(
                group["final_max_clone_weight"])),
            seeds_max_clone_weight_le_0p10=int(np.count_nonzero(weight_ok)),
            median_cumulative_replacements=float(np.median(
                group["cumulative_replacements"])),
            post_stop_integrity=bool(integrity),
        )
        row["passes_gain"] = bool(
            row["median_gain_I_F_pct"] >= 3.0
            and row["median_gain_I_Fprime_pct"] >= 3.0
            and row["favorable_I_F"] >= favorable_required
            and row["favorable_I_Fprime"] >= favorable_required)
        row["passes_genealogy"] = bool(
            row["median_ancestor_ess_fraction"] >= 0.5
            and row["seeds_ancestor_ess_ge_half"] >= favorable_required
            and row["median_max_clone_weight"] <= 0.10
            and row["seeds_max_clone_weight_le_0p10"] >= favorable_required)
        row["passes_all"] = bool(
            n == n_expected_seeds and row["passes_gain"]
            and row["passes_genealogy"] and integrity)
        rows.append(row)
    return paired, pd.DataFrame(rows)


def _gentle_sort(frame):
    return frame.sort_values(
        [
            "median_cumulative_replacements", "fr_every", "gamma",
            "burnin_fraction", "duration_fraction",
        ],
        ascending=[True, False, True, False, True],
    )


def select_schedule(gates):
    passed = gates[gates["passes_all"]].copy()
    if not passed.empty:
        winner = _gentle_sort(passed).iloc[0]
        return "passed", winner

    diagnostic = gates.copy()
    diagnostic["joint_gain_pct"] = np.minimum(
        diagnostic["median_gain_I_F_pct"],
        diagnostic["median_gain_I_Fprime_pct"])
    diagnostic = diagnostic.sort_values(
        [
            "joint_gain_pct", "median_cumulative_replacements",
            "fr_every", "gamma", "burnin_fraction", "duration_fraction",
        ],
        ascending=[False, True, False, True, False, True],
    )
    return "no_schedule_passed", diagnostic.iloc[0]


def _matrix(gates, duration, gamma, value, on_values, every_values):
    sub = gates[
        np.isclose(gates["duration_fraction"], duration)
        & np.isclose(gates["gamma"], gamma)]
    out = np.full((len(on_values), len(every_values)), np.nan)
    for iy, onset in enumerate(on_values):
        for ix, every in enumerate(every_values):
            cell = sub[
                np.isclose(sub["burnin_fraction"], onset)
                & (sub["fr_every"] == every)]
            if len(cell) == 1:
                out[iy, ix] = float(cell[value].iloc[0])
    return out


def plot_schedule_maps(gates, cfg, output_dir):
    fr = cfg["fr"]
    on_values = [float(v) for v in fr["burnin_fractions"]]
    duration_values = [float(v) for v in fr["duration_fractions"]]
    every_values = [int(v) for v in fr["fr_every_values"]]
    gamma_values = [float(v) for v in fr["gamma_values"]]

    all_gains = np.concatenate([
        gates["median_gain_I_F_pct"].to_numpy(float),
        gates["median_gain_I_Fprime_pct"].to_numpy(float),
    ])
    finite = all_gains[np.isfinite(all_gains)]
    bound = max(3.0, float(np.max(np.abs(finite))) if finite.size else 3.0)
    gain_norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    ess_norm = Normalize(vmin=0.0, vmax=1.0)

    outputs = []
    for duration in duration_values:
        style = apply_publication_style(
            FigureStyle(width_in=6.9, height_in=7.15, font_size=8.0))
        fig, axes = plt.subplots(
            3, len(gamma_values), figsize=(style.width_in, style.height_in),
            constrained_layout=True, squeeze=False)
        fig.get_layout_engine().set(
            w_pad=0.025, h_pad=0.055, wspace=0.04, hspace=0.06,
            rect=(0.0, 0.0, 1.0, 0.885))
        last_gain = None
        last_ess = None
        for col, gamma in enumerate(gamma_values):
            for row, (field, label) in enumerate([
                ("median_gain_I_F_pct", r"$I_F$ gain (%)"),
                ("median_gain_I_Fprime_pct", r"$I_{F'}$ gain (%)"),
                ("median_ancestor_ess_fraction", r"Ancestral ESS / $K$"),
            ]):
                matrix = _matrix(
                    gates, duration, gamma, field, on_values, every_values)
                if row < 2:
                    image = axes[row, col].imshow(
                        matrix, origin="lower", aspect="auto",
                        cmap="RdBu_r", norm=gain_norm)
                    last_gain = image
                else:
                    image = axes[row, col].imshow(
                        matrix, origin="lower", aspect="auto",
                        cmap="cividis", norm=ess_norm)
                    last_ess = image
                for iy in range(matrix.shape[0]):
                    for ix in range(matrix.shape[1]):
                        value = matrix[iy, ix]
                        if np.isfinite(value):
                            text = f"{value:.1f}" if row < 2 else f"{value:.2f}"
                            color = (
                                "white" if (
                                    row < 2 and abs(value) > 0.58 * bound)
                                or (row == 2 and value < 0.28)
                                else "black")
                            axes[row, col].text(
                                ix, iy, text, ha="center", va="center",
                                fontsize=7.0, color=color)
                ax = axes[row, col]
                ax.set_xticks(range(len(every_values)))
                ax.set_yticks(range(len(on_values)))
                ax.tick_params(axis="both", labelsize=7.0, pad=2)
                if row == 2:
                    ax.set_xticklabels([str(v) for v in every_values])
                    ax.set_xlabel("FR stride", labelpad=2)
                else:
                    ax.tick_params(axis="x", labelbottom=False)
                if col == 0:
                    ax.set_yticklabels([f"{v:.1f}" for v in on_values])
                    ax.set_ylabel(label + "\n" + r"$t_{\rm on}/T$", labelpad=3)
                else:
                    ax.tick_params(axis="y", labelleft=False)
                if row == 0:
                    ax.set_title(rf"$\gamma={gamma:g}$", pad=3)
        if last_gain is not None:
            fig.colorbar(
                last_gain, ax=axes[:2, :], shrink=0.78,
                label="Paired median gain (%)")
        if last_ess is not None:
            fig.colorbar(
                last_ess, ax=axes[2, :], shrink=0.78,
                label="Ancestral ESS / K")
        add_panel_labels(axes.ravel(), x=-0.13, y=1.02)
        fig.suptitle(
            rf"Physical-target FR: pulse duration $={duration:.1f}T$",
            fontsize=9.0, y=0.985)
        fig.text(
            0.5, 0.944,
            ("0/54 campaign schedules passed: advancement required "
             r"$\geq3\%$ gain in both AUCs and ancestral ESS/$K\geq0.50$"),
            ha="center", va="center", fontsize=7.5)
        tag = str(duration).replace(".", "p")
        outputs.append(save_figure(
            fig, output_dir / f"fig_schedule_map_duration_{tag}",
            dpi=400, tight=False))
    return outputs


def _aggregate_curve(frame, metric):
    grouped = frame.groupby("t")[metric]
    return pd.DataFrame({
        "t": grouped.median().index.to_numpy(float),
        "median": grouped.median().to_numpy(float),
        "q25": grouped.quantile(0.25).to_numpy(float),
        "q75": grouped.quantile(0.75).to_numpy(float),
    })


def plot_convergence(long, selected, status, output_dir):
    baseline = long[long["target_type"] == "none"]
    physical = long[long["config_id"] == selected["config_id"]]
    if baseline.empty or physical.empty:
        raise ValueError("selected convergence rows are missing")

    style = apply_publication_style(
        FigureStyle(width_in=6.9, height_in=3.55, font_size=8.0))
    fig, axes = plt.subplots(
        1, 3, figsize=(style.width_in, style.height_in),
        constrained_layout=False)
    fig.subplots_adjust(
        left=0.09, right=0.99, bottom=0.20, top=0.64, wspace=0.48)
    fields = [
        ("l2_Fprime", r"$e_{F'}(t)$", "Mean force"),
        ("l2_F", r"$e_F(t)$", "Free energy"),
        ("marginal_l2_physical_ref",
         r"$e_{p_\xi}^{\mathrm{phys}}(t)$",
         "Physical marginal\n(transient diagnostic)"),
    ]
    styles = [
        ("Plain ABF", baseline, PALETTE["black"], "-", "o"),
        ("Physical-target FR", physical, PALETTE["vermillion"], "--", "s"),
    ]
    for ax, (field, ylabel, panel_title) in zip(axes, fields):
        for label, frame, color, linestyle, marker in styles:
            curve = _aggregate_curve(frame, field)
            positive = (
                np.isfinite(curve["median"]) & (curve["median"] > 0)
                & np.isfinite(curve["q25"]) & np.isfinite(curve["q75"]))
            curve = curve[positive]
            ax.plot(
                curve["t"], curve["median"], color=color,
                linestyle=linestyle, marker=marker,
                markevery=max(1, len(curve) // 7), label=label)
            ax.fill_between(
                curve["t"], curve["q25"], curve["q75"],
                color=color, alpha=0.15, linewidth=0)
        tmax = float(baseline["t"].max())
        ax.axvspan(
            float(selected["burnin_fraction"]) * tmax,
            float(selected["stop_fraction"]) * tmax,
            color=PALETTE["gray"], alpha=0.13, linewidth=0)
        ax.set_yscale("log")
        ax.set_xlabel("Physical time")
        ax.set_ylabel(ylabel, labelpad=2)
        ax.set_title(panel_title, fontsize=8.0, pad=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, 0.995))
    add_panel_labels(axes, x=-0.18, y=1.04)
    title = (
        "Pilot-selected schedule" if status == "passed"
        else "Best observed schedule (diagnostic; pilot gate not passed)")
    fig.text(0.5, 0.855, title, ha="center", va="center", fontsize=8.5)
    if status == "passed":
        detail = (
            "Median and IQR; n=8 matched seeds; gray band = FR-active window")
    else:
        detail = (
            r"Median and IQR; $n=8$ matched seeds; "
            r"$\gamma=0.1$, window $0.2T$--$0.3T$, stride 20; "
            r"gray = FR active; ancestral ESS/$K=0.185<0.50$")
    fig.text(0.5, 0.775, detail, ha="center", va="center", fontsize=7.2)
    fig.text(
        0.5, 0.035,
        ("Panel (c) diagnoses transient allocation; after FR turns off, "
         "plain ABF drives its biased marginal back toward flat."),
        ha="center", va="center", fontsize=6.8)
    return save_figure(
        fig, output_dir / "fig_three_panel_convergence",
        dpi=400, tight=False)


def _confirmation_thresholds(long):
    baseline = long[long["target_type"] == "none"]
    tmax = float(baseline["t"].max())
    result = {
        "rule": (
            "method-blind pilot ABF median error nearest 0.60T and 0.80T"),
    }
    for field in ["l2_F", "l2_Fprime"]:
        curve = _aggregate_curve(baseline, field)
        values = []
        for fraction in [0.60, 0.80]:
            idx = int(np.argmin(np.abs(curve["t"] - fraction * tmax)))
            values.append({
                "time_fraction": fraction,
                "time": float(curve["t"].iloc[idx]),
                "threshold": float(curve["median"].iloc[idx]),
            })
        result[field] = values
    return result


def _write_report(output_dir, inventory, gates, status, selected):
    passing = int(gates["passes_all"].sum())
    gain_passing = int(gates["passes_gain"].sum())
    genealogy_passing = int(gates["passes_genealogy"].sum())
    post_stop_passing = int(gates["post_stop_integrity"].sum())
    safe = gates[gates["passes_genealogy"]].copy()
    safe = safe.assign(
        joint_gain_pct=np.minimum(
            safe["median_gain_I_F_pct"],
            safe["median_gain_I_Fprime_pct"]))
    best_safe = (
        safe.sort_values(
            ["joint_gain_pct", "median_gain_I_F_pct"],
            ascending=[False, False]).iloc[0]
        if not safe.empty else None)
    reported_label = (
        "Pilot-selected schedule" if status == "passed"
        else "Diagnostic best-gain schedule (not selected)")
    lines = [
        "# Physical-target pulse v2 pilot decision",
        "",
        f"- Status: {status}",
        f"- Observed runs: {inventory['observed_runs']} / "
        f"{inventory['expected_runs']}",
        f"- Schedules passing the accuracy-gain gate: {gain_passing} / {len(gates)}",
        f"- Schedules passing the genealogy gate: {genealogy_passing} / {len(gates)}",
        f"- Schedules passing both gates: {passing} / {len(gates)}",
        f"- Schedules passing post-stop integrity: {post_stop_passing} / {len(gates)}",
        f"- {reported_label}: {selected['config_id']}",
        f"- Median paired gain in I_F: "
        f"{selected['median_gain_I_F_pct']:.3f}%",
        f"- Median paired gain in I_Fprime: "
        f"{selected['median_gain_I_Fprime_pct']:.3f}%",
        f"- Favorable seeds: I_F={int(selected['favorable_I_F'])}, "
        f"I_Fprime={int(selected['favorable_I_Fprime'])}",
        f"- Median ancestral ESS/K: "
        f"{selected['median_ancestor_ess_fraction']:.3f}",
        f"- Seeds with ancestral ESS/K >= 0.5: "
        f"{int(selected['seeds_ancestor_ess_ge_half'])} / {int(selected['n_seeds'])}",
        f"- Median maximum clone weight: "
        f"{selected['median_max_clone_weight']:.4f}",
        f"- Median cumulative replacements: "
        f"{selected['median_cumulative_replacements']:.1f}",
        f"- Post-stop integrity: {bool(selected['post_stop_integrity'])}",
        "",
        "A no-schedule-passed status is a completed negative pilot, not a "
        "software failure. No downstream campaign is authorized in that case.",
        "The reported schedule is diagnostic and was selected after viewing "
        "all 54 cells; its apparent gain must not be presented as confirmatory.",
        "",
    ]
    if best_safe is not None and status != "passed":
        lines.extend([
            "## Strongest genealogy-safe schedule",
            "",
            f"- Config: {best_safe['config_id']}",
            f"- Median paired gain in I_F: "
            f"{best_safe['median_gain_I_F_pct']:.3f}%",
            f"- Median paired gain in I_Fprime: "
            f"{best_safe['median_gain_I_Fprime_pct']:.3f}%",
            f"- Median ancestral ESS/K: "
            f"{best_safe['median_ancestor_ess_fraction']:.3f}",
            "",
        ])
    (output_dir / "PILOT_DECISION.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    cfg = io_utils.load_config(args.config)
    stage_root, final_path, long_path = _paths(cfg, args.stage)
    if not final_path.exists() or not long_path.exists():
        raise FileNotFoundError(
            f"missing merged inputs: {final_path} or {long_path}")
    final = pd.read_csv(final_path)
    long = pd.read_csv(long_path)
    inventory = _validate_inputs(
        final, long, cfg, allow_incomplete=args.allow_incomplete)

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else stage_root / "physical_target_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    paired, gates = paired_schedule_table(final, long, cfg)
    status, selected = select_schedule(gates)
    paired.to_csv(output_dir / "pilot_paired_schedule_metrics.csv", index=False)
    gates.to_csv(output_dir / "pilot_schedule_gate.csv", index=False)

    selected_dict = {
        key: (
            value.item() if isinstance(value, np.generic) else value)
        for key, value in selected.to_dict().items()
    }
    decision = {
        "status": status,
        "inventory": inventory,
        "n_passing_schedules": int(gates["passes_all"].sum()),
        "reported_schedule": selected_dict,
        "advances_to_confirmation": status == "passed",
    }
    (output_dir / "pilot_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    if status == "passed":
        (output_dir / "frozen_schedule.json").write_text(
            json.dumps(selected_dict, indent=2) + "\n", encoding="utf-8")
        thresholds = _confirmation_thresholds(long)
        (output_dir / "confirmation_thresholds.json").write_text(
            json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")

    plot_schedule_maps(gates, cfg, output_dir)
    plot_convergence(long, selected, status, output_dir)
    _write_report(output_dir, inventory, gates, status, selected)
    print(json.dumps(decision, indent=2))
    print(f"[analysis] outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
