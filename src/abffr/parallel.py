"""Batching, checkpoint/resume and CSV assembly for the GPU backend.

This module turns a list of :class:`abffr.io_utils.RunSpec` into batched calls
to :func:`abffr.simulation_torch.run_batch` and writes outputs whose schema is a
*superset* of the CPU runner's (so the existing plotting/table scripts and the
study-spec columns are both satisfied).  Metric and conditional-diagnostic rows
are produced with the *same* :mod:`abffr.metrics` / :mod:`abffr.diagnostics`
code the CPU runner uses.

Resumption
----------
Each run has a unique ``run_id``.  A finished run drops a marker
``<stage>/completed/<safe_run_id>.done``; a crashed run drops
``<stage>/failed/<safe_run_id>.json`` with the error and config.  On restart,
only run ids with post-flush completion markers are skipped unless forced. Partial CSV rows without a marker are rerun and later deduplicated. Each process writes tag-suffixed CSVs
(``<prefix>_<kind>__<tag>.csv``); :func:`merge_stage_csvs` concatenates the tags
into the canonical ``<prefix>_<kind>.csv`` (deduped by ``run_id``) for the
plotting/table scripts.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import time
import traceback
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from . import (diagnostics, io_utils, metrics, reference, simulation_torch,
               torch_utils as tu)
from .io_utils import RunSpec

CSV_KINDS = ("runs_long", "final_summary", "profiles", "fr_events",
             "conditional_diagnostics", "v3_events", "fr_pulses")

# Columns that uniquely identify a row of each CSV kind, used to de-duplicate
# when merging tag-suffixed shard CSVs (a profile has one row per (run, x); a
# long row one per (run, snapshot); etc.).  ``keep="last"`` lets a forced re-run
# override an earlier row.
DEDUP_SUBSET = {
    "runs_long": ["run_id", "step"],
    "final_summary": ["run_id"],
    "profiles": ["run_id", "x"],
    "fr_events": ["run_id", "step"],
    # Amendment 4c: one row per FR opportunity per run.
    "v3_events": ["run_id", "step"],
    # clean-v2: one row per FR *pulse* per run (Gates D/E and the genealogy
    # appendix read this and nothing else).
    "fr_pulses": ["run_id", "step"],
    # conditional_diagnostics rows carry only (method, target_type, seed) as
    # identity (not the full config), so several configs share a key -- like the
    # CPU runner we keep them all (plotting averages over them); no dedup.
}


# --------------------------------------------------------------------------- #
# Stage setup (device/dtype/reference/eval), shared by all GPU scripts
# --------------------------------------------------------------------------- #
def prepare_stage(cfg: Dict, stage: str, require_csv: bool = True, logger=print
                  ) -> Dict:
    """Resolve device/dtype, the stage output dir, the reference (gated) and the
    evaluation window for ``stage``."""
    device = tu.resolve_device(cfg.get("device"))
    dtype = tu.resolve_dtype(cfg.get("dtype"))
    stage_root = io_utils.stage_dir(cfg, stage)
    prefix = io_utils.stage_prefix(stage)
    x_grid, ref, csv_path = reference.load_reference_for_run(
        cfg, require_csv=require_csv, logger=logger)
    ev = metrics.EvalConfig.from_domain(cfg["domain"])
    return dict(device=device, dtype=dtype, stage_root=stage_root, prefix=prefix,
                x_grid=x_grid, ref=ref, ev=ev, csv_path=csv_path)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
def batch_key(spec: RunSpec):
    """Batch runs with a common target, smoothing, cadence, onset and stop."""
    return (
        spec.target_type, float(spec.eta), int(spec.fr_every),
        float(spec.burnin_fraction), float(spec.stop_fraction))


def build_batches(specs: List[RunSpec], batch_size: int) -> List[List[RunSpec]]:
    groups: Dict = defaultdict(list)
    for s in specs:
        groups[batch_key(s)].append(s)
    batches: List[List[RunSpec]] = []
    for key in sorted(groups):
        gs = sorted(groups[key], key=lambda s: (float(s.gamma), int(s.seed)))
        for i in range(0, len(gs), batch_size):
            batches.append(gs[i:i + batch_size])
    return batches


# --------------------------------------------------------------------------- #
# Resume bookkeeping
# --------------------------------------------------------------------------- #
def _safe_run_id(run_id: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]", "_", run_id)
    return f"{s[:120]}__{hashlib.sha1(run_id.encode()).hexdigest()[:8]}"


def completed_dir(stage_root: str) -> str:
    return io_utils.ensure_dir(os.path.join(stage_root, "completed"))


def failed_dir(stage_root: str) -> str:
    return io_utils.ensure_dir(os.path.join(stage_root, "failed"))


def load_completed(stage_root: str, prefix: str) -> set:
    """Return run ids with post-flush completion markers.

    CSV rows alone are not proof of completion because a crash can occur between
    atomically replacing different CSV kinds. A marker is written only after
    every kind has flushed, so marker-only resume is conservative and complete.
    """
    del prefix
    done = set()
    cdir = os.path.join(stage_root, "completed")
    if os.path.isdir(cdir):
        for path in glob.glob(os.path.join(cdir, "*.done")):
            try:
                with open(path) as fh:
                    done.add(json.load(fh)["run_id"])
            except Exception:
                pass
    return done


def _write_marker(stage_root: str, spec: RunSpec, summary: Dict) -> None:
    path = os.path.join(completed_dir(stage_root), _safe_run_id(spec.run_id) + ".done")
    io_utils.save_json(path, dict(run_id=spec.run_id, config_id=spec.config_id,
                                  final_l2_F=summary.get("final_l2_F"),
                                  final_l2_Fprime=summary.get("final_l2_Fprime")))


def _write_failure(stage_root: str, spec: RunSpec, err: str) -> None:
    path = os.path.join(failed_dir(stage_root), _safe_run_id(spec.run_id) + ".json")
    io_utils.save_json(path, dict(run_id=spec.run_id, **spec.to_row(), error=err))


# --------------------------------------------------------------------------- #
# Row assembly (reuses metrics.py / diagnostics.py, like the CPU runner)
# --------------------------------------------------------------------------- #
def _v3_event_rows(meta: Dict, events, b: int):
    """Amendment 4c per-opportunity diagnostics for batch row ``b``.

    Computed in the engine since Amendment 4c but previously discarded here, so
    theta, the consistency residue and the carrier error never reached disk.
    Retention is recomputed from the two ancestral ESS values rather than stored
    separately, so the CSV cannot carry a G_t that disagrees with them.
    """
    rows = []
    for e in (events or []):
        before = float(e["ess_anc_before"][b])
        after = float(e["ess_anc_after"][b])
        rows.append({
            **meta, "step": int(e["step"]),
            "theta": float(e["theta"][b]),
            "ess_w_frac": float(e["ess_w"][b]),
            "dtau": float(e["dtau"][b]),
            "q90_absS": float(e["q90"][b]),
            "p_event_mean": float(e["pev_mean"][b]),
            "p_event_max": float(e["pev_max"][b]),
            "replacements": int(e["repl"][b]),
            "kl_before": float(e["kl_before"][b]),
            "kl_after": float(e["kl_after"][b]),
            "ess_anc_before": before, "ess_anc_after": after,
            "wmax_before": float(e["wmax_before"][b]),
            "wmax_after": float(e["wmax_after"][b]),
            "retention": after / before if before > 0 else float("nan"),
            "carrier_err": float(e["carrier_err"][b]),
            "dcons": float(e["dcons"][b]),
        })
    return rows


def _clean_event_rows(meta: Dict, events, b: int):
    """One row per clean-v2 FR pulse for batch row ``b``.

    Everything a gate needs to check that the operator ran as written lives
    here: the reaction time actually used, the event probabilities it produced
    (uncapped, so ``p_event_max`` may sit near 1 and that is faithful rather
    than a fault), the raw score spread, and the genealogy paid for the move.
    ``logp_floored_fraction`` must be 0: a non-zero value means ``log phat`` hit
    its NaN backstop, which at particle positions should be impossible.
    """
    rows = []
    for e in (events or []):
        # The engine records one entry per batched opportunity, and a batch may
        # mix an FR run with its own gamma = 0 control.  dtau = gamma L dt is
        # zero exactly for the inactive rows, so filtering on it keeps
        # ``fr_pulses`` a table of pulses that actually happened -- counting its
        # rows must not over-report the FR dose.
        if float(e["dtau"][b]) <= 0.0:
            continue
        before = float(e["ess_anc_before"][b])
        after = float(e["ess_anc_after"][b])
        row = {
            **meta, "step": int(e["step"]), "t": float(e["t"]),
            "dtau": float(e["dtau"][b]),
            "p_event_mean": float(e["p_event_mean"][b]),
            "p_event_max": float(e["p_event_max"][b]),
            "replacements": int(e["repl"][b]),
            "event_fraction": float(e["event_fraction"][b]),
            "s_min": float(e["s_min"][b]), "s_max": float(e["s_max"][b]),
            "s_absmax": float(e["s_absmax"][b]),
            "s_span": float(e["s_max"][b]) - float(e["s_min"][b]),
            "kl_before": float(e["kl_before"][b]),
            "kl_after": float(e["kl_after"][b]),
            "ess_anc_before": before, "ess_anc_after": after,
            "wmax_before": float(e["wmax_before"][b]),
            "wmax_after": float(e["wmax_after"][b]),
            "retention": after / before if before > 0 else float("nan"),
            "logp_floored_fraction": float(e["logp_floored_fraction"][b]),
        }
        if "information_risk" in e:
            row.update(
                information_risk=float(e["information_risk"][b]),
                uniform_information_risk=float(
                    e["uniform_information_risk"][b]),
                information_risk_ratio=float(
                    e["information_risk_ratio"][b]),
            )
            masses = np.asarray(e["q_cell_masses"][b], dtype=float)
            variances = np.asarray(e["force_variance_cells"][b], dtype=float)
            for j, value in enumerate(masses):
                row[f"q_cell_{j:02d}"] = float(value)
            for j, value in enumerate(variances):
                row[f"force_var_cell_{j:02d}"] = float(value)
        for lab in ("q01", "q10", "q50", "q90", "q99"):
            key = f"s_{lab}"
            if key in e:
                row[key] = float(e[key][b])
        rows.append(row)
    return rows


def _rows_for_run(spec: RunSpec, diag: Dict, cfg: Dict, x_grid, ref, ev,
                  runtime_seconds: float, conditional: str,
                  v3_events=None, batch_index: int = 0, clean_events=None):
    """Assemble durable CSV payloads for one completed GPU run."""
    meta = spec.to_row()
    h = float(cfg["abf"]["h"])
    ramp_fraction = float(cfg.get("fr", {}).get("ramp_fraction", 0.1))
    beta = float(cfg["simulation"]["beta"])
    p_ref = ref.get("p_ref")

    ts = metrics.time_series_metrics(
        diag, x_grid, ref["F_ref"], ref["Fprime_ref"], ev, p_ref=p_ref,
        scopes=metrics.evaluation_scopes(
            x_grid, ref["F_ref"], float(cfg["simulation"]["beta"]), ev))
    long_rows, fr_rows = [], []
    for k, row in enumerate(ts):
        rf = metrics.region_fractions(diag["X_snap"][k], ev)
        long_rows.append({
            **meta, **row, "h": h, "ramp_fraction": ramp_fraction,
            "x_l2_to_target": row["marginal_l2_target"],
            "x_l2_to_uniform": row["marginal_l2_uniform"],
            "x_l2_to_physical_ref": row["marginal_l2_physical_ref"],
            "fr_score_std": row["score_std"],
            "fr_score_max": row["score_max"],
            "left_frac": rf["frac_left"],
            "barrier_frac": rf["frac_barrier"],
            "right_frac": rf["frac_right"],
        })
        fr_row = {
            **meta,
            "step": row["step"], "t": row["t"],
            "gamma_eff": row["gamma_eff"],
            "fr_applied": row["fr_applied"],
            "fr_event_fraction": row["fr_event_fraction"],
            "fr_event_fraction_max": row["fr_event_fraction_max"],
            "fr_events_total": row["fr_events_total"],
            "num_deaths": int(diag["fr_events_total"][k]),
            "num_births": int(diag["fr_events_total"][k]),
            "event_fraction": row["fr_event_fraction"],
            "score_mean": row["score_mean"],
            "score_std": row["score_std"],
            "score_min": row["score_min"],
            "score_max": row["score_max"],
            "target_l2": (
                float(diag["target_l2"][k])
                if "target_l2" in diag else float("nan")),
            "n_unique_ancestors": row["n_unique_ancestors"],
        }
        for key in [
            "ancestor_ess", "max_clone_multiplicity", "max_clone_weight",
            "cumulative_fr_events", "cumulative_replacements",
            "score_clipped_fraction",
            *[f"score_raw_{q}" for q in ("q01", "q10", "q50", "q90", "q99")],
            *[f"score_applied_{q}" for q in ("q01", "q10", "q50", "q90", "q99")],
        ]:
            if key in row:
                fr_row[key] = row[key]
        fr_rows.append(fr_row)

    summary = metrics.final_summary(
        diag, x_grid, ref["F_ref"], ref["Fprime_ref"], ev, p_ref=p_ref)
    final_row = {
        **meta, **summary, "runtime_seconds": float(runtime_seconds),
        "total_barrier_crossings": int(summary["barrier_crossings"]),
    }

    Fp = diag["Fprime_hat"][-1]
    F = diag["F_hat"][-1]
    p = diag["p_hat_grid"][-1]
    q = diag["q_target_grid"][-1]
    profile_rows = []
    for j in range(len(x_grid)):
        profile_rows.append({
            **meta, "x": float(x_grid[j]),
            "F_ref": float(ref["F_ref"][j]),
            "Fprime_ref": float(ref["Fprime_ref"][j]),
            "p_ref": (
                float(p_ref[j]) if p_ref is not None else float("nan")),
            "Fprime_hat": float(Fp[j]), "F_hat": float(F[j]),
            "p_hat": float(p[j]), "q_target": float(q[j]),
            "p_hat_x": float(p[j]), "target_x": float(q[j]),
        })

    cmeta = dict(
        method=spec.method, target_type=spec.target_type, seed=int(spec.seed))
    snap_idx = None if conditional == "final" else list(range(len(diag["steps"])))
    cond_rows = diagnostics.conditional_diagnostics(
        diag, cmeta, beta, cfg["domain"], snapshot_indices=snap_idx)

    return dict(
        runs_long=long_rows, final_summary=[final_row],
        profiles=profile_rows, fr_events=fr_rows,
        conditional_diagnostics=cond_rows,
        v3_events=_v3_event_rows(meta, v3_events, batch_index),
        fr_pulses=_clean_event_rows(meta, clean_events, batch_index)), summary


# --------------------------------------------------------------------------- #
# Tag-suffixed CSV writer (process-local, append + flush)
# --------------------------------------------------------------------------- #
class _CsvBuffer:
    def __init__(self, stage_root: str, prefix: str, tag: str,
                 seed_from_disk: bool = True):
        self.stage_root, self.prefix, self.tag = stage_root, prefix, tag
        self.rows = {k: [] for k in CSV_KINDS}
        # Seed from this tag's prior partial output (resume within a process).
        # Skipped on --force so a forced re-run overwrites cleanly rather than
        # appending duplicate rows (conditional_diagnostics is not deduped).
        if not seed_from_disk:
            return
        for k in CSV_KINDS:
            path = self._path(k)
            if os.path.exists(path):
                try:
                    self.rows[k] = pd.read_csv(path).to_dict("records")
                except Exception:
                    self.rows[k] = []

    def _path(self, kind: str) -> str:
        return os.path.join(self.stage_root, f"{self.prefix}_{kind}__{self.tag}.csv")

    def extend(self, payload: Dict[str, list]) -> None:
        for k, rows in payload.items():
            self.rows[k].extend(rows)

    def flush(self) -> None:
        """Atomically replace each process-local CSV after a complete write."""
        for k in CSV_KINDS:
            if not self.rows[k]:
                continue
            path = self._path(k)
            tmp = path + ".tmp"
            pd.DataFrame(self.rows[k]).to_csv(tmp, index=False)
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def run_specs(
    specs: List[RunSpec],
    *,
    cfg: Dict,
    stage_root: str,
    prefix: str,
    x_grid: np.ndarray,
    ref: Dict,
    ev,
    device: torch.device,
    dtype: torch.dtype,
    estimator: str = "binned_smooth",
    batch_size: int = 16,
    base_seed: int = 0,
    tag: str = "main",
    resume: bool = True,
    force: bool = False,
    conditional: str = "final",
    logger=print,
) -> Dict:
    """Run all ``specs`` on one ``device`` with checkpoint/resume.

    Returns a small summary dict (counts, runtimes, NaN count).
    """
    completed = load_completed(stage_root, prefix) if (resume and not force) else set()
    todo = [s for s in specs if s.run_id not in completed]
    skipped = len(specs) - len(todo)
    batches = build_batches(todo, batch_size)
    logger(f"[parallel] device={device} estimator={estimator} "
           f"runs={len(specs)} todo={len(todo)} skipped(resume)={skipped} "
           f"batches={len(batches)} batch_size<={batch_size} tag={tag}")

    buf = _CsvBuffer(stage_root, prefix, tag, seed_from_disk=not force)
    n_done, n_failed, n_nan = 0, 0, 0
    t_start = time.time()

    for bi, batch in enumerate(batches):
        try:
            res = simulation_torch.run_batch(
                batch, cfg=cfg, x_grid=x_grid, F_ref=ref["F_ref"],
                Fprime_ref=ref["Fprime_ref"],
                force_var_ref=ref.get("force_var_ref"),
                ev=ev, device=device, dtype=dtype,
                estimator=estimator, base_seed=base_seed)
        except Exception as exc:  # whole-batch failure
            err = f"{exc}\n{traceback.format_exc()}"
            for spec in batch:
                _write_failure(stage_root, spec, err)
            n_failed += len(batch)
            logger(f"[parallel] BATCH {bi+1}/{len(batches)} FAILED "
                   f"({len(batch)} runs): {exc}")
            continue

        per_run_runtime = res.runtime_seconds / max(len(batch), 1)
        pending = []
        for b, (spec, diag) in enumerate(zip(batch, res.diags)):
            try:
                payload, summary = _rows_for_run(
                    spec, diag, cfg, x_grid, ref, ev,
                    per_run_runtime, conditional,
                    v3_events=getattr(res, "v3_events", None), batch_index=b,
                    clean_events=getattr(res, "clean_events", None))
                buf.extend(payload)
                pending.append((spec, summary))
            except Exception as exc:
                _write_failure(
                    stage_root, spec, f"{exc}\n{traceback.format_exc()}")
                n_failed += 1

        # Result rows must reach an atomically replaced CSV before any marker
        # can make a run eligible for resume skipping.
        buf.flush()
        for spec, summary in pending:
            _write_marker(stage_root, spec, summary)
            n_done += 1
            n_nan += int(bool(summary.get("any_nan")))
        logger(f"[parallel] batch {bi+1}/{len(batches)} done "
               f"({len(batch)} runs, {res.runtime_seconds:.1f}s, "
               f"{res.runtime_seconds/max(len(batch),1):.2f}s/run); "
               f"cumulative done={n_done} failed={n_failed} nan={n_nan}")

    buf.flush()
    return dict(n_runs=len(specs), n_done=n_done, n_failed=n_failed,
                n_skipped=skipped, n_nan=n_nan, wall_seconds=time.time() - t_start,
                tag=tag)


# --------------------------------------------------------------------------- #
# Merge + config aggregation (canonical CSVs for plotting / tables)
# --------------------------------------------------------------------------- #
def merge_stage_csvs(stage_root: str, prefix: str, logger=print) -> None:
    """Concatenate all tag-suffixed CSVs into canonical ``<prefix>_<kind>.csv``."""
    for kind in CSV_KINDS:
        parts = sorted(glob.glob(os.path.join(stage_root, f"{prefix}_{kind}__*.csv")))
        if not parts:
            continue
        frames = []
        for p in parts:
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                pass
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        subset = [c for c in DEDUP_SUBSET.get(kind, []) if c in df.columns]
        if subset:
            df = df.drop_duplicates(subset=subset, keep="last")
        out = os.path.join(stage_root, f"{prefix}_{kind}.csv")
        df.to_csv(out, index=False)
        logger(f"[parallel] merged {len(parts)} part(s) -> "
               f"{os.path.relpath(out)} ({len(df)} rows)")


def _iqr(s):
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    return float(np.percentile(s, 75) - np.percentile(s, 25)) if s.size else float("nan")


def summarize_configs(final_df: pd.DataFrame) -> pd.DataFrame:
    """Per-config median/IQR over matched seeds."""
    rows = []
    keys = [
        "config_id", "method", "target_type", "gamma", "eta",
        "burnin_fraction", "stop_fraction", "fr_every",
    ]
    aggregate = [
        "final_l2_F", "final_l2_Fprime", "integrated_l2_F",
        "integrated_l2_Fprime", "final_marginal_l2_uniform",
        "final_marginal_l2_target", "final_marginal_l2_physical_ref",
        "integrated_marginal_l2_physical_ref", "final_deltaF_error",
        "integrated_deltaF_error", "final_barrier_height_error",
        "integrated_barrier_height_error", "mean_fr_event_fraction",
        "max_fr_event_fraction", "barrier_crossings", "frac_left",
        "frac_barrier", "frac_right", "final_ancestor_ess",
        "final_max_clone_multiplicity", "final_max_clone_weight",
        "cumulative_fr_events", "cumulative_replacements",
    ]
    for _, group in final_df.groupby("config_id"):
        row = {
            key: (
                group[key].iloc[0] if key in group
                else (1.0 if key == "stop_fraction" else float("nan")))
            for key in keys
        }
        row["n_seeds"] = len(group)
        for col in aggregate:
            if col in group:
                row[f"median_{col}"] = float(np.nanmedian(group[col]))
                row[f"iqr_{col}"] = _iqr(group[col])
        row["frac_nan"] = (
            float(np.mean(group["any_nan"])) if "any_nan" in group else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_configs(config_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Compatibility selector; preregistered pilot gating is done separately."""
    cap = float(cfg.get("fr", {}).get("max_event_fraction", 0.10))
    rows = []
    targets = [
        "none", "estimated", "uniform", "oracle", "self",
        "physical", "physical_oracle",
    ]
    for target in targets:
        sub = config_df[config_df["target_type"] == target].copy()
        if sub.empty:
            continue
        passed = (
            (sub.get("frac_nan", 0.0) == 0.0)
            & (sub.get("median_mean_fr_event_fraction", 0.0) <= cap + 1e-9)
            & (sub.get("median_max_fr_event_fraction", 0.0)
               <= 1.5 * cap + 1e-9)
        )
        sub["passed_safety"] = passed
        pool = sub[passed] if passed.any() else sub
        pool = pool.sort_values(
            ["median_integrated_l2_F", "median_final_l2_F"]).reset_index(
                drop=True)
        for rank, (_, row) in enumerate(pool.iterrows(), start=1):
            rows.append(dict(
                rank_within_target=rank, selected=(rank == 1),
                method=row["method"], target_type=row["target_type"],
                gamma=float(row["gamma"]), eta=float(row["eta"]),
                burnin_fraction=float(row["burnin_fraction"]),
                stop_fraction=float(row.get("stop_fraction", 1.0)),
                fr_every=int(row["fr_every"]),
                median_integrated_l2_F=float(
                    row["median_integrated_l2_F"]),
                median_final_l2_F=float(row["median_final_l2_F"]),
                median_final_l2_Fprime=float(
                    row["median_final_l2_Fprime"]),
                median_mean_fr_event_fraction=float(
                    row["median_mean_fr_event_fraction"]),
                passed_safety=bool(row["passed_safety"]),
                n_seeds=int(row["n_seeds"]),
            ))
    return pd.DataFrame(rows)


def write_config_summaries(stage_root: str, prefix: str, cfg: Dict, logger=print):
    """Write the config summary and, when authorized, a generic best-config file."""
    path = os.path.join(stage_root, f"{prefix}_final_summary.csv")
    if not os.path.exists(path):
        logger(f"[parallel] {os.path.relpath(path)} missing; skipping config summary.")
        return
    final_df = pd.read_csv(path)
    if final_df.empty:
        return
    config_df = summarize_configs(final_df)
    config_df.to_csv(os.path.join(stage_root, f"{prefix}_config_summary.csv"), index=False)
    write_generic = bool(
        cfg.get("selection", {}).get("write_generic_best", True))
    if not write_generic:
        logger(
            f"[parallel] wrote {prefix}_config_summary.csv ({len(config_df)} configs); "
            "generic best-config selection disabled (preregistered gate required)")
        return
    best_df = select_best_configs(config_df, cfg)
    best_df.to_csv(os.path.join(stage_root, "best_configs.csv"), index=False)
    logger(f"[parallel] wrote {prefix}_config_summary.csv ({len(config_df)} configs) "
           f"and best_configs.csv "
           f"({int(best_df['selected'].sum()) if len(best_df) else 0} selected)")
