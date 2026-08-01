#!/usr/bin/env python3
"""mFR mechanism audit -- STEP 1: cross-system paired-seed target-ablation audit.

Builds ONE master paired-seed table over every cell anywhere in this repository
that has an ABF arm plus at least two of {fr_estimated, fr_uniform, fr_oracle}
at MATCHED configuration, and quantifies the scope of the analytic identity

    F_target = B + eps   =>   q_est(x) ~ exp(-beta * eps(x))

using the stored q_target / p_hat profiles.

READ-ONLY.  This script never writes outside
``results/mfr_mechanism_audit/target_ablation/`` and never touches src/,
results/*/raw/, or report/.

------------------------------------------------------------------------------
PREDECLARED PRACTICAL EQUIVALENCE MARGIN  (fixed before any number was looked
at; applied uniformly to every cell; never tuned)
------------------------------------------------------------------------------
For a cell, let  m = 0.05 * mean_over_ABF_seeds( final L2(F) ).
Two methods are PRACTICALLY EQUIVALENT in that cell iff the 95% paired
bootstrap CI of their per-seed difference in final L2(F) is *contained* in
[-m, +m].  Verdicts (a partition):

    EQUIVALENT     CI subset of [-m, +m]
    DIFFERENT      CI disjoint from [-m, +m]           (materially different)
    UNDERPOWERED   CI straddles an equivalence boundary -- neither conclusion
                   is licensed.  This necessarily includes every case with
                   CI width > 2m.

Cell classification from the three final-L2(F) contrasts:
    A   est ~= uni ~= oracle              (target shape empirically unimportant)
    B   est ~= uni, oracle differs        (deployable target collapsed toward
                                           uniform; informative steering has
                                           real headroom)
    C   all three materially differ       (target direction is first-order)
    U   any contrast UNDERPOWERED, or the oracle arm is missing
The raw verdict triple is always reported alongside, so no pattern is smoothed.

------------------------------------------------------------------------------
DATA-SHAPE HAZARDS HANDLED EXPLICITLY (see excluded_and_filters.csv)
------------------------------------------------------------------------------
* Summary CSVs mix stages and mix FR *rates* under one ``method`` label.
  Every loader states the stage / rate / name filter it applied.
* ``fr_active`` (alkanes, fr_rate=0.2) and ``fr_r*`` tuning arms carry
  ``method == 'fr_estimated'`` but are NOT the matched-rate arm; excluded.
* WCA production ``main`` carries fr_estimated at 4 rates; only fr_rate == 0.10
  matches fr_uniform / fr_oracle.
* One .npz per JOB can bundle many seeds (alkanes: ``per_seed`` JSON string,
  profile arrays shaped (n_seeds, n_grid)).  Handled.
* ``report/main.aux`` is stale and is never read.
* eb_abffr_core / edb_abffr_core store ``q_target`` as the ESTIMATED target
  snapshot for EVERY method row (``_finalize``: ``q_final = fr_target_from(
  F_target, Bbias, ...)``).  The oracle / uniform arms' actual targets are NOT
  recoverable from those files.  Flagged in the shape table.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "mfr_mechanism_audit", "target_ablation")

EQUIV_MARGIN_FRAC = 0.05          # PREDECLARED.  Do not tune.
N_BOOT = 20000
BOOT_SEED = 20260721
PRIMARY_METRIC = "final_l2_F"     # the metric the margin/verdict is defined on

ROLES = ("abf", "estimated", "uniform", "oracle")

# rows appended by every loader
_SEED_ROWS: List[dict] = []
_CELLS: List[dict] = []
_EXCL: List[dict] = []


def note_filter(system: str, source: str, action: str, detail: str) -> None:
    _EXCL.append(dict(system=system, source=source, action=action, detail=detail))


def add_cell(cell_id: str, system: str, stage: str, cell_label: str,
             config: str, source: str, pair_key_desc: str = "seed",
             notes: str = "") -> None:
    _CELLS.append(dict(cell_id=cell_id, system=system, stage=stage,
                       cell_label=cell_label, config=config, source=source,
                       pair_key=pair_key_desc, notes=notes))


def add_rows(cell_id: str, role: str, df: pd.DataFrame, cols: Dict[str, str],
             pair_key: Optional[pd.Series] = None,
             ess_kind: str = "", ev_kind: str = "") -> None:
    """Append canonical per-seed rows.  ``cols`` maps canonical->source column."""
    if len(df) == 0:
        return
    out = pd.DataFrame(dict(
        cell_id=cell_id, role=role,
        seed=df[cols["seed"]].to_numpy(),
    ))
    out["pair_key"] = (pair_key.to_numpy() if pair_key is not None
                       else df[cols["seed"]].astype(str).to_numpy())
    for canon in ("final_l2_F", "int_l2_F", "final_l2_Fp",
                  "ancestor_ess", "event_fraction"):
        src = cols.get(canon)
        out[canon] = (pd.to_numeric(df[src], errors="coerce").to_numpy()
                      if src is not None and src in df.columns else np.nan)
    out["ancestor_ess_kind"] = ess_kind
    out["event_fraction_kind"] = ev_kind
    _SEED_ROWS.append(out)


# ============================================================================
# 1. metastability toy  --  results/two_dim_xi_x
# ============================================================================
def load_two_dim() -> None:
    src = os.path.join(REPO, "results/two_dim_xi_x/production_gpu/"
                             "production_gpu_final_summary.csv")
    df = pd.read_csv(src)
    note_filter("two_dim_xi_x", src, "stage filter",
                "only the shard-merged production_gpu final summary is used; "
                "results/two_dim_xi_x/{validation,benchmark} carry no FR "
                "target arms.")
    note_filter("two_dim_xi_x", src, "headline warning",
                "production_gpu/table_main_results.csv reports each FR method "
                "at its OWN best config (best-of-36 selection). This audit "
                "never uses that table; it pairs at MATCHED config.")
    mmap = {"abf_only": "abf", "abf_fr_estimated": "estimated",
            "abf_fr_uniform": "uniform", "abf_fr_oracle": "oracle"}
    df["role"] = df["method"].map(mmap)
    cols = dict(seed="seed", final_l2_F="final_l2_F", int_l2_F="integrated_l2_F",
                final_l2_Fp="final_l2_Fprime",
                ancestor_ess="n_unique_ancestors",
                event_fraction="mean_fr_event_fraction")
    abf = df[df.role == "abf"]
    key = ["gamma", "eta", "burnin_fraction", "fr_every"]
    fr = df[df.role != "abf"]

    # ---- per-config cells --------------------------------------------------
    for kv, g in fr.groupby(key):
        tag = "g%g_eta%g_bi%g_fe%d" % kv
        cid = f"twodim|prod|{tag}"
        add_cell(cid, "metastability_toy_2D", "production_gpu", tag,
                 "gamma=%g eta=%g burnin=%g fr_every=%d" % kv, src,
                 notes="ABF arm has a single config (gamma=0) and is reused by "
                       "all 36 FR configs; ancestor ESS not stored, "
                       "n_unique_ancestors used as proxy.")
        add_rows(cid, "abf", abf, cols, ess_kind="n_unique_ancestors(proxy)",
                 ev_kind="mean_fr_event_fraction")
        for role, gg in g.groupby("role"):
            add_rows(cid, role, gg, cols, ess_kind="n_unique_ancestors(proxy)",
                     ev_kind="mean_fr_event_fraction")

    # ---- pooled-over-configs cell (paired on config|seed) ------------------
    cid = "twodim|prod|POOLED_36cfg"
    add_cell(cid, "metastability_toy_2D", "production_gpu",
             "POOLED over 36 matched FR configs",
             "gamma in {.01,.02,.05,.1} x eta in {.075,.1,.15} x burnin in "
             "{0,.2,.4} x fr_every=5", src,
             pair_key_desc="config|seed",
             notes="Pooled paired comparison at MATCHED config (180 pairs). "
                   "Mixes gentle and aggressive gamma by design -- it is a "
                   "power-boosting pooled contrast, not a single physical cell. "
                   "The ABF arm (5 seeds, gamma=0) is reused 36x.")
    pk = fr[key].astype(str).agg("|".join, axis=1) + "|s" + fr["seed"].astype(str)
    for role, gg in fr.groupby("role"):
        add_rows(cid, role, gg, cols, pair_key=pk.loc[gg.index],
                 ess_kind="n_unique_ancestors(proxy)",
                 ev_kind="mean_fr_event_fraction")
    abf_rep = []
    for kv, g in fr.groupby(key):
        a = abf.copy()
        a["_pk"] = "|".join(str(v) for v in kv) + "|s" + a["seed"].astype(str)
        abf_rep.append(a)
    abf_rep = pd.concat(abf_rep, ignore_index=True)
    add_rows(cid, "abf", abf_rep, cols, pair_key=abf_rep["_pk"],
             ess_kind="n_unique_ancestors(proxy)",
             ev_kind="mean_fr_event_fraction")


# ============================================================================
# 2. entropic bottleneck  --  results/entropic_bottleneck
# ============================================================================
def load_entropic_bottleneck() -> None:
    src = os.path.join(REPO, "results/entropic_bottleneck/summaries/summary.csv")
    df = pd.read_csv(src)
    rmap = {"abf": "abf", "fr_estimated": "estimated",
            "fr_uniform": "uniform", "fr_oracle": "oracle"}
    df["role"] = df["method"].map(rmap)
    cols = dict(seed="seed", final_l2_F="final_l2_f", int_l2_F="int_l2_f",
                final_l2_Fp="final_l2_fp", ancestor_ess="final_ess",
                event_fraction="repl_fraction")
    for stage, g in df.groupby("stage"):
        roles = set(g.role.dropna())
        if len(roles & {"estimated", "uniform", "oracle"}) < 2:
            note_filter("entropic_bottleneck", src, "cell EXCLUDED",
                        f"stage={stage}: only methods {sorted(set(g.method))} "
                        "-- fewer than two of {estimated,uniform,oracle}.")
            continue
        cfg = ("beta=%g omega_in=%g gamma=%g N=%d n_steps=%d ema=%g" % (
            g.beta.iloc[0], g.omega_in.iloc[0], g.gamma.iloc[0],
            g.N.iloc[0], g.n_steps.iloc[0], g.target_ema_rate.iloc[0]))
        cid = f"eb|{stage}"
        add_cell(cid, "entropic_bottleneck", stage, stage, cfg, src,
                 notes="event_fraction = repl_fraction = (n_die+n_clone)/"
                       "(n_fr_apply*N).")
        for role, gg in g.groupby("role"):
            add_rows(cid, role, gg, cols, ess_kind="final_ess",
                     ev_kind="repl_fraction")


# ============================================================================
# 3. entropy-dominant bottleneck  --  main phi sweep
# ============================================================================
def load_edb_main() -> None:
    src = os.path.join(REPO, "results/entropy_dominant_bottleneck/"
                             "sweep_20260614_015145/raw_runs.csv")
    df = pd.read_csv(src)
    note_filter("entropy_dominant_bottleneck", src, "sweep filter",
                "sweep=='main' only.  sweep=='rate' (gamma in {1,5,15,30,50}) "
                "has ONLY abf + fr_estimated -> excluded (needs >=2 of "
                "{estimated,uniform,oracle}).")
    df = df[df.sweep == "main"]
    rmap = {"abf": "abf", "fr_estimated": "estimated",
            "fr_uniform": "uniform", "fr_oracle": "oracle"}
    df["role"] = df["method"].map(rmap)
    cols = dict(seed="seed", final_l2_F="final_l2_f", int_l2_F="int_l2_f",
                final_l2_Fp="final_l2_fp", ancestor_ess="final_ess",
                event_fraction="repl_fraction")
    for phi, g in df.groupby("phi"):
        cid = f"edb|main|phi{phi:g}"
        add_cell(cid, "entropy_dominant_bottleneck", "main_sweep",
                 f"phi={phi:g}",
                 "beta=%g m=%d B0=%g gamma=%g N=%d n_steps=%d ema=%g" % (
                     g.beta.iloc[0], g.m.iloc[0], g.B0.iloc[0], g.gamma.iloc[0],
                     g.N.iloc[0], g.n_steps.iloc[0], g.target_ema_rate.iloc[0]),
                 src)
        for role, gg in g.groupby("role"):
            add_rows(cid, role, gg, cols, ess_kind="final_ess",
                     ev_kind="repl_fraction")


# ============================================================================
# 4. entropy-dominant bottleneck  --  EMA-tuning rerun
# ============================================================================
def _edb_npz_rows(pattern: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        rows.append(dict(
            method=str(z["method"]), seed=int(z["seed"]),
            phi=float(z["cfg__phi"]),
            target_ema_rate=float(z["cfg__target_ema_rate"]),
            final_l2_f=float(z["final_l2_f"]), int_l2_f=float(z["int_l2_f"]),
            final_l2_fp=float(z["final_l2_fp"]), final_ess=float(z["final_ess"]),
            repl_fraction=float(z["repl_fraction"]), path=f))
    return pd.DataFrame(rows)


def load_edb_ema() -> None:
    base = os.path.join(REPO, "results/entropy_dominant_bottleneck_ema_tuning/"
                              "ema_sweep_20260615_175039")
    src_est = os.path.join(base, "ema_tuning_raw_scalars.csv")
    est = pd.read_csv(src_est)
    src_bl = os.path.join(base, "oracle_uniform_baseline/raw/main/*.npz")
    bl = _edb_npz_rows(src_bl)
    note_filter("entropy_dominant_bottleneck_ema_tuning", src_bl,
                "npz read (no per-seed CSV)",
                "matched_oracle_uniform_summary.csv holds medians only; the "
                "per-seed fr_uniform / fr_oracle scalars exist ONLY as raw "
                ".npz under oracle_uniform_baseline/raw/main.")
    note_filter("entropy_dominant_bottleneck_ema_tuning", base,
                "arm REUSE across cells",
                "fr_uniform and fr_oracle do not depend on target_ema_rate by "
                "construction (q_uni is flat; q_orc = exp(-beta(F_ref-B))), and "
                "were run once at ema=0.005. The SAME 20 uniform/oracle seeds "
                "are therefore reused in all 8 rate cells of a given phi -- the "
                "8 cells within a phi are NOT independent.")
    ab = _edb_npz_rows(os.path.join(base, "oracle_uniform_baseline/raw/main/abf__*.npz"))
    ab_csv = est[est.method == "abf"]
    # ABF is target-independent: verify the rate-dir ABF equals the baseline ABF
    chk = ab.merge(ab_csv, on=["phi", "seed"], suffixes=("_bl", "_csv"))
    maxdev = (float(np.nanmax(np.abs(chk["final_l2_f_bl"] - chk["final_l2_f_csv"])))
              if len(chk) else np.nan)
    note_filter("entropy_dominant_bottleneck_ema_tuning", base, "consistency check",
                f"ABF arm is identical across all 8 ema dirs and the "
                f"oracle/uniform baseline dir: max |dFinal L2(F)| = {maxdev:.3g}.")

    cols_csv = dict(seed="seed", final_l2_F="final_l2_f", int_l2_F="int_l2_f",
                    final_l2_Fp="final_l2_fp", ancestor_ess="final_ess",
                    event_fraction="repl_fraction")
    for (rate, phi), g in est.groupby(["target_ema_rate", "phi"]):
        cid = f"edbema|phi{phi:g}|ema{rate:g}"
        add_cell(cid, "entropy_dominant_bottleneck_ema_tuning", "ema_sweep",
                 f"phi={phi:g}, target_ema_rate={rate:g}",
                 "beta=4 m=2 B0=8 gamma=%g N=%g n_steps=%g" % (
                     g.gamma.iloc[0], g.N.iloc[0], g.n_steps.iloc[0]),
                 src_est + " + " + src_bl,
                 notes="fr_uniform/fr_oracle arms are the ema=0.005 baseline "
                       "runs (EMA-independent by construction) and are shared "
                       "across the 8 rate cells at this phi.")
        add_rows(cid, "abf", g[g.method == "abf"], cols_csv,
                 ess_kind="final_ess", ev_kind="repl_fraction")
        add_rows(cid, "estimated", g[g.method == "fr_estimated"], cols_csv,
                 ess_kind="final_ess", ev_kind="repl_fraction")
        for meth, role in (("fr_uniform", "uniform"), ("fr_oracle", "oracle")):
            sub = bl[(bl.method == meth) & (np.isclose(bl.phi, phi))]
            add_rows(cid, role, sub, cols_csv, ess_kind="final_ess",
                     ev_kind="repl_fraction")


# ============================================================================
# 5. slow-transverse ORPHAN sweep
# ============================================================================
def load_slow_transverse() -> None:
    src = os.path.join(REPO, "results/entropy_dominant_bottleneck_slow_transverse/"
                             "reanalysis/slow_transverse_runs_long.csv")
    df = pd.read_csv(src)
    note_filter("slow_transverse_ORPHAN", src, "PROVENANCE",
                "Generating code was NEVER committed (docs/"
                "ORPHAN_ARTIFACT_PROVENANCE.md). Method labels are inferred "
                "from filenames. UNVERIFIED -- do not cite.")
    note_filter("slow_transverse_ORPHAN", src, "arm EXCLUDED",
                "fr_estimated_condrefresh and fr_estimated_estrefresh are "
                "excluded from the target-ablation contrasts: they are not "
                "members of {estimated,uniform,oracle} and their mechanism is "
                "unverified (estrefresh was additionally built from a "
                "different code version -- missing none__F_moment).")
    note_filter("slow_transverse_ORPHAN", src, "pilot DUPLICATION",
                "pilot_20260616_225737 and pilot_20260617_001656 share "
                "bit-identical ABF runs (max |d final L2(F)| = 1.45e-07 over "
                "200 matched seeds) but their FR arms differ. Pilots are kept "
                "as SEPARATE cells and must not be pooled or double counted.")
    rmap = {"abf": "abf", "fr_estimated": "estimated",
            "fr_uniform": "uniform", "fr_oracle": "oracle"}
    df["role"] = df["method"].map(rmap)
    df = df[df.role.notna()]
    cols = dict(seed="seed", final_l2_F="final_l2_f", int_l2_F="int_l2_f",
                final_l2_Fp="final_l2_fp", ancestor_ess="final_ess",
                event_fraction="repl_fraction")
    for (pilot, phi, muy), g in df.groupby(["pilot", "phi", "muy"]):
        if len(set(g.role) & {"estimated", "uniform", "oracle"}) < 2:
            continue
        short = pilot.replace("pilot_", "")
        cid = f"slowtrans|{short}|phi{phi:g}|muy{muy:g}"
        add_cell(cid, "slow_transverse_ORPHAN", pilot,
                 f"phi={phi:g}, mu_y={muy:g}",
                 "beta=4 m=2 B0=8 gamma=15 N=512 n_steps=80000 dt=5e-4 "
                 "(T=40) ema=0.005; tau_y_well=%.3g" % g.tau_y_well.iloc[0], src,
                 notes="ORPHAN / UNVERIFIED provenance.")
        for role, gg in g.groupby("role"):
            add_rows(cid, role, gg, cols, ess_kind="final_ess",
                     ev_kind="repl_fraction")


# ============================================================================
# 6. WCA production (main)
# ============================================================================
def load_wca_production() -> None:
    src = os.path.join(REPO, "results/wca_production/summaries/summary.csv")
    df = pd.read_csv(src)
    note_filter("wca_production", src, "stage filter",
                "stage=='main' only. Stages difficulty_budget / "
                "difficulty_crowding / difficulty_replicas / failure / pilot / "
                "stage0_reproduce carry ONLY abf + fr_estimated -> excluded.")
    note_filter("wca_production", src, "RATE filter (critical)",
                "stage main carries fr_estimated at fr_rate in "
                "{0.05,0.10,0.20,0.50} under one method label. Only "
                "fr_rate==0.10 matches fr_uniform / fr_oracle; the other three "
                "rate arms (fr_est_gentle / _strong / _aggressive) are "
                "EXCLUDED. Pooling them would average gentle with aggressive.")
    g = df[(df.stage == "main")].copy()
    g = g[(g.method == "abf") | (np.isclose(g.fr_rate, 0.10))]
    rmap = {"abf": "abf", "fr_estimated": "estimated",
            "fr_uniform": "uniform", "fr_oracle": "oracle"}
    g["role"] = g["method"].map(rmap)
    # event fraction is not stored for this study -> derive it
    n_apply = (g.n_steps - g.fr_start_steps) / g.fr_every + 1.0
    g["ev_frac"] = g.total_replacement_events / (n_apply * g.n_replicas)
    cols = dict(seed="seed", final_l2_F="l2_f", int_l2_F="integrated_l2_f",
                final_l2_Fp="l2_fp", ancestor_ess="final_ancestor_ess",
                event_fraction="ev_frac")
    cid = "wcaprod|main"
    add_cell(cid, "wca_production", "main", "WCA dimer production main cell",
             "a=1.5 N=1024 n_steps=250000 fr_rate=0.10 fr_every=5 "
             "fr_start=20000 ema=0.005 max_event_fraction=0.02", src,
             notes="event_fraction DERIVED as total_replacement_events / "
                   "(n_applications * n_replicas); n_fr_applications is not "
                   "stored by wca_prod_v1. ABF ancestor ESS is not stored "
                   "(NaN) -- ABF never resamples, so ESS == n_replicas.")
    for role, gg in g.groupby("role"):
        add_rows(cid, role, gg, cols, ess_kind="final_ancestor_ess",
                 ev_kind="derived_from_total_events")


# ============================================================================
# 7. WCA phase diagram (production / pilot / smoke)
# ============================================================================
def load_wca_phase() -> None:
    for stage in ("production", "pilot", "smoke"):
        src = os.path.join(REPO, f"results/wca_phase_diagram/{stage}/"
                                 "summaries/phase_final_summary.csv")
        if not os.path.exists(src):
            continue
        df = pd.read_csv(src)
        rmap = {"abf": "abf", "fr_estimated": "estimated",
                "fr_uniform": "uniform", "fr_oracle": "oracle"}
        df["role"] = df["method"].map(rmap)
        cols = dict(seed="seed", final_l2_F="l2_f", int_l2_F="integrated_l2_f",
                    final_l2_Fp="l2_fp", ancestor_ess="final_ancestor_ess",
                    event_fraction="fr_event_fraction")
        if stage == "pilot":
            note_filter("wca_phase_diagram", src, "arm MISSING",
                        "pilot stage has NO fr_oracle arm (abf + fr_estimated "
                        "+ fr_uniform only) -> only the est-vs-uni contrast is "
                        "computable; cells classify as U.")
        for tag, g in df.groupby("physics_tag"):
            if len(set(g.role.dropna()) & {"estimated", "uniform", "oracle"}) < 2:
                continue
            cid = f"wcaphase|{stage}|{tag}"
            add_cell(cid, "wca_phase_diagram", stage, tag,
                     "beta=%g h=%g w=%g n_dim=%d M=%d a=%g N=%d n_steps=%d "
                     "fr_rate=%g" % (g.beta.iloc[0], g.h.iloc[0], g.w.iloc[0],
                                     g.n_dim.iloc[0], g.M.iloc[0], g.a.iloc[0],
                                     g.n_replicas.iloc[0], g.n_steps.iloc[0],
                                     float(np.nanmax(g.fr_rate))), src)
            for role, gg in g.groupby("role"):
                add_rows(cid, role, gg, cols, ess_kind="final_ancestor_ess",
                         ev_kind="fr_event_fraction")


# ============================================================================
# 8. WCA representative cells
# ============================================================================
def load_wca_representative() -> None:
    src = os.path.join(REPO, "results/wca_representative/summaries/"
                             "wca_representative_run_summary.csv")
    df = pd.read_csv(src)
    note_filter("wca_representative", src, "arm EXCLUDED",
                "fr_estimated_adaptive is present at 10 seeds per cell but is "
                "a different METHOD (online-gated adaptive rate), not a target "
                "variant; excluded from the three target contrasts.")
    rmap = {"abf": "abf", "fr_estimated": "estimated",
            "fr_uniform": "uniform", "fr_oracle": "oracle"}
    df["role"] = df["method"].map(rmap)
    df = df[df.role.notna()]
    cols = dict(seed="seed", final_l2_F="l2_f", int_l2_F="integrated_l2_f",
                final_l2_Fp="l2_fp", ancestor_ess="final_ancestor_ess",
                event_fraction="fr_event_fraction")
    for tag, g in df.groupby("physics_tag"):
        cid = f"wcarep|{tag}"
        add_cell(cid, "wca_representative", "representative", tag,
                 "beta=%g h=%g w=%g n_dim=%d M=%d a=%g N=%d n_steps=%d "
                 "fr_rate=%g" % (g.beta.iloc[0], g.h.iloc[0], g.w.iloc[0],
                                 g.n_dim.iloc[0], g.M.iloc[0], g.a.iloc[0],
                                 g.n_replicas.iloc[0], g.n_steps.iloc[0],
                                 float(np.nanmax(g.fr_rate))), src)
        for role, gg in g.groupby("role"):
            add_rows(cid, role, gg, cols, ess_kind="final_ancestor_ess",
                     ev_kind="fr_event_fraction")


# ============================================================================
# 9. alkanes (butane / pentane torsion)
# ============================================================================
def load_alkanes() -> None:
    for stage in ("production", "smoke"):
        src = os.path.join(REPO, f"results/alkanes/{stage}/summaries/"
                                 "alkanes_runs_long.csv")
        if not os.path.exists(src):
            continue
        df = pd.read_csv(src)
        if stage == "production":
            note_filter("alkanes", src, "NAME filter (critical)",
                        "pentane p1 cells carry TWO arms under "
                        "method=='fr_estimated': name=='fr_estimated' "
                        "(fr_rate=0.02, matched to uniform/oracle) and "
                        "name=='fr_active' (fr_rate=0.20). Only "
                        "name=='fr_estimated' is used. opes arms excluded.")
            note_filter("alkanes", src, "stage EXCLUDED",
                        "stage 'tuning' has only abf + fr_r0.02..0.40 + opes "
                        "(no uniform/oracle) -> excluded.")
        df = df[df["name"].isin(["abf", "fr_estimated", "fr_uniform",
                                 "fr_oracle"])]
        rmap = {"abf": "abf", "fr_estimated": "estimated",
                "fr_uniform": "uniform", "fr_oracle": "oracle"}
        df["role"] = df["name"].map(rmap)
        cols = dict(seed="seed", final_l2_F="final_l2_F",
                    int_l2_F="integrated_l2_F", final_l2_Fp="final_l2_Fp",
                    ancestor_ess="final_ancestor_ess",
                    event_fraction="fr_event_fraction")
        for (st, cell), g in df.groupby(["stage", "cell"]):
            if len(set(g.role) & {"estimated", "uniform", "oracle"}) < 2:
                continue
            cid = f"alk|{st}|{cell}"
            add_cell(cid, "alkanes_torsion", f"{stage}/{st}", cell,
                     "molecule=%s beta=%g sigma=%g init=%s N=%d n_steps=%d" % (
                         g.molecule.iloc[0], g.beta.iloc[0], g.sigma.iloc[0],
                         g.init_mode.iloc[0], g.n_replicas.iloc[0],
                         g.n_steps.iloc[0]), src,
                     notes="fr_rate=0.02 for all three FR arms (spec_json).")
            for role, gg in g.groupby("role"):
                add_rows(cid, role, gg, cols, ess_kind="final_ancestor_ess",
                         ev_kind="fr_event_fraction")


# ============================================================================
# 10. pentane R15 distance CV
# ============================================================================
def load_r15() -> None:
    src = os.path.join(REPO, "results/alkanes_cv_extension/r15_methods/"
                             "summaries/cv_runs_long.csv")
    df = pd.read_csv(src)
    note_filter("alkanes_cv_extension_R15", src, "stage + NAME filter",
                "stage=='production' only (tuning / runlength / opes_tuning "
                "have no uniform/oracle). Within it, name=='fr_active' "
                "(fr_rate=0.2) and opes are excluded; only the matched "
                "name=='fr_estimated' arm is used.")
    note_filter("alkanes_cv_extension_R15",
                "results/alkanes_cv_extension/2d_methods/summaries/cv_runs_long.csv",
                "cell EXCLUDED",
                "2-D torsion-torus cells have abf + fr_estimated + fr_active + "
                "opes only -- no uniform/oracle arm.")
    note_filter("alkanes_cv_extension_R15",
                "results/alkanes_cv_extension/{r14,r15,2d}/summaries/cv_runs_long.csv",
                "cell EXCLUDED", "screen/resgate stages are ABF-only.")
    df = df[(df.stage == "production") &
            df["name"].isin(["abf", "fr_estimated", "fr_uniform", "fr_oracle"])]
    rmap = {"abf": "abf", "fr_estimated": "estimated",
            "fr_uniform": "uniform", "fr_oracle": "oracle"}
    df["role"] = df["name"].map(rmap)
    cols = dict(seed="seed", final_l2_F="final_l2_F",
                int_l2_F="integrated_l2_F", final_l2_Fp="final_l2_Fp",
                ancestor_ess="final_ancestor_ess",
                event_fraction="fr_event_fraction")
    for cell, g in df.groupby("cell"):
        cid = f"r15|{cell}"
        add_cell(cid, "pentane_R15_distance_CV", "production", cell,
                 "molecule=pentane beta=%g N=%d n_steps=%d CV=R15 distance" % (
                     g.beta.iloc[0], g.n_replicas.iloc[0], g.n_steps.iloc[0]),
                 src)
        for role, gg in g.groupby("role"):
            add_rows(cid, role, gg, cols, ess_kind="final_ancestor_ess",
                     ev_kind="fr_event_fraction")


# ============================================================================
# statistics
# ============================================================================
def paired_boot(d: np.ndarray, rng: np.random.Generator):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = float(d.mean())
    med = float(np.median(d))
    if n < 2:
        return mean, med, np.nan, np.nan, n
    idx = rng.integers(0, n, size=(N_BOOT, n))
    bm = d[idx].mean(axis=1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return mean, med, float(lo), float(hi), n


def verdict(lo: float, hi: float, m: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(m)) or m <= 0:
        return "UNDETERMINED"
    if lo >= -m and hi <= m:
        return "EQUIVALENT"
    if lo > m or hi < -m:
        return "DIFFERENT"
    return "UNDERPOWERED"


def classify(v_eu: str, v_ou: str, v_oe: str, have_oracle: bool):
    vs = (v_eu, v_ou, v_oe)
    if not have_oracle:
        return "U", "no_oracle_arm"
    if any(v in ("UNDETERMINED", "UNDERPOWERED") for v in vs):
        return "U", "underpowered:" + "/".join(vs)
    if all(v == "EQUIVALENT" for v in vs):
        return "A", "all_equivalent"
    if all(v == "DIFFERENT" for v in vs):
        return "C", "all_different"
    if v_eu == "EQUIVALENT" and (v_ou == "DIFFERENT" or v_oe == "DIFFERENT"):
        return "B", "est~uni, oracle differs"
    return "C", "FALLBACK_C:" + "/".join(vs)


CONTRASTS = (("estimated", "uniform", "est_minus_uni"),
             ("oracle", "uniform", "orc_minus_uni"),
             ("oracle", "estimated", "orc_minus_est"))
METRICS = ("final_l2_F", "int_l2_F", "final_l2_Fp")


def analyse(seeds: pd.DataFrame, cells: pd.DataFrame):
    rng = np.random.default_rng(BOOT_SEED)
    cell_rows, contrast_rows = [], []
    for _, c in cells.iterrows():
        cid = c.cell_id
        s = seeds[seeds.cell_id == cid]
        wide = {r: g.set_index("pair_key") for r, g in s.groupby("role")}
        if "abf" not in wide:
            continue
        abf = wide["abf"]
        base = float(np.nanmean(abf[PRIMARY_METRIC]))
        margin = EQUIV_MARGIN_FRAC * base

        row = dict(c)
        row["abf_n_seeds"] = int(abf[PRIMARY_METRIC].notna().sum())
        row["abf_mean_final_l2_F"] = base
        row["abf_median_final_l2_F"] = float(np.nanmedian(abf[PRIMARY_METRIC]))
        row["equiv_margin_abs"] = margin
        row["equiv_margin_frac_of_abf"] = EQUIV_MARGIN_FRAC

        have = {}
        for role in ("estimated", "uniform", "oracle"):
            if role not in wide:
                row[f"{role}_n"] = 0
                continue
            g = wide[role]
            have[role] = g
            j = g.join(abf, how="inner", rsuffix="_abf")
            row[f"{role}_n"] = int(len(g))
            row[f"{role}_n_matched_vs_abf"] = int(len(j))
            for m in METRICS:
                row[f"{role}_{m}_mean"] = float(np.nanmean(g[m]))
                row[f"{role}_{m}_median"] = float(np.nanmedian(g[m]))
            row[f"{role}_ancestor_ess_median"] = float(np.nanmedian(g["ancestor_ess"]))
            row[f"{role}_event_fraction_median"] = float(np.nanmedian(g["event_fraction"]))
            d = (j[PRIMARY_METRIC] - j[PRIMARY_METRIC + "_abf"]).to_numpy(float)
            d = d[np.isfinite(d)]
            row[f"{role}_winrate_vs_abf"] = (float((d < 0).mean()) if d.size else np.nan)
            row[f"{role}_n_wins_vs_abf"] = int((d < 0).sum())
            mn, md, lo, hi, n = paired_boot(d, rng)
            row[f"{role}_minus_abf_mean"] = mn
            row[f"{role}_minus_abf_ci_lo"] = lo
            row[f"{role}_minus_abf_ci_hi"] = hi
            row[f"{role}_gain_pct_vs_abf_median"] = (
                float(np.median(-(j[PRIMARY_METRIC] - j[PRIMARY_METRIC + "_abf"])
                                / j[PRIMARY_METRIC + "_abf"] * 100.0)))
        row["abf_ancestor_ess_median"] = float(np.nanmedian(abf["ancestor_ess"]))
        row["abf_event_fraction_median"] = float(np.nanmedian(abf["event_fraction"]))
        row["ancestor_ess_kind"] = (s["ancestor_ess_kind"].dropna().iloc[0]
                                    if s["ancestor_ess_kind"].notna().any() else "")
        row["event_fraction_kind"] = (s["event_fraction_kind"].dropna().iloc[0]
                                      if s["event_fraction_kind"].notna().any() else "")

        verds = {}
        for a, b, name in CONTRASTS:
            if a not in have or b not in have:
                verds[name] = "UNDETERMINED"
                row[f"{name}_verdict"] = "ARM_MISSING"
                row[f"{name}_n_pairs"] = 0
                continue
            j = have[a].join(have[b], how="inner", rsuffix="_B")
            for m in METRICS:
                d = (j[m] - j[m + "_B"]).to_numpy(float)
                mn, md, lo, hi, n = paired_boot(d, rng)
                v = verdict(lo, hi, margin) if m == PRIMARY_METRIC else ""
                contrast_rows.append(dict(
                    cell_id=cid, system=c.system, stage=c.stage,
                    cell_label=c.cell_label, contrast=name, metric=m,
                    n_pairs=n, mean_diff=mn, median_diff=md,
                    ci95_lo=lo, ci95_hi=hi,
                    ci_width=(hi - lo) if np.isfinite(hi) and np.isfinite(lo) else np.nan,
                    abf_mean_final_l2_F=base, equiv_margin_abs=margin,
                    equiv_band_width=2 * margin,
                    ci_width_over_band=((hi - lo) / (2 * margin)
                                        if np.isfinite(hi) and np.isfinite(lo)
                                        and margin > 0 else np.nan),
                    ci_wider_than_band=(bool((hi - lo) > 2 * margin)
                                        if np.isfinite(hi) and np.isfinite(lo)
                                        and margin > 0 else None),
                    mean_diff_pct_of_abf=(100.0 * mn / base
                                          if np.isfinite(mn) and base > 0 else np.nan),
                    equivalence_verdict=v,
                    frac_pairs_A_lower=float((d[np.isfinite(d)] < 0).mean())
                    if np.isfinite(d).any() else np.nan))
                if m == PRIMARY_METRIC:
                    verds[name] = v
                    row[f"{name}_n_pairs"] = n
                    row[f"{name}_mean_diff"] = mn
                    row[f"{name}_ci_lo"] = lo
                    row[f"{name}_ci_hi"] = hi
                    row[f"{name}_pct_of_abf"] = (100.0 * mn / base if base > 0 else np.nan)
                    row[f"{name}_verdict"] = v

        have_oracle = "oracle" in have
        cls, note = classify(verds.get("est_minus_uni", "UNDETERMINED"),
                             verds.get("orc_minus_uni", "UNDETERMINED"),
                             verds.get("orc_minus_est", "UNDETERMINED"),
                             have_oracle)
        row["classification"] = cls
        row["classification_note"] = note
        row["verdict_triple"] = "/".join((verds.get("est_minus_uni", "NA"),
                                          verds.get("orc_minus_uni", "NA"),
                                          verds.get("orc_minus_est", "NA")))
        oe = row.get("orc_minus_est_mean_diff", np.nan)
        row["oracle_direction_vs_estimated"] = (
            "oracle_BETTER" if np.isfinite(oe) and oe < 0 else
            "oracle_WORSE" if np.isfinite(oe) and oe > 0 else "n/a")
        row["oracle_effect_pct_of_abf"] = row.get("orc_minus_est_pct_of_abf", np.nan)
        cell_rows.append(row)
    return pd.DataFrame(cell_rows), pd.DataFrame(contrast_rows)


# ============================================================================
# target-shape scope:  how far is q_target from uniform?
# ============================================================================
def shape_metrics(q: np.ndarray, interior: float = 0.10) -> Optional[dict]:
    q = np.asarray(q, float).ravel()
    if q.size < 8 or not np.isfinite(q).all() or q.min() <= 0:
        if q.size < 8 or not np.isfinite(q).all():
            return None
    out = {}
    for tag, sl in (("full", slice(None)),
                    ("interior", slice(int(interior * q.size),
                                       q.size - int(interior * q.size)))):
        v = q[sl]
        v = np.clip(v, 1e-300, None)
        v = v / v.mean()          # unit-mean == uniform reference is 1.0
        out[f"q_max_rel_dev_{tag}"] = float(np.max(np.abs(v - 1.0)))
        out[f"q_rms_rel_dev_{tag}"] = float(np.sqrt(np.mean((v - 1.0) ** 2)))
        out[f"q_tv_to_uniform_{tag}"] = float(0.5 * np.mean(np.abs(v - 1.0)))
        lg = np.log(v)
        out[f"log_q_range_kT_{tag}"] = float(lg.max() - lg.min())
        out[f"log_q_std_kT_{tag}"] = float(lg.std())
    return out


def recon_oracle_from_profiles(F_ref, F_hat, beta) -> Optional[dict]:
    """q_oracle ~ exp(-beta (F_ref - B)).  Any additive constant in either
    profile cancels under normalisation, so the *centred* stored F_hat may be
    used in place of the raw bias B.  Validated against the stored WCA oracle
    target (see oracle_target_reconstruction_check.csv)."""
    F_ref = np.asarray(F_ref, float).ravel()
    F_hat = np.asarray(F_hat, float).ravel()
    if F_ref.shape != F_hat.shape or not np.isfinite(F_ref).all() \
            or not np.isfinite(F_hat).all() or not np.isfinite(beta):
        return None
    lg = -float(beta) * (F_ref - F_hat)
    lg -= lg.max()
    q = np.exp(lg)
    m = shape_metrics(q)
    if m is None:
        return None
    return {"recon_oracle_" + k: v for k, v in m.items()}


def _push(rows, base, q, extra=None):
    m = shape_metrics(q)
    if m is None:
        return
    r = dict(base)
    r.update(m)
    if extra:
        r.update(extra)
    rows.append(r)


def scan_shapes() -> pd.DataFrame:
    rows: List[dict] = []

    # --- entropic bottleneck (stage0) -- stored q is ALWAYS the ESTIMATED one
    for f in sorted(glob.glob(os.path.join(
            REPO, "results/entropic_bottleneck/raw/stage0_reproduce/*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "q_target" not in z.files:
            continue
        q = z["q_target"]
        if q.dtype == object or q.shape == ():
            continue
        _push(rows, dict(system="entropic_bottleneck", stage="stage0_reproduce",
                         cell="stage0", cell_id="eb|stage0_reproduce",
                         run_method=str(z["method"]),
                         seed=int(z["seed"]),
                         stored_target_semantics="ESTIMATED-for-every-method "
                         "(eb_abffr_core._finalize)",
                         beta=float(z["cfg__beta"]),
                         ema=float(z["cfg__target_ema_rate"]), source=f), q,
              recon_oracle_from_profiles(z["F_ref"], z["F_hat"],
                                         float(z["cfg__beta"])))

    # --- entropy-dominant main sweep
    for f in sorted(glob.glob(os.path.join(
            REPO, "results/entropy_dominant_bottleneck/sweep_20260614_015145/"
                  "raw/main/*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "q_target" not in z.files:
            continue
        q = z["q_target"]
        if q.dtype == object or q.shape == ():
            continue
        _push(rows, dict(system="entropy_dominant_bottleneck", stage="main_sweep",
                         cell="phi%g" % float(z["cfg__phi"]),
                         cell_id="edb|main|phi%g" % float(z["cfg__phi"]),
                         run_method=str(z["method"]), seed=int(z["seed"]),
                         stored_target_semantics="ESTIMATED-for-every-method "
                         "(edb_abffr_core._finalize)",
                         beta=float(z["cfg__beta"]),
                         ema=float(z["cfg__target_ema_rate"]), source=f), q,
              recon_oracle_from_profiles(z["F_ref"], z["F_hat"],
                                         float(z["cfg__beta"])))

    # --- EMA-tuning rerun: the direct EMA-lag probe
    for f in sorted(glob.glob(os.path.join(
            REPO, "results/entropy_dominant_bottleneck_ema_tuning/"
                  "ema_sweep_20260615_175039/*/raw/main/fr_*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "q_target" not in z.files:
            continue
        q = z["q_target"]
        if q.dtype == object or q.shape == ():
            continue
        _push(rows, dict(system="entropy_dominant_bottleneck_ema_tuning",
                         stage="ema_sweep",
                         cell="phi%g|ema%g" % (float(z["cfg__phi"]),
                                               float(z["cfg__target_ema_rate"])),
                         cell_id="edbema|phi%g|ema%g" % (
                             float(z["cfg__phi"]),
                             float(z["cfg__target_ema_rate"])),
                         run_method=str(z["method"]), seed=int(z["seed"]),
                         stored_target_semantics="ESTIMATED-for-every-method "
                         "(edb_abffr_core._finalize)",
                         beta=float(z["cfg__beta"]),
                         ema=float(z["cfg__target_ema_rate"]), source=f), q,
              recon_oracle_from_profiles(z["F_ref"], z["F_hat"],
                                         float(z["cfg__beta"])))

    # --- slow-transverse ORPHAN (largest pilot only, to bound the scan)
    for f in sorted(glob.glob(os.path.join(
            REPO, "results/entropy_dominant_bottleneck_slow_transverse/"
                  "pilot_20260617_001656/raw/*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "q_target" not in z.files:
            continue
        q = z["q_target"]
        if q.dtype == object or q.shape == ():
            continue
        _push(rows, dict(system="slow_transverse_ORPHAN",
                         stage="pilot_20260617_001656",
                         cell="phi%g|muy%g" % (float(z["cfg__phi"]),
                                               float(z["cfg__mu_y"])),
                         cell_id="slowtrans|20260617_001656|phi%g|muy%g" % (
                             float(z["cfg__phi"]), float(z["cfg__mu_y"])),
                         run_method=str(z["method"]), seed=int(z["seed"]),
                         stored_target_semantics="PRESUMED estimated-for-every-"
                         "method (uncommitted generator; UNVERIFIED)",
                         beta=float(z["cfg__beta"]),
                         ema=float(z["cfg__target_ema_rate"]), source=f), q,
              recon_oracle_from_profiles(z["F_ref"], z["F_hat"],
                                         float(z["cfg__beta"])))

    # --- 2-D metastability toy (final profiles CSV)
    prof = os.path.join(REPO, "results/two_dim_xi_x/production_gpu/"
                              "production_gpu_profiles.csv")
    if os.path.exists(prof):
        p = pd.read_csv(prof, usecols=["run_id", "method", "target_type", "seed",
                                       "gamma", "eta", "burnin_fraction",
                                       "fr_every", "q_target"])
        for rid, g in p.groupby("run_id"):
            _tag2d = "g%g_eta%g_bi%g_fe%d" % (
                g.gamma.iloc[0], g.eta.iloc[0],
                g.burnin_fraction.iloc[0], g.fr_every.iloc[0])
            _push(rows, dict(system="metastability_toy_2D", stage="production_gpu",
                             cell=_tag2d,
                             cell_id=("twodim|prod|" + _tag2d
                                      if g.method.iloc[0] != "abf_only" else ""),
                             run_method=g.method.iloc[0], seed=int(g.seed.iloc[0]),
                             stored_target_semantics="METHOD-SPECIFIC "
                             "(target_type=%s)" % g.target_type.iloc[0],
                             beta=np.nan, ema=np.nan, source=prof),
                  g["q_target"].to_numpy())

    # --- WCA (method-specific q IS stored)
    wca_sets = [
        ("wca_production", "main",
         os.path.join(REPO, "results/wca_production/raw/main__*.npz")),
        ("wca_phase_diagram", "production",
         os.path.join(REPO, "results/wca_phase_diagram/production/raw/*.npz")),
        ("wca_representative", "representative",
         os.path.join(REPO, "results/wca_representative/raw/*.npz")),
    ]
    for system, stage, pat in wca_sets:
        for f in sorted(glob.glob(pat)):
            z = np.load(f, allow_pickle=True)
            if "final_q_target" not in z.files:
                continue
            q = np.asarray(z["final_q_target"], float)
            if not np.isfinite(q).all():
                continue
            # NOTE: the raw .npz does NOT carry ``physics_tag`` (that field
            # exists only in the summary CSVs); rebuild it from the stored
            # physics scalars.
            if {"beta", "h", "w", "n_dim", "a"} <= set(z.files):
                tag = "b%g_h%g_w%g_n%d_a%g" % (
                    float(z["beta"]), float(z["h"]), float(z["w"]),
                    int(z["n_dim"]), float(z["a"]))
            else:
                tag = "a%g" % float(z["a"])
            extra = {}
            # reconstruct the ORACLE target from stored profiles (validation)
            if {"ref_free_energy", "final_pmf", "beta"} <= set(z.files):
                extra = recon_oracle_from_profiles(z["ref_free_energy"],
                                                   z["final_pmf"],
                                                   float(z["beta"])) or {}
            cid = ("wcaprod|main" if system == "wca_production" else
                   f"wcaphase|{stage}|{tag}" if system == "wca_phase_diagram"
                   else f"wcarep|{tag}")
            _push(rows, dict(system=system, stage=stage, cell=tag, cell_id=cid,
                             run_method=str(z["method"]), seed=int(z["seed"]),
                             stored_target_semantics="METHOD-SPECIFIC "
                             "(wca_abffr_core._build_fr_target)",
                             beta=float(z["beta"]) if "beta" in z.files else np.nan,
                             ema=float(z["target_ema_rate"]), source=f), q,
                  extra)

    # --- alkanes (one npz per JOB, profiles shaped (n_seeds, n_grid))
    for f in sorted(glob.glob(os.path.join(
            REPO, "results/alkanes/production/raw/*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "final_q_target" not in z.files:
            continue
        Q = np.asarray(z["final_q_target"], float)
        if Q.ndim != 2:
            continue
        seeds = np.asarray(z["seeds"]).ravel()
        spec = json.loads(str(z["spec_json"])) if "spec_json" in z.files else {}
        for i in range(Q.shape[0]):
            if not np.isfinite(Q[i]).all():
                continue
            _acell = "%s_b%g_s%g_%s_%s" % (
                str(z["molecule"]), float(z["beta"]), float(z["sigma"]),
                "dec" if bool(z["decouple"]) else "full", str(z["init_mode"]))
            _push(rows, dict(system="alkanes_torsion", stage=str(z["stage"]),
                             cell=_acell,
                             cell_id="alk|%s|%s" % (str(z["stage"]), _acell),
                             run_method=str(z["name"]),
                             seed=int(seeds[i]),
                             stored_target_semantics="METHOD-SPECIFIC "
                             "(alkanes.core._fr_target)",
                             beta=float(z["beta"]),
                             ema=float(spec.get("target_ema_rate", np.nan)),
                             source=f), Q[i])

    # --- pentane R15: q_target NOT stored; reconstruct the ORACLE target only
    for f in sorted(glob.glob(os.path.join(
            REPO, "results/alkanes_cv_extension/r15_methods/raw/production__*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "final_q_target" in z.files:
            continue
        if not {"ref_F", "final_pmf", "beta"} <= set(z.files):
            continue
        P = np.asarray(z["final_pmf"], float)
        ref = np.asarray(z["ref_F"], float)
        seeds = np.asarray(z["seeds"]).ravel()
        # restrict to the THERMAL window: outside it ABF has no support and the
        # reconstructed oracle target is dominated by unvisited bins.
        thr = float(z["thermal_delta"]) if "thermal_delta" in z.files else np.inf
        win = np.isfinite(ref) & ((ref - np.nanmin(ref)) <= thr)
        for i in range(P.shape[0]):
            lo = -float(z["beta"]) * (ref - P[i])
            good = np.isfinite(lo) & win
            if good.sum() < 8:
                continue
            l = lo[good] - np.nanmax(lo[good])
            q = np.exp(l)
            _push(rows, dict(system="pentane_R15_distance_CV", stage="production",
                             cell=str(z["stage"]) + "_" + str(z["name"]),
                             run_method="RECONSTRUCTED_oracle_target",
                             seed=int(seeds[i]),
                             stored_target_semantics="q_target NOT STORED; "
                             "oracle target RECONSTRUCTED as "
                             "exp(-beta*(ref_F - final_pmf)) on the THERMAL "
                             "window only. The ESTIMATED target is NOT "
                             "recoverable (F_target_ema absent).",
                             beta=float(z["beta"]), ema=np.nan, source=f), q)

    return pd.DataFrame(rows)


# ============================================================================
# time-resolved lower-bound witness for target-vs-uniform (2-D toy only)
# ============================================================================
def target_uniform_time_witness() -> pd.DataFrame:
    src = os.path.join(REPO, "results/two_dim_xi_x/production_gpu/"
                             "production_gpu_runs_long.csv")
    if not os.path.exists(src):
        return pd.DataFrame()
    d = pd.read_csv(src, usecols=["run_id", "method", "target_type", "seed",
                                  "gamma", "eta", "burnin_fraction", "fr_every",
                                  "t", "marginal_l2_uniform", "marginal_l2_target"])
    d = d[d.method != "abf_only"]
    # |L2(p,q) - L2(p,u)| <= L2(q,u)  (reverse triangle inequality)
    d["witness"] = (d.marginal_l2_target - d.marginal_l2_uniform).abs()
    g = d.groupby(["method", "target_type", "gamma", "eta", "burnin_fraction",
                   "fr_every"])
    out = g.agg(n_rows=("witness", "size"),
                witness_median=("witness", "median"),
                witness_p90=("witness", lambda s: float(np.nanpercentile(s, 90))),
                witness_max=("witness", "max"),
                witness_final_median=("witness", lambda s: np.nan)).reset_index()
    fin = d.sort_values("t").groupby(
        ["method", "target_type", "gamma", "eta", "burnin_fraction", "fr_every"]
    ).tail(1).groupby(["method", "target_type", "gamma", "eta",
                       "burnin_fraction", "fr_every"])["witness"].median()
    out = out.drop(columns=["witness_final_median"]).merge(
        fin.rename("witness_final_median").reset_index(),
        on=["method", "target_type", "gamma", "eta", "burnin_fraction", "fr_every"])
    out["quantity"] = ("|L2(p_hat,q_target) - L2(p_hat,uniform)|  "
                       "== LOWER BOUND on L2(q_target, uniform)")
    out["source"] = src
    return out


# ============================================================================
# main
# ============================================================================
def main() -> int:
    os.makedirs(OUT, exist_ok=True)

    load_two_dim()
    load_entropic_bottleneck()
    load_edb_main()
    load_edb_ema()
    load_slow_transverse()
    load_wca_production()
    load_wca_phase()
    load_wca_representative()
    load_alkanes()
    load_r15()

    seeds = pd.concat(_SEED_ROWS, ignore_index=True)
    cells = pd.DataFrame(_CELLS)
    seeds.to_csv(os.path.join(OUT, "per_seed_matched.csv"), index=False)

    cell_tab, contrast_tab = analyse(seeds, cells)
    cell_tab.to_csv(os.path.join(OUT, "master_cell_table.csv"), index=False)
    contrast_tab.to_csv(os.path.join(OUT, "contrast_table.csv"), index=False)

    counts = (cell_tab.groupby(["system", "classification"]).size()
              .unstack(fill_value=0).reset_index())
    counts.to_csv(os.path.join(OUT, "classification_counts.csv"), index=False)

    pd.DataFrame(_EXCL).to_csv(os.path.join(OUT, "excluded_and_filters.csv"),
                               index=False)

    shp = scan_shapes()
    shp.to_csv(os.path.join(OUT, "target_shape_scope_per_run.csv"), index=False)
    if len(shp):
        agg = (shp.groupby(["system", "stage", "cell", "run_method",
                            "stored_target_semantics"])
               .agg(n_runs=("q_max_rel_dev_full", "size"),
                    beta=("beta", "first"), ema=("ema", "first"),
                    max_rel_dev_full_median=("q_max_rel_dev_full", "median"),
                    max_rel_dev_interior_median=("q_max_rel_dev_interior", "median"),
                    rms_rel_dev_full_median=("q_rms_rel_dev_full", "median"),
                    tv_to_uniform_full_median=("q_tv_to_uniform_full", "median"),
                    log_q_range_kT_full_median=("log_q_range_kT_full", "median"),
                    log_q_range_kT_interior_median=("log_q_range_kT_interior", "median"),
                    log_q_std_kT_full_median=("log_q_std_kT_full", "median"))
               .reset_index())
        agg.to_csv(os.path.join(OUT, "target_shape_scope_by_cell.csv"), index=False)

        # validate the analytic reconstruction of the ORACLE target against the
        # stored one (only WCA stores both).  Used to license the R15
        # reconstruction, where q_target was never written out.
        v = shp[shp["run_method"] == "fr_oracle"].dropna(
            subset=["recon_oracle_q_max_rel_dev_full"])
        if len(v):
            v = v.assign(
                abs_err=(v["q_max_rel_dev_full"]
                         - v["recon_oracle_q_max_rel_dev_full"]).abs(),
                rel_err=((v["q_max_rel_dev_full"]
                          - v["recon_oracle_q_max_rel_dev_full"]).abs()
                         / v["q_max_rel_dev_full"]))
            (v.groupby(["system", "stage"])
             .agg(n_runs=("abs_err", "size"),
                  stored_max_rel_dev_median=("q_max_rel_dev_full", "median"),
                  recon_max_rel_dev_median=("recon_oracle_q_max_rel_dev_full", "median"),
                  abs_err_median=("abs_err", "median"),
                  abs_err_max=("abs_err", "max"),
                  rel_err_median=("rel_err", "median"),
                  rel_err_max=("rel_err", "max"))
             .reset_index()
             .to_csv(os.path.join(OUT, "oracle_target_reconstruction_check.csv"),
                     index=False))

        # ---- does a less-collapsed deployable target change the outcome? ----
        # join per-cell ESTIMATED-target deviation to the est-vs-uni contrast.
        est = shp[shp["run_method"].isin(["fr_estimated", "abf_fr_estimated"])]
        est = est[est["cell_id"].astype(str) != ""]
        gest = (est.groupby("cell_id")
                .agg(n_runs=("log_q_range_kT_full", "size"),
                     est_log_q_range_kT=("log_q_range_kT_full", "median"),
                     est_max_rel_dev=("q_max_rel_dev_full", "median"),
                     recon_oracle_log_q_range_kT=("recon_oracle_log_q_range_kT_full", "median"))
                .reset_index())
        gest["oracle_over_estimated_shape_ratio"] = (
            gest["recon_oracle_log_q_range_kT"] / gest["est_log_q_range_kT"])
        cc = contrast_tab[(contrast_tab.contrast == "est_minus_uni")
                          & (contrast_tab.metric == PRIMARY_METRIC)]
        j = gest.merge(cc, on="cell_id", how="inner")
        j.to_csv(os.path.join(OUT, "shape_vs_effect.csv"), index=False)

    # compact headline view of the master table
    keep = ["cell_id", "system", "stage", "cell_label", "abf_n_seeds",
            "abf_mean_final_l2_F", "equiv_margin_abs",
            "estimated_n", "uniform_n", "oracle_n",
            "estimated_final_l2_F_median", "uniform_final_l2_F_median",
            "oracle_final_l2_F_median",
            "estimated_int_l2_F_median", "uniform_int_l2_F_median",
            "oracle_int_l2_F_median",
            "estimated_final_l2_Fp_median", "uniform_final_l2_Fp_median",
            "oracle_final_l2_Fp_median",
            "estimated_ancestor_ess_median", "uniform_ancestor_ess_median",
            "oracle_ancestor_ess_median",
            "estimated_event_fraction_median", "uniform_event_fraction_median",
            "oracle_event_fraction_median",
            "estimated_winrate_vs_abf", "uniform_winrate_vs_abf",
            "oracle_winrate_vs_abf",
            "est_minus_uni_mean_diff", "est_minus_uni_ci_lo",
            "est_minus_uni_ci_hi", "est_minus_uni_verdict",
            "orc_minus_uni_mean_diff", "orc_minus_uni_ci_lo",
            "orc_minus_uni_ci_hi", "orc_minus_uni_verdict",
            "orc_minus_est_mean_diff", "orc_minus_est_ci_lo",
            "orc_minus_est_ci_hi", "orc_minus_est_verdict",
            "verdict_triple", "classification", "classification_note",
            "oracle_direction_vs_estimated", "oracle_effect_pct_of_abf"]
    keep = [k for k in keep if k in cell_tab.columns]
    cell_tab[keep].to_csv(os.path.join(OUT, "master_cell_table_headline.csv"),
                          index=False)

    wit = target_uniform_time_witness()
    if len(wit):
        wit.to_csv(os.path.join(OUT, "target_uniform_time_witness_2Dtoy.csv"),
                   index=False)

    manifest = dict(
        script=os.path.abspath(__file__),
        generated_utc=pd.Timestamp.utcnow().isoformat(),
        equivalence_margin_fraction_of_abf_baseline=EQUIV_MARGIN_FRAC,
        equivalence_margin_metric=PRIMARY_METRIC,
        equivalence_rule=("EQUIVALENT iff 95% paired bootstrap CI of the "
                          "per-seed difference is CONTAINED in "
                          "[-0.05*ABF_mean_final_L2F, +0.05*ABF_mean_final_L2F]; "
                          "DIFFERENT iff CI disjoint from that band; "
                          "UNDERPOWERED otherwise (includes every CI wider "
                          "than the band)."),
        n_bootstrap=N_BOOT, bootstrap_rng_seed=BOOT_SEED,
        n_cells=int(len(cell_tab)),
        n_seed_rows=int(len(seeds)),
        classification_counts=cell_tab["classification"].value_counts().to_dict(),
        numpy=np.__version__, pandas=pd.__version__,
        python=sys.version.split()[0],
    )
    with open(os.path.join(OUT, "audit_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
