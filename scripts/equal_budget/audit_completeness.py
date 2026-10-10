#!/usr/bin/env python
"""MANDATORY completeness audit of the equal-budget replica ladders (docs/equal_budget/SCIENTIFIC_PLAN.md).

    python scripts/equal_budget/audit_completeness.py --system gateway|lta300|lta150|all \
        [--results-root DIR] [--config PATH | --config SYSTEM=PATH ...] [--fig-root DIR] [--analysis-root DIR] \
        [--out PATH]

Exit code 0 = PASS for every audited system, 1 = FAIL (at least one failure), 2 = usage error.  Prints one table
per system and writes <analysis-root>/completeness_audit.json (or --out).  The JSON covers exactly the systems audited
in this invocation (the final audit is --system all); every failure / warning carries a code, a location and a
message.  Flagged confidence intervals are listed but do not fail the audit.

Layout (every root holds one directory per config ``out_dir``: gateway, lta_T300, lta_T150)
  <results-root>/<out_dir>/N<N>/s<seed>_<abf|fr>.npz    run files (s<seed>_<m>.npz.ckpt.npz while running)
  <results-root>/<out_dir>/ledger.csv                   run_ladder.py's ledger (cross-checked when present)
  <results-root>/<out_dir>/STATUS.json                  OPTIONAL: the reason for every NOT RUN / partial N
  <analysis-root>/<out_dir>/analysis/summary.json       analyze_ladder.py output, with its run_metrics/ cache
                                                        (fallback: <analysis-root>/summary.json of that system)
  <fig-root>/<out_dir>/N<N>/MANIFEST.json               plot_config.py manifest of the configuration (system, N)
  <fig-root>/<out_dir>/synthesis/MANIFEST.json          plot_synthesis.py manifest (S1-S8s); fallback, with a WARN,
                                                        <results-root>/<out_dir>/synthesis/ (a synthesis drawn with
                                                        --fig-root set to the results root)
  Defaults: results-root results/equal_budget_v2, analysis-root = results-root (analyze_ladder's default output),
  fig-root figures/equal_budget_v2 (plot_config.py's default), configs configs/equal_budget_v2/<system>_production.json.
  The references are ALWAYS the frozen results/equal_budget_v2/references/*.npz (eqb_metrics.REF_DIR).

What is checked (FAIL unless marked WARN)
1. Plan status: every planned N of the config gets complete / partial / NOT RUN (planned (method, seed) pairs:
   ABF at every N, FR at N >= 2).  A partial or NOT RUN N needs a non-empty reason in STATUS.json (N_UNEXPLAINED);
   STATUS.json must not contradict the files (STATUS_CONTRADICTION: e.g. 'NOT RUN' while runs exist, 'complete'
   while runs are missing, an explicit 'missing' list that differs) and must not say 'running' (STATUS_NOT_FINAL).
   Checkpoints of unfinished jobs are reported as running.  Once any run file exists, ledger.csv must exist
   (LEDGER_MISSING).  A ledger row claiming 'complete' for an absent or invalid file fails (LEDGER_CLAIM) even when
   its N is documented, unless STATUS.json names that run under "deleted" ({method: [seeds]}) with a reason (then
   WARN): a completed run may not silently become NOT RUN.  Unplanned files: WARN.
2. Run files (every file present): readable, meta status 'complete' (FILE_STATUS), N / seed / method of the file
   equal to its path, system, T, B, h, n_steps = B / N, engine knobs = config engine_cfg (eqb_metrics.check_plan),
   diagnostics on; every REQUIRED array of its engine (GATEWAY_ARRAYS / LTA_ARRAYS below) present (ARRAY_MISSING)
   with the contract shape and value class (ARRAY_SHAPE / ARRAY_VALUES); the save grid equal to the production
   driver's run_ladder.save_grid(P, n_steps) and, checked one by one, the last save at u = 1, the 200 uniform
   u = k/200 (exact integer steps), the profile snapshot fractions of the config and the physical checkpoints
   t <= T_N (GRID); integrity (INTEGRITY): exact deposit totals (gateway pre-deposit sum C_all = N k; LTA
   sum C_all = N (k + 1), sum C_prod = N max(0, k - burn_in + 1)), cumulative counters non-decreasing,
   hist_inst rows sum to N, region fractions sum to 1, events-per-opportunity histogram total = opportunities,
   exact force-evaluation count, no FR activity in an ABF arm, FR opportunities > 0 in an FR arm (WARN when the FR
   arm never realised a death: it is then identical to ABF).
   PAIRING: the ABF and FR arms of one (N, seed) share initial conditions and Langevin noise (plan section 6):
   gateway meta noise_seed equal and M_all / C_all / hist_inst bitwise identical at every save before the FR arm's
   first opportunity with a candidate (fr_opp_event_cum == 0, no slot gather yet); LTA meta rng entropy equal and
   M_all / C_all / M_prod / C_prod / hist_inst identical at every save before its first realised death.
   ENGINE_MIXED: complete runs of one ladder from more than one engine build (engine, engine_sha256) FAIL unless
   STATUS.json documents it ({"engine_builds": {"reason": ...}}); a file without engine_sha256: ENGINE_UNIDENTIFIED
   (WARN).  CFG_INCONSISTENT: cfg_json must be identical across every run of the ladder except the per-run fields
   N, seed, n_steps, cap, n_trace (physics defaults outside the config included).
3. References: the frozen npz exists and documents its units and gauge (REF_UNDOCUMENTED); LTA units are the
   threshold units kJ/mol and kJ/mol/rad, T_K and kT match the config, the source is the exact MC (never the
   published umbrella reference); the stated gauge holds numerically (REF_GAUGE: mean of F_ref over the 180 bins,
   gateway over the 151 eval-window nodes, is 0); the accepted scorer reproduces the reference (REF_SCORER); the
   summary's reference block names the same file.
4. Analysis summary (required once any run is complete): strict JSON, schema / system / metrics version / 10 000
   bootstrap resamples, plan block = config (B, h, N ladder, seeds, thresholds); its status block = the files on
   disk; EVERY per-N metric has per-seed values for exactly the complete files of that (N, method) (SUMMARY_UNBACKED
   / SUMMARY_STALE) and its n / n_inf agree with them; contrasts, max-transient and best-allocation seed / N sets
   match the files (an N whose two arms share no seed carries n = 0 transients); the plan's metric list is present
   (SUMMARY_INCOMPLETE); provenance: every run's run_metrics cache entry has the current key (the file's CONTENT
   sha256, metrics version, config digest, scoring-code sha256, frozen-reference sha256) and every per-seed value in
   the summary equals that cached scalar (without a cache: summary not older than any run file); the summary's
   reference sha256 equals the frozen file's (REF_SUMMARY).  SUMMARY_AGGREGATE: every aggregate block (per_N
   statistics, contrasts, max transient, best allocation, FR activity, threshold floors, establishment null) is
   RECOMPUTED from the run-metrics cache with analyze_ladder.summarize_runs and must equal the summary (rel. 1e-12).
5. Confidence intervals: every key ending in 'ci95' in summary.json is classified (CI_UNAUDITABLE otherwise) and
   its seed count n is compared with the planned seeds: paired contrasts n (seeds with both arms finite), tau
   contrasts n_both_finite, best allocation the per-N seed counts of the candidate N AND the candidate N set against
   the planned ladder (a best-allocation CI over a reduced ladder is flagged, naming the missing N).
   n = planned -> 'full'; otherwise 'flagged' with the reason (planned seeds missing on disk, seeds excluded as
   censored / non-finite, undefined CI, planned N missing).  n > planned fails (CI_COUNT).
6. Figures.  Per configuration (system, N), from plot_config.py's <fig-root>/<out_dir>/N<N>/MANIFEST.json
   ({"system", "system_dir", "N", "fixture", "metrics_version", "seeds": {method: [...]}, "figures": [{"tag",
   "description", "files": [paths relative to the manifest]}, ...]}): every COMPLETE N needs its manifest
   (MANIFEST_MISSING) and the items A (time, budget, profiles), B (time, budget, profiles), C1-C5, D, E, F (time,
   budget), each satisfied by >= 1 figure whose tag matches PER_N_ITEMS (FIG_MISSING); a placeholder (description
   'placeholder...' or a 'note') stands in for a figure only for E at N = 1 (FIG_NOTE otherwise).  Every listed
   file must exist (FIG_MISSING), be non-empty (FIG_EMPTY) and carry a valid png / pdf / svg / jpg / gif header
   (FIG_CORRUPT), supplementary figures included.  The figures must be drawn from the current files (FIG_STALE):
   manifest seeds = complete seeds on disk, current metrics version, manifest newer than every run file of that N,
   and every source's recorded sha256 equal to the run file's current content;
   system / N / fixture flag must match (MANIFEST_MISMATCH).  For an incomplete N an existing manifest is checked
   the same way, staleness and placeholders as WARN.  Synthesis, from plot_synthesis.py's
   <out_dir>/synthesis/MANIFEST.json ({"system", "summary": {"sha256", ...}, "freshness": {"stale"}, "plan",
   "figures": [{"id", "files"}, ...]}): every id of SYNTHESIS_FIGURES present (S1, S2, S2s, S3, S4a, S4b, S5a,
   S5b, S6a, S6b, S7, S7s, S8, S8s: tau in physical time AND budget fraction, the secondary reference;
   SYNTH_MISSING), files sound, drawn from the CURRENT summary.json (sha256) and not from a stale one
   (MANIFEST_STALE), plan = config (MANIFEST_MISMATCH); required, like the summary, once any run is complete.

Mechanism cells (docs/mechanism/SCIENTIFIC_PLAN.md): --system gateway_family --config <cell config> audits one cell
(configs/mechanism/cells/<experiment>/<variant>.json); a cell config with any other --system, or gateway_family with
a non-cell config, is refused (eqb_family.check_invocation).  Defaults come from the cell: results root = parent of
its results_dir, analysis = its analysis_dir (or --analysis-root itself / <analysis-root>/<cell>/analysis), figures =
its fig_dir (or <fig-root>/<cell>), --out = <analysis_dir>/completeness_audit.json (never inside results/ or
figures/equal_budget_v2).  In addition: every run's engine must be the cell's engine_version and its model fields the
cell's (eqb_family.check_plan -> PLAN_MISMATCH; gateway_ladder_numba/2 files carry no model fields and are read as
alpha 1, lam 1), the meta 'system' label is 'gateway' or 'gateway_family', the reference is the cell's frozen
results/mechanism/references/<experiment>_<variant>_reference.npz (model, h, gauge; its primary arrays bitwise the
equal-budget gateway reference: REF_PRIMARY), the per-run metric list is the cell's (tau on e_F, e_Fp_stat incl.
tau_bgrid, TV_half; no raw e_F' tau below its hard floor), the ledger is the cell's 'ledger' (production: ONE
results/mechanism/<experiment>/ledger.csv per experiment written by scripts/mechanism/run_cells.py, rows filtered to
the cell by 'ledger_filter' {experiment, variant}; reuse cells: results/equal_budget_v2/gateway/ledger.csv), and a
reuse cell's bitwise-reuse gate record (plan section 5; scripts/mechanism/reuse_gate.py) must license the reuse
(REUSE_GATE).

STATUS.json (results/equal_budget_v2/<out_dir>/STATUS.json)
  {"N": {"16": {"status": "NOT RUN", "reason": "resource ceiling reached at ..."},
         "8":  {"status": "partial", "reason": "...", "missing": {"fr": [30003]}}}}
  The per-N block may also be called "per_N" or be the top level; keys "16" or "N16"; a bare string value is a
  reason.  status: "NOT RUN" (also "not_run"), "partial", "complete"; "missing" (optional) is {method: [seeds]} or a
  list of [method, seed] pairs; "deleted" (optional, same format) names COMPLETED runs that were removed afterwards
  (the reason must say why).  Top level "engine_builds": {"reason": "..."} documents a ladder that mixes engine
  builds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eqb_metrics as M  # noqa: E402
import eqb_family as MF  # noqa: E402  (system / config dispatch; mechanism cells)
import analyze_ladder as A  # noqa: E402
import run_ladder as RL  # noqa: E402  (the production driver's save grid)

ROOT = M.ROOT
SCHEMA = "eqb_completeness_audit/2"
SYSTEM_ORDER = ("gateway", "lta300", "lta150")
KB = 0.008314462618
N_TRACE_MAX = 32

# ---- required raw arrays of each engine: (key, shape, value class) ------------------------------------------------
# shape tokens: "S" saves, "nb" 180 production bins, "K" trace times, "ntr" = min(N, 32) traced walkers, "cap1" =
# meta cap + 1, integers are literal sizes; "one" = a single value (0-d or shape (1,)).
# value classes: "int" integral and finite, "count" integral, finite and >= 0, "float" numeric and finite, "num"
# numeric (NaN allowed, e.g. ESS of an ABF arm), "str" a JSON string.
GATEWAY_ARRAYS = (
    ("save_step", ("S",), "int"), ("save_t", ("S",), "float"), ("save_u", ("S",), "float"),
    ("M_all", ("S", "nb"), "float"), ("C_all", ("S", "nb"), "count"), ("hist_inst", ("S", "nb"), "count"),
    ("region_frac", ("S", 3), "float"), ("events_cum", ("S", 2), "count"), ("lineage_visited", ("S", 1), "count"),
    ("first_arrival_step", "one", "int"),
    ("gen_nuniq_run", ("S",), "num"), ("gen_ess_run", ("S",), "num"), ("gen_maxfam_run", ("S",), "num"),
    ("gen_nuniq_win", ("S",), "num"), ("gen_ess_win", ("S",), "num"), ("gen_maxfam_win", ("S",), "num"),
    ("fr_deaths_cum", ("S",), "count"), ("fr_kd_cum", ("S",), "count"), ("fr_kc_cum", ("S",), "count"),
    ("fr_opp_cum", ("S",), "count"), ("fr_opp_event_cum", ("S",), "count"), ("fr_events_hist", ("cap1",), "count"),
    ("traces_step", ("K",), "int"), ("traces_t", ("K",), "float"), ("traces", ("K", "ntr"), "float"),
    ("traces_anc", ("K", "ntr"), "int"),
    ("n_force_evals", "one", "int"), ("wall_s", "one", "float"), ("peak_rss_mb", "one", "float"),
    ("meta_json", "one", "str"), ("cfg_json", "one", "str"),
)
# gateway_ladder_numba/2 additions: shape-checked when present, not required
GATEWAY_OPTIONAL = (
    ("gen_win_age_steps", ("S",), "int"), ("fr_opp_death_cum", ("S",), "count"),
    ("fr_events_hist_cum", ("S", "cap1"), "count"), ("fr_deaths_hist", ("cap1",), "count"),
    ("fr_deaths_hist_cum", ("S", "cap1"), "count"), ("traces_rebirth", ("K", "ntr"), "int"),
)
LTA_ARRAYS = (
    ("save_step", ("S",), "int"), ("save_t", ("S",), "float"), ("save_u", ("S",), "float"),
    ("M_all", ("S", "nb"), "float"), ("C_all", ("S", "nb"), "count"),
    ("M_prod", ("S", "nb"), "float"), ("C_prod", ("S", "nb"), "count"), ("hist_inst", ("S", "nb"), "count"),
    ("region_frac", ("S", 3), "float"),
    ("events_window_crossings", ("S",), "count"), ("events_translocations", ("S",), "count"),
    ("lineage_ever_window", ("S",), "count"), ("lineage_ever_opposite", ("S",), "count"),
    ("first_window_step", "one", "int"), ("first_opposite_step", "one", "int"),
    ("gen_n_unique", ("S",), "num"), ("gen_ess", ("S",), "num"), ("gen_max_frac", ("S",), "num"),
    ("gen_n_unique_win", ("S",), "num"), ("gen_ess_win", ("S",), "num"), ("gen_max_frac_win", ("S",), "num"),
    ("cum_deaths", ("S",), "count"), ("cum_opps", ("S",), "count"), ("cum_opps_with_event", ("S",), "count"),
    ("events_per_opp_hist", ("S", "cap1"), "count"),
    ("traces_step", ("K",), "int"), ("traces", ("K", "ntr"), "float"), ("traces_anc", ("K", "ntr"), "int"),
    ("n_force_evals", "one", "int"), ("wall_s", "one", "float"), ("peak_rss_mb", "one", "float"),
    ("meta_json", "one", "str"), ("cfg_json", "one", "str"),
)
LTA_OPTIONAL = ()
# cumulative arrays (non-decreasing along the saves).  NOT the lineage counts (lineage_visited, lineage_ever_*): a
# flagged lineage can die under FR and be replaced by a copy of an unflagged one, so they may decrease.
CUMULATIVE = {
    "gateway": ("C_all", "events_cum", "fr_deaths_cum", "fr_kd_cum", "fr_kc_cum", "fr_opp_cum", "fr_opp_event_cum"),
    "lta": ("C_all", "C_prod", "events_window_crossings", "events_translocations", "cum_deaths", "cum_opps",
            "cum_opps_with_event", "events_per_opp_hist"),
}

# ---- figures ----------------------------------------------------------------------------------------------------
# required per-N items and the plot_config.py tags that satisfy them (A1_eF_vs_t, A2_eF_vs_u, A3_F_profiles,
# B1_eFp_vs_t, B2_eFp_vs_u, B3_Fp_profiles [+ B4_Fp_profiles_projected], C1_*, C2_TV_vs_t/u, C3_density_heatmap_u/t,
# C4_*, C5_traces, D_establishment_vs_t/u, E_genealogy_vs_t/u, F_summary_t/u); every matching figure is file-checked
PER_N_ITEMS = (
    ("A_time", r"^A\d*_.*(_vs_t|_t)$"), ("A_budget", r"^A\d*_.*(_vs_u|_u)$"), ("A_profiles", r"^A\d*_.*profiles"),
    ("B_time", r"^B\d*_.*(_vs_t|_t)$"), ("B_budget", r"^B\d*_.*(_vs_u|_u)$"), ("B_profiles", r"^B\d*_.*profiles"),
    ("C1", r"^C1(_|$)"), ("C2", r"^C2(_|$)"), ("C3", r"^C3(_|$)"), ("C4", r"^C4(_|$)"), ("C5", r"^C5(_|$)"),
    ("D", r"^D(_|$)"), ("E", r"^E(_|$)"), ("F_time", r"^F(_.*)?(_vs_t|_t)$"), ("F_budget", r"^F(_.*)?(_vs_u|_u)$"),
)
NOTE_ALLOWED = {("E", 1)}                    # (item, N): at N = 1, E may be a placeholder / note (no FR arm)
SYNTHESIS_FIGURES = ("S1", "S2", "S2s", "S3", "S4a", "S4b", "S5a", "S5b", "S6a", "S6b", "S7", "S7s", "S8", "S8s")
CFG_PER_RUN = {"N", "seed", "n_steps", "cap", "n_trace"}      # cfg_json fields allowed to differ between runs
MAGIC = {".png": b"\x89PNG\r\n\x1a\n", ".pdf": b"%PDF", ".jpg": b"\xff\xd8\xff", ".jpeg": b"\xff\xd8\xff",
         ".gif": b"GIF8"}

# ---- the plan's per-run metric list that the summary must carry (section 4) -------------------------------------
REQUIRED_RUN_METRICS = (
    ["Ibar_F", "Ibar_Fp", "final_e_F", "final_e_Fp"]
    + [f"tau_{e}_{n}_{f}" for e in ("e_F", "e_Fp") for n in M.THR_NAMES for f in ("t", "u", "censored")]
    + ["tau_TV_half_t", "tau_TV_half_u", "tau_TV_half_censored", "Ibar_TV_half", "final_TV_half", "Ibar_TV_inst",
       "final_TV_inst", "TV_inst_floor", "est_t", "est_u", "est_censored", "est_cum_u", "est_unreliable",
       "est_null_p_hold_second_half", "final_transitions", "final_gen_ess_run", "final_gen_nuniq_run",
       "min_gen_ess_win", "max_gen_maxfam_win", "fr_deaths", "fr_zero_deaths", "fr_mean_event_frac_per_opp",
       "fr_deaths_frac_per_opp", "n_force_evals", "wall_s", "peak_rss_mb"]
    + [f"tau_bgrid_{e}_{n}_u" for e in ("e_F", "e_Fp") for n in M.THR_NAMES] + ["tau_bgrid_TV_half_u"])
def required_run_metrics(system, P, floors=None):
    """The plan's per-run metric list for ``system``: REQUIRED_RUN_METRICS + the kind's (equal-budget systems,
    unchanged); a mechanism cell also tau (and tau_bgrid) on the companions with thresholds of their own (e_Fp_stat)
    and NOT the tau at thresholds below a hard floor (``floors`` = the summary's threshold_floors)."""
    kind = MF.system_kind(system)
    req = REQUIRED_RUN_METRICS + REQUIRED_RUN_METRICS_KIND[kind]
    if not MF.is_family(system):
        return req
    comp = MF.tau_companions(kind, P)
    req = (req + [f"tau_{e}_{n}_{f}" for e in comp for n in M.THR_NAMES for f in ("t", "u", "censored")]
           + [f"tau_bgrid_{e}_{n}_u" for e in comp for n in M.THR_NAMES])
    drop = MF.unreachable_tau_stems(floors)
    return [k for k in req if not any(k.startswith(st + "_") or k.startswith(st.replace("tau_", "tau_bgrid_", 1) + "_")
                                      for st in drop)]


REQUIRED_RUN_METRICS_KIND = {
    "gateway": ["Ibar_F_em", "Ibar_Fp_em", "final_e_F_em", "final_e_Fp_em", "Ibar_Fp_stat", "final_e_Fp_stat",
                "fr_candidates", "first_right_u", "first_right_t"],
    "lta": ["Ibar_Fp_proj", "final_e_Fp_proj", "first_window_u", "first_window_t", "first_opposite_u",
            "first_opposite_t"],
}


# =================================================================================================== utilities
class Report:
    """Failures / warnings of one system, each {code, where, message}."""

    def __init__(self):
        self.failures, self.warnings = [], []

    def fail(self, code, where, msg):
        self.failures.append(dict(code=code, where=where, message=msg))

    def warn(self, code, where, msg):
        self.warnings.append(dict(code=code, where=where, message=msg))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def num(v):
    """summary.json value -> float ('inf' -> inf, null -> NaN, bool -> 0/1)."""
    if v is None:
        return math.nan
    if isinstance(v, str):
        return float(v)                      # "inf" / "-inf"
    return float(v)


def same(a, b, rtol=1e-12):
    a, b = num(a), num(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))


def strict_json(path):
    def bad(x):
        raise ValueError(f"non-strict JSON constant {x}")
    with open(path) as fh:
        return json.load(fh, parse_constant=bad)


def nkey(k):
    """'1024' / 'N1024' / 1024 -> 1024 (None if not an N key)."""
    m = re.match(r"^N?(\d+)$", str(k))
    return int(m.group(1)) if m else None


def short(paths, k=5):
    paths = list(paths)
    return ", ".join(str(p) for p in paths[:k]) + (f", ... ({len(paths)} in all)" if len(paths) > k else "")


# =================================================================================================== run files
def _shape_ok(a, spec, dims):
    if spec == "one":
        return a.size == 1 and a.ndim <= 1
    want = tuple(dims[d] if isinstance(d, str) else d for d in spec)
    return a.shape == want


def _values_ok(a, cls):
    if cls == "str":
        return a.dtype.kind in "US"
    if a.dtype.kind not in "biuf":
        return False
    if cls == "num":
        return True
    if not np.all(np.isfinite(a)):
        return False
    if cls in ("int", "count") and not np.all(a == np.rint(a)):
        return False
    if cls == "count" and np.any(a < 0):
        return False
    return True


def expected_grid(system, P, n_steps):
    return np.asarray(RL.save_grid(dict(P, system_key=MF.engine_key(system)), int(n_steps)), dtype=np.int64)


def audit_run_file(path, system, P, N, seed, method, grid):
    """Returns (status 'complete' | 'invalid', problems [(code, message)], info)."""
    kind = MF.system_kind(system)
    probs, info = [], {}
    try:
        with np.load(path, allow_pickle=False) as z:
            arr = {k: z[k] for k in z.files}
    except Exception as e:  # noqa: BLE001
        return "invalid", [("FILE_UNREADABLE", f"cannot read: {type(e).__name__}: {e}")], info
    required = GATEWAY_ARRAYS if kind == "gateway" else LTA_ARRAYS
    optional = GATEWAY_OPTIONAL if kind == "gateway" else LTA_OPTIONAL
    missing = [k for k, _, _ in required if k not in arr]
    if missing:
        probs.append(("ARRAY_MISSING", f"required {kind} arrays missing: {missing}"))
    try:
        meta = json.loads(str(arr["meta_json"]))
        cfg = json.loads(str(arr["cfg_json"]))
    except Exception as e:  # noqa: BLE001
        probs.append(("FILE_META", f"meta_json / cfg_json unreadable: {type(e).__name__}: {e}"))
        return "invalid", probs, info
    st = meta.get("status")
    if st != "complete":
        probs.append(("FILE_STATUS", f"meta status is {st!r}, not 'complete'"))
    # ---- identity and plan
    B, h = int(P["B"]), float(P["engine_cfg"]["h"])
    n_steps = B // N
    want = dict(N=N, seed=seed, method=method, system=("gateway" if kind == "gateway" else "lta"), n_steps=n_steps,
                B=B)
    if MF.is_family(system):           # gateway family: the label may be 'gateway' (reuse cells) or 'gateway_family'
        want.pop("system")
        if meta.get("system") not in MF.FAMILY_SYSTEM_LABELS:
            probs.append(("FILE_IDENTITY", f"meta system = {meta.get('system')!r}, expected one of "
                                           f"{MF.FAMILY_SYSTEM_LABELS}"))
    for k, v in want.items():
        if meta.get(k) != v:
            probs.append(("FILE_IDENTITY", f"meta {k} = {meta.get(k)!r}, expected {v!r} (path / config)"))
    if float(meta.get("h", math.nan)) != h:
        probs.append(("FILE_IDENTITY", f"meta h = {meta.get('h')!r}, config h = {h!r}"))
    if kind == "lta" and float(meta.get("T_K", math.nan)) != float(P["T_K"]):
        probs.append(("FILE_IDENTITY", f"meta T_K = {meta.get('T_K')!r}, config T_K = {P['T_K']!r}"))
    try:
        MF.check_plan(dict(meta=meta, cfg=cfg), P, path, dict(N=N, seed=seed, method=method))
    except M.MetricsError as e:
        probs.append(("PLAN_MISMATCH", str(e)))
    except Exception as e:  # noqa: BLE001
        probs.append(("PLAN_MISMATCH", f"cannot compare with the config: {type(e).__name__}: {e}"))
    if meta.get("diagnostics") is not True:
        probs.append(("FILE_DIAGNOSTICS", f"meta diagnostics = {meta.get('diagnostics')!r} (diagnostic arrays absent "
                                          f"or meaningless)"))
    info.update(engine=meta.get("engine"), engine_sha256=meta.get("engine_sha256"), git_commit=meta.get("git_commit"),
                cap=meta.get("cap"), noise_seed=meta.get("noise_seed"), rng_entropy=(meta.get("rng") or {}).get("entropy"),
                compile_target=meta.get("compile_target"))
    info["_cfg"] = cfg
    # ---- shapes and value classes
    S = int(arr["save_step"].shape[0]) if "save_step" in arr and arr["save_step"].ndim == 1 else None
    K = int(arr["traces_step"].shape[0]) if "traces_step" in arr and arr["traces_step"].ndim == 1 else None
    cap = meta.get("cap")
    dims = dict(S=S, K=K, nb=M.NB, ntr=min(N, N_TRACE_MAX), cap1=(int(cap) + 1 if isinstance(cap, int) else None))
    info.update(S=S, K=K)
    for k, spec, cls in tuple(required) + tuple(optional):
        if k not in arr:
            continue
        a = arr[k]
        if spec != "one" and any(isinstance(d, str) and dims[d] is None for d in spec):
            probs.append(("ARRAY_SHAPE", f"{k}: cannot resolve its contract shape {spec} (save_step / traces_step / "
                                         f"meta cap unusable)"))
            continue
        if not _shape_ok(a, spec, dims):
            exp = spec if spec == "one" else tuple(dims[d] if isinstance(d, str) else d for d in spec)
            probs.append(("ARRAY_SHAPE", f"{k}: shape {a.shape}, contract {exp}"))
        elif not _values_ok(a, cls):
            probs.append(("ARRAY_VALUES", f"{k}: values are not of class {cls!r} (dtype {a.dtype})"))
    if S is not None and meta.get("n_saves") is not None and int(meta["n_saves"]) != S:
        probs.append(("ARRAY_SHAPE", f"meta n_saves {meta['n_saves']} != {S} saves"))
    if any(c in ("ARRAY_MISSING", "ARRAY_SHAPE", "ARRAY_VALUES") for c, _ in probs) or S is None:
        return "invalid", probs, info

    # ---- save grid
    step = arr["save_step"].astype(np.int64)
    if not np.array_equal(step, grid):
        miss = np.setdiff1d(grid, step)
        extra = np.setdiff1d(step, grid)
        probs.append(("GRID", f"save grid != run_ladder.save_grid(P, {n_steps}): {miss.size} planned saves missing "
                              f"{miss[:5].tolist()}, {extra.size} unplanned {extra[:5].tolist()}"))
    if step.size == 0 or step[-1] != n_steps:
        probs.append(("GRID", f"last save at step {step[-1] if step.size else None}, not at u = 1 (n_steps {n_steps})"))
    if not np.allclose(arr["save_t"], step * h, rtol=1e-12, atol=0):
        probs.append(("GRID", "save_t != save_step h"))
    if not np.allclose(arr["save_u"], step / float(n_steps), rtol=1e-12, atol=0):
        probs.append(("GRID", "save_u != save_step / n_steps"))
    sset = {int(s): i for i, s in enumerate(step)}
    kk = np.arange(1, M.N_UNIFORM + 1)
    if n_steps % M.N_UNIFORM == 0:
        tgt = kk * (n_steps // M.N_UNIFORM)
    else:
        tgt = np.clip(np.round(kk / M.N_UNIFORM * n_steps).astype(np.int64), 1, n_steps)
        probs.append(("GRID", f"n_steps {n_steps} is not a multiple of {M.N_UNIFORM}: the uniform u are not exact"))
    absent = [int(k) for k, s in zip(kk, tgt) if int(s) not in sset]
    if absent:
        probs.append(("GRID", f"{len(absent)} of the {M.N_UNIFORM} uniform budget fractions u = k/{M.N_UNIFORM} "
                              f"are not saves (k = {absent[:5]}...)"))
    elif n_steps % M.N_UNIFORM == 0:
        uu = arr["save_u"][[sset[int(s)] for s in tgt]]
        if not np.allclose(uu, kk / M.N_UNIFORM, rtol=0, atol=1e-12):
            probs.append(("GRID", "save_u at the uniform saves differs from k/200"))
    info["n_uniform_saves"] = M.N_UNIFORM - len(absent)
    prof_missing = []
    for u in P["profile_snapshot_u"]:
        x = float(u) * n_steps
        s = int(round(x))
        if abs(x - s) > 1e-6 or s not in sset or abs(arr["save_u"][sset[s]] - float(u)) > 1e-12:
            prof_missing.append(u)
    if prof_missing:
        probs.append(("GRID", f"profile snapshot fractions u {prof_missing} are not exact saves"))
    phys_missing = []
    for tt in P["physical_checkpoints_t"]:
        s = int(round(float(tt) / h))
        if 0 < s <= n_steps and (s not in sset or abs(arr["save_t"][sset[s]] - float(tt)) > 1e-9 * max(1.0, tt)):
            phys_missing.append(tt)
    if phys_missing:
        probs.append(("GRID", f"physical checkpoints t {phys_missing} (<= T_N) are not saves"))
    tr = arr["traces_step"].astype(np.int64)
    if tr.size and (np.any(np.diff(tr) <= 0) or tr[0] < 0 or tr[-1] > n_steps):
        probs.append(("GRID", "traces_step is not strictly increasing inside [0, n_steps]"))

    # ---- integrity
    C = arr["C_all"]
    if kind == "gateway":
        dep = N * step                                          # pre-deposit convention: C_all[k].sum() = N k
        dep_txt = "N k (gateway pre-deposit)"
    else:
        dep = N * (step + 1)                                    # LTA: includes the saved step's deposit
        dep_txt = "N (k + 1) (LTA)"
    if not np.array_equal(C.sum(1), dep.astype(C.dtype)):
        probs.append(("INTEGRITY", f"sum C_all != {dep_txt} at {int(np.sum(C.sum(1) != dep))} saves"))
    if kind == "lta":
        try:
            b = int(meta["knobs"]["burn_in"]["steps"])
        except Exception:  # noqa: BLE001
            b = int(P["engine_cfg"]["burn_in_steps"])
        dp = N * np.maximum(0, step - b + 1)
        if not np.array_equal(arr["C_prod"].sum(1), dp.astype(arr["C_prod"].dtype)):
            probs.append(("INTEGRITY", f"sum C_prod != N max(0, k - burn_in + 1) (burn_in {b})"))
    if not np.all(np.isfinite(arr["M_all"])):
        probs.append(("INTEGRITY", "M_all holds non-finite values"))
    hs = arr["hist_inst"].sum(1)
    if not np.all(hs == N):
        probs.append(("INTEGRITY", f"hist_inst rows do not sum to N = {N} at {int(np.sum(hs != N))} saves"))
    rf = arr["region_frac"]
    if np.any(rf < -1e-12) or np.any(rf > 1 + 1e-12) or not np.allclose(rf.sum(1), 1.0, rtol=0, atol=1e-9):
        probs.append(("INTEGRITY", "region_frac rows are not fractions summing to 1"))
    for k in CUMULATIVE[kind]:
        a = arr[k]
        if a.shape[0] > 1 and np.any(np.diff(a, axis=0) < 0):
            probs.append(("INTEGRITY", f"cumulative {k} decreases between saves"))
    if kind == "gateway":
        opps = int(arr["fr_opp_cum"][-1])
        deaths = int(arr["fr_deaths_cum"][-1])
        cand = int(arr["fr_kd_cum"][-1] + arr["fr_kc_cum"][-1])
        htot = int(arr["fr_events_hist"].sum())
        nfe_exp = N * n_steps
    else:
        opps = int(arr["cum_opps"][-1])
        deaths = int(arr["cum_deaths"][-1])
        cand = deaths
        htot = int(arr["events_per_opp_hist"][-1].sum())
        nfe_exp = N * (n_steps + 1)
    if htot != opps:
        probs.append(("INTEGRITY", f"events-per-opportunity histogram total {htot} != opportunities {opps}"))
    nfe = int(np.asarray(arr["n_force_evals"]).ravel()[0])
    if nfe != nfe_exp:
        probs.append(("INTEGRITY", f"n_force_evals {nfe} != expected {nfe_exp}"))
    if method == "abf" and (opps or deaths or cand):
        probs.append(("INTEGRITY", f"ABF arm shows FR activity (opportunities {opps}, deaths {deaths})"))
    if method == "fr":
        if opps <= 0:
            probs.append(("INTEGRITY", "FR arm with zero FR opportunities (FR never active)"))
        if kind == "lta" and meta.get("fr_active") is not True:
            probs.append(("INTEGRITY", f"FR arm with meta fr_active = {meta.get('fr_active')!r}"))
    info.update(n_opps=opps, n_deaths=deaths, n_force_evals=nfe, wall_s=float(np.asarray(arr["wall_s"]).ravel()[0]))
    return ("complete" if not probs else "invalid"), probs, info


# =================================================================================================== pairing / cfg
PAIR_ARRAYS = {"gateway": ("M_all", "C_all", "hist_inst"), "lta": ("M_all", "C_all", "M_prod", "C_prod", "hist_inst")}


def pairing_problems(kind, pa, pf, ia, if_):
    """ABF / FR arms of one (N, seed) share initial conditions and Langevin noise: identical noise identity in meta,
    and bitwise-identical accumulators / instantaneous histograms at every save before FR changed the population
    (gateway: no opportunity with a candidate yet -- no slot gather; LTA: no realised death yet).  Returns
    (problems, number of saves compared)."""
    probs = []
    if kind == "gateway":
        if ia.get("noise_seed") != if_.get("noise_seed"):
            probs.append(f"Langevin noise_seed differs: ABF {ia.get('noise_seed')} vs FR {if_.get('noise_seed')}")
        gate = "fr_opp_event_cum"
    else:
        if ia.get("rng_entropy") != if_.get("rng_entropy"):
            probs.append(f"rng entropy differs: ABF {ia.get('rng_entropy')} vs FR {if_.get('rng_entropy')}")
        gate = "cum_deaths"
    if ia.get("compile_target") != if_.get("compile_target"):
        probs.append(f"compile target differs (not bitwise paired): {ia.get('compile_target')} vs {if_.get('compile_target')}")
    keys = PAIR_ARRAYS[kind]
    with np.load(pa, allow_pickle=False) as za, np.load(pf, allow_pickle=False) as zf:
        if not np.array_equal(za["save_step"], zf["save_step"]):
            return probs + ["save grids differ between the arms"], 0
        q = np.asarray(zf[gate]) == 0
        n = int(q.sum())
        bad = [k for k in keys if n and not np.array_equal(np.asarray(za[k])[q], np.asarray(zf[k])[q])]
    if bad:
        probs.append(f"{bad} differ between the arms at saves before FR's first population change ({n} saves): the arms "
                     f"do not share initial conditions / Langevin noise")
    return probs, n


def cfg_differences(ref, cfg):
    keys = (set(ref) | set(cfg)) - CFG_PER_RUN
    return sorted(k for k in keys if ref.get(k, "<absent>") != cfg.get(k, "<absent>"))


# =================================================================================================== STATUS / ledger
def norm_status(s):
    if s is None:
        return None
    t = str(s).strip().lower().replace("_", " ").replace("-", " ")
    return {"not run": "NOT RUN", "notrun": "NOT RUN", "partial": "partial", "complete": "complete",
            "running": "running"}.get(t, f"?{s}")


def load_status_json(path, rep, where):
    if not os.path.exists(path):
        return None
    try:
        d = strict_json(path)
    except Exception as e:  # noqa: BLE001
        rep.fail("STATUS_JSON_INVALID", where, f"{path}: {type(e).__name__}: {e}")
        return {}
    if not isinstance(d, dict):
        rep.fail("STATUS_JSON_INVALID", where, f"{path}: top level is not an object")
        return {}
    blk = d.get("N", d.get("per_N"))
    if blk is None:
        blk = {k: v for k, v in d.items() if nkey(k) is not None}
    if not isinstance(blk, dict):
        rep.fail("STATUS_JSON_INVALID", where, f"{path}: the per-N block is not an object")
        return {}
    out = {}
    for k, v in blk.items():
        N = nkey(k)
        if N is None:
            rep.fail("STATUS_JSON_INVALID", where, f"{path}: key {k!r} is not an N")
            continue
        if isinstance(v, str):
            out[N] = dict(status=None, reason=v.strip(), missing=None, deleted=None)
        elif isinstance(v, dict):
            out[N] = dict(status=norm_status(v.get("status")), reason=str(v.get("reason") or "").strip(),
                          missing=v.get("missing"), deleted=v.get("deleted"))
        else:
            rep.fail("STATUS_JSON_INVALID", where, f"{path}: entry for N {k} is neither a reason nor an object")
    return out


def status_missing_set(m):
    """STATUS.json 'missing' -> {(method, seed)} (None if not given / unreadable)."""
    if m is None:
        return None
    out = set()
    try:
        if isinstance(m, dict):
            for meth, seeds in m.items():
                out |= {(str(meth), int(s)) for s in seeds}
        else:
            out = {(str(a), int(b)) for a, b in m}
    except Exception:  # noqa: BLE001
        return "unreadable"
    return out


def load_ledger(path, where=None):
    """Last ledger row per (N, seed, method), in file order; None if no ledger.  ``where``: {column: value} rows must
    match (a mechanism experiment ledger holds every variant of the experiment: rows filtered to one cell BEFORE
    keying, so variants never overwrite each other); a filter column absent from the ledger matches nothing."""
    if not os.path.exists(path):
        return None
    last = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if where and any(str(row.get(k)) != str(v) for k, v in where.items()):
                continue
            try:
                key = (int(row["N"]), int(row["seed"]), str(row["method"]))
            except Exception:  # noqa: BLE001
                continue
            last[key] = row
    return last


# =================================================================================================== references
def audit_family_reference(system, P, rep, summary):
    """Gateway family: the cell's frozen reference (scripts/mechanism/build_references.py)."""
    where = "references"
    path = MF.reference_path(system, P)
    out = dict(path=os.path.relpath(path, ROOT), frozen_dir=os.path.relpath(os.path.dirname(path), ROOT))
    if not os.path.exists(path):
        rep.fail("REF_MISSING", where, f"frozen cell reference {path} does not exist (scripts/mechanism/build_references.py)")
        return out
    out["sha256"] = sha256_file(path)
    with np.load(path, allow_pickle=False) as z:
        ref = {k: z[k] for k in z.files}
    need = MF.FAMILY_REF_ARRAYS + ("h", "model_json")
    miss = [k for k in need if k not in ref]
    if miss:
        rep.fail("REF_MISSING", where, f"{out['path']}: arrays missing {miss}")
        return out
    for k in ("units", "gauge", "secondary_definition"):
        v = ref.get(k)
        v = str(v.item()) if (v is not None and v.ndim == 0 and v.dtype.kind in "US") else None
        if not v:
            rep.fail("REF_UNDOCUMENTED", where, f"{out['path']}: field {k!r} absent or empty")
        out[k] = v
    m = ref["eval_mask"].astype(bool)
    if float(ref["h"]) != float(P["engine_cfg"]["h"]):
        rep.fail("REF_MISMATCH", where, f"{out['path']}: h {float(ref['h'])} != config h {P['engine_cfg']['h']}")
    if int(m.sum()) != 151:
        rep.fail("REF_MISMATCH", where, f"{out['path']}: eval window has {int(m.sum())} nodes, the plan says 151")
    try:
        model = json.loads(str(ref["model_json"]))
    except Exception:  # noqa: BLE001
        model = None
    want_model = MF.family_model(P["engine_cfg"])
    out["model"] = model
    if model != want_model:
        rep.fail("REF_MISMATCH", where, f"{out['path']}: model {model} != the cell's {want_model}")
    g = {k: float(np.mean(ref[k][m])) for k in ("F_ref", "F_ref_em")}
    out["gauge_check"] = dict(mean_F_ref_eval_window=g["F_ref"], mean_F_ref_em_eval_window=g["F_ref_em"])
    if any(abs(v) > 1e-9 for v in g.values()):
        rep.fail("REF_GAUGE", where, f"{out['path']}: stated gauge 'centred on the eval window' violated: {g}")
    eqb_path = os.path.join(M.REF_DIR, "gateway_reference.npz")
    with np.load(eqb_path, allow_pickle=False) as z:
        eqb = {k: z[k] for k in z.files}
    diff = [k for k in MF.PRIMARY_REF_ARRAYS if not np.array_equal(ref[k], eqb[k])]
    out["primary_identical_to"] = os.path.relpath(eqb_path, ROOT)
    out["primary_digest"] = MF.primary_digest(ref)
    if diff:
        rep.fail("REF_PRIMARY", where, f"{out['path']}: primary reference arrays {diff} differ from the equal-budget "
                                       f"{out['primary_identical_to']} (the primary reference must be identical for "
                                       f"every variant)")
    d = (ref["F_ref_em"] - ref["F_ref"])[m]
    out["em_floor_F_rms"] = float(np.sqrt(np.mean((d - d.mean()) ** 2)))
    try:
        sc = MF.make_scorer(system, P)            # verifies the file against the scorer, bitwise primary included
        out["scorer"] = {k: v for k, v in sc.reference_info.items() if not isinstance(v, dict)}
    except Exception as e:  # noqa: BLE001
        rep.fail("REF_SCORER", where, f"the family scorer refuses the frozen reference: {type(e).__name__}: {e}")
    if summary is not None:
        sref = summary.get("reference") or {}
        if sref.get("path") != out["path"]:
            rep.fail("REF_SUMMARY", where, f"summary.json reference path {sref.get('path')!r} != {out['path']!r}")
        if sref.get("sha256") != out.get("sha256"):
            rep.fail("REF_SUMMARY", where, f"summary.json was scored against reference sha256 "
                                           f"{str(sref.get('sha256'))[:12]}, the frozen file is {str(out.get('sha256'))[:12]}")
        if (sref.get("primary_reference") or {}).get("digest") != out["primary_digest"]:
            rep.fail("REF_SUMMARY", where, "summary.json primary-reference digest differs from the frozen file's")
    return out


def audit_reference(system, P, rep, summary):
    if MF.is_family(system):
        return audit_family_reference(system, P, rep, summary)
    kind = MF.system_kind(system)
    where = "references"
    fname = "gateway_reference.npz" if kind == "gateway" else f"lta_T{int(round(float(P['T_K'])))}_reference.npz"
    path = os.path.join(M.REF_DIR, fname)
    out = dict(path=os.path.relpath(path, ROOT), frozen_dir=os.path.relpath(M.REF_DIR, ROOT))
    if not os.path.exists(path):
        rep.fail("REF_MISSING", where, f"frozen reference {path} does not exist")
        return out
    out["sha256"] = sha256_file(path)
    with np.load(path, allow_pickle=False) as z:
        ref = {k: z[k] for k in z.files}

    def text(k):
        v = ref.get(k)
        return str(v.item()) if (v is not None and v.ndim == 0 and v.dtype.kind in "US") else None

    if kind == "gateway":
        need = ("x_grid", "eval_mask", "F_ref", "F_ref_em", "Fp_ref", "h")
        docs = ("units", "gauge")
    else:
        need = ("T_K", "kT", "grid_phi", "F_ref", "F_ref_se", "gamma_ref", "gamma_se", "F_ref_mf")
        docs = ("units_F", "units_gamma", "gauge", "source")
    miss = [k for k in need if k not in ref]
    if miss:
        rep.fail("REF_MISSING", where, f"{fname}: arrays missing {miss}")
        return out
    for k in docs:
        v = text(k)
        if not v:
            rep.fail("REF_UNDOCUMENTED", where, f"{fname}: field {k!r} absent or empty (units / gauge must be "
                                                f"documented in the frozen reference)")
        out[k] = v
    if kind == "gateway":
        m = ref["eval_mask"].astype(bool)
        if float(ref["h"]) != float(P["engine_cfg"]["h"]):
            rep.fail("REF_MISMATCH", where, f"{fname}: h {float(ref['h'])} != config h {P['engine_cfg']['h']}")
        if int(m.sum()) != 151:
            rep.fail("REF_MISMATCH", where, f"{fname}: eval window has {int(m.sum())} nodes, the plan says 151")
        g = {k: float(np.mean(ref[k][m])) for k in ("F_ref", "F_ref_em")}
        out["gauge_check"] = dict(mean_F_ref_eval_window=g["F_ref"], mean_F_ref_em_eval_window=g["F_ref_em"])
        if any(abs(v) > 1e-9 for v in g.values()):
            rep.fail("REF_GAUGE", where, f"{fname}: stated gauge 'centred on the eval window' violated: {g}")
        out["units_Fp"] = None
        rep.warn("REF_UNITS_IMPLICIT", where, f"{fname}: one 'units' field ({out.get('units')!r}) documents F; the "
                                             f"units of F' (per unit x) are implicit")
        d = (ref["F_ref_em"] - ref["F_ref"])[m]
        out["em_floor_F_rms"] = float(np.sqrt(np.mean((d - d.mean()) ** 2)))
    else:
        T = float(P["T_K"])
        if float(ref["T_K"]) != T:
            rep.fail("REF_MISMATCH", where, f"{fname}: T_K {float(ref['T_K'])} != config T_K {T}")
        if abs(float(ref["kT"]) - KB * T) > 1e-9 * KB * T:
            rep.fail("REF_MISMATCH", where, f"{fname}: kT {float(ref['kT'])} != kB T {KB * T}")
        for k in ("grid_phi", "F_ref", "F_ref_se", "gamma_ref", "gamma_se", "F_ref_mf"):
            if ref[k].shape != (M.NB,):
                rep.fail("REF_MISMATCH", where, f"{fname}: {k} shape {ref[k].shape}, expected ({M.NB},)")
        if out.get("units_F") and out["units_F"] != "kJ/mol":
            rep.fail("REF_UNITS", where, f"{fname}: units_F {out['units_F']!r}; the frozen e_F thresholds are kJ/mol")
        if out.get("units_gamma") and out["units_gamma"] != "kJ/mol/rad":
            rep.fail("REF_UNITS", where, f"{fname}: units_gamma {out['units_gamma']!r}; the frozen e_F' thresholds "
                                         f"are kJ/mol/rad")
        src = out.get("source") or ""
        if "/MC" not in src or "umbrella" in src.lower():
            rep.fail("REF_SOURCE", where, f"{fname}: source {src!r} is not the exact MC reference (plan section 2)")
        mF = float(np.mean(ref["F_ref"]))
        gm = float(np.mean(ref["gamma_ref"]))
        gse = float(np.sqrt(np.mean(ref["gamma_se"] ** 2)) / math.sqrt(M.NB))
        out["gauge_check"] = dict(mean_F_ref=mF, gamma_circular_mean=gm, gamma_circular_mean_z=gm / gse if gse else None)
        if abs(mF) > 1e-9:
            rep.fail("REF_GAUGE", where, f"{fname}: stated gauge 'mean over the 180 bins = 0' violated (mean {mF:.3g})")
        if gse and abs(gm / gse) > 3:
            rep.warn("REF_GAMMA_MEAN", where, f"{fname}: circular mean of gamma_ref {gm:.4g} is {gm / gse:.1f} SE "
                                              f"from 0")
        out["noise"] = dict(F_ref_se_rms=float(np.sqrt(np.mean(ref["F_ref_se"] ** 2))),
                            gamma_se_rms=float(np.sqrt(np.mean(ref["gamma_se"] ** 2))))
    try:
        sc = MF.make_scorer(system, P)            # gateway: the accepted scorer must reproduce the frozen reference
        out["scorer"] = {k: v for k, v in sc.reference_info.items() if not isinstance(v, dict)}
    except Exception as e:  # noqa: BLE001
        rep.fail("REF_SCORER", where, f"the metrics scorer refuses the frozen reference: {type(e).__name__}: {e}")
    if summary is not None:
        sref = summary.get("reference") or {}
        if sref.get("path") != out["path"]:
            rep.fail("REF_SUMMARY", where, f"summary.json reference path {sref.get('path')!r} != {out['path']!r}")
        if sref.get("sha256") != out.get("sha256"):
            rep.fail("REF_SUMMARY", where, f"summary.json was scored against reference sha256 "
                                           f"{str(sref.get('sha256'))[:12]}, the frozen file is {str(out.get('sha256'))[:12]}")
        if kind == "lta" and sref.get("units") not in (None, dict(F=out.get("units_F"), Fp=out.get("units_gamma"))):
            rep.fail("REF_SUMMARY", where, f"summary.json reference units {sref.get('units')} != the npz units")
    return out


# =================================================================================================== summary + CIs
def resolve_analysis_dir(analysis_root, out_dir, system, P=None):
    if MF.is_cell(P):              # mechanism cell: its analysis_dir, or --analysis-root itself / <root>/<cell>/analysis
        if analysis_root is None:
            return MF.cell_analysis_dir(P)
        direct = os.path.join(analysis_root, "summary.json")
        if os.path.exists(direct):
            try:
                S = strict_json(direct)
                if S.get("system") == system and (S.get("cell") or {}).get("cell") == P["cell"]:
                    return analysis_root
            except Exception:  # noqa: BLE001
                pass
        return os.path.join(analysis_root, P["cell"], "analysis")
    cand = os.path.join(analysis_root, out_dir, "analysis")
    if os.path.exists(os.path.join(cand, "summary.json")):
        return cand
    direct = os.path.join(analysis_root, "summary.json")
    if os.path.exists(direct):
        try:
            if strict_json(direct).get("system") == system:
                return analysis_root
        except Exception:  # noqa: BLE001
            pass
    return cand


def audit_summary(system, P, rep, analysis_dir, results_root, D, file_paths, any_complete):
    """Checks section 4 of the module docstring.  D: {(N, method): sorted complete seeds}."""
    kind = MF.system_kind(system)
    where = "summary"
    path = os.path.join(analysis_dir, "summary.json")
    out = dict(path=path, present=os.path.exists(path))
    if not out["present"]:
        if any_complete:
            rep.fail("SUMMARY_MISSING", where, f"{path} does not exist (run analyze_ladder.py --system {system})")
        return out, None
    try:
        S = strict_json(path)
    except Exception as e:  # noqa: BLE001
        rep.fail("SUMMARY_FORMAT", where, f"{path} is not strict JSON: {type(e).__name__}: {e}")
        return out, None
    out.update(generated_utc=S.get("generated_utc"), sha256=sha256_file(path), mtime=os.path.getmtime(path),
               metrics_version=S.get("metrics_version"), n_boot=S.get("n_boot"))
    if S.get("schema") != A.SCHEMA:
        rep.fail("SUMMARY_FORMAT", where, f"schema {S.get('schema')!r} != {A.SCHEMA!r}")
    if S.get("system") != system:
        rep.fail("SUMMARY_PLAN", where, f"summary system {S.get('system')!r} != {system!r}")
    if S.get("metrics_version") != M.METRICS_VERSION:
        rep.fail("SUMMARY_VERSION", where, f"metrics version {S.get('metrics_version')!r} != current "
                                           f"{M.METRICS_VERSION!r}")
    if S.get("n_boot") != M.N_BOOT:
        rep.fail("SUMMARY_NBOOT", where, f"n_boot {S.get('n_boot')!r}; the plan fixes {M.N_BOOT} bootstrap resamples")
    pl = S.get("plan") or {}
    want = dict(B=int(P["B"]), h=float(P["engine_cfg"]["h"]), N_ladder=[int(n) for n in P["N_ladder"]],
                seeds=[int(s) for s in P["seeds"]], thresholds=P["thresholds"])
    for k, v in want.items():
        if pl.get(k) != v:
            rep.fail("SUMMARY_PLAN", where, f"summary plan {k} = {pl.get(k)!r} != config {v!r}")
    if S.get("results_root") and os.path.realpath(S["results_root"]) != os.path.realpath(results_root):
        rep.warn("SUMMARY_MOVED", where, f"summary was made from results root {S['results_root']} (audited: "
                                         f"{results_root}); provenance is checked by relative path, mtime and size")
    for w in S.get("warnings") or []:
        rep.warn("ANALYSIS_WARNING", where, str(w))
    for f in ("tables.md", "median_curves.npz"):
        if not os.path.exists(os.path.join(analysis_dir, f)):
            rep.warn("SUMMARY_OUTPUTS", where, f"{f} missing next to summary.json")
    rows = A.plan(P)
    status = S.get("status") or {}
    per_N = S.get("per_N") or {}
    contrasts = S.get("contrasts") or {}
    transient = S.get("max_transient") or {}
    backed = {}
    n_values = 0
    # ---- status block and per-N metrics vs the files on disk
    for r in rows:
        N = r["N"]
        sN = status.get(str(N))
        bN = per_N.get(str(N)) or {}
        backed[N] = {}
        for m in r["methods"]:
            disk = D[(N, m)]
            sm = ((sN or {}).get("methods") or {}).get(m)
            if sm is None:
                rep.fail("SUMMARY_STALE", f"N{N} {m}", "summary status block lacks this (N, method)")
            else:
                if sorted(int(s) for s in sm.get("complete", [])) != disk:
                    rep.fail("SUMMARY_STALE", f"N{N} {m}", f"summary lists complete seeds {sm.get('complete')}, the "
                                                          f"disk has {disk}")
                if sm.get("invalid"):
                    rep.fail("SUMMARY_SKIPPED_INVALID", f"N{N} {m}", f"the analysis skipped invalid runs "
                                                                     f"{sorted(sm['invalid'])} (--skip-invalid)")
            blk = bN.get(m)
            if not disk:
                if blk:
                    rep.fail("SUMMARY_UNBACKED", f"N{N} {m}", "per-N metrics present but no complete run file")
                backed[N][m] = dict(n_files=0, n_metrics=0, n_seeds_summary=0)
                continue
            if not blk:
                rep.fail("SUMMARY_STALE", f"N{N} {m}", f"{len(disk)} complete run files but no per-N metrics")
                backed[N][m] = dict(n_files=len(disk), n_metrics=0, n_seeds_summary=0)
                continue
            req = required_run_metrics(system, P, S.get("threshold_floors"))
            lack = [k for k in req if k not in blk]
            if lack:
                rep.fail("SUMMARY_INCOMPLETE", f"N{N} {m}", f"plan metrics missing from per_N: {lack}")
            seen, unb, stl, intl = set(), {}, {}, []
            for k, d in blk.items():
                ps = d.get("per_seed") or {}
                keys = {int(s) for s in ps}
                seen |= keys
                if not keys <= set(disk):
                    unb[k] = keys - set(disk)
                if not set(disk) <= keys:
                    stl[k] = set(disk) - keys
                vals = [num(v) for v in ps.values()]
                n_ok = sum(1 for v in vals if not math.isnan(v))
                n_inf = sum(1 for v in vals if math.isinf(v))
                if d.get("n") != n_ok or d.get("n_inf") != n_inf:
                    intl.append(k)
                n_values += len(ps)
            if unb:
                rep.fail("SUMMARY_UNBACKED", f"N{N} {m}", f"per-seed values for seeds without a complete run file "
                                                          f"{sorted(set().union(*unb.values()))} in {len(unb)} metric(s): "
                                                          f"{short(unb)}")
            if stl:
                rep.fail("SUMMARY_STALE", f"N{N} {m}", f"complete run files {sorted(set().union(*stl.values()))} absent "
                                                       f"from {len(stl)} metric(s): {short(stl)}")
            if intl:
                rep.fail("SUMMARY_INTERNAL", f"N{N} {m}", f"n / n_inf disagree with the per-seed values in "
                                                          f"{len(intl)} metric(s): {short(intl)}")
            backed[N][m] = dict(n_files=len(disk), n_metrics=len(blk), n_seeds_summary=len(seen))
        # ---- contrasts and transients
        if "fr" in r["methods"]:
            common = sorted(set(D[(N, "abf")]) & set(D[(N, "fr")]))
            have = D[(N, "abf")] and D[(N, "fr")]
            c = contrasts.get(str(N))
            if not have:
                if c:
                    rep.fail("SUMMARY_UNBACKED", f"N{N} contrasts", "contrasts present without both arms on disk")
                continue
            if not c:
                rep.fail("SUMMARY_STALE", f"N{N} contrasts", "both arms on disk but no paired contrasts")
                continue
            tau_prim, tau_sec = A.tau_keys(system, P, S.get("threshold_floors"))
            groups = dict(primary=A.PRIMARY, secondary=A.SECONDARY[kind], marginal=A.MARGINAL, tau=tau_prim,
                          tau_secondary=tau_sec)
            for grp, keys in groups.items():
                lack = [k for k in keys if k not in (c.get(grp) or {})]
                if lack:
                    rep.fail("SUMMARY_INCOMPLETE", f"N{N} contrasts {grp}", f"missing {lack}")
                stl, unb, intl = [], set(), []
                for k, cc in (c.get(grp) or {}).items():
                    ps = {int(s) for s in (cc.get("per_seed") or {})}
                    if cc.get("n_common") != len(common):
                        stl.append(f"{k} (n_common {cc.get('n_common')})")
                    if not ps <= set(common):
                        unb |= ps - set(common)
                    if grp.startswith("tau"):        # internal: against the summary's own n_common (disk: above)
                        rk = cc.get("rank") or {}
                        nc = cc.get("n_common")
                        if len(ps) != nc or sum(rk.get(x, 0) for x in ("wins", "losses", "ties")) != nc:
                            intl.append(k)
                    elif cc.get("n") != len(ps):
                        intl.append(k)
                if stl:
                    rep.fail("SUMMARY_STALE", f"N{N} contrasts {grp}", f"{len(common)} seeds have both arms on disk; "
                                                                       f"{len(stl)} contrast(s) differ: {short(stl)}")
                if unb:
                    rep.fail("SUMMARY_UNBACKED", f"N{N} contrasts {grp}", f"per-seed contrasts for seeds without both "
                                                                          f"arms on disk: {sorted(unb)}")
                if intl:
                    rep.fail("SUMMARY_INTERNAL", f"N{N} contrasts {grp}", f"n / rank counts disagree with the per-seed "
                                                                          f"contrasts in {short(intl)}")
            bad_t = [f"{k} (n {t.get('n')})" for k, t in (transient.get(str(N)) or {}).items()
                     if t.get("n") != len(common)]
            if bad_t:
                rep.fail("SUMMARY_STALE", f"N{N} max_transient", f"{len(common)} seeds have both arms on disk; "
                                                                 f"{short(bad_t)}")
            if not transient.get(str(N)):
                rep.fail("SUMMARY_STALE", f"N{N} max_transient", "both arms on disk but no transient comparison")
    # ---- best allocation
    ba = S.get("best_allocation") or {}
    exp_N = dict(abf=[r["N"] for r in rows if D[(r["N"], "abf")]],
                 fr=[r["N"] for r in rows if r["N"] >= 2 and D.get((r["N"], "fr"))])
    if any_complete:
        lack = [k for k in A.BEST_PRIMARY + A.BEST_SECONDARY if k not in ba]
        if lack:
            rep.fail("SUMMARY_INCOMPLETE", "best_allocation", f"missing {lack}")
    for k, b in ba.items():
        for m in ("abf", "fr"):
            x = b.get(m)
            if x is None:
                if exp_N[m]:
                    rep.fail("SUMMARY_STALE", f"best_allocation {k} {m}", f"absent although N {exp_N[m]} have runs")
                continue
            if sorted(int(n) for n in x.get("Ns", [])) != sorted(exp_N[m]):
                rep.fail("SUMMARY_STALE", f"best_allocation {k} {m}", f"candidate N {x.get('Ns')} != N with runs "
                                                                      f"{exp_N[m]}")
            for n, v in (x.get("medians") or {}).items():
                d = ((per_N.get(str(n)) or {}).get(m) or {}).get(k)
                if d is None or not same(v, d.get("median")):
                    rep.fail("SUMMARY_INTERNAL", f"best_allocation {k} {m} N{n}", f"median {v!r} != per_N median "
                                                                                  f"{None if d is None else d.get('median')!r}")
    # ---- provenance: run-metrics cache keyed to the current files, summary values = cached scalars
    cache_dir = os.path.join(analysis_dir, "run_metrics")
    if not os.path.isdir(cache_dir) and S.get("cache", {}).get("dir") and os.path.isdir(S["cache"]["dir"]):
        cache_dir = S["cache"]["dir"]
    prov = dict(cache_dir=cache_dir, runs_checked=0, values_checked=0, aggregates_recomputed=False)
    cur = A.provenance(system, P)
    runs_cache, all_valid = {}, bool(file_paths)
    if os.path.isdir(cache_dir):
        for (N, m, s), fp in sorted(file_paths.items()):
            w = f"N{N} s{s} {m}"
            key, blob, kprob = A.cache_entry_check(fp, system, P, cache_dir, dict(N=N, seed=s, method=m), cur)
            if blob is None:
                all_valid = False
                rep.fail("SUMMARY_STALE", w, f"{kprob[0]} in {cache_dir}: the summary was not computed from this file")
                continue
            sc = blob.get("scalars") or {}
            src = ((blob.get("file") or {}).get("src_path")) or ""
            tail = os.path.normpath(str(src)).split(os.sep)[-3:]
            probs = list(kprob)
            if tail != [P["out_dir"], f"N{N}", f"s{s}_{m}.npz"]:
                probs.append(f"cache entry is for {src}")
            if probs:
                all_valid = False
                rep.fail("SUMMARY_STALE", w, "; ".join(probs) + " -- rerun analyze_ladder.py")
                continue
            prov["runs_checked"] += 1
            try:
                with np.load(os.path.join(cache_dir, f"N{N}", f"s{s}_{m}.npz"), allow_pickle=False) as z:
                    runs_cache[(N, m, s)] = ({k: z[k] for k in z.files}, sc)
            except Exception as e:  # noqa: BLE001
                all_valid = False
                rep.fail("SUMMARY_STALE", w, f"unreadable run-metrics curves: {type(e).__name__}: {e}")
            blk = (per_N.get(str(N)) or {}).get(m) or {}
            bad = []
            for k, d in blk.items():
                if str(s) not in (d.get("per_seed") or {}):
                    continue                      # already reported as SUMMARY_STALE above
                v = d["per_seed"][str(s)]
                if k not in sc or not same(v, sc[k]):
                    bad.append(f"{k} {v!r} vs {sc.get(k)!r}")
                prov["values_checked"] += 1
            if bad:
                rep.fail("SUMMARY_UNBACKED", w, f"{len(bad)} summary value(s) differ from the run's scored value "
                                                f"(summary vs run-metrics cache): {short(bad)}")
        # ---- every aggregate block recomputed from the cache (catches an edited or stale-aggregate summary)
        if all_valid and not any(f["code"] == "SUMMARY_PLAN" for f in rep.failures):
            try:
                blocks, _ = A.summarize_runs(P, system, runs_cache, rows, int(S.get("n_boot") or M.N_BOOT),
                                             MF.make_scorer(system, P))
                want_blocks = M.json_safe(blocks)
                diffs = []
                for name, blk in want_blocks.items():
                    compare_json(S.get(name), blk, name, diffs)
                prov["aggregates_recomputed"] = True
                prov["aggregate_blocks"] = sorted(want_blocks)
                if diffs:
                    rep.fail("SUMMARY_AGGREGATE", where, f"{len(diffs)} aggregate value(s) differ from a recomputation "
                                                         f"from the run-metrics cache: {short(diffs, 6)}")
            except Exception as e:  # noqa: BLE001
                rep.fail("SUMMARY_AGGREGATE", where, f"cannot recompute the aggregates: {type(e).__name__}: {e}")
        elif file_paths:
            rep.warn("SUMMARY_AGGREGATE_SKIPPED", where, "aggregates not recomputed: the run-metrics cache does not match "
                                                         "every run file (see SUMMARY_STALE)")
    elif file_paths:
        rep.warn("SUMMARY_NO_CACHE", where, f"no run_metrics cache at {cache_dir}: per-run provenance not verifiable, "
                                            f"falling back to modification times")
        newest = max(os.path.getmtime(p) for p in file_paths.values())
        if newest > out["mtime"]:
            rep.fail("SUMMARY_STALE", where, "a run file is newer than summary.json -- rerun analyze_ladder.py")
    out.update(per_N_backing=backed, n_per_seed_values=n_values, provenance=prov)
    return out, S


def compare_json(got, want, path, diffs, rtol=1e-12):
    """Recursive comparison of two JSON-safe structures (numbers within rtol; 'inf' strings; null = NaN)."""
    if isinstance(want, dict):
        if not isinstance(got, dict):
            diffs.append(f"{path}: missing / not an object")
            return
        for k in set(want) | set(got):
            if k not in got:
                diffs.append(f"{path}/{k}: missing from the summary")
            elif k not in want:
                diffs.append(f"{path}/{k}: not produced by the recomputation")
            else:
                compare_json(got[k], want[k], f"{path}/{k}", diffs, rtol)
        return
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            diffs.append(f"{path}: list differs ({got!r} vs {want!r})"[:200])
            return
        for i, (a, b) in enumerate(zip(got, want)):
            compare_json(a, b, f"{path}/{i}", diffs, rtol)
        return
    num_like = lambda v: v is None or isinstance(v, (int, float)) or v in ("inf", "-inf")  # noqa: E731
    if isinstance(want, bool) or isinstance(got, bool) or not (num_like(want) and num_like(got)):
        if got != want:
            diffs.append(f"{path}: {got!r} vs recomputed {want!r}"[:200])
        return
    if not same(got, want, rtol):
        diffs.append(f"{path}: {got!r} vs recomputed {want!r}")


def audit_cis(S, P, rep, D):
    """Every '*ci95*' key of the summary: its seed count vs the planned seeds (full / flagged)."""
    planned = len(P["seeds"])
    found = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if "ci95" in str(k):
                    found.append((path + [str(k)], v, o))
                else:
                    walk(v, path + [str(k)])
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, path + [str(i)])

    walk(S, [])
    per_N = S.get("per_N") or {}
    out = []
    for p, val, parent in found:
        pstr = "/".join(p)
        undefined = not (isinstance(val, list) and len(val) == 2 and all(v is not None for v in val))
        reasons, n, detail = [], None, {}
        if len(p) == 5 and p[0] == "contrasts" and p[4] == "G_ci95" and nkey(p[1]) is not None:
            N = nkey(p[1])
            common = set(D.get((N, "abf"), [])) & set(D.get((N, "fr"), []))
            tau = "n_both_finite" in parent
            n = parent.get("n_both_finite") if tau else parent.get("n")
            n_common = parent.get("n_common")
            if not isinstance(n, int) or not isinstance(n_common, int):
                rep.fail("CI_UNAUDITABLE", pstr, "CI without an integer seed count (n / n_both_finite, n_common)")
                continue
            if planned - len(common) > 0:
                reasons.append(f"{planned - len(common)} planned seed(s) without both arms on disk")
            if n_common - n > 0:
                reasons.append(f"{n_common - n} seed(s) excluded: " + (
                    "censored in at least one arm (the ratio uses both-finite seeds; ranks use all)" if tau
                    else "non-finite value or ABF value 0"))
            detail = dict(n_common=n_common)
        elif len(p) == 4 and p[0] == "best_allocation" and p[2] in ("abf", "fr") and p[3] == "boot_best_value_ci95":
            k, m = p[1], p[2]
            counts = {int(N): (((per_N.get(str(N)) or {}).get(m) or {}).get(k) or {}).get("n", 0)
                      for N in parent.get("Ns", [])}
            if not counts:
                rep.fail("CI_UNAUDITABLE", pstr, "best-allocation CI without candidate N")
                continue
            n = min(counts.values())
            short_N = {N: c for N, c in counts.items() if c < planned}
            if short_N:
                reasons.append(f"candidate N with fewer than {planned} usable seeds: {short_N}")
            plan_N = [int(x) for x in P["N_ladder"] if m == "abf" or int(x) >= 2]
            miss_N = sorted(set(plan_N) - set(counts))
            if miss_N:
                reasons.append(f"best N selected over a REDUCED ladder: planned N {miss_N} have no data")
            if (parent.get("boot_frac_censored") or 0) > 0:
                detail["boot_frac_censored"] = parent["boot_frac_censored"]
            if (parent.get("boot_frac_no_data") or 0) > 0:
                detail["boot_frac_no_data"] = parent["boot_frac_no_data"]
            detail["seeds_per_N"] = counts
        elif len(p) == 4 and p[0] == "best_allocation" and p[2] == "fr_minus_abf" and p[3] in ("ci95", "rel_ci95"):
            k = p[1]
            b = (S.get("best_allocation") or {}).get(k) or {}
            counts = {}
            for m in ("abf", "fr"):
                for N in (b.get(m) or {}).get("Ns", []):
                    counts[f"{m} N{N}"] = (((per_N.get(str(N)) or {}).get(m) or {}).get(k) or {}).get("n", 0)
            if not counts:
                rep.fail("CI_UNAUDITABLE", pstr, "best-allocation difference CI without candidate N")
                continue
            n = min(counts.values())
            short_N = {a: c for a, c in counts.items() if c < planned}
            if short_N:
                reasons.append(f"arms with fewer than {planned} usable seeds: {short_N}")
            miss = [f"{m} N{x}" for m in ("abf", "fr") for x in P["N_ladder"] if (m == "abf" or int(x) >= 2)
                    and f"{m} N{int(x)}" not in counts]
            if miss:
                reasons.append(f"best values selected over a REDUCED ladder: no data at {miss}")
            if parent.get("frac_resamples_undefined"):
                detail["frac_resamples_undefined"] = parent["frac_resamples_undefined"]
        else:
            rep.fail("CI_UNAUDITABLE", pstr, "confidence interval at a location the audit does not know; its seed "
                                             "count cannot be checked")
            continue
        if n > planned:
            rep.fail("CI_COUNT", pstr, f"n = {n} seeds exceeds the {planned} planned seeds")
        if undefined:
            reasons.append("CI undefined (no usable seed pairs)")
        status = "full" if (n == planned and not reasons) else "flagged"
        out.append(dict(path=pstr, n_seeds=n, planned=planned, status=status, reasons=reasons, ci=val, **detail))
    return dict(n_total=len(out), n_full=sum(c["status"] == "full" for c in out),
                n_flagged=sum(c["status"] == "flagged" for c in out),
                flagged=[c for c in out if c["status"] == "flagged"],
                full=[c["path"] for c in out if c["status"] == "full"])


# =================================================================================================== figures
def check_file(f, base):
    p = f if os.path.isabs(f) else os.path.normpath(os.path.join(base, f))
    if not os.path.isfile(p):
        return "FIG_MISSING", f"{p} does not exist", p
    sz = os.path.getsize(p)
    if sz == 0:
        return "FIG_EMPTY", f"{p} is empty (0 bytes)", p
    ext = os.path.splitext(p)[1].lower()
    with open(p, "rb") as fh:
        head = fh.read(4096)
    if ext in MAGIC and not head.startswith(MAGIC[ext]):
        return "FIG_CORRUPT", f"{p} does not start with a valid {ext} header", p
    if ext == ".svg" and b"<svg" not in head:
        return "FIG_CORRUPT", f"{p} has no <svg element in its first 4 kB", p
    return None, None, p


def load_manifest(path, rep, where):
    try:
        d = strict_json(path)
    except Exception as e:  # noqa: BLE001
        rep.fail("MANIFEST_INVALID", where, f"{path}: {type(e).__name__}: {e}")
        return None
    if not isinstance(d, dict) or not isinstance(d.get("figures"), list):
        rep.fail("MANIFEST_INVALID", where, f"{path}: no 'figures' list")
        return None
    return d


def is_placeholder(e):
    return bool(e.get("note")) or str(e.get("description", "")).strip().lower().startswith("placeholder")


def check_entries(entries, base, rep, where, n_files):
    """File checks of manifest entries; returns True when every listed file is present, non-empty and valid."""
    ok = True
    for e in entries:
        fl = e.get("files")
        if isinstance(fl, str):
            fl = [fl]
        if not isinstance(fl, list) or not all(isinstance(x, str) for x in fl):
            rep.fail("MANIFEST_INVALID", where, f"entry {e.get('tag', e.get('id'))!r}: 'files' is not a list of paths")
            ok = False
            continue
        if not fl and not is_placeholder(e):
            rep.fail("FIG_MISSING", where, f"entry {e.get('tag', e.get('id'))!r} lists no file")
            ok = False
        for f in fl:
            code, msg, _ = check_file(f, base)
            n_files[0] += 1
            if code:
                rep.fail(code, where, msg)
                ok = False
    return ok


def audit_figures(system, P, rep, fig_root, results_root, N_status, D, file_paths, summary_meta, fig_base=None):
    """Per-N manifests of plot_config.py and the synthesis manifest of plot_synthesis.py (section 6).  fig_base: the
    directory holding N<N>/ and synthesis/ (default <fig_root>/<out_dir>; mechanism cells: their fig_dir)."""
    out_dir = P["out_dir"]
    fig_base = fig_base or os.path.join(fig_root, out_dir)
    fixture = bool(P.get("_fixture_of"))
    out = dict(per_N={}, synthesis={}, n_files_checked=0)
    n_files = [0]
    for N in [int(n) for n in P["N_ladder"]]:
        req = N_status.get(N) == "complete"
        mp = os.path.join(fig_base, f"N{N}", "MANIFEST.json")
        rec = dict(required=req, manifest=mp, present=os.path.exists(mp), items={})
        out["per_N"][N] = rec
        if not rec["present"]:
            if req:
                rep.fail("MANIFEST_MISSING", f"N{N} figures", f"{mp} does not exist (scripts/equal_budget/plot_config.py "
                                                              f"--system {system})")
                rec["items"] = {name: "missing" for name, _ in PER_N_ITEMS}
            rec.update(n_ok=0, n_items=len(PER_N_ITEMS))
            continue
        man = load_manifest(mp, rep, f"N{N} figures")
        if man is None:
            rec.update(n_ok=0, n_items=len(PER_N_ITEMS))
            continue
        where = f"N{N} figures"
        fail_or_warn = rep.fail if req else rep.warn
        if man.get("system") not in (None, system) or man.get("system_dir", out_dir) != out_dir or man.get("N") != N:
            rep.fail("MANIFEST_MISMATCH", where, f"manifest is for system {man.get('system')!r} / {man.get('system_dir')!r}"
                                                 f" / N {man.get('N')!r}, not {system} / {out_dir} / N {N}")
        if "fixture" in man and bool(man["fixture"]) != fixture:
            rep.fail("MANIFEST_MISMATCH", where, f"figures drawn from a {'fixture' if man['fixture'] else 'production'} "
                                                 f"config, the audited config is {'a fixture' if fixture else 'production'}")
        if man.get("metrics_version") not in (None, M.METRICS_VERSION):
            fail_or_warn("FIG_STALE", where, f"figures drawn with metrics {man.get('metrics_version')!r}, current "
                                             f"{M.METRICS_VERSION!r}")
        seeds = man.get("seeds") or {}
        for m in (["abf"] if N == 1 else ["abf", "fr"]):
            drawn = sorted(int(s) for s in seeds.get(m, []))
            if drawn != D[(N, m)]:
                fail_or_warn("FIG_STALE", where, f"{m} figures drawn from seeds {drawn}, the disk has complete "
                                                 f"{D[(N, m)]} -- rerun plot_config.py")
        newest = max([os.path.getmtime(p) for (n, _, _), p in file_paths.items() if n == N] or [0.0])
        if newest > os.path.getmtime(mp):
            fail_or_warn("FIG_STALE", where, "a run file of this N changed after the figures were drawn")
        changed = []                      # content check (plot_config records each source's sha256)
        for src in man.get("sources") or []:
            key = (N, str(src.get("method")), int(src.get("seed", -1)))
            if src.get("src_sha256") and key in file_paths and M.sha256_file(file_paths[key]) != src["src_sha256"]:
                changed.append(f"s{key[2]}_{key[1]}")
        if changed:
            fail_or_warn("FIG_STALE", where, f"figures drawn from a different CONTENT of run file(s) {short(changed)} "
                                             f"-- rerun plot_config.py")
        entries = [e for e in man["figures"] if isinstance(e, dict)]
        base = os.path.dirname(mp)
        for name, rx in PER_N_ITEMS:
            match = [e for e in entries if re.match(rx, str(e.get("tag", "")))]
            if not match:
                if req:
                    rep.fail("FIG_MISSING", f"N{N} {name}", f"no figure in {mp} (tag pattern {rx})")
                rec["items"][name] = "missing"
                continue
            real = [e for e in match if not is_placeholder(e)]
            note_ok = (name, N) in NOTE_ALLOWED
            if not real and not note_ok:
                fail_or_warn("FIG_NOTE", f"N{N} {name}", f"only a placeholder / note in place of the figure "
                                                         f"({[e.get('tag') for e in match]})")
            good = check_entries(match, base, rep, f"N{N} {name}", n_files)
            rec["items"][name] = ("note" if not real and note_ok else "ok") if good and (real or note_ok) else "bad"
        listed = {e.get("tag") for e in entries}
        matched = {e.get("tag") for e in entries for _, rx in PER_N_ITEMS if re.match(rx, str(e.get("tag", "")))}
        extra = [e for e in entries if e.get("tag") in listed - matched]
        check_entries(extra, base, rep, where, n_files)                # supplementary figures: files must be sound
        rec.update(n_ok=sum(v in ("ok", "note") for v in rec["items"].values()), n_items=len(PER_N_ITEMS),
                   supplementary=sorted(str(t) for t in listed - matched))

    # ---- synthesis S1-S8: required as soon as any run is complete (like the summary)
    need = any(len(v) > 0 for v in D.values())
    cands = [os.path.join(fig_base, "synthesis", "MANIFEST.json")]
    if not MF.is_cell(P):          # a cell's results_dir may be the equal-budget tree (reuse): no fallback there
        cands.append(os.path.join(results_root, out_dir, "synthesis", "MANIFEST.json"))
    found = [c for c in dict.fromkeys(cands) if os.path.exists(c)]
    syn = dict(manifest=found[0] if found else cands[0], present=bool(found), items={})
    out["synthesis"] = syn
    if len(found) > 1:
        rep.warn("SYNTH_LOCATION", "synthesis", f"two synthesis manifests ({found}); auditing the one under --fig-root")
    elif found and len(cands) > 1 and found[0] == cands[1] and cands[0] != cands[1]:
        rep.warn("SYNTH_LOCATION", "synthesis", f"synthesis figures are under the results root ({found[0]}), not under "
                                                f"--fig-root (plot_synthesis.py's default --fig-root is the results root)")
    man = load_manifest(found[0], rep, "synthesis") if found else None
    if not found and need:
        rep.fail("MANIFEST_MISSING", "synthesis", f"{cands[0]} does not exist (scripts/equal_budget/plot_synthesis.py "
                                                  f"--system {system})")
    if man is not None:
        if man.get("system") not in (None, system):
            rep.fail("MANIFEST_MISMATCH", "synthesis", f"manifest is for system {man.get('system')!r}")
        sm = man.get("summary") or {}
        if summary_meta and summary_meta.get("present"):
            if sm.get("sha256") != summary_meta.get("sha256"):
                rep.fail("MANIFEST_STALE", "synthesis", f"drawn from summary sha256 {str(sm.get('sha256'))[:12]} "
                                                        f"(generated {sm.get('generated_utc')}), the current summary is "
                                                        f"{str(summary_meta.get('sha256'))[:12]} "
                                                        f"(generated {summary_meta.get('generated_utc')})")
        elif need:
            rep.fail("MANIFEST_STALE", "synthesis", "no current summary.json to compare the synthesis manifest with")
        fr = man.get("freshness") or {}
        if fr.get("stale"):
            rep.fail("MANIFEST_STALE", "synthesis", f"synthesis plotted from a STALE summary (--allow-stale): "
                                                    f"{fr.get('stale_reasons')}")
        pl = man.get("plan") or {}
        want = dict(B=int(P["B"]), h=float(P["engine_cfg"]["h"]), N_ladder=[int(n) for n in P["N_ladder"]],
                    seeds=[int(s) for s in P["seeds"]], thresholds=P["thresholds"])
        for k, v in want.items():
            if k in pl and pl[k] != v:
                rep.fail("MANIFEST_MISMATCH", "synthesis", f"synthesis plan {k} = {pl[k]!r} != config {v!r}")
        if "fixture" in man and bool(man["fixture"]) != fixture:
            rep.fail("MANIFEST_MISMATCH", "synthesis", "fixture / production flag differs from the audited config")
    entries = [e for e in (man or {}).get("figures", []) if isinstance(e, dict)]
    base = os.path.dirname(found[0]) if found else ""
    for k in SYNTHESIS_FIGURES:
        match = [e for e in entries if re.match(rf"^{k}(_|$)", str(e.get("id", e.get("tag", ""))))]
        if not match:
            if need:
                rep.fail("SYNTH_MISSING", k, "no synthesis figure with this id" + ("" if found else " (no manifest)"))
            syn["items"][k] = "missing" if need else "unlisted"
            continue
        real = [e for e in match if not is_placeholder(e)]
        if not real:
            (rep.fail if need else rep.warn)("FIG_NOTE", k, "only a placeholder in place of the synthesis figure")
        syn["items"][k] = "ok" if check_entries(match, base, rep, k, n_files) and real else "bad"
    ids = [str(e.get("id", e.get("tag"))) for e in entries]
    syn["ids"] = ids
    rest = [e for e in entries if not any(re.match(rf"^{k}(_|$)", str(e.get("id", ""))) for k in SYNTHESIS_FIGURES)]
    check_entries(rest, base, rep, "synthesis", n_files)
    out["n_files_checked"] = n_files[0]
    return out


# =================================================================================================== one system
def audit_system(system, cfg_path, results_root, analysis_root, fig_root):
    """results_root / analysis_root / fig_root None: the equal-budget defaults, or a mechanism cell's own locations."""
    rep = Report()
    t0 = time.time()
    P = json.load(open(cfg_path))
    MF.check_invocation(system, P, cfg_path)      # refused: a cell config as 'gateway', or gateway_family without one
    kind = MF.system_kind(system)
    out_dir = P["out_dir"]
    cell = MF.is_cell(P)
    if cell:
        results_root = results_root or MF.cell_results_root(P)
        fig_base = MF.cell_fig_dir(P, fig_root)
    else:
        results_root = results_root or os.path.join(ROOT, "results", "equal_budget_v2")
        analysis_root = analysis_root or results_root
        fig_root = fig_root or os.path.join(ROOT, "figures", "equal_budget_v2")
        fig_base = os.path.join(fig_root, out_dir)
    seeds = [int(s) for s in P["seeds"]]
    rows = A.plan(P)
    res_dir = os.path.join(results_root, out_dir)
    if kind == "lta" and float(P.get("T_K", MF.SYSTEMS[system].get("T_K"))) != float(MF.SYSTEMS[system]["T_K"]):
        rep.fail("PLAN_MISMATCH", "config", f"config T_K {P.get('T_K')} is not {system}'s {MF.SYSTEMS[system]['T_K']}")
    for k in ("profile_snapshot_u", "physical_checkpoints_t", "thresholds", "seeds", "N_ladder"):
        if not P.get(k):
            rep.fail("PLAN_MISMATCH", "config", f"config lacks {k!r}")
    stat = load_status_json(os.path.join(res_dir, "STATUS.json"), rep, "STATUS.json")
    stat_top = {}
    if stat is not None:
        try:
            stat_top = strict_json(os.path.join(res_dir, "STATUS.json"))
        except Exception:  # noqa: BLE001  (already reported)
            stat_top = {}
    if cell:      # the cell's ledger (production: one per experiment, rows of this variant only)
        ledger_path, ledger_where = MF.cell_ledger(P)
    else:
        ledger_path, ledger_where = os.path.join(res_dir, "ledger.csv"), None
    ledger = load_ledger(ledger_path, ledger_where)
    gate = MF.reuse_gate_status(P) if cell else None
    if gate and gate["reuse"] and not gate["licensed"]:
        rep.fail("REUSE_GATE", "reuse", "the cell reuses results/equal_budget_v2 files but the plan's bitwise-reuse gate "
                                        "(section 5) is not established: " + "; ".join(gate["problems"]))

    # ---- run files
    files, D, file_paths, eng, cfgs, unident = {}, {}, {}, {}, {}, []
    for r in rows:
        N = r["N"]
        grid = expected_grid(system, P, r["n_steps"])
        for m in r["methods"]:
            ok = []
            for s in seeds:
                p = A.run_path(results_root, P, N, s, m)
                rel = os.path.relpath(p, res_dir)
                if os.path.exists(p):
                    st, probs, info = audit_run_file(p, system, P, N, s, m, grid)
                    cfg_run = info.pop("_cfg", None)
                    for code, msg in probs:
                        rep.fail(code, rel, msg)
                    if st == "complete":
                        ok.append(s)
                        file_paths[(N, m, s)] = p
                        cfgs[(N, m, s)] = cfg_run
                        e = (info.get("engine"), info.get("engine_sha256"))
                        eng[e] = eng.get(e, 0) + 1
                        if not info.get("engine_sha256"):
                            unident.append(rel)
                        if m == "fr" and info.get("n_deaths", 1) == 0:
                            rep.warn("FR_NO_EVENTS", rel, f"FR arm realised no death in {info.get('n_opps')} "
                                                          f"opportunities: identical to the ABF arm")
                    files[(N, m, s)] = dict(status=st, problems=[c for c, _ in probs], **info)
                elif os.path.exists(p + ".ckpt.npz"):
                    files[(N, m, s)] = dict(status="running")
                else:
                    files[(N, m, s)] = dict(status="missing")
            D[(N, m)] = sorted(ok)
    if len(eng) > 1:
        builds = {f"{a} {str(b or '')[:12]}": c for (a, b), c in eng.items()}
        eb = stat_top.get("engine_builds") if isinstance(stat_top, dict) else None
        why = (eb.get("reason") if isinstance(eb, dict) else eb) if eb else None
        if why and str(why).strip():
            rep.warn("ENGINE_MIXED", "run files", f"complete runs come from {len(eng)} engine builds {builds}; documented "
                                                  f"in STATUS.json: {why}")
        else:
            rep.fail("ENGINE_MIXED", "run files", f"complete runs come from {len(eng)} engine builds {builds}; a ladder "
                                                  f"must be one build unless STATUS.json documents it (engine_builds)")
    if unident:
        rep.warn("ENGINE_UNIDENTIFIED", "run files", f"{len(unident)} run file(s) record no engine_sha256 (the build "
                                                     f"that produced them cannot be identified): {short(unident)}")
    # ---- one cfg for the whole ladder (physics defaults outside the config included)
    if cfgs:
        k0 = sorted(cfgs)[0]
        ref_cfg = cfgs[k0]
        for key, c in sorted(cfgs.items()):
            if c is None or ref_cfg is None:
                continue
            diff = cfg_differences(ref_cfg, c)
            if diff:
                rep.fail("CFG_INCONSISTENT", f"N{key[0]}/s{key[2]}_{key[1]}.npz",
                         f"cfg_json differs from N{k0[0]}/s{k0[2]}_{k0[1]}.npz in {short(diff)}: "
                         + "; ".join(f"{d} {ref_cfg.get(d, '<absent>')!r} -> {c.get(d, '<absent>')!r}" for d in diff[:4]))
    # ---- pairing of the two arms of every (N, seed)
    pairing = dict(n_pairs=0, n_saves_compared=0, meta_only=[])
    for r in rows:
        if "fr" not in r["methods"]:
            continue
        N = r["N"]
        for s in sorted(set(D[(N, "abf")]) & set(D[(N, "fr")])):
            pa, pf = file_paths[(N, "abf", s)], file_paths[(N, "fr", s)]
            probs, n = pairing_problems(kind, pa, pf, files[(N, "abf", s)], files[(N, "fr", s)])
            pairing["n_pairs"] += 1
            pairing["n_saves_compared"] += n
            if n == 0:
                pairing["meta_only"].append(f"N{N} s{s}")
            for msg in probs:
                rep.fail("PAIRING", f"N{N} s{s}", msg)
    planned_keys = {(r["N"], s, m) for r in rows for m in r["methods"] for s in seeds}
    extra = A.unplanned_files(results_root, P, planned_keys)
    if extra:
        rep.warn("UNPLANNED_FILES", out_dir, f"{len(extra)} result files not in the plan: {short(extra)}")

    # ---- N status vs STATUS.json and ledger
    N_status, N_rows = {}, []
    for r in rows:
        N = r["N"]
        cnt = {m: dict(complete=len(D[(N, m)]), planned=len(seeds),
                       running=sum(files[(N, m, s)]["status"] == "running" for s in seeds),
                       invalid=sum(files[(N, m, s)]["status"] == "invalid" for s in seeds),
                       missing=sum(files[(N, m, s)]["status"] == "missing" for s in seeds)) for m in r["methods"]}
        n_planned = len(seeds) * len(r["methods"])
        n_done = sum(c["complete"] for c in cnt.values())
        n_run = sum(c["running"] for c in cnt.values())
        obs = "complete" if n_done == n_planned else ("NOT RUN" if n_done == 0 else "partial")
        miss_set = {(m, s) for m in r["methods"] for s in seeds if files[(N, m, s)]["status"] != "complete"}
        e = (stat or {}).get(N)
        documented = False
        if obs == "complete":
            if e and e["status"] not in (None, "complete"):
                rep.fail("STATUS_CONTRADICTION", f"N{N}", f"STATUS.json says {e['status']!r} but all {n_planned} "
                                                          f"planned runs are complete")
        else:
            if not e or not e["reason"]:
                rep.fail("N_UNEXPLAINED", f"N{N}", f"{obs}: {n_done}/{n_planned} planned runs complete"
                                                   f"{f', {n_run} running' if n_run else ''}; no reason in STATUS.json")
            elif e["status"] == "running":
                rep.fail("STATUS_NOT_FINAL", f"N{N}", "STATUS.json says 'running': the audit is not final")
            elif e["status"] == "complete":
                rep.fail("STATUS_CONTRADICTION", f"N{N}", f"STATUS.json says 'complete', the disk has {n_done}/"
                                                          f"{n_planned}")
            elif e["status"] == "NOT RUN" and n_done > 0:
                rep.fail("STATUS_CONTRADICTION", f"N{N}", f"STATUS.json says 'NOT RUN' but {n_done} runs are complete")
            elif e["status"] == "partial" and n_done == 0:
                rep.fail("STATUS_CONTRADICTION", f"N{N}", "STATUS.json says 'partial' but no run is complete")
            elif e["status"] is not None and e["status"].startswith("?"):
                rep.fail("STATUS_JSON_INVALID", f"N{N}", f"unknown status {e['status'][1:]!r}")
            else:
                documented = True
                ms = status_missing_set(e.get("missing"))
                if ms == "unreadable":
                    rep.fail("STATUS_JSON_INVALID", f"N{N}", "'missing' is neither {method: [seeds]} nor pairs")
                elif ms is not None and ms != miss_set:
                    rep.fail("STATUS_CONTRADICTION", f"N{N}", f"STATUS.json missing {sorted(ms)} != not-complete on "
                                                              f"disk {sorted(miss_set)}")
            if n_run and documented:
                rep.warn("CHECKPOINT_LEFT", f"N{N}", f"{n_run} checkpoint(s) of unfinished jobs remain")
        if ledger is not None:
            dele = status_missing_set((e or {}).get("deleted")) if e else None
            dele = dele if isinstance(dele, set) else set()
            for m in r["methods"]:
                for s in seeds:
                    row = ledger.get((N, s, m))
                    fst = files[(N, m, s)]["status"]
                    where = f"N{N} s{s} {m}"
                    if row is not None and row.get("status") == "complete" and fst != "complete":
                        named = (m, s) in dele and bool((e or {}).get("reason"))
                        (rep.warn if named else rep.fail)(
                            "LEDGER_CLAIM", where, f"ledger.csv records 'complete' ({row.get('finished_utc')}) but the "
                                                   f"file is {fst}" + ("; STATUS.json names it as deleted" if named else
                                                                       "; a completed run may not become NOT RUN unless "
                                                                       "STATUS.json lists it under 'deleted' with a reason"))
                    elif fst == "complete" and row is None:
                        rep.warn("LEDGER_GAP", where, "complete file without a ledger row")
                    elif fst == "complete" and row.get("status") != "complete":
                        rep.warn("LEDGER_GAP", where, f"complete file, last ledger status {row.get('status')!r}")
                    elif (fst == "complete" and row.get("n_force_evals") not in (None, "")
                          and int(float(row["n_force_evals"])) != files[(N, m, s)].get("n_force_evals")):
                        rep.warn("LEDGER_GAP", where, f"ledger n_force_evals {row['n_force_evals']} != file "
                                                      f"{files[(N, m, s)].get('n_force_evals')}")
        N_status[N] = obs
        N_rows.append(dict(N=N, n_steps=r["n_steps"], T=r["T"], status=obs, documented=documented,
                           reason=(e or {}).get("reason") if e else None,
                           status_json=(e or {}).get("status") if e else None, methods=cnt))
    if ledger is not None:
        for (N, s, m) in sorted(set(ledger) - planned_keys):
            rep.warn("LEDGER_UNPLANNED", f"N{N} s{s} {m}", "ledger row for a job outside the plan")
    elif any(v["status"] != "missing" for v in files.values()):
        rep.fail("LEDGER_MISSING", out_dir, f"{ledger_path} does not exist although run files do "
                                            f"(run_ladder.py / scripts/mechanism/run_cells.py write one row per finished job)")
    for N in sorted(set(stat or {}) - {r["N"] for r in rows}):
        rep.warn("STATUS_UNPLANNED", f"N{N}", "STATUS.json entry for an N outside the plan")
    any_complete = any(D.values())

    # ---- summary, CIs, references, figures
    adir = resolve_analysis_dir(analysis_root, out_dir, system, P)
    summ, S = audit_summary(system, P, rep, adir, results_root, D, file_paths, any_complete)
    cis = audit_cis(S, P, rep, D) if S is not None else dict(n_total=0, n_full=0, n_flagged=0, flagged=[], full=[])
    refs = audit_reference(system, P, rep, S)
    figs = audit_figures(system, P, rep, fig_root, results_root, N_status, D, file_paths, summ, fig_base=fig_base)
    for row in N_rows:
        f = figs["per_N"].get(row["N"], {})
        row["figures"] = dict(required=f.get("required"), present=f.get("present"), n_ok=f.get("n_ok"),
                              n_items=f.get("n_items"))
    n_files = sum(1 for v in files.values() if v["status"] in ("complete", "invalid"))
    out = dict(
        system=system, kind=kind, verdict="PASS" if not rep.failures else "FAIL", config=os.path.abspath(cfg_path),
        config_sha256=sha256_file(cfg_path), out_dir=out_dir, results_dir=res_dir, analysis_dir=adir,
        figure_root=fig_base, synthesis_manifest=figs["synthesis"]["manifest"],
        n_failures=len(rep.failures), n_warnings=len(rep.warnings),
        plan=dict(B=int(P["B"]), h=float(P["engine_cfg"]["h"]), N_ladder=[r["N"] for r in rows], seeds=seeds,
                  n_planned_runs=len(planned_keys), profile_snapshot_u=P["profile_snapshot_u"],
                  physical_checkpoints_t=P["physical_checkpoints_t"]),
        status_json=os.path.join(res_dir, "STATUS.json") if stat is not None else None,
        N=N_rows,
        run_files=dict(n_present=n_files, n_complete=len(file_paths), n_invalid=n_files - len(file_paths),
                       n_running=sum(v["status"] == "running" for v in files.values()),
                       n_missing=sum(v["status"] == "missing" for v in files.values()),
                       required_arrays=[k for k, _, _ in (GATEWAY_ARRAYS if kind == "gateway" else LTA_ARRAYS)],
                       optional_arrays=[k for k, _, _ in (GATEWAY_OPTIONAL if kind == "gateway" else LTA_OPTIONAL)],
                       engines={f"{a} {b or ''}".strip(): c for (a, b), c in eng.items()}, pairing=pairing,
                       files={f"N{N}/s{s}_{m}.npz": v for (N, m, s), v in sorted(files.items())}),
        references=refs, summary=summ, confidence_intervals=cis, figures=figs,
        failures=rep.failures, warnings=rep.warnings, elapsed_s=time.time() - t0)
    if cell:
        out["cell"] = {k: P.get(k) for k in ("experiment", "cell", "cell_label", "model", "engine_version", "results_dir",
                                              "analysis_dir", "fig_dir", "reference_file", "reuse")}
        out["cell"].update(ledger=ledger_path, ledger_filter=ledger_where, reuse_gate=gate)
    return out


# =================================================================================================== printing
def print_system(a, max_lines=40):
    pr = print
    where = (f"cell {a['cell']['experiment']}/{a['cell']['cell']}, runs in {a['out_dir']}" if a.get("cell")
             else a["out_dir"])
    pr(f"\n== {a['system']} ({where}): {a['verdict']}  [{a['n_failures']} failure(s), {a['n_warnings']} "
       f"warning(s)]")
    pr(f"   config {a['config']}")
    pr(f"   results {a['results_dir']}   analysis {a['analysis_dir']}   figures {a['figure_root']}")
    hdr = f"   {'N':>5} {'n_steps':>13} {'T_N':>9}  {'status':<9} {'ABF':>6} {'FR':>6} {'run':>4} {'inv':>4}  " \
          f"{'figures':<10} reason / note"
    pr(hdr)
    pr("   " + "-" * (len(hdr) - 3))
    for r in a["N"]:
        mm = r["methods"]
        ab = f"{mm['abf']['complete']}/{mm['abf']['planned']}"
        fr = f"{mm['fr']['complete']}/{mm['fr']['planned']}" if "fr" in mm else "n/a"
        run = sum(c["running"] for c in mm.values())
        inv = sum(c["invalid"] for c in mm.values())
        f = r["figures"]
        fig = (f"{f['n_ok']}/{f['n_items']}" if f.get("required") else
               (f"({f['n_ok']} drawn)" if f.get("present") else "n/a"))
        note = ""
        if r["status"] != "complete":
            note = (f"STATUS.json {r['status_json'] or ''}: {r['reason']}" if r["documented"]
                    else "NO REASON (STATUS.json)")
        pr(f"   {r['N']:>5} {r['n_steps']:>13,} {r['T']:>9g}  {r['status']:<9} {ab:>6} {fr:>6} {run:>4} {inv:>4}  "
           f"{fig:<10} {note}")
    rf = a["run_files"]
    pr(f"   run files: {rf['n_present']} present, {rf['n_complete']} valid, {rf['n_invalid']} invalid, "
       f"{rf['n_running']} running, {rf['n_missing']} missing; {len(rf['required_arrays'])} required arrays per file; "
       f"engines {rf['engines']}")
    ref = a["references"]
    units = (f"F {ref.get('units')!r} (F' implicit)" if a["kind"] == "gateway"
             else f"F {ref.get('units_F')!r}, F' {ref.get('units_gamma')!r}")
    pr(f"   reference: {ref['path']} sha256 {str(ref.get('sha256'))[:12]}; units {units}; gauge {ref.get('gauge')!r}")
    s = a["summary"]
    if s.get("present"):
        pv = s.get("provenance", {})
        pr(f"   summary: {s['path']} (generated {s.get('generated_utc')}, n_boot {s.get('n_boot')}); "
           f"{s.get('n_per_seed_values', 0)} per-seed values; provenance {pv.get('runs_checked', 0)} runs / "
           f"{pv.get('values_checked', 0)} values verified against the run-metrics cache")
    else:
        pr(f"   summary: {s['path']} ABSENT")
    c = a["confidence_intervals"]
    pr(f"   confidence intervals: {c['n_total']} in summary.json, {c['n_full']} at the planned "
       f"{len(a['plan']['seeds'])} seeds, {c['n_flagged']} flagged")
    for x in c["flagged"][:6]:
        pr(f"      flagged {x['path']}: n {x['n_seeds']}/{x['planned']}; {'; '.join(x['reasons'])}")
    if c["n_flagged"] > 6:
        pr(f"      ... {c['n_flagged'] - 6} more flagged CIs in the JSON")
    syn = a["figures"]["synthesis"]
    pr(f"   synthesis figures ({syn['manifest']}): " + ", ".join(f"{k} {v}" for k, v in syn["items"].items())
       + f"; {a['figures']['n_files_checked']} figure files checked")
    if a["failures"]:
        pr(f"   FAILURES ({len(a['failures'])}):")
        for x in a["failures"][:max_lines]:
            pr(f"      [{x['code']}] {x['where']}: {x['message']}")
        if len(a["failures"]) > max_lines:
            pr(f"      ... {len(a['failures']) - max_lines} more in the JSON")
    if a["warnings"]:
        pr(f"   warnings ({len(a['warnings'])}):")
        for x in a["warnings"][:15]:
            pr(f"      [{x['code']}] {x['where']}: {x['message']}")
        if len(a["warnings"]) > 15:
            pr(f"      ... {len(a['warnings']) - 15} more in the JSON")


# =================================================================================================== main
def parse_configs(items, systems):
    out = {s: (None if MF.is_family(s) else MF.default_config_path(s)) for s in systems}
    for it in items or []:
        if "=" in it and it.split("=", 1)[0] in MF.SYSTEMS:
            s, p = it.split("=", 1)
            if s not in systems:
                raise SystemExit(f"--config {it}: {s} is not audited (--system)")
            out[s] = p
        elif len(systems) == 1:
            out[systems[0]] = it
        else:
            raise SystemExit("with several systems give --config SYSTEM=PATH")
    miss = [s for s, p in out.items() if p is None]
    if miss:
        raise SystemExit(f"--system {miss[0]} needs --config <cell config> (configs/mechanism/cells/<experiment>/<variant>.json)")
    return out


def run_audit(systems, configs, results_root, analysis_root, fig_root, out_path, verbose=True, argv=None):
    results = {}
    for s in systems:
        results[s] = audit_system(s, configs[s], results_root, analysis_root, fig_root)
        if verbose:
            print_system(results[s])
    verdict = "PASS" if all(a["verdict"] == "PASS" for a in results.values()) else "FAIL"
    doc = dict(schema=SCHEMA, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), argv=argv,
               verdict=verdict, systems_audited=list(systems), results_root=results_root, analysis_root=analysis_root,
               fig_root=fig_root, references_dir=M.REF_DIR, metrics_version=M.METRICS_VERSION,
               scope=("covers exactly systems_audited, measured at generated_utc; the final completeness audit is "
                      "--system all"),
               required_figures_per_N={name: rx for name, rx in PER_N_ITEMS},
               note_allowed=[f"{f} at N = {n}" for f, n in sorted(NOTE_ALLOWED)],
               synthesis_figures=list(SYNTHESIS_FIGURES), systems=results)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + f".tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(M.json_safe(doc), fh, indent=1, allow_nan=False)
    os.replace(tmp, out_path)
    if verbose:
        print(f"\nCOMPLETENESS AUDIT {verdict}: " + ", ".join(f"{s} {a['verdict']} ({a['n_failures']} failures)"
                                                         for s, a in results.items()))
        print(f"wrote {out_path}")
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=list(SYSTEM_ORDER) + ["gateway_family", "all"])
    ap.add_argument("--results-root", default=None, help="default results/equal_budget_v2 (a cell: its own)")
    ap.add_argument("--config", action="append", default=None,
                    help="config path (one system) or SYSTEM=PATH (repeatable); default the production configs")
    ap.add_argument("--fig-root", default=None, help="default figures/equal_budget_v2 (a cell: its fig_dir)")
    ap.add_argument("--analysis-root", default=None, help="default: --results-root (a cell: its analysis_dir)")
    ap.add_argument("--out", default=None, help="default: <analysis-root>/completeness_audit.json (a cell: "
                                                "<analysis_dir>/completeness_audit.json)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    systems = list(SYSTEM_ORDER) if a.system == "all" else [a.system]
    configs = parse_configs(a.config, systems)
    if MF.is_family(a.system):        # mechanism cell: defaults from the cell config
        P = json.load(open(configs[a.system]))
        try:
            MF.check_invocation(a.system, P, configs[a.system])
        except M.MetricsError as e:
            raise SystemExit(str(e))
        results_root = os.path.abspath(a.results_root) if a.results_root else None
        analysis_root = os.path.abspath(a.analysis_root) if a.analysis_root else None
        fig_root = os.path.abspath(a.fig_root) if a.fig_root else None
        out = os.path.abspath(a.out or os.path.join(resolve_analysis_dir(analysis_root, P["out_dir"], a.system, P),
                                                    "completeness_audit.json"))
        MF.guard_output(P, out, a.system)
    else:

        results_root = os.path.abspath(a.results_root or os.path.join(ROOT, "results", "equal_budget_v2"))
        analysis_root = os.path.abspath(a.analysis_root or results_root)
        fig_root = os.path.abspath(a.fig_root or os.path.join(ROOT, "figures", "equal_budget_v2"))
        out = os.path.abspath(a.out or os.path.join(analysis_root, "completeness_audit.json"))
    try:
        doc = run_audit(systems, configs, results_root, analysis_root, fig_root, out, verbose=not a.quiet,
                        argv=list(sys.argv if argv is None else argv))
    except M.MetricsError as e:          # e.g. a cell config audited as an equal-budget system
        raise SystemExit(f"audit refused: {e}")
    return 0 if doc["verdict"] == "PASS" else 1



if __name__ == "__main__":
    sys.exit(main())
