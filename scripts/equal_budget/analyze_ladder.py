#!/usr/bin/env python
"""Analysis of an equal-budget replica ladder (docs/equal_budget/SCIENTIFIC_PLAN.md, section 4 metrics).

    python scripts/equal_budget/analyze_ladder.py --system gateway|lta300|lta150 \
        [--results-root PATH] [--config PATH] [--out PATH] [--n-boot 10000] [--skip-invalid]

--results-root  directory that holds <out_dir>/N<N>/s<seed>_<abf|fr>.npz (default results/equal_budget_v2)
--config        ladder config (default configs/equal_budget_v2/<system>_production.json)
--out           output directory (default <results-root>/<out_dir>/analysis)

Per run (eqb_metrics.run_metrics, cached in <out>/run_metrics/N<N>/s<seed>_<method>.{npz,json}; an entry is reused
only when its key (eqb_metrics.run_cache_key: the run file's CONTENT sha256, the metrics version, the config digest of
every engine knob / threshold / grid, the sha256 of the scoring code and of the frozen reference) equals the current
one, and eqb_metrics.check_plan is re-run on the file's meta / cfg at every cache hit).  Then per (N, method) medians /
IQRs / per-seed values; paired FR-vs-ABF contrasts at each N (same seed) G = (FR - ABF)/ABF with a 10 000-resample
seed bootstrap and wins; censoring-aware tau comparisons (thresholds below a hard reference floor are marked
'unreachable by construction': both censored there is not a tie); the max transient improvement of the median curves
(n = 0 when no seed has both arms); best allocation with re-selection of the best N in every bootstrap resample
(ties share the credit; the planned candidate N and the N missing from the candidate set are recorded); the FR
activity per N (realised deaths; FR arms with zero deaths are FR-inactive: G = 0 there is not equivalence); the null
calibration of the establishment criterion per N; the status of every planned (N, method, seed).
Writes <out>/summary.json (strict JSON: +inf = the string "inf", i.e. censored; NaN = null), <out>/tables.md and
<out>/median_curves.npz (median / q25 / q75 over seeds on the 200 uniform budget fractions).
A result that violates the engine contract stops the analysis with a clear message (--skip-invalid records it
as invalid instead).

Mechanism cells (docs/mechanism/SCIENTIFIC_PLAN.md):
    python scripts/equal_budget/analyze_ladder.py --system gateway_family \
        --config configs/mechanism/cells/<experiment>/<variant>.json [--out PATH]
The cell config names its own locations: --results-root defaults to the parent of its results_dir, --out to its
analysis_dir; an output inside results/ or figures/equal_budget_v2 is refused (eqb_family.guard_output).  A cell
config is accepted only with --system gateway_family and gateway_family only with a cell config
(eqb_family.check_invocation).  The family code lives in eqb_family.py (eqb_metrics.py is byte-identical to the
committed file, so the equal-budget code digest and every committed equal-budget analysis stay valid); the scorer is
eqb_family.GatewayFamilyScorer.  tau (plan section 6): the 'tau' contrasts of a cell are tau on e_F, on the
floor-free e_Fp_stat (thresholds.e_Fp_stat) and on TV_half; 'tau_secondary' holds tau on e_F_em and on raw e_Fp /
e_Fp_em at the thresholds above their hard floor only (raw e_F' is reported; no tau below its 0.03227 floor).
summary.json carries a 'cell' block (incl. the reuse-gate status of a reuse cell).  Equal-budget outputs are unchanged.
"""
from __future__ import annotations

import argparse
import functools
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
import run_ladder as RL  # noqa: E402  (the production driver's save grid: establishment null calibration per N)

ROOT = M.ROOT
SCHEMA = "eqb_ladder_summary/2"
PRIMARY = ["Ibar_F", "Ibar_Fp", "final_e_F", "final_e_Fp"]
# secondary read-outs; the gateway's *_Fp_stat are the labelled floor-free companions of e_F' (descriptive)
SECONDARY = {"gateway": ["Ibar_F_em", "Ibar_Fp_em", "final_e_F_em", "final_e_Fp_em", "Ibar_Fp_stat", "final_e_Fp_stat"],
             "lta": ["Ibar_Fp_proj", "final_e_Fp_proj"]}
MARGINAL = ["Ibar_TV_half", "final_TV_half", "Ibar_TV_inst", "final_TV_inst"]
TAU_PRIMARY = ([f"tau_e_F_{n}_u" for n in M.THR_NAMES] + [f"tau_e_Fp_{n}_u" for n in M.THR_NAMES] + ["tau_TV_half_u"])
TAU_SECONDARY = {"gateway": [f"tau_e_F_em_{n}_u" for n in M.THR_NAMES] + [f"tau_e_Fp_em_{n}_u" for n in M.THR_NAMES],
                 "lta": [f"tau_e_Fp_proj_{n}_u" for n in M.THR_NAMES]}


def tau_keys(system, P, floors=None):
    """(primary, secondary) tau contrast keys: TAU_PRIMARY and TAU_SECONDARY[kind] for every equal-budget system
    (unchanged); a mechanism cell: eqb_family.tau_groups (plan section 6; ``floors`` = threshold_floors drops raw
    e_F' tau below its hard floor)."""
    g = MF.tau_groups(system, P, floors)
    return g if g is not None else (TAU_PRIMARY, TAU_SECONDARY[M.system_kind(system)])


CURVE_KEYS = {"gateway": ["e_F", "e_Fp", "e_F_em", "e_Fp_em", "e_Fp_stat", "TV_half", "TV_inst", "far_frac"],
              "lta": ["e_F", "e_Fp", "e_Fp_proj", "TV_half", "TV_inst", "far_frac"]}
TRANSIENT_KEYS = {"gateway": ["e_F", "e_Fp", "e_F_em", "TV_half"], "lta": ["e_F", "e_Fp", "e_Fp_proj", "TV_half"]}
# tau compared ACROSS N uses the common budget grid only (tau_bgrid_*: identical u grid at every N)
BEST_PRIMARY = ["Ibar_F", "final_e_F", "tau_bgrid_e_F_mid_u"]
BEST_SECONDARY = ["Ibar_Fp", "final_e_Fp", "tau_bgrid_TV_half_u"]
FILE_RE = re.compile(r"^s(\d+)_(abf|fr)\.npz$")
NON_STAT = {"system", "kind", "N", "seed", "method", "n_steps", "h", "T", "B", "metrics_version"}


# --------------------------------------------------------------------------------------------- plan / status
def plan(P):
    B = int(P["B"])
    out = []
    for N in P["N_ladder"]:
        N = int(N)
        n_steps = B // N
        if n_steps * N != B:
            raise M.MetricsError(f"B {B} is not divisible by N {N}")
        out.append(dict(N=N, n_steps=n_steps, T=n_steps * float(P["engine_cfg"]["h"]),
                        methods=["abf"] if N == 1 else ["abf", "fr"]))
    return out


def run_path(res_root, P, N, seed, method):
    return os.path.join(res_root, P["out_dir"], f"N{N}", f"s{seed}_{method}.npz")


def file_status(path):
    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as z:
                st = json.loads(str(z["meta_json"])).get("status")
        except Exception as e:  # noqa: BLE001
            return "invalid", f"unreadable: {type(e).__name__}: {e}"
        return ("complete", None) if st == "complete" else ("invalid", f"meta status {st!r}")
    if os.path.exists(path + ".ckpt.npz"):
        return "running", None
    return "missing", None


def unplanned_files(res_root, P, planned):
    d = os.path.join(res_root, P["out_dir"])
    extra = []
    if not os.path.isdir(d):
        return extra
    for sub in sorted(os.listdir(d)):
        m = re.match(r"^N(\d+)$", sub)
        if not m or not os.path.isdir(os.path.join(d, sub)):
            continue
        for f in sorted(os.listdir(os.path.join(d, sub))):
            mm = FILE_RE.match(f)
            if mm and (int(m.group(1)), int(mm.group(1)), mm.group(2)) not in planned:
                extra.append(os.path.join(sub, f))
    return extra


# --------------------------------------------------------------------------------------------- per-run cache
def provenance(system, P, config_path=None):
    """Scoring-code and reference provenance of this analysis (also the cache-key ingredients)."""
    ref = MF.reference_path(system, P)
    return dict(code=MF.code_provenance(system), reference_path=os.path.relpath(ref, ROOT),
                reference_sha256=M.sha256_file(ref), config_digest=MF.config_digest(P),
                config_path=(os.path.abspath(config_path) if config_path else None),
                config_sha256=(M.sha256_file(config_path) if config_path else None))


def cache_paths(cache_dir, N, seed, method):
    stem = os.path.join(cache_dir, f"N{N}", f"s{seed}_{method}")
    return stem + ".json", stem + ".npz"


def read_meta_cfg(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta_json"])), json.loads(str(z["cfg_json"]))


def cache_entry_check(path, system, P, cache_dir, expect, prov):
    """(current key, stored blob or None, problems): problems = [] iff the cached entry of ``path`` is valid."""
    key = MF.run_cache_key(path, system, P, code=prov["code"], ref_sha=prov["reference_sha256"])
    jp, npz = cache_paths(cache_dir, expect["N"], expect["seed"], expect["method"])
    if not (os.path.exists(jp) and os.path.exists(npz)):
        return key, None, ["no cache entry"]
    try:
        with open(jp) as fh:
            blob = json.load(fh)
    except Exception as e:  # noqa: BLE001
        return key, None, [f"unreadable cache entry: {type(e).__name__}: {e}"]
    return key, blob, M.cache_key_problems(blob.get("key"), key)


def cached_metrics(path, system, P, get_scorer, cache_dir, expect, prov):
    key, blob, probs = cache_entry_check(path, system, P, cache_dir, expect, prov)
    jp, npz = cache_paths(cache_dir, expect["N"], expect["seed"], expect["method"])
    if blob is not None and not probs:
        meta, cfg = read_meta_cfg(path)                 # the plan is checked at EVERY use, cached or not
        if meta.get("status") != "complete":
            raise M.MetricsError(f"{path}: meta status is {meta.get('status')!r}, not 'complete'")
        MF.check_plan(dict(meta=meta, cfg=cfg), P, path, expect)
        try:
            with np.load(npz, allow_pickle=False) as z:
                curves = {k: z[k] for k in z.files}
            return curves, blob["scalars"], True
        except Exception:  # noqa: BLE001  (a damaged cache entry is recomputed)
            pass
    res = MF.load_run(path, system)
    MF.check_plan(res, P, path, expect)
    curves, sc = MF.run_metrics(res, system, P, get_scorer(), path)
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    tmp = npz + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **curves)
    os.replace(tmp, npz)
    tmpj = jp + f".tmp{os.getpid()}"
    with open(tmpj, "w") as fh:            # python JSON (Infinity kept): internal cache
        json.dump(dict(key=key, file=M.file_key(path), scalars=sc), fh, indent=0)
    os.replace(tmpj, jp)
    return curves, sc, False


# --------------------------------------------------------------------------------------------- aggregation
def numeric(v):
    return isinstance(v, (bool, int, float)) and not isinstance(v, str)


def aggregate(P, system, runs, rows, floors=None):
    kind = MF.system_kind(system)
    tau_prim, tau_sec = tau_keys(system, P, floors)
    per_N, contrasts, transient, curves_out = {}, {}, {}, {}
    u_grid = np.arange(1, M.N_UNIFORM + 1) / M.N_UNIFORM
    for r in rows:
        N = r["N"]
        blk = dict(n_steps=r["n_steps"], T=r["T"])
        for m in r["methods"]:
            sel = {s: runs[(N, m, s)][1] for s in P["seeds"] if (N, m, s) in runs}
            if not sel:
                blk[m] = None
                continue
            keys = [k for k, v in next(iter(sel.values())).items()
                    if numeric(v) and k not in NON_STAT and not k.endswith("_eps")]
            blk[m] = {k: M.describe({s: float(sel[s][k]) for s in sel if k in sel[s]}) for k in keys}
            for ck in CURVE_KEYS[kind]:
                E = np.stack([np.asarray(runs[(N, m, s)][0][ck], float)[runs[(N, m, s)][0]["uniform_index"]]
                              for s in sorted(sel)])
                curves_out[f"N{N}_{m}_{ck}_median"] = np.median(E, 0)
                curves_out[f"N{N}_{m}_{ck}_q25"] = np.percentile(E, 25, 0)
                curves_out[f"N{N}_{m}_{ck}_q75"] = np.percentile(E, 75, 0)
                curves_out[f"N{N}_{m}_{ck}_n"] = np.array(E.shape[0])
        per_N[N] = blk
        if "fr" not in r["methods"]:
            continue
        pa = {s: runs[(N, "abf", s)][1] for s in P["seeds"] if (N, "abf", s) in runs}
        pf = {s: runs[(N, "fr", s)][1] for s in P["seeds"] if (N, "fr", s) in runs}
        if not pa or not pf:
            continue
        c = dict(primary={}, secondary={}, marginal={}, tau={}, tau_secondary={})
        for grp, keys in (("primary", PRIMARY), ("secondary", SECONDARY[kind]), ("marginal", MARGINAL)):
            for k in keys:
                c[grp][k] = M.paired_contrast({s: v[k] for s, v in pa.items()}, {s: v[k] for s, v in pf.items()})
        for grp, keys in (("tau", tau_prim), ("tau_secondary", tau_sec)):
            for k in keys:
                c[grp][k] = M.paired_tau_contrast({s: float(v[k]) for s, v in pa.items()},
                                                  {s: float(v[k]) for s, v in pf.items()})
        contrasts[N] = c
        both = sorted(set(pa) & set(pf))
        tr = {}
        for ck in TRANSIENT_KEYS[kind]:
            if not both:                    # both arms have runs but no seed in common (mid-campaign state)
                tr[ck] = dict(max_rel=math.nan, u_at_max=math.nan, min_rel=math.nan, u_at_min=math.nan,
                              final_rel=math.nan, n=0, t_at_max=math.nan, t_at_min=math.nan)
                continue
            Ea = np.stack([np.asarray(runs[(N, "abf", s)][0][ck], float)[runs[(N, "abf", s)][0]["uniform_index"]] for s in both])
            Ef = np.stack([np.asarray(runs[(N, "fr", s)][0][ck], float)[runs[(N, "fr", s)][0]["uniform_index"]] for s in both])
            mt = M.max_transient_improvement(u_grid, Ea, Ef)
            curves_out[f"N{N}_rel_improvement_{ck}"] = mt.pop("rel")
            mt.update(n=len(both), t_at_max=mt["u_at_max"] * r["T"], t_at_min=mt["u_at_min"] * r["T"])
            tr[ck] = mt
        transient[N] = tr
    curves_out["u"] = u_grid
    return per_N, contrasts, transient, curves_out


def best_allocations(P, runs, rows, n_boot):
    out = {}
    Ns_abf = [r["N"] for r in rows]
    Ns_fr = [r["N"] for r in rows if r["N"] >= 2]
    n_pl = len(P["seeds"])
    for grp, keys in (("primary", BEST_PRIMARY), ("secondary", BEST_SECONDARY)):
        for k in keys:
            table = {}
            for r in rows:
                table[r["N"]] = {m: {s: float(runs[(r["N"], m, s)][1][k]) for s in P["seeds"] if (r["N"], m, s) in runs}
                                 for m in r["methods"]}
            ba = M.best_allocation(table, Ns_abf, Ns_fr, n_boot=n_boot)
            cands = {}
            for m, planned in (("abf", Ns_abf), ("fr", Ns_fr)):
                x = ba.get(m)
                have = list(x["Ns"]) if x else []
                n_seed = {N: sum(1 for v in table.get(N, {}).get(m, {}).values() if not math.isnan(v)) for N in planned}
                info = dict(Ns_planned=list(planned), Ns_missing=[N for N in planned if N not in have],
                            Ns_short_seeds={N: n_seed[N] for N in have if n_seed[N] < n_pl}, n_seeds_planned=n_pl)
                info["complete_ladder"] = not info["Ns_missing"] and not info["Ns_short_seeds"]
                cands[m] = info
                if x:
                    x.update(info)
            ba["candidates"] = cands
            ba["group"] = grp
            ba["axis"] = "budget fraction u (common budget grid)" if k.endswith("_u") else "error"
            out[k] = ba
    return out


def fr_activity(P, runs, rows):
    """Per N with FR runs: realised deaths per seed; zero-death FR arms are FR-INACTIVE (for the LTA bitwise equal
    to the ABF arm, so G = 0 exactly there): a tie at such an N is FR inactivity, not equivalence."""
    out = {}
    for r in rows:
        N = r["N"]
        if "fr" not in r["methods"]:
            continue
        sel = {s: runs[(N, "fr", s)][1] for s in P["seeds"] if (N, "fr", s) in runs}
        if not sel:
            continue
        d = np.array([float(sel[s]["fr_deaths"]) for s in sorted(sel)])
        dpw = d / N
        zero = [int(s) for s in sorted(sel) if int(sel[s]["fr_deaths"]) == 0]
        out[N] = dict(n_fr_seeds=len(sel), zero_death_seeds=zero, n_zero_deaths=len(zero),
                      median_deaths=float(np.median(d)), median_deaths_per_walker=float(np.median(dpw)),
                      min_deaths_per_walker=float(np.min(dpw)), max_deaths_per_walker=float(np.max(dpw)),
                      inactive=bool(len(zero) == len(sel)), any_inactive=bool(zero),
                      note=("FR realised no death in any seed: the FR arm equals ABF, G = 0 is FR inactivity, not "
                            "equivalence" if len(zero) == len(sel) else
                            (f"FR realised no death in {len(zero)}/{len(sel)} seeds (those seeds tie by construction)"
                             if zero else None)))
    return out


def establishment_null_table(system, P, rows):
    """Null calibration of the establishment criterion on every planned N's save grid (independent of the data)."""
    kind = MF.system_kind(system)
    out = {}
    for r in rows:
        steps = RL.save_grid(dict(P, system_key=MF.engine_key(system)), r["n_steps"])
        out[r["N"]] = M.establishment_null(r["N"], np.asarray(steps, float) / r["n_steps"], M.FAR[kind]["target"])
    return out


def annotate_tau(contrasts, floors):
    """Mark every tau contrast whose threshold lies below a HARD reference floor (unreachable by construction)."""
    for c in contrasts.values():
        for grp in ("tau", "tau_secondary"):
            for k, t in (c.get(grp) or {}).items():
                tf = floors.get(k[:-2]) if k.endswith("_u") else None
                t["unreachable_by_construction"] = bool(tf is not None and not tf["reachable"])
                t["floor"] = tf["floor"] if tf is not None else None
                if t["unreachable_by_construction"]:
                    t["note"] = (f"threshold {tf['eps']:g} < hard floor {tf['floor']:.4g}: censored in every arm by "
                                 f"construction; the ties are NOT evidence of equivalence")


def summarize_runs(P, system, runs, rows, n_boot, scorer):
    """Every aggregate block of summary.json from the per-run (curves, scalars) dict -- also used by
    audit_completeness.py to recompute the summary.  Returns (blocks, median curves)."""
    floors = MF.threshold_floors(system, P, scorer)
    per_N, contrasts, transient, curves_out = aggregate(P, system, runs, rows, floors)
    annotate_tau(contrasts, floors)
    blocks = dict(per_N=per_N, contrasts=contrasts, max_transient=transient,
                  best_allocation=best_allocations(P, runs, rows, n_boot) if runs else {},
                  fr_activity=fr_activity(P, runs, rows), threshold_floors=floors,
                  establishment_null=establishment_null_table(system, P, rows))
    return blocks, curves_out


DEFINITIONS = dict(
    G="(X_FR - X_ABF) / X_ABF per seed (same N, same seed); median, seed bootstrap 95 % CI, wins = G < 0",
    tau="first save after which the metric stays <= eps at every later save (all saves of the run); inf (JSON 'inf') "
        "= censored.  Paired at one N (both arms share the grid).",
    tau_bgrid="the same on the COMMON budget grid only (200 uniform + 24 log-spaced u, identical at every N); used "
              "whenever tau is compared ACROSS N (best allocation, budget-axis tau vs N)",
    tau_rank="FR wins when tau_FR < tau_ABF (finite beats censored; censored vs censored ties); exact sign test.  "
             "A tau whose threshold lies below a hard reference floor (threshold_floors, unreachable_by_construction) "
             "is censored in every arm by construction: its ties are not evidence of equivalence",
    threshold_floors="zero-noise floor of each error (the scorer fed the exact bin-averaged reference); gateway "
                     "e_F' and e_F'_em: HARD lower bound 0.0323 (within-bin variation of F'_ref, e^2 = stat^2 + "
                     "floor^2), so the frozen strict threshold 0.012 is unreachable by construction and mid 0.035 "
                     "needs a statistical part <= 0.0136; Ibar_Fp_stat / final_e_Fp_stat are the labelled floor-free "
                     "companions (RMS of Gamma - bin-averaged F'_ref), descriptive, not frozen metrics",
    Ibar="mean of the error over the 200 uniform budget fractions u = k/200",
    TV_half="TV(18 coarse bins) of C_all(t) - C_all(s_lo), s_lo = save nearest t/2 (or the run start)",
    TV_inst_floor="E_N[TV] under multinomial(N, uniform 18), exact binomial formula",
    establishment="first save after which the instantaneous far-state fraction stays >= 0.5 x target (frozen); "
                  "ill-defined at N <= 2 (est_cum_*: the same on the cumulative visitation C_all)",
    establishment_null=("null calibration on each N's save grid: far count ~ Bin(N, target) independently per save "
                        "(an EXACTLY uniform population); est_unreliable = N <= 2 or P(no failure among the saves "
                        f"with u > 1/2) < {M.EST_RELIABLE_P}: there est_* is censored / late by chance even when "
                        "established; read est_cum_* and first arrival instead.  Uniform-target FR birth-death clones "
                        "into the under-filled far state and kills over-represented walkers, which can push the "
                        "instantaneous count BELOW binomial scatter: at unreliable N an FR establishment advantage is "
                        "NOT evidence for faster convergence (Outcome C)"),
    fr_events=("realised deaths (= realised replacements, one death + one copy) per FR opportunity in BOTH systems; "
               "gateway from fr_deaths_hist / fr_opp_death_cum (engine /2), its capped candidates kd + kc reported "
               "separately as fr_candidate*; fr_activity: FR arms with zero realised deaths equal ABF (LTA: "
               "bitwise), so their G = 0 is FR inactivity, not equivalence"),
    fr_cap_extension="LTA only: cap max(1, floor(0.02 N)) where the historical int(0.02 N) is 0 (N < 50); the "
                     "gateway's max(1, floor(0.08 N)) is its historical rule (no extension)",
    best_allocation="min over N of the seed median (ABF all N, FR N >= 2); bootstrap resamples seeds per N (one draw "
                    "shared by both methods) and re-selects the minimising N; ties share the credit (boot_tie_frac); "
                    "an N without usable seeds in a resample is skipped (boot_frac_no_data when none is left); "
                    "candidates.Ns_missing lists planned N absent from the candidate set (the minimum is then over a "
                    "REDUCED ladder); tau via tau_bgrid",
    max_transient="(ABF - FR)/ABF of the seed-median curves over the 200 uniform u (seeds with both arms; n = 0: none)",
)


# --------------------------------------------------------------------------------------------- tables.md
def f_val(v, kind="g", tau=None, T=None):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if isinstance(v, float) and math.isinf(v):
        if tau == "u":
            return "> 1"
        if tau == "t":
            return f"> {T:g}" if T else "> T_N"
        return "inf"
    if kind == "pct":
        return f"{100 * v:+.1f} %"
    if kind == "int":
        return f"{v:.0f}"
    return f"{v:.4g}"


def med_iqr(d, tau=None, T=None):
    if not d:
        return "--"
    return f"{f_val(d['median'], tau=tau, T=T)} [{f_val(d['q25'], tau=tau, T=T)}, {f_val(d['q75'], tau=tau, T=T)}]"


def contrast_cell(c):
    if not c or c["n"] == 0:
        return "--"
    return f"{f_val(c['G_median'], 'pct')} [{f_val(c['G_ci95'][0], 'pct')}, {f_val(c['G_ci95'][1], 'pct')}] ({c['wins']}/{c['n']})"


def tables_md(S, kind):
    L = []
    pl = S["plan"]
    cl = S.get("cell") or {}
    L += [f"# Equal-budget ladder analysis: {S['system']}"
          + (f" -- mechanism cell {cl.get('experiment')}/{cl.get('cell')}: {cl.get('cell_label')}" if cl else ""), "",
          f"Config `{S['config']}`; results `{S['results_root']}`; generated {S['generated_utc']}; {S['metrics_version']}.",
          f"B = {pl['B']:,} walker-steps per arm per seed; h = {pl['h']:g}; seeds {len(pl['seeds'])}; "
          f"thresholds e_F {pl['thresholds']['e_F']}, e_F' {pl['thresholds']['e_Fp']}, TV_half {pl['thresholds']['TV_half']}"
          + "".join(f", {k} {v}" for k, v in pl["thresholds"].items() if k not in ("e_F", "e_Fp", "TV_half")) + ".",
          "Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, "
          f"{S['n_boot']} resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the "
          "budget axis u (censored = '> 1').", ""]
    ref = S["reference"]
    L.append("Reference: " + ", ".join(f"{k} {v}" for k, v in ref.items() if not isinstance(v, dict)) + ".")
    fl = ref.get("zero_noise_floor") or {}
    L.append("Zero-noise floors (scorer fed the exact bin-averaged reference): " + ", ".join(
        f"{k} {v:.4g}" + (" (HARD lower bound)" if (ref.get("floor_is_lower_bound") or {}).get(k) else "")
        for k, v in fl.items()) + ".")
    L += ["", "## Frozen thresholds vs the zero-noise floors", "",
          "| tau | eps | floor | hard bound | reachable | statistical budget sqrt(eps^2 - floor^2) | note |",
          "|---|---|---|---|---|---|---|"]
    for k, tf in (S.get("threshold_floors") or {}).items():
        L.append(f"| {k[4:]} | {tf['eps']:g} | {f_val(tf['floor'])} | {'yes' if tf['floor_is_lower_bound'] else 'no'} | "
                 f"{'yes' if tf['reachable'] else '**NO (unreachable by construction)**'} | {f_val(tf['stat_budget'])} | "
                 f"{tf.get('note') or ''} |")
    L += ["", "## Status", "", "| N | n_steps | T_N | ABF complete | FR complete | missing / running / invalid |", "|---|---|---|---|---|---|"]
    for N, st in S["status"].items():
        a = st["methods"]["abf"]
        f = st["methods"].get("fr")
        miss = []
        for m, d in st["methods"].items():
            for k in ("missing", "running", "invalid"):
                if d[k]:
                    miss.append(f"{m} {k} {len(d[k])}")
        L.append(f"| {N} | {st['n_steps']:,} | {st['T']:g} | {len(a['complete'])}/{a['planned']} | "
                 f"{(str(len(f['complete'])) + '/' + str(f['planned'])) if f else 'n/a'} | {'; '.join(miss) or '--'} |")
    pn = S["per_N"]

    def g(N, m, k):
        b = pn.get(N, {}).get(m)
        return b.get(k) if b else None

    L += ["", "## Absolute values (median [IQR] over seeds)", "",
          "| N | T_N | ABF Ibar_F | FR Ibar_F | ABF Ibar_F' | FR Ibar_F' | ABF e_F(1) | FR e_F(1) | ABF e_F'(1) | FR e_F'(1) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for N in pn:
        T = pn[N]["T"]
        L.append(f"| {N} | {T:g} | " + " | ".join(med_iqr(g(N, m, k)) for k in ("Ibar_F", "Ibar_Fp", "final_e_F", "final_e_Fp")
                                                   for m in ("abf", "fr")) + " |")
    L += ["", "## Paired contrasts FR vs ABF at each N", "",
          "| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |",
          "|---|---|---|---|---|---|---|---|"]
    for N, c in S["contrasts"].items():
        fa = (S.get("fr_activity") or {}).get(N) or {}
        act = ("**FR INACTIVE (no death in any seed): G = 0 is not equivalence**" if fa.get("inactive") else
               (f"**no death in {fa['n_zero_deaths']}/{fa['n_fr_seeds']} seeds**" if fa.get("n_zero_deaths") else
                (f"{fa['median_deaths_per_walker']:.3g} deaths / walker" if fa else "--")))
        L.append(f"| {N} | " + " | ".join(contrast_cell(c["primary"].get(k)) for k in PRIMARY) + " | "
                 + " | ".join(contrast_cell(c["marginal"].get(k)) for k in ("Ibar_TV_half", "final_TV_half"))
                 + f" | {act} |")
    L += ["", "## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')", "",
          "| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for N, c in S["contrasts"].items():
        # a mechanism cell also lists its secondary tau (e_F_em; raw e_F' above its floor), labelled
        items = [(k, t, "") for k, t in c["tau"].items()]
        if cl:
            items += [(k, t, " (secondary)") for k, t in (c.get("tau_secondary") or {}).items()]
        for k, t, lab in items:
            r = t["rank"]
            unr = " **(unreachable by construction: ties are not equivalence)**" if t.get("unreachable_by_construction") else ""
            L.append(f"| {N} | {k[4:-2]}{lab}{unr} | {f_val(t['abf_median'], tau='u')} | {f_val(t['censored_frac_abf'])} | "
                     f"{f_val(t['fr_median'], tau='u')} | {f_val(t['censored_frac_fr'])} | {r['wins']}/{r['losses']}/{r['ties']} | "
                     f"{r['sign_test_p']:.3g} | {f_val(t['G_median'], 'pct')} [{f_val(t['G_ci95'][0], 'pct')}, "
                     f"{f_val(t['G_ci95'][1], 'pct')}] (n {t['n_both_finite']}) |")
    tau_prim = TAU_PRIMARY if not cl else tau_keys(S["system"], S["plan"], S.get("threshold_floors"))[0]
    L += ["", "### tau per arm (all N; median u [IQR], fraction censored)", "",
          "| N | method | " + " | ".join(k[4:-2] for k in tau_prim) + " |", "|---|---|" + "---|" * len(tau_prim)]
    for N in pn:
        for m in ("abf", "fr"):
            if not pn[N].get(m):
                continue
            cells = []
            for k in tau_prim:
                d = g(N, m, k)
                cz = g(N, m, k[:-2] + "_censored")
                cells.append(f"{med_iqr(d, tau='u')} ({f_val(cz['mean'])})" if d and cz else "--")
            L.append(f"| {N} | {m} | " + " | ".join(cells) + " |")
    L += ["", "## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)", "",
          "| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |",
          "|---|---|---|---|---|---|---|"]
    for N, e in (S.get("establishment_null") or {}).items():
        L.append(f"| {N} | {e['k_min']} | {e['p_fail_per_save']:.3g} | {e['p_censored']:.3g} | {e['p_hold_second_half']:.3f} | "
                 f"{f_val(e['null_median_u'], tau='u')} | {'**yes** (read cum. est. / first arrival)' if e['unreliable'] else 'no'} |")
    L += ["", "## Marginal and population (median [IQR])", "",
          "| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    fa_key = "first_right_u" if kind == "gateway" else "first_window_u"
    en = S.get("establishment_null") or {}
    for N in pn:
        for m in ("abf", "fr"):
            if not pn[N].get(m):
                continue
            ill = (" (ill-defined, N <= 2)" if int(N) <= 2 else
                   (" (UNRELIABLE: fails by chance at this N)" if (en.get(N) or {}).get("unreliable") else ""))
            cz = g(N, m, "est_censored")
            L.append(f"| {N} | {m} | {med_iqr(g(N, m, 'final_TV_half'))} | {med_iqr(g(N, m, 'tau_TV_half_u'), tau='u')} | "
                     f"{med_iqr(g(N, m, 'final_TV_inst'))} | {S['tv_floor']['table'][str(N)]['mean']:.4f} | "
                     f"{med_iqr(g(N, m, 'est_u'), tau='u')}{ill} | {f_val(cz['mean'] if cz else None)} | "
                     f"{med_iqr(g(N, m, 'est_cum_u'), tau='u')} | {med_iqr(g(N, m, fa_key), tau='u')} | "
                     f"{med_iqr(g(N, m, 'final_transitions'))} |")
    L += ["", "## FR genealogy and activity (FR arm, median [IQR])", "",
          "Event = a REALISED death (= one replacement: death + copy) in both systems"
          + ("; the gateway's capped candidates kd + kc are the last column." if kind == "gateway" else ".")
          + " 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).", "",
          "| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | "
          "max events/N per opp | final ESS run | min ESS win | final unique run | max family win |"
          + (" candidates/N per opp |" if kind == "gateway" else ""),
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" if kind == "gateway" else "")]
    fa_all = S.get("fr_activity") or {}
    for N in pn:
        if not pn[N].get("fr"):
            continue
        fa = fa_all.get(N) or {}
        ina = (f"**{fa['n_zero_deaths']}/{fa['n_fr_seeds']}**" if fa.get("n_zero_deaths") else f"0/{fa.get('n_fr_seeds', '--')}")
        L.append(f"| {N} | {ina} | {f_val(g(N, 'fr', 'fr_cap')['median'], 'int')} | {f_val(g(N, 'fr', 'fr_cap_extension')['median'], 'int')} | "
                 f"{med_iqr(g(N, 'fr', 'fr_deaths'))} | {med_iqr(g(N, 'fr', 'fr_deaths_per_walker'))} | "
                 f"{med_iqr(g(N, 'fr', 'fr_frac_opps_with_event'))} | "
                 f"{med_iqr(g(N, 'fr', 'fr_mean_event_frac_per_opp'))} | {med_iqr(g(N, 'fr', 'fr_max_event_frac_per_opp'))} | "
                 f"{med_iqr(g(N, 'fr', 'final_gen_ess_run'))} | {med_iqr(g(N, 'fr', 'min_gen_ess_win'))} | "
                 f"{med_iqr(g(N, 'fr', 'final_gen_nuniq_run'))} | {med_iqr(g(N, 'fr', 'max_gen_maxfam_win'))} |"
                 + (f" {med_iqr(g(N, 'fr', 'fr_mean_candidate_frac_per_opp'))} |" if kind == "gateway" else ""))
    L += ["", "## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u", "",
          "| N | metric | max | at u | min | at u | at u = 1 | n seeds |", "|---|---|---|---|---|---|---|---|"]
    for N, tr in S["max_transient"].items():
        for k, t in tr.items():
            L.append(f"| {N} | {k} | {f_val(t['max_rel'], 'pct')} | {f_val(t['u_at_max'])} | {f_val(t['min_rel'], 'pct')} | "
                     f"{f_val(t['u_at_min'])} | {f_val(t['final_rel'], 'pct')} | {t['n']} |")
    L += ["", "## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)", "",
          "| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, b in S["best_allocation"].items():
        a, f, d = b.get("abf"), b.get("fr"), b.get("fr_minus_abf")
        tau = "u" if k.endswith("_u") else None

        def freq(x):
            txt = ", ".join(f"{n}: {p:.2f}" for n, p in sorted(x["boot_best_N_freq"].items(), key=lambda kv: -kv[1]) if p > 0)
            for lab, key in (("all censored", "boot_frac_censored"), ("no data", "boot_frac_no_data"),
                             ("ties", "boot_tie_frac")):
                v = x.get(key) or 0.0
                if v > 0:
                    txt += (", " if txt else "") + f"{lab}: {v:.2f}"
            if x.get("Ns_missing"):
                txt += f"; **REDUCED ladder: N {x['Ns_missing']} missing**"
            if x.get("Ns_short_seeds"):
                txt += f"; N with fewer seeds {x['Ns_short_seeds']}"
            return txt
        row = [k]
        for x in (a, f):
            if x:
                bn = str(x["best_N"]) if x["best_N"] is not None else "all censored"
                if x.get("best_N_tied"):
                    bn += f" (tied: {x['best_N_tied']})"
                row += [bn, f_val(x["best_value"], tau=tau),
                        f"[{f_val(x['boot_best_value_ci95'][0], tau=tau)}, {f_val(x['boot_best_value_ci95'][1], tau=tau)}]", freq(x)]
            else:
                row += ["--"] * 4
        row += [f"{f_val(d['point'])} [{f_val(d['ci95'][0])}, {f_val(d['ci95'][1])}]" if d else "--",
                f"{f_val(d['point_rel'], 'pct')} [{f_val(d['rel_ci95'][0], 'pct')}, {f_val(d['rel_ci95'][1], 'pct')}]" if d else "--"]
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## Secondary (" + ("EM-consistent reference; *_Fp_stat = floor-free companion of e_F' (descriptive)"
                                  if kind == "gateway" else "periodically projected F'") + ")", "",
          "| N | " + " | ".join(SECONDARY[kind]) + " |", "|---|" + "---|" * len(SECONDARY[kind])]
    for N, c in S["contrasts"].items():
        L.append(f"| {N} | " + " | ".join(f"ABF {f_val(c['secondary'][k]['abf_median'])}, FR {f_val(c['secondary'][k]['fr_median'])}; "
                                          f"{contrast_cell(c['secondary'][k])}" for k in SECONDARY[kind]) + " |")
    L += ["", "## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)", "",
          f"{S['tv_floor']['method']}.", "", "| N | E_N[TV] |", "|---|---|"]
    for N, d in S["tv_floor"]["table"].items():
        L.append(f"| {N} | {d['mean']:.5f} |")
    L += ["", "## Cost (median per run)", "", "| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |", "|---|---|---|---|---|---|"]
    for N in pn:
        for m in ("abf", "fr"):
            if not pn[N].get(m):
                continue
            ok = g(N, m, "n_force_evals_ok")
            L.append(f"| {N} | {m} | {f_val(g(N, m, 'wall_s')['median'])} | {f_val(g(N, m, 'us_per_walker_step')['median'])} | "
                     f"{f_val(g(N, m, 'peak_rss_mb')['median'])} | {int(sum(ok['per_seed'].values()))}/{ok['n']} |")
    if S["warnings"]:
        L += ["", "## Warnings", ""] + [f"* {w}" for w in S["warnings"]]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------- main
def analyze(system, results_root=None, config=None, out=None, n_boot=M.N_BOOT, skip_invalid=False, verbose=True):
    kind = MF.system_kind(system)
    config = config or MF.default_config_path(system)
    P = json.load(open(config))
    MF.check_invocation(system, P, config)     # a cell config only as gateway_family, gateway_family only with one
    cell = MF.is_cell(P)                       # mechanism cell: its own result / analysis locations
    results_root = os.path.abspath(results_root or (MF.cell_results_root(P) if cell else
                                                    os.path.join(ROOT, "results", "equal_budget_v2")))
    out = os.path.abspath(out or (MF.cell_analysis_dir(P) if cell else os.path.join(results_root, P["out_dir"], "analysis")))
    MF.guard_output(P, out, system)
    cache_dir = os.path.join(out, "run_metrics")
    os.makedirs(cache_dir, exist_ok=True)
    get_scorer = functools.lru_cache(maxsize=1)(lambda: MF.make_scorer(system, P))
    prov = provenance(system, P, config)
    rows = plan(P)
    planned = {(r["N"], s, m) for r in rows for m in r["methods"] for s in P["seeds"]}
    runs, status, warnings = {}, {}, []
    hits = computed = 0
    t0 = time.time()
    for r in rows:
        N = r["N"]
        st = dict(n_steps=r["n_steps"], T=r["T"], methods={})
        for m in r["methods"]:
            d = dict(planned=len(P["seeds"]), complete=[], missing=[], running=[], invalid={})
            for s in P["seeds"]:
                p = run_path(results_root, P, N, s, m)
                fs, why = file_status(p)
                if fs == "complete":
                    try:
                        curves, sc, hit = cached_metrics(p, system, P, get_scorer, cache_dir,
                                                         dict(N=N, seed=int(s), method=m), prov)
                    except M.MetricsError as e:
                        if not skip_invalid:
                            raise
                        d["invalid"][int(s)] = str(e)
                        continue
                    runs[(N, m, int(s))] = (curves, sc)
                    d["complete"].append(int(s))
                    hits += hit
                    computed += not hit
                    if not sc["n_force_evals_ok"]:
                        warnings.append(f"N {N} seed {s} {m}: n_force_evals {sc['n_force_evals']} != expected {sc['n_force_evals_expected']}")
                    if not sc["u_grid_exact"]:
                        warnings.append(f"N {N} seed {s} {m}: n_steps not a multiple of 200, uniform u grid by rounding")
                    for k in [k for k in sc if k.startswith("n_nan_") and sc[k]]:
                        warnings.append(f"N {N} seed {s} {m}: {sc[k]} non-finite values in {k[6:]}")
                elif fs == "invalid":
                    if not skip_invalid:
                        raise M.MetricsError(f"{p}: {why}")
                    d["invalid"][int(s)] = why
                else:
                    d[fs].append(int(s))
            st["methods"][m] = d
        status[N] = st
    extra = unplanned_files(results_root, P, planned)
    if extra:
        warnings.append(f"{len(extra)} result files not in the plan were ignored: {extra[:10]}")
    for N, st in status.items():
        for m, d in st["methods"].items():
            if d["missing"] or d["running"] or d["invalid"]:
                warnings.append(f"N {N} {m}: {len(d['complete'])}/{d['planned']} complete "
                                f"(missing {len(d['missing'])}, running {len(d['running'])}, invalid {len(d['invalid'])})")
    if verbose:
        print(f"{system}: {len(runs)} runs ({computed} scored, {hits} from cache) in {time.time() - t0:.1f} s", flush=True)
    scorer = get_scorer()                      # also verifies the frozen reference when nothing ran yet
    agg, curves_out = summarize_runs(P, system, runs, rows, n_boot, scorer)
    for N, fa in agg["fr_activity"].items():
        if fa["note"]:
            warnings.append(f"N {N}: {fa['note']}")
    for k, tf in agg["threshold_floors"].items():
        if not tf["reachable"]:
            warnings.append(f"{k}: {tf['note']}")
    S = dict(
        schema=SCHEMA, system=system, kind=kind, config=os.path.abspath(config), results_root=results_root, out=out,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), metrics_version=M.METRICS_VERSION,
        n_boot=int(n_boot), boot_seed=M.BOOT_SEED,
        plan=dict(B=int(P["B"]), h=float(P["engine_cfg"]["h"]), N_ladder=[int(n) for n in P["N_ladder"]],
                  seeds=[int(s) for s in P["seeds"]], thresholds=P["thresholds"], far_state=M.FAR[kind],
                  uniform_u=f"{M.N_UNIFORM} saves at u = k/{M.N_UNIFORM}", coarse_bins=M.N_COARSE),
        reference=scorer.reference_info, provenance=prov,
        tv_floor=dict(method=M.TV_FLOOR_METHOD, table=M.tv_floor_table(P["N_ladder"])),
        status=status, **agg, warnings=warnings, cache=dict(hits=hits, computed=computed, dir=cache_dir),
        definitions=DEFINITIONS,
    )
    if cell:
        S["cell"] = {k: P.get(k) for k in ("experiment", "cell", "cell_label", "model", "engine_version", "results_dir",
                                            "analysis_dir", "fig_dir", "reference_file", "reuse")}
        S["cell"]["reuse_gate"] = MF.reuse_gate_status(P)
        if S["cell"]["reuse_gate"]["problems"]:
            warnings.append("reuse cell: the plan's bitwise-reuse gate is NOT established ("
                            + "; ".join(S["cell"]["reuse_gate"]["problems"]) + "): these results may not be used")
    S = M.json_safe(S)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(S, fh, indent=1, allow_nan=False)
    with open(os.path.join(out, "tables.md"), "w") as fh:
        fh.write(tables_md(_reload(S), kind))
    np.savez_compressed(os.path.join(out, "median_curves.npz"), **curves_out)
    if verbose:
        print(f"wrote {out}/summary.json, tables.md, median_curves.npz", flush=True)
    return S


def _reload(S):
    """JSON-safe summary -> numbers again ('inf' strings -> inf, null -> nan) for the markdown tables."""
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [conv(v) for v in o]
        if o == "inf":
            return math.inf
        if o == "-inf":
            return -math.inf
        return o
    return conv(S)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=list(MF.SYSTEMS))

    ap.add_argument("--results-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=M.N_BOOT)
    ap.add_argument("--skip-invalid", action="store_true")
    a = ap.parse_args(argv)
    analyze(a.system, a.results_root, a.config, a.out, a.n_boot, a.skip_invalid)


if __name__ == "__main__":
    main()
