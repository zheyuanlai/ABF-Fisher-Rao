#!/usr/bin/env python
"""Analysis of the Experiment III wall-clock benchmark (docs/mechanism/SCIENTIFIC_PLAN.md section 8;
scripts/mechanism/parallel_benchmark.py writes the runs).

    python scripts/mechanism/analyze_parallel_benchmark.py                       # results/mechanism/parallel_benchmark
    python scripts/mechanism/analyze_parallel_benchmark.py --root DIR --out DIR --fig-dir DIR

Per run (the SAME scorer as the equal-budget production: scripts/equal_budget/eqb_metrics.make_scorer -> e_F at every
save from the saved accumulators M_all, C_all [+ M_prod, C_prod]):
  persistent time-to-accuracy at the frozen MID threshold (gateway 0.0055, LTA 300 K 0.094, 150 K 0.13 kJ/mol):
  eqb_metrics.tau_persistent on every save of the run (both arms share the grid) -> its first save i of the persistent
  window, reported as simulation time save_t[i], force evaluations n_force_evals_at_save[i] and WALL seconds
  wall_s_at_save[i] (wall at the first save of the persistent window; censored -> inf);
  Ibar_F (mean e_F over the 200 uniform budget fractions; needs the full grid), final e_F; total wall (timed run),
  warm-up, us per step, peak memory, FR device time (gpu_torch).
Per (system, backend), paired ABF vs ABF+FR per seed:
  tau ratio FR / ABF for t, force evaluations and wall: median over the seeds where both are finite + 10 000-resample
  seed-bootstrap 95 % CI (eqb_metrics.boot_median_ci), with the censoring-aware rank part (FR wins when tau_FR <
  tau_ABF; any finite beats inf) and its exact sign test (eqb_metrics.paired_tau_contrast); G(Ibar_F)
  (eqb_metrics.paired_contrast);
  W1 (plan section 8) = the paired WALL ratio on each backend under W1_RULE below (uncontended pairs; all pairs as a
  sensitivity analysis);
  FR overhead = (wall_FR - wall_ABF) / wall_ABF of the timed runs (equal steps), median + CI, on the uncontended
  pairs (primary) and on all pairs;
  total resource use: timed device-seconds (sum of wall_total_s) and allocated device-seconds (process start to the
  end of each run, process_wall_s).
Across backends (same system, method, seed and n_steps): effective parallelism = wall(cpu_numba_1core) /
  wall(gpu_torch) at equal steps, median + CI, on the pairs with no SMT contention in either run (primary) and on all.
  One CPU core is never equated with a GPU: the ratio is reported as what it is.
Statistical equivalence of a torch backend with the validated numba production at N = 512 (EQUIVALENCE_RULE below,
  stated here before any benchmark data exist; reported, never tuned): needs the full budget and >= 8 complete pairs.
Resource ceilings (plan section 9): the projected GPU-h of each system's full-budget gpu_torch runs is compared with
  the 8 GPU-h ceiling (and the <= 3 GPU-h Experiment III estimate): a system whose projection exceeds the ceiling
  has an equivalence gate that CANNOT BE EVALUATED within the plan (summary.json cost_projection.ceiling_check).
Backends with no runs are reported NOT TESTED (cpu_numba_threads always: no walker-parallel build exists).

Outputs: <out>/summary.json, <out>/tables.md, figures <fig-dir>/parallel_benchmark_<tag>.{pdf,png} with
<fig-dir>/MANIFEST.json (legibility of every figure: scripts/equal_budget/fig_legibility.check_figure_safe):
  T_time_to_accuracy   preregistered endpoints (t, force evaluations, wall) per backend x arm, seeds paired
  C_cost               preregistered 'also reported': us per step, effective parallelism, FR overhead, peak memory
                       (host RSS of every run; GPU device memory incl. the CUDA context for gpu_torch)
  X_error_vs_wall      EXPLORATORY: e_F against wall-clock seconds (median and IQR across seeds)
  Q_equivalence        EXPLORATORY (tolerance stated in this harness, not in the plan): torch backend vs numba
                       production, Ibar_F per arm and G(Ibar_F)
CPU only.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts", "equal_budget"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import eqb_metrics as M  # noqa: E402
import fig_legibility as LG  # noqa: E402
import parallel_benchmark as PB  # noqa: E402

ANALYSIS_VERSION = "analyze_parallel_benchmark/1"
N_BOOT = M.N_BOOT
BOOT_SEED = M.BOOT_SEED
BACKEND_ORDER = ("cpu_numba_1core", "gpu_torch", "cpu_torch_test")
BACKENDS_EXP = ("cpu_numba_1core", "gpu_torch")          # Experiment III backends (cpu_torch_test: harness tests only)
BACKEND_LABEL = {"cpu_numba_1core": "1 CPU core (numba)", "gpu_torch": "1 GPU (torch)",
                 "cpu_torch_test": "torch on CPU (test only)", "cpu_numba_threads": "k CPU cores (numba)"}
METHOD_LABEL = {"abf": "ABF", "fr": "ABF+FR"}
C_ABF, C_FR = "#2a78d6", "#eb6834"
METHOD_COLOR = {"abf": C_ABF, "fr": C_FR}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BACKEND_MARK = {"cpu_numba_1core": "o", "gpu_torch": "s", "cpu_torch_test": "D"}
BACKEND_SHORT = {"cpu_numba_1core": "1 core", "gpu_torch": "GPU", "cpu_torch_test": "torch-CPU"}
BACKEND_LS = {"cpu_numba_1core": "-", "gpu_torch": "--", "cpu_torch_test": ":"}
PRIMARY_BACKEND_FOR_EQUIV = "cpu_numba_1core"
# --------------------------------------------------------------------------------------------------------------
# EQUIVALENCE_RULE (fixed 2026-10-10 in this harness before any Experiment III benchmark data exist; reported, not
# tuned; revised the same day after review, still before any data: the first version compared the backend median
# with resampled medians of the production values, treating the production seeds as the population, which is
# anti-conservative -- family-wise false-INCONSISTENT 8-12 % by Monte Carlo).  For a torch backend (full budget)
# against the validated numba production at N = 512 (results/equal_budget_v2/<out_dir>/analysis/summary.json; 32 / 16
# / 16 production seeds):
#   reference  the production per-seed values MINUS the seeds whose runs share randomness with the backend's runs
#              (the gateway torch driver reproduces production's x0 / y0 for a production seed: those seeds are
#              dropped from the reference; the LTA torch draws are independent of production's, nothing is dropped);
#   R1/R2  per arm (abf, fr): two-sample PERMUTATION test of median(backend Ibar_F) - median(reference Ibar_F),
#          20 000 random relabellings of the pooled values, two-sided p = (1 + #{|d*| >= |d|}) / (1 + 20 000) -- exact
#          under H0 (both samples iid from one law), the uncertainty of BOTH medians included;
#   R3/R4  per arm: two-sided Mann-Whitney U test of the per-seed Ibar_F, backend vs reference;
#   R5     permutation test as R1 on the per-seed paired G(Ibar_F) = (FR - ABF) / ABF (backend complete pairs vs the
#          reference's per-seed G).
#   Holm-Bonferroni over R1-R5 at family-wise 0.05 -> CONSISTENT (no rejection) or INCONSISTENT (rejections listed),
#   issued ONLY with >= EQUIV_MIN_PAIRS = 8 complete (ABF, FR) backend pairs (plan: seeds per arm '>= 8'); below
#   that the tests are reported and the status is INCOMPLETE (no verdict).  Next to every verdict: the minimum
#   detectable shift of each median contrast (normal approximation from the permutation null, 80 % power, per-test
#   alpha 0.01 = Holm's strictest step and 0.05 = its most lenient), relative to the reference median for Ibar_F.
#   This is a non-rejection (consistency) check, not a TOST equivalence proof: CONSISTENT means 'no difference
#   larger than about the reported minimum detectable shift', nothing smaller.
EQUIV_ALPHA = 0.05
EQUIV_NPERM = 20_000
EQUIV_SEED = 20261010
EQUIV_MIN_PAIRS = PB.DEFAULT_SEEDS_PER_ARM
MDS_POWER = 0.80
MDS_ALPHAS = (0.01, 0.05)
EQUIVALENCE_RULE = ("reference = production per-seed values minus seeds sharing randomness with the backend runs; "
                    "R1/R2: two-sample permutation test of the median Ibar_F difference per arm (20 000 relabellings); "
                    "R3/R4: two-sided Mann-Whitney U per arm; R5: permutation test of the median paired G(Ibar_F) "
                    "difference; Holm over R1-R5 at 0.05 -> CONSISTENT / INCONSISTENT, only with >= 8 complete backend "
                    "pairs (else INCOMPLETE); minimum detectable shift (80 % power) reported with every verdict")
SMT_BUSY_FLAG = PB.SMT_BUSY_FLAG
# W1_RULE (fixed 2026-10-10 before any Experiment III benchmark data exist).  A busy SMT sibling slows a run 1.4-1.7x
# (measured), the same order as the effects W1 measures, so:
W1_RULE = ("W1 on one backend = the paired FR / ABF wall-clock time-to-accuracy ratio (median over both-finite pairs "
           "+ seed-bootstrap 95 % CI, censoring-aware rank part) computed ONLY on the pairs in which NEITHER run was "
           "SMT-contended (each run pinned to one logical CPU, sibling busy <= 0.10 of its timed run, telemetry "
           "present); excluded pairs (contended or unknown) are counted; the same ratio on ALL pairs is reported as a "
           "sensitivity analysis and AGREES iff its median lies inside the primary 95 % CI; with no uncontended pair "
           "W1 is NOT EVALUABLE on that backend.  The same primary / all-pairs split is applied to the FR overhead "
           "and to the effective parallelism.")
PLAN_GPU_CEILING_H = PB.PLAN_CEILING_H["gpu_torch"]
PLAN_GPU_ESTIMATE_H = 3.0
PLAN_CPU_ESTIMATE_H = PB.PLAN_CEILING_H["cpu"]


# ================================================================================================ loading / scoring
def discover(root):
    """{(system, backend): {(seed, method): npz path}} of complete runs under root."""
    out = {}
    for system in PB.SYSTEMS:
        for backend in BACKEND_ORDER:
            d = os.path.join(root, system, backend)
            for p in sorted(glob.glob(os.path.join(d, "s*_*.npz"))):
                b = os.path.basename(p)[1:-4]
                seed, _, method = b.partition("_")
                js = p[:-4] + ".json"
                if method not in PB.METHODS or not os.path.exists(js):
                    continue
                with open(js) as fh:
                    m = json.load(fh)
                if m.get("status") != "complete":
                    continue
                out.setdefault((system, backend), {})[(int(seed), method)] = p
    return out


_SCORERS = {}


def scorer_for(system):
    if system not in _SCORERS:
        P, N, thr = PB.system_spec(system)
        _SCORERS[system] = (M.make_scorer(system, P), P, N, thr)
    return _SCORERS[system]


def score_run(path, system):
    sc, P, N, thr = scorer_for(system)
    r = PB.load_run(path)
    meta = r["meta"]
    if meta["system"] != system or int(meta["N"]) != N:
        raise M.MetricsError(f"{path}: system / N mismatch")
    res = dict(M_all=r["M_all"], C_all=r["C_all"], meta=dict(T_K=meta.get("T_K")))
    if "M_prod" in r:
        res.update(M_prod=r["M_prod"], C_prod=r["C_prod"])
    eF = np.asarray(sc.errors(res)["e_F"], float)
    t = np.asarray(r["save_t"], float)
    u = np.asarray(r["save_u"], float)
    step = np.asarray(r["save_step"], np.int64)
    wall = np.asarray(r["wall_s_at_save"], float)
    nfe = np.asarray(r["n_force_evals_at_save"], float)
    n_steps = int(meta["n_steps"])
    tau = M.tau_persistent(eF, t, u, thr)
    i = tau["index"]
    sc_ = r if "peak_gpu_device_mb" in r else {}
    dev_mb = float(sc_.get("peak_gpu_device_mb", math.nan)) if sc_ else math.nan
    if meta["backend"] == "gpu_torch" and not np.isfinite(dev_mb):        # older files: the post-run nvidia-smi
        own = [PB.parse_mib(p.get("used_memory")) for p in (meta.get("gpu_processes_after") or [])
               if int(p.get("pid", -1)) == int(meta.get("pid", -2))]
        dev_mb = max(own) if own else math.nan
    out = dict(path=os.path.relpath(path, ROOT), system=system, backend=meta["backend"], method=meta["method"],
               seed=int(meta["seed"]), N=N, n_steps=n_steps, budget_fraction=float(meta["budget_fraction"]),
               threshold=thr, tau_censored=bool(tau["censored"]),
               tau_t=(float(t[i]) if i >= 0 else math.inf), tau_u=(float(u[i]) if i >= 0 else math.inf),
               tau_fe=(float(nfe[i]) if i >= 0 else math.inf), tau_wall=(float(wall[i]) if i >= 0 else math.inf),
               final_e_F=float(eF[-1]), wall_total_s=float(r["wall_total_s"]),
               warmup_s=float(meta.get("warmup_s", math.nan)), cuda_init_s=float(meta.get("cuda_init_s", 0.0) or 0.0),
               import_s=float(meta.get("import_s", math.nan)), engine_import_s=float(meta.get("engine_import_s", math.nan)),
               us_per_step=float(r["us_per_step"]), ns_per_walker_step=float(r["ns_per_walker_step"]),
               peak_rss_mb=float(r["peak_rss_mb"]), peak_gpu_alloc_mb=float(r["peak_gpu_alloc_mb"]),
               peak_gpu_reserved_mb=float(r["peak_gpu_reserved_mb"]), peak_gpu_device_mb=dev_mb,
               process_wall_s=PB.run_cost_s(meta),
               shares_production_init=bool((meta.get("engine_meta") or {}).get(
                   "shares_production_init", meta["backend"] != "cpu_numba_1core" and meta.get("kind") == "gateway")),
               fr_time_s=float(np.asarray(r["fr_time_s_at_save"], float)[-1]),
               n_fr_opps=int(r["n_fr_opps"]),
               bitwise_vs_production=meta.get("bitwise_vs_production", {}).get("status"),
               other_gpu_processes=[p for p in ((meta.get("gpu_processes_before") or [])
                                                + (meta.get("gpu_processes_after") or [])
                                                + ((meta.get("gpu_monitor") or {}).get("other_processes") or []))
                                    if int(p.get("pid", -1)) != int(meta.get("pid", -2))],
               gpu_occupancy=_gpu_occupancy(meta),
               cpu_busy_fraction=meta.get("cpu_busy_fraction"), smt_siblings=meta.get("smt_siblings"),
               affinity=meta.get("affinity"), loadavg=[meta.get("loadavg_before"), meta.get("loadavg_after")],
               device_name=meta.get("device_name"))
    state, sbm = PB.smt_state(meta.get("affinity"), meta.get("smt_siblings"), meta.get("cpu_busy_fraction") or {})
    out["smt_state"] = state
    out["smt_sibling_busy_max"] = sbm
    out["smt_contended"] = None if state == "unknown" else (state == "contended")      # tri-state
    try:
        iu, exact = M.uniform_indices(step, n_steps)
        out["Ibar_F"] = float(np.mean(eF[iu]))
        out["Ibar_F_exact_grid"] = bool(exact)
    except M.MetricsError:
        out["Ibar_F"] = math.nan
        out["Ibar_F_exact_grid"] = False
    curve = dict(save_step=step, save_t=t, save_u=u, wall=wall, nfe=nfe, e_F=eF)
    return out, curve


def _gpu_occupancy(meta):
    """gpu_torch: 'exclusive' | 'shared' | 'unknown' (nvidia-smi unreadable before / during / after); CPU: 'n/a'."""
    if meta.get("backend") != "gpu_torch":
        return "n/a"
    if meta.get("gpu_occupancy"):
        return meta["gpu_occupancy"]
    b, a = meta.get("gpu_processes_before"), meta.get("gpu_processes_after")     # older files
    if b is None or a is None:
        return "unknown"
    pid = int(meta.get("pid", -2))
    return "shared" if any(int(p.get("pid", -1)) != pid for p in b + a) else "exclusive"


def uncontended(runs):
    """{seed: row} restricted to the runs verified uncontended (smt_contended is False; None = unknown excluded)."""
    return {s: v for s, v in runs.items() if v.get("smt_contended") is False}


# ================================================================================================ statistics
def ratio_stats(abf, fr, key):
    """Paired FR / ABF ratio of a tau (inf = censored) per seed: median + bootstrap CI over the both-finite seeds,
    rank part with censoring (eqb_metrics.paired_tau_contrast)."""
    a = {s: v[key] for s, v in abf.items()}
    f = {s: v[key] for s, v in fr.items()}
    pc = M.paired_tau_contrast(a, f, N_BOOT, BOOT_SEED)
    both = [s for s in sorted(set(a) & set(f)) if np.isfinite(a[s]) and np.isfinite(f[s]) and a[s] > 0]
    rat = np.array([f[s] / a[s] for s in both], float)
    med, lo, hi = M.boot_median_ci(rat, N_BOOT, BOOT_SEED)
    return dict(ratio_median=med, ratio_ci95=[lo, hi], n_both_finite=len(both), rank=pc["rank"],
                censored_frac_abf=pc["censored_frac_abf"], censored_frac_fr=pc["censored_frac_fr"],
                abf_median=pc["abf_median"], fr_median=pc["fr_median"],
                per_seed={int(s): [float(a[s]), float(f[s])] for s in sorted(set(a) & set(f))})


def w1_stats(abf, fr):
    """W1_RULE: the paired wall-to-accuracy ratio on the uncontended pairs (primary) and on all pairs."""
    both = sorted(set(abf) & set(fr))
    ua, uf = uncontended(abf), uncontended(fr)
    clean = [s for s in both if s in ua and s in uf]
    excl = [s for s in both if s not in clean]
    out = dict(rule=W1_RULE, n_pairs=len(both), n_pairs_uncontended=len(clean), n_pairs_excluded=len(excl),
               excluded=[dict(seed=int(s), abf=abf[s].get("smt_state"), fr=fr[s].get("smt_state")) for s in excl],
               sensitivity_all_pairs=ratio_stats(abf, fr, "tau_wall"))
    if not clean:
        out.update(status="NOT EVALUABLE (no pair with both runs verified uncontended)", primary=None, agree=None)
        return out
    pr = ratio_stats({s: abf[s] for s in clean}, {s: fr[s] for s in clean}, "tau_wall")
    lo, hi = pr["ratio_ci95"]
    am = out["sensitivity_all_pairs"]["ratio_median"]
    agree = bool(np.isfinite(am) and np.isfinite(lo) and lo <= am <= hi) if np.isfinite(am) else None
    out.update(status="EVALUATED", primary=pr, agree=agree)
    return out


def overhead_stats(abf, fr, key="wall_total_s"):
    both = sorted(set(abf) & set(fr))
    x = np.array([(fr[s][key] - abf[s][key]) / abf[s][key] for s in both], float)
    med, lo, hi = M.boot_median_ci(x, N_BOOT, BOOT_SEED)
    return dict(n=len(both), median=med, ci95=[lo, hi], per_seed={int(s): float(v) for s, v in zip(both, x)})


def perm_median_test(backend, reference, nperm=EQUIV_NPERM, seed=EQUIV_SEED):
    """Two-sample permutation test of d = median(backend) - median(reference): nperm random relabellings of the
    pooled values, two-sided p = (1 + #{|d*| >= |d|}) / (1 + nperm) (exact under H0: both iid from one law).  Also the
    central 95 % of median(reference) + d* (where the backend median falls under H0), the null sd of d and the
    minimum detectable shift |d| at MDS_POWER for each alpha in MDS_ALPHAS (normal approximation)."""
    from scipy.stats import norm
    b = np.asarray([v for v in backend if np.isfinite(v)], float)
    r = np.asarray([v for v in reference if np.isfinite(v)], float)
    out = dict(n=int(b.size), n_reference=int(r.size), nperm=int(nperm))
    if b.size == 0 or r.size == 0:
        return dict(out, p=math.nan, d=math.nan, backend_median_null95=[math.nan, math.nan], sd_null=math.nan,
                    mds_abs={str(a): math.nan for a in MDS_ALPHAS})
    d_obs = float(np.median(b) - np.median(r))
    pooled = np.concatenate([b, r])
    rng = np.random.default_rng(seed)
    perm = rng.permuted(np.broadcast_to(pooled, (int(nperm), pooled.size)), axis=1)
    d = np.median(perm[:, :b.size], axis=1) - np.median(perm[:, b.size:], axis=1)
    tol = 1e-12 * max(1.0, abs(d_obs))
    p = (1.0 + float(np.sum(np.abs(d) >= abs(d_obs) - tol))) / (1.0 + nperm)
    sd = float(np.std(d))
    mr = float(np.median(r))
    zb = norm.ppf(MDS_POWER)
    return dict(out, p=min(1.0, p), d=d_obs, backend_median=float(np.median(b)), reference_median=mr,
                backend_median_null95=[mr + float(np.percentile(d, 2.5)), mr + float(np.percentile(d, 97.5))],
                sd_null=sd, mds_abs={str(a): float((norm.ppf(1 - a / 2) + zb) * sd) for a in MDS_ALPHAS},
                mds_rel={str(a): (float((norm.ppf(1 - a / 2) + zb) * sd / abs(mr)) if mr else math.nan)
                         for a in MDS_ALPHAS})


def holm(pvals, alpha=EQUIV_ALPHA):
    """Holm-Bonferroni: {name: p} -> {name: (p_adj, rejected)}."""
    items = sorted(((float(p) if np.isfinite(p) else 1.0), k) for k, p in pvals.items())
    m = len(items)
    out, running, stop = {}, 0.0, False
    for i, (p, k) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        rej = (not stop) and p <= alpha / (m - i)
        if not rej:
            stop = True
        out[k] = (adj, bool(rej))
    return out


def production_summary(system):
    P, N, _ = PB.system_spec(system)
    p = os.path.join(ROOT, "results", "equal_budget_v2", PB.SYSTEMS[system]["out_dir"], "analysis", "summary.json")
    if not os.path.exists(p):
        return None, p
    with open(p) as fh:
        d = json.load(fh)
    key = str(N)
    if key not in d.get("per_N", {}):
        return None, p
    pn, ct = d["per_N"][key], d["contrasts"][key]

    def num(v):              # json_safe convention: "inf" = censored, null = undefined (NaN)
        return math.nan if v is None else float(v)
    out = dict(path=os.path.relpath(p, ROOT), N=N,
               Ibar_F={m: {int(s): num(v) for s, v in pn[m]["Ibar_F"]["per_seed"].items()} for m in PB.METHODS},
               G_Ibar_F={int(s): num(v) for s, v in ct["primary"]["Ibar_F"]["per_seed"].items()},
               G_Ibar_F_median=num(ct["primary"]["Ibar_F"]["G_median"]),
               G_Ibar_F_ci95=[num(v) for v in ct["primary"]["Ibar_F"]["G_ci95"]],
               tau_t={m: {int(s): num(v) for s, v in pn[m]["tau_e_F_mid_t"]["per_seed"].items()} for m in PB.METHODS})
    return out, p


def consistency_tests(back, ref, ref_G=None, drop=(), nperm=EQUIV_NPERM, min_pairs=EQUIV_MIN_PAIRS):
    """The tests of EQUIVALENCE_RULE on per-seed values: back / ref = {method: {seed: value}} (backend / production);
    ref_G = the production per-seed G (default: (FR - ABF) / ABF of ref); seeds in ``drop`` leave the reference.
    Returns detail, p_values, holm, rejected, n_complete_pairs, min_detectable_shift and the status."""
    from scipy.stats import mannwhitneyu
    drop = set(int(x) for x in drop)
    a, f = back.get("abf", {}), back.get("fr", {})
    pairs = [s_ for s_ in sorted(set(a) & set(f)) if np.isfinite(a[s_]) and np.isfinite(f[s_])]
    if ref_G is None:
        ra, rf = ref.get("abf", {}), ref.get("fr", {})
        ref_G = {s_: (rf[s_] - ra[s_]) / ra[s_] for s_ in set(ra) & set(rf)}
    pv, detail = {}, {}
    for m in PB.METHODS:
        b = np.array(list(back.get(m, {}).values()), float)
        b = b[np.isfinite(b)]
        pr = np.array([v for s_, v in ref.get(m, {}).items() if s_ not in drop], float)
        pr = pr[np.isfinite(pr)]
        if b.size == 0 or pr.size == 0:
            continue
        pt = perm_median_test(b, pr, nperm=nperm)
        mw = mannwhitneyu(b, pr, alternative="two-sided")
        pmed, plo, phi = M.boot_median_ci(pr, N_BOOT, BOOT_SEED)
        detail[m] = dict(backend_median=float(np.median(b)), backend_n=int(b.size), production_median=pmed,
                         production_n=int(pr.size), production_ci95_own_median=[plo, phi], perm=pt,
                         mannwhitney_p=float(mw.pvalue),
                         inside_production_ci95_descriptive=bool(plo <= np.median(b) <= phi))
        pv[f"R_median_{m}"] = pt["p"]
        pv[f"R_mannwhitney_{m}"] = float(mw.pvalue)
    G = np.array([(f[s_] - a[s_]) / a[s_] for s_ in pairs], float)
    G = G[np.isfinite(G)]
    if G.size:
        pg = np.array([v for s_, v in ref_G.items() if s_ not in drop], float)
        pg = pg[np.isfinite(pg)]
        if pg.size:
            pt = perm_median_test(G, pg, nperm=nperm)
            detail["G"] = dict(backend_median=float(np.median(G)), backend_n=int(G.size),
                               production_median=float(np.median(pg)), production_n=int(pg.size), perm=pt)
            pv["R_median_G"] = pt["p"]
    hm = holm(pv)
    rej = sorted(k for k, v in hm.items() if v[1])
    out = dict(detail=detail, p_values=pv, holm={k: dict(p_adj=v[0], rejected=v[1]) for k, v in hm.items()},
               rejected=rej, n_complete_pairs=len(pairs),
               min_detectable_shift={k: dict(rel_to_reference_median=d["perm"].get("mds_rel"),
                                             abs=d["perm"]["mds_abs"], power=MDS_POWER,
                                             note="normal approximation from the permutation null; alpha = per-test "
                                                  "level (0.01 = Holm's strictest step, 0.05 = its most lenient)")
                                     for k, d in detail.items()})
    if len(pv) < 5:
        out["status"] = "INCOMPLETE (a test could not be computed)"
    elif len(pairs) < min_pairs or any(d["backend_n"] < min_pairs for d in detail.values()):
        out["status"] = (f"INCOMPLETE ({len(pairs)} complete pairs < {min_pairs}: tests reported, no verdict"
                         + (f"; {len(rej)} Holm rejection(s) at this n" if rej else "") + ")")
    else:
        out["status"] = "INCONSISTENT" if rej else "CONSISTENT"
    return out


def _shared_drop(runs_by_method):
    """Seeds whose runs share randomness with the production run of the same seed (gateway torch: x0 / y0)."""
    allr = [r for m in PB.METHODS for r in runs_by_method.get(m, {}).values()]
    shared = any(r.get("shares_production_init", False) for r in allr)
    return (sorted({int(s_) for m in PB.METHODS for s_ in runs_by_method.get(m, {})}) if shared else []), shared


def equivalence(system, backend, runs_by_method, nperm=EQUIV_NPERM, min_pairs=EQUIV_MIN_PAIRS):
    """EQUIVALENCE_RULE for one torch backend of one system."""
    rec = dict(rule=EQUIVALENCE_RULE, backend=backend, system=system, min_pairs=int(min_pairs))
    allr = [r for m in PB.METHODS for r in runs_by_method.get(m, {}).values()]
    if not allr:
        rec["status"] = "NOT TESTED (no runs)"
        return rec
    if any(r["budget_fraction"] < 1.0 for r in allr):
        rec["status"] = "NOT APPLICABLE (shortened budget: the production comparison needs the full budget)"
        return rec
    prod, p = production_summary(system)
    if prod is None:
        rec["status"] = f"NOT APPLICABLE (no production summary at {os.path.relpath(p, ROOT)})"
        return rec
    drop, shared = _shared_drop(runs_by_method)
    rec["production"] = dict(path=prod["path"], n_seeds={m: len(prod["Ibar_F"][m]) for m in PB.METHODS},
                             G_median=prod["G_Ibar_F_median"], G_ci95=prod["G_Ibar_F_ci95"],
                             reference_excludes_seeds=[s_ for s_ in drop if any(s_ in prod["Ibar_F"][m]
                                                                                for m in PB.METHODS)],
                             why_excluded=("the backend's runs reproduce these production seeds' initial conditions"
                                           if shared else "nothing excluded: independent randomness"))
    back = {m: {s_: r["Ibar_F"] for s_, r in runs_by_method.get(m, {}).items()} for m in PB.METHODS}
    ct = consistency_tests(back, prod["Ibar_F"], ref_G=prod["G_Ibar_F"], drop=drop, nperm=nperm, min_pairs=min_pairs)
    if "G" in ct["detail"]:
        g = ct["detail"].pop("G")
        g.update(production_median_all_seeds=prod["G_Ibar_F_median"], production_ci95_own_median=prod["G_Ibar_F_ci95"],
                 inside_production_ci95_descriptive=bool(
                     prod["G_Ibar_F_ci95"][0] <= g["backend_median"] <= prod["G_Ibar_F_ci95"][1]))
        ct["detail"]["G_Ibar_F"] = g
        ct["min_detectable_shift"]["G_Ibar_F"] = ct["min_detectable_shift"].pop("G")
    rec.update(ct)
    return rec


# EARLY_RULE (fixed 2026-10-10 before any Experiment III benchmark data; EXPLORATORY: the plan preregisters no
# reduced-cost check, this one is offered for the Experiment III amendment because the full-budget gateway gate does
# not fit the GPU ceiling).
EARLY_RULE = ("EXPLORATORY (not preregistered; proposed for the Experiment III amendment): for shortened-budget runs "
              "of one n_steps, Ebar_common = mean e_F over the saves common to the run's grid and the production "
              "N = 512 grid (identical integration steps, hence identical simulation times), scored with the same "
              "scorer; the EQUIVALENCE_RULE tests (reference without seeds sharing randomness; permutation median and "
              "Mann-Whitney per arm, permutation on the paired G; Holm at 0.05; >= 8 complete pairs, else "
              "INCOMPLETE) against the production per-seed Ebar_common at the same saves.  It tests equality in law "
              "up to the run's end only.")
_PROD_EARLY = {}


def production_early_values(system, steps):
    """{method: {seed: mean e_F over the production saves at ``steps``}} from the equal-budget N = 512 files."""
    key = (system, tuple(int(x) for x in steps))
    if key in _PROD_EARLY:
        return _PROD_EARLY[key]
    sc, P, N, _ = scorer_for(system)
    out = {}
    for m in PB.METHODS:
        for seed in P["seeds"]:
            p = PB.production_path(system, seed, m)
            if not os.path.exists(p):
                continue
            with np.load(p, allow_pickle=False) as z:
                pos = np.searchsorted(z["save_step"], steps)
                if np.any(pos >= z["save_step"].size) or not np.array_equal(z["save_step"][pos], steps):
                    raise M.MetricsError(f"{p}: the common saves are not on its grid")
                res = dict(M_all=z["M_all"][pos], C_all=z["C_all"][pos], meta=dict(T_K=P.get("T_K")))
                if "M_prod" in z.files:
                    res.update(M_prod=z["M_prod"][pos], C_prod=z["C_prod"][pos])
            out.setdefault(m, {})[int(seed)] = float(np.mean(np.asarray(sc.errors(res)["e_F"], float)))
    _PROD_EARLY[key] = out
    return out


def early_window_consistency(system, backend, runs_by_method, curves_by_method, nperm=EQUIV_NPERM,
                             min_pairs=EQUIV_MIN_PAIRS):
    """EARLY_RULE for one backend of one system (shortened-budget runs only)."""
    rec = dict(rule=EARLY_RULE, backend=backend, system=system, label="EXPLORATORY")
    allr = [r for m in PB.METHODS for r in runs_by_method.get(m, {}).values()]
    if not allr:
        return dict(rec, status="NOT TESTED (no runs)")
    nst = sorted({r["n_steps"] for r in allr})
    if any(r["budget_fraction"] >= 1.0 for r in allr) or len(nst) != 1:
        return dict(rec, status="NOT APPLICABLE (needs shortened-budget runs of ONE n_steps)")
    P, N, _ = PB.system_spec(system)
    p0 = PB.production_path(system, P["seeds"][0], "abf")
    if not os.path.exists(p0):
        return dict(rec, status=f"NOT APPLICABLE (no production file {os.path.relpath(p0, ROOT)})")
    with np.load(p0, allow_pickle=False) as z:
        prod_steps = np.asarray(z["save_step"], np.int64)
    cv0 = next(iter(next(iter(curves_by_method.values())).values()))
    common = np.intersect1d(np.asarray(cv0["save_step"], np.int64), prod_steps)
    if common.size < 3:
        return dict(rec, status=f"NOT APPLICABLE ({common.size} saves common with the production grid < 3)")
    back = {}
    for m in PB.METHODS:
        for s_, cv in curves_by_method.get(m, {}).items():
            pos = np.searchsorted(cv["save_step"], common)
            back.setdefault(m, {})[int(s_)] = float(np.mean(np.asarray(cv["e_F"], float)[pos]))
    drop, shared = _shared_drop(runs_by_method)
    ref = production_early_values(system, common)
    rec.update(n_steps=nst[0], common_steps=common.tolist(), common_t=(common * float(P["engine_cfg"]["h"])).tolist(),
               reference_excludes_seeds=[s_ for s_ in drop if any(s_ in ref.get(m, {}) for m in PB.METHODS)],
               backend_values=back)
    rec.update(consistency_tests(back, ref, drop=drop, nperm=nperm, min_pairs=min_pairs))
    rec["status"] = "EXPLORATORY " + rec["status"]
    return rec


def production_crosscheck(system, runs_by_method, tol=1e-12):
    """Full-budget numba runs of production seeds reproduce the production computation bitwise, so their Ibar_F and
    simulation-time tau must equal the frozen equal-budget summary's per-seed values (a check of the whole pipeline:
    engine, save grid, timing hook, scorer)."""
    allr = [r for m in PB.METHODS for r in runs_by_method.get(m, {}).values()]
    if not allr or any(r["budget_fraction"] < 1.0 for r in allr):
        return dict(status="NOT APPLICABLE (shortened budget)")
    prod, p = production_summary(system)
    if prod is None:
        return dict(status=f"NOT APPLICABLE (no production summary at {os.path.relpath(p, ROOT)})")
    bad, n = [], 0
    for m in PB.METHODS:
        for s, r in runs_by_method.get(m, {}).items():
            if s not in prod["Ibar_F"][m]:
                continue
            n += 1
            pi, pt = prod["Ibar_F"][m][s], prod["tau_t"][m].get(s, math.nan)
            if not (abs(r["Ibar_F"] - pi) <= tol * max(1.0, abs(pi))):
                bad.append(f"{m} s{s} Ibar_F {r['Ibar_F']!r} vs {pi!r}")
            if not ((math.isinf(pt) and math.isinf(r["tau_t"])) or abs(r["tau_t"] - pt) <= tol * max(1.0, abs(pt))):
                bad.append(f"{m} s{s} tau_t {r['tau_t']!r} vs {pt!r}")
    return dict(status=("EQUAL" if not bad else "DIFFERENT") if n else "NOT APPLICABLE (no production seeds)",
                n_runs_compared=n, mismatches=bad[:20])


def cost_projection(per_backend_runs, systems, seeds_per_arm=PB.DEFAULT_SEEDS_PER_ARM):
    """Full-campaign projection: per (system, backend, method) the median measured us/step of the analysed runs x the
    full n_steps x seeds_per_arm, + the median per-run overhead (process_wall_s - wall_total_s); a (system, method)
    with no analysed run uses parallel_benchmark.FALLBACK_US_PER_STEP (2026-10-10 measurements, host sibling busy)
    and RUN_OVERHEAD_FALLBACK_S, marked as such.  Labelled a projection; the per-step cost of a shortened run
    under-represents phases it did not reach (e.g. FR after the LTA FR start).  ceiling_check compares the gpu_torch
    projection, per system and in total, with the plan section 9 GPU ceiling (8 GPU-h) and estimate (<= 3 GPU-h):
    the equivalence gate of a system needs exactly its full-budget runs, so a system projected above the ceiling
    cannot have its GPU gate evaluated within the plan."""
    out = {}
    for backend in BACKENDS_EXP:
        tot_s, rows, per_sys = 0.0, {}, {}
        for system in systems:
            P, N, _ = PB.system_spec(system)
            nf = PB.full_n_steps(P, N)
            for m in PB.METHODS:
                rr = per_backend_runs.get((system, backend), {}).get(m, {})
                if rr:
                    us = float(np.median([r["us_per_step"] for r in rr.values()]))
                    ov = float(np.median([max(r["process_wall_s"] - r["wall_total_s"], 0.0) for r in rr.values()]))
                    src = f"measured ({len(rr)} runs)"
                    meas = sorted({r["n_steps"] for r in rr.values()})
                    cont = sorted({str(r["smt_state"]) for r in rr.values()})
                else:
                    us = PB.FALLBACK_US_PER_STEP[(backend, system)]
                    ov, src, meas, cont = PB.RUN_OVERHEAD_FALLBACK_S, "fallback (FALLBACK_US_PER_STEP)", [], ["contended"]
                per_run = us * 1e-6 * nf + ov
                rows[f"{system}/{m}"] = dict(us_per_step=us, n_steps_full=nf, s_per_run=per_run, overhead_s=ov,
                                             runs=int(seeds_per_arm), s_total=per_run * seeds_per_arm, source=src,
                                             measured_n_steps=meas, host_smt_states=cont)
                tot_s += per_run * seeds_per_arm
                per_sys[system] = per_sys.get(system, 0.0) + per_run * seeds_per_arm / 3600.0
        unit = "GPU-h (one device, runs sequential)" if backend == "gpu_torch" else "core-h"
        out[backend] = dict(rows=rows, total_h=tot_s / 3600.0, per_system_h=per_sys, unit=unit,
                            seeds_per_arm=int(seeds_per_arm),
                            all_measured=all(rows[f"{s_}/{m}"]["source"].startswith("measured")
                                             for s_ in systems for m in PB.METHODS))
    g = out.get("gpu_torch")
    if g:
        g["ceiling_check"] = dict(
            plan_gpu_ceiling_h=PLAN_GPU_CEILING_H, plan_exp3_gpu_estimate_h=PLAN_GPU_ESTIMATE_H,
            total_within_ceiling=bool(g["total_h"] <= PLAN_GPU_CEILING_H),
            per_system={s_: dict(gpu_h=h, within_ceiling=bool(h <= PLAN_GPU_CEILING_H),
                                 within_estimate=bool(h <= PLAN_GPU_ESTIMATE_H),
                                 equivalence_gate=("evaluable within the GPU ceiling" if h <= PLAN_GPU_CEILING_H else
                                                   "CANNOT BE EVALUATED within the plan's 8 GPU-h ceiling (its "
                                                   f"{seeds_per_arm} x 2 full-budget runs alone need {h:.1f} GPU-h)"))
                        for s_, h in g["per_system_h"].items()})
    c = out.get("cpu_numba_1core")
    if c:
        c["ceiling_check"] = dict(plan_exp3_cpu_estimate_core_h=PLAN_CPU_ESTIMATE_H,
                                  total_within_estimate=bool(c["total_h"] <= PLAN_CPU_ESTIMATE_H))
    return out


# ================================================================================================ figures
def setup_style():
    try:
        import plot_config as PC
        PC.setup_style()
    except Exception:  # noqa: BLE001
        plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                             "axes.spines.right": False, "pdf.fonttype": 42})


def save_fig(fig, fig_dir, tag, manifest, waivers=()):
    os.makedirs(fig_dir, exist_ok=True)
    base = os.path.join(fig_dir, f"parallel_benchmark_{tag}")
    fig.savefig(base + ".png", dpi=160)
    fig.savefig(base + ".pdf")
    leg = LG.check_figure_safe(fig, waivers=waivers)
    plt.close(fig)
    manifest[tag] = dict(files=[os.path.relpath(base + ".png", ROOT), os.path.relpath(base + ".pdf", ROOT)],
                         legibility=leg)
    if leg.get("status") != "pass":
        print(LG.one_line(tag, leg), file=sys.stderr)


def _log_ticks(ax, axis="y"):
    """Labelled 1-2-5 ticks on a log axis (a log axis spanning less than a decade would otherwise show one label)."""
    import matplotlib.ticker as mt
    a = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi / lo < 1e4):
        return
    a.set_major_locator(mt.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(mt.NullFormatter())


def _cats(systems, data):
    """Backend x arm categories that have runs, in display order."""
    cats = []
    for b in BACKEND_ORDER:
        for m in PB.METHODS:
            if any(data.get((s, b), {}).get(m) for s in systems):
                cats.append((b, m))
    return cats


def fig_time_to_accuracy(systems, data, stats, fig_dir, manifest):
    keys = [("tau_t", "simulation time to accuracy (t.u.)"), ("tau_fe", "force evaluations to accuracy"),
            ("tau_wall", "wall-clock to accuracy (s)")]
    cats = _cats(systems, data)
    if not cats:
        return
    fig, axs = plt.subplots(len(systems), 3, figsize=(12.6, 3.4 * len(systems) + 1.6), squeeze=False)
    for r, system in enumerate(systems):
        _, _, _, thr = scorer_for(system)
        for c, (key, lab) in enumerate(keys):
            ax = axs[r, c]
            fin_all, n_cens = [], {}
            for j, (b, m) in enumerate(cats):
                rr = data.get((system, b), {}).get(m, {})
                v = np.array([x[key] for x in rr.values()], float)
                fin_all += [x for x in v if np.isfinite(x)]
                n_cens[(b, m)] = int(np.sum(~np.isfinite(v)))
            top = (max(fin_all) * 3.0) if fin_all else 1.0
            bot = (min(fin_all) / 3.0) if fin_all else 0.1
            pos = {}
            for j, (b, m) in enumerate(cats):
                rr = data.get((system, b), {}).get(m, {})
                if not rr:
                    continue
                seeds = sorted(rr)
                v = np.array([rr[s][key] for s in seeds], float)
                x = j + np.linspace(-0.12, 0.12, len(seeds)) if len(seeds) > 1 else np.array([float(j)])
                pos[(b, m)] = dict(zip(seeds, x.tolist()))
                fin = np.isfinite(v)
                unc = np.array([rr[s]["smt_contended"] is False for s in seeds]) if key == "tau_wall" \
                    else np.ones(len(seeds), bool)
                ax.scatter(x[fin & unc], v[fin & unc], s=16, color=METHOD_COLOR[m], marker=BACKEND_MARK[b], alpha=0.75,
                           zorder=3, linewidths=0)
                if np.any(fin & ~unc):
                    ax.scatter(x[fin & ~unc], v[fin & ~unc], s=18, marker=BACKEND_MARK[b], facecolors="none",
                               edgecolors=METHOD_COLOR[m], linewidths=0.9, zorder=3)
                if np.any(~fin):
                    ax.scatter(x[~fin], np.full(int(np.sum(~fin)), top), s=22, marker="^", facecolors="none",
                               edgecolors=METHOD_COLOR[m], zorder=3)
                if np.any(fin):
                    med, lo, hi = M.boot_median_ci(v[fin], N_BOOT, BOOT_SEED)
                    if np.all(fin):
                        ax.errorbar([j + 0.28], [med], yerr=[[med - lo], [hi - med]], fmt="_", color=INK, ms=10,
                                    capsize=2.5, lw=1.1, zorder=4)
            for (b, m), px in pos.items():       # same seed, ABF to ABF+FR: from point to point
                if m != "fr" or (b, "abf") not in pos:
                    continue
                ra, rr = data[(system, b)]["abf"], data[(system, b)]["fr"]
                for s, xf in px.items():
                    if s in ra and np.isfinite(ra[s][key]) and np.isfinite(rr[s][key]):
                        ax.plot([pos[(b, "abf")][s], xf], [ra[s][key], rr[s][key]], color=MUTED, lw=0.5, alpha=0.6,
                                zorder=2)
            ax.set_yscale("log")
            ax.set_ylim(bot, top * 1.8)
            _log_ticks(ax)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels([f"{BACKEND_SHORT[b]}\n{'FR' if m == 'fr' else 'ABF'}"
                                + (f"\n{n_cens[(b, m)]} cens." if n_cens[(b, m)] else "") for b, m in cats], fontsize=7.5)
            ax.set_xlim(-0.5, len(cats) - 0.4)
            ax.set_ylabel(lab, fontsize=9)
            ax.set_xlabel("backend / arm", fontsize=9)
            ax.set_title(f"{system}, persistent e_F <= {thr:g}", fontsize=9)
            if not fin_all:
                ax.text(0.5, 0.3, "every seed censored\n(threshold not reached persistently)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color=MUTED)
    hs = [Line2D([], [], color=METHOD_COLOR[m], marker="o", ls="none", label=METHOD_LABEL[m]) for m in PB.METHODS]
    hs += [Line2D([], [], color=INK, marker="_", ms=10, ls="none", label="median, 95 % bootstrap CI (no seed censored)"),
           Line2D([], [], color=MUTED, lw=0.8, label="same seed, ABF to ABF+FR"),
           Line2D([], [], marker="^", color=INK, mfc="none", ls="none", label="censored (never persistently below)"),
           Line2D([], [], marker="o", color=INK, mfc="none", ls="none",
                  label="wall: host SMT sibling busy or unknown (excluded from W1)")]
    fig.legend(handles=hs, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Experiment III: persistent time-to-accuracy at the frozen mid e_F threshold, N = 512 "
                 "(preregistered endpoints; one point per seed)", fontsize=10)
    H = fig.get_figheight()
    fig.subplots_adjust(left=0.07, right=0.99, top=1.0 - 0.75 / H, bottom=1.6 / H, hspace=0.62, wspace=0.28)
    save_fig(fig, fig_dir, "T_time_to_accuracy", manifest)


def fig_error_vs_wall(systems, curves, fig_dir, manifest):
    """EXPLORATORY: e_F against the wall clock of the timed run, one row per system, one column per backend, the x
    axis shared along a row (so the backends' walls compare directly); one figure-level key."""
    backends = [b for b in BACKEND_ORDER if any((s, b) in curves for s in systems)]
    if not backends:
        return
    ncol = len(backends)
    W = max(4.6 * ncol + 1.2, 8.0)                      # wide enough for the two-line title even with one column
    H = 3.0 * len(systems) + 1.9
    fig, axs = plt.subplots(len(systems), ncol, figsize=(W, H), squeeze=False, sharex="row")
    for r, system in enumerate(systems):
        _, _, _, thr = scorer_for(system)
        for c, b in enumerate(backends):
            ax = axs[r, c]
            ns = []
            for m in PB.METHODS:
                cs = curves.get((system, b), {}).get(m, {})
                if not cs:
                    continue
                Wl = np.array([cs[s]["wall"] for s in sorted(cs)], float)
                E = np.array([cs[s]["e_F"] for s in sorted(cs)], float)
                x = np.nanmedian(Wl, axis=0)
                med = np.nanmedian(E, axis=0)
                lo, hi = np.nanpercentile(E, 25, axis=0), np.nanpercentile(E, 75, axis=0)
                ok = x > 0
                ax.plot(x[ok], med[ok], color=METHOD_COLOR[m], lw=1.5)
                ax.fill_between(x[ok], lo[ok], hi[ok], color=METHOD_COLOR[m], alpha=0.18, lw=0)
                ns.append(f"{'FR' if m == 'fr' else 'ABF'} n = {len(cs)}")
            ax.axhline(thr, color=INK, lw=0.9, ls=(0, (5, 2.5)))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("wall clock of the timed run (s)", fontsize=9)
            ax.set_ylabel("free-energy error e_F", fontsize=9)
            ax.set_title(f"{system}: {BACKEND_LABEL[b]}" + (f" ({', '.join(ns)})" if ns else ""), fontsize=9)
            if not ns:
                ax.text(0.5, 0.5, "NOT TESTED", transform=ax.transAxes, ha="center", va="center", color=MUTED)
    hs = [Line2D([], [], color=METHOD_COLOR[m], lw=1.5, label=f"{METHOD_LABEL[m]}: median across seeds")
          for m in PB.METHODS]
    hs += [plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.3, lw=0, label="interquartile range across seeds"),
           Line2D([], [], color=INK, lw=0.9, ls=(0, (5, 2.5)), label="frozen mid threshold")]
    fig.legend(handles=hs, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("EXPLORATORY (not a preregistered endpoint): e_F against wall-clock seconds\n"
                 "x = median wall at each save across seeds; panels in a row share the wall axis", fontsize=9.5)
    fig.subplots_adjust(left=0.75 / W + 0.04, right=0.98, top=1.0 - 0.85 / H, bottom=1.25 / H, hspace=0.62,
                        wspace=0.28)
    save_fig(fig, fig_dir, "X_error_vs_wall", manifest)


def fig_cost(systems, data, stats, fig_dir, manifest):
    cats = _cats(systems, data)
    if not cats:
        return
    fig, axs = plt.subplots(1, 4, figsize=(15.5, 5.0))
    xs = np.arange(len(systems))
    width = 0.8 / max(len(cats), 1)
    # (a) us per step
    ax = axs[0]
    for j, (b, m) in enumerate(cats):
        for i, system in enumerate(systems):
            rr = data.get((system, b), {}).get(m, {})
            if not rr:
                continue
            v = np.array([x["us_per_step"] for x in rr.values()], float)
            cont = np.array([x["smt_contended"] is not False for x in rr.values()])     # contended or unknown
            x0 = i - 0.4 + width * (j + 0.5)
            ax.scatter(np.full(int(np.sum(~cont)), x0), v[~cont], s=14, color=METHOD_COLOR[m], marker=BACKEND_MARK[b],
                       alpha=0.8, linewidths=0)
            if np.any(cont):
                ax.scatter(np.full(int(np.sum(cont)), x0), v[cont], s=18, marker=BACKEND_MARK[b], facecolors="none",
                           edgecolors=METHOD_COLOR[m], linewidths=0.9)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(systems)
    ax.set_xlabel("system")
    ax.set_ylabel("wall per integration step (us)")
    _log_ticks(ax)
    ax.set_title("(a) per-step wall, N = 512\n(open: SMT sibling busy > 10 % or unknown)", fontsize=9)
    # (b) effective parallelism
    ax = axs[1]
    any_b = False
    for i, system in enumerate(systems):
        for k, m in enumerate(PB.METHODS):
            ep = stats.get(system, {}).get("effective_parallelism", {})
            e = ep.get(m) if isinstance(ep, dict) else None
            if not e or not np.isfinite(e.get("median", math.nan)):
                continue
            any_b = True
            x0 = i + (k - 0.5) * 0.3
            for vv, dx_, mfc in ((e.get("uncontended", {}), -0.05, METHOD_COLOR[m]), (e, 0.05, "white")):
                med = vv.get("median", math.nan)
                if not np.isfinite(med):
                    continue
                lo, hi = vv["ci95"]
                ax.errorbar([x0 + dx_], [med], yerr=[[med - lo], [hi - med]], fmt="o", color=METHOD_COLOR[m],
                            mfc=mfc, capsize=3)
    ax.axhline(1.0, color=INK, lw=0.8, ls=(0, (5, 2.5)))
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(systems)
    ax.set_xlabel("system")
    ax.set_ylabel("wall(1 CPU core) / wall(1 GPU)")
    _log_ticks(ax)
    ax.set_title("(b) effective parallelism, equal steps\n(filled: uncontended pairs; open: all pairs)", fontsize=9)
    if not any_b:
        ax.text(0.5, 0.5, "NOT TESTED\n(needs both backends)", transform=ax.transAxes, ha="center", va="center",
                color=MUTED)
    # (c) FR overhead
    ax = axs[2]
    backs = [b for b in BACKEND_ORDER if any((s, b) in data for s in systems)]
    for k, b in enumerate(backs):
        for i, system in enumerate(systems):
            bs_ = stats.get(system, {}).get("backends", {}).get(b, {})
            x0 = i + (k - (len(backs) - 1) / 2) * 0.25
            for o, dx_, mfc in ((bs_.get("fr_overhead_uncontended"), -0.05, INK), (bs_.get("fr_overhead"), 0.05, "white")):
                if not o or not np.isfinite(o.get("median", math.nan)):
                    continue
                med, (lo, hi) = 100 * o["median"], [100 * o["ci95"][0], 100 * o["ci95"][1]]
                ax.errorbar([x0 + dx_], [med], yerr=[[med - lo], [hi - med]], fmt=BACKEND_MARK[b], color=INK, mfc=mfc,
                            capsize=3)
    ax.axhline(0.0, color=MUTED, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(systems)
    ax.set_xlabel("system")
    ax.set_ylabel("FR added wall (%) = (FR - ABF) / ABF")
    ax.set_title("(c) FR overhead, paired seeds\n(filled: uncontended pairs; open: all pairs)", fontsize=9)
    # (d) peak memory
    ax = axs[3]
    for j, (b, m) in enumerate(cats):
        for i, system in enumerate(systems):
            rr = data.get((system, b), {}).get(m, {})
            if not rr:
                continue
            x0 = i - 0.4 + width * (j + 0.5)
            v = np.array([x["peak_rss_mb"] for x in rr.values()], float)
            ax.scatter(np.full(v.size, x0), v, s=14, color=METHOD_COLOR[m], marker=BACKEND_MARK[b], alpha=0.8,
                       linewidths=0)
            if b == "gpu_torch":
                v = np.array([x["peak_gpu_device_mb"] for x in rr.values()], float)
                v = v[np.isfinite(v)]
                ax.scatter(np.full(v.size, x0), v, s=34, color=METHOD_COLOR[m], marker="*", alpha=0.9, linewidths=0)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(systems)
    ax.set_xlabel("system")
    ax.set_ylabel("peak memory (MB)")
    _log_ticks(ax)
    ax.set_title("(d) peak memory: host RSS of every run\n(circle / square), GPU device (star)", fontsize=9)
    hs = [Line2D([], [], color=METHOD_COLOR[m], marker=BACKEND_MARK[b], ls="none",
                 label=f"{BACKEND_SHORT[b]}, {METHOD_LABEL[m]}") for b, m in cats]
    hs += [Line2D([], [], color=INK, marker=BACKEND_MARK[b], mfc="white", ls="none",
                  label=f"FR overhead, {BACKEND_SHORT[b]}") for b in backs]
    hs.append(Line2D([], [], color=INK, lw=0.8, ls=(0, (5, 2.5)), label="(b) equal wall"))
    if any(b == "gpu_torch" for b, _ in cats):
        hs.append(Line2D([], [], color=INK, marker="*", ms=8, ls="none",
                         label="(d) GPU device memory incl. CUDA context (nvidia-smi)"))
    fig.legend(handles=hs, loc="lower center", ncol=min(len(hs), 5), fontsize=8, frameon=False)
    fig.suptitle("Experiment III cost (preregistered 'also reported'); points = seeds, error bars = median and 95 % "
                 "seed-bootstrap CI", fontsize=10)
    for ax in axs:                       # categorical system axis: room for the outer tick labels in every layout
        ax.set_xlim(-0.6, len(systems) - 0.4)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.8, bottom=0.27, wspace=0.42)
    save_fig(fig, fig_dir, "C_cost", manifest)


def fig_equivalence(systems, data, stats, fig_dir, manifest):
    """EXPLORATORY: torch backend vs the validated numba production at N = 512 (EQUIVALENCE_RULE)."""
    rows = []
    for system in systems:
        for b in ("gpu_torch", "cpu_torch_test"):
            eq = stats.get(system, {}).get("backends", {}).get(b, {}).get("equivalence")
            if eq and eq.get("detail"):
                rows.append((system, b, eq))
    if not rows:
        return
    H = 3.1 * len(rows) + 1.7
    fig, axs = plt.subplots(len(rows), 3, figsize=(12.5, H), squeeze=False)
    for r, (system, b, eq) in enumerate(rows):
        prod, _ = production_summary(system)
        for c, key in enumerate(["abf", "fr", "G_Ibar_F"]):
            ax = axs[r, c]
            d = eq["detail"].get(key)
            if not d:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color=MUTED)
                ax.set_xlabel("engine")
                ax.set_ylabel(key)
                continue
            drop = set(eq.get("production", {}).get("reference_excludes_seeds", []))
            if key == "G_Ibar_F":
                pv = np.array([v for s_, v in prod["G_Ibar_F"].items() if s_ not in drop], float)
                a, f = data[(system, b)].get("abf", {}), data[(system, b)].get("fr", {})
                bv = np.array([(f[s]["Ibar_F"] - a[s]["Ibar_F"]) / a[s]["Ibar_F"] for s in sorted(set(a) & set(f))])
                ylab, col = "G(Ibar_F) = (FR - ABF) / ABF", INK
            else:
                pv = np.array([v for s_, v in prod["Ibar_F"][key].items() if s_ not in drop], float)
                bv = np.array([x["Ibar_F"] for x in data[(system, b)].get(key, {}).values()], float)
                ylab, col = f"Ibar_F, {METHOD_LABEL[key]}", METHOD_COLOR[key]
            pv, bv = pv[np.isfinite(pv)], bv[np.isfinite(bv)]
            ax.scatter(np.linspace(-0.15, 0.15, max(pv.size, 1))[:pv.size], pv, s=12, color=MUTED, linewidths=0)
            ax.scatter(1.0 + np.linspace(-0.15, 0.15, max(bv.size, 1))[:bv.size], bv, s=18, color=col,
                       marker=BACKEND_MARK[b], linewidths=0)
            lo, hi = d["perm"]["backend_median_null95"]
            ax.fill_between([0.72, 1.28], [lo, lo], [hi, hi], color=MUTED, alpha=0.25, lw=0)
            ax.plot([0.72, 1.28], [d["backend_median"]] * 2, color=INK, lw=1.4)
            ax.plot([-0.28, 0.28], [d["production_median"]] * 2, color=INK, lw=1.4)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"numba production\n(n = {d['perm']['n_reference']} in reference)",
                                f"{BACKEND_SHORT[b]} (n = {bv.size})"])
            ax.set_xlim(-0.5, 1.5)
            ax.set_xlabel("engine")
            ax.set_ylabel(ylab, fontsize=9)
            ax.set_title(f"{system} [{b}]: perm. p(median) = {d['perm']['p']:.2g}"
                         + (f", MWU p = {d['mannwhitney_p']:.2g}" if "mannwhitney_p" in d else ""), fontsize=8.5)
        axs[r, 0].annotate(f"verdict (Holm over R1-R5, >= {EQUIV_MIN_PAIRS} pairs): {eq['status'][:70]}",
                           xy=(0.0, 1.16), xycoords="axes fraction", fontsize=8.5, color=INK)
    hs = [Line2D([], [], color=MUTED, marker="o", ls="none", label="numba production, one point per seed"),
          Line2D([], [], color=INK, marker="s", ls="none", label="torch backend, one point per seed"),
          plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.25, lw=0,
                        label="central 95 % of the backend median under H0 (permutation null)"),
          Line2D([], [], color=INK, lw=1.4, label="median")]
    fig.legend(handles=hs, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("EXPLORATORY (tolerance stated in analyze_parallel_benchmark.py, not in the plan): torch backend vs "
                 "validated numba production, N = 512", fontsize=9.5)
    fig.subplots_adjust(left=0.08, right=0.99, top=1.0 - 0.9 / H, bottom=1.35 / H, hspace=1.0, wspace=0.32)
    save_fig(fig, fig_dir, "Q_equivalence", manifest)


# ================================================================================================ main analysis
def analyze(root, out_dir, fig_dir, seeds_per_arm=PB.DEFAULT_SEEDS_PER_ARM, figures=True):
    found = discover(root)
    data, curves = {}, {}
    for (system, backend), runs in found.items():
        for (seed, method), p in runs.items():
            row, cv = score_run(p, system)
            data.setdefault((system, backend), {}).setdefault(method, {})[seed] = row
            curves.setdefault((system, backend), {}).setdefault(method, {})[seed] = cv
    systems = [s for s in PB.SYSTEMS if any((s, b) in data for b in BACKEND_ORDER)]
    stats = {}
    for system in PB.SYSTEMS:
        st = dict(backends={}, not_tested=[])
        for b in BACKEND_ORDER + tuple(PB.UNAVAILABLE_BACKENDS):
            if (system, b) not in data:
                why = PB.UNAVAILABLE_BACKENDS.get(b, "no runs")
                if b != "cpu_torch_test":
                    st["not_tested"].append(dict(backend=b, label="NOT TESTED", why=why))
                continue
            dm = data[(system, b)]
            a, f = dm.get("abf", {}), dm.get("fr", {})
            bst = dict(n_seeds={m: len(dm.get(m, {})) for m in PB.METHODS},
                       budget_fraction=sorted({r["budget_fraction"] for m in dm for r in dm[m].values()}),
                       n_steps=sorted({r["n_steps"] for m in dm for r in dm[m].values()}))
            for m in PB.METHODS:
                rr = dm.get(m, {})
                if not rr:
                    continue
                def med(k):
                    return M.quantile_inf([x[k] for x in rr.values()], 50)
                bst[m] = dict(tau_t_median=med("tau_t"), tau_fe_median=med("tau_fe"), tau_wall_median=med("tau_wall"),
                              n_censored=int(sum(x["tau_censored"] for x in rr.values())),
                              Ibar_F_median=med("Ibar_F"), final_e_F_median=med("final_e_F"),
                              wall_total_s_median=med("wall_total_s"), us_per_step_median=med("us_per_step"),
                              ns_per_walker_step_median=med("ns_per_walker_step"), warmup_s_median=med("warmup_s"),
                              cuda_init_s_median=med("cuda_init_s"), import_s_median=med("import_s"),
                              engine_import_s_median=med("engine_import_s"),
                              peak_rss_mb_max=float(np.nanmax([x["peak_rss_mb"] for x in rr.values()])),
                              peak_gpu_alloc_mb_max=float(np.nanmax([x["peak_gpu_alloc_mb"] for x in rr.values()]))
                              if b == "gpu_torch" else math.nan,
                              peak_gpu_reserved_mb_max=float(np.nanmax([x["peak_gpu_reserved_mb"] for x in rr.values()]))
                              if b == "gpu_torch" else math.nan,
                              peak_gpu_device_mb_max=(float(np.nanmax([x["peak_gpu_device_mb"] for x in rr.values()]))
                                                      if b == "gpu_torch" and any(np.isfinite(x["peak_gpu_device_mb"])
                                                                                  for x in rr.values()) else math.nan),
                              gpu_occupancy=sorted({x["gpu_occupancy"] for x in rr.values()}),
                              fr_time_frac_median=(float(np.median([x["fr_time_s"] / x["wall_total_s"]
                                                                    for x in rr.values()]))
                                                   if m == "fr" and b != "cpu_numba_1core" else math.nan),
                              bitwise_vs_production=sorted({str(x["bitwise_vs_production"]) for x in rr.values()}),
                              other_gpu_processes=sum(len(x["other_gpu_processes"]) for x in rr.values()),
                              n_smt_contended=int(sum(x["smt_contended"] is True for x in rr.values())),
                              n_smt_unknown=int(sum(x["smt_contended"] is None for x in rr.values())),
                              per_seed={int(s): {k: x[k] for k in ("tau_t", "tau_fe", "tau_wall", "Ibar_F", "final_e_F",
                                                                   "wall_total_s", "us_per_step", "warmup_s",
                                                                   "smt_sibling_busy_max", "smt_contended",
                                                                   "smt_state", "peak_rss_mb", "peak_gpu_device_mb")}
                                        for s, x in sorted(rr.items())})
            if a and f:
                bst["paired"] = {k: ratio_stats(a, f, k) for k in ("tau_t", "tau_fe", "tau_wall")}
                bst["W1"] = w1_stats(a, f)
                bst["G_Ibar_F"] = M.paired_contrast({s: v["Ibar_F"] for s, v in a.items()},
                                                    {s: v["Ibar_F"] for s, v in f.items()}, N_BOOT, BOOT_SEED)
                bst["fr_overhead"] = overhead_stats(a, f)
                ca, cf = uncontended(a), uncontended(f)
                bst["fr_overhead_uncontended"] = overhead_stats(ca, cf)      # pairs with neither run SMT-contended
            dev_s = sum(r["wall_total_s"] for m in dm for r in dm[m].values())
            alloc_s = sum(r["process_wall_s"] for m in dm for r in dm[m].values())
            bst["resources"] = dict(timed_device_s=dev_s, allocated_device_s=alloc_s,
                                    unit="GPU-seconds" if b == "gpu_torch" else "core-seconds",
                                    allocated_definition="process start to the end of each run (process_wall_s)")
            if b in ("gpu_torch", "cpu_torch_test"):
                bst["equivalence"] = equivalence(system, b, dm)
                try:
                    bst["early_window_consistency"] = early_window_consistency(system, b, dm, curves[(system, b)])
                except M.MetricsError as e:
                    bst["early_window_consistency"] = dict(rule=EARLY_RULE, status=f"NOT APPLICABLE ({e})")
            if b == "cpu_numba_1core":
                bst["production_crosscheck"] = production_crosscheck(system, dm)
            st["backends"][b] = bst
        # effective parallelism (equal steps)
        ep = {}
        for m in PB.METHODS:
            c = data.get((system, "cpu_numba_1core"), {}).get(m, {})
            g = data.get((system, "gpu_torch"), {}).get(m, {})
            both = [s for s in sorted(set(c) & set(g)) if c[s]["n_steps"] == g[s]["n_steps"]]
            if both:
                x = np.array([c[s]["wall_total_s"] / g[s]["wall_total_s"] for s in both], float)
                med, lo, hi = M.boot_median_ci(x, N_BOOT, BOOT_SEED)
                clean = [k for k, s in enumerate(both)
                         if c[s]["smt_contended"] is False and g[s]["smt_contended"] is False]
                cm, cl, ch = M.boot_median_ci(x[clean], N_BOOT, BOOT_SEED)
                ep[m] = dict(n=len(both), median=med, ci95=[lo, hi], per_seed={int(s): float(v) for s, v in zip(both, x)},
                             n_smt_contended_pairs=len(both) - len(clean),
                             uncontended=dict(n=len(clean), median=cm, ci95=[cl, ch]),
                             definition="wall_total(cpu_numba_1core) / wall_total(gpu_torch), same seed and n_steps")
        st["effective_parallelism"] = ep if ep else "NOT TESTED (needs runs of both cpu_numba_1core and gpu_torch)"
        # cross-backend wall-to-accuracy (descriptive)
        st["wall_to_accuracy_by_backend"] = {
            f"{b}/{m}": M.quantile_inf([x["tau_wall"] for x in data.get((system, b), {}).get(m, {}).values()], 50)
            for b in BACKEND_ORDER for m in PB.METHODS if data.get((system, b), {}).get(m)}
        stats[system] = st
    per_backend_runs = data
    summary = dict(version=ANALYSIS_VERSION, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   root=os.path.abspath(root), plan="docs/mechanism/SCIENTIFIC_PLAN.md section 8",
                   config=os.path.relpath(PB.BENCH_CONFIG, ROOT), equivalence_rule=EQUIVALENCE_RULE,
                   w1_rule=W1_RULE, early_rule=EARLY_RULE, thresholds={s: PB.system_spec(s)[2] for s in PB.SYSTEMS},
                   unavailable_backends=PB.UNAVAILABLE_BACKENDS, systems=stats,
                   cost_projection=cost_projection(per_backend_runs, list(PB.SYSTEMS), seeds_per_arm),
                   definitions=dict(
                       tau="eqb_metrics.tau_persistent at the frozen mid e_F threshold over every save of the run; "
                           "t / force evaluations / wall at its first save; inf = censored",
                       ratio="FR / ABF per seed (both finite), median + 10 000-resample seed bootstrap 95 % CI; rank "
                             "part with censoring and exact sign test",
                       W1="backends.<b>.W1 (W1_RULE): FR / ABF wall-clock to accuracy on the SAME backend, primary on "
                          "the uncontended pairs; paired.tau_wall = the same ratio on all pairs (sensitivity)",
                       fr_overhead="(wall_total FR - wall_total ABF) / wall_total ABF, paired seeds, equal steps; "
                                   "fr_overhead_uncontended (primary): the pairs in which neither run had its pinned "
                                   f"CPU's SMT sibling busy > {SMT_BUSY_FLAG:g} of the timed run (unknown excluded)",
                       effective_parallelism="wall_total(1 CPU core) / wall_total(1 GPU), same seed and n_steps; "
                                             "'uncontended' (primary) restricts to pairs with both runs uncontended",
                       smt_state="per run: contended (sibling busy > 0.10 of the timed run) / uncontended / unknown "
                                 "(not pinned to one logical CPU, or telemetry missing)",
                       memory="peak_rss_mb = host VmHWM of the timed run (every backend); peak_gpu_device_mb = the "
                              "process's nvidia-smi device memory incl. CUDA context (gpu_torch, THE GPU footprint); "
                              "peak_gpu_reserved/alloc_mb = torch tensor memory only (secondary)",
                       resources="timed = sum of wall_total_s; allocated = sum of process_wall_s (process start to the "
                                 "end of the run)"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(M.json_safe(summary), fh, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "tables.md"), "w") as fh:
        fh.write(tables_md(summary))
    manifest = {}
    if figures and systems:
        setup_style()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fig_time_to_accuracy(systems, data, stats, fig_dir, manifest)
            fig_cost(systems, data, stats, fig_dir, manifest)
            fig_error_vs_wall(systems, curves, fig_dir, manifest)
            fig_equivalence(systems, data, stats, fig_dir, manifest)
        with open(os.path.join(fig_dir, "MANIFEST.json"), "w") as fh:
            json.dump(dict(version=ANALYSIS_VERSION, figures=manifest,
                           legibility_summary=LG.summarize({k: v["legibility"] for k, v in manifest.items()})),
                      fh, indent=1, sort_keys=True)
    return summary, manifest


def _f(v, nd=3):
    if v is None:
        return "--"
    if isinstance(v, str):
        return v
    v = float(v)
    if math.isnan(v):
        return "--"
    if math.isinf(v):
        return "censored"
    return f"{v:.{nd}g}"


def tables_md(S):
    L = [f"# Experiment III benchmark: {S['generated_utc']}", "", f"Root: `{S['root']}`", "",
         f"Equivalence rule: {S['equivalence_rule']}", "", f"W1 rule: {S['w1_rule']}", ""]
    for system, st in S["systems"].items():
        L += [f"## {system} (mid threshold {S['thresholds'][system]})", ""]
        for nt in st["not_tested"]:
            L.append(f"* {nt['backend']}: **{nt['label']}** ({nt['why']})")
        L.append("")
        if not st["backends"]:
            continue
        L += ["| backend | arm | n | budget frac | tau t | tau force evals | tau wall (s) | censored | Ibar_F | "
              "wall total (s) | us/step | warm-up (s) | SMT contended / unknown runs | host RSS max (MB) | "
              "GPU device max (MB) | GPU occupancy | bitwise vs prod |",
              "|" + "---|" * 17]
        for b, bs in st["backends"].items():
            for m in PB.METHODS:
                if m not in bs:
                    continue
                x = bs[m]
                L.append(f"| {b} | {m} | {bs['n_seeds'][m]} | {','.join(_f(v) for v in bs['budget_fraction'])} | "
                         f"{_f(x['tau_t_median'])} | {_f(x['tau_fe_median'])} | {_f(x['tau_wall_median'])} | "
                         f"{x['n_censored']} | {_f(x['Ibar_F_median'])} | {_f(x['wall_total_s_median'])} | "
                         f"{_f(x['us_per_step_median'])} | {_f(x['warmup_s_median'])} | {x['n_smt_contended']} / "
                         f"{x['n_smt_unknown']} | {_f(x['peak_rss_mb_max'])} | {_f(x['peak_gpu_device_mb_max'])} | "
                         f"{', '.join(x['gpu_occupancy'])} | {'; '.join(x['bitwise_vs_production'])} |")
        L.append("")
        for b, bs in st["backends"].items():
            L.append(f"* **{b}**")
            if "paired" in bs:
                p = bs["paired"]
                L.append(f"  FR/ABF ratio, all pairs (median [95 % CI], n both finite; wins-losses-ties, sign p): "
                         + "; ".join(
                             f"{k} {_f(v['ratio_median'])} [{_f(v['ratio_ci95'][0])}, {_f(v['ratio_ci95'][1])}] "
                             f"n={v['n_both_finite']} ({v['rank']['wins']}-{v['rank']['losses']}-{v['rank']['ties']}, "
                             f"p={_f(v['rank']['sign_test_p'])})" for k, v in p.items()))
                w = bs["W1"]
                if w["primary"] is None:
                    L.append(f"  **W1** (wall, uncontended pairs): {w['status']}; {w['n_pairs_excluded']} of "
                             f"{w['n_pairs']} pairs excluded (contended / unknown)")
                else:
                    v = w["primary"]
                    L.append(f"  **W1** (wall, uncontended pairs): {_f(v['ratio_median'])} [{_f(v['ratio_ci95'][0])}, "
                             f"{_f(v['ratio_ci95'][1])}] n={v['n_both_finite']} ({v['rank']['wins']}-"
                             f"{v['rank']['losses']}-{v['rank']['ties']}, p={_f(v['rank']['sign_test_p'])}); "
                             f"{w['n_pairs_excluded']} of {w['n_pairs']} pairs excluded; all-pairs sensitivity "
                             f"{_f(w['sensitivity_all_pairs']['ratio_median'])}, agrees: {w['agree']}")
                o = bs["fr_overhead"]
                ou = bs["fr_overhead_uncontended"]
                L.append(f"  FR overhead {_f(100 * o['median'])} % [{_f(100 * o['ci95'][0])}, {_f(100 * o['ci95'][1])}] "
                         f"(n={o['n']}; pairs without SMT contention: {_f(100 * ou['median'])} % "
                         f"[{_f(100 * ou['ci95'][0])}, {_f(100 * ou['ci95'][1])}], n={ou['n']}); "
                         f"G(Ibar_F) {_f(bs['G_Ibar_F']['G_median'])}")
            r = bs["resources"]
            L.append(f"  resources: timed {r['timed_device_s']:.1f} {r['unit']}, allocated {r['allocated_device_s']:.1f}")
            if "equivalence" in bs:
                eq = bs["equivalence"]
                L.append(f"  equivalence vs numba production: **{eq['status']}**")
                if eq.get("min_detectable_shift"):
                    L.append("  minimum detectable shift (80 % power, per-test alpha 0.01 / 0.05): " + "; ".join(
                        f"{k} " + (f"{_f(100 * v['rel_to_reference_median']['0.01'])} / "
                                   f"{_f(100 * v['rel_to_reference_median']['0.05'])} % of the reference median"
                                   if k != "G_Ibar_F" else
                                   f"{_f(v['abs']['0.01'])} / {_f(v['abs']['0.05'])} (absolute G)")
                        for k, v in eq["min_detectable_shift"].items()))
                if eq.get("production", {}).get("reference_excludes_seeds"):
                    L.append(f"  reference excludes production seeds {eq['production']['reference_excludes_seeds']} "
                             f"({eq['production']['why_excluded']})")
            if "early_window_consistency" in bs:
                ew = bs["early_window_consistency"]
                L.append(f"  early-window consistency vs production (EXPLORATORY, not preregistered): **{ew['status']}**"
                         + (f" over {len(ew['common_steps'])} common saves, t <= {ew['common_t'][-1]:g}; p: "
                            + ", ".join(f"{k} {_f(v)}" for k, v in ew.get("p_values", {}).items())
                            if ew.get("common_steps") else ""))
            if "production_crosscheck" in bs:
                L.append(f"  Ibar_F / tau_t vs the equal-budget production summary: {bs['production_crosscheck']['status']}")
        ep = st["effective_parallelism"]
        L.append("")
        L.append("* effective parallelism (1 CPU core / 1 GPU wall, equal steps): " + (ep if isinstance(ep, str) else
                 "; ".join(f"{m} {_f(v['median'])} [{_f(v['ci95'][0])}, {_f(v['ci95'][1])}] (n={v['n']}, "
                           f"{v['n_smt_contended_pairs']} SMT-contended; uncontended {_f(v['uncontended']['median'])}, "
                           f"n={v['uncontended']['n']})" for m, v in ep.items())))
        L.append("")
    L += ["## Full-campaign projection (per-step wall x full budget x seeds)", ""]
    for b, cp in S["cost_projection"].items():
        L.append(f"* {b}: {cp['total_h']:.2f} {cp['unit']} for {cp['seeds_per_arm']} seeds x 2 arms"
                 + ("" if cp["all_measured"] else " (some rows from the harness fallback costs, not this root)"))
        for k, v in cp["rows"].items():
            L.append(f"  * {k}: {v['us_per_step']:.1f} us/step x {v['n_steps_full']} steps + {v['overhead_s']:.0f} s "
                     f"= {v['s_per_run']:.0f} s/run [{v['source']}; host SMT {', '.join(v['host_smt_states'])}]")
        cc = cp.get("ceiling_check")
        if cc and "per_system" in cc:
            L.append(f"  * plan GPU ceiling {cc['plan_gpu_ceiling_h']:g} GPU-h (Exp III estimate "
                     f"{cc['plan_exp3_gpu_estimate_h']:g}): total within ceiling: {cc['total_within_ceiling']}")
            for s_, v in cc["per_system"].items():
                L.append(f"    * {s_}: {v['gpu_h']:.2f} GPU-h -> equivalence gate {v['equivalence_gate']}")
        elif cc:
            L.append(f"  * plan Exp III CPU estimate {cc['plan_exp3_cpu_estimate_core_h']:g} core-h: within: "
                     f"{cc['total_within_estimate']}")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=PB.EXP_ROOT)
    ap.add_argument("--out", default=None, help="default <root>/analysis")
    ap.add_argument("--fig-dir", default=None, help="default figures/mechanism/parallel_benchmark")
    ap.add_argument("--seeds-per-arm", type=int, default=PB.DEFAULT_SEEDS_PER_ARM)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.root, "analysis")
    fig_dir = a.fig_dir or os.path.join(ROOT, "figures", "mechanism", "parallel_benchmark")
    S, man = analyze(a.root, out, fig_dir, a.seeds_per_arm, figures=not a.no_figures)
    print(open(os.path.join(out, "tables.md")).read())
    bad = {k: v["legibility"].get("status") for k, v in man.items() if v["legibility"].get("status") != "pass"}
    print(f"figures: {len(man)} written to {fig_dir}; legibility failures: {bad or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
