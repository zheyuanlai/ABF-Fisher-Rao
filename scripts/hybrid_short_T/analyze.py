#!/usr/bin/env python
"""Stage IV analysis of the hybrid short-T study (prereg section 6 with Amendment A2).

    python scripts/hybrid_short_T/analyze.py [--select] [--systems ...]

Reads results/hybrid_short_T/{prod,confirm}/<system>/<arm>/s<seed>[.rep<k>].{json,npz} (+ telemetry sidecars), the
benchmark summary and the frozen routing, the stored equal-budget production runs' metric cache (law check, null
control, hybrid projection), and writes results/hybrid_short_T/analysis/{summary.json, tables.md,
curves_<kind>_<system>.npz} and, with --select (exploration seeds only, ONCE), analysis/confirm_selection.json.

Per run (hst_common.run_endpoints; e_F = the frozen equal-budget scorer):
  W_eps   persistent on the A2.1 persistence grid (original budget grid U dense log grid, the same fractions u/u* at
          every N, relative spacing <= 2 %), seconds since the engine's timed start; W_eps_e2e = W_eps + the run's
          start overhead (process start .. timed start, minus the monitor hold); W_eps_interp (log-linear crossing,
          sensitivity); censoring (inf, never imputed); tau_u; force evaluations / walker-steps / used and allocated
          core-hours to tau; e_F(T_N); Ibar_F (k* original uniform saves); completion wall of the replicate used.
Replicate choice (A2.4): per endpoint the first replicate whose science is bitwise replicate 0's and whose
  telemetry shows no contention over that endpoint's window ([0, W], whole run if censored / for whole-run
  quantities); none -> replicate 0, FLAGGED for that endpoint.
Statistics (10 000 resamples):
  per N    paired FR / ABF: median per-seed log-ratio with a seed bootstrap (e_F(T_N), Ibar_F); W (inf allowed):
           ratio of inf-aware medians with a paired seed bootstrap; wins / losses / ties (both censored = tie)
  best     exploratory best allocation: one index draw per N shared by both methods (eqb_metrics.best_allocation's
           design), the minimising N re-selected per method in every resample; R_wall = best FR median / best ABF
           median, CI from the resampled best values (A2.3: x / inf = 0, inf / x = inf, inf / inf undefined and
           counted); W and W_e2e (co-primary, A2.2), plus sensitivities
  confirm  A2.7: R_conf = min over the confirmed FR arms / min over the confirmed ABF arms of the median W (and W_e2e),
           independent seeds per arm, the minimum re-selected within each method in every resample; decision per
           system needs both W and W_e2e CIs on the same side of 1; study-level Holm over the three systems on the
           one-sided bootstrap p (fraction of resamples with R >= 1 for 'FR faster', <= 1 for 'ABF faster')
  frontier per method, min over N of the median e_F(w), e_F(w) = the latest save at or before w (e_flat before the
           first save; the final estimate after completion, flagged 'completed')
Sensitivities: W_interp; flagged runs excluded (pooled and per backend); throughput uncertainty (the benchmark
  max / min c_ss ratio of the configuration applied to one method's W); hybrid projection with the validated
  engine's law (production_reference_seeds truncated at u*: tau step x N x routed c_ss -- the seeds on which T* was
  chosen); deployment-optimised routing (projection where a method's own route differs).
Law check (A2.8): campaign runs vs production_reference_seeds truncated at u* (e_F(T_N), Ibar_F, tau_mid u), per N
  and method; null control: production block vs the equal-budget best-allocation confirmation block (another
  validated block) at the N where it exists.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import hst_common as H  # noqa: E402
import expB_common as C  # noqa: E402
import expB_alloc as A  # noqa: E402
import eqb_metrics as M  # noqa: E402

AN_DIR = os.path.join(H.OUT_ROOT, "analysis")
SELECTION = os.path.join(AN_DIR, "confirm_selection.json")
N_BOOT = 10_000
BOOT_SEED = 20261018
THR = H.THR_NAMES
NULL_BLOCKS = {"gateway": ("confirm_best_gateway", 16), "lta300": ("confirm_best_lta_T300", 16),
               "lta150": ("confirm_best_lta_T150", 2)}


# ================================================================================================ loading
def _seed_rep(fname):
    m = re.match(r"s(\d+)(?:\.rep(\d+))?\.json$", fname)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


def load_kind(kind, system, out_root=H.OUT_ROOT):
    """{(method, N): {seed: [complete records, replicate order]}}."""
    root = os.path.join(out_root, kind, system)
    out = {}
    if not os.path.isdir(root):
        return out
    for arm in sorted(os.listdir(root)):
        d = os.path.join(root, arm)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            sr = _seed_rep(f)
            if sr is None:
                continue
            rec = C.read_json(os.path.join(d, f))
            if rec.get("status") != "complete":
                continue
            j = rec["job"]
            out.setdefault((j["method"], int(j["N"])), {}).setdefault(sr[0], []).append(rec)
    for v in out.values():
        for recs in v.values():
            recs.sort(key=lambda r: r["job"]["replicate"])
    return out


def _tel(rec):
    return A.load_telemetry_npz(H.job_paths(rec["job"])["tel_npz"])


def run_summary(recs, spec, scorer):
    """One (method, N, seed): scored endpoints with the per-endpoint replicate choice (A2.4)."""
    j0 = recs[0]["job"]
    sci = H.load_science(j0)
    e = scorer.e_F(sci)
    sha0 = recs[0].get("science_sha256")
    reps = []
    nondet = False
    for r in recs:
        if r.get("science_sha256") != sha0:
            nondet = True                    # never used for timing (A2.4)
            continue
        jr = r["job"]
        res = dict(H.load_science(jr)) if jr["replicate"] > 0 else dict(sci)
        res["wall_total_s"] = float(r["wall_total_s"])          # the replicate's own completion wall
        ep, _ = H.run_endpoints(res, spec, scorer, jr, rec=r, err=e)
        reps.append((r, res, ep, _tel(r)))
    out = dict(method=j0["method"], N=int(j0["N"]), seed=int(j0["seed"]), impl=j0["impl"], threads=int(j0["threads"]),
               backend=H.backend_label(j0["impl"], j0["threads"]), n_replicates=len(recs), nondeterministic=nondet)
    flags = {}
    for name in THR:
        pick = None
        for (r, res, ep, tel) in reps:
            W = ep[f"W_{name}"]
            win = W if math.isfinite(W) else float(r["wall_total_s"])
            ct = A.contention(tel, float(r["t0_epoch"]), win, int(r["threads"]), pid=r.get("pid"))
            if not ct["contended"]:
                pick = (r, ep)
                break
        flags[name] = pick is None
        if pick is None:                         # A2.4b: the least-contended replicate over this window, flagged
            best = None
            for (r, res, ep, tel) in reps:
                W = ep[f"W_{name}"]
                win = W if math.isfinite(W) else float(r["wall_total_s"])
                fb = A.contention(tel, float(r["t0_epoch"]), win, int(r["threads"]), pid=r.get("pid")).get(
                    "max_foreign_busy")
                fb = math.inf if fb is None else float(fb)
                if best is None or fb < best[0]:
                    best = (fb, r, ep)
            pick = (best[1], best[2])
        r, ep = pick
        for k in ("W_{}", "W_{}_e2e", "W_{}_interp", "idx_{}", "censored_{}", "step_{}", "t_{}", "u_{}",
                  "force_evals_{}", "walker_steps_{}", "used_core_h_{}", "alloc_core_h_{}", "gap_ratio_{}"):
            key = k.format(name)
            out[key] = ep[key]
        out[f"replicate_{name}"] = int(r["job"]["replicate"])
    whole = next(((r, res, ep) for (r, res, ep, tel) in reps if not r.get("contended", True)), None)
    flags["whole"] = whole is None
    if whole is None:                            # A2.4b: least whole-run foreign busy
        whole = min(((r, res, ep) for (r, res, ep, tel) in reps),
                    key=lambda x: float((x[0].get("contention_whole") or {}).get("max_foreign_busy") or math.inf))
    r, res, ep = whole
    out.update(final_e_F=ep["final_e_F"], Ibar_F=ep["Ibar_F"], completion_wall_s=ep["completion_wall_s"],
               start_overhead_s=ep["start_overhead_s"], replicate_whole=int(r["job"]["replicate"]),
               e_flat=ep["e_flat"], flags=flags, wall=np.asarray(res["wall_s_at_save"], float),
               e_F=np.asarray(e, float), save_step=np.asarray(sci["save_step"], np.int64),
               n_force_evals=int(r["n_force_evals"]), used_core_h_run=float(r.get("used_core_h_run", math.nan)),
               alloc_core_h_run=float(r.get("alloc_core_h_run", math.nan)),
               max_busy_used_at_launch=float((r.get("allocation") or {}).get("max_busy_used_at_launch", math.nan)),
               mean_freq_mhz=float((r.get("contention_whole") or {}).get("mean_freq_mhz") or math.nan))
    return out


def load_all(kind, system, spec, scorer):
    raw = load_kind(kind, system)
    return {key: {s: run_summary(recs, spec, scorer) for s, recs in sorted(by.items())} for key, by in raw.items()}


# ================================================================================================ statistics
def qinf(x, q):
    return M.quantile_inf(np.asarray(x, float), q)


def med_inf(x):
    return qinf(x, 50)


def boot_med_inf(X, idx):
    """Inf-aware medians of X[idx] rows (NaN entries = missing seeds are not allowed here)."""
    S = np.sort(X[idx], axis=1)
    n = S.shape[1]
    p = 0.5 * (n - 1)
    lo, hi = int(math.floor(p)), int(math.ceil(p))
    a, b = S[:, lo], S[:, hi]
    with np.errstate(invalid="ignore"):
        v = a + (b - a) * (p - lo)
    return np.where(np.isinf(b) | np.isinf(a), np.where(np.isinf(b), b, a), v)


def ratio_arr(num, den):
    """num / den with x / inf = 0, inf / x = inf, inf / inf = NaN (undefined)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    r = np.where(np.isinf(den) & np.isfinite(num), 0.0, r)
    r = np.where(np.isinf(num) & np.isfinite(den), np.inf, r)
    r = np.where(np.isinf(num) & np.isinf(den), np.nan, r)
    return r


def ratio_ci(num, den):
    r = ratio_arr(num, den)
    und = float(np.mean(np.isnan(r))) if r.size else math.nan
    r = r[~np.isnan(r)]
    if r.size == 0:
        return dict(lo=math.nan, hi=math.nan, frac_lt1=math.nan, frac_ge1=math.nan, frac_le1=math.nan,
                    frac_undefined=und)
    return dict(lo=qinf(r, 2.5), hi=qinf(r, 97.5), frac_lt1=float(np.mean(r < 1.0)),
                frac_ge1=float(np.mean(r >= 1.0)), frac_le1=float(np.mean(r <= 1.0)), frac_undefined=und)


def point_ratio(f, a):
    return float(ratio_arr([f], [a])[0])


def paired_logratio(a, f, rng):
    a, f = np.asarray(a, float), np.asarray(f, float)
    ok = (a > 0) & (f > 0) & np.isfinite(a) & np.isfinite(f)
    lr = np.log(f[ok] / a[ok])
    if lr.size == 0:
        return dict(ratio=math.nan, lo=math.nan, hi=math.nan, wins=0, losses=0, n=0)
    idx = rng.integers(0, lr.size, (N_BOOT, lr.size))
    b = np.exp(np.median(lr[idx], 1))
    return dict(ratio=float(np.exp(np.median(lr))), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                wins=int(np.sum(lr < 0)), losses=int(np.sum(lr > 0)), n=int(lr.size))


def paired_W(a, f, rng):
    """Ratio of inf-aware medians FR / ABF with a paired seed bootstrap; wins / losses / ties."""
    a, f = np.asarray(a, float), np.asarray(f, float)
    n = a.size
    idx = rng.integers(0, n, (N_BOOT, n))
    ci = ratio_ci(boot_med_inf(f, idx), boot_med_inf(a, idx))
    pa, pf = med_inf(a), med_inf(f)
    return dict(abf_median=pa, fr_median=pf, ratio=point_ratio(pf, pa), lo=ci["lo"], hi=ci["hi"],
                frac_boot_fr_faster=ci["frac_lt1"], frac_undefined=ci["frac_undefined"],
                wins=int(np.sum(f < a)), losses=int(np.sum(f > a)), ties=int(np.sum(f == a)), n=int(n),
                n_censored_abf=int(np.sum(np.isinf(a))), n_censored_fr=int(np.sum(np.isinf(f))))


def decision(lo, hi):
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return "undefined"
    if hi < 1.0:
        return "FR faster"
    if lo > 1.0:
        return "ABF faster"
    return "unresolved"


def joint_decision(dW, dE):
    """A2.2: a directional verdict needs W and W_e2e to agree."""
    if dW == dE:
        return dW
    if "undefined" in (dW, dE):
        return "undefined"
    return f"unresolved (W: {dW}; W_e2e: {dE})"


def best_alloc(runs, key, Ns, transform=None, exclude_flag=None, n_boot=N_BOOT, seed=BOOT_SEED):
    """Exploratory best allocation on runs[(m, N)][seed][key] (lower is better, inf allowed) -- A2.3.

    Same resampling design as eqb_metrics.best_allocation: for every N one index draw over the union of that N's
    seeds, shared by both methods; per method the minimising N is re-selected in every resample (an N whose
    resample has no usable seed for that method is not selectable there).  Returns per method the point best N /
    value, the medians, the selection frequencies, and R_wall = best FR / best ABF with the CI from the resampled
    best values (ratio convention of ratio_arr)."""
    rng = np.random.default_rng(seed)
    table = {}
    for (m, N), by in runs.items():
        if N not in Ns:
            continue
        for s, r in by.items():
            if exclude_flag and r["flags"].get(exclude_flag):
                continue
            v = r[key] if transform is None else transform(r, m, N)
            table.setdefault(N, {}).setdefault(m, {})[s] = float(v)
    med = {"abf": {}, "fr": {}}
    boot = {"abf": {}, "fr": {}}
    for N in sorted(table):
        d = table[N]
        seeds = sorted(set(d.get("abf", {})) | set(d.get("fr", {})))
        X = np.array([[d.get(m, {}).get(s, np.nan) for m in ("abf", "fr")] for s in seeds], float)
        idx = rng.integers(0, len(seeds), size=(int(n_boot), len(seeds)))
        for c, m in enumerate(("abf", "fr")):
            col = X[:, c]
            if np.all(np.isnan(col)):
                continue
            med[m][N] = med_inf(col[~np.isnan(col)])
            Xb = col[idx]
            bm = np.full(int(n_boot), np.nan)
            cnt = np.sum(~np.isnan(Xb), axis=1)
            for nn in np.unique(cnt):
                if nn == 0:
                    continue
                rows = cnt == nn
                Srt = np.sort(np.where(np.isnan(Xb[rows]), np.inf, Xb[rows]), axis=1)[:, :nn]
                p = 0.5 * (nn - 1)
                lo, hi = int(math.floor(p)), int(math.ceil(p))
                a, b = Srt[:, lo], Srt[:, hi]
                with np.errstate(invalid="ignore"):
                    v = a + (b - a) * (p - lo)
                bm[rows] = np.where(np.isinf(a) | np.isinf(b), np.where(np.isinf(b), b, a), v)
            boot[m][N] = bm
    out = {}
    for m in ("abf", "fr"):
        cand = sorted(med[m])
        if not cand:
            out[m] = None
            continue
        vals = np.array([med[m][N] for N in cand])
        k = int(np.argmin(vals))
        Bm = np.stack([boot[m][N] for N in cand], axis=1)
        nodata = np.all(np.isnan(Bm), axis=1)
        Bf = np.where(np.isnan(Bm), np.inf, Bm)
        bmin = Bf.min(axis=1)
        arg = Bf.argmin(axis=1)
        freq = {int(N): float(np.mean((arg == i) & np.isfinite(bmin))) for i, N in enumerate(cand)}
        out[m] = dict(best_N=(int(cand[k]) if math.isfinite(vals[k]) else None), best_value=float(vals[k]),
                      medians={int(N): float(med[m][N]) for N in cand}, boot_best_N_freq=freq,
                      boot_frac_censored=float(np.mean(np.isinf(bmin) & ~nodata)),
                      boot_best_value_ci95=[qinf(bmin[~nodata], 2.5), qinf(bmin[~nodata], 97.5)],
                      _boot=np.where(nodata, np.nan, bmin))
    if out.get("abf") and out.get("fr"):
        da, df = out["abf"]["_boot"], out["fr"]["_boot"]
        ok = ~(np.isnan(da) | np.isnan(df))
        ci = ratio_ci(df[ok], da[ok])
        out["R_wall"] = dict(point=point_ratio(out["fr"]["best_value"], out["abf"]["best_value"]),
                             ci95=[ci["lo"], ci["hi"]], frac_resamples_fr_better=ci["frac_lt1"],
                             frac_resamples_undefined=ci["frac_undefined"], p_fr_faster=ci["frac_ge1"],
                             p_abf_faster=ci["frac_le1"])
    for m in ("abf", "fr"):
        if out.get(m):
            out[m].pop("_boot")
    return out


def holm(pvals, alpha=0.05):
    """{key: (p, adjusted p, reject)} by Holm's step-down."""
    items = sorted(((float(p), k) for k, p in pvals.items() if p is not None and not math.isnan(p)))
    m = len(items)
    out, prev = {}, 0.0
    for i, (p, k) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        prev = adj
        out[k] = dict(p=p, p_holm=adj, reject=bool(adj < alpha))
    return out


# ================================================================================================ production reference
def _eqb_cache(dirname, N, m):
    return sorted(glob.glob(os.path.join(H.ROOT, "results", "equal_budget_v2", dirname, "analysis", "run_metrics",
                                         f"N{N}", f"s*_{m}.npz")))


def truncated_stats(f, spec, N, c_ss=None):
    """Statistics of one stored validated run truncated at u*: e_F(T_N), Ibar_F, tau (u, step) on the original
    budget grid (the stored runs have no dense saves), projected W = step x N x c_ss."""
    h = spec["h"]
    steps, bmask, umask = H.save_grid(spec, N)
    st_orig = H.orig_budget_steps(spec, N)
    usteps = steps[umask]
    n = H.n_steps_for(spec, N)
    z = np.load(f)
    st = np.rint(np.asarray(z["save_t"], float) / h).astype(np.int64)
    e = np.asarray(z["e_F"], float)
    ib = np.isin(st, st_orig)
    iu = np.isin(st, usteps)
    if ib.sum() != st_orig.size or iu.sum() != usteps.size:
        raise H.HSTError(f"{f}: production grid does not contain the short budget grid")
    eb, sb = e[ib], st[ib]
    rec = dict(final_e_F=float(e[st == n][0]), Ibar_F=float(np.mean(e[iu])))
    for name in THR:
        i = M.persistent_first(eb <= spec["thresholds"][name])
        rec[f"step_{name}"] = -1 if i is None else int(sb[i])
        rec[f"u_{name}"] = math.inf if i is None else float(sb[i]) / n
        if c_ss is not None:
            rec[f"W_proj_{name}"] = math.inf if i is None else float(sb[i]) * N * c_ss
    return rec


def production_reference(spec, routing, summ):
    """production_reference_seeds truncated at u*: {(method, N): {seed: stats}} (A2.8)."""
    out = {}
    keep = set(spec["production_reference_seeds"])
    for N in spec["N_ladder"]:
        route = routing["routes"][spec["system"]][str(N)]["common"]
        for m in H.methods_for(N):
            c = c_ss_of(summ, spec["system"], m, N, route)
            d = {}
            for f in _eqb_cache(os.path.basename(spec["eqb_dir"]), N, m):
                seed = int(re.search(r"/s(\d+)_", f).group(1))
                if seed in keep:
                    d[seed] = truncated_stats(f, spec, N, c)
            out[(m, N)] = d
    return out


def c_ss_of(summ, system, method, N, route):
    key = f"{system}|{H.config_name(method, N, route['impl'], route['threads'])}"
    if route["impl"] == "parallel" and route["threads"] >= 4 and N >= 16:
        kl = key + "_L"
        if summ.get(kl, {}).get("c_ss_median") is not None:
            key = kl
    return summ[key]["c_ss_median"]


def _ratio_of_medians_boot(a, b, rng):
    """b / a medians (inf-aware) with independent bootstraps."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ma = boot_med_inf(a, rng.integers(0, a.size, (N_BOOT, a.size)))
    mb = boot_med_inf(b, rng.integers(0, b.size, (N_BOOT, b.size)))
    ci = ratio_ci(mb, ma)
    return dict(ratio=point_ratio(med_inf(b), med_inf(a)), lo=ci["lo"], hi=ci["hi"])


def law_and_null(spec, routing, summ, runs, prodref, rng):
    law = []
    for N in spec["N_ladder"]:
        route = routing["routes"][spec["system"]][str(N)]["common"]
        for m in H.methods_for(N):
            by, ref = runs.get((m, N), {}), prodref.get((m, N), {})
            if not by or not ref:
                continue
            row = dict(N=N, method=m, backend=route["backend"], n_campaign=len(by), n_reference=len(ref),
                       role=("same engine (S1): a seed-block comparison" if route["impl"] == "serial"
                             else "counter-noise law check"))
            for q in ("final_e_F", "Ibar_F", "u_mid"):
                row[q] = _ratio_of_medians_boot([v[q] for v in ref.values()], [v[q] for v in by.values()], rng)
            law.append(row)
    null = []
    dname, Nn = NULL_BLOCKS[spec["system"]]
    keep = set(spec["production_reference_seeds"])
    for m in H.methods_for(Nn):
        a = [truncated_stats(f, spec, Nn) for f in _eqb_cache(os.path.basename(spec["eqb_dir"]), Nn, m)
             if int(re.search(r"/s(\d+)_", f).group(1)) in keep]
        b = [truncated_stats(f, spec, Nn) for f in _eqb_cache(dname, Nn, m)]
        if not a or not b:
            continue
        row = dict(N=Nn, method=m, n_a=len(a), n_b=len(b), blocks=f"production vs {dname}")
        for q in ("final_e_F", "Ibar_F", "u_mid"):
            row[q] = _ratio_of_medians_boot([v[q] for v in a], [v[q] for v in b], rng)
        null.append(row)
    return law, null


# ================================================================================================ frontier / curves
def frontier(runs, w_grid):
    out = {}
    for m in ("abf", "fr"):
        Ns = sorted(N for (mm, N) in runs if mm == m)
        if not Ns:
            continue
        med = np.full((len(Ns), w_grid.size), np.nan)
        done = np.zeros((len(Ns), w_grid.size))
        for i, N in enumerate(Ns):
            by = runs[(m, N)]
            E = np.array([C.e_at_wall(r["e_F"], r["wall"], w_grid, r["e_flat"]) for r in by.values()])
            med[i] = np.median(E, 0)
            done[i] = np.mean([w_grid >= r["wall"][-1] for r in by.values()], 0)
        j = np.nanargmin(med, 0)
        cols = np.arange(w_grid.size)
        out[m] = dict(best=med[j, cols].tolist(), best_N=[Ns[k] for k in j], completed_frac=done[j, cols].tolist(),
                      per_N={str(N): med[i].tolist() for i, N in enumerate(Ns)})
    return out


def save_curves(path, runs, spec):
    arrays = {}
    for (m, N), by in runs.items():
        seeds = sorted(by)
        tag = f"{m}{N}"
        arrays[f"{tag}_seeds"] = np.array(seeds, np.int64)
        arrays[f"{tag}_e"] = np.array([by[s]["e_F"] for s in seeds])
        arrays[f"{tag}_wall"] = np.array([by[s]["wall"] for s in seeds])
        arrays[f"{tag}_t"] = by[seeds[0]]["save_step"] * spec["h"]
        for name in THR:
            arrays[f"{tag}_W_{name}"] = np.array([by[s][f"W_{name}"] for s in seeds], float)
            arrays[f"{tag}_W_{name}_e2e"] = np.array([by[s][f"W_{name}_e2e"] for s in seeds], float)
        arrays[f"{tag}_completion"] = np.array([by[s]["completion_wall_s"] for s in seeds], float)
        arrays[f"{tag}_backend"] = np.array(by[seeds[0]]["backend"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


# ================================================================================================ main analysis
def analyse_system(system, routing, bench, cfg=None):
    spec = H.system_spec(system, cfg)
    scorer = H.Scorer(spec)
    runs = load_all("prod", system, spec, scorer)
    if not runs:
        return None, None
    rng = np.random.default_rng(BOOT_SEED)
    summ = bench["configs"]
    Ns = spec["N_ladder"]
    per_N = []
    for N in Ns:
        route = routing["routes"][system][str(N)]["common"]
        row = dict(N=N, T_N=H.T_N(spec, N), backend=route["backend"], threads=route["threads"], impl=route["impl"])
        for m in H.methods_for(N):
            by = runs.get((m, N), {})
            row[f"n_{m}"] = len(by)
            if not by:
                continue
            fe = np.array([r["final_e_F"] for r in by.values()])
            ib = np.array([r["Ibar_F"] for r in by.values()])
            row[f"final_{m}"] = dict(median=float(np.median(fe)), q25=float(np.percentile(fe, 25)),
                                     q75=float(np.percentile(fe, 75)))
            row[f"Ibar_{m}"] = dict(median=float(np.median(ib)), q25=float(np.percentile(ib, 25)),
                                    q75=float(np.percentile(ib, 75)))
            for name in THR:
                for suf in ("", "_e2e", "_interp"):
                    W = np.array([r[f"W_{name}{suf}"] for r in by.values()])
                    row[f"W_{name}{suf}_{m}"] = dict(median=med_inf(W), q25=qinf(W, 25), q75=qinf(W, 75),
                                                     n_censored=int(np.sum(np.isinf(W))), n=int(W.size))
                row[f"gap_ratio_{name}_{m}"] = med_inf([r[f"gap_ratio_{name}"] for r in by.values()
                                                        if math.isfinite(r[f"gap_ratio_{name}"])] or [math.nan])
            row[f"completion_{m}"] = float(np.median([r["completion_wall_s"] for r in by.values()]))
            row[f"start_overhead_{m}"] = float(np.nanmedian([r["start_overhead_s"] for r in by.values()]))
            row[f"flagged_{m}"] = {k: int(sum(r["flags"][k] for r in by.values())) for k in ("mid", "whole")}
            row[f"replicates_{m}"] = int(sum(r["n_replicates"] for r in by.values()))
            row[f"nondeterministic_{m}"] = int(sum(r["nondeterministic"] for r in by.values()))
            row[f"mean_freq_mhz_{m}"] = float(np.nanmedian([r["mean_freq_mhz"] for r in by.values()]))
            for q in ("force_evals_mid", "walker_steps_mid", "used_core_h_mid", "alloc_core_h_mid"):
                row[f"{q}_{m}"] = med_inf([r[q] for r in by.values()])
            c = c_ss_of(summ, system, m, N, route)
            row[f"timing_model_{m}"] = float(np.median([r["completion_wall_s"] / (spec["B"] * c)
                                                         for r in by.values()]))
        if N >= 2 and runs.get(("abf", N)) and runs.get(("fr", N)):
            A_, F_ = runs[("abf", N)], runs[("fr", N)]
            seeds = sorted(set(A_) & set(F_))
            for q in ("final_e_F", "Ibar_F"):
                row[f"pair_{q}"] = paired_logratio([A_[s][q] for s in seeds], [F_[s][q] for s in seeds], rng)
            for name in THR:
                row[f"pair_W_{name}"] = paired_W([A_[s][f"W_{name}"] for s in seeds],
                                                 [F_[s][f"W_{name}"] for s in seeds], rng)
                row[f"pair_W_{name}_e2e"] = paired_W([A_[s][f"W_{name}_e2e"] for s in seeds],
                                                     [F_[s][f"W_{name}_e2e"] for s in seeds], rng)
        per_N.append(row)
    best = {}
    for name in THR:
        best[name] = best_alloc(runs, f"W_{name}", Ns)
        best[f"{name}_e2e"] = best_alloc(runs, f"W_{name}_e2e", Ns)
    sens = {"interp": best_alloc(runs, "W_mid_interp", Ns),
            "exclude_flagged": best_alloc(runs, "W_mid", Ns, exclude_flag="mid")}

    def thr_scale(against):
        def tf(r, m, N):
            route = routing["routes"][system][str(N)]["common"]
            key = f"{system}|{H.config_name(m, N, route['impl'], route['threads'])}"
            if route["impl"] == "parallel" and route["threads"] >= 4 and N >= 16 and summ.get(key + "_L", {}).get(
                    "c_ss_median") is not None:
                key += "_L"
            s = summ[key]
            return r["W_mid"] * (s["c_ss_max"] / s["c_ss_min"]) if m == against else r["W_mid"]
        return tf
    sens["throughput_against_fr"] = best_alloc(runs, "W_mid", Ns, transform=thr_scale("fr"))
    sens["throughput_against_abf"] = best_alloc(runs, "W_mid", Ns, transform=thr_scale("abf"))

    def own_route(r, m, N):
        c = routing["routes"][system][str(N)]["c_ss_on_route"]
        f = c.get(f"{m}_own", math.nan) / c.get(f"{m}_common", math.nan)
        return r["W_mid"] * (f if math.isfinite(f) else 1.0)
    sens["deployment_optimised_projection"] = best_alloc(runs, "W_mid", Ns, transform=own_route)
    prodref = production_reference(spec, routing, summ)
    hp = {k: {s: dict(W_mid=v["W_proj_mid"], flags={}) for s, v in d.items()} for k, d in prodref.items()}
    sens["hybrid_projection_validated_law"] = best_alloc(hp, "W_mid", Ns)
    flag_rates = {}
    for (m, N), by in runs.items():
        b = next(iter(by.values()))["backend"]
        fr_ = flag_rates.setdefault(b, dict(n=0, flagged_mid=0, flagged_whole=0, replicates=0))
        fr_["n"] += len(by)
        fr_["flagged_mid"] += sum(r["flags"]["mid"] for r in by.values())
        fr_["flagged_whole"] += sum(r["flags"]["whole"] for r in by.values())
        fr_["replicates"] += sum(r["n_replicates"] for r in by.values())
    by_backend = {}
    for b in flag_rates:
        sub = {k: v for k, v in runs.items() if next(iter(v.values()))["backend"] == b}
        if any(k[0] == "abf" for k in sub) and any(k[0] == "fr" for k in sub):
            by_backend[b] = best_alloc(sub, "W_mid", Ns, exclude_flag="mid")
    sens["exclude_flagged_within_backend"] = by_backend
    law, null = law_and_null(spec, routing, summ, runs, prodref, rng)
    # censored configurations that finish before the winner of the other method reaches accuracy (A2.10)
    cens_notes = []
    for name in ("mid",):
        b = best[name]
        for m, other in (("abf", "fr"), ("fr", "abf")):
            if not b.get(other) or b[other]["best_N"] is None:
                continue
            Wo = b[other]["best_value"]
            for (mm, N), by in runs.items():
                if mm != m:
                    continue
                W = np.array([r[f"W_{name}"] for r in by.values()])
                comp = float(np.median([r["completion_wall_s"] for r in by.values()]))
                if math.isinf(med_inf(W)) and comp < Wo:
                    cens_notes.append(dict(threshold=name, method=m, N=N, median_completion_s=comp,
                                           n_censored=int(np.sum(np.isinf(W))), n=int(W.size),
                                           best_other=f"{other}{b[other]['best_N']}", best_other_W=Wo))
    N0 = spec["N0"]
    fixed = next((r for r in per_N if r["N"] == N0), {})
    w_hi = max(r["wall"][-1] for by in runs.values() for r in by.values())
    w_grid = np.geomspace(0.02, w_hi * 1.05, 240)
    save_curves(os.path.join(AN_DIR, f"curves_prod_{system}.npz"), runs, spec)
    return dict(system=system, label=spec["label"], kind="prod", thresholds=spec["thresholds"], B=spec["B"],
                n_runs=sum(len(v) for v in runs.values()), per_N=per_N, best=best, sensitivity=sens,
                flag_rates=flag_rates, law_check=law, null_control=null, censored_notes=cens_notes,
                fixed_N0=dict(N=N0, **{k: fixed.get(k) for k in ("pair_W_mid", "pair_W_mid_e2e", "pair_W_loose",
                                                                 "pair_final_e_F", "pair_Ibar_F", "W_mid_abf",
                                                                 "W_mid_fr")}),
                frontier=dict(w=w_grid.tolist(), **frontier(runs, w_grid))), runs


def select_confirm(results, routing, k=3):
    """A2.7: per system and method the top-k configurations by exploration median W_mid (finite only; ties ->
    fewer threads, then smaller N)."""
    sel = dict(rule="A2.7: top-k per method by exploration median W_mid (finite); ties -> fewer threads, smaller N",
               k=k, systems={}, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    for s, res in results.items():
        arms, notes = [], []
        for m in ("abf", "fr"):
            rows = []
            for r in res["per_N"]:
                w = r.get(f"W_mid_{m}")
                if w and math.isfinite(w["median"]):
                    rows.append((w["median"], r["threads"], r["N"], r))
            rows.sort(key=lambda x: (x[0], x[1], x[2]))
            if not rows:
                notes.append(f"{m}: every configuration's median W_mid is censored; nothing to confirm")
            for rank, (w, t, N, r) in enumerate(rows[:k]):
                route = routing["routes"][s][str(N)]["common"]
                arms.append(dict(method=m, N=int(N), impl=route["impl"], threads=int(route["threads"]),
                                 backend=route["backend"], rank=rank + 1, exploration_median_W_mid=w))
        sel["systems"][s] = dict(arms=arms, notes=notes)
    return sel


def confirm_analysis(system, routing, selection, cfg=None):
    spec = H.system_spec(system, cfg)
    scorer = H.Scorer(spec)
    runs = load_all("confirm", system, spec, scorer)
    if not runs:
        return None
    sel = (selection.get("systems") or {}).get(system, {})
    want = {(a["method"], int(a["N"])) for a in sel.get("arms", [])}
    have = set(runs)
    if have - want:
        raise H.HSTError(f"{system}: confirmation records of arms not in confirm_selection.json: {have - want}")
    rng = np.random.default_rng(BOOT_SEED + 1)
    out = dict(system=system, n_arms_selected=len(want), missing_arms=sorted(f"{m}{N}" for m, N in want - have),
               arms={f"{m}{N}": dict(n=len(v), W_mid_median=med_inf([r["W_mid"] for r in v.values()]),
                                     W_mid_e2e_median=med_inf([r["W_mid_e2e"] for r in v.values()]),
                                     n_censored=int(sum(math.isinf(r["W_mid"]) for r in v.values())),
                                     n_flagged_mid=int(sum(r["flags"]["mid"] for r in v.values())),
                                     force_evals_mid=med_inf([r["force_evals_mid"] for r in v.values()]),
                                     used_core_h_mid=med_inf([r["used_core_h_mid"] for r in v.values()]),
                                     alloc_core_h_mid=med_inf([r["alloc_core_h_mid"] for r in v.values()]),
                                     backend=next(iter(v.values()))["backend"])
                     for (m, N), v in sorted(runs.items())})

    def stat(key, exclude_flag=False):
        per = {}
        for m in ("abf", "fr"):
            arms = [(N, v) for (mm, N), v in runs.items() if mm == m]
            if not arms:
                return None
            pts, boots = [], []
            for N, v in arms:
                x = np.array([r[key] for r in v.values() if not (exclude_flag and r["flags"]["mid"])], float)
                if x.size == 0:
                    continue
                pts.append(med_inf(x))
                boots.append(boot_med_inf(x, rng.integers(0, x.size, (N_BOOT, x.size))))
            if not pts:
                return None
            per[m] = (min(pts), np.min(np.stack(boots), axis=0), [N for N, _ in arms][int(np.argmin(pts))])
        ci = ratio_ci(per["fr"][1], per["abf"][1])
        return dict(ratio=point_ratio(per["fr"][0], per["abf"][0]), lo=ci["lo"], hi=ci["hi"],
                    p_fr_faster=ci["frac_ge1"], p_abf_faster=ci["frac_le1"], frac_undefined=ci["frac_undefined"],
                    best_abf=f"abf{per['abf'][2]}", best_fr=f"fr{per['fr'][2]}", abf_value=per["abf"][0],
                    fr_value=per["fr"][0], decision=decision(ci["lo"], ci["hi"]))
    W = stat("W_mid")
    E = stat("W_mid_e2e")
    if W and E:
        out.update(R_conf=W, R_conf_e2e=E, decision=joint_decision(W["decision"], E["decision"]),
                   R_conf_interp=stat("W_mid_interp"), R_conf_unflagged=stat("W_mid", exclude_flag=True))
    save_curves(os.path.join(AN_DIR, f"curves_confirm_{system}.npz"), runs, spec)
    return out


# ================================================================================================ tables
def fmt(x, p=3):
    if x is None:
        return "--"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isinf(x):
        return "censored" if x > 0 else "-inf"
    if math.isnan(x):
        return "n/a"
    return f"{x:.{p}g}"


def _ci(d, key="ci95"):
    if not d:
        return "--"
    if key in d:
        lo, hi = d[key]
        return f"{fmt(d['point'], 3)} [{fmt(lo, 3)}, {fmt(hi, 3)}]"
    return f"{fmt(d['ratio'], 3)} [{fmt(d['lo'], 3)}, {fmt(d['hi'], 3)}]"


def tables_md(results, confirm, holm_res=None):
    L = ["# Hybrid short-T wall-clock study: analysis tables", "",
         "Generated by `scripts/hybrid_short_T/analyze.py`. W = measured wall seconds from the engine's timed start "
         "to the first save of the persistence grid (original budget grid U dense log grid, Amendment A2.1) after "
         "which e_F stays <= the threshold; censored = never sustained within the budget. W_e2e adds the run's "
         "start-up (process start, imports, JIT, engine warm-up). Ratios are FR / ABF; < 1 favours FR. "
         "Exploratory unless labelled confirmatory.", ""]
    for s, res in results.items():
        thr = res["thresholds"]
        L += [f"## {res['label']} ({res['n_runs']} runs)", "",
              f"Thresholds e_F strict / mid / loose = {thr['strict']} / {thr['mid']} / {thr['loose']}.", "",
              "| N | T_N | backend | e_F(T_N) ABF | ABF+FR | ratio [95 % CI] | W_mid ABF (s) [cens] | W_mid FR (s) "
              "[cens] | W ratio [95 % CI] | W_e2e ratio | FR first / ABF first / ties | completion ABF / FR (s) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in res["per_N"]:
            fa = r.get("final_abf", {}).get("median")
            ff = r.get("final_fr", {}).get("median")
            pr = r.get("pair_final_e_F")
            wa, wf = r.get("W_mid_abf", {}), r.get("W_mid_fr", {})
            pw, pe = r.get("pair_W_mid"), r.get("pair_W_mid_e2e")
            L.append(f"| {r['N']} | {fmt(r['T_N'], 4)} | {r['backend']} | {fmt(fa)} | {fmt(ff)} | "
                     + (_ci(pr) if pr else "--") + " | "
                     + f"{fmt(wa.get('median'))} [{wa.get('n_censored', '-')}/{wa.get('n', '-')}] | "
                     + (f"{fmt(wf.get('median'))} [{wf.get('n_censored')}/{wf.get('n')}]" if wf else "--") + " | "
                     + (_ci(pw) if pw else "--") + " | " + (_ci(pe) if pe else "--") + " | "
                     + (f"{pw['wins']} / {pw['losses']} / {pw['ties']}" if pw else "--") + " | "
                     + f"{fmt(r.get('completion_abf'))} / {fmt(r.get('completion_fr'))} |")
        L.append("")
        for name in ("mid", "mid_e2e", "loose", "loose_e2e", "strict", "strict_e2e"):
            b = res["best"][name]
            if not (b.get("abf") and b.get("fr")):
                continue
            L.append(f"* Best allocation, W_{name} (exploratory): ABF N = {b['abf']['best_N']} "
                     f"({fmt(b['abf']['best_value'])} s), ABF+FR N = {b['fr']['best_N']} "
                     f"({fmt(b['fr']['best_value'])} s); R_wall = {_ci(b['R_wall'])}; fraction of resamples with FR "
                     f"better {fmt(b['R_wall']['frac_resamples_fr_better'], 2)} (undefined "
                     f"{fmt(b['R_wall']['frac_resamples_undefined'], 2)}).")
        L.append("")
        for k, b in res["sensitivity"].items():
            if k == "exclude_flagged_within_backend":
                for bk, bb in b.items():
                    if bb.get("R_wall"):
                        L.append(f"* Sensitivity flagged excluded, within backend {bk}: ABF N {bb['abf']['best_N']}, "
                                 f"FR N {bb['fr']['best_N']}; R = {_ci(bb['R_wall'])}.")
                continue
            if not b.get("R_wall"):
                continue
            L.append(f"* Sensitivity `{k}`: ABF N = {b['abf']['best_N']} ({fmt(b['abf']['best_value'])}), FR N = "
                     f"{b['fr']['best_N']} ({fmt(b['fr']['best_value'])}); R = {_ci(b['R_wall'])}.")
        L.append("")
        if res["censored_notes"]:
            L.append("Censored configurations that finish their budget before the other method's best reaches the "
                     "mid threshold (A2.10):")
            for c in res["censored_notes"]:
                L.append(f"* {c['method']}{c['N']}: {c['n_censored']}/{c['n']} censored, median completion "
                         f"{fmt(c['median_completion_s'])} s < {c['best_other']} W_mid {fmt(c['best_other_W'])} s")
            L.append("")
        L.append("Flag rates by backend: " + "; ".join(
            f"{b}: {v['flagged_mid']}/{v['n']} flagged (mid), {v['replicates']} replicates" for b, v in
            sorted(res["flag_rates"].items())))
        L += ["", "Law check (campaign vs production_reference_seeds truncated at u*; ratio of medians, 95 % CI):", "",
              "| N | method | backend | role | e_F(T_N) | Ibar_F | tau_mid u |", "|---|---|---|---|---|---|---|"]
        for x in res["law_check"]:
            L.append(f"| {x['N']} | {x['method']} | {x['backend']} | {x['role']} | {_ci(x['final_e_F'])} | "
                     f"{_ci(x['Ibar_F'])} | {_ci(x['u_mid'])} |")
        L += ["", "Null control (two validated-engine blocks, truncated at u*):", ""]
        for x in res["null_control"]:
            L.append(f"* N {x['N']} {x['method']} ({x['blocks']}, {x['n_a']} vs {x['n_b']}): e_F(T_N) "
                     f"{_ci(x['final_e_F'])}, Ibar_F {_ci(x['Ibar_F'])}, tau_mid u {_ci(x['u_mid'])}")
        L.append("")
    if confirm:
        L += ["## Confirmation (fresh seeds, selection fixed in confirm_selection.json)", ""]
        for s, c in confirm.items():
            if not c or "R_conf" not in c:
                L.append(f"* {s}: {c}")
                continue
            u, e = c["R_conf"], c["R_conf_e2e"]
            L.append(f"* {s}: R_conf (W) = {_ci(u)} [{u['best_fr']} vs {u['best_abf']}], R_conf (W_e2e) = {_ci(e)}; "
                     f"**{c['decision']}**; one-sided p (FR faster) W {fmt(u['p_fr_faster'], 3)}, W_e2e "
                     f"{fmt(e['p_fr_faster'], 3)}")
        if holm_res:
            L.append("")
            L.append(f"Holm across systems (A2.7): {json.dumps(C.json_safe(holm_res))}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--select", action="store_true", help="write analysis/confirm_selection.json (exploration)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--systems", nargs="*", default=list(H.SYSTEMS))
    a = ap.parse_args(argv)
    import benchmark as BM
    routing = BM.load_routing()
    bench = C.read_json(os.path.join(BM.BENCH_DIR, "summary.json"))
    results = {}
    for s in a.systems:
        r, _ = analyse_system(s, routing, bench)
        if r is not None:
            results[s] = r
            print(f"{s}: {r['n_runs']} runs analysed")
    confirm, holm_res = {}, None
    if os.path.exists(SELECTION):
        selection = C.read_json(SELECTION)
        for s in a.systems:
            c = confirm_analysis(s, routing, selection)
            if c is not None:
                confirm[s] = c
        if confirm:
            pw = {s: max(c["R_conf"]["p_fr_faster"], c["R_conf_e2e"]["p_fr_faster"]) for s, c in confirm.items()
                  if "R_conf" in c}
            pa = {s: max(c["R_conf"]["p_abf_faster"], c["R_conf_e2e"]["p_abf_faster"]) for s, c in confirm.items()
                  if "R_conf" in c}
            holm_res = dict(fr_faster=holm(pw), abf_faster=holm(pa),
                            note="one-sided bootstrap p = fraction of resamples on the wrong side of 1, the larger of "
                                 "W and W_e2e (both must hold)")
    os.makedirs(AN_DIR, exist_ok=True)
    out = dict(version=H.VERSION, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), results=results,
               confirm=confirm, holm=holm_res, routing_frozen_utc=routing.get("frozen_utc"))
    C.write_json(os.path.join(AN_DIR, "summary.json"), out)
    md = tables_md(results, confirm, holm_res)
    with open(os.path.join(AN_DIR, "tables.md"), "w") as fh:
        fh.write(md)
    if a.select:
        if os.path.exists(SELECTION):
            raise H.HSTError(f"{SELECTION} exists: the confirmation selection is made once")
        C.write_json(SELECTION, select_confirm(results, routing, a.k))
        print(f"wrote {SELECTION}")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
