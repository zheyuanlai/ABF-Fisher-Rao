#!/usr/bin/env python
"""Stage IV analysis of the hybrid short-T study (prereg section 6).

    python scripts/hybrid_short_T/analyze.py [--kinds prod confirm] [--select]

Reads results/hybrid_short_T/{prod,confirm}/<system>/<arm>/s<seed>[.rep<k>].{json,npz} (+ telemetry sidecars), the
benchmark summary and the frozen routing, the stored equal-budget production runs' metric cache (law check and the
hybrid projection), and writes results/hybrid_short_T/analysis/{summary.json, tables.md, curves_<kind>_<system>.npz}
and, with --select (exploration seeds only), analysis/confirm_selection.json.

Per run (hst_common.run_endpoints, on the frozen budget grid; e_F = the frozen equal-budget scorer):
  W_eps (persistent, s since the timed start), W_eps_e2e, censoring, force evaluations / walker-steps / used and
  allocated core-hours to tau, e_F(T_N), Ibar_F, completion wall.
Replicate choice (campaign.json hardware.replicate_choice): per endpoint, the first replicate whose telemetry shows no
  contention over that endpoint's window ([0, W], or the whole run when W is censored or for whole-run quantities);
  none -> replicate 0 and the run is FLAGGED for that endpoint (excluded in the sensitivity analysis).  Replicates
  share the science bitwise (checked: science_sha256), so only the timing differs.
Statistics (10 000 resamples, default_rng(20261018)):
  per N  paired FR / ABF: median of the per-seed log-ratio with a seed bootstrap (e_F(T_N), Ibar_F); for W (inf
         allowed): ratio of inf-aware medians with a paired seed bootstrap, wins / losses / ties (both censored = tie)
  best   eqb_metrics.best_allocation on W_mid (re-selection of the minimising N in every resample), R_wall = best FR
         median / best ABF median, CI = 1 + the resampled relative difference -- EXPLORATORY
  conf   confirmation seeds: R_wall^conf = ratio of inf-aware medians, unpaired bootstrap (selection held fixed);
         decision 'FR faster' / 'ABF faster' / 'unresolved'
  frontier  per method, min over N of the median e_F(w), e_F(w) = the latest save at or before w (e_flat before the
         first save; the final estimate after completion, flagged 'completed')
Sensitivity: W_e2e; contended-flagged runs excluded; throughput uncertainty (bench max / min c_ss against each
  method); hybrid projection with the validated engine's law (stored production runs truncated at u*: tau step x N x
  routed c_ss); deployment-optimised routing (projection where the method's own route differs); fixed largest N.
Law check (post hoc, prereg section 3): campaign runs vs the stored production runs truncated at u*, ratio of medians
  of e_F(T_N) and Ibar_F with a 95 % bootstrap CI, per N and method; S1-routed rungs are the null control (same
  engine, different seeds), parallel-routed rungs the counter-noise law check.
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
N_BOOT = 10_000
BOOT_SEED = 20261018
THR = H.THR_NAMES


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


def _tel(rec, kind, system):
    j = rec["job"]
    p = H.job_paths(j)["tel_npz"]
    return A.load_telemetry_npz(p)


def run_summary(recs, spec, scorer, kind):
    """One (method, N, seed): scored endpoints with the per-endpoint replicate choice."""
    j0 = recs[0]["job"]
    sci = H.load_science(j0)
    e = scorer.e_F(sci)
    sha0 = recs[0].get("science_sha256")
    reps = []
    for r in recs:
        jr = r["job"]
        res = H.load_science(jr) if jr["replicate"] > 0 else sci
        ep, _ = H.run_endpoints(res, spec, scorer, jr, rec=r, err=e)
        tel = _tel(r, kind, spec["system"])
        reps.append((r, res, ep, tel))
    out = dict(method=j0["method"], N=int(j0["N"]), seed=int(j0["seed"]), impl=j0["impl"], threads=int(j0["threads"]),
               backend=H.backend_label(j0["impl"], j0["threads"]), n_replicates=len(recs),
               science_identical=all((r.get("science_sha256") == sha0) for r in recs))
    flags = {}
    for name in THR:
        pick = None
        for (r, res, ep, tel) in reps:
            W = ep[f"W_{name}"]
            win = W if math.isfinite(W) else float(r["wall_total_s"])
            ct = A.contention(tel, float(r["t0_epoch"]), win, int(r["threads"]), pid=r.get("pid"))
            if not ct["contended"]:
                pick = (r, ep, ct)
                break
        flagged = pick is None
        if pick is None:
            r, res, ep, tel = reps[0]
            ct = A.contention(tel, float(r["t0_epoch"]), float(r["wall_total_s"]), int(r["threads"]),
                              pid=r.get("pid"))
            pick = (r, ep, ct)
        r, ep, ct = pick
        for k in ("W", "W_{}_e2e", "idx", "censored", "step", "t", "u", "force_evals", "walker_steps",
                  "used_core_h", "alloc_core_h"):
            key = f"{k}_{name}" if "{}" not in k else k.format(name)
            out[key] = ep[key]
        out[f"replicate_{name}"] = int(r["job"]["replicate"])
        flags[name] = flagged
    whole = next(((r, res, ep) for (r, res, ep, tel) in reps if not r.get("contended", True)), None)
    flags["whole"] = whole is None
    r, res, ep = whole if whole is not None else (reps[0][0], reps[0][1], reps[0][2])
    out.update(final_e_F=ep["final_e_F"], Ibar_F=ep["Ibar_F"], completion_wall_s=ep["completion_wall_s"],
               start_overhead_s=ep["start_overhead_s"], replicate_whole=int(r["job"]["replicate"]),
               e_flat=ep["e_flat"], flags=flags, wall=np.asarray(res["wall_s_at_save"], float),
               e_F=np.asarray(e, float), save_step=np.asarray(sci["save_step"], np.int64),
               n_force_evals=int(r["n_force_evals"]), used_core_h_run=float(r.get("used_core_h_run", math.nan)),
               alloc_core_h_run=float(r.get("alloc_core_h_run", math.nan)),
               max_busy_used_at_launch=float((r.get("allocation") or {}).get("max_busy_used_at_launch", math.nan)),
               contention_whole_status=(r.get("contention_whole") or {}).get("status"))
    return out


def load_all(kind, system, spec, scorer):
    raw = load_kind(kind, system)
    runs = {}
    for key, by_seed in raw.items():
        runs[key] = {s: run_summary(recs, spec, scorer, kind) for s, recs in sorted(by_seed.items())}
    return runs


# ================================================================================================ statistics
def qinf(x, q):
    return M.quantile_inf(np.asarray(x, float), q)


def med_inf(x):
    return qinf(x, 50)


def boot_med_inf(X, idx):
    """Inf-aware medians of X[idx] rows: X (n,), idx (B, n)."""
    S = np.sort(X[idx], axis=1)
    n = S.shape[1]
    p = 0.5 * (n - 1)
    lo, hi = int(math.floor(p)), int(math.ceil(p))
    a, b = S[:, lo], S[:, hi]
    with np.errstate(invalid="ignore"):
        v = a + (b - a) * (p - lo)
    return np.where(np.isinf(b) | np.isinf(a), np.where(np.isinf(b), b, a), v)


def ratio_ci(num, den):
    """2.5 / 97.5 % of num / den arrays with inf handling (x / inf -> 0, inf / x -> inf, inf / inf -> NaN dropped)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(num, float) / np.asarray(den, float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return math.nan, math.nan, 0.0
    return qinf(r, 2.5), qinf(r, 97.5), float(np.mean(r < 1.0))


def paired_logratio(a, f, rng):
    a, f = np.asarray(a, float), np.asarray(f, float)
    ok = (a > 0) & (f > 0) & np.isfinite(a) & np.isfinite(f)
    lr = np.log(f[ok] / a[ok])
    if lr.size == 0:
        return dict(ratio=math.nan, lo=math.nan, hi=math.nan, wins=0, n=0)
    idx = rng.integers(0, lr.size, (N_BOOT, lr.size))
    b = np.exp(np.median(lr[idx], 1))
    return dict(ratio=float(np.exp(np.median(lr))), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                wins=int(np.sum(lr < 0)), losses=int(np.sum(lr > 0)), n=int(lr.size))


def paired_W(a, f, rng):
    """Ratio of inf-aware medians FR / ABF with a paired seed bootstrap; wins / losses / ties."""
    a, f = np.asarray(a, float), np.asarray(f, float)
    n = a.size
    idx = rng.integers(0, n, (N_BOOT, n))
    ma, mf = boot_med_inf(a, idx), boot_med_inf(f, idx)
    lo, hi, frac = ratio_ci(mf, ma)
    pa, pf = med_inf(a), med_inf(f)
    with np.errstate(divide="ignore", invalid="ignore"):
        pt = pf / pa if not (math.isinf(pa) and math.isinf(pf)) else math.nan
    return dict(abf_median=pa, fr_median=pf, ratio=float(pt), lo=lo, hi=hi, frac_boot_fr_faster=frac,
                wins=int(np.sum(f < a)), losses=int(np.sum(f > a)), ties=int(np.sum(f == a)), n=int(n),
                n_censored_abf=int(np.sum(np.isinf(a))), n_censored_fr=int(np.sum(np.isinf(f))))


def unpaired_ratio(a, f, rng):
    a, f = np.asarray(a, float), np.asarray(f, float)
    ma = boot_med_inf(a, rng.integers(0, a.size, (N_BOOT, a.size)))
    mf = boot_med_inf(f, rng.integers(0, f.size, (N_BOOT, f.size)))
    lo, hi, frac = ratio_ci(mf, ma)
    pa, pf = med_inf(a), med_inf(f)
    with np.errstate(divide="ignore", invalid="ignore"):
        pt = pf / pa if not (math.isinf(pa) and math.isinf(pf)) else math.nan
    return dict(abf_median=pa, fr_median=pf, ratio=float(pt), lo=lo, hi=hi, frac_boot_fr_faster=frac,
                n_abf=int(a.size), n_fr=int(f.size), n_censored_abf=int(np.sum(np.isinf(a))),
                n_censored_fr=int(np.sum(np.isinf(f))))


def decision(lo, hi):
    if not (math.isfinite(lo) or math.isinf(lo)) or math.isnan(lo) or math.isnan(hi):
        return "undefined"
    if hi < 1.0:
        return "FR faster"
    if lo > 1.0:
        return "ABF faster"
    return "unresolved"


def best_alloc(runs, key, Ns, transform=None, exclude_flag=None):
    """eqb_metrics.best_allocation on runs[(m, N)][seed][key] (lower is better, inf allowed)."""
    table = {}
    for (m, N), by in runs.items():
        vals = {}
        for s, r in by.items():
            if exclude_flag and r["flags"].get(exclude_flag):
                continue
            v = r[key] if transform is None else transform(r, m, N)
            vals[s] = float(v)
        table.setdefault(N, {})[m] = vals
    Na = [N for N in Ns if "abf" in table.get(N, {})]
    Nf = [N for N in Ns if "fr" in table.get(N, {})]
    ba = M.best_allocation(table, Na, Nf, n_boot=N_BOOT, seed=BOOT_SEED)
    if ba.get("abf") and ba.get("fr"):
        a0, f0 = ba["abf"]["best_value"], ba["fr"]["best_value"]
        with np.errstate(divide="ignore", invalid="ignore"):
            R = f0 / a0 if not (math.isinf(a0) and math.isinf(f0)) else math.nan
        rel = ba["fr_minus_abf"]["rel_ci95"]
        ba["R_wall"] = dict(point=float(R), ci95=[1.0 + rel[0], 1.0 + rel[1]],
                            frac_resamples_fr_better=ba["fr_minus_abf"]["frac_resamples_fr_better"])
    return ba


# ================================================================================================ production reference
def production_reference(spec, routing, summ):
    """Stored validated production runs truncated at u*: per (method, N) {seed: dict(final, Ibar, W_proj_mid...)}."""
    out = {}
    h = spec["h"]
    for N in spec["N_ladder"]:
        steps, bmask, umask = H.save_grid(spec, N)
        bsteps = steps[bmask]
        usteps = steps[umask]
        n = H.n_steps_for(spec, N)
        route = routing["routes"][spec["system"]][str(N)]["common"]
        for m in H.methods_for(N):
            key = f"{spec['system']}|{H.config_name(m, N, route['impl'], route['threads'])}"
            c = summ[key]["c_ss_median"]
            d = {}
            for f in sorted(glob.glob(os.path.join(spec["eqb_dir"], "analysis", "run_metrics", f"N{N}",
                                                   f"s*_{m}.npz"))):
                seed = int(re.search(r"/s(\d+)_", f).group(1))
                z = np.load(f)
                st = np.rint(np.asarray(z["save_t"], float) / h).astype(np.int64)
                e = np.asarray(z["e_F"], float)
                ib = np.isin(st, bsteps)
                iu = np.isin(st, usteps)
                if ib.sum() != bsteps.size or iu.sum() != usteps.size:
                    raise H.HSTError(f"{f}: production grid does not contain the short budget grid")
                eb, sb = e[ib], st[ib]
                rec = dict(final_e_F=float(e[st == n][0]), Ibar_F=float(np.mean(e[iu])))
                for name in THR:
                    i = M.persistent_first(eb <= spec["thresholds"][name])
                    rec[f"step_{name}"] = -1 if i is None else int(sb[i])
                    rec[f"W_proj_{name}"] = math.inf if i is None else float(sb[i]) * N * c
                d[seed] = rec
            out[(m, N)] = d
    return out


# ================================================================================================ frontier / curves
def frontier(runs, w_grid, key_e="e_F"):
    """Per method: min over N of the median e_F(w); returns arrays and the argmin N and completed fraction."""
    out = {}
    for m in ("abf", "fr"):
        Ns = sorted(N for (mm, N) in runs if mm == m)
        if not Ns:
            continue
        med = np.full((len(Ns), w_grid.size), np.nan)
        done = np.zeros((len(Ns), w_grid.size))
        for i, N in enumerate(Ns):
            by = runs[(m, N)]
            E = np.array([C.e_at_wall(r[key_e], r["wall"], w_grid, r["e_flat"]) for r in by.values()])
            med[i] = np.median(E, 0)
            done[i] = np.mean([w_grid >= r["wall"][-1] for r in by.values()], 0)
        j = np.nanargmin(med, 0)
        cols = np.arange(w_grid.size)
        out[m] = dict(best=med[j, cols].tolist(), best_N=[Ns[k] for k in j], completed_frac=done[j, cols].tolist(),
                      per_N={str(N): med[i].tolist() for i, N in enumerate(Ns)})
    return out


def save_curves(path, runs, spec):
    """Per (method, N): e_F (seeds, saves), wall (seeds, saves), save_t; plus W endpoints, for the figures."""
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
        arrays[f"{tag}_completion"] = np.array([by[s]["completion_wall_s"] for s in seeds], float)
        arrays[f"{tag}_backend"] = np.array(by[seeds[0]]["backend"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


# ================================================================================================ main analysis
def analyse_system(system, kind, routing, bench, cfg=None):
    spec = H.system_spec(system, cfg)
    scorer = H.Scorer(spec)
    runs = load_all(kind, system, spec, scorer)
    if not runs:
        return None, None
    rng = np.random.default_rng(BOOT_SEED)
    summ = bench["configs"]
    Ns = spec["N_ladder"]
    per_N = []
    for N in Ns:
        route = routing["routes"][system][str(N)]["common"]
        row = dict(N=N, T_N=H.T_N(spec, N), backend=route["backend"], threads=route["threads"])
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
                W = np.array([r[f"W_{name}"] for r in by.values()])
                row[f"W_{name}_{m}"] = dict(median=med_inf(W), q25=qinf(W, 25), q75=qinf(W, 75),
                                            n_censored=int(np.sum(np.isinf(W))), n=int(W.size))
            We = np.array([r["W_mid_e2e"] for r in by.values()])
            row[f"W_mid_e2e_{m}"] = dict(median=med_inf(We))
            row[f"completion_{m}"] = float(np.median([r["completion_wall_s"] for r in by.values()]))
            row[f"start_overhead_{m}"] = float(np.nanmedian([r["start_overhead_s"] for r in by.values()]))
            row[f"flagged_{m}"] = {k: int(sum(r["flags"][k] for r in by.values())) for k in ("mid", "whole")}
            row[f"replicates_{m}"] = int(sum(r["n_replicates"] for r in by.values()))
            for q in ("force_evals_mid", "walker_steps_mid", "used_core_h_mid", "alloc_core_h_mid"):
                row[f"{q}_{m}"] = med_inf([r[q] for r in by.values()])
            key = f"{system}|{H.config_name(m, N, route['impl'], route['threads'])}"
            c = summ[key]["c_ss_median"]
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
            row["pair_W_mid_e2e"] = paired_W([A_[s]["W_mid_e2e"] for s in seeds],
                                             [F_[s]["W_mid_e2e"] for s in seeds], rng)
        per_N.append(row)
    best = {}
    for name in THR:
        best[name] = best_alloc(runs, f"W_{name}", Ns)
    sens = {}
    sens["e2e"] = best_alloc(runs, "W_mid_e2e", Ns)
    sens["exclude_flagged"] = best_alloc(runs, "W_mid", Ns, exclude_flag="mid")

    def thr_scale(direction):
        def tf(r, m, N):
            route = routing["routes"][system][str(N)]["common"]
            s = summ[f"{system}|{H.config_name(m, N, route['impl'], route['threads'])}"]
            fac = (s["c_ss_max"] / s["c_ss_median"]) if (m == "fr") == (direction == "against_fr") else (
                s["c_ss_min"] / s["c_ss_median"])
            return r["W_mid"] * fac
        return tf
    sens["throughput_against_fr"] = best_alloc(runs, "W_mid", Ns, transform=thr_scale("against_fr"))
    sens["throughput_against_abf"] = best_alloc(runs, "W_mid", Ns, transform=thr_scale("against_abf"))

    def own_route(r, m, N):
        row = routing["routes"][system][str(N)]
        c = row["c_ss_on_route"]
        f = c.get(f"{m}_own", math.nan) / c.get(f"{m}_common", math.nan)
        return r["W_mid"] * (f if math.isfinite(f) else 1.0)
    sens["deployment_optimised_projection"] = best_alloc(runs, "W_mid", Ns, transform=own_route)
    # hybrid projection with the validated engine's law
    prodref = production_reference(spec, routing, summ)
    hp = {k: {s: dict(W_mid=v["W_proj_mid"], flags={}) for s, v in d.items()} for k, d in prodref.items()}
    sens["hybrid_projection_validated_law"] = best_alloc(hp, "W_mid", Ns)
    # law check: campaign vs stored production (truncated)
    law = []
    for N in Ns:
        route = routing["routes"][system][str(N)]["common"]
        for m in H.methods_for(N):
            by, ref = runs.get((m, N), {}), prodref.get((m, N), {})
            if not by or not ref:
                continue
            r1 = {}
            for q in ("final_e_F", "Ibar_F"):
                a = np.array([v[q] for v in ref.values()])
                b = np.array([v[q] for v in by.values()])
                ia = rng.integers(0, a.size, (N_BOOT, a.size))
                ib_ = rng.integers(0, b.size, (N_BOOT, b.size))
                rb = np.median(b[ib_], 1) / np.median(a[ia], 1)
                r1[q] = dict(ratio=float(np.median(b) / np.median(a)), lo=float(np.percentile(rb, 2.5)),
                             hi=float(np.percentile(rb, 97.5)))
            law.append(dict(N=N, method=m, backend=route["backend"], n_campaign=len(by), n_reference=len(ref),
                            role="null control (same engine)" if route["impl"] == "serial" else
                            "counter-noise law check", **r1))
    # fixed largest N
    N0 = spec["N0"]
    fixed = next((r for r in per_N if r["N"] == N0), {})
    w_hi = max(r["wall"][-1] for by in runs.values() for r in by.values())
    w_grid = np.unique(np.concatenate([np.geomspace(0.02, w_hi * 1.05, 240)]))
    fr_front = frontier(runs, w_grid)
    save_curves(os.path.join(AN_DIR, f"curves_{kind}_{system}.npz"), runs, spec)
    out = dict(system=system, label=spec["label"], kind=kind, thresholds=spec["thresholds"], B=spec["B"],
               n_runs=sum(len(v) for v in runs.values()), per_N=per_N, best=best, sensitivity=sens, law_check=law,
               fixed_N0=dict(N=N0, **{k: fixed.get(k) for k in ("pair_W_mid", "pair_final_e_F", "pair_Ibar_F",
                                                                "W_mid_abf", "W_mid_fr")}),
               frontier=dict(w=w_grid.tolist(), **fr_front))
    return out, runs


def select_confirm(results, routing):
    """Per system the best ABF and the best FR configuration (argmin median W_mid on the exploration seeds)."""
    sel = dict(rule="argmin of the inf-aware median W_mid over the exploration seeds per method (prereg section 5); "
                    "a method whose best median is censored (inf) is not confirmed", systems={},
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    for s, res in results.items():
        arms = []
        notes = []
        for m in ("abf", "fr"):
            b = res["best"]["mid"].get(m)
            if not b or b["best_N"] is None or not math.isfinite(b["best_value"]):
                notes.append(f"{m}: best median W_mid censored; nothing to confirm")
                continue
            N = int(b["best_N"])
            route = routing["routes"][s][str(N)]["common"]
            arms.append(dict(method=m, N=N, impl=route["impl"], threads=int(route["threads"]),
                             backend=route["backend"], exploration_median_W_mid=b["best_value"]))
        sel["systems"][s] = dict(arms=arms, notes=notes)
    return sel


def confirm_analysis(system, routing, cfg=None):
    spec = H.system_spec(system, cfg)
    scorer = H.Scorer(spec)
    runs = load_all("confirm", system, spec, scorer)
    if not runs:
        return None
    rng = np.random.default_rng(BOOT_SEED + 1)
    abf = [(k, v) for k, v in runs.items() if k[0] == "abf"]
    fr = [(k, v) for k, v in runs.items() if k[0] == "fr"]
    out = dict(system=system, arms={f"{k[0]}{k[1]}": dict(n=len(v), W_mid_median=med_inf([r["W_mid"] for r in
                                                                                          v.values()]),
                                                        n_censored=int(sum(math.isinf(r["W_mid"]) for r in
                                                                           v.values())))
                                    for k, v in runs.items()})
    if abf and fr:
        a = [r["W_mid"] for r in abf[0][1].values()]
        f = [r["W_mid"] for r in fr[0][1].values()]
        u = unpaired_ratio(a, f, rng)
        u["decision"] = decision(u["lo"], u["hi"])
        ae = [r["W_mid_e2e"] for r in abf[0][1].values()]
        fe = [r["W_mid_e2e"] for r in fr[0][1].values()]
        ue = unpaired_ratio(ae, fe, rng)
        ue["decision"] = decision(ue["lo"], ue["hi"])
        ax = [r["W_mid"] for r in abf[0][1].values() if not r["flags"]["mid"]]
        fx = [r["W_mid"] for r in fr[0][1].values() if not r["flags"]["mid"]]
        ux = unpaired_ratio(ax, fx, rng) if ax and fx else None
        if ux:
            ux["decision"] = decision(ux["lo"], ux["hi"])
        out.update(best_abf=f"abf{abf[0][0][1]}", best_fr=f"fr{fr[0][0][1]}", R_wall_conf=u, R_wall_conf_e2e=ue,
                   R_wall_conf_unflagged=ux,
                   force_evals_mid=dict(abf=med_inf([r["force_evals_mid"] for r in abf[0][1].values()]),
                                        fr=med_inf([r["force_evals_mid"] for r in fr[0][1].values()])),
                   used_core_h_mid=dict(abf=med_inf([r["used_core_h_mid"] for r in abf[0][1].values()]),
                                        fr=med_inf([r["used_core_h_mid"] for r in fr[0][1].values()])),
                   alloc_core_h_mid=dict(abf=med_inf([r["alloc_core_h_mid"] for r in abf[0][1].values()]),
                                         fr=med_inf([r["alloc_core_h_mid"] for r in fr[0][1].values()])))
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


def tables_md(results, confirm):
    L = ["# Hybrid short-T wall-clock study: analysis tables", "",
         "Generated by `scripts/hybrid_short_T/analyze.py`. W = measured wall seconds from the engine's timed start "
         "to the first budget-grid save after which e_F stays <= the threshold (persistent; censored = never "
         "sustained within the budget). Ratios are FR / ABF; < 1 favours FR. Exploratory unless labelled "
         "confirmatory.", ""]
    for s, res in results.items():
        thr = res["thresholds"]
        L += [f"## {res['label']} ({res['kind']}, {res['n_runs']} runs)", "",
              f"Thresholds e_F strict / mid / loose = {thr['strict']} / {thr['mid']} / {thr['loose']}.", "",
              "| N | T_N | backend | e_F(T_N) ABF | ABF+FR | ratio [95 % CI] | W_mid ABF (s) [cens] | W_mid FR (s) "
              "[cens] | W ratio [95 % CI] | FR first / ABF first / ties | completion ABF / FR (s) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in res["per_N"]:
            fa = r.get("final_abf", {}).get("median")
            ff = r.get("final_fr", {}).get("median")
            pr = r.get("pair_final_e_F")
            wa = r.get("W_mid_abf", {})
            wf = r.get("W_mid_fr", {})
            pw = r.get("pair_W_mid")
            L.append(f"| {r['N']} | {fmt(r['T_N'], 4)} | {r['backend']} | {fmt(fa)} | {fmt(ff)} | "
                     + (f"{fmt(pr['ratio'], 2)} [{fmt(pr['lo'], 2)}, {fmt(pr['hi'], 2)}]" if pr else "--") + " | "
                     + f"{fmt(wa.get('median'))} [{wa.get('n_censored', '-')}/{wa.get('n', '-')}] | "
                     + (f"{fmt(wf.get('median'))} [{wf.get('n_censored')}/{wf.get('n')}]" if wf else "--") + " | "
                     + (f"{fmt(pw['ratio'], 2)} [{fmt(pw['lo'], 2)}, {fmt(pw['hi'], 2)}]" if pw else "--") + " | "
                     + (f"{pw['wins']} / {pw['losses']} / {pw['ties']}" if pw else "--") + " | "
                     + f"{fmt(r.get('completion_abf'))} / {fmt(r.get('completion_fr'))} |")
        L.append("")
        for name in THR:
            b = res["best"][name]
            if not (b.get("abf") and b.get("fr")):
                continue
            R = b.get("R_wall", {})
            L.append(f"* Best allocation, W_{name} (exploratory): ABF N = {b['abf']['best_N']} "
                     f"({fmt(b['abf']['best_value'])} s), ABF+FR N = {b['fr']['best_N']} "
                     f"({fmt(b['fr']['best_value'])} s); R_wall = {fmt(R.get('point'), 3)} "
                     f"[{fmt(R.get('ci95', [math.nan])[0], 3)}, {fmt(R.get('ci95', [0, math.nan])[1], 3)}]; "
                     f"fraction of resamples with FR better {fmt(R.get('frac_resamples_fr_better'), 2)}.")
        L.append("")
        for k, b in res["sensitivity"].items():
            R = b.get("R_wall", {})
            if not R:
                continue
            L.append(f"* Sensitivity `{k}`: ABF N = {b['abf']['best_N']} ({fmt(b['abf']['best_value'])}), FR N = "
                     f"{b['fr']['best_N']} ({fmt(b['fr']['best_value'])}); R = {fmt(R['point'], 3)} "
                     f"[{fmt(R['ci95'][0], 3)}, {fmt(R['ci95'][1], 3)}].")
        L.append("")
        L += ["Law check (campaign vs stored production runs truncated at u*; ratio of medians, 95 % CI):", "",
              "| N | method | backend | role | e_F(T_N) ratio [CI] | Ibar_F ratio [CI] |", "|---|---|---|---|---|---|"]
        for x in res["law_check"]:
            L.append(f"| {x['N']} | {x['method']} | {x['backend']} | {x['role']} | {fmt(x['final_e_F']['ratio'], 3)} "
                     f"[{fmt(x['final_e_F']['lo'], 3)}, {fmt(x['final_e_F']['hi'], 3)}] | "
                     f"{fmt(x['Ibar_F']['ratio'], 3)} [{fmt(x['Ibar_F']['lo'], 3)}, {fmt(x['Ibar_F']['hi'], 3)}] |")
        L.append("")
    if confirm:
        L += ["## Confirmation (fresh seeds, selection fixed)", ""]
        for s, c in confirm.items():
            if not c or "R_wall_conf" not in c:
                L.append(f"* {s}: {c}")
                continue
            u = c["R_wall_conf"]
            L.append(f"* {s}: best ABF {c['best_abf']} median W_mid {fmt(u['abf_median'])} s "
                     f"({u['n_censored_abf']}/{u['n_abf']} censored) vs best FR {c['best_fr']} "
                     f"{fmt(u['fr_median'])} s ({u['n_censored_fr']}/{u['n_fr']}); R_wall^conf = {fmt(u['ratio'], 3)} "
                     f"[{fmt(u['lo'], 3)}, {fmt(u['hi'], 3)}] -> **{u['decision']}** (e2e: "
                     f"{fmt(c['R_wall_conf_e2e']['ratio'], 3)} [{fmt(c['R_wall_conf_e2e']['lo'], 3)}, "
                     f"{fmt(c['R_wall_conf_e2e']['hi'], 3)}] {c['R_wall_conf_e2e']['decision']}).")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--select", action="store_true", help="write analysis/confirm_selection.json (exploration)")
    ap.add_argument("--systems", nargs="*", default=list(H.SYSTEMS))
    a = ap.parse_args(argv)
    import benchmark as BM
    routing = BM.load_routing()
    bench = C.read_json(os.path.join(BM.BENCH_DIR, "summary.json"))
    results = {}
    for s in a.systems:
        r, _ = analyse_system(s, "prod", routing, bench)
        if r is not None:
            results[s] = r
            print(f"{s}: {r['n_runs']} runs analysed")
    confirm = {}
    for s in a.systems:
        c = confirm_analysis(s, routing)
        if c is not None:
            confirm[s] = c
    os.makedirs(AN_DIR, exist_ok=True)
    out = dict(version=H.VERSION, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), results=results,
               confirm=confirm, routing_frozen_utc=routing.get("frozen_utc"))
    C.write_json(os.path.join(AN_DIR, "summary.json"), out)
    md = tables_md(results, confirm)
    with open(os.path.join(AN_DIR, "tables.md"), "w") as fh:
        fh.write(md)
    if a.select:
        sel_path = os.path.join(AN_DIR, "confirm_selection.json")
        if os.path.exists(sel_path):
            raise H.HSTError(f"{sel_path} exists: the confirmation selection is made once")
        C.write_json(sel_path, select_confirm(results, routing))
        print(f"wrote {sel_path}")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
