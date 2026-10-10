#!/usr/bin/env python
"""Appendix tables of the equal-budget replica ladders -> docs/equal_budget/APPENDIX_TABLES.md.

    python scripts/equal_budget/report_tables.py [--results-root PATH] [--out PATH] [--n-boot 10000]

Reads ONLY
  <results-root>/{gateway,lta_T300,lta_T150}/analysis/summary.json   (analyze_ladder.py, schema eqb_ladder_summary/2)
  <results-root>/<dir>/analysis/run_metrics/N<N>/s<seed>_<abf|fr>.npz (its per-run curve cache; used ONLY for the
                                                                       per-RUN max transient of section 4)
Nothing is re-scored and no result file is opened.  Every cached curve that is used is first checked against the
summary (Ibar and final value of the same metric recomputed from the cached curve must equal the summary's
per-seed value), so a stale cache stops the script instead of being reported.

Per system:
  1. absolute values per N and arm (median [IQR] over seeds) of Ibar_F, final e_F, Ibar_F', final e_F' (gateway also
     the floor-free *_Fp_stat companions), Ibar_TV_half, final TV_half, final TV_inst vs its exact finite-N floor;
  2. MEAN-based paired contrasts G_X(N) = (mean_s X^FR - mean_s X^ABF) / mean_s X^ABF over the seeds with both arms,
     95 % CI from a paired seed bootstrap (resample seeds with replacement, recompute the ratio of means), next to
     the production median-of-per-seed-ratios G; and a MEAN-based best allocation for Ibar_F (min over N of the
     seed means; the bootstrap re-selects N in every resample, ties share the credit) next to the production
     median-based one;
  3. persistent time-to-accuracy tau per N and arm in physical time t and budget fraction u (median [IQR],
     censored fraction) with the paired FR-vs-ABF wins/losses/ties and exact sign-test p;
  4. maximum transient improvement (ABF - FR)/ABF: of the seed-median curves (summary max_transient) and per RUN
     (per seed, max over the 200 uniform u, and where), summarised as median [IQR] over seeds;
  5. discovery (first arrival) vs establishment (instantaneous / cumulative) vs accumulated information
     (cumulative far-state fraction, region occupation, true events, lineage coverage).

The bootstrap uses eqb_metrics.N_BOOT resamples and seed eqb_metrics.BOOT_SEED (the production constants), with
one fresh generator per contrast, like the production paired_contrast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eqb_metrics as M  # noqa: E402  (constants and quantile_inf only; the sign-test p is read from the summary)

ROOT = M.ROOT
SYSTEMS = (
    dict(dir="gateway", tag="G", label="Entropic gateway (reduced units)", kind="gateway"),
    dict(dir="lta_T300", tag="L300", label="LTA 300 K", kind="lta"),
    dict(dir="lta_T150", tag="L150", label="LTA 150 K", kind="lta"),
)
MEAN_KEYS = ("Ibar_F", "final_e_F", "Ibar_Fp", "final_e_Fp")
TRANSIENT = ("e_F", "e_Fp", "TV_half")
# per-run curve -> (summary Ibar key, summary final key) used to verify the cache against the summary
CURVE_CHECK = {"e_F": ("Ibar_F", "final_e_F"), "e_Fp": ("Ibar_Fp", "final_e_Fp"),
               "TV_half": ("Ibar_TV_half", "final_TV_half")}
LABEL = {"Ibar_F": "Ibar_F", "final_e_F": "e_F(1)", "Ibar_Fp": "Ibar_F'", "final_e_Fp": "e_F'(1)",
         "Ibar_Fp_stat": "Ibar_F'_stat", "final_e_Fp_stat": "e_F'_stat(1)", "e_F": "e_F", "e_Fp": "e_F'",
         "TV_half": "TV_half"}


# --------------------------------------------------------------------------------------------- loading
def _conv(o):
    """summary.json -> numbers ('inf' strings -> inf, null -> nan), like analyze_ladder._reload."""
    if isinstance(o, dict):
        return {k: _conv(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_conv(v) for v in o]
    if o is None:
        return math.nan
    if o == "inf":
        return math.inf
    if o == "-inf":
        return -math.inf
    return o


def load_summary(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return _conv(json.loads(raw)), hashlib.sha256(raw).hexdigest()


def per_seed(S, N, m, k):
    """{seed (int): value} of per_N[N][m][k].per_seed (empty if the arm / key is absent)."""
    b = (S["per_N"].get(str(N)) or {}).get(m)
    if not b or k not in b:
        return {}
    return {int(s): float(v) for s, v in b[k]["per_seed"].items()}


# --------------------------------------------------------------------------------------------- statistics
def mean_ratio_contrast(abf, fr, n_boot=M.N_BOOT, seed=M.BOOT_SEED):
    """MEAN-based paired contrast.  abf, fr: {seed: value} at one N.  Seeds with both arms finite are used;
    G = (mean FR - mean ABF) / mean ABF (a ratio of means, NOT the mean or median of per-seed ratios).  CI: paired
    seed bootstrap (one index draw resamples the seeds of BOTH arms), G recomputed as a ratio of means in every
    resample, percentile 2.5 / 97.5.  frac_boot_neg = fraction of resamples with G < 0 (FR better)."""
    common = sorted(set(abf) & set(fr))
    use = [s for s in common if math.isfinite(abf[s]) and math.isfinite(fr[s])]
    a = np.array([abf[s] for s in use], float)
    f = np.array([fr[s] for s in use], float)
    out = dict(n=len(use), n_common=len(common), mean_abf=math.nan, mean_fr=math.nan, G=math.nan,
               ci95=[math.nan, math.nan], frac_boot_neg=math.nan, frac_boot_undefined=math.nan,
               n_boot=int(n_boot), seed=int(seed))
    if not use:
        return out
    ma0, mf0 = float(a.mean()), float(f.mean())
    out.update(mean_abf=ma0, mean_fr=mf0, G=(mf0 - ma0) / ma0 if ma0 != 0 else math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(use), size=(int(n_boot), len(use)))
    ma, mf = a[idx].mean(axis=1), f[idx].mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        Gb = np.where(ma != 0, (mf - ma) / np.where(ma != 0, ma, 1.0), np.nan)
    ok = ~np.isnan(Gb)
    if ok.any():
        out.update(ci95=[float(np.percentile(Gb[ok], 2.5)), float(np.percentile(Gb[ok], 97.5))],
                   frac_boot_neg=float(np.mean(Gb[ok] < 0)))
    out["frac_boot_undefined"] = float(np.mean(~ok))
    return out


def mean_best_allocation(table, Ns_abf, Ns_fr, n_boot=M.N_BOOT, seed=M.BOOT_SEED):
    """MEAN-based best allocation (lower is better; finite values).  table: {N: {"abf": {seed: v}, "fr": {...}}}.
    Point: min over Ns_abf of the ABF seed mean vs min over Ns_fr of the FR seed mean.  Bootstrap, mirroring the
    production eqb_metrics.best_allocation with the mean in place of the median: for every N (ascending) one seed
    index draw shared by both methods, independent across N; the minimising N is RE-SELECTED in every resample;
    ties share the credit (boot_tie_frac = fraction of resamples with a tie); an N whose resample has no usable
    seed is skipped in that resample (boot_frac_no_data when no candidate is left)."""
    rng = np.random.default_rng(seed)
    Ns = sorted(set(Ns_abf) | set(Ns_fr))
    mean = {"abf": {}, "fr": {}}
    boot = {"abf": {}, "fr": {}}
    for N in Ns:
        d = table.get(N, {})
        seeds = sorted(set(d.get("abf", {})) | set(d.get("fr", {})))
        if not seeds:
            continue
        X = np.array([[d.get(m, {}).get(s, np.nan) for m in ("abf", "fr")] for s in seeds], float)
        X[~np.isfinite(X)] = np.nan
        idx = rng.integers(0, len(seeds), size=(int(n_boot), len(seeds)))
        Xb = X[idx]                                           # (n_boot, n_seeds, 2)
        for c, m in enumerate(("abf", "fr")):
            if np.all(np.isnan(X[:, c])):
                continue
            mean[m][N] = float(np.nanmean(X[:, c]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN resample -> NaN (no usable seed)
                boot[m][N] = np.nanmean(Xb[:, :, c], axis=1)
    out = {}
    for m, Nset in (("abf", Ns_abf), ("fr", Ns_fr)):
        cand = [N for N in sorted(Nset) if N in mean[m]]
        if not cand:
            out[m] = None
            continue
        vals = np.array([mean[m][N] for N in cand])
        k = int(np.argmin(vals))
        tied_pt = [int(N) for N, v in zip(cand, vals) if v == vals[k]]
        B = np.stack([boot[m][N] for N in cand], axis=1)       # (n_boot, n_cand)
        nodata = np.all(np.isnan(B), axis=1)
        Bf = np.where(np.isnan(B), np.inf, B)
        bmin = Bf.min(axis=1)
        finb = np.isfinite(bmin)
        tied = (Bf == bmin[:, None]) & finb[:, None]
        n_tied = tied.sum(axis=1)
        credit = np.where(finb[:, None], tied / np.maximum(n_tied, 1)[:, None], 0.0)
        bestb = np.where(nodata, np.nan, bmin)
        okb = ~np.isnan(bestb)
        out[m] = dict(best_N=int(cand[k]), best_value=float(vals[k]), best_N_tied=tied_pt if len(tied_pt) > 1 else [],
                      means={int(N): float(mean[m][N]) for N in cand}, Ns=[int(N) for N in cand],
                      boot_best_N_freq={int(N): float(np.mean(credit[:, i])) for i, N in enumerate(cand)},
                      boot_tie_frac=float(np.mean(n_tied > 1)), boot_frac_no_data=float(np.mean(nodata)),
                      boot_best_value_ci95=([float(np.percentile(bestb[okb], 2.5)), float(np.percentile(bestb[okb], 97.5))]
                                            if okb.any() else [math.nan, math.nan]),
                      _boot=bestb)
    if out.get("abf") and out.get("fr"):
        da, df = out["abf"]["_boot"], out["fr"]["_boot"]
        diff = df - da
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(da != 0, diff / np.where(da != 0, da, 1.0), np.nan)
        ok, okr = ~np.isnan(diff), ~np.isnan(rel)
        a0, f0 = out["abf"]["best_value"], out["fr"]["best_value"]
        out["fr_minus_abf"] = dict(
            point=f0 - a0, point_rel=(f0 - a0) / a0 if a0 != 0 else math.nan,
            ci95=[float(np.percentile(diff[ok], 2.5)), float(np.percentile(diff[ok], 97.5))] if ok.any() else [math.nan] * 2,
            rel_ci95=[float(np.percentile(rel[okr], 2.5)), float(np.percentile(rel[okr], 97.5))] if okr.any() else [math.nan] * 2,
            frac_resamples_fr_better=float(np.mean(df[ok] < da[ok])) if ok.any() else math.nan)
    for m in ("abf", "fr"):
        if out.get(m):
            out[m].pop("_boot")
    out["n_boot"], out["seed"] = int(n_boot), int(seed)
    return out


def describe(x):
    """median / q25 / q75 / n of a 1-D array (eqb_metrics.quantile_inf: numpy's linear rule, inf-aware)."""
    x = np.asarray(x, float)
    return dict(median=M.quantile_inf(x, 50), q25=M.quantile_inf(x, 25), q75=M.quantile_inf(x, 75),
                n=int(np.sum(~np.isnan(x))))


def per_run_transient(cache_dir, N, T, seeds, keys, S=None, n_uniform=M.N_UNIFORM):
    """Per seed with both arms cached: rel(u) = (e_ABF(u) - e_FR(u)) / e_ABF(u) on the n_uniform budget fractions
    (curve[uniform_index], the grid of analyze_ladder), its max and the u (t = u T) where it is reached (first
    occurrence; points with e_ABF = 0 or non-finite rel are skipped), and rel at u = 1.  When S (the summary) is
    given, every curve used is verified first: mean over the uniform u and the last value must equal the summary's
    per-seed Ibar / final value (CURVE_CHECK).  Returns {key: dict(max_rel, u_at_max, t_at_max, final_rel arrays,
    seeds)}."""
    u_grid = np.arange(1, n_uniform + 1) / n_uniform
    out = {k: dict(max_rel=[], u_at_max=[], t_at_max=[], final_rel=[], seeds=[]) for k in keys}
    for s in seeds:
        z = {}
        for m in ("abf", "fr"):
            p = os.path.join(cache_dir, f"N{N}", f"s{s}_{m}.npz")
            if not os.path.exists(p):
                z = None
                break
            with np.load(p, allow_pickle=False) as d:
                iu = np.asarray(d["uniform_index"], int)
                if iu.size != n_uniform:
                    raise SystemExit(f"{p}: uniform_index has {iu.size} entries, expected {n_uniform}")
                if not np.allclose(np.asarray(d["save_u"], float)[iu], u_grid, rtol=0, atol=1e-9):
                    raise SystemExit(f"{p}: save_u[uniform_index] is not the uniform grid k/{n_uniform}")
                z[m] = {k: np.asarray(d[k], float) for k in keys}
                z[m]["_iu"] = iu
        if z is None:
            continue
        for k in keys:
            ea, ef = z["abf"][k][z["abf"]["_iu"]], z["fr"][k][z["fr"]["_iu"]]
            if S is not None and k in CURVE_CHECK:
                for m, e, full in (("abf", ea, z["abf"][k]), ("fr", ef, z["fr"][k])):
                    for sk, val in ((CURVE_CHECK[k][0], float(np.mean(e))), (CURVE_CHECK[k][1], float(full[-1]))):
                        ref = per_seed(S, N, m, sk).get(int(s))
                        if ref is None or not np.isclose(val, ref, rtol=1e-9, atol=1e-15):
                            raise SystemExit(f"stale run_metrics cache: {cache_dir}/N{N}/s{s}_{m}.npz {k} gives "
                                             f"{sk} = {val!r}, summary.json per_seed has {ref!r}")
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.where((ea != 0) & np.isfinite(ea), (ea - ef) / np.where(ea != 0, ea, 1.0), np.nan)
            if not np.any(np.isfinite(rel)):
                continue
            j = int(np.nanargmax(rel))
            r = out[k]
            r["max_rel"].append(float(rel[j]))
            r["u_at_max"].append(float(u_grid[j]))
            r["t_at_max"].append(float(u_grid[j] * T))
            r["final_rel"].append(float(rel[-1]))
            r["seeds"].append(int(s))
    return out


# --------------------------------------------------------------------------------------------- formatting
# the tables.md number style (analyze_ladder.f_val / med_iqr / contrast_cell)
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


def med_iqr(d, tau=None, T=None, kind="g"):
    if not d:
        return "--"
    return (f"{f_val(d['median'], kind, tau=tau, T=T)} [{f_val(d['q25'], kind, tau=tau, T=T)}, "
            f"{f_val(d['q75'], kind, tau=tau, T=T)}]")


def contrast_cell(c):
    if not c or c["n"] == 0:
        return "--"
    return (f"{f_val(c['G_median'], 'pct')} [{f_val(c['G_ci95'][0], 'pct')}, {f_val(c['G_ci95'][1], 'pct')}] "
            f"({c['wins']}/{c['n']})")


def freq_cell(x):
    txt = ", ".join(f"{n}: {p:.2f}" for n, p in sorted(x["boot_best_N_freq"].items(), key=lambda kv: -kv[1]) if p > 0)
    for lab, key in (("all censored", "boot_frac_censored"), ("no data", "boot_frac_no_data"), ("ties", "boot_tie_frac")):
        v = x.get(key) or 0.0
        if v > 0:
            txt += (", " if txt else "") + f"{lab}: {v:.2f}"
    if x.get("Ns_missing"):
        txt += f"; **REDUCED ladder: N {x['Ns_missing']} missing**"
    return txt


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]


def caption(tag, title, source, note=None):
    s = f"**Table {tag}. {title}** Source: {source}."
    return ["", s + (f" {note}" if note else ""), ""]


# --------------------------------------------------------------------------------------------- sections
class Report:
    def __init__(self, S, sysd, cache_dir, n_boot):
        self.S, self.sysd, self.kind, self.tag = S, sysd, sysd["kind"], sysd["tag"]
        self.cache_dir, self.n_boot = cache_dir, n_boot
        self.pn = S["per_N"]
        self.Ns = list(self.pn)                       # summary order (descending N)
        self.cN = list(S["contrasts"])
        self.check = []                               # completeness notes

    def g(self, N, m, k):
        b = self.pn.get(N, {}).get(m)
        return b.get(k) if b else None

    def T(self, N):
        return self.pn[N]["T"]

    def arms(self, N):
        return [m for m in ("abf", "fr") if self.pn[N].get(m)]

    def fr_act(self, N):
        fa = (self.S.get("fr_activity") or {}).get(N) or {}
        if fa.get("inactive"):
            return "**FR INACTIVE (no death in any seed)**"
        if fa.get("n_zero_deaths"):
            return f"**no death in {fa['n_zero_deaths']}/{fa['n_fr_seeds']} seeds**"
        return f"{fa['median_deaths_per_walker']:.3g} deaths / walker" if fa else "--"

    # ---- 1. absolute values
    def sec_absolute(self):
        t = self.tag
        L = ["", "### 1. Absolute values per N and arm (median [IQR] over seeds)"]
        keys = ["Ibar_F", "final_e_F", "Ibar_Fp", "final_e_Fp"]
        L += caption(f"{t}.1a", "Errors.", "`per_N[N][abf|fr][k].{median, q25, q75}` for k in "
                     "`Ibar_F, final_e_F, Ibar_Fp, final_e_Fp`",
                     "N = 1 has no FR arm ('--'). Ibar = mean over the 200 uniform budget fractions u; (1) = at u = 1.")
        head = ["N", "T_N"] + [f"{m.upper()} {LABEL[k]}" for k in keys for m in ("abf", "fr")]
        rows = [[N, f"{self.T(N):g}"] + [med_iqr(self.g(N, m, k)) for k in keys for m in ("abf", "fr")] for N in self.Ns]
        L += table(head, rows)
        if self.kind == "gateway":
            keys = ["Ibar_Fp_stat", "final_e_Fp_stat"]
            L += caption(f"{t}.1b", "Floor-free companion of e_F' (gateway only, descriptive).",
                         "`per_N[N][abf|fr][k]` for k in `Ibar_Fp_stat, final_e_Fp_stat`",
                         "e_F'_stat = RMS over the eval bins of Gamma_j - <F'_ref>_j; e_F'^2 = e_F'_stat^2 + floor^2 with the "
                         f"hard floor {self.S['threshold_floors']['tau_e_Fp_mid']['floor']:.4g} "
                         "(`threshold_floors.tau_e_Fp_*.floor`).")
            head = ["N", "T_N"] + [f"{m.upper()} {LABEL[k]}" for k in keys for m in ("abf", "fr")]
            rows = [[N, f"{self.T(N):g}"] + [med_iqr(self.g(N, m, k)) for k in keys for m in ("abf", "fr")] for N in self.Ns]
            L += table(head, rows)
        L += caption(f"{t}.1c", "Marginal: TV_half and TV_inst vs its exact finite-N floor.",
                     "`per_N[N][abf|fr][k]` for k in `Ibar_TV_half, final_TV_half, final_TV_inst, final_TV_inst_excess`; "
                     "floor E_N[TV] from `tv_floor.table[N].mean` (identical to `per_N[N][arm].TV_inst_floor`)",
                     "TV on 18 coarse bins vs uniform; E_N[TV] = exact multinomial(N, uniform 18) expectation "
                     "(`tv_floor.method`); excess = TV_inst(1) - E_N[TV] per seed.")
        keys = ["Ibar_TV_half", "final_TV_half", "final_TV_inst"]
        head = (["N"] + [f"{m.upper()} {k.replace('final_', '')}{'(1)' if k.startswith('final_') else ''}"
                         for k in keys for m in ("abf", "fr")] + ["E_N[TV]", "ABF excess(1)", "FR excess(1)"])
        rows = []
        for N in self.Ns:
            fl = self.S["tv_floor"]["table"][N]["mean"]
            pf = self.g(N, "abf", "TV_inst_floor")
            if pf and not math.isclose(pf["median"], fl, rel_tol=1e-12):
                raise SystemExit(f"{self.sysd['dir']} N {N}: per_N TV_inst_floor {pf['median']} != tv_floor.table {fl}")
            rows.append([N] + [med_iqr(self.g(N, m, k)) for k in keys for m in ("abf", "fr")]
                        + [f"{fl:.4f}", med_iqr(self.g(N, "abf", "final_TV_inst_excess")),
                           med_iqr(self.g(N, "fr", "final_TV_inst_excess"))])
        L += table(head, rows)
        return L

    # ---- 2. mean-based contrasts and best allocation
    def sec_mean(self):
        t, S = self.tag, self.S
        L = ["", "### 2. MEAN-based paired contrasts and MEAN-based best allocation",
             "", f"G_X(N) = (mean_seeds X^FR - mean_seeds X^ABF) / mean_seeds X^ABF over the seeds that have both arms "
                 f"(ratio of seed means); 95 % CI from a paired seed bootstrap ({self.n_boot} resamples, seed "
                 f"{M.BOOT_SEED}: resample seeds with replacement, recompute the ratio of means); P_boot(G < 0) = "
                 "share of resamples in which FR is better. The production column is the frozen median of the "
                 "per-seed ratios (`contrasts[N].primary[k]`, tables.md) for comparison."]
        self.mean_contrasts = {}
        for i, k in enumerate(MEAN_KEYS):
            L += caption(f"{t}.2{'abcd'[i]}", f"Mean-based contrast of {LABEL[k]}.",
                         f"per-seed values `per_N[N][abf|fr].{k}.per_seed` (mean contrast, computed here); production "
                         f"median contrast `contrasts[N].primary.{k}`; FR activity `fr_activity[N]`")
            rows = []
            for N in self.cN:
                c = mean_ratio_contrast(per_seed(S, N, "abf", k), per_seed(S, N, "fr", k), self.n_boot, M.BOOT_SEED)
                self.mean_contrasts[(N, k)] = c
                rows.append([N, str(c["n"]), f_val(c["mean_abf"]), f_val(c["mean_fr"]),
                             f"{f_val(c['G'], 'pct')} [{f_val(c['ci95'][0], 'pct')}, {f_val(c['ci95'][1], 'pct')}]",
                             f_val(c["frac_boot_neg"]), contrast_cell(S["contrasts"][N]["primary"].get(k)), self.fr_act(N)])
            L += table(["N", "n seeds", "mean ABF", "mean FR", "G_mean [95 % CI]", "P_boot(G < 0)",
                        "production G median [95 % CI] (wins/n)", "FR activity"], rows)
        # best allocation
        k = "Ibar_F"
        Ns_abf = [int(N) for N in self.Ns]
        Ns_fr = [int(N) for N in self.Ns if int(N) >= 2]
        tab = {int(N): {m: per_seed(S, N, m, k) for m in self.arms(N)} for N in self.Ns}
        ba = mean_best_allocation(tab, Ns_abf, Ns_fr, self.n_boot, M.BOOT_SEED)
        self.mean_best = ba
        L += caption(f"{t}.2e", "Selection inputs of the best allocation for Ibar_F: seed mean and seed median per N.",
                     "`per_N[N][abf|fr].Ibar_F.{mean, median}` (the mean is recomputed from `per_seed` and checked "
                     "against `.mean`)")
        rows = []
        for N in self.Ns:
            cells = [N]
            for stat in ("mean", "median"):
                for m in ("abf", "fr"):
                    d = self.g(N, m, k)
                    if d is None:
                        cells.append("--")
                        continue
                    if stat == "mean":
                        v = ba[m]["means"].get(int(N)) if ba.get(m) else None
                        if v is None or not math.isclose(v, d["mean"], rel_tol=1e-12):
                            raise SystemExit(f"{self.sysd['dir']} N {N} {m}: recomputed Ibar_F seed mean {v} != "
                                             f"summary {d['mean']}")
                    cells.append(f_val(d[stat]))
            rows.append(cells)
        L += table(["N", "ABF mean", "FR mean", "ABF median", "FR median"], rows)
        prod = S["best_allocation"][k]
        L += caption(f"{t}.2f", "Best allocation for Ibar_F: production MEDIAN-based vs MEAN-based.",
                     "production row `best_allocation.Ibar_F` (median over seeds, eqb_metrics.best_allocation); mean row "
                     "computed here from `per_N[N][abf|fr].Ibar_F.per_seed`",
                     "ABF over all N, FR over N >= 2. The bootstrap resamples the seeds of every N (one draw shared by "
                     "both arms, independent across N) and re-selects the minimising N in each resample; ties share the "
                     "credit. FR - ABF = best FR value - best ABF value (CI from the same resamples).")
        rows = []
        for lab, b, vk in (("production: seed MEDIAN", prod, "medians"), ("this appendix: seed MEAN", ba, "means")):
            row = [lab]
            for m in ("abf", "fr"):
                x = b.get(m)
                if not x:
                    row += ["--"] * 4
                    continue
                bn = str(x["best_N"]) if x["best_N"] is not None else "all censored"
                if x.get("best_N_tied"):
                    bn += f" (tied: {x['best_N_tied']})"
                row += [bn, f_val(x["best_value"]), f"[{f_val(x['boot_best_value_ci95'][0])}, "
                        f"{f_val(x['boot_best_value_ci95'][1])}]", freq_cell(x)]
            d = b.get("fr_minus_abf")
            row += ([f"{f_val(d['point'])} [{f_val(d['ci95'][0])}, {f_val(d['ci95'][1])}]",
                     f"{f_val(d['point_rel'], 'pct')} [{f_val(d['rel_ci95'][0], 'pct')}, {f_val(d['rel_ci95'][1], 'pct')}]",
                     f_val(d["frac_resamples_fr_better"])] if d else ["--"] * 3)
            rows.append(row)
        L += table(["statistic", "ABF best N", "ABF best", "ABF 95 % CI", "ABF N frequency", "FR best N (N >= 2)",
                    "FR best", "FR 95 % CI", "FR N frequency", "best FR - best ABF [95 % CI]", "rel. [95 % CI]",
                    "P(FR better)"], rows)
        return L

    # ---- 3. tau
    def tau_metrics(self):
        thr = self.S["plan"]["thresholds"]
        out = []
        for fam, lab in (("e_F", "e_F"), ("e_Fp", "e_F'")):
            for i, n in enumerate(M.THR_NAMES):
                out.append((f"tau_{fam}_{n}", f"{lab} {n}", thr[fam][i]))
        out.append(("tau_TV_half", "TV_half", thr["TV_half"]))
        return out

    def sec_tau(self):
        t, S = self.tag, self.S
        L = ["", "### 3. Persistent time-to-accuracy tau (physical time t in t.u. and budget fraction u)", "",
             "tau = first save after which the metric stays <= eps at every later save (all saves of the run); "
             "censored = never met (shown as '> T_N' in t and '> 1' in u; the median / quartile is censored when its "
             "interpolation neighbour is). FR wins when tau_FR < tau_ABF (finite beats censored; censored vs censored "
             "ties); sign p = exact two-sided binomial test, ties excluded."]
        for i, (base, lab, eps) in enumerate(self.tau_metrics()):
            tf = (S.get("threshold_floors") or {}).get(base)
            unr = bool(tf and not tf["reachable"])
            note = f"eps = {eps:g}."
            if unr:
                note += (f" **UNREACHABLE BY CONSTRUCTION** (`threshold_floors.{base}`: hard floor {tf['floor']:.4g} > eps): "
                         "censored in every arm, the ties are NOT evidence of equivalence.")
            L += caption(f"{t}.3{chr(ord('a') + i)}", f"tau for {lab}.",
                         f"`per_N[N][abf|fr].{{{base}_t, {base}_u}}.{{median, q25, q75}}`, censored fraction "
                         f"`per_N[N][abf|fr].{base}_censored.mean`; paired rank test `contrasts[N].tau.{base}_u.rank`", note)
            rows = []
            for N in self.Ns:
                T = self.T(N)
                cells = [N, f"{T:g}"]
                for m in ("abf", "fr"):
                    if not self.pn[N].get(m):
                        cells += ["--"] * 3
                        continue
                    cz = self.g(N, m, base + "_censored")
                    cells += [med_iqr(self.g(N, m, base + "_t"), tau="t", T=T), med_iqr(self.g(N, m, base + "_u"), tau="u"),
                              f_val(cz["mean"]) if cz else "--"]
                c = (S["contrasts"].get(N) or {}).get("tau", {}).get(base + "_u")
                if c:
                    r = c["rank"]
                    cells += [f"{r['wins']}/{r['losses']}/{r['ties']}", f"{r['sign_test_p']:.3g}"]
                    if c.get("unreachable_by_construction") != unr:
                        raise SystemExit(f"{self.sysd['dir']} N {N} {base}: unreachable flag disagrees")
                else:
                    cells += ["--", "--"]
                rows.append(cells)
            L += table(["N", "T_N", "ABF tau t", "ABF tau u", "ABF cens.", "FR tau t", "FR tau u", "FR cens.",
                        "FR W/L/T", "sign p"], rows)
        return L

    # ---- 4. max transient
    def sec_transient(self):
        t, S = self.tag, self.S
        L = ["", "### 4. Maximum transient improvement (ABF - FR)/ABF over the 200 uniform u", "",
             "Median-curve columns: the seed-median curves of each arm, rel(u) = (med ABF - med FR)/med ABF, its max, "
             "where (u and t = u T_N) and its value at u = 1 (`max_transient`, as in tables.md). Per-run columns: for "
             "every seed with both arms, rel_s(u) = (e_ABF,s(u) - e_FR,s(u))/e_ABF,s(u) on the same 200 u from the "
             "per-run curve cache (`run_metrics/N<N>/s<seed>_<arm>.npz`, curve[uniform_index]; every curve is first "
             "checked against the summary's per-seed Ibar and final value), max_u rel_s and its u; median [IQR] over "
             "seeds. Positive = FR better. The per-run max is a max over 200 noisy points, so it is biased upwards "
             "(it is positive even for two statistically identical arms); compare it across N and with its u, not "
             "with zero."]
        self.transient_n = {}
        prs = {}
        for N in self.cN:                             # one pass over the cache per N (all three curves)
            seeds = sorted(set(per_seed(S, N, "abf", "Ibar_F")) & set(per_seed(S, N, "fr", "Ibar_F")))
            prs[N] = (seeds, per_run_transient(self.cache_dir, N, self.T(N), seeds, list(TRANSIENT), S))
        for i, ck in enumerate(TRANSIENT):
            L += caption(f"{t}.4{'abc'[i]}", f"Transient improvement of {LABEL[ck]}.",
                         f"median curves `max_transient[N].{ck}.{{max_rel, u_at_max, t_at_max, final_rel, n}}`; per run "
                         f"`run_metrics/N<N>/s<seed>_<abf|fr>.npz` key `{ck}` (computed here)")
            rows = []
            for N in self.cN:
                T = self.T(N)
                mt = (S["max_transient"].get(N) or {}).get(ck)
                seeds, pr = prs[N][0], prs[N][1][ck]
                self.transient_n[(N, ck)] = (len(pr["seeds"]), len(seeds))
                if mt and int(mt["n"]) != len(seeds):          # the median curves must be over the same seeds
                    self.check.append(f"N {N} {ck}: max_transient n = {mt['n']} != {len(seeds)} paired seeds")
                rows.append([N, f"{T:g}"]
                            + ([f_val(mt["max_rel"], "pct"), f_val(mt["u_at_max"]), f_val(mt["t_at_max"]),
                                f_val(mt["final_rel"], "pct")] if mt else ["--"] * 4)
                            + [med_iqr(describe(pr["max_rel"]), kind="pct"), med_iqr(describe(pr["u_at_max"])),
                               med_iqr(describe(pr["t_at_max"])), f"{len(pr['seeds'])}/{len(seeds)}"])
            L += table(["N", "T_N", "median-curve max", "at u", "at t", "at u = 1", "per-run max [IQR]",
                        "per-run u at max [IQR]", "per-run t at max [IQR]", "seeds (cached/paired)"], rows)
        return L

    # ---- 5. discovery vs establishment vs accumulated information
    def sec_discovery(self):
        t, S, kind = self.tag, self.S, self.kind
        far = S["plan"]["far_state"]
        L = ["", "### 5. Discovery vs establishment vs accumulated information", "",
             f"Far state: {far['name']} (uniform-target fraction {far['target']:g}). Discovery = first arrival of any "
             "walker; establishment (instantaneous) = first save after which the instantaneous far-state fraction "
             "stays >= 1/2 target; establishment (cumulative) = the same on the cumulative visitation C_all; "
             "accumulated information = the final cumulative far-state fraction, the occupation over the run and "
             "the true events."]
        if kind == "gateway":
            fa = [("first_right", "right well")]
        else:
            fa = [("first_window", "window"), ("first_opposite", "opposite cage")]
        L += caption(f"{t}.5a", "Discovery: first arrival (median [IQR], censored fraction).",
                     "; ".join(f"`per_N[N][abf|fr].{{{k}_u, {k}_t}}`, `{k}_censored.mean`" for k, _ in fa),
                     "u = budget fraction, t in t.u.; a run in which no walker ever arrives is censored.")
        head = ["N", "arm"]
        for _, lab in fa:
            head += [f"first {lab} u", f"first {lab} t", "cens."]
        rows = []
        for N in self.Ns:
            T = self.T(N)
            for m in self.arms(N):
                cells = [N, m]
                for k, _ in fa:
                    cz = self.g(N, m, k + "_censored")
                    cells += [med_iqr(self.g(N, m, k + "_u"), tau="u"), med_iqr(self.g(N, m, k + "_t"), tau="t", T=T),
                              f_val(cz["mean"]) if cz else "--"]
                rows.append(cells)
        L += table(head, rows)
        b0 = self.pn[self.Ns[0]]["abf"]
        L += caption(f"{t}.5b", "Establishment: instantaneous (with the null-calibration flag) and cumulative.",
                     "`per_N[N][abf|fr].{est_u, est_t}`, `est_censored.mean`, `est_unreliable.mean`, `est_ill_defined.mean`; "
                     "`per_N[N][abf|fr].{est_cum_u, est_cum_censored, final_far_frac_cum}`; flag cross-checked with "
                     "`establishment_null[N].unreliable`",
                     f"Instantaneous threshold {b0['est_threshold']['median']:g} (= 1/2 x {far['target']:g}); cumulative "
                     f"threshold {b0['est_cum_threshold']['median']:.4g} (= 1/2 x the far-bin share "
                     f"{b0['est_cum_bin_target']['median']:.4g}). **UNRELIABLE** = the null calibration (an exactly uniform "
                     "population, Bin(N, target) per save) fails the instantaneous criterion by chance at this N "
                     f"(P(no failure, u > 1/2) < {M.EST_RELIABLE_P}) or N <= 2: read the cumulative establishment and the "
                     "first arrival instead; an FR advantage in est. u there is not evidence of faster convergence.")
        rows = []
        en = S.get("establishment_null") or {}
        for N in self.Ns:
            T = self.T(N)
            for m in self.arms(N):
                ur, ill = self.g(N, m, "est_unreliable"), self.g(N, m, "est_ill_defined")
                urv = ur["mean"] if ur else math.nan
                if N in en and bool(en[N]["unreliable"]) != (urv > 0):
                    raise SystemExit(f"{self.sysd['dir']} N {N} {m}: est_unreliable disagrees with establishment_null")
                flag = ("**UNRELIABLE** (ill-defined, N <= 2)" if ill and ill["mean"] > 0 else
                        ("**UNRELIABLE**" if urv == 1 else (f"**UNRELIABLE in {urv:.2f} of seeds**" if urv > 0 else "reliable")))
                cz, czc = self.g(N, m, "est_censored"), self.g(N, m, "est_cum_censored")
                rows.append([N, m, med_iqr(self.g(N, m, "est_u"), tau="u"), med_iqr(self.g(N, m, "est_t"), tau="t", T=T),
                             f_val(cz["mean"]) if cz else "--", flag, med_iqr(self.g(N, m, "est_cum_u"), tau="u"),
                             f_val(czc["mean"]) if czc else "--", med_iqr(self.g(N, m, "final_far_frac_cum"))])
        L += table(["N", "arm", "inst. est. u", "inst. est. t", "cens.", "flag", "cum. est. u", "cum. cens.",
                    "final far frac (cum.)"], rows)
        if kind == "gateway":
            regs = ["left", "gate", "right"]
            ev = [("final_transitions", "transitions L<->R"), ("final_trans_LR", "L->R"), ("final_trans_RL", "R->L"),
                  ("transitions_per_t", "transitions / t.u.")]
            lin = [("final_lineage_right_frac", "lineage ever right")]
            regdef = "left x < -0.5 / gate / right x > 0.5"
            evdef = ("TRUE cross-well transitions under the hysteresis label (x < -0.5 / x > 0.5), summed over walkers "
                     "(clones inherit the label, so a copy is never a transition)")
        else:
            regs = ["cage", "neck", "window"]
            ev = [("final_window_crossings", "window crossings"), ("final_transitions", "translocations"),
                  ("transitions_per_t", "translocations / t.u.")]
            lin = [("final_lineage_window_frac", "lineage ever window"),
                   ("final_lineage_opposite_frac", "lineage ever translocated")]
            regdef = "cage |z| > 4 A / neck / window |z| < 1.5 A"
            evdef = ("TRUE events summed over walkers: window-plane crossings = changes of floor(x_COM / a); "
                     "translocation = arrival in a cage of a different x-cell than the lineage's last cage (clones "
                     "inherit labels, so a copy is never an event)")
        keys = [f"mean_frac_{r}" for r in regs] + [k for k, _ in ev] + [k for k, _ in lin]
        L += caption(f"{t}.5c", "Accumulated information: region occupation, true events and lineage coverage "
                                "(median [IQR]).",
                     "`per_N[N][abf|fr].{" + ", ".join(keys) + "}`",
                     f"Occupation = instantaneous region fraction averaged over the 200 uniform u ({regdef}). Events: "
                     f"{evdef}, cumulative at u = 1. Lineage coverage = share of the N final slots whose lineage ever "
                     "visited (flags are inherited by FR clones, so for the FR arm it is not a count of independent "
                     "discoveries).")
        head = ["N", "arm"] + [f"occ. {r}" for r in regs] + [lab for _, lab in ev] + [lab for _, lab in lin]
        rows = [[N, m] + [med_iqr(self.g(N, m, k)) for k in keys] for N in self.Ns for m in self.arms(N)]
        L += table(head, rows)
        if kind == "gateway":
            L += ["", "Round trips: NOT computed. The gateway engine (src/gateway_ladder_numba.py) saves its event "
                      "counts only walker-summed: the cumulative directional transition counts `events_cum` [L->R, "
                      "R->L], the lineage count `lineage_visited` and `first_arrival_step`. There is no per-walker leg "
                      "pairing over all N walkers, so an exact round-trip count cannot be reconstructed (min(L->R, "
                      "R->L) is NOT one). The raw run files also hold position traces of min(N, 32) walkers "
                      "(`traces`, every 0.1 t.u. up to t = 40, then every >= 10 t.u.; a trace jumps when its walker "
                      "dies under FR). A round-trip estimate from them would cover at most 32 walkers, would miss "
                      "returns between trace points and, in the FR arm, would mix in replacements. It is not "
                      "computed here because the traces are outside this script's inputs. Only transitions are "
                      "reported."]
        else:
            L += ["", "Round trips: NOT computed. The LTA engine (src/lta_ladder_numba.py) saves its event counts "
                      "only walker-summed: `events_window_crossings`, `events_translocations` and the torch-equivalent "
                      "`n_transitions` / `n_cage_crossings_lineage`, the lineage counts `lineage_ever_window` / "
                      "`lineage_ever_opposite` and `first_window_step` / `first_opposite_step`. There is no "
                      "per-walker leg pairing over all N walkers, so an exact round-trip count cannot be "
                      "reconstructed. The raw run files also hold SLOT traces of min(N, 32) slots (`traces`, every "
                      "0.05 t.u. up to t = 60, then every >= 10 t.u.; a slot trace jumps when the slot is overwritten "
                      "by an FR copy). A round-trip estimate from them would cover at most 32 slots, would miss "
                      "returns between trace points and, in the FR arm, would mix in replacements. It is not "
                      "computed here because the traces are outside this script's inputs. Only crossings and "
                      "translocations are reported."]
        return L

    def render(self, path_rel, sha):
        S, pl = self.S, self.S["plan"]
        L = ["", f"## {self.sysd['label']} (`{S['system']}`)", "",
             f"Input `{path_rel}` (sha256 `{sha}`; generated {S['generated_utc']}; {S['metrics_version']}; schema "
             f"{S['schema']}). B = {pl['B']:,} walker-steps per arm per seed; h = {pl['h']:g}; seeds {len(pl['seeds'])} "
             f"({pl['seeds'][0]}-{pl['seeds'][-1]}); N ladder {pl['N_ladder']}; thresholds e_F {pl['thresholds']['e_F']}, "
             f"e_F' {pl['thresholds']['e_Fp']}, TV_half {pl['thresholds']['TV_half']}. T_N = B h / N."]
        if S.get("warnings"):
            L += ["", "Analysis warnings carried by the summary (`warnings`): " + "; ".join(S["warnings"])]
        L += self.sec_absolute() + self.sec_mean() + self.sec_tau() + self.sec_transient() + self.sec_discovery()
        return L

    def completeness(self):
        """Completeness of the inputs used: every planned N has ABF data, every N >= 2 has FR data and a contrast,
        every per-run transient covers all paired seeds, the status block reports no missing / running / invalid."""
        probs = list(self.check)
        pl = self.S["plan"]
        for N in pl["N_ladder"]:
            N = str(N)
            if N not in self.pn or not self.pn[N].get("abf"):
                probs.append(f"N {N}: no ABF block")
            if int(N) >= 2:
                if not self.pn.get(N, {}).get("fr"):
                    probs.append(f"N {N}: no FR block")
                if N not in self.S["contrasts"]:
                    probs.append(f"N {N}: no contrast")
            for m, d in ((self.S.get("status") or {}).get(N, {}).get("methods") or {}).items():
                for k in ("missing", "running", "invalid"):
                    if d.get(k):
                        probs.append(f"N {N} {m}: {k} {len(d[k])}")
                if len(d.get("complete", [])) != d.get("planned"):
                    probs.append(f"N {N} {m}: {len(d.get('complete', []))}/{d.get('planned')} complete")
        for (N, ck), (a, b) in self.transient_n.items():
            if a != b or b != len(pl["seeds"]):
                probs.append(f"N {N} {ck}: per-run transient covers {a}/{b} paired seeds of {len(pl['seeds'])}")
        for (N, k), c in self.mean_contrasts.items():
            if c["n"] != len(pl["seeds"]):
                probs.append(f"N {N} {k}: mean contrast over {c['n']} of {len(pl['seeds'])} seeds")
        return probs


def build(results_root, n_boot=M.N_BOOT, systems=SYSTEMS, verbose=True):
    L = ["# Equal-budget ladders: appendix tables", "",
         f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by `scripts/equal_budget/report_tables.py` "
         "from the three production analyses ONLY: `results/equal_budget_v2/{gateway,lta_T300,lta_T150}/analysis/"
         "summary.json` (written by `scripts/equal_budget/analyze_ladder.py`) and, for the per-run maximum transient "
         "of section 4, its per-run curve cache `analysis/run_metrics/N<N>/s<seed>_<arm>.npz` (each cached curve is "
         "verified against the summary's per-seed Ibar and final value before use). Nothing is re-scored.", "",
         "Number style as in each system's `analysis/tables.md`: 4 significant digits, median [q25, q75] over seeds "
         "(linear quantiles, censored-aware), relative differences as signed percentages; censored tau is '> T_N' "
         "in physical time and '> 1' in budget fraction; '--' = not applicable (N = 1 has no FR arm). Every table "
         "caption names the `summary.json` key(s) it is read from (`per_N[N][arm][key]` holds n, median, q25, q75, "
         "mean and per_seed). Sign convention: G < 0 = FR better; transient improvement > 0 = FR better.", "",
         "Two definitions differ from the production tables.md and are labelled where used: the MEAN-based "
         "contrasts / best allocation of section 2 (ratio of seed means, computed here from `per_seed`) and the "
         "per-RUN transient of section 4 (computed here from the curve cache).", "",
         "Systems: " + "; ".join(f"{s['label']} (tables {s['tag']}.*)" for s in systems) + ". Sections per system: "
         "1 absolute values; 2 mean-based contrasts and best allocation; 3 tau; 4 maximum transient improvement; "
         "5 discovery / establishment / accumulated information."]
    status = {}
    for sysd in systems:
        p = os.path.join(results_root, sysd["dir"], "analysis", "summary.json")
        S, sha = load_summary(p)
        if S.get("schema") != "eqb_ladder_summary/2":
            raise SystemExit(f"{p}: unexpected schema {S.get('schema')!r}")
        R = Report(S, sysd, os.path.join(results_root, sysd["dir"], "analysis", "run_metrics"), n_boot)
        rel = os.path.relpath(p, ROOT) if p.startswith(ROOT) else p
        t0 = time.time()
        L += R.render(rel, sha)
        probs = R.completeness()
        status[sysd["dir"]] = probs
        if verbose:
            print(f"{sysd['dir']}: {len(R.Ns)} N, {len(R.cN)} contrasts, {time.time() - t0:.1f} s; "
                  f"{'COMPLETE' if not probs else 'INCOMPLETE: ' + '; '.join(probs)}", flush=True)
    L += ["", "## Completeness", ""]
    for d, probs in status.items():
        L.append(f"* `{d}`: " + ("complete: every planned N has both arms (ABF only at N = 1), every paired seed is "
                                 "in the mean contrasts and the per-run transient, the summary's status block reports "
                                 "nothing missing / running / invalid." if not probs else "**INCOMPLETE**: " + "; ".join(probs)))
    return "\n".join(L) + "\n", status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=os.path.join(ROOT, "results", "equal_budget_v2"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "equal_budget", "APPENDIX_TABLES.md"))
    ap.add_argument("--n-boot", type=int, default=M.N_BOOT)
    a = ap.parse_args(argv)
    txt, status = build(os.path.abspath(a.results_root), a.n_boot)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(txt)
    print(f"wrote {a.out}", flush=True)
    return 0 if not any(status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
