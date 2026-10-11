"""Experiment B estimators and decision rule (docs/final_campaign/PREREGISTRATION.md section 2.7, frozen).

Pure functions on per-seed values, unit-tested on censored fixtures (tests/test_final_expB_stats.py) before any
production data exist.  W values: seconds, +inf = censored (never reached persistently within the budget), NaN =
missing (a seed without a usable run; never produced by censoring).  eqb_metrics.boot_median_ci, paired_contrast and
the Exp III W1_RULE are NOT used for any W quantity (they drop non-finite values).

* medians are inf-aware: eqb_metrics.quantile_inf on the full vector including inf (a median whose upper
  interpolation neighbour is inf is inf);
* bootstrap: 10 000 resamples, default_rng(20261010); seeds resampled independently per arm, except the PAIRED arms
  (ABF 512 and FR 512) which share ONE index draw over their common seed labels; units (single arms and paired
  groups) draw their indices in sorted order of their names, so every result is reproducible.  A paired resample
  index that hits a seed missing in one arm (NaN) is skipped for that arm only;
* ratio rho(f, a) = median W_f / median W_a: finite/inf = 0, inf/finite = inf, inf/inf = undefined (also 0/0 and any
  NaN); undefined resamples are COUNTED and reported; percentiles are taken over the defined resamples; a contrast
  with > 2.5 % undefined resamples is 'unresolved';
* decision (per system and threshold): FR advantage iff some f in F has, for EVERY a in A, the upper end of the
  two-sided (1 - 0.05/|F|) percentile CI of rho(f, a) < 1; ABF advantage iff some a in A has, for every f in F, the
  lower end of the (1 - 0.05/|A|) CI > 1; otherwise unresolved (the two cannot both hold: for the same contrast the
  0.05/(2|A|) quantile is <= the 1 - 0.05/(2|F|) quantile);
* rho_best = min_f median W_f / min_a median W_a, both sides re-selected in every resample (descriptive; with its
  percentile CI and the fraction of resamples with rho_best < 1 -- not a probability);
* cross-N contrasts: Gehan-Wilcoxon two-sample test (inf ranked last, inf vs inf a tie; Mantel's permutational
  variance, normal p, plus a Monte-Carlo permutation p); ABF 512 vs FR 512 only: exact paired sign test;
* budget artefact: a contrast is 'not comparable (budget artefact)' if a censored arm (median W = inf) has a median
  completion wall time shorter than the comparator's median W;
* W* = the smallest of the arms' median completion wall times over uncontended runs, fixed at its observed value
  before the bootstrap (w_star).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "scripts", "equal_budget"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.append(_p)
import eqb_metrics as M  # noqa: E402

STATS_VERSION = "expB_stats/1"
N_BOOT = 10_000
BOOT_SEED = 20261010
ALPHA = 0.05
UNDEF_MAX = 0.025
N_PERM = 20_000


# ================================================================================================ medians
def _vec(x):
    return np.asarray(list(x.values()) if isinstance(x, dict) else x, dtype=float)


def median_inf(x):
    """inf-aware median of the full vector (inf = censored); NaN (missing) entries are not values."""
    return M.quantile_inf(_vec(x), 50)


def quantile_inf(x, q):
    return M.quantile_inf(_vec(x), q)


def row_quantile_inf(A, q=50.0):
    """Per-row inf-aware quantile (quantile_inf's interpolation) of a (B, n) array, NaN = missing (skipped); a row
    without any value gives NaN."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    B, n = A.shape
    out = np.full(B, np.nan)
    if n == 0:
        return out
    S = np.sort(A, axis=1)                      # finite < inf < NaN
    cnt = np.sum(~np.isnan(A), axis=1)
    ok = cnt > 0
    p = float(q) / 100.0 * (np.maximum(cnt, 1) - 1)
    lo = np.floor(p).astype(np.int64)
    hi = np.ceil(p).astype(np.int64)
    a = np.take_along_axis(S, lo[:, None], axis=1)[:, 0]
    b = np.take_along_axis(S, hi[:, None], axis=1)[:, 0]
    with np.errstate(invalid="ignore"):
        v = a + (b - a) * (p - lo)
    v = np.where(np.isinf(b), b, np.where(np.isinf(a), a, v))
    out[ok] = v[ok]
    return out


# ================================================================================================ bootstrap
def bootstrap_medians(values, pairs=(), n_boot=N_BOOT, seed=BOOT_SEED, q=50.0):
    """values: {arm: {seed: W}} (inf allowed, NaN = missing); pairs: iterable of tuples of arms sharing one index
    draw.  Returns ({arm: (n_boot,) bootstrap medians}, info).  Units draw in sorted order of their names."""
    arms = sorted(values)
    paired = {}
    for g in pairs:
        g = tuple(sorted(a for a in g if a in values))
        if len(g) >= 2:
            for a in g:
                if a in paired:
                    raise ValueError(f"arm {a} appears in two paired groups")
                paired[a] = g
    units = sorted({paired.get(a, (a,)) for a in arms}, key=lambda g: "+".join(g))
    rng = np.random.default_rng(seed)
    out, info = {}, {}
    for g in units:
        seeds = sorted(set().union(*[set(values[a]) for a in g]))
        n = len(seeds)
        if n == 0:
            for a in g:
                out[a] = np.full(int(n_boot), np.nan)
            continue
        idx = rng.integers(0, n, size=(int(n_boot), n))
        for a in g:
            x = np.array([values[a].get(s, np.nan) for s in seeds], dtype=float)
            out[a] = row_quantile_inf(x[idx], q)
            info[a] = dict(unit="+".join(g), n_seeds=int(np.sum(~np.isnan(x))), n_union=n, paired=len(g) > 1)
    return out, info


def boot_ci_median(values, n_boot=N_BOOT, seed=BOOT_SEED, level=0.95):
    """(median, lo, hi) of an inf-aware seed bootstrap of the median (inf kept, NaN = missing dropped)."""
    x = _vec(values)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    b = row_quantile_inf(x[rng.integers(0, x.size, size=(int(n_boot), x.size))])
    a = (1.0 - level) / 2.0
    return median_inf(x), M.quantile_inf(b, 100 * a), M.quantile_inf(b, 100 * (1 - a))


def boot_ci_fraction(flags, n_boot=N_BOOT, seed=BOOT_SEED, level=0.95):
    """(fraction, lo, hi) of booleans (e.g. runs reaching a threshold)."""
    x = np.asarray(_vec(flags), dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, x.size, size=(int(n_boot), x.size))].mean(axis=1)
    a = (1.0 - level) / 2.0
    return float(x.mean()), float(np.percentile(b, 100 * a)), float(np.percentile(b, 100 * (1 - a)))


def paired_diff_ci(a, b, n_boot=N_BOOT, seed=BOOT_SEED, level=0.95):
    """Median over common seeds of b[s] - a[s] (finite values only; e.g. FR's added wall time) with a seed bootstrap
    CI.  Returns dict(n, median, ci, per_seed)."""
    common = sorted(s for s in set(a) & set(b) if np.isfinite(a[s]) and np.isfinite(b[s]))
    d = np.array([b[s] - a[s] for s in common], dtype=float)
    if d.size == 0:
        return dict(n=0, median=math.nan, ci=[math.nan, math.nan], per_seed={})
    rng = np.random.default_rng(seed)
    bm = np.median(d[rng.integers(0, d.size, size=(int(n_boot), d.size))], axis=1)
    al = (1.0 - level) / 2.0
    return dict(n=int(d.size), median=float(np.median(d)),
                ci=[float(np.percentile(bm, 100 * al)), float(np.percentile(bm, 100 * (1 - al)))],
                per_seed={int(s): float(v) for s, v in zip(common, d)})


# ================================================================================================ ratios
def ratio(num, den):
    """Elementwise rho = num / den with the frozen conventions: finite/inf = 0, inf/finite = inf, inf/inf =
    undefined (NaN); 0/0 and any NaN input undefined; x/0 (x > 0 finite) = inf."""
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    n, d = np.broadcast_arrays(n, d)
    out = np.full(n.shape, np.nan)
    fn, fd = np.isfinite(n), np.isfinite(d)
    inn, ind = np.isposinf(n), np.isposinf(d)
    out = np.where(fn & ind, 0.0, out)
    out = np.where(inn & fd, np.inf, out)
    both = fn & fd
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d != 0, n / np.where(d != 0, d, 1.0), np.where(n > 0, np.inf, np.nan))
    out = np.where(both, q, out)
    return out if out.ndim else float(out)


def defined_quantile(r, q):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    return M.quantile_inf(r, q) if r.size else math.nan


def ratio_ci(r, level):
    """Two-sided percentile CI (over defined resamples) of a bootstrap ratio array."""
    a = (1.0 - float(level)) / 2.0
    return [defined_quantile(r, 100 * a), defined_quantile(r, 100 * (1 - a))]


def contrast(boot_f, boot_a, med_f, med_a, levels=(0.95,)):
    r = ratio(boot_f, boot_a)
    und = int(np.sum(np.isnan(r)))
    dfd = r[~np.isnan(r)]
    out = dict(point=ratio(med_f, med_a), n_boot=int(r.size), n_undefined=und, frac_undefined=und / max(r.size, 1),
               unresolved=bool(und / max(r.size, 1) > UNDEF_MAX),
               frac_below_1=float(np.mean(dfd < 1)) if dfd.size else math.nan)
    for lv in levels:
        out[f"ci{_lvtag(lv)}"] = ratio_ci(r, lv)
    return out, r


def _lvtag(lv):
    s = f"{100 * float(lv):.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


# ================================================================================================ decision
def iut_decision(boot, med, fr_arms, abf_arms, alpha=ALPHA):
    """The frozen intersection-union decision (module docstring).  boot / med: {arm: bootstrap medians / point
    median}.  Returns dict(decision, fr_advantage_arms, abf_advantage_arms, levels, contrasts {f|a: ...})."""
    F, A = list(fr_arms), list(abf_arms)
    if not F or not A:
        raise ValueError("need at least one FR arm and one ABF arm")
    lvF, lvA = 1.0 - alpha / len(F), 1.0 - alpha / len(A)
    table = {}
    for f in F:
        for a in A:
            c, r = contrast(boot[f], boot[a], med[f], med[a], levels=(0.95, lvF, lvA))
            c["upper_FR_level"] = defined_quantile(r, 100 * (1 - alpha / (2 * len(F))))
            c["lower_ABF_level"] = defined_quantile(r, 100 * (alpha / (2 * len(A))))
            c["fr_better"] = bool(not c["unresolved"] and c["upper_FR_level"] < 1)
            c["abf_better"] = bool(not c["unresolved"] and c["lower_ABF_level"] > 1)
            table[f"{f}|{a}"] = c
    fr_adv = [f for f in F if all(table[f"{f}|{a}"]["fr_better"] for a in A)]
    abf_adv = [a for a in A if all(table[f"{f}|{a}"]["abf_better"] for f in F)]
    if fr_adv and abf_adv:                                   # impossible by construction (docstring)
        raise AssertionError("FR and ABF advantage both hold: quantile ordering violated")
    decision = "FR advantage" if fr_adv else ("ABF advantage" if abf_adv else "unresolved")
    return dict(decision=decision, fr_advantage_arms=fr_adv, abf_advantage_arms=abf_adv,
                levels=dict(fr_side=lvF, abf_side=lvA, alpha=alpha, n_fr=len(F), n_abf=len(A)),
                unresolved_contrasts=[k for k, v in table.items() if v["unresolved"]], contrasts=table)


def rho_best(boot, med, fr_arms, abf_arms, level=0.95):
    """Descriptive rho_best with re-selection of the best arm on both sides in every resample."""
    F, A = list(fr_arms), list(abf_arms)
    mf = min(med[f] for f in F)
    ma = min(med[a] for a in A)
    bf = np.stack([boot[f] for f in F], axis=1)
    ba = np.stack([boot[a] for a in A], axis=1)
    with np.errstate(invalid="ignore"):
        minf = np.where(np.all(np.isnan(bf), axis=1), np.nan, np.nanmin(np.where(np.isnan(bf), np.inf, bf), axis=1))
        mina = np.where(np.all(np.isnan(ba), axis=1), np.nan, np.nanmin(np.where(np.isnan(ba), np.inf, ba), axis=1))
    r = ratio(minf, mina)
    dfd = r[~np.isnan(r)]

    def freq(Bm, names):
        Bm = np.where(np.isnan(Bm), np.inf, Bm)
        m = Bm.min(axis=1)
        fin = np.isfinite(m)
        tied = (Bm == m[:, None]) & fin[:, None]
        cr = np.where(fin[:, None], tied / np.maximum(tied.sum(axis=1), 1)[:, None], 0.0)
        return {n: float(np.mean(cr[:, i])) for i, n in enumerate(names)}
    return dict(point=ratio(mf, ma), best_fr_point=F[int(np.argmin([med[f] for f in F]))],
                best_abf_point=A[int(np.argmin([med[a] for a in A]))], ci=ratio_ci(r, level), level=level,
                frac_below_1=float(np.mean(dfd < 1)) if dfd.size else math.nan,
                n_undefined=int(np.sum(np.isnan(r))), best_fr_freq=freq(bf, F), best_abf_freq=freq(ba, A),
                note="descriptive (winner's curse not removed); frac_below_1 is not a probability")


# ================================================================================================ tests
def gehan_test(x, y, n_perm=N_PERM, seed=BOOT_SEED):
    """Gehan-Wilcoxon two-sample test of W samples x, y with inf ranked last (inf vs inf tied).  G = #(x > y) -
    #(x < y); Mantel's permutational variance n m sum(u_k^2) / ((n+m)(n+m-1)) with u_k = #(z_k > z_l) - #(z_k <
    z_l) over the pooled sample; normal two-sided p and a Monte-Carlo permutation p.  Also P(x < y) + P(tie)/2."""
    from scipy.stats import norm
    x = _vec(x)
    y = _vec(y)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    n, m = x.size, y.size
    if n == 0 or m == 0:
        return dict(n_x=int(n), n_y=int(m), G=math.nan, z=math.nan, p_normal=math.nan, p_perm=math.nan,
                    prob_x_less=math.nan)
    z = np.concatenate([x, y])
    gt = (z[:, None] > z[None, :]).astype(np.int64)
    lt = (z[:, None] < z[None, :]).astype(np.int64)
    u = (gt - lt).sum(axis=1)
    G = float(u[:n].sum())
    N_ = n + m
    var = n * m * float(np.sum(u.astype(float) ** 2)) / (N_ * (N_ - 1)) if N_ > 1 else 0.0
    zst = G / math.sqrt(var) if var > 0 else 0.0
    p_norm = float(2.0 * (1.0 - norm.cdf(abs(zst)))) if var > 0 else 1.0
    rng = np.random.default_rng(seed)
    if var > 0 and n_perm:
        perms = rng.permuted(np.broadcast_to(np.arange(N_), (int(n_perm), N_)), axis=1)
        Gp = u[perms[:, :n]].sum(axis=1)
        p_perm = (1.0 + float(np.sum(np.abs(Gp) >= abs(G) - 1e-9))) / (1.0 + n_perm)
    else:
        p_perm = 1.0
    xl = float(np.sum(x[:, None] < y[None, :]))
    tie = float(np.sum(x[:, None] == y[None, :]))
    return dict(n_x=int(n), n_y=int(m), G=G, var=var, z=zst, p_normal=min(1.0, p_norm), p_perm=min(1.0, p_perm),
                prob_x_less=(xl + 0.5 * tie) / (n * m), n_perm=int(n_perm),
                convention="inf ranked last (censored), inf vs inf tied; G > 0: x tends to be slower")


def paired_sign_test(abf, fr):
    """Exact two-sided sign test on the common seeds: FR wins iff W_FR < W_ABF (finite beats inf; inf vs inf and
    equal values tie)."""
    common = sorted(s for s in set(abf) & set(fr) if not (np.isnan(abf[s]) or np.isnan(fr[s])))
    a = np.array([abf[s] for s in common], dtype=float)
    f = np.array([fr[s] for s in common], dtype=float)
    wins, losses = int(np.sum(f < a)), int(np.sum(f > a))
    return dict(n_common=len(common), fr_wins=wins, fr_losses=losses, ties=len(common) - wins - losses,
                p=M.sign_test_p(wins, losses))


# ================================================================================================ budget, W*
def budget_artefact(W_x, C_x, W_y, C_y):
    """'not comparable (budget artefact)' iff a censored arm (median W = inf) completed (median completion wall)
    sooner than the comparator's median W.  Returns dict(label, x_limited, y_limited, details)."""
    def side(W, C, Wo):
        mW = median_inf(W)
        if not math.isinf(mW):
            return False, None
        mc = median_inf(C)
        mo = median_inf(Wo)
        return bool(mc < mo), dict(median_completion=mc, comparator_median_W=mo)
    xl, xd = side(W_x, C_x, W_y)
    yl, yd = side(W_y, C_y, W_x)

    def partial(W, C, Wo):
        mo = median_inf(Wo)
        Wv, Cv = _vec(W), _vec(C)
        cens = np.isinf(Wv)
        return float(np.mean(cens & (Cv < mo))) if Wv.size else math.nan
    return dict(label="not comparable (budget artefact)" if (xl or yl) else "comparable", x_limited=xl,
                y_limited=yl, x_detail=xd, y_detail=yd,
                frac_runs_budget_limited_x=partial(W_x, C_x, W_y), frac_runs_budget_limited_y=partial(W_y, C_y, W_x))


def w_star(completion_by_arm):
    """W* = min over arms of the median completion wall time (pass uncontended runs only); fixed before any
    bootstrap."""
    meds = {a: median_inf(v) for a, v in completion_by_arm.items() if len(_vec(v))}
    if not meds:
        raise ValueError("no completion times")
    a = min(meds, key=lambda k: meds[k])
    return float(meds[a]), a, meds


# ================================================================================================ costs
def effective_parallelism(c_serial, walker_steps, wall, threads):
    """Serial-equivalent speed-up = validated-serial wall per walker-step x walker-steps / wall; efficiency =
    speed-up / threads."""
    s = float(c_serial) * float(walker_steps) / float(wall) if (wall and math.isfinite(wall)) else math.nan
    return dict(speedup=s, efficiency=s / float(threads) if threads else math.nan)


# ================================================================================================ full report
def compare_arms(values, fr_arms, abf_arms, pairs=(("abf512", "fr512"),), completion=None, n_boot=N_BOOT,
                 seed=BOOT_SEED, alpha=ALPHA, gehan_perm=N_PERM):
    """Everything of prereg 2.7 for one endpoint: per-arm medians (inf-aware) and bootstrap CIs, the IUT decision,
    rho_best, the Gehan-Wilcoxon tests of the cross-N (unpaired) FR/ABF contrasts, the paired sign test, and the
    budget-artefact label of every contrast (needs completion = {arm: {seed: completion wall}})."""
    missing = [a for a in list(fr_arms) + list(abf_arms) if not values.get(a)]
    arms = [a for a in list(fr_arms) + list(abf_arms) if values.get(a)]
    vals = {a: values[a] for a in arms}
    for a, v in vals.items():
        if any(isinstance(x, float) and math.isnan(x) for x in v.values()):
            raise ValueError(f"arm {a}: NaN W values (missing runs must be absent, censored = inf)")
    boot, binfo = bootstrap_medians(vals, pairs=pairs, n_boot=n_boot, seed=seed)
    med = {a: median_inf(vals[a]) for a in arms}
    per_arm = {}
    for a in arms:
        x = _vec(vals[a])
        per_arm[a] = dict(n=int(x.size), n_censored=int(np.sum(np.isinf(x))), median=med[a],
                          ci95=[M.quantile_inf(boot[a][~np.isnan(boot[a])], 2.5),
                                M.quantile_inf(boot[a][~np.isnan(boot[a])], 97.5)],
                          frac_reached=float(np.mean(np.isfinite(x))) if x.size else math.nan,
                          boot_unit=binfo.get(a, {}).get("unit"))
    F = [f for f in fr_arms if f in vals]
    A = [a for a in abf_arms if a in vals]
    out = dict(per_arm=per_arm, n_boot=int(n_boot), seed=int(seed), pairs=[list(p) for p in pairs],
               arms_without_data=missing)
    if F and A:
        out["decision"] = iut_decision(boot, med, F, A, alpha)
        if missing:                  # the IUT is over ALL arms: an arm without usable runs blocks any advantage
            out["decision"].update(decision="unresolved", fr_advantage_arms=[], abf_advantage_arms=[],
                                   reason=f"arms without usable runs: {missing} (the decision needs every arm)")
        out["rho_best"] = rho_best(boot, med, F, A)
        paired_set = {tuple(sorted(p)) for p in pairs}
        tests = {}
        for f in F:
            for a in A:
                key = f"{f}|{a}"
                if tuple(sorted((f, a))) in paired_set:
                    tests[key] = dict(kind="paired sign test", **paired_sign_test(vals[a], vals[f]))
                else:
                    tests[key] = dict(kind="Gehan-Wilcoxon (unpaired)", **gehan_test(vals[f], vals[a],
                                                                                    n_perm=gehan_perm, seed=seed))
                if completion is not None:
                    tests[key]["budget"] = budget_artefact(list(vals[f].values()),
                                                           [completion[f][s] for s in vals[f]],
                                                           list(vals[a].values()),
                                                           [completion[a][s] for s in vals[a]])
                    out["decision"]["contrasts"][key]["budget_label"] = tests[key]["budget"]["label"]
        out["tests"] = tests
    out["_boot"] = boot
    return out
