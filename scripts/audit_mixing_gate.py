#!/usr/bin/env python3
"""mFR mechanism audit, step 3: does an ABF-ONLY mixing indicator predict the
sign and magnitude of the mFR gain across systems?

PURE READ-ONLY RE-ANALYSIS.  Reads existing artifacts under ``results/`` and
writes only under ``results/mfr_mechanism_audit/mixing_gate/``.  It launches no
dynamics, touches no GPU, and modifies nothing that already exists.

Design constraints (from docs/MFR_MECHANISM_AUDIT_PLAN.md step 3)
-----------------------------------------------------------------
*  Gate INPUTS may only use quantities computable from an **ABF baseline run**.
   No mFR result, no ancestor ESS of an FR run, and nothing that requires the
   reference free energy, may enter as a feature.  The mFR gain is the
   prediction TARGET.
*  Raw round-trips per replica are not comparable across systems, so every
   indicator below is standardised (dimensionless, or divided by the horizon).
*  Validation is leave-ONE-SYSTEM-out.  Direction and threshold of the gate are
   fitted on the training systems only; the held-out system is never used to
   pick either.
*  Two incumbent predictors are scored with the identical protocol:
     INC-A  measured ABF baseline error (needs the reference -> NOT deployable)
     INC-B  nominal low-support fraction of the ABF run (deployable)

Cells and pooling filters (stated explicitly; see the printed manifest)
----------------------------------------------------------------------
wca              results/wca_phase_diagram/production/raw, stage='production',
                 methods abf vs fr_estimated (fr_rate=0.1, fr_every=5,
                 max_event_fraction=0.02, target_ema_rate=0.005), 4 seeds/cell.
wca_rep          results/wca_representative/raw, stage='representative', same FR
                 spec, 10 seeds/cell.  Used only as a robustness variant; it is
                 the SAME physical system as `wca` and is never treated as an
                 independent LOSO fold.
alkane_torsion   results/alkanes/production/raw (stages b1,b2,p1), phi CV,
                 fr_estimated fr_rate=0.02.
alkane_dist      results/alkanes_cv_extension/r15_methods/raw, stage='production'
                 only (tuning/opes_tuning stages excluded), R15 distance CV.
alkane_torus     results/alkanes_cv_extension/2d_methods/raw, stages
                 'production' and 'control', 2-D torsion torus CV.
edb              results/entropy_dominant_bottleneck/sweep_20260614_015145/raw,
                 abf vs fr_estimated.  PHYSICS cells = the phi sweep at the
                 locked FR rate gamma=15 only.  The gamma rate sweep is split
                 into a separate `rate_confound.csv`: gamma is the FR birth-death
                 intensity and does not touch the ABF run, so those cells share
                 one ABF indicator value while their gains differ -- pooling them
                 into the main table would mix gentle and aggressive FR arms
                 under one label.
edb_slow         results/entropy_dominant_bottleneck_slow_transverse/
                 pilot_20260616_225737 ONLY (ORPHAN artifact; four pilots exist
                 with overlapping seed ids and differing sha256, so they are NOT
                 pooled).  Flagged orphan in every output.
eb               results/entropic_bottleneck/raw (stages 2/3/4, deduplicated).
                 PHYSICS cells = omega_in and beta sweeps at the locked FR rate
                 gamma=15; the gamma sweep goes to `rate_confound.csv` for the
                 same reason as edb.
                 NOTE: this system stores no transition/occupancy time series, so
                 the transition-based indicators CANNOT be built for it.
toy2d            results/two_dim_xi_x/production_gpu summary CSVs.

Usage
-----
    python scripts/audit_mixing_gate.py            # writes CSV/JSON outputs
    python scripts/audit_mixing_gate.py --quiet
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "mfr_mechanism_audit", "mixing_gate")
EPS = 1e-12

# ---------------------------------------------------------------------------
# PRE-DECLARED analysis choices (fixed before looking at any performance number)
# ---------------------------------------------------------------------------
PRACTICAL_MARGIN_PCT = 5.0     # |gain| <= 5 %  => practically equivalent
N_BOOT = 4000                  # bootstrap resamples
RNG_SEED = 20260721
GATE_PRIMARY = "log10_n_relax"  # designated candidate gate variable
INCUMBENT_A = "abf_norm_l2_f"   # measured ABF baseline error (reference-based)
INCUMBENT_B = "low_support_fraction"  # nominal low-support fraction (ABF-only)


# ===========================================================================
# generic helpers
# ===========================================================================
def _v(d, k, default=None):
    """npz/dict tolerant getter; unwraps 0-d arrays."""
    keys = d.files if hasattr(d, "files") else d
    if k not in keys:
        return default
    x = d[k]
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode()
    return x


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def rankdata(a):
    """Average-rank transform (avoids a scipy dependency)."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[m]), rankdata(y[m])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry / den) if den > 0 else float("nan")


def auc_score(scores, labels):
    """AUC via the Mann-Whitney U identity, ties counted as 0.5."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    m = np.isfinite(scores)
    scores, labels = scores[m], labels[m]
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(scores)
    rpos = r[labels == 1].sum()
    return float((rpos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def boot_ci(fn, n, rng, n_boot=N_BOOT, alpha=0.05, groups=None):
    """Percentile bootstrap CI of ``fn(idx)`` over ``n`` items.

    If ``groups`` is given, resampling is stratified within group.
    """
    vals = []
    if groups is not None:
        groups = np.asarray(groups)
        gidx = [np.where(groups == g)[0] for g in np.unique(groups)]
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in gidx])
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 20:
        return float("nan"), float("nan")
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def two_state_relax(cross_per_replica, p_a, p_b):
    """Standardised mixing indicator: number of 2-state relaxation times per
    replica over the whole run.

    For a lumped 2-state Markov chain with stationary probabilities (p_a, p_b)
    and total (both-direction) boundary crossings ``C`` per replica over horizon
    T, the per-direction rate estimates are k_ab = (C/2)/(T p_a) and
    k_ba = (C/2)/(T p_b).  The relaxation rate is lambda2 = k_ab + k_ba, so

        n_relax = lambda2 * T = (C/2) * (1/p_a + 1/p_b) = (C/2) / (p_a p_b).

    T cancels: the indicator is dimensionless and horizon-normalised, which is
    exactly what is needed to compare systems with different units and horizons.
    """
    if not np.isfinite(cross_per_replica) or not np.isfinite(p_a) or not np.isfinite(p_b):
        return float("nan")
    pa = min(max(p_a, EPS), 1 - EPS)
    pb = min(max(p_b, EPS), 1 - EPS)
    return float(0.5 * cross_per_replica / (pa * pb))


def coverage_stats(support, mask=None):
    """Reference-free support diagnostics from an ABF density / count profile.

    Returns (cv_coverage, low_support_fraction, support_min_over_median):
      cv_coverage        fraction of evaluated bins with support > 1e-3 * max
      low_support_fraction  fraction of evaluated bins with support < 0.25 *
                         median-of-positive-bins  (matches the project's
                         `dist_support_metrics` definition in
                         src/alkanes/metrics_cv.py).  This is the incumbent
                         "nominal low-support fraction"; it saturates at 0 in
                         several systems, so a graded companion is also returned.
      support_min_over_median  min bin support / median bin support -- a graded,
                         reference-free version of the same idea (analogous to
                         `abf_neff_transition_min_ratio` in
                         scripts/analyze_wca_starvation.py).
    """
    s = np.asarray(support, float)
    if mask is not None:
        s = s[mask]
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan"), float("nan"), float("nan")
    mx = float(np.max(s))
    cov = float(np.mean(s > 1e-3 * mx)) if mx > 0 else 0.0
    pos = s[s > 0]
    med = float(np.median(pos)) if pos.size else 0.0
    low = float(np.mean(s < 0.25 * med)) if med > 0 else 1.0
    ratio = float(np.min(s) / med) if med > 0 else float("nan")
    return cov, low, ratio


def profile_range(profile, mask=None):
    p = np.asarray(profile, float)
    if mask is not None:
        p = p[mask]
    p = p[np.isfinite(p)]
    return float(p.max() - p.min()) if p.size else float("nan")


def split_prob(grid, dens, cut):
    """Probability mass below / above ``cut`` for a density on ``grid``."""
    g = np.asarray(grid, float)
    d = np.clip(np.asarray(dens, float), 0, None)
    lo = g <= cut
    hi = g > cut
    if lo.sum() < 2 or hi.sum() < 2:
        return float("nan"), float("nan")
    a = float(np.trapezoid(d[lo], g[lo]))
    b = float(np.trapezoid(d[hi], g[hi]))
    tot = a + b
    if tot <= 0:
        return float("nan"), float("nan")
    return a / tot, b / tot


def matrix_spectral_gap(counts):
    """Second eigenvalue modulus of the row-normalised state-transition matrix.

    ``counts[i, j]`` = number of observed i -> j transitions.  Returns
    (lam2, gap) with gap = 1 - lam2.  Rows with no mass are dropped.
    """
    C = np.asarray(counts, float)
    keep = C.sum(1) > 0
    C = C[np.ix_(keep, keep)]
    if C.shape[0] < 2:
        return float("nan"), float("nan")
    P = C / C.sum(1, keepdims=True)
    ev = np.linalg.eigvals(P)
    ev = np.sort(np.abs(ev))[::-1]
    lam2 = float(ev[1])
    return lam2, float(1.0 - lam2)


# ===========================================================================
# per-run record
# ===========================================================================
def blank_run(**kw):
    r = dict(system="", cell="", method="", seed=-1, stage="",
             n_replicas=np.nan, n_steps=np.nan, dt=np.nan, t_phys=np.nan,
             l2_f=np.nan, int_l2_f=np.nan,
             n_cross=np.nan, n_roundtrips=np.nan,
             p_a=np.nan, p_b=np.nan,
             cv_coverage=np.nan, low_support_fraction=np.nan,
             support_min_over_median=np.nan,
             f_range=np.nan,
             spectral_gap_matrix=np.nan,
             n_states_visited_frac=np.nan,
             fpt_frac=np.nan,
             cross_series=None, cross_series_t=None,
             occ_series=None, occ_series_t=None,
             sweep="physics", orphan=False)
    r.update(kw)
    return r


def finalize_run(r):
    """Derive the standardised indicators from the raw per-run fields."""
    N = r["n_replicas"]
    T = r["t_phys"]
    cpr = r["n_cross"] / N if (np.isfinite(r["n_cross"]) and N and np.isfinite(N)) else np.nan
    r["cross_per_replica"] = cpr
    r["log10_cross_per_replica"] = math.log10(cpr + 1.0) if np.isfinite(cpr) else np.nan
    r["cross_rate_per_time"] = cpr / T if (np.isfinite(cpr) and T and np.isfinite(T)) else np.nan
    rt = r["n_roundtrips"] / N if (np.isfinite(r["n_roundtrips"]) and N and np.isfinite(N)) else np.nan
    r["roundtrips_per_replica"] = rt
    r["roundtrip_rate_per_time"] = rt / T if (np.isfinite(rt) and T and np.isfinite(T)) else np.nan
    nr = two_state_relax(cpr, r["p_a"], r["p_b"])
    r["n_relax"] = nr
    r["log10_n_relax"] = math.log10(nr + 1.0) if np.isfinite(nr) else np.nan
    pa, pb = r["p_a"], r["p_b"]
    r["min_state_occupancy"] = (min(pa, pb) if (np.isfinite(pa) and np.isfinite(pb))
                                else np.nan)
    return r


# ===========================================================================
# loaders (one per system)
# ===========================================================================
def load_wca(raw_dir, system, stage_filter, fr_name="fr_estimated"):
    """WCA dimer.  Crossings = per-step sign changes of z about the barrier
    midpoint 0.5 (see src/wca_abffr_core.py ~line 906); round trips =
    sum_r min(#c->s, #s->c).  Occupancies are taken from the cumulative
    kernel-weighted ABF counts (`final_eff_counts`) split at z = 0.5, which is
    reference-free and averages over the entire run."""
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        stage = str(_v(d, "stage"))
        if stage != stage_filter:
            continue
        method = str(_v(d, "method"))
        if method not in ("abf", fr_name):
            continue
        spec = json.loads(str(_v(d, "spec_json", "{}")))
        grid = np.asarray(_v(d, "grid"), float)
        emask = (grid >= -0.1) & (grid <= 1.1)
        n_rep = float(_v(d, "n_replicas"))
        n_steps = float(_v(d, "n_steps"))
        dt = float(spec.get("dt", 2.0e-3))
        eff = np.asarray(_v(d, "final_eff_counts", np.full_like(grid, np.nan)), float)
        pa, pb = split_prob(grid[emask], eff[emask], 0.5)
        cov, low, smr = coverage_stats(eff, emask)
        cell = (f"b{_v(d,'beta'):g}_h{_v(d,'h'):g}_w{_v(d,'w'):g}"
                f"_n{int(_v(d,'n_dim'))}_a{_v(d,'a'):g}")
        # ensemble occupancy series -> first-passage-to-far-basin fraction
        times = np.asarray(_v(d, "times", []), float)
        fstr = np.asarray(_v(d, "frac_stretched", []), float)
        fcom = np.asarray(_v(d, "frac_compact", []), float)
        r = blank_run(system=system, cell=cell, method=("abf" if method == "abf" else "fr"),
                      seed=int(_v(d, "seed")), stage=stage,
                      n_replicas=n_rep, n_steps=n_steps, dt=dt, t_phys=n_steps * dt,
                      l2_f=float(_v(d, "l2_f", np.nan)),
                      int_l2_f=float(_v(d, "integrated_l2_f", np.nan)),
                      n_cross=float(_v(d, "n_barrier_crossings", np.nan)),
                      n_roundtrips=float(_v(d, "n_round_trips", np.nan)),
                      p_a=pa, p_b=pb, cv_coverage=cov, low_support_fraction=low,
                      support_min_over_median=smr,
                      f_range=profile_range(_v(d, "ref_free_energy"), emask),
                      occ_series=(fstr if fstr.size else None),
                      occ_series_t=(times if times.size else None))
        # first passage: first save time where the minority (final) basin is
        # occupied at >= 10 % of its final level, divided by the horizon.
        if fstr.size and times.size == fstr.size:
            far = fstr if fstr[-1] <= fcom[-1] else fcom
            thr = 0.1 * far[-1]
            hit = np.where(far >= thr)[0]
            if hit.size:
                r["fpt_frac"] = float(times[hit[0]] / max(times[-1], EPS))
        # 3-region visitation
        vis = 0
        for arr in (fcom, np.asarray(_v(d, "frac_transition", []), float), fstr):
            if arr.size and np.nanmax(arr) > 0:
                vis += 1
        r["n_states_visited_frac"] = vis / 3.0
        rows.append(finalize_run(r))
    return rows


def load_alkane_torsion(raw_dir, system):
    """Butane/pentane torsion CV.  n_transitions = basin-index changes (T,G+,G-)
    summed over replicas; n_round_trips = T->G->T completions.  Lumped 2-state
    occupancies use the final torsion marginal split at |phi| = 61.6 deg."""
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        method = str(_v(d, "method"))
        if method not in ("abf", "fr_estimated"):
            continue
        spec = json.loads(str(_v(d, "spec_json", "{}")))
        cell = (f"{_v(d,'molecule')}_b{float(_v(d,'beta')):g}_{_v(d,'init_mode')}"
                f"_T{int(_v(d,'n_steps'))}")
        grid = np.asarray(_v(d, "grid"), float)
        ref_F = np.asarray(_v(d, "ref_F"), float)
        n_rep = float(_v(d, "n_replicas"))
        n_steps = float(_v(d, "n_steps"))
        dt = float(spec.get("dt", 5.0e-4))
        barrier = math.radians(61.6)
        seeds = np.asarray(_v(d, "seeds"), int)
        p_hats = np.asarray(_v(d, "final_p_hat"), float)
        per_seed = json.loads(str(_v(d, "per_seed")))
        for k, ps in enumerate(per_seed):
            ph = p_hats[k]
            inside = np.abs(grid) < barrier
            a = float(np.trapezoid(ph[inside], grid[inside]))
            tot = float(np.trapezoid(ph, grid))
            pa = a / tot if tot > 0 else np.nan
            cov, low, smr = coverage_stats(ph)
            r = blank_run(system=system, cell=cell,
                          method=("abf" if method == "abf" else "fr"),
                          seed=int(ps["seed"]), stage=str(_v(d, "stage")),
                          n_replicas=n_rep, n_steps=n_steps, dt=dt,
                          t_phys=n_steps * dt,
                          l2_f=float(ps["final_l2_F"]),
                          int_l2_f=float(ps["integrated_l2_F"]),
                          n_cross=float(ps["n_transitions"]),
                          n_roundtrips=float(ps["n_round_trips"]),
                          p_a=pa, p_b=(1 - pa) if np.isfinite(pa) else np.nan,
                          cv_coverage=cov, low_support_fraction=low,
                      support_min_over_median=smr,
                          f_range=profile_range(ref_F))
            occ = [ps.get("final_frac_T"), ps.get("final_frac_Gp"), ps.get("final_frac_Gm")]
            occ = [o for o in occ if o is not None]
            r["n_states_visited_frac"] = (sum(o > 0 for o in occ) / 3.0) if occ else np.nan
            rows.append(finalize_run(r))
        _ = seeds
    return rows


def load_alkane_dist(raw_dir, system, stages=("production",)):
    """Pentane R15 distance CV.  3 extension states (compact / intermediate /
    extended) by R terciles; n_transitions counts all state changes, round trips
    are compact -> ... -> extended completions.  Lumped occupancies use the
    extreme terciles of the final R marginal, renormalised."""
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        if str(_v(d, "stage")) not in stages:
            continue
        method = str(_v(d, "method"))
        if method not in ("abf", "fr_estimated"):
            continue
        spec = json.loads(str(_v(d, "spec_json", "{}")))
        R_lo, R_hi = float(_v(d, "R_lo")), float(_v(d, "R_hi"))
        c1 = R_lo + (R_hi - R_lo) / 3.0
        c2 = R_lo + 2.0 * (R_hi - R_lo) / 3.0
        grid = np.asarray(_v(d, "grid"), float)
        n_rep = float(_v(d, "n_replicas"))
        n_steps = float(_v(d, "n_steps"))
        dt = float(spec.get("dt", 5.0e-4))
        p_hats = np.asarray(_v(d, "final_p_hat"), float)
        effs = np.asarray(_v(d, "final_eff_counts"), float)
        per_seed = json.loads(str(_v(d, "per_seed")))
        cell = (f"pentaneR15_b{float(_v(d,'beta')):g}_{_v(d,'init_mode')}"
                f"_T{int(n_steps)}")
        for k, ps in enumerate(per_seed):
            ph = p_hats[k]
            lo = grid <= c1
            hi = grid >= c2
            a = float(np.trapezoid(ph[lo], grid[lo])) if lo.sum() > 1 else np.nan
            b = float(np.trapezoid(ph[hi], grid[hi])) if hi.sum() > 1 else np.nan
            pa = a / (a + b) if (np.isfinite(a) and np.isfinite(b) and a + b > 0) else np.nan
            cov, _low, smr = coverage_stats(effs[k])
            r = blank_run(system=system, cell=cell,
                          method=("abf" if method == "abf" else "fr"),
                          seed=int(ps["seed"]), stage=str(_v(d, "stage")),
                          n_replicas=n_rep, n_steps=n_steps, dt=dt,
                          t_phys=n_steps * dt,
                          l2_f=float(ps["final_l2_F"]),
                          int_l2_f=float(ps["integrated_l2_F"]),
                          n_cross=float(ps["n_transitions"]),
                          n_roundtrips=float(ps["n_round_trips"]),
                          p_a=pa, p_b=(1 - pa) if np.isfinite(pa) else np.nan,
                          cv_coverage=cov,
                          low_support_fraction=float(ps.get("low_support_fraction", np.nan)),
                          support_min_over_median=smr,
                          f_range=float(_v(d, "F_range_thermal", np.nan)))
            fd = [ps.get("fd_compact"), ps.get("fd_intermediate"), ps.get("fd_extended")]
            fd = [x for x in fd if x is not None]
            if fd:
                r["n_states_visited_frac"] = sum(x >= 0 for x in fd) / 3.0
                if all(x >= 0 for x in fd):
                    r["fpt_frac"] = float(max(fd)) / n_steps
            rows.append(finalize_run(r))
    return rows


def load_alkane_torus(raw_dir, system, stages=("production", "control")):
    """Pentane 2-D torsion torus (9 basins).  A genuine 9-state transition-count
    matrix is stored per seed, so the spectral-gap proxy is exact here."""
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        if str(_v(d, "stage")) not in stages:
            continue
        method = str(_v(d, "method"))
        if method not in ("abf", "fr_estimated"):
            continue
        spec = json.loads(str(_v(d, "spec_json", "{}")))
        n_rep = float(_v(d, "n_replicas"))
        n_steps = float(_v(d, "n_steps"))
        dt = float(spec.get("dt", 5.0e-4))
        per_seed = json.loads(str(_v(d, "per_seed")))
        tm = np.asarray(_v(d, "trans_matrix"), float)
        fdisc = np.asarray(_v(d, "first_discovery"), float)
        jh = np.asarray(_v(d, "joint_hist"), float)
        cell = (f"pentane2D_b{float(_v(d,'beta')):g}_{_v(d,'init_mode')}"
                f"_T{int(n_steps)}")
        for k, ps in enumerate(per_seed):
            occ = None
            if tm.ndim == 3 and k < tm.shape[0]:
                occ = tm[k].sum(1)
                occ = occ / max(occ.sum(), EPS)
            pa = pb = np.nan
            if occ is not None and occ.size >= 2:
                srt = np.sort(occ)[::-1]
                pa = float(srt[0] / max(srt[0] + srt[1], EPS))
                pb = 1.0 - pa
            lam2, gap = (matrix_spectral_gap(tm[k]) if tm.ndim == 3 and k < tm.shape[0]
                         else (np.nan, np.nan))
            cov, low, smr = (coverage_stats(jh[k].ravel())
                             if jh.ndim == 3 and k < jh.shape[0]
                             else (np.nan, np.nan, np.nan))
            r = blank_run(system=system, cell=cell,
                          method=("abf" if method == "abf" else "fr"),
                          seed=int(ps["seed"]), stage=str(_v(d, "stage")),
                          n_replicas=n_rep, n_steps=n_steps, dt=dt,
                          t_phys=n_steps * dt,
                          l2_f=float(ps["final_l2_F"]),
                          int_l2_f=float(ps["integrated_l2_F"]),
                          n_cross=float(ps["n_transitions"]),
                          n_roundtrips=float(ps["n_round_trips"]),
                          p_a=pa, p_b=pb, cv_coverage=cov, low_support_fraction=low,
                      support_min_over_median=smr,
                          f_range=float(_v(d, "F_range_thermal", np.nan)),
                          spectral_gap_matrix=gap)
            if ps.get("n_basins_visited") is not None:
                r["n_states_visited_frac"] = float(ps["n_basins_visited"]) / 9.0
            if fdisc.ndim == 2 and k < fdisc.shape[0]:
                fk = fdisc[k]
                if np.all(fk >= 0):
                    r["fpt_frac"] = float(np.max(fk)) / n_steps
            rows.append(finalize_run(r))
    return rows


def _load_edb_like(paths, system, cell_of, orphan=False, locked_gamma=None):
    """Entropy-dominant bottleneck family (edb / edb_slow).

    ``final_cross`` counts genuine Langevin sign changes of x about x = 0,
    accumulated on the PROPOSAL before any FR resampling (src/edb_abffr_core.py
    ~line 623), so it is directly comparable between abf and FR arms.  Lumped
    occupancies come from the sampled marginal p_hat split at x = 0.
    ``cross_t`` is a cumulative time series -> early-window indicator available.
    """
    rows = []
    seen = set()
    for path in paths:
        d = np.load(path, allow_pickle=True)
        method = str(_v(d, "method"))
        if method not in ("abf", "fr_estimated"):
            continue
        key = (cell_of(d), method, int(_v(d, "seed")))
        if key in seen:          # the locked-rate cell recurs in both sub-sweeps
            continue
        seen.add(key)
        n_rep = float(_v(d, "cfg__N"))
        n_steps = float(_v(d, "cfg__n_steps"))
        dt = float(_v(d, "cfg__dt"))
        grid = np.asarray(_v(d, "x_grid"), float)
        ph = np.asarray(_v(d, "p_hat"), float)
        emask = (grid >= -1.5) & (grid <= 1.5)
        pa, pb = split_prob(grid[emask], ph[emask], 0.0)
        cov, low, smr = coverage_stats(ph, emask)
        ct = np.asarray(_v(d, "cross_t", []), float)
        tt = np.asarray(_v(d, "t", []), float)
        r = blank_run(system=system, cell=cell_of(d),
                      method=("abf" if method == "abf" else "fr"),
                      seed=int(_v(d, "seed")), stage="main",
                      n_replicas=n_rep, n_steps=n_steps, dt=dt, t_phys=n_steps * dt,
                      l2_f=float(_v(d, "final_l2_f", np.nan)),
                      int_l2_f=float(_v(d, "int_l2_f", np.nan)),
                      n_cross=float(_v(d, "final_cross", np.nan)),
                      n_roundtrips=np.nan,
                      p_a=pa, p_b=pb, cv_coverage=cov, low_support_fraction=low,
                      support_min_over_median=smr,
                      f_range=profile_range(_v(d, "F_ref"), emask),
                      cross_series=(ct if ct.size else None),
                      cross_series_t=(tt if tt.size else None),
                      sweep=("physics" if (locked_gamma is None
                                           or float(_v(d, "cfg__gamma")) == locked_gamma)
                             else "fr_rate"),
                      orphan=orphan)
        r["n_states_visited_frac"] = (1.0 if (np.isfinite(pa) and min(pa, 1 - pa) > 0)
                                      else np.nan)
        rows.append(finalize_run(r))
    return rows


def load_edb(root, system):
    paths = (sorted(glob.glob(os.path.join(root, "raw", "main", "*.npz")))
             + sorted(glob.glob(os.path.join(root, "raw", "rate", "*.npz"))))

    def cell_of(d):
        phi = float(_v(d, "cfg__phi"))
        gam = float(_v(d, "cfg__gamma"))
        return f"phi{phi:g}_gamma{gam:g}"
    return _load_edb_like(paths, system, cell_of, locked_gamma=15.0)


def load_edb_slow(pilot_dir, system):
    paths = sorted(glob.glob(os.path.join(pilot_dir, "raw", "main", "*.npz")))

    def cell_of(d):
        phi = float(_v(d, "cfg__phi"))
        muy = float(_v(d, "cfg__mu_y"))
        return f"phi{phi:g}_muy{muy:g}"
    return _load_edb_like(paths, system, cell_of, orphan=True, locked_gamma=15.0)


def load_eb(raw_root, system):
    """Entropic bottleneck.  NOTE: these runs store NO crossing counter and NO
    occupancy time series, so every transition-based indicator is unavailable.
    Only the support/coverage indicators can be built.  Occupancies of the two
    wells of U = H (x^2 - 1)^2 are still recoverable from p_hat."""
    rows = []
    seen = set()
    for stage_dir in sorted(glob.glob(os.path.join(raw_root, "stage*"))):
        stage = os.path.basename(stage_dir)
        if stage == "stage0_reproduce":
            continue
        for path in sorted(glob.glob(os.path.join(stage_dir, "*.npz"))):
            d = np.load(path, allow_pickle=True)
            method = str(_v(d, "method"))
            if method not in ("abf", "fr_estimated"):
                continue
            cell = (f"om{float(_v(d,'cfg__omega_in')):g}"
                    f"_b{float(_v(d,'cfg__beta')):g}"
                    f"_g{float(_v(d,'cfg__gamma')):g}")
            key = (cell, method, int(_v(d, "seed")))
            if key in seen:          # the base cell recurs in stages 2/3/4
                continue
            seen.add(key)
            grid = np.asarray(_v(d, "x_grid"), float)
            ph = np.asarray(_v(d, "p_hat"), float)
            emask = (grid >= -1.5) & (grid <= 1.5)
            pa, pb = split_prob(grid[emask], ph[emask], 0.0)
            cov, low, smr = coverage_stats(ph, emask)
            r = blank_run(system=system, cell=cell,
                          method=("abf" if method == "abf" else "fr"),
                          seed=int(_v(d, "seed")), stage=stage,
                          n_replicas=float(_v(d, "cfg__N")),
                          n_steps=float(_v(d, "cfg__n_steps")),
                          dt=float(_v(d, "cfg__dt")),
                          t_phys=float(_v(d, "cfg__n_steps")) * float(_v(d, "cfg__dt")),
                          l2_f=float(_v(d, "final_l2_f", np.nan)),
                          int_l2_f=float(_v(d, "int_l2_f", np.nan)),
                          n_cross=np.nan, n_roundtrips=np.nan,
                          p_a=pa, p_b=pb, cv_coverage=cov, low_support_fraction=low,
                          support_min_over_median=smr,
                          f_range=profile_range(_v(d, "F_ref"), emask),
                          sweep=("physics" if float(_v(d, "cfg__gamma")) == 15.0
                                 else "fr_rate"))
            rows.append(finalize_run(r))
    return rows


def load_toy2d(prod_dir, system):
    """2-D metastability toy (xi = x).  Read from the tracked summary CSVs.

    ``barrier_crossings`` counts sign changes of (x - 0) on the genuine Langevin
    proposal (src/abffr/simulation.py ~line 328).  Occupancies come from the
    time-averaged left/right region fractions.  ABF has ONE config; the FR arm
    has 36 tuning configs, so the cell gain is the MEDIAN over all 36 configs
    (not best-of-N) -- the best-of-N value is reported separately.
    """
    import csv as _csv
    long_path = os.path.join(prod_dir, "production_gpu_runs_long.csv")
    fin_path = os.path.join(prod_dir, "production_gpu_final_summary.csv")
    series = defaultdict(list)
    with open(long_path) as fh:
        for row in _csv.DictReader(fh):
            if row["method"] != "abf_only":
                continue
            series[row["run_id"]].append(row)
    ref_range = np.nan
    ref_csv = os.path.join(os.path.dirname(prod_dir), "reference", "reference_profile.csv")
    if os.path.exists(ref_csv):
        xs, fs = [], []
        with open(ref_csv) as fh:
            for row in _csv.DictReader(fh):
                key_x = "x" if "x" in row else list(row.keys())[0]
                key_f = next((k for k in row if k.lower().startswith("f")
                              and "prime" not in k.lower()), None)
                if key_f is None:
                    continue
                xs.append(float(row[key_x]))
                fs.append(float(row[key_f]))
        if fs:
            xs = np.asarray(xs)
            fs = np.asarray(fs)
            m = (xs >= -2.5) & (xs <= 2.5)
            ref_range = profile_range(fs, m)
    rows = []
    with open(fin_path) as fh:
        for row in _csv.DictReader(fh):
            meth = row["method"]
            if meth not in ("abf_only", "abf_fr_estimated"):
                continue
            is_abf = meth == "abf_only"
            rid = row["run_id"]
            n_rep, n_steps, dt = 1000.0, 100000.0, 2.0e-3
            ser = series.get(rid, [])
            lf = np.array([float(s["left_frac"]) for s in ser]) if ser else np.array([])
            rf = np.array([float(s["right_frac"]) for s in ser]) if ser else np.array([])
            tt = np.array([float(s["t"]) for s in ser]) if ser else np.array([])
            cs = np.array([float(s["barrier_crossings"]) for s in ser]) if ser else np.array([])
            if lf.size:
                a, b = float(lf.mean()), float(rf.mean())
                pa = a / max(a + b, EPS)
            else:
                a = float(row.get("frac_left", "nan") or "nan")
                b = float(row.get("frac_right", "nan") or "nan")
                pa = a / max(a + b, EPS) if np.isfinite(a) and np.isfinite(b) else np.nan
            cfg = row["config_id"]
            r = blank_run(system=system,
                          cell="toy2d_base",
                          method=("abf" if is_abf else "fr"),
                          seed=int(row["seed"]), stage="production_gpu",
                          n_replicas=n_rep, n_steps=n_steps, dt=dt, t_phys=n_steps * dt,
                          l2_f=float(row["final_l2_F"]),
                          int_l2_f=float(row["integrated_l2_F"]),
                          n_cross=float(row["barrier_crossings"]),
                          n_roundtrips=np.nan,
                          p_a=pa, p_b=(1 - pa) if np.isfinite(pa) else np.nan,
                          f_range=ref_range,
                          cross_series=(cs if cs.size else None),
                          cross_series_t=(tt if tt.size else None),
                          occ_series=(lf if lf.size else None),
                          occ_series_t=(tt if tt.size else None))
            r["fr_config"] = cfg
            r["n_states_visited_frac"] = 1.0
            rows.append(finalize_run(r))
    return rows


# ===========================================================================
# cell aggregation: ABF-only indicators + matched-seed mFR gain
# ===========================================================================
def build_cells(runs, rng, sweep="physics"):
    """One row per (system, cell): ABF indicator medians + matched-seed gain.

    ``sweep`` selects which runs enter: "physics" keeps only cells whose FR arm
    uses the study's locked FR rate (so a single cell = a single FR setting);
    "fr_rate" keeps the rate-sweep cells, which share one ABF indicator value per
    physics point and are reported only as a confound diagnostic.
    """
    runs = [r for r in runs if r.get("sweep", "physics") == sweep]
    by = defaultdict(lambda: {"abf": [], "fr": []})
    for r in runs:
        by[(r["system"], r["cell"])][r["method"]].append(r)

    ind_keys = ["cross_per_replica", "log10_cross_per_replica", "cross_rate_per_time",
                "roundtrips_per_replica", "roundtrip_rate_per_time",
                "n_relax", "log10_n_relax", "min_state_occupancy",
                "cv_coverage", "low_support_fraction", "support_min_over_median",
                "spectral_gap_matrix", "n_states_visited_frac", "fpt_frac", "p_a"]
    cells = []
    for (system, cell), grp in sorted(by.items()):
        abf, fr = grp["abf"], grp["fr"]
        if not abf:
            continue
        row = dict(system=system, cell=cell, sweep=sweep,
                   n_abf_seeds=len(abf), n_fr_seeds=len(fr),
                   n_replicas=abf[0]["n_replicas"], n_steps=abf[0]["n_steps"],
                   t_phys=abf[0]["t_phys"], orphan=bool(abf[0]["orphan"]))
        for k in ind_keys:
            vals = np.array([a[k] for a in abf], float)
            row[k] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")
            row[k + "_iqr"] = (float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))
                               if np.isfinite(vals).sum() > 1 else float("nan"))
        abf_l2 = np.array([a["l2_f"] for a in abf], float)
        row["abf_final_l2_f"] = float(np.nanmedian(abf_l2))
        rng_f = row["f_range"] = float(np.nanmedian([a["f_range"] for a in abf]))
        row["abf_norm_l2_f"] = (row["abf_final_l2_f"] / rng_f
                                if np.isfinite(rng_f) and rng_f > 0 else float("nan"))

        # ---- matched-seed gain ------------------------------------------- #
        gains, gains_int = [], []
        if fr:
            abf_by_seed = defaultdict(list)
            for a in abf:
                abf_by_seed[a["seed"]].append(a)
            fr_by_seed = defaultdict(list)
            for f in fr:
                fr_by_seed[f["seed"]].append(f)
            common = sorted(set(abf_by_seed) & set(fr_by_seed))
            n_fr_cfg = int(np.median([len(v) for v in fr_by_seed.values()])) if fr_by_seed else 0
            if common:
                # ONE gain per seed.  Where several FR tuning configs share a seed
                # (only the 2-D toy: 36 configs x 5 seeds) the per-seed value is
                # the MEDIAN over configs -- a typical-config gain, deliberately
                # not a best-of-N selection.  This keeps the bootstrap unit = seed.
                for s in common:
                    b = float(np.nanmedian([x["l2_f"] for x in abf_by_seed[s]]))
                    fv = float(np.nanmedian([x["l2_f"] for x in fr_by_seed[s]]))
                    if b > 0 and np.isfinite(fv):
                        gains.append(100.0 * (b - fv) / b)
                    bi = float(np.nanmedian([x["int_l2_f"] for x in abf_by_seed[s]]))
                    fi = float(np.nanmedian([x["int_l2_f"] for x in fr_by_seed[s]]))
                    if bi > 0 and np.isfinite(fi):
                        gains_int.append(100.0 * (bi - fi) / bi)
                row["pairing"] = ("matched_seed" if n_fr_cfg <= 1
                                  else f"matched_seed_median_over_{n_fr_cfg}_fr_configs")
                # best-of-N (reported, never used as the target)
                if n_fr_cfg > 1:
                    per_cfg = defaultdict(list)
                    for f in fr:
                        per_cfg[f.get("fr_config", "?")].append(f)
                    bmed = float(np.nanmedian(abf_l2))
                    best = np.nanmin([float(np.nanmedian([x["l2_f"] for x in v]))
                                      for v in per_cfg.values()])
                    row["gain_pct_best_of_n"] = (100.0 * (bmed - best) / bmed
                                                 if bmed > 0 else float("nan"))
                    row["n_fr_configs"] = len(per_cfg)
            else:   # unmatched seed ids -> unpaired medians (flagged)
                b = float(np.nanmedian(abf_l2))
                f = float(np.nanmedian([x["l2_f"] for x in fr]))
                if b > 0:
                    gains = [100.0 * (b - f) / b]
                row["pairing"] = "unpaired_median"
        gains = np.array(gains, float)
        row["n_gain_pairs"] = int(np.isfinite(gains).sum())
        if np.isfinite(gains).sum() >= 1:
            g = gains[np.isfinite(gains)]
            row["gain_pct"] = float(np.mean(g))
            row["gain_pct_median"] = float(np.median(g))
            if g.size >= 3:
                lo, hi = boot_ci(lambda idx: float(np.mean(g[idx])), g.size, rng)
            else:
                lo = hi = float("nan")
            row["gain_lo"], row["gain_hi"] = lo, hi
            row["win_rate"] = float(np.mean(g > 0))
        else:
            row["gain_pct"] = row["gain_pct_median"] = float("nan")
            row["gain_lo"] = row["gain_hi"] = row["win_rate"] = float("nan")
        gi = np.array(gains_int, float)
        row["gain_int_pct"] = float(np.nanmean(gi)) if np.isfinite(gi).any() else float("nan")

        # ---- label -------------------------------------------------------- #
        gp, lo, hi = row["gain_pct"], row["gain_lo"], row["gain_hi"]
        if not np.isfinite(gp):
            lab = "unknown"
        elif gp > PRACTICAL_MARGIN_PCT and (not np.isfinite(lo) or lo > 0):
            lab = "help"
        elif gp < -PRACTICAL_MARGIN_PCT and (not np.isfinite(hi) or hi < 0):
            lab = "harm"
        else:
            lab = "neutral"
        row["label"] = lab
        row["y_help"] = 1 if lab == "help" else (0 if lab in ("neutral", "harm") else -1)
        cells.append(row)
    return cells


# ===========================================================================
# validation machinery
# ===========================================================================
def fit_gate(x, y):
    """Fit direction + threshold maximising Youden's J on the TRAINING set.

    Returns (sign, threshold).  score = sign * x; predict help when
    score >= sign * threshold.
    """
    m = np.isfinite(x)
    x, y = x[m], y[m]
    if x.size < 3 or len(set(y.tolist())) < 2:
        return 0.0, float("nan")
    best = (-2.0, 1.0, float("nan"))
    cand = np.unique(x)
    cuts = np.concatenate([[-np.inf], (cand[:-1] + cand[1:]) / 2.0, [np.inf]]) \
        if cand.size > 1 else np.array([cand[0]])
    for sgn in (1.0, -1.0):
        s = sgn * x
        for c in cuts:
            pred = s >= sgn * c
            tpr = np.mean(pred[y == 1]) if (y == 1).any() else 0.0
            fpr = np.mean(pred[y == 0]) if (y == 0).any() else 0.0
            J = tpr - fpr
            if J > best[0]:
                best = (J, sgn, c)
    return best[1], best[2]


def loso(cells, feature, rng, exclude_systems=()):
    """Leave-one-SYSTEM-out evaluation of a single feature as a gate."""
    use = [c for c in cells
           if c["y_help"] in (0, 1) and np.isfinite(c[feature])
           and c["system"] not in exclude_systems]
    systems = sorted({c["system"] for c in use})
    x = np.array([c[feature] for c in use], float)
    y = np.array([c["y_help"] for c in use], int)
    g = np.array([c["gain_pct"] for c in use], float)
    sysarr = np.array([c["system"] for c in use])

    oos_score, oos_y, oos_sys, oos_gain = [], [], [], []
    oos_pred, folds, signs = [], [], []
    for s in systems:
        tr = sysarr != s
        te = ~tr
        if len(set(y[tr].tolist())) < 2:
            folds.append(dict(held_out=s, n_test=int(te.sum()),
                              status="train folds single-class"))
            continue
        sgn, thr = fit_gate(x[tr], y[tr])
        if sgn == 0.0:
            folds.append(dict(held_out=s, n_test=int(te.sum()), status="no gate"))
            continue
        # Scores from different folds are only comparable after they are put on a
        # common scale, because each fold fits its own direction AND the feature
        # has very different location/spread in different systems.  Standardise
        # the held-out scores with the TRAINING median/IQR of that fold (no
        # held-out information used).
        med = float(np.median(x[tr]))
        iqr = float(np.percentile(x[tr], 75) - np.percentile(x[tr], 25))
        scale = iqr if iqr > 0 else (float(np.std(x[tr])) or 1.0)
        sc = sgn * (x[te] - med) / scale
        pred = ((x[te] - thr) * sgn >= 0).astype(int)
        yt = y[te]
        tpr = float(np.mean(pred[yt == 1])) if (yt == 1).any() else float("nan")
        fpr = float(np.mean(pred[yt == 0])) if (yt == 0).any() else float("nan")
        acc = float(np.mean(pred == yt))
        bal = float(np.nanmean([tpr, 1 - fpr]))
        fold_auc = auc_score(sgn * x[te], yt)   # within-fold, scale-free
        rho_oos = spearman(sgn * x[te], g[te])
        rho_tr = spearman(sgn * x[tr], g[tr])
        folds.append(dict(held_out=s, n_test=int(te.sum()),
                          n_help_test=int((yt == 1).sum()),
                          train_sign=float(sgn), train_threshold=float(thr),
                          test_tpr=tpr, test_fpr=fpr, test_accuracy=acc,
                          test_balanced_accuracy=bal, test_auc_within_fold=fold_auc,
                          test_spearman_signed=rho_oos,
                          train_spearman_signed=rho_tr, status="ok"))
        signs.append(sgn)
        oos_score.extend(sc.tolist())
        oos_pred.extend(pred.tolist())
        oos_y.extend(yt.tolist())
        oos_sys.extend(sysarr[te].tolist())
        oos_gain.extend(g[te].tolist())

    out = dict(feature=feature, n_cells=len(use), systems=systems, folds=folds)
    out["fold_signs"] = signs
    out["sign_stable_across_folds"] = bool(len(set(signs)) <= 1) if signs else False
    if oos_y and len(set(oos_y)) > 1:
        sc = np.array(oos_score, float)
        yy = np.array(oos_y, int)
        pr = np.array(oos_pred, int)
        out["pooled_oos_auc"] = auc_score(sc, yy)
        lo, hi = boot_ci(lambda idx: auc_score(sc[idx], yy[idx]), len(yy), rng,
                         groups=np.array(oos_sys))
        out["pooled_oos_auc_lo"], out["pooled_oos_auc_hi"] = lo, hi
        tpr = float(np.mean(pr[yy == 1]))
        fpr = float(np.mean(pr[yy == 0]))
        out["pooled_oos_tpr"], out["pooled_oos_fpr"] = tpr, fpr
        out["pooled_oos_balanced_accuracy"] = 0.5 * (tpr + (1 - fpr))

        def _bal(idx):
            yb, pb = yy[idx], pr[idx]
            if not (yb == 1).any() or not (yb == 0).any():
                return float("nan")
            return 0.5 * (np.mean(pb[yb == 1]) + (1 - np.mean(pb[yb == 0])))
        lo, hi = boot_ci(_bal, len(yy), rng, groups=np.array(oos_sys))
        out["pooled_oos_balacc_lo"], out["pooled_oos_balacc_hi"] = lo, hi
        aucs = [f["test_auc_within_fold"] for f in folds
                if f.get("status") == "ok" and np.isfinite(f.get("test_auc_within_fold", np.nan))]
        out["mean_within_fold_auc"] = float(np.mean(aucs)) if aucs else float("nan")
        bals = [f["test_balanced_accuracy"] for f in folds if f.get("status") == "ok"]
        out["mean_fold_balanced_accuracy"] = float(np.nanmean(bals)) if bals else float("nan")
        out["pooled_oos_spearman"] = spearman(sc, np.array(oos_gain, float))
    else:
        for k in ("pooled_oos_auc", "pooled_oos_auc_lo", "pooled_oos_auc_hi",
                  "pooled_oos_tpr", "pooled_oos_fpr", "pooled_oos_balanced_accuracy",
                  "pooled_oos_balacc_lo", "pooled_oos_balacc_hi",
                  "mean_within_fold_auc", "mean_fold_balanced_accuracy",
                  "pooled_oos_spearman"):
            out[k] = float("nan")
        out["status"] = ("no out-of-sample positives and negatives: the 'help' "
                         "label is confined to systems that cannot be scored on "
                         "this feature")

    # in-sample (for contrast only -- NOT the headline)
    out["in_sample_auc"] = auc_score(x, y)
    out["in_sample_spearman"] = spearman(x, g)
    lo, hi = boot_ci(lambda idx: spearman(x[idx], g[idx]), len(x), rng)
    out["in_sample_spearman_lo"], out["in_sample_spearman_hi"] = lo, hi
    lo, hi = boot_ci(lambda idx: auc_score(x[idx], y[idx]), len(x), rng)
    out["in_sample_auc_lo"], out["in_sample_auc_hi"] = lo, hi
    out["within_system_spearman"] = {
        s: spearman(x[sysarr == s], g[sysarr == s]) for s in systems}
    return out


def system_identity_check(cells, feature):
    """Is the indicator a disguised system label?

    eta2 = between-system variance / total variance of the indicator.  Also
    reports the mean |within-system Spearman| and the discriminability of the
    system label itself (best achievable AUC using system identity alone).
    """
    use = [c for c in cells if np.isfinite(c[feature])]
    if len(use) < 4:
        return dict(feature=feature, status="insufficient")
    x = np.array([c[feature] for c in use], float)
    g = np.array([c["gain_pct"] for c in use], float)
    sysarr = np.array([c["system"] for c in use])
    grand = x.mean()
    ss_tot = float(((x - grand) ** 2).sum())
    ss_between = 0.0
    within_rho, spans = {}, {}
    for s in np.unique(sysarr):
        xs = x[sysarr == s]
        ss_between += len(xs) * (xs.mean() - grand) ** 2
        within_rho[s] = spearman(xs, g[sysarr == s])
        spans[s] = (float(xs.min()), float(xs.max()))
    eta2 = float(ss_between / ss_tot) if ss_tot > 0 else float("nan")
    # pairwise range overlap
    keys = sorted(spans)
    n_pairs = n_overlap = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = spans[keys[i]], spans[keys[j]]
            n_pairs += 1
            if min(a[1], b[1]) >= max(a[0], b[0]):
                n_overlap += 1
    finite_rho = [v for v in within_rho.values() if np.isfinite(v)]
    # AUC obtainable from system identity alone (in-sample upper bound):
    y = np.array([c["y_help"] for c in use], int)
    m = y >= 0
    sys_rate = {}
    for s in np.unique(sysarr[m]):
        sys_rate[s] = float(np.mean(y[m][sysarr[m] == s]))
    label_score = np.array([sys_rate[s] for s in sysarr[m]], float)
    return dict(feature=feature, eta2_system=eta2,
                mean_abs_within_system_spearman=(float(np.mean(np.abs(finite_rho)))
                                                 if finite_rho else float("nan")),
                within_system_spearman=within_rho,
                per_system_range=spans,
                pairwise_range_overlap_fraction=(n_overlap / n_pairs) if n_pairs else float("nan"),
                system_label_in_sample_auc=auc_score(label_score, y[m]),
                n_cells=len(use))


def early_window_indicator(runs, frac):
    """Recompute the indicator on the first ``frac`` of each ABF run, for the
    systems that stored a CUMULATIVE crossing time series."""
    out = defaultdict(list)
    for r in runs:
        if r["method"] != "abf":
            continue
        cs, ts = r.get("cross_series"), r.get("cross_series_t")
        if cs is None or ts is None or len(cs) < 4:
            continue
        cs = np.asarray(cs, float)
        ts = np.asarray(ts, float)
        k = max(1, int(round(frac * (len(ts) - 1))))
        c_early = cs[k] - cs[0]
        cpr = c_early / r["n_replicas"]
        # occupancies: use the same full-run lumping (p_hat is only stored at the
        # end); this makes the test *favourable* to the early indicator.
        nr = two_state_relax(cpr, r["p_a"], r["p_b"])
        out[(r["system"], r["cell"])].append(
            (math.log10(nr + 1.0) if np.isfinite(nr) else np.nan,
             math.log10(cpr + 1.0) if np.isfinite(cpr) else np.nan))
    agg = {}
    for k, v in out.items():
        arr = np.array(v, float)
        agg[k] = (float(np.nanmedian(arr[:, 0])), float(np.nanmedian(arr[:, 1])))
    return agg


# ===========================================================================
# CSV writing
# ===========================================================================
def write_csv(rows, path, cols=None):
    import csv as _csv
    if not rows:
        print(f"  (no rows) skip {path}")
        return
    cols = cols or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"  wrote {path}  ({len(rows)} rows)")


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    R = os.path.join(REPO, "results")
    print("loading ABF/FR runs ...")
    runs = []
    runs += load_wca(os.path.join(R, "wca_phase_diagram/production/raw"),
                     "wca", "production")
    wca_rep = load_wca(os.path.join(R, "wca_representative/raw"),
                       "wca_rep", "representative")
    runs += load_alkane_torsion(os.path.join(R, "alkanes/production/raw"),
                                "alkane_torsion")
    runs += load_alkane_dist(os.path.join(R, "alkanes_cv_extension/r15_methods/raw"),
                             "alkane_dist")
    runs += load_alkane_torus(os.path.join(R, "alkanes_cv_extension/2d_methods/raw"),
                              "alkane_torus")
    runs += load_edb(os.path.join(R, "entropy_dominant_bottleneck/sweep_20260614_015145"),
                     "edb")
    runs += load_edb_slow(os.path.join(
        R, "entropy_dominant_bottleneck_slow_transverse/pilot_20260616_225737"), "edb_slow")
    runs += load_eb(os.path.join(R, "entropic_bottleneck/raw"), "eb")
    runs += load_toy2d(os.path.join(R, "two_dim_xi_x/production_gpu"), "toy2d")

    print(f"  {len(runs)} runs (+{len(wca_rep)} wca_rep robustness runs)")
    write_csv([{k: v for k, v in r.items()
                if not isinstance(v, (np.ndarray, list, type(None)))}
               for r in runs],
              os.path.join(args.out, "runs_long.csv"))

    print("aggregating cells ...")
    cells = build_cells(runs, rng, sweep="physics")
    cells_rate = build_cells(runs, rng, sweep="fr_rate")
    cells_rep = build_cells(wca_rep, rng, sweep="physics")
    cell_cols = ["system", "cell", "sweep", "label", "y_help", "gain_pct",
                 "gain_pct_median", "gain_lo", "gain_hi", "gain_int_pct", "win_rate",
                 "gain_pct_best_of_n", "n_fr_configs", "n_gain_pairs",
                 "n_abf_seeds", "n_fr_seeds", "pairing", "n_replicas", "n_steps",
                 "t_phys", "orphan",
                 "cross_per_replica", "log10_cross_per_replica", "cross_rate_per_time",
                 "roundtrips_per_replica", "roundtrip_rate_per_time",
                 "n_relax", "log10_n_relax", "min_state_occupancy", "p_a",
                 "cv_coverage", "low_support_fraction", "support_min_over_median",
                 "spectral_gap_matrix", "n_states_visited_frac", "fpt_frac",
                 "abf_final_l2_f", "abf_norm_l2_f", "f_range"]
    write_csv(cells, os.path.join(args.out, "indicator_table.csv"), cell_cols)
    write_csv(cells_rate, os.path.join(args.out, "rate_confound.csv"), cell_cols)
    write_csv(cells_rep, os.path.join(args.out, "indicator_table_wca_representative.csv"),
              cell_cols)

    # ------------------------------------------------------------------ #
    features = [GATE_PRIMARY, "log10_cross_per_replica", "cross_rate_per_time",
                "roundtrips_per_replica", "roundtrip_rate_per_time", "n_relax",
                "min_state_occupancy", "cv_coverage", "n_states_visited_frac",
                "fpt_frac", "spectral_gap_matrix", "support_min_over_median",
                INCUMBENT_A, INCUMBENT_B]
    print("leave-one-system-out validation ...")
    loso_res = {}
    for f in features:
        loso_res[f] = loso(cells, f, rng)
        loso_res[f + "__no_orphan"] = loso(cells, f, rng, exclude_systems=("edb_slow",))

    ident = {f: system_identity_check(cells, f) for f in features}

    print("short-window check ...")
    early = {}
    for frac in (0.10, 0.25, 0.50):
        agg = early_window_indicator(runs, frac)
        early[f"{frac:.2f}"] = agg
    # attach early features to the cells that have them, then re-run LOSO
    early_res = {}
    for frac_key, agg in early.items():
        avail = set(agg.keys())
        for c in cells:
            k = (c["system"], c["cell"])
            c["log10_n_relax_early"] = agg.get(k, (np.nan, np.nan))[0]
            c["log10_cross_per_replica_early"] = agg.get(k, (np.nan, np.nan))[1]
        sub = [c for c in cells if (c["system"], c["cell"]) in avail]
        rho_full_vs_early = spearman([c["log10_n_relax"] for c in sub],
                                     [c["log10_n_relax_early"] for c in sub])
        early_res[frac_key] = dict(
            n_cells_with_early_series=len(sub),
            systems=sorted({c["system"] for c in sub}),
            spearman_early_vs_full=rho_full_vs_early,
            loso_early=loso(sub, "log10_n_relax_early", rng),
            loso_full_same_subset=loso(sub, "log10_n_relax", rng))

    # ------------------------------------------------------------------ #
    # availability audit of every requested indicator
    avail_rows = []
    requested = {
        "transition rate per unit physical time": "cross_rate_per_time",
        "round trips per replica per unit physical time": "roundtrip_rate_per_time",
        "fraction of replicas with >=1 round trip": None,
        "median first-passage time / horizon": None,
        "ensemble first-passage / horizon (surrogate)": "fpt_frac",
        "state-transition spectral-gap proxy (2-state rate)": "log10_n_relax",
        "state-transition spectral-gap proxy (matrix eigenvalue)": "spectral_gap_matrix",
        "basin-discovery completeness (states visited)": "n_states_visited_frac",
        "basin-discovery completeness (CV coverage)": "cv_coverage",
        "number of distinct crossing lineages": None,
    }
    systems = sorted({c["system"] for c in cells})
    for name, key in requested.items():
        row = dict(indicator=name, column=(key or "UNAVAILABLE"))
        for s in systems:
            sub = [c for c in cells if c["system"] == s]
            if key is None:
                row[s] = "no"
            else:
                row[s] = "yes" if any(np.isfinite(c[key]) for c in sub) else "no"
        avail_rows.append(row)
    write_csv(avail_rows, os.path.join(args.out, "indicator_availability.csv"),
              ["indicator", "column"] + systems)

    # ------------------------------------------------------------------ #
    manifest = dict(
        git_commit=_git_commit(), argv=sys.argv, rng_seed=RNG_SEED,
        n_boot=N_BOOT, practical_margin_pct=PRACTICAL_MARGIN_PCT,
        gate_primary=GATE_PRIMARY, incumbent_a=INCUMBENT_A, incumbent_b=INCUMBENT_B,
        n_runs=len(runs), n_cells=len(cells),
        cells_per_system={s: sum(1 for c in cells if c["system"] == s) for s in systems},
        labels_per_system={s: {lab: sum(1 for c in cells
                                        if c["system"] == s and c["label"] == lab)
                               for lab in ("help", "neutral", "harm", "unknown")}
                           for s in systems},
    )
    # ------------------------------------------------------------------ #
    # within-system view: is the indicator informative INSIDE a system?
    ws_rows = []
    for f in features:
        for s in systems:
            sub = [c for c in cells if c["system"] == s and np.isfinite(c[f])]
            if len(sub) < 2:
                continue
            xs = np.array([c[f] for c in sub], float)
            gs = np.array([c["gain_pct"] for c in sub], float)
            ws_rows.append(dict(feature=f, system=s, n_cells=len(sub),
                                spearman_vs_gain=spearman(xs, gs),
                                indicator_min=float(np.min(xs)),
                                indicator_max=float(np.max(xs)),
                                indicator_median=float(np.median(xs)),
                                gain_min=float(np.nanmin(gs)),
                                gain_max=float(np.nanmax(gs)),
                                n_help=sum(1 for c in sub if c["label"] == "help"),
                                n_harm=sum(1 for c in sub if c["label"] == "harm"),
                                n_neutral=sum(1 for c in sub if c["label"] == "neutral")))
    write_csv(ws_rows, os.path.join(args.out, "within_system.csv"))

    # ------------------------------------------------------------------ #
    # sensitivity of the verdict to the practical-equivalence margin
    margin_rows = []
    for margin in (0.0, 5.0, 10.0, 20.0):
        tmp = []
        for c in cells:
            c2 = dict(c)
            gp, lo, hi = c["gain_pct"], c["gain_lo"], c["gain_hi"]
            if not np.isfinite(gp):
                lab = "unknown"
            elif gp > margin and (not np.isfinite(lo) or lo > 0):
                lab = "help"
            elif gp < -margin and (not np.isfinite(hi) or hi < 0):
                lab = "harm"
            else:
                lab = "neutral"
            c2["label"] = lab
            c2["y_help"] = 1 if lab == "help" else (0 if lab in ("neutral", "harm") else -1)
            tmp.append(c2)
        row = dict(margin_pct=margin,
                   n_help=sum(1 for c in tmp if c["label"] == "help"))
        for f in (GATE_PRIMARY, INCUMBENT_A, INCUMBENT_B, "support_min_over_median"):
            row[f + "_loso_auc"] = loso(tmp, f, rng)["pooled_oos_auc"]
        margin_rows.append(row)
    write_csv(margin_rows, os.path.join(args.out, "margin_sensitivity.csv"))

    # summary CSV of the head-to-head, easiest thing to read
    hh = []
    for f in features:
        r = loso_res[f]
        r2 = loso_res[f + "__no_orphan"]
        hh.append(dict(feature=f,
                       deployable=("no" if f == INCUMBENT_A else "yes"),
                       n_cells_scored=r["n_cells"], n_systems=len(r["systems"]),
                       systems="|".join(r["systems"]),
                       loso_auc=r["pooled_oos_auc"],
                       loso_auc_lo=r["pooled_oos_auc_lo"],
                       loso_auc_hi=r["pooled_oos_auc_hi"],
                       loso_balacc=r["pooled_oos_balanced_accuracy"],
                       loso_balacc_lo=r["pooled_oos_balacc_lo"],
                       loso_balacc_hi=r["pooled_oos_balacc_hi"],
                       loso_tpr=r["pooled_oos_tpr"], loso_fpr=r["pooled_oos_fpr"],
                       mean_within_fold_auc=r["mean_within_fold_auc"],
                       loso_spearman=r["pooled_oos_spearman"],
                       sign_stable_across_folds=r["sign_stable_across_folds"],
                       fold_signs="|".join(f"{s:+.0f}" for s in r["fold_signs"]),
                       in_sample_auc=r["in_sample_auc"],
                       in_sample_auc_lo=r["in_sample_auc_lo"],
                       in_sample_auc_hi=r["in_sample_auc_hi"],
                       in_sample_spearman=r["in_sample_spearman"],
                       in_sample_spearman_lo=r["in_sample_spearman_lo"],
                       in_sample_spearman_hi=r["in_sample_spearman_hi"],
                       loso_auc_no_orphan=r2["pooled_oos_auc"],
                       loso_auc_no_orphan_lo=r2["pooled_oos_auc_lo"],
                       loso_auc_no_orphan_hi=r2["pooled_oos_auc_hi"],
                       n_cells_no_orphan=r2["n_cells"],
                       eta2_system=ident[f].get("eta2_system", float("nan")),
                       mean_abs_within_system_rho=ident[f].get(
                           "mean_abs_within_system_spearman", float("nan")),
                       range_overlap=ident[f].get(
                           "pairwise_range_overlap_fraction", float("nan")),
                       status=r.get("status", "ok")))
    write_csv(hh, os.path.join(args.out, "head_to_head.csv"))

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(dict(manifest=manifest, loso=loso_res,
                       system_identity=ident, short_window=early_res),
                  fh, indent=1, default=float)
    print(f"  wrote {os.path.join(args.out, 'results.json')}")

    # ------------------------------------------------------------------ #
    if not args.quiet:
        print("\n=== cells per system (physics cells only) ===")
        for s in systems:
            print(f"  {s:16s} n_cells={manifest['cells_per_system'][s]:3d} "
                  f"labels={manifest['labels_per_system'][s]}")
        print("\n=== head-to-head (leave-one-SYSTEM-out) ===")
        print(f"  {'feature':30s} {'nC':>3s} {'AUC_oos':>8s} {'[95% CI]':>16s} "
              f"{'balacc':>7s} {'rho_oos':>8s} {'AUCin':>6s} {'sign':>6s}")
        for f in features:
            r = loso_res[f]
            print(f"  {f:30s} {r['n_cells']:3d} {r['pooled_oos_auc']:8.3f} "
                  f"[{r['pooled_oos_auc_lo']:5.3f},{r['pooled_oos_auc_hi']:5.3f}] "
                  f"{r['pooled_oos_balanced_accuracy']:7.3f} "
                  f"{r['pooled_oos_spearman']:8.3f} {r['in_sample_auc']:6.3f} "
                  f"{'stable' if r['sign_stable_across_folds'] else 'FLIPS':>6s}")
        print("\n=== per-fold detail, primary gate + incumbents ===")
        for f in (GATE_PRIMARY, INCUMBENT_A, INCUMBENT_B):
            print(f"  -- {f}")
            for fo in loso_res[f]["folds"]:
                if fo.get("status") != "ok":
                    print(f"     held out {fo['held_out']:15s} SKIPPED ({fo['status']})")
                    continue
                print(f"     held out {fo['held_out']:15s} n={fo['n_test']:2d} "
                      f"help={fo['n_help_test']:2d} sign={fo['train_sign']:+.0f} "
                      f"thr={fo['train_threshold']:.4g} balacc={fo['test_balanced_accuracy']:.3f} "
                      f"AUC={fo['test_auc_within_fold']:.3f} "
                      f"rho_test={fo['test_spearman_signed']:+.3f} "
                      f"rho_train={fo['train_spearman_signed']:+.3f}")
        print("\n=== system-identity check ===")
        for f in (GATE_PRIMARY, INCUMBENT_A, INCUMBENT_B):
            i = ident[f]
            print(f"  {f:26s} eta2_system={i.get('eta2_system', float('nan')):.3f} "
                  f"mean|rho_within|={i.get('mean_abs_within_system_spearman', float('nan')):.3f} "
                  f"range_overlap={i.get('pairwise_range_overlap_fraction', float('nan')):.3f} "
                  f"sys_label_AUC={i.get('system_label_in_sample_auc', float('nan')):.3f}")
            print(f"      within-system rho: "
                  + ", ".join(f"{k}={v:+.2f}" for k, v in
                              sorted(i.get("within_system_spearman", {}).items())
                              if np.isfinite(v)))
        print("\n=== short-window check ===")
        for k, v in early_res.items():
            print(f"  first {float(k)*100:.0f}% : n_cells={v['n_cells_with_early_series']} "
                  f"systems={v['systems']} rho(early,full)={v['spearman_early_vs_full']:.3f} "
                  f"AUC_oos_early={v['loso_early']['pooled_oos_auc']:.3f} "
                  f"AUC_oos_full(same subset)={v['loso_full_same_subset']['pooled_oos_auc']:.3f}")
        print("\n=== robustness: drop the ORPHAN system (edb_slow) ===")
        for f in features:
            r2 = loso_res[f + "__no_orphan"]
            print(f"  {f:30s} n={r2['n_cells']:3d} AUC_oos={r2['pooled_oos_auc']:6.3f} "
                  f"[{r2['pooled_oos_auc_lo']:5.3f},{r2['pooled_oos_auc_hi']:5.3f}]")
        print("\n=== robustness: WCA gains from the 10-seed `representative` stage ===")
        rep_map = {c["cell"]: c for c in cells_rep}
        for c in cells:
            if c["system"] != "wca" or c["cell"] not in rep_map:
                continue
            r = rep_map[c["cell"]]
            print(f"  {c['cell']:20s} phase(4 seeds) gain={c['gain_pct']:+8.2f}% "
                  f"[{c['label']:7s}]   representative(10 seeds) gain={r['gain_pct']:+8.2f}% "
                  f"[{r['label']}]")
        print("\n=== practical-equivalence margin sensitivity ===")
        for m in margin_rows:
            print(f"  margin={m['margin_pct']:5.1f}%  n_help={m['n_help']:2d}  "
                  + "  ".join(f"{k.replace('_loso_auc',''):>22s}={m[k]:.3f}"
                              for k in m if k.endswith("_loso_auc")))
        print("\n=== FR-rate confound (ABF indicator fixed, FR rate varied) ===")
        for c in cells_rate:
            print(f"  {c['system']:9s} {c['cell']:22s} log10_n_relax="
                  f"{c['log10_n_relax'] if np.isfinite(c['log10_n_relax']) else float('nan'):.4f} "
                  f"gain={c['gain_pct']:+8.2f}%  label={c['label']}")
    print("\ndone.")


if __name__ == "__main__":
    main()
