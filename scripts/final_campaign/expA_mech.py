#!/usr/bin/env python
"""Experiment A mechanism read-outs, guards and descriptors (docs/final_campaign/PREREGISTRATION.md 1.2, 1.5-1.7).

    python scripts/final_campaign/expA_mech.py [--cells-dir configs/final_campaign/cells/expA]
        [--out-dir results/final_campaign/expA/synthesis] [--exp1-summary results/mechanism/matched_free_energy/
        shift/analysis/summary.json] [--n-boot 10000] [--no-verify-runs] [--provisional] [--force] [--recompute]
        [--dry-run]

Inputs: the raw runs of every cell (results_dir/N<N>/s<seed>_<method>.npz; each plan-checked against its cell config
by eqb_family.check_plan through mech_analysis.load_run, whose D1-D3 consistency identities must hold), the cells'
analysis summaries (scripts/equal_budget/analyze_ladder.py), and, read only, the Experiment I shifted-fibre summary
(replication table).  The D2 machinery of scripts/mechanism/mech_analysis.py is used by import (D2Calc, d2_arrays,
d2_summarise, d2_paired with matched=True, the reflected / basic seed bootstrap, cell_model,
exact_bin_force_moments); that file is not edited and its dynamics_name() (which maps every shifted-fibre lambda to
'shift') is not used: cells are keyed by lambda here.

Read-outs (seeds are the independent units; default_rng(20261010), n_boot resamples; one fresh generator per
quantity unless stated):
  D2 per lambda and N (end snapshot, the seeds with D2 in both arms): FR B_c per clone-age class ([0,0.01),
     [0.01,0.1), [0.1,1), >=1, never, all) and ABF's B, debiased RMS with the reflected (basic) CI
     (mech_analysis.d2_summarise); the matched-position excess of each FR class over ABF's deposits (d2_paired(FR, c,
     ABF, all, matched=True): B_excess = B_c|J - B_ABF|c, CI, p); the young-clone class [0.01, 0.1) singled out.
  MC3 (PREREGISTRATION 1.2) for lambda_hi in {4, 16, 128} at each N: the change from lambda = 1 of the young-clone
     matched excess, B_excess_did = excess(lambda_hi) - excess(1) (d2_paired(FR_hi, young, FR_1, young, calc_c=ABF_hi,
     cc=all, calc_d=ABF_1, cd=all, matched=True), seeds common to the four arms); Holm over the two N per lambda_hi;
     MC3 passes at (lambda_hi, N) iff Holm p < 0.05 and B_excess_did < 0 (fewer than 10 seeds: no claim).
  Gate decompositions (descriptive, analysis-defined: PREREGISTRATION 1.5(7) asks for "a descriptive decomposition of
     FR's gate-region excess bias into age-class contributions" without defining it), in the gate |x| <= 0.5 AND in
     each flank separately (gate_neg x < 0, gate_pos x > 0), so that opposite-sign bias of the two flanks (and of the
     two transit directions) cannot cancel unnoticed:
       signed     K_c = sum_j n^FR_cj (delta^FR_cj - delta^ABF_j) / sum_j n^FR_all,j  (eval bins of the region with
                  ABF deposits), sum_c K_c = K_all exactly; phi_c = the class's share of FR's deposits there;
       quadratic  Q_c = sum_j n^FR_cj [(delta^FR_cj^2 - V^FR_cj) - (delta^ABF_j^2 - V^ABF_j)] / sum_j n^FR_all,j
                  (bins where ABF's and the class's debiased squared deviation is usable; V = the D2 variance of the
                  bin mean, the debiasing of B and B_excess), Q_sum = sum_c Q_c exactly (sign-insensitive: the
                  deposit-weighted mean class-resolved excess squared bias over ABF's).
     Percentile seed-bootstrap CIs (FR and ABF resampled with the same seeds).  'gate_excess_decomposition' keeps the
     pooled-gate signed rows in the flat layout.
  Genealogy (FR arm): time-averaged windowed ancestor ESS/N = mean of gen_ess_win over the 200 uniform-u saves with
     step >= the genealogy window (1.6 t.u. in production); maximum windowed lineage fraction = max of gen_maxfam_win
     over all saves (equals the summary's max_gen_maxfam_win, checked); min windowed ESS/N, unique ancestors / N,
     realised deaths per walker per t.u. (summary); seed medians with bootstrap CIs.
  Guards at (lambda, N) (PREREGISTRATION 1.7): g1 fails iff the 95 % CI of G(final e_F) lies entirely above 0
     (lower end > 0); g2 the same for G(Ibar_F'_stat); g3 (lambda != 1) fails iff the upper 95 % CI of the seed-median
     ratio lambda / lambda 1 of FR's time-averaged windowed ESS/N is < 0.8 OR the lower 95 % CI of the seed-median
     ratio of FR's maximum windowed lineage fraction is > 1.25 (median of per-seed ratios, seeds coupled across
     lambda); g4 fails iff any run of the cell at that N (both arms) has a non-finite state (X_final, Y_final, M_all,
     C_all, hist_inst, region_frac, max_abs_bias; FR arms also the genealogy arrays, which are NaN by contract in an
     ABF arm) or a non-finite error curve (summary n_nan_*), or
     an FR cap violation (meta cap != max(cap_min, floor(0.08 N)), realised-death histogram longer than cap + 1 or
     inconsistent with the death / opportunity counters, deaths in an ABF arm).  A guard that cannot be evaluated is
     null (the outcome classifier treats it as not holding).
  HA3 at (lambda, N): holds iff the upper 95 % CI of G(Ibar_TV_half) < 0; persists iff it holds in every cell.
  ABF-only cross-lambda descriptors: ABF Ibar_F, gate transitions per run (final_transitions), tau(TV_half <= 0.1)
     and first arrival x > 0.5 (u and t) per lambda and N (inf-aware seed medians with bootstrap CIs) and the ratio of
     seed medians lambda / lambda 1 (paired: one index draw resamples the coupled seeds of both cells; finite / inf = 0,
     inf / finite = inf, inf / inf undefined, the undefined fraction reported).
  Cost per successful threshold, per (lambda, N, arm, threshold) for the persistent tau at e_F 0.002 / 0.0055 / 0.02,
     e_F'_stat 0.004 / 0.0136 / 0.0946 and TV_half 0.1: (a) sum over runs of the full cost / #successes and (b) oracle
     stopping (sum over successes of tau_u x cost + sum over failures of the full cost) / #successes, in walker-steps
     (exact integers: full = B, tau = round(tau_u n_steps) N; the numerator is stored as an integer) and in
     core-seconds (full = the run's wall_s from the summary, tau = tau_u wall_s); 'inf (0 successes)' when none.
  Replication (descriptive): lambda = 1 vs the Experiment I shifted fibre at each common N (Ibar_F, final e_F,
     Ibar_F'_stat, Ibar_TV_half: arm medians and paired G from the two summaries).

  g4 also fails for a run whose engine sha256 is not the frozen 69087f2c... (full 64 characters) or whose plan check /
     D1-D3 identities fail; such runs also make the output incomplete.

Before computing, every summary must belong to its CURRENT config / scoring code / reference and be fresh against the
run tree (cross_cell.check_summary / check_freshness; --no-verify-runs skips freshness and forbids complete), since
g1, g2, HA3, the cost and the ABF descriptors are read from the summaries.  complete =
expA_analysis_common.completeness() (incl. every cell's completeness audit PASS on the current summary, n_boot = 10 000
here and in every summary, no --provisional) AND every planned run loaded with D2 AND only frozen-engine runs.

Writes <out-dir>/mech.json (strict JSON; input_signature over every run file's sha256, the summaries, the Exp I
summary, the code and the completeness basis; complete; completeness; provenance) and mech.md (placeholder JSON
first, then the markdown, then the final JSON).  Output policy as expA_analysis_common.decide_write.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import expA_analysis_common as C  # noqa: E402
import numpy as np  # noqa: E402
import eqb_metrics as M  # noqa: E402
import eqb_family as MF  # noqa: E402
import mech_analysis as MA  # noqa: E402  (scripts/mechanism/mech_analysis.py, by import, unmodified)

SCHEMA = "final_expA_mech/2"
YOUNG = 1                                     # clone-age class [0.01, 0.1) t.u.
ALL = MA.ALL
N_AGE = MA.N_AGE
CLASS_NAMES = list(MA.AGE_DEFAULT) + ["all"]
G3_ESS_BOUND = 0.8
G3_MAXFAM_BOUND = 1.25
HI_LAMS = (4.0, 16.0, 128.0)
TAU_THRESHOLDS = ([(f"tau_e_F_{n}_u", "e_F", n) for n in M.THR_NAMES]
                  + [(f"tau_e_Fp_stat_{n}_u", "e_Fp_stat", n) for n in M.THR_NAMES]
                  + [("tau_TV_half_u", "TV_half", "0.1")])
ABF_DESCRIPTORS = (("Ibar_F", "ABF Ibar_F"), ("final_transitions", "ABF gate transitions per run"),
                   ("tau_TV_half_u", "ABF tau(TV_half <= 0.1), budget fraction u"),
                   ("tau_TV_half_t", "ABF tau(TV_half <= 0.1), t.u."),
                   ("first_right_u", "ABF first arrival x > 0.5, budget fraction u"),
                   ("first_right_t", "ABF first arrival x > 0.5, t.u."))
REPL_METRICS = ("Ibar_F", "final_e_F", "Ibar_Fp_stat", "Ibar_TV_half")
FINITE_KEYS = ("X_final", "Y_final", "M_all", "C_all", "hist_inst", "region_frac", "max_abs_bias")
GEN_KEYS = ("gen_ess_win", "gen_maxfam_win", "gen_ess_run", "gen_maxfam_run")     # NaN by contract in an ABF arm
EXTRA_KEYS = FINITE_KEYS + GEN_KEYS + ("save_step", "fr_deaths_hist", "fr_deaths_cum", "fr_opp_cum")
NAN_KEYS = ("n_nan_e_F", "n_nan_e_Fp", "n_nan_e_Fp_stat", "n_nan_e_F_em", "n_nan_e_Fp_em")
CODE_FILES = [os.path.abspath(__file__), os.path.join(HERE, "expA_analysis_common.py"),
              os.path.join(C.ROOT, "scripts", "mechanism", "mech_analysis.py"),
              os.path.join(C.ROOT, "scripts", "mechanism", "cross_cell.py"),
              os.path.join(C.ROOT, "scripts", "equal_budget", "eqb_metrics.py"),
              os.path.join(C.ROOT, "scripts", "equal_budget", "eqb_family.py"),
              os.path.join(C.ROOT, "src", "gateway_family_validation.py")]


# ============================================================================================ pure functions
def ess_time_average(save_step, gen_ess_win, n_steps, window_steps):
    """Mean of the windowed ancestor ESS/N over the 200 uniform-u saves whose step is >= window_steps (the windowed
    genealogy has covered a full window); NaN if there is none."""
    step = np.asarray(save_step, np.int64)
    iu, _ = M.uniform_indices(step, int(n_steps))
    sel = iu[step[iu] >= int(window_steps)]
    if sel.size == 0:
        return math.nan
    return float(np.mean(np.asarray(gen_ess_win, float)[sel]))


def g3_rule(ess_hi, ess_ref, maxfam_hi, maxfam_ref, n_boot=C.N_BOOT, seed=C.BOOT_SEED):
    """Guard g3 (PREREGISTRATION 1.7) from per-seed {seed: value} dicts of FR's time-averaged windowed ESS/N and
    maximum windowed lineage fraction at the higher lambda and at lambda 1 (seeds coupled): fails iff the upper 95 %
    CI of the seed-median ratio of ESS < 0.8 or the lower 95 % CI of the seed-median ratio of the lineage fraction
    > 1.25.  holds = None when a ratio cannot be computed."""
    e = C.median_of_ratios(ess_hi, ess_ref, n_boot, seed)
    f = C.median_of_ratios(maxfam_hi, maxfam_ref, n_boot, seed)
    ok = e["n"] > 0 and f["n"] > 0
    fail_e = bool(ok and e["ci95"][1] < G3_ESS_BOUND)
    fail_f = bool(ok and f["ci95"][0] > G3_MAXFAM_BOUND)
    return dict(holds=(None if not ok else not (fail_e or fail_f)), fails_ess=fail_e, fails_maxfam=fail_f,
                ess_ratio=dict((k, e[k]) for k in ("n", "median", "ci95")),
                maxfam_ratio=dict((k, f[k]) for k in ("n", "median", "ci95")),
                n=min(e["n"], f["n"]), calibrated=bool(min(e["n"], f["n"]) >= C.SMALL_N),
                rule=f"fails iff upper CI(ESS ratio) < {G3_ESS_BOUND} or lower CI(max lineage ratio) > {G3_MAXFAM_BOUND}")


def ci_guard(G_ci95):
    """g1 / g2: holds iff the 95 % CI of G does not lie entirely above 0 (lower end <= 0); None if undefined."""
    lo = C.fnum((G_ci95 or [None])[0])
    if math.isnan(lo):
        return None
    return bool(not lo > 0)


def cost_per_success(tau_u, full_cost):
    """(a) sum(full) / n_success and (b) (sum_success tau_u full + sum_failure full) / n_success for per-run arrays
    (tau_u = inf when censored; NaN runs are dropped).  inf when no run succeeds."""
    t = np.asarray(tau_u, float)
    c = np.asarray(full_cost, float)
    keep = ~(np.isnan(t) | np.isnan(c))
    t, c = t[keep], c[keep]
    succ = np.isfinite(t)
    ns = int(succ.sum())
    if ns == 0:
        return dict(n_runs=int(t.size), n_success=0, a=math.inf, b=math.inf, note="inf (0 successes)")
    a = float(c.sum()) / ns
    b = (float(np.sum(t[succ] * c[succ])) + float(np.sum(c[~succ]))) / ns
    return dict(n_runs=int(t.size), n_success=ns, a=a, b=b, note=None)


def walker_step_cost(tau_u, n_steps, N):
    """Exact walker-step version of cost_per_success: full run = n_steps N, success at tau_u = round(tau_u n_steps) N
    (tau_u is a save fraction step / n_steps).  Numerators are exact integers."""
    t = [C.fnum(x) for x in tau_u if not math.isnan(C.fnum(x))]
    full = int(n_steps) * int(N)
    succ_steps = []
    for x in t:
        if math.isfinite(x):
            k = int(round(x * int(n_steps)))
            if abs(k - x * int(n_steps)) > 1e-6 * max(1, int(n_steps)):
                raise C.ExpAError(f"tau_u {x} is not a save fraction of n_steps {n_steps}")
            succ_steps.append(k)
    ns, nf = len(succ_steps), len(t) - len(succ_steps)
    num_a = len(t) * full
    num_b = sum(k * int(N) for k in succ_steps) + nf * full
    if ns == 0:
        return dict(n_runs=len(t), n_success=0, a=math.inf, b=math.inf, numerator_a=num_a, numerator_b=num_b,
                    note="inf (0 successes)")
    return dict(n_runs=len(t), n_success=ns, a=num_a / ns, b=num_b / ns, numerator_a=num_a, numerator_b=num_b,
                note=None)


def _region_mask(name):
    return np.asarray(MA.DEV_REGIONS[name] & MA.EVAL, bool)


def _decomp_point(a, b, reg):
    """Signed and quadratic class decompositions of FR's gate excess over ABF in the bins ``reg`` from the per-bin
    D2 statistics a (FR) and b (ABF) of D2Calc.stats(perbin=True); every array has the leading bootstrap axis."""
    nF, dF, VF, uF = a["Nk"], a["delta"], a["V"], a["use"]
    nA, dA, VA, uA = b["Nk"][:, ALL, :], b["delta"][:, ALL, :], b["V"][:, ALL, :], b["use"][:, ALL, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        # signed: bins of the region where ABF deposited
        msk = reg[None, :] & (nA > 0)
        w = np.where(msk[:, None, :], nF, 0.0)
        den = w[:, ALL, :].sum(-1)
        dpos = den[:, None] > 0
        dsafe = np.where(dpos, den[:, None], 1.0)
        Kc = np.where(dpos, (w * (dF - dA[:, None, :])).sum(-1) / dsafe, np.nan)
        phi = np.where(dpos, w.sum(-1) / dsafe, np.nan)
        tot = np.where(msk, nF[:, ALL, :], 0.0).sum(-1)
        allreg = np.where(reg[None, :], nF[:, ALL, :], 0.0).sum(-1)
        cov = np.where(allreg > 0, tot / np.where(allreg > 0, allreg, 1.0), np.nan)
        # quadratic: bins of the region where ABF's debiased squared deviation is usable (D2 'use': >= 2 seeds);
        # a class enters a bin only where its own is usable; normaliser = FR's deposits in those bins
        mq = reg[None, :] & uA
        Wq = np.where(mq, nF[:, ALL, :], 0.0).sum(-1)
        qpos = Wq[:, None] > 0
        qsafe = np.where(qpos, Wq[:, None], 1.0)
        wq = np.where(mq[:, None, :] & uF, nF, 0.0)
        exc = (dF * dF - VF) - (dA * dA - VA)[:, None, :]
        Qc = np.where(qpos, (wq * exc).sum(-1) / qsafe, np.nan)
        Qsum = np.where(Wq > 0, np.nansum(Qc[:, :N_AGE], axis=1), np.nan)
        cls_reg = np.where(reg[None, None, :], nF, 0.0).sum(-1)
        qcov = np.where(cls_reg > 0, wq.sum(-1) / np.where(cls_reg > 0, cls_reg, 1.0), np.nan)
    return dict(K=Kc, phi=phi, coverage=cov, Q=Qc, Q_sum=Qsum, Q_coverage=qcov)


DECOMP_REGIONS = ("gate", "gate_neg", "gate_pos")
SIGNED_DEF = ("signed gate deviation decomposition: K_c = sum_{j in region & eval, n_ABF,j > 0} n^FR_cj (delta^FR_cj - "
              "delta^ABF_j) / sum_j n^FR_all,j (signed, deposit-weighted; delta = bin mean force - bin-averaged F*'); "
              "sum over the 5 classes = K_all exactly; phi_c = class share of FR's deposits there; coverage = share of "
              "FR's region deposits in bins where ABF deposited.  Opposite-sign bias in different bins (e.g. the two "
              "flanks) CANCELS here: read with the per-flank rows and the quadratic decomposition.  Descriptive, "
              "analysis-defined (PREREGISTRATION 1.5(7) does not define the decomposition)")
QUAD_DEF = ("quadratic (sign-insensitive) decomposition: Q_c = sum_{j in region & eval, ABF usable, class usable} "
            "n^FR_cj [(delta^FR_cj^2 - V^FR_cj) - (delta^ABF_j^2 - V^ABF_j)] / sum_{j in region & eval, ABF usable} "
            "n^FR_all,j, with V the D2 variance of the bin mean (debiased squared bias, as in the matched B_excess); "
            "Q_sum = sum over the 5 classes (exactly additive by construction: the deposit-weighted mean class-resolved "
            "excess squared bias of FR's deposits over ABF's); Q['all'] = the same with FR's pooled bin means (not "
            "additive, reported for reference); Q_coverage = share of the class's region deposits that enter.  "
            "Descriptive, analysis-defined")


def gate_decompositions(calc_fr, calc_abf, n_boot=C.N_BOOT, seed=C.BOOT_SEED, regions=DECOMP_REGIONS):
    """Signed and quadratic clone-age decompositions of FR's excess over ABF in the gate |x| <= 0.5 and in each flank
    (gate_neg x < 0, gate_pos x > 0): point estimates and percentile seed-bootstrap CIs (FR and ABF resampled with the
    same seed weights, mech_analysis.boot_counts(default_rng(seed)))."""
    if calc_fr.S != calc_abf.S:
        raise C.ExpAError("gate decomposition with unaligned seeds")
    masks = {r: _region_mask(r) for r in regions}

    def f(K):
        a = calc_fr.stats(K, full=False, perbin=True)
        b = calc_abf.stats(K, full=False, perbin=True)
        return {r: _decomp_point(a, b, m) for r, m in masks.items()}
    pt = f(np.ones((1, calc_fr.S)))
    boot = None
    if n_boot > 0 and calc_fr.S >= 2:
        Kb = MA.boot_counts(calc_fr.S, n_boot, seed)
        parts = [f(Kb[i:i + MA.BOOT_CHUNK]) for i in range(0, Kb.shape[0], MA.BOOT_CHUNK)]
        boot = {r: {k: np.concatenate([p[r][k] for p in parts]) for k in pt[r]} for r in regions}
    out = dict(classes=CLASS_NAMES, n_seeds=calc_fr.S, calibrated=bool(calc_fr.S >= C.SMALL_N), signed={}, quadratic={},
               definitions=dict(signed=SIGNED_DEF, quadratic=QUAD_DEF))
    for r in regions:
        p, bb = pt[r], (boot or {}).get(r)

        def ci(k):
            if bb is None:
                n = np.shape(p[k])[1:] if np.ndim(p[k]) > 1 else ()
                nan = np.full(n, np.nan) if n else math.nan
                return [nan, nan]
            return [MA.pct(bb[k], 2.5), MA.pct(bb[k], 97.5)]
        out["signed"][r] = dict(region=MA.REGION_LABEL.get(r, r), K=p["K"][0], K_ci95=ci("K"), phi=p["phi"][0],
                                phi_ci95=ci("phi"), coverage=float(p["coverage"][0]),
                                sum_classes_minus_all=float(np.nansum(p["K"][0, :N_AGE]) - p["K"][0, ALL]))
        out["quadratic"][r] = dict(region=MA.REGION_LABEL.get(r, r), Q=p["Q"][0], Q_ci95=ci("Q"),
                                   Q_sum=float(p["Q_sum"][0]), Q_sum_ci95=ci("Q_sum"), Q_coverage=p["Q_coverage"][0],
                                   sum_classes_minus_Q_sum=float(np.nansum(p["Q"][0, :N_AGE]) - p["Q_sum"][0]))
    return out


def gate_excess_decomposition(calc_fr, calc_abf, n_boot=C.N_BOOT, seed=C.BOOT_SEED, region="gate"):
    """The SIGNED decomposition of one region (gate_decompositions) in the flat layout: K, K_ci95, phi, phi_ci95,
    coverage, sum_classes_minus_all, n_seeds, calibrated, definition."""
    d = gate_decompositions(calc_fr, calc_abf, n_boot, seed, regions=(region,))
    sgn = d["signed"][region]
    return dict(classes=CLASS_NAMES, K=sgn["K"], K_ci95=sgn["K_ci95"], phi=sgn["phi"], phi_ci95=sgn["phi_ci95"],
                coverage=sgn["coverage"], sum_classes_minus_all=sgn["sum_classes_minus_all"], n_seeds=d["n_seeds"],
                calibrated=d["calibrated"], definition=SIGNED_DEF, region=region)


# ============================================================================================ run loading
def bin_reference(P):
    """(r, etot): the scorer's bin-averaged F*' (verified against the cell's frozen reference) and the exact
    within-bin variance of f (mech_analysis.analyse_cell's construction)."""
    sc = MF.make_scorer("gateway_family", P)
    fbar, w, s2h, mk = sc.bin_reference(sc.sa)
    e = sc.sa.est(MA.NB, 1)[0]
    ex = MA.exact_bin_force_moments(MA.cell_model(P), e.sub_x.numpy(), w, s2h)
    ev_bins = np.zeros(MA.NB, bool)
    ev_bins[np.unique(s2h[mk])] = True
    if not np.array_equal(ev_bins, MA.EVAL):
        raise C.ExpAError("scorer eval bins != the 150 bins with centres in (-1.5, 1.5)")
    dmax = float(np.max(np.abs(ex["fbar"] - fbar)))
    if dmax > 1e-9:
        raise C.ExpAError(f"exact bin-averaged F*' disagrees with the scorer's bin reference by {dmax:.3g}")
    return dict(r=np.asarray(fbar, float), etot=ex["etot"], check_max_abs=dmax)


def load_extra(path, meta, P, N, method):
    """Genealogy arrays and the g4 checks of one complete run."""
    with np.load(path, allow_pickle=False) as z:
        a = {k: np.asarray(z[k]) for k in EXTRA_KEYS if k in z.files}
    probs = []
    for k in FINITE_KEYS + (GEN_KEYS if method == "fr" else ()):
        if k not in a:
            probs.append(f"{k} missing")
        elif not np.all(np.isfinite(np.asarray(a[k], float))):
            probs.append(f"{k} holds non-finite values")
    ec = P["engine_cfg"]
    cap_rule = max(int(ec["cap_min"]), int(math.floor(float(ec["max_event_fraction"]) * int(N))))
    cap = int(meta.get("cap", -1))
    hist = np.asarray(a.get("fr_deaths_hist", np.zeros(0)), np.int64).ravel()
    deaths = int(np.asarray(a.get("fr_deaths_cum", [0]))[-1])
    opps = int(np.asarray(a.get("fr_opp_cum", [0]))[-1])
    if cap != cap_rule:
        probs.append(f"cap {cap} != max(cap_min, floor(0.08 N)) = {cap_rule}")
    if hist.size != cap + 1:
        probs.append(f"realised-death histogram has {hist.size} bins, cap + 1 = {cap + 1}")
    elif int(hist.sum()) != opps or int((np.arange(hist.size) * hist).sum()) != deaths:
        probs.append("realised-death histogram inconsistent with the opportunity / death counters")
    if method == "abf" and deaths != 0:
        probs.append(f"ABF arm with {deaths} FR deaths")
    if meta.get("engine_sha256") != C.ENGINE_SHA256:
        probs.append(f"engine sha256 {str(meta.get('engine_sha256'))[:16]} != the frozen {C.ENGINE_SHA256[:16]}...")
    n_steps = int(meta["n_steps"])
    W = int(meta.get("window_steps", 0))
    if method == "fr" and "gen_ess_win" in a:
        ess_t = ess_time_average(a["save_step"], a["gen_ess_win"], n_steps, W)
        mf = np.asarray(a["gen_maxfam_win"], float)
    else:
        ess_t, mf = math.nan, np.full(1, np.nan)
    return dict(ess_win_tavg=ess_t, maxfam_win_max=float(np.nanmax(mf)) if np.any(np.isfinite(mf)) else math.nan,
                g4_problems=probs, window_steps=W)


def collect(rec, N):
    """{method: {seed: run}} of the complete, plan-checked runs (with D2 + genealogy), status counts, the sha256 of
    EVERY file whose meta status is 'complete' (the same set run_shas_of hashes, whether or not the run then passes
    the plan check and the D1-D3 identities: those runs are reported through g4, never dropped from the signature)
    and the engine builds of the loaded runs."""
    P = rec["P"]
    root = MF._abs(P["results_dir"])
    runs, status, shas, engines = {"abf": {}, "fr": {}}, {}, {}, set()
    for m in ("abf", "fr"):
        st = dict(planned=len(P["seeds"]), complete=[], missing=[], running=[], invalid={}, diag_problems={})
        for s in [int(x) for x in P["seeds"]]:
            path = os.path.join(root, f"N{int(N)}", f"s{s}_{m}.npz")
            if MA.file_status(path)[0] == "complete":
                shas[C.rel(path)] = C.sha256_file(path)
            r = MA.load_run(path, P, int(N), s, m)
            if r["status"] == "complete":
                engines.add((r["meta"].get("engine"), r["meta"].get("engine_sha256")))
                if not r["has_diag"] or r["diag_problems"]:
                    st["diag_problems"][s] = r["diag_problems"] or [f"missing {r['diag_missing']}"]
                    continue
                r.update(load_extra(path, r["meta"], P, N, m))
                runs[m][s] = r
                st["complete"].append(s)
            elif r["status"] == "missing":
                st["missing"].append(s)
            elif r["status"] == "running":
                st["running"].append(s)
            else:
                st["invalid"][s] = r.get("why")
        status[m] = st
    return runs, status, shas, engines


def class_names_of(runs):
    for m in ("fr", "abf"):
        for r in runs[m].values():
            return list(r["meta"].get("diag_classes", {}).get("clone_age_class_names", MA.AGE_DEFAULT))
    return list(MA.AGE_DEFAULT)


# ============================================================================================ assembly
def d2_block(runs, ref, n_boot):
    common = sorted(set(runs["fr"]) & set(runs["abf"]))
    if len(common) < 2:
        return dict(missing=f"fewer than 2 seeds with D2 in both arms ({common})"), None
    arrays = {m: MA.d2_arrays([runs[m][s] for s in common], -1) for m in ("fr", "abf")}
    cf = MA.D2Calc(*arrays["fr"], ref["r"], ref["etot"])
    ca = MA.D2Calc(*arrays["abf"], ref["r"], ref["etot"])
    sf, sa = MA.d2_summarise(cf, n_boot), MA.d2_summarise(ca, n_boot)
    pv = MA.d2_paired(cf, list(range(N_AGE + 1)), ca, [ALL] * (N_AGE + 1), n_boot, matched=True)
    regs = list(MA.DEV_REGIONS)
    gi = regs.index("gate")
    matched = {}
    for c in range(N_AGE + 1):
        p = pv[c]
        matched[CLASS_NAMES[c]] = dict(
            B_excess=p["B_excess"], B_excess_ci95=p["B_excess_ci95"], B_excess_p=p["B_excess_p"],
            Bm_a=p["Bm_a"], Bm_a_ci95=p["Bm_a_ci95"], Bm_b=p["Bm_b"], Bm_b_ci95=p["Bm_b_ci95"],
            dev_excess_gate=p["dev_excess"][gi], dev_excess_gate_ci95=[p["dev_excess_ci95"][0][gi],
                                                                       p["dev_excess_ci95"][1][gi]],
            matched_coverage=p["matched_coverage"], p_calibrated=p["p_calibrated"])
    blk = dict(n_seeds=len(common), seeds=common, classes=CLASS_NAMES,
               fr=dict(B=sf["B"], B_ci95=[sf["B_lo"], sf["B_hi"]], coverage=sf["coverage"],
                       dev_gate=sf["dev"][:, gi], dev_gate_ci95=[sf["dev_lo"][:, gi], sf["dev_hi"][:, gi]]),
               abf=dict(B=float(sa["B"][ALL]), B_ci95=[float(sa["B_lo"][ALL]), float(sa["B_hi"][ALL])],
                        dev_gate=float(sa["dev"][ALL, gi])),
               fr_vs_abf_matched=matched, young_clone_class=CLASS_NAMES[YOUNG],
               young_clone=dict(B=float(sf["B"][YOUNG]), B_ci95=[float(sf["B_lo"][YOUNG]), float(sf["B_hi"][YOUNG])],
                                **matched[CLASS_NAMES[YOUNG]]),
               calibrated=bool(len(common) >= C.SMALL_N))
    dec = gate_decompositions(cf, ca, n_boot)
    sg = dec["signed"]["gate"]
    blk["gate_decomposition"] = dec
    blk["gate_excess_decomposition"] = dict(        # the pooled-gate SIGNED rows (flat layout; see gate_decomposition)
        classes=CLASS_NAMES, K=sg["K"], K_ci95=sg["K_ci95"], phi=sg["phi"], phi_ci95=sg["phi_ci95"],
        coverage=sg["coverage"], sum_classes_minus_all=sg["sum_classes_minus_all"], n_seeds=dec["n_seeds"],
        calibrated=dec["calibrated"], definition=SIGNED_DEF, region="gate")
    return blk, dict(seeds=common, arrays=arrays)


def mc3_block(keep, refs, Ns, n_boot):
    out = {}
    for lam in HI_LAMS:
        rows = {}
        for N in Ns:
            a, b = keep.get((lam, N)), keep.get((C.REF_LAM, N))
            if a is None or b is None:
                rows[str(N)] = dict(missing="D2 arrays of one of the cells are missing")
                continue
            common = sorted(set(a["seeds"]) & set(b["seeds"]))
            if len(common) < 2:
                rows[str(N)] = dict(missing=f"fewer than 2 common seeds ({common})")
                continue

            def calc(k, m, r):
                idx = [k["seeds"].index(s) for s in common]
                return MA.D2Calc(*(x[idx] for x in k["arrays"][m]), r["r"], r["etot"])
            cfa, caa = calc(a, "fr", refs[lam]), calc(a, "abf", refs[lam])
            cfb, cab = calc(b, "fr", refs[C.REF_LAM]), calc(b, "abf", refs[C.REF_LAM])
            p = MA.d2_paired(cfa, YOUNG, cfb, YOUNG, n_boot, calc_c=caa, cc=ALL, calc_d=cab, cd=ALL, matched=True)
            rows[str(N)] = dict(n_seeds=len(common), p_calibrated=p["p_calibrated"],
                                B_excess_did=p["B_excess_did"], B_excess_did_ci95=p["B_excess_did_ci95"],
                                B_excess_did_p=p["B_excess_did_p"], excess_hi=p["B_excess_a"],
                                excess_hi_ci95=p["B_excess_a_ci95"], excess_ref=p["B_excess_b"],
                                excess_ref_ci95=p["B_excess_b_ci95"])
        ps = [rows[str(N)].get("B_excess_did_p", math.nan) for N in Ns]
        adj = C.holm(ps, m=len(Ns)) if Ns else []
        for N, ph in zip(Ns, adj):
            r = rows[str(N)]
            if "missing" in r:
                r.update(p_holm=math.nan, passes=None)
                continue
            r["p_holm"] = ph
            r["passes"] = (None if not r["p_calibrated"] else
                           bool(ph < C.ALPHA and C.fnum(r["B_excess_did"]) < 0))
        out[C.lam_key(lam)] = dict(per_N=rows, passes_all_N=(all(rows[str(N)].get("passes") is True for N in Ns)
                                                              if Ns else None),
                                   passes_any_N=any(rows[str(N)].get("passes") is True for N in Ns))
    return out


def genealogy_block(runs, S, N, n_boot):
    fr = runs["fr"]
    ess = {s: r["ess_win_tavg"] for s, r in fr.items()}
    mf = {s: r["maxfam_win_max"] for s, r in fr.items()}
    sm = C.per_seed(S, N, "fr", "max_gen_maxfam_win") if S else {}
    mism = [s for s in mf if s in sm and not (math.isclose(mf[s], sm[s], rel_tol=0, abs_tol=1e-12))]
    blk = dict(n_fr=len(fr), ess_win_tavg=C.boot_median_inf(ess, n_boot), maxfam_win_max=C.boot_median_inf(mf, n_boot),
               per_seed=dict(ess_win_tavg=ess, maxfam_win_max=mf),
               maxfam_matches_summary=(not mism), maxfam_mismatch_seeds=mism)
    for k in ("min_gen_ess_win", "final_gen_nuniq_run_frac", "fr_deaths_per_walker_per_t", "fr_deaths_per_walker"):
        blk[k] = C.boot_median_inf(C.per_seed(S, N, "fr", k), n_boot) if S else None
    return blk, ess, mf


def g4_block(runs, status, S, N):
    probs = []
    for m in ("abf", "fr"):
        for s, r in runs[m].items():
            probs += [f"{m} seed {s}: {p}" for p in r["g4_problems"]]
        for s, ps in status[m]["diag_problems"].items():
            probs.append(f"{m} seed {s}: D1-D3 diagnostics {ps}")
        for s, why in status[m]["invalid"].items():
            probs.append(f"{m} seed {s}: invalid run ({why})")
        if S is not None:
            for k in NAN_KEYS:
                for s, v in C.per_seed(S, N, m, k).items():
                    if C.fnum(v) != 0:
                        probs.append(f"{m} seed {s}: {k} = {v}")
    n_runs = sum(len(runs[m]) for m in ("abf", "fr"))
    holds = None if n_runs == 0 else (not probs)
    return dict(holds=holds, n_runs_checked=n_runs, problems=probs[:50], n_problems=len(probs))


def abf_descriptors(cells, Ns, n_boot):
    out = {}
    for key, label in ABF_DESCRIPTORS:
        blk = dict(label=label, per_lam={})
        for lam, rec in sorted(cells.items()):
            if rec["S"] is None:
                continue
            for N in Ns:
                v = C.per_seed(rec["S"], N, "abf", key)
                e = dict(C.boot_median_inf(v, n_boot))
                if lam != C.REF_LAM and C.REF_LAM in cells and cells[C.REF_LAM]["S"] is not None:
                    e["ratio_vs_lam1"] = C.ratio_of_medians(v, C.per_seed(cells[C.REF_LAM]["S"], N, "abf", key), n_boot)
                blk["per_lam"].setdefault(C.lam_key(lam), {})[str(N)] = e
        out[key] = blk
    return out


def cost_block(cells, Ns):
    out = {}
    for lam, rec in sorted(cells.items()):
        S = rec["S"]
        if S is None:
            continue
        for N in Ns:
            n_steps = int(rec["P"]["B"]) // int(N)
            for m in ("abf", "fr"):
                wall = C.per_seed(S, N, m, "wall_s")
                d = {}
                for key, metric, thr in TAU_THRESHOLDS:
                    tau = C.per_seed(S, N, m, key)
                    seeds = sorted(s for s in tau if s in wall)
                    if not seeds:
                        continue
                    t = [tau[s] for s in seeds]
                    d[key] = dict(metric=metric, threshold=thr,
                                  walker_steps=walker_step_cost(t, n_steps, N),
                                  core_seconds=cost_per_success(t, [wall[s] for s in seeds]))
                out.setdefault(C.lam_key(lam), {}).setdefault(str(N), {})[m] = d
    return out


def replication_block(cells, exp1_path, n_boot):
    rec = cells.get(C.REF_LAM)
    if rec is None or rec["S"] is None:
        return dict(missing="no lambda = 1 summary")
    if not os.path.exists(exp1_path):
        return dict(missing=f"no Experiment I summary at {exp1_path}")
    import json
    E1 = json.load(open(exp1_path))
    model = (E1.get("cell") or {}).get("model") or {}
    if model.get("variant") != "shift" or C.fnum(model.get("lam")) != 1.0 or C.fnum(model.get("kappa")) != 1.0:
        return dict(missing=f"{exp1_path} is not the Experiment I shifted fibre (model {model})")
    S = rec["S"]
    Ns = [int(n) for n in rec["P"]["N_ladder"] if str(n) in (E1.get("per_N") or {})]
    out = dict(exp1_summary=C.rel(exp1_path), exp1_summary_sha256=C.sha256_file(exp1_path),
               exp1_seeds=[int(s) for s in (E1.get("plan") or {}).get("seeds", [])],
               expA_seeds=[int(s) for s in rec["P"]["seeds"]], N=Ns, per_N={},
               note="descriptive, not tested (PREREGISTRATION 1.6): fresh seeds, same dynamics and settings")
    for N in Ns:
        for metric in REPL_METRICS:
            row = {}
            for tag, SS in (("exp1_shift", E1), ("expA_lam1", S)):
                g = C.summary_contrast(SS, N, metric) or {}
                row[tag] = dict(abf_median=C.fnum(g.get("abf_median")), fr_median=C.fnum(g.get("fr_median")),
                                G_median=C.fnum(g.get("G_median")),
                                G_ci95=[C.fnum(x) for x in (g.get("G_ci95") or [None, None])],
                                wins=g.get("wins"), n=g.get("n"))
            out["per_N"].setdefault(str(N), {})[metric] = row
    return out


def analyse(cells, exp1_path=C.EXP1_SHIFT_SUMMARY, n_boot=C.N_BOOT):
    production, fixture, design_probs = C.design_status(cells)
    Ns = C.common_N(cells)
    doc = dict(schema=SCHEMA, n_boot=int(n_boot), boot_seed=C.BOOT_SEED, N=Ns, lams=[C.lam_key(l) for l in C.LAMS],
               production_design=production, fixture=fixture, design_problems=design_probs, small_n=C.SMALL_N,
               cells={}, d2={}, genealogy={}, guards={}, ha3=dict(cells={}), warnings=[], run_files={},
               engine_problems=[])
    if fixture:
        doc["warnings"].append("FIXTURE cells: synthetic tiny-budget data, never production evidence")
    refs, keep, gen = {}, {}, {}
    all_complete = True
    for lam, rec in sorted(cells.items()):
        lk = C.lam_key(lam)
        refs[lam] = bin_reference(rec["P"])
        info = dict(config=C.rel(rec["config"]), config_sha256=C.sha256_file(rec["config"]), cell=rec["name"],
                    summary=C.rel(rec["summary_path"]), summary_available=rec["S"] is not None,
                    summary_sha256=(C.sha256_file(rec["summary_path"]) if rec["S"] is not None else None),
                    per_N={}, engines=[])
        engines = set()
        for N in Ns:
            runs, status, shas, eng = collect(rec, N)
            engines |= eng
            doc["run_files"].update(shas)
            info["per_N"][str(N)] = {m: dict(complete=len(status[m]["complete"]), planned=status[m]["planned"],
                                             missing=status[m]["missing"], running=status[m]["running"],
                                             invalid=status[m]["invalid"], diag_problems=status[m]["diag_problems"])
                                     for m in ("abf", "fr")}
            all_complete &= all(len(status[m]["complete"]) == status[m]["planned"] for m in ("abf", "fr"))
            names = class_names_of(runs)
            if names != list(MA.AGE_DEFAULT):
                raise C.ExpAError(f"lambda {lam:g} N {N}: clone-age classes {names} != {MA.AGE_DEFAULT}")
            blk, k = d2_block(runs, refs[lam], n_boot)
            doc["d2"].setdefault(lk, {})[str(N)] = blk
            if k is not None:
                keep[(lam, N)] = k
            gb, ess, mf = genealogy_block(runs, rec["S"], N, n_boot)
            doc["genealogy"].setdefault(lk, {})[str(N)] = gb
            gen[(lam, N)] = (ess, mf)
            if not gb["maxfam_matches_summary"]:
                doc["warnings"].append(f"lambda {lk} N {N}: max windowed lineage fraction differs from the summary "
                                       f"for seeds {gb['maxfam_mismatch_seeds']}")
            S = rec["S"]
            gd = dict(g4=g4_block(runs, status, S, N))
            for gname, metric in (("g1", "final_e_F"), ("g2", "Ibar_Fp_stat")):
                g = C.summary_contrast(S, N, metric) if S else None
                ci = [C.fnum(x) for x in ((g or {}).get("G_ci95") or [None, None])]
                gd[gname] = dict(holds=ci_guard(ci) if g else None, metric=metric, G_median=C.fnum((g or {}).get("G_median")),
                                 G_ci95=ci, rule="fails iff the 95 % CI of G lies entirely above 0")
            doc["guards"].setdefault(lk, {})[str(N)] = gd
            g = C.summary_contrast(S, N, "Ibar_TV_half") if S else None
            hi = C.fnum(((g or {}).get("G_ci95") or [None, None])[1])
            doc["ha3"]["cells"].setdefault(lk, {})[str(N)] = dict(
                holds=(None if math.isnan(hi) else bool(hi < 0)), G_median=C.fnum((g or {}).get("G_median")),
                G_ci95=[C.fnum(x) for x in ((g or {}).get("G_ci95") or [None, None])], n=(g or {}).get("n"))
        info["engines"] = sorted(f"{e} {s}" for e, s in engines)
        bad_eng = sorted(f"{e} {s}" for e, s in engines if s != C.ENGINE_SHA256)
        if bad_eng:
            msg = f"lambda {lk}: runs from an engine other than the frozen sha256 {C.ENGINE_SHA256[:12]}...: {bad_eng}"
            doc["warnings"].append(msg)
            doc["engine_problems"].append(msg)
        doc["cells"][lk] = info
    # g3 needs lambda 1
    for lam in HI_LAMS:
        for N in Ns:
            gd = doc["guards"].get(C.lam_key(lam), {}).get(str(N))
            if gd is None:
                continue
            if (lam, N) in gen and (C.REF_LAM, N) in gen:
                (eh, mh), (er, mr) = gen[(lam, N)], gen[(C.REF_LAM, N)]
                gd["g3"] = g3_rule(eh, er, mh, mr, n_boot)
            else:
                gd["g3"] = dict(holds=None, note="lambda 1 or this cell has no FR runs")
    hs = [v["holds"] for byN in doc["ha3"]["cells"].values() for v in byN.values()]
    doc["ha3"]["persists"] = (None if (not hs or any(h is None for h in hs) or len(hs) != len(C.LAMS) * len(Ns))
                              else all(hs))
    doc["ha3"]["rule"] = ("HA3 holds at (lambda, N) iff the upper 95 % CI of G(Ibar_TV_half) < 0; persists iff it holds "
                          "in every cell (null when a cell is missing or undefined)")
    doc["mc3"] = mc3_block(keep, refs, Ns, n_boot)
    doc["abf_descriptors"] = abf_descriptors(cells, Ns, n_boot)
    doc["cost"] = cost_block(cells, Ns)
    doc["replication"] = replication_block(cells, exp1_path, n_boot)
    doc["runs_complete"] = bool(all_complete and all(l in cells for l in C.LAMS))
    doc["definitions"] = dict(
        d2="mech_analysis D2 at the end snapshot (seeds with D2 in both arms); B: debiased RMS with the reflected CI",
        matched_excess="d2_paired(FR class c, ABF all, matched=True): B_excess = B_c|J - B_ABF|c",
        mc3="B_excess_did = young-clone matched excess at lambda_hi minus at lambda 1; Holm over N; passes iff Holm p < "
            "0.05 and estimate < 0",
        genealogy="ess_win_tavg = mean gen_ess_win over the 200 uniform-u saves with step >= genealogy window; "
                  "maxfam_win_max = max gen_maxfam_win over all saves",
        cost="(a) sum full / successes; (b) (sum_success tau cost + sum_failure full) / successes; walker-steps exact "
             "(numerators integer), core-seconds from wall_s",
        abf_ratio="ratio of inf-aware seed medians lambda / lambda 1, paired bootstrap, censoring conventions")
    return doc


def input_signature(cells, run_shas, exp1_path, n_boot, audits=None, provisional=False, verify_runs=True):
    ins = {C.lam_key(l): dict(config=C.sha256_file(r["config"]),
                              summary=(C.sha256_file(r["summary_path"]) if r["S"] is not None else None))
           for l, r in sorted(cells.items())}
    return C.digest(dict(schema=SCHEMA, inputs=ins, runs=dict(sorted(run_shas.items())),
                         exp1=(C.sha256_file(exp1_path) if os.path.exists(exp1_path) else None),
                         code=C.code_digests(CODE_FILES), n_boot=int(n_boot), audits=C.audit_signature(audits or {}),
                         provisional=bool(provisional), verify_runs=bool(verify_runs)))


def run_shas_of(cells):
    """sha256 of every complete run file of every cell (the signature input; read before deciding to skip)."""
    out = {}
    for lam, rec in sorted(cells.items()):
        root = MF._abs(rec["P"]["results_dir"])
        for N in rec["P"]["N_ladder"]:
            for m in ("abf", "fr"):
                for s in rec["P"]["seeds"]:
                    p = os.path.join(root, f"N{int(N)}", f"s{int(s)}_{m}.npz")
                    if MA.file_status(p)[0] == "complete":
                        out[C.rel(p)] = C.sha256_file(p)
    return out


# ============================================================================================ markdown
def md(doc):
    f = C.fmt
    L = ["# Experiment A: mechanism read-outs, guards and descriptors", "",
         f"Generated {doc.get('provenance', {}).get('generated_utc')} by scripts/final_campaign/expA_mech.py ({SCHEMA}); "
         f"{doc['n_boot']} seed-bootstrap resamples.  Complete: {doc.get('complete')}.", ""]
    if doc.get("fixture"):
        L += ["**FIXTURE INPUT: synthetic tiny-budget data, never production evidence.**", ""]
    comp = doc.get("completeness") or {}
    if comp.get("reasons"):
        L += ["Not complete because:", ""] + [f"* {r}" for r in comp["reasons"][:20]] + [""]
    L += ["## MC3: change from lambda 1 of the young-clone ([0.01, 0.1) t.u.) matched excess over ABF", "",
          "| lambda | N | n | B_excess_did [95 % CI] | p | p Holm (over N) | passes | excess at lambda | excess at 1 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for lk, blk in doc["mc3"].items():
        for N, r in blk["per_N"].items():
            if "missing" in r:
                L.append(f"| {lk} | {N} | -- | MISSING: {r['missing']} | | | | | |")
                continue
            L.append(f"| {lk} | {N} | {r['n_seeds']} | {f(r['B_excess_did'], 4)} {C.fmt_ci(r['B_excess_did_ci95'], 4)} | "
                     f"{f(r['B_excess_did_p'])} | {f(r['p_holm'])} | {r['passes']} | {f(r['excess_hi'], 4)} | "
                     f"{f(r['excess_ref'], 4)} |")
    L += ["", "## Guards (PREREGISTRATION 1.7) and HA3", "",
          "| lambda | N | g1 final e_F | g2 Ibar_F'_stat | g3 genealogy (ESS ratio; max-lineage ratio) | g4 | HA3 G(Ibar_TV_half) |",
          "|---|---|---|---|---|---|---|"]
    for lk, byN in doc["guards"].items():
        for N, g in byN.items():
            g3 = g.get("g3") or {}
            g3s = ("--" if not g3 or g3.get("holds") is None and "ess_ratio" not in g3 else
                   f"{g3.get('holds')} ({f((g3.get('ess_ratio') or {}).get('median'), 3)} "
                   f"{C.fmt_ci((g3.get('ess_ratio') or {}).get('ci95'))}; {f((g3.get('maxfam_ratio') or {}).get('median'), 3)} "
                   f"{C.fmt_ci((g3.get('maxfam_ratio') or {}).get('ci95'))})")
            h = doc["ha3"]["cells"].get(lk, {}).get(N, {})
            L.append(f"| {lk} | {N} | {g['g1']['holds']} {C.fmt_ci(g['g1']['G_ci95'])} | {g['g2']['holds']} "
                     f"{C.fmt_ci(g['g2']['G_ci95'])} | {g3s} | {g['g4']['holds']} ({g['g4']['n_problems']} problems) | "
                     f"{h.get('holds')} {C.fmt_ci(h.get('G_ci95'))} |")
    L += ["", f"HA3 persists: {doc['ha3'].get('persists')}", "",
          "## ABF-only cross-lambda descriptors (seed median [95 % CI]; ratio of medians vs lambda 1)", "",
          "| descriptor | lambda | N | median [CI] | ratio vs 1 [CI] (undefined frac) |", "|---|---|---|---|---|"]
    for key, blk in doc["abf_descriptors"].items():
        for lk, byN in blk["per_lam"].items():
            for N, e in byN.items():
                r = e.get("ratio_vs_lam1")
                rs = f"{f(r['ratio'], 4)} {C.fmt_ci(r['ci95'], 4)} ({f(r['undefined_frac'], 2)})" if r else "--"
                L.append(f"| {key} | {lk} | {N} | {f(e['median'], 4)} {C.fmt_ci(e['ci95'], 4)} | {rs} |")
    L += ["", "## Genealogy (FR arm; seed median [95 % CI])", "",
          "| lambda | N | ESS/N time average | max lineage fraction | min ESS/N | deaths / walker / t.u. |",
          "|---|---|---|---|---|---|"]
    for lk, byN in doc["genealogy"].items():
        for N, g in byN.items():
            def mc(x):
                return f"{f(x['median'], 4)} {C.fmt_ci(x['ci95'], 4)}" if x else "--"
            L.append(f"| {lk} | {N} | {mc(g['ess_win_tavg'])} | {mc(g['maxfam_win_max'])} | {mc(g.get('min_gen_ess_win'))} | "
                     f"{mc(g.get('fr_deaths_per_walker_per_t'))} |")
    L += ["", "## Young clones (FR) and the gate decompositions by clone age", "",
          "| lambda | N | n | young B_c [CI] | young matched excess [CI] | ABF B |", "|---|---|---|---|---|---|"]
    for lk, byN in doc["d2"].items():
        for N, b in byN.items():
            if "missing" in b:
                L.append(f"| {lk} | {N} | -- | MISSING: {b['missing']} | | |")
                continue
            y = b["young_clone"]
            L.append(f"| {lk} | {N} | {b['n_seeds']} | {f(y['B'], 4)} {C.fmt_ci(y['B_ci95'], 4)} | {f(y['B_excess'], 4)} "
                     f"{C.fmt_ci(y['B_excess_ci95'], 4)} | {f(b['abf']['B'], 4)} |")
    L += ["", "Signed gate deviation decomposition K_c (sum over the 5 classes = all; opposite-sign bias cancels: read "
          "the flank rows and the quadratic rows) and quadratic decomposition Q_c (debiased excess squared bias over "
          "ABF; Q_sum = sum of the 5 classes).  Descriptive, analysis-defined.", "",
          "| lambda | N | region | kind | " + " | ".join(CLASS_NAMES) + " | total |",
          "|---|---|---|---|" + "---|" * (len(CLASS_NAMES) + 1)]
    for lk, byN in doc["d2"].items():
        for N, b in byN.items():
            dec = b.get("gate_decomposition")
            if not dec:
                continue
            for r in DECOMP_REGIONS:
                sg, qd = dec["signed"][r], dec["quadratic"][r]
                L.append(f"| {lk} | {N} | {r} | signed K | " + " | ".join(f(k, 3) for k in sg["K"]) +
                         f" | {f(sg['K'][ALL], 3)} |")
                L.append(f"| {lk} | {N} | {r} | quadratic Q | " + " | ".join(f(q, 3) for q in qd["Q"]) +
                         f" | {f(qd['Q_sum'], 3)} {C.fmt_ci(qd['Q_sum_ci95'], 3)} |")
    L += ["", "## Cost per successful threshold, mid thresholds (a) / (b)", "",
          "| lambda | N | arm | threshold | successes / runs | walker-steps (a) | (b) | core-s (a) | (b) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for lk, byN in doc["cost"].items():
        for N, arms in byN.items():
            for m, d in arms.items():
                for key in ("tau_e_F_mid_u", "tau_e_Fp_stat_mid_u", "tau_TV_half_u"):
                    if key not in d:
                        continue
                    w, c = d[key]["walker_steps"], d[key]["core_seconds"]
                    L.append(f"| {lk} | {N} | {m} | {key[4:-2]} | {w['n_success']}/{w['n_runs']} | {f(w['a'], 4)} | "
                             f"{f(w['b'], 4)} | {f(c['a'], 4)} | {f(c['b'], 4)} |")
    rp = doc.get("replication") or {}
    L += ["", "## Replication: lambda = 1 vs Experiment I shifted fibre (descriptive)", ""]
    if "missing" in rp:
        L.append(f"MISSING: {rp['missing']}")
    else:
        L += ["| N | metric | Exp I G [CI] (wins/n) | Exp A lambda 1 G [CI] (wins/n) | Exp I ABF / FR median | "
              "Exp A ABF / FR median |", "|---|---|---|---|---|---|"]
        for N, byM in rp["per_N"].items():
            for metric, row in byM.items():
                a, b = row["exp1_shift"], row["expA_lam1"]
                L.append(f"| {N} | {metric} | {100 * C.fnum(a['G_median']):+.1f} % {C.fmt_ci([100 * C.fnum(x) for x in a['G_ci95']])}"
                         f" ({a['wins']}/{a['n']}) | {100 * C.fnum(b['G_median']):+.1f} % "
                         f"{C.fmt_ci([100 * C.fnum(x) for x in b['G_ci95']])} ({b['wins']}/{b['n']}) | "
                         f"{f(a['abf_median'], 4)} / {f(a['fr_median'], 4)} | {f(b['abf_median'], 4)} / {f(b['fr_median'], 4)} |")
    if doc["warnings"]:
        L += ["", "## Warnings", ""] + [f"* {w}" for w in doc["warnings"]]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", default=C.CELLS_DIR)
    ap.add_argument("--out-dir", default=C.SYN_DIR)
    ap.add_argument("--exp1-summary", default=C.EXP1_SHIFT_SUMMARY)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--no-verify-runs", action="store_true",
                    help="skip the summaries' run-tree freshness check (forbids complete)")
    ap.add_argument("--provisional", action="store_true",
                    help="force complete = false (the synthesis runs over a failed completeness audit)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cells = C.load_cells(a.cells_dir)
    if not cells:
        raise C.ExpAError(f"no cell config under {a.cells_dir}")
    out = os.path.abspath(a.out_dir)
    jp = os.path.join(out, "mech.json")
    t0 = time.time()
    verify = not a.no_verify_runs
    shas = run_shas_of(cells)
    comp = C.completeness(cells, a.n_boot, verify, a.provisional)
    sig = input_signature(cells, shas, a.exp1_summary, a.n_boot, comp["audits"], a.provisional, verify)
    if a.dry_run:
        sp = C.summary_problems(cells, verify)
        try:
            decision = C.decide_write(jp, sig, a.force, a.recompute)
        except C.ExpAError as e:
            decision = f"REFUSE ({e})"
        print(f"{len(shas)} complete run files hashed in {time.time() - t0:.1f} s; complete-eligible (before reading "
              f"runs) {comp['complete']}; would {decision} {jp} (signature {sig[:16]})"
              + (f"; summaries not current for lambda {[f'{l:g}' for l in sp]}: would stop" if sp else ""))
        return 0
    C.require_current_summaries(cells, verify)
    decision = C.decide_write(jp, sig, a.force, a.recompute)
    if decision == "skip":
        print(f"{jp} is up to date (signature {sig[:16]}); nothing to do (--recompute rewrites it)")
        return 0
    prev = C.previous_signature(jp)
    doc = analyse(cells, a.exp1_summary, a.n_boot)
    if set(doc["run_files"]) != set(shas) or any(doc["run_files"][k] != shas[k] for k in shas):
        changed = sorted(set(doc["run_files"]) ^ set(shas)) + sorted(k for k in set(doc["run_files"]) & set(shas)
                                                                     if doc["run_files"][k] != shas[k])
        raise C.ExpAError(f"run files changed while the analysis read them ({changed[:4]}); re-run")
    extra = []
    if not doc["runs_complete"]:
        extra.append("not every planned run was loaded with D2 (missing, running, invalid or D1-D3 problems: see "
                     "cells.<lambda>.per_N)")
    extra += doc["engine_problems"]
    comp = C.completeness(cells, a.n_boot, verify, a.provisional, extra_reasons=extra)
    doc.update(input_signature=sig, provenance=C.provenance(CODE_FILES), cells_dir=C.rel(a.cells_dir),
               complete=comp["complete"], completeness=comp, provisional=bool(a.provisional),
               freshness_verified=verify, wall_s=time.time() - t0)
    if prev and prev != sig:
        doc["replaced_signature"] = prev
    C.begin_output(jp, SCHEMA, prev)
    C.write_text(os.path.join(out, "mech.md"), md(C.reload_json(M.json_safe(doc))))
    C.write_json(jp, doc)
    for lk, blk in doc["mc3"].items():
        for N, r in blk["per_N"].items():
            print(f"MC3 lambda {lk} vs 1, N {N}: {C.fmt(r.get('B_excess_did'), 4)} {C.fmt_ci(r.get('B_excess_did_ci95'), 4)}"
                  f" Holm p {C.fmt(r.get('p_holm'))} -> passes {r.get('passes')}", flush=True)
    print(f"wrote {jp} and mech.md in {time.time() - t0:.1f} s (complete {doc['complete']})", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except C.ExpAError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
