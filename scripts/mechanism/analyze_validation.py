#!/usr/bin/env python
"""Mechanism campaign: numerical-validation gate for Experiments I and II (docs/mechanism/SCIENTIFIC_PLAN.md section 3,
thresholds of configs/equal_budget_v2/gateway_validation.json) -> <res>/summary.json and
figures/mechanism/validation/exp1_timestep_validation.{pdf,png}.

    python scripts/mechanism/analyze_validation.py [--res results/mechanism/validation] [--fig figures/mechanism/validation]
           [--abf-smoke <res>/abf_smoke.json] [--smoke]

Data: scripts/mechanism/run_validation.py.  Every statistic is a jackknife over the independent groups (16 in the gate);
with fewer than 2 groups the statistic is NaN and its gate reads NO_DATA, never PASS.

Flat-bias design (as the frozen gateway gate): each chain targets exp(-beta (V - F*(x))), whose x-marginal is uniform
for EVERY model (F_model = F* + C), so  F_h(x) - F*(x) = -kT log p_h(x) + const  (density route) and the conditional
mean force <d_x V | x>_h estimates F_h'(x) (mean-force route); exact: p uniform, <d_x V | x> = F*'(x).

Design check (review fix 2026-10-10): every record set must match the frozen design of run_validation.py
(SCIENTIFIC_PLAN section 3): exactly 16 groups with ids 0-15, the preregistered seed of each group, 256 walkers,
T = 500/lam t.u. (MALA 500) after 10 t.u. burn-in, h equal to the tag, flat bias on (unbiased runs: 8 walkers, 200 t.u.),
exact draws 2^24 (flat) / 2^22 (unbiased), meta.smoke false.  A set that deviates is EXCLUDED (its gates read NO_DATA,
never PASS) and the deviations are listed under dynamics.<d>.design; with --smoke the deviations are only listed.

Gates per dynamics d and timestep h (PASS / FAIL / NO_DATA; FAIL wins over NO_DATA):
  V1  closed form 1/(1 - lam k_max h/2) - 1 <= 0.02, and in the two central fine bins (width 0.0025) the measured
      relative inflation of Var(Y|x) [residual y - E[Y|x]] satisfies m <= 0.02 + 2 se -- the plan's own definition of
      "consistent" (SCIENTIFIC_PLAN section 3: "consistent with it (<= 0.02 + 2 se)"); shifted fibre also (plan): the
      residual mean E[Y - m(x) | bin] in the same two bins within 3 se of 0, and the pooled residual variance over the
      production bins with |x| <= 0.3 within 2 % of 1/(beta kappa^2).
      STRICT READING (reported, not gating): the frozen gateway gate's V1 text also requires |m - closed form EM|
      <= 3 se.  It is evaluated (V1_strict, with the z values) and the whole selection is repeated under it
      (selection.strict_reading); if the two readings disagree the summary says so and the main session decides.
      Reason: over the 7 dynamics x 2 bins that clause has a family-wise false-FAIL probability of ~12 % per h with
      every sampler exact (t_15 tails, 0.9 % per test), and the frozen-x closed form omits the x-coupling (accepted alpha = 1 data:
      z = 1.56 / 1.97), so a chance or model-inadequacy FAIL would force a 99.5 core-h refinement.
  V2  D(F_density, F*) <= 0.00185 with bootstrap 95 % upper bound <= 0.0028, and the same for F_MF.
  V3  D(conditional mean force, bin-averaged F*') <= 0.0115 with upper bound <= 0.017, and the closed-form EM mean-force
      bias RMS <= 0.0115 (frozen gateway gate).
  V3b Var(f|x) measured as the within-bin variance of the residual f - F*'(x) (pointwise conditional variance; the
      within-bin variance of f itself also contains Var(F*'(x) | bin)) vs the bin-averaged exact (2 alpha^2/beta^2)
      (omega'/omega)^2 (shift: kappa^2 m'^2/beta): debiased RMS relative deviation <= 0.05 over the eval-window bins
      where the exact value is > 0.05; alpha = 0: pooled measured variance <= 1e-12 x pooled mean f^2.
  V5  no non-finite state (EM flat, EM unbiased); unbiased wall reflections < 1e-6 per walker-step; flat-bias walkers
      cross the gate (in every group >= 50 % of the walkers cross x = 0); ABF-bias smoke stability from the hook
      (abf_smoke.json, written by the main session from engine smoke runs: >= 2 seeds, finite, max|Gamma| < 2 max|F*'| on
      visited bins) -- PENDING while absent.
Harness gate (h-INDEPENDENT, a precondition as in the frozen gateway selection rule "with V4, V5 passed"; review fix
2026-10-10: it is NOT part of the per-h status, so a V4 failure can never trigger a refinement it cannot change):
  V4  per dynamics, MALA at h 1e-4 and the exact i.i.d. sampler each: central-bin Var(Y|x) within 3 se of exact
      (shift: also the central residual mean within 3 se and the |x| <= 0.3 pooled variance within 2 %), the V3b
      statistic within its threshold, F_density upper bound <= 0.0028, no non-finite state (MALA finiteness lives here).
      Any V4 FAIL makes the verdict HARNESS_FAIL: stop and investigate (the h selection the per-h gates would make is
      still reported).  The 3-se clauses are 32 tests (7 dynamics x 2 samplers x 2 bins + the shift's 2 x 2 mean bins):
      family-wise false-FAIL probability ~25 % with every sampler exact; summary.multiplicity gives the counts, the
      t_{G-1} per-test rate and the Bonferroni-adjusted p of the largest |z| of each family (descriptive -- a
      multiplicity rule would be a preregistration amendment, which is the main session's decision).
Selection: h_common = the largest candidate (2.5e-5, then 1.25e-5) at which all seven dynamics PASS (V1, V2, V3, V3b,
V5) and V4 passes for all.  If 2.5e-5 fails and 1.25e-5 has not been run: REFINE_REQUIRED (all dynamics at 1.25e-5).
If some dynamics fail at every tested h they are reported UNRESOLVED and the common h of the rest is given.  With every
harness gate passed but the ABF smoke pending the selection is PROVISIONAL.  No ABF/FR performance is read.
Descriptive (not gated): conditional ACF of z_y and of the sign-invariant z_f (gateway_family_validation module
docstring) in x-windows -1, -0.21, 0 (+-0.02), integrated times to 5 tau and 1/e times vs the frozen-x OU predictions
tau_y = 1/(lam k(x)), tau_f = tau_y/2 (alpha family) or tau_y (shift); a window is 'resolved' for a quantity when the
MEASURED rho at the first nonzero lag is >= 0.8 (not from the frozen-x tau: the shifted fibre relaxes ~30x faster than
it); exact-unbiased read-out of F; the within-bin variance of f itself vs exact.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_family_validation as GF  # noqa: E402


def _load_run_validation():
    """scripts/mechanism/run_validation.py (the frozen run design: groups, walkers, run lengths, seed rule)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mech_run_validation", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                                     "run_validation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RV = _load_run_validation()

RES_DEFAULT = os.path.join(ROOT, "results", "mechanism", "validation")
FIG_DEFAULT = os.path.join(ROOT, "figures", "mechanism", "validation")
GATE_CFG = os.path.join(ROOT, "configs", "equal_budget_v2", "gateway_validation.json")
NBF, NB = GF.NB_FINE, GF.NB_PROD
# frozen thresholds (configs/equal_budget_v2/gateway_validation.json; SCIENTIFIC_PLAN section 3) -- checked against the
# config text at import so a drift cannot pass silently
TOL_F, TOL_F_UP, TOL_FP, TOL_FP_UP = 0.00185, 0.0028, 0.0115, 0.017
TOL_V1 = 0.02
TOL_V1_SHIFT_VAR, V1_SHIFT_REGION = 0.02, 0.3
TOL_V3B, V3B_MIN_EXACT, V3B_ALPHA0_RATIO = 0.05, 0.05, 1e-12
V5_REFL, V5_CROSS_FRAC, ABF_SMOKE_FACTOR, ABF_SMOKE_MIN_SEEDS = 1e-6, 0.5, 2.0, 2
H_CANDIDATES = (2.5e-5, 1.25e-5)
H_MALA = 1e-4
EVAL_LO, EVAL_HI = -1.5, 1.5
PASS, FAIL, NO_DATA, PENDING = "PASS", "FAIL", "NO_DATA", "PENDING"
N_BOOT = 1000
SDF_MIN_ACF = 1e-8          # windows whose mean exact sd(f|x) is below this report no z_f ACF (rounding noise)
RHO_RESOLVED = 0.8          # a conditional ACF is 'resolved' when the MEASURED rho at the first nonzero lag is >= this
Z_3SE = 3.0                 # the frozen 'within 3 se' clauses
if tuple(H_CANDIDATES) != tuple(RV.H_CANDIDATES) or H_MALA != RV.H_MALA:
    raise RuntimeError("analysis and run script disagree on the candidate h / MALA h")


def _check_frozen_thresholds():
    g = json.load(open(GATE_CFG))["gates"]
    txt = json.dumps(g)
    for s in ("<= 0.00185", "<= 0.0028", "<= 0.0115", "<= 0.017", "<= 2 %"):
        if s not in txt:
            raise RuntimeError(f"frozen gate text changed: '{s}' not in {GATE_CFG}")


_check_frozen_thresholds()


# ---------------------------------------------------------------------------------------------------------------
# loading and statistics (conventions of scripts/equal_budget/analyze_gateway_validation.py)
# ---------------------------------------------------------------------------------------------------------------
def load(res, dyn, tag):
    fs = sorted(f for f in glob.glob(os.path.join(res, dyn, tag, "g*.npz")) if ".tmp." not in f)
    out = []
    for f in fs:
        with np.load(f) as d:
            r = {k: d[k] for k in d.files if k != "meta_json"}
            r["meta"] = json.loads(str(d["meta_json"]))
        out.append(r)
    return out


def design_issues(recs, chain, m, d_index, h, smoke=False):
    """Deviations of one record set (chain of dynamics m, index d_index in DYNAMICS_ORDER, step h; h ignored for the
    exact chains) from the frozen design of run_validation.py (SCIENTIFIC_PLAN section 3).  [] = conforms."""
    iss = []
    exp_groups = [0] if chain == "em_unbiased" else list(range(RV.G))
    gids = sorted(int(r["meta"].get("group", -1)) for r in recs)
    if gids != exp_groups:
        iss.append(f"groups {gids} != {exp_groups}")
    for r in recs:
        mt = r["meta"]
        g = int(mt.get("group", -1))
        pre = f"g{g:02d}: "
        if bool(mt.get("smoke", True)) != bool(smoke):
            iss.append(pre + f"meta smoke={mt.get('smoke')} (gate run expects smoke={bool(smoke)})")
        if mt.get("chain") != chain or mt.get("dynamics") != m["name"]:
            iss.append(pre + f"chain/dynamics {mt.get('chain')}/{mt.get('dynamics')} != {chain}/{m['name']}")
        try:
            want_seed = RV.seed_of(chain, d_index, h, g)
        except Exception as e:                                    # h not a candidate
            want_seed = f"<{e}>"
        if mt.get("seed") != want_seed:
            iss.append(pre + f"seed {mt.get('seed')} != preregistered {want_seed}")
        if chain.startswith("exact"):
            flat = chain == "exact_flat"
            n_want = RV.N_EXACT_FLAT if flat else RV.N_EXACT_UNB
            if mt.get("n_draws") != n_want or bool(mt.get("flat")) != flat:
                iss.append(pre + f"n_draws/flat {mt.get('n_draws')}/{mt.get('flat')} != {n_want}/{flat}")
            continue
        T_want, burn_want = RV.settings(chain, m, False)
        nw_want = 8 if chain == "em_unbiased" else RV.NW
        flat_want = 0 if chain == "em_unbiased" else 1
        T, burn, hh = mt.get("T"), mt.get("burn"), mt.get("h")
        if mt.get("n_walkers") != nw_want:
            iss.append(pre + f"n_walkers {mt.get('n_walkers')} != {nw_want}")
        if T is None or abs(T - T_want) > 1e-9 * T_want:
            iss.append(pre + f"T {T} != {T_want}")
        if burn is None or abs(burn - burn_want) > 1e-12:
            iss.append(pre + f"burn {burn} != {burn_want}")
        if hh is None or abs(hh - h) > 1e-12 * h:
            iss.append(pre + f"h {hh} != {h}")
        if mt.get("flat_bias") != flat_want:
            iss.append(pre + f"flat_bias {mt.get('flat_bias')} != {flat_want}")
        if hh and T and burn is not None and mt.get("n_steps") != int(round((T_want + burn_want) / h)):
            iss.append(pre + f"n_steps {mt.get('n_steps')} != {int(round((T_want + burn_want) / h))}")
    return iss


def jack(fn, recs):
    """(estimate, jackknife se) of fn(list of group records); se NaN with fewer than 2 groups."""
    n = len(recs)
    est = np.asarray(fn(recs), float)
    if n < 2:
        return est, np.full_like(est, np.nan)
    loo = np.array([fn(recs[:i] + recs[i + 1:]) for i in range(n)], float)
    return est, np.sqrt((n - 1) / n * ((loo - loo.mean(0)) ** 2).sum(0))


def debias(d, se, centre=True):
    """(rms, D = sqrt(max(0, rms^2 - mean se^2)), noise = sqrt(mean se^2)) of a profile error d."""
    d = np.asarray(d, float)
    if centre:
        d = d - d.mean()
    rms = float(np.sqrt(np.mean(d ** 2)))
    noise = float(np.sqrt(np.mean(np.asarray(se, float) ** 2)))
    D = float(np.sqrt(max(0.0, rms ** 2 - noise ** 2))) if np.isfinite(noise) else float("nan")
    return rms, D, noise


def boot_upper(fn, recs, se_mean2, centre=True, n=N_BOOT, seed=0):
    """95 % bootstrap upper bound of the debiased RMS: groups resampled with replacement, each resample debiased by
    2 x the full-sample noise variance (the resample adds its own sampling noise), as analyze_gateway_validation.py."""
    if len(recs) < 2 or not np.isfinite(se_mean2):
        return float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(recs), len(recs))
        d = np.asarray(fn([recs[i] for i in idx]), float)
        if centre:
            d = d - d.mean()
        vals.append(np.sqrt(max(0.0, np.mean(d ** 2) - 2 * se_mean2)))
    return float(np.quantile(vals, 0.95))


def _le(a, b):
    """True / False, or None when a is not a finite number (no data)."""
    if a is None or not np.isfinite(a):
        return None
    return bool(a <= b)


def status(*conds):
    """FAIL if any condition is False; else NO_DATA if any is None; else PASS."""
    flat = []
    for c in conds:
        flat += list(c) if isinstance(c, (list, tuple)) else [c]
    if any(c is False for c in flat):
        return FAIL
    if any(c is None for c in flat):
        return NO_DATA
    return PASS


def _sum(rs, key):
    return sum(r[key] for r in rs)


# ---------------------------------------------------------------------------------------------------------------
# per-chain analysis
# ---------------------------------------------------------------------------------------------------------------
_REF_CACHE = {}


def ref_bins(m, nb, h=None):
    """GF.bin_averages, cached per (model, nb, h)."""
    key = (tuple(sorted((k, str(v)) for k, v in m.items())), nb, h)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = GF.bin_averages(m, nb, h=h)
    return _REF_CACHE[key]


def analyse_flat(recs, m, h=None):
    """Read-out of flat-bias records (EM at h, or an exact sampler with h=None: predicted discretisation error 0)."""
    ref2 = ref_bins(m, NB, h)
    ref1 = ref_bins(m, NBF, h)
    cen = ref2["centres"]
    W = (cen >= EVAL_LO) & (cen <= EVAL_HI)
    kT = 1.0 / m["beta"]
    o = dict(n_groups=len(recs), h=h)
    if not recs:
        return o
    o["walker_steps"] = float(sum(r["meta"]["walker_steps"] for r in recs))
    o["wall_s"] = float(sum(r["meta"]["wall_s"] for r in recs))
    o["nonfinite"] = int(sum(int(r["nonfinite"]) for r in recs))
    if "n_prop" in recs[0]:
        o["acceptance"] = float(sum(float(r["n_acc"]) for r in recs) / max(1.0, sum(float(r["n_prop"]) for r in recs)))
    if "crossings" in recs[0]:
        fr = [float(np.mean(r["crossings"] > 0)) for r in recs]
        o["cross_frac_min"] = float(min(fr))
        o["cross_frac_mean"] = float(np.mean(fr))
        o["crossings_per_walker"] = float(np.mean([r["crossings"].mean() for r in recs]))

    def F_den(rs):
        C = _sum(rs, "A2")[0]
        F = -kT * np.log(np.maximum(C, 1e-300))
        return F[W] - F[W].mean()

    def mf_err(rs):
        A = _sum(rs, "A2")
        return (A[1] / np.maximum(A[0], 1e-300) - ref2["Fp_flat"])[W]

    def F_mf(rs):
        A = _sum(rs, "A2")
        d = A[1] / np.maximum(A[0], 1e-300) - ref2["Fp_flat"]
        dz = (GF.XMAX - GF.XMIN) / NB
        F = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * dz)])
        return F[W] - F[W].mean()

    fd, sfd = jack(F_den, recs)
    o["F_density"] = dict(zip(("rms", "D", "noise"), debias(fd, sfd)))
    o["F_density"]["upper95"] = boot_upper(F_den, recs, float(np.mean(sfd ** 2)))
    fm, sfm = jack(F_mf, recs)
    o["F_MF"] = dict(zip(("rms", "D", "noise"), debias(fm, sfm)))
    o["F_MF"]["upper95"] = boot_upper(F_mf, recs, float(np.mean(sfm ** 2)))
    me, sme = jack(mf_err, recs)
    o["mean_force"] = dict(zip(("rms", "D", "noise"), debias(me, sme, centre=False)))
    o["mean_force"]["upper95"] = boot_upper(mf_err, recs, float(np.mean(sme ** 2)), centre=False, seed=1)
    em_bias = (ref2["Fp_em_flat"] - ref2["Fp_flat"])[W] if h is not None else np.zeros(int(W.sum()))
    o["mean_force"]["closed_form_EM_bias_rms"] = float(np.sqrt(np.mean(em_bias ** 2)))

    # V1: transverse conditional variance at the stiffest point (two central fine bins), residual y - E[Y|x]
    def var_rel(rs):
        A = _sum(rs, "A1")
        C = np.maximum(A[0], 1.0)
        return (A[8] / C - (A[7] / C) ** 2) / ref1["var_flat"] - 1.0

    vr, svr = jack(var_rel, recs)
    c2 = np.sort(np.argsort(np.abs(ref1["centres"]))[:2])
    pred = (ref1["var_em_flat"] / ref1["var_flat"] - 1.0) if h is not None else np.zeros(NBF)
    o["var_centre"] = dict(x=ref1["centres"][c2].tolist(), measured=vr[c2].tolist(), se=svr[c2].tolist(),
                           predicted_EM=pred[c2].tolist(), closed_form_peak=(GF.closed_form_peak(m, h) if h is not None else 0.0))
    sel = np.abs(ref1["centres"]) < 0.3
    o["var_profile"] = dict(x=ref1["centres"][sel].tolist(), rel=vr[sel].tolist(), se=svr[sel].tolist(), pred=pred[sel].tolist())

    # transverse conditional mean (residual) and the pooled |x| <= 0.3 variance (constant-width fibres)
    def mean_res(rs):
        A = _sum(rs, "A1")
        return A[7] / np.maximum(A[0], 1.0)

    mr, smr = jack(mean_res, recs)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = mr[c2] / smr[c2]
    o["mean_centre"] = dict(resid_mean=mr[c2].tolist(), se=smr[c2].tolist(), z=z.tolist())
    Rg = np.abs(cen) <= V1_SHIFT_REGION + 1e-12
    if m["variant"] == "shift" or m.get("alpha") == 0.0:
        v_exact = 1.0 / (m["beta"] * float(GF.stiffness(np.array([0.0]), m)[0]))

        def var_region(rs):
            A = _sum(rs, "A2")
            C = A[0][Rg].sum()
            if C <= 0:
                return np.array(np.nan)
            return np.array((A[8][Rg].sum() / C - (A[7][Rg].sum() / C) ** 2) / v_exact - 1.0)

        vg, svg = jack(var_region, recs)
        o["var_region"] = dict(rel=float(vg), se=float(svg), x_max=V1_SHIFT_REGION, exact=v_exact)
    sel6 = np.abs(cen) <= 0.6
    mrp, smrp = jack(lambda rs: _sum(rs, "A2")[7] / np.maximum(_sum(rs, "A2")[0], 1.0), recs)
    o["mean_profile"] = dict(x=cen[sel6].tolist(), resid_mean=mrp[sel6].tolist(), se=smrp[sel6].tolist())

    # V3b: pointwise conditional force variance (residual r = f - F*'(x))
    def fvar(rs):
        A = _sum(rs, "A2")
        C = np.maximum(A[0], 1.0)
        return A[6] / C - (A[5] / C) ** 2

    fv, sfv = jack(fvar, recs)
    ex = ref2["varf_flat"]
    fsel = W & (ex > V3B_MIN_EXACT)
    fo = dict(n_bins=int(fsel.sum()))
    if fsel.any():
        def frel(rs):
            return fvar(rs)[fsel] / ex[fsel] - 1.0
        rel, srel = jack(frel, recs)
        rms, D, noise = debias(rel, srel, centre=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            mz = float(np.nanmax(np.abs(rel / srel))) if np.all(np.isfinite(srel)) and np.all(srel > 0) else float("nan")
        fo.update(rms_rel=rms, D_rel=D, noise_rel=noise, max_abs_z=mz,
                  upper95_rel=boot_upper(frel, recs, float(np.mean(srel ** 2)), centre=False, seed=2))
        pr = (ref2["varf_em_flat"][fsel] / ex[fsel] - 1.0) if h is not None else np.zeros(int(fsel.sum()))
        fo["predicted_EM_rms_rel"] = float(np.sqrt(np.mean(pr ** 2)))
    else:
        def ratio(rs):
            A = _sum(rs, "A2")
            C = A[0][W]
            v = (A[6][W] - A[5][W] ** 2 / np.maximum(C, 1.0)).sum() / max(C.sum(), 1.0)
            return np.array(v / max((A[2][W].sum() / max(C.sum(), 1.0)), 1e-300))
        fo["ratio_var_to_mean_f2"] = float(ratio(recs))
        A = _sum(recs, "A2")
        C = np.maximum(A[0], 1.0)
        fo["max_bin_ratio"] = float(np.max((fv / np.maximum(A[2] / C, 1e-300))[W]))
    # descriptive: the within-bin variance of f itself vs E[Var(f|x)|bin] + Var(F*'(x)|bin)
    A = _sum(recs, "A2")
    C = np.maximum(A[0], 1.0)
    tot = A[2] / C - (A[1] / C) ** 2
    tot_ex = ex + ref2["Fp2_flat"] - ref2["Fp_flat"] ** 2
    tsel = W & (tot_ex > V3B_MIN_EXACT)
    fo["total_f_var_rms_rel"] = float(np.sqrt(np.mean((tot[tsel] / tot_ex[tsel] - 1) ** 2))) if tsel.any() else float("nan")
    o["force_var"] = fo
    o["force_var_profile"] = dict(x=cen[sel6].tolist(), measured=fv[sel6].tolist(), se=sfv[sel6].tolist(), exact=ex[sel6].tolist(),
                                  em=(ref2["varf_em_flat"][sel6] if h is not None else ex[sel6]).tolist())
    o["profiles"] = dict(centres=cen[W].tolist(), F_density_err=fd.tolist(), F_density_se=sfd.tolist(), F_MF_err=fm.tolist(),
                         mf_err=me.tolist(), mf_se=sme.tolist(), em_bias_mf=em_bias.tolist())
    if "zhist" in recs[0]:
        o["zhist"] = _sum(recs, "zhist").tolist()
    return o


def gates_em(o, m):
    """V1, V2, V3, V3b of one (dynamics, h) EM flat-bias read-out."""
    if not o.get("n_groups"):
        return dict(V1=NO_DATA, V2=NO_DATA, V3=NO_DATA, V3b=NO_DATA)
    v = o["var_centre"]
    c_cf = _le(v["closed_form_peak"], TOL_V1)
    c_le = [_le(mm - 2 * s, TOL_V1) for mm, s in zip(v["measured"], v["se"])]
    comp = dict(closed_form=c_cf, central_le_2pct_plus_2se=c_le)
    if m["variant"] == "shift":
        comp["shift_mean_3se"] = [None if not np.isfinite(zz) else bool(abs(zz) <= Z_3SE) for zz in o["mean_centre"]["z"]]
        comp["shift_var_region_2pct"] = _le(abs(o["var_region"]["rel"]), TOL_V1_SHIFT_VAR) if np.isfinite(o["var_region"]["se"]) else None
    # the frozen gateway gate's extra consistency clause |m - closed form EM| <= 3 se: STRICT READING, reported only
    z_cons = [(mm - p) / s if np.isfinite(s) and s > 0 else float("nan") for mm, s, p in zip(v["measured"], v["se"], v["predicted_EM"])]
    c_cons = [None if not np.isfinite(z) else bool(abs(z) <= Z_3SE) for z in z_cons]
    g = dict(V1=status(*comp.values()), V1_components=comp, V1_consistency_3se=c_cons, V1_consistency_z=z_cons,
             V1_strict=status(*comp.values(), c_cons))
    g["V2"] = status(_le(o["F_density"]["D"], TOL_F), _le(o["F_density"]["upper95"], TOL_F_UP),
                     _le(o["F_MF"]["D"], TOL_F), _le(o["F_MF"]["upper95"], TOL_F_UP))
    g["V3"] = status(_le(o["mean_force"]["D"], TOL_FP), _le(o["mean_force"]["upper95"], TOL_FP_UP),
                     _le(o["mean_force"]["closed_form_EM_bias_rms"], TOL_FP))
    g["V3b"] = v3b_status(o)
    return g


def v3b_status(o):
    fo = o["force_var"]
    if fo["n_bins"] > 0:
        return status(_le(fo["D_rel"], TOL_V3B))
    return status(_le(fo["ratio_var_to_mean_f2"], V3B_ALPHA0_RATIO))


def exact_sampler_z(o, m):
    """The z values of the 3-se clauses of V4 for one exact sampler: {'central_var': [..], 'shift_mean': [..]}."""
    if not o.get("n_groups"):
        return {}
    v = o["var_centre"]
    z = dict(central_var=[mm / s if np.isfinite(s) and s > 0 else float("nan") for mm, s in zip(v["measured"], v["se"])])
    if m["variant"] == "shift":
        z["shift_mean"] = [float(zz) for zz in o["mean_centre"]["z"]]
    return z


def gate_exact_sampler(o, m):
    """V4 components for one exact sampler (MALA or i.i.d.): predicted discretisation error 0."""
    if not o.get("n_groups"):
        return NO_DATA, {}
    z = exact_sampler_z(o, m)
    comp = dict(central_var_3se=[None if not np.isfinite(zz) else bool(abs(zz) <= Z_3SE) for zz in z["central_var"]],
                F_density_upper=_le(o["F_density"]["upper95"], TOL_F_UP), force_var=v3b_status(o) == PASS if v3b_status(o) != NO_DATA else None,
                finite=o["nonfinite"] == 0)
    if m["variant"] == "shift":
        comp["shift_mean_3se"] = [None if not np.isfinite(zz) else bool(abs(zz) <= Z_3SE) for zz in z["shift_mean"]]
        comp["shift_var_region_2pct"] = _le(abs(o["var_region"]["rel"]), TOL_V1_SHIFT_VAR) if np.isfinite(o["var_region"]["se"]) else None
    return status(*comp.values()), comp


def three_se_only(comp):
    """True if every False component of a V4 component dict is a 3-se clause (a chance exceedance is possible)."""
    bad = [k for k, c in comp.items() if any(x is False for x in (c if isinstance(c, list) else [c]))]
    return bool(bad) and all(k.endswith("_3se") for k in bad)


def t_two_sided(z, dof):
    """Two-sided tail probability of Student t_dof at |z| (the jackknife z over G groups is ~ t_{G-1})."""
    from scipy.stats import t as _t
    return float(2.0 * _t.sf(abs(z), dof))


def family_rate(k, dof, z=Z_3SE):
    """P(at least one of k independent |t_dof| > z): the family-wise false-FAIL probability of k 3-se clauses."""
    return float(1.0 - (1.0 - t_two_sided(z, dof)) ** k) if k > 0 else 0.0


def family_summary(zs, dof):
    """Counts, per-test null rate, family-wise null rate and the Bonferroni-adjusted p of the largest |z| of a family
    of 3-se clauses (descriptive; zs = list of finite z values)."""
    zs = [float(z) for z in zs if np.isfinite(z)]
    k = len(zs)
    out = dict(n_tests=k, dof=dof, per_test_null_rate=t_two_sided(Z_3SE, dof) if dof > 0 else float("nan"),
               familywise_null_rate=family_rate(k, dof) if dof > 0 else float("nan"))
    if k and dof > 0:
        zmax = max(zs, key=abs)
        out.update(max_abs_z=abs(zmax), n_exceed=int(sum(abs(z) > Z_3SE for z in zs)),
                   bonferroni_p_of_max=min(1.0, k * t_two_sided(zmax, dof)))
    return out


def analyse_unbiased_em(recs):
    if not recs:
        return None
    r = recs[0]
    return dict(nonfinite=int(r["nonfinite"]), reflections_per_walker_step=float(r["n_reflect"]) / r["meta"]["walker_steps"],
                max_dx=float(r["max_dx"]), walker_steps=float(r["meta"]["walker_steps"]), wall_s=float(r["meta"]["wall_s"]))


def analyse_exact_unbiased(recs, m):
    """Read-out of the unbiased exact draws: F_density on the eval-window bins with >= 100 counts vs the exact
    bin-integrated F (descriptive)."""
    if not recs:
        return None
    ref = ref_bins(m, NB)
    cen = ref["centres"]
    C = _sum(recs, "A2")[0]
    sel = (cen >= EVAL_LO) & (cen <= EVAL_HI) & (C >= 100)
    kT = 1.0 / m["beta"]

    def err(rs):
        Cr = _sum(rs, "A2")[0][sel]
        e = -kT * np.log(np.maximum(Cr, 1e-300)) - ref["F"][sel]
        return e - e.mean()
    e, se = jack(err, recs)
    rms, D, noise = debias(e, se)
    return dict(n_bins=int(sel.sum()), rms=rms, D=D, noise=noise, upper95=boot_upper(err, recs, float(np.mean(se ** 2))))


# ---------------------------------------------------------------------------------------------------------------
# conditional autocorrelation (descriptive covariate, SCIENTIFIC_PLAN section 6)
# ---------------------------------------------------------------------------------------------------------------
def analyse_acf(recs, m, h):
    recs = [r for r in recs if "acf0" in r]
    if not recs:
        return None
    stride = recs[0]["acf_stride"]
    lags = (recs[0]["lags0"], recs[0]["lags1"])
    out = dict(stride_steps=stride.tolist(), stride_t=(stride * h).tolist(), n_groups=len(recs), windows={})
    has_f_model = m["variant"] == "shift" or m["alpha"] != 0.0
    for w, c in enumerate(GF.ACF_WINDOWS):
        ou = GF.ou_prediction(m, c)
        # z_f = (f - F*'(x)) / sd_f(x) is meaningful only where the exact force fluctuation exceeds the rounding of the
        # subtraction f - F*' (~1e-16 |f|): in the wells omega'/omega ~ 1e-20, so z_f there is pure rounding noise
        xw = np.linspace(c - GF.ACF_HALF_WIDTH, c + GF.ACF_HALF_WIDTH, 401)
        sdf_win = float(np.mean(np.sqrt(GF.cond_var_f(xw, m))))
        has_f = has_f_model and sdf_win > SDF_MIN_ACF
        ty, tf = ou["tau_y_centre"], ou["tau_f_centre"]
        # the finest level whose lags reach 5 tau_y (frozen); 'resolved' is judged below from the MEASURED decay
        lev = next((L for L in (0, 1) if lags[L][-1] * stride[L] * h >= 5.0 * ty), None)
        reaches = lev is not None
        if lev is None:
            lev = int(np.argmax([lags[L][-1] * stride[L] for L in (0, 1)]))
        key = f"acf{lev}"
        t = lags[lev] * stride[lev] * h
        k1 = int(np.argmax(lags[lev] > 0))                     # first nonzero lag

        def rho(rs, q):
            return GF.acf_from_sums(_sum(rs, key))["rho_" + q][w]

        ry, sry = jack(lambda rs: rho(rs, "y"), recs)
        n0 = float(_sum(recs, key)[w, 0, 0])
        tt = np.linspace(0, max(t[-1], 5 * ty), 20001)
        e = dict(x_centre=c, level=lev, reaches_5tau=reaches, t=t.tolist(), n_origin_samples=n0, sd_f_window_mean=sdf_win,
                 f_acf_reported=bool(has_f), first_lag_t=float(t[k1]), rho_y_first_lag=float(ry[k1]),
                 resolved_y=bool(np.isfinite(ry[k1]) and ry[k1] >= RHO_RESOLVED),
                 tau_y_frozen=ty, tau_f_frozen=tf if has_f else None, f_factor=ou["f_factor"],
                 rho_y=ry.tolist(), rho_y_se=sry.tolist(), rho_y_pred=ou["rho_y"](t).tolist(),
                 m01_y_lag0=float(GF.acf_from_sums(_sum(recs, key))["m01_y"][w, 0]))
        tau_int = lambda rs, q, cut: np.array(GF.tau_integrated(t, rho(rs, q), cut))
        v, sv = jack(lambda rs: tau_int(rs, "y", 5 * ty), recs)
        e["tau_int_y_5tau"], e["tau_int_y_5tau_se"] = float(v), float(sv)
        e["tau_int_y_5tau_pred"] = GF.tau_integrated(tt, ou["rho_y"](tt), 5 * ty)
        e["tau_e_y"] = GF.tau_efold(t, ry)
        if has_f:
            rf, srf = jack(lambda rs: rho(rs, "f"), recs)
            e.update(rho_f=rf.tolist(), rho_f_se=srf.tolist(), rho_f_pred=ou["rho_f"](t).tolist())
            v, sv = jack(lambda rs: tau_int(rs, "f", 5 * tf), recs)
            e["tau_int_f_5tau"], e["tau_int_f_5tau_se"] = float(v), float(sv)
            e["tau_int_f_5tau_pred"] = GF.tau_integrated(tt, ou["rho_f"](tt), 5 * tf)
            e["tau_e_f"] = GF.tau_efold(t, rf)
            e["rho_f_first_lag"] = float(rf[k1])
            e["resolved_f"] = bool(np.isfinite(rf[k1]) and rf[k1] >= RHO_RESOLVED)
        e["resolved"] = bool(e["resolved_y"] and e.get("resolved_f", True))
        out["windows"][f"{c:g}"] = e
    return out


# ---------------------------------------------------------------------------------------------------------------
# V5 and the ABF-bias smoke hook
# ---------------------------------------------------------------------------------------------------------------
def abf_smoke_gate(entries):
    """ABF-bias stability (SCIENTIFIC_PLAN V5): entries = list of per-seed dicts from the engine smoke runs
    (N = 2048, 4 t.u., live ABF + FR) with keys finite (bool), max_abs_Gamma_visited, max_abs_Fp_visited.
    PASS iff >= 2 seeds, all finite and max|Gamma| < 2 max|F*'| on visited bins; PENDING if no entries."""
    if not entries:
        return PENDING
    ok = len(entries) >= ABF_SMOKE_MIN_SEEDS
    for e in entries:
        ok = ok and bool(e.get("finite")) and np.isfinite(e.get("max_abs_Gamma_visited", np.nan)) and \
            e["max_abs_Gamma_visited"] < ABF_SMOKE_FACTOR * e["max_abs_Fp_visited"]
    return PASS if ok else FAIL


def load_abf_smoke(path):
    """The V5 ABF-bias smoke hook, written by the main session from the engine smoke runs (default
    <res>/abf_smoke.json): {dynamics name: {f"{h:g}" (e.g. "2.5e-05"): [ {"seed": 8100, "finite": true,
    "max_abs_Gamma_visited": float, "max_abs_Fp_visited": float}, ... ]}}; {} if the file does not exist."""
    if path and os.path.exists(path):
        return json.load(open(path))
    return {}


def gate_v5(o_em, unb, abf_entries):
    """Per-h V5 (the MALA finiteness check is h-independent and lives in V4)."""
    comp = dict(em_flat_finite=(o_em["nonfinite"] == 0) if o_em.get("n_groups") else None,
                em_flat_cross=_le(-o_em.get("cross_frac_min", float("nan")), -V5_CROSS_FRAC) if o_em.get("n_groups") else None,
                unbiased_finite=(unb["nonfinite"] == 0) if unb else None,
                unbiased_reflections=bool(unb["reflections_per_walker_step"] < V5_REFL) if unb else None)
    core = status(*comp.values())
    abf = abf_smoke_gate(abf_entries)
    if core == FAIL or abf == FAIL:
        full = FAIL
    elif core == NO_DATA:
        full = NO_DATA
    elif abf == PENDING:
        full = PENDING
    else:
        full = PASS
    return dict(V5_core=core, V5_abf_smoke=abf, V5=full, V5_components=comp)


def combine(*sts):
    if any(s == FAIL for s in sts):
        return FAIL
    if any(s == NO_DATA for s in sts):
        return NO_DATA
    if any(s == PENDING for s in sts):
        return PENDING
    return PASS


def select_common_h(table, names, hs=H_CANDIDATES, key="full"):
    """table[dyn][f'{h:g}'][key] in {PASS, FAIL, NO_DATA, PENDING}.  Candidates are visited largest first; at each h
    the state is PASS (all dynamics pass), FAIL (any fails), NO_DATA (any missing, none failing) or PENDING.  The first
    PASS is the common h.  A FAIL moves on to the next candidate (refinement for ALL dynamics); NO_DATA after a FAIL is
    REFINE_REQUIRED (next_h), NO_DATA at the first candidate is INCOMPLETE, PENDING stops the selection.  If every
    candidate FAILs, the dynamics failing at every candidate are UNRESOLVED and the common h of the rest is given."""
    st = lambda d, h: table.get(d, {}).get(f"{h:g}", {}).get(key, NO_DATA)

    def hstate(h):
        s = [st(d, h) for d in names]
        if all(x == PASS for x in s):
            return PASS
        if any(x == FAIL for x in s):
            return FAIL
        if any(x == NO_DATA for x in s):
            return NO_DATA
        return PENDING

    failed = []
    for h in hs:
        s = hstate(h)
        if s == PASS:
            return dict(h_common=h, verdict="PASS", failed_at=failed, unresolved=[])
        if s == FAIL:
            failed.append(h)
            continue
        failing = sorted({d for d in names for hf in failed if st(d, hf) == FAIL})
        if s == PENDING:
            return dict(h_common=None, verdict="PENDING", pending_h=h, failed_at=failed, failing=failing, unresolved=[])
        return dict(h_common=None, verdict="REFINE_REQUIRED" if failed else "INCOMPLETE", next_h=h, failed_at=failed,
                    failing=failing, missing=[d for d in names if st(d, h) == NO_DATA], unresolved=[])
    unresolved = [d for d in names if all(st(d, h) != PASS for h in hs)]
    rest = [d for d in names if d not in unresolved]
    sub = next((h for h in hs if rest and all(st(d, h) == PASS for d in rest)), None)
    return dict(h_common=None, verdict="UNRESOLVED", failed_at=failed, unresolved=unresolved, subset=rest, subset_h=sub)


# ---------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------
def decide(table, names, v4, key_core="core", key_full="full"):
    """Verdict from the per-h table (V1, V2, V3, V3b, V5; select_common_h) and the h-independent harness gate
    v4 = {dynamics: V4 status}.  A V4 FAIL is HARNESS_FAIL (stop and investigate), never a refinement; missing V4 data
    is INCOMPLETE; otherwise the h selection stands.  The h selection is always reported (h_selection_*)."""
    sel_core = select_common_h(table, names, key=key_core)
    sel_full = select_common_h(table, names, key=key_full)
    if sel_full["verdict"] == "PASS":
        vh, hh = "PASS", sel_full["h_common"]
    elif sel_core["verdict"] == "PASS" and sel_full["verdict"] == "PENDING" and sel_full.get("pending_h") == sel_core["h_common"]:
        vh, hh = "PROVISIONAL (harness gates pass; ABF-bias smoke PENDING)", sel_core["h_common"]
    else:
        vh, hh = sel_full["verdict"], sel_full.get("h_common")
    v4_fail = [d for d in names if v4.get(d, NO_DATA) == FAIL]
    v4_missing = [d for d in names if v4.get(d, NO_DATA) not in (PASS, FAIL)]
    out = dict(core=sel_core, full=sel_full, h_selection_verdict=vh, h_selection_h=hh,
               harness=dict(V4_fail=v4_fail, V4_missing=v4_missing))
    if v4_fail:
        out.update(verdict=f"HARNESS_FAIL (V4 fails for {', '.join(v4_fail)}: h-independent, refinement cannot change it -- stop "
                           f"and investigate; per-h gates alone: {vh})", h_common=None)
    elif v4_missing:
        out.update(verdict=f"INCOMPLETE (V4 data missing for {', '.join(v4_missing)}; per-h gates alone: {vh})", h_common=None)
    else:
        out.update(verdict=vh, h_common=hh)
    return out


def analyse_all(res, abf_path, smoke):
    dyns = GF.load_dynamics()
    abf = load_abf_smoke(abf_path)
    dof_plan = RV.G - 1
    S = dict(prereg=["docs/mechanism/SCIENTIFIC_PLAN.md section 3", "configs/equal_budget_v2/gateway_validation.json"],
             generated_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), res=os.path.relpath(res, ROOT), smoke=bool(smoke),
             thresholds=dict(V1=TOL_V1, V1_shift_var=TOL_V1_SHIFT_VAR, V1_shift_region=V1_SHIFT_REGION, V2_D=TOL_F,
                             V2_upper=TOL_F_UP, V3_D=TOL_FP, V3_upper=TOL_FP_UP, V3b=TOL_V3B, V3b_min_exact=V3B_MIN_EXACT,
                             V3b_alpha0_ratio=V3B_ALPHA0_RATIO, V4_F_upper=TOL_F_UP, V5_reflections=V5_REFL,
                             V5_cross_frac=V5_CROSS_FRAC, V5_abf_factor=ABF_SMOKE_FACTOR, z_3se=Z_3SE,
                             acf_rho_resolved=RHO_RESOLVED),
             readings=dict(V1="plan (SCIENTIFIC_PLAN s3): closed form <= 0.02 and measured <= 0.02 + 2 se [+ shift mean 3 se, "
                                "pooled var 2 %]; strict (gateway config V1 text) adds |measured - closed form| <= 3 se: "
                                "reported as selection.strict_reading",
                           V4="h-independent harness precondition (gateway selection rule 'with V4, V5 passed'): FAIL -> "
                              "HARNESS_FAIL, never REFINE_REQUIRED"),
             candidates=list(H_CANDIDATES), abf_smoke_file=abf_path if abf else None, dynamics={})
    nd = len(dyns)
    n_shift = sum(m["variant"] == "shift" for m in dyns)
    S["multiplicity"] = dict(
        note="3-se clauses under the frozen design (G = 16 groups, jackknife z ~ t_15), assuming independent tests; "
             "descriptive -- a multiplicity rule would be a preregistration amendment (main session's decision)",
        planned=dict(V4=dict(n_tests=2 * (2 * nd + 2 * n_shift), familywise_null_rate=family_rate(2 * (2 * nd + 2 * n_shift), dof_plan)),
                     V1_plan_per_h=dict(n_tests=2 * n_shift, familywise_null_rate=family_rate(2 * n_shift, dof_plan)),
                     V1_strict_per_h=dict(n_tests=2 * nd + 2 * n_shift, familywise_null_rate=family_rate(2 * nd + 2 * n_shift, dof_plan)),
                     per_test_null_rate=t_two_sided(Z_3SE, dof_plan)),
        observed={})
    table = {}
    v4 = {}
    z_v4 = []
    z_h = {f"{h:g}": dict(plan=[], strict=[]) for h in H_CANDIDATES}
    dof_seen = []
    for d_index, m in enumerate(dyns):
        name = m["name"]
        D = dict(model={k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in m.items()}, per_h={}, design={})

        def get(chain, h):
            """Records of one chain, EXCLUDED (-> []) unless they match the frozen design (listed either way)."""
            tg = f"{chain}_h{h:g}" if not chain.startswith("exact") else chain
            recs = load(res, name, tg)
            if not recs:
                return recs
            iss = design_issues(recs, chain, m, d_index, h, smoke=smoke)
            if iss:
                D["design"][tg] = iss[:20] + ([f"... {len(iss) - 20} more"] if len(iss) > 20 else [])
                if not smoke:
                    print(f"{name:9s} {tg}: EXCLUDED, does not match the frozen design: {iss[:3]}", flush=True)
                    return []
            return recs

        o_mala = analyse_flat(get("mala_flat", H_MALA), m, None)
        o_ex = analyse_flat(get("exact_flat", None), m, None)
        s_mala, c_mala = gate_exact_sampler(o_mala, m)
        s_ex, c_ex = gate_exact_sampler(o_ex, m)
        zm, ze = exact_sampler_z(o_mala, m), exact_sampler_z(o_ex, m)
        for zz in (zm, ze):
            for vals in zz.values():
                z_v4 += vals
        dof_seen += [o["n_groups"] - 1 for o in (o_mala, o_ex) if o.get("n_groups")]
        fails = [c for st_, c in ((s_mala, c_mala), (s_ex, c_ex)) if st_ == FAIL]
        D["V4"] = dict(status=combine(s_mala, s_ex), mala=s_mala, exact=s_ex, mala_components=c_mala, exact_components=c_ex,
                       mala_z=zm, exact_z=ze, fail_3se_clauses_only=bool(fails) and all(three_se_only(c) for c in fails))
        v4[name] = D["V4"]["status"]
        D["mala"] = o_mala
        D["exact_flat"] = o_ex
        D["exact_unbiased_readout"] = analyse_exact_unbiased(get("exact_unbiased", None), m)
        table[name] = {}
        for h in H_CANDIDATES:
            raw_em = load(res, name, f"em_flat_h{h:g}")
            raw_unb = load(res, name, f"em_unbiased_h{h:g}")
            if not raw_em and not raw_unb:
                continue
            recs = get("em_flat", h)
            unb = analyse_unbiased_em(get("em_unbiased", h))
            o = analyse_flat(recs, m, h)
            g = gates_em(o, m)
            g.update(gate_v5(o, unb, abf.get(name, {}).get(f"{h:g}", [])))
            g["V4"] = D["V4"]["status"]                     # reported here; h-independent, NOT in the per-h status
            core = combine(g["V1"], g["V2"], g["V3"], g["V3b"], g["V5_core"])
            full = combine(core, g["V5"])
            core_s = combine(g["V1_strict"], g["V2"], g["V3"], g["V3b"], g["V5_core"]) if o.get("n_groups") else core
            full_s = combine(core_s, g["V5"])
            table[name][f"{h:g}"] = dict(core=core, full=full, core_strict=core_s, full_strict=full_s)
            if o.get("n_groups"):
                zs = list(o["mean_centre"]["z"]) if m["variant"] == "shift" else []
                z_h[f"{h:g}"]["plan"] += zs
                z_h[f"{h:g}"]["strict"] += zs + list(g["V1_consistency_z"])
                dof_seen.append(o["n_groups"] - 1)
            D["per_h"][f"{h:g}"] = dict(em_flat=o, em_unbiased=unb, gates=g, status_core=core, status=full,
                                        status_strict=full_s, acf=analyse_acf(recs, m, h))
            print(f"{name:9s} h {h:g}: V1 {g['V1']} (strict {g.get('V1_strict', '-')}) V2 {g['V2']} V3 {g['V3']} V3b {g['V3b']} "
                  f"V5 {g['V5_core']}/abf {g['V5_abf_smoke']} -> per-h {core}, full {full} | V4 (h-indep.) {g['V4']} | "
                  f"var(c) {np.round(o['var_centre']['measured'], 5) if o.get('n_groups') else '-'} "
                  f"pred {o['var_centre']['closed_form_peak'] if o.get('n_groups') else '-'}", flush=True)
        S["dynamics"][name] = D
    names = [m["name"] for m in dyns]
    dof = min(dof_seen) if dof_seen else 0
    S["multiplicity"]["observed"] = dict(V4=family_summary(z_v4, dof),
                                         per_h={hk: dict(V1_plan=family_summary(v["plan"], dof), V1_strict=family_summary(v["strict"], dof))
                                                for hk, v in z_h.items() if v["strict"]})
    sel = decide(table, names, v4)
    sel["table"] = table
    v4f = sel["harness"]["V4_fail"]
    if v4f:
        only3 = all(S["dynamics"][d]["V4"]["fail_3se_clauses_only"] for d in v4f)
        sel["harness"]["V4_fail_3se_clauses_only"] = only3
        o4 = S["multiplicity"]["observed"]["V4"]
        sel["harness"]["note"] = (("only 3-se clauses fail: " if only3 else "a tolerance clause (not only 3 se) fails: ") +
                                  f"largest |z| {o4.get('max_abs_z', float('nan')):.2f} over {o4['n_tests']} V4 3-se tests, "
                                  f"Bonferroni p {o4.get('bonferroni_p_of_max', float('nan')):.3f}; family-wise null rate "
                                  f"{o4['familywise_null_rate']:.1%} with every sampler exact")
    strict = decide(table, names, v4, "core_strict", "full_strict")
    sel["strict_reading"] = dict(verdict=strict["verdict"], h_common=strict["h_common"],
                                 differs=(strict["verdict"] != sel["verdict"] or strict["h_common"] != sel["h_common"]))
    if sel["strict_reading"]["differs"]:
        sel["strict_reading"]["note"] = ("the frozen gateway V1 consistency clause |m - closed form| <= 3 se changes the verdict: "
                                         "main session decides (see readings.V1 and multiplicity)")
    h_common = sel["h_common"]
    if h_common:
        B = 2048 * 40 / h_common
        sel.update(B=B, T_N={str(N): B * h_common / N for N in (2048, 512, 128)})
    if smoke:
        sel["verdict"] = "SMOKE -- NOT A GATE RESULT (" + sel["verdict"] + ")"
        sel["strict_reading"]["verdict"] = "SMOKE -- NOT A GATE RESULT (" + sel["strict_reading"]["verdict"] + ")"
    S["selection"] = sel
    print("SELECTION:", sel["verdict"], "h_common", h_common, "| strict reading:", sel["strict_reading"]["verdict"],
          sel["strict_reading"]["h_common"], flush=True)
    return S


def _json_default(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(f"not JSON serialisable: {type(v)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", default=RES_DEFAULT)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--abf-smoke", default=None, help="default <res>/abf_smoke.json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    res = os.path.abspath(a.res)
    fig = os.path.abspath(a.fig) if a.fig else FIG_DEFAULT
    if a.smoke and (os.path.realpath(res) == os.path.realpath(RES_DEFAULT) or os.path.realpath(fig) == os.path.realpath(FIG_DEFAULT)):
        raise SystemExit("--smoke needs --res and --fig outside the production directories")
    S = analyse_all(res, a.abf_smoke or os.path.join(res, "abf_smoke.json"), a.smoke)
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, "summary.json"), "w") as f:
        json.dump(S, f, indent=1, default=_json_default)
    os.makedirs(fig, exist_ok=True)
    figures(S, res, fig)


# ---------------------------------------------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------------------------------------------
COLS = {"alpha1": "#1f4e9c", "alpha0.5": "#2a9d8f", "alpha0": "#e9a03b", "shift": "#c0392b", "lam0.5": "#7b5ea7",
        "lam0.25": "#a0739b", "lam0.1": "#d17fb8"}


def figures(S, res, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    fig, axs = plt.subplots(3, 4, figsize=(21, 13.5))
    D = S["dynamics"]

    # the h shown in the per-dynamics panels: the selected common h, else the h the per-h gates alone would select
    # (e.g. under HARNESS_FAIL), else the first candidate with data (review fix: never silently the rejected h)
    sel = S["selection"]
    h_show = sel.get("h_common") or sel.get("h_selection_h")

    def first_h(d):
        order = ([h_show] if h_show else []) + [h for h in H_CANDIDATES if h != h_show]
        for h in order:
            e = D[d]["per_h"].get(f"{h:g}")
            if e and e["em_flat"].get("n_groups"):
                return h, e
        return None, None

    shown = sorted({first_h(d)[0] for d in D if first_h(d)[0] is not None})
    h_lab = "h shown: " + (", ".join(f"{h:g}" for h in shown) if shown else "none") + \
        (" (selected)" if sel.get("h_common") and shown == [sel["h_common"]] else
         " (h the per-h gates select; harness gate not passed)" if sel.get("h_selection_h") and shown == [sel["h_selection_h"]] else
         " (NOT a selected h)")

    # (a) transverse conditional variance
    ax = axs[0, 0]
    for d in D:
        h, e = first_h(d)
        if e is None:
            continue
        vp = e["em_flat"]["var_profile"]
        ax.errorbar(vp["x"], 100 * np.array(vp["rel"]), yerr=200 * np.nan_to_num(np.array(vp["se"])), fmt="o", ms=1.5, lw=0.4,
                    color=COLS[d], label=f"{d} EM h {h:g}")
        ax.plot(vp["x"], 100 * np.array(vp["pred"]), color=COLS[d], lw=1)
    ax.axhline(2, color="#888", ls="--", lw=0.8)
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("Var(Y-E[Y|x] | x) / exact - 1 (%)")
    ax.set_title("(a) transverse conditional variance (lines: frozen-x EM)", loc="left")
    ax.legend(fontsize=6, ncol=2)
    # (b, c) free-energy error profiles
    for ax, key, lab in ((axs[0, 1], "F_density_err", "density route"), (axs[0, 2], "F_MF_err", "mean-force route")):
        for d in D:
            h, e = first_h(d)
            if e is None:
                continue
            p = e["em_flat"]["profiles"]
            ax.plot(p["centres"], p[key], color=COLS[d], lw=0.9, label=d)
        ax.axhspan(-TOL_F, TOL_F, color="#ddd", alpha=0.5, lw=0)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("F_h - F* (centred)")
        ax.set_title(f"(b) free-energy error, {lab} (band: +-{TOL_F})" if key == "F_density_err" else f"(c) free-energy error, {lab}", loc="left")
        ax.legend(fontsize=6, ncol=2)
    # (d) conditional mean force error
    ax = axs[0, 3]
    for d in D:
        h, e = first_h(d)
        if e is None:
            continue
        p = e["em_flat"]["profiles"]
        ax.plot(p["centres"], p["mf_err"], color=COLS[d], lw=0.9, label=d)
        ax.plot(p["centres"], p["em_bias_mf"], color=COLS[d], lw=0.7, ls=":")
    ax.set_xlim(-0.6, 0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("<d_x V | x>_h - F*'")
    ax.set_title("(d) conditional mean-force error (dotted: frozen-x EM)", loc="left")
    ax.legend(fontsize=6, ncol=2)
    # (e) conditional force variance
    ax = axs[1, 0]
    for d in D:
        h, e = first_h(d)
        if e is None:
            continue
        fp = e["em_flat"]["force_var_profile"]
        ax.plot(fp["x"], fp["exact"], color=COLS[d], lw=1)
        ax.errorbar(fp["x"], fp["measured"], yerr=2 * np.nan_to_num(np.array(fp["se"])), fmt="o", ms=1.6, lw=0.4, color=COLS[d], label=d)
    ax.set_xlabel("x")
    ax.set_ylabel("Var(f | x)")
    ax.set_title("(e) conditional-force variance: measured (dots) vs exact (lines)", loc="left")
    ax.legend(fontsize=6, ncol=2)
    # (f) conditional y distributions at representative x
    ax = axs[1, 1]
    zb = np.linspace(-GF.ZH_LIM, GF.ZH_LIM, GF.ZH_NB + 1)
    zc = 0.5 * (zb[1:] + zb[:-1])
    gauss = np.exp(-zc ** 2 / 2) / np.sqrt(2 * np.pi)
    for w, c in enumerate(GF.ACF_WINDOWS):
        off = 0.5 * w
        ax.plot(zc, gauss + off, color="k", lw=1.2, ls="--")
        ax.text(-4.9, off + 0.02, f"x = {c:g} +- {GF.ACF_HALF_WIDTH:g}", fontsize=7)
        for d in D:
            h, e = first_h(d)
            if e is None or "zhist" not in e["em_flat"]:
                continue
            zh = np.array(e["em_flat"]["zhist"][w])
            if zh.sum() > 0:
                ax.plot(zc, zh / (zh.sum() * (zb[1] - zb[0])) + off, color=COLS[d], lw=0.8, label=d if w == 0 else None)
    ax.set_xlabel("z_y = (y - E[Y|x]) / sd(Y|x)")
    ax.set_ylabel("density (offset per window)")
    ax.set_title("(f) conditional law of y (dashed: N(0,1))", loc="left")
    ax.legend(fontsize=6, ncol=2)
    # (g, h) conditional ACFs on the frozen-x time axis
    mk = {"-1": "o", "-0.21": "s", "0": "^"}
    for ax, q, title in ((axs[1, 2], "y", "(g) conditional ACF of z_y vs t / tau_y(x0)"),
                         (axs[1, 3], "f", "(h) conditional ACF of z_f vs t / tau_y(x0)")):
        tt = np.linspace(0, 5, 200)
        ax.plot(tt, np.exp(-tt), color="k", lw=1, ls="--", label="exp(-t/tau)")
        if q == "f":
            ax.plot(tt, np.exp(-2 * tt), color="k", lw=1, ls=":", label="exp(-2t/tau) (alpha family)")
        for d in D:
            h, e = first_h(d)
            if e is None or not e.get("acf"):
                continue
            for wk, a in e["acf"]["windows"].items():
                if ("rho_" + q) not in a:
                    continue
                t = np.array(a["t"]) / a["tau_y_frozen"]
                r = np.array(a["rho_" + q], float)
                k = t <= 5
                ax.plot(t[k], r[k], marker=mk.get(wk, "."), ms=2, lw=0.6, color=COLS[d], label=f"{d}" if wk == "-1" else None)
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlabel("t / tau_y(x0),  tau_y = 1/(lam k(x0));  markers o -1, s -0.21, ^ 0")
        ax.set_ylabel("rho")
        ax.set_title(title, loc="left")
        ax.legend(fontsize=6, ncol=2)
    # (i) integrated ACF times measured vs predicted
    ax = axs[2, 0]
    for d in D:
        for hk, e in D[d]["per_h"].items():
            if not e.get("acf"):
                continue
            for wk, a in e["acf"]["windows"].items():
                for q, mkq in (("y", "o"), ("f", "s")):
                    if f"tau_int_{q}_5tau" in a and np.isfinite(a[f"tau_int_{q}_5tau"] or np.nan):
                        ax.errorbar(a[f"tau_int_{q}_5tau_pred"], a[f"tau_int_{q}_5tau"], yerr=2 * np.nan_to_num(a[f"tau_int_{q}_5tau_se"] or 0),
                                    fmt=mkq, ms=3, color=COLS[d], mfc="none" if q == "f" else COLS[d])
    xs = np.geomspace(1e-4, 20, 10)
    ax.plot(xs, xs, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frozen-x OU prediction of int_0^{5 tau} rho")
    ax.set_ylabel("measured")
    ax.set_title("(i) integrated conditional ACF times (o: y, open s: f)", loc="left")
    # (j) trajectories
    ax = axs[2, 1]
    for d in D:
        h, e = first_h(d)
        if h is None:
            continue
        rs = load(res, d, f"em_unbiased_h{h:g}")
        if rs:
            tr = rs[0]["traces_x"]
            t = np.arange(tr.shape[0]) * rs[0]["meta"]["trace_every"] * h
            for i in range(tr.shape[1]):
                ax.plot(t, tr[:, i], color=COLS[d], lw=0.35, alpha=0.8, label=f"{d} unbiased" if i == 0 else None)
        rf = load(res, d, f"em_flat_h{h:g}")
        if rf and rf[0]["traces_x"].shape[0] > 2:
            tr = rf[0]["traces_x"][:, 0]
            t = np.arange(tr.shape[0]) * rf[0]["meta"]["trace_every"] * h
            ax.plot(t, tr, color=COLS[d], lw=0.3, ls="-", alpha=0.35)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("(j) trajectories: unbiased EM (8 walkers) and one flat-bias walker (faint)", loc="left")
    ax.legend(fontsize=6, ncol=2)
    # (k) gate statistics relative to their thresholds
    ax = axs[2, 2]
    stats = (("V1", lambda o: max(np.abs(o["var_centre"]["measured"])) / TOL_V1), ("V2 dens", lambda o: o["F_density"]["upper95"] / TOL_F_UP),
             ("V2 MF", lambda o: o["F_MF"]["upper95"] / TOL_F_UP), ("V3", lambda o: o["mean_force"]["upper95"] / TOL_FP_UP),
             ("V3b", lambda o: (o["force_var"].get("D_rel", 0.0) or 0.0) / TOL_V3B))
    names = list(D)
    width = 0.8 / max(1, len(stats))
    for k, (lab, fn) in enumerate(stats):
        vals = []
        for d in names:
            h, e = first_h(d)
            try:
                vals.append(float(fn(e["em_flat"])) if e else np.nan)
            except Exception:
                vals.append(np.nan)
        ax.bar(np.arange(len(names)) + (k - len(stats) / 2 + 0.5) * width, vals, width, label=lab)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, fontsize=7)
    ax.set_ylabel("statistic / threshold (upper bounds where defined)")
    ax.set_title("(k) gate statistics at the h shown (< 1 passes)", loc="left")
    ax.legend(fontsize=6, ncol=3)
    # (l) gate table
    ax = axs[2, 3]
    ax.axis("off")
    rows = []
    for d in names:
        for hk, e in D[d]["per_h"].items():
            g = e["gates"]
            rows.append([d, hk, g["V1"], g["V2"], g["V3"], g["V3b"], g["V4"], g["V5_core"], g["V5_abf_smoke"], e["status"]])
    # (status = per-h V1, V2, V3, V3b, V5; V4 is the h-independent harness precondition)
    if rows:
        tb = ax.table(cellText=rows, colLabels=["dynamics", "h", "V1", "V2", "V3", "V3b", "V4", "V5", "ABF smoke", "status"],
                      loc="upper center", cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(6.5)
        tb.scale(1, 1.25)
        for (r, c), cell in tb.get_celld().items():
            if r > 0 and c >= 2:
                txt = cell.get_text().get_text()
                cell.set_facecolor({"PASS": "#d8f0d8", "FAIL": "#f6d0d0", "PENDING": "#f6ecc8"}.get(txt, "#eeeeee"))
    verdict = S["selection"]["verdict"]
    short = ("SMOKE: " + verdict[len("SMOKE -- NOT A GATE RESULT ("):].split(" (")[0]) if verdict.startswith("SMOKE") else verdict.split(" (")[0]
    ax.set_title("(l) gates (V4 h-independent); verdict " + short +
                 (f", h = {S['selection']['h_common']:g}" if S["selection"].get("h_common") else ""), loc="left")
    sup = f"Experiment I/II numerical-validation gate (SCIENTIFIC_PLAN section 3) -- panels (a)-(h), (j), (k): {h_lab}"
    if S.get("smoke"):
        sup = "SMOKE RUN -- NOT A GATE RESULT -- " + sup
    fig.suptitle(sup, fontsize=11, color="#c0392b" if S.get("smoke") else "k")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"exp1_timestep_validation.{ext}"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
