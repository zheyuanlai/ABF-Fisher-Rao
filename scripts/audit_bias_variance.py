#!/usr/bin/env python
"""mFR mechanism audit, step 6 -- bias / variance decomposition of the *mean-force*
estimator, from EXISTING seeds only.  Read-only re-analysis: no simulation is run.

For each system, cell (physical config) and method, using independent seeds ``s``:

    mean_Fp(z)  = mean_s  Fp_hat_s(z)
    bias2(z)    = [ mean_Fp(z) - Fp_ref(z) ]^2          (plug-in; biased up by var/S)
    variance(z) = Var_s Fp_hat_s(z)                     (sample variance, ddof=1)

and the exact per-bin identity

    mean_s [Fp_hat_s(z) - Fp_ref(z)]^2 = bias2_plugin(z) + (S-1)/S * variance(z)

Both are integrated under two spatial weightings (plus one sensitivity window):

  * ``thermal_uniform_10kT``  uniform over  eval-window AND  F_ref - min(F_ref) <= 10 kT
  * ``thermal_uniform_20kT``  same with 20 kT (sensitivity)
  * ``equilibrium``           w(z) proportional to exp(-beta [F_ref(z) - min F_ref]),
                              restricted to the eval window and renormalised

All weights are normalised to sum to 1, so every integrated quantity is a
*weighted mean over bins* and ``sqrt(int_mse_direct)`` is directly comparable to the
RMS L2(F') numbers the repo reports.

Reported per (system, cohort, cell, method, weighting):

  int_bias2_plugin      sum_z w(z) bias2_plugin(z)
  int_bias2_debiased    int_bias2_plugin - int_var / S      (unbiased for the true bias^2;
                                                             may be negative when bias ~ 0)
  int_var               sum_z w(z) variance(z)
  int_mse_direct        sum_z w(z) mean_s [Fp_hat_s - Fp_ref]^2   (what one seed actually costs)
  int_mse_pop           int_bias2_debiased + int_var  (unbiased estimate of population MSE)
  bias_fraction         int_bias2_debiased / int_mse_pop
  + seed-level bootstrap 95% percentile CIs.

IMPORTANT -- what this script does NOT do.  A "split-half" allocation-variance proxy
requires partitioning the population by ANCESTOR FAMILY.  No engine in this repository
persists per-particle ancestor labels (only ancestor ESS / n_unique / max-fraction
scalars) and no engine keeps per-family mean-force accumulators, so a family-level split
cannot be reconstructed from existing artifacts.  That is recorded as a blocking data gap
in ``feasibility.json``; no particle-level fallback is computed.

Outputs (all under results/mfr_mechanism_audit/bias_variance/):
  feasibility.json                  per-system feasibility verdict + split-half verdict
  bias_variance_integrated.csv      the decomposition table
  bias_variance_bins.csv            per-bin bias2 / variance profiles
  validation_l2_crosscheck.csv      recomputed repo-native L2(F') vs the stored scalar
"""
from __future__ import annotations

import glob
import json
import math
import os
import zlib
from collections import OrderedDict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "mfr_mechanism_audit", "bias_variance")
RES = os.path.join(REPO, "results")
EPS = 1.0e-12
N_BOOT = 2000
BOOT_SEED = 20260721

# ---------------------------------------------------------------------------
# core decomposition
# ---------------------------------------------------------------------------


def build_weightings(F_ref, beta, eval_mask, deltas_kT=(10.0, 20.0)):
    """Normalised (sum == 1) spatial weight maps.  ``F_ref`` in energy units."""
    F = np.asarray(F_ref, float)
    em = np.asarray(eval_mask, bool)
    Fm = F - F[em].min()
    out = OrderedDict()
    for d in deltas_kT:
        m = em & (Fm <= d / beta)
        w = m.astype(float)
        out["thermal_uniform_%gkT" % d] = (w / max(w.sum(), EPS), int(m.sum()))
    w = np.where(em, np.exp(-beta * Fm), 0.0)
    out["equilibrium"] = (w / max(w.sum(), EPS), int(em.sum()))
    return out


def decompose(Fp, Fp_ref, comp_axis=None):
    """Per-bin bias^2 / variance / mean squared error.

    ``Fp``: (S, ...) per-seed profiles; ``Fp_ref``: (...) reference.  If ``comp_axis`` is
    given it indexes a vector component axis of ``Fp`` (after the seed axis) which is
    summed over, i.e. bias2 = sum_c (.)^2 and var = sum_c Var_s(.).
    """
    Fp = np.asarray(Fp, float)
    ref = np.asarray(Fp_ref, float)
    S = Fp.shape[0]
    mean = Fp.mean(axis=0)
    var = Fp.var(axis=0, ddof=1)
    b2 = (mean - ref) ** 2
    sq = ((Fp - ref) ** 2).mean(axis=0)
    if comp_axis is not None:
        b2 = b2.sum(axis=comp_axis - 1)
        var = var.sum(axis=comp_axis - 1)
        sq = sq.sum(axis=comp_axis - 1)
        mean = None
    return S, mean, b2, var, sq


def integrate(w, S, b2, var, sq):
    ib2 = float(np.sum(w * b2))
    iv = float(np.sum(w * var))
    isq = float(np.sum(w * sq))
    ib2d = ib2 - iv / S
    return dict(int_bias2_plugin=ib2, int_bias2_debiased=ib2d, int_var=iv,
                int_mse_direct=isq, int_mse_pop=ib2d + iv,
                bias_fraction=ib2d / (ib2d + iv) if abs(ib2d + iv) > EPS else float("nan"),
                rms_total=math.sqrt(max(isq, 0.0)))


BOOT_KEYS = ["int_bias2_debiased", "int_var", "int_mse_direct", "bias_fraction"]


def boot_index_matrix(S, n_boot, seed):
    """(n_boot, S) resample-with-replacement index matrix; rows with <2 distinct seeds
    are redrawn so a bootstrap replicate always admits a sample variance."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, S, size=(n_boot, S))
    bad = np.array([len(np.unique(r)) < 2 for r in idx])
    while bad.any():
        idx[bad] = rng.integers(0, S, size=(int(bad.sum()), S))
        bad = np.array([len(np.unique(r)) < 2 for r in idx])
    return idx


def bootstrap_curves(Fp, ref, w, idx, comp_axis=None):
    """Bootstrap replicates of the integrated quantities, one row per resample."""
    out = {k: np.empty(idx.shape[0]) for k in BOOT_KEYS}
    for j, ii in enumerate(idx):
        Sb, _m, b2, var, sq = decompose(Fp[ii], ref, comp_axis=comp_axis)
        r = integrate(w, Sb, b2, var, sq)
        for k in BOOT_KEYS:
            out[k][j] = r[k]
    return out


def pct_ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size < 20:
        return (float("nan"), float("nan"))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# ---------------------------------------------------------------------------
# repo-native L2(F') reproductions, used only as a validation cross-check
# ---------------------------------------------------------------------------


def l2_wca(profile, ref, grid, mask):
    p, r, g = np.asarray(profile, float)[mask], np.asarray(ref, float)[mask], np.asarray(grid, float)[mask]
    return math.sqrt(np.trapezoid((p - r) ** 2, g) / (g[-1] - g[0]))


def l2_rms(profile, ref, mask):
    d = (np.asarray(profile, float) - np.asarray(ref, float))[mask]
    return math.sqrt(float(np.mean(d * d)))


def l2_interval(profile, ref, dz, mask):
    p, r = np.asarray(profile, float), np.asarray(ref, float)
    w = mask.astype(float)
    width = np.sum(w) * dz
    return math.sqrt(np.sum((p - r) ** 2 * w) * dz / max(width, EPS))


def spectral_gradient_np(B, dz1, dz2):
    n1, n2 = B.shape[-2], B.shape[-1]
    k1 = (2.0 * math.pi * np.fft.fftfreq(n1, d=dz1))[:, None]
    k2 = (2.0 * math.pi * np.fft.fftfreq(n2, d=dz2))[None, :]
    Bh = np.fft.fft2(B)
    return np.real(np.fft.ifft2(1j * k1 * Bh)), np.real(np.fft.ifft2(1j * k2 * Bh))


# ---------------------------------------------------------------------------
# loaders -- each returns a list of "cases"
# ---------------------------------------------------------------------------
# A case is a dict with:
#   system, cohort, cell, method, arm_note, source, n_seeds,
#   Fp (S,...), Fp_ref (...), F_ref (...), beta, eval_mask, grid (or grid1/grid2),
#   comp_axis (None or 1), validate: list of (seed, recomputed_l2, stored_l2, convention)


# wca eval window: SimConfig defaults are 0.0/1.0 but every production yaml
# (configs/wca_representative.yaml, configs/wca_phase_diagram_production.yaml) sets
# eval_z_lo=-0.1, eval_z_hi=1.1.  The window is not stored in the npz; the
# validation_l2_crosscheck.csv output confirms this choice reproduces the stored l2_fp.
WCA_EVAL_LO, WCA_EVAL_HI = -0.1, 1.1


def _wca_cases(subdir, prefix, cells, methods, cohort):
    cases = []
    for cell in cells:
        for meth in methods:
            fs = sorted(glob.glob(os.path.join(
                RES, subdir, "raw", "%s__%s__%s__seed*__*.npz" % (prefix, meth, cell))))
            if not fs:
                continue
            Fp, seeds, val = [], [], []
            grid = ref_mf = ref_F = None
            beta = None
            for f in fs:
                d = np.load(f, allow_pickle=True)
                grid = d["grid"].astype(float)
                ref_mf = d["ref_mean_force"].astype(float)
                ref_F = d["ref_free_energy"].astype(float)
                beta = float(d["beta"])
                mf = d["final_mean_force"].astype(float)
                Fp.append(mf)
                seeds.append(int(d["seed"]))
                em = (grid >= WCA_EVAL_LO) & (grid <= WCA_EVAL_HI)
                val.append((int(d["seed"]), l2_wca(mf, ref_mf, grid, em), float(d["l2_fp"]),
                            "trapezoid over z in [%.1f,%.1f] / span "
                            "(wca_abffr_core.profile_l2_error_np)" % (WCA_EVAL_LO, WCA_EVAL_HI)))
            Fp = np.stack(Fp)
            cases.append(dict(system="wca_dimer", cohort=cohort, cell=cell, method=meth,
                              arm_note="fr_rate=0.1, fr_every=5, target_ema_rate=0.005, "
                                       "N=1024, T=120000" if meth != "abf" else
                                       "fr_rate=0 (baseline), N=1024, T=120000",
                              source=os.path.join("results", subdir, "raw"),
                              n_seeds=len(seeds), seeds=seeds, Fp=Fp, Fp_ref=ref_mf,
                              F_ref=ref_F, beta=beta,
                              eval_mask=(grid >= WCA_EVAL_LO) & (grid <= WCA_EVAL_HI),
                              grid=grid, comp_axis=None, validate=val))
    return cases


def load_wca():
    cells6 = ["b1_h2_w2_n10_a1.5", "b1_h4_w2_n10_a1.5", "b2_h4_w2_n10_a1.5",
              "b2_h6_w2_n10_a1.5", "b4_h1_w2_n10_a1.5", "b4_h2_w2_n10_a1.5"]
    m5 = ["abf", "fr_estimated", "fr_uniform", "fr_oracle", "fr_estimated_adaptive"]
    cases = _wca_cases("wca_representative", "representative", cells6, m5, "representative_10seed")
    allcells = sorted({os.path.basename(p).split("__")[2]
                       for p in glob.glob(os.path.join(RES, "wca_phase_diagram", "production",
                                                       "raw", "production__abf__*.npz"))})
    cases += _wca_cases("wca_phase_diagram/production", "production", allcells,
                        ["abf", "fr_estimated", "fr_uniform", "fr_oracle"],
                        "phase_production_4seed")
    return cases


def _eb_like(root, files_by_key, system, cohort, xlo, xhi, arm_notes):
    cases = []
    for (cell, meth), fs in sorted(files_by_key.items()):
        Fp, seeds, val = [], [], []
        xg = Fpref = Fref = None
        beta = None
        for f in fs:
            d = np.load(f, allow_pickle=True)
            xg = d["x_grid"].astype(float)
            Fpref = d["Fp_ref"].astype(float)
            Fref = d["F_ref"].astype(float)
            beta = float(d["cfg__beta"])
            fp = d["Fp_hat"].astype(float)
            Fp.append(fp)
            seeds.append(int(d["seed"]))
            em = (xg >= xlo) & (xg <= xhi)
            val.append((int(d["seed"]), l2_rms(fp, Fpref, em), float(d["final_l2_fp"]),
                        "RMS over interior window x in [%.1f,%.1f] (eb/edb core l2_error)" % (xlo, xhi)))
        Fp = np.stack(Fp)
        em = (xg >= xlo) & (xg <= xhi)
        cases.append(dict(system=system, cohort=cohort, cell=cell, method=meth,
                          arm_note=arm_notes.get(meth, ""), source=root,
                          n_seeds=len(seeds), seeds=seeds, Fp=Fp, Fp_ref=Fpref, F_ref=Fref,
                          beta=beta, eval_mask=em, grid=xg, comp_axis=None, validate=val))
    return cases


def load_eb():
    """Entropic bottleneck.  Base cell = omega_in25_beta8_gamma15.

    Stages are NOT pooled: the same seed index in different stages gives different
    trajectories, and stage2_omega uses a different dt / n_steps.  stage1_seeds
    (20 seeds, abf + fr_estimated) and stage0_reproduce (5 seeds, 4 methods) are
    reported as separate cohorts.
    """
    cases = []
    for stage, cohort in [("stage1_seeds", "stage1_seeds_20seed"),
                          ("stage0_reproduce", "stage0_reproduce_5seed")]:
        by = {}
        for f in sorted(glob.glob(os.path.join(RES, "entropic_bottleneck", "raw", stage,
                                               "*omega_in25_beta8_gamma15_seed*.npz"))):
            meth = os.path.basename(f).split("__")[0]
            by.setdefault(("omega_in25_beta8_gamma15", meth), []).append(f)
        cases += _eb_like(os.path.join("results/entropic_bottleneck/raw", stage),
                          by, "entropic_bottleneck", cohort, -1.5, 1.5,
                          {"abf": "fr off", "fr_estimated": "eta=locked stage config",
                           "fr_uniform": "eta=locked stage config",
                           "fr_oracle": "eta=locked stage config"})
    return cases


def load_edb():
    """Entropy-dominant bottleneck, sweep_20260614_015145 raw/main only.

    raw/rate (the fr-rate probe arms) is deliberately excluded so that one method label
    never mixes two FR rates.
    """
    by = {}
    for f in sorted(glob.glob(os.path.join(
            RES, "entropy_dominant_bottleneck", "sweep_20260614_015145", "raw", "main", "*.npz"))):
        b = os.path.basename(f)
        meth = b.split("__")[0]
        cell = b.split("__")[1].split("_seed")[0]
        by.setdefault((cell, meth), []).append(f)
    return _eb_like("results/entropy_dominant_bottleneck/sweep_20260614_015145/raw/main",
                    by, "entropy_dominant_bottleneck", "main_sweep_20seed", -1.5, 1.5,
                    {"abf": "fr off", "fr_estimated": "main-arm rate (raw/main only)",
                     "fr_uniform": "main-arm rate", "fr_oracle": "main-arm rate"})


def load_r15():
    """Pentane R15 distance CV, beta=2, production stage (8 seeds bundled per npz)."""
    cases = []
    for f in sorted(glob.glob(os.path.join(RES, "alkanes_cv_extension", "r15_methods", "raw",
                                           "production__dist__pentane__*.npz"))):
        d = np.load(f, allow_pickle=True)
        spec = json.loads(str(d["spec_json"]))
        label = os.path.basename(f).split("__")[3]          # abf / fr_estimated / fr_active / ...
        beta = float(d["beta"])
        grid = d["grid"].astype(float)
        F_ref = d["ref_F"].astype(float)
        Fp_ref = d["ref_Fprime"].astype(float)
        dz = float(d["dz"])
        td = float(d["thermal_delta"])
        mf = d["final_mean_force"].astype(float)            # (S, 256) per seed
        per_seed = json.loads(str(d["per_seed"]))
        tmask = (F_ref - F_ref.min()) <= td
        val = [(int(ps["seed"]), l2_interval(mf[i], Fp_ref, dz, tmask),
                float(ps["final_l2_Fp"]),
                "interval L2 over repo thermal mask (F_ref-min<=%.1f) (metrics_cv._interval_l2, align=False)" % td)
               for i, ps in enumerate(per_seed)]
        cases.append(dict(system="pentane_R15", cohort="production_8seed",
                          cell="pentane_R15_beta2_trans",
                          method=label,
                          arm_note="engine method=%s, fr_rate=%g, N=1024, T=80000" %
                                   (spec["method"], spec["fr_rate"]),
                          source="results/alkanes_cv_extension/r15_methods/raw",
                          n_seeds=mf.shape[0], seeds=[int(s) for s in d["seeds"]],
                          Fp=mf, Fp_ref=Fp_ref, F_ref=F_ref, beta=beta,
                          eval_mask=np.ones_like(F_ref, bool), grid=grid,
                          comp_axis=None, validate=val))
    return cases


def load_torus():
    """2-D torsion torus.  The raw mean-force accumulator is NOT persisted; only the
    per-seed reconstructed PMF is.  The mean force used here is the SPECTRAL GRADIENT of
    that stored per-seed PMF -- i.e. its curl-free part -- which is exactly the repo's own
    2-D mean-force error convention (metrics_cv.meanforce_vector_error).  Labelled DERIVED.
    """
    cases = []
    for f in sorted(glob.glob(os.path.join(RES, "alkanes_cv_extension", "2d_methods", "raw",
                                           "production__joint2d__pentane__*.npz"))):
        d = np.load(f, allow_pickle=True)
        spec = json.loads(str(d["spec_json"]))
        label = os.path.basename(f).split("__")[3]
        beta = float(d["beta"])
        g1 = d["grid1"].astype(float)
        g2 = d["grid2"].astype(float)
        dphi = float(d["dphi"])
        F_ref = d["ref_joint_F"].astype(float)
        pmf = d["final_pmf"].astype(float)                  # (S, 48, 48)
        r1, r2 = spectral_gradient_np(F_ref, dphi, dphi)
        ref_vec = np.stack([r1, r2])                        # (2, 48, 48)
        grads = np.stack([np.stack(spectral_gradient_np(pmf[s], dphi, dphi))
                          for s in range(pmf.shape[0])])    # (S, 2, 48, 48)
        td = float(d["thermal_delta"])
        tmask = (F_ref - F_ref.min()) <= td
        per_seed = json.loads(str(d["per_seed"]))
        val = []
        for i, ps in enumerate(per_seed):
            err = ((grads[i, 0] - r1) ** 2 + (grads[i, 1] - r2) ** 2)
            rec = math.sqrt(float(np.sum(err * tmask) / max(np.sum(tmask), EPS)))
            val.append((int(ps["seed"]), rec, float(ps["meanforce_vec_err"]),
                        "RMS |grad F_hat - grad F_ref| over repo thermal mask "
                        "(metrics_cv.meanforce_vector_error)"))
        cases.append(dict(system="pentane_2d_torsion_torus", cohort="production_6seed_DERIVED",
                          cell="pentane_joint2d_beta2_trans", method=label,
                          arm_note="DERIVED mean force = spectral grad of stored per-seed PMF; "
                                   "engine method=%s, fr_rate=%g, N=2048, T=45000" %
                                   (spec["method"], spec["fr_rate"]),
                          source="results/alkanes_cv_extension/2d_methods/raw",
                          n_seeds=grads.shape[0], seeds=[int(s) for s in d["seeds"]],
                          Fp=grads, Fp_ref=ref_vec, F_ref=F_ref, beta=beta,
                          eval_mask=np.ones_like(F_ref, bool), grid=g1, grid2=g2,
                          comp_axis=1, validate=val))
    return cases


def load_two_dim_xi_x():
    """2-D metastability toy (xi = x).  Per-seed F'(x) lives in the profiles CSV.

    Filter: production_gpu stage; eta=0.075 and burnin_fraction=0.0 for every FR arm so
    the estimator settings match the abf_only baseline; FR arms are kept SEPARATE per
    gamma (FR rate) -- pooling gammas would average gentle and aggressive arms.
    """
    import csv
    path = os.path.join(RES, "two_dim_xi_x", "production_gpu", "production_gpu_profiles.csv")
    if not os.path.exists(path):
        return []
    rows = {}
    xs = {}
    ref = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            meth, tt = r["method"], r["target_type"]
            gamma, eta, bi = float(r["gamma"]), float(r["eta"]), float(r["burnin_fraction"])
            if meth != "abf_only" and not (abs(eta - 0.075) < 1e-9 and abs(bi) < 1e-9):
                continue
            key = ("abf_only", 0.0) if meth == "abf_only" else (meth, gamma)
            rows.setdefault(key, {}).setdefault(int(r["seed"]), []).append(
                (float(r["x"]), float(r["Fprime_hat"])))
            ref.setdefault(key, []).append((float(r["x"]), float(r["Fprime_ref"]), float(r["F_ref"])))
    cases = []
    for key, byseed in sorted(rows.items()):
        seeds = sorted(byseed)
        arr = []
        for s in seeds:
            pts = sorted(byseed[s])
            arr.append([v for _x, v in pts])
            xs[key] = np.array([x for x, _v in pts])
        Fp = np.asarray(arr, float)
        rr = sorted(set(ref[key]))
        x = np.array([a for a, _b, _c in rr])
        Fp_ref = np.array([b for _a, b, _c in rr])
        F_ref = np.array([c for _a, _b, c in rr])
        assert np.allclose(x, xs[key])
        cases.append(dict(system="two_dim_xi_x_toy", cohort="production_gpu_5seed",
                          cell="xi_eq_x_beta4", method=key[0] if key[0] == "abf_only"
                          else "%s_g%g" % (key[0], key[1]),
                          arm_note="eta=0.075, burnin=0.0, gamma(FR rate)=%g, beta=4, 5 seeds" % key[1],
                          source="results/two_dim_xi_x/production_gpu/production_gpu_profiles.csv",
                          n_seeds=len(seeds), seeds=seeds, Fp=Fp, Fp_ref=Fp_ref, F_ref=F_ref,
                          beta=4.0, eval_mask=np.ones_like(F_ref, bool), grid=x,
                          comp_axis=None, validate=[]))
    return cases


# ---------------------------------------------------------------------------
# split-half feasibility probe (data-availability only -- computes nothing)
# ---------------------------------------------------------------------------


def splithalf_feasibility():
    """Check every raw family for a persisted per-particle ancestor label array."""
    probes = [
        ("wca_dimer", "results/wca_representative/raw/*.npz"),
        ("wca_dimer", "results/wca_phase_diagram/production/raw/*.npz"),
        ("entropic_bottleneck", "results/entropic_bottleneck/raw/stage1_seeds/*.npz"),
        ("entropy_dominant_bottleneck",
         "results/entropy_dominant_bottleneck/sweep_20260614_015145/raw/main/*.npz"),
        ("pentane_R15", "results/alkanes_cv_extension/r15_methods/raw/*.npz"),
        ("pentane_2d_torsion_torus", "results/alkanes_cv_extension/2d_methods/raw/*.npz"),
        ("alkanes_dihedral", "results/alkanes/production/raw/*.npz"),
    ]
    out = []
    for system, pat in probes:
        fs = sorted(glob.glob(os.path.join(REPO, pat)))
        if not fs:
            out.append(dict(system=system, pattern=pat, files=0, ancestor_fields=[],
                            per_particle_ancestor_labels=False, note="no files"))
            continue
        d = np.load(fs[0], allow_pickle=True)
        anc = [k for k in d.files
               if ("ancestor" in k.lower() or "ess" in k.lower().split("_"))]
        shapes = {k: list(np.shape(d[k])) for k in anc}
        # replica count, however this engine family names it
        nrep = None
        for k in ("n_replicas", "cfg__N", "N"):
            if k in d.files:
                try:
                    nrep = int(d[k])
                    break
                except Exception:
                    pass
        if nrep is None:
            try:
                nrep = int(json.loads(str(d["spec_json"]))["n_replicas"])
            except Exception:
                nrep = None
        # ANY stored array carrying a replica axis would be needed for a particle-level
        # (let alone family-level) partition; list them all, not just ancestor-named ones
        rep_axis_fields = ({k: list(np.shape(d[k])) for k in d.files
                            if nrep is not None and nrep in np.shape(d[k])}
                           if nrep is not None else {})
        per_particle = any(nrep is not None and nrep in shapes[k] for k in anc)
        out.append(dict(system=system, pattern=pat, files=len(fs), n_replicas=nrep,
                        ancestor_or_ess_fields=shapes,
                        fields_with_a_replica_axis=rep_axis_fields,
                        per_particle_ancestor_labels=bool(per_particle)))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUT, exist_ok=True)
    loaders = [("wca_dimer", load_wca), ("entropic_bottleneck", load_eb),
               ("entropy_dominant_bottleneck", load_edb), ("pentane_R15", load_r15),
               ("pentane_2d_torsion_torus", load_torus), ("two_dim_xi_x_toy", load_two_dim_xi_x)]
    cases = []
    feas = []
    for name, fn in loaders:
        cs = fn()
        cases += cs
        feas.append(dict(system=name, n_cases=len(cs),
                         cohorts=sorted({c["cohort"] for c in cs}),
                         methods=sorted({c["method"] for c in cs}),
                         seeds_per_case=sorted({c["n_seeds"] for c in cs})))

    # one bootstrap index matrix per (system, cohort, cell, n_seeds) so that arms with the
    # same seed list are resampled MATCHED -- the ratio CIs below are then matched-seed.
    idx_cache = {}

    def get_idx(c):
        k = (c["system"], c["cohort"], c["cell"], c["n_seeds"])
        if k not in idx_cache:
            # deterministic (str hashing is salted per process, crc32 is not)
            idx_cache[k] = boot_index_matrix(
                c["n_seeds"], N_BOOT, BOOT_SEED + zlib.crc32(repr(k).encode()) % 10_000)
        return idx_cache[k]

    int_rows, bin_rows, val_rows = [], [], []
    boot_store = {}
    for c in cases:
        S, mean, b2, var, sq = decompose(c["Fp"], c["Fp_ref"], comp_axis=c["comp_axis"])
        W = build_weightings(c["F_ref"], c["beta"], c["eval_mask"])
        bidx = get_idx(c) if S >= 3 else None
        for wname, (w, nbin) in W.items():
            r = integrate(w, S, b2, var, sq)
            if bidx is not None:
                bc = bootstrap_curves(c["Fp"], c["Fp_ref"], w, bidx, comp_axis=c["comp_axis"])
                boot_store[(c["system"], c["cohort"], c["cell"], c["method"], wname)] = bc
                for k in BOOT_KEYS:
                    lo, hi = pct_ci(bc[k])
                    r[k + "_lo95"], r[k + "_hi95"] = lo, hi
            int_rows.append(dict(system=c["system"], cohort=c["cohort"], cell=c["cell"],
                                 method=c["method"], arm_note=c["arm_note"],
                                 source=c["source"], n_seeds=S,
                                 seeds=";".join(str(s) for s in c["seeds"]),
                                 weighting=wname, n_bins_in_window=nbin, beta=c["beta"],
                                 **r))
        # per-bin profiles under the 10 kT window flag + equilibrium weight
        w10 = W["thermal_uniform_10kT"][0]
        weq = W["equilibrium"][0]
        flat = b2.ravel()
        idx = np.arange(flat.size)
        if c["comp_axis"] is None:
            z1 = c["grid"]
            z2 = np.full_like(z1, np.nan)
            ref_flat = np.asarray(c["Fp_ref"], float).ravel()
            mean_flat = np.asarray(mean, float).ravel()
        else:
            g1, g2 = np.meshgrid(c["grid"], c["grid2"], indexing="ij")
            z1, z2 = g1.ravel(), g2.ravel()
            ref_flat = np.full(flat.size, np.nan)
            mean_flat = np.full(flat.size, np.nan)
        for i in idx:
            bin_rows.append(dict(system=c["system"], cohort=c["cohort"], cell=c["cell"],
                                 method=c["method"], bin_index=int(i),
                                 z=float(z1.ravel()[i]), z2=float(z2.ravel()[i]),
                                 Fp_ref=float(ref_flat[i]), mean_Fp=float(mean_flat[i]),
                                 bias2=float(flat[i]), variance=float(var.ravel()[i]),
                                 mean_sq_err=float(sq.ravel()[i]),
                                 w_thermal10=float(w10.ravel()[i]), w_eq=float(weq.ravel()[i]),
                                 n_seeds=S))
        for seed, rec, stored, conv in c["validate"]:
            val_rows.append(dict(system=c["system"], cohort=c["cohort"], cell=c["cell"],
                                 method=c["method"], seed=seed, recomputed_l2_fp=rec,
                                 stored_l2_fp=stored, abs_diff=abs(rec - stored),
                                 convention=conv))

    # paired ratios vs the abf arm in the same (system, cohort, cell, weighting)
    key = lambda r: (r["system"], r["cohort"], r["cell"], r["weighting"])
    base = {key(r): r for r in int_rows if r["method"] in ("abf", "abf_only")}
    for r in int_rows:
        b = base.get(key(r))
        if b is None:
            r["var_ratio_vs_abf"] = float("nan")
            r["bias2_ratio_vs_abf"] = float("nan")
            r["mse_ratio_vs_abf"] = float("nan")
        else:
            r["var_ratio_vs_abf"] = r["int_var"] / b["int_var"] if b["int_var"] > 0 else float("nan")
            r["bias2_ratio_vs_abf"] = (r["int_bias2_debiased"] / b["int_bias2_debiased"]
                                       if abs(b["int_bias2_debiased"]) > EPS else float("nan"))
            r["mse_ratio_vs_abf"] = (r["int_mse_direct"] / b["int_mse_direct"]
                                     if b["int_mse_direct"] > 0 else float("nan"))
        # matched-seed bootstrap CI on each ratio (same resample index rows in both arms)
        bk = (r["system"], r["cohort"], r["cell"], r["method"], r["weighting"])
        bb = None if b is None else (b["system"], b["cohort"], b["cell"], b["method"],
                                     b["weighting"])
        for lab, src in (("var_ratio_vs_abf", "int_var"),
                         ("bias2_ratio_vs_abf", "int_bias2_debiased"),
                         ("mse_ratio_vs_abf", "int_mse_direct")):
            lo = hi = float("nan")
            if bb is not None and bk in boot_store and bb in boot_store and bk != bb:
                num, den = boot_store[bk][src], boot_store[bb][src]
                with np.errstate(divide="ignore", invalid="ignore"):
                    lo, hi = pct_ci(np.where(np.abs(den) > EPS, num / den, np.nan))
            r[lab + "_lo95"], r[lab + "_hi95"] = lo, hi

    import csv as _csv

    def dump(path, rows):
        if not rows:
            return
        cols = list(OrderedDict((k, None) for r in rows for k in r))
        with open(path, "w", newline="") as fh:
            wtr = _csv.DictWriter(fh, fieldnames=cols)
            wtr.writeheader()
            for r in rows:
                wtr.writerow(r)

    dump(os.path.join(OUT, "bias_variance_integrated.csv"), int_rows)
    dump(os.path.join(OUT, "bias_variance_bins.csv"), bin_rows)
    dump(os.path.join(OUT, "validation_l2_crosscheck.csv"), val_rows)

    sh = splithalf_feasibility()
    feasibility = dict(
        generated_by="scripts/audit_bias_variance.py",
        read_only=True,
        definition=dict(
            bias2="[mean_s Fp_hat_s(z) - Fp_ref(z)]^2 (plug-in) and its debiased form "
                  "plug-in - Var_s/S",
            variance="sample variance over independent seeds, ddof=1",
            identity="mean_s (Fp_hat_s - Fp_ref)^2 = bias2_plugin + (S-1)/S * variance",
            weightings=["thermal_uniform_10kT: uniform over eval window AND "
                        "F_ref - min <= 10 kT",
                        "thermal_uniform_20kT: same at 20 kT (sensitivity)",
                        "equilibrium: w proportional to exp(-beta (F_ref - min)) over the "
                        "eval window, renormalised"],
            note="all weights sum to 1, so integrated quantities are weighted means over "
                 "bins and sqrt(int_mse_direct) is an RMS comparable to reported L2(F')"),
        per_system=feas,
        split_half_proxy=dict(
            attempted=False,
            verdict="IMPOSSIBLE from existing artifacts -- blocking data gap",
            reason="An ancestor-FAMILY split-half requires (a) per-particle ancestor labels "
                   "at the final time and (b) per-family mean-force accumulators. Neither is "
                   "persisted: the engines keep ancestor labels only in memory and save "
                   "ancestor ESS / n_unique / max-fraction scalars, and every system uses a "
                   "single global mean-force accumulator shared by all replicas. No "
                   "particle-level fallback was computed, per the audit constraint.",
            probe=sh,
            requires="new instrumented runs that dump per-particle ancestor labels and "
                     "family-partitioned mean-force accumulators"),
    )
    with open(os.path.join(OUT, "feasibility.json"), "w") as fh:
        json.dump(feasibility, fh, indent=2)

    print("cases: %d ; integrated rows: %d ; bin rows: %d ; validation rows: %d"
          % (len(cases), len(int_rows), len(bin_rows), len(val_rows)))
    if val_rows:
        mx = max(v["abs_diff"] for v in val_rows)
        print("max |recomputed - stored| L2(F') over %d validation rows: %.3e" % (len(val_rows), mx))
    print("outputs -> %s" % OUT)


if __name__ == "__main__":
    main()
