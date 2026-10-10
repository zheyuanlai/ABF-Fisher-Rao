#!/usr/bin/env python
"""Mechanism-specific analysis of Experiments I and II (docs/mechanism/SCIENTIFIC_PLAN.md sections 4, 6, 7, incl.
Amendment 1): the passive engine diagnostics D1-D4 per cell and N, and the synthesis figures S1-S6 + the descriptive
mechanistic plots.

    python scripts/mechanism/mech_analysis.py [--cells-root configs/mechanism/cells]
        [--results-override DIR]   runs read from DIR/<experiment>/<out_dir>/N<N>/s<seed>_<method>.npz
                                   (default: each cell config's results_dir)
        [--analysis-override DIR]  cell summaries / per-run metric caches from DIR/<experiment>/<cell>[/analysis]/
                                   (default: each cell config's analysis_dir)
        [--validation results/mechanism/validation/summary.json] [--cross-cell results/mechanism/synthesis/cross_cell.json]
        [--out ROOT]               outputs under ROOT/figures/mechanism/... and ROOT/results/mechanism/... (default:
                                   the repository root)
        [--n-boot 10000] [--experiment ...] [--cells ...] [--N ...] [--no-figures] [--dpi 110]

Inputs.  Raw runs (src/gateway_family_numba.py result contract: D1 fr_death_pos_hist_cum / fr_birth_pos_hist_cum,
D2 dep_age_C/M/M2 (n_snap, 5, nb), D3 dep_cross_C (n_snap, 5, nb), D4 traces_y, X_final / Y_final); each run is
checked against its cell config by eqb_family.check_plan (B, h, engine knobs, engine, model fields; path N / seed /
method).  The cells' analysis summaries (scripts/equal_budget/analyze_ladder.py --system gateway_family) and their
per-run metric caches (run_metrics/N<N>/s<seed>_<method>.npz: e_F, e_Fp_stat curves).  The validation summary
(scripts/mechanism/analyze_validation.py: conditional ACF times and measured Var(f | x)) and cross_cell.json
(scripts/mechanism/cross_cell.py) are optional: when absent the script says so (completeness table, JSON, the
affected panels) and falls back to the exact / frozen-x predictions (S4, mechanistic plots) or computes the
cross-cell contrasts itself with cross_cell.analyse (verify_runs=False, labelled 'provisional').  Missing cells,
runs, diagnostics and summaries are listed, never skipped silently; a completeness table is printed.

Outputs.  Per cell and N: figures/mechanism/<experiment>/<cell>/N<N>/mech_{d1_fr_events, d2_age_fractions,
d2_mean_force_bias, d2_force_variance, d3_crossing, ylaw, ytraces}.{pdf,png}; per cell
results/mechanism/<experiment>/<cell>/analysis/mech_diagnostics.json.  Synthesis: figures/mechanism/synthesis/
mech_{s1_models, s2_convergence_<experiment>_N<N>_{u,t}, s3_gain_vs_entropic_fraction, s3b_cross_cell_contrasts,
s4_force_variance_relaxation, s5_lambda, s5b_d2_conditional_relaxation, s5c_d2_matched_free_energy,
s6_marginal_vs_free_energy, m1_gain_vs_covariates}.{pdf,png} and results/mechanism/synthesis/mech_synthesis.json
(S5b_D2_readout: the Experiment II D2 readouts per lambda and N, the preregistered B_c / B_c(lambda) -
B_c(1) and the EXPLORATORY analysis-defined matched excess, side by side).  Every figure is checked with fig_legibility.check_figure_safe (the
result is stored with the figure's entry); figures of quantities the plan does not preregister say EXPLORATORY,
the mechanistic covariate plots say 'descriptive, not causal', and non-production (smoke) data are stamped.

Definitions (bins: the 180 production bins of width 0.02 on [-1.8, 1.8]; regions by bin centre: left x < -0.5,
gate |x| <= 0.5 (halves gate- x < 0, gate+ x > 0), right x > 0.5; EVAL = the scorer's 150 eval-window bins
[-1.5, 1.5]; seeds s = 1..S are the independent units everywhere; a statistic of pooled deposits is never given a
deposit-level error bar).
  D1  per seed, deaths (births) per opportunity per walker in bin j: d_sj = hist_cum[end, j] / (opp_cum[end] N);
      interval rates between snapshots use the increments of both; region fractions = deaths_R / deaths; seed median
      and IQR.  EXPLORATORY: occupancy hazard = deaths_j fr_every / C_all_j (deaths per walker-opportunity spent in
      the bin; C_all counts walker-steps, an opportunity comes every fr_every steps).
  D2  per seed s, clone-age class c (+ 'all' = every deposit), bin j: n_scj, S_scj = sum f, Q_scj = sum f^2 at a
      snapshot (cumulative).  Reference r_j = <F*'>_j, the scorer's bin reference (eqb_metrics.GatewayScorer.
      bin_reference on the analytic scorer: Gauss-Legendre bin average, the e_Fp_stat convention), verified against
      the cell's frozen reference by eqb_family.GatewayFamilyScorer.
        pooled          n_cj = sum_s n_scj,  m_cj = sum_s S_scj / n_cj,  delta_cj = m_cj - r_j
        cluster var     V_cj = G/(G-1) sum_s (S_scj - m_cj n_scj)^2 / n_cj^2, G = #seeds with n_scj > 0 (>= 2,
                        else the bin is excluded): the seed-cluster (delta-method) variance of the ratio m_cj, which
                        absorbs the within-seed autocorrelation of deposits and the clone duplication
        signed          D_cR = sum_{j in R & EVAL} n_cj delta_cj / sum_{j in R & EVAL} n_cj  (deposit-weighted)
        contribution    K_cR = sum_{j in R & EVAL} n_cj delta_cj / sum_{j in R & EVAL} n_all,j  (sum_c K_cR = D_all,R)
        debiased RMS    B_c^2 = sum_{j in E_c} n_cj (delta_cj^2 - V_cj) / sum_{j in E_c} n_cj,
                        E_c = EVAL bins with G >= 2;  B_c = sqrt(max(B_c^2, 0))  ('bias'); raw = same without -V,
                        noise^2 = sum n V / sum n, coverage = sum_{E_c} n_cj / sum_EVAL n_cj
        CI              seed bootstrap (n_boot resamples of S seeds with replacement, numpy default_rng(BOOT_SEED)):
                        every quantity above recomputed from the resampled seed sums.  B_c^2 is a DEBIASED quadratic:
                        in the bootstrap world its parameter is the raw plug-in raw_c (the resampled seeds' pooled
                        means are the truth there), so the replicates B2* centre on raw_c, about noise^2 above the
                        estimate; the interval is the BASIC (pivotal) one, from the reflected draws
                        B2_c - (B2* - raw_c): [B2_c - q97.5(B2* - raw_c), B2_c - q2.5(B2* - raw_c)], then
                        B_c's interval = sqrt(max(., 0)) of its ends; every function of several debiased quadratics
                        (B differences, the matched excess and its changes) is evaluated on the jointly reflected
                        draws.  Signed deviations, contributions, variance ratios, coverage: plain 2.5 / 97.5
                        percentiles.  raw2 and noise2: descriptive, no interval
        matched excess  ANALYSIS-DEFINED, EXPLORATORY (not in the plan, which names B_c / D_cR against the bin-averaged
                        F*'): the class's error vs ABF's error AT THE CLASS'S POSITIONS (paired seeds): over the bins
                        J_c usable in both arms, with the FR class's counts as weights,
                          B_ABF|c^2 = sum_{J_c} n_cj (delta_ABF,j^2 - V_ABF,j) / sum_{J_c} n_cj,
                          excess_c  = B_c|J_c - B_ABF|c,
                          dev_excess_cR = sum_{j in R} n_cj (delta_cj - delta_ABF,j) / sum_{j in R} n_cj;
                        zero for a class that only deposits where every walker's error is large (young clones in the
                        gate flank), so a positive excess is error carried by the clone lineage itself
        contrasts       paired over the common seeds (one draw resamples the same seeds in every arm and cell): FR
                        class c vs the ABF arm ('all' = 'never'); cell vs the reference cell (alpha1 / lam1) at the same
                        N: B_c differences B_c(cell) - B_c(ref) (the PREREGISTERED Experiment II reading, plan section
                        7: 'recently cloned deposits carrying a mean-force error that grows as lambda decreases'), D_cR
                        differences, the difference in differences of (B_c^FR - B^ABF) and of the matched excess,
                        excess_c(cell) - excess_c(ref) ('change from ref', EXPLORATORY, analysis-defined); two-sided
                        bootstrap p (cross_cell.bootstrap_p convention) on the (reflected) draws, UNADJUSTED per test,
                        plus Holm within each readout family (experiment x N x statistic, over cells x classes) in the
                        synthesis readout; with < 10 seeds the CIs and p are not calibrated: flagged per cell, and
                        NaN in the synthesis readout
      Class-wise variance (EXPLORATORY): within-bin variance pooled over seeds, SSW_cj / n_cj, vs the exact
      e_j = <Var(f|x)>_j + <F*'^2>_j - <F*'>_j^2 (the within-bin variance of f also holds the variation of F*' across
      the bin) on the scorer's sub-grid; region ratio sum SSW / sum n e with a seed-bootstrap CI.
      Class fractions per region and snapshot: per seed n_cR / n_R (cumulative and interval), seed median and IQR.
  D3  deposit counts by time since the walker's last gate crossing, fractions per region and snapshot as D2's
      fractions, FR and ABF; paired FR - ABF difference of the per-seed fractions at the end (median, seed
      bootstrap CI).
  y law (EXPLORATORY)  walkers of X_final / Y_final with |x - x0| <= 0.05, x0 in {-1, -0.21, 0, 0.21, 1}, pooled over
      seeds; z = (y - m(x)) sqrt(beta k(x)) is exactly N(0, 1) under the conditional law (k = omega^(2 alpha) or
      kappa^2); KS D against N(0, 1) with its NOMINAL p (walkers treated as independent: FR copies are not), mean and
      variance of z with seed-bootstrap CIs, and the frozen-x EM expectation <1/(1 - lam k h/2)> of var(z).
  traces (EXPLORATORY) x, y and z_y of the first four traced walkers of the lowest complete seed.
  S1  V(x, y) of the four matched-F* models on one colour scale (beta (V - min V), kT) with the conditional band
      m(x) +- 2 sd(y | x); barrier decomposition: energetic E(x) = min_y V(x, y) (alpha: H(x^2-1)^2 + ((1-alpha)/beta)
      log omega; shift: F*), energetic barrier E(0) - E(-1), entropic = F* barrier - energetic barrier.
  S3 / S5 / S6  G values, CIs and per-seed values are read from the summaries (paired G, 10000-resample seed
      bootstrap; analyze_ladder); nothing is re-scored.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import textwrap
import time
import warnings

for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts", "equal_budget"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import eqb_metrics as M  # noqa: E402
import eqb_family as MF  # noqa: E402
import fig_legibility as LG  # noqa: E402
import gateway_family_validation as GF  # noqa: E402  (numpy model functions, frozen-x OU predictions)

SCHEMA_CELL = "mech_diagnostics/1"
SCHEMA_SYN = "mech_synthesis/1"
NB = M.NB
XMIN, XMAX = -1.8, 1.8
DELTA = (XMAX - XMIN) / NB
CENTRES = XMIN + DELTA * (np.arange(NB) + 0.5)
EVAL_LO, EVAL_HI = -1.5, 1.5
EVAL = (CENTRES > EVAL_LO) & (CENTRES < EVAL_HI)
X_BASIN = 0.5
REGIONS = dict(left=CENTRES < -X_BASIN, gate=np.abs(CENTRES) <= X_BASIN, right=CENTRES > X_BASIN)
DEV_REGIONS = dict(left=REGIONS["left"], gate=REGIONS["gate"], gate_neg=REGIONS["gate"] & (CENTRES < 0),
                   gate_pos=REGIONS["gate"] & (CENTRES > 0), right=REGIONS["right"])
REGION_LABEL = dict(left="left well x < -0.5", gate="gate |x| <= 0.5", gate_neg="gate, x < 0", gate_pos="gate, x > 0",
                    right="right well x > 0.5")
COARSE = 5                                   # production bins per display bin (0.1 wide)
NC = NB // COARSE
CENTRES_C = XMIN + COARSE * DELTA * (np.arange(NC) + 0.5)
N_AGE = 5
ALL = 5                                      # D2 pseudo-class index: every deposit
AGE_DEFAULT = ["[0,0.01)", "[0.01,0.1)", "[0.1,1)", ">=1", "never"]
CROSS_DEFAULT = ["[0,0.1)", "[0.1,1)", "[1,10)", ">=10", "never"]
Y_WINDOWS = (-1.0, -0.21, 0.0, 0.21, 1.0)
Y_HW = 0.05
Z_EDGES = np.linspace(-5.0, 5.0, 41)
ACF_X = (-1.0, -0.21, 0.0)
N_BOOT = M.N_BOOT
BOOT_SEED = M.BOOT_SEED
BOOT_CHUNK = 1000
SMALL_N = 10                                 # below this many seeds a bootstrap p is not calibrated (cross_cell)
EXP_ORDER = ("matched_free_energy", "conditional_relaxation")
CELL_ORDER = dict(matched_free_energy=("alpha1", "alpha0.5", "alpha0", "shift"),
                  conditional_relaxation=("lam1", "lam0.5", "lam0.25", "lam0.1"))
REF_CELL = dict(matched_free_energy="alpha1", conditional_relaxation="lam1")
EXP_LABEL = dict(matched_free_energy="Experiment I (matched free energy)",
                 conditional_relaxation="Experiment II (transverse mobility lambda)")
RUN_KEYS = ("save_step", "save_t", "save_u", "C_all", "fr_opp_cum", "fr_deaths_cum", "fr_death_pos_hist_cum",
            "fr_birth_pos_hist_cum", "snap_step", "snap_t", "snap_u", "dep_age_C", "dep_age_M", "dep_age_M2",
            "dep_cross_C", "traces_t", "traces", "traces_y", "traces_rebirth", "X_final", "Y_final")
D_KEYS = ("fr_death_pos_hist_cum", "fr_birth_pos_hist_cum", "snap_step", "dep_age_C", "dep_age_M", "dep_age_M2",
          "dep_cross_C", "traces_y")
PROTECTED = MF.PROTECTED_OUTPUT_DIRS

# ---- palette (equal-budget method slots; classes ordered young -> old; neutral chrome) ----
C_ABF, C_FR = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CLASS_COLOR = ["#7f0000", "#c51b7d", "#6a51a3", "#238b45", "#737373"]
N_COLOR = {2048: "#08306b", 512: "#4292c6", 128: "#9e9ac8"}
N_MARKER = {2048: "o", 512: "s", 128: "^"}
CELL_COLOR = {"alpha1": "#d95f02", "alpha0.5": "#386cb0", "alpha0": "#1b9e77", "shift": "#e7298a",
              "lam1": "#d95f02", "lam0.5": "#e6ab02", "lam0.25": "#b15928", "lam0.1": "#6a3d9a"}
SNAP_CMAP = plt.get_cmap("viridis")
BANNER_EXPLORATORY = "EXPLORATORY: this quantity is not preregistered (SCIENTIFIC_PLAN sections 4 and 6)"
BANNER_DESCRIPTIVE = "Descriptive, not causal (SCIENTIFIC_PLAN section 6: mechanistic covariates)"


class MechError(RuntimeError):
    pass


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "mathtext.default": "regular",
    })


# =================================================================================================== small helpers
def fnum(v):
    if v is None:
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def miq(Y, axis=0):
    """(median, q25, q75) over axis, NaN-aware (all-NaN columns stay NaN)."""
    Y = np.asarray(Y, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanmedian(Y, axis=axis), np.nanpercentile(Y, 25, axis=axis), np.nanpercentile(Y, 75, axis=axis))


def stats_of(Y, axis=0):
    med, lo, hi = miq(Y, axis)
    return dict(median=med, q25=lo, q75=hi)


def coarse(a):
    """Sum of COARSE consecutive production bins along the last axis."""
    a = np.asarray(a, dtype=float)
    return a.reshape(a.shape[:-1] + (NC, COARSE)).sum(-1)


def pct(b, q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(b, q, axis=0)


def boot_p(boot):
    """Two-sided percentile-bootstrap p of H0: value = 0 (cross_cell.bootstrap_p, NaN resamples dropped)."""
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return math.nan
    lo = (np.sum(b <= 0) + 1.0) / (b.size + 1.0)
    hi = (np.sum(b >= 0) + 1.0) / (b.size + 1.0)
    return float(min(1.0, 2.0 * min(lo, hi)))


def boot_counts(n_units, n_boot, seed=BOOT_SEED):
    """(n_boot, n_units) multiplicities of a with-replacement resample of n_units units (cross_cell's draw:
    default_rng(seed).integers(0, n, size=(n_boot, n)))."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_units, size=(int(n_boot), int(n_units)))
    K = np.zeros((int(n_boot), int(n_units)))
    np.add.at(K, (np.repeat(np.arange(int(n_boot)), int(n_units)), idx.ravel()), 1.0)
    return K


def wrap(s, width):
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False)) if s else ""


def cell_model(P):
    """GF model dict of a cell (physical constants from its engine_cfg)."""
    ec = P["engine_cfg"]
    m = MF.family_model(ec)
    phys = dict(beta=float(ec["beta"]), H=float(ec["H"]), omega_out=float(ec["omega_out"]),
                omega_in=float(ec["omega_in"]), s=float(ec["s"]))
    return GF.make_model(m["variant"], m["alpha"] if m["alpha"] is not None else 1.0,
                         m["kappa"] if m["kappa"] is not None else 1.0, m["lam"], **phys)


def dynamics_name(model):
    """Validation dynamics name (gateway_family_validation.DYNAMICS_ORDER) of a cell model."""
    if model["variant"] == "shift":
        return "shift"
    if model["lam"] == 1.0:
        return f"alpha{model['alpha']:g}"
    return f"lam{model['lam']:g}" if model["alpha"] == 1.0 else None


def is_production(P):
    h = float(P["engine_cfg"]["h"])
    return int(P["B"]) == int(round(2048 * 40 / h)) and len(P["seeds"]) == 32


def em_inflation(x, m, h):
    return 1.0 / (1.0 - m["lam"] * GF.stiffness(np.asarray(x, float), m) * h / 2.0)


# =================================================================================================== model facts
def barrier_decomposition(m, n=7201):
    """Exact barrier of F* and its energetic (min_y V along x) and entropic (F* - energetic) parts, in kT."""
    x = np.linspace(XMIN, XMAX, n)
    F = GF.F_exact(x, m)
    if m["variant"] == "shift":
        E = F.copy()
    else:
        a = m["alpha"]
        E = m["H"] * (x * x - 1.0) ** 2 + ((1.0 - a) / m["beta"]) * np.log(GF.omega(x, m))
    i0, iw = int(np.argmin(np.abs(x))), int(np.argmin(np.abs(x + 1.0)))
    b = m["beta"]
    Fb, Eb = F[i0] - F[iw], E[i0] - E[iw]
    return dict(x=x, F_kT=b * (F - F[iw]), E_kT=b * (E - E[iw]), S_kT=b * ((F - F[iw]) - (E - E[iw])),
                barrier_kT=b * Fb, energetic_kT=b * Eb, entropic_kT=b * (Fb - Eb), entropic_fraction=(Fb - Eb) / Fb)


def exact_bin_force_moments(m, sub_x, sub_w, s2h):
    """Per production bin on the scorer's sub-grid: <F*'>_j, <Var(f|x)>_j and the exact within-bin variance of f,
    e_j = <Var(f|x)>_j + <F*'^2>_j - <F*'>_j^2."""
    sub_x = np.asarray(sub_x, float)
    wsum = np.bincount(s2h, weights=sub_w, minlength=NB)
    fp = GF.Fp_exact(sub_x, m)
    vf = GF.cond_var_f(sub_x, m)
    fbar = np.bincount(s2h, weights=sub_w * fp, minlength=NB) / wsum
    f2 = np.bincount(s2h, weights=sub_w * fp * fp, minlength=NB) / wsum
    vbar = np.bincount(s2h, weights=sub_w * vf, minlength=NB) / wsum
    return dict(fbar=fbar, varf=vbar, etot=vbar + np.maximum(f2 - fbar * fbar, 0.0))


# =================================================================================================== discovery / IO
def discover(cells_root, experiments=None, only=None):
    """[(experiment, cell, P, config path)] in the fixed order, plus the expected cells that have no config."""
    found, expected_missing = {}, []
    for p in sorted(glob.glob(os.path.join(cells_root, "*", "*.json"))):
        try:
            P = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            raise MechError(f"unreadable cell config {p}: {e}")
        if not MF.is_cell(P):
            continue
        found[(P["experiment"], P["cell"])] = (P, p)
    out = []
    for exp in EXP_ORDER:
        if experiments and exp not in experiments:
            continue
        for c in CELL_ORDER[exp]:
            if only and c not in only:
                continue
            if (exp, c) in found:
                out.append((exp, c) + found[(exp, c)])
            else:
                expected_missing.append(f"{exp}/{c}")
    extra = sorted(k for k in found if k[1] not in CELL_ORDER.get(k[0], ()))
    return out, expected_missing, [f"{a}/{b}" for a, b in extra]


def runs_root(P, override):
    if override:
        return os.path.join(os.path.abspath(override), P["experiment"], P["out_dir"])
    return MF._abs(P["results_dir"])


def analysis_dir(P, override):
    if override:
        base = os.path.join(os.path.abspath(override), P["experiment"], P["cell"])
        return os.path.join(base, "analysis") if os.path.isdir(os.path.join(base, "analysis")) else base
    return MF.cell_analysis_dir(P)


def file_status(path):
    """analyze_ladder.file_status: complete / invalid / running (checkpoint only) / missing."""
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


def load_run(path, P, N, seed, method):
    """The arrays this analysis reads (RUN_KEYS present in the file) + status; plan-checked against the cell."""
    st, why = file_status(path)
    if st != "complete":
        return dict(status=st, why=why)
    try:
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            out = {k: np.asarray(z[k]) for k in RUN_KEYS if k in files}
            meta = json.loads(str(z["meta_json"]))
            cfg = json.loads(str(z["cfg_json"]))
    except Exception as e:  # noqa: BLE001
        return dict(status="invalid", why=f"unreadable: {type(e).__name__}: {e}")
    try:
        MF.check_plan(dict(meta=meta, cfg=cfg), P, path, expect=dict(N=int(N), seed=int(seed), method=method))
    except M.MetricsError as e:
        return dict(status="invalid", why=str(e))
    out.update(status="complete", meta=meta, cfg=cfg, path=path)
    miss = [k for k in D_KEYS if k not in out]
    out["has_diag"] = not miss
    out["diag_missing"] = miss
    out["diag_problems"] = diag_consistency(out) if not miss else []
    if not miss and not out["diag_problems"]:
        thin_save_axis(out)
    return out


SAVE_AXIS_KEYS = ("save_step", "save_t", "save_u", "C_all", "fr_opp_cum", "fr_deaths_cum", "fr_death_pos_hist_cum",
                  "fr_birth_pos_hist_cum")


def thin_save_axis(r):
    """Keep only the save rows this analysis reads (the snapshot saves and the last save) of the save-axis arrays:
    memory for 32 seeds x 2 arms at production save grids.  Done after the consistency checks, which read every row."""
    n = np.asarray(r["save_step"]).size
    rows = np.unique(np.concatenate([snap_save_index(r), [n - 1]]))
    for k in SAVE_AXIS_KEYS:
        if k in r and np.ndim(r[k]) >= 1 and np.shape(r[k])[0] == n:
            r[k] = np.asarray(r[k])[rows]
    r["n_saves_original"] = int(n)


def snap_save_index(r):
    """Save index of every snapshot step (snapshots are a subset of the save grid); raises if one is missing."""
    ss, sv = np.asarray(r["snap_step"]), np.asarray(r["save_step"])
    idx = np.searchsorted(sv, ss)
    if np.any(idx >= sv.size) or np.any(sv[np.minimum(idx, sv.size - 1)] != ss):
        raise MechError("a snapshot step is not on the save grid")
    return idx


def diag_consistency(r):
    """The engine's exact D2 / D3 / D1 identities (module docstring of gateway_family_numba)."""
    probs = []
    try:
        idx = snap_save_index(r)
    except MechError as e:
        return [str(e)]
    C = np.asarray(r["C_all"])[idx]
    if not np.array_equal(np.asarray(r["dep_age_C"]).sum(1), C):
        probs.append("dep_age_C summed over classes != C_all at the snapshots")
    if not np.array_equal(np.asarray(r["dep_cross_C"]).sum(1), C):
        probs.append("dep_cross_C summed over classes != C_all at the snapshots")
    if "fr_deaths_cum" in r:
        d = np.asarray(r["fr_death_pos_hist_cum"]).sum(1)
        b = np.asarray(r["fr_birth_pos_hist_cum"]).sum(1)
        if not (np.array_equal(d, np.asarray(r["fr_deaths_cum"])) and np.array_equal(b, np.asarray(r["fr_deaths_cum"]))):
            probs.append("D1 histogram totals != fr_deaths_cum")
    return probs


def collect_N(P, res_root, N):
    """{method: {seed: run dict or status dict}} for the planned seeds and methods of a cell at one N (one N at a
    time keeps the memory to 2 x 32 runs)."""
    out = {}
    for method in ("abf", "fr"):
        out[method] = {}
        for seed in [int(s) for s in P["seeds"]]:
            path = os.path.join(res_root, f"N{int(N)}", f"s{seed}_{method}.npz")
            out[method][seed] = load_run(path, P, int(N), seed, method)
    return out


def status_counts(runs_by_seed):
    c = dict(planned=len(runs_by_seed), complete=0, with_diag=0, missing=[], running=[], invalid={}, diag_missing={},
             diag_problems={})
    for s, r in runs_by_seed.items():
        st = r["status"]
        if st == "complete":
            c["complete"] += 1
            if r["has_diag"] and not r["diag_problems"]:
                c["with_diag"] += 1
            elif not r["has_diag"]:
                c["diag_missing"][s] = r["diag_missing"]
            else:
                c["diag_problems"][s] = r["diag_problems"]
        elif st == "missing":
            c["missing"].append(s)
        elif st == "running":
            c["running"].append(s)
        else:
            c["invalid"][s] = r.get("why")
    return c


def diag_runs(runs_by_seed):
    """{seed: run} of the complete runs that carry consistent D1-D4 diagnostics."""
    return {s: r for s, r in sorted(runs_by_seed.items())
            if r["status"] == "complete" and r["has_diag"] and not r["diag_problems"]}


def complete_runs(runs_by_seed):
    return {s: r for s, r in sorted(runs_by_seed.items()) if r["status"] == "complete"}


def load_summary(adir):
    p = os.path.join(adir, "summary.json")
    if not os.path.exists(p):
        return None, f"no summary at {p}"
    try:
        return json.load(open(p)), None
    except Exception as e:  # noqa: BLE001
        return None, f"unreadable summary {p}: {e}"


def load_run_curves(adir, N, seeds, method):
    """{seed: dict(save_u, save_t, e_F, e_Fp_stat, TV_half)} from the per-run metric cache of the cell analysis."""
    out, miss = {}, []
    for s in seeds:
        p = os.path.join(adir, "run_metrics", f"N{N}", f"s{s}_{method}.npz")
        if not os.path.exists(p):
            miss.append(s)
            continue
        with np.load(p, allow_pickle=False) as z:
            out[s] = {k: np.asarray(z[k], float) for k in ("save_u", "save_t", "e_F", "e_Fp_stat", "TV_half")
                      if k in z.files}
    return out, miss


# =================================================================================================== D1
def d1_compute(fr_runs, N):
    """D1 statistics from {seed: FR run}: per-seed rates, then seed median / IQR."""
    seeds = sorted(fr_runs)
    if not seeds:
        return None
    dr, br, hz, dI, bI, reg_d, reg_b, totals, snaps_u = [], [], [], [], [], [], [], [], None
    for s in seeds:
        r = fr_runs[s]
        opp = np.asarray(r["fr_opp_cum"], float)
        dh = np.asarray(r["fr_death_pos_hist_cum"], float)
        bh = np.asarray(r["fr_birth_pos_hist_cum"], float)
        o = opp[-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            dr.append(dh[-1] / (o * N) if o > 0 else np.full(NB, np.nan))
            br.append(bh[-1] / (o * N) if o > 0 else np.full(NB, np.nan))
            fe = float(r["meta"].get("fr_every", 1))
            Cc = coarse(np.asarray(r["C_all"], float)[-1])
            hz.append(np.where(Cc > 0, coarse(dh[-1]) * fe / Cc, np.nan))
        idx = snap_save_index(r)
        snaps_u = np.asarray(r["snap_u"], float) if snaps_u is None else snaps_u
        rows_d, rows_b = [], []
        prev_d, prev_b, prev_o = np.zeros(NB), np.zeros(NB), 0.0
        for k in idx:
            do = opp[k] - prev_o
            with np.errstate(invalid="ignore", divide="ignore"):
                rows_d.append(coarse(dh[k] - prev_d) / (do * N) if do > 0 else np.full(NC, np.nan))
                rows_b.append(coarse(bh[k] - prev_b) / (do * N) if do > 0 else np.full(NC, np.nan))
            prev_d, prev_b, prev_o = dh[k], bh[k], opp[k]
        dI.append(rows_d)
        bI.append(rows_b)
        td, tb = dh[-1].sum(), bh[-1].sum()
        reg_d.append([dh[-1][REGIONS[R]].sum() / td if td > 0 else np.nan for R in REGIONS])
        reg_b.append([bh[-1][REGIONS[R]].sum() / tb if tb > 0 else np.nan for R in REGIONS])
        totals.append([td / (o * N) if o > 0 else np.nan, o, td])
    dr, br, hz = np.array(dr), np.array(br), np.array(hz)
    dI, bI = np.array(dI), np.array(bI)                     # (S, n_snap, NC)
    reg_d, reg_b, totals = np.array(reg_d), np.array(reg_b), np.array(totals)
    return dict(
        seeds=seeds, n=len(seeds), snap_u=snaps_u,
        end=dict(death_rate=stats_of(dr), birth_rate=stats_of(br),
                 death_rate_coarse=stats_of(coarse(dr)), birth_rate_coarse=stats_of(coarse(br)),
                 net_rate_coarse=stats_of(coarse(br - dr)), hazard_coarse_EXPLORATORY=stats_of(hz)),
        interval=dict(death_rate_coarse=stats_of(dI), birth_rate_coarse=stats_of(bI)),
        region_fraction=dict(deaths={R: stats_of(reg_d[:, i]) for i, R in enumerate(REGIONS)},
                             births={R: stats_of(reg_b[:, i]) for i, R in enumerate(REGIONS)}),
        totals=dict(deaths_per_opp_per_walker=stats_of(totals[:, 0]), opportunities=stats_of(totals[:, 1]),
                    deaths=stats_of(totals[:, 2])))


# =================================================================================================== D2 core
def d2_arrays(runs, si):
    """(n, s1, s2) of shape (S, 6, NB) at snapshot index si: the five clone-age classes + 'all'."""
    out = []
    for key in ("dep_age_C", "dep_age_M", "dep_age_M2"):
        a = np.stack([np.asarray(r[key][si], float) for r in runs])
        out.append(np.concatenate([a, a.sum(1, keepdims=True)], axis=1))
    return out


class D2Calc:
    """D2 statistics of one arm (module docstring) for arbitrary seed weights K (n_boot, S): K = 1 is the point
    estimate, a row of bootstrap multiplicities a seed-bootstrap replicate."""

    def __init__(self, n, s1, s2, ref, etot, evalmask=EVAL):
        n, s1, s2 = (np.asarray(a, float) for a in (n, s1, s2))
        S, C, J = n.shape
        self.S, self.C, self.J = S, C, J
        self.ref = np.asarray(ref, float)
        self.etot = np.asarray(etot, float)
        self.ev = np.asarray(evalmask, bool)
        Np = n.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m0 = np.where(Np > 0, s1.sum(0) / np.where(Np > 0, Np, 1.0), 0.0)
        A = s1 - m0[None] * n                            # centred sums (no cancellation in the bootstrap)
        Qc = s2 - 2.0 * m0[None] * s1 + m0[None] ** 2 * n
        f = lambda a: a.reshape(S, C * J)                # noqa: E731
        self.m0 = m0
        self.n2, self.A2, self.AA, self.An, self.nn = f(n), f(A), f(A * A), f(A * n), f(n * n)
        self.Qc, self.pos = f(Qc), f((n > 0).astype(float))
        self.Rm = np.stack([(mk & self.ev) for mk in DEV_REGIONS.values()], -1).astype(float)     # (J, nR)
        self.Cm = (np.arange(J)[:, None] // COARSE == np.arange(J // COARSE)[None, :]).astype(float)  # (J, NC)

    def stats(self, K, full=True, profiles=False, perbin=False):
        """Every D2 quantity for the seed weights K (n_boot, S).  full=False: only B (B2) and the signed region
        deviations (paired contrasts); profiles=True adds the 0.1-wide display profiles; perbin=True adds the per-bin
        n (Nk), delta, V and the usable-bin mask (matched-position contrasts)."""
        K = np.atleast_2d(np.asarray(K, float))
        B = K.shape[0]
        shp = (B, self.C, self.J)
        Nk = (K @ self.n2).reshape(shp)
        Ak = (K @ self.A2).reshape(shp)
        has = Nk > 0
        Nsafe = np.where(has, Nk, 1.0)
        out = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(has, Ak / Nsafe, 0.0)            # m - m0 (0 where empty)
            R2 = (K @ self.AA).reshape(shp) - 2.0 * d * (K @ self.An).reshape(shp) + d * d * (K @ self.nn).reshape(shp)
            G = (K @ self.pos).reshape(shp)
            Gd = ((K > 0).astype(float) @ self.pos).reshape(shp)
            use = self.ev[None, None, :] & has & (Gd >= 2)
            V = np.where(use, G / np.maximum(G - 1.0, 1.0) * np.maximum(R2, 0.0) / Nsafe ** 2, 0.0)
            delta = np.where(has, self.m0[None] + d - self.ref[None, None, :], 0.0)
            ndel = Nk * delta                              # 0 where empty
            den = Nk @ self.Rm                             # (B, C, nR)
            num = ndel @ self.Rm
            out["dev"] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
            w = np.where(use, Nk, 0.0)
            ws = w.sum(-1)
            okw = ws > 0
            wss = np.where(okw, ws, 1.0)
            wd2 = (w * delta * delta).sum(-1)
            wv = (w * V).sum(-1)
            out["B2"] = np.where(okw, (wd2 - wv) / wss, np.nan)
            out["B"] = np.where(okw, np.sqrt(np.maximum(out["B2"], 0.0)), np.nan)
            out["raw2"] = np.where(okw, wd2 / wss, np.nan)      # the plug-in (bootstrap-world parameter of B2)
            if perbin:
                out.update(Nk=Nk, delta=delta, V=V, use=use)
            if full:
                out["noise2"] = np.where(okw, wv / wss, np.nan)
                ev_n = (Nk * self.ev[None, None, :]).sum(-1)
                out["coverage"] = np.where(ev_n > 0, ws / np.where(ev_n > 0, ev_n, 1.0), np.nan)
                out["n_bins"] = use.sum(-1).astype(float)
                da = (Nk[:, ALL, :] @ self.Rm)[:, None, :]
                out["contrib"] = np.where(da > 0, num / np.where(da > 0, da, 1.0), np.nan)
                SSW = np.maximum((K @ self.Qc).reshape(shp) - Nk * d * d, 0.0)
                de = (Nk * self.etot[None, None, :]) @ self.Rm
                out["var_ratio"] = np.where((den > 0) & (de > 0), (SSW @ self.Rm) / np.where(de > 0, de, 1.0), np.nan)
                if profiles:
                    Nc = Nk @ self.Cm
                    ok = Nc > 0
                    Ncs = np.where(ok, Nc, 1.0)
                    out["dev_coarse"] = np.where(ok, (ndel @ self.Cm) / Ncs, np.nan)
                    out["var_coarse"] = np.where(ok, (SSW @ self.Cm) / Ncs, np.nan)
                    out["etot_coarse"] = np.where(ok, ((Nk * self.etot[None, None, :]) @ self.Cm) / Ncs, np.nan)
        return out


SCALAR_KEYS = ("B", "B2", "raw2", "noise2", "coverage", "n_bins")
REGION_KEYS = ("dev", "contrib", "var_ratio")
PCT_SCALAR_KEYS = ("coverage", "n_bins")                 # descriptive scalars: plain percentile interval
NO_CI_KEYS = ("raw2", "noise2")                          # descriptive, no interval (see D2_CI in DEFINITIONS)


def root(q):
    """B = sqrt(max(B^2, 0)) of a debiased squared bias (NaN stays NaN)."""
    q = np.asarray(q, float)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(q), np.sqrt(np.maximum(q, 0.0)), np.nan)


def reflect(q_pt, raw_pt, q_boot):
    """Basic (pivotal) seed-bootstrap draws of a DEBIASED quadratic estimator Q = sum w (delta^2 - V) / sum w.
    In the bootstrap world the resampled seeds' pooled bin means are the truth, so the parameter there is the RAW
    plug-in raw = sum w delta_hat^2 / sum w (it still holds the sampling noise of delta_hat), and the replicates Q* are
    centred on raw, i.e. about noise^2 ABOVE the estimate Q_hat: their plain percentiles would give an interval that
    sits above the debiased estimate and rarely covers the truth.  The distribution of Q* - raw approximates that of
    Q_hat - q, so the reflected draws Q_hat - (Q* - raw) are confidence draws of q: their 2.5 / 97.5 percentiles are
    the basic interval [Q_hat - q97.5(Q* - raw), Q_hat - q2.5(Q* - raw)], and a function of several q (B = root(q),
    differences of B, the matched excess) is evaluated on the jointly reflected draws (one resample reflects every
    quadratic of every arm / cell with the same seeds)."""
    return q_pt + raw_pt - q_boot


def d2_summarise(calc, n_boot, seed=BOOT_SEED, profiles=False):
    """Point estimate + seed-bootstrap 95 % CI of every D2 quantity of one arm: B2 / B by the basic (reflected)
    bootstrap of the debiased quadratic (reflect), the linear / descriptive quantities by plain percentiles; raw2 and
    noise2 carry no interval."""
    pt = calc.stats(np.ones((1, calc.S)), profiles=profiles)
    keys = SCALAR_KEYS + REGION_KEYS + (("dev_coarse", "var_coarse") if profiles else ())
    pkeys = PCT_SCALAR_KEYS + REGION_KEYS + (("dev_coarse", "var_coarse") if profiles else ())
    acc = {k: [] for k in pkeys + ("B2",)}
    if n_boot > 0 and calc.S >= 2:
        K = boot_counts(calc.S, n_boot, seed)
        for i in range(0, K.shape[0], BOOT_CHUNK):
            st = calc.stats(K[i:i + BOOT_CHUNK], profiles=profiles)
            acc["B2"].append(reflect(pt["B2"], pt["raw2"], st["B2"]))
            for k in pkeys:
                acc[k].append(st[k])
    out = {}
    for k in keys:
        out[k] = pt[k][0]
        if k in acc and acc[k]:
            b = np.concatenate(acc[k], 0)
            out[k + "_lo"], out[k + "_hi"] = pct(b, 2.5), pct(b, 97.5)
        else:
            out[k + "_lo"] = out[k + "_hi"] = np.full_like(pt[k][0], np.nan)
    out["B_lo"], out["B_hi"] = root(out["B2_lo"]), root(out["B2_hi"])
    if profiles:
        out["etot_coarse"] = pt["etot_coarse"][0]
    return out


def _matched(sa, ca, sb, cb, Rm, ev):
    """Matched-position comparison of class ca of arm/cell a with class cb of b, both weighted by a's class-ca
    deposit counts (stats dicts with perbin=True; ca, cb: index arrays of equal length):
        A2 = B_{a|a}^2 = sum_{j in J} n_a,j (delta_a,j^2 - V_a,j) / sum_{j in J} n_a,j,   J = bins usable in both,
        B2 = B_{b|a}^2 = sum_{j in J} n_a,j (delta_b,j^2 - V_b,j) / sum_{j in J} n_a,j   (b's error at a's positions),
        A2raw, B2raw: the same without -V (the bootstrap-world parameters, see reflect),
        B_excess = root(A2) - root(B2),
        dev_excess_R = [sum_{j in R} n_a,j (delta_a,j - delta_b,j)] / sum_{j in R} n_a,j over bins with data in both,
    so a class that merely deposits where every walker's error is large (young clones in the gate flank) has zero
    excess.  ANALYSIS-DEFINED (not in SCIENTIFIC_PLAN): reported as EXPLORATORY."""
    na, da, Va, ua = (sa[k][:, ca, :] for k in ("Nk", "delta", "V", "use"))
    nb_, db, Vb, ub = (sb[k][:, cb, :] for k in ("Nk", "delta", "V", "use"))
    with np.errstate(invalid="ignore", divide="ignore"):
        J = ua & ub
        w = np.where(J, na, 0.0)
        ws = w.sum(-1)
        ok = ws > 0
        wss = np.where(ok, ws, 1.0)
        A2raw = np.where(ok, (w * da * da).sum(-1) / wss, np.nan)
        B2raw = np.where(ok, (w * db * db).sum(-1) / wss, np.nan)
        A2 = np.where(ok, A2raw - (w * Va).sum(-1) / wss, np.nan)
        B2 = np.where(ok, B2raw - (w * Vb).sum(-1) / wss, np.nan)
        Ba, Bb = root(A2), root(B2)
        hb = (na > 0) & (nb_ > 0) & ev[None, None, :]
        wr = np.where(hb, na, 0.0)
        den = wr @ Rm
        dex = np.where(den > 0, ((wr * (da - db)) @ Rm) / np.where(den > 0, den, 1.0), np.nan)
        tot = (na * ev[None, None, :]).sum(-1)
        cov = np.where(tot > 0, ws / np.where(tot > 0, tot, 1.0), np.nan)
    return dict(A2=A2, B2=B2, A2raw=A2raw, B2raw=B2raw, Bm_a=Ba, Bm_b=Bb, B_excess=Ba - Bb, dev_excess=dex,
                matched_coverage=cov)


def d2_paired(calc_a, ca, calc_b, cb, n_boot, seed=BOOT_SEED, calc_c=None, cc=None, calc_d=None, cd=None,
              matched=False):
    """Paired seed-bootstrap differences a[ca_i] - b[cb_i] of B and of the signed region deviations for every class
    pair i (ca, cb: int or equal-length sequences); with c, d also the difference in differences
    (a[ca_i] - c[cc]) - (b[cb_i] - d[cd]) of B.  matched=True adds the matched-position excess (_matched) of a[ca_i]
    over b[cb_i] (without c, d) or, with c, d, the matched excesses E_a = a[ca_i] over c[cc] and E_b = b[cb_i] over
    d[cd] and their difference E_a - E_b ('B_excess_did').  All calcs must hold the SAME seeds in the same order (the
    caller aligns them); one resample draws the same seeds in every calc.  Quantities built from the debiased squared
    biases (B differences, B_did, Bm_*, B_excess*) use the jointly reflected (basic-bootstrap) draws of every
    quadratic (reflect); the signed deviations, dev_excess and the coverage use plain percentiles.  The p values are
    two-sided bootstrap p of H0: value = 0 on those draws, UNADJUSTED for multiplicity.  Returns one dict per pair (a
    dict if ca was an int)."""
    scalar = np.ndim(ca) == 0
    ca, cb = np.atleast_1d(ca).astype(int), np.atleast_1d(cb).astype(int)
    if ca.size != cb.size:
        raise MechError("d2_paired: class lists of different length")
    S = calc_a.S
    for c in (calc_b, calc_c, calc_d):
        if c is not None and c.S != S:
            raise MechError("paired D2 contrast with unaligned seeds")
    did = calc_c is not None
    if did and calc_d is None:
        raise MechError("d2_paired: calc_c without calc_d")
    Rm, ev = calc_a.Rm, calc_a.ev
    ccv = np.full(ca.size, int(cc)) if did else None
    cdv = np.full(ca.size, int(cd)) if did else None

    def prim(K):
        """(debiased quadratics q, their raw plug-ins r, linear / descriptive quantities lin) for seed weights K."""
        a = calc_a.stats(K, full=False, perbin=matched)
        b = calc_b.stats(K, full=False, perbin=matched)
        q = dict(a=a["B2"][:, ca], b=b["B2"][:, cb])
        r = dict(a=a["raw2"][:, ca], b=b["raw2"][:, cb])
        lin = dict(dev=a["dev"][:, ca, :] - b["dev"][:, cb, :])
        if did:
            c = calc_c.stats(K, full=False, perbin=matched)
            d = calc_d.stats(K, full=False, perbin=matched)
            q.update(c=c["B2"][:, [cc]], d=d["B2"][:, [cd]])
            r.update(c=c["raw2"][:, [cc]], d=d["raw2"][:, [cd]])
            if matched:
                for tag, e in (("ea", _matched(a, ca, c, ccv, Rm, ev)), ("eb", _matched(b, cb, d, cdv, Rm, ev))):
                    q[tag + "A"], q[tag + "B"] = e["A2"], e["B2"]
                    r[tag + "A"], r[tag + "B"] = e["A2raw"], e["B2raw"]
        elif matched:
            e = _matched(a, ca, b, cb, Rm, ev)
            q["mA"], q["mB"], r["mA"], r["mB"] = e["A2"], e["B2"], e["A2raw"], e["B2raw"]
            lin.update(dev_excess=e["dev_excess"], matched_coverage=e["matched_coverage"])
        return q, r, lin

    def derive(q, lin):
        Ba, Bb = root(q["a"]), root(q["b"])
        o = dict(B=Ba - Bb, dev=lin["dev"])
        if did:
            o["B_did"] = (Ba - root(q["c"])) - (Bb - root(q["d"]))
            if matched:
                xa = root(q["eaA"]) - root(q["eaB"])
                xb = root(q["ebA"]) - root(q["ebB"])
                o.update(B_excess_a=xa, B_excess_b=xb, B_excess_did=xa - xb)
        elif matched:
            Ma, Mb = root(q["mA"]), root(q["mB"])
            o.update(Bm_a=Ma, Bm_b=Mb, B_excess=Ma - Mb, dev_excess=lin["dev_excess"],
                     matched_coverage=lin["matched_coverage"])
        return o

    q0, r0, l0 = prim(np.ones((1, S)))
    pt = derive(q0, l0)
    acc = {k: [] for k in pt}
    if n_boot > 0 and S >= 2:
        K = boot_counts(S, n_boot, seed)
        for i in range(0, K.shape[0], BOOT_CHUNK):
            qb, _, lb = prim(K[i:i + BOOT_CHUNK])
            st = derive({k: reflect(q0[k], r0[k], qb[k]) for k in qb}, lb)
            for k in st:
                acc[k].append(st[k])
    boot = {k: (np.concatenate(v, 0) if v else None) for k, v in acc.items()}
    res = []
    for i in range(ca.size):
        out = dict(n_seeds=S, p_calibrated=bool(S >= SMALL_N), n_boot=int(n_boot))
        for k in pt:
            out[k] = pt[k][0, i]
            b = boot[k]
            if b is None:
                out[k + "_ci95"] = [math.nan, math.nan]
                out[k + "_p"] = math.nan
                continue
            bi = b[:, i]
            out[k + "_ci95"] = [pct(bi, 2.5), pct(bi, 97.5)]
            out[k + "_p"] = boot_p(bi) if bi.ndim == 1 else [boot_p(bi[:, r]) for r in range(bi.shape[1])]
            fin = np.isfinite(bi) if bi.ndim == 1 else np.all(np.isfinite(bi), axis=1)
            out[k + "_boot_finite_frac"] = float(np.mean(fin))
        res.append(out)
    return res[0] if scalar else res


def class_fractions(runs, key, regions=REGIONS):
    """Per seed, per snapshot, per class, per region: cumulative and interval fractions of the deposits."""
    cum, inter = [], []
    for r in runs:
        a = np.asarray(r[key], float)                    # (n_snap, 5, NB)
        R = np.stack([a[:, :, m].sum(-1) for m in regions.values()], -1)        # (n_snap, 5, nR)
        tot = R.sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cum.append(np.where(tot > 0, R / np.where(tot > 0, tot, 1.0), np.nan))
            dR = np.diff(np.concatenate([np.zeros_like(R[:1]), R], 0), axis=0)
            dt = dR.sum(1, keepdims=True)
            inter.append(np.where(dt > 0, dR / np.where(dt > 0, dt, 1.0), np.nan))
    return np.array(cum), np.array(inter)                # (S, n_snap, 5, nR)


# =================================================================================================== conditional y law
def ylaw_compute(runs_by_method, model, h, n_boot, seed=BOOT_SEED):
    """EXPLORATORY: standardised conditional residual z of the final walkers in the representative windows."""
    from scipy.stats import kstest
    out = {}
    for meth, runs in runs_by_method.items():
        seeds = sorted(runs)
        res = {}
        for c in Y_WINDOWS:
            zs, sums, infl = [], [], []
            for s in seeds:
                X = np.asarray(runs[s]["X_final"], float)
                Y = np.asarray(runs[s]["Y_final"], float)
                sel = np.abs(X - c) <= Y_HW
                x, y = X[sel], Y[sel]
                z = (y - GF.m_centre(x, model)) / np.sqrt(GF.cond_var_y(x, model))
                zs.append(z)
                sums.append([z.size, z.sum(), (z * z).sum()])
                infl.append(em_inflation(x, model, h))
            z = np.concatenate(zs) if zs else np.zeros(0)
            sums = np.array(sums, float).reshape(-1, 3)
            e = dict(x0=c, half_width=Y_HW, n=int(z.size), n_seeds=len(seeds),
                     n_per_seed_median=float(np.median(sums[:, 0])) if sums.size else math.nan)
            if z.size >= 2:
                ks = kstest(z, "norm")
                e.update(ks_D=float(ks.statistic), ks_p_nominal=float(ks.pvalue), mean_z=float(z.mean()),
                         var_z=float(z.var()),
                         em_var_z_expected=float(np.mean(np.concatenate(infl))),
                         hist_density=(np.histogram(np.clip(z, Z_EDGES[0], Z_EDGES[-1]), Z_EDGES, density=True)[0]))
                if len(seeds) >= 2 and n_boot > 0:
                    K = boot_counts(len(seeds), n_boot, seed)
                    T = K @ sums
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mu = T[:, 1] / T[:, 0]
                        var = T[:, 2] / T[:, 0] - mu ** 2
                    e.update(mean_z_ci95=[pct(mu, 2.5), pct(mu, 97.5)], var_z_ci95=[pct(var, 2.5), pct(var, 97.5)])
            res[f"{c:g}"] = e
        out[meth] = dict(seeds=seeds, windows=res)
    return out


# =================================================================================================== figures
class FigWriter:
    def __init__(self, out_dir, dpi, rel_root, enabled=True):
        self.out_dir, self.dpi, self.rel_root, self.enabled = out_dir, dpi, rel_root, enabled
        self.entries, self.skipped = [], []

    def skip(self, name, why):
        self.skipped.append(dict(name=name, why=why))
        print(f"  FIGURE SKIPPED {os.path.relpath(os.path.join(self.out_dir, name), self.rel_root)}: {why}", flush=True)

    def save(self, fig, name, description, waivers=()):
        if not self.enabled:
            plt.close(fig)
            return
        os.makedirs(self.out_dir, exist_ok=True)
        files = []
        for ext in ("pdf", "png"):
            p = os.path.join(self.out_dir, f"{name}.{ext}")
            fig.savefig(p, dpi=self.dpi if ext == "png" else None)
            files.append(os.path.relpath(p, self.rel_root))
        res = LG.check_figure_safe(fig, waivers=waivers)
        plt.close(fig)
        if res["status"] != "pass":
            print("  " + LG.one_line(name, res), flush=True)
        self.entries.append(dict(name=name, description=description, files=files, legibility=res))


def new_fig(w, h, nrows=1, ncols=1, **kw):
    fig, axs = plt.subplots(nrows, ncols, figsize=(w, h), layout="constrained", squeeze=False, **kw)
    return fig, axs


def title(fig, lines, width_in, banners=()):
    wchar = max(40, int(width_in * 12.0))
    txt = [wrap(lines[0], wchar)] + [wrap(s, int(wchar * 1.1)) for s in lines[1:]] + [wrap(b, int(wchar * 1.1)) for b in banners]
    fig.suptitle("\n".join(t for t in txt if t), fontsize=10, color=INK, linespacing=1.3)


def fig_legend(fig, handles, ncol):
    """Figure-level key below the panels; at most one column per ~2.9 in of figure width."""
    if handles:
        ncol = max(1, min(int(ncol), len(handles), int(fig.get_figwidth() / 2.9)))
        fig.legend(handles=handles, loc="outside lower center", ncol=ncol, frameon=False, fontsize=8.5)


def pos(a):
    """Non-positive values -> NaN (log axes: a zero is 'no data', not a plunge to -inf)."""
    a = np.asarray(a, dtype=float).copy()
    a[~(a > 0)] = np.nan
    return a


def log_ticks(ax, which="y"):
    """Readable log ticks (1, 2, 5 per decade, plain numbers) when the axis spans few decades."""
    axis = ax.yaxis if which == "y" else ax.xaxis
    lo, hi = (ax.get_ylim() if which == "y" else ax.get_xlim())
    if lo > 0 and hi > 0 and math.log10(hi / lo) < 2.5:
        axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_locator(NullLocator())


def ratio_ticks(ax):
    """Plain-number ticks on a log ratio axis (0.5, 0.8, 0.9, 1, 1.11, 1.25, 2, ...) inside the current limits."""
    lo, hi = ax.get_ylim()
    if lo > 0 and math.log10(hi / lo) > 0.6:
        cand = np.array([0.1, 0.2, 0.33, 0.5, 0.67, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    else:
        cand = np.array([0.5, 0.67, 0.8, 0.9, 1.0, 1.11, 1.25, 1.5, 2.0])
    t = cand[(cand >= lo) & (cand <= hi)]
    if t.size >= 2:
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(t))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(NullLocator())


def lam_axis(ax, lams):
    ax.set_xscale("log")
    ax.set_xlim(min(lams) / 1.35, max(lams) * 1.35)
    ax.set_xticks(sorted(lams))
    ax.set_xticklabels([f"{v:g}" for v in sorted(lams)])
    ax.xaxis.set_minor_locator(NullLocator())


def u_axis(ax, u):
    u = np.asarray(u, float)
    u = u[np.isfinite(u) & (u > 0)]
    ax.set_xscale("log")
    if u.size:
        ax.set_xlim(u.min() / 1.6, u.max() * 1.6)


def band(ax, x, st, color, lw=1.6, ls="-", alpha=0.18, zorder=3, logy=False):
    x = np.asarray(x, float)
    med, lo, hi = (np.asarray(st[k], float) for k in ("median", "q25", "q75"))
    if logy:
        med, lo, hi = pos(med), pos(lo), pos(hi)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, zorder=zorder - 1)
    ax.plot(x, med, color=color, lw=lw, ls=ls, zorder=zorder)


def ci_band(ax, x, mid, lo, hi, color, lw=1.6, ls="-", alpha=0.18, marker=None, logy=False):
    x = np.asarray(x, float)
    mid, lo, hi = (np.asarray(v, float) for v in (mid, lo, hi))
    if logy:
        mid, lo, hi = pos(mid), pos(lo), pos(hi)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)
    ax.plot(x, mid, color=color, lw=lw, ls=ls, marker=marker, ms=4)


def errpt(ax, x, mid, lo, hi, color, marker="o", ms=6, filled=True, zorder=4):
    mid, lo, hi = fnum(mid), fnum(lo), fnum(hi)
    if not math.isfinite(mid):
        return
    yerr = None
    if math.isfinite(lo) and math.isfinite(hi):
        yerr = [[max(mid - lo, 0.0)], [max(hi - mid, 0.0)]]
    ax.errorbar([x], [mid], yerr=yerr, fmt=marker, color=color, ms=ms, mfc=color if filled else "white",
                mec=color, capsize=2.5, lw=1.2, zorder=zorder)


def safe_log(ax, which, data):
    d = np.asarray(data, float)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size:
        (ax.set_yscale if which == "y" else ax.set_xscale)("log")
    return d.size > 0


def class_handles(names, with_all=True, arms=True):
    hs = [Line2D([], [], color=CLASS_COLOR[i], lw=2, label=f"clone age {names[i]} t.u." if i < 4 else "never cloned")
          for i in range(N_AGE)]
    if with_all:
        hs.append(Line2D([], [], color=C_FR, lw=2, ls="--", label="ABF+FR, all deposits"))
        hs.append(Line2D([], [], color=C_ABF, lw=2, ls="--", label="ABF (all deposits = never cloned)"))
    return hs


# --------------------------------------------------------------------------------------------------- per-cell figures
def _rng(*arrs, positive=False):
    """(min, max) of the finite (positive) values of the arrays, or None."""
    vals = [np.asarray(a, float).ravel() for a in arrs if a is not None]
    v = np.concatenate(vals) if vals else np.zeros(0)
    v = v[np.isfinite(v)]
    if positive:
        v = v[v > 0]
    return (float(v.min()), float(v.max())) if v.size else None


def _st_rng(st, positive=False):
    return _rng(st["median"], st["q25"], st["q75"], positive=positive)


def merge_ext(exts):
    """Panel-wise union of extent dicts {panel: (lo, hi) or None} (shared limits across the cells of one experiment
    and N)."""
    out = {}
    for e in exts:
        for k, r in (e or {}).items():
            if r is None:
                continue
            lo, hi = out.get(k, (math.inf, -math.inf))
            out[k] = (min(lo, r[0]), max(hi, r[1]))
    return out


def set_lin(ax, r, zero=False):
    if r is None:
        return
    lo, hi = r
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi) * 0.1, 1e-6)
    ax.set_ylim(lo - pad, hi + pad)


def set_log(ax, r, cap=None):
    """Log y axis on the shared extent r (padded by x1.5); cap: upper limit (fractions: 1)."""
    if r is None:
        return
    lo, hi = r[0] / 1.5, r[1] * 1.5
    if cap is not None:
        hi = min(hi, cap)
        lo = min(lo, cap / 10.0)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    log_ticks(ax, "y")


def ext_d1(D1):
    e, it = D1["end"], D1["interval"]
    return dict(a=_st_rng(e["death_rate_coarse"]), b=_st_rng(e["birth_rate_coarse"]),
                c=_rng(0.0, *(e["net_rate_coarse"][k] for k in ("median", "q25", "q75"))),
                d=_st_rng(it["death_rate_coarse"]), e=_st_rng(it["birth_rate_coarse"]),
                f=_st_rng(e["hazard_coarse_EXPLORATORY"], positive=True))


def ext_d2_fractions(F_fr, F_abf=None):
    out = {}
    for i, R in enumerate(REGIONS):
        st = stats_of(F_fr[..., i], 0)
        out[R] = _st_rng(st, positive=True)
    return out


def ext_d2_bias(d2fr, d2abf, paired=None):
    Bv, dv, pv = [], [], []
    for per in (d2fr, d2abf):
        for st in per or []:
            Bv += [st["B"], st["B_lo"], st["B_hi"]]
        if per:
            last = per[-1]
            Bv.append(root(last["raw2"]))
            dv += [last["dev"], last["dev_lo"], last["dev_hi"]]
            if "dev_coarse" in last:
                sel = (CENTRES_C > EVAL_LO) & (CENTRES_C < EVAL_HI)
                pv += [np.asarray(last[k], float)[..., sel] for k in ("dev_coarse", "dev_coarse_lo", "dev_coarse_hi")]
    for e in paired or []:
        Bv += [e.get("Bm_b"), *(e.get("Bm_b_ci95") or [])]
    return dict(B=_rng(0.0, *[np.asarray(fnum_arr(x), float) for x in Bv]), dev=_rng(0.0, *dv),
                prof=_rng(0.0, *pv))


def fnum_arr(x):
    a = np.asarray(x, dtype=object)
    return np.vectorize(fnum, otypes=[float])(a) if a.dtype == object else np.asarray(x, float)


def ext_d3(X_fr, X_abf):
    out = {}
    for j in (0, 1):
        for i, R in enumerate(REGIONS):
            sts = [stats_of(Xm[j][..., i], 0) for Xm in (X_fr, X_abf) if Xm is not None]
            out[f"{j}{R}"] = _rng(*[st[k] for st in sts for k in ("median", "q25", "q75")], positive=True)
    return out


def fig_d1(W, ctx, D1, lims=None):
    lims = lims or ext_d1(D1)
    fig, axs = new_fig(14.5, 7.8, 2, 3)
    x = CENTRES_C
    e = D1["end"]
    panels = ((axs[0, 0], e["death_rate_coarse"], "(a) FR deaths at the end", "deaths / (opportunity x walker)"),
              (axs[0, 1], e["birth_rate_coarse"], "(b) FR births (extra copies) at the end", "births / (opportunity x walker)"),
              (axs[0, 2], e["net_rate_coarse"], "(c) net births - deaths at the end", "net / (opportunity x walker)"))
    for (ax, st, t, yl), pk in zip(panels, "abc"):
        band(ax, x, st, C_FR)
        ax.axhline(0, color=MUTED, lw=0.8)
        set_lin(ax, lims.get(pk), zero=True)
        ax.set_title(t, loc="left")
        ax.set_xlabel("x (0.1-wide bins)")
        ax.set_ylabel(yl)
    u = D1["snap_u"]
    cols = [SNAP_CMAP(i / max(1, len(u) - 1)) for i in range(len(u))]
    for ax, st, t, yl, pk in ((axs[1, 0], D1["interval"]["death_rate_coarse"], "(d) death rate per snapshot interval",
                               "deaths / (opportunity x walker)", "d"),
                              (axs[1, 1], D1["interval"]["birth_rate_coarse"], "(e) birth rate per snapshot interval",
                               "births / (opportunity x walker)", "e")):
        med, q25, q75 = (np.asarray(st[k], float) for k in ("median", "q25", "q75"))
        for i in range(len(u)):
            ax.fill_between(x, q25[i], q75[i], color=cols[i], alpha=0.10, lw=0)
            ax.plot(x, med[i], color=cols[i], lw=1.4)
        set_lin(ax, lims.get(pk), zero=True)
        ax.set_title(t, loc="left")
        ax.set_xlabel("x (0.1-wide bins)")
        ax.set_ylabel(yl)
    ax = axs[1, 2]
    st = e["hazard_coarse_EXPLORATORY"]
    band(ax, x, st, INK2, logy=True)
    set_log(ax, lims.get("f"))
    ax.set_title("(f) EXPLORATORY: deaths per walker-opportunity in the bin", loc="left")
    ax.set_xlabel("x (0.1-wide bins)")
    ax.set_ylabel("death hazard per occupancy")
    for a in axs.ravel():
        for xb in (-0.5, 0.5):
            a.axvline(xb, color=AXIS, lw=0.8, ls=":")
    hs = [Line2D([], [], color=C_FR, lw=2, label=f"ABF+FR seed median (n = {D1['n']})"),
          Patch(facecolor=C_FR, alpha=0.18, label="interquartile range over seeds (d, e: faint, per interval)")]
    hs += [Line2D([], [], color=cols[i], lw=2, label=f"interval ending u = {u[i]:g}") for i in range(len(u))]
    fig_legend(fig, hs, 5)
    title(fig, [f"D1 FR death and birth positions: {ctx['label']}, N = {ctx['N']}",
                "Rates per FR opportunity per walker in 0.1-wide x bins (5 production bins); seed median and IQR. "
                "Dotted lines: region boundaries x = +-0.5. y ranges shared by every cell of this experiment at this "
                "N."], 14.5, ctx["banners"])
    W.save(fig, "mech_d1_fr_events", "D1: FR death / birth x-position histograms (per opportunity per walker), end "
           "and per snapshot interval; panel (f) exploratory occupancy hazard")


def fig_d2_fractions(W, ctx, F_fr, F_abf, u, names, lims=None):
    lims = lims or ext_d2_fractions(F_fr, F_abf)
    fig, axs = new_fig(15.5, 5.0, 1, 4)
    for i, R in enumerate(REGIONS):
        ax = axs[0, i]
        st = stats_of(F_fr[..., i], 0)                  # (n_snap, 5)
        for c in range(N_AGE):
            band(ax, u, dict(median=st["median"][:, c], q25=st["q25"][:, c], q75=st["q75"][:, c]), CLASS_COLOR[c],
                 logy=True)
        set_log(ax, lims.get(R), cap=1.0)
        u_axis(ax, u)
        ax.set_title(f"({'abc'[i]}) ABF+FR, {REGION_LABEL[R]}", loc="left")
        ax.set_xlabel("budget fraction u (snapshot)")
        ax.set_ylabel("cumulative fraction of deposits")
    ax = axs[0, 3]
    pos, labels = [], []
    k = 0
    for i, R in enumerate(REGIONS):
        for meth, Fm, col in (("ABF", F_abf, C_ABF), ("FR", F_fr, C_FR)):
            if Fm is None or Fm.size == 0:
                k += 1
                continue
            med = np.nanmedian(Fm[:, -1, :, i], axis=0)
            bottom = 0.0
            for c in range(N_AGE):
                v = float(np.nan_to_num(med[c]))
                ax.bar(k, v, bottom=bottom, color=CLASS_COLOR[c], width=0.7, edgecolor="white", lw=0.4)
                bottom += v
            pos.append(k)
            labels.append(f"{R}\n{meth}")
            k += 1
        k += 0.5
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_title("(d) composition at the end (median)", loc="left")
    ax.set_xlabel("region and arm")
    ax.set_ylabel("fraction of deposits")
    hs = class_handles(names, with_all=False) + [Patch(facecolor=INK2, alpha=0.18, label="interquartile range over seeds")]
    fig_legend(fig, hs, 6)
    title(fig, [f"D2(i) clone-age composition of the force deposits: {ctx['label']}, N = {ctx['N']}",
                f"Cumulative fractions at the snapshots (seed median, IQR; ABF+FR n = {F_fr.shape[0]}). ABF never "
                "clones: its deposits are all 'never cloned'. y ranges (a-c) shared by every cell of this experiment "
                "at this N, capped at 1."], 15.5, ctx["banners"])
    W.save(fig, "mech_d2_age_fractions", "D2(i): fraction of deposits per clone-age class, per region and snapshot")


def fig_d2_bias(W, ctx, d2fr, d2abf, u, names, paired=None, lims=None):
    lims = lims or ext_d2_bias(d2fr, d2abf, paired)
    fig, axs = new_fig(14.5, 9.0, 2, 2)
    end_fr = d2fr[-1] if d2fr else None
    end_abf = d2abf[-1] if d2abf else None
    # (a) debiased RMS bias at the end, per class; ABF's error at the same positions (matched, paired seeds)
    ax = axs[0, 0]
    ticks, tl = [], []
    for c in range(N_AGE + 1):
        if end_fr is None:
            break
        col = CLASS_COLOR[c] if c < N_AGE else C_FR
        errpt(ax, c - 0.12, end_fr["B"][c], end_fr["B_lo"][c], end_fr["B_hi"][c], col)
        ax.plot([c + 0.06], [np.sqrt(fnum(end_fr["raw2"][c]))], marker="D", ms=4.5, mfc="white", mec=col, ls="none")
        if paired is not None:
            e = paired[c]
            errpt(ax, c + 0.22, e["Bm_b"], e["Bm_b_ci95"][0], e["Bm_b_ci95"][1], C_ABF, marker="s", ms=5, filled=False)
        ticks.append(c)
        tl.append(names[c] if c < N_AGE else "FR all")
    if end_abf is not None:
        errpt(ax, N_AGE + 1 - 0.12, end_abf["B"][ALL], end_abf["B_lo"][ALL], end_abf["B_hi"][ALL], C_ABF)
        ax.plot([N_AGE + 1.06], [np.sqrt(fnum(end_abf["raw2"][ALL]))], marker="D", ms=4.5, mfc="white", mec=C_ABF,
                ls="none")
        ticks.append(N_AGE + 1)
        tl.append("ABF all")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tl, fontsize=8)
    set_lin(ax, lims.get("B"), zero=True)
    ax.set_title("(a) debiased RMS mean-force bias B_c at the end", loc="left")
    ax.set_xlabel("clone-age class (t.u.) / arm")
    ax.set_ylabel("B_c (reduced force)")
    # (b) B over the snapshots
    ax = axs[0, 1]
    for c in range(N_AGE):
        if not d2fr:
            break
        mid = [d["B"][c] for d in d2fr]
        ci_band(ax, u, mid, [d["B_lo"][c] for d in d2fr], [d["B_hi"][c] for d in d2fr], CLASS_COLOR[c], marker="o")
    if d2fr:
        ci_band(ax, u, [d["B"][ALL] for d in d2fr], [d["B_lo"][ALL] for d in d2fr], [d["B_hi"][ALL] for d in d2fr], C_FR,
                ls="--")
    if d2abf:
        ci_band(ax, u, [d["B"][ALL] for d in d2abf], [d["B_lo"][ALL] for d in d2abf], [d["B_hi"][ALL] for d in d2abf],
                C_ABF, ls="--")
    u_axis(ax, u)
    set_lin(ax, lims.get("B"), zero=True)
    ax.set_title("(b) B_c of the cumulative deposits at each snapshot", loc="left")
    ax.set_xlabel("budget fraction u (snapshot)")
    ax.set_ylabel("B_c (reduced force)")
    # (c) signed deviation per region at the end
    ax = axs[1, 0]
    regs = list(DEV_REGIONS)
    offs = np.linspace(-0.3, 0.3, N_AGE + 2)
    for ri, R in enumerate(regs):
        for c in range(N_AGE + 1):
            if end_fr is None:
                break
            col = CLASS_COLOR[c] if c < N_AGE else C_FR
            errpt(ax, ri + offs[c], end_fr["dev"][c][ri], end_fr["dev_lo"][c][ri], end_fr["dev_hi"][c][ri], col, ms=5)
        if end_abf is not None:
            errpt(ax, ri + offs[-1], end_abf["dev"][ALL][ri], end_abf["dev_lo"][ALL][ri], end_abf["dev_hi"][ALL][ri],
                  C_ABF, ms=5)
    ax.axhline(0, color=MUTED, lw=0.8)
    set_lin(ax, lims.get("dev"), zero=True)
    ax.set_xticks(range(len(regs)))
    ax.set_xticklabels(["left well", "gate (all)", "gate x<0", "gate x>0", "right well"], fontsize=8)
    ax.set_title("(c) signed deposit-weighted deviation D_cR at the end (eval window)", loc="left")
    ax.set_xlabel("region")
    ax.set_ylabel("mean f - <F*'>_bin (reduced force)")
    # (d) bin profile of the deviation
    ax = axs[1, 1]
    for c, col, ls, src in ((0, CLASS_COLOR[0], "-", end_fr), (1, CLASS_COLOR[1], "-", end_fr),
                            (4, CLASS_COLOR[4], "-", end_fr), (ALL, C_FR, "--", end_fr), (ALL, C_ABF, "--", end_abf)):
        if src is None or "dev_coarse" not in src:
            continue
        sel = (CENTRES_C > EVAL_LO) & (CENTRES_C < EVAL_HI)
        ci_band(ax, CENTRES_C[sel], src["dev_coarse"][c][sel], src["dev_coarse_lo"][c][sel],
                src["dev_coarse_hi"][c][sel], col, ls=ls, alpha=0.12)
    ax.axhline(0, color=MUTED, lw=0.8)
    set_lin(ax, lims.get("prof"), zero=True)
    ax.set_title("(d) deviation profile at the end (0.1-wide bins)", loc="left")
    ax.set_xlabel("x")
    ax.set_ylabel("mean f - <F*'>_bin (reduced force)")
    hs = class_handles(names) + [Line2D([], [], marker="D", ms=5, mfc="white", mec=INK2, ls="none",
                                        label="raw RMS (not debiased)"),
                                 Line2D([], [], marker="s", ms=5, mfc="white", mec=C_ABF, color=C_ABF, ls="none",
                                        label="(a) EXPLORATORY: ABF's B at the class's positions (matched, paired)"),
                                 Patch(facecolor=INK2, alpha=0.18, label="95 % seed-bootstrap CI")]
    fig_legend(fig, hs, 5)
    nfr = d2fr[-1]["n_seeds"] if d2fr else 0
    nab = d2abf[-1]["n_seeds"] if d2abf else 0
    title(fig, [f"D2(ii) class-wise conditional mean-force error vs the bin-averaged F*': {ctx['label']}, N = {ctx['N']}",
                "B_c^2 = sum_j n_cj [(m_cj - <F*'>_j)^2 - V_cj] / sum_j n_cj over eval bins (V_cj: seed-cluster variance "
                f"of the pooled bin mean); CIs: {ctx['n_boot']} seed-bootstrap resamples (ABF+FR n = {nfr}, ABF n = {nab}; "
                "B_c: basic interval of the debiased B_c^2). The ABF arm is the reference level. y ranges shared by "
                "every cell of this experiment at this N."], 14.5, ctx["banners"])
    W.save(fig, "mech_d2_mean_force_bias", "D2(ii): debiased RMS and signed class-wise conditional mean-force "
           "deviation from the bin-averaged F*' (the plan's D2 quantity; open squares in (a): EXPLORATORY matched ABF "
           "level)")


def fig_d2_var(W, ctx, end_fr, end_abf, names, exact):
    fig, axs = new_fig(14.5, 5.4, 1, 2)
    ax = axs[0, 0]
    allv = []
    for c, col, src, ls in [(i, CLASS_COLOR[i], end_fr, "-") for i in range(N_AGE)] + [(ALL, C_ABF, end_abf, "--")]:
        if src is None:
            continue
        v = np.asarray(src["var_coarse"][c], float)
        ok = np.isfinite(v) & (v > 0)
        lo = np.asarray(src["var_coarse_lo"][c], float)
        hi = np.asarray(src["var_coarse_hi"][c], float)
        okb = ok & np.isfinite(lo) & np.isfinite(hi)
        ax.fill_between(CENTRES_C, np.where(okb, pos(lo), np.nan), np.where(okb, hi, np.nan), color=col, alpha=0.13,
                        lw=0)
        ax.plot(CENTRES_C[ok], v[ok], color=col, lw=1.4, ls=ls, marker="o", ms=3)
        allv.append(v[ok])
    ev = exact["etot"][EVAL]
    floor = float(np.min(ev[ev > 0])) / 10.0 if np.any(ev > 0) else 1e-8
    ax.plot(CENTRES, pos(exact["etot"]), color=MUTED, lw=1.0)
    ax.plot(CENTRES, np.where(exact["varf"] > floor, exact["varf"], np.nan), color=MUTED, lw=1.2, ls=":")
    ref_src = end_fr if end_fr is not None else end_abf
    if ref_src is not None and "etot_coarse" in ref_src:
        ec = np.asarray(ref_src["etot_coarse"][ALL], float)
        ok = np.isfinite(ec) & (ec > 0)
        ax.plot(CENTRES_C[ok], ec[ok], color=INK, lw=0, marker="x", ms=6, mew=1.4, zorder=6)
    allv.append(ev[ev > 0])
    if safe_log(ax, "y", np.concatenate(allv) if allv else []):
        top = max(float(np.nanmax(np.concatenate(allv))) * 3.0, floor * 10)
        ax.set_ylim(floor, top)
    ax.set_xlim(EVAL_LO, EVAL_HI)
    ax.set_title("(a) within-bin variance of f per clone-age class", loc="left")
    ax.set_xlabel("x (0.1-wide bins)")
    ax.set_ylabel("Var(f | bin) (reduced force^2)")
    ax = axs[0, 1]
    regs = ["left", "gate", "right"]
    ridx = [list(DEV_REGIONS).index(r) for r in regs]
    offs = np.linspace(-0.3, 0.3, N_AGE + 1)
    for k, ri in enumerate(ridx):
        for c in range(N_AGE):
            if end_fr is None:
                break
            errpt(ax, k + offs[c], end_fr["var_ratio"][c][ri], end_fr["var_ratio_lo"][c][ri],
                  end_fr["var_ratio_hi"][c][ri], CLASS_COLOR[c], ms=5)
        if end_abf is not None:
            errpt(ax, k + offs[-1], end_abf["var_ratio"][ALL][ri], end_abf["var_ratio_lo"][ALL][ri],
                  end_abf["var_ratio_hi"][ALL][ri], C_ABF, ms=5)
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(regs)))
    ax.set_xticklabels([REGION_LABEL[r] for r in regs], fontsize=8)
    ax.set_title("(b) measured / exact within-bin variance (eval window)", loc="left")
    ax.set_xlabel("region")
    ax.set_ylabel("ratio sum SSW / sum n e_j")
    hs = class_handles(names, with_all=False) + [
        Line2D([], [], color=C_ABF, lw=2, ls="--", label="ABF (all deposits)"),
        Line2D([], [], color=INK, lw=0, marker="x", ms=6, mew=1.4,
               label="exact e_j, deposit-weighted over the 0.1 bin (compare with the curves)"),
        Line2D([], [], color=MUTED, lw=1.2, label="exact e_j = <Var(f|x)> + Var(F*' | bin), 0.02 bins"),
        Line2D([], [], color=MUTED, lw=1.4, ls=":", label="exact <Var(f|x)>_bin"),
        Patch(facecolor=INK2, alpha=0.13, label="(a, b) 95 % seed-bootstrap CI")]
    fig_legend(fig, hs, 5)
    title(fig, [f"D2(iii) class-wise conditional force variance: {ctx['label']}, N = {ctx['N']}",
                "Within-bin variance of the deposited force pooled over seeds vs the exact (2 alpha^2/beta^2)(w'/w)^2 "
                "(shift: (2/beta^2)(w'/w)^2) bin average plus the within-bin variance of F*'; bands (a) and error bars "
                "(b): 95 % seed-bootstrap CIs."],
          14.5, ctx["banners"] + [BANNER_EXPLORATORY])
    W.save(fig, "mech_d2_force_variance", "D2(iii) EXPLORATORY: class-wise conditional force variance vs exact")


def fig_d3(W, ctx, X_fr, X_abf, u, names, lims=None):
    lims = lims or ext_d3(X_fr, X_abf)
    fig, axs = new_fig(15.0, 8.4, 2, 3)
    for row, (lab, j) in enumerate((("cumulative", 0), ("per snapshot interval", 1))):
        for i, R in enumerate(REGIONS):
            ax = axs[row, i]
            allm = []
            for Fm, ls in ((X_fr, "-"), (X_abf, "--")):
                if Fm is None:
                    continue
                st = stats_of(Fm[j][..., i], 0)
                for c in range(N_AGE):
                    band(ax, u, dict(median=st["median"][:, c], q25=st["q25"][:, c], q75=st["q75"][:, c]),
                         CLASS_COLOR[c], ls=ls, alpha=0.10, logy=True)
                allm.append(st["median"])
            set_log(ax, lims.get(f"{j}{R}"), cap=1.0)
            u_axis(ax, u)
            ax.set_title(f"({'abcdef'[row * 3 + i]}) {lab}, {REGION_LABEL[R]}", loc="left")
            ax.set_xlabel("budget fraction u (snapshot)")
            ax.set_ylabel("fraction of deposits")
    hs = [Line2D([], [], color=CLASS_COLOR[c], lw=2, label=(f"crossed {names[c]} t.u. ago" if c < 4 else "never crossed"))
          for c in range(N_AGE)]
    hs += [Line2D([], [], color=INK2, lw=2, label="ABF+FR (solid)"), Line2D([], [], color=INK2, lw=2, ls="--",
                                                                          label="ABF (dashed)"),
           Patch(facecolor=INK2, alpha=0.12, label="interquartile range over seeds")]
    fig_legend(fig, hs, 4)
    nf = X_fr[0].shape[0] if X_fr is not None else 0
    na = X_abf[0].shape[0] if X_abf is not None else 0
    title(fig, [f"D3 deposits by time since the walker last crossed the gate: {ctx['label']}, N = {ctx['N']}",
                f"Fractions of the deposits per region (seed median and IQR; ABF+FR n = {nf}, ABF n = {na}). Clones "
                "inherit the crossing time. y ranges shared by every cell of this experiment at this N, capped at 1."],
          15.0, ctx["banners"])
    W.save(fig, "mech_d3_crossing", "D3: deposits by time-since-crossing class per region and snapshot, FR vs ABF")


def fig_ylaw(W, ctx, Y):
    fig, axs = new_fig(17.5, 5.6, 1, len(Y_WINDOWS))
    zz = np.linspace(-5, 5, 401)
    pdf = np.exp(-0.5 * zz ** 2) / math.sqrt(2 * math.pi)
    centres = 0.5 * (Z_EDGES[1:] + Z_EDGES[:-1])
    for i, c in enumerate(Y_WINDOWS):
        ax = axs[0, i]
        stat = []
        for meth, col in (("abf", C_ABF), ("fr", C_FR)):
            e = (Y.get(meth) or {}).get("windows", {}).get(f"{c:g}")
            if not e or "hist_density" not in e:
                stat.append(f"{meth.upper()} n {e['n'] if e else 0}")
                continue
            ax.step(centres, e["hist_density"], where="mid", color=col, lw=1.5)
            ci = e.get("var_z_ci95")
            cis = f" [{fnum(ci[0]):.2f}, {fnum(ci[1]):.2f}]" if ci else ""
            stat.append(f"{'ABF' if meth == 'abf' else 'FR'} n {e['n']}: KS D {e['ks_D']:.2f}, mean {e['mean_z']:+.2f}\n"
                        f"var {e['var_z']:.2f}{cis}")
        ax.plot(zz, pdf, color=INK, lw=1.1, ls=":")
        ax.set_title(f"x = {c:g} +- {Y_HW:g}", loc="left")
        ax.set_xlabel("z = (y - m(x)) / sd(y|x)\n" + "\n".join(stat), fontsize=8.5)
        ax.set_ylabel("density")
    hs = [Line2D([], [], color=C_ABF, lw=2, label="ABF final walkers"),
          Line2D([], [], color=C_FR, lw=2, label="ABF+FR final walkers"),
          Line2D([], [], color=INK, lw=1.4, ls=":", label="exact conditional law N(0, 1)")]
    fig_legend(fig, hs, 3)
    title(fig, [f"Conditional y law at representative x: {ctx['label']}, N = {ctx['N']}",
                "Final walkers pooled over seeds, standardised by the exact Y | x law; KS D vs N(0, 1) (nominal: FR "
                "copies are not independent), mean and var of z (95 % seed-bootstrap CI of var; exact: 0 and 1, EM "
                "frozen-x var ~ 1/(1 - lam k h / 2) <= 1.013)."],
          17.5, ctx["banners"] + [BANNER_EXPLORATORY])
    W.save(fig, "mech_ylaw", "EXPLORATORY: conditional y law of the final walkers at x in {-1, -0.21, 0, 0.21, 1} vs "
           "the exact law, KS statistic")


def _trace_segments(t, v, reb):
    """(t, v) with a NaN inserted before every sample at which the id was reborn (no line across a rebirth)."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    if reb is None or t.size < 2:
        return t, v
    jump = np.flatnonzero(np.diff(np.asarray(reb)) != 0) + 1
    return np.insert(t, jump, np.nan), np.insert(v, jump, np.nan)


def fig_traces(W, ctx, runs, model):
    fig, axs = new_fig(14.5, 9.0, 3, 2)
    seeds = {}
    for j, (meth, lab) in enumerate((("abf", "ABF"), ("fr", "ABF+FR"))):
        rr = runs.get(meth) or {}
        if not rr:
            for i in range(3):
                axs[i, j].set_visible(False)
            continue
        s = min(rr)
        seeds[meth] = s
        r = rr[s]
        t = np.asarray(r["traces_t"], float)
        X = np.asarray(r["traces"], float)
        Yv = np.asarray(r["traces_y"], float)
        reb = np.asarray(r["traces_rebirth"]) if "traces_rebirth" in r else None
        for k in range(min(4, X.shape[1])):
            col = ["#008300", "#4a3aa7", "#e87ba4", "#eda100"][k]
            rk = reb[:, k] if reb is not None else None
            axs[0, j].plot(*_trace_segments(t, X[:, k], rk), color=col, lw=1.0)
            axs[1, j].plot(*_trace_segments(t, Yv[:, k], rk), color=col, lw=1.0)
            z = (Yv[:, k] - GF.m_centre(X[:, k], model)) / np.sqrt(GF.cond_var_y(X[:, k], model))
            axs[2, j].plot(*_trace_segments(t, z, rk), color=col, lw=1.0)
        for i in range(3):
            axs[i, j].set_xlabel("physical time t (t.u.)")
        axs[0, j].set_ylabel("x")
        axs[1, j].set_ylabel("y")
        axs[2, j].set_ylabel("z_y = (y - m(x)) / sd(y|x)")
        for yv in (-2, 2):
            axs[2, j].axhline(yv, color=MUTED, lw=0.8, ls=":")
        axs[0, j].set_title(f"({'ab'[j]}) {lab}, seed {s}: x(t)", loc="left")
        axs[1, j].set_title(f"({'cd'[j]}) {lab}: y(t)", loc="left")
        axs[2, j].set_title(f"({'ef'[j]}) {lab}: standardised y (dotted: +-2)", loc="left")
    hs = [Line2D([], [], color=["#008300", "#4a3aa7", "#e87ba4", "#eda100"][k], lw=2, label=f"traced walker id {k}")
          for k in range(4)]
    fig_legend(fig, hs, 4)
    title(fig, [f"Walker traces with y (D4): {ctx['label']}, N = {ctx['N']}",
                "First four traced walker ids; FR lines break where an id is reborn (copy of another walker); "
                "samples every 0.1 t.u. to t = 40, sparser later."], 14.5, ctx["banners"] + [BANNER_EXPLORATORY])
    W.save(fig, "mech_ytraces", "EXPLORATORY: example x / y / standardised-y traces (D4)")
    return seeds


# =================================================================================================== per-cell driver
def analyse_cell(exp, cname, P, cfg_path, args, out_root, ref_cache):
    print(f"[{exp}/{cname}]", flush=True)
    model = cell_model(P)
    h = float(P["engine_cfg"]["h"])
    rroot = runs_root(P, args.results_override)
    adir = analysis_dir(P, args.analysis_override)
    t0 = time.time()
    doc = dict(schema=SCHEMA_CELL, experiment=exp, cell=cname, cell_label=P.get("cell_label"), config=cfg_path,
               model=dict(variant=model["variant"], alpha=P["engine_cfg"].get("alpha"), kappa=P["engine_cfg"].get("kappa"),
                          lam=model["lam"]),
               production=is_production(P), results_root=rroot, analysis_dir=adir, n_boot=int(args.n_boot),
               boot_seed=BOOT_SEED, regions={k: REGION_LABEL[k] for k in DEV_REGIONS}, warnings=[], per_N={},
               definitions=DEFINITIONS)
    # scorer bin reference (frozen cell reference verified by GatewayFamilyScorer)
    ref = None
    try:
        sc = MF.make_scorer("gateway_family", P)
        fbar, w, s2h, mk = sc.bin_reference(sc.sa)
        e = sc.sa.est(NB, 1)[0]
        ex = exact_bin_force_moments(model, e.sub_x.numpy(), w, s2h)
        ev_bins = np.zeros(NB, bool)
        ev_bins[np.unique(s2h[mk])] = True
        if not np.array_equal(ev_bins, EVAL):
            raise MechError("scorer eval bins != the 150 bins with centres in (-1.5, 1.5)")
        dmax = float(np.max(np.abs(ex["fbar"] - fbar)))
        if dmax > 1e-9:
            raise MechError(f"exact bin-averaged F*' disagrees with the scorer's bin reference by {dmax:.3g}")
        ref = dict(r=np.asarray(fbar, float), etot=ex["etot"], varf=ex["varf"], check_max_abs=dmax)
        doc["reference"] = dict(source="eqb_family.GatewayFamilyScorer(...).bin_reference(scorer.sa)",
                                reference_file=P["reference_file"], max_abs_vs_exact_bin_average=dmax)
    except Exception as e:  # noqa: BLE001
        doc["reference"] = dict(error=f"{type(e).__name__}: {e}")
        doc["warnings"].append(f"no bin reference ({type(e).__name__}: {e}): D2 deviations and variances not computed")
        print(f"  WARNING: no bin reference: {e}", flush=True)
    ref_cache[(exp, cname)] = ref
    keep, defer = {}, []
    for N in [int(n) for n in P["N_ladder"]]:
        if args.N and N not in args.N:
            continue
        ctx = dict(label=P.get("cell_label") or f"{exp}/{cname}", N=N, n_boot=int(args.n_boot), banners=[])
        if not is_production(P):
            ctx["banners"].append(f"NON-PRODUCTION DATA (B = {P['B']:.4g}, {len(P['seeds'])} seeds): pipeline check, "
                                  "never interpreted (SCIENTIFIC_PLAN section 9)")
        fdir = os.path.join(out_root, "figures", "mechanism", exp, cname, f"N{N}")
        W = FigWriter(fdir, args.dpi, out_root, enabled=not args.no_figures)
        runsN = collect_N(P, rroot, N)
        R = dict(status={m: status_counts(runsN[m]) for m in ("abf", "fr")})
        dr = {m: diag_runs(runsN[m]) for m in ("abf", "fr")}
        cr = {m: complete_runs(runsN[m]) for m in ("abf", "fr")}
        names_age, names_cross = AGE_DEFAULT, CROSS_DEFAULT
        any_run = next((r for m in ("fr", "abf") for r in dr[m].values()), None)
        if any_run is not None:
            dc = any_run["meta"].get("diag_classes", {})
            names_age = list(dc.get("clone_age_class_names", AGE_DEFAULT))
            names_cross = list(dc.get("cross_age_class_names", CROSS_DEFAULT))
            for m in ("abf", "fr"):
                for s, r in dr[m].items():
                    d = r["meta"].get("diag_classes", {})
                    if d.get("clone_age_class_names", names_age) != names_age or \
                            d.get("cross_age_class_names", names_cross) != names_cross:
                        doc["warnings"].append(f"N{N} {m} seed {s}: class edges differ from the other runs")
        R["class_names"] = dict(clone_age=names_age, crossing=names_cross)
        snap_u = None
        # ---- D1
        if dr["fr"]:
            D1 = d1_compute(dr["fr"], N)
            snap_u = D1["snap_u"]
            R["D1"] = D1
            defer.append(dict(kind="d1", key=(exp, N), fn=fig_d1, W=W, ctx=ctx, args=(D1,), kw={}, ext=ext_d1(D1)))
        else:
            R["D1"] = dict(missing="no complete ABF+FR run with D1-D4 diagnostics")
            W.skip("mech_d1_fr_events", R["D1"]["missing"])
        # ---- D2
        d2 = {}
        for m in ("abf", "fr"):
            rs = [dr[m][s] for s in sorted(dr[m])]
            if not rs or ref is None:
                continue
            nsnap = np.asarray(rs[0]["snap_step"]).size
            if any(np.asarray(r["snap_step"]).size != nsnap for r in rs):
                doc["warnings"].append(f"N{N} {m}: runs have different snapshot grids; D2 not computed")
                continue
            snap_u = np.asarray(rs[0]["snap_u"], float) if snap_u is None else snap_u
            per = []
            for si in range(nsnap):
                n, s1, s2 = d2_arrays(rs, si)
                calc = D2Calc(n, s1, s2, ref["r"], ref["etot"])
                st = d2_summarise(calc, int(args.n_boot), BOOT_SEED, profiles=(si == nsnap - 1))
                st["n_seeds"] = calc.S
                per.append(st)
                if si == nsnap - 1:
                    keep[(N, m)] = dict(seeds=sorted(dr[m]), arrays=(n, s1, s2))
            d2[m] = per
        if "fr" in d2 or "abf" in d2:
            R["D2"] = d2_json(d2, snap_u, names_age)
        else:
            R["D2"] = dict(missing="no complete run with D2 diagnostics" if ref is not None else "no bin reference")
        # class fractions (i)
        Ffr = class_fractions([dr["fr"][s] for s in sorted(dr["fr"])], "dep_age_C") if dr["fr"] else None
        Fab = class_fractions([dr["abf"][s] for s in sorted(dr["abf"])], "dep_age_C") if dr["abf"] else None
        if Ffr is not None:
            R["D2_fractions"] = dict(
                snap_u=snap_u, classes=names_age, regions=list(REGIONS),
                fr_cumulative=stats_of(Ffr[0]), fr_interval=stats_of(Ffr[1]),
                abf_cumulative=stats_of(Fab[0]) if Fab is not None else None,
                recent_lt_0p1_gate_end=stats_of(Ffr[0][:, -1, 0, 1] + Ffr[0][:, -1, 1, 1]),
                recent_lt_0p1_by_region_end={R_: stats_of(Ffr[0][:, -1, 0, i] + Ffr[0][:, -1, 1, i])
                                             for i, R_ in enumerate(REGIONS)},
                axes="(snapshot, class, region)")
            fa = (Ffr[0], Fab[0] if Fab is not None else None, snap_u, names_age)
            defer.append(dict(kind="d2f", key=(exp, N), fn=fig_d2_fractions, W=W, ctx=ctx, args=fa, kw={},
                              ext=ext_d2_fractions(fa[0], fa[1])))
        else:
            R["D2_fractions"] = dict(missing="no complete ABF+FR run with D2 diagnostics")
            W.skip("mech_d2_age_fractions", R["D2_fractions"]["missing"])
        if d2:
            # paired FR class vs the ABF reference level (same seeds), incl. the matched-position excess
            common = sorted(set(dr["fr"]) & set(dr["abf"]))
            pr = None
            if len(common) >= 2 and ref is not None:
                ca = D2Calc(*d2_arrays([dr["fr"][s] for s in common], -1), ref["r"], ref["etot"])
                cb = D2Calc(*d2_arrays([dr["abf"][s] for s in common], -1), ref["r"], ref["etot"])
                pr = d2_paired(ca, list(range(N_AGE + 1)), cb, [ALL] * (N_AGE + 1), int(args.n_boot), matched=True)
                R["D2_fr_vs_abf"] = {(names_age[c] if c < N_AGE else "all"): jsonify_paired(pr[c])
                                     for c in range(N_AGE + 1)}
                R["D2_fr_vs_abf_definition"] = (
                    "paired seed bootstrap over the seeds with both arms, end snapshot: B_diff = B_c(FR) - B(ABF all "
                    "deposits), dev_diff = D_cR(FR) - D_R(ABF); matched-position: Bm_a = B_c(FR) and Bm_b = B(ABF) both "
                    "weighted by the FR class's deposit counts n_cj over the bins usable in both arms, B_excess = Bm_a - "
                    "Bm_b, dev_excess_R = sum_R n_cj (delta_cj^FR - delta_j^ABF) / sum_R n_cj; p two-sided bootstrap")
            else:
                R["D2_fr_vs_abf"] = dict(missing=f"fewer than 2 seeds with both arms ({common})")
            defer.append(dict(kind="d2b", key=(exp, N), fn=fig_d2_bias, W=W, ctx=ctx,
                              args=(d2.get("fr"), d2.get("abf"), snap_u, names_age), kw=dict(paired=pr),
                              ext=ext_d2_bias(d2.get("fr"), d2.get("abf"), pr)))
            fig_d2_var(W, ctx, d2["fr"][-1] if "fr" in d2 else None, d2["abf"][-1] if "abf" in d2 else None, names_age,
                       dict(etot=ref["etot"], varf=ref["varf"]))
        else:
            for nm in ("mech_d2_mean_force_bias", "mech_d2_force_variance"):
                W.skip(nm, R["D2"].get("missing", "no D2"))
        # ---- D3
        Xfr = class_fractions([dr["fr"][s] for s in sorted(dr["fr"])], "dep_cross_C") if dr["fr"] else None
        Xab = class_fractions([dr["abf"][s] for s in sorted(dr["abf"])], "dep_cross_C") if dr["abf"] else None
        if Xfr is not None or Xab is not None:
            D3 = dict(snap_u=snap_u, classes=names_cross, regions=list(REGIONS), axes="(snapshot, class, region)")
            for tag, Xm in (("fr", Xfr), ("abf", Xab)):
                if Xm is not None:
                    D3[f"{tag}_cumulative"] = stats_of(Xm[0])
                    D3[f"{tag}_interval"] = stats_of(Xm[1])
            common = sorted(set(dr["fr"]) & set(dr["abf"]))
            if Xfr is not None and Xab is not None and common:
                fi = [sorted(dr["fr"]).index(s) for s in common]
                ai = [sorted(dr["abf"]).index(s) for s in common]
                diff = Xfr[0][fi, -1] - Xab[0][ai, -1]           # (S, 5, nR)
                D3["fr_minus_abf_end"] = {
                    R_: {names_cross[c]: dict(zip(("median", "ci_lo", "ci_hi"),
                                                  M.boot_median_ci(diff[:, c, i], int(args.n_boot), BOOT_SEED)))
                         for c in range(N_AGE)} for i, R_ in enumerate(REGIONS)}
                D3["fr_minus_abf_n_seeds"] = len(common)
            R["D3"] = D3
            defer.append(dict(kind="d3", key=(exp, N), fn=fig_d3, W=W, ctx=ctx, args=(Xfr, Xab, snap_u, names_cross),
                              kw={}, ext=ext_d3(Xfr, Xab)))
        else:
            R["D3"] = dict(missing="no complete run with D3 diagnostics")
            W.skip("mech_d3_crossing", R["D3"]["missing"])
        # ---- conditional y law (final walkers; base contract, not a D key)
        yl = {m: {s: r for s, r in cr[m].items() if "X_final" in r and "Y_final" in r} for m in ("abf", "fr")}
        if any(yl.values()):
            Y = ylaw_compute({m: v for m, v in yl.items() if v}, model, h, int(args.n_boot))
            R["ylaw_EXPLORATORY"] = Y
            fig_ylaw(W, ctx, Y)
        else:
            R["ylaw_EXPLORATORY"] = dict(missing="no complete run with X_final / Y_final")
            W.skip("mech_ylaw", R["ylaw_EXPLORATORY"]["missing"])
        # ---- traces
        tr = {m: {s: r for s, r in dr[m].items() if "traces" in r} for m in ("abf", "fr")}
        if any(tr.values()):
            R["traces_EXPLORATORY"] = dict(seed=fig_traces(W, ctx, tr, model), walkers=[0, 1, 2, 3])
        else:
            R["traces_EXPLORATORY"] = dict(missing="no complete run with traces_y")
            W.skip("mech_ytraces", R["traces_EXPLORATORY"]["missing"])
        R["figures"] = W.entries
        R["figures_skipped"] = W.skipped
        doc["per_N"][str(N)] = R
    doc["wall_s"] = time.time() - t0
    jp = os.path.join(out_root, "results", "mechanism", exp, cname, "analysis", "mech_diagnostics.json")
    print(f"  analysed in {time.time() - t0:.1f} s (D1 / D2 / D3 figures drawn after every cell: shared y ranges)",
          flush=True)
    return doc, keep, defer, jp


def draw_deferred(defer):
    """Draw the per-cell D1 / D2(i) / D2(ii) / D3 figures with y ranges shared by every cell of one experiment at
    one N (panel-wise union of the cells' data extents)."""
    lims = {}
    for d in defer:
        lims.setdefault((d["kind"], d["key"]), []).append(d["ext"])
    lims = {k: merge_ext(v) for k, v in lims.items()}
    for d in defer:
        d["fn"](d["W"], d["ctx"], *d["args"], lims=lims[(d["kind"], d["key"])], **d["kw"])
    return {f"{k[0]}:{k[1][0]}:N{k[1][1]}": v for k, v in lims.items()}


def _region_block(p, key, regs):
    v, ci, pv = p[key], p[key + "_ci95"], p[key + "_p"]
    out = {key: {R: v[i] for i, R in enumerate(regs)}}
    out[key + "_ci95"] = ({R: [ci[0][i], ci[1][i]] for i, R in enumerate(regs)} if np.ndim(ci[0])
                          else {R: [math.nan, math.nan] for R in regs})
    out[key + "_p"] = {R: (pv[i] if isinstance(pv, list) else math.nan) for i, R in enumerate(regs)}
    return out


def jsonify_paired(p):
    """JSON block of one d2_paired result (scalars with _ci95 / _p; region quantities as {region: value})."""
    regs = list(DEV_REGIONS)
    out = dict(n_seeds=p["n_seeds"], p_calibrated=p["p_calibrated"], B_diff=p["B"], B_diff_ci95=p["B_ci95"],
               B_diff_p=p["B_p"])
    rb = _region_block(p, "dev", regs)
    out.update(dev_diff=rb["dev"], dev_diff_ci95=rb["dev_ci95"], dev_diff_p=rb["dev_p"])
    for k in ("B_did", "Bm_a", "Bm_b", "B_excess", "matched_coverage", "B_excess_a", "B_excess_b", "B_excess_did"):
        if k in p:
            out.update({k: p[k], k + "_ci95": p[k + "_ci95"], k + "_p": p[k + "_p"]})
    out["boot_finite_frac"] = {k[:-len("_boot_finite_frac")]: v for k, v in p.items() if k.endswith("_boot_finite_frac")}
    if "dev_excess" in p:
        out.update(_region_block(p, "dev_excess", regs))
    if not p["p_calibrated"]:
        out["note"] = f"{p['n_seeds']} seeds (< {SMALL_N}): bootstrap CIs / p are not calibrated, never interpreted"
    return out


def d2_json(d2, snap_u, names):
    """Array layout (compact): value / ci_lo / ci_hi [snapshot][class] or [snapshot][class][region]."""
    cls = list(names) + ["all"]
    out = dict(classes=cls, regions=list(DEV_REGIONS), snap_u=snap_u,
               layout=("value / ci_lo / ci_hi: [snapshot][class] for " + ", ".join(SCALAR_KEYS) + "; [snapshot][class]"
                       "[region] for " + ", ".join(REGION_KEYS) + "; profiles_end: [class][0.1-wide bin] at the end"))
    for m, per in d2.items():
        e = dict(n_seeds=[st["n_seeds"] for st in per])
        if min(e["n_seeds"]) < SMALL_N:
            e["note"] = (f"{min(e['n_seeds'])} seeds (< {SMALL_N}): the seed-bootstrap CIs are not calibrated (with 2 "
                         "seeds every finite resample repeats the sample), never interpreted")
        for k in SCALAR_KEYS + REGION_KEYS:
            e[k] = dict(value=np.array([st[k] for st in per]), ci_lo=np.array([st[k + "_lo"] for st in per]),
                        ci_hi=np.array([st[k + "_hi"] for st in per]))
        last = per[-1]
        if "dev_coarse" in last:
            e["profiles_end"] = dict(x=CENTRES_C, dev=last["dev_coarse"], dev_ci_lo=last["dev_coarse_lo"],
                                     dev_ci_hi=last["dev_coarse_hi"], var=last["var_coarse"],
                                     var_ci_lo=last["var_coarse_lo"], var_ci_hi=last["var_coarse_hi"],
                                     exact_var=last["etot_coarse"])
        out[m] = e
    return out


def rnd(o, sig=6):
    """JSON-safe object with floats rounded to sig significant digits (output size)."""
    if isinstance(o, dict):
        return {k: rnd(v, sig) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v, sig) for v in o]
    if isinstance(o, float):
        return float(f"{o:.{sig}g}")
    return o


def write_json(path, doc):
    guard(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(rnd(M.json_safe(doc)), fh, allow_nan=False, separators=(",", ":"))
    os.replace(path + ".tmp", path)


def guard(path):
    rp = os.path.realpath(os.path.abspath(path))
    for d in PROTECTED:
        dd = os.path.realpath(d)
        if rp == dd or rp.startswith(dd + os.sep):
            raise MechError(f"refusing to write {path}: inside the protected equal-budget tree {d}")


# =================================================================================================== synthesis
def g_of(S, N, group, metric):
    """(G_median, lo, hi, per_seed dict) of the summary's paired contrast, or None."""
    try:
        b = S["contrasts"][str(N)][group][metric]
    except (KeyError, TypeError):
        return None
    lo, hi = (b.get("G_ci95") or [None, None])
    return dict(G=fnum(b.get("G_median")), lo=fnum(lo), hi=fnum(hi), n=b.get("n"),
                per_seed={int(k): fnum(v) for k, v in (b.get("per_seed") or {}).items()})


G_METRICS = (("primary", "Ibar_F", "G(Ibar_F)"), ("secondary", "Ibar_Fp_stat", "G(Ibar_F'_stat)"),
             ("marginal", "Ibar_TV_half", "G(Ibar_TV_half)"))


def abf_scalar(S, N, key):
    try:
        b = S["per_N"][str(N)]["abf"][key]
    except (KeyError, TypeError):
        return None
    return dict(median=fnum(b.get("median")), q25=fnum(b.get("q25")), q75=fnum(b.get("q75")), n_inf=b.get("n_inf"),
                n=b.get("n"))


def validation_lookup(V, dyn, h):
    """(per-h block of a dynamics in the validation summary, h key) or (None, reason)."""
    if V is None:
        return None, "validation summary absent"
    d = (V.get("dynamics") or {}).get(dyn)
    if d is None:
        return None, f"dynamics {dyn} not in the validation summary"
    ph = d.get("per_h") or {}
    for hk in (f"{h:g}", f"{fnum((V.get('selection') or {}).get('h_common')):g}"):
        if hk in ph:
            return ph[hk], hk
    return None, f"no per-h block for h {h:g} ({sorted(ph)})"


def fig_s1(W, models, banners):
    fig = plt.figure(figsize=(16.0, 9.6), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0])
    xs = np.linspace(XMIN, XMAX, 361)
    ys = np.linspace(-1.0, 2.2, 321)
    XX, YY = np.meshgrid(xs, ys)
    norm = Normalize(0.0, 24.0)
    levels = np.arange(0.0, 24.01, 1.0)
    cs = None
    axes_top = []
    out = {}
    for i, (name, m) in enumerate(models):
        ax = fig.add_subplot(gs[0, i])
        V = GF.V_np(XX, YY, m)
        Vk = m["beta"] * (V - V.min())
        cs = ax.contourf(XX, YY, np.minimum(Vk, 24.0), levels=levels, cmap="magma_r", norm=norm)
        mu = GF.m_centre(xs, m)
        sd = np.sqrt(GF.cond_var_y(xs, m))
        ax.plot(xs, mu, color="white", lw=1.1)
        ax.plot(xs, mu + 2 * sd, color="#5ee0f0", lw=1.0, ls="--")
        ax.plot(xs, mu - 2 * sd, color="#5ee0f0", lw=1.0, ls="--")
        ax.set_title(f"({'abcd'[i]}) V(x, y): {name}", loc="left")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)
        axes_top.append(ax)
        bd = barrier_decomposition(m)
        out[name] = {k: v for k, v in bd.items() if not isinstance(v, np.ndarray)}
    fig.colorbar(cs, ax=axes_top, label="beta (V - min V) (kT), clipped at 24", shrink=0.9)
    ax = fig.add_subplot(gs[1, 0:2])
    bd0 = barrier_decomposition(models[0][1])
    ax.plot(bd0["x"], bd0["F_kT"], color=INK, lw=2.2)
    for name, m in models:
        bd = barrier_decomposition(m)
        ax.plot(bd["x"], bd["E_kT"], color=CELL_COLOR.get(name, INK2), lw=1.4, ls="--")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(-0.5, 20)
    ax.set_title("(e) common F*(x) and the energetic part min_y V(x, y) per model", loc="left")
    ax.set_xlabel("x")
    ax.set_ylabel("beta (F - F(-1)) (kT)")
    ax = fig.add_subplot(gs[1, 2:4])
    for i, (name, m) in enumerate(models):
        b = out[name]
        ax.bar(i, b["energetic_kT"], color="#9ecae1", width=0.6, edgecolor="white")
        ax.bar(i, b["entropic_kT"], bottom=b["energetic_kT"], color="#fd8d3c", width=0.6, edgecolor="white")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([f"{n}\nentropic {out[n]['entropic_fraction']:.1%}" for n, _ in models], fontsize=8.5)
    ax.set_ylim(0, 13.5)
    ax.set_title("(f) barrier decomposition F*(0) - F*(-1) = 11.47 kT", loc="left")
    ax.set_xlabel("model")
    ax.set_ylabel("barrier part (kT)")
    hs = [Line2D([], [], color="white", lw=2, label="E[Y | x] = m(x)", markeredgecolor=INK, marker="s", ms=5,
                 mfc="white"),
          Line2D([], [], color="#5ee0f0", lw=2, ls="--", label="m(x) +- 2 sd(Y | x)"),
          Line2D([], [], color=INK, lw=2.2, label="F*(x) (every model)")]
    hs += [Line2D([], [], color=CELL_COLOR.get(n, INK2), lw=1.6, ls="--", label=f"energetic part, {n}") for n, _ in models]
    hs += [Patch(facecolor="#9ecae1", label="energetic barrier"), Patch(facecolor="#fd8d3c", label="entropic barrier")]
    fig_legend(fig, hs, 5)
    title(fig, ["S1 matched-free-energy family: one F*(x), four 2-D potentials (Experiment I)",
                "Exact model quantities. alpha family V = H(x^2-1)^2 + ((1-alpha)/beta) log w + w^(2 alpha) y^2 / 2; "
                "shifted fibre V = F* + (y - m(x))^2 / 2. Energetic = min_y V along x; entropic = F* - energetic."],
          16.0, banners)
    W.save(fig, "mech_s1_models", "S1: exact F*, four V(x, y) on one colour scale with the conditional y band, barrier "
           "decomposition")
    return out


def fig_s2(W, exp, N, curves, xkey, banners, thresholds=None):
    names = [c for c in CELL_ORDER[exp] if c in curves and any(curves[c].get(m) for m in ("abf", "fr"))]
    absent = [c for c in CELL_ORDER[exp] if c not in names]
    if not names:
        W.skip(f"mech_s2_convergence_{exp}_N{N}_{xkey}", "no per-run metric curves (run_metrics caches of the cell "
               "analyses) for any cell")
        return
    width = max(9.0, 4.0 * len(names) + 1.0)
    fig, axs = new_fig(width, 7.6, 2, len(names), sharey="row")
    for j, c in enumerate(names):
        for i, (k, lab) in enumerate((("e_F", "e_F"), ("e_Fp_stat", "e_F'_stat (floor-free)"))):
            ax = axs[i, j]
            for meth, col in (("abf", C_ABF), ("fr", C_FR)):
                cv = curves[c].get(meth) or {}
                if not cv:
                    continue
                ref = cv[min(cv)]
                xg = ref["save_" + xkey]
                Ys = []
                for s, d in cv.items():
                    if k not in d:
                        continue
                    y = d[k]
                    if d["save_" + xkey].shape != xg.shape or not np.allclose(d["save_" + xkey], xg):
                        xs_ = d["save_" + xkey]
                        okx = xs_ > 0
                        y = np.interp(np.log(np.maximum(xg, 1e-300)), np.log(xs_[okx]), y[okx], left=np.nan,
                                      right=np.nan)
                    Ys.append(y)
                if not Ys:
                    continue
                band(ax, xg, stats_of(np.array(Ys)), col, logy=True)
            for v in ((thresholds or {}).get("e_F" if k == "e_F" else "e_Fp_stat") or []):
                ax.axhline(float(v), color=MUTED, lw=0.8, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            log_ticks(ax)
            ax.set_title(f"({'abcdefgh'[i * len(names) + j]}) {c}: {lab}", loc="left")
            ax.set_xlabel("budget fraction u" if xkey == "u" else "physical time t (t.u.)")
            ax.set_ylabel(lab)
    ns = {c: (len(curves[c].get("abf") or {}), len(curves[c].get("fr") or {})) for c in names}
    hs = [Line2D([], [], color=C_ABF, lw=2, label="ABF seed median"), Line2D([], [], color=C_FR, lw=2,
                                                                             label="ABF+FR seed median"),
          Patch(facecolor=INK2, alpha=0.18, label="interquartile range over seeds")]
    fig_legend(fig, hs, 3)
    title(fig, [f"S2 free-energy and floor-free mean-force convergence, {EXP_LABEL[exp]}, N = {N} (vs "
                f"{'budget fraction' if xkey == 'u' else 'physical time'})",
                "Frozen scorer curves (analytic F*, F*'); dotted: the frozen tau thresholds (e_F " +
                " / ".join(f"{float(v):g}" for v in (thresholds or {}).get("e_F", [])) + "; e_F'_stat " +
                " / ".join(f"{float(v):g}" for v in (thresholds or {}).get("e_Fp_stat", [])) +
                "). Seeds per cell (ABF, FR): " +
                ", ".join(f"{c} {a}/{f}" for c, (a, f) in ns.items()) +
                (f". MISSING (no curves): {', '.join(absent)}" if absent else "")], width, banners)
    W.save(fig, f"mech_s2_convergence_{exp}_N{N}_{xkey}", f"S2: e_F and e_F'_stat seed median + IQR, ABF vs FR, "
           f"{exp} N {N}, x = {xkey}")


def lr(g):
    """G = (FR - ABF)/ABF -> log(FR/ABF) = log1p(G): the log-ratio axis of the gain figures (a single seed with FR 75x
    worse otherwise flattens every median on a linear axis; added 2026-10-10 after reading the production S3)."""
    g = fnum(g)
    return math.log1p(g) if math.isfinite(g) and g > -1.0 else float("nan")


def logratio_axis(ax, lab):
    """Ticks of a log(FR/ABF) axis labelled as G in %."""
    lo, hi = ax.get_ylim()
    cand = [-0.95, -0.9, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 24.0, 49.0, 99.0]
    t0 = [math.log1p(c) for c in cand if lo <= math.log1p(c) <= hi]
    gap = 0.09 * (hi - lo)                       # thin: labels at least 9 % of the span apart, 0 always kept
    t = [0.0] if lo <= 0.0 <= hi else []
    for v in sorted(t0, key=lambda v: abs(v)):
        if all(abs(v - u) >= gap for u in t):
            t.append(v)
    t = sorted(t)
    ax.set_yticks(t)
    ax.set_yticklabels([f"{100 * math.expm1(v):+.0f} %" for v in t])
    ax.set_ylabel(f"{lab} = (FR - ABF)/ABF (log-ratio axis)")


def seed_strip(ax, x0, vals, color, dx=0.004, spread=0.0025, marker="o", seed=0):
    """Per-seed values as a jittered strip just right of a summary marker."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    j = np.random.default_rng(seed).uniform(-spread, spread, v.size)
    ax.plot(x0 + dx + j, v, ls="none", marker=marker, ms=2.6, color=color, alpha=0.45, mew=0)


def fig_s3(W, summ, ent, Ns, banners):
    fig, axs = new_fig(15.5, 5.9, 1, 3, sharex=True)
    offs = {N: o for N, o in zip(Ns, np.linspace(-0.014, 0.014, max(1, len(Ns))))}
    alpha_of = {"alpha1": 1.0, "alpha0.5": 0.5, "alpha0": 0.0}
    rows, missing = {}, []
    for k, (grp, met, lab) in enumerate(G_METRICS):
        ax = axs[0, k]
        for N in Ns:
            line = []
            for c in CELL_ORDER["matched_free_energy"]:
                S = summ.get(("matched_free_energy", c))
                g = g_of(S, N, grp, met) if (S is not None and c in ent) else None
                if g is None:
                    if k == 0:
                        missing.append(f"{c} N{N}")
                    continue
                x0 = ent[c] + offs[N] + (0.045 if c == "shift" else 0.0)
                seed_strip(ax, x0, [lr(v) for v in g["per_seed"].values()], N_COLOR.get(N, INK2), seed=N)
                errpt(ax, x0, lr(g["G"]), lr(g["lo"]), lr(g["hi"]), N_COLOR.get(N, INK2), marker="D" if c == "shift" else
                      N_MARKER.get(N, "o"), filled=(c != "shift"))
                if c in alpha_of and math.isfinite(lr(g["G"])):
                    line.append((x0, lr(g["G"])))
                rows.setdefault(met, []).append(dict(cell=c, N=N, entropic_fraction=ent[c],
                                                     **{kk: g[kk] for kk in ("G", "lo", "hi", "n")}))
            if len(line) >= 2:
                line.sort()
                ax.plot([a for a, _ in line], [b for _, b in line], color=N_COLOR.get(N, INK2), lw=0.9, alpha=0.6)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"({'abc'[k]}) {lab}", loc="left")
        ax.set_xlabel("entropic fraction of the F* barrier (shift: offset right of 0)")
        logratio_axis(ax, lab)
        sec = ax.secondary_xaxis("top")
        sec.set_xticks([ent[c] for c in alpha_of if c in ent] + ([ent["shift"] + 0.045] if "shift" in ent else []))
        sec.set_xticklabels([f"alpha {alpha_of[c]:g}" for c in alpha_of if c in ent] + (["shift"] if "shift" in ent
                                                                                          else []), fontsize=8)
        sec.tick_params(length=2)
    hs = [Line2D([], [], color=N_COLOR.get(N, INK2), marker=N_MARKER.get(N, "o"), ls="-", lw=0.9, ms=7,
                 label=f"alpha family, N = {N}: median, 95 % CI") for N in Ns]
    hs += [Line2D([], [], color=N_COLOR.get(N, INK2), marker="D", mfc="white", ls="none", ms=7,
                  label=f"shifted fibre (entropic 0), N = {N}") for N in Ns]
    hs += [Line2D([], [], color=INK2, marker="o", ls="none", ms=3, alpha=0.5, label="per seed (jittered strip)")]
    fig_legend(fig, hs, 4)
    title(fig, ["S3 FR relative improvement vs the entropic fraction of the barrier (Experiment I)",
                "alpha = 0 / 0.5 / 1 at entropic fraction 0 / 0.151 / 0.302 (identical F*); G median with the summary's "
                "paired 10 000-resample seed-bootstrap 95 % CI and the per-seed values. Negative = FR better." +
                (f" MISSING (no summary G): {', '.join(missing)}." if missing else "")], 15.5, banners)
    W.save(fig, "mech_s3_gain_vs_entropic_fraction", "S3: G(Ibar_F), G(Ibar_F'_stat), G(Ibar_TV_half) vs entropic "
           "fraction; shifted fibre separate")
    return dict(rows=rows, missing=missing)


def fig_s3b(W, cc, src, banners, keep_row=None):
    keep_row = keep_row or (lambda e, r: True)
    exps = [e for e in EXP_ORDER if e in (cc.get("experiments") or {})]
    if not any(keep_row(e, r) for e in exps for m in (cc["experiments"][e].get("metrics") or {}).values()
               for r in m.get("contrasts", [])):
        W.skip("mech_s3b_cross_cell_contrasts", "no cross-cell contrast among the analysed cells and N")
        return
    fig, axs = new_fig(15.5, 5.6, 1, 3)
    for k, (grp, met, lab) in enumerate(G_METRICS):
        ax = axs[0, k]
        rows = []
        for e in exps:
            for r in ((cc["experiments"][e].get("metrics") or {}).get(met) or {}).get("contrasts", []):
                if keep_row(e, r):
                    rows.append(r)
        for i, r in enumerate(rows):
            col = N_COLOR.get(int(r["N"]), INK2)
            lo, hi = (r.get("ratio_ci95") or [None, None])
            errpt(ax, i, r.get("ratio"), lo, hi, col, marker=N_MARKER.get(int(r["N"]), "o"),
                  filled=(r.get("kind") == "primary"))
        ax.axhspan(0.90, 1.11, color="#d9f0d3", alpha=0.6, lw=0)
        ax.axhline(1.0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"{r['id']} N{r['N']}" for r in rows], rotation=90, fontsize=7.5)
        ax.set_yscale("log")
        ratio_ticks(ax)
        ax.set_title(f"({'abc'[k]}) exp(Delta) on {met}", loc="left")
        ax.set_xlabel("contrast and N")
        ax.set_ylabel("ratio of FR/ABF ratios (a vs b)")
    Ns = sorted({int(r["N"]) for e in exps for m in (cc["experiments"][e].get("metrics") or {}).values()
                 for r in m.get("contrasts", []) if keep_row(e, r)}, reverse=True)
    hs = [Line2D([], [], color=N_COLOR.get(N, INK2), marker=N_MARKER.get(N, "o"), ls="none", ms=7,
                 label=f"N = {N} (filled: primary, open: trend)") for N in Ns]
    hs += [Patch(facecolor="#d9f0d3", label="equivalence band [0.90, 1.11]")]
    fig_legend(fig, hs, 4)
    title(fig, ["S3b preregistered cross-cell contrasts exp(Delta_ab) (< 1: cell a has the larger FR gain)",
                "C1 alpha0 vs alpha1, C2 shift vs alpha1, C3 shift vs alpha0, C4 alpha0.5 vs alpha1; D1 lam0.1, T1 "
                "lam0.25, T2 lam0.5 vs lam1. 95 % paired seed-bootstrap CIs; Holm and verdicts are in the source.",
                f"Source: {src}."], 15.5, banners)
    W.save(fig, "mech_s3b_cross_cell_contrasts", "S3b: cross-cell contrasts exp(Delta) with CIs and the equivalence band")


def fig_s4(W, models, V, h, banners):
    fig, axs = new_fig(16.0, 5.6, 1, 3)
    ax = axs[0, 0]
    xs = np.linspace(-0.6, 0.6, 1201)
    meas_any = False
    for name, m in models:
        ax.plot(xs, GF.cond_var_f(xs, m), color=CELL_COLOR.get(name, INK2), lw=1.6,
                ls=("-" if name != "shift" else (0, (4, 2))))
        blk, _ = validation_lookup(V, dynamics_name(m), h)
        fv = ((blk or {}).get("em_flat") or {}).get("force_var_profile")
        if fv:
            meas_any = True
            ax.errorbar(fv["x"], fv["measured"], yerr=2 * np.nan_to_num(np.asarray(fv["se"], float)), fmt="o", ms=3,
                        color=CELL_COLOR.get(name, INK2), mfc="white", lw=0.8)
    ax.set_title("(a) exact Var(f | x)" + (" and validation measurement" if meas_any else ""), loc="left")
    ax.set_xlabel("x")
    ax.set_ylabel("Var(f | x) (reduced force^2)")
    dyns = [d for d in GF.DYNAMICS_ORDER]
    for k, q in enumerate(("y", "f")):
        ax = axs[0, k + 1]
        meas = False
        for i, dn in enumerate(dyns):
            m = dyn_model(dn)
            blk, _ = validation_lookup(V, dn, h)
            wins = ((blk or {}).get("acf") or {}).get("windows") or {}
            for j, c in enumerate(ACF_X):
                xo = i + (j - 1) * 0.22
                pred = predicted_tau(m, c, q)
                if pred is not None and math.isfinite(pred):
                    ax.plot([xo], [pred], marker=PRED_MK[j], color=C_PRED, mew=1.6, ls="none", ms=8, zorder=6)
                w = wins.get(f"{c:g}") or {}
                val = fnum(w.get(f"tau_int_{q}_5tau"))
                if math.isfinite(val) and val > 0:
                    meas = True
                    se = fnum(w.get(f"tau_int_{q}_5tau_se"))
                    ax.errorbar([xo], [val], yerr=[[min(2 * se, 0.99 * val)], [2 * se]] if math.isfinite(se) else None,
                                fmt=MEAS_MK[j], color=INK if w.get(f"resolved_{q}", w.get("resolved", True))
                                else MUTED, ms=5, capsize=2, zorder=5)
        ax.set_yscale("log")
        ax.set_xticks(range(len(dyns)))
        ax.set_xticklabels(dyns, rotation=30, fontsize=8)
        ax.set_title(f"({'bc'[k]}) integrated conditional ACF time of {q}" + ("" if meas else
                                                                               ": frozen-x prediction only"), loc="left")
        ax.set_xlabel("dynamics")
        ax.set_ylabel(f"tau_int,{q} to 5 tau (t.u.)")
    hs = [Line2D([], [], color=CELL_COLOR.get(n, INK2), lw=2, label=f"(a) exact Var(f|x), {n}") for n, _ in models]
    hs += [Line2D([], [], marker="o", color=INK2, mfc="white", ls="none", ms=5,
                  label="(a) measured Var(f|x) (validation EM-flat chains, +-2 se), in the model's colour")]
    hs += [Line2D([], [], marker=mk, color=C_PRED, mew=1.6, ls="none", ms=8, label=f"(b, c) frozen-x OU prediction, "
                  f"x = {c:g}") for mk, c in zip(PRED_MK, ACF_X)]
    hs += [Line2D([], [], marker=mk, color=INK, ls="none", ms=5, label=f"(b, c) measured, x = {c:g} (+-2 se)")
           for mk, c in zip(MEAS_MK, ACF_X)]
    hs += [Line2D([], [], marker="o", color=MUTED, ls="none", ms=6, label="(b, c) measured, not resolved at the first "
                  "lag")]
    fig_legend(fig, hs, 4)
    vnote = ("validation summary present" if V is not None else
             "validation summary ABSENT (results/mechanism/validation/summary.json): no measured values shown")
    title(fig, ["S4 conditional force variance and conditional relaxation per dynamics",
                f"{vnote}. Frozen-x OU: tau_y = 1/(lam k(x)), tau_f = tau_y / 2 (alpha family) or tau_y (shift), "
                "integrated to 5 tau over the window x0 +- 0.02; no tau_f where f does not depend on y (alpha = 0; "
                "x = -1, where w' = 0). alpha1, shift and the lambda dynamics share one Var(f | x) (curves overlap)."],
          16.0, banners)
    W.save(fig, "mech_s4_force_variance_relaxation", "S4: exact (and measured) Var(f|x); conditional ACF times vs "
           "frozen-x OU predictions")


MEAS_MK = ("o", "s", "^")                       # S4 (b, c): measured ACF times per window x in ACF_X
PRED_MK = ("x", "+", "1")                       # S4 (b, c): frozen-x OU predictions (line markers, never filled)
C_PRED = "#b15928"


def dyn_model(name):
    phys = dict(GF.PHYS)
    if name == "shift":
        return GF.make_model("shift", 1.0, 1.0, 1.0, **phys)
    if name.startswith("alpha"):
        return GF.make_model("alpha", float(name[5:]), 1.0, 1.0, **phys)
    return GF.make_model("alpha", 1.0, 1.0, float(name[3:]), **phys)


def predicted_tau(m, c, q):
    """Frozen-x OU integrated ACF time to 5 tau in the window c +- ACF_HALF_WIDTH (the validation's prediction)."""
    if q == "f" and m["variant"] == "alpha" and m["alpha"] == 0.0:
        return None
    if q == "f" and float(np.max(GF.cond_var_f(np.linspace(c - GF.ACF_HALF_WIDTH, c + GF.ACF_HALF_WIDTH, 41), m))) \
            < 1e-12:
        return None                                  # f does not depend on y in this window (the wells: w' = 0)
    ou = GF.ou_prediction(m, c)
    tau = ou["tau_y_centre"] if q == "y" else ou["tau_f_centre"]
    tt = np.linspace(0.0, 5.0 * tau, 20001)
    return GF.tau_integrated(tt, ou["rho_" + q](tt), 5.0 * tau)


def fig_s5(W, summ, d2keep, lam_cells, Ns, banners):
    """S5 (a-c): G of Ibar_F / Ibar_F'_stat / Ibar_TV_half vs lambda per N; (d) young-clone deposit share in the gate."""
    fig, axs = new_fig(19.0, 5.8, 1, 4)
    lamv = {c: float(c[3:]) for c in lam_cells}
    lams = list(lamv.values())
    rows, missing = {}, []
    nshift = {N: f for N, f in zip(Ns, np.linspace(-0.05, 0.05, max(1, len(Ns))))}
    for k, (grp, met, lab) in enumerate(G_METRICS):
        ax = axs[0, k]
        for N in Ns:
            xs, gs = [], []
            for c in lam_cells:
                S = summ.get(("conditional_relaxation", c))
                g = g_of(S, N, grp, met) if S else None
                if g is None:
                    if k == 0:
                        missing.append(f"{c} N{N}")
                    continue
                x0 = lamv[c] * (1.0 + nshift[N])
                vals = np.array([lr(v) for v in g["per_seed"].values()], float)
                vals = vals[np.isfinite(vals)]
                jit = np.random.default_rng(N).uniform(-0.012, 0.012, vals.size)
                ax.plot(x0 * (1.035 + jit), vals, ls="none", marker="o", ms=2.6, color=N_COLOR.get(N, INK2),
                        alpha=0.45, mew=0)
                errpt(ax, x0, lr(g["G"]), lr(g["lo"]), lr(g["hi"]), N_COLOR.get(N, INK2), marker=N_MARKER.get(N, "o"))
                xs.append(x0)
                gs.append(lr(g["G"]))
                rows.setdefault(met, []).append(dict(cell=c, lam=lamv[c], N=N,
                                                     **{kk: g[kk] for kk in ("G", "lo", "hi", "n")}))
            if xs:
                o = np.argsort(xs)
                ax.plot(np.array(xs)[o], np.array(gs)[o], color=N_COLOR.get(N, INK2), lw=1.0, alpha=0.7)
        ax.axhline(0, color=MUTED, lw=0.8)
        lam_axis(ax, lams)
        ax.set_title(f"({'abc'[k]}) {lab} vs lambda", loc="left")
        ax.set_xlabel("transverse mobility lambda")
        logratio_axis(ax, lab)
    ax = axs[0, 3]
    for N in Ns:
        for c in lam_cells:
            fr = (d2keep.get(("conditional_relaxation", c, N, "fr_frac")) or {})
            if not fr:
                continue
            st = fr["recent_lt_0p1_gate_end"]
            errpt(ax, lamv[c] * (1.0 + nshift[N]), st["median"], st["q25"], st["q75"], N_COLOR.get(N, INK2),
                  marker=N_MARKER.get(N, "o"))
    lam_axis(ax, lams)
    ax.set_title("(d) gate deposits with clone age < 0.1 t.u. (end)", loc="left")
    ax.set_xlabel("transverse mobility lambda")
    ax.set_ylabel("fraction of gate deposits (seed median, IQR)")
    hs = [Line2D([], [], color=N_COLOR.get(N, INK2), marker=N_MARKER.get(N, "o"), ls="-", lw=1.0, ms=7,
                 label=f"N = {N}: median with 95 % CI (a-c) / IQR (d)") for N in Ns]
    hs += [Line2D([], [], color=INK2, marker="o", ls="none", ms=3, alpha=0.5, label="per seed (a-c)")]
    fig_legend(fig, hs, 3)
    title(fig, ["S5 Experiment II: FR improvement vs transverse mobility lambda (V_1, identical Gibbs law)",
                "G = (FR - ABF)/ABF per seed; median with the summary's paired 10 000-resample seed-bootstrap 95 % CI. "
                "The D2 recently-cloned mean-force bias vs lambda: mech_s5b_d2_conditional_relaxation." +
                (f" MISSING (no summary G): {', '.join(missing)}." if missing else "")], 19.0, banners)
    W.save(fig, "mech_s5_lambda", "S5: G(Ibar_F / F'_stat / TV_half) vs lambda per N, and the young-clone share of "
           "the gate deposits")
    return dict(rows=rows, missing=missing)


D2_SHOW = (0, 1, 2, 3, 4)
D2_SERIES = (0, 1, 2, 3, 4, "fr_all", "abf")


def _series_style(k):
    if k == "abf":
        return C_ABF, "--"
    if k == "fr_all":
        return C_FR, "-"
    return CLASS_COLOR[k], "-"


def fig_d2_cells(W, exp, cells_present, d2keep, d2c, Ns, banners):
    """S5b (Experiment II) / S5c (Experiment I): the D2 class-wise mean-force bias across the cells of one experiment
    at each N.  Columns: (1) B_c (FR classes, ABF) and (2) its paired change B_c(cell) - B_c(ref) -- the quantities the
    plan names (sections 4 and 7; preregistered for Experiment II); (3) the matched-position excess over ABF and (4)
    its paired change from the reference cell (difference in differences) -- analysis-defined, EXPLORATORY.  S5c draws
    the shifted fibre in its own narrow panel next to each alpha-family panel, on its own y scale (its B_c is an order
    of magnitude above the alpha family's and it is not on the alpha continuum)."""
    ref = REF_CELL[exp]
    prereg = exp == "conditional_relaxation"
    side = [c for c in cells_present if c == "shift"] if not prereg else []
    main_cells = [c for c in cells_present if c not in side]
    if not main_cells:
        side, main_cells = [], list(cells_present)
    if prereg:
        xpos = {c: float(c[3:]) for c in cells_present}
        logx = True
    else:
        xpos = {c: float(i) for i, c in enumerate(main_cells)}
        xpos.update({c: 0.0 for c in side})
        logx = False
    span = 0.07 if logx else 0.34
    offs = {k: o for k, o in zip(D2_SERIES, np.linspace(-span, span, len(D2_SERIES)))}

    def xat(c, k):
        return xpos[c] * (1.0 + offs[k]) if logx else xpos[c] + offs[k]

    def setx(ax, cl):
        if logx:
            lam_axis(ax, [xpos[c] for c in cl])
            ax.set_xlabel("transverse mobility lambda")
        else:
            ax.set_xticks([xpos[c] for c in cl])
            ax.set_xticklabels(cl)
            ax.set_xlim(min(xpos[c] for c in cl) - 0.6, max(xpos[c] for c in cl) + 0.6)
            ax.set_xlabel("model (Experiment I cell)" if cl is main_cells else "cell")

    def nm(k):
        return AGE_DEFAULT[k] if isinstance(k, int) else "all"

    def get_B(c, N, k):
        arm = "abf" if k == "abf" else "fr"
        st = (d2keep.get((exp, c, N, arm)) or {}).get("end")
        if st is None:
            return None
        i = ALL if k in ("abf", "fr_all") else k
        return fnum(st["B"][i]), fnum(st["B_lo"][i]), fnum(st["B_hi"][i])

    def get_cc(c, N, k, key):
        blk = ((d2c or {}).get(exp) or {}).get(f"{c}_vs_{ref}_N{N}") or {}
        e = blk.get("abf_all") if k == "abf" else blk.get(nm(k))
        if not e or key not in e:
            return None
        lo, hi = e.get(key + "_ci95") or [math.nan, math.nan]
        return fnum(e[key]), fnum(lo), fnum(hi)

    def get_ex(c, N, k):
        e = (d2keep.get((exp, c, N, "fr_vs_abf")) or {}).get(nm(k))
        if not e or "B_excess" not in e:
            return None
        lo, hi = e.get("B_excess_ci95") or [math.nan, math.nan]
        return fnum(e["B_excess"]), fnum(lo), fnum(hi)

    def draw(ax, getter, series, cl, ref_zero=False):
        """Points with CIs of the cells cl; a line joins the cells of one panel (the side panel has one cell)."""
        for k in series:
            col, ls = _series_style(k)
            line = []
            for c in cl:
                if ref_zero and c == ref:
                    if any(getter(cc, k) is not None for cc in cl if cc != ref):
                        line.append((xat(c, k), 0.0))
                    continue
                g = getter(c, k)
                if g is None:
                    continue
                v, lo, hi = g
                errpt(ax, xat(c, k), v, lo, hi, col, ms=5)
                if math.isfinite(v):
                    line.append((xat(c, k), v))
            if len(line) >= 2:
                ax.plot(*zip(*sorted(line)), color=col, lw=0.8, alpha=0.6, ls=ls)

    nr = max(1, len(Ns))
    if side:
        fig = plt.figure(figsize=(24.0, 5.2 * nr + 2.2), layout="constrained")
        gs = fig.add_gridspec(nr, 8, width_ratios=[3.3, 1.0] * 4)
        axs = np.array([[fig.add_subplot(gs[r, 2 * j]) for j in range(4)] for r in range(nr)])
        sax = np.array([[fig.add_subplot(gs[r, 2 * j + 1]) for j in range(4)] for r in range(nr)])
    else:
        fig, axs = new_fig(21.0, 4.9 * nr + 2.2, nr, 4)
        sax = None
    missing = []

    def panel(r, j, getter, series, title_txt, ylab, ref_zero=False, zero_line=True):
        ax = axs[r, j]
        draw(ax, getter, series, main_cells, ref_zero=ref_zero)
        if zero_line:
            ax.axhline(0, color=MUTED, lw=0.8)
        setx(ax, main_cells)
        ax.set_title(title_txt, loc="left")
        ax.set_ylabel(ylab)
        if sax is not None:
            a2 = sax[r, j]
            draw(a2, getter, series, side)
            if zero_line:
                a2.axhline(0, color=MUTED, lw=0.8)
            setx(a2, side)
            a2.set_title("shift", loc="left")
            a2.set_ylabel("same quantity, own y scale")
    tagp = "[preregistered]" if prereg else "[EXPLORATORY]"
    for r, N in enumerate(Ns):
        for c in cells_present:
            for arm in ("fr", "abf"):
                if (d2keep.get((exp, c, N, arm)) or {}).get("end") is None:
                    missing.append(f"{c} N{N} {arm}")
            if c != ref and (((d2c or {}).get(exp) or {}).get(f"{c}_vs_{ref}_N{N}") or {}).get("missing"):
                missing.append(f"{c}_vs_{ref} N{N}")
        # (1) B_c per class, FR classes + ABF
        panel(r, 0, lambda c, k: get_B(c, N, k), list(D2_SHOW) + ["abf"],
              f"({'aeim'[r]}) N = {N}: B_c at the end\n{tagp}", "B_c (reduced force)")
        # (2) paired change of B_c from the reference cell (FR classes, FR all, ABF)
        panel(r, 1, lambda c, k: get_cc(c, N, k, "B_diff"), list(D2_SERIES),
              f"({'bfjn'[r]}) N = {N}: B_c(cell) - B_c({ref})\npaired seeds {tagp}",
              f"B_c - B_c({ref}) (reduced force)", ref_zero=True)
        # (3) matched-position excess over ABF -- EXPLORATORY
        panel(r, 2, lambda c, k: get_ex(c, N, k), list(D2_SHOW) + ["fr_all"],
              f"({'cgko'[r]}) N = {N}: EXPLORATORY\nexcess over ABF at the class's positions",
              "B_c(FR) - B_ABF|c (reduced force)")
        # (4) change of the excess from the reference cell -- EXPLORATORY
        panel(r, 3, lambda c, k: get_cc(c, N, k, "B_excess_did"), list(D2_SHOW) + ["fr_all"],
              f"({'dhlp'[r]}) N = {N}: EXPLORATORY\nchange of that excess from {ref}",
              f"excess(cell) - excess({ref}) (reduced force)", ref_zero=True)
    hs = [Line2D([], [], color=CLASS_COLOR[c], marker="o", ls="-", lw=0.8, ms=5,
                 label=f"FR clone age {AGE_DEFAULT[c]} t.u." if c < 4 else "FR never cloned") for c in D2_SHOW]
    hs += [Line2D([], [], color=C_FR, marker="o", ls="-", lw=0.8, ms=5, label="FR, all deposits (columns 2-4)"),
           Line2D([], [], color=C_ABF, marker="o", ls="--", lw=0.8, ms=5, label="ABF, all deposits (reference level)"),
           Line2D([], [], color=INK2, lw=1.2, label="error bars: 95 % seed-bootstrap CI (basic, debiased)")]
    fig_legend(fig, hs, 4)
    if prereg:
        head = ("S5b Experiment II: D2 class-wise mean-force bias vs lambda. Columns 1-2: the plan's readout (sections 4 "
                "and 7: B_c and its paired change from lambda = 1); columns 3-4: EXPLORATORY, analysis-defined")
    else:
        head = ("S5c Experiment I: D2 class-wise mean-force bias across the matched-free-energy models (all columns "
                "EXPLORATORY); the shifted fibre in its own narrow panel on its own y scale (not on the alpha "
                "continuum)")
    title(fig, [head,
                "B_c^2 = sum_j n_cj [(m_cj - <F*'>_j)^2 - V_cj] / sum_j n_cj over the eval bins (seed-cluster debiased). "
                "Columns 3-4 (not in the plan): B_ABF|c = ABF's debiased error weighted by the FR class's deposit counts "
                f"(same bins); column 4 = [B_c(FR) - B_ABF|c]_cell - [same]_{ref}. Paired over the common seeds; 0 at "
                f"{ref} by construction. Per-class p values are unadjusted (Holm within each readout family in "
                "mech_synthesis.json)." + (f" MISSING: {', '.join(sorted(set(missing)))}." if missing else "")],
          fig.get_figwidth(), banners + ([BANNER_EXPLORATORY.replace("this quantity is not preregistered",
                                                       "columns 3-4 (matched-position excess) are analysis-defined, "
                                                       "not preregistered")] if prereg else
                           [BANNER_EXPLORATORY.replace("this quantity is not preregistered",
                                                       "the D2 contrast across alpha is not a preregistered test")]))
    name = "mech_s5b_d2_conditional_relaxation" if prereg else "mech_s5c_d2_matched_free_energy"
    W.save(fig, name, ("S5b: D2 B_c and its paired change from lam1 (plan readout); EXPLORATORY matched-position excess "
                       "over ABF and its change, per N") if prereg else
           ("S5c EXPLORATORY: D2 B_c, its paired change from alpha1, matched-position excess over ABF and its "
            "change, per N (alpha family; shifted fibre on its own scale)"))
    return dict(missing=sorted(set(missing)))


def fig_s6(W, summ, cells, banners):
    fig, axs = new_fig(10.5, 8.0, 1, 1)
    ax = axs[0, 0]
    pts = []
    hs = []
    seen = set()
    missing = [f"{e}/{c}" for (e, c) in cells if (e, c) not in summ]
    for (exp, c), S in summ.items():
        if exp == "conditional_relaxation" and c == "lam1" and ("matched_free_energy", "alpha1") in summ:
            continue                                     # identical runs (alias of alpha1): plotted once
        for N in [int(n) for n in cells[(exp, c)]["N_ladder"]]:
            gF, gT = g_of(S, N, "primary", "Ibar_F"), g_of(S, N, "marginal", "Ibar_TV_half")
            if not gF or not gT:
                missing.append(f"{exp}/{c} N{N}")
                continue
            col = CELL_COLOR.get(c, INK2)
            com = sorted(set(gF["per_seed"]) & set(gT["per_seed"]))
            ax.plot([gT["per_seed"][s] for s in com], [gF["per_seed"][s] for s in com], ls="none",
                    marker=N_MARKER.get(N, "o"), ms=3.5, color=col, alpha=0.35)
            xe = [[max(gT["G"] - gT["lo"], 0.0)], [max(gT["hi"] - gT["G"], 0.0)]] \
                if math.isfinite(gT["lo"]) and math.isfinite(gT["hi"]) else None
            ye = [[max(gF["G"] - gF["lo"], 0.0)], [max(gF["hi"] - gF["G"], 0.0)]] \
                if math.isfinite(gF["lo"]) and math.isfinite(gF["hi"]) else None
            ax.errorbar([gT["G"]], [gF["G"]], xerr=xe, yerr=ye, fmt=N_MARKER.get(N, "o"), ms=10, color=col, mec=INK,
                        mew=0.8, ecolor=col, elinewidth=1.3, capsize=3, zorder=5)
            pts.append(dict(experiment=exp, cell=c, N=N, G_TV_half=gT["G"], G_TV_half_ci95=[gT["lo"], gT["hi"]],
                            G_F=gF["G"], G_F_ci95=[gF["lo"], gF["hi"]], n=len(com)))
            if c not in seen:
                hs.append(Line2D([], [], color=col, marker="o", ls="none", ms=8, label=c))
                seen.add(c)
    lim = [-1.0, 1.0]
    if pts:
        v = [p["G_TV_half"] for p in pts] + [p["G_F"] for p in pts] + [x for p in pts for x in
                                                                        p["G_TV_half_ci95"] + p["G_F_ci95"]]
        v = [x for x in v if math.isfinite(x)]
        if v:
            lim = [min(-0.1, min(v) - 0.1), max(0.1, max(v) + 0.1)]
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("G(Ibar_TV_half): marginal improvement (negative = FR better)")
    ax.set_ylabel("G(Ibar_F): free-energy improvement (negative = FR better)")
    ax.set_title("marginal vs free-energy improvement, all cells and N", loc="left")
    hs += [Line2D([], [], color=INK2, marker=N_MARKER[N], ls="none", ms=8, label=f"N = {N}") for N in (2048, 512, 128)]
    hs += [Line2D([], [], color=INK2, marker="o", ls="-", lw=1.3, ms=10, mec=INK,
                  label="cell median (large) with 95 % CI of each G (crosshair)"),
           Line2D([], [], color=INK2, marker="o", ls="none", ms=3.5, alpha=0.5, label="per seed (small)"),
           Line2D([], [], color=MUTED, ls="--", lw=1, label="equal improvement")]
    fig_legend(fig, hs, 5)
    title(fig, ["S6 reaction-coordinate marginal improvement vs free-energy improvement",
                "Paired per-seed G from the cell summaries; crosshairs: the summaries' 10 000-resample paired seed-"
                "bootstrap 95 % CIs of the median G on each axis. Experiment II lam1 is the alpha1 run set (plotted "
                "once)." +
                (f" MISSING: {', '.join(missing)}." if missing else "")],
          10.5, banners)
    W.save(fig, "mech_s6_marginal_vs_free_energy", "S6: per-seed and median G(Ibar_TV_half) vs G(Ibar_F), all cells "
           "and N")
    return dict(points=pts, missing=missing)


def fig_m1(W, cov, banners):
    fig, axs = new_fig(17.0, 5.9, 1, 4)
    specs = (("var_f_gate", "(a) gate-mean conditional force variance", "mean Var(f|x), |x| <= 0.5", False),
             ("tau_TV_half", "(b) ABF establishment tau(TV_half <= 0.1)", "ABF median tau (t.u.)", True),
             ("est_cum", "(c) ABF cumulative establishment", "ABF median t (t.u.)", True),
             ("tau_f", "(d) conditional relaxation time of f at x = -0.21", "tau_int,f (t.u.)", True))
    Nlist = sorted({r["N"] for r in cov}, reverse=True)
    nidx = {N: i - (len(Nlist) - 1) / 2.0 for i, N in enumerate(Nlist)}
    undefined = {}
    for k, (key, t, xl, logx) in enumerate(specs):
        ax = axs[0, k]
        pts = []
        for r in cov:
            x = fnum(r.get(key))
            cens = bool(r.get(key + "_censored"))
            if cens:
                x = fnum(r.get("T_N"))
            if not math.isfinite(x):
                if str(r.get(key + "_source", "")).startswith("undefined"):
                    undefined.setdefault(key, set()).add(r["cell"])
                continue
            pts.append((r, x, cens))
        xs = [x for _, x, _ in pts]
        rng = (max(xs) - min(xs)) if xs else 1.0
        for r, x, cens in pts:
            j = nidx.get(r["N"], 0.0)
            xj = x * 10 ** (0.025 * j) if (logx and x > 0) else x + 0.012 * (rng if rng > 0 else 1.0) * j
            col = CELL_COLOR.get(r["cell"], INK2)
            mk = ">" if cens else N_MARKER.get(r["N"], "o")
            errpt(ax, xj, r["G"], r["lo"], r["hi"], col, marker=mk, filled=not cens)
            if key in ("tau_TV_half", "est_cum") and not cens:
                q25, q75 = fnum(r.get(key + "_q25")), fnum(r.get(key + "_q75"))
                if math.isfinite(q25) and math.isfinite(q75):
                    ax.plot([q25, q75], [r["G"], r["G"]], color=col, lw=0.8, alpha=0.6)
        if logx and any(x > 0 for x in xs):
            ax.set_xscale("log")
            pxs = [x for x in xs if x > 0] + [v for r, _, cens in pts if not cens for v in
                                               (fnum(r.get(key + "_q25")), fnum(r.get(key + "_q75")))
                                               if math.isfinite(v) and v > 0]
            ax.set_xlim(min(pxs) / 1.4, max(pxs) * 1.4)
            log_ticks(ax, "x")
        ax.axhline(0, color=MUTED, lw=0.8)
        src = {r.get(key + "_source") for r in cov if r.get(key + "_source")
               and not str(r.get(key + "_source")).startswith("undefined")}
        und = sorted(undefined.get(key, ()))
        ax.set_title(t, loc="left")
        ax.set_xlabel(xl + (f"\nsource: {'; '.join(sorted(src))}" if src else "")
                      + (f"\nnot defined for {', '.join(und)} (f independent of y)" if und else ""), fontsize=8.5)
        ax.set_ylabel("G(Ibar_F) median, 95 % CI")
    cells = sorted({r["cell"] for r in cov}, key=lambda c: list(CELL_COLOR).index(c) if c in CELL_COLOR else 99)
    hs = [Line2D([], [], color=CELL_COLOR.get(c, INK2), marker="o", ls="none", ms=7, label=c) for c in cells]
    hs += [Line2D([], [], color=INK2, marker=N_MARKER[N], ls="none", ms=7, label=f"N = {N}") for N in Nlist
           if N in N_MARKER]
    hs += [Line2D([], [], color=INK2, marker=">", mfc="white", ls="none", ms=7, label="censored: plotted at T_N")]
    fig_legend(fig, hs, 6)
    title(fig, ["Mechanistic covariates vs the FR gain G(Ibar_F) (all cells and N)",
                "Each point: one cell at one N (small horizontal offsets separate the N). Establishment times: ABF arm, "
                "seed median (line: IQR). Experiment II lam1 = the alpha1 runs (shown once)."], 17.0,
          banners + [BANNER_DESCRIPTIVE])
    W.save(fig, "mech_m1_gain_vs_covariates", "Descriptive, not causal: G(Ibar_F) vs conditional-force variance, ABF "
           "establishment times and conditional relaxation time")


# =================================================================================================== main
DEFINITIONS = dict(
    regions="production-bin centres: left x < -0.5, gate |x| <= 0.5 (gate_neg x < 0, gate_pos x > 0), right x > 0.5; "
            "D2 deviations / variances only on the scorer's 150 eval-window bins [-1.5, 1.5]",
    D1="per seed: count_j / (opp_cum N) at the end (and per snapshot interval from the increments); seed median / IQR; "
       "EXPLORATORY hazard = deaths_j fr_every / C_all_j",
    D2_reference="r_j = <F*'>_j: eqb_metrics.GatewayScorer.bin_reference(scorer.sa) (analytic, Gauss-Legendre bin "
                 "average) of the cell's eqb_family.GatewayFamilyScorer",
    D2_pooled="n_cj = sum_s n_scj, m_cj = sum_s S_scj / n_cj, delta_cj = m_cj - r_j",
    D2_cluster_variance="V_cj = G/(G-1) sum_s (S_scj - m_cj n_scj)^2 / n_cj^2, G = seeds with n_scj > 0 (>= 2 distinct)",
    D2_signed="D_cR = sum_{j in R & EVAL} n_cj delta_cj / sum_{j in R & EVAL} n_cj",
    D2_contribution="K_cR = sum_{j in R & EVAL} n_cj delta_cj / sum_{j in R & EVAL} n_all,j",
    D2_B="B_c^2 = sum_{j in E_c} n_cj (delta_cj^2 - V_cj) / sum_{j in E_c} n_cj, E_c = EVAL bins with >= 2 seeds; "
         "B_c = sqrt(max(B_c^2, 0)); raw2 without -V; noise2 = sum n V / sum n; coverage = deposits in E_c / in EVAL",
    D2_CI="seed bootstrap: n_boot resamples of the seeds with replacement (default_rng(BOOT_SEED)); every quantity "
          "recomputed from the resampled seed sums.  Debiased quadratics (B_c^2, the matched A2 / B2): BASIC (pivotal) "
          "interval from the reflected draws B2_hat - (B2* - raw2_hat), raw2_hat = the plug-in without -V (the "
          "bootstrap-world parameter; plain percentiles of B2* would sit ~noise2 above the estimate); B_c interval = "
          "sqrt(max(., 0)) of the B2 ends; differences of B / matched excess / their changes evaluated on the jointly "
          "reflected draws.  Signed deviations, contributions, variance ratios, coverage: plain 2.5 / 97.5 percentiles. "
          "raw2, noise2: descriptive, no interval",
    D2_matched_excess="EXPLORATORY, analysis-defined (not in SCIENTIFIC_PLAN): over the bins usable in both arms, "
                      "weights n_cj of the FR class: B_ABF|c^2 = sum n_cj "
                      "(delta_ABF,j^2 - V_ABF,j) / sum n_cj; B_excess = B_c - B_ABF|c (both on those bins); dev_excess_R "
                      "= sum_R n_cj (delta_cj - delta_ABF,j) / sum_R n_cj",
    D2_contrasts="paired over common seeds with the same resamples; B_diff (cross-cell: B_c(cell) - B_c(ref), the "
                 "PREREGISTERED Experiment II reading), dev_diff and (cross-cell) the difference in differences "
                 "(B_c^FR - B^ABF)_a - (B_c^FR - B^ABF)_b and of the matched excess (B_excess_did = change from the "
                 "reference cell, EXPLORATORY analysis-defined); p two-sided bootstrap on the reflected draws, "
                 "unadjusted (Holm within each readout family in the synthesis readout); < 10 seeds: not calibrated",
    D2_variance_EXPLORATORY="pooled within-bin variance SSW_cj / n_cj vs e_j = <Var(f|x)>_j + <F*'^2>_j - <F*'>_j^2; "
                            "region ratio sum SSW / sum n e",
    D3="per seed fractions n_cR / n_R of dep_cross_C, cumulative and per snapshot interval; FR - ABF per-seed "
       "difference at the end: median and seed-bootstrap CI (eqb_metrics.boot_median_ci)",
    ylaw_EXPLORATORY="z = (y - m(x)) sqrt(beta k(x)) of the final walkers with |x - x0| <= 0.05, pooled over seeds; KS D vs "
                     "N(0,1) with nominal p; mean / var of z with seed-bootstrap CIs; EM expectation <1/(1 - lam k h/2)>",
)


DEFAULT_CELLS_ROOT = os.path.join(ROOT, "configs", "mechanism", "cells")
REPO_CROSS_CELL = os.path.join(ROOT, "results", "mechanism", "synthesis", "cross_cell.json")


def resolve_cross_cell_path(args):
    """(path or None, reason): the explicit --cross-cell, else the repository file only for the default inputs."""
    if args.cross_cell:
        return args.cross_cell, None
    overridden = [f for f, v in (("--results-override", args.results_override),
                                 ("--analysis-override", args.analysis_override)) if v]
    if os.path.realpath(args.cells_root) != os.path.realpath(DEFAULT_CELLS_ROOT):
        overridden.append("--cells-root")
    if overridden:
        return None, (f"inputs are overridden ({', '.join(overridden)}): the repository cross_cell.json is not used "
                      "without an explicit --cross-cell")
    return REPO_CROSS_CELL, None


def verify_cross_cell(cc, cells, args, prod):
    """Problems that make cross_cell.json foreign to this run: every analysed cell must be in it with exactly the
    summary file read here (same path, same sha256 -> same runs, same seeds, same n per contrast); a fixture document
    is never used for production cells."""
    probs = []
    if cc.get("fixture") and prod:
        probs.append("cross_cell.json was computed on FIXTURE cells, the analysed cells are production")
    ccc = cc.get("cells") or {}
    for (exp, c), P in cells.items():
        k = f"{exp}/{c}"
        sp = os.path.join(analysis_dir(P, args.analysis_override), "summary.json")
        e = ccc.get(k)
        if e is None:
            probs.append(f"{k}: not in cross_cell.json")
            continue
        have = os.path.exists(sp)
        if bool(e.get("available")) != have:
            probs.append(f"{k}: summary available here {have}, in cross_cell.json {bool(e.get('available'))}")
            continue
        if not have:
            continue
        cs = str(e.get("summary") or "")
        cs = cs if os.path.isabs(cs) else os.path.join(ROOT, cs)
        if os.path.realpath(cs) != os.path.realpath(sp):
            probs.append(f"{k}: cross_cell.json used {e.get('summary')}, this run reads {sp}")
        elif e.get("summary_sha256") != M.sha256_file(sp):
            probs.append(f"{k}: {sp} changed since cross_cell.json was made (sha256 "
                         f"{str(e.get('summary_sha256'))[:12]} -> {M.sha256_file(sp)[:12]})")
    return probs


def s3b_row_filter(cells, Nsel):
    """Keep a cross-cell contrast row only if both of its cells are analysed here and its N is selected."""
    have = {(e, c) for (e, c) in cells}

    def keep(exp, r):
        return ((exp, r.get("a")) in have and (exp, r.get("b")) in have
                and (not Nsel or int(r.get("N", -1)) in Nsel))
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells-root", default=os.path.join(ROOT, "configs", "mechanism", "cells"))
    ap.add_argument("--results-override", default=None)
    ap.add_argument("--analysis-override", default=None)
    ap.add_argument("--validation", default=os.path.join(ROOT, "results", "mechanism", "validation", "summary.json"))
    ap.add_argument("--cross-cell", default=None,
                    help="cross_cell.json to use (default: the repository's results/mechanism/synthesis/cross_cell.json "
                         "when no input override is given; with --results-override / --analysis-override / a "
                         "non-default --cells-root the contrasts are recomputed provisionally unless this is given).  "
                         "It is used only if it was made from exactly the summaries analysed here (path + sha256)")
    ap.add_argument("--out", default=ROOT, help="output root (figures/mechanism/..., results/mechanism/... under it)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--experiment", nargs="*", default=None, choices=list(EXP_ORDER))
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--N", type=int, nargs="*", default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args(argv)
    setup_style()
    out_root = os.path.abspath(args.out)
    guard(os.path.join(out_root, "figures", "mechanism"))
    guard(os.path.join(out_root, "results", "mechanism"))
    t0 = time.time()
    found, exp_missing, extra = discover(args.cells_root, args.experiment, args.cells)
    print(f"cells: {len(found)} found under {args.cells_root}; missing configs: {exp_missing or 'none'}; "
          f"unexpected: {extra or 'none'}", flush=True)
    cells, summ, docs, d2keep, ref_cache = {}, {}, {}, {}, {}
    deferred, doc_paths = [], {}
    summ_why, curves = {}, {}
    for exp, cname, P, cfg_path in found:
        cells[(exp, cname)] = P
        doc, keep, defer, jp = analyse_cell(exp, cname, P, cfg_path, args, out_root, ref_cache)
        docs[(exp, cname)] = doc
        deferred += defer
        doc_paths[(exp, cname)] = jp
        for (N, m), v in keep.items():
            d2keep[(exp, cname, N, m + "_arrays")] = v
        for N, R in doc["per_N"].items():
            for m in ("fr", "abf"):
                e = (R.get("D2") or {}).get(m)
                if e:
                    d2keep[(exp, cname, int(N), m)] = dict(end=_d2_end_arrays(e, R["D2"]["classes"]))
            if isinstance(R.get("D2_fractions"), dict) and "recent_lt_0p1_gate_end" in R["D2_fractions"]:
                d2keep[(exp, cname, int(N), "fr_frac")] = R["D2_fractions"]
            if R.get("D2_fr_vs_abf"):
                d2keep[(exp, cname, int(N), "fr_vs_abf")] = R["D2_fr_vs_abf"]
        adir = analysis_dir(P, args.analysis_override)
        S, why = load_summary(adir)
        if S is not None:
            summ[(exp, cname)] = S
        else:
            summ_why[(exp, cname)] = why
        curves[(exp, cname)] = {}
        for N in [int(n) for n in P["N_ladder"]]:
            if args.N and N not in args.N:
                continue
            seeds = [int(s) for s in P["seeds"]]
            curves[(exp, cname)][N] = {m: load_run_curves(adir, N, seeds, m) for m in ("abf", "fr")}
    # ---------------------------------------------------------------- per-cell figures with shared ranges + JSON
    print("[per-cell D1 / D2 / D3 figures, shared y ranges per experiment and N]", flush=True)
    shared = draw_deferred(deferred)
    for k, doc in docs.items():
        doc["shared_y_ranges"] = {kk: v for kk, v in shared.items() if kk.split(":")[1] == k[0]}
        write_json(doc_paths[k], doc)
        print(f"  wrote {os.path.relpath(doc_paths[k], out_root)}", flush=True)
    del deferred
    # ---------------------------------------------------------------- synthesis
    print("[synthesis]", flush=True)
    syn = dict(schema=SCHEMA_SYN, n_boot=int(args.n_boot), boot_seed=BOOT_SEED, warnings=[], inputs={},
               definitions=DEFINITIONS)
    prod = all(is_production(P) for P in cells.values()) and bool(cells)
    banners = [] if prod else ["NON-PRODUCTION DATA in at least one cell: pipeline check, never interpreted "
                               "(SCIENTIFIC_PLAN section 9)"]
    syn["production"] = prod
    W = FigWriter(os.path.join(out_root, "figures", "mechanism", "synthesis"), args.dpi, out_root,
                  enabled=not args.no_figures)
    V, vwhy = None, None
    if os.path.exists(args.validation):
        try:
            V = json.load(open(args.validation))
        except Exception as e:  # noqa: BLE001
            vwhy = f"unreadable: {e}"
    else:
        vwhy = f"absent: {args.validation}"
    syn["inputs"]["validation_summary"] = dict(path=args.validation, available=V is not None, why=vwhy,
                                               note=None if V is not None else
                                               "S4 and the mechanistic plots fall back to the exact Var(f|x) and the "
                                               "frozen-x OU predictions; no measured conditional ACF time is shown")
    if V is None:
        print(f"  validation summary {vwhy}: S4 / m1 use exact / frozen-x predicted covariates", flush=True)
    cc, ccsrc, cc_problems = None, None, []
    cc_path, cc_why = resolve_cross_cell_path(args)
    if cc_path and os.path.exists(cc_path):
        try:
            cc = json.load(open(cc_path))
            ccsrc = os.path.relpath(cc_path, ROOT) if os.path.abspath(cc_path).startswith(ROOT + os.sep) else cc_path
        except Exception as e:  # noqa: BLE001
            cc_why = f"{cc_path} unreadable ({e})"
            syn["warnings"].append(f"cross_cell.json unreadable: {e}")
        if cc is not None:
            cc_problems = verify_cross_cell(cc, cells, args, prod)
            if cc_problems:
                msg = (f"{ccsrc} does NOT belong to the summaries analysed here ({len(cc_problems)} problems, e.g. "
                       f"{cc_problems[0]}): not used; contrasts recomputed provisionally")
                syn["warnings"].append(msg)
                print(f"  WARNING: {msg}", flush=True)
                cc_why = f"{ccsrc} rejected: {cc_problems[0]}"
                cc = None
            elif not cc.get("freshness_verified", True):
                syn["warnings"].append(f"{ccsrc}: run-tree freshness was NOT verified when it was made")
    elif cc_path:
        cc_why = f"{cc_path} is absent"
    if cc is None:
        try:
            import cross_cell as XC
            recs = {}
            for (exp, cname), P in cells.items():
                recs.setdefault(exp, {})[cname] = dict(
                    config=os.path.join(args.cells_root, exp, cname + ".json"), P=P,
                    summary_path=os.path.join(analysis_dir(P, args.analysis_override), "summary.json"),
                    S=summ.get((exp, cname)), why=summ_why.get((exp, cname)))
            cc = XC.analyse(recs, n_boot=int(args.n_boot), production=prod, verify_runs=False)
            ccsrc = f"PROVISIONAL: computed here by cross_cell.analyse(verify_runs=False) because {cc_why}"
        except Exception as e:  # noqa: BLE001
            syn["warnings"].append(f"cross-cell contrasts unavailable: cross_cell.json absent and cross_cell.analyse "
                                   f"failed ({type(e).__name__}: {e})")
            cc = None
    syn["inputs"]["cross_cell"] = dict(path=cc_path, available=bool(cc_path and os.path.exists(cc_path)), source=ccsrc,
                                       rejected_because=cc_problems or None)
    print(f"  cross-cell contrasts: {ccsrc or 'unavailable'}", flush=True)
    # S1
    mf_models = []
    for c in CELL_ORDER["matched_free_energy"]:
        P = cells.get(("matched_free_energy", c))
        mf_models.append((c, cell_model(P) if P else dyn_model(c)))
    syn["S1_barrier_decomposition"] = fig_s1(W, mf_models, banners)
    ent = {c: v["entropic_fraction"] for c, v in syn["S1_barrier_decomposition"].items()}
    # S2
    for exp in EXP_ORDER:
        Ns = sorted({N for (e, c), byN in curves.items() if e == exp for N in byN}, reverse=True)
        for N in Ns:
            cv = {c: {m: curves[(e, c)][N][m][0] for m in ("abf", "fr")} for (e, c) in curves if e == exp
                  and N in curves[(e, c)]}
            thr = next((P["thresholds"] for (e, c), P in cells.items() if e == exp), None)
            for xk in ("u", "t"):
                fig_s2(W, exp, N, cv, xk, banners, thr)
    syn["S2_missing_run_metric_curves"] = {f"{e}/{c}/N{N}/{m}": v[m][1] for (e, c), byN in curves.items()
                                           for N, v in byN.items() for m in ("abf", "fr") if v[m][1]}
    # S3
    Ns1 = sorted({int(n) for (e, c), P in cells.items() if e == "matched_free_energy" for n in P["N_ladder"]}, reverse=True)
    if args.N:
        Ns1 = [n for n in Ns1 if n in args.N]
    if any(e == "matched_free_energy" for e, _ in summ):
        syn["S3"] = fig_s3(W, summ, ent, Ns1, banners)
    else:
        W.skip("mech_s3_gain_vs_entropic_fraction", "no Experiment I summary")
    if cc is not None:
        keep_row = s3b_row_filter(cells, args.N)
        fig_s3b(W, cc, ccsrc, banners, keep_row)
        syn["S3b_cross_cell"] = {e: {met: [{k: r.get(k) for k in ("id", "a", "b", "N", "kind", "n", "ratio", "ratio_ci95",
                                                                   "p", "p_holm", "verdict", "equivalent")}
                                           for r in (E.get("metrics", {}).get(met) or {}).get("contrasts", [])
                                           if keep_row(e, r)]
                                     for met in ("Ibar_F", "Ibar_TV_half", "Ibar_Fp_stat")}
                                 for e, E in (cc.get("experiments") or {}).items()}
        syn["S3b_source"] = ccsrc
        syn["S3b_note"] = ("rows restricted to the analysed cells and N; p_holm and verdicts are over the source's "
                           "frozen families")
    else:
        W.skip("mech_s3b_cross_cell_contrasts", "no cross-cell contrasts (file absent and not computable)")
    # S4
    h0 = float(next(iter(cells.values()))["engine_cfg"]["h"]) if cells else 2.5e-5
    fig_s4(W, mf_models, V, h0, banners)
    syn["S4"] = s4_table(V, h0)
    # S5 + D2 cross-cell contrasts
    lam_cells = [c for c in CELL_ORDER["conditional_relaxation"] if ("conditional_relaxation", c) in cells]
    Ns2 = sorted({int(n) for (e, c), P in cells.items() if e == "conditional_relaxation" for n in P["N_ladder"]},
                 reverse=True)
    if args.N:
        Ns2 = [n for n in Ns2 if n in args.N]
    d2c = d2_cross_cell(d2keep, ref_cache, cells, int(args.n_boot))
    syn["D2_cross_cell"] = d2c
    if lam_cells:
        syn["S5"] = fig_s5(W, summ, d2keep, lam_cells, Ns2, banners)
        syn["S5b"] = fig_d2_cells(W, "conditional_relaxation", lam_cells, d2keep, d2c, Ns2, banners)
        syn["S5_D2_B_vs_lambda"] = {f"{c}/N{N}": {arm: _d2_end_table((d2keep.get(("conditional_relaxation", c, N, arm))
                                                                      or {}).get("end"))
                                                  for arm in ("fr", "abf")}
                                    for c in lam_cells for N in Ns2}
        syn["S5b_D2_readout"] = d2_readout_table("conditional_relaxation", lam_cells, d2keep, d2c, Ns2)
    else:
        W.skip("mech_s5_lambda", "no Experiment II cell")
        W.skip("mech_s5b_d2_conditional_relaxation", "no Experiment II cell")
    mf_cells = [c for c in CELL_ORDER["matched_free_energy"] if ("matched_free_energy", c) in cells]
    if mf_cells:
        syn["S5c"] = fig_d2_cells(W, "matched_free_energy", mf_cells, d2keep, d2c, Ns1, banners)
        syn["S5c_D2_readout_EXPLORATORY"] = d2_readout_table("matched_free_energy", mf_cells, d2keep, d2c, Ns1)
    else:
        W.skip("mech_s5c_d2_matched_free_energy", "no Experiment I cell")
    # S6
    if summ:
        syn["S6"] = fig_s6(W, summ, cells, banners)
    else:
        W.skip("mech_s6_marginal_vs_free_energy", "no cell summary")
    # mechanistic covariates
    cov = covariates(cells, summ, V, args.N)
    syn["mechanistic_covariates"] = cov
    if cov:
        fig_m1(W, cov, banners)
    else:
        W.skip("mech_m1_gain_vs_covariates", "no cell summary with G(Ibar_F)")
    # completeness
    comp = completeness(docs, cells, summ, summ_why, curves, exp_missing, extra, V, vwhy, cc, ccsrc, args)
    syn["completeness"] = comp
    syn["figures"] = W.entries
    syn["figures_skipped"] = W.skipped
    allfig = W.entries + [f for d in docs.values() for R in d["per_N"].values() for f in R.get("figures", [])]
    syn["legibility"] = LG.summarize({f["files"][0]: f["legibility"] for f in allfig})
    syn["wall_s"] = time.time() - t0
    jp = os.path.join(out_root, "results", "mechanism", "synthesis", "mech_synthesis.json")
    write_json(jp, syn)
    print_completeness(comp)
    lg = syn["legibility"]
    print(f"figures: {lg['n_figures']} written, legibility pass {lg['n_pass']}, fail {lg['n_fail']}"
          + (f" {lg['failing']}" if lg["n_fail"] else ""), flush=True)
    print(f"wrote {os.path.relpath(jp, out_root)} ({time.time() - t0:.1f} s)", flush=True)
    return syn


def holm_adjust(pvals, m=None):
    """Holm step-down adjusted p (cross_cell.holm convention: a NaN p is not computed, counted in the family as 1,
    and its adjusted p is NaN); m = planned family size (default: the number of entries)."""
    raw = [fnum(v) for v in pvals]
    p = np.array([v if math.isfinite(v) else 1.0 for v in raw], float)
    m = int(m or p.size)
    order = np.argsort(p, kind="stable")
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return [float(a) if math.isfinite(v) else math.nan for a, v in zip(adj, raw)]


READOUT_DEFINITION = dict(
    preregistered=("SCIENTIFIC_PLAN sections 4 and 7: the class-wise conditional-mean-force error against the "
                   "bin-averaged F*' and whether it grows as lambda decreases: B_c per FR clone-age class (and 'all', "
                   "and ABF's B as the reference level) with its 95 % seed-bootstrap CI, and the paired change "
                   "B_c(cell) - B_c(ref) over the common seeds (CI, unadjusted p, Holm p)"),
    exploratory_analysis_defined=("NOT in the plan: the matched-position excess B_c - B_ABF|c (ABF's debiased error "
                                  "weighted by the FR class's deposit counts) and its paired change from the reference "
                                  "cell (difference in differences); CI, unadjusted p, Holm p"),
    holm=("Holm within each family = (statistic, N) over the cells (non-reference cells for the changes) x the six "
          "FR classes; the per-test p are unadjusted"),
    small_n=f"with fewer than {SMALL_N} seeds the CIs and p are NaN (bootstrap not calibrated) and the row says so",
    missing="a quantity that could not be computed is 'MISSING: <reason>'",
)


def d2_readout_table(exp, cell_list, d2keep, d2c, Ns):
    """Compact D2 readout per cell and N (end snapshot) with TWO readouts side by side, neither called 'the' readout
    (READOUT_DEFINITION): 'preregistered' (B_c, ABF's B, the paired change B_c(cell) - B_c(ref)) and
    'exploratory_analysis_defined' (the matched-position excess and its change from the reference cell).  Plan section
    7: Experiment II's conditional-relaxation reading is 'proven' only if recently cloned deposits carry a mean-force
    error that grows as lambda decreases."""
    ref = REF_CELL[exp]
    rows = {}
    fam = {}                                             # (statistic, N) -> [(row dict, key)]

    def ci_or_nan(ci, ok):
        return list(ci) if ok and ci is not None else [math.nan, math.nan]

    for N in Ns:
        for c in cell_list:
            fr = (d2keep.get((exp, c, N, "fr")) or {}).get("end")
            ab = (d2keep.get((exp, c, N, "abf")) or {}).get("end")
            pv = d2keep.get((exp, c, N, "fr_vs_abf")) or {}
            pv_missing = (pv.get("missing") if isinstance(pv.get("missing"), str) else
                          (None if pv else "no FR-vs-ABF D2 block (FR or ABF arm without D2)"))
            cc = ((d2c or {}).get(exp) or {}).get(f"{c}_vs_{ref}_N{N}") or {}
            cc_missing = None if c == ref else (cc.get("missing") or (None if cc else "no cross-cell D2 contrast"))
            e = dict(n_seeds=dict(fr=(fr or {}).get("n_seeds"), abf=(ab or {}).get("n_seeds"),
                                  common_with_ref=cc.get("n_common_seeds")))
            ok_ab = ab is not None and (ab.get("n_seeds") or 0) >= SMALL_N
            e["abf_B"] = (dict(B=ab["B"][ALL], ci95=ci_or_nan([ab["B_lo"][ALL], ab["B_hi"][ALL]], ok_ab))
                          if ab else "MISSING: no ABF D2 arrays")
            if c != ref:
                xa = cc.get("abf_all")
                if xa:
                    okx = bool(xa.get("p_calibrated"))
                    e["abf_B_diff_from_ref"] = dict(value=xa["B_diff"], ci95=ci_or_nan(xa["B_diff_ci95"], okx),
                                                    p=xa["B_diff_p"] if okx else math.nan)
                else:
                    e["abf_B_diff_from_ref"] = f"MISSING: {cc_missing or 'no ABF arm in one of the cells'}"
            notes = []
            for i in range(N_AGE + 1):
                nm = AGE_DEFAULT[i] if i < N_AGE else "all"
                pre, exp_ = {}, {}
                if fr:
                    okf = (fr.get("n_seeds") or 0) >= SMALL_N
                    pre["B"], pre["B_ci95"] = fr["B"][i], ci_or_nan([fr["B_lo"][i], fr["B_hi"][i]], okf)
                else:
                    pre["B"] = "MISSING: no FR D2 arrays"
                x = pv.get(nm) or {}
                if "B_excess" in x:
                    okp = bool(x.get("p_calibrated"))
                    exp_.update(B_excess=x["B_excess"], B_excess_ci95=ci_or_nan(x["B_excess_ci95"], okp),
                                B_excess_p=x["B_excess_p"] if okp else math.nan,
                                Bm_b=x.get("Bm_b"), Bm_b_ci95=ci_or_nan(x.get("Bm_b_ci95"), okp),
                                matched_coverage=x.get("matched_coverage"), p_calibrated=okp)
                    fam.setdefault(("B_excess", N), []).append((exp_, "B_excess"))
                    if not okp:
                        notes.append(x.get("note") or "FR-vs-ABF bootstrap not calibrated")
                else:
                    exp_["B_excess"] = f"MISSING: {pv_missing or 'class absent from the FR-vs-ABF block'}"
                if c != ref:
                    y = cc.get(nm) or {}
                    if "B_diff" in y:
                        oky = bool(y.get("p_calibrated"))
                        pre.update(B_diff_from_ref=y["B_diff"], B_diff_from_ref_ci95=ci_or_nan(y["B_diff_ci95"], oky),
                                   B_diff_from_ref_p=y["B_diff_p"] if oky else math.nan, p_calibrated=oky)
                        fam.setdefault(("B_diff_from_ref", N), []).append((pre, "B_diff_from_ref"))
                        if not oky:
                            notes.append(y.get("note") or "cross-cell bootstrap not calibrated")
                    else:
                        pre["B_diff_from_ref"] = f"MISSING: {cc_missing or 'class absent from the cross-cell block'}"
                    if "B_excess_did" in y:
                        oky = bool(y.get("p_calibrated"))
                        exp_.update(change_from_ref=y["B_excess_did"],
                                    change_from_ref_ci95=ci_or_nan(y["B_excess_did_ci95"], oky),
                                    change_from_ref_p=y["B_excess_did_p"] if oky else math.nan)
                        fam.setdefault(("change_from_ref", N), []).append((exp_, "change_from_ref"))
                    else:
                        exp_["change_from_ref"] = ("MISSING: " + (cc_missing or "no matched excess in the cross-cell "
                                                                  "block (an ABF arm is missing in one of the cells)"))
                e[nm] = dict(preregistered=pre, exploratory_analysis_defined=exp_)
            e["p_calibrated"] = not notes
            if notes:
                e["note"] = sorted(set(notes))[0]
            rows[f"{c}/N{N}"] = e
    for (stat, N), lst in fam.items():
        ncells = len(cell_list) - (1 if stat != "B_excess" else 0)
        adj = holm_adjust([d.get(k + "_p") for d, k in lst], m=max(len(lst), ncells * (N_AGE + 1)))
        for (d, k), a in zip(lst, adj):
            d[k + "_p_holm"] = a
    return dict(definition=READOUT_DEFINITION, rows=rows)


def _d2_end_arrays(e, classes):
    """Class arrays of B at the end snapshot from d2_json's layout (for S5)."""
    return dict(classes=list(classes), B=np.asarray(e["B"]["value"][-1], float),
                B_lo=np.asarray(e["B"]["ci_lo"][-1], float), B_hi=np.asarray(e["B"]["ci_hi"][-1], float),
                n_seeds=int(e["n_seeds"][-1]))


def _d2_end_table(st):
    if st is None:
        return None
    return {c: dict(B=st["B"][i], ci95=[st["B_lo"][i], st["B_hi"][i]]) for i, c in enumerate(st["classes"])}


def d2_cross_cell(d2keep, ref_cache, cells, n_boot):
    """Paired (common seeds) D2 contrasts of every cell vs its experiment's reference cell at the same N: per class
    B_c(a) - B_c(ref), D_cR differences, and the difference in differences of B_c^FR - B^ABF."""
    out = {}
    for (exp, c), P in cells.items():
        refc = REF_CELL[exp]
        if c == refc:
            continue
        for N in [int(n) for n in P["N_ladder"]]:
            ka = {m: d2keep.get((exp, c, N, m + "_arrays")) for m in ("fr", "abf")}
            kb = {m: d2keep.get((exp, refc, N, m + "_arrays")) for m in ("fr", "abf")}
            ra, rb = ref_cache.get((exp, c)), ref_cache.get((exp, refc))
            key = f"{c}_vs_{refc}_N{N}"
            if not (ka["fr"] and kb["fr"] and ra is not None and rb is not None):
                out.setdefault(exp, {})[key] = dict(missing="D2 arrays of one of the cells are missing")
                continue
            sets = [set(ka["fr"]["seeds"]), set(kb["fr"]["seeds"])]
            both = ka["abf"] is not None and kb["abf"] is not None
            if both:
                sets += [set(ka["abf"]["seeds"]), set(kb["abf"]["seeds"])]
            common = sorted(set.intersection(*sets))
            if len(common) < 2:
                out.setdefault(exp, {})[key] = dict(missing=f"fewer than 2 common seeds ({common})")
                continue

            def calc(k, r):
                idx = [k["seeds"].index(s) for s in common]
                return D2Calc(*(a[idx] for a in k["arrays"]), r["r"], r["etot"])

            cfa, cfb = calc(ka["fr"], ra), calc(kb["fr"], rb)
            caa = calc(ka["abf"], ra) if both else None
            cab = calc(kb["abf"], rb) if both else None
            E = dict(n_common_seeds=len(common), a=c, b=refc, N=N)
            cls = list(range(N_AGE + 1))
            if both:
                pr = d2_paired(cfa, cls, cfb, cls, n_boot, calc_c=caa, cc=ALL, calc_d=cab, cd=ALL, matched=True)
            else:
                pr = d2_paired(cfa, cls, cfb, cls, n_boot)
            for cl in cls:
                E[AGE_DEFAULT[cl] if cl < N_AGE else "all"] = jsonify_paired(pr[cl])
            if both:
                E["abf_all"] = jsonify_paired(d2_paired(caa, ALL, cab, ALL, n_boot))
            out.setdefault(exp, {})[key] = E
    return out


def s4_table(V, h):
    out = {}
    for dn in GF.DYNAMICS_ORDER:
        m = dyn_model(dn)
        blk, why = validation_lookup(V, dn, h)
        e = dict(var_f_peak_exact=float(np.max(GF.cond_var_f(np.linspace(-0.6, 0.6, 2401), m))), windows={})
        for c in ACF_X:
            w = ((((blk or {}).get("acf") or {}).get("windows") or {}).get(f"{c:g}")) or {}
            e["windows"][f"{c:g}"] = dict(tau_y_pred=predicted_tau(m, c, "y"), tau_f_pred=predicted_tau(m, c, "f"),
                                          tau_y_measured=w.get("tau_int_y_5tau"), tau_f_measured=w.get("tau_int_f_5tau"),
                                          resolved=w.get("resolved"))
        e["validation"] = "present" if blk else why
        out[dn] = e
    return out


def covariates(cells, summ, V, Nsel):
    rows = []
    for (exp, c), P in cells.items():
        S = summ.get((exp, c))
        if S is None:
            continue
        if exp == "conditional_relaxation" and c == "lam1" and ("matched_free_energy", "alpha1") in summ:
            continue
        m = cell_model(P)
        h = float(P["engine_cfg"]["h"])
        dn = dynamics_name(m)
        blk, why = validation_lookup(V, dn, h)
        fv = ((blk or {}).get("em_flat") or {}).get("force_var_profile")
        if fv:
            x = np.asarray(fv["x"], float)
            vf, vsrc = float(np.mean(np.asarray(fv["measured"], float)[np.abs(x) <= 0.5])), "validation (measured)"
        else:
            xx = CENTRES[np.abs(CENTRES) <= 0.5]
            sub = xx[:, None] + DELTA * (np.linspace(-0.5, 0.5, 41)[None, :])
            vf, vsrc = float(np.mean(GF.cond_var_f(sub, m))), "exact (validation absent)"
        w = (((blk or {}).get("acf") or {}).get("windows") or {}).get(f"{-0.21:g}") or {}
        tf = fnum(w.get("tau_int_f_5tau"))
        tsrc = "validation (measured)"
        if not math.isfinite(tf):
            p = predicted_tau(m, -0.21, "f")
            tf = fnum(p)
            if p is None:
                tsrc = "undefined: f does not depend on y (alpha = 0)"
            else:
                tsrc = "frozen-x OU prediction" + ("" if blk else " (validation absent)")
        for N in [int(n) for n in P["N_ladder"]]:
            if Nsel and N not in Nsel:
                continue
            g = g_of(S, N, "primary", "Ibar_F")
            if g is None:
                continue
            r = dict(experiment=exp, cell=c, N=N, G=g["G"], lo=g["lo"], hi=g["hi"], var_f_gate=vf,
                     var_f_gate_source=vsrc, tau_f=tf, tau_f_source=tsrc, T_N=float(P["B"]) * h / N)
            for key, sk in (("tau_TV_half", "tau_TV_half_t"), ("est_cum", "est_cum_t"), ("first_right", "first_right_t"),
                            ("transitions", "final_transitions")):
                a = abf_scalar(S, N, sk)
                if a is None:
                    r[key + "_source"] = f"MISSING: summary per_N.{N}.abf.{sk}"
                    continue
                r[key] = a["median"]
                r[key + "_censored"] = not math.isfinite(a["median"])
                r[key + "_q25"], r[key + "_q75"] = a["q25"], a["q75"]
                r[key + "_n_censored"] = a["n_inf"]
                r[key + "_source"] = "summary per_N.abf (seed median)"
            rows.append(r)
    return rows


def completeness(docs, cells, summ, summ_why, curves, exp_missing, extra, V, vwhy, cc, ccsrc, args):
    rows = []
    for (exp, c), d in docs.items():
        for N, R in d["per_N"].items():
            for m in ("abf", "fr"):
                st = R["status"][m]
                cv = (curves.get((exp, c)) or {}).get(int(N), {}).get(m, ({}, []))
                rows.append(dict(cell=f"{exp}/{c}", N=int(N), method=m, planned=st["planned"], complete=st["complete"],
                                 with_diag=st["with_diag"], missing=len(st["missing"]), running=len(st["running"]),
                                 invalid=len(st["invalid"]), diag_missing=len(st["diag_missing"]),
                                 diag_problems=len(st["diag_problems"]), run_metric_curves=len(cv[0]),
                                 summary=(exp, c) in summ,
                                 detail=dict(missing=st["missing"], running=st["running"], invalid=st["invalid"],
                                             diag_missing={k: v for k, v in list(st["diag_missing"].items())[:3]},
                                             diag_problems=st["diag_problems"])))
    return dict(rows=rows, cells_without_config=exp_missing, unexpected_cells=extra,
                summaries_missing={f"{e}/{c}": w for (e, c), w in summ_why.items()},
                validation_summary=("present" if V is not None else vwhy),
                cross_cell=("present and verified against the analysed summaries (path + sha256): " + ccsrc
                            if (cc is not None and ccsrc and not ccsrc.startswith("PROVISIONAL")) else
                            (ccsrc or "unavailable")))


def print_completeness(comp):
    print("\nCOMPLETENESS (runs: complete / planned; diag = with consistent D1-D4; curves = per-run metric caches)")
    hdr = f"{'cell':40s} {'N':>5s} {'arm':4s} {'complete':>9s} {'diag':>5s} {'miss':>5s} {'run':>4s} {'inval':>5s} " \
          f"{'curves':>6s} summary"
    print(hdr)
    print("-" * len(hdr))
    for r in comp["rows"]:
        print(f"{r['cell']:40s} {r['N']:5d} {r['method']:4s} {r['complete']:4d}/{r['planned']:<4d} {r['with_diag']:5d} "
              f"{r['missing']:5d} {r['running']:4d} {r['invalid']:5d} {r['run_metric_curves']:6d} "
              f"{'yes' if r['summary'] else 'MISSING'}")
    if comp["cells_without_config"]:
        print(f"cells WITHOUT a config (missing): {comp['cells_without_config']}")
    if comp["unexpected_cells"]:
        print(f"unexpected cell configs (not analysed): {comp['unexpected_cells']}")
    for k, v in comp["summaries_missing"].items():
        print(f"summary MISSING {k}: {v}")
    print(f"validation summary: {comp['validation_summary']}")
    print(f"cross-cell contrasts: {comp['cross_cell']}")


if __name__ == "__main__":
    main()
