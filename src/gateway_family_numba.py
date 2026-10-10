"""Gateway-FAMILY replica-ladder engine (mechanism campaign): histogram ABF (+ uniform FR), ONE arm per call, on a
family of 2-D models that all share the gateway's free energy F*(x).

Mechanism campaign (docs/mechanism/SCIENTIFIC_PLAN.md, configs/mechanism/*.json), 2026-10-10.  Derived by copy +
extension from ``src/gateway_ladder_numba.py`` (the equal-budget engine), which is NOT imported by the kernel and is
NOT modified (its sha256 is recorded in the equal-budget results).  The P0 histogram estimator, the FR operator, the
two random streams, the chunked/checkpointed driver and the result contract are that engine's, op for op; read its
docstring for everything not restated here.  tests/test_gateway_family_numba.py proves the equivalences.

Models (SCIENTIFIC_PLAN section 1; cfg keys variant, alpha, kappa, lam)
--------------------------------------------------------------------
omega(x) = w_out + (w_in - w_out) exp(-x^2 / (2 s^2)) (= 1 + 31 exp(-x^2/(2 s^2)) at the frozen cell), omega' its
derivative, F*(x) = H (x^2-1)^2 + log(omega)/beta, F*'(x) = 4Hx(x^2-1) + (omega'/omega)/beta.  x reflected into
[-1.8, 1.8] (_reflect, unchanged), y unbounded.
  variant 'alpha' (alpha float; kappa must be None):
      V = H(x^2-1)^2 + ((1-alpha)/beta) log(omega) + omega^(2 alpha) y^2 / 2
      f_x = dV/dx = 4Hx(x^2-1) + ((1-alpha)/beta)(omega'/omega) + alpha omega^(2 alpha) (omega'/omega) y^2
      dV/dy = omega^(2 alpha) y;   Y | x ~ N(0, 1/(beta omega^(2 alpha)));  alpha = 1 is the original gateway.
  variant 'shift' (kappa > 0; alpha must be None):
      V = F*(x) + kappa^2 (y - m(x))^2 / 2,  m = (sqrt(2/beta)/kappa) log(omega),  m' = (sqrt(2/beta)/kappa) omega'/omega
      f_x = F*'(x) - kappa^2 (y - m) m',  dV/dy = kappa^2 (y - m);  Y | x ~ N(m(x), 1/(beta kappa^2)).
  transverse mobility lam > 0 (both variants): y <- y - lam (dV/dy) h + sqrt(2 lam h / beta) zy (drift AND noise
      scaled, Gibbs law unchanged); the x update is unchanged.
Exact for every model: F = F* + C, E[f | x] = F*'(x), Var(f | x) = (2 alpha^2/beta^2)(omega'/omega)^2 (alpha) or
kappa^2 m'^2/beta = (2/beta^2)(omega'/omega)^2 (shift); the EM y-update at frozen x has the stationary variance
(1/(beta k)) / (1 - lam k h / 2), k = omega^(2 alpha) or kappa^2 (the 'stiffness').  Reference functions (numpy):
omega_np, F_star, dF_star, potential, grad_V, cond_y_law, cond_force_var, stiffness, em_mean_force,
em_var_inflation.

Code paths (model id computed OUTSIDE the kernel by model_params, MODEL_* constants):
  MODEL_ORIG  (variant 'alpha', alpha == 1.0, any lam): the ORIGINAL force expressions of gateway_ladder_numba,
              f = 4Hx(x^2-1) + om dom y y, fy = om om y, op for op;
  MODEL_ALPHA (variant 'alpha', alpha != 1.0): omega^(2 alpha) = om ** (2 alpha) -- numba's float ** float, i.e. libm
              pow (NOT exp(2 alpha log om)); the Python reference / init use numpy's ** (pow); bitwise agreement of
              the two is not required and not claimed;
  MODEL_SHIFT (variant 'shift'): log(om) via math.log.
  The y-update keeps the ORIGINAL expression y - fy h + sqrt(2h/beta) zy when lam == 1.0 and uses
  y - lam fy h + sqrt(2 lam h/beta) zy otherwise.  So at (alpha, alpha = 1, lam = 1) every floating-point operation
  of the dynamics is the original's, and the run is BITWISE gateway_ladder_numba.run_arm (both arms; tested at
  N 2, 128, 2048 with every diagnostic on).

Initial state (init_family).  EXACTLY init_left's draws: numpy default_rng(1000 + seed), x ~ N(-1, 0.05) reflected,
then z ~ N(0, 1) (N each, in that order); y = z sqrt(1/(beta omega^(2 alpha))) (written like init_left's
z sqrt(1/(beta om**2)), which it IS at alpha = 1: bitwise init_left), or y = m(x) + z sqrt(1/(beta kappa^2)) (shift).
lam does not enter.  So x and z are coupled across every variant and lam.

Random streams, pairing, FR, checkpoints: unchanged (Langevin noise = numba MT19937 seeded with default_noise_seed
(seed, N), 2N normals per step, nothing else; FR uniforms = PCG64(SeedSequence([seed, N, 1])), fixed blocks of
L = N + cap + max(N, cap) doubles per opportunity; the ABF and FR arms of one (variant, N, seed) see identical noise;
the noise seed does not depend on the model, so the noise is also coupled across cells).  Test-only switches:
``_fr_internal_rng`` (as before) and ``_freeze_x_test`` (x is not moved: the frozen-x EM test of the y integrator;
the normals are still drawn, so the streams are unchanged).

New PASSIVE diagnostics (SCIENTIFIC_PLAN section 4; skipped and their keys OMITTED when diagnostics=False; they only
READ the dynamics state -- inertness is tested bitwise for every variant)
------------------------------------------------------------------------------------------------------------
D1 fr_death_pos_hist_cum, fr_birth_pos_hist_cum (S, nb) int64, cumulative, at every save: production-bin (deposit
   bin rule) histograms of the x of the REALISED deaths (slots with zero offspring, at their pre-gather x) and of the
   realised births (each extra copy, i.e. offspring - 1 per source with >= 2 offspring, at its source's x).  Their
   totals both equal fr_deaths_cum.
D2 per-walker clone_step (int64, -1 = never cloned).  At an FR gather, every new slot whose source has >= 2 offspring
   (the source's surviving slot AND all its copies) gets clone_step = the completed-step count of the state the
   gather acted on (loop step + 1, the FR acts on the post-move state); every other slot inherits its source's
   value.  At every deposit (the state at completed-step count k), the walker's clone age a = (k - clone_step) h is
   classed into [0, e0), [e0, e1), [e1, e2), >= e2, never (cfg clone_age_edges_t, default 0.01, 0.1, 1 t.u.; the
   first deposit after a clone has a = 0).  Per class and production bin: count, sum f, sum f^2 (float64; the
   count is integral, like C_all).  Saved ONLY at the snapshot saves (default snapshot_grid: the steps of
   PROFILE_SNAPSHOT_U = 0.01 ... 1.00 of n_steps, the same rule as prereg_save_grid, which includes the end):
   dep_age_C, dep_age_M, dep_age_M2 (n_snap, 5, nb), snap_step / snap_t / snap_u (n_snap,).  Same pre-deposit
   convention as C_all: dep_age_C[s].sum(0) == C_all at the same step, exactly.  The all-deposit sum of f^2 per bin
   is NOT stored separately: it is dep_age_M2[s].sum(0) (the classes partition the deposits).
D3 per-walker cross_step (int64, -1 = never): the completed-step count at which the walker's hysteresis well label
   (events_cum's) last flipped -1 <-> +1 (a true transition; the initial 0 -> +-1 assignment of a walker starting in
   the gate is not a crossing); clones inherit it with the label.  Deposit COUNTS by time since crossing, classes
   [0, c0), [c0, c1), [c1, c2), >= c2, never (cfg cross_age_edges_t, default 0.1, 1, 10 t.u.):
   dep_cross_C (n_snap, 5, nb) float64 (integral), at the snapshot saves; dep_cross_C[s].sum(0) == C_all exactly.
   Class edges are applied in integer steps: a < e  <=>  d < ceil(e / h) (e / h rounded when within 1e-6 of an
   integer); meta.diag_classes records both.
D4 traces_y (K, n_tr): the y of the traced walkers (same persistent-id columns as traces).

Result contract: gateway_ladder_numba's keys and meanings, unchanged, PLUS the D1-D4 keys above; cfg carries
variant / alpha / kappa / lam (and the class edges); meta.model records them with the model id, the derived
stiffness (gate / well / max), lam * stiffness, the frozen-x relaxation time 1/(lam k) and the EM variance
inflation at the stiffest point; ENGINE_VERSION 'gateway_family_numba/1'; run_signature includes the model
parameters (and the snapshot grid).  meta.system = 'gateway_family'.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import time

import numpy as np
import numba
from numba import njit, _helperlib

ENGINE_VERSION = "gateway_family_numba/1"
XMIN, XMAX = -1.8, 1.8
N_GRID = 181
EPS = 1e-30
X_BASIN = 0.5
NB_PROD = 180
N_TRACE_MAX = 32
N_AGE_CLASSES = 5            # 3 edges -> [0,e0), [e0,e1), [e1,e2), >= e2, never

MODEL_ORIG, MODEL_ALPHA, MODEL_SHIFT = 0, 1, 2
MODEL_NAMES = {MODEL_ORIG: "orig (alpha = 1)", MODEL_ALPHA: "alpha", MODEL_SHIFT: "shift"}

DEFAULT_CFG = dict(
    system="gateway_family",
    beta=16.0, H=0.5, omega_out=1.0, omega_in=32.0, s=0.1,
    h=2.5e-5,
    nb=NB_PROD, min_count=1.0,
    gamma=1.5, eta=0.1, fr_interval_t=0.004, ramp_t=4.0, score_clip=3.0,
    max_event_fraction=0.08, cap_min=1,
    genealogy_window_t=1.6,
    trace_early_dt_t=0.1, trace_early_until_t=40.0, trace_late_dt_t=10.0, trace_max=5000,
    # model family (SCIENTIFIC_PLAN section 1); alpha None -> 1.0 for variant 'alpha', kappa None -> 1.0 for 'shift'
    variant="alpha", alpha=None, kappa=None, lam=1.0,
    # D2 / D3 class edges in time units (SCIENTIFIC_PLAN section 4)
    clone_age_edges_t=[0.01, 0.1, 1.0], cross_age_edges_t=[0.1, 1.0, 10.0],
)
GATEWAY_PHYSICAL_CHECKPOINTS_T = (0.5, 1, 2, 4, 10, 20, 40, 100, 400, 1000, 4000, 10000, 40000)
PROFILE_SNAPSHOT_U = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00)

# ---- integer-counter slots of the kernel state vector ``ictr`` ----
(I_SPTR, I_TPTR, I_NOPP, I_NOPP_EV, I_KD, I_KC, I_DEATHS, I_LR, I_RL, I_FIRST, I_NFE, I_STEP, I_NOPP_D,
 I_NPTR) = range(14)
N_ICTR = 14

NEW_DIAG_KEYS = ("fr_death_pos_hist_cum", "fr_birth_pos_hist_cum", "snap_step", "snap_t", "snap_u", "dep_age_C",
                 "dep_age_M", "dep_age_M2", "dep_cross_C", "traces_y")
# keys present only with diagnostics=True (omitted, not filled with initial values, when diagnostics=False)
DIAG_ONLY_KEYS = ("hist_inst", "region_frac", "events_cum", "lineage_visited", "first_arrival_step", "first_arrival_t",
                  "gen_nuniq_run", "gen_ess_run", "gen_maxfam_run", "gen_nuniq_win", "gen_ess_win", "gen_maxfam_win",
                  "gen_win_age_steps", "traces_step", "traces_t", "traces", "traces_anc", "traces_rebirth") \
    + NEW_DIAG_KEYS

# common (lta_ladder_numba) name -> this engine's key, for the quantities both engines save
COMMON_KEYS = dict(
    save_step="save_step", save_t="save_t", save_u="save_u", M_all="M_all", C_all="C_all", hist_inst="hist_inst",
    region_frac="region_frac",
    gen_n_unique="gen_nuniq_run", gen_ess="gen_ess_run", gen_max_frac="gen_maxfam_run",
    gen_n_unique_win="gen_nuniq_win", gen_ess_win="gen_ess_win", gen_max_frac_win="gen_maxfam_win",
    cum_deaths="fr_deaths_cum", cum_opps="fr_opp_cum", cum_opps_with_event="fr_opp_death_cum",
    events_per_opp_hist="fr_deaths_hist_cum",
    traces_step="traces_step", traces="traces", traces_anc="traces_anc",
    n_force_evals="n_force_evals", wall_s="wall_s", peak_rss_mb="peak_rss_mb",
    n_checkpoints_written="n_checkpoints_written", resumed_from="resumed_from",
)
_COMMON_INT = ("gen_n_unique", "gen_n_unique_win")
_COMMON_FLOAT = ("hist_inst",)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _engine_sha256():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git_info():
    rel = os.path.relpath(os.path.abspath(__file__), ROOT)
    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=20).stdout.strip() or "unknown"
        st = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", rel], capture_output=True, text=True,
                            timeout=20)
        if st.returncode != 0:
            return dict(git_commit=commit, git_dirty=None, git_status="")
        return dict(git_commit=commit, git_dirty=bool(st.stdout.strip()), git_status=st.stdout.strip())
    except Exception:  # pragma: no cover
        return dict(git_commit="unknown", git_dirty=None, git_status="")


_PROVENANCE = dict(engine_sha256=_engine_sha256(), **_git_info(),
                   imported_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class RunInterrupted(RuntimeError):
    """Raised by run_arm(stop_after_chunks=k): the run stopped after k chunks (state is in the checkpoint)."""


# ---------------------------------------------------------------------------------------------------------
# configuration and the model family
# ---------------------------------------------------------------------------------------------------------
def _edges(v, name):
    e = [float(a) for a in v]
    if len(e) != 3 or not all(math.isfinite(a) and a > 0 for a in e) or not (e[0] < e[1] < e[2]):
        raise ValueError(f"{name} must be 3 strictly increasing positive times (got {v!r})")
    return e


def full_cfg(cfg=None):
    """DEFAULT_CFG updated with cfg, with the model parameters normalised and validated: variant 'alpha' takes alpha
    (None -> 1.0) and refuses a kappa; variant 'shift' takes kappa > 0 (None -> 1.0) and refuses an alpha; lam > 0."""
    c = dict(DEFAULT_CFG)
    if cfg:
        c.update(cfg)
    v = c["variant"]
    if v == "alpha":
        if c["kappa"] is not None:
            raise ValueError("variant 'alpha' takes no kappa (pass kappa=None)")
        c["alpha"] = 1.0 if c["alpha"] is None else float(c["alpha"])
        if not math.isfinite(c["alpha"]):
            raise ValueError("alpha must be finite")
    elif v == "shift":
        if c["alpha"] is not None:
            raise ValueError("variant 'shift' takes no alpha (pass alpha=None)")
        c["kappa"] = 1.0 if c["kappa"] is None else float(c["kappa"])
        if not (math.isfinite(c["kappa"]) and c["kappa"] > 0):
            raise ValueError("kappa must be > 0")
    else:
        raise ValueError(f"variant must be 'alpha' or 'shift' (got {v!r})")
    c["lam"] = float(c["lam"])
    if not (math.isfinite(c["lam"]) and c["lam"] > 0):
        raise ValueError("lam must be > 0")
    c["clone_age_edges_t"] = _edges(c["clone_age_edges_t"], "clone_age_edges_t")
    c["cross_age_edges_t"] = _edges(c["cross_age_edges_t"], "cross_age_edges_t")
    return c


def model_params(cfg):
    """Kernel constants of the model (computed OUTSIDE the kernel) and the derived stiffness / relaxation facts."""
    c = full_cfg(cfg)
    beta, h, lam = float(c["beta"]), float(c["h"]), float(c["lam"])
    oout, oin = float(c["omega_out"]), float(c["omega_in"])
    if c["variant"] == "alpha":
        a = float(c["alpha"])
        mid = MODEL_ORIG if a == 1.0 else MODEL_ALPHA
        kappa, k2, msc = None, 0.0, 0.0
        k_gate, k_well = oin ** (2.0 * a), oout ** (2.0 * a)
    else:
        a = 1.0                                         # unused by MODEL_SHIFT
        mid = MODEL_SHIFT
        kappa = float(c["kappa"])
        k2 = kappa * kappa
        msc = math.sqrt(2.0 / beta) / kappa
        k_gate = k_well = k2
    k_max = max(k_gate, k_well)
    lk = lam * k_max * h / 2.0
    return dict(
        model_id=mid, model_name=MODEL_NAMES[mid], variant=c["variant"], alpha=(c["alpha"]), kappa=kappa, lam=lam,
        k_alpha=a, two_alpha=2.0 * a, c_ent=(1.0 - a) / beta, msc=msc, k2=k2, inv_beta=1.0 / beta,
        lam1=(lam == 1.0), amp_y=math.sqrt(2.0 * lam * h / beta),
        omega_power="pow (numba float ** float = libm pow; numpy ** in the reference / init)",
        stiffness_gate=k_gate, stiffness_well=k_well, stiffness_max=k_max,
        lam_stiffness_gate=lam * k_gate, tau_y_gate=1.0 / (lam * k_gate), tau_y2_gate=1.0 / (2.0 * lam * k_gate),
        em_var_inflation_max=(1.0 / (1.0 - lk) if lk < 1.0 else float("inf")),
        em_stable=bool(lam * k_max * h < 2.0),
    )


def _steps_edges(edges_t, h):
    """Class edges in integer steps: a = d h < e  <=>  d < ceil(e / h) (e / h rounded when within 1e-6 of an integer)."""
    out = []
    for e in edges_t:
        r = e / h
        out.append(int(round(r)) if abs(r - round(r)) < 1e-6 else int(math.ceil(r)))
    return np.asarray(out, dtype=np.int64)


# ---- numpy reference functions of the family (analysis / tests; the kernel has its own inline arithmetic) ----
def omega_np(cfg, x):
    c = full_cfg(cfg)
    x = np.asarray(x, dtype=np.float64)
    return c["omega_out"] + (c["omega_in"] - c["omega_out"]) * np.exp(-x * x / (2.0 * c["s"] ** 2))


def omega_prime_np(cfg, x):
    c = full_cfg(cfg)
    x = np.asarray(x, dtype=np.float64)
    return -(c["omega_in"] - c["omega_out"]) * (x / c["s"] ** 2) * np.exp(-x * x / (2.0 * c["s"] ** 2))


def F_star(cfg, x):
    c = full_cfg(cfg)
    x = np.asarray(x, dtype=np.float64)
    return c["H"] * (x * x - 1.0) ** 2 + np.log(omega_np(c, x)) / c["beta"]


def dF_star(cfg, x):
    c = full_cfg(cfg)
    x = np.asarray(x, dtype=np.float64)
    return 4.0 * c["H"] * x * (x * x - 1.0) + omega_prime_np(c, x) / omega_np(c, x) / c["beta"]


def m_shift(cfg, x):
    """Fibre centre m(x) of the shifted fibre (0 for the alpha family)."""
    c = full_cfg(cfg)
    if c["variant"] != "shift":
        return np.zeros_like(np.asarray(x, dtype=np.float64))
    return math.sqrt(2.0 / c["beta"]) / c["kappa"] * np.log(omega_np(c, x))


def stiffness(cfg, x):
    """Transverse stiffness d^2V/dy^2 at x: omega^(2 alpha) or kappa^2."""
    c = full_cfg(cfg)
    x = np.asarray(x, dtype=np.float64)
    if c["variant"] == "alpha":
        return omega_np(c, x) ** (2.0 * c["alpha"])
    return np.full_like(x, c["kappa"] ** 2)


def potential(cfg, x, y):
    c = full_cfg(cfg)
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    om = omega_np(c, x)
    if c["variant"] == "alpha":
        a = c["alpha"]
        return c["H"] * (x * x - 1.0) ** 2 + (1.0 - a) / c["beta"] * np.log(om) + 0.5 * om ** (2.0 * a) * y * y
    return F_star(c, x) + 0.5 * c["kappa"] ** 2 * (y - m_shift(c, x)) ** 2


def grad_V(cfg, x, y):
    """(dV/dx, dV/dy) analytically; f_x = dV/dx is the deposited local force (lam does not enter)."""
    c = full_cfg(cfg)
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    om, dom = omega_np(c, x), omega_prime_np(c, x)
    lr = dom / om
    base = 4.0 * c["H"] * x * (x * x - 1.0)
    if c["variant"] == "alpha":
        a = c["alpha"]
        k = om ** (2.0 * a)
        return base + (1.0 - a) / c["beta"] * lr + a * k * lr * y * y, k * y
    msc = math.sqrt(2.0 / c["beta"]) / c["kappa"]
    d = y - msc * np.log(om)
    k2 = c["kappa"] ** 2
    return base + lr / c["beta"] - k2 * d * msc * lr, k2 * d


def cond_y_law(cfg, x):
    """Exact Y | x: (mean, variance)."""
    c = full_cfg(cfg)
    return m_shift(c, x), 1.0 / (c["beta"] * stiffness(c, x))


def cond_force_var(cfg, x):
    """Exact Var(f_x | x): (2 alpha^2/beta^2)(omega'/omega)^2 or (2/beta^2)(omega'/omega)^2 (any kappa)."""
    c = full_cfg(cfg)
    lr = omega_prime_np(c, x) / omega_np(c, x)
    a2 = c["alpha"] ** 2 if c["variant"] == "alpha" else 1.0
    return 2.0 * a2 / c["beta"] ** 2 * lr * lr


def em_var_inflation(cfg, x):
    """Stationary EM y-variance at frozen x over the exact one: 1 / (1 - lam k h / 2)."""
    c = full_cfg(cfg)
    return 1.0 / (1.0 - c["lam"] * stiffness(c, x) * c["h"] / 2.0)


def em_mean_force(cfg, x):
    """Secondary reference (SCIENTIFIC_PLAN section 6): the frozen-x EM-consistent conditional mean force."""
    c = full_cfg(cfg)
    x = np.asarray(x, dtype=np.float64)
    if c["variant"] == "shift":
        return dF_star(c, x)
    a = c["alpha"]
    lr = omega_prime_np(c, x) / omega_np(c, x)
    return 4.0 * c["H"] * x * (x * x - 1.0) + (1.0 - a) / c["beta"] * lr + (a / c["beta"]) * lr * em_var_inflation(c, x)


# ---------------------------------------------------------------------------------------------------------
# helpers shared with gateway_ladder_numba (copies; equality asserted in the tests)
# ---------------------------------------------------------------------------------------------------------
def gaussian_kernel_np(bw, dx):
    r = max(1, int(round(4.0 * bw / dx)))
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t * dx / bw) ** 2)
    return k / (k.sum() * dx), r


def grid_dx():
    xg = np.linspace(XMIN, XMAX, N_GRID)
    return float(xg[1] - xg[0])


def _init_draws(seed, N):
    """init_left's draws: default_rng(1000 + seed); x ~ N(-1, 0.05) reflected, then z ~ N(0, 1)."""
    rng = np.random.default_rng(1000 + int(seed))
    x = rng.normal(-1.0, 0.05, N)
    span = XMAX - XMIN
    qm = np.remainder(x - XMIN, 2.0 * span)
    x = np.where(qm > span, 2.0 * span - qm, qm) + XMIN
    z = rng.normal(0.0, 1.0, N)
    return x, z


def init_left(seed, N, beta=16.0, oout=1.0, oin=32.0, s=0.1):
    """gateway_ladder_numba.init_left (copy): the original gateway's initial state."""
    x, z = _init_draws(seed, N)
    om = oout + (oin - oout) * np.exp(-x * x / (2.0 * s * s))
    y = z * np.sqrt(1.0 / (beta * om ** 2))
    return x.astype(np.float64), y.astype(np.float64)


def y_from_z(cfg, x, z):
    """Map standard normals z to the model's exact Y | x (alpha: z sqrt(1/(beta om^(2 alpha))), init_left's form,
    which is init_left's exact expression at alpha = 1; shift: m(x) + z sqrt(1/(beta kappa^2)))."""
    c = full_cfg(cfg)
    x = np.asarray(x, dtype=np.float64); z = np.asarray(z, dtype=np.float64)
    beta, s = c["beta"], c["s"]
    om = c["omega_out"] + (c["omega_in"] - c["omega_out"]) * np.exp(-x * x / (2.0 * s * s))
    if c["variant"] == "alpha":
        if c["alpha"] == 1.0:
            return z * np.sqrt(1.0 / (beta * om ** 2))
        return z * np.sqrt(1.0 / (beta * om ** (2.0 * c["alpha"])))
    kappa = c["kappa"]
    return math.sqrt(2.0 / beta) / kappa * np.log(om) + z * np.sqrt(1.0 / (beta * kappa ** 2))


def init_family(seed, N, cfg=None):
    """The coupled initial state of every cell: init_left's x and z, y = y_from_z(cfg, x, z).  At variant 'alpha',
    alpha 1 it is init_left bitwise (any lam)."""
    c = full_cfg(cfg)
    x, z = _init_draws(seed, N)
    y = y_from_z(c, x, z)
    return x.astype(np.float64), np.asarray(y, dtype=np.float64)


def default_noise_seed(seed, N):
    return int(1_000_003 * int(seed) + 7919 * int(N)) % (2 ** 31 - 1)


def fr_seed_entropy(seed, N):
    return [int(seed), int(N), 1]


def budget_save_grid(n_steps, n_lin=200, n_log=24, u_min=1e-4):
    """gateway_numba.budget_save_grid: completed-step counts at u = k/n_lin plus log-spaced u in [u_min, 1/n_lin)."""
    u_lin = np.arange(1, n_lin + 1) / n_lin
    u_log = np.logspace(np.log10(u_min), np.log10(1.0 / n_lin), n_log, endpoint=False)
    u = np.unique(np.concatenate([u_log, u_lin]))
    steps = np.unique(np.clip(np.round(u * n_steps).astype(np.int64), 1, n_steps))
    return steps, steps / float(n_steps)


def snapshot_grid(n_steps, profile_u=PROFILE_SNAPSHOT_U):
    """The profile-snapshot steps (prereg_save_grid's rule): min(max(round(u n), 1), n) for u in profile_u, unique."""
    n_steps = int(n_steps)
    return np.unique(np.asarray([min(max(int(round(u * n_steps)), 1), n_steps) for u in profile_u], dtype=np.int64))


def prereg_save_grid(n_steps, h, physical_t=GATEWAY_PHYSICAL_CHECKPOINTS_T, profile_u=PROFILE_SNAPSHOT_U):
    """gateway_ladder_numba.prereg_save_grid (copy): budget grid U physical checkpoints t <= T_N U profile snapshots."""
    n_steps = int(n_steps)
    steps, _ = budget_save_grid(n_steps)
    extra = [int(round(t / h)) for t in physical_t if 0 < round(t / h) <= n_steps]
    prof = [min(max(int(round(u * n_steps)), 1), n_steps) for u in profile_u]
    return np.unique(np.concatenate([steps, np.asarray(extra + prof, dtype=np.int64)]))


def default_trace_grid(n_steps, h, early_dt_t=0.1, early_until_t=40.0, late_dt_t=10.0, max_k=5000):
    """gateway_ladder_numba.default_trace_grid (copy)."""
    T = n_steps * h
    t_early = np.arange(0, int(math.floor(min(early_until_t, T) / early_dt_t + 1e-9)) + 1) * early_dt_t
    n_late_raw = int(math.floor((T - early_until_t) / late_dt_t + 1e-9)) if T > early_until_t else 0
    room = max(1, max_k - 1 - len(t_early))
    mult = max(1, int(math.ceil(n_late_raw / room))) if n_late_raw > 0 else 1
    ldt = late_dt_t * mult
    n_late = int(math.floor((T - early_until_t) / ldt + 1e-9)) if T > early_until_t else 0
    t_late = early_until_t + np.arange(1, n_late + 1) * ldt
    t = np.concatenate([t_early, t_late])
    steps = np.unique(np.clip(np.round(t / h).astype(np.int64), 0, n_steps))
    assert len(steps) < max_k
    return steps, ldt


def derived_knobs(cfg, N):
    """Physical-time knobs -> steps at the frozen h (raises if fr_interval_t is not an integer number of steps)."""
    h = float(cfg["h"])
    fr_every = int(round(cfg["fr_interval_t"] / h))
    if fr_every < 1 or abs(fr_every * h - cfg["fr_interval_t"]) > 1e-9 * max(1.0, cfg["fr_interval_t"]):
        raise ValueError(f"fr_interval_t {cfg['fr_interval_t']} is not an integer number of steps at h {h}")
    ramp_steps = float(cfg["ramp_t"]) / h if cfg["ramp_t"] > 0 else 0.0
    if ramp_steps > 0 and abs(ramp_steps - round(ramp_steps)) < 1e-6:
        ramp_steps = float(round(ramp_steps))
    window_steps = int(round(cfg["genealogy_window_t"] / h)) if cfg["genealogy_window_t"] > 0 else 0
    cap = max(int(cfg["cap_min"]), int(math.floor(cfg["max_event_fraction"] * N)))
    L = N + cap + max(N, cap)
    return dict(fr_every=fr_every, ramp_steps=ramp_steps, window_steps=window_steps, cap=cap, fr_block=L,
                dt_fr=h * fr_every)


# ---------------------------------------------------------------------------------------------------------
# numba MT19937 (np.random inside njit) state handling -- private API (gateway_ladder_numba's, copied)
# ---------------------------------------------------------------------------------------------------------
_MT_N = 624
_OFF_HAS_GAUSS = 4 + 4 * _MT_N
_OFF_IS_INIT = _OFF_HAS_GAUSS + 4 + 8
_MT_LAYOUT_OK = False


@njit(cache=True)
def _mt_seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _mt_normals(n):
    out = np.empty(n)
    for i in range(n):
        out[i] = np.random.standard_normal()
    return out


@njit(cache=True)
def _mt_uniforms(n):
    out = np.empty(n)
    for i in range(n):
        out[i] = np.random.random()
    return out


def _mt_ptr():
    return _helperlib.rnd_get_np_state_ptr()


def mt_has_gauss():
    return ctypes.c_int.from_address(_mt_ptr() + _OFF_HAS_GAUSS).value


def check_mt_layout():
    """Verify the assumed rnd_state_t layout (has_gauss / is_initialized offsets) once per process."""
    global _MT_LAYOUT_OK
    if _MT_LAYOUT_OK:
        return True
    _mt_seed(12345)
    ok = (ctypes.c_int.from_address(_mt_ptr() + _OFF_IS_INIT).value == 1 and mt_has_gauss() == 0)
    _mt_normals(1)
    ok = ok and mt_has_gauss() == 1
    _mt_normals(1)
    ok = ok and mt_has_gauss() == 0
    _mt_normals(3)
    ok = ok and mt_has_gauss() == 1
    _mt_seed(12345)
    ok = ok and mt_has_gauss() == 0
    if not ok:
        raise RuntimeError("numba rnd_state_t layout differs from the assumed one; refusing to checkpoint")
    _MT_LAYOUT_OK = True
    return True


def mt_get():
    """(index, key uint32[624]) of numba's np.random state in THIS thread; asserts no cached gaussian."""
    if mt_has_gauss() != 0:
        raise RuntimeError("numba MT state holds a cached gaussian at a chunk boundary (odd normal count)")
    idx, key = _helperlib.rnd_get_state(_mt_ptr())
    return int(idx), np.asarray(key, dtype=np.uint32)


def mt_set(idx, key):
    _helperlib.rnd_set_state(_mt_ptr(), (int(idx), [int(v) for v in np.asarray(key, dtype=np.uint32)]))


# ---------------------------------------------------------------------------------------------------------
# numba kernels (copied from gateway_ladder_numba; the force block and the y update are extended per model)
# ---------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _reflect(q, lo, hi):
    span = hi - lo
    qm = (q - lo) % (2.0 * span)
    if qm > span:
        qm = 2.0 * span - qm
    return qm + lo


@njit(cache=True)
def kde_density(X, n, kern, r, dx, p_out, hist):
    """gateway_numba.kde_density (eb_abffr_core.binned_density) for the first n entries of X."""
    G = N_GRID
    for k in range(G):
        hist[k] = 0.0
        p_out[k] = 0.0
    for i in range(n):
        k = int(np.rint((X[i] - XMIN) / dx))
        if k < 0:
            k = 0
        elif k > G - 1:
            k = G - 1
        hist[k] += 1.0
    pad = r if r < G - 1 else G - 1
    off = r - pad
    for k in range(G):
        c = hist[k]
        if c == 0.0:
            continue
        for img in range(3):
            if img == 0:
                o = k
            elif img == 1:
                if k < 1 or k > pad:
                    continue
                o = -k
            else:
                if k > G - 2 or k < G - 1 - pad:
                    continue
                o = 2 * (G - 1) - k
            lo_i = o - pad
            if lo_i < 0:
                lo_i = 0
            hi_i = o + pad
            if hi_i > G - 1:
                hi_i = G - 1
            for i in range(lo_i, hi_i + 1):
                m = o - i + pad
                p_out[i] += c * kern[off + m]
    inv_n = 1.0 / n
    s = 0.0
    for k in range(G):
        p_out[k] *= inv_n
        s += p_out[k]
    mass = dx * (s - 0.5 * (p_out[0] + p_out[G - 1]))
    if mass < EPS:
        mass = EPS
    for k in range(G):
        v = p_out[k] / mass
        p_out[k] = v if v > EPS else EPS
    return mass


@njit(cache=True)
def uniform_scores(X, n, p, dx, clip, S_out):
    """gateway_numba.uniform_scores: S = log p(x) - log q - KL(p || q), linear interpolation, clipped."""
    G = N_GRID
    q = 1.0 / (dx * (G - 1))
    logq = math.log(q)
    kl = 0.0
    prev = p[0] * (math.log(p[0]) - logq)
    for k in range(1, G):
        cur = p[k] * (math.log(p[k]) - logq)
        kl += 0.5 * (cur + prev) * dx
        prev = cur
    for i in range(n):
        pos = (X[i] - XMIN) / dx
        if pos < 0.0:
            pos = 0.0
        elif pos > G - 1.0:
            pos = G - 1.0
        i0 = int(math.floor(pos))
        if i0 > G - 2:
            i0 = G - 2
        frac = pos - i0
        v = p[i0] + frac * (p[i0 + 1] - p[i0])
        if v < EPS:
            v = EPS
        s = math.log(v) - logq - kl
        if s > clip:
            s = clip
        elif s < -clip:
            s = -clip
        S_out[i] = s
    return kl


@njit(cache=True)
def fr_resample(S, n, g, dt_fr, cap, sel, die_c, clone_c, pool, ubuf, u0, internal):
    """gateway_ladder_numba.fr_resample (copy): uniforms read sequentially from ubuf[u0:] (or np.random if
    ``internal``); writes the gather index into sel (new = old[sel]); returns (kd, kc, n_uniforms_used)."""
    k = u0
    nd = 0
    nc = 0
    for i in range(n):
        if internal:
            u = np.random.random()
        else:
            u = ubuf[k]
        k += 1
        s = S[i]
        if s > 0.0:
            pd = 1.0 - math.exp(-g * s * dt_fr)
            if u < pd:
                die_c[nd] = i
                nd += 1
        elif s < 0.0:
            pc = 1.0 - math.exp(g * s * dt_fr)
            if u < pc:
                clone_c[nc] = i
                nc += 1
    nev = nd + nc
    if nev > cap:
        den = nev if nev > 1 else 1
        kd = int(np.rint(cap * nd / den))
        if kd > nd:
            kd = nd
        kc = cap - kd
        if kc > nc:
            kc = nc
    else:
        kd = nd
        kc = nc
    if kd + kc == 0:
        return 0, 0, k - u0
    for t in range(kd):
        if internal:
            u = np.random.random()
        else:
            u = ubuf[k]
        k += 1
        j = t + int(u * (nd - t))
        if j > nd - 1:
            j = nd - 1
        tmp = die_c[t]; die_c[t] = die_c[j]; die_c[j] = tmp
    for t in range(kc):
        if internal:
            u = np.random.random()
        else:
            u = ubuf[k]
        k += 1
        j = t + int(u * (nc - t))
        if j > nc - 1:
            j = nc - 1
        tmp = clone_c[t]; clone_c[t] = clone_c[j]; clone_c[j] = tmp
    for i in range(n):
        sel[i] = 0
    for t in range(kd):
        sel[die_c[t]] = 1
    P = 0
    for i in range(n):
        if sel[i] == 0:
            pool[P] = i
            P += 1
    n_surv = P
    for t in range(kc):
        pool[P] = clone_c[t]
        P += 1
    if P >= n:
        for t in range(n):
            if internal:
                u = np.random.random()
            else:
                u = ubuf[k]
            k += 1
            j = t + int(u * (P - t))
            if j > P - 1:
                j = P - 1
            tmp = pool[t]; pool[t] = pool[j]; pool[j] = tmp
        for t in range(n):
            sel[t] = pool[t]
    else:
        for t in range(P):
            sel[t] = pool[t]
        for t in range(P, n):
            if internal:
                u = np.random.random()
            else:
                u = ubuf[k]
            k += 1
            j = int(u * n_surv)
            if j > n_surv - 1:
                j = n_surv - 1
            sel[t] = pool[j]
    return kd, kc, k - u0


@njit(cache=True)
def _bin_of(x, delta, nb):
    j = int(math.floor((x - XMIN) / delta + 1e-9))
    if j < 0:
        j = 0
    elif j > nb - 1:
        j = nb - 1
    return j


@njit(cache=True)
def _age_class(d_or_neg, ref_set, thr):
    """Class of an age d = k - ref (steps) against thr[0..2]; 4 = never (ref < 0)."""
    if not ref_set:
        return 4
    d = d_or_neg
    if d < thr[0]:
        return 0
    if d < thr[1]:
        return 1
    if d < thr[2]:
        return 2
    return 3


@njit(cache=True)
def _family_stats(lab, N, cnt):
    for i in range(N):
        cnt[i] = 0
    for i in range(N):
        cnt[lab[i]] += 1
    ss = 0.0
    mx = 0
    nz = 0
    for i in range(N):
        c = cnt[i]
        if c > 0:
            nz += 1
            ss += float(c) * float(c)
            if c > mx:
                mx = c
    return nz, (N * N / ss) / N, mx / N


@njit(cache=True)
def _rec_save(ptr, N, nb, is_fr, diag, X, Mh, Ch, anc_w, anc_r, visited, ictr, ev_hist, dh_hist, dpos, bpos,
              o_M, o_C, o_hist, o_reg, o_ev, o_vis, o_gen, o_fr, o_evh, o_dth, o_dpos, o_bpos, cnt):
    delta = (XMAX - XMIN) / nb
    for j in range(nb):
        o_M[ptr, j] = Mh[j]
        o_C[ptr, j] = Ch[j]
    o_fr[ptr, 0] = ictr[I_DEATHS]
    o_fr[ptr, 1] = ictr[I_KD]
    o_fr[ptr, 2] = ictr[I_KC]
    o_fr[ptr, 3] = ictr[I_NOPP]
    o_fr[ptr, 4] = ictr[I_NOPP_EV]
    o_fr[ptr, 5] = ictr[I_NOPP_D]
    for k in range(ev_hist.shape[0]):
        o_evh[ptr, k] = ev_hist[k]
        o_dth[ptr, k] = dh_hist[k]
    if not diag:
        return
    for j in range(nb):
        o_dpos[ptr, j] = dpos[j]
        o_bpos[ptr, j] = bpos[j]
    nl = 0
    nr = 0
    nv = 0
    for i in range(N):
        x = X[i]
        o_hist[ptr, _bin_of(x, delta, nb)] += 1
        if x < -X_BASIN:
            nl += 1
        elif x > X_BASIN:
            nr += 1
        if visited[i] != 0:
            nv += 1
    o_reg[ptr, 0] = nl / N
    o_reg[ptr, 1] = (N - nl - nr) / N
    o_reg[ptr, 2] = nr / N
    o_ev[ptr, 0] = ictr[I_LR]
    o_ev[ptr, 1] = ictr[I_RL]
    o_vis[ptr, 0] = nv
    if is_fr:
        nz, ess, mf = _family_stats(anc_r, N, cnt)
        o_gen[ptr, 0] = nz; o_gen[ptr, 1] = ess; o_gen[ptr, 2] = mf
        nz, ess, mf = _family_stats(anc_w, N, cnt)
        o_gen[ptr, 3] = nz; o_gen[ptr, 4] = ess; o_gen[ptr, 5] = mf
    else:
        o_gen[ptr, 0] = N; o_gen[ptr, 3] = N


@njit(cache=True)
def _rec_trace(ptr, n_tr, N, X, Y, anc_r, wid, wgen, inv, tr_x, tr_y, tr_anc, tr_gen):
    """Trace column k = the walker with persistent id k (ids are a permutation of 0..N-1)."""
    for i in range(N):
        w = wid[i]
        if w < n_tr:
            inv[w] = i
    for k in range(n_tr):
        i = inv[k]
        tr_x[ptr, k] = X[i]
        tr_y[ptr, k] = Y[i]
        tr_anc[ptr, k] = anc_r[i]
        tr_gen[ptr, k] = wgen[k]


@njit(cache=True)
def _rec_snap(ptr, dagC, dagM, dagM2, dcrC, o_sC, o_sM, o_sM2, o_sX):
    for a in range(dagC.shape[0]):
        for j in range(dagC.shape[1]):
            o_sC[ptr, a, j] = dagC[a, j]
            o_sM[ptr, a, j] = dagM[a, j]
            o_sM2[ptr, a, j] = dagM2[a, j]
            o_sX[ptr, a, j] = dcrC[a, j]


@njit(cache=True)
def _record_initial(N, nb, is_fr, diag, n_tr, save_at, trace_at, snap_at, X, Y, Mh, Ch, anc_w, anc_r, visited, wid,
                    wgen, ictr, ev_hist, dh_hist, dpos, bpos, dagC, dagM, dagM2, dcrC,
                    o_M, o_C, o_hist, o_reg, o_ev, o_vis, o_gen, o_fr, o_evh, o_dth, o_dpos, o_bpos,
                    tr_x, tr_y, tr_anc, tr_gen, o_sC, o_sM, o_sM2, o_sX):
    """Saves / traces / snapshots requested at completed-step count 0 (the initial state)."""
    cnt = np.zeros(N, dtype=np.int64)
    inv = np.zeros(max(n_tr, 1), dtype=np.int64)
    while ictr[I_SPTR] < save_at.shape[0] and save_at[ictr[I_SPTR]] == 0:
        _rec_save(ictr[I_SPTR], N, nb, is_fr, diag, X, Mh, Ch, anc_w, anc_r, visited, ictr, ev_hist, dh_hist,
                  dpos, bpos, o_M, o_C, o_hist, o_reg, o_ev, o_vis, o_gen, o_fr, o_evh, o_dth, o_dpos, o_bpos, cnt)
        ictr[I_SPTR] += 1
    while diag and ictr[I_TPTR] < trace_at.shape[0] and trace_at[ictr[I_TPTR]] == 0:
        _rec_trace(ictr[I_TPTR], n_tr, N, X, Y, anc_r, wid, wgen, inv, tr_x, tr_y, tr_anc, tr_gen)
        ictr[I_TPTR] += 1
    while diag and ictr[I_NPTR] < snap_at.shape[0] and snap_at[ictr[I_NPTR]] == 0:
        _rec_snap(ictr[I_NPTR], dagC, dagM, dagM2, dcrC, o_sC, o_sM, o_sM2, o_sX)
        ictr[I_NPTR] += 1


@njit(cache=True)
def _advance(step0, step1, N, h, beta, H, oout, oin, s, nb, min_count,
             model, alpha, two_alpha, c_ent, msc, k2, inv_beta, lam, lam1, amp_y, freeze_x,
             is_fr, gamma, ramp_steps, fr_every, score_clip, cap, kern, r_eta, dx, ubuf, ublk, fr_internal,
             diag, win_steps, save_at, trace_at, snap_at, n_tr, age_thr, crs_thr,
             X, Y, Mh, Ch, anc_w, anc_r, label, visited, wid, wgen, clone_step, cross_step, ictr, fctr, ev_hist,
             dh_hist, dpos, bpos, dagC, dagM, dagM2, dcrC,
             o_M, o_C, o_hist, o_reg, o_ev, o_vis, o_gen, o_fr, o_evh, o_dth, o_dpos, o_bpos,
             tr_x, tr_y, tr_anc, tr_gen, o_sC, o_sM, o_sM2, o_sX):
    """Advance one arm from completed step ``step0`` to ``step1`` in place (gateway_ladder_numba._advance, with the
    model-dependent force / y update and the D1-D4 diagnostics).  Everything under ``diag`` only reads the
    dynamics state."""
    fx = np.empty(N); fy = np.empty(N); jb = np.empty(N, dtype=np.int64)
    zx = np.empty(N); zy = np.empty(N)
    p = np.zeros(N_GRID); hist = np.zeros(N_GRID); S = np.empty(N)
    sel = np.empty(N, dtype=np.int64); die_c = np.empty(N, dtype=np.int64)
    clone_c = np.empty(N, dtype=np.int64); pool = np.empty(2 * N, dtype=np.int64)
    tmpx = np.empty(N); tmpy = np.empty(N)
    tmpi = np.empty(N, dtype=np.int64); ocnt = np.zeros(N, dtype=np.int64)
    cnt = np.zeros(N, dtype=np.int64)
    claimed = np.zeros(N, dtype=np.int64); freeid = np.empty(N, dtype=np.int64)
    inv = np.zeros(max(n_tr, 1), dtype=np.int64)
    n_saves = save_at.shape[0]
    n_traces = trace_at.shape[0]
    n_snaps = snap_at.shape[0]
    amp = math.sqrt(2.0 * h / beta)
    dt_fr = h * fr_every
    two_s2 = 2.0 * s * s
    s2 = s * s
    dom_amp = oin - oout
    delta = (XMAX - XMIN) / nb
    do_fr_arm = is_fr and N >= 2
    sptr = ictr[I_SPTR]
    tptr = ictr[I_TPTR]
    nptr = ictr[I_NPTR]
    mb = fctr[0]
    n_lr = ictr[I_LR]
    n_rl = ictr[I_RL]
    first = ictr[I_FIRST]
    kopp = 0  # opportunity index within this call (ubuf block)
    for step in range(step0, step1):
        if do_fr_arm and diag and win_steps > 0 and step % win_steps == 0:
            for i in range(N):
                anc_w[i] = i
        for i in range(N):
            zx[i] = np.random.standard_normal()
        for i in range(N):
            zy[i] = np.random.standard_normal()
        # --- deposit (all walkers before any bias read) ---
        for i in range(N):
            x = X[i]; y = Y[i]
            e = math.exp(-x * x / two_s2)
            om = oout + dom_amp * e
            dom = -dom_amp * (x / s2) * e
            if model == MODEL_ORIG:
                # the original gateway expressions, op for op
                f = 4.0 * H * x * (x * x - 1.0) + om * dom * y * y
                fx[i] = f
                fy[i] = om * om * y
            elif model == MODEL_ALPHA:
                lr = dom / om
                om2a = om ** two_alpha
                f = 4.0 * H * x * (x * x - 1.0) + c_ent * lr + alpha * om2a * lr * y * y
                fx[i] = f
                fy[i] = om2a * y
            else:
                lr = dom / om
                dyv = y - msc * math.log(om)
                f = 4.0 * H * x * (x * x - 1.0) + lr * inv_beta - k2 * dyv * (msc * lr)
                fx[i] = f
                fy[i] = k2 * dyv
            j = int(math.floor((x - XMIN) / delta + 1e-9))
            if j < 0:
                j = 0
            elif j > nb - 1:
                j = nb - 1
            jb[i] = j
            Ch[j] += 1.0
            Mh[j] += f
            if diag:
                cs = clone_step[i]
                a = _age_class(step - cs, cs >= 0, age_thr)
                dagC[a, j] += 1.0
                dagM[a, j] += f
                dagM2[a, j] += f * f
                xs = cross_step[i]
                b = _age_class(step - xs, xs >= 0, crs_thr)
                dcrC[b, j] += 1.0
        ictr[I_NFE] += N
        # --- move ---
        for i in range(N):
            j = jb[i]
            den = Ch[j] + min_count
            bias = Mh[j] / den if den > 0.0 else 0.0
            if abs(bias) > mb:
                mb = abs(bias)
            xn = _reflect(X[i] + (-fx[i] + bias) * h + amp * zx[i], XMIN, XMAX)
            if freeze_x:
                xn = X[i]
            X[i] = xn
            if lam1:
                Y[i] = Y[i] - fy[i] * h + amp * zy[i]
            else:
                Y[i] = Y[i] - lam * fy[i] * h + amp_y * zy[i]
            if diag:
                if xn > X_BASIN:
                    if label[i] == -1:
                        n_lr += 1
                        cross_step[i] = step + 1
                    label[i] = 1
                    visited[i] = 1
                    if first < 0:
                        first = step + 1
                elif xn < -X_BASIN:
                    if label[i] == 1:
                        n_rl += 1
                        cross_step[i] = step + 1
                    label[i] = -1
        # --- uniform FR on the post-move positions ---
        if do_fr_arm and (step % fr_every) == 0:
            if ramp_steps > 0:
                g = gamma * (1.0 - math.exp(-step / ramp_steps))
            else:
                g = gamma
            kde_density(X, N, kern, r_eta, dx, p, hist)
            uniform_scores(X, N, p, dx, score_clip, S)
            kd, kc, nu = fr_resample(S, N, g, dt_fr, cap, sel, die_c, clone_c, pool, ubuf, kopp * ublk, fr_internal)
            kopp += 1
            ictr[I_NOPP] += 1
            ev_hist[kd + kc] += 1
            nde = 0
            if kd + kc > 0:
                ictr[I_NOPP_EV] += 1
                ictr[I_KD] += kd
                ictr[I_KC] += kc
                # offspring counts (integer, from sel only; computed before the gather so that D1 can read the
                # pre-gather positions -- no floating-point operation of the dynamics moves)
                for i in range(N):
                    ocnt[i] = 0
                for i in range(N):
                    ocnt[sel[i]] += 1
                if diag:
                    # D1: realised deaths at their pre-gather x; births (offspring - 1) at the source's x
                    for i in range(N):
                        oc = ocnt[i]
                        if oc == 0:
                            dpos[_bin_of(X[i], delta, nb)] += 1
                        elif oc >= 2:
                            bpos[_bin_of(X[i], delta, nb)] += oc - 1
                for i in range(N):
                    tmpx[i] = X[sel[i]]; tmpy[i] = Y[sel[i]]
                for i in range(N):
                    X[i] = tmpx[i]; Y[i] = tmpy[i]
                for i in range(N):
                    if ocnt[i] == 0:
                        nde += 1
                ictr[I_DEATHS] += nde
                if nde > 0:
                    ictr[I_NOPP_D] += 1
                if diag:
                    for i in range(N):
                        tmpi[i] = anc_w[sel[i]]
                    for i in range(N):
                        anc_w[i] = tmpi[i]
                    for i in range(N):
                        tmpi[i] = anc_r[sel[i]]
                    for i in range(N):
                        anc_r[i] = tmpi[i]
                    for i in range(N):
                        tmpi[i] = label[sel[i]]
                    for i in range(N):
                        label[i] = tmpi[i]
                    for i in range(N):
                        tmpi[i] = visited[sel[i]]
                    for i in range(N):
                        visited[i] = tmpi[i]
                    # D2: a slot whose source has >= 2 offspring is (re)marked; others inherit
                    for i in range(N):
                        src = sel[i]
                        if ocnt[src] >= 2:
                            tmpi[i] = step + 1
                        else:
                            tmpi[i] = clone_step[src]
                    for i in range(N):
                        clone_step[i] = tmpi[i]
                    # D3: clones inherit the time of the last crossing (with the label)
                    for i in range(N):
                        tmpi[i] = cross_step[sel[i]]
                    for i in range(N):
                        cross_step[i] = tmpi[i]
                    # persistent walker ids (unchanged)
                    for i in range(N):
                        claimed[i] = 0
                    for i in range(N):
                        src = sel[i]
                        if claimed[src] == 0:
                            claimed[src] = 1
                            tmpi[i] = wid[src]
                        else:
                            tmpi[i] = -1
                    nf = 0
                    for i in range(N):
                        if claimed[i] == 0:
                            freeid[nf] = wid[i]
                            nf += 1
                    q = 0
                    for i in range(N):
                        if tmpi[i] < 0:
                            w = freeid[q]
                            q += 1
                            tmpi[i] = w
                            wgen[w] += 1
                    for i in range(N):
                        wid[i] = tmpi[i]
            if nde > cap:
                raise RuntimeError("realised deaths exceed the cap at one FR opportunity")
            dh_hist[nde] += 1
        # --- traces, snapshots and saves after step + 1 completed steps ---
        if diag:
            while tptr < n_traces and trace_at[tptr] == step + 1:
                _rec_trace(tptr, n_tr, N, X, Y, anc_r, wid, wgen, inv, tr_x, tr_y, tr_anc, tr_gen)
                tptr += 1
            while nptr < n_snaps and snap_at[nptr] == step + 1:
                _rec_snap(nptr, dagC, dagM, dagM2, dcrC, o_sC, o_sM, o_sM2, o_sX)
                nptr += 1
        if sptr < n_saves and save_at[sptr] == step + 1:
            ictr[I_LR] = n_lr; ictr[I_RL] = n_rl
            while sptr < n_saves and save_at[sptr] == step + 1:
                _rec_save(sptr, N, nb, is_fr, diag, X, Mh, Ch, anc_w, anc_r, visited, ictr, ev_hist, dh_hist,
                          dpos, bpos, o_M, o_C, o_hist, o_reg, o_ev, o_vis, o_gen, o_fr, o_evh, o_dth, o_dpos,
                          o_bpos, cnt)
                sptr += 1
    ictr[I_SPTR] = sptr
    ictr[I_TPTR] = tptr
    ictr[I_NPTR] = nptr
    ictr[I_LR] = n_lr
    ictr[I_RL] = n_rl
    ictr[I_FIRST] = first
    if step1 > step0:
        ictr[I_STEP] = step1
    fctr[0] = mb


# ---------------------------------------------------------------------------------------------------------
# Python driver: state, chunks, checkpoints, result contract
# ---------------------------------------------------------------------------------------------------------
_STATE_ARRAYS = ("X", "Y", "Mh", "Ch", "anc_w", "anc_r", "label", "visited", "wid", "wgen", "clone_step",
                 "cross_step", "ictr", "fctr", "ev_hist", "dh_hist", "dpos", "bpos", "dagC", "dagM", "dagM2", "dcrC",
                 "o_M", "o_C", "o_hist", "o_reg", "o_ev", "o_vis", "o_gen", "o_fr", "o_evh", "o_dth", "o_dpos",
                 "o_bpos", "tr_x", "tr_y", "tr_anc", "tr_gen", "o_sC", "o_sM", "o_sM2", "o_sX")


def _git_commit():
    return _git_info()["git_commit"]


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _proc_status_mb(field):
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _peak_rss_mb():
    v = _proc_status_mb("VmHWM")
    return v if v is not None else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _reset_peak_rss():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def _proc_start_epoch():
    try:
        with open("/proc/self/stat") as f:
            s = f.read()
        ticks = int(s[s.rindex(")") + 2:].split()[19])
        with open("/proc/stat") as f:
            btime = next(int(line.split()[1]) for line in f if line.startswith("btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


def _atomic_savez(path, compressed=False, **arrays):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "." + os.path.basename(path) + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        (np.savez_compressed if compressed else np.savez)(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _check_grid(a, n_steps, name):
    """Strictly increasing integer completed-step counts in [0, n_steps] (integral floats accepted, never truncated)."""
    a = np.asarray(a).ravel()
    if a.size and not np.issubdtype(a.dtype, np.integer):
        if (not np.issubdtype(a.dtype, np.floating) or not np.all(np.isfinite(a))
                or not np.all(a == np.rint(a))):
            raise ValueError(f"{name} must be integer step counts (dtype {a.dtype}, first values {a[:3].tolist()}); "
                             f"round explicitly, e.g. np.rint(t / h), instead of relying on truncation")
    a = a.astype(np.int64)
    if a.size and (np.any(np.diff(a) <= 0) or a[0] < 0 or a[-1] > n_steps):
        raise ValueError(f"{name} must be strictly increasing completed-step counts in [0, n_steps]")
    return a


def _model_sig(c):
    mp = model_params(c)
    return dict(variant=c["variant"], alpha=c["alpha"], kappa=c["kappa"], lam=c["lam"], model_id=mp["model_id"])


def _prepare(cfg, method, N, seed, n_steps, save_steps, trace_steps=None, noise_seed=None, diagnostics=True,
             x0=None, y0=None, fr_internal=False, snapshot_steps=None, freeze_x=False):
    """Validate the request and build everything that defines the run, including its signature."""
    c = full_cfg(cfg)
    if method not in ("abf", "fr"):
        raise ValueError("method must be 'abf' or 'fr'")
    N = int(N); n_steps = int(n_steps); seed = int(seed)
    if N < 1 or n_steps < 1:
        raise ValueError("N and n_steps must be >= 1")
    is_fr = method == "fr"
    if is_fr and N < 2:
        raise ValueError("no FR at N = 1 (the 'fr' arm needs N >= 2)")
    kn = derived_knobs(c, N)
    if is_fr and kn["cap"] >= N:
        raise ValueError(f"FR cap {kn['cap']} >= N {N} allows total extinction")
    mp = model_params(c)
    h = float(c["h"])
    if noise_seed is None:
        noise_seed = default_noise_seed(seed, N)
    noise_seed = int(noise_seed)
    save_at = _check_grid(save_steps, n_steps, "save_steps")
    trace_late_dt = trace_mult = None
    if trace_steps is None:
        trace_steps, trace_late_dt = default_trace_grid(n_steps, h, c["trace_early_dt_t"], c["trace_early_until_t"],
                                                        c["trace_late_dt_t"], int(c["trace_max"]))
        trace_mult = int(round(trace_late_dt / c["trace_late_dt_t"]))
    trace_at = _check_grid(trace_steps, n_steps, "trace_steps")
    if snapshot_steps is None:
        snapshot_steps = snapshot_grid(n_steps)
    snap_at = _check_grid(snapshot_steps, n_steps, "snapshot_steps")
    if x0 is None or y0 is None:
        x0, y0 = init_family(seed, N, c)
    x0 = np.ascontiguousarray(x0, dtype=np.float64); y0 = np.ascontiguousarray(y0, dtype=np.float64)
    if x0.shape != (N,) or y0.shape != (N,):
        raise ValueError("x0 / y0 must have shape (N,)")
    diag = bool(diagnostics)
    sig = dict(engine=ENGINE_VERSION, cfg=c, model=_model_sig(c), method=method, N=N, seed=seed, n_steps=n_steps,
               noise_seed=noise_seed, fr_entropy=fr_seed_entropy(seed, N), save_sha=_sha(save_at),
               trace_sha=_sha(trace_at), snap_sha=_sha(snap_at), x0_sha=_sha(x0), y0_sha=_sha(y0), diag=diag,
               fr_internal=bool(fr_internal), freeze_x=bool(freeze_x), engine_sha=_PROVENANCE["engine_sha256"])
    sig = json.loads(json.dumps(sig, sort_keys=True))
    return dict(c=c, kn=kn, mp=mp, h=h, N=N, seed=seed, n_steps=n_steps, is_fr=is_fr, noise_seed=noise_seed,
                save_at=save_at, trace_at=trace_at, snap_at=snap_at, trace_late_dt=trace_late_dt,
                trace_mult=trace_mult, x0=x0, y0=y0, diag=diag, fr_internal=bool(fr_internal),
                freeze_x=bool(freeze_x), sig=sig)


def run_signature(cfg, method, N, seed, n_steps, save_steps, *, trace_steps=None, noise_seed=None,
                  diagnostics=True, x0=None, y0=None, snapshot_steps=None, _fr_internal_rng=False,
                  _freeze_x_test=False):
    """The signature run_arm stores in checkpoints and in meta.run_signature for this request."""
    return _prepare(cfg, method, N, seed, n_steps, save_steps, trace_steps, noise_seed, diagnostics, x0, y0,
                    _fr_internal_rng, snapshot_steps, _freeze_x_test)["sig"]


def signature_mismatch(old, new, allow_engine_change=False):
    """Sorted list of the signature fields that differ (['run_signature missing'] if old is None)."""
    if not isinstance(old, dict):
        return ["run_signature missing"]
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)
                  and not (allow_engine_change and k == "engine_sha"))


def run_arm(cfg, method, N, seed, n_steps, save_steps, *, trace_steps=None, noise_seed=None, snapshot_steps=None,
            chunk_steps=None, checkpoint_path=None, checkpoint_every_s=600.0, stop_after_chunks=None,
            diagnostics=True, x0=None, y0=None, allow_engine_change=False, _fr_internal_rng=False,
            _freeze_x_test=False):
    """Run ONE arm (method 'abf' or 'fr') of the (cfg-model, N, seed) job for n_steps steps; returns the result dict.

    gateway_ladder_numba.run_arm's contract, plus snapshot_steps (default snapshot_grid(n_steps): the D2/D3
    snapshot saves) and the test-only _freeze_x_test (x never moves)."""
    t_wall0 = time.monotonic()
    epoch0 = time.time()
    hwm_before = _proc_status_mb("VmHWM")
    peak_reset = _reset_peak_rss()
    rss_start = _proc_status_mb("VmRSS")
    fr_internal = bool(_fr_internal_rng)
    P = _prepare(cfg, method, N, seed, n_steps, save_steps, trace_steps, noise_seed, diagnostics, x0, y0, fr_internal,
                 snapshot_steps, _freeze_x_test)
    c, kn, mp, h, N, seed, n_steps, is_fr = P["c"], P["kn"], P["mp"], P["h"], P["N"], P["seed"], P["n_steps"], P["is_fr"]
    save_at, trace_at, snap_at, diag, sig = P["save_at"], P["trace_at"], P["snap_at"], P["diag"], P["sig"]
    x0, y0, noise_seed, freeze_x = P["x0"], P["y0"], P["noise_seed"], P["freeze_x"]
    nb = int(c["nb"])
    dx = grid_dx()
    kern, r_eta = gaussian_kernel_np(float(c["eta"]), dx)
    S, K, NS = len(save_at), len(trace_at), len(snap_at)
    n_tr = min(N, N_TRACE_MAX)
    cap, L = kn["cap"], kn["fr_block"]
    NC = N_AGE_CLASSES
    age_thr = _steps_edges(c["clone_age_edges_t"], h)
    crs_thr = _steps_edges(c["cross_age_edges_t"], h)
    if chunk_steps is None:
        chunk_steps = max(1, (1 << 23) // N)
    if int(chunk_steps) != chunk_steps or int(chunk_steps) < 1:
        raise ValueError(f"chunk_steps must be an integer >= 1 (got {chunk_steps!r})")
    chunk_steps = int(chunk_steps)
    check_mt_layout()
    sig_json = json.dumps(sig, sort_keys=True)
    session = dict(pid=os.getpid(), host=platform.node(), proc_start_epoch=_proc_start_epoch(), run_start_epoch=epoch0,
                   start_step=0, engine_sha256=_PROVENANCE["engine_sha256"], git_commit=_PROVENANCE["git_commit"],
                   git_dirty=_PROVENANCE["git_dirty"], vmrss_start_mb=rss_start, vmhwm_before_mb=hwm_before,
                   peak_reset=peak_reset,
                   last_ckpt_epoch=None, last_ckpt_step=None, end_epoch=None, end_step=None, peak_rss_mb=None,
                   wall_s=0.0)

    st = None
    resumed_from, wall_prev, n_ckpt, rss_prev, sessions = [], 0.0, 0, 0.0, []
    fr_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(fr_seed_entropy(seed, N))))
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        with np.load(checkpoint_path, allow_pickle=False) as z:
            old = json.loads(str(z["sig_json"]))
            mism = signature_mismatch(old, sig, allow_engine_change)
            if mism:
                raise ValueError(f"checkpoint {checkpoint_path} does not match this run: {mism}")
            st = {k: z[k].copy() for k in _STATE_ARRAYS}
            mt_idx, mt_key = int(z["mt_index"]), z["mt_key"].copy()
            fr_rng.bit_generator.state = json.loads(str(z["fr_state_json"]))
            resumed_from = [int(v) for v in z["resumed_from"]]
            wall_prev = float(z["wall_s"]); n_ckpt = int(z["n_checkpoints_written"]); rss_prev = float(z["peak_rss_mb"])
            sessions = json.loads(str(z["sessions_json"]))
        resumed_from.append(int(st["ictr"][I_STEP]))
    if st is None:
        lab0 = np.where(x0 < -X_BASIN, -1, np.where(x0 > X_BASIN, 1, 0)).astype(np.int64)
        st = dict(X=x0.copy(), Y=y0.copy(), Mh=np.zeros(nb), Ch=np.zeros(nb),
                  anc_w=np.arange(N, dtype=np.int64), anc_r=np.arange(N, dtype=np.int64),
                  label=lab0, visited=(x0 > X_BASIN).astype(np.int64),
                  wid=np.arange(N, dtype=np.int64), wgen=np.zeros(N, dtype=np.int64),
                  clone_step=np.full(N, -1, dtype=np.int64), cross_step=np.full(N, -1, dtype=np.int64),
                  ictr=np.zeros(N_ICTR, dtype=np.int64), fctr=np.zeros(1),
                  ev_hist=np.zeros(cap + 1, dtype=np.int64), dh_hist=np.zeros(cap + 1, dtype=np.int64),
                  dpos=np.zeros(nb, dtype=np.int64), bpos=np.zeros(nb, dtype=np.int64),
                  dagC=np.zeros((NC, nb)), dagM=np.zeros((NC, nb)), dagM2=np.zeros((NC, nb)), dcrC=np.zeros((NC, nb)),
                  o_M=np.zeros((S, nb)), o_C=np.zeros((S, nb)), o_hist=np.zeros((S, nb), dtype=np.int64),
                  o_reg=np.full((S, 3), np.nan), o_ev=np.zeros((S, 2), dtype=np.int64),
                  o_vis=np.zeros((S, 1), dtype=np.int64), o_gen=np.full((S, 6), np.nan),
                  o_fr=np.zeros((S, 6), dtype=np.int64), o_evh=np.zeros((S, cap + 1), dtype=np.int64),
                  o_dth=np.zeros((S, cap + 1), dtype=np.int64),
                  o_dpos=np.zeros((S, nb), dtype=np.int64), o_bpos=np.zeros((S, nb), dtype=np.int64),
                  tr_x=np.full((K, n_tr), np.nan), tr_y=np.full((K, n_tr), np.nan),
                  tr_anc=np.full((K, n_tr), -1, dtype=np.int64), tr_gen=np.full((K, n_tr), -1, dtype=np.int64),
                  o_sC=np.zeros((NS, NC, nb)), o_sM=np.zeros((NS, NC, nb)), o_sM2=np.zeros((NS, NC, nb)),
                  o_sX=np.zeros((NS, NC, nb)))
        st["ictr"][I_FIRST] = 0 if np.any(x0 > X_BASIN) else -1
        _mt_seed(noise_seed)
        mt_idx, mt_key = mt_get()
        _record_initial(N, nb, is_fr, diag, n_tr, save_at, trace_at, snap_at, st["X"], st["Y"], st["Mh"], st["Ch"],
                        st["anc_w"], st["anc_r"], st["visited"], st["wid"], st["wgen"], st["ictr"], st["ev_hist"],
                        st["dh_hist"], st["dpos"], st["bpos"], st["dagC"], st["dagM"], st["dagM2"], st["dcrC"],
                        st["o_M"], st["o_C"], st["o_hist"], st["o_reg"], st["o_ev"], st["o_vis"], st["o_gen"],
                        st["o_fr"], st["o_evh"], st["o_dth"], st["o_dpos"], st["o_bpos"], st["tr_x"], st["tr_y"],
                        st["tr_anc"], st["tr_gen"], st["o_sC"], st["o_sM"], st["o_sM2"], st["o_sX"])
    session["start_step"] = int(st["ictr"][I_STEP])
    sessions.append(session)

    def write_ckpt():
        nonlocal n_ckpt
        n_ckpt += 1
        now = time.time()
        session.update(last_ckpt_epoch=now, last_ckpt_step=int(st["ictr"][I_STEP]), peak_rss_mb=_peak_rss_mb(),
                       wall_s=time.monotonic() - t_wall0)
        rss = max(rss_prev, session["peak_rss_mb"])
        _atomic_savez(checkpoint_path, compressed=False, sig_json=np.array(sig_json), mt_index=np.int64(mt_idx),
                      mt_key=mt_key, fr_state_json=np.array(json.dumps(fr_rng.bit_generator.state)),
                      resumed_from=np.array(resumed_from, dtype=np.int64),
                      wall_s=np.float64(wall_prev + time.monotonic() - t_wall0),
                      n_checkpoints_written=np.int64(n_ckpt), peak_rss_mb=np.float64(rss),
                      sessions_json=np.array(json.dumps(sessions)),
                      **{k: st[k] for k in _STATE_ARRAYS})

    fe = kn["fr_every"]
    empty_u = np.zeros(0)
    t_last_ckpt = time.monotonic()
    n_chunks = 0
    step = int(st["ictr"][I_STEP])
    while step < n_steps:
        step1 = min(n_steps, step + chunk_steps)
        if is_fr and not fr_internal:
            n_opp = -(-step1 // fe) - (-(-step // fe))  # FR steps in [step, step1)
            ubuf = fr_rng.random(n_opp * L) if n_opp > 0 else empty_u
        else:
            ubuf = empty_u
        mt_set(mt_idx, mt_key)
        _advance(step, step1, N, h, float(c["beta"]), float(c["H"]), float(c["omega_out"]), float(c["omega_in"]),
                 float(c["s"]), nb, float(c["min_count"]),
                 int(mp["model_id"]), float(mp["k_alpha"]), float(mp["two_alpha"]), float(mp["c_ent"]),
                 float(mp["msc"]), float(mp["k2"]), float(mp["inv_beta"]), float(mp["lam"]), bool(mp["lam1"]),
                 float(mp["amp_y"]), bool(freeze_x),
                 is_fr, float(c["gamma"]), float(kn["ramp_steps"]), int(fe), float(c["score_clip"]), int(cap),
                 kern, int(r_eta), float(dx), ubuf, int(L), fr_internal,
                 diag, int(kn["window_steps"]), save_at, trace_at, snap_at, int(n_tr), age_thr, crs_thr,
                 st["X"], st["Y"], st["Mh"], st["Ch"], st["anc_w"], st["anc_r"], st["label"], st["visited"],
                 st["wid"], st["wgen"], st["clone_step"], st["cross_step"], st["ictr"], st["fctr"], st["ev_hist"],
                 st["dh_hist"], st["dpos"], st["bpos"], st["dagC"], st["dagM"], st["dagM2"], st["dcrC"],
                 st["o_M"], st["o_C"], st["o_hist"], st["o_reg"], st["o_ev"], st["o_vis"], st["o_gen"], st["o_fr"],
                 st["o_evh"], st["o_dth"], st["o_dpos"], st["o_bpos"],
                 st["tr_x"], st["tr_y"], st["tr_anc"], st["tr_gen"], st["o_sC"], st["o_sM"], st["o_sM2"], st["o_sX"])
        mt_idx, mt_key = mt_get()
        step = step1
        assert int(st["ictr"][I_STEP]) == step
        n_chunks += 1
        if checkpoint_path is not None and step < n_steps and (time.monotonic() - t_last_ckpt >= checkpoint_every_s):
            write_ckpt()
            t_last_ckpt = time.monotonic()
        if stop_after_chunks is not None and n_chunks >= stop_after_chunks and step < n_steps:
            raise RunInterrupted(f"stopped after {n_chunks} chunks at step {step}")

    ic = st["ictr"]
    if (int(ic[I_SPTR]) != S or (diag and (int(ic[I_TPTR]) != K or int(ic[I_NPTR]) != NS))
            or int(ic[I_NFE]) != N * n_steps):
        raise RuntimeError("run incomplete: not every save/trace/snapshot fired or the force count is wrong")
    wall = wall_prev + time.monotonic() - t_wall0
    session.update(end_epoch=time.time(), end_step=int(ic[I_STEP]), peak_rss_mb=_peak_rss_mb(),
                   wall_s=time.monotonic() - t_wall0)
    rss = max(rss_prev, session["peak_rss_mb"])
    lost = 0.0
    for a, b in zip(sessions[:-1], sessions[1:]):
        if a.get("last_ckpt_epoch") is not None:
            lost += max(0.0, b["run_start_epoch"] - a["last_ckpt_epoch"])
    W = int(kn["window_steps"])
    if W > 0:
        win_age = np.where(save_at >= 1, save_at - W * ((save_at - 1) // W), 0).astype(np.int64)
    else:
        win_age = save_at.astype(np.int64).copy()
    class_names_age = [f"[0,{c['clone_age_edges_t'][0]:g})", f"[{c['clone_age_edges_t'][0]:g},"
                       f"{c['clone_age_edges_t'][1]:g})", f"[{c['clone_age_edges_t'][1]:g},{c['clone_age_edges_t'][2]:g})",
                       f">={c['clone_age_edges_t'][2]:g}", "never"]
    class_names_crs = [f"[0,{c['cross_age_edges_t'][0]:g})", f"[{c['cross_age_edges_t'][0]:g},"
                       f"{c['cross_age_edges_t'][1]:g})", f"[{c['cross_age_edges_t'][1]:g},{c['cross_age_edges_t'][2]:g})",
                       f">={c['cross_age_edges_t'][2]:g}", "never"]
    mp_meta = {k: v for k, v in mp.items()}
    meta = dict(
        engine=ENGINE_VERSION, engine_sha256=_PROVENANCE["engine_sha256"], system="gateway_family", method=method,
        status="complete",
        N=N, seed=seed, n_steps=n_steps, B=N * n_steps, T=n_steps * h, h=h,
        model=mp_meta, variant=c["variant"], alpha=c["alpha"], kappa=c["kappa"], lam=c["lam"],
        stiffness_gate=mp["stiffness_gate"], stiffness_max=mp["stiffness_max"],
        noise_seed=noise_seed, noise_seed_formula="(1_000_003*seed + 7919*N) % (2**31-1) unless given",
        noise_stream="numba np.random MT19937 (Langevin normals only: N zx then N zy per step)",
        fr_stream="numpy Generator(PCG64(SeedSequence(fr_seedseq_entropy))); opportunity k owns doubles [kL,(k+1)L)",
        fr_seedseq_entropy=fr_seed_entropy(seed, N), fr_block_L=L, fr_internal_rng=fr_internal,
        freeze_x_test=freeze_x,
        init=("init_family: init_left's draws (default_rng(1000+seed): x ~ N(-1,0.05) reflected, then z ~ N(0,1)); "
              "y = z sqrt(1/(beta om^(2 alpha))) or m(x) + z sqrt(1/(beta kappa^2)); x0/y0 if given"),
        cap=cap, fr_every=kn["fr_every"], fr_interval_t=c["fr_interval_t"], dt_fr=kn["dt_fr"],
        ramp_steps=kn["ramp_steps"], ramp_t=c["ramp_t"], window_steps=kn["window_steps"],
        genealogy_window_t=c["genealogy_window_t"], gamma=c["gamma"], eta=c["eta"], score_clip=c["score_clip"],
        max_event_fraction=c["max_event_fraction"], cap_min=c["cap_min"], nb=nb, min_count=c["min_count"],
        cell=dict(beta=c["beta"], H=c["H"], omega_out=c["omega_out"], omega_in=c["omega_in"], s=c["s"]),
        n_saves=S, n_traces=K, n_trace_slots=n_tr, n_snapshots=NS, trace_late_dt_t=P["trace_late_dt"],
        trace_late_multiplier=P["trace_mult"], chunk_steps=chunk_steps,
        diagnostics=diag, omitted_keys=([] if diag else list(DIAG_ONLY_KEYS)),
        event_names=["trans_LR", "trans_RL"], visit_names=["right_well"],
        region_names=["left x<-0.5", "gate", "right x>0.5"], first_arrival_names=["right_well"],
        diag_classes=dict(
            clone_age_edges_t=list(c["clone_age_edges_t"]), clone_age_edges_steps=[int(v) for v in age_thr],
            clone_age_class_names=class_names_age,
            cross_age_edges_t=list(c["cross_age_edges_t"]), cross_age_edges_steps=[int(v) for v in crs_thr],
            cross_age_class_names=class_names_crs,
            rule="age d = k - ref steps (k = completed-step count of the deposited state); class i iff d < edge_steps[i]"
                 " (first such i), 3 if d >= edge_steps[2], 4 (never) if ref = -1; a < e <=> d < ceil(e/h)"),
        diag_definitions=dict(
            fr_death_pos_hist_cum="cumulative production-bin histogram of the pre-gather x of realised deaths "
                                  "(slots with zero offspring)",
            fr_birth_pos_hist_cum="cumulative production-bin histogram of realised births (offspring - 1 per source "
                                  "with >= 2 offspring) at the source's x; totals of both = fr_deaths_cum",
            clone_step="per walker: completed-step count of the last FR gather at which its source had >= 2 offspring "
                       "(source slot and all copies marked; others inherit; -1 never); NOT saved in the result",
            dep_age_C_M_M2="cumulative deposits (count, sum f, sum f^2) per clone-age class and bin at the snapshot "
                           "saves; sum over classes of dep_age_C = C_all; all-deposit sum f^2 = dep_age_M2.sum(class "
                           "axis) (not stored twice)",
            cross_step="per walker: completed-step count of its last true hysteresis-label flip (events_cum's); "
                       "clones inherit; -1 never; NOT saved in the result",
            dep_cross_C="cumulative deposit counts per time-since-crossing class and bin at the snapshot saves",
            traces_y="y of the traced walkers (columns as traces)",
            snap_step="snapshot saves (default: profile_snapshot_u fractions of n_steps, prereg_save_grid's rule)"),
        accumulator_convention=("pre-deposit (gateway_numba): the save at completed-step count k holds the deposits "
                                "of the states 0..k-1 (C_all[k].sum() = N k); the saved state's own deposit enters "
                                "the next save. lta_ladder_numba includes the saved step's deposit. The D2/D3 "
                                "snapshots follow the same convention."),
        force_eval_convention="N per completed step: n_force_evals = N n_steps (lta_ladder_numba: N (n_steps + 1))",
        event_definitions=dict(
            events_cum="true cross-well transitions [L->R, R->L] under the hysteresis label (x < -0.5 / x > 0.5)",
            fr_kd_cum="capped death candidates kd (gateway_numba die)",
            fr_kc_cum="capped clone candidates kc (gateway_numba clone)",
            fr_opp_event_cum="opportunities with kd + kc > 0 (candidates; may realise zero deaths)",
            fr_events_hist="histogram over opportunities of kd + kc, bins 0..cap (fr_events_hist_cum per save)",
            fr_deaths_cum="realised deaths = slots with zero offspring = realised births",
            fr_opp_death_cum="opportunities with >= 1 realised death",
            fr_deaths_hist=("histogram over opportunities of realised deaths, bins 0..cap "
                            "(fr_deaths_hist_cum per save)")),
        trace_semantics=("column j follows the walker with persistent id j (j < n_tr): ids are a permutation of "
                         "0..N-1; at an FR gather the first new slot holding a copy of an old walker keeps its id "
                         "and the ids of the dead (ascending old slot) pass to the remaining copies (ascending new "
                         "slot); traces_rebirth counts the deaths/rebirths of id j, the only places a trace jumps "
                         "(lta_ladder_numba's in-place slot-trace meaning). ABF: id = slot."),
        genealogy_window=("windowed labels reset at the START of every step multiple of window_steps (before that "
                          "step's FR); gen_win_age_steps = k - W floor((k - 1) / W) steps are covered at save k"),
        common_keys=dict(COMMON_KEYS),
        run_signature=sig, run_signature_sha256=hashlib.sha256(sig_json.encode()).hexdigest(),
        sessions=sessions, wall_lost_upper_s=lost,
        wall_s_definition=("sum over sessions of the time inside run_arm up to the session's last checkpoint or "
                           "completion; excludes work lost after the last checkpoint of a killed session "
                           "(bounded by wall_lost_upper_s) and process start-up before run_arm"),
        peak_rss_source=(("max over sessions of VmHWM, reset to the current RSS at run_arm entry "
                          "(/proc/self/clear_refs 5)" if all(x.get("peak_reset") for x in sessions) else
                          "max over sessions of VmHWM (NOT reset: includes earlier work of the process)")
                         if _proc_status_mb("VmHWM") is not None else "ru_maxrss (inherits a parent's peak)"),
        peak_rss_over_start_mb=max([x["peak_rss_mb"] - x["vmrss_start_mb"] for x in sessions
                                    if x.get("peak_rss_mb") is not None and x.get("vmrss_start_mb") is not None],
                                   default=None),
        git_commit=_PROVENANCE["git_commit"], git_dirty=_PROVENANCE["git_dirty"],
        provenance_note="engine_sha256 / git_commit / git_dirty are taken when the module is imported (per session "
                        "in sessions)",
        numba=numba.__version__, numpy=np.__version__, python=platform.python_version(),
        host=platform.node(), finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    first = int(ic[I_FIRST])
    res = dict(
        save_step=save_at.copy(), save_t=save_at * h, save_u=save_at / float(n_steps),
        M_all=st["o_M"], C_all=st["o_C"], hist_inst=st["o_hist"], region_frac=st["o_reg"],
        events_cum=st["o_ev"], lineage_visited=st["o_vis"],
        first_arrival_step=np.array([first], dtype=np.int64),
        first_arrival_t=np.array([first * h if first >= 0 else np.nan]),
        gen_nuniq_run=st["o_gen"][:, 0], gen_ess_run=st["o_gen"][:, 1], gen_maxfam_run=st["o_gen"][:, 2],
        gen_nuniq_win=st["o_gen"][:, 3], gen_ess_win=st["o_gen"][:, 4], gen_maxfam_win=st["o_gen"][:, 5],
        gen_win_age_steps=win_age,
        fr_deaths_cum=st["o_fr"][:, 0], fr_kd_cum=st["o_fr"][:, 1], fr_kc_cum=st["o_fr"][:, 2],
        fr_opp_cum=st["o_fr"][:, 3], fr_opp_event_cum=st["o_fr"][:, 4], fr_opp_death_cum=st["o_fr"][:, 5],
        fr_events_hist=st["ev_hist"], fr_events_hist_cum=st["o_evh"],
        fr_deaths_hist=st["dh_hist"], fr_deaths_hist_cum=st["o_dth"],
        traces_step=trace_at.copy(), traces_t=trace_at * h, traces=st["tr_x"], traces_anc=st["tr_anc"],
        traces_rebirth=st["tr_gen"],
        X_final=st["X"], Y_final=st["Y"], max_abs_bias=np.float64(st["fctr"][0]),
        n_fr_steps=np.int64(ic[I_NOPP]),
        n_force_evals=np.int64(ic[I_NFE]), wall_s=np.float64(wall), peak_rss_mb=np.float64(rss),
        n_checkpoints_written=np.int64(n_ckpt), resumed_from=np.array(resumed_from, dtype=np.int64),
        # ---- D1-D4 ----
        fr_death_pos_hist_cum=st["o_dpos"], fr_birth_pos_hist_cum=st["o_bpos"],
        snap_step=snap_at.copy(), snap_t=snap_at * h, snap_u=snap_at / float(n_steps),
        dep_age_C=st["o_sC"], dep_age_M=st["o_sM"], dep_age_M2=st["o_sM2"], dep_cross_C=st["o_sX"],
        traces_y=st["tr_y"],
        cfg=c, meta=meta,
    )
    if not diag:
        for k in DIAG_ONLY_KEYS:
            del res[k]
    return res


def common_view(res):
    """The result under the names and dtypes shared with lta_ladder_numba (COMMON_KEYS)."""
    out = {}
    for common, key in COMMON_KEYS.items():
        if key not in res:
            continue
        v = np.asarray(res[key])
        if common in _COMMON_INT:
            if not np.all(np.isfinite(v)):
                raise ValueError(f"{key} holds non-finite values; cannot view it as integer {common}")
            v = v.astype(np.int64)
        elif common in _COMMON_FLOAT:
            v = v.astype(np.float64)
        out[common] = v
    return out


# ---------------------------------------------------------------------------------------------------------
# result I/O (compressed npz, no pickle; cfg / meta as JSON strings)
# ---------------------------------------------------------------------------------------------------------
def save_result(path, res):
    """Atomically write a COMPLETE result (refuses a partial one)."""
    if res.get("meta", {}).get("status") != "complete":
        raise ValueError("refusing to save a result whose status is not 'complete'")
    arrays = {k: np.asarray(v) for k, v in res.items() if k not in ("cfg", "meta")}
    for k, v in arrays.items():
        if v.dtype == object:
            raise TypeError(f"{k} would need pickle")
    arrays["cfg_json"] = np.array(json.dumps(res["cfg"], sort_keys=True))
    arrays["meta_json"] = np.array(json.dumps(res["meta"], sort_keys=True))
    _atomic_savez(path, compressed=True, **arrays)


def load_result(path):
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k not in ("cfg_json", "meta_json")}
        out["cfg"] = json.loads(str(z["cfg_json"]))
        out["meta"] = json.loads(str(z["meta_json"]))
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray) and v.ndim == 0:
            out[k] = v[()]
    return out


def is_complete(path, signature=None, allow_engine_change=False):
    """True iff path holds a result with status 'complete' (and, if ``signature`` is given, a matching
    meta.run_signature)."""
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
    except Exception:
        return False
    if meta.get("status") != "complete":
        return False
    return signature is None or not signature_mismatch(meta.get("run_signature"), signature, allow_engine_change)


_SIG_KW = ("trace_steps", "noise_seed", "diagnostics", "x0", "y0", "snapshot_steps", "_fr_internal_rng",
           "_freeze_x_test")


def run_job(cfg, method, N, seed, n_steps, out_path, save_steps=None, checkpoint_every_s=600.0, **kw):
    """Runner convenience (gateway_ladder_numba.run_job's semantics): return an existing complete result IF its run
    signature matches this request (else raise); otherwise run (resuming from out_path + '.ckpt.npz'), save the
    result atomically, then delete the checkpoint.  Default save grid: prereg_save_grid(n_steps, h)."""
    if save_steps is None:
        save_steps = prereg_save_grid(n_steps, float(full_cfg(cfg)["h"]))
    if is_complete(out_path):
        old = load_result(out_path)
        want = run_signature(cfg, method, N, seed, n_steps, save_steps, **{k: kw[k] for k in _SIG_KW if k in kw})
        mism = signature_mismatch(old["meta"].get("run_signature"), want, bool(kw.get("allow_engine_change", False)))
        if mism:
            raise ValueError(f"{out_path} holds a complete result of a DIFFERENT run (fields {mism}); "
                             f"refusing to return it -- move it away or use another out_path")
        return old
    ckpt = out_path + ".ckpt.npz"
    res = run_arm(cfg, method, N, seed, n_steps, save_steps, checkpoint_path=ckpt,
                  checkpoint_every_s=checkpoint_every_s, **kw)
    save_result(out_path, res)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    return res
