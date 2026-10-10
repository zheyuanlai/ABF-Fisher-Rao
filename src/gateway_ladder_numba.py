"""Equal-budget replica-ladder engine for the entropic gateway: histogram ABF (+ uniform FR), ONE arm per call.

Equal-budget campaign (docs/equal_budget/), 2026-10-10.  The arithmetic of the dynamics, of the P0 histogram
estimator and of the FR operator is copied op for op from ``src/gateway_numba.py`` (spec with line refs:
docs/equal_budget/audit/AUDIT_0A_gateway.md); that file is NOT imported by the kernel and is NOT modified.
tests/test_gateway_ladder_numba.py proves the equivalences listed below.

Model and dynamics (unchanged from gateway_numba)
-------------------------------------------------
V(x, y) = H (x^2-1)^2 + omega(x)^2 y^2 / 2,  omega = w_out + (w_in - w_out) exp(-x^2 / (2 s^2)); overdamped
Euler-Maruyama at step h, unit mobility, x reflected into [-1.8, 1.8], y unbounded.  Per step:
  1. N standard normals zx (slot order), then N normals zy;
  2. DEPOSIT: every walker evaluates f_x at its pre-move (x, y) and deposits into its bin
     j = clamp(floor((x + 1.8)/delta + 1e-9), 0, nb-1):  C_j += 1, M_j += f_x   (one force evaluation per walker);
  3. MOVE: bias Gamma = M_j / (C_j + min_count) of the walker's OWN pre-move bin, read AFTER all deposits;
     x <- reflect(x + (-f_x + Gamma) h + sqrt(2h/beta) zx),  y <- y - omega^2 y h + sqrt(2h/beta) zy;
  4. FR (method 'fr', N >= 2, every fr_every steps starting at step 0, on the post-move positions): binned KDE
     (eta), uniform score S = log p(x) - log q - KL(p||q) clipped at +-score_clip, rate g = gamma (1 - exp(-step /
     ramp_steps)), death/clone candidates with prob 1 - exp(-+ g S dt_fr), cap = max(cap_min, floor(mef N)) on the
     total, pool law of gateway_numba.fr_resample; slots are gathered only when >= 1 event fires.

Deliberate differences from gateway_numba (and why)
---------------------------------------------------
* ONE arm per call (method 'abf' or 'fr'), on 1-D state arrays.
* Two random streams.  Langevin noise: numba's internal np.random MT19937, seeded ONCE with ``noise_seed``
  (default (1_000_003 seed + 7919 N) mod (2^31-1), gateway_numba's formula) and used for NOTHING else, so the ABF
  arm and the FR arm of one (N, seed), run in separate processes, see identical noise slot by slot.  FR
  uniforms: numpy ``Generator(PCG64(SeedSequence([seed, N, 1])))``; opportunity k (the k-th FR step) owns the
  fixed block of L = N + cap + max(N, cap) consecutive doubles [k L, (k+1) L) of that stream (L bounds the
  draws of one opportunity: N candidate tests + kd + kc Fisher-Yates picks + max(N, kd - kc) pool picks), of
  which the first n_used are read in gateway_numba's order.  A fixed block makes the stream position a
  function of k alone (chunk-size independent).  (gateway_numba draws normals and FR uniforms from ONE stream,
  so its arms are not bitwise separable -- audit section 3.)  Test-only switch ``_fr_internal_rng=True`` draws
  the FR uniforms from the Langevin MT stream exactly as gateway_numba does, which makes a single-FR-arm run
  BITWISE equal to gateway_numba (test Q2c).
* Explicit chunked state and atomic checkpoints.  numba's MT19937 state is saved/restored with the private
  ``numba._helperlib.rnd_get_state / rnd_set_state`` on the np-state pointer, restored immediately before EVERY
  kernel call in the calling thread (never reseeded on resume).  rnd_set_state clears the cached Box-Muller
  gaussian; that is exact because each step draws 2N normals (even), so the cache is empty at every step
  boundary -- verified after every chunk by reading ``has_gauss`` from the rnd_state_t struct (layout checked
  at first use).  The FR Generator is saved via ``bit_generator.state``.
* Diagnostics (below) only READ the dynamics state; ``diagnostics=False`` skips them and changes no bit.
  The persistent walker ids behind the traces recycle the ids of the dead walkers instead of the audit's
  next_id++ (AUDIT_0A_gateway.md section 5): the id space stays 0..N-1, a traced id never ends, and the trace
  has lta_ladder_numba's in-place meaning (a dead walker's place is taken by a copy).
* An FR call with cap >= N is refused: it would allow every walker to die, where gateway_numba.fr_resample
  reads pool[-1] (an uninitialised slot).  Unreachable with the production cap max(1, floor(0.08 N)) < N.
* fr_time_s is not reported (no clock inside the nopython kernel); FR cost shows in ns per walker-step.
* Physical-time knobs are given in time units in ``cfg`` and converted with the frozen h:
  fr_every = round(fr_interval_t / h) (must be an integer number of steps), ramp_steps = ramp_t / h,
  window_steps = round(genealogy_window_t / h).  No FR at N = 1 (an 'fr' call with N = 1 raises).

Result contract (run_arm -> dict; save_result/load_result <-> compressed .npz, no pickle)
---------------------------------------------------------------------------------------
S saves (the caller's sorted unique completed-step counts, 0 allowed), nb = 180 bins, K traces, n_tr = min(N, 32).
Grids are integer step counts; a non-integral value is refused, never truncated (pass np.rint(t / h)).
run_job's default grid is prereg_save_grid (budget grid + the frozen physical-time checkpoints t <= T_N).
  save_step (S,) int64, save_t = save_step h, save_u = save_step / n_steps
  M_all, C_all (S, nb)       ABF accumulators, gateway_numba's PRE-DEPOSIT convention (required for Q1): the save at
                             completed-step count k holds the deposits of the states 0..k-1, so C_all[k].sum() = N k;
                             the saved state is deposited by the next step, so it first appears in the save at
                             k + 1 (with saves at k and k + 1, C_all[k+1] - C_all[k] = hist_inst[k]).
                             lta_ladder_numba INCLUDES the saved step's deposit (and counts N (n_steps + 1) force
                             evaluations); meta.accumulator_convention records this.
  hist_inst (S, nb) int64    instantaneous histogram of the N walkers on the production bins (deposit bin rule)
  region_frac (S, 3)         instantaneous fractions left x<-0.5 / gate / right x>0.5
  events_cum (S, 2) int64    cumulative TRUE cross-well transitions [L->R, R->L]: hysteresis label -1 (x<-0.5) /
                             +1 (x>0.5), updated after each move; a flip counts when it happens; clones inherit the
                             label, so a copy is never a transition
  lineage_visited (S, 1)     number of slots whose lineage has ever been in the right well (flag set after a
                             move, inherited by clones)
  first_arrival_step (1,)    first completed-step count at which any walker was in the right well; -1 if never
  gen_{nuniq,ess,maxfam}_{run,win} (S,)  genealogy at saves: distinct ancestors, ESS/N = (sum c)^2/sum c^2 / N,
                             largest family / N; 'run' labels never reset, 'win' labels reset at the START of every
                             step multiple of window_steps (gateway_numba's ess_window rule, before that step's FR).
                             ABF arm: nuniq = N, ess/maxfam = NaN.
  gen_win_age_steps (S,)     steps covered by the windowed labels at the save: k - W floor((k - 1) / W) (k >= 1; a
                             save on a window boundary reports the full window W); compare gen_*_win across N only
                             at equal age
  fr_deaths_cum (S,)         REALISED deaths (slots with zero offspring; = realised births)
  fr_opp_death_cum (S,)      opportunities with >= 1 realised death
  fr_deaths_hist_cum (S, cap+1)  cumulative histogram over opportunities of realised deaths (<= max(kd, kc) <= cap);
                             fr_deaths_hist (cap+1,) is its run-level row
  fr_kd_cum, fr_kc_cum (S,)  capped death / clone CANDIDATE counts (= gateway_numba's die / clone)
  fr_opp_cum (S,)            FR opportunities (ABF: 0)
  fr_opp_event_cum (S,)      opportunities with >= 1 candidate (kd + kc > 0); with kc >= kd the pool law can drop
                             the extra copy, so such an opportunity may realise no death (a pure slot permutation)
  fr_events_hist_cum (S, cap+1)  cumulative histogram over opportunities of candidates kd + kc;
                             fr_events_hist (cap+1,) is its run-level row
  traces_step (K,), traces_t, traces (K, n_tr), traces_anc (K, n_tr), traces_rebirth (K, n_tr):
                             WALKER traces.  Each walker carries a persistent id (a permutation of 0..N-1, diagnostic
                             only, no RNG).  At an FR gather the first new slot (ascending) holding a copy of an old
                             walker keeps that walker's id; the ids of the walkers that died (zero offspring,
                             ascending old slot) pass to the remaining copies (ascending new slot).  Column j follows
                             the walker with id j: x, its run ancestor label, and the number of times id j has died
                             and been reborn as a copy (traces_rebirth).  A trace therefore jumps only when its walker
                             dies (traces_rebirth increments), exactly the in-place 'slot trace' meaning of
                             lta_ladder_numba; the pool law's slot permutations do not show.  ABF: id = slot.
                             Default grid t = 0, 0.1, ..., 40 then every 10 t.u., the late interval multiplied by the
                             smallest integer keeping K < 5000 (20 t.u. at N = 1; meta trace_late_dt_t /
                             trace_late_multiplier, the same rule as lta_ladder_numba's sparse_multiplier)
  X_final, Y_final (N,), max_abs_bias, n_fr_steps
  n_force_evals (exact count, N per completed step), wall_s, peak_rss_mb, n_checkpoints_written, resumed_from
  cfg (dict), meta (dict; knobs in steps and time units, seeds, cap, conventions, run_signature, sessions,
                             provenance, status 'complete')
diagnostics=False OMITS every diagnostic-only key (DIAG_ONLY_KEYS, listed in meta.omitted_keys) instead of leaving
initial values that would read as 'no transitions / never arrived'.

Cost and provenance.  wall_s sums, over the sessions of a resumed run, the time inside run_arm up to the session's
last checkpoint (or completion): it EXCLUDES work lost between a kill and the last checkpoint, and process start-up
before run_arm.  meta.sessions lists every durable session (process start, run start, start step, last checkpoint
time / step, end, VmRSS / VmHWM at start, peak, engine sha256 at import, git commit + dirty flag at import);
meta.wall_lost_upper_s bounds the lost time (run start of a resume minus the previous session's last checkpoint,
including any idle gap).  peak_rss_mb is the max over sessions of VmHWM (/proc/self/status), which run_arm resets to
the current RSS on entry (/proc/self/clear_refs 5), so it is the peak of THIS run even in a reused pool worker or a
fork child of a large parent (whose shared pages still count: meta.sessions[].vmrss_start_mb is the baseline and
meta.peak_rss_over_start_mb the excess); ru_maxrss (which inherits a parent's peak across exec) is only a fallback.
meta.engine_sha256 / git_commit / git_dirty are taken when the module is IMPORTED (the code that runs), not at
completion.

Common contract with lta_ladder_numba.  The shared quantities have different key names in the two engines; COMMON_KEYS
(also in meta.common_keys) maps the LTA/common name to this engine's key and common_view(res) returns the result
under the common names and dtypes (tested against a live lta_ladder_numba result).  'cum_deaths',
'cum_opps_with_event' and 'events_per_opp_hist' map to the REALISED-death quantities, which are what LTA's
replacement events are (one death + one copy each).
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

ENGINE_VERSION = "gateway_ladder_numba/2"
XMIN, XMAX = -1.8, 1.8
N_GRID = 181
EPS = 1e-30
X_BASIN = 0.5
NB_PROD = 180
N_TRACE_MAX = 32

DEFAULT_CFG = dict(
    system="gateway",
    beta=16.0, H=0.5, omega_out=1.0, omega_in=32.0, s=0.1,
    h=2.5e-5,
    nb=NB_PROD, min_count=1.0,
    gamma=1.5, eta=0.1, fr_interval_t=0.004, ramp_t=4.0, score_clip=3.0,
    max_event_fraction=0.08, cap_min=1,
    genealogy_window_t=1.6,
    trace_early_dt_t=0.1, trace_early_until_t=40.0, trace_late_dt_t=10.0, trace_max=5000,
)
# preregistered save points (docs/equal_budget/SCIENTIFIC_PLAN.md section 3;
# configs/equal_budget_v2/gateway_production.json)
GATEWAY_PHYSICAL_CHECKPOINTS_T = (0.5, 1, 2, 4, 10, 20, 40, 100, 400, 1000, 4000, 10000, 40000)
PROFILE_SNAPSHOT_U = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00)

# ---- integer-counter slots of the kernel state vector ``ictr`` ----
I_SPTR, I_TPTR, I_NOPP, I_NOPP_EV, I_KD, I_KC, I_DEATHS, I_LR, I_RL, I_FIRST, I_NFE, I_STEP, I_NOPP_D = range(13)
N_ICTR = 13

# keys present only with diagnostics=True (omitted, not filled with initial values, when diagnostics=False)
DIAG_ONLY_KEYS = ("hist_inst", "region_frac", "events_cum", "lineage_visited", "first_arrival_step", "first_arrival_t",
                  "gen_nuniq_run", "gen_ess_run", "gen_maxfam_run", "gen_nuniq_win", "gen_ess_win", "gen_maxfam_win",
                  "gen_win_age_steps", "traces_step", "traces_t", "traces", "traces_anc", "traces_rebirth")

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
_COMMON_INT = ("gen_n_unique", "gen_n_unique_win")      # integer-valued here, stored as float64 with NaN sentinels
_COMMON_FLOAT = ("hist_inst",)                          # int64 here, float64 in lta_ladder_numba

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


# provenance of the code this process imported (and therefore runs), taken once at import
_PROVENANCE = dict(engine_sha256=_engine_sha256(), **_git_info(),
                   imported_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class RunInterrupted(RuntimeError):
    """Raised by run_arm(stop_after_chunks=k): the run stopped after k chunks (state is in the checkpoint)."""


# ---------------------------------------------------------------------------------------------------------
# helpers shared with gateway_numba (copies; equality asserted in the tests)
# ---------------------------------------------------------------------------------------------------------
def gaussian_kernel_np(bw, dx):
    r = max(1, int(round(4.0 * bw / dx)))
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t * dx / bw) ** 2)
    return k / (k.sum() * dx), r


def grid_dx():
    xg = np.linspace(XMIN, XMAX, N_GRID)
    return float(xg[1] - xg[0])


def init_left(seed, N, beta=16.0, oout=1.0, oin=32.0, s=0.1):
    """gateway_numba.init_left: numpy default_rng(1000 + seed); x ~ N(-1, 0.05) reflected, y from the
    continuum conditional."""
    rng = np.random.default_rng(1000 + int(seed))
    x = rng.normal(-1.0, 0.05, N)
    span = XMAX - XMIN
    qm = np.remainder(x - XMIN, 2.0 * span)
    x = np.where(qm > span, 2.0 * span - qm, qm) + XMIN
    z = rng.normal(0.0, 1.0, N)
    om = oout + (oin - oout) * np.exp(-x * x / (2.0 * s * s))
    y = z * np.sqrt(1.0 / (beta * om ** 2))
    return x.astype(np.float64), y.astype(np.float64)


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


def prereg_save_grid(n_steps, h, physical_t=GATEWAY_PHYSICAL_CHECKPOINTS_T, profile_u=PROFILE_SNAPSHOT_U):
    """The preregistered save grid (SCIENTIFIC_PLAN.md section 3): budget_save_grid (200 uniform + 24 log-spaced u)
    union the physical-time checkpoints t <= T_N at the nearest integration step (the same rule as
    scripts/equal_budget/run_ladder.save_grid) union the profile snapshots u (already on the uniform grid)."""
    n_steps = int(n_steps)
    steps, _ = budget_save_grid(n_steps)
    extra = [int(round(t / h)) for t in physical_t if 0 < round(t / h) <= n_steps]
    prof = [min(max(int(round(u * n_steps)), 1), n_steps) for u in profile_u]
    return np.unique(np.concatenate([steps, np.asarray(extra + prof, dtype=np.int64)]))


def default_trace_grid(n_steps, h, early_dt_t=0.1, early_until_t=40.0, late_dt_t=10.0, max_k=5000):
    """Completed-step counts of the walker traces: t = 0, early_dt, ... up to early_until, then every late_dt
    (late_dt multiplied by the smallest integer that keeps K < max_k).  Returns (steps, late_dt_used)."""
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
    # gateway_numba passes ramp_steps as a float; keep the integral value exact when it is one
    if ramp_steps > 0 and abs(ramp_steps - round(ramp_steps)) < 1e-6:
        ramp_steps = float(round(ramp_steps))
    window_steps = int(round(cfg["genealogy_window_t"] / h)) if cfg["genealogy_window_t"] > 0 else 0
    cap = max(int(cfg["cap_min"]), int(math.floor(cfg["max_event_fraction"] * N)))
    L = N + cap + max(N, cap)
    return dict(fr_every=fr_every, ramp_steps=ramp_steps, window_steps=window_steps, cap=cap, fr_block=L,
                dt_fr=h * fr_every)


# ---------------------------------------------------------------------------------------------------------
# numba MT19937 (np.random inside njit) state handling -- private API, see module docstring
# ---------------------------------------------------------------------------------------------------------
_MT_N = 624
_OFF_HAS_GAUSS = 4 + 4 * _MT_N            # rnd_state_t {int index; uint mt[624]; int has_gauss; double gauss; int is_initialized}
_OFF_IS_INIT = _OFF_HAS_GAUSS + 4 + 8      # has_gauss at 2500, gauss (8-byte aligned) at 2504, is_initialized at 2512
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
# numba kernels (dynamics copied from gateway_numba.py:81-177, 251-326, 367-433)
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
    """gateway_numba.fr_resample with the uniforms read sequentially from ubuf[u0:] (or, if ``internal``, drawn
    from np.random exactly as gateway_numba does).  Writes the gather index into sel (new = old[sel]) and
    returns (kd, kc, n_uniforms_used).  sel is stale when kd + kc == 0."""
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
def _rec_save(ptr, N, nb, is_fr, diag, X, Mh, Ch, anc_w, anc_r, visited, ictr, ev_hist, dh_hist,
              o_M, o_C, o_hist, o_reg, o_ev, o_vis, o_gen, o_fr, o_evh, o_dth, cnt):
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
def _rec_trace(ptr, n_tr, N, X, anc_r, wid, wgen, inv, tr_x, tr_anc, tr_gen):
    """Trace column k = the walker with persistent id k (ids are a permutation of 0..N-1)."""
    for i in range(N):
        w = wid[i]
        if w < n_tr:
            inv[w] = i
    for k in range(n_tr):
        i = inv[k]
        tr_x[ptr, k] = X[i]
        tr_anc[ptr, k] = anc_r[i]
        tr_gen[ptr, k] = wgen[k]


@njit(cache=True)
def _record_initial(N, nb, is_fr, diag, n_tr, save_at, trace_at, X, Mh, Ch, anc_w, anc_r, visited, wid, wgen, ictr,
                    ev_hist, dh_hist, o_M, o_C, o_hist, o_reg, o_ev, o_vis, o_gen, o_fr, o_evh, o_dth,
                    tr_x, tr_anc, tr_gen):
    """Saves / traces requested at completed-step count 0 (the initial state)."""
    cnt = np.zeros(N, dtype=np.int64)
    inv = np.zeros(max(n_tr, 1), dtype=np.int64)
    while ictr[I_SPTR] < save_at.shape[0] and save_at[ictr[I_SPTR]] == 0:
        _rec_save(ictr[I_SPTR], N, nb, is_fr, diag, X, Mh, Ch, anc_w, anc_r, visited, ictr, ev_hist, dh_hist,
                  o_M, o_C, o_hist, o_reg, o_ev, o_vis, o_gen, o_fr, o_evh, o_dth, cnt)
        ictr[I_SPTR] += 1
    while diag and ictr[I_TPTR] < trace_at.shape[0] and trace_at[ictr[I_TPTR]] == 0:
        _rec_trace(ictr[I_TPTR], n_tr, N, X, anc_r, wid, wgen, inv, tr_x, tr_anc, tr_gen)
        ictr[I_TPTR] += 1


@njit(cache=True)
def _advance(step0, step1, N, h, beta, H, oout, oin, s, nb, min_count,
             is_fr, gamma, ramp_steps, fr_every, score_clip, cap, kern, r_eta, dx, ubuf, ublk, fr_internal,
             diag, win_steps, save_at, trace_at, n_tr,
             X, Y, Mh, Ch, anc_w, anc_r, label, visited, wid, wgen, ictr, fctr, ev_hist, dh_hist,
             o_M, o_C, o_hist, o_reg, o_ev, o_vis, o_gen, o_fr, o_evh, o_dth, tr_x, tr_anc, tr_gen):
    """Advance one arm from completed step ``step0`` to ``step1`` in place.  Langevin normals come from numba's
    np.random (state restored by the caller); FR uniforms from ubuf (block ublk per opportunity, the first block
    belonging to the first FR step >= step0), or from np.random if fr_internal.  Everything under ``diag`` (labels,
    genealogy, persistent ids, traces) only reads the dynamics state."""
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
    amp = math.sqrt(2.0 * h / beta)
    dt_fr = h * fr_every
    two_s2 = 2.0 * s * s
    s2 = s * s
    dom_amp = oin - oout
    delta = (XMAX - XMIN) / nb
    do_fr_arm = is_fr and N >= 2
    sptr = ictr[I_SPTR]
    tptr = ictr[I_TPTR]
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
            f = 4.0 * H * x * (x * x - 1.0) + om * dom * y * y
            fx[i] = f
            fy[i] = om * om * y
            j = int(math.floor((x - XMIN) / delta + 1e-9))
            if j < 0:
                j = 0
            elif j > nb - 1:
                j = nb - 1
            jb[i] = j
            Ch[j] += 1.0
            Mh[j] += f
        ictr[I_NFE] += N
        # --- move ---
        for i in range(N):
            j = jb[i]
            den = Ch[j] + min_count
            bias = Mh[j] / den if den > 0.0 else 0.0
            if abs(bias) > mb:
                mb = abs(bias)
            xn = _reflect(X[i] + (-fx[i] + bias) * h + amp * zx[i], XMIN, XMAX)
            X[i] = xn
            Y[i] = Y[i] - fy[i] * h + amp * zy[i]
            if diag:
                if xn > X_BASIN:
                    if label[i] == -1:
                        n_lr += 1
                    label[i] = 1
                    visited[i] = 1
                    if first < 0:
                        first = step + 1
                elif xn < -X_BASIN:
                    if label[i] == 1:
                        n_rl += 1
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
                for i in range(N):
                    tmpx[i] = X[sel[i]]; tmpy[i] = Y[sel[i]]
                for i in range(N):
                    X[i] = tmpx[i]; Y[i] = tmpy[i]
                for i in range(N):
                    ocnt[i] = 0
                for i in range(N):
                    ocnt[sel[i]] += 1
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
                    # persistent walker ids: the first new slot holding a copy of an old walker keeps its id; the
                    # ids of the dead (ascending old slot) pass to the remaining copies (ascending new slot)
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
        # --- traces and saves after step + 1 completed steps ---
        if diag:
            while tptr < n_traces and trace_at[tptr] == step + 1:
                _rec_trace(tptr, n_tr, N, X, anc_r, wid, wgen, inv, tr_x, tr_anc, tr_gen)
                tptr += 1
        if sptr < n_saves and save_at[sptr] == step + 1:
            ictr[I_LR] = n_lr; ictr[I_RL] = n_rl
            while sptr < n_saves and save_at[sptr] == step + 1:
                _rec_save(sptr, N, nb, is_fr, diag, X, Mh, Ch, anc_w, anc_r, visited, ictr, ev_hist, dh_hist,
                          o_M, o_C, o_hist, o_reg, o_ev, o_vis, o_gen, o_fr, o_evh, o_dth, cnt)
                sptr += 1
    ictr[I_SPTR] = sptr
    ictr[I_TPTR] = tptr
    ictr[I_LR] = n_lr
    ictr[I_RL] = n_rl
    ictr[I_FIRST] = first
    if step1 > step0:
        ictr[I_STEP] = step1
    fctr[0] = mb


# ---------------------------------------------------------------------------------------------------------
# Python driver: state, chunks, checkpoints, result contract
# ---------------------------------------------------------------------------------------------------------
_STATE_ARRAYS = ("X", "Y", "Mh", "Ch", "anc_w", "anc_r", "label", "visited", "wid", "wgen", "ictr", "fctr",
                 "ev_hist", "dh_hist", "o_M", "o_C", "o_hist", "o_reg", "o_ev", "o_vis", "o_gen", "o_fr", "o_evh",
                 "o_dth", "tr_x", "tr_anc", "tr_gen")


def _git_commit():
    """Current HEAD (not necessarily the code this process imported: see _PROVENANCE)."""
    return _git_info()["git_commit"]


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _proc_status_mb(field):
    """A /proc/self/status memory field (VmRSS, VmHWM, ...) in MB, or None where unavailable."""
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
    """Reset this process's VmHWM to its current RSS (Linux >= 4.0: write 5 to /proc/self/clear_refs), so that a
    later VmHWM is the peak of THIS run, not of earlier work in the same (pooled / forked) process."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def _proc_start_epoch():
    """Wall-clock start time of this process (Linux /proc), or None."""
    try:
        with open("/proc/self/stat") as f:
            s = f.read()
        ticks = int(s[s.rindex(")") + 2:].split()[19])          # field 22 (starttime) counted from 'state' = 3
        with open("/proc/stat") as f:
            btime = next(int(line.split()[1]) for line in f if line.startswith("btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


def full_cfg(cfg=None):
    c = dict(DEFAULT_CFG)
    if cfg:
        c.update(cfg)
    return c


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
    """Strictly increasing integer completed-step counts in [0, n_steps]; integral floats are accepted, anything
    else is refused (never truncated: pass np.rint(t / h) for the nearest integration step)."""
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


def _prepare(cfg, method, N, seed, n_steps, save_steps, trace_steps=None, noise_seed=None, diagnostics=True,
             x0=None, y0=None, fr_internal=False):
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
        # every walker could die (no survivor to copy): gateway_numba would read pool[-1]; never in production
        raise ValueError(f"FR cap {kn['cap']} >= N {N} allows total extinction")
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
    if x0 is None or y0 is None:
        x0, y0 = init_left(seed, N, c["beta"], c["omega_out"], c["omega_in"], c["s"])
    x0 = np.ascontiguousarray(x0, dtype=np.float64); y0 = np.ascontiguousarray(y0, dtype=np.float64)
    if x0.shape != (N,) or y0.shape != (N,):
        raise ValueError("x0 / y0 must have shape (N,)")
    diag = bool(diagnostics)
    sig = dict(engine=ENGINE_VERSION, cfg=c, method=method, N=N, seed=seed, n_steps=n_steps, noise_seed=noise_seed,
               fr_entropy=fr_seed_entropy(seed, N), save_sha=_sha(save_at), trace_sha=_sha(trace_at),
               x0_sha=_sha(x0), y0_sha=_sha(y0), diag=diag, fr_internal=bool(fr_internal),
               engine_sha=_PROVENANCE["engine_sha256"])
    sig = json.loads(json.dumps(sig, sort_keys=True))          # the form it takes after a JSON round trip
    return dict(c=c, kn=kn, h=h, N=N, seed=seed, n_steps=n_steps, is_fr=is_fr, noise_seed=noise_seed,
                save_at=save_at, trace_at=trace_at, trace_late_dt=trace_late_dt, trace_mult=trace_mult,
                x0=x0, y0=y0, diag=diag, fr_internal=bool(fr_internal), sig=sig)


def run_signature(cfg, method, N, seed, n_steps, save_steps, *, trace_steps=None, noise_seed=None,
                  diagnostics=True, x0=None, y0=None, _fr_internal_rng=False):
    """The signature run_arm stores in checkpoints and in meta.run_signature for this request."""
    return _prepare(cfg, method, N, seed, n_steps, save_steps, trace_steps, noise_seed, diagnostics, x0, y0,
                    _fr_internal_rng)["sig"]


def signature_mismatch(old, new, allow_engine_change=False):
    """Sorted list of the signature fields that differ (['run_signature missing'] if old is None)."""
    if not isinstance(old, dict):
        return ["run_signature missing"]
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)
                  and not (allow_engine_change and k == "engine_sha"))


def run_arm(cfg, method, N, seed, n_steps, save_steps, *, trace_steps=None, noise_seed=None,
            chunk_steps=None, checkpoint_path=None, checkpoint_every_s=600.0, stop_after_chunks=None,
            diagnostics=True, x0=None, y0=None, allow_engine_change=False, _fr_internal_rng=False):
    """Run ONE arm (method 'abf' or 'fr') of the (N, seed) job for n_steps steps; returns the result dict.

    save_steps / trace_steps: strictly increasing integer completed-step counts in [0, n_steps] (trace default:
    default_trace_grid; non-integral values are refused).  With checkpoint_path, an atomic checkpoint is written
    every checkpoint_every_s seconds of wall time (0 = after every chunk) and the run resumes from it if present
    (its signature must match; allow_engine_change=True tolerates only a different engine sha256).
    stop_after_chunks=k raises RunInterrupted after k chunks of THIS call (testing hook)."""
    t_wall0 = time.monotonic()
    epoch0 = time.time()
    hwm_before = _proc_status_mb("VmHWM")
    peak_reset = _reset_peak_rss()
    rss_start = _proc_status_mb("VmRSS")
    fr_internal = bool(_fr_internal_rng)
    P = _prepare(cfg, method, N, seed, n_steps, save_steps, trace_steps, noise_seed, diagnostics, x0, y0, fr_internal)
    c, kn, h, N, seed, n_steps, is_fr = P["c"], P["kn"], P["h"], P["N"], P["seed"], P["n_steps"], P["is_fr"]
    save_at, trace_at, diag, sig = P["save_at"], P["trace_at"], P["diag"], P["sig"]
    x0, y0, noise_seed = P["x0"], P["y0"], P["noise_seed"]
    nb = int(c["nb"])
    dx = grid_dx()
    kern, r_eta = gaussian_kernel_np(float(c["eta"]), dx)
    S, K = len(save_at), len(trace_at)
    n_tr = min(N, N_TRACE_MAX)
    cap, L = kn["cap"], kn["fr_block"]
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
                  ictr=np.zeros(N_ICTR, dtype=np.int64), fctr=np.zeros(1),
                  ev_hist=np.zeros(cap + 1, dtype=np.int64), dh_hist=np.zeros(cap + 1, dtype=np.int64),
                  o_M=np.zeros((S, nb)), o_C=np.zeros((S, nb)), o_hist=np.zeros((S, nb), dtype=np.int64),
                  o_reg=np.full((S, 3), np.nan), o_ev=np.zeros((S, 2), dtype=np.int64),
                  o_vis=np.zeros((S, 1), dtype=np.int64), o_gen=np.full((S, 6), np.nan),
                  o_fr=np.zeros((S, 6), dtype=np.int64), o_evh=np.zeros((S, cap + 1), dtype=np.int64),
                  o_dth=np.zeros((S, cap + 1), dtype=np.int64), tr_x=np.full((K, n_tr), np.nan),
                  tr_anc=np.full((K, n_tr), -1, dtype=np.int64), tr_gen=np.full((K, n_tr), -1, dtype=np.int64))
        st["ictr"][I_FIRST] = 0 if np.any(x0 > X_BASIN) else -1
        _mt_seed(noise_seed)
        mt_idx, mt_key = mt_get()
        _record_initial(N, nb, is_fr, diag, n_tr, save_at, trace_at, st["X"], st["Mh"], st["Ch"], st["anc_w"],
                        st["anc_r"], st["visited"], st["wid"], st["wgen"], st["ictr"], st["ev_hist"], st["dh_hist"],
                        st["o_M"], st["o_C"], st["o_hist"], st["o_reg"], st["o_ev"], st["o_vis"], st["o_gen"],
                        st["o_fr"], st["o_evh"], st["o_dth"], st["tr_x"], st["tr_anc"], st["tr_gen"])
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
                 is_fr, float(c["gamma"]), float(kn["ramp_steps"]), int(fe), float(c["score_clip"]), int(cap),
                 kern, int(r_eta), float(dx), ubuf, int(L), fr_internal,
                 diag, int(kn["window_steps"]), save_at, trace_at, int(n_tr),
                 st["X"], st["Y"], st["Mh"], st["Ch"], st["anc_w"], st["anc_r"], st["label"], st["visited"],
                 st["wid"], st["wgen"], st["ictr"], st["fctr"], st["ev_hist"], st["dh_hist"],
                 st["o_M"], st["o_C"], st["o_hist"], st["o_reg"], st["o_ev"], st["o_vis"], st["o_gen"], st["o_fr"],
                 st["o_evh"], st["o_dth"], st["tr_x"], st["tr_anc"], st["tr_gen"])
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
    if int(ic[I_SPTR]) != S or (diag and int(ic[I_TPTR]) != K) or int(ic[I_NFE]) != N * n_steps:
        raise RuntimeError("run incomplete: not every save/trace fired or the force count is wrong")
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
    meta = dict(
        engine=ENGINE_VERSION, engine_sha256=_PROVENANCE["engine_sha256"], system="gateway", method=method,
        status="complete",
        N=N, seed=seed, n_steps=n_steps, B=N * n_steps, T=n_steps * h, h=h,
        noise_seed=noise_seed, noise_seed_formula="(1_000_003*seed + 7919*N) % (2**31-1) unless given",
        noise_stream="numba np.random MT19937 (Langevin normals only: N zx then N zy per step)",
        fr_stream="numpy Generator(PCG64(SeedSequence(fr_seedseq_entropy))); opportunity k owns doubles [kL,(k+1)L)",
        fr_seedseq_entropy=fr_seed_entropy(seed, N), fr_block_L=L, fr_internal_rng=fr_internal,
        cap=cap, fr_every=kn["fr_every"], fr_interval_t=c["fr_interval_t"], dt_fr=kn["dt_fr"],
        ramp_steps=kn["ramp_steps"], ramp_t=c["ramp_t"], window_steps=kn["window_steps"],
        genealogy_window_t=c["genealogy_window_t"], gamma=c["gamma"], eta=c["eta"], score_clip=c["score_clip"],
        max_event_fraction=c["max_event_fraction"], cap_min=c["cap_min"], nb=nb, min_count=c["min_count"],
        cell=dict(beta=c["beta"], H=c["H"], omega_out=c["omega_out"], omega_in=c["omega_in"], s=c["s"]),
        n_saves=S, n_traces=K, n_trace_slots=n_tr, trace_late_dt_t=P["trace_late_dt"],
        trace_late_multiplier=P["trace_mult"], chunk_steps=chunk_steps,
        diagnostics=diag, omitted_keys=([] if diag else list(DIAG_ONLY_KEYS)),
        event_names=["trans_LR", "trans_RL"], visit_names=["right_well"],
        region_names=["left x<-0.5", "gate", "right x>0.5"], first_arrival_names=["right_well"],
        accumulator_convention=("pre-deposit (gateway_numba): the save at completed-step count k holds the deposits "
                                "of the states 0..k-1 (C_all[k].sum() = N k); the saved state's own deposit enters "
                                "the next save. lta_ladder_numba includes the saved step's deposit."),
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
        cfg=c, meta=meta,
    )
    if not diag:
        for k in DIAG_ONLY_KEYS:
            del res[k]
    return res


def common_view(res):
    """The result under the names and dtypes shared with lta_ladder_numba (COMMON_KEYS).  Keys absent from res
    (diagnostics off) stay absent.  Conventions that still differ are in res['meta'] (accumulator_convention,
    force_eval_convention)."""
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
    """True iff path holds a result with status 'complete' (and, if ``signature`` is given, whose stored
    meta.run_signature matches it; see run_signature)."""
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


_SIG_KW = ("trace_steps", "noise_seed", "diagnostics", "x0", "y0", "_fr_internal_rng")


def run_job(cfg, method, N, seed, n_steps, out_path, save_steps=None, checkpoint_every_s=600.0, **kw):
    """Runner convenience: return an existing complete result IF its run signature matches this request (any
    mismatch -- cfg, method, N, seed, n_steps, grids, seeds, initial state, diagnostics, engine sha256 unless
    allow_engine_change=True -- raises ValueError instead of returning another job's result); else run (resuming
    from out_path + '.ckpt.npz'), save the result atomically, then delete the checkpoint.  The default save grid is
    prereg_save_grid(n_steps, h) (budget grid + physical-time checkpoints)."""
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
