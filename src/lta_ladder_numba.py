"""CPU (numba) engine of the ethane/LTA histogram-ABF sampler, with and without uniform-target
marginal Fisher-Rao birth-death, for the equal-budget replica ladder (2026-10-10).

Why this exists
---------------
The equal-budget ladder holds the force-evaluation budget B = N x n_steps = 3.072e8 fixed and
moves along N in {1024, 512, ..., 2, 1} at T = 300 K and 150 K, h = 2e-4 (n_steps = 3.072e8 / N, up
to 3.07e8 sequential steps at N = 1).  The production torch engine (``src/lta/core_lta.py``) is
dispatch-bound for small N, so the long-thin end of the ladder needs a compiled scalar loop.

What is ported (op for op from ``core_lta.run_sampler``, methods 'abf' and 'fr_uniform',
``abf_estimator='histogram'``; porting spec ``docs/equal_budget/audit/AUDIT_0A_lta.md``)
-------------------------------------------------------------------------------------------
* physics: 384 rigid framework O, shifted LJ (eps 93 K kB, sigma 3.48 A, rc 10 A) with the minimum
  image d - L rint(d / L) on bead-O displacements only, harmonic bond (k 400, r0 1.54), coordinates
  never wrapped; overdamped Euler-Maruyama q <- (q + h (F + bias)) + sqrt(2 h kT) xi;
* CV phi = remainder((2 pi / a) x_COM + pi, 2 pi) - pi in the fmod form of torch.remainder, bin
  j = floor((phi + pi) / dphi) % n_grid, local mean force f = clip(-(a / 2 pi)(F0x + F1x), +-480);
* the step order: forces/CV at the PRE-move positions -> deposits of ALL N walkers (per-step bin
  sums first, then added to the accumulators, like torch's scatter_add into zeros) -> production
  accumulators from step >= burn_in -> bias = ramp * clip(M_j / max(C_j, 1), +-60) * (pi / a) read
  from the ALL-STEPS accumulators INCLUDING this step's deposit (ramp = min(1, step / warmup)) ->
  region bookkeeping -> save -> move -> FR at the POST-move positions when nxt = step + 1 >= fr_start
  and (nxt - fr_start) % fr_every == 0;
* FR score: bin counts of the post-move phi smoothed by the UNNORMALISED single-image wrapped
  Gaussian matrix K (bandwidth 0.10 rad), clamped and normalised; raw = log p(phi_i) - log q - KL with
  periodic linear interpolation; recentred, then 3 x (clip +-score_clip, recentre);
* birth-death: death prob 1 - exp(-rate S+ h fr_every); candidates u_i < dp_i (ascending slot
  order); the 'sum(S-) <= EPS or sum(S+) <= EPS -> no event' guard (after the uniforms are drawn);
  when the candidates exceed the cap a uniformly random ordered cap-subset survives; sources iid
  categorical proportional to S- (with replacement); whole-molecule copy + ancestor copy; deaths and
  sources are disjoint (S > 0 vs S < 0), so the in-place copy equals torch's copy from the old state;
* the torch-equivalent bookkeeping (n_transitions, lineage-following n_cage_crossings, birth/death
  histograms, per-opportunity event counts, score std / absmax) for the replay equivalence test.

Deliberate differences from the torch engine
--------------------------------------------
* RANDOMNESS.  Three independent PCG64 streams from SeedSequence([seed, N, round(T_K)]).spawn(3):
  child 0 the initial conditions (same LAW as LTASystem.initial_conditions: uniform alpha-cage centre
  of the 8, + 0.5 N(0, 1) per coordinate, uniform orientation, bond 1.54); child 1 the Langevin noise,
  standard_normal in the order [step][replica][bead][xyz]; child 2 the FR draws: one block of
  N + 2 cap uniforms per FR opportunity (whenever FR is active: cap >= 1 and rate > 0) -- [0, N) the
  firing uniforms, [N, N + cap) a partial Fisher-Yates for the cap subset, [N + cap, N + 2 cap) the
  inverse-CDF source draws.  The block size is fixed, so the FR stream position depends only on the
  opportunity index.  The ABF and the FR arm of one (seed, N, T) share the IC and the noise slot by
  slot.  torch's CPU randperm / multinomial algorithms are not reproduced (same law, tested E2); a
  REPLAY mode consumes torch-recorded draws instead (E1).
* The FINITE-N CAP EXTENSION cap(N) = max(cap_min, floor(0.02 N)); cap_min = 1 for the ladder (FR
  live down to N = 2), cap_min = 0 is the historical int(0.02 N) (FR silently off below N = 50).
* Reduction orders: the LJ pair sum is a vectorised reduction (fastmath 'reassoc' on the sums only;
  every pair term is evaluated in plain IEEE arithmetic in torch's operation order), the KDE product
  is an ascending sum over occupied bins, means are sequential sums -- ulp-level differences from
  torch's own (unspecified) orders; measured in E1.
* BITWISE CLAIMS HOLD PER NUMBA COMPILE TARGET.  The vector width LLVM picks for the 'reassoc' sum
  depends on the compile target (CPU name + features; NUMBA_CPU_NAME / NUMBA_CPU_FEATURES override the
  host), so the same job compiled for another target differs at the ulp level from step 1 (measured:
  znver4 vs skylake-avx512 / generic |dq| ~ 1e-14 after 3000 steps; haswell happened to agree).
  Bitwise checkpoint resume, chunking invariance, cross-process determinism and the ABF/FR pairing
  before fr_start therefore hold for runs compiled for ONE target.  The target is recorded
  (meta.compile_target) and is part of the checkpoint signature: a resume on a different target is
  refused unless run_arm(..., allow_platform_change=True), which records the change in meta.sessions.
  The two arms of a (seed, N, T) must run on the same target to be bitwise paired before fr_start.
* deposit_clip = 8 x abf_force_clip (core_lta's f_loc clamp) unless deposit_clip_override = True; an
  explicit deposit_clip that differs from 8 x abf_force_clip without the override is refused.
* The FR arm at N = 1 is refused (no FR at N = 1: the score is identically 0).  At N = 2 the
  uniform-target score is zero BY SYMMETRY (each walker sees K(0) + K(d)) apart from the within-bin
  linear interpolation, so |S| ~ 5e-3 and the arm realises a handful of deaths per run: N = 2 is an
  FR-INACTIVE cell (meta.fr_score_degenerate = True); N = 3-4 have small scores too (walkers mostly
  farther apart than the 0.10 rad bandwidth), read the realised total_replacement_events.
* No torch CPU eff_counts aliasing bug: every save stores copies.

Conventions of the outputs (COMMON RESULT CONTRACT, see ``run_arm``)
--------------------------------------------------------------------
* A save at completed-step count k records the state at the START of step k (after k moves and the
  FR opportunity at k), with the accumulators INCLUDING step k's deposit -- torch's save point.
* Force evaluations: every evaluated step k = 0..n_steps evaluates the N molecules once (the final
  evaluation at k = n_steps deposits the final state, like torch); a complete run has
  n_force_evals = N (n_steps + 1).  B = N n_steps counts moves.
* TRUE events (not lineage-inherited): window-plane crossings = changes of the integer cell
  floor(x_COM / a) of a slot; a translocation = arrival in the cage region |z| > 4 A of a DIFFERENT
  x-cell than the cage region last occupied by that slot's lineage (a walker that starts outside
  every cage region adopts its first cage without counting).  Labels are updated at every evaluated
  step AND at every FR opportunity on the post-move positions BEFORE the copies (so every executed
  move is inspected exactly once for every walker that made it, including walkers about to die);
  a clone inherits the source's labels, so a copy is never counted as an event.  Lineage flags
  (ever visited the window region; ever translocated) are inherited by clones.
* Genealogy: whole-run ancestor labels (never reset) and windowed labels reset to arange(N) at every
  step that is a multiple of the window (4 t.u. = 20000 steps) AFTER that step's save, so a save at a
  window boundary reports the full window.  ESS = (sum c)^2 / sum c^2 / N.
* Traces: every trace_dense_every_tu (0.05 t.u.) up to 60 t.u., then every m x 10 t.u. with the
  smallest integer m keeping K <= max_traces (4999).  Only N = 1 needs m = 2 (20 t.u., K = 4270; a 10
  t.u. cadence would need K = 7339); meta.traces.sparse_multiplier / sparse_every_t record it.
* diagnostics=False OMITS every diagnostic-only key (DIAG_ONLY_KEYS, listed in meta.omitted_keys)
  instead of leaving initial values that would read as 'never reached' / 'no events'.
* peak_rss_mb is the peak resident set of THIS run: run_arm resets the process's VmHWM to the current
  RSS on entry (/proc/self/clear_refs 5) and reads VmHWM at every checkpoint and at completion; the
  value is the max over the sessions of a resumed run (meta.sessions[].peak_rss_mb, vmrss_start_mb =
  the baseline the process already held).  ru_maxrss (process lifetime) is only a fallback
  (meta.peak_rss_method).
* Provenance: meta.engine_sha256 (sha256 of this file), git_commit and git_dirty (git status of this
  file) are taken when the module is IMPORTED, i.e. for the code that runs; every session of a resumed
  run records its own.  The checkpoint signature holds the job fingerprint, ENGINE, the engine sha256,
  the numpy / numba versions and the compile target; a resume refuses any mismatch unless
  allow_engine_change (engine sha / ENGINE) or allow_platform_change (numpy, numba, compile target).

Public API
----------
  cfg = make_cfg(T_K, N, seed, **overrides)      # frozen ladder knobs by default (DEFAULT_CFG)
  save_steps, u = budget_save_grid(cfg["n_steps"])
  res = run_arm(cfg, "abf" | "fr", save_steps, checkpoint_path=..., checkpoint_every_s=...)
  save_result(path, res); res = load_result(path)
  helpers: initial_conditions, seed_streams, trace_grid, kde_matrix, mean_force, histogram_pmf,
           fr_select_native (the FR selection law, njit), run_signature, signature_mismatch,
           compile_target, InterruptedRun, COST_KEYS, DIAG_ONLY_KEYS
Tests: tests/test_lta_ladder_numba.py (E1-E7 + X*).
"""
from __future__ import annotations

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
from numba import njit

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
KB = 0.008314462618                 # kJ/mol/K (core_lta.KB)
PI = math.pi
TWO_PI = 2.0 * math.pi
EPS = 1.0e-12
SENT = -(2 ** 62)                   # "no cage yet" label
BUDGET = 307_200_000
LADDER_N = (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
METHODS = ("abf", "fr")
ENGINE = "lta_ladder_numba/2"
# result keys present only with diagnostics=True (omitted, and listed in meta.omitted_keys, when off)
DIAG_ONLY_KEYS = ("hist_inst", "region_frac", "events_window_crossings", "events_translocations",
                  "lineage_ever_window", "lineage_ever_opposite", "first_window_step", "first_opposite_step",
                  "gen_n_unique", "gen_ess", "gen_max_frac", "gen_n_unique_win", "gen_ess_win", "gen_max_frac_win",
                  "n_transitions", "n_cage_crossings_lineage", "traces_step", "traces", "traces_anc")


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


def compile_target():
    """The numba CPU compile target of this process (LLVM triple, CPU name, CPU features): the jitted kernels
    (and their on-disk cache entries) are built for it, and the vectorised 'reassoc' LJ sum is bitwise
    reproducible only within one target.  NUMBA_CPU_NAME / NUMBA_CPU_FEATURES override the host values."""
    try:
        from numba.core.registry import cpu_target
        triple, name, feats = cpu_target.target_context.codegen().magic_tuple()
        return dict(triple=str(triple), cpu_name=str(name),
                    cpu_features_sha256=hashlib.sha256(str(feats).encode()).hexdigest(), cpu_features=str(feats))
    except Exception:  # pragma: no cover
        return dict(triple="unknown", cpu_name=platform.processor() or "unknown", cpu_features_sha256="unknown",
                    cpu_features="")


# provenance of the code this process imported (and therefore runs), taken once at import
_PROVENANCE = dict(engine_sha256=_engine_sha256(), **_git_info(),
                   imported_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

DEFAULT_CFG = dict(
    system="lta",
    T_K=300.0, N=1024, seed=0,
    h=2.0e-4,
    budget=BUDGET,                  # B = N * n_steps (moves)
    n_steps=None,                   # None -> budget // N
    n_grid=180,
    warmup_steps=20000,             # bias ramp (4 t.u.)
    burn_in_steps=20000,            # production accumulators from this step (4 t.u.)
    fr_start_steps=20000,
    fr_every=5,                     # 0.001 t.u.
    fr_rate=0.2,
    score_clip=2.0,
    kde_bandwidth=0.10,
    max_event_fraction=0.02,
    cap_min=1,                      # finite-N extension; 0 = historical int(0.02 N)
    abf_force_clip=60.0,            # bias read clip
    deposit_clip=None,              # f_loc clip; None -> 8 * abf_force_clip (core_lta's rule)
    deposit_clip_override=False,    # True allows a deposit_clip != 8 * abf_force_clip (departs from torch)
    abf_bias_scale=1.0,
    window_half=1.5, cage_min=4.0,  # A, region convention of core_lta.region_index
    genealogy_window_tu=4.0,
    trace_dense_every_tu=0.05, trace_dense_until_tu=60.0, trace_sparse_every_tu=10.0,
    n_trace_max=32, max_traces=4999,
    accumulate_u=False,             # production U(z) accumulators (torch u_of_z); costs a 2nd LJ pass
    diagnostics=True,               # regions, true events, genealogy stats, traces, hist_inst
    record_events=False,            # per-opportunity counts + (step, death, source) log (tests)
    eps_go_K=93.0, sigma_go=3.48, rc=10.0, r0_bond=1.54, k_bond=400.0,
    framework_npz="cache/lta/framework.npz",
)

# ---- packed parameter / state layouts (module constants are compile-time constants in numba) ----
F_DT, F_NS, F_A, F_L, F_INVL, F_RC2, F_S2, F_C24, F_KB, F_R0, F_C4, F_VRC = range(12)
F_CPHI, F_CF, F_GPX, F_DPHI, F_FCLIP, F_DCLIP, F_RATE, F_DTEFF, F_SCLIP = range(12, 21)
F_WH, F_CM, F_BSCALE, F_GRID0, F_QV, F_LQ = range(21, 27)
NFP = 27
I_N, I_NG, I_NSTEPS, I_WARM, I_BURN, I_FRSTART, I_FREVERY, I_CAP, I_ISFR, I_FRON = range(10)
I_GWIN, I_ACCU, I_DIAG, I_MODE, I_NTR, I_REC = range(10, 16)
NIP = 16
(S_STEP, S_SP, S_TP, S_TRANS, S_PREVOK, S_REPL, S_NOPP, S_NOPPEV, S_WCROSS, S_TRANSLOC,
 S_FIRSTWIN, S_FIRSTOPP, S_NSCORE, S_RPU, S_RPP, S_RPS, S_NFE, S_EVLOG, S_BADSRC) = range(19)
NIST = 19
G_SSTD, G_SMAX = 0, 1
NFST = 2
W_HLC, W_REPC, W_PREV, W_LCELL, W_LCAGE, W_EVWIN, W_EVOPP = range(7)
NW = 7
A_FS, A_CS, A_FSP, A_CSP, A_USP, A_BH, A_DH = range(7)
NA = 7
# per-save integer columns
V_WCROSS, V_TRANSLOC, V_EVWIN, V_EVOPP, V_NU, V_NUW, V_DEATHS, V_OPPS, V_OPPSEV, V_TRANS, V_CAGEX = range(11)
NV = 11
# per-save float columns
U_ESS, U_MAXF, U_ESSW, U_MAXFW = range(4)
NU = 4


# =====================================================================================
# numba kernels
# =====================================================================================
@njit(cache=True, error_model="numpy")
def _pair(x, y, z, ox, oy, oz, L, invL, rc2, s2, c24):
    """Force of one framework O on one bead, torch's operation order; 0 outside the cutoff.
    (rint(d * invL) and rint(d / L) can only differ at |d_mi| = L/2 > rc, i.e. outside the cutoff.)"""
    dx = x - ox
    dx = dx - L * np.rint(dx * invL)
    dy = y - oy
    dy = dy - L * np.rint(dy * invL)
    dz = z - oz
    dz = dz - L * np.rint(dz * invL)
    r2 = (dx * dx + dy * dy) + dz * dz
    inv = 1.0 / (r2 if r2 > EPS else EPS)
    s = s2 * inv
    sr6 = (s * s) * s
    coef = (c24 * ((2.0 * sr6) * sr6 - sr6)) * inv
    if r2 < rc2:
        return coef * dx, coef * dy, coef * dz
    return 0.0, 0.0, 0.0


@njit(cache=True, fastmath={"reassoc"}, error_model="numpy")
def _lj_bead(x, y, z, Ox, Oy, Oz, L, invL, rc2, s2, c24):
    """Sum over the framework of the bead's LJ force; 'reassoc' only lets the SUM vectorise."""
    ax = 0.0
    ay = 0.0
    az = 0.0
    for k in range(Ox.shape[0]):
        tx, ty, tz = _pair(x, y, z, Ox[k], Oy[k], Oz[k], L, invL, rc2, s2, c24)
        ax += tx
        ay += ty
        az += tz
    return ax, ay, az


@njit(cache=True, error_model="numpy")
def _pair_e(x, y, z, ox, oy, oz, L, invL, rc2, s2, c4, v_rc):
    dx = x - ox
    dx = dx - L * np.rint(dx * invL)
    dy = y - oy
    dy = dy - L * np.rint(dy * invL)
    dz = z - oz
    dz = dz - L * np.rint(dz * invL)
    r2 = (dx * dx + dy * dy) + dz * dz
    inv = 1.0 / (r2 if r2 > EPS else EPS)
    s = s2 * inv
    sr6 = (s * s) * s
    v = (c4 * (sr6 * sr6 - sr6)) - v_rc
    if r2 < rc2:
        return v
    return 0.0


@njit(cache=True, fastmath={"reassoc"}, error_model="numpy")
def _lj_bead_energy(x, y, z, Ox, Oy, Oz, L, invL, rc2, s2, c4, v_rc):
    e = 0.0
    for k in range(Ox.shape[0]):
        e += _pair_e(x, y, z, Ox[k], Oy[k], Oz[k], L, invL, rc2, s2, c4, v_rc)
    return e


@njit(cache=True, error_model="numpy")
def _mol_force(q, i, Ox, Oy, Oz, fp, F):
    L = fp[F_L]
    invL = fp[F_INVL]
    rc2 = fp[F_RC2]
    s2 = fp[F_S2]
    c24 = fp[F_C24]
    kb = fp[F_KB]
    r0 = fp[F_R0]
    dx = q[i, 0, 0] - q[i, 1, 0]
    dy = q[i, 0, 1] - q[i, 1, 1]
    dz = q[i, 0, 2] - q[i, 1, 2]
    r = math.sqrt((dx * dx + dy * dy) + dz * dz)
    if r < EPS:
        r = EPS
    c = (-kb) * (r - r0)
    fbx = (c * dx) / r
    fby = (c * dy) / r
    fbz = (c * dz) / r
    ax, ay, az = _lj_bead(q[i, 0, 0], q[i, 0, 1], q[i, 0, 2], Ox, Oy, Oz, L, invL, rc2, s2, c24)
    F[i, 0, 0] = fbx + ax
    F[i, 0, 1] = fby + ay
    F[i, 0, 2] = fbz + az
    bx, by, bz = _lj_bead(q[i, 1, 0], q[i, 1, 1], q[i, 1, 2], Ox, Oy, Oz, L, invL, rc2, s2, c24)
    F[i, 1, 0] = (-fbx) + bx
    F[i, 1, 1] = (-fby) + by
    F[i, 1, 2] = (-fbz) + bz


@njit(cache=True, error_model="numpy")
def _mol_energy(q, i, Ox, Oy, Oz, fp):
    L = fp[F_L]
    invL = fp[F_INVL]
    dx = q[i, 0, 0] - q[i, 1, 0]
    dy = q[i, 0, 1] - q[i, 1, 1]
    dz = q[i, 0, 2] - q[i, 1, 2]
    r = math.sqrt((dx * dx + dy * dy) + dz * dz)
    e = (0.5 * fp[F_KB]) * ((r - fp[F_R0]) * (r - fp[F_R0]))
    e0 = _lj_bead_energy(q[i, 0, 0], q[i, 0, 1], q[i, 0, 2], Ox, Oy, Oz, L, invL, fp[F_RC2], fp[F_S2],
                         fp[F_C4], fp[F_VRC])
    e1 = _lj_bead_energy(q[i, 1, 0], q[i, 1, 1], q[i, 1, 2], Ox, Oy, Oz, L, invL, fp[F_RC2], fp[F_S2],
                         fp[F_C4], fp[F_VRC])
    return e + (e0 + e1)


@njit(cache=True, error_model="numpy")
def _wrap(p):
    """torch.remainder(p + pi, 2 pi) - pi (fmod form, bit-identical to ATen's CPU kernel)."""
    t = p + PI
    m = np.fmod(t, TWO_PI)
    if m != 0.0 and m < 0.0:
        m += TWO_PI
    return m - PI


@njit(cache=True, error_model="numpy")
def _bin(ph, dphi, ng):
    j = np.int64(np.floor((ph + PI) / dphi))
    return j % ng


@njit(cache=True, error_model="numpy")
def _region(ph, a, wh, cm):
    z = (abs(ph) * a) / TWO_PI
    if z > cm:
        return 0
    if z < wh:
        return 2
    return 1


@njit(cache=True, error_model="numpy")
def _true_event(step, i, x, r, a, wst, ist):
    cell = np.int64(np.floor(x / a))
    lc = wst[W_LCELL, i]
    if cell != lc:
        d = cell - lc
        ist[S_WCROSS] += d if d > 0 else -d
        wst[W_LCELL, i] = cell
    if r == 2:
        wst[W_EVWIN, i] = 1
        if ist[S_FIRSTWIN] < 0:
            ist[S_FIRSTWIN] = step
    elif r == 0:
        lcc = wst[W_LCAGE, i]
        if lcc == SENT:
            wst[W_LCAGE, i] = cell
        elif cell != lcc:
            ist[S_TRANSLOC] += 1
            wst[W_LCAGE, i] = cell
            wst[W_EVOPP, i] = 1
            if ist[S_FIRSTOPP] < 0:
                ist[S_FIRSTOPP] = step


@njit(cache=True, error_model="numpy")
def _init_labels(q, fp, wst):
    a = fp[F_A]
    for i in range(q.shape[0]):
        x = (q[i, 0, 0] + q[i, 1, 0]) * 0.5
        cell = np.int64(np.floor(x / a))
        r = _region(_wrap(fp[F_CPHI] * x), a, fp[F_WH], fp[F_CM])
        for w in range(NW):
            wst[w, i] = 0
        wst[W_LCELL, i] = cell
        wst[W_LCAGE, i] = cell if r == 0 else SENT


@njit(cache=True, error_model="numpy")
def _family_stats(lab, N, fam):
    for i in range(N):
        fam[lab[i]] += 1
    nu = 0
    ssq = 0
    mx = 0
    for i in range(N):
        c = fam[i]
        if c > 0:
            nu += 1
            ssq += c * c
            if c > mx:
                mx = c
            fam[i] = 0
    return nu, float(N) / float(ssq), float(mx) / float(N)


@njit(cache=True, error_model="numpy")
def _fr_score(phin, jn, N, ng, dphi, Kk, grid0, qv, lq, clip, cnt, pk, raw, sc, touched):
    """Uniform-target FR score at the post-move positions (alkanes.core._fr_score); returns
    (std ddof 0, max |s|) of the final score for the torch diagnostics."""
    nt = 0
    for i in range(N):
        j = jn[i]
        if cnt[j] == 0.0:
            touched[nt] = j
            nt += 1
        cnt[j] += 1.0
    # ascending occupied bins: the sum over j of cnt_j K[j, k] in ascending j (zero bins add +0.0)
    for t in range(1, nt):
        v = touched[t]
        u = t - 1
        while u >= 0 and touched[u] > v:
            touched[u + 1] = touched[u]
            u -= 1
        touched[u + 1] = v
    for k in range(ng):
        pk[k] = 0.0
    for t in range(nt):
        j = touched[t]
        c = cnt[j]
        for k in range(ng):
            pk[k] += c * Kk[j, k]
        cnt[j] = 0.0
    tot = 0.0
    for k in range(ng):
        v = pk[k]
        if v < 0.0:
            v = 0.0
        pk[k] = v
        tot += v
    mass = tot * dphi
    if mass < EPS:
        mass = EPS
    for k in range(ng):
        pk[k] = pk[k] / mass
    kl = 0.0
    for k in range(ng):
        v = pk[k]
        kl += v * (math.log(v if v > EPS else EPS) - lq)
    kl = kl * dphi
    for i in range(N):
        xx = (phin[i] - grid0) / dphi
        f0 = np.floor(xx)
        fr = xx - f0
        i0 = np.int64(f0)
        p0 = pk[i0 % ng]
        p1 = pk[(i0 + 1) % ng]
        pat = (1.0 - fr) * p0 + fr * p1
        qat = (1.0 - fr) * qv + fr * qv
        raw[i] = (math.log(pat if pat > EPS else EPS) - math.log(qat if qat > EPS else EPS)) - kl
    m = 0.0
    for i in range(N):
        m += raw[i]
    m = m / N
    for i in range(N):
        sc[i] = raw[i] - m
    for _ in range(3):
        m = 0.0
        for i in range(N):
            v = sc[i]
            if v < -clip:
                v = -clip
            elif v > clip:
                v = clip
            sc[i] = v
            m += v
        m = m / N
        for i in range(N):
            sc[i] = sc[i] - m
    m = 0.0
    amax = 0.0
    for i in range(N):
        m += sc[i]
        if abs(sc[i]) > amax:
            amax = abs(sc[i])
    m = m / N
    v2 = 0.0
    for i in range(N):
        d = sc[i] - m
        v2 += d * d
    return math.sqrt(v2 / N), amax


@njit(cache=True, error_model="numpy")
def fr_select_native(sc, N, cap, rate, dt_eff, ublk, cand, dsel, ssel, dp, cdf):
    """The engine's own FR selection for one opportunity (score ``sc`` given, uniforms ``ublk`` of
    length N + 2 cap).  Fills dsel[:n] (dying slots) and ssel[:n] (sources); returns n.  Exposed for
    the law tests (E2); the kernel calls it."""
    sbw = 0.0
    sdw = 0.0
    for i in range(N):
        s = sc[i]
        dw = s if s > 0.0 else 0.0
        bw = -s if -s > 0.0 else 0.0
        if dw > 0.0:
            dp[i] = 1.0 - math.exp(((-rate) * dw) * dt_eff)
        else:
            dp[i] = 0.0
        cdf[i] = bw
        sbw += bw
        sdw += dw
    if sbw <= EPS or sdw <= EPS:
        return 0
    n = 0
    for i in range(N):
        if ublk[i] < dp[i]:
            cand[n] = i
            n += 1
    if n == 0:
        return 0
    if n > cap:
        for t in range(cap):        # partial Fisher-Yates: a uniform random ORDERED cap-subset
            m = n - t
            jj = np.int64(ublk[N + t] * m)
            if jj > m - 1:
                jj = m - 1
            jj += t
            tmp = cand[t]
            cand[t] = cand[jj]
            cand[jj] = tmp
        n = cap
    tot = 0.0
    for i in range(N):
        tot += cdf[i]
        cdf[i] = tot
    for t in range(n):
        x = ublk[N + cap + t] * tot
        lo = 0
        hi = N - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            if cdf[mid] > x:
                hi = mid
            else:
                lo = mid + 1
        ssel[t] = lo
        dsel[t] = cand[t]
    return n


@njit(cache=True, error_model="numpy")
def _run_chunk(n_eval, fp, ip, Ox, Oy, Oz, Kk, save_steps, trace_steps, noise, frb, rp_u, rp_perm, rp_src,
               q, anc, ancw, wst, acc, ist, fst, sv_acc, sv_reg, sv_i, sv_f, sv_evh, traces, traces_anc,
               evh, ev_counts, ev_log):
    """Advance ``n_eval`` evaluated steps in place.  The state between calls sits between the FR
    opportunity of step k-1 -> k and the evaluation of step k (ist[S_STEP] = k)."""
    N = ip[I_N]
    ng = ip[I_NG]
    n_steps = ip[I_NSTEPS]
    warm = ip[I_WARM]
    burn = ip[I_BURN]
    fr_start = ip[I_FRSTART]
    fr_every = ip[I_FREVERY]
    cap = ip[I_CAP]
    is_fr = ip[I_ISFR] != 0
    fr_on = ip[I_FRON] != 0
    gwin = ip[I_GWIN]
    accu = ip[I_ACCU] != 0
    diag = ip[I_DIAG] != 0
    mode = ip[I_MODE]
    ntr = ip[I_NTR]
    rec = ip[I_REC] != 0
    dt = fp[F_DT]
    ns = fp[F_NS]
    a = fp[F_A]
    cphi = fp[F_CPHI]
    cf = fp[F_CF]
    gpx = fp[F_GPX]
    dphi = fp[F_DPHI]
    fclip = fp[F_FCLIP]
    dclip = fp[F_DCLIP]
    bscale = fp[F_BSCALE]
    wh = fp[F_WH]
    cm = fp[F_CM]
    S = save_steps.shape[0]
    K = trace_steps.shape[0]
    F = np.empty((N, 2, 3))
    phi = np.empty(N)
    xc = np.empty(N)
    fl = np.empty(N)
    jb = np.empty(N, np.int64)
    bxs = np.empty(N)
    reg = np.empty(N, np.int64)
    uu = np.zeros(N)
    tf = np.zeros(ng)
    tc = np.zeros(ng)
    tu = np.zeros(ng)
    touched = np.empty(ng, np.int64)
    phin = np.empty(N)
    xn = np.empty(N)
    jn = np.empty(N, np.int64)
    sc = np.empty(N)
    raw = np.empty(N)
    dp = np.empty(N)
    cdf = np.empty(N)
    cand = np.empty(N, np.int64)
    dsel = np.empty(N, np.int64)
    ssel = np.empty(N, np.int64)
    cnt = np.zeros(ng)
    pk = np.empty(ng)
    fam = np.zeros(N, np.int64)
    imove = 0
    iopp = 0
    for _it in range(n_eval):
        step = ist[S_STEP]
        # (1) forces, CV, local mean force at the PRE-move positions
        for i in range(N):
            _mol_force(q, i, Ox, Oy, Oz, fp, F)
            x = (q[i, 0, 0] + q[i, 1, 0]) * 0.5
            xc[i] = x
            ph = _wrap(cphi * x)
            phi[i] = ph
            f = cf * (F[i, 0, 0] + F[i, 1, 0])
            if f < -dclip:
                f = -dclip
            elif f > dclip:
                f = dclip
            fl[i] = f
            jb[i] = _bin(ph, dphi, ng)
        ist[S_NFE] += N
        # (2) deposits of ALL walkers before any bias is read (per-step bin sums, then added)
        prod = step >= burn
        if prod and accu:
            for i in range(N):
                uu[i] = _mol_energy(q, i, Ox, Oy, Oz, fp)
        nt = 0
        for i in range(N):
            j = jb[i]
            if tc[j] == 0.0:
                touched[nt] = j
                nt += 1
            tf[j] += fl[i]
            tc[j] += 1.0
            if prod and accu:
                tu[j] += uu[i]
        for t in range(nt):
            j = touched[t]
            acc[A_FS, j] += tf[j]
            acc[A_CS, j] += tc[j]
            if prod:
                acc[A_FSP, j] += tf[j]
                acc[A_CSP, j] += tc[j]
                if accu:
                    acc[A_USP, j] += tu[j]
            tf[j] = 0.0
            tc[j] = 0.0
            tu[j] = 0.0
        # (3) bias from the ALL-STEPS accumulators (this step's deposit included)
        ramp = step / (warm if warm > 1 else 1)
        if ramp > 1.0:
            ramp = 1.0
        scl = bscale * ramp
        for i in range(N):
            j = jb[i]
            c = acc[A_CS, j]
            if c > 0.0:
                G = acc[A_FS, j] / (c if c > 1.0 else 1.0)
            else:
                G = 0.0
            if G < -fclip:
                G = -fclip
            elif G > fclip:
                G = fclip
            bxs[i] = (scl * G) * gpx
        # (4) region bookkeeping (torch) + true events, pre-move positions
        if diag:
            prevok = ist[S_PREVOK] != 0
            for i in range(N):
                r = _region(phi[i], a, wh, cm)
                reg[i] = r
                if prevok:
                    if r != wst[W_PREV, i]:
                        ist[S_TRANS] += 1
                    if wst[W_HLC, i] != 0 and r == 0:
                        wst[W_REPC, i] += 1
                if r == 0:
                    wst[W_HLC, i] = 0
                elif r == 2:
                    wst[W_HLC, i] = 1
                wst[W_PREV, i] = r
                _true_event(step, i, xc[i], r, a, wst, ist)
            ist[S_PREVOK] = 1
        # (5) save (state at the START of this step; accumulators include this step)
        sp = ist[S_SP]
        if sp < S and save_steps[sp] == step:
            for k in range(ng):
                sv_acc[sp, 0, k] = acc[A_FS, k]
                sv_acc[sp, 1, k] = acc[A_CS, k]
                sv_acc[sp, 2, k] = acc[A_FSP, k]
                sv_acc[sp, 3, k] = acc[A_CSP, k]
            sv_i[sp, V_DEATHS] = ist[S_REPL]
            sv_i[sp, V_OPPS] = ist[S_NOPP]
            sv_i[sp, V_OPPSEV] = ist[S_NOPPEV]
            for k in range(evh.shape[0]):
                sv_evh[sp, k] = evh[k]
            if diag:
                for k in range(ng):
                    sv_acc[sp, 4, k] = 0.0
                n0 = 0
                n1 = 0
                n2 = 0
                new = 0
                neo = 0
                rcx = 0
                for i in range(N):
                    sv_acc[sp, 4, jb[i]] += 1.0
                    if reg[i] == 0:
                        n0 += 1
                    elif reg[i] == 1:
                        n1 += 1
                    else:
                        n2 += 1
                    new += wst[W_EVWIN, i]
                    neo += wst[W_EVOPP, i]
                    rcx += wst[W_REPC, i]
                sv_reg[sp, 0] = n0 / N
                sv_reg[sp, 1] = n1 / N
                sv_reg[sp, 2] = n2 / N
                sv_i[sp, V_WCROSS] = ist[S_WCROSS]
                sv_i[sp, V_TRANSLOC] = ist[S_TRANSLOC]
                sv_i[sp, V_EVWIN] = new
                sv_i[sp, V_EVOPP] = neo
                sv_i[sp, V_TRANS] = ist[S_TRANS]
                sv_i[sp, V_CAGEX] = rcx
                if is_fr:
                    nu, ess, mxf = _family_stats(anc, N, fam)
                    sv_i[sp, V_NU] = nu
                    sv_f[sp, U_ESS] = ess
                    sv_f[sp, U_MAXF] = mxf
                    nu, ess, mxf = _family_stats(ancw, N, fam)
                    sv_i[sp, V_NUW] = nu
                    sv_f[sp, U_ESSW] = ess
                    sv_f[sp, U_MAXFW] = mxf
                else:
                    sv_i[sp, V_NU] = N
                    sv_i[sp, V_NUW] = N
            ist[S_SP] = sp + 1
        tp = ist[S_TP]
        if diag and tp < K and trace_steps[tp] == step:
            for k in range(ntr):
                traces[tp, k] = phi[k]
                traces_anc[tp, k] = anc[k]
            ist[S_TP] = tp + 1
        if is_fr and gwin > 0 and step > 0 and step % gwin == 0:
            for i in range(N):
                ancw[i] = i
        if step >= n_steps:
            ist[S_STEP] = step + 1
            break
        # (6) Euler-Maruyama move
        for i in range(N):
            bxi = bxs[i]
            for b in range(2):
                q[i, b, 0] = (q[i, b, 0] + dt * (F[i, b, 0] + bxi)) + ns * noise[imove, i, b, 0]
                q[i, b, 1] = (q[i, b, 1] + dt * F[i, b, 1]) + ns * noise[imove, i, b, 1]
                q[i, b, 2] = (q[i, b, 2] + dt * F[i, b, 2]) + ns * noise[imove, i, b, 2]
        imove += 1
        nxt = step + 1
        # (7) Fisher-Rao at the POST-move positions
        if is_fr and nxt >= fr_start and (nxt - fr_start) % fr_every == 0:
            for i in range(N):
                x = (q[i, 0, 0] + q[i, 1, 0]) * 0.5
                xn[i] = x
                ph = _wrap(cphi * x)
                phin[i] = ph
                jn[i] = _bin(ph, dphi, ng)
            if diag:
                for i in range(N):
                    _true_event(nxt, i, xn[i], _region(phin[i], a, wh, cm), a, wst, ist)
            ist[S_NOPP] += 1
            sstd, sabs = _fr_score(phin, jn, N, ng, dphi, Kk, fp[F_GRID0], fp[F_QV], fp[F_LQ], fp[F_SCLIP],
                                   cnt, pk, raw, sc, touched)
            fst[G_SSTD] += sstd
            if sabs > fst[G_SMAX]:
                fst[G_SMAX] = sabs
            ist[S_NSCORE] += 1
            nev = 0
            if fr_on:
                if mode == 0:
                    nev = fr_select_native(sc, N, cap, fp[F_RATE], fp[F_DTEFF], frb[iopp], cand, dsel, ssel,
                                           dp, cdf)
                    iopp += 1
                else:
                    # replay of torch's draws: rand(N) always; randperm(n) if n > cap; multinomial idx
                    urow = rp_u[ist[S_RPU]]
                    ist[S_RPU] += 1
                    rate = fp[F_RATE]
                    dte = fp[F_DTEFF]
                    sbw = 0.0
                    sdw = 0.0
                    for i in range(N):
                        s = sc[i]
                        dw = s if s > 0.0 else 0.0
                        bw = -s if -s > 0.0 else 0.0
                        dp[i] = 1.0 - math.exp(((-rate) * dw) * dte) if dw > 0.0 else 0.0
                        sbw += bw
                        sdw += dw
                    if not (sbw <= EPS or sdw <= EPS):
                        n = 0
                        for i in range(N):
                            if urow[i] < dp[i]:
                                cand[n] = i
                                n += 1
                        if n > 0:
                            if n > cap:
                                p0 = ist[S_RPP]
                                for t in range(cap):
                                    dsel[t] = cand[rp_perm[p0 + t]]
                                for t in range(cap):
                                    cand[t] = dsel[t]
                                ist[S_RPP] = p0 + n
                                n = cap
                            p0 = ist[S_RPS]
                            for t in range(n):
                                ssel[t] = rp_src[p0 + t]
                                dsel[t] = cand[t]
                                if not (sc[ssel[t]] < 0.0):
                                    ist[S_BADSRC] += 1
                            ist[S_RPS] = p0 + n
                            nev = n
            for t in range(nev):
                d = dsel[t]
                s_ = ssel[t]
                acc[A_BH, jn[s_]] += 1.0
                acc[A_DH, jn[d]] += 1.0
                for b in range(2):
                    for c3 in range(3):
                        q[d, b, c3] = q[s_, b, c3]
                anc[d] = anc[s_]
                ancw[d] = ancw[s_]
                for w in range(NW):
                    wst[w, d] = wst[w, s_]
                if rec:
                    e = ist[S_EVLOG]
                    if e < ev_log.shape[0]:
                        ev_log[e, 0] = nxt
                        ev_log[e, 1] = d
                        ev_log[e, 2] = s_
                    ist[S_EVLOG] = e + 1
            evh[nev] += 1
            if nev > 0:
                ist[S_NOPPEV] += 1
                ist[S_REPL] += nev
            if rec:
                o = ist[S_NOPP] - 1
                if o < ev_counts.shape[0]:
                    ev_counts[o] = nev
        ist[S_STEP] = nxt
    return imove, iopp


# =====================================================================================
# Python driver
# =====================================================================================
class InterruptedRun(RuntimeError):
    """Raised by the ``_stop_after_chunks`` test hook after the checkpoint of that chunk."""


def framework(path=None):
    z = np.load(os.path.join(ROOT, path or DEFAULT_CFG["framework_npz"]), allow_pickle=True)
    O = np.ascontiguousarray(z["o_pos"], dtype=np.float64)
    return O, float(z["a_pseudo"]), float(z["box"])


def make_cfg(T_K, N, seed, **overrides):
    c = dict(DEFAULT_CFG)
    c.update(T_K=float(T_K), N=int(N), seed=int(seed))
    for k, v in overrides.items():
        if k not in DEFAULT_CFG:
            raise KeyError(f"unknown cfg key {k!r}")
        c[k] = v
    return resolve_cfg(c)


def resolve_cfg(cfg):
    """Defaults filled, derived integers computed (n_steps, cap, window steps), validated."""
    c = dict(DEFAULT_CFG)
    for k, v in cfg.items():
        if k not in DEFAULT_CFG and k not in ("cap", "genealogy_window_steps", "n_trace"):
            raise KeyError(f"unknown cfg key {k!r}")
        c[k] = v
    # deposit clip: core_lta clamps f_loc at +-8 abf_force_clip; a different value must be asked for explicitly
    c["abf_force_clip"] = float(c["abf_force_clip"])
    c["deposit_clip_override"] = bool(c["deposit_clip_override"])
    dc_torch = 8.0 * c["abf_force_clip"]
    if c["deposit_clip"] is None:
        c["deposit_clip"] = dc_torch
    else:
        c["deposit_clip"] = float(c["deposit_clip"])
        if c["deposit_clip"] != dc_torch and not c["deposit_clip_override"]:
            raise ValueError(f"deposit_clip {c['deposit_clip']} != 8 x abf_force_clip = {dc_torch} (core_lta's rule); "
                             f"leave deposit_clip unset to derive it, or set deposit_clip_override=True")
    N = int(c["N"])
    if N < 1:
        raise ValueError("N must be >= 1")
    c["N"] = N
    c["seed"] = int(c["seed"])
    c["T_K"] = float(c["T_K"])
    if c["n_steps"] is None:
        if int(c["budget"]) % N:
            raise ValueError(f"budget {c['budget']} not divisible by N {N}")
        c["n_steps"] = int(c["budget"]) // N
    c["n_steps"] = int(c["n_steps"])
    for k in ("n_grid", "warmup_steps", "burn_in_steps", "fr_start_steps", "fr_every", "cap_min",
              "n_trace_max", "max_traces"):
        c[k] = int(c[k])
    if c["fr_every"] < 1:
        raise ValueError("fr_every must be >= 1")
    c["cap"] = max(int(c["cap_min"]), int(float(c["max_event_fraction"]) * N))
    c["genealogy_window_steps"] = int(round(float(c["genealogy_window_tu"]) / float(c["h"])))
    c["n_trace"] = min(N, int(c["n_trace_max"]))
    return c


def seed_streams(seed, N, T_K):
    """SeedSequence([seed, N, round(T_K)]) and its three PCG64 children (IC, noise, FR)."""
    ss = np.random.SeedSequence([int(seed), int(N), int(round(float(T_K)))])
    kids = ss.spawn(3)
    gens = [np.random.Generator(np.random.PCG64(k)) for k in kids]
    info = dict(entropy=[int(seed), int(N), int(round(float(T_K)))],
                spawn_keys=[list(k.spawn_key) for k in kids],
                streams={"ic": "child 0", "noise": "child 1", "fr": "child 2"})
    return gens, info


def initial_conditions(gen, N, a, L, r0=1.54):
    """Law of LTASystem.initial_conditions for one run of N molecules."""
    S = int(round(L / a))
    cages = np.array([[i + 0.5, j + 0.5, k + 0.5] for i in range(S) for j in range(S) for k in range(S)],
                     dtype=np.float64) * a
    pick = gen.integers(0, cages.shape[0], size=N)
    com = cages[pick] + 0.5 * gen.standard_normal((N, 3))
    u = gen.standard_normal((N, 3))
    u = u / np.maximum(np.sqrt((u * u).sum(-1, keepdims=True)), EPS)
    half = 0.5 * r0
    return np.ascontiguousarray(np.stack([com + half * u, com - half * u], axis=1))


def kde_matrix(ng, bw):
    """alkanes.periodic.wrapped_gaussian_kernel_matrix on the cell-centred grid (unnormalised)."""
    dphi = TWO_PI / ng
    grid = -PI + (np.arange(ng, dtype=np.float64) + 0.5) * dphi
    d = grid[:, None] - grid[None, :]
    d = d - TWO_PI * np.rint(d / TWO_PI)
    return np.ascontiguousarray(np.exp(-0.5 * (d / max(float(bw), EPS)) ** 2)), grid, dphi


def mean_force(M, C):
    """P0 bin force Gamma = M / max(C, 1) on populated bins, 0 elsewhere (core_lta.histogram_mean_force)."""
    M = np.asarray(M, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    return np.where(C > 0, M / np.maximum(C, 1.0), 0.0)


def histogram_pmf(G, dphi=TWO_PI / 180):
    """core_lta.histogram_pmf: exact integral of the piecewise-constant force at the cell centres."""
    g0 = G - G.mean(-1, keepdims=True)
    Fr = np.cumsum(g0 * dphi, axis=-1)
    Fc = Fr - 0.5 * g0 * dphi
    return Fc - Fc.mean(-1, keepdims=True)


def budget_save_grid(n_steps, n_lin=200, n_log=24, u_min=1e-4):
    """Completed-step counts at shared budget fractions u = b/B (gateway_numba.budget_save_grid):
    k/n_lin plus log-spaced fractions in [u_min, 1/n_lin).  Returns (save_steps, u)."""
    u_lin = np.arange(1, n_lin + 1) / n_lin
    u_log = np.logspace(np.log10(u_min), np.log10(1.0 / n_lin), n_log, endpoint=False)
    u = np.unique(np.concatenate([u_log, u_lin]))
    steps = np.unique(np.clip(np.round(u * n_steps).astype(np.int64), 1, n_steps))
    return steps, steps / float(n_steps)


def trace_grid(c):
    """Physical-time trace cadence: every trace_dense_every_tu up to trace_dense_until_tu, then every
    m * trace_sparse_every_tu with the smallest integer m >= 1 keeping K <= max_traces.
    Returns (steps (K,), m)."""
    h = float(c["h"])
    n_steps = int(c["n_steps"])
    de = max(1, int(round(float(c["trace_dense_every_tu"]) / h)))
    du = int(round(float(c["trace_dense_until_tu"]) / h))
    se = max(1, int(round(float(c["trace_sparse_every_tu"]) / h)))
    dense = np.arange(0, min(du, n_steps) + 1, de, dtype=np.int64)
    m = 1
    while True:
        start = (dense[-1] // (se * m) + 1) * (se * m)
        sparse = np.arange(start, n_steps + 1, se * m, dtype=np.int64)
        if dense.size + sparse.size <= int(c["max_traces"]) or m > 10 ** 6:
            break
        m += 1
    return np.unique(np.concatenate([dense, sparse])), m


def _n_opps_in(lo, hi, fr_start, fr_every):
    """Number of FR opportunities nxt in [lo, hi]."""
    lo2 = max(lo, fr_start)
    if hi < lo2:
        return 0
    first = fr_start + ((lo2 - fr_start + fr_every - 1) // fr_every) * fr_every
    if first > hi:
        return 0
    return (hi - first) // fr_every + 1


def _params(c, method, O, a, L, mode):
    T = float(c["T_K"])
    beta = 1.0 / (KB * T)
    h = float(c["h"])
    eps = float(c["eps_go_K"]) * KB
    sig = float(c["sigma_go"])
    rc = float(c["rc"])
    sr6c = (sig / rc) ** 6
    v_rc = 4.0 * eps * (sr6c * sr6c - sr6c)
    ng = int(c["n_grid"])
    dphi = TWO_PI / ng
    grid0 = -PI + (0.0 + 0.5) * dphi
    qv = 1.0 / max(float(ng) * dphi, EPS)       # normalize_density(ones): sum(ones) = ng exactly
    fp = np.zeros(NFP)
    fp[F_DT] = h
    fp[F_NS] = math.sqrt(2.0 * h / beta)
    fp[F_A] = a
    fp[F_L] = L
    fp[F_INVL] = 1.0 / L
    fp[F_RC2] = rc ** 2
    fp[F_S2] = sig ** 2
    fp[F_C24] = 24.0 * eps
    fp[F_KB] = float(c["k_bond"])
    fp[F_R0] = float(c["r0_bond"])
    fp[F_C4] = 4.0 * eps
    fp[F_VRC] = v_rc
    fp[F_CPHI] = 2.0 * PI / a
    fp[F_CF] = -(a / (2.0 * PI))
    fp[F_GPX] = PI / a
    fp[F_DPHI] = dphi
    fp[F_FCLIP] = float(c["abf_force_clip"])
    fp[F_DCLIP] = float(c["deposit_clip"])
    fp[F_RATE] = float(c["fr_rate"])
    fp[F_DTEFF] = h * max(int(c["fr_every"]), 1)
    fp[F_SCLIP] = float(c["score_clip"])
    fp[F_WH] = float(c["window_half"])
    fp[F_CM] = float(c["cage_min"])
    fp[F_BSCALE] = float(c["abf_bias_scale"])
    fp[F_GRID0] = grid0
    fp[F_QV] = qv
    fp[F_LQ] = math.log(max(qv, EPS))
    is_fr = method == "fr"
    fr_on = is_fr and int(c["cap"]) >= 1 and float(c["fr_rate"]) > 0.0
    ip = np.zeros(NIP, dtype=np.int64)
    ip[I_N] = int(c["N"])
    ip[I_NG] = ng
    ip[I_NSTEPS] = int(c["n_steps"])
    ip[I_WARM] = int(c["warmup_steps"])
    ip[I_BURN] = int(c["burn_in_steps"])
    ip[I_FRSTART] = int(c["fr_start_steps"])
    ip[I_FREVERY] = int(c["fr_every"])
    ip[I_CAP] = int(c["cap"])
    ip[I_ISFR] = 1 if is_fr else 0
    ip[I_FRON] = 1 if fr_on else 0
    ip[I_GWIN] = int(c["genealogy_window_steps"])
    ip[I_ACCU] = 1 if c["accumulate_u"] else 0
    ip[I_DIAG] = 1 if c["diagnostics"] else 0
    ip[I_MODE] = mode
    ip[I_NTR] = int(c["n_trace"])
    ip[I_REC] = 1 if c["record_events"] else 0
    return fp, ip, beta, fr_on


_STATE_KEYS = ("q", "anc", "ancw", "wst", "acc", "ist", "fst", "sv_acc", "sv_reg", "sv_i", "sv_f", "sv_evh",
               "traces", "traces_anc", "evh", "ev_counts", "ev_log")


def _new_state(c, q0, S, K, n_opps_total):
    N = int(c["N"])
    ng = int(c["n_grid"])
    cap = int(c["cap"])
    st = dict(
        q=np.ascontiguousarray(q0, dtype=np.float64).copy(),
        anc=np.arange(N, dtype=np.int64),
        ancw=np.arange(N, dtype=np.int64),
        wst=np.zeros((NW, N), dtype=np.int64),
        acc=np.zeros((NA, ng)),
        ist=np.zeros(NIST, dtype=np.int64),
        fst=np.zeros(NFST),
        sv_acc=np.zeros((S, 5, ng)),
        sv_reg=np.full((S, 3), np.nan),
        sv_i=np.full((S, NV), -1, dtype=np.int64),
        sv_f=np.full((S, NU), np.nan),
        sv_evh=np.zeros((S, cap + 1), dtype=np.int64),
        traces=np.full((K, int(c["n_trace"])), np.nan),
        traces_anc=np.full((K, int(c["n_trace"])), -1, dtype=np.int64),
        evh=np.zeros(cap + 1, dtype=np.int64),
        ev_counts=np.zeros(n_opps_total if c["record_events"] else 0, dtype=np.int32),
        ev_log=np.zeros((n_opps_total * max(cap, 1) if c["record_events"] else 0, 3), dtype=np.int64),
    )
    st["sv_acc"][:, 4, :] = np.nan
    st["ist"][S_FIRSTWIN] = -1
    st["ist"][S_FIRSTOPP] = -1
    return st


def _host_cpu():
    try:
        import llvmlite.binding as ll
        return ll.get_host_cpu_name()
    except Exception:
        return platform.processor() or "unknown"


def _fingerprint(c, method, save_steps, trace_steps, mode):
    """Job identity: the resolved cfg, the arm, the save / trace grids and the randomness mode."""
    h = hashlib.sha256()
    h.update(json.dumps(c, sort_keys=True, default=str).encode())
    h.update(method.encode())
    h.update(np.ascontiguousarray(save_steps, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(trace_steps, dtype=np.int64).tobytes())
    h.update(str(mode).encode())
    return h.hexdigest()


_ENGINE_FIELDS = ("engine", "engine_sha256")
_PLATFORM_FIELDS = ("numpy", "numba", "compile_target")


def run_signature(c, method, save_steps, trace_steps, mode=0):
    """What a checkpoint must match to be resumed BITWISE: the job fingerprint, the engine code, the numpy /
    numba versions (numpy does not promise Generator.standard_normal streams across versions) and the numba
    compile target (the 'reassoc' LJ sum)."""
    tgt = compile_target()
    return dict(job=_fingerprint(resolve_cfg(c), method, save_steps, trace_steps, mode), engine=ENGINE,
                engine_sha256=_PROVENANCE["engine_sha256"], numpy=np.__version__, numba=numba.__version__,
                compile_target=dict(triple=tgt["triple"], cpu_name=tgt["cpu_name"],
                                    cpu_features_sha256=tgt["cpu_features_sha256"]))


def signature_mismatch(old, new):
    """Sorted list of the signature fields that differ (['signature missing'] if ``old`` is not a dict)."""
    if not isinstance(old, dict):
        return ["signature missing"]
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


def _proc_status_mb(field):
    """A /proc/self/status memory field (VmHWM, VmRSS) in MB, or None (non-Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _reset_peak_rss():
    """Reset this process's VmHWM to its current RSS (Linux >= 4.0: write 5 to /proc/self/clear_refs), so that a
    later VmHWM is the peak of THIS run, not of earlier work in the same (pooled / forked) process."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def _peak_rss_mb(reset_ok):
    v = _proc_status_mb("VmHWM") if reset_ok else None
    if v is not None:
        return v, "VmHWM after reset at run start"
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, "ru_maxrss (process lifetime; fallback)"


def _write_checkpoint(path, st, extra):
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp{os.getpid()}")
    arrays = {k: st[k] for k in _STATE_KEYS}
    arrays["extra_json"] = np.array(json.dumps(extra))
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_checkpoint(path):
    with np.load(path, allow_pickle=False) as z:
        st = {k: np.array(z[k]) for k in _STATE_KEYS}
        extra = json.loads(str(z["extra_json"]))
    return st, extra


def run_arm(cfg, method, save_steps, *, checkpoint_path=None, checkpoint_every_s=900.0, chunk_steps=None,
            trace_steps=None, replay=None, allow_engine_change=False, allow_platform_change=False,
            _stop_after_chunks=None):
    """Run ONE arm ('abf' or 'fr' = uniform-target FR) of one (lta, N, seed) job; returns the result dict.

    cfg          dict with at least T_K, N, seed (every other knob defaults to DEFAULT_CFG; see make_cfg)
    save_steps   sorted unique completed-step counts in [0, n_steps] (the caller's save grid)
    checkpoint_path / checkpoint_every_s
                 atomic checkpoint (np.savez to a temp file + os.replace) after a chunk once
                 checkpoint_every_s of wall time has passed since the last one, and at completion;
                 an existing checkpoint whose signature (run_signature: job fingerprint, ENGINE, engine
                 sha256, numpy / numba versions, compile target) matches is resumed BITWISE; a checkpoint
                 of a different job is refused, and so is any other mismatch unless
    allow_engine_change    tolerates a different engine sha256 / ENGINE (e.g. a comment edit), or
    allow_platform_change  tolerates different numpy / numba versions or compile target (the run is then
                 no longer bitwise reproducible across the cut; recorded in sessions_json).
    chunk_steps  evaluated steps per kernel call (default max(1, 1_000_000 // N))
    trace_steps  override of the physical-time trace cadence (tests)
    replay       dict(q0 (N,2,3), noise (n_steps,N,2,3), fr_u (rows,N), fr_perm (flat int), fr_src (flat int))
                 replays torch-recorded draws (equivalence test only)

    Result keys (arrays; S saves, nb = n_grid bins, K traces, n_tr traced slots):
      save_step, save_t, save_u; M_all, C_all, M_prod, C_prod, hist_inst (S, nb); region_frac (S, 3)
      [cage |z| > 4, neck, window |z| < 1.5]; events_window_crossings, events_translocations,
      lineage_ever_window, lineage_ever_opposite (S,); first_window_step, first_opposite_step;
      gen_n_unique, gen_ess, gen_max_frac, gen_n_unique_win, gen_ess_win, gen_max_frac_win (S,);
      cum_deaths, cum_opps, cum_opps_with_event (S,); events_per_opp_hist (S, cap+1);
      traces_step (K,), traces, traces_anc (K, n_tr); torch-equivalent n_transitions,
      n_cage_crossings_lineage (S,), birth_hist, death_hist, total_replacement_events, fr_score_std,
      fr_score_absmax; final state q_final, anc_final, M/C_all_final, M/C_prod_final; cost
      n_force_evals, wall_s, peak_rss_mb, n_checkpoints_written, resumed_from, sessions_json; meta_json,
      cfg_json.  diagnostics=False omits DIAG_ONLY_KEYS (listed in meta.omitted_keys).
    """
    hwm_before = _proc_status_mb("VmHWM")
    reset_ok = _reset_peak_rss()
    rss_start = _proc_status_mb("VmRSS")
    run_start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    c = resolve_cfg(cfg)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    N = c["N"]
    if method == "fr" and N < 2:
        raise ValueError("no FR arm at N = 1 (the uniform score is identically 0): run ABF only")
    n_steps = c["n_steps"]
    h = float(c["h"])
    save_steps = np.asarray(save_steps, dtype=np.int64).ravel()
    if save_steps.size and (np.any(np.diff(save_steps) <= 0) or save_steps[0] < 0 or save_steps[-1] > n_steps):
        raise ValueError("save_steps must be sorted, unique and inside [0, n_steps]")
    if trace_steps is None:
        trace_steps, trace_mult = trace_grid(c)
    else:
        trace_steps = np.asarray(trace_steps, dtype=np.int64).ravel()
        trace_mult = None
        if trace_steps.size and (np.any(np.diff(trace_steps) <= 0) or trace_steps[0] < 0 or trace_steps[-1] > n_steps):
            raise ValueError("trace_steps must be sorted, unique and inside [0, n_steps]")
    mode = 1 if replay is not None else 0
    O, a, L = framework(c["framework_npz"])
    Ox, Oy, Oz = (np.ascontiguousarray(O[:, k]) for k in range(3))
    fp, ip, beta, fr_on = _params(c, method, O, a, L, mode)
    Kk, _, dphi = kde_matrix(int(c["n_grid"]), float(c["kde_bandwidth"]))
    cap = int(c["cap"])
    fr_start, fr_every = int(c["fr_start_steps"]), int(c["fr_every"])
    n_opps_total = _n_opps_in(1, n_steps, fr_start, fr_every) if method == "fr" else 0
    if chunk_steps is None:
        chunk_steps = max(1, 1_000_000 // N)
    chunk_steps = int(chunk_steps)
    if chunk_steps < 1:
        raise ValueError("chunk_steps must be >= 1")
    sig = run_signature(c, method, save_steps, trace_steps, mode)
    fpr = sig["job"]
    gens, rng_info = seed_streams(c["seed"], N, c["T_K"])
    gen_ic, gen_noise, gen_fr = gens
    session = dict(pid=os.getpid(), host=platform.node(), run_start_utc=run_start_utc, start_step=0,
                   engine=ENGINE, engine_sha256=_PROVENANCE["engine_sha256"], git_commit=_PROVENANCE["git_commit"],
                   git_dirty=_PROVENANCE["git_dirty"], imported_utc=_PROVENANCE["imported_utc"],
                   numpy=np.__version__, numba=numba.__version__, compile_target=sig["compile_target"],
                   tolerated_mismatch=[], vmhwm_before_mb=hwm_before, vmrss_start_mb=rss_start, peak_reset=reset_ok,
                   peak_rss_mb=None, wall_s=0.0, end_step=None)

    resumed_from = []
    sessions = []
    n_ckpt = 0
    wall_prev = 0.0
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        st, extra = _read_checkpoint(checkpoint_path)
        old = extra.get("signature")
        if not isinstance(old, dict):
            raise ValueError(f"checkpoint {checkpoint_path} carries no run signature (written by an older engine); "
                             f"delete it to restart the job")
        mism = signature_mismatch(old, sig)
        if "job" in mism:
            raise ValueError(f"checkpoint {checkpoint_path} belongs to a different job (fingerprint mismatch)")
        refused = [k for k in mism if not ((k in _ENGINE_FIELDS and allow_engine_change)
                                           or (k in _PLATFORM_FIELDS and allow_platform_change))]
        if refused:
            raise ValueError(f"checkpoint {checkpoint_path} was written under a different {refused} "
                             f"(checkpoint {[old.get(k) for k in refused]} vs now {[sig.get(k) for k in refused]}); "
                             f"a resume would not be bitwise continuous -- pass allow_engine_change / "
                             f"allow_platform_change to accept that")
        session["tolerated_mismatch"] = mism
        gen_noise.bit_generator.state = extra["rng_noise"]
        gen_fr.bit_generator.state = extra["rng_fr"]
        resumed_from = list(extra.get("resumed_from", [])) + [int(st["ist"][S_STEP])]
        sessions = list(extra.get("sessions", []))
        n_ckpt = int(extra.get("n_checkpoints_written", 0))
        wall_prev = float(extra.get("wall_s", 0.0))
        session["start_step"] = int(st["ist"][S_STEP])
    else:
        if replay is not None:
            q0 = np.asarray(replay["q0"], dtype=np.float64).reshape(N, 2, 3)
        else:
            q0 = initial_conditions(gen_ic, N, a, L, float(c["r0_bond"]))
        st = _new_state(c, q0, save_steps.size, trace_steps.size, n_opps_total)
        _init_labels(st["q"], fp, st["wst"])
    if replay is not None:
        rp_noise = np.ascontiguousarray(replay["noise"], dtype=np.float64).reshape(-1, N, 2, 3)
        if rp_noise.shape[0] != n_steps:
            raise ValueError(f"replay noise has {rp_noise.shape[0]} steps, the run makes {n_steps} moves")
        rp_u = np.ascontiguousarray(replay.get("fr_u", np.zeros((0, N))), dtype=np.float64).reshape(-1, N)
        rp_perm = np.ascontiguousarray(replay.get("fr_perm", np.zeros(0)), dtype=np.int64).ravel()
        rp_src = np.ascontiguousarray(replay.get("fr_src", np.zeros(0)), dtype=np.int64).ravel()
    else:
        rp_u = np.zeros((0, N))
        rp_perm = np.zeros(0, dtype=np.int64)
        rp_src = np.zeros(0, dtype=np.int64)
    empty_frb = np.zeros((0, N + 2 * cap))

    def close_session():
        session.update(peak_rss_mb=_peak_rss_mb(reset_ok)[0], wall_s=time.perf_counter() - t0,
                       end_step=int(st["ist"][S_STEP]), end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return sessions + [dict(session)]

    def extra_json(status):
        return dict(fingerprint=fpr, signature=sig, rng_noise=gen_noise.bit_generator.state,
                    rng_fr=gen_fr.bit_generator.state, resumed_from=resumed_from, n_checkpoints_written=n_ckpt,
                    wall_s=wall_prev + (time.perf_counter() - t0), status=status, method=method,
                    sessions=close_session())

    t0 = time.perf_counter()
    t_last_ckpt = time.perf_counter()
    chunks = 0
    ist = st["ist"]
    while ist[S_STEP] <= n_steps:
        step0 = int(ist[S_STEP])
        n_eval = min(chunk_steps, n_steps + 1 - step0)
        last = step0 + n_eval - 1
        n_moves = n_eval if last < n_steps else n_eval - 1
        if replay is not None:
            noise = rp_noise[step0:step0 + n_moves]
            if noise.shape[0] != n_moves:
                raise ValueError("replay noise shorter than n_steps")
        else:
            noise = gen_noise.standard_normal((n_moves, N, 2, 3))
        if method == "fr" and fr_on and replay is None:
            n_opp = _n_opps_in(step0 + 1, step0 + n_moves, fr_start, fr_every)
            frb = gen_fr.random((n_opp, N + 2 * cap))
        else:
            n_opp = 0
            frb = empty_frb
        imove, iopp = _run_chunk(n_eval, fp, ip, Ox, Oy, Oz, Kk, save_steps, trace_steps, noise, frb, rp_u, rp_perm,
                                 rp_src, st["q"], st["anc"], st["ancw"], st["wst"], st["acc"], ist, st["fst"],
                                 st["sv_acc"], st["sv_reg"], st["sv_i"], st["sv_f"], st["sv_evh"], st["traces"],
                                 st["traces_anc"], st["evh"], st["ev_counts"], st["ev_log"])
        if int(ist[S_STEP]) != step0 + n_eval or imove != n_moves or iopp != n_opp:
            raise RuntimeError(f"chunk accounting error: step {ist[S_STEP]} vs {step0 + n_eval}, "
                               f"moves {imove}/{n_moves}, opps {iopp}/{n_opp}")
        chunks += 1
        done = ist[S_STEP] > n_steps
        if checkpoint_path is not None and (done or time.perf_counter() - t_last_ckpt >= float(checkpoint_every_s)):
            n_ckpt += 1
            _write_checkpoint(checkpoint_path, st, extra_json("complete_state" if done else "partial"))
            t_last_ckpt = time.perf_counter()
        if _stop_after_chunks is not None and chunks >= int(_stop_after_chunks) and not done:
            raise InterruptedRun(f"stopped after {chunks} chunks at step {int(ist[S_STEP])}")
    wall = wall_prev + (time.perf_counter() - t0)
    if replay is not None and st["ist"][S_BADSRC] != 0:
        raise RuntimeError("replayed source with non-negative score")
    all_sessions = close_session()
    return _build_result(c, method, st, save_steps, trace_steps, trace_mult, rng_info, fr_on, beta, h, a, L,
                         wall, n_ckpt, resumed_from, replay is not None, sessions=all_sessions, signature=sig,
                         peak_rss_method=_peak_rss_mb(reset_ok)[1])


def _build_result(c, method, st, save_steps, trace_steps, trace_mult, rng_info, fr_on, beta, h, a, L, wall, n_ckpt,
                  resumed_from, is_replay, sessions=None, signature=None, peak_rss_method=None):
    N = c["N"]
    n_steps = c["n_steps"]
    ist = st["ist"]
    sv_acc, sv_i, sv_f = st["sv_acc"], st["sv_i"], st["sv_f"]
    S = save_steps.size
    if int(ist[S_SP]) != S:
        raise RuntimeError(f"only {int(ist[S_SP])} of {S} saves written")
    nfe = int(ist[S_NFE])
    if nfe != N * (n_steps + 1):
        raise RuntimeError(f"force-evaluation count {nfe} != N (n_steps + 1) = {N * (n_steps + 1)}")
    is_fr = method == "fr"
    knob_steps = dict(warmup=int(c["warmup_steps"]), burn_in=int(c["burn_in_steps"]),
                      fr_start=int(c["fr_start_steps"]), fr_every=int(c["fr_every"]),
                      genealogy_window=int(c["genealogy_window_steps"]), n_steps=int(n_steps))
    meta = dict(
        engine=ENGINE, status="complete", system="lta", method=method,
        torch_method=("fr_uniform" if is_fr else "abf"), abf_estimator="histogram",
        T_K=c["T_K"], beta=beta, kT=1.0 / beta, N=N, seed=c["seed"], h=h, n_steps=int(n_steps),
        B=int(N * n_steps), t_total=float(n_steps * h),
        knobs={k: dict(steps=v, t=v * h) for k, v in knob_steps.items()},
        fr_rate=float(c["fr_rate"]), score_clip=float(c["score_clip"]), kde_bandwidth=float(c["kde_bandwidth"]),
        max_event_fraction=float(c["max_event_fraction"]), cap=int(c["cap"]), cap_min=int(c["cap_min"]),
        cap_rule=("finite-N extension cap = max(cap_min, floor(0.02 N))" if int(c["cap_min"]) >= 1
                  else "historical cap = int(0.02 N) (FR off below N = 50)"),
        cap_historical=int(float(c["max_event_fraction"]) * N), fr_active=bool(fr_on),
        fr_score_degenerate=bool(is_fr and N == 2),
        fr_small_n_note=("N = 2: the uniform-target score is zero by symmetry (each walker sees K(0) + K(d)) "
                         "apart from within-bin interpolation (|S| ~ 5e-3); FR-INACTIVE cell, not evidence about FR"
                         if is_fr and N == 2 else
                         "N = 3-4: small scores (walkers mostly farther apart than the KDE bandwidth); read "
                         "total_replacement_events before interpreting the FR contrast" if is_fr and N <= 4 else None),
        abf_force_clip=float(c["abf_force_clip"]), deposit_clip=float(c["deposit_clip"]),
        deposit_clip_rule=("8 x abf_force_clip (core_lta)" if float(c["deposit_clip"]) == 8.0 * float(c["abf_force_clip"])
                           else "EXPLICIT OVERRIDE (deposit_clip_override; core_lta uses 8 x abf_force_clip)"),
        n_grid=int(c["n_grid"]), dphi=TWO_PI / int(c["n_grid"]), window_half=float(c["window_half"]),
        cage_min=float(c["cage_min"]), a_pseudo=a, box=L,
        traces=dict(n_trace=int(c["n_trace"]), K=int(trace_steps.size),
                    dense_every=dict(steps=max(1, int(round(c["trace_dense_every_tu"] / h))), t=c["trace_dense_every_tu"]),
                    dense_until_t=c["trace_dense_until_tu"],
                    sparse_every_t=(None if trace_mult is None else c["trace_sparse_every_tu"] * trace_mult),
                    sparse_multiplier=trace_mult, override=trace_mult is None),
        rng=dict(rng_info, noise_layout="standard_normal((n_moves, N, 2, 3)) per chunk, sequential",
                 fr_layout=f"random((n_opportunities, N + 2 cap)) per chunk; [0,N) firing, [N,N+cap) cap subset, "
                           f"[N+cap,N+2cap) sources", replay=is_replay),
        conventions=dict(
            save="state at the start of the saved step (after k moves and the FR at k); accumulators include "
                 "that step's deposit",
            force_evals="N per evaluated step; steps 0..n_steps are evaluated -> N (n_steps + 1)",
            budget_fraction="save_u = save_step / n_steps = b / B with B = N n_steps",
            events="true events: window-plane crossings = changes of floor(x_COM / a); translocation = arrival "
                   "in the cage region (|z| > cage_min) of a different x-cell than the lineage's last cage; "
                   "labels updated at every evaluated step and at FR opportunities before the copies; clones "
                   "inherit labels",
            genealogy=f"windowed labels reset after the save of every step multiple of "
                      f"{int(c['genealogy_window_steps'])}; ESS = (sum c)^2 / sum c^2 / N",
            torch_equivalent="n_transitions, n_cage_crossings_lineage, birth/death hist, score std as core_lta"),
        diagnostics=bool(c["diagnostics"]), accumulate_u=bool(c["accumulate_u"]),
        record_events=bool(c["record_events"]),
        omitted_keys=([] if c["diagnostics"] else list(DIAG_ONLY_KEYS)),
        lj_sum="vectorised reduction (fastmath reassoc on the sums only); bitwise reproducible per numba compile "
               "target (meta.compile_target), which is part of the checkpoint signature",
        compile_target=compile_target(), host_cpu=_host_cpu(),
        engine_sha256=_PROVENANCE["engine_sha256"], git_commit=_PROVENANCE["git_commit"],
        git_dirty=_PROVENANCE["git_dirty"], git_status=_PROVENANCE["git_status"],
        provenance_note="engine_sha256 / git_commit / git_dirty are taken when the module is imported by the process "
                        "that completed the run; sessions_json lists every session's",
        run_signature=signature, numba=numba.__version__, numpy=np.__version__,
        python=platform.python_version(), n_saves=int(S),
        peak_rss_method=(peak_rss_method or "ru_maxrss (process lifetime; fallback)"),
    )
    if sessions:
        peak = max(float(x["peak_rss_mb"]) for x in sessions if x.get("peak_rss_mb") is not None)
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    res = dict(
        save_step=save_steps.copy(), save_t=save_steps * h, save_u=save_steps / float(n_steps),
        M_all=sv_acc[:, 0].copy(), C_all=sv_acc[:, 1].copy(), M_prod=sv_acc[:, 2].copy(),
        C_prod=sv_acc[:, 3].copy(), hist_inst=sv_acc[:, 4].copy(), region_frac=st["sv_reg"].copy(),
        events_window_crossings=sv_i[:, V_WCROSS].copy(), events_translocations=sv_i[:, V_TRANSLOC].copy(),
        lineage_ever_window=sv_i[:, V_EVWIN].copy(), lineage_ever_opposite=sv_i[:, V_EVOPP].copy(),
        first_window_step=np.int64(ist[S_FIRSTWIN]), first_opposite_step=np.int64(ist[S_FIRSTOPP]),
        gen_n_unique=sv_i[:, V_NU].copy(), gen_ess=sv_f[:, U_ESS].copy(), gen_max_frac=sv_f[:, U_MAXF].copy(),
        gen_n_unique_win=sv_i[:, V_NUW].copy(), gen_ess_win=sv_f[:, U_ESSW].copy(),
        gen_max_frac_win=sv_f[:, U_MAXFW].copy(),
        cum_deaths=sv_i[:, V_DEATHS].copy(), cum_opps=sv_i[:, V_OPPS].copy(),
        cum_opps_with_event=sv_i[:, V_OPPSEV].copy(), events_per_opp_hist=st["sv_evh"].copy(),
        n_transitions=sv_i[:, V_TRANS].copy(), n_cage_crossings_lineage=sv_i[:, V_CAGEX].copy(),
        traces_step=np.asarray(trace_steps, dtype=np.int64).copy(), traces=st["traces"].copy(),
        traces_anc=st["traces_anc"].copy(),
        q_final=st["q"].copy(), anc_final=st["anc"].copy(),
        M_all_final=st["acc"][A_FS].copy(), C_all_final=st["acc"][A_CS].copy(),
        M_prod_final=st["acc"][A_FSP].copy(), C_prod_final=st["acc"][A_CSP].copy(),
        birth_hist=st["acc"][A_BH].copy(), death_hist=st["acc"][A_DH].copy(),
        total_replacement_events=np.int64(ist[S_REPL]), n_fr_opps=np.int64(ist[S_NOPP]),
        fr_score_std=np.float64(st["fst"][G_SSTD] / max(int(ist[S_NSCORE]), 1)),
        fr_score_absmax=np.float64(st["fst"][G_SMAX]),
        n_force_evals=np.int64(ist[S_NFE]), wall_s=np.float64(wall),
        peak_rss_mb=np.float64(peak),
        n_checkpoints_written=np.int64(n_ckpt), resumed_from=np.asarray(resumed_from, dtype=np.int64),
        sessions_json=json.dumps(sessions or [], sort_keys=True),
        meta_json=json.dumps(meta, sort_keys=True), cfg_json=json.dumps(c, sort_keys=True, default=str),
    )
    if not c["diagnostics"]:
        for k in DIAG_ONLY_KEYS:
            del res[k]
    if c["accumulate_u"]:
        res["u_of_z"] = st["acc"][A_USP] / np.maximum(st["acc"][A_CSP], 1.0)
        res["U_prod_final"] = st["acc"][A_USP].copy()
    if is_replay:
        res["replay_consumed"] = np.array([ist[S_RPU], ist[S_RPP], ist[S_RPS]], dtype=np.int64)
    if c["record_events"]:
        n_ev = int(ist[S_EVLOG])
        res["event_counts"] = st["ev_counts"][:int(ist[S_NOPP])].copy()
        res["event_log"] = st["ev_log"][:n_ev].copy()
    return res


# keys that legitimately differ between two executions of one job (cost / execution record)
COST_KEYS = ("wall_s", "peak_rss_mb", "n_checkpoints_written", "resumed_from", "sessions_json")


def save_result(path, res):
    """Atomic compressed .npz (no pickle): arrays as is, strings as 0-d unicode arrays."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp{os.getpid()}")
    arrays = {}
    for k, v in res.items():
        arrays[k] = np.array(v) if isinstance(v, str) else np.asarray(v)
        if arrays[k].dtype == object:
            raise TypeError(f"result key {k!r} is not a plain array")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_result(path):
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            v = z[k]
            if v.ndim == 0:
                v = v.item()
            out[k] = v
    return out
