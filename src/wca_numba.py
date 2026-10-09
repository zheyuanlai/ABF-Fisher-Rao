"""CPU (numba) port of the WCA-dimer sampler: histogram (P0) ABF +/- uniform Fisher-Rao.

Why this exists
---------------
The WCA replica ladder holds the force-evaluation budget B = N_replicas x n_steps = 1024 x 120000
fixed and moves along N in {1024, 256, 64, 16, 4, 1}.  The accepted torch engine
(``wca_abffr_core.run_sampler_gpu`` with ``SimConfig(abf_estimator='histogram', abf_n_bins=160)``)
costs ~0.25 ms per replica-step on CPU torch, i.e. ~8.5 h per arm-run, and is dispatch-bound on
the GPU at small N.  This module runs the same dynamics as one compiled scalar loop, all arms of
one (N, seed) job on SHARED Langevin noise.

What is ported (op for op, file:line of the torch original in src/wca_abffr_core.py)
------------------------------------------------------------------------------------
* ``WCADimerEngine.force`` (scatter path, :300-338): WCA pairs within r0 = 2^(1/6) sigma (dimer
  pair excluded), r clamped at min_r*sigma, the double-well dimer bond; pair forces accumulated
  in torch's CPU order (all pairs (k, j>k) ascending, then all pairs (i<k, k) ascending, then
  the dimer) so the sum is the same sequence of float64 additions.  A vectorised screen only
  FINDS the active pairs; the inactive pairs contribute exact zeros in torch.
  ``torch.linalg.norm`` of a 2-vector is sqrt(fma(y, y, x*x)) under the AVX2/AVX512 kernels this
  host dispatches to and sqrt(x*x + y*y) under ATEN_CPU_CAPABILITY=default (measured; ``fma_norm``).
  The one op not reproduced bit for bit is torch's vectorised ``pow(inv, 6)`` (Sleef, <= 1 ulp
  from the libm pow used here); with ATEN_CPU_CAPABILITY=default torch uses libm pow too and the
  ABF path is BITWISE equal for thousands of steps (test V1a).
* ``clip_forces`` (:229), ``reaction_coordinate`` (:350), ``local_mean_force`` (:361) -- note that
  torch evaluates ``python_float / tensor`` as ``tensor.reciprocal() * python_float``
  (``Tensor.__rtruediv__``), which rounds differently from a division and is reproduced as such;
  ``add_abf_force`` (:369), ``add_reaction_coordinate_wall_force`` (:377), ``wrap_positions``
  (:221, Python-style remainder = numba ``%``, measured identical).
* ``HistogramABFEstimator`` (:542-626): bin = clamp(floor((z - z_min)/delta + 1e-9)), every
  replica deposits (C_j += 1, M_j += f_local) BEFORE any bias is read; bias = M_j/(C_j + 0) of
  the replica's OWN bin, edge-extrapolated; two instances as in ``run_sampler_gpu`` (:1498-1499):
  the bias estimator sees every step, the production (reported) one only steps >= burn-in.
* the main loop of ``run_sampler_gpu`` (:1664-1811): deposit at state ``step``, bias with the
  linear warm-up ramp and the +-abf_force_clip clamp, the three force clips, the soft RC wall,
  Euler-Maruyama move ``q + dt*T + sqrt(2 dt/beta) * xi``, wrap; uniform-target FR on the
  POST-move positions at ``next_step >= fr_start`` and ``(next_step - fr_start) % fr_every == 0``
  (:1812-1929).
* the FR score (``fr_score_torch`` :1029, ``kde_1d_torch`` :460, ``normalize_density_on_grid_torch``
  :454, ``interp_uniform_grid_edge`` :491, ``fr_target_uniform_torch`` :1010,
  ``recentered_clipped_score_torch`` :981, applied twice exactly as torch does: once in
  ``fr_score_torch``, once in ``fixed_population_birth_death_torch``) and the birth-death law
  (``fixed_population_birth_death_torch`` :1086-1147): death candidates u_i < 1 - exp(-r S_i+ dt_fr),
  a uniformly random ``cap``-subset if there are more than ``cap``, every death replaced by an
  iid copy of a replica drawn proportionally to S_i- (negative scores; never a death), the copy
  taking the whole configuration and the ancestor label.
* genealogy: run-long ancestor ESS / max-lineage share / unique ancestors at saves
  (``ancestor_ess`` :1367, ``ancestor_max_fraction`` :1378) and the windowed ESS minimum
  (``_track_window`` :1567: updated after every event with >= 1 death, reset at FR
  opportunities with next_step % max(ess_window, 1) == 0); barrier crossings (:1669-1680, a clone
  inherits only the source's side label, :1914-1921).

Deliberate differences (each statistically inert; tests/test_wca_numba.py)
--------------------------------------------------------------------------
* float64 state and accumulators throughout.  The accepted engine is float32 on CUDA (positions,
  forces, and the bin sums M_j, C_j).  Rounding is negligible EXCEPT one real artefact of the
  accepted runs: a float32 count saturates at 2^24, and the boundary bin 159 (z >= 1.19125, which
  also collects the wall tail z > 1.2) gets ~2e7 deposits per run, so in all 32 accepted
  histogram runs its C is exactly 16777216 at the end.  The artefact is NOT neutral between the
  arms (measured on the 32 accepted files): the hist_abf arm reaches 2^24 at the 105000-107500
  save (the last ~10-12 % of the run) and its bin-159 mean force drifts from 37.2-37.3 to
  44.2-45.6; the hist_fr_uniform arm saturates only at the 115000-117500 save (the last ~2-4 %)
  and drifts from 37.3-37.4 to 39.2-40.4.  Accepted bias_clip_fraction: ABF 4.7-5.2 %, FR
  3.4-3.7 %; in float64 (seed 3105, this port) ABF 2.9 % and FR 3.2 % -- the ordering reverses.
  Bin 159 is outside the eval window, but it feeds the late dynamics beyond z ~ 1.19 unequally
  in the two arms, so the accepted FR-vs-ABF contrast carries this artefact; its effect on the
  eval-window metrics is not measured here.  ``run_ladder_point(..., accum_float32=True)``
  emulates the float32 accumulator STORAGE (the default is float64, which the ladder must use:
  at fixed budget B every ladder point deposits ~2e7 into bin 159, so in float32 every point
  would saturate).  The emulation reproduces the saturation (C = 2^24 in both arms) and puts the
  eval-window metrics (l2_f, I_F, e_F'), replacements, ESS and round trips inside the accepted
  16-seed ranges, but NOT the boundary bin's drift: on fresh seeds 3105/3110/3113 the final
  bin-159 mean force is ABF 46.1-46.4 (accepted max 45.6) and FR 40.7-41.0 (accepted max
  40.4), probably because the accepted engine's float32 positions and forces (not emulated)
  shift when the bin saturates.  Judge bin-159 diagnostics (its mean force, and to a lesser
  degree bias_clip_fraction) as like-for-like only to ~1-2 force units.  The CPU
  float64 torch engine driven by the same noise is reproduced BITWISE (V1a, under
  ATEN_CPU_CAPABILITY=default) and to round-off over 300 steps under the default AVX512 kernels
  (V1b); beyond that the WCA collisions amplify 1 ulp to O(1) within ~1000 steps (V1c), which is
  a property of the dynamics, not of the port.
* Random numbers: Langevin noise from numpy's PCG64 ``Generator.standard_normal`` (one draw per
  coordinate per step, order [replica][particle][x, y], drawn ONCE per step and shared by every
  arm slot by slot); FR randomness (death uniforms, cap subset, copy sources) from SEPARATE PCG64
  streams, ONE PER FR ARM: ``SeedSequence(noise_seed).spawn(1 + n_fr)``, child 0 = the noise,
  child 1 + k = the k-th FR arm in ``arms`` order (with one FR arm this is exactly the earlier
  two-stream layout).  So an arm's trajectory never depends on what the other arms do (V1d, and
  V1f for two FR arms); an FR arm moved to a different FR position draws from a different child,
  i.e. an independent realisation of the same law.  In the torch engine the FR draws (rand(R),
  randperm, multinomial) come from the SAME global stream as the Langevin noise, so the torch abf
  and fr_uniform arms share noise only until the first FR opportunity (step 20000); here they
  share it for the whole run (stronger pairing, identical per-arm law).
* The torch cap subset is ``randperm(n)[:cap]`` and the sources one ``multinomial`` call; here a
  partial Fisher-Yates and iid inverse-CDF draws (same law: V3 checks both against the exact law
  with statistics shown to reject four plausible wrong laws, incl. at N = 1024 with the cap binding).
* The event cap is ``max(cap_min, floor(max_event_fraction * N))``; ``cap_min = 0`` is the
  accepted law exactly (torch returns before any draw when the cap is < 1, so at N < 50 the
  accepted fr_uniform arm IS the ABF arm), ``cap_min = 1`` is the ladder's floor.
* Means inside the recentred clip use a sequential sum (torch: pairwise); the KDE sum skips
  kernel terms below exp(-72) of the peak (< 1e-30 absolute); both are round-off level (V2).
* The FR grid is taken from ``torch.linspace`` itself (``fr_grid``): torch's CPU kernel fills it
  with a vectorised arange whose last bits depend on the SIMD width (the accepted CUDA float32
  grid differs at the 1e-8 level anyway).  The ABF path never reads the grid.
* The uniform target has no EMA and the WCA FR rate has no ramp (constant ``fr_rate`` from
  ``fr_start_steps``), exactly as in the torch engine; nothing to port there.
* The error read-out is NOT reimplemented: the engine returns the RAW accumulators (M_j, C_j) of
  both estimators at every save; ``score_accumulators`` feeds them into the accepted
  ``HistogramABFEstimator`` + ``final_l2_errors`` / ``timeseries_l2`` / ``histogram_fp_error_np``
  (the same calls as ``wca_phase_jobs.execute_run`` :275-276 and :398-413; V1a, V5).
  ``integrated_l2_f`` is the accepted PHYSICAL-time integral (it scales with n_steps = B/N, so it
  is not comparable across ladder points); ``score_ladder_result`` adds the budget-axis integral
  ``integrated_l2_f_u`` = trapezoid of e_F over u = s/n_steps across the saves and its average
  ``mean_l2_f_u`` (the segment [0, u_first) before the first save is dropped, not extrapolated;
  ``budget_save_grid`` starts at u = 1e-4).  Each result carries the ``cfg`` it was run with and
  the scorer builds its SimConfig from it (``setup_from_cfg``).
* CPU only: every torch import goes through ``_cpu_torch``, which hides CUDA
  (CUDA_VISIBLE_DEVICES='') before torch / wca_abffr_core is first imported and refuses to run if
  wca_abffr_core already picked a non-CPU device.
* Performance: the particle count P and the norm mode are compiled as numba LITERALS (a runtime
  P halves the speed of the vectorised pair screen); 5.6-6.2 us per replica-step per arm on one
  EPYC 9554 core (N = 1 ... 1024, FR on), unchanged with 96 concurrent processes: one 2-arm
  ladder job at B = 1.2288e8 replica-steps takes ~23-25 min.

Conventions
-----------
* ``save_at`` = COMPLETED-STEP counts s in [0, n_steps]: the state after s Langevin steps (and
  the FR event at s, if any), with its own deposit made -- exactly the state the torch engine
  saves at its loop index ``step = s`` (it saves at s = 0, 2500, ..., 120000).  The force is
  evaluated n_steps + 1 times per replica (the torch loop does the same).
* seed label -> engine: the accepted runner maps label -> ``SimConfig.seed`` by identity
  (``wca_phase_jobs.build_sim``), and the engine uses that seed for BOTH the lattice init
  (``lattice_initial_conditions(..., seed=label)``, its own generator) and the noise
  (``torch.manual_seed(label)``).  ``lattice_init`` reproduces the init with the torch CPU
  generator at the engine dtype (float32): bit-identical to the torch engine run on CPU.  The
  accepted runs were on CUDA (Philox), whose init values for the same seed are different draws
  of the same law; the numba noise stream is seeded from (label, N) (``default_noise_seed``).
  ``lattice_init`` imports torch (CPU); workers should set OMP_NUM_THREADS=1 before importing it.
* Outputs per arm and save: the raw (M_j, C_j) of the bias (all-steps) and production
  (post-burn-in) estimators, and ``M_rep``/``C_rep`` = the one the torch engine reports there
  (production iff s >= estimator_burn_in_steps); run-long ancestor ESS / max lineage share /
  unique ancestors (for an ABF arm NaN / NaN / N, the torch engine's convention :1791-1794),
  the windowed ESS now and its running minimum (the torch statistic
  ``min_ancestor_ess_window``), cumulative replacements (deaths = copies), opportunities with an
  event and with the cap binding, region fractions, the z histogram of the walkers on the
  estimator bins, barrier crossings and round trips.  ``budget_save_grid`` (identical to the
  gateway port's) contains no s = 0 save; pass ``save_at`` explicitly to get one.
"""
from __future__ import annotations

import math
import os

import numpy as np
from numba import literally, njit, types
from numba.extending import intrinsic
from llvmlite import ir

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REFERENCE_NPZ = os.path.join(ROOT, "cache", "phase_hp_v3", "wca_ti_b1_h2_w2_n10_a1.5_g160.npz")
EPS = 1.0e-12                 # wca_abffr_core.EPS
KDE_TAIL = 12.0               # KDE terms with |d| > 12 bw (< exp(-72) of the peak) are skipped
CLIP_SAFE = 1.0 - 1e-12       # |F|^2 below CLIP_SAFE * clip^2 => the torch clip scale is exactly 1.0


@intrinsic
def _fma(typingctx, a, b, c):
    """IEEE fused multiply-add (llvm.fma.f64): torch.linalg.norm(v) == sqrt(fma(v1, v1, v0*v0))."""
    sig = types.float64(types.float64, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        d = ir.DoubleType()
        fn = builder.module.declare_intrinsic("llvm.fma", [d], ir.FunctionType(d, [d, d, d]))
        return builder.call(fn, args)
    return sig, codegen


@njit(cache=True)
def _sq2d(x, y, fm):
    """x^2 + y^2 the way torch.linalg.norm accumulates it: fma(y, y, x*x) under the AVX2/AVX512
    kernels this host dispatches to (fm = 1), x*x + y*y under ATEN_CPU_CAPABILITY=default (fm = 0)."""
    if fm == 1:
        return _fma(y, y, x * x)
    return x * x + y * y


@njit(cache=True)
def _norm2d(x, y, fm):
    return math.sqrt(_sq2d(x, y, fm))


@njit(cache=True)
def _clip_vec(fx, fy, clip, fm):
    """clip_forces for one particle: returns the (possibly) rescaled force."""
    n2 = _sq2d(fx, fy, fm)
    if n2 < CLIP_SAFE * clip * clip:
        return fx, fy                       # torch: scale = clamp(clip/norm, max=1) == 1.0 exactly
    nrm = math.sqrt(n2)
    if nrm < EPS:
        nrm = EPS
    sc = (1.0 / nrm) * clip                 # torch: python_float / tensor == tensor.reciprocal() * float
    if sc > 1.0:
        sc = 1.0
    return fx * sc, fy * sc


# ---------------------------------------------------------------------------------------------
# Physical force (WCADimerEngine.force, scatter path) for ONE replica
# ---------------------------------------------------------------------------------------------
@njit(cache=True)
def wca_force(q, P, L, sigma, eps4, cutoff, rmin, h, w, r0, xs, ys, r2b,
              plist_i, plist_j, plist_fx, plist_fy, F, fm):
    """Raw force of ONE replica (``WCADimerEngine.force``, scatter path) into F (P, 2); returns the
    number of active WCA pairs.  q: (P, 2) positions in [0, L]; ``eps4`` = 4.0*epsilon.

    A branch-free (vectorisable) screen over all pairs i < j flags candidates with r^2 <=
    cutoff^2 (1 + 1e-6); each candidate is then recomputed with torch's exact expressions
    (minimum image d - L*round(d/L), r = norm, r >= cutoff -> inactive, which in torch contributes
    an exact 0).  Active pairs come out in torch's lexicographic (i, j) order and are accumulated
    exactly like the two CPU ``scatter_add_`` calls: all (k, j>k) contributions in j order, then all
    -(i<k, k) contributions in i order, then the dimer bond.  The screen only decides which pairs
    are LOOKED AT, never the arithmetic (its 1e-6 margin is ~10 orders above its rounding)."""
    invL = 1.0 / L
    cut2 = cutoff * cutoff * (1.0 + 1e-6)
    for p in range(P):
        xs[p] = q[p, 0]
        ys[p] = q[p, 1]
    npair = 0
    for i in range(P):
        qix = xs[i]
        qiy = ys[i]
        for j in range(i + 1, P):                # screen (vectorised by LLVM)
            dx = qix - xs[j]
            dx = dx - L * np.rint(dx * invL)
            dy = qiy - ys[j]
            dy = dy - L * np.rint(dy * invL)
            r2b[j] = dx * dx + dy * dy
        j0 = 2 if i == 0 else i + 1              # the dimer pair (0, 1) is not a WCA pair
        for j in range(j0, P):
            if r2b[j] <= cut2:
                dx = qix - xs[j]
                dx = dx - L * np.rint(dx / L)    # minimum_image, exact torch expression
                dy = qiy - ys[j]
                dy = dy - L * np.rint(dy / L)
                r = _norm2d(dx, dy, fm)
                if r <= cutoff:
                    rs = r if r > rmin else rmin
                    inv = (1.0 / rs) * sigma            # sigma / r_safe (__rtruediv__)
                    inv6 = math.pow(inv, 6.0)
                    inv12 = inv6 * inv6
                    dvdr = eps4 * (-12.0 * inv12 / rs + 6.0 * inv6 / rs)
                    rsc = rs if rs > EPS else EPS
                    coef = -dvdr / rsc
                    plist_i[npair] = i
                    plist_j[npair] = j
                    plist_fx[npair] = coef * dx
                    plist_fy[npair] = coef * dy
                    npair += 1
    for p in range(P):
        F[p, 0] = 0.0
        F[p, 1] = 0.0
    for k in range(npair):                    # scatter_add_(idx_i, f_pair), pairs in order
        F[plist_i[k], 0] += plist_fx[k]
        F[plist_i[k], 1] += plist_fy[k]
    for k in range(npair):                    # scatter_add_(idx_j, -f_pair), pairs in order
        F[plist_j[k], 0] += -plist_fx[k]
        F[plist_j[k], 1] += -plist_fy[k]
    # dimer bond
    dx = q[0, 0] - q[1, 0]
    dx = dx - L * np.rint(dx / L)
    dy = q[0, 1] - q[1, 1]
    dy = dy - L * np.rint(dy / L)
    r01 = _norm2d(dx, dy, fm)
    if r01 < EPS:
        r01 = EPS
    u = (r01 - r0 - w) / w
    dvdr_dim = -4.0 * h * u * (1.0 - u * u) / w
    c01 = -dvdr_dim / r01
    fx = c01 * dx
    fy = c01 * dy
    F[0, 0] += fx
    F[0, 1] += fy
    F[1, 0] -= fx
    F[1, 1] -= fy
    return npair


@njit(cache=True)
def dimer_geometry(q, L, r0, w, fm):
    """(d01x, d01y, r01, z) of one replica: ``dimer_displacement_and_length`` + ``reaction_coordinate``."""
    dx = q[0, 0] - q[1, 0]
    dx = dx - L * np.rint(dx / L)
    dy = q[0, 1] - q[1, 1]
    dy = dy - L * np.rint(dy / L)
    r01 = _norm2d(dx, dy, fm)
    if r01 < EPS:
        r01 = EPS
    return dx, dy, r01, (r01 - r0) / (2.0 * w)


@njit(cache=True)
def hist_bin(z, z_min, delta, nb):
    """HistogramABFEstimator.bin_index."""
    j = int(math.floor((z - z_min) / delta + 1e-9))
    if j < 0:
        j = 0
    elif j > nb - 1:
        j = nb - 1
    return j


# ---------------------------------------------------------------------------------------------
# Fisher-Rao pieces (uniform target)
# ---------------------------------------------------------------------------------------------
@njit(cache=True)
def kde_density(zs, n, grid, bw, z_min, z_max, p_out):
    """normalize_density_on_grid_torch(kde_1d_torch(grid, zs[:n], bw, z_min, z_max), grid)."""
    G = grid.shape[0]
    norm = bw * math.sqrt(2.0 * math.pi)
    tail = KDE_TAIL * bw
    for k in range(G):
        p_out[k] = 0.0
    g0 = grid[0]
    gN = grid[G - 1]
    dzg = grid[1] - grid[0]
    for img in range(3):
        for i in range(n):
            if img == 0:
                c = zs[i]
            elif img == 1:
                c = 2.0 * z_min - zs[i]
            else:
                c = 2.0 * z_max - zs[i]
            if c + tail < g0 or c - tail > gN:
                continue
            k_lo = int(math.floor((c - tail - g0) / dzg)) - 1
            k_hi = int(math.ceil((c + tail - g0) / dzg)) + 1
            if k_lo < 0:
                k_lo = 0
            if k_hi > G - 1:
                k_hi = G - 1
            for k in range(k_lo, k_hi + 1):
                d = grid[k] - c
                if d > tail or d < -tail:
                    continue
                t = d / bw
                p_out[k] += math.exp(-0.5 * (t * t)) / norm
    den = float(n if n > 1 else 1)
    for k in range(G):
        v = p_out[k] / den
        p_out[k] = v if v > EPS else EPS          # kde clamp, then the normalize clamp (same)
    mass = 0.0
    for k in range(G - 1):
        mass += 0.5 * (p_out[k + 1] + p_out[k]) * (grid[k + 1] - grid[k])
    if mass < EPS:
        mass = EPS
    for k in range(G):
        p_out[k] = p_out[k] / mass


@njit(cache=True)
def uniform_target(grid, q_out):
    """fr_target_uniform_torch: normalize_density_on_grid_torch(ones)."""
    G = grid.shape[0]
    mass = 0.0
    for k in range(G - 1):
        mass += 0.5 * (1.0 + 1.0) * (grid[k + 1] - grid[k])
    if mass < EPS:
        mass = EPS
    for k in range(G):
        q_out[k] = 1.0 / mass


@njit(cache=True)
def interp_edge(prof, grid, z):
    """interp_uniform_grid_edge for one point."""
    G = grid.shape[0]
    dz = grid[1] - grid[0]
    x = (z - grid[0]) / dz
    i0 = int(math.floor(x))
    if i0 < 0:
        i0 = 0
    elif i0 > G - 2:
        i0 = G - 2
    fr = x - i0
    if fr < 0.0:
        fr = 0.0
    elif fr > 1.0:
        fr = 1.0
    v = (1.0 - fr) * prof[i0] + fr * prof[i0 + 1]
    if z < grid[0]:
        v = prof[0]
    if z > grid[G - 1]:
        v = prof[G - 1]
    return v


@njit(cache=True)
def raw_scores(zs, n, grid, p, qd, S_out):
    """fr_score_torch before the clip: S = log p(z) - log q(z) - KL(p || q).  Returns KL."""
    G = grid.shape[0]
    kl = 0.0
    prev = p[0] * (math.log(p[0] if p[0] > EPS else EPS) - math.log(qd[0] if qd[0] > EPS else EPS))
    for k in range(1, G):
        cur = p[k] * (math.log(p[k] if p[k] > EPS else EPS) - math.log(qd[k] if qd[k] > EPS else EPS))
        kl += 0.5 * (cur + prev) * (grid[k] - grid[k - 1])
        prev = cur
    for i in range(n):
        pa = interp_edge(p, grid, zs[i])
        qa = interp_edge(qd, grid, zs[i])
        S_out[i] = math.log(pa if pa > EPS else EPS) - math.log(qa if qa > EPS else EPS) - kl
    return kl


@njit(cache=True)
def recenter_clip(S, n, clip):
    """recentered_clipped_score_torch, in place."""
    m = 0.0
    for i in range(n):
        m += S[i]
    m /= n
    for i in range(n):
        S[i] = S[i] - m
    for _ in range(3):
        m = 0.0
        for i in range(n):
            v = S[i]
            if v > clip:
                v = clip
            elif v < -clip:
                v = -clip
            S[i] = v
            m += v
        m /= n
        for i in range(n):
            S[i] = S[i] - m


@njit(cache=True)
def death_birth_weights(S, n, fr_rate, dt_eff, dprob, bweight):
    """Death probabilities and birth weights of fixed_population_birth_death_torch.
    Returns (death_mass, birth_mass)."""
    dm = 0.0
    bm = 0.0
    for i in range(n):
        s = S[i]
        dw = s if s > 0.0 else 0.0
        bwt = -s if -s > 0.0 else 0.0
        dm += dw
        bm += bwt
        bweight[i] = bwt
        if dw > 0.0:
            dprob[i] = 1.0 - math.exp(-fr_rate * dw * dt_eff)
        else:
            dprob[i] = 0.0
    return dm, bm


@njit(cache=True)
def fr_select(dprob, bweight, n, cap, rng, deaths, sources, cum):
    """The birth-death law on fixed probabilities: death candidates u_i < dprob_i (one uniform per
    replica), a uniformly random ``cap``-subset if more, iid sources ~ bweight.  Writes
    deaths[:k], sources[:k]; returns (k, n_candidates) (k = 0: no event)."""
    nd = 0
    for i in range(n):
        if rng.random() < dprob[i]:
            deaths[nd] = i
            nd += 1
    n_cand = nd
    if nd == 0:
        return 0, 0
    if nd > cap:
        for t in range(cap):                     # uniform cap-subset (partial Fisher-Yates)
            j = t + int(rng.random() * (nd - t))
            if j > nd - 1:
                j = nd - 1
            tmp = deaths[t]
            deaths[t] = deaths[j]
            deaths[j] = tmp
        nd = cap
    tot = 0.0
    last = -1
    for i in range(n):
        tot += bweight[i]
        cum[i] = tot
        if bweight[i] > 0.0:
            last = i
    for t in range(nd):
        u = rng.random() * tot
        lo = 0
        hi = n - 1
        while lo < hi:                           # first index with cum > u
            mid = (lo + hi) // 2
            if cum[mid] > u:
                hi = mid
            else:
                lo = mid + 1
        if not (cum[lo] > u) or bweight[lo] <= 0.0:
            lo = last
        sources[t] = lo
    return nd, n_cand


@njit(cache=True)
def sham_select(N, k, rng, perm, deaths, sources):
    """The matched-sham law for k replacements among N replicas: deaths = k distinct replicas drawn
    uniformly (a full Fisher-Yates permutation, first k), sources iid uniform among the N - k
    survivors (with replacement) -- ``uniform_birth_death_torch`` / ``lta.core_lta._sham_birth_death``."""
    for i in range(N):
        perm[i] = i
    for t in range(N):
        j = t + int(rng.random() * (N - t))
        if j > N - 1:
            j = N - 1
        tmp = perm[t]
        perm[t] = perm[j]
        perm[j] = tmp
    for t in range(k):
        deaths[t] = perm[t]
    for t in range(k):
        u = int(rng.random() * (N - k))
        if u > N - k - 1:
            u = N - k - 1
        sources[t] = perm[k + u]


@njit(cache=True)
def fr_uniform_event(zs, n, grid, kde_bw, z_min, z_max, score_clip, fr_rate, dt_eff, cap, rng,
                     p, qd, S, dprob, bweight, deaths, sources, cum):
    """One uniform-FR opportunity on post-move RC values zs[:n]: score, guards, law.
    Returns (k, n_candidates): deaths[:k] <- copies of sources[:k]."""
    if cap < 1 or fr_rate <= 0.0:
        return 0, 0                              # torch: max_events < 1 -> no event, no RNG
    kde_density(zs, n, grid, kde_bw, z_min, z_max, p)
    uniform_target(grid, qd)
    raw_scores(zs, n, grid, p, qd, S)
    recenter_clip(S, n, score_clip)              # fr_score_torch
    recenter_clip(S, n, score_clip)              # fixed_population_birth_death_torch (again)
    dm, bm = death_birth_weights(S, n, fr_rate, dt_eff, dprob, bweight)
    if dm <= EPS or bm <= EPS:
        return 0, 0
    return fr_select(dprob, bweight, n, cap, rng, deaths, sources, cum)


# ---------------------------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------------------------
@njit(cache=True)
def simulate(N, n_steps, arm_fr, arm_nb, max_bins, physp, simp, intp, grid, q0, save_at,
             rng_noise, rng_frs, arm_slot, ext_noise, use_ext_noise, P, fm, Mb, Cb, Mp, Cp, arm_partner):
    """All arms of one job on shared Langevin noise.

    physp = [L, sigma, 4*epsilon, cutoff, min_r*sigma, h, w, r0, force_clip, beta]
    simp  = [dt, z_min, z_max, mean_force_sample_clip, abf_bias_scale, wall, abf_force_clip,
             kde_bw, fr_rate, score_clip, transition_lo, transition_hi, min_count, dt_eff]
    intp  = [warmup, burn_in, fr_start, fr_every, ess_window, cap, use_clipped_mf]
    P     = physical particle count n_dim^2;  fm = 1: torch.linalg.norm as fma (AVX2/AVX512 host
            kernels), 0: plain (ATEN_CPU_CAPABILITY=default).  Both are compiled as LITERALS (one
            specialisation per value): a runtime P halves the speed of the vectorised pair screen.
    q0: (N, P, 2) initial positions (every arm starts from it).  save_at: sorted unique
    completed-step counts in [0, n_steps].  ext_noise: (n_steps, N*P*2) if use_ext_noise.
    rng_frs: a tuple of FR Generators, one per FR arm (length >= 1); arm a draws its FR randomness
    from rng_frs[arm_slot[a]] (arm_slot = -1 for an ABF arm, never read).
    Mb, Cb, Mp, Cp: (A, max_bins) zero arrays holding the bias / production accumulators; float64,
    or float32 to emulate the accepted CUDA engine's float32 storage (``accum_float32``).
    arm_fr: 0 ABF, 1 uniform FR, 2 MATCHED SHAM (2026-10-09): at every FR opportunity a sham arm
    replaces exactly as many replicas as its partner FR arm ``arm_partner[a]`` (an earlier arm, so
    processed first in the same step) did at that opportunity, with deaths uniform without
    replacement and sources uniform among the survivors (with replacement) -- the law of
    ``wca_abffr_core.uniform_birth_death_torch`` / ``lta.core_lta._sham_birth_death`` -- drawn from
    its own FR stream.  Arms with codes 0/1 are unaffected (bitwise; tests/test_wca_numba.py)."""
    literally(P)
    literally(fm)
    L, sigma, eps4, cutoff, rmin = physp[0], physp[1], physp[2], physp[3], physp[4]
    h, w, r0, fclip, beta = physp[5], physp[6], physp[7], physp[8], physp[9]
    dt, z_min, z_max, mf_clip, bias_scale = simp[0], simp[1], simp[2], simp[3], simp[4]
    wall, abf_clip, kde_bw, fr_rate, score_clip = simp[5], simp[6], simp[7], simp[8], simp[9]
    tr_lo, tr_hi, min_count, dt_eff = simp[10], simp[11], simp[12], simp[13]
    warmup, burn_in, fr_start = intp[0], intp[1], intp[2]
    fr_every, ess_window, cap, use_clipped_mf = intp[3], intp[4], intp[5], intp[6]
    win = ess_window if ess_window > 1 else 1        # torch: _win = max(int(ess_window_steps), 1)
    A = arm_fr.shape[0]
    S_n = save_at.shape[0]
    z_b = 0.5 * (tr_lo + tr_hi)
    noise_scale = math.sqrt(2.0 * dt / beta)
    G = grid.shape[0]

    # ---- state ----
    q = np.empty((A, N, P, 2))
    for a in range(A):
        for i in range(N):
            for p in range(P):
                q[a, i, p, 0] = q0[i, p, 0]
                q[a, i, p, 1] = q0[i, p, 1]
    anc = np.empty((A, N), dtype=np.int64); anc_w = np.empty((A, N), dtype=np.int64)
    cnt_r = np.ones((A, N), dtype=np.int64); cnt_w = np.ones((A, N), dtype=np.int64)
    s2_r = np.full(A, N, dtype=np.int64); s2_w = np.full(A, N, dtype=np.int64)
    for a in range(A):
        for i in range(N):
            anc[a, i] = i
            anc_w[a, i] = i
    min_ess_w = np.full(A, float(N))
    side = np.zeros((A, N), dtype=np.bool_)
    rep_c2s = np.zeros((A, N), dtype=np.int64); rep_s2c = np.zeros((A, N), dtype=np.int64)
    tot_c2s = np.zeros(A, dtype=np.int64); tot_s2c = np.zeros(A, dtype=np.int64)
    tot_repl = np.zeros(A, dtype=np.int64); n_event_opp = np.zeros(A, dtype=np.int64)
    n_cap_bind = np.zeros(A, dtype=np.int64)
    bias_absmax = np.zeros(A); clip_n = np.zeros(A); eval_n = np.zeros(A)

    # ---- outputs ----
    o_Mb = np.zeros((A, S_n, max_bins)); o_Cb = np.zeros((A, S_n, max_bins))
    o_Mp = np.zeros((A, S_n, max_bins)); o_Cp = np.zeros((A, S_n, max_bins))
    o_repl = np.zeros((A, S_n), dtype=np.int64)
    o_ess = np.full((A, S_n), np.nan); o_wmax = np.full((A, S_n), np.nan)
    o_nuniq = np.full((A, S_n), N, dtype=np.int64)      # torch reports N for a non-FR arm
    o_ess_w = np.full((A, S_n), np.nan); o_min_ess_w = np.full((A, S_n), np.nan)
    o_frac = np.zeros((A, S_n, 3)); o_zhist = np.zeros((A, S_n, max_bins))
    o_c2s = np.zeros((A, S_n), dtype=np.int64); o_s2c = np.zeros((A, S_n), dtype=np.int64)
    o_rt = np.zeros((A, S_n), dtype=np.int64)
    o_nev = np.zeros((A, S_n), dtype=np.int64); o_ncap = np.zeros((A, S_n), dtype=np.int64)

    # ---- work ----
    F = np.empty((P, 2)); Fph = np.empty((N, P, 2))
    zc = np.empty(N); jb = np.empty(N, dtype=np.int64)
    d0x = np.empty(N); d0y = np.empty(N); r01a = np.empty(N)
    xs = np.empty(P); ys = np.empty(P); r2b = np.empty(P)
    npmax = P * (P - 1) // 2 + 1
    pl_i = np.empty(npmax, dtype=np.int64); pl_j = np.empty(npmax, dtype=np.int64)
    pl_fx = np.empty(npmax); pl_fy = np.empty(npmax)
    noise = np.empty(N * P * 2)
    zs = np.empty(N); pk = np.empty(G); qk = np.empty(G); Sc = np.empty(N)
    dprob = np.empty(N); bwt = np.empty(N); cum = np.empty(N)
    deaths = np.empty(N, dtype=np.int64); sources = np.empty(N, dtype=np.int64)
    k_now = np.zeros(A, dtype=np.int64)            # events of each arm at the current opportunity (sham replay)
    perm = np.empty(N, dtype=np.int64)

    ptr = 0
    for step in range(n_steps + 1):
        last = step == n_steps
        if not last:
            if use_ext_noise:
                for k in range(N * P * 2):
                    noise[k] = ext_noise[step, k]
            else:
                for k in range(N * P * 2):
                    noise[k] = rng_noise.standard_normal()
        ramp = step / (warmup if warmup > 1 else 1)
        if ramp > 1.0:
            ramp = 1.0
        abf_scale = bias_scale * ramp
        next_step = step + 1
        do_fr = (not last) and next_step >= fr_start and (next_step - fr_start) % (fr_every if fr_every > 1 else 1) == 0
        is_save = ptr < S_n and save_at[ptr] == step
        for a in range(A):
            nb = arm_nb[a]
            delta = (z_max - z_min) / nb
            # ---------- pass A: forces, RC, local mean force, deposit ----------
            for i in range(N):
                qi = q[a, i]
                wca_force(qi, P, L, sigma, eps4, cutoff, rmin, h, w, r0, xs, ys, r2b,
                          pl_i, pl_j, pl_fx, pl_fy, F, fm)
                for p in range(P):
                    fx, fy = _clip_vec(F[p, 0], F[p, 1], fclip, fm)
                    Fph[i, p, 0] = fx
                    Fph[i, p, 1] = fy
                dx, dy, r01, z = dimer_geometry(qi, L, r0, w, fm)
                d0x[i] = dx; d0y[i] = dy; r01a[i] = r01; zc[i] = z
                cur = z > z_b
                if step == 0:
                    side[a, i] = cur
                else:
                    if cur and not side[a, i]:
                        tot_c2s[a] += 1
                        rep_c2s[a, i] += 1
                    elif side[a, i] and not cur:
                        tot_s2c[a] += 1
                        rep_s2c[a, i] += 1
                    side[a, i] = cur
                if use_clipped_mf == 1:
                    g0x = Fph[i, 1, 0] - Fph[i, 0, 0]
                    g0y = Fph[i, 1, 1] - Fph[i, 0, 1]
                else:
                    g0x = F[1, 0] - F[0, 0]
                    g0y = F[1, 1] - F[0, 1]
                energetic = ((1.0 / r01) * w) * (dx * g0x + dy * g0y)      # (w / r01) * sum(...)
                entropic = (1.0 / (beta * r01)) * (2.0 * w)              # (2 w) / (beta r01)
                fl = energetic - entropic
                if fl < -mf_clip:
                    fl = -mf_clip
                elif fl > mf_clip:
                    fl = mf_clip
                j = hist_bin(z, z_min, delta, nb)
                jb[i] = j
                Cb[a, j] += 1.0
                Mb[a, j] += fl
                if step >= burn_in:
                    Cp[a, j] += 1.0
                    Mp[a, j] += fl
            # ---------- save (state after `step` completed steps, deposit made) ----------
            if is_save:
                for j in range(nb):
                    o_Mb[a, ptr, j] = Mb[a, j]; o_Cb[a, ptr, j] = Cb[a, j]
                    o_Mp[a, ptr, j] = Mp[a, j]; o_Cp[a, ptr, j] = Cp[a, j]
                for i in range(N):
                    z = zc[i]
                    if z < tr_lo:
                        o_frac[a, ptr, 0] += 1.0 / N
                    elif z <= tr_hi:
                        o_frac[a, ptr, 1] += 1.0 / N
                    else:
                        o_frac[a, ptr, 2] += 1.0 / N
                    o_zhist[a, ptr, jb[i]] += 1.0
                o_repl[a, ptr] = tot_repl[a]
                o_c2s[a, ptr] = tot_c2s[a]; o_s2c[a, ptr] = tot_s2c[a]
                rt = 0
                for i in range(N):
                    rt += rep_c2s[a, i] if rep_c2s[a, i] < rep_s2c[a, i] else rep_s2c[a, i]
                o_rt[a, ptr] = rt
                o_nev[a, ptr] = n_event_opp[a]; o_ncap[a, ptr] = n_cap_bind[a]
                if arm_fr[a] >= 1:
                    mx = 0
                    nu = 0
                    for i in range(N):
                        c = cnt_r[a, i]
                        if c > mx:
                            mx = c
                        if c > 0:
                            nu += 1
                    o_ess[a, ptr] = (float(N) * float(N)) / float(s2_r[a])
                    o_wmax[a, ptr] = mx / float(N)
                    o_nuniq[a, ptr] = nu
                    o_ess_w[a, ptr] = (float(N) * float(N)) / float(s2_w[a])
                    o_min_ess_w[a, ptr] = min_ess_w[a]
            # ---------- pass B: bias, transport, move ----------
            for i in range(N):
                j = jb[i]
                den = Cb[a, j] + min_count
                raw = Mb[a, j] / den if den > 0.0 else 0.0
                ab = abs(raw)
                if ab > bias_absmax[a]:
                    bias_absmax[a] = ab
                if ab > abf_clip:
                    clip_n[a] += 1.0
                eval_n[a] += 1.0
                if last:
                    continue
                rc = raw
                if rc < -abf_clip:
                    rc = -abf_clip
                elif rc > abf_clip:
                    rc = abf_clip
                mfz = abf_scale * rc
                den4 = 2.0 * w * r01a[i]
                gx = d0x[i] / den4
                gy = d0y[i] / den4
                z = zc[i]
                up = z - z_max
                if up < 0.0:
                    up = 0.0
                lo = z - z_min
                if lo > 0.0:
                    lo = 0.0
                mw = -(wall * (up + lo))
                base = i * P * 2
                for p in range(P):
                    tx = Fph[i, p, 0]
                    ty = Fph[i, p, 1]
                    if p == 0:
                        tx = tx + mfz * gx
                        ty = ty + mfz * gy
                    elif p == 1:
                        tx = tx + mfz * (-gx)
                        ty = ty + mfz * (-gy)
                    tx, ty = _clip_vec(tx, ty, fclip, fm)
                    if p == 0:
                        tx = tx + mw * gx
                        ty = ty + mw * gy
                    elif p == 1:
                        tx = tx + mw * (-gx)
                        ty = ty + mw * (-gy)
                    tx, ty = _clip_vec(tx, ty, fclip, fm)
                    nx = (q[a, i, p, 0] + dt * tx) + noise_scale * noise[base + 2 * p]
                    ny = (q[a, i, p, 1] + dt * ty) + noise_scale * noise[base + 2 * p + 1]
                    # wrap_positions = torch.remainder (Python %): identity on (0, L)
                    q[a, i, p, 0] = nx if 0.0 < nx < L else nx % L
                    q[a, i, p, 1] = ny if 0.0 < ny < L else ny % L
            # ---------- uniform FR (or its matched sham) on the post-move positions ----------
            if do_fr and arm_fr[a] >= 1:
                if arm_fr[a] == 1:
                    for i in range(N):
                        zs[i] = dimer_geometry(q[a, i], L, r0, w, fm)[3]
                    k, n_cand = fr_uniform_event(zs, N, grid, kde_bw, z_min, z_max, score_clip, fr_rate, dt_eff, cap,
                                                 rng_frs[arm_slot[a]], pk, qk, Sc, dprob, bwt, deaths, sources, cum)
                else:
                    k = k_now[arm_partner[a]]
                    if k > N - 1:
                        k = N - 1
                    n_cand = k
                    if k > 0:
                        sham_select(N, k, rng_frs[arm_slot[a]], perm, deaths, sources)
                k_now[a] = k
                if k > 0:
                    n_event_opp[a] += 1
                    for t in range(k):
                        d = deaths[t]
                        s = sources[t]
                        for p in range(P):
                            q[a, d, p, 0] = q[a, s, p, 0]
                            q[a, d, p, 1] = q[a, s, p, 1]
                        side[a, d] = side[a, s]
                        # run-long genealogy (counts and sum of squares kept exactly)
                        o = anc[a, d]; nw = anc[a, s]
                        s2_r[a] += 1 - 2 * cnt_r[a, o]
                        cnt_r[a, o] -= 1
                        s2_r[a] += 2 * cnt_r[a, nw] + 1
                        cnt_r[a, nw] += 1
                        anc[a, d] = nw
                        o = anc_w[a, d]; nw = anc_w[a, s]
                        s2_w[a] += 1 - 2 * cnt_w[a, o]
                        cnt_w[a, o] -= 1
                        s2_w[a] += 2 * cnt_w[a, nw] + 1
                        cnt_w[a, nw] += 1
                        anc_w[a, d] = nw
                    tot_repl[a] += k
                    e = (float(N) * float(N)) / float(s2_w[a])
                    if e < min_ess_w[a]:
                        min_ess_w[a] = e
                    if n_cand > k:                   # the cap bound (candidates truncated to cap)
                        n_cap_bind[a] += 1
                if next_step % win == 0:
                    for i in range(N):
                        anc_w[a, i] = i
                        cnt_w[a, i] = 1
                    s2_w[a] = N
        if is_save:
            ptr += 1
    return (o_Mb, o_Cb, o_Mp, o_Cp, o_repl, o_ess, o_wmax, o_nuniq, o_ess_w, o_min_ess_w, o_frac, o_zhist,
            o_c2s, o_s2c, o_rt, o_nev, o_ncap, bias_absmax, clip_n, eval_n, min_ess_w, q, anc, anc_w)


# ---------------------------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------------------------
def _cpu_torch():
    """Import torch and wca_abffr_core on the CPU without ever touching a GPU; returns (torch, core).

    wca_abffr_core picks its device at import (``choose_device`` -> ``torch.cuda.is_available()``),
    so CUDA is hidden (CUDA_VISIBLE_DEVICES='') BEFORE that first import.  If torch has already
    initialised CUDA, or wca_abffr_core was already imported with a non-CPU device, this refuses
    (RuntimeError) instead of running.  Every torch import in this module goes through here."""
    import sys
    src = os.path.join(ROOT, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    if "wca_abffr_core" not in sys.modules and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        t = sys.modules.get("torch")
        if t is not None and t.cuda.is_initialized():
            raise RuntimeError("wca_numba is CPU-only, but torch has already initialised CUDA in this process")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import wca_abffr_core as core
    if core.DEVICE.type != "cpu":
        raise RuntimeError(f"wca_numba is CPU-only, but wca_abffr_core.DEVICE is {core.DEVICE}; "
                           "run with CUDA_VISIBLE_DEVICES=''")
    return torch, core


def accepted_setup(n_bins=160, method="fr_uniform", seed=3100):
    """(SimConfig, DimerWCAParams) of the accepted histogram confirmation run, built through the
    accepted runner's own path (scripts/run_histogram_abf_wca.py load_frozen + make_spec ->
    wca_phase_jobs.build_sim / build_params).  Imports torch (CPU)."""
    import importlib.util
    _cpu_torch()
    import wca_phase_jobs as jobs
    spec_ = importlib.util.spec_from_file_location("_run_histogram_abf_wca", os.path.join(ROOT, "scripts", "run_histogram_abf_wca.py"))
    runner = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(runner)
    c, base, fr = runner.load_frozen()
    sp = runner.make_spec("confirmation", f"hist_{method}", method, seed, c, fr, "histogram", n_bins)
    return jobs.build_sim(sp, base), jobs.build_params(sp)


def cfg_from_setup(sim, params):
    """Engine knobs (every time knob in STEPS, as in the accepted config)."""
    return dict(
        n_dim=int(params.n_dim), a=float(params.a), sigma=float(params.sigma), epsilon=float(params.epsilon),
        h=float(params.h), w=float(params.w), beta=float(params.beta), min_r=float(params.min_r),
        force_clip=float(params.force_clip),
        dt=float(sim.dt), z_min=float(sim.z_min), z_max=float(sim.z_max), n_grid=int(sim.n_grid),
        n_bins=int(sim.abf_n_bins), kde_bandwidth=float(sim.kde_bandwidth),
        mean_force_sample_clip=float(sim.mean_force_sample_clip),
        use_clipped_force_for_mean_force=bool(sim.use_clipped_force_for_mean_force),
        abf_bias_scale=float(sim.abf_bias_scale), abf_edge_extrapolate=bool(sim.abf_edge_extrapolate),
        boundary_wall_strength=float(sim.boundary_wall_strength), abf_force_clip=float(sim.abf_force_clip),
        abf_warmup_steps=int(sim.abf_warmup_steps), estimator_burn_in_steps=int(sim.estimator_burn_in_steps),
        fr_rate=float(sim.fr_rate), score_clip=float(sim.score_clip), fr_start_steps=int(sim.fr_start_steps),
        fr_every=int(sim.fr_every), max_event_fraction=float(sim.max_event_fraction),
        ess_window_steps=int(sim.ess_window_steps), transition_lo=float(sim.transition_lo),
        transition_hi=float(sim.transition_hi), eval_z_lo=float(sim.eval_z_lo), eval_z_hi=float(sim.eval_z_hi),
        hist_min_count=0.0)


#: The accepted values (tests assert ``cfg_from_setup(*accepted_setup()) == ACCEPTED_CFG``).
ACCEPTED_CFG = dict(
    n_dim=10, a=1.5, sigma=1.0, epsilon=1.0, h=2.0, w=2.0, beta=1.0, min_r=0.65, force_clip=250.0,
    dt=0.002, z_min=-0.2, z_max=1.2, n_grid=160, n_bins=160, kde_bandwidth=0.07,
    mean_force_sample_clip=500.0, use_clipped_force_for_mean_force=True, abf_bias_scale=1.0,
    abf_edge_extrapolate=True, boundary_wall_strength=80.0, abf_force_clip=40.0,
    abf_warmup_steps=10000, estimator_burn_in_steps=10000, fr_rate=0.1, score_clip=2.0,
    fr_start_steps=20000, fr_every=5, max_event_fraction=0.02, ess_window_steps=4000,
    transition_lo=0.25, transition_hi=0.75, eval_z_lo=-0.1, eval_z_hi=1.1, hist_min_count=0.0)

#: cfg keys that live in DimerWCAParams (every other key is the SimConfig field of the same name,
#: except n_bins = abf_n_bins and hist_min_count, which the torch histogram estimator fixes at 0).
PHYS_KEYS = ("n_dim", "a", "sigma", "epsilon", "h", "w", "beta", "min_r", "force_clip")
#: what the accepted TI reference depends on (physics and z grid): the scorer only defaults to it
#: when a job's cfg matches the accepted values on these keys.
REFERENCE_KEYS = PHYS_KEYS + ("z_min", "z_max", "n_grid")
#: the SimConfig fields the read-out (score_accumulators) reads.
READOUT_SIM_FIELDS = ("dt", "z_min", "z_max", "n_grid", "abf_n_bins", "abf_estimator", "abf_edge_extrapolate",
                      "eval_z_lo", "eval_z_hi", "transition_lo", "transition_hi")


def setup_from_cfg(cfg):
    """(SimConfig, DimerWCAParams) of an engine cfg: the accepted setup (``accepted_setup``) with
    every cfg knob substituted, checked by the round trip ``cfg_from_setup(sim, params) == cfg``."""
    import dataclasses
    cfg = dict(cfg)
    assert set(cfg) == set(ACCEPTED_CFG), sorted(set(cfg) ^ set(ACCEPTED_CFG))
    assert float(cfg["hist_min_count"]) == 0.0, "the torch histogram estimator has min_count = 0"
    sim, params = accepted_setup(n_bins=int(cfg["n_bins"]))
    sim = dataclasses.replace(sim, **{("abf_n_bins" if k == "n_bins" else k): v for k, v in cfg.items()
                                      if k not in PHYS_KEYS and k != "hist_min_count"})
    params = dataclasses.replace(params, **{k: cfg[k] for k in PHYS_KEYS})
    assert cfg_from_setup(sim, params) == cfg, "cfg does not round-trip through SimConfig / DimerWCAParams"
    return sim, params


def fr_grid(cfg=ACCEPTED_CFG):
    """The FR/KDE z-grid exactly as the torch engine builds it on this CPU,
    ``torch.linspace(z_min, z_max, n_grid)`` in float64 (its CPU kernel fills with a vectorised
    arange, so the last bits depend on the SIMD width and are taken from torch, not recomputed).
    The ABF path never reads it (bins are computed from z_min and delta)."""
    torch, _ = _cpu_torch()
    return torch.linspace(float(cfg["z_min"]), float(cfg["z_max"]), int(cfg["n_grid"]), dtype=torch.float64).numpy().copy()


def _r0(cfg):
    return 2.0 ** (1.0 / 6.0) * cfg["sigma"]


def event_cap(N, cfg, cap_min):
    """max(cap_min, floor(max_event_fraction * N)); cap_min = 0 is the accepted torch law
    (``int(max_event_fraction * R)``, no event at all when it is < 1)."""
    return max(int(cap_min), int(math.floor(cfg["max_event_fraction"] * N)))


def lattice_init(seed, N, cfg=ACCEPTED_CFG, dtype="float32"):
    """The torch engine's initial condition for engine seed ``seed`` (``lattice_initial_conditions``
    on the CPU generator at the engine dtype, float32 by default), as float64 (N, P, 2)."""
    torch, core = _cpu_torch()
    params = core.DimerWCAParams(n_dim=int(cfg["n_dim"]), a=float(cfg["a"]), sigma=float(cfg["sigma"]),
                                 epsilon=float(cfg["epsilon"]), h=float(cfg["h"]), w=float(cfg["w"]),
                                 beta=float(cfg["beta"]))
    dt = dict(float32=torch.float32, float64=torch.float64)[dtype]
    q = core.lattice_initial_conditions(params, int(N), torch.device("cpu"), dt, seed=int(seed))
    return q.numpy().astype(np.float64)


def default_noise_seed(seed, N):
    return int(1_000_003 * int(seed) + 7919 * int(N)) % (2 ** 31 - 1)


def make_rngs(noise_seed, n_fr=1):
    """PCG64 streams from one integer: ``SeedSequence(noise_seed).spawn(1 + n_fr)``, child 0 = the
    Langevin noise, child 1 + k = the FR selection of the k-th FR arm.  Returns (rng_noise, tuple of
    n_fr FR generators); n_fr = 1 is the earlier (noise, fr) pair exactly."""
    ss = np.random.SeedSequence(int(noise_seed)).spawn(1 + int(n_fr))
    gens = [np.random.Generator(np.random.PCG64(s)) for s in ss]
    return gens[0], tuple(gens[1:])


def budget_save_grid(n_steps, n_lin=200, n_log=24, u_min=1e-4):
    """Completed-step counts at the shared budget fractions u = b/B: the linear grid k/n_lin
    plus log-spaced fractions in [u_min, 1/n_lin).  Returns (save_at, u).  (Identical to
    gateway_numba.budget_save_grid.)"""
    u_lin = np.arange(1, n_lin + 1) / n_lin
    u_log = np.logspace(np.log10(u_min), np.log10(1.0 / n_lin), n_log, endpoint=False)
    u = np.unique(np.concatenate([u_log, u_lin]))
    steps = np.unique(np.clip(np.round(u * n_steps).astype(np.int64), 1, n_steps))
    return steps, steps / float(n_steps)


def _pack(cfg, N, cap):
    L = cfg["n_dim"] * cfg["a"]
    r0 = _r0(cfg)
    physp = np.array([L, cfg["sigma"], 4.0 * cfg["epsilon"], r0, cfg["min_r"] * cfg["sigma"], cfg["h"], cfg["w"],
                      r0, cfg["force_clip"], cfg["beta"]], dtype=np.float64)
    dt_eff = cfg["dt"] * max(int(cfg["fr_every"]), 1)
    simp = np.array([cfg["dt"], cfg["z_min"], cfg["z_max"], cfg["mean_force_sample_clip"], cfg["abf_bias_scale"],
                     cfg["boundary_wall_strength"], cfg["abf_force_clip"], cfg["kde_bandwidth"], cfg["fr_rate"],
                     cfg["score_clip"], cfg["transition_lo"], cfg["transition_hi"], cfg["hist_min_count"], dt_eff],
                    dtype=np.float64)
    intp = np.array([cfg["abf_warmup_steps"], cfg["estimator_burn_in_steps"],
                     cfg["fr_start_steps"], cfg["fr_every"], cfg["ess_window_steps"], cap,
                     1 if cfg["use_clipped_force_for_mean_force"] else 0], dtype=np.int64)
    assert cfg["abf_edge_extrapolate"], "only the accepted edge-extrapolated read of the bias is ported"
    return physp, simp, intp


def run_ladder_point(seed, N, n_steps, arms, cfg=ACCEPTED_CFG, save_at=None, cap_min=1, noise_seed=None,
                     q0=None, ext_noise=None, fma_norm=True, accum_float32=False):
    """One (seed, N) job.  ``arms``: list of (name, use_fr) or (name, use_fr, n_bins) (default
    n_bins = cfg['n_bins']); all arms start from the torch engine's lattice init for ``seed`` and
    share the Langevin noise; each FR arm has its own FR stream (``make_rngs``).  ``save_at``:
    completed-step counts in [0, n_steps] (default ``budget_save_grid(n_steps)``).  Returns a dict
    of arrays (raw accumulators of BOTH estimators at every save; ``M_rep``/``C_rep`` = the
    estimator the torch engine reports there) plus the ``cfg`` it ran with (``save_result`` /
    ``load_result`` store it; ``score_ladder_result`` scores it).

    ``accum_float32=True`` stores the four histogram accumulators in float32, emulating the
    accepted CUDA engine (``HistogramABFEstimator`` allocates C and M in the grid dtype, float32):
    a bin's count then saturates at 2^24 = 16777216 (C + 1 rounds back) while M keeps growing.
    In the accepted WCA cell the boundary bin z >= 1.19125 (it also collects the whole wall tail
    z > 1.2) receives ~2e7 deposits per 1.2288e8-replica-step run, so in all 32 accepted
    histogram runs its count is exactly 2^24 at the end: hist_abf saturates over the last
    ~10-12 % of the run (bin-159 mean force 37.2 -> 44.2-45.6), hist_fr_uniform only over the
    last ~2-4 % (37.3 -> 39.2-40.4), so the artefact is not neutral between the arms.  The
    emulation reproduces the saturation and the eval-window metrics, but its bin-159 mean force
    runs ~1-2 units above the accepted range on fresh seeds (module docstring); use it only for
    like-for-like validation against the accepted runs.  Default False: the intended
    (unsaturated) algorithm, which the ladder must use (in float32 every ladder point would
    saturate, since every point deposits ~2e7 into bin 159 at fixed budget)."""
    _cpu_torch()                                 # hide CUDA before anything below imports torch
    N, n_steps = int(N), int(n_steps)
    cfg = dict(cfg)
    assert set(cfg) == set(ACCEPTED_CFG), sorted(set(cfg) ^ set(ACCEPTED_CFG))
    if save_at is None:
        save_at, _ = budget_save_grid(n_steps)
    save_at = np.asarray(save_at, dtype=np.int64)
    assert np.all(np.diff(save_at) > 0) and save_at[0] >= 0 and save_at[-1] <= n_steps, "save_at: sorted, unique, in [0, n_steps]"
    names = [a[0] for a in arms]
    assert len(set(names)) == len(names), "arm names must be unique"
    arm_fr = np.array([2 if isinstance(a[1], str) else (1 if a[1] else 0) for a in arms], dtype=np.int64)
    arm_nb = np.array([int(a[2]) if len(a) > 2 else int(cfg["n_bins"]) for a in arms], dtype=np.int64)
    arm_partner = np.full(len(arms), -1, dtype=np.int64)
    for i, a in enumerate(arms):
        if isinstance(a[1], str):                # ("sham", "sham:<partner arm name>"): matched sham
            assert a[1].startswith("sham:"), a
            pi = names.index(a[1][5:])
            assert pi < i and arm_fr[pi] == 1, "a sham's partner must be an EARLIER uniform-FR arm"
            assert arm_nb[pi] == arm_nb[i]
            arm_partner[i] = pi
    n_fr = int((arm_fr == 1).sum())
    n_sham = int((arm_fr == 2).sum())
    arm_slot = np.full(len(arms), -1, dtype=np.int64)
    arm_slot[arm_fr == 1] = np.arange(n_fr)
    arm_slot[arm_fr == 2] = n_fr + np.arange(n_sham)       # sham streams AFTER the FR ones: FR streams unchanged
    if q0 is None:
        q0 = lattice_init(seed, N, cfg)
    q0 = np.ascontiguousarray(q0, dtype=np.float64)
    P = cfg["n_dim"] ** 2
    assert q0.shape == (N, P, 2)
    cap = event_cap(N, cfg, cap_min)
    if noise_seed is None:
        noise_seed = default_noise_seed(seed, N)
    rng_noise, rng_frs = make_rngs(noise_seed, max(n_fr + n_sham, 1))   # >= 1 stream; SeedSequence children are index-stable
    physp, simp, intp = _pack(cfg, N, cap)
    grid = fr_grid(cfg)
    use_ext = ext_noise is not None
    ext = np.ascontiguousarray(ext_noise, dtype=np.float64).reshape(n_steps, N * P * 2) if use_ext else np.zeros((1, 1))
    adt = np.float32 if accum_float32 else np.float64
    acc = [np.zeros((len(arms), int(arm_nb.max())), dtype=adt) for _ in range(4)]
    out = simulate(N, n_steps, arm_fr, arm_nb, int(arm_nb.max()), physp, simp, intp, grid, q0, save_at,
                   rng_noise, rng_frs, arm_slot, ext, use_ext, int(P), 1 if fma_norm else 0, *acc, arm_partner)
    (Mb, Cb, Mp, Cp, repl, ess, wmax, nuniq, ess_w, min_ess_w_t, frac, zhist, c2s, s2c, rt, nev, ncap,
     bias_absmax, clip_n, eval_n, min_ess_w, q_final, anc, anc_w) = out
    burn = int(cfg["estimator_burn_in_steps"])
    prod = save_at >= burn                       # production estimator has n_updates > 0
    M_rep = np.where(prod[None, :, None], Mp, Mb)
    C_rep = np.where(prod[None, :, None], Cp, Cb)
    return dict(arms=names, arm_fr=arm_fr, arm_nbins=arm_nb, arm_fr_stream=arm_slot, save_at=save_at,
                u=save_at / float(n_steps), times=save_at * float(cfg["dt"]), N=N, n_steps=n_steps, seed=int(seed),
                cap=int(cap), cap_min=int(cap_min), noise_seed=int(noise_seed), accum_float32=bool(accum_float32),
                cfg=cfg,
                M_bias=Mb, C_bias=Cb, M_prod=Mp, C_prod=Cp, M_rep=M_rep, C_rep=C_rep, rep_is_prod=prod,
                repl_cumulative=repl, ancestor_ess=ess, max_ancestor_frac=wmax, n_unique_ancestor=nuniq,
                ancestor_ess_window=ess_w, min_ancestor_ess_window_t=min_ess_w_t,
                min_ancestor_ess_window=np.where(arm_fr >= 1, min_ess_w, np.nan),
                frac_regions=frac, z_hist=zhist, n_c2s=c2s, n_s2c=s2c, n_round_trips=rt,
                n_event_opportunities=nev, n_cap_binding=ncap,
                bias_absmax=bias_absmax, bias_clip_fraction=clip_n / np.maximum(eval_n, 1.0), q_final=q_final,
                ancestors_final=anc, ancestors_window_final=anc_w)


def save_result(res, path, keep_state=True):
    """Save a run_ladder_point result as an .npz (no pickle): arrays as they are, ``cfg`` and
    ``arms`` as JSON.  ``keep_state=False`` drops the final positions / ancestor labels (the bulk
    of the file at large N).  Reload with ``load_result``."""
    import json
    drop = {"cfg", "arms"} | (set() if keep_state else {"q_final", "ancestors_final", "ancestors_window_final"})
    arrays = {k: np.asarray(v) for k, v in res.items() if k not in drop}
    np.savez_compressed(path, cfg_json=json.dumps(res["cfg"]), arms_json=json.dumps(list(res["arms"])), **arrays)


def load_result(path):
    """Inverse of ``save_result``: a dict that ``score_ladder_result`` accepts."""
    import json
    with np.load(path, allow_pickle=False) as d:
        res = {k: d[k] for k in d.files if k not in ("cfg_json", "arms_json")}
        res["cfg"] = json.loads(str(d["cfg_json"]))
        res["arms"] = json.loads(str(d["arms_json"]))
    return res


# ---------------------------------------------------------------------------------------------
# Offline read-out through the ACCEPTED code (no reimplementation)
# ---------------------------------------------------------------------------------------------
def load_reference(sim=None, params=None):
    """The accepted v2 TI reference, loaded by ``core.load_or_compute_ti_reference`` (asserted to
    exist first: that function would silently recompute a missing one)."""
    _, core = _cpu_torch()
    import wca_phase_jobs as jobs
    assert os.path.exists(REFERENCE_NPZ), f"accepted TI reference missing: {REFERENCE_NPZ}"
    if sim is None:
        sim, params = accepted_setup()
    ref = core.load_or_compute_ti_reference(REFERENCE_NPZ, params, sim, jobs.build_ti_config({}, sim), None, verbose=False)
    assert "v2" in str(ref["label"]), ref["label"]
    return ref


def score_accumulators(M_t, C_t, steps, dt=None, n_bins=None, sim=None, reference=None, dtype="float64"):
    """Score saved RAW accumulators (M_t, C_t: (n_saves, n_bins)) of the REPORTED estimator with the
    accepted read-out: the accepted ``HistogramABFEstimator`` (bin_mean_force / pmf_profile /
    mean_force_profile on the torch grid), ``final_l2_errors`` + ``timeseries_l2`` (e_F, aligned on
    the eval window; I_F = trapezoid over the saves' times) and ``histogram_fp_error_np`` (own
    e_F', the primary F' metric of the histogram arms) -- the calls of wca_phase_jobs.execute_run.
    ``sim`` defaults to the ACCEPTED SimConfig; to score a ladder job use ``score_ladder_result``,
    which builds the SimConfig from the job's own cfg."""
    torch, core = _cpu_torch()
    M_t = np.asarray(M_t, dtype=np.float64)
    C_t = np.asarray(C_t, dtype=np.float64)
    if n_bins is None:
        n_bins = M_t.shape[-1]
    if sim is None:
        sim, params = accepted_setup(n_bins=int(n_bins))
        if reference is None:
            reference = load_reference(sim, params)
    if reference is None:
        reference = load_reference(sim)
    assert int(sim.abf_n_bins) == int(n_bins) and sim.abf_estimator == "histogram"
    if dt is None:
        dt = float(sim.dt)
    tdt = dict(float32=torch.float32, float64=torch.float64)[dtype]
    grid_t = torch.linspace(sim.z_min, sim.z_max, sim.n_grid, dtype=tdt)
    est = core.make_abf_estimator(sim, grid_t)
    pmf, mf, mfb = [], [], []
    for k in range(M_t.shape[0]):
        est.M = torch.as_tensor(M_t[k, :n_bins], dtype=tdt)
        est.C = torch.as_tensor(C_t[k, :n_bins], dtype=tdt)
        pmf.append(core.to_numpy(est.pmf_profile()))
        mf.append(core.to_numpy(est.mean_force_profile()))
        mfb.append(core.to_numpy(est.bin_mean_force()))
    steps = np.asarray(steps)
    diag = dict(times=steps * float(dt), pmf=np.asarray(pmf), mean_force=np.asarray(mf))
    fin = core.final_l2_errors(diag, reference, sim)
    ts = core.timeseries_l2(diag, reference, sim)
    mfb = np.asarray(mfb, dtype=np.float64)
    edges = core.to_numpy(est.edges)
    own_t = core.histogram_fp_error_np(mfb, edges, reference["grid"], reference["mean_force"], sim.eval_z_lo, sim.eval_z_hi)
    out = dict(times=ts["times"], l2_f=fin["l2_f"], integrated_l2_f=ts["integrated_l2_f"], l2_f_t=ts["l2_f_t"],
               l2_fp=float(own_t[-1]), l2_fp_t=np.asarray(own_t), l2_fp_nodes=fin["l2_fp"], l2_fp_nodes_t=ts["l2_fp_t"],
               hist_mf_bins_t=mfb)
    for k in ("l2_f_compact", "l2_f_transition", "l2_f_stretched"):
        out[k] = fin[k]
    return out


def score_ladder_result(res, arm, reference=None, sim=None):
    """``score_accumulators`` for arm ``arm`` (name or index) of a ``run_ladder_point`` result, or of
    one re-loaded with ``load_result``, on the SimConfig of the cfg the job RAN with
    (``setup_from_cfg(res['cfg'])`` with the arm's bin count).  A passed ``sim`` must agree with that
    cfg on every field the read-out reads (``READOUT_SIM_FIELDS``).  ``reference`` defaults to the
    accepted TI reference, which is allowed only when the job's physics and z grid are the
    accepted ones (``REFERENCE_KEYS``); otherwise pass it.

    Adds the budget axis: ``u`` = s / n_steps at the saves, ``integrated_l2_f_u`` = trapezoid of
    l2_f_t over u across the saves ([0, u_first) is dropped, not extrapolated) and ``mean_l2_f_u``
    = integrated_l2_f_u / (u_last - u_first).  At fixed budget B these compare across ladder
    points; ``integrated_l2_f`` (the accepted physical-time integral) scales as n_steps = B / N."""
    a = list(res["arms"]).index(arm) if isinstance(arm, str) else int(arm)
    nb = int(np.asarray(res["arm_nbins"])[a])
    cfg = dict(res["cfg"], n_bins=nb)
    sim_c, params_c = setup_from_cfg(cfg)
    if sim is None:
        sim = sim_c
    else:
        bad = {f: (getattr(sim, f), getattr(sim_c, f)) for f in READOUT_SIM_FIELDS if getattr(sim, f) != getattr(sim_c, f)}
        if bad:
            raise ValueError(f"sim disagrees with the job's cfg on read-out fields (passed, job): {bad}")
    if reference is None:
        bad = {k: (cfg[k], ACCEPTED_CFG[k]) for k in REFERENCE_KEYS if cfg[k] != ACCEPTED_CFG[k]}
        if bad:
            raise ValueError(f"the accepted TI reference does not apply to this job (job, accepted): {bad}; pass reference=")
        reference = load_reference(sim_c, params_c)
    save_at = np.asarray(res["save_at"], dtype=np.int64)
    out = score_accumulators(np.asarray(res["M_rep"])[a, :, :nb], np.asarray(res["C_rep"])[a, :, :nb], save_at,
                             n_bins=nb, sim=sim, reference=reference)
    if not np.array_equal(out["times"], np.asarray(res["times"])):
        raise ValueError("save times of the result and of the scorer's SimConfig differ")
    u = save_at / float(res["n_steps"])
    out["u"] = u
    if len(u) > 1:
        out["integrated_l2_f_u"] = float(np.trapezoid(out["l2_f_t"], u))
        out["mean_l2_f_u"] = out["integrated_l2_f_u"] / float(u[-1] - u[0])
    else:
        out["integrated_l2_f_u"] = float("nan")
        out["mean_l2_f_u"] = float(out["l2_f_t"][0])
    return out


def benchmark(Ns=(1, 16, 256, 1024), replica_steps=2_000_000, seed=3100):
    """us per replica-step per arm with the accepted knobs and FR active from the first step
    (fr_start_steps = 0, so the cost includes the FR marginal, score and law every fr_every steps).
    Returns {N: (us_abf_only, us_per_arm_two_arms)}."""
    import time
    cfg = dict(ACCEPTED_CFG, fr_start_steps=0)
    run_ladder_point(seed, 4, 20, [("abf", False), ("fr", True)], cfg)          # compile / load the cache
    out = {}
    for N in Ns:
        n_steps = max(200, replica_steps // N)
        q0 = lattice_init(seed, N, cfg)
        t0 = time.perf_counter()
        run_ladder_point(seed, N, n_steps, [("abf", False)], cfg, q0=q0)
        t1 = time.perf_counter()
        run_ladder_point(seed, N, n_steps, [("abf", False), ("fr", True)], cfg, q0=q0)
        t2 = time.perf_counter()
        out[N] = ((t1 - t0) / (N * n_steps) * 1e6, (t2 - t1) / (2 * N * n_steps) * 1e6)
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    B = 1024 * 120000
    res = benchmark()
    for N, (a, b) in res.items():
        print(f"N={N:5d}: ABF alone {a:6.2f} us/replica-step; ABF+FR job {b:6.2f} us/replica-step/arm"
              f" -> one (N, seed) job at B = {B:.4g}, 2 arms: {2 * B * b * 1e-6 / 60:6.1f} min", flush=True)
