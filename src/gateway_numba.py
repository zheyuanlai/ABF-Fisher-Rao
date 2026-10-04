"""CPU (numba) port of the entropic-gateway sampler, histogram ABF +/- uniform Fisher-Rao.

Why this exists
---------------
The replica-ladder study (docs/GATEWAY_REPLICA_LADDER.md) holds the force-evaluation budget
B = N_replicas x n_steps fixed and moves along (N, n_steps) = (2048, 1e5) ... (1, 2.048e8).
The torch engine (``gateway_core.simulate_batch``) is dispatch-bound on the GPU (~0.9 ms per
step whatever the batch), so the long-thin end of the ladder (1e7-2e8 sequential steps) would
take days there.  A compiled scalar loop costs ~10-30 ns per walker-step, so every rung of
the ladder costs the same few seconds of one CPU core and all seeds run in parallel.

What is ported (op for op from ``gateway_core.simulate_batch`` with estimator='histogram')
----------------------------------------------------------------------------------------
* the Brownian dynamics on V(x, y) = H (x^2-1)^2 + omega(x)^2 y^2 / 2, reflection of x into
  [XMIN, XMAX] (``reflect_into``), y unbounded;
* the P0 histogram estimator: bins [e_j, e_{j+1}) with the 1e-9 nudge, every walker deposits
  (C_j += 1, M_j += f_x) BEFORE any bias is evaluated, bias = M_j / (C_j + min_count) of the
  walker's OWN bin;
* uniform-target FR every ``fr_every`` steps on the post-move positions: nearest-node binned
  KDE (``binned_density``: reflect-padded Gaussian smoothing at eta, /N, normalised by trapz,
  floored at EPS), score S = log p(x) - log q - KL(p||q) with linear interpolation, clipped at
  +-score_clip; death/clone probabilities 1 - exp(-+ g S dt_fr), the proportional cap on the
  total number of events, uniformly random choice of which candidates survive the cap, and
  the pool law (survivors + clones; a uniform N-subset if the pool is too big, uniform
  survivors with replacement if it is too small) -- ``resample_indices`` / ``fr_resample_indices``;
* the FR rate ramp g(t) = gamma (1 - exp(-step / ramp_steps)) and the windowed ancestor ESS.

Deliberate differences (statistically inert, see tests/test_gateway_numba.py)
---------------------------------------------------------------------------
* Random numbers come from numba's generator, not torch's, so runs are NOT bitwise equal to the
  torch engine; with externally supplied Langevin noise the ABF dynamics ARE (test V1).
* When an FR opportunity fires no event the slots are left in place instead of being shuffled
  (walkers are exchangeable; the torch engine applies a random permutation).
* The error read-out is NOT reimplemented: the engine returns the raw bin accumulators (M_j, C_j)
  at every save and ``scripts/analyze_gateway_replica_ladder.py`` scores them with the accepted
  ``eb_abffr_core.HistogramABFEstimator`` code.

All arms of one call share the Langevin noise slot by slot (the paired design of the torch
engine: arms of one row share initial conditions and noise).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

# Fixed domain/grid -- must equal eb_abffr_core's (asserted in the tests).
XMIN, XMAX = -1.8, 1.8
N_GRID = 181
EPS = 1e-30
X_BASIN = 0.5


def gaussian_kernel_np(bw, dx):
    """``eb_abffr_core.gaussian_kernel``: radius round(4 bw / dx), normalised by sum * dx."""
    r = max(1, int(round(4.0 * bw / dx)))
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t * dx / bw) ** 2)
    return k / (k.sum() * dx), r


def grid_dx():
    xg = np.linspace(XMIN, XMAX, N_GRID)
    return float(xg[1] - xg[0])


def init_left(seed, N, beta, oout, oin, s):
    """``gateway_core.init_conditions`` for init='left' (same numpy stream, 1000 + seed)."""
    rng = np.random.default_rng(1000 + int(seed))
    x = rng.normal(-1.0, 0.05, N)
    span = XMAX - XMIN
    qm = np.remainder(x - XMIN, 2.0 * span)
    x = np.where(qm > span, 2.0 * span - qm, qm) + XMIN
    z = rng.normal(0.0, 1.0, N)
    om = oout + (oin - oout) * np.exp(-x * x / (2.0 * s * s))
    y = z * np.sqrt(1.0 / (beta * om ** 2))
    return x.astype(np.float64), y.astype(np.float64)


@njit(cache=True)
def _reflect(q, lo, hi):
    span = hi - lo
    qm = (q - lo) % (2.0 * span)
    if qm > span:
        qm = 2.0 * span - qm
    return qm + lo


@njit(cache=True)
def kde_density(X, n, kern, r, dx, p_out, hist):
    """``eb_abffr_core.binned_density`` for the first ``n`` entries of X, written into p_out."""
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
    # out[i] = sum_m kern[off + m] * v[refl(i + m - pad)], m = 0..2 pad  (torch reflect pad + conv1d)
    for k in range(G):
        c = hist[k]
        if c == 0.0:
            continue
        # images of source k in the padded index o = i + m - pad
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
    """Score of the uniform target: S = log p(x) - log q - KL(p || q), clipped (gateway_core)."""
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
def _self_weight(i, k, kern, pad, off):
    """Weight a unit mass binned at node k puts on output node i through the reflect-padded smoothing."""
    G = N_GRID
    w = 0.0
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
        d = o - i
        if -pad <= d <= pad:
            w += kern[off + d + pad]
    return w


@njit(cache=True)
def centred_scores(X, n, p, dx, clip, S_out, mode, kern, r, mass):
    """EXPLORATORY score variants (configs/gateway_replica_ladder/exploratory_score_fix.json).
    mode 1: S = log p_hat(x_i) - mean_j log p_hat(x_j)  (empirical centring; uniform q cancels)
    mode 2: the same with the leave-one-out density p_loo(x_i) = (p_hat(x_i) - self_i) n / (n - 1)."""
    G = N_GRID
    pad = r if r < G - 1 else G - 1
    off = r - pad
    tot = 0.0
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
        if mode == 2:
            k = int(np.rint((X[i] - XMIN) / dx))
            if k < 0:
                k = 0
            elif k > G - 1:
                k = G - 1
            sw = ((1.0 - frac) * _self_weight(i0, k, kern, pad, off)
                  + frac * _self_weight(i0 + 1, k, kern, pad, off)) / (n * mass)
            v = (v - sw) * n / (n - 1.0)
        if v < EPS:
            v = EPS
        lv = math.log(v)
        S_out[i] = lv
        tot += lv
    mean = tot / n
    for i in range(n):
        s = S_out[i] - mean
        if s > clip:
            s = clip
        elif s < -clip:
            s = -clip
        S_out[i] = s


@njit(cache=True)
def _round_half_even(v):
    return np.rint(v)


@njit(cache=True)
def fr_resample(S, n, g, dt_fr, cap, sel, die_c, clone_c, pool):
    """One FR opportunity on scores S[:n]: writes the gather index into ``sel`` (new = old[sel]) and
    returns (kd, kc).  Law of ``gateway_core.resample_indices`` / ``eb.fr_resample_indices``."""
    nd = 0
    nc = 0
    for i in range(n):
        u = np.random.random()
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
        kd = int(_round_half_even(cap * nd / den))
        if kd > nd:
            kd = nd
        kc = cap - kd
        if kc > nc:
            kc = nc
    else:
        kd = nd
        kc = nc
    if kd + kc == 0:
        return 0, 0
    # uniformly random kd of the nd death candidates (partial Fisher-Yates), same for clones
    for t in range(kd):
        j = t + int(np.random.random() * (nd - t))
        if j > nd - 1:
            j = nd - 1
        tmp = die_c[t]; die_c[t] = die_c[j]; die_c[j] = tmp
    for t in range(kc):
        j = t + int(np.random.random() * (nc - t))
        if j > nc - 1:
            j = nc - 1
        tmp = clone_c[t]; clone_c[t] = clone_c[j]; clone_c[j] = tmp
    # pool = survivors then extra copies of the cloned walkers
    for i in range(n):
        sel[i] = 0  # reuse as a death flag first
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
        # uniform n-subset of the pool
        for t in range(n):
            j = t + int(np.random.random() * (P - t))
            if j > P - 1:
                j = P - 1
            tmp = pool[t]; pool[t] = pool[j]; pool[j] = tmp
        for t in range(n):
            sel[t] = pool[t]
    else:
        for t in range(P):
            sel[t] = pool[t]
        for t in range(P, n):
            j = int(np.random.random() * n_surv)
            if j > n_surv - 1:
                j = n_surv - 1
            sel[t] = pool[j]
    return kd, kc


@njit(cache=True)
def simulate(noise_seed, N, n_steps, dt, beta, H, oout, oin, s,
             arm_fr, arm_nbins, arm_score, max_bins, min_count,
             gamma, ramp_steps, fr_every, score_clip, cap,
             kern, r_eta, dx, x0, y0, save_at, ess_window,
             ext_noise, use_ext_noise):
    """Run all arms on shared Langevin noise.  ``save_at`` = sorted completed-step counts (>= 1)."""
    A = arm_fr.shape[0]
    n_saves = save_at.shape[0]
    X = np.empty((A, N)); Y = np.empty((A, N))
    for a in range(A):
        for i in range(N):
            X[a, i] = x0[i]; Y[a, i] = y0[i]
    Mh = np.zeros((A, max_bins)); Ch = np.zeros((A, max_bins))
    anc = np.empty((A, N), dtype=np.int64)
    out_M = np.zeros((A, n_saves, max_bins)); out_C = np.zeros((A, n_saves, max_bins))
    out_ess = np.zeros((A, n_saves)); out_wmax = np.zeros((A, n_saves))
    out_P = np.zeros((A, n_saves, 3))
    out_die = np.zeros((A, n_saves)); out_clone = np.zeros((A, n_saves))
    out_maxbias = np.zeros(A)
    tot_die = np.zeros(A); tot_clone = np.zeros(A)
    n_fr = 0

    fx = np.empty(N); fy = np.empty(N); jb = np.empty(N, dtype=np.int64)
    zx = np.empty(N); zy = np.empty(N)
    p = np.zeros(N_GRID); hist = np.zeros(N_GRID); S = np.empty(N)
    sel = np.empty(N, dtype=np.int64); die_c = np.empty(N, dtype=np.int64)
    clone_c = np.empty(N, dtype=np.int64); pool = np.empty(2 * N, dtype=np.int64)
    tmpx = np.empty(N); tmpy = np.empty(N); tmpa = np.empty(N, dtype=np.int64)
    cnt = np.zeros(N)

    np.random.seed(noise_seed)
    amp = math.sqrt(2.0 * dt / beta)
    dt_fr = dt * fr_every
    two_s2 = 2.0 * s * s
    s2 = s * s
    dom_amp = oin - oout
    ptr = 0
    for step in range(n_steps):
        if ess_window > 0 and step % ess_window == 0:
            for a in range(A):
                for i in range(N):
                    anc[a, i] = i
        if use_ext_noise:
            for i in range(N):
                zx[i] = ext_noise[2 * step, i]
            for i in range(N):
                zy[i] = ext_noise[2 * step + 1, i]
        else:
            for i in range(N):
                zx[i] = np.random.standard_normal()
            for i in range(N):
                zy[i] = np.random.standard_normal()
        do_fr = (step % fr_every) == 0
        if ramp_steps > 0:
            g = gamma * (1.0 - math.exp(-step / ramp_steps))
        else:
            g = gamma
        for a in range(A):
            nb = arm_nbins[a]
            delta = (XMAX - XMIN) / nb
            # --- deposit (all walkers before any bias read) ---
            for i in range(N):
                x = X[a, i]; y = Y[a, i]
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
                Ch[a, j] += 1.0
                Mh[a, j] += f
            # --- move ---
            mb = out_maxbias[a]
            for i in range(N):
                j = jb[i]
                den = Ch[a, j] + min_count
                bias = Mh[a, j] / den if den > 0.0 else 0.0
                if abs(bias) > mb:
                    mb = abs(bias)
                X[a, i] = _reflect(X[a, i] + (-fx[i] + bias) * dt + amp * zx[i], XMIN, XMAX)
                Y[a, i] = Y[a, i] - fy[i] * dt + amp * zy[i]
            out_maxbias[a] = mb
            # --- uniform FR on the post-move positions ---
            if do_fr and arm_fr[a] == 1 and N >= 2:
                xa = X[a]
                mass = kde_density(xa, N, kern, r_eta, dx, p, hist)
                if arm_score[a] == 0:
                    uniform_scores(xa, N, p, dx, score_clip, S)
                else:
                    centred_scores(xa, N, p, dx, score_clip, S, arm_score[a], kern, r_eta, mass)
                kd, kc = fr_resample(S, N, g, dt_fr, cap, sel, die_c, clone_c, pool)
                if kd + kc > 0:
                    for i in range(N):
                        tmpx[i] = X[a, sel[i]]; tmpy[i] = Y[a, sel[i]]; tmpa[i] = anc[a, sel[i]]
                    for i in range(N):
                        X[a, i] = tmpx[i]; Y[a, i] = tmpy[i]; anc[a, i] = tmpa[i]
                    tot_die[a] += kd
                    tot_clone[a] += kc
        if do_fr:
            n_fr += 1
        # --- saves, after `step + 1` completed steps ---
        while ptr < n_saves and save_at[ptr] == step + 1:
            for a in range(A):
                nb = arm_nbins[a]
                for j in range(nb):
                    out_M[a, ptr, j] = Mh[a, j]
                    out_C[a, ptr, j] = Ch[a, j]
                for i in range(N):
                    cnt[i] = 0.0
                for i in range(N):
                    cnt[anc[a, i]] += 1.0
                ss = 0.0; mx = 0.0
                for i in range(N):
                    ss += cnt[i] * cnt[i]
                    if cnt[i] > mx:
                        mx = cnt[i]
                out_ess[a, ptr] = (N * N / ss) / N
                out_wmax[a, ptr] = mx / N
                for i in range(N):
                    x = X[a, i]
                    if x < -X_BASIN:
                        out_P[a, ptr, 0] += 1.0 / N
                    elif x > X_BASIN:
                        out_P[a, ptr, 2] += 1.0 / N
                    else:
                        out_P[a, ptr, 1] += 1.0 / N
                out_die[a, ptr] = tot_die[a]
                out_clone[a, ptr] = tot_clone[a]
            ptr += 1
    return (out_M, out_C, out_ess, out_wmax, out_P, out_die, out_clone, out_maxbias, X, Y, n_fr)


def budget_save_grid(n_steps, n_lin=200, n_log=24, u_min=1e-4):
    """Completed-step counts at the shared budget fractions u = b/B: the linear grid k/n_lin
    plus log-spaced fractions in [u_min, 1/n_lin).  Returns (save_at, u)."""
    u_lin = np.arange(1, n_lin + 1) / n_lin
    u_log = np.logspace(np.log10(u_min), np.log10(1.0 / n_lin), n_log, endpoint=False)
    u = np.unique(np.concatenate([u_log, u_lin]))
    steps = np.unique(np.clip(np.round(u * n_steps).astype(np.int64), 1, n_steps))
    return steps, steps / float(n_steps)


def run_ladder_point(seed, N, n_steps, arms, cell, fr, save_n_lin=200, save_n_log=24,
                     ess_window=4000, noise_seed=None):
    """One (seed, N) job: every arm in ``arms`` = list of (name, use_fr, n_bins), shared noise.

    cell: dict(beta, H, omega_out, omega_in, s, dt);  fr: dict(gamma, eta, fr_every, ramp_steps,
    score_clip, max_event_fraction, min_count, cap_min)."""
    dx = grid_dx()
    kern, r = gaussian_kernel_np(fr["eta"], dx)
    x0, y0 = init_left(seed, N, cell["beta"], cell["omega_out"], cell["omega_in"], cell["s"])
    save_at, u = budget_save_grid(n_steps, save_n_lin, save_n_log)
    cap = max(int(fr["cap_min"]), int(math.floor(fr["max_event_fraction"] * N)))
    arm_fr = np.array([1 if a[1] else 0 for a in arms], dtype=np.int64)
    arm_nb = np.array([int(a[2]) for a in arms], dtype=np.int64)
    arm_sc = np.array([int(a[3]) if len(a) > 3 else 0 for a in arms], dtype=np.int64)
    if noise_seed is None:
        noise_seed = int(1_000_003 * int(seed) + 7919 * int(N)) % (2 ** 31 - 1)
    res = simulate(int(noise_seed), int(N), int(n_steps), float(cell["dt"]), float(cell["beta"]), float(cell["H"]),
                   float(cell["omega_out"]), float(cell["omega_in"]), float(cell["s"]),
                   arm_fr, arm_nb, arm_sc, int(arm_nb.max()), float(fr["min_count"]),
                   float(fr["gamma"]), float(fr["ramp_steps"]), int(fr["fr_every"]), float(fr["score_clip"]), int(cap),
                   kern, int(r), float(dx), x0, y0, save_at, int(ess_window),
                   np.zeros((0, N)), False)
    out_M, out_C, ess, wmax, P, die, clone, maxbias, X, Y, n_fr = res
    return dict(save_at=save_at, u=u, M=out_M, C=out_C, ess=ess, wmax=wmax, P=P, die=die, clone=clone,
                maxbias=maxbias, X_final=X, Y_final=Y, n_fr=n_fr, cap=cap, noise_seed=noise_seed)


@njit(cache=True)
def _seed_for_tests(seed):
    """Seed numba's generator (it is separate from numpy's) for the law tests."""
    np.random.seed(seed)
