"""Energetic-metastability toy (2-D Gaussian-quartic multiwell, xi = x): histogram ABF +/- uniform Fisher-Rao.

New engine for the UROPS 2026 confirmation (deliverables/UROPS_2026/metastability/PREREGISTRATION.md).  The model is
the archived "Case I" of report/sections/04_case_metastability.tex, potential copied term for term from
src/abffr/potentials.py (tested against it and against central differences in test_meta_engine.py):

    V(x, y) = 3 e^{-x^2} (e^{-(y-1/3)^2} - e^{-(y-5/3)^2}) - 5 e^{-y^2} (e^{-(x-1)^2} + e^{-(x+1)^2})
              + 0.2 x^4 + 0.2 (y - 1/3)^4,
box x in [-3, 3], y in [-2.5, 3.5], both coordinates specularly reflected (src/abffr/simulation.reflect_1d), as in
configs/two_dim_xi_x_production_gpu.yaml.

Dynamics: overdamped Euler-Maruyama, unit mobility (the archived engine's convention),
    x <- reflect(x + (-dV/dx + b) h + sqrt(2 h / beta) zx),  y <- reflect(y - dV/dy h + sqrt(2 h / beta) zy).
ABF: the textbook P0 histogram estimator of the validated gateway engine (src/gateway_ladder_numba.py): nb equal
bins on [-3, 3]; at every step every walker deposits dV/dx at its PRE-move state into its bin (C_j += 1, M_j += f)
and then moves with the bias b = Gamma_j = M_j / (C_j + min_count) of its own pre-move bin, read after all
deposits (pre-deposit convention: the save at completed-step count k holds the deposits of the states 0..k-1).
No ABF ramp.  Reported free energy = exact integral of the piecewise-constant Gamma.
FR (method 'fr'): every fr_every steps on the post-move positions; binned KDE (nearest node of a 601-node grid on
[-3, 3], reflect-padded Gaussian of width eta, normalised); uniform target q = 1/6; score S = log p(x_i) - log q -
KL(p || q) clipped at +-score_clip; rate r(t) = gamma (1 - exp(-t / ramp_t)); the WCA / ITE / LTA birth-death law:
death candidates u_i < 1 - exp(-r S_i^+ dt_fr) (dt_fr = h fr_every), a uniformly random cap-subset when more than
cap = max(1, floor(max_event_fraction N)), every death replaced by an iid copy (state and ancestor label) of a
walker drawn proportionally to S^- among the S < 0 walkers.
RNG: Langevin normals from numba's internal MT19937, seeded once per run with noise_seed and used for nothing else
(N zx then N zy per step, slot order), so the ABF and FR arms of one seed see identical noise slot by slot and are
BITWISE identical until the first FR event.  FR uniforms from numpy Generator(PCG64(SeedSequence([seed, N, 7]))),
opportunity k owning the fixed block of N + 2 cap doubles.  Initial conditions from
Generator(PCG64(SeedSequence([seed, N, 3]))), "mixed" mode of src/abffr/simulation._init_positions: the first
N/2 walkers x ~ U[-2.4, -0.6], y ~ U[-1.9, -0.1], the others x ~ U[0.6, 2.4], y ~ U[1.1, 2.9].
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
from numba import njit

XMIN, XMAX = -3.0, 3.0
YMIN, YMAX = -2.5, 3.5
YA, YB, YQ = 1.0 / 3.0, 5.0 / 3.0, 1.0 / 3.0
N_KDE = 601
ENGINE_VERSION = "meta_engine/1"


# ------------------------------------------------------------------------------------------------ potential
@njit(cache=True)
def V(x, y):
    t1 = 3.0 * math.exp(-x * x) * (math.exp(-(y - YA) ** 2) - math.exp(-(y - YB) ** 2))
    t2 = -5.0 * math.exp(-y * y) * (math.exp(-(x - 1.0) ** 2) + math.exp(-(x + 1.0) ** 2))
    t3 = 0.2 * x ** 4 + 0.2 * (y - YQ) ** 4
    return t1 + t2 + t3


@njit(cache=True)
def grad(x, y):
    ex = math.exp(-x * x)
    ea = math.exp(-(y - YA) ** 2)
    eb = math.exp(-(y - YB) ** 2)
    ey = math.exp(-y * y)
    em = math.exp(-(x - 1.0) ** 2)
    ep = math.exp(-(x + 1.0) ** 2)
    A = ea - eb
    dx = 3.0 * (-2.0 * x) * ex * A - 5.0 * ey * ((-2.0 * (x - 1.0)) * em + (-2.0 * (x + 1.0)) * ep) + 0.8 * x ** 3
    dy = (3.0 * ex * ((-2.0 * (y - YA)) * ea - (-2.0 * (y - YB)) * eb)
          - 5.0 * (-2.0 * y) * ey * (em + ep) + 0.8 * (y - YQ) ** 3)
    return dx, dy


@njit(cache=True)
def reflect(v, lo, hi):
    """Specular reflection into [lo, hi] (src/abffr/simulation.reflect_1d, folded for any overshoot)."""
    L = hi - lo
    w = (v - lo) % (2.0 * L)
    if w > L:
        w = 2.0 * L - w
    return lo + w


@njit(cache=True)
def bin_of(x, nb, delta):
    j = int(math.floor((x - XMIN) / delta + 1e-9))
    if j < 0:
        j = 0
    elif j > nb - 1:
        j = nb - 1
    return j


@njit(cache=True)
def seed_numba(s):
    np.random.seed(s)


# ------------------------------------------------------------------------------------------------ dynamics
@njit(cache=True)
def advance(x, y, M, C, n_steps, h, beta, nb, min_count, zx, zy, jb):
    """n_steps of ABF-biased EM with the P0 histogram estimator (deposit all, then move all)."""
    N = x.shape[0]
    delta = (XMAX - XMIN) / nb
    sig = math.sqrt(2.0 * h / beta)
    for _ in range(n_steps):
        for i in range(N):
            zx[i] = np.random.standard_normal()
        for i in range(N):
            zy[i] = np.random.standard_normal()
        for i in range(N):
            fx, fy = grad(x[i], y[i])
            j = bin_of(x[i], nb, delta)
            jb[i] = j
            C[j] += 1.0
            M[j] += fx
            zy[i] = -fy * h + sig * zy[i]           # y increment (force at the pre-move state)
            zx[i] = -fx * h + sig * zx[i]
        for i in range(N):
            j = jb[i]
            b = M[j] / (C[j] + min_count)
            x[i] = reflect(x[i] + zx[i] + b * h, XMIN, XMAX)
            y[i] = reflect(y[i] + zy[i], YMIN, YMAX)


@njit(cache=True)
def advance_fixed_bias(x, y, n_steps, h, beta, fb_grid, fb_x0, fb_dx, nbv, Cv, Mv, Yv, Y2v, accumulate, zx, zy):
    """EM with a FIXED bias force b(x) (linear interpolation of fb_grid); optional accumulation of validation
    histograms on nbv bins: counts, sum dV/dx, sum y, sum y^2 at the pre-move state (timestep validation)."""
    N = x.shape[0]
    G = fb_grid.shape[0]
    deltav = (XMAX - XMIN) / nbv
    sig = math.sqrt(2.0 * h / beta)
    for _ in range(n_steps):
        for i in range(N):
            zx[i] = np.random.standard_normal()
        for i in range(N):
            zy[i] = np.random.standard_normal()
        for i in range(N):
            xi, yi = x[i], y[i]
            fx, fy = grad(xi, yi)
            if accumulate:
                j = bin_of(xi, nbv, deltav)
                Cv[j] += 1.0
                Mv[j] += fx
                Yv[j] += yi
                Y2v[j] += yi * yi
            u = (xi - fb_x0) / fb_dx
            k = int(math.floor(u))
            if k < 0:
                k = 0
            elif k > G - 2:
                k = G - 2
            w = u - k
            b = (1.0 - w) * fb_grid[k] + w * fb_grid[k + 1]
            x[i] = reflect(xi + (-fx + b) * h + sig * zx[i], XMIN, XMAX)
            y[i] = reflect(yi - fy * h + sig * zy[i], YMIN, YMAX)


@njit(cache=True)
def mala(x, y, n_steps, h, beta, fb_grid, fb_x0, fb_dx, Fb_grid, nbv, Cv, Mv, Yv, Y2v, accumulate, acc):
    """Exact MALA on pi(x, y) ~ exp(-beta (V - F_b(x))) restricted to the box (proposals outside rejected),
    F_b = the integral of the bias force (Fb_grid, linear interpolation); acc[0] accepted, acc[1] proposed."""
    N = x.shape[0]
    G = fb_grid.shape[0]
    deltav = (XMAX - XMIN) / nbv
    sig = math.sqrt(2.0 * h / beta)
    for _ in range(n_steps):
        for i in range(N):
            xi, yi = x[i], y[i]
            fx, fy = grad(xi, yi)
            if accumulate:
                j = bin_of(xi, nbv, deltav)
                Cv[j] += 1.0
                Mv[j] += fx
                Yv[j] += yi
                Y2v[j] += yi * yi
            u = (xi - fb_x0) / fb_dx
            k = min(max(int(math.floor(u)), 0), G - 2)
            w = u - k
            b = (1.0 - w) * fb_grid[k] + w * fb_grid[k + 1]
            Fb = (1.0 - w) * Fb_grid[k] + w * Fb_grid[k + 1]
            gx, gy = fx - b, fy                          # gradient of U = V - F_b
            xp = xi - gx * h + sig * np.random.standard_normal()
            yp = yi - gy * h + sig * np.random.standard_normal()
            ur = np.random.random()
            acc[1] += 1.0
            if xp < XMIN or xp > XMAX or yp < YMIN or yp > YMAX:
                continue
            fxp, fyp = grad(xp, yp)
            up = (xp - fb_x0) / fb_dx
            kp = min(max(int(math.floor(up)), 0), G - 2)
            wp = up - kp
            bp = (1.0 - wp) * fb_grid[kp] + wp * fb_grid[kp + 1]
            Fbp = (1.0 - wp) * Fb_grid[kp] + wp * Fb_grid[kp + 1]
            gxp, gyp = fxp - bp, fyp
            U0 = V(xi, yi) - Fb
            U1 = V(xp, yp) - Fbp
            # log q(old | new) - log q(new | old)
            dfx = xi - (xp - gxp * h)
            dfy = yi - (yp - gyp * h)
            dbx = xp - (xi - gx * h)
            dby = yp - (yi - gy * h)
            lq = -(dfx * dfx + dfy * dfy - dbx * dbx - dby * dby) * beta / (4.0 * h)
            la = -beta * (U1 - U0) + lq
            if la >= 0.0 or ur < math.exp(la):
                x[i], y[i] = xp, yp
                acc[0] += 1.0


# ------------------------------------------------------------------------------------------------ Fisher-Rao
def kde_matrix(eta, n=N_KDE):
    """Reflect-padded Gaussian smoothing matrix on the n-node grid of [-3, 3] (nearest-node binned KDE)."""
    g = np.linspace(XMIN, XMAX, n)
    d = g[1] - g[0]
    K = np.zeros((n, n))
    for img in (lambda c: c, lambda c: 2 * XMIN - c, lambda c: 2 * XMAX - c):
        c = img(g)
        K += np.exp(-0.5 * ((g[:, None] - c[None, :]) / eta) ** 2)
    K /= eta * math.sqrt(2.0 * math.pi)
    return g, d, K


@njit(cache=True)
def fr_step(x, y, anc, ancw, r, dt_fr, Kmat, gd, clip, cap, u, score_out):
    """One uniform-FR opportunity in place.  u: N + 2 cap uniforms.  Returns the number of deaths."""
    N = x.shape[0]
    n = Kmat.shape[0]
    cnt = np.zeros(n)
    for i in range(N):
        k = int(math.floor((x[i] - XMIN) / gd + 0.5))
        if k < 0:
            k = 0
        elif k > n - 1:
            k = n - 1
        cnt[k] += 1.0
    p = Kmat @ cnt
    for k in range(n):
        p[k] /= N
    # trapezoid normalisation on [-3, 3]
    m = 0.5 * (p[0] + p[n - 1])
    for k in range(1, n - 1):
        m += p[k]
    m *= gd
    q = 1.0 / (XMAX - XMIN)
    kl = 0.0
    for k in range(n):
        p[k] = max(p[k] / m, 1e-300)
    for k in range(n):
        wk = 0.5 if (k == 0 or k == n - 1) else 1.0
        kl += wk * p[k] * math.log(p[k] / q)
    kl *= gd
    for i in range(N):
        t = (x[i] - XMIN) / gd
        k = int(math.floor(t))
        if k < 0:
            k = 0
        elif k > n - 2:
            k = n - 2
        w = t - k
        pi = (1.0 - w) * p[k] + w * p[k + 1]
        s = math.log(pi) - math.log(q) - kl
        if s > clip:
            s = clip
        elif s < -clip:
            s = -clip
        score_out[i] = s
    # death candidates
    cand = np.empty(N, dtype=np.int64)
    nc = 0
    for i in range(N):
        s = score_out[i]
        if s > 0.0 and u[i] < 1.0 - math.exp(-r * s * dt_fr):
            cand[nc] = i
            nc += 1
    if nc == 0:
        return 0
    if nc > cap:                                     # uniform cap-subset (partial Fisher-Yates)
        for t_ in range(cap):
            jj = t_ + int(u[N + t_] * (nc - t_))
            if jj > nc - 1:
                jj = nc - 1
            tmp = cand[t_]
            cand[t_] = cand[jj]
            cand[jj] = tmp
        nc = cap
    # sources ~ S^- among S < 0 walkers (never a death)
    neg = 0.0
    for i in range(N):
        if score_out[i] < 0.0:
            neg += -score_out[i]
    if neg <= 0.0:
        return 0
    xs = x.copy()
    ys = y.copy()
    an = anc.copy()
    aw = ancw.copy()
    for t_ in range(nc):
        target = u[N + cap + t_] * neg
        acc = 0.0
        src = -1
        for i in range(N):
            if score_out[i] < 0.0:
                acc += -score_out[i]
                src = i
                if acc >= target:
                    break
        d = cand[t_]
        x[d], y[d], anc[d], ancw[d] = xs[src], ys[src], an[src], aw[src]
    return nc


def family_stats(labels, N):
    c = np.bincount(labels, minlength=N).astype(float) / N
    return 1.0 / np.sum(c * c) / N, float(c.max())


# ------------------------------------------------------------------------------------------------ driver
DEFAULT = dict(beta=4.0, h=5e-4, N=1000, T=200.0, nb=120, min_count=1.0, save_dt=0.5,
               fr_interval_t=0.01, gamma=0.2, eta=0.1, score_clip=2.0, max_event_fraction=0.02, ramp_t=20.0,
               genealogy_window_t=10.0)


def noise_seed(seed, N):
    return (1_000_003 * int(seed) + 7919 * int(N)) % (2 ** 31 - 1)


def init_mixed(seed, N):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(N), 3])))
    h = N // 2
    x = np.empty(N)
    y = np.empty(N)
    x[:h] = rng.uniform(-2.4, -0.6, h)
    y[:h] = rng.uniform(-1.9, -0.1, h)
    x[h:] = rng.uniform(0.6, 2.4, N - h)
    y[h:] = rng.uniform(1.1, 2.9, N - h)
    return x, y


def run_arm(cfg, method, seed, keep_traj=False):
    """One arm.  Returns a dict of numpy arrays (saves at every save_dt, t = 0 included)."""
    c = dict(DEFAULT, **cfg)
    h, N, nb = float(c["h"]), int(c["N"]), int(c["nb"])
    n_steps = int(round(c["T"] / h))
    fe = int(round(c["fr_interval_t"] / h))
    sv = int(round(c["save_dt"] / h))
    W = int(round(c["genealogy_window_t"] / h))
    assert abs(fe * h - c["fr_interval_t"]) < 1e-12 and sv % fe == 0 and n_steps % sv == 0 and W % fe == 0
    cap = max(1, int(math.floor(c["max_event_fraction"] * N)))
    x, y = init_mixed(seed, N)
    seed_numba(noise_seed(seed, N))
    frg = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), N, 7])))
    _, gd, Kmat = kde_matrix(float(c["eta"]))
    M = np.zeros(nb)
    C = np.zeros(nb)
    zx, zy, jb = np.zeros(N), np.zeros(N), np.zeros(N, np.int64)
    anc = np.arange(N, dtype=np.int64)
    ancw = np.arange(N, dtype=np.int64)
    score = np.zeros(N)
    delta = (XMAX - XMIN) / nb
    n_save = n_steps // sv + 1
    out = dict(save_step=np.zeros(n_save, np.int64), M_all=np.zeros((n_save, nb)), C_all=np.zeros((n_save, nb)),
               hist_inst=np.zeros((n_save, nb), np.int64), deaths_cum=np.zeros(n_save, np.int64),
               ess_run=np.zeros(n_save), maxfam_run=np.zeros(n_save), ess_win=np.zeros(n_save),
               maxfam_win=np.zeros(n_save), frac_left=np.zeros(n_save), frac_gate=np.zeros(n_save),
               frac_right=np.zeros(n_save), frac_upper=np.zeros(n_save))
    deaths = 0
    min_ess_win, max_fam_win, max_ev_frac, n_opp, n_cap_bind = 1.0, 0.0, 0.0, 0, 0
    t0 = time.perf_counter()

    def save(isv, k):
        out["save_step"][isv] = k
        out["M_all"][isv] = M
        out["C_all"][isv] = C
        idx = np.clip(np.floor((x - XMIN) / delta + 1e-9).astype(np.int64), 0, nb - 1)
        out["hist_inst"][isv] = np.bincount(idx, minlength=nb)
        out["deaths_cum"][isv] = deaths
        out["ess_run"][isv], out["maxfam_run"][isv] = family_stats(anc, N)
        out["ess_win"][isv], out["maxfam_win"][isv] = family_stats(ancw, N)
        out["frac_left"][isv] = np.mean(x < -0.5)
        out["frac_gate"][isv] = np.mean(np.abs(x) <= 0.5)
        out["frac_right"][isv] = np.mean(x > 0.5)
        out["frac_upper"][isv] = np.mean((np.abs(x) <= 0.5) & (y > 0.9))

    save(0, 0)
    k = 0
    isv = 1
    while k < n_steps:
        advance(x, y, M, C, fe, h, float(c["beta"]), nb, float(c["min_count"]), zx, zy, jb)
        k += fe
        if method == "fr":
            r = float(c["gamma"]) * (1.0 - math.exp(-(k * h) / float(c["ramp_t"]))) if c["ramp_t"] > 0 else float(c["gamma"])
            u = frg.random(N + 2 * cap)
            nd = fr_step(x, y, anc, ancw, r, h * fe, Kmat, gd, float(c["score_clip"]), cap, u, score)
            deaths += nd
            n_opp += 1
            max_ev_frac = max(max_ev_frac, nd / N)
            n_cap_bind += int(nd == cap)
            e_w, f_w = family_stats(ancw, N)
            min_ess_win = min(min_ess_win, e_w)
            max_fam_win = max(max_fam_win, f_w)
        if k % W == 0:
            ancw[:] = np.arange(N)
        if k % sv == 0:
            save(isv, k)
            isv += 1
    out["save_t"] = out["save_step"] * h
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise FloatingPointError("non-finite state")
    out.update(X_final=x.copy(), Y_final=y.copy(), n_force_evals=np.int64(N * n_steps), wall_s=time.perf_counter() - t0,
               min_ess_win=min_ess_win, max_fam_win=max_fam_win, max_event_fraction_realised=max_ev_frac,
               n_opportunities=n_opp, n_cap_binding=n_cap_bind, cap=cap)
    out["meta_json"] = json.dumps(dict(engine=ENGINE_VERSION, method=method, seed=int(seed), cfg=c, n_steps=n_steps,
                                       fr_every=fe, save_every=sv, window_steps=W, cap=cap,
                                       noise_seed=noise_seed(seed, N), fr_stream=[int(seed), N, 7],
                                       init_stream=[int(seed), N, 3], init="mixed"), sort_keys=True)
    return out


def save_npz(path, res):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **res)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ reference
def reference(beta=4.0, nx=12001, ny=4001):
    """Independent y-quadrature reference (src/abffr/reference.compute_reference) on a fine x grid."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.environ.get("ABFFR_DATA_ROOT", os.path.abspath(os.path.join(here, "..", "..", "..")))
    sys.path.insert(0, os.path.join(root, "src"))
    from abffr.reference import compute_reference
    xg = np.linspace(XMIN, XMAX, nx)
    yg = np.linspace(YMIN, YMAX, ny)
    r = compute_reference(xg, yg, beta)
    return xg, r["F_ref"], r["Fprime_ref"], r["p_ref"]


class Scorer:
    """Histogram read-out vs the quadrature reference with the archived metric conventions (src/abffr/metrics.py):
    e_F = centred domain-RMS on [-2.5, 2.5] (trapezoid over n_eval nodes) of F_hat - F_ref, F_hat = exact integral
    of the piecewise-constant Gamma; e_F' = domain-RMS of Gamma(x) - F'_ref(x) (no centring; it includes the
    piecewise-constant floor = the error of the exact bin-averaged Gamma)."""

    def __init__(self, nb, min_count=1.0, beta=4.0, lo=-2.5, hi=2.5, n_eval=2001, ref=None):
        self.nb, self.min_count = nb, min_count
        xg, F, Fp, p = ref if ref is not None else reference(beta)
        self.xe = np.linspace(lo, hi, n_eval)
        self.F_ref = np.interp(self.xe, xg, F)
        self.Fp_ref = np.interp(self.xe, xg, Fp)
        self.delta = (XMAX - XMIN) / nb
        self.edges = XMIN + self.delta * np.arange(nb + 1)
        self.jx = np.clip(np.floor((self.xe - XMIN) / self.delta + 1e-9).astype(int), 0, nb - 1)
        self.width = hi - lo
        # zero-noise floor of e_F': Gamma = exact bin average of F'_ref (fine-grid mean per bin)
        jf = np.clip(np.floor((xg - XMIN) / self.delta + 1e-9).astype(int), 0, nb - 1)
        fbar = np.bincount(jf, weights=Fp, minlength=nb) / np.maximum(np.bincount(jf, minlength=nb), 1)
        self.fbar = fbar
        self.floor_eFp = self._rms(fbar[self.jx] - self.Fp_ref)
        self.floor_eF = self.errors(fbar[None, :] * (1e12 + min_count), np.full((1, nb), 1e12))[0][0]

    def _rms(self, d):
        return float(np.sqrt(np.trapezoid(d * d, self.xe) / self.width))

    def gamma(self, M, C):
        return np.where(C > 0, M / (C + self.min_count), 0.0)

    def errors(self, M_all, C_all):
        G = self.gamma(np.asarray(M_all, float), np.asarray(C_all, float))
        Fe = np.concatenate([np.zeros((G.shape[0], 1)), np.cumsum(G * self.delta, axis=1)], axis=1)
        eF, eFp = np.zeros(G.shape[0]), np.zeros(G.shape[0])
        for s in range(G.shape[0]):
            Fh = np.interp(self.xe, self.edges, Fe[s])
            d = (Fh - Fh.mean()) - (self.F_ref - self.F_ref.mean())       # metrics.center: arithmetic mean over nodes
            eF[s] = self._rms(d)
            eFp[s] = self._rms(G[s, self.jx] - self.Fp_ref)
        return eF, eFp
