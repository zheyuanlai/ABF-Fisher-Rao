"""Independent validation harness for the entropic-gateway model (equal-budget campaign, 2026-10-10).

docs/equal_budget/GATEWAY_TIMESTEP_VALIDATION.md.  Never imported by the production engines.

Model (src/gateway_numba.py, src/eb_abffr_core.py; checked equal here):
    V(x, y) = H (x^2 - 1)^2 + omega(x)^2 y^2 / 2,   omega(x) = w_out + (w_in - w_out) exp(-x^2 / (2 s^2)),
    x reflected into [XMIN, XMAX] = [-1.8, 1.8], y unbounded, unit mobility, inverse temperature beta.
Exact Gibbs law on the domain:  p(x) ~ exp(-beta F(x)),  F(x) = H (x^2-1)^2 + beta^-1 log omega(x),
    Y | X = x ~ N(0, 1 / (beta omega(x)^2)),  E[d_x V | x] = F'(x) = 4 H x (x^2-1) + beta^-1 omega'(x)/omega(x).
Euler-Maruyama in y at fixed x has stationary variance 1 / (beta omega^2 (1 - omega^2 h / 2)).

Chains (all accumulate, every step after burn-in, on ``nb`` equal x-bins of [XMIN, XMAX]):
    EM      production Euler-Maruyama, optional frozen bias +F'_analytic(x) on the x drift (flat target);
    MALA    EM proposal + Metropolis-Hastings on V - F_analytic (exact for the flat-bias target; proposals
            leaving [XMIN, XMAX] are rejected -- the target is zero there);
    EXACT   i.i.d. draws from the exact Gibbs law (inverse-CDF in x on a fine grid, Gaussian y).
Accumulators per x-bin: count, sum d_xV (the production local mean force), sum y, sum y^2.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

XMIN, XMAX = -1.8, 1.8
EM, MALA = 0, 1


def omega(x, p):
    return p["omega_out"] + (p["omega_in"] - p["omega_out"]) * np.exp(-x * x / (2 * p["s"] ** 2))


def domega(x, p):
    return -(p["omega_in"] - p["omega_out"]) * (x / p["s"] ** 2) * np.exp(-x * x / (2 * p["s"] ** 2))


def V_np(x, y, p):
    return p["H"] * (x * x - 1.0) ** 2 + 0.5 * omega(x, p) ** 2 * y * y


def grad_V_np(x, y, p):
    """Analytic gradient (dV/dx, dV/dy), written from the definition."""
    om, dom = omega(x, p), domega(x, p)
    return 4.0 * p["H"] * x * (x * x - 1.0) + om * dom * y * y, om * om * y


def F_exact(x, p):
    return p["H"] * (x * x - 1.0) ** 2 + np.log(omega(x, p)) / p["beta"]


def Fp_exact(x, p):
    return 4.0 * p["H"] * x * (x * x - 1.0) + domega(x, p) / (omega(x, p) * p["beta"])


def Fp_em(x, p, h):
    """The mean force the EM chain converges to at fixed x (fibre variance inflated)."""
    om = omega(x, p)
    return 4.0 * p["H"] * x * (x * x - 1.0) + domega(x, p) / (om * p["beta"]) / (1.0 - om * om * h / 2.0)


def bin_averages(p, nb, h=None, n_sub=2001):
    """Exact bin-averaged (under p(x)) quantities on nb equal bins: probabilities, F_bin = -kT log P_bin,
    mean force, Var(Y | bin) exact and (if h) the EM prediction, for the UNBIASED Gibbs law and for the
    flat-bias law (weights uniform in x)."""
    edges = np.linspace(XMIN, XMAX, nb + 1)
    out = {k: np.zeros(nb) for k in ("P", "Fp", "var", "var_em", "Fp_flat", "var_flat", "var_em_flat", "Fp_em_flat")}
    for j in range(nb):
        xs = np.linspace(edges[j], edges[j + 1], n_sub)
        w = np.exp(-p["beta"] * (F_exact(xs, p) - F_exact(np.array([0.0]), p)[0]))
        om = omega(xs, p)
        v = 1.0 / (p["beta"] * om ** 2)
        out["P"][j] = np.trapezoid(w, xs)
        out["Fp"][j] = np.trapezoid(w * Fp_exact(xs, p), xs) / out["P"][j]
        out["var"][j] = np.trapezoid(w * v, xs) / out["P"][j]
        out["Fp_flat"][j] = np.mean(Fp_exact(xs, p))
        out["var_flat"][j] = np.mean(v)
        if h is not None:
            ve = v / (1.0 - om ** 2 * h / 2.0)
            out["var_em"][j] = np.trapezoid(w * ve, xs) / out["P"][j]
            out["var_em_flat"][j] = np.mean(ve)
            out["Fp_em_flat"][j] = np.mean(Fp_em(xs, p, h))
    out["F"] = -np.log(out["P"]) / p["beta"]
    out["edges"] = edges
    out["centres"] = 0.5 * (edges[1:] + edges[:-1])
    return out


@njit(cache=True)
def _reflect(q, lo, hi):
    span = hi - lo
    qm = (q - lo) % (2.0 * span)
    if qm > span:
        qm = 2.0 * span - qm
    return qm + lo


@njit(cache=True)
def _forces(x, y, H, oout, oin, s):
    """The production force expressions (gateway_numba.simulate, deposit block)."""
    two_s2 = 2.0 * s * s
    e = math.exp(-x * x / two_s2)
    om = oout + (oin - oout) * e
    dom = -(oin - oout) * (x / (s * s)) * e
    return 4.0 * H * x * (x * x - 1.0) + om * dom * y * y, om * om * y, om, dom


@njit(cache=True)
def _Fexact(x, H, oout, oin, s, beta):
    om = oout + (oin - oout) * math.exp(-x * x / (2.0 * s * s))
    return H * (x * x - 1.0) ** 2 + math.log(om) / beta


@njit(cache=True)
def _Fpexact(x, H, oout, oin, s, beta):
    e = math.exp(-x * x / (2.0 * s * s))
    om = oout + (oin - oout) * e
    dom = -(oin - oout) * (x / (s * s)) * e
    return 4.0 * H * x * (x * x - 1.0) + dom / (om * beta)


@njit(cache=True)
def run_chain(scheme, x0, y0, n_steps, burn, h, beta, H, oout, oin, s, flat_bias, nb, seed, trace_every, n_trace):
    """N independent walkers (x0, y0) for n_steps.  flat_bias=1 adds +F'_exact(x) to the x drift (EM) /
    uses the target exp(-beta (V - F_exact(x))) (MALA).  Returns per-bin count, sum f_x, sum y, sum y^2,
    acceptance (MALA), number of reflections, max |dx| per step, non-finite count, x traces of the first
    n_trace walkers every trace_every steps."""
    np.random.seed(seed)
    N = x0.shape[0]
    X = x0.copy()
    Y = y0.copy()
    C = np.zeros(nb); Mf = np.zeros(nb); Sy = np.zeros(nb); Sy2 = np.zeros(nb)
    amp = math.sqrt(2.0 * h / beta)
    delta = (XMAX - XMIN) / nb
    n_acc = 0.0; n_prop = 0.0; n_refl = 0.0; maxdx = 0.0; nonfin = 0
    nt = n_steps // trace_every + 1
    tr = np.zeros((nt, n_trace))
    kt = 0
    for step in range(n_steps):
        if step % trace_every == 0 and kt < nt:
            for i in range(n_trace):
                tr[kt, i] = X[i]
            kt += 1
        for i in range(N):
            x = X[i]; y = Y[i]
            fx, fy, om, dom = _forces(x, y, H, oout, oin, s)
            if step >= burn:
                j = int(math.floor((x - XMIN) / delta + 1e-9))
                if j < 0:
                    j = 0
                elif j > nb - 1:
                    j = nb - 1
                C[j] += 1.0; Mf[j] += fx; Sy[j] += y; Sy2[j] += y * y
            b = _Fpexact(x, H, oout, oin, s, beta) if flat_bias == 1 else 0.0
            zx = np.random.standard_normal()
            zy = np.random.standard_normal()
            if scheme == EM:
                xn = x + (-fx + b) * h + amp * zx
                if xn < XMIN or xn > XMAX:
                    n_refl += 1.0
                xn = _reflect(xn, XMIN, XMAX)
                yn = y - fy * h + amp * zy
                d = abs(xn - x)
                if d > maxdx:
                    maxdx = d
                if not (math.isfinite(xn) and math.isfinite(yn)):
                    nonfin += 1
                X[i] = xn; Y[i] = yn
            else:
                # MALA on U = V - flat_bias * F_exact
                xn = x + (-fx + b) * h + amp * zx
                yn = y - fy * h + amp * zy
                n_prop += 1.0
                if xn < XMIN or xn > XMAX:
                    continue
                fxn, fyn, omn, domn = _forces(xn, yn, H, oout, oin, s)
                bn = _Fpexact(xn, H, oout, oin, s, beta) if flat_bias == 1 else 0.0
                U = H * (x * x - 1.0) ** 2 + 0.5 * om * om * y * y
                Un = H * (xn * xn - 1.0) ** 2 + 0.5 * omn * omn * yn * yn
                if flat_bias == 1:
                    U -= _Fexact(x, H, oout, oin, s, beta)
                    Un -= _Fexact(xn, H, oout, oin, s, beta)
                gx, gy = fx - b, fy             # grad U at x
                gxn, gyn = fxn - bn, fyn        # grad U at xn
                fw = (xn - x + gx * h) ** 2 + (yn - y + gy * h) ** 2
                bw = (x - xn + gxn * h) ** 2 + (y - yn + gyn * h) ** 2
                la = -beta * (Un - U) - beta * (bw - fw) / (4.0 * h)
                if la >= 0.0 or np.random.random() < math.exp(la):
                    n_acc += 1.0
                    d = abs(xn - x)
                    if d > maxdx:
                        maxdx = d
                    X[i] = xn; Y[i] = yn
    return C, Mf, Sy, Sy2, n_acc, n_prop, n_refl, maxdx, nonfin, tr[:kt], X, Y


def exact_samples(n, p, nb, seed):
    """i.i.d. draws from the exact Gibbs law (unbiased or flat?  -- unbiased), binned like run_chain."""
    g = np.random.default_rng(seed)
    xs = np.linspace(XMIN, XMAX, 400001)
    w = np.exp(-p["beta"] * (F_exact(xs, p) - F_exact(xs, p).min()))
    cdf = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(xs))])
    cdf /= cdf[-1]
    x = np.interp(g.random(n), cdf, xs)
    y = g.standard_normal(n) / (np.sqrt(p["beta"]) * omega(x, p))
    fx, _ = grad_V_np(x, y, p)
    j = np.clip(np.floor((x - XMIN) / ((XMAX - XMIN) / nb) + 1e-9).astype(int), 0, nb - 1)
    C = np.bincount(j, minlength=nb).astype(float)
    return C, np.bincount(j, fx, nb), np.bincount(j, y, nb), np.bincount(j, y * y, nb)


def production_force(x, y, p):
    """Extract the PRODUCTION force from gateway_numba.simulate itself: one step, external zero noise,
    min_count 1e300 (bias exactly 0), dt 1e-7: f = (q0 - q1) / dt (x kept away from the walls)."""
    import gateway_numba as gn
    x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
    N = len(x)
    dt = 1e-7
    kern, r = gn.gaussian_kernel_np(0.1, gn.grid_dx())
    # ess_window must be >= 1: simulate() initialises its ancestor labels only at step % ess_window == 0
    # (with ess_window <= 0 they stay uninitialised and the save block indexes garbage -- latent; production uses 4000)
    out = gn.simulate(1, N, 1, dt, p["beta"], p["H"], p["omega_out"], p["omega_in"], p["s"],
                      np.array([0], np.int64), np.array([180], np.int64), np.array([0], np.int64), 180, 1e300,
                      0.0, 0.0, 1, 3.0, 0, kern, int(r), gn.grid_dx(), x.copy(), y.copy(), np.array([1], np.int64), 1,
                      np.zeros((2, N)), True)
    X1, Y1 = out[8][0], out[9][0]
    return (x - X1) / dt, (y - Y1) / dt
