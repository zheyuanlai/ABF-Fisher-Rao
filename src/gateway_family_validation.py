"""Numerical-validation harness for the matched-free-energy gateway family and the transverse-mobility dynamics.

Mechanism campaign, Experiments I and II (docs/mechanism/SCIENTIFIC_PLAN.md section 3; gate thresholds frozen in
configs/equal_budget_v2/gateway_validation.json).  Generalises src/gateway_validation.py (which is NOT modified) to
the four potentials and the y-mobility lambda.  Never imported by the production engines.

Models (beta 16, H 0.5, s 0.1, omega = 1 + 31 exp(-x^2 / (2 s^2)); x reflected into [-1.8, 1.8], y unbounded):
  alpha family  V = H (x^2-1)^2 + ((1-alpha)/beta) log omega + omega^(2 alpha) y^2 / 2,   alpha in {1, 0.5, 0}
                f = d_x V = 4Hx(x^2-1) + ((1-alpha)/beta) omega'/omega + alpha omega^(2 alpha) (omega'/omega) y^2
                Y | x ~ N(0, 1/(beta omega^(2 alpha))),  stiffness k(x) = omega^(2 alpha)
  shifted fibre V = F*(x) + kappa^2 (y - m(x))^2 / 2,  m = (sqrt(2/beta)/kappa) log omega
                f = F*'(x) - kappa^2 (y - m) m'(x),  Y | x ~ N(m(x), 1/(beta kappa^2)),  k = kappa^2
  every model:  F = F* + C,  F*(x) = H (x^2-1)^2 + log(omega)/beta,  E[f | x] = F*'(x)
                Var(f | x) = (2 alpha^2/beta^2)(omega'/omega)^2 (alpha family), (2/beta^2)(omega'/omega)^2 (shift, kappa 1)
  mobility:     x <- reflect(x + (-f + b) h + sqrt(2h/beta) zx),  y <- y - lam (d_y V) h + sqrt(2 lam h/beta) zy
                (drift AND noise of y scaled: the Gibbs law is lambda-independent); b = +F*'(x) for the flat-bias
                target exp(-beta (V - F*)) (uniform x marginal), else 0.
  EM at frozen x: Var_h(Y | x) = Var(Y | x) / (1 - lam k h / 2); for the alpha family
                E_h[f | x] = 4Hx(x^2-1) + ((1-alpha)/beta) r + (alpha/beta) r / (1 - lam k h/2),  r = omega'/omega,
                Var_h(f | x) = Var(f | x) / (1 - lam k h/2)^2; for the shift E_h[f | x] = F*'(x),
                Var_h(f | x) = Var(f | x) / (1 - lam kappa^2 h/2).

Chains (``run_chain``; all accumulate on the PRE-move state of every step after burn-in):
  EM    the production integrator with lam (at alpha = 1, lam = 1 BITWISE src/gateway_validation.run_chain:
        same force expression, same draw order zx, zy per walker, same accumulation order);
  MALA  EM proposal with the y-mobility preconditioner diag(1, lam) + Metropolis-Hastings on U = V - b F*; the
        proposal density exp(-beta [(dx + h g_x)^2 + (dy + lam h g_y)^2 / lam] / (4h)) enters the ratio, so it is
        exact for ANY h and lam (the target does not depend on lam); proposals leaving [XMIN, XMAX] are rejected.
        At alpha = 1, lam = 1 it is bitwise gateway_validation's MALA.
Accumulators per x-bin, on the 1440 fine bins AND on the 180 production bins (bin rule floor((x+1.8)/delta + 1e-9),
clamped): ACC_KEYS = C, Sf, Sf2 (f = d_x V), Sy, Sy2, Sr, Sr2 (r = f - F*'(x): the force residual at the walker's
own x, so its within-bin variance is the POINTWISE conditional variance, free of the within-bin spread of F*'),
Sd, Sd2 (d = y - E[Y | x]: the transverse residual; = y for the alpha family).

Conditional autocorrelation (passive): for the first ``n_acf`` walkers, the normalised deviations
z_y = (y - mu(x))/sigma(x) and the SIGN-INVARIANT z_f = sign(omega'(x)) (f - F*'(x))/sd_f(x) (sd_f from the exact
Var(f|x); z_f = 0 where sd_f = 0, i.e. at x = 0 exactly and everywhere at alpha = 0; in the wells omega'/omega ~ 1e-20,
so z_f there is rounding noise of f - F*' and the analysis reports no z_f ACF in such windows) are sampled at two
strides (level 0 fine, level 1 coarse, in steps after burn-in).  Exactly, z_f = (z_y^2 - 1)/sqrt(2) (alpha family) and
-z_y (shift).  The factor sign(omega') is a deviation from the plain normalised fluctuation (f - F*')/sd_f: that one
equals sign(omega') times the above, and omega' changes sign at x = 0, so its ACF in the x = 0 +- 0.02 window would
mostly measure how often x crosses 0 (lambda-independent), not the conditional relaxation (review 2026-10-10).  The
plain signed version is recoverable from the raw excerpts (they store x).
A ring buffer per walker and level holds the last R = max lag + 1 samples with the window label of x at that
sample; at each sample t and lag l the product z(t - l) z(t) is accumulated into the window of x(t - l)
(conditioning on the ORIGIN).  ACF_KEYS per (window, lag): n, sum z0, z1, z0^2, z1^2, z0 z1 for y then f.  This
is exactly the estimator ``acf_offline`` applies to recorded traces (tested equal); storing the lag sums instead
of full-resolution traces keeps the files small.  The first ``acf_rec`` samples of each level are also recorded
raw (x, z_y, z_f) for the offline cross-check and the figure.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from numba import njit

XMIN, XMAX = -1.8, 1.8
EM, MALA = 0, 1
V_ALPHA, V_SHIFT = 0, 1
NB_FINE, NB_PROD = 1440, 180
ACC_KEYS = ("C", "Sf", "Sf2", "Sy", "Sy2", "Sr", "Sr2", "Sd", "Sd2")
ACF_KEYS = ("n", "y0", "y1", "y00", "y11", "y01", "f0", "f1", "f00", "f11", "f01")
ACF_WINDOWS = (-1.0, -0.21, 0.0)
ACF_HALF_WIDTH = 0.02
ZH_NB, ZH_LIM = 100, 5.0
PHYS = dict(beta=16.0, H=0.5, omega_out=1.0, omega_in=32.0, s=0.1)
DYNAMICS_ORDER = ("alpha1", "alpha0.5", "alpha0", "shift", "lam0.5", "lam0.25", "lam0.1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---------------------------------------------------------------------------------------------------------------
# model specification
# ---------------------------------------------------------------------------------------------------------------
def make_model(variant="alpha", alpha=1.0, kappa=1.0, lam=1.0, name=None, **phys):
    """Model dict: physical constants (PHYS defaults) + variant ('alpha' | 'shift'), alpha, kappa, lam."""
    m = dict(PHYS)
    m.update(phys)
    if variant not in ("alpha", "shift"):
        raise ValueError(variant)
    m.update(variant=variant, alpha=float(alpha) if variant == "alpha" else float("nan"),
             kappa=float(kappa) if variant == "shift" else float("nan"), lam=float(lam))
    if m["lam"] <= 0:
        raise ValueError("lam must be > 0")
    m["name"] = name or (f"alpha{alpha:g}" if variant == "alpha" else "shift") + ("" if lam == 1 else f"_lam{lam:g}")
    return m


def load_dynamics(root=ROOT):
    """The seven dynamics of SCIENTIFIC_PLAN section 3, parameters read from the frozen mechanism configs, in the
    fixed order DYNAMICS_ORDER (its index is the dynamics_index of the seed rule)."""
    out = {}
    for fn in ("matched_free_energy.json", "conditional_relaxation.json"):
        c = json.load(open(os.path.join(root, "configs", "mechanism", fn)))
        ec = c["engine_cfg"]
        phys = dict(beta=ec["beta"], H=ec["H"], omega_out=ec["omega_out"], omega_in=ec["omega_in"], s=ec["s"])
        for v in c["variants"]:
            if v["name"] in out or v["name"] == "lam1":          # lam1 is alpha1
                continue
            out[v["name"]] = make_model(v["variant"], v["alpha"] if v["alpha"] is not None else 1.0,
                                        v["kappa"] if v["kappa"] is not None else 1.0, v["lam"], name=v["name"], **phys)
    missing = [n for n in DYNAMICS_ORDER if n not in out]
    if missing or len(out) != len(DYNAMICS_ORDER):
        raise RuntimeError(f"configs do not define exactly the seven dynamics: missing {missing}, have {sorted(out)}")
    return [out[n] for n in DYNAMICS_ORDER]


def numba_args(m):
    """(var_code, alpha, kappa, lam) for the kernels."""
    if m["variant"] == "alpha":
        return V_ALPHA, float(m["alpha"]), 1.0, float(m["lam"])
    return V_SHIFT, 0.0, float(m["kappa"]), float(m["lam"])


# ---------------------------------------------------------------------------------------------------------------
# closed forms (numpy)
# ---------------------------------------------------------------------------------------------------------------
def omega(x, p):
    return p["omega_out"] + (p["omega_in"] - p["omega_out"]) * np.exp(-x * x / (2 * p["s"] ** 2))


def domega(x, p):
    return -(p["omega_in"] - p["omega_out"]) * (x / p["s"] ** 2) * np.exp(-x * x / (2 * p["s"] ** 2))


def dlogomega(x, p):
    return domega(x, p) / omega(x, p)


def F_exact(x, p):
    return p["H"] * (x * x - 1.0) ** 2 + np.log(omega(x, p)) / p["beta"]


def Fp_exact(x, p):
    return 4.0 * p["H"] * x * (x * x - 1.0) + domega(x, p) / (omega(x, p) * p["beta"])


def _cshift(m):
    return math.sqrt(2.0 / m["beta"]) / m["kappa"]


def m_centre(x, m):
    """Conditional mean of Y: m(x) (shift) or 0 (alpha family)."""
    if m["variant"] == "shift":
        return _cshift(m) * np.log(omega(x, m))
    return np.zeros_like(np.asarray(x, float))


def dm_centre(x, m):
    if m["variant"] == "shift":
        return _cshift(m) * dlogomega(x, m)
    return np.zeros_like(np.asarray(x, float))


def stiffness(x, m):
    """d^2 V / dy^2: omega^(2 alpha) or kappa^2."""
    x = np.asarray(x, float)
    if m["variant"] == "shift":
        return np.full_like(x, m["kappa"] ** 2)
    return omega(x, m) ** (2.0 * m["alpha"])


def V_np(x, y, m):
    if m["variant"] == "shift":
        return F_exact(x, m) + 0.5 * m["kappa"] ** 2 * (y - m_centre(x, m)) ** 2
    a = m["alpha"]
    om = omega(x, m)
    return m["H"] * (x * x - 1.0) ** 2 + ((1.0 - a) / m["beta"]) * np.log(om) + 0.5 * om ** (2 * a) * y * y


def grad_V_np(x, y, m):
    """Analytic (d_x V, d_y V), written from the definitions."""
    if m["variant"] == "shift":
        k2 = m["kappa"] ** 2
        d = y - m_centre(x, m)
        return Fp_exact(x, m) - k2 * d * dm_centre(x, m), k2 * d
    a = m["alpha"]
    om = omega(x, m)
    r = dlogomega(x, m)
    o2a = om ** (2 * a)
    return 4.0 * m["H"] * x * (x * x - 1.0) + ((1.0 - a) / m["beta"]) * r + a * o2a * r * y * y, o2a * y


def cond_var_y(x, m):
    return 1.0 / (m["beta"] * stiffness(x, m))


def cond_var_f(x, m):
    """Exact Var(f | x)."""
    r = dlogomega(x, m)
    if m["variant"] == "shift":
        return m["kappa"] ** 2 * dm_centre(x, m) ** 2 / m["beta"]
    return 2.0 * m["alpha"] ** 2 * r * r / m["beta"] ** 2


def em_factor(x, m, h):
    """Frozen-x EM inflation of Var(Y | x): 1 / (1 - lam k h / 2)."""
    return 1.0 / (1.0 - m["lam"] * stiffness(x, m) * h / 2.0)


def Fp_em(x, m, h):
    """The conditional mean force the EM chain converges to at frozen x (SCIENTIFIC_PLAN section 6)."""
    if m["variant"] == "shift":
        return Fp_exact(x, m)
    a = m["alpha"]
    r = dlogomega(x, m)
    return 4.0 * m["H"] * x * (x * x - 1.0) + ((1.0 - a) / m["beta"]) * r + (a / m["beta"]) * r * em_factor(x, m, h)


def var_f_em(x, m, h):
    if m["variant"] == "shift":
        return cond_var_f(x, m) * em_factor(x, m, h)
    return cond_var_f(x, m) * em_factor(x, m, h) ** 2


def closed_form_peak(m, h):
    """1/(1 - lam k_max h/2) - 1 at the stiffest point (k_max = omega(0)^(2 alpha) or kappa^2)."""
    return float(em_factor(np.array([0.0]), m, h)[0] - 1.0)


def tau_y_frozen(x, m):
    """Frozen-x OU relaxation time of y: 1/(lam k(x))."""
    return 1.0 / (m["lam"] * stiffness(x, m))


def bin_averages(m, nb, h=None, n_sub=2001):
    """Exact per-bin averages on nb equal bins of [XMIN, XMAX] (midpoint rule, n_sub points per bin).
    Flat law (uniform x within the bin, the flat-bias target): Fp_flat, Fp2_flat (E[F*'^2]), var_flat (Var(Y|x)),
    mu_flat (E[Y|x]), varf_flat (Var(f|x)), and with h the EM predictions var_em_flat, Fp_em_flat, varf_em_flat.
    Unbiased Gibbs law exp(-beta F*): P (bin probability up to a constant), F = -kT log P, Fp, var."""
    edges = np.linspace(XMIN, XMAX, nb + 1)
    w = edges[1] - edges[0]
    out = dict(edges=edges, centres=0.5 * (edges[1:] + edges[:-1]))
    for k in ("P", "Fp", "var", "Fp_flat", "Fp2_flat", "var_flat", "mu_flat", "varf_flat", "var_em_flat", "Fp_em_flat",
              "varf_em_flat"):
        out[k] = np.zeros(nb)
    step = max(1, 2_000_000 // n_sub)
    F0 = F_exact(np.array([0.0]), m)[0]
    for j0 in range(0, nb, step):
        j1 = min(nb, j0 + step)
        xs = edges[j0:j1, None] + (np.arange(n_sub)[None, :] + 0.5) * (w / n_sub)
        fp = Fp_exact(xs, m)
        v = cond_var_y(xs, m)
        out["Fp_flat"][j0:j1] = fp.mean(1)
        out["Fp2_flat"][j0:j1] = (fp * fp).mean(1)
        out["var_flat"][j0:j1] = v.mean(1)
        out["mu_flat"][j0:j1] = m_centre(xs, m).mean(1)
        out["varf_flat"][j0:j1] = cond_var_f(xs, m).mean(1)
        wt = np.exp(-m["beta"] * (F_exact(xs, m) - F0))
        P = wt.sum(1) * (w / n_sub)
        out["P"][j0:j1] = P
        out["Fp"][j0:j1] = (wt * fp).sum(1) * (w / n_sub) / P
        out["var"][j0:j1] = (wt * v).sum(1) * (w / n_sub) / P
        if h is not None:
            out["var_em_flat"][j0:j1] = (v * em_factor(xs, m, h)).mean(1)
            out["Fp_em_flat"][j0:j1] = Fp_em(xs, m, h).mean(1)
            out["varf_em_flat"][j0:j1] = var_f_em(xs, m, h).mean(1)
    out["F"] = -np.log(out["P"]) / m["beta"]
    return out


def bin_index(x, nb):
    """The production bin rule (gateway_ladder_numba._bin_of)."""
    j = np.floor((np.asarray(x) - XMIN) / ((XMAX - XMIN) / nb) + 1e-9).astype(np.int64)
    return np.clip(j, 0, nb - 1)


# ---------------------------------------------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _reflect(q, lo, hi):
    """gateway_ladder_numba._reflect / gateway_validation._reflect."""
    span = hi - lo
    qm = (q - lo) % (2.0 * span)
    if qm > span:
        qm = 2.0 * span - qm
    return qm + lo


@njit(cache=True)
def _Fexact(x, H, oout, oin, s, beta):
    """gateway_validation._Fexact (op for op)."""
    om = oout + (oin - oout) * math.exp(-x * x / (2.0 * s * s))
    return H * (x * x - 1.0) ** 2 + math.log(om) / beta


@njit(cache=True)
def _eval(x, y, var_code, alpha, kappa, H, oout, oin, s, beta):
    """(f = d_x V, d_y V, F*'(x), E[Y|x], omega, omega').  The alpha = 1 branch is gateway_validation._forces op
    for op; F*' is gateway_validation._Fpexact op for op (same exp argument, same omega, omega')."""
    two_s2 = 2.0 * s * s
    e = math.exp(-x * x / two_s2)
    om = oout + (oin - oout) * e
    dom = -(oin - oout) * (x / (s * s)) * e
    g = 4.0 * H * x * (x * x - 1.0)
    Fp = 4.0 * H * x * (x * x - 1.0) + dom / (om * beta)
    # the shift / generic-alpha branches are written op for op like src/gateway_family_numba._advance (MODEL_SHIFT,
    # MODEL_ALPHA with its host constants msc, k2, inv_beta, c_ent, two_alpha), so the unbiased harness EM is the
    # production integrator bit for bit for every variant (test M3b)
    if var_code == V_SHIFT:
        c = math.sqrt(2.0 / beta) / kappa
        lr = dom / om
        mu = c * math.log(om)
        fy = (kappa * kappa) * (y - mu)
        fx = g + lr * (1.0 / beta) - fy * (c * lr)
        return fx, fy, Fp, mu, om, dom
    if alpha == 1.0:
        return 4.0 * H * x * (x * x - 1.0) + om * dom * y * y, om * om * y, Fp, 0.0, om, dom
    r = dom / om
    o2a = om ** (2.0 * alpha)
    fx = g + ((1.0 - alpha) / beta) * r + alpha * o2a * r * y * y
    return fx, o2a * y, Fp, 0.0, om, dom


@njit(cache=True)
def _cond_sd(om, dom, var_code, alpha, kappa, beta):
    """(sigma_y(x), sd_f(x)): exact conditional standard deviations of Y and of f."""
    r = dom / om
    if var_code == V_SHIFT:
        c = math.sqrt(2.0 / beta) / kappa
        return 1.0 / (math.sqrt(beta) * kappa), kappa * abs(c * r) / math.sqrt(beta)
    return 1.0 / (math.sqrt(beta) * math.pow(om, alpha)), math.sqrt(2.0) * abs(alpha * r) / beta


@njit(cache=True)
def _V(x, y, om, var_code, alpha, kappa, H, oout, oin, s, beta):
    """V(x, y); the alpha = 1 branch is gateway_validation's MALA expression op for op."""
    if var_code == V_SHIFT:
        c = math.sqrt(2.0 / beta) / kappa
        d = y - c * math.log(om)
        return H * (x * x - 1.0) ** 2 + math.log(om) / beta + 0.5 * kappa * kappa * d * d
    if alpha == 1.0:
        return H * (x * x - 1.0) ** 2 + 0.5 * om * om * y * y
    return H * (x * x - 1.0) ** 2 + ((1.0 - alpha) / beta) * math.log(om) + 0.5 * math.pow(om, 2.0 * alpha) * y * y


@njit(cache=True)
def _window(x, win_c, win_hw):
    for k in range(win_c.shape[0]):
        if abs(x - win_c[k]) <= win_hw:
            return k
    return -1


@njit(cache=True)
def acf_push(i, cnt, zy, zf, w, ry, rf, rw, lags, acc):
    """Push sample number ``cnt`` (z_y, z_f, window label w of x) of walker i into its ring and accumulate, for
    every lag l <= cnt (lags ascending), z(cnt - l) * z(cnt) into acc[window of the ORIGIN sample, lag]."""
    R = ry.shape[1]
    p = cnt % R
    ry[i, p] = zy
    rf[i, p] = zf
    rw[i, p] = w
    for k in range(lags.shape[0]):
        l = lags[k]
        if l > cnt:
            break
        q = p - l
        if q < 0:
            q += R
        wo = rw[i, q]
        if wo < 0:
            continue
        y0 = ry[i, q]
        f0 = rf[i, q]
        acc[wo, k, 0] += 1.0
        acc[wo, k, 1] += y0
        acc[wo, k, 2] += zy
        acc[wo, k, 3] += y0 * y0
        acc[wo, k, 4] += zy * zy
        acc[wo, k, 5] += y0 * zy
        acc[wo, k, 6] += f0
        acc[wo, k, 7] += zf
        acc[wo, k, 8] += f0 * f0
        acc[wo, k, 9] += zf * zf
        acc[wo, k, 10] += f0 * zf


@njit(cache=True)
def run_chain(scheme, x0, y0, n_steps, burn, h, beta, H, oout, oin, s, var_code, alpha, kappa, lam, flat_bias,
              nb1, nb2, seed, trace_every, n_trace, n_acf, acf_stride, lags0, lags1, win_c, win_hw, acf_rec):
    """N independent walkers (x0, y0) for n_steps (see the module docstring).  Returns
    (A1 (9, nb1), A2 (9, nb2), n_acc, n_prop, n_reflect, max|dx|, n_nonfinite, traces_x, traces_y, X, Y,
     crossings (N,) of x = 0, zhist (W, ZH_NB) of z_y for x in each window (all walkers),
     acc0 (W, K0, 11), acc1 (W, K1, 11), rec0 (acf_rec, n_acf, 3), rec1, n_samples (2,))."""
    np.random.seed(seed)
    N = x0.shape[0]
    X = x0.copy()
    Y = y0.copy()
    A1 = np.zeros((9, nb1))
    A2 = np.zeros((9, nb2))
    amp = math.sqrt(2.0 * h / beta)
    ampy = math.sqrt(2.0 * lam * h / beta)
    d1 = (XMAX - XMIN) / nb1
    d2 = (XMAX - XMIN) / nb2
    n_acc = 0.0
    n_prop = 0.0
    n_refl = 0.0
    maxdx = 0.0
    nonfin = 0
    nt = n_steps // trace_every + 1
    trx = np.zeros((nt, n_trace))
    tryy = np.zeros((nt, n_trace))
    kt = 0
    cross = np.zeros(N, np.int64)
    nwin = win_c.shape[0]
    zh = np.zeros((nwin, ZH_NB))
    zh_w = ZH_NB / (2.0 * ZH_LIM)
    has_f = var_code == V_SHIFT or alpha != 0.0
    R0 = lags0[lags0.shape[0] - 1] + 1
    R1 = lags1[lags1.shape[0] - 1] + 1
    na = n_acf if n_acf > 0 else 0
    ry0 = np.zeros((na, R0))
    rf0 = np.zeros((na, R0))
    rw0 = np.full((na, R0), -1, np.int64)
    ry1 = np.zeros((na, R1))
    rf1 = np.zeros((na, R1))
    rw1 = np.full((na, R1), -1, np.int64)
    acc0 = np.zeros((nwin, lags0.shape[0], 11))
    acc1 = np.zeros((nwin, lags1.shape[0], 11))
    rec0 = np.zeros((acf_rec, na, 3))
    rec1 = np.zeros((acf_rec, na, 3))
    cnt0 = 0
    cnt1 = 0
    for step in range(n_steps):
        if step % trace_every == 0 and kt < nt:
            for i in range(n_trace):
                trx[kt, i] = X[i]
                tryy[kt, i] = Y[i]
            kt += 1
        post = step >= burn
        s0 = na > 0 and post and (step - burn) % acf_stride[0] == 0
        s1 = na > 0 and post and (step - burn) % acf_stride[1] == 0
        for i in range(N):
            x = X[i]
            y = Y[i]
            fx, fy, Fp, mu, om, dom = _eval(x, y, var_code, alpha, kappa, H, oout, oin, s, beta)
            if post:
                r = fx - Fp
                d = y - mu
                j = int(math.floor((x - XMIN) / d1 + 1e-9))
                if j < 0:
                    j = 0
                elif j > nb1 - 1:
                    j = nb1 - 1
                A1[0, j] += 1.0
                A1[1, j] += fx
                A1[2, j] += fx * fx
                A1[3, j] += y
                A1[4, j] += y * y
                A1[5, j] += r
                A1[6, j] += r * r
                A1[7, j] += d
                A1[8, j] += d * d
                j = int(math.floor((x - XMIN) / d2 + 1e-9))
                if j < 0:
                    j = 0
                elif j > nb2 - 1:
                    j = nb2 - 1
                A2[0, j] += 1.0
                A2[1, j] += fx
                A2[2, j] += fx * fx
                A2[3, j] += y
                A2[4, j] += y * y
                A2[5, j] += r
                A2[6, j] += r * r
                A2[7, j] += d
                A2[8, j] += d * d
                w = _window(x, win_c, win_hw)
                rec_i = i < na and (s0 or s1)
                if w >= 0 or rec_i:
                    sig, sdf = _cond_sd(om, dom, var_code, alpha, kappa, beta)
                    zy_ = d / sig
                    if w >= 0:
                        b_ = int(math.floor((zy_ + ZH_LIM) * zh_w))
                        if b_ >= 0 and b_ < ZH_NB:
                            zh[w, b_] += 1.0
                    if rec_i:
                        # sign-invariant (module docstring): sd_f > 0 implies omega' != 0
                        zf_ = (r / sdf if dom > 0.0 else -r / sdf) if (has_f and sdf > 0.0) else 0.0
                        if s0:
                            if cnt0 < acf_rec:
                                rec0[cnt0, i, 0] = x
                                rec0[cnt0, i, 1] = zy_
                                rec0[cnt0, i, 2] = zf_
                            acf_push(i, cnt0, zy_, zf_, w, ry0, rf0, rw0, lags0, acc0)
                        if s1:
                            if cnt1 < acf_rec:
                                rec1[cnt1, i, 0] = x
                                rec1[cnt1, i, 1] = zy_
                                rec1[cnt1, i, 2] = zf_
                            acf_push(i, cnt1, zy_, zf_, w, ry1, rf1, rw1, lags1, acc1)
            b = Fp if flat_bias == 1 else 0.0
            zx = np.random.standard_normal()
            zy = np.random.standard_normal()
            if scheme == EM:
                xn = x + (-fx + b) * h + amp * zx
                if xn < XMIN or xn > XMAX:
                    n_refl += 1.0
                xn = _reflect(xn, XMIN, XMAX)
                yn = y - lam * fy * h + ampy * zy
                dd = abs(xn - x)
                if dd > maxdx:
                    maxdx = dd
                if not (math.isfinite(xn) and math.isfinite(yn)):
                    nonfin += 1
                if (xn < 0.0) != (x < 0.0):
                    cross[i] += 1
                X[i] = xn
                Y[i] = yn
            else:
                # MALA on U = V - flat_bias F*, preconditioner diag(1, lam)
                xn = x + (-fx + b) * h + amp * zx
                yn = y - lam * fy * h + ampy * zy
                n_prop += 1.0
                if xn < XMIN or xn > XMAX:
                    continue
                fxn, fyn, Fpn, mun, omn, domn = _eval(xn, yn, var_code, alpha, kappa, H, oout, oin, s, beta)
                bn = Fpn if flat_bias == 1 else 0.0
                U = _V(x, y, om, var_code, alpha, kappa, H, oout, oin, s, beta)
                Un = _V(xn, yn, omn, var_code, alpha, kappa, H, oout, oin, s, beta)
                if flat_bias == 1:
                    U -= _Fexact(x, H, oout, oin, s, beta)
                    Un -= _Fexact(xn, H, oout, oin, s, beta)
                gx, gy = fx - b, fy
                gxn, gyn = fxn - bn, fyn
                fw = (xn - x + gx * h) ** 2 + (yn - y + lam * gy * h) ** 2 / lam
                bw = (x - xn + gxn * h) ** 2 + (y - yn + lam * gyn * h) ** 2 / lam
                la = -beta * (Un - U) - beta * (bw - fw) / (4.0 * h)
                if la >= 0.0 or np.random.random() < math.exp(la):
                    n_acc += 1.0
                    dd = abs(xn - x)
                    if dd > maxdx:
                        maxdx = dd
                    if not (math.isfinite(xn) and math.isfinite(yn)):
                        nonfin += 1
                    if (xn < 0.0) != (x < 0.0):
                        cross[i] += 1
                    X[i] = xn
                    Y[i] = yn
        if s0:
            cnt0 += 1
        if s1:
            cnt1 += 1
    ns = np.zeros(2, np.int64)
    ns[0] = cnt0
    ns[1] = cnt1
    return (A1, A2, n_acc, n_prop, n_refl, maxdx, nonfin, trx[:kt], tryy[:kt], X, Y, cross, zh, acc0, acc1,
            rec0, rec1, ns)


# ---------------------------------------------------------------------------------------------------------------
# python wrappers
# ---------------------------------------------------------------------------------------------------------------
_NOLAG = np.zeros(1, np.int64)


def acf_plan(m, h, n_lags=80, n_small=16, max_ring=8192):
    """Two-stride sampling plan for the conditional ACF of model m at step h.
    level 0 (fine):   stride = floor(tau_fast / (10 h)) >= 1 step (and <= the level-1 stride), with
                      tau_fast = min(tau_min, tau_couple): tau_min = 1/(lam k_max) is the frozen-x gate time, and
                      tau_couple = 1/max_x k(x)(lam + m'(x)^2) the coupled relaxation time of the transverse residual
                      d = y - m(x) when x moves (shifted fibre: the x drift kappa^2 d m' re-aligns the curved channel,
                      dd/dt = -kappa^2 (lam + m'^2) d, ~30x faster than 1/(lam kappa^2) on the flank; alpha family:
                      m' = 0, so tau_fast = tau_min and the plan is unchanged); lags up to 6 tau(x = -0.21) (or
                      6 tau_min if larger), ring <= max_ring;
    level 1 (coarse): stride = floor(tau_max / (20 h)) >= 1, tau_max = tau(x = -1.0) (~ 1/lam); lags up to
                      6 tau_max (>= 5/lam t.u.).
    Lags (in samples): 0..n_small then n_lags log-spaced up to the maximum (unique integers)."""
    tau = {c: float(tau_y_frozen(np.array([c]), m)[0]) for c in ACF_WINDOWS}
    tau_min = float(tau_y_frozen(np.array([0.0]), m)[0])
    tau_max = max(tau.values())
    if m["variant"] == "shift":
        xg = np.linspace(XMIN, XMAX, 72001)
        tau_couple = float(1.0 / np.max(stiffness(xg, m) * (m["lam"] + dm_centre(xg, m) ** 2)))
    else:                                                                # m' = 0: exactly tau_min (no rounding)
        tau_couple = tau_min
    tau_fast = min(tau_min, tau_couple)
    st1 = max(1, int(math.floor(tau_max / (20.0 * h))))
    st0 = min(st1, max(1, int(math.floor(tau_fast / (10.0 * h)))))     # never coarser than level 1
    L0 = int(min(max_ring - 1, math.ceil(6.0 * max(tau[-0.21], tau_min) / (st0 * h))))
    L1 = int(min(max_ring - 1, math.ceil(6.0 * tau_max / (st1 * h))))

    def grid(L):
        g = np.unique(np.concatenate([np.arange(0, min(L, n_small) + 1),
                                      np.round(np.geomspace(max(1, n_small), max(L, n_small), n_lags)).astype(np.int64)]))
        return g[g <= L].astype(np.int64)
    return dict(stride=np.array([st0, st1], np.int64), lags0=grid(L0), lags1=grid(L1), tau_windows=tau,
                tau_min=tau_min, tau_max=tau_max, tau_couple=tau_couple, tau_fast=tau_fast)


def initial_y(x, z, m):
    """y from standard normals z through the exact conditional law of model m."""
    return m_centre(x, m) + z * np.sqrt(cond_var_y(x, m))


def chain(m, scheme, x0, y0, n_steps, burn, h, flat_bias, seed, trace_every=10 ** 9, n_trace=1, n_acf=0, acf=None,
          acf_rec=0, nb1=NB_FINE, nb2=NB_PROD):
    """Run run_chain for model m; returns a dict."""
    vc, a, k, lam = numba_args(m)
    if n_acf > 0:
        st, l0, l1 = acf["stride"], acf["lags0"], acf["lags1"]
    else:
        st, l0, l1 = np.ones(2, np.int64), _NOLAG, _NOLAG
    out = run_chain(int(scheme), np.ascontiguousarray(x0, dtype=np.float64), np.ascontiguousarray(y0, dtype=np.float64),
                    int(n_steps), int(burn), float(h), m["beta"], m["H"], m["omega_out"], m["omega_in"], m["s"], vc, a, k,
                    lam, int(flat_bias), int(nb1), int(nb2), int(seed), int(trace_every), int(n_trace), int(n_acf),
                    np.asarray(st, np.int64), np.asarray(l0, np.int64), np.asarray(l1, np.int64),
                    np.asarray(ACF_WINDOWS, np.float64), float(ACF_HALF_WIDTH), int(acf_rec))
    keys = ("A1", "A2", "n_acc", "n_prop", "n_reflect", "max_dx", "nonfinite", "traces_x", "traces_y", "X", "Y",
            "crossings", "zhist", "acf0", "acf1", "acf_rec0", "acf_rec1", "acf_nsamples")
    r = dict(zip(keys, out))
    r["acf_stride"] = np.asarray(st, np.int64)
    r["lags0"] = np.asarray(l0, np.int64)
    r["lags1"] = np.asarray(l1, np.int64)
    return r


def exact_samples(n, m, seed, flat=True, chunk=1 << 21, nb1=NB_FINE, nb2=NB_PROD):
    """i.i.d. draws from the exact law: x uniform on [XMIN, XMAX] (flat-bias target, flat=True) or by inverse CDF of
    exp(-beta F*) on a 400001-point grid (unbiased Gibbs law); y = E[Y|x] + sd(Y|x) z.  Returns the accumulators
    A1, A2 (same layout as run_chain) and zhist."""
    g = np.random.default_rng(seed)
    if not flat:
        xs = np.linspace(XMIN, XMAX, 400001)
        w = np.exp(-m["beta"] * (F_exact(xs, m) - F_exact(xs, m).min()))
        cdf = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(xs))])
        cdf /= cdf[-1]
    A1 = np.zeros((9, nb1))
    A2 = np.zeros((9, nb2))
    zh = np.zeros((len(ACF_WINDOWS), ZH_NB))
    done = 0
    while done < n:
        k = min(chunk, n - done)
        x = g.uniform(XMIN, XMAX, k) if flat else np.interp(g.random(k), cdf, xs)
        z = g.standard_normal(k)
        mu = m_centre(x, m)
        y = mu + z * np.sqrt(cond_var_y(x, m))
        f, _ = grad_V_np(x, y, m)
        r = f - Fp_exact(x, m)
        d = y - mu
        vals = (np.ones(k), f, f * f, y, y * y, r, r * r, d, d * d)
        for A, nb in ((A1, nb1), (A2, nb2)):
            j = bin_index(x, nb)
            for q, v in enumerate(vals):
                A[q] += np.bincount(j, v, nb)
        for wi, c in enumerate(ACF_WINDOWS):
            sel = np.abs(x - c) <= ACF_HALF_WIDTH
            b = np.floor((z[sel] + ZH_LIM) * (ZH_NB / (2 * ZH_LIM))).astype(np.int64)
            b = b[(b >= 0) & (b < ZH_NB)]
            zh[wi] += np.bincount(b, minlength=ZH_NB)
        done += k
    return A1, A2, zh


# ---------------------------------------------------------------------------------------------------------------
# conditional autocorrelation: offline estimator and read-out
# ---------------------------------------------------------------------------------------------------------------
def acf_offline(xs, zy, zf, lags, win_c=ACF_WINDOWS, win_hw=ACF_HALF_WIDTH):
    """The conditional-ACF lag sums from recorded traces (T samples, n walkers): for each window and lag l, over
    every origin t with x[t] in the window and t + l < T: n, sum z0, z1, z0^2, z1^2, z0 z1 (y then f) with
    z0 = z[t], z1 = z[t + l].  Identical estimator to the in-chain ``acf_push`` (its window label is taken with the
    same |x - c| <= hw rule, first window first)."""
    xs, zy, zf = (np.asarray(a, float) for a in (xs, zy, zf))
    T = xs.shape[0]
    lab = np.full(xs.shape, -1, np.int64)
    for k in range(len(win_c) - 1, -1, -1):
        lab[np.abs(xs - win_c[k]) <= win_hw] = k
    acc = np.zeros((len(win_c), len(lags), 11))
    for kl, l in enumerate(lags):
        if l >= T:
            continue
        lo = lab[:T - l]
        for w in range(len(win_c)):
            sel = lo == w
            y0, y1 = zy[:T - l][sel], zy[l:][sel]
            f0, f1 = zf[:T - l][sel], zf[l:][sel]
            acc[w, kl] = (sel.sum(), y0.sum(), y1.sum(), (y0 * y0).sum(), (y1 * y1).sum(), (y0 * y1).sum(),
                          f0.sum(), f1.sum(), (f0 * f0).sum(), (f1 * f1).sum(), (f0 * f1).sum())
    return acc


def acf_from_sums(acc):
    """Normalised conditional ACF rho(l) = cov(z0, z1) / sqrt(var z0 var z1) per (window, lag) for y and f, and the
    raw second moment E[z0 z1] (z is normalised by the exact conditional law, so its expectation is 1 at lag 0)."""
    n = np.maximum(acc[..., 0], 1e-300)
    out = {}
    for q, o in (("y", 1), ("f", 6)):
        m0, m1 = acc[..., o] / n, acc[..., o + 1] / n
        v0 = acc[..., o + 2] / n - m0 ** 2
        v1 = acc[..., o + 3] / n - m1 ** 2
        c = acc[..., o + 4] / n - m0 * m1
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = c / np.sqrt(v0 * v1)
        rho[acc[..., 0] < 2] = np.nan
        out["rho_" + q] = rho
        out["m01_" + q] = np.where(acc[..., 0] > 0, acc[..., o + 4] / n, np.nan)
    out["n"] = acc[..., 0]
    return out


def tau_integrated(t, rho, t_cut):
    """Trapezoid integral of rho over t in [0, t_cut] (t ascending, t[0] = 0; linear interpolation at t_cut).
    NaN if the lags do not reach t_cut or rho is NaN inside."""
    t = np.asarray(t, float)
    rho = np.asarray(rho, float)
    if t.size < 2 or t[-1] < t_cut or t[0] != 0.0:
        return float("nan")
    k = int(np.searchsorted(t, t_cut, side="right"))
    tt = np.concatenate([t[:k], [t_cut]]) if t[k - 1] < t_cut else t[:k]
    rr = np.interp(tt, t, rho)
    if not np.all(np.isfinite(rr)):
        return float("nan")
    return float(np.sum(0.5 * (rr[1:] + rr[:-1]) * np.diff(tt)))


def tau_efold(t, rho):
    """First time rho falls below 1/e (log-linear interpolation); NaN if it never does."""
    t = np.asarray(t, float)
    rho = np.asarray(rho, float)
    target = math.exp(-1.0)
    for k in range(1, len(t)):
        if np.isfinite(rho[k]) and rho[k] < target:
            r0, r1 = rho[k - 1], rho[k]
            if r0 > 0 and r1 > 0:
                return float(t[k - 1] + (t[k] - t[k - 1]) * (math.log(r0) - math.log(target)) / (math.log(r0) - math.log(r1)))
            return float(t[k - 1] + (t[k] - t[k - 1]) * (r0 - target) / (r0 - r1))
    return float("nan")


def ou_prediction(m, c, hw=ACF_HALF_WIDTH, n=401):
    """Frozen-x OU predictions in the window c +- hw (uniform x, the flat-bias law): tau_y(c), and the window-averaged
    rho_y(t) = <exp(-t / tau_y(x))>, rho_f(t) = <exp(-2 t / tau_y(x))> (alpha family: z_f = (z_y^2 - 1)/sqrt 2) or
    <exp(-t / tau_y(x))> (shift: z_f = -z_y).  Frozen x: x motion is ignored (for the shifted fibre it shortens the
    relaxation of y - m(x) to ~ 1/(kappa^2 (lam + m'^2)), see acf_plan)."""
    xs = np.linspace(c - hw, c + hw, n)
    tau = tau_y_frozen(xs, m)
    fac = 2.0 if m["variant"] == "alpha" else 1.0
    return dict(tau_y_centre=float(tau_y_frozen(np.array([c]), m)[0]), tau_f_centre=float(tau_y_frozen(np.array([c]), m)[0] / fac),
                rho_y=lambda t: np.mean(np.exp(-np.asarray(t, float)[..., None] / tau), -1),
                rho_f=lambda t: np.mean(np.exp(-fac * np.asarray(t, float)[..., None] / tau), -1), f_factor=fac)
