"""Reference read-outs for the ERI campaign (docs/eri_campaign/PREREGISTRATION.md sections 2-4).

* F_density : periodic WHAM on the NFINE fine bins (bias at fine-bin centres), probabilities summed to the 180 bins,
              F = -kT log p, centred (mean zero over the 180 bins).
* F_MF      : pooled conditional mean force per 180-bin (sum Mf / sum C over all windows), integrated as the EXACT
              integral of the piecewise-constant force with the circular mean subtracted (core_lta.histogram_pmf
              convention), centred.
* U(phi)    : pooled conditional mean potential per bin; -TS = F - U.
* D(A, B)   : debiased RMS of the centred difference on the 180 bins, noise from leave-one-group-out jackknife,
              95 % upper bound from group bootstrap debiased by 2 x noise (LTA validation statistic).
* bead profiles: WHAM-unbiased joint (phi bin, bead coordinate) histograms, marginalised over phi.
"""
from __future__ import annotations

import glob
import math
import os

import numpy as np

NB = 180
TWO_PI = 2.0 * math.pi
DPHI = TWO_PI / NB


def load(dirpath, groups=None, keys=("C_fine", "C", "Mf", "Mf2", "MU", "MU2", "Mbond", "Mbond2", "Mux2", "Muz2",
                                       "Meta", "Hs", "Hx", "C_fine_h1", "C_h1", "Mf_h1", "cnt")):
    """Per-job accumulators stacked as (G, K, ...).  Missing jobs raise."""
    files = sorted(glob.glob(os.path.join(dirpath, "g*_w*.npz")))
    gs = sorted({int(os.path.basename(f)[1:3]) for f in files})
    ws = sorted({int(os.path.basename(f)[5:7]) for f in files})
    if groups is not None:
        gs = [g for g in gs if g in groups]
        missing = set(groups) - set(gs)
        if missing:
            raise FileNotFoundError(f"groups {sorted(missing)} missing in {dirpath}")
    out = {k: [] for k in keys}
    centres = None
    meta = None
    for g in gs:
        rows = {k: [] for k in keys}
        cs = []
        for w in ws:
            z = np.load(os.path.join(dirpath, f"g{g:02d}_w{w:02d}.npz"))
            for k in keys:
                rows[k].append(z[k])
            cs.append(float(z["centre"]))
            if meta is None:
                meta = dict(kappa=float(z["kappa"]), s_max=float(z["s_max"]), scheme=str(z["scheme"]),
                            dt=float(z["dt"]), variant=str(z["variant"]))
        for k in keys:
            out[k].append(np.array(rows[k]))
        centres = np.array(cs)
    out = {k: np.array(v) for k, v in out.items()}
    out["centres"] = centres
    out["groups"] = np.array(gs)
    out.update(meta)
    return out


def wham(H, centres, kappa, kT, tol=1e-11, max_iter=500000, f0=None):
    """Periodic binned WHAM.  H (K, nb) per-window histograms on nb equal bins of [-pi, pi), bias at bin centres.
    Returns (p (nb,) summing to 1, f (K,) dimensionless window free energies, n_iter).

    Solved as the convex minimisation of A(f) = sum_j n_j log sum_w N_w exp(f_w - w_wj) - sum_w N_w f_w (Zhu &
    Hummer) by L-BFGS with an analytic gradient, then polished with fixed-point WHAM iterations until
    max |f_new - f| < tol (the same fixed point as the textbook iteration)."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    H = np.asarray(H, float)
    K, nb = H.shape
    mid = (np.arange(nb) + 0.5) / nb * TWO_PI - math.pi
    d = mid[None, :] - np.asarray(centres)[:, None]
    d = d - TWO_PI * np.floor((d + math.pi) / TWO_PI)
    w = 0.5 * kappa * d * d / kT
    N = H.sum(1)
    num = H.sum(0)
    occ = num > 0
    lN = np.log(np.maximum(N, 1e-300))
    wo = w[:, occ]
    no = num[occ]

    def obj(fr):
        f = np.concatenate([[0.0], fr])
        L = lN[:, None] + f[:, None] - wo                    # (K, nocc)
        lden = logsumexp(L, axis=0)
        A = float((no * lden).sum() - (N * f).sum())
        g = (no[None, :] * np.exp(L - lden[None, :])).sum(1) - N
        return A, g[1:]

    x0 = np.zeros(K - 1) if f0 is None else np.asarray(f0, float)[1:] - float(f0[0])
    r = minimize(obj, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=20000, gtol=1e-12, ftol=1e-16))
    f = np.concatenate([[0.0], r.x])
    ew = np.exp(-w)
    it = 0
    for it in range(max_iter):
        den = (N[:, None] * np.exp(f[:, None]) * ew).sum(0)
        p = np.where(num > 0, num / np.maximum(den, 1e-300), 0.0)
        p /= p.sum()
        fn = -np.log((p[None, :] * ew).sum(1))
        fn -= fn[0]
        if np.max(np.abs(fn - f)) < tol:
            f = fn
            break
        f = fn
    den = (N[:, None] * np.exp(f[:, None]) * ew).sum(0)
    p = np.where(num > 0, num / np.maximum(den, 1e-300), 0.0)
    p /= p.sum()
    return p, f, it


def centre(F):
    return F - F.mean(-1, keepdims=True)


_WARM = {}


def F_density(acc, sel, kT, key="C_fine"):
    """F on the 180 bins from the fine-bin WHAM of the pooled groups ``sel`` (index array into acc[...]).
    Warm-started from the first solution computed for the same (acc, key) (same fixed point, fewer iterations)."""
    H = acc[key][sel].sum(0)
    wk = (id(acc), key)
    p, f, _ = wham(H, acc["centres"], acc["kappa"], kT, f0=_WARM.get(wk))
    _WARM.setdefault(wk, f)
    nf = H.shape[1]
    p180 = p.reshape(NB, nf // NB).sum(1)
    with np.errstate(divide="ignore"):
        F = -kT * np.log(p180)
    return centre(F), f, p


def histogram_pmf(gamma):
    """Exact integral of the piecewise-constant force on the circle at the cell centres (bin width from the array
    length), circular mean of the force subtracted, centred."""
    dphi = TWO_PI / gamma.shape[-1]
    g0 = gamma - gamma.mean(-1, keepdims=True)
    F_right = np.cumsum(g0 * dphi, axis=-1)
    F_c = F_right - 0.5 * g0 * dphi
    return F_c - F_c.mean(-1, keepdims=True)


def F_MF(acc, sel, C="C", M="Mf"):
    c = acc[C][sel].sum((0, 1))
    m = acc[M][sel].sum((0, 1))
    gamma = m / np.maximum(c, 1.0)
    return histogram_pmf(gamma), gamma, c


def cond_mean(acc, sel, key):
    c = acc["C"][sel].sum((0, 1))
    return acc[key][sel].sum((0, 1)) / np.maximum(c, 1.0)


class Profile:
    """A profile estimator with leave-one-group-out jackknife and group-bootstrap resampling."""

    def __init__(self, label, fn, n_groups):
        self.label = label
        self.fn = fn                    # fn(index array of groups) -> profile (NB,)
        self.G = n_groups
        self.F = fn(np.arange(n_groups))
        loo = np.array([fn(np.delete(np.arange(n_groups), g)) for g in range(n_groups)])
        self.se = np.sqrt((n_groups - 1) / n_groups * ((loo - loo.mean(0)) ** 2).sum(0))

    def of(self, idx):
        return self.fn(idx)


class Fixed:
    def __init__(self, label, F):
        self.label = label
        self.F = F
        self.se = np.zeros_like(F)

    def of(self, idx):
        return self.F


def compare(A, B, nboot=2000, seed=12345):
    d = centre(A.F - B.F)
    noise2 = float(np.mean(A.se ** 2 + B.se ** 2))
    rms = float(np.sqrt(np.mean(d ** 2)))
    D = math.sqrt(max(0.0, rms ** 2 - noise2))
    rng = np.random.default_rng(seed)
    GA = getattr(A, "G", 1)
    GB = getattr(B, "G", 1)
    Db = np.zeros(nboot)
    for b in range(nboot):
        ia = rng.integers(0, GA, GA)
        ib = rng.integers(0, GB, GB)
        db = centre(A.of(ia) - B.of(ib))
        Db[b] = math.sqrt(max(0.0, float(np.mean(db ** 2)) - 2.0 * noise2))
    return dict(A=A.label, B=B.label, D=D, D_upper95=float(np.quantile(Db, 0.95)), rms_obs=rms,
                noise=math.sqrt(noise2), max_abs=float(np.abs(d).max()))


def bead_profile(acc, sel, kT, key):
    """WHAM-unbiased marginal of a bead coordinate from the joint (phi bin, bead bin) histograms ``key`` (Hs or Hx).
    Returns beta F over the bead bins (min zero; inf where empty)."""
    H = acc["C_fine"][sel].sum(0)
    p, f, _ = wham(H, acc["centres"], acc["kappa"], kT)
    N = H.sum(1)                                            # samples per window
    mid = (np.arange(NB) + 0.5) / NB * TWO_PI - math.pi
    d = mid[None, :] - acc["centres"][:, None]
    d = d - TWO_PI * np.floor((d + math.pi) / TWO_PI)
    w = 0.5 * acc["kappa"] * d * d / kT
    den = (N[:, None] * np.exp(f[:, None] - w)).sum(0)      # (NB,)
    J = acc[key][sel].sum((0, 1))                           # (NB, nbead)
    pb = (J / den[:, None]).sum(0)
    with np.errstate(divide="ignore"):
        bF = -np.log(pb / pb.sum())
    return bF - bF[np.isfinite(bF)].min()


def barrier_stats(F, U=None, kT=1.0, half_width_bins=2):
    """Barrier = mean F over the window bins (|phi| within half_width_bins of 0) minus min F; U and -TS at the same
    two points (window bins vs the bin of min F, its 2*hw+1 neighbourhood)."""
    j0 = NB // 2
    win = np.arange(j0 - half_width_bins, j0 + half_width_bins)
    jm = int(np.argmin(F))
    cg = (jm + np.arange(-half_width_bins, half_width_bins + 1)) % NB
    out = dict(F_barrier=float(F[win].mean() - F[cg].mean()), jmin=jm, phi_min=float((jm + 0.5) * DPHI - math.pi))
    if U is not None:
        out["U_barrier"] = float(U[win].mean() - U[cg].mean())
        out["mTS_barrier"] = out["F_barrier"] - out["U_barrier"]
    out["F_barrier_kT"] = out["F_barrier"] / kT
    return out
