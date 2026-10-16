"""Read-outs of ERI engine runs (docs/eri_campaign/PREREGISTRATION.md, Amendments 1-2).

Reference: the Gate 1 MC profile on the run's grid (``reference_on_grid``): the pooled MC conditional mean force
per bin (P0-compatible: the exact long-time limit of the histogram estimator on that grid, up to the umbrella's
within-bin reweighting, Gate 1 check 2) integrated with the same histogram_pmf convention.  For n_grid that does not
divide 180 the density route (fine WHAM summed to the grid) is used.

e_F(t)   = RMS over ALL bins of the centred difference F_hat(t) - F_ref  (kJ/mol; additive constant removed)
I_F      = int_0^T e_F dt (trapezoid on the save grid);  Ibar_F = I_F / T
e_F'(t)  = RMS over all bins of Gamma_hat(t) - Gamma_ref  (kJ/mol/rad)
marginal = instantaneous walker histogram; TV to uniform and to the bias-aware target
           Q*_t(phi) ∝ exp(-beta (F_ref - A_t)), A_t = histogram_pmf(ramp(t) w(C_t) Gamma_t) the applied bias potential.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

TWO_PI = 2.0 * math.pi
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def histogram_pmf(G):
    dphi = TWO_PI / G.shape[-1]
    g0 = G - G.mean(-1, keepdims=True)
    Fr = np.cumsum(g0 * dphi, axis=-1)
    Fc = Fr - 0.5 * g0 * dphi
    return Fc - Fc.mean(-1, keepdims=True)


def mean_force(M, C):
    return np.where(C > 0, M / np.maximum(C, 1.0), 0.0)


def reference_on_grid(ng, variant="gard_tait"):
    """(F_ref, Gamma_ref, F_se) on ng bins from results/eri_campaign/reference/<variant>/reference_profile.npz."""
    z = np.load(os.path.join(ROOT, "results", "eri_campaign", "reference", variant, "reference_profile.npz"))
    if 180 % ng == 0:
        k = 180 // ng
        g = z["gamma"] * z["counts"]
        C = z["counts"].reshape(ng, k).sum(1)
        G = g.reshape(ng, k).sum(1) / C
        return histogram_pmf(G), G, None
    raise ValueError("grid must divide 180 (finer grids: use reference_density_on_grid)")


def centred_rms(D):
    D = D - D.mean(-1, keepdims=True)
    return np.sqrt((D * D).mean(-1))


def load_run(path):
    z = np.load(path, allow_pickle=False)
    r = {k: z[k] for k in z.files}
    r["cfg"] = json.loads(str(r["cfg"]))
    return r


def applied_bias_potential(r, s):
    """A_t at save s: histogram_pmf of the APPLIED bin force ramp(t) * w(C) * Gamma."""
    c = r["cfg"]
    k = int(r["save_steps"][s])
    ramp = 1.0 if c["warmup_steps"] <= 0 else min(1.0, k / c["warmup_steps"])
    C = r["sv_C"][s]
    G = mean_force(r["sv_M"][s], C)
    w = np.ones_like(C) if c["trust_count"] <= 0 else np.minimum(1.0, C / c["trust_count"])
    return histogram_pmf(ramp * w * G)


def errors(r, F_ref, G_ref):
    """e_F(t), e_F'(t) over the saves, and I_F, Ibar_F, final errors."""
    G = mean_force(r["sv_M"], r["sv_C"])
    F = histogram_pmf(G)
    eF = centred_rms(F - F_ref[None, :])
    eG = np.sqrt(((G - G_ref[None, :]) ** 2).mean(-1))
    t = r["times"]
    I = float(np.trapezoid(eF, t))
    return dict(t=t, eF=eF, eG=eG, I_F=I, Ibar_F=I / float(t[-1]), eF_T=float(eF[-1]), eG_T=float(eG[-1]))


def marginal(r, F_ref, kT, window_phi=None, n_coarse=18):
    """Per save: window-region walker fraction, its bias-aware target Q*_win, TV(p, uniform), TV(p, Q*) on coarse
    bins, and the exact finite-N floor of TV to a known law."""
    c = r["cfg"]
    ng = r["sv_h"].shape[1]
    N = int(c["N"])
    wphi = c["window_phi"] if window_phi is None else window_phi
    grid = (np.arange(ng) + 0.5) / ng * TWO_PI - math.pi
    win = np.abs(grid) < wphi
    out = {k: [] for k in ("frac_win", "Q_win", "tv_unif", "tv_Q", "tv_floor_Q")}
    for s in range(len(r["save_steps"])):
        h = r["sv_h"][s]
        A = applied_bias_potential(r, s)
        lq = -(F_ref - A) / kT
        Q = np.exp(lq - lq.max())
        Q /= Q.sum()
        p = h / max(h.sum(), 1.0)
        pc = p.reshape(n_coarse, -1).sum(1)
        Qc = Q.reshape(n_coarse, -1).sum(1)
        out["frac_win"].append(float(p[win].sum()))
        out["Q_win"].append(float(Q[win].sum()))
        out["tv_unif"].append(0.5 * float(np.abs(pc - 1.0 / n_coarse).sum()))
        out["tv_Q"].append(0.5 * float(np.abs(pc - Qc).sum()))
        out["tv_floor_Q"].append(tv_floor(N, Qc))
    return {k: np.array(v) for k, v in out.items()}


def tv_floor(N, Q, n_mc=0):
    """E_N[TV(empirical, Q)] for N iid draws from Q: (1/2) sum_k E|X_k/N - Q_k|, X_k ~ Bin(N, Q_k) (exact)."""
    from scipy.stats import binom
    tot = 0.0
    for q in Q:
        if q <= 0:
            continue
        k = np.arange(N + 1)
        pk = binom.pmf(k, N, q)
        tot += float((pk * np.abs(k / N - q)).sum())
    return 0.5 * tot


def establishment(t, frac, Q, N, t_from, z=2.0):
    """Under-established at a save if the window count is below 0.5 N Q* by more than z binomial sd's.
    Returns (T_est = first save time >= t_from after which never under-established again, fraction of saves in
    [t_from, T] under-established, longest contiguous under-established span)."""
    n = frac * N
    thr = 0.5 * N * Q - z * np.sqrt(np.maximum(N * Q * (1 - Q), 0.0))
    under = (n < thr) & (t >= t_from)
    sel = t >= t_from
    if not under.any():
        T_est = float(t_from)
    else:
        last = np.nonzero(under)[0].max()
        T_est = float(t[last + 1]) if last + 1 < len(t) else float("inf")
    span = 0.0
    cur = 0.0
    for i in range(1, len(t)):
        if under[i]:
            cur += t[i] - t[i - 1]
            span = max(span, cur)
        else:
            cur = 0.0
    return T_est, float(under[sel].mean()), span


def conditional_lag(r, ref, n_coarse=18):
    """Population conditional means (eta, u_x^2) per coarse bin vs the MC reference conditional means (count
    weighted), per save: RMS over populated coarse bins of (pop - ref) / ref_sd_proxy is not used; raw RMS."""
    ce = ref["eta"].reshape(n_coarse, -1)
    cc = ref["counts"].reshape(n_coarse, -1)
    eta_ref = (ce * cc).sum(1) / cc.sum(1)
    ux = ref["ux2"].reshape(n_coarse, -1)
    ux_ref = (ux * cc).sum(1) / cc.sum(1)
    sv = r["sv_cond"]
    cnt = sv[:, :, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        eta = sv[:, :, 1] / cnt
        ux2 = sv[:, :, 2] / cnt
    w = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
    d_eta = np.sqrt(np.nansum(w * (eta - eta_ref[None, :]) ** 2, 1))
    d_ux = np.sqrt(np.nansum(w * (ux2 - ux_ref[None, :]) ** 2, 1))
    return dict(d_eta=d_eta, d_ux2=d_ux)


def reference_smooth(ng, variant="gard_tait"):
    """P0-compatible reference on ANY grid from the smooth Fourier representation (K = 32) of the 1800-bin MC WHAM
    profile: Gamma_ref,j = [F_s(e_{j+1}) - F_s(e_j)] / dphi (the exact long-time P0 limit for a uniform within-bin
    density), F_ref = histogram_pmf(Gamma_ref).  Used for the bin-width ladder; production at the frozen grid is
    scored with reference_on_grid (the unsmoothed MC mean force)."""
    z = np.load(os.path.join(ROOT, "results", "eri_campaign", "reference", variant, "reference_fine.npz"))
    c, K = z["fourier_c"], int(z["fourier_K"])

    def fs(x):
        x = np.asarray(x, float)
        X = np.stack([np.ones_like(x)] + [np.cos(k * x) for k in range(1, K + 1)]
                     + [np.sin(k * x) for k in range(1, K + 1)], 1)
        return X @ c
    d = TWO_PI / ng
    e = -math.pi + np.arange(ng + 1) * d
    G = (fs(e[1:]) - fs(e[:-1])) / d
    return histogram_pmf(G), G


def establishment_blocks(t, frac, Q, N, t_from=2.0, block=2.0, deficit=0.25, z=3.0, tau_corr=0.5):
    """Powered establishment test (PREREGISTRATION Amendment 2, replaces the vacuous per-save rule of A1.4-3E).

    The window occupancy ratio R(t) = frac(t) / Q*_W(t) is averaged over consecutive blocks of ``block`` t.u. in
    [t_from, T].  A block is DEFICIENT if  mean R < 1 - deficit  AND  1 - mean R > z * sd_null, where sd_null is the
    binomial null sd of the block mean for an exactly equilibrated population, sd_null^2 = mean_t[(1 - Q)/(N Q)] / n_eff
    with n_eff = block / max(tau_corr, save spacing) effectively independent saves.  Returns (block ratios,
    deficiency flags, longest run of consecutive deficient blocks in t.u., sd_null per block)."""
    t = np.asarray(t, float)
    frac = np.asarray(frac, float)
    Q = np.maximum(np.asarray(Q, float), 1e-12)
    T = float(t[-1])
    edges = np.arange(t_from, T + 1e-9, block)
    ds = float(np.median(np.diff(t)))
    n_eff = block / max(tau_corr, ds)
    ratios, flags, sds = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (t >= a) & (t < b)
        if sel.sum() == 0:
            continue
        Rm = float(np.mean(frac[sel] / Q[sel]))
        sd = float(np.sqrt(np.mean((1 - Q[sel]) / (N * Q[sel])) / n_eff))
        ratios.append(Rm)
        sds.append(sd)
        flags.append(bool(Rm < 1.0 - deficit and (1.0 - Rm) > z * sd))
    run = best = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return np.array(ratios), np.array(flags), best * block, np.array(sds)


def conditional_lag_timeavg(r, ref_means, var, t_from_frac=0.5, tau_corr=0.5, n_coarse=18):
    """Time-averaged population conditional lag (Amendment 2, replaces A1.4-2b).  For each hidden variable X in
    {eta, u_x^2} (columns 1, 2 of sv_cond) and coarse bin b, the population mean over saves t >= t_from_frac T is
    compared with the reference conditional mean; statistic = count-weighted RMS over bins of the lag, floor = the
    same RMS for iid sampling from the reference conditional law with n_eff = sum_t n_b(t) * ds / max(tau_corr, ds).
    Returns {name: (stat, floor)}."""
    sv = r["sv_cond"]
    t = r["times"]
    sel = t >= t_from_frac * t[-1]
    ds = float(np.median(np.diff(t)))
    w_eff = ds / max(tau_corr, ds)
    out = {}
    for col, name in ((1, "eta"), (2, "ux2")):
        cnt = sv[sel, :, 0].sum(0)
        s = sv[sel, :, col].sum(0)
        occ = cnt > 0
        m = np.where(occ, s / np.maximum(cnt, 1), np.nan)
        lag = m - ref_means[name]
        w = cnt / cnt.sum()
        stat = float(np.sqrt(np.nansum(w * lag ** 2)))
        n_eff = np.maximum(cnt * w_eff, 1e-12)
        floor = float(np.sqrt(np.nansum(np.where(occ, w * var[name] / n_eff, 0.0))))
        out[name] = (stat, floor)
    return out
