"""Reference-free mean-force information adequacy: the clock the ZIF-8 result says
the campaign should have been using.

ABF estimates a CONDITIONAL mean force

    F'(z) = E_nu[ f(q) | xi(q) = z ],

whose accuracy is governed by  SE[F'(z)] ~ sqrt( Var(f|z) / n_eff(z) )  -- NOT by
how uniform the reaction-coordinate marginal p^xi(z) is.  Two bins whose counts
differ 100:1 can both have an accurate conditional mean.  The ethane/ZIF-8 300 K
cell made the distinction concrete: the free energy was within 5% of its final
value at 54 ps while the marginal did not establish until 77 ps, so uniform FR
spent the whole run correcting a mismatch that no longer mattered and paid for it
in bias.

Everything here is computed from a run's OWN accumulators.  No reference free
energy is read, so the resulting clock is deployable as a gate, not merely as a
post-hoc diagnostic.

The recoverable quantities: engines in this repo save, per save t and bin g, the
kernel-smoothed cumulative count `eff_counts` and the regularized mean force

    mean_force(t,g) = smooth(fsum_p)(t,g) / ( smooth(csum_p)(t,g) + min_count ),

so with the post-burn-in offset removed both accumulators come back exactly:

    W(t,g) = eff_counts(t,g) - eff_counts(t_burn,g)
    Y(t,g) = mean_force(t,g) * ( W(t,g) + min_count ).
"""
from __future__ import annotations

import numpy as np

__all__ = ["block_accumulators", "pmf_from_mean_force", "uncertainty_block_bootstrap",
           "uncertainty_across_seeds", "relative_clock"]


def block_accumulators(eff_counts, mean_force, i_burn, min_count):
    """(dW, dY) per BLOCK, shape (n_saves-1, ..., G), from cumulative saves."""
    W = eff_counts - eff_counts[i_burn][None]
    Y = mean_force * (W + min_count)
    dW = np.diff(W[i_burn:], axis=0)
    dY = np.diff(Y[i_burn:], axis=0)
    return dW, dY


def pmf_from_mean_force(mf, dz, periodic):
    """Integrate F'(z) -> F(z), gauge-removed. Matches the engines' convention."""
    if periodic:
        mf = mf - mf.mean(-1, keepdims=True)          # enforce zero net drift
        incr = 0.5 * (mf + np.roll(mf, 1, axis=-1)) * dz
        F = np.cumsum(incr, axis=-1)
    else:
        incr = 0.5 * (mf[..., 1:] + mf[..., :-1]) * dz
        F = np.concatenate([np.zeros_like(mf[..., :1]),
                            np.cumsum(incr, axis=-1)], axis=-1)
    return F - F.mean(-1, keepdims=True)


def uncertainty_block_bootstrap(dW, dY, dz, min_count, periodic, block=5,
                                n_boot=200, seed=0, stride=1):
    """U_F(t) for a SINGLE run, by resampling blocks of its own trajectory.

    This is the deployable estimate: one run, no reference, no seed replicas.
    ``block`` groups consecutive save intervals so that a block is longer than
    the mean force's own correlation time; the caller should check the answer is
    insensitive to it rather than trusting one value.
    """
    rng = np.random.default_rng(seed)
    nb = dW.shape[0] // block
    if nb < 4:
        return np.array([]), np.array([])
    gW = dW[:nb * block].reshape(nb, block, -1).sum(1)
    gY = dY[:nb * block].reshape(nb, block, -1).sum(1)
    out_t, out_u = [], []
    for k in range(4, nb + 1, stride):
        idx = rng.integers(0, k, size=(n_boot, k))
        Wb = gW[idx].sum(1)                    # (n_boot, G)
        Yb = gY[idx].sum(1)
        Fb = pmf_from_mean_force(Yb / (Wb + min_count), dz, periodic)
        out_t.append(k * block)                # in units of save intervals
        out_u.append(float(np.sqrt(Fb.var(axis=0).mean())))
    return np.asarray(out_t), np.asarray(out_u)


def uncertainty_across_seeds(pmf):
    """U_F(t) from the spread of INDEPENDENT seeds -- the honest uncertainty.

    Used only to validate the single-run block bootstrap above; it is not
    deployable, because a deployment has one run."""
    d = pmf - pmf.mean(-1, keepdims=True)
    return np.sqrt(d.var(axis=1, ddof=1).mean(-1))


def relative_clock(t, curve, frac=0.2, hold_frac=0.1, t0=0.0):
    """The SAME relative rule already frozen for T_marg, applied to any curve.

    Returns (time, C0, C_inf, threshold, status). A curve that gets worse is
    'degrading' (inf), not 'converged at t=0'; the sustain window must fit
    inside the data, so a late dip is censored rather than imputed."""
    t = np.asarray(t, float); c = np.asarray(curve, float)
    T = t[-1]
    late = c[t >= 0.8 * T]; late = late[np.isfinite(late)]
    if late.size == 0:
        return float("inf"), np.nan, np.nan, np.nan, "no_data"
    C_inf = float(np.median(late))
    post = np.nonzero((t >= t0) & np.isfinite(c))[0]
    if post.size == 0:
        return float("inf"), np.nan, C_inf, np.nan, "no_data"
    C0 = float(c[post[0]])
    if not (C0 > C_inf):
        return float("inf"), C0, C_inf, np.nan, "degrading"
    thr = C_inf + frac * (C0 - C_inf)
    hold = hold_frac * T
    for i in range(post[0], len(t)):
        if t[i] + hold > T + 1e-12:
            break
        seg = c[(t >= t[i]) & (t <= t[i] + hold)]
        if np.isfinite(seg).all() and (seg <= thr).all():
            return float(t[i]), C0, C_inf, float(thr), "ok"
    return float("inf"), C0, C_inf, float(thr), "never_sustained"
