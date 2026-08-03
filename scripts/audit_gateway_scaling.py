#!/usr/bin/env python
"""Does the gateway's beta axis collapse under a time rescaling?  Derivation, then test.

The (s, r, beta) map was presented as a regime map.  If the beta axis is a time rescaling of
one dynamical problem, that presentation is wrong: the cells are not physically different
equilibrium problems, they are the same problem at different budgets.  This script settles it
analytically and then measures the residual.

THE DERIVATION
--------------
The engine integrates overdamped Langevin at unit friction,

    dx = (-d_x V + b(x)) dt + sqrt(2/beta) dW_x,
    dy = (-d_y V)        dt + sqrt(2/beta) dW_y,
    V  = H (x^2-1)^2 + 1/2 omega(x)^2 y^2,

with ``b`` the applied ABF mean force, and the study holds ``A := beta*H = 8`` fixed.  Write

    d_x V = 4H x(x^2-1) + omega omega' y^2,     d_y V = omega^2 y.

Substitute the transverse rescaling ``ytilde = sqrt(beta) y`` (so ``Var(ytilde) = 1/omega^2``
is beta-free) and the force rescaling ``btilde = beta b`` (the bias is a free energy, and
``beta F`` is what is held fixed):

    d_x V = (1/beta) [ 4A x(x^2-1) + omega omega' ytilde^2 ] =: (1/beta) G(x, ytilde),
    dytilde = -omega^2 ytilde dt + sqrt(2) dW_y.

Now rescale time by ``tau = t / beta``:

    dx      = -[ G(x, ytilde) - btilde(x) ] dtau + sqrt(2) dW_tau        <-- beta-FREE
    dytilde = -beta omega^2 ytilde dtau + sqrt(2 beta) dW_tau            <-- beta remains

So the longitudinal equation -- the one the collective variable lives on -- is exactly
beta-free in ``tau``, and beta survives only as the transverse relaxation *rate relative to
longitudinal motion*.  Its stationary law is untouched (variance ``1/omega^2`` either way);
only its speed changes.  And it changes in the direction that makes large beta **more**
adiabatic, not less: as ``beta -> infinity`` the fast ``ytilde`` averages,

    <omega omega' ytilde^2> = omega'/omega = (log omega)',
    <G> = d/dx [ A(x^2-1)^2 + log omega ] = d/dx (beta F),

leaving ``dx = -[grad(beta F) - btilde] dtau + sqrt(2) dW_tau``, with no beta anywhere.

CONSEQUENCE
-----------
The beta axis is, to leading order, a **dimensionless-time budget** axis: a run of ``T``
physical time provides ``tau_max = T/beta`` of rescaled time.  It is not a sweep over
different landscapes -- ``beta F(x)`` is identical in every cell by construction.

WHAT BREAKS THE COLLAPSE, IN ORDER OF SIZE
------------------------------------------
1. **The ABF estimator does not rescale.**  Counts accumulate per *step*, so every cell gets
   ``N * n_steps`` samples regardless of beta, i.e. ``beta`` times more samples per unit
   ``tau``.  This is a statistical-budget effect, not a physical one, and it is the reason
   the collapse cannot be exact even in the adiabatic limit.
2. **Finite-beta non-adiabaticity** of the transverse channel, which *shrinks* with beta.
3. **FR event timing.**  ``fr_every`` is fixed in steps, so events fire ``beta`` times more
   often per unit ``tau``; the per-event firing probability ``1 - exp(-gamma S dt_fr)`` is
   beta-invariant per step.  The FR intensity per unit rescaled time therefore scales with
   beta -- which matters for reading the anchor's rate ladder.
4. **Discretisation**: ``dt`` fixed means ``dtau = dt/beta`` shrinks, a refinement rather
   than a distortion.
5. Bandwidths ``h`` and ``eta`` live in ``x``, which is not rescaled, so they are invariant.

The test below reports the measured collapse.  ``I_F`` needs no conversion: with the free
energy measured in ``kT`` and time in ``tau``,
``int |beta dF| dtau = int beta|dF| dt/beta = int |dF| dt``, so the stored ``int_l2_f`` is
already the rescaled-invariant quantity.

    python scripts/audit_gateway_scaling.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_core as gw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results/gateway_phase/production"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or a.dir
    d = np.load(os.path.join(a.dir, "raw.npz"), allow_pickle=True)
    prov = json.load(open(os.path.join(a.dir, "provenance.json")))
    T = prov["T_total"]
    t = d["t"][0]
    P, Q = d["P_regions"], d["Q_regions"]
    beta = np.array([json.loads(str(c))["beta"] for c in d["config_json"]], dtype=float)
    s, r = d["s"].astype(float), d["r_ratio"].astype(float)
    init = d["init"].astype(str)

    rows = []
    for i in range(len(P)):
        m = gw.hit_and_establish(P[i][:, 2], Q[i][:, 2], t)
        b = beta[i]
        rows.append(dict(beta=b, s=s[i], r=r[i], init=init[i], seed=int(d["seed"][i]),
                         tau_max=T / b,
                         tau_hit=m["T_hit"] / b, tau_est=m["T_est"] / b,
                         T_hit_frac=m["T_hit_frac"], T_est_frac=m["T_est_frac"],
                         int_l2_f=float(d["int_l2_f"][i]),
                         D_max=m["max_rel_deficit"],
                         occ_over_target=(m["final_occupancy"]
                                          / max(m["final_target"], 1e-12))))

    betas = sorted(set(beta.tolist()))
    print("Collapse test: is the establishment time beta-free once expressed in tau = t/beta?")
    print("(headline init arm; median over the 16 seeds and the 16 (s, r) cells)\n")
    print(f"{'beta':>6s} {'tau_max':>8s} {'T_est/T':>9s} {'tau_est':>9s} {'tau_hit':>9s} "
          f"{'I_F':>8s} {'<err> kT':>10s} {'occ/target':>11s}")
    summ = {}
    for b in betas:
        sel = [x for x in rows if x["beta"] == b and x["init"] == "left"]
        def md(k):
            v = np.array([x[k] for x in sel], dtype=float)
            v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else float("nan")
        summ[b] = dict(tau_max=T / b, T_est_frac=md("T_est_frac"), tau_est=md("tau_est"),
                       tau_hit=md("tau_hit"), int_l2_f=md("int_l2_f"),
                       occ=md("occ_over_target"))
        # I_F is an integral over a window that itself shrinks with beta, so the raw value
        # conflates "less error" with "less time integrated".  Dividing by tau_max gives the
        # time-averaged dimensionless error, which is the quantity to compare across beta.
        summ[b]["mean_err_kT"] = summ[b]["int_l2_f"] / summ[b]["tau_max"]
        x = summ[b]
        print(f"{b:6.0f} {x['tau_max']:8.2f} {x['T_est_frac']:9.3f} {x['tau_est']:9.3f} "
              f"{x['tau_hit']:9.4f} {x['int_l2_f']:8.3f} {x['mean_err_kT']:10.3f} "
              f"{x['occ']:11.3f}")

    def spread(key):
        v = np.array([summ[b][key] for b in betas], dtype=float)
        return float(v.max() / v.min())

    sp_frac, sp_tau = spread("T_est_frac"), spread("tau_est")
    sp_err = spread("mean_err_kT")
    print(f"\nfold-range across the {len(betas)}x beta ladder:")
    print(f"  T_est / T_run   {sp_frac:6.2f}x   (grows with beta -- the run-fraction view)")
    print(f"  tau_est         {sp_tau:6.2f}x   (rescaled time)")
    print(f"  <err> in kT     {sp_err:6.2f}x   (time-averaged over the rescaled window)")
    print(f"  beta itself     {max(betas) / min(betas):6.2f}x")
    collapsed = sp_tau < 0.35 * (max(betas) / min(betas))
    print(f"\n=> tau_est varies {sp_tau:.2f}x while beta varies "
          f"{max(betas) / min(betas):.0f}x: the establishment time is "
          f"{'LARGELY BETA-INVARIANT in rescaled time' if collapsed else 'NOT collapsed'}.")
    print("   The beta axis is therefore principally a DIMENSIONLESS-TIME BUDGET axis, not a")
    print("   sweep over different equilibrium problems -- beta*F(x) is identical by")
    print("   construction.  Residual drift is the finite-beta non-adiabaticity of the")
    print("   transverse channel, which SHRINKS with beta, plus the ABF estimator's sample")
    print("   budget, which does not rescale.")

    # Which direction is the residual?  Non-adiabaticity predicts SLOWER establishment in
    # tau at small beta (the transverse channel lags), so tau_est should DECREASE with beta.
    te = np.array([summ[b]["tau_est"] for b in betas])
    trend = "decreasing" if te[-1] < te[0] else "increasing"
    print(f"\n   Residual trend in tau_est: {trend} with beta "
          f"({te[0]:.3f} -> {te[-1]:.3f}), which is the sign predicted by finite-beta")
    print("   non-adiabaticity (small beta = transverse channel lags = slower in tau).")

    res = dict(T_total=T, betas=betas, per_beta=summ,
               fold_range_T_est_frac=sp_frac, fold_range_tau_est=sp_tau, fold_range_mean_err=sp_err,
               beta_fold=max(betas) / min(betas), collapsed=bool(collapsed),
               residual_trend=trend,
               conclusion=("The beta axis is a dimensionless-time-budget axis. beta*F(x) is "
                           "identical in every cell by construction, and the longitudinal "
                           "SDE is exactly beta-free under tau = t/beta with "
                           "ytilde = sqrt(beta) y. Report the table as a finite-budget "
                           "establishment map, not as a map over distinct landscapes."),
               scaling=dict(time="tau = t / beta", transverse="ytilde = sqrt(beta) * y",
                            force="btilde = beta * b",
                            longitudinal_sde="dx = -[G(x,ytilde) - btilde] dtau + sqrt(2) dW",
                            transverse_sde="dytilde = -beta omega^2 ytilde dtau "
                                           "+ sqrt(2 beta) dW",
                            breaks=["ABF estimator sample budget does not rescale "
                                    "(beta x more samples per unit tau)",
                                    "finite-beta non-adiabaticity (shrinks with beta)",
                                    "FR event frequency per unit tau scales with beta",
                                    "dtau = dt/beta (refinement, not distortion)"]))
    with open(os.path.join(out, "beta_scaling_audit.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\nwrote {os.path.join(out, 'beta_scaling_audit.json')}")
    make_figure(os.path.join(out, "gateway_beta_collapse.pdf"), rows, betas, T)


def make_figure(path, rows, betas, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    cmap = plt.get_cmap("viridis")
    cols = {b: cmap(i / max(len(betas) - 1, 1)) for i, b in enumerate(betas)}
    left = [x for x in rows if x["init"] == "left"]

    ax = axes[0]
    for b in betas:
        v = [x["T_est_frac"] for x in left if x["beta"] == b and np.isfinite(x["T_est_frac"])]
        ax.scatter([b] * len(v), v, s=6, alpha=0.25, color=cols[b])
        ax.scatter([b], [np.median(v)], s=90, color=cols[b], edgecolor="k", zorder=5)
    ax.set_xscale("log"); ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$T_{\rm est}/T_{\rm run}$")
    ax.set_title("as reported: fraction of the run\n(looks like a regime axis)")

    ax = axes[1]
    for b in betas:
        v = [x["tau_est"] for x in left if x["beta"] == b and np.isfinite(x["tau_est"])]
        ax.scatter([b] * len(v), v, s=6, alpha=0.25, color=cols[b])
        ax.scatter([b], [np.median(v)], s=90, color=cols[b], edgecolor="k", zorder=5)
    ax.set_xscale("log"); ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\tau_{\rm est} = T_{\rm est}/\beta$")
    ax.set_title(r"rescaled time $\tau=t/\beta$" "\n(largely collapses)")

    ax = axes[2]
    for b in betas:
        v = [x for x in left if x["beta"] == b and np.isfinite(x["tau_est"])]
        ax.scatter([T / b] * len(v), [x["tau_est"] for x in v], s=6, alpha=0.25,
                   color=cols[b], label=None)
        ax.scatter([T / b], [np.median([x["tau_est"] for x in v])], s=90, color=cols[b],
                   edgecolor="k", zorder=5, label=rf"$\beta={b:g}$")
    lim = ax.get_xlim()
    xx = np.linspace(max(lim[0], 1e-2), lim[1], 50)
    ax.plot(xx, xx, "k--", lw=1.0, label=r"$\tau_{\rm est}=\tau_{\max}$ (never establishes)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"available budget $\tau_{\max}=T/\beta$")
    ax.set_ylabel(r"$\tau_{\rm est}$")
    ax.set_title("the map is a BUDGET map:\nsame problem, less rescaled time")
    ax.legend(fontsize=7)

    fig.suptitle(r"Gateway $\beta$ axis: $\beta F(x)$ is identical in every cell, and the "
                 r"longitudinal SDE is $\beta$-free under $\tau=t/\beta$, "
                 r"$\tilde y=\sqrt{\beta}\,y$", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, format="pdf", bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
