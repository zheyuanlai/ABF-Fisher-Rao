"""Offline audit of the IO-ABF risk model.  No simulations, no fitting to the answer.

The question: why does ``sum_j a_j Gamma_j / r_j`` predict a full-domain
*improvement* (0.757 / 0.770 / 0.430 relative to A0) where the measurement shows
1.399 / 1.031 / 1.880 *degradation*?

The derivation behind that functional uses only

    E[e_A^2] ~ tr(Q diag Sigma_f) = sum_j a_j Var[f_hat_j]

whereas the exact finite-sample identity is

    E[e_A^2] = b' Q b + tr(Q Sigma),      b = E[f_hat] - f,   Sigma = Cov(f_hat).

So the model can be missing either of two terms: the finite-time mean-force bias
``b`` (which it assumes away) or the off-diagonal covariance (which it assumes
zero).  This script measures both, on stored profiles only.

Everything is computed in the *scoring operator's own* coordinates, so the
identity is checked rather than asserted: ``Q`` is built from the engine's actual
reconstruction (cumulative trapezoid, then centring, then the mask), and the
reconstruction offset ``A f'_ref - F_ref`` is reported separately rather than
folded silently into the bias.

No matrix is inverted anywhere: with 32 seeds and 181 grid points ``Sigma`` has
rank <= 31, and none of these statistics needs an inverse.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from abffr import allocation as al                                # noqa: E402

OUT = os.path.join(ROOT, "results", "io_abf_overnight", "audit")


# --------------------------------------------------------------------------- #
# the scoring operator, built to match each engine exactly
# --------------------------------------------------------------------------- #
def scoring(xgrid, mask):
    """``A`` and ``Q`` for  e^2 = (1/n) sum_{g in mask} (A f - F_ref)_g^2.

    ``A = P C H``: cumulative trapezoid, then centre on the mask, then keep the
    mask rows -- the same three operations the engines apply, in the same order.
    """
    G = xgrid.size
    dx = float(xgrid[1] - xgrid[0])
    H = al.cumulative_trapezoid_matrix(G, dx)
    idx = np.flatnonzero(mask)
    C = np.eye(G)
    C[np.ix_(np.arange(G), idx)] -= 1.0 / idx.size     # subtract the mask mean
    A = (C @ H)[idx, :]                                 # (n_mask, G)
    Q = (A.T @ A) / idx.size
    return A, Q, idx


def decompose(fhat, fref, Fref_c, A, Q, idx):
    """Exact split of the measured mean squared error into bias and variance.

    ``fhat`` is (S, G) stored mean-force profiles; ``Fref_c`` the engine's stored
    reference free energy, already centred the way the metric centres.
    """
    S = fhat.shape[0]
    d = fhat @ A.T - Fref_c[idx]                        # (S, n_mask) residuals
    e2 = (d ** 2).mean(axis=1)                          # per-seed e_A^2
    dbar = d.mean(axis=0)
    R_bias = float((dbar ** 2).mean())
    R_var = float(d.var(axis=0, ddof=0).mean())

    Sigma = np.cov(fhat, rowvar=False, ddof=0)          # (G, G), rank <= S-1
    tr_full = float(np.sum(Q * Sigma))
    tr_diag = float(np.sum(np.diag(Q) * np.diag(Sigma)))

    b_f = fhat.mean(axis=0) - fref                      # mean-force bias
    recon = A @ fref - Fref_c[idx]                      # reconstruction offset
    return dict(
        n_seeds=S,
        measured=float(e2.mean()),
        identity=R_bias + R_var,
        R_bias=R_bias, R_var=R_var,
        tr_Q_Sigma=tr_full, tr_Q_diagSigma=tr_diag,
        eta_bias=R_bias / max(R_bias + tr_full, 1e-300),
        eta_cov=(tr_full - tr_diag) / max(tr_full, 1e-300),
        R_bias_from_meanforce=float(((A @ b_f) ** 2).mean()),
        R_reconstruction=float((recon ** 2).mean()),
        Sigma=Sigma)


def lag_profile(Sigma, xgrid, n_bins=30):
    """Mean correlation against |z_i - z_j|, for the kernel-smearing question."""
    sd = np.sqrt(np.clip(np.diag(Sigma), 1e-300, None))
    Corr = Sigma / np.outer(sd, sd)
    D = np.abs(xgrid[:, None] - xgrid[None, :])
    keep = np.isfinite(Corr)
    edges = np.linspace(0, D.max() * 0.5, n_bins + 1)
    out = []
    for k in range(n_bins):
        m = keep & (D >= edges[k]) & (D < edges[k + 1])
        if m.sum() > 20:
            out.append((0.5 * (edges[k] + edges[k + 1]), float(Corr[m].mean())))
    return np.array(out)


def corr_length(lag, target=1.0 / np.e):
    """First lag at which the mean correlation drops below 1/e."""
    for z, c in lag:
        if c < target:
            return float(z)
    return float("nan")


# --------------------------------------------------------------------------- #
# the 1/r law, tested without fitting to the arm it predicts
# --------------------------------------------------------------------------- #
def grid_density(occ_cells, cell_of_grid, J, floor=1e-12):
    """Cell occupancy -> per-grid-point density, normalised."""
    n_per = np.bincount(cell_of_grid, minlength=J).astype(float)
    dens = occ_cells[cell_of_grid] / np.maximum(n_per[cell_of_grid], 1.0)
    return np.maximum(dens / dens.sum(), floor)


def one_over_r_test(Sig0, r0, r1, Q):
    """Predict Sigma(r1) from Sigma(r0) under Sigma(r) = D(r)^-1/2 K D(r)^-1/2.

    ``K`` is estimated from the A0 arm alone and never sees the A6b covariance,
    so this is a prediction rather than a fit.
    """
    s0, s1 = np.sqrt(r0), np.sqrt(r1)
    K = (s0[:, None] * Sig0) * s0[None, :]
    Sig_pred = (K / s1[:, None]) / s1[None, :]
    return float(np.sum(Q * Sig_pred))


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_batched(system, arm):
    """EB / gateway: final mean-force profile per seed (all that is stored)."""
    F, occ = [], []
    for p in sorted(glob.glob(os.path.join(
            ROOT, "results/io_abf_overnight", system, "confirmatory", f"{arm}__*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            F.append(d["Fp_hat"]); occ.append(d["io_occupancy_t"][-1])
            meta = dict(xgrid=d["x_grid"], fref=d["Fp_ref"], Fref=d["F_ref"],
                        a_cell=d["io_a_cell"], edges=d["io_cell_edges"])
    return np.array(F), np.array(occ), meta


def load_wca(phase, arm, frame=-1):
    """WCA: mean force at a chosen save frame -- the only system that stores them."""
    import wca_abffr_core as core
    ref = np.load(os.path.join(ROOT, "cache/phase_hp_v3",
                               "wca_ti_b1_h2_w2_n10_a1.5_g160.npz"), allow_pickle=True)
    F, occ = [], []
    for p in sorted(glob.glob(os.path.join(
            ROOT, "results/io_abf_overnight/wca", phase, f"{arm}__*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            F.append(np.asarray(d["mean_force"], float)[frame])
            occ.append(np.asarray(d["io_occupancy_t"], float)[-1])
            meta = dict(a_cell=d["io_a_cell"], edges=d["io_cell_edges"],
                        times=np.asarray(d["times"], float))
    sim = core.SimConfig()
    meta.update(xgrid=np.asarray(ref["grid"], float),
                fref=np.asarray(ref["mean_force"], float),
                Fref=np.asarray(ref["free_energy"], float),
                emask=core.eval_window_mask_np(np.asarray(ref["grid"], float), sim))
    return np.array(F), np.array(occ), meta


#: The engines' own evaluation windows, so the audit scores exactly the metric the
#: campaign reported.  Deriving the mask from ``a_cell > 0`` instead would widen it
#: to whole cells that merely straddle the boundary, and the audit would then be
#: decomposing a different number from the one it is trying to explain.
EVAL_WINDOW = {"eb_beta4": (-1.5, 1.5), "eb_beta8": (-1.5, 1.5),
               "gateway": (-1.5, 1.5)}


def masks_for(meta, system):
    xg = meta["xgrid"]
    if system == "wca":
        primary = meta["emask"]
    else:
        lo, hi = EVAL_WINDOW[system]
        primary = (xg >= lo) & (xg <= hi)
    return primary, np.ones_like(xg, bool)


def centred(Fref, mask):
    return Fref - Fref[mask].mean()


def run_system(system, load, arms=("A0", "A6b")):
    res = {}
    for arm in arms:
        F, occ, meta = load(arm)
        if F.size == 0:
            continue
        xg = meta["xgrid"]
        prim, full = masks_for(meta, system)
        entry = {"n_seeds": len(F), "occ": occ, "meta": meta}
        for scope, mask in (("primary", prim), ("full", full)):
            A, Q, idx = scoring(xg, mask)
            entry[scope] = decompose(F, meta["fref"], centred(meta["Fref"], mask),
                                     A, Q, idx)
            entry[scope]["Q"] = Q
        entry["lag"] = lag_profile(entry["full"]["Sigma"], xg)
        entry["corr_length"] = corr_length(entry["lag"])
        res[arm] = entry
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    systems = {}
    for s in ("eb_beta4", "eb_beta8", "gateway"):
        systems[s] = run_system(s, lambda arm, s=s: load_batched(s, arm))
    systems["wca"] = run_system("wca", lambda arm: (
        load_wca("screening", "A0") if arm == "A0" else load_wca("pilot", arm)))

    # ---- 1. does the exact identity reproduce the measured error? ----------
    print("=" * 96)
    print("1. IDENTITY CHECK   E[e^2] =?= b'Qb + tr(Q Sigma)     (exact, no model)")
    print("=" * 96)
    print("%-10s %-4s %-8s %12s %12s %12s %10s" %
          ("system", "arm", "scope", "measured", "bias+var", "rel.err", "recon"))
    print("-" * 84)
    for s, r in systems.items():
        for arm, e in r.items():
            for scope in ("primary", "full"):
                d = e[scope]
                rel = abs(d["identity"] - d["measured"]) / max(d["measured"], 1e-300)
                print("%-10s %-4s %-8s %12.6g %12.6g %12.2e %10.4g" %
                      (s, arm, scope, d["measured"], d["identity"], rel,
                       d["R_reconstruction"]))

    # ---- 2. which term is the model missing? ------------------------------
    print()
    print("=" * 96)
    print("2. WHICH TERM IS MISSING     eta_bias = bias share,  eta_cov = off-diagonal share")
    print("=" * 96)
    print("%-10s %-4s %-8s %10s %10s %12s %12s %12s" %
          ("system", "arm", "scope", "eta_bias", "eta_cov", "R_bias", "tr(Q dSig)",
           "tr(Q Sig)"))
    print("-" * 88)
    for s, r in systems.items():
        for arm, e in r.items():
            for scope in ("primary", "full"):
                d = e[scope]
                print("%-10s %-4s %-8s %10.4f %10.4f %12.4g %12.4g %12.4g" %
                      (s, arm, scope, d["eta_bias"], d["eta_cov"], d["R_bias"],
                       d["tr_Q_diagSigma"], d["tr_Q_Sigma"]))

    # ---- 3. how far does the covariance reach? ----------------------------
    print()
    print("=" * 96)
    print("3. COVARIANCE RANGE   mean corr vs |z_i - z_j|, A0 arm, full domain")
    print("=" * 96)
    bw = {"eb_beta4": 0.07, "eb_beta8": 0.07, "gateway": 0.07, "wca": 0.025}
    for s, r in systems.items():
        if "A0" not in r:
            continue
        e = r["A0"]; lag = e["lag"]; xg = e["meta"]["xgrid"]
        cw = float(e["meta"]["edges"][1] - e["meta"]["edges"][0])
        print("  %-10s corr length (1/e) = %.4f | ABF bandwidth h = %.3f | "
              "cell width = %.4f | grid dx = %.4f"
              % (s, e["corr_length"], bw[s], cw, float(xg[1] - xg[0])))
        take = [lag[np.argmin(np.abs(lag[:, 0] - z))] for z in
                (0.02, 0.05, 0.1, 0.2, 0.4) if z <= lag[:, 0].max()]
        print("     " + "  ".join("corr(%.2f)=%+.3f" % (z, c) for z, c in take))

    # ---- 4. does Sigma scale as 1/r? --------------------------------------
    print()
    print("=" * 96)
    print("4. THE 1/r LAW      K estimated from A0 ONLY, then used to predict A6b")
    print("=" * 96)
    print("%-10s %-8s %14s %14s %10s" %
          ("system", "scope", "predicted", "measured", "pred/meas"))
    print("-" * 60)
    for s, r in systems.items():
        if not {"A0", "A6b"} <= set(r):
            continue
        e0, e1 = r["A0"], r["A6b"]
        xg = e0["meta"]["xgrid"]
        J = e0["meta"]["a_cell"].size
        cog = np.clip(np.digitize(xg, e0["meta"]["edges"]) - 1, 0, J - 1)
        r0 = grid_density(e0["occ"].mean(axis=0), cog, J)
        r1 = grid_density(e1["occ"].mean(axis=0), cog, J)
        for scope in ("primary", "full"):
            pred = one_over_r_test(e0[scope]["Sigma"], r0, r1, e1[scope]["Q"])
            meas = e1[scope]["tr_Q_Sigma"]
            print("%-10s %-8s %14.6g %14.6g %10.3f" %
                  (s, scope, pred, meas, pred / max(meas, 1e-300)))

    # ---- 5. inside vs outside the mask ------------------------------------
    print()
    print("=" * 96)
    print("5. SPATIAL SPLIT    the same decomposition restricted to outside-mask cells")
    print("=" * 96)
    print("%-10s %-4s %10s %10s %12s %12s" %
          ("system", "arm", "eta_bias", "eta_cov", "R_bias", "tr(Q Sig)"))
    print("-" * 62)
    outside = {}
    for s, r in systems.items():
        for arm, e in r.items():
            xg = e["meta"]["xgrid"]
            prim, _ = masks_for(e["meta"], s)
            if (~prim).sum() < 3:
                continue
            A, Q, idx = scoring(xg, ~prim)
            F, occ, meta = (load_batched(s, arm) if s != "wca" else
                            (load_wca("screening", arm) if arm == "A0"
                             else load_wca("pilot", arm)))
            d = decompose(F, meta["fref"], centred(meta["Fref"], ~prim), A, Q, idx)
            outside[(s, arm)] = d
            print("%-10s %-4s %10.4f %10.4f %12.4g %12.4g" %
                  (s, arm, d["eta_bias"], d["eta_cov"], d["R_bias"], d["tr_Q_Sigma"]))

    # ---- 6. time resolution, where the data allows it ---------------------
    print()
    print("=" * 96)
    print("6. TIME COURSE      WCA only -- EB and gateway store no profile time series")
    print("=" * 96)
    _, _, m0 = load_wca("screening", "A0")
    times = m0["times"]; T = times[-1]
    print("%-6s %-4s %10s %10s %12s %12s" %
          ("t/T", "arm", "eta_bias", "eta_cov", "R_bias", "tr(Q Sig)"))
    print("-" * 58)
    for frac in (0.3, 0.5, 0.7, 1.0):
        k = int(np.argmin(np.abs(times - frac * T)))
        for arm, phase in (("A0", "screening"), ("A6b", "pilot")):
            F, occ, meta = load_wca(phase, arm, frame=k)
            prim, _ = masks_for(meta, "wca")
            A, Q, idx = scoring(meta["xgrid"], prim)
            d = decompose(F, meta["fref"], centred(meta["Fref"], prim), A, Q, idx)
            print("%-6.1f %-4s %10.4f %10.4f %12.4g %12.4g" %
                  (frac, arm, d["eta_bias"], d["eta_cov"], d["R_bias"], d["tr_Q_Sigma"]))

    payload = {}
    for s, r in systems.items():
        payload[s] = {arm: {sc: {k: v for k, v in e[sc].items()
                                 if k not in ("Sigma", "Q")}
                            for sc in ("primary", "full")}
                      for arm, e in r.items()}
        for arm in r:
            if (s, arm) in outside:
                payload[s][arm]["outside"] = {
                    k: v for k, v in outside[(s, arm)].items()
                    if k not in ("Sigma", "Q")}
        payload[s]["_corr_length"] = {arm: r[arm]["corr_length"] for arm in r}
    with open(os.path.join(OUT, "risk_model_audit.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print("\nwrote %s" % os.path.relpath(os.path.join(OUT, "risk_model_audit.json"), ROOT))


if __name__ == "__main__":
    main()
