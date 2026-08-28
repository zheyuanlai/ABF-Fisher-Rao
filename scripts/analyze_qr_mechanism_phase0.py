"""Phase 0 decomposition: was A6b > A6a on the K-family bias- or variance-driven?

Frozen protocol: ``docs/MECHANISM_CAMPAIGN_PREREGISTRATION.md``.

Works directly on the saved F_hat profiles (the identity check and eta_bias),
and through the reconstruction operator on Fprime_hat (eta_cov, which needs the
diagonal to be diagonal in MEAN-FORCE space, where the Neyman model lives).
``Q`` is built from the campaign's own metric -- trapezoid quadrature on the
mask, mask-centred -- and verified against ``metrics.l2_error_F`` per profile.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from abffr import allocation as al, metrics                       # noqa: E402

P0 = os.path.join(ROOT, "results", "qr_mechanism", "phase0")
CELLS = ("K0", "K1", "K2", "K3")
ARMS = ("A0", "A6a", "A6b")


def scoring(xg, mask):
    """W^{1/2} (C H) rows for the kappa metric: trapezoid weights, mask centring."""
    G = xg.size
    dx = float(xg[1] - xg[0])
    H = al.cumulative_trapezoid_matrix(G, dx)
    idx = np.flatnonzero(mask)
    xa = xg[idx]
    w = np.zeros(idx.size)
    w[1:-1] = 0.5 * (xa[2:] - xa[:-2])
    w[0] = 0.5 * (xa[1] - xa[0]); w[-1] = 0.5 * (xa[-1] - xa[-2])
    w = w / (xa[-1] - xa[0])
    C = np.eye(G)
    C[np.ix_(np.arange(G), idx)] -= 1.0 / idx.size
    A = np.sqrt(w)[:, None] * (C @ H)[idx, :]
    return A, A.T @ A, idx, w


def load(cell, arm):
    F, Fp, E = [], [], []
    for p in sorted(glob.glob(os.path.join(P0, cell, arm, "seed*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            F.append(d["F_hat"]); Fp.append(d["Fprime_hat"]); E.append(d["e_F"])
            meta = dict(x=d["x_grid"], Fref=d["F_ref"], fpref=d["Fprime_ref"],
                        mask=d["primary_mask"], t=d["t"])
    return np.array(F), np.array(Fp), np.array(E), meta


def decompose_frame(Fp_k, meta, A, Q, idx, w):
    """Bias/variance split of the metric's own e_F^2 at one snapshot."""
    Fref_c = metrics.center(meta["Fref"], meta["mask"])
    d = Fp_k @ A.T - np.sqrt(w) * Fref_c[idx]
    e2 = (d ** 2).sum(axis=1)
    dbar = d.mean(axis=0)
    R_bias = float((dbar ** 2).sum())
    R_var = float(d.var(axis=0, ddof=0).sum())
    Sig = np.cov(Fp_k, rowvar=False, ddof=0)
    tr_full = float(np.sum(Q * Sig))
    tr_diag = float(np.sum(np.diag(Q) * np.diag(Sig)))
    return dict(measured=float(e2.mean()), identity=R_bias + R_var,
                R_bias=R_bias, tr_Q_Sigma=tr_full, tr_Q_diagSigma=tr_diag,
                eta_bias=R_bias / max(R_bias + tr_full, 1e-300),
                eta_cov=(tr_full - tr_diag) / max(tr_full, 1e-300))


def main():
    print("=" * 100)
    print("PHASE 0 -- was the Stage-2 A6b > A6a margin bias- or variance-driven?")
    print("=" * 100)
    # fidelity first: if the rerun is not the old experiment, stop.
    print("\nFidelity gate (rerun vs archived profiles.csv, final e_F):")
    ok = True
    for cell in CELLS:
        row = []
        for arm in ARMS:
            with open(os.path.join(P0, cell, arm, "fidelity.json")) as fh:
                g = json.load(fh)
            row.append("%s %.1e%s" % (arm, g["median_final_rel_dev"],
                                      "" if g["fidelity_pass"] else " FAIL"))
            ok &= g["fidelity_pass"]
        print("  %-4s " % cell + " | ".join(row))
    if not ok:
        print("\n*** FIDELITY GATE FAILED -- decomposition would not be of Stage 2 ***")
    out = {}
    for cell in CELLS:
        per = {}
        for arm in ARMS:
            F, Fp, E, meta = load(cell, arm)
            A, Q, idx, w = scoring(meta["x"], meta["mask"])
            # verify the operator against the campaign's own metric
            k = -1
            chk = np.sqrt(((Fp[0, k] @ A.T - np.sqrt(w)
                            * metrics.center(meta["Fref"], meta["mask"])[idx]) ** 2).sum())
            ref = metrics.l2_error_F(F[0, k], meta["Fref"], meta["x"], meta["mask"])
            assert abs(chk - ref) / ref < 1e-10, (cell, arm, chk, ref)
            frames = {"T": Fp.shape[1] - 1, "0.5T": (Fp.shape[1] - 1) // 2}
            per[arm] = {lab: decompose_frame(Fp[:, kk], meta, A, Q, idx, w)
                        for lab, kk in frames.items()}
        out[cell] = per

    for lab in ("T", "0.5T"):
        print("\n" + "=" * 100)
        print("Decomposition at %s   (primary metric; identity check vs measured e^2)" % lab)
        print("=" * 100)
        print("%-4s %-5s %11s %11s %9s %9s %11s %11s" %
              ("cell", "arm", "measured", "identity", "eta_bias", "eta_cov",
               "R_bias", "tr(QSig)"))
        print("-" * 76)
        for cell in CELLS:
            for arm in ARMS:
                d = out[cell][arm][lab]
                print("%-4s %-5s %11.5g %11.5g %9.4f %9.4f %11.4g %11.4g" %
                      (cell, arm, d["measured"], d["identity"], d["eta_bias"],
                       d["eta_cov"], d["R_bias"], d["tr_Q_Sigma"]))

        print("\nThe decisive ratios, %s:" % lab)
        print("%-4s %14s %14s %14s %16s" %
              ("cell", "MSE 6b/6a", "B-ratio", "V-ratio", "margin carried by"))
        print("-" * 66)
        for cell in CELLS:
            a6a, a6b = out[cell]["A6a"][lab], out[cell]["A6b"][lab]
            mr = a6b["measured"] / a6a["measured"]
            br = a6b["R_bias"] / max(a6a["R_bias"], 1e-300)
            vr = a6b["tr_Q_Sigma"] / max(a6a["tr_Q_Sigma"], 1e-300)
            dm = a6b["measured"] - a6a["measured"]
            db = a6b["R_bias"] - a6a["R_bias"]
            carried = "BIAS" if abs(db) > 0.5 * abs(dm) else "variance"
            print("%-4s %14.3f %14.3f %14.3f %16s" % (cell, mr, br, vr, carried))

    with open(os.path.join(P0, "phase0_decomposition.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\nwrote %s" % os.path.relpath(os.path.join(P0, "phase0_decomposition.json"), ROOT))


if __name__ == "__main__":
    main()
