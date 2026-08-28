"""Phase-5 verdict: does Neyman allocation work where variance dominates?

Frozen protocol: ``docs/MECHANISM_CAMPAIGN_PREREGISTRATION.md`` (Phase 5).
Committed BEFORE the arm data exists, so the verdict logic cannot be tuned to it.

Order of operations, and it matters:
  1. GATE: eta_bias < 0.1 on A0 at T, primary metric.  If the gate fails, the
     regime is not variance-dominated and NO Neyman claim may be made either
     way -- the report then stops at the gate.
  2. Only behind the gate: the preregistered comparison.
       measured:   tr(Q Sigma) for A0 / A6a / A6b,   MSE for the same
       predicted:  the diagonal-model ratio  [sum a G / r_b] / [sum a G / r_a]
                   with G = A0's Gamma_hat and r = each arm's realised exposure
     A6b validates the asymptotic theory if tr(Q Sigma) falls from A6a to A6b
     in the predicted direction and by at least half the predicted log-margin;
     it retires the theory if it fails to fall (or rises) with the gate passed.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from abffr import allocation as al                                # noqa: E402

OUT = os.path.join(ROOT, "results", "qr_mechanism", "phase5_arms")
ARMS = ("A0", "A6a", "A6b")


def scoring(xg, mask):
    G = xg.size; dx = float(xg[1] - xg[0])
    H = al.cumulative_trapezoid_matrix(G, dx)
    idx = np.flatnonzero(mask)
    C = np.eye(G); C[np.ix_(np.arange(G), idx)] -= 1.0 / idx.size
    A = (C @ H)[idx, :] / math.sqrt(idx.size)
    return A, A.T @ A, idx


def load(arm):
    F, Cc = [], []
    for p in sorted(glob.glob(os.path.join(OUT, f"{arm}__seed*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            F.append(d["Fp_hat_t"]); Cc.append(d["C_t"])
            meta = dict(x=d["x_grid"], apref=d["Ap_ref"], Aref=d["A_ref"],
                        mask=d["eval_mask"].astype(bool), t=d["t"],
                        gam=d["io_gamma"], a_cell=d["io_a_cell"],
                        edges=d["io_cell_edges"])
    return np.array(F), np.array(Cc), meta


def decompose(Fp_k, meta, A, Q, idx):
    Aref_c = meta["Aref"] - meta["Aref"][meta["mask"]].mean()
    d = Fp_k @ A.T - Aref_c[idx] / math.sqrt(idx.size)
    e2 = (d ** 2).sum(axis=1)
    dbar = d.mean(axis=0)
    Sig = np.cov(Fp_k, rowvar=False, ddof=0)
    return dict(measured=float(e2.mean()),
                R_bias=float((dbar ** 2).sum()),
                tr_Q_Sigma=float(np.sum(Q * Sig)),
                tr_Q_diagSigma=float(np.sum(np.diag(Q) * np.diag(Sig))))


def main():
    data = {arm: load(arm) for arm in ARMS}
    meta = data["A0"][2]
    xg = meta["x"]
    A, Q, idx = scoring(xg, meta["mask"])

    print("=" * 96)
    print("PHASE 5 ARM VERDICT   (analysis committed before the data existed)")
    print("=" * 96)
    out = {}
    for lab, kf in (("T", -1), ("0.5T", 50)):
        print(f"\n--- frame {lab} ---")
        print("%-5s %12s %12s %12s %10s" %
              ("arm", "MSE", "b'Qb", "tr(QSig)", "eta_bias"))
        for arm in ARMS:
            F, Cc, m = data[arm]
            d = decompose(F[:, kf], m, A, Q, idx)
            d["eta_bias"] = d["R_bias"] / max(d["R_bias"] + d["tr_Q_Sigma"], 1e-300)
            out.setdefault(lab, {})[arm] = d
            print("%-5s %12.5g %12.5g %12.5g %10.4f" %
                  (arm, d["measured"], d["R_bias"], d["tr_Q_Sigma"], d["eta_bias"]))

    gate = out["T"]["A0"]["eta_bias"] < 0.1
    print("\nGATE  eta_bias(A0, T) = %.4f  < 0.1  ->  %s"
          % (out["T"]["A0"]["eta_bias"], "PASS" if gate else "FAIL"))
    verdict = {"gate_eta_bias": out["T"]["A0"]["eta_bias"], "gate_pass": bool(gate)}

    if gate:
        # predicted diagonal ratio from A0's Gamma_hat and realised exposures
        _, C0, m0 = data["A0"]
        gam = m0["gam"]                       # A0 rows carry the measured Gamma
        J = m0["a_cell"].size
        cog = np.clip(np.digitize(xg, m0["edges"]) - 1, 0, J - 1)

        def exposure(arm):
            Cc = data[arm][1][:, -1].mean(axis=0)
            r = np.zeros(J)
            np.add.at(r, cog, Cc)
            return r / r.sum()

        g = m0["a_cell"] * np.stack([data["A0"][2]["gam"]]).mean(axis=0)
        risks = {arm: float(np.sum(g / np.maximum(exposure(arm), 1e-12)))
                 for arm in ARMS}
        pred_ratio = risks["A6b"] / risks["A6a"]
        meas_ratio = out["T"]["A6b"]["tr_Q_Sigma"] / out["T"]["A6a"]["tr_Q_Sigma"]
        print("\nNeyman check (behind the gate):")
        print("  predicted tr(QSig) A6b/A6a from diag model + realised exposure: %.3f"
              % pred_ratio)
        print("  measured  tr(QSig) A6b/A6a:                                    %.3f"
              % meas_ratio)
        validated = (meas_ratio < 1.0 and pred_ratio < 1.0
                     and math.log(meas_ratio) <= 0.5 * math.log(pred_ratio))
        print("  variance falls, and by >= half the predicted log-margin -> %s"
              % ("VALIDATED" if validated else "NOT VALIDATED"))
        verdict.update(pred_ratio=pred_ratio, meas_ratio=meas_ratio,
                       neyman_validated=bool(validated))
    else:
        print("\nRegime not variance-dominated: no Neyman claim either way.")

    with open(os.path.join(OUT, "verdict.json"), "w") as fh:
        json.dump(dict(verdict=verdict, decomposition=out), fh, indent=2,
                  default=float)
    print("\nwrote", os.path.relpath(os.path.join(OUT, "verdict.json"), ROOT))


if __name__ == "__main__":
    main()
