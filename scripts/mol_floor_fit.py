"""Fit e_F(h, b_mf) and say which term owns the ~0.020 plateau.

The three error sources are separated by construction rather than by fitting all
of them at once:

* the **smoothing** term is computed analytically, by pushing the reference
  profile itself through the same kernel and grid the estimator uses, so it is
  known and does not have to be fitted;
* the **statistical** term is estimated from the spread across independent rows;
* whatever is left is the **discretisation** term, and its `h`-dependence gives
  the integrator's observed order `p`.

Errors are L2 norms of roughly orthogonal deviations, so they combine in
quadrature.  Subtracting the known smoothing contribution from `e_F^2` should
therefore leave a residual that depends on `h` but NOT on the bandwidth -- which
is a real falsifiable check on the decomposition, not a fitted degree of freedom.
"""
from __future__ import annotations

import argparse, json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.estimators import gauge_l2, smoothing_floor as _sfloor
from rcwfr.grid import Grid1D
from rcwfr.mol.refdata import load_reference


def smoothing_floor(g, ref_path, bw, dev, dt):
    """L2 error a PERFECT mean-force estimator still makes at this (grid, bw)."""
    ref = load_reference(ref_path, g, g, dev, dt)
    return float(_sfloor(g, ref["F_ref"], bw)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/mol/floor/BUT_floor_n257.npz")
    ap.add_argument("--ref", default="results/mol/ref/BUT_ref.npz")
    ap.add_argument("--ref2", default=None, help="same system, different h")
    ap.add_argument("--out", default="results/mol/floor/BUT_floor_fit.json")
    a = ap.parse_args()
    dev, dt = torch.device("cuda"), torch.float64
    d = np.load(a.npz)
    hs, bws, ng = d["h"], d["bw"], int(d["ngrid"])
    e, F = d["e_F"], d["F"]
    g = Grid1D(-math.pi, math.pi, ng, -math.pi, math.pi, "periodic")
    mask = g.eval_mask(dev, dt)
    B = np.array([smoothing_floor(g, a.ref, float(b), dev, dt) for b in bws])

    print(f"grid n={ng}   analytic smoothing floors: "
          + "  ".join(f"bw={b:.3f}:{B[i]:.5f}" for i, b in enumerate(bws)))
    print("\ne_F (median over rows), and the residual after removing smoothing:")
    print(f"{'h':>9} | " + " | ".join(f"bw={b:<5.3f}" for b in bws)
          + " || " + " | ".join(f"res {b:<5.3f}" for b in bws))
    res = np.zeros((len(hs), len(bws)))
    for i, h in enumerate(hs):
        med = np.median(e[i], axis=-1)
        res[i] = np.sqrt(np.maximum(med ** 2 - B ** 2, 0.0))
        print(f"{h:9.2e} | " + " | ".join(f"{m:9.5f}" for m in med)
              + " || " + " | ".join(f"{r:9.5f}" for r in res[i]))

    # statistical floor: spread of independent rows about their own mean
    stat = np.zeros((len(hs), len(bws)))
    for i in range(len(hs)):
        for j in range(len(bws)):
            Fi = torch.tensor(F[i, j], device=dev, dtype=dt)
            Fbar = Fi.mean(0, keepdim=True)
            per = gauge_l2(Fi, Fbar[0], mask).cpu().numpy()
            # e_F is a PER-ROW error, so the statistical part of it is the row
            # scatter itself (de-biased for using the sample mean as centre),
            # not the error of the row mean.
            R = F.shape[2]
            stat[i, j] = np.median(per) * math.sqrt(R / max(1, R - 1))
    print("\nstatistical component of a single row's e_F:")
    for i, h in enumerate(hs):
        print(f"{h:9.2e} | " + " | ".join(f"{s:9.5f}" for s in stat[i]))

    # what is left once BOTH known terms are removed: this is the discretisation
    # bias, and it should agree across bandwidths
    bias = np.sqrt(np.maximum(res ** 2 - stat ** 2, 0.0))
    print("\ndiscretisation bias (smoothing AND statistics removed) -- "
          "should agree across bandwidths:")
    for i, h in enumerate(hs):
        print(f"{h:9.2e} | " + " | ".join(f"{b:9.5f}" for b in bias[i]))
    jb = int(np.argmin(bws))

    # reference-free: the constrained arm against ITSELF at the smallest h
    print("\nreference-free self-difference  ||F(h) - F(h_min)||  (no reference used):")
    Fm = torch.tensor(F[-1, jb].mean(0), device=dev, dtype=dt)
    self_d = []
    for i, h in enumerate(hs):
        Fi = torch.tensor(F[i, jb].mean(0), device=dev, dtype=dt)
        v = float(gauge_l2(Fi.unsqueeze(0), Fm, mask)[0]); self_d.append(v)
        print(f"{h:9.2e} | {v:9.5f}")

    sd = np.array(self_d)
    # the self-difference compares two row MEANS, so its noise floor is the
    # error of the mean (stat / sqrt(R)) on each side, added in quadrature
    R = F.shape[2]
    sd_floor = float(np.median(stat)) * math.sqrt(2.0 / R)
    ok = sd > 2 * sd_floor
    pfit = (np.polyfit(np.log(hs[ok]), np.log(sd[ok]), 1) if ok.sum() >= 2
            else [float("nan"), 0.0])
    if ok.sum() >= 2:
        print(f"\nobserved order of the constrained integrator: p = {pfit[0]:.2f}")
    else:
        # only the largest h resolves the bias at all; the rest sit in the noise,
        # so the order can be BOUNDED but not fitted
        j = int(np.argmax(hs[1:])) + 1
        lo = max(sd[1], 2 * sd_floor)
        pl = math.log(sd[0] / lo) / math.log(hs[0] / hs[1])
        print(f"\nonly h={hs[0]:.1e} resolves the bias ({sd[0]:.5f}); every smaller "
              f"h sits under the {2*sd_floor:.5f} noise floor,")
        print(f"  so the order is BOUNDED rather than fitted:  p > {pl:.2f}")
        pfit = [pl, 0.0]
    out = {"h": hs.tolist(), "bw": bws.tolist(), "ngrid": ng,
           "smoothing_floor": B.tolist(),
           "e_F": np.median(e, axis=-1).tolist(), "residual": res.tolist(),
           "stat": stat.tolist(), "bias": bias.tolist(), "order_p": float(pfit[0]),
           "self_diff_vs_hmin": self_d}

    out["self_diff_floor"] = sd_floor

    # what the campaign's own convention was paying for smoothing alone
    print("\nsmoothing floor at other conventions (perfect estimator, no dynamics):")
    for ncmp in (129, 257, 513):
        gc = Grid1D(-math.pi, math.pi, ncmp, -math.pi, math.pi, "periodic")
        row = [smoothing_floor(gc, a.ref, b, dev, dt) for b in (0.08, 0.05, 0.04, 0.02)]
        print(f"  n={ncmp:4d} | " + "  ".join(f"bw={b:.2f}:{v:.5f}"
                                              for b, v in zip((0.08, 0.05, 0.04, 0.02), row)))
        out[f"smoothing_floor_n{ncmp}"] = row

    # the campaign's own plateau, reconstructed from the two measured terms.
    # Only meaningful when the sweep actually CONTAINS the campaign's step --
    # otherwise sd[0] is a self-difference between two other h values.
    if abs(hs[0] - 2e-3) < 1e-9:
        B_cam = out["smoothing_floor_n129"][1]      # bw = 0.05
        D_cam = float(sd[0])                        # integrator bias at h = 2e-3
        tot = math.hypot(B_cam, D_cam)
        print(f"\nthe campaign ran at h=2e-3, b_mf=0.05, n=129.  Its two KNOWN "
              f"numerical terms are")
        print(f"  smoothing {B_cam:.5f} and constrained-integrator {D_cam:.5f} "
              f"kcal/mol, giving {tot:.5f}")
        print(f"  in quadrature before any statistical error -- against an "
              f"observed plateau of ~0.020.")
        out["campaign_smoothing"] = B_cam
        out["campaign_integrator"] = D_cam
        out["campaign_predicted_floor"] = tot

    if a.ref2 and os.path.exists(a.ref2):
        r1 = load_reference(a.ref, g, g, dev, dt)["F_ref"]
        r2 = load_reference(a.ref2, g, g, dev, dt)["F_ref"]
        db = float(gauge_l2(r1, r2[0], mask)[0])
        out["reference_own_h_bias"] = db
        print(f"\nthe REFERENCE's own discretisation bias  ||F_ref(h0) - F_ref(h0/4)|| "
              f"= {db:.5f} kcal/mol")
        print("  (an O(h) reference bias of this size is a floor on any e_F "
              "measured against it)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
