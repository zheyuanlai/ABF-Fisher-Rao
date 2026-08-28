"""Gates and predictions for the prescribed-r experiments (Phases 1, 2, 4).

Frozen protocol: ``docs/MECHANISM_CAMPAIGN_PREREGISTRATION.md``.

The bias model under test, computed from the measured cumulative counts and the
exact mean force, never from the estimator itself:

    f_tilde = smooth(C_t * f) / (smooth(C_t) + m)         [full prediction]
    b_smooth ~ (mu2_eff/2) [ f'' + 2 f' dlog r ]          [asymptotic form]
    b_pseudo = - m f / (smooth(C_t) + m)                  [starvation term]

Gates: P1-a corr(pred, meas) > 0.95 on the eval window; P1-b predicted b'Qb
within 20 % of measured, per target.
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
import torch                                                      # noqa: E402
import eb_abffr_core as eb                                        # noqa: E402
from abffr import allocation as al                                # noqa: E402

OUT = os.path.join(ROOT, "results", "qr_mechanism")
DEV, DT = torch.device("cpu"), torch.float64


def smooth_np(v, h, dx):
    k, r = eb.gaussian_kernel(h, dx, DEV, DT)
    t = torch.as_tensor(np.atleast_2d(v), dtype=DT)
    return eb.smooth(t, k, r, dx).numpy()


def mu2_eff(h, dx):
    k, r = eb.gaussian_kernel(h, dx, DEV, DT)
    k = k.numpy(); t = np.arange(-r, r + 1) * dx
    w = k / k.sum()
    return float(np.sum(w * t ** 2))


def load_tag(tag):
    by = {}
    for p in sorted(glob.glob(os.path.join(OUT, tag, "a*_k*_seed*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            key = (float(d["alpha"]), int(d["k"]))
            by.setdefault(key, []).append({k: d[k] for k in d.files})
    return by


def scoring_F(xg, mask):
    """EB metric: uniform-weight RMS on the mask, mask-centred, cumtrapz."""
    G = xg.size; dx = float(xg[1] - xg[0])
    H = al.cumulative_trapezoid_matrix(G, dx)
    idx = np.flatnonzero(mask)
    C = np.eye(G); C[np.ix_(np.arange(G), idx)] -= 1.0 / idx.size
    A = (C @ H)[idx, :] / math.sqrt(idx.size)
    return A, idx


def phase1(tag="phase1", frame=-1, report_key=None):
    by = load_tag(tag)
    first = next(iter(by.values()))[0]
    xg = first["x_grid"]; dx = float(xg[1] - xg[0])
    mask = first["eval_mask"].astype(bool)
    f = first["Fp_ref"]; h = float(first["h"]); m = float(first["min_count"])
    A, idx = scoring_F(xg, mask)
    mu2 = mu2_eff(h, dx)

    print("=" * 100)
    print(f"[{tag}] bias-model gates   h={h:g}  m={m:g}  mu2_eff={mu2:.3e}  "
          f"frame={frame}")
    print("=" * 100)
    print("%-12s %8s %10s %10s %12s %12s %8s %8s" %
          ("target", "corr", "corr_asym", "amp_meas", "bQb_meas", "bQb_pred",
           "P1-a", "P1-b"))
    print("-" * 88)
    res = {}
    for (a_, k_), recs in sorted(by.items()):
        Fp = np.stack([r["Fp_hat_t"][frame] for r in recs])
        Ct = np.stack([r["C_t"][frame] for r in recs])
        b_meas = Fp.mean(axis=0) - f
        # full prediction from measured counts + exact f
        pred = np.stack([
            (smooth_np(C * f, h, dx)[0]) / (smooth_np(C, h, dx)[0] + m + eb.EPS)
            for C in Ct])
        b_pred = pred.mean(axis=0) - f
        # asymptotic smooth term (interior only; needs r > 0 everywhere)
        L = eb.XMAX
        dlogr = -a_ * (k_ * math.pi / L) * np.sin(k_ * math.pi * xg / L)
        fpp = np.gradient(np.gradient(f, xg), xg)
        fp1 = np.gradient(f, xg)
        b_asym = 0.5 * mu2 * (fpp + 2.0 * fp1 * dlogr)
        c = float(np.corrcoef(b_pred[mask], b_meas[mask])[0, 1])
        ca = float(np.corrcoef(b_asym[mask], b_meas[mask])[0, 1])
        bQb_m = float(((A @ b_meas) ** 2).sum())
        bQb_p = float(((A @ b_pred) ** 2).sum())
        ok_a = c > 0.95
        ok_b = abs(bQb_p - bQb_m) / max(bQb_m, 1e-300) < 0.20
        res[f"a{a_:+g}_k{k_}"] = dict(
            corr=c, corr_asym=ca, bQb_meas=bQb_m, bQb_pred=bQb_p,
            amp_meas=float(np.abs(b_meas[mask]).max()),
            gate_a=bool(ok_a), gate_b=bool(ok_b))
        print("%-12s %8.4f %10.4f %10.4g %12.4g %12.4g %8s %8s" %
              (f"a={a_:+g},k={k_}", c, ca, np.abs(b_meas[mask]).max(),
               bQb_m, bQb_p, "PASS" if ok_a else "FAIL",
               "PASS" if ok_b else "FAIL"))
    if report_key:
        with open(os.path.join(OUT, f"{report_key}.json"), "w") as fh:
            json.dump(res, fh, indent=2)
    return res


def phase2_h():
    print("\n" + "=" * 100)
    print("PHASE 2 -- h scaling of the interior bias amplitude (prediction: ~ h^2)")
    print("=" * 100)
    amps = {}
    for tag, h in (("phase2_h0.035", 0.035), ("phase1", 0.07), ("phase2_h0.14", 0.14)):
        by = load_tag(tag)
        key = (2.0, 1)
        if key not in by:
            continue
        recs = by[key]
        f = recs[0]["Fp_ref"]; mask = recs[0]["eval_mask"].astype(bool)
        Fp = np.stack([r["Fp_hat_t"][-1] for r in recs])
        b = Fp.mean(axis=0) - f
        amps[h] = float(np.sqrt(np.mean(b[mask] ** 2)))
        print("  h=%.3f  rms interior bias = %.5g" % (h, amps[h]))
    if len(amps) == 3:
        hs = np.array(sorted(amps)); bs = np.array([amps[x] for x in hs])
        slope = np.polyfit(np.log(hs), np.log(bs), 1)[0]
        print("  fitted exponent d log b / d log h = %.2f   "
              "(gate: within [1.6, 2.4] -> %s)"
              % (slope, "PASS" if 1.6 <= slope <= 2.4 else "FAIL"))
        return dict(amps={f"{k:g}": v for k, v in amps.items()},
                    exponent=float(slope), gate=bool(1.6 <= slope <= 2.4))


def phase2_m():
    print("\n" + "=" * 100)
    print("PHASE 2 -- pseudocount term: b in low-support regions vs -m f/(smooth(C)+m)")
    print("=" * 100)
    out = {}
    for tag, m in (("phase2_m0.1", 0.1), ("phase1", 1.0), ("phase2_m10", 10.0)):
        by = load_tag(tag)
        key = (2.0, 1)
        if key not in by:
            continue
        recs = by[key]
        xg = recs[0]["x_grid"]; dx = float(xg[1] - xg[0])
        f = recs[0]["Fp_ref"]; h = float(recs[0]["h"])
        Fp = np.stack([r["Fp_hat_t"][-1] for r in recs])
        Ct = np.stack([r["C_t"][-1] for r in recs])
        b = Fp.mean(axis=0) - f
        smC = np.stack([smooth_np(C, h, dx)[0] for C in Ct]).mean(axis=0)
        pred = -m * f / (smC + m)
        starved = smC < 10.0 * m           # where the term should matter
        lab = ("starved cells: %3d" % starved.sum())
        if starved.sum() >= 3:
            c = float(np.corrcoef(pred[starved], b[starved])[0, 1])
            ratio = float(np.median(b[starved] / np.where(
                np.abs(pred[starved]) > 1e-12, pred[starved], np.nan)))
            print("  m=%-5g %s  corr(pred,b)=%.4f  median b/pred=%.3f" %
                  (m, lab, c, ratio))
            out[f"{m:g}"] = dict(corr=c, ratio=ratio, n_starved=int(starved.sum()))
        else:
            print("  m=%-5g %s  (too few starved cells to test)" % (m, lab))
            out[f"{m:g}"] = dict(n_starved=int(starved.sum()))
    return out


def phase4():
    print("\n" + "=" * 100)
    print("PHASE 4 -- realizability: same target (a=+2,k=1), beta sweep")
    print("=" * 100)
    print("%-6s %12s %12s %12s %12s" %
          ("beta", "C_force", "TV(occ, r)", "rms bias", "bQb"))
    print("-" * 60)
    out = {}
    for tag, beta in (("phase4_beta1", 1.0), ("phase4_beta2", 2.0),
                      ("phase4_beta4", 4.0), ("phase1", 8.0),
                      ("phase4_beta16", 16.0)):
        by = load_tag(tag)
        key = (2.0, 1)
        if key not in by:
            continue
        recs = by[key]
        xg = recs[0]["x_grid"]; dx = float(xg[1] - xg[0])
        mask = recs[0]["eval_mask"].astype(bool)
        f = recs[0]["Fp_ref"]
        A, idx = scoring_F(xg, mask)
        L = eb.XMAX
        logr = 2.0 * np.cos(math.pi * xg / L)
        r = np.exp(logr - logr.max()); r /= r.sum() * dx
        dlr = -2.0 * (math.pi / L) * np.sin(math.pi * xg / L)
        cforce = float(np.trapezoid((dlr / beta) ** 2 * r, xg))
        # realised occupancy from the last inter-save count increment
        occ = np.stack([r_["C_t"][-1] - r_["C_t"][-2] for r_ in recs]).mean(axis=0)
        occ = occ / max(occ.sum(), 1e-300)
        tv = float(0.5 * np.abs(occ - r * dx).sum())
        Fp = np.stack([r_["Fp_hat_t"][-1] for r_ in recs])
        b = Fp.mean(axis=0) - f
        out[f"{beta:g}"] = dict(C_force=cforce, TV=tv,
                                rms_bias=float(np.sqrt(np.mean(b[mask] ** 2))),
                                bQb=float(((A @ b) ** 2).sum()))
        d = out[f"{beta:g}"]
        print("%-6g %12.4g %12.4f %12.4g %12.4g" %
              (beta, d["C_force"], d["TV"], d["rms_bias"], d["bQb"]))
    return out


def main():
    rep = {"phase1": phase1(report_key="phase1_gates")}
    if glob.glob(os.path.join(OUT, "phase2_h*", "*.npz")):
        rep["phase2_h"] = phase2_h()
        rep["phase2_m"] = phase2_m()
    if glob.glob(os.path.join(OUT, "phase4_beta*", "*.npz")):
        rep["phase4"] = phase4()
    with open(os.path.join(OUT, "prescribed_r_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=float)
    print("\nwrote results/qr_mechanism/prescribed_r_report.json")


if __name__ == "__main__":
    main()
