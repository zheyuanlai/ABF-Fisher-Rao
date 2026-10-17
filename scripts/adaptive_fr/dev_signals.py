"""DEVELOPMENT ONLY: online-observable time series from dev_monitor records (no reference, no ABF arm).
    python scripts/adaptive_fr/dev_signals.py results/adaptive_fr_campaign/dev/<file>.npz [--W 1.0 --R 18]"""
import argparse
import math

import numpy as np


def signals(z, W=1.0, R=18):
    t = z["t"]
    BH, DH = z["BH"].astype(float), z["DH"].astype(float)
    ng = BH.shape[1]
    out = []
    for i, ti in enumerate(t):
        j = int(np.searchsorted(t, ti - W - 1e-9))
        if ti - t[j] < W - 1e-9 or i == j:
            continue
        b = (BH[i] - BH[j]).reshape(R, ng // R).sum(1)
        d = (DH[i] - DH[j]).reshape(R, ng // R).sum(1)
        n = b + d
        net = b - d
        occ = n > 0
        X = float((net[occ] ** 2 / n[occ]).sum())
        r_eff = int(occ.sum())
        zchi = (X - r_eff) / math.sqrt(2 * max(r_eff, 1))
        moved = 0.5 * np.abs(net).sum()
        noise = 0.5 * np.sqrt(2 * n / math.pi).sum()
        K = int(z["occ"][i].sum())
        jdir = max(0.0, moved - noise) / (K * W)            # systematically relocated fraction of K per t.u.
        occc = z["occ"][i].reshape(R, -1).sum(1) / K
        out.append(dict(t=float(ti), events=float(n.sum() / 2), zchi=zchi, jdir=jdir, rawfrac=float(moved / max(n.sum() / 2, 1)),
                        occ_min=float(occc.min() * R), essw=float(z["essw"][i]), maxfw=float(z["maxfw"][i])))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--W", type=float, default=1.0)
    ap.add_argument("--R", type=int, default=18)
    ap.add_argument("--every", type=float, default=1.0)
    a = ap.parse_args()
    for f in a.files:
        z = np.load(f)
        print("==", f.split("/")[-1])
        print("   t    events/t.u.  z_chi   j_dir(%K/tu)  dir.frac  occ_min/unif  ESSw   maxlin")
        for s in signals(z, a.W, a.R):
            if abs(s["t"] / a.every - round(s["t"] / a.every)) < 1e-6:
                print(f"  {s['t']:5.1f}  {s['events'] / a.W:8.1f}  {s['zchi']:7.2f}  {100 * s['jdir']:9.2f}   {s['rawfrac']:6.3f}   {s['occ_min']:6.3f}   {s['essw']:.3f}  {s['maxfw']:.3f}")
