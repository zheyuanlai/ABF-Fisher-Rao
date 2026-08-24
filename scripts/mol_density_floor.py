"""Does the estimator's uniform-window assumption explain the last ~0.003?

The smoothing floor of `MOLECULAR_RESULTS.md` section 20 was computed for a
UNIFORM sampling density.  The estimator is a Nadaraya-Watson ratio,

    f_hat(z)  =  (K * rho f)(z) / (K * rho)(z),

whose O(b^2) bias is

    (b^2/2) [ f''  +  2 f' rho'/rho ],

and only the first term survives when rho is flat.  The second term is real
whenever the windows are not evenly spread -- and they never are, because the
deposit carries the Fixman weight (det G)^{-1/2}, which varies along z even for
windows placed on a perfect grid.  RC-WFR's windows are equidistributed by
transport rather than placed, so its rho differs again.

This measures each arm's ACTUAL rho, recomputes the floor with it, and asks
whether the difference is the size of the residual left in section 22.
"""
from __future__ import annotations

import argparse, json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from rcwfr.estimators import smoothing_floor
from rcwfr.mol import systems as S
from rcwfr.mol.refdata import load_reference


def floor_with_density(g, F_ref, bw, rho, mask):
    """The estimator's infinite-sample error at this bandwidth AND this density."""
    return smoothing_floor(g, F_ref, bw, mask, rho=rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="PEN")
    ap.add_argument("--tag", default="dens")
    ap.add_argument("--arms", default="ti_warm,wfr_lmh,ti_cold")
    ap.add_argument("--bw", type=float, default=0.02)
    ap.add_argument("--bws", default="0.04,0.02,0.01")
    ap.add_argument("--ngrid", type=int, default=257)
    ap.add_argument("--h", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--run", action="store_true", help="measure rho first")
    ap.add_argument("--out", default="results/mol/floor2/density_floor.json")
    a = ap.parse_args()
    arms = a.arms.split(",")
    bws = [float(x) for x in a.bws.split(",")]
    cam = "results/mol/campaign"

    if a.run:
        from mol_campaign import run_one
        for arm in arms:
            kw = dict(system=a.system, arm=arm, seeds=a.seeds, N=a.N,
                      steps=a.steps, h=a.h, ngrid=a.ngrid, bw_mf=a.bw,
                      save_every=a.steps, tag=a.tag)
            if arm == "wfr_lmh":
                kw.update(kappa="0.6", theta="0.3", decay="0.999", lift_bw_z=0.25)
            run_one(**kw)

    dev, dt = torch.device("cuda"), torch.float64
    sy = S.REGISTRY[a.system](dev, dt, n_grid=a.ngrid)
    g = sy.grid
    mask = g.eval_mask(dev, dt)
    ref = load_reference(f"results/mol/ref/{a.system}_ref.npz", g, g, dev, dt,
                         cv_shift=sy.cv_shift)
    Fr = ref["F_ref"]

    print(f"# {a.system}: the estimator floor with each arm's OWN sampling density\n")
    print("| arm | " + " | ".join(f"b_mf={b:g}" for b in bws)
          + " | uniform-rho floor |")
    print("|---|" + "---|" * (len(bws) + 1))
    out = {"bws": bws, "uniform": {}, "actual": {}, "extra": {}}
    for b in bws:
        out["uniform"][f"{b}"] = float(smoothing_floor(g, Fr, b, mask)[0])
    for arm in arms:
        p = os.path.join(cam, f"{a.system}_{arm}_{a.tag}.npz")
        if not os.path.exists(p):
            print(f"| {arm} | (no density archive -- rerun with --run) |")
            continue
        d = np.load(p)
        if d["dens"].size == 0:
            print(f"| {arm} | (archive predates the density field) |")
            continue
        rho = torch.as_tensor(d["dens"], device=dev, dtype=dt)
        rho = rho.mean(0, keepdim=True)
        cells, rec = [], {}
        for b in bws:
            v = float(floor_with_density(g, Fr, b, rho, mask)[0])
            cells.append(f"{v:.5f}")
            rec[f"{b}"] = v
        out["actual"][arm] = rec
        out["extra"][arm] = {k: math.sqrt(max(rec[k] ** 2
                                              - out["uniform"][k] ** 2, 0.0))
                             for k in rec}
        print(f"| {arm} | " + " | ".join(cells) + " | "
              + " / ".join(f"{out['uniform'][f'{b}']:.5f}" for b in bws) + " |")

    if out["extra"]:
        print("\nextra error from the density gradient alone "
              "(quadrature difference):\n")
        print("| arm | " + " | ".join(f"b_mf={b:g}" for b in bws) + " |")
        print("|---|" + "---|" * len(bws))
        for arm, rec in out["extra"].items():
            print(f"| {arm} | " + " | ".join(f"{rec[f'{b}']:.5f}" for b in bws) + " |")
        print("\nratio between successive bandwidths (b^2 scaling predicts 4):")
        for arm, rec in out["extra"].items():
            r = [rec[f"{bws[i]}"] / max(rec[f"{bws[i+1]}"], 1e-12)
                 for i in range(len(bws) - 1)]
            print(f"  {arm:9s} " + "  ".join(f"{v:.2f}" for v in r))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
