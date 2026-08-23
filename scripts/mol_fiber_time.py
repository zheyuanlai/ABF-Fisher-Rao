"""How long does the hidden fiber mode take to relax at fixed z?

tau_y sets the entire campaign's budget scale: a run shorter than tau_y has an
essentially frozen fiber, so the LIFT is the only thing that can put the
conditional in the right place, and a run much longer than tau_y does not need
a lift at all.  Every claim about lifts has to say where it sits relative to
this number.

Measured by pinning phi1 = z, starting every replica in the trans basin of
phi2, and watching P(phi2 gauche) approach its reference value.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.mol import systems as S
from rcwfr.mol.dynamics import constrained_step
from rcwfr.mol.ff import _wrap as _wrapd
from rcwfr.mol.geom import TorsionCV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="PEN")
    ap.add_argument("--y0", type=float, default=0.0)
    ap.add_argument("--thr", type=float, default=50.0)
    ap.add_argument("--B", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--every", type=int, default=5_000)
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = f"results/mol/{a.system}_fiber_time.npz"
    dev, dt = torch.device("cuda"), torch.float64
    sy = S.REGISTRY[a.system](dev, dt)
    top, cv, beta, h = sy.top, sy.cv, sy.beta, sy.h
    full = TorsionCV(top.tor_idx, top.mass, shift=cv.shift)
    torch.manual_seed(3)
    z = torch.full((a.B, 1), a.z, device=dev, dtype=dt)
    phis = torch.full((a.B, top.tor_idx.shape[0]), a.y0, device=dev, dtype=dt)
    phis[:, 0] = a.z
    q = sy.ideal(phis)
    step = torch.compile(lambda q, z: constrained_step(top, cv, q, z, h, beta,
                                                       n_newton=6,
                                                       drift_cap=sy.drift_cap),
                         dynamic=False)
    yv = torch.compile(lambda q: full.value(q), dynamic=False)
    ts, ps = [], []
    thr = a.thr * math.pi / 180.0
    nfib = top.tor_idx.shape[0] - 1
    for it in range(a.steps + 1):
        if it % a.every == 0:
            ts.append(it)
            yy = yv(q)
            ps.append([float((_wrapd(yy[:, k + 1] - a.y0).abs() > thr).to(dt).mean())
                       for k in range(nfib)])
        q = step(q, z)
    ts, ps = np.array(ts), np.array(ps)          # ps: (n_t, n_fib)
    peq = ps[-1]
    # exponential relaxation time per fiber torsion, from the last-value normalisation
    m = (ts > 0) & (ts < 0.75 * a.steps)
    tau = np.array([-1.0 / np.polyfit(
        ts[m], np.log(np.clip(1.0 - ps[m, k] / max(peq[k], 1e-9), 1e-6, None)), 1)[0]
        for k in range(ps.shape[1])])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, t=ts, p=ps, tau=tau, p_eq=peq, z=a.z, h=h,
                        system=a.system)
    print(json.dumps({"system": a.system, "tau_steps": tau.tolist(),
                      "p_final": peq.tolist(), "z": a.z}, indent=1))


if __name__ == "__main__":
    main()
