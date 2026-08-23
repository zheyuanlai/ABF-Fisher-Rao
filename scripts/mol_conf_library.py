"""A library of exact conditional configurations, bucketed by z.

The lift arms all approximate `nu^xi(. | z')` in one way or another.  This is
the object they approximate: configurations drawn from the UNBIASED Boltzmann
measure and conditioned on `xi(q) = z` by binning.  An arm that replaces a
walker's whole configuration with a fresh draw from this library has, by
construction, no lift error at all -- and whatever error it still shows is the
estimator plus the z-marginal, i.e. the ceiling every other lift is measured
against.

Filling is a ring buffer per z-bin, so the rarely visited bins (the cis barrier,
1.6e-5 of the mass) accumulate over the whole run instead of being crowded out.
"""
from __future__ import annotations

import argparse, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.mol import systems as S
from rcwfr.mol.dynamics import free_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="PEN")
    ap.add_argument("--B", type=int, default=65536)
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--burn", type=int, default=60_000)
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--nz", type=int, default=90)
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="results/mol/ref")
    a = ap.parse_args()
    dev, dt = torch.device("cuda"), torch.float64
    torch.manual_seed(a.seed)
    sy = S.REGISTRY[a.system](dev, dt)
    top, cv, beta, h = sy.top, sy.cv, sy.beta, sy.h
    A = top.n_atoms
    phis = (torch.rand(a.B, top.tor_idx.shape[0], device=dev, dtype=dt) * 2 - 1) * math.pi
    q = sy.ideal(phis)
    step = torch.compile(lambda q: free_step(top, q, h, beta), dynamic=False)
    zfun = torch.compile(lambda q: cv.value(q)[..., 0], dynamic=False)

    g = sy.grid                      # bins span the CAMPAIGN grid, not [-pi, pi):
    lo, span = g.xmin, g.volume      # a restricted-arc CV has no weight outside it
    lib = torch.zeros((a.nz, a.M, A, 3), device=dev, dtype=dt)
    ptr = torch.zeros(a.nz, device=dev, dtype=torch.long)
    t0 = time.time()
    for it in range(a.steps):
        q = step(q)
        if it < a.burn or it % a.every:
            continue
        z = zfun(q)
        b = torch.clamp(((z - lo) / span * a.nz).long(), 0, a.nz - 1)
        order = torch.argsort(b)
        bs = b[order]
        cnt = torch.bincount(bs, minlength=a.nz)
        off = torch.cat([torch.zeros(1, device=dev, dtype=torch.long),
                         torch.cumsum(cnt, 0)[:-1]])
        rank = torch.arange(a.B, device=dev) - off[bs]
        slot = (ptr[bs] + rank) % a.M
        lib[bs, slot] = q[order]
        ptr = ptr + cnt
        if it % 100_000 == 0 and it:
            print(f"  {it}/{a.steps} {time.time()-t0:.0f}s min fill "
                  f"{int(torch.clamp(ptr, max=a.M).min())}/{a.M}", flush=True)
    fill = torch.clamp(ptr, max=a.M)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"{a.system}_conflib.npz")
    np.savez_compressed(p, lib=lib.cpu().numpy().astype(np.float32),
                        fill=fill.cpu().numpy(), nz=a.nz, M=a.M,
                        lo=float(lo), span=float(span))
    print(f"done {time.time()-t0:.0f}s  min fill {int(fill.min())}  -> {p}", flush=True)


if __name__ == "__main__":
    main()
