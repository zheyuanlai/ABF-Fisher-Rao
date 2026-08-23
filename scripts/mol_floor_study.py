"""Where does the ~0.020 kcal/mol plateau come from?

Every constrained arm in this project converges to about the same `e_F` and stops
there, whatever its transport does, so that number is a property of the numerics
rather than of any method.  It decomposes as

    e_F  ~  C_h h^p  +  C_b b_mf^2  +  C_stat N^{-1/2}

and this script separates the three.  Warm stratified constrained TI only -- no
transport, no exploration, no hidden mode -- so nothing about sampling difficulty
is involved.

Two design points matter.

* **Fixed physical time, not fixed step count.**  `n_steps` scales as `1/h`, so
  every `h` sees the same amount of dynamics and the statistical error is held
  roughly constant while the discretisation bias varies.
* **The reference has its own `h`.**  `F_ref` came from unbiased Brownian
  dynamics at `h_0`, so it carries an O(h) bias of its own; shrinking `h` in the
  constrained arm alone would make `e_F` grow, not shrink, if that bias
  dominated.  The script therefore reports the constrained arm's SELF-difference
  `F(h) - F(h_min)`, which needs no reference at all, alongside `e_F`.

Several bandwidths are accumulated from ONE trajectory per `h`, since the
bandwidth changes only the estimator and not the dynamics.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.estimators import MeanForceAccumulator, gauge_l2
from rcwfr.grid import Grid1D
from rcwfr.mol import systems as S
from rcwfr.mol.dynamics import constrained_step
from rcwfr.mol.engines import _invcdf
from rcwfr.mol.lift import ReferenceFiberCDF
from rcwfr.mol.refdata import load_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="BUT")
    ap.add_argument("--hs", default="2e-3,1e-3,5e-4,2.5e-4")
    ap.add_argument("--bws", default="0.08,0.04,0.02")
    ap.add_argument("--ngrid", type=int, default=257)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--time", type=float, default=400.0,
                    help="physical time = n_steps * h, held fixed across h")
    ap.add_argument("--eq-frac", type=float, default=0.1)
    ap.add_argument("--dep-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--warm", action="store_true",
                    help="draw the fiber torsions from the reference conditional; "
                         "required whenever the fiber holds a SLOW mode, since a "
                         "delta start would otherwise be relaxing, not equilibrated")
    ap.add_argument("--out", default="results/mol/floor")
    a = ap.parse_args()
    hs = [float(x) for x in a.hs.split(",")]
    bws = [float(x) for x in a.bws.split(",")]
    dev, dt = torch.device("cuda"), torch.float64
    g = Grid1D(-math.pi, math.pi, a.ngrid, -math.pi, math.pi, "periodic")
    refp = a.ref or f"results/mol/ref/{a.system}_ref.npz"
    ref = load_reference(refp, g, g, dev, dt)
    mask = g.eval_mask(dev, dt)
    out = {"h": hs, "bw": bws, "F": np.zeros((len(hs), len(bws), a.rows, a.ngrid)),
           "e_F": np.zeros((len(hs), len(bws), a.rows)), "wall": []}
    for ih, h in enumerate(hs):
        sy = S.REGISTRY[a.system](dev, dt, h=h, n_grid=a.ngrid)
        top, cv, beta = sy.top, sy.cv, sy.beta
        nt = top.tor_idx.shape[0]
        n_steps = int(round(a.time / h))
        n_eq = int(a.eq_frac * n_steps)
        # deposit at a fixed PHYSICAL interval too, so the number of deposits and
        # hence the statistical error are held constant while h varies
        dep = max(1, int(round(a.dep_every * hs[0] / h)))
        torch.manual_seed(a.seed + ih)
        zs = torch.linspace(g.xmin, g.xmax, a.N + 1, device=dev, dtype=dt)[:a.N]
        z = zs.unsqueeze(0).expand(a.rows, a.N).contiguous()
        phis = torch.full((a.rows, a.N, nt), sy.y0, device=dev, dtype=dt)
        phis[..., 0] = z
        if a.warm and nt > 1:
            tab = ReferenceFiberCDF(a.rows, ref["gz"], sy.y_grid or g, dev, dt,
                                    ref["H2"])
            u = torch.rand((a.rows, a.N), device=dev, dtype=dt)
            phis[..., 1] = _invcdf(tab, z, u)
        q = sy.ideal(phis.reshape(-1, nt)).reshape(a.rows, a.N, top.n_atoms, 3)
        z = z.unsqueeze(-1)
        step = torch.compile(lambda q, z: constrained_step(top, cv, q, z, h, beta,
                                                           n_newton=6,
                                                           drift_cap=sy.drift_cap),
                             dynamic=False)
        gradV = torch.compile(lambda q: top.grad(q), dynamic=False)
        accs = [MeanForceAccumulator(a.rows, g, b, 1.0, dev, dt) for b in bws]
        t0 = time.time()
        for n in range(n_steps):
            q = step(q, z)
            if n >= n_eq and (n + 1) % dep == 0:
                f, G = cv.mean_force(q, gradV(q), beta)
                w = G[..., 0, 0] ** -0.5
                for acc in accs:
                    acc.deposit(z[..., 0], f[..., 0], weights=w)
        wall = time.time() - t0
        out["wall"].append(wall)
        for ib, acc in enumerate(accs):
            F = acc.free_energy(mask)
            out["F"][ih, ib] = F.cpu().numpy()
            out["e_F"][ih, ib] = gauge_l2(F, ref["F_ref"], mask).cpu().numpy()
        print(f"h={h:.2e} steps={n_steps} dep_every={dep} wall={wall:.0f}s  "
              + "  ".join(f"bw={b:.3f}:e_F={np.median(out['e_F'][ih, ib]):.5f}"
                          for ib, b in enumerate(bws)), flush=True)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"{a.system}_floor_n{a.ngrid}.npz")
    np.savez_compressed(p, F_ref=ref["F_ref"].cpu().numpy(),
                        ngrid=a.ngrid, N=a.N, rows=a.rows, time=a.time, **out)
    print("wrote", p)


if __name__ == "__main__":
    main()
