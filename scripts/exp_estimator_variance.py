"""E6: two valid mean-force estimators, and the free variance win between them.

Any function with conditional mean F'(z) is a valid mean-force sample.  Two are
available once the reaction coordinate is nonlinear:

    f_LRS   = (grad xi . grad V)/G - beta^-1 div(grad xi / G)      (Chapter 3)
    f_graph = d Psi / dz                                            (fiber frame)

They differ by a function of conditional mean zero, so any convex combination is
also valid -- and their variances differ sharply and in opposite directions
across z, so the optimal combination beats both.  Reported as the variance ratio
against the better of the two, which is the honest baseline.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from rcwfr.grid import DEVICE, DTYPE
from rcwfr.systems.graph import build_graph

M = 2_000_000
rows = []
for name in ("EB", "CHANNEL"):
    for a, k in ((0.3, 1.4), (0.6, 1.4), (0.6, 2.8)):
        s = build_graph(name, a=a, k=k)
        gen = torch.Generator(device=DEVICE); gen.manual_seed(21)
        for z0 in (-1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2):
            z = torch.full((M,), z0, device=DEVICE, dtype=DTYPE)
            y = s.sample_fiber(z, gen)
            q = torch.stack([z - s.cv.s(y), y], -1)
            fa = s.cv.mean_force(q, s.grad_V_ambient(q), s.p.beta)
            fb = s.mean_force_z(z, y)
            va, vb = float(fa.var()), float(fb.var())
            cab = float(((fa - fa.mean()) * (fb - fb.mean())).mean())
            # d = f_LRS - f_graph has conditional mean ZERO, so it is an exact
            # control variate and lambda is unconstrained: f_lam = f_graph + lam d
            # is unbiased for every real lambda.  (For a LINEAR xi the two
            # estimators coincide, d == 0, and this control variate does not exist.)
            den = va + vb - 2 * cab
            lam = float((vb - cab) / den) if abs(den) > 1e-14 else 0.5
            vmix = max(lam * lam * va + (1 - lam) ** 2 * vb
                       + 2 * lam * (1 - lam) * cab, 1e-30)
            corr = cab / max((va * vb) ** 0.5, 1e-30)
            rows.append(dict(system=name, a=a, k=k, z=z0, corr=corr, var_lrs=va, var_graph=vb,
                             cov=cab, lam=lam, var_mix=vmix,
                             gain_vs_best=min(va, vb) / vmix,
                             ratio_lrs_graph=va / vb))
        del s; torch.cuda.empty_cache()

os.makedirs("results/manifold", exist_ok=True)
json.dump(rows, open("results/manifold/estimator_variance.json", "w"), indent=1)

print(f"{'sys':8s}{'a':>5s}{'k':>5s}{'z':>6s}{'Var f_LRS':>11s}{'Var f_gr':>10s}"
      f"{'lam*':>8s}{'gain vs best':>13s}")
for r in rows:
    if r["a"] == 0.6 and r["k"] == 1.4:
        print(f"{r['system']:8s}{r['a']:5.2f}{r['k']:5.2f}{r['z']:6.2f}"
              f"{r['var_lrs']:11.4f}{r['var_graph']:10.4f}{r['lam']:8.3f}"
              f"{r['gain_vs_best']:13.3f}")
g = [r["gain_vs_best"] for r in rows]
print(f"\nvariance gain over the better single estimator: median {np.median(g):.3f}, "
      f"max {max(g):.3f}")
rr = [r["ratio_lrs_graph"] for r in rows]
print(f"Var(f_LRS)/Var(f_graph) ranges over {min(rr):.3f} .. {max(rr):.1f} "
      f"-- neither estimator dominates")
cc = [r["corr"] for r in rows]
print(f"correlation between the two estimators: median {np.median(cc):.5f}, "
      f"max {max(cc):.6f}")
print("The gain is 1/(1-corr^2)-like, so it is entirely a statement about how")
print("nearly AFFINE the two estimators are ON THE FIBER.  Measured here on a")
print("ONE-dimensional fiber with the spectator block integrated analytically;")
print("it must be re-measured, not extrapolated, on a many-dimensional fiber.")
