"""M7: the rigid-measure route to F, and what it costs in variance.

Chapter 3 offers two ways to reach the standard free energy on a nonlinear CV.

  (A) target nu^xi directly, using V^xi = V + (1/2 beta) log det G, and estimate
      the mean force with the LRS formula -- which needs second derivatives of xi.

  (B) sample the RIGID measure  nu_Sigma(dq) propto e^{-beta V} sigma_Sigma(dq)
      with plain SHAKE/RATTLE, and correct statistically:

          F(z) = F_rgd(z) - beta^{-1} log E_{nu_Sigma(z)}[ (det G)^{-1/2} ].

(B) is far more attractive for an atomistic implementation: no Hessians of the
collective variable, and the constrained-dynamics module is the standard one.
The identity is exact, so the only question is the VARIANCE of the reweighting --
if E[(det G)^{-1/2}] needs a huge sample to resolve, route (B) is a false economy.

Measured here: the relative standard error of the correction, its effective
sample size, and the resulting error in F against the estimator floor, as a
function of sample count and of how nonlinear the coordinate is.
"""
import sys, os, json, math, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from rcwfr.grid import DEVICE, DTYPE
from rcwfr.systems.graph import build_graph

FLOOR = 0.004
rows = []
print(f"{'a':>5}{'k':>5}{'a*k':>6}{'|F-Frgd|max':>13}{'ESS frac':>10}"
      f"{'rel.SE @1e3':>12}{'rel.SE @1e5':>12}{'n for 0.1x floor':>18}")
print("-" * 81)
for a, k in ((0.3, 1.4), (0.6, 1.4), (0.6, 2.8), (1.0, 2.8)):
    s = build_graph("EB", a=a, k=k)
    beta, y = s.p.beta, s._yq
    G = s.cv.G(y)
    w = G ** -0.5                                  # (det G)^{-1/2}, the reweight
    mask = s.grid.eval_mask(s.device, s.dtype)
    zq = s.grid.x(s.device, s.dtype)

    # rigid conditional at each z:  nu_Sigma propto e^{-beta Psi} sqrt(G)
    e = s._pdf * torch.sqrt(G).unsqueeze(0)
    e = e / torch.trapezoid(e, dx=s._dy, dim=1, ).unsqueeze(1)
    m1 = torch.trapezoid(e * w.unsqueeze(0), dx=s._dy, dim=1)
    m2 = torch.trapezoid(e * w.unsqueeze(0) ** 2, dx=s._dy, dim=1)
    var = torch.clamp(m2 - m1 * m1, min=0.0)
    rel_var = var / (m1 * m1)                      # per-sample relative variance
    ess_frac = float((1.0 / (1.0 + rel_var))[mask].min())

    # the correction enters F as -beta^{-1} log m1, so its SE is
    #   sd(log m1_hat) ~ sqrt(rel_var / n),  and the F error is that over beta
    def f_err(n):
        return float((torch.sqrt(rel_var / n) / beta)[mask].max())

    gap = float((s.F_ref - s.F_rgd_ref)[0, mask].abs().max())
    target = 0.1 * FLOOR
    n_need = float((rel_var[mask].max() / (beta * target) ** 2))
    rows.append(dict(a=a, k=k, ak=a * k, gap=gap, ess_frac=ess_frac,
                     se_1e3=f_err(1e3), se_1e5=f_err(1e5), n_for_tenth_floor=n_need))
    print(f"{a:5.2f}{k:5.2f}{a*k:6.2f}{gap:13.5f}{ess_frac:10.4f}"
          f"{f_err(1e3):12.6f}{f_err(1e5):12.6f}{n_need:18.0f}")
    del s
    torch.cuda.empty_cache()

os.makedirs("results/manifold", exist_ok=True)
json.dump(dict(floor=FLOOR, rows=rows), open("results/manifold/rigid_route.json", "w"),
          indent=1)
print(f"\nestimator floor {FLOOR}; 'n' is per z-bin, and a production run deposits")
print("O(1e5-1e6) samples per bin, so anything below ~1e4 is free.")
