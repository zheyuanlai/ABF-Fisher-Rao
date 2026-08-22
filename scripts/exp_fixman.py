"""E1: how large is the Fixman / rigid-vs-standard error, and when does it matter?

A constrained sampler that omits (det G)^{-1/2} converges to the RIGID measure and
returns F_rgd, not F.  For a linear reaction coordinate G = 1 and the two coincide,
which is why the existing (xi = x) campaign could never see this.  Here G varies
along the fiber and the gap is exactly computable by quadrature.

Reported against the campaign's measured estimator floor e_F = 0.004 (bw_mf = 0.02),
because an error below the floor is not a real error.
"""
import sys, json, itertools
sys.path.insert(0, "src")
import numpy as np
import torch

from rcwfr.systems.graph import build_graph

FLOOR = 0.004
rows = []
for name in ("EB", "SLOWFIB", "CHANNEL"):
    for a, k in itertools.product((0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0),
                                  (0.7, 1.4, 2.8)):
        s = build_graph(name, a=a, k=k)
        mask = s.grid.eval_mask(s.device, s.dtype)
        d = (s.F_ref - s.F_rgd_ref)[0, mask]
        d = d - d.mean()
        rmse = float(torch.sqrt((d * d).mean()))
        span = float((s.F_ref[0, mask].max() - s.F_ref[0, mask].min()))
        # the Fixman term is (1/2 beta) log G; its spread over the fiber is what
        # drives the gap
        G = s.cv.G(s._yq)
        rows.append(dict(system=name, a=a, k=k, ak=a * k,
                         rmse_F_minus_Frgd=rmse, floor_multiple=rmse / FLOOR,
                         F_span=span, rel_to_span=rmse / span,
                         logG_range=float(torch.log(G).max() - torch.log(G).min())))
        del s
        torch.cuda.empty_cache()

with open("results/manifold/fixman.json", "w") as fh:
    json.dump(dict(floor=FLOOR, rows=rows), fh, indent=2)

hdr = f"{'system':9s} {'a':>5s} {'k':>5s} {'a*k':>6s} {'RMSE(F-Frgd)':>13s} {'/floor':>8s} {'/span':>8s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    if r["a"] in (0.0, 0.3, 0.6, 1.0):
        print(f"{r['system']:9s} {r['a']:5.2f} {r['k']:5.2f} {r['ak']:6.2f} "
              f"{r['rmse_F_minus_Frgd']:13.5f} {r['floor_multiple']:8.1f} "
              f"{r['rel_to_span']:8.4f}")
