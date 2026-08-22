"""M0b: what does dropping the divergence term of the local mean force cost?

    f = (grad xi . grad V)/G  -  beta^{-1} div(grad xi / G)

The second term is identically zero when xi is linear, so the frozen campaign never
exercised it.  Here the exact conditional means are pushed through the SAME
thermodynamic integration the estimator uses, so the answer comes out in FREE-ENERGY
units and can be quoted against the estimator floor -- quoting a mean-force error
against a free-energy floor compares different units and overstates the effect.

Keeping the term is the control: it should return an error below the floor (pure
quadrature error), which also validates the implementation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import json, torch
from rcwfr.estimators import gauge_l2
from rcwfr.grid import cumtrapz
from rcwfr.systems.graph import build_mfib, build_graph

out = {}
for tag, s in (("MFIB a=0.6 k=1.4", build_mfib(omega=1.0, a=0.6, k=1.4)),
               ("EB   a=0.6 k=1.4", build_graph("EB", a=0.6, k=1.4)),
               ("EB   a=1.0 k=2.8", build_graph("EB", a=1.0, k=2.8))):
    cv, beta = s.cv, s.p.beta
    zq = s.grid.x(s.device, s.dtype)
    y = s._yq
    Z1, Y1 = zq.unsqueeze(1), y.unsqueeze(0)
    nu = s._pdf                                   # (G, n_yq), exact conditional
    x = Z1 - cv.s(Y1)
    q = torch.stack([x, Y1.expand_as(x)], -1)
    gV = s.grad_V_ambient(q)
    c = cv.c(Y1); G = 1.0 + c * c
    f_full = cv.mean_force(q, gV, beta)
    f_nodiv = (gV[..., 0] + c * gV[..., 1]) / G
    dF_full = torch.trapezoid(nu * f_full, dx=s._dy, dim=1)
    dF_nodiv = torch.trapezoid(nu * f_nodiv, dx=s._dy, dim=1)
    mask = s.grid.eval_mask(s.device, s.dtype)
    F_full = cumtrapz(dF_full.unsqueeze(0), s.grid.dx)
    F_nodiv = cumtrapz(dF_nodiv.unsqueeze(0), s.grid.dx)
    e_full = float(gauge_l2(F_full, s.F_ref, mask))
    e_nodiv = float(gauge_l2(F_nodiv, s.F_ref, mask))
    dfe = float((dF_nodiv - s.dF_ref[0])[mask].abs().max())
    out[tag] = dict(max_abs_dF_err=dfe, L2_F_err_nodiv=e_nodiv,
                    L2_F_err_full=e_full, floor=0.004,
                    floor_multiple=e_nodiv / 0.004)
    print(f"{tag}: max |dF' err| = {dfe:.5f}   L2 F error = {e_nodiv:.5f} "
          f"({e_nodiv/0.004:.1f}x floor)   [control, full formula: {e_full:.2e}]",
          flush=True)
    del s; torch.cuda.empty_cache()
json.dump(out, open("results/manifold/nodiv.json","w"), indent=1)
