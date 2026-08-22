"""Which fiber block actually poses a lift problem?

E11 found that lifting the spectators naively costs nothing.  Two explanations are
possible and they have different consequences: either the spectators relax fast
(the design rule, confirmed), or their conditional barely changes along z (the
design rule, untested).  This measures which, by computing each block's FROZEN
lift-lag coefficient -- the D_cond a naive lift produces per unit dz with no
relaxation at all, so speed cannot enter.

  y_1 block:  C_y = integral [ d_y(nu delta) ]^2 / nu dy      (as in the audit)
  S block:    the conditional is N(0, sigma(z,y_1)^2 I) and a naive lift leaves it
              at the old width, so per spectator
                  KL = log r + 1/(2 r^2) - 1/2,  r = sigma_new / sigma_old,
              giving  C_S = 2 m ( d log sigma / dz )^2  to leading order.

If C_S << C_y the spectators were never a test of anything.
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from rcwfr.grid import DEVICE, DTYPE
from rcwfr.systems.graph import build_graph_nd, lag_coefficients


def block_lags(s, z0):
    """(C_y1 for the cartesian lift, C_S for the naive spectator lift)."""
    C_y = lag_coefficients(s, z0, "cartesian")["C"]
    y = s._yq
    z = torch.full_like(y, z0)
    i0, fz = s._z_index(torch.tensor([z0], device=DEVICE, dtype=DTYPE))
    nu = (s._pdf[i0] * (1 - fz.unsqueeze(-1))
          + s._pdf[i0 + 1] * fz.unsqueeze(-1)).squeeze(0)
    x = z - s.cv.s(y)
    # cartesian lift on y_1 => dy/dz = 0 => dx/dz = 1
    dlog_sigma_dz = -(s.domega_s(x) / s.omega_s(x))
    width = 2.0 * torch.trapezoid(nu * dlog_sigma_dz ** 2, x=y)
    # a stale CENTRE costs (d mu/dz)^2 / (2 sigma^2) per spectator, doubled to match
    # the C dz^2 / 2 convention
    dmu = s._dmu_dx(x)
    sd2 = 1.0 / (s.p.beta * s.omega_s(x) ** 2)
    shift = torch.trapezoid(nu * dmu * dmu / sd2, x=y)
    return C_y, float(s.p.m_spec * (width + shift))


def calibrate(m, om, A_list):
    """C_S in units of C_y1, for shift amplitudes expressed in CONDITIONAL WIDTHS
    per unit z.  mu_amp = A / omega_s keeps the shift-to-width ratio fixed, so the
    lift error stays constant while the relaxation time 1/omega_s^2 varies."""
    out = []
    for A in A_list:
        s = build_graph_nd("CHANNEL", a=0.6, k=1.4, m_spec=m, oms_out=om,
                           oms_ratio=4.0, mu_amp=A / om)
        Cy, CS = np.mean([block_lags(s, z) for z in (-0.9, -0.3, 0.3)], axis=0)
        out.append((A, float(Cy), float(CS), float(CS / Cy)))
        del s
        torch.cuda.empty_cache()
    return out


rows = []
print(f"{'m':>4}{'oms_out':>9}{'ratio':>7}{'C_y1':>12}{'C_S':>12}{'C_S/C_y1':>11}"
      f"{'tau_spec':>10}")
print("-" * 65)
for m, om, ratio in itertools.product((4, 16), (0.25, 1.0), (4.0, 12.0)):
    try:
        s = build_graph_nd("CHANNEL", a=0.6, k=1.4, m_spec=m,
                           oms_out=om, oms_ratio=ratio)
    except AssertionError as e:
        print(f"{m:4d}{om:9.2f}{ratio:7.1f}   unbuildable: {str(e)[:34]}")
        continue
    Cy, CS = np.mean([block_lags(s, z) for z in (-0.9, -0.3, 0.3)], axis=0)
    rows.append(dict(m_spec=m, oms_out=om, ratio=ratio, C_y1=float(Cy),
                     C_S=float(CS), frac=float(CS / Cy), tau=1.0 / om ** 2))
    print(f"{m:4d}{om:9.2f}{ratio:7.1f}{Cy:12.2f}{CS:12.4f}{CS/Cy:11.5f}"
          f"{1.0/om**2:10.2f}")
    del s
    torch.cuda.empty_cache()

print()
print("With a SHIFTED spectator block, mu_amp = A / omega_s (A = shift in conditional")
print("widths per unit z), so C_S is held fixed while tau_spec = 1/omega_s^2 varies:")
print(f"  {'A':>6}{'omega_s':>9}{'C_y1':>10}{'C_S':>10}{'C_S/C_y1':>11}{'tau':>8}")
cal = {}
for om in (0.25, 1.0, 4.0):
    for A, Cy, CS, fr in calibrate(4, om, (0.0, 0.25, 0.5, 1.0)):
        print(f"  {A:6.2f}{om:9.2f}{Cy:10.2f}{CS:10.3f}{fr:11.4f}{1/om**2:8.2f}")
        cal[f"A{A}_om{om}"] = dict(A=A, omega_s=om, C_y1=Cy, C_S=CS, frac=fr)
os.makedirs("results/manifold", exist_ok=True)
json.dump(dict(width_only=rows, shifted=cal),
          open("results/manifold/block_lag.json", "w"), indent=1)
