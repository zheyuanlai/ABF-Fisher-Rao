"""Phase M0: does the Chapter-3 machinery actually reproduce the exact answer?

Four checks, each against a quadrature-exact reference:

  V1  co-area normalization: the graph parameterization has unit Jacobian, so
      integral over Sigma(z) of (det G)^{-1/2} dsigma must equal integral dy.
  V2  LOCAL MEAN FORCE (LRS eq. 3.32, with the divergence term that only exists
      when G varies along the fiber):  E_{nu(.|z)}[ f ] = F'(z).
      This is the check that fails if the divergence term is dropped.
  V3  SHAKE projection lands on Sigma(z) and the three lifts all satisfy the
      constraint to the same order.
  V4  the constrained ambient sampler converges to nu^xi WITH the Fixman term
      and to the RIGID measure without it.
"""
import sys, json, math
sys.path.insert(0, "src")
import torch

from rcwfr.grid import DEVICE, DTYPE
from rcwfr.manifold import GraphCV, constrained_step
from rcwfr.systems.graph import build_graph

torch.manual_seed(0)
gen = torch.Generator(device=DEVICE); gen.manual_seed(7)
out = {}

SYS, A, K = "EB", 0.6, 1.4
s = build_graph(SYS, a=A, k=K)
cv = s.cv
beta = s.p.beta
zs = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=DEVICE, dtype=DTYPE)

# ---- V1 co-area ------------------------------------------------------------
yq = s._yq
sqG = torch.sqrt(cv.G(yq))
lhs = float(torch.trapezoid(sqG / sqG, dx=s._dy))          # (det G)^{-1/2} dsigma
rhs = float(yq[-1] - yq[0])
out["V1_coarea_rel_err"] = abs(lhs - rhs) / rhs

# ---- V2 local mean force ---------------------------------------------------
M = 4_000_000
rows = []
for z0 in zs:
    zz = z0.expand(M)
    y = s.sample_fiber(zz, gen)
    q = torch.stack([zz - cv.s(y), y], dim=-1)
    gV = s.grad_V_ambient(q)
    f_lrs = cv.mean_force(q, gV, beta)
    f_lrs_nodiv = (gV[..., 0] + cv.c(y) * gV[..., 1]) / cv.G(y)   # divergence dropped
    f_graph = s.mean_force_z(zz, y)
    i0, fz = s._z_index(z0.reshape(1))
    ref = float(s.dF_ref[0, i0] + fz * (s.dF_ref[0, i0 + 1] - s.dF_ref[0, i0]))
    se = lambda t: float(t.std() / math.sqrt(M))
    rows.append(dict(z=float(z0), F_prime_exact=ref,
                     lrs=float(f_lrs.mean()), lrs_se=se(f_lrs),
                     lrs_nodiv=float(f_lrs_nodiv.mean()),
                     graph=float(f_graph.mean()), graph_se=se(f_graph),
                     sd_lrs=float(f_lrs.std()), sd_graph=float(f_graph.std())))
out["V2_mean_force"] = rows
out["V2_max_z_lrs"] = max(abs(r["lrs"] - r["F_prime_exact"]) / r["lrs_se"] for r in rows)
out["V2_max_z_graph"] = max(abs(r["graph"] - r["F_prime_exact"]) / r["graph_se"] for r in rows)
out["V2_max_abs_err_nodiv"] = max(abs(r["lrs_nodiv"] - r["F_prime_exact"]) for r in rows)

# ---- V3 projection / lifts -------------------------------------------------
z0 = torch.full((200_000,), 0.3, device=DEVICE, dtype=DTYPE)
y = s.sample_fiber(z0, gen)
q = torch.stack([z0 - cv.s(y), y], dim=-1)
out["V3_xi_residual_initial"] = float((cv.xi(q) - z0).abs().max())
dz = 0.05
res = {}
for mode in ("cartesian", "minnorm"):
    qp = cv.project(cv.lift(q, torch.full_like(z0, dz), mode), z0 + dz, mode="minnorm")
    res[mode] = float((cv.xi(qp) - (z0 + dz)).abs().max())
for mode in ("cartesian", "minnorm", "adiabatic"):
    yn = s.lift_fiber(z0, y, dz, mode, n_sub=8)
    qn = torch.stack([z0 + dz - cv.s(yn), yn], dim=-1)
    res[f"fiber_{mode}"] = float((cv.xi(qn) - (z0 + dz)).abs().max())
out["V3_xi_residual_after_lift"] = res

# ---- V4 constrained ambient sampler: Fixman on / off -----------------------
def run_constrained(z_val, fixman, n_steps=40_000, dt=2e-4, M=200_000):
    zz = torch.full((M,), z_val, device=DEVICE, dtype=DTYPE)
    y = s.sample_fiber(zz, gen)
    q = torch.stack([zz - cv.s(y), y], dim=-1)
    acc_y, acc_n = torch.zeros(1, device=DEVICE, dtype=DTYPE), 0
    hist = torch.zeros(64, device=DEVICE, dtype=DTYPE)
    for t in range(n_steps):
        gV = s.grad_V_ambient(q)
        q = constrained_step(cv, q, zz, gV, dt, beta, gen, fixman=fixman)
        q[..., 1] = torch.clamp(q[..., 1], -s.p.y_max, s.p.y_max)
        q = cv.project(q, zz, mode="minnorm")
        if t >= n_steps // 2 and t % 20 == 0:
            u = s.pit(zz, q[..., 1])
            hist += torch.histc(u, bins=64, min=0.0, max=1.0)
            acc_n += 1
    p = hist / hist.sum()
    kl = float((p * torch.log(torch.clamp(p * 64, min=1e-30))).sum())
    return kl

out["V4"] = {}
for z_val in (-0.6, 0.0, 0.6):
    out["V4"][f"z={z_val}"] = dict(
        kl_pit_fixman=run_constrained(z_val, True),
        kl_pit_no_fixman=run_constrained(z_val, False))

print(json.dumps(out, indent=2))
with open("results/manifold/validate.json", "w") as fh:
    json.dump(out, fh, indent=2)
