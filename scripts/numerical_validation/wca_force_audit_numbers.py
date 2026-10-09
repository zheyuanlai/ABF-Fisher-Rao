#!/usr/bin/env python
"""Numbers for docs/numerical_validation/WCA_FORCE_AUDIT.md -> results/numerical_validation/wca/force_audit.json."""
import json, math, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))
import wca_validation as V
import test_wca_force_audit as T

out = {}
cfgs = {}
for z, s in ((0.0, 11), (0.25, 12), (0.5, 13), (1.0, 14)):
    cfgs[f"restrained z={z}"] = T.restrained_config(z, s)
cfgs["free (MC) a"] = V.chain("MC", V.lattice_q0(21), n_steps=3000, seed=21, mc_delta=0.4)["q_final"]
cfgs["free (MC) b"] = V.chain("MC", V.lattice_q0(22), n_steps=3000, seed=22, mc_delta=0.4)["q_final"]
PH, PH0 = T.PH, T.PH0
fd = {}
for name, q in cfgs.items():
    U, F, m2 = T.ef(q)
    r = T.pair_distances(q)
    row = dict(U=U, min_pair_r=float(np.sqrt(m2)), max_abs_F=float(np.abs(F).max()), z=float(V.dimer_geom(q, PH[0], PH[6], PH[4])[3]),
               newton_sum=float(np.abs(F.sum(0)).max()))
    for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        errs = []
        for p in range(q.shape[0]):
            if np.min(np.abs(r[p] - T.RC)) < 10 * eps:
                continue
            for k in range(2):
                qp = q.copy(); qp[p, k] += eps
                qm = q.copy(); qm[p, k] -= eps
                d = -(V.energy_only(qp, PH) - V.energy_only(qm, PH)) / (2 * eps)
                errs.append(abs(F[p, k] - d) / max(abs(F[p, k]), 1.0))
        row[f"fd_max_rel_err_eps{eps:g}"] = float(max(errs))
    U0, F0, _ = T.ef(q, PH0)
    row["prod_force_maxdiff_fma"] = float(np.abs(V.production_force(q, fma=True) - F0).max())
    row["prod_force_maxdiff_plain"] = float(np.abs(V.production_force(q, fma=False) - F0).max())
    Ut, Ft = V.implemented_energy_torch(q)
    row["torch_force_maxdiff"] = float(np.abs(Ft - F0).max()); row["torch_energy_diff"] = float(Ut - U0)
    fd[name] = row
out["configurations"] = fd
# min_r clamp
rmin = T.RMIN; x = 1 / rmin
dV = 4 * (-12 * x ** 12 / rmin + 6 * x ** 6 / rmin)
out["min_r_clamp"] = dict(min_r=rmin, V_at_min_r=4 * (x ** 12 - x ** 6) + 1, dVdr_at_min_r=dV,
                          force_at_r060=-(dV / rmin) * 0.60, implemented_energy_gradient_below_min_r=0.0,
                          note="force below min_r = -grad of U_q(r) = V(rmin) + V'(rmin)(r^2-rmin^2)/(2 rmin); implemented energy flat = V(rmin)")
# single-pair force reaching the clip
from scipy.optimize import brentq
fpair = lambda r: -4 * (-12 * r ** -13 + 6 * r ** -7)
r250 = brentq(lambda r: fpair(r) - 250.0, 0.7, 1.12)
out["force_clip"] = dict(force_clip=250.0, single_pair_r_at_clip=r250, V_at_that_r=4 * (r250 ** -12 - r250 ** -6) + 1,
                         dt_times_clip={dt: dt * 250 for dt in (0.002, 0.001, 0.0005, 0.00025, 0.000125)},
                         noise_sd_per_step={dt: math.sqrt(2 * dt) for dt in (0.002, 0.001, 0.0005, 0.00025, 0.000125)})
# EM linear stability: dt * V''(r) for a pair at typical contact distances
V2 = lambda r: 4 * (156 * r ** -14 - 42 * r ** -8)
out["stiffness"] = {f"r={r}": dict(Vpp=V2(r), Vpair=4 * (r ** -12 - r ** -6) + 1, **{f"dt*Vpp(dt={dt:g})": dt * V2(r) for dt in (0.002, 0.001, 0.0005, 0.00025, 0.000125)})
                    for r in (0.90, 0.95, 1.0, 1.05, 1.1)}
# clip asymmetry for the constructed 3-particle case
q = V.lattice_q0(6, jitter=0.0); c = 22
q[32] = (q[c] + np.array([0.86, 0.0])) % T.L; q[23] = (q[c] + np.array([0.0, 0.84])) % T.L
Fc = V.production_clipped_force(q); Fr = V.production_force(q)
h = 1e-6
def J(i, a, k, b, field):
    qp = q.copy(); qp[k, b] += h; qm = q.copy(); qm[k, b] -= h
    return (field(qp)[i, a] - field(qm)[i, a]) / (2 * h)
out["clip_counterexample"] = dict(raw_norm_center=float(np.linalg.norm(Fr[c])), clipped_sum=Fc.sum(0).tolist(), raw_sum=Fr.sum(0).tolist(),
                                  J_raw=[J(c, 0, 23, 1, V.production_force), J(23, 1, c, 0, V.production_force)],
                                  J_clipped=[J(c, 0, 23, 1, V.production_clipped_force), J(23, 1, c, 0, V.production_clipped_force)])
json.dump(out, open(os.path.join(ROOT, "results", "numerical_validation", "wca", "force_audit.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:6000])
