#!/usr/bin/env python
"""Freeze the references of the equal-budget ladders (independent of every ladder run).
LTA T: exact Metropolis-MC Gibbs reference of the 2026-10-09 validation (16 groups x 40 umbrella windows):
  F_ref     fine-bin (1800) WHAM summed to the 180 production bins, -kT log p, centred (kJ/mol)
  F_ref_se  leave-one-group-out jackknife
  gamma_ref conditional mean force per bin, sum Mf / sum C over all windows (kJ/mol/rad), gamma_se jackknife
  F_ref_mf  histogram_pmf(gamma_ref) (the mean-force route), for consistency
Gateway: exact analytic F, F' on the accepted scorer grid and the EM-consistent profile at h_G.
-> results/equal_budget_v2/references/{lta_T300,lta_T150,gateway}_reference.npz (+ .json summary)"""
import glob, json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts", "numerical_validation"))
OUT = os.path.join(ROOT, "results", "equal_budget_v2", "references")
KB = 0.008314462618


def lta(T):
    import analyze_lta_validation as A
    D = A.load(float(T), "MC")
    assert D["n_groups"] == 16 and D["n_files"] == 640, (D["n_groups"], D["n_files"])
    kT = KB * T
    pr = A.Prof("MC density", D, "density", kT, A.KAPPA[float(T)])
    G = D["n_groups"]
    def gam(idx):
        M = D["Mf"][idx].sum((0, 1)); C = D["C"][idx].sum((0, 1)); return M / C
    g = gam(np.arange(G))
    loo = np.array([gam(np.delete(np.arange(G), i)) for i in range(G)])
    gse = np.sqrt((G - 1) / G * ((loo - loo.mean(0)) ** 2).sum(0))
    Fmf = A.hist_pmf(g)
    out = dict(T_K=float(T), kT=kT, grid_phi=A.CEN, F_ref=pr.F, F_ref_se=pr.se, gamma_ref=g, gamma_se=gse, F_ref_mf=Fmf,
               units_F="kJ/mol", units_gamma="kJ/mol/rad", gauge="F centred (mean over the 180 bins = 0); compare after removing the mean difference",
               source=f"results/numerical_validation/lta/T{T}/MC (640 files: 16 groups x 40 windows)")
    np.savez(os.path.join(OUT, f"lta_T{T}_reference.npz"), **{k: np.asarray(v) for k, v in out.items()})
    summ = dict(T_K=T, F_noise_rms=float(np.sqrt(np.mean(pr.se ** 2))), gamma_se_rms=float(np.sqrt(np.mean(gse ** 2))),
                gamma_circular_mean=float(g.mean()), rms_F_mf_minus_F_density=float(np.sqrt(np.mean(((Fmf - Fmf.mean()) - (pr.F - pr.F.mean())) ** 2))),
                barrier_window_minus_cage=float(pr.F[np.abs(A.CEN) * 11.919 / (2 * np.pi) < 0.4].mean() - pr.F[np.abs(A.CEN) * 11.919 / (2 * np.pi) > 4].mean()))
    print(T, json.dumps(summ))
    return summ


def gateway():
    import torch
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import analyze_gateway_replica_ladder as GL
    cell = dict(beta=16.0, H=0.5, omega_out=1.0, omega_in=32.0, s=0.1, dt=2.5e-5)
    sa = GL.Scorer(cell, "analytic"); se = GL.Scorer(cell, "em")
    out = dict(x_grid=sa.x_grid.numpy(), eval_mask=sa.eval_mask.numpy(), F_ref=sa.F_ref.numpy()[0], F_ref_em=se.F_ref.numpy()[0],
               Fp_ref=sa.Fp_ref.numpy()[0], h=2.5e-5, units="model energy units (beta = 16)",
               gauge="F centred on the eval window [-1.5, 1.5] (151 grid nodes); errors via analyze_gateway_replica_ladder.Scorer")
    np.savez(os.path.join(OUT, "gateway_reference.npz"), **{k: np.asarray(v) for k, v in out.items()})
    m = sa.eval_mask.numpy()
    d = (se.F_ref.numpy()[0] - sa.F_ref.numpy()[0])[m]
    s = dict(em_floor_rms=float(np.sqrt(np.mean((d - d.mean()) ** 2))))
    print("gateway", json.dumps(s))
    return s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    S = {"lta_T300": lta(300), "lta_T150": lta(150), "gateway": gateway()}
    json.dump(S, open(os.path.join(OUT, "references_summary.json"), "w"), indent=1)
