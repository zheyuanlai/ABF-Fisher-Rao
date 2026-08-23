"""Gate I: does the constrained Chapter-3 engine reproduce unbiased molecular MD?

Three independent checks, in increasing order of what they can break:

  A  mean-force formula.  F from the unbiased histogram vs F from thermodynamic
     integration of the SAME unbiased samples.  Sampler-free: only grad xi, the
     Gram matrix, the Hessian trace and the divergence term are on trial.
  B  constrained sampler + Fixman reweighting.  Stratified constrained TI (all
     windows, warm) vs the unbiased histogram.
  C  diagnostics: Fixman reweighting ESS, SHAKE residual, and what happens if
     the (det G)^{-1/2} weight is dropped (the F_rgd control).

Failing A or B is an engine bug, not a result about RC-WFR.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.estimators import MeanForceAccumulator, gauge_l2
from rcwfr.grid import cumtrapz
from rcwfr.mol import systems as S
from rcwfr.mol.engines import MolCfg, run_constrained
from rcwfr.mol.refdata import load_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="BUT")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--out", default="results/mol/gate1")
    a = ap.parse_args()
    dev, dt = torch.device("cuda"), torch.float64
    sy = S.REGISTRY[a.system](dev, dt)
    refp = a.ref or f"results/mol/ref/{a.system}_ref.npz"
    ref = load_reference(refp, sy.grid, sy.y_grid or sy.grid, dev, dt)
    mask = sy.grid.eval_mask(dev, dt)
    res = {"system": a.system}

    # ---- A: mean force vs histogram, both from the unbiased trajectories ----
    d = np.load(refp)
    beta = float(d["beta"])
    ctr = d["centers"]
    mf = d["S1"][:, 0].sum(0) / np.maximum(d["S0"][:, 0].sum(0), 1e-9)
    dz = ctr[1] - ctr[0]
    F_ti = np.concatenate([[0.0], np.cumsum(0.5 * (mf[1:] + mf[:-1]) * dz)])
    p = d["H1"][:, 0].sum(0)
    F_h = -np.log(np.maximum(p / p.sum(), 1e-300)) / beta
    F_ti -= F_ti.mean(); F_h -= F_h.mean()
    Fb = d["H1"][:, 0]
    Fbl = np.stack([-np.log(np.maximum(b / b.sum(), 1e-300)) / beta for b in Fb])
    Fbl -= Fbl.mean(1, keepdims=True)
    sd = Fbl.std(0, ddof=1).mean() / math.sqrt(Fb.shape[0])
    res["A_rms_TI_vs_hist"] = float(np.sqrt(((F_ti - F_h) ** 2).mean()))
    res["A_ref_block_sd"] = float(sd)
    res["A_span_kcal"] = float(F_h.max() - F_h.min())
    # the same TI with the WRONG (rigid) weighting, as a control
    mfw = d["W1"][:, 0].sum(0) / np.maximum(d["W0"][:, 0].sum(0), 1e-9)
    F_w = np.concatenate([[0.0], np.cumsum(0.5 * (mfw[1:] + mfw[:-1]) * dz)])
    F_w -= F_w.mean()
    res["A_rms_wrongweight"] = float(np.sqrt(((F_w - F_h) ** 2).mean()))

    # ---- B: stratified constrained TI ---------------------------------------
    cfg = MolCfg(N=a.N, n_steps=a.steps, n_cond=50, dep_every=5, save_every=5_000,
                 n_eq=3_000, init="grid_cold", n_windows=a.N, w_mode="none",
                 fr_rule="none", bw_mf=0.10)
    t0 = time.time()
    out = run_constrained(sy, cfg, a.rows, seed=1234, ref=ref)
    eF = gauge_l2(out["F"][-1], ref["F_ref"], mask)
    res["B_eF_final"] = eF.cpu().numpy().tolist()
    res["B_ess_fixman"] = out["ess_fix"][-1].cpu().numpy().tolist()
    res["B_resid_max"] = float(out["resid"].max())
    res["B_wall_s"] = time.time() - t0
    res["B_fe"] = float(out["fe"][-1])
    res["ref_F_rms"] = float(torch.sqrt((ref["F_ref"][0][mask] ** 2).mean()))
    res["ref_sd_rms"] = float(torch.sqrt((ref["F_sd"][mask] ** 2).mean()))
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, f"{a.system}_gate1.npz"),
                        F=out["F"].cpu().numpy(), fe=out["fe"].cpu().numpy(),
                        F_ref=ref["F_ref"].cpu().numpy(),
                        F_sd=ref["F_sd"].cpu().numpy(),
                        F_ti_unb=F_ti, F_hist_unb=F_h, F_wrong=F_w, centers=ctr)
    with open(os.path.join(a.out, f"{a.system}_gate1.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
