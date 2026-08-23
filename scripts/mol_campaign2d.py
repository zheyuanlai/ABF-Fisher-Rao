"""Two-dimensional reaction-coordinate campaign: alanine with z = (phi, psi).

Reference is a stratified 2-CV constrained TI run: with BOTH torsions pinned the
only coordinates left in the fiber are fast, so each window converges quickly and
no oracle conditional is needed anywhere.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.mol import systems as S
from rcwfr.mol.engines import MolCfg
from rcwfr.mol.engines2d import run2d, run_abf2d
from rcwfr.mol.grid2d import gauge_l2_2d

ARMS2D = {
    "wfr":      dict(w_mode="sde",  fr_rule="fr",   init="point"),
    "w_only":   dict(w_mode="sde",  fr_rule="none", init="point"),
    "ti_cold":  dict(w_mode="none", fr_rule="none", init="grid"),
    "abf":      dict(),
}
_CACHE = {}


def load_ref2d(path, device, dtype):
    d = np.load(path)
    F = torch.as_tensor(d["F"], device=device, dtype=dtype)     # (rows, Gx, Gy)
    F = F - F.reshape(F.shape[0], -1).mean(1).view(-1, 1, 1)
    return dict(F_ref=F.mean(0, keepdim=True),
                F_sd=F.std(0, unbiased=True) / np.sqrt(F.shape[0]),
                blocks=F)


def run_one2d(**kw):
    a = dict(system="ALA2D", arm="wfr", seeds=8, seed0=1000, N=1024, steps=200_000,
             n_cond=20, dep_every=20, save_every=10_000, n_eq=4_000, bw_mf=0.05,
             bw_kde=0.30, kappa=1.2, theta=0.3, z0=-0.5236, n_win=64,
             abf_nmin=200.0, t_switch=0, tag="", out="results/mol/campaign2d")
    a.update(kw)
    dev, dt = torch.device("cuda"), torch.float64
    if "sys" not in _CACHE:
        _CACHE["sys"] = S.alanine2d(dev, dt)
    sy, g2 = _CACHE["sys"]
    ref = None
    rp = "results/mol/ref/ALA2D_tiref.npz"
    if os.path.exists(rp):
        ref = load_ref2d(rp, dev, dt)
    cfg = MolCfg(N=a["N"], n_steps=a["steps"], n_cond=a["n_cond"],
                 dep_every=a["dep_every"], save_every=a["save_every"],
                 n_eq=a["n_eq"], bw_mf=a["bw_mf"], bw_kde=a["bw_kde"],
                 kappa=a["kappa"], theta=a["theta"], z0=a["z0"],
                 abf_n_min=a["abf_nmin"], t_switch=a["t_switch"])
    t0 = time.time()
    if a["arm"] == "abf":
        out = run_abf2d(sy, sy.cv, g2, cfg, a["seeds"], seed=a["seed0"])
    else:
        out = run2d(sy, sy.cv, g2, cfg, a["seeds"], seed=a["seed0"],
                    n_win=a["n_win"], **ARMS2D[a["arm"]])
    torch.cuda.synchronize()
    mask = g2.mask(dev, dt)
    ns = out["F"].shape[0]
    if ref is not None:
        e = torch.stack([gauge_l2_2d(out["F"][i], ref["F_ref"], mask)
                         for i in range(ns)]).cpu().numpy()
        ep = torch.stack([gauge_l2_2d(out["F_prod"][i], ref["F_ref"], mask)
                          for i in range(ns)]).cpu().numpy()
    else:
        e = ep = np.zeros((ns, a["seeds"]))
    fe = out["fe"].cpu().numpy()
    IF = np.trapezoid(e, fe, axis=0) / max(fe[-1] - fe[0], 1.0)
    wall = time.time() - t0
    name = f"ALA2D_{a['arm']}{('_' + a['tag']) if a['tag'] else ''}"
    os.makedirs(a["out"], exist_ok=True)
    np.savez_compressed(os.path.join(a["out"], name + ".npz"),
                        e_F=e, e_F_prod=ep, fe=fe, I_F=IF, e_F_final=e[-1],
                        e_F_prod_final=ep[-1], F=out["F"][-1].cpu().numpy(),
                        cov=out["cov"].cpu().numpy(), curl=out["curl"].cpu().numpy(),
                        ess_fix=out["ess_fix"].cpu().numpy(),
                        n_cfg=1, n_seed=a["seeds"], wall=wall)
    cov = float(out["cov"][-1].median()); curl = float(out["curl"][-1].median())
    print(f"{name}: e_F={np.median(e[-1]):.4f} e_Fprod={np.median(ep[-1]):.4f} "
          f"I_F={np.median(IF):.4f} cov={cov:.3f} curl={curl:.3f} "
          f"fe={fe[-1]:.3g} wall={wall:.0f}s", flush=True)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    for i, kw in enumerate(spec):
        print(f"[{i+1}/{len(spec)}] {kw.get('arm')} {kw.get('tag','')}", flush=True)
        try:
            run_one2d(**kw)
        except Exception:
            import traceback; traceback.print_exc()
    print("ALL2D_DONE", flush=True)


if __name__ == "__main__":
    main()
