"""Run one molecular arm (all seeds batched into the row axis) and store it.

Every arm in a comparison gets the same N, the same n_steps and the same
deposit rate, so force evaluations match by construction; `fe` is stored and
checked anyway.  Seeds are rows, so a 32-seed confirmation is ONE GPU job.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from dataclasses import replace, asdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.estimators import gauge_l2
from rcwfr.mol import systems as S
from rcwfr.mol.engines import (MolCfg, run_abf, run_constrained, run_opes,
                                run_opes_true)
from rcwfr.mol.refdata import load_reference


ARM_LIBRARY = {
    # --- classical stratified baselines -------------------------------------
    "ti_cold":     ("constrained", dict(init="grid_cold", w_mode="none", fr_rule="none")),
    "ti_warm":     ("constrained", dict(init="grid_warm", w_mode="none", fr_rule="none")),
    # --- adaptive biasing ----------------------------------------------------
    "abf":         ("abf", dict()),
    "abp":         ("opes", dict()),          # visit-count adaptive biasing potential
    "opes":        ("opes_true", dict()),     # OPES proper
    # --- RC-WFR --------------------------------------------------------------
    "wfr_shake":   ("constrained", dict(init="point", w_mode="sde", fr_rule="fr",
                                        lift="shake")),
    "wfr_rot":     ("constrained", dict(init="point", w_mode="sde", fr_rule="fr",
                                        lift="rot")),
    "wfr_ymap":    ("constrained", dict(init="point", w_mode="sde", fr_rule="fr",
                                        lift="ymap_oracle")),
    "wfr_yref":    ("constrained", dict(init="point", w_mode="sde", fr_rule="fr",
                                        lift="yref_oracle")),
    "wfr_lmap":    ("constrained", dict(init="point", w_mode="sde", fr_rule="fr",
                                        lift="ymap_learned")),
    "wfr_lref":    ("constrained", dict(init="point", w_mode="sde", fr_rule="fr",
                                        lift="yref_learned")),
    # the practical arm: a Metropolis-corrected learned conditional move on y
    "wfr_lmh":     ("constrained", dict(init="point", w_mode="sde", fr_rule="fr",
                                        lift="ymh_learned")),
    "wfr_ymh":     ("constrained", dict(init="point", w_mode="sde", fr_rule="fr",
                                        lift="ymh_oracle")),
    "wfr_qref":    ("constrained", dict(init="point", w_mode="sde", fr_rule="fr",
                                        lift="qref_oracle")),
    "wfr_qref_uw": ("constrained", dict(init="point", w_mode="sde", fr_rule="fr",
                                        lift="qref_oracle", fixman_weight=False)),
    # --- mechanism ablations -------------------------------------------------
    "w_only":      ("constrained", dict(init="point", w_mode="sde", fr_rule="none",
                                        lift="rot")),
    "w_only_y":    ("constrained", dict(init="point", w_mode="sde", fr_rule="none",
                                        lift="yref_oracle")),
    "fr_only":     ("constrained", dict(init="point", w_mode="none", fr_rule="fr",
                                        lift="rot")),
    "w_count":     ("constrained", dict(init="point", w_mode="sde", fr_rule="count",
                                        lift="rot")),
    "wfr_flow":    ("constrained", dict(init="point", w_mode="flow", fr_rule="fr",
                                        lift="rot")),
    "wfr_flow_y":  ("constrained", dict(init="point", w_mode="flow", fr_rule="fr",
                                        lift="yref_oracle")),
}
ENGINES = {"constrained": run_constrained, "abf": run_abf, "opes": run_opes,
           "opes_true": run_opes_true}


def score(out, ref, sy):
    mask = sy.grid.eval_mask(sy.device, sy.dtype)
    n_saves = out["F"].shape[0]
    eF = torch.stack([gauge_l2(out["F"][i], ref["F_ref"], mask) for i in range(n_saves)])
    fe = out["fe"].cpu().numpy()
    e = eF.cpu().numpy()
    IF = np.trapezoid(e, fe, axis=0) / max(fe[-1] - fe[0], 1.0)
    ep = (torch.stack([gauge_l2(out["F_prod"][i], ref["F_ref"], mask)
                       for i in range(n_saves)]).cpu().numpy()
          if "F_prod" in out else e)
    return {"e_F": e, "e_F_prod": ep, "fe": fe, "I_F": IF, "e_F_final": e[-1],
            "e_F_prod_final": ep[-1],
            "kl": out["kl"].cpu().numpy(), "cov": out["cov"].cpu().numpy(),
            "dcond": out["dcond"].cpu().numpy(),
            "dcond_all": (out["dcond_all"].cpu().numpy()
                          if "dcond_all" in out else np.zeros((0,))),
            "ess_fix": out["ess_fix"].cpu().numpy(),
            "ess_anc": out["ess_anc"].cpu().numpy(),
            "lift_cov": out["lift_cov"].cpu().numpy(),
            "resid": out["resid"].cpu().numpy()}


DEFAULTS = dict(system="PEN", arm=None, seeds=8, seed0=1000, N=256, steps=100_000,
                n_cond=20, n_windows=64, kappa="0.30", theta="0.30", decay=None,
                bw_mf=0.05, bw_kde=0.25, lift_bw_z=0.25, lift_bw_y=0.30,
                lift_decay=0.999, lift_nmin=150.0, lift_start=0.0, abf_nmin=200.0,
                shus_gain=1.0, opes_sigma=0.10, opes_barrier=20.0,
                promote=(1,), z0=0.0, t_switch=0, snap=False, freeze_lift=False,
                snap_windows=0, auto_switch=False, eps_snap=0.0,
                eps_learn=0.0, eps_ens=0.0,
                fr_jitter=0.0, dep_every=20, save_every=5_000, n_eq=2_000,
                tag="", out="results/mol/campaign")


class _NS(dict):
    __getattr__ = dict.__getitem__


_CACHE = {}


def run_one(**kw):
    """One arm, all seeds and hyper-parameter configurations in the row axis.

    Reference tables and the system are cached across calls so a driver that
    loops over arms in ONE process pays the torch.compile warm-up once per
    distinct code path instead of once per arm.
    """
    a = _NS({**DEFAULTS, **kw})
    dev, dt = torch.device("cuda"), torch.float64
    key = a.system
    if key not in _CACHE:
        sy = S.REGISTRY[a.system](dev, dt)
        tip = f"results/mol/ref/{a.system}_tiref.npz"
        cnd = f"results/mol/ref_cond/{a.system}_tiref.npz"
        ref = load_reference(f"results/mol/ref/{a.system}_ref.npz", sy.grid,
                             sy.y_grid or sy.grid, dev, dt,
                             cv_shift=sy.cv_shift,
                             ti_path=(tip if os.path.exists(tip) else None),
                             cond_path=(cnd if os.path.exists(cnd) else None))
        _CACHE[key] = (sy, ref)
    sy, ref = _CACHE[key]
    eng, defaults = ARM_LIBRARY[a.arm]
    kap = [float(x) for x in str(a.kappa).split(",")]
    the = [float(x) for x in str(a.theta).split(",")]
    dec = [float(x) for x in str(a.decay).split(",")] if a.decay else [a.lift_decay]
    grid_cfg = [(k, t, d) for k in kap for t in the for d in dec]
    n_cfg = len(grid_cfg)
    rows = n_cfg * a.seeds
    kv = torch.tensor([g[0] for g in grid_cfg], dtype=dt).repeat_interleave(a.seeds)
    tv = torch.tensor([g[1] for g in grid_cfg], dtype=dt).repeat_interleave(a.seeds)
    dv = torch.tensor([g[2] for g in grid_cfg], dtype=dt).repeat_interleave(a.seeds)
    cfg = MolCfg(z0=a.z0, N=a.N, n_steps=a.steps, n_cond=a.n_cond, dep_every=a.dep_every,
                 save_every=a.save_every, n_eq=a.n_eq, bw_mf=a.bw_mf,
                 bw_kde=a.bw_kde, kappa=kap[0], theta=the[0],
                 n_windows=a.n_windows, fr_jitter=a.fr_jitter,
                 lift_bw_z=a.lift_bw_z, lift_bw_y=a.lift_bw_y,
                 lift_decay=dec[0], lift_nmin=a.lift_nmin,
                 lift_start=a.lift_start, abf_n_min=a.abf_nmin,
                 t_switch=a.t_switch, shus_gain=a.shus_gain,
                 opes_sigma=a.opes_sigma, opes_barrier=a.opes_barrier,
                 snap_at_switch=a.snap, freeze_lift_at_switch=a.freeze_lift,
                 snap_windows=a.snap_windows, auto_switch=a.auto_switch,
                 eps_snap=a.eps_snap, eps_learn=a.eps_learn, eps_ens=a.eps_ens,
                 promote=tuple(int(x) for x in str(a.promote).split(','))
                 if isinstance(a.promote, str) else tuple(a.promote), **defaults)
    t0 = time.time()
    if eng in ("abf", "opes", "opes_true"):
        out = ENGINES[eng](sy, cfg, rows, seed=a.seed0, ref=ref)
    else:
        out = run_constrained(sy, cfg, rows, seed=a.seed0, ref=ref,
                              kappa_vec=kv, theta_vec=tv, decay_vec=dv)
    torch.cuda.synchronize()
    sc = score(out, ref, sy)
    wall = time.time() - t0
    name = f"{a.system}_{a.arm}{('_' + a.tag) if a.tag else ''}"
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, name + ".npz"),
                        F=out["F"].cpu().numpy(), F_ref=ref["F_ref"].cpu().numpy(),
                        wall=wall, n_cfg=n_cfg, n_seed=a.seeds,
                        fe_switch=float(out.get("fe_switch", 0.0)),
                        diag=(out["diag"].cpu().numpy() if "diag" in out
                              else np.zeros((0, 0, 3))),
                        cfg_grid=np.array(grid_cfg), **sc)
    with open(os.path.join(a.out, name + ".json"), "w") as f:
        json.dump({"arm": a.arm, "system": a.system, "wall_s": wall,
                   "cfg": {k: v for k, v in asdict(cfg).items()},
                   "seeds": a.seeds, "seed0": a.seed0, "n_cfg": n_cfg,
                   "promote": list(cfg.promote),
                   "cfg_grid": [list(g) for g in grid_cfg],
                   "e_F_final_median": float(np.median(sc["e_F_final"])),
                   "e_F_prod_final_median": float(np.median(sc["e_F_prod_final"])),
                   "I_F_median": float(np.median(sc["I_F"])),
                   "dcond_final_median": float(np.median(sc["dcond"][-1])),
                   "cov_final_median": float(np.median(sc["cov"][-1])),
                   "ess_fix_median": float(np.median(sc["ess_fix"][-1])),
                   "fe_final": float(sc["fe"][-1])}, f, indent=1)
    print(f"{name}: e_F={np.median(sc['e_F_final']):.4f} "
          f"e_Fprod={np.median(sc['e_F_prod_final']):.4f} I_F={np.median(sc['I_F']):.4f} "
          f"dcond={np.median(sc['dcond'][-1]):.4f} cov={np.median(sc['cov'][-1]):.3f} "
          f"fe={sc['fe'][-1]:.3g} wall={wall:.0f}s", flush=True)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="PEN")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--n-cond", type=int, default=20)
    ap.add_argument("--n-windows", type=int, default=64)
    ap.add_argument("--kappa", default="0.30")     # comma list -> screened in the row axis
    ap.add_argument("--theta", default="0.30")
    ap.add_argument("--decay", default=None)
    ap.add_argument("--bw-mf", type=float, default=0.08)
    ap.add_argument("--bw-kde", type=float, default=0.25)
    ap.add_argument("--lift-bw-z", type=float, default=0.25)
    ap.add_argument("--lift-bw-y", type=float, default=0.30)
    ap.add_argument("--lift-decay", type=float, default=0.999)
    ap.add_argument("--lift-nmin", type=float, default=150.0)
    ap.add_argument("--lift-start", type=float, default=0.0)
    ap.add_argument("--abf-nmin", type=float, default=200.0)
    ap.add_argument("--shus-gain", type=float, default=1.0)
    ap.add_argument("--opes-sigma", type=float, default=0.10)
    ap.add_argument("--opes-barrier", type=float, default=20.0)
    ap.add_argument("--promote", default="1")
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--t-switch", type=int, default=0)
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--snap-windows", type=int, default=0)
    ap.add_argument("--auto-switch", action="store_true")
    ap.add_argument("--eps-snap", type=float, default=0.0)
    ap.add_argument("--eps-learn", type=float, default=0.0)
    ap.add_argument("--eps-ens", type=float, default=0.0)
    ap.add_argument("--freeze-lift", action="store_true")
    ap.add_argument("--fr-jitter", type=float, default=0.0)
    ap.add_argument("--dep-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=5_000)
    ap.add_argument("--n-eq", type=int, default=2_000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/mol/campaign")
    a = ap.parse_args()
    run_one(**{k: v for k, v in vars(a).items()})


if __name__ == "__main__":
    main()
