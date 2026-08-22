"""Arm dispatch, scoring, and storage for the RC-WFR campaign.

An ARM is (engine, RunConfig overrides).  Every arm in a comparison is run with
the SAME N, dt and n_steps, so the force-evaluation axis is shared by
construction; `fe` is stored anyway and asserted to agree within 10%.
"""
from __future__ import annotations

import json, os
from dataclasses import replace

import numpy as np
import torch

from .engines import ARMS, RunConfig
from .estimators import gauge_l2
from .grid import Grid1D, trapz
from .systems.base import SepSystem


ARM_LIBRARY = {
    # --- RC-WFR family -------------------------------------------------------
    "wfr":        ("wfr",  dict(w_mode="sde",  fr_rule="fr",    init="point")),
    "w_only":     ("wfr",  dict(w_mode="sde",  fr_rule="none",  init="point")),
    "fr_only":    ("wfr",  dict(w_mode="none", fr_rule="fr",    init="point")),
    "w_count":    ("wfr",  dict(w_mode="sde",  fr_rule="count", init="point")),
    "w_sham":     ("wfr",  dict(w_mode="sde",  fr_rule="sham",  init="point")),
    "wfr_flow":   ("wfr",  dict(w_mode="flow", fr_rule="fr",    init="point")),
    "wfr_gmm":    ("wfr",  dict(w_mode="flow", fr_rule="fr",    init="point",
                                density_model="gmm")),
    "wfr_gmm_sde":("wfr",  dict(w_mode="sde",  fr_rule="fr",    init="point",
                                density_model="gmm")),
    "wfr_oracle": ("wfr",  dict(w_mode="sde",  fr_rule="fr",    init="point",
                                lift="oracle")),
    # --- classical stratified baselines -------------------------------------
    "ti_warm":    ("wfr",  dict(w_mode="none", fr_rule="none",  init="grid_warm")),
    "ti_cold":    ("wfr",  dict(w_mode="none", fr_rule="none",  init="grid_cold")),
    "reti_warm":  ("reti", dict(init="grid_warm")),
    "reti_cold":  ("reti", dict(init="grid_cold")),
    # --- adaptive biasing ----------------------------------------------------
    "abf":        ("abf",  dict(init="point")),
    "shus":       ("shus", dict(init="point")),
    "unbiased":   ("unbiased", dict(init="point")),
}


def run_arm(sys: SepSystem, arm: str, base: RunConfig, rows: int, seed: int,
            overrides=None, sham_source=None):
    engine, defaults = ARM_LIBRARY[arm]
    cfg = replace(base, **defaults)
    if overrides:
        cfg = replace(cfg, **overrides)
    fn = ARMS[engine]
    if engine == "wfr":
        return fn(sys, cfg, rows, seed, sham_source=sham_source), cfg
    return fn(sys, cfg, rows, seed), cfg


def score(run, sys: SepSystem, eps_ladder=(0.30, 0.20, 0.15, 0.10, 0.07, 0.05)):
    """Per-row metrics from a stored run."""
    g = sys.grid
    mask = g.eval_mask(sys.device, sys.dtype)
    n_saves, rows, _ = run["F"].shape
    eF = torch.stack([gauge_l2(run["F"][i], sys.F_ref, mask) for i in range(n_saves)])
    fe = run["fe"].cpu().numpy()
    eFn = eF.cpu().numpy()
    IF = np.trapezoid(eFn, fe, axis=0) / max(fe[-1] - fe[0], 1.0)
    tau = {}
    for eps in eps_ladder:
        t = np.full(rows, np.nan)
        below = eFn <= eps
        T = fe[-1]
        for r in range(rows):
            for i in range(n_saves):
                j = np.searchsorted(fe, fe[i] + 0.2 * T, side="right")
                j = max(j, i + 1)
                if below[i:j, r].all():
                    t[r] = fe[i]
                    break
        tau[f"tau_{eps}"] = t
    nrm = float(torch.sqrt(((sys.F_ref[0][mask] - sys.F_ref[0][mask].mean()) ** 2).mean()))
    out = {"e_F": eFn, "fe": fe, "I_F": IF, "e_F_final": eFn[-1],
           "F_rms": nrm, "I_F_rel": IF / nrm, "e_F_rel_final": eFn[-1] / nrm,
           "kl": run["kl"].cpu().numpy(), "cov": run["cov"].cpu().numpy(),
           "chan": run["chan"].cpu().numpy(),
           "ess_anc": run["ess_anc"].cpu().numpy(),
           "surv_anc": run["surv_anc"].cpu().numpy()}
    out.update(tau)
    if "ex_accept" in run:
        out["ex_accept"] = run["ex_accept"]
    return out


def paired_bootstrap(values, stat=np.median, n_boot=10_000, alpha=0.05, seed=20260822):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    s = np.array([stat(v[rng.integers(0, v.size, v.size)]) for _ in range(n_boot)])
    lo, hi = np.quantile(s, [alpha / 2, 1 - alpha / 2])
    return float(stat(v)), float(lo), float(hi)


def rel_change(arm_vals, base_vals):
    """Paired relative change (arm - base)/base, per seed."""
    a, b = np.asarray(arm_vals, float), np.asarray(base_vals, float)
    return (a - b) / b


def estimator_floor(sys: SepSystem, cfg: RunConfig, n_samples_list, rows=8, seed=99):
    """e_F an i.i.d. oracle sample (Z ~ u, Y ~ nu^xi(.|Z)) reaches through the SAME
    estimator.  Anything below this is unreachable; anything at it is converged."""
    from .estimators import MeanForceAccumulator
    g, dev, dt = sys.grid, sys.device, sys.dtype
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    mask = g.eval_mask(dev, dt)
    acc = MeanForceAccumulator(rows, g, cfg.bw_mf, cfg.n_min, dev, dt)
    out, done = {}, 0
    chunk = 131072
    for target in sorted(n_samples_list):
        while done < target:
            b = min(chunk, target - done)
            X = (torch.rand((rows, b), device=dev, dtype=dt, generator=gen)
                 * (g.eval_hi - g.eval_lo) + g.eval_lo)
            Y = sys.sample_conditional(X, gen)
            acc.deposit(X, sys.mean_force(X, Y))
            done += b
        out[target] = gauge_l2(acc.free_energy(mask), sys.F_ref, mask).cpu().numpy()
    return out


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
