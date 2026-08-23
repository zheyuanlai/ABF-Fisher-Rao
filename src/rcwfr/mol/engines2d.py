"""RC-WFR with a TWO-dimensional reaction coordinate.

This is the control the campaign needs, not a new capability for its own sake.
With `z = (phi, psi)` alanine's hidden torsion is no longer hidden, so whatever
advantage survives cannot be "the method repairs an incomplete reaction
coordinate".  Two readings, both useful:

  * the advantage largely survives  -> low-dimensional reaction-coordinate
    transport is worth something by itself;
  * it largely disappears           -> the whole benefit was hidden-mode repair,
    which is a cleaner and more honest mechanism story.

Everything else is held fixed: same constrained dynamics, same Chapter-3 mean
force (now a two-vector), same Fixman weight (now `det G^{-1/2}` of a 2x2), same
cost currency.
"""
from __future__ import annotations

import math

import torch

from ..fisher_rao import fr_weights, theta_backoff
from ..grid import EPS
from ..resampling import ancestor_stats, systematic_resample, turnover_counts
from .dynamics import constrained_step
from .ff import _wrap, rotate_about_bond
from .grid2d import (Grid2D, MeanForceAccumulator2D, gauge_l2_2d, interp2d,
                     kde2d, scatter2d)


def _gather_conf(q, sel):
    A, D = q.shape[-2], q.shape[-1]
    return torch.gather(q, 1, sel[:, :, None, None].expand(-1, -1, A, D))


def _coverage2d(Z, g2, nb=24):
    R = Z.shape[0]
    ix = torch.clamp(((Z[..., 0] - g2.gx.xmin) / g2.gx.volume * nb).long(), 0, nb - 1)
    iy = torch.clamp(((Z[..., 1] - g2.gy.xmin) / g2.gy.volume * nb).long(), 0, nb - 1)
    h = torch.zeros((R, nb * nb), device=Z.device, dtype=Z.dtype)
    h.scatter_add_(1, ix * nb + iy, torch.ones_like(Z[..., 0]))
    return (h > 0).to(Z.dtype).mean(1)


def run2d(sy, cv2, g2: Grid2D, cfg, rows: int, seed: int, ref=None,
          w_mode="sde", fr_rule="fr", init="point", n_win=32, lift="rot",
          abf_mode=False):
    """One arm on a two-dimensional reaction coordinate.

    `lift='rot'` rotates BOTH torsions to their new values, exactly and without
    distortion.  There is no fiber torsion left to promote -- that is the point
    of the experiment.
    """
    dev, dt = sy.device, sy.dtype
    top, beta, h = sy.top, sy.beta, sy.h
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    torch.manual_seed(seed + 4241)
    specs = [(0, sy.z_bond, sy.z_movers), (1, sy.y_bond, sy.y_movers)]

    # --- initial placement -------------------------------------------------
    N = cfg.N
    if init == "point":
        Z = torch.zeros((rows, N, 2), device=dev, dtype=dt)
        Z[..., 0] = cfg.z0; Z[..., 1] = sy.y0
    else:
        M = int(round(math.sqrt(n_win)))
        ax = torch.linspace(g2.gx.xmin, g2.gx.xmax, M + 1, device=dev, dtype=dt)[:M]
        ay = torch.linspace(g2.gy.xmin, g2.gy.xmax, M + 1, device=dev, dtype=dt)[:M]
        gridpts = torch.stack(torch.meshgrid(ax, ay, indexing="ij"), -1).reshape(-1, 2)
        assert N % gridpts.shape[0] == 0, (N, gridpts.shape[0])
        Z = gridpts.repeat_interleave(N // gridpts.shape[0], 0).unsqueeze(0)
        Z = Z.expand(rows, N, 2).clone()
    q = sy.ideal(Z.reshape(-1, 2)).reshape(rows, N, top.n_atoms, 3)

    step = torch.compile(lambda q, z: constrained_step(top, cv2, q, z, h, beta,
                                                       n_newton=cfg.n_newton,
                                                       drift_cap=sy.drift_cap),
                         dynamic=False)
    gradV = torch.compile(lambda q: top.grad(q), dynamic=False)
    mfun = lambda q, gv: cv2.mean_force(q, gv, beta)

    acc = MeanForceAccumulator2D(rows, g2, cfg.bw_mf, cfg.n_min, dev, dt)
    acc_prod = MeanForceAccumulator2D(rows, g2, cfg.bw_mf, cfg.n_min, dev, dt)
    mask = g2.mask(dev, dt)
    n_outer = cfg.n_steps // cfg.n_cond
    save_outer = max(1, cfg.save_every // cfg.n_cond)
    n_saves = n_outer // save_outer
    dtau = cfg.n_cond * h
    theta0 = torch.full((rows,), cfg.theta, device=dev, dtype=dt)
    ar = torch.arange(N, device=dev).unsqueeze(0).expand(rows, N)
    anc = ar.clone()
    sw_outer = (cfg.t_switch // cfg.n_cond) if cfg.t_switch else None
    Zf = lambda *s: torch.zeros(s, device=dev, dtype=dt)
    out = {"F": Zf(n_saves, rows, g2.gx.n, g2.gy.n),
           "F_prod": Zf(n_saves, rows, g2.gx.n, g2.gy.n),
           "cov": Zf(n_saves, rows), "curl": Zf(n_saves, rows),
           "ess_fix": Zf(n_saves, rows), "ess_anc": Zf(n_saves, rows),
           "resid": Zf(n_saves, rows), "fe": Zf(n_saves)}
    fe, si, out_w = 0.0, 0, None
    eq_outer = cfg.n_eq // cfg.n_cond
    bias_n = torch.full((rows, 1), cfg.abf_n_min, device=dev, dtype=dt)

    for it in range(n_outer):
        transporting = sw_outer is None or it < sw_outer
        if sw_outer is not None and it == sw_outer:
            acc_prod.zero_()
        for k in range(cfg.n_cond):
            q = step(q, Z)
            fe += 1.0
            if it >= eq_outer and ((k + 1) % cfg.dep_every == 0):
                gv = gradV(q); fe += 1.0
                f, G = mfun(q, gv)
                w = torch.linalg.det(G) ** -0.5
                acc.deposit(Z, f, weights=w)
                acc_prod.deposit(Z, f, weights=w)
                out_w = w
        if w_mode == "sde" and transporting:
            Zn = g2.enforce(Z + math.sqrt(2.0 * cfg.kappa * dtau)
                            * torch.randn(Z.shape, device=dev, dtype=dt, generator=gen))
        else:
            Zn = Z
        if w_mode != "none" and transporting:
            for i, (ti, bd, mv) in enumerate(specs):
                q = rotate_about_bond(q, bd[0], bd[1], list(mv),
                                      -_wrap(Zn[..., i] - Z[..., i]))
            Z = Zn
        if fr_rule != "none" and transporting:
            p = kde2d(Z, g2, cfg.bw_kde)
            p_at = torch.clamp(interp2d(Z, p, g2), min=EPS)
            vol = g2.gx.volume * g2.gy.volume
            lr = math.log(1.0 / vol) - torch.log(p_at)
            wsel, th, essf = theta_backoff(lr, theta0, cfg.alpha_ess)
            sel = systematic_resample(wsel, gen)
            q = _gather_conf(q, sel)
            Z = torch.gather(Z, 1, sel.unsqueeze(-1).expand(-1, -1, 2))
            anc = torch.gather(anc, 1, sel)
        if (it + 1) % save_outer == 0 and si < n_saves:
            F, curl = acc.free_energy(mask), acc.curl_fraction()
            out["F"][si] = F
            out["F_prod"][si] = acc_prod.free_energy(mask)
            out["curl"][si] = curl
            out["cov"][si] = _coverage2d(Z, g2)
            out["ess_fix"][si] = (1.0 if out_w is None else
                                  (out_w.sum(1) ** 2) / torch.clamp(
                                      (out_w * out_w).sum(1) * out_w.shape[1], min=EPS))
            e, _ = ancestor_stats(anc, N)
            out["ess_anc"][si] = e / N
            out["resid"][si] = cv2.dz_residual(cv2.value(q), Z).abs().amax(dim=(1, 2))
            out["fe"][si] = fe * N
            anc = ar.clone()
            si += 1
    out["Z_final"], out["q_final"] = Z, q
    return out


def run_abf2d(sy, cv2, g2: Grid2D, cfg, rows: int, seed: int, ref=None,
              k_wall=400.0):
    """Multiple-walker ABF on a two-dimensional CV, for the same comparison."""
    dev, dt = sy.device, sy.dtype
    top, beta, h = sy.top, sy.beta, sy.h
    torch.manual_seed(seed + 991)
    N = cfg.N
    phis = torch.zeros((rows, N, 2), device=dev, dtype=dt)
    phis[..., 0] = cfg.z0; phis[..., 1] = sy.y0
    q = sy.ideal(phis.reshape(-1, 2)).reshape(rows, N, top.n_atoms, 3)
    acc = MeanForceAccumulator2D(rows, g2, cfg.bw_mf, cfg.n_min, dev, dt)
    mask = g2.mask(dev, dt)

    def one(q, bg):
        gv = top.grad(q)
        gs = cv2.grad_local(q)                       # (R,N,2,S,3)
        zc = cv2.value(q)
        b = torch.stack([interp2d(zc, bg[..., c], g2) for c in range(2)], -1)
        if k_wall > 0.0:
            for c, gg in enumerate((g2.gx, g2.gy)):
                if gg.bc != "periodic":
                    b[..., c] = b[..., c] + k_wall * (
                        torch.clamp(zc[..., c] - gg.xmax, min=0.0)
                        + torch.clamp(zc[..., c] - gg.xmin, max=0.0))
        gb = gv.clone()
        gb[..., cv2.support, :] = gb[..., cv2.support, :] - (
            b[..., 0, None, None] * gs[..., 0, :, :]
            + b[..., 1, None, None] * gs[..., 1, :, :])
        minv = (1.0 / top.mass).view(-1, 1)
        nz = torch.randn(q.shape, device=q.device, dtype=q.dtype)
        amp = ((2.0 * h / beta) ** 0.5) * torch.sqrt(minv)
        drift = -h * minv * gb
        if sy.drift_cap is not None:
            drift = torch.clamp(drift, -sy.drift_cap * amp, sy.drift_cap * amp)
        return q + drift + amp * nz, zc

    step = torch.compile(one, dynamic=False)
    gradV = torch.compile(lambda q: top.grad(q), dynamic=False)
    n_saves = cfg.n_steps // cfg.save_every
    Zf = lambda *s: torch.zeros(s, device=dev, dtype=dt)
    out = {"F": Zf(n_saves, rows, g2.gx.n, g2.gy.n),
           "F_prod": Zf(n_saves, rows, g2.gx.n, g2.gy.n),
           "cov": Zf(n_saves, rows), "curl": Zf(n_saves, rows),
           "ess_fix": Zf(n_saves, rows) + 1.0, "ess_anc": Zf(n_saves, rows) + 1.0,
           "resid": Zf(n_saves, rows), "fe": Zf(n_saves)}
    si, fe = 0, 0.0
    bias_n = torch.full((rows, 1, 1, 1), cfg.abf_n_min, device=dev, dtype=dt)
    for n in range(cfg.n_steps):
        bg = acc.S1 / (acc.S0.unsqueeze(-1) + bias_n)
        q, zc = step(q, bg)
        fe += 1.0
        if (n + 1) % cfg.dep_every == 0 and n >= cfg.n_eq:
            f, G = cv2.mean_force(q, gradV(q), beta)
            acc.deposit(zc, f)
        if (n + 1) % cfg.save_every == 0 and si < n_saves:
            out["F"][si] = acc.free_energy(mask)
            out["F_prod"][si] = out["F"][si]
            out["curl"][si] = acc.curl_fraction()
            out["cov"][si] = _coverage2d(zc, g2)
            out["fe"][si] = fe * N
            si += 1
    out["Z_final"], out["q_final"] = zc, q
    return out
