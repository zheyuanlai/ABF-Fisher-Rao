"""Timestep / discretisation validation tools for the pentane R15 cell (docs/PENTANE_R15_DT_FR.md).

Validation instruments only -- nothing here is an ABF or Fisher--Rao comparison arm, and nothing
here is imported by the production sampler.  Three pieces:

* :func:`exact_gibbs_pool` / :func:`sample_exact` -- independent exact Gibbs samples of the pentane
  united-atom model from the internal-coordinate proposal of the v2 reference builder (bonds
  ``r^2 exp(-beta k0 (r-d0)^2/2)``, bends ``sin(theta) exp(-beta k_theta (theta-theta0)^2/2)``,
  dihedrals ``exp(-beta V4)``) with the exact weight ``exp(-beta V_nb)``, resampled either to the
  Gibbs law (``target='gibbs'``) or to the Gibbs law reweighted by ``exp(+beta B(R))`` on a window
  of R bins (``target='flat'``, ``B`` = a fixed bias profile).  Within each R bin the draw is a
  weighted reservoir sample (Efraimidis--Spirakis keys), so it is the exact conditional law per bin.
* :func:`hessian_batch` -- the exact 15 x 15 Hessian of the physical potential by autograd.
* :func:`em_validation_run` -- the production overdamped Euler--Maruyama update, verbatim
  (``q <- remove_com(q + dt (F + b) + sqrt(2 dt / beta) xi)`` with the production per-atom force
  clip and the production local mean force), under either no bias or a FIXED bias profile, started
  from exact samples, with every accumulator the timestep audit needs (R histogram, binned local
  mean force, bond/bend histograms, torsional-family populations through time, clip activity,
  per-step displacement, non-finite states).

``em_validation_run`` deposits nothing into any estimator and has no adaptive state: its stationary
law is the integrator's own, so comparing its R-free energy against the exact reference isolates the
discretisation error of the integrator from the discovery / conditional-mixing problem of ABF.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from . import geometry as geom
from . import interval as iv
from . import potentials as pot
from .distance_cv import DistanceCV
from .reference import sample_bond_lengths, sample_bond_angles
from .reference_cv import sample_dihedral_v4

PI = math.pi
EPS = 1.0e-12
BASIN_BARRIER = math.radians(61.6)
FAMILY_NAMES = ("TT", "TG", "GG_same", "GG_opp")      # TG = T/G+-, G+-/T; same = G+G+ / G-G-; opp = G+G- / G-G+


def basin_of(phi, barrier=BASIN_BARRIER):
    """0 = trans (|phi| < barrier), 1 = G+, 2 = G-."""
    b = torch.zeros_like(phi, dtype=torch.long)
    b = torch.where(phi >= barrier, torch.ones_like(b), b)
    return torch.where(phi <= -barrier, 2 * torch.ones_like(b), b)


def family_of(phi1, phi2, barrier=BASIN_BARRIER):
    """Torsional family 0 TT, 1 TG (either order), 2 same-sign GG, 3 opposite-sign GG."""
    b1 = basin_of(phi1, barrier); b2 = basin_of(phi2, barrier)
    ng = (b1 > 0).long() + (b2 > 0).long()
    fam = torch.where(ng == 0, torch.zeros_like(b1), torch.ones_like(b1))
    both = ng == 2
    fam = torch.where(both & (b1 == b2), 2 * torch.ones_like(b1), fam)
    fam = torch.where(both & (b1 != b2), 3 * torch.ones_like(b1), fam)
    return fam


def basin9_of(phi1, phi2, barrier=BASIN_BARRIER):
    return 3 * basin_of(phi1, barrier) + basin_of(phi2, barrier)


# ---------------------------------------------------------------------------
# exact Gibbs samples
# ---------------------------------------------------------------------------
def exact_gibbs_pool(m, p: pot.AlkaneParams, gen, device="cpu", dtype=torch.float64):
    """``m`` proposals from the internal-coordinate law and their exact log-weights ``-beta V_nb``."""
    nb = p.n_atoms - 1; na = p.n_atoms - 2; nd = p.n_atoms - 3
    bonds = torch.stack([sample_bond_lengths(m, p, gen, device, dtype) for _ in range(nb)], 1)
    angles = torch.stack([sample_bond_angles(m, p, gen, device, dtype) for _ in range(na)], 1)
    dih = torch.stack([sample_dihedral_v4(m, p, gen, device, dtype) for _ in range(nd)], 1)
    q = geom.place_chain_internal(bonds, angles, dih, p.n_atoms, device=device, dtype=dtype)
    logw = -p.beta * pot.nonbonded_energy(q, p)
    return q, logw


def sample_exact(n_out, p: pot.AlkaneParams, seed, *, target="gibbs", n_pool=20_000_000, chunk=1_000_000,
                 grid=None, bias=None, window_mask=None, cv=None, device="cpu", dtype=torch.float64):
    """Exact samples from the pentane Gibbs law (``target='gibbs'``) or from the window-restricted
    Gibbs law reweighted by ``exp(+beta * bias(R))`` (``target='flat'``; ``bias`` on ``grid``, the
    window = the grid bins with ``window_mask``).  Returns ``(q (n_out, A, 3), info)``.

    gibbs: one weighted reservoir of size ``n_out`` over the whole pool (exact Gibbs, sampling
    without replacement from a pool >> n_out).  flat: one reservoir per window bin with quota
    ``n_out / n_window_bins`` (uniform over window bins, exact Gibbs conditional within each bin;
    with ``bias = F_ref`` this is the flat-bias stationary law up to within-bin resolution).
    """
    cv = cv or DistanceCV(0, p.n_atoms - 1)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    gen_key = torch.Generator(device=device).manual_seed(int(seed) + 7919)
    if target == "gibbs":
        keys_best = torch.full((0,), -float("inf"), dtype=dtype); q_best = torch.zeros(0, p.n_atoms, 3, dtype=dtype)
    elif target == "flat":
        grid_t = torch.as_tensor(np.asarray(grid), dtype=dtype); n_grid = grid_t.numel()
        dz = float(grid_t[1] - grid_t[0]); lo = float(grid_t[0]) - dz / 2; hi = float(grid_t[-1]) + dz / 2
        win = np.nonzero(np.asarray(window_mask))[0]
        quota = int(math.ceil(n_out / win.size))
        in_win = torch.zeros(n_grid, dtype=torch.bool); in_win[torch.as_tensor(win)] = True
        res_keys = torch.zeros(0, dtype=dtype); res_bin = torch.zeros(0, dtype=torch.long)
        res_q = torch.zeros(0, p.n_atoms, 3, dtype=dtype)
    else:
        raise ValueError(target)
    done = 0; wsum = 0.0; w2sum = 0.0
    with torch.no_grad():
        while done < n_pool:
            m = min(chunk, n_pool - done)
            q, logw = exact_gibbs_pool(m, p, gen, device, dtype)
            w = torch.exp(logw); wsum += float(w.sum()); w2sum += float((w * w).sum())
            u = torch.rand(m, generator=gen_key, device=device, dtype=dtype).clamp_min(1e-300)
            if target == "gibbs":
                key = torch.log(u) / w.clamp_min(1e-300)          # Efraimidis-Spirakis: keep the largest u^(1/w)
                allk = torch.cat([keys_best, key]); allq = torch.cat([q_best, q])
                top = torch.topk(allk, min(n_out, allk.numel())).indices
                keys_best, q_best = allk[top], allq[top]
            else:
                # per-bin reservoir: the factor exp(beta bias(bin)) is constant within a bin, so the
                # conditional law per bin needs only the Gibbs weight w; the bin quota sets the flat marginal
                R = cv.value(q)
                idx = iv.bin_index(R, n_grid, lo, hi)
                sel = (R >= lo) & (R < hi) & in_win[idx]
                key = torch.log(u[sel]) / w[sel].clamp_min(1e-300)
                allk = torch.cat([res_keys, key]); allb = torch.cat([res_bin, idx[sel]]); allq = torch.cat([res_q, q[sel]])
                o1 = torch.argsort(allk, descending=True)
                o2 = torch.argsort(allb[o1], stable=True)
                order = o1[o2]
                bsorted = allb[order]
                start = torch.searchsorted(bsorted, bsorted, right=False)
                rank = torch.arange(bsorted.numel()) - start
                keep = order[rank < quota]
                res_keys, res_bin, res_q = allk[keep], allb[keep], allq[keep]
            done += m
    info = dict(n_pool=n_pool, global_ess_frac=(wsum ** 2 / w2sum) / n_pool, target=target)
    if target == "gibbs":
        perm = torch.randperm(q_best.shape[0], generator=gen)
        return q_best[perm], info
    counts = torch.bincount(res_bin, minlength=n_grid).numpy()[win]
    info["min_bin_count"] = int(counts.min()); info["quota"] = quota
    perm = torch.randperm(res_q.shape[0], generator=gen)[:n_out]
    return res_q[perm], info


# ---------------------------------------------------------------------------
# Hessian
# ---------------------------------------------------------------------------
def hessian_batch(q, p: pot.AlkaneParams):
    """Exact Hessian of the physical potential, ``(B, 3A, 3A)`` (no clip -- the clip is not a potential)."""
    B, A, _ = q.shape

    def e_single(x):
        return pot.total_energy(x.reshape(1, A, 3), p).sum()
    H = torch.func.vmap(torch.func.hessian(e_single))(q.reshape(B, 3 * A).detach())
    return H


def minimise(q, p: pot.AlkaneParams, n_iter=20000, lr=2.0e-4):
    """Plain steepest descent (unclipped force) to the nearest local minimum."""
    p0 = pot.AlkaneParams(**{**p.__dict__, "force_clip": 0.0})
    for _ in range(n_iter):
        q = geom.remove_com(q + lr * pot.forces(q, p0))
    return q


# ---------------------------------------------------------------------------
# EM validation run (production update, fixed bias, exact start)
# ---------------------------------------------------------------------------
def _wall_g(R, wall_lo, wall_hi, k_wall):
    f = torch.zeros_like(R)
    f = torch.where(R > wall_hi, -k_wall * (R - wall_hi), f)
    return torch.where(R < wall_lo, -k_wall * (R - wall_lo), f)


def clipped_forces_with_stats(q, p: pot.AlkaneParams):
    """Production force (identical arithmetic to :func:`potentials.forces`) + clip activity.

    Returns ``(f_clipped, n_clipped_atoms, max_raw_norm, sum_excess)`` where ``sum_excess`` is the
    summed norm of the force removed by the clip.
    """
    q = q.detach().requires_grad_(True)
    e = pot.total_energy(q, p).sum()
    (g,) = torch.autograd.grad(e, q, create_graph=False)
    f = -g
    norm = torch.linalg.norm(f, dim=-1, keepdim=True)
    if p.force_clip and p.force_clip > 0:
        scale = torch.clamp(p.force_clip / norm.clamp_min(pot.EPS), max=1.0)
        f = f * scale
        over = norm[..., 0] > p.force_clip
        n_clip = int(over.sum()); excess = float((norm[..., 0] - p.force_clip).clamp_min(0.0).sum())
    else:
        n_clip = 0; excess = 0.0
    return f.detach(), n_clip, float(norm.max()), excess


def em_validation_run(q0, p: pot.AlkaneParams, dt, n_steps, seed, *, grid, bias_mf=None, wall_lo=None, wall_hi=None,
                      k_wall=200.0, n_burn=0, n_snap=20, f_sample_clip=480.0, bond_edges=None, bend_edges=None,
                      cv=None, deposit_stride=1, verbose=False):
    """Production overdamped EM from ``q0`` under a FIXED generalized bias force ``bias_mf(R)`` (the
    derivative profile on ``grid``; ``None`` = unbiased), optional soft walls; returns accumulators.

    Accumulated after ``n_burn`` steps every ``deposit_stride`` steps (all walkers):
    R histogram, binned local mean force (production clip of +-``f_sample_clip``) and its square,
    bond-length and bend-angle histograms, 9-basin torsion counts per R bin.  Every step: clip
    activity, max per-atom displacement, non-finite count.  ``n_snap`` snapshots of the 4-family
    populations (all walkers) through time.
    """
    cv = cv or DistanceCV(0, p.n_atoms - 1)
    grid_t = torch.as_tensor(np.asarray(grid), dtype=q0.dtype); n_grid = grid_t.numel()
    dz = float(grid_t[1] - grid_t[0]); lo = float(grid_t[0]) - dz / 2; hi = float(grid_t[-1]) + dz / 2
    mf_t = None if bias_mf is None else torch.as_tensor(np.asarray(bias_mf), dtype=q0.dtype)[None, :]
    gen = torch.Generator().manual_seed(int(seed))
    q = q0.clone(); M, A, _ = q.shape; beta = p.beta
    noise_scale = math.sqrt(2.0 * dt / beta)
    bond_edges = torch.linspace(0.85, 1.15, 121, dtype=q.dtype) if bond_edges is None else bond_edges
    bend_edges = torch.linspace(0.6, 1.8, 121, dtype=q.dtype) if bend_edges is None else bend_edges
    acc = dict(c=torch.zeros(n_grid, dtype=q.dtype), fs=torch.zeros(n_grid, dtype=q.dtype), f2=torch.zeros(n_grid, dtype=q.dtype),
               bond=torch.zeros(p.n_atoms - 1, bond_edges.numel() - 1, dtype=q.dtype),
               bend=torch.zeros(p.n_atoms - 2, bend_edges.numel() - 1, dtype=q.dtype),
               b9=torch.zeros(n_grid, 9, dtype=q.dtype))
    n_clip = 0; n_atom_evals = 0; max_raw = 0.0; excess = 0.0; max_disp = 0.0; n_nonfinite = 0; n_fsample_clip = 0
    n_dep = 0; bond_sum = torch.zeros(p.n_atoms - 1, dtype=q.dtype); bond_sq = torch.zeros(p.n_atoms - 1, dtype=q.dtype)
    snap_steps = np.unique(np.linspace(0, n_steps, n_snap + 1).astype(int))
    fam_series = []; b9_series = []; snap_done = []
    t0 = time.perf_counter()
    for step in range(n_steps + 1):
        F, nc, mr, ex = clipped_forces_with_stats(q, p)
        n_clip += nc; n_atom_evals += M * A; max_raw = max(max_raw, mr); excess += ex
        f_loc, R, grad_full = cv.local_mean_force(q, F, beta)
        if step in snap_steps:
            phi1 = geom.signed_dihedral(q, 0, 1, 2, 3); phi2 = geom.signed_dihedral(q, 1, 2, 3, 4)
            fam_series.append(torch.bincount(family_of(phi1, phi2), minlength=4).numpy())
            b9_series.append(torch.bincount(basin9_of(phi1, phi2), minlength=9).numpy()); snap_done.append(step)
        if step >= n_burn and (step - n_burn) % deposit_stride == 0:
            fl = torch.clamp(f_loc, -f_sample_clip, f_sample_clip); n_fsample_clip += int((f_loc.abs() > f_sample_clip).sum())
            idx = iv.bin_index(R, n_grid, lo, hi); ins = ((R >= lo) & (R < hi)).to(q.dtype)
            acc["c"].scatter_add_(0, idx, ins); acc["fs"].scatter_add_(0, idx, ins * fl); acc["f2"].scatter_add_(0, idx, ins * fl * fl)
            for k in range(p.n_atoms - 1):
                r = torch.linalg.norm(q[:, k + 1] - q[:, k], dim=-1)
                acc["bond"][k] += torch.histc(r, bins=bond_edges.numel() - 1, min=float(bond_edges[0]), max=float(bond_edges[-1]))
                bond_sum[k] += r.sum(); bond_sq[k] += (r * r).sum()
            for k in range(p.n_atoms - 2):
                b1 = q[:, k + 1] - q[:, k]; b2 = q[:, k + 2] - q[:, k + 1]
                c = ((b1 * b2).sum(-1) / (torch.linalg.norm(b1, dim=-1) * torch.linalg.norm(b2, dim=-1))).clamp(-1, 1)
                acc["bend"][k] += torch.histc(torch.arccos(c), bins=bend_edges.numel() - 1, min=float(bend_edges[0]), max=float(bend_edges[-1]))
            phi1 = geom.signed_dihedral(q, 0, 1, 2, 3); phi2 = geom.signed_dihedral(q, 1, 2, 3, 4)
            lin = idx * 9 + basin9_of(phi1, phi2)
            acc["b9"].view(-1).scatter_add_(0, lin, ins)
            n_dep += M
        if step == n_steps:
            break
        g = torch.zeros_like(R)
        if mf_t is not None:
            g = g + iv.interval_interp(mf_t, grid_t, R[None, :])[0]
        if wall_lo is not None:
            g = g + _wall_g(R, wall_lo, wall_hi, k_wall)
        bias_force = g[:, None, None] * grad_full
        noise = torch.randn(q.shape, generator=gen, dtype=q.dtype)
        dq = dt * (F + bias_force) + noise_scale * noise
        q_new = geom.remove_com(q + dq)
        disp = float(torch.linalg.norm(q_new - q, dim=-1).max())
        bad = ~torch.isfinite(q_new).all(dim=(-2, -1))
        if bool(bad.any()):
            n_nonfinite += int(bad.sum())
            q_new[bad] = q[bad]                     # freeze a non-finite walker (counted, reported)
        max_disp = max(max_disp, disp if math.isfinite(disp) else float("inf"))
        q = q_new
        if verbose and step % max(n_steps // 10, 1) == 0:
            print(f"    step {step}/{n_steps} {time.perf_counter() - t0:.0f}s clip {n_clip / max(n_atom_evals, 1):.2e}", flush=True)
    out = {k: v.numpy() for k, v in acc.items()}
    out.update(dict(n_clip=n_clip, n_atom_evals=n_atom_evals, clip_frac=n_clip / max(n_atom_evals, 1), max_raw_force=max_raw,
                    mean_clip_excess=excess / max(n_clip, 1), max_disp=max_disp, n_nonfinite=n_nonfinite,
                    n_fsample_clip=n_fsample_clip, n_dep=n_dep, bond_mean=(bond_sum / max(n_dep, 1)).numpy(),
                    bond_var=(bond_sq / max(n_dep, 1) - (bond_sum / max(n_dep, 1)) ** 2).numpy(),
                    fam_series=np.asarray(fam_series), b9_series=np.asarray(b9_series), snap_steps=np.asarray(snap_done),
                    bond_edges=bond_edges.numpy(), bend_edges=bend_edges.numpy(), wall_s=time.perf_counter() - t0, q_final=q.numpy()))
    return out


def exact_accumulators(q, p: pot.AlkaneParams, grid, cv=None, f_sample_clip=480.0, bond_edges=None, bend_edges=None, weights=None):
    """The same accumulators as :func:`em_validation_run` on a batch of exact samples (optionally weighted)."""
    cv = cv or DistanceCV(0, p.n_atoms - 1)
    grid_t = torch.as_tensor(np.asarray(grid), dtype=q.dtype); n_grid = grid_t.numel()
    dz = float(grid_t[1] - grid_t[0]); lo = float(grid_t[0]) - dz / 2; hi = float(grid_t[-1]) + dz / 2
    bond_edges = torch.linspace(0.85, 1.15, 121, dtype=q.dtype) if bond_edges is None else bond_edges
    bend_edges = torch.linspace(0.6, 1.8, 121, dtype=q.dtype) if bend_edges is None else bend_edges
    w = torch.ones(q.shape[0], dtype=q.dtype) if weights is None else weights
    F, nc, mr, _ = clipped_forces_with_stats(q, p)
    f_loc, R, _ = cv.local_mean_force(q, F, p.beta)
    fl = torch.clamp(f_loc, -f_sample_clip, f_sample_clip)
    idx = iv.bin_index(R, n_grid, lo, hi); ins = ((R >= lo) & (R < hi)).to(q.dtype) * w
    c = torch.zeros(n_grid, dtype=q.dtype).scatter_add_(0, idx, ins)
    fs = torch.zeros(n_grid, dtype=q.dtype).scatter_add_(0, idx, ins * fl)
    bond = []; bend = []
    for k in range(p.n_atoms - 1):
        r = torch.linalg.norm(q[:, k + 1] - q[:, k], dim=-1)
        bond.append(torch.histogram(r, bins=bond_edges, weight=w).hist)
    for k in range(p.n_atoms - 2):
        b1 = q[:, k + 1] - q[:, k]; b2 = q[:, k + 2] - q[:, k + 1]
        cc = ((b1 * b2).sum(-1) / (torch.linalg.norm(b1, dim=-1) * torch.linalg.norm(b2, dim=-1))).clamp(-1, 1)
        bend.append(torch.histogram(torch.arccos(cc), bins=bend_edges, weight=w).hist)
    return dict(c=c.numpy(), fs=fs.numpy(), bond=torch.stack(bond).numpy(), bend=torch.stack(bend).numpy(),
                n_clip=nc, max_raw_force=mr, n_atoms=q.shape[0] * q.shape[1], n_fsample_clip=int((f_loc.abs() > f_sample_clip).sum()))
