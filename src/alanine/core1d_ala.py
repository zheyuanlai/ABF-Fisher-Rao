"""1-D ABF (+ uniform-target Fisher--Rao) sampler for Ace-Ala-Nme with an INCOMPLETE CV.

The reaction coordinate is ONE backbone dihedral, ``phi`` or ``psi``; the other one is the
hidden coordinate and is recorded at every save (joint walker histogram on the 97 x 97 grid,
basin labels from the accepted 2-D watershed) so the conditional distribution of the omitted
angle can be checked against the reference (docs/ALANINE_1D_CV.md).

Everything that is not the CV dimension is the accepted 2-D engine's code or convention:
force field, BAOAB, dt, gamma, temperature, dtype, IUPAC angles, the initial ensemble, the
fixed-consumption RNG streams, the full-state kill-and-clone step (imported from
``core2d_ala``), the age-aware genealogy window, the non-finite containment.

Estimator (1-D, periodic, cell-centred 97-bin grid, ``alkanes.periodic``):
  * ``histogram``  adaptive-box P0: per bin the smallest periodic box of ``2k+1`` bins
                   (k <= abf_hist_levels) whose count reaches ``abf_min_count`` gives
                   ``M_k / C_k``; no such box -> zero force.  The walker feels its OWN bin's
                   value (textbook piecewise-constant bias); the reported PMF is the exact
                   integral of that piecewise-constant force (trapezoid between bin centres
                   is exact for it) with the circular mean of F' removed (periodicity).
  * ``kernel``     the alkanes/alanine wrapped-Gaussian ratio at ``abf_bandwidth``, trust on
                   the smoothed count, linearly interpolated (the alkanes convention).
The mean force itself is den Otter's 1-D instantaneous force for a single dihedral
(``alkanes.cv.DihedralCV``): NOT the phi-component of the 2-D vector mean force, because the
two dihedrals share three atoms and their gradients are not orthogonal.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from alkanes import periodic as per
from alkanes.core import _recentered_clipped_score
from alkanes.cv import DihedralCV

from .core2d_ala import EPS, _ancestor_stats_t, _birth_death_ala, _dihedral_iupac_t
from .cv2d import rb_to_iupac
from .dynamics import KB, SeedFailure

TWO_PI = 2.0 * math.pi
METHODS = ("abf", "fr_uniform")
CV_ATOMS = {"phi": (4, 6, 8, 14), "psi": (6, 8, 14, 16)}


class BackboneCV1D(DihedralCV):
    """One backbone dihedral with IUPAC values; geometry inherited unchanged."""

    def __init__(self, name):
        if name not in CV_ATOMS:
            raise ValueError(f"cv must be one of {tuple(CV_ATOMS)}, got {name!r}")
        super().__init__(CV_ATOMS[name])
        self.name = name
        self.hidden = "psi" if name == "phi" else "phi"

    def value(self, q):
        return rb_to_iupac(super().value(q))

    def _idx(self, device):
        # cached DEVICE index tensor: indexing with the Python tuple builds a CPU index tensor on
        # every call, which is a host->device copy and is illegal inside CUDA-graph capture (the
        # 2-D FastBackboneCV2D caches its indices for the same reason)
        t = self.__dict__.get("_idx_t")
        if t is None or t.device != device:
            t = torch.tensor(self.atoms, dtype=torch.long, device=device)
            self.__dict__["_idx_t"] = t
        return t

    def geometry(self, q):
        """As ``DihedralCV.geometry`` (same arithmetic), with cached index tensors."""
        from alkanes.cv import _grad_phi4, _hess_phi4
        B, n_atoms, _ = q.shape
        idx = self._idx(q.device)
        sub = q.index_select(1, idx).reshape(B, 12).detach()
        g = _grad_phi4(sub)
        H = _hess_phi4(sub)
        gg = (g * g).sum(-1)
        lap = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)
        gHg = torch.einsum("bi,bij,bj->b", g, H, g)
        div_v = lap / gg.clamp_min(EPS) - 2.0 * gHg / gg.clamp_min(EPS) ** 2
        phi = self.value(q)
        grad_full = q.new_zeros(B, n_atoms, 3)
        grad_full.index_copy_(1, idx, g.reshape(B, 4, 3))
        return phi.detach(), grad_full.detach(), div_v.detach()


@dataclass(frozen=True)
class Ala1DSimConfig:
    cv: str = "phi"
    # --- dynamics (frozen physical model, = 2-D) ---
    dt: float = 0.001
    gamma: float = 1.0
    temperature: float = 300.0
    n_steps: int = 100_000
    n_replicas: int = 2048
    save_every: int = 1_000
    rng_seed: int = 20260903
    # --- estimator ---
    n_grid: int = 97
    abf_estimator: str = "histogram"
    abf_hist_levels: int = 4
    abf_hist_fixed: bool = False
    abf_min_count: float = 800.0
    abf_bandwidth: float = 0.08
    abf_force_clip: float = 200.0
    abf_warmup_steps: int = 5_000
    # --- Fisher--Rao (object B, = 2-D) ---
    kde_bandwidth: float = 0.15
    fr_rate: float = 0.02
    score_clip: float = 2.0
    fr_start_steps: int = 0
    fr_every: int = 500
    max_event_fraction: float = 0.05
    lineage_reset_steps: int = 6_000
    store_accumulators: bool = False

    _HASH_ALWAYS_DROP = ("store_accumulators",)

    def config_hash(self):
        import hashlib, json
        d = asdict(self)
        for k in self._HASH_ALWAYS_DROP:
            d.pop(k)
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- estimators
def box_sum1(x, k):
    """Periodic sum over offsets ``-k..k`` along the last dim (box of ``2k+1`` bins)."""
    out = x
    for s in range(1, int(k) + 1):
        out = out + torch.roll(x, s, dims=-1) + torch.roll(x, -s, dims=-1)
    return out


def adaptive_box_mean_force_1d(fs, cs, levels, min_count, fixed=False):
    """1-D analogue of ``core2d_ala.adaptive_box_mean_force``: ``(g, den, level)``."""
    g = torch.zeros_like(cs)
    den = torch.zeros_like(cs)
    chosen = torch.zeros_like(cs, dtype=torch.bool)
    level = torch.full_like(cs, -1, dtype=torch.long)
    C = cs
    for k in ([int(levels)] if fixed else range(int(levels) + 1)):
        C, M = box_sum1(cs, k), box_sum1(fs, k)
        ok = (~chosen) & (C >= min_count)
        g = torch.where(ok, M / C.clamp_min(EPS), g)
        den = torch.where(ok, C, den)
        level = torch.where(ok, torch.full_like(level, int(k)), level)
        chosen = chosen | ok
    return g, torch.where(chosen, den, C), level


def mean_force_1d(fs, cs, sim, K_abf):
    """``(profile (R,n), den, level|None)`` with the engine's trust rule applied."""
    if sim.abf_estimator == "histogram":
        g, den, level = adaptive_box_mean_force_1d(fs, cs, sim.abf_hist_levels,
                                                   sim.abf_min_count, sim.abf_hist_fixed)
    elif sim.abf_estimator == "kernel":
        den = per.smooth(cs, K_abf)
        g = torch.where(den > EPS, per.smooth(fs, K_abf) / den.clamp_min(EPS), torch.zeros_like(den))
        level = None
    else:
        raise ValueError(f"unknown abf_estimator {sim.abf_estimator!r}")
    trust = den >= sim.abf_min_count
    return torch.where(trust, g, torch.zeros_like(g)), den, level, trust


def bias_value_at(profile, grid, dphi, phi, estimator):
    """Applied CV-space bias at the walkers: own bin (histogram) or linear interpolation (kernel)."""
    if estimator == "histogram":
        n = grid.shape[0]
        idx = torch.floor((phi + math.pi) / dphi).long() % n
        return torch.gather(profile, -1, idx)
    return per.circular_interp(profile, grid, phi)


# ----------------------------------------------------------------------------- sampler
def run_sampler_ala1d(method, tff, cv: BackboneCV1D, sim: Ala1DSimConfig, seeds, init_positions,
                      basin_labels, device, dtype=torch.float64, dump_dir=None, force_fn=None,
                      rare_basin=2, verbose=True):
    """``R = len(seeds)`` matched-seed replicas of ``method`` with the 1-D CV ``cv``."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    base_cv = getattr(cv, "cv", cv)            # GraphedCV wraps the CV; read names from the base
    if getattr(base_cv, "name", None) != sim.cv:
        raise ValueError(f"cv object is {getattr(base_cv, 'name', None)!r} but the config says {sim.cv!r}")
    is_fr = method == "fr_uniform"
    R, N = len(seeds), sim.n_replicas
    A = tff.n_atoms
    beta = 1.0 / (KB * sim.temperature)
    n = sim.n_grid
    grid, dphi = per.periodic_grid(n, device=device, dtype=dtype)
    K_abf = per.wrapped_gaussian_kernel_matrix(grid, sim.abf_bandwidth)
    K_kde = per.wrapped_gaussian_kernel_matrix(grid, sim.kde_bandwidth)
    q_uniform = torch.full((R, n), 1.0 / TWO_PI, device=device, dtype=dtype)
    hidden_atoms = CV_ATOMS[base_cv.hidden]
    cv_is_phi = base_cv.name == "phi"

    gen_dyn = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    gens_fr = [torch.Generator(device=device).manual_seed(int(sim.rng_seed) + 987654321 + 1000 * r)
               for r in range(R)]

    q = torch.as_tensor(init_positions, device=device, dtype=dtype).reshape(R, N, A, 3).contiguous()
    m = tff.masses.reshape(-1, 1)
    kT = KB * sim.temperature
    sigma_v = math.sqrt(kT) / m.sqrt()
    c1 = math.exp(-sim.gamma * sim.dt)
    c2 = math.sqrt(1.0 - c1 * c1)
    v = torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype) * sigma_v
    phys = force_fn or (lambda x: tff.forces(x))

    fs = torch.zeros(R, n, device=device, dtype=dtype)
    cs = torch.zeros(R, n, device=device, dtype=dtype)
    anc = torch.arange(N, device=device).expand(R, N).contiguous()
    anc_age = anc.clone()
    n_nonfinite = torch.zeros(R, device=device, dtype=torch.long)
    n_clip = torch.zeros((), device=device, dtype=torch.long)
    n_force_eval = torch.zeros((), device=device, dtype=torch.long)
    n_events = torch.zeros(R, device=device, dtype=torch.long)
    Tsum = torch.zeros((), device=device, dtype=dtype); n_T = 0
    n_basins = int(basin_labels.max().item()) + 1
    first_hit = torch.full((R, n_basins), -1, device=device, dtype=torch.long)
    prev_basin = None
    trans = torch.zeros(R, n_basins, n_basins, device=device, dtype=torch.long)
    score_std = torch.zeros(R, device=device, dtype=dtype)
    score_absmax = torch.zeros(R, device=device, dtype=dtype)
    hist_untrusted = torch.full((R,), float("nan"), device=device, dtype=dtype)
    hist_level_mean = torch.full((R,), float("nan"), device=device, dtype=dtype)
    diag = {k: [] for k in ("steps", "times", "pmf", "mean_force", "counts", "basin_frac",
                            "ess_perm", "ess_age", "n_unique", "wmax", "wmax_rare", "ess_age_rare",
                            "events_cum", "trust_frac", "clip_frac", "temperature",
                            "score_std", "score_absmax", "marg_hist", "kl_uniform",
                            "joint_hist", "hist_level_mean", "hist_untrusted_frac")}
    if sim.store_accumulators:
        diag["acc_fs"] = []; diag["acc_cs"] = []
    t0 = time.perf_counter()

    qf = q.reshape(R * N, A, 3)
    f_phys = phys(qf).reshape(R, N, A, 3)
    floc, phi, gfull = cv.local_mean_force(qf, f_phys.reshape(R * N, A, 3), beta)
    n_force_eval += R * N

    def _abort(step, why):
        import os
        path = None
        if dump_dir is not None:
            os.makedirs(dump_dir, exist_ok=True)
            bad = int(torch.argmax(n_nonfinite).item())
            path = os.path.join(dump_dir, f"FAILED_{method}_{sim.cv}_seed{bad}_step{step}.npz")
            np.savez_compressed(path, step=step, method=method, seed_index=bad, reason=why,
                                q=q[bad].detach().cpu().numpy(), fs=fs[bad].cpu().numpy(),
                                cs=cs[bad].cpu().numpy(), n_nonfinite=n_nonfinite.cpu().numpy())
        raise SeedFailure(f"{why} in method={method} cv={sim.cv} at step {step}; dump -> {path}",
                          int(torch.argmax(n_nonfinite).item()), step, path)

    def _bias(profile, phi_t, gfull_t, ramp):
        z = torch.nan_to_num(phi_t.reshape(R, N), 0.0, 0.0, 0.0)
        c = ramp * bias_value_at(profile, grid, dphi, z, sim.abf_estimator)
        nclip = (c.abs() > sim.abf_force_clip).sum()
        c = c.clamp(-sim.abf_force_clip, sim.abf_force_clip)
        cart = (c.reshape(R * N)[:, None, None] * gfull_t).reshape(R, N, A, 3)
        return cart, z, nclip

    profile = torch.zeros(R, n, device=device, dtype=dtype)
    trust_frac = 0.0
    for step in range(sim.n_steps + 1):
        okm = (torch.isfinite(floc) & torch.isfinite(phi)).reshape(R, N)
        okm = okm & torch.isfinite(f_phys.reshape(R, N, -1)).all(-1)
        n_nonfinite += (~okm).sum(1)
        okd = okm.to(dtype)
        f1 = torch.nan_to_num(floc.reshape(R, N), 0.0, 0.0, 0.0) * okd
        z = torch.nan_to_num(phi.reshape(R, N), 0.0, 0.0, 0.0)
        fs += per.bin_sum(z, f1, n)
        cs += per.bin_sum(z, okd, n)
        profile, den, level_map, trust = mean_force_1d(fs, cs, sim, K_abf)
        trust_frac = float(trust.to(dtype).mean())
        if level_map is not None:
            tr = level_map >= 0
            hist_untrusted = 1.0 - tr.to(dtype).mean(-1)
            hist_level_mean = ((level_map.to(dtype) * tr.to(dtype)).sum(-1)
                               / tr.to(dtype).sum(-1).clamp_min(1.0))
        ramp = min(1.0, step / max(sim.abf_warmup_steps, 1))
        bias_cart, z, nc = _bias(profile, phi, gfull, ramp)
        n_clip += nc

        # ---- basin bookkeeping on the FULL (phi, psi) grid: the hidden angle is measured ----
        hid = _dihedral_iupac_t(q.reshape(R * N, A, 3), hidden_atoms).reshape(R, N)
        a_phi, a_psi = (z, hid) if cv_is_phi else (hid, z)
        bi = torch.floor((a_phi + math.pi) / dphi).long().clamp(0, n - 1)
        bj = torch.floor((a_psi + math.pi) / dphi).long().clamp(0, n - 1)
        cur = basin_labels[bi, bj]
        for k in range(n_basins):
            seen = (cur == k).any(1)
            fh = first_hit[:, k]
            first_hit[:, k] = torch.where((fh < 0) & seen, torch.full_like(fh, step), fh)
        if prev_basin is not None:
            ch = (cur != prev_basin) & (cur >= 0) & (prev_basin >= 0)
            lin = (torch.arange(R, device=device)[:, None].expand(R, N) * n_basins * n_basins
                   + prev_basin.clamp_min(0) * n_basins + cur.clamp_min(0))
            trans.view(-1).scatter_add_(0, lin.reshape(-1), ch.reshape(-1).long())
        prev_basin = cur

        if step % sim.save_every == 0 or step == sim.n_steps:
            if int(n_nonfinite.sum().item()) > 0:
                _abort(step, "non-finite local mean force / CV / physical force")
            ess_p, nuq, wmx = _ancestor_stats_t(anc, N)
            ess_a, _, _ = _ancestor_stats_t(anc_age, N)
            in_rare = (cur == int(rare_basin))
            frac = torch.stack([(cur == k).to(dtype).mean(1) for k in range(n_basins)], -1)
            wmax_rare = torch.zeros(R, device=device, dtype=torch.float64)
            ess_a_rare = torch.zeros(R, device=device, dtype=torch.float64)
            for r in range(R):
                sel = anc_age[r][in_rare[r]]
                if sel.numel() > 0:
                    cnt = torch.bincount(sel, minlength=N).to(torch.float64)
                    w = cnt / cnt.sum()
                    wmax_rare[r] = w.max(); ess_a_rare[r] = 1.0 / (w * w).sum().clamp_min(EPS)
            pmf = per.free_energy_from_mean_force(profile, grid, dphi)
            diag["steps"].append(step); diag["times"].append(step * sim.dt)
            diag["pmf"].append(pmf.detach().cpu().numpy())
            diag["mean_force"].append(profile.detach().cpu().numpy())
            diag["counts"].append(cs.detach().to(torch.float32).cpu().numpy())
            diag["basin_frac"].append(frac.cpu().numpy())
            diag["ess_perm"].append((ess_p / N).cpu().numpy()); diag["ess_age"].append((ess_a / N).cpu().numpy())
            diag["ess_age_rare"].append((ess_a_rare / N).cpu().numpy())
            diag["n_unique"].append(nuq.cpu().numpy()); diag["wmax"].append(wmx.cpu().numpy())
            diag["wmax_rare"].append(wmax_rare.cpu().numpy()); diag["events_cum"].append(n_events.cpu().numpy())
            diag["trust_frac"].append(trust_frac)
            diag["clip_frac"].append(float(n_clip.item()) / max(float(n_force_eval.item()), 1.0))
            diag["temperature"].append(float(Tsum / max(n_T, 1)) if n_T else float("nan"))
            diag["score_std"].append(score_std.cpu().numpy()); diag["score_absmax"].append(score_absmax.cpu().numpy())
            p_bin = per.bin_counts(z, n); p_bin = p_bin / p_bin.sum(-1, keepdim=True).clamp_min(1.0)
            kl_u = (p_bin * (torch.log(p_bin.clamp_min(EPS)) + math.log(float(n)))).sum(-1)
            diag["marg_hist"].append(p_bin.to(torch.float32).cpu().numpy())
            diag["kl_uniform"].append(kl_u.cpu().numpy())
            jh = torch.zeros(R, n * n, device=device, dtype=torch.float32)
            jh.scatter_add_(1, (bi * n + bj).reshape(R, N), torch.ones(R, N, device=device, dtype=torch.float32))
            diag["joint_hist"].append(jh.reshape(R, n, n).cpu().numpy())
            diag["hist_level_mean"].append(hist_level_mean.cpu().numpy())
            diag["hist_untrusted_frac"].append(hist_untrusted.cpu().numpy())
            if sim.store_accumulators:
                diag["acc_fs"].append(fs.to(torch.float32).cpu().numpy()); diag["acc_cs"].append(cs.to(torch.float32).cpu().numpy())
            Tsum = torch.zeros((), device=device, dtype=dtype); n_T = 0
        if step == sim.n_steps:
            break

        # ---- BAOAB (= 2-D engine) ----
        v = v + (0.5 * sim.dt) * (f_phys + bias_cart) / m
        q = q + (0.5 * sim.dt) * v
        v = c1 * v + c2 * torch.randn(v.shape, generator=gen_dyn, device=device, dtype=dtype) * sigma_v
        q = q + (0.5 * sim.dt) * v
        qf = q.reshape(R * N, A, 3)
        f_phys = phys(qf).reshape(R, N, A, 3)
        n_force_eval += R * N
        floc, phi, gfull = cv.local_mean_force(qf, f_phys.reshape(R * N, A, 3), beta)
        bias_new, _, nc2 = _bias(profile, phi, gfull, ramp)
        n_clip += nc2
        v = v + (0.5 * sim.dt) * (f_phys + bias_new) / m
        Tsum = Tsum + (m * v * v).sum() / (3.0 * R * N * A * KB); n_T += 1

        if sim.lineage_reset_steps > 0 and (step + 1) % sim.lineage_reset_steps == 0:
            anc_age = torch.arange(N, device=device).expand(R, N).contiguous()

        # ---- Fisher--Rao birth--death on the 1-D marginal (uniform target on the circle) ----
        if is_fr:
            nxt = step + 1
            if nxt >= sim.fr_start_steps and (nxt - sim.fr_start_steps) % sim.fr_every == 0:
                z1 = torch.nan_to_num(phi.reshape(R, N), 0.0, 0.0, 0.0)
                p_hat = per.kde_marginal(z1, K_kde, n, dphi)
                p_at = per.circular_interp(p_hat, grid, z1)
                q_at = per.circular_interp(q_uniform, grid, z1)
                log_ratio = torch.log(p_hat.clamp_min(EPS)) - torch.log(q_uniform.clamp_min(EPS))
                kl = (p_hat * log_ratio).sum(-1) * dphi
                raw = torch.log(p_at.clamp_min(EPS)) - torch.log(q_at.clamp_min(EPS)) - kl[:, None]
                score = _recentered_clipped_score(raw, sim.score_clip)
                score_std = score.std(1); score_absmax = score.abs().amax(1)
                q, v, f_phys, anc, anc_age, ne = _birth_death_ala(
                    q, v, f_phys, score, anc, anc_age, gens_fr, sim, sigma_v)
                n_events += ne.to(n_events.device)
                if int(ne.sum()) > 0:
                    qf = q.reshape(R * N, A, 3)
                    floc, phi, gfull = cv.local_mean_force(qf, f_phys.reshape(R * N, A, 3), beta)

    if int(n_nonfinite.sum().item()) > 0:
        _abort(sim.n_steps, "non-finite detected")
    wall = time.perf_counter() - t0
    out = dict(method=method, cv=sim.cv, seeds=np.asarray(seeds), n_replicas=N, n_steps=sim.n_steps,
               grid=grid.cpu().numpy(), dphi=float(dphi), n_grid=n, rare_basin=int(rare_basin),
               final_pmf=per.free_energy_from_mean_force(profile, grid, dphi).cpu().numpy(),
               final_mean_force=profile.cpu().numpy(), final_fs=fs.cpu().numpy(), final_cs=cs.cpu().numpy(),
               first_hit=first_hit.cpu().numpy(), trans_matrix=trans.cpu().numpy(),
               total_events=n_events.cpu().numpy(),
               clip_fraction=float(n_clip.item()) / max(float(n_force_eval.item()), 1.0),
               force_evaluations=int(n_force_eval.item()),
               aggregate_simulated_ps=float(sim.n_steps * sim.dt * R * N),
               wall_seconds=wall, ms_per_step=wall / max(sim.n_steps, 1) * 1e3,
               peak_cuda_gib=(torch.cuda.max_memory_allocated() / 2 ** 30 if device != "cpu" else 0.0),
               n_nonfinite=n_nonfinite.cpu().numpy(), config_hash=sim.config_hash())
    for k in diag:
        out[k] = np.asarray(diag[k])
    if verbose:
        print(f"  {method:10s} cv={sim.cv} R={R} N={N}: {wall:.1f}s  {out['ms_per_step']:.2f} ms/step  "
              f"events={int(n_events.sum())}  clip={out['clip_fraction']:.2e}  peak={out['peak_cuda_gib']:.2f} GiB",
              flush=True)
    return out


# ----------------------------------------------------------------------------- reference
def marginal_reference_1d(F2, kT, axis_keep, mask=None):
    """``F1(xi) = -kT log sum_hidden exp(-F2/kT)`` over the OTHER axis of the 97 x 97 reference;
    non-finite (unvisited) cells contribute nothing; ``mask`` (bool, 97 x 97) restricts the sum
    (e.g. the phi < 0 half, for the visited-side conditional reference).  ``axis_keep`` 0 = phi."""
    F = np.asarray(F2, float)
    ok = np.isfinite(F) if mask is None else (np.isfinite(F) & mask)
    w = np.where(ok, np.exp(-(np.where(ok, F, 0.0) - np.nanmin(F[np.isfinite(F)])) / kT), 0.0)
    s = w.sum(axis=1 - axis_keep)
    F1 = np.where(s > 0, -kT * np.log(np.where(s > 0, s, 1.0)), np.inf)
    return F1 - np.nanmin(F1[np.isfinite(F1)])
