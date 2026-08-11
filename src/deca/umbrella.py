"""Umbrella sampling + MBAR reference for the deca-alanine end-to-end distance.

Why umbrella and not a long ABF run
-----------------------------------
The reference has to be independent of the method under test.  A converged ABF calculation
would share the estimator, the bias channel and the failure modes of the arms it is meant to
score, so a systematic error in the estimator would cancel out of the comparison and hide
itself.  Umbrella sampling plus MBAR shares only the force field and the integrator.

Convention
----------
Matched to the R15 distance-CV reference in :mod:`alkanes.reference_cv`, so the two studies are
directly comparable:

    p_ref(R)  =  marginal density of R under the configurational Boltzmann measure
    F_ref(R)  =  -beta^-1 log p_ref(R) + C

No Jacobian is divided out.  This is exactly the free energy whose derivative the den Otter
local mean force in :mod:`alkanes.distance_cv` estimates -- its ``-beta^-1 div v`` term is what
accounts for the geometry -- so reference and estimator agree by construction rather than by
luck.

Seeding
-------
Windows are *not* seeded from the helix alone.  Vacuum deca-alanine's difficulty is hidden
conformational structure, and a reference seeded from one basin would inherit precisely the
bias under test.  Each window is filled from a deliberately diverse pool spanning helical,
extended and collapsed backbones, then pulled to its restraint centre under the umbrella
potential itself.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from alanine.dynamics import BAOAB, check_finite, make_seed_streams
from alkanes.distance_cv import DistanceCV, dist_bias_force

from . import system as dsys
from .labels import DecaLabels

KB = 0.008314462618          # kJ/mol/K


@dataclass
class UmbrellaConfig:
    """Frozen per §6.3 of ``docs/V2_PREREGISTRATION.md``, as revised by **Amendment 1**.

    The umbrella centre range deliberately BRACKETS the evaluation domain.  With centres ending
    at ``R_hi`` exactly, the steep end of the PMF displaces the last window down by ~0.105 nm
    and the top evaluation bin ends up resting on one window's tail instead of sitting between
    two windows -- a structural asymmetry in the estimator, not a tuning preference.
    """
    n_windows: int = 96
    R_lo: float = 1.20                     # nm, evaluation domain (scoring)
    R_hi: float = 3.60
    win_lo: float = 1.15                   # nm, umbrella centre range (sampling)
    win_hi: float = 3.70
    k_umbrella: float = 3200.0             # kJ/mol/nm^2  (window sd ~0.037 vs spacing 0.028)
    n_rep: int = 32                        # replicas per window
    dt: float = 0.001                      # ps
    gamma: float = 1.0                     # ps^-1
    temperature: float = 300.0             # K
    n_pull_steps: int = 20_000             # 20 ps, restrained pull to the window centre
    n_equil_steps: int = 200_000           # 200 ps discarded
    n_prod_steps: int = 4_000_000          # 4 ns collected
    sample_every: int = 500                # 0.5 ps -> 8000 samples per replica
    n_grid: int = 129                      # ODD: no Nyquist row
    kde_bandwidth: float = 0.03            # nm
    rng_seed: int = 20260811

    @property
    def beta(self):
        return 1.0 / (KB * self.temperature)

    def config_hash(self):
        import hashlib, json
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def window_centers(cfg: UmbrellaConfig):
    """Uniformly spaced restraint centres over ``[win_lo, win_hi]``, which brackets the
    evaluation domain ``[R_lo, R_hi]`` -- see :class:`UmbrellaConfig`."""
    return np.linspace(cfg.win_lo, cfg.win_hi, cfg.n_windows)


def diverse_pool(n, cfg: UmbrellaConfig, rng):
    """A structurally diverse starting pool: helical, extended, polyproline-like and mixed.

    Uniform ``(phi, psi)`` per structure, plus a per-residue jitter so replicas within a window
    do not start identical.  Every member passes the builder gate before it is used.
    """
    presets = [(-57.0, -47.0),      # right-handed alpha helix
               (-150.0, 150.0),     # extended / beta
               (-75.0, 145.0),      # polyproline II
               (-90.0, 0.0),        # bridge / compact
               (-140.0, 60.0),      # mixed
               (60.0, 40.0)]        # left-handed
    out = []
    for i in range(n):
        phi, psi = presets[i % len(presets)]
        phi += float(rng.normal(0.0, 8.0))
        psi += float(rng.normal(0.0, 8.0))
        out.append(dsys.build_helix(phi, psi, n_res=dsys.N_RES))
    return np.stack(out)


def _restraint_force(grad_full, R, centers, k):
    """Cartesian force from ``V = 0.5 k (R - R_c)^2``, i.e. ``-k (R - R_c) grad R``."""
    return dist_bias_force(grad_full, -k * (R - centers))


def run_umbrella(engine, cfg: UmbrellaConfig, build_index=0, device="cuda",
                 dtype=torch.float64, verbose=True, progress_every=200_000):
    """Run all windows in one batch.  Returns a dict of samples and diagnostics.

    Batch layout is ``(n_windows * n_rep, 112, 3)`` -- one process, one GPU, one noise stream,
    which is the v1 design and the reason arm-to-arm comparisons stay paired.
    """
    n_w, n_r = cfg.n_windows, cfg.n_rep
    B = n_w * n_r
    beta = cfg.beta
    centers_np = window_centers(cfg)
    centers = torch.as_tensor(np.repeat(centers_np, n_r), device=device, dtype=dtype)

    i_cv, j_cv = dsys.terminal_carbonyls(dsys.N_RES)
    cv = DistanceCV(i_cv, j_cv)
    labels = DecaLabels(device=device, dtype=dtype)

    rng = np.random.default_rng(cfg.rng_seed + 7919 * build_index)
    X0 = diverse_pool(B, cfg, rng)
    q = torch.as_tensor(X0, device=device, dtype=dtype).contiguous()

    gen = torch.Generator(device=device).manual_seed(int(cfg.rng_seed) + 104729 * build_index)
    integ = BAOAB(engine.masses, dt=cfg.dt, gamma=cfg.gamma, temperature=cfg.temperature,
                  force_fn=engine.forces, device=device, dtype=dtype)
    v = integ.maxwell((B, engine.n_atoms, 3), gen, device, dtype)
    f = engine.forces(q)

    def step(q, v, f, k_scale=1.0):
        R, grad_full, _ = cv.geometry(q)
        fr = _restraint_force(grad_full, R, centers, cfg.k_umbrella * k_scale)
        v = v + (0.5 * cfg.dt) * (f + fr) / integ.m
        q = q + (0.5 * cfg.dt) * v
        v = integ.c1 * v + integ.c2 * torch.randn(v.shape, generator=gen, device=device,
                                                  dtype=dtype) * integ.sigma
        q = q + (0.5 * cfg.dt) * v
        f = engine.forces(q)
        R2, grad2, _ = cv.geometry(q)
        fr2 = _restraint_force(grad2, R2, centers, cfg.k_umbrella * k_scale)
        v = v + (0.5 * cfg.dt) * (f + fr2) / integ.m
        return q, v, f

    t0 = time.perf_counter()
    # --- pull: ramp the restraint so a structure far from its centre is not slammed into it ---
    for s in range(cfg.n_pull_steps):
        q, v, f = step(q, v, f, k_scale=min(1.0, (s + 1) / max(cfg.n_pull_steps * 0.5, 1)))
    check_finite(cfg.n_pull_steps, ("q", q[None]), ("v", v[None]), tag="deca_umbrella_pull")

    R_after_pull = cv.value(q).cpu().numpy()
    pull_err = np.abs(R_after_pull - np.repeat(centers_np, n_r))
    # Amendment 1: a hard pull is exactly the operation that can flip a stereocentre or
    # isomerise a peptide bond, so screen here rather than discover it in the samples.
    ok_pull, rep_pull = dsys.validate_thermal(q.cpu().numpy(), dsys.N_RES)
    if verbose:
        print(f"  pull done ({time.perf_counter()-t0:.0f}s): |R - R_c| median "
              f"{np.median(pull_err):.4f} max {pull_err.max():.4f} nm; "
              f"structural {int(ok_pull.sum())}/{B} pass "
              f"(cis {int(rep_pull['n_fail_cis'])}, chirality {int(rep_pull['n_fail_chirality'])})")

    # --- equilibration (discarded) ---
    for _ in range(cfg.n_equil_steps):
        q, v, f = step(q, v, f)
    check_finite(cfg.n_equil_steps, ("q", q[None]), ("v", v[None]), tag="deca_umbrella_equil")

    ok_equil, rep_equil = dsys.validate_thermal(q.cpu().numpy(), dsys.N_RES)
    keep = ok_pull & ok_equil
    if verbose:
        print(f"  equil done: structural {int(ok_equil.sum())}/{B} pass; "
              f"{int((~keep).sum())} replicas excluded from the reference")
    if keep.sum() < 0.9 * B:
        raise RuntimeError(
            f"structural screening excluded {int((~keep).sum())}/{B} replicas (>10%); the pull "
            "or the restraint stiffness is damaging structures, do not build a reference on this")

    # --- production ---
    xi_s, y_s, lab_s = [], [], {k: [] for k in ("n_hbonds", "alpha_frac", "rg", "ca_rmsd_helix")}
    n_sample = 0
    for s in range(cfg.n_prod_steps):
        q, v, f = step(q, v, f)
        if (s + 1) % cfg.sample_every == 0:
            xi_s.append(cv.value(q).to(torch.float32).cpu().numpy())
            L = labels.all_labels(q)
            y_s.append(L["y"].to(torch.int8).cpu().numpy())
            for k in lab_s:
                lab_s[k].append(L[k].to(torch.float32).cpu().numpy())
            n_sample += 1
        if verbose and progress_every and (s + 1) % progress_every == 0:
            el = time.perf_counter() - t0
            frac = (s + 1) / cfg.n_prod_steps
            print(f"    prod {100*frac:5.1f}%  {el/60:6.1f} min elapsed, "
                  f"~{el/frac*(1-frac)/60:6.1f} min left", flush=True)
    check_finite(cfg.n_prod_steps, ("q", q[None]), ("v", v[None]), tag="deca_umbrella_prod")

    ok_thermal, rep_thermal = dsys.validate_thermal(q.cpu().numpy(), dsys.N_RES)

    return dict(
        xi=np.stack(xi_s),                                   # (n_sample, B)
        y=np.stack(y_s),
        keep=keep,                                           # (B,) structural screening mask
        n_excluded=int((~keep).sum()),
        n_fail_cis_pull=int(rep_pull["n_fail_cis"]),
        n_fail_chirality_pull=int(rep_pull["n_fail_chirality"]),
        n_fail_cis_equil=int(rep_equil["n_fail_cis"]),
        n_fail_chirality_equil=int(rep_equil["n_fail_chirality"]),
        centers=centers_np, n_windows=n_w, n_rep=n_r, n_sample=n_sample,
        build_index=build_index, config=asdict(cfg), config_hash=cfg.config_hash(),
        pull_error_nm=pull_err, runtime_seconds=time.perf_counter() - t0,
        final_thermal_pass=int(np.sum(ok_thermal)), final_thermal_n=int(ok_thermal.size),
        final_n_cis=int(np.sum(rep_thermal["n_cis_bonds"])),
        final_min_chirality=float(np.min(rep_thermal["min_chirality"])),
        **{k: np.stack(v) for k, v in lab_s.items()})


# --------------------------------------------------------------------------- MBAR
def mbar_weights(xi, centers, n_rep, k_umbrella, beta, subsample_stride=1, keep=None,
                 aux=None):
    """Unbiased-ensemble MBAR weights for every sample.

    ``xi`` is ``(n_sample, n_windows * n_rep)`` with columns ordered window-major.  The unbiased
    state is appended as an extra state with zero reduced potential and zero samples, so
    ``weights()[:, -1]`` is exactly the reweighting to the Boltzmann ensemble.

    ``keep`` is the structural screening mask over replicas (Amendment 1).  Excluded replicas
    are dropped and ``N_k`` is rebuilt per window from what survives, so a window that lost a
    replica is not silently credited with samples it does not have.  ``aux`` is an optional dict
    of ``(n_sample, B)`` arrays (structural labels) reordered the same way, returned alongside.
    """
    from pymbar import MBAR

    n_s, B = xi.shape
    n_w = len(centers)
    if keep is None:
        keep = np.ones(B, dtype=bool)
    keep = np.asarray(keep, dtype=bool).reshape(n_w, n_rep)

    x = xi[::subsample_stride]                                # (m, B)
    m = x.shape[0]
    x_w = x.reshape(m, n_w, n_rep)

    parts, N_k = [], np.zeros(n_w + 1, dtype=int)
    aux_parts = {k: [] for k in (aux or {})}
    for w in range(n_w):
        sel = keep[w]
        block = x_w[:, w, sel].reshape(-1)                    # (m * n_kept,)
        parts.append(block)
        N_k[w] = block.size
        for k, arr in (aux or {}).items():
            a = arr[::subsample_stride].reshape(m, n_w, n_rep)
            aux_parts[k].append(a[:, w, sel].reshape(-1))
    x_all = np.concatenate(parts)
    N_k[-1] = 0

    # u_kn[k, n] = beta * 0.5 * k_u * (xi_n - R_k)^2 ; unbiased row is zero
    u_kn = np.zeros((n_w + 1, x_all.size))
    d = x_all[None, :] - centers[:, None]
    u_kn[:n_w] = beta * 0.5 * k_umbrella * d ** 2

    mbar = MBAR(u_kn, N_k, solver_protocol="robust")
    W = mbar.weights()                                        # (N_total, n_w + 1)
    out_aux = {k: np.concatenate(v) for k, v in aux_parts.items()}
    return x_all, W[:, -1], mbar, out_aux


def pmf_from_weights(xi_all, w, cfg: UmbrellaConfig, device="cpu", smooth_F_bandwidth=0.0):
    """``F_ref = -beta^-1 log p_ref`` on the frozen grid, from binned MBAR weights.

    **Smoothing is applied to F, never to p.**  The R15 reference smooths the *density* with a
    reflected KDE, which is correct there because that PMF spans a few kT.  This one does not:
    the deca-alanine end-to-end PMF spans roughly 185 kJ/mol, about **74 kT**, so ``p`` varies
    by a factor of order ``e^74``.  Convolving a density with that dynamic range makes every
    high-``R`` bin a leakage artifact of the peak near the helix rather than a measurement --
    a smoothed ``p`` would look perfectly smooth and be meaningless above ~2.5 nm.  Working in
    ``F`` keeps the operation linear in the quantity that is actually of order tens of kT.

    ``smooth_F_bandwidth = 0`` (the default) applies no smoothing at all, which is what a
    reference should do unless there is a stated reason.

    Returns ``(grid, dz, p, F, counts)``; ``counts`` is the raw per-bin sample count, so a bin
    resting on too few samples can be identified rather than silently trusted.
    """
    from alkanes import interval as iv

    grid, dz = iv.interval_grid(cfg.n_grid, cfg.R_lo, cfg.R_hi, device=device,
                                dtype=torch.float64)
    z = torch.as_tensor(np.asarray(xi_all, dtype=np.float64), device=device)[None]
    wt = torch.as_tensor(np.asarray(w, dtype=np.float64), device=device)[None]
    hist = iv.bin_sum(z, wt, cfg.n_grid, cfg.R_lo, cfg.R_hi)
    counts = iv.bin_counts(z, cfg.n_grid, cfg.R_lo, cfg.R_hi)

    p = hist / (hist.sum(-1, keepdim=True).clamp_min(1e-300) * dz)
    F = -(1.0 / cfg.beta) * torch.log(p.clamp_min(1e-300))
    if smooth_F_bandwidth and smooth_F_bandwidth > 0:
        K = iv.gaussian_kernel_matrix(grid, smooth_F_bandwidth)
        Fs = iv.smooth(F, K) / iv.smooth(torch.ones_like(F), K)
        F = torch.where(torch.isfinite(F), Fs, F)
    F = F - F[torch.isfinite(F)].mean()
    return (grid.cpu().numpy(), float(dz), p[0].cpu().numpy(), F[0].cpu().numpy(),
            counts[0].cpu().numpy())
