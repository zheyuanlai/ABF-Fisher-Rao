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


def relax_pool(engine, x, n_steps=1500, lr=2.0e-5, max_disp=0.004,
               max_energy_above_min=500.0, max_force=1.0e5):
    """Capped-displacement steepest descent to relieve the steric clashes in a built pool.

    A rigid internal-coordinate build preserves every bond and angle and can still drive
    non-bonded atoms into each other.  :mod:`alanine.system` documents the consequence exactly:
    such a seed is *finite* but explodes on the first BAOAB step, reaching absurd kinetic
    temperatures **without ever producing a NaN**, so a finiteness check does not catch it.

    Measured here before this function existed: 2 of 3072 deca-alanine replicas blew up during
    the umbrella pull, ending 13 283 nm from their restraint centre.  The structural screening
    excluded them and the reference was never corrupted -- but the compute was wasted and a
    failure that a finiteness check cannot see should not be left to a downstream gate.

    Each step is capped in displacement so a huge initial force cannot throw the structure.
    Returns ``(x_relaxed, ok, report)``; ``ok`` gates on energy above the pool minimum and on
    maximum force, which is what actually predicts an explosion.
    """
    y = x.detach().clone() if torch.is_tensor(x) else torch.as_tensor(x)
    y = y.clone()
    for _ in range(int(n_steps)):
        f = engine.forces(y)
        d = lr * f
        nrm = d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        y = (y + d * torch.clamp(max_disp / nrm, max=1.0)).detach()
    e = engine.energy(y)
    fmax = engine.forces(y).abs().amax(dim=(-2, -1))
    finite = torch.isfinite(e) & torch.isfinite(fmax)
    e_min = e[finite].min() if bool(finite.any()) else torch.tensor(float("inf"))
    ok = finite & (e <= e_min + max_energy_above_min) & (fmax <= max_force)
    rep = dict(energy=e.detach().cpu().numpy(), force_max=fmax.detach().cpu().numpy(),
               energy_min=float(e_min), n_fail=int((~ok).sum()))
    return y, ok.cpu().numpy(), rep


def _restraint_force(grad_full, R, centers, k):
    """Cartesian force from ``V = 0.5 k (R - R_c)^2``, i.e. ``-k (R - R_c) grad R``."""
    return dist_bias_force(grad_full, -k * (R - centers))


def run_umbrella(engine, cfg: UmbrellaConfig, build_index=0, n_builds=1, device="cuda",
                 dtype=torch.float64, verbose=True, progress_every=200_000,
                 checkpoint_every=0, on_checkpoint=None):
    """Run every window of every build in one batch.  Returns samples and diagnostics.

    Batch layout is ``(n_builds * n_windows * n_rep, 112, 3)`` -- one process, one GPU, one
    noise stream, which is the v1 design and the reason comparisons stay paired.

    **All builds advance together, on purpose.**  §4.5 acceptance is a statement about the
    spread *between* independent builds, so it cannot be evaluated until every build has
    reached the same amount of sampling.  Running them sequentially would mean no acceptance
    test until the last one finished, which forecloses stopping early.  Interleaving also puts
    the batch past the per-state cost knee: measured 0.99 us/state-step at ``B = 2048`` against
    0.79-0.82 us at ``B >= 4096``.

    ``on_checkpoint(n_sample, samples) -> bool`` is called every ``checkpoint_every``
    production steps; returning ``True`` stops production early.  ``samples`` carries the
    arrays accumulated so far.  This is how the §4.5 convergence-versus-compute trace is
    produced -- as a by-product of the run rather than as extra work.
    """
    n_w, n_r = cfg.n_windows, cfg.n_rep
    per_build = n_w * n_r
    B = n_builds * per_build
    beta = cfg.beta
    centers_np = window_centers(cfg)
    centers = torch.as_tensor(np.tile(np.repeat(centers_np, n_r), n_builds),
                              device=device, dtype=dtype)

    i_cv, j_cv = dsys.terminal_carbonyls(dsys.N_RES)
    cv = DistanceCV(i_cv, j_cv)
    labels = DecaLabels(device=device, dtype=dtype)

    # Each build gets its own RNG, so "independently initialised" means what it says.
    rng = np.random.default_rng(cfg.rng_seed + 7919 * build_index)
    X0 = np.concatenate([
        diverse_pool(per_build, cfg, np.random.default_rng(cfg.rng_seed + 7919 * (build_index + b)))
        for b in range(n_builds)])
    q = torch.as_tensor(X0, device=device, dtype=dtype).contiguous()

    # Relieve build-induced steric clashes BEFORE any dynamics.  Without this, a finite but
    # clashing seed explodes on the first BAOAB step and no finiteness check sees it.
    q, ok_seed, rep_seed = relax_pool(engine, q)
    n_bad = int((~ok_seed).sum())
    if n_bad:
        if n_bad > 0.05 * B:
            raise RuntimeError(f"{n_bad}/{B} relaxed seeds still fail the energy/force gate; "
                               "the pool builder is producing unusable structures")
        good = np.flatnonzero(ok_seed)
        repl = good[rng.integers(0, good.size, size=n_bad)]
        q[torch.as_tensor(np.flatnonzero(~ok_seed), device=device)] = \
            q[torch.as_tensor(repl, device=device)].clone()
    if verbose:
        print(f"  seed relaxation: {B - n_bad}/{B} pass "
              f"(E_min {rep_seed['energy_min']:.1f} kJ/mol, "
              f"|F|max {rep_seed['force_max'].max():.2e}); {n_bad} replaced", flush=True)

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
    pull_err = np.abs(R_after_pull - np.tile(np.repeat(centers_np, n_r), n_builds))
    # Amendment 1: a hard pull is exactly the operation that can flip a stereocentre or
    # isomerise a peptide bond, so screen here rather than discover it in the samples.
    # A replica that blew up is finite, so the CV itself is the tell: 13 283 nm was observed
    # before seed relaxation existed.  Screen on it explicitly rather than trusting chirality
    # to happen to catch every explosion.
    sane_cv = np.isfinite(R_after_pull) & (R_after_pull > 0.2) & (R_after_pull < 10.0)
    ok_pull, rep_pull = dsys.validate_thermal(q.cpu().numpy(), dsys.N_RES)
    ok_pull = ok_pull & sane_cv
    rep_pull["n_fail_cv_sanity"] = int((~sane_cv).sum())
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
    stopped_early_at = 0

    t_prod0 = time.perf_counter()

    def _snapshot():
        return dict(xi=np.stack(xi_s), y=np.stack(y_s), keep=keep,
                    **{k: np.stack(v) for k, v in lab_s.items()})

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
            # Rate is measured from the START OF PRODUCTION, not from t0.  Including the pull
            # and equilibration in the rate inflates the remaining-time estimate by roughly the
            # ratio of setup to production steps -- at 5 % done that read 775 min against a
            # true 6.2 h, which is alarming and wrong.
            el_prod = time.perf_counter() - t_prod0
            el_all = time.perf_counter() - t0
            frac = (s + 1) / cfg.n_prod_steps
            print(f"    prod {100*frac:5.1f}%  {el_all/60:6.1f} min elapsed "
                  f"({el_prod/60:6.1f} min of production), "
                  f"~{el_prod/frac*(1-frac)/60:6.1f} min left", flush=True)
        if (on_checkpoint is not None and checkpoint_every
                and (s + 1) % checkpoint_every == 0 and n_sample > 1):
            if bool(on_checkpoint(s + 1, _snapshot())):
                stopped_early_at = s + 1
                if verbose:
                    print(f"    STOP: acceptance met at {(s+1)*cfg.dt:.2f} ps per replica",
                          flush=True)
                break
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
        n_builds=n_builds, per_build=per_build, stopped_early_at=stopped_early_at,
        build_index=build_index, config=asdict(cfg), config_hash=cfg.config_hash(),
        pull_error_nm=pull_err, runtime_seconds=time.perf_counter() - t0,
        final_thermal_pass=int(np.sum(ok_thermal)), final_thermal_n=int(ok_thermal.size),
        final_n_cis=int(np.sum(rep_thermal["n_cis_bonds"])),
        final_min_chirality=float(np.min(rep_thermal["min_chirality"])),
        **{k: np.stack(v) for k, v in lab_s.items()})


# --------------------------------------------------------------------------- MBAR
#: MBAR solves a dense ``(n_states, n_samples)`` reduced-potential matrix.  At the production
#: budget the raw sample count is ~24.6 M over 97 states, i.e. **~19 GB** of float64 -- it does
#: not run.  Free energies are therefore solved on a strided subsample and then applied to every
#: sample; see :func:`mbar_weights`.
MBAR_MAX_SAMPLES = 400_000


def _order_by_window(arr, n_w, n_rep, keep, stride=1):
    """Flatten ``(n_sample, n_w * n_rep)`` to window-major 1-D, dropping screened replicas."""
    a = arr[::stride]
    m = a.shape[0]
    a = a.reshape(m, n_w, n_rep)
    parts, counts = [], np.zeros(n_w, dtype=int)
    for w in range(n_w):
        block = a[:, w, keep[w]].reshape(-1)
        parts.append(block)
        counts[w] = block.size
    return np.concatenate(parts), counts


def mbar_weights(xi, centers, n_rep, k_umbrella, beta, keep=None, aux=None,
                 max_mbar_samples=MBAR_MAX_SAMPLES):
    """Unbiased-ensemble weights for every sample, via MBAR free energies.

    ``xi`` is ``(n_sample, n_windows * n_rep)`` with columns ordered window-major.

    **Two-stage, because one stage does not fit.** MBAR's dense reduced-potential matrix is
    ``n_states x n_samples``; at the production budget that is ~24.6 M samples over 97 states,
    about 19 GB. Stage 1 solves the state free energies ``f_k`` on a uniformly strided subsample
    sized to ``max_mbar_samples``. Stage 2 applies those ``f_k`` to **every** sample in chunks:

        w_n  proportional to  1 / sum_k N_k exp(f_k - u_k(x_n))

    ``f_k`` are properties of the states, not of how many samples were drawn, so this is exact
    given ``f_k`` -- it simply spends the histogram's statistics on all the data while spending
    the solver's on as much as it can hold. Striding uniformly per replica keeps the per-window
    proportions identical between the two stages, which is what makes the ``N_k`` consistent.
    Samples 0.5 ps apart are in any case strongly correlated, so the subsample loses little.

    ``keep`` is the Amendment 1 structural screening mask over replicas; excluded replicas are
    dropped and ``N_k`` rebuilt per window from what survives. ``aux`` is an optional dict of
    ``(n_sample, B)`` arrays reordered identically and returned alongside.

    Returns ``(xi_all, weights, info, aux_out)``.
    """
    from pymbar import MBAR

    n_s, B = xi.shape
    n_w = len(centers)
    if keep is None:
        keep = np.ones(B, dtype=bool)
    keep = np.asarray(keep, dtype=bool).reshape(n_w, n_rep)
    n_kept = int(keep.sum())

    total = n_s * n_kept
    stride = max(1, int(np.ceil(total / max(max_mbar_samples, 1))))

    # ---- stage 1: free energies on a strided subsample -----------------------------------
    x_sub, N_sub = _order_by_window(xi, n_w, n_rep, keep, stride=stride)
    u_kn = np.zeros((n_w + 1, x_sub.size))
    u_kn[:n_w] = beta * 0.5 * k_umbrella * (x_sub[None, :] - centers[:, None]) ** 2
    N_k_sub = np.concatenate([N_sub, [0]])
    mbar = MBAR(u_kn, N_k_sub, solver_protocol="robust")
    f_k = np.asarray(mbar.f_k, dtype=np.float64)

    # ---- stage 2: apply f_k to every sample, in chunks ------------------------------------
    x_all, N_full = _order_by_window(xi, n_w, n_rep, keep, stride=1)
    N_k_full = np.concatenate([N_full, [0]]).astype(np.float64)
    logN = np.log(np.maximum(N_k_full[:n_w], 1e-300))
    w = np.empty(x_all.size, dtype=np.float64)
    chunk = 1_000_000
    for s in range(0, x_all.size, chunk):
        xc = x_all[s:s + chunk].astype(np.float64)
        u = beta * 0.5 * k_umbrella * (xc[None, :] - centers[:, None]) ** 2
        # log sum_k N_k exp(f_k - u_k), stabilised
        a = (logN + f_k[:n_w])[:, None] - u
        mx = a.max(0)
        w[s:s + chunk] = -(mx + np.log(np.exp(a - mx[None, :]).sum(0)))
    w -= w.max()
    w = np.exp(w)
    w /= w.sum()

    aux_out = {k: _order_by_window(np.asarray(v), n_w, n_rep, keep, stride=1)[0]
               for k, v in (aux or {}).items()}
    info = dict(f_k=f_k, N_k_subsample=N_k_sub, N_k_full=N_k_full, stride=int(stride),
                n_mbar_samples=int(x_sub.size), n_total_samples=int(x_all.size),
                n_kept_replicas=n_kept, mbar=mbar)
    return x_all, w, info, aux_out


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

    # **Drop out-of-domain samples; do NOT let them be clamped into the edge bins.**
    # ``iv.bin_counts`` clamps out-of-range samples into bin 0 / bin n-1, which is harmless in
    # the alkane study because soft walls make it rare -- but Amendment 1 deliberately places
    # the umbrella centres at [win_lo, win_hi] BRACKETING [R_lo, R_hi], so entire windows sample
    # outside the evaluation domain by design.  Clamping them piles that mass into the edge bins
    # and carves a spurious ~2.7 kT well at grid[0]: a fake basin, which propagated into a fake
    # ESTABLISHMENT-LIMITED verdict on the first screen.  The coverage fix created the artifact.
    x = np.asarray(xi_all, dtype=np.float64)
    wv = np.asarray(w, dtype=np.float64)
    inside = (x >= cfg.R_lo) & (x <= cfg.R_hi)
    n_dropped = int((~inside).sum())
    x, wv = x[inside], wv[inside]

    z = torch.as_tensor(x, device=device)[None]
    wt = torch.as_tensor(wv, device=device)[None]
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
            counts[0].cpu().numpy(), n_dropped)
