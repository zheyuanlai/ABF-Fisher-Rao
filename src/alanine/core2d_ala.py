"""Corrected 2-D ABF (+ oracle marginal Fisher--Rao) sampler for atomistic Ace-Ala-Nme.

Two arms only, and **the only difference between them is the birth--death step**:

    ``abf``        2-D ABF on (phi, psi)
    ``fr_oracle``  the same, plus mFR birth--death against the ACCEPTED reference target
                   ``q ~ exp(-beta (F_ref - B))``

Everything else is shared and frozen: force field, integrator, dt, gamma, temperature, dtype,
CV convention, grid parity, estimator settings, projection, warmup, clipping, initial ensemble,
and the per-step dynamical noise stream (so a paired seed sees *identical* Langevin noise in
both arms until a replacement actually occurs).

Deliberately absent -- these are separate future ``mFR-ABF-v2`` ablations, not part of the first
comparison: exponential forgetting, weighted projection, clone-discounted effective counts,
lagged targets, tempered targets.

Correctness properties enforced here
------------------------------------
* **Fixed-consumption RNG.** The dynamical noise is one fixed-shape draw per step from a
  method-independent generator.  The FR generator is per-seed and consumes a fixed number of
  values per FR *event opportunity*, independent of how many replacements fire, so changing one
  seed's event pattern cannot perturb any other seed.
* **Full-state cloning.** A birth copies the parent position and cached physical force, inherits
  both genealogy labels, and draws fresh Maxwell momenta; the parent is untouched.
* **Non-finite containment.** Every quantity feeding the accumulator is checked on-device.  Bad
  samples are masked out of *both* the force sums and the counts, so one bad replica can never
  contaminate a seed's global accumulator; the event is counted and the run aborts at the next
  save boundary with a full diagnostic dump.
* **Online/frozen consistency.** The projection asserts ``gB == spectral_gradient(B)``; the
  applied CV force is that same gradient, magnitude-clipped (never per-component, which would
  reintroduce the curl the projection exists to remove).
* **No host sync in the hot loop.** All diagnostics accumulate on device; transfers happen only
  at ``save_every``.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from alkanes import density2d as d2
from alkanes import poisson2d as ps

from .dynamics import KB, SeedFailure
from .projection import clip_magnitude, require_odd_grid

EPS = 1.0e-12
TWO_PI = 2.0 * math.pi

METHODS = ("abf", "fr_oracle")
FR_METHODS = ("fr_oracle",)


@dataclass(frozen=True)
class AlaSimConfig:
    """Frozen configuration.  Estimator settings are the accepted shared values."""
    # --- dynamics (frozen physical model) ---
    dt: float = 0.001                      # ps  (1 fs)
    gamma: float = 1.0                     # ps^-1
    temperature: float = 300.0             # K
    n_steps: int = 100_000
    n_replicas: int = 2048
    save_every: int = 1_000
    rng_seed: int = 20260810
    # --- estimator (frozen, shared by both arms) ---
    n_grid: int = 97                       # ODD: no Nyquist row exists
    abf_bandwidth: float = 0.08            # rad
    kde_bandwidth: float = 0.15            # rad
    abf_min_count: float = 200.0
    abf_force_clip: float = 200.0          # kJ/mol/rad -- a GUARD, not a regulariser
    project_every: int = 50
    estimator_stride: int = 1
    abf_warmup_steps: int = 5_000
    estimator_burn_in_steps: int = 0
    # --- Fisher--Rao (oracle arm only) ---
    fr_rate: float = 0.05
    score_clip: float = 2.0
    fr_start_steps: int = 20_000           # 20 ps
    fr_every: int = 500                    # 0.5 ps
    max_event_fraction: float = 0.05
    lineage_reset_steps: int = 6_000       # 6 ps age-aware genealogy window

    def config_hash(self):
        import hashlib, json
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def assert_no_reference_leakage(method, reference_F):
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if method == "fr_oracle":
        if reference_F is None:
            raise ValueError("fr_oracle requires the reference free energy.")
        return
    if reference_F is not None:
        raise AssertionError(
            f"NO-REFERENCE-LEAKAGE VIOLATION: method={method!r} received a reference free "
            "energy; only fr_oracle may.")


def _project(f1s, f2s, csum, K1, K2, dz1, dz2, min_count, check=True, tol=1e-9):
    """Mean-force sums -> conservative bias ``B`` and its spectral gradient."""
    g1, g2, den = d2.mean_force_fields(f1s, f2s, csum, K1, K2)
    trust = den >= min_count
    g1 = torch.where(trust, g1, torch.zeros_like(g1))
    g2 = torch.where(trust, g2, torch.zeros_like(g2))
    B, gB1, gB2 = ps.poisson_projection(g1, g2, dz1, dz2)
    resid = 0.0
    if check:
        s1, s2 = ps.spectral_gradient(B, dz1, dz2)
        resid = float(max((gB1 - s1).abs().max(), (gB2 - s2).abs().max()))
        if resid > tol:
            raise AssertionError(
                f"projection returned gB != grad(B) (max abs {resid:.3e} > {tol:.1e}); the "
                "online and frozen runs would apply different fields")
    return B, gB1, gB2, g1, g2, float(trust.to(B.dtype).mean()), resid


def sanitize_reference(F_ref, kT, cap_kT=30.0):
    """Make the reference FES usable as an oracle target.

    The accepted reference stores ``+inf`` in bins no umbrella window ever visited (they are
    excluded by the evaluation mask).  Feeding that straight in is fatal and *silent*:
    ``F - F.mean()`` becomes ``inf - inf = NaN`` over the whole grid, so the target is NaN, the
    Fisher-Rao score is NaN, no death or birth weight is ever positive, and ``fr_oracle``
    degenerates into ``abf`` while reporting zero events.  That would have been read as
    "mFR is EQUIVALENT to ABF" when in fact the mechanism was never switched on.

    Unvisited cells are therefore pinned to ``F_min + cap_kT * kT``, i.e. a region the oracle
    target treats as essentially unpopulated (``exp(-30) ~ 1e-13``), and the zero of the scale is
    set from the FINITE cells only.
    """
    F = torch.as_tensor(F_ref)
    finite = torch.isfinite(F)
    if not bool(finite.any()):
        raise ValueError("reference FES has no finite cells")
    fmin = F[finite].min()
    F = torch.where(finite, F, fmin + cap_kT * kT)
    return F - F.mean()


def _oracle_target(F_ref, B, beta, dz1, dz2):
    """``q ~ exp(-beta (F_ref - B))`` normalised on the torus.  ``F_ref`` must be finite."""
    log_q = -beta * (F_ref[None] - B)
    log_q = log_q - log_q.amax(dim=(-2, -1), keepdim=True)
    q = d2.normalize2(torch.exp(log_q), dz1, dz2)
    if not bool(torch.isfinite(q).all()):
        raise AssertionError("oracle target is non-finite; the reference FES was not sanitised")
    return q


def _ancestor_stats_t(labels, N):
    """ESS / n_unique / max-fraction per seed, computed ON DEVICE (labels ``(R,N)``)."""
    R = labels.shape[0]
    oh = torch.zeros(R, N, device=labels.device, dtype=torch.float64)
    oh.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.float64))
    w = oh / oh.sum(1, keepdim=True).clamp_min(1.0)
    ess = 1.0 / (w * w).sum(1).clamp_min(EPS)
    return ess, (oh > 0).sum(1), w.amax(1)


def _dihedral_iupac_t(x, idx):
    """IUPAC signed dihedral of ``x (B,A,3)`` for one atom 4-tuple.  Used only for the optional
    omitted-coordinate recorder below, never in the estimator path."""
    p0, p1, p2, p3 = (x[:, i] for i in idx)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    return torch.atan2((torch.linalg.cross(b1n, v, dim=-1) * w).sum(-1), (v * w).sum(-1))


def run_sampler_ala(method, tff, cv, sim: AlaSimConfig, seeds, init_positions, basin_labels,
                    device, dtype=torch.float64, reference_F=None, dump_dir=None,
                    force_fn=None, rare_basin=2, extra_angle_atoms=None, verbose=True):
    """Run ``R = len(seeds)`` matched-seed replicas of ``method``.

    ``init_positions``  ``(R, N, A, 3)`` -- identical across arms for a given seed.
    ``basin_labels``    ``(n_grid, n_grid)`` long tensor from the reference watershed.
    ``force_fn``        optional compiled physical-force callable ``(B,A,3)->(B,A,3)``.
    ``rare_basin``      index of the basin whose in-basin genealogy is tracked separately.
                        Defaults to 2, which is C7ax for alanine's prominence ordering
                        (C7eq=0, C5=1, C7ax=2) -- but the ordering is by DEPTH and is therefore
                        system-specific, so any other system must pass its own index
                        (e.g. ``basins.index["<name>"]``). Hardcoding it silently reports the
                        wrong basin's ancestor statistics.
    ``extra_angle_atoms`` optional atom 4-tuple of a dihedral that is NOT in the CV, recorded at
                        every save as ``extra_angle`` ``(T, R, N)``.  This is how the OMITTED
                        coordinate is checked: a 2-D CV can look perfectly converged while the
                        coordinate it hides is not equilibrated, and nothing else in this
                        sampler would notice.  ``None`` (the default) reproduces the accepted
                        alanine behaviour exactly -- no extra evaluation, no extra output key.
    """
    if not (0 <= int(rare_basin) <= int(basin_labels.max())):
        raise ValueError(f"rare_basin={rare_basin} outside the basin labels present "
                         f"(0..{int(basin_labels.max())})")
    assert_no_reference_leakage(method, reference_F)
    require_odd_grid(sim.n_grid)
    is_fr = method in FR_METHODS
    R, N = len(seeds), sim.n_replicas
    A = tff.n_atoms
    beta = 1.0 / (KB * sim.temperature)
    n = sim.n_grid
    g1c, g2c, dz1, dz2 = d2.torus_grid(n, n, device=device, dtype=dtype)
    K1, K2 = d2.kernels(g1c, g2c, sim.abf_bandwidth, sim.abf_bandwidth)
    Kk1, Kk2 = d2.kernels(g1c, g2c, sim.kde_bandwidth, sim.kde_bandwidth)

    # --- RNG: method-independent dynamics, per-seed FR ---
    gen_dyn = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    gens_fr = [torch.Generator(device=device).manual_seed(int(sim.rng_seed) + 987654321 + 1000 * r)
               for r in range(R)]

    F_ref_t = None
    if is_fr:
        F_ref_t = sanitize_reference(
            torch.as_tensor(reference_F, device=device, dtype=dtype), KB * sim.temperature)

    q = torch.as_tensor(init_positions, device=device, dtype=dtype).reshape(R, N, A, 3).contiguous()
    m = tff.masses.reshape(-1, 1)
    kT = KB * sim.temperature
    sigma_v = math.sqrt(kT) / m.sqrt()
    c1 = math.exp(-sim.gamma * sim.dt)
    c2 = math.sqrt(1.0 - c1 * c1)
    v = torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype) * sigma_v
    phys = force_fn or (lambda x: tff.forces(x))

    # --- accumulators ---
    f1s = torch.zeros(R, n, n, device=device, dtype=dtype)
    f2s = torch.zeros(R, n, n, device=device, dtype=dtype)
    csum = torch.zeros(R, n, n, device=device, dtype=dtype)
    B = torch.zeros(R, n, n, device=device, dtype=dtype)
    gB1 = torch.zeros(R, n, n, device=device, dtype=dtype)
    gB2 = torch.zeros(R, n, n, device=device, dtype=dtype)
    anc = torch.arange(N, device=device).expand(R, N).contiguous()      # permanent
    anc_age = anc.clone()                                              # age-aware (6 ps window)

    # device-side diagnostics (no host sync in the loop)
    n_nonfinite = torch.zeros(R, device=device, dtype=torch.long)
    n_clip = torch.zeros((), device=device, dtype=torch.long)
    n_force_eval = torch.zeros((), device=device, dtype=torch.long)
    n_events = torch.zeros(R, device=device, dtype=torch.long)
    Tsum = torch.zeros((), device=device, dtype=dtype)
    n_T = 0
    first_hit = torch.full((R, int(basin_labels.max().item()) + 1), -1,
                           device=device, dtype=torch.long)
    n_basins = first_hit.shape[1]
    prev_basin = None
    trans = torch.zeros(R, n_basins, n_basins, device=device, dtype=torch.long)

    # ``*_rare`` are the in-basin genealogy diagnostics for basin ``rare_basin``.  They were
    # once named ``*_c7ax``, which was true only for alanine's depth ordering and would have
    # silently mislabelled every other system's tracked basin; the artifact now records
    # ``rare_basin`` explicitly so the name cannot drift from the index.
    diag = {k: [] for k in ("steps", "times", "pmf", "basin_frac", "ess_perm", "ess_age",
                            "n_unique", "wmax", "wmax_rare", "events_cum", "trust_frac",
                            "clip_frac", "temperature", "proj_resid", "curl_pre",
                            "score_std", "score_absmax", "ess_age_rare")}
    if extra_angle_atoms is not None:
        diag["extra_angle"] = []
    trust_frac = 0.0
    proj_resid = 0.0
    curl_pre = torch.zeros(R, device=device, dtype=dtype)
    score_std = torch.zeros(R, device=device, dtype=dtype)
    score_absmax = torch.zeros(R, device=device, dtype=dtype)
    t0 = time.perf_counter()

    # BAOAB needs the force at the CURRENT position on entry, and the loop is arranged so that
    # exactly ONE physical-force evaluation and ONE CV-geometry evaluation happen per step: the
    # pair computed after the position update is carried into the next iteration's estimator.
    qf = q.reshape(R * N, A, 3)
    f_phys = phys(qf).reshape(R, N, A, 3)
    floc, phi, gfull, geo = cv.local_mean_force(qf, f_phys.reshape(R * N, A, 3), beta)
    n_force_eval += R * N

    def _abort(step, why):
        path = None
        if dump_dir is not None:
            os.makedirs(dump_dir, exist_ok=True)
            bad = int(torch.argmax(n_nonfinite).item())
            path = os.path.join(dump_dir, f"FAILED_{method}_seed{bad}_step{step}.npz")
            np.savez_compressed(
                path, step=step, method=method, seed_index=bad, reason=why,
                q=q[bad].detach().cpu().numpy(), v=v[bad].detach().cpu().numpy(),
                f=f_phys[bad].detach().cpu().numpy(), B=B[bad].detach().cpu().numpy(),
                f1s=f1s[bad].detach().cpu().numpy(), csum=csum[bad].detach().cpu().numpy(),
                n_nonfinite=n_nonfinite.detach().cpu().numpy())
        raise SeedFailure(f"{why} in method={method} at step {step}; dump -> {path}",
                          int(torch.argmax(n_nonfinite).item()), step, path)

    def _bias_at(phi_t, gfull_t, ramp):
        """Magnitude-clipped gradient of the saved potential, pushed to Cartesian."""
        a1 = torch.nan_to_num(phi_t[:, 0].reshape(R, N), 0.0, 0.0, 0.0)
        a2 = torch.nan_to_num(phi_t[:, 1].reshape(R, N), 0.0, 0.0, 0.0)
        c_1 = d2.bilinear_interp2(ramp * gB1, g1c, g2c, dz1, dz2, a1, a2)
        c_2 = d2.bilinear_interp2(ramp * gB2, g1c, g2c, dz1, dz2, a1, a2)
        nclip = ((c_1 * c_1 + c_2 * c_2) > sim.abf_force_clip ** 2).sum()
        c_1, c_2 = clip_magnitude(c_1, c_2, sim.abf_force_clip)
        cart = (c_1.reshape(R * N)[:, None, None] * gfull_t[:, 0]
                + c_2.reshape(R * N)[:, None, None] * gfull_t[:, 1]).reshape(R, N, A, 3)
        return cart, a1, a2, nclip

    for step in range(sim.n_steps + 1):
        # ---- non-finite containment: mask BEFORE anything reaches the accumulator ----
        okm = (torch.isfinite(floc).all(-1) & torch.isfinite(phi).all(-1)
               & torch.isfinite(geo["lam_min"]) & (geo["lam_min"] > 0)).reshape(R, N)
        okm = okm & torch.isfinite(f_phys.reshape(R, N, -1)).all(-1)
        n_nonfinite += (~okm).sum(1)
        okd = okm.to(dtype)
        f1 = torch.nan_to_num(floc[:, 0].reshape(R, N), 0.0, 0.0, 0.0) * okd
        f2 = torch.nan_to_num(floc[:, 1].reshape(R, N), 0.0, 0.0, 0.0) * okd

        ramp = min(1.0, step / max(sim.abf_warmup_steps, 1))
        bias_cart, p1s, p2s, nc = _bias_at(phi, gfull, ramp)
        n_clip += nc

        if step % sim.estimator_stride == 0 and step >= sim.estimator_burn_in_steps:
            f1s += d2.scatter_sum(p1s, p2s, f1, n, n, dz1, dz2)
            f2s += d2.scatter_sum(p1s, p2s, f2, n, n, dz1, dz2)
            csum += d2.scatter_sum(p1s, p2s, okd, n, n, dz1, dz2)   # bad samples add no count

        if step % sim.project_every == 0:
            B, gB1, gB2, g1f, g2f, trust_frac, proj_resid = _project(
                f1s, f2s, csum, K1, K2, dz1, dz2, sim.abf_min_count)
            curl_pre = ps.curl_norm(g1f, g2f, dz1, dz2)
            bias_cart, p1s, p2s, _ = _bias_at(phi, gfull, ramp)     # refresh with the new field

        # ---- basin bookkeeping (device-side) ----
        dzc = TWO_PI / n
        bi = torch.floor((p1s + math.pi) / dzc).long().clamp(0, n - 1)
        bj = torch.floor((p2s + math.pi) / dzc).long().clamp(0, n - 1)
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

        # ---- save ----
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
                    wmax_rare[r] = w.max()
                    ess_a_rare[r] = 1.0 / (w * w).sum().clamp_min(EPS)
            Tnow = float(Tsum / max(n_T, 1)) if n_T else float("nan")
            diag["steps"].append(step); diag["times"].append(step * sim.dt)
            diag["pmf"].append(B.detach().cpu().numpy())
            diag["basin_frac"].append(frac.detach().cpu().numpy())
            diag["ess_perm"].append((ess_p / N).cpu().numpy())
            diag["ess_age"].append((ess_a / N).cpu().numpy())
            diag["ess_age_rare"].append((ess_a_rare / N).cpu().numpy())
            diag["n_unique"].append(nuq.cpu().numpy())
            diag["wmax"].append(wmx.cpu().numpy())
            diag["wmax_rare"].append(wmax_rare.cpu().numpy())
            diag["events_cum"].append(n_events.cpu().numpy())
            diag["trust_frac"].append(trust_frac)
            diag["clip_frac"].append(float(n_clip.item()) / max(float(n_force_eval.item()), 1.0))
            diag["temperature"].append(Tnow)
            diag["proj_resid"].append(proj_resid)
            diag["curl_pre"].append(curl_pre.detach().cpu().numpy())
            diag["score_std"].append(score_std.detach().cpu().numpy())
            diag["score_absmax"].append(score_absmax.detach().cpu().numpy())
            if extra_angle_atoms is not None:
                diag["extra_angle"].append(
                    _dihedral_iupac_t(q.reshape(R * N, A, 3), extra_angle_atoms)
                    .reshape(R, N).to(torch.float32).cpu().numpy())
            Tsum = torch.zeros((), device=device, dtype=dtype); n_T = 0

        if step == sim.n_steps:
            break

        # ---- BAOAB: one physical-force and one CV-geometry evaluation per step ----
        v = v + (0.5 * sim.dt) * (f_phys + bias_cart) / m
        q = q + (0.5 * sim.dt) * v
        v = c1 * v + c2 * torch.randn(v.shape, generator=gen_dyn, device=device,
                                      dtype=dtype) * sigma_v
        q = q + (0.5 * sim.dt) * v
        qf = q.reshape(R * N, A, 3)
        f_phys = phys(qf).reshape(R, N, A, 3)
        n_force_eval += R * N
        floc, phi, gfull, geo = cv.local_mean_force(qf, f_phys.reshape(R * N, A, 3), beta)
        bias_new, _, _, nc2 = _bias_at(phi, gfull, ramp)
        n_clip += nc2
        v = v + (0.5 * sim.dt) * (f_phys + bias_new) / m
        Tsum = Tsum + (m * v * v).sum() / (3.0 * R * N * A * KB); n_T += 1

        # ---- age-aware genealogy window ----
        if sim.lineage_reset_steps > 0 and (step + 1) % sim.lineage_reset_steps == 0:
            anc_age = torch.arange(N, device=device).expand(R, N).contiguous()

        # ---- Fisher--Rao birth--death (the ONLY difference between the arms) ----
        if is_fr:
            nxt = step + 1
            if nxt >= sim.fr_start_steps and (nxt - sim.fr_start_steps) % sim.fr_every == 0:
                z1 = torch.nan_to_num(phi[:, 0].reshape(R, N), 0.0, 0.0, 0.0)
                z2 = torch.nan_to_num(phi[:, 1].reshape(R, N), 0.0, 0.0, 0.0)
                p_hat = d2.kde2(z1, z2, Kk1, Kk2, n, n, dz1, dz2)
                q_tgt = _oracle_target(F_ref_t, B, beta, dz1, dz2)
                score, _ = d2.fr_score_2d(z1, z2, p_hat, q_tgt, g1c, g2c, dz1, dz2, sim.score_clip)
                score_std = score.std(1)
                score_absmax = score.abs().amax(1)
                q, v, f_phys, anc, anc_age, ne = _birth_death_ala(
                    q, v, f_phys, score, anc, anc_age, gens_fr, sim, sigma_v)
                n_events += ne.to(n_events.device)
                if int(ne.sum()) > 0:      # state changed: refresh the cached geometry
                    qf = q.reshape(R * N, A, 3)
                    floc, phi, gfull, geo = cv.local_mean_force(
                        qf, f_phys.reshape(R * N, A, 3), beta)

    if int(n_nonfinite.sum().item()) > 0:
        _abort(sim.n_steps, "non-finite detected")
    wall = time.perf_counter() - t0
    out = dict(method=method, seeds=np.asarray(seeds), n_replicas=N, n_steps=sim.n_steps,
               grid=g1c.cpu().numpy(), dz=float(dz1), n_grid=n,
               rare_basin=int(rare_basin),
               final_pmf=B.detach().cpu().numpy(),
               first_hit=first_hit.cpu().numpy(), trans_matrix=trans.cpu().numpy(),
               total_events=n_events.cpu().numpy(),
               clip_fraction=float(n_clip.item()) / max(float(n_force_eval.item()), 1.0),
               force_evaluations=int(n_force_eval.item()),
               aggregate_simulated_ps=float(sim.n_steps * sim.dt * R * N),
               wall_seconds=wall, ms_per_step=wall / max(sim.n_steps, 1) * 1e3,
               peak_cuda_gib=(torch.cuda.max_memory_allocated() / 2 ** 30
                              if device != "cpu" else 0.0),
               n_nonfinite=n_nonfinite.cpu().numpy(), config_hash=sim.config_hash())
    for k in diag:
        out[k] = np.asarray(diag[k])
    if verbose:
        print(f"  {method:10s} R={R} N={N}: {wall:.1f}s  {out['ms_per_step']:.2f} ms/step  "
              f"events={int(n_events.sum())}  clip={out['clip_fraction']:.2e}  "
              f"peak={out['peak_cuda_gib']:.2f} GiB", flush=True)
    return out


def _birth_death_ala(q, v, f, score, anc, anc_age, gens, sim, sigma_v):
    """Fixed-population full-state kill-and-clone; fixed RNG consumption per seed."""
    R, N = score.shape
    dt_eff = sim.dt * sim.fr_every
    max_ev = int(sim.max_event_fraction * N)
    qn, vn, fn = q.clone(), v.clone(), f.clone()
    an, ag = anc.clone(), anc_age.clone()
    ne = torch.zeros(R, dtype=torch.long, device=q.device)
    if max_ev < 1 or sim.fr_rate <= 0.0:
        return qn, vn, fn, an, ag, ne
    death_w = torch.clamp(score, min=0.0)
    birth_w = torch.clamp(-score, min=0.0)
    p_die = torch.where(death_w > 0, 1.0 - torch.exp(-sim.fr_rate * death_w * dt_eff),
                        torch.zeros_like(death_w))
    A = q.shape[-2]
    for r in range(R):
        g = gens[r]
        # FIXED consumption per opportunity, independent of the number of events
        u_fire = torch.rand(N, generator=g, device=q.device, dtype=q.dtype)
        u_pick = torch.rand(N, generator=g, device=q.device, dtype=q.dtype)
        vfresh = torch.randn((N, A, 3), generator=g, device=q.device, dtype=q.dtype) * sigma_v
        if float(birth_w[r].sum()) <= 0.0 or float(death_w[r].sum()) <= 0.0:
            continue
        fire = u_fire < p_die[r]
        di = torch.nonzero(fire, as_tuple=False).flatten()
        if di.numel() == 0:
            continue
        if di.numel() > max_ev:
            di = di[torch.argsort(score[r, di], descending=True)[:max_ev]]
        k = int(di.numel())
        cdf = torch.cumsum(birth_w[r], 0)
        cdf = cdf / cdf[-1].clamp_min(1e-30)
        src = torch.searchsorted(cdf, u_pick[:k].clamp(0, 1 - 1e-12)).clamp_(0, N - 1)
        qn[r, di] = q[r, src]
        fn[r, di] = f[r, src]                 # cached PHYSICAL force follows the position
        an[r, di] = anc[r, src]
        ag[r, di] = anc_age[r, src]
        vn[r, di] = vfresh[:k]                # fresh Maxwell momenta for the child
        ne[r] = k
    return qn, vn, fn, an, ag, ne
