"""OPES_METAD adapter for the WCA dimer engine.

``run_opes_gpu`` mirrors ``wca_abffr_core.run_sampler_gpu`` step-for-step, but the
per-replica biasing mean force comes from an :class:`opes_core.OPESState` instead of
the online ABF estimator, and there is no Fisher--Rao birth--death.  All
``n_replicas`` walkers share ONE OPES bias (multi-walker OPES), depositing into a
single weighted KDE every ``pace`` steps -- the natural equal-force-evaluation match
to ABF/mFR at the same ``N`` and ``n_steps``.

The returned ``diag`` dict uses the SAME keys as ``run_sampler_gpu`` so the existing
metrics / frozen-bias / analysis / summary pipeline consumes OPES runs unchanged.
Fisher--Rao-specific fields (ancestors, replacements, scores) are filled with the
neutral no-op values (NaN / 0 / n_replicas) that those methods use when inactive.

No TI reference is ever consulted (no-leakage): see
``opes_core.assert_no_reference_leakage``.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

import wca_abffr_core as core
import opes_core as oc


def build_opes_config(sim: core.SimConfig, params: core.DimerWCAParams,
                      barrier: float, pace: int, sigma, gamma=float("inf"),
                      gamma_from_barrier: bool = True, sigma_mode: str = "fixed",
                      warmup_steps=None) -> oc.OPESConfig:
    """Construct an OPESConfig on the WCA CV grid, matched to the sampler grid."""
    return oc.OPESConfig(
        z_min=sim.z_min, z_max=sim.z_max, n_grid=sim.n_grid, beta=params.beta,
        barrier=float(barrier), pace=int(pace),
        sigma=(float(sigma) if sigma_mode == "fixed" else float(sim.abf_bandwidth)),
        sigma_mode=sigma_mode, gamma=float(gamma), gamma_from_barrier=bool(gamma_from_barrier),
        bias_force_clip=sim.abf_force_clip,
        warmup_steps=(sim.abf_warmup_steps if warmup_steps is None else int(warmup_steps)),
        fill_edges=sim.abf_edge_extrapolate,
    )


@torch.inference_mode()
def run_opes_gpu(params, sim, opes_cfg, engine, initial_q=None,
                 collect_diagnostics=True, verbose=True, track_crossings=True):
    """Run one multi-walker OPES_METAD sampler on the GPU (WCA dimer).

    Signature parallels ``run_sampler_gpu`` (minus the FR-only args). Returns a
    ``diag`` dict schema-compatible with ``run_sampler_gpu``.
    """
    oc.assert_no_reference_leakage(False, "opes")  # OPES never gets a reference target
    torch.manual_seed(sim.seed)
    q = (core.lattice_initial_conditions(params, sim.n_replicas, engine.device, engine.dtype, seed=sim.seed)
         if initial_q is None else initial_q.clone())
    grid = torch.linspace(sim.z_min, sim.z_max, sim.n_grid, device=engine.device, dtype=engine.dtype)
    noise_scale = math.sqrt(2.0 * sim.dt / params.beta)

    opes = oc.OPESState(opes_cfg, engine.device, dtype=engine.dtype)
    # production ABF estimator kept ONLY as an independent mean-force cross-check
    # readout (never fed back into dynamics); the reported PMF is OPES-native.
    production_estimator = core.TorchKernelABFEstimator(
        grid, sim.abf_bandwidth, sim.abf_smooth_sigma, edge_extrapolate=sim.abf_edge_extrapolate)

    # barrier-crossing tracker (read-only; identical bookkeeping to run_sampler_gpu)
    z_barrier = 0.5 * (sim.transition_lo + sim.transition_hi)
    side = None
    n_c2s = torch.zeros((), device=engine.device, dtype=torch.long) if track_crossings else 0
    n_s2c = torch.zeros((), device=engine.device, dtype=torch.long) if track_crossings else 0
    rep_c2s = torch.zeros(sim.n_replicas, device=engine.device, dtype=torch.long) if track_crossings else None
    rep_s2c = torch.zeros(sim.n_replicas, device=engine.device, dtype=torch.long) if track_crossings else None

    # OPES per-deposit diagnostic log (analogue of the adaptive-FR log)
    opes_log = {k: [] for k in ["step", "neff_frac", "n_kernels", "zed", "sigma_cur",
                                "max_bias", "bias_range"]}

    diag = {k: [] for k in ["steps", "times", "mean_force", "pmf",
                            "mean_force_reweight", "pmf_reweight",
                            "p_hat", "q_target", "pq_l2", "kl_pq", "eff_counts",
                            "frac_compact", "frac_transition", "frac_stretched",
                            "ancestor_ess", "n_unique_ancestor", "max_ancestor_frac",
                            "repl_cumulative"]}
    t0 = time.perf_counter()

    for step in range(sim.n_steps + 1):
        forces_raw = engine.force(q, compute_energy=False)
        forces_physical = core.clip_forces(forces_raw, params.force_clip)
        z = core.reaction_coordinate(q, params)

        if track_crossings:
            cur = z > z_barrier
            if side is None:
                side = cur
            else:
                up = (~side) & cur; down = side & (~cur)
                n_c2s += up.sum(); n_s2c += down.sum()
                rep_c2s += up.long(); rep_s2c += down.long()
                side = cur

        # independent mean-force readout (not fed back to dynamics)
        mf_in = forces_physical if sim.use_clipped_force_for_mean_force else forces_raw
        f_local = torch.clamp(core.local_mean_force(q, mf_in, params),
                              -sim.mean_force_sample_clip, sim.mean_force_sample_clip)
        if step >= sim.estimator_burn_in_steps:
            production_estimator.update(z, f_local)

        # ---- OPES biasing mean force along z (the ONLY bias on the dynamics) ----
        bias_at_z = opes.bias_force_at(z, step=step)
        transport = core.clip_forces(core.add_abf_force(q, forces_physical, bias_at_z, params), params.force_clip)
        transport = core.clip_forces(core.add_reaction_coordinate_wall_force(q, transport, z, sim, params), params.force_clip)

        if step % sim.save_every == 0 or step == sim.n_steps:
            # PRIMARY estimate: integrate the local mean force accumulated under the
            # OPES-biased dynamics -- identical reconstruction to ABF/mFR, so OPES-vs-ABF
            # differs ONLY in the biasing strategy. Native reweight kept as secondary.
            mf_primary = production_estimator.mean_force_profile()
            F_primary = production_estimator.pmf_profile()
            F_native = opes.free_energy(); mf_native = opes.mean_force()
            diag["steps"].append(step); diag["times"].append(step * sim.dt)
            diag["mean_force"].append(core.to_numpy(mf_primary))
            diag["pmf"].append(core.to_numpy(F_primary))
            diag["mean_force_reweight"].append(core.to_numpy(mf_native))
            diag["pmf_reweight"].append(core.to_numpy(F_native))
            diag["repl_cumulative"].append(0)
            if collect_diagnostics:
                fc, ft, fs = core.region_fractions_torch(z, sim)
                diag["frac_compact"].append(fc); diag["frac_transition"].append(ft); diag["frac_stretched"].append(fs)
                diag["eff_counts"].append(core.to_numpy(opes.marginal()))
                p_grid = core.normalize_density_on_grid_torch(
                    core.kde_1d_torch(grid, z, sim.kde_bandwidth, sim.z_min, sim.z_max), grid)
                diag["p_hat"].append(core.to_numpy(p_grid))
                # OPES has no FR target; log the well-tempered target for reference
                q_grid = core.normalize_density_on_grid_torch(opes.marginal() ** (1.0 / opes.gamma), grid) \
                    if math.isfinite(opes.gamma) else core.normalize_density_on_grid_torch(torch.ones_like(grid), grid)
                diag["q_target"].append(core.to_numpy(q_grid))
                diag["pq_l2"].append(float(math.sqrt(core.trapz_torch((p_grid - q_grid) ** 2, grid).item())))
                lr = torch.log(torch.clamp(p_grid, min=core.EPS)) - torch.log(torch.clamp(q_grid, min=core.EPS))
                diag["kl_pq"].append(float(core.trapz_torch(p_grid * lr, grid).item()))
                diag["ancestor_ess"].append(float("nan"))
                diag["n_unique_ancestor"].append(sim.n_replicas)
                diag["max_ancestor_frac"].append(float("nan"))

        if step == sim.n_steps:
            break
        q = core.wrap_positions(q + sim.dt * transport + noise_scale * torch.randn_like(q), params.box_length)
        # deposit into the shared OPES bias every `pace` steps
        if (step + 1) % max(int(opes_cfg.pace), 1) == 0:
            z_new = core.reaction_coordinate(q, params)
            opes.deposit(z_new)
            d = opes.diagnostics()
            opes_log["step"].append(int(step + 1))
            for k in ["neff_frac", "n_kernels", "zed", "sigma_cur", "max_bias", "bias_range"]:
                opes_log[k].append(d[k])

    diag["runtime_seconds"] = time.perf_counter() - t0
    diag["method"] = "opes"
    diag["grid"] = core.to_numpy(grid)
    diag["total_replacement_events"] = 0
    diag["F_target_ema"] = None
    # birth/death histograms N/A for OPES (no resampling)
    diag["birth_hist"] = np.zeros(sim.n_grid, dtype=np.float64)
    diag["death_hist"] = np.zeros(sim.n_grid, dtype=np.float64)
    diag["hist_edges"] = np.linspace(sim.z_min, sim.z_max, sim.n_grid + 1)
    diag["fr_score_std"] = float("nan")
    diag["fr_score_absmax"] = float("nan")
    diag["fr_score_clip_fraction"] = float("nan")
    # reuse the adaptive_log container to carry the OPES per-deposit log
    diag["adaptive_log"] = {k: np.asarray(v, dtype=float) for k, v in opes_log.items()}
    diag["opes_log"] = diag["adaptive_log"]
    if track_crossings:
        c2s = int(n_c2s.item()); s2c = int(n_s2c.item())
        diag["n_compact_to_stretched"] = c2s
        diag["n_stretched_to_compact"] = s2c
        diag["n_barrier_crossings"] = c2s + s2c
        diag["n_round_trips"] = int(torch.minimum(rep_c2s, rep_s2c).sum().item())
    else:
        diag["n_compact_to_stretched"] = -1
        diag["n_stretched_to_compact"] = -1
        diag["n_barrier_crossings"] = -1
        diag["n_round_trips"] = -1
    for key in ["steps", "times", "mean_force", "pmf", "mean_force_reweight", "pmf_reweight",
                "p_hat", "q_target", "eff_counts",
                "pq_l2", "kl_pq", "frac_compact", "frac_transition", "frac_stretched",
                "ancestor_ess", "n_unique_ancestor", "max_ancestor_frac", "repl_cumulative"]:
        diag[key] = np.asarray(diag[key])
    if verbose:
        d = opes.diagnostics()
        print(f"opes         : {diag['runtime_seconds']:.1f}s  neff_frac={d['neff_frac']:.2f} "
              f"nker={d['n_kernels']} bias_range={d['bias_range']:.2f}")
    return diag
