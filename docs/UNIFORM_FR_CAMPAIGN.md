# Uniform-FR acceleration campaign — preregistration

**Frozen 2026-08-29, before any production run of this campaign.**
Branch: `main` (per instruction — no side branch). Base commit at freeze: 662f2fc.

## Question

Does periodic **uniform-target** marginal Fisher–Rao reallocation make ABF compute
F(z) **faster at the same physical sampling budget**?

The target is frozen to q_uni(z) = 1/|M| (normalized on each system's grid /
torus). No EMA target, no oracle target, no sham, no OPES, no rate ladder.
Exactly **two arms everywhere**:

    abf            plain ABF
    fr_uniform     identical ABF backbone + periodic FR toward the uniform marginal

Everything else — initial conditions, seeds, replica count, timestep, ABF
estimator/grid/bandwidth, warm-up, reference, evaluation mask, run length — is
matched within a system, inherited verbatim from that system's closed frozen
configuration. FR mechanics (rate, cadence, burn-in, KDE bandwidth, score clip,
event cap) are inherited from each system's previously frozen FR arm and are
**not** re-tuned for the uniform target.

## Hypothesis (paper-level)

Uniform mFR is a finite-time **establishment accelerator** for ABF, not a change
of the asymptotic estimator. Expected regime pattern:

- establishment-limited (Gateway, WCA Case IX): acceleration;
- ABF-sufficient (alanine): neutral / self-throttling;
- discovery-limited: no rescue (not tested this round).

## Systems and inherited frozen configurations

### Stage 1 — Entropic Gateway (establishment-limited positive control)
- Engine: `src/gateway_core.py`; kernel target_mode `uniform` (already implemented).
- All sampler/cell values inherited from
  `results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json`
  (beta=16, s=0.10, r=32, beta_H=8 kT; N=2048, n_steps=100000, dt=4e-4,
  save_every=500; eta=0.10, h=0.07, fr_every=10, fr_burnin=0, ramp 0.10,
  score_clip=3.0, max_event_fraction=0.08, target_ema_rate=0.005 [inert for a
  uniform target], ess_window_steps=4000).
- gamma(fr_uniform) = **1.5**, inherited from the frozen practical rate. Not tuned.
- Seeds 100–131 (32), inits [left, one_right]; both arms in ONE batch per chunk
  → shared initial conditions and Langevin noise (paired by construction).
- Frozen-bias endpoint stage kept as in the confirmatory protocol.
- Campaign prereg twin: `configs/uniform_campaign/gateway_prereg.json`.
- Output: `results/uniform_campaign/gateway/`.

### Stage 2 — WCA dimer, corrected-reference Case IX cell
- Engine: `src/wca_abffr_core.py` via `src/wca_phase_jobs.py`; `fr_uniform`
  target already implemented (`fr_target_uniform_torch`).
- Cell and run frozen from Case IX v2: beta=1, h=2, w=2, n_dim=10, a=1.5;
  N=1024, 120000 steps, save_every=2500; config
  `configs/wca_phase_diagram_production.yaml` (eval window [-0.1, 1.1]).
- FR knobs identical to the frozen `fr_estimated` block = the YAML's existing
  `fr_uniform` block: rate 0.10, fr_every 5, fr_start 20000, score_clip 2.0,
  max_event_fraction 0.02. Not tuned.
- Reference: **corrected** TI reference `cache/phase_hp_v3/` ONLY (the old
  `results/wca_production` / `cache/phase` reference is superseded and is not
  evidence).
- Seeds 400–415 (16). Both arms run **fresh in the same process per shard**
  (paired = same initial conditions; the process-level determinism caveat from
  the v2 prereg is why the old abf npz files are not reused).
- `--store-profiles` on, so pmf_t / mean_force_t / kl_pq_t series are saved.
- Output: `results/uniform_campaign/wca/`.

### Stage 3 — Alanine dipeptide, vacuum (phi,psi) (ABF-sufficient neutrality control)
- Engine: `src/alanine/core2d_ala.py` + `scripts/run_alanine_study.py`.
- This code path supports only (abf, fr_oracle) today; the campaign ADDS
  `fr_uniform` (uniform density on the torus grid) with the same score/event
  mechanics and NO reference access (extends the structural no-leakage assert).
- All dynamics/estimator values inherited from `configs/alanine/pilot.yaml`:
  T=300 K, dt=0.001 ps, 100 ps, N=2048, n_grid=97, abf_bandwidth=0.08,
  kde_bandwidth=0.15, fr_start 20 ps, fr_every 0.5 ps, score_clip 2.0,
  max_event_fraction 0.05, lineage_reset 6 ps.
- fr_rate = **0.02**, inherited from the safety-only calibration
  (`results/alanine_oracle/calibration/fr_rate_selection.json`). Not re-tuned.
- Seeds 0–15 (16 paired seeds; label semantics as in the original study),
  init c7eq, init_seed 4242, rng_seed 20260903.
- Reference: `results/alanine/reference/reference.npz`; eval window 20–100 ps;
  primary endpoint kernel-matched integrated aligned-L2
  (`int_eF_km_equilibrium`), exactly as in `src/alanine/metrics_ala.py`.
- Output: `results/uniform_campaign/alanine/`.

Stage 4 (ethane/LTA zeolite) is licensed only by the gate below and is out of
scope for this round.

## Endpoints (identical across systems)

Let e_F(t) = the system's own frozen error norm vs its frozen reference.

1. **Primary: I_F = ∫ e_F(t) dt** over the system's full saved horizon
   (trapezoid on the saved checkpoints, exactly as each engine already computes
   it). Statistic: per-seed paired relative change
   ΔI_F = 100·(I_F^uni − I_F^abf)/I_F^abf, median over seeds, 10000-resample
   bootstrap CI of the median. ΔI_F < 0 means faster.
2. **Time-to-accuracy** T_eps with the convergence-atlas convention:
   first t with e_F ≤ eps sustained for 0.2·T; eps ∈ {e0/2, e0/4, e0/8,
   abf_final}. Speedup S_eps = T_eps^abf / T_eps^uni. Censoring is reported,
   never imputed.
3. **Final error** e_F(T), reported always (transient vs persistent
   acceleration must be visible; the Gateway EMA arm's late reversal is the
   reason this cannot be hidden).
4. **Mechanism**: KL(p_t || uniform) (or the closest stored marginal-mismatch
   series), region/basin occupancy, and F/mean-force profile snapshots.
5. **Genealogy**: ancestor ESS/N and event fractions, against each system's
   previously declared health floors (gateway 0.30/0.05; WCA 0.10/0.05;
   alanine ess 0.30 / wmax 0.05 / cumulative events ≤ 5 %).

## Success criteria (frozen now)

Per system:
- **Acceleration-positive**: median ΔI_F ≤ −10 % AND bootstrap CI95 upper < 0.
- **Safe accelerator**: additionally median final relative change ≤ +5 %
  (non-inferiority margin).
- **Neutral**: |median ΔI_F| < 10 % and final change within the same margin.
- Any health-floor violation is reported next to the verdict and blocks the
  "safe" label regardless of the error numbers.

Expected/desired outcomes: Gateway accelerates (question: does uniform keep the
early gain without the EMA arm's late reversal?), WCA accelerates persistently,
alanine is neutral with FR events self-throttling toward zero.

**Gate to Stage 4 (LTA)**: at least one of {Gateway, WCA} acceleration-positive
with no genealogy collapse, AND alanine not catastrophically degraded.

## Held fixed / prohibited

- No tuning of any FR parameter against e_F, I_F, or T_eps.
- No additional arms, targets, cells, systems, or seed extensions after data
  are seen.
- Thresholds and this file do not change after the first production run starts.
- Compute: GPU 3 only (H200), per current user instruction; multiple processes
  on that one device are allowed.

## Existing-evidence side deliverable (no new runs)

ABF vs fr_uniform curves already on disk are re-plotted as context, clearly
labeled, with the superseded-reference WCA tree excluded from evidence:
butane phi1, pentane phi1, pentane R15, entropic bottleneck beta=8,
entropy-dominant bottleneck, WCA representative cells, 2-D toy. These carry
their original selection caveats and are context, not confirmatory results.
