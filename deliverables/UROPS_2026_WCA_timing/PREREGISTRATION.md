# WCA dimer: does earlier Fisher–Rao activation improve validated ABF convergence? (preregistration)

*Written 2026-10-20 (UTC) before any production run of this study. Machine-readable design:
`frozen_config.json`. Both files, the engine extension and its tests are committed before the first confirmation
run; every raw result records that commit.*

## Background

In the validated Phase-V confirmation (`docs/numerical_validation/WCA_CONFIRMATORY.md`; dt* = 1.25 × 10⁻⁴, N = 256,
T = 240, exact Metropolis-MC reference), uniform FR from t = 40 is harmful: I_F +9.9 % [+2.3, +13.2], e_F(T)
+131 %. ABF alone is already accurate by t ≈ 40 (median e_F 0.023 kT).

The earlier start-time ladder (`docs/WCA_FR_START.md`) found early starts beneficial. It ran at dt = 0.002, whose
invariant law is wrong, and was scored against a reference affected by that integrator, so its gains are not
validated free-energy improvements. This study re-asks the timing question with the validated physics and reference.

## Hypotheses (tested, not assumed)

* **H1 (early activation).** FR from t = 0 improves early free-energy convergence relative to FR from t = 40.
* **H2 (early stopping).** FR from t = 0, stopped at t = 20 (or 40), avoids the late-time penalty of continuous
  FR.
* **H3 (marginal vs free energy).** Gains in marginal uniformity need not translate into free-energy accuracy.
* **H4 (schedule sensitivity).** The FR effect depends on the activation and stopping times at fixed physics and
  rate.

## Frozen setup (identical to Phase V except the FR schedule)

* WCA solvent + double-well dimer, 100 particles per replica, β = 1, K = N = 256 replicas.
* Validated float64 Euler–Maruyama, dt = 1.25 × 10⁻⁴, T = 240 (1 920 000 steps).
* P0 histogram ABF, 160 bins on [−0.2, 1.2]. ABF warm-up (linear ramp) to t = 20; reported-estimator burn-in /
  reset at t = 20.
* FR:
  * uniform target, KDE bandwidth 0.07;
  * rate γ = 0.1, constant while active, no ramp;
  * opportunities every 0.01 t.u. (80 steps); score clip 2;
  * cap ⌊0.02 K⌋ = 5 per opportunity, cap_min = 0;
  * the existing death-driven, score-weighted-refill law;
  * constant population.
* Every other knob is `ACCEPTED_CFG` / the Phase-V cfg, unchanged: force field, clips, soft RC wall, initial-condition
  law, mean-force estimator, genealogy (ESS window 8 t.u.). See `frozen_config.json → engine_cfg_template`.
* **Reference.** Exact Gibbs by Metropolis MC (64 chains; `rescore_wca_history.mc_reference`), frozen in
  `data/summary/reference_mc.json` (array sha256 f58f069c…). The analysis recomputes it and asserts equality. No
  trajectory of this study touches it.

## Engine extension (smallest backward-compatible change)

`src/wca_numba.run_ladder_point` gains two keyword-only options. `ACCEPTED_CFG` is unchanged.

* `fr_stop_steps`: None (default) = never stop. Otherwise FR opportunities occur only at completed-step counts
  `next_step < fr_stop_steps`, so FR events happen only at t < t_stop. Afterwards the replicas continue plain ABF
  with their configurations and the shared estimator: no reset, no restart.
* `fr_stream_ids`: a fixed FR stream per arm, SeedSequence child 1 + id. An arm's stream never depends on which other
  arms exist.

The engine also saves cumulative ABF bias-clip counters at every checkpoint. This changes no dynamics.

Tests, all to pass before production:

* `tests/test_wca_numba.py` and `test_wca_numba_sham.py`: the original 43, on the modified engine;
* `tests/test_wca_numba_stop.py`, 12 new tests:
  * bitwise equality with the original engine when stopping is disabled;
  * pre-activation identity;
  * common prefix of continuous and stopped arms on one stream;
  * no events or opportunities after the stop;
  * stop = 0 equals ABF;
  * population and histogram conservation;
  * matched physics;
  * inert clip counters;
  * `ACCEPTED_CFG` unchanged;
  * scorer compatibility with the exact MC reference;
* `scripts/regression_validated_prefix.py`: the extended engine reproduces the stored Phase-V seed-5100 ABF and
  FR-from-40 trajectories bitwise through t = 45 (FR active over 40–45).

## Arms (all run; decided from the cost estimate before any production run)

| arm | FR start | FR stop | FR stream id | priority |
|---|---|---|---|---|
| `abf` | — | — | — | mandatory |
| `fr_start0` | 0 | never | 1 | mandatory |
| `fr_start40` | 40 | never | 0 | mandatory (the validated schedule) |
| `fr_start0_stop20` | 0 | 20 | 2 | mandatory |
| `fr_start0_stop40` | 0 | 40 | 3 | mandatory |
| `fr_start5` | 5 | never | 4 | secondary |
| `fr_start20` | 20 | never | 5 | secondary |

* **Cost.** Validated Phase-V wall times are 0.85–0.88 core-h per 256-replica, 1.92 × 10⁶-step run, so the full
  design is 7 × 16 × 0.88 ≈ 99 core-h, under the 200 core-h cap. The secondary arms are therefore included.
* **Seeds.** 16 fresh paired seeds, **5600–5615**. They overlap neither 5100–5115 (Phase V) nor 3300–3307 (the
  dt = 0.002 ladder), and no WCA result or document uses them.
* **Pairing.** Every arm of a seed uses the seed's lattice initial condition and its Langevin noise (one process per
  seed and arm, shared noise stream). FR arms differ in their own FR streams.

## Saving schedule

* Every 0.5 t.u. on [0, 20], every 1 t.u. on (20, 80], every 5 t.u. on (80, 240]: 133 checkpoints, including
  t = 0, 5, 20, 40, 80 and 240.
* Each checkpoint stores actual simulation states (no interpolation):
  * both estimators' accumulators;
  * the instantaneous z histogram and region fractions;
  * cumulative replacements, event opportunities and cap binding;
  * run-long and windowed genealogy;
  * cumulative bias-clip counters.

## Metrics

* **e_F(t):** the Phase-V read-out `wca_numba.score_ladder_result(...)['l2_f_t']`, aligned L² on [−0.1, 1.1].
* **e_F′(t):** `l2_fp_t`, the histogram arms' own mean-force error on [−0.1, 1.1].
* **e_ξ(t):** ‖p̂ − 1/1.4‖ on [−0.2, 1.2] from `z_hist`, the full-domain diagnostic. Also reported, each labelled:
  * the **interior-only** error (158 interior bins, renormalised);
  * the **edge-bin fraction** (z_hist[0] + z_hist[159]) / N, since the engine clamps the soft-wall tails into the
    edge bins.
* **Windows:** I_F^{[a,b]} = trapezoid of e_F over the saves with a ≤ t ≤ b, for [0, 20], [0, 40], [20, 40],
  [20, 80], [40, 240] and [0, 240]. Also e_F(240).
  * Sensitivity (pre-registered): (20, 80], i.e. the trapezoid from t = 21. The t = 20 save is the estimator reset,
    where the production estimator holds one step of deposits (validated ABF median e_F there: 0.55).
* **Time to accuracy:** sustained first-passage times at ε ∈ {0.05, 0.04, 0.03, 0.02, 0.015} kT.
  * These were chosen from the validated ABF-only arm alone: each ε is reached by ABF in ≥ 14/16 seeds, with
    medians 27.5 to 137.5 t.u.
  * Censored if e_F(240) > ε, never counted as reached.
  * Per-seed paired ratio min(τ_ABF, T) / min(τ_FR, T), with censoring counts.
* **Genealogy and safety:**
  * replacements in total and on [0, 20), [20, 40) and [40, 240];
  * minimum windowed ancestor ESS / K;
  * maximum ancestor family fraction;
  * unique ancestors;
  * bias-clip fraction;
  * non-finite states.

## Statistics and decision rules

* **Unit and summary.** The seed is the unit. Per seed G = I_arm / I_base − 1 (the same for final errors). Report the
  median G with a percentile bootstrap CI (10 000 resamples, `default_rng(20261020)`) and the number of seeds with
  G < 0.
* **Primary contrasts** (Bonferroni over two; 97.5 % CI decides, 95 % CI reported alongside):
  * **P1:** `fr_start0` vs `fr_start40` on I_F^{[0,40]}. SUPPORTED if the CI is < 0, CONTRADICTED if > 0, otherwise
    UNRESOLVED. Note: `fr_start40` equals `abf` bitwise before t = 40, so P1 is in effect early FR against ABF on
    [0, 40].
  * **P2:** `fr_start0_stop20` vs `fr_start0` on e_F(240). Same rule.
* **Secondary** (descriptive):
  * every FR arm vs `abf` on I_F^{[20,80]}, I_F^{[0,240]}, e_F(240) and the other windows;
  * the stop arms vs `abf` and vs `fr_start40`;
  * `fr_start0_stop40` vs `fr_start0` on e_F(240);
  * e_F′, the marginal metrics, genealogy and time to accuracy.
* **Early-convergence benefit** (an FR arm vs `abf`): the 95 % CI of G on I_F^{[20,80]} lies below 0.
* **Safe early-stopping benefit** (a stop arm): an early-convergence benefit, and final-error non-inferiority vs
  `abf`, meaning a 95 % CI upper bound of G on e_F(240) below +5 %.
  * DEGRADED if the CI lower bound is above +5 %.
  * Otherwise the final-accuracy conclusion is INCONCLUSIVE.
* No best-looking schedule is promoted post hoc. Neutral, negative or inconclusive results are reported as such.

## Exclusions, failures, tuning

* No seed is excluded for its outcome.
* A crashed or non-finite run is rerun once. The engine is deterministic, so a reproducible failure is reported as a
  failure, listed in the validation report, and the statistics use the seeds with complete data for the arms
  compared.
* Nothing is tuned after any production result is read: physics, estimator, FR parameters, seeds, windows,
  tolerances, statistics.
* Raw `.npz` results are git-ignored, in `results/urops_2026_wca_timing/raw/` of the main checkout.
