# Equal-budget replica ladders: ABF vs ABF + uniform Fisher–Rao (preregistration)

*Written 2026-10-10, before any production ladder run. Frozen in git together with
`configs/equal_budget_v2/*.json`.*

Seen before writing:
* the numerical-validation results (`docs/equal_budget/GATEWAY_TIMESTEP_VALIDATION.md`,
  `docs/numerical_validation/LTA_VALIDATION.md`);
* the historical ladders at the old gateway dt (`docs/GATEWAY_REPLICA_LADDER.md`) and the WCA ladders;
* the published fixed-N LTA results.

No new ABF/FR ladder data exists.

## 1. Question and design

**Question.** At a fixed total budget of base walker-steps B = N × n_steps, how should computation be split
between the number of replicas N and the trajectory length T_N = B h / N? How does uniform-target marginal
Fisher–Rao (FR) birth–death change convergence at each N?

**Algorithms.** Two only:
* histogram (P0) ABF;
* the same ABF + uniform-target marginal FR.

No sham, no other sampler, no hyperparameter search.

**Systems, timesteps and budgets.** Each timestep was validated independently, without reading FR performance.

| system | h | anchor (N₀, T₀) | B (walker-steps per arm per seed) | N ladder | seeds |
|---|---|---|---|---|---|
| entropic gateway | 2.5 × 10⁻⁵ (validated: `GATEWAY_TIMESTEP_VALIDATION.md`) | 2048, 40 | 2048 × 1.6 × 10⁶ = 3.2768 × 10⁹ | 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 | 32 (8100-8131) |
| ethane/LTA 300 K | 2 × 10⁻⁴ (validated vs exact MC: `LTA_TIMESTEP_VALIDATION.md`) | 1024, 60 | 1024 × 300 000 = 3.072 × 10⁸ | 1024, 512, ..., 2, 1 | 16 (30000-30015) |
| ethane/LTA 150 K | 2 × 10⁻⁴ | 1024, 60 | 3.072 × 10⁸ | same | 16 (15000-15015) |

* n_steps(N) = B/N exactly; every (N, T_N) of a system has the same B.
* ABF and ABF + FR run at every N ≥ 2; ABF alone runs at N = 1.
* Physical model, reference, estimator, 180-bin histogram and FR algorithm are identical across the ladder.

**Frozen algorithm constants** (physical time, converted to steps with the frozen h).

*Gateway:*
* histogram ABF, 180 bins, min_count 1 (Γ = M/(C+1), own bin);
* FR: γ 1.5, KDE bandwidth η 0.1, score clip 3, update interval 0.004 t.u. (160 steps), strength ramp timescale
  4 t.u.;
* cap max(1, ⌊0.08 N⌋);
* left-well initialisation `init_left(seed, N)`.

*LTA:*
* histogram ABF, 180 bins, own-bin P0, bias clip ±60, deposit clip ±480;
* ABF warm-up 4 t.u., estimator burn-in 4 t.u.;
* FR: start 4 t.u., rate 0.2 (both T), KDE bandwidth 0.10 rad, score clip 2, update interval 0.001 t.u. (5 steps),
  max event fraction 0.02;
* molecules start at random α-cage centres.

**Finite-N extension (LTA).** The historical cap `int(0.02 N)` is 0 for N < 50 and silently switches FR off. The
ladder uses **cap(N) = max(1, ⌊0.02 N⌋) for N ≥ 2**.
* This keeps the historical value wherever it is already ≥ 1 (N ≥ 50).
* The probabilistic event law is unchanged: an event is permitted, never forced.
* Every N < 50 point is flagged as using the extension.
* The realised event fraction and ancestor degeneracy are reported.
* It is not a continuum FR limit.

The gateway cap `max(1, ⌊0.08 N⌋)` already contains this floor (historical ladder rule).

## 2. References (independent of the ladder)

**Gateway.** The exact analytic F(x) = H(x² − 1)² + β⁻¹ log ω(x) and F′(x), on the eval window [−1.5, 1.5].
* At h_G the closed-form EM floor is 8 × 10⁻⁵ (F) and 8 × 10⁻⁴ (F′).
* The EM-consistent profile is reported as a secondary reference.

**LTA.** The exact Metropolis-MC Gibbs reference of the 2026-10-09 validation:
* 16 groups × 40 umbrella windows; free energy from fine-bin WHAM; noise 0.006 kJ/mol;
* mean-force reference = MC conditional mean force per bin, per-bin SE 0.014 (300 K) / 0.010 (150 K)
  kJ/mol/rad, circular mean consistent with 0.

Never the published umbrella reference (300 K MARGINAL).

## 3. Saved history (every (N, seed, method))

Common budget grid: 200 uniform u = b/B ∈ (0, 1], plus 24 log-spaced u ∈ [10⁻⁴, 5 × 10⁻³), plus physical-time
checkpoints:
* gateway t ∈ {0.5, 1, 2, 4, 10, 20, 40, 100, 400, 1000, 4000, 10000, 40000};
* LTA t ∈ {0.5, 1, 2, 4, 10, 20, 60, 120, 480, 1920, 7680, 30720}.

Each is kept only if t ≤ T_N, and saved at the nearest integration step with its true t and u. Profile snapshots
are taken at u ∈ {0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00}.

At every save:
* raw ABF accumulators (all-steps; LTA also production), from which F̂_t and F̂′_t (raw and periodically
  projected) are reconstructed;
* instantaneous CV histogram, region fractions, true event counts, first arrivals;
* FR genealogy: whole-run and windowed, unique ancestors, ESS, largest family, realised deaths,
  events-per-opportunity histogram;
* slot traces with ancestor labels;
* cost (force evaluations, wall-clock, peak memory).

## 4. Metrics (frozen)

**Errors.**
* e_F(t): RMS over the eval window (gateway, 151 grid nodes; the accepted scorer) or the full circle (LTA, 180
  bins) of F̂_t − F_ref, after removing the mean difference.
* e_F′(t): gateway, the accepted function-space RMS; LTA, the RMS over bins of the raw histogram mean force minus
  the MC conditional mean force. The periodically projected version is also reported.
* Marginal distance, two quantities on 18 coarse bins against the uniform target:
  * **TV_half(t)**, the total variation of the walkers' visitation histogram over the trailing half of the
    elapsed run, C_all(t) − C_all(≈ t/2), using the nearest saved step and recording the exact window. It is
    defined at every N, including N = 1, and is not dominated by the initial transient. **τ for the marginal uses
    this quantity.**
  * **TV_inst(t)**, the instantaneous walker histogram, reported descriptively with its finite-N floor
    E_N[TV] under multinomial(N, uniform), computed exactly by simulation. At small N it is dominated by
    discreteness (at N = 1 it is always 17/18) and is never used for τ.

**Integrated errors on the budget axis.** Ī_F = mean of e_F over the 200 uniform u; Ī_F′ likewise.

**Persistent time-to-accuracy.** τ(ε) = the first save after which the error stays ≤ ε at every later save.
* It is reported in physical time and in budget fraction u.
* An arm that never meets ε is **censored**: reported as "> T_N" or "> 1", never as success at T.

**Frozen thresholds.** The same for every N within a system/temperature.

| system | e_F strict / mid / loose | e_F′ strict / mid / loose | marginal: TV_half |
|---|---|---|---|
| gateway | 0.002 / 0.0055 / 0.02 | 0.012 / 0.035 / 0.10 | ≤ 0.10 |
| LTA 300 K (kJ/mol; kJ/mol/rad) | 0.03 / 0.094 / 0.5 | 0.04 / 0.12 / 1.0 | ≤ 0.10 |
| LTA 150 K | 0.03 / 0.13 / 0.5 | 0.04 / 0.15 / 1.0 | ≤ 0.10 |

How the thresholds were set:
* **mid** = the previously accepted high-N ABF endpoint accuracy at the anchor (gateway at h_G; LTA vs the MC
  reference);
* **strict** = about ⅓ of the anchor, above the reference/estimator noise floors (gateway EM floor 8e-5; LTA MC
  noise 0.006 kJ/mol, P0 floor 0.003, mean-force reference SE 0.014);
* **loose** = early establishment.

**Population establishment.** The first save after which the instantaneous fraction in the far state stays ≥ ½ of
its uniform-target value:
* gateway: right well x > 0.5, target 0.361;
* LTA: window |z| < 1.5 Å, target 0.252.

It is reported with the caveat that it is ill-defined at N ≤ 2, where cumulative visitation is used instead.
Also reported: first arrival (gateway right well; LTA window, opposite cage) and cumulative true crossings.

**Paired comparisons at each N.** For Ī_F, Ī_F′, final e_F, final e_F′ and τ:
* G_X(N) = (X_FR − X_ABF)/X_ABF per seed;
* median with a 10 000-resample seed bootstrap 95 % CI, and wins;
* absolute values always reported alongside.

**Best allocation.** min_N Ī_F^ABF(N) vs min_{N≥2} Ī_F^FR(N), each with a bootstrap over seeds that re-selects
the minimising N in every resample. This handles winner's-curse bias, reported as the frequency with which each N
wins. The same is done for final error and τ.

## 5. Interpretation map (decided before data)

* **Outcome A**: FR faster at the same N, but the best ABF-only N beats the best FR N at equal B.
* **Outcome B**: FR's best beats every ABF-only N.
* **Outcome C**: FR improves the marginal (TV, establishment) but not F̂ or F̂′.
* **Outcome D**: FR helps at intermediate/large N and is neutral or harmful at small N.
* **Outcome E**: system or temperature dependence.

All are valid outcomes, and several can hold at once. Finite-N mechanisms (establishment starvation, KDE score
noise, excessive birth–death, genealogy collapse, ABF converging without help) are tested against the saved
diagnostics, not assumed.

## 6. Execution and resources

* **Order.** LTA 300 K on the coarse N {1024, 256, 64, 16, 4, 1}, then {512, 128, 32, 8, 2}; then LTA 150 K in the
  same order; then the gateway ladder.
* **Engines.** numba CPU engines `src/lta_ladder_numba.py` and `src/gateway_ladder_numba.py`, validated against
  the production engines (replay equivalence, law tests, bitwise checkpoint/resume, inert diagnostics) **before**
  any production run.
* **Arms.** One process per (system, T, N, seed, method). ABF and FR at the same (N, seed) share initial
  conditions and Langevin noise slot by slot; FR uses an independent random stream.
* **Ceilings.** 1000 CPU core-hours and 24 GPU device-hours.
  * Estimated: LTA about 130 core-h per temperature; gateway about 60 core-h; no GPU.
  * A resource ledger is kept in `docs/equal_budget/EXPERIMENT_LOG.md`.
* **Stopping rules.** A failed validation or equivalence gate, an unavailable reference, a numerical instability,
  or the resource ceiling. A poor FR result is **not** a stopping reason. No N is dropped or re-budgeted. A run
  that cannot finish is reported as NOT RUN with the reason.
