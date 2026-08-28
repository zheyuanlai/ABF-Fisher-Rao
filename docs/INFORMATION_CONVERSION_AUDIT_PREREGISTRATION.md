# Information-Conversion Audit — Preregistration

**Campaign**: `information-allocation-fr`
**Frozen**: 2026-08-28, before any scientific run of this campaign.
**Base commit**: `c7b2d44` (branch `q-r-decoupling`, "No accessible variance-dominated
regime: established on three independent levers").
**Owner**: zheyuanlai (single owner; GPU 2 only, one process).

## The question

Can a **single, genealogy-safe, standard Fisher–Rao birth–death pulse** toward the
**finite-horizon information-optimal** reaction-coordinate allocation produce genuinely
more *effective conditional-force information* — not merely moved particle occupancy?

This is a mechanism audit, not a schedule search. The q–r campaign established that
repeated BD toward an information allocation pays compounding genealogy (A4b: ancestor
ESS ~8/256), while the *bias* realization of the same allocation works; the
`unflattened-target` oracle campaign used an asymptotic target with no counts, no
horizon, no cooldown and slowed both F thresholds (0.960×, 0.962×). Neither isolates
the missing arrow:

    instantaneous FR population → future sampling effort → realized estimator information.

One pulse toward the *actual* finite-horizon optimum, followed by a decorrelation-length
cooldown, is the last clean way to ask it.

## Hard methodological constraints

Particle reallocation ONLY via the existing `fr_v3.bd_standard` (with `fr_v3.FRScore`
and `fr_v3.bd_timestep` for the dose). Forbidden anywhere in the experimental runner:
systematic/multinomial/stratified resampling, random or count-balancing turnover,
transport relocation, bias-held realization of the target, score clipping, event caps,
jitter after cloning, q/r weighted-particle reallocation, holdout of clones. A unit test
asserts the runner module references none of the forbidden operators; a config test
asserts the pulse count is exactly one. FR birth/death itself deposits no ABF
observation (the pulse executes *after* the step's deposit; a clone first speaks after
its next propagation — clean-v2 Gate-B semantics, tested).

Forking a saved checkpoint into arms is experimental replication, not a reallocation
operator.

## Provenance and code reuse

- Branch `information-allocation-fr` is based on `q-r-decoupling` (NOT on
  `unflattened-target`, which stays frozen) because the validated K2/K3 kappa-family
  instrument, the leverage/allocation machinery, and the Stage-1B–validated difficulty
  reference machinery live there (Spearman(Γ̂, Γ_ref) = 0.976/0.976 on K2/K3, mirror
  rank correlation 0.980, `results/qr_decoupling/stage1b/gamma_validation.json`).
- The experiment runner is a NEW module (`src/abffr/info_conversion.py`) that reuses
  the audited building blocks verbatim: `potentials.*_torch`, `torch_utils`,
  `kappa_family.kappa_at_torch`, `allocation.leverage`/`cell_reduce`,
  `fr_v3.{FRScore, bd_timestep, bd_standard, clone_mask, replacement_count}`,
  `reference.compute_reference`, `io_utils.make_rng_streams`,
  `simulation._init_positions`. It is structurally unable to invoke the q-r arms
  (`qr_arms`), `balanced_representation`, or any `v3`/`v4`/`clean_v2` FR path.
- **Engine-parity gate** (runs before science): the new runner's plain-ABF step map
  (propagation + deposit + estimator arithmetic) is compared against
  `simulation_torch.run_batch` with identical injected noise on a short run; max abs
  difference of `F_hat` and of the trajectory must be < 1e-9 (float64). This is the
  "testing the engine ≠ testing the pipeline" lesson applied in advance.

## Frozen backbone (identical to the validated q-r stage-2 configuration)

| Quantity | Value |
| --- | --- |
| Systems | kappa cells **K2** (a=log16, shift 0) and **K3** (a=log16, shift 3.0) |
| Potential | 2-D double-well, `x_tilt = 0.1021665783` |
| Domain | x ∈ [−3, 3], y ∈ [−2.5, 3.5], reflecting |
| β, dt | 4.0, 0.002 |
| K (replicas) | 256 |
| Grid | 401 nodes on x; reference ny = 801 |
| Allocation cells | J = 32 equal cells on [−3, 3] |
| Evaluation window | geometric, [−2.5, 2.5] (margin 0.5) — fixed a priori |
| ABF estimator | `binned_smooth`, h = 0.05, `update_every` = 10, `min_count` = 1.0 |
| Observation order | `post_propagation` |
| KDE bandwidth η | 0.10 |
| Init | x, y uniform; `make_rng_streams(seed)` (matched-seed convention) |
| dtype / device | float64 / CUDA (GPU 2), single process |
| Burn-in checkpoint | step 10,000 (t = 20) |
| Full horizon (continuation only) | 50,000 steps (T = 100) |

**Discovered deposition semantics (inspected, not assumed).** In
`simulation_torch.run_batch` with `observation_order: post_propagation` and the
`binned_smooth` estimator, the ABF accumulators `C_acc`/`S_acc` receive **one
observation per replica per physical step** (deposit of `dV/dx(X_prop, Y_prop)+tilt`
at the nearest grid node, every step); `abf.update_every = 10` controls only the
cadence of the grid-estimate refresh (`recompute_grid`) and of the difficulty stream.
Therefore the number of raw deposition opportunities per replica during a horizon of
`H` steps is `H`, and

    M = K × H.

A unit test asserts total deposited counts = K × n_steps on a plain run.

**Noise realization.** The new runner uses chunk-keyed Langevin noise: chunk `c`
(500 steps) for seed `s` is drawn from a fresh generator seeded
`stable_seed("langevin-chunk", 0, s, c)`. Noise is keyed by (seed, step, slot) and is
therefore identical across all arms forked from one seed — FR changes which
configuration occupies a slot, never which variates the slot receives (Appendix-A.6
convention). This also makes the continuation resumable from a saved state without
generator-state archaeology. The FR pulse consumes a separate stream keyed
`stable_seed("fr-pulse", 0, s)`, shared across doses of one seed (common random
numbers: a higher dose fires a superset of the first uniform draw).

**Seeds** (fresh, disjoint from every block used in this repo: 5100s/5200s/5300s q-r,
7000s C60, 7200 elsewhere):

- Pilot: **8000–8007** (8 seeds), both cells.
- Confirmation: **8100–8131** (32 seeds), both cells.
- Stage-0A reference: **5100–5103** — deliberately the *same* seeds as the validated
  Stage-1B reference runs, because Stage 0A *is* that machinery rerun to save the full
  per-cell profiles (only summary statistics were archived). The reference is
  evaluation-only/oracle and never an experimental arm.

## Stage 0A — oracle local-information reference

For each cell j: `V_j = sigma_f,j^2 · tau_j`, from the **validated long-run reference
machinery** of `scripts/validate_qr_gamma.py::gamma_reference`: arm A2 (estimator on,
allocation off), 200,000 steps, K = 256, seeds 5100–5103, full history capacity;
read `qr_gamma_final` (the frozen decomposed Γ̂ = σ̂²τ̂) and `qr_tau_final` (τ̂ by the
frozen bias-corrected AR(1) fit, in physical time units; observation interval
`update_every · dt = 0.02`). `V_j` and `tau_j` are the across-seed **medians**.

Recorded honestly: the brief prefers "fixed-x hidden-fibre" machinery. The repo's
reusable, *validated* reference machinery is the long-run A2 estimator above (validated
against exactly this construction in Stage 1B); the fixed-x τ measurement quoted in
`information.py` comments was a one-off not present as reusable code. Per the brief's
own rule ("do not build a new IAT estimator tonight"), we freeze the A2 long-run
reference. Its τ is a *lower* bound on the pure fibre τ (cell-population turnover
decorrelates the cell-mean series faster), so the resulting H is conservative in the
direction of a *shorter* claimed horizon.

Leverage: `a_j = cell_reduce(leverage(x_grid, mask), cell_of_grid, J)` — pure grid
geometry plus the geometric mask; identical construction to the q-r arms. Neither
`a_j` nor `V_j` reads R12, basin labels, barrier locations, or any arm outcome.

## Stage 0B — planning/cooldown horizon

`tau_max` = max of reference `tau_j` over cells with `a_j > 0` (the evaluation cells).
Per cell family (K2 and K3 separately):

    H = H_plan = H_cool = ceil(tau_max / dt)   [integration steps]
    M = K × H                                  [raw deposition opportunities]

No safety factor. Units test: counts and M both count per-replica-per-step deposits.

## Stage 0C — actual finite-horizon information target

At each seed's burn-in checkpoint, `C_j` = raw hard-cell deposit counts accumulated by
the **diagnostic hard-cell accumulator** (cell = the J=32 allocation cells; one count
per replica per step; identical units as M; never touches the ABF dynamics).

Solve, per seed:

    min_{pi in Delta_J} sum_j a_j V_j / (C_j + M pi_j)
    s.t. sum_j pi_j = 1,  pi_j >= 1/K  for all j

by monotone bisection on the KKT multiplier λ:
`pi_j(λ) = max(1/K, (sqrt(a_j V_j M / λ) − C_j)/M)`; `sum_j pi_j(λ)` is continuous and
non-increasing, so the root is unique. The 1/K floor is a coverage constraint (≥ 1
expected replica per cell) and guarantees a strictly positive FR target; it is not
tunable. Unit tests: normalization to 1e-12; floor satisfaction; KKT stationarity on
free cells (equal `a_j V_j M/(C_j+M pi_j)^2`); predicted risk ≤ uniform (`pi = 1/J`);
invariance under common rescaling of `a_j V_j`.

## Stage 0D — oracle opportunity gate (computed BEFORE any FR)

Per seed: `R_opt = sum_j a_j V_j/(C_j + M pi*_j)`, `R_unif = sum_j a_j V_j/(C_j + M/J)`,
`G_ideal = 1 − R_opt/R_unif`. Also reported for context (not gating): the asymptotic
Neyman comparator `R_asym(r)/R_asym(unif)` with `r ∝ sqrt(a_j V_j)`, `R_asym = Σ a_j V_j/r_j`
— so the earlier favorable *asymptotic* risk ratio can be compared with the *finite-
horizon* opportunity at this checkpoint.

**STOP rule**: if `median_seeds(G_ideal) < 0.10` in **BOTH** K2 and K3, all FR stages
are cancelled and the verdict is `NO_FINITE_HORIZON_ALLOCATION_OPPORTUNITY`. No tuning
of burn-in, M, τ, or the target to evade this stop. (If at least one cell passes, the
frontier runs in both cells; the pilot gates below still require both cells to pass.)

## Stage 1 — single-pulse FR mechanism frontier

At the checkpoint (immediately after the step-10,000 deposit and grid refresh), fork
each seed into 5 arms sharing the identical state and future noise:

- **plain ABF** (no pulse),
- **one `bd_standard` pulse** at dose `p90 ∈ {0.02, 0.05, 0.10, 0.20}`.

Score: existing KDE `p_hat` (η = 0.10, binned, normalized) and piecewise-constant
target density `q*(x) = pi*_cell(x) / cell_width`, both evaluated at particle positions
by the engine's `interp1d`; `S = FRScore(log p̂, log q*).S` (centered; no clipping).
Dose: `dtau = fr_v3.bd_timestep(score, p_max=p90, quantile=0.90)` — i.e.
`dtau = −log(1−p90)/s90`, `s90 = Q_0.9(|S|)`. If `s90 = 0`: record
"target indistinguishable", no dose invented. Then exactly ONE `bd_standard` call;
FR permanently off; ordinary ABF for `H_cool` steps. No holdout, no jitter; clones
first deposit at the next step's propagation.

### Stage-1 diagnostics (recorded per seed × arm)

1. Target movement: `KL_post/KL_pre` and `TV_post/TV_pre` of (p̂ ‖ q*) across the pulse.
2. Genealogy: ancestor ESS/K (post-pulse; unchanged during cooldown), local ancestor
   ESS in cells with `pi*_j >` pre-pulse occupancy (min and median across gaining
   cells), max family fraction, events and replacements.
3. Future raw allocation: `N_future,j` (hard-cell deposits during cooldown),
   `r_future = N_future/ΣN`, and `TV(r_future, pi*)` compared FR vs plain ABF.
4. **PRIMARY**: realized effective information. `fhat_s,j` = cumulative hard-cell mean
   force (diagnostic accumulator, from t = 0 through t_b + H_cool; never feeds the
   bias). Reference `f_ref,j` = node-average of `Fprime_ref` over the grid nodes of
   cell j (the estimand of a within-cell-uniform sampler; both arms share it, and the
   within-cell occupancy-weighting residual is reported by the bias term below).
   `R_s = sum_j a_j (fhat_s,j − f_ref,j)^2`; across seeds the paired ratio
   `R_FR/R_ABF`, plus the decomposition `R_var = Σ a_j Var_s(fhat)` and
   `R_bias = Σ a_j (mean_s fhat − f_ref)^2`.
5. Sibling local-force correlation vs time since pulse (clone/continuation pairs,
   pooled across seeds) — diagnostic only, never a gate.
6. `e_F`/`e_F'` time profiles are recorded (eval cadence 500 steps) but are NOT inputs
   to dose selection; the selection code takes only the mechanism-gate table.

## Pilot decision rule (frozen)

8 matched seeds (8000–8007), both cells, plain ABF + all four doses. A dose is
pilot-positive only if in **BOTH** K2 and K3:

- median `KL_post/KL_pre` ≤ 0.90;
- median ancestor ESS/K ≥ 0.90;
- median `[TV(r_future_FR, pi*) / TV(r_future_ABF, pi*)]` ≤ 0.90;
- realized information-risk ratio `mean_s(R_FR)/mean_s(R_ABF)` ≤ 0.90 **and** its
  paired-bootstrap (10,000 resamples over seeds, seed-paired) 95% upper bound < 1.

Selection: the **smallest** p90 passing all gates in both cells. Frozen outcomes:

- No dose reaches KL ≤ 0.90 with ESS/K ≥ 0.90 → `FR_STRENGTH_GENEALOGY_CONFLICT`.
- Some dose moves the target safely but no safe dose improves realized risk →
  `FR_DOES_NOT_CONVERT_REPRESENTATION_TO_INFORMATION`.
- Favorable direction but < 10% reduction → `WEAK_SIGNAL`, not confirmed.

## Confirmation

Selected dose + plain ABF on 32 fresh seeds (8100–8131), both cells; identical target
construction, H rule, one-pulse operator; no parameter changes. Mechanism PASS = the
same four gates.

## Optional long continuation (only after mechanism PASS in both cells)

Continue the same confirmed trajectories (state saved at t_b + H_cool) to step 50,000
(T = 100); **no second pulse**. Endpoints: the frozen q-r thresholds
(`results/qr_decoupling/thresholds.json`) — stringent `eps_2`, secondary `eps_1` —
with the frozen `analyze_qr_stage2.py` conventions verbatim: τ_ε = first frame with 3
consecutive frames ≤ ε on the geometric window; restricted-mean speedup
`S_ε = E[min(τ_ABF,T)]/E[min(τ_FR,T)]`; paired bootstrap; asymmetric censoring veto.
Practical PASS: `S_ε ≥ 1.15` at `eps_2` with paired 95% lower bound > 1, no extra
censoring, final `e_F` ratio ≤ 1.05. Secondary only; can never change the mechanism
verdict or the dose.

## Top-level verdicts (exactly one)

1. `NO_FINITE_HORIZON_ALLOCATION_OPPORTUNITY`
2. `FR_STRENGTH_GENEALOGY_CONFLICT`
3. `FR_DOES_NOT_CONVERT_REPRESENTATION_TO_INFORMATION`
4. `MECHANISM_PASS_FEC_WEAK`
5. `MECHANISM_AND_FEC_PASS`

No new target, cadence, repeated-pulse schedule, online IAT algorithm, or extra
hyperparameter may be created from these results during this autonomous run.

## Audits (before science)

- Existing test suite (relevant subset) green — recorded.
- New unit tests: constrained water-filling (5 properties); runner can invoke only
  `bd_standard`; BD deposits zero observations at the pulse step; counts/M unit
  consistency; configured pulse count exactly one; dose selection is structurally
  blind to FEC columns; torch/numpy cell-assignment parity; engine-parity gate.

## After runs

Independent re-analysis from saved raw CSVs; machine-readable summaries compared;
verdict JSON; full result markdown; git commit, configs, seeds, environment, failures
and NaNs recorded; raw tables retained. Negative results are reported as negatives.

## Amendments

(append-only, timestamped, with reason)

### Amendment 1 — 2026-08-28, after Stage 0A, BEFORE any Stage-0C/0D computation

**Observation.** The frozen reference machinery returned `tau_max(eval)` = 0.096
(K2) / 0.108 (K3) time units, so H = 49/55 steps and M/ΣC ≈ 0.5% of the
checkpoint's accumulated observations. The campaign's own recorded one-off
measurements (comment block in `abffr/information.py`, commit `a52e955`) put the
slow-cell force decorrelation at ~4.7 time units at fixed x and ~1.2 with
x-motion — an order of magnitude above the A2 cell-mean-series τ̂, which is
shortened by cell-population turnover. The A2 τ̂ is validated as an input to
Γ̂ = σ²τ̂; as an estimate of the *fibre* decorrelation the horizon rule intends,
it is now known to be a large underestimate.

**What does NOT change.** Stage 0D runs exactly as frozen: H from the A2
machinery, the 0.10 gate, and the stop semantics. No FR runs if the frozen gate
stops. No experimental arm, dose, target, or endpoint changes.

**What is added (reported-only).** A sensitivity sidecar, computed from the
already-saved Stage-0 checkpoint counts by pure arithmetic (no simulation of any
experimental arm, no FR): `G_ideal(H')` on the horizon grid
`H' ∈ {H_frozen, 250, 600, 1200, 2350, 6000}` steps, plus `H_fib` derived from a
fixed-x fibre τ measured at the cell centres with the frozen AR(1) estimator
(`information.tau_from_lag1` applied per-walker at fixed x — the frozen
estimator, the Gate-0I sampling design). These numbers are attribution context
for a possible `NO_FINITE_HORIZON_ALLOCATION_OPPORTUNITY` verdict — whether the
stop means "no exploitable heterogeneity at this checkpoint" or "none within a
horizon that is short by construction". They cannot change tonight's verdict,
cannot license FR runs tonight, and are not inputs to any gate.

**Reason recorded before the gate:** this amendment is written before any
`C_j`, `pi*`, or `G_ideal` has been computed, so it cannot be outcome-contingent
on the gate it annotates.
