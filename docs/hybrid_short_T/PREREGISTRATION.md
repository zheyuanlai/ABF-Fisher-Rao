# Hybrid wall-clock study at the shortened equal budgets: ABF vs ABF + FR (preregistration)

*Written 2026-10-18 and frozen in git, together with `configs/hybrid_short_T/campaign.json`, before any benchmark,
validation run or production run of this study. Changes after the freeze are dated amendments at the end of this
file.*

## 0. What was seen before writing

Seen before writing:
* every result of the equal-budget study, including `results/equal_budget_v2/short_T/{tables.md,summary.json}`;
* the earlier wall-clock work:
  * Experiment III: torch on an H200 at N = 512 (`docs/mechanism/EXP3_PARALLEL_EFFICIENCY.md`);
  * Experiment B: the walker-parallel CPU backend, its validation and its wall-clock comparison
    (`docs/final_campaign/EXPB_*.md`);
* the FR overhead breakdown (`docs/equal_budget/FR_OVERHEAD.md`).

No data of this study exists.

Facts of the task statement, checked against the repository at `0d683ccb`. All still hold.

| fact | where | status |
|---|---|---|
| torch on one H200 at N = 512 is slower than one CPU core for LTA (0.64× ABF, 0.52× FR) and FR costs ≈ 25 % per step there | `EXP3_PARALLEL_EFFICIENCY.md` §2 | confirmed |
| the walker-parallel CPU backend gives 8.2–9.1× at N = 512, and its best-FR vs best-ABF wall-clock comparison is unresolved (LTA 150 K 0.717 [0.536, 1.196]; gateway, exploratory, 0.763 [0.515, 1.109]) | `EXPB_WALLCLOCK.md` | confirmed |
| the short-T results are truncations of the stored equal-budget runs (no fresh dynamics, no wall-clock) | `scripts/equal_budget/short_T_ladder.py` docstring | confirmed |
| the torch gateway run was never done in Experiment III (projected 16.9 GPU-h, over its ceiling; about 9× slower per step than one core at N = 512) | `EXP3_PARALLEL_EFFICIENCY.md` §1 | confirmed |
| the small-N event caps differ between torch and numba: gateway torch `floor(0.08 N)`, numba `max(1, floor(0.08 N))` (differ at N < 13); LTA torch `int(0.02 N)`, numba `max(1, int(0.02 N))` (differ at N < 50: torch FR is silently off) | `scripts/mechanism/bench_torch_engines.py` (`gateway_knobs`, `lta_sim_config` raise `EquivalenceError`) | confirmed |

## 1. Deviation from the requested design: CPU only

All four H200 NVL on this node were at 98–100 % utilisation with other users' jobs when this study started
(2026-10-18 07:13 UTC):
* GPU 0/1: `yifanchen` and `yesom`;
* GPU 2/3: `hyt35`, 24 GB and 58 GB training jobs.

GPU 3 is the project's standing device. The task forbids benchmarking on a GPU with competing workloads. Asked
which device to use, the user answered **"CPU only"**.

Consequences, all fixed now:
* **No H200 is used in this study.** The CPU/H200 crossover is **not measured** and the GPU-hour guard is 0.
* **Figure A** shows the H200 as "not measured". The 2026-10-10 Experiment III point (LTA, N = 512, torch eager,
  H200 #2) appears only as a historical marker, labelled as such.
* **The accelerator for large N becomes the validated walker-parallel CPU backend** (`src/parallel/`, up to 16
  cores). The hybrid deployment is "one core for small N, several cores for large N where this is measurably
  faster". The routing rule is §4.
* The task's "stronger practical CPU baseline" (walker-parallel CPU) is thereby **inside** the primary analysis,
  for both methods, not hidden in a side analysis.
* The GPU numerical-validation items of Stage I (torch vs numba) are answered **by code reading only**, with no
  execution, and recorded as such in `VALIDATION.md`.

The scientific question keeps its form: can many short, walker-parallel ABF + FR replicas beat the strongest
eligible ABF deployment (any N, any validated CPU backend) in measured wall-clock time to a fixed accuracy, at the
shortened equal budgets?

## 2. Systems, budgets, algorithms (frozen; no tuning)

| system | h | N₀, T* | B (walker-steps) | N ladder | thresholds e_F (strict / **mid** / loose) | production seeds |
|---|---|---|---|---|---|---|
| entropic gateway (β = 16) | 2.5 × 10⁻⁵ | 2048, 12.6 | 1 032 192 000 | 2048 … 1 | 0.002 / **0.0055** / 0.02 (reduced units) | 9600–9631 (32) |
| ethane / LTA 300 K | 2 × 10⁻⁴ | 1024, 51.3 | 262 656 000 | 1024 … 1 | 0.03 / **0.094** / 0.5 kJ/mol | 30500–30515 (16) |
| ethane / LTA 150 K | 2 × 10⁻⁴ | 1024, 23.1 | 118 272 000 | 1024 … 1 | 0.03 / **0.13** / 0.5 kJ/mol | 15700–15715 (16) |

**Budget rule.**
* n_steps(N) = B / N, an integer at every rung (checked by `tests/test_hybrid_short_T.py`).
* T_N = n_steps(N) h. ABF runs at every N; ABF + FR runs at N ≥ 2, never at N = 1.
* B counts integration steps. Force evaluations follow each engine's accounting: gateway N k at completed step k;
  LTA N (k + 1), because the LTA engine evaluates forces at step 0. Both numbers are reported, never conflated.

**Algorithms.** The `engine_cfg` of `configs/equal_budget_v2/*_production.json` is unchanged:
* histogram (P0) ABF;
* uniform-target marginal FR;
* gateway: physical-time ramp 4 t.u., FR every 0.004 t.u., cap max(1, ⌊0.08 N⌋);
* LTA: FR from step 20 000, every 5 steps, cap max(1, ⌊0.02 N⌋);
* clips, warm-up and burn-in as in production; float64.

Diagnostics are off; both engines' diagnostics are bit-inert.

**Reference and scorer.** The frozen references `results/equal_budget_v2/references/` and the accepted e_F scorer
(`eqb_metrics.make_scorer`) are used unchanged.

**Save grid.** The rung's original equal-budget production grid (200 uniform + 24 log-spaced budget fractions of
B_orig/N, plus the physical-time checkpoints) restricted to steps ≤ n_steps(N).
* The last save is exactly n_steps(N): the k*-th original uniform point, with k* = 63 / 171 / 77.
* Every save of this study is therefore at a step where the original production run also saved.
* "Budget grid" = the 24 log points + the k* uniform points. It is the same set of fractions of the run at every N
  and is the grid on which persistence is judged (the repository's cross-N `tau_bgrid` convention). The
  physical-time checkpoints are extra saves used for figures only.

## 3. Stage I: validation gates (before any benchmark result is used)

Run by `scripts/hybrid_short_T/validate.py` and recorded in `docs/hybrid_short_T/VALIDATION.md`. Any FAIL blocks
the affected backend for the affected system, which then runs on S1 only; the reason is recorded.

**V1. Budget and grid.** Integer n_steps; T_N equals the short-T table's; the last save is n_steps; every save is a
save of the original production grid.

**V2. Truncation identity.** The serial engine (S1) driven by this harness, at a production seed and the short
budget, reproduces the stored equal-budget production run bitwise at every common save (M_all, C_all; LTA also
M_prod, C_prod).
* Configurations: one large N and one small N per system, ABF and FR.
* This checks the claim that a short run is the long run truncated, and that the harness drives the engine
  exactly.

**V3. Parallel replay identity at the campaign configurations.** The parallel engine in replay mode equals S1
bitwise at threads {1, 4, 16}, ABF and FR, short windows:
* gateway: N 2048 and N 64;
* LTA 300 K: N 1024, 256 and 16. This is the first validation at 300 K; the Experiment B gates were 150 K only;
* LTA 150 K: N 1024.

**V4. Counter-mode thread invariance.** Counter mode is bitwise invariant over threads {1, 8, 16}: LTA 300 K
N 256 and gateway N 2048, both arms.

**V5. Arm pairing.** In both backends, the ABF and FR arms of one (N, seed) are bitwise identical up to FR's first
realised event:
* S1 and P_t, all three systems;
* checked on a short window at N = 64 (gateway) and N = 1024 (LTA, from FR start).

The torch GPU path is not run (§1). Its pairing defect is documented from code:
* gateway `resample_indices` permutes every slot at every FR opportunity, so the arms are unpaired after step 0;
* its caps differ at small N (§0).

**V6. Timing hooks are inert.** S1 with the SaveClock hooks equals plain `run_arm` bitwise. The parallel engine
with timers on equals timers off bitwise.

**V7. Scorer.** The harness's e_F on stored production files equals the equal-budget metric cache
(`results/equal_budget_v2/<sys>/analysis/run_metrics`) at every common save.

**V8. Schedules.**
* LTA: no FR opportunity before step 20 000; the first one is at nxt = 20 000.
* Gateway: fr_every = 160 steps, ramp_steps = 160 000, FR from step 0.
* Caps at N = 2, 4, 16, 64 and 2048 are identical between S1 and P_t.

**V9. Timing instrumentation.**
* Per-save stamps are monotone and finite.
* The engine's wall_total_s is ≤ the process wall measured by the driver.
* The parallel engine's in-kernel CLOCK_MONOTONIC stamps agree with the driver's perf_counter span to within 1 %
  + 0.05 s.

**Post hoc (reported, not a gate).** Law equivalence of counter vs replay noise:
* At every parallel-routed N, the campaign's counter-mode runs (fresh seeds) are compared with the stored validated
  production runs truncated at u* (32 / 16 seeds): ratio of median e_F(T_N) and Ibar_F, 95 % bootstrap CI.
* A null control (two validated-engine blocks) calibrates the band. The Experiment B V-B4 lesson was that a fixed
  ±18 % band failed the validated engine against itself.
* A clear law difference (the CI excludes 1 by more than the null control's spread) is reported. It triggers the
  law-insensitive sensitivity analysis (§6, hybrid projection).

## 4. Stage II: CPU throughput benchmark and the frozen routing rule

**Configurations.**
* Every N of each ladder; ABF at every N, FR at N ≥ 2.
* Backends: S1, and P_t for t ∈ {1, 2, 4, 8, 16} with t ≤ N.

**Window.**
* The production configuration truncated to n_b(N) = max(n_min, ⌈W_b / N⌉) steps: gateway W_b = 6.4 × 10⁷,
  n_min = 0; LTA W_b = 1.2 × 10⁷, n_min = 25 000.
* Saves at 20 uniform steps (+ step 20 000 for LTA).
* Steady-state cost c_ss = Δwall / (N Δsteps) from the first save ≥ max(n_b/2, fr_start) to the end. FR is active
  over the whole window in every FR configuration.

**Protocol.**
* 3 repeats, with seeds bench_seeds[r]. The configuration list is in a fixed shuffled order, rotated by one third
  per repeat.
* Up to 6 runs at a time, each on its own exclusive CCD set with telemetry, under the same conditions as
  production.
* Contended runs are repeated (§5).
* Recorded: warm-up / JIT time, end-to-end time, peak memory, per-phase timers (parallel engine), FR overhead
  (FR / ABF c_ss at the same N and backend, per repeat).

**Routing rule (frozen here; its output depends only on ABF timings).** Per system and N:
1. P* = the fastest parallel thread count by median ABF c_ss. Configurations within 3 % of it count as tied; ties go
   to fewer threads.
2. Route N to P* iff median c_ss(P*) ≤ median c_ss(S1) / 1.05 **and** every repeat of P* is faster than every
   repeat of S1. Otherwise route N to S1.
3. ABF and FR at that N both run on the routed backend. This is the common-backend primary analysis.

The resulting `routing.json` is written once, with the engine source hashes, before any production run. Production
refuses to start without it or with changed engine sources.

**Secondary (deployment-optimised).** The same rule is applied to each method's own timings. Where it disagrees
with the common route, the alternative wall time is a **projection**: measured W × c_ss(own route) / c_ss(common
route). It is labelled as a projection everywhere.

**Resource projection.** After the benchmark, the used core-hours of production and confirmation are projected
from the routed c_ss (+ 15 s per run). If over the cap, see §8.

## 5. Stage III: production

**Runs.**
* Per system: every N; ABF and (N ≥ 2) FR; prod_seeds; the routed backend; the full shortened budget B.
* Each run is a fresh process pinned to its exclusive CCD set, with 1 Hz telemetry (the Experiment B allocator
  and monitor, `scripts/final_campaign/expB_alloc.py`).

**Order.**
* Seed-major blocks; within a seed, the arm order is `default_rng(20261018 + seed).permutation`.
* The three systems are interleaved by block, so slow host drift cannot align with N or method.

**Smoke.** The first 2 seeds of every system run first. They are inspected for pipeline faults only: contract,
completeness, scorer, contention, ledger. Accuracy outcomes are not inspected. The runs are kept as production runs.

**Records.** `results/hybrid_short_T/prod/<system>/<arm>/s<seed>[.rep<k>].{npz,json}`:
* science arrays and per-save wall stamps and force evaluations;
* configuration, N, T_N, B, method, backend (implementation, threads, noise), CPU allocation and host;
* seed, precision, engine source hashes, git commit;
* process start, import, warm-up and timed-start epochs;
* telemetry sidecar, contention verdicts, status.

Writes are atomic (temp + rename). Complete runs are skipped by signature, and a complete file of a different
signature is never overwritten. A manifest with sha256 is built at the end.

**Contention and repeats.**
* A run contended over its whole timed window is repeated (new process, new allocation), at most 3 repeats.
* Engines are deterministic: a repeat must reproduce the science arrays bitwise (sha256). Only its timing is new.
* Per endpoint, the first replicate uncontended over that endpoint's window is used. Runs without one are flagged;
  they enter the primary analysis with replicate 0 and are excluded in a sensitivity analysis.

**Confirmation stage.** After the exploration analysis (§6):
* per system, the best ABF configuration and the best FR configuration (argmin of median W_mid; inf = censored) are
  re-run on 32 fresh confirm_seeds each, on the same backend;
* a method whose best median W_mid is inf has nothing to confirm; this is reported.

## 6. Stage IV: analysis (`scripts/hybrid_short_T/analyze.py`)

**Per run.**
* e_F at every save: the frozen scorer.
* **τ_ε**: the first budget-grid save after which e_F ≤ ε at **every** later budget-grid save. This is a
  retrospective diagnostic, not an online stopping rule.
* **W_ε** = wall_s_at_save at τ, in seconds since the engine's timed start. It excludes process start, imports,
  JIT / cache load and the engine's warm-up run.
* **W_ε^e2e** = W_ε + (timed start − process start).
* **Censored**: the threshold is not sustained by the end of the budget. W = ∞; nothing is imputed.
* Also per run: force evaluations, used core-hours and allocated core-hours to τ; e_F(T_N) (last save); Ibar_F (mean
  over the k* uniform saves); completion wall; the timing-model check (completion wall vs B × benchmark c_ss).

**A. Endpoint accuracy per (system, N, method).**
* Median and IQR of e_F(T_N) and Ibar_F.
* Paired FR / ABF ratio: median paired log-ratio, 10 000 seed resamples, 95 % CI; wins.

**B. Wall-clock time to accuracy per (system, N, method).**
* Median W_mid with censoring counts. The inf-aware median is ∞ if more than half the runs are censored.
* Paired FR / ABF W ratio and wins, at every N.
* Also W_loose and W_strict, descriptive.

**C. Best allocation (exploratory).**
* `eqb_metrics.best_allocation` on W_mid. Candidates: ABF at every N, FR at every N ≥ 2, on their routed backends.
  Resampling is per N over seeds, with the minimising N re-selected in every resample.
* R_wall = best FR median / best ABF median, with a 95 % CI from the resampled relative difference.
* Reported: best N and backend per method, the frequency with which each N wins, success / censoring counts,
  used / allocated core-hours, and force evaluations to accuracy.

**Confirmatory.**
* R_wall^conf on the confirm seeds, with the selection held fixed (unpaired bootstrap of the ratio of medians).
* Decision: **"FR faster"** iff the 95 % CI is below 1; **"ABF faster"** iff above 1; otherwise **"unresolved"**.
* A CI that crosses 1 is never called equivalence.

**D. Wall-clock frontier (descriptive).**
* Per method, at wall budget w: the minimum over the eligible (N, backend) of the median e_F(w).
* e_F(w) is the error of the latest save at or before w.
* Runs that completed before w carry their final estimate forward; this is drawn distinctly and is never presented
  as continued sampling.

**E. Confounders.** For each, how it is checked:
* backend: the same backend for both arms at fixed N;
* precision: float64 throughout;
* estimator and reference: frozen scorer, frozen references;
* FR strength: the frozen `engine_cfg`;
* GPU setup / sync overhead: no GPU used;
* contention: telemetry, repeats, and the sensitivity analysis excluding flagged runs;
* initial conditions: shared slot by slot between arms;
* selection bias: confirmation on fresh seeds, plus the re-selection bootstrap;
* selection of the shortened budget: T* was chosen on the original production seeds; this study uses fresh seeds.

**Sensitivity analyses.**
* W^e2e;
* throughput uncertainty: every W of a configuration multiplied by its benchmark max/min c_ss ratio, in the
  direction against the method;
* contended runs excluded;
* hybrid projection with the validated engine's law: τ_u of the stored production runs truncated at u*, times B,
  times the routed c_ss;
* deployment-optimised routing (projection where it differs);
* the fixed-largest-N comparison (FR N₀ vs ABF N₀).

## 7. Figures (`scripts/hybrid_short_T/plot.py`)

PNG + PDF in `figures/hybrid_short_T/`, styled like `figures/equal_budget_v2/short_T`.

| figure | content |
|---|---|
| A | Throughput vs N per system: S1 ABF / FR and best-parallel ABF / FR, benchmark ranges, chosen routes; H200 "not measured" plus the historical Experiment III marker |
| B | All-N–T e_F vs **measured** wall-clock, one multi-panel figure per system: median and IQR, both methods, N, T_N, backend, threshold; N = 1 is ABF only |
| C | The same vs physical time |
| D | Best-N frontier vs wall budget |
| E | Time to accuracy per N and the best N of each method, with censoring marked |
| F | Synthesis: three aligned panels (best hybrid FR vs best hybrid ABF, and the fixed largest N) |

Projections are never drawn as measurements: a dashed style with an explicit legend entry.

## 8. Guards and stopping

* **CPU.** 100 **used** core-hours for the whole study (threads × process wall of every run process). Allocated
  (exclusive-CCD) core-hours are reported beside it.
* **GPU.** 0 device-hours.
* **Per run.** Timeout 2 h.
* **Disk.** Pause below 2 GB free.
* **Over the cap.** If the post-benchmark projection exceeds the cap, production proceeds seed-major (every
  configuration gets the same seeds) until the cap. Missing seeds are reported as NOT RUN, which is a resource
  outcome, not a scientific negative. No N and no method is dropped for performing poorly.

**Stopping rules.**
* a validation gate that fails for every backend of a system;
* a missing reference;
* an engine-source change against routing.json;
* the cap.

A poor FR result is not a stopping reason.

## 9. Interpretation map (decided before data)

* **H1 (the hypothesis).** R_wall^conf < 1 with its CI below 1: many short parallel ABF + FR replicas beat the best
  ABF deployment in wall-clock to the mid threshold.
* **H0-type outcomes.**
  * The CI includes 1: unresolved.
  * The CI is above 1: ABF faster.
  * The best FR configuration is at small N (serial): FR's wall-clock advantage, if any, is not a parallelism effect.
* **Hardware outcome.** If no parallel configuration beats S1 by ≥ 5 % at some N, that N runs on S1 and the
  parallel backend plays no role there; this is reported as a hardware result.
* **Claim types kept apart.** Verified numerical identities (Stage I); measured timings (Stage II and the per-save
  stamps); timing-model projections (sensitivity only, labelled); hypotheses; statistically supported conclusions
  (confirmatory decision only); unresolved or incomplete comparisons.

## Amendment A1 (2026-10-18, after Stage I and a 24-job scheduler smoke test in a scratch directory; before any benchmark or production record of this study)

1. **Allocation wait: 60 s → 10 s** (`hst_sched.ALLOC_MAX_WAIT_S`). The campaign.json text above is kept as frozen;
   this amendment overrides it.
   * Every CCD of this host carries 1–2 of other users' unpinned processes. During validation no 16-core set ever
     qualified (max busy of the used cpus 0.28–1.0).
   * A 60 s head-of-line wait per launch would add hours and does not improve timing hygiene: the scheduler moves
     roaming tasks within milliseconds.
   * Contention is judged, as frozen, by the run's own 1 Hz telemetry over its timed window (repeat if contended).
   * The busy fraction of the used cpus at launch is recorded in every record (`max_busy_used_at_launch`).
2. **V9 wording.** An external perf_counter span of an engine call necessarily contains untimed set-up: framework
   load, LTA self-check, result assembly. "Agree within 1 % + 0.05 s" therefore does not test the clock. V9 is
   implemented as follows:
   * stamps finite and non-decreasing;
   * the engine's warm-up + timed loop ≤ the caller's span;
   * the untimed remainder is reported (measured 0.001–0.21 s);
   * `time.perf_counter` is `clock_gettime(CLOCK_MONOTONIC)`, the clock of the parallel kernel's in-kernel stamps.
3. **Validation harness fix (no effect on any timed run).**
   * `scripts/mechanism/parallel_benchmark.py` sets `NUMBA_NUM_THREADS=1` in `os.environ` at import, and numba
     re-reads its config at the next compilation.
   * A validation check that ran S1 before P_t in one process therefore got a one-thread pool, and 8 V3 LTA checks
     raised (ERROR, not FAIL).
   * `validate.py one` now imports it first and restores the launch environment; the 8 checks were re-run and pass.
   * Production children never mix the two (serial children use one thread; parallel children do not import
     parallel_benchmark).
