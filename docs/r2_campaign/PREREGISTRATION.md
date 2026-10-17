# Preregistration — online FR stop rule R2 (occupancy-triggered): cross-system test

* Branch `claude/ecstatic-bhaskara-b3b3ea` (worktree `.claude/worktrees/ecstatic-bhaskara-b3b3ea`), cut from `main`
  at 6716aefc. Raw run files are git-ignored and live in the main checkout under
  `/home/zheyuanlai/ABF-Fisher-Rao/results/r2_campaign/raw` (symlinked as `results/r2_campaign/raw`); analysis JSONs
  are committed under `results/r2_campaign/{dev,analysis}`.
* **Frozen by the commit that adds this file, before any confirmatory run.** Later changes are dated amendments or
  errata at the end; this header is never edited.
* Development record (replay only, no new runs): `DEVELOPMENT.md`, written and fixed before this file.

## 0. Rules (unchanged from the adaptive-FR campaign)

* **Exactly two arms in every cell**: histogram (P0) ABF, and the same ABF + uniform-target marginal Fisher–Rao
  birth–death stopped online by R2. No sham, placebo, matched-turnover, oracle, OPES or metadynamics arm. Exact
  references appear only as references.
* **Pairing.** 16 fresh paired seeds per cell; the arms share the initial-condition and noise streams and are bitwise
  identical before the first FR opportunity (asserted by the analysis). Force evaluations are equal (asserted).
  Zero FR events after the stop (asserted).
* **Inference.** The seed is the unit; per-seed effects first, then the median over seeds with a 95 % percentile
  bootstrap CI (10 000 resamples, seed 2026). No seed is discarded; censoring at the horizon.
* **Metrics** as frozen in `docs/convergence_fr_campaign/PREREGISTRATION.md` §2 and implemented in
  `scripts/convergence_fr/atlas_lib.py`. The sustained time to accuracy is an offline metric, never an input to the
  controller.
* **Compute.** Incremental on the adaptive-FR campaign's 100 CPU core-h authorisation (≈ 44.1 used before this
  campaign). The estimated cost of every block is stated in §6 before launch; `COMPUTE_LEDGER.md` keeps the account.
  No GPU.

## 1. The frozen rule R2

* **Code.** `src/eri/fr_controller.py` at this commit (`R2`, `OnlineStop`, `window_mask`); both engines construct the
  controller with their own window mask (`src/eri/engine.py` `Run.__init__`; `src/lta_ladder_numba.py` `run_arm`).
  The default paths of both engines are bitwise unchanged (existing tests). R1 is untouched: its code path, spec
  and outputs are identical (the 35 pre-existing tests pass unchanged).
* **Spec.**
  ```
  R2 = dict(name="R2", trigger="occupancy", check_every=0.25, W=1.0, R=18, level=0.75, persist=1, stop_factor=2.0,
            support_count=200.0, stability_tol=0.05, stability_gate=False, essw_min=0.15, maxlin_min=0.10, maxlin_k=6.0)
  ```
* **Definition** (all times in t.u., checks at `t = t_start + n × 0.25`, FR starts at each cell's validated fixed
  start = the end of the bias ramp; only the stop is online):
  * window bins = the fine deposit bins whose centre lies in the engine's barrier window (`|phi| < window_phi`
    for ITE/ERI, 28 of 180 bins; `|z| < window_half`, `z = |phi| a / 2π`, for LTA, 46 of 180 bins);
    `share` = (window bins)/n_grid, the flat-bias deposit share of those bins;
  * `occ(t)` = Σ_window (C_j(t) − C_j(t−W)) / Σ_all (C_j(t) − C_j(t−W)) over the trailing W = 1 t.u.
    (every walker deposits every step, so this is the time-averaged window fraction);
  * `established(t)` = `occ(t) ≥ 0.75 × share` and every bin has `C_j ≥ 200` and `t ≥ t_start + W`;
  * `t_est` = the first check with `established`; the decision is taken at that check (persist 1);
  * **FR stops at `t_stop = max(2 × t_est, t_est)` = `2 t_est`**, absolute time from t = 0;
  * the stability statistic `‖Γ_t − Γ_{t−W}‖/‖Γ_{t−W}‖` (over bins supported at t − W) is logged at every check with a
    `stable` flag (≤ 0.05) but **does not gate** the decision (`stability_gate: False`; DEVELOPMENT.md §3.1);
  * the genealogy-safety backstop of R1 is unchanged: at every check until FR has stopped, windowed ancestor
    ESS/K < 0.15 or max lineage share > max(0.10, 6/K) stops FR at that check; safety only brings a stop earlier;
  * a stop is never postponed, FR never restarts; a run in which neither condition fires keeps FR to the horizon and
    is reported as "never stopped".
* **Reference-free by construction.** The module imports only `math` and `numpy` and does no I/O (AST test). The
  controller receives, at each check, only the run's own cumulative birth/death histograms, deposit counts and
  force sums and its windowed ancestor labels. The window mask is the engine's region-bookkeeping convention, not a
  reference.
* **Tests** (`tests/test_fr_controller.py` 25, `tests/test_lta_fr_stop.py` 7, plus `test_fr_stop.py`,
  `test_eri_engine.py`, `test_diag_age.py`: 49 passed at this commit): R2 synthetic decision logic (ramp → stop at
  2 t_est; support guard; stability logged not gating; safety; irrevocability; persist/factor semantics; state
  round-trip with the mask), engine-level determinism under chunking, equality with the fixed-stop path,
  checkpoint/resume, inertness for ABF, exclusivity with a fixed stop, mask conventions in both engines.
* **Development.** ITE (all blocks) and ERI K1024 only; LTA C3/C4 were a read-only check; ERI K256 was not loaded.
  Every candidate examined is in DEVELOPMENT.md §3. **One departure from the brief's first-stated primary
  candidate:** the stability guard is logged, not gating (it never binds in ITE or LTA and in ERI it would delay the
  stop from 4.0 to 5.6 t.u., past continuous FR's crossing time 5.2 [4.9, 5.6]).
* **No candidate comparison in the confirmatory cells.** No alternative rule is evaluated against R2 below.

## 2. Primary statistic and verdict rules (frozen)

* **Analysis script.** `scripts/adaptive_fr/analyze_online.py`, **unchanged** from the adaptive-FR campaign (commit
  6716aefc; it is not modified by this campaign). It asserts pairing, equal force evaluations, config differences
  limited to `fr_controller` (and inert keys), and zero FR events after the stop.
* **PRIMARY per cell.** S_geo = the per-seed geometric mean, over the cell's five-rung ladder, of the restricted
  sustained time-to-accuracy speedup τ_ABF/τ_FR (censored at T); median over seeds with bootstrap CIs. Ladders as in
  the adaptive-FR campaign: kT ladder {0.080, 0.064, 0.048, 0.040, 0.032} k_BT for ITE 300 K and ERI;
  ABF-anchored ladder {3.0, 2.5, 2.0, 1.6, 1.3} × median ABF e_F(T) for ITE 150 K and all LTA cells.
* **Verdict rules** (`scripts/r2_campaign/verdicts.py`, frozen with this commit; it only re-reads the per-seed S_geo
  and the secondary CIs from the analysis JSON and applies the rules below; the bootstrap is the campaign's):
  * **Superiority cells** (R2C1, R2C2, R2C3, R2C4, R2H80, R2H350): SUPPORTED if the CI of the median S_geo is > 1,
    CONTRADICTED if < 1, else UNRESOLVED. Reported at 95 % and at the **Bonferroni level for 8 cells**
    (1 − 0.05/8 = 99.375 %); **cross-system claims use the Bonferroni CI.**
  * **Equivalence cells** (R2C5, R2N256; ERI): the ceiling in ERI is "no harm", not a gain (continuous FR's early
    post-ramp Ī_F over [1, 10] was only −2.5 % [−3.7, −1.0] at K1024 and −4.3 % [−8.3, +3.1] at K256; its 0.1 T bias
    ratio was 0.89–0.91; R1's early window was +6 % [−0, +13]). **TOST**: EQUIVALENT if the **90 % CI** of the median
    S_geo lies inside **[1/1.25, 1.25] = [0.80, 1.25]** (two one-sided tests at 5 %); HARMFUL if the 95 % CI is < 1;
    BENEFICIAL if the 95 % CI is > 1 and not inside the margin; else UNRESOLVED. The margin ±log 1.25 is about
    1.5–2.5× the 95 % half-widths seen in the R1 cells (log-scale 0.08–0.15), so a true null is resolvable with 16
    seeds; it is far inside the harm seen before (R1 0.61 [0.53, 0.71]; continuous 0.51 [0.49, 0.58]).
  * **Endpoint flag** (secondary, both kinds of cell): e_F(T) relative change, 95 % CI: "penalty" if the lower bound
    > 0; "gain" if the upper bound < 0; "removed" if the upper bound < +10 %; else "unresolved".
* **Rule-level questions (prespecified):**
  * **Q-ITE (in-sample).** Does R2 reproduce R1's acceleration? R2C1 and R2C2 SUPPORTED.
  * **Q-LTA (held out).** Does R2 preserve the benefit on systems it never saw? R2C3 (positive control) SUPPORTED and
    R2C4 not CONTRADICTED.
  * **Q-NEW (held out, new temperatures).** R2H80 SUPPORTED (strong positive candidate); R2H350 not CONTRADICTED
    (borderline candidate; its prediction is S_geo ≥ 1 with a CI that may include 1).
  * **Q-ERI (expected-negative).** R2C5 and R2N256 EQUIVALENT or BENEFICIAL by TOST; HARMFUL in either = R2 harms ERI.
  * **"R2 preserves the acceleration"** iff, at the Bonferroni level: R2C3 and R2H80 SUPPORTED; R2C1, R2C2, R2C4 and
    R2H350 not CONTRADICTED.
  * **"R2 avoids ERI-type harm"** iff both ERI cells are EQUIVALENT or BENEFICIAL.
  * **"R2 generalises by the preregistered definition"** iff both of the above hold. Otherwise the failing cells are
    reported with their observables and the verdict is "does not generalise".
* **Online-rule outcomes per cell** (reported regardless): stop-time distribution (median, IQR, range), reasons,
  never-stopped count, FR replacements per run, events after the stop (must be 0), min windowed ESS/K and max
  lineage, window establishment ABF → FR, translocations, wall-clock.
* **Secondary outcomes** as in the adaptive-FR campaign §C.2 (post-ramp, full, early and late Ī_F; e_F at T/4, T/2,
  3T/4, T; e_F′(T); post-stop-only estimate; crossing; marginal TV; genealogy).
* **Record.** No ERI run with an FR stop below 5.5 t.u. has ever been performed (R1's earliest ERI stop was 5.5,
  S 0.96; the fixed-stop rule of the convergence campaign refused 5 t.u. because it lay above the crossing's 2.5 %
  quantile 4.9). R2's predicted ERI stop is 4.0 t.u. in every replayed run. The ERI cells are therefore the first
  test of an ERI stop inside FR's useful window.

## 3. Cells

| block | system (validated config reused unchanged) | K | T | dt | FR rate / start | genealogy window | seeds | ladder | reference | role | predicted R2 stop (DEVELOPMENT.md §4) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R2C1 | ITE 300 K (`jobs_E1.json` config) | 1024 | 50 | 1e-4 | 2.0 / 1 t.u. | 2.5 t.u. | 1400000–015 | kT | `results/overnight/ite/reference` (exact MC) | in-sample, superiority | 5.0 [4.75, 5.25] |
| R2C2 | ITE 150 K (`jobs_E2.json` config) | 256 | 100 | 1e-4 | 0.5 / 1 t.u. | 5 t.u. | 1410000–015 | ABF-anchored | `results/overnight/ite150/reference` (exact MC) | in-sample, superiority | 8.5 [7.25, 10.75] |
| R2C5 | ERI 600 K (`jobs_ERI_rate2_K1024.json` config) | 1024 | 40 | 2e-4 | 2.0 / 1 t.u. | 2 t.u. | 1440000–015 | kT | `results/eri_campaign/reference` (exact MC) | in-sample, **equivalence** | 4.0 |
| R2C3 | LTA 150 K (histogram production config, numba engine) | 1024 | 60 | 2e-4 | 0.2 / 4 t.u. | 4 t.u. | 1420000–015 | ABF-anchored | `results/equal_budget_v2/references/lta_T150_reference.npz` (exact MC) | **held out** (positive control), superiority | 18.6 [17.6, 20.6] |
| R2C4 | LTA 300 K (same) | 1024 | 60 | 2e-4 | 0.2 / 4 t.u. | 4 t.u. | 1430000–015 | ABF-anchored | `lta_T300_reference.npz` (exact MC) | **held out**, superiority | 14.6 [13.4, 15.8] |
| R2H80 | LTA 80 K (same; rate 0.10 = the sweep's safety-ladder rate at 80 K) | 1024 | 60 | 2e-4 | 0.1 / 4 t.u. | 4 t.u. | 1450000–015 | ABF-anchored | `lta_T80_reference.npz` (built 2026-10-17, see §5) | **held out, new temperature**, superiority | about 32 |
| R2H350 | LTA 350 K (same; rate 0.2) | 1024 | 60 | 2e-4 | 0.2 / 4 t.u. | 4 t.u. | 1460000–015 | ABF-anchored | `lta_T350_reference.npz` (built 2026-10-17, see §5) | **held out, new temperature**, superiority (borderline) | about 14 |
| R2N256 | ERI 600 K (`jobs_ERI_rate2_K256.json` config) | 256 | 40 | 2e-4 | 2.0 / 1 t.u. | 2 t.u. | 1470000–015 | kT | `results/eri_campaign/reference` (exact MC) | **held-out allocation**, **equivalence** | about 4 |

* Job files: `configs/adaptive_fr/jobs_R2*.json` (from `scripts/adaptive_fr/make_jobs.py R2C1 …`), committed with this
  file. ITE/ERI cells: both arms carry the same config except `fr_controller`; LTA cells: both arms carry the same
  `fr_rate` (inert for ABF) and differ only in `fr_controller`. LTA saves every 500 steps (0.1 t.u.); ITE every
  1250 steps (0.125 t.u.); ERI every 500 steps (0.1 t.u.).
* Seeds 1400000–1470015 have never been used in this project (the adaptive-FR campaign used 1140000–1380002 and the
  earlier campaigns smaller values).

## 4. Why these held-out cells

* **LTA 150 K and 300 K.** R2 never saw an LTA trajectory while any choice was made (DEVELOPMENT.md §2); C3/C4 were
  read-only checks. They test the rule on a different engine, estimator bookkeeping and window convention, with the
  exact MC references of the equal-budget campaign.
* **LTA 80 K (new strong-positive candidate).** No stop rule has seen this temperature. Under the histogram
  estimator continuous FR gave Δe_F(T) −83 % and ΔI_F −35 % (16/16) with 1.9× the window crossings
  (`docs/LTA_HISTOGRAM_REPLICATION.md` §4.7); ABF establishes the window at 37 t.u. (FR 16 t.u.) on the historical
  0.6-t.u. save grid, so it is the most establishment-limited cell in the project. The sweep's rate at 80 K was 0.10
  (ESS floor 0.26 was below the 0.30 "safe" label there; R2's backstop is 0.15 and will be reported if it fires).
* **LTA 350 K (new borderline candidate).** No stop rule has seen it. Continuous FR gave ΔI_F −12 % (16/16) but
  Δe_F(T) only −9 % [−22, −0.15] and τ to ABF's final accuracy 1.03×; ABF establishes at 8.4 t.u. (FR 6.9). It tests
  R2 where the benefit is smallest among the positive LTA temperatures.
* **References at 80 and 350 K.** No Metropolis-MC mean-force reference exists at these temperatures. The files
  `results/equal_budget_v2/references/lta_T{80,350}_reference.npz` were built on 2026-10-17 (before any R2 run) in
  the format `analyze_online.py` reads, from the sweep's umbrella-sampling/WHAM references
  (`results/uniform_campaign/lta/reference/reference_T{80,350}.npz`: 40 windows, overdamped EM at the production dt
  2e-4). At 150 and 300 K that same protocol agrees with the exact MC reference to 0.017 / 0.034 kJ/mol RMS
  (`docs/numerical_validation/LTA_VALIDATION.md`: 150 K admissible, 300 K marginal). `F_ref` is that F, centred.
  `gamma_ref` is the central finite difference of `F_ref` on the circle (at 150/300 K this is within 0.03–0.04
  kJ/mol/rad RMS of the exact mean force, 0.3 % of its RMS); **the secondary e_F′ statistics at 80 and 350 K are
  therefore finite-difference-referenced and are reported with that label; the primary S_geo uses only F_ref.**
* **ERI K256 (expected-negative, held-out allocation).** The brief's criterion for an expected-negative cell is a
  system in which ABF establishes the window within about 1 t.u. of the ramp end. Verified from existing ABF data:
  ERI 600 K establishes at 2.1 t.u. with the ramp ending at 1.0 (both K256 and K1024; ERI_RESULTS.md). Among cells
  with an existing exact reference it is the only one meeting the criterion; continuous FR was most harmful there
  (e_F(T) +113 %, S 0.52). Its K256 data were **not loaded** during development, so the allocation is held out
  (the system is not). An ITE cell at a higher temperature would be a genuinely new expected-negative framework
  but has no reference and no dt validation at that temperature; its cost (Gate 1 MC reference ≈ 2.2–3.6 core-h by
  the ITE precedent, Gate 2 dt check ≈ 1–4 core-h, the cell ≈ 4.8 core-h, plus the pipeline work) was judged beyond
  this campaign and is listed as a follow-up. The ethane-in-rigid-Si-IHW option (CANDIDATES.md) needs the structure,
  engine adaptation and an MC reference (≥ 12 core-h and the same pipeline work) and is deferred for the same reason.

## 5. Predictions (recorded before the runs)

* Stops as in the table of §3. Every run stops as "established"; no safety stop at K1024/K256 (R1 saw none).
* R2C1, R2C2: S_geo close to R1's (1.67, 1.64), since the stops (5.0, 8.5) are within 0.5 t.u. of R1's (5.5, 8.25).
* R2C3, R2C4: S_geo close to R1's (2.52, 1.46), stops 2–3 t.u. earlier than R1's (18.6 vs 21.5; 14.6 vs 16).
* R2H80: large S_geo (the fixed-stop benefit should dominate; stop about 32 of 60 t.u.).
* R2H350: S_geo ≥ 1, possibly UNRESOLVED.
* R2C5, R2N256: EQUIVALENT (stop 4.0, inside FR's useful window; the seed-level R1 data at stops 5.5–9 gave S
  0.96–0.73 with the earliest stop best). The failure mode to watch is an endpoint "penalty" from the FR-period
  deposits even if S_geo is equivalent.

## 6. Order, cost and stop rules

* **Order.** Wave 1 = the held-out cells R2C3, R2C4, R2H80, R2H350 and R2N256 (5 × 32 single-threaded processes,
  160 cores). Wave 2 = the in-sample cells R2C1, R2C2 and R2C5 (96 cores), launched when wave 1 has finished.
  Launchers: `scripts/r2_campaign/launch.sh` (ITE/ERI engine) and `launch_lta.sh` (LTA), registry
  `results/r2_campaign/logs/launches.tsv`.
* **Cost estimate** (measured single-thread costs of the adaptive-FR campaign, COMPUTE_LEDGER.md there):

  | block | per run | runs | core-h |
  |---|---|---|---|
  | R2C1 ITE K1024 × 50 | 0.227 | 32 | 7.2 |
  | R2C2 ITE K256 × 100 | 0.110 | 32 | 3.5 |
  | R2C5 ERI K1024 × 40 | 0.150 | 32 | 4.8 |
  | R2N256 ERI K256 × 40 | 0.033 | 32 | 1.1 |
  | R2C3, R2C4, R2H80, R2H350 LTA K1024 × 60 | 0.075 | 4 × 32 | 9.6 |
  | development replay, tests, analyses | | | ≈ 0.3 |
  | **total** | | | **≈ 26.5** (campaign total ≈ 70.6 of 100; aggregate ≈ 269 of ≈ 298.4) |

* **Stop rules.** Seeds are fixed; no interim look; no extension. A failed run is resumed from its checkpoint, never
  replaced. The rule is never changed between cells. If the host becomes overloaded, blocks are launched later, not
  shortened.
* **Analysis** is run once per cell after its last run file exists, with the commands recorded in
  `scripts/r2_campaign/analyze_all.sh` (written with the results; it only calls the frozen scripts with the cell
  parameters of §3).
