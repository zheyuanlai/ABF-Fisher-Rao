# Preregistration — efficient and adaptive Fisher–Rao acceleration of histogram ABF

* Branch `research/adaptive-fr-equal-budget`, worktree `/home/zheyuanlai/ABF-Fisher-Rao-adaptive`, from `main`
  0f225325.
* Every section is frozen by the commit that adds it, before the block it governs is run. Later changes are dated
  amendments at the end.

## 0. Rules for the whole campaign

* **Exactly two algorithms in every comparison.**
  * **A** = histogram (P0) ABF.
  * **B** = the same ABF + marginal Fisher–Rao birth–death with a uniform target. Its schedule may be fixed-stop or
    online-stop.
  * One schedule per two-arm block. No sham, placebo, matched-turnover, randomised-resampling, oracle, OPES,
    metadynamics or other adaptive arm.
  * Exact MC / umbrella / WHAM appears only as the reference.
* **Inference.**
  * The seed is the unit.
  * Per-seed effects are computed first, then the median over seeds, with a 95 % percentile bootstrap CI (10 000
    resamples, seed 2026).
  * No seed is ever discarded; a tolerance not reached is right-censored at the horizon.
  * Cross-K comparisons are unpaired and use independent bootstrap resampling within each allocation.
* **Metrics.** As frozen in `docs/convergence_fr_campaign/PREREGISTRATION.md` §2 and implemented in
  `scripts/convergence_fr/atlas_lib.py`. The sustained time to accuracy is an *offline* evaluation metric, never an
  input to any FR controller.
* **Compute ceiling.** 100 CPU core-h for this campaign, on top of the ≈ 198.4 core-h used before; the aggregate
  ceiling is ≈ 298.4.
  * No GPU: GPUs 0–3 were occupied by other users when checked on 2026-10-17 at 06:19 UTC.
  * Runs are single-threaded; core-h = wall × threads.

## A. Experiment A — early-stopping FR at the efficient ABF allocation (CONFIRMATORY; frozen 2026-10-17 06:23 UTC)

### A.1 Cell (identical to the historical equal-budget ABF block EB)

* **Model.** Rigid Si-ITE (ITQ-3), Dubbeldam-2004 UA ethane, 300 K, infinite dilution. The validated model,
  reaction coordinate and exact-MC reference (`results/overnight/ite/reference/ite_itq3`) are reused unchanged.
* **Allocation.** K = 256, T = 200 t.u. (2 000 000 steps of overdamped EM at dt 1e-4).
* **Estimator.** 180 bins, trust 200 counts, 1 t.u. bias ramp.
* **Initialisation.** 5 t.u. unbiased pre-equilibration from cage starts.
* **Bookkeeping.** Saves every 5000 steps (0.5 t.u.). Genealogy window 25 000 steps (2.5 t.u.), the same as the ABF
  baseline and as the K256 calibration that admitted rate 1.0.

### A.2 Arms and pairing

* **ABF arm: REUSED.** The 16 historical EB runs, `results/convergence_fr_campaign/EB_ite300_abf_K256_T200/abf_N256_s{1140000…1140015}.npz`
  in the main checkout. They are read-only and never overwritten.
  * All 16 are verified complete: 2 000 000 steps, 256 × 2 000 001 production and 256 × 50 000 equilibration force
    evaluations, zero FR events.
  * Their configs equal `configs/convergence_fr/jobs_EB.json`.
  * They were produced by the engine that imports now: engine sha256 `a165f0a9…`, identical.
* **FR arm: NEW.** 16 runs on the same seeds, with config = EB config + `fr_rate = 1.0` + `fr_stop_steps = 100000`.
* **Pairing evidence (Gate F, pilot on seed 1140000, `scripts/adaptive_fr/pilot_pairing.py`).**
  * The FR configuration is bitwise identical to the historical ABF file at every save before the first FR
    opportunity.
  * An ABF re-run carrying `fr_rate = 1.0` is bitwise identical to the file through step 15 000. So the ABF arm's
    historical `fr_rate = 2.0` is inert. `tests/test_fr_stop.py::test_fr_rate_and_genealogy_fields_are_inert_for_abf`
    passes.
  * The production analysis re-asserts pairing for all 16 seeds before reading any error. It allows the arms' configs
    to differ ONLY in {`fr_rate`, `fr_stop_steps`}, and asserts zero FR opportunities and events in the ABF arm.

### A.3 FR schedule

* **Schedule.** γ(t) = 1.0 for 1 ≤ t < 10 t.u., 0 afterwards. After the stop there is no FR event and no FR-stream
  draw, and ABF continues on the same cumulative accumulators (Gate E; `tests/test_fr_stop.py`).
* **Rate 1.0, by the blind safety calibration** `results/convergence_fr_campaign/ITEcal_T300/frcal_K256.json`
  (K256, 300 K, the same 2.5 t.u. genealogy window). Over 49 t.u. of continuous FR:
  * min median windowed ESS/K 0.531 ≥ 0.30;
  * max median lineage share 0.027 ≤ 0.05;
  * 23 replacements per t.u. ≤ 512.

  Rate 2.0 failed the same gate. The stopped schedule's FR period [1, 10) has exactly the law of those runs' first 9
  t.u. Production FR runs are re-checked blind on the same criteria before any error is read. If they fail, the
  block is reported as a safety failure and the next lower rung (0.5) is preregistered before any accuracy is
  examined.
* **t_stop = 10 t.u., by the frozen stop rule of the previous campaign** (§3 there), applied to the K256 / 300 K
  exploratory-cell data (M300_256):
  * the smallest multiple of 5 t.u. ≥ 2 × the median ABF window-establishment time (2 × 4.125 = 8.25) is 10;
  * 10 is below the 2.5 % quantile of continuous FR's e_F crossing time in that cell (15.25).
* **Horizon independence.** The establishment time does not depend on the horizon: the dynamics up to t are
  identical whatever n_steps is. No EB error data enter this choice.

### A.4 Outcomes (frozen; analysis by `scripts/convergence_fr/analyze_block.py`, extended only by the cross-directory pairing option)

* **PRIMARY.** The per-seed geometric mean, over the kT ladder ε ∈ {0.080, 0.064, 0.048, 0.040, 0.032} k_BT, of the
  restricted SUSTAINED speedup τ_ABF/τ_FR (censoring at T = 200).
  * The exact values are ε = c × k_BT with k_BT = 0.0083144626 × 300 kJ/mol, i.e. 0.1996 / 0.1597 / 0.1197 /
    0.0998 / 0.0798 kJ/mol.
  * Take the median over seeds and its 95 % bootstrap CI.
  * **SUPPORTED** if the CI is > 1; **CONTRADICTED** if < 1; otherwise **UNRESOLVED**.
* **Secondary, with the frozen rules of the previous campaign:**
  * H1: post-ramp Ī_F over [1, 20] (= [t_FR, 2 t_stop]); supported if the CI < 0.
  * H2: e_F(T). "Penalty removed" if the CI upper bound < +10 %; "persists" if the lower bound > 0.
  * H4: the post-stop-only estimate (deposits after t_stop, both arms). Supported if the CI upper bound < +10 %;
    contradicted if the lower bound > 0.
* **Reported regardless of direction:**
  * every rung, as sustained, first-hit and windowed (0.1 T window) τ, with reach and censoring counts;
  * restricted mean τ per arm;
  * e_F(t), e_F′(t), post-ramp and full Ī_F, e_F′(T);
  * window establishment, marginal TV, ESS/K, max lineage, replacements, true translocations;
  * circulation, the e_F crossing, and wall-clock.

### A.5 Equal-budget analysis (prespecified; descriptive, unpaired across K)

* **Four configurations = two algorithms × two allocations**, all at B = K × T/dt = 5.12 × 10⁸ production force
  evaluations:
  * ABF at K1024 × 50 (the E1 ABF arm);
  * FR-stop-10 at K1024 × 50 (the E1 FR arm, rate 2.0);
  * ABF at K256 × 200 (EB);
  * FR-stop-10 at K256 × 200 (this block, rate 1.0).
* **Fixed-K comparisons:** paired, as in A.4 and E1.
* **Cross-K comparisons.**
  * Measure: the median per-seed sustained τ at each rung, in production force evaluations (B = K × steps), censored
    at the common budget.
  * Each allocation's summary: the per-seed geometric mean of τ_B over the five rungs.
  * The ratio of medians, with an independent bootstrap CI.
* **"Best tested allocation" of an algorithm** = the allocation with the smaller median per-seed geometric-mean τ_B.
  * Question B: is best FR / best ABF < 1, i.e. is the CI entirely < 1? This is answered separately from A.4's
    fixed-K question, and "B" is never concluded from "A".
* **All-in variant.** Equilibration force evaluations are added to every τ_B: K × 50 000 = 1.28 × 10⁷ (K256) and
  5.12 × 10⁷ (K1024). The budget then becomes 5.248 × 10⁸ / 5.632 × 10⁸. The comparison is made at each
  allocation's own all-in budget and at the common smaller one (5.248 × 10⁸).
* **Status.** Single-thread wall-clock per run is reported. Parallel wall-clock is Experiment E.

### A.6 Cost

* 16 FR runs × about 0.21 core-h. The historical EB runs took 742 s each at 1 thread, giving ≈ 3.4 core-h.
* Seeds are fixed (1140000–1140015). There is no interim look and no extension.

## B. Experiment B — the online stopping rule R1 (FROZEN 2026-10-17 07:10 UTC, by the commit that adds this section)

* **The rule.** The single primary online rule is R1. Its frozen spec is
  {"name": "R1", "check_every": 0.25, "W": 1.0, "R": 18, "theta": 0.25, "z_floor": 2.0, "persist": 4, "stop_factor": 2.0,
  "support_count": 200.0, "stability_tol": 0.05, "essw_min": 0.15, "maxlin_min": 0.1, "maxlin_k": 6.0}.
* **Implementation.** The code of `src/eri/fr_controller.py` at this commit. The mathematical definition and rationale
  are in `ONLINE_STOPPING_RULE.md`.
* **What R1 controls.** FR starts at each cell's validated fixed start; only the stop is online.
* **Development.** ITE only: the historical exploratory data plus nine monitor runs on seeds 1200000–1200500.
* **Not used for any choice:** LTA, ERI, Experiment A outcomes, or any confirmatory outcome.
* **No candidate comparison.** No alternative rule is evaluated against R1 in Experiment C.

## C. Experiment C — cross-system validation of the frozen R1 (CONFIRMATORY; frozen with section B)

### C.1 Arms and cells

* **Arms.** In every cell: ABF vs ABF + FR stopped online by R1. The arms are identical except for `fr_controller`
  (asserted). They share seeds and streams, so they are bitwise equal before FR start (asserted). 16 fresh pairs per
  cell.

| cell | system (validated config reused unchanged) | K | T (t.u.) | dt | FR rate / start | genealogy window | seeds | ladder | status |
|---|---|---|---|---|---|---|---|---|---|
| C1 | ITE 300 K (the E1 config, `jobs_E1.json`) | 1024 | 50 | 1e-4 | 2.0 / 1 t.u. | 2.5 t.u. | 1300000–015 | kT | in-sample cell (ITE) |
| C2 | ITE 150 K (the E2 config) | 256 | 100 | 1e-4 | 0.5 / 1 t.u. | 5 t.u. | 1310000–015 | ABF-anchored | in-sample cell (ITE) |
| C3 | LTA 150 K (LTA numba engine; historical histogram production config) | 1024 | 60 | 2e-4 | 0.2 / 4 t.u. | 4 t.u. | 1320000–015 | ABF-anchored | **held out** (positive control) |
| C4 | LTA 300 K (same) | 1024 | 60 | 2e-4 | 0.2 / 4 t.u. | 4 t.u. | 1330000–015 | ABF-anchored | **held out** |
| C5 | ERI 600 K (the ERI K1024 config) | 1024 | 40 | 2e-4 | 2.0 / 1 t.u. | 2 t.u. | 1340000–015 | kT | **held out** (negative control) |

* **LTA config.** fr_every 5, cap 20, KDE 0.10, score clip 2, bias ramp and production burn-in 4 t.u., bias read
  clip 60 and deposit clip 480. These are the engine defaults, identical to the validated 2026-10-03 histogram
  production. Saves are every 500 steps.
* **LTA reference.** The frozen exact-MC references `results/equal_budget_v2/references/lta_T{150,300}_reference.npz`,
  identical to the atlas's MC density profiles (checked). Γ uses the production accumulators once they hold counts,
  as in the equal-budget campaign.

### C.2 Analysis

* **Script.** `scripts/adaptive_fr/analyze_online.py`, frozen with this commit. On the E1 data it reproduces
  `analyze_block.py` exactly.
* **PRIMARY per cell.** S_geo, the per-seed geometric mean of the restricted sustained speedup over the cell's ladder.
  * SUPPORTED / CONTRADICTED / UNRESOLVED by the 95 % CI.
  * **Cross-system claims use the Bonferroni 99 % CI** (5 cells).
* **Secondary, per cell:**
  * post-ramp, full, early ([t_FR, 0.4 T]) and late-window Ī_F;
  * e_F(T), e_F′(T), e_F at T/4, T/2 and 3T/4;
  * the post-stop-only estimate (deposits after each FR run's own stop, both arms);
  * every rung as sustained, first-hit and windowed τ, with censoring and restricted means;
  * crossing, establishment, time-averaged marginal TV, genealogy, true translocations and wall-clock.
* **Online-rule outcomes, per cell:**
  * the distribution of stop times (median, IQR, range);
  * the reason (established / safety);
  * the number of never-stopped runs;
  * FR replacements per run;
  * FR events after the stop, which must be 0 (asserted).
* **Prespecified rule-level questions:**
  * **Q-ITE (in-sample).** Does R1 reproduce the fixed-schedule acceleration? C1 and C2 are SUPPORTED.
  * **Q-LTA150 (positive control).** Does R1 preserve the benefit? C3 is SUPPORTED. The descriptive context is the
    continuous-FR LTA 150 K effect: S_geo 2.42 in the equal-budget numba runs, cross-block and unpaired.
  * **Q-ERI (negative control).** Does R1 avoid the harm of continuous FR? C5 is NOT CONTRADICTED. The descriptive
    context is continuous FR at ERI K1024: S_geo 0.51.
  * **"R1 generalises across the tested systems"** iff, at the Bonferroni 99 % level:
    * C3 is SUPPORTED;
    * C4 and C5 are not CONTRADICTED;
    * C1 and C2 are not CONTRADICTED.
    Otherwise the failing cells are reported with their observables.

### C.3 Order, cost and stop rules

* **Order.** Wave 1 is C3, C4 and C5 (held out); wave 2 is C1 and C2.
* **Cost.** Measured single-thread costs: ITE K1024 × 50 is 0.227, ITE K256 × 100 is 0.108, ERI K1024 × 40 is 0.15
  and LTA K1024 × 60 is about 0.08 core-h per run. In total about 7.3 + 3.5 + 2.6 + 2.6 + 4.8 ≈ 21 core-h.
* **Stop rules.** Seeds are fixed, with no interim look and no extension. A failed run is resumed from its
  checkpoint, never replaced. The rule is never changed between cells.

## D. Experiment D — mechanism (EXPLORATORY; declared 07:22 UTC before the D2 runs)

* **D1, bias–variance decomposition.** Existing data only: `scripts/adaptive_fr/bias_variance.py`.
* **D2, clone age.** The FR arms of two existing continuous-FR blocks are re-run with the read-only diagnostic
  `diag_age_steps` = 1 t.u.:
  * ERI 600 K / K256, rate 2, seeds 1111000–015;
  * ITE 300 K / K256, rate 1, seeds 1121000–015.
  * The diagnostic splits each deposit by whether its walker was cloned within the last 1 t.u. Trajectories must be
    bitwise identical to the historical FR arms; this is checked, and the historical ABF arms stay valid partners.
* **D2 analysis**, pooled over seeds, with a seed-jackknife s.e., against the exact-MC reference:
  * the per-bin mean force and free energy from "young" deposits vs from "old" deposits;
  * the η and u_x² conditionals by age class;
  * the share of deposits that are young, per region.
  * Descriptive; no confirmatory claim.

## E. Experiment E — efficiency (frozen 07:32 UTC, before any E run)

### E.1 Block EA_R1: R1 at the efficient allocation (CONFIRMATORY for R1 at K256 × T200)

* **Cell.** Experiment A's cell exactly: ITE 300 K, K256, T200, rate 1.0, genealogy window 2.5 t.u.
* **FR arm (new).** The EB config + `fr_rate` 1.0 + `fr_controller` R1, on the EB seeds 1140000–015.
* **ABF arm.** The 16 historical EB runs, reused. Pairing is asserted; the allowed config differences are
  `fr_rate` and `fr_controller`.
* **Primary.** S_geo on the kT ladder (as in A).
* **Secondary.** As in C, via `analyze_online.py`.

### E.2 Block EB64: frontier cell K64 × T800 (B = 5.12 × 10⁸, CONFIRMATORY for its own two-arm comparison)

* **Settings.** ITE 300 K, K64, 8 000 000 steps, saves every 20 000 steps (the same 1.28 × 10⁶ force evaluations
  per save as the K256/K1024 blocks), genealogy window 2.5 t.u. Seeds 1350000–015; both arms; the FR arm is stopped
  by R1.
* **FR rate by blind, top-down calibration at K64.** Short T 50 continuous-FR runs, 3 seeds (1360000–2), rungs
  1.0 → 0.5 → 0.2, using the overnight criteria:
  * median windowed ESS/K ≥ 0.30;
  * lineage ≤ max(0.05, 3/K) = 0.05;
  * ≤ 2K replacements per t.u.

  The first safe rung is used.
* **K64 is below R1's development range** (K256–1024). The result is reported as R1's behaviour at K64, whatever it
  is.

### E.3 Equal-budget frontier (descriptive, unpaired across K)

* **Configurations.** ABF at K1024 (the C1 ABF arm), K256 (EB) and K64 (EB64), and FR-R1 at K1024 (C1), K256 (EA_R1)
  and K64 (EB64).
* **Statistic.** As in §A.5: the median per-seed geometric-mean sustained τ in production force evaluations, on the
  kT ladder, censored at 5.12 × 10⁸. Plus an all-in variant.
* **Best allocation per algorithm** by the smaller median. Best FR / best ABF, with an independent bootstrap CI.

### E.4 Parallel wall-clock (measured)

* **Configurations.** ABF and FR(R1) at K1024, K256 and K64.
* **Measurement.** The wall-clock time of a fixed 2-t.u. production segment (after the 5 t.u. equilibration), at
  numba thread counts 1, 4, 8, 16 and 32, on this host (2 × AMD EPYC, 256 logical CPUs, shared).
* **Recorded alongside.** Concurrent load, measured with `uptime`. Run while none of our other jobs runs.
* **Wall-clock time to accuracy** = measured seconds per step at P threads × the configuration's median sustained τ
  in steps. This is labelled as combining measured throughput with measured τ. It is not a full re-run at every P.

### E.5 Cost

| item | core-h |
|---|---|
| calibration | about 0.04 per rung |
| EA_R1 | 16 × 0.21 ≈ 3.4 |
| EB64 | 32 × 0.21 ≈ 6.8 |
| wall-clock | about 1–2 |
| total | about 12 |
* **E-A1 (07:45 UTC, integrity handling only; no statistic changed).** The engine key `diag_age_steps`
  (D2's read-only diagnostic, default 0) appears in configs written after it was added, but not in the historical EB
  files. `analyze_online.py` now allows that key to differ only when it is 0 or absent in every run, i.e. when the
  diagnostic is off; it asserts this. The bitwise pre-FR pairing assertion is unchanged.

* **E-A2 (07:58 UTC, frozen before any K16 run; adds one cell, changes no earlier statistic).** The brief's allocation
  ladder runs K ∈ {16, 64, 256, 1024}. E.3 found the best ABF and the best FR allocation both at the smallest K tried
  (K64), so the frontier is not bracketed from below. Block **EB16** adds K16 × T3200 at the same budget.
  * **Settings.** ITE 300 K, K16, 32 000 000 steps, saves every 80 000 steps (1.28 × 10⁶ force evaluations per save,
    as in every frontier cell), genealogy window 2.5 t.u. Seeds 1370000–015; both arms; the FR arm is stopped by R1
    (frozen spec, unchanged).
  * **FR rate by the same blind top-down calibration** at K16: T 50 continuous-FR runs, seeds 1380000–2, rungs
    1.0 → 0.5 → 0.2. Criteria: median windowed ESS/K ≥ 0.30; lineage ≤ max(0.05, 3/K) = 0.1875; ≤ 2K = 32
    replacements per t.u. The three rungs run concurrently (they are blind and cheap). The selection is the largest
    safe rung, which is the same as the first safe rung top-down. If no rung is safe, the cell runs at 0.2 and that is
    reported, because the brief forbids optimising the rate on the outcome and a fourth rung would be new tuning.
  * **K16 is further below R1's development range** (K256–1024). R1's lineage backstop is max(0.10, 6/K) = 0.375 at
    K16. The result is reported as R1's behaviour at K16.
  * **Analysis.** The EB16 two-arm comparison is analysed exactly as EB64 (`analyze_online.py`, kT ladder, primary
    S_geo, 95 % CI; it is a fourth E cell, so no Bonferroni). E.3 is re-run with the two K16 configurations added
    (eight in total). The best allocation per algorithm is still chosen by the smaller median.
  * **Wall-clock.** E.4's benchmark is re-run with K16 added, once no campaign job is running.
  * **Cost.** About 0.25 core-h per run × 32, plus calibration ≈ 9 core-h. Campaign total before this ≈ 36 of 100.
* **Errata (08:34 UTC, timing records only; nothing in any rule or statistic changes).**
  1. **E-A2's header time is wrong.** "07:58 UTC" was an estimate written into the text. The amendment was committed
     at **07:56:23** (d0929f7a), before the K16 calibration launch at 07:56:29 and the EB16 launch at 07:57:02 (both
     in `logs/launches.tsv`). The header was briefly edited in place in f98d3ba8; that edit is reverted here, and the
     record is this erratum.
  2. **E-A2 was prompted by data.** It was triggered by E.3's result, which put both algorithms' best allocation at
     K64, the smallest K tried. The amendment text says so.
  3. **E-A1 came after the EA_R1 data.** It was written at about 07:45, after EA_R1 finished (07:44:45). The frozen
     analysis refused the EA_R1 files, which was the trigger. It was first committed at 07:56:23 with E-A2, after EB64
     (07:47:47) and the six-configuration frontier (07:48) had been computed. It is integrity handling only and
     changes no statistic.
  4. **§D's commit came after the D2 runs.** §D was written at about 07:22, before the D2 runs, but first committed at
     07:30:24 (11477fef), after they had finished at 07:25:35. D is exploratory.

* **E-A3 (2026-10-17, about 11:35 UTC; frozen by the commit that adds it, before any K < 16 run; adds three frontier
  cells and two blind calibrations, changes no earlier statistic).** Prompted by data and by a reviewer. The
  eight-configuration frontier (E.3 + E-A2) put the best allocation of both algorithms at K16, the smallest K tried,
  and K16 vs K64 is itself unresolved (ABF 0.75 [0.44, 1.13]). The reviewer (2026-10-17) made two points: the best K
  is selected on the same data that scores it while the bootstrap holds the chosen configurations fixed (addressed
  without new runs by the selection-aware nested bootstrap, EFFICIENCY.md §7, added to `equal_budget_frontier.py` under
  new keys only), and the frontier is not bracketed below K16. This amendment brackets it.
  * **Cells.** ITE 300 K, 180 bins, B = 5.12 × 10⁸ production force evaluations per run; the EB config with only `N`,
    `n_steps`, `save_every` (and `fr_rate` for EB8) changed; genealogy window 2.5 t.u.; 16 seeds per cell from fresh
    blocks (`make_jobs.py`).

    | block | K × T (t.u.) | arms | seeds | steps per run |
    |---|---|---|---|---|
    | EB8 (`EB8_ite300_K8_T6400_R1`) | 8 × 6400 | ABF, ABF + FR stopped by R1 (frozen spec) | 1400000–015 | 64 000 000 |
    | EB4 (`EB4_ite300_K4_T12800_abf`) | 4 × 12800 | ABF only (see the K4 FR criterion) | 1410000–015 | 128 000 000 |
    | EB1 (`EB1_ite300_K1_T51200_abf`) | 1 × 51200 | ABF only (one walker cannot be reallocated) | 1420000–015 | 512 000 000 |
  * **Save grid in physical time.** Every new cell saves every 10 000 steps = 1 t.u. (the existing frontier cells save
    every 1.28 × 10⁶ force evaluations, which is 8 t.u. at K16). Time to accuracy is read at saves, so its resolution
    in force evaluations is 8 × 10⁴ at K8, 4 × 10⁴ at K4 and 10⁴ at K1, against 1.28 × 10⁶ in every existing cell;
    establishment is resolved at 1 t.u. **No existing cell is re-run and none of its statistics changes.** The
    production budget is 5.12 × 10⁸ at every K (8 × 64 × 10⁶ = 4 × 128 × 10⁶ = 1 × 512 × 10⁶), so the production
    rows of the existing configurations in the extended table are identical to E.3's. The all-in common budget of the
    extended set is min over all configurations = 5.125 × 10⁸ (K1: 51 200 000 + 50 000 equilibration steps) instead
    of 5.128 × 10⁸, so the all-in variant of the extended table truncates 0.06 % earlier and the all-in rows of the
    existing cells may differ in the last decimal; this is stated with the table. `E_frontier.json`, the
    eight-configuration analysis, keeps its budget and is not rewritten by this amendment (the extended analysis is
    `E_frontier_ext.json`).
  * **FR rate at K8 by the same blind top-down safety calibration** as E.2 / E-A2 (`scripts/overnight/fr_calibrate.py`):
    T 50 continuous-FR runs, seeds 1430000–2, rungs 1.0 → 0.5 → 0.2 run concurrently. Criteria: median windowed
    ESS/K ≥ 0.30; windowed max lineage share ≤ max(0.05, 3/K) = **0.375 at K8** (a family of 4 of 8 walkers fails, 3
    of 8 passes); ≤ 2K = 16 replacements per t.u. The largest safe rung is used; if no rung is safe, the cell runs at
    0.2 and that is reported (as in E-A2). Whether the lineage criterion constrained anything is reported: which rungs
    it rejected, and where the median lineage share sat relative to 0.375.
  * **R1 at K8.** R1's own safety backstops are max(0.10, 6/K) = 0.75 on the windowed lineage share (fires only when
    one family holds 7 or 8 of 8 walkers) and ESS/K < 0.15 (fires only at complete collapse: 1/8 = 0.125; a 7 + 1
    split gives 0.16). Both can fire, but only at near-collapse. K8 is far below R1's development range (K256–1024);
    the result is reported as R1's behaviour at K8, whatever it is. The FR machinery is otherwise unchanged (cap
    max(1, ⌊0.02 K⌋) = 1 per opportunity, KDE 0.10 rad, score clip 2, `fr_every` 10).
  * **K4 FR arm: criterion stated in advance.** A K4 FR arm is run only if ALL of the following hold. (i) R1's
    genealogical backstops can fire at K4: max(0.10, 6/K) < 1 and 1/K < essw_min = 0.15. (ii) The blind calibration
    at K4 (seeds 1440000–2, rungs 1.0 / 0.5 / 0.2, the standard criteria with 3/K = 0.75, **and additionally** windowed
    max lineage share ≤ 0.5, i.e. no family ever holds more than half the population) passes at some rung. The safety
    limits are not weakened for small K. **Condition (i) fails at K4 by arithmetic** (6/4 = 1.5 ≥ 1; 1/4 = 0.25 ≥
    0.15): R1 could never stop FR for a genealogical reason there, so an R1-stopped K4 FR arm would run without any
    genealogical protection. **The K4 FR arm is therefore not run**, and this is the reported finding. The K4
    calibration (9 runs of T 50, ≈ 0.01 core-h) is run anyway as a descriptive diagnostic of how degenerate
    birth–death is with four walkers (replacements per t.u., lineage, ESS); it has no production consequence.
  * **Analysis (frozen as before).**
    1. The EB8 two-arm comparison exactly as EB64 / EB16 (`analyze_online.py`, kT ladder, primary S_geo with 95 % CI,
       secondaries, integrity checks). It is a fifth E cell, so no Bonferroni.
    2. E.3 re-run on the extended set of 12 configurations (ABF at K1, 4, 8, 16, 64, 256, 1024; FR-R1 at K8, 16, 64,
       256, 1024) by the same script → `E_frontier_ext.json`. Per-rung medians, best allocation per algorithm by the
       smaller median, same-K and adjacent-K ratios of medians with independent bootstrap CIs.
    3. **Verdicts.**
       * *Does ABF's best allocation lie below K16?* SUPPORTED if the best ABF allocation (smaller median) is at K < 16
         and its ratio to ABF K16 has a 95 % CI with upper bound < 1; CONTRADICTED if the best ABF allocation is at K
         ≥ 16 and ABF K8 / ABF K16 has a CI lower bound > 1; otherwise UNRESOLVED, with the point estimates. The same
         rule for FR with K8 vs K16.
       * *Is the frontier bracketed from below?* For an algorithm, yes if some tested K below its best allocation has a
         cost ratio to the best whose CI lower bound is > 1; otherwise it remains unbracketed and that is said.
       * *Best FR / best ABF.* The verdict label (BEATS / BEATEN / UNRESOLVED) is taken from the **nested
         selection-aware CI** (EFFICIENCY.md §7); the fixed-selection CI of E.3 is reported alongside for continuity,
         and a disagreement between the two, if any, is stated.
    4. **Establishment on the 1 t.u. grid.** The frozen measure of every cell (`atlas_lib.establishment_time`: first
       save at which the trailing 1 t.u. mean of the instantaneous window fraction reaches 0.75 of the uniform window
       share), per seed, median per configuration, with the save resolution stated. Because the instantaneous
       fraction takes only K + 1 values and is weak at K ≤ 8, a supplementary descriptive measure is added: the same
       first-passage rule applied to the window share of the deposits collected during the trailing save interval
       (difference of the cumulative window-bin counts C over the interval divided by the difference of all counts;
       10⁴ K samples per t.u.). Both are computed for every frontier configuration by
       `scripts/adaptive_fr/establishment_frontier.py` → `E_establishment.json`. Both are descriptive and feed no
       verdict.
  * **Cost.** Measured by a 1-minute timing pilot (11:21:44–11:22:16 UTC; three concurrent scratch ABF runs of
    2 × 10⁶ steps with 1 t.u. saves, deleted, nothing read from them but the wall time): 12.8 µs per step at K8, 7.4
    at K4, 2.7 at K1 (single thread, 1-minute load 10 on 256 logical CPUs). Expected: EB8 32 × 64 × 10⁶ × 12.8 µs ≈
    7.3 core-h; EB4 16 × 128 × 10⁶ × 7.4 µs ≈ 4.2; EB1 16 × 512 × 10⁶ × 2.7 µs ≈ 6.1; calibrations ≈ 0.03; total ≈
    17.7 core-h (the brief's estimate was about 30). Campaign total before this amendment ≈ 44.1 of 100 authorised;
    after it ≈ 62. Disk: about 6 GB of run files because of the 1 t.u. saves, in the data root only.
  * **Order.** Commit this amendment → launch EB4, EB1 and the K8 / K4 calibration ladders concurrently (64 + 18
    single-threaded processes) → select the K8 rate blind → launch EB8. No production file is read before its block
    is complete.
* **Erratum to E-A3 (11:33 UTC, timing record only; no rule or statistic changes).** E-A3's header time "about 11:35
  UTC" was an estimate written into the text. The amendment was committed at **11:28:15** (4e848bb5), before the
  launches of the K8 / K4 calibration ladders, EB4 and EB1 (11:28:15) and of EB8 (11:29:40), all in
  `logs/launches.tsv`. The header is left as written; this erratum is the record.
* **Erratum 2 to E-A3 (12:05 UTC, arithmetic in the text only; the rule is unchanged and was applied as written).**
  E-A3 states the all-in common budget of the extended set as "5.125 × 10⁸ (K1: 51 200 000 + 50 000 equilibration
  steps)". K1's run is 512 000 000 steps, so the figure is 512 050 000 = **5.1205 × 10⁸**, 0.01 % above the production
  budget and 0.15 % below the eight-configuration all-in budget 5.128 × 10⁸. The rule, "min over all configurations",
  is what the script computes; the all-in medians of the eight earlier configurations are unchanged to four decimals.
  The matched-save-grid re-read of the E-A3 cells (EFFICIENCY.md §8.4, item 5) is a post-hoc sensitivity analysis that
  was not preregistered; it is labelled as such and changes no verdict.
