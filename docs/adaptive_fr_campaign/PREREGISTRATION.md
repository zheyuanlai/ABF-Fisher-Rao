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
