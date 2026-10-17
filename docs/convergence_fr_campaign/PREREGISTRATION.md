# Preregistration — convergence acceleration of histogram ABF by Fisher–Rao

* Branch `research/convergence-fr-stopping`, worktree `/home/zheyuanlai/ABF-Fisher-Rao-convergence`, based on
  `main` at 5223d13e.
* Written 2026-10-17 03:11–03:25 UTC.
* Committed **before** any run of any block below. The commit hash is the freeze; later changes are dated
  amendments at the end of this file.

## 0. Rules that hold everywhere

* **Exactly two algorithms.**
  * **A** = histogram (P0) ABF.
  * **B** = the same ABF + marginal Fisher–Rao birth–death.
  * Continuous FR and early-stopping FR are two *schedules* of B. Each block compares one schedule of B against A
    and nothing else.
  * No sham, placebo, randomised-resampling, oracle or OPES arm exists anywhere in this campaign.
  * MC umbrella/WHAM appears only as the reference free energy.
* **Same everything within a cell.** Both arms share:
  * the physical model, timestep, estimator (180-bin histogram, trust 200 counts, 1 t.u. global ramp), reference
    and replica count K;
  * force evaluations, which are identical (asserted);
  * the seed. The three PCG64 streams (IC, noise, FR) are derived from the seed, so both arms are bitwise
    identical before the first FR opportunity. This is checked first.
* **Seeds.** Fresh blocks, disjoint from every earlier block (earlier blocks used 900000–981015):

  | block | seeds |
  |---|---|
  | E1 | 1100000–1100015 |
  | ERI calibration | 1110000–1110002 |
  | ERI production | 1111000–1111015 |
  | ITE common-rate calibration | 1120000–1120002 |
  | matrix 300 K / K256 | 1121000–015 |
  | matrix 150 K / K256 | 1122000–015 |
  | matrix 150 K / K1024 | 1123000–015 |
  | matrix 300 K / K1024 (only if needed) | 1124000–015 |

  * The exploratory ladder E1x uses the exploratory X1 seeds 980000–980003, never a confirmatory seed.
* **Seed handling.**
  * The seed is the unit of inference.
  * Per-seed relative changes (B/A − 1) or per-seed ratios are computed first, then the median over seeds, with a
    95 % percentile bootstrap CI (10 000 resamples, seed 2026).
  * Every seed is kept. A run that never reaches a tolerance is right-censored at the horizon, never dropped.
* **Analysis code is frozen with this commit.**
  * Descriptive atlas: `scripts/convergence_fr/atlas.py` and `atlas_lib.py`.
  * Every block: `scripts/convergence_fr/analyze_block.py`.
* **Compute.**
  * Runs are single-threaded numba (`run_jobs.py`, `threads: 1`). Single- and multi-threaded runs are bitwise
    identical; the pilot checked this against X1.
  * Core-h = wall × threads.
  * Aggregate ceiling ≤ 200 CPU core-h and ≤ 24 GPU-h. About 171.1 core-h were already used at the start of this
    campaign.
  * No GPU is available to this project: GPUs 0/1 belong to `yesom` and GPUs 2/3 to `hyt35` (GPU 3 was checked:
    occupied by another user since about 00:00 UTC). So **CPU only**.

## 1. What is already known (exploratory; motivates, does not confirm)

| quantity | value |
|---|---|
| X1 (ITE 300 K, K1024, rate 2.0, T 50): continuous FR ΔĪ_F | −14.9 % [−17.0, −10.1], 16/16 |
| X1: e_F(T) | +14.0 % [+0.8, +21.3] |
| X1: estimate from deposits after fr_start | +16.5 % worse |
| X1: atlas window establishment (trailing 1-t.u. mean ≥ 0.75 × the flat-bias window share), ABF | 4.3 t.u. |
| X1: same, FR | 2.4 t.u. (paired −1.8 [−2.1, −1.6]) |
| X1: e_F permanent-crossing time of the median paired curve | 29.1 t.u.; bootstrap 2.5 / 50 / 97.5 % = 19.9 / 29.1 / 46.8 t.u. |
| X1: e_F permanent crossing, per seed | 12/16 seeds cross |
| X1: restricted sustained speedup S at 0.20 / 0.16 / 0.12 / 0.10 / 0.08 kJ/mol | 1.29 / 1.07 / 0.96 / 0.90 / 1.00 |
| X1: geometric mean of those (the §3 primary metric applied to continuous FR) | 1.06 [0.95, 1.15] |

Source: `results/convergence_fr_campaign/atlas/atlas_summary.json`. The geometric-mean value came from the
analysis-script smoke test on X1, before this freeze.

## 2. Metric definitions (frozen; implemented in `atlas_lib.py`)

* **Error.** e_F(t) is the aligned full-circle RMS of F̂_t − F_ref, in kJ/mol: the L² norm on the circle divided by
  √(2π), after removing the additive constant.
* **Mean-force error.** e_F′(t) is the RMS over bins of Γ̂_t − Γ_ref.
* **Time grid.** Every quantity is evaluated at the saves: every 0.125 t.u. for ITE, 0.1 t.u. for ERI and 0.6 t.u.
  for the historical LTA.
* **Integrated error.**
  * I_F(t) = ∫₀ᵗ e_F (trapezoid); Ī_F(t) = I_F(t)/t.
  * Post-ramp Ī_F(t) = ∫_{t_FR}^{t} e_F / (t − t_FR), where t_FR is the FR start, which equals the end of the 1 t.u.
    bias ramp in every block.
* **Time to accuracy at tolerance ε** (save resolution):
  * *first*: the first save with e_F ≤ ε;
  * *sustained* (primary kind): the first save after which e_F ≤ ε at every later save up to T;
  * *windowed*: e_F ≤ ε on [t, t + 0.1 T].
  * A tolerance not met is censored at T.
  * The restricted per-seed speedup is S = min(τ_A, T)/min(τ_B, T). If both arms are censored, S = 1.
* **Tolerance ladders.**
  * **kT ladder:** ε ∈ {0.080, 0.064, 0.048, 0.040, 0.032} k_BT. At 300 K that is
    0.20 / 0.16 / 0.12 / 0.10 / 0.08 kJ/mol; at 600 K it is 0.40 / 0.32 / 0.24 / 0.20 / 0.16 kJ/mol.
    * This ladder was written into `atlas.py` before that script was first run. It extends the user's candidates
      0.12 / 0.10 / 0.08 kJ/mol upward, so that it spans the range where ABF converges within the horizon.
    * Feasibility for the ABF arm, from ABF-only data: ITE 300 K X1 ABF reaches every rung in ≥ 9/16 seeds and the
      top four in ≥ 15/16. The ERI ABF-only screen at K1024 has e_F 0.42 / 0.22 / 0.125 kJ/mol at t = 10 / 20 / 40.
  * **ABF-anchored ladder** (matrix cells, §5): ε_k = m_k × median_seeds e_F^ABF(T) of the same block, with
    m ∈ {3.0, 2.5, 2.0, 1.6, 1.3}.
    * This ladder is FR-blind. It is needed because at 150 K / T 50 the ABF arm reaches no rung of the kT ladder:
      X2's truncated ABF e_F(50) is 0.142 kJ/mol = 0.114 k_BT.
* **Primary time-to-accuracy statistic of every block.**
  * Per seed: the geometric mean over the five rungs of the restricted *sustained* speedup.
  * Then the median over seeds, with a 95 % bootstrap CI.
  * **SUPPORTED** if the CI is entirely > 1; **CONTRADICTED** if entirely < 1; otherwise **UNRESOLVED**.
* **Companion.** Post-ramp Ī_F(T), as the per-seed relative change B/A − 1.
* **Window establishment** (descriptive): the first save at which the trailing 1-t.u. mean window occupancy
  (|φ| < 0.5 rad) reaches 0.75 × the flat-bias share (1/2π).
* **Crossing.** The first save after which e_F^B > e_F^A at every later save. It is reported:
  * per seed;
  * for the median paired relative curve, with its bootstrap distribution.

  It is reported as a distribution, never as a precise time read off a plot.
* **Circulation.** c(t) = (1/2π)∫Γ_t dφ is the mean of the read-out Γ over the equal bins. c_app(t) is the same for
  the applied force, ramp × trust × Γ.
  * The exact mean force of the periodic linear CV integrates to zero; the MC reference closure is
    0.0002 ± 0.0008 kJ/mol.
  * The read-out PMF (`histogram_pmf`) already projects out c. The *applied* bias does not, so c ≠ 0 is a
    non-conservative tilt of 2π c per period.

## 3. Block E1 — ITE 300 K early-stopping FR (CONFIRMATORY, highest priority)

* **Cell.** Exactly X1's validated configuration (`configs/overnight/ite/jobs_prod.json`):
  * rigid Si-ITE, Dubbeldam-2004 UA ethane, 300 K, infinite dilution;
  * overdamped EM with dt 1e-4, which passed Gate 2 against exact MC;
  * K 1024, T 50 t.u. (500 000 steps), 180 bins, trust 200, 1 t.u. bias ramp;
  * 5 t.u. unbiased pre-equilibration from cage starts;
  * saves every 1250 steps;
  * reference: exact MC umbrella/WHAM F_MF at 180 bins (`results/overnight/ite/reference/ite_itq3`).
* **Arm B schedule.**
  * γ(t) = γ0 = 2.0 (X1's blind safety-calibrated rate) for 1 ≤ t < t_stop, else 0.
  * `fr_every` 10, KDE bandwidth 0.10, score clip 2, cap ⌊0.02K⌋ = 20.
  * After t_stop: no FR event and no FR-stream draw. Dynamics, cumulative C_j and M_j, the ramp and the trust
    rule all continue unchanged, with no reset of any kind (engine `fr_stop_steps`; `tests/test_fr_stop.py`).
* **Stop time t_stop = 10 t.u. (100 000 steps)**, by this rule applied to the X1 exploratory data only:
  * t_stop is the smallest multiple of 5 t.u. that is ≥ 2 × the median ABF window-establishment time
    (2 × 4.3 = 8.6);
  * and it must lie below the 2.5 % bootstrap quantile of the X1 e_F crossing time (19.9).
  * FR is therefore active for 9 t.u. and covers ABF's establishment phase twice over, while stopping well before
    continuous FR's error falls behind.
* **Hypotheses and frozen decision rules.**
  * **H3 (PRIMARY).** Early-stopping FR improves time to accuracy relative to ABF.
    * Statistic: the primary time-to-accuracy statistic of §2 on the kT ladder (0.20 / 0.16 / 0.12 / 0.10 /
      0.08 kJ/mol).
    * SUPPORTED / CONTRADICTED / UNRESOLVED as in §2.
  * **H1.** The early-time error reduction is preserved.
    * Statistic: post-ramp Ī_F over [1, 20] t.u. (= [t_FR, 2 t_stop]), per-seed relative change.
    * Supported if the CI < 0.
  * **H2.** The late-time penalty is removed.
    * Statistic: per-seed relative change of e_F(T).
    * **Removed** if the CI upper bound is < +10 % (non-inferiority margin, fixed now).
    * **Persists** if the CI lower bound is > 0.
    * Otherwise inconclusive.
    * Also reported: late-window Ī_F over [40, 50].
  * **H4.** After the stop, the late-only estimate is accurate.
    * Statistic: per-seed relative change of the error of the post-stop estimate F̂_post. F̂_post is computed
      read-only from M_j(T) − M_j(t_stop) and C_j(T) − C_j(t_stop) in *both* arms; t_stop is a save point.
    * Supported if the CI upper bound is < +10 %.
    * Contradicted if the CI lower bound is > 0.
    * Also reported: the pooled-over-seeds post-stop estimates against MC and against each other.
* **Companion.** Post-ramp Ī_F over [1, 50].
* **Secondary, reported regardless of direction:**
  * per-tolerance sustained, first and windowed speedups, with censoring counts and the fraction of seeds where
    FR is faster;
  * full Ī_F;
  * e_F at 0.25 T, 0.5 T and 0.75 T;
  * e_F′(T);
  * crossing distribution;
  * establishment;
  * circulation before and after the stop;
  * genealogy (min windowed ESS/K, max lineage share, replacements);
  * true translocations;
  * wall-clock per arm.
* **Multiplicity.** One primary (H3). H1, H2 and H4 are secondary and each carries its own rule. No claim is
  promoted from a secondary to the primary.
* **Sample size and stopping.** 16 seeds, fixed: no interim look and no extension. A run that fails is resumed from
  its checkpoint; a seed is never replaced.
* **Cost.** 32 runs × about 0.25 core-h ≈ 8 core-h. The pilot measured 1.4–1.5 µs per molecule-step at one thread;
  equilibration takes 69 s.
* **Exploratory ladder E1x (optional, last priority).**
  * t_stop ∈ {5, 10, 20} t.u. on X1 seeds 980000–980003, B arms only.
  * The X1 ABF arms of those seeds already exist and are bitwise the same runs, so each schedule is still a
    two-arm contrast against A.
  * Descriptive only; never pooled with E1.
* **ITE 150 K early-stopping extension.** Only if E1 gives a useful result or an interpretable mechanism and budget
  remains. It would need a separately preregistered amendment (its own t_stop by the same rule applied to 150 K
  ABF data, and fresh seeds).

## 4. Block ERI — ERI 600 K, ABF vs ABF + continuous FR (CONFIRMATORY, second priority)

* **Cell.** The validated ERI configuration (`configs/eri_campaign/jobs/screen_dilute.json`):
  * Gard & Tait rigid ERI surrogate, Schüring UA ethane, 600 K, infinite dilution;
  * EM dt 2e-4 (Gate 2 admissible);
  * T 40 t.u.; 180 bins; trust 200; 1 t.u. ramp; 5 t.u. pre-equilibration; saves every 0.1 t.u.;
  * exact MC reference (`results/eri_campaign/reference/gard_tait`).
* **FR start.** 1 t.u., the end of the ramp, as in ITE.
* **Not gated.** The old eligibility classifier is **not** a gate. Its ABF-sufficient label is reported only
  descriptively.
* **Safety-only rate calibration (blind).**
  * Top-down over the frozen ladder {2, 1, 0.5, 0.2, 0.1, 0.05}, 3 seeds (1110000–2) per rung per K, full horizon,
    using `scripts/overnight/fr_calibrate.py`.
  * Criteria: median windowed ancestor ESS/K ≥ 0.30 at every save after the FR start; median windowed max lineage
    share ≤ max(0.05, 3/K); replacements ≤ 2K per t.u.
  * The genealogy window is 2 t.u. (= T/20, the existing convention).
  * The selected rate is the first safe rung, which equals the largest safe rung of the ladder.
  * No free-energy, mean-force or occupancy error is read during calibration.
  * One rate per K. The rates may differ between K; they are reported.
* **Production.** K 1024 and K 256; seeds 1111000–1111015; both arms; continuous FR.
* **Primary (per K).** The §2 statistic on the kT ladder at 600 K (0.40 / 0.32 / 0.24 / 0.20 / 0.16 kJ/mol).
  * The two K are co-primary, with Bonferroni correction. A claim that FR changes ERI time to accuracy at a given
    K requires that K's **97.5 %** bootstrap CI (`primary.ci975`) to exclude 1. The 95 % CIs are reported
    alongside.
* **Companion and secondary.** As in §3, except the H1/H4 windows. The post-window estimate uses the deposits after
  the FR start.
* **Conditional early stopping.**
  * Trigger: at a K where continuous FR shows *both* a resolved early gain (post-ramp Ī_F over [1, 10] CI < 0) and a
    resolved late penalty (e_F(T) CI > 0).
  * Then an early-stopping block may follow as a dated amendment with fresh seeds. Its t_stop follows the §3 rule
    applied to the ERI production data: 2 × the median ABF establishment time, rounded up to 5 t.u., and below the
    crossing 2.5 % quantile.
  * Otherwise no ERI early-stopping block, unless ABF's own establishment is late (median > 0.2 T).
* **Cost.** Single-thread historical ERI K1024 runs took 472 s → about 0.14 core-h with equilibration, and K256
  about 0.035. Production ≈ 32 × 0.14 + 32 × 0.035 ≈ 5.6 core-h; calibration 0.5–1.5 core-h.

## 5. Block M — ITE temperature × K matrix (third priority; confirmatory per cell)

* **Question.** Is the 300 K vs 150 K difference in the FR effect still present when K, horizon, bins and FR rate
  are equal?
* **Common settings.** X1's settings at both temperatures:
  * T 50 t.u., dt 1e-4, **180 bins**, trust 200, 1 t.u. ramp, 5 t.u. pre-equilibration;
  * genealogy window 2.5 t.u. (T/20); saves every 1250 steps; continuous FR from 1 t.u.
* **150 K reference.** Exact MC at 180 bins (`results/overnight/ite150/reference/ite_itq3`).
* **Bin protocol.** 180 bins at both temperatures.
  * The frozen Gate-3c rule switches bins only if the finer grid improves e_F(T) by > 10 %. Under one common
    reference, 360 improved on 180 by 7.1 % at 150 K (audit F4), so the rule keeps 180.
  * The historical X2 (360 bins) is not altered. It is analysed as it ran, plus a read-out-only 180-bin
    coarsening; the atlas shows the two agree to < 1 %.
* **Common FR rate** (blind, safety only).
  * Top-down over {2, 1, 0.5} at 300 K / K256 and 150 K / K256 (T 50, seeds 1120000–2).
  * The existing X1 calibration covers 300 K / K1024 at the same T 50 and window, with 2.0 safe.
  * The common rate r_c is the largest rung safe in all three.
  * At 150 K / K1024, r_c is used and that cell's FR genealogy is checked by the same blind criteria before any
    error is read. A failure is reported as a safety failure.
* **Cells.**
  * 300 K / K256, 150 K / K256 and 150 K / K1024: new, 16 pairs each, at r_c.
  * 300 K / K1024: if r_c = 2.0, the exploratory X1 block has exactly these settings and serves as this cell
    (status: exploratory seeds, analysed with the same frozen code). Otherwise a new 300 K / K1024 block at r_c is
    run, with 16 pairs if the budget allows, else 8 pairs labelled exploratory.
* **Per-cell outcomes.**
  * Primary: the §2 statistic on the **ABF-anchored** ladder.
  * Companion: post-ramp Ī_F.
  * Also: e_F(T), e_F′(T), establishment, conditional relaxation (`conditional_lag_timeavg`), marginal TV,
    genealogy and translocations.
* **Temperature contrast at fixed K.** The difference between temperatures of the per-cell median relative change
  of post-ramp Ī_F, with a 95 % CI from independent bootstrap resampling of seeds within each cell. A difference
  is "present" if the CI excludes 0. No causal claim about barrier type follows from it.
* **Long-horizon 150 K.** X2 (K256, T 200, rate 0.5) is retained as the existing long-horizon analysis.
* **Cost.**
  * K256 cells: 32 × about 0.06 core-h each.
  * 150 K / K1024: 32 × about 0.25 ≈ 8 core-h.
  * Calibration: about 0.4 core-h.

## 6. Circulation diagnostics (read-only, fourth priority)

* Computed from saved accumulators with no change to either algorithm:
  * c(t) and c_app(t) per seed;
  * the late-half mean-force error;
  * FR event rates and genealogy;
  * FR − ABF profile differences;
  * the across-seed Spearman correlation between late-half c and e_F(T), and between late-half c and the late-window
    estimate error;
  * circulation before vs after the stop in E1.
* Scope: X1, X2, E1, ERI and the matrix cells.
* **Correction experiment.** The zero-mean projection Γ − c(t) in *both* arms is run only if the diagnostics show
  strong evidence: within-arm |ρ| ≥ 0.6 between circulation and late error in both ITE temperatures, **and** the
  FR−ABF circulation difference is resolved. It also needs budget. It would be a separately preregistered, new
  configuration on fresh seeds.

## 7. Efficiency (reported for every completed cell)

1. Simulation-time acceleration at fixed K (the §2 speedups).
2. Equal force evaluations at fixed K. FR adds no molecular force evaluation (asserted), so this equals (1).
3. Measured single-thread wall-clock per arm, from `wall_seconds_session`. The device, thread count and concurrent
   load are recorded in `EXPERIMENT_LOG.md`. The FR overhead is the KDE and selection every 10 steps.
4. Equal budget with K free only after a robust fixed-K gain. Not budgeted in this campaign; stated as not run.

## 8. Order, budget and stop rules

* **Order.**
  1. Atlas (done; descriptive).
  2. E1, launched together with the cheap blind calibrations (ERI rate 2.0 rung; ITE K256 rate 2.0 rung).
  3. ERI production.
  4. Matrix K256 cells.
  5. Matrix 150 K / K1024.
  6. E1x.
  7. Any conditional block (by amendment).
* **Budget.** Before each batch the projected cost is checked against the remaining ceiling, with ≥ 1 core-h
  reserved for analysis.
* **Stop rule.** A block that cannot be completed within the ceiling is **not started**. It is not shortened
  silently into something confirmatory; a smaller block is run only if it is labelled exploratory.

## Amendments

* **A0 (2026-10-17 03:15 UTC, clerical).** The header's "written 03:11–03:25" was an estimate. The freeze is commit
  4431ce30 at 03:13:20 UTC (git). E1, the ERI rate-2.0 rung and the ITE rate-2.0 rung were launched at 03:13:31,
  after it. Nothing in the design changed.
