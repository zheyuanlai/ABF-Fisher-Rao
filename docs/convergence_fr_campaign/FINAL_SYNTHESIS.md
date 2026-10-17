# FINAL SYNTHESIS — convergence acceleration of histogram ABF by Fisher–Rao

*Campaign of 2026-10-17, run on branch `research/convergence-fr-stopping`, now on `main` (a96b8532). Exactly two algorithms throughout: histogram (P0)
ABF, and the same ABF + marginal Fisher–Rao birth–death, run continuously or stopped early. No sham, oracle or other
control arm was run or presented. MC umbrella/WHAM appears only as the reference. Every confirmatory block was
preregistered and committed before it ran (`PREREGISTRATION.md`, A0–A3).*

## 1. All paired blocks (frozen analysis; median of per-seed effects with a 95 % bootstrap CI; FR better if < 0, or S > 1)

| block | system | K | rate | FR schedule | T (t.u.) | status | **primary S_geo** (FR faster) | post-ramp Ī_F | e_F(T) |
|---|---|---|---|---|---|---|---|---|---|
| E1 | ITE 300 K | 1024 | 2.0 | stop 10 | 50 | confirmatory | **1.59 [1.31, 1.85]** 16/16 | -38 % [-42, -29] | -32 % [-43, -24] |
| E2 | ITE 150 K | 256 | 0.5 | stop 15 | 100 | confirmatory | **1.25 [1.11, 1.58]** 14/16 | -18 % [-28, -13] | -19 % [-32, -6] |
| M300_256 | ITE 300 K | 256 | 1.0 | continuous | 50 | confirmatory | **1.03 [0.92, 1.21]** 9/16 | -8 % [-15, +1] | +12 % [-0, +43] |
| M150_256 | ITE 150 K | 256 | 1.0 | continuous | 50 | confirmatory | **1.63 [1.34, 2.73]** 14/16 | -22 % [-29, -17] | +29 % [+7, +51] |
| ERI256 | ERI 600 K | 256 | 2.0 | continuous | 40 | confirmatory | **0.52 [0.46, 0.63]** 0/16 | +32 % [+22, +46] | +113 % [+79, +236] |
| ERI1024 | ERI 600 K | 1024 | 2.0 | continuous | 40 | confirmatory | **0.51 [0.49, 0.58]** 0/16 | +33 % [+28, +35] | +119 % [+103, +141] |
| X1 | ITE 300 K | 1024 | 2.0 | continuous | 50 | exploratory (historical) | **1.06 [0.95, 1.15]** 10/16 | -22 % [-24, -15] | +14 % [+1, +21] |
| X2 | ITE 150 K | 256 | 0.5 | continuous | 200 | exploratory (historical) | **0.82 [0.60, 0.95]** 4/16 | -1 % [-12, +15] | +57 % [+29, +95] |

* **S_geo** is the per-seed geometric mean, over a frozen five-rung tolerance ladder, of the restricted sustained
  time-to-accuracy speedup τ_ABF/τ_FR.
* **Ladders.** E1, ERI and X1 use k_BT rungs; E2, the matrix cells and X2 use ABF-anchored rungs (§2 of the prereg).
* **Integrity.** Every block passed:
  * bitwise pairing before the first FR opportunity;
  * equal force evaluations;
  * identical configurations apart from the schedule;
  * for the stopped schedules, zero FR events after the stop.
* **Descriptive atlas** (`CONVERGENCE_ATLAS.md`): LTA 150/300 K and the historical X1/X2.

## 2. Answers

**Q1 — Does early-stopping FR improve convergence or time to accuracy in ITE 300 K?**
* **Yes (E1, confirmatory, 16 fresh paired seeds).** FR from 1 to 10 t.u. at rate 2.0, then plain ABF.
* Preregistered S_geo **1.59 [1.31, 1.85]**, with S_geo > 1 in **every seed** (16/16). Per rung, S is 1.45–1.73.
  * FR is faster in 16/16 seeds at the four rungs from 0.20 to 0.10 kJ/mol.
  * At 0.08 kJ/mol FR is faster in 15/16 seeds, with one tie in which both arms are censored. ABF reaches that rung
    in only 9/16 seeds, FR in 15/16.
* Post-ramp integrated error is −38 % [−42, −29]. No seed's error curve falls permanently behind ABF's (0/16 crossings).
* The stop time came from a rule applied to exploratory data only.
* **Replicated at 150 K (E2):** stop at 15 t.u., rate 0.5, K 256, T 100. S_geo **1.25 [1.11, 1.58]**, 14/16.
* **Exploratory ladder (E1x, 4 X1 seeds, paired with their X1 ABF arms; stop 10 was not run on these seeds):**
  * stop at 5 t.u.: S_geo 1.39–1.48, final error −23 to −31 %;
  * stop at 20 t.u.: S_geo 0.99–1.19, final error −27 to +4 %;
  * continuous: S_geo 0.97–1.07, final error 0 to +15 %.
* **Ordering in the ladder.**
  * S_geo orders stop 5 > stop 20 > continuous on all 4 seeds.
  * For the final error, stop 20 beats continuous on all 4 seeds, but stop 5 beats stop 20 on only 3. On seed 980002
    stop 20 gives −27 % against −23 % for stop 5.
* **No ITE early-stopping schedule was optimised.**

**Q2 — Does the stopping strategy mitigate the late-time error penalty?**
* **Yes, and it reverses it.**
* Final error with early stopping:
  * E1: **−32 % [−43, −24]**, 16/16;
  * E2: **−19 % [−32, −6]**, 13/16.
* Continuous FR in the same model (other blocks, other seeds):
  * 300 K: +14 % (X1, K1024, rate 2.0), and +12 % (M300, K256, rate 1.0; not resolved, CI lower bound −0.1 %);
  * 150 K: +57 % (X2, T 200) and +29 % (M150, T 50).
* **After the stop the FR arm is exactly ABF** (bitwise by test).
  * Pooled over seeds, its post-stop deposits are as close to MC as ABF's: 0.011 vs 0.0095 kJ/mol in E1, and 0.018 vs
    0.017 in E2.
* **H4 by the frozen rules:**
  * **CONTRADICTED at 150 K (E2).** The per-seed error of the post-stop-only estimate is +15 % [+0.4, +58] larger than
    ABF's, and the CI's lower bound is above 0. The pooled post-stop estimates do not differ, so the excess is per-seed
    (scatter-like), not a shared bias. Its cause is not established.
  * **Inconclusive at 300 K (E1):** −0 % [−21, +25].

**Q3 — After controlling K, horizon and safe FR rates, is the 300 K vs 150 K difference still present?**
* **A difference is present, but it is the opposite of the old one.**
* Controlled cells: K 256, T 50, 180 bins, and the largest common blind-safe rate, 1.0 (2.0 is unsafe at K256).
* FR's finite-time benefit is *larger* at 150 K:
  * post-ramp Ī_F −22 % vs −8 %; difference +13.9 points [+4.4, +25.2];
  * time to accuracy S_geo 1.63 (supported) vs 1.03 (unresolved).
* Both temperatures carry a late penalty: +29 % (resolved) and +12 % (borderline).
* **The old "300 K helps, 150 K does not" was a horizon effect.** X2's first 50 t.u. alone give −12.7 %
  [−14.3, −8.3], against X1's −14.9 %. The remaining old confounds were K, the bin count (immaterial) and the rate.
* **K and rate are not separated at 300 K.** The K1024 row of the matrix was not run (budget).

**Q4 — Does FR accelerate ABF in ERI 600 K, the strongly entropic-barrier example?**
* **No. It slows it** (confirmatory, 16 + 16 pairs, blind safety rate 2.0).
* S_geo **0.52** (K256) and **0.51** (K1024). ABF is faster in 16/16 seeds at both K, FR in 0/16. The result is
  Bonferroni-significant.
* Final error +113 % and +119 %.
* FR is ahead only at 1.5–5 t.u. ABF establishes the window in 2.1 t.u. anyway.
* Continuous FR's estimator converges to a biased profile: pooled late-half error 0.13 vs 0.03 kJ/mol. It distorts
  the cage conditional of the c-axis coordinate η by 2.5–5×.
* **Early stopping could not be applied.** The frozen stop-time rule had no admissible value.
  * Twice ABF's establishment time is 4.2 t.u. The rule rounds it up to 5 t.u.
  * The stop must lie below the crossing time's 2.5 % quantile, which is 4.3 t.u. at K256 and 4.9 at K1024.
  * 5 t.u. fails that condition.

**Q5 — Do the experiments support a relationship between FR acceleration and entropic-barrier fraction,
population-establishment delay, or conditional mixing?**
* **Entropic fraction: no.** ERI (84 % entropic) is the clearest *negative*.
  * ITE 150 K (33 % entropic) gains more than ITE 300 K (48 %).
  * LTA's gain is largest at its coldest temperature.
* **Establishment delay: consistent, but not established causally.** Across the cells, FR's finite-time gain grows
  with ABF's own window-establishment time:

  | cell | ABF establishment (t.u.) | FR gain |
  |---|---|---|
  | ERI | 2.1 | harmful (+32 %) |
  | ITE 300 K / K256 | 4.1 | −8 % |
  | ITE 300 K / K1024 | 4.3 | −22 % |
  | ITE 150 K / K256 | 6.9–7.5 | −18 % (X2, first 50 t.u.) to −22 % (M150) |
  | LTA 300 K | 9.3 | −23 % |
  | LTA 150 K | 16.2 | −46 % (post-ramp) |

  * This matches the earlier repository's "establishment-starvation" predictor. It is a cross-system association
    over different models, K and rates.
* **Conditional mixing.**
  * FR's late damage is largest where a slow hidden coordinate is distorted (ERI η).
  * In ITE the hidden-coordinate conditionals change only slightly in absolute terms. The η RMS from MC is 0.0005–0.0007
    under continuous FR (X1, X2) vs 0.016–0.017 in ERI, although it is still 1.4–1.9× ABF's. The FR − ABF late mean-force difference sits
    in the flank and cage (in X1, cage 0.061 vs flank 0.049 kJ/mol/rad).
  * Conditional distortion is therefore one mechanism, not the only one.

**Q6 — Is nonzero circulation of the histogram bias a plausible explanation for the late errors?**
* **Not as the main explanation.**
* The applied bias does circulate in every cell and both arms (0.03–0.3 k_BT per period). Within the FR arm it
  correlates with final error e_F(T): ρ 0.63–0.79 at 150 K and 0.18–0.64 at 300 K.
* Against it as the main cause:
  1. FR increases circulation resolvably only in X2;
  2. early-stopping FR (E1) circulates about as much as continuous FR (X1): 0.025 vs 0.030 kJ/mol/rad. That is a
     cross-block comparison on different seeds, and the difference was not tested. Yet E1 has no late penalty;
  3. the largest late penalty (ERI) shows no resolved FR − ABF circulation difference.
* **The correction experiment was not run.**
  * The preregistered trigger (§6) asks for "within-arm |ρ| ≥ 0.6 between circulation and late error in both ITE
    temperatures, and a resolved FR − ABF difference". It reads two ways:
    * **"late error" = the late-window estimate error** (the §6 diagnostic): every ITE |ρ| ≤ 0.47, so the trigger is
      **not met**;
    * **"late error" = e_F(T):** the FR arm reaches ρ ≥ 0.6 at both temperatures (M300 0.64; X2 0.63, E2 0.79, M150
      0.75), but ABF only in X2. The difference clause holds only in X2.
  * Under the first reading the trigger is not met; under the most permissive one it is met only partly.
  * Either way, the trigger's budget clause was not satisfied: about 1.6 core-h remained, against the ≥ 7 core-h a
    two-arm corrected block needs.

**Q7 — What acceleration factors are actually supported?**

| cost model | supported factor (ITE 300 K, E1) | notes |
|---|---|---|
| simulation time at fixed K = 1024 | **1.45–1.73** per tolerance; S_geo 1.59 [1.31, 1.85] | ITE 150 K (E2): 1.24–1.33; continuous FR: 0.90–1.29 (X1) |
| equal force evaluations at fixed K | identical to the row above | FR adds no molecular force evaluation (asserted) |
| measured wall-clock at fixed K (1 thread per run) | identical to the row above within **0.2 %** | FR overhead with early stopping: +0.2 %; continuous FR: −1.0 to +1.0 % in the 1-thread blocks (+2 to +3 % in X1/X2, which ran at 8 / 2 threads). The overhead is the KDE and selection every 10 steps |
| **equal force evaluations, K free** (EB: ABF at K 256 for 200 t.u. vs E1 at K 1024 for 50 t.u.) | **< 1: ABF at smaller K is about 2× more efficient** | table below |

Equal budget (5.12 × 10⁸ molecular force evaluations). The table gives the median sustained time to accuracy in
units of 10⁸ force evaluations, with the number of seeds reaching each rung by the budget.

| ε (kJ/mol) | ABF, K 256 × 200 t.u. | ABF, K 1024 × 50 t.u. | ABF + FR stop 10, K 1024 × 50 t.u. | ABF K1024 / FR-stop K1024 | ABF K256 / FR-stop K1024 |
|---|---|---|---|---|---|
| 0.20 | 0.57 (16/16) | 1.80 (16/16) | 1.09 (16/16) | 1.66 [1.31, 2.10] | 0.52 [0.38, 0.68] |
| 0.16 | 0.68 (16/16) | 2.32 (16/16) | 1.48 (16/16) | 1.57 [1.32, 1.97] | 0.46 [0.40, 0.63] |
| 0.12 | 0.95 (16/16) | 3.10 (16/16) | 1.93 (16/16) | 1.61 [1.34, 1.94] | 0.49 [0.41, 0.63] |
| 0.10 | 1.29 (16/16) | 3.91 (16/16) | 2.40 (16/16) | 1.63 [1.35, 1.98] | 0.54 [0.41, 0.66] |
| 0.08 | 1.66 (16/16) | 4.96 (9/16) | 3.06 (15/16) | 1.62 [1.26, 2.07] | 0.54 [0.42, 0.72] |

* **e_F at the full budget:** ABF K256 0.034; ABF K1024 0.076; FR-stop K1024
  0.052 kJ/mol. The ratio ABF K256 / FR-stop is 0.66 [0.46,
  1.003]; the CI just includes 1.
* **The allocations are unpaired** (independent bootstrap). The equilibration force evaluations differ (1.3 × 10⁷
  vs 5.1 × 10⁷) and are excluded.
* **A plausible reading, not tested by EB.** If ABF's error is the diluting memory of an early transient that lasts
  a fixed *physical* time, fewer replicas run for longer dilute it faster per force evaluation. The decay slopes
  (`MECHANISM.md` §1) are consistent with this.
* **Consequence.** Early-stopping FR at K 1024 is a genuine 1.6× accelerator *of that allocation*. It is **not**
  faster than the better ABF allocation at the same budget, which reaches each tolerance in about half the force
  evaluations.
* **Untested.** Early-stopping FR at the better allocation (K 256 × 200 t.u.) was not run (budget). Its fixed-K
  speedup there is unknown; continuous FR at K256 / T 50 was unresolved (M300_256).
* **Parallel hardware.** On many cores the K-wide replica loop parallelises; X1's K1024 runs took 262 s at 8 threads
  vs 747 s at 1. That favours large K in wall-clock, but it was not measured for K256.

## 3. Supported conclusions, hypotheses, open problems

**Observations (supported by the data here).**
1. **Continuous marginal FR accelerates the early convergence** in every establishment-limited molecular cell (ITE,
   LTA). The gain is transient in ITE and LTA 300 K. In LTA 150 K it persists to the end of the run (e_F(T) −54 %,
   16/16 seeds, no crossing).
   * FR loses in ITE at late times, and in ERI from about 5 t.u. on.
   * Its post-start deposits are biased: their per-seed and pooled errors exceed ABF's.
   * Its error decays as about t^−½ (−0.45 to −0.69), against ABF's 1/t (−0.78 to −1.00).
2. **Early-stopping FR** (FR only while ABF establishes the window, then plain ABF) keeps the early gain, removes
   the late penalty, and ends with a *lower* final error than ABF at fixed K. This is confirmatory in two ITE cells
   (300 K, 150 K).
3. **In the ITE cells, the data are consistent with a memory-limited cumulative histogram ABF.** The evidence:
   * the late error decays as about 1/t;
   * the pooled whole-run bias is about equal to the per-seed error;
   * the post-stop deposits are nearly unbiased in both arms (E1, E2).

   On this reading, early FR lowers the memory error and continuous FR adds a new, slowly decaying bias. That
   attribution is an inference, not a direct test.
4. **None of this is an equal-budget win over a better ABF allocation** (EB). The repository's earlier equal-budget
   finding stands for the early-stopping variant too.

**Hypotheses (consistent with the data, not established).**
* FR's late bias comes from an FR + dynamics steady state that replaces equilibrated residents of the dense (cage)
  region by recent arrivals. Only the ERI η distortion is directly observed; the replacement mechanism is
  inferred.
* FR's benefit scales with ABF's own establishment delay.

**Open problems.**
* Early-stopping FR at the better ABF allocation (small K, long T).
* A K1024 controlled row at the common rate.
* An *online* stop criterion, for example stopping when the window occupancy reaches its flat-bias share.
* Whether any FR schedule can help when ABF is not establishment-limited (ERI).
* Every system here is a rigid-zeolite, single-guest, overdamped surrogate.

## 4. Limitations

* **Scope.** One stop time per cell, chosen by a frozen rule; no schedule optimisation. The exploratory ladder E1x
  covers four seeds.
* **Rates.** They are safety-calibrated, not tuned.
* **Overdamped dynamics.** Euler–Maruyama with unit mobility, at the validated timesteps (Gate 2).
* **Fixed reference grid.** 180 bins.
* **Budget.** The remaining budget (≤ 200 CPU core-h in aggregate, no GPU available) excluded the matrix K1024 row,
  ERI loading studies, a circulation-correction experiment and early-stopping FR at small K.
