# Experiment B: wall-clock comparison on genuinely parallel CPU hardware

**2026-10-11.** Preregistration `docs/final_campaign/PREREGISTRATION.md` §2.5–§2.8 (frozen `c5a2264f`) with Amendments
2–4. Backend, validation and feasibility: `docs/final_campaign/EXPB_PARALLEL_BACKEND.md`. Every statement is typed
**[identity]**, **[observed]** or **[hypothesis]**. The four claim types are kept separate. This report is about claim
type **(3)**: wall-clock benefit on specified hardware. It also reports type (2) quantities, force evaluations, as
secondary endpoints.

**Status of the two systems.**
* **LTA 150 K — confirmatory.** V-B1…V-B5 PASS (V-B4 after its one pre-specified extension) and F1/F1b PASS, so
  production was licensed by the frozen rules.
* **Gateway — exploratory.** The frozen V-B4 verdict was FAIL, which stops the confirmatory gateway study (§3 rule ii).
  The gateway was continued under Amendment 3 (A3.3) after a post-hoc diagnostic showed the gate could not certify
  equivalence even for the validated engine against itself. Every gateway number below carries that label, and its
  decision is read together with the frozen law-insensitive hybrid projection (A3.3 b).

**Hardware and protocol (both systems).** 2 × AMD EPYC 9554 (128 cores), shared with other users. Each timed run has its
own CCD(s): t ≤ 8 threads on one CCD, t = 16 on two CCDs of one node. 1 Hz telemetry; contention judged on the run's own
cores and their SMT siblings (A2.4), repeat until uncontended (≤ 3 repeats). Fixed walker-step budget B per run, mid
threshold as the primary endpoint (W_mid = wall seconds from the timed start to the first save of the persistent window;
censored = ∞). Bootstrap 10 000, `default_rng(20261010)`; IUT decision over all FR × ABF arms (Bonferroni over FR arms).

## 1. LTA 150 K (confirmatory): **decision UNRESOLVED**

32 seeds 15200–15231 × 6 arms; 192/192 runs plus 5 contention repeats; 25.6 used core-h; drift reference stable
(1.055–1.072 × 10⁸ walker-steps/s over the run). Stage COMPLETE (0 missing runs, 0 pending repeats). B = 3.072 × 10⁸
walker-steps per run, mid threshold 0.13 kJ/mol.

### 1.1 Primary endpoint W_mid (median wall seconds, 95 % CI; 0 censored in every arm)

| arm | implementation | median W_mid (s) | ρ(FR 64, arm) [95 % CI] |
|---|---|---|---|
| **FR 64** (completion arm) | parallel t 8 | **2.37** [1.84, 3.76] | — |
| **ABF 32** (completion arm) | parallel t 8 | **3.31** [2.46, 3.69] | **0.717 [0.534, 1.196]** |
| ABF 64 | parallel t 8 | 4.41 [3.47, 4.98] | 0.538 [0.406, 0.896] |
| FR 512 | parallel t 16 | 7.76 [6.82, 9.28] | — |
| ABF 2 | serial engine | 11.25 [8.31, 15.31] | 0.211 [0.140, 0.365] |
| ABF 512 | parallel t 16 | 15.05 [13.52, 17.01] | 0.158 [0.119, 0.260] |

* **[observed] Frozen decision: unresolved.** FR 64 beats ABF 2, ABF 64 and ABF 512 at the IUT level, but not the best
  ABF allocation, ABF 32: the FR-side upper bound at level 0.975 (two FR arms) is 1.30 > 1. No ABF arm beats every FR
  arm either. ρ_best (best FR / best ABF, re-selected in every resample) = 0.717 [0.536, 1.196]. FR 64 is the best FR
  arm in 100 % of resamples and ABF 32 the best ABF arm in 97 %. The fraction of resamples with ρ_best < 1 is 0.89 (a
  descriptive number, not a probability).
* **Sensitivities (all unresolved):**
  * W_mid interpolated: ρ_best 0.72 [0.54, 1.19];
  * flagged seeds included: 0.72 [0.54, 1.20];
  * **CCD-wide contention verdict** (A2.4): 0.72 [0.54, 1.20];
  * hybrid projection (validated-engine τ × measured speed; descriptive for LTA): 0.635 [0.44, 1.37]. It agrees with the
    measured decision.
* **[observed] Fixed N (claim types 1/3 at N 512).** FR 512 vs ABF 512, paired on shared noise at the same t 16: FR
  reaches the mid threshold first in 31/32 seeds (sign test p 1.5 × 10⁻⁸), ρ 0.515 [0.440, 0.650]. FR's added wall
  time per run is +2.94 s [2.75, 3.20], about +10 % of the ≈ 29–32 s run. The single-core result of the mechanism
  campaign (C6) therefore holds on 16 cores.

### 1.2 Secondary endpoints

| arm | W_loose (s) | W_strict (s) (censored /32) | fraction reaching strict | final e_F (kJ/mol) | force evaluations to mid | used core-h to mid (×10⁻³) | allocated core-h to mid (×10⁻³) | speed-up vs serial (efficiency) |
|---|---|---|---|---|---|---|---|---|
| FR 64 | **0.56** | ∞ (16) | 0.50 | 0.030 | **1.22e7** | 5.27 | 10.6 | 4.46 (0.56) |
| ABF 32 | 0.64 | 45.1 (9) | 0.72 | 0.024 | 1.48e7 | 7.35 | 14.7 | 3.85 (0.48) |
| ABF 64 | 0.79 | **43.2** (7) | 0.78 | 0.023 | 2.48e7 | 9.81 | 19.6 | 4.81 (0.60) |
| FR 512 | 1.57 | ∞ (24) | 0.25 | 0.037 | 7.67e7 | 34.5 | 34.5 | 8.35 (0.52) |
| ABF 2 | 1.04 | 200.8 (9) | 0.72 | 0.023 | 1.32e7 | **3.13** | 50.0 | 1.00 (1.00) |
| ABF 512 | 3.50 | ∞ (32) | 0.00 | 0.063 | 1.60e8 | 66.9 | 66.9 | 9.12 (0.57) |

* **[observed] Loose threshold (0.5 kJ/mol):** unresolved, ρ_best 0.87 [0.66, 1.22].
* **[observed] Strict threshold (0.03 kJ/mol):** FR does *worse*. FR 64 reaches it within the budget in only 50 % of
  runs and FR 512 in 25 %, against 72–78 % for ABF 2/32/64. The median W_strict of both FR arms is censored (∞), and
  ρ_best is ∞ [0.98, ∞]. The frozen rule labels FR 64 vs ABF 32 and vs ABF 64 "comparable" and the other contrasts
  "not comparable (budget artefact)"; the decision is unresolved. FR's final errors (0.030, 0.037) are higher than
  ABF's at N ≤ 64 (0.023–0.024).
* **[observed] Integrated error against wall-clock**, Ī_F^wall at W\* = 28.7 s (ABF 512's median completion):
  unresolved. FR 64 0.113, ABF 32 0.108, ABF 64 0.124, ABF 2 0.193, FR 512 0.226, ABF 512 0.301.
* **[observed] Force evaluations to the mid threshold (claim type 2):** FR 64 needs the fewest (1.22e7), but ABF 2
  (1.32e7) and ABF 32 (1.48e7) are within about 20 %. This matches the equal-budget finding that FR does not change the
  best allocation (`docs/equal_budget/FINAL_RESULTS.md`, Result 3).
* **[observed] Core-hours.** The cheapest arm in used core-hours is single-core ABF 2 (3.1 × 10⁻³ core-h to mid,
  against 5.3 × 10⁻³ for FR 64 on 8 threads). FR 64's advantage is in wall time on idle cores, not in CPU cost.

### 1.3 Reading (LTA)

* **[observed]** On 8 dedicated cores, the best FR allocation (FR 64) reaches mid accuracy about 28 % sooner than the
  best ABF allocation (ABF 32) by point estimate, but the preregistered test does not resolve it (CI up to 1.20). It
  loses at the strict threshold, where FR runs more often fail to converge within the budget.
* **[observed]** The parallel backend makes large N affordable in wall-clock (FR 512 is 8.3× the serial engine), but
  large N itself is not the best allocation for either method: both best arms sit at N 32–64.
* **[hypothesis]** FR's lower strict-threshold success reflects a residual finite-time bias of FR at fixed budget,
  the same pattern as FR's higher final error at N ≤ 64. A stationary bias, as in WCA, is not established here; the
  LTA validation found none at the budgets used.

## 2. Gateway (exploratory, Amendment 3)

*(pending: production in progress)*
