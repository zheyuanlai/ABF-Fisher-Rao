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

## 2. Gateway (exploratory, Amendment 3): **exploratory decision UNRESOLVED; robust to the backend's law**

64 seeds 8400–8463 × 5 arms; 320/320 runs plus contention repeats; 28.2 used core-h; drift reference 1.04–1.07 × 10⁸
walker-steps/s. Stage COMPLETE. B = 3.2768 × 10⁹ walker-steps per run, mid threshold e_F 0.0055. Every number in this
section is **exploratory (after a frozen V-B4 FAIL; Amendment 3)**.

### 2.1 Primary endpoint W_mid (0 censored in every arm)

| arm | implementation | median W_mid (s) | ρ(FR 64, arm) [95 % CI] |
|---|---|---|---|
| **FR 64** (completion arm) | parallel t 8 | **1.30** [0.89, 1.67] | — |
| **ABF 64** | parallel t 8 | **1.71** [1.41, 2.07] | **0.763 [0.488, 1.087]** |
| ABF 16 | parallel t 2 | 1.77 [1.42, 2.46] | 0.735 [0.429, 1.026] |
| FR 512 | parallel t 8 | 2.86 [2.64, 3.33] | — |
| ABF 512 | parallel t 8 | 7.61 [7.18, 8.10] | 0.171 [0.116, 0.221] |

* **[observed, exploratory] Decision: unresolved.** FR 64 beats ABF 512 but not ABF 16 or ABF 64 at the IUT level (FR-side
  upper bounds 1.10 and 1.12). ρ_best = 0.763 [0.515, 1.109]. FR 64 is the best FR arm in 100 % of resamples; the best
  ABF arm is ABF 64 in 64 % and ABF 16 in 36 %. The fraction of resamples with ρ_best < 1 is 0.90 (descriptive).
* **[observed] A3.3 (b) hybrid projection** (validated-engine τ_mid × measured wall per walker-step): unresolved,
  ρ_best 0.719 [0.527, 1.187]. The measured and projected labels agree, so the exploratory decision is **robust to the
  backend's law**: the V-B4 question (whether counter-mode noise changes the law) does not change the answer.
* **Sensitivities (all unresolved):** interpolated 0.756 [0.513, 1.110]; flagged seeds included and CCD-wide
  contention scope both 0.763 [0.515, 1.109].
* **[observed, exploratory] Fixed N.** FR 512 vs ABF 512, paired at t 8: FR first in 60/64 seeds (sign test
  p 7 × 10⁻¹⁴), ρ 0.375 [0.340, 0.447]; FR adds +0.91 s [0.77, 1.00] per ≈ 30 s run (≈ 3 %).

### 2.2 Secondary endpoints (exploratory)

| arm | W_loose (s) | W_strict (s) (censored /64) | final e_F | force evaluations to mid | used core-h to mid (×10⁻³) | speed-up vs serial (efficiency) |
|---|---|---|---|---|---|---|
| FR 64 | **0.21** | **5.53** (1) | 5.1e-4 | 8.81e7 | 2.90 | 5.20 (0.65) |
| ABF 64 | 0.32 | 6.30 (0) | **4.4e-4** | 1.21e8 | 3.80 | 5.39 (0.67) |
| ABF 16 | 0.39 | 16.9 (0) | 5.2e-4 | **5.16e7** | **0.99** | 2.21 (1.10) |
| FR 512 | 0.71 | 12.2 (1) | 7.5e-4 | 3.10e8 | 6.35 | 8.19 (1.02) |
| ABF 512 | 1.22 | 24.0 (13) | 1.6e-3 | 8.51e8 | 16.9 | 8.44 (1.06) |

* **Loose threshold (0.02):** FR advantage, ρ_best 0.654 [0.581, 0.732]: FR 64 beats every ABF arm.
* **Strict threshold (0.002):** unresolved, ρ_best 0.878 [0.662, 1.485]; all contrasts comparable.
* **Integrated error against wall-clock**, Ī_F^wall at W\* = 29.3 s: FR advantage, ρ_best 0.85 [0.79, 0.94]. FR 64
  0.0026, ABF 64 0.0030, ABF 16 0.0038.
* **Force evaluations (claim type 2):** ABF 16 needs the fewest to the mid threshold (5.2e7, against 8.8e7 for FR 64).
  This matches the equal-budget result that the best gateway allocation is ABF at N 16.
* **Core-hours:** ABF 16 on 2 threads is the cheapest (1.0 × 10⁻³ core-h to mid), about 3× cheaper than FR 64.
* Efficiencies above 1 mean the parallel engine's counter-mode noise is cheaper than the serial engine's MT19937 noise
  (1.24–1.52×, §4.1 of the backend report). They are not superlinear scaling.

## 3. Combined reading (claim type 3)

* **[observed] Neither system resolves a wall-clock advantage of the best FR allocation over the best ABF allocation at
  the mid threshold** on 8 dedicated cores (LTA confirmatory: ρ_best 0.717 [0.536, 1.196]; gateway exploratory: 0.763
  [0.515, 1.109]). The point estimates favour FR by 24–28 % in both. The preregistered resolution statement predicted
  CI widths of ×/÷1.22–1.3 (from resampling τ at fixed throughput), but the observed ρ_best CIs are ×/÷1.47–1.49, so
  the design was underpowered for an effect of this size.
* **[observed] The best FR allocation is N 64 in both systems, not N 512.** On this hardware the parallel backend
  makes N 512 fast (8.2–9.1× the serial engine), but N 512 is not the best allocation for either method. The decisive
  comparison is therefore moderate-N FR (8 threads) against moderate-N ABF (8 threads, or 2 threads at N 16), not "large
  N against small N".
* **[observed] FR helps earlier accuracy more than late accuracy.** Gateway: FR wins at the loose threshold and on
  Ī_F^wall, and is unresolved at mid and strict. LTA: unresolved at loose and mid, and worse at strict (FR reaches it
  less often within the budget).
* **[observed] FR does not reduce CPU cost.** In core-hours and force evaluations to mid accuracy, a small-N ABF arm is
  cheapest in both systems (gateway ABF 16; LTA ABF 2). This agrees with the equal-budget result (`EVIDENCE_LEDGER.md`
  C5).
* **[observed] At fixed N, FR's simulation-time gain carries over to wall-clock on 8–16 cores** (FR 512 first in 60/64
  and 31/32 seeds; overhead 3 % / 10 %), extending C6 beyond a single core.
* **[hypothesis]** A larger study (≥ 128 seeds per arm) could resolve an FR advantage of ≈ 25 % at the mid threshold
  on idle cores. Even then it would be a wall-time advantage at higher CPU cost, confined to early and intermediate
  accuracy.

## 4. Compute and files

* Used core-hours: LTA production 25.6, gateway production 28.2. The whole of Experiment B used ≈ 92 of its 160 cap
  (Amendment 4); 0 GPU-h.
* Records: `results/final_campaign/expB/prod/<system>/<arm>/s<seed>[.rep<k>].json` (science arrays `.npz` gitignored,
  sha256 in the records); analysis `results/final_campaign/expB/analysis/<system>/{summary.json, hybrid_projection.json}`;
  seed tables `results/final_campaign/expB/tables/<system>_seeds.{csv,md}`; figures `figures/final_campaign/expB/` B1–B7
  (all pass the legibility check).
