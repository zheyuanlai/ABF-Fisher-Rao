# Experiment A — early-stopping FR at the efficient ABF allocation, and the equal-budget question

*Preregistered: `PREREGISTRATION.md` §A (frozen bc78e255, 2026-10-17 06:22 UTC). Run 06:22–06:35 UTC. Analysed with the
frozen `scripts/convergence_fr/analyze_block.py` (the ABF arm read from the historical EB directory) and
`scripts/adaptive_fr/equal_budget4.py`. Figures: `figures/adaptive_fr/equal_budget_ite300.png` and `block_A.png`.*

## A.1 Pairing and validity (Gate F, Gate E, safety)

* **ABF arm.** The 16 historical EB runs (seeds 1140000–015), reused read-only.
  * They are complete, their configurations equal their job file, and they were produced by the identical engine
    (sha256).
  * The FR arm (new) differs only in `fr_rate` (1.0; inert in the ABF arm, tested) and `fr_stop_steps` (100 000).
* **Pairing.** The arms are bitwise identical at every save before FR start (16/16, asserted). Force evaluations are
  equal: 512 000 256 per run.
* **Schedule.** Exactly 9000 FR opportunities per run, and no FR event after the stop.
* **Blind genealogy check, before any error was read:** min median ESSw/K 0.535, max lineage 0.029, 26 replacements
  per t.u. → PASS.

## A.2 Result (fixed K = 256, T = 200 t.u., paired)

#### A: K 256, T 200 t.u., 300 K, 180 bins, FR rate 1 from 1 t.u., stopped at 10 t.u.; 16 paired seeds 1140000–1140015

* integrity: bitwise paired before the first FR opportunity 16/16; force evaluations per run 512,000,256 (equal in both arms); arms differ only in ['fr_rate', 'fr_stop_steps']; FR opportunities per run 9000, FR events after the stop 0
* ladder (kJ/mol): 0.200, 0.160, 0.120, 0.100, 0.080 (= [0.08, 0.064, 0.048, 0.04, 0.032] k_BT)
* **PRIMARY S_geo 1.281 [0.904, 1.426]** (97.5 %: [0.879, 1.480]); FR faster in 11/16 seeds → **UNRESOLVED**

| statistic | median per-seed FR/ABF − 1 [95 % CI] (FR better) | ABF median | FR median |
|---|---|---|---|
| post-ramp Ī_F (companion) | -15.3 % [-22.4, -4.1] (12/16) | 0.1079 | 0.0901 |
| full-horizon Ī_F | -11.4 % [-16.6, -2.3] (12/16) | 0.1454 | 0.1295 |
| post-ramp Ī_F over [t_FR, 10] | -25.1 % [-29.6, -15.7] (16/16) | 0.7019 | 0.5374 |
| post-ramp Ī_F over [t_FR, 2 t_stop] (H1) | -25.5 % [-29.9, -16.3] (16/16) | 0.4980 | 0.3761 |
| e_F(T/4) | -20.6 % [-28.5, +12.2] (11/16) | 0.0937 | 0.0826 |
| e_F(T/2) | -5.6 % [-24.4, +12.4] (9/16) | 0.0599 | 0.0436 |
| e_F(3T/4) | +3.5 % [-15.3, +22.5] (7/16) | 0.0379 | 0.0374 |
| late-window Ī_F [0.8T, T] | +0.4 % [-15.9, +17.3] (8/16) | 0.0361 | 0.0354 |
| e_F(T) (H2) | -0.9 % [-14.1, +14.0] (8/16) | 0.0344 | 0.0310 |
| e_F′(T) | +2.8 % [-9.4, +8.6] (7/16) | 0.0698 | 0.0670 |
| estimate from deposits after t_stop (H4) | +8.3 % [-9.1, +21.4] (7/16) | 0.0248 | 0.0240 |

| ε (kJ/mol) | sustained S [95 % CI] | reached ABF / FR | FR faster / ABF faster | first-hit S |
|---|---|---|---|---|
| 0.200 | 1.36 [1.00, 1.79] | 16 / 16 | 11 / 4 | 1.35 |
| 0.160 | 1.33 [1.04, 1.74] | 16 / 16 | 12 / 4 | 1.38 |
| 0.120 | 1.14 [0.92, 1.32] | 16 / 16 | 10 / 6 | 1.25 |
| 0.100 | 1.20 [0.92, 1.44] | 16 / 16 | 10 / 6 | 1.18 |
| 0.080 | 1.07 [0.90, 1.33] | 16 / 16 | 9 / 6 | 1.03 |

* crossing (FR permanently worse after): 8/16 seeds; median curve none, bootstrap 2.5/50/97.5 % 59.5/197.0/198.5 t.u. (fraction crossing 0.44)
* window establishment: ABF 4.50, FR 3.00 t.u.; paired -1.50 [-1.50, -1.00]
* FR genealogy: min windowed ESS/K 0.535, max lineage share 0.029, replacements 237 (median over seeds); true translocations ABF 1712 vs FR 1698
* pooled over seeds vs MC: whole run ABF 0.0263 / FR 0.0248; post-window deposits ABF 0.0127 / FR 0.0145; FR vs ABF post-window 0.0038
* wall-clock per run (single thread, production phase): ABF 742 s, FR 751 s (+1.2 %)

### Hypotheses (frozen rules)

| hypothesis | outcome |
|---|---|
| **PRIMARY**: FR-stop improves time to accuracy | **UNRESOLVED**: S_geo 1.28 [0.90, 1.43]; 11/16 seeds faster; every rung S 1.07–1.36, none resolved except 0.16 kJ/mol |
| H1: early gain preserved ([1, 20] t.u.) | **SUPPORTED**: −25.5 % [−29.9, −16.3], 16/16 |
| H2: no late penalty (e_F(T); margin +10 %) | **INCONCLUSIVE**: −0.9 % [−14.1, +14.0] |
| H4: post-stop estimate not worse | **INCONCLUSIVE**: +8.3 % [−9.1, +21.4] |
| companion: post-ramp Ī_F | −15.3 % [−22.4, −4.1], 12/16 |

## A.3 Two algorithms × two allocations at an equal force-evaluation budget (§A.5)

### Production force evaluations only (B = 5.12 × 10⁸)

| configuration | median per-seed geometric-mean τ_B (10⁸) | τ at 0.20 (reached) | τ at 0.16 (reached) | τ at 0.12 (reached) | τ at 0.10 (reached) | τ at 0.08 (reached) | e_F at budget |
|---|---|---|---|---|---|---|---|
| ABF, K1024 × 50 t.u. (E1 ABF arm) | 2.977 | 1.80 (16) | 2.32 (16) | 3.10 (16) | 3.91 (16) | 4.96 (9) | 0.0763 |
| FR-stop-10, K1024 × 50 t.u. (E1, rate 2.0) | 1.797 | 1.09 (16) | 1.48 (16) | 1.93 (16) | 2.40 (16) | 3.06 (15) | 0.0519 |
| ABF, K256 × 200 t.u. (EB) | 0.931 | 0.57 (16) | 0.68 (16) | 0.95 (16) | 1.29 (16) | 1.66 (16) | 0.0344 |
| FR-stop-10, K256 × 200 t.u. (A, rate 1.0) | 0.850 | 0.46 (16) | 0.60 (16) | 0.88 (16) | 1.08 (16) | 1.41 (16) | 0.0310 |

| ratio of medians (unpaired, independent bootstrap) | value [95 % CI] |
|---|---|
| FR_K256 / ABF_K256 | 0.913 [0.646, 1.086] |
| FR_K1024 / ABF_K1024 | 0.604 [0.512, 0.747] |
| ABF_K256 / ABF_K1024 | 0.313 [0.272, 0.369] |
| FR_K256 / FR_K1024 | 0.473 [0.323, 0.574] |
| e_F at budget, best FR / best ABF | 0.900 [0.728, 1.240] |

### All-in (equilibration added), compared at the common smaller budget 5.248e+08

| configuration | median per-seed geometric-mean τ_B (10⁸) | τ at 0.20 (reached) | τ at 0.16 (reached) | τ at 0.12 (reached) | τ at 0.10 (reached) | τ at 0.08 (reached) | e_F at budget |
|---|---|---|---|---|---|---|---|
| ABF, K1024 × 50 t.u. (E1 ABF arm) | 3.510 | 2.32 (16) | 2.83 (16) | 3.61 (16) | 4.42 (16) | 5.25 (7) | 0.0849 |
| FR-stop-10, K1024 × 50 t.u. (E1, rate 2.0) | 2.330 | 1.60 (16) | 1.99 (16) | 2.44 (16) | 2.91 (16) | 3.57 (15) | 0.0567 |
| ABF, K256 × 200 t.u. (EB) | 1.074 | 0.70 (16) | 0.81 (16) | 1.08 (16) | 1.41 (16) | 1.79 (16) | 0.0344 |
| FR-stop-10, K256 × 200 t.u. (A, rate 1.0) | 0.994 | 0.59 (16) | 0.72 (16) | 1.00 (16) | 1.20 (16) | 1.54 (16) | 0.0310 |

| ratio of medians (unpaired, independent bootstrap) | value [95 % CI] |
|---|---|
| FR_K256 / ABF_K256 | 0.925 [0.691, 1.075] |
| FR_K1024 / ABF_K1024 | 0.664 [0.587, 0.789] |
| ABF_K256 / ABF_K1024 | 0.306 [0.272, 0.353] |
| FR_K256 / FR_K1024 | 0.427 [0.311, 0.495] |
| e_F at budget, best FR / best ABF | 0.900 [0.728, 1.240] |

**Question B (best FR allocation vs best ABF allocation): UNRESOLVED** (production) and
**UNRESOLVED** (all-in).
* For both algorithms, the best tested allocation is K256 × 200 t.u.
* Best FR / best ABF = 0.91 [0.65, 1.09] (production) and 0.93 [0.69, 1.08] (all-in).

## A.4 Why the fixed-K effect shrinks at the efficient allocation (from `MECHANISM.md` D1)

Bias–variance decomposition of the final F error over the 16 seeds (squared bias corrected for finite seeds and the
reference noise):

| configuration | final RMSE | \|bias\| | seed s.d. | bias share of MSE |
|---|---|---|---|---|
| ABF K1024 × 50 (E1) | 0.0770 | 0.0734 | 0.0226 | 0.91 |
| FR-stop K1024 × 50 (E1) | 0.0535 | 0.0480 | 0.0233 | 0.81 |
| ABF K256 × 200 (EB) | 0.0351 | 0.0252 | 0.0241 | 0.51 |
| FR-stop K256 × 200 (A) | 0.0340 | 0.0237 | 0.0241 | 0.49 |

* **At K1024 × 50 the final error is about 90 % bias.** That is the memory of early non-equilibrium deposits, and
  early FR reduces it (0.073 → 0.048).
* **At K256 × 200 the same memory has been diluted by four times the physical time.** The error is about half
  variance, the bias left for FR to remove is small (0.025 → 0.024), and the seed scatter is unchanged.
* **Consequence.** FR's mechanism acts on the early transient. A longer run already removes most of that transient,
  which is why it is also the more efficient ABF allocation.

## A.5 Answers

1. **At ITE 300 K, K = 256, T = 200, does early-stopping FR outperform ABF in time to accuracy?**
   * **Not resolved.** The point estimate favours FR (S_geo 1.28; 11/16 seeds), but the 95 % CI [0.90, 1.43]
     includes 1.
   * The early error is resolvably lower (−25 % over [1, 20], 16/16), and so is the post-ramp integrated error
     (−15 %).
   * The final error is unchanged (−1 % [−14, +14]).
2. **Does early-stopping FR beat the best ABF allocation at equal force evaluations?**
   * **Not resolved.** The best FR allocation is also K256: 0.91 [0.65, 1.09] (production) and 0.93 [0.69, 1.08]
     (all-in).
   * FR at K1024 remains about 2× worse than either K256 configuration.
   * B does not follow from A, and neither is established.
