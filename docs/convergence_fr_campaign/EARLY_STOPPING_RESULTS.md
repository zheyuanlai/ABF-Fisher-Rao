# Early-stopping Fisher–Rao (Experiment 3)

*Frozen in `PREREGISTRATION.md` §3 (commit 4431ce30, 2026-10-17 03:13 UTC) and A1 (9b044bc8). Run 03:13–03:27 UTC.
Analysed 03:27:50 with the frozen `scripts/convergence_fr/analyze_block.py`.
Figures: `figures/convergence_fr/block_E1.png` and `early_stopping_vs_continuous.png`.*

## Design

* **Algorithm.** ABF + FR with γ(t) = γ0 = 2.0 for 1 ≤ t < t_stop and 0 afterwards. After the stop no FR event and
  no FR random draw occur. ABF continues on the same cumulative C_j and M_j, with no reset of positions, estimator
  or samples (engine `fr_stop_steps`, `tests/test_fr_stop.py` 6/6).
* **Comparator.** Plain histogram ABF on the same seeds. These are the only two arms.
* **Stop time.** t_stop = **10 t.u.**, chosen before any confirmatory run by the frozen rule on the exploratory X1
  data: the smallest multiple of 5 t.u. ≥ 2 × ABF's window-establishment time (4.3), and below the 2.5 % quantile of
  continuous FR's crossing time (19.9).
* **Cell.** ITE 300 K, K 1024, T 50, dt 1e-4, 180 bins: X1's validated configuration. 16 fresh paired seeds.

## E1 result (CONFIRMATORY)

#### E1: K 1024, T 50 t.u., 300 K, 180 bins, FR rate 2 from 1 t.u., stopped at 10 t.u.; 16 paired seeds 1100000–1100015

* integrity: bitwise paired before the first FR opportunity 16/16; force evaluations per run 512,001,024 (equal in both arms); arms differ only in ['fr_stop_steps']; FR opportunities per run 9000, FR events after the stop 0
* ladder (kJ/mol): 0.200, 0.160, 0.120, 0.100, 0.080 (= [0.08, 0.064, 0.048, 0.04, 0.032] k_BT)
* **PRIMARY S_geo 1.586 [1.311, 1.852]** (97.5 %: [1.306, 1.921]); FR faster in 16/16 seeds → **SUPPORTED (FR+ reaches the ladder sooner)**

| statistic | median per-seed FR/ABF − 1 [95 % CI] (FR better) | ABF median | FR median |
|---|---|---|---|
| post-ramp Ī_F (companion) | -37.7 % [-42.2, -28.5] (16/16) | 0.2301 | 0.1425 |
| full-horizon Ī_F | -24.5 % [-28.3, -19.1] (16/16) | 0.3369 | 0.2515 |
| post-ramp Ī_F over [t_FR, 10] | -36.8 % [-39.6, -30.0] (16/16) | 0.5943 | 0.3803 |
| post-ramp Ī_F over [t_FR, 2 t_stop] (H1) | -37.4 % [-40.0, -29.2] (16/16) | 0.4075 | 0.2578 |
| e_F(T/4) | -36.9 % [-43.0, -26.6] (16/16) | 0.2753 | 0.1803 |
| e_F(T/2) | -34.0 % [-45.8, -27.9] (16/16) | 0.1449 | 0.0930 |
| e_F(3T/4) | -37.7 % [-47.0, -25.0] (16/16) | 0.0988 | 0.0664 |
| late-window Ī_F [0.8T, T] | -34.5 % [-47.2, -24.3] (16/16) | 0.0897 | 0.0590 |
| e_F(T) (H2) | -32.3 % [-42.6, -24.1] (16/16) | 0.0763 | 0.0519 |
| e_F′(T) | -22.4 % [-29.0, -13.7] (16/16) | 0.1109 | 0.0881 |
| estimate from deposits after t_stop (H4) | -0.0 % [-21.0, +24.9] (8/16) | 0.0264 | 0.0239 |

| ε (kJ/mol) | sustained S [95 % CI] | reached ABF / FR | FR faster / ABF faster | first-hit S |
|---|---|---|---|---|
| 0.200 | 1.73 [1.39, 1.95] | 16 / 16 | 16 / 0 | 1.73 |
| 0.160 | 1.60 [1.31, 1.82] | 16 / 16 | 16 / 0 | 1.60 |
| 0.120 | 1.53 [1.30, 1.83] | 16 / 16 | 16 / 0 | 1.49 |
| 0.100 | 1.51 [1.31, 1.85] | 16 / 16 | 16 / 0 | 1.51 |
| 0.080 | 1.45 [1.28, 1.68] | 9 / 15 | 15 / 0 | 1.45 |

* crossing (FR permanently worse after): 0/16 seeds; median curve none
* window establishment: ABF 4.38, FR 2.38 t.u.; paired -1.94 [-2.00, -1.75]
* FR genealogy: min windowed ESS/K 0.338, max lineage share 0.020, replacements 1016 (median over seeds); true translocations ABF 1592 vs FR 1604
* pooled over seeds vs MC: whole run ABF 0.0738 / FR 0.0484; post-window deposits ABF 0.0095 / FR 0.0113; FR vs ABF post-window 0.0045
* wall-clock per run (single thread, production phase): ABF 747 s, FR 749 s (+0.2 %)

## Hypotheses (frozen decision rules)

| hypothesis | statistic | rule | outcome |
|---|---|---|---|
| **H3 (primary)** early-stopping FR improves time to accuracy | S_geo over the kT ladder | CI > 1 | **SUPPORTED**: 1.59 [1.31, 1.85]; each rung 1.45–1.73; 16/16 seeds faster |
| H1 the early-time reduction is preserved | post-ramp Ī_F over [1, 20] | CI < 0 | **SUPPORTED**: −37.4 % [−40.0, −29.2], 16/16 |
| H2 the late penalty is removed | e_F(T) | CI upper < +10 % → removed; CI lower > 0 → persists | **REMOVED, and reversed into a gain**: −32.3 % [−42.6, −24.1], 16/16; late window [40, 50] −34.5 % |
| H4 the post-stop-only estimate is accurate | error of F̂ from deposits after t_stop, FR-stop vs ABF | CI upper < +10 % → supported; CI lower > 0 → contradicted | **INCONCLUSIVE by the frozen rule**: −0.0 % [−21.0, +24.9], 8/16. Pooled over the 16 seeds the two post-stop estimates are 0.011 vs 0.0095 kJ/mol from MC, 0.0045 apart. The per-seed post-stop estimates are noise-limited, about 0.025 each |

## Continuous vs early-stopping FR (descriptive cross-block comparison: X1 exploratory vs E1 confirmatory, different seeds)

| | X1: continuous FR vs its ABF | E1: FR stopped at 10 t.u. vs its ABF |
|---|---|---|
| primary S_geo | 1.06 [0.95, 1.15], unresolved | 1.59 [1.31, 1.85], supported |
| post-ramp Ī_F | -22.3 % | -37.7 % |
| e_F(T) | +14.0 % [+0.8, +21.3] | -32.3 % [-42.6, -24.1] |
| estimate from deposits after FR start (X1) or after the stop (E1) | +16.5 % | -0.0 % |
| e_F crossing | 12/16 seeds; median curve 29.1 t.u. [19.9, 46.8] | **0/16 seeds; none** |
| log-log decay slope of e_F on [20, 50], ABF / FR | -0.84 [-1.08, -0.73] / -0.69 [-0.73, -0.51] | -1.00 [-1.07, -0.82] / -1.04 [-1.23, -0.84] |
| FR replacements per run | 4640 | 1016 |
| wall-clock FR vs ABF | – | +0.2 % |

* The two ABF arms agree: median e_F(T) 0.0764 (X1) and 0.0763 (E1).
* Before the stop, the FR arms agree within seed-block noise: median e_F at t = 5 is 0.336 vs 0.315.

## What the data say about the mechanism

1. **ABF's error over the second half of the run decays like 1/t** (slope −0.84 to −1.00), not like the 1/√t of
   statistical noise.
   * The end-of-run error of cumulative histogram ABF is the slowly diluted **memory of early, non-equilibrium
     deposits**.
   * Pooled over 16 seeds, the whole-run ABF estimate is 0.074 kJ/mol from MC. That is nearly the per-seed error
     (0.076), so it is systematic, not noise. The post-t_stop deposits alone are only 0.0095 from MC.
2. **FR during establishment makes those early deposits less biased.** The window is populated 1.9 t.u. sooner. The
   pooled whole-run estimate of FR-stop is 0.048 from MC. That smaller memory error then dilutes at the same 1/t rate
   (slope −1.04), so the proportional advantage (−35 %) persists to T.
3. **Continuous FR adds an ongoing error term.**
   * Its late deposits are biased: X1's post-start estimate is +16.5 %, and its pooled late-half estimate is 0.035
     vs 0.013 from MC.
   * Its error decays only as t^−0.69. The two effects cross at about 29 t.u.
4. **After the stop the FR arm is exactly ABF.** This is bitwise by test. Its late deposits are indistinguishable
   from ABF's (pooled late-half FR − ABF D = 0.000, noise 0.011).
5. **Circulation falls after the stop in both arms.** Read-out c, median over 5 t.u.:
   * ABF 0.079 → 0.062;
   * FR 0.063 → 0.044.
   * The FR − ABF difference in the 5 t.u. after the stop is -0.021 [-0.026, +0.000]. Circulation is not what separates the arms here.

## Answers

* **Q1 (ITE 300 K).** Yes. Early-stopping FR reaches every rung of the frozen 0.20–0.08 kJ/mol tolerance ladder
  sooner than ABF, in every seed, by a factor of 1.45–1.73, with a preregistered geometric mean of 1.59
  [1.31, 1.85]. That is at fixed K = 1024 and equal force evaluations (equal wall-clock to 0.2 %).
* **Q2.** Yes. The late penalty of continuous FR (+14 % final error in X1, exploratory) is not present with
  early-stopping FR. Its final error is 32 % **lower** than ABF's, and no seed's error curve crosses ABF's.
* **Limits.**
  * One cell, one stop time (chosen by a rule), one rate, fixed K.
  * The equal-budget comparison with ABF at a different K is in `FINAL_SYNTHESIS.md` (block EB).
