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

## E2 — ITE 150 K early-stopping FR (CONFIRMATORY; Amendment A2)

* **Cell.** X2's model, K and rate (K 256, rate 0.5), with the campaign's 180-bin protocol and T 100.
* **Stop time.** t_stop = **15 t.u.**, by the same frozen rule applied to X2 only: 2 × ABF's establishment time of
  7.5 t.u., below X2's crossing 2.5 % quantile of 32.5 t.u.
* **Primary ladder.** ABF-anchored (A2).

#### E2: K 256, T 100 t.u., 150 K, 180 bins, FR rate 0.5 from 1 t.u., stopped at 15 t.u.; 16 paired seeds 1130000–1130015

* integrity: bitwise paired before the first FR opportunity 16/16; force evaluations per run 256,000,256 (equal in both arms); arms differ only in ['fr_stop_steps']; FR opportunities per run 14000, FR events after the stop 0
* ladder (kJ/mol): 0.241, 0.201, 0.161, 0.129, 0.105 (= [3.0, 2.5, 2.0, 1.6, 1.3] × median ABF e_F(T) 0.0805)
* **PRIMARY S_geo 1.250 [1.115, 1.582]** (97.5 %: [1.088, 1.669]); FR faster in 14/16 seeds → **SUPPORTED (FR+ reaches the ladder sooner)**

| statistic | median per-seed FR/ABF − 1 [95 % CI] (FR better) | ABF median | FR median |
|---|---|---|---|
| post-ramp Ī_F (companion) | -18.1 % [-27.9, -13.1] (15/16) | 0.2371 | 0.2081 |
| full-horizon Ī_F | -14.4 % [-20.9, -9.7] (15/16) | 0.3104 | 0.2804 |
| post-ramp Ī_F over [t_FR, 10] | -17.7 % [-22.6, -14.0] (16/16) | 0.8964 | 0.7398 |
| post-ramp Ī_F over [t_FR, 2 t_stop] (H1) | -18.2 % [-24.4, -16.2] (16/16) | 0.5067 | 0.4222 |
| e_F(T/4) | -17.0 % [-28.3, -12.2] (15/16) | 0.2654 | 0.2344 |
| e_F(T/2) | -22.6 % [-34.6, -7.9] (14/16) | 0.1520 | 0.1197 |
| e_F(3T/4) | -20.9 % [-31.7, -4.5] (12/16) | 0.1079 | 0.0856 |
| late-window Ī_F [0.8T, T] | -20.0 % [-34.1, -6.3] (12/16) | 0.0881 | 0.0805 |
| e_F(T) (H2) | -19.0 % [-31.6, -6.3] (13/16) | 0.0805 | 0.0764 |
| e_F′(T) | -14.0 % [-21.6, -1.5] (13/16) | 0.1438 | 0.1221 |
| estimate from deposits after t_stop (H4) | +14.9 % [+0.4, +58.0] (4/16) | 0.0361 | 0.0413 |

| ε (kJ/mol) | sustained S [95 % CI] | reached ABF / FR | FR faster / ABF faster | first-hit S |
|---|---|---|---|---|
| 0.241 | 1.29 [1.13, 1.76] | 16 / 16 | 13 / 2 | 1.30 |
| 0.201 | 1.29 [1.07, 1.52] | 16 / 16 | 12 / 3 | 1.39 |
| 0.161 | 1.24 [1.06, 1.56] | 16 / 16 | 14 / 2 | 1.29 |
| 0.129 | 1.25 [1.12, 1.48] | 16 / 16 | 14 / 2 | 1.25 |
| 0.105 | 1.33 [1.02, 1.55] | 14 / 14 | 12 / 3 | 1.22 |

* crossing (FR permanently worse after): 3/16 seeds; median curve none, bootstrap 2.5/50/97.5 % 57.0/80.2/99.0 t.u. (fraction crossing 0.01)
* window establishment: ABF 6.69, FR 4.12 t.u.; paired -2.69 [-3.00, -1.88]
* FR genealogy: min windowed ESS/K 0.452, max lineage share 0.035, replacements 194 (median over seeds); true translocations ABF 404 vs FR 404
* pooled over seeds vs MC: whole run ABF 0.0748 / FR 0.0595; post-window deposits ABF 0.0168 / FR 0.0179; FR vs ABF post-window 0.0047
* wall-clock per run (single thread, production phase): ABF 372 s, FR 373 s (+0.2 %)

| hypothesis | outcome (frozen rule) |
|---|---|
| **H3 (primary)** | **SUPPORTED**: S_geo 1.25 [1.11, 1.58], FR faster in 14/16; every rung > 1 |
| H1 (post-ramp Ī_F over [1, 30]) | **SUPPORTED**: -18.2 % [-24.4, -16.2] |
| H2 (e_F(T)) | **REMOVED, and reversed**: -19.0 % [-31.6, -6.3], 13/16 |
| H4 (post-stop estimate) | **CONTRADICTED by the frozen rule**: per seed +14.9 % [+0.4, +58.0], 4/16 FR better |

**H4 in more detail.**
* **No pooled bias.** Pooled over the 16 seeds, the post-stop estimates of the two arms are 0.018 vs 0.017 kJ/mol
  from MC and 0.005 apart. The late-half pooled FR − ABF D is 0.000 (noise 0.016).
* **The per-seed deficit is therefore scatter, not a bias.** After the stop both arms run identical ABF dynamics, so
  the larger per-seed scatter must come from the state that FR left behind.
* **Possible causes (hypotheses, not established).** The applied bias of the FR-stop arm differs. Clones made just
  before the stop could still share their slow hidden coordinates. At 300 K (E1) the same statistic was
  −0.0 % [−21, +25].

**Context (cross-block and descriptive, same model at 150 K and K 256):**

| block | FR schedule | rate | horizon | Δe_F(T) | ΔĪ_F post-ramp | S_geo (ABF ladder) |
|---|---|---|---|---|---|---|
| X2 (exploratory) | continuous | 0.5 | 200 (360 bins) | +56.8 % | -1.4 % | 0.82 [0.60, 0.95] |
| X2 truncated at 50 (atlas) | continuous | 0.5 | 50 | +10.5 % | -18.0 % | – |
| M150_256 (confirmatory) | continuous | 1.0 | 50 | +28.9 % | -22.1 % | 1.63 [1.34, 2.73] |
| **E2 (confirmatory)** | **stopped at 15** | 0.5 | 100 | **-19.0 %** | **-18.1 %** | **1.25 [1.11, 1.58]** |

* **Late-time decay.** ABF -0.87, FR-stop -0.81. Continuous FR in the
  150 K cells: -0.52 (X2) and -0.45 (M150_256).
* **Memory bias.** The pooled whole-run estimate vs MC is ABF 0.074 and FR-stop 0.058.
  Hidden-coordinate conditionals are clean in both arms.

## E1x — exploratory t_stop ladder (Amendment A3; 4 X1 seeds; descriptive only)

* **Design.** Each schedule is paired with the X1 ABF arm of the same seed. Every stopped arm is bitwise identical to
  the X1 continuous-FR arm of the same seed at every save before its stop (asserted).
* **The t_stop = 10 point** comes from E1, on different seeds.

| schedule (same 4 seeds 980000–980003) | S_geo per seed | post-ramp Ī_F per seed | e_F(T) per seed |
|---|---|---|---|
| t_stop 5 | 1.39, 1.46, 1.40, 1.48 (median 1.43) | -33 %, -30 %, -33 %, -30 % | -25 %, -31 %, -23 %, -24 % |
| t_stop 20 | 1.14, 0.99, 1.19, 1.09 (median 1.11) | -27 %, -15 %, -28 %, -19 % | -14 %, -1 %, -27 %, +4 % |
| continuous (X1) | 1.07, 0.97, 1.07, 1.01 (median 1.04) | -24 %, -15 %, -24 %, -16 % | -0 %, +6 %, +14 %, +15 % |
| t_stop 10 (E1, 16 other seeds; for orientation only) | median 1.59 | median −38 % | median −32 % |

* **Reading (exploratory).** On every seed, the earlier stops (5 and 10 t.u.) beat the later one (20 t.u.), which in
  turn beats continuous FR. Stopping at 5 t.u., about 2× FR's own establishment time, is as good as 10.
* **Interpretation.** FR's useful work is done within the first few t.u., and every additional t.u. of FR costs
  accuracy. This is consistent with the mechanism in `MECHANISM.md`.
* **Status.** It has 4 seeds and is not a confirmatory schedule comparison.

## Answers

* **Q1 (ITE 300 K).** Yes. Early-stopping FR reaches every rung of the frozen 0.20–0.08 kJ/mol tolerance ladder
  sooner than ABF, in every seed, by a factor of 1.45–1.73, with a preregistered geometric mean of 1.59
  [1.31, 1.85]. That is at fixed K = 1024 and equal force evaluations (equal wall-clock to 0.2 %).
* **Q2.** Yes. The late penalty of continuous FR (+14 % final error in X1, exploratory) is not present with
  early-stopping FR. Its final error is 32 % **lower** than ABF's, and no seed's error curve crosses ABF's.
* **Replication at 150 K (E2).** The result holds in the colder, more establishment-limited cell, with its own
  frozen stop time (15 t.u.) and rate (0.5):
  * time-to-accuracy speedup 1.25 [1.12, 1.58];
  * final error −19 % [−32, −6] at T 100.

  Continuous FR in the same model, K and rate ended **+57 %** at T 200 (X2) and +29 % at T 50 (M150_256, rate 1.0).
* **Limits.**
  * Two cells, one stop time each (chosen by a rule), fixed K. ERI could not be tested: the frozen rule gave no
    admissible stop.
  * The equal-budget comparison with ABF at a different K is in `FINAL_SYNTHESIS.md` (block EB).
