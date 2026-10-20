# Results: WCA Fisher–Rao activation time and early stopping (validated dt, exact MC reference)

## Setup

* 16 paired seeds (5600–5615), K = 256, dt = 1.25 × 10⁻⁴, T = 240.
* 112 runs, all complete. All ran from commit e25edb3e, which holds the preregistration, pushed before any run.
* Errors are scored against the frozen exact Metropolis-MC reference.
* Every number below is the median over seeds of the per-seed relative change (arm / base − 1). Brackets give the
  95 % percentile bootstrap CI (10 000 resamples), and the count is the seeds where the arm is better.
* **Negative = the FR schedule has the lower error.**
* t = 20 is both the end of the ABF warm-up and the **reset of the reported estimator**. Before it the reported
  estimate is the all-steps (warm-up) estimator. After it, only production accumulators with deposits from t ≥ 20
  are reported.
* Under ABF, **72 % of the total I_F^{[0,240]} lies in [0, 20]** (median ABF window integrals: I_F[0,20] 10.0,
  I_F[20,80] 2.17, I_F[40,240] 2.61, I_F[0,240] 14.1).

## Primary contrasts (preregistered; Bonferroni over two, decided by the 97.5 % CI)

| id | contrast | metric | change [95 % CI] | 97.5 % CI | arm better | verdict |
|---|---|---|---|---|---|---|
| **P1** | FR from t = 0 vs FR from t = 40 | I_F^{[0,40]} | **−7.5 % [−13.5, −4.0]** | [−14.4, −4.0] | 15/16 | **SUPPORTED** |
| **P2** | FR on [0, 20) then ABF vs FR from t = 0 continuous | e_F(240) | **−61.8 % [−68.5, −54.3]** | [−69.3, −52.2] | 16/16 | **SUPPORTED** |

* FR from t = 40 equals ABF bitwise before t = 40. Its only difference on [0, 40] is the first FR event at the t = 40
  checkpoint. P1 is therefore effectively early FR against ABF on [0, 40].

## Every schedule vs ABF: required windows and final error

| schedule vs ABF | I_F[0,40] | I_F[20,80] | I_F[0,240] | e_F(240) |
|---|---|---|---|---|
| FR from t = 0, continuous | -7.5 % [-13.5, -4.0] 15/16 | -3.9 % [-26.3, +4.2] 11/16 | -1.2 % [-7.6, +2.5] 11/16 | +152.9 % [+69.0, +229.6] 1/16 |
| FR from t = 5, continuous | -1.7 % [-4.0, +0.5] 10/16 | +10.9 % [-9.6, +30.7] 7/16 | +14.2 % [+1.7, +23.0] 3/16 | +199.2 % [+119.6, +308.5] 1/16 |
| FR from t = 20, continuous | -0.3 % [-2.1, +1.7] 8/16 | +6.3 % [-10.3, +42.2] 7/16 | +11.8 % [+1.0, +21.4] 4/16 | +160.9 % [+89.8, +243.5] 0/16 |
| FR from t = 40, continuous (validated schedule) | +0.0 % [+0.0, +0.0] 1/16 | -1.5 % [-10.6, +11.7] 8/16 | +8.9 % [+1.4, +16.5] 3/16 | +172.3 % [+124.8, +230.9] 1/16 |
| FR on 0 ≤ t < 20, then ABF | -6.3 % [-9.1, -4.6] 15/16 | -11.8 % [-29.2, +7.4] 11/16 | -8.1 % [-12.6, -4.4] 15/16 | -6.0 % [-22.3, +10.3] 10/16 |
| FR on 0 ≤ t < 40, then ABF | -6.5 % [-9.0, -1.9] 13/16 | +0.2 % [-19.4, +31.3] 8/16 | -0.5 % [-7.8, +2.4] 9/16 | +18.9 % [-16.9, +132.8] 7/16 |

### Preregistered verdicts (vs ABF)

* **Early-convergence benefit**, defined as a 95 % CI of G on I_F^{[20,80]} entirely below 0: **NOT SUPPORTED for
  any schedule.**
* **Safe early-stopping benefit** (early-convergence benefit and final-error non-inferiority at +5 %):
  * **NOT SUPPORTED** for FR on [0, 20) and for FR on [0, 40);
  * both have an INCONCLUSIVE final-accuracy conclusion, since their CIs straddle +5 %.

### The other windows

| schedule vs ABF | I_F[0,20] | I_F[20,40] | I_F(20,80] (sensitivity) | I_F[40,240] | e_F′(240) |
|---|---|---|---|---|---|
| FR from t = 0, continuous | -7.0 % [-9.7, -5.4] 15/16 | -12.1 % [-22.1, +8.3] 10/16 | -3.2 % [-33.2, +32.2] 9/16 | +30.3 % [+0.2, +62.4] 4/16 | +7.3 % [+4.2, +10.6] 2/16 |
| FR from t = 5, continuous | -1.3 % [-3.4, -0.3] 14/16 | -6.0 % [-26.8, +20.7] 9/16 | +17.5 % [-13.4, +51.3] 6/16 | +75.4 % [+25.9, +125.1] 4/16 | +11.0 % [+7.0, +14.8] 1/16 |
| FR from t = 20, continuous | +0.0 % [+0.0, +0.0] 0/16 | -2.5 % [-17.0, +17.9] 8/16 | +9.5 % [-13.0, +53.0] 7/16 | +67.1 % [+11.0, +113.9] 2/16 | +11.2 % [+4.5, +16.2] 2/16 |
| FR from t = 40, continuous (validated schedule) | +0.0 % [+0.0, +0.0] 0/16 | +0.0 % [+0.0, +0.0] 1/16 | -2.8 % [-12.0, +15.2] 8/16 | +56.2 % [+9.5, +96.5] 3/16 | +10.7 % [+7.0, +14.0] 2/16 |
| FR on 0 ≤ t < 20, then ABF | -6.1 % [-8.9, -3.9] 14/16 | -14.1 % [-31.0, +16.8] 12/16 | -12.8 % [-38.2, +20.7] 11/16 | -17.1 % [-31.5, +3.2] 12/16 | -1.6 % [-4.4, +2.0] 10/16 |
| FR on 0 ≤ t < 40, then ABF | -5.7 % [-9.4, -1.3] 12/16 | -2.1 % [-16.1, +20.1] 8/16 | +1.6 % [-24.1, +39.4] 8/16 | +13.3 % [-13.2, +47.9] 6/16 | -0.0 % [-3.0, +5.3] 8/16 |

* The sensitivity window (20, 80], which excludes the t = 20 reset save, gives the same conclusions as [20, 80].

## Direct contrasts between schedules

| direct contrast | metric | change (median [95 % CI], arm better) |
|---|---|---|
| `fr_start0_stop20` vs `fr_start0` | e_F(240) | -61.8 % [-68.5, -54.3] 16/16 |
| `fr_start0_stop20` vs `fr_start0` | I_F[40,240] | -35.3 % [-45.2, -24.1] 15/16 |
| `fr_start0_stop20` vs `fr_start0` | I_F[0,240] | -4.6 % [-10.3, -2.2] 14/16 |
| `fr_start0_stop40` vs `fr_start0` | e_F(240) | -30.4 % [-61.7, -16.4] 14/16 |
| `fr_start0_stop20` vs `fr_start0_stop40` | e_F(240) | -23.2 % [-49.7, -2.8] 13/16 |
| `fr_start0` vs `fr_start40` | I_F[0,40] | -7.5 % [-13.5, -4.0] 15/16 |
| `fr_start0` vs `fr_start40` | I_F[20,80] | -8.2 % [-32.2, +17.7] 8/16 |
| `fr_start0` vs `fr_start40` | I_F[0,240] | -11.2 % [-15.0, -1.5] 12/16 |
| `fr_start0` vs `fr_start40` | e_F(240) | +0.1 % [-37.5, +18.3] 8/16 |
| `fr_start5` vs `fr_start40` | I_F[0,240] | +4.6 % [-6.2, +10.4] 6/16 |
| `fr_start20` vs `fr_start40` | I_F[0,240] | +5.4 % [-8.7, +13.1] 6/16 |
| `fr_start0_stop20` vs `fr_start40` | I_F[0,240] | -13.6 % [-24.2, -10.8] 15/16 |
| `fr_start0_stop20` vs `fr_start40` | e_F(240) | -62.5 % [-72.7, -47.8] 16/16 |
| `fr_start0_stop40` vs `fr_start40` | e_F(240) | -53.5 % [-65.0, -12.2] 13/16 |

## Time to accuracy

The table gives the per-seed restricted sustained speed-up S = min(τ_ABF, T) / min(τ_FR, T), as a geometric median
[95 % CI] with the number of FR seeds censored at T. ABF is censored for 0 seeds at every tolerance. ABF median τ:
31.5, 37, 55.5, 68 and 112.5 t.u.

| schedule | ε = 0.05 | 0.04 | 0.03 | 0.02 | 0.015 kT |
|---|---|---|---|---|---|
| FR from t = 0 | 1.07 [0.83, 1.29] (0 cens.) | 1.19 [0.83, 1.50] (0 cens.) | 1.18 [0.81, 1.75] (1 cens.) | 0.54 [0.35, 0.91] (5 cens.) | 0.47 [0.38, 0.75] (12 cens.) |
| FR from t = 5 | 0.98 [0.76, 1.28] (0 cens.) | 0.82 [0.67, 1.27] (0 cens.) | 0.69 [0.32, 1.33] (0 cens.) | 0.33 [0.21, 0.65] (13 cens.) | 0.56 [0.42, 0.73] (14 cens.) |
| FR from t = 20 | 1.00 [0.70, 1.11] (0 cens.) | 1.02 [0.56, 1.38] (0 cens.) | 0.79 [0.53, 1.83] (2 cens.) | 0.49 [0.26, 0.84] (7 cens.) | 0.47 [0.38, 0.71] (15 cens.) |
| FR from t = 40 | 1.00 [1.00, 1.00] (0 cens.) | 1.00 [1.00, 1.00] (0 cens.) | 1.00 [0.50, 1.44] (5 cens.) | 0.55 [0.25, 1.00] (8 cens.) | 0.66 [0.42, 0.76] (11 cens.) |
| FR on [0, 20) | 1.22 [0.93, 1.35] (0 cens.) | 1.28 [1.00, 1.64] (0 cens.) | 1.20 [0.89, 1.59] (0 cens.) | 1.20 [0.85, 1.59] (0 cens.) | 1.09 [0.81, 1.67] (0 cens.) |
| FR on [0, 40) | 1.01 [0.75, 1.25] (0 cens.) | 0.93 [0.80, 1.23] (0 cens.) | 0.99 [0.74, 1.47] (0 cens.) | 0.57 [0.45, 0.84] (0 cens.) | 0.75 [0.50, 1.21] (3 cens.) |

* Continuous FR (from any start) reaches the loose tolerances at about ABF's time, with CIs including 1.
* It fails to reach 0.02 to 0.015 kT in 5–15 of 16 seeds, because of its late plateau near 0.018 kT.
* FR on [0, 20) reaches every tolerance in every seed, with S 1.09–1.28. Every CI includes 1 or touches it, so the
  speed-up is not resolved.

## Marginal uniformity vs accuracy (H3)

| schedule vs ABF | mean e_ξ full [0,40] | mean e_ξ full [20,80] | mean e_ξ interior [20,80] | mean edge fraction [20,80] | I_F′[20,80] |
|---|---|---|---|---|---|
| FR from t = 0, continuous | -3.6 % [-5.6, -2.0] 14/16 | -5.8 % [-7.0, -4.5] 15/16 | -1.3 % [-1.7, -0.3] 13/16 | -4.3 % [-5.9, -3.0] 15/16 | -3.0 % [-5.9, +3.3] 9/16 |
| FR from t = 5, continuous | -3.8 % [-4.4, -2.0] 14/16 | -7.1 % [-8.0, -3.4] 15/16 | -1.0 % [-1.3, -0.2] 14/16 | -5.8 % [-6.7, -3.0] 13/16 | -0.2 % [-5.8, +3.7] 8/16 |
| FR from t = 20, continuous | -2.5 % [-4.5, -1.4] 15/16 | -4.1 % [-7.2, -3.6] 16/16 | -1.4 % [-1.7, -0.5] 13/16 | -3.3 % [-6.9, -2.4] 15/16 | +0.9 % [-3.5, +6.6] 8/16 |
| FR from t = 40, continuous (validated schedule) | +0.0 % [+0.0, +0.0] 1/16 | -4.4 % [-5.1, -3.3] 16/16 | -0.8 % [-1.2, -0.5] 14/16 | -4.1 % [-4.6, -3.1] 15/16 | +1.3 % [-0.2, +2.2] 6/16 |
| FR on 0 ≤ t < 20, then ABF | -1.8 % [-2.8, -0.3] 13/16 | +0.1 % [-4.7, +1.3] 8/16 | -0.5 % [-0.9, +0.0] 11/16 | -0.6 % [-3.4, +1.7] 10/16 | -2.5 % [-8.2, +7.8] 10/16 |
| FR on 0 ≤ t < 40, then ABF | -4.0 % [-5.7, -2.4] 15/16 | -2.8 % [-4.3, +0.9] 10/16 | -0.4 % [-0.9, +0.4] 10/16 | -1.0 % [-4.7, +0.6] 8/16 | -1.3 % [-4.2, +7.2] 10/16 |

## Genealogy and numerical safety

| arm | replacements total | on [0,20) | on [20,40) | on [40,240] | min windowed ESS/K (median; worst seed) | max family share (median; worst) | unique ancestors at T | ABF bias-clip fraction |
|---|---|---|---|---|---|---|---|---|
| `abf` | 0 | 0 | 0 | 0 | —; — | —; — | 256 | 0.0485 |
| `fr_start0` | 818 | 92 | 64 | 664 | 0.70; 0.62 | 0.082; 0.164 | 58 | 0.0501 |
| `fr_start5` | 800 | 56 | 70 | 672 | 0.78; 0.75 | 0.078; 0.109 | 60 | 0.0496 |
| `fr_start20` | 750 | 0 | 66 | 679 | 0.78; 0.72 | 0.080; 0.141 | 62 | 0.0493 |
| `fr_start40` | 662 | 0 | 0 | 662 | 0.80; 0.76 | 0.064; 0.094 | 69 | 0.0493 |
| `fr_start0_stop20` | 86 | 86 | 0 | 0 | 0.70; 0.66 | 0.023; 0.031 | 188 | 0.0491 |
| `fr_start0_stop40` | 158 | 88 | 72 | 0 | 0.73; 0.62 | 0.027; 0.051 | 154 | 0.0489 |

* There were no non-finite states and no failed runs.
* The ABF bias-clip fraction (|bin force| > 40) is about 0.049 in every arm, the same as Phase V.
* The minimum windowed ESS/K is at least 0.61 in every seed, well above the repository's 0.30 floor.
* Continuous FR concentrates the run-long genealogy: about 60 of 256 founders survive, and the largest family holds
  up to 16 % of the population.

## What the data establish (numerical observations)

1. **Starting FR at t = 0 rather than t = 40 lowers I_F^{[0,40]}** (P1, −7.5 %).
   * The gain sits in the warm-up window [0, 20]: −7.0 % [−9.7, −5.4], 15/16.
   * It is **not resolved after the reset**: [20, 40] −12.1 % [−22.1, +8.3]; [20, 80] −3.9 % [−26.3, +4.2].
2. **Stopping FR removes the late penalty of continuous FR.**
   * Final error vs continuous FR from t = 0: −61.8 % with the stop at 20 (P2); −30.4 % [−61.7, −16.4] with the
     stop at 40.
   * Stopping at 20 beats stopping at 40 by −23.2 % [−49.7, −2.8].
3. **No schedule meets the preregistered early-convergence criterion against ABF.**
   * FR on [0, 20) has the only resolved total-integrated gain over ABF: I_F^{[0,240]} −8.1 % [−12.6, −4.4], 15/16.
     This is a secondary, descriptive comparison, and the gain comes mostly from the warm-up window.
   * Its final error is ABF-like: −6.0 % [−22.3, +10.3], inconclusive against the +5 % margin.
4. **Continuous FR is harmful at the end from every start.**
   * e_F(240) is +153 % to +199 % above ABF, and e_F′(240) is +7 % to +11 %.
   * The validated schedule (start 40) replicates Phase V on fresh seeds: I_F +8.9 % [+1.4, +16.5]; e_F(240)
     +172 % [+125, +231].
5. **Marginal gains do not carry over to accuracy (H3).**
   * Continuous FR lowers the full-domain marginal error over [20, 80] by 4–7 %, mostly by holding fewer replicas in
     the soft-wall edge bins (edge fraction −3 to −6 %). The interior marginal improves by only about 1 %.
   * Over the same arms, the free-energy and mean-force errors end worse.
6. **The outcome depends on the schedule at fixed physics and rate (H4).**
   * Ordering of I_F^{[0,240]} against ABF: FR on [0, 20) (−8.1 %); then FR from 0, FR on [0, 40) and ABF, which
     are indistinguishable; then starts at 40, 20 and 5 (+9 % to +14 %).

## Mechanistic hypotheses (not tested here)

* **Late penalty.** The plateau of continuous FR near 0.018 kT matches the stationary tilt of score-directed FR
  measured in Phase V (0.017 kT). The Phase-V "fibre-lag" reading (cloning walkers whose solvent has not relaxed)
  was not tested here either.
* **Why the stop at 20 works best.** The FR stop coincides with the estimator reset, so the reported production
  accumulators never receive deposits from an FR-active period. With the stop at 40, the production accumulators
  include FR-period deposits from [20, 40), and the final error is +18.9 % [−16.9, +132.8] against ABF, not
  resolved. This suggests that FR-period deposits carry the bias; it is a hypothesis.
* **Early gain.** The early benefit is consistent with FR moving the population over the barrier while the ABF bias
  is still ramping. That gain is realised in the warm-up estimate, which the reset discards.
