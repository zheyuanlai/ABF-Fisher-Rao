# Convergence and time-to-accuracy atlas (Experiment 1)

*2026-10-17. Descriptive analysis of existing paired runs. Each system compares ABF and ABF + FR only; LTA's
historical third arm is never read.*

* Code: `scripts/convergence_fr/atlas.py`, `atlas_lib.py` (definitions) and `systems.py` (loaders).
* Figures: `fig_atlas.py`. Tables: `atlas_tables.py`.
* Data: `results/convergence_fr_campaign/atlas/atlas_summary.json`. The tables below are generated from it.
* Figures: `figures/convergence_fr/{convergence_eF, convergence_mean_force, running_integrated_error,
  time_to_accuracy, speedup_vs_tolerance, early_population_establishment}.{pdf,png}`.

## Systems, references and statistical conventions

| system | model | runs | reference |
|---|---|---|---|
| LTA 150 K, 300 K | rigid all-silica LTA, 2-bead UA ethane, EM dt 2e-4, K 1024, T 60, 180 bins, FR rate 0.2 from 4 t.u. | `results/lta_histogram/production_T{150,300}` (2026-10-03/04, 16 paired seeds) | exact Metropolis-MC umbrella/WHAM (`results/numerical_validation/lta/T*/`, the profile used by the validated rescoring) |
| ITE 300 K (X1) | rigid Si-ITE, Dubbeldam-2004 UA ethane, EM dt 1e-4, K 1024, T 50, 180 bins, rate 2.0 from 1 t.u. | `results/overnight/ite/prod_confirm` (exploratory, 16 paired seeds) | exact MC F_MF at 180 bins |
| ITE 150 K (X2) | same model, K 256, T 200, **360 bins**, rate 0.5 from 1 t.u. | `results/overnight/ite150/prod_confirm` (exploratory, 16 paired seeds) | as run: smooth MC reference at 360 bins; sensitivity: accumulators summed to 180 bins vs the 180-bin MC mean force |

* **Arithmetic check.** The atlas reproduces every published contrast exactly:
  * LTA rescoring vs exact MC: −28.5 % / −54.4 % at 150 K and −13.7 % / −7.9 % at 300 K;
  * X1: −14.9 % / +14.0 %;
  * X2: −1.1 % / +56.8 %.
* **Errors.** e_F is the aligned RMS over the circle (L² / √2π) after removing the additive constant. e_F′ is the
  RMS of the bin mean force.
* **Bands.**
  * Light: 25–75 % of seeds.
  * Dark: the 95 % percentile bootstrap CI of the per-save median (2000 resamples of seeds).
* **Paired statistics.** Per-seed relative change B/A − 1, then the median over seeds, with a 95 % bootstrap CI
  (10 000 resamples) and the number of seeds where FR is better.
* **Time to accuracy (sustained).** The first save after which e_F ≤ ε at every later save up to T.
  * The speedup S = τ_ABF/τ_FR is computed per seed, censored at T (both censored → 1). The median over seeds is
    reported.
  * A "lower bound" note means ABF is censored in > 25 % of seeds, so S understates FR's advantage. An "upper
    bound" note means the same for FR.
* **Tolerance ladder.** {0.080, 0.064, 0.048, 0.040, 0.032} k_BT, i.e. 0.20 / 0.16 / 0.12 / 0.10 / 0.08 kJ/mol at
  300 K. It was written into the script before its first run.
* **Status.** All of this is **descriptive**. X1 and X2 are exploratory blocks, and LTA is historical.

## 1. Integrated and final errors

| system | K | T | bins | rate | Ī_F full | Ī_F post-ramp | e_F(T) | e_F′(T) |
|---|---|---|---|---|---|---|---|---|
| LTA 150 K | 1024 | 60 | 180 | 0.2 | -28.5 % [-29.4, -26.6] (16/16) | -46.1 % [-51.2, -44.4] (16/16) | -54.4 % [-60.5, -46.1] (16/16) | -42.6 % [-49.7, -35.8] (16/16) |
| LTA 300 K | 1024 | 60 | 180 | 0.2 | -13.7 % [-14.8, -10.9] (16/16) | -22.7 % [-24.5, -19.0] (16/16) | -7.9 % [-12.8, +0.7] (11/16) | -5.4 % [-12.0, +4.7] (11/16) |
| ITE 300 K | 1024 | 50 | 180 | 2 | -14.9 % [-17.0, -10.1] (16/16) | -22.3 % [-24.2, -15.1] (16/16) | +14.0 % [+0.8, +21.3] (4/16) | +11.0 % [+0.0, +28.2] (4/16) |
| ITE 150 K | 256 | 200 | 360 | 0.5 | -1.1 % [-8.3, +11.9] (9/16) | -1.4 % [-11.6, +15.2] (9/16) | +56.8 % [+29.3, +95.0] (0/16) | +34.1 % [+22.7, +70.1] (1/16) |
| ITE 150 K (180-bin read-out) | 256 | 200 | 180 | 0.5 | -1.1 % [-8.5, +12.3] (9/16) | -1.5 % [-11.8, +15.7] (9/16) | +56.8 % [+28.5, +93.3] (0/16) | +36.1 % [+23.6, +71.8] (1/16) |
| ITE 150 K, first 50 t.u. | 256 | 50 | 360 | 0.5 | -12.7 % [-14.3, -8.3] (14/16) | -18.0 % [-21.3, -12.1] (14/16) | +10.5 % [-12.1, +24.2] (7/16) | +5.2 % [-12.0, +20.8] (8/16) |
| ITE 150 K, first 50 t.u. (180-bin read-out) | 256 | 50 | 180 | 0.5 | -12.8 % [-14.4, -8.3] (14/16) | -17.9 % [-21.5, -12.1] (14/16) | +10.6 % [-12.0, +24.2] (7/16) | +5.6 % [-11.9, +21.1] (8/16) |

Absolute medians (ABF → FR), kJ/mol: 
* LTA 150 K: Ī_F 0.460 → 0.337; post-ramp 0.288 → 0.157; e_F(T) 0.1293 → 0.0621
* LTA 300 K: Ī_F 0.394 → 0.339; post-ramp 0.249 → 0.191; e_F(T) 0.0937 → 0.0763
* ITE 300 K: Ī_F 0.317 → 0.288; post-ramp 0.224 → 0.184; e_F(T) 0.0764 → 0.0865
* ITE 150 K: Ī_F 0.182 → 0.195; post-ramp 0.145 → 0.159; e_F(T) 0.0477 → 0.0802
* ITE 150 K (180-bin read-out): Ī_F 0.183 → 0.195; post-ramp 0.146 → 0.160; e_F(T) 0.0491 → 0.0815
* ITE 150 K, first 50 t.u.: Ī_F 0.534 → 0.464; post-ramp 0.385 → 0.318; e_F(T) 0.1424 → 0.1532
* ITE 150 K, first 50 t.u. (180-bin read-out): Ī_F 0.532 → 0.461; post-ramp 0.385 → 0.319; e_F(T) 0.1437 → 0.1545

## 2. Time to accuracy (sustained criterion; first-hit table in `atlas_tables.md`)

— S median [95 % CI]; reached A/B of n; FR faster / ABF faster; median τ ABF → FR (t.u.; '–' if ≤ half reach)

| system | ε (kJ/mol) | ε/k_BT | S | reached ABF / FR | FR faster / ABF faster | τ ABF → FR | note |
|---|---|---|---|---|---|---|---|
| LTA 150 K | 0.100 | 0.080 | 1.84 [1.54, 2.00] | 0 / 16 of 16 | 16 / 0 | – → 32.7 | lower bound (ABF censored in > 25 % of seeds) |
| LTA 150 K | 0.080 | 0.064 | 1.43 [1.20, 1.67] | 0 / 13 of 16 | 13 / 0 | – → 42.0 | lower bound (ABF censored in > 25 % of seeds) |
| LTA 150 K | 0.060 | 0.048 | 1.00 [1.00, 1.09] | 0 / 6 of 16 | 6 / 0 | – → – | both arms heavily censored |
| LTA 150 K | 0.050 | 0.040 | 1.00 [1.00, 1.00] | 0 / 2 of 16 | 2 / 0 | – → – | both arms heavily censored |
| LTA 150 K | 0.040 | 0.032 | 1.00 [1.00, 1.00] | 0 / 2 of 16 | 2 / 0 | – → – | both arms heavily censored |
| LTA 300 K | 0.200 | 0.080 | 1.39 [1.31, 1.53] | 16 / 16 of 16 | 16 / 0 | 27.6 → 19.2 | estimable |
| LTA 300 K | 0.160 | 0.064 | 1.33 [1.22, 1.37] | 16 / 16 of 16 | 16 / 0 | 32.7 → 24.9 | estimable |
| LTA 300 K | 0.120 | 0.048 | 1.17 [1.15, 1.29] | 14 / 16 of 16 | 15 / 1 | 45.6 → 36.6 | estimable |
| LTA 300 K | 0.100 | 0.040 | 1.06 [1.00, 1.18] | 9 / 11 of 16 | 10 / 1 | 56.7 → 45.3 | both arms heavily censored |
| LTA 300 K | 0.080 | 0.032 | 1.01 [1.00, 1.10] | 5 / 10 of 16 | 8 / 1 | – → 58.5 | both arms heavily censored |
| ITE 300 K | 0.200 | 0.080 | 1.29 [1.15, 1.60] | 16 / 16 of 16 | 14 / 1 | 16.6 → 13.6 | estimable |
| ITE 300 K | 0.160 | 0.064 | 1.07 [0.99, 1.23] | 16 / 16 of 16 | 11 / 5 | 20.9 → 20.2 | estimable |
| ITE 300 K | 0.120 | 0.048 | 0.96 [0.91, 1.09] | 15 / 15 of 16 | 6 / 9 | 28.9 → 30.1 | estimable |
| ITE 300 K | 0.100 | 0.040 | 0.90 [0.81, 1.00] | 15 / 13 of 16 | 4 / 11 | 35.9 → 40.3 | estimable |
| ITE 300 K | 0.080 | 0.032 | 1.00 [0.94, 1.00] | 9 / 6 of 16 | 3 / 7 | 48.4 → – | both arms heavily censored |
| ITE 150 K | 0.100 | 0.080 | 0.84 [0.57, 0.97] | 16 / 13 of 16 | 3 / 12 | 75.5 → 121.5 | estimable |
| ITE 150 K | 0.080 | 0.064 | 0.65 [0.47, 0.88] | 16 / 8 of 16 | 2 / 14 | 94.8 → – | upper bound (FR censored in > 25 % of seeds) |
| ITE 150 K | 0.060 | 0.048 | 0.82 [0.65, 0.92] | 14 / 4 of 16 | 0 / 14 | 145.5 → – | upper bound (FR censored in > 25 % of seeds) |
| ITE 150 K | 0.050 | 0.040 | 0.97 [0.73, 1.00] | 10 / 2 of 16 | 0 / 10 | 194.5 → – | both arms heavily censored |
| ITE 150 K | 0.040 | 0.032 | 1.00 [0.89, 1.00] | 4 / 0 of 16 | 0 / 4 | – → – | both arms heavily censored |
| ITE 150 K (180-bin read-out) | 0.100 | 0.080 | 0.82 [0.57, 0.97] | 16 / 13 of 16 | 4 / 12 | 76.5 → 124.2 | estimable |
| ITE 150 K (180-bin read-out) | 0.080 | 0.064 | 0.65 [0.47, 0.88] | 15 / 7 of 16 | 2 / 13 | 96.8 → – | upper bound (FR censored in > 25 % of seeds) |
| ITE 150 K (180-bin read-out) | 0.060 | 0.048 | 0.82 [0.65, 0.99] | 14 / 4 of 16 | 2 / 12 | 146.0 → – | upper bound (FR censored in > 25 % of seeds) |
| ITE 150 K (180-bin read-out) | 0.050 | 0.040 | 0.98 [0.73, 1.00] | 9 / 2 of 16 | 0 / 9 | 195.5 → – | both arms heavily censored |
| ITE 150 K (180-bin read-out) | 0.040 | 0.032 | 1.00 [0.89, 1.00] | 4 / 0 of 16 | 0 / 4 | – → – | both arms heavily censored |
| ITE 150 K, first 50 t.u. | 0.100 | 0.080 | 1.00 [1.00, 1.00] | 4 / 2 of 16 | 2 / 3 | – → – | both arms heavily censored |
| ITE 150 K, first 50 t.u. | 0.080 | 0.064 | 1.00 [1.00, 1.00] | 0 / 1 of 16 | 1 / 0 | – → – | both arms heavily censored |
| ITE 150 K, first 50 t.u. | 0.060 | 0.048 | 1.00 [1.00, 1.00] | 0 / 0 of 16 | 0 / 0 | – → – | both arms heavily censored |
| ITE 150 K, first 50 t.u. | 0.050 | 0.040 | 1.00 [1.00, 1.00] | 0 / 0 of 16 | 0 / 0 | – → – | both arms heavily censored |
| ITE 150 K, first 50 t.u. | 0.040 | 0.032 | 1.00 [1.00, 1.00] | 0 / 0 of 16 | 0 / 0 | – → – | both arms heavily censored |
| ITE 150 K, first 50 t.u. (180-bin read-out) | 0.100 | 0.080 | 1.00 [1.00, 1.00] | 3 / 2 of 16 | 2 / 2 | – → – | both arms heavily censored |
| ITE 150 K, first 50 t.u. (180-bin read-out) | 0.080 | 0.064 | 1.00 [1.00, 1.00] | 0 / 1 of 16 | 1 / 0 | – → – | both arms heavily censored |
| ITE 150 K, first 50 t.u. (180-bin read-out) | 0.060 | 0.048 | 1.00 [1.00, 1.00] | 0 / 0 of 16 | 0 / 0 | – → – | both arms heavily censored |
| ITE 150 K, first 50 t.u. (180-bin read-out) | 0.050 | 0.040 | 1.00 [1.00, 1.00] | 0 / 0 of 16 | 0 / 0 | – → – | both arms heavily censored |
| ITE 150 K, first 50 t.u. (180-bin read-out) | 0.040 | 0.032 | 1.00 [1.00, 1.00] | 0 / 0 of 16 | 0 / 0 | – → – | both arms heavily censored |

## 3. Speedup as a function of tolerance (`speedup_vs_tolerance.pdf`, 41 tolerances from 0.016 to 0.32 k_BT)

| system | largest median sustained speedup | where the 95 % CI is > 1 | where it is < 1 |
|---|---|---|---|
| LTA 150 K | 2.48 [2.27, 2.61] at 0.15 k_BT (0.19 kJ/mol) | 0.057–0.32 k_BT; below 0.12 k_BT ABF is censored, so S there is a lower bound | nowhere |
| LTA 300 K | 1.49 [1.42, 1.56] at 0.14 k_BT (0.35 kJ/mol) | 0.046–0.28 k_BT | nowhere |
| ITE 300 K (X1) | 2.21 [1.94, 2.32] at 0.24 k_BT (0.59 kJ/mol) | 0.066–0.32 k_BT | 0.039 k_BT (≈ 0.10 kJ/mol) |
| ITE 150 K (X2, T 200) | 1.83 [1.50, 2.00] at 0.32 k_BT (0.40 kJ/mol) | 0.18–0.32 k_BT | 0.046–0.090 k_BT |

* **Reading.** In every system FR reaches *loose* accuracy sooner, by up to about 2× in time.
* **The direct time-to-accuracy calculation supports a ≥ 2× acceleration only:**
  * ITE 300 K at 0.15–0.32 k_BT (0.4–0.8 kJ/mol);
  * LTA 150 K at 0.1–0.3 k_BT.
* **At tight tolerances the systems split.**
  * LTA: FR is faster or tied everywhere.
  * ITE 300 K: tied, with one rung (≈ 0.10 kJ/mol) where FR is slower (S 0.90 [0.81, 1.00]; ABF faster in 11/16
    seeds).
  * ITE 150 K over T 200: FR is slower (S 0.65–0.84), because its error plateaus.

## 4. Crossing and early window establishment

**Crossing (FR permanently worse than ABF after this save) and window establishment**

| system | seeds crossing | per-seed median | median-curve crossing | bootstrap: fraction crossing; 2.5/50/97.5 % | t_est ABF | t_est FR | paired Δ t_est |
|---|---|---|---|---|---|---|---|
| LTA 150 K | 0/16 | – | none | 0.00; – | 16.2 | 9.0 | -7.2 [-7.8, -6.6] |
| LTA 300 K | 5/16 | – | none | 0.04; 52.8/57.6/60.0 | 9.3 | 7.5 | -1.8 [-2.1, -1.5] |
| ITE 300 K | 12/16 | 30.9 | 29.1 | 0.99; 19.9/29.1/46.8 | 4.3 | 2.4 | -1.8 [-2.1, -1.6] |
| ITE 150 K | 16/16 | 65.8 | 38.5 | 1.00; 32.5/38.5/103.5 | 7.5 | 4.5 | -3.0 [-3.8, -2.5] |
| ITE 150 K (180-bin read-out) | 16/16 | 65.8 | 38.5 | 1.00; 32.5/38.5/103.5 | 7.5 | 4.5 | -3.0 [-3.8, -2.5] |
| ITE 150 K, first 50 t.u. | 9/16 | 38.2 | 38.5 | 0.74; 32.5/37.5/47.0 | 7.5 | 4.5 | -3.0 [-3.8, -2.5] |
| ITE 150 K, first 50 t.u. (180-bin read-out) | 9/16 | 38.2 | 38.5 | 0.74; 32.5/37.5/49.5 | 7.5 | 4.5 | -3.0 [-3.8, -2.5] |

* **Establishment rule.** The first save at which the trailing 1-t.u. mean window occupancy reaches 0.75 × the
  flat-bias window share (`early_population_establishment.pdf`).
* **FR establishes the window sooner in every system:**
  * by 1.8 t.u. in ITE 300 K and LTA 300 K;
  * by 3.0 t.u. in ITE 150 K;
  * by 7.2 t.u. in LTA 150 K.
  * The largest establishment gain (LTA 150 K) goes with the largest and only permanent error gain.
* **ABF's window occupancy overshoots the flat-bias share in ITE.** Mean of the per-save median, ABF vs FR, share
  0.159:

  | window (t.u.) | ITE 300 K | ITE 150 K |
  |---|---|---|
  | 5–20 | 0.176 vs 0.160 | 0.160 vs 0.162 |
  | 20–30 | 0.169 vs 0.159 | 0.195 vs 0.168 |
  | 30–50 | 0.164 vs 0.159 | 0.182 vs 0.162 |
  | 50–100 | – | 0.168 vs 0.161 |

  * FR holds the occupancy at the share.
  * ABF's bias over-flattens the window for tens of t.u. after establishment. The overshoot is largest at 150 K.
  * The LTA overshoot is smaller: 0.26 vs 0.25 at 300 K, and 0.276 vs 0.254 only after 40 t.u. at 150 K.
  * The atlas cannot tell whether this matters for e_F.
* **Crossing.**
  * ITE 300 K: the median paired e_F curve crosses permanently at 29 t.u. [19.9, 46.8] (12/16 seeds cross).
  * ITE 150 K: 38.5 t.u. [32.5, 103.5] (16/16 seeds cross by T = 200).
  * LTA: no crossing at 150 K; 5/16 seeds cross only in the last 7 t.u. at 300 K.

## 5. The ITE 150 K "null" is a horizon effect (new observation)

| | ITE 300 K (X1, T 50) | ITE 150 K, first 50 t.u. of X2 | ITE 150 K, X2 full T 200 |
|---|---|---|---|
| ΔĪ_F full | −14.9 % [−17.0, −10.1], 16/16 | **−12.7 % [−14.3, −8.3], 14/16** | −1.1 % [−8.3, +11.9], 9/16 |
| ΔĪ_F post-ramp | −22.3 % | −18.0 % | −1.4 % |
| Δe_F(T) | +14.0 % [+0.8, +21.3] | +10.5 % [−12.1, +24.2] | +56.8 % [+29.3, +95.0] |

* **Truncation is exact.** The engine's dynamics never depend on n_steps, so the first 50 t.u. of a T-200 run *is* a
  T-50 run.
* **What is observed.** Over a common 50 t.u. window, the 150 K cell shows an integrated FR gain about as large as
  300 K's, despite the smaller K (256 vs 1024) and the 4× lower FR rate. The "−1.1 %" of X2 is FR's early gain
  averaged against its growing late deficit over 150 more t.u.
* **Bins do not matter.** 360 vs a 180-bin read-out changes nothing (< 0.2 points), so the 360-bin choice is
  immaterial to these conclusions.
* **What is not yet established.** This is exploratory, and K and the rate still differ between the cells. The
  controlled matrix (`ITE_RESULTS.md`) tests it.

## 6. What the atlas does and does not show

* **Observed in all four systems:**
  * FR establishes the window sooner;
  * FR reduces the post-ramp integrated error (−18 % to −46 % over the first 50–60 t.u.);
  * FR reaches loose tolerances sooner.
* **Observed in ITE only:** FR's error curve falls behind ABF's after about 20–40 t.u. and stays behind. In LTA it
  does not (150 K), or only marginally at the very end (300 K).
* **Not shown.**
  * That FR improves *final* accuracy anywhere except LTA 150 K.
  * Any equal-cost advantage. Every comparison here is at fixed K and equal force evaluations.
* **Hypothesis motivating E1, not established here.** Ongoing FR after establishment is what makes the ITE curves
  cross. Stopping FR after establishment should keep the early gain without the late deficit.
