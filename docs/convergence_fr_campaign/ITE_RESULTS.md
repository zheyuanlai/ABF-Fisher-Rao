# ITE — controlled temperature × replica-count study (Experiment 2) and context

*Frozen in `PREREGISTRATION.md` §5 and A2. Run 2026-10-17. Frozen analysis `analyze_block.py --ladder abf`.
Figures: `figures/convergence_fr/block_M300_256.png` and `block_M150_256.png`.
Atlas context: `CONVERGENCE_ATLAS.md` §5.*

## 1. Why the old comparison was confounded

| | X1 (300 K) | X2 (150 K) |
|---|---|---|
| K | 1024 | 256 |
| horizon T | 50 | 200 |
| bins | 180 | 360 |
| FR rate | 2.0 | 0.5 |
| ΔĪ_F / Δe_F(T) | −14.9 % / +14.0 % | −1.1 % / +56.8 % |

* **Horizon (atlas, exact truncation).** X2's first 50 t.u. give ΔĪ_F −12.7 % [−14.3, −8.3] (14/16), close to
  X1's −14.9 %. Most of the original "300 K helps, 150 K does not" contrast is the 4× longer 150 K horizon.
* **Bins.** A 180-bin read-out of X2 changes ΔĪ_F, post-ramp Ī_F and Δe_F(T) by ≤ 0.2 points. Δe_F′(T) moves by
  2 points (+34.1 → +36.1 %). Under one common reference, the
  frozen Gate-3c rule keeps 180 bins at 150 K (audit F4), so every new cell uses 180 bins.
* **Rate.** The rate is safety-limited, and the limit depends on K and on the genealogy window (T/20).
  * Blind calibration at T 50: rate 2.0 is **unsafe** at K256 for both temperatures (ESSw/K 0.293 and 0.258; lineage
    0.051 and 0.078). Rate **1.0 is safe** at both (0.531 and 0.490).
  * 300 K / K1024 is safe up to 2.0 (X1 calibration).
  * The largest common safe rate is therefore **1.0**. X1, at rate 2.0, cannot be the controlled 300 K / K1024 cell.

## 2. Controlled cells (common rate 1.0, T 50, 180 bins, dt 1e-4; 16 fresh paired seeds each)

#### M300_256: K 256, T 50 t.u., 300 K, 180 bins, FR rate 1 from 1 t.u., continuous; 16 paired seeds 1121000–1121015

* integrity: bitwise paired before the first FR opportunity 16/16; force evaluations per run 128,000,256 (equal in both arms); arms differ only in the method
* ladder (kJ/mol): 0.271, 0.226, 0.181, 0.145, 0.117 (= [3.0, 2.5, 2.0, 1.6, 1.3] × median ABF e_F(T) 0.0903)
* **PRIMARY S_geo 1.035 [0.917, 1.212]** (97.5 %: [0.887, 1.224]); FR faster in 9/16 seeds → **UNRESOLVED**

| statistic | median per-seed FR/ABF − 1 [95 % CI] (FR better) | ABF median | FR median |
|---|---|---|---|
| post-ramp Ī_F (companion) | -8.2 % [-14.5, +0.9] (11/16) | 0.2486 | 0.2274 |
| full-horizon Ī_F | -5.1 % [-8.4, +0.5] (11/16) | 0.3902 | 0.3737 |
| post-ramp Ī_F over [t_FR, 10] | -20.3 % [-24.8, -12.9] (16/16) | 0.6244 | 0.5142 |
| e_F(T/4) | -14.5 % [-30.4, -4.3] (15/16) | 0.2636 | 0.2348 |
| e_F(T/2) | +11.5 % [-1.4, +25.5] (4/16) | 0.1663 | 0.1857 |
| e_F(3T/4) | +16.1 % [+8.2, +35.8] (3/16) | 0.1144 | 0.1477 |
| late-window Ī_F [0.8T, T] | +13.4 % [+1.2, +42.8] (4/16) | 0.0982 | 0.1274 |
| e_F(T) (H2) | +12.4 % [-0.1, +42.6] (5/16) | 0.0903 | 0.1252 |
| e_F′(T) | +4.2 % [-7.6, +27.4] (7/16) | 0.1531 | 0.1796 |
| estimate from deposits after t_FR | +11.0 % [-1.0, +36.4] (5/16) | 0.0800 | 0.1217 |

| ε (kJ/mol) | sustained S [95 % CI] | reached ABF / FR | FR faster / ABF faster | first-hit S |
|---|---|---|---|---|
| 0.271 | 1.23 [1.12, 1.49] | 16 / 16 | 13 / 3 | 1.21 |
| 0.226 | 1.14 [0.97, 1.51] | 16 / 16 | 9 / 5 | 1.16 |
| 0.181 | 0.96 [0.74, 1.29] | 16 / 16 | 6 / 10 | 0.98 |
| 0.145 | 0.94 [0.68, 1.10] | 15 / 12 | 7 / 9 | 0.82 |
| 0.117 | 0.96 [0.82, 1.12] | 13 / 7 | 4 / 9 | 0.93 |

* crossing (FR permanently worse after): 11/16 seeds; median curve 17.2, bootstrap 2.5/50/97.5 % 15.2/19.8/44.8 t.u. (fraction crossing 0.96)
* window establishment: ABF 4.12, FR 2.75 t.u.; paired -1.38 [-1.50, -1.25]
* FR genealogy: min windowed ESS/K 0.538, max lineage share 0.029, replacements 1120 (median over seeds); true translocations ABF 408 vs FR 378
* pooled over seeds vs MC: whole run ABF 0.0818 / FR 0.1042; post-window deposits ABF 0.0753 / FR 0.0981; FR vs ABF post-window 0.0235
* wall-clock per run (single thread, production phase): ABF 190 s, FR 191 s (+1.0 %)

#### M150_256: K 256, T 50 t.u., 150 K, 180 bins, FR rate 1 from 1 t.u., continuous; 16 paired seeds 1122000–1122015

* integrity: bitwise paired before the first FR opportunity 16/16; force evaluations per run 128,000,256 (equal in both arms); arms differ only in the method
* ladder (kJ/mol): 0.394, 0.328, 0.263, 0.210, 0.171 (= [3.0, 2.5, 2.0, 1.6, 1.3] × median ABF e_F(T) 0.1314)
* **PRIMARY S_geo 1.633 [1.341, 2.727]** (97.5 %: [1.239, 2.800]); FR faster in 14/16 seeds → **SUPPORTED (FR+ reaches the ladder sooner)**

| statistic | median per-seed FR/ABF − 1 [95 % CI] (FR better) | ABF median | FR median |
|---|---|---|---|
| post-ramp Ī_F (companion) | -22.1 % [-28.6, -16.9] (15/16) | 0.3500 | 0.2685 |
| full-horizon Ī_F | -14.6 % [-19.2, -11.4] (15/16) | 0.4884 | 0.4126 |
| post-ramp Ī_F over [t_FR, 10] | -27.9 % [-32.9, -23.4] (16/16) | 0.8373 | 0.5816 |
| e_F(T/4) | -38.6 % [-47.0, -37.2] (16/16) | 0.4377 | 0.2613 |
| e_F(T/2) | -15.2 % [-28.1, -7.6] (14/16) | 0.2511 | 0.2081 |
| e_F(3T/4) | +8.8 % [-5.9, +14.8] (6/16) | 0.1803 | 0.1715 |
| late-window Ī_F [0.8T, T] | +23.0 % [+5.9, +31.2] (3/16) | 0.1476 | 0.1554 |
| e_F(T) (H2) | +28.9 % [+6.7, +51.0] (2/16) | 0.1314 | 0.1445 |
| e_F′(T) | +27.0 % [+12.9, +40.7] (3/16) | 0.2274 | 0.2446 |
| estimate from deposits after t_FR | +30.6 % [+6.9, +55.5] (2/16) | 0.1269 | 0.1442 |

| ε (kJ/mol) | sustained S [95 % CI] | reached ABF / FR | FR faster / ABF faster | first-hit S |
|---|---|---|---|---|
| 0.394 | 2.40 [2.02, 2.90] | 16 / 16 | 16 / 0 | 2.37 |
| 0.328 | 2.25 [1.79, 2.83] | 16 / 16 | 14 / 2 | 2.30 |
| 0.263 | 1.95 [1.48, 2.53] | 16 / 16 | 13 / 3 | 1.95 |
| 0.210 | 1.16 [0.97, 1.92] | 16 / 12 | 12 / 4 | 1.37 |
| 0.171 | 0.97 [0.88, 1.45] | 15 / 10 | 4 / 9 | 0.98 |

* crossing (FR permanently worse after): 14/16 seeds; median curve 33.4, bootstrap 2.5/50/97.5 % 29.1/33.4/42.9 t.u. (fraction crossing 1.00)
* window establishment: ABF 6.88, FR 3.56 t.u.; paired -3.12 [-3.75, -2.75]
* FR genealogy: min windowed ESS/K 0.425, max lineage share 0.047, replacements 1102 (median over seeds); true translocations ABF 182 vs FR 162
* pooled over seeds vs MC: whole run ABF 0.1182 / FR 0.1467; post-window deposits ABF 0.1145 / FR 0.1437; FR vs ABF post-window 0.0295
* wall-clock per run (single thread, production phase): ABF 186 s, FR 188 s (+0.8 %)

## 3. Temperature contrast at fixed K = 256 (prereg §5: difference of per-cell medians, independent bootstrap)

| statistic | 300 K | 150 K | 300 K − 150 K [95 % CI] |
|---|---|---|---|
| post-ramp Ī_F (preregistered contrast) | -8.2 % | -22.1 % | **+13.9 points [+4.4, +25.2]** |
| full-horizon Ī_F | -5.1 % | -14.6 % | **+9.5 points [+4.3, +17.3]** |
| post-ramp Ī_F over [1, 10] | -20.3 % | -27.9 % | **+7.6 points [+1.3, +16.6]** |
| e_F(T) | +12.4 % | +28.9 % | **-16.5 points [-39.8, +18.6]** |
| late-window Ī_F [40, 50] | +13.4 % | +23.0 % | **-9.6 points [-25.2, +21.5]** |
| primary log S_geo | +0.034 | +0.491 | -0.457 [-0.948, -0.181] (speedup ratio ×1.58 in favour of 150 K) |

* **Verdict (preregistered statistic): a temperature difference is PRESENT, but in the opposite direction to the old
  X1/X2 comparison.**
  * At fixed K, rate, horizon and bins, FR's finite-time benefit is **larger at 150 K**. Post-ramp Ī_F is −22 % vs
    −8 %, and time to accuracy S_geo is 1.63 (supported) vs 1.03 (unresolved).
  * Both temperatures show a late penalty in e_F(T): +29 % [+7, +51] at 150 K (resolved) and +12 % [−0, +43] at
    300 K (borderline). The difference between them is not resolved.
* **The colder cell is the more establishment-limited one.**
  * ABF's window establishment takes 6.9 t.u. at 150 K vs 4.1 at 300 K. FR shortens it by 3.1 vs 1.4 t.u. (paired medians)
  * ABF makes fewer true translocations at 150 K: 182 vs 408 per 256 walkers.
  * This matches the starvation reading of the LTA temperature sweep, where the gain is largest at the coldest
    temperature, and contradicts the idea that a more energetic barrier makes FR intrinsically ineffective.
  * It is an association between two cells, not a causal law.
* **Late-time decay.** ABF's error decays as about 1/t in both cells:
  * 300 K: -0.85;
  * 150 K: -0.98.

  Continuous FR's decays as about t^−½: -0.54 and -0.45. This is the same signature as in X1, X2 and ERI
  (`MECHANISM.md`).

## 4. Replica count

* **Fixed rate.** Within 300 K, K256 at rate 1.0 (post-ramp Ī_F −8 %) vs X1's K1024 at rate 2.0 (−22 %) is
  confounded by the rate.
* **Not run (budget).** The controlled K1024 row at rate 1.0 (preregistered §5) was not run: about 7.3 core-h even at
  8 pairs (A2). So K and rate are not separated at 300 K, and no K-dependence claim is made from this study.
* **Earlier repository evidence:** at equal force evaluations, FR's gain grows with the replica count
  (`docs/equal_budget/`).

## 5. 150 K early stopping (E2)

* **E2** (confirmatory; K 256, rate 0.5, T 100, 180 bins; FR stopped at 15 t.u.).
  * Primary S_geo 1.25 [1.11, 1.58], SUPPORTED.
  * e_F(T) -19.0 % [-31.6, -6.3].
  * Post-ramp Ī_F -18.1 %.
* **Reading.** At 150 K, stopping FR removes the late penalty that continuous FR shows in this cell (X2 at T 200:
  +57 %; M150_256 at T 50: +29 %) and keeps the early gain.
* **Details.** `EARLY_STOPPING_RESULTS.md`.

## Answer to Q3

**Yes, a difference remains, but it is not the old one.**
* With K (256), horizon (50 t.u.), bins (180) and the safe common rate (1.0) controlled, the colder ITE cell shows
  the *larger* finite-time FR acceleration. Its early and integrated gains are larger, together with a comparable or
  larger late penalty.
* The original "300 K positive / 150 K null" came mainly from comparing a 50 t.u. horizon with a 200 t.u. one, in
  which FR's late deficit outweighs its early gain.
