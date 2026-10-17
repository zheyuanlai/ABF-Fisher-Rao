# R2 campaign — results

*Preregistered in `PREREGISTRATION.md` (commit 5effe58d, 2026-10-17 11:38 UTC, before any run). Wave 1 (held out:
R2C3, R2C4, R2H80, R2H350, R2N256) ran 11:38–11:47 UTC; wave 2 (in-sample: R2C1, R2C2, R2C5) ran 11:47–12:02 UTC.
Frozen analysis: `scripts/adaptive_fr/analyze_online.py` (unchanged since 6716aefc) called by
`scripts/r2_campaign/analyze_all.sh`; verdict rules `scripts/r2_campaign/verdicts.py`; tables
`scripts/r2_campaign/results_tables.py` (formatting only; `results/r2_campaign/analysis/tables.md`).
Analysis JSONs: `results/r2_campaign/analysis/<block>.json` (+ `_curves.npz`), `verdicts.json`. Nothing was retuned,
re-run or re-analysed after the first analysis of each cell.*

Every cell compares two arms on 16 fresh paired seeds: histogram ABF, and the same ABF + FR stopped online by the
frozen rule R2. The arms differ only in `fr_controller`, are bitwise equal before FR start, use equal force
evaluations, and have zero FR events after the stop (all asserted; §5).

## 0. Summary

**R2 generalises by the preregistered definition.** All six positive cells are SUPPORTED at the Bonferroni-8 level:
the in-sample ITE cells (300 K K1024 S_geo 1.80, 150 K K256 1.44) and the four held-out LTA cells, two of which no
stop rule had ever seen (150 K 2.45, 300 K 1.39, **80 K 1.84, 350 K 1.29**; FR faster in 95 of 96 seeds). In ERI, R2
stopped at 4.0 t.u. in all 32 runs and both ERI cells are EQUIVALENT by the preregistered TOST margin [0.80, 1.25]
(K1024 0.92 [0.86, 1.04]; K256 0.87 [0.81, 0.98]), against R1's 0.61 and continuous FR's 0.51. The honest reading of
ERI is "inside the no-harm margin, not neutral": a residual 8–13 % slowdown remains, resolved at K256 (ABF faster in
12/16 seeds; e_F(T/2) +11 % [+6, +34]), and both ERI endpoints are about +10 % worse with CIs touching zero. The
residual is the memory of the 3 t.u. of FR-period deposits, which is what stopping cannot remove.

## 1. Primary outcome

| cell | system | role | test | R2 stop, t.u.: median [IQR] (range) | S_geo [95 % CI] | 90 % CI | Bonferroni-8 CI | FR faster | verdict |
|---|---|---|---|---|---|---|---|---|---|
| R2C1 | ITE 300 K K1024 T50 | in-sample | superiority | 5.00 [5.00, 5.00] (5.00–5.50) | **1.80 [1.55, 2.04]** | [1.55, 2.00] | [1.52, 2.13] | 16/16 | **SUPPORTED** |
| R2C2 | ITE 150 K K256 T100 | in-sample | superiority | 8.50 [8.38, 9.00] (7.00–10.00) | **1.44 [1.29, 1.70]** | [1.30, 1.65] | [1.28, 1.78] | 16/16 | **SUPPORTED** |
| R2C5 | ERI 600 K K1024 T40 | in-sample; expected-negative | equivalence | 4.00 [4.00, 4.00] (4.00–4.00) | **0.92 [0.86, 1.04]** | [0.87, 1.00] | [0.83, 1.05] | 5/16 | **EQUIVALENT** |
| R2C3 | LTA 150 K K1024 T60 | held out; positive control | superiority | 18.50 [17.88, 19.12] (17.50–20.50) | **2.45 [2.26, 2.61]** | [2.27, 2.60] | [2.23, 2.74] | 16/16 | **SUPPORTED** |
| R2C4 | LTA 300 K K1024 T60 | held out | superiority | 14.50 [14.00, 15.12] (13.50–16.00) | **1.39 [1.34, 1.51]** | [1.36, 1.48] | [1.31, 1.56] | 16/16 | **SUPPORTED** |
| R2H80 | LTA 80 K K1024 T60 | held out; new strong-positive candidate | superiority | 31.25 [30.38, 32.50] (28.00–34.50) | **1.84 [1.64, 2.19]** | [1.64, 2.18] | [1.45, 2.37] | 15/16 | **SUPPORTED** |
| R2H350 | LTA 350 K K1024 T60 | held out; new borderline candidate | superiority | 14.00 [13.50, 14.50] (13.00–15.00) | **1.29 [1.25, 1.45]** | [1.26, 1.38] | [1.20, 1.49] | 16/16 | **SUPPORTED** |
| R2N256 | ERI 600 K K256 T40 | held-out allocation; expected-negative | equivalence | 4.00 [4.00, 4.00] (4.00–4.00) | **0.87 [0.81, 0.98]** | [0.81, 0.96] | [0.75, 1.02] | 4/16 | **EQUIVALENT** |

* **Stop times landed where the offline replay predicted** (DEVELOPMENT.md §4: 5.0 / 8.5 / 4.0 / 18.6 / 14.6 / ≈ 32
  / ≈ 14 / ≈ 4): realised medians 5.0 / 8.5 / 4.0 / 18.5 / 14.5 / 31.25 / 14.0 / 4.0. Every run stopped as
  "established" at the first check that met the criterion; no safety stop fired; no run kept FR to the horizon.
* **t_est tracks the offline establishment time of the same FR arm** (§3: 2.50 vs 2.38, 4.25 vs 4.12, 2.00 vs 1.90,
  9.25 vs 9.10, 7.25 vs 7.20, 15.6 vs 15.55, 7.00 vs 6.95, 2.00 vs 1.90 t.u.), i.e. the online statistic is the
  offline one up to the 0.25-t.u. check grid.
* **Against R1 on the same cells (other seeds):** ITE 300 K K1024 1.80 vs 1.67; ITE 150 K 1.44 vs 1.64; LTA 150 K
  2.45 vs 2.52; LTA 300 K 1.39 vs 1.46; ERI K1024 0.92 vs 0.61. R2 stops 2–3 t.u. earlier than R1 in LTA (18.5 vs
  21.5; 14.5 vs 16.0) with the benefit unchanged within the CIs, and 11 t.u. earlier in ERI.

## 2. Secondary outcomes

| cell | post-ramp Ī_F | early Ī_F [t_FR, 0.4 T] | late-window Ī_F [0.8 T, T] | e_F(T/2) | e_F(T) | e_F′(T) | post-stop-only estimate | endpoint flag |
|---|---|---|---|---|---|---|---|---|
| R2C1 | -39 % [-45, -35] (16/16) | -39 % [-42, -35] (16/16) | -40 % [-47, -32] (16/16) | -40 % [-50, -36] (16/16) | -42 % [-49, -27] (16/16) | -24 % [-40, -16] (16/16) | -14 % [-36, +3] (11/16) | gain |
| R2C2 | -25 % [-32, -18] (16/16) | -26 % [-29, -19] (16/16) | -19 % [-36, -9] (13/16) | -29 % [-37, -16] (15/16) | -20 % [-32, -8] (13/16) | -17 % [-26, -0] (12/16) | -8 % [-31, +23] (8/16) | gain |
| R2C5 | -1 % [-5, +3] (10/16) | -5 % [-8, -1] (12/16) | +11 % [-1, +17] (5/16) | +13 % [-6, +22] (5/16) | +10 % [-2, +13] (6/16) | +5 % [+3, +13] (2/16) | +24 % [-6, +49] (5/16) | unresolved |
| R2C3 | -49 % [-54, -47] (16/16) | -39 % [-43, -36] (16/16) | -59 % [-71, -56] (16/16) | -63 % [-69, -56] (16/16) | -59 % [-71, -54] (16/16) | -49 % [-54, -42] (16/16) | -53 % [-68, -39] (16/16) | gain |
| R2C4 | -26 % [-28, -21] (16/16) | -22 % [-26, -20] (16/16) | -29 % [-35, -20] (16/16) | -30 % [-33, -21] (16/16) | -24 % [-39, -16] (16/16) | -15 % [-21, -11] (15/16) | -4 % [-18, +11] (9/16) | gain |
| R2H80 | -62 % [-66, -61] (16/16) | -41 % [-45, -36] (16/16) | -83 % [-85, -81] (16/16) | -75 % [-81, -75] (16/16) | -83 % [-85, -81] (16/16) | -57 % [-62, -55] (16/16) | -78 % [-80, -74] (16/16) | gain |
| R2H350 | -26 % [-29, -22] (16/16) | -24 % [-25, -21] (16/16) | -27 % [-35, -16] (15/16) | -34 % [-39, -27] (16/16) | -24 % [-31, -14] (15/16) | -1 % [-2, -0] (12/16) | +9 % [-2, +27] (6/16) | gain |
| R2N256 | +2 % [-6, +10] (7/16) | -2 % [-4, +3] (10/16) | +16 % [-11, +26] (5/16) | +11 % [+6, +34] (3/16) | +12 % [-6, +27] (6/16) | +5 % [-11, +19] (7/16) | -10 % [-29, +11] (9/16) | unresolved |

## 3. The online schedule actually realised

| cell | never stopped | reasons | t_est median (range) | FR replacements per run | events after stop | min windowed ESS/K | max lineage | window establishment ABF → FR (t.u.) | paired diff. | translocations ABF / FR | wall s ABF / FR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R2C1 | 0 | established | 2.50 (2.50–2.75) | 582 | 0 | 0.324 | 0.019 | 4.12 → 2.38 | -1.75 [-1.81, -1.62] | 1601 / 1640 | 776 / 768 |
| R2C2 | 0 | established | 4.25 (3.50–5.00) | 116 | 0 | 0.499 | 0.031 | 7.00 → 4.12 | -2.62 [-3.50, -1.88] | 398 / 402 | 408 / 418 |
| R2C5 | 0 | established | 2.00 (2.00–2.00) | 398 | 0 | 0.671 | 0.006 | 2.05 → 1.90 | -0.20 [-0.20, -0.10] | 3627 / 3610 | 523 / 532 |
| R2C3 | 0 | established | 9.25 (8.75–10.25) | 364 | 0 | 0.603 | 0.008 | 16.75 → 9.10 | -7.50 [-7.90, -7.20] | 317 / 383 | 490 / 501 |
| R2C4 | 0 | established | 7.25 (6.75–8.00) | 198 | 0 | 0.759 | 0.005 | 8.95 → 7.20 | -1.80 [-1.90, -1.55] | 926 / 949 | 495 / 485 |
| R2H80 | 0 | established | 15.62 (14.00–17.25) | 491 | 0 | 0.716 | 0.007 | 36.40 → 15.55 | -20.85 [-24.00, -19.80] | 66 / 138 | 495 / 485 |
| R2H350 | 0 | established | 7.00 (6.50–7.50) | 178 | 0 | 0.782 | 0.005 | 8.20 → 6.95 | -1.10 [-1.40, -1.00] | 1150 / 1166 | 497 / 500 |
| R2N256 | 0 | established | 2.00 (2.00–2.00) | 158 | 0 | 0.546 | 0.023 | 2.05 → 1.90 | -0.20 [-0.20, -0.10] | 902 / 902 | 239 / 232 |

* **R2C1** (kT ladder): ε 0.200: S 1.85 [1.51, 1.99] reach A16/B16; ε 0.160: S 1.80 [1.53, 2.09] reach A16/B16; ε 0.120: S 1.90 [1.59, 2.18] reach A16/B16; ε 0.100: S 1.85 [1.61, 2.16] reach A13/B16; ε 0.080: S 1.64 [1.38, 1.91] reach A6/B16
* **R2C2** (abf ladder): ε 0.216: S 1.52 [1.33, 1.66] reach A16/B16; ε 0.180: S 1.47 [1.33, 1.66] reach A16/B16; ε 0.144: S 1.46 [1.31, 1.54] reach A16/B16; ε 0.115: S 1.48 [1.19, 1.72] reach A15/B16; ε 0.094: S 1.32 [1.04, 1.68] reach A14/B15
* **R2C5** (kT ladder): ε 0.399: S 0.95 [0.89, 1.00] reach A16/B16; ε 0.319: S 0.95 [0.90, 0.96] reach A16/B16; ε 0.239: S 0.93 [0.84, 1.00] reach A16/B16; ε 0.200: S 0.93 [0.81, 1.02] reach A16/B16; ε 0.160: S 0.92 [0.84, 1.05] reach A16/B16
* **R2C3** (abf ladder): ε 0.399: S 2.20 [2.00, 2.49] reach A16/B16; ε 0.333: S 2.50 [2.29, 2.70] reach A16/B16; ε 0.266: S 2.47 [2.34, 2.68] reach A16/B16; ε 0.213: S 2.55 [2.35, 2.81] reach A16/B16; ε 0.173: S 2.51 [2.36, 2.88] reach A16/B16
* **R2C4** (abf ladder): ε 0.295: S 1.45 [1.38, 1.54] reach A16/B16; ε 0.246: S 1.45 [1.36, 1.58] reach A16/B16; ε 0.197: S 1.40 [1.26, 1.54] reach A16/B16; ε 0.158: S 1.36 [1.27, 1.48] reach A16/B16; ε 0.128: S 1.36 [1.30, 1.45] reach A15/B16
* **R2H80** (abf ladder): ε 0.498: S 1.00 [1.00, 1.00] reach A16/B16; ε 0.415: S 1.00 [1.00, 1.05] reach A16/B16; ε 0.332: S 1.01 [1.00, 1.84] reach A16/B16; ε 0.266: S 4.10 [2.85, 5.70] reach A16/B16; ε 0.216: S 4.35 [3.98, 5.36] reach A15/B16
* **R2H350** (abf ladder): ε 0.163: S 1.37 [1.18, 1.41] reach A16/B16; ε 0.136: S 1.26 [1.18, 1.36] reach A16/B16; ε 0.109: S 1.30 [1.25, 1.49] reach A16/B16; ε 0.087: S 1.27 [1.14, 1.56] reach A14/B16; ε 0.071: S 1.23 [1.19, 1.40] reach A13/B15
* **R2N256** (kT ladder): ε 0.399: S 0.92 [0.81, 1.01] reach A16/B16; ε 0.319: S 0.93 [0.83, 0.96] reach A16/B16; ε 0.239: S 0.85 [0.72, 0.99] reach A16/B16; ε 0.200: S 0.87 [0.78, 0.99] reach A15/B15; ε 0.160: S 0.94 [0.87, 1.00] reach A12/B12

## 4. Rule-level questions (preregistered §2)

| question | rule | outcome |
|---|---|---|
| Q-ITE (in-sample) | R2C1 and R2C2 SUPPORTED | **yes** (1.80, 1.44) |
| Q-LTA (held out) | R2C3 SUPPORTED, R2C4 not CONTRADICTED | **yes** (2.45, 1.39; both SUPPORTED) |
| Q-NEW (held out, new temperatures) | R2H80 SUPPORTED, R2H350 not CONTRADICTED | **yes** (1.84; 1.29 SUPPORTED, better than the "borderline" prediction) |
| Q-ERI (expected-negative) | R2C5 and R2N256 EQUIVALENT or BENEFICIAL | **yes, EQUIVALENT** (0.92 [90 %: 0.87, 1.00]; 0.87 [90 %: 0.81, 0.96]); neither is HARMFUL by the 95 % rule at K1024, but R2N256's 95 % CI excludes 1 |
| "R2 preserves the acceleration" (Bonferroni-8) | R2C3, R2H80 SUPPORTED; R2C1, R2C2, R2C4, R2H350 not CONTRADICTED | **yes** (all six SUPPORTED) |
| "R2 avoids ERI-type harm" | both ERI cells EQUIVALENT or BENEFICIAL | **yes by the margin**; see §6.3 for what the margin hides |
| **"R2 generalises by the preregistered definition"** | both of the above | **YES** |

## 5. Integrity

* R2C1: paired bitwise before FR start 16/16; force evaluations per run 512,001,024 (equal in both arms); arms differ only in ['fr_controller']; seeds 1400000–1400015.
* R2C2: paired bitwise before FR start 16/16; force evaluations per run 256,000,256 (equal in both arms); arms differ only in ['fr_controller']; seeds 1410000–1410015.
* R2C5: paired bitwise before FR start 16/16; force evaluations per run 204,801,024 (equal in both arms); arms differ only in ['fr_controller']; seeds 1440000–1440015.
* R2C3: paired bitwise before FR start 16/16; force evaluations per run 307,201,024 (equal in both arms); arms differ only in ['fr_controller']; seeds 1420000–1420015.
* R2C4: paired bitwise before FR start 16/16; force evaluations per run 307,201,024 (equal in both arms); arms differ only in ['fr_controller']; seeds 1430000–1430015.
* R2H80: paired bitwise before FR start 16/16; force evaluations per run 307,201,024 (equal in both arms); arms differ only in ['fr_controller']; seeds 1450000–1450015.
* R2H350: paired bitwise before FR start 16/16; force evaluations per run 307,201,024 (equal in both arms); arms differ only in ['fr_controller']; seeds 1460000–1460015.
* R2N256: paired bitwise before FR start 16/16; force evaluations per run 51,200,256 (equal in both arms); arms differ only in ['fr_controller']; seeds 1470000–1470015.

* Events after the stop: 0 in every run (asserted). Reasons: 'established' in 128/128 FR runs.
* Genealogy stayed healthy everywhere: median over runs of the per-run minimum windowed ESS/K ≥ 0.32 (ITE 300 K K1024,
  the most active FR) and ≤ 0.78; maximum lineage share ≤ 0.031; R2's backstops (0.15; 0.10) were never approached.

## 6. Readings

### 6.1 The online rule is the offline rule

The trailing-1 t.u. deposit share of the window bins reaches 0.75 × the flat share at the same time as the offline
`establishment_time` of the same run (§1), and 2 × that time reproduces the stop times the offline rule would have
chosen with hindsight. R2 therefore adapts to the system through the one quantity the mechanism work identified
(the establishment delay FR removes, MECHANISM.md §3.1), with no system-specific setting: stops run from 4.0 t.u.
(ERI) to 31 t.u. (LTA 80 K).

### 6.2 Where ABF is establishment-limited, the acceleration is preserved and grows with the delay

S_geo follows the establishment delay FR removes, as in the R1 cells: LTA 80 K −20.9 t.u. → fine-rung speedups
4.1–4.35× (S_geo 1.84 only because the three coarse rungs of the ABF-anchored ladder, 0.50 / 0.41 / 0.33 kJ/mol at an
ABF endpoint error of 0.166, are reached by both arms within 5 t.u. and contribute S = 1; the endpoint is −83 %
and the post-stop-only estimate −78 %, the largest post-stop advantage in the project: at the stop, 31 t.u., the ABF
arm has not yet established the window at all, 36.4 t.u.); LTA 150 K −7.5 → 2.45; ITE 150 K −2.6 → 1.44; ITE
300 K K1024 −1.75 → 1.80; LTA 300 K −1.8 → 1.39; LTA 350 K −1.1 → 1.29; ERI −0.2 → 0.92 / 0.87. LTA 350 K, where the
continuous-FR endpoint gain had been borderline (−9 %), gives a clear speedup and a −24 % endpoint under R2: stopping
at 14 t.u. removes continuous FR's late cost there as well.

### 6.3 ERI: the harm is cut, not removed

* **What R2 changed.** R1 stopped ERI at a median 15.25 t.u. (range 5.5–25.5), after FR's useful window had closed;
  R2 stops at 4.0 t.u. in every run, 1.1 t.u. before continuous FR's measured crossing (5.2; here the paired curves
  cross at 5.1 t.u. in the median and in 10/16 seeds at K1024). The early window [1, 16] is now resolvably better
  under FR at K1024 (−4.8 % [−7.7, −1.4]) and neutral at K256 (−1.5 % [−3.7, +3.1]); R1 had +6 %.
* **What remains.** From T/2 on, the FR arm is about 10 % worse: K1024 e_F(T/2) +13 % [−6, +22], e_F(T) +10 %
  [−2, +13], e_F′(T) +5 % [+3, +13]; K256 e_F(T/2) +11 % [+6, +34], e_F(T) +12 % [−6, +27]. The time-to-accuracy
  speedups are 0.92 and 0.87, with the K256 CI excluding 1 and ABF faster in 11/16 and 12/16 seeds.
* **Where it sits.** The post-stop-only estimates are not resolved worse (K1024 +24 % [−6, +49]; K256 −10 %
  [−29, +11]), so after 4.0 t.u. the FR arm samples like ABF. The excess is the memory of the deposits made during
  1–4 t.u., the period in which ERI's FR-sustained conditional distortion (MECHANISM.md §2) is active; in ITE the
  same period's deposits are *less* biased than ABF's (establishment memory), in ERI they are more biased, and the
  cumulative estimator keeps them. A stop rule cannot remove this; only starting FR later, not at all, or not
  accumulating during FR could, none of which was tested here.
* **How to read the TOST verdict.** The preregistered margin was ±25 % on S_geo. Both ERI cells fall inside it, so
  the rule-level question is answered "yes"; but "equivalent" here means "a slowdown smaller than 25 %", and at K256
  the slowdown is statistically resolved. A reviewer who wants "no harm" in the strict sense should read ERI as a
  residual 8–13 % cost. By the earlier campaigns' endpoint rule (CI upper bound < +10 % = "penalty removed") both
  ERI endpoints are "unresolved" (+13 %, +27 %).

### 6.4 The stability guard (development decision, DEVELOPMENT.md §3.1)

Had the stability guard gated the decision, the replay puts the ERI stop at 5.6 t.u., at the crossing. The realised
early-window advantage at the 4.0-t.u. stop (−4.8 % at K1024) is consistent with the earlier stop being the right
side of the crossing, but this is a single-rule campaign: no cell ran both variants, and the difference between a
4.0 and a 5.6 stop in ERI is not measured.

## 7. What this campaign does not establish

* Equal-budget or wall-clock claims: none were made or tested; every comparison is at fixed K and equal force
  evaluations (the adaptive-FR campaign's EFFICIENCY.md stands: no ABF + FR allocation was shown to beat the best
  ABF allocation).
* The 80 K and 350 K references are the sweep's umbrella/WHAM (overdamped EM) references, not Metropolis MC; their
  `gamma_ref` is a central finite difference of F, so the e_F′ rows at those temperatures are finite-difference
  referenced (PREREGISTRATION.md §4). The primary statistic uses only F.
* ERI is the only system in which R2 was tested where ABF establishes within about 1 t.u. of the ramp end, and ERI
  K1024 was development data; the held-out ERI cell is held out at the allocation level only. A genuinely new
  expected-negative framework (an ITE cell at higher temperature, or the ethane-in-rigid-Si-IHW system) needs a new
  reference and dt validation and was costed out of this campaign (PREREGISTRATION.md §4).
* No sham, placebo, matched-turnover or oracle arm was run (campaign rule), so nothing here attributes the ERI
  residual to the Fisher–Rao direction rather than to turnover.
* The stop factor 2 and level 0.75 were carried over from the offline rule, not optimised; the development replay
  (DEVELOPMENT.md §3) shows factors 1.5–3 would move every stop proportionally, with ERI's crossing at 5.2 t.u. as
  the constraint from above and the earliest tested ITE stop (5 t.u.) as the constraint from below.

## 8. Compute

36.7 CPU core-h (`COMPUTE_LEDGER.md`): wave 1 19.8, wave 2 16.5, development and analyses ≈ 0.4. The preregistered
estimate was 26.5; the per-run wall was 1.1–1.8× the adaptive-FR campaign's under a host load of 125–220, and the
account is wall-based. Adaptive-FR authorisation ≈ 80.8 of 100 used; aggregate ≈ 279.2 of ≈ 298.4.
