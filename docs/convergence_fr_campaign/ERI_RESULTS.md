# ERI 600 K — ABF vs ABF + Fisher–Rao (Experiment 4)

*Confirmatory blocks frozen in `PREREGISTRATION.md` §4, run 2026-10-17 03:20–03:32 UTC.
Frozen analysis: `scripts/convergence_fr/analyze_block.py --ladder kT`, with the Bonferroni field of A1.
Figures: `figures/convergence_fr/block_ERI256.png` and `block_ERI1024.png`.*

* **Model.** The validated rigid all-silica ERI surrogate (Gard & Tait), Schüring UA ethane, 600 K, infinite
  dilution. Barrier 6.67 k_BT, 84 % entropic.
* **Integration.** EM dt 2e-4 (Gate 2 admissible), T 40 t.u., 180 bins, trust 200, 1 t.u. ramp, 5 t.u.
  pre-equilibration.
* **Reference.** The exact MC umbrella/WHAM reference of the ERI campaign.
* **The old classifier was NOT used as a gate.** For the record, it labelled these cells
  K64 ABF-SUFFICIENT, K256 ABF-SUFFICIENT, K1024 ABF-SUFFICIENT.
* **Safety-only calibration (blind, top-down).** Rate 2.0, the top rung, was safe at both K:
  * K256: ESSw/K 0.554, lineage 0.023;
  * K1024: 0.675, 0.006.

## Results

#### ERI256: K 256, T 40 t.u., 600 K, 180 bins, FR rate 2 from 1 t.u., continuous; 16 paired seeds 1111000–1111015

* integrity: bitwise paired before the first FR opportunity 16/16; force evaluations per run 51,200,256 (equal in both arms); arms differ only in the method
* ladder (kJ/mol): 0.399, 0.319, 0.239, 0.200, 0.160 (= [0.08, 0.064, 0.048, 0.04, 0.032] k_BT)
* **PRIMARY S_geo 0.517 [0.460, 0.628]** (97.5 %: [0.453, 0.670]); FR faster in 0/16 seeds → **CONTRADICTED (ABF reaches the ladder sooner)**

| statistic | median per-seed FR/ABF − 1 [95 % CI] (FR better) | ABF median | FR median |
|---|---|---|---|
| post-ramp Ī_F (companion) | +32.3 % [+21.7, +45.7] (2/16) | 0.4058 | 0.5451 |
| full-horizon Ī_F | +26.4 % [+17.8, +39.5] (2/16) | 0.4645 | 0.5967 |
| post-ramp Ī_F over [t_FR, 10] | -4.3 % [-8.3, +3.1] (11/16) | 1.0646 | 1.0309 |
| e_F(T/4) | +53.3 % [+27.5, +70.0] (0/16) | 0.3903 | 0.5876 |
| e_F(T/2) | +100.9 % [+61.3, +120.5] (1/16) | 0.2066 | 0.4221 |
| e_F(3T/4) | +108.5 % [+73.3, +174.1] (2/16) | 0.1456 | 0.3354 |
| late-window Ī_F [0.8T, T] | +116.3 % [+81.9, +181.9] (1/16) | 0.1294 | 0.3048 |
| e_F(T) (H2) | +113.0 % [+79.0, +236.0] (1/16) | 0.1240 | 0.2825 |
| e_F′(T) | +62.1 % [+36.6, +94.1] (1/16) | 0.2335 | 0.3561 |
| estimate from deposits after t_FR | +126.2 % [+88.2, +280.0] (1/16) | 0.1117 | 0.2703 |

| ε (kJ/mol) | sustained S [95 % CI] | reached ABF / FR | FR faster / ABF faster | first-hit S |
|---|---|---|---|---|
| 0.399 | 0.47 [0.36, 0.75] | 16 / 16 | 0 / 16 | 0.47 |
| 0.319 | 0.42 [0.32, 0.62] | 16 / 13 | 1 / 15 | 0.44 |
| 0.239 | 0.53 [0.39, 0.65] | 15 / 6 | 1 / 14 | 0.50 |
| 0.200 | 0.59 [0.47, 0.76] | 15 / 1 | 1 / 14 | 0.55 |
| 0.160 | 0.74 [0.62, 1.00] | 12 / 1 | 1 / 11 | 0.69 |

* crossing (FR permanently worse after): 15/16 seeds; median curve 5.2, bootstrap 2.5/50/97.5 % 4.3/5.2/6.2 t.u. (fraction crossing 1.00)
* window establishment: ABF 2.10, FR 1.90 t.u.; paired -0.20 [-0.30, -0.10]
* FR genealogy: min windowed ESS/K 0.517, max lineage share 0.031, replacements 1817 (median over seeds); true translocations ABF 914 vs FR 836
* pooled over seeds vs MC: whole run ABF 0.1174 / FR 0.2608; post-window deposits ABF 0.1015 / FR 0.2473; FR vs ABF post-window 0.1473
* wall-clock per run (single thread, production phase): ABF 118 s, FR 118 s (+0.1 %)

#### ERI1024: K 1024, T 40 t.u., 600 K, 180 bins, FR rate 2 from 1 t.u., continuous; 16 paired seeds 1111000–1111015

* integrity: bitwise paired before the first FR opportunity 16/16; force evaluations per run 204,801,024 (equal in both arms); arms differ only in the method
* ladder (kJ/mol): 0.399, 0.319, 0.239, 0.200, 0.160 (= [0.08, 0.064, 0.048, 0.04, 0.032] k_BT)
* **PRIMARY S_geo 0.510 [0.487, 0.577]** (97.5 %: [0.485, 0.598]); FR faster in 0/16 seeds → **CONTRADICTED (ABF reaches the ladder sooner)**

| statistic | median per-seed FR/ABF − 1 [95 % CI] (FR better) | ABF median | FR median |
|---|---|---|---|
| post-ramp Ī_F (companion) | +32.6 % [+27.5, +34.7] (0/16) | 0.3959 | 0.4952 |
| full-horizon Ī_F | +27.4 % [+23.8, +30.4] (0/16) | 0.4423 | 0.5438 |
| post-ramp Ī_F over [t_FR, 10] | -2.5 % [-3.7, -1.0] (13/16) | 1.0201 | 0.9889 |
| e_F(T/4) | +42.7 % [+34.2, +47.3] (0/16) | 0.4097 | 0.5603 |
| e_F(T/2) | +97.1 % [+65.6, +105.3] (0/16) | 0.2055 | 0.3771 |
| e_F(3T/4) | +112.0 % [+83.0, +130.6] (0/16) | 0.1476 | 0.3015 |
| late-window Ī_F [0.8T, T] | +115.6 % [+98.1, +140.4] (0/16) | 0.1280 | 0.2736 |
| e_F(T) (H2) | +119.3 % [+102.9, +140.6] (0/16) | 0.1193 | 0.2581 |
| e_F′(T) | +95.9 % [+80.7, +110.6] (0/16) | 0.1486 | 0.3040 |
| estimate from deposits after t_FR | +141.6 % [+123.2, +165.9] (0/16) | 0.1019 | 0.2442 |

| ε (kJ/mol) | sustained S [95 % CI] | reached ABF / FR | FR faster / ABF faster | first-hit S |
|---|---|---|---|---|
| 0.399 | 0.56 [0.50, 0.62] | 16 / 16 | 0 / 16 | 0.56 |
| 0.319 | 0.46 [0.44, 0.53] | 16 / 16 | 0 / 16 | 0.46 |
| 0.239 | 0.42 [0.40, 0.52] | 16 / 5 | 0 / 16 | 0.43 |
| 0.200 | 0.54 [0.45, 0.62] | 16 / 1 | 0 / 16 | 0.52 |
| 0.160 | 0.65 [0.53, 0.78] | 15 / 0 | 0 / 15 | 0.65 |

* crossing (FR permanently worse after): 16/16 seeds; median curve 5.2, bootstrap 2.5/50/97.5 % 4.9/5.2/5.6 t.u. (fraction crossing 1.00)
* window establishment: ABF 2.10, FR 1.90 t.u.; paired -0.20 [-0.20, -0.15]
* FR genealogy: min windowed ESS/K 0.666, max lineage share 0.007, replacements 3941 (median over seeds); true translocations ABF 3619 vs FR 3383
* pooled over seeds vs MC: whole run ABF 0.1156 / FR 0.2520; post-window deposits ABF 0.0983 / FR 0.2367; FR vs ABF post-window 0.1389
* wall-clock per run (single thread, production phase): ABF 485 s, FR 480 s (-1.0 %)

## Mechanism (`scripts/overnight/fr_mechanism.py`, read-only; `results/convergence_fr_campaign/mechanism/fr_mechanism_ERI*.json`)

| quantity | ABF | ABF + FR | note |
|---|---|---|---|
| **K256** | | | |
| pooled late-half estimate vs MC, D | 0.027 | 0.130 | FR vs ABF 0.100 (noise 0.036) |
| pooled whole-run estimate vs MC, D | 0.116 | 0.260 | FR vs ABF 0.141 |
| deposit-weighted η conditional, RMS vs MC | 0.0033 | 0.0160 | |
| deposit-weighted u_x² conditional, RMS vs MC | 0.0007 | 0.0019 | |
| late-half FR − ABF mean-force RMS: window / flank / cage | | | 0.058 / 0.187 / 0.040 kJ/mol/rad |
| **K1024** | | | |
| pooled late-half estimate vs MC, D | 0.029 | 0.127 | FR vs ABF 0.098 (noise 0.016) |
| pooled whole-run estimate vs MC, D | 0.115 | 0.252 | FR vs ABF 0.136 |
| deposit-weighted η conditional, RMS vs MC | 0.0067 | 0.0171 | |
| deposit-weighted u_x² conditional, RMS vs MC | 0.0009 | 0.0017 | |
| late-half FR − ABF mean-force RMS: window / flank / cage | | | 0.062 / 0.169 / 0.022 kJ/mol/rad |

### Observations

* **FR's estimator converges to the wrong profile.** At both K, the pooled late-half estimate of the FR arm sits
  about 0.13 kJ/mol from exact MC, against 0.03 for ABF, with noise about 0.01–0.02.
  * The FR − ABF difference is about 3× its noise at K256 (0.100 vs 0.036) and 6× at K1024 (0.098 vs 0.016).
  * It is an FR-induced **stationary estimator bias**, not variance.
  * It sits mostly in the flank mean force.
* **FR distorts the slow hidden coordinate in the cages.** η measures the molecule's position along the cage's long
  c axis. The deposit-weighted conditional mean of η in the cage coarse bins is 2.5–5× further from MC under FR.
  This is the first FR-induced conditional distortion measured in these molecular cells; ITE's are clean (X1, X2,
  E1).
* **FR's useful window is short.**
  * FR is ahead only at about 1.5–5 t.u., by up to −21 % (K256).
  * The paired e_F curves cross permanently at 5.2 t.u. (K256 bootstrap 4.3–6.2; K1024 4.9–5.6), in 15/16 and
    16/16 seeds.
* **ABF is not establishment-limited here.** ABF's window establishment takes 2.1 t.u., FR's 1.9. The entropic
  6.67 k_BT barrier at 600 K is crossed easily: ABF makes about 900 translocations per 256 walkers in 40 t.u.

### Hypothesis (not established)

* **Proposed mechanism.** The uniform-target FR keeps draining the denser cage and refilling the window and flank.
  * In steady state this replaces long-resident cage walkers, whose η is equilibrated, by recently arrived ones,
    whose η is not yet relaxed.
  * That is a non-equilibrium steady state whose conditional mean force differs from the Gibbs one.
* **Why the bias can persist.** The resulting estimator bias keeps the applied bias inexact, which keeps the natural
  marginal non-uniform, which keeps FR pumping. A self-consistent biased fixed point is possible.
* **What has been ruled out.** Circulation does not explain it: the ERI circulation (late half) is negative, −0.02 to −0.05 kJ/mol/rad,
  in both arms. The paired FR − ABF difference is not resolved: K256 +0.002 [−0.047, +0.025]; K1024 +0.007 [−0.001, +0.013].

## Early stopping (prereg §4)

* **Trigger.** Met at K1024: early post-ramp Ī_F over [1, 10] is −2.5 % [−3.7, −1.0] and e_F(T) is +119 %, both
  resolved. Not met at K256: −4.3 % [−8.3, +3.1].
* **Stop-time rule.** The frozen rule takes 2 × the ABF establishment time (2.1 t.u.), rounded up to 5 t.u.,
  giving 5 t.u. The stop must also lie below the crossing's 2.5 % quantile, which is 4.9 t.u. at K1024 and 4.3 at
  K256. **5 t.u. fails that condition.**
* **No ERI early-stopping block was run.** FR's advantage window closes before a stop time chosen by the frozen rule
  could be applied.
  * A shorter, ad-hoc stop such as 3 t.u. was not tried, because it would be chosen after seeing these
    confirmatory curves.
  * At best, early stopping could keep the −2.5 to −4 % gain over [1, 10].

## Answer to Q4

**No.** At fixed K and equal force evaluations, continuous marginal FR **slows** ABF's convergence in ERI 600 K, the
strongly entropic-barrier molecular example.
* It takes about twice as long to reach any rung of the frozen tolerance ladder: S_geo 0.52 (K256) and 0.51
  (K1024), 0/16 seeds faster, Bonferroni-significant.
* It roughly doubles the final error.
* The large entropic fraction did not make FR useful here, because ABF was never establishment-limited.
