# Experiment D — mechanism: why continuous FR biases late estimates, and what stopping changes

*Status: EXPLORATORY throughout (PREREGISTRATION §D, declared 07:22 UTC before the D2 runs). No statistic here is
confirmatory, and none was used to choose an FR rate, a stop time or the online rule.
Scripts: `scripts/adaptive_fr/bias_variance.py` (D1), `scripts/adaptive_fr/clone_age.py` (D2). Outputs:
`results/adaptive_fr_campaign/mechanism/{bias_variance.json, bias_variance_curves.npz, clone_age.json}`.
Figures: `figures/adaptive_fr/bias_variance.png` (6) and `mechanism_clone_age_bias.png` (7).*

All comparisons are ABF vs the same ABF + FR (no third arm). Errors are against the exact-MC references used by every
confirmatory analysis (ITE: `results/overnight/ite{,150}/reference`; ERI: `results/eri_campaign/reference`).

## 0. Answer to Q4 in one paragraph

Continuous FR makes the **deposits it collects while it is active** biased. The late-only decomposition (D1) puts the
squared bias of FR's late-window estimate at 0.41–0.89 of its MSE. ABF's late window is variance-dominated (bias share
0.05–0.26). The absolute late-window bias is about 3–6× ABF's in every block. Once FR is stopped, the deposits made after the
stop are close to unbiased: bias 0.009–0.015 kJ/mol, against 0.035–0.13 under continuous FR. The bias is **not
confined to recently cloned walkers** (D2). Dropping every deposit made within 1 t.u. of a cloning removes only part
of FR's excess late error: ERI 0.132 → 0.110 kJ/mol against ABF 0.038; ITE 0.050 → 0.043 against 0.024. In ERI the
fibre conditionals of the whole FR population are displaced from equilibrium, for walkers that were not recently
cloned too: η 4.6× and u_x² 2.6× ABF's deviation. The data therefore point to a **population-level distortion of the
conditional (fibre) distribution, maintained for as long as reallocation continues.** They do not point to a
per-clone transient, and they do not point to a residue of early deposits. Stopping FR removes the source and keeps
the early establishment gain. The specific selection mechanism that produces the distortion is **not identified**:
in ITE the two monitored fibre coordinates are barely displaced, although the late mean-force bias is real (§2.3).

## 1. D1 — bias–variance decomposition across seeds (existing data)

For each arm, at each save, over the 16 independent seeds:
* MSE = mean over seeds of ‖F − F_ref‖²/n_bins, with F centred;
* V = the mean per-bin variance over seeds;
* B² = ‖mean F − F_ref‖²/n_bins − V/n − s_ref².

The last term, s_ref², is the reference's own squared standard error. "Bias" below is √B², with the sign of B²; a
small negative value means noise. *Late-only* is the estimate built only from deposits after the split: after the FR
stop for stopped schedules, after T/2 otherwise. It is computed for both arms of a block over the same window.

### 1.1 Table (final time T; kJ/mol)

| block | arm | final RMSE | final bias | final sd | bias share | late-only window | late-only bias | late-only sd | late-only bias share |
|---|---|---|---|---|---|---|---|---|---|
| X1 ITE 300 K K1024 T50, continuous FR (rate 2) | ABF | 0.078 | 0.073 | 0.026 | 0.88 | t > 25 | 0.012 | 0.033 | 0.11 |
| | FR | 0.088 | 0.084 | 0.026 | 0.91 | t > 25 | **0.035** | 0.029 | **0.58** |
| E1 ITE 300 K K1024 T50, FR stopped at 10 | ABF | 0.077 | 0.073 | 0.023 | 0.91 | t > 25 | −0.002 | 0.035 | ≈0 |
| | FR | **0.054** | 0.048 | 0.023 | 0.81 | t > 10 | 0.009 | 0.024 | 0.11 |
| A ITE 300 K K256 T200, FR stopped at 10 | ABF | 0.035 | 0.025 | 0.024 | 0.51 | t > 100 | 0.007 | 0.039 | 0.03 |
| | FR | 0.034 | 0.024 | 0.024 | 0.49 | t > 10 | 0.013 | 0.024 | 0.21 |
| M300 ITE 300 K K256 T50, continuous FR | ABF | 0.094 | 0.081 | 0.049 | 0.73 | t > 25 | 0.017 | 0.065 | 0.07 |
| | FR | 0.117 | 0.103 | 0.054 | 0.78 | t > 25 | **0.048** | 0.057 | **0.41** |
| M150 ITE 150 K K256 T50, continuous FR | ABF | 0.129 | 0.118 | 0.053 | 0.83 | t > 25 | 0.015 | 0.062 | 0.05 |
| | FR | 0.161 | 0.146 | 0.066 | 0.83 | t > 25 | **0.095** | 0.052 | **0.76** |
| X2 ITE 150 K K256 T200, continuous FR (360 bins) | ABF | 0.049 | 0.040 | 0.028 | 0.67 | t > 100 | 0.010 | 0.033 | 0.09 |
| | FR | 0.079 | 0.070 | 0.036 | 0.79 | t > 100 | **0.039** | 0.036 | **0.54** |
| E2 ITE 150 K K256 T100, FR stopped at 15 | ABF | 0.084 | 0.075 | 0.039 | 0.79 | t > 50 | 0.004 | 0.046 | 0.01 |
| | FR | 0.075 | 0.059 | 0.047 | 0.61 | t > 15 | 0.015 | 0.043 | 0.10 |
| ERI 600 K K256 T40, continuous FR | ABF | 0.141 | 0.116 | 0.079 | 0.68 | t > 20 | 0.026 | 0.107 | 0.06 |
| | FR | 0.278 | 0.260 | 0.099 | 0.87 | t > 20 | **0.130** | 0.099 | **0.63** |
| ERI 600 K K1024 T40, continuous FR | ABF | 0.122 | 0.115 | 0.038 | 0.90 | t > 20 | 0.028 | 0.046 | 0.26 |
| | FR | 0.256 | 0.252 | 0.043 | 0.97 | t > 20 | **0.127** | 0.044 | **0.89** |

**Early-time bias.** At 0.1 T, from the same JSON: FR has the smaller bias in every block. ITE 300 K K1024: 0.33 vs
0.58 (X1) and 0.32 vs 0.57 (E1). ITE 150 K K256: 0.43 vs 0.68. ERI K1024: 1.00 vs 1.10, the smallest relative gain.

### 1.2 Readings

1. **Histogram-ABF endpoints are bias-dominated, and the bias is memory.** At K1024 × T50 the final ABF error is
   88–91 % squared bias. It is not ongoing bias: ABF's late-only estimate is nearly unbiased (|bias| ≤ 0.03 kJ/mol,
   share ≤ 0.26). The final bias is therefore the **memory of the early, unestablished deposits**. Those deposits
   still sit in the cumulative accumulators M_j, C_j.
2. **FR's early gain is a reduction of that memory.** By 0.1 T, FR has about half the bias in ITE. It establishes
   the marginal sooner (§3.1), so fewer badly placed early deposits enter the accumulators.
3. **Continuous FR adds a new, ongoing bias.** Its late-only deposits are biased in every block: 2.9× (X1) to 6.3× (M150) ABF's late-only
   bias, and bias share 0.41–0.89 against 0.05–0.26. Over a long run this ongoing bias overtakes the memory reduction.
   That is why continuous FR loses at the endpoint in X1, M300, M150, X2 and both ERI blocks, even where it won early.
4. **Stopping removes the ongoing bias.** After the stop the deposits are nearly unbiased: E1 0.009, A 0.013,
   E2 0.015. These are within about 0.01 kJ/mol of ABF's late-only values, and 3–6× below the matching continuous-FR
   blocks (X1 0.035, M300 0.048, M150 0.095).
   The stopped arm keeps the memory reduction. In E1 this turns X1's endpoint loss (0.088 vs 0.078) into an endpoint
   gain (0.054 vs 0.077).
5. **Why the gain is smaller at the efficient allocation.** At K256 × T200 (block A) the ABF endpoint is only 51 %
   bias. The long trajectory dilutes the early transient's memory with 190 t.u. of nearly unbiased deposits. The term
   that FR reduces is therefore a smaller part of the endpoint error. Hence A's e_F(T) is −0.9 % [−14, +14] and its
   H1 early window is −25.5 %. The C1 / E1 endpoints are −35 % / −32 %.

## 2. D2 — clone-age split (two re-runs of existing continuous-FR blocks, bitwise identical)

* **Re-runs.** The FR arms of ERI 600 K K256 (rate 2, seeds 1111000–015) and ITE 300 K K256 (rate 1, seeds
  1121000–015) were re-run with the read-only `diag_age_steps` = 1 t.u.
* **Bitwise identity.** 32/32 runs reproduced their historical trajectories exactly (sv_C, sv_M, q_final), so the
  historical ABF arms remain valid partners.
* **The split.** A deposit is "young" if its walker was cloned within the last 1 t.u. Every other deposit is "old";
  old walkers can still be descendants of earlier clones.

### 2.1 Pooled late-half estimates (t > T/2; 16 seeds pooled; seed-jackknife s.e.)

| block | estimate from | share of late deposits | mean-force wRMS vs ref | F RMS vs ref |
|---|---|---|---|---|
| ERI 600 K K256 | young FR deposits | 15.5–17.3 % | 0.301 ± 0.057 | 0.250 |
| | old FR deposits | 83–85 % | 0.126 ± 0.019 | **0.110** |
| | all FR deposits | 100 % | 0.151 ± 0.020 | 0.132 |
| | ABF arm | 100 % | 0.086 ± 0.017 | **0.038** |
| ITE 300 K K256 | young FR deposits | 7.9–9.1 % | 0.211 ± 0.036 | 0.132 |
| | old FR deposits | 91–92 % | 0.061 ± 0.013 | **0.043** |
| | all FR deposits | 100 % | 0.069 ± 0.014 | 0.050 |
| | ABF arm | 100 % | 0.044 ± 0.009 | **0.024** |

**Caveat: sample size.** The young estimates rest on 6–12× fewer deposits than the others. Their larger RMS is
therefore mostly variance, and **the young-vs-old contrast is not evidence that clones carry the bias.** The robust
contrast is **old FR vs ABF**: 83–92 % of the deposits against 100 %, which is nearly equal variance. Old-only FR is
still 2.9× (ERI) and 1.8× (ITE) worse than ABF in F.

### 2.2 Fibre conditionals by age class (whole run, 18 coarse bins, RMS deviation from the reference)

| block | coordinate | young FR | old FR | ABF |
|---|---|---|---|---|
| ERI 600 K K256 | η (cage-axis coordinate), absolute | 0.0213 | **0.0152** | 0.0033 |
| | u_x² (transverse), relative | 0.0165 | **0.0068** | 0.0026 |
| ITE 300 K K256 | η, absolute | 0.0009 | 0.0008 | 0.0007 |
| | u_x², relative | 0.0071 | 0.0045 | 0.0038 |

### 2.3 Readings

1. **The FR bias is a property of the population, not of the newborn.** Only about 16 % (ERI) or 8 % (ITE) of
   deposits are young. Removing them closes 23 % (ERI) or 27 % (ITE) of FR's excess F error over ABF.
2. **In ERI the carrier is visible.** The walkers that were not recently cloned have a displaced η and u_x²
   conditional: 4.6× and 2.6× ABF's deviation from the exact-MC conditional. This fits ERI's mean-force error being
   limited by slow conditional (cage-axis) relaxation. Marginal reallocation acts on ξ only; by the project's theorem,
   d/dt p(y|ξ)|_FR = 0, it cannot relax p(y|ξ). Any correlation between the walkers FR selects and their fibre state
   therefore persists. Because births and deaths continue for the whole run (1562 replacements per C5 run, the most
   of any cell), the displacement is continually refreshed.
3. **In ITE the carrier is not identified.** η and u_x² are nearly undisplaced (old 0.0008 / 0.0045 vs ABF 0.0007 /
   0.0038). Yet the late mean-force bias is real: old-only FR has a flank mean-force error of +0.030 against ABF's
   +0.015, and late-only F bias of 0.035–0.048 against 0.012–0.017. The engine accumulates only two fibre
   coordinates, so the ITE carrier, for example the molecule's orientation or bond, was not measured. **The selection
   mechanism behind the conditional distortion is a hypothesis, not a result.**
4. **Possible selection mechanism (not tested here).** Late in the run, a bin falls below the FR target because
   walkers flowed out of it. The walkers that remain are biased toward fibre states that resist leaving. Cloning
   them enriches those states at that ξ. A direct test would split deposits by the walker's fibre state at birth, or
   by birth vs death bins, in a system with a known slow fibre. That was not run; it would need a new prereg.

## 3. D3 — the role of continued resampling (existing confirmatory blocks; descriptive)

### 3.1 Early establishment (FR's source of benefit)

Median window-establishment time, ABF → FR (t.u.), with the paired median difference FR − ABF [95 % CI]:

| cell | ABF → FR | paired median difference [95 % CI] |
|---|---|---|
| C1 ITE 300 K K1024 | 4.25 → 2.50 | −1.75 [−1.88, −1.63] |
| EA_R1 ITE 300 K K256 | 4.5 → 3.0 | −1.5 [−1.5, −1.0] |
| EB64 ITE 300 K K64 (rate 0.2) | 4.0 → 4.0 | **0 [0, 0]** |
| C2 ITE 150 K K256 | 7.06 → 4.19 | −3.0 [−3.6, −2.5] |
| C3 LTA 150 K | 16.8 → 9.3 | −7.4 [−7.9, −7.0] |
| C4 LTA 300 K | 9.0 → 7.3 | −1.4 [−2.0, −1.3] |
| C5 ERI 600 K | 2.1 → 1.9 | −0.2 [−0.2, −0.2] |

* **The benefit scales with the establishment delay FR removes.** Where ABF establishes slowly (LTA 150 K) the gain
  is largest (S_geo 2.52). Where ABF establishes almost at once (ERI, 2.1 t.u.) there is nothing to remove, and FR
  shows only its cost (S_geo 0.61).
* **At K64 the calibrated FR rate (0.2) establishes no faster than ABF**, and EB64 is neutral (S_geo 1.11
  [0.92, 1.33], 9/16).
* **ABF's establishment time in t.u. hardly depends on K** (4.0–4.5 t.u. at K64–K1024). This is why small K is
  cheaper in force evaluations (EFFICIENCY.md).

### 3.2 Late bias (FR's source of cost)

Continuous FR's ongoing late-window bias (§1, reading 3) is the cost. It grows with run length, which is why
continuous FR's endpoint error can exceed ABF's even where FR wins early.

### 3.3 The stop

* **Post-stop deposits are ABF-like.** The post-stop-only estimate, built only from deposits after the stop, is
  equivalent to ABF's over the same window in ITE:
  * C1 −1 % [−17, +20];
  * EA_R1 +1 % [−6, +19];
  * A +8 % [−9, +21].
  So **the stopped arm's advantage is carried by the accumulators filled before the stop**, not by better sampling
  afterwards.
* **Exception: LTA 150 K.** Its post-stop-only estimate is −49 % [−60, −33]. At its stop (21.5 t.u.) the ABF arm
  is still poorly spread (ABF establishment 16.8 t.u.; its bias force is not yet converged). The FR arm's population
  and bias force are better at the stop, and that advantage carries into the post-stop sampling.
* **Harm under R1 comes from late stops.** In ERI, R1 stops late (median 15.25 t.u., range 5.5–25.5) because the
  directedness never decays below 30–60 % of its peak. FR therefore accumulates biased deposits for 15 of 40 t.u.
  ERI's post-stop-only estimate is +11 % [−17, +23], neutral. The **damage sits in the pre-stop accumulators, and
  there was no establishment deficit to offset it.**

### 3.4 Relaxation and turnover

* **Turnover does not track benefit.** FR replacements per run: ERI 1562 (harmful); C1 611, C3 400, C4 210, C2 114
  (beneficial).
* **Genealogy stayed healthy** in all C cells: minimum windowed ESS/K ≥ 0.32, maximum lineage ≤ 0.037. The ERI harm
  is therefore not a genealogy collapse.

### 3.5 Circulation (secondary)

FR does not increase barrier crossings: true translocations, ABF / FR medians:

| cell | ABF / FR |
|---|---|
| C1 | 1595 / 1630 |
| C2 | 412 / 406 |
| C4 | 938 / 958 |
| C5 | 3617 / 3484 |
| A | 1712 / 1698 (paired −0.3 % [−1.6, +0.3]) |

Only LTA 150 K shows more crossings (328 / 386, +18 %). The acceleration therefore does not come from enhanced
circulation; it comes from faster marginal establishment.

## 4. What D does not establish

* Every D quantity is exploratory and descriptive, and comes from blocks whose seeds and settings were fixed before D.
* D2 covers two blocks (ERI and ITE at K256, continuous FR). It was not repeated for LTA, for stopped schedules or for
  K1024.
* The mechanism by which reallocation biases the fibre conditional is a hypothesis (§2.3, readings 3–4).
* No placebo or matched-turnover control was run; the campaign rules forbid them. So "the bias comes from the
  reallocation itself rather than from any population turnover" is not tested.
