# Experiment D — mechanism: why continuous FR biases late estimates, and what stopping changes

*Status: EXPLORATORY throughout (PREREGISTRATION §D). §D was written into the working file at about 07:22 UTC,
before the D2 runs were launched, but was first committed at 07:30:24 (11477fef), after those runs had finished at
07:25:35. Its timing is therefore not verifiable from git. The D2 runs are bitwise re-runs of historical arms. No
statistic here is confirmatory, and none was used to choose an FR rate, a stop time or the online rule.
Scripts: `scripts/adaptive_fr/bias_variance.py` (D1), `scripts/adaptive_fr/clone_age.py` (D2),
`scripts/adaptive_fr/selection_split.py` (D4, §2.4; preregistered in `MECHANISM_SELECTION_PREREG.md`, committed
49ae642d before its runs). Outputs: `results/adaptive_fr_campaign/mechanism/{bias_variance.json,
bias_variance_curves.npz, clone_age.json, selection_split.json}`. Figures: `figures/adaptive_fr/bias_variance.png` (6),
`mechanism_clone_age_bias.png` and `mechanism_meanforce_genealogy.png` (7), `mechanism_selection.png` (9).*

All comparisons are ABF vs the same ABF + FR (no third arm). Errors are against the exact-MC references used by every
confirmatory analysis (ITE: `results/overnight/ite{,150}/reference`; ERI: `results/eri_campaign/reference`).

## 0. Answer to Q4 in one paragraph

Continuous FR makes the **deposits it collects while it is active** biased. The late-only decomposition (D1) puts the
squared bias of FR's late-window estimate at 0.41–0.89 of its MSE. ABF's late window is variance-dominated (bias share
0.05–0.26). The absolute late-window bias is 2.7–6.4× ABF's in every continuous-FR block. Once FR is stopped, the
deposits made after the stop are close to unbiased: bias 0.009–0.014 kJ/mol, against 0.035–0.13 under continuous FR.
Over the same post-stop window, ABF's bias is 0.006–0.014, within 0.003 kJ/mol of the stopped FR arm's. The bias is **not
confined to recently cloned walkers** (D2). Dropping every deposit made within 1 t.u. of a cloning removes only part
of FR's excess late error: ERI 0.132 → 0.110 kJ/mol against ABF 0.038; ITE 0.050 → 0.043 against 0.024. In ERI the
fibre conditionals of the whole FR population are displaced from equilibrium, for walkers that were not recently
cloned too: η 4.6× and u_x² 2.6× ABF's deviation. The data therefore point to a **population-level distortion of the
conditional (fibre) distribution, maintained for as long as reallocation continues.** They do not point to a
per-clone transient, and they do not point to a residue of early deposits. Stopping FR removes the source and keeps
the early establishment gain. The selection mechanism proposed in §2.3 (reading 4) was tested directly in D4 (§2.4): the
walkers FR clones are **not** fibre-selected within their bin, and the birth–death exchange is fibre-neutral. The
late error follows the **birth region** of a lineage: cage-born lineages carry the pooled error pattern, window-born
lineages its mirror image, and the carried fibre state (η in ERI) relaxes over 1–3 t.u. Late FR makes a net
window → cage transfer, so arrivals in the flank are unbalanced. That is an arrival-law (net CV-flux) mechanism, not
selection. In ITE the same birth-region pattern holds, but none of the four measured fibre coordinates carries it.

## 1. D1 — bias–variance decomposition across seeds (existing data)

For each arm, at each save, over the 16 independent seeds:
* MSE = mean over seeds of ‖F − F_ref‖²/n_bins, with F centred;
* V = the mean per-bin variance over seeds;
* B² = ‖mean F − F_ref‖²/n_bins − V/n − s_ref².

The last term, s_ref², is the reference's own squared standard error. "Bias" below is √B², with the sign of B²; a
small negative value means noise. *Late-only* is the estimate built only from deposits after each arm's own split:
* the FR arm splits at its stop (stopped schedules) or at T/2;
* the ABF arm splits at T/2.

The windows therefore differ in the stopped blocks E1, A and E2, as the table's window column shows. For those blocks
the ABF arm also has a **matched-window** variant over the FR arm's post-stop window (`late_only_matched_to_fr_stop`
in the JSON; reading 4).

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
| | FR | 0.075 | 0.059 | 0.047 | 0.61 | t > 15 | 0.014 | 0.043 | 0.10 |
| ERI 600 K K256 T40, continuous FR | ABF | 0.141 | 0.116 | 0.079 | 0.68 | t > 20 | 0.026 | 0.107 | 0.06 |
| | FR | 0.278 | 0.260 | 0.099 | 0.87 | t > 20 | **0.130** | 0.099 | **0.63** |
| ERI 600 K K1024 T40, continuous FR | ABF | 0.122 | 0.115 | 0.038 | 0.90 | t > 20 | 0.028 | 0.046 | 0.26 |
| | FR | 0.256 | 0.252 | 0.043 | 0.97 | t > 20 | **0.127** | 0.044 | **0.89** |

**Early-time bias.** At 0.1 T, from the same JSON, FR has the smaller bias in every block. FR / ABF:

| block | FR / ABF at 0.1 T |
|---|---|
| ITE 300 K K1024 | 0.56–0.58 (X1 0.33 vs 0.58; E1 0.32 vs 0.57) |
| ITE K256 blocks (M300, M150, X2, E2, A) | 0.63–0.81 |
| ERI | 0.89–0.91, the smallest relative gain |

### 1.2 Readings

1. **Histogram-ABF endpoints are bias-dominated, and the bias is memory.** At K1024 × T50 the final ABF error is
   88–91 % squared bias. It is not ongoing bias: ABF's late-only estimate is nearly unbiased (|bias| ≤ 0.03 kJ/mol,
   share ≤ 0.26). The final bias is therefore the **memory of the early, unestablished deposits**. Those deposits
   still sit in the cumulative accumulators M_j, C_j.
2. **FR's early gain is a reduction of that memory.** By 0.1 T, FR has about half the bias in ITE at K1024, and
   0.63–0.81 of it at K256. It establishes the marginal sooner (§3.1), so fewer badly placed early deposits enter the
   accumulators.
3. **Continuous FR adds a new, ongoing bias.** Its late-only deposits are biased in every continuous-FR block: from
   2.7× (M300) to 6.4× (M150) ABF's late-only bias over the same window (both arms split at T/2 in these blocks), and
   bias share 0.41–0.89 against 0.05–0.26. Over a long run this ongoing bias overtakes the memory reduction.
   That is why continuous FR loses at the endpoint in X1, M300, M150, X2 and both ERI blocks, even where it won early.
4. **Stopping removes the ongoing bias.** After the stop, the deposits are nearly unbiased: E1 0.009, A 0.013,
   E2 0.014. Over the **same** post-stop window (matched-window variant), ABF's bias is 0.006, 0.010 and 0.014.
   Stopped FR is within 0.003 kJ/mol of ABF there, and its post-stop seed sd is similar (FR vs ABF: 0.024 vs 0.025,
   0.024 vs 0.025, 0.043 vs 0.040). The continuous-FR blocks at the same temperature and allocation sit 3–7× higher
   (X1 0.035, M300 0.048, M150 0.095).
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

### 2.2b Where in CV space the late mean-force error sits (late half, pooled, count-weighted per coarse bin)

`clone_age.json` → `signed_mean_force_error_by_coarse_bin`; figure 7b, left panels.
* **ERI.** The FR error is antisymmetric across the window. All FR deposits run from +0.12 to +0.27 on the bins
  before the window (coarse bins 3–6) and from −0.16 to −0.27 on the bins after it (11–13). The old walkers alone
  show the same pattern (+0.13 to +0.24 and −0.13 to −0.22). ABF over the same bins runs from −0.03 to +0.16 and from
  −0.19 to +0.06. FR thus systematically steepens the mean force on both flanks of the window, which is consistent
  with the displaced η conditional.
* **ITE.** The excess is on one flank only: FR old +0.10 to +0.14 in coarse bins 2–3, against ABF +0.03 to +0.07.

### 2.3 Readings

1. **The FR bias is a property of the population, not of the newborn.** Only about 16 % (ERI) or 8 % (ITE) of
   deposits are young. Removing them closes 24 % (ERI) or 27 % (ITE) of FR's excess F error over ABF.
2. **In ERI the carrier is visible.** The walkers that were not recently cloned have a displaced η and u_x²
   conditional: 4.6× and 2.6× ABF's deviation from the exact-MC conditional. This fits ERI's mean-force error being
   limited by slow conditional (cage-axis) relaxation. Marginal reallocation acts on ξ only; by the project's theorem,
   d/dt p(y|ξ)|_FR = 0, it cannot relax p(y|ξ). Any correlation between the walkers FR selects and their fibre state
   therefore persists. In these continuous-FR blocks births and deaths continue for the whole run (median 1817
   replacements per ERI run and 1120 per ITE run in the D2 re-runs), so the displacement is continually refreshed.
3. **In ITE the carrier is not identified.** η and u_x² are nearly undisplaced (old 0.0008 / 0.0045 vs ABF 0.0007 /
   0.0038). Yet the late mean-force bias is real: old-only FR has a flank mean-force error of +0.030 against ABF's
   +0.015, and late-only F bias of 0.035–0.048 against 0.012–0.017. The engine accumulates only two fibre
   coordinates, so the ITE carrier, for example the molecule's orientation or bond, was not measured. **The selection
   mechanism behind the conditional distortion is a hypothesis, not a result.**
4. **Possible selection mechanism (tested in §2.4; not supported).** Late in the run, a bin falls below the FR
   target because walkers flowed out of it. The walkers that remain are biased toward fibre states that resist
   leaving. Cloning them enriches those states at that ξ. D4 split the deposits by the walker's fibre state at birth
   and by birth / death bins (preregistered in `MECHANISM_SELECTION_PREREG.md`); the result is §2.4.


## 2.4 D4 — direct test of the selection hypothesis (both arms of the two D2 blocks re-run bitwise)

*Preregistered in `MECHANISM_SELECTION_PREREG.md` (committed 49ae642d at 11:30 UTC, runs launched at 11:30:27,
finished 11:35:19; 3.66 core-h). Exploratory. Script `scripts/adaptive_fr/selection_split.py`; JSON
`results/adaptive_fr_campaign/mechanism/selection_split.json`; figure 9 `mechanism_selection`. Post-launch analysis
changes are listed, dated, in the preregistration's §9; the one post-hoc statistic (P6) is marked below.*

**Instrument.** The engine's read-only `diag_sel` record (default off, bitwise-inert; `tests/test_diag_sel.py`, which
also checks the default path against a trajectory frozen from the pre-change engine): every FR birth / death with the
step, parent and killed slot, their fine CV bins, and the fibre state of both — η (ERI: cos(4πZ/c), +1 at window
heights, −1 at cage middles; ITE: cos(2πZ/a), +1 on the channel axis), u_x², the bond length r and u_z²; an ancestor
tag per slot (the most recent birth event that wrote into it; founders = never overwritten); per-event deposit
accumulators (C, M per fine bin, early / late half; count and f, η, u_x², r, u_z² sums per coarse bin and
time-since-birth band); per-bin fibre histograms in both arms (the ABF arm's late-half histogram defines the classes).

**Runs and gate.** ERI 600 K K256 rate 2 (seeds 1111000–015) and ITE 300 K K256 rate 1 (seeds 1121000–015), both
arms, `diag_sel = 1`. 64/64 runs reproduced their historical arms bitwise (sv_C, sv_M, sv_h, sv_i, sv_u, sv_cond,
q_final). Events per run: 1817 (ERI) and 1120 (ITE) medians, = the replacement counts; late-half event rate 46.1
(ERI) and 22.5 (ITE) per t.u. per run. Late half = t > 20 (ERI), t > 25 (ITE). Pooled over 16 seeds; seed-jackknife
s.e. throughout (every pooled law and class assignment is recomputed per replicate).

### 2.4.1 P0 — which fibre coordinate is displaced (late-half population conditional, FR − ABF)

Count-weighted RMS over 18 coarse bins of (⟨y | bin⟩_FR − ⟨y | bin⟩_ABF) / sd_ABF(bin), with a floor of 1 % of the
coordinate's range under the sd; "floor" = the ABF split-half estimate of the same statistic's noise.

| block | η | u_x² | bond r | u_z² |
|---|---|---|---|---|
| ERI | **0.0157 ± 0.0064** (floor 0.0052; natural 0.0095) | 0.0049 ± 0.0012 (0.0033) | 0.0007 ± 0.0001 (0.0004) | 0.0027 ± 0.0009 (0.0032) |
| ITE | degenerate (η ≈ 1 with sd ≈ 0 over the window side; 0.0048 ± 0.0033) | 0.0044 ± 0.0013 (0.0029) | 0.0005 ± 0.0001 (0.0006) | 0.0035 ± 0.0011 (0.0022) |

* **ERI.** η is the carrier, as preregistered (D2). Its per-bin displacement is negative on both flanks (coarse bins
  3–4: −0.028; 13–14: −0.031 / −0.028 sd): the FR population on the flanks sits at **lower η, i.e. cage-like
  heights**. It is small in absolute terms (1.6 % of a sd), and the P0 rule's 3 s.e. threshold is not met (2.5 s.e.).
* **ITE.** u_x² is the largest and formally qualifies (3.4 s.e., above the floor), but at 0.4 % of a sd. The bond
  length is undisplaced (and relaxes in ≈ 3 steps, so it cannot carry anything). u_z² 0.35 % of a sd. The ITE
  carrier is therefore **not among the four coordinates** in any useful sense; §2.4.6 confirms this.

### 2.4.2 P1 — are the cloned walkers fibre-selected? (no)

u′ = quantile of the parent's fibre value, at birth, within the FR population's own late-half law at its bin
(mid-rank; the exact null for a random draw is tabulated from the same histograms); u = the same within the ABF law;
v′ = the killed walker's quantile. "Tail" = share with u′ < 0.25 or ≥ 0.75.

| block, coordinate | births: mean u′ (null 0.5) | births: tail (null) | births vs ABF law: mean u | deaths: mean v′ | net exchange (births − deaths) |
|---|---|---|---|---|---|
| ERI η | 0.4974 ± 0.0017 | 0.4144 ± 0.0033 (0.4091) | 0.4949 ± 0.0021 | 0.4948 ± 0.0015 | **+0.0026 ± 0.0020** |
| ERI u_x² | 0.5007 ± 0.0017 | 0.5016 ± 0.0031 (0.5033) | 0.4998 ± 0.0017 | 0.4978 ± 0.0015 | +0.0029 ± 0.0022 |
| ERI r | 0.5011 ± 0.0015 | 0.4846 ± 0.0036 (0.4896) | 0.5011 ± 0.0015 | 0.4995 ± 0.0022 | +0.0016 ± 0.0031 |
| ERI u_z² | 0.5003 ± 0.0015 | 0.5242 ± 0.0030 (0.5251) | 0.5006 ± 0.0014 | 0.5009 ± 0.0012 | −0.0006 ± 0.0021 |
| ITE u_x² | 0.4983 ± 0.0023 | 0.4782 ± 0.0043 (0.4803) | 0.4984 ± 0.0024 | 0.5015 ± 0.0022 | −0.0032 ± 0.0038 |
| ITE r | 0.4988 ± 0.0025 | 0.4865 ± 0.0034 (0.4928) | 0.4988 ± 0.0025 | 0.4964 ± 0.0019 | +0.0024 ± 0.0031 |
| ITE u_z² | 0.5005 ± 0.0017 | 0.4443 ± 0.0019 (0.4391) | 0.5006 ± 0.0015 | 0.5022 ± 0.0021 | −0.0017 ± 0.0027 |

Moment form (exact deposit-sum means; sd units): ERI η births −0.026 ± 0.010, deaths −0.024 ± 0.007, FR population
at the birth bins −0.012 ± 0.006; every other coordinate within ±0.012 of zero. By birth region (ERI η, mean u′):
window 0.5001 ± 0.0017, flank 0.497 ± 0.003, cage 0.497 ± 0.004. Late births alone: tail 0.4079 (null 0.4091).

* The parents FR clones are a **fibre-neutral draw from the population at their bin**: no coordinate deviates from
  the null by more than 2.4 s.e. in rank, and the walkers FR kills are displaced by the same amount as the ones it
  clones (η: −0.024 vs −0.026 sd), so the birth–death exchange moves the population's η by **+0.003 ± 0.002**
  rank units — in the direction *opposite* to the population's displacement.
* Sustaining the observed −0.012 sd displacement by selection would need a net exchange of about −0.0035 rank units
  (−0.01 once the 1–3 t.u. relaxation of §2.4.4 and the 46 births per t.u. in 256 walkers are allowed for). The
  measured net exchange excludes that at 3–6 s.e. **P1 is null.** The small displacement of the births relative to
  the ABF law (−0.005 rank units, −0.026 sd) is the population's own displacement, which the births inherit.

### 2.4.3 P2 — late-half estimates by the ancestor's fibre class at birth (ERI, η classes vs the ABF law)

| estimate from late FR deposits of … | share of late FR deposits | mean-force wRMS vs ref | F RMS vs ref | Γ − Γ_ref, coarse bin 4 / 13 |
|---|---|---|---|---|
| founders (slot never overwritten) | 0.8 % | 0.82 ± 0.18 | 0.59 ± 0.28 | +0.90 / −1.30 |
| ancestor's η in **Q1** (lowest quarter of the ABF law at the birth bin) | 22.7 % | **0.86 ± 0.06** | 0.64 ± 0.04 | **+1.93 / −2.28** |
| Q2 | 17.0 % | 0.42 ± 0.03 | 0.34 ± 0.03 | +0.26 / −0.36 |
| Q3 | 41.1 % | 0.35 ± 0.02 | 0.27 ± 0.02 | **−0.74 / +0.83** |
| Q4 (highest quarter) | 18.4 % | 0.38 ± 0.04 | 0.25 ± 0.04 | +0.19 / +0.18 |
| tail Q1 ∪ Q4 | 41.2 % | 0.54 ± 0.04 | 0.45 ± 0.03 | +1.13 / −1.18 |
| central Q2 ∪ Q3 | 58.1 % | 0.18 ± 0.02 | 0.11 ± 0.02 | −0.43 / +0.46 |
| all descendants | 99.2 % | 0.148 ± 0.020 | 0.129 ± 0.017 | +0.24 / −0.23 |
| all FR deposits (= D2) | 100 % | 0.151 ± 0.020 | 0.132 ± 0.017 | +0.25 / −0.24 |
| ABF arm | 100 % | 0.086 ± 0.017 | 0.038 ± 0.016 | −0.01 / +0.06 |
| variance-matched ABF: 4-seed pools (≈ ¼ of the deposits) / 8-seed pools | | 0.133 / 0.113 | 0.055 / 0.047 | |

Classes defined against the FR population's own law instead: tail 0.535 ± 0.036, central 0.173 ± 0.022 (same). Lineage
born before T/2: 26 % of late deposits, wRMS 0.138 ± 0.023; born after: 73 %, 0.166 ± 0.024.

* **Every class is biased far beyond the variance-matched ABF value**, and the biases have **opposite signs**: lineages
  born with η below the median of their bin's law steepen the mean force on both flanks (Q1: +1.93 / −2.28), lineages
  born with η above the median flatten it (Q3: −0.74 / +0.83). The pooled FR bias (+0.25 / −0.24) is the incomplete
  cancellation of class biases ten times larger.
* The preregistered P2 criterion (tail excess ≥ 2 × central excess) is formally met (0.45 vs 0.09 above ABF), but
  the reading it was written for — "the tail classes carry the error, the central classes are clean" — is wrong:
  the central group is itself biased (0.18 against 0.11 variance-matched), in the opposite direction.
* The quartile labels are blurred by the shape of η's law, which is piled against η = 1 on the window side (Q3 and
  Q4 share the top histogram bin), so below / above the median is the robust split; §2.4.6 uses it.
* **ITE** (u_x² classes, the P0 pick): Q1–Q4 0.109 ± 0.018, 0.097 ± 0.029, 0.090 ± 0.019, 0.087 ± 0.020 (shares
  22–25 %); tail 0.072 ± 0.018 vs central 0.078 ± 0.022; all FR 0.069 ± 0.014; ABF 0.044 ± 0.009; variance-matched
  ABF 0.075 / 0.055. **No class contrast**: u_x² does not carry ITE's bias. u_z² gives tail 0.120 ± 0.028 vs central
  0.052 ± 0.008 (F 0.087 ± 0.019 vs 0.019 ± 0.013), a 2–3 s.e. contrast with a null P1 (mean u′ 0.5005 ± 0.0017)
  among four coordinates and several splits; §2.4.6 shows it is not a carrier either. Lineages born before T/2:
  36 % of late deposits, wRMS 0.065 ± 0.008 (F 0.027 ± 0.008); born after: 60 %, 0.096 ± 0.019 (F 0.069 ± 0.017).

### 2.4.4 P3 — the carried fibre state relaxes with time since birth

Count-weighted RMS over coarse bins of |⟨η | bin⟩_class,band − ⟨η | bin⟩_ABF| / sd_ABF, late half, ERI:

| time since birth (t.u.) | < 0.1 | 0.1–0.3 | 0.3–1 | 1–3 | 3–10 | ≥ 10 |
|---|---|---|---|---|---|---|
| tail classes Q1 ∪ Q4 | 0.368 ± 0.018 | 0.322 ± 0.021 | 0.256 ± 0.022 | 0.129 ± 0.016 | 0.018 ± 0.010 | 0.016 ± 0.012 |
| central classes Q2 ∪ Q3 | 0.278 ± 0.005 | 0.252 ± 0.007 | 0.184 ± 0.008 | 0.063 ± 0.011 | 0.009 ± 0.006 | 0.011 ± 0.010 |
| signed mean, tail / central | −0.37 / +0.23 | −0.30 / +0.18 | −0.22 / +0.13 | −0.11 / +0.05 | −0.01 / 0.00 | −0.01 / −0.01 |

* Descendants carry their birth η for a long time: the deviation falls by e over ≈ 2 t.u. and is gone by 3–10 t.u.
  This is ERI's slow cage-axis relaxation measured directly. It is **present in both class groups**, with opposite
  sign (the tail group is dominated by Q1, low η; the central group by Q3, high η), so it is not a property of
  "selected" births but of **every** birth: a clone starts with its parent's fibre state and keeps it for ≈ 2 t.u.
* ITE (u_x²): 0.099 ± 0.016 → 0.037 → 0.025 → 0.016 → 0.008 (tail) and 0.072 ± 0.009 → 0.034 → 0.018 → 0.010 → 0.011
  (central): the orientation relaxes within ≈ 0.3 t.u., too fast to carry a late bias.

### 2.4.5 P4 — the late error follows the ancestor's BIRTH REGION (figure 9, left column)

Late-half error from the deposits of lineages born in each region (window |φ| < 0.5, flank, cage |φ| ≥ 2):

| block | born in | share of late FR deposits | wRMS | F RMS | Γ − Γ_ref at coarse bins 4–5 / 12–13 (ERI), 2–3 / 12–14 (ITE) |
|---|---|---|---|---|---|
| ERI | window | 13.6 % | 0.65 ± 0.07 | 0.55 ± 0.06 | **−1.17, −1.36 / +0.85, +1.04** |
| | flank | 46.6 % | 0.156 ± 0.025 | 0.063 ± 0.013 | −0.17, +0.04 / +0.07, +0.28 |
| | cage | 39.0 % | 0.63 ± 0.05 | 0.57 ± 0.03 | **+1.18, +1.08 / −1.23, −1.25** |
| | all FR | 100 % | 0.151 | 0.132 | +0.25, +0.22 / −0.27, −0.24 |
| | ABF | 100 % | 0.086 | 0.038 | −0.01, +0.03 / −0.19, +0.06 |
| ITE | window | 13.6 % | 0.22 ± 0.04 | 0.17 ± 0.03 | **−0.34, −0.73** / +0.15, +0.16, +0.13 |
| | flank | 43.5 % | 0.136 ± 0.020 | 0.090 ± 0.011 | −0.22, −0.36 / +0.09, +0.11, +0.20 |
| | cage | 38.8 % | 0.32 ± 0.03 | 0.25 ± 0.02 | **+0.46, +0.94** / −0.31, −0.31, −0.29 |
| | all FR | 100 % | 0.069 | 0.050 | +0.11, +0.18 / −0.04, −0.05, −0.03 |
| | ABF | 100 % | 0.044 | 0.024 | +0.07, +0.03 / −0.05, 0.00, 0.00 |

By the region of the walker that was *killed* at the event: ERI 0.235 / 0.172 / 0.111 (window / flank / cage), ITE
0.096 / 0.091 / 0.101 — a weak dependence. Deposits inside the ancestor's own birth coarse bin (7.6 % / 8.5 %) have
wRMS 0.38 ± 0.05 / 0.22 ± 0.03 against 0.150 / 0.064 outside it.

* **Cage-born lineages carry the pooled error pattern, five times amplified; window-born lineages carry its mirror
  image; flank-born lineages are nearly clean** (ERI F RMS 0.063, close to ABF's variance-matched 0.055). This holds
  in **both** systems, including ITE where no fibre coordinate is displaced.
* Where the two arrival directions meet — the flank — a lineage's error depends on where it came from. In
  equilibrium sampling the two directions balance exactly. The pooled FR error is the imbalance: cage-born deposits
  outnumber window-born deposits 2.9 : 1 (ERI) and 2.8 : 1 (ITE), against region sizes of 2.25 : 1, so the +1.2 and
  −1.2 patterns do not cancel (ERI: 0.39 × 1.18 − 0.14 × 1.17 − 0.47 × 0.17 ≈ +0.22, the observed +0.25).
* **The late birth–death balance makes the transfer explicit.** Births − deaths per t.u. per walker, late half:
  ERI window **−0.0067 ± 0.0013**, flank −0.0033 ± 0.0019, cage **+0.0100 ± 0.0023**; ITE window −0.0022 ± 0.0007,
  flank −0.0034 ± 0.0012, cage +0.0055 ± 0.0013. Late in the run FR clones in the cage and kills in the window, and
  the compensating return flow runs cage → window through the flank.

### 2.4.6 P6 (post hoc, not preregistered) — birth region × fibre class: what carries the error

Late error of lineages born in a region, split by whether the ancestor's η (ERI) or u_x² (ITE) at birth was below or
above the median of the ABF law at its birth bin (mean Γ − Γ_ref over coarse bins 3–6 / 11–14):

| block | born in | below the median | above the median |
|---|---|---|---|
| ERI (η) | window | 1 % of deposits: +… / … (too few) | 12.5 %: wRMS 0.64 ± 0.07, **−0.88 / +0.78** |
| | flank | 19 %: 0.36 ± 0.04, **+0.45 / −0.45** | 28 %: 0.36 ± 0.03, **−0.28 / +0.48** |
| | cage | 20 %: **0.97 ± 0.06, +1.17 / −1.40** (Q1 alone: 1.54, +1.72 / −2.05) | 19 %: 0.41 ± 0.04, +0.53 / −0.40 |
| ITE (u_x²) | window | 11.5 %: 0.24 ± 0.04, −0.29 / +0.13 | 2 %: 0.26 ± 0.07, −0.10 / +0.10 |
| | flank | 23 %: 0.13 ± 0.02, −0.07 / +0.16 | 20 %: 0.17 ± 0.04, −0.06 / +0.05 |
| | cage | 19 %: 0.34 ± 0.04, +0.43 / −0.29 | 20 %: 0.30 ± 0.03, +0.38 / −0.22 |

(ITE by quartile, cage-born: Q1–Q4 wRMS 0.36, 0.34, 0.31, 0.32 with the same +0.34…0.43 / −0.18…0.35 pattern; the same
holds with u_z² classes.)

* **ERI: η is the carrier.** Among flank-born lineages the η class *flips the sign* of the error: a flank birth with
  cage-like η (below the median) behaves like a cage-born lineage, one with window-like η like a window-born lineage.
  Among cage-born lineages the lowest-η births carry 3× the error of the rest. The fibre state at birth, not the
  birth bin itself, predicts the error — and the fibre state at birth records where the walker came from.
* **ITE: none of the four coordinates is the carrier.** Within every birth region the class leaves the error
  unchanged (cage-born: 0.34 vs 0.30, same pattern). The birth-region dependence is as sharp as in ERI, so the same
  mechanism acts through a coordinate that is not measured (candidates: the molecule's transverse COM position in the
  channel cross-section, or the sub-bin φ position of an arriving walker, which needs no fibre coordinate at all).

### 2.4.7 P5 — by time since birth (coarse-level wRMS, late half)

| block | band (t.u.) | < 0.1 | 0.1–0.3 | 0.3–1 | 1–3 | 3–10 | ≥ 10 | ABF / all FR (coarse) |
|---|---|---|---|---|---|---|---|---|
| ERI | all FR (share) | 0.43 ± 0.10 (1.8 %) | 0.37 ± 0.07 (3.5 %) | 0.29 ± 0.07 (11 %) | 0.136 ± 0.029 (26 %) | 0.131 ± 0.028 (42 %) | 0.136 ± 0.037 (15 %) | 0.081 / 0.151 |
| | tail / central | 2.42 / 1.17 | 1.93 / 0.76 | 1.37 / 0.54 | 0.67 / 0.27 | 0.19 / 0.11 | 0.17 / 0.19 | |
| ITE | all FR (share) | 0.30 ± 0.08 (0.9 %) | 0.33 ± 0.07 (1.7 %) | 0.18 ± 0.04 (5.8 %) | 0.159 ± 0.030 (15 %) | 0.100 ± 0.017 (36 %) | 0.093 ± 0.009 (37 %) | 0.092 / 0.099 |

Deposits older than 3 t.u. (57 % in ERI) still sit 1.6× above ABF's coarse error, as D2 found; the young bands are
3–5× worse but hold 17 % of the deposits. Within every band the tail classes are worse than the central ones in ERI,
consistent with §2.4.4: the carried η decays with age but is never zero while births continue.

### 2.4.8 Verdict

* **By the preregistered rule: UNRESOLVED** (P1 null; P2, P3 and P4 strongly non-null). The rule's "supported"
  needed fibre selection at birth; its "not supported" needed everything null.
* **Plain reading.** The selection hypothesis of §2.3 reading 4 — *FR clones walkers whose fibre state resists
  leaving, and that enriches those states* — is **not supported**: the cloned walkers are a fibre-neutral draw from
  the population at their bin, and the exchange (births minus deaths) is neutral to 0.003 rank units, in the wrong
  direction. What *is* supported is the second half of the hypothesis, that an unrelaxed fibre state is carried by
  descendants and biases their deposits (ERI: η, relaxing over ≈ 2 t.u.). Its source is **where** FR clones and
  kills, not **which** walkers: late FR performs a net window → cage transfer (births − deaths per t.u. per walker
  −0.007 in the window, +0.010 in the cage), the return flow runs cage → window through the flank, and arrivals from
  the cage (cage-like η, mean force too steep) outnumber arrivals from the window (window-like η, too flat) by
  2.9 : 1. This is the **arrival-law / net CV-flux mechanism** named in advance as alternative 7.2 in the
  preregistration. It needs only two ingredients that both systems have: births and deaths concentrated in
  different CV regions, and a conditional that relaxes more slowly than a walker crosses a region.
* **ITE.** The same birth-region signature (cage-born +0.94 / window-born −0.73 at coarse bin 3, mirror images), but
  none of η (degenerate on the channel axis), u_x² (0.4 % of a sd, no class contrast), the bond (undisplaced,
  relaxes in steps) or u_z² carries it. **The ITE carrier remains unmeasured**; the candidates are the transverse COM
  position and the sub-bin φ position of arriving walkers.
* **What would test the mechanism directly.** A per-walker record of the bin-entry time and entry side, so that the
  flank conditional and the flank mean force can be split by arrival direction and residence time in *both* arms
  (ABF should show the two directions balancing exactly; FR should show the imbalance scaling with the net source).
  For ITE, add the transverse COM coordinates to the fibre record.
* **Caveats.** Exploratory, two blocks, continuous FR at K256 only; the class estimates rest on 10–40 % of the
  deposits each (variance-matched ABF controls given); quartile labels are blurred for a law piled against the edge
  of its range (below / above median is the robust split); the P6 cross-split was not preregistered; the ITE P0
  carrier rule picked u_x² formally, at 0.4 % of a sd.

## 3. D3 — the role of continued resampling (existing confirmatory blocks; descriptive)

### 3.1 Early establishment (FR's source of benefit)

Median window-establishment time, ABF → FR (t.u.), with the paired median difference FR − ABF [95 % CI]. The
establishment time is the first **save** at which the criterion holds, so its resolution is the save interval.

| cell | save interval (t.u.) | ABF → FR | paired median difference [95 % CI] |
|---|---|---|---|
| C1 ITE 300 K K1024 | 0.125 | 4.25 → 2.50 | −1.75 [−1.88, −1.63] |
| EA_R1 ITE 300 K K256 | 0.5 | 4.5 → 3.0 | −1.5 [−1.5, −1.0] |
| EB64 ITE 300 K K64 (rate 0.2) | **2** | 4.0 → 4.0 | 0 [0, 0]: a gain < 2 t.u. is not resolvable |
| EB16 ITE 300 K K16 (rate 1.0) | **8** | 8.0 → 8.0 | 0 [0, 0]: both arms established by the first save; uninformative |
| C2 ITE 150 K K256 | 0.125 | 7.06 → 4.19 | −3.0 [−3.6, −2.5] |
| C3 LTA 150 K | 0.1 | 16.8 → 9.3 | −7.4 [−7.9, −7.0] |
| C4 LTA 300 K | 0.1 | 9.0 → 7.3 | −1.4 [−2.0, −1.3] |
| C5 ERI 600 K | 0.1 | 2.1 → 1.9 | −0.2 [−0.2, −0.2] |

* **S_geo roughly follows the establishment delay that FR removes** (rows where it is resolved): LTA 150 K, −7.4 t.u.
  → 2.52; ITE 150 K, −3.0 → 1.64; ITE 300 K K1024, −1.75 → 1.67; K256, −1.5 → 1.31; LTA 300 K, −1.4 → 1.46; ERI,
  −0.2 → 0.61.
  * This is a cross-cell correlation, not a controlled test.
  * It does **not** hold for ABF's establishment time itself: LTA 300 K establishes at 9.0 t.u. but gains less than
    ITE K1024 at 4.25.
  * In ERI there is almost nothing to remove, and FR shows its cost.
* **K64 and K16 cannot be read this way.** Their save grids (2 and 8 t.u.) are too coarse to measure an establishment
  gain of the size seen at K256 (1.5 t.u.). Their S_geo are unresolved: 1.11 [0.92, 1.33] and 0.93 [0.70, 1.29].
* **ABF's establishment time in t.u. hardly depends on K where it is resolved** (4.25–4.5 t.u. at K256–K1024). This is
  consistent with ABF's time to accuracy in t.u. being nearly K-independent for K ≥ 64 (EFFICIENCY.md §3).

### 3.2 Late bias (FR's source of cost)

Continuous FR's ongoing late-window bias (§1, reading 3) is the cost. It grows with run length, which is why
continuous FR's endpoint error can exceed ABF's even where FR wins early.

### 3.3 The stop

* **Post-stop deposits show no resolved difference from ABF in ITE 300 K.** The post-stop-only estimate is built
  only from deposits after the stop and compared with ABF's over the same window:
  * C1 −1 % [−17, +20];
  * EA_R1 +1 % [−6, +19];
  * A +8 % [−9, +21];
  * ITE 150 K (C2): −16 % [−29, +1], favouring FR, borderline.

  These CIs are about ±20 % and no equivalence margin was set, so "no resolved difference" is all they license. Most
  of the stopped arm's advantage is therefore carried by the accumulators filled before the stop. Better sampling
  afterwards, if any, is not resolved, except in LTA 150 K below.
* **Exception: LTA 150 K.** Its post-stop-only estimate is −49 % [−60, −33]. At its stop (21.5 t.u.) the ABF arm
  is still poorly spread (ABF establishment 16.8 t.u.; its bias force is not yet converged). The FR arm's population
  and bias force are better at the stop, and that advantage carries into the post-stop sampling.
* **Harm under R1 comes from late stops.** In ERI, R1 stops late (median 15.25 t.u., range 5.5–25.5) because the
  directedness plateaus at 30–60 % of its peak, above the 25 % threshold, for 5–10 t.u. after the early peak; t1 is
  7–11 t.u. (CROSS_SYSTEM_RESULTS.md §5). Every ERI run did eventually stop as "established". FR therefore runs for a
  median 15 of 40 t.u. ERI's post-stop-only estimate is +11 % [−17, +23], unresolved. The **damage sits in the
  pre-stop accumulators, and there was almost no establishment deficit to offset it** (−0.2 t.u.).

### 3.4 Relaxation and turnover

* **Turnover does not track benefit.** FR replacements per run: ERI 1562 (harmful); C1 611, C3 400, C4 210, C2 114
  (beneficial).
* **Genealogy stayed healthy** in all C cells. Across cells, the median over runs of the per-run minimum windowed
  ESS/K is ≥ 0.32, and of the per-run maximum lineage ≤ 0.037. The ERI harm is therefore not a genealogy collapse.

### 3.4b Genealogy over time in the C cells (figure 7b, right panels)

* **Ancestor diversity dips only while FR is active.** The windowed ancestor ESS/K (median over seeds) dips to
  0.32–0.45 in ITE (C1, C2) during the first 2–5 t.u., and stays ≥ 0.6 in LTA and ERI.
* **Lineage concentration is negligible:** maximum windowed lineage share ≤ 0.04 everywhere, against R1's backstop of
  0.10.
* **Both recover once R1 stops**, with ESS/K returning to 1 within one genealogy window.
* So the stopped schedules never approached genealogy collapse. ERI's harm happens with a healthy genealogy, which
  agrees with §3.4.

### 3.5 Circulation (secondary)

In the ITE and ERI cells FR does not change barrier crossings: true translocations, ABF / FR medians:

| cell | ABF / FR |
|---|---|
| C1 | 1595 / 1630 |
| C2 | 412 / 406 |
| C4 | 938 / 958 |
| C5 | 3617 / 3484 |
| A | 1712 / 1698 (paired −0.3 % [−1.6, +0.3]) |

LTA 150 K, the cell with the largest gain, shows 18 % more crossings (328 / 386). In ITE and ERI the acceleration
(or harm) therefore does not come from changed circulation. In LTA 150 K, circulation may contribute alongside faster
establishment; the two were not separated.

## 4. What D does not establish

* Every D quantity is exploratory and descriptive, and comes from blocks whose seeds and settings were fixed before D.
* D2 and D4 cover two blocks (ERI and ITE at K256, continuous FR). They were not repeated for LTA, for stopped
  schedules or for K1024.
* The selection mechanism of §2.3 reading 4 is not supported (§2.4); the arrival-law mechanism that fits the D4 data
  was named in advance as an alternative but its direct test (bin-entry record) has not been run. The ITE carrier is
  not identified.
* No placebo or matched-turnover control was run; the campaign rules forbid them. So "the bias comes from the
  reallocation itself rather than from any population turnover" is not tested.
