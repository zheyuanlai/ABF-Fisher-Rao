# Final synthesis — efficient and adaptive Fisher–Rao acceleration of ABF (campaign of 2026-10-17)

* **What was compared.** Every adaptive comparison had exactly two arms: histogram ABF, and the identical ABF + marginal
  Fisher–Rao birth–death. No sham, placebo, matched-turnover, oracle-target, OPES or metadynamics arm was run.
* **Statistics.**
  * The seed is the unit of inference: 16 paired seeds per cell, bitwise identical before FR start.
  * The primary statistic is the median per-seed geometric-mean restricted sustained speedup S_geo, with a bootstrap
    CI.
  * Cross-K comparisons are unpaired.
* **Status labels.** Confirmatory results carry their preregistered verdicts. Everything labelled exploratory or
  descriptive is so.
* **Where the details are.**
  * EQUAL_BUDGET_RESULTS.md: A.
  * ONLINE_STOPPING_RULE.md: B.
  * CROSS_SYSTEM_RESULTS.md: C.
  * MECHANISM.md: D.
  * EFFICIENCY.md: E.
  * PREREGISTRATION.md: every frozen rule and amendment.

## Headline

1. **Early-stopping FR is a reliable fixed-K accelerator where ABF is establishment-limited.** The frozen online rule
   R1 stops FR without a reference and gave S_geo:

   | cell | S_geo |
   |---|---|
   | ITE 300 K, K1024 | 1.67 |
   | ITE 300 K, K256 | 1.31 |
   | ITE 150 K | 1.64 |
   | held-out LTA 150 K | 2.52 |
   | held-out LTA 300 K | 1.46 |

2. **It was harmful in the one held-out cell where ABF establishes almost at once.** At ERI 600 K, S_geo was 0.61 and
   FR was slower in all 16 seeds. In the small-K ITE cells (K64, K16) it was unresolved, not harmful.
3. **No allocation of ABF + FR was shown to beat the best tested allocation of plain ABF**, at equal force evaluations
   or in parallel wall-clock.
   * On the eight configurations K16–K1024, best FR / best ABF is 1.04 [0.74, 1.33] in force evaluations (1.04
     [0.78, 1.28] with the selection of the best K re-done in every bootstrap replicate, EFFICIENCY.md §7) and
     1.03–1.07 in wall-clock, all unresolved.
   * With the frontier extended to K8, K4 and K1 (amendment E-A3, 2026-10-17; EFFICIENCY.md §8) the best ABF allocation
     moves to K1 and the best FR allocation stays at K16: **1.10, fixed selection [0.75, 1.51], nested [0.87, 1.43],
     UNRESOLVED.** The point estimates are on ABF's side; the CIs allow FR to be from 13 % cheaper to 43 % dearer.
   * ABF's cost is flat from K1 to K64 within the resolution of 16 unpaired seeds; the frontier has no resolved minimum
     to bracket. FR cannot run below K8 under the genealogical safety rules, and at K8 it neither shortens establishment
     (4.5 → 4.5 t.u., resolved at 1 t.u.) nor changes the error (S_geo 1.03 [0.72, 1.42]).
4. **What limits FR is largely characterised (exploratory).**
   * FR shortens the early establishment transient; a longer run with fewer replicas also dilutes that transient, at
     lower cost.
   * Continuous FR adds an ongoing bias, which stopping removes.
   * What carries that bias is identified in ERI only.

## The eight questions

### 1. At ITE 300 K, K256 × T200 (the efficient allocation of the earlier campaign), does early-stopping FR improve ABF?

**Fixed stop at 10 t.u. (A, rate 1.0): not resolved.**
* S_geo 1.28 [0.90, 1.43]; FR faster in 11/16 seeds.
* The early error is resolvably lower: H1 window −25.5 % [−29.9, −16.3], 16/16.
* The post-ramp integrated error is lower: −15 % [−22, −4].
* The final error is unchanged: −1 % [−14, +14].

**Online stop R1 (EA_R1, same ABF runs): yes.**
* S_geo 1.31 [1.07, 1.47] (99 % [1.02, 1.69]); FR faster in 13/16.
* R1 stopped earlier than the fixed schedule (median 6.5 t.u., range 5.5–9).
* Post-ramp −17 %; final error −9 % [−21, +6].

The improvement is modest and concentrated early.

### 2. Does it beat the best ABF allocation at equal force evaluations?

**No resolved advantage; the point estimate favours ABF.**
* **The frontier.** The eight-configuration frontier at B = 5.12 × 10⁸ (K ∈ {1024, 256, 64, 16} × {ABF, FR-R1}) finds
  the best allocation of both algorithms at K16.
* **Best FR / best ABF = 1.04 [0.74, 1.33] → UNRESOLVED.** The all-in variant gives 1.04 [0.75, 1.31], and e_F at the
  budget gives 1.13 [0.80, 1.40].
* **K256 is far from optimal for ABF.** ABF K64 costs 0.26× ABF K256.
* **FR's paired fixed-K advantage becomes unresolved as K approaches ABF's best tested allocation.** S_geo is 1.67
  at K1024, 1.31 at K256, 1.11 [0.92, 1.33] at K64 and 0.93 [0.70, 1.29] at K16.
* **The frontier is not bracketed (as of E-A2).** K16 vs K64 is unresolved for both algorithms, and nothing below K16
  was tested. Superseded by E-A3, below.
* **The K16 cell was prompted by the data.** It was added by amendment E-A2 because the six-configuration frontier
  put both optima at K64, then the smallest K tried. It was committed before any K16 run.

**Update, 2026-10-17 (nested bootstrap and amendment E-A3; EFFICIENCY.md §7–8).**
* **The published best-vs-best CI omitted selection uncertainty** (the best K was chosen on the scoring data and held
  fixed in the bootstrap). Re-selecting the best K in every replicate gives 1.04 [0.78, 1.28] on the eight
  configurations: not wider, because K16 is re-selected for both algorithms in 87 % of replicates and the minimum over K
  compresses the upper tail. Verdict unchanged.
* **The frontier was extended to K8 (both arms, rate 1.0 from the blind calibration), K4 and K1 (ABF only), all saving
  every 1 t.u.** Best ABF = K1 (0.170 × 10⁸), best FR = K16 (0.188): best FR / best ABF = 1.10, fixed [0.75, 1.51],
  nested [0.87, 1.43], UNRESOLVED; P(ratio < 1) = 0.19. ABF K1 / ABF K16 = 0.94 [0.62, 1.38]; FR K8 / FR K16 = 1.06
  [0.75, 1.37]; ABF K4 / K16 = 1.32 [0.83, 1.66]. Nothing in K ≤ 64 is resolvably different from anything else in
  K ≤ 64: ABF's cost is 0.17–0.24 × 10⁸ at every K from 1 to 64. The frontier is explored to its floor for ABF (K1) and
  to its safety floor for FR (K8); it has a flat minimum, not a bracketed optimum.
* **Why no K4 FR arm.** R1's genealogical backstops (lineage > max(0.10, 6/K), ESS/K < 0.15) cannot fire at K4 (1.5 and
  0.25), so the preregistered criterion for a K4 FR arm failed by arithmetic. The diagnostic K4 calibration showed
  0–10 replacements in 50 t.u. and one family holding half the population for 13–25 % of saves. The safety limits were
  not relaxed.
* **Caveat on the statistic at small K.** The ladder's tightest rung (0.080 kJ/mol) is reached in the first 3–5 % of
  the budget for K ≤ 16, so the small-K frontier is a frontier of the early transient; by the end-of-budget error ABF
  K1 is the worst of K ≤ 64 (0.027 kJ/mol) and FR K16's error is lower than ABF K1's (0.84 [0.59, 1.13]). A
  matched-save-grid re-read (the new cells subsampled to the earlier cells' 1.28 × 10⁶-force-evaluation grid) moves
  every point estimate further to ABF's side (best FR / best ABF 1.27, nested [0.89, 1.56]) and resolves nothing.

### 3. Can an online, reference-free rule decide when to stop FR?

**Yes, within its domain.**
* **Inputs.** R1 uses only FR's own birth/death flux over 18 coarse regions, a noise-scaled directedness floor, bin
  support and estimate stability.
* **Leakage.** It has no access to F_ref, F′, the ABF arm or future states; an AST test enforces this.
* **Stop times.** R1 stopped every run (128 R1 runs in the confirmatory cells, none never-stopped, none by the safety
  backstop). The stop time adapted to the system: 5.5 t.u. at ITE 300 K, 8.25 at ITE 150 K, 16 at LTA 300 K and 21.5
  at LTA 150 K.
* **Benefit kept.** It matched or exceeded the fixed schedules it replaced: C1 1.67 vs E1 1.59; C2 1.64 vs E2 1.25
  (other seeds).
* **It cannot tell when FR should not run at all.** At ERI it stopped late (median 15.25 t.u., range 5.5–25.5).
  After its early peak the directed flux plateaued at 30–60 % of that peak, above the 25 % threshold, for 5–10 t.u.
  All 16 ERI runs eventually stopped as "established".

### 4. Does one frozen rule generalise across systems (LTA 150 K, ERI)?

**Partly; by the preregistered definition, no.**
* **LTA 150 K, the held-out positive control: generalises.** S_geo 2.52 [2.33, 2.60] (99 % [2.21, 2.67]), 16/16. R1
  kept FR on 2.6–3.9× longer than in ITE (21.5 vs 8.25 / 5.5 t.u.) and preserved the full continuous-FR benefit;
  e_F(T) −59 %.
* **LTA 300 K, held out:** 1.46, 16/16.
* **ERI 600 K, the held-out negative control: fails.** S_geo 0.61 [0.53, 0.71] (99 % [0.49, 0.73]), 0/16; e_F(T)
  +55 %.
* R1 does not avoid harm where FR has nothing to repair, so it is not a general safety rule. It was not retuned after
  ERI.

### 5. What is the evidence that FR acts through establishment delay?

**Consistent across the resolved cells, but correlational.**
* **FR shortens window establishment in every cell where it helps** (ABF → FR, t.u.): ITE 300 K K1024 4.25 → 2.50;
  K256 4.5 → 3.0; ITE 150 K 7.06 → 4.19; LTA 150 K 16.8 → 9.3; LTA 300 K 9.0 → 7.3.
* **Where establishment is already fast, there is almost nothing to remove:** ERI 2.1 → 1.9.
* **K64 and K16 are uninformative here.** Their save grids are 2 and 8 t.u., too coarse to see a gain of about 1.5
  t.u.; their 4.0 → 4.0 and 8.0 → 8.0 are resolution artefacts.
* **K8, saved every 1 t.u. (E-A3), is informative and shows no gain:** 4.5 → 4.5 t.u. (paired difference 0.0
  [−0.5, 0.0]; FR shorter in 4/16 seeds). ABF's own establishment is 4.25–4.5 t.u. at every K from 1024 down to 8 where
  it is resolved, 6 t.u. at K4 and 27 t.u. at K1. R1 stopped later as K fell (5.5 → 16.5 t.u.) although establishment
  did not move, because its directedness statistic is built from fewer events (12 per K8 run).
* **S_geo roughly follows the size of the delay removed** (paired median reduction → S_geo; resolved cells only):

  | cell | delay removed (t.u.) | S_geo |
  |---|---|---|
  | LTA 150 K | 7.4 | 2.52 |
  | ITE 150 K | 3.0 | 1.64 |
  | ITE 300 K K1024 | 1.75 | 1.67 |
  | ITE 300 K K256 | 1.5 | 1.31 |
  | LTA 300 K | 1.4 | 1.46 |
  | ERI | 0.2 | 0.61 |

  * ERI is the outlier on the low side: little to remove, plus FR's late bias (question 6).
  * The ordering does **not** follow ABF's own establishment time: LTA 300 K establishes slower than ITE K1024 but
    gains less.
* **The decomposition agrees** (MECHANISM.md §1). FR's early advantage is a smaller bias at 0.1 T, i.e. less memory
  of unestablished early deposits in the cumulative accumulators.
* **Circulation.** FR leaves translocation counts within ±5 % everywhere except LTA 150 K (+18 %), the largest-gain
  cell. There circulation may contribute; the two effects were not separated.

### 6. What causes the late error of continuous FR?

**An ongoing bias in the deposits collected while FR is active.** It is not left over from early deposits, and it is
not confined to newly cloned walkers. All of the following is exploratory.
* **Size.** Continuous FR's late-only deposits carry 2.7–6.4× ABF's late-only bias over the same window. Bias accounts
  for 41–89 % of the late-window MSE, against 5–26 % for ABF.
* **Stopping removes it.** Post-stop deposits have a bias of 0.009–0.014 kJ/mol. Over the same post-stop window ABF's
  bias is 0.006–0.014, a difference of at most 0.003.
* **Population-level.** Dropping deposits from walkers cloned within 1 t.u. closes only 24–27 % of FR's excess error.
  Walkers that were not recently cloned are 1.8× (ITE) and 2.9× (ERI) worse than ABF.
* **Carrier.** In ERI the whole FR population's fibre conditionals are displaced (η 4.6×, u_x² 2.6× ABF's deviation).
  This is consistent with marginal reallocation, which cannot relax p(y|ξ), continually re-selecting walkers whose
  fibre state is out of equilibrium.
* **Not identified.** The precise selection mechanism, and the carrier in ITE, were not identified.

### 7. Fixed K vs equal cost vs wall-clock

| question | answer |
|---|---|
| fixed K (paired) | FR helps at K1024 (1.67) and K256 (1.31); unresolved at K64 (1.11 [0.92, 1.33]) and K16 (0.93 [0.70, 1.29]) |
| equal force evaluations, same K | FR/ABF cost 0.58 at K1024, 0.83 at K256, 1.02 at K64, 1.04 at K16, 1.03 at K8 (E-A3) |
| equal force evaluations, best vs best | eight configurations: 1.04 [0.74, 1.33] fixed selection, [0.78, 1.28] nested, UNRESOLVED (best = K16 for both); K1–K1024 (E-A3): 1.10 [0.75, 1.51] fixed, [0.87, 1.43] nested, UNRESOLVED (best ABF K1, best FR K16) |
| all-in | same: 1.04 [0.75, 1.31]; K1–K1024 1.16 [0.79, 1.58], nested [0.88, 1.48] |
| FR overhead per step | −2 to +1 % in single-thread production runs; −5 to +6 % in benchmarks for K ≤ 256 or one thread; +5 to +15 % at K1024 with 4–32 threads |
| parallel wall-clock, best vs best | 1.03–1.07 at 1–32 threads, all UNRESOLVED |
| best wall-clock configuration | ABF K16 at 1 thread; ABF K64 at 4–32 threads (10.8 s at 16 threads vs 76.6 s for ABF K1024 at 32) |

Therefore:
* **A fixed-K speedup is real, but it was not shown to be an efficiency gain.** It is largest exactly where ABF is
  least efficient.
* **A cheaper lever exists.** Running fewer replicas for longer removes most of the same early transient at lower
  cost.
* **When FR is still useful.** If K must be large for reasons outside this comparison, FR stopped by R1 gives 1.3–2.5×
  at that fixed K in the establishment-limited systems tested (ITE, LTA).
  * On this host, a large K with FR is still slower in wall-clock than ABF at K64: 47.6 s for FR at K1024 against
    10.8 s.
  * Multi-process parallelism over walkers was not measured.

### 8. How mature is the method?

**A well-characterised research method, not a production accelerator.**
* **What is established.** Early-stopping FR has a clear mechanism, a working reference-free stop rule within
  establishment-limited systems, and preregistered positive replications on two framework families (ITE, LTA) at
  fixed K.
* **What is missing.**
  1. **Applicability test.** There is no online test of whether ABF is establishment-limited before committing to FR.
     ERI shows the cost of running FR where ABF is not establishment-limited: about 1.6× ABF's time to accuracy
     (S_geo 0.61) and +55 % final error. R1 cannot detect that.
  2. **Equal-cost advantage.** None was shown on the one system where the frontier was measured, now from K1 to K1024
     (CIs wide, point estimates on ABF's side, 1.10 best vs best with the selection-aware CI [0.87, 1.43]). ABF's own
     allocation is the larger lever there, and below K64 its cost is flat within what 16 seeds resolve.
  3. **Late-bias mechanism.** It is identified only as population-level; there is no design that would let FR run
     longer safely.
  4. **Small K.** R1 has not been validated below K256. At K64, K16 and K8 it ran and stopped every run, and the
     comparison was unresolved each time; at K8 it stopped 3–4× later than establishment because its statistic sees
     only 12 events per run. Below K8 FR cannot run under the genealogical safety rules (R1's backstops cannot fire at
     K4), which was reported rather than relaxed.
* **Novelty.** The closest prior art is selection/resampling in adaptive biasing (Lelièvre–Rousset–Stoltz 2007 and
  descendants). An "accelerator" framing is not supported by the equal-budget evidence. What is supported is "a
  reference-free schedule that makes marginal FR useful at fixed K in the establishment-limited systems tested". It is
  not a safety rule: ERI was harmed.

## Compute

* **Campaign total.** About 65.9 of the 100 authorised CPU core-h (COMPUTE_LEDGER.md; 44.1 before amendment E-A3,
  21.8 for it); aggregate about 264.3 of about 298.4.
* **Hardware.** No GPU and no external or paid resource were used.
* **Seeds.** No seed was dropped. Every confirmatory cell has 16/16 paired seeds.
