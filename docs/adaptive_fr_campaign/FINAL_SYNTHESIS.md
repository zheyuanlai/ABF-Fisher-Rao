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
   * Best FR / best ABF is 1.04 [0.74, 1.33] in force evaluations and 1.03–1.07 in wall-clock, all unresolved.
   * The point estimates are on ABF's side, but the CIs allow FR to be up to about 26 % cheaper.
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
* **The frontier is not bracketed.** K16 vs K64 is unresolved for both algorithms, and nothing below K16 was tested.
* **The K16 cell was prompted by the data.** It was added by amendment E-A2 because the six-configuration frontier
  put both optima at K64, then the smallest K tried. It was committed before any K16 run.

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
| equal force evaluations, same K | FR/ABF cost 0.58 at K1024, 0.83 at K256, 1.02 at K64, 1.04 at K16 |
| equal force evaluations, best vs best | 1.04 [0.74, 1.33], UNRESOLVED (best = K16 for both) |
| all-in | same (1.04 [0.75, 1.31]) |
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
  2. **Equal-cost advantage.** None was shown on the one system where the frontier was measured (CIs wide, point
     estimates on ABF's side). ABF's own allocation is the larger lever there.
  3. **Late-bias mechanism.** It is identified only as population-level; there is no design that would let FR run
     longer safely.
  4. **Small K.** R1 has not been validated below K256. At K64 and K16 it ran and stopped, and the comparison was
     unresolved.
* **Novelty.** The closest prior art is selection/resampling in adaptive biasing (Lelièvre–Rousset–Stoltz 2007 and
  descendants). An "accelerator" framing is not supported by the equal-budget evidence. What is supported is "a
  reference-free schedule that makes marginal FR useful at fixed K in the establishment-limited systems tested". It is
  not a safety rule: ERI was harmed.

## Compute

* **Campaign total.** About 44.1 of the 100 authorised CPU core-h (COMPUTE_LEDGER.md); aggregate about 242.5 of about
  298.4.
* **Hardware.** No GPU and no external or paid resource were used.
* **Seeds.** No seed was dropped. Every confirmatory cell has 16/16 paired seeds.
