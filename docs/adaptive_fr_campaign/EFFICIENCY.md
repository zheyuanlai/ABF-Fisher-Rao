# Experiment E — efficiency: fixed K, equal force evaluations, all-in cost and parallel wall-clock

*Preregistered: `PREREGISTRATION.md` §E (fd3b10ed, 07:32 UTC), with amendments E-A1 and E-A2.
* **E-A1** (integrity handling only; no statistic changed) was written at about 07:45, after EA_R1 had finished
  (07:44:45). It was needed because the frozen analysis refused to run on EA_R1. It was first committed at 07:56:23 in
  d0929f7a, after EB64 and the six-configuration frontier had been computed.
* **E-A2** (the K16 cell) was **prompted by the data**: E.3 had found the best allocation of both algorithms at the
  smallest K tried. It was committed at 07:56:23 (d0929f7a), before the K16 calibration (07:56:29) and the K16 cell
  (07:57:02) were launched; see the erratum in PREREGISTRATION.md about its header time.
Scripts: `analyze_online.py` (paired cells), `equal_budget_frontier.py` (E.3), `wallclock_bench.py` +
`wallclock_tta.py` (E.4).
Outputs: `results/adaptive_fr_campaign/analysis/{EA_R1, EB64, EB16, E_frontier, E_wallclock_bench,
E_wallclock_bench_run2, E_wallclock_tta}.json`.
Figures: `figures/adaptive_fr/equal_budget_frontier.png` (3) and `wallclock_ite300.png` (8).*

**System and comparison.** Everything here is ITE 300 K, 180 bins, comparing two algorithms only: histogram ABF, and
the same ABF + FR stopped online by the frozen rule R1. The budget is B = 5.12 × 10⁸ production molecular force
evaluations per run at every allocation, i.e. K replicas × T = B / K.

## 1. Answer to Q5

**Not shown.** A fixed-K speedup of ABF + FR over ABF was not shown to translate into a gain at equal force
evaluations, all-in cost or parallel wall-clock once ABF itself runs at its best tested allocation. The best-vs-best
CIs are wide: they allow FR to be anywhere from about 26 % cheaper to 33 % dearer. The point estimates all sit on
ABF's side.

1. **The fixed-K speedup is largest where ABF is least efficient, and it becomes unresolved as K shrinks toward
   ABF's best tested allocation.** Paired S_geo is 1.67 at K1024, 1.31 at K256, 1.11 [0.92, 1.33] at K64 and 0.93
   [0.70, 1.29] at K16.
2. **At equal force evaluations**, the best FR allocation (K16) is 1.04× the best ABF allocation (K16) in time to
   accuracy: 95 % CI [0.74, 1.33], UNRESOLVED. The point estimate is on ABF's side.
3. **In parallel wall-clock**, at every thread count, the best FR configuration is 1.03–1.07× the best ABF
   configuration (all UNRESOLVED; point estimates on ABF's side). The best ABF configuration is K16 at 1 thread and
   K64 at 4–32 threads.

FR's per-step overhead is small:
* −2 to +1 % in the single-threaded production runs;
* −5 to +6 % in the benchmarks for K ≤ 256 and at one thread;
* systematically +5 to +15 % at K1024 with 4–32 threads (7 of 8 rows; one row −0.2 %).

Overhead is therefore not the main reason FR does not win. The likely reason is that its acceleration acts on a
transient the efficient allocation has already diluted (MECHANISM.md §1.2, reading 5).

## 2. Fixed K: the paired two-arm cells (each confirmatory for its own comparison)

| K × T (t.u.) | block | FR rate | R1 stop, t.u.: median (range) | **S_geo [95 % CI]** | FR faster | post-ramp Ī_F | e_F(T) | window establishment ABF → FR |
|---|---|---|---|---|---|---|---|---|
| 1024 × 50 | C1 | 2.0 | 5.5 (5.5–6.0) | **1.67 [1.45, 1.98]** SUPPORTED | 16/16 | −39 % | −35 % [−40, −25] | 4.25 → 2.50 |
| 256 × 200 | EA_R1 | 1.0 | 6.5 (5.5–9.0) | **1.31 [1.07, 1.47]** SUPPORTED | 13/16 | −17 % | −9 % [−21, +6] | 4.5 → 3.0 |
| 256 × 200 | A (fixed stop 10) | 1.0 | 10 (fixed) | 1.28 [0.90, 1.43] UNRESOLVED | 11/16 | −15 % | −1 % [−14, +14] | 4.5 → 3.0 |
| 64 × 800 | EB64 | 0.2 | 7.5 (6.5–9.0) | 1.11 [0.92, 1.33] UNRESOLVED | 9/16 | +1 % [−11, +15] | −1 % [−15, +19] | 4.0 → 4.0 (save grid 2 t.u.) |
| 16 × 3200 | EB16 | 1.0 | 11.0 (9.0–14.0) | 0.93 [0.70, 1.29] UNRESOLVED | 7/16 | +10 % [−14, +21] | +3 % [−10, +19] | 8.0 → 8.0 (save grid 8 t.u.) |

**Establishment resolution.** Establishment is read at saves. Every frontier cell saves every 1.28 × 10⁶ force
evaluations, which is 0.125 t.u. at K1024, 0.5 at K256, 2 at K64 and 8 at K16. So the K64 and K16 establishment
values cannot show a gain of the size seen at K256 (1.5 t.u.), and the K16 value only says both arms were
established by the first save. Time to accuracy in force evaluations has the same resolution at every K.

**Integrity, all cells:**
* bitwise-paired before FR start (16/16);
* equal force evaluations (asserted);
* zero FR events after any stop;
* no run that never stopped.

**FR rates came from the blind top-down safety calibration** (E.2 / E-A2) and were never chosen on an outcome.
* At K64, rates 1.0 and 0.5 failed the lineage criterion (0.078 and 0.0625 > 0.05).
* At K16 every rung passed, because the limit 3/K = 0.1875 is looser there.
* The K64 cell therefore ran a weaker FR (13 replacements per run) than the K16 cell (22) or the K256 cell (163).
* Whether FR shortened establishment at K64 or K16 cannot be resolved on their save grids (above).

## 3. Equal force evaluations: the eight-configuration frontier (E.3 + E-A2; descriptive, unpaired across K)

**Statistic.** Per seed, the geometric mean over the kT ladder {0.200, 0.160, 0.120, 0.100, 0.080} kJ/mol of the
sustained time to accuracy, in production force evaluations, restricted at B. The table gives the median over 16
seeds with a bootstrap 95 % CI of the median; the per-rung columns give the median restricted τ and, in brackets, the
number of seeds that reached the rung.

| configuration | median geo τ_B (10⁸) [95 % CI] | 0.200 | 0.160 | 0.120 | 0.100 | 0.080 | e_F at B (kJ/mol) | τ in t.u. |
|---|---|---|---|---|---|---|---|---|
| ABF K1024 | 3.299 [2.702, 3.465] | 1.97 (16) | 2.50 (16) | 3.48 (16) | 4.20 (15) | 5.12 (7) | 0.0831 | 32.2 |
| FR K1024 | 1.907 [1.685, 2.095] | 1.07 (16) | 1.32 (16) | 1.92 (16) | 2.53 (16) | 3.44 (16) | 0.0503 | 18.6 |
| ABF K256 | 0.931 [0.853, 1.071] | 0.57 (16) | 0.68 (16) | 0.95 (16) | 1.29 (16) | 1.66 (16) | 0.0344 | 36.4 |
| FR K256 | 0.770 [0.518, 0.893] | 0.41 (16) | 0.53 (16) | 0.76 (16) | 1.06 (16) | 1.40 (16) | 0.0314 | 30.1 |
| ABF K64 | 0.241 [0.176, 0.397] | 0.13 (16) | 0.19 (16) | 0.29 (16) | 0.35 (16) | 0.47 (16) | 0.0243 | 37.7 |
| FR K64 | 0.246 [0.191, 0.292] | 0.12 (16) | 0.13 (16) | 0.26 (16) | 0.32 (16) | 0.48 (16) | 0.0242 | 38.5 |
| **ABF K16** | **0.180 [0.158, 0.243]** | 0.09 (16) | 0.12 (16) | 0.17 (16) | 0.22 (16) | 0.56 (16) | **0.0201** | 112.7 |
| **FR K16** | **0.188 [0.168, 0.237]** | 0.08 (16) | 0.13 (16) | 0.20 (16) | 0.26 (16) | 0.39 (16) | 0.0228 | 117.3 |

| ratio of medians (independent bootstrap) | production | all-in (common budget 5.128 × 10⁸) |
|---|---|---|
| **best FR (K16) / best ABF (K16)** | **1.041 [0.739, 1.326] → UNRESOLVED** | 1.040 [0.745, 1.309] → UNRESOLVED |
| FR K64 / ABF K64 | 1.021 [0.554, 1.464] | 1.015 [0.589, 1.411] |
| FR K256 / ABF K256 | 0.827 [0.552, 0.975] | 0.855 [0.607, 0.984] |
| FR K1024 / ABF K1024 | 0.578 [0.512, 0.698] | 0.654 [0.599, 0.753] |
| ABF K16 / ABF K64 | 0.748 [0.439, 1.133] | 0.682 [0.425, 1.006] |
| ABF K64 / ABF K256 | 0.259 [0.183, 0.434] | 0.258 [0.192, 0.410] |
| ABF K256 / ABF K1024 | 0.282 [0.257, 0.350] | 0.286 [0.262, 0.338] |
| FR K16 / FR K64 | 0.762 [0.621, 1.041] | 0.699 [0.575, 0.932] |
| e_F at B, best FR / best ABF | 1.133 [0.795, 1.404] | 1.133 [0.795, 1.404] |

### Readings

1. **Why small K is cheaper.** For K ≥ 64, ABF's time to accuracy in **physical time** is almost independent of K
   (32–38 t.u.), and its window establishment is 4.25–4.5 t.u. wherever it is resolved (K256–K1024). The error is
   then limited mostly by how long the early transient's memory takes to dilute, not by the number of samples per
   unit time. Each 4× reduction of K therefore cuts the force-evaluation cost by about 3.5–4× (ABF K64/K256 0.26,
   K256/K1024 0.28).
2. **Below K64 the returns flatten. The frontier is not bracketed.**
   * At K16 the time in t.u. triples (113 t.u.).
   * The force-evaluation gain over K64 is 0.75 [0.44, 1.13] for ABF and 0.76 [0.62, 1.04] for FR, both
     unresolved.
   * At the tightest rung (0.080 kJ/mol), ABF K16 is slower than ABF K64 (0.56 vs 0.47 × 10⁸), while FR K16 (0.39)
     is the fastest of all eight configurations there.
   * Nothing below K16 was tested, so where the minimum lies is not known.
3. **FR's fixed-K advantage acts like a fix for an inefficient allocation, and a smaller K is a cheaper fix.**
   * At K1024, FR roughly halves ABF's cost (0.58).
   * Reducing K from 1024 to 256 without FR saves more (0.28), and reducing it to 64 or 16 saves much more again.
   * FR's relative gain shrinks along the same path and is unresolved at K64 (1.02 [0.55, 1.46]) and K16
     (1.04 [0.74, 1.33]).
4. **The all-in variant changes nothing.** It adds equilibration (5 t.u. at the same K) and compares at the common
   all-in budget 5.128 × 10⁸. It gives the same best allocations and the same verdict.

## 4. FR overhead per step

| source | ABF | FR | FR − ABF |
|---|---|---|---|
| production runs, single thread, median wall per run (s): C1 K1024 | 744 | 744 | 0 % |
| EA_R1 K256 (ABF = historical EB runs) | 742 | 729 | −2 % |
| A K256 (fixed stop) | 742 | 751 | +1 % |
| EB64 K64 | 734 | 733 | 0 % |
| EB16 K16 | 764 | 765 | 0 % |
| benchmark, 2 t.u. segment with FR active, P = 1, K16–K1024 (two runs) | – | – | +0.6 to +5.7 % |
| benchmark, P = 4–32, K ≤ 256 (two runs) | – | – | −5 to +5 % |
| benchmark, P = 4–32, K1024 (two runs) | – | – | −0.2 to +15 % |

* **R1's controller check is host-side** and runs every 0.25 t.u. until the stop. The birth–death step costs one
  gather/scatter of a few walkers per opportunity.
* **The single-threaded per-run overhead is within the run-to-run noise** of a shared host. The FR arm stops after
  5.5–14 t.u. of 50–3200 t.u., so for most of a run it is plain ABF.
* **The K1024 threaded overhead is systematic** (+5 to +15 % in 7 of 8 rows) and applies only while FR is active.

## 5. Parallel wall-clock (E.4 + E-A2; measured)

**Throughput.**
* **Host.** atlas: 2 × AMD EPYC 9554, 256 logical CPUs, shared with other users.
* **Load.** The 1-minute load average was 22 → 24 (run 1, 07:49–07:53) and 28 → 19 (run 2, 08:10–08:15). None of our
  jobs were running.
* **Replicability.** Run 2 / run 1 seconds per step: median 0.998, range 0.91–1.06 over the 30 shared rows.

Measured µs per MD step (run 2):

| K | ABF P=1 | P=4 | P=8 | P=16 | P=32 | FR P=1 | P=4 | P=8 | P=16 | P=32 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1441.8 | 518.3 | 395.6 | 271.5 | 237.8 | 1524.3 | 586.2 | 394.6 | 311.9 | 255.6 |
| 256 | 363.0 | 153.2 | 102.4 | 78.7 | 71.8 | 371.4 | 158.9 | 105.3 | 82.2 | 74.8 |
| 64 | 91.6 | 45.4 | 33.1 | 28.8 | 30.1 | 92.6 | 45.8 | 34.1 | 30.1 | 31.7 |
| 16 | 23.8 | 17.5 | 15.6 | 15.9 | 18.9 | 24.2 | 17.3 | 14.8 | 15.3 | 19.8 |

**Wall-clock time to accuracy.** This is the measured seconds per step at P threads × each seed's geometric-mean
sustained τ in steps (§3). It combines measured throughput with measured τ; it is not a re-run at every P. Medians
over 16 seeds, production:

| P | ABF K1024 | FR K1024 | ABF K256 | FR K256 | ABF K64 | FR K64 | ABF K16 | FR K16 | best ABF | best FR | **best FR / best ABF [95 % CI]** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 465 s | 284 s | 132 s | 112 s | 34.5 s | 35.6 s | 26.9 s | 28.3 s | K16 | K16 | 1.05 [0.75, 1.34] UNRESOLVED |
| 4 | 167 | 109 | 55.7 | 47.8 | 17.1 | 17.6 | 19.8 | 20.3 | K64 | K64 | 1.03 [0.56, 1.48] UNRESOLVED |
| 8 | 127 | 73.5 | 37.2 | 31.7 | 12.5 | 13.1 | 17.6 | 17.4 | K64 | K64 | 1.05 [0.57, 1.51] UNRESOLVED |
| 16 | 87.5 | 58.1 | 28.6 | 24.7 | **10.8** | 11.6 | 17.9 | 18.0 | K64 | K64 | 1.07 [0.58, 1.53] UNRESOLVED |
| 32 | 76.6 | 47.6 | 26.1 | 22.5 | 11.3 | 12.2 | 21.2 | 23.2 | K64 | K64 | 1.07 [0.58, 1.54] UNRESOLVED |

The all-in values (`E_wallclock_tta.json`, `all_in_common`) are 5–29 % larger. They have the same best configurations
and best FR / best ABF 1.02–1.07, all UNRESOLVED.

### Readings

1. **Same-K wall-clock ratios equal the force-evaluation ratios.** FR K1024 / ABF K1024 is 0.58–0.66 at every P;
   FR K256 / ABF K256 is 0.85–0.86. If K1024 is required for reasons outside this comparison, FR gains at that K.
   It does not make K1024 competitive with ABF at K64 in wall-clock on this host (reading 2).
2. **Fewer replicas also win in wall-clock on this host.** The fastest configuration is ABF at K64 on 16 threads
   (10.8 s), about 7× faster than ABF at K1024 on 32 threads (76.6 s) and 4.4× faster than FR at K1024 on 32 threads
   (47.6 s).
   * The numba threading of one run saturates at about 16 threads for K ≤ 64.
   * At K16, more than 8 threads slow the run down.
   * Using more cores would need parallelism across walkers beyond one numba process (multi-process, multi-node or
     GPU). That was not measured.
3. **At K16, the K16 advantage in force evaluations is lost in wall-clock** once P ≥ 4, because K16 cannot use the
   threads (16 µs per step at best) and needs 3× the physical time of K64.

## 6. Limitations

* **One system.** E is ITE 300 K only. The frontier was not measured for LTA or ERI, where FR's fixed-K effects are
  larger (LTA 150 K) or negative (ERI).
* **Unpaired across K.** Cross-K comparisons are unpaired (independent seeds), so their CIs are wide. The
  best-vs-best comparisons are UNRESOLVED, not "equivalent"; no equivalence margin was preregistered.
* **Different FR intensities by K.** Small-K FR rates were fixed by the blind safety calibration (K64 0.2, K16 1.0).
  The K64 FR is therefore weaker than at the other K. A stronger FR at K64 was excluded by the preregistered genealogy
  criterion.
* **Wall-clock is projected.** It is measured throughput × measured τ, on a shared host, with numba threads within
  one process. Multi-process or multi-node parallelism over walkers, and GPU execution, were not measured.

## 7. Selection-aware (nested) bootstrap of best FR / best ABF (added 2026-10-17; no new runs)

**Why.** A reviewer (2026-10-17) pointed out that the best K of each algorithm in §3 is selected on the same 16 seeds
that score it, and that the bootstrap behind 1.04 [0.74, 1.33] holds the two chosen configurations fixed
(`equal_budget_frontier.py`: the best allocation is chosen once, then `ratio_ci` resamples only the two chosen
per-seed arrays). **The published interval therefore omitted the uncertainty of the selection.**

**What was added.** In each of the 10 000 replicates the seeds are resampled within every configuration
(independently, as in the fixed-selection bootstrap), the best K of each algorithm is re-selected on the resampled
medians, and best FR / best ABF is taken. The results sit under new keys only (`nested_bootstrap`,
`question_B_verdict_nested`, `save_resolution_force_evals`); every pre-existing key of `E_frontier.json` is
byte-identical (checked), so `reproduce_analyses.sh` still verifies.

| best FR / best ABF | fixed selection (published) | nested: selection re-done in every replicate |
|---|---|---|
| geo τ_B, production | **1.041 [0.739, 1.326]** | **1.041; 95 % [0.781, 1.281]**; 99 % [0.686, 1.428] |
| geo τ_B, all-in (common budget 5.128 × 10⁸) | 1.040 [0.745, 1.309] | 1.040 (bootstrap median 1.039); 95 % [0.773, 1.291]; 99 % [0.687, 1.475] |
| e_F at B, production | 1.133 [0.795, 1.404] | bootstrap median 1.081; 95 % [0.771, 1.420] |

Selection frequencies (production): ABF K16 90.5 %, K64 9.5 %, K256 and K1024 0 %; FR K16 95.8 %, K64 4.2 %. The
fixed choice (K16, K16) is re-selected in 86.7 % of replicates (95.6 % all-in). P(ratio < 1) = 0.30.

**Reading.**
* The published 1.04 [0.74, 1.33] omitted selection uncertainty. Accounting for it does not widen the interval here:
  the nested 95 % interval is [0.78, 1.28]. The selection is stable (K16 for both algorithms in 87 % of replicates),
  and when it switches it does so in replicates where the K16 resample is unusually slow, so an extreme ratio is
  replaced by a K64 one: a minimum over K has a compressed upper tail. The nested statistic also carries the
  winner's-curse bias of a minimum in both numerator and denominator; it cancels in the ratio here (bootstrap median
  1.041 = fixed ratio).
* **Verdict unchanged: UNRESOLVED by both intervals.** The point estimate is still on ABF's side and the 99 %
  interval still spans 0.69–1.43.
* The statement to quote is: the best tested FR allocation costs 1.04× the best tested ABF allocation, 95 %
  [0.78, 1.28] with the selection re-done in every replicate, [0.74, 1.33] with the selection held fixed.
* Neither interval addresses the second reviewer point, that nothing below K16 was tested; amendment E-A3 does.
