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

**Added 2026-10-17 (§7, §8).** The best-vs-best interval above held the selected configurations fixed; with the
selection re-done in every bootstrap replicate it is 1.04 [0.78, 1.28] on the eight configurations (§7). With the
frontier extended to K8, K4 and K1 (amendment E-A3, §8) the best ABF allocation moves to K1 and the best FR allocation
stays at K16: best FR / best ABF = **1.10, fixed selection [0.75, 1.51], nested [0.87, 1.43], UNRESOLVED**. ABF's cost
is flat from K1 to K64 within the resolution of 16 unpaired seeds, so the frontier has no resolved minimum to bracket;
it is now explored to its floor (K1) for ABF and to its safety floor (K8) for FR.

## 2. Fixed K: the paired two-arm cells (each confirmatory for its own comparison)

| K × T (t.u.) | block | FR rate | R1 stop, t.u.: median (range) | **S_geo [95 % CI]** | FR faster | post-ramp Ī_F | e_F(T) | window establishment ABF → FR |
|---|---|---|---|---|---|---|---|---|
| 1024 × 50 | C1 | 2.0 | 5.5 (5.5–6.0) | **1.67 [1.45, 1.98]** SUPPORTED | 16/16 | −39 % | −35 % [−40, −25] | 4.25 → 2.50 |
| 256 × 200 | EA_R1 | 1.0 | 6.5 (5.5–9.0) | **1.31 [1.07, 1.47]** SUPPORTED | 13/16 | −17 % | −9 % [−21, +6] | 4.5 → 3.0 |
| 256 × 200 | A (fixed stop 10) | 1.0 | 10 (fixed) | 1.28 [0.90, 1.43] UNRESOLVED | 11/16 | −15 % | −1 % [−14, +14] | 4.5 → 3.0 |
| 64 × 800 | EB64 | 0.2 | 7.5 (6.5–9.0) | 1.11 [0.92, 1.33] UNRESOLVED | 9/16 | +1 % [−11, +15] | −1 % [−15, +19] | 4.0 → 4.0 (save grid 2 t.u.) |
| 16 × 3200 | EB16 | 1.0 | 11.0 (9.0–14.0) | 0.93 [0.70, 1.29] UNRESOLVED | 7/16 | +10 % [−14, +21] | +3 % [−10, +19] | 8.0 → 8.0 (save grid 8 t.u.) |
| 8 × 6400 | EB8 (E-A3) | 1.0 | 16.5 (11.5–24.5) | 1.03 [0.72, 1.42] UNRESOLVED | 8/16 | +2 % [−12, +11] | +3 % [−10, +21] | 4.5 → 4.5 (save grid 1 t.u.; resolved) |

**Establishment resolution.** Establishment is read at saves. Every frontier cell of E.3 / E-A2 saves every 1.28 × 10⁶
force evaluations, which is 0.125 t.u. at K1024, 0.5 at K256, 2 at K64 and 8 at K16; the E-A3 cells (K8, K4, K1) save
every 1 t.u. (§8). So the K64 and K16 establishment
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

## 8. Amendment E-A3: the frontier below K16 (K8 both arms; K4 and K1 ABF; 1 t.u. save grid)

*Preregistered: `PREREGISTRATION.md` E-A3, committed 4e848bb5 at 11:28:15 UTC on 2026-10-17, before any K < 16 run
(the header's "about 11:35" is an estimate; see the erratum). Prompted by data (E.3 + E-A2 put both optima at K16, the
smallest K tried) and by a reviewer (selection on the scoring data, §7; frontier not bracketed). Scripts:
`analyze_online.py` (EB8), `equal_budget_frontier.py` (E_frontier_ext), `establishment_frontier.py`. Outputs:
`analysis/{EB8, E_frontier_ext, E_frontier_ext_matched, E_establishment, frcal_K8, frcal_K4}.json`. Figures 3c
(`equal_budget_frontier_ext.png`) and 9 (`establishment_frontier.png`).*

### 8.1 Cells, cost, save grid

| block | K × T (t.u.) | arms | seeds | FR rate | saves | runs | core-h | wall per run (s), median (range) |
|---|---|---|---|---|---|---|---|---|
| EB8 | 8 × 6400 | ABF, ABF + FR(R1) | 1400000–015 | 1.0 (blind, §8.2) | every 1 t.u. = 8 × 10⁴ FE | 32 | 9.30 | ABF 1048 (983–1158); FR 1040 (991–1089) |
| EB4 | 4 × 12800 | ABF only | 1410000–015 | – | every 1 t.u. = 4 × 10⁴ FE | 16 | 5.11 | 1147 (1079–1219) |
| EB1 | 1 × 51200 | ABF only | 1420000–015 | – | every 1 t.u. = 10⁴ FE | 16 | 7.27 | 1617 (1575–1710) |
| Ecal_K8, Ecal_K4 | T 50, rates 1.0 / 0.5 / 0.2 × 3 seeds | FR only | 1430000–2, 1440000–2 | – | every 0.125 t.u. | 18 | 0.045 | 6–12 |

* **Cost.** 21.8 core-h including the pilot (0.02) and analyses (≈ 0.1), against the pilot-based estimate of 17.7:
  the host's 1-minute load was ≈ 130 during the runs (80 of ours), and the per-step cost rose to 16.4 µs at K8, 9.0 at
  K4 and 3.2 at K1 (pilot at load 10: 12.8, 7.4, 2.7). Campaign total ≈ 65.9 of 100 core-h.
* **FR overhead.** −1 % per run at K8 (FR active for 11.5–24.5 of 6400 t.u.).
* **Save grid.** The E-A3 cells save every 1 t.u.; time to accuracy is therefore read at 8 × 10⁴, 4 × 10⁴ and 10⁴
  force evaluations against 1.28 × 10⁶ in every earlier cell. No earlier cell was re-run: the production rows of the
  eight earlier configurations in the extended table are identical to §3 (checked key by key), and `E_frontier.json`
  keeps every published key byte for byte. The all-in common budget of the extended set is 5.1205 × 10⁸ (K1's run
  plus its 50 000 equilibration steps; E-A3's text said 5.125 × 10⁸ by an arithmetic slip, see the erratum); the
  all-in medians of the eight earlier configurations are unchanged to four decimals.
* **Disk.** 5.7 GB of run files (1 t.u. saves), in the data root only.

### 8.2 Blind calibration at K8; why there is no K4 FR arm

K8, T 50 continuous FR, 3 seeds, median over seeds (`frcal_K8.json`; criteria ESS/K ≥ 0.30, lineage ≤ 3/K = 0.375,
≤ 2K = 16 replacements per t.u.):

| rate | min median ESS/K | max median lineage share (limit 0.375) | replacements per t.u. (limit 16) | safe |
|---|---|---|---|---|
| 0.2 | 0.571 | 0.375 | 0.2 | yes |
| 0.5 | 0.571 | 0.250 | 0.4 | yes |
| 1.0 | 0.500 | 0.375 | 0.8 | yes |

* **Rate 1.0 selected** (largest safe rung), the same as at K16.
* **Did the lineage criterion constrain anything?** No rung was rejected. The medians sat exactly on the limit (3 of 8
  walkers in one family) at rates 1.0 and 0.2; individual seeds at rate 1.0 held 4 of 8 in one family at 0.3–3.8 % of
  saves. So the criterion was touched but not binding, and a 4-of-8 family in the median would have failed it.
* **FR at K8 is a rare pairwise event, not a marginal reallocation.** 35–41 replacements in 50 t.u. at rate 1.0, one
  event per ≈ 1300 opportunities. The score is the recentred log-ratio of an 8-point KDE (bandwidth 0.10 rad on 180
  bins) to the uniform target; it is non-zero only when two walkers sit within about two bandwidths, so the operator
  acts as an occasional repulsion between walker pairs. In production, 12 replacements per run (range 8–24) before R1
  stopped; 22 at K16, 13 at K64 (weaker rate), 163 at K256, 611 at K1024.
* **K4: no FR arm, by the criterion stated in E-A3.** Condition (i), that R1's genealogical backstops can fire, fails by
  arithmetic at K4 (lineage limit max(0.10, 6/K) = 1.5 ≥ 1; ESS floor 1/K = 0.25 ≥ 0.15): an R1-stopped K4 FR arm would
  run with no genealogical protection at all. The diagnostic K4 calibration (`frcal_K4.json`) shows what that would
  mean: 0–10 replacements in 50 t.u.; at rates 0.5–1.0 one family held half the population for 13–25 % of saves and one
  seed reached 3 of 4 (0.75). Under the standard criteria every K4 rung is "safe" (limit 3/K = 0.75), i.e. the
  criterion is vacuous there. **This is the genealogy requirement small-K FR fails; it was reported, not relaxed.** K1
  has nothing to reallocate.

### 8.3 EB8: the paired K8 cell (confirmatory for its own comparison)

* **PRIMARY S_geo 1.029 [0.722, 1.419]** (99 % [0.704, 1.433]); FR faster in 8/16 seeds → **UNRESOLVED.** Per rung
  (0.200 → 0.080 kJ/mol): 1.02 [0.83, 1.29], 1.00 [0.78, 1.43], 1.10 [0.75, 2.22], 0.94 [0.54, 1.52], 1.02
  [0.89, 1.75]; every rung reached by every seed in both arms.
* **Integrity.** Bitwise paired before FR start 16/16; equal force evaluations; zero events after any stop; no run
  never stopped.
* **R1 at K8.** Stopped every run as "established" (none by the safety backstop): median 16.5 t.u., IQR 13.9–18.5,
  range 11.5–24.5. That is later than at K16 (11.0), K64 (7.5), K256 (6.5) and K1024 (5.5), while the establishment
  it tracks stays at 4.5 t.u. (§8.5): with 12 events per run the directedness statistic is too sparse for its "low"
  test to persist, so FR stays on 3–4× past establishment with no measurable effect.
* **Genealogy during FR** (medians over seeds): min windowed ESS/K 0.571, max lineage share 0.375 (= the calibration
  limit; 3 of 8 walkers), 12 replacements.
* **Secondaries (FR − ABF, paired, median [95 % CI], FR better in n/16).** Post-ramp Ī_F +1.9 % [−12.2, +10.6], 7;
  early-window Ī_F −2.7 % [−11.7, +5.8], 11; late-window Ī_F +8.0 % [−20.4, +19.5], 7; e_F(T) +3.0 % [−10.2, +20.6],
  8; e_F′(T) +3.4 % [−5.2, +9.9], 5; post-stop-deposit estimate +4.3 % [−9.1, +20.3], 7. Translocations 1704 vs 1719.
  All unresolved; the only sign of FR's early effect is the 11/16 early-window count.
* **Establishment, resolved at 1 t.u. for the first time below K256.** Frozen measure 4.5 → 4.5 t.u. (paired
  difference 0.0 [−0.5, 0.0]; FR shorter in 4/16, equal in 9); deposit-share measure 7.0 → 6.5 (0.0 [−1.0, 0.0]; FR
  shorter in 6/16, equal in 7). **At K8 FR does not shorten establishment.** The resolved gains were −1.75 t.u. at
  K1024 and −1.5 at K256 (16/16 each).

### 8.4 The extended frontier K1–K1024 (descriptive, unpaired across K)

Same statistic as §3; B = 5.12 × 10⁸ at every K. Median over 16 seeds [bootstrap 95 % CI of the median]; per-rung
median restricted τ (seeds that reached the rung); e_F at B; τ in t.u.; the save resolution at which τ is read.

| configuration | median geo τ_B (10⁸) [95 % CI] | 0.200 | 0.160 | 0.120 | 0.100 | 0.080 | e_F at B (kJ/mol) | τ in t.u. | τ resolution (FE) |
|---|---|---|---|---|---|---|---|---|---|
| ABF K1024 | 3.299 [2.702, 3.465] | 1.97 (16) | 2.50 (16) | 3.48 (16) | 4.20 (15) | 5.12 (7) | 0.0831 | 32.2 | 1.28 × 10⁶ |
| FR K1024 | 1.907 [1.685, 2.095] | 1.07 (16) | 1.32 (16) | 1.92 (16) | 2.53 (16) | 3.44 (16) | 0.0503 | 18.6 | 1.28 × 10⁶ |
| ABF K256 | 0.931 [0.853, 1.071] | 0.57 (16) | 0.68 (16) | 0.95 (16) | 1.29 (16) | 1.66 (16) | 0.0344 | 36.4 | 1.28 × 10⁶ |
| FR K256 | 0.770 [0.518, 0.893] | 0.41 (16) | 0.53 (16) | 0.76 (16) | 1.06 (16) | 1.40 (16) | 0.0314 | 30.1 | 1.28 × 10⁶ |
| ABF K64 | 0.241 [0.176, 0.397] | 0.13 (16) | 0.19 (16) | 0.29 (16) | 0.35 (16) | 0.47 (16) | 0.0243 | 37.7 | 1.28 × 10⁶ |
| FR K64 | 0.246 [0.191, 0.292] | 0.12 (16) | 0.13 (16) | 0.26 (16) | 0.32 (16) | 0.48 (16) | 0.0242 | 38.5 | 1.28 × 10⁶ |
| ABF K16 | 0.180 [0.158, 0.243] | 0.09 (16) | 0.12 (16) | 0.17 (16) | 0.22 (16) | 0.56 (16) | 0.0201 | 112.7 | 1.28 × 10⁶ |
| **FR K16** | **0.188 [0.168, 0.237]** | 0.08 (16) | 0.13 (16) | 0.20 (16) | 0.26 (16) | 0.39 (16) | 0.0228 | 117.3 | 1.28 × 10⁶ |
| ABF K8 | 0.194 [0.157, 0.271] | 0.07 (16) | 0.10 (16) | 0.22 (16) | 0.28 (16) | 0.61 (16) | 0.0206 | 242.2 | 8 × 10⁴ |
| FR K8 | 0.199 [0.147, 0.249] | 0.07 (16) | 0.10 (16) | 0.14 (16) | 0.35 (16) | 0.54 (16) | 0.0228 | 249.2 | 8 × 10⁴ |
| ABF K4 | 0.237 [0.173, 0.288] | 0.09 (16) | 0.15 (16) | 0.27 (16) | 0.44 (16) | 0.54 (16) | 0.0237 | 593.0 | 4 × 10⁴ |
| **ABF K1** | **0.170 [0.132, 0.252]** | 0.06 (16) | 0.09 (16) | 0.18 (16) | 0.29 (16) | 0.39 (16) | 0.0270 | 1697.5 | 10⁴ |

| ratio of medians (independent bootstrap) | production | all-in (common budget 5.1205 × 10⁸) | matched save grid (§8.4, item 5) |
|---|---|---|---|
| **best FR (K16) / best ABF (K1), fixed selection** | **1.105 [0.752, 1.514]** | 1.155 [0.786, 1.578] | 1.271 [0.812, 1.719] |
| **best FR / best ABF, nested (selection re-done per replicate)** | **1.105; 95 % [0.866, 1.430]; 99 % [0.816, 1.551]** | 1.141 (boot. median); [0.884, 1.480] | 1.221 (boot. median); [0.888, 1.556] |
| P(best FR / best ABF < 1), nested | 0.19 | 0.15 | 0.10 |
| verdict (nested; fixed agrees) | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| e_F at B, best FR / best ABF (fixed; nested) | 0.842 [0.588, 1.129]; 0.898 [0.608, 1.335] | 0.832 [0.585, 1.124] | – |
| ABF K1 / ABF K16 | 0.942 [0.620, 1.379] | 0.901 [0.596, 1.316] | 0.818 [0.558, 1.278] |
| ABF K8 / ABF K16 | 1.075 [0.714, 1.493] | 1.062 [0.709, 1.454] | 1.023 [0.661, 1.511] |
| ABF K4 / ABF K16 | 1.317 [0.830, 1.661] | 1.269 [0.804, 1.594] | 1.261 [0.822, 1.577] |
| ABF K1 / ABF K4 | 0.715 [0.511, 1.130] | 0.709 [0.509, 1.120] | 0.649 [0.500, 1.082] |
| ABF K4 / ABF K8 | 1.225 [0.797, 1.629] | 1.195 [0.789, 1.592] | 1.233 [0.793, 1.748] |
| ABF K16 / ABF K64 | 0.748 [0.439, 1.133] | 0.682 [0.425, 1.006] | 0.748 [0.439, 1.133] |
| FR K8 / FR K16 | 1.063 [0.750, 1.367] | 1.047 [0.744, 1.333] | 1.066 [0.724, 1.321] |
| FR K8 / ABF K8 | 1.029 [0.682, 1.414] | 1.026 [0.690, 1.404] | 1.084 [0.664, 1.546] |

Nested-bootstrap selection frequencies (production): ABF K1 69.8 %, K16 17.0 %, K8 10.8 %, K4 1.4 %, K64 1.1 %;
FR K16 59.9 %, K8 38.6 %, K64 1.6 %. The fixed pair (FR K16, ABF K1) is re-selected in 41.5 % of replicates.

**Verdicts by the E-A3 rules.**
1. **Does ABF's best allocation lie below K16? UNRESOLVED.** The best tested ABF allocation is K1 (0.170 vs 0.180 at
   K16), but ABF K1 / ABF K16 = 0.94 [0.62, 1.38]. ABF's cost is 0.17–0.24 × 10⁸ at every K from 1 to 64 (a factor 1.4
   between the extremes) while the unpaired CIs have half-widths of ±35–40 %. Nothing in K ≤ 64 is resolvably
   different from anything else in K ≤ 64.
2. **Does FR's best allocation lie below K16? UNRESOLVED.** Best FR stays at K16; FR K8 / FR K16 = 1.06 [0.75, 1.37].
   K8 is the smallest K at which FR may run under the safety rules (§8.2).
3. **Is the frontier bracketed from below?** For ABF there is nothing below K1 to test: the frontier is explored to its
   floor, and no neighbour of the point-estimate minimum is resolvably worse. The honest description is **a flat
   minimum from K1 to K64, not a bracketed optimum**. For FR, K8 is not resolvably worse than K16, so its frontier is
   still unbracketed and cannot be extended under the safety rules.
4. **Best FR / best ABF: UNRESOLVED by both intervals**, 1.10 with fixed selection [0.75, 1.51] and nested
   [0.87, 1.43]. The point estimate moved from 1.04 (eight configurations) to 1.10 because ABF gained an allocation
   (K1) that FR cannot use. The CIs allow FR to be from 13 % cheaper to 43 % dearer.
5. **Sensitivity to the save grid (post hoc).** The sustained criterion is stricter on the 1 t.u. grid of the new cells
   (16–128× more saves at which an excursion above ε can be caught), while τ is rounded up to the next save on the
   coarse grid. Re-reading the E-A3 cells on the 1.28 × 10⁶-force-evaluation grid of the earlier cells
   (`E_frontier_ext_matched.json`) gives ABF K1 0.147, K8 0.184, K4 0.237 → 0.227 and FR K8 0.200: every point estimate
   moves further to ABF's side (best FR / best ABF 1.27 [0.81, 1.72], nested [0.89, 1.56], P(< 1) 0.10; ABF K1 / K16
   0.82 [0.56, 1.28]) and every verdict stays UNRESOLVED. The finer grid of the new cells therefore biased the primary
   comparison against them, i.e. conservatively for the question asked.
6. **What the statistic measures at small K.** For K ≤ 16 the ladder's tightest rung (0.080 kJ/mol) is reached within
   the first 3–5 % of the budget (τ ≤ 0.24 × 10⁸ of 5.12 × 10⁸), and e_F at B is 0.020–0.027 kJ/mol, three to four
   times below that rung. The frontier at small K is therefore a frontier of the early transient, decided in 110 t.u.
   at K16, 240 at K8, 590 at K4 and 1700 at K1. By the end-of-budget error the ordering differs: ABF K1 has the
   largest e_F at B of all K ≤ 64 (0.027) and FR K16's is lower than ABF K1's (0.84 [0.59, 1.13], unresolved). A ladder
   with tighter rungs would be needed to compare the small-K allocations over the whole budget; none was
   preregistered, so none is reported.

### 8.5 Establishment at every K on each cell's own grid (descriptive; `E_establishment.json`, figure 9)

Frozen measure: the first save at which the trailing 1-t.u. mean of the instantaneous window fraction reaches 0.75 of
the uniform window share. Deposit-share measure (E-A3): the same first-passage rule on the window share of the
deposits collected during the trailing max(1 t.u., save interval). Medians over 16 seeds [95 % CI]; no seed failed to
establish.

| configuration | save interval (t.u.) | frozen measure (t.u.) | deposit-share measure (t.u.) | R1 stop, median (t.u.) |
|---|---|---|---|---|
| ABF K1024 | 0.125 | 4.25 [4.12, 4.31] | 4.38 [4.12, 4.50] | – |
| FR K1024 | 0.125 | 2.50 [2.38, 2.50] | 2.50 [2.50, 2.50] | 5.5 |
| ABF K256 | 0.5 | 4.50 [3.50, 4.50] | 4.50 [4.00, 5.00] | – |
| FR K256 | 0.5 | 3.00 [2.50, 3.00] | 3.00 [3.00, 3.50] | 6.5 |
| ABF K64 | 2 | 4.00 [4.00, 6.00] | 6.00 [6.00, 6.00] | – |
| FR K64 | 2 | 4.00 [4.00, 5.00] | 6.00 [6.00, 6.00] | 7.5 |
| ABF K16 | 8 | 8.00 [8.00, 16.00] | 16.00 [16.00, 16.00] | – |
| FR K16 | 8 | 8.00 [8.00, 16.00] | 16.00 [16.00, 16.00] | 11.0 |
| ABF K8 | 1 | 4.50 [4.00, 6.00] | 7.00 [6.00, 8.00] | – |
| FR K8 | 1 | 4.50 [3.50, 6.00] | 6.50 [5.00, 7.00] | 16.5 |
| ABF K4 | 1 | 6.00 [5.00, 8.00] | 7.00 [6.00, 9.00] | – |
| ABF K1 | 1 | 27.0 [15.0, 43.0] | 22.0 [15.0, 30.0] | – |

Readings.
* **ABF's establishment is 4.25–4.5 t.u. at every K where it is resolved from K1024 down to K8**; the K64 and K16
  values are save-grid artefacts (as stated in §2). At K4 it is 6 t.u.; at K1 it is 27 t.u. (per-seed 6–80): a single
  walker needs many window passages before the bias flattens the marginal.
* **The deposit-share measure agrees with the frozen one at K ≥ 256 (4.4–4.5) but reads 7 t.u. at K8 and K4 and 22 at
  K1:** with few walkers, the deposits of the first few t.u. are concentrated in the cage region until the walkers
  themselves cross, so the time-averaged window share lags the instantaneous one.
* **FR shortens establishment where it helps and not at K8**: −1.75 t.u. (K1024), −1.5 (K256), 0.0 [−0.5, 0.0] (K8,
  resolved at 1 t.u.). The K16 / K64 values remain unresolvable on their grids.
* **R1 stops later as K falls (5.5 → 16.5 t.u.) while establishment does not move**, because its directedness
  statistic is built from fewer and fewer events.

### 8.6 What E-A3 changes in the answer to Q5

Nothing in direction, and it closes the open end. No allocation of ABF + FR was shown to beat the best tested ABF
allocation; the best-vs-best point estimate is now 1.10 (nested 95 % [0.87, 1.43]); ABF's cost is flat from K1 to
K64 within what 16 unpaired seeds resolve, so there is no sharp optimum left to bracket; FR cannot run below K8 under
the genealogical safety rules, and at K8 it neither shortens establishment nor changes the error. The remaining
limitation is the statistic itself (§8.4, item 6): at small K the ladder is reached in the first few per cent of the
budget.
