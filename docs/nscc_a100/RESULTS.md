# A100 wall-clock study on NSCC ASPIRE2A: results

*2026-10-26 to 2026-10-27, NSCC ASPIRE2A, project personal-e1297740.*
* Design: `PREREGISTRATION.md`, frozen before any benchmark (freeze manifest
  `results/nscc_a100/freeze/design.json`), with Amendment A1.
* Validation: `VALIDATION.md`. How to run or resume: `HANDOFF.md`.
* Generated tables: `results/nscc_a100/analysis/*_tables.md` and `results/nscc_a100/bench/tables.md`. Every number
  below is taken from them.
* Figures: `figures/nscc_a100/A-E`, and Figure F in `figures/nscc_a100/all_N_T_wallclock/`.

**Claim labels.**
* **[measured]**: timings on NSCC hardware.
* **[exploratory]**: selection and estimation on the same 24 seeds.
* **[confirmatory]**: fresh seeds, with the selection frozen beforehand.

## 1. Answer

**GPU parallelisation.**
* **[measured]** One A100 runs the validated algorithm much faster per walker-step than the best CPU route (EPYC
  7713, 1-16 cores), once N is large enough to fill the GPU:
  * entropic gateway: from N = 128 (1.3x) to N = 2048 (4.8x ABF, 4.3x FR);
  * ethane / LTA: from N = 32 (1.3-1.4x) to N = 1024 (10x ABF, 7.6-7.8x FR).
* Below those N the CPU is faster. The per-N routing gives each rung its faster backend.

**Wall-clock to accuracy, best FR deployment vs best ABF deployment (primary claim):**

| system | best ABF (frozen) | best FR (frozen) | **R_W [95 % CI]** | **R_e2e [95 % CI]** | **decision** |
|---|---|---|---|---|---|
| ethane / LTA 150 K (primary) | N 128, A100: 3.06 s | N 256, A100: 2.20 s | **0.72 [0.64, 0.83]** | **0.78 [0.72, 0.88]** | **FR faster** (Holm-significant) |
| entropic gateway (primary) | N 32, 4 cores: 2.83 s | N 32, 4 cores: 2.47 s | 0.87 [0.65, 1.23] | 0.98 [0.82, 1.14] | **unresolved** |
| ethane / LTA 300 K (secondary) | N 512, A100: 3.61 s | N 512, A100: 4.50 s | 1.25 [1.05, 1.55] | 1.22 [1.05, 1.49] | **ABF faster** |

* R = median wall seconds to sustained mid accuracy of the best FR deployment / the same for the best ABF
  deployment. R < 1 favours FR. W counts from the engine's timed start; W_e2e from process start.
* **[confirmatory]** 128 fresh seeds per arm (64 for LTA 300 K). Unpaired 10 000-resample bootstrap. Holm across
  the two primary systems.

**What the GPU changed.**
* **LTA.** On the A100, the best deployment of both methods moves from N = 16-32 on CPU cores to N = 128-512 on the
  GPU. There:
  * on LTA 150 K, FR at N = 256 is 28 % faster in wall-clock than the best ABF (N = 128), and the CI excludes 1;
  * on LTA 300 K, the easier temperature, ABF at N = 512 is 25 % faster than FR at N = 512.

  The hybrid short-T study on CPUs left all three systems unresolved. The A100 resolves both LTA systems, in opposite
  directions.
* **Gateway.** The best W_mid deployments of both methods stay on 4 CPU cores at N = 32.
  * The A100 arms are close: FR at N = 256 reaches mid accuracy in 1.96 s vs 1.95 s for the CPU FR at N = 32
    (exploration medians).
  * Counted from process start, the A100 arms win: 0.7 s start-up vs 2.3 s for the walker-parallel CPU engine, so the
    exploratory best W_e2e allocation is ABF N = 128 / FR N = 256, both on the A100.
  * The preregistered selection uses W_mid, so the confirmed gateway arms are the CPU ones, and their comparison is
    unresolved.

## 2. Throughput: A100 vs NSCC CPU (Figure A, `results/nscc_a100/bench/tables.md`)

Steady-state ns per walker-step (median of 3 repeats on the hybrid short-T benchmark windows, FR active).
* "Best CPU" is the fastest of S1 and P_t (t <= 16) on the EPYC 7713.
* Speed-up = best CPU / A100 (> 1: the A100 is faster).
* FR rows are the Amendment-A1 re-measurements.
* Routing used the ABF rows.

| N | gateway ABF: CPU / A100 (speed-up) | gateway FR | LTA 300 K ABF | LTA 300 K FR | LTA 150 K ABF | LTA 150 K FR |
|---|---|---|---|---|---|---|
| 16 | 49.8 / 101 (0.49x) | 55.7 / 121 (0.46x) | 464 / 521 (0.89x) | 523 / 743 (0.70x) | 455 / 523 (0.87x) | 532 / 744 (0.71x) |
| 32 | 33.7 / 51.3 (0.66x) | 35.0 / 61.2 (0.57x) | 374 / 266 (**1.41x**) | 386 / 387 (1.00x) | 355 / 266 (**1.34x**) | 396 / 388 (1.02x) |
| 64 | 21.2 / 25.6 (0.83x) | 22.2 / 30.3 (0.73x) | 257 / 137 (**1.87x**) | 284 / 197 (**1.44x**) | 251 / 137 (**1.83x**) | 299 / 197 (**1.52x**) |
| 128 | 17.5 / 13.8 (**1.26x**) | 18.6 / 16.0 (**1.16x**) | 198 / 70.4 (**2.81x**) | 232 / 106 (**2.18x**) | 197 / 70.3 (**2.80x**) | 232 / 107 (**2.18x**) |
| 256 | 12.9 / 9.7 (**1.32x**) | 13.7 / 10.5 (**1.30x**) | 174 / 41.6 (**4.19x**) | 195 / 62.2 (**3.14x**) | 168 / 41.4 (**4.06x**) | 192 / 61.6 (**3.11x**) |
| 512 | 10.5 / 7.0 (**1.50x**) | 11.0 / 7.6 (**1.44x**) | 166 / 22.8 (**7.29x**) | 171 / 35.8 (**4.78x**) | 153 / 22.7 (**6.72x**) | 172 / 35.3 (**4.88x**) |
| 1024 | 9.3 / 3.5 (**2.64x**) | 9.7 / 3.9 (**2.50x**) | 152 / 14.7 (**10.4x**) | 174 / 22.2 (**7.81x**) | 152 / 14.7 (**10.3x**) | 168 / 22.0 (**7.62x**) |
| 2048 | 8.4 / 1.8 (**4.75x**) | 8.8 / 2.1 (**4.27x**) | -- | -- | -- | -- |

Per step, the A100 has a nearly flat floor:
* gateway: 1.6 us/step up to N = 128, 3.6 us/step at N >= 512;
* LTA: 5.6-9 us/step up to N = 128, 15 us/step at N = 1024.

The floor is one block's critical path (FP64 latency of force, noise, deposit) plus, for multi-block grids, one grid
barrier (about 1 us). The time is split by phase in Figure B. FR costs per step on the A100 (after A1):
* gateway: +8-17 %;
* LTA: +44-57 %, because FR runs every 5 steps; its per-opportunity cost (KDE, scores, selection, one extra grid
  barrier) is about 20 us.

The NSCC CPU cores are about 2x slower per walker-step than the EPYC 9554 of the hybrid short-T study:
* gateway S1: 118 vs 52 ns;
* LTA S1: 1450 vs 845 ns.

For that reason this study re-measured every CPU route on NSCC, and pairs no number with the earlier host.

## 3. Exploration (24 seeds per arm, `results/nscc_a100/analysis/prod_tables.md`, Figure C)

Median W_mid (s) on the routed backend. Bold: the best arm per method. "cens." = censored runs out of 24.

| N | gateway ABF | gateway FR | LTA 300 K ABF | LTA 300 K FR | LTA 150 K ABF | LTA 150 K FR |
|---|---|---|---|---|---|---|
| 16 | 4.24 (P2) | 4.44 (P2) | 14.5 (P8) | 14.1 (P8) | 4.31 (P8) | 5.25 (P8) |
| 32 | **2.91** (P4) | **1.95** (P4) | 8.73 (G1) | 11.1 (G1) | 4.28 (G1) | 6.33 (G1) |
| 64 | 3.09 (P8) | 2.37 (P8) | 4.87 (G1) | 6.41 (G1) | 3.00 (G1) | 2.87 (G1) |
| 128 | 3.62 (G1) | 2.30 (G1) | 4.36 (G1) | 6.08 (G1) | **2.91** (G1) | 2.45 (G1) |
| 256 | 4.46 (G1) | 1.96 (G1) | 4.00 (G1) | 5.44 (G1) | 3.99 (G1, 3 cens.) | **2.33** (G1) |
| 512 | 6.24 (G1, 4 cens.) | 2.24 (G1) | **3.27** (G1) | **4.38** (G1, 1 cens.) | censored (22) | 2.49 (G1) |
| 1024 | censored (22) | 2.46 (G1) | censored (15) | 5.39 (G1, 10 cens.) | censored (24) | censored (15) |

**[exploratory]** Best-allocation R_wall (W_mid):

| system | R_wall [95 % CI] | best ABF N | best FR N |
|---|---|---|---|
| gateway | 0.67 [0.35, 1.04] | 32 | 32 |
| LTA 300 K | 1.34 [1.10, 1.70] | 512 | 512 |
| LTA 150 K | 0.80 [0.68, 0.98] | 128 | 256 |

The confirmation shrank the gateway and LTA 300 K effects toward 1 (the winner's curse), and the LTA 150 K effect
grew slightly (0.80 to 0.72).

FR keeps more large-N rungs inside the budget: ABF is censored where FR is not. This is why the GPU, which is fastest
at large N, helps FR more on LTA 150 K, the establishment-limited system.

**Figure F** (`figures/nscc_a100/all_N_T_wallclock/`) shows the full convergence curve at every rung of this table.
* Curves: e_F against measured wall-clock, ABF vs FR on the routed backend, median and 25-75 % over the 24 seeds.
* Each panel marks the median W_mid shown above, with its censored count.
* There is one page per system and two clocks: `timed` (engine start) and `e2e` (process start, so W_mid_e2e).
* The plotted quantiles and the W_mid quartiles are written to CSV next to the figures.
* The curves span only the window where every seed has a save; final values are not carried past a run's end.
  Figure E now uses the same window. Its first version carried each run's final value forward.

## 4. Confirmation details (`results/nscc_a100/analysis/confirm_tables.md`, Figures D-E)

| system | arms | n | R_W | R_e2e | R loose (W) | R strict (W) | final e_F ABF / FR |
|---|---|---|---|---|---|---|---|
| gateway | ABF 32 P4 vs FR 32 P4 | 128 / 128 | 0.87 [0.65, 1.23] | 0.98 [0.82, 1.14] | 0.68 [0.59, 0.79] | 1.21 [0.90, 1.55] | 0.0009 / 0.0010 |
| LTA 300 K | ABF 512 G1 vs FR 512 G1 | 64 / 64 | 1.25 [1.05, 1.55] | 1.22 [1.05, 1.49] | 0.95 [0.87, 1.03] | both censored | 0.061 / 0.054 |
| LTA 150 K | ABF 128 G1 vs FR 256 G1 | 128 / 128 | 0.72 [0.64, 0.83] | 0.78 [0.72, 0.88] | 0.78 [0.69, 0.90] | both censored | 0.053 / 0.057 |

* Censored at mid: LTA 300 K 4 (ABF) and 5 (FR) of 64; none elsewhere.
* Holm, over gateway and LTA 150 K: LTA 150 K p < 1e-4 at alpha 0.025 is significant; gateway p = 0.82 is not.
* The gateway has an early-accuracy advantage of FR at the loose threshold (R 0.68, CI below 1), which vanishes at
  mid and strict. By the preregistered interpretation this is an early-accuracy acceleration, not a general speed-up.

## 5. Validity

* **Stage I** (`VALIDATION.md`, all PASS on the frozen code). GPU and CPU trajectories from identical IC, Philox
  keys and FR streams stay pathwise coupled over whole runs, with identical bin counts and identical FR deaths. The
  FR selection is identical on 1 800 random opportunities. Results are bitwise independent of the launch
  configuration and reproducible.
* **V4 law check: PASS.** GPU vs CPU on 16 validation seeds per arm, full budgets, 12 arms. Every ratio of medians
  (final e_F, Ibar_F, u_mid) has a 95 % CI containing 1. The gateway ABF medians agree within 0.2 %.
* **Amendment A1**, an FR-path speed-up before any production run, changed timing only: 7 re-runs reproduced the
  frozen-code science bitwise.
* **Deviations, all recorded in the preregistration:**
  * the TRAJ gate tolerance was set to 1e-6 (see the prereg);
  * CPU jobs ran on one 64-core socket per PBS job, since whole nodes were unavailable, with exclusive CCDs per run;
  * the LTA kernel race found by the TRAJ gate was fixed before the freeze;
  * `bench/summary.json` was regenerated after the A1 FR re-measurement. The frozen routing (`routing/routing.json`,
    ABF-based) was not changed.

## 6. Resources

* GPU: 7.7 A100 card-hours of process wall (production 4.3, law check 1.2, confirmation 0.9, benchmarks 1.25). The
  PBS reservations were longer, because they also covered builds, the Stage I runs and idle tails.
* CPU: 140 used core-hours (production 96, law check 23, confirmation 12, benchmark 10).
* 3 898 timed runs in the result records (each a fresh process), plus 7 Amendment-A1 re-runs; 0 failed.

## 7. Interpretation (the preregistered map)

* **LTA 150 K:** "FR passes both W and W_e2e superiority tests against the best eligible ABF allocation". This is a
  supported, hardware-specific practical advantage of FR on one A100 under the frozen budgets and thresholds.
* **LTA 300 K:** the A100 makes large-N ABF fast enough that ABF wins. The FR gain per N does not survive at the
  best deployment.
* **Gateway:** "A100 beats CPU at large N, but best moderate-N CPU deployments still win" on W. The FR advantage is
  unresolved at mid accuracy and present only at loose accuracy.

The per-step cost of GPU FR on LTA (+44-57 % after A1) is the main remaining engineering lever. A further cut would
move the LTA 300 K comparison toward FR, but it was not attempted after the selection was frozen.
