# FR computational overhead: component breakdown (gateway and ethane/LTA engines)

*2026-10-10.* Harness: `scripts/equal_budget/measure_fr_overhead.py`. Raw numbers:
`results/equal_budget_v2/fr_overhead/fr_overhead.json`. Per-case worker output and logs:
`results/equal_budget_v2/fr_overhead/parts/`. Driver log: `results/equal_budget_v2/fr_overhead/driver.log`.

The walker-step budget B counts force evaluations only. This note measures what FR costs on top of that, in CPU
time, and splits it into components. Engines: `src/gateway_ladder_numba.py` and `src/lta_ladder_numba.py`, with
the production `engine_cfg` of `configs/equal_budget_v2/{gateway,lta_300K}_production.json`.

## Summary

* **FR adds a roughly fixed cost per FR opportunity, c_opp ≈ a + b·N.**
  * Gateway: a ≈ 1.6 µs and b ≈ 25 ns per walker (2.4 µs at N = 16, 53 µs at N = 2048).
  * LTA: a ≈ 1.0 µs and b ≈ 41 ns per walker (2.0 µs at N = 16, 43–45 µs at N = 1024).
  * That cost is spread over fr_every·N walker-steps: 160·N × 77 ns for the gateway and 5·N × 850 ns for LTA. The
    relative overhead therefore falls as 1/N toward a floor of b / (fr_every · cost per walker-step), about 0.2 %
    for the gateway and about 1 % for LTA.
* **Measured FR overhead with diagnostics off** (median [min, max] over 5 interleaved repetitions):
  * Gateway: 8.66 % [8.37, 8.88] at N = 2, 1.45 % [−0.40, 2.24] at N = 16, 0.32 % [0.20, 0.47] at N = 2048.
  * LTA 300 K: 13.93 % [13.85, 13.99] at N = 2, 3.27 % [2.95, 3.36] at N = 16, 1.09 % [0.99, 1.36] at N = 1024.
* **What the cost is made of.**
  * At small N it is the N-independent loops over the 181- (gateway) or 180-point (LTA) grid, which run once per
    opportunity. The largest is the KL term of the score, which takes one log per grid point. The others are zeroing,
    normalising and clamping the density.
  * At large N it is per-walker work. Both engines take a log per walker in the score. The gateway also evaluates an
    exp per walker in selection and draws the RNG block. LTA recomputes the post-move CV (an fmod per walker) and
    takes a second log per walker.
  * **Resampling is small everywhere:** at most 0.63 %, counting selection, state copy and RNG. Realised events are
    rare at the production steady state.
* **Per-step scheduling check (gateway only).** The gateway kernel tests `step % fr_every` on every step. This
  costs 2.75 ns per step, or 1.8 % at N = 2, and is not resolved at larger N. In LTA it is negligible (≤ 0.1 %).
* **Diagnostics.**
  * Cost to both arms (ABF diagnostics on vs off): gateway 3.1 % / 1.3 % / 0.8 % (N = 2 / 16 / 2048); LTA 1.1 % /
    1.5 % / 1.4 % (N = 2 / 16 / 1024).
  * Diagnostics that only the FR arm runs add another 0.0–1.2 %: gateway 1.2 % / 0.9 % / 0.0 %, LTA 1.0 % /
    0.3 % / 0.3 % (N = 2 / 16 / large N). These are differences of differences. A re-run reproduced gateway N = 2
    and large N but not gateway N = 16 or LTA N = 2 (§5).
* **Cross-check.** Where the overhead is resolved, the micro-benchmarked components plus the scheduling check give
  0.90–0.96 of the measured total: gateway N = 2, and LTA at all three N. Gateway N = 16 (0.86) and N = 2048 (0.67)
  agree within the timing noise of the full runs. The micro-benchmarks run on cache-hot data, which plausibly
  explains the 5–10 % underestimate (this attribution was not tested). An independent reviewer check, which fitted
  the slope of full-run wall time against the number of FR opportunities (fr_every 160/80/40 for the gateway,
  20/10/5 for LTA, N = 2, 3 reps), gave 1.67 µs per opportunity for the gateway and 1.17 µs for LTA. That agrees
  with the measured (FR − FRsched) 1.72 and 1.19 µs. Its gateway FRsched − ABF difference gave 3.0 ns per step,
  against 2.75 ns here.
* **Production ledgers agree where they can resolve the overhead.**
  * Gateway N = 2: 9.71 % in the ledger vs 9.54 % measured here (diagnostics on).
  * LTA N = 2: 14.27 % vs 14.75 %. LTA N = 16: 3.52 % vs 3.50 %.
  * At large N the ledger ratios are dominated by noise: the per-seed IQR spans 4–5 percentage points.
  * Consequence: the `FINAL_RESULTS.md` §7 figure "−0.7 % at N = 1024 (300 K)" is noise; it is the ratio of medians
    of noisy wall times. The clean value is **+1.4 %** with diagnostics on (as in production) and +1.1 % with them off,
    both with FR active for the whole short run. On the production schedule FR is active for 93 % of the steps at
    N = 1024 (from step 20000 of 300000), so the expected production value is ≈ +1.3 %.

## 1. Method

**Full short runs (parts a and b).**
* Each case calls the engine's own `run_arm` with the production `engine_cfg` and the production save-grid rule
  (`run_ladder.save_grid`). The seed is the first production seed (gateway 8100, LTA 30000).
* `n_steps` is shortened so that one ABF run takes about 12 s. It is calibrated on the spot after one warm-up run of
  every variant.
* There are five variants, each run 5 times:
  * ABF, diagnostics off and on;
  * ABF+FR, diagnostics off and on;
  * "FRsched": the FR arm with fr_every = n_steps. It has a single FR opportunity, so it pays the per-step FR
    scheduling test and nothing else.
* The run order is rotated in every repetition, so each variant occupies each position once. Ratios are formed
  **within a repetition** and summarised by their median and min–max over the 5 repetitions.
* All runs of a variant are deterministic: same seed, same n_steps. The spread across repetitions is therefore
  timing noise only.
* LTA only: `fr_start_steps = 5` (production: 20000) so that FR is active for the whole short run. In production
  it is active for 93 % (N = 1024) to ~100 % (N ≤ 16) of the steps.

**Micro-benchmarks (part c).**
* Each component runs in an njit loop with no Python call per iteration. The loops call the engines' own njit
  helpers.
* They use five representative population states:
  * the final states of the production FR arm at the same N (first 4 seeds, read only);
  * the end state of the case's own short FR run.
* Gateway components:
  * KDE = `kde_density`; score = `uniform_scores`;
  * selection = `fr_resample`, with the engine's uniform-block layout and the fully ramped rate γ = 1.5;
  * gather = an op-for-op copy of the inline slot gather in `_advance`, charged per opportunity with ≥ 1 candidate;
  * RNG = the Python-side PCG64 draw of the FR uniform blocks, at the engine's chunk size.
* LTA components:
  * post-move CV/binning = an op-for-op copy of the inline loop in `_run_chunk` step (7);
  * KDE and score = an op-for-op split of `_fr_score` into its density part and its score part. The split is asserted
    **bitwise equal** to `_fr_score` on every state, and the sum of the two parts is 0.93–1.04× the timed whole
    `_fr_score`;
  * selection = `fr_select_native`; copy = an op-for-op copy of the event-application loop, charged per realised
    event; RNG = `gen_fr.random((n_opp, N + 2 cap))` at the engine's chunk size;
  * diagnostics-only FR work = `_true_event` on the post-move positions.
* Component time per opportunity × opportunities per short run ÷ the ABF diagnostics-off wall time gives the
  component's share in %.

**Host.**
* 2 × AMD EPYC 9554 (256 logical CPUs, SMT-2). numba 0.68.0, numpy 2.4.6, Python 3.14.4.
* `OMP_NUM_THREADS = NUMBA_NUM_THREADS = 1`, `NUMBA_CACHE_DIR = ~/.cache/numba_eqb`.
* The six cases ran concurrently, each pinned with `taskset -c` to one idle logical CPU on its own physical core:
  CPUs 178, 209, 113, 215, 17 and 114. CPU 0 was excluded. Six cores were used in total.

## 2. Overhead breakdown (short runs, % of the ABF diagnostics-off wall time)

The time per walker-step is the run's wall time / (N · n_steps), with median [min, max] over 5 repetitions.
Columns:
* "per-step FR check" = FRsched − ABF.
* "sum" = KDE + score + resampling + the scheduling check, with a negative scheduling estimate counted as 0.
* For LTA, the KDE column includes the post-move CV recomputation, which is also shown on its own in parentheses.
* Component columns are evaluated at the end state of the case's own short FR run.

| system | N | n_steps (short run) | FR opps/run | ABF ns/ws, diag off | ABF ns/ws, diag on | diagnostics cost, ABF (%) | FR total overhead, diag off (%) | KDE (%) | score (%) | resampling (%) | per-step FR check (%) | sum (%) | sum / measured |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gateway | 2 | 77,313,280 | 483,208 | 77.6 [77.5, 77.8] | 80.0 [79.9, 80.2] | 3.06 [2.79, 3.37] | **8.66** [8.37, 8.88] | 3.28 | 3.11 | 0.18 | 1.77 [1.69, 1.99] | 8.35 | 0.96 |
| gateway | 16 | 9,675,200 | 60,470 | 76.4 [76.3, 77.8] | 77.3 [77.3, 77.8] | 1.29 [−0.62, 1.95] | **1.45** [−0.40, 2.24] | 0.69 | 0.47 | 0.09 | −0.12 [−0.97, 0.83] | 1.25 | 0.86 |
| gateway | 2048 | 78,240 | 489 | 75.0 [74.9, 75.1] | 75.7 [75.5, 75.7] | 0.81 [0.72, 1.05] | **0.32** [0.20, 0.47] | 0.03 | 0.08 | 0.11 | −0.01 [−0.08, 0.14] | 0.22 | 0.67 |
| LTA 300 K | 2 | 7,028,635 | 1,405,727 | 853.1 [852.2, 853.3] | 862.8 [861.8, 864.1] | 1.11 [1.02, 1.32] | **13.93** [13.85, 13.99] | 3.01 (CV 0.19) | 8.80 | 0.63 | 0.04 [−0.11, 0.07] | 12.48 | 0.90 |
| LTA 300 K | 16 | 881,790 | 176,358 | 850.4 [849.9, 852.7] | 863.4 [861.2, 863.4] | 1.47 [1.11, 1.54] | **3.27** [2.95, 3.36] | 1.32 (CV 0.23) | 1.49 | 0.18 | 0.11 [−0.25, 0.21] | 3.09 | 0.95 |
| LTA 300 K | 1024 | 13,845 | 2,769 | 847.5 [847.0, 847.9] | 859.5 [859.0, 860.2] | 1.44 [1.31, 1.56] | **1.09** [0.99, 1.36] | 0.38 (CV 0.24) | 0.47 | 0.15 | −0.09 [−0.13, 0.27] | 0.99 | 0.91 |

## 3. Per-opportunity cross-check (diagnostics off)

Columns:
* "measured FR−ABF" = (ABF+FR − ABF) wall time per FR opportunity, median [min, max] over repetitions.
* "measured FR−FRsched" = the same with the scheduling check removed.
* Component values are µs per opportunity at the case's own end state.
* "4 production final states" = the total over the four production final states, read from the production FR arms.

| system | N | measured FR−ABF (µs/opp) | measured FR−FRsched (µs/opp) | KDE | score | selection | gather / copy | RNG | components total | total, 4 production final states | occupied cells / touched bins | candidates (gw) / events (LTA) per opp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gateway | 2 | 2.151 [2.082, 2.203] | 1.724 [1.587, 1.764] | 0.814 | 0.773 | 0.035 | 0.000 | 0.011 | 1.632 | 1.588–1.637 | 2 | 0.007 |
| gateway | 16 | 2.842 [−0.797, 4.387] | 2.633 [1.131, 4.273] | 1.352 | 0.913 | 0.103 | 0.001 | 0.071 | 2.440 | 2.367–2.428 | 16 | 0.028 |
| gateway | 2048 | 79.3 [50.2, 114.6] | 81.3 [33.7, 112.3] | 8.131 | 19.033 | 13.697 | 3.300 | 9.225 | 53.385 | 51.149–52.798 | 118 | 3.468 |
| LTA 300 K | 2 | 1.189 [1.182, 1.192] | 1.186 [1.179, 1.198] | 0.241 + CV 0.016 | 0.751 | 0.045 | 0.000 | 0.009 | 1.061 | 1.077–1.083 | 1 | 0.000 |
| LTA 300 K | 16 | 2.224 [2.011, 2.287] | 2.146 [2.118, 2.180] | 0.739 + CV 0.159 | 1.012 | 0.081 | 0.000 | 0.039 | 2.031 | 2.015–2.064 | 16 | 0.000 |
| LTA 300 K | 1024 | 47.2 [42.9, 59.1] | 48.5 [43.3, 55.9] | 6.004 + CV 10.269 | 20.246 | 4.112 | 0.000 | 2.300 | 42.932 | 44.681–45.496 | 145 | 0.035 |

Execution-only ratios, components / (FR − FRsched): gateway 0.95, 0.93 and 0.66; LTA 0.89, 0.95 and 0.88.

* At gateway N = 2048 FR is 0.3 % of the run. A paired spread of about ±0.15 % then becomes ±40 % per opportunity,
  so the measured minimum (50 µs/opp) brackets the prediction (53 µs).
* The gateway N = 2048 short run is early in the trajectory: 118 occupied cells, rate ramp incomplete. Its component
  total still equals the total at the production final states (all 181 cells occupied) to within 5 % (1.1–4.4 %).

## 4. Comparison with the production ledgers

Production ledgers: `results/equal_budget_v2/{gateway,lta_T300}/ledger.csv`, last row per (N, seed, method). The
FR and ABF arms of one seed are paired. Production ran with diagnostics on in both arms, so it compares with the
"diag on" column. The last column predicts FR's overhead for a production run of this N: the components
(diagnostics off) at the production final states × the production number of FR opportunities (LTA: from step
20000) ÷ the short-run ABF diagnostics-on cost per walker-step. It excludes the scheduling check and the
diagnostics that only the FR arm runs.

| system | N | FR overhead, diag on, short runs (%) | FR-arm-only diagnostics (%) | production ledger FR/ABF − 1 (%): paired median [IQR] | ratio of medians | ratio of sums | n pairs | production ABF ns/ws | predicted FR overhead, production schedule, components only (%) |
|---|---|---|---|---|---|---|---|---|---|
| gateway | 2 | 9.54 [9.42, 9.73] | 1.16 | 9.71 [9.30, 10.07] | 10.04 | 8.90 | 32 | 80.7 | 6.21–6.40 |
| gateway | 16 | 2.33 [1.18, 3.41] | 0.91 | 1.82 [1.17, 2.05] | 1.89 | 1.78 | 32 | 79.5 | 1.20–1.23 |
| gateway | 2048 | 0.32 [0.22, 0.40] | 0.01 | 0.60 [−0.93, 3.61] | 0.74 | 0.97 | 32 | 81.5 | 0.21 |
| LTA 300 K | 2 | 14.75 [14.57, 15.06] | 1.04 | 14.27 [13.91, 14.56] | 14.10 | 14.31 | 16 | 899.0 | 12.48–12.55 |
| LTA 300 K | 16 | 3.50 [3.33, 3.60] | 0.29 | 3.52 [2.82, 3.76] | 3.32 | 3.46 | 16 | 893.7 | 2.92–2.99 |
| LTA 300 K | 1024 | 1.42 [1.34, 1.49] | 0.27 | 0.31 [−1.36, 3.97] | −0.66 | 1.11 | 16 | 893.6 | 0.95–0.96 |

**Gateway N = 2 reconstruction.** Adding the parts gives 9.3 %, against 9.7 % in the production ledger:
* components 6.3 %;
* scheduling check 1.8 %;
* diagnostics that only the FR arm runs 1.2 %.

**LTA 150 K** (ledger only; not measured here) follows the same pattern:
* 13.89 % [13.40, 14.56] at N = 2;
* 4.00 % [3.35, 4.18] at N = 16;
* 1.37 % [0.26, 2.46] at N = 1024.

The full per-N ledger table for all three systems is in the JSON under `production_ledger`.

**Diagnostics run only by the FR arm.** This is (FR on − FR off) − (ABF on − ABF off).
* Gateway: the arm with diagnostics on evaluates a second per-step modulo, the genealogy-window test
  `step % win_steps`. It also maintains the genealogy and persistent walker-id permutation at gathers. The 1.16 %
  at N = 2 is the size expected from one more per-step modulo, but it was not isolated.
* LTA: `_true_event` at every opportunity costs 0.02 / 0.16 / 10.5 µs per opportunity by micro-benchmark. That is
  about 0.2 % of the ABF time at every N. The other 0.8 % at N = 2 is unattributed. It is a difference of
  differences of four timings.

## 5. Noise and validity

* **Machine state during these measurements.**
  * The 1-min load average was 19–23 on 256 logical CPUs, read before and after every run.
  * The SMT sibling of every pinned CPU was ≤ 2 % busy during every run (from `/proc/stat`). Thread CPU time / wall
    was ≥ 0.9998, so there was no preemption. There were 5–39 nonvoluntary context switches per 12 s run.
  * When the harness started, 11 logical CPUs were > 50 % busy. The equal-budget confirmation run, which an earlier
    snapshot showed on ~140 cores, was no longer loading the machine. The measurements therefore come from an almost
    idle host.
* **Production numbers come from a heavily loaded host.** They were measured under 110–120 concurrent
  single-threaded processes, many of them sharing physical cores with an SMT sibling.
  * The per-walker-step cost there was higher than here with diagnostics on: by 3–6 % for LTA (890–912 ns vs
    860–863 ns) and by 1–8 % for the gateway (79–82 ns vs 76–80 ns).
  * The paired per-seed FR/ABF spread is far larger: as wide as −12 % to +11 % (min to max) for the gateway at
    large N.
  * Production ratios therefore resolve the FR overhead only where it exceeds a few %, roughly N ≤ 32. Above that,
    any single summary of the ledger (paired median, ratio of medians, ratio of sums) can carry the wrong sign.
* **Short-run noise.**
  * The max–min spread of a variant over 5 repetitions was 0.1–0.6 % of its median.
  * Exception: gateway N = 16, where every variant spread 0.7–1.9 % and one ABF diagnostics-off repetition (rep 2)
    was 1.9 % slow with an idle sibling. That repetition sets the negative minima in that row; the medians are
    robust to it.
  * Differences below ~0.3 % are not resolved per case. Per-opportunity values at gateway N = 2048 and LTA N = 1024
    carry ±20–40 %.
  * **The [min, max] ranges are within one launch and understate launch-to-launch variation.** An independent
    reviewer re-run (same harness, `--reps 5 --target-s 12`, other idle CPUs 25/31/33/35/42/45, load 16–22)
    reproduced every FR-overhead median to within 0.42 percentage points. Two pairs of ranges did not overlap:
    * LTA N = 2, FR overhead (diagnostics off): 14.22 [14.17, 14.27] vs 13.93 [13.85, 13.99] here;
    * gateway N = 16, diagnostics cost: 2.34 [2.16, 2.46] vs 1.29 [−0.62, 1.95] here.

    Other re-run medians (diagnostics off / on):
    * gateway: 8.63 / 9.54 % (N = 2), 1.87 / 1.99 % (N = 16), 0.30 / 0.25 % (N = 2048);
    * LTA: 14.22 / 14.66 % (N = 2), 3.01 / 3.51 % (N = 16), 1.20 / 1.49 % (N = 1024).

    The component totals per opportunity agreed to within 6 % (largest: LTA N = 16, 1.93 vs 2.03 µs). Treat about
    ±0.3 points (±1 point for the gateway at N = 16) as the realistic uncertainty of a single launch.
    * The re-run's cross-check (components + scheduling check) / measured was 0.96 / 0.96 / 0.50 for the gateway
      and 0.90 / 0.94 / 0.84 for LTA (N = 2 / 16 / large N). The gateway N = 2048 value comes from a repetition
      disturbed by a busy SMT sibling, with per-variant spreads up to 8 %.
    * The FR-arm-only diagnostics, a difference of differences, re-measured as 1.36 / 0.08 / 0.02 % for the gateway
      and 0.63 / 0.54 / 0.31 % for LTA. It reproduces at gateway N = 2 (1.2–1.4 %) and at large N (≤ 0.3 %), but
      not at gateway N = 16 (0.91 vs 0.08 %) or LTA N = 2 (1.04 vs 0.63 %).
* **Short runs vs production.**
  * Each short run covers about 5 % of a production run's steps (12 s vs 260–280 s). Per-step costs do not depend on
    the trajectory length, but the state does: e.g. the gateway N = 2048 short run ends with 118 of 181 grid cells
    occupied. Every component was therefore also timed at the production final states (last columns of §3 and §4).
  * LTA's `fr_start_steps` was set to 5. The production-schedule column accounts for the 20000-step delay.
* **Micro-benchmarks** run on L1/L2-resident data with the same state repeated. In the kernel, the FR arrays and
  the 180 × 180 KDE matrix (LTA) compete with the dynamics for cache. This systematic underestimate is consistent
  with the 0.88–0.96 ratios where the overhead is resolved.
* **The FRsched variant** uses fr_every = n_steps, a larger divisor than in production. On Zen 4 a smaller quotient
  can shorten the integer division, so the scheduling-check cost is a lower bound.
* **Selection** is timed at the fully ramped rate γ (gateway) or `fr_rate` (LTA). The rate changes only the event
  count, and the gather/copy cost is ≤ 3.3 µs per opportunity in every case.
* **Single compile target** (znver4); bitwise behaviour is unchanged. The harness checks the KDE/score split
  bitwise against `_fr_score`.

## 6. Reproduce

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate abffr
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=~/.cache/numba_eqb
python scripts/equal_budget/measure_fr_overhead.py --reps 5 --target-s 12      # ~6.5 min on 6 CPUs
python scripts/equal_budget/measure_fr_overhead.py --summarize                 # re-merge parts, print the tables
```

The CPUs are auto-picked: idle for 4 s, on distinct physical cores, CPU 0 excluded. Use `--cpus a,b,...` to set
them by hand. The harness writes only under `results/equal_budget_v2/fr_overhead/`.
