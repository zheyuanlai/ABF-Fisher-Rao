# Hybrid wall-clock study at the shortened equal budgets: results

*2026-10-18, 07:13–15:28 UTC.*
* Design: `PREREGISTRATION.md`, frozen at `ef3b6453`, with Amendments A1, A2 (A2.4 revised, A2.4b) and A3.
* Validation: `VALIDATION.md`. Hardware: `BENCHMARK_RESULTS.md`. Reproduce or resume: `HANDOFF.md`.
* Tables: `results/hybrid_short_T/analysis/tables.md`; all numbers: `analysis/summary.json`.
* Figures A–F: `figures/hybrid_short_T/`, all passing the legibility check.

**Claim labels.**
* **[identity]**: verified bitwise.
* **[measured]**: timings on this host.
* **[exploratory]**: selection and estimation on the same seeds.
* **[confirmatory]**: fresh seeds with the selection fixed beforehand.
* **[projection]**: timing model; never presented as a measurement.

## 1. Answer

**No: a wall-clock advantage of the best ABF + FR deployment over the best ABF deployment is not established on any
of the three systems.** The preregistered confirmatory test (top-3 arms per method, 32 fresh seeds per arm, W and
W_e2e co-primary) is **unresolved** everywhere. The Holm-adjusted one-sided p is ≥ 0.48 for "FR faster" and ≥ 0.82
for "ABF faster".

| system | exploratory R_wall (W) [95 % CI] | **confirmatory R (W)** | **confirmatory R (W_e2e)** | **decision** |
|---|---|---|---|---|
| entropic gateway | 0.72 [0.46, 0.95] | **0.78 [0.49, 1.20]** (FR 256 / ABF 32) | **0.90 [0.74, 1.10]** | **unresolved** |
| ethane / LTA 300 K | 1.24 [0.78, 1.94] | **1.11 [0.85, 1.59]** (FR 64 / ABF 32) | **1.10 [0.85, 1.52]** | **unresolved** |
| ethane / LTA 150 K | 0.74 [0.41, 1.13] | **0.95 [0.76, 1.33]** (FR 64 / ABF 16) | **0.95 [0.79, 1.29]** | **unresolved** |

R = (best ABF + FR median wall seconds to sustained mid accuracy) / (best ABF median); R < 1 favours FR.
"Unresolved" means the CI contains 1. It is not a claim of equivalence. The data remain compatible with:
* gateway: from FR about 2× faster to ABF about 1.2× faster;
* LTA 300 K: from FR 1.2× faster to ABF 1.6× faster;
* LTA 150 K: from FR 1.3× faster to ABF 1.3× faster.

**The gateway's exploratory advantage did not replicate.** Its exploratory CI excluded 1 for both W and W_e2e. The
exploration winners then regressed on fresh seeds:
* ABF N 32: 1.42 → 1.83 s;
* FR N 128: 1.03 → 1.44 s.

This is the winner's curse the confirmation stage was designed to catch.

**The central question.** Can many short, parallel ABF + FR replicas beat the strongest ABF deployment that uses
fewer, longer trajectories? On this CPU hardware the strongest ABF deployment is **itself parallel and at moderate
N**: N = 16–64 on 2–8 cores, never the serial small-N run. Against it, many-replica FR is not measurably faster.
* **[measured]** Few-replica ABF on one core (N ≤ 4; S1 for LTA, P1 for the gateway) is 1.6–5.5× slower in
  wall-clock than the best ABF deployment.
* **[measured]** FR at the largest N (2048 / 1024), where its per-N advantage is largest, is not competitive. Large N
  spends so much of the fixed budget per unit physical time that ABF there does not reach mid accuracy within B at
  all, and FR does so in only a third to a half of the runs.

**The H200 was not used** (all four were busy with other users' jobs; user instruction "CPU only"). The CPU/H200
crossover is unmeasured. The "accelerator" for large N is the validated walker-parallel CPU engine, up to 16 cores
per run.

## 2. What was executed

| stage | configurations | jobs | attempts (incl. contention repeats) | failed / blocked | used core-h |
|---|---|---|---|---|---|
| I validation | 49 checks | 49 | 54 (8 environment errors re-run) | 0 FAIL; no backend blocked | 3.1 |
| II benchmark | 342 configurations × 3 repeats | 1026 | 1224 complete (+6 terminated at driver restarts, re-run) | 0 | 11.4 |
| II long-window supplement (A2.5) | 66 × 3 | 198 | 206 | 0 | 8.3 |
| III production (exploration) | 3 systems × every N × ABF / FR (65 arms); 32 / 16 / 16 seeds | 1408 | 1502 | 0 failed; 13 jobs with no uncontended replicate (flagged) | 97.2 |
| III confirmation | 3 systems × 6 arms × 32 fresh seeds | 576 | 594 | 0 failed; 1 flagged | 49.2 |
| **total** | | | | | **169.2 used core-h** (cap 300, A3); **0 H200-h** |

**Completeness.** All 1408 production and 576 confirmation jobs are complete. Nothing was skipped or trimmed: all 32
seed blocks fit the cap.

**Allocated core-hours.** Exclusive CCD reservations count idle cores too: about 281 core-h (benchmark 35.7,
production 192.9, confirmation 52.4).

**Seeds.** Fresh: gateway 9600–9631 / 9640–9671; LTA 300 K 30500–30515 / 30520–30551; LTA 150 K 15700–15715 /
15720–15751. They were checked against every earlier seed and random stream (`seed_check.json`).

**Data and integrity.**
* Manifest `results/hybrid_short_T/SHA256SUMS.txt`: 2.37 GB, 0 completeness problems.
* Integrity check (`smoke_check.py`) on every production and confirmation record: 0 problems. It covers the save
  grid, force-evaluation accounting, science sha256, finite scores, contention verdicts, ledger marks and the frozen
  route.

## 3. Hardware: CPU throughput and routing (summary of `BENCHMARK_RESULTS.md`)

* **[measured]** Gateway: no CPU crossover.
  - The walker-parallel engine beats the serial engine at every N. On one thread (P1), 1.24–1.46× comes from a
    faster implementation (C kernel + Philox noise), **not** from parallelism.
  - Parallel speed-up starts at N 16 (P2) and reaches 12.9× at N 2048 (P16).
* **[measured]** LTA: the serial engine is fastest for N ≤ 4. P8 takes over at N 8 (1.7×); P16 gives 6–10.5× at
  N ≥ 128.
* **[measured]** FR's extra cost per step:
  - gateway 4–7 % (P2–P16), up to 14.5 % at N 2 (P1);
  - LTA 9–14 % on P8 / P16 (FR is serial while forces run on 8–16 threads), 7–15 % on one core at N 4 / 2.
* **[measured]** Start-up (process → timed start): S1 0.44 s, LTA parallel 0.53 s, gateway parallel 1.72 s (1.45 s of
  it the engine warm-up).
* **[measured]** The benchmark predicts production completion walls to a median ratio of 1.000 (range 0.93–1.26 over
  2096 records).
* Frozen routes. The common route equals each method's own fastest route at every N.

  | system | route by N |
  |---|---|
  | gateway | P1 (N ≤ 8), P2 (16), P4 (32), P8 (64–512), P16 (1024–2048) |
  | LTA | S1 (N ≤ 4), P8 (8–64), P16 (128–1024) |

## 4. Accuracy at the shortened budget (fresh seeds; reproduces the short-T study)

e_F(T_N), median, FR / ABF paired ratio [95 % CI] **[measured]**:

| system | N₀ | mid-N | small N | original short-T (truncated stored runs) at N₀ |
|---|---|---|---|---|
| gateway | 2048: 0.0124 vs 0.0055, **0.45 [0.39, 0.50]** (32/32 FR better) | 128: 0.61 [0.47, 0.92]; 32: **1.74 [1.23, 2.57]** (FR worse) | 2–8: 1.2–1.4 (only N 8 excludes 1) | 0.0115 vs 0.0055, 0.46 [0.41, 0.53] |
| LTA 300 K | 1024: 0.109 vs 0.095, 0.90 [0.80, 0.97] | 128: 1.01; 32: 0.91 [0.82, 1.34] | ≈ 1 | 0.107 vs 0.094, 0.91 [0.82, 1.00] |
| LTA 150 K | 1024: 0.358 vs 0.156, **0.45 [0.40, 0.49]** | 128: 0.77 [0.55, 1.06]; 32: 0.89 [0.75, 0.93] | ≈ 1.0–1.3 | 0.342 vs 0.129, 0.39 [0.34, 0.44] |

* The equal-budget pattern replicates on fresh seeds: FR helps strongly at large N and is neutral or harmful at
  small N.
* At gateway N 32, FR's endpoint error is worse (1.74×).
* Figure C shows the physical-time view at every N.

## 5. Measured wall-clock time to the mid threshold, every N (Figures B, E)

W_mid: median seconds since the engine's timed start, censored fraction in brackets, FR / ABF paired ratio
**[measured]**. Full rows with the e2e and loose / strict variants are in `analysis/tables.md`.

| N | gateway ABF | gateway FR | ratio | LTA 300 K ABF | LTA 300 K FR | ratio | LTA 150 K ABF | LTA 150 K FR | ratio |
|---|---|---|---|---|---|---|---|---|---|
| N₀ | cens. (32/32) | cens. (16/32) | — | cens. (13/16) | cens. (8/16) | — | cens. (16/16) | cens. (11/16) | — |
| N₀/2 | cens. (30/32) | 3.66 (3/32) | FR only | 16.1 (3/16) | 12.5 (1/16) | 0.77 [0.64, 1.19] | cens. (16/16) | 6.44 | FR only |
| 256 | 3.99 | 1.75 | **0.44 [0.27, 0.60]** | 8.48 | 10.9 | 1.29 [0.76, 1.81] | 8.97 | 4.62 | **0.52 [0.44, 0.65]** |
| 128 | 2.40 | **1.03** | **0.43 [0.33, 0.55]** | 10.5 | 7.15 | 0.68 [0.47, 1.30] | 5.32 | 3.08 | **0.58 [0.41, 0.78]** |
| 64 | 2.09 | 1.28 | **0.61 [0.31, 0.88]** | 6.40 | 9.12 | **1.42 [1.13, 2.46]** | 4.77 | 3.03 | 0.64 [0.30, 1.07] |
| 32 | **1.42** | 1.75 | 1.23 [0.94, 1.83] | **5.75** | 8.29 | 1.44 [0.74, 2.92] | **3.64** | **2.69** | 0.74 [0.43, 1.32] |
| 16 | 2.21 | 2.74 | 1.24 [0.64, 1.81] | 8.44 | 9.65 | 1.14 [0.68, 2.08] | 4.71 | 3.97 | 0.84 [0.62, 1.10] |
| 8 | 2.46 | 3.72 | 1.51 [0.86, 2.28] | 13.1 | 18.5 | 1.41 [0.71, 2.03] | 4.84 | 6.57 | 1.36 [0.75, 3.08] |
| 4 | 2.32 | 3.29 | 1.42 [0.81, 2.33] | 31.9 | 37.4 | 1.17 | 9.84 | 11.5 | 1.17 |
| 2 | 2.61 | 3.39 | 1.30 | 27.1 | 27.1 | 1.00 | 9.66 | 12.5 | 1.29 |
| 1 | 3.06 | — | — | 19.9 | — | — | 13.6 | — | — |

Notes:
* N₀ is 2048 (gateway) or 1024 (LTA); N₀/2 is 1024 or 512.
* "FR only" means the ABF median is censored while FR reaches the threshold.
* **[measured]** At fixed N, FR reaches mid accuracy sooner at large N: gateway N ≥ 64 and LTA 150 K N ≥ 128, CI
  below 1. At small N it is slower or neutral.
* **[measured]** In wall-clock, the large-N advantage is cancelled because large N is slow per unit physical time:
  B / N steps per walker-step budget. Both methods' best N therefore sit at moderate N (16–128), where FR's per-N
  advantage is small or reversed.
* **[measured]** Fixed largest N:
  - Gateway N 2048: ABF never reaches mid within B (32/32 censored); FR reaches it in 16/32 runs.
  - LTA 150 K N 1024: ABF 16/16 censored, FR 11/16.

  This is degenerate by construction: B was anchored where the FR median at N₀ just settles (A2.10). The loose
  threshold gives a measurable fixed-N₀ contrast: FR / ABF W_loose 0.60 [0.56, 0.66] (gateway), 0.79 [0.67, 0.85]
  (LTA 300 K), 0.55 [0.45, 0.62] (LTA 150 K), FR first in every run.

## 6. Best allocations, resource cost and the frontier

**Exploratory** (re-selection bootstrap, exploration seeds):

| system | best ABF | best FR | how often that N wins the resampled selection (ABF / FR) |
|---|---|---|---|
| gateway | N 32 (P4) 1.42 s | N 128 (P8) 1.03 s | 0.99 / 0.72 |
| LTA 300 K | N 32 (P8) 5.75 s | N 128 (P16) 7.15 s | 0.66 / 0.50 |
| LTA 150 K | N 32 (P8) 3.64 s | N 32 (P8) 2.69 s | 0.76 / 0.55 |

**Confirmatory, per arm** (32 fresh seeds each; W_mid median s; force evaluations and used core-h to the mid
threshold, medians) **[measured]**:

| system | ABF arms | FR arms |
|---|---|---|
| gateway | **N 32 P4 1.83 s** (8.6 × 10⁷ f.e., 0.0020 core-h); N 64 P8 1.84 s; N 16 P2 2.00 s (5.7 × 10⁷, 0.0011 core-h) | **N 256 P8 1.43 s** (1.38 × 10⁸, 0.0032 core-h); N 128 P8 1.44 s (1.21 × 10⁸); N 64 P8 1.93 s |
| LTA 300 K | **N 32 P8 5.40 s** (2.4 × 10⁷, 0.012 core-h); N 64 P8 6.39 s; N 16 P8 10.1 s | **N 64 P8 5.99 s** (3.1 × 10⁷, 0.013 core-h); N 32 P8 7.96 s; N 128 P16 8.15 s |
| LTA 150 K | **N 16 P8 3.03 s** (9.6 × 10⁶, 0.0067 core-h); N 32 P8 3.08 s; N 64 P8 4.55 s | **N 64 P8 2.88 s** (1.5 × 10⁷, 0.0064 core-h); N 128 P16 3.35 s; N 32 P8 4.51 s |

**Resource cost [measured].**
* Gateway: the fastest FR arm needs **2.4× the force evaluations and 2.9× the used core-hours** of the cheapest
  near-fastest ABF arm (N 16 on 2 cores, 2.00 s). Its point-estimate wall-clock lead of about 20–25 % is bought with
  more parallel hardware, not with fewer force evaluations.
* LTA: FR's best confirmed arm uses 28–54 % more force evaluations than ABF's, at similar used core-hours (−5 to
  +11 %).

**Frontier (Figure D, descriptive) [measured].** The best-over-N median e_F at wall budget w is nearly identical for
the two methods on all three systems:
* gateway: FR a little lower between about 0.15 and 0.4 s;
* LTA 150 K: FR a little lower between about 2 and 20 s;
* LTA 300 K: ABF a little lower between about 3 and 20 s.

Near the threshold crossing, the minimising N is 16–128 for both methods; it is noisy at the shortest budgets, where
every configuration is still near its initial error. Runs that have completed are carried forward
(dashed), never shown as continued sampling.

## 7. Confounders and sensitivity

**Confounders, and how each is controlled.**
* **Backend.**
  - At fixed N both arms share the route. Across N the backends differ by design.
  - **[identity]** Replay mode equals the serial engine bitwise (V3), so the parallel engines run the same
    algorithm.
  - **[measured]** Law check: counter-noise campaign runs vs the stored validated runs truncated at u*, about 190
    comparisons of e_F(T_N), Ibar_F and τ_mid u. 9 exclude 1 at 95 %, about what chance gives, with no consistent
    direction.
  - Validated-vs-validated null blocks spread just as widely: median ratios 0.82–1.67.
  - The hybrid projection with the validated engine's law gives the same conclusion.
* **Precision, estimator, reference.** float64 throughout; the frozen scorer and references (V7 bitwise).
* **FR strength.** The frozen production `engine_cfg`. V8: LTA FR from step 20 000, gateway ramp 4 t.u.
* **Initial conditions and noise.** Shared slot by slot between the arms (V5).
* **GPU overhead.** None: no GPU was used.
* **Contention.**
  - 1 Hz telemetry and repeats.
  - A contended replicate is a median 0.3–3.5 % slower.
  - Flagged: 13 / 1408 production jobs, 1 / 576 confirmation jobs.
  - Excluding flagged runs leaves every R unchanged to ≤ 0.08.
* **Selection bias.** Fresh-seed confirmation with min-of-3 re-selection on both sides. T* was chosen on the original
  seeds, never on these.
* **Save-grid quantization.** A2.1: persistence on a ≤ 2 % grid. The gap ratio at τ was 1.01–1.02 for every best
  configuration; W_interp changes R by ≤ 0.01.

**Sensitivity of the exploratory R_wall** (gateway / LTA 300 K / LTA 150 K):

| variant | gateway | LTA 300 K | LTA 150 K |
|---|---|---|---|
| W (primary) | 0.72 | 1.24 | 0.74 |
| W_e2e (start-up included) | 0.87 [0.74, 0.98] | 1.23 | 0.77 |
| W_interp | 0.72 | 1.25 | 0.74 |
| flagged runs excluded | 0.72 | 1.16 | 0.74 |
| throughput ranges against FR | 0.75 | 1.25 | 0.76 |
| throughput ranges against ABF | 0.71 | 1.23 | 0.72 |
| deployment-optimised routes | identical routes | identical | identical |
| hybrid projection, validated law | 0.71 [0.43, 1.27] [projection] | 1.21 [0.87, 1.55] | 0.64 [0.46, 1.18] |

The confirmatory decision is the same under W, W_e2e, W_interp and unflagged-only.

## 8. Limitations

* **No H200 data.** The CPU/H200 crossover and any GPU-backed large-N deployment are untested in this study. On the
  only H200 data (Experiment III, torch eager, 2026-10-10), the GPU was slower than one CPU core at N 512 with about
  25 % FR overhead. A fused or graph-captured GPU kernel could change the large-N economics. That is a separate study.
* **The wall-clock times are short** (1–15 s for the best configurations). Start-up overhead (0.4–1.7 s) is
  therefore material; W and W_e2e were treated as co-primary for that reason.
* **The conclusions are conditional.** They hold for the frozen shortened budgets, which are anchored at FR's T* at
  N₀ and make large-N ABF censored by construction. They hold for the mid threshold as primary. And they hold for this
  host's 16-core-per-run convention on a machine shared with other users' processes; telemetry was recorded and the
  contention rule enforced.
* **The confirmatory design detects about a 1.3–1.6× effect.** With 32 seeds per arm and the observed spread, the
  CIs span about ×/÷1.25–1.6. A smaller real difference (e.g. the gateway's point estimate of 0.78) could exist
  undetected.
* **Other ABF deployments were not compared.** For example, averaging independent ABF runs on the same cores lies
  outside this design.
* **Amendments.** A1, A2 (incl. the A2.4 revisions and A2.4b) and A3 changed allocation timing, repeat policy, W
  resolution, CI computation, the confirmation design and the CPU cap.
  * Every amendment preceded the first production run. A1, A2 and A3 were made before or during the benchmark;
    A2.4b came after it.
  * The routing was frozen before production. The confirmation selection was frozen before any confirmation run.
  * No amendment was made after any accuracy outcome of this study's seeds had been seen.
