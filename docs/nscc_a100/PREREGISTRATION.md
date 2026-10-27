# A100 wall-clock study on NSCC ASPIRE2A: ABF vs ABF + FR with a GPU backend (preregistration)

*Written 2026-10-26 and frozen in git, together with `configs/nscc_a100/campaign.json`, after the Stage I correctness
runs and the launch-tuning sweeps of the new A100 engine, and before any benchmark, law-check, production or
confirmation run. Changes after the freeze are dated amendments at the end of this file.*

## 0. What was seen before writing

* Everything of the hybrid short-T study (`docs/hybrid_short_T/*`, `results/hybrid_short_T/analysis/tables.md`): CPU
  only (2 x EPYC 9554), confirmatory best-FR / best-ABF wall-clock ratios unresolved on all three systems.
* The earlier GPU attempt (`docs/mechanism/EXP3_PARALLEL_EFFICIENCY.md`): torch eager on an H200 was slower than one
  CPU core (LTA N = 512: 0.64x ABF, 0.52x FR), launch-bound.
* The external study plan `NSCC_A100_ABF_FR_Experiment_Plan.md` (a proposal; this file is the binding design).
* Stage I output of this study's A100 engine (`results/nscc_a100/validation/`) and the launch-tuning sweeps
  (`results/nscc_a100/bench/tune_*.jsonl`). These contain no free-energy accuracy and no timing of a production
  configuration.

## 1. Question and claims

Primary claim (C4 of the plan): after each method picks its best deployment, any N of the ladder on any eligible
backend (one EPYC 7713 core, the walker-parallel CPU engine on up to 16 cores, one A100), is the best ABF + FR
deployment faster in measured wall-clock time to the frozen accuracy than the best ABF deployment, on fresh
confirmation seeds? Secondary: the A100/CPU throughput crossover per N (C2), and FR / ABF at fixed N on the A100
(C3). A GPU speed-up at fixed N is not evidence for C4.

## 2. Science (frozen, programmatic)

Systems, budgets B, ladders, thresholds, save grids, benchmark windows, engines' algorithms and the e_F scorer are
those of the hybrid short-T study, read through `hst_common.system_spec` (cross-checked against
`configs/equal_budget_v2/*_production.json`):

| system | h | B | ladder | e_F strict / **mid** / loose |
|---|---|---|---|---|
| entropic gateway | 2.5e-5 | 1 032 192 000 | 2048 ... 1 | 0.002 / **0.0055** / 0.02 |
| ethane / LTA 300 K | 2e-4 | 262 656 000 | 1024 ... 1 | 0.03 / **0.094** / 0.5 kJ/mol |
| ethane / LTA 150 K | 2e-4 | 118 272 000 | 1024 ... 1 | 0.03 / **0.13** / 0.5 kJ/mol |

n_steps(N) = B / N; ABF at every N, FR at N >= 2. No parameter of the potentials, of ABF or of FR is changed.

## 3. The A100 engine (backend G1)

`src/gpu/` (design in the headers of `gw_cuda.cu` and `lta_cuda.cu`):
* one persistent cooperative kernel per host segment (segments end at saves); all steps, FR included, run on
  the device; the host stamps saves after the device-to-host copy;
* counter-mode Philox noise with the CPU engine's keys and counter map; the CPU engine's IC and FR uniform streams
  (numpy PCG64 reproduced in C++, bitwise);
* per-step histogram deposits rounded to a fixed-point grid (gateway 2^-34, LTA 2^-43) and summed exactly with
  integer atomics. This makes the reduction deterministic and launch-configuration invariant;
* FR on the device, computed redundantly by every block, with the validated selection logic;
* float64; no FMA contraction.

Deviations from the CPU engine are ulp-level: CUDA libm, force-sum order (LTA), the fixed-point deposit grid
(<= 2.9e-11 gateway, <= 5.7e-14 LTA per deposit), and tree sums in the LTA FR means.

## 4. Stage I gates and the law check

RNG1, RNG2, FORCE, FR, TRAJ, INV, DET and CTR are defined in `campaign.json` `stage1_gates`. They ran repeatedly
during development. The formal record is their run on the frozen code at the start of the benchmark job
(`results/nscc_a100/validation/<system>_validation.json`). A failing gate removes G1 for that system, whatever its
speed.

Engineering changes made during development, all before this freeze, all timing-only, all with the gates re-run:
* gateway: 32-bit limb atomics instead of 64-bit shared atomics;
* gateway: helper warps computing the noise;
* gateway: a touched-bin list;
* LTA: a block barrier after the move (a correctness fix, above);
* LTA: a 128-register launch bound (no spills);
* LTA: a dedicated noise warp, three lanes per walker;
* LTA: an unrolled force reduction;
* LTA: framework O atoms in spatial order.

The TRAJ gate did its job during development. Two configurations of the LTA engine with G > 32 threads per walker
decoupled from the CPU within 50-100 steps, while every G <= 32 configuration stayed coupled. The cause was a missing
block barrier after the leader thread's position update: a group spanning several warps read the pre-move
positions. The barrier was added (`lta_cuda.cu`). With it, every TRAJ run of both systems is pathwise coupled to the
CPU engine over the whole run, including FR events. The gate is the same for both systems: identical bin counts
and FR opportunity counts over the first quarter of the run, and accumulators within 1e-6 relative. The tolerance was
first 1e-9. At 300 K, ulp-level force-sum differences grow in the coordinates that do not enter the CV. The bins and
FR events stay identical, but the accumulators then differed by up to 7.8e-7. The tolerance was set to 1e-6 after
that observation. It is still five orders of magnitude below the statistical noise of an estimate.

**V4 law check (before production).** Full-budget runs, GPU vs CPU P_t, on the same validation seeds (16 per
arm): gateway N = 64 and 512, LTA N = 64 and 256, both methods. Two quantities are reported:
* the per-seed paired differences;
* the ratio of medians of final e_F, Ibar_F and u_mid, with a paired 10 000-resample bootstrap 95 % CI.

Pass iff every CI contains 1 or lies inside [0.8, 1.25].

## 5. Benchmark, launch tuning and routing

* **Launch tuning (done before the freeze; timing only).** For every N and candidate launch, ABF c_ss on the 0.25-scaled
  benchmark window. The fastest launch per N is frozen in `configs/nscc_a100/gpu_launch.json`.
* **Benchmark.** Hybrid short-T windows, 3 repeats with the bench seeds, ABF and FR, every N. Backends: G1 on the
  A100, and S1 and P_t (t <= min(16, N)) on a whole EPYC 7713 node with exclusive CCDs.
* **Routing (common backend).** The hybrid short-T rule chooses the CPU route. G1 replaces it iff:
  * median c_ss(G1) <= median c_ss(CPU route) / 1.05, and
  * every G1 repeat is faster than every CPU-route repeat.

  The rule uses the ABF benchmark. ABF and FR at that N use the routed backend. The routing is frozen (with the
  engines' source hashes) before production.

## 6. Production, selection, confirmation

* **Production (exploration):**
  * every N of the ladder on its routed backend, 24 fresh seeds per arm (`prod_seeds`);
  * seed-major order, arms within a seed permuted by `default_rng(20261026 + seed)`;
  * every run is a fresh process (W_e2e includes imports, JIT / cache loads and, for G1, CUDA context creation).
* **Exploratory best allocation:** `eqb_metrics.best_allocation` on W_mid and W_mid_e2e.
* **Selection (frozen before confirmation):**
  * best ABF arm and best FR arm = argmin of the median W_mid (censored = inf);
  * the selected arm keeps its N and its routed backend.
* **Confirmation:**
  * fresh `confirm_seeds`: 128 per arm (gateway, LTA 150 K), 64 (LTA 300 K);
  * R_W = median W_mid(best FR) / median W_mid(best ABF), and R_e2e the same on W_mid_e2e;
  * unpaired 10 000-resample bootstrap, 95 % CIs.
* **Decision:**
  * 'FR faster' iff both CIs lie below 1; 'ABF faster' iff both lie above 1; otherwise 'unresolved';
  * Holm across gateway and LTA 150 K (primary); LTA 300 K is secondary;
  * censored runs count as inf, never imputed.

## 7. Secondary outputs

* Throughput crossover c_ss per backend and N (figure).
* GPU phase split (deposit, barrier, move + force, FR).
* Fixed-N FR / ABF ratios on G1.
* W_loose and W_strict.
* GPU card-seconds and CPU core-seconds to accuracy.
* The historical EPYC 9554 numbers of the hybrid short-T study, as context only (another machine; never paired
  with this study's runs).

## 8. Stopping rules

* A failing Stage I or V4 gate removes G1 for that system.
* Caps: 40 GPU card-hours and 1 500 CPU core-hours; a run is aborted after 2 h.
* Missing runs are listed.
* Nothing is stopped, extended or re-selected because of which method is winning.

## Amendments

### A1 (2026-10-26, after the benchmark and the routing freeze, before any production run)

**Observation.** On the A100 the FR path was expensive (benchmark tables, FR vs ABF rows):
* LTA FR steps cost 1.6-1.9x an ABF step, against about 1.1x on the CPU;
* gateway FR steps cost up to 1.45x (N = 2048).

The cause is the per-opportunity smoothing. LTA read the 180 x 180 kernel matrix from L2 one dependent load at a
time. The gateway scanned all 181 source nodes per grid node. A slow FR path biases the primary FR-vs-ABF
comparison against FR, so it is fixed before any production data exists.

**Change (timing only).**
* LTA: the occupied bins are compacted (ascending), and the matrix entries are fetched 8 at a time ahead of the
  in-order accumulation.
* Gateway: the source loop runs over the window [i - pad, i + pad], which contains every contribution, in the same
  ascending order.
* Both engines compute the same terms in the same order.

**Checks.**
* Re-runs of law-check jobs on the amended code must reproduce the frozen-code records' science sha256 bitwise
  (`scripts/nscc_a100/check_a1.py`). If they do, the V4 law check carries over.
* The G1 FR benchmark rows are re-measured on the amended code (3 repeats). The frozen-code rows are kept in
  `results/nscc_a100/bench_preA1/`.
* The ABF rows and the routing (ABF-based) are unchanged. The deployment-optimised secondary routing is recomputed.
* Freeze manifest stage `amendment1`.
