# Experiment III: wall-clock time-to-accuracy at fixed N = 512

*2026-10-10. Preregistration: `SCIENTIFIC_PLAN.md` §8 (endpoints) and Amendment 2 (d3595341, backends, seeds and guards,
written before any run).*

**Harness and data**
* `scripts/mechanism/parallel_benchmark.py` runs the jobs; `scripts/mechanism/analyze_parallel_benchmark.py` analyses
  them. Both were built, reviewed (12 findings: 11 fixed, 1 fixed in the reporting) and tested.
* Runs: `results/mechanism/parallel_benchmark/<system>/<backend>/`.
* Analysis: `results/mechanism/parallel_benchmark/analysis/{summary.json,tables.md}`.
* Figures: `figures/mechanism/parallel_benchmark/parallel_benchmark_{T_time_to_accuracy, X_error_vs_wall,
  Q_equivalence, C_cost}.{pdf,png}`. The **T** figure is the campaign's synthesis figure 7.

## 1. What was measured

**Comparison.** ABF vs ABF + FR at **fixed N = 512** on **identical hardware**, for the gateway (h 2.5e-5), LTA 300 K
and LTA 150 K (h 2e-4), with the frozen production algorithms.

**Endpoint.** Wall-clock seconds to the **persistent** mid e_F threshold (gateway 0.0055, LTA 300 K 0.094, LTA 150 K
0.13).
* Timestamps are taken at every save, with a CUDA synchronisation before each for the GPU.
* Warm-up and JIT are measured separately: 0.18–0.35 s per run, negligible.
* The headline ratio W1 uses only pairs in which neither run's CPU core shared its SMT sibling with other load.

**Backends**

| backend | systems | status |
|---|---|---|
| `cpu_numba_1core`: the validated production engines, one pinned core per run | all 3 | **complete**: 8 seeds × 2 arms each. Every run is bitwise equal to its production file. |
| `gpu_torch`: the existing torch engines, one run at a time on one H200 (GPU 3) | LTA 300 K | **complete**: 8 × 2. Equivalence vs the numba production: **CONSISTENT** (minimum detectable shift 17–32 % of the median). |
| `gpu_torch` | LTA 150 K | **INCOMPLETE**: 5 complete pairs + 1 unpaired FR run. The last 5 jobs were refused by the harness guard because another user's process started on GPU 3 (18:5x UTC), and no GPU was idle. Equivalence: no verdict (< 8 pairs). Resumable: completed runs are skipped. |
| `gpu_torch` | gateway | **NOT TESTED**: projected 16.9 GPU-h, over the 8 GPU-h ceiling. The torch gateway engine is about 9× slower per step than one CPU core at N = 512. |
| walker-parallel CPU | all | **NOT TESTED**: no such build exists, and the plan forbids writing one into this comparison. |

## 2. Results

FR/ABF ratios are medians of paired per-seed ratios with seed-bootstrap 95 % CIs. Below 1 means FR is faster.

| system | backend | simulation time to accuracy, FR/ABF | wall-clock to accuracy, FR/ABF (**W1**) | FR wins | FR overhead per step | ABF wall to accuracy (median s) | FR wall (s) |
|---|---|---|---|---|---|---|---|
| gateway | 1 core | 0.353 [0.298, 0.578] | **0.350 [0.299, 0.559]** | 8/8 (p 0.008) | 0.5 % [0.1, 1.2] | 60 | 25 |
| LTA 300 K | 1 core | 0.801 [0.652, 1.10] | **0.755 [0.662, 1.11]**, n = 7 | 5/7 (p 0.45) | 1.6 % | 122 | 107 |
| LTA 300 K | GPU | 0.857 [0.469, 1.57] | **1.04 [0.558, 1.97]** | 4/8 (p 1) | **24.9 % [22.9, 26.0]** | 198 | 203 |
| LTA 150 K | 1 core | 0.341 [0.318, 0.451] | **0.361 [0.327, 0.518]**, n = 6 | 6/6 (p 0.03) | 1.5 % | 152 | 55 |
| LTA 150 K | GPU (5 pairs) | 0.376 [0.246, 0.583] | **0.449 [0.283, 0.713]** | 5/5 (p 0.06) | **25.5 % [22.9, 28.1]** | 212 | 83 |

**Hardware facts**
* **Effective parallelism**, one CPU core's wall over one GPU's wall at equal steps: 0.64 (ABF) and 0.52 (FR) for both
  LTA systems. **The GPU is slower than one CPU core at N = 512.**
* GPU memory: 654 MB (ABF) and 752 MB (FR) on the device, plus about 1 GB of host memory.
* One CPU core uses 166–263 MB.

**Resources**
* CPU: 3.5 core-h (12 500 core-seconds timed).
* GPU: 3.5 GPU-h allocated, 7 507 s (LTA 300 K) + 5 226 s (LTA 150 K), plus about 0.2 GPU-h of smoke runs.

## 3. Interpretation

* **At fixed N on one core per run, FR's simulation-time gain becomes a wall-clock gain almost one for one.** Its
  overhead is 0.5–1.6 %, giving W1 = 0.35 (gateway) and 0.36 (LTA 150 K). Where its simulation-time gain is weak
  (LTA 300 K, 0.80), the wall-clock gain is weak and not significant (0.76).
* **On the GPU tested, FR's overhead rises to about 25 % per step.** The force evaluation is cheap and parallel on the
  device, while the KDE, score and resampling are a serial-ish global step. This erodes the gain:
  * LTA 150 K: 0.38 in simulation time → 0.45 wall (5/5);
  * LTA 300 K: 0.86 → 1.04, no advantage.
* **The GPU does not provide walker-level speed-up at N = 512 here.** It is about 1.6–1.9× slower than one CPU core.
  So this experiment cannot show that FR's large-N, short-T_N advantage becomes a wall-clock advantage over a small-N
  ABF allocation on parallel hardware: the hardware tested does not make large N cheap.
* The cross-N question remains as in the equal-budget campaign. At equal force evaluations, few long ABF replicas match
  or beat FR's best. On one core, equal force evaluations means equal wall-clock time.

**Answer to W1.** Yes at fixed N on CPU: about 2.8× for the gateway and LTA 150 K, and not significant for
LTA 300 K. Reduced or absent on the GPU tested, where FR's overhead is 25 %. A parallel speed-up over the best small-N
ABF allocation is **not demonstrated**, because no backend tested makes large N cheaper per walker-step.
