# NSCC A100 study: how to run, resume and audit

Read first: `PREREGISTRATION.md` (the design), then `RESULTS.md`. This file covers the mechanics.

## Code

| path | role |
|---|---|
| `src/gpu/gw_cuda.cu`, `src/gpu/gateway_gpu.py` | gateway A100 engine; `run_arm_gpu` returns the CPU engines' result contract |
| `src/gpu/lta_cuda.cu`, `src/gpu/lta_gpu.py` | ethane / LTA A100 engine, same contract |
| `src/gpu/gpu_common.cuh` | Philox4x64-10, Box-Muller, numpy PCG64, block scans, fixed-point limb deposits |
| `src/gpu/gpu_build.py` | nvcc build (tagged by source / flag / compiler hashes) into `$GPU_BUILD_DIR` |
| `scripts/nscc_a100/nscc_common.py` | jobs, paths, the run child (backends S1, P_t, G1), endpoints via `hst_common` |
| `scripts/nscc_a100/child.py`, `runner.py` | one job per fresh process; GPU jobs one at a time, CPU jobs on exclusive CCDs |
| `scripts/nscc_a100/validate_gpu.py`, `validate_gpu_lta.py` | Stage I gates (RNG, forces, FR selection, pathwise coupling, invariance, determinism) |
| `scripts/nscc_a100/bench_gpu.py` | launch-tuning sweeps (scaled windows) |
| `scripts/nscc_a100/plan.py` | job lists: `launch` (freeze the tuned launch table), `bench`, `law`, `prod`, `confirm` |
| `scripts/nscc_a100/route.py` | benchmark summary and the frozen routing |
| `scripts/nscc_a100/analyze.py` | law check, exploration and best allocation, selection freeze, confirmation, Holm |
| `scripts/nscc_a100/freeze.py` | sha256 manifests of the design and run-path code per stage |
| `scripts/nscc_a100/report.py`, `plot.py` | generated tables and figures |
| `scripts/nscc_a100/pbs/{gpu,gpu2,cpu}_task.pbs`, `tasks/*.sh` | PBS wrappers (one A100; two A100s; one EPYC 7713 socket) and the task scripts |
| `scripts/nscc_a100/check_a1.py` | Amendment A1: re-runs law-check jobs on the amended engines, compares the science sha256 |

## Environment on NSCC ASPIRE2A

* `source scripts/nscc_a100/env.sh`. Every cache lives on scratch, because the home quota was full. The script
  sets:
  * the CUDA 12.8 toolkit;
  * the uv venv `.venv`: Python 3.14, numpy 2.4.6, numba 0.68.0, scipy 1.17.1, torch 2.12 CPU (the gateway scorer);
  * `NUMBA_CACHE_DIR`, `GW_PARALLEL_BUILD_DIR` and `GPU_BUILD_DIR` on scratch.
* Build the CUDA libraries inside a GPU job (`tasks/build_gpu.sh`). The login nodes throttle per-user CPU, and
  ptxas takes over 10 minutes there.
* `cache/lta/framework.npz` is gitignored: `python scripts/build_lta_framework.py --super 2`.

## The order the study actually ran (task scripts in `scripts/nscc_a100/tasks/`)

1. `perf_and_tune.sh`, `lta_perf_tune.sh`: Stage I during development and the launch sweeps; then
   `plan.py launch` and `freeze.py write --stage design` (login node).
2. `gpu_validate_bench.sh` (A100) and `cpu_bench.sh` (CPU socket):
   * the Stage I record on the frozen code;
   * the benchmark;
   * both halves of the V4 law check.

   Then `route.py freeze`, `analyze.py law`, and `freeze.py write --stage routing`.
3. Amendment A1. Archive `bench/*/fr*_G1` into `bench_preA1/`, then `freeze.py write --stage amendment1`.
4. `prod_g2.sh` (2 A100s: the A1 bitwise check, production, the re-measured G1 FR benchmark rows) and
   `prod_cpu_a.sh` / `prod_cpu_b.sh` (two CPU sockets). Then `analyze.py select` and
   `freeze.py write --stage selection`.
5. `confirm_g2.sh` + `confirm_cpu.sh`. Then `analyze.py confirm`, `route.py summarize`, `report.py`, `plot.py`.

Development-only helpers: `cpu_smoke.sh`, `diag_lta_perf.sh` (`diag_lta.py`, `diag_lta_perf.py`: GPU-CPU coupling
per launch, LTA step split by phase).

## Running a stage

```bash
qsub -P personal-e1297740 -v TASK=scripts/nscc_a100/tasks/<task>.sh scripts/nscc_a100/pbs/gpu_task.pbs
qsub -P personal-e1297740 -v TASK=scripts/nscc_a100/tasks/<task>.sh scripts/nscc_a100/pbs/cpu_task.pbs
```

* The runner skips every job whose record is complete with a matching signature, so a resubmitted task resumes.
* Logs: `/scratch/users/nus/e1297740/logs/<task>_<jobid>.log`. Failed children: `results/nscc_a100/logs/fail_*`.
* Records: `results/nscc_a100/<kind>/<system>/<arm>_<backend>/s<seed>[_r<r>].json` (+ `.npz` arrays).

## PBS facts learned

* GPU jobs: `-l select=1:ncpus=16:ngpus=1:mem=110gb`, walltime > 2 h (queue g1). Each GPU queue (g1, g2, g3)
  allows one running job per user. A 2-GPU job (g2) runs alongside a 1-GPU job, with one runner per GPU. The CPU
  queue q2 allows 64 running jobs per user.
* Whole CPU nodes are rarely free (node_pool). One socket (`ncpus=64`) starts at once in q2. A walltime <= 2 h
  routes to qdev, which is often full.
