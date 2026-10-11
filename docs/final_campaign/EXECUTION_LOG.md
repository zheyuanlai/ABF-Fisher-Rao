# Final campaign — execution log

All times UTC, 2026-10-11, measured from command output or commit metadata (not estimated). Compute is tracked in
`results/final_campaign/compute_ledger.csv` (used core-hours = threads × wall).

| time | event | evidence |
|---|---|---|
| 04:59 | Session start. HEAD = origin/main = 189c2408. Host: 2 × EPYC 9554 (128 cores), load 13.4, disk 14 GB free, GPUs 0–2 busy | `uptime`, `df`, `nvidia-smi` |
| 05:00–05:16 | Repository audit: seven parallel code-mapping agents (engine, validation harness, pipeline, analysis, LTA engine, Exp III harness, seed inventory) | workflow wf_b94e70d7 |
| 05:05 | Synchronisation micro-benchmarks (scratch): OpenMP barrier 0.36–1.10 µs at 1–16 threads; numba `prange` launch 0.9–3.8 µs at 1–8 threads (63.8 µs at 16 threads on a partly loaded CCD) | scratchpad `bench/` |
| 05:18–05:30 | Evidence ledger, preregistration draft, configs; seed check PASS | `check_seeds.py` |
| 05:25–05:50 | Five-lens adversarial review of the draft + skeptic verification; 61 findings, all critical/major resolved | `PREREG_REVIEW_RESOLUTION.md` |
| 05:45 | Baseline test state recorded at 189c2408: engine, validation-harness, metrics and LTA suites pass; 10 pre-existing failures in old-campaign bookkeeping tests (`test_mech_pipeline` ×7, `test_f12_driver_reuse_policy`, `test_parallel_benchmark` GPU-guard ×2) | `results/final_campaign/baseline_tests.md` |
| 05:52–05:54 | Pre-freeze scouting (flat bias, no ABF/FR): transverse mobility changes gate transport ×2.5 / ×4.8 / ×6.9 and flank relaxation ×1.1 / ×1.6 / ×5.6 at λ = 4 / 16 / 128 vs λ = 1. 4 runs × 70 s | `results/final_campaign/expA/scout/` |
| 05:58 | Host at freeze: load 19.0 / 21.7 / 20.5; disk 6.1 GB free (falling ≈ 8 GB/h from other users' activity; `/data1`, `/data2` not writable); GPU 3 idle, GPUs 0–2 busy | `uptime`, `df`, `nvidia-smi` |
