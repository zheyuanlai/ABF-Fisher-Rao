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
| 05:59 | **Preregistration frozen and pushed: c5a2264f** (prereg, ledger, review resolution, configs, seed check, scouting record) | `git log` |
| 06:03–06:05 | Exp A tooling and Exp B backend implementation workflows launched (no experiment runs) | workflows wf_49b9d228, wf_63252a8d |
| 06:49 | **Disk emergency:** free space on / reached 2.08 GB (other users plus the user's own concurrent `levy-GM-theory` job writing 0.5–1 GB checkpoints). This campaign's footprint was ≈ 45 MB. Purged the user's pip HTTP download cache (`pip cache purge`, 12.2 GB, 2452 files, regenerable; no locally built wheels) → 13.7 GB free. User notified | `df`, `pip cache info` |
| 07:01–07:13 | Disk fell again (13.7 → 6.4 GB, ≈ 5–19 GB/h) from accounts not readable here; the user's own `levy-GM-theory` job writes 0.5–1 GB checkpoints every ≈ 40 min | `df`, `find`, `du` snapshots |
| 07:24 | Both build workflows resumed after a subagent session-limit interruption (fix stages and the Exp B harness review had failed; completed implement/review stages replayed from cache) | workflows wf_49b9d228, wf_63252a8d |
| 07:26 | **Disk at 1.07 GB.** With the user's approval ("delete old Claude scratch"), removed past-session Claude scratch data: ABF sessions 01755893 and c27bfa13 (whole, no open handles) and the `scratchpad/` subfolders of b25e8a63 and f927818a (their `tasks/` dirs are held open by still-running idle sessions and were kept). The levy-GM-theory session 4ee891d2 was **not** touched: it is held open by a live `claude --continue` process (pid 1262331) writing task output. Freed 3.6 GB → 4.6 GB free | `lsof`, `df` |
