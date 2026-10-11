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
| 07:50 | Exp A tooling workflow complete (implement → review → fix; 45 + 28 + 9 + 62 tests pass) | wf_49b9d228 |
| 07:52 | **Amendment 1 + Exp A tooling committed and pushed: 014c4be0** (before any gate/smoke/production run) | `git log` |
| 07:52 | Checks 1/7 PASS (39 tests), 8c harness inertness PASS (6 tests), 2 references PASS (bitwise across λ and vs equal-budget reference) | `results/final_campaign/expA/checks/` |
| 07:53 | Launched (detached): G-A1 gate chains (121 jobs, 32 workers, cores 8–39), check 4b (core 0), check 8 (4 workers, 48–51), smoke (32 runs, 8 workers, 52–59) | `results/final_campaign/expA/logs/` |
| ≈ 07:56 | Check 4b PASS: 0/30 Holm rejections; resolvable-inflation cases (λ 1, 16 at h = 0.1/λ) match the EM AR(1) law | `check4b_frozen_x.json` |
| ≈ 08:03 | Check 8 PASS: 4/4 stored production jobs re-run bitwise equal (48 arrays each); 8c PASS | `check8_old_bitwise.json` |
| ≈ 08:00 | Smoke: 32/32 runs complete, 0 failed | `logs/smoke.log` |
| 08:05 | Check 9 PASS (32/32 files) incl. check 6 PASS on the frozen first-death rule (16/16 pairs, 16–25 compared saves each) | `check9_smoke.json`, `check6_pairing_smoke.json` |
| 08:06 | V5 ABF-bias smoke PASS for λ 4, 16, 128 (finite; final max|Γ| 7.89–7.90 vs 15.78) | `validation/abf_smoke.json` |
| 08:06 | G-A1 chains 32/121 done; every λ = 1 replay so far bitwise equal to its frozen record | `logs/gate_run.log` |
| 08:16–08:26 | Exp B backend workflow complete (implement → review → fix; gateway 36, LTA 37, harness+stats 64 tests pass); Amendment-2 code defaults applied (replicated_dissem variant, tau_bgrid competitor completion, own-core contention scope) and tests re-run (64 + 36 pass) | wf_63252a8d |
| 08:30 | User: another session removed large files (32 GB free); **user instruction: no GPU, CPU only** (campaign already 0 GPU-h; verified no GPU process of this account) | `nvidia-smi`, `df` |
| 08:23 | G-A1 chains 121/121 complete; λ = 1 replay 16/16 bitwise | `logs/gate_run.log` |
| 08:31 | **G-A1 SELECTION PASS, h_A = 2.5e-5** (λ 4, 16, 128: V1, V2, V3, V3b, V5 PASS, V4 holds). MC1 flank 1/e ×1.14 / ×1.65 / ×6.2; MC2 transport ×2.0 / ×3.7 / ×5.3 vs λ = 1 | `validation/summary.{json,md}` |
| 08:31 | Amendment 2 committed with gate records and Exp B code: **ef8ebc20** | `git log` |
| 08:32 | Production gate SATISFIED (`expA_gate.py`) | — |
| 08:33 | **Exp A production launched**: 512 runs, 48 workers on idle physical cores, seed-major | `logs/production.log` |
| 08:34 | Exp B formal V-B1/V-B2/V-B3/V-B5 correctness gates launched on cores 64–79 (sequential driver `expB_run_vb_gates.sh`) | `results/final_campaign/expB/validation/` |
| 08:47 | **Exp B correctness gates PASS** (766 s on 16 cores): V-B1 replay bitwise vs validated engines (gateway N 16/64/512 both arms, LTA N 2/16/64/512, threads 1/4/16); V-B2 counter-mode thread invariance (threads 1–16); V-B5 pairing (frozen first-death rule and event rule); V-B3 RNG (Philox = numpy oracle, counter-map injectivity, 10⁸-normal battery, kernel noise = Philox module) | `results/final_campaign/expB/validation/` |
| 08:47 | Exp B gateway V-B4 equivalence runs launched (96 counter-mode runs: ABF 16, ABF 512, FR 512 × seeds 8500–8531, 16 single-core workers) | `results/final_campaign/expB/validation/vb4_runs.log` |
| 09:04 | **Gateway V-B4 frozen verdict FAIL**: ABF 16 median-Ī_F ratio 1.30 [1.07, 1.46] (point outside the extension window → no extension); G difference at N 512 +0.056 [−0.034, +0.180] (point outside ±0.05 → FAIL); ABF 512 PASS 0.94 [0.87, 1.08]; FR 512 EXTEND 1.06 [0.91, 1.21]. Under §3 stopping rule (ii) confirmatory Exp B for the gateway stops | `results/final_campaign/expB/equiv/gateway/vb4.json` |
| 09:04 | LTA 150 K V-B4 runs launched (144 jobs, 16 single-core workers, cores 64–79) | `vb4_runs.log` |
| 09:24 | **Exp A production complete**: 512/512 runs, 0 failed, 0.87 h wall on 48 cores (40.5 used core-h) | `expA/logs/production.log` |
| 09:25 | Check 6 on production: 256/256 pairs PASS the frozen first-death rule (4 via replay) | `expA/checks/check6_pairing_production.json` |
| 09:25–09:28 | Exp A analysis (per-cell, cross-cell, mechanism, outcome, tables, figures; 0.15 core-h): **frozen outcome A at both N for E1 (λ 16 vs 1) and E2 (λ 128 vs 1)** | `expA/synthesis/outcome.md` |
| 09:29 | **Exp A production + analysis committed and pushed: 1ec25ca5** | `git log` |
| 09:25 | Post-hoc V-B4 diagnostic started (not preregistered; seeds outside every preregistered block): (A) validated serial engine on the counter seeds 8500–8531, (B) counter mode on fresh 8740–8771, (C) parallel engine in replay mode at full length on 8500–8503. First attempt as a 32-process pool: the parallel engine's libgomp bound every unpinned worker's master thread to CPU 0 (≈ 4 % CPU each); killed and relaunched at 09:32 as 32 separately pinned single-core workers | `diagnostics/vb4_abf16/` |
| 09:39 | Interim diagnostic: (C) replay identity bitwise at full length 4/4; (A) validated engine on 8500–8531 median 0.00111 vs counter 0.00113 on the same labels | `diagnostic_summary.json` |
| 09:39 | Development probe (not the benchmark), N 512 FR, 8 threads on cores 32–39: replay mode 33.2 µs/step (36.4 at 1 thread: the serial MT noise fill dominates), counter mode 5.34 µs/step — replay mode cannot provide the parallel speed-up | console |
| 09:40 | Folded Exp A validation compute (G-A1 15.20, V5 smoke 0.10, check 4b 0.017, check 6 ≈ 0.33 core-h) into the compute ledger | `compute_ledger.csv` |
