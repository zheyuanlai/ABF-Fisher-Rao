# Compute ledger — convergence-FR campaign

Ceiling (aggregate over the ERI campaign, the overnight molecular extension and this campaign; no newer authorisation
exists): **≤ 200 CPU core-h, ≤ 24 GPU device-h.**

* Core-h for a run = wall × numba threads. Every run here is single-threaded, so the count is close to actual CPU
  time.
* Opening balance: **171.1 CPU core-h, 0 GPU-h.**
  * 170.8 at 2026-10-16 21:45 in `docs/overnight_molecular_extension/COMPUTE_LEDGER.md`.
  * About 0.3 for the later analyses and the push.
  * Reconciled 2026-10-17 02:52 UTC: no job of this project was running (`ps`), and no checkpoint was in flight.
* GPU: none used. GPUs 0/1 have been occupied by user `yesom` for 50 days. GPUs 2/3 have run a job of user `hyt35`
  since about 00:00 UTC on 2026-10-17. **GPU 3 is therefore not available**, and this campaign is CPU only.

| date (UTC) | item | CPU core-h | GPU h | cumulative CPU |
|---|---|---|---|---|
| 10-17 02:52–03:10 | engine tests (12), reproduction pilot (X1 s0, 20 k steps, 1 thread), atlas (2 runs of the analysis, 1 process) | 0.10 | 0 | 171.2 |
| 10-17 03:13–03:17 | ITE common-rate calibration rung 2.0 (6 × K256, T 50) | 0.343 | 0 | 171.5 |
| 03:13–03:22 | ERI calibration rung 2.0 (3 × K256 + 3 × K1024, T 40) | 0.528 | 0 | 172.1 |
| 03:17–03:21 | ITE calibration rung 1.0 (6 × K256, T 50) | 0.339 | 0 | 172.4 |
| 03:20–03:23 | ERI K256 production (32 runs) | 1.168 | 0 | 173.6 |
| 03:13–03:27 | **E1** ITE 300 K early-stopping (32 × K1024, T 50; 812–850 s each) | 7.259 | 0 | 180.8 |
| 03:23–03:30 | matrix M300_256 (32 runs) | 1.853 | 0 | 182.7 |
| 03:23–03:30 | matrix M150_256 (32 runs) | 1.824 | 0 | 184.5 |
| 03:22–03:32 | ERI K1024 production (32 runs, 559–577 s each) | 4.801 | 0 | 189.3 |
| 03:18–03:35 | analyses, figures, audit agent checks (single processes) | ~0.2 | 0 | 189.5 |
