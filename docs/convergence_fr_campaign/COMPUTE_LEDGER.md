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
