# Compute ledger — adaptive-FR campaign

* **Authorisation.** 100 CPU core-h for this campaign, incremental on the previous campaigns' ≈ 198.4. The aggregate
  ceiling is ≈ 298.4.
* **GPU.** None. GPUs 0/1 belong to `yesom` and GPUs 2/3 to `hyt35`; all were busy at 06:19 UTC on 2026-10-17.
* **Accounting.** Core-h = wall × threads. Every simulation is single-threaded.

| date (UTC) | item | CPU core-h | campaign total | aggregate |
|---|---|---|---|---|
| 10-17 06:19–06:23 | inspection, 14 engine tests, pairing pilot (2 × 15 000 steps at K256) | 0.03 | 0.03 | 198.43 |
| 06:22–06:35 | **Experiment A** FR arm (16 × K256, T 200; 775 s wall each) | 3.415 | 3.45 | 201.8 |
| 06:26–06:32 | Exp B development runs (6 continuous-FR monitor runs, ITE only, seeds 1200000/1/100/101/200/201) | ~0.30 | 3.75 | 202.1 |
| 06:40–07:10 | Exp B: 3 weak-rate dev runs, engine/LTA tests (×3), dev engine checks, replays, review agent checks | ~0.4 | 4.15 | 202.6 |
| 07:10–07:15 | **C3** LTA 150 K (32 runs) | 2.391 | 6.54 | 205.0 |
| 07:10–07:15 | **C4** LTA 300 K (32 runs) | 2.383 | 8.92 | 207.4 |
| 07:10–07:20 | **C5** ERI 600 K K1024 (32 runs) | 4.801 | 13.72 | 212.2 |
