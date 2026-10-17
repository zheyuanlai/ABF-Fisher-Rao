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
| 07:15–07:29 | **C1** ITE 300 K K1024 (32 runs) | 7.242 | 20.96 | 219.4 |
| 07:15–07:22 | **C2** ITE 150 K K256 (32 runs) | 3.516 | 24.48 | 222.9 |
| 07:22–07:26 | **D2** clone-age re-runs (32 FR runs, ERI + ITE K256) | 1.501 | 25.98 | 224.4 |
| 07:32–07:35 | K64 blind calibration (3 rungs × 3 seeds, T 50) | 0.137 | 26.12 | 224.5 |
| 07:32–07:45 | **EA_R1** (16 FR runs, K256 × T200, R1) | 3.321 | 29.44 | 227.8 |
| 07:35–07:48 | **EB64** (32 runs, K64 × T800, R1, rate 0.2) | 6.566 | 36.00 | 234.4 |
| 07:49–07:53 | E.4 wall-clock benchmark run 1 (threads 1–32; measured 0.31 + warm-up) | ~0.5 | 36.50 | 234.9 |
| 07:56 | K16 blind calibration (3 rungs × 3 seeds, T 50) | 0.042 | 36.54 | 234.9 |
| 07:57–08:10 | **EB16** (32 runs, K16 × T3200, R1, rate 1.0) | 6.825 | 43.37 | 241.8 |
| 08:10–08:15 | E.4 wall-clock benchmark run 2 (with K16; measured 0.31 + warm-up) | ~0.5 | 43.87 | 242.3 |
| 08:15–08:20 | analyses, figures, documents (single process, minutes each) | ~0.1 | ≈ 44.0 | ≈ 242.4 |
| 08:21–08:35 | reproduction run, tests, D1 matched-window and D2 per-bin additions, figure 7b, doc audit fixes | ~0.1 | ≈ 44.1 | ≈ 242.5 |

**Campaign total: ≈ 44.1 of 100 authorised CPU core-h; aggregate ≈ 242.5 of ≈ 298.4. No GPU; no external or paid
resource.**

### Amendment E-A3 (frontier bracket below K16; branch `claude/priceless-rubin-9602df`)

| date (UTC) | item | CPU core-h | campaign total | aggregate |
|---|---|---|---|---|
| 10-17 11:21–11:22 | E-A3 timing pilot: 3 concurrent scratch ABF runs of 2 × 10⁶ steps (K8 12.8, K4 7.4, K1 2.7 µs/step; + warm-up) | 0.02 | ≈ 44.1 | ≈ 242.5 |
