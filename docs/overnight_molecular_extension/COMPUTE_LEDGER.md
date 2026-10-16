# Compute ledger — overnight molecular extension

The ceiling is **aggregate** over the ERI campaign and this extension: ≤ 200 CPU core-h and ≤ 24 GPU device-h in
total (the 2026-10-16 instruction preserves the existing allocation).
* GPUs 0/1 belong to another user and are never touched. GPU 3 is the project device; it is unused so far.
* Core-h for threaded runs = wall × threads, which counts idle threads too (an over-estimate).

| date (UTC) | item | CPU core-h | GPU h | cumulative CPU |
|---|---|---|---|---|
| 2026-10-16 ≤ 17:10 | ERI campaign up to the dilute screen | 18.9 | 0 | 18.9 |
| 17:10–17:30 | ERI classification, audit checks, tests | 0.3 | 0 | 19.2 |
| 17:45–17:55 | ITE build, percolation, pilots, Gate 0 | 0.3 | 0 | 19.5 |
| 17:56–18:12 | ITE Gate 1 MC (2.24) + Gate 2 EM 2e-4 / 1e-4 / LM 2e-4 / EM 5e-5 (0.70 / 1.33 / 0.70 / 2.63) | 7.60 | 0 | 27.1 |
| 18:03–18:31 | ITE Gate 1/2 analyses (5 processes, 2000 bootstrap resamples) | ~1.6 | 0 | 28.7 |
| 18:06–18:32 | ITE Gate 3a horizon (6 runs, 400 t.u.) | 10.68 | 0 | 39.4 |
| 18:32–18:37 | ITE Gate 3b ramp / pre-eq (24 runs) | 6.00 | 0 | 45.4 |
| 18:37–18:42 | ITE Gate 3c bins (18 runs) | 5.78 | 0 | 51.2 |
| 18:37–18:42 | ITE ABF-only screen (24 runs) | 5.82 | 0 | 57.0 |

**Remaining authorisation at 18:45:** about 143 CPU core-h and 24.0 GPU-h.
