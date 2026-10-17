# Compute ledger — R2 campaign

* **Authorisation.** Incremental on the adaptive-FR campaign's 100 CPU core-h (≈ 44.1 used before this campaign;
  aggregate before ≈ 242.5 of ≈ 298.4). Two other sessions of the same project were working on the host at the same
  time; their consumption is not in this ledger and is not visible to it.
* **GPU.** None.
* **Accounting.** Core-h = wall × threads; every simulation is single-threaded. Host: 2 × AMD EPYC, 256 logical
  CPUs, load average ≈ 10 from other users at 11:16 UTC.

| date (UTC) | item | CPU core-h | campaign total | adaptive-FR total | aggregate |
|---|---|---|---|---|---|
| 10-17 11:16–11:45 | inspection; R2 implementation; 49 engine/controller tests (×4 runs); offline replay (two passes over ≈ 400 run files, single process); 80/350 K reference files | ≈ 0.3 | 0.3 | ≈ 44.4 | ≈ 242.8 |
| 11:38–11:47 | **wave 1** (held out): R2C3 LTA 150 K (32 runs, 4.364), R2C4 LTA 300 K (4.337), R2H80 LTA 80 K (4.368), R2H350 LTA 350 K (4.413), R2N256 ERI K256 (2.335); 160 single-threaded processes at once; per-run wall 490 s (LTA) / 263 s (ERI K256) against 270 / 118 s in the adaptive-FR campaign: the host load average was 130–220 (our 160 plus other users' jobs) on 128 physical cores, so the wall-clock-based account is inflated by SMT sharing | 19.82 | 20.1 | ≈ 64.2 | ≈ 262.6 |
| 11:47–12:02 | **wave 2** (in-sample): R2C1 ITE 300 K K1024 (32 runs, 7.528), R2C2 ITE 150 K K256 (3.832), R2C5 ERI 600 K K1024 (5.165); 96 processes; per-run wall 847 / 431 / 581 s against 744 / 377 / 486 s in the adaptive-FR campaign (load average 125–160) | 16.53 | 36.6 | ≈ 80.7 | ≈ 279.1 |
| 11:47–12:10 | frozen analyses of the eight cells, verdicts, tables (single process, under a minute each) | ≈ 0.1 | ≈ 36.7 | ≈ 80.8 | ≈ 279.2 |

**Campaign total ≈ 36.7 CPU core-h (estimate was 26.5: the measured per-run wall was 1.1–1.8× the adaptive-FR
campaign's because of host contention, and the account is wall-based); adaptive-FR authorisation ≈ 80.8 of 100 used;
aggregate ≈ 279.2 of ≈ 298.4. No GPU; no external or paid resource.**
