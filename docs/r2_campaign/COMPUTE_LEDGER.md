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
