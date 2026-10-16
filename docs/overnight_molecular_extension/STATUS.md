# Overnight molecular extension — STATUS

*Started 2026-10-16 17:12 UTC. Two adaptive arms only: histogram ABF and histogram ABF + uniform marginal FR. No sham,
oracle, OPES or any other algorithmic arm.*

## Current stage (2026-10-16 18:50 UTC)

**Cell A′-150 (ethane / rigid Si-ITE / Dubbeldam 2004, now at 150 K): Gate 1–2 reference running.** The cell was
preregistered as Amendment 3 before any 150 K computation. The motivation is the repo's ethane/LTA, which uses the same
O–CH3 pair and was establishment-limited at 150 K.

## Results so far (two adaptive arms only; details in `RESULTS.md`)

| cell | barrier (exact MC) | entropic share | ABF-only classification | FR run |
|---|---|---|---|---|
| ethane / ERI 600 K | 6.67 kBT | 84 % | ABF-sufficient (K 64/256/1024) | not licensed |
| ethane / ITE 300 K | 15.9 kBT | 48 % | ABF-sufficient (K 64/256/1024) | not licensed |
| ethane / ITE 150 K | pilot ≈ 23.9 kBT | – | pending | – |

## Validation gates

| cell | Gate 0 | Gate 1 | Gate 2 (dt) | Gate 3 | Gate 4 |
|---|---|---|---|---|---|
| ERI 600 K | PASS | PASS | EM 2e-4 | frozen | PASS (tests) |
| ITE 300 K | PASS | PASS | EM 1e-4 (2e-4 failed) | T 50, ramp 1, pre-eq 5, 180 bins | PASS (tests) |
| ITE 150 K | same model | running | running | – | – |

## Candidates

A (Combariza 2011), B, C and E are not reproducible. D is feasible with caveats and deprioritised. See
`CANDIDATES.md`.

## Active jobs

* `run_reference.py` chain, ITE 150 K: MC (1024 jobs) → EM 2e-4 → EM 1e-4. 40 single-thread processes. Log:
  `results/eri_campaign/logs/ite150_gate12.log`.
* No GPU in use.

## Next automatic action

Gate 1/2 analysis at 150 K → Gate 3 (horizon, ramp/pre-eq, bins) → ABF-only screen → frozen classifier. Two-arm
production (calibration, then explore 8 + confirm 16 seeds) runs only if a K is ESTABLISHMENT-LIMITED.

## Resource use

57 of 200 CPU core-h (start of A′-150) and 0 of 24 GPU-h. See `COMPUTE_LEDGER.md`.
