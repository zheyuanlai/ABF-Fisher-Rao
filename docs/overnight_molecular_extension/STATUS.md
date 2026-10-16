# Overnight molecular extension — STATUS

*Started 2026-10-16 17:12 UTC. Two adaptive arms only: histogram ABF and histogram ABF + uniform marginal FR. No sham,
oracle, OPES or any other algorithmic arm.*

## Current stage (2026-10-16 ~20:45 UTC)

**All three cells are complete and every one is ABF-sufficient, so no ABF vs ABF + FR comparison was licensed.** An
independent read-only review of the overnight work is the last step. No experiment is running.

## Results (two adaptive arms only; details in `RESULTS.md`)

| cell | barrier (exact MC) | entropic share | ABF-only classification | FR run |
|---|---|---|---|---|
| ethane / ERI 600 K | 6.67 kBT | 84 % | ABF-sufficient (K 64/256/1024) | not licensed |
| ethane / ITE 300 K | 15.9 kBT | 48 % | ABF-sufficient (K 64/256/1024) | not licensed |
| ethane / ITE 150 K | 24.0 kBT | 33 % | ABF-sufficient (K 64/256/1024) | not licensed |

**Pattern.**
* ABF establishes the window population at its bias-aware target within 3 / 5 / 11 t.u. That is 5.5–10 % of the
  horizon, and the same at every K.
* After that the window carries ≤ 7 % of the final error.
* **Open lead (untested hypothesis):** 34–57 % of ABF's integrated error accrues in that short early phase. A new,
  preregistered test of FR during the bias build-up is the natural next experiment (`HANDOFF.md`).

**Summary figure:** `figures/overnight/summary/fig_overnight_summary.png`.

## Validation gates

| cell | Gate 0 | Gate 1 | Gate 2 (production dt) | Gate 3 | Gate 4 |
|---|---|---|---|---|---|
| ERI 600 K | PASS | PASS | EM 2e-4 | 180 bins, T 40 | PASS |
| ITE 300 K | PASS | PASS | EM 1e-4 (2e-4 failed) | 180 bins, T 50 | PASS |
| ITE 150 K | PASS (shared) | PASS | EM 1e-4 (2e-4 failed the second clause) | 360 bins, T 200 | PASS |

## Candidates

* A (Combariza 2011), B, C and E: not reproducible.
* D: feasible with caveats; deprioritised (expected ABF-sufficient).
* See `CANDIDATES.md`.

## Active jobs

None. Nothing is running on CPU or GPU.

## Next action (needs the user)

1. Decide whether to preregister an "early-phase FR" experiment. Recommended: ITE 300 K, K 1024, FR from t = 0 or
   from the end of the ramp, endpoint I_F; two arms only.
2. Optionally, the ERI loading study on GPU 3 (a new torch engine).

## Resource use

About 117 of 200 CPU core-h and 0 of 24 GPU-h. See `COMPUTE_LEDGER.md`.
