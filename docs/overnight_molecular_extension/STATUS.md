# Overnight molecular extension — STATUS

*Started 2026-10-16 17:12 UTC. Two adaptive arms only: histogram ABF and histogram ABF + uniform marginal FR. No sham,
oracle, OPES or any other algorithmic arm.*

## Current stage (2026-10-16 18:00 UTC)

**Candidate A′ — ethane in rigid Si-ITE (ITQ-3), Dubbeldam-2004 UA force field, 300 K, infinite dilution: Gates 1–2
running.**

* A′ is a labelled model of the same physical system, not a reproduction of Combariza et al. 2011. That paper used a
  flexible charged framework and all-atom guests, and both the paper and its parameters are unrecoverable
  (`CANDIDATES.md`).
* Running: exact MC umbrella reference (16 groups × 40 windows), then the timestep ladder EM 2e-4 / 1e-4 / LM 2e-4 /
  EM 5e-5. CPU, 32 single-thread processes. Log: `results/eri_campaign/logs/ite_gate12.log`.

## Where things stand

| item | status |
|---|---|
| ERI dilute (ethane, Gard & Tait ERI, 600 K) | **done.** Every gate passes; ABF-only screen **ABF-sufficient at K 64/256/1024** → no FR run licensed (`docs/eri_campaign/RESULTS.md`) |
| ERI loading study | not run (GPU engine needed; deprioritised) |
| A (Combariza 2011 model) | **not reproducible** (flexible BKS + all-atom Oie; paywalled; parameters closed) |
| **A′ (ITE, labelled)** | Gate 0 PASS (5 tests). Pilot: barrier on φ ≈ 15.8 kBT, ΔU ≈ +20.7 kJ/mol (about half entropic). Gates 1–2 running |
| B (DDR CO2), C (MFI CH4), E (H-SAPO-34) | not reproducible (paywalled / closed parameters / beyond budget). A labelled CH4/rigid-DDR substitute is in reserve |
| D (CH4 pair in water pore) | feasible with caveats; expected ABF-sufficient; low priority |

## Active jobs

* `scripts/eri_campaign/run_reference.py` chain for `ite_itq3` (MC → EM 2e-4 → EM 1e-4 → LM 2e-4 → EM 5e-5).
* Nothing on any GPU.

## Next automatic action

1. Analyse Gates 1–2 (tolerance 0.03 kJ/mol).
2. Gate 3 ABF-only calibration: horizon → ramp / pre-eq → bins.
3. ABF-only screen and the frozen classifier.
4. Two-arm production only if a cell is ESTABLISHMENT-LIMITED.

## Resource use

See `COMPUTE_LEDGER.md`.
