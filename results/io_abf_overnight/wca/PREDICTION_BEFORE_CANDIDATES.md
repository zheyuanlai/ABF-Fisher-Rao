# A prediction registered before any WCA candidate arm runs

Written **2026-08-27 18:00 UTC**. At this moment exactly **one** A0 calibration record exists
(`A0__seed1000.npz`, 592 s). No A6b or A6c run for WCA exists, and none has been started — the
phase chain runs the remaining 15 A0 calibration seeds first. Verifiable from the file
modification times in `results/io_abf_overnight/wca/screening/` and from the chain log.

## Why this is being written

The campaign lost its Q4 test. β=4 was designated the flat-Γ control on the repository's
"ABF-sufficient" classification, and measured it carries R_Γ = 21.6 — *larger* than the β=8
candidate's 12.4. Every system that ran tonight has large Γ heterogeneity (12–124), so
"A6b ≈ A0 where Γ is flat" was never testable. That is the same failure the q-r campaign recorded
for K0, and I do not want to discover a third instance of it after the fact.

The first WCA A0 record suggests WCA may be the missing control. On that seed, over the 28 scored
allocation cells:

| | min | max | ratio |
|---|---:|---:|---:|
| σ² | 1848 | 4219 | 2.3 |
| τ | 4.88e-4 | 1.04e-3 | 2.1 |
| Γ | 1.285 | 3.234 | **2.5** |

against R_Γ = 12.4, 20.6 and 124.7 on the three systems that completed. One seed and a min/max
range are not the pooled Q₉₀/Q₁₀ statistic the protocol uses, so this is a hint, not a number.

## The prediction

Let `R_Γ^WCA` be the pooled Q₉₀/Q₁₀ over scored cells from the 16-seed A0 calibration.

1. **If `R_Γ^WCA < 5`** — small against 12–124 — then WCA is the flat-Γ control this campaign
   lacked, and the frozen theory predicts **A6b ties A0**: `S(ε₂) ∈ [0.95, 1.15]`, i.e. it fails
   the `S ≥ 1.15` bar *from below rather than from a correctness failure*, and the full-domain
   damage is small because a near-uniform `a Γ` produces a near-uniform `r*`.
2. **If `R_Γ^WCA ≥ 5`**, this prediction does not apply and WCA is another heterogeneous system.

A tie under (1) would be the campaign's first genuine Q4 evidence. **A speedup under (1) would
falsify the mechanism story** — it would mean the gain does not come from following Γ.

## A second thing already visible, and it is a limit not a result

WCA's measured τ is **4.9e-4 to 1.0e-3 time units against a timestep of 2.0e-3** — that is
**0.25 to 0.5 steps**. The correlation time of the per-cell mean force is *below the integration
timestep*, so the lag-1 AR(1) fit sits in its `φ → 0` failure mode and 28.6 % of scored cells came
back invalid on this seed. No sampling cadence can fix this: you cannot sample faster than `dt`.

If the 16-seed calibration confirms valid-τ < 0.80, WCA is marked **`Gamma unresolved`** under
§8 of the preregistration. Candidate arms still run, but **a candidate failure there may not be
attributed to the theory** — only to the fact that half of `Γ = σ²τ` is unmeasurable in this
system at this timestep. The estimator will not be adjusted to rescue it.
