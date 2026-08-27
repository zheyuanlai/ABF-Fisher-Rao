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

---

# Amendment, 2026-08-27 18:20 UTC — still before any candidate arm

Two A0 calibration records now exist (`seed1000`, `seed1001`); the remaining 14 are running and
**no A6b or A6c run for WCA has been started**. The amendment is derived entirely from A0 data,
which is what Phase 1A exists to produce, and it *sharpens* the prediction rather than relaxing it.

## What the calibration shows so far (n = 2, will firm up at n = 16)

| | WCA | β=4 | β=8 | gateway |
|---|---:|---:|---:|---:|
| R_σ² | **2.00** | 73.7 | 18.4 | 648.5 |
| R_τ | **1.67** | 3.7 | 3.9 | 1.9 |
| R_Γ | **1.68** | 20.6 | 12.5 | 124.7 |
| valid-τ | **0.696** | 0.999 | 0.867 | 0.998 |
| ρ_s(Γ early, late) | **0.674** | 0.981 | 0.980 | 0.992 |

## The consequence I did not state precisely enough the first time

The right question is not "is R_Γ small" but **"does Γ move the target at all"**. Computing both
targets from A0 data alone — the full `r* ∝ sqrt(aΓ̂)` and the pure-geometry `r* ∝ sqrt(a)`, both
floored at 0.25/J:

| System | R_Γ | TV(r_aΓ, r_a) | max ratio | reading |
|---|---:|---:|---:|:--|
| β=4 | 20.1 | 0.199 | 2.03 | Γ reshapes the target |
| β=8 | 11.1 | 0.158 | 1.92 | Γ reshapes the target |
| gateway | 113.7 | 0.341 | 3.20 | Γ reshapes the target |
| **WCA** | **1.56** | **0.016** | **1.08** | **Γ barely moves the target** |

**On WCA, `r* ∝ sqrt(aΓ̂)` *is* `r* ∝ sqrt(a)` to within 1.6 % total variation.** The online
difficulty channel contributes essentially nothing there. So the WCA arm is not a test of the
difficulty half of the theory at all — it is a test of the **static geometric leverage half**,
which is a different (and separately interesting) experiment.

## Revised prediction

1. **A6b at WCA measures leverage alone.** Its advantage should be *smaller* than at the three
   heterogeneous systems. Quantitatively: the fixed-time error ratio at the horizon should be
   **≥ 0.92**, i.e. at most an 8 % improvement, against 0.652 / 0.879 / 0.923 elsewhere.
2. **The full-domain damage should be smallest of the four**, since the target is least lopsided —
   consistent with the damage being monotone in R_Γ (1.03 / 1.40 / 1.92 at R_Γ 12 / 21 / 124).
3. **Falsification.** If A6b at WCA improves *more* than the heterogeneous systems (ratio < 0.65),
   the gain does not follow Γ and the information-optimal framing is wrong. My earlier wording —
   "a speedup would falsify the mechanism" — was too strong: a *modest* speedup from the geometric
   factor alone is consistent with the theory and is what item 1 predicts.

## A limit that must be stated with the flatness, not after it

**WCA's Γ looks flat partly because its τ cannot be measured.** τ comes in at 4.9e-4–1.0e-3 time
units against a timestep of 2.0e-3 — a quarter to a half of one step — so 30 % of scored cells
return no valid fit and the rest are compressed by the median fallback and the 0.3 shrinkage. The
σ² half *is* solidly measured (an instantaneous spread needs no window) and spans only **2.0×**,
which on its own bounds R_Γ small unless τ were strongly heterogeneous. So the defensible claim is:

> **WCA's conditional force noise is nearly homogeneous across the reaction coordinate (σ² spans
> 2×), and its correlation time is unmeasurable at this timestep.**

Not: "WCA has flat Γ, therefore the theory predicts a tie." If the 16-seed calibration confirms
valid-τ < 0.80, WCA is marked **`Gamma unresolved`** and that caveat travels with every WCA number.
The estimator will not be adjusted to rescue it.
