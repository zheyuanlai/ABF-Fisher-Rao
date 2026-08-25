# v2 physical-target pulse: post-mortem

Stage 0 of `docs/V3_PREREGISTRATION.md` (frozen v3.1). Executed 2026-08-25 on the
v2 engine with score-shape instrumentation added; 5 variants × 16 runs = 80 runs,
0 failures. Stage 0's only preregistered decision is the wording below.

## What Stage 0 was asked

Whether `score_clip = 5` collapsed the Fisher–Rao score, whether
`max_event_fraction` bound and silently changed the dose, and whether the
apparent gain is edge evacuation. All three were audit *inferences*; Stage 0
measures them.

## Result 1 — the clip did collapse the score, but by less than the audit estimated

Medians over 8 seeds and all FR events, at particle positions:

| variant | score_clip | raw span (nats) | applied span | fraction clipped |
|---|---|---|---|---|
| clip05 | 5 | 27.19 | 8.11 | **0.554** |
| clip20 | 20 | 26.20 | 25.23 | 0.035 |
| clipinf | none | 26.22 | 26.22 | 0 |

The v2 operator compressed a 27-nat score into 8 nats and clipped 55 % of
particles. Confirmed.

**Correction to the audit.** The audit reported a ~78-nat target range and ~88 %
of particles clipped. The 78 nats is the range of log(p̂/q) **on the grid**,
including |x| ≈ 3 where q ≈ 10⁻³⁴ and no particle ever sits. At the population
that actually carries a score — the particles — the span is 27 nats and the
clipped fraction is 55 %. Both numbers are correct about their own population;
the audit quoted the grid population for a particle-level claim. This is the
project's standing ratio/population defect class, found in our own audit.

## Result 2 — the clip was NOT load-bearing for the headline gain

| variant | median gain I_F | favorable | median final F′ ratio | ESS/K | replacements |
|---|---|---|---|---|---|
| clip05 | +11.32 % | 6/8 | 1.095 | 0.207 | 357 |
| clip20 | +8.68 % | 5/8 | 1.279 | 0.150 | 508 |
| clipinf | +8.87 % | 7/8 | **1.408** | 0.151 | 504 |

Removing the clip entirely leaves a gain of the same order (+8.9 % vs +11.3 %,
n = 8, seed range −18.7 % to +19.1 % — these medians are not separable). The
faithful operator does **not** rescue the method: it turns over ~40 % more
particles, ends with worse genealogy, and makes the final mean-force error
**41 % worse than plain ABF** where the clipped operator was 9.5 % worse.

**The damage grows monotonically with faithfulness: 1.095 → 1.279 → 1.408.**

## Result 3 — the target's estimation error is not the problem

The oracle-target arm (q ∝ exp(−βF_ref), no estimation error at all) gives
+12.38 % median I_F and a final F′ ratio of **1.379** on 6/8 seeds. A perfect
physical target under full ABF flattening is as harmful at the endpoint as an
estimated one. What fails is the target's *shape* against the applied bias, not
our estimate of it.

## Result 4 — the event cap did bind and did change the dose

At γ = 0.1, `fr_every` = 500 the realized event fraction with the cap released is
0.124 mean / 0.219 max, far above the 0.0977 the 0.10 cap permits: the cap bound
at essentially every firing and cut that cell's dose. Confirmed. Released, the
cell gives +4.52 % I_F and is the **only variant whose final F′ beats plain ABF**
(0.969, worse on just 2/8 seeds) — sparse opportunities with a full-strength dose
behave differently from frequent capped ones, which the v2 grid could not see.

## Result 5 — edge evacuation confirmed, and transient

Population outside the evaluation mask at pulse end versus at T = 100:
clip05 0.055 → 0.168; clipinf 0.008 → 0.164; oracle 0.047 → 0.164. Every variant
evacuates the un-evaluated strips during the pulse and every one relaxes back to
the plain-ABF level (~0.16–0.17) by the end of the run. The mechanism is real and
the benefit is not retained.

## Post-mortem wording (the decision Stage 0 was run to make)

> The v2 pilot's operator was unfaithful: `score_clip = 5` compressed a 27-nat
> Fisher–Rao score to 8 nats and clipped 55 % of particles, and
> `max_event_fraction` bound at high dose, so v2's grid did not vary the dose it
> believed it was varying. **The unfaithfulness was not, however, the reason the
> pilot failed.** With the clip removed the transient gain persists at the same
> order while the final mean-force error rises to 1.41× plain ABF, and an
> exact-oracle target is no better. v2's preregistered negative therefore stands
> and is strengthened rather than voided: physical-target Fisher–Rao reallocation
> under a *fully flattening* ABF bias buys a transient that it repays at the
> endpoint, and neither a faithful score nor a perfect target repairs it.

An earlier characterization — that v2 "never tested the physical target" — is too
strong and is withdrawn. v2 tested a distorted version of it; the undistorted
version, measured here, does not behave better.

## Consequences for v3.1

- Track P's registered prediction P2 (transient gain, endpoint damage, genealogy
  failure) is now supported *before* Track P runs. Track P remains in the
  campaign: it must be run at the frozen thresholds, on the same scopes and
  metrics as Track C, and it now has a pre-committed expectation.
- P1 as written ("removing the clip … materially changes behavior") is **half
  confirmed**: the score shape changes completely, the I_F outcome does not, the
  final-F′ outcome changes for the worse. Recorded as such; not amended.
- The target–bias conflict, not the clip, is the load-bearing defect. This is the
  hypothesis Track C is built on, and Stage 0 raises its prior.
- The `cap_fe500` result (only variant with final F′ < 1) is a hint that sparse,
  full-strength opportunities behave differently from frequent capped ones. v3.1
  already uses one stride (L_FR = 500) with no cap; no change is made to the
  frozen protocol on the strength of one diagnostic cell.
