# v3 diagnostic replay: what the per-opportunity series show

Diagnostic-only. Four arms replayed into `results/v3/diagnostic_replay/` after
the frozen pilot closed, with the Amendment 4c per-opportunity series finally
persisted. **These runs never replace frozen-pilot values.**

Contract verified: 61 rows per run, steps 10000–40000, `retention` equal to
`ess_anc_after / ess_anc_before` exactly.

## 1. The feedback loop is real — and my first metric could not see it

Carrier error on the primary scope R₁₂ (this is exactly `l2_F_R12`, the centred
error of the carrier the target is built from), as a ratio to the **same-bias
no-FR control**:

| step | deployable capped12+FT | oracle-target |
|---|---|---|
| 10000 (FR starts) | 1.000 | 1.000 |
| 12500 | 1.383 | 1.006 |
| 15000 | 1.758 | 1.020 |
| 25000 | 2.541 | 0.842 |
| 40000 (FR ends) | 3.474 | 0.877 |
| 50000 | 3.740 | 0.937 |

The deployable arm's carrier learning is **arrested at the moment FR begins**:
its R₁₂ error stalls at ≈0.086 and barely moves for 40 000 steps while the
control falls 0.0884 → 0.0206. The oracle arm tracks the control throughout and
finishes slightly ahead of it. Both arms are identical up to step 10000, so the
divergence is caused by FR and by nothing else.

This is the time-ordered evidence the endpoint numbers alone could not give:
**A_t error → q_t error → FR movement → A_t deterioration**, with the loop absent
when the target is correct.

**A logging defect of my own, recorded.** The `carrier_err` series I specified in
Amendment 4c is computed **full-domain**, and on that metric the two arms look
nearly identical (1.2673 → 1.1383 deployable versus 1.2673 → 1.1464 oracle). I
initially read that as refuting the feedback loop. It does not: the full-domain
carrier error is dominated by the far tails, where the capped bias deliberately
stops flattening, and is insensitive to the region the science is about. A
scope-blind diagnostic answered a scope-specific question and gave the wrong
answer. Any future logging of `carrier_err` should be scoped to R₁₂.

## 2. The consistency residue behaves exactly as Amendment 2 predicts

Median D_cons = KL(p*_{B_t} ‖ q_t):

| arm | at 10000 | at 40000 |
|---|---|---|
| oracle-target | **0.0000** | **0.0000** |
| deployable capped12+FT | 0.0597 | 0.0485 |
| P_FT ρ=0.85 | **10.62** | **9.80** |

The oracle target is exactly the frozen-bias marginal, so its residue is
identically zero — an in-situ confirmation of the Gate 1B algebra. The
deployable residue is small and slowly shrinking as A_t → F, exactly the
asymptotic-consistency claim. Track P's residue is ~10 nats and does **not**
shrink: the structural conflict, now quantified rather than argued.

## 3. The ESS governor never binds for Track C, and binds hard for Track P

| arm | median θ | fraction of opportunities at θ = 1 | median ESS_w/K |
|---|---|---|---|
| capped12 + FT | 1.000 | 0.99 | 0.962 |
| oracle-target | 1.000 | 1.00 | 0.964 |
| P_FT ρ=0.85 | **0.087** | 0.00 | 0.850 (= ρ) |

For every consistent arm the FT step jumps **all the way to q** at essentially
every opportunity, so ρ was inert: the ρ = 0.70 and ρ = 0.85 arms were the same
algorithm, which is why they returned identical gate rows. Under Track P's
inconsistent target the governor throttles hard to θ ≈ 0.087 and sits exactly at
its ESS floor.

**P9 re-scored: partially confirmed, not refuted.** I predicted the governor
would self-throttle under an inconsistent target *and* that this would leave
P-FT nearly inert. The throttling is confirmed and is dramatic (θ 0.087 against
1.000). The inertness is refuted — P_FT still made 2706 replacements, because a
small θ applied to a ~10-nat discrepancy still moves a great deal of mass. My
earlier scorecard called P9 simply "refuted" on the replacement count alone;
that was too coarse.

## 4. Why self-limitation fails even with a perfect target

Per-opportunity KL(p̂ ‖ q) drop, against the regrowth before the next opportunity:

| arm | median drop | median regrowth | regrowth / drop |
|---|---|---|---|
| capped12 + FT | 0.0143 | 0.0139 | 0.975 |
| **oracle-target** | 0.0157 | 0.0148 | **0.941** |
| capped12 + hold-out | 0.0130 | 0.0137 | 1.052 |
| P_FT ρ=0.85 | 2.858 | 2.825 | 0.988 |

Each FR event closes the discrepancy and the dynamics reopens ~95 % of it before
the next one — **including with an exactly correct target**. So the absence of
dose decay is not caused by target error. At finite K the empirical marginal
cannot stay at q: it wanders at a rate set by the dynamics and the KDE, FR keeps
pulling it back, and each pull spends ancestry irreversibly.

This is the mechanism behind the two-failure separation, and it explains why the
oracle arm has excellent accuracy and still-collapsed genealogy. It also confirms
Amendment 4b's correction: the residual dose is a finite-time and finite-K floor,
not a readout of carrier error.

## Consequence for any successor

A perfect deployable target estimator would fix failure mode 1 and leave failure
mode 2 untouched. The regrowth ratio of 0.94 with a *correct* target says the
current design would keep resampling at full rate indefinitely, so genealogy
safety has to come from the representation — how often mass movement is realized
as birth–death at all — rather than from better targeting.
