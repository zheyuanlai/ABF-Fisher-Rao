# v4-A: genealogy-safe Fisher–Rao under an oracle target

## Material Passport

- Artifact type: code-experiment protocol
- Status: **FROZEN 2026-08-25, before any v4 run.** Changes require a numbered amendment.
- Date drafted: 2026-08-25
- Predecessor: `docs/V3_POST_MORTEM.md` (v3 closed, 0/11 candidates passed)
- Revision: draft v1's Gate 0 proposed a *weighted* ABF estimator; that is
  withdrawn (see "The red-line that shaped this draft").

## The single research question

> Can repeated Fisher–Rao mass updates be realized **without repeatedly
> destroying genealogy**?

The target is held at the oracle q_t ∝ exp(−β(F_ref + B_t)) throughout, so
target-estimation error cannot excuse an operator failure. Target estimation is
v4-B and is out of scope.

## The red-line that shaped this draft

Draft v1 proposed weighting the ABF estimator by the FR masses. **Withdrawn.**
After persistent updates, w_i(t) ∝ ∏_{k<t} G_k(ξ(q_i(t_k))) is a *path* functional:
every multiplier depends only on ξ at the moment it was applied, but two replicas
at the *same current* ξ can carry wildly different weights. ABF's justification
needs the biasing factor to be constant on the current fibre Σ(z), which is what
makes π_B(dq | ξ=z) = π(dq | ξ=z). A path weight has no such property.

Measured on this benchmark (4096 replicas, 6000 steps, synthetic multipliers
depending on ξ alone): within a single narrow ξ-bin the accumulated weights span
a factor of **8 × 10⁶**. They are nowhere near constant on the fibre. In this
particular toy the correlation between log w and the fibre coordinate y is small
(0.02–0.08), so the leading damage would be **collapse of the effective sample
size within each fibre** rather than a large directional bias — but neither is
acceptable, and neither is guaranteed to stay small on a system with stronger
ξ–fibre coupling.

Letting FR masses weight the estimator would therefore re-conflate the two things
v4-A exists to separate, and would make v4-A *harder* to interpret than v3.

## The governing distinction

    FR weights          ->  probability mass
    physical propagation ->  statistical information
    resampling          ->  particle representation

Two empirical measures are maintained, and they are not interchangeable:

    nu_t  = (1/K) sum_i  delta_{q_i}          physical-information ensemble
    mu_t  = sum_i w_i    delta_{q_i}          Fisher-Rao mass ensemble

`nu` feeds ABF. `mu` feeds the FR marginal and the resampling decision.

**The ABF estimator is the unweighted v3 estimator, unchanged.** The binding rule:

> Changing w_i alone can never change F̂′. Only physical propagation creates an
> ABF observation.

This is strictly stronger than draft v1's gate and is the direct analogue of the
clone-information rule v3 already enforced.

## The candidate algorithm

State per replica: configuration q_i, FR mass w_i, ancestor a_i, hold-out h_i.

1. **Propagate** every replica under the current bias using the matched
   physical-noise bank. Weights ride along unchanged: w_i' = w_i.
2. **Accumulate ABF** — if h_i = 0 the propagated configuration contributes
   exactly **one** observation, *independent of w_i*; if h_i > 0 it contributes
   zero. Then decrement h_i.
3. **Update bias** exactly as capped-12 in v3: A′_t = F̂′_t, B_t = g(A_t) − A_t.
4. **FR opportunity** (61 of them, steps 10000…40000 stride 500): compute the
   weighted marginal p̂_w(z) = Σ_i w_i K_η(z − z_i) and the oracle target, then
   move mass only —
   w_i⁺ ∝ w_i · (q_t(z_i) / p̂_w(z_i))^θ, normalized. **No cloning. No
   accumulator change. Positions unchanged.**
5. **Representation check** — if ESS_w = 1/Σ_i (w_i⁺)² < ρ_resample · K, systematic
   resample K configurations with probabilities w_i⁺ and reset w_i = 1/K;
   otherwise do nothing.

Terminology: **degeneracy-triggered resampling** (or ESS-triggered), never
"conditional resampling" — "conditional" in this project means π(dq | ξ = z) and
the collision would be dangerous.

## Frozen parameter decisions (no grids)

| choice | value | reason |
|---|---|---|
| θ | **1** for the persistent-weight arms | v3's oracle arm already ran at θ ≈ 1 at essentially every opportunity; reusing the ESS governor would let dose vary with accumulated degeneracy, changing FR dose *and* representation at once |
| ρ_resample | **0.50, single value** | a transparent degeneracy threshold; v4-A is not a tuning search. If it fails, it is **not** rescued with 0.3 or 0.7 inside v4-A |
| L_hold | **500, unchanged** | persistent weighting creates no clones, but a triggered resampling does; the v3 continuation convention applies verbatim |
| ABF estimator | **v3 cumulative-from-0, unweighted, no forgetting** | a forgetting factor would make v4-A test three changes at once |

Everything else inherits from v3: β = 4, K = 256, 50 000 steps, seeds 0–7, scope
R₁₂, the 61-opportunity window, and the thresholds already frozen in
`results/v3/V3_THRESHOLDS.json` — which are **not** re-frozen.

## Arms — only two are generated

| arm | role | source |
|---|---|---|
| 1. capped-12 no FR | physical baseline | **reuse frozen v3 run** |
| 2. capped-12 oracle FT, resample every opportunity | the bad-representation reference | **reuse frozen v3 run** |
| 3. capped-12 oracle, persistent weights, **never** resample | representation diagnostic, *not* a candidate | new |
| 4. capped-12 oracle, persistent weights + degeneracy-triggered resampling | **the only v4-A candidate** | new |

Arms 1 and 2 already exist in the frozen v3 dataset and are reused rather than
regenerated. Arm 3 must physically reproduce arm 1; if it does not, that is an
engineering failure, not a result.

## Gate 0 (all must pass before any scientific run)

- **0A — mass/information separation.** An FR weight update with zero physical
  propagation leaves the ABF count accumulators, force accumulators, A_t and B_t
  *exactly* unchanged.
- **0B — arm-3 physical identity.** Persistent weights with no resampling produce
  the same physical trajectories and the same ABF estimator as capped-12 no-FR
  under the same noise bank: exact on discrete counters, within the Amendment-1
  1e-5 tolerance on profiles. Only the sidecar weights may differ.
- **0C — equal-weight reduction.** With w_i = 1/K the weighted KDE reproduces the
  v3 FR density estimate exactly.
- **0D — log-space invariance.** Adding any constant C to every log weight leaves
  w, p̂_w, ESS_w and the next FR update unchanged. Tested at score ranges far
  beyond the measured 8 × 10⁶ ratio — up to 1000 nats.
- **0E — resampling is representation-consistent.** For a fixed weighted cloud,
  E[(1/K) Σ_j φ(q_j⁺)] = Σ_i w_i φ(q_i) over test observables, within
  Monte-Carlo tolerance.
- **0F — no information from clones.** Immediately after a triggered resampling,
  Δ(ABF accumulators) = 0; extra children contribute only after L_hold steps.

## Genealogy metrics: count *and* mass

In v3 every replica carried 1/K, so family size and family mass were the same
number. They no longer are, and reporting only one would be misleading — arm 3
has ESS_anc^count = K *by construction* while its mass may concentrate almost
entirely on one ancestor.

    c_a = (1/K) #{i : a_i = a}        ESS_anc^count = 1 / sum_a c_a^2
    m_a = sum_{i : a_i = a} w_i       ESS_anc^mass  = 1 / sum_a m_a^2
    c_max = max_a c_a                 m_max = max_a m_a

Both are reported for every arm, always as a pair.

## Success criteria

**Mechanism-positive** — arm 4 must satisfy all of:

1. **N_resample ≤ 12** on ≥ 6/8 seeds (≥ 5× fewer representation resets than arm 2's 61).
2. **Count genealogy:** ESS_anc^count/K ≥ 0.5 and c_max ≤ 0.10, on ≥ 6/8 seeds.
3. **Mass genealogy:** ESS_anc^mass/K ≥ 0.5 and m_max ≤ 0.10, on ≥ 6/8 seeds.

   Both are evaluated at the **final** frame, not as a minimum over time. The
   trigger fires *after* particle ESS crosses 0.5K, so a minimum-over-time
   condition at the same threshold would be structurally self-contradictory.
4. **No harm vs capped-12 no-FR** on R₁₂: final e_F ≤ 1.05 × and final e_F′ ≤ 1.05 ×.
5. **FR still adds something:** S at ε_F,2 ≥ 1.05 versus capped-12 no-FR,
   favorable on ≥ 6/8 seeds.

**Full success (secondary, descriptive).** With final error E,

    R_preserve = (E_C12 - E_v4) / (E_C12 - E_oracle)

is 0 when nothing beyond capped ABF is gained and 1 when the whole v3 oracle gain
is retained. Pre-registered target: **R_preserve ≥ 0.75** for both final e_F on
R₁₂ and barrier e_F′.

## Registered predictions

- **V4-P1** Arm 3 is physically trajectory-identical to capped-12 no-FR, and its
  persistent weights develop substantial mass concentration: the v3 failure
  *moves* from count genealogy to weight/mass degeneracy rather than vanishing.
- **V4-P2** Arm 4 triggers resampling on ≤ 12 of 61 opportunities.
- **V4-P3** Arm 4 keeps both count- and mass-ancestral ESS above 0.5K with max
  family fractions below 0.10 on ≥ 6/8 seeds.
- **V4-P4** Arm 4 improves ε_F,2 time-to-accuracy by ≥ 5 % over capped-12 no-FR
  with no final F/F′ harm.
- **V4-P5** Arm 4 retains ≥ 75 % of the v3 oracle arm's incremental endpoint gain
  over capped-12 no-FR.
- **V4-P6** Persistent weighting reduces the *genealogical cost of representing*
  repeated FR corrections. It is **not** predicted to reduce the physical
  between-opportunity discrepancy regeneration itself.

**Not predicted:** that v3's 0.94 regrowth ratio is unchanged. Arm 4's resampling
alters the finite ensemble and therefore its subsequent physical evolution, so
the measured regrowth may change. Arm 3, whose trajectories are identical to the
no-FR control, is the arm that exposes the intrinsic regrowth mechanism without
genealogy loss. Whether representation alters regrowth in arm 4 is measured, not
predicted.

## Arm 3 has no stopping rule

Arm 3 exists to answer: *what happens to FR probability mass if it is never
converted into genealogy?* It therefore runs the full 50 000 steps regardless of
how degenerate its weights become. ESS_w/K → 10⁻² or one replica carrying 99 % of
the mass is **the result**, not an engineering failure: it would show the v3
degeneracy moving from ancestry into mass. The only abort condition is genuine
numerical pathology (non-finite log weights), and that fails closed.

Recorded instead of stopping, using arm 4's own threshold so the two are
comparable:

    T_deg  = min{ t : ESS_w(t) < 0.50 K }     "when arm 4 would have resampled"
    T_0.1  = min{ t : ESS_w(t) < 0.10 K }     descriptive only, not a parameter
    min_t ESS_w(t)/K ,  max_{i,t} w_i(t)

## All mass arithmetic is in log space

The scientific state is **log w**, never normalized w. Every update renormalizes
by log-sum-exp so that LSE(ℓ) = 0 and every log w ≤ 0:

    ESS_w   = exp( − LSE(2ℓ) )          w_max = exp( max_i ℓ_i )

both safe to exponentiate since ESS_w ∈ [1, K] and max ℓ ∈ [−log K, 0]. The
weighted marginal is likewise a log-sum-exp,
log p̂_w(z_i) = LSE_j [ ℓ_j + log K_η(z_i − z_j) ], never a float sum of w·K, and
the oracle target stays in log form. At θ = 1 the update is simply

    ℓ_i⁺ = ℓ_i + log q_t(z_i) − log p̂_w(z_i)

followed by one normalization. **log p̂_w is only ever required up to an additive
constant**, because the normalization that follows absorbs it — which is what
lets the weighted grid density be built from max-shifted weights and reduce
*exactly* to the v3 construction at equal weights.

No ε floor is introduced: the Gaussian KDE is positive and every particle
contributes its own self-kernel. If a required log quantity is non-finite, **fail
closed** — never clip it to a plausible value.


## Implementation order (binding)

1. A standalone `PersistentMass` object owning **only** {log w_i}: log-sum-exp
   normalization, weighted KDE in log space, the FR mass update, particle ESS,
   count/mass ancestry, the trigger condition, and reset-to-uniform. It holds
   **no reference to any ABF accumulator**, so the architecture enforces the
   mathematics rather than relying on discipline.
2. Cloud-level tests of that object.
3. Wire it into the engine as a sidecar; a separate module owns resampling.
   Only the orchestration layer connects physical dynamics, mass, and
   representation.
4. **Gate 0B is the decisive pre-science experiment.** Run arm 3 first. Expect
   its trajectories and F̂ to match capped-12 no-FR while ESS_w/K → 0. That
   simultaneous outcome validates the decomposition: probability mass can
   degenerate dramatically without changing physical information, until we
   choose to alter its representation.
5. Only after Gate 0B passes may arm 4 run.

v4-B is not touched. If v4-A fails even with an oracle target and persistent
weights, the problem is deeper than target estimation or resampling frequency.
