# v4-A: genealogy-safe Fisher–Rao under an oracle target — DRAFT

## Material Passport

- Artifact type: code-experiment protocol
- Status: **DRAFT for red-line — not frozen; no scientific run until approved**
- Date drafted: 2026-08-25
- Predecessor: `docs/V3_POST_MORTEM.md` (v3 closed, 0/11 candidates passed)

## The single research question

> Can repeated Fisher–Rao mass updates be realized **without repeatedly
> destroying genealogy**?

The target is held at the oracle throughout, q_t ∝ exp(−β(F_ref + B_t)), so that
**target-estimation error cannot excuse an operator failure**. Target estimation
is v4-B and is not in scope here.

## What v3 established that this design answers

With D_cons ≡ 0, the v3 oracle arm still showed dose decay ≈ 1.0 and ancestral
ESS/K ≈ 0.15, because each FR event closes the marginal discrepancy and physical
dynamics reopens 94 % of it before the next. Every correction was immediately
converted into birth–death, so discrepancy regenerated and ancestry did not.

The response is to stop conflating three distinct objects, which v3 merged every
500 steps:

    probability mass   |  physical information  |  particle representation

## The candidate: persistent weights, conditional resampling

Replicas carry explicit weights w_i. At an FR opportunity, mass moves in weight
space only:

    p̂_w(z)   = Σ_i w_i K_h(z − z_i)              (weighted marginal estimate)
    w̃_i⁺     = w_i · ( q(z_i) / p̂_w(z_i) )^θ
    w_i⁺     = w̃_i⁺ / Σ_j w̃_j⁺

**No cloning occurs merely because an opportunity arrived.** Resampling is a
numerical representation operation, invoked only on degeneracy:

    resample when   ESS_w = 1 / Σ_i w_i²  <  ρ_resample · K

This is closer to the continuum flow than forcing an equal-weight representation
after every step, and it attacks the exact asymmetry v3 exposed.

## Gate 0 (theory/engineering, before any science): the weighted estimator

Unequal weights mean the ABF estimator can no longer treat every observation as
1/K. The weighted form

    F̂′_t(z) = Σ_i w_i K_h(z − ξ(q_i)) f(q_i) / Σ_i w_i K_h(z − ξ(q_i))

must be **derived from the intended particle interpretation, not inserted
heuristically**, and the derivation is itself a gate artifact.

The binding requirement, directly analogous to the clone-information rule v3
already enforced:

> FR changes weights **without generating new physical samples**. It must not be
> able to create information in the ABF accumulators by multiplying a replica's
> weight.

Proposed gates: (i) a written derivation of the weighted estimator from the
particle interpretation; (ii) an FR step applied with **zero** physical
propagation must leave every ABF accumulator unchanged; (iii) with all weights
equal the weighted estimator must reduce exactly to the v3 estimator; (iv) a
frozen-bias no-FR run must recover F_ref under weighted estimation to the same
tolerance as the unweighted one.

## Arms (deliberately narrow — 4 arms × 8 matched seeds)

| # | arm | purpose |
|---|---|---|
| 1 | capped-12, no FR | the strongest deployable baseline from v3 |
| 2 | capped-12 + oracle target, FT + resample every event | the v3 reference point being improved on |
| 3 | capped-12 + oracle target, persistent-weight FR, no conditional resample | isolates weight-space mass movement |
| 4 | capped-12 + oracle target, persistent-weight FR + conditional resampling | the candidate |

**No Track P. No tempered family. No c_cut = 8. No target-estimator variants.
No molecules.** The question is isolated enough that four arms should settle it.

Everything else inherits from v3 unchanged: β = 4, K = 256, 50 000 steps, seeds
0–7, scope R₁₂, the 61-opportunity window, and the frozen thresholds in
`results/v3/V3_THRESHOLDS.json` (they were fixed from plain ABF and remain valid;
they are **not** re-frozen).

## Success criterion

Arm 2's accuracy in the frozen v3 pilot is the standard to preserve:

    S at ε_F,2                       ≈ 2.39
    final e_F  / plain ABF           ≈ 0.49
    final barrier e_F′ / plain ABF   ≈ 0.52

**v4-A succeeds if a candidate retains most of that accuracy while achieving
ancestral ESS/K ≥ 0.5 and max family weight ≤ 0.10** — i.e. escaping arm 2's
ESS/K ≈ 0.15. Precise retention fractions, the resampling-trigger values ρ_resample
to test, and the censoring convention carry over from v3 and are to be fixed in
this document **before** any run.

## Registered predictions (to be completed before freezing)

- Conditional resampling reduces total resampling events by ≥ 5× relative to
  arm 2 at comparable accuracy.
- Arm 3 (weights only, never resampled) has excellent genealogy and degrades in
  effective sample size instead — i.e. the failure moves rather than vanishing.
- The regrowth ratio measured in v3 (0.94 with a correct target) is a property of
  the dynamics and will be **unchanged** by the representation; what changes is
  how much ancestry each correction costs.

## Open questions for red-line

1. Should the weighted estimator use a forgetting factor, given that v3's
   accumulators are cumulative from t = 0 and weights now vary in time?
2. Does the hold-out clone policy still apply, or is it subsumed by weights?
3. ρ_resample values to test, and whether the FR step size θ keeps the v3 ESS
   governor or becomes a fixed schedule now that resampling is decoupled.
