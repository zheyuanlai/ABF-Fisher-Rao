# v3 post-mortem: the obstacle is representation, not target choice

Closes the v3 consistent-target ABF–FR campaign. Frozen protocol
`docs/V3_PREREGISTRATION.md` (v3.1 + Amendments 1–6); results in
`docs/V3_PILOT_RESULT.md`; per-opportunity mechanism in
`docs/V3_DIAGNOSTIC_REPLAY.md`. Nothing in the frozen pilot is re-run or
re-scored by this document.

## The project statement

    v2                  physical target conflicts structurally with full ABF
    v3 family law       removes that asymptotic conflict
    v3 pilot            but finite-time ESTIMATED targets still corrupt learning
    v3 oracle arm       and even EXACT targets cannot make repeated resampling
                        genealogy-safe
    diagnostic replay   because physical dynamics regenerates 94-99 % of the
                        marginal discrepancy between events

The last line is the new mechanism, and it moves the problem:

> **The obstacle is no longer how to choose the Fisher–Rao target.**
> **It is how to represent a continuously reweighted population without
> repeatedly converting every discrepancy into irreversible genealogical
> resampling.**

Statistical discrepancy regenerates. Ancestry does not. That asymmetry is the
fundamental v3 failure.

## Two independent failure modes

**1. Wrong-target reallocation damages learning.** On R₁₂ the deployable arm's
carrier tracks its no-FR control exactly until FR begins at step 10 000, then
stalls at ≈0.086 while the control continues to 0.021 — ratio 1.00 → 3.74. The
oracle-target arm shows no arrest. The trajectories agreeing *before* FR is
switched on is what makes this causal rather than correlational. The error is
independent of the family member g and is exponentiated into the target, so an
estimate good enough to bias forces can be far too poor to serve as a population
target.

**2. Perfect targeting does not restore self-limitation.** With D_cons ≡ 0 the
oracle arm still shows dose decay ≈ 1.0 and ancestral ESS/K ≈ 0.15, because each
event closes the discrepancy and the dynamics reopens 94 % of it before the next
one. This also explains θ = 1.0 at essentially every Track-C opportunity: the
governor keeps finding enough discrepancy to permit the full map. The finite-K
population simply does not remain near the target between opportunities.

## The deployable result worth keeping

**Capped ABF without FR is the strongest deployable method in this campaign**:
R_shape 1.33–1.91 against plain ABF, ε_F,2 reached on 8/8 seeds versus 5/8, and
final error *below* plain ABF's on R₁₂. Full flattening wastes a large share of a
finite budget in tails that carry no probability. Every future FR variant must
beat this baseline, not merely plain ABF.

No novelty is claimed for the capped family — partial flattening, well-tempered
targets and barrier-capped probability targeting have close relatives — but the
measurement stands on its own.

## Defects recorded against this campaign

1. **The window bug.** `io_utils` hardcodes burnin = 0 / stop = 1 for the
   `abf_only` method, so the first execution of all 17 FR arms ran FR over the
   whole run. Fixed by moving the window into the v3 block; arms re-run;
   baseline and controls proven unaffected on exact discrete counters.
2. **A gate that could not catch it.** The schedule gate built its own RunSpecs
   and never entered through the YAML path production uses. *Testing the engine
   is not testing the pipeline.*
3. **An unsatisfiable gate condition.** Advancement condition 7 (full-domain
   e_F ≤ 1.25) cannot be met by the capped family by construction — its no-FR
   controls already sit at 15–19×. Not amended after the fact; the verdict is
   unaffected because every candidate fails on 2–4 independent grounds.
4. **A scope-blind diagnostic.** Amendment 4c's `carrier_err` is full-domain and
   made failure mode 1 invisible; on R₁₂ it is unmistakable. *A diagnostic must
   be evaluated on the scope corresponding to the scientific claim.*
5. **Interpretation slips corrected on review:** the three reported medians do
   not factorize (the identity is per-seed); "same dose" was false for the oracle
   contrast (1201 vs 1123 replacements); and P9 is partially confirmed —
   throttling happened (θ 0.087 vs 1.000), predicted inertness did not.

## What is authorized next

Nothing from this pilot licenses β = 8, WCA, molecules, a new ρ search, a
retuned c_cut, or any rescue of the v3 schedule.

The single defensible successor is **v4-A**: hold the target fixed at the oracle
q ∝ exp(−β(F_ref + B_t)) so target error cannot excuse an operator failure, and
ask only whether repeated Fisher–Rao mass updates can be realized without
repeatedly destroying genealogy. The leading candidate is persistent weighted
replicas with resampling triggered by representation degeneracy rather than by
the arrival of an FR opportunity — separating **probability mass**, **physical
information**, and **particle representation**, three things v3 conflated every
500 steps.

v4-B — a practical target estimator, judged as an estimator against the oracle
marginal before being coupled to FR — is authorized only after v4-A succeeds.
