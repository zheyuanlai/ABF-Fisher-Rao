# WCA Case IX: uniform Fisher–Rao + online sensitivity-targeted constrained solvent relaxation

**Date:** 2026-09-02/03 (overnight). **Status:** IN PROGRESS — this file is the morning report and is completed stage by stage by the detached orchestration (`scripts/overnight_wca_targeted_relax.sh`, log `results/targeted_relax_campaign/wca/overnight.log`).
**Prereg:** [`wca_fr_targeted_relax_prereg.json`](../configs/targeted_relax_campaign/wca_fr_targeted_relax_prereg.json) (frozen at 199ef32 before any run; analyzer at 4c34b8c; orchestration at ca36c32).
**Parents:** the four gateway stages ([GATEWAY_TARGETED_RELAXATION.md](GATEWAY_TARGETED_RELAXATION.md) and its parents): the fibre, not the marginal, is the ingredient; targeted relaxation at 1× cost; FR is the allocator to pair it with. **No transport is implemented.**
**GPU:** 3 only, shared with another user's jobs throughout.

## The method under test

ABF + uniform Fisher–Rao reallocation (the accepted Case IX arm, every knob verbatim) + **targeted constrained solvent
relaxation at fixed dimer coordinate**: at every FR opportunity, after birth–death, selected replicas have their dimer
frozen exactly while the solvent evolves under the physical potential for `m_i` inner steps. *Where*: importance
`a_i = v̂_t(z_i)`, the online conditional variance of the local mean force from a second-moment ABF accumulator
(per-bin variance first, then the inherited kernel smoothing; never the reference). *How much*: water-filling
`t_i = τ_i/2 [log(2a_i/(λτ_i))]₊` with `τ_i = τ_f(z_i)` from the W0 map and `Σ m_i = ρ N fr_every` replica-steps per
opportunity (ρ = the notional extra cost). Inner steps deposit nothing anywhere. A cost-matched random control permutes
the same durations across replicas. Full compute accounting: `C(t) = N·step + Σ inner replica-steps`.

## Morning report (in the order asked)

1. **Did the online conditional-force-variance estimator validate?** — *pending W0*
2. **Where is WCA conditionally sensitive?** — *pending W0*
3. **What is the measured τ_f(z)?** — *pending W0*
4. **Where did the water-filling policy spend its compute?** — *pending W1*
5. **Which ρ passed, if any?** — *pending W1*
6. **Does FR + targeted relaxation beat plain FR?** — *pending W2*
7. **Does it beat cost-matched random relaxation?** — *pending W2*
8. **Does it remain faster on total force evaluations?** — *pending W2*
9. **Does the known FR-vs-ABF positive reproduce?** — *pending W2*
10. **Final verdict** — *pending*

## Stage W0 — sensitivity and timescale maps

*pending*

## Stage W1 — cost ladder

*pending*

## Stage W2 — fresh confirmation

*pending*

## Sequencing note

W0-A (plain ABF, instrument only) was first launched from the tool after the engine/runner/prereg commit and before the
analyzer commit; it produced no completed run before it was stopped and relaunched by the orchestration after the
analyzer was committed (4c34b8c). No error metric of any arm was read before the committed analyzers ran.
