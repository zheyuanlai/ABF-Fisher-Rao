# WCA Case IX: uniform Fisher–Rao + online sensitivity-targeted constrained solvent relaxation

**Date:** 2026-09-02/03 (overnight). **Status:** W0 CLOSED (frozen gate failed; continued under amendment A1); W1 IN PROGRESS — this file is the morning report and is completed stage by stage by the detached orchestration (`scripts/overnight_wca_targeted_relax.sh`, log `results/targeted_relax_campaign/wca/overnight.log`).
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

1. **Did the online conditional-force-variance estimator validate?** — By value, yes: Pearson(v̂, v_constr) = 0.970 and v̂/v_constr = 1.67 ± 6 % across all twelve constrained sites. By the *frozen* rank gate, no: Spearman 0.455 (< 0.6), because the landscape is a plateau (below). Recorded as `SENSITIVITY_INVALID` under the frozen rule and `W0_PASS_A1` under the post-hoc amendment A1 (value-based criterion on the same data; [AMENDMENT_A1_W0_validity.md](../configs/targeted_relax_campaign/AMENDMENT_A1_W0_validity.md)).
2. **Where is WCA conditionally sensitive?** — Almost everywhere equally. Constrained Var(f | z) is 2102 ± 68 for every site with z ≥ 0.18 and 1211 at the compact end (z ≤ 0.01); the online field has the same shape (1850 vs 3570). There is no localised fibre-sensitive region like the gateway's flank; the only structure is a 40 % drop at the compact end.
3. **What is the measured τ_f(z)?** — 0.0064–0.0115 time units, i.e. 3–6 outer steps, at every site (replica-halves and time-halves agree within 10 %; the ACF has a weak positive tail out to 0.3–0.7 time units that the integral already includes; block-mean variances are 3–6× the i.i.d. value, consistent). The solvent force on the dimer decorrelates within one FR interval (5 steps). Every measured τ_f lies below the frozen floor of 0.02, so the frozen τ map is effectively flat at 10 dt.
4. **Where did the water-filling policy spend its compute?** — *pending W1*
5. **Which ρ passed, if any?** — *pending W1*
6. **Does FR + targeted relaxation beat plain FR?** — *pending W2*
7. **Does it beat cost-matched random relaxation?** — *pending W2*
8. **Does it remain faster on total force evaluations?** — *pending W2*
9. **Does the known FR-vs-ABF positive reproduce?** — *pending W2*
10. **Final verdict** — *pending*

## Stage W0 — sensitivity and timescale maps

**W0-A:** 8 plain-ABF runs (seeds 800–807, 476–533 s each on the shared GPU) with the second-moment accumulator and the
final configurations recorded. **Site selection (mechanical):** 10 quantiles of the cumulative sensitivity mass of the
median online field plus the argmin of v̂ in each half of the window → z = 0.012, 0.176, 0.281, 0.386, 0.498, 0.605,
0.713, 0.823, 0.931, 1.040 and controls −0.094, 0.760. **W0-B:** 64 pooled replicas per site, dimer projected exactly
to z_k, frozen-dimer solvent dynamics 4000 + 16000 steps, force recorded every 2 steps (2 min for all twelve sites).

| z | kind | τ_f | v_constr | v̂ (online) | v̂ / v_constr |
|---|---|---|---|---|---|
| −0.094 | control | 0.0064 | 1161 | 1775 | 1.53 |
| 0.012 | site | 0.0071 | 1260 | 1927 | 1.53 |
| 0.176 | site | 0.0115 | 2017 | 3153 | 1.56 |
| 0.281 | site | 0.0102 | 2095 | 3916 | 1.87 |
| 0.386 | site | 0.0069 | 1939 | 3500 | 1.80 |
| 0.498 | site | 0.0089 | 2190 | 3560 | 1.63 |
| 0.605 | site | 0.0080 | 2113 | 3663 | 1.73 |
| 0.713 | site | 0.0070 | 2126 | 3566 | 1.68 |
| 0.760 | control | 0.0071 | 2138 | 3556 | 1.66 |
| 0.823 | site | 0.0074 | 2128 | 3580 | 1.68 |
| 0.931 | site | 0.0077 | 2123 | 3608 | 1.70 |
| 1.040 | site | 0.0072 | 2147 | 3578 | 1.67 |

The frozen rank gate (Spearman ≥ 0.6, top-half sites above both controls) failed: ten of the twelve sites sit on one
plateau within ±3 %, and the right-half "control" is a plateau point because that half has no low-sensitivity region.
The estimator is proportional to the truth within ±6 % everywhere (the 1.67 factor is the online ensemble's extra
variance — moving dimer, ABF bias, kernel window — and does not affect the allocation, which uses the field's shape).
Amendment A1 was recorded before W1 with this diagnosis and with the prediction that targeting cannot matter on a flat
field with a five-step fibre. The frozen chain stopped at W0 (commit bebb8e7); the chain was relaunched from W1 under A1
(commit 12714b5).

## Stage W1 — cost ladder

*pending*

## Stage W2 — fresh confirmation

*pending*

## Sequencing note

W0-A (plain ABF, instrument only) was first launched from the tool after the engine/runner/prereg commit and before the
analyzer commit; it produced no completed run before it was stopped and relaunched by the orchestration after the
analyzer was committed (4c34b8c). No error metric of any arm was read before the committed analyzers ran.
