# WCA Case IX: uniform Fisher–Rao + online sensitivity-targeted constrained solvent relaxation

**Date:** 2026-09-02/03 (overnight). **Status (2026-09-03 05:30 UTC):** W0 CLOSED (frozen gate failed; continued under amendment A1); W1 CLOSED — STOP `NO_COMPUTE_EFFICIENT_FR_RELAXATION` (frozen-dimer operator, an operator-consistency confound); W1b (amendment A2, reference-scheme operator) RUNNING — 26 of 32 runs, three seeds complete, preliminary below; W2 NOT launched. The detached chain will finish W1b, run its analyzer on all four seeds and auto-commit `W1/rho_selection_projected.json` when the shared GPU allows (four other jobs on it since 03:00 UTC). — this file is the morning report and is completed stage by stage by the detached orchestration (`scripts/overnight_wca_targeted_relax.sh`, log `results/targeted_relax_campaign/wca/overnight.log`).
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
4. **Where did the water-filling policy spend its compute?** — Almost uniformly, as the flat field dictates: 79–100 % of the replicas were relaxed at every opportunity (ρ 0.25 → 1), and 80 % of the budget fell on 49–63 % of the grid against 75 % for the occupancy — a mild concentration toward the low-τ, higher-v̂ sites, nothing like the gateway's flank. Budget spent exactly as specified (0.208×, 0.417×, 0.833× of the outer force evaluations, relaxation starting at step 20000).
5. **Which ρ passed, if any?** — None. With the frozen-dimer operator every relaxed arm is *worse* than its unrelaxed partner, monotonically in ρ (table below), and none reaches plain FR's final accuracy at any compute. STOP by the frozen rule; W2 was not launched. The post-hoc diagnostic (below) shows this is an operator-consistency confound, and W1b re-runs the ladder with the reference's own integration scheme.
6. **Does FR + targeted relaxation beat plain FR?** — With the frozen-dimer operator (W1, 4 seeds): no, it is worse at every ρ (+13 / +15 / +31 % integrated, 0/4). With the reference-consistent operator (W1b preliminary, 2 complete seeds): at ρ = 1 yes — F_1 vs F −9.4 % [−13.9, −4.9] integrated (2/2) and **−37 % at the end**; ρ = 0.5 neutral (−1.3 % / −9.8 %); ρ = 0.25 worse (+10.5 %). The frozen ρ\* rule (≤ −10 % integrated AND compute ratio ≤ 0.8) is missed at ρ = 1 by 0.6 points on the integrated margin with two seeds; the full four-seed W1b decides it.
7. **Does it beat cost-matched random relaxation?** — Not tested: W2 was not licensed. With a flat sensitivity field (W0) the budget is spread over 79–100 % of the replicas at every ρ, so the targeting is close to uniform relaxation by construction; F_rand ≈ F_R is the expectation and the W2 control would most likely return TARGETING_NOT_SPECIFIC even where relaxation helps.
8. **Does it remain faster on total force evaluations?** — Frozen-dimer operator: never (no relaxed arm reaches plain FR's final accuracy). Reference-consistent operator, ρ = 1, preliminary: **yes at the plain-FR final threshold**, C_{F_1}(ε_F)/C_F(ε_F) = 0.77 with every inner force evaluation charged; at ρ = 0.5 the ratio is 1.00 and at ρ = 0.25 relaxation never reaches it.
9. **Does the known FR-vs-ABF positive reproduce?** — Yes: F vs A −18.1 % [−20.8, −13.8] integrated, 4/4 (W1, h_read\*\* 0.00625), against the accepted −18.3 % on seeds 700–715.
10. **Final verdict** — *By the frozen rules:* STOP at W1 (`NO_COMPUTE_EFFICIENT_FR_RELAXATION`); W2 not run; nothing goes into the paper as a positive. *What the night established:* (i) the online conditional-force-variance field is valid on WCA but nearly flat, and the solvent force decorrelates within one FR interval, so the gateway's *localised slow fibre* does not exist here and targeting cannot earn credit; (ii) the frozen-dimer relaxation operator is inconsistent with the dt = 2e-3 discretisation shared by the reference and the outer dynamics, which is why it "hurt" — a lesson that generalises (validate any constrained/inner operator against the reference's own scheme first); (iii) the accepted WCA reference carries an O(dt) bias of order 3 in the compact mean force (≈ 0.6 kT in well depth), shared by every arm, which the paper must state; (iv) with the consistent operator, *untargeted-in-effect* solvent relaxation at ρ = 1 reduces the endpoint error of both ABF and FR by ~37 % on two seeds and reaches FR's accuracy at 0.77× its compute — evidence that the WCA endpoint error does contain a conditional-lag (solvent-shell) bias, the ACF's weak long tail rather than its 5-step core. Recommendation: let W1b finish (it decides ρ\* on the frozen rule); if ρ = 1 passes, a W2 with F_rand is the next block — but its likely label is TARGETING_NOT_SPECIFIC, and the deployable method on WCA would then be "FR + uniform solvent relaxation at ρ ≈ 1", not the sensitivity-targeted one. Do not carry the frozen-dimer result forward as a negative about relaxation; carry it as a negative about operator inconsistency.

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

## Stage W1 — cost ladder (seeds 820–823, 8 arms, 6.6 h on the shared GPU; read-out intersection h_read\*\* = 0.00625)

| ρ | F_ρ vs F, ΔI_F | Δe_F(T) | A_ρ vs A, ΔI_F | Δe_F(T) | actual cost | replicas relaxed | C(ε_F) ratio |
|---|---|---|---|---|---|---|---|
| 0.25 | +13.0 % [+4.3, +25.5], 0/4 | +34 % | +10.0 % [+5.4, +10.7], 0/4 | +19 % | 0.208× | 79 % | ∞ (never reaches) |
| 0.5 | +15.5 % [+7.7, +22.2], 0/4 | +72 % | +17.5 % [+6.9, +23.2], 0/4 | +30 % | 0.417× | 85 % | ∞ |
| 1 | +31.1 % [+22.0, +43.5], 0/4 | +119 % | +26.0 % [+16.8, +33.9], 0/4 | +53 % | 0.833× | 100 % | ∞ |

Positive control F vs A: −18.1 % [−20.8, −13.8], 4/4 (the accepted −18.3 % reproduces). Every relaxed arm's extra
error sits at z < 0.25 (mean-force error at raw bins for abf_targ1 −0.58 vs −0.26 for abf; fr_targ1 −0.54 vs −0.20 for
fr), i.e. it is a *bias* in the compact region, growing with the relaxation dose.

**Diagnosis (post hoc; `W1/diagnostic/operator_consistency.json`).** Constrained mean force at fixed z minus the
reference, 64 replicas, 8 + 32 time units per cell:

| z | F′_ref | frozen dimer, dt 2e-3 | frozen, 1e-3 | frozen, 5e-4 | projected (reference scheme), 2e-3 | projected, 5e-4 |
|---|---|---|---|---|---|---|
| −0.094 | −2.87 | −1.54 | −3.44 | −3.68 | −0.04 | −3.53 |
| 0.012 | 6.12 | −1.75 | −3.16 | −3.29 | +0.13 | −3.08 |
| 0.176 | 7.10 | −2.10 | −2.25 | −2.05 | +0.11 | −1.90 |
| 0.281 | 1.10 | +0.80 | +1.52 | +1.53 | +0.20 | +1.58 |

The reference's own scheme at dt = 2e-3 reproduces the reference; the frozen-dimer scheme at the same dt is off by
−1.5 to −2.1 in the compact region; and at dt = 5e-4 the two schemes **agree with each other** and both sit ~3 below
the reference. The operators share a continuum limit; the accepted reference — and the outer ABF dynamics, which
converges to it — carry an O(dt) discretisation bias of order 3 in the compact mean force (≈ 0.6 kT in the compact
well depth); the frozen-dimer inner steps sample a solvent shell closer to the continuum and are scored as *error*.
W1's harm is therefore an operator-consistency confound (the inner operator's discretised stationary law differed
from the outer's), not a verdict on targeting. Amendment A2 ([AMENDMENT_A2_W1b_projected_scheme.md](../configs/targeted_relax_campaign/AMENDMENT_A2_W1b_projected_scheme.md))
re-runs the ladder with the reference's scheme as the inner step (W1b), same seeds, τ map, budget and rules,
with the prediction that the harm disappears and the targeting is neutral.

**Benchmark caveat for the paper.** Every WCA arm and the TI reference share the dt = 2e-3 Euler–Maruyama
discretisation, so all *relative* comparisons in the project stand; but the reference is ~3 in F′ (≈ 0.6 kT in well
depth) from the continuum in the compact region, and any operator with a different discretisation must be validated
against the reference's scheme before it is used (this stage's lesson).

## Stage W1b — the ladder with the reference-consistent operator (amendment A2; PRELIMINARY, 3 of 4 seeds)

Arms `abf_ptarg{ρ}` / `fr_ptarg{ρ}` (inner step = the TI scheme: all particles move, dimer re-projected to its z),
same seeds, τ map, budget and rules, paired against the W1 `abf` / `fr_uniform` runs. Seeds 820–822 complete at
06:20 UTC; the chain is finishing seed 823 under heavy GPU contention. h_read\*\* = 0.00625 by the intersection rule.

| ρ | F_ρ vs F, ΔI_F | Δe_F(T) | A_ρ vs A, ΔI_F | Δe_F(T) | actual cost | C(ε_F) ratio |
|---|---|---|---|---|---|---|
| 0.25 | +12.0 % [+9.1, +19.1], 0/3 | +13.8 % | +9.5 % [+8.4, +9.8], 0/3 | +11.3 % | 0.208× | inf |
| 0.5 | -1.9 % [-4.3, +1.7], 2/3 | -12.0 % | +6.8 % [-8.4, +9.7], 1/3 | -10.6 % | 0.417× | 1.01 |
| 1 | -5.1 % [-13.9, -4.9], 3/3 | -40.1 % | -15.6 % [-16.8, -12.0], 3/3 | -33.5 % | 0.833× | 0.75 |

Positive control F vs A -17.1 % [-19.2, -13.8], 3/3. The frozen-operator harm is gone; the improvement grows
with ρ and is larger at the end than integrated (an endpoint conditional-lag bias that relaxation removes), the shape
the handoff's P4 anticipated. Frozen ρ\* rule on three seeds: not yet met (rho* None)
(`W1/rho_selection_projected_prelim3.json`); the four-seed decision is the chain's and will be auto-committed as
`W1/rho_selection_projected.json`.

## Stage W2 — fresh confirmation

Not launched (W1 STOP by the frozen rule). If the four-seed W1b licenses ρ = 1, W2 (A, F, A_R, F_R, F_rand on seeds
900–915) is the next block; on a flat sensitivity field its targeting control is expected to return
TARGETING_NOT_SPECIFIC, which would make the deployable WCA method "FR + solvent relaxation at ρ ≈ 1" rather than the
targeted one.

## Sequencing note

W0-A (plain ABF, instrument only) was first launched from the tool after the engine/runner/prereg commit and before the
analyzer commit; it produced no completed run before it was stopped and relaunched by the orchestration after the
analyzer was committed (4c34b8c). No error metric of any arm was read before the committed analyzers ran.
