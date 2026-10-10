# Experiment II: does FR need fast conditional (fibre) relaxation?

*2026-10-10. Preregistration: `SCIENTIFIC_PLAN.md` §1.2, §2, §7 and Amendment 1. Validation:
`EXP1_TIMESTEP_VALIDATION.md` (PASS for λ = 0.5, 0.25, 0.1 at h = 2.5 × 10⁻⁵).*

**Data and figures**
* Raw runs: `results/mechanism/conditional_relaxation/lam<λ>/N<N>/`. λ = 1 is hard-linked to the identical
  Experiment I α = 1 runs.
* Per-cell analysis: `.../analysis/`.
* Contrasts: `results/mechanism/synthesis/cross_cell.{json,md}`.
* Figures:
  * `figures/mechanism/conditional_relaxation/lam<λ>/...` (A–F, mech_\*, S1–S8);
  * `figures/mechanism/synthesis/mech_s5_lambda`, `mech_s5b_d2_conditional_relaxation`,
    `mech_s2_convergence_conditional_relaxation_*`.

## 1. Design (as frozen)

**Model.** The original potential V₁. Only the transverse mobility changes: dY = −λ∂_yV dt + √(2λ/β) dW, with
λ ∈ {1, 0.5, 0.25, 0.1}.
* The Gibbs law, F\* and F\*′ are unchanged. The validation confirmed the conditional law and Var(f | x) at every λ.
* N ∈ {2048, 512} (T = 40, 160), 32 seeds, both arms, the same budget and constants as Experiment I.

**Measured conditional relaxation** (validation chains, integrated ACF time):

| λ | τ_y in the wells (x = −1) | τ_f at the flank (x = −0.21) | τ_y at the gate (x = 0) |
|---|---|---|---|
| 1 | 0.97 | 0.013 | 0.0010 |
| 0.5 | 1.93 | 0.020 | 0.0020 |
| 0.25 | 3.82 | 0.028 | 0.0040 |
| 0.1 | 9.56 | 0.041 | 0.0104 |

The y times scale as 1/λ, as the frozen-x OU predicts. The flank force time scales more weakly because x moves.

## 2. Results

| λ | N | ABF Ī_F | FR Ī_F | **G(Ī_F)** | G(final e_F) | G(Ī_F′_stat) | G(Ī_TV_half) |
|---|---|---|---|---|---|---|---|
| 1 | 2048 | 0.0227 | 0.0156 | **−31.4 % [−34.7, −27.7]** 31/32 | −61.4 % | −29.9 % | −71.5 % |
| | 512 | 0.0083 | 0.0050 | **−41.0 % [−49.6, −33.4]** 32/32 | −50.5 % | −32.6 % | −72.7 % |
| 0.5 | 2048 | 0.0305 | 0.0212 | **−30.1 % [−34.4, −25.3]** 30/32 | −48.0 % [−53.8, −44.6] | −26.2 % | −69.4 % |
| | 512 | 0.0126 | 0.0089 | **−34.4 % [−43.3, −19.9]** 30/32 | −28.3 % [−39.5, −14.0] | −30.5 % | −72.4 % |
| 0.25 | 2048 | 0.0445 | 0.0338 | **−28.8 % [−31.4, −23.3]** 28/32 | −38.2 % [−40.6, −27.9] | −26.9 % | −65.3 % |
| | 512 | 0.0192 | 0.0164 | **−14.3 % [−47.3, +5.2]** 21/32 | +27.6 % [−5.2, +76.5] | −19.1 % [−47.7, +0.5] | −75.7 % |
| 0.1 | 2048 | 0.0864 | 0.0800 | **−23.2 % [−31.7, −3.4]** 23/32 | −9.1 % [−27.3, +11.6] | −21.5 % [−34.7, −4.2] | −62.8 % |
| | 512 | 0.0462 | 0.0585 | **+4.1 % [−12.1, +98.1]** 15/32 | **+126 % [+63, +366]** 8/32 | +0.1 % [−14.1, +101] | −84.1 % |

* **ABF's own error grows as λ decreases:** Ī_F × 3.8 (N 2048) and × 5.6 (N 512) from λ = 1 to 0.1.
* At N = 2048, λ ≤ 0.25, neither arm reaches the mid threshold of e_F within the budget.
* FR's first arrival in the right well is unchanged by λ: ABF u = 0.032–0.035 at N 2048, 0.0093–0.0102 at 512.

### 2.1 Preregistered contrasts

D1 compares λ = 0.1 with λ = 1 (Holm over 2). exp(Δ) > 1 means FR's relative gain is smaller at λ = 0.1.

| quantity | N = 2048 | N = 512 |
|---|---|---|
| **Ī_F (primary)** | **1.21 [1.05, 1.44], Holm p 0.033: smaller gain** | **1.98 [1.33, 3.43], Holm p 0.006: smaller gain** |
| Ī_F′_stat | 1.06 [0.93, 1.40], inconclusive | 1.54 [1.23, 3.18], Holm p 0.024: smaller gain |
| Ī_TV_half (marginal) | 1.29 [1.26, 1.33]: smaller marginal gain | **0.55 [0.53, 0.66]: larger marginal gain** |

The trend (unadjusted, descriptive):
* λ = 0.25 vs 1 on Ī_F: 1.09 [1.03, 1.15] at N 2048 and 1.11 [0.92, 2.09] at 512.
* λ = 0.5 vs 1: 1.015 (equivalent) and 1.05.

### 2.2 Mechanism readout (preregistered: D2 must show that recently cloned deposits carry a mean-force error that grows as λ decreases)

B_c is the debiased RMS deviation of class-c deposits from the bin-averaged F\*′, with a seed-bootstrap CI corrected
for the debiasing (`mech_s5b_d2_conditional_relaxation`).

| λ (N = 2048) | ABF deposits | FR, clone age [0, 0.01) | FR, [0.01, 0.1) | FR [0.01, 0.1) − ABF (matched excess, exploratory) |
|---|---|---|---|---|
| 1 | 0.012 | 0.015 | 0.018 | +0.006 [0.002, 0.009] |
| 0.5 | 0.021 | 0.031 | 0.036 | +0.016 [0.010, 0.020] |
| 0.25 | 0.040 | 0.065 | 0.085 | +0.045 [0.026, 0.057] |
| 0.1 | 0.115 | 0.305 | 0.359 | +0.244 [0.145, 0.310] |

* The preregistered change B_c(λ) − B_c(1) is positive for both young classes at every λ < 1, Holm p = 0.0036
  throughout.
* **At N = 2048 the preregistered criterion is met.**
* At N = 512 the same direction holds: the young-class excess over ABF is 0.0014, 0.0075, 0.053 and 0.36. The
  preregistered changes are **not** significant there (Holm p 0.33–0.64): few young deposits, and wide CIs whose lower
  ends reach 0.
* Young clones are a small share of the data: 1.3–1.6 % of gate deposits at N 2048 and 2.3–2.4 % at N 512.
* ABF's own deposits also become more biased as λ falls. This is the fibre lag that ABF suffers without FR.

## 3. Interpretation (preregistered map, §7)

* **"The marginal gain persists while the F/F′ gain shrinks as λ → 0.1" → supports a conditional-relaxation
  limitation.**
  * At N = 512 this is exactly what happens. The free-energy gain goes from −41 % to +4 % (the final error is
    +126 % worse with FR, 24 of 32 seeds), while the marginal gain *increases*, from −73 % to −84 %.
  * At N = 2048 both shrink, F's more than the marginal's in relative terms. FR still helps the integrated error
    there (−23 %) but no longer the endpoint.
* **"Proven" only if D2 shows the clone error growing with decreasing λ.** At N = 2048 it does, significantly at every
  λ. At N = 512 the direction is the same but not significant. So the fibre-lag mechanism is **established at
  N = 2048 by the preregistered criterion** and **consistent but not established at N = 512**.
* **Discovery vs equilibration.** First arrivals and the marginal establishment that FR provides are unchanged or
  better at small λ. What degrades is the force information carried by the walkers, ABF's and FR's clones' alike, but
  clones' most. Slower discovery does not explain the loss.
* **What the data do not separate.**
  * Young clones are ≤ 2.4 % of gate deposits. Whether their bias alone explains FR's endpoint loss at N = 512, or FR
    also degrades the remaining deposits (for example by over-populating the gate entry side, where lag is largest),
    is not tested here.
  * The magnitude attribution is exploratory.

**Negative result preserved.** FR is harmful at λ = 0.1, N = 512 on the endpoint, and neutral on the integrated
error, while giving the largest marginal improvement of the λ series.
