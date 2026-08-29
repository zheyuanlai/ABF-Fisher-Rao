# The clone-correlation penalty: what is exact, and what it does not explain

Companion to `docs/INFORMATION_TARGET_FR_BD_CLOSURE.md`. Written 2026-08-29 from the
fibre-horizon audit data (`results/fibre_horizon/`).

This note has two halves and they point in different directions. The first is an exact
effective-sample-size identity for exact-copy branching, which is worth keeping. The second is
the measurement showing that on this benchmark **that identity is not the mechanism of the
observed failure** — a conclusion reached by trying to make it predictive and finding it
predicts the wrong sign.

---

## 1. The identity

For trajectories `Y_1..Y_K` with common marginal variance `σ²` and correlation matrix `R`,

    Var( (1/K) Σ_i Y_i ) = (σ²/K²) · 1ᵀ R 1

so the natural effective sample size is

    K_eff = K² / (1ᵀ R 1).                                            (1)

`K_eff` is the object of interest; ancestor ESS is only a proxy for it, and a poor one after a
cooldown (an ancestor label is permanent, correlation is not).

For a single paired kill/clone event: two trajectories that were independent before the event
become, after evolving for time `t`, a sibling pair with observable correlation `ρ(t)`. Their
correlation matrix is `[[1, ρ], [ρ, 1]]`, so `1ᵀR1 = 2 + 2ρ` and

    K_eff,pair(t) = 4 / (2 + 2ρ(t)) = 2 / (1 + ρ(t)).                 (2)

Relative to the two independent trajectories that existed before the event, the exact
effective-sample loss is

    ΔK_eff(t) = 2 − 2/(1 + ρ(t)) = 2ρ(t) / (1 + ρ(t)).                (3)

Two limits, both immediate:

    ρ(0) = 1   ⟹   ΔK_eff(0) = 1
    ρ(t) → 0   ⟹   ΔK_eff(t) → 0

**At the instant of branching an exact copy converts two independent information sources into
one independent-equivalent source, and no choice of target can prevent it**, because the loss
depends only on `ρ(0) = 1`, which is a property of exact copying. The penalty then decays as the
siblings decorrelate.

### Scope — what (3) does and does not license

**Licensed:** the finite-time correlation penalty is *intrinsic to exact-copy fixed-`K`
kill/clone realization*. One independent trajectory is destroyed and its slot is filled by a
duplicate; at `t = 0` that is a full unit of `K_eff`, unconditionally.

**Not licensed, and not established anywhere in this project:** that *all* Fisher–Rao particle
realizations must pay it. Untouched by this analysis are variable-`K` branching, weighted FR
particle systems, continuous mass/weight evolution with delayed materialization, and any other
particle approximation of the same FR PDE that does not replace an independent trajectory with an
exact clone. Each would need its own `1ᵀR1`.

### Measured decay

On the fibre-horizon benchmark, fitting `ρ(t) = exp(−t/τ_ρ)` over FR-born pairs:

| cell | arm | `τ_ρ` (steps) | `H_fib` (steps) | `τ_ρ/H_fib` | `ρ` at `H_fib` | `ΔK_eff` |
|---|---|---:|---:|---:|---:|---:|
| K2 | p0.1 | 2768 | 2635 | 1.05 | 0.416 | 0.588 |
| K2 | p0.2 | 3415 | 2635 | 1.30 | 0.412 | 0.583 |
| K3 | p0.1 | 2775 | 2325 | 1.19 | 0.487 | 0.655 |
| K3 | p0.2 | 3344 | 2325 | 1.44 | 0.507 | 0.673 |

`τ_ρ ≈ H_fib`, so **one decorrelation time leaves `ρ ≈ e⁻¹ ≈ 0.37` by definition.** "Wait one
τ" is not "the clones are independent"; `ρ < 0.05` needs ≈ 3τ. That is a genuine correction to an
assumption this project carried implicitly, and it survives everything below.

---

## 2. The measurement that says this is not the operative mechanism here

The tempting next step is to treat (3) as the explanation of the audit's failure. It is not, and
the way to find that out is to make it predictive rather than narrative.

### 2a. The break-even test

Put both effects into the same units — the finite-horizon risk functional the target was solved
under — by discounting the future budget for the correlation loss:

    R(arm) = Σ_j a_j V_j / (C_j + M_eff · r_j),
    M_eff  = M · (1 − n_events · 2ρ/(1+ρ) / K),      r_j = realised future allocation

| cell | dose | events | `ρ(H)` | `M_eff/M` | **predicted** ratio | **measured** ratio |
|---|---|---:|---:|---:|---:|---:|
| K2 | 0.02 | 3.5 | 0.214 | 0.9952 | 0.9906 | 1.0088 |
| K2 | 0.05 | 8.0 | 0.394 | 0.9823 | 0.9848 | 1.0145 |
| K2 | 0.10 | 14.5 | 0.416 | 0.9667 | 0.9798 | 1.0264 |
| K2 | 0.20 | 31.0 | 0.412 | 0.9294 | 0.9826 | 1.0246 |
| K3 | 0.02 | 3.5 | 0.482 | 0.9911 | 0.9924 | 1.0089 |
| K3 | 0.05 | 8.5 | 0.340 | 0.9831 | 0.9894 | 1.0134 |
| K3 | 0.10 | 14.5 | 0.487 | 0.9629 | 0.9837 | 1.0201 |
| K3 | 0.20 | 31.5 | 0.507 | 0.9172 | 0.9725 | 1.0379 |

**Sign wrong in 8 of 8.** Even after paying the full correlation penalty, the risk model still
says FR should *help* by 1–3 %; the measurement says it *hurts* by 1–4 %. Rank correlation
between predicted and measured is **−0.976** — the model does not merely miss the magnitude, it
orders the doses backwards. Solving for the break-even correlation at the strongest dose gives
`ρ* = 0.028` (K2), i.e. the model would need the clones to be *essentially independent* before it
reproduced the observed harm — which is the opposite of the proposed explanation.

### 2b. Why: the endpoint is bias-dominated, and the change is bias

Decomposing the audit's own endpoint across its 8 seeds, `E[R_s] = Σ_j a_j (bias_j² + Var_j)`:

| cell | arm | `E[R_s]` | bias part | var part | `η_bias` |
|---|---|---:|---:|---:|---:|
| K2 | abf | 0.010940 | 0.010482 | 4.574e-4 | **0.958** |
| K2 | p0.2 | 0.011224 | 0.010755 | 4.688e-4 | 0.958 |
| K3 | abf | 0.010948 | 0.010188 | 7.603e-4 | **0.931** |
| K3 | p0.2 | 0.011363 | 0.010591 | 7.721e-4 | 0.932 |

And of the ABF → strongest-dose change:

| cell | Δ total | Δ bias part | Δ var part | bias share of the change |
|---|---:|---:|---:|---:|
| K2 | +2.841e-4 | +2.727e-4 | +1.139e-5 | **96 %** |
| K3 | +4.148e-4 | +4.030e-4 | +1.174e-5 | **97 %** |

The variance part — the *only* term equations (1)–(3) govern — carries 4–7 % of the endpoint and
about 3 % of the damage, and moves by 0.4–5.7 % (K3 at p0.1 it is 0.996, i.e. slightly *better*).

**So the harm FR does is 96–97 % in the finite-time bias.** The clone-correlation penalty is real,
exact, and almost irrelevant to this outcome.

### 2c. This is the same finding as the mechanism campaign

`η_bias` = 0.93–0.96 here sits alongside 0.28–0.79 on the K-family Phase-0 re-audit and
0.93–0.999 on the four IO-ABF transfer systems and the τ-benchmark. Every failure this project
has diagnosed at the mechanism level has turned out to live in the finite-time estimator bias,
not in the asymptotic variance the theory optimises. Cloning relocates that bias — as a change in
the cumulative exposure `C_t`, which is exactly the argument of the bias model
`b ≈ (μ₂h²/2)[f″ + 2f′ ∂_z log r̄]` validated in the mechanism campaign — and here it relocates
it the wrong way.

---

## 3. What to carry forward

* **Keep (1)–(3).** They are exact, they correct a real misconception (`one τ` ⇏ independent), and
  they will bind on any future exact-copy scheme. `K_eff`, not ancestor ESS, is the right object.
* **Do not use (3) to explain the fibre-horizon result.** It predicts the wrong sign. Saying "the
  correlation cost exceeded the placement gain" is a narrative that the arithmetic refuses.
* **Do not compare a TV improvement to an effective-particle loss.** Different units; the earlier
  "3–8 % of the population to buy 1–2 % of placement" framing is retired by §2a, which does the
  comparison properly and gets the opposite answer.
* **The open theoretical question is unchanged in form but changed in target.** It is no longer
  "can a particle realization avoid the correlation penalty" alone, but: *can any reallocation
  operator move `C_t` toward `π*` without moving the finite-time bias more than it moves the
  variance?* On a bias-dominated endpoint that is the binding constraint, and it applies to
  bias-held realizations as much as to birth–death.
