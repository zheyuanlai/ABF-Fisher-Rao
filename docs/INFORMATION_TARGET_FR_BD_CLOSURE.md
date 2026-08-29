# Closure: information-targeted Fisher–Rao birth–death as an ABF accelerator

2026-08-29. Scope of this document: what is closed, what is not, and on what evidence.
Companion: `docs/FR_CLONE_CORRELATION_PENALTY.md`.

## The conclusion

> **The information-allocation objective remains viable. Sparse standard fixed-`K` exact
> kill/clone Fisher–Rao birth–death is closed as an ABF accelerator on this benchmark.**

## The evidence chain, in order

Each link was established under a preregistration frozen before its run, and each one removes an
explanation for the next link's failure.

**1. The opportunity exists.** `G_ideal` = **0.621 (K2) / 0.606 (K3)** — the oracle finite-horizon
allocation problem, solved at the physically correct fibre horizon with fibre-consistent
difficulty `V = σ²τ_fib`, on the run's own accumulated counts. So the benchmark is not one where
there was nothing to win. (`results/fibre_horizon/stage0/`)

**2. The target is reasonable, not a delta.** `max_j π*_j` = 0.368 / 0.365, against 0.826 / 0.879
at the earlier, too-short horizon. The mass concentrates in about five information-important
cells, not one.

**3. The FR operator moves the population the right way.** `KL_post/KL_pre` = 0.987 → 0.921,
monotone in dose. The operator does what its mathematics says.

**4. The reallocation persists into the observations.** `TV(r_future, π*)` = 0.8328 → 0.8198 (K2),
0.8355 → 0.8171 (K3). New deposits did land closer to the target.

**5. Genealogy does not collapse.** Ancestor ESS 0.81–0.97, max family fraction 0.008–0.012 —
nothing like clean-v2's `ESS/K ~ 0.1`. This is not a lineage-diversity failure.

**6. And the estimator got worse anyway.**

    R_FR / R_ABF = 1.008 – 1.038,  every 95 % CI excluding 1 FROM ABOVE,
    monotone in dose, at all four doses in both mirror cells.

Against a preregistered pass condition of ≤ 0.90. **Link 6 is the proof of failure.** Links 1–5
are what make it interpretable: they eliminate "no headroom", "wrong target", "target too sharp",
"wrong horizon", "operator points the wrong way", "particles didn't go there", and "genealogy
collapsed", in that order.

## The mechanism, corrected

The natural reading of link 6 — that exact cloning costs more effective sample than the placement
gain is worth — is **wrong on this benchmark**, and we established that by making it predictive
instead of narrative.

Discounting the future budget by the exact clone-correlation loss `2ρ/(1+ρ)` and re-evaluating the
same risk functional the target was solved under predicts FR should **help** by 1–3 %, where it
measurably hurts by 1–4 %: **sign wrong in 8 of 8 (cell, dose) points, rank correlation −0.976.**

Decomposing the audit's own endpoint says why. `E[R_s] = Σ_j a_j(bias_j² + Var_j)` gives
`η_bias` = **0.958 (K2) / 0.931 (K3)**, and of the ABF → strongest-dose change, **96 % / 97 % is
bias**. The variance term — the only thing the Neyman objective and the correlation penalty
govern — carries 4–7 % of the endpoint and ~3 % of the damage.

> **The failure is that cloning relocated the finite-time estimator bias, not that it destroyed
> effective sample size.**

That is the same diagnosis as the mechanism campaign, which found `η_bias` = 0.93–0.999 on the
four IO-ABF transfer systems and 0.28–0.79 on the K-family, and validated the bias model
`b ≈ (μ₂h²/2)[f″ + 2f′ ∂_z log r̄]`. A pulse changes the cumulative exposure `r̄`; on a
bias-dominated endpoint that is the dominant thing it does.

## What is closed, and what is not

| | status |
|---|---|
| information-allocation objective (`min_π Σ a_j V_j /(C_j + Mπ_j)`) | **open / supported** — `G_ideal` ≈ 0.61 is real headroom |
| Fisher–Rao geometry as a basis for reallocation | **not closed** — nothing here tests the geometry |
| standard fixed-`K` exact kill/clone FR-BD, information-targeted, as an ABF accelerator | **CLOSED on this benchmark** |

Closed under the strongest conditions the project could construct for it: an *oracle* target (no
online estimation error), 61 % finite-horizon headroom, the physically correct horizon, a
non-degenerate target, healthy genealogy, four doses spanning 16×, and two mirror cells.

Deliberately **not** attempted, and recorded so the boundary is explicit: a `2τ` / `3τ` / `4τ`
cooldown ladder. The preregistration said Outcome B closes the direction; the previous repeated-BD
campaigns already failed independently; extending the cooldown after seeing the result would be a
moving goalpost, and §2 of the companion note now shows the correlation penalty was not the
operative term anyway, so a longer cooldown targets the wrong quantity.

**Also not started:** full adaptive Info-FR-ABF (online `τ̂`, `σ̂²`, drifting `C_j`, repeated
pulses, adaptive triggers). A single oracle pulse already gives `R_FR/R_ABF > 1`; adding
estimation noise and repetition has no mechanism by which to do better, and would make
attribution worse.

## Provenance

The preregistration (`docs/FIBRE_HORIZON_AUDIT_PREREGISTRATION.md`) was written at 05:24:24 UTC
and the pilot ran at 05:27:39 per its own receipt, but `docs/` was blanket-ignored by `.gitignore`
so it was **not committed before the run**; both documents entered git together at `3c78f1e`. The
ordering evidence is file mtimes plus the run receipt, not commit order — weaker than the previous
audit achieved (prereg `fba1b4b`, runner `6484574`, verdict later). Nothing in the protocol changed
after the run. Recorded because the weakness is in what the repository can prove, and a closure
document that hid it would be worth less than one that states it.

## The question this leaves

The earlier formulation — *can a particle realization of the Fisher–Rao reaction term avoid
destroying an independent trajectory whenever mass is created elsewhere?* — is well posed and its
answer is still unknown. But §2 of the companion note moves the binding constraint:

> Can **any** reallocation operator move the cumulative exposure toward `π*` without moving the
> finite-time estimator bias more than it moves the variance?

On a bias-dominated endpoint that constraint binds on bias-held realizations too, not only on
birth–death — which is consistent with the IO-ABF transfer campaign, where a bias-held allocation
with no cloning at all also lost, and also through the bias channel. Answering the allocation
question may require changing the *estimator* (so that the endpoint is variance-limited) before
changing the *operator*.
