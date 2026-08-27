# IO-ABF transfer campaign — preregistration

**Frozen 2026-08-27, before any scientific run of this campaign.**
Branch `q-r-decoupling`. Everything in this document is fixed; nothing in it may
be changed after the first candidate run starts.

## 1. The question

The q-r campaign measured, on the kappa family, that allocating replicas by
estimator risk and holding that allocation with the bias accelerates ABF by
1.55–1.87×, and that every birth–death arm loses. That is one instrument. This
campaign asks whether it **transfers**:

> When the cost of obtaining mean-force information is strongly heterogeneous
> across reaction-coordinate bins, does r*(z) ∝ sqrt(a(z) Γ(z)) significantly
> accelerate free-energy convergence on the project's existing benchmarks?

Four preregistered questions:

| Q  | Comparison        | Asks                                                        |
|----|-------------------|-------------------------------------------------------------|
| Q1 | A6b vs A0         | does estimator-risk allocation accelerate ABF?               |
| Q2 | A6c vs A6b        | what does keeping the physical mass q cost in speed?         |
| Q3 | R_Γ vs gain       | does the gain appear where difficulty is heterogeneous?      |
| Q4 | easy control      | is IO-ABF neutral where there is no difficulty heterogeneity?|

## 2. The three arms

**A0 — standard ABF.** Uniform reaction-coordinate allocation, `B_t = Â_t`.
Implemented as `eb.IO_A0`, which carries `use_fr=False, target_mode="none"` and
therefore runs the accepted `abf` code path exactly (gate G0.1).

**A6b — IO-ABF.** Per allocation cell `I_j`,

    Γ̂_j = σ̂_j² τ̂_j ,  g_j = a_j Γ̂_j ,  r̃_j = sqrt(g_j) / Σ_k sqrt(g_k)
    r*_j = 0.75 r̃_j + 0.25 / J                    (shared floor, FLOOR_FRACTION = 0.25)
    B_t(z) = Â_t(z) + β⁻¹ log r*_t(z)

No birth, no death, no clone, no resampling, no weight. The occupancy `p_t(z) →
r*_t(z)` is the *stationary* distribution of the biased dynamics.

**A6c — FR-constrained IO-ABF.** Same mechanism, plus the genuine Fisher–Rao
cell mass `M⁺ ∝ M^(1-θ) q^θ` at **θ = 1** with `q_j ∝ exp(-β Â_j)`, and

    min_r Σ_j a_j Γ̂_j / r_j    s.t.   ESS_M/K = [Σ_j M_j²/r_j]⁻¹ ≥ ρ = 0.5
    ⇒  r*_j ∝ sqrt(a_j Γ̂_j + λ M_j²),   λ by bisection, never tuned

## 3. Γ̂ is frozen

    Γ̂_j = σ̂_j² τ̂_j

`σ̂_j²` is the instantaneous spread of the local force residual `f(X_i) −
Â'(ξ(X_i))` — against `F̂'` at each replica's own position, never a raw
within-bin variance, so the bin's own slope in `A'` is not charged to noise.
`τ̂_j` is the bias-corrected lag-1 AR(1) estimate `τ̂ = −Δt_obs / log φ̂` with
Kendall's `φ + (1+3φ)/n` correction, already implemented on this branch. Cells
that cannot be estimated are **never** assigned 0; they take the pooled/median
fallback and keep the frozen shrinkage 0.3.

**No estimator parameter may be changed after a candidate run.**

## 4. Grid and allocation cells are separate

The ABF grid of each benchmark is kept exactly as accepted. Equal-width
allocation cells are overlaid on top of it, with

    J_alloc = min(32, floor(K / 8))

so a cell holds ≳8 walkers under uniform occupancy. If `K / J < 8` the **cell
count** gives way, never the estimator.

## 5. Cadence — rule R-CAD and rule R-OBS

Both are structural functions of the run, not per-system knobs:

    opportunity_every = round(0.60 · n_steps / 48)     # 48 refreshes in [0.2T, 0.8T]
    obs_every         = round(0.15 · n_steps / 600)    # history spans 15% of the run
    burn-in 0.20 T,  allocation stops at 0.80 T,  history capacity 600

`obs_every` may be overridden **once per system, before any candidate run**, by
rule **R-OBS**: an A0-only probe measures the per-cell mean-force
autocorrelation and sets the sampling interval near `τ_med / 2`. The AR(1) fit
fails in opposite ways on either side — `φ→0` when the interval is long against
τ (the cell reads as unresolved) and `φ→1` when it is short (the fitted φ
crosses 1 on noise) — so the interval is a *measurement design* decision, of the
same kind as freezing ε from A0. It never reads a candidate arm and never reads
a reference free energy. The value used is recorded per system.

## 6. Phase 0 — engineering gates (hard stops)

| Gate | Check | Status |
|------|-------|--------|
| G0.1 | A6b with `r* = uniform` is trajectory- and estimator-identical to A0; and an A0 row inside an IO batch reproduces accepted `eb.ABF` bit for bit | PASS |
| G0.2 | nothing on the allocation path executes `F_ref`, `q_ref`, barrier or landscape names (tokenised source, docstrings stripped) | PASS |
| G0.3 | no clone/kill/resample path; every public method returns a float field, never indices; ancestor ESS stays at N end-to-end | PASS |
| G0.4 | `r*` normalised, strictly positive, floor exactly 0.25/J | PASS |
| G0.5 | constant Γ gives `r* ∝ sqrt(a)` | PASS |
| G0.6 | synthetic AR(1) force stream recovers σ², τ and Γ ordering (Spearman > 0.85, valid-τ ≥ 80 %) | PASS |
| G0.7 | A6c satisfies `ESS_M/K ≥ 0.5` **on the allocation it applies** | PASS after fix |
| G0.8 | particle weights never enter the ABF accumulator; the allocation force is added to the drift only | PASS |
| G0.9 | with no valid τ the target falls back to `sqrt(a)`, not to an extreme allocation | PASS |
| REG  | the 160 existing q-r / kappa / clean-v2 regression tests still pass | PASS |

**G0.7 found a real defect and it is recorded here rather than quietly fixed.**
The shared floor was applied *after* the ESS bisection, so the applied target
missed the constraint it was solved for — measured at ESS 0.420 against a stated
0.500 — and the number the run reported was the pre-floor target's, not the one
it used. `allocation.r_ess_constrained` now takes `floor_fraction` and solves the
floored problem. The unfloored path is left intact, so the closed q-r campaign's
arithmetic is unchanged; **its reported "mass ESS 0.500" is the unfloored
target's and should be read that way.**

## 7. Systems, in order

1. **Entropic bottleneck β = 4** — repo-classified ABF-sufficient. *Control.*
2. **Entropic bottleneck β = 8** — repo-classified establishment-limited. *Candidate.*
3. **Entropic gateway** — accepted production sampler `gateway_core.simulate_batch`,
   not the Gate-0 audit's simplified re-implementation.
4. **WCA dimer** — only after its high-precision reference is revalidated
   (physical parameters, CV, grid, evaluation mask, integration convention). On
   a mismatch: A0 diagnostics only, **no speedup may be reported.**

**Deca-alanine is not a candidate.** The repository has already classified it as
conditional-equilibration-limited (mean-force error up to 61 % at fixed ξ);
IO-ABF is an allocation correction, not an orthogonal-ergodicity repair.
Alanine is likewise a negative control at best, and its periodic CV needs a
correct periodic leverage operator that this campaign does not build.

## 8. Design per system

* **Phase 1A** — 16 A0-only calibration seeds. Records σ², τ, Γ, e_A(t), the
  robust difficulty spread `R_Γ = Q₀.₉(Γ)/Q₀.₁(Γ)`, and the valid-τ fraction.
  **Reliability gate: ≥ 80 % of scored allocation cells have a valid τ̂.** Below
  that the system is marked `Gamma unresolved`; candidate runs still happen, but
  a candidate failure may not be attributed to the theory.
* **Phase 1B** — 8 paired pilot seeds, arms A0/A6b/A6c. Checks crashes, NaN, the
  estimator working, occupancy sane, no leakage, no gross pathology.
  **The pilot may not change the algorithm.**
* **Phase 1C** — 32 **fresh** paired confirmatory seeds per arm.

Identical across arms: initial states, RNG seed pairing, walker count, force
evaluations, timestep, ABF kernel, ABF update cadence, horizon, evaluation
cadence. Only the algorithm definition differs. Arms are **columns of one batch**
so they share the Langevin noise stream exactly.

## 9. Thresholds, frozen from A0 calibration

    ε₁ = median e_A(0.4 T)      ε₂ = median e_A(0.6 T)   (stringent)
    τ_ε = inf{ t : e_A(t), e_A(t+Δ), e_A(t+2Δ) ≤ ε }     (three consecutive frames)

Frozen once. **May not be changed afterwards.**

## 10. Primary endpoint

    S_ε = E[min(τ_ε^A0, T)] / E[min(τ_ε^method, T)]        S > 1 = faster

Paired bootstrap CIs, 10 000 resamples, seed 20260827.

## 11. Positive criterion — A6b vs A0, at ε₂

A6b is a **positive** only if all of:

1. `S_ε₂ ≥ 1.15`;
2. paired 95 % CI lower bound `> 1`;
3. `P(τ_ε₂ ≤ T)` for A6b is not worse than A0 by more than 5 percentage points
   (censoring flatters a candidate, so it can only block a positive — it may not
   be used to block a negative);
4. `e_A(T)^A6b ≤ 1.10 e_A(T)^A0` **and** `e_A,full(T)^A6b ≤ 1.10 e_A,full(T)^A0`.

## 12. A6c is a Pareto point, not a pass/fail

Reported separately: speed `S_ε^A6c/A0`, and mass fidelity `ESS_M/K` with the
headline `ρ = 0.5`. The retention `R_retain = (S_A6c − 1)/(S_A6b − 1)` is
reported as a number, **not** as a gate. There is no "must retain 80 %" rule.

## 13. Required outputs per system

Decomposition table: Q₁₀/Q₉₀/ratio for σ², τ and Γ; valid-τ fraction;
Spearman(Γ early, Γ late). Six figures: convergence; time-to-accuracy; σ²(z),
τ(z), Γ(z); a(z), a(z)Γ(z), r*(z); r* vs realised occupancy; and for A6c q(z),
r*_A6b(z), r*_A6c(z) with ESS_M/K and λ annotated. Both `e_A,primary(t)` and
`e_A,full(t)` are reported everywhere.

## 14. Prohibited tonight

Changing `FLOOR_FRACTION = 0.25`; sweeping ρ (headline A6c is ρ = 0.5 only);
sweeping θ (stays 1); reintroducing birth–death; changing the Γ estimator on
pilot results; changing the evaluation window on results; using `A_ref` to
decide `a`, `r*`, warm-up or which bin is hard; picking the best allocation
cadence from results; extending A6b's horizon because it lost; silently deleting
a blown seed; explaining deca's failure as an IO-ABF result either way.

A single system going negative is **recorded and the campaign continues to the
next preregistered system.** No algorithm change follows a negative.
