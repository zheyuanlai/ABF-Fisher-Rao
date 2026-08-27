# q-r decoupling: preregistration

**Branch** `q-r-decoupling`. **Opened** 2026-08-26, after clean-v2 Stage 2 closed
0/9 (`docs/CLEAN_V2_STAGE2_RESULT.md`). **Status:** Stage 0 written and passing;
Stage 1 not started.

## The hypothesis

> The physical mass density `q` and the information-optimal replica density `r`
> are different objects, and equal-weight birth--death failed because it forced
> them to be the same one.

`q` is fixed for this campaign at the deployable physical target
`q_t(z) ∝ exp(-β Â_t(z))`. The campaign is about `r`.

### Why the mass channel cannot be what failed

ABF's estimator is `F̂'(bin) = Σf / count`. The Fisher--Rao score depends on `z`
alone, so within a bin the operator kills and clones *exchangeably*: it does not
select on the fibre coordinate. The conditional law on each fibre is therefore
preserved **in expectation**, and a marginal reallocation reaches the estimator
only through

1. **`r`** -- where replicas propagate afterwards, and
2. **genealogy** -- how independent those replicas are.

This is weaker than pathwise invariance, and the difference matters: a clone is
unbiased but not free, and the deviation it introduces is precisely the
genealogy term. What it rules out is a *third* channel. In an equal-weight
scheme the choice of FR target was never a statement about probability mass at
all -- it was a replica-allocation policy and nothing else. As a policy,
`e^{-βF}` is anti-Neyman: it depletes the high-`F` cells relative to the roughly
uniform counts ABF's own flattening produces.

### What that retrodicts

| Recorded result | `r`-channel reading |
| --- | --- |
| 0/54, 0/11, 0/9 across three campaigns | `r` pushed from ~uniform to `e^{-βF}`, genealogy paid for it |
| Stage 0: damage grows with target faithfulness (1.095 → 1.279 → 1.408) | more faithful target = `r` further from uniform |
| Oracle targets never rescue (pulse-v2 1.379; R15 0/8; v3 KL regrows 94-99%) | the oracle fixes `q`; the `r`-policy is unchanged |
| v3's oracle flip (2.09× → 0.49×) | also `r`: the estimated target closes a feedback loop the oracle does not |
| The gateway positive (-12.12%) | establishment-limited = large local `Γ`; FR accidentally did a Neyman correction |
| WCA: count balancing ties mFR | count balancing *is* the uniform-`r` policy |

The last row is measured **at a different endpoint and on a different run tree**
(`final L²(F')`, not time-to-accuracy). It is a motivating observation, not a
settled fact, and A3 is re-run here rather than cited.

**The clean-v2 early transient (~18%) is not evidence for this pivot.** Stage 0
attributed the analogous early gain to edge evacuation that fully relaxes back.
The gateway result is the only load-bearing positive.

## The two corrections that changed the design

### 1. The endpoint is `F`, not `F'`, so bins do not have equal influence

`F̂ = cumulative_trapezoid(F̂')` and the metric centres on the evaluation mask.
Writing `F = H f`, `C` for centring and `W` for the mask quadrature, and taking
`Cov(f̂)` diagonal,

```
E[e_F²] = Σ_j a_j Var[f̂_j],    a_j = (1/L) [Hᵀ Cᵀ W C H]_jj
```

so the allocation optimum is `r* ∝ sqrt(a_j Γ_j)`, not `sqrt(Γ_j)`.

**Measured on the clean-v2 grid** (401 points on [-3,3], mask [-2.5,2.5], J=32):
`a` spans **14×** across cells, is exactly zero in the 4 cells outside the mask,
and gives the outer 8 cells 3.9% of the replicas against 25% under uniform.

This is not a footnote, and it breaks the tie prediction as originally stated.
`r ∝ sqrt(a)` already differs from count balancing by an order of magnitude at
**flat** `Γ`, so "A4 ≈ A3 when `Γ` is homogeneous" is false before any dynamics
run. The arm is therefore split:

- **A4a** `r ∝ sqrt(a)` -- static leverage only, no online estimation at all
- **A4b** `r ∝ sqrt(a Γ̂)` -- leverage × measured difficulty

and the tie prediction moves to **A4b ≈ A4a where `Γ` is flat**, which is a real
prediction about the estimator rather than an artefact of the metric's geometry.
If A4a captures most of the gain, that is a simpler and more deployable result
than the one we set out to find, and it must be reported as such.

**`a_j` requires the evaluation mask to be geometric.** Under a thermal scope
such as R12 the mask is a function of `F_ref`, which would make `a_j` -- and
therefore the entire allocation -- an oracle quantity. Dropping R12 as primary
is a structural requirement, not a preference. Primary scope: the fixed
geometric window `[-2.5, 2.5]`; R12 continues to be reported as legacy secondary.

**Declared asymmetry:** A3/A4/A5 are told the evaluation window; A0 has no
allocation mechanism and cannot use it. Part of any A4-over-A0 margin is that
extra input. The claims that carry the science are the ones between mask-aware
arms: **A4a vs A3** (leverage) and **A4b vs A4a** (information).

**Known confound, pre-registered:** `sqrt(a)` de-allocates the domain edges, and
the v2 Stage 0 audit attributed that campaign's transient gain to edge
evacuation that fully relaxed back. Same action, different reason. Any A4a gain
must be shown to survive to the stringent threshold, and the mirror test below
is what separates the two mechanisms.

### 2. The ESS constraint is not a performance argument

A5 minimises the same risk as A4b under an added constraint, so its predicted
risk is **weakly worse** -- A5 cannot beat A4b on the free-energy endpoint except
through model error or noise. This is asserted as a gate, not hoped for.

What the constraint buys is fidelity of the physical-mass representation, which
is a *separate reported endpoint* (`marginal_l2_to_physical_ref`). A5's role is
therefore a Pareto point -- free-energy speed against physical-marginal fidelity
-- and the frontier is the deliverable, not a pass/fail. `ρ = 0.5` is the single
preregistered headline; `ρ ∈ {0.3, 0.7}` are reported as the frontier and may
not be used to select a winner.

The family also explains the old algorithm: `r ∝ sqrt(g + λ q²)` runs from A4b at
`λ = 0` to `r = q` as `λ → ∞`. **clean-v2 was this family at `ρ = 1`** -- the
corner that demands every replica carry equal mass. Gate 0C checks that corner.

## The algorithm (MI-FR-ABF)

Per step: ordinary ABF propagation, ordinary **unweighted** accumulation, update
`Â_t`. Per opportunity (every 500 steps):

1. `q̂_j ∝ exp(-β Â_j)` on cells; FR step `M_j ← M_j^(1-θ) q̂_j^θ`, `θ = 1`.
2. `Γ̂_j` by batch means over the accumulator's **eligible** stream.
3. `g_j = Σ_{grid ∈ j} a Γ̂`.
4. `r*` per arm; mix with the shared floor (0.25 × uniform, identical for every arm).
5. Resample **only if** predicted risk falls by ≥ 10%.
6. Balanced offspring within cells; clones held out for `Δ_j = (τ̂_j/2) log(D/(ε n))`.
7. Project mass: `w_i = M_j / n_j`.

Weights never enter the accumulator. `θ = 1` is not swept: the mass cannot move
the primary endpoint, and a knob that cannot move the headline should not be tuned.

### Two feedback loops, and what closes them

- **`Γ̂` → allocation → clones → `Γ̂`.** Siblings inflate measured variance; if
  that fed the allocator, a cell receiving clones would measure as harder and
  receive more. Closed by computing `Γ̂` from the post-hold eligible stream only,
  which bounds the inflation by the same `ε_gene = 0.1` that sets the hold.
- **`q̂` → allocation → `F̂` → `q̂`.** This is the v3 oracle-flip shape. **A4a and
  A4b are structurally immune** (no `q̂` in the allocation); A5 partially
  reintroduces it through `λq²`. Reported, not assumed away.

### Blind by construction

The runner may read `X_i, ξ(X_i), f_i, Â_t, Γ̂_t, n_j, lineage`, and the
a-priori evaluation window. It may not read `F_ref`, `p_ref`, barrier or basin
definitions, `R12`, `κ(z)`, `T_hit`, or `T_turnover`; and it may not switch
reallocation off because a curve has started to worsen. Gate 0A enforces this on
the import graph.

## Stage 0 -- engineering gates (`tests/test_qr_decoupling.py`, 111 passing)

| Gate | Property |
| --- | --- |
| 0A | allocation modules cannot reach the reference, transitively; mask is geometric |
| 0B | `Σ_{i∈j} w_i = M_j` exactly; weights normalise |
| 0C | fibre constancy after a long FR history; `ρ=1` recovers `r=q`; A5 risk ≥ A4b risk |
| 0E | balanced offspring is the *exact* minimiser of `Σ m_a(m_a-1)` (enumerated) |
| 0F | resampling never moves a configuration between cells; cell-conditional mean preserved |
| 0G | empty cells stay empty; assigning one is an error, not a silent fix |
| 0H | leverage vanishes off-mask, is translation-invariant, and matches a Monte-Carlo risk to 5% |
| 0I | κ ≤ 1 in every cell; the implemented integrator samples `exp(-βV)` at fixed `x`; K3 relocates difficulty rather than rescaling it |

Not yet written: **0D**, the mass-only identity `A2 ≡ A0`, which needs the engine
wiring. Run identity pairs **in one process** -- the engine is not bitwise
reproducible across processes.

## Stage 1 -- calibration, and two estimator gates that can stop the campaign

Backbone frozen from clean-v2: `β=4, K=256, dt=0.002, n_steps=50000, h=0.05,
update_every=10, eval_every=500`. Cells `J=32` equal width. Seeds 5100-5115
(16 ABF-only per κ-cell). Thresholds `ε₁, ε₂` = median ABF `e_F` at `0.4T, 0.6T`;
`τ_ε` needs 3 consecutive frames; every run goes to full `T`.

**1B -- is `Γ̂` usable?** This gate has already fired once, before any
scientific run, and it changed the estimator.

Measured on the campaign's own potential (fixed `x`, hidden channel isolated):
`τ ≈ 0.19` time units at `κ=1`, `1.12` at `κ=1/4`, `4.73` at `κ=1/16`; in-run
`x`-motion turns a cell over ~4× faster, so the slow end sits near `1.2` time
units. Batch means need a block ≫ `τ` **and** `B` of them: `B=10 × 10τ ≈ 60000`
steps against a **50000-step run**. The frozen `W=5000/B=10` budget was short by
more than an order of magnitude, and short in the direction that **hides**
difficulty rather than inventing it.

Simulated at exactly that budget (`τ ∈ {30,120,480}` steps, 8 replicas/cell,
3000-step window), median over 12 realisations:

| estimator | recovers, of a true 16× spread |
| --- | --- |
| batch means, `B=10` | **2.7×** |
| `Γ = σ² τ`, decomposed | **13.1×** |

So the estimator is **`Γ̂_j = σ̂²_j τ̂_j`**. The two factors need different amounts
of data and the decomposition exploits that: `σ̂²` is an instantaneous spread
across the replicas in a cell (no window, error does not grow with `τ`, residuals
taken against `F̂'` at each replica's own position so a steep mean force is not
charged to noise); `τ̂` is a shape parameter fitted by **bias-corrected lag-1
AR(1) regression**, which for a relaxing hidden coordinate is the MLE. Fitting
`log ρ_k` against lag was tried and rejected: the sample autocorrelation carries
a downward bias ≈ `2τ/n` that is roughly constant in the lag, which steepens the
fitted decay and reports the hard cell as easy (16× came back as 5.8×).

Only *ratios* of `Γ̂` reach the allocator, so the factor of 2 in the asymptotic
variance is deliberately not carried. A cell that cannot be fitted returns NaN
and is filled from the pooled median -- never a small number, because "unmeasured"
and "easy" must not look alike. Single-cell estimates remain noisy at `n/τ ≈ 6`
(±30%); the frozen shrinkage of 0.3 damps it, and it degrades toward *uniform*
allocation rather than toward an actively wrong one.

Still to do in 1B: validate `Γ̂` against `Γ_ref` from 4 seeds × 4T on the real
engine -- rank correlation, multiplicative error, top-cell overlap. If `Γ̂` cannot
see the K2/K3 difficulty inversion there, **do not run A4b/A5.** That is an
estimator failure, not a method failure, and fixing it is Stage 1 work.

**1C -- is the diagonal `Cov(f̂)` assumption sound?** The ABF estimator smooths
with `h=0.05` against a cell width of 0.1875, so cross-cell correlation should be
modest -- but report `‖Σ - diag Σ‖_F / ‖Σ‖_F`. If it is large, the theory owes a
covariance-aware allocation and `r* ∝ sqrt(aΓ)` may not be quoted as exact.

## Amendment 1 -- the mechanism changed, so the arms did

`r*` says where physical trajectories should be. Birth--death was one way to put
them there; it is not the natural one. Under a bias `A(z)` the marginal is
`p_A ∝ exp(-β(F-A))`, so

    A_t(z) = F̂_t(z) + β⁻¹ log r*_t(z)      ⟹      p_t(z) → r*_t(z)

once `F̂_t ≈ F`. The allocation becomes the **stationary** state of the dynamics
instead of something that has to be re-imposed against them, and because the
added term depends on `z` alone the fibre conditional is unchanged --
`μ_A(dx | ξ=z) = μ(dx | ξ=z)` -- so the ordinary unweighted ABF mean-force
estimator stays justified. No cloning, no genealogy, no leak.

Three measurements forced this, each correcting the previous guess:

1. The occupancy gate stopped the arms fighting count noise, but A4b still fired
   at 22 of 24 opportunities and drove ancestor ESS to **8 of 256**.
2. The allocation is **not** erased between opportunities -- imposed under the
   exact bias it recovers only **36%** of the way back to equilibrium in 500
   steps -- so leakage is not the explanation.
3. Opportunity-to-opportunity drift in `r*` is **0.000** TV for A4a and 6% of
   the target-occupancy gap for A4b, while the gap holds at 0.18-0.29. **The
   target is stable.** The dynamics pull the population off it continuously and
   birth--death buys a fraction back each time, at a genealogy cost it never
   recovers.

Exploratory, one seed, 20k steps, final `e_F` rather than time-to-accuracy.
**Directional, and explicitly not a basis for selecting an arm:**

| arm | mechanism | K0 | K2 | resamplings (K0) | ancestor ESS (K0) |
| --- | --- | --- | --- | --- | --- |
| A0 | none | 0.1238 | 0.2307 | 0 | 256 |
| A3 | birth--death | 0.1502 | 0.2469 | 1 | 209 |
| A4a | birth--death | 0.2102 | 0.2501 | 10 | 71 |
| **A6a** | **bias** | **0.0789** | **0.1894** | 0 | 256 |
| A4b | birth--death | 0.2675 | 0.2678 | 21 | 8 |
| **A6b** | **bias** | **0.0527** | **0.0964** | 0 | 256 |

### What this is NOT

**A6a and A6b are not Fisher--Rao particle reallocation.** They are
information-optimal adaptive biasing: there is no reaction term, no birth or
death, and no `q_phys`. The exploratory run makes the point itself -- A6b's mass
ESS is **0.000** in K2, so the arm that wins does not represent the physical
mass at all. Saying "we fixed Fisher--Rao by replacing birth--death with a bias"
would be conceptually false.

### A6c -- where Fisher--Rao earns its place

Keep a genuine FR mass layer on cells with the exact finite-time step
`M⁺ ∝ M^(1-θ) q^θ`, `q_t ∝ exp(-β F̂_t)`. Compute `g_j = a_j Γ̂_j`
independently. Then require the replica population to still represent that mass:

    min_r  Σ_j g_j / r_j      s.t.   Σ_j r_j = 1,   [Σ_j (M⁺_j)² / r_j]⁻¹ ≥ ρ

    ⟹   r*_j ∝ sqrt( a_j Γ̂_j + λ_t (M⁺_j)² )

realised through the same bias. Three objects, three questions:

    Fisher--Rao   →  M_t     what probability mass each region should carry
    Neyman        →  r*_t    where physical trajectories should spend effort
    adaptive bias →  p ≈ r*  how that allocation is actually realised

### Amended arm table

| arm | desired `r` | realised by | role |
| --- | --- | --- | --- |
| A0 | uniform | ordinary ABF bias | baseline -- converged ABF *is* the uniform-`r` bias arm, so no separate one is needed |
| A6a | `sqrt(a)` | bias | leverage only |
| A6b | `sqrt(aΓ̂)` | bias | pure information optimum |
| **A6c** | `sqrt(aΓ̂ + λM²)` | bias | **main candidate** |

Mechanism controls only, never candidates: **A2** (mass-only identity, must equal
A0), **A3** (legacy count balancing), **A4a**/**A4b** (identical `r` to A6a/A6b,
birth--death realisation). **A4a and A4b are frozen: no further tuning of their
resampling frequency.**

### The leverage has a closed form

`a(s) ∝ (s-L)(R-s)/(R-L)²` on the evaluation window, verified against the
computed diagonal to 0.3% (correlation 0.9999993). The centred cumulative
integral of an uncorrelated mean-force error is a Brownian bridge, and a
bridge's pointwise variance is `s(1-s)`. So the leverage is a property of the
endpoint's definition, not of the grid.

**The confound this creates.** `a` vanishing at the window edges means A6a spends
no effort where the metric does not score. Some of an A6a gain over A0 could be
that alone. So **every A6 result reports both the primary `e_F` and a fixed
full-domain `e_F`**, and the headline incremental claim is **A6b vs A6a**, which
holds `a` fixed and varies only `Γ̂`.

## Stage 0.5 -- the mechanism replication (runs before Stage 2)

Same `r`, different realisation, 8 fresh matched seeds on K0 and K2:
**A4a vs A6a** and **A4b vs A6b**, identical in every respect but mechanism.
Report `D_r(t) = ‖r_empirical(t) - r*_t‖_TV`, ancestor ESS, `N_replacements`,
`e_F(t)`. Prediction: the bias realisation tracks `r*` at no genealogy cost while
birth--death repeatedly pays to oppose the natural dynamics. If it replicates,
A4a/A4b are permanently demoted.

### Stage 0.5 result (run 2026-08-27, 8 paired seeds, 50k steps)

**H7 confirmed in both cells.** Identical `r*`, differing only in realisation;
paired bootstrap on shared seeds:

| cell | pair | birth--death | bias | ratio | 95% CI | ancestor ESS (bd) | replacements |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K0 | A4a/A6a | 0.1358 | **0.0193** | **6.55** | [5.25, 7.76] | 30.4 | 446 |
| K0 | A4b/A6b | 0.2116 | **0.0235** | **8.58** | [7.30, 11.79] | **4.2** | 2383 |
| K2 | A4a/A6a | 0.2572 | **0.1085** | **2.67** | [2.33, 3.78] | 45.2 | 212 |
| K2 | A4b/A6b | 0.3363 | **0.0493** | **7.08** | [5.58, 9.93] | **6.3** | 1536 |

**A4a and A4b are hereby demoted to mechanism controls permanently.** No further
tuning of their resampling frequency, cadence or gate.

**And the κ-family produced the two-sided pattern it was built for.** A6a vs A6b
differ only in whether `Γ̂` enters the allocation:

| cell | `Γ` | A6a | A6b | A6a/A6b | 95% CI | reading |
| --- | --- | --- | --- | --- | --- | --- |
| K0 | flat | 0.0193 | 0.0235 | 0.94 | [0.74, 1.14] | tie -- nothing for the difficulty channel to find |
| K2 | 16× spread | 0.1085 | **0.0493** | **2.09** | [1.29, 3.32] | the difficulty channel pays |

A theory that predicts its own ties is much harder to fish, and this is that
prediction coming out right before the confirmatory stage rather than after it.

**The safety metric attenuates but does not overturn it.** On K2, A6b's
full-domain error (0.0705) is 1.43× its primary (0.0493), so some of the margin
does come from spending nothing where the window does not score. But A6b's
full-domain error still beats A6a's full-domain error (0.1044) by 1.48×, so H2's
direction survives the confound at roughly two-thirds the apparent size. Both
numbers are reported; the primary alone would overstate the effect.

**Status: exploratory.** Final error, not the frozen time-to-accuracy endpoint;
8 seeds, not 32; A6c not included. Stage 2 remains the confirmatory test and
none of these numbers may be used to select an arm.

## Stage 2 -- the decisive κ-family experiment

`dY = -κ(X) ∂_y V dt + sqrt(2κ(X)/β) dW`, `κ_a(z) = exp(a(h(z)-1)/2)` with `h` a
fixed sinusoid the algorithm never sees; K0/K1/K2/K3. 32 paired seeds
(5200-5231). K0/K2/K3 confirmatory, K1 secondary dose.

Primary `S^(T)_ε = E[min(τ^base,T)] / E[min(τ^method,T)]`, paired bootstrap,
`P(τ≤T)` reported per arm. **If the candidate is censored more than its baseline,
that threshold cannot return a positive verdict** -- the clean-v2 amendment.

| | Prediction |
| --- | --- |
| H1 | K0: `A6b ≈ A6a` within [0.95, 1.05]. Flat `Γ` ⇒ nothing left for the difficulty channel. The corrected "theory predicts its own tie" -- it is **not** stated against uniform, which `a` alone already beats. |
| H2 | K2/K3: `S_{A6b/A6a} ≥ 1.10` at the stringent threshold, paired 95% CI lower bound > 1. **The cleanest test that `Γ̂` does useful work**, and what the κ-family was built for. |
| H3 | `S_{A6b/A0} ≥ 1.15`. Secondary to H2: A6b carries both `a` and `Γ̂`, so this cannot separate them. |
| H4 | A6c keeps `ESS_M/K ≥ ρ = 0.5` **by construction**, and retains `R_retain = (S_{A6c/A0}-1)/(S_{A6b/A0}-1) ≥ 0.8`. A6c solves a *more constrained* problem than A6b, so `A6c > A6b` is backwards and is not hypothesised. This is a viability criterion, not an optimality theorem. |
| H5 | mirror: `r*_K3` mirrors `r*_K2` while `q` is unchanged -- allocation tracks difficulty, not density |
| H6 | `A2 ≡ A0` on `F̂`. Otherwise mass leaked into the estimator; fail the campaign. |
| H7 | mechanism: `A6a > A4a` and `A6b > A4b` at identical `r*` (Stage 0.5) |

### The FR-relevance diagnostic, and why it is preregistered

Record `λ_t` on every opportunity. If `λ_t = 0` almost always then
`r*_A6c = r*_A6b` and **Fisher--Rao contributed nothing, however well A6c
performs.** So also report `P(λ_t > 0)`, `‖r_A6c - r_A6b‖_TV`, and A6b's
unconstrained mass ESS. Measured on a short A6c run: `P(λ>0) = 1.00`,
`TV = 0.162`, mass ESS 0.500 constrained against 0.199 unconstrained -- the
constraint is strongly active, which is what makes K2 the informative cell.

### Decision tree

- **A.** `A6b > A6a > A0`, `A6c ≈ A6b`, `ESS_M/K ≥ 0.5`, `λ > 0`. Information-optimal
  sampling accelerates ABF **and** the FR physical mass is retained cheaply. The
  q/r-decoupled FR project survives.
- **B.** `A6b > A6a > A0` but A6c loses most of the gain. Optimal allocation works and
  `q_phys` representation conflicts with it. The strongest method is then not an FR
  method; pivot to information-optimal ABF and say so.
- **C.** `A6a > A0` but `A6b ≈ A6a`. The benefit is `a(z)`, not hidden difficulty --
  metric-aware allocation, not a `Γ`-adaptive theory. Simpler, still real.
- **D.** Fresh-seed A6a/A6b do not beat A0. The one-seed pilot misled us. **Stop before
  molecular transfer.**

### Novelty, stated before the result rather than after

"Choose a target distribution and construct a bias that realises it" is **not** new:
that is the framing of VES and OPES, and optimised-ensemble work has long used local
dynamical difficulty to allocate effort toward hard regions. ABF has also been combined
with further adaptive bias mechanisms (meta-eABF, WTM-eABF, FK-eABF). So the claim
cannot be "we use another bias to make the RC marginal non-uniform." What may be new is
narrower and must be stated that way:

> derive the optimal replica marginal from the **actual free-energy estimator risk**,
> including the leverage `a(z)` and the local difficulty `Γ(z)`; estimate it **online
> without landscape knowledge**; and reconcile it optimally with a **separate physical
> Fisher--Rao mass**.

## Stages 3-5 -- only on Q2-A

3. Metastable quartic (old Example 2) + entropic gateway. Arms A0/A3/A4b/A5,
   every setting transferred unchanged. Success needs `S/A0 ≥ 1.15`, `S/A3 ≥ 1.10`,
   **and** `e_F(T) ≤ 1.05 × e_F^A0(T)` -- no early lead paid for at the end.
4. WCA. **Re-verify the cached TI reference first**: an audit branch found it can
   inflate some `F'` contrasts ~2×, and whether that reaches the integrated-`F`
   endpoint is unresolved. A0/A3/A5, 32 paired seeds.
5. ABF-sufficient negative control (butane or alanine). Want `A5 ≈ A0` **and**
   `N_resample ≈ 0`: a method that intervenes when it should and stands down when
   it should not.

Sequential. If Stage 2 fails, Stages 3-5 do not run.

## Reporting

Secondary: `e_F(t)`, `e_F'(t)`, `∫e_F dt`, `e_F(T)`, `e_F'(T)`, physical-marginal
fidelity. **No arm may be selected on final or integrated error.** Mechanism:
`Γ̂_j(t)`, `r_j(t)`, `r*_j(t)`, `q_j(t)`, mass ESS, ancestor ESS, `D_j`,
`N_resample`, `Δ_j`, predicted vs realised risk reduction. Figures: convergence;
paired `τ_ε`; **`q` vs `r` vs `r*`** at 4 times; `Γ̂` vs `Γ_ref`; the K2/K3 mirror;
genealogy; the speed-vs-mass-fidelity Pareto.

Shams are supplementary attribution only, run after a positive. They may not
gate, schedule or steer any arm.
