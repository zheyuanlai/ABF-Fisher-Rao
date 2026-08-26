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

Not yet written: **0D**, the mass-only identity `A2 ≡ A0`, which needs the engine
wiring. Run identity pairs **in one process** -- the engine is not bitwise
reproducible across processes.

## Stage 1 -- calibration, and two estimator gates that can stop the campaign

Backbone frozen from clean-v2: `β=4, K=256, dt=0.002, n_steps=50000, h=0.05,
update_every=10, eval_every=500`. Cells `J=32` equal width. Seeds 5100-5115
(16 ABF-only per κ-cell). Thresholds `ε₁, ε₂` = median ABF `e_F` at `0.4T, 0.6T`;
`τ_ε` needs 3 consecutive frames; every run goes to full `T`.

**1B -- is `Γ̂` usable?** Against `Γ_ref` from 4 seeds × 4T. Report rank
correlation, multiplicative error, top-cell overlap. Two hard stops:

- Batch means are **biased low when the block is shorter than `τ_int`**, and
  worst exactly where difficulty is highest -- the anti-detection failure mode.
  The κ-family varies `τ` by 16× *on purpose*, so `W=5000/B=10` is a guess that
  may be wrong for the slow cells. **Measure `τ̂_j` first and set the block length
  from it.** `information.block_length_adequacy` reports the ratio.
- `B=10` gives ~47% relative error on a variance (checked in Stage 0). Shrinkage
  is frozen at 0.3 toward the pooled value; unshrunk `Γ̂` hands the allocator
  dispersion shaped like heterogeneity.

If `Γ̂` cannot see the K2/K3 difficulty inversion, **do not run A4b/A5.** That is
an estimator failure, not a method failure, and fixing it is Stage 1 work.

**1C -- is the diagonal `Cov(f̂)` assumption sound?** The ABF estimator smooths
with `h=0.05` against a cell width of 0.1875, so cross-cell correlation should be
modest -- but report `‖Σ - diag Σ‖_F / ‖Σ‖_F`. If it is large, the theory owes a
covariance-aware allocation and `r* ∝ sqrt(aΓ)` may not be quoted as exact.

## Stage 2 -- the decisive κ-family experiment

`dY = -κ(X) ∂_y V dt + sqrt(2κ(X)/β) dW`. Mobility on the hidden coordinate
alone leaves the invariant density -- and hence `F` and `q_phys` -- unchanged
while moving conditional mixing. `κ_a(z) = exp(a h(z))` with `h` a fixed
sinusoid the algorithm never sees; `a ∈ {0, log4, log16, -log16}` = K0/K1/K2/K3.

Arms: **A0** plain ABF · **A1** legacy physical BD (failure control) · **A2**
mass-only (identity gate) · **A3** count balancing · **A4a** `sqrt(a)` · **A4b**
`sqrt(aΓ̂)` · **A5** `sqrt(aΓ̂+λq²)`, `ρ=0.5`. A3/A4a/A4b/A5 share one resampler,
one gate, one floor, one rejuvenation rule and one RNG split; only `r*` differs.
32 paired seeds (5200-5231). K0/K2/K3 confirmatory, K1 secondary dose.

Primary `S^(T)_ε = E[min(τ^base,T)] / E[min(τ^method,T)]`, paired bootstrap,
`P(τ≤T)` reported per arm. **If the candidate is censored more than its baseline,
that threshold cannot return a positive verdict** -- the clean-v2 amendment.

| | Prediction |
| --- | --- |
| H1 | K0: `A4b ≈ A4a` within [0.95, 1.05]. Flat `Γ` ⇒ no information left to exploit. |
| H2 | K2/K3: `A4b/A4a ≥ 1.10` and `A4b/A0 ≥ 1.15`, CI lower bound > 1 |
| H3 | mirror: `r*_K3` mirrors `r*_K2` while `q` is unchanged -- allocation tracks difficulty, not density |
| H4 | `A2 ≡ A0` on `F̂`. Otherwise mass leaked into the estimator; fail the campaign. |
| H5 | `A5/A3 ≥ 1.10` **and** mass ESS ≥ 0.5 **and** `A5` retains ≥ 80% of A4b's margin over A0 |
| H6 | A4a's margin does not relax back by `T` (the edge-evacuation confound) |

**Kill criteria.** Q2-A (`A4b > A4a > A3`, A5 holds ESS): continue. Q2-B (A4b
wins, A5 does not): information allocation works, insisting on `q_phys` costs it
-- the method is variance-aware ABF and the project's identity must be restated.
Q2-C (`A4b ≈ A4a` in K2/K3 with `Γ̂` validated in 1B): Neyman hypothesis
falsified; stop. Q2-D (nothing beats A0): population reallocation does not buy
finite-time ABF acceleration; **stop the direction -- no benchmark shopping.**

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
