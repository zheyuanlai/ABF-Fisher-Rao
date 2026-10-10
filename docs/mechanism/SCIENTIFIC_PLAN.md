# When and why does marginal Fisher–Rao improve ABF? — preregistration

*Written 2026-10-10 (≈ 10:30 UTC), before any new engine code, validation run or performance run of this campaign.
Frozen in git together with `configs/mechanism/{matched_free_energy,conditional_relaxation,parallel_benchmark}.json`.*

## 0. What is already known, and what this campaign can and cannot claim

Seen before writing (`docs/equal_budget/FINAL_RESULTS.md`, `CONFIRM_BEST_ALLOCATION.md`, `GATEWAY_TIMESTEP_VALIDATION.md`):

* On the original entropic gateway (this plan's α = 1, λ = 1), at the validated h_G = 2.5 × 10⁻⁵ with seeds 8100–8131,
  FR lowers the integrated free-energy error Ī_F by −31.4 % (N = 2048), −41.0 % (512) and −30.3 % (128).
  * It improves the trailing-half marginal by about −71 %.
  * It is neutral at N ≤ 16.
  * At the fresh-seed best allocation (N = 16) it is significantly worse (+24 %).
* In LTA the gain tracks how late ABF itself establishes the marginal (the 150 K vs 300 K test). In the gateway, the
  seed-level version of that test had the *opposite* sign.
* The α = 1 / λ = 1 cells of this plan therefore **re-use already analysed data** (§5). H1 is a known result here, not
  a fresh confirmatory test. The new information is in the α < 1, shifted-fibre and λ < 1 cells, and in the
  cross-variant contrasts, which no one has seen.

Earlier fibre-relaxation studies (`docs/GATEWAY_FIBRE_RELAXATION.md`, `GATEWAY_TRANSPORT_REFRESH.md`, 2026-09-02) found
that an exact transverse refresh helped ABF strongly. Those runs used h = 4 × 10⁻⁴, where the Euler–Maruyama fibre
variance is 26 % too wide at the gate. Their "fibre-lag" reading is therefore confounded by integrator bias. They are
**not** used as evidence here.

## 1. Models

All models share β = 16, H = 0.5, s = 0.1, ω(x) = 1 + 31 exp(−x²/(2s²)) and ξ(x, y) = x. x is reflected into
[−1.8, 1.8], the repository's convention (`gateway_ladder_numba._reflect`); y is unbounded. The target free energy is

  F\*(x) = H(x² − 1)² + β⁻¹ log ω(x) + C,  F\*′(x) = 4Hx(x² − 1) + β⁻¹ ω′/ω.

The barrier F\*(0) − F\*(±1) = 11.466 kT, made of 8 kT energetic and log 32 = 3.466 kT entropic.

### 1.1 Matched-free-energy family (Experiment I)

**α family:** V_α(x, y) = H(x² − 1)² + ((1 − α)/β) log ω(x) + ½ ω(x)^{2α} y², α ∈ {1, 0.5, 0}.
* Y | X = x ~ N(0, 1/(β ω^{2α})), so F_α = F\* + C exactly.
* Local force (the production convention: the deposited sample is ∂ₓV, with E[∂ₓV | x] = F′):
  f_α = 4Hx(x² − 1) + ((1 − α)/β)(ω′/ω) + α ω^{2α}(ω′/ω) y².
* E[f_α | x] = F\*′(x). Var(f_α | x) = (2α²/β²)(ω′/ω)², using Var(y²) = 2σ⁴.
* α = 1 is the original gateway.

**Shifted fibre:** V_sh(x, y) = F\*(x) + ½κ²(y − m(x))², with m(x) = (√(2/β)/κ) log ω(x) and κ = 1.
* Y | x ~ N(m(x), 1/(βκ²)), so F_sh = F\* + C.
* f_sh = F\*′(x) − κ²(y − m) m′(x), giving E[f_sh | x] = F\*′ and Var(f_sh | x) = κ² m′²/β = (2/β²)(ω′/ω)², the α = 1
  variance.

**Pre-registration numerical check** (scratch, not the formal gate), by quadrature in y at 11 x values and a 721-point
x grid:
* E[f | x] = F\*′ to 2 × 10⁻¹⁵;
* the variance identity to 2 × 10⁻¹⁵ relative;
* F_model − F\* constant to 2 × 10⁻¹⁶;
* analytic gradients equal central differences to 1 × 10⁻¹⁰;
* max |ω′/ω| = 16.25 at |x| = 0.21, so max Var(f | x) = 2.06 (SD 1.44) for α = 1 and the shifted fibre;
* shifted fibre: max m = 1.225, max |m′| = 5.74, max |m″| = 53.1, σ_y = 0.25.

**The confounds this family cannot remove (stated now so they are not over-read later):**

| variant | Var(f\|x) | gate transverse stiffness (λ = 1) | frozen-x transverse relaxation time at the gate, 1/(λ·stiffness) | other |
|---|---|---|---|---|
| α = 1 | original | ω² = 1024 | 0.001 t.u. (1 t.u. in the wells) | fibre narrows 32× at the gate |
| α = 0.5 | ¼ original | ω = 32 | 0.031 | fibre narrows 5.7× |
| α = 0 | 0 (deterministic) | 1 | 1 (irrelevant: f does not depend on y) | constant width; the barrier is purely energetic in x |
| shifted fibre | = original | κ² = 1 | 1 (1000× slower than α = 1 at the gate) | constant width, but the centre moves 4.9 σ_y across the gate: a curved channel, so y lags m(x) when x moves |

* The shifted fibre matches the conditional-force **variance**, but not the conditional **relaxation time** or the
  higher moments, and it adds a curved-channel lag. It is a variance-matched mechanistic control, **not** a clean
  "entropy off, everything else fixed" manipulation.
* The α family moves Var(f | x), the gate relaxation time and the entropic share together.

### 1.2 Transverse mobility (Experiment II)

The potential is V₁ (original), and only y's mobility changes:

  dY = −λ ∂_y V dt + √(2λ/β) dW^y,  λ ∈ {0.1, 0.25, 0.5, 1}.

* Both the drift and the noise are scaled, so the Gibbs law, F\* and F\*′ are unchanged for every λ > 0.
* The frozen-x relaxation time is τ_y(x) = 1/(λ ω(x)²); for y², which carries the force, it is 1/(2λω²).
* Var(f | x) is unchanged.
* The x dynamics, ABF and FR are unchanged.

## 2. Hypotheses (frozen)

**Experiment I**
* **H1.** FR accelerates ABF on the original gateway (α = 1). This is a known result; see §0.
* **H2.** The FR gain changes as the entropic contribution is moved into the energetic part (α = 1 → 0.5 → 0), at
  identical F\*.
* **H3.** Part of any α-dependence is explained by conditional-force variance rather than by entropy itself.
* **H4.** The shifted-fibre control (variance matched, constant width, slow curved fibre) separates the influence of
  conditional-force fluctuations from fibre narrowing, within the limits of §1.1.

**Experiment II**
* **R1.** FR keeps improving the marginal at small λ.
* **R2.** FR's free-energy and mean-force gain shrinks as λ decreases (slow fibre → copies carry unrelaxed y).
* **R3.** If both the marginal and the F gains vanish, slower discovery and slower conditional equilibration must
  be separated with the diagnostics of §6.

**Experiment III**
* **W1.** At fixed N = 512 and identical hardware, FR's lower simulation time-to-accuracy translates into lower
  wall-clock time-to-accuracy.

No hypothesis predicts a sign in advance: all outcomes are reported.

## 3. Numerical-validation gate (Experiments I and II), frozen

The gate is applied separately to **each** of the seven dynamics:
* α ∈ {1, 0.5, 0} and the shifted fibre, at λ = 1;
* λ ∈ {0.5, 0.25, 0.1} on V₁.

The candidate is **h = 2.5 × 10⁻⁵**, with refinement 1.25 × 10⁻⁵ if any fails. The harness is a generalisation of
`src/gateway_validation.py` (EM with a frozen flat bias +F\*′(x), MALA and exact i.i.d. sampling), with the same 16
groups × 256 walkers. Run length: 500 t.u. at λ = 1, extended to 500/λ t.u. at λ < 1 so that every chain covers the
same number of transverse relaxation times.

Gates, with thresholds identical to the frozen gateway gate (`configs/equal_budget_v2/gateway_validation.json`):

* **V1 conditional law.**
  * The EM conditional variance at the stiffest point is within 2 % of exact: the closed form
    1/(1 − λ·stiffness·h/2) − 1 ≤ 0.02, and the measured value is consistent with it (≤ 0.02 + 2 se).
  * For the shifted fibre, the measured conditional mean of y in the gate bins is within 3 se of m(x).
  * The variance is within 2 % of 1/(βκ²) at |x| ≤ 0.3.
* **V2 free energy.** D(F_density, F\*) ≤ 0.00185 with 95 % upper bound ≤ 0.0028, and the same for F from the
  integrated conditional mean force.
* **V3 mean force.** D(conditional mean force, bin-averaged F\*′) ≤ 0.0115 with upper bound ≤ 0.017.
* **V3b conditional-force variance.** The measured Var(f | x), bin-averaged over the eval bins, is consistent with the
  exact (2α²/β²)(ω′/ω)². The RMS relative deviation is ≤ 5 % on bins where the exact value is > 0.05; α = 0 has a
  measured variance ≤ 1e-12 × mean f².
* **V4 exact sampler.** MALA at h 1e-4 for each potential, and exact i.i.d. draws, reproduce the conditional law and
  F (upper bound ≤ 0.0028). This validates the harness and the read-out per model.
* **V5 sanity.**
  * No non-finite state; wall reflections < 1e-6 per walker-step in unbiased runs.
  * Flat-bias walkers cross the gate.
  * **ABF-bias stability:** for each dynamics, a 2-seed smoke run at N = 2048 for 4 t.u. with live ABF + FR has finite
    states and max |Γ| below 2 × max|F\*′| on visited bins.

**Selection.** The common h is the largest candidate at which **all seven dynamics pass**. If one fails at
2.5 × 10⁻⁵, all are refined to 1.25 × 10⁻⁵, or the failing variant is isolated as unresolved and reported.
* B = 2048 × 40/h, and T_N = Bh/N is the same physical horizon for every variant.
* The choice reads no ABF/FR performance.
* Deliverables: `docs/mechanism/EXP1_TIMESTEP_VALIDATION.md` and `figures/mechanism/validation/exp1_timestep_validation.{pdf,png}`.

## 4. Engine (implemented and reviewed before any validation or production run)

**Module.** A new `src/gateway_family_numba.py`, derived from `src/gateway_ladder_numba.py`. That file is NOT
modified: its sha256 is recorded in the equal-budget results.
* Model parameters: `variant ∈ {alpha, shift}`, α, κ, λ.
* At (alpha, α = 1, λ = 1) it must reproduce `gateway_ladder_numba` **bitwise**, for both arms, at N ∈ {2, 128, 2048},
  with the new diagnostics on.

**New passive diagnostics.** They read state only, and are tested inert, bitwise:
* **D1.** FR death and birth x-position histograms on the 180 production bins, cumulative, at the saves.
* **D2.** Clone-age-stratified deposits.
  * Each walker carries the step of its last clone event: the source and all its copies are marked at the gather.
  * Per production bin, the count, Σf and Σf² of the deposits are split into age classes [0, 0.01), [0.01, 0.1),
    [0.1, 1), ≥ 1 t.u. or never cloned.
  * Saved at the seven profile-snapshot fractions and at the end.
  * This yields the fraction of local force samples from recently cloned lineages, and the class-wise
    conditional-mean-force error against the bin-averaged F\*′.
* **D3.** Deposit counts per bin stratified by time since the walker last crossed the gate (its well label changed):
  < 0.1, 0.1–1, 1–10, ≥ 10 t.u., or never. Clones inherit the time. Saved at the same snapshots.
* **D4.** Walker traces also record y (32 walkers).

**Tests.** Gradients, exact F/F′, the conditional laws, the conditional-force variance, the integrator, population-size
invariance, FR resampling law, randomness and pairing, bitwise checkpoint/resume, and inertness of every diagnostic
(including D1–D4).

**Review.** Independent review before the gate.

## 5. Production design (frozen)

**Shared settings**
* h = the common validated h. B = 2048 × 40/h walker-steps per arm per seed, identical for every cell.
* **Seeds 8100–8131 (32)** in every cell.
* Initial conditions are coupled across all variants and λ: x from `init_left(seed, N)`, and y from the same standard
  normals z mapped through each model's exact conditional law, z/(√β ω^α) or m(x) + z/(√β κ).
* The Langevin noise seed is the same for every model.
* Within a cell the ABF and FR arms share initial conditions and noise; FR uses a separate PCG64 stream (as before).
* Algorithm: 180 bins, own-bin P0, min_count 1, γ 1.5, η 0.1, score clip 3, FR interval 0.004 t.u., ramp 4 t.u., cap
  max(1, ⌊0.08N⌋). These are the frozen gateway constants, never varied.

**Experiment I**
* α ∈ {1, 0.5, 0} and the shifted fibre.
* N ∈ {2048, 512, 128}, i.e. T_N = 40, 160, 640.
* Both arms, 32 seeds: 4 × 3 × 2 × 32 = 768 cells.
* The α = 1 cells re-use `results/equal_budget_v2/gateway/N{2048,512,128}/` if and only if the new engine at α = 1
  reproduces those files' arrays bitwise for ≥ 4 re-run jobs (both arms, N = 128 and 2048). Otherwise α = 1 is re-run.

**Experiment II**
* λ ∈ {0.1, 0.25, 0.5, 1} on V₁.
* N ∈ {2048, 512}, both arms, 32 seeds.
* λ = 1 re-uses the same production files under the same condition.

**Throughput.** 576 + 384 new runs at about 270 s each ≈ 72 core-h.

## 6. Metrics and statistics (frozen)

**Reference and scorer.** The common analytic F\*, F\*′ with the accepted gateway scorer and conventions:
* 151-node eval window [−1.5, 1.5], each profile centred on the window;
* e_F, function-space e_F′, and the floor-free e_F′_stat with e_F′² = e_F′_stat² + 0.03227².

Identical for every variant and λ. The secondary reference is each dynamics' own frozen-x EM-consistent mean force:
* α family: 4Hx(x² − 1) + ((1 − α)/β)(ω′/ω) + (α/β)(ω′/ω)/(1 − λω^{2α}h/2);
* shifted fibre: exactly F\*′ at frozen x.

**Endpoints**, all per (cell, N), paired per seed:
* **Primary:** G(Ī_F) = (FR − ABF)/ABF, the median over seeds with a 10 000-resample seed bootstrap 95 % CI and wins.
  These are the equal-budget conventions.
* **Secondary:**
  * G(final e_F), G(Ī_F′_stat), G(final e_F′_stat), G(Ī_TV_half), G(final TV_half);
  * the mean-based companions;
  * persistent τ at the thresholds below, in t and u, with censoring and the sign test.
* **Thresholds:**
  * e_F: 0.002 / 0.0055 / 0.02 (the frozen gateway values);
  * e_F′_stat: 0.004 / 0.0136 / 0.0946 (the raw mid and loose statistical budgets, plus 1/3 of the α = 1 N = 2048 ABF
    final e_F′_stat);
  * TV_half ≤ 0.1.
  * Raw e_F′ is reported, but no τ is computed below its 0.03227 floor.

**Cross-cell contrasts (the new tests)**
* Seeds are coupled across cells, so for cells a and b at the same N:
  Δ_ab = median over seeds of [log(Ī_F^FR/Ī_F^ABF)_a − log(…)_b].
  * Bootstrap 95 % CI and p over seeds.
  * Reported as the ratio exp(Δ): < 1 means a has the larger FR gain.
* **Experiment I primary contrasts:**
  * C1 α = 0 vs α = 1;
  * C2 shifted vs α = 1;
  * C3 shifted vs α = 0;
  * C4 α = 0.5 vs α = 1.
  * Each at N ∈ {2048, 512, 128}: 12 tests with Holm correction for any "differs" claim.
* **Experiment II primary contrasts:** D1 λ = 0.1 vs λ = 1 at N ∈ {2048, 512}, with Holm over 2. λ = 0.25 and 0.5 are
  reported as the trend.
* **Equivalence:** two cells' FR gains are called equivalent only if the 95 % CI of exp(Δ) lies inside [0.90, 1.11].
* The same contrasts are computed on Ī_TV_half and on Ī_F′_stat as secondary results.

**Mechanistic covariates.** These are descriptive and are never used as causal proof.
* Exact Var(f | x).
* Measured conditional autocorrelation times of y and of f. These come from the validation chains, conditioned on x
  in bins around x ∈ {−1, −0.21, 0}, at lags up to 5τ.
* ABF establishment τ(TV_half), first arrivals and transitions per cell.
* D1–D3: event positions, recently-cloned deposit fractions and their mean-force error, and time-since-crossing
  stratification.

## 7. Interpretation map (decided before data)

**Experiment I**
* **C1 equivalent.** An entropic bottleneck is not necessary for the FR gain.
* **C1 shows a smaller gain at α = 0.** Read C2 before attributing it to entropy.
  * C2 equivalent (the shifted fibre gains like α = 1): conditional-force fluctuations or coupled dynamics matter
    more than fibre narrowing.
  * The shifted fibre gains like α = 0: narrowing or confinement geometry matters. Because of §1.1 this is not
    isolated causally: the shifted fibre also has slow, curved conditional relaxation.
* **C1 shows a larger gain at α = 0.** Entropy is not needed, and conditional-force fluctuations may *limit* FR.
* **C4** tests whether the dependence is graded with α.
* Every claim must be checked against the marginal endpoints. Differing F gains with equal marginal gains point to the
  force estimate, not to discovery.

**Experiment II**
* **The marginal gain persists while the F/F′ gain shrinks as λ → 0.1.** Supports a conditional-relaxation
  limitation. This is called "proven" only if D2 shows recently cloned deposits carrying a mean-force error that grows
  as λ decreases.
* **Both shrink.** Separate slower discovery (first arrivals, establishment) from slower equilibration (D2, conditional
  ACF).
* **The F gain is unchanged at every λ.** Fast conditional relaxation is not essential.

**Falsification of the working hypothesis** ("FR helps when establishment limits ABF and clones are conditionally
informative"):
* it is weakened if the FR gain at fixed N does not order with the ABF establishment time across the α family;
* or if the gain is unchanged by λ.

## 8. Experiment III protocol (frozen endpoints; implementation details amended before its runs)

**Comparison.** At fixed N = 512, on identical allocated hardware, ABF vs ABF + FR for:
* the original gateway (h 2.5e-5);
* LTA 300 K and 150 K (h 2e-4).

These are the existing validated configurations.

**Primary endpoint:** wall-clock time to the persistent mid threshold of e_F (gateway 0.0055, LTA 300 K 0.094 kJ/mol,
150 K 0.13).
* Wall time is measured by timestamps at every save, and it excludes JIT/compile time, which is reported separately.

**Also reported:**
* simulation time and force evaluations to accuracy;
* peak memory;
* FR's added time;
* effective parallelism (the measured speed-up over one core);
* total resource use.

**Hardware candidates**, in preference order:
1. One GPU (H200 #2 was idle at writing) with the existing torch production engines, if a GPU implementation with
   the same algorithm exists and passes a statistical-equivalence check against the validated numba engine.
2. A walker-parallel CPU build on k cores.
3. The validated single-core numba engines.

Any backend not run is marked **NOT TESTED**. One CPU core is never equated with a GPU. Experiment III starts only
after Experiments I and II are analysed.

## 9. Execution, resources, stopping

**Order:** engine and tests → review → validation gate → 2-seed smoke runs (stability, checkpoints, diagnostics,
plots; never interpreted) → Experiment I → Experiment II → analysis → Experiment III → synthesis.

**Ceilings:** 400 CPU core-h and 8 GPU-h. Estimated use: about 100 core-h for Experiments I + II with validation, and
≤ 20 core-h + ≤ 3 GPU-h for Experiment III. No other user's process is touched.

**Stopping rules:**
* a failed gate for every candidate h;
* an engine-equivalence failure;
* a numerical instability;
* the resource ceiling.

A neutral or negative FR result is not a stopping reason. Incomplete cells are labelled, never dropped.

**Storage:** the disk was 98 % full (78 GB free) at writing. Raw files are about 0.6 MB per run, so about 0.6 GB in
total.

---

## Amendment 1 (2026-10-10, committed 11:51 UTC (bf56da4) — the header first said "≈ 12:30", an estimate, corrected from the commit time; after the implementation review and BEFORE any validation-gate run, smoke run or production run)

The implementation review raised four points that the frozen text left ambiguous, or that would make it fail for
reasons unrelated to its purpose. No gate, smoke or performance data existed when this was written.

**A1. V5 ABF-bias stability: which Γ.** "max |Γ| below 2 × max|F\*′| on visited bins" is now read on the **final** Γ
profile of the smoke run, over bins with ≥ 100 deposits, together with finite states and accumulators.
* The running maximum over all bias reads is reported descriptively, not gated.
* Reason: at λ = 0.1 the running maximum reaches 19.6–30.4 against the 15.8 limit (engine test, N = 2048, 1 t.u.)
  while every state stays finite and the final profile is ≤ 9.2.
* The cause is physical, not numerical. Walkers entering the gate flank still carry well-width y, because the
  transverse relaxation 1/(λω²) is slow there, so their local forces are large. They deposit into bins with only a few
  counts, where Γ = M/(C + 1) is a noisy average.
* This phenomenon is the Experiment II question itself, and it is reported as such.

**A2. V1 reading.** V1 passes iff the closed form 1/(1 − λ·stiffness·h/2) − 1 ≤ 0.02 **and** the measured inflation
≤ 0.02 + 2 se.
* For the shifted fibre it also needs the conditional-mean 3-se test in the two central fine bins and the pooled 2 %
  variance check at |x| ≤ 0.3.
* The gateway config's additional "|measured − closed form| ≤ 3 se" clause is computed and reported as
  `V1_strict`. A strict-only failure triggers an investigation note, not a refinement. Reason: 16 such tests per h
  give a 13 % chance of a false failure even with exact discretisation.

**A3. V4 multiplicity.** The V4 family consists of all 3-se exact-sampler checks across the seven dynamics (32 tests).
* It is evaluated with Holm–Bonferroni at family-wise 5 %, using per-test two-sided p from the t₁₅ distribution
  (16-group jackknife).
* Unadjusted, the family would read HARNESS_FAIL about 25 % of the time with every sampler exact.
* V4 stays an h-independent precondition, as in the frozen gateway rule.

**A4. α = 1 / λ = 1 cells are re-run, not re-used.**
* The re-used equal-budget files carry no D1–D3 diagnostics, but §7 requires the λ = 1 point of D2.
* The new engine therefore runs all 192 α = 1 jobs (N 2048 / 512 / 128 × 2 arms × 32 seeds, about 15 core-h). The
  λ = 1 cell of Experiment II aliases the identical α = 1 N = 2048 / 512 jobs.
* **Every** re-run is compared bitwise, on all shared arrays, with its equal-budget counterpart. This replaces the
  ≥ 4-job reuse gate with a 192-job one.
* Any mismatch stops the campaign for investigation. H1 still re-uses the same seeds, so §0's caveat stands.

**A5. Cost.** The frozen validation design (500/λ t.u. per chain) measures about 58 core-h, or about 157 if the
refinement to 1.25e-5 is needed. The production total is now about 1 152 runs × 290 s ≈ 93 core-h. The plan's
estimate of about 100 core-h for Experiments I + II with validation becomes about 150 (about 250 with refinement).
All of this is under the 400 core-h ceiling, so no design change is needed.
