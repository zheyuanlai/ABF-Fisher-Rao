# When and why does marginal Fisher–Rao help ABF? Final interpretation

*2026-10-10.*
* Plan: `SCIENTIFIC_PLAN.md` (44735ea), with Amendments 1 (bf56da4) and 2 (d3595341).
* Reports: `EXP1_TIMESTEP_VALIDATION.md`, `EXP1_MATCHED_FREE_ENERGY.md`, `EXP2_CONDITIONAL_RELAXATION.md`,
  `EXP3_PARALLEL_EFFICIENCY.md`.
* Log and ledger: `EXECUTION_LOG.md`.

This document keeps three kinds of statement apart:
* **[identity]** mathematical facts verified exactly;
* **[observed]** measured results with seed-level uncertainty;
* **[hypothesis]** explanations the data support or contradict but do not prove.

## 1. Answers to the five questions

### Q1. Does marginal FR help ABF on a purely energetic realisation of the same free-energy landscape?

**Yes [observed].** For α = 0, the same F\*(x) with the barrier entirely energetic and a deterministic local force:
* FR lowers the integrated free-energy error by **−24.0 % [−29.4, −15.2]** (31/32 seeds) at N = 2048,
  **−31.7 % [−37.6, −20.2]** (30/32) at N = 512 and −7.2 % [−19.4, −3.2] at N = 128.
* It improves the reaction-coordinate marginal by −71 % to −76 %, as much as on the entropic gateway.

**An entropic bottleneck is therefore not necessary.** The relative free-energy gain is somewhat smaller than at α = 1
(preregistered contrast C1: ratio 1.12 [1.05, 1.22] and 1.20 [1.08, 1.41], Holm-significant at N 2048 and 512).

### Q2. Is the entropic-versus-energetic difference explained by conditional-force variance?

**No [observed, mechanism a hypothesis].**
* **[identity]** The α family has Var(f | x) = (2α²/β²)(ω′/ω)². The shifted fibre matches α = 1's variance exactly.
* **[observed]** The variance-matched shifted fibre is the **worst** case for FR: −21 % at N = 2048, +2 % at 512 and
  **+105 %** at 128 (final error +831 %, 0/32 seeds). It does not behave like α = 1.
* **[observed]** The smaller relative gain at α = 0 coincides with a much stronger ABF baseline. Without conditional
  fluctuations ABF's bin means are exact once visited: ABF's Ī_F is 0.0129 versus 0.0227 for α = 1 at N = 2048, with
  final error 2 × 10⁻⁵. FR then mainly brings forward the moment the population is established.
* **[hypothesis]** The free-energy gain FR can deliver is limited by how much of ABF's finite-time error is due to
  population establishment. That share is smaller when the force carries no conditional noise.
* What separates the shifted fibre is not its variance but its slow, curved conditional relaxation: τ_f at the flank
  is 28× α = 1's, and the fibre centre moves 4.9 σ across the gate. FR's recent clones there deposit strongly biased
  forces (excess RMS bias +0.15 to +0.18 over ABF's, CIs far from 0).

### Q3. How strongly does FR depend on transverse relaxation speed?

**Strongly, at fixed budget [observed].** On the original potential with only the transverse mobility λ changed
(identical Gibbs law):

| | λ = 1 | λ = 0.1 |
|---|---|---|
| FR gain, N = 512 | −41 % | +4 % (final error +126 % worse) |
| FR gain, N = 2048 | −31 % | −23 % |
| marginal gain, N = 512 | −73 % | −84 % |

* The preregistered λ = 0.1 vs 1 contrast is significant at both N after Holm (ratios 1.21 and 1.98).
* The marginal gain *grows* at N = 512 while the free-energy gain disappears.
* **[observed, preregistered mechanism criterion]** At N = 2048, deposits from walkers cloned < 0.1 t.u. earlier carry
  a mean-force bias that grows 20-fold from λ = 1 to 0.1 (0.018 → 0.359, Holm p 0.004). It rises much faster than
  ABF's own deposit bias (0.012 → 0.115).
* By the plan's own rule the fibre-lag mechanism is **established at N = 2048** and **consistent but not significant
  at N = 512**.
* **[hypothesis]** FR copies a walker's y together with its x. If y has not relaxed for the walker's current x, the
  copy multiplies a biased force sample in exactly the under-populated bins where the estimator has the least other
  data. Slow conditional relaxation therefore turns FR's marginal repair into a mean-force bias.
* Young clones are only 1.3–2.4 % of gate deposits. Whether they explain the whole loss, or whether FR also biases the
  other deposits, is not resolved.

### Q4. Does FR give a practical wall-clock acceleration on parallel hardware?

**Partly answered** (see `EXP3_PARALLEL_EFFICIENCY.md`).
* **[observed] At fixed N = 512 on identical hardware (one CPU core per run, validated engines):** FR reaches the
  frozen mid accuracy in 0.35× (gateway, 8/8 seeds) and 0.36× (LTA 150 K, 6/6 uncontended) of ABF's wall-clock time.
  LTA 300 K gives 0.76 [0.66, 1.11], not significant. FR's added cost is 0.5–1.6 %.
* **[observed] GPU (existing torch engines, LTA only):** at N = 512 a single run is *slower* than one CPU core, and FR
  costs about 21 % more per step there. The time-to-accuracy ratio is in `EXP3_PARALLEL_EFFICIENCY.md`
  (**{{GPU_RESULT}}**).
* **NOT TESTED:**
  * a walker-parallel CPU build (none exists);
  * the gateway on the GPU (would need 16.9 GPU-h, over the 8 GPU-h ceiling);
  * a true multi-device scaling study.
* **So FR does shorten the wall-clock time at fixed N**, by roughly the same factor as its simulation-time gain, since
  its overhead is small on CPU.
* Whether a *large-N* FR allocation beats a *small-N* ABF allocation in wall-clock time on parallel hardware is **not**
  answered. It needs an implementation in which large N actually runs faster per walker-step than small N. On the GPU
  tested, it does not at N = 512. The equal-budget campaign already showed that, in force evaluations, few long ABF
  replicas match or beat FR's best allocation.

### Q5. When should FR be used, and when should ABF simply be run longer?

Use FR when **all** of the following hold [observed across Experiments I–II and the equal-budget campaign]:
1. **Many replicas, each short.** ABF's finite-time error must be dominated by the reaction-coordinate marginal still
   being under-populated, i.e. N large and T_N short. This holds for entropic **and** energetic barriers.
2. **Fast conditional relaxation.** The coordinates orthogonal to ξ must re-equilibrate quickly after a walker moves
   or is copied, relative to how often FR copies and how long walkers stay in the under-populated region.
   * Gateway λ ≥ 0.5 (τ_f at the flank ≲ 0.02 t.u.) is safe.
   * λ ≤ 0.25, or the curved shifted fibre (τ_f ≈ 0.38), is not.
3. **Not too few replicas.** At N ≲ 16 the kernel score is noise (equal-budget campaign).

Simply run ABF longer, or with fewer, longer replicas, when:
* the budget can be spent on per-replica time, since at equal force evaluations the best ABF allocation is
  never beaten;
* the orthogonal degrees of freedom are slow or curved relative to ξ, where FR's marginal gain then comes with a
  mean-force bias;
* ABF already converges once bins are visited (low conditional-force noise and a short establishment time), where FR
  adds little.

**A practical diagnostic**, which is a hypothesis to test in new systems: compare ABF's establishment time of the
marginal with T_N, and the conditional relaxation time of the local force with the FR event interval. FR helps when the
first is large and the second is small.

## 2. What is established, and what is not

**Established**
* [identity] The four Experiment I systems share F\*(x) and F\*′(x) exactly; only the microscopic realisation differs.
  Verified to 10⁻¹⁵ and by the gate.
* [observed] The timestep h = 2.5 × 10⁻⁵ is valid for all seven dynamics, by a frozen gate against exact samplers.
* [observed] The FR gains and losses above, with seed-paired CIs. The α = 1 cells are bitwise the equal-budget data.
* [observed] FR's marginal improvement is large in **every** cell (−63 % to −92 %), including those where the free
  energy gets worse. A better marginal is not a better free energy.

**Supported but not established**
* [hypothesis] Fibre lag of clones as the cause of FR's loss. This is preregistered-established at N = 2048 in
  Experiment II, consistent at N = 512, and observed in the shifted fibre, but the magnitude attribution is open.
* [hypothesis] The smaller relative gain at α = 0 arises because ABF's error has a smaller establishment share there.

**Not established**
* Whether fibre *narrowing* (entropy) has any effect of its own once relaxation and variance are controlled. The
  shifted fibre changes relaxation together with variance placement (plan §1.1), so it cannot isolate narrowing.
* A parallel wall-clock advantage of large-N FR over small-N ABF.

## 3. Recommendations for further experiments

At most two, and only because each answers a question this campaign could not.

1. **Separating relaxation from geometry in the curved fibre: the shifted fibre at κ ∈ {1, 2, 4}.**
   * With m = √(2/β) log ω/κ, Var(f | x) is unchanged for every κ ([identity]).
   * The conditional relaxation becomes κ² times faster and the centre displacement 1/κ smaller.
   * This single knob tests whether FR's harm in the curved fibre disappears when it relaxes fast. Same gate, same
     statistics, about 40 core-h.
2. **A real walker-parallel benchmark.** Use a multi-threaded or kernel-fused implementation in which N = 512 runs
   materially faster per walker-step than N = 16. Then compare wall-clock time-to-accuracy of large-N FR against the
   best small-N ABF on the same hardware. Without such an implementation the "FR saves wall-clock" claim stays limited
   to fixed N.
