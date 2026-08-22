# RC-WFR-TI: what a bias-free reaction-coordinate Wasserstein–Fisher–Rao sampler
# can and cannot do

*Campaign record. Every number is reproduced by a script in `scripts/`; every
measurement, in the order it was taken, is in `docs/RESULTS_LOG.md`.*

---

## 1. The question

The previous campaigns (`ABF-Fisher-Rao`, branch `abp-fisher-rao`) established that a
marginal Fisher–Rao correction is **redundant on top of adaptive biasing**: the
adaptive bias already owns the reaction-coordinate marginal, so FR either repeats what
the bias is doing or cannot see the problem that remains. The natural response was to
remove the adaptive bias entirely and let a **Wasserstein–Fisher–Rao** flow own the
marginal outright:

```
conditional MD on the fiber  ->  W transport of z  ->  lift  ->  FR reallocation  ->  TI
```

with `F` recovered from the conditional mean force by thermodynamic integration, and no
bias potential anywhere. The preregistered question (`docs/PREREGISTRATION.md`):

> Can a bias-free reaction-coordinate WFR sampler compute free energies faster than
> adaptive biasing — and than classical stratified thermodynamic integration?

---

## 2. Verdict

**Against adaptive biasing: yes, in an identifiable and predictable regime.**
**Against classical stratification: it depends on the fiber, and on which form of the
Wasserstein step is used.**

Two things had to be got right before the method could be judged at all, and both were
found by this campaign rather than assumed:

1. the **stochastic** Wasserstein step is the wrong one. Replacing
   `Z <- Z + sqrt(2 kappa dtau) eta` by the deterministic probability flow
   `Z <- Z - kappa dtau grad log p_hat(Z)` changes the error by up to an order of
   magnitude, because the flow velocity vanishes as `p -> u`, so its hysteresis
   self-annihilates instead of persisting forever;
2. deterministic transport and Fisher-Rao resampling are **incompatible** unless a
   small resample-move jitter is added — clones follow identical trajectories and the
   ensemble collapses. The jitter window is narrow (`sigma = 0.01` optimal on a
   3.6-wide domain; `sigma = 0.05` costs a factor 6).

With both fixed, RC-WFR is a real method with a real regime:

All figures below are paired median relative changes in `I_F` on 32 fresh confirmation
seeds at matched force evaluations, with every baseline screened at least as hard as
RC-WFR.  Bold entries have 95% bootstrap CIs excluding zero.

| comparison                                            | result |
|-------------------------------------------------------|--------|
| vs **ABF**, hidden-channel fiber                       | **-82.4%** [-85, -73] |
| vs **cold-start RE-TI**, hidden-channel fiber          | **-50.1%** [-55, -14] |
| vs **cold-start fixed-window TI**, hidden-channel fiber| **-70.5%** [-75, -50] |
| vs **ABF**, easy fiber                                 | **-62.6%** [-66, -58] |
| vs **fixed-window TI**, easy fiber, no fiber model     | +40.5% [+29, +54]     |
| vs **fixed-window TI**, easy fiber, exact analytic lift| **-36.1%** [-40, -34] |
| vs **ABF**, long torsional CV (L = 24)                 | **-89%** (0.0224 vs 0.198) |
| vs **fixed-window TI**, long torsional CV (L = 24)     | **-12%** (0.0224 vs 0.0254) |
| vs **RE-TI**, long torsional CV (L = 24)               | +57% (0.0224 vs 0.0142) |
| vs **ABF**, short torsional CV (L = 3)                 | +191% [+163, +209] |
| vs **SHUS / ABP**                                      | better by 1-2 orders of magnitude everywhere |
| vs oracle-initialized TI / RE-TI                       | worse (they use information nobody has) |

and it retains one hard, structural limitation that no amount of compute removes.

## 2b. The structural limitation

> Any move that changes `xi(q)` without knowing `F` cannot be Metropolis-corrected,
> because the acceptance ratio for the target `u(z) nu^xi(dq|z)` contains
> `exp(+beta F(xi(q)))`. Replica exchange escapes this by swapping between two
> **occupied** windows, where the unknown weights cancel identically. RC-WFR instead
> moves unconditionally and does not correct — buying CV transport at the price of a
> hysteresis bias set by `kappa * tau_fiber` summed over EVERY fiber mode it drags,
> including the slowest one, which is precisely the mode that made physical CV
> transport slow to begin with.

So RC-WFR converts a *convergence* problem into a *bias* problem. More compute fixes
the first and not the second. The deterministic flow mitigates this (its velocity, and
therefore its hysteresis, decays to zero) but does not repeal it: the bias is still
extensive in the number of dragged fiber modes, which is why RC-WFR falls further
behind RE-TI as the fiber grows rather than catching up (Section 4, F5).

The one component of RC-WFR that is entirely free of this problem is **Fisher-Rao**:
selection copies a walker together with its fiber configuration and drags nothing, so
it reallocates population at zero hysteresis cost. That is why the best configurations
found here use a SMALL `kappa` and a LARGE `theta` — minimum dragging, maximum
birth-death — and why removing FR costs a factor 2.4-2.5.

The sharpest form of that statement: on the hidden-channel system, RC-WFR **with**
Fisher-Rao beats cold-start replica exchange by 50.1%; the identical arm **without** it
loses to replica exchange by 27.5%. The birth-death half is not a refinement — it is
the difference between winning and losing.

## 3. What was confirmed

**M1. The marginal WFR flow is implemented correctly and its complementarity is real.**
Particle KL(p_t‖u) tracks the WFR PDE to 0.28% (W only) and 3.8% (W+FR) median relative
deviation. FR alone provably cannot enlarge the particle support (support width
unchanged to four decimals over 500 events at two bandwidths) and correspondingly fails
to track the Eulerian PDE. Time to `KL < 0.05` scales as **L²** for W alone and as **L**
for W+FR — a reaction–diffusion front of speed `~2 sqrt(kappa lambda)` replacing
diffusive relaxation. (Figure 1.)

**M2. The mechanism survives into the free-energy setting.** On the confirmation run
`wfr` (0.0361) beats `w_only` (0.0765) and `fr_only` (1.037, coverage 0.067). Both
halves are necessary.

**M3. RC-WFR does beat ABF, in the regime the marginal argument predicts.** ABF's CV
equilibration is diffusive, so its cost grows like L² while RC-WFR's grows like L. On
a periodic torsional landscape with fixed well spacing, RC-WFR moves from **+191%**
worse than ABF at L = 3 to a tie at L = 6, and on the high-barrier and large-fiber
systems it beats ABF by 45–87%.

---

## 4. What was falsified

**F1. The lift is the whole problem.** With an *oracle* lift (exact conditional
refresh) RC-WFR sits at 1.0-1.1x the estimator floor at every transport rate from
`kappa = 0.03` to `8.0`, on both systems. Every error the method makes is lift
hysteresis; none of it is the WFR flow. (Figure 2.)

**F2. With the SDE step the bias is irreducible.** It grows monotonically with `kappa`
(3.3x -> 28x the floor) and is **independent of `n_cond`** — a steady-state hysteresis,
not a post-jump transient, so relaxing longer after each move does not remove it. Only
the deterministic flow form escapes this, and only because its velocity decays.

**F3. A model-based lift only repairs the modes it models.** Rescaling the fiber
coordinate by `omega(x)/omega(x')` — exact for a harmonic fiber — makes RC-WFR the best
non-oracle arm on the easy system (`I_F` 0.0071 vs fixed TI's 0.0114). On the
hidden-channel system, where the slow mode is *which* channel is occupied, the same
lift makes the error **1.1-1.7x worse than doing nothing**. A lift built from a local
model cannot be trusted: it repairs what one already understands and can damage what
one does not — which is, by construction, what made the problem hard.

**F4. Fast CV transport and correct conditional sampling are in direct conflict.** On
the hidden-channel system RC-WFR's channel error *grows* with its own transport rate
(0.12 at `kappa = 0.03` to 0.45 at `kappa = 8`): walkers are dragged through the switch
region faster than the slow mode can equilibrate. This is why the winning configuration
uses a small `kappa` and leans on Fisher-Rao, which reallocates population without
dragging anything.

**F5. The hoped-for large-system crossover does not exist.** Exchange acceptance decays
only slowly with fiber size (0.975 -> 0.814 over `m_spec = 0 -> 512`), while RC-WFR's
lift bias is extensive in dragged modes, so the gap widens instead of closing:
RC-WFR goes from +33% to +270% worse than RE-TI. (Figure 5b.) **Prediction P2 is
falsified.**

**F6. Smooth Fisher-Rao is not needed.** Count balancing ties smooth FR three separate
times (EB SDE 0.03502 vs 0.03612; EB flow 0.01504 vs 0.01513; CHANNEL flow 0.07804 vs
0.06519), while the matched-turnover sham is 2.3x worse. The *direction* of the
reallocation matters and its Fisher-Rao geometry does not. This reproduces the ABF/ABP
campaign's result in a setting with no adaptive bias to be redundant with, which makes
it a property of the uniform target rather than of the host method. **Hypothesis H4 is
rejected for the count control and upheld for the sham control.**

---

## 5. Prediction P1, confirmed quantitatively

RC-WFR's standing against ABF improves **monotonically** with the CV domain length,
exactly as the marginal argument predicts (ABF's CV equilibration is diffusive,
`O(L^2)`; W+FR is a reaction-diffusion front, `O(L)`). Periodic landscape with wells at
fixed spacing, identical local physics at every L, 25.6M force evaluations per arm,
each family free to pick its replica count:

| L  | wells | RC-WFR vs best ABF        | RC-WFR vs best fixed TI |
|----|------:|---------------------------|-------------------------|
| 3  |     2 | +191.3% [+163, +209]      | +164.9% [+95, +269]     |
| 6  |     4 | -19.1% [-37, +20] (tie)   | +95.8% [+61, +119]      |
| 12 |     8 | **-72.2%** [-75, -58]     | +39.9% [+26, +94]       |
| 24 |    16 | **-82.5%** [-86, -79]     | +31.4% [+9, +67]        |

Stratified TI also degrades with L at fixed budget — it needs `M ~ L` windows for fixed
CV resolution, so samples per window fall like `1/L` — and RC-WFR closes on it steadily
(+165% -> +96% -> +40% -> +31%).

Those RC-WFR numbers are **conservative**: the scan fixed the marginal-density bandwidth
at `bw_kde = max(0.10, L/60)`, which is far too coarse at large L. Screening it properly
at L = 24 gives RC-WFR `I_F = 0.0224`, which is **12% better than the best fixed-window
stratified TI (0.0254) and 89% better than the best ABF (0.198)**, though still 57%
behind RE-TI (0.0142). That screen also shows the probability flow is not universally
the better W step: on this long torsional domain the stochastic step at moderate `kappa`
wins (0.0224 vs 0.0279). The flow's advantage is specific to regimes where residual
hysteresis dominates the error; where transport itself is the bottleneck, the SDE's
larger effective step wins.

---

## 6. Where this leaves the idea

RC-WFR is a real method with a narrow, *predictable in advance* regime:

| condition | RC-WFR |
|---|---|
| CV domain long relative to physical CV diffusion | beats ABF, and the margin grows with L |
| high enthalpic or entropic barrier the bias must learn | beats ABF and SHUS decisively |
| fiber has a slow mode with a localized switch region | **beats cold-start RE-TI by 50%, stratified TI by 70%, ABF by 82%** — with small kappa, strong FR and the flow step |
| fiber has an exact, cheap analytic lift | beats cold-start stratified TI by 36% |
| easy unimodal fiber, short CV domain | loses to stratified TI and to ABF |
| system size grows | loses further; the lift bias is extensive in fiber modes |
| only one starting structure is available | the flow form cannot start at all (zero score at a delta ensemble); use the SDE form for a few steps first |

The honest positioning:

> **Reaction-coordinate WFR is a grid-free, continuum alternative to stratified
> thermodynamic integration whose CV transport is unconditional and therefore biased by
> exactly the fiber modes it drags. Its Fisher-Rao half is hysteresis-free and does most
> of the useful work; its Wasserstein half should be run deterministically and gently.
> It beats adaptive biasing by a margin that grows with CV domain length, and it beats
> Hamiltonian replica-exchange TI when the fiber's slow mode has a localized gateway —
> but it is not a general replacement for either, and its bias does not go away with
> more compute.**

## 7. What would have to be true for the method to be general

A lift that is asymptotically exact *without knowing F*. The campaign found four kinds
and none qualifies: the oracle (not implementable); a model-based rescaling (repairs
only modelled modes, damages unmodelled ones); annealing `kappa -> 0` (removes the bias
only by removing the transport, converging to stratified TI); and the deterministic
probability flow (self-annihilating, which is why it is the best variant found — but
still extensive in dragged modes). Any fifth candidate must supply the missing
`exp(+beta F)` weight from somewhere other than an estimate of `F`, and the only known
mechanism that does is exchange between occupied windows.

Two directions the campaign did NOT close and that look worth a look:

* a **hybrid**: use RC-WFR's front to establish coverage, then hand the resulting
  configurations to exact RE-TI. The two mechanisms are complementary — RC-WFR is fast
  and biased, RE-TI is exact and slow to mobilize — and RC-WFR's own annealed variant
  is already a crude version of this;
* **variance-optimal targets**. Everything here targets `u(z)` uniform, which allocates
  computation evenly rather than where the mean-force variance is. That is an
  allocation question, entirely separate from the bias question, and the FR machinery
  can carry any target at no extra cost.

## 8. Reproducibility

Every number in this report is produced by a script in `scripts/` and recorded, in the
order it was measured, in `docs/RESULTS_LOG.md`. Arms in a comparison share `N`,
`n_steps`, the estimator, the initial ensemble and the seed base, so all comparisons are
paired; RE-TI's exchange energy evaluations are charged to its force budget and its
inner loop shortened to match. Every claim is quoted against a measured estimator floor
and no difference at or below that floor is claimed. Wall clock is reported separately:
RC-WFR's marginal machinery costs about 1.5x the wall clock of stratified TI in this
toy, where a "force evaluation" is a two-term polynomial; in any real system the force
cost dominates and that overhead vanishes.
