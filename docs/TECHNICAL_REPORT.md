# Fisher-Rao population reallocation for adaptive free-energy computation

### What the hypothesis was, what happened to it, and what the campaign established instead

**Status: FROZEN 2026-08-20.**  Branch `application-map`, tag `v1-application-map-final`.
Every number below is taken from a recorded outcome in `PREREGISTRATION_GATEWAY.md`,
`PREREGISTRATION_WCA.md` or `PREREGISTRATION_APPLICATION_MAP.md`; none of those outcome
sections were edited after the fact, and the two places where an *interpretation* was
later narrowed are marked as scope corrections in the text they narrow.  143 tests green.

---

## 0. Conclusion

> Fisher-Rao population reallocation can produce faster finite-time decay of
> free-energy error, but the observed speedups are not robustly attributable to
> Fisher-Rao itself.  Marginal speedups disappear relative to properly tuned adaptive
> biasing, or are reproduced exactly by simple count balancing.  Large *conditional*
> speedups arise when equal-weight selection pushes the represented conditional
> distribution toward an externally chosen target; preserving the represented measure
> removes essentially all of that acceleration.  Across the tested regimes there is
> therefore no evidence for a target-free Fisher-Rao acceleration of adaptive
> free-energy computation.

The campaign's value is not that sentence but the map underneath it: *why* apparent
acceleration occurs, *when* it disappears, and *which pre-run diagnostics* tell the two
cases apart.

---

## 1. The hypothesis, and why it was reasonable

An adaptive-biasing method estimates `F(xi)` from the occupancy of a walker ensemble
under a bias it is simultaneously learning.  If the ensemble does not populate a region,
the estimator learns nothing there.  The predecessor campaign (mFR-ABF) had found cases
where that population deficit, not the sampling rate, was the binding constraint.

Fisher-Rao gradient flow toward a uniform target,

    d_t p = -p ( log(p/u) - KL(p||u) ),

is the canonical way to move a population toward flatness without touching the physics.
Realized on particles as a finite-`theta` birth-death step,

    p^+ propto p^{1-theta} u^theta,   particle weight a_k = (u(x_k) / p_hat(x_k))^theta,

followed by systematic resampling.  The hypothesis: a **temporary, moderate** FR step
should damp the post-discovery establishment transient of an accumulating bias,
improving both integrated error `I_F` and time-to-accuracy `tau_eps`, while a
persistent/strong step should hurt through estimator-resampling feedback.

The adaptive method chosen was a mollified **SHUS** accumulator (an adaptive-biasing
*potential*), because its estimator is the occupancy itself — the quantity a population
method acts on — so the interaction is direct rather than mediated by a force estimate.

Two conventions were frozen before any run and never moved: the FR step may gather
walker arrays only, never accumulator state (**estimator protection**); and every FR arm
is accompanied by a **matched-turnover sham** that performs the same number of
births/deaths with the direction destroyed.

---

## 2. What was built

| Component | What it is |
|---|---|
| `shus.py`, `shus2d.py`, `grid1p.py` | mollified SHUS on an interval, a torus, a circle; analytic fixed point `R* = K_eps e^{-beta F}` and its error floor `e*` |
| `fisher_rao.py`, `events*.py` | finite-`theta` FR step, ESS backoff, systematic resampling, matched shams |
| `fisher_rao_cond.py` | fiber-wise (conditional) reallocation inside `xi`-strata; histogram and discrete-state controls; weighted (measure-preserving) selection |
| `systems/gateway.py` | entropic gateway, analytic reference |
| `systems/wca.py` | WCA dimer in solvent, `hp_v3` reference |
| `systems/torus2d.py` | 2D periodic model with an exact reference |
| `systems/alanine.py` | ff14SB vacuum alanine dipeptide: analytic forces and dihedral gradients, CUDA-graph block replay, MBAR reference on the engine's own grid |
| `systems/bichannel.py` | the designed Type-C system: two channels spanning the CV with a barrier orthogonal to it, exact quadrature reference |
| `metrics.py`, `diagnostics.py` | `e_F`, `I_F`, `tau_eps`/`S_eps` (0.2 T persistence, right-censored), `T_hit`/`T_est`, KDE noise floors, paired bootstrap |

All production runs batch every `(seed x arm)` row into one GPU call, so arms of a seed
share initial conditions and Langevin noise and every comparison is paired.

---

## 3. Five turns of the hypothesis

### Turn 1 — the positive (gateway, Stages 1-3)

Plain SHUS on the gateway showed a genuinely underdamped establishment transient:
occupancy overshoot, sign changes in the slow mode, `e_F` ringing.  Temporary FR
(`theta = 0.01`, stride 10 blocks, window [6, 14]) gave

* `dI_F = -11.4% [-12.5, -10.8]`, better terminal error,
* persistent FR **overdamped**: `+2.5% I_F`, ancestry collapse to `0.17 K` — the
  predicted failure mode, observed.

The preregistered speed target `S_eps* >= 1.25` **failed** (`1.02`, 26/32 seeds
censored); the uncensored ladder gave `1.10 / 1.08 / 1.04`.  Recorded as a Level-1
accuracy pass with a modest speedup.

Already in that stage: **count balancing tied FR exactly.**

### Turn 2 — the confounds (Phases A, B, D, ALA)

Four independent attacks on the positive, all preregistered:

* **Adaptation gain.**  `g_SHUS` multiplies the accumulator increment.  On the gateway,
  slowing it hurts badly (`+85%` at `g = 0.25`) and `g = 1.5` gives `-9.4%` — i.e. most
  of what FR gave.  On fresh seeds, FR vs the tuned baseline: **`-0.9% [-3.5, 0.6]`, a
  tie.**
* **Resources.**  WCA at `K in {32, 64, 128, 256, 1024}`: every cell SHUS-sufficient,
  `T_hit/T <= 0.008`, no establishment window ever opens.  Small `K` costs estimator
  *noise*, which reallocation cannot repair.  Preregistered rule: no WCA FR run, ever.
* **Geometry.**  On a 2D torus cell that looked establishment-limited, all reallocation
  arms were inert (`+0.3%`) while `g = 8` gave `-73.7%` and moved `T_est` from 162 to
  13.8.  On the tuned baseline, FR added `-0.1% [-0.5, 0.4]`.
* **A real molecule.**  Vacuum alanine at `xi = phi` and `xi = (phi, psi)`, `K` and `g`
  ladders: no eligible establishment-limited cell anywhere, so no FR run was justified.
  `E_cond` sat at its own finite-`K` floor for `K >= 128` — no hidden-`psi` deficit.

Two lessons crystallized here, and they drove everything after:

> **Lemma.** Marginal FR and count balancing realize the *same* flow and differ only in
> the estimator of `log p(xi)`.  A tie at `K = 1024` in 1-2D is the correct answer, not
> bad luck; FR can only separate where the density estimator binds, i.e. descriptor
> dimension >= 3.

> **Corollary.** The ABP *owns* the `xi`-marginal — its own occupancy flooding performs
> the marginal correction FR was meant to supply.  The only regime left for a population
> method is a coordinate the bias does not act on.

### Turn 3 — the reframe (Phase F): Type C

A four-type taxonomy replaced the original `T_hit`/`T_est` gate:

| type | what limits the estimator | what fixes it |
|---|---|---|
| **A** | oscillatory establishment transient | damping: FR *or* gain tuning *or* count balancing |
| **B** | adaptation rate | raise `g_SHUS` (up to `-73.7%`) |
| **C** | `p(z|xi)` for an unbiased coordinate `z` | nothing the bias can do |
| **D** | discovery / estimator noise | more walkers or longer runs |

`bichannel.py` was built to exhibit **C** on purpose: two channels spanning `phi` with a
`psi`-barrier exactly orthogonal to it, `Ha = Hb` so the channel ratio is analytic.  It
delivered the campaign's first deficit that base-method tuning cannot touch:
`e_F(T)` at **49-93 x** the mollifier floor while the ABP's own marginal gate reports
convergence at `t = 0`, and an eightfold gain increase moves it by `<= 3%`, wrong sign.

Fiber-wise (conditional) reallocation — the same FR step applied to `p(z | xi)` inside
`xi`-strata, leaving the `xi`-marginal invariant at stratum resolution — gave

* `dI_F = -15.3% / -12.6% / -31.4%`, every CI below zero;
* **marginal** FR at the same dose: `+0.19% / -0.04% / -0.18%` (blind by construction —
  two walkers at the same `phi` in different channels get the same score);
* matched-turnover sham: null;
* stratified **count** balancing: ties conditional FR again (the fourth replication).

A companion measurement gave the mechanism and a *pre-run* diagnostic: `tau_clone` is
at the first recorded lag in the CV (one event stride) and **160-290 strides in the
hidden channel**, in the
same runs.  A clone forgets the coordinate the bias already controls almost immediately
and remembers the rare channel for a long time — which is exactly the condition under
which cloning can carry information.

### Turn 4 — the baselines Phase F was missing (F3a, F3b, F4)

* **Augmented CV.**  Just add `psi` to the biased CV.  Same walkers, same steps, same
  noise, scored on the same reduced `F(phi)`: `-83% / -83% / -80%` versus `-15/-13/-31%`,
  reaching `2.1-2.6 e*` where conditional FR sits at `18-78 e*`.  Directly: the augmented
  CV beats conditional reallocation by **-71% to -81%**.
* **Sample complexity.**  The obvious defence — a `96 x 96` accumulator needs walkers —
  fails: the margin moves only from `-80.4%` (`K = 1024`) to `-74.4%` (`K = 64`).  An
  ABP's accumulator is filled by **trajectory time**, not population size.
* **Target.**  Re-declaring "uniform" in `psi' = psi +- 0.8 sin psi` — an arbitrary
  reparametrization of the descriptor — moved the result from `-14.9%` to **`+5.1%`** at
  comparable turnover.  The oracle target gave `-64% / -54%`.  Structurally: `q/p_hat` is
  reparametrization-invariant only if `q` is a density in a *fixed* measure, so "uniform"
  is a modelling choice on the same footing as choosing a CV, and F4 measured what a bad
  one costs: all of it, with the wrong sign.

### Turn 5 — the mechanism (Phases I and J)

**Phase I** separated the two jobs the equal-weight step conflates.  Carry statistical
weights through the *identical* selection — same score, same `theta`, same draw, so the
arms are dose-matched by construction — with descendants of a score-driven resample
taking `W_k / (cnt_j w_k)`, renormalized to hold each stratum's total weight exactly.
The score then allocates computational effort; the weights keep the represented law.

| | equal weight | weighted |
|---|---|---|
| spread over three choosable targets | 72.9 / 40.0 points | **1.52 / 2.93** |
| uniform target vs its own sham | `-14.7% / -29.2%` | `+1.2% / -0.5%` |
| **oracle target** | **`-62.9% / -55.4%`** | **`-2.8% / -0.6%`** |

The decoupling is visible in the same rows: `wfr_cond_oracle` puts **66% of its
particles** in the rare channel while the population it *represents* stays at the
plain-SHUS value.  One apparent survivor (a `-30%` hot-dose oracle arm) was traced to an
`O(1/walkers-per-stratum)` ratio bias in the weight bookkeeping — measured with no
dynamics at `+0.0142` (32/stratum) vs `+0.0032` (128/stratum) — and vanished when the
per-stratum sample size was quadrupled (`-31.8% -> +0.09%` against its sham) while the
particle allocation got *stronger*.

**Phase J** tested the complementary regime, where selection methods are actually built
to work: the conditional **correct in expectation** but resolved by too few walkers.
Starting the ensemble at the exact stationary law of the converged bias and warm-starting
the accumulator at its fixed point produced the campaign's first variance-dominated cell
(every cell's represented conditional within 0.75 of a binomial sd of the exact one,
**77-92%** of the residual MSE seed scatter).  Measure-preserving allocation still bought nothing: no arm improved
the deliverable; the one variance reduction (`-31%`, gentlest dose, exchange-active cell)
did not replicate on the slower cell and was mostly *not direction* — the equal-weight
sham, which allocates nothing, already buys `-21%`.  Stronger allocation lost more to
weight degeneracy (`ESS_w` 0.51-0.66, and 0.03 at equal-count-per-state) than it gained.

And the mirror image, on the same system: the equal-weight arm that gained `-15 to -31%`
when the conditional was wrong **costs `+24%` and `+88%`** when it is right, with its
`bias^2` up 276% and 2676% while its *variance falls*.  One operation, applied to a wrong
and to a right conditional.

---

## 4. The speed map

`I_F` mixes how fast the error fell with how low it ended.  Every stored run was rescored
with the campaign's other frozen endpoint (`tau_eps`, 0.2 T persistence, right-censored;
`S_eps = tau^base / tau^arm`) on a ladder of thresholds in units of each cell's `e*`.
No new simulations.

| comparison | `S_eps` |
|---|---|
| gateway FR vs **untuned** SHUS | **1.29 -> 1.03** as the rung tightens |
| gateway **count** vs untuned SHUS | matches FR to two decimals at every rung |
| gateway sham | 1.00 exactly |
| gateway FR vs **tuned** SHUS | 1.06, 0.87, 1.29, 1.11, 0.97 — inconsistent in sign |
| torus FR on a tuned baseline (`g* = 8`) | **1.00** at every rung (untuned run: 0.13-0.35) |
| Type-C conditional FR | **1.57 / 3.66** (count ties; marginal FR and sham exactly 1.00) |
| Type-C **augmented CV** | **6.25 / 7.31** |
| Type-C **oracle target, equal weight** | **4.84 / 5.37** |
| Type-C **oracle target, weighted** | **1.04 / 1.00** |

The last two rows are the campaign in two lines: the same selection, the same
information, the same dose — and the acceleration exists only when the target is
permitted to move the represented law.

Three claims had been travelling together under the word "faster", and they separate:

* **A. Did the error curve sit lower?**  Yes — gateway FR, Type-C conditional FR.
* **B. Did it reach a fixed accuracy sooner?**  Yes — `1.03-1.29x` (gateway, untuned),
  `1.6-3.7x` (Type C).  Both real.
* **C. Is that a target-free acceleration, after controlling for base-method tuning and
  for changes of the represented law?**  **No evidence, anywhere in the campaign.**

---

## 5. What the campaign established

1. **A limitation taxonomy for adaptive biasing, with interventions attached.**  Types
   A/B/C/D above.  `T_hit` and `T_est` do not distinguish them: the 2D torus cell that
   looked establishment-limited was adaptation-rate limited, and reallocation was inert
   there while the gain was worth `-73.7%`.
2. **Marginal convergence can be badly misleading.**  On the Type-C system,
   `D_KL(p_xi || u) ~ 0` — the ABP's own convergence gate reports success at `t = 0` —
   while `e_F` sits at 49-93 estimator floors because `p(z | xi)` has not mixed.  Any
   adaptive-biasing practitioner reading a marginal-flatness diagnostic should know this
   failure exists.
3. **`tau_clone` is a pre-run test for whether cloning can help.**  Selection can only
   transmit information the descendants retain: measure the descriptor's clone
   decorrelation time against the event stride *before* building a method on it.  Under
   one stride in the CV (WCA and bichannel alike) means cloning on it is redundant;
   160-290 strides in the hidden channel means it is not.
4. **Population selection requires knowing what should be over-represented.**  Equal-weight
   reallocation converges toward its target, so its accuracy *is* the target's accuracy.
   With a good target it is fast (up to `5.4x`); with an arbitrary reparametrization of
   the same descriptor it reverses sign.  It does not discover the missing conditional.
5. **Making the step measure-preserving is safe and, here, inert.**  Weights remove a
   20-40 point target risk for 1-2% cost, and remove the benefit with it — in the
   bias-limited regime (Phase I) and the variance-limited one (Phase J) alike.
6. **When the limiting coordinate is known and biasable, bias it.**  `-71 to -81%` in
   `I_F` and `4x` in time-to-accuracy over the best reallocation arm, at identical cost.

---

## 6. Method-level warnings (transferable, and mostly learned the hard way)

* **A matched-turnover sham is not a valid null once walkers carry weights.**  Random
  weight-conserving churn fragments weights multiplicatively; the weighted sham's
  `ESS_w` random-walked to 0.22 and it degraded `I_F` by `+88-93%`.  Every weighted arm
  scored against it looked like a `-64 to -92%` win; none of it was real.  Match ESS
  degradation, or use an analytically neutral scheme — not the number of births.
* **Seed-variance endpoints in selection experiments carry a churn null.**  Resampling
  couples seeds through shared ancestry, so *any* resampling shrinks seed-to-seed
  scatter — about `-20%` here, before any allocation.
* **"Uniform" is a choice of reference measure, not a canonical target.**  It moves under
  reparametrization of the descriptor; the only invariant conditional target is the
  unknown physical one.  Tempering is not an escape (tempering a target is algebraically
  equivalent to reducing `theta`).
* **Preregister metric floors before interpreting them.**  `E_cond` has a KDE-smoothing
  floor that consumed the entire apparent alanine deficit; the analytic mollifier floor
  `e*` is computable before any run and every error is quoted in units of it.
* **Per-stratum weight renormalization is a ratio estimator.**  It holds the marginal
  exactly and pays an `O(1/walkers-per-stratum)` bias *toward the target*.  Measure it
  with the dynamics switched off; it is otherwise indistinguishable from a result.
* **A warm-started run has no convergence phase.**  It starts at the fixed point, is
  driven up to its own sampling-noise level and relaxes back, so `tau_eps` is undefined
  and the comparable quantity is the late-run noise level.

---

## 7. Ruled out, and still open

**Ruled out by preregistered experiment:** marginal FR as a general addition to a tuned
ABP; FR as a substitute for walkers (WCA, every `K`); FR wherever `T_hit << T_est` (that
cell was adaptation-rate limited); smooth FR beating count balancing in 1-2D (five
replications); clone redundancy as the WCA limiter; conditional reallocation as
preferable to biasing the hidden coordinate; a low-`K` crossover at `d_z = 1`; the
uniform conditional target as robust; and target-free acceleration in either the
bias-limited or the variance-limited regime.

**Not tested, and honestly so:**

* **Non-differentiable descriptors** — birth-death needs only to *evaluate* `z`, biasing
  needs its gradient.  Hydrogen-bond counts, hard-cutoff contacts, cluster labels.  This
  is the strongest remaining argument for the method's *applicability*, though not for
  its performance, and it is weakened by the fact that conditioning and biasing require
  the same knowledge of *which* coordinate is slow.
* **Descriptor dimension `d_z >= 3`** (Phase G), where a smooth conditional estimator
  might beat fixed bins and where a dense augmented accumulator becomes infeasible in
  memory rather than merely under-sampled.  That is a computational argument, weaker than
  the statistical one the phase was opened on, and it was not run because it would refine
  a step with no demonstrated benefit.
* **The reaction law itself** (Phase H: FR vs convex mixture vs a `chi^2`-type score at
  matched target, KDE, event times and turnover).  Same reason.
* **Rare-event fluxes and rates.**  Everything here scores a free-energy profile.
  Weighted-ensemble methods have established mathematical grounds for helping with flux
  and rate observables, and nothing in this campaign speaks against that — it is a
  different observable and would be a new project with its own preregistration.

---

## 8. If you are running an adaptive-biasing calculation

1. **Tune the base method first.**  The adaptation rate was worth `-9%` to `-74%` on
   different systems and was mistaken for a population problem twice.
2. **Diagnose the deficit type** (A/B/C/D) before reaching for a population method, and
   do not trust marginal flatness: check a conditional diagnostic against its own
   finite-`K` floor.
3. **If the limiting coordinate is known and biasable, add it to the CV.**  Nothing else
   in this campaign came close.
4. **If it is not biasable**, conditional reallocation is a fallback that works when you
   can defend the target on independent grounds — symmetry-related states, discrete states
   with known relative free energies, a validated prior.  Measure `tau_clone` on the
   descriptor first, and carry statistical weights unless you are prepared to defend the
   target as a physical claim, because that is exactly what an equal-weight step turns it
   into.
5. **In low descriptor dimension, count balancing is as good as any smooth estimator.**
   Five head-to-head replications, two campaigns.

---

## 9. Reproducing any of it

```bash
pip install -e ".[dev]"
python -m pytest                                        # 143 tests, ~40 s
python scripts/run_appmap_phaseF1_screen.py             # the Type-C screen
python scripts/run_appmap_phaseF2_realloc.py            # conditional reallocation
python scripts/run_appmap_phaseF3a_augcv.py             # the augmented-CV baseline
python scripts/run_appmap_phaseF4_target.py             # target sensitivity
python scripts/run_appmap_phaseI_weighted.py            # weighted selection
python scripts/run_appmap_phaseI2_hotdose.py            # the bookkeeping-bias control
python scripts/run_appmap_phaseJ_variance.py --scan     # the rarity/relevance scan
python scripts/run_appmap_phaseJ_variance.py --screen   # the variance-limited screen
python scripts/run_appmap_phaseJ_variance.py            # the variance experiment
python scripts/analyze_speed_map.py                     # the speed map (no simulation)
python scripts/analyze_appmap_phaseJ_vs_shus.py         # Phase J re-scored vs SHUS
```

Each run writes one `.npz` + `.json` per row under `results/`, carrying the full PMF and
marginal time series, so any later change of reference, window or metric can be rescored
without re-running dynamics.  Every phase's design was committed *before* its first row
existed; the commit history is the audit trail.
