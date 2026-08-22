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

**Against adaptive biasing: sometimes, and predictably.**
**Against classical stratification: no, except when an exact analytic model of the
fiber is available — which is precisely when the problem was not hard.**

The mechanism claim (`W = discovery, FR = establishment`) is **confirmed** and is
quantitatively sharp at the marginal level. The free-energy claim fails for a reason
that is structural rather than numerical, and that is worth stating on its own:

> Any move that changes `xi(q)` without knowing `F` cannot be Metropolis-corrected,
> because the acceptance ratio for the target `u(z) nu^xi(dq|z)` contains
> `exp(+beta F(xi(q)))`. Replica exchange avoids this by swapping between two
> **occupied** windows, where the unknown weights cancel identically. RC-WFR instead
> moves unconditionally and does not correct — buying CV transport at the price of a
> hysteresis bias whose magnitude is set by `kappa * tau_fiber` summed over every fiber
> mode, including the slowest one, which is exactly the mode that made physical CV
> transport slow in the first place.

So RC-WFR converts a *convergence* problem into a *bias* problem. More compute fixes
the first and not the second.

---

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

**F1. The lift is the whole problem, and it is irreducible.** With an *oracle* lift
(exact conditional refresh) RC-WFR sits at 1.0–1.1× the estimator floor at every
transport rate from `kappa = 0.03` to `8.0`. With the implementable *identity* lift the
error grows monotonically with `kappa` (3.3× → 28× the floor) and is **independent of
`n_cond`** — it is a steady-state hysteresis, not a post-jump transient, so it cannot
be removed by relaxing longer after each move. (Figure 2a.)

**F2. A model-based lift only repairs the modes it models.** Rescaling the fiber
coordinate by `omega(x)/omega(x')` — exact for a harmonic fiber — restores RC-WFR to
1.5× the floor on `EB` and makes it the best non-oracle arm there. On the hidden-channel
system, where the slow mode is *which* channel is occupied, the same lift makes the
error **1.1–1.7× worse than doing nothing**. (Figure 2b.)

**F3. Fast CV transport and correct conditional sampling are in direct conflict.** On
the hidden-channel system RC-WFR's channel error *grows* with its own transport rate
(0.12 at `kappa = 0.03` to 0.45 at `kappa = 8`): walkers are dragged through the switch
region faster than the slow mode can equilibrate. The move that buys coverage destroys
the conditional law the estimator needs.

**F4. Replica exchange does the same job exactly.** Cold-start Hamiltonian RE-TI repairs
the hidden channel completely (`chan` 0.297 → 0.032, matching the oracle arms) at the
same budget and with no bias.

**F5. The hoped-for large-system crossover does not exist.** Exchange acceptance decays
only slowly with fiber size (0.975 → 0.899 over `m_spec = 0 → 128`), while RC-WFR's
lift bias grows with *every* dragged mode, so the gap widens: RC-WFR goes from +33% to
+276% worse than RE-TI. (Figure 5b.)

**F6. Smooth Fisher–Rao is not needed.** Plain count balancing ties smooth FR
(0.03502 vs 0.03612), while the matched-turnover sham is 2.3× worse. The *direction* of
the reallocation matters; its Fisher–Rao geometry does not. This reproduces the ABF/ABP
result in a setting with no adaptive bias to be redundant with, which makes it a
property of the uniform target rather than of the host method.

---

## 5. The one genuinely useful algorithmic finding

Replacing the stochastic Wasserstein step by the **deterministic probability flow**

    Z <- Z - kappa * dtau * grad log p_hat(Z)

changes the bias picture qualitatively, because the velocity vanishes as `p -> u`: the
hysteresis self-annihilates once the marginal is flat. On `EB` it holds `e_F` near 2×
the floor across `kappa = 0.03 ... 2.0` where the SDE form degrades from 3.2× to 28×.
This is also the formulation that connects to the Gaussian-mixture picture, since
`grad log p` is analytic for a GMM and needs no KDE differentiation.

Two caveats found with it, both structural:

* it cannot start from a single structure (the score of a delta ensemble vanishes at the
  particles, so the ensemble never moves), and
* deterministic transport plus FR resampling is degenerate — clones follow identical
  trajectories and coverage collapses to 0.33–0.44. The standard SMC *resample–move*
  jitter repairs it, but only in a narrow window (`sigma = 0.01` optimal; `sigma = 0.05`
  reintroduces SDE hysteresis and costs a factor 6).

---

## 6. Where this leaves the idea

RC-WFR is a **correct and well-behaved marginal sampler bolted onto an incorrect
conditional transport**. Its useful regime is narrow and identifiable in advance:

| condition | RC-WFR |
|---|---|
| CV domain long relative to physical CV diffusion | beats ABF, grows with L |
| high enthalpic or entropic barrier the bias must learn | beats ABF and SHUS |
| fiber has an exact, cheap analytic model | competitive with, or better than, stratified TI |
| fiber has a slow mode not in the model | loses to cold-start RE-TI, often to plain fixed-window TI |
| system size grows | loses further; the lift bias is extensive in fiber modes |
| only one starting structure is available | the flow form cannot start; the SDE form pays full hysteresis |

The honest positioning is not "a faster free-energy method". It is:

> **Reaction-coordinate WFR is a continuum, grid-free way to GENERATE stratified
> window configurations by continuation, whose free-energy estimate is biased by
> exactly the fiber modes it drags. Where a cheap exact lift exists it is competitive
> with stratified TI; where one does not, Hamiltonian replica exchange achieves the
> same CV-space mobility without the bias, and should be preferred.**

## 7. What would have to be true for the idea to work

A lift that is *asymptotically exact without knowing F*. The campaign found only three
kinds: the oracle (not implementable), a model-based rescaling (only repairs modelled
modes, and damages unmodelled ones), and annealing `kappa -> 0` (removes the bias only
by removing the transport, converging to stratified TI). If a fourth exists it is the
thing to look for; nothing in the structural argument of Section 2 forbids it, but any
candidate must supply the missing `exp(+beta F)` weight from somewhere other than an
estimate of `F` — and the only known mechanism that does is exchange between occupied
windows.
