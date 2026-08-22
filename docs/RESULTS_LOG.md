# Running results log (append-only; every entry names the script that produced it)

## Phase 0 - marginal WFR operator validation  (`scripts/phase0_marginal.py`)

Domain [-1,1], N = 65536 particles, kappa = 0.05, lambda = 5, T = 1, bw = 0.03.
Particle KL(p_t||u) vs the explicit-Euler solution of
`d_t p = kappa Lap p - lambda p (log p - E_p log p)` smoothed by the same kernel.

| arm     | KL(T) particle | KL(T) PDE | median rel. dev. |
|---------|---------------:|----------:|-----------------:|
| W only  | 0.33314        | 0.33377   | **0.28%**        |
| WFR     | 0.00026        | 0.00014   | **3.8%**         |
| FR only | 0.37733        | 0.00099   | 320%  (expected) |

The FR-only mismatch is the POINT, not a bug: the Eulerian PDE has p > 0 everywhere
so its FR term converges to uniform, while the particle FR cannot move mass to where
there are no particles.

**FR cannot expand support (exact).**  FR-only, 500 iterations, lambda = 5:
particle support width 1.1367 -> 1.1367 at bw = 0.20 AND at bw = 0.02 (unchanged to
4 decimals).  KL stalls at 1.167 / 1.623.

**Domain-size scaling** - time to KL < 0.05, kappa = 0.25, lambda = 5, domain [-L, L]:

| L   | W only | FR only | W+FR |
|-----|-------:|--------:|-----:|
| 1   | 0.54   | never   | 0.26 |
| 2   | 2.38   | never   | 0.54 |
| 4   | 9.64   | never   | 1.02 |
| 8   | 38.52  | never   | 2.06 |

W scales as **L^2** (ratios 4.4, 4.05, 4.00); W+FR scales as **L** (ratios 2.08, 1.89,
2.02) - a reaction-diffusion front of speed ~2 sqrt(kappa lambda) instead of diffusive
relaxation.  FR alone never converges.

=> The mechanism claim `W = discovery, FR = establishment` is CONFIRMED at the marginal
level, and it predicts a free-energy advantage that GROWS with the CV domain size.
That prediction is what Phase 1 must test against ABF (whose CV equilibration is also
diffusive, hence O(L^2)) and against stratified TI (whose coverage is O(1) by
construction).

## Estimator floor calibration  (2026-08-22)

`e_F` reached by 2^24 i.i.d. ORACLE samples (Z ~ u, Y ~ nu^xi(.|Z)) pushed through the
shared binned mean-force estimator.  This is a SYSTEMATIC floor (kernel smoothing of a
curved F'), not a variance floor: it does not decrease with more samples.

| grid G | bw_mf | floor e_F |
|--------|-------|-----------|
| 181    | 0.07  | 0.0444    |
| 361    | 0.07  | 0.0438    |
| 721    | 0.07  | 0.0436    |
| 361    | 0.04  | 0.0152    |
| 361    | 0.02  | 0.0040    |
| 721    | 0.01  | 0.0009    |

Floor ~ bw_mf^2, essentially independent of grid resolution.

**This invalidated the first smoke comparison**: with bw_mf = 0.07 every stratified arm
(WFR 0.038, TI-warm 0.043, TI-cold 0.039, RE-TI 0.038) was sitting AT the floor, so
their differences carried no information.  Only ABF (0.246) and SHUS (1.05) were
resolvable.

**FROZEN numerical convention**: domain [-1.8, 1.8], G = 361, eval window [-1.5, 1.5],
`bw_mf = 0.02`, `n_min = 1.0`  =>  floor e_F = 0.0040.  Every reported error is quoted
against that floor.

## Phase 1a - lift-bias audit  (EB, N=256, 10.24M force evals, theta=0.6)

`identity` lift = carry the fiber configuration across the W move unchanged (the only
thing implementable without knowing nu^xi).  `oracle` lift = redraw Y from the exact
conditional at the new fibre (not implementable; an upper bound).

| lift     | n_cond | kappa | I_F     | e_F_final | / floor |
|----------|-------:|------:|--------:|----------:|--------:|
| identity |      5 | 0.03  | 0.01677 | 0.01305   |  3.3x   |
| identity |      5 | 0.125 | 0.03741 | 0.03225   |  8.1x   |
| identity |      5 | 0.5   | 0.07572 | 0.07305   | 18.3x   |
| identity |      5 | 2.0   | 0.11182 | 0.11043   | 27.6x   |
| identity |     20 | 0.125 | 0.03465 | 0.03156   |  7.9x   |
| identity |    100 | 0.125 | 0.04036 | 0.03093   |  7.7x   |
| oracle   |      5 | 0.03  | 0.00827 | 0.00445   |  1.1x   |
| oracle   |      5 | 2.0   | 0.00450 | 0.00444   |  1.1x   |
| oracle   |    100 | 2.0   | 0.00584 | 0.00540   |  1.3x   |

**Finding L1.**  With the oracle lift RC-WFR reaches the estimator floor at EVERY
kappa - the marginal WFR machinery itself is sound and carries no intrinsic bias.

**Finding L2.**  With the implementable identity lift there is a SYSTEMATIC bias floor
that grows monotonically with the transport rate kappa (3x -> 28x the estimator floor
over kappa = 0.03 -> 2.0) and that MORE COMPUTE DOES NOT REMOVE.

**Finding L3.**  That bias is independent of n_cond at fixed kappa (5 / 20 / 100 give
the same e_F_final to 2 significant figures).  It is therefore not "not enough
relaxation per jump": it is the continuum hysteresis of a fibre measure being dragged
at rate kappa through a fibre that relaxes at rate 1/tau_fiber.  The only control on it
is kappa itself - i.e. **RC-WFR cannot buy CV transport without paying free-energy
bias**, which is the central practical limitation of the method.

## Phase 1b - baseline calibration on EB (`scripts/calibrate.py`, 4 seeds, 10.24M fe)

| arm        | best knob         | I_F     | e_F_final |
|------------|-------------------|--------:|----------:|
| reti_warm  | n_ex = 5          | 0.00589 | 0.00550   |
| ti_warm    | -                 | 0.00644 | 0.00555   |
| ti_cold    | -                 | 0.01061 | 0.00625   |
| **wfr (best identity lift)** | kappa 0.03, n_cond 5, theta 0.6 | 0.01677 | 0.01305 |
| abf        | bias_n_min = 1    | 0.04088 | 0.00604   |
| shus       | gain = 1 (none help) | 1.494 | 1.308    |
| unbiased   | -                 | 0.90578 | 0.90388   |

**Finding E1.**  On the easy system RC-WFR beats ABF by 2.4x in `I_F` - but EVERY
classical stratified baseline beats RC-WFR, including plain cold-start fixed-window TI.
The advantage over ABF is an advantage of STRATIFICATION, which fixed-window TI already
delivers for free and without any lift bias.  **H2 fails on EB.**

**Finding E2 (SHUS).**  Not a bug: the EB barrier is 23.2 kT and SHUS is a
histogram-filling ABP.  At gain 1e6 it has filled F_hat to 6.9 of the true 11.1 within
the budget.  A force-based method (ABF) is simply far more efficient on a high
enthalpic barrier.  ABF is used as the primary adaptive-biasing baseline.

## Phase 2 - hidden-channel system (CHANNEL), 25.6M force evals, 16 seeds

System: two fiber channels (y_1 > 0 / y_1 < 0) whose CORRECT occupancy runs
P(y>0|x): 1.0 at x = -1.4  ->  0.0 at x = +1.4.  The channels interconvert only near
x_sw = 0.  Measured sign-change time at fixed x:

| x            | 0.0   | 0.3  | 0.6  | 1.2  |
|--------------|------:|-----:|-----:|-----:|
| tau_switch   | 0.025 | 0.15 | 16.7 | 83.3 |

so the switch region is |x| < 0.4 and the budget is T = 100.  Estimator floor 0.0035.

| arm         | I_F     | e_F_final | /floor | chan  | cov   |
|-------------|--------:|----------:|-------:|------:|------:|
| wfr_oracle  | 0.00378 | 0.00356   |  1.0   | 0.033 | 1.000 |
| ti_cold     | 0.22149 | 0.16112   | 46.6   | 0.297 | 1.000 |
| wfr         | 0.22450 | 0.18477   | 53.4   | 0.172 | 1.000 |
| w_count     | 0.24052 | 0.18656   | 53.9   | 0.201 | 1.000 |
| w_only      | 0.26080 | 0.18042   | 52.1   | 0.198 | 1.000 |
| fr_only     | 0.64543 | 0.64543   | 186.6  | 0.000 | 0.022 |

**Finding C1 (the mechanism inverts).**  RC-WFR's hidden-channel error GROWS with its
own transport rate.  From the kappa sweep (8 seeds, same budget, n_cond = 5):

| kappa | 0.03 | 0.125 | 0.5  | 2.0  | 8.0  |
|-------|-----:|------:|-----:|-----:|-----:|
| chan  | 0.12 | 0.20  | 0.31 | 0.40 | 0.45 |
| e_F   | 0.13 | 0.18  | 0.25 | 0.25 | 0.34 |

Faster transport in z drags walkers through the switch region before the slow fiber
mode can equilibrate, so the very move that buys CV coverage destroys the conditional
law the estimator needs.  **Fast CV transport and correct conditional sampling are in
direct conflict**, and the conflict is governed by the SLOWEST fiber mode - the same
mode that made the physical transport slow in the first place.

**Finding C2 (the lift is the whole story).**  `wfr_oracle` - identical marginal WFR
dynamics, exact conditional refresh - sits at 1.0x the estimator floor at EVERY kappa
from 0.03 to 8.0 and at every theta.  All of RC-WFR's error is lift hysteresis, none of
it is the WFR flow.

**Finding C3.**  `fr_only` confirms Phase 0 in the free-energy setting: coverage 0.022,
`chan` trivially 0 (it never leaves the start), I_F 187x the floor.  FR alone cannot
discover.

**Finding C4 (H4, geometry).**  `wfr` 0.2245 vs `w_count` 0.2405 - the smooth
Fisher-Rao score and plain count balancing are within a few percent, consistent with
the identity `uniform-target FR with histogram density == count balancing` (unit test
`test_uniform_target_fr_equals_count_balancing_in_the_histogram_limit`, r > 0.99).

## Phase 2 (cont.) - the decisive CHANNEL table, 25.6M force evals, 16 seeds

| arm         | I_F     | e_F_final | /floor | chan  | note                    |
|-------------|--------:|----------:|-------:|------:|-------------------------|
| wfr_oracle  | 0.00378 | 0.00356   |   1.0  | 0.033 | exact lift, not usable  |
| reti_warm   | 0.00953 | 0.00674   |   1.9  | 0.029 | oracle-initialized      |
| ti_warm     | 0.01152 | 0.00551   |   1.6  | 0.025 | oracle-initialized      |
| **reti_cold**  | **0.12974** | **0.03897** | **11.3** | **0.032** | acc 0.975; NO oracle |
| wfr (annealed, best) | 0.18993 | 0.09885 | 28.0 | 0.082 | kappa 0.5 -> 0.003   |
| ti_cold     | 0.22149 | 0.16112   |  46.6  | 0.297 |                         |
| wfr         | 0.22450 | 0.18477   |  53.4  | 0.172 |                         |
| w_count     | 0.24052 | 0.18656   |  53.9  | 0.201 |                         |
| w_only      | 0.26080 | 0.18042   |  52.1  | 0.198 |                         |
| fr_only     | 0.64543 | 0.64543   | 186.6  | 0.000 | coverage 0.022          |

**Finding C5 (the decisive one).**  On the system built specifically to reward
CV-space population mobility, cold-start Hamiltonian replica exchange REPAIRS the
hidden channel completely (chan 0.297 -> 0.032, the same value the oracle arms reach)
and reaches e_F = 0.039.  RC-WFR, with the same information and the same budget,
reaches e_F = 0.185 (0.099 with its best kappa anneal) and chan = 0.172 (0.082).
**RE-TI beats the best RC-WFR configuration by 4.7x (1.9x annealed) in final error.**
The reason is structural: RE swaps between two OCCUPIED windows, so the unknown
exp(+beta F) weights cancel and the move is EXACT.  RC-WFR's unconditional move
cannot be corrected without knowing F (docs/METHOD.md), so it is biased.

**Finding C6 (annealing helps, does not rescue).**  Annealing kappa 0.5 -> 0.003 with
the mean-force accumulator reset after 30-60% of the budget cuts e_F from 0.185 to
0.069-0.099 - still 20-28x the floor and still worse than cold-start RE-TI.  Freezing
late does not repair the conditional, because repairing it needs precisely the slow
fiber relaxation the transport was meant to avoid.

## Phase 2b - can a BETTER lift rescue it?  (EB, 10.24M fe, 8 seeds, theta=0.6)

`scaled` lift = rescale each fiber coordinate by omega(x)/omega(x'), i.e. the exact
adiabatic map for a harmonic fiber - the best lift a practitioner could build from a
known local model.

| kappa | identity e_F | scaled e_F | oracle e_F |
|-------|-------------:|-----------:|-----------:|
| 0.03  | 0.0135       | 0.0061     | 0.0044     |
| 0.125 | 0.0323       | 0.0058     | 0.0044     |
| 0.5   | 0.0729       | 0.0061     | 0.0045     |
| 2.0   | 0.1120       | 0.0090     | 0.0045     |
| 8.0   | 0.0948       | 0.0180     | 0.0045     |

**Finding B1.**  A model-based lift removes the hysteresis of exactly the modes it
models: on the purely harmonic EB fiber the scaled lift restores RC-WFR to 1.5x the
estimator floor up to kappa = 0.5.  Best scaled-lift `I_F` = 0.0069, versus ti_warm
0.0064 and reti_warm 0.0059 - **a tie with stratified TI at best, not a win**, and it
requires an analytic model of the fiber that no real system provides.

## Phase 2c - a model-based lift only helps when the model is RIGHT

`scaled` lift on the CHANNEL fiber (whose slow mode is WHICH channel is occupied, a
mode the omega-rescaling model does not contain):

| kappa | 0.03 | 0.125 | 0.5  | 2.0  | 8.0  |
|-------|-----:|------:|-----:|-----:|-----:|
| identity e_F | 0.128 | 0.184 | 0.246 | 0.245 | 0.335 |
| scaled   e_F | 0.143 | 0.267 | 0.372 | 0.411 | 0.396 |

**Finding B2.**  On EB (harmonic fiber, the model is exact) the scaled lift cuts e_F by
5x.  On CHANNEL (the slow mode is not in the model) the SAME lift makes e_F 1.1-1.7x
WORSE than doing nothing, because rescaling y distorts the channel minima at y = +-c.
A lift built from a local model therefore cannot be trusted: it repairs exactly the
modes one already understands and can actively damage the ones one does not - which
are, by construction, the modes that made the problem hard.

## Phase 2d - the deterministic probability-flow W step

`w_mode='flow'` replaces `Z <- Z + sqrt(2 kappa dtau) eta` by the probability flow
`Z <- Z - kappa dtau grad log p_hat(Z)`.  Started from a single structure it NEVER
MOVES: the score of a delta initial ensemble vanishes at the particles, so coverage
stays at 0.02 and e_F at 1.08 for every kappa and theta tested (10 configurations).
The deterministic form of the Wasserstein step therefore requires a non-degenerate
starting ensemble - a structural disadvantage relative to the SDE form, which is
exactly the regime ("we have one equilibrated structure") the method is sold for.

## Phase 3 - the best RC-WFR variant: probability-flow W + FR + resample-move jitter

Replacing the stochastic W step `Z <- Z + sqrt(2 kappa dtau) eta` by the DETERMINISTIC
probability flow `Z <- Z - kappa dtau grad log p_hat(Z)` changes the bias picture
qualitatively, because the flow velocity vanishes as p -> u: the hysteresis
self-annihilates once the marginal is flat.

EB, 10.24M fe, 8 seeds, n_cond = 5, identity lift, floor 0.00398:

| W step | fr_jitter | kappa | theta | I_F     | e_F_final | /floor | coverage |
|--------|----------:|------:|------:|--------:|----------:|-------:|---------:|
| sde    | -         | 0.03  | 0.6   | 0.02328 | 0.01255   |  3.2   | 1.00 |
| sde    | -         | 2.0   | 0.6   | 0.11738 | 0.11022   | 27.7   | 1.00 |
| flow   | 0         | 0.5   | 0.0   | 0.02802 | 0.00930   |  2.3   | 1.00 |
| flow   | 0         | 0.5   | 0.6   | 0.14675 | 0.01127   |  2.8   | **0.33** |
| flow   | 0.01      | 0.5   | 0.3   | 0.01474 | 0.00888   |  2.2   | 1.00 |
| flow   | 0.01      | 2.0   | 0.3   | **0.01397** | **0.00794** | **2.0** | 1.00 |
| flow   | 0.05      | 2.0   | 0.3   | 0.05435 | 0.05028   | 12.6   | 1.00 |

**Finding F1.**  The deterministic flow keeps `e_F` at ~2x the floor for kappa from
0.03 to 2.0 - essentially kappa-INDEPENDENT - where the SDE form degrades from 3.2x to
28x.  This is the single most important algorithmic improvement found in the campaign.

**Finding F2 (an incompatibility).**  Deterministic transport plus Fisher-Rao
resampling is degenerate: clones are exact duplicates and the flow moves them
identically, so they never separate and coverage collapses to 0.33-0.44.  The standard
SMC "resample-move" fix - a small z-jitter after each FR event - repairs it, but only
in a narrow window: sigma = 0.01 is optimal, sigma = 0.05 reintroduces the SDE
hysteresis and costs a factor 6 in e_F.

**Finding F3 (H0 supported, in the flow formulation only).**  With the jitter,
theta = 0.3 beats theta = 0 at every kappa (0.0308 -> 0.0147 at kappa = 0.5), so the
Fisher-Rao term does real work once the Wasserstein step is deterministic.  In the SDE
formulation theta helps far less because the W noise already redistributes population.

## Phase 4 - scaling tests of the two mechanism predictions

### P1: does RC-WFR's advantage over ABF grow with the CV domain length L?

Periodic torsional landscape, wells at fixed spacing 1.5, beta*dF = 9.8 per barrier,
budget 25.6M force evaluations for every arm; N is each arm's own knob at fixed budget.
Paired median relative change in I_F (negative = RC-WFR better), 8 seeds:

| L  | wells | RC-WFR vs best ABF          | RC-WFR vs best fixed TI      |
|----|------:|-----------------------------|------------------------------|
| 3  |     2 | **+191.3%** [+163.0,+209.3] | +164.9% [ +95.1,+269.2]      |
| 6  |     4 | -19.1% [-37.2,+19.5] (tie)  | **+95.8%** [ +60.7,+118.7]   |

The DIRECTION of P1 is confirmed - RC-WFR gains on ABF as the CV domain lengthens,
exactly as the O(L) vs O(L^2) marginal argument predicts.  But fixed-window stratified
TI is L-INDEPENDENT by construction (its coverage is O(1)), so the same axis that helps
RC-WFR against ABF does nothing for it against TI.

### P2: does RC-WFR overtake RE-TI as the fiber grows?  NO - the opposite.

CHANNEL fiber plus m inert-in-x-but-not-in-z spectator dofs; errors relative to
|F_ref| so the axis is comparable across m.  8 seeds, 15.4M force evaluations.

| m_spec | RE acceptance | wfr I_F_rel | ti_cold I_F_rel | reti I_F_rel | wfr vs reti      |
|--------|--------------:|------------:|----------------:|-------------:|------------------|
| 0      | 0.975         | 0.401       | 0.385           | 0.309        | +33.0% [+23,+48] |
| 32     | 0.944         | 0.195       | 0.105           | 0.108        | +78.6% [+67,+98] |
| 128    | 0.899         | 0.226       | 0.057           | 0.062        | (worse still)    |

**Finding P2-neg.**  Exchange acceptance decays only slowly (0.975 -> 0.899 over
m = 0 -> 128) while RC-WFR's lift bias grows with EVERY fiber mode it drags, so the
gap widens rather than closing.  The hoped-for crossover does not exist in this family.
RC-WFR does beat ABF by a wide margin here (-86.5% at m = 32) - but only because a
large entropic barrier is very hard for ABF, and stratified TI beats both.

## Phase 5 - frozen confirmation, EB (32 fresh seeds, 10.24M force evals, floor 0.00403)

`scripts/confirm.py --system EB --steps 40000 --seeds 32`

| arm         | I_F     | e_F_final | /floor | note |
|-------------|--------:|----------:|-------:|------|
| wfr_oracle  | 0.00456 | 0.00444   |  1.1   | exact conditional refresh (upper bound) |
| reti_warm   | 0.00627 | 0.00551   |  1.4   | oracle-initialized |
| ti_warm     | 0.00669 | 0.00566   |  1.4   | oracle-initialized |
| **wfr_scaled** | **0.00711** | 0.00615 | 1.5 | needs an exact analytic fiber model |
| reti_cold   | 0.00999 | 0.00612   |  1.5   | acc 0.983; no oracle |
| ti_cold     | 0.01140 | 0.00639   |  1.6   | no oracle |
| wfr_flow    | 0.02858 | 0.01004   |  2.5   | uncalibrated kappa/theta, see Phase 6 |
| w_count     | 0.03502 | 0.03283   |  8.1   | |
| wfr         | 0.03612 | 0.03362   |  8.3   | |
| wfr_anneal  | 0.03767 | 0.01265   |  3.1   | |
| abf         | 0.04138 | 0.00749   |  1.9   | |
| w_only      | 0.07650 | 0.03656   |  9.1   | |
| w_sham      | 0.08185 | 0.03786   |  9.4   | |
| fr_only     | 1.03684 | 1.03676   | 257.2  | coverage 0.067 |
| shus        | 2.48788 | 0.78684   | 195.2  | coverage 0.644 |

**Finding H0 (mechanism, SUPPORTED).**  `wfr` (0.0361) beats `w_only` (0.0765) and
`fr_only` (1.037).  The W+FR decomposition does what it claims: W discovers, FR
establishes, and neither alone suffices.

**Finding H4 (geometry, FAILS for count, PASSES for sham).**  `w_count` 0.03502 vs
`wfr` 0.03612 - count balancing is a hair BETTER than smooth Fisher-Rao, i.e. an exact
tie; `w_sham` (0.08185) is 2.3x worse, so the DIRECTION of the reallocation matters
but its Fisher-Rao geometry does not.  This reproduces the ABF/ABP campaign's finding
in a setting with no adaptive bias to be redundant with, which strengthens it: the
result is a property of the uniform target, not of the host method.

## Phase 5b - EB confirmation: paired relative change in I_F (32 fresh seeds)

`* = 95% bootstrap CI excludes 0`

| arm         | vs ti_cold                | vs reti_cold              | vs abf                    |
|-------------|---------------------------|---------------------------|---------------------------|
| wfr_oracle  | -59.2% [-61.6,-57.4]*     | -54.7% [-57.2,-52.4]*     | -88.7% [-89.4,-88.2]*     |
| reti_warm   | -43.8% [-47.1,-39.1]*     | -37.1% [-40.6,-32.6]*     | -84.9% [-85.7,-83.3]*     |
| ti_warm     | -40.3% [-43.0,-37.4]*     | -32.8% [-39.5,-28.2]*     | -83.4% [-84.5,-82.4]*     |
| **wfr_scaled** | **-36.1% [-39.9,-33.8]\*** | **-26.9% [-34.2,-23.0]\*** | -82.8% [-83.7,-80.5]*  |
| reti_cold   | -12.7% [-19.2,-2.2]*      | -                         | -75.6% [-76.4,-73.4]*     |
| ti_cold     | -                         | +14.5% [+2.2,+23.7]       | -71.1% [-73.5,-69.3]*     |
| wfr_flow    | +147.4%                   | +188.3%                   | -27.2% [-32.4,-22.1]*     |
| wfr         | +223.7%                   | +265.7%                   | -5.8% [-17.2,+3.2] (tie)  |
| w_only      | +579.0%                   | +670.7%                   | +85.9%                    |
| w_sham      | +612.7%                   | +697.8%                   | +99.4%                    |

**Finding E3.**  Given an EXACT analytic model of the fiber (`wfr_scaled`), RC-WFR is
the best non-oracle arm on EB: -36% vs cold-start stratified TI and -27% vs cold-start
RE-TI, both with CIs excluding 0.  Without such a model it ties ABF and loses heavily
to both stratified baselines.  The method's usable regime therefore requires knowing
the fiber well enough to map it between neighbouring reaction-coordinate values - which
is exactly the situation in which the sampling problem was not hard.

## Phase 6 - P1 in full: RC-WFR vs ABF vs stratified TI as the CV domain lengthens

`scripts/torsion_scaling.py`, periodic landscape with wells at fixed spacing 1.5,
beta*dF = 9.8 per barrier, 25.6M force evaluations for every arm, 8 seeds, N is each
arm's own knob at fixed budget.  Best I_F per family:

| L  | wells | RC-WFR   | ABF      | fixed TI | RE-TI    | RC-WFR vs ABF            | RC-WFR vs fixed TI |
|----|------:|---------:|---------:|---------:|---------:|--------------------------|--------------------|
| 3  |     2 | 0.00889  | 0.00305  | 0.00334  | 0.00428  | **+191.3%** [+163,+209]  | +164.9% [+95,+269] |
| 6  |     4 | 0.01254  | 0.01539  | 0.00682  | 0.00664  | -19.1% [-37,+20] (tie)   | +95.8% [+61,+119]  |
| 12 |     8 | 0.01628  | 0.05678  | 0.01187  | 0.00835  | **-72.2%** [-75,-58]     | +39.9% [+26,+94]   |
| 24 |    16 | 0.03332  | 0.19x    | 0.02540  | 0.01423  | **-82.5%** [-86,-79]     | +31.4% [+9,+67]    |

**Finding P1-pos.**  RC-WFR's standing against ABF improves MONOTONICALLY with the CV
domain length, from 3x worse at L = 3 to 3.6x better at L = 12 and 5.7x better at L = 24, exactly as the marginal
argument predicts: ABF's CV equilibration is diffusive (O(L^2)) while W+FR is a
reaction-diffusion front (O(L)).  This is the campaign's clearest positive result and it
is the direct answer to "is it better than adaptive biasing?": **yes, on a long CV
domain, and by a wide and growing margin.**

**Finding P1-caveat.**  Stratified TI also degrades with L at fixed budget - it needs
M ~ L windows for fixed CV resolution, so samples per window fall like 1/L - and it
degrades more slowly than ABF.  RC-WFR closes on it (+165% -> +96% -> +40% -> +31%) but had not
overtaken it at L = 24 in the first scan.  A fairer re-scan (probability-flow arm,
bandwidth matched to the well structure, each family free to pick its replica count,
L up to 48) is `results/torsion/torsion_scaling_flowfair.json`.

**Calibrated flow arm on TORSION L=12** (`sweep_TORSION12`, 8 seeds): flow, kappa 0.5,
theta 0.6, jitter 0.01 gives I_F = 0.01353 vs the SDE arm's 0.01628 - so the flow form
narrows the gap to fixed TI (0.01187) to about 14%.

## Phase 7 - the baselines' own screens, and the calibrated flow arm

### RE-TI screen on CHANNEL (`scripts/screen_reti.py`, 8 seeds, 25.6M fe, floor 0.00353)

Window count M trades CV resolution against window-space mobility; the exchange
period n_ex trades mobility against the energy evaluations it is charged for.

| M   | n_ex | I_F     | e_F     | chan   | acceptance |
|-----|-----:|--------:|--------:|-------:|-----------:|
| 256 |    1 | 0.16887 | 0.05763 | 0.0272 | 0.975 |
| 256 |    5 | 0.13944 | 0.04679 | 0.0313 | 0.975 |
| 256 |   20 | 0.15748 | 0.05658 | 0.0298 | 0.975 |
| 128 |    5 | 0.12610 | 0.04237 | 0.0276 | 0.950 |
| 64  |    5 | **0.12214** | **0.04118** | 0.0280 | 0.901 |
| 64  |   20 | 0.13138 | 0.04174 | 0.0286 | 0.902 |

Tuned RE-TI on CHANNEL: **I_F = 0.1221** (M = 64, n_ex = 5).  Every setting repairs the
hidden channel (`chan` ~ 0.028, the oracle value).

### The calibrated probability-flow RC-WFR arm

CHANNEL screen (`sweep_CHANNEL_flowjit`, 8 seeds): the best configuration is
flow, kappa = 0.125, theta = 0.6, fr_jitter = 0.005 ->
**I_F = 0.0733, e_F = 0.0308 (8.7x floor), chan = 0.0469** - i.e. it repairs most of
the hidden channel AND beats the tuned RE-TI baseline (0.1221) by ~40% on the screen.
Note the winning kappa is SMALL: with a deterministic, self-annihilating W step the
method wants slow, careful transport (so walkers linger in the switch region and
equilibrate the slow mode) and lets Fisher-Rao - which reallocates population WITHOUT
dragging any fiber, and therefore contributes no hysteresis at all - do the
amplification.  That is the WFR division of labour working exactly as designed.

This is a screen result on 8 seeds and must be confirmed on fresh seeds before it is
claimed; see Phase 8.

## Phase 7b - GMM score vs KDE score  ("Version A" of the Gaussian-mixture plan)

`src/rcwfr/gmm.py`: a batched 1-D Gaussian mixture with a uniform background, fitted by
WARM-STARTED EM (a fresh fit each step would make the induced Wasserstein velocity
discontinuous), supplying BOTH quantities RC-WFR needs analytically:
`p(z) = sum_k w_k N(z; m_k, s_k^2)` and `grad log p(z) = sum_k r_k(z)(m_k - z)/s_k^2`.
Validated against numerical differentiation to 3e-8 and against grid quadrature for
mass, on reflecting and periodic domains.

EB, 10.24M fe, 8 seeds, probability-flow W, fr_jitter = 0.01:

| density model | K  | kappa | theta | I_F     | e_F_final | /floor | coverage |
|---------------|---:|------:|------:|--------:|----------:|-------:|---------:|
| GMM           |  8 | 0.5   | 0.6   | 0.14054 | 0.12949   | 32.5   | **0.00** |
| GMM           | 24 | 0.5   | 0.6   | **0.01381** | 0.00831 | 2.1  | 1.00 |
| GMM           | 24 | 2.0   | 0.3   | 0.01601 | 0.01169   | 2.9    | 0.98 |
| GMM           | 64 | 0.5   | 0.6   | 0.01742 | 0.00934   | 2.3    | 0.99 |
| KDE           |  - | 2.0   | 0.3   | **0.01366** | 0.00823 | 2.1  | 1.00 |

**Finding G1.**  With enough components the GMM score reproduces the KDE score's
performance EXACTLY (0.01381 vs 0.01366 - within seed noise) at the same wall clock.
The Gaussian-mixture representation is therefore a valid, analytic, grid-free drop-in
- attractive for `d_xi >= 2` where KDE and grid differentiation get expensive - but it
does NOT change any conclusion of this campaign, because the limiting error is the
lift, not the density estimate.

**Finding G2.**  Too few components is catastrophic, not merely inaccurate: at K = 8
the mixture score is wrong enough to drive the entire ensemble into the domain walls
(coverage 0.00).  K must resolve the marginal's structure; K = 24 over a 3.6-wide
domain (component spacing 0.15) was the optimum here.
