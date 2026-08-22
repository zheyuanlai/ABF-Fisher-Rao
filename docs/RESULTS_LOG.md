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

## Phase 8 - calibrated confirmation on fresh seeds (32 seeds, `--tag _cal`)

Stage-1 winners frozen and re-run at base seed 9000, which no screen used.

### EB (10.24M fe, floor 0.00403)

| arm          | I_F     | e_F_final | /floor | note |
|--------------|--------:|----------:|-------:|------|
| wfr_scaled   | 0.00711 | 0.00615   |  1.5   | exact analytic fiber model |
| wfr_flow     | **0.01513** | 0.00834 |  2.1 | flow, kappa 2.0, theta 0.3, jitter 0.01 |
| wfr_flow_cnt | 0.01504 | 0.00879   |  2.2   | flow + COUNT balancing (ties FR exactly) |
| wfr_gmm      | 0.02139 | 0.01257   |  3.1   | GMM score, K = 24 |
| wfr_flow_w   | 0.03648 | 0.01273   |  3.2   | flow, theta = 0: **FR removed** |
| wfr          | 0.03612 | 0.03362   |  8.3   | SDE form |
| wfr_anneal   | 0.03767 | 0.01265   |  3.1   | |

**Finding F4 (H0, decisive).**  Turning Fisher-Rao OFF in the best RC-WFR variant costs
a factor **2.4** on EB (0.01513 -> 0.03648) and a factor **2.5** on CHANNEL
(0.06519 -> 0.16549).  The birth-death term is doing real, large work once the
Wasserstein step is deterministic - and it does it WITHOUT any hysteresis, because
selection copies a walker together with its fiber configuration and drags nothing.

**Finding F5 (H4 again).**  `wfr_flow_cnt` (count balancing) 0.01504 vs `wfr_flow`
(smooth FR) 0.01513: an exact tie for the third time in this campaign.  What matters is
*that* population is reallocated toward uniform, not the Fisher-Rao geometry of the
reallocation.

### CHANNEL (25.6M fe, floor 0.00346)

| arm         | I_F     | e_F_final | /floor | chan   |
|-------------|--------:|----------:|-------:|-------:|
| **wfr_flow**| **0.06519** | 0.03862 | 11.2 | 0.0402 |
| wfr_flow_w  | 0.16549 | 0.05982   | 17.3   | 0.0561 |
| wfr_anneal  | 0.19239 | 0.10397   | 30.1   | 0.1040 |
| wfr (SDE)   | 0.23933 | 0.19399   | 56.1   | 0.1677 |

Tuned cold-start RE-TI on the same system is `I_F = 0.1221` (M = 64, n_ex = 5, screen)
/ `0.14482` (M = 256, 32-seed confirmation).  **The calibrated probability-flow RC-WFR
(0.0652) is roughly 2x better than the strongest classical baseline that does not use
oracle information.**  Its winning configuration uses a SMALL kappa (0.125) and a strong
FR dose (theta = 0.6): slow, careful transport so walkers equilibrate the slow mode
while they are in the switch region, with the hysteresis-free birth-death term doing the
amplification.  This is the WFR division of labour behaving exactly as the theory says
it should.

### EB calibrated confirmation - paired comparisons (32 fresh seeds, `*` = CI excludes 0)

| arm          | I_F     | vs ti_cold             | vs reti_cold           | vs abf                 |
|--------------|--------:|------------------------|------------------------|------------------------|
| wfr_oracle   | 0.00456 | -59.2% [-62,-57]*      | -57.8% [-60,-56]*      | -88.7% [-89,-88]*      |
| reti_warm    | 0.00637 | -42.8% [-45,-40]*      | -39.7% [-42,-36]*      | -83.3% [-85,-81]*      |
| ti_warm      | 0.00669 | -40.3% [-43,-37]*      | -38.4% [-43,-35]*      | -83.4% [-85,-82]*      |
| **wfr_scaled** | 0.00711 | **-36.1% [-40,-34]\*** | **-31.4% [-42,-26]\*** | -82.8% [-84,-81]*   |
| reti_cold    | 0.01086 | -2.5% [-14,+2] (tie)   | -                      | -72.6% [-74,-70]*      |
| ti_cold      | 0.01140 | -                      | +2.6% [-2,+16] (tie)   | -71.1% [-74,-69]*      |
| wfr_flow_cnt | 0.01504 | +25.9% [+11,+52]       | +40.0% [+22,+50]       | -61.9% [-67,-59]*      |
| **wfr_flow** | 0.01513 | +40.5% [+29,+54]       | +46.0% [+33,+65]       | **-62.6% [-66,-58]\*** |
| wfr_gmm      | 0.02139 | +80.8% [+73,+109]      | +100.9% [+75,+116]     | -47.8% [-52,-42]*      |
| wfr (SDE)    | 0.03612 | +223.7%                | +235.1%                | -5.8% [-17,+3] (tie)   |
| abf          | 0.04138 | +246.4%                | +264.5%                | -                      |
| w_only       | 0.07650 | +579.0%                | +617.8%                | +85.9%                 |
| w_sham       | 0.08185 | +612.7%                | +620.4%                | +99.4%                 |
| shus         | 2.51069 | +22061%                | +22910%                | +5968%                 |

**Finding E4.**  On the easy system the model-free probability-flow RC-WFR **beats ABF
by 62.6%** (CI excludes 0) and loses to cold-start stratified TI by 40.5%.  The SDE form
merely ties ABF.  Given an exact analytic lift, RC-WFR beats every non-oracle baseline
by 31-83%.

**Finding G3.**  The GMM score arm (`wfr_gmm`, K = 24) confirms at 0.02139 - worse than
the KDE score at the same kappa/theta (0.01513).  On the screen at its own best
kappa/theta the two were identical (0.01381 vs 0.01366).  The GMM is therefore an
equal-quality but NOT a drop-in replacement: it needs its own K and kappa calibration,
and transplanting the KDE arm's settings into it costs ~40%.

### CHANNEL calibrated confirmation - paired comparisons (32 fresh seeds, floor 0.00346)

Baselines at their own screen winners: RE-TI at M = 64, n_ex = 5 (screened over 12
configurations); ABF at its screened ramp.  `*` = 95% bootstrap CI excludes 0.

| arm          | I_F     | e_F     | chan   | vs ti_cold             | vs reti_cold           | vs abf                 |
|--------------|--------:|--------:|-------:|------------------------|------------------------|------------------------|
| wfr_oracle   | 0.00377 | 0.00359 | 0.032  | -98.3%*                | -97.2%*                | -99.0%*                |
| reti_warm    | 0.01015 | 0.00636 | 0.031  | -95.5%*                | -92.4%*                | -97.4%*                |
| ti_warm      | 0.01066 | 0.00582 | 0.030  | -95.3%*                | -91.7%*                | -97.4%*                |
| **wfr_flow** | **0.06519** | 0.03862 | 0.040 | **-70.5% [-75,-50]\*** | **-50.1% [-55,-14]\*** | **-82.4% [-85,-73]\*** |
| wfr_flow_cnt | 0.07804 | 0.03581 | 0.051  | -62.7% [-74,-55]*      | -40.7% [-55,-20]*      | -80.5% [-86,-72]*      |
| reti_cold    | 0.13122 | 0.03954 | 0.030  | -40.7% [-43,-38]*      | -                      | -66.6% [-68,-65]*      |
| wfr_flow_w   | 0.16549 | 0.05982 | 0.056  | -24.0% [-27,-22]*      | **+27.5% [+23,+33]**   | -57.7% [-60,-56]*      |
| wfr_anneal   | 0.19239 | 0.10397 | 0.104  | -12.3% [-14,-8]*       | +43.4%                 | -50.6%*                |
| wfr_gmm      | 0.20254 | 0.11910 | 0.183  | -5.0% [-19,-2]*        | +53.6%                 | -49.1%*                |
| ti_cold      | 0.21816 | 0.15432 | 0.299  | -                      | +68.8%                 | -43.5%*                |
| w_count      | 0.23212 | 0.18598 | 0.171  | +6.3% [-0.2,+15]       | +75.5%                 | -41.9%*                |
| wfr (SDE)    | 0.23933 | 0.19399 | 0.168  | +11.0%                 | +82.3%                 | -39.5%*                |
| w_only       | 0.25834 | 0.18153 | 0.192  | +16.1%                 | +97.7%                 | -35.2%*                |
| w_sham       | 0.26029 | 0.18806 | 0.161  | +20.9%                 | +104.1%                | -31.9%*                |
| wfr_scaled   | 0.29172 | 0.26642 | 0.219  | +33.1%                 | +119.3%                | -27.0%*                |
| abf          | 0.39734 | 0.18966 | 0.059  | +77.1%                 | +199.0%                | -                      |
| fr_only      | 0.62760 | 0.62765 | 0.000  | +187.5%                | +375.8%                | +57.7%                 |
| unbiased     | 0.68097 | 0.71962 | 0.000  | +205.9%                | +417.6%                | +76.7%                 |
| shus         | 2.52239 | 0.35457 | 0.033  | +1023%                 | +1817%                 | +533%                  |

**Finding C7 (the campaign's headline positive).**  On a fiber whose slow mode has a
localized gateway, the probability-flow RC-WFR beats **tuned cold-start Hamiltonian
replica-exchange TI by 50.1%**, cold-start stratified TI by 70.5% and ABF by 82.4%, all
with 95% CIs excluding zero, at matched force evaluations and with the baselines
screened at least as hard as it was.

**Finding C8 (Fisher-Rao is what makes it win).**  Remove the FR term from the same
arm (`wfr_flow_w`) and it *loses* to RE-TI by 27.5% [+23, +33].  The birth-death half is
not a refinement here - it is the difference between winning and losing, and it is the
only half that carries no hysteresis, because selection copies a walker together with
its fiber configuration.

## Phase 9 - bandwidth tuning at L = 24, and where RC-WFR overtakes stratified TI

The torsion L-scan used `bw_kde = max(0.10, L/60)`, which is badly coarse at large L.
Screening it properly at L = 24 (8 seeds, 25.6M fe, floor 0.00300):

| W step | bw_kde | kappa | I_F     |
|--------|-------:|------:|--------:|
| flow   | 0.1    | 2.0   | 0.02792 |
| sde    | 0.1    | 0.5   | 0.02253 |
| **sde**| **0.3**| **0.5** | **0.02237** |
| sde    | 0.3    | 2.0   | 0.02506 |
| flow   | 1.0    | 2.0   | 0.07203 |

Best baselines at L = 24: fixed TI **0.02540** (N = 1024), RE-TI **0.01423**
(N = 1024), ABF **0.19833**.

**Finding P1b.**  With its density bandwidth tuned, RC-WFR at L = 24 is **12% better
than the best fixed-window stratified TI** and **89% better than the best ABF**, while
still 57% behind RE-TI.  The L-scan's WFR numbers are therefore conservative.

**Finding F6.**  The probability flow is NOT universally the better W step: on this
long torsional domain the stochastic step at moderate `kappa` wins (0.02237 vs 0.02792).
The flow's advantage is specific to regimes where the residual hysteresis dominates the
error; where transport itself is the bottleneck, the SDE's larger effective step wins.

---

# Manifold phase - the Chapter-3 reformulation on a NONLINEAR reaction coordinate

Everything above uses `xi(q) = x`, for which `G = 1` and the entire manifold
construction degenerates (see `docs/MANIFOLD_FORMULATION.md` §2.1). A nonlinear
test family was built to make it testable: same potentials, `xi = x + a sin(k y)`,
`Sigma(z)` a graph over the fiber, everything exact by quadrature, `a = 0`
reproducing the frozen systems bit for bit (`tests/test_manifold.py`).

## Phase M0 - does the Chapter-3 machinery reproduce the exact answer?

`scripts/validate_manifold.py`, `results/manifold/validate.json`.

| check | result |
|---|---|
| co-area normalization, `(det G)^{-1/2} dsigma = dy` | rel. error **3.8e-14** |
| local mean force, `E_nu[f] = F'(z)`, 4e6 samples, 5 values of `z` | worst deviation **1.9 SE** |
| same with the divergence term **dropped** | **3.2-4.0x the estimator floor** in `F` |
| SHAKE projection and all three lifts land on `Sigma(z)` | residual **< 2e-16** |
| constrained ambient sampler, Fixman on | PIT-KL at the histogram floor |

**Finding M1.** The Chapter-3 formulae are correct as stated and are correctly
implemented here. The divergence term of the local mean force — the one that is
identically zero when `xi` is linear, and therefore never exercised by the frozen
campaign — is not optional on a nonlinear coordinate. Pushing the exact conditional
means through the estimator by quadrature (`results/manifold/nodiv.json`), dropping
it costs up to **0.13 in `F'`** and **0.013-0.016 in `F`, i.e. 3.2-4.0x the estimator
floor**, on three (system, a, k) settings. Keeping it, the same computation returns
1e-4 to 7e-4 — below the floor, i.e. pure quadrature error, which also validates the
implementation.

**Correction.** An earlier draft of this section quoted "0.052 in `F'`, 13x the
estimator floor". That compared a MEAN-FORCE error against a FREE-ENERGY floor; the
two have different units. The corrected figure above is 3.2-4.0x. The conclusion is
unchanged in direction but the divergence term is comparable to the Fixman factor
(<= 7x, Phase M1) rather than an order of magnitude larger, and both are dwarfed by
the lift error of Phase M2.

**Finding M2 (variance).** `f_LRS` and the fiber-frame `dPsi/dz` are both valid
mean-force samples and neither dominates: `Var(f_LRS)/Var(f_graph)` ranges over
**0.047 to 1074** across `z`. At the barrier top of `EB` the ratio is 263 in favour
of the fiber frame; in the wells it is 0.35 the other way.

## Phase M1 - how much is the Fixman factor worth?

`scripts/exp_fixman.py`, `results/manifold/fixman.json`, `figures/figM4_fixman.png`.
Exact by quadrature: `F` vs `F_rgd = -beta^-1 log int e^{-beta Psi} sqrt(G) dy`.

Full grid in [`MANIFOLD_TABLES.md`](MANIFOLD_TABLES.md), max over the three
systems at each `a k`:

| `a k` | RMSE(`F - F_rgd`) | multiple of the 0.004 floor |
|---|---|---|
| 0.21 | 0.0009 | 0.2 |
| 0.42 | 0.0036 | 0.9 |
| 0.84 | 0.0118 | 2.9 |
| 1.68 | 0.0289 | 7.2 |
| 2.80 | 0.0298 | 7.4 |

**Finding M3.** The Fixman/rigid error scales as `(a k)^2` and crosses the
estimator floor at `a k ~ 0.55`, saturating at **7x the floor**. The corresponding
conditional error is at most `KL(nu_rgd || nu) = 0.018`. So the geometric factor is
**real but second-order**: it must be implemented (it is three lines) and it is
one to two orders of magnitude smaller than the lift error measured next.

## Phase M2 - the lift, and the law that governs it

`scripts/exp_lift.py`, `results/manifold/lift_a_*.json`, `figures/figM1_lift.png`.
Exact pushforward of `nu(.|z0)` through each lift map, cross-checked by 4e5-sample
Monte Carlo (agreement to 3 digits wherever the 64-bin PIT is unsaturated).

**Finding M4 (the lift-lag law).** `D_cond(z0 + dz) = C(z0, lift) dz^2 / 2 + O(dz^3)`
with `C = int [d_y(nu delta)]^2 / nu dy` and `delta = w - w*`. Measured/predicted
= 0.987, 0.982, 0.976, 1.018 at `dz = 0.0125` across four `z`, and the exact KL
scales as `dz^2` (successive ratios 3.95, 3.90, 3.80 for doublings of `dz`).

**Finding M5 (minimum-norm is not the answer).** Repeated at `(a,k)` = (0.3, 1.4),
(0.6, 0.7) and (0.6, 1.4), 24 values of `z`: `C(minnorm)/C(cartesian)` is **0.33-0.71
in the wells** — a real factor 1.4-3 — and **1.006 / 1.018 / 1.054 at the barrier
top**, marginally worse at every setting, where `C` is 6x larger than anywhere else and
where the profile is determined. At `(0.3, 1.4)`, `z = +-0.30`, the ratio is **3.65**:
the minimum-norm lift is over three times WORSE than moving one ambient coordinate.
The law itself holds throughout — measured/predicted in [0.86, 1.16] over every
`(z, a, k)`. The lift that solves the fiber continuity equation gives `C = 0` exactly,
and the measurement confirms it at all three settings: max PIT-KL over all `(z0, dz)`
is 3.0e-5, the noise floor.

**Trap M-a.** `w*` itself diverges across any low-density valley of the conditional
(moving mass through it needs unbounded velocity), so integrating `dy/dz = w*` as an
ODE fails exactly on the multimodal fibers the method is meant to help with. Use
the monotone map `y' = CDF^{-1}_{z'}(CDF_z(y))`, which is the same flow integrated
exactly. First attempt reported `D_cond = 210` for the "exact" lift; that was the
integrator, not the lift.

## Phase M3 - the timescale condition, made parameter-free

`scripts/exp_timescale.py`, `scripts/analyze_timescale.py`,
`results/manifold/timescale.json`. Ensemble of 1e5 swept at constant `dz/dt = v`
with the fiber relaxing; `v = 0` control gives a discretization floor of `1e-5`.

**Finding M6.** `D_cond(steady) = C_eff v^2 / 2` with `C_eff = beta^2 Var_nu(int delta)`
— closed form, no eigenproblem, no fitted constant. In the linear-response regime
(left well, predicted `D < 0.1`): measured/predicted median **1.03** [0.99, 1.07]
over 13 points (cartesian) and **1.05** [0.99, 1.29] (min-norm). Fitted `v`-exponent
**1.96-2.04** in every case.

**Finding M7.** The relevant timescale is `tau_eff = sqrt(C_eff / C)`, not the
fiber spectral-gap time `1/omega^2`. They differ by 1.4-5.4x here, i.e. up to **29x
in `D_cond`** — using the spectral gap would force `kappa_W` down ~5x for nothing.

**Finding M8.** The adiabatic lift has `D_cond <= 1.8e-4` at **every** speed and
every fiber stiffness tested (cartesian reaches 4.11 on the same sweep). For a
conditionally correct lift there is no timescale condition at all.

**Finding M9 (where the law fails).** At the barrier top, where the conditional is
multimodal, linear response over-predicts by up to 2500x while the measured
`D_cond` saturates near 1.6. The prediction is an upper bound there, not an
estimate; quote it only where `C_eff v^2 / 2 < 0.1`.

## Phase M4 - does lift correctness change the free energy, not just the lag?

`scripts/exp_arms.py`, `results/manifold/arms/`. Nonlinear-CV `CHANNEL`,
`N = 256`, 1e5 steps, 16 seeds, matched force evaluations, frozen Stage-1
hyper-parameters (flow `kappa = 2.0`, `theta = 0.3`, jitter 0.01), floor 0.00458.

**Caveat, stated once and applying to Phases M4-M5.** `scripts/exp_arms.py` is a
FRESH engine on the graph systems, not `src/rcwfr/engines.py`. Its fiber dynamics is
the intrinsic flat-`y` Langevin (exact invariant measure, different from the ambient
constrained dynamics), its `ti_*` baselines place all `N` windows across the eval
window, and it carries none of the burn-in / ancestry / annealing machinery. Its
absolute numbers are NOT comparable to the linear-`xi` campaign above and are not
offered as a reproduction of it. Every claim drawn from these tables is a
WITHIN-EXPERIMENT contrast between arms that share the code and differ in one call.

| arm | `e_F` | / floor | vs `wfr_cart` |
|---|---|---|---|
| `ti_cold` fixed windows, cold fiber | 0.6501 | 142 | +0.9% |
| `wfr_cart` W+FR, cartesian lift | 0.6445 | 141 | — |
| `wfr_minnorm` W+FR, **minimum-norm lift** | 0.6157 | 134 | **-4.5%** |
| `fr_only` birth-death only, no transport | 0.2626 | 57 | -59.3% |
| `ti_warm` fixed windows, equilibrium start | 0.0355 | 7.8 | -94.5% |
| `wfr_adiab` W+FR, **adiabatic lift** | 0.0088 | **1.9** | **-98.6%** |
| `wfr_oracle` W+FR, conditional refresh | 0.0046 | 1.0 | -99.3% |

**Finding M9b (the degeneracy, measured).** Re-running the same arm comparison with
`a = 0` — the only change being that the reaction coordinate is linear — the cartesian
and minimum-norm arms return `e_F = 0.25614` and `D_cond = 0.8509` for BOTH. Identical
to every digit: with `G = 1` the minimum-norm horizontal lift and the identity lift are
the same operation, so the frozen campaign could not have tested the proposed fix even
in principle.

**Finding M9c (the lift error is not created by the nonlinearity).** The full matched
pair (floors 0.00349 linear / 0.00458 nonlinear; % against the cartesian lift on the
same coordinate):

| arm | linear `xi` | nonlinear `xi` |
|---|---|---|
| `ti_cold` | 0.1569 (-38.7%) | 0.6501 (+0.9%) |
| `wfr_cart` | 0.2561 (--) | 0.6445 (--) |
| `wfr_minnorm` | **0.2561 (+0.0%)** | 0.6157 (-4.5%) |
| `wfr_fit` | 0.3008 (+17.4%) | 0.8302 (+28.8%) |
| `wfr_fit_decay` | 0.0966 (-62.3%) | 0.5211 (-19.2%) |
| `wfr_adiab` | **0.0141 (-94.5%)** | 0.0088 (-98.6%) |
| `wfr_oracle` | 0.0036 (-98.6%) | 0.0046 (-99.3%) |

On the campaign's OWN linear systems the lift error is already worth **94.5%** of the
reachable error, and there the minimum-norm lift addresses none of it -- not because it
approximates badly but because it IS the cartesian lift. The nonlinearity is only what
lets the two differ, and they differ by 4.5%.

**Finding M9d.** The self-built lift with forgetting reaches **-62.3%** on the linear
coordinate against **-19.2%** on the nonlinear one: its viability tracks how hard the
fiber conditional is to estimate, which is the same thing that makes a correct lift
necessary.

**Finding M10.** On the end metric the minimum-norm lift is worth **4.5%** and the
conditionally correct lift is worth **98.6%**, landing within 2x of the estimator floor
and within 2x of the oracle. `wfr_adiab` and `wfr_oracle` both use the EXACT conditional
and are upper bounds, not implementable arms (Phase M5 is where that is tested). What
the contrast establishes is where the achievable gain lives: the 98.6% is unreachable by
any amount of care about the ambient metric, because `wfr_minnorm` is another arm in the
same table on the same z-trajectories and it sits at 4.5%.

**Finding M11.** With a nonlinear coordinate and a naive lift, the **Wasserstein
half is actively harmful**: `fr_only` (0.263) beats `wfr_cart` (0.645) by a factor
2.5, and `wfr_cart` does not beat cold fixed windows at all. This is the frozen
campaign's Finding C8 sharpened — Fisher-Rao carries no lift, so it carries no
lift bias, and once the coordinate is nonlinear that is the whole difference.

## Phase M5 - can the correct lift be built from the run's own samples?

`src/rcwfr/adaptive_lift.py`, `scripts/exp_arms.py --arms wfr_fit ...`.
Running smoothed `(z, y)` histogram -> `nu_hat(y|z)` -> CDF-matching lift, with a
count-based fallback to cartesian.

Fed **exact** samples the construction works: `D_cond` after a `dz = 0.2` step falls
from 0.762 (cartesian) to **0.0026**, a 290x reduction. That residual is a
**bandwidth floor**, flat under a 4x increase in samples, and `bw_z` is the
sensitive knob (0.08 gives 0.041, 0.03 gives 0.003) because the conditional changes
fast with `z`.

Fed its **own** samples inside the algorithm (1e5 steps, 16 seeds):

| arm | `e_F` | vs `wfr_cart` |
|---|---|---|
| `wfr_cart` | 0.6445 | — |
| `wfr_fit` running average | 0.8302 | **+28.8% (worse)** |
| `wfr_fit_decay` with forgetting | 0.5211 | -19.2% |
| `wfr_adiab` exact | 0.0088 | -98.6% |

**Finding M12 (forgetting is not optional).** A plain running average of the
`(z, y)` histogram makes the lift **worse than doing nothing** at both budgets tested
(+28.8% at 1e5 steps, +25% at 4e5), with a log-error slope of **-0.02**: it is parked.
The estimate is made from the ensemble the lift is steering, and without forgetting it
never sheds the ensemble's early error.

**Finding M13 (with forgetting the bootstrap works, at a warm-up cost).** Rerunning at
4x the budget (4e5 steps, 8 seeds) against the 1e5 / 16-seed run:

| arm | 1e5 steps | 4e5 steps | change | vs `wfr_cart`, same budget |
|---|---|---|---|---|
| `wfr_cart` | 0.6445 | 0.6469 | **+0.4%** | -- |
| `wfr_fit` | 0.8302 | 0.8107 | -2.3% | +25% |
| `wfr_fit_decay` | 0.5211 | **0.2357** | **-55%** | **-64%** |
| `wfr_adiab` | 0.0088 | 0.0056 | -37% | -99.1% |

Log-error slopes over the final decade: `wfr_cart` **-0.06**, `wfr_fit` **-0.02**
(both on a bias floor), `wfr_fit_decay` **-0.22**, `wfr_adiab` **-0.76**. The
self-built lift's conditional lag falls monotonically 0.596 -> 0.075 -> 0.027 ->
0.016 -> 0.0096 as its estimate improves. It is an implementable method with a
measured convergence rate, still 40x short of the exact lift, and the cost is the
deposits it makes while the lift is still harmful.

**Trap M-d (grade a self-consistent scheme by its slope, not its endpoint).** At one
budget unit `wfr_fit_decay` beat `wfr_cart` by 19% and would have been written off; at
four it beat it by 64%, because `wfr_cart` was on a floor and `wfr_fit_decay` was not.
Two arms 1.6x apart at 1x were 3.4x apart at 4x. Any comparison involving an arm that
estimates something from its own trajectory needs at least two budgets.

**Trap M-b (pooled vs z-resolved conditional error).** `wfr_fit_decay` ends with a
**pooled** PIT-KL of 0.0097 — near the floor — while its free-energy error is 114x
the floor. A single PIT histogram over the whole ensemble lets errors at different
`z` cancel. The quantity in the error functional is
`D_z = int KL[rho(.|z) || nu(.|z)] p(z) dz`, and it must be accumulated per `z`-bin
over the production half of the run. On a spot check `wfr_cart` reads 0.338 pooled
and **0.552** z-resolved. Grading a lift on the pooled number will reward the wrong
thing.

## Phase M6 - transport rate, and what a burn-in sweep cannot test

**Finding M16 (the trade-off is a property of the lift).** Sweeping the transport rate
`kappa` over a factor 32, 16 seeds each, everything else frozen (floor 0.00458):

| `kappa` | `wfr_cart` | `wfr_minnorm` | `wfr_adiab` |
|---|---|---|---|
| 0.25 | 0.26338 | 0.25749 | 0.01244 |
| 0.50 | 0.27124 | 0.27124 | 0.01241 |
| 1.00 | 0.35403 | 0.33595 | 0.01076 |
| 2.00 | 0.64454 | 0.61574 | 0.00881 |
| 4.00 | 0.95036 | 0.88297 | 0.00694 |
| 8.00 | **1.16727** | **1.04286** | **0.00669** |

The two families run in OPPOSITE directions. The naive lifts degrade monotonically by
a factor 4.4 and their best point is the SLOWEST rate tested, where the value (0.2634)
is just the birth-death-only arm (0.2626): their optimum is the limit in which the
Wasserstein transport is switched off. The adiabatic lift improves monotonically by a
factor 1.86 and its best point is the FASTEST rate tested, at 1.5x the estimator floor.
The gap widens from 21x to 175x. `figures/figM5_kappa.png`.

(The `kappa = 0.5` medians agree to five decimals by coincidence -- 0.27124354 vs
0.27124095, with per-seed differences up to 1.6e-2 -- not a bug.)

This is the condition `tau_mix << tau_WFR` shown to be a property of the lift and not
of the method: with `delta = 0` there is no rate to tune, and faster is strictly
better because faster transport buys coverage at no cost.


**Trap M-c (the burn-in sweep is budget-limited).** The obvious test of "transport,
freeze, equilibrate, then estimate" is to raise `n_eq` at fixed `n_cond`. It cannot
work: at `n_cond = 20`, `dt = 1e-3`, even `n_eq = 19` buys 0.019 time units of
relaxation against a fiber time of order 1.

| `n_eq` | deposits kept | `ti_cold` | `wfr_cart` | `wfr_minnorm` | `wfr_adiab` |
|---|---|---|---|---|---|
| 0 | 100% | 0.65007 | 0.95554 | 0.89317 | 0.00756 |
| 5 | 75% | 0.65006 | 0.95102 | 0.89099 | 0.00759 |
| 10 | 50% | 0.65001 | 0.94677 | 0.88892 | 0.00758 |
| 15 | 25% | 0.64995 | 0.94272 | 0.88700 | 0.00755 |
| 19 | 5% | 0.64997 | **0.93961** | 0.88545 | 0.00755 |
| | | -0.0% | **-1.7%** | -0.9% | -0.2% |

The correct reading is "short burn-in is worthless", not "Version I is refuted".

**Finding M15 (the arm is bias-dominated).** The curve above is monotone improving,
not U-shaped: discarding 95% of the deposits still helps, so the variance cost of
losing them is smaller than the bias removed. With N = 256 over 5000 epochs even 5%
of deposits is 1.3M samples, so variance is not binding -- the error is essentially
all bias, and the burn-in affordable inside an epoch removes 1.7% of it. `wfr_adiab`
is unmoved because it has no such bias to remove.

**Finding M14 (an exact lift is free to transport fast).** Quadrupling the transport
step by raising `n_cond` from 5 to 20 at fixed `kappa`, 16 seeds, matched force
evaluations:

| arm | n_cond = 5 | n_cond = 20 | change |
|---|---|---|---|
| `wfr_cart` | 0.6445 | 0.9555 | **+48%** |
| `wfr_minnorm` | 0.6157 | 0.8932 | **+45%** |
| `wfr_adiab` | 0.0088 | **0.0076** | **-14%** |

The two naive lifts degrade in the direction the `dz^2` law requires; the exact lift
does not degrade at all and in fact improves, because larger W steps transport more
efficiently and an exact lift carries no penalty for taking them. This is Finding M8
(no timescale condition for a correct lift) at the level of the whole algorithm.

## Phase M7 - the warm-up policy, and the rigid-measure route

**Finding M17 (discarding the warm-up works, and the control confirms why).**
Zeroing the mean-force accumulator partway through, 16 seeds, everything else frozen:

| reset at | `wfr_cart` | `wfr_fit_decay` | `wfr_adiab` |
|---|---|---|---|
| none | 0.64454 | 0.52105 | 0.00881 |
| 0.50 | 0.66254 (**+2.8%**) | 0.36270 (**-30.4%**) | 0.00885 (+0.4%) |
| 0.75 | 0.67343 (**+4.5%**) | **0.31493 (-39.6%)** | 0.01385 (**+57.2%**) |

The three arms respond in three different directions and each is the predicted one:

* `wfr_cart` has no warm-up, so discarding deposits is pure variance cost -- monotone
  worse, and mildly so.
* `wfr_fit_decay` has a warm-up whose bias dominates that variance cost even when 75%
  of the deposits go -- monotone better, and still improving at the largest fraction
  tested, which says the optimum is later still.
* `wfr_adiab` is already within 2x of the estimator floor, so it is variance-limited:
  +0.4% at half, **+57%** at three quarters.

That divergence is a usable diagnostic in its own right: **if a reset helps, the arm
has a warm-up; if it hurts, the arm is variance-limited.** A fixed fraction is the
crude version -- the right policy resets when the lift's own conditional diagnostic
stops moving.

Combining with M13: the self-built lift is at **-51%** against the cartesian lift at
the same reset (0.3149 vs 0.6734) at one budget unit, having been -19% with no warm-up
policy at all.

**Finding M18 (the rigid-measure route is free).** Chapter 3's alternative to targeting
`nu^xi` directly is to sample the rigid measure with plain SHAKE/RATTLE and correct
statistically,

    F(z) = F_rgd(z) - beta^-1 log E_{nu_Sigma(z)}[ (det G)^{-1/2} ],

which avoids second derivatives of `xi` entirely. The identity is exact, so the only
question is the reweighting variance (`scripts/exp_rigid_route.py`):

| `a k` | max \|F - F_rgd\| | ESS fraction | samples/bin for 0.1x floor |
|---|---|---|---|
| 0.42 | 0.0054 | 0.999 | 89 |
| 0.84 | 0.0187 | 0.985 | 1510 |
| 1.68 | 0.0442 | 0.976 | 2371 |
| 2.80 | 0.0859 | 0.948 | 5374 |

A production run deposits 1e5-1e6 samples per z-bin, so the correction is resolved to a
tenth of the estimator floor at a cost of nothing. **For an atomistic implementation,
take route (B):** standard rigid constrained dynamics, Fixman handled statistically.

## Phase M8 - making the learned lift deployable, and stressing the design rule

**Finding M19 (the learned lift's two knobs, and where its optimum is).** Screening
the forgetting factor against the z-bandwidth of the conditional estimate, warm-up
discard on throughout (reset at 0.5), 16 seeds, floor 0.00458:

| decay \ bw_z | 0.015 | 0.03 | 0.06 | 0.10 | 0.16 | 0.24 | 0.36 |
|---|---|---|---|---|---|---|---|
| 1.0 (none) | 0.858 | 0.820 | 0.591 | -- | -- | -- | -- |
| 0.9995 | 0.683 | 0.451 | 0.244 | -- | -- | -- | -- |
| **0.999** | 0.552 | 0.363 | 0.213 | 0.206 | 0.185 | **0.181** | 0.250 |
| 0.997 | 0.566 | 0.545 | 0.534 | -- | -- | -- | -- |
| 0.99 | 0.662 | 0.662 | 0.662 | -- | -- | -- | -- |

Best is decay 0.999, `bw_z` 0.24: **0.1809**, which is **-72.7%** against the cartesian
lift at the same budget and reset (0.6625), 39.5x the estimator floor, and 20x the
exact lift. The bandwidth optimum is interior -- 0.36 is worse -- so this is a real
optimum, not an edge.

**Finding M20 (the deployed optimum bandwidth is NOT the oracle-fed one).** Fed exact
samples, the learned lift is best at `bw_z = 0.03` and degrades by 13x at 0.08
(Phase M5). Fed its own samples it is best at `bw_z = 0.24` -- **8x larger** -- and
0.03 is 2x worse than the optimum. Heavy z-smoothing damps the self-reinforcement
that makes the estimate track its own ensemble. Tuning this knob against an oracle
would land on the wrong value by a factor of eight.

**Finding M21 (the design rule, properly stressed).** Using the SHIFTED spectator
block, with `A = 1.0` so that `C_S / C_y1 = 0.92` and the block's lift error is held
FIXED while its relaxation time sweeps 256x:

| `tau_spec` | naive | promote (naive on S) | both | oracle | excess/floor | Dz(S) promote |
|---|---|---|---|---|---|---|
| 16.0 | 0.6749 | 0.2521 | 0.0624 | 0.0050 | **38.6** | 0.269 |
| 1.0 | 0.5734 | 0.0788 | 0.0135 | 0.0050 | **13.3** | 0.103 |
| 0.062 | 0.6353 | 0.0199 | 0.0087 | 0.0050 | **2.3** | 0.0075 |

The right statistic is the EXCESS of `promote` over `both` in units of the estimator
floor, because the ratio is contaminated by `both` itself approaching the floor. That
excess falls **17x** (38.6 -> 2.3 floors) across a 256x fall in relaxation time, with
the lift error held constant, and `Dz(S)` falls 36x alongside it. **The same lift
error costs less on a faster mode** -- which is the design rule, and the width-only
block of Phase M6 could not have shown it (promote/both = 0.89 there, i.e. no
difference at all).

Two limits, stated plainly:

* **The gap never closes.** At `tau_spec = 0.062` promoting one mode is still 2.3
  floors behind correcting both, and 4.1x the floor. "Fast modes are free to lift
  naively" is too strong over the range reachable here; they are *cheaper*, not free.
* **Promotion is worth 8-32x on its own** at every stiffness (0.675->0.252,
  0.573->0.079, 0.635->0.020). Branch B works; it does not reach the ceiling alone.

**Efficiency note (not a result, but it changed the campaign).** The inner loop is
launch-latency-bound, not bandwidth-bound: 4096 and 40960 particles both cost 2.7 ms
per step. Running arms sequentially wasted a factor of the arm count for nothing.
Batching the arms into the row axis and striding the deposits (the fiber
autocorrelation time is O(1) against `dt = 1e-3`, so consecutive deposits are ~1000x
redundant) gave **~9x** with no change to any result.
