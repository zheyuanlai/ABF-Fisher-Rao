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
