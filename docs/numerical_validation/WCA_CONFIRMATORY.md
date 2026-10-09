# WCA dimer at the validated time step: ABF vs ABF + FR vs matched sham (Phase V)

*2026-10-09, branch `research/langevin-validation-oct2026`.*

* Preregistration (written before any FR run at a validated dt): `configs/numerical_validation/wca_confirmatory_prereg.json`.
* Runner: `scripts/numerical_validation/run_wca_confirmatory.py`.
* Analyzer: `scripts/numerical_validation/analyze_wca_confirmatory.py`.
* Data: `results/numerical_validation/wca_confirmatory/dt0.000125_N256_T240/` (`summary.json`, `scoreboard.md`,
  `figures/fig_wca_confirmatory.png`).

## Licence and design

The WCA dynamics gate (`WCA_LANGEVIN_VALIDATION.md`) chose dt* = 0.000125, the largest dt at which the
production integrator reproduces exact Gibbs to ≤ 0.005 kT. By the rule fixed beforehand (N = 1024 would
take about 10 h single-threaded per seed), the experiment uses N = 256 and T = 240 t.u., the accepted
physical time.

* Every physical-time knob of the accepted histogram confirmation is kept:
  * ABF warm-up and estimator burn-in 20 t.u.;
  * FR from 40 t.u., every 0.01 t.u., rate 0.1, score clip 2, KDE bandwidth 0.07;
  * cap floor(0.02 N) = 5 per opportunity.
* 16 fresh seeds, 5100-5115.
* Arms: histogram ABF, ABF + uniform FR, and the matched-turnover sham. The sham replays FR's realised
  count at every opportunity, with uniform deaths and sources drawn from survivors.
* All three arms run on the same Langevin noise.
* Scored against the exact Gibbs (Metropolis MC) reference with the accepted read-out.
* Equal force evaluations per arm (N × T/dt). The FR + sham process costs +4 % wall-clock per arm over the
  ABF process.

## Result: **FR_HARMFUL** (frozen decision rule)

| arm | median I_F | median e_F(T) | median own e_F′ | replacements | min windowed ancestor ESS/N | max family share |
|---|---|---|---|---|---|---|
| ABF | 13.94 | 0.0095 | 0.136 | 0 | — | — |
| ABF + FR | 15.37 | 0.0179 | 0.150 | 677 | 0.755 | 0.068 |
| sham | 14.65 | 0.0096 | 0.136 | 677 | 0.764 | 0.074 |

Each cell gives the median paired % change [95 % bootstrap CI] (wins out of 16).

| contrast | ΔI_F | Δe_F(T) | Δe_F′ |
|---|---|---|---|
| FR vs ABF | **+9.9 % [+2.3, +13.2]** (4/16) | **+131 % [+93, +194]** (1/16) | +10.5 % [+7.0, +16.5] (2/16) |
| sham vs ABF | −0.8 % [−4.6, +5.0] (8/16) | −8.4 % [−40, +55] (8/16) | +1.0 % [−3.2, +7.4] (7/16) |
| FR vs sham | **+8.7 % [+1.1, +14.7]** (3/16) | +113 % [+34, +241] (2/16) | +10.7 % [+5.8, +13.7] (2/16) |

* FR tracks ABF until t ≈ 50 t.u. Then it stops improving, at e_F ≈ 0.018 kT. ABF and the sham keep
  converging to ≈ 0.010.
* There is no establishment-limited phase in which FR gets ahead.
* The sham, with the same turnover and genealogy and only the direction randomised, is neutral. So the
  harm comes from the *score-directed* selection, not from resampling per se.
* Predictions: **P2 PASS**, the accepted −24 % gain does not reappear. **P1 partially**, see below.

## The stationary FR tilt is intrinsic, not an integrator artefact

Pooled long-run limits (16 seeds), D vs exact Gibbs in kT:

* ABF 0.0038, at the level of the dt* integrator tolerance;
* sham 0.0000;
* **ABF + FR 0.0198**;
* FR limit vs ABF limit **0.0169 [upper 0.0212]**;
* sham vs ABF 0.0047.

The FR profile is tilted mainly in the stretched well, +0.05 kT at z = 1.1 relative to the compact well.

FR tilt at the same N = 256 across time steps:

| dt | D(FR limit, ABF limit) |
|---|---|
| 0.002 | 0.090 |
| 0.0005 | 0.022 |
| **0.000125** | **0.017** |

From 0.002 to 0.0005 the tilt fell 4×, as an integrator interaction would. From 0.0005 to 0.000125 it
fell only 1.3×; an O(dt) effect would have fallen to ≈ 0.0055. So the tilt at a valid dt is
**predominantly intrinsic to finite-population, score-directed FR**. The integrator only amplified it at
the accepted dt. Prediction P1 ("the tilt at dt* is smaller than at dt 0.0005") holds in direction but not
in the O(dt) magnitude it allowed.

Because the matched sham shows no tilt, cloning and genealogy alone do not cause it. The selection rule
does. A plausible mechanism, **not tested here**:

* the score depends on the walkers' instantaneous z-density;
* walkers in locally sparse z-regions are preferentially those in transit, whose solvent configuration has
  not relaxed;
* cloning them over-weights non-equilibrium conditional configurations, which biases the conditional mean
  force.

This is the "fibre-lag" mechanism. It is irrelevant when the marginal is starved (LTA, gateway: there FR
helps). It dominates when ABF converges smoothly (WCA). It is consistent with the absence of any FR tilt
at 300 K in LTA (FR limit vs ABF limit D = 0).

## Conclusion for WCA

At a time step that reproduces the physical free energy, uniform FR does not accelerate ABF on the WCA
dimer. In the accepted physical time it is harmful: +10 % integrated, +131 % final error. The harm comes
from a stationary bias of 0.017 kT that its score-directed selection introduces and its sham does not.
**WCA is a validated negative control**, not a positive.
