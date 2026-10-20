# Results summary: ABF vs ABF + uniform marginal Fisher–Rao (paired by seed)

**Statistic.** Per-seed relative change (FR − ABF)/ABF, then the median across seeds [95 % bootstrap CI of the
median]. "Wins" counts the seeds where FR's error is lower. A negative change means FR is better.

**Integrated vs final error.** The *integrated* error summarises the whole convergence curve; the *final* error is
the endpoint e_F(T). They are different questions and are reported separately. The integration windows are each
experiment's own validated conventions (column "integrated definition").

## Free-energy error

| figure | system (seeds, N) | integrated definition | integrated free-energy error | final error e_F(T) | verdict |
|---|---|---|---|---|---|
| gateway | entropic gateway (32, 2048) | mean of e_F over 200 uniform budget fractions on [0, 40] | **−31.4 % [−34.7, −27.7]**, 31/32 | **−61.4 % [−65.1, −55.2]**, 32/32 | FR better (integrated and final) |
| lta150 | ethane / LTA 150 K (16, 1024) | trapezoid over [0, 60] | **−28.5 % [−29.4, −26.6]**, 16/16 | **−54.4 % [−58.3, −46.1]**, 16/16 | FR better (integrated and final) |
| lta300 | ethane / LTA 300 K (16, 1024) | trapezoid over [0, 60] | **−13.7 % [−14.8, −11.1]**, 16/16 | −7.9 % [−14.6, +0.7], 11/16 | FR better integrated; final **not resolved** |
| ite300 | ethane / ITE 300 K, FR on 1 ≤ t < 10 only (16, 1024) | post-ramp mean over [1, 50] | **−37.7 % [−42.2, −28.5]**, 16/16 | **−32.3 % [−42.6, −24.1]**, 16/16 | FR better; time-to-accuracy S_geo **1.59 [1.31, 1.85]** |
| meta | energetic metastability toy, NEW (32, 1000) | trapezoid over [0, 200] | **+51.9 % [+40.2, +60.7]**, 0/32 | **+334 % [+291, +422]**, 0/32 | **FR harmful** |
| alanine | alanine dipeptide, (φ, ψ) (16, 2048) | trapezoid over own window [20, 100] ps | +0.04 % [−1.36, +0.64], 8/16 | −0.60 % [−0.91, +0.03], 10/16 | **neutral** (equivalent within ±4 %) |
| wca | WCA dimer, validated dt (16, 256) | trapezoid over [0, 240] | **+9.9 % [+2.3, +13.2]**, 4/16 | **+131 % [+93, +194]**, 1/16 | **FR harmful** (validated negative) |

Notes on the table:

* The gateway's independent dt-refinement run set gives −29.7 % (30/32) at the same validated timestep.
* ITE's full-window [0, 50] mean is −24.5 %.
* Alanine uses a BCa bootstrap CI (the study's convention).
* The other experiments use a 10 000-resample percentile bootstrap.

## Mean-force error (secondary)

| figure | final e_F′(T) | note |
|---|---|---|
| gateway | e_F′ floor-limited (0.0323) | Ī_F′ −20.9 % |
| lta150 | −42.6 % [−49.7, −35.8], 16/16 | |
| lta300 | −5.4 % [−12.0, +4.7] | not resolved |
| ite300 | −22.4 % [−29.0, −13.7], 16/16 | |
| meta | +13.9 % [+11.4, +18.2], 0/32 | I_F′ −1.0 % [−2.3, +2.4]; e_F′ dominated by the 0.139 piecewise-constant floor |
| alanine | −0.17 % [−0.24, −0.04] | e_F′ is reference-noise limited (≈ 8.25 kJ/mol/rad) |
| wca | +10.5 % [+7.0, +16.5], 2/16 | |

## Where FR helps and where it hurts

* **FR helps** in the three entropic-barrier systems: gateway, LTA, and ITE with early stopping. The integrated error
  falls by 14–38 %.
* The **final** benefit is resolved for gateway, LTA 150 K and ITE E1, but **not at LTA 300 K**.
* **FR is neutral** for alanine.
* **FR is harmful** for the validated WCA dimer and for the new metastability toy.
* In the metastability toy, FR flattens the marginal faster but biases the free energy across the barrier. A
  diagnostic places the bias in the hidden y-conditional (see `metastability/RESULTS.md`).
* None of these are equal-cost allocation comparisons: both arms use the same N and T, so the force evaluations are
  equal only at fixed N. Earlier repository studies found that, with N free at equal force evaluations, the best
  ABF-only allocation (a few long replicas) ties or beats the best ABF + FR allocation
  (`docs/equal_budget/FINAL_RESULTS.md`). For ITE, ABF at K = 256 for 200 t.u. is about 2× more efficient than E1
  (`docs/convergence_fr_campaign/FINAL_SYNTHESIS.md`). Those studies are not part of this package.

## What was computed

* Six figures are post-processing of original validated data: gateway, LTA ×2, ITE, alanine, WCA.
* The metastability figure is a new preregistered experiment (≈ 6 CPU core-hours).
