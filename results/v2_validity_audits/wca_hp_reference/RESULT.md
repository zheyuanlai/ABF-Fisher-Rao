# Stage A — high-precision WCA TI reference. The z ~ 0.25 defect is REAL and 23 sigma.

41 z-values x **4 independent preparations** x 128 replicas = 20 992 states; 20 k prep at each
preparation's own z, 20 k re-equilibration at target, 50 k production. 22 min on one H200.

## The falsifiable target, answered

| | `F'(0.255)` |
|---|---|
| cached reference | **2.013** |
| Gate 0 pools predicted | 0.931 |
| **high-precision build** | **0.601 +- 0.060** (preparation SE) |

Deviation from cached: **-1.412, i.e. 23.4 sigma.** All four preparations agree that the cached
value is wrong, and by a similar amount:

```
  lattice 0.762   from_lo 0.566   from_hi 0.604   hot 0.473
```

Note the `lattice` preparation -- the one the cached reference uses -- is itself at 0.762, not
2.013. So the defect is **not** simply "the lattice preparation is biased": it is a
under-equilibration in the cached build that longer sampling of the same preparation fixes.

## The defect is localised, with a smaller broad component

| z | `F'_new` | cached | delta | delta/se_prep |
|---|---|---|---|---|
| 0.150 | 7.991 | 7.556 | +0.436 | 11.5 |
| 0.185 | 6.617 | 6.124 | +0.493 | 21.9 |
| 0.220 | 4.023 | 3.867 | +0.156 | 3.8 |
| **0.255** | **0.601** | **2.013** | **-1.412** | **-23.4** |
| 0.290 | 1.535 | 1.986 | -0.450 | -6.8 |
| 0.325 | 3.166 | 2.878 | +0.288 | 5.5 |
| 0.360 | 3.708 | 3.356 | +0.351 | 13.0 |

A sharp spike at `z = 0.255-0.290`, on top of a broad systematic offset (~+0.3 to +0.5 for
`z` in [-0.06, 0.36], ~-0.1 to -0.25 for `z > 0.5`). Mean `|delta|` over the transition region
0.2-0.8 is **0.2102**; max is 1.4119.

## Stage B — what it does to the free energy, and to Case IX

The whole mean-force curve was reintegrated with Case IX's own convention (smooth -> cumulative
trapezoid -> zero at midpoint), not patched pointwise.

```
  L2(F_new - F_cached)                         = 0.0608
  mean |F_new - F_cached| over z in [0.2, 0.8]  = 0.0103
```

Against the Case IX numbers:

| | |
|---|---|
| ABF arm, median final `L2(F)` | 0.0901 |
| mFR arm, median final `L2(F)` | 0.0429 |
| **reference shift** `L2(F_new - F_cached)` | **0.0608** |
| shift / ABF's own error | **0.675** |
| Case IX effect size (22.83 % of ABF error) | 0.0206 |
| **shift / effect size** | **2.96** |

**The reference correction is ~3x the size of the effect it was used to measure.** That does not
mean the effect is wrong -- the shift is common to both arms and much of it will cancel in a
paired comparison -- but it does mean the ratio cannot be assumed to cancel:

    |F_A - F_new|^2 - |F_B - F_new|^2  =  |F_A - F_old|^2 - |F_B - F_old|^2
                                          - 2<F_A - F_B, delta>

The cross term vanishes only if the two arms have the same error pattern where `delta` lives,
and `delta` peaks at `z = 0.255`, inside the transition region where the arms differ most. So
the sign is very likely safe and the magnitude is not.

## Status

* **-22.83 % is an uncalibrated effect size.** It is scored against a reference now known to be
  wrong by 23 sigma at one transition-region point.
* The **mechanism** is unaffected: Gate 0 passed at 0.040 (all z) / 0.039 (transition), so WCA is
  genuinely establishment-limited.
* **Stage C is required, not optional.** Case IX raw retains only scored `l2_f_t` scalars and
  `final_pmf`, never `F_hat_t(z)`, so no rescore is possible; the dynamics must be re-run against
  `wca_hp_reference.npz`.

## Caveats

* `z = -0.2` and `z = 1.2` show large deltas (-2.672, +20.879) because the cached file carries
  zeros at its outermost grid points. Domain edges, excluded from the evaluation mask.
* 41 z-values (spacing 0.035), not the cached 51 or the 160-point eval grid; the eval-grid output
  is interpolated. Sufficient to establish the defect and its size; a denser build would be
  needed to quote a final corrected `F'` curve pointwise.
