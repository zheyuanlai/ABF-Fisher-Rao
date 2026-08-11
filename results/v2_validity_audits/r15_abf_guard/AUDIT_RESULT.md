# v2 validity audit — frozen v1 pentane R15 ABF baseline

v1 is immutable and was not modified. This audit re-ran the R15 ABF screen with the standard
ABF sample-count guard (`abf_min_count_dist = 200`, NAMD's default `fullSamples` and the value
the trusted alanine 2-D sampler uses) and **nothing else changed**: same molecule, beta grid,
init modes, CV, walls, grid, bandwidths, dt, budget, seeds 0-7, 1024 replicas. The config is
verified identical to `configs/alkanes_cv_extension/r15_screen.yaml` apart from name, output
root and that one field.

Evaluation reproduces v1's published `norm_final_l2_F` exactly (0.0677 / 0.0756 / 0.1590 /
0.1436), confirming the comparison is on v1's own thermal mask and metric.

## Result 1 — the missing guard was NOT material for R15

| cell | span ratio Δ | normL2(F) Δ | L2(F') Δ | lowSupport Δ |
|---|---|---|---|---|
| dispersed β1 | +0.000 | +0.0000 | +0.001 | +0.000 |
| dispersed β2 | −0.000 | −0.0000 | −0.000 | +0.000 |
| trans β1 | +0.001 | −0.0000 | −0.003 | +0.000 |
| trans β2 | +0.003 | +0.0004 | +0.000 | +0.000 |

**The v1 R15 numbers are robust to adding the guard.** Unlike deca-alanine (128 walkers,
129 bins, 72 kT), R15 runs 1024 walkers over 256 bins in a confined interval, so effective
counts pass the threshold almost immediately and the guard is a no-op in practice.

The internal v1 inconsistency that motivated the audit is nonetheless real and is now recorded:
`Sim2DConfig` carried `abf_min_count = 5.0` and masked untrusted cells, while `DistSimConfig`
had no such field, so v1's 2-D torsion cell ran **with** the guard and R15 ran **without** it.
That asymmetry did not change the numbers, but it was never stated.

## Result 2 — Gate 0's span clause is mis-specified

The span-ratio clause of Amendment 7 fires on **all four** cells (1.31-1.51), including the
β=1 cells v1 classifies as *easy* with `normL2(F) = 0.068` and **zero** low-support bins. A
max−min statistic over 183 bins is dominated by tail noise and does not measure bias validity.
**The clause is wrong, not R15.** It must be replaced before Gate 0 is applied anywhere.

Clause 2 (population pinning) does not fire: max tercile occupancy 0.36-0.71, far below 0.90.

## Result 3 — the statistic that DID diagnose deca puts R15 β=2 in the same place

Relative mean-force error against the reference, on the thermal mask:

| system / cell | relative \|mf − dF_ref\| / \|dF_ref\| | v1 verdict |
|---|---|---|
| deca-alanine screen | **0.61** | conditional-equilibration-limited |
| R15 dispersed β1 | 0.264 | easy |
| R15 trans β1 | 0.265 | easy |
| **R15 dispersed β2** | **0.564** | starved / discovery-limited |
| **R15 trans β2** | **0.593** | starved / discovery-limited |

The β=2 cells — v1's discovery-limited pillar — sit at essentially deca's error level, and the
guard does not move them (Δ = 0.000).

**This is a signal, not a verdict.** For deca the attribution was settled by a controlled
experiment: the same `f_loc` estimator accumulated inside umbrella-restrained windows returned
8.4 % error, exonerating the estimator, the CV geometry, the reference and the integration, and
leaving conditional equilibration as the only remaining explanation. **No equivalent test has
been run for R15.** Until it is, a 0.56-0.59 error is consistent with several causes — including
error in the R15 importance-sampling reference itself, which is a different object from the
deca umbrella+MBAR reference.

## What this does and does not license

* The audit's primary question is answered: **the guard does not change R15**, so the v1 R15
  numbers stand as published.
* Whether R15's β=2 classification should remain *discovery-limited* or become
  *conditional-equilibration-limited* is **open**, and is exactly the four-way question
  Amendment 8 says must no longer be forced into a three-way box.
* No threshold for a revised Gate 0 is set here. Choosing one now would determine R15's verdict
  after seeing its number, which is the failure mode this project keeps guarding against.
