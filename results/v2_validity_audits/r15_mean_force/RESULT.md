# R15 restrained mean-force test — INCONCLUSIVE

Question: is pentane R15 at beta=2 *discovery-limited* (v1's classification) or
*conditional-equilibration-limited* (the deca signature)? **Not answered.**

| statistic | value | threshold |
|---|---|---|
| build-vs-build relative difference | **0.261** | 0.15 |
| all-trans vs reference | 0.691 | — |
| all-gauche+ vs reference | 0.768 | — |
| pooled vs reference | 0.640 | — |
| (ABF screen at beta=2) | 0.564 / 0.593 | — |
| (deca) | 0.61 | — |

The two builds started from all-trans and all-gauche+ dihedrals and **did not converge to the
same conditional average** after 60k pull + 400k equilibration + 200k production steps at fixed
R. `cos phi1` at identical R still carries the initial state:

| R | trans | gauche |
|---|---|---|
| 2.066 | -0.248 | -0.600 |
| 2.518 | +0.238 | -0.406 |
| 2.647 | +0.235 | -0.274 |

Where the builds agree (R > 2.78 and R < 1.68) `cos phi1` agrees too, so the disagreement is
precisely the hidden torsional coordinate failing to relax.

## What this establishes, and what it does not

**Establishes:** R15's orthogonal torsional coordinate is slower than 600k steps at fixed R —
7.5x the entire v1 screen budget. That is consistent with conditional-equilibration-limited.

**Does not establish:** the three-way attribution. With my own controlled sampler failing to
equilibrate, "ABF's conditional sampling is bad" cannot be separated from "the
importance-sampling reference is wrong". The test's premise — that restraining the CV leaves
only fast coordinates — is false here.

## Defect in this test, to fix before any rerun

`k_umbrella = 400` is too soft against the steep compact-side PMF. Window `R_c = 1.550` sampled
at `<xi> = 1.943`; every window below ~1.9 collapsed to the same place, so **R < 1.9 is not
covered at all**. A rerun needs an R-dependent or much stiffer restraint there.

## What would settle it

Not more of the same. Enumerate the torsional states explicitly at fixed R — sample each
(phi1, phi2) basin separately and reweight to get p(y | R) — instead of hoping a single long
trajectory mixes between them. That measures the conditional distribution directly rather than
relying on it to equilibrate.
