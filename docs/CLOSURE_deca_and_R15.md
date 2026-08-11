# Closure — deca-alanine and pentane R15

Both closed 2026-08-11. Neither yields an mFR positive. Neither is a method failure: **neither
system ever reaches the regime in which marginal selection can act**, and for one of them there
is now a theorem saying it never could.

---

## 1. Deca-alanine — CLOSED as `conditional-equilibration-limited`

**Model.** Ace-(Ala)10-Nme, 112 atoms, ff14SB vacuum, BAOAB, 300 K, dt 1 fs. CV = terminal
carbonyl C-C distance. OpenMM parity 2.9e-8 / 8.7e-9. *Declared deviation: literature uses
CHARMM/NAMD; ff14SB carries no CMAP and is covered exactly by an already-validated energy path,
so this is "same molecule, same CV, same budget, different modern force field", not a
reproduction of the published PMF.*

**Reference (usable, retained).** Umbrella + MBAR, 96 windows, 3 builds interleaved, each seeded
from a different conformational pool. 2.0 ns/replica, 18 432 ns aggregate (~144x the 128 ns
literature benchmark). Between-build max pairwise L2 0.607 kJ/mol; **time drift 3.18 kJ/mol =
0.178 of a resolvable 10 % effect**, i.e. 5.2x the between-build spread — drift is the honest
uncertainty. Span 71.7 kT, minimum 1.637 nm, single-basin monotone.

**Gate A PASSED** at max pairwise TV = 0.754 against a 0.30 threshold.

**Gate 0 FAILED.** ABF-only screen at the literature budget (8 ensembles x 16 walkers x 0.5 ns):

| | |
|---|---|
| learned `A_hat` span | 86.7-110.3 kT against a 72.0 kT reference |
| walkers above 2.80 nm, 2nd half | 95.1-99.96 % |
| seed 0 population | `[1,0,0]` at 0 ps -> `[0,0,1]` by 100 ps, never returns |
| mean force vs `dF_ref/dR` | **61 % error at up to 2e6 effective counts per bin** |

**Attribution, by controlled experiment.** The same `f_loc` estimator inside umbrella-restrained
windows gives **8.4 %** relative error (interior bins agree sub-1 %: 103.66 vs 103.20, 127.08 vs
127.40, 133.04 vs 132.49), and integrating `<f_loc>` gives 69.4 kT against the reference's
67.1 kT. **The estimator, CV geometry, reference and integration are mutually consistent. There
is no bug in the mean force.** ABF's *conditional* ensemble at fixed `xi` is not equilibrated.

**Why mFR cannot help, structurally.** For any score `S_t(xi)` depending on the CV alone,
`d/dt p_t(y|xi)|_FR = 0` — the mean-field selection step leaves the conditional distribution
exactly invariant. This covers count balancing and the Chapter-6 Laplacian rule as well as
Fisher-Rao. mFR moves population *along* `xi`; the fault is *orthogonal* to `xi`.

**Not done, deliberately.** No budget extension. 8 ns aggregate is the literature benchmark, run
faithfully with 16 walkers sharing one estimator. Raising it until ABF becomes tractable and
then hunting an mFR gain would be searching for a budget that manufactures the comparison.

**Value retained.** Deca is the first system where **Gate A passes and ABF still fails** — a CV
can separate the structural states well and the conditional equilibrium along it still not be
reached. The alkanes and dipeptides never showed this.

---

## 2. Pentane R15 — CLOSED; v1's `discovery-limited` classification STANDS

Three separate questions were asked. All are answered.

**(a) Does the missing ABF sample-count guard change v1?** No.
`Delta normL2(F) <= 3.6e-4`, `Delta span ratio <= 0.003`, `Delta lowSupport = 0.000`, and the
recomputation reproduces v1's published values exactly (0.0677 / 0.0756 / 0.1590 / 0.1436).
**The v1 R15 numbers stand as published.** A real v1 inconsistency was found on the way:
`Sim2DConfig` carried `abf_min_count = 5.0` and masked untrusted cells while `DistSimConfig` had
no such field, so v1's 2-D torsion cell ran *with* the guard and R15 *without* it. It changed
nothing numerically, but the two classifications were never like-for-like and that is now
recorded.

**(b) Does R15 fail Gate 0?** Locally yes, and the mechanism is clean. At fixed R, nine
torsional pools propagated independently for 200 time units (5x the v1 screen budget):

| R | pools reaching it | `<f_loc>` spread | \|dF_ref\| | spread/\|ref\| | Gate 0 |
|---|---|---|---|---|---|
| 1.70 | **0/9** | — | 38.90 | — | unusable, geometrically unreachable |
| 2.00 | 9/9 | 1.72 | 37.00 | 0.05 | PASS |
| 2.30 | 9/9 | 12.12 | 10.76 | 1.13 | **FAIL** |
| 2.60 | 9/9 | 19.27 | 1.98 | 9.71 | **FAIL** |
| 2.90 | 9/9 | 0.02 | 0.17 | 0.10 | PASS |
| 3.20 | 9/9 | 0.09 | 10.88 | 0.01 | PASS |

At R = 2.60 eight of nine pools agree to ~0.3 while the two double-gauche states sit at **+3.49,
opposite in sign**. The conditional mean force is bimodal in the hidden torsional coordinate and
the double-gauche weight never equilibrates.

**(c) Does that failure explain v1's error? NO — and this is the reason R15 is not
reclassified.** Decomposing v1's squared `L2(F)` on the thermal mask:

| R window | share of squared error |
|---|---|
| [1.4, 2.0) | 0.0 % |
| [2.0, 2.3) | 11.9 % |
| **[2.3, 2.6)  Gate 0 FAILS** | **12.4 %** |
| [2.6, 2.9) | 19.9 % |
| **[2.9, 3.7)  Gate 0 PASSES** | **55.7 %** |

The Gate-0-failing band carries 12.4 % of the error across 18.1 % of the mask — *sub*-proportional
— while the dominant error region is Gate-0-clean. Amendment 10 classifies by the region
responsible for the free-energy error, so by its own rule **R15 remains discovery-limited**, with
a documented local conditional-equilibration caveat at 2.3-2.6 nm.

**Correction recorded.** An interim reading of this session proposed reclassifying R15 as
conditional-equilibration-limited on the strength of the Gate 0 failure alone. The error
decomposition refutes that, and the interim reading is withdrawn. A second interim error is also
recorded: the per-run scripts printed `maxTV(pool,pool)` as the Gate 0 criterion, which is wrong
— at R = 2.90 the pools are maximally unmixed (TV = 1.000) while the mean force is
pool-*independent* (spread 0.02). Non-mixing is harmless where `f_loc` does not depend on `Y`.
The correct statistic is the spread of `<f_loc>` across pools relative to `|dF_ref|`, used above.

---

## 3. What both closures give the project

The regime map is four-way, with Gate 0 evaluated first:

| regime | what fails | mFR |
|---|---|---|
| ABF-sufficient | nothing important | neutral |
| discovery-limited | state never reached | cannot clone what does not exist |
| **establishment-limited** | state reached, marginal population slow | **the useful regime** |
| conditional-equilibration-limited | `p_t(q\|xi)` is wrong | cannot directly repair |

and three timescales rather than two: `T_hit << tau_perp << T_est`. **mFR works when the system
locally forgets a cloned configuration faster than the marginal population equilibrates.**

The project's claim is therefore not "birth-death added to ABF sometimes helps" but *"we
characterise when marginal population selection can and cannot accelerate adaptive free-energy
estimation"*, with `d/dt p(y|xi)|_FR = 0` as the structural reason for one of the boundaries.

## 4. Open, and NOT pursued here

* WCA dimer and the entropic gateway were classified establishment-limited **before Gate 0
  existed**. Their interpretation is **provisional** until each passes a Gate 0 backfill. This is
  the highest-value remaining task.
* Deca budget scaling (8/32/128 ns) answers an ABF question, not an mFR question. Optional
  appendix at most.
