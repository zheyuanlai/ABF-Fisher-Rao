# Ace-Val-Nme -- decision brief

**Verdict: FAIL-B ABF already sufficient. Ace-Val-Nme is an atomistic neutrality control, not an mFR-positive benchmark.**

## The decisive numbers

Over a 300 ps run with 16 seeds x 2048 walkers, multi-replica ABF on xi = (phi, chi1) reaches every one of the 8 regions within **5.4 ps** (first touch, every seed), holds them persistently from **18.0 ps**, and establishes every one of them within **52 ps**. The rarest region carries a pilot population of 0.0014 and still ends at 1.46 of its bias-aware target. ABF's own free energy lands **0.248 kT** from the pilot reference (marginal TV 0.069).

The worst relative deficit over the second half of the run is **0.223** against a 0.50 threshold, and no region sits below half its target for more than **6.0%** of the run against a 20 % threshold. The necessary condition for mFR to act -- a discovered state that stays under-populated -- fails, and it fails by a wide margin rather than marginally.

## What this licenses

* Val joins alanine as a **second atomistic neutrality control**, and it is the stronger of the two: alanine was neutral on a CV with no meaningfully rare state, whereas Val was selected for a real side-chain barrier and cleared every prior gate -- V1, the distinguishability gate at 0.973 balanced accuracy -- before failing V3.

* The corrected Stage-0 reading must travel with the result: the 11-18 kT chi1 barriers were **backbone-clamped conditional** barriers. With the backbone free the 2-D min-max path costs 1.1-7.4 kT and rotamers interconvert at 2.70 changes per walker per ns. The genuinely slow coordinate is **phi** (4 crossings in 2581 ns), and phi is in the CV -- which is exactly why ABF succeeds here.

## What this forbids

* Do **not** run the Val oracle mFR arm, the sham arm, or the full 576-window reference. All three existed only to support or defend a positive result.

* Do **not** shorten the run, cut walkers, or lower the establishment band until a deficit appears. Both `T_hit < 0.1 T_run` and "starved for >= 0.2 T_run" scale with run length, so a shorter run flatters the result instead of testing it.

* Do **not** read this as evidence that mFR fails in general. It is evidence about a **regime**: when ABF's CV contains the slow coordinate, ABF establishes the populations by itself and marginal reallocation has no deficit to repair.

## Known limitations of the artifacts, corrected but not re-run

* The omitted-psi check was confounded by interior weighting; it is re-derived at matched CV cells here (worst-region TV 0.055 matched, against 0.108 region-aggregated). Conditions 4 and 5 only ever gated a PASS, so neither can change a FAIL-B.

* The entry counter reported zero entries into every region behind the corridor. That was a counting artifact, now fixed; the regions were demonstrably reached, as their finite T_hit shows.

* An earlier target normalised over cells the pilot never sampled and put 97 % of the target mass there. The guard is now an assertion rather than a printed diagnostic.
