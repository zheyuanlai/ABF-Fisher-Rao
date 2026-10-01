# Alanine dipeptide, CV = (phi, psi): ABF vs ABF + Fisher-Rao birth-death

`alanine_ala2d_abf_vs_fr.mp4` (1920x1080, 30 fps, 35 s, 39 MB): one paired seed of the closed
alanine histogram study (`docs/ALANINE_HISTOGRAM_ABF.md`), recorded every 50 steps (0.05 ps),
two snapshots per frame. Both columns start from the same thermalised C7eq ensemble and see the
same Langevin noise; they differ only by the birth-death step.

| item | value |
|---|---|
| system | vacuum Ace-Ala-Nme, ff14SB, 300 K, BAOAB dt 1 fs, gamma 1 ps^-1; CV = (phi, psi) on the 97 x 97 torus grid |
| sampler | N 2048 walkers, 100 ps; histogram mean-force estimator (adaptive boxes, c_min 800), 5 ps bias ramp |
| FR arm | `fr_uniform`: torus-uniform target, from 0.5 ps, rate 0.15 (the dose at which events are visible; the campaign's frozen rate 0.02 fires ~3 per opportunity), fr_every 500, cap 5 % of N, score clip 2 |
| seed / init | seed 0, C7eq, `init_seed` 4242, `rng_seed` 20260903 |
| this seed | FES error (raw reference, equilibrium weight): ABF 0.532, FR 0.530 kJ/mol at 100 ps (floor 0.559 = reference noise); integrated over the run 95.8 vs 92.6 (-3 %); 4 183 deaths / 4 183 births; C7ax occupancy at the end 4.9 % vs 4.5 % |
| the closed study (16 seeds, `results/alanine_histogram/analysis/scoreboard.md`) | FR from 0.5 ps at rate 0.15: -0.68 % [-2.11, +0.93] integrated, NEUTRAL; at rate 0.02 -0.68 % [-2.18, +1.24], NEUTRAL |

Provenance: `configs/alanine_movie/ala2d.yaml`, `scripts/alanine_movie/launch.sh`, engine
`src/alanine/core2d_ala.py` with the report-only `store_snapshots` record (bit-inert,
`tests/test_alanine_movie.py`), renderer `scripts/make_alanine_movie.py --case ala2d`. Raw record
`abf/raw/*.npz`, `fr/raw/*.npz` (ignored, ~60 MB each; reproduce with the launcher on GPU 3).

## What each panel shows

* **Top left / top middle** - the Ramachandran plot of every walker (phi, psi in degrees) over the
  reference free energy on its full 0-90 kJ/mol range (white = wells, dark = the ~90 kJ/mol tops; colour
  bar on the left panel; contours at 5, 10, 20, 40, 60, 80 kJ/mol; hatched = the few cells the umbrella
  reference never sampled). Both angles are periodic, so a cloud cut off at one edge continues at the
  opposite edge, and because ABF flattens F the biased walkers cover the whole torus, high-energy
  regions included: dots in the dark areas are the ABF ensemble doing its job, not walkers leaving the domain. Dots are
  coloured by the watershed basin of the reference: C7eq blue, C5 green, C7ax orange, elsewhere
  purple. In the FR column the deaths of the last six snapshots are red crosses (at the dying
  walker's angles) and the births green rings (at the parent's angles); the counters are cumulative.
  The box at the bottom right gives the instantaneous basin occupancies.
* **Bottom left / bottom middle** - the free energy each arm has learned so far (its own exact
  piecewise-linear PMF, 0-90 kJ/mol, viridis) with the reference contours overlaid in white, and its
  error against the reference.
* **Right column** - the error of both arms against time (the dot marks the current frame); the
  KL distance of the walker marginal to the uniform torus target (what FR reads: it floors at 1.67
  because the sterically excluded part of the torus cannot be populated); the occupancy of the
  three wells against time, solid ABF, dashed FR, dotted the reference equilibrium populations.

## What to point at while it plays

* 0-4 ps: both clouds spill out of C7eq into C5 along psi as the bias ramps up; the first walkers
  cross the phi barrier into C7ax at ~4 ps in BOTH columns at the same moment (same noise).
* 5-10 ps: C7ax overshoots to ~25 % occupancy as the bias flattens the crossing, then relaxes; the
  marginal reaches its floor by ~15 ps. This is the whole establishment phase: 7 ps.
* FR column: the deaths cluster where the instantaneous marginal is densest (the well the cloud
  just arrived in) and the births land in the sparse regions, but the two learned surfaces and the
  two error curves stay on top of each other to the end: the walkers loop between the three wells
  identically with or without the birth-death step, which is the closed study's NEUTRAL verdict.
