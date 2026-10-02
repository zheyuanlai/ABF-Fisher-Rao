# WCA-dimer movie, histogram estimator, Fisher-Rao from the START: ABF vs ABF + Fisher-Rao birth-death

`wca_abf_vs_fr_histogram.mp4` (1920x1080, 30 fps, 42 s, 30 MB) is the companion of
`results/wca_movie_histogram/wca_abf_vs_fr_histogram.mp4` with ONE change: Fisher-Rao switches on at
step 0 (first opportunity at step 5, t = 0.01) instead of step 20 000 (t = 40). Same frozen Case IX
cell, same histogram estimator (160 bins), same seed 3200, same lattice initial condition; in the
accepted setting FR only starts once ABF has already populated both dimer states (both marginals
flatten by t ~ 6.5 on their own), which is what the user asked to change.

| item | value |
|---|---|
| cell / sampler / estimator | as `results/wca_movie_histogram/README.md`: beta 1, h 2, w 2, 100 particles; N 1024, dt 0.002, 120 000 steps (T = 240), ABF ramp 10 000 steps, reported estimator from the 10 000-step burn-in; histogram 160 bins, own read-out |
| FR arm | `fr_uniform`, rate 0.1, every 5 steps **from step 0** (t = 0), score clip 2, cap 2 %, KDE 0.07 |
| this seed (3200) | e_F(T): ABF 0.0915, FR 0.0534 (**-41.6 %**); I_F 39.1 vs 29.6 (**-24.4 %**); 4 671 replacements; windowed ESS/N 0.658, max lineage share 0.033; FR reaches ABF's final accuracy for good at **t = 23** (ABF: never before T; the t = 40 movie: t = 63) |
| same seed with FR from t = 40 (`results/wca_movie_histogram`) | e_F(T) -49 %, I_F -19 %, 3 511 replacements, windowed ESS/N 0.717 |
| confirmation medians (16 seeds, FR from t = 40) | -50.5 % final, -23.1 % integrated |
| multi-seed backing for the early start | `results/wca_fr_start/scoreboard.md` (8 fresh seeds x FR start {0, 2 500, 10 000, 20 000}; `docs/WCA_FR_START.md`) |

Provenance: `scripts/make_wca_movie.py simulate --seed 3200 --fr-start 0 --out results/wca_movie_histogram_fr0`
(stage label `movie_fr0`; the frames read the FR arm's own `fr_start_steps`), engine and record as in
the t = 40 movie (`store_snapshots`, bit-inert). Data `movie_data.npz`, numbers `movie_summary.json`.
The ABF arm is a fresh run of the same seed (0.0915 vs 0.0863 in the t = 40 movie: the WCA force
kernel's atomics make repeats differ in detail; the columns are paired in distribution).

## What each panel shows

Identical to the t = 40 movie (see its README): configuration of replica 0 | every replica as a bead
on beta F_ref(xi) with FR deaths (red x) and births (green o) flashed | replica histogram vs the
uniform target | learned histogram PMF vs the TI reference | e_F(t) and the stretched fraction.
The "FR starts" cursor in the bottom-left panel now sits at t = 0.

## What to point at while it plays

1. t < 7, the part that is new: Fisher-Rao is acting while the ABF bias is still ramping up and
   the replicas are still leaving the compact lattice start. Every one of the 185 deaths before
   t = 7 is in the over-represented COMPACT state, and their births land in the transition region
   (109) and the stretched state (48): selection pushes the population over the barrier before the
   bias has flattened it. At t = 5 the stretched fraction is 0.28 in the FR column against 0.24 for
   ABF, and the FR error is already 0.91x ABF's at t = 5 and 0.79x at t = 10.
2. t = 20: the reported estimator restarts (burn-in spike in both columns; the ratio is meaningless
   for a few time units). t = 23: the FR column is already at the accuracy ABF only reaches at the
   end of the run (the t = 40 movie needed t = 63 for this).
3. t > 20: the familiar steady state. 4 434 of the 4 671 deaths are in the stretched state (ABF
   alone leaves ~1.4x the uniform density there), births go to transition (2 141) and compact
   (1 959); the error ratio is 0.48 at t = 40, 0.38 at t = 80, 0.48 at t = 120 and 0.58 at the end.
4. The honest comparison with the t = 40 start on this seed: the early start buys a larger
   INTEGRATED gain (-24 % vs -19 %) through the head start in t < 25, at the cost of more
   replacements (4 671 vs 3 511) and a lower windowed ESS (0.66 vs 0.72); the endpoint is noise-level
   different (-42 % vs -49 %). Whether that holds across seeds is what the 8-seed ladder measures.
