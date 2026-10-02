# WCA-dimer movie, histogram estimator, Fisher-Rao from t = 20: ABF vs ABF + Fisher-Rao birth-death

`wca_abf_vs_fr_histogram.mp4` (1920x1080, 30 fps, 42 s, 30 MB): the same seed 3200, cell, estimator and
layout as `results/wca_movie_histogram/` (FR from t = 40) and `results/wca_movie_histogram_fr0/` (FR from
t = 0), with Fisher-Rao switched on at step 10000 (t = 20): the end of the ABF bias ramp, when the reported estimator restarts. One of four movies that differ only
in the FR start; the 8-seed ladder behind them is `results/wca_fr_start/scoreboard.md` (`docs/WCA_FR_START.md`).

| item | value |
|---|---|
| cell / sampler / estimator | as the t = 40 movie's README: beta 1, h 2, w 2, 100 particles; N 1024, dt 0.002, 120 000 steps (T = 240); histogram 160 bins, own read-out; ABF ramp 10 000 steps, reported estimator from the 10 000-step burn-in |
| FR arm | `fr_uniform`, rate 0.1, every 5 steps **from step 10000 (t = 20)**, score clip 2, cap 2 %, KDE 0.07 |
| this seed (3200) | e_F(T): ABF 0.1021, FR 0.0571 (-44.0 %); I_F 50.4 vs 29.4 (-41.7 %); 4,226 replacements (0 before t = 7); windowed ESS/N 0.714, max lineage share 0.040; FR reaches ABF's final accuracy for good at t = 21; error ratio FR/ABF 0.99 at t = 10, 0.30 at 40, 0.49 at 120, 0.56 at the end |
| ladder medians for this start (8 seeds) | final -45.8 % [-49.6, -40.3], integrated -27.3 % [-36.0, -22.0], both 8/8; time to ABF's final accuracy t = 32 (ABF 185) |
| the accepted start (t = 40), same ladder | final -50.4 %, integrated -26.2 %, t = 60 |

Provenance: `scripts/make_wca_movie.py simulate --seed 3200 --fr-start 10000 --out results/wca_movie_histogram_fr10000`
then `render`; the frames read the FR arm's own `fr_start_steps`; data `movie_data.npz`, numbers
`movie_summary.json`. The ABF column is a fresh run of seed 3200 (the WCA force kernel's atomics make
repeats differ in detail; columns are paired in distribution).

## What each panel shows

Identical to the t = 40 movie (see its README): configuration of replica 0 | every replica as a bead
on beta F_ref(xi) with FR deaths (red x) and births (green o) flashed | replica histogram vs the uniform
target | learned histogram PMF vs the TI reference | e_F(t) and the stretched fraction. The "FR starts"
cursor sits at t = 20.

## What to point at while it plays

1. t < 20: plain ABF in both columns; both marginals flatten by t ~ 6.5 on their own, so by the time FR
   switches on the two dimer states are already populated (no compact-start deaths at all).
2. t = 20: FR switches on exactly when the bias ramp ends and the reported estimator restarts (spike in
   both columns). The ratio falls from 0.48 at t = 20 to 0.30 at t = 40; FR reaches ABF's final
   accuracy at t = 21 (the t = 40 start on this seed: t = 63).
3. t > 20: the familiar steady state: deaths in the stretched state, births in transition / compact;
   ratio ~0.3-0.55 to the end. Across the 8-seed ladder this start is indistinguishable from the
   accepted t = 40 one on both endpoint and integrated error.
