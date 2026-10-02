# WCA-dimer movie, histogram estimator, Fisher-Rao from t = 5: ABF vs ABF + Fisher-Rao birth-death

`wca_abf_vs_fr_histogram.mp4` (1920x1080, 30 fps, 42 s, 30 MB): the same seed 3200, cell, estimator and
layout as `results/wca_movie_histogram/` (FR from t = 40) and `results/wca_movie_histogram_fr0/` (FR from
t = 0), with Fisher-Rao switched on at step 2500 (t = 5): mid-ramp (the ABF bias is at 25 % of full strength). One of four movies that differ only
in the FR start; the 8-seed ladder behind them is `results/wca_fr_start/scoreboard.md` (`docs/WCA_FR_START.md`).

| item | value |
|---|---|
| cell / sampler / estimator | as the t = 40 movie's README: beta 1, h 2, w 2, 100 particles; N 1024, dt 0.002, 120 000 steps (T = 240); histogram 160 bins, own read-out; ABF ramp 10 000 steps, reported estimator from the 10 000-step burn-in |
| FR arm | `fr_uniform`, rate 0.1, every 5 steps **from step 2500 (t = 5)**, score clip 2, cap 2 %, KDE 0.07 |
| this seed (3200) | e_F(T): ABF 0.0934, FR 0.0509 (-45.5 %); I_F 40.0 vs 27.6 (-31.0 %); 4,427 replacements (58 before t = 7, all in the compact start); windowed ESS/N 0.686, max lineage share 0.030; FR reaches ABF's final accuracy for good at t = 21; error ratio FR/ABF 0.92 at t = 10, 0.40 at 40, 0.49 at 120, 0.54 at the end |
| ladder medians for this start (8 seeds) | final -43.5 % [-45.6, -42.2], integrated -35.1 % [-45.5, -23.6], both 8/8; time to ABF's final accuracy t = 25 (ABF 185) |
| the accepted start (t = 40), same ladder | final -50.4 %, integrated -26.2 %, t = 60 |

Provenance: `scripts/make_wca_movie.py simulate --seed 3200 --fr-start 2500 --out results/wca_movie_histogram_fr2500`
then `render`; the frames read the FR arm's own `fr_start_steps`; data `movie_data.npz`, numbers
`movie_summary.json`. The ABF column is a fresh run of seed 3200 (the WCA force kernel's atomics make
repeats differ in detail; columns are paired in distribution).

## What each panel shows

Identical to the t = 40 movie (see its README): configuration of replica 0 | every replica as a bead
on beta F_ref(xi) with FR deaths (red x) and births (green o) flashed | replica histogram vs the uniform
target | learned histogram PMF vs the TI reference | e_F(t) and the stretched fraction. The "FR starts"
cursor sits at t = 5.

## What to point at while it plays

1. t < 5: plain ABF in both columns; the clouds leave the compact start as the bias ramps up.
2. t = 5: FR switches on mid-ramp. The first deaths (58 before t = 7) are all in the still
   over-represented compact state and the births land in the transition / stretched regions: selection
   finishes the job the ramping bias has started. The error ratio is 0.92 at t = 10 and 0.40 at t = 40.
3. t = 20: the reported estimator restarts (spike in both columns). t = 21: the FR column is already at
   the accuracy ABF reaches only at the end of the run (the t = 40 movie needs t = 63).
4. t > 20: the steady state of every start: deaths in the stretched state (ABF leaves ~1.4x the uniform
   density there), births in transition / compact; ratio ~0.4-0.55 to the end.
