# WCA dimer: how early can Fisher-Rao start? (FR-start ladder under the histogram estimator)

**Date:** 2026-10-01. **Status:** design and predictions written BEFORE any run. **GPU:** 3.

## Question

The accepted Case IX WCA cell starts FR at step 20 000 (t = 40) of 120 000: after the 10 000-step ABF
bias ramp and the 10 000-step estimator burn-in, by which time ABF alone has already populated both
dimer states (both marginals flatten by t ~ 6.5 on their own; WCA is not barrier-limited). The user
asks whether FR introduced EARLIER, before the two modes are populated, changes the picture, and
wants the WCA movie redone with the earlier start. The movie is one seed; this ladder is the paired
multi-seed backing it needs.

## Design

Frozen cell, sampler and estimator of the histogram replication (`configs/histogram_abf/campaign.json`,
`selected_bins.json`: 160 bins), FR knobs unchanged except `fr_start_steps` in {0, 2 500, 10 000,
20 000} (t = 0, 5, 20, 40: before the ramp has acted, mid-ramp, ramp end, the accepted value).
Seeds 3300-3307 (fresh labels), every arm of a seed in ONE process from the same lattice init and
noise stream (the campaign convention; the arms are paired in distribution, bitwise only to t ~ 0.4).
Arms: `hist_abf`, `hist_fr_s0`, `hist_fr_s2500`, `hist_fr_s10000`, `hist_fr_s20000`.
Endpoints (own read-out, the campaign's): e_F(T), I_F over the run, paired relative change vs the
seed's ABF arm (median, BCa CI, wins); replacements, windowed ESS/N, max lineage share; time to ABF's
final accuracy. Script `scripts/run_wca_fr_start_ladder.py`, outputs `results/wca_fr_start/`.

## Predictions (recorded before any run)

- P1 s20000 reproduces the confirmation (-50 % final / -23 % integrated, 16/16 there).
- P2 s10000 (ramp end) ~ s20000: the marginal is already flat at t = 20, so the start barely matters.
- P3 s0 and s2500 act while the bias is still ramping: a transient head start on I_F (more negative
  than s20000 by a few points) but no better endpoint, and more replacements with a lower windowed
  ESS; the early-dose harm seen on alanine/R15 is possible but the WCA marginal is the kind FR helps.
- P4 no start is HARMFUL at the frozen rate 0.10.

## Results (ladder run 2026-10-01 16:46-18:35 UTC; movies 18:35-18:47 UTC; read 2026-10-02 03:10 UTC)

### Ladder, 8 seeds (3300-3307), FR vs the seed's own ABF arm (`results/wca_fr_start/scoreboard.md`)

| arm | FR start (step / t) | e_F(T) median ABF / FR | d e_F(T) | I_F median ABF / FR | d I_F | replacements | windowed ESS/N min | max lineage share | time to ABF final accuracy, FR / ABF |
|---|---|---|---|---|---|---|---|---|---|
| hist_fr_s0 | 0 / t = 0 | 0.0911 / 0.0487 | -45.2 % [-48.0, -42.9] 8/8 | 41.45 / 26.27 | -37.7 % [-44.9, -30.8] 8/8 | 4634 | 0.637 | 0.037 | 28 / 185 |
| hist_fr_s10000 | 10000 / t = 20 | 0.0911 / 0.0501 | -45.8 % [-49.6, -40.3] 8/8 | 41.45 / 29.43 | -27.3 % [-36.0, -22.0] 8/8 | 4154 | 0.706 | 0.034 | 32 / 185 |
| hist_fr_s20000 | 20000 / t = 40 | 0.0911 / 0.0464 | -50.4 % [-52.3, -42.5] 8/8 | 41.45 / 31.13 | -26.2 % [-38.6, -17.3] 8/8 | 3584 | 0.726 | 0.034 | 60 / 185 |
| hist_fr_s2500 | 2500 / t = 5 | 0.0911 / 0.0499 | -43.5 % [-45.6, -42.2] 8/8 | 41.45 / 27.18 | -35.1 % [-45.5, -23.6] 8/8 | 4494 | 0.693 | 0.038 | 25 / 185 |

Direct paired contrasts between starts (same seeds):

| contrast | I_F | e_F(T) |
|---|---|---|
| FR from t = 0 vs FR from t = 40 | -13.0 % [-25.4, -6.3] 8/8 | +4.4 % [-1.3, +11.0] 2/8 |
| FR from t = 5 vs FR from t = 40 | -12.9 % [-16.9, -10.4] 7/8 | +6.5 % [-3.5, +13.9] 2/8 |
| FR from t = 20 vs FR from t = 40 | -4.1 % [-16.6, +7.3] 6/8 | +6.4 % [+1.4, +10.5] 0/8 |

**Reading.** P1 PASS: the accepted start reproduces the confirmation (-50.4 % final, -26.2 % integrated,
8/8). P2 PASS: the ramp-end start (t = 20) is indistinguishable from t = 40 (-45.8 % / -27.3 %). P3
PASS, in the predicted direction and larger than predicted: starting FR during the ramp buys a
substantial INTEGRATED gain (-37.7 % at t = 0 and -35.1 % at t = 5 vs -26.2 % at t = 40, 8/8 at every
start; directly, FR from t = 0 beats FR from t = 40 on I_F by -13.0 % [-25.4, -6.3], 8/8) and
reaches ABF's final accuracy at t = 25-28 instead of t = 60 (ABF alone: t = 185). The ENDPOINT pays
a small price: the endpoint gain shrinks from -50 % to -44..-46 %, i.e. the early-start arms end
4-6 % above the t = 40 arm (direct contrasts +4.4 % [-1.3, +11.0] at t = 0, +6.5 % [-3.5, +13.9] at
t = 5, +6.4 % [+1.4, +10.5] at t = 20; only the last excludes zero), and the genealogy cost is more
replacements (4 634 vs 3 584) and a lower windowed ESS (0.64 vs 0.73, floors met). P4 PASS: no start
is harmful. The
mechanism is visible in the movies: before t = 7 every FR death is in the over-represented compact
start and the births go to the transition / stretched regions, so selection moves the population over
the barrier while the ABF bias is still ramping; from t ~ 10 on the two settings run the same
steady state (deaths in the stretched state, births in transition / compact).

**Verdict.** On the WCA dimer, FR does not need the 40-time-unit burn-in: switching it on from the
start (or mid-ramp) trades ~5 points of endpoint gain for ~10 points of integrated gain and a 2.2x
earlier arrival at ABF's final accuracy, at a modest genealogy cost; the ramp-end start (t = 20) is
the accepted setting's equal integrated and slightly worse at the end. This is the opposite of the
alanine / R15 result (earlier start neutral-to-harmful), and consistent with the LTA rule "FR from
the warm-up end": the early start helps where the marginal is the bottleneck.

### Movies, seed 3200, one per start (`results/wca_movie_histogram{,_fr0,_fr2500,_fr10000}/`)

| FR start | this seed: e_F(T) ABF / FR | I_F ABF / FR | replacements (before t = 7, all compact) | FR reaches ABF's final accuracy at | windowed ESS/N |
|---|---|---|---|---|---|
| t = 0 | 0.0915 / 0.0534 (-42 %) | 39.1 / 29.6 (-24 %) | 4,671 (185) | t = 23 | 0.66 |
| t = 5 | 0.0934 / 0.0509 (-46 %) | 40.0 / 27.6 (-31 %) | 4,427 (58) | t = 21 | 0.69 |
| t = 20 | 0.1021 / 0.0571 (-44 %) | 50.4 / 29.4 (-42 %) | 4,226 (0) | t = 21 | 0.71 |
| t = 40 (original movie) | 0.0863 / 0.0443 (-49 %) | 38.3 / 30.9 (-19 %) | 3,511 (0) | t = 63 | 0.72 |

(The ABF column is re-simulated for every movie; its seed-to-seed scatter is the ~0.086-0.102 spread,
which is why the per-movie percentages wander more than the ladder's paired medians.)
