# WCA FR-start ladder (histogram estimator, 160 bins, frozen cell): FR vs the seed's ABF arm

Seeds 3300-3307 (n = 8); own read-out; paired relative change (median, BCa 95 % CI, wins).

| arm | FR start (step / t) | e_F(T) median ABF / FR | d e_F(T) | I_F median ABF / FR | d I_F | replacements | windowed ESS/N min | max lineage share | time to ABF final accuracy, FR / ABF |
|---|---|---|---|---|---|---|---|---|---|
| hist_fr_s0 | 0 / t = 0 | 0.0911 / 0.0487 | -45.2 % [-48.0, -42.9] 8/8 | 41.45 / 26.27 | -37.7 % [-44.9, -30.8] 8/8 | 4634 | 0.637 | 0.037 | 28 / 185 |
| hist_fr_s10000 | 10000 / t = 20 | 0.0911 / 0.0501 | -45.8 % [-49.6, -40.3] 8/8 | 41.45 / 29.43 | -27.3 % [-36.0, -22.0] 8/8 | 4154 | 0.706 | 0.034 | 32 / 185 |
| hist_fr_s20000 | 20000 / t = 40 | 0.0911 / 0.0464 | -50.4 % [-52.3, -42.5] 8/8 | 41.45 / 31.13 | -26.2 % [-38.6, -17.3] 8/8 | 3584 | 0.726 | 0.034 | 60 / 185 |
| hist_fr_s2500 | 2500 / t = 5 | 0.0911 / 0.0499 | -43.5 % [-45.6, -42.2] 8/8 | 41.45 / 27.18 | -35.1 % [-45.5, -23.6] 8/8 | 4494 | 0.693 | 0.038 | 25 / 185 |
