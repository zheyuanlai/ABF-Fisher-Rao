# EXPLORATORY: Laplacian selection with a wider KDE bandwidth (LTA 150 K and 80 K, fresh seeds)

Not part of the predeclared protocol (docs/LRS_LAPLACIAN_BASELINE.md).  Motivation: score_noise.md shows that at the 0.1 rad bandwidth shared with FR the Laplacian score is ~99 % finite-K noise.  Here ABF, FR (unchanged, bandwidth 0.1) and Laplacian variants with kde_bandwidth {0.3, 0.5} rad x scale {0.1, 0.3, 1, 3} run on the same 8 fresh seeds.  G = (second - first) / first, median with seed-bootstrap 95 % CI, (wins / n).  Negative favours the second arm.  TV(time-avg) = total variation of the time-averaged post-fr_start marginal from uniform; 'density at deaths' = marginal density relative to uniform where the arm killed (a flattening rule kills at > 1).

## LTA 150 K K1024 (8 fresh seeds)

| arm | bw | scale | replacements | min ESS/K | safety stops | TV(time-avg) | density at deaths | e_F(T) median | I_F median | G e_F: vs ABF | G I_F: vs ABF | G e_F: vs FR | G I_F: vs FR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| ABF | | | 0 | | | 0.052 | | 0.1334 | 0.4547 | | | | |
| FR | 0.1 | | 378 | 0.589 | 0 | 0.038 | 1.04 | 0.05494 | 0.3105 | -61.5 % [-68.1, -55.3] (8/8) | -32.8 % [-34.8, -28.0] (8/8) | | |
| Laplacian | 0.3 | 0.1 | 212 | 0.915 | 0 | 0.048 | 1.00 | 0.1209 | 0.428 | -9.1 % [-15.0, -0.6] (7/8) | -4.9 % [-6.3, -3.2] (8/8) | +137.0 % [+104.4, +199.1] (0/8) | +43.3 % [+32.3, +44.5] (0/8) |
| Laplacian | 0.3 | 0.3 | 505 | 0.783 | 0 | 0.040 | 1.01 | 0.09584 | 0.3967 | -27.4 % [-32.7, -18.8] (8/8) | -12.4 % [-16.7, -12.1] (8/8) | +90.9 % [+50.0, +154.3] (0/8) | +26.6 % [+20.7, +32.5] (0/8) |
| Laplacian | 0.3 | 1 | 994 | 0.536 | 0 | 0.032 | 1.04 | 0.07241 | 0.3453 | -45.3 % [-51.2, -30.1] (8/8) | -25.9 % [-28.0, -19.9] (8/8) | +50.5 % [+17.2, +117.0] (0/8) | +9.8 % [+6.8, +14.5] (0/8) |
| Laplacian | 0.3 | 3 | 1542 | 0.390 | 0 | 0.028 | 1.03 | 0.049 | 0.3244 | -66.0 % [-69.0, -57.2] (8/8) | -32.1 % [-38.9, -27.9] (8/8) | -9.8 % [-29.0, +27.3] (5/8) | -2.4 % [-5.8, +5.6] (5/8) |
| Laplacian | 0.5 | 0.1 | 106 | 0.938 | 0 | 0.048 | 0.98 | 0.1216 | 0.4356 | -8.0 % [-15.9, -6.1] (8/8) | -4.8 % [-6.7, -4.0] (8/8) | +136.2 % [+102.8, +178.4] (0/8) | +41.8 % [+33.4, +45.1] (0/8) |
| Laplacian | 0.5 | 0.3 | 256 | 0.833 | 0 | 0.040 | 1.00 | 0.09187 | 0.3934 | -27.7 % [-33.8, -23.5] (8/8) | -15.4 % [-17.0, -12.3] (8/8) | +82.0 % [+48.3, +139.6] (0/8) | +27.4 % [+17.6, +30.5] (0/8) |
| Laplacian | 0.5 | 1 | 510 | 0.569 | 0 | 0.034 | 1.05 | 0.05735 | 0.3201 | -58.0 % [-61.9, -49.6] (8/8) | -29.6 % [-31.5, -27.7] (8/8) | +9.4 % [-5.7, +48.2] (2/8) | +3.2 % [-1.1, +7.0] (2/8) |
| Laplacian | 0.5 | 3 | 730 | 0.388 | 0 | 0.031 | 1.06 | 0.03719 | 0.2909 | -73.7 % [-77.4, -58.6] (8/8) | -38.8 % [-39.9, -34.1] (8/8) | -23.4 % [-45.9, +1.3] (6/8) | -8.8 % [-12.3, -4.1] (8/8) |
