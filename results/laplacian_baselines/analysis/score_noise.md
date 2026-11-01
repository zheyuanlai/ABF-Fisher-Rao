# Mechanism diagnostic: what does the Laplacian selection act on? (LTA cells)

Medians over up to 8 runs per arm.  TV(inst) = total variation between the instantaneous K-walker marginal and uniform, time-averaged after fr_start; TV(time-avg) = the same for the time-averaged marginal (the finite-K sampling noise is the difference).  'density at deaths / births' = marginal density, relative to uniform, where the arm killed / cloned (a flattening rule kills at > 1 and clones at < 1).

## LTA 150 K K1024

| arm | n | deaths | TV(inst) | TV(time-avg) | density at deaths | density at births | score std |
|---|---:|---:|---:|---:|---:|---:|---:|
| abf (production seeds) | 8 | 0 | 0.198 | 0.052 |  |  | 0.000 |
| fr (production seeds) | 8 | 364 | 0.181 | 0.037 | 1.06 | 0.98 | 0.286 |
| laplacian frozen scale (production seeds) | 8 | 254 | 0.198 | 0.051 | 0.99 | 0.99 | 0.106 |
| fr (pilot seeds) | 8 | 354 | 0.181 | 0.036 | 1.04 | 0.99 | 0.291 |
| laplacian 0.003 (pilot seeds) | 8 | 74 | 0.199 | 0.053 | 1.00 | 0.99 | 0.032 |
| laplacian 0.01 (pilot seeds) | 8 | 258 | 0.199 | 0.054 | 0.99 | 0.99 | 0.106 |
| laplacian 0.03 (pilot seeds) | 8 | 766 | 0.200 | 0.054 | 1.00 | 0.99 | 0.315 |
| laplacian 0.1 (pilot seeds) | 8 | 2482 | 0.202 | 0.055 | 0.99 | 0.98 | 0.960 |
| laplacian 0.3 (pilot seeds) | 8 | 5126 | 0.203 | 0.059 | 0.99 | 0.98 | 1.581 |

Emulated score field (production Laplacian run, K = 1024, engine bandwidth 0.1 rad, scale 0.01; saved score std 0.106 vs emulated 0.123):

| KDE bandwidth (rad) | corr(instantaneous p''/p, time-averaged p''/p) | std instantaneous (rad^-2) | std time-averaged (rad^-2) | noise variance fraction |
|---:|---:|---:|---:|---:|
| 0.1 | +0.07 | 12.32 | 0.82 | 0.996 |
| 0.2 | +0.22 | 2.24 | 0.56 | 0.937 |
| 0.3 | +0.43 | 0.90 | 0.43 | 0.777 |
| 0.5 | +0.61 | 0.30 | 0.24 | 0.362 |
| 0.8 | +0.30 | 0.11 | 0.10 | 0.219 |

## LTA 300 K K1024

| arm | n | deaths | TV(inst) | TV(time-avg) | density at deaths | density at births | score std |
|---|---:|---:|---:|---:|---:|---:|---:|
| abf (production seeds) | 8 | 0 | 0.178 | 0.017 |  |  | 0.000 |
| fr (production seeds) | 8 | 196 | 0.174 | 0.015 | 1.01 | 1.00 | 0.224 |
| laplacian frozen scale (production seeds) | 8 | 122 | 0.178 | 0.016 | 1.00 | 1.00 | 0.105 |
| fr (pilot seeds) | 8 | 202 | 0.173 | 0.016 | 1.01 | 1.00 | 0.220 |
| laplacian 0.003 (pilot seeds) | 8 | 34 | 0.176 | 0.019 | 1.01 | 0.99 | 0.032 |
| laplacian 0.01 (pilot seeds) | 8 | 119 | 0.176 | 0.019 | 0.99 | 0.99 | 0.106 |
| laplacian 0.03 (pilot seeds) | 8 | 364 | 0.177 | 0.018 | 1.00 | 1.00 | 0.316 |
| laplacian 0.1 (pilot seeds) | 8 | 1210 | 0.177 | 0.017 | 1.00 | 1.00 | 0.974 |
| laplacian 0.3 (pilot seeds) | 8 | 2300 | 0.178 | 0.017 | 1.00 | 1.00 | 1.594 |

Emulated score field (production Laplacian run, K = 1024, engine bandwidth 0.1 rad, scale 0.01; saved score std 0.106 vs emulated 0.114):

| KDE bandwidth (rad) | corr(instantaneous p''/p, time-averaged p''/p) | std instantaneous (rad^-2) | std time-averaged (rad^-2) | noise variance fraction |
|---:|---:|---:|---:|---:|
| 0.1 | +0.01 | 11.41 | 0.50 | 0.998 |
| 0.2 | +0.09 | 1.98 | 0.23 | 0.987 |
| 0.3 | +0.17 | 0.78 | 0.17 | 0.952 |
| 0.5 | +0.26 | 0.25 | 0.09 | 0.860 |
| 0.8 | +0.34 | 0.09 | 0.04 | 0.844 |

## LTA 80 K K1024

| arm | n | deaths | TV(inst) | TV(time-avg) | density at deaths | density at births | score std |
|---|---:|---:|---:|---:|---:|---:|---:|
| abf (production seeds) | 8 | 0 | 0.274 | 0.199 |  |  | 0.000 |
| fr (production seeds) | 8 | 492 | 0.213 | 0.108 | 1.21 | 0.94 | 0.413 |
| laplacian frozen scale (production seeds) | 8 | 703 | 0.274 | 0.198 | 1.01 | 0.99 | 0.316 |
| fr (pilot seeds) | 8 | 492 | 0.210 | 0.104 | 1.17 | 0.95 | 0.409 |
| laplacian 0.003 (pilot seeds) | 8 | 64 | 0.268 | 0.194 | 1.02 | 0.99 | 0.032 |
| laplacian 0.01 (pilot seeds) | 8 | 232 | 0.269 | 0.196 | 1.02 | 0.99 | 0.106 |
| laplacian 0.03 (pilot seeds) | 8 | 704 | 0.271 | 0.194 | 1.02 | 0.99 | 0.316 |
| laplacian 0.1 (pilot seeds) | 8 | 2196 | 0.279 | 0.211 | 1.02 | 1.00 | 0.944 |
| laplacian 0.3 (pilot seeds) | 8 | 4100 | 0.276 | 0.206 | 1.02 | 1.00 | 1.560 |

Emulated score field (production Laplacian run, K = 1024, engine bandwidth 0.1 rad, scale 0.03; saved score std 0.317 vs emulated 0.392):

| KDE bandwidth (rad) | corr(instantaneous p''/p, time-averaged p''/p) | std instantaneous (rad^-2) | std time-averaged (rad^-2) | noise variance fraction |
|---:|---:|---:|---:|---:|
| 0.1 | +0.10 | 13.07 | 1.61 | 0.985 |
| 0.2 | +0.44 | 2.26 | 1.09 | 0.768 |
| 0.3 | +0.72 | 1.01 | 0.86 | 0.266 |
| 0.5 | +0.88 | 0.47 | 0.55 | -0.410 |
| 0.8 | +0.95 | 0.27 | 0.32 | -0.408 |

## LTA 350 K K1024

| arm | n | deaths | TV(inst) | TV(time-avg) | density at deaths | density at births | score std |
|---|---:|---:|---:|---:|---:|---:|---:|
| abf (production seeds) | 8 | 0 | 0.174 | 0.013 |  |  | 0.000 |
| fr (production seeds) | 8 | 184 | 0.172 | 0.013 | 1.01 | 1.01 | 0.207 |
| laplacian frozen scale (production seeds) | 8 | 330 | 0.174 | 0.012 | 1.00 | 0.99 | 0.316 |
| fr (pilot seeds) | 8 | 178 | 0.172 | 0.014 | 0.99 | 1.00 | 0.213 |
| laplacian 0.003 (pilot seeds) | 8 | 31 | 0.174 | 0.014 | 1.00 | 1.01 | 0.032 |
| laplacian 0.01 (pilot seeds) | 8 | 96 | 0.174 | 0.014 | 1.00 | 1.00 | 0.105 |
| laplacian 0.03 (pilot seeds) | 8 | 320 | 0.175 | 0.014 | 1.00 | 1.00 | 0.315 |
| laplacian 0.1 (pilot seeds) | 8 | 1049 | 0.175 | 0.014 | 1.00 | 1.00 | 0.978 |
| laplacian 0.3 (pilot seeds) | 8 | 1908 | 0.174 | 0.014 | 1.00 | 1.00 | 1.595 |

Emulated score field (production Laplacian run, K = 1024, engine bandwidth 0.1 rad, scale 0.03; saved score std 0.317 vs emulated 0.345):

| KDE bandwidth (rad) | corr(instantaneous p''/p, time-averaged p''/p) | std instantaneous (rad^-2) | std time-averaged (rad^-2) | noise variance fraction |
|---:|---:|---:|---:|---:|
| 0.1 | +0.07 | 11.51 | 0.59 | 0.997 |
| 0.2 | +0.12 | 2.20 | 0.17 | 0.994 |
| 0.3 | +0.15 | 0.82 | 0.11 | 0.981 |
| 0.5 | +0.23 | 0.25 | 0.06 | 0.948 |
| 0.8 | +0.25 | 0.07 | 0.02 | 0.927 |
