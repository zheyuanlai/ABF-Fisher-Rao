# pilot: six arms on the compute axis (C* = 81.92 M walker-steps, v2 reference, window [2.0828125, 3.5382812500000003])

| arm | I_F^(C) | e_F(C*) | e_F(end of run) | D_cond(C*) | legacy I_F | C(eps_A)/C* | inner steps/seed | wall min |
|---|---|---|---|---|---|---|---|---|
| A | 2.526 | 1.419 | 1.419 | 0.368 | 2.518 | 0.97 (4/8) | 0.0 M | 5 |
| F | 2.546 | 1.463 | 1.463 | 0.369 | 2.536 | 0.93 (2/8) | 0.0 M | 14 |
| T | 2.133 | 1.061 | 1.061 | 0.366 | 2.191 | 0.68 (8/8) | 0.0 M | 6 |
| R | 2.934 | 2.049 | 1.338 | 0.382 | 2.893 | 1.75 (8/8) | 69.6 M | 6 |
| F+R | 2.937 | 2.051 | 1.344 | 0.383 | 2.897 | 1.75 (8/8) | 69.6 M | 14 |
| T+R | 2.572 | 1.503 | 1.033 | 0.364 | 2.561 | 1.15 (8/8) | 69.6 M | 7 |

| contrast | dI_F^(C) median [CI95] wins | d e_F(C*) | d D_cond(C*) | positive? | legacy dI_F |
|---|---|---|---|---|---|
| T vs A | -15.4% [-15.8, -14.1] 8/8 | -24.4% [-25.5, -22.0] 8/8 | -0.5% [-1.5, +0.1] 6/8 | YES | -12.7% |
| T vs F | -16.1% [-16.5, -14.9] 8/8 | -27.0% [-27.8, -24.3] 8/8 | -0.6% [-1.8, +0.2] 6/8 | YES | -13.3% |
| F vs A | +0.9% [+0.7, +1.0] 0/8 | +3.5% [+2.7, +3.6] 0/8 | +0.2% [+0.0, +0.3] 1/8 | no | +0.7% |
| R vs A | +16.9% [+15.8, +18.2] 0/8 | +45.5% [+41.0, +49.6] 0/8 | +3.7% [+3.2, +4.1] 0/8 | no | +15.8% |
| F+R vs F | +15.9% [+14.9, +17.6] 0/8 | +40.6% [+37.2, +44.6] 0/8 | +3.6% [+3.0, +3.8] 0/8 | no | +14.9% |
| T+R vs T | +20.4% [+18.1, +22.3] 0/8 | +42.1% [+37.8, +43.1] 0/8 | -0.8% [-1.3, +0.4] 5/8 | no | +16.8% |
| T+R vs R | -12.5% [-13.3, -12.0] 8/8 | -25.9% [-27.4, -24.6] 8/8 | -4.6% [-5.1, -4.3] 8/8 | YES | -11.8% |
| T+R vs F+R | -12.6% [-13.3, -12.2] 8/8 | -25.9% [-27.5, -24.9] 8/8 | -4.7% [-5.1, -4.3] 8/8 | YES | -11.8% |
| T+R vs A | +2.0% [+1.3, +3.5] 0/8 | +7.2% [+4.6, +11.4] 0/8 | -1.2% [-1.5, -0.9] 8/8 | no | +1.9% |
| F+R vs A | +17.0% [+15.9, +18.2] 0/8 | +45.6% [+41.2, +49.7] 0/8 | +3.7% [+3.1, +4.1] 0/8 | no | +15.8% |

Mechanism (deposit-free, moved walkers, window bins with >= 200 samples): RMS of <f | R> - F'_v2 and conditional TV, before (pre) and after (post) repair.

- A: final smoothed mean-force RMS error 7.477
- F: final smoothed mean-force RMS error 7.597
- T: final smoothed mean-force RMS error 7.115; post-event deposit bias RMS pre nan / post 8.264 (|F'| RMS 16.75); conditional TV of moved walkers pre nan / post 0.369; |dR| mean 0.0025, capped 0.000, moved 1.000
- R: final smoothed mean-force RMS error 7.255
- F+R: final smoothed mean-force RMS error 7.265
- T+R: final smoothed mean-force RMS error 7.358; post-event deposit bias RMS pre 8.322 / post 8.230 (|F'| RMS 16.75); conditional TV of moved walkers pre 0.378 / post 0.378; |dR| mean 0.0026, capped 0.000, moved 1.000

Go to confirmatory: **True**
