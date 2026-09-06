# confirmatory: six arms on the compute axis (C* = 81.92 M walker-steps, v2 reference, window [2.0828125, 3.5382812500000003])

| arm | I_F^(C) | e_F(C*) | e_F(end of run) | D_cond(C*) | legacy I_F | C(eps_A)/C* | inner steps/seed | wall min |
|---|---|---|---|---|---|---|---|---|
| A | 2.500 | 1.358 | 1.358 | 0.368 | 2.494 | 1.00 (8/16) | 0.0 M | 5 |
| F | 2.514 | 1.390 | 1.390 | 0.368 | 2.503 | 1.00 (3/16) | 0.0 M | 17 |
| T | 2.097 | 1.025 | 1.025 | 0.368 | 2.170 | 0.70 (16/16) | 0.0 M | 7 |
| R | 2.945 | 2.016 | 1.285 | 0.381 | 2.909 | 1.75 (16/16) | 69.6 M | 7 |
| F+R | 2.946 | 2.021 | 1.290 | 0.381 | 2.911 | 1.75 (15/16) | 69.6 M | 18 |
| T+R | 2.556 | 1.476 | 1.030 | 0.364 | 2.550 | 1.15 (16/16) | 69.6 M | 8 |

| contrast | dI_F^(C) median [CI95] wins | d e_F(C*) | d D_cond(C*) | positive? | legacy dI_F |
|---|---|---|---|---|---|
| T vs A | -16.0% [-16.4, -15.6] 16/16 | -24.6% [-25.2, -23.5] 16/16 | +0.3% [-0.3, +0.7] 6/16 | YES | -13.1% |
| T vs F | -16.4% [-17.0, -16.1] 16/16 | -26.0% [-27.0, -25.2] 16/16 | +0.2% [-0.3, +0.6] 7/16 | YES | -13.4% |
| F vs A | +0.5% [+0.4, +0.8] 0/16 | +2.1% [+1.4, +2.9] 0/16 | +0.1% [+0.0, +0.1] 2/16 | no | +0.4% |
| R vs A | +17.8% [+17.1, +18.5] 0/16 | +47.2% [+45.7, +50.5] 0/16 | +3.7% [+3.5, +3.8] 0/16 | no | +16.6% |
| F+R vs F | +17.0% [+16.6, +17.6] 0/16 | +44.4% [+43.7, +47.6] 0/16 | +3.6% [+3.4, +3.8] 0/16 | no | +15.9% |
| T+R vs T | +21.8% [+20.8, +22.4] 0/16 | +44.0% [+41.9, +45.4] 0/16 | -1.4% [-1.8, -0.6] 16/16 | no | +17.6% |
| T+R vs R | -13.2% [-13.6, -12.7] 16/16 | -26.8% [-27.7, -25.6] 16/16 | -4.5% [-4.6, -4.4] 16/16 | YES | -12.3% |
| T+R vs F+R | -13.3% [-13.6, -12.7] 16/16 | -26.9% [-27.8, -25.7] 16/16 | -4.5% [-4.6, -4.4] 16/16 | YES | -12.4% |
| T+R vs A | +2.4% [+1.7, +3.1] 0/16 | +8.4% [+6.3, +11.3] 0/16 | -1.1% [-1.1, -0.9] 16/16 | no | +2.2% |
| F+R vs A | +17.9% [+17.2, +18.5] 0/16 | +47.7% [+45.9, +50.3] 0/16 | +3.7% [+3.5, +3.9] 0/16 | no | +16.7% |

Mechanism (deposit-free, moved walkers, window bins with >= 200 samples): RMS of <f | R> - F'_v2 and conditional TV, before (pre) and after (post) repair.

- A: final smoothed mean-force RMS error 7.329; raw deposit bias RMS (all deposits) 9.175
- F: final smoothed mean-force RMS error 7.405; raw deposit bias RMS (all deposits) 9.181
- T: final smoothed mean-force RMS error 7.191; raw deposit bias RMS (all deposits) 8.313; post-event deposit bias RMS pre nan / post 8.294 (|F'| RMS 16.75); conditional TV of moved walkers pre nan / post 0.373; |dR| mean 0.0025, capped 0.000, moved 1.000
- R: final smoothed mean-force RMS error 7.137; raw deposit bias RMS (all deposits) 9.115
- F+R: final smoothed mean-force RMS error 7.143; raw deposit bias RMS (all deposits) 9.111
- T+R: final smoothed mean-force RMS error 7.382; raw deposit bias RMS (all deposits) 8.299; post-event deposit bias RMS pre 8.330 / post 8.236 (|F'| RMS 16.75); conditional TV of moved walkers pre 0.379 / post 0.379; |dR| mean 0.0026, capped 0.000, moved 1.000

Go to confirmatory: **True**
