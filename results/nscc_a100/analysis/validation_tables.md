# Stage I gates of the A100 engine (generated)

## gateway

host x1000c3s0b0n1, 2026-10-26T06:47:59Z, 5 s

| gate | passed | key numbers |
|---|---|---|
| RNG1 | PASS | 1048576 doubles, bitwise True |
| RNG2 | PASS | 204800 normals, max abs diff 4.44e-16, bitwise fraction 0.862 |
| FR | PASS | 600 opportunities (N 16, 64, 512, 2048), 218 with events, mismatches 0 |
| TRAJ | PASS | abf64 40000 steps: counts identical over the whole run True, max rel dM (first quarter) 6.37e-11, deaths CPU/GPU 0/0, final position diff 1.39e-13<br>fr64 40000 steps: counts identical over the whole run True, max rel dM (first quarter) 6.37e-11, deaths CPU/GPU 2/2, final position diff 2.73e-13<br>fr512 20000 steps: counts identical over the whole run True, max rel dM (first quarter) 4.48e-11, deaths CPU/GPU 7/7, final position diff 3.46e-14<br>fr2048 8000 steps: counts identical over the whole run True, max rel dM (first quarter) 2.57e-11, deaths CPU/GPU 9/9, final position diff 2.44e-14 |
| INV | PASS | 8 launch configurations, identical True |
| DET | PASS |  |
| CTR | PASS | check_contract passed on every TRAJ GPU run |

## lta300

host x1000c3s0b0n1, 2026-10-26T06:48:25Z, 12 s

| gate | passed | key numbers |
|---|---|---|
| RNG1 | PASS | 1048576 doubles, bitwise True |
| RNG2 | PASS | 38400 normals, max abs diff 4.44e-16, bitwise fraction 0.863 |
| FORCE | PASS | G=8: max rel 1.87e-16, bins identical True; G=32: max rel 1.87e-16, bins identical True; G=128: max rel 1.87e-16, bins identical True; G=256: max rel 1.87e-16, bins identical True |
| FR | PASS | 600 opportunities (N 2, 16, 64, 256, 1024), 180 with events, mismatches 0, max score diff 8.88e-16 |
| TRAJ | PASS | fr16 40000 steps: counts identical over the whole run True, max rel dM (first quarter) 9.75e-12, deaths CPU/GPU 3/3, final position diff 1.18e-08<br>fr64 30000 steps: counts identical over the whole run True, max rel dM (first quarter) 5.48e-12, deaths CPU/GPU 3/3, final position diff 2.33e-10<br>abf256 24000 steps: counts identical over the whole run True, max rel dM (first quarter) 1.41e-11, deaths CPU/GPU 0/0, final position diff 1.65e-10<br>fr1024 23000 steps: counts identical over the whole run True, max rel dM (first quarter) 7.02e-12, deaths CPU/GPU 24/24, final position diff 5.68e-11 |
| INV | PASS | 4 launch configurations, identical True |
| DET | PASS |  |
| CTR | PASS | check_contract passed on every TRAJ GPU run |

## lta150

host x1000c3s0b0n1, 2026-10-26T06:48:12Z, 12 s

| gate | passed | key numbers |
|---|---|---|
| RNG1 | PASS | 1048576 doubles, bitwise True |
| RNG2 | PASS | 38400 normals, max abs diff 4.44e-16, bitwise fraction 0.862 |
| FORCE | PASS | G=8: max rel 1.42e-16, bins identical True; G=32: max rel 1.42e-16, bins identical True; G=128: max rel 1.42e-16, bins identical True; G=256: max rel 2.13e-16, bins identical True |
| FR | PASS | 600 opportunities (N 2, 16, 64, 256, 1024), 181 with events, mismatches 0, max score diff 1.11e-15 |
| TRAJ | PASS | fr16 40000 steps: counts identical over the whole run True, max rel dM (first quarter) 9.23e-12, deaths CPU/GPU 4/4, final position diff 2.52e-10<br>fr64 30000 steps: counts identical over the whole run True, max rel dM (first quarter) 2.49e-11, deaths CPU/GPU 6/6, final position diff 2.79e-13<br>abf256 24000 steps: counts identical over the whole run True, max rel dM (first quarter) 3.94e-11, deaths CPU/GPU 0/0, final position diff 8.29e-12<br>fr1024 23000 steps: counts identical over the whole run True, max rel dM (first quarter) 4.95e-11, deaths CPU/GPU 31/31, final position diff 7.04e-12 |
| INV | PASS | 4 launch configurations, identical True |
| DET | PASS |  |
| CTR | PASS | check_contract passed on every TRAJ GPU run |

