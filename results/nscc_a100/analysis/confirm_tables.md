# Confirmation (fresh seeds, selection frozen) (generated)

| system | ABF arm | FR arm | n ABF / FR | median W_mid ABF / FR (s) | R_W [95 % CI] | R_e2e [95 % CI] | censored ABF / FR | Holm alpha | decision |
|---|---|---|---|---|---|---|---|---|---|
| gateway | ABF 32 P4 | FR 32 P4 | 128 / 128 | 2.83 / 2.47 | 0.871 [0.654, 1.23] | 0.978 [0.822, 1.14] | 0 / 0 | 0.05 | **unresolved** |
| lta300 | ABF 512 G1 | FR 512 G1 | 64 / 64 | 3.61 / 4.5 | 1.25 [1.05, 1.55] | 1.22 [1.05, 1.49] | 4 / 5 | -- | **ABF faster (secondary)** |
| lta150 | ABF 128 G1 | FR 256 G1 | 128 / 128 | 3.06 / 2.2 | 0.718 [0.642, 0.833] | 0.783 [0.717, 0.876] | 0 / 0 | 0.025 | **FR faster** |

R = median FR / median ABF; < 1 favours FR. Secondary thresholds:

| system | R loose (W) | R strict (W) |
|---|---|---|
| gateway | 0.677 [0.585, 0.786] | 1.21 [0.904, 1.55] |
| lta300 | 0.947 [0.87, 1.03] | n/a [n/a, n/a] |
| lta150 | 0.776 [0.693, 0.904] | n/a [n/a, n/a] |
