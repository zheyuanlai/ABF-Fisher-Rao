# V4 law check: A100 vs CPU on the validation seeds (generated)

Ratio of medians GPU / CPU (paired bootstrap 95 % CI); 'identical' = fraction of seeds whose value is bitwise equal on both engines.

| system | arm | pairs | final e_F | Ibar_F | u_mid | verdict |
|---|---|---|---|---|---|---|
| gateway | abf64 | 16 | 1 [0.999, 1] (identical 0.00) | 1 [0.999, 1] (identical 0.00) | 1 [1, 1] (identical 1.00) | PASS |
| gateway | fr64 | 16 | 1.06 [0.566, 1.43] (identical 0.00) | 1.01 [0.805, 1.24] (identical 0.00) | 1 [0.804, 1.3] (identical 0.75) | PASS |
| gateway | abf512 | 16 | 0.995 [0.989, 1] (identical 0.00) | 1 [1, 1] (identical 0.00) | 0.992 [0.983, 1] (identical 0.94) | PASS |
| gateway | fr512 | 16 | 1.27 [0.892, 1.47] (identical 0.00) | 1.01 [0.951, 1.03] (identical 0.00) | 1 [0.861, 1.05] (identical 0.75) | PASS |
| lta300 | abf64 | 16 | 1.07 [0.873, 1.21] (identical 0.00) | 1.03 [0.946, 1.13] (identical 0.00) | 1.16 [0.896, 1.43] (identical 0.25) | PASS |
| lta300 | fr64 | 16 | 0.757 [0.608, 1.03] (identical 0.00) | 0.876 [0.71, 1.05] (identical 0.00) | 0.835 [0.493, 1.94] (identical 0.00) | PASS |
| lta300 | abf256 | 16 | 0.856 [0.736, 1.22] (identical 0.00) | 0.991 [0.95, 1.02] (identical 0.00) | 1.01 [0.692, 1.12] (identical 0.25) | PASS |
| lta300 | fr256 | 16 | 0.901 [0.721, 1.34] (identical 0.00) | 0.986 [0.901, 1.12] (identical 0.00) | 0.926 [0.543, 1.31] (identical 0.06) | PASS |
| lta150 | abf64 | 16 | 0.973 [0.755, 1.25] (identical 0.00) | 1.02 [0.807, 1.08] (identical 0.00) | 1.07 [0.888, 1.27] (identical 0.44) | PASS |
| lta150 | fr64 | 16 | 1.26 [0.814, 1.69] (identical 0.00) | 1.05 [0.884, 1.26] (identical 0.00) | 1.11 [0.713, 2] (identical 0.19) | PASS |
| lta150 | abf256 | 16 | 0.981 [0.913, 1.09] (identical 0.00) | 1 [0.996, 1.01] (identical 0.00) | 0.983 [0.98, 1.04] (identical 0.50) | PASS |
| lta150 | fr256 | 16 | 1.59 [0.938, 1.85] (identical 0.00) | 1.03 [0.974, 1.07] (identical 0.00) | 1.01 [0.885, 1.13] (identical 0.38) | PASS |

Overall: PASS
