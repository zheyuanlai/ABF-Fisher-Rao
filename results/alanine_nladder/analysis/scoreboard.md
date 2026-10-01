# Low-T alanine scoreboard (results/alanine_nladder; reference T = 300.0 K, kT = 2.494 kJ/mol; raw read-out floor 0.559 kJ/mol)

Generated 2026-10-01T05:13:39Z. Reference acceptance: "see acceptance.json"

## Predictor on `N256_abf` (N = 256): **COUNT-STARVED, ERROR-CONVERGED (t_F < t_est: the ZIF-8 pattern, NOT allocation-limited)**

- t_est (95 % of mask cells at count >= c_min, and C7ax visited): median 37 ps (max 38); coverage-only t_cov 6 ps; t_F (error within 2x final): 12 ps; rule: FAST if t_est < 30 ps; favourable iff t_F >= t_est (amendment A1, docs/ALANINE_NLADDER.md)
- C7ax first hit median 4.87 ps, censored 0/16; ABF raw error @1/5/20/50/100 ps: 1.659 / 5.730 / 0.725 / 0.591 / 0.554; I_F W1 77.21; KL(p||U) @1/100: 3.95/3.62; KL(p||support) @1/100: 2.11/3.62

## Predictor on `N64_abf` (N = 64): **COUNT-STARVED, ERROR-CONVERGED (t_F < t_est: the ZIF-8 pattern, NOT allocation-limited)**

- t_est (95 % of mask cells at count >= c_min, and C7ax visited): median inf ps (max inf); coverage-only t_cov 9 ps; t_F (error within 2x final): 15 ps; rule: FAST if t_est < 30 ps; favourable iff t_F >= t_est (amendment A1, docs/ALANINE_NLADDER.md)
- C7ax first hit median 7.09 ps, censored 0/16; ABF raw error @1/5/20/50/100 ps: 1.455 / 3.015 / 0.755 / 0.641 / 0.571; I_F W1 90.59; KL(p||U) @1/100: 5.12/4.99; KL(p||support) @1/100: 2.90/4.99

## FR vs matched ABF (own window, raw read-out; frozen rules)

| arm | method | rate | dI_F own | dI_F W1 | final | S(e0/2) | S(e0/4) | S(e0/8) | ratio min (t) | ratio final | events/opp | ESS min | KL_U final arm/abf | KL_support final arm/abf | C7ax final arm/abf | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N256_u02_t0 | fr_uniform | 0.02 |  -0.75 % [ -1.42,  +0.09] 10/16 |  -0.68 % [ -1.65,  +0.09] 10/16 |  -0.59 % [ -2.62,  +0.39] 10/16 | 1.00 | - | - | 0.973 (6) | 0.994 | 0.4 | 0.90 | 3.623/3.620 | 3.623/3.620 | 0.039/0.043 | NEUTRAL |
| N256_u15_t0 | fr_uniform | 0.15 |  +0.47 % [ -0.66,  +1.80] 5/16 |  +0.78 % [ -0.83,  +2.16] 5/16 |  -0.15 % [ -2.29,  +1.85] 8/16 | 0.94 | - | - | 0.978 (2) | 0.998 | 3.0 | 0.62 | 3.639/3.620 | 3.639/3.620 | 0.043/0.043 | NEUTRAL_FLOOR_VIOLATION |
| N64_u02_t0 | fr_uniform | 0.02 |  +5.61 % [ -1.80,  +7.59] 6/16 |  +6.08 % [ -1.76,  +8.37] 6/16 |  +0.21 % [ -2.42,  +2.44] 8/16 | 0.95 | - | - | 0.985 (77) | 1.002 | 0.1 | 0.86 | 4.991/4.991 | 4.991/4.991 | 0.047/0.047 | NEUTRAL_FLOOR_VIOLATION |
| N64_u15_t0 | fr_uniform | 0.15 |  +2.15 % [ -2.51,  +5.32] 7/16 |  +2.54 % [ -2.42,  +5.90] 6/16 |  -0.07 % [ -2.53,  +4.89] 8/16 | 0.94 | - | - | 0.959 (3) | 0.999 | 0.5 | 0.67 | 4.991/4.991 | 4.991/4.991 | 0.047/0.047 | NEUTRAL_FLOOR_VIOLATION |
