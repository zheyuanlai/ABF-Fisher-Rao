# Equal-budget ladder analysis: gateway_family -- mechanism cell matched_free_energy/alpha0: I: alpha = 0 (purely energetic barrier in x; deterministic force)

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/mechanism/cells/matched_free_energy/alpha0.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/mechanism/matched_free_energy`; generated 2026-10-10T14:19:18Z; eqb_metrics/2.
B = 3,276,800,000 walker-steps per arm per seed; h = 2.5e-05; seeds 32; thresholds e_F [0.002, 0.0055, 0.02], e_F' [0.012, 0.035, 0.1], TV_half 0.1, e_Fp_stat [0.004, 0.0136, 0.0946].
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/mechanism/references/matched_free_energy_alpha0_reference.npz, sha256 b586ad0c62c3ac3107d57a10dc9291370133f93a549fbdfb0c886f7e0f3ea8ee, n_eval_nodes 151, em_floor_F_rms 1.0496374093956405e-09, em_bias_Fp_bin_rms 0.0, floor_definition errors of the scorer fed the exact bin-averaged reference mean force (Gamma_j = <F'_ref>_j on the Gauss-Legendre sub-grid, infinite counts); for e_Fp / e_Fp_em it is a HARD lower bound: e_Fp^2 = e_Fp_stat^2 + floor^2 for every Gamma (within-bin variation of F'_ref, steep at the gate); e_F at Gamma = <F'_ref> is ~0, units reduced (energy; force per unit x), scorer scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, identical for every variant and lambda); secondary = eqb_family.family_mean_force (per-dynamics EM-consistent), em_bias_definition em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j, F'_em = this dynamics' frozen-x EM-consistent mean force (NOT the e_F' floor), secondary_definition per-dynamics frozen-x Euler-Maruyama-consistent mean force at the cell's h and lambda: alpha family 4Hx(x^2-1) + ((1-alpha)/beta) w'/w + (alpha/beta)(w'/w) / (1 - lam w^(2 alpha) h/2) (the EM y-update y <- (1 - lam w^(2 alpha) h) y + sqrt(2 lam h/beta) z has stationary variance 1/(beta w^(2 alpha) (1 - lam w^(2 alpha) h/2))); shifted fibre: F*'(x) exactly (the frozen-x EM chain of y has mean m(x) exactly), secondary_equals_equal_budget_em None.
Zero-noise floors (scorer fed the exact bin-averaged reference): e_F 4.326e-16, e_Fp 0.03227 (HARD lower bound), e_Fp_stat 0 (HARD lower bound), e_F_em 1.05e-09, e_Fp_em 0.03227 (HARD lower bound), e_Fp_em_stat 0 (HARD lower bound).

## Frozen thresholds vs the zero-noise floors

| tau | eps | floor | hard bound | reachable | statistical budget sqrt(eps^2 - floor^2) | note |
|---|---|---|---|---|---|---|
| e_F_strict | 0.002 | 4.326e-16 | no | yes | -- |  |
| e_F_mid | 0.0055 | 4.326e-16 | no | yes | -- |  |
| e_F_loose | 0.02 | 4.326e-16 | no | yes | -- |  |
| e_Fp_strict | 0.012 | 0.03227 | yes | **NO (unreachable by construction)** | 0 | UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp itself is reported) |
| e_Fp_mid | 0.035 | 0.03227 | yes | yes | 0.01356 | reachable only if the statistical part (RMS of Gamma - bin-averaged reference) is <= 0.01356 |
| e_Fp_loose | 0.1 | 0.03227 | yes | yes | 0.09465 |  |
| e_F_em_strict | 0.002 | 1.05e-09 | no | yes | -- |  |
| e_F_em_mid | 0.0055 | 1.05e-09 | no | yes | -- |  |
| e_F_em_loose | 0.02 | 1.05e-09 | no | yes | -- |  |
| e_Fp_em_strict | 0.012 | 0.03227 | yes | **NO (unreachable by construction)** | 0 | UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp_em itself is reported) |
| e_Fp_em_mid | 0.035 | 0.03227 | yes | yes | 0.01356 | reachable only if the statistical part (RMS of Gamma - bin-averaged reference) is <= 0.01356 |
| e_Fp_em_loose | 0.1 | 0.03227 | yes | yes | 0.09465 |  |
| e_Fp_stat_strict | 0.004 | 0 | yes | yes | 0.004 |  |
| e_Fp_stat_mid | 0.0136 | 0 | yes | yes | 0.0136 |  |
| e_Fp_stat_loose | 0.0946 | 0 | yes | yes | 0.0946 |  |

## Status

| N | n_steps | T_N | ABF complete | FR complete | missing / running / invalid |
|---|---|---|---|---|---|
| 2048 | 1,600,000 | 40 | 32/32 | 32/32 | -- |
| 512 | 6,400,000 | 160 | 32/32 | 32/32 | -- |
| 128 | 25,600,000 | 640 | 32/32 | 32/32 | -- |

## Absolute values (median [IQR] over seeds)

| N | T_N | ABF Ibar_F | FR Ibar_F | ABF Ibar_F' | FR Ibar_F' | ABF e_F(1) | FR e_F(1) | ABF e_F'(1) | FR e_F'(1) |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 40 | 0.01288 [0.01162, 0.01381] | 0.009921 [0.009207, 0.01042] | 0.1043 [0.09746, 0.1084] | 0.08475 [0.07899, 0.08841] | 1.755e-05 [1.694e-05, 1.909e-05] | 7.46e-06 [6.735e-06, 8.151e-06] | 0.03227 [0.03227, 0.03227] | 0.03227 [0.03227, 0.03227] |
| 512 | 160 | 0.004002 [0.003499, 0.004441] | 0.002639 [0.002472, 0.002984] | 0.05283 [0.05124, 0.05674] | 0.04717 [0.04416, 0.04784] | 5.835e-06 [5.348e-06, 6.721e-06] | 3.602e-06 [2.69e-06, 4.414e-06] | 0.03227 [0.03227, 0.03227] | 0.03227 [0.03227, 0.03227] |
| 128 | 640 | 0.0006625 [0.0006222, 0.0007367] | 0.0006282 [0.0005706, 0.0006666] | 0.03613 [0.036, 0.03867] | 0.03599 [0.03597, 0.036] | 4.203e-06 [3.179e-06, 5.319e-06] | 2.769e-06 [2.264e-06, 3.067e-06] | 0.03227 [0.03227, 0.03227] | 0.03227 [0.03227, 0.03227] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 2048 | -24.0 % [-29.4 %, -15.2 %] (31/32) | -18.6 % [-21.9 %, -14.1 %] (30/32) | -57.7 % [-61.1 %, -54.7 %] (32/32) | -0.0 % [-0.0 %, -0.0 %] (32/32) | -73.4 % [-73.8 %, -73.1 %] (32/32) | -96.4 % [-96.6 %, -96.0 %] (32/32) | 2.64 deaths / walker |
| 512 | -31.7 % [-37.6 %, -20.2 %] (30/32) | -14.1 % [-17.5 %, -7.6 %] (31/32) | -36.6 % [-52.6 %, -27.6 %] (29/32) | -0.0 % [-0.0 %, -0.0 %] (22/32) | -76.4 % [-77.6 %, -75.5 %] (32/32) | -71.3 % [-77.0 %, -57.2 %] (32/32) | 18.5 deaths / walker |
| 128 | -7.2 % [-19.4 %, -3.2 %] (23/32) | -1.0 % [-6.3 %, -0.1 %] (25/32) | -30.1 % [-48.8 %, -10.8 %] (24/32) | -0.0 % [-0.0 %, +0.0 %] (19/32) | -71.4 % [-73.3 %, -70.0 %] (32/32) | -72.0 % [-75.2 %, -58.9 %] (32/32) | 151 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 2048 | e_F_strict | 0.1 | 0 | 0.075 | 0 | 30/1/1 | 2.98e-08 | -28.6 % [-33.3 %, -20.5 %] (n 32) |
| 2048 | e_F_mid | 0.1 | 0 | 0.075 | 0 | 30/1/1 | 2.98e-08 | -28.6 % [-33.3 %, -20.5 %] (n 32) |
| 2048 | e_F_loose | 0.0975 | 0 | 0.07 | 0 | 30/1/1 | 2.98e-08 | -29.0 % [-35.0 %, -19.9 %] (n 32) |
| 2048 | e_Fp_stat_strict | 0.105 | 0 | 0.075 | 0 | 32/0/0 | 4.66e-10 | -28.9 % [-33.3 %, -25.0 %] (n 32) |
| 2048 | e_Fp_stat_mid | 0.1 | 0 | 0.075 | 0 | 30/1/1 | 2.98e-08 | -27.9 % [-33.3 %, -20.5 %] (n 32) |
| 2048 | e_Fp_stat_loose | 0.1 | 0 | 0.075 | 0 | 30/1/1 | 2.98e-08 | -28.6 % [-33.3 %, -20.5 %] (n 32) |
| 2048 | TV_half | > 1 | 1 | 0.1725 | 0 | 32/0/0 | 4.66e-10 | -- [--, --] (n 0) |
| 2048 | e_F_em_strict (secondary) | 0.1 | 0 | 0.075 | 0 | 30/1/1 | 2.98e-08 | -28.6 % [-33.3 %, -20.5 %] (n 32) |
| 2048 | e_F_em_mid (secondary) | 0.1 | 0 | 0.075 | 0 | 30/1/1 | 2.98e-08 | -28.6 % [-33.3 %, -20.5 %] (n 32) |
| 2048 | e_F_em_loose (secondary) | 0.0975 | 0 | 0.07 | 0 | 30/1/1 | 2.98e-08 | -29.0 % [-35.0 %, -19.9 %] (n 32) |
| 2048 | e_Fp_mid (secondary) | 0.1 | 0 | 0.075 | 0 | 30/1/1 | 2.98e-08 | -27.9 % [-33.3 %, -20.5 %] (n 32) |
| 2048 | e_Fp_loose (secondary) | 0.1 | 0 | 0.075 | 0 | 30/1/1 | 2.98e-08 | -28.6 % [-33.3 %, -20.5 %] (n 32) |
| 2048 | e_Fp_em_mid (secondary) | 0.1 | 0 | 0.075 | 0 | 30/1/1 | 2.98e-08 | -27.9 % [-33.3 %, -20.5 %] (n 32) |
| 2048 | e_Fp_em_loose (secondary) | 0.1 | 0 | 0.075 | 0 | 30/1/1 | 2.98e-08 | -28.6 % [-33.3 %, -20.5 %] (n 32) |
| 512 | e_F_strict | 0.0325 | 0 | 0.025 | 0 | 25/0/7 | 5.96e-08 | -31.0 % [-37.5 %, -16.7 %] (n 32) |
| 512 | e_F_mid | 0.0325 | 0 | 0.025 | 0 | 25/0/7 | 5.96e-08 | -31.0 % [-37.5 %, -16.7 %] (n 32) |
| 512 | e_F_loose | 0.0325 | 0 | 0.025 | 0 | 25/0/7 | 5.96e-08 | -28.6 % [-37.5 %, -16.7 %] (n 32) |
| 512 | e_Fp_stat_strict | 0.035 | 0 | 0.025 | 0 | 30/0/2 | 1.86e-09 | -31.0 % [-37.5 %, -28.6 %] (n 32) |
| 512 | e_Fp_stat_mid | 0.0325 | 0 | 0.025 | 0 | 26/0/6 | 2.98e-08 | -31.0 % [-37.5 %, -16.7 %] (n 32) |
| 512 | e_Fp_stat_loose | 0.0325 | 0 | 0.025 | 0 | 25/0/7 | 5.96e-08 | -31.0 % [-37.5 %, -16.7 %] (n 32) |
| 512 | TV_half | 0.2925 | 0 | 0.05 | 0 | 32/0/0 | 4.66e-10 | -83.6 % [-84.5 %, -82.9 %] (n 32) |
| 512 | e_F_em_strict (secondary) | 0.0325 | 0 | 0.025 | 0 | 25/0/7 | 5.96e-08 | -31.0 % [-37.5 %, -16.7 %] (n 32) |
| 512 | e_F_em_mid (secondary) | 0.0325 | 0 | 0.025 | 0 | 25/0/7 | 5.96e-08 | -31.0 % [-37.5 %, -16.7 %] (n 32) |
| 512 | e_F_em_loose (secondary) | 0.0325 | 0 | 0.025 | 0 | 25/0/7 | 5.96e-08 | -28.6 % [-37.5 %, -16.7 %] (n 32) |
| 512 | e_Fp_mid (secondary) | 0.0325 | 0 | 0.025 | 0 | 26/0/6 | 2.98e-08 | -31.0 % [-37.5 %, -16.7 %] (n 32) |
| 512 | e_Fp_loose (secondary) | 0.0325 | 0 | 0.025 | 0 | 25/0/7 | 5.96e-08 | -31.0 % [-37.5 %, -16.7 %] (n 32) |
| 512 | e_Fp_em_mid (secondary) | 0.0325 | 0 | 0.025 | 0 | 26/0/6 | 2.98e-08 | -31.0 % [-37.5 %, -16.7 %] (n 32) |
| 512 | e_Fp_em_loose (secondary) | 0.0325 | 0 | 0.025 | 0 | 25/0/7 | 5.96e-08 | -31.0 % [-37.5 %, -16.7 %] (n 32) |
| 128 | e_F_strict | 0.01 | 0 | 0.01 | 0 | 20/1/11 | 2.1e-05 | -33.3 % [-35.4 %, +0.0 %] (n 32) |
| 128 | e_F_mid | 0.01 | 0 | 0.01 | 0 | 19/1/12 | 4.01e-05 | -33.3 % [-35.4 %, +0.0 %] (n 32) |
| 128 | e_F_loose | 0.01 | 0 | 0.01 | 0 | 17/1/14 | 0.000145 | -33.3 % [-37.5 %, +0.0 %] (n 32) |
| 128 | e_Fp_stat_strict | 0.015 | 0 | 0.01 | 0 | 21/1/10 | 1.1e-05 | -33.3 % [-33.3 %, -16.7 %] (n 32) |
| 128 | e_Fp_stat_mid | 0.01 | 0 | 0.01 | 0 | 19/1/12 | 4.01e-05 | -33.3 % [-33.3 %, +0.0 %] (n 32) |
| 128 | e_Fp_stat_loose | 0.01 | 0 | 0.01 | 0 | 20/1/11 | 2.1e-05 | -33.3 % [-35.4 %, +0.0 %] (n 32) |
| 128 | TV_half | 0.075 | 0 | 0.015 | 0 | 32/0/0 | 4.66e-10 | -80.0 % [-81.2 %, -78.6 %] (n 32) |
| 128 | e_F_em_strict (secondary) | 0.01 | 0 | 0.01 | 0 | 20/1/11 | 2.1e-05 | -33.3 % [-35.4 %, +0.0 %] (n 32) |
| 128 | e_F_em_mid (secondary) | 0.01 | 0 | 0.01 | 0 | 19/1/12 | 4.01e-05 | -33.3 % [-35.4 %, +0.0 %] (n 32) |
| 128 | e_F_em_loose (secondary) | 0.01 | 0 | 0.01 | 0 | 17/1/14 | 0.000145 | -33.3 % [-37.5 %, +0.0 %] (n 32) |
| 128 | e_Fp_mid (secondary) | 0.01 | 0 | 0.01 | 0 | 19/1/12 | 4.01e-05 | -33.3 % [-33.3 %, +0.0 %] (n 32) |
| 128 | e_Fp_loose (secondary) | 0.01 | 0 | 0.01 | 0 | 20/1/11 | 2.1e-05 | -33.3 % [-35.4 %, +0.0 %] (n 32) |
| 128 | e_Fp_em_mid (secondary) | 0.01 | 0 | 0.01 | 0 | 19/1/12 | 4.01e-05 | -33.3 % [-33.3 %, +0.0 %] (n 32) |
| 128 | e_Fp_em_loose (secondary) | 0.01 | 0 | 0.01 | 0 | 20/1/11 | 2.1e-05 | -33.3 % [-35.4 %, +0.0 %] (n 32) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_stat_strict | e_Fp_stat_mid | e_Fp_stat_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 2048 | abf | 0.1 [0.09, 0.1062] (0) | 0.1 [0.09, 0.1062] (0) | 0.0975 [0.08875, 0.105] (0) | 0.105 [0.1, 0.1113] (0) | 0.1 [0.09, 0.1062] (0) | 0.1 [0.09, 0.1062] (0) | > 1 [> 1, > 1] (1) |
| 2048 | fr | 0.075 [0.065, 0.08] (0) | 0.075 [0.065, 0.08] (0) | 0.07 [0.065, 0.075] (0) | 0.075 [0.07, 0.08] (0) | 0.075 [0.065, 0.08] (0) | 0.075 [0.065, 0.08] (0) | 0.1725 [0.17, 0.175] (0) |
| 512 | abf | 0.0325 [0.03, 0.04] (0) | 0.0325 [0.03, 0.04] (0) | 0.0325 [0.03, 0.035] (0) | 0.035 [0.03, 0.04] (0) | 0.0325 [0.03, 0.04] (0) | 0.0325 [0.03, 0.04] (0) | 0.2925 [0.2788, 0.3075] (0) |
| 512 | fr | 0.025 [0.02, 0.025] (0) | 0.025 [0.02, 0.025] (0) | 0.025 [0.02, 0.025] (0) | 0.025 [0.02375, 0.025] (0) | 0.025 [0.02, 0.025] (0) | 0.025 [0.02, 0.025] (0) | 0.05 [0.045, 0.05] (0) |
| 128 | abf | 0.01 [0.01, 0.015] (0) | 0.01 [0.01, 0.015] (0) | 0.01 [0.01, 0.015] (0) | 0.015 [0.01, 0.015] (0) | 0.01 [0.01, 0.015] (0) | 0.01 [0.01, 0.015] (0) | 0.075 [0.06875, 0.0825] (0) |
| 128 | fr | 0.01 [0.00625, 0.01] (0) | 0.01 [0.00625, 0.01] (0) | 0.01 [0.00625, 0.01] (0) | 0.01 [0.01, 0.01] (0) | 0.01 [0.00625, 0.01] (0) | 0.01 [0.00625, 0.01] (0) | 0.015 [0.015, 0.015] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 2048 | 370 | 1.74e-72 | 1.74e-72 | 1.000 | 0.0001 | no |
| 512 | 93 | 1.53e-19 | 1.53e-19 | 1.000 | 0.0001 | no |
| 128 | 24 | 5.53e-06 | 5.53e-06 | 0.999 | 0.0001 | no |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | abf | 0.1226 [0.1172, 0.1273] | > 1 [> 1, > 1] | 0.07794 [0.07099, 0.0824] | 0.0363 | 0.48 [0.465, 0.5] | 0 | > 1 [> 1, > 1] | 0.03678 [0.03249, 0.03955] | 2524 [2485, 2557] |
| 2048 | fr | 0.004454 [0.003866, 0.004939] | 0.1725 [0.17, 0.175] | 0.02767 [0.02627, 0.0344] | 0.0363 | 0.105 [0.1, 0.105] | 0 | 0.2125 [0.205, 0.215] | 0.03301 [0.02709, 0.03709] | 2739 [2697, 2763] |
| 512 | abf | 0.01602 [0.009626, 0.01896] | 0.2925 [0.2788, 0.3075] | 0.0714 [0.06169, 0.08512] | 0.0728 | 0.125 [0.1187, 0.13] | 0 | 0.275 [0.2638, 0.295] | 0.01064 [0.008911, 0.01331] | 2755 [2709, 2799] |
| 512 | fr | 0.00427 [0.003818, 0.00471] | 0.05 [0.045, 0.05] | 0.06391 [0.0555, 0.07275] | 0.0728 | 0.03 [0.025, 0.03] | 0 | 0.055 [0.055, 0.06] | 0.009536 [0.008264, 0.01036] | 2812 [2785, 2837] |
| 128 | abf | 0.01457 [0.01023, 0.01717] | 0.075 [0.06875, 0.0825] | 0.1536 [0.1337, 0.1684] | 0.1447 | 0.03062 [0.03, 0.035] | 0 | 0.07 [0.06187, 0.08] | 0.003682 [0.002997, 0.004335] | 2814 [2799, 2860] |
| 128 | fr | 0.004446 [0.004017, 0.004816] | 0.015 [0.015, 0.015] | 0.1267 [0.1083, 0.1354] | 0.1447 | 0.01 [0.01, 0.01] | 0 | 0.015 [0.015, 0.01672] | 0.003197 [0.002743, 0.003825] | 2817 [2802, 2837] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems; the gateway's capped candidates kd + kc are the last column. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win | candidates/N per opp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 0/32 | 163 | 0 | 5396 [5341, 5434] | 2.635 [2.608, 2.653] | 0.4304 [0.4265, 0.4322] | 0.0002635 [0.0002608, 0.0002653] | 0.00293 [0.002441, 0.00293] | 0.02734 [0.02193, 0.03588] | 0.3237 [0.2935, 0.3484] | 407 [399.2, 415.5] | 0.01636 [0.01355, 0.02014] | 0.0003054 [0.0003028, 0.0003089] |
| 512 | 0/32 | 40 | 0 | 9470 [9403, 9546] | 18.5 [18.37, 18.64] | 0.2207 [0.2193, 0.2229] | 0.0004624 [0.0004591, 0.0004661] | 0.007812 [0.005859, 0.007812] | 0.02005 [0.0166, 0.0232] | 0.3338 [0.3045, 0.3685] | 22 [19.75, 23] | 0.03906 [0.0332, 0.04541] | 0.0004924 [0.0004884, 0.0004957] |
| 128 | 0/32 | 10 | 0 | 1.932e+04 [1.918e+04, 1.939e+04] | 150.9 [149.9, 151.5] | 0.1168 [0.1159, 0.1174] | 0.0009431 [0.0009368, 0.0009468] | 0.02344 [0.02344, 0.02539] | 0.007812 [0.007812, 0.008564] | 0.4197 [0.4045, 0.4414] | 1 [1, 2] | 0.07422 [0.07031, 0.08008] | 0.0009777 [0.000972, 0.0009825] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 2048 | e_F | +99.8 % | 0.085 | -6.8 % | 0.045 | +57.5 % | 32 |
| 2048 | e_Fp | +95.8 % | 0.075 | -0.4 % | 0.01 | +0.0 % | 32 |
| 2048 | e_F_em | +99.8 % | 0.085 | -6.8 % | 0.045 | +57.5 % | 32 |
| 2048 | TV_half | +97.8 % | 0.365 | -0.0 % | 0.005 | +96.4 % | 32 |
| 512 | e_F | +99.9 % | 0.025 | -0.9 % | 0.015 | +38.3 % | 32 |
| 512 | e_Fp | +95.6 % | 0.025 | +0.0 % | 0.85 | +0.0 % | 32 |
| 512 | e_F_em | +99.9 % | 0.025 | -0.9 % | 0.015 | +38.3 % | 32 |
| 512 | TV_half | +95.8 % | 0.095 | +0.4 % | 0.005 | +73.3 % | 32 |
| 128 | e_F | +81.8 % | 0.01 | -1.7 % | 0.005 | +34.1 % | 32 |
| 128 | e_Fp | +0.9 % | 0.01 | -0.0 % | 0.3 | +0.0 % | 32 |
| 128 | e_F_em | +81.8 % | 0.01 | -1.7 % | 0.005 | +34.1 % | 32 |
| 128 | TV_half | +91.3 % | 0.025 | +12.7 % | 0.005 | +69.5 % | 32 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 128 | 0.0006625 | [0.0006411, 0.0006893] | 128: 1.00 | 128 | 0.0006282 | [0.000595, 0.0006538] | 128: 1.00 | -3.434e-05 [-7.897e-05, 2.506e-06] | -5.2 % [-11.8 %, +0.4 %] |
| final_e_F | 128 | 4.203e-06 | [3.746e-06, 4.905e-06] | 128: 1.00, 512: 0.00 | 128 | 2.769e-06 | [2.424e-06, 2.907e-06] | 128: 1.00, 512: 0.00 | -1.434e-06 [-2.325e-06, -8.822e-07] | -34.1 % [-48.6 %, -23.4 %] |
| tau_bgrid_e_F_mid_u | 128 | 0.01 | [0.01, 0.015] | 128: 1.00 | 128 | 0.01 | [0.01, 0.01] | 128: 1.00 | 0 [-0.005, 0] | +0.0 % [-33.3 %, +0.0 %] |
| Ibar_Fp | 128 | 0.03613 | [0.03601, 0.03814] | 128: 1.00 | 128 | 0.03599 | [0.03598, 0.03599] | 128: 1.00 | -0.0001436 [-0.002148, -1.772e-05] | -0.4 % [-5.6 %, -0.0 %] |
| final_e_Fp | 128 | 0.03227 | [0.03227, 0.03227] | 128: 1.00, 512: 0.00 | 128 | 0.03227 | [0.03227, 0.03227] | 128: 0.84, 512: 0.16 | -2.039e-09 [-6.604e-09, 4.134e-10] | -0.0 % [-0.0 %, +0.0 %] |
| tau_bgrid_TV_half_u | 128 | 0.075 | [0.07, 0.08] | 128: 1.00 | 128 | 0.015 | [0.015, 0.015] | 128: 1.00 | -0.06 [-0.065, -0.055] | -80.0 % [-81.2 %, -78.6 %] |

## Secondary (EM-consistent reference; *_Fp_stat = floor-free companion of e_F' (descriptive))

| N | Ibar_F_em | Ibar_Fp_em | final_e_F_em | final_e_Fp_em | Ibar_Fp_stat | final_e_Fp_stat |
|---|---|---|---|---|---|---|
| 2048 | ABF 0.01288, FR 0.009921; -24.0 % [-29.4 %, -15.2 %] (31/32) | ABF 0.1043, FR 0.08475; -18.6 % [-21.9 %, -14.1 %] (30/32) | ABF 1.755e-05, FR 7.46e-06; -57.7 % [-61.1 %, -54.7 %] (32/32) | ABF 0.03227, FR 0.03227; -0.0 % [-0.0 %, -0.0 %] (32/32) | ABF 0.07532, FR 0.05478; -27.0 % [-32.7 %, -21.0 %] (30/32) | ABF 0.0001045, FR 6.488e-05; -37.2 % [-41.3 %, -35.2 %] (32/32) |
| 512 | ABF 0.004002, FR 0.002639; -31.7 % [-37.6 %, -20.2 %] (30/32) | ABF 0.05283, FR 0.04717; -14.1 % [-17.5 %, -7.6 %] (31/32) | ABF 5.835e-06, FR 3.603e-06; -36.6 % [-52.6 %, -27.6 %] (29/32) | ABF 0.03227, FR 0.03227; -0.0 % [-0.0 %, -0.0 %] (22/32) | ABF 0.0216, FR 0.01563; -36.9 % [-39.8 %, -21.7 %] (31/32) | ABF 5.823e-05, FR 5.38e-05; -8.6 % [-13.4 %, -1.3 %] (22/32) |
| 128 | ABF 0.0006625, FR 0.0006282; -7.2 % [-19.4 %, -3.2 %] (23/32) | ABF 0.03613, FR 0.03599; -1.0 % [-6.3 %, -0.1 %] (25/32) | ABF 4.203e-06, FR 2.769e-06; -30.1 % [-48.8 %, -10.8 %] (24/32) | ABF 0.03227, FR 0.03227; -0.0 % [-0.0 %, +0.0 %] (19/32) | ABF 0.004167, FR 0.003977; -9.5 % [-38.9 %, -1.7 %] (27/32) | ABF 5.326e-05, FR 5.201e-05; -1.7 % [-10.4 %, +3.0 %] (19/32) |

## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)

exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial).

| N | E_N[TV] |
|---|---|
| 128 | 0.14473 |
| 512 | 0.07281 |
| 2048 | 0.03635 |

## Cost (median per run)

| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |
|---|---|---|---|---|---|
| 2048 | abf | 296.7 | 0.09053 | 159.2 | 32/32 |
| 2048 | fr | 302.9 | 0.09243 | 160.2 | 32/32 |
| 512 | abf | 301.9 | 0.09214 | 159.2 | 32/32 |
| 512 | fr | 307.8 | 0.09394 | 159.2 | 32/32 |
| 128 | abf | 307.2 | 0.09376 | 159.2 | 32/32 |
| 128 | fr | 313.5 | 0.09569 | 159.2 | 32/32 |

## Warnings

* tau_e_Fp_strict: UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp itself is reported)
* tau_e_Fp_em_strict: UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp_em itself is reported)
