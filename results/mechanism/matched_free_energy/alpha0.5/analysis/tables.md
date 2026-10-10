# Equal-budget ladder analysis: gateway_family -- mechanism cell matched_free_energy/alpha0.5: I: alpha = 0.5 (half the log-omega entropy moved into the energy; Var(f|x) / 4)

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/mechanism/cells/matched_free_energy/alpha0.5.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/mechanism/matched_free_energy`; generated 2026-10-10T14:19:19Z; eqb_metrics/2.
B = 3,276,800,000 walker-steps per arm per seed; h = 2.5e-05; seeds 32; thresholds e_F [0.002, 0.0055, 0.02], e_F' [0.012, 0.035, 0.1], TV_half 0.1, e_Fp_stat [0.004, 0.0136, 0.0946].
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/mechanism/references/matched_free_energy_alpha0.5_reference.npz, sha256 194ea8553d0758120b69d3afea81aa6b3e50fecc5877edf1dc6b78617641e3b9, n_eval_nodes 151, em_floor_F_rms 2.7558266101825967e-06, em_bias_Fp_bin_rms 2.0765655167769617e-05, floor_definition errors of the scorer fed the exact bin-averaged reference mean force (Gamma_j = <F'_ref>_j on the Gauss-Legendre sub-grid, infinite counts); for e_Fp / e_Fp_em it is a HARD lower bound: e_Fp^2 = e_Fp_stat^2 + floor^2 for every Gamma (within-bin variation of F'_ref, steep at the gate); e_F at Gamma = <F'_ref> is ~0, units reduced (energy; force per unit x), scorer scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, identical for every variant and lambda); secondary = eqb_family.family_mean_force (per-dynamics EM-consistent), em_bias_definition em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j, F'_em = this dynamics' frozen-x EM-consistent mean force (NOT the e_F' floor), secondary_definition per-dynamics frozen-x Euler-Maruyama-consistent mean force at the cell's h and lambda: alpha family 4Hx(x^2-1) + ((1-alpha)/beta) w'/w + (alpha/beta)(w'/w) / (1 - lam w^(2 alpha) h/2) (the EM y-update y <- (1 - lam w^(2 alpha) h) y + sqrt(2 lam h/beta) z has stationary variance 1/(beta w^(2 alpha) (1 - lam w^(2 alpha) h/2))); shifted fibre: F*'(x) exactly (the frozen-x EM chain of y has mean m(x) exactly), secondary_equals_equal_budget_em None.
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
| 2048 | 40 | 0.01888 [0.01778, 0.0199] | 0.01355 [0.01235, 0.01445] | 0.1065 [0.1003, 0.1095] | 0.08861 [0.08417, 0.09238] | 0.003531 [0.003073, 0.004029] | 0.001303 [0.0007652, 0.001626] | 0.03301 [0.03285, 0.03318] | 0.03239 [0.03233, 0.03248] |
| 512 | 160 | 0.006423 [0.006021, 0.007155] | 0.003899 [0.003543, 0.004472] | 0.05276 [0.05015, 0.05608] | 0.04658 [0.04469, 0.04799] | 0.0008737 [0.0007287, 0.001279] | 0.0004023 [0.0002845, 0.0006191] | 0.03237 [0.03233, 0.03241] | 0.03231 [0.03229, 0.03233] |
| 128 | 640 | 0.0019 [0.001548, 0.002133] | 0.001478 [0.001093, 0.001725] | 0.03656 [0.0364, 0.03924] | 0.03622 [0.03615, 0.03634] | 0.0003536 [0.0001931, 0.0005499] | 0.0004013 [0.0002341, 0.0007006] | 0.03231 [0.03229, 0.03232] | 0.03231 [0.0323, 0.03233] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 2048 | -26.2 % [-34.8 %, -21.7 %] (32/32) | -16.5 % [-19.6 %, -12.2 %] (30/32) | -63.8 % [-71.5 %, -56.6 %] (32/32) | -1.9 % [-2.2 %, -1.4 %] (32/32) | -72.2 % [-72.8 %, -72.0 %] (32/32) | -95.9 % [-96.5 %, -95.6 %] (32/32) | 2.6 deaths / walker |
| 512 | -36.3 % [-43.2 %, -28.4 %] (32/32) | -14.3 % [-17.6 %, -9.4 %] (30/32) | -55.9 % [-71.9 %, -36.1 %] (27/32) | -0.2 % [-0.3 %, -0.1 %] (25/32) | -74.0 % [-75.4 %, -72.6 %] (32/32) | -70.0 % [-75.4 %, -62.0 %] (32/32) | 18.5 deaths / walker |
| 128 | -21.4 % [-37.7 %, -15.2 %] (26/32) | -1.4 % [-5.7 %, -0.6 %] (27/32) | +4.0 % [-17.7 %, +142.1 %] (15/32) | +0.0 % [-0.0 %, +0.1 %] (13/32) | -72.6 % [-74.0 %, -70.6 %] (32/32) | -64.4 % [-74.2 %, -46.2 %] (32/32) | 150 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 2048 | e_F_strict | > 1 | 1 | 0.625 | 0.125 | 28/0/4 | 7.45e-09 | -- [--, --] (n 0) |
| 2048 | e_F_mid | 0.4925 | 0 | 0.175 | 0 | 32/0/0 | 4.66e-10 | -65.8 % [-69.2 %, -55.7 %] (n 32) |
| 2048 | e_F_loose | 0.0975 | 0 | 0.0725 | 0 | 31/1/0 | 1.54e-08 | -22.2 % [-27.3 %, -17.9 %] (n 32) |
| 2048 | e_Fp_stat_strict | > 1 | 1 | 0.65 | 0.1562 | 27/0/5 | 1.49e-08 | -- [--, --] (n 0) |
| 2048 | e_Fp_stat_mid | 0.3475 | 0 | 0.15 | 0 | 32/0/0 | 4.66e-10 | -56.2 % [-62.6 %, -53.2 %] (n 32) |
| 2048 | e_Fp_stat_loose | 0.1 | 0 | 0.075 | 0 | 29/3/0 | 2.56e-06 | -23.0 % [-27.3 %, -16.7 %] (n 32) |
| 2048 | TV_half | > 1 | 0.7188 | 0.17 | 0 | 32/0/0 | 4.66e-10 | -82.5 % [-82.9 %, -82.4 %] (n 9) |
| 2048 | e_F_em_strict (secondary) | > 1 | 1 | 0.625 | 0.125 | 28/0/4 | 7.45e-09 | -- [--, --] (n 0) |
| 2048 | e_F_em_mid (secondary) | 0.4925 | 0 | 0.175 | 0 | 32/0/0 | 4.66e-10 | -65.8 % [-69.2 %, -55.7 %] (n 32) |
| 2048 | e_F_em_loose (secondary) | 0.0975 | 0 | 0.0725 | 0 | 31/1/0 | 1.54e-08 | -22.2 % [-27.3 %, -17.9 %] (n 32) |
| 2048 | e_Fp_mid (secondary) | 0.3475 | 0 | 0.1525 | 0 | 32/0/0 | 4.66e-10 | -57.5 % [-62.6 %, -52.5 %] (n 32) |
| 2048 | e_Fp_loose (secondary) | 0.1 | 0 | 0.075 | 0 | 29/3/0 | 2.56e-06 | -23.0 % [-27.3 %, -16.7 %] (n 32) |
| 2048 | e_Fp_em_mid (secondary) | 0.3475 | 0 | 0.1525 | 0 | 32/0/0 | 4.66e-10 | -57.5 % [-62.6 %, -52.5 %] (n 32) |
| 2048 | e_Fp_em_loose (secondary) | 0.1 | 0 | 0.075 | 0 | 29/3/0 | 2.56e-06 | -23.0 % [-27.3 %, -16.7 %] (n 32) |
| 512 | e_F_strict | 0.5025 | 0 | 0.1475 | 0 | 30/2/0 | 2.46e-07 | -70.4 % [-80.5 %, -53.6 %] (n 32) |
| 512 | e_F_mid | 0.1375 | 0 | 0.05 | 0 | 29/2/1 | 4.63e-07 | -63.7 % [-67.8 %, -47.0 %] (n 32) |
| 512 | e_F_loose | 0.03 | 0 | 0.025 | 0 | 28/0/4 | 7.45e-09 | -28.6 % [-33.3 %, -16.7 %] (n 32) |
| 512 | e_Fp_stat_strict | 0.5525 | 0.03125 | 0.24 | 0 | 28/3/1 | 4.65e-06 | -48.5 % [-60.0 %, -40.6 %] (n 31) |
| 512 | e_Fp_stat_mid | 0.105 | 0 | 0.05 | 0 | 29/3/0 | 2.56e-06 | -50.4 % [-62.8 %, -28.3 %] (n 32) |
| 512 | e_Fp_stat_loose | 0.03 | 0 | 0.025 | 0 | 27/2/3 | 1.62e-06 | -28.6 % [-33.3 %, -16.7 %] (n 32) |
| 512 | TV_half | 0.26 | 0 | 0.045 | 0 | 32/0/0 | 4.66e-10 | -82.0 % [-82.7 %, -80.8 %] (n 32) |
| 512 | e_F_em_strict (secondary) | 0.5025 | 0 | 0.1475 | 0 | 30/2/0 | 2.46e-07 | -70.4 % [-80.5 %, -53.6 %] (n 32) |
| 512 | e_F_em_mid (secondary) | 0.1375 | 0 | 0.05 | 0 | 29/2/1 | 4.63e-07 | -63.7 % [-67.8 %, -47.0 %] (n 32) |
| 512 | e_F_em_loose (secondary) | 0.03 | 0 | 0.025 | 0 | 28/0/4 | 7.45e-09 | -28.6 % [-33.3 %, -16.7 %] (n 32) |
| 512 | e_Fp_mid (secondary) | 0.105 | 0 | 0.05 | 0 | 29/3/0 | 2.56e-06 | -50.4 % [-62.8 %, -28.3 %] (n 32) |
| 512 | e_Fp_loose (secondary) | 0.03 | 0 | 0.025 | 0 | 27/2/3 | 1.62e-06 | -28.6 % [-33.3 %, -16.7 %] (n 32) |
| 512 | e_Fp_em_mid (secondary) | 0.105 | 0 | 0.05 | 0 | 29/3/0 | 2.56e-06 | -50.4 % [-62.8 %, -28.3 %] (n 32) |
| 512 | e_Fp_em_loose (secondary) | 0.03 | 0 | 0.025 | 0 | 27/2/3 | 1.62e-06 | -28.6 % [-33.3 %, -16.7 %] (n 32) |
| 128 | e_F_strict | 0.1481 | 0 | 0.0875 | 0 | 23/9/0 | 0.0201 | -55.7 % [-64.5 %, -11.1 %] (n 32) |
| 128 | e_F_mid | 0.045 | 0 | 0.0175 | 0 | 28/3/1 | 4.65e-06 | -53.8 % [-66.7 %, -40.0 %] (n 32) |
| 128 | e_F_loose | 0.01 | 0 | 0.01 | 0 | 18/1/13 | 7.63e-05 | -33.3 % [-37.5 %, +0.0 %] (n 32) |
| 128 | e_Fp_stat_strict | 0.275 | 0 | 0.235 | 0 | 22/10/0 | 0.0501 | -12.4 % [-33.0 %, -2.3 %] (n 32) |
| 128 | e_Fp_stat_mid | 0.04 | 0 | 0.025 | 0 | 25/5/2 | 0.000325 | -47.7 % [-55.6 %, -16.7 %] (n 32) |
| 128 | e_Fp_stat_loose | 0.01 | 0 | 0.01 | 0 | 19/1/12 | 4.01e-05 | -33.3 % [-37.5 %, +0.0 %] (n 32) |
| 128 | TV_half | 0.0675 | 0 | 0.015 | 0 | 32/0/0 | 4.66e-10 | -78.6 % [-78.6 %, -76.0 %] (n 32) |
| 128 | e_F_em_strict (secondary) | 0.1481 | 0 | 0.0875 | 0 | 23/9/0 | 0.0201 | -55.7 % [-64.5 %, -11.1 %] (n 32) |
| 128 | e_F_em_mid (secondary) | 0.045 | 0 | 0.0175 | 0 | 28/3/1 | 4.65e-06 | -53.8 % [-66.7 %, -40.0 %] (n 32) |
| 128 | e_F_em_loose (secondary) | 0.01 | 0 | 0.01 | 0 | 18/1/13 | 7.63e-05 | -33.3 % [-37.5 %, +0.0 %] (n 32) |
| 128 | e_Fp_mid (secondary) | 0.04 | 0 | 0.025 | 0 | 25/5/2 | 0.000325 | -47.7 % [-55.6 %, -16.7 %] (n 32) |
| 128 | e_Fp_loose (secondary) | 0.01 | 0 | 0.01 | 0 | 19/1/12 | 4.01e-05 | -33.3 % [-37.5 %, +0.0 %] (n 32) |
| 128 | e_Fp_em_mid (secondary) | 0.04 | 0 | 0.025 | 0 | 25/5/2 | 0.000325 | -47.7 % [-55.6 %, -16.7 %] (n 32) |
| 128 | e_Fp_em_loose (secondary) | 0.01 | 0 | 0.01 | 0 | 19/1/12 | 4.01e-05 | -33.3 % [-37.5 %, +0.0 %] (n 32) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_stat_strict | e_Fp_stat_mid | e_Fp_stat_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 2048 | abf | > 1 [> 1, > 1] (1) | 0.4925 [0.4237, 0.5762] (0) | 0.0975 [0.09, 0.105] (0) | > 1 [> 1, > 1] (1) | 0.3475 [0.315, 0.39] (0) | 0.1 [0.09, 0.105] (0) | > 1 [0.9988, > 1] (0.7188) |
| 2048 | fr | 0.625 [0.4163, 0.8362] (0.125) | 0.175 [0.1537, 0.2062] (0) | 0.0725 [0.07, 0.08] (0) | 0.65 [0.495, 0.9225] (0.1562) | 0.15 [0.1237, 0.165] (0) | 0.075 [0.07, 0.08] (0) | 0.17 [0.17, 0.1762] (0) |
| 512 | abf | 0.5025 [0.3638, 0.57] (0) | 0.1375 [0.0975, 0.1862] (0) | 0.03 [0.03, 0.035] (0) | 0.5525 [0.4363, 0.66] (0.03125) | 0.105 [0.075, 0.1313] (0) | 0.03 [0.03, 0.035] (0) | 0.26 [0.25, 0.28] (0) |
| 512 | fr | 0.1475 [0.085, 0.2362] (0) | 0.05 [0.03, 0.07625] (0) | 0.025 [0.02, 0.025] (0) | 0.24 [0.1737, 0.365] (0) | 0.05 [0.035, 0.06438] (0) | 0.025 [0.02, 0.025] (0) | 0.045 [0.045, 0.05] (0) |
| 128 | abf | 0.1481 [0.09125, 0.2388] (0) | 0.045 [0.03406, 0.06] (0) | 0.01 [0.01, 0.015] (0) | 0.275 [0.2188, 0.3762] (0) | 0.04 [0.03094, 0.055] (0) | 0.01 [0.01, 0.015] (0) | 0.0675 [0.06, 0.07] (0) |
| 128 | fr | 0.0875 [0.03875, 0.1553] (0) | 0.0175 [0.01, 0.025] (0) | 0.01 [0.00625, 0.01] (0) | 0.235 [0.1737, 0.2825] (0) | 0.025 [0.01547, 0.03031] (0) | 0.01 [0.00625, 0.01] (0) | 0.015 [0.015, 0.015] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 2048 | 370 | 1.74e-72 | 1.74e-72 | 1.000 | 0.0001 | no |
| 512 | 93 | 1.53e-19 | 1.53e-19 | 1.000 | 0.0001 | no |
| 128 | 24 | 5.53e-06 | 5.53e-06 | 0.999 | 0.0001 | no |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | abf | 0.1061 [0.09928, 0.111] | > 1 [0.9988, > 1] | 0.06104 [0.05652, 0.06482] | 0.0363 | 0.4425 [0.4237, 0.4563] | 0 | 1 [0.9788, > 1] | 0.0352 [0.03056, 0.03945] | 2251 [2230, 2291] |
| 2048 | fr | 0.004105 [0.003718, 0.004703] | 0.17 [0.17, 0.1762] | 0.02789 [0.02577, 0.03185] | 0.0363 | 0.105 [0.1, 0.105] | 0 | 0.21 [0.205, 0.2162] | 0.03344 [0.02887, 0.03503] | 2404 [2375, 2421] |
| 512 | abf | 0.01488 [0.009383, 0.02113] | 0.26 [0.25, 0.28] | 0.07335 [0.06337, 0.07547] | 0.0728 | 0.115 [0.11, 0.12] | 0 | 0.25 [0.2387, 0.2612] | 0.01046 [0.009097, 0.01193] | 2372 [2345, 2414] |
| 512 | fr | 0.004594 [0.004278, 0.005309] | 0.045 [0.045, 0.05] | 0.06098 [0.05143, 0.06532] | 0.0728 | 0.03 [0.02875, 0.03] | 0 | 0.055 [0.055, 0.05625] | 0.008953 [0.00836, 0.01028] | 2404 [2370, 2439] |
| 128 | abf | 0.01371 [0.008923, 0.01884] | 0.0675 [0.06, 0.07] | 0.1493 [0.1276, 0.166] | 0.1447 | 0.03 [0.025, 0.035] | 0 | 0.06125 [0.05875, 0.07] | 0.003488 [0.002933, 0.004113] | 2420 [2392, 2468] |
| 128 | fr | 0.004822 [0.004428, 0.005729] | 0.015 [0.015, 0.015] | 0.1181 [0.1079, 0.1369] | 0.1447 | 0.01 [0.01, 0.01] | 0 | 0.015 [0.015, 0.015] | 0.003144 [0.002523, 0.003413] | 2422 [2394, 2442] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems; the gateway's capped candidates kd + kc are the last column. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win | candidates/N per opp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 0/32 | 163 | 0 | 5335 [5304, 5393] | 2.605 [2.59, 2.633] | 0.4283 [0.424, 0.4329] | 0.0002605 [0.000259, 0.0002633] | 0.00293 [0.002441, 0.00293] | 0.03163 [0.02831, 0.0416] | 0.3494 [0.3144, 0.3684] | 417.5 [407, 424] | 0.01514 [0.01306, 0.02014] | 0.0003023 [0.0002996, 0.0003059] |
| 512 | 0/32 | 40 | 0 | 9462 [9395, 9526] | 18.48 [18.35, 18.61] | 0.221 [0.2191, 0.2221] | 0.000462 [0.0004588, 0.0004651] | 0.007812 [0.005859, 0.007812] | 0.02184 [0.01579, 0.02442] | 0.3566 [0.3205, 0.3913] | 21 [19.75, 23] | 0.03906 [0.0293, 0.04736] | 0.0004912 [0.0004869, 0.0004933] |
| 128 | 0/32 | 10 | 0 | 1.924e+04 [1.915e+04, 1.931e+04] | 150.3 [149.6, 150.9] | 0.1164 [0.1159, 0.1169] | 0.0009396 [0.0009353, 0.0009429] | 0.02344 [0.02344, 0.02344] | 0.007812 [0.007812, 0.007812] | 0.4142 [0.3827, 0.4354] | 1 [1, 1] | 0.07031 [0.0625, 0.08594] | 0.0009741 [0.0009695, 0.0009772] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 2048 | e_F | +84.8 % | 0.075 | -2.5 % | 0.055 | +63.1 % | 32 |
| 2048 | e_Fp | +94.1 % | 0.08 | -0.1 % | 0.025 | +1.9 % | 32 |
| 2048 | e_F_em | +84.8 % | 0.075 | -2.5 % | 0.055 | +63.1 % | 32 |
| 2048 | TV_half | +97.6 % | 0.325 | -0.0 % | 0.005 | +96.1 % | 32 |
| 512 | e_F | +86.5 % | 0.025 | -2.8 % | 0.005 | +54.0 % | 32 |
| 512 | e_Fp | +94.2 % | 0.025 | +0.1 % | 0.005 | +0.2 % | 32 |
| 512 | e_F_em | +86.5 % | 0.025 | -2.8 % | 0.005 | +54.0 % | 32 |
| 512 | TV_half | +95.1 % | 0.095 | +0.4 % | 0.005 | +69.1 % | 32 |
| 128 | e_F | +71.2 % | 0.05 | -47.4 % | 0.895 | -13.5 % | 32 |
| 128 | e_Fp | +21.6 % | 0.01 | -0.1 % | 0.53 | -0.0 % | 32 |
| 128 | e_F_em | +71.2 % | 0.05 | -46.2 % | 0.895 | -13.2 % | 32 |
| 128 | TV_half | +90.7 % | 0.025 | +15.1 % | 0.005 | +64.8 % | 32 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 128 | 0.0019 | [0.00171, 0.002052] | 128: 1.00 | 128 | 0.001478 | [0.001177, 0.001709] | 128: 1.00 | -0.0004222 [-0.0007347, -0.0001332] | -22.2 % [-37.9 %, -7.7 %] |
| final_e_F | 128 | 0.0003536 | [0.0002469, 0.0005135] | 128: 1.00 | 128 | 0.0004013 | [0.0002738, 0.0005026] | 128: 0.50, 512: 0.50 | 4.762e-05 [-0.000193, 0.0001934] | +13.5 % [-39.2 %, +70.0 %] |
| tau_bgrid_e_F_mid_u | 128 | 0.045 | [0.035, 0.055] | 128: 1.00 | 128 | 0.0175 | [0.015, 0.0225] | 128: 1.00 | -0.0275 [-0.04, -0.0175] | -61.1 % [-72.7 %, -46.7 %] |
| Ibar_Fp | 128 | 0.03656 | [0.03643, 0.03776] | 128: 1.00 | 128 | 0.03622 | [0.03617, 0.03633] | 128: 1.00 | -0.0003415 [-0.001544, -0.0001426] | -0.9 % [-4.1 %, -0.4 %] |
| final_e_Fp | 128 | 0.03231 | [0.0323, 0.03232] | 128: 1.00 | 512 | 0.03231 | [0.0323, 0.03231] | 512: 0.90, 128: 0.10 | 1.329e-06 [-1.454e-05, 1.198e-05] | +0.0 % [-0.0 %, +0.0 %] |
| tau_bgrid_TV_half_u | 128 | 0.0675 | [0.06, 0.07] | 128: 1.00 | 128 | 0.015 | [0.015, 0.015] | 128: 1.00 | -0.0525 [-0.055, -0.045] | -77.8 % [-78.6 %, -75.0 %] |

## Secondary (EM-consistent reference; *_Fp_stat = floor-free companion of e_F' (descriptive))

| N | Ibar_F_em | Ibar_Fp_em | final_e_F_em | final_e_Fp_em | Ibar_Fp_stat | final_e_Fp_stat |
|---|---|---|---|---|---|---|
| 2048 | ABF 0.01888, FR 0.01355; -26.2 % [-34.8 %, -21.7 %] (32/32) | ABF 0.1065, FR 0.08861; -16.5 % [-19.6 %, -12.2 %] (30/32) | ABF 0.003531, FR 0.001303; -63.8 % [-71.5 %, -56.5 %] (32/32) | ABF 0.03301, FR 0.03239; -1.9 % [-2.2 %, -1.4 %] (32/32) | ABF 0.08599, FR 0.06266; -25.8 % [-29.4 %, -19.8 %] (31/32) | ABF 0.006966, FR 0.002767; -60.8 % [-67.8 %, -52.1 %] (32/32) |
| 512 | ABF 0.006423, FR 0.003898; -36.3 % [-43.2 %, -28.4 %] (32/32) | ABF 0.05276, FR 0.04658; -14.3 % [-17.6 %, -9.4 %] (30/32) | ABF 0.0008729, FR 0.0004017; -56.1 % [-72.0 %, -36.2 %] (27/32) | ABF 0.03237, FR 0.03231; -0.2 % [-0.2 %, -0.1 %] (25/32) | ABF 0.02694, FR 0.01774; -33.2 % [-40.3 %, -24.6 %] (32/32) | ABF 0.002612, FR 0.00162; -36.1 % [-44.6 %, -21.4 %] (25/32) |
| 128 | ABF 0.001899, FR 0.001477; -21.5 % [-37.7 %, -15.2 %] (26/32) | ABF 0.03656, FR 0.03622; -1.4 % [-5.7 %, -0.6 %] (27/32) | ABF 0.0003541, FR 0.0004008; +4.2 % [-17.7 %, +141.0 %] (15/32) | ABF 0.03231, FR 0.03231; +0.0 % [-0.0 %, +0.1 %] (13/32) | ABF 0.008029, FR 0.007029; -17.5 % [-29.7 %, -8.7 %] (27/32) | ABF 0.001593, FR 0.001739; +12.1 % [-8.9 %, +36.2 %] (13/32) |

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
| 2048 | abf | 330.4 | 0.1008 | 159.2 | 32/32 |
| 2048 | fr | 334.4 | 0.1021 | 159.2 | 32/32 |
| 512 | abf | 330.9 | 0.101 | 159.2 | 32/32 |
| 512 | fr | 335 | 0.1022 | 159.2 | 32/32 |
| 128 | abf | 333.7 | 0.1018 | 159.2 | 32/32 |
| 128 | fr | 338.2 | 0.1032 | 159.2 | 32/32 |

## Warnings

* tau_e_Fp_strict: UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp itself is reported)
* tau_e_Fp_em_strict: UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp_em itself is reported)
