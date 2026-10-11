# Equal-budget ladder analysis: gateway_family -- mechanism cell expA/shift_lam128: A: shifted fibre, kappa = 1, transverse mobility lambda = 128 (preregistered extension)

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/final_campaign/cells/expA/shift_lam128.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/final_campaign/expA`; generated 2026-10-11T09:25:19Z; eqb_metrics/2.
B = 3,276,800,000 walker-steps per arm per seed; h = 2.5e-05; seeds 32; thresholds e_F [0.002, 0.0055, 0.02], e_F' [0.012, 0.035, 0.1], TV_half 0.1, e_Fp_stat [0.004, 0.0136, 0.0946].
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/final_campaign/references/expA_shift_lam128_reference.npz, sha256 f6ce66cda7442f98b810e9ea0957d3c094cc8e80385be45ee70156f4c933906a, n_eval_nodes 151, em_floor_F_rms 1.0496374093956405e-09, em_bias_Fp_bin_rms 0.0, floor_definition errors of the scorer fed the exact bin-averaged reference mean force (Gamma_j = <F'_ref>_j on the Gauss-Legendre sub-grid, infinite counts); for e_Fp / e_Fp_em it is a HARD lower bound: e_Fp^2 = e_Fp_stat^2 + floor^2 for every Gamma (within-bin variation of F'_ref, steep at the gate); e_F at Gamma = <F'_ref> is ~0, units reduced (energy; force per unit x), scorer scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, identical for every variant and lambda); secondary = eqb_family.family_mean_force (per-dynamics EM-consistent), em_bias_definition em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j, F'_em = this dynamics' frozen-x EM-consistent mean force (NOT the e_F' floor), secondary_definition per-dynamics frozen-x Euler-Maruyama-consistent mean force at the cell's h and lambda: alpha family 4Hx(x^2-1) + ((1-alpha)/beta) w'/w + (alpha/beta)(w'/w) / (1 - lam w^(2 alpha) h/2) (the EM y-update y <- (1 - lam w^(2 alpha) h) y + sqrt(2 lam h/beta) z has stationary variance 1/(beta w^(2 alpha) (1 - lam w^(2 alpha) h/2))); shifted fibre: F*'(x) exactly (the frozen-x EM chain of y has mean m(x) exactly), secondary_equals_equal_budget_em None.
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
| 512 | 6,400,000 | 160 | 32/32 | 32/32 | -- |
| 128 | 25,600,000 | 640 | 32/32 | 32/32 | -- |

## Absolute values (median [IQR] over seeds)

| N | T_N | ABF Ibar_F | FR Ibar_F | ABF Ibar_F' | FR Ibar_F' | ABF e_F(1) | FR e_F(1) | ABF e_F'(1) | FR e_F'(1) |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 160 | 0.005089 [0.004471, 0.005266] | 0.003248 [0.002993, 0.003561] | 0.052 [0.0507, 0.05515] | 0.04626 [0.04448, 0.04768] | 0.0005884 [0.0002902, 0.0007215] | 0.0001915 [0.0001322, 0.0003681] | 0.03233 [0.03231, 0.03235] | 0.0323 [0.03229, 0.03232] |
| 128 | 640 | 0.001454 [0.001257, 0.001686] | 0.001006 [0.0006425, 0.00116] | 0.03775 [0.03634, 0.03981] | 0.03591 [0.03462, 0.03621] | 0.0002478 [0.0001673, 0.0003782] | 0.0002492 [0.0001623, 0.0004144] | 0.03231 [0.0323, 0.03232] | 0.03231 [0.0323, 0.03232] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 512 | -34.2 % [-38.0 %, -30.2 %] (30/32) | -13.3 % [-15.3 %, -10.2 %] (28/32) | -58.2 % [-68.9 %, -23.7 %] (25/32) | -0.1 % [-0.1 %, -0.0 %] (22/32) | -75.7 % [-76.5 %, -74.9 %] (32/32) | -68.9 % [-76.0 %, -63.2 %] (32/32) | 18.5 deaths / walker |
| 128 | -32.0 % [-43.8 %, -23.3 %] (28/32) | -8.2 % [-9.6 %, -2.5 %] (30/32) | +5.0 % [-39.2 %, +51.5 %] (16/32) | -0.0 % [-0.0 %, +0.0 %] (17/32) | -72.8 % [-74.4 %, -70.6 %] (32/32) | -68.4 % [-72.3 %, -58.6 %] (31/32) | 150 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 512 | e_F_strict | 0.2175 | 0 | 0.0525 | 0 | 31/1/0 | 1.54e-08 | -72.1 % [-80.0 %, -60.6 %] (n 32) |
| 512 | e_F_mid | 0.05 | 0 | 0.025 | 0 | 28/0/4 | 7.45e-09 | -50.0 % [-54.5 %, -37.5 %] (n 32) |
| 512 | e_F_loose | 0.03 | 0 | 0.02 | 0 | 26/2/4 | 3.03e-06 | -31.0 % [-33.3 %, -28.6 %] (n 32) |
| 512 | e_Fp_stat_strict | 0.2975 | 0 | 0.18 | 0 | 24/7/1 | 0.00333 | -41.9 % [-47.4 %, -23.9 %] (n 32) |
| 512 | e_Fp_stat_mid | 0.05 | 0 | 0.03 | 0 | 25/2/5 | 5.65e-06 | -38.8 % [-50.0 %, -29.2 %] (n 32) |
| 512 | e_Fp_stat_loose | 0.0325 | 0 | 0.025 | 0 | 28/3/1 | 4.65e-06 | -28.6 % [-33.3 %, -24.3 %] (n 32) |
| 512 | TV_half | 0.2725 | 0 | 0.05 | 0 | 32/0/0 | 4.66e-10 | -83.3 % [-84.1 %, -81.6 %] (n 32) |
| 512 | e_F_em_strict (secondary) | 0.2175 | 0 | 0.0525 | 0 | 31/1/0 | 1.54e-08 | -72.1 % [-80.0 %, -60.6 %] (n 32) |
| 512 | e_F_em_mid (secondary) | 0.05 | 0 | 0.025 | 0 | 28/0/4 | 7.45e-09 | -50.0 % [-54.5 %, -37.5 %] (n 32) |
| 512 | e_F_em_loose (secondary) | 0.03 | 0 | 0.02 | 0 | 26/2/4 | 3.03e-06 | -31.0 % [-33.3 %, -28.6 %] (n 32) |
| 512 | e_Fp_mid (secondary) | 0.05 | 0 | 0.03 | 0 | 25/2/5 | 5.65e-06 | -38.8 % [-50.0 %, -29.2 %] (n 32) |
| 512 | e_Fp_loose (secondary) | 0.0325 | 0 | 0.025 | 0 | 28/3/1 | 4.65e-06 | -28.6 % [-33.3 %, -24.3 %] (n 32) |
| 512 | e_Fp_em_mid (secondary) | 0.05 | 0 | 0.03 | 0 | 25/2/5 | 5.65e-06 | -38.8 % [-50.0 %, -29.2 %] (n 32) |
| 512 | e_Fp_em_loose (secondary) | 0.0325 | 0 | 0.025 | 0 | 28/3/1 | 4.65e-06 | -28.6 % [-33.3 %, -24.3 %] (n 32) |
| 128 | e_F_strict | 0.0675 | 0 | 0.04 | 0 | 20/9/3 | 0.0614 | -45.6 % [-61.7 %, +0.0 %] (n 32) |
| 128 | e_F_mid | 0.015 | 0 | 0.01 | 0 | 22/4/6 | 0.000534 | -33.3 % [-37.5 %, -4.0 %] (n 32) |
| 128 | e_F_loose | 0.01 | 0 | 0.00625 | 0 | 23/0/9 | 2.38e-07 | -37.5 % [-50.0 %, -33.3 %] (n 32) |
| 128 | e_Fp_stat_strict | 0.2025 | 0 | 0.1825 | 0 | 19/13/0 | 0.377 | -23.0 % [-34.1 %, +7.7 %] (n 32) |
| 128 | e_Fp_stat_mid | 0.025 | 0 | 0.01562 | 0 | 25/5/2 | 0.000325 | -25.0 % [-40.0 %, -17.1 %] (n 32) |
| 128 | e_Fp_stat_loose | 0.0125 | 0 | 0.00625 | 0 | 25/1/6 | 8.05e-07 | -37.5 % [-43.8 %, -33.3 %] (n 32) |
| 128 | TV_half | 0.075 | 0 | 0.015 | 0 | 32/0/0 | 4.66e-10 | -80.0 % [-82.4 %, -79.3 %] (n 32) |
| 128 | e_F_em_strict (secondary) | 0.0675 | 0 | 0.04 | 0 | 20/9/3 | 0.0614 | -45.6 % [-61.7 %, +0.0 %] (n 32) |
| 128 | e_F_em_mid (secondary) | 0.015 | 0 | 0.01 | 0 | 22/4/6 | 0.000534 | -33.3 % [-37.5 %, -4.0 %] (n 32) |
| 128 | e_F_em_loose (secondary) | 0.01 | 0 | 0.00625 | 0 | 23/0/9 | 2.38e-07 | -37.5 % [-50.0 %, -33.3 %] (n 32) |
| 128 | e_Fp_mid (secondary) | 0.025 | 0 | 0.01562 | 0 | 25/6/1 | 0.000878 | -25.0 % [-40.0 %, -17.1 %] (n 32) |
| 128 | e_Fp_loose (secondary) | 0.0125 | 0 | 0.00625 | 0 | 25/1/6 | 8.05e-07 | -37.5 % [-43.8 %, -33.3 %] (n 32) |
| 128 | e_Fp_em_mid (secondary) | 0.025 | 0 | 0.01562 | 0 | 25/6/1 | 0.000878 | -25.0 % [-40.0 %, -17.1 %] (n 32) |
| 128 | e_Fp_em_loose (secondary) | 0.0125 | 0 | 0.00625 | 0 | 25/1/6 | 8.05e-07 | -37.5 % [-43.8 %, -33.3 %] (n 32) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_stat_strict | e_Fp_stat_mid | e_Fp_stat_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 512 | abf | 0.2175 [0.1388, 0.285] (0) | 0.05 [0.04, 0.055] (0) | 0.03 [0.025, 0.035] (0) | 0.2975 [0.2575, 0.3475] (0) | 0.05 [0.04, 0.05625] (0) | 0.0325 [0.03, 0.035] (0) | 0.2725 [0.26, 0.31] (0) |
| 512 | fr | 0.0525 [0.035, 0.075] (0) | 0.025 [0.025, 0.03] (0) | 0.02 [0.02, 0.025] (0) | 0.18 [0.1437, 0.2462] (0) | 0.03 [0.025, 0.035] (0) | 0.025 [0.02, 0.025] (0) | 0.05 [0.045, 0.05] (0) |
| 128 | abf | 0.0675 [0.04, 0.115] (0) | 0.015 [0.015, 0.02] (0) | 0.01 [0.01, 0.015] (0) | 0.2025 [0.1775, 0.265] (0) | 0.025 [0.02, 0.03] (0) | 0.0125 [0.01, 0.015] (0) | 0.075 [0.07, 0.1] (0) |
| 128 | fr | 0.04 [0.02375, 0.0575] (0) | 0.01 [0.01, 0.015] (0) | 0.00625 [0.00625, 0.01] (0) | 0.1825 [0.145, 0.2162] (0) | 0.01562 [0.015, 0.02] (0) | 0.00625 [0.00625, 0.01] (0) | 0.015 [0.015, 0.015] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 512 | 93 | 1.53e-19 | 1.53e-19 | 1.000 | 0.0001 | no |
| 128 | 24 | 5.53e-06 | 5.53e-06 | 0.999 | 0.0001 | no |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | abf | 0.0139 [0.01006, 0.01833] | 0.2725 [0.26, 0.31] | 0.07172 [0.06478, 0.08371] | 0.0728 | 0.115 [0.11, 0.125] | 0 | 0.265 [0.2487, 0.28] | 0.01003 [0.008748, 0.01161] | 2578 [2527, 2612] |
| 512 | fr | 0.004053 [0.003669, 0.004735] | 0.05 [0.045, 0.05] | 0.06141 [0.05138, 0.07118] | 0.0728 | 0.03 [0.03, 0.03] | 0 | 0.055 [0.055, 0.055] | 0.009196 [0.00871, 0.01038] | 2606 [2587, 2638] |
| 128 | abf | 0.01349 [0.01083, 0.01647] | 0.075 [0.07, 0.1] | 0.1493 [0.1335, 0.1684] | 0.1447 | 0.03313 [0.03, 0.03625] | 0 | 0.07 [0.06438, 0.075] | 0.00375 [0.003154, 0.004331] | 2637 [2586, 2672] |
| 128 | fr | 0.004675 [0.004057, 0.005145] | 0.015 [0.015, 0.015] | 0.1181 [0.1042, 0.1369] | 0.1447 | 0.01 [0.00625, 0.01] | 0 | 0.015 [0.015, 0.015] | 0.002784 [0.002285, 0.003191] | 2600 [2563, 2630] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems; the gateway's capped candidates kd + kc are the last column. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win | candidates/N per opp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 0/32 | 40 | 0 | 9476 [9378, 9532] | 18.51 [18.32, 18.62] | 0.2206 [0.2189, 0.2221] | 0.0004627 [0.0004579, 0.0004654] | 0.007812 [0.005859, 0.007812] | 0.02044 [0.01702, 0.02224] | 0.3315 [0.2904, 0.369] | 20.5 [19, 23] | 0.03809 [0.03125, 0.05176] | 0.0004914 [0.0004881, 0.0004945] |
| 128 | 0/32 | 10 | 0 | 1.926e+04 [1.918e+04, 1.931e+04] | 150.5 [149.9, 150.8] | 0.1166 [0.116, 0.1169] | 0.0009406 [0.0009366, 0.0009427] | 0.02344 [0.02344, 0.02344] | 0.007812 [0.007812, 0.007812] | 0.4197 [0.3926, 0.4369] | 1 [1, 1] | 0.07031 [0.06836, 0.07812] | 0.0009746 [0.0009699, 0.0009784] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 512 | e_F | +94.6 % | 0.025 | -10.4 % | 0.015 | +67.5 % | 32 |
| 512 | e_Fp | +95.0 % | 0.025 | -0.0 % | 0.005 | +0.1 % | 32 |
| 512 | e_F_em | +94.6 % | 0.025 | -10.4 % | 0.015 | +67.5 % | 32 |
| 512 | TV_half | +95.9 % | 0.085 | +0.5 % | 0.005 | +70.8 % | 32 |
| 128 | e_F | +69.4 % | 0.01 | -21.4 % | 0.46 | -0.6 % | 32 |
| 128 | e_Fp | +78.9 % | 0.01 | -0.0 % | 0.895 | +0.0 % | 32 |
| 128 | e_F_em | +69.4 % | 0.01 | -21.4 % | 0.46 | -0.6 % | 32 |
| 128 | TV_half | +91.1 % | 0.025 | +14.5 % | 0.005 | +65.3 % | 32 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 128 | 0.001454 | [0.001331, 0.001592] | 128: 1.00 | 128 | 0.001006 | [0.000888, 0.001129] | 128: 1.00 | -0.0004484 [-0.0006204, -0.0002375] | -30.8 % [-39.6 %, -17.5 %] |
| final_e_F | 128 | 0.0002478 | [0.0002208, 0.0002826] | 128: 1.00, 512: 0.00 | 512 | 0.0001915 | [0.0001506, 0.0002427] | 512: 0.92, 128: 0.08 | -5.629e-05 [-0.0001134, 7.563e-06] | -22.7 % [-41.7 %, +3.3 %] |
| tau_bgrid_e_F_mid_u | 128 | 0.015 | [0.015, 0.02] | 128: 1.00 | 128 | 0.01 | [0.01, 0.015] | 128: 1.00 | -0.005 [-0.01, -0.0025] | -33.3 % [-50.0 %, -14.3 %] |
| Ibar_Fp | 128 | 0.03775 | [0.03636, 0.0397] | 128: 1.00 | 128 | 0.03591 | [0.03537, 0.03618] | 128: 1.00 | -0.001841 [-0.004011, -0.0003038] | -4.9 % [-10.1 %, -0.8 %] |
| final_e_Fp | 128 | 0.03231 | [0.0323, 0.03231] | 128: 0.99, 512: 0.01 | 512 | 0.0323 | [0.03229, 0.03231] | 512: 0.85, 128: 0.15 | -5.058e-06 [-1.681e-05, 5.156e-06] | -0.0 % [-0.1 %, +0.0 %] |
| tau_bgrid_TV_half_u | 128 | 0.075 | [0.075, 0.0825] | 128: 1.00 | 128 | 0.015 | [0.015, 0.015] | 128: 1.00 | -0.06 [-0.0675, -0.06] | -80.0 % [-81.8 %, -80.0 %] |

## Secondary (EM-consistent reference; *_Fp_stat = floor-free companion of e_F' (descriptive))

| N | Ibar_F_em | Ibar_Fp_em | final_e_F_em | final_e_Fp_em | Ibar_Fp_stat | final_e_Fp_stat |
|---|---|---|---|---|---|---|
| 512 | ABF 0.005089, FR 0.003248; -34.2 % [-38.0 %, -30.2 %] (30/32) | ABF 0.052, FR 0.04626; -13.3 % [-15.3 %, -10.2 %] (28/32) | ABF 0.0005884, FR 0.0001915; -58.2 % [-68.9 %, -23.7 %] (25/32) | ABF 0.03233, FR 0.0323; -0.1 % [-0.1 %, -0.0 %] (22/32) | ABF 0.02398, FR 0.01694; -30.4 % [-35.6 %, -26.3 %] (28/32) | ABF 0.00194, FR 0.001509; -16.2 % [-31.4 %, -3.3 %] (22/32) |
| 128 | ABF 0.001454, FR 0.001006; -32.0 % [-43.8 %, -23.3 %] (28/32) | ABF 0.03775, FR 0.03591; -8.2 % [-9.6 %, -2.5 %] (30/32) | ABF 0.0002478, FR 0.0002492; +5.0 % [-39.2 %, +51.5 %] (16/32) | ABF 0.03231, FR 0.03231; -0.0 % [-0.0 %, +0.0 %] (17/32) | ABF 0.008909, FR 0.006342; -32.1 % [-39.1 %, -17.4 %] (30/32) | ABF 0.001614, FR 0.001611; -0.6 % [-6.8 %, +14.5 %] (17/32) |

## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)

exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial).

| N | E_N[TV] |
|---|---|
| 128 | 0.14473 |
| 512 | 0.07281 |

## Cost (median per run)

| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |
|---|---|---|---|---|---|
| 512 | abf | 281.3 | 0.08586 | 155.8 | 32/32 |
| 512 | fr | 285.6 | 0.08717 | 155.8 | 32/32 |
| 128 | abf | 282.5 | 0.08621 | 157.8 | 32/32 |
| 128 | fr | 288.1 | 0.08791 | 158.8 | 32/32 |

## Warnings

* tau_e_Fp_strict: UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp itself is reported)
* tau_e_Fp_em_strict: UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp_em itself is reported)
