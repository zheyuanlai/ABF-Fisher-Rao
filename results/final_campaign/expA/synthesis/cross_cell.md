# Experiment A: cross-cell contrasts (PREREGISTRATION.md 1.2, 1.6)

Generated 2026-10-11T09:27:19Z by scripts/final_campaign/expA_cross_cell.py (final_expA_cross_cell/2); 10000 paired seed-bootstrap resamples (default_rng(20261010)).

R = exp(median over coupled seeds of [L_a - L_b]), L = log(Ibar^FR / Ibar^ABF): **R < 1 means FR's performance relative to ABF is better at lambda_a**.  E1 (16 vs 1) and E2 (128 vs 1) are separate Holm families over the two N; the 97.5 % CI is the Holm-consistent CI of the first step.  Equivalent iff the 95 % CI of R lies in [0.9, 1.11].  Contrasts with fewer than 10 seed pairs make no claim.

Complete: True; production design: True; freshness verified: True; input signature `f4292a660a44813a`.

Completeness audits: lambda 1: PASS (current True, passes True); lambda 4: PASS (current True, passes True); lambda 16: PASS (current True, passes True); lambda 128: PASS (current True, passes True)

## Cells

| lambda | cell | summary | complete ABF / FR per N | h | B |
|---|---|---|---|---|---|
| 1 | shift_lam1 | yes | N 512: 32/32; N 128: 32/32 (of 32) | 2.5e-05 | 3,276,800,000 |
| 4 | shift_lam4 | yes | N 512: 32/32; N 128: 32/32 (of 32) | 2.5e-05 | 3,276,800,000 |
| 16 | shift_lam16 | yes | N 512: 32/32; N 128: 32/32 (of 32) | 2.5e-05 | 3,276,800,000 |
| 128 | shift_lam128 | yes | N 512: 32/32; N 128: 32/32 (of 32) | 2.5e-05 | 3,276,800,000 |

## Primary: Ibar_F, families E1 and E2

| contrast | N | n | R [95 % CI] | R [97.5 % CI] | p | p Holm | wins a / n | equivalent | ABF ratio a/b [95 % CI] | FR ratio a/b [95 % CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 16 vs 1 | 512 | 32 | 0.6525 [0.6268, 0.7848] | 0.6525 [0.6109, 0.7862] | 0.0002 | 0.0004 | 29/32 | no | 0.133 [0.1287, 0.1414] | 0.09165 [0.0856, 0.09981] | Holm p < 0.05: R < 1 (FR relatively better at the higher lambda) |
| E1 16 vs 1 | 128 | 32 | 0.3726 [0.2937, 0.4566] | 0.3726 [0.2765, 0.4731] | 0.0002 | 0.0004 | 29/32 | no | 0.1209 [0.1062, 0.1428] | 0.04405 [0.03929, 0.05659] | Holm p < 0.05: R < 1 (FR relatively better at the higher lambda) |
| E2 128 vs 1 | 512 | 32 | 0.7797 [0.6629, 0.8184] | 0.7797 [0.6626, 0.8267] | 0.0002 | 0.0004 | 30/32 | no | 0.0485 [0.04493, 0.05131] | 0.03446 [0.0336, 0.03747] | Holm p < 0.05: R < 1 (FR relatively better at the higher lambda) |
| E2 128 vs 1 | 128 | 32 | 0.324 [0.2682, 0.3841] | 0.324 [0.2652, 0.3876] | 0.0002 | 0.0004 | 31/32 | no | 0.04022 [0.03613, 0.04909] | 0.01382 [0.01147, 0.01648] | Holm p < 0.05: R < 1 (FR relatively better at the higher lambda) |

E1 (primary (task-specified)): complete True, claim allowed True.

E2 (extension (separate family)): complete True, claim allowed True.

## Trend contrasts (Ibar_F, unadjusted)

| contrast | N | n | R [95 % CI] | p | wins a / n | ABF ratio a/b | FR ratio a/b | verdict |
|---|---|---|---|---|---|---|---|---|
| T1 4 vs 1 | 512 | 32 | 1.166 [1.117, 1.278] | 0.0002 | 2/32 | 0.433 | 0.4949 | trend (unadjusted): CI excludes 1, FR relatively better at lambda_b; p = 0.0002 |
| T1 4 vs 1 | 128 | 32 | 1.129 [0.9585, 1.287] | 0.305 | 13/32 | 0.3851 | 0.4161 | trend (unadjusted): CI includes 1; p = 0.305 |
| T2 16 vs 4 | 512 | 32 | 0.5724 [0.5156, 0.6462] | 0.0002 | 31/32 | 0.3136 | 0.1843 | trend (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| T2 16 vs 4 | 128 | 32 | 0.333 [0.2831, 0.42] | 0.0002 | 31/32 | 0.3371 | 0.1121 | trend (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| T3 128 vs 16 | 512 | 32 | 1.009 [0.8803, 1.172] | 0.433 | 14/32 | 0.368 | 0.3775 | trend (unadjusted): CI includes 1; p = 0.433 |
| T3 128 vs 16 | 128 | 32 | 0.82 [0.6764, 1.073] | 0.0864 | 21/32 | 0.3405 | 0.2807 | trend (unadjusted): CI includes 1; p = 0.0864 |

## Secondary: final_e_F (every row unadjusted)

| contrast | N | n | R [95 % CI] | p | wins a / n | verdict |
|---|---|---|---|---|---|---|
| E1 16 vs 1 | 512 | 32 | 0.5227 [0.4365, 0.5827] | 0.0002 | 32/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| E1 16 vs 1 | 128 | 32 | 0.09545 [0.07563, 0.1537] | 0.0002 | 32/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| E2 128 vs 1 | 512 | 32 | 0.2962 [0.2168, 0.5033] | 0.0002 | 28/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| E2 128 vs 1 | 128 | 32 | 0.0941 [0.04402, 0.1479] | 0.0002 | 31/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| T1 4 vs 1 | 512 | 32 | 1.633 [1.487, 1.787] | 0.0002 | 1/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_b; p = 0.0002 |
| T1 4 vs 1 | 128 | 32 | 0.6577 [0.4816, 1.115] | 0.113 | 20/32 | secondary (unadjusted): CI includes 1; p = 0.113 |
| T2 16 vs 4 | 512 | 32 | 0.3026 [0.2475, 0.3798] | 0.0002 | 32/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| T2 16 vs 4 | 128 | 32 | 0.1509 [0.1095, 0.2956] | 0.0002 | 30/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| T3 128 vs 16 | 512 | 32 | 0.5748 [0.3779, 0.9685] | 0.023 | 22/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.023 |
| T3 128 vs 16 | 128 | 32 | 0.6431 [0.3786, 1.351] | 0.311 | 19/32 | secondary (unadjusted): CI includes 1; p = 0.311 |

## Secondary: Ibar_Fp_stat (every row unadjusted)

| contrast | N | n | R [95 % CI] | p | wins a / n | verdict |
|---|---|---|---|---|---|---|
| E1 16 vs 1 | 512 | 32 | 0.5766 [0.546, 0.6411] | 0.0002 | 32/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| E1 16 vs 1 | 128 | 32 | 0.3529 [0.2944, 0.4242] | 0.0002 | 32/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| E2 128 vs 1 | 512 | 32 | 0.6115 [0.5703, 0.6617] | 0.0002 | 31/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| E2 128 vs 1 | 128 | 32 | 0.2778 [0.2542, 0.3826] | 0.0002 | 32/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| T1 4 vs 1 | 512 | 32 | 0.9065 [0.8624, 0.9634] | 0.001 | 24/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.001 |
| T1 4 vs 1 | 128 | 32 | 0.8624 [0.7551, 0.9959] | 0.0484 | 21/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0484 |
| T2 16 vs 4 | 512 | 32 | 0.6296 [0.5773, 0.7187] | 0.0002 | 31/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| T2 16 vs 4 | 128 | 32 | 0.3958 [0.3558, 0.4572] | 0.0002 | 32/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0002 |
| T3 128 vs 16 | 512 | 32 | 0.9786 [0.9133, 1.186] | 0.861 | 17/32 | secondary (unadjusted): CI includes 1; p = 0.861 |
| T3 128 vs 16 | 128 | 32 | 0.8911 [0.7396, 0.9827] | 0.003 | 24/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.003 |

## Secondary: Ibar_TV_half (every row unadjusted)

| contrast | N | n | R [95 % CI] | p | wins a / n | verdict |
|---|---|---|---|---|---|---|
| E1 16 vs 1 | 512 | 32 | 2.236 [2.029, 2.38] | 0.0002 | 0/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_b; p = 0.0002 |
| E1 16 vs 1 | 128 | 32 | 3.163 [2.996, 3.462] | 0.0002 | 0/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_b; p = 0.0002 |
| E2 128 vs 1 | 512 | 32 | 2.043 [1.966, 2.14] | 0.0002 | 0/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_b; p = 0.0002 |
| E2 128 vs 1 | 128 | 32 | 3.449 [3.11, 3.763] | 0.0002 | 0/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_b; p = 0.0002 |
| T1 4 vs 1 | 512 | 32 | 1.235 [1.213, 1.33] | 0.0002 | 0/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_b; p = 0.0002 |
| T1 4 vs 1 | 128 | 32 | 1.898 [1.705, 2.09] | 0.0002 | 0/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_b; p = 0.0002 |
| T2 16 vs 4 | 512 | 32 | 1.662 [1.611, 1.847] | 0.0002 | 0/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_b; p = 0.0002 |
| T2 16 vs 4 | 128 | 32 | 1.752 [1.636, 1.901] | 0.0002 | 0/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_b; p = 0.0002 |
| T3 128 vs 16 | 512 | 32 | 0.9647 [0.8887, 0.9729] | 0.0008 | 25/32 | secondary (unadjusted): CI excludes 1, FR relatively better at lambda_a; p = 0.0008 |
| T3 128 vs 16 | 128 | 32 | 1.046 [0.9581, 1.145] | 0.196 | 12/32 | secondary (unadjusted): CI includes 1; p = 0.196 |

## Per cell: FR/ABF ratio exp(median L) and paired G

| lambda | N | metric | n | exp(median L) [95 % CI] | G median [95 % CI] (wins/n) | mean G [95 % CI] | mean L [95 % CI] | ABF median [CI] | FR median [CI] |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 512 | Ibar_F | 32 | 0.9285 [0.8601, 0.999] | -7.1 % [-14.0, -0.0] (21/32) | -7.4 % [-12.2, -2.6] | -0.0879 [-0.142, -0.036] | 0.09899 [0.09737, 0.104] | 0.09492 [0.08521, 0.09978] |
| 1 | 512 | final_e_F | 32 | 1.459 [1.313, 1.573] | +45.9 % [+31.3, +57.3] (1/32) | +47.6 % [+39.0, +56.4] | 0.375 [0.315, 0.434] | 0.04961 [0.04841, 0.05165] | 0.07505 [0.06782, 0.07955] |
| 1 | 512 | Ibar_Fp_stat | 32 | 1.154 [1.084, 1.195] | +15.4 % [+8.4, +19.5] (2/32) | +15.0 % [+10.8, +19.2] | 0.134 [0.096, 0.17] | 0.2206 [0.2172, 0.2277] | 0.2563 [0.239, 0.267] |
| 1 | 512 | Ibar_TV_half | 32 | 0.1203 [0.1172, 0.1245] | -88.0 % [-88.3, -87.5] (32/32) | -87.8 % [-88.2, -87.5] | -2.11 [-2.14, -2.08] | 0.2727 [0.2668, 0.2821] | 0.03317 [0.03194, 0.03415] |
| 1 | 128 | Ibar_F | 32 | 1.923 [1.744, 2.414] | +92.3 % [+74.4, +141.4] (1/32) | +107.8 % [+83.9, +131.7] | 0.674 [0.547, 0.792] | 0.03739 [0.03328, 0.03939] | 0.07339 [0.06797, 0.07758] |
| 1 | 128 | final_e_F | 32 | 11.93 [9.292, 14.45] | +1093.7 % [+829.2, +1345.2] (0/32) | +1333.0 % [+1035.8, +1652.2] | 2.47 [2.24, 2.68] | 0.005337 [0.004115, 0.006304] | 0.05927 [0.05565, 0.06364] |
| 1 | 128 | Ibar_Fp_stat | 32 | 2.283 [2.026, 2.463] | +128.3 % [+102.8, +146.3] (0/32) | +130.9 % [+111.7, +149.9] | 0.807 [0.72, 0.892] | 0.08187 [0.07737, 0.08633] | 0.1853 [0.1697, 0.1972] |
| 1 | 128 | Ibar_TV_half | 32 | 0.08014 [0.07573, 0.08486] | -92.0 % [-92.4, -91.5] (32/32) | -91.9 % [-92.3, -91.5] | -2.53 [-2.58, -2.47] | 0.2388 [0.2303, 0.2461] | 0.0188 [0.01803, 0.01933] |
| 4 | 512 | Ibar_F | 32 | 1.101 [1.028, 1.206] | +10.1 % [+2.8, +20.6] (7/32) | +12.0 % [+6.4, +17.9] | 0.103 [0.0529, 0.153] | 0.04342 [0.04203, 0.04473] | 0.04762 [0.0449, 0.05145] |
| 4 | 512 | final_e_F | 32 | 2.456 [2.297, 2.548] | +145.6 % [+129.7, +154.9] (0/32) | +142.0 % [+130.2, +154.2] | 0.874 [0.824, 0.924] | 0.01226 [0.01183, 0.01319] | 0.02934 [0.028, 0.03166] |
| 4 | 512 | Ibar_Fp_stat | 32 | 1.047 [1.012, 1.146] | +4.7 % [+1.2, +14.6] (10/32) | +6.9 % [+2.5, +11.5] | 0.0598 [0.0186, 0.101] | 0.09305 [0.09158, 0.09721] | 0.1013 [0.09516, 0.1051] |
| 4 | 512 | Ibar_TV_half | 32 | 0.1566 [0.1502, 0.1612] | -84.3 % [-85.0, -83.9] (32/32) | -84.5 % [-85.1, -84.0] | -1.87 [-1.91, -1.84] | 0.1827 [0.1764, 0.192] | 0.02794 [0.02738, 0.02866] |
| 4 | 128 | Ibar_F | 32 | 2.251 [2.023, 2.354] | +125.1 % [+102.3, +135.4] (0/32) | +127.3 % [+105.9, +149.5] | 0.782 [0.684, 0.879] | 0.01404 [0.01277, 0.01499] | 0.0295 [0.02874, 0.03159] |
| 4 | 128 | final_e_F | 32 | 9.106 [6.088, 9.458] | +810.7 % [+510.2, +845.8] (0/32) | +951.7 % [+701.1, +1267.3] | 2.15 [1.95, 2.37] | 0.002318 [0.001916, 0.002875] | 0.01918 [0.01756, 0.02111] |
| 4 | 128 | Ibar_Fp_stat | 32 | 1.968 [1.716, 2.08] | +96.8 % [+71.6, +108.0] (0/32) | +99.3 % [+83.6, +115.8] | 0.662 [0.581, 0.744] | 0.0308 [0.02862, 0.03388] | 0.05927 [0.05769, 0.06452] |
| 4 | 128 | Ibar_TV_half | 32 | 0.1445 [0.1347, 0.1592] | -85.5 % [-86.5, -84.1] (32/32) | -85.0 % [-86.0, -84.1] | -1.92 [-1.98, -1.85] | 0.1001 [0.09251, 0.106] | 0.01474 [0.01441, 0.01537] |
| 16 | 512 | Ibar_F | 32 | 0.6631 [0.6031, 0.7012] | -33.7 % [-39.7, -29.9] (32/32) | -34.3 % [-38.8, -30.1] | -0.44 [-0.514, -0.373] | 0.01344 [0.01262, 0.01413] | 0.00879 [0.008434, 0.009209] |
| 16 | 512 | final_e_F | 32 | 0.7324 [0.6237, 0.9425] | -26.8 % [-37.6, -5.7] (25/32) | -19.5 % [-30.1, -8.3] | -0.292 [-0.435, -0.156] | 0.003171 [0.002889, 0.003647] | 0.002373 [0.002232, 0.002886] |
| 16 | 512 | Ibar_Fp_stat | 32 | 0.6547 [0.6084, 0.7688] | -34.5 % [-39.2, -23.1] (31/32) | -30.6 % [-35.1, -26.0] | -0.382 [-0.447, -0.318] | 0.03892 [0.03527, 0.04014] | 0.02592 [0.02402, 0.02774] |
| 16 | 512 | Ibar_TV_half | 32 | 0.262 [0.2518, 0.2749] | -73.8 % [-74.8, -72.5] (32/32) | -73.7 % [-74.9, -72.6] | -1.34 [-1.39, -1.3] | 0.08477 [0.0815, 0.08969] | 0.02248 [0.02221, 0.02274] |
| 16 | 128 | Ibar_F | 32 | 0.7711 [0.6489, 0.9289] | -22.9 % [-35.1, -7.1] (24/32) | -13.9 % [-27.1, +0.5] | -0.255 [-0.428, -0.0951] | 0.004281 [0.003676, 0.004763] | 0.003468 [0.003061, 0.003927] |
| 16 | 128 | final_e_F | 32 | 1.451 [0.908, 2.393] | +45.2 % [-9.2, +139.3] (11/32) | +107.3 % [+48.9, +175.2] | 0.375 [0.0771, 0.67] | 0.000805 [0.0004876, 0.001207] | 0.00119 [0.001044, 0.001372] |
| 16 | 128 | Ibar_Fp_stat | 32 | 0.8159 [0.7097, 0.9081] | -18.4 % [-29.0, -9.2] (28/32) | -18.7 % [-25.6, -11.9] | -0.239 [-0.333, -0.151] | 0.01391 [0.0135, 0.01454] | 0.01103 [0.0104, 0.012] |
| 16 | 128 | Ibar_TV_half | 32 | 0.2634 [0.2341, 0.2811] | -73.7 % [-76.6, -71.8] (32/32) | -73.5 % [-75.5, -71.6] | -1.35 [-1.43, -1.28] | 0.04214 [0.03809, 0.04587] | 0.01079 [0.0106, 0.01129] |
| 128 | 512 | Ibar_F | 32 | 0.6582 [0.6198, 0.6982] | -34.2 % [-38.0, -30.2] (30/32) | -29.8 % [-35.2, -23.6] | -0.379 [-0.447, -0.304] | 0.005089 [0.004668, 0.005216] | 0.003248 [0.003045, 0.003404] |
| 128 | 512 | final_e_F | 32 | 0.4176 [0.3114, 0.7627] | -58.2 % [-68.9, -23.7] (25/32) | -17.1 % [-47.2, +19.9] | -0.69 [-1.02, -0.349] | 0.0005884 [0.0004147, 0.0006139] | 0.0001915 [0.0001458, 0.0002595] |
| 128 | 512 | Ibar_Fp_stat | 32 | 0.6958 [0.644, 0.7371] | -30.4 % [-35.6, -26.3] (28/32) | -25.7 % [-32.7, -17.3] | -0.332 [-0.411, -0.242] | 0.02398 [0.02341, 0.02693] | 0.01694 [0.0156, 0.01856] |
| 128 | 512 | Ibar_TV_half | 32 | 0.2429 [0.2347, 0.2506] | -75.7 % [-76.5, -74.9] (32/32) | -75.5 % [-76.2, -74.8] | -1.41 [-1.44, -1.38] | 0.09123 [0.08683, 0.09357] | 0.0223 [0.02179, 0.02251] |
| 128 | 128 | Ibar_F | 32 | 0.6803 [0.5624, 0.7667] | -32.0 % [-43.8, -23.3] (28/32) | -31.2 % [-41.9, -20.0] | -0.496 [-0.686, -0.32] | 0.001454 [0.001331, 0.001592] | 0.001006 [0.000888, 0.001129] |
| 128 | 128 | final_e_F | 32 | 1.047 [0.6067, 1.515] | +5.0 % [-39.2, +51.5] (16/32) | +46.5 % [+3.2, +97.1] | -0.000456 [-0.302, 0.302] | 0.0002478 [0.0002208, 0.0002826] | 0.0002492 [0.0001929, 0.0003313] |
| 128 | 128 | Ibar_Fp_stat | 32 | 0.6787 [0.6088, 0.8256] | -32.1 % [-39.1, -17.4] (30/32) | -32.4 % [-40.1, -25.0] | -0.456 [-0.595, -0.331] | 0.008909 [0.007475, 0.01022] | 0.006342 [0.00614, 0.00661] |
| 128 | 128 | Ibar_TV_half | 32 | 0.2724 [0.2564, 0.2937] | -72.8 % [-74.4, -70.6] (32/32) | -72.4 % [-73.8, -71.0] | -1.3 [-1.35, -1.25] | 0.0401 [0.03701, 0.04194] | 0.01085 [0.01047, 0.01131] |
