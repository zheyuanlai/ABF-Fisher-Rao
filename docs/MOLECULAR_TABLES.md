# Molecular campaign: every table, machine-generated

Regenerate with `bash scripts/make_mol_tables.sh`.  Numbers here come from
the stored `.npz` archives, never retyped.

## Pentane: frozen hyper-parameters (chosen on 8 screening seeds)

| arm | frozen | screen I_F | screen e_F | screen D_cond |
|---|---|---|---|---|
| wfr_rot | kappa=1.2 theta=0.3 decay=0.999 - | 0.0579 | 0.0466 | 0.2109 |
| wfr_shake | kappa=0.075 theta=0.6 decay=0.999 - | 0.1219 | 0.1014 | 0.1918 |
| wfr_ymap | kappa=0.15 theta=0.6 decay=0.999 - | 0.0346 | 0.0247 | 0.0411 |
| wfr_yref | kappa=0.075 theta=0.3 decay=0.999 - | 0.0322 | 0.0273 | 0.0093 |
| wfr_lmap | kappa=0.3 theta=0.3 decay=1 bz0.4 | 0.0851 | 0.0799 | 0.2233 |
| wfr_lref | kappa=0.3 theta=0.3 decay=0.999 bz0.4 | 0.0783 | 0.0499 | 1.0288 |
| wfr_ymh | kappa=0.15 theta=0.15 decay=0.999 - | 0.0258 | 0.0210 | 0.0099 |
| wfr_lmh | kappa=0.6 theta=0.3 decay=0.999 bz0.25 | 0.0297 | 0.0236 | 0.0105 |
| wfr_qref | kappa=0.15 theta=0.3 decay=0.999 - | 0.0881 | 0.0884 | 0.0122 |
| ti_cold | kappa=0.3 theta=0.3 decay=0.999 w32 | 0.0796 | 0.0641 | 0.1723 |
| ti_warm | kappa=0.3 theta=0.3 decay=0.999 w64 | 0.0474 | 0.0343 | 0.1756 |
| abf | kappa=0.3 theta=0.3 decay=0.999 n50 | 0.0667 | 0.0505 | 0.0525 |

12 confirmation runs -> results/mol/confirm_spec.json

## Pentane: confirmation, 32 fresh seeds

### PEN: free-energy error at three budgets (32 fresh seeds, median [IQR], kcal/mol)

| arm | 0.5x | 1x | 4x | D_cond (4x) | ESS_Fix | wall (s) |
|---|---|---|---|---|---|---|
| stratified constrained TI (cold) | 0.0818 [0.0701,0.0919] | 0.0633 [0.0522,0.0768] | 0.0378 [0.0327,0.0442] | 0.1352 | 0.980 | 539 |
| stratified constrained TI (warm start, oracle) | 0.0470 [0.0323,0.0612] | 0.0410 [0.0331,0.0564] | 0.0299 [0.0222,0.0367] | 0.1526 | 0.980 | 535 |
| ABF, multiple walkers | 0.0568 [0.0522,0.0660] | 0.0540 [0.0501,0.0605] | 0.0314 [0.0281,0.0344] | 0.0259 | 1.000 | 471 |
| RC-WFR, min-norm SHAKE lift | 0.1286 [0.1155,0.1424] | 0.1181 [0.1033,0.1253] | 0.1003 [0.0931,0.1087] | 0.1704 | 0.979 | 695 |
| RC-WFR, naive rotation lift | 0.0714 [0.0597,0.0809] | 0.0596 [0.0530,0.0703] | 0.0475 [0.0407,0.0523] | 0.1940 | 0.980 | 711 |
| RC-WFR + oracle y CDF-map | 0.0334 [0.0265,0.0397] | 0.0290 [0.0238,0.0324] | 0.0223 [0.0205,0.0242] | 0.0270 | 0.979 | 709 |
| RC-WFR + oracle y refresh | 0.0471 [0.0400,0.0515] | 0.0410 [0.0394,0.0468] | 0.0419 [0.0396,0.0432] | 0.0297 | 0.979 | 705 |
| RC-WFR + learned y CDF-map | 0.0794 [0.0738,0.0926] | 0.0758 [0.0714,0.0834] | 0.0710 [0.0662,0.0754] | 0.1917 | 0.980 | 636 |
| RC-WFR + learned y refresh | 0.0713 [0.0605,0.0803] | 0.0516 [0.0439,0.0649] | 0.2917 [0.2786,0.3106] | 1.4860 | 0.974 | 614 |
| RC-WFR + Metropolis y-move, oracle proposal | 0.0296 [0.0231,0.0330] | 0.0256 [0.0203,0.0280] | 0.0196 [0.0183,0.0232] | 0.0049 | 0.980 | 627 |
| RC-WFR + Metropolis y-move, LEARNED proposal | 0.0264 [0.0196,0.0334] | 0.0259 [0.0219,0.0288] | 0.0215 [0.0189,0.0235] | 0.0051 | 0.980 | 663 |
| RC-WFR + full conditional refresh (ceiling) | 0.0861 [0.0822,0.0896] | 0.0848 [0.0827,0.0882] | 0.0835 [0.0821,0.0852] | 0.0049 | 0.980 | 649 |
| ablation: W only (no Fisher-Rao) | 0.0659 [0.0532,0.0764] | 0.0565 [0.0528,0.0647] | 0.0444 [0.0416,0.0473] | 0.1829 | 0.980 | 568 |
| ablation: Fisher-Rao only (no transport) | 1.2622 [1.2622,1.2622] | 1.2622 [1.2622,1.2622] | 1.2622 [1.2622,1.2622] | 0.0027 | 0.999 | 558 |
| ablation: W + count balancing | 0.0771 [0.0519,0.0952] | 0.0664 [0.0503,0.0711] | 0.0448 [0.0390,0.0514] | 0.2277 | 0.980 | 582 |
| ablation: W only, oracle y-refresh | 0.0476 [0.0413,0.0523] | 0.0429 [0.0405,0.0450] | 0.0415 [0.0403,0.0424] | 0.0292 | 0.980 | 586 |
| probability-flow W, naive lift | 1.2622 [1.2622,1.2622] | 1.2622 [1.2622,1.2622] | 1.2622 [1.2622,1.2622] | 0.0027 | 0.999 | 606 |
| probability-flow W, oracle y-refresh | 1.2622 [1.2622,1.2622] | 1.2622 [1.2622,1.2622] | 1.2622 [1.2622,1.2622] | 0.0042 | 0.999 | 614 |

Estimator floor at this bandwidth: **0.0127** kcal/mol.
Force evaluations at 4x: 1.07e+08.

### Paired relative change vs each comparator (median, 95% bootstrap CI)

| arm | vs | budget | change in e_F | change in I_F |
|---|---|---|---|---|
| wfr_rot | wfr_shake | 1x | **-45.2%** [-54.9, -38.0] | -44.0% [-50.8, -34.7] |
| wfr_rot | wfr_shake | 4x | **-53.7%** [-56.6, -49.0] | -47.9% [-53.1, -44.7] |
| wfr_ymap | wfr_rot | 1x | **-56.2%** [-59.7, -51.0] | -51.9% [-55.5, -48.2] |
| wfr_ymap | wfr_rot | 4x | **-52.0%** [-57.3, -47.9] | -51.0% [-56.0, -46.4] |
| wfr_yref | wfr_rot | 1x | **-27.1%** [-37.1, -20.4] | -34.7% [-42.9, -24.2] |
| wfr_yref | wfr_rot | 4x | **-10.8%** [-18.5, -0.2] | -23.4% [-31.2, -12.7] |
| wfr_ymh | wfr_rot | 1x | **-61.0%** [-65.4, -48.9] | -55.8% [-64.7, -49.9] |
| wfr_ymh | wfr_rot | 4x | **-54.4%** [-60.2, -49.4] | -57.2% [-60.7, -50.1] |
| wfr_lmh | wfr_rot | 1x | **-60.5%** [-63.9, -52.5] | -60.5% [-64.2, -52.1] |
| wfr_lmh | wfr_rot | 4x | **-53.6%** [-57.7, -50.1] | -57.1% [-59.4, -52.6] |
| wfr_lmh | wfr_ymh | 1x | +3.8% [-1.3, +11.2] | -0.8% [-8.6, +3.8] |
| wfr_lmh | wfr_ymh | 4x | +1.2% [-2.6, +15.3] | +3.5% [-6.6, +15.6] |
| wfr_lmap | wfr_rot | 1x | **+20.8%** [+12.9, +42.4] | +18.1% [+10.1, +28.2] |
| wfr_lmap | wfr_rot | 4x | **+52.8%** [+36.9, +69.8] | +34.6% [+23.7, +53.7] |
| wfr_lref | wfr_rot | 1x | -20.7% [-29.5, +17.5] | +1.5% [-15.3, +13.1] |
| wfr_lref | wfr_rot | 4x | **+506.9%** [+473.8, +577.4] | +140.1% [+132.3, +165.9] |
| wfr_qref | wfr_yref | 1x | **+102.7%** [+95.3, +108.1] | +80.6% [+70.2, +97.9] |
| wfr_qref | wfr_yref | 4x | **+99.1%** [+95.3, +104.7] | +98.0% [+88.8, +101.2] |
| wfr_lmh | wfr_qref | 1x | **-70.8%** [-73.3, -66.8] | -64.8% [-68.9, -61.0] |
| wfr_lmh | wfr_qref | 4x | **-74.4%** [-76.6, -72.8] | -71.7% [-73.1, -69.4] |
| wfr_lmh | ti_cold | 1x | **-62.3%** [-64.9, -54.6] | -65.3% [-67.4, -58.9] |
| wfr_lmh | ti_cold | 4x | **-47.1%** [-52.7, -35.6] | -58.2% [-61.6, -51.0] |
| wfr_lmh | abf | 1x | **-55.0%** [-61.3, -48.0] | -53.0% [-58.4, -48.0] |
| wfr_lmh | abf | 4x | **-31.4%** [-38.1, -25.3] | -46.7% [-49.9, -42.4] |
| wfr_lmh | ti_warm | 1x | **-43.5%** [-50.5, -26.1] | -39.1% [-49.2, -25.2] |
| wfr_lmh | ti_warm | 4x | **-25.1%** [-38.6, -19.2] | -36.5% [-46.9, -30.4] |
| wfr_yref | ti_cold | 1x | **-33.8%** [-43.5, -23.6] | -43.3% [-47.0, -38.3] |
| wfr_yref | ti_cold | 4x | +9.0% [-0.6, +20.6] | -20.7% [-28.4, -16.0] |
| wfr_yref | abf | 1x | **-23.3%** [-29.7, -18.5] | -25.5% [-28.7, -18.1] |
| wfr_yref | abf | 4x | **+33.0%** [+21.5, +47.0] | -8.3% [-11.8, +3.2] |

## Pentane: convergence rate

| arm | e_F at max budget | late-time d log e_F / d log fe | reading |
|---|---|---|---|
| wfr_ymh | 0.0196 | -0.113 | bias floor |
| wfr_lmh | 0.0215 | -0.133 | partly bias-limited |
| wfr_ymap | 0.0223 | -0.146 | partly bias-limited |
| ti_warm | 0.0299 | -0.249 | partly bias-limited |
| abf | 0.0314 | -0.416 | still converging |
| ti_cold | 0.0378 | -0.412 | still converging |
| w_only_y | 0.0415 | -0.037 | bias floor |
| wfr_yref | 0.0419 | -0.008 | bias floor |
| w_only | 0.0444 | -0.208 | partly bias-limited |
| w_count | 0.0448 | -0.290 | partly bias-limited |
| wfr_rot | 0.0475 | -0.183 | partly bias-limited |
| wfr_qref_uw | 0.0567 | -0.006 | bias floor |
| wfr_lmap | 0.0710 | -0.059 | bias floor |
| wfr_qref | 0.0835 | -0.007 | bias floor |
| wfr_shake | 0.1003 | -0.081 | bias floor |
| opes | 0.1728 | -0.135 | partly bias-limited |
| wfr_lref | 0.2917 | +1.612 | bias floor |
| wfr_flow_y | 1.2622 | -0.000 | bias floor |
| fr_only | 1.2622 | +0.000 | bias floor |
| wfr_flow | 1.2622 | +0.000 | bias floor |

## Alanine dipeptide: frozen hyper-parameters

| arm | frozen | screen I_F | screen e_F | screen D_cond |
|---|---|---|---|---|
| wfr_rot | kappa=1.2 theta=0.3 decay=0.999 - | 3.7977 | 4.3894 | 0.8665 |
| wfr_shake | kappa=0.075 theta=0.3 decay=0.999 - | 6.0185 | 7.1046 | 0.9992 |
| wfr_ymap | kappa=1.2 theta=0.3 decay=0.999 - | 1.7870 | 1.2114 | 0.3247 |
| wfr_yref | kappa=0.075 theta=0.3 decay=0.999 - | 14.4045 | 15.3433 | 4.1308 |
| wfr_ymh | kappa=1.2 theta=0.3 decay=0.999 - | 1.0371 | 0.5694 | 0.3026 |
| wfr_lmh | kappa=1.2 theta=0.3 decay=0.999 bz0.25 | 1.6327 | 0.7997 | 0.1800 |
| wfr_qref | kappa=0.075 theta=0.3 decay=0.999 - | 3.2665 | 1.6368 | 0.1571 |
| ti_cold | kappa=0.3 theta=0.3 decay=0.999 w64 | 3.1963 | 3.8935 | 1.0808 |
| ti_warm | kappa=0.3 theta=0.3 decay=0.999 w64 | 1.1207 | 0.8426 | 0.4822 |
| abf | kappa=0.3 theta=0.3 decay=0.999 n800 | 16.1595 | 13.7391 | 1.2027 |

10 confirmation runs -> results/mol/ala_confirm_spec.json

## Alanine dipeptide: confirmation, 16 fresh seeds (kJ/mol)

### ALA: free-energy error at three budgets (16 fresh seeds, median [IQR], kJ/mol)

| arm | 0.5x | 1x | 4x | D_cond (4x) | ESS_Fix | wall (s) |
|---|---|---|---|---|---|---|
| stratified constrained TI (cold) | 2.7730 [2.6788,3.3352] | 3.8356 [3.6926,4.1137] | 2.9799 [2.8680,3.1359] | 0.4289 | 0.974 | 560 |
| stratified constrained TI (warm start, oracle) | 0.8174 [0.7450,1.0892] | 0.7814 [0.7196,0.8894] | 0.6527 [0.6132,0.7168] | 0.3977 | 0.976 | 567 |
| ABF, multiple walkers | 16.5582 [15.5210,17.2193] | 14.3408 [13.1657,14.9457] | 8.7071 [7.9324,9.3695] | 1.0864 | 1.000 | 441 |
| RC-WFR, min-norm SHAKE lift | 5.3257 [4.8365,6.0937] | 7.2104 [6.7347,7.6144] | 3.3573 [2.7739,4.2063] | 0.4549 | 0.975 | 731 |
| RC-WFR, naive rotation lift | 3.5122 [2.6176,4.1271] | 4.1416 [3.3138,5.0747] | 3.3245 [2.8315,3.5216] | 0.3228 | 0.976 | 612 |
| RC-WFR + oracle y CDF-map | 1.6460 [1.5549,2.0401] | 1.2718 [1.0942,1.4816] | 0.6814 [0.6505,0.7213] | 0.2600 | 0.975 | 648 |
| RC-WFR + oracle y refresh | 16.2002 [16.0744,16.9633] | 15.1634 [14.9773,15.3551] | 15.2039 [15.1298,15.2729] | 4.0953 | 0.978 | 624 |
| RC-WFR + Metropolis y-move, oracle proposal | 0.6964 [0.6079,0.8877] | 0.6093 [0.5696,0.6849] | 0.5790 [0.5396,0.6059] | 0.2472 | 0.975 | 642 |
| RC-WFR + Metropolis y-move, LEARNED proposal | 1.0559 [0.7489,1.5208] | 0.6587 [0.6154,0.7621] | 0.5257 [0.5146,0.5567] | 0.1613 | 0.975 | 687 |
| RC-WFR + full conditional refresh (ceiling) | 3.9549 [3.8704,4.0193] | 2.1801 [1.7798,2.7959] | 7.3293 [6.7425,8.0335] | 0.1813 | 0.975 | 669 |

Estimator floor at this bandwidth: **0.1560** kJ/mol.
Force evaluations at 4x: 1.07e+08.

### Paired relative change vs each comparator (median, 95% bootstrap CI)

| arm | vs | budget | change in e_F | change in I_F |
|---|---|---|---|---|
| wfr_rot | wfr_shake | 1x | **-38.8%** [-54.5, -25.8] | -40.2% [-47.1, -16.4] |
| wfr_rot | wfr_shake | 4x | -5.3% [-21.3, +33.4] | -23.9% [-36.3, -13.0] |
| wfr_ymap | wfr_rot | 1x | **-69.7%** [-75.4, -59.6] | -48.4% [-59.1, -29.0] |
| wfr_ymap | wfr_rot | 4x | **-78.3%** [-80.4, -75.9] | -72.7% [-75.5, -63.8] |
| wfr_yref | wfr_rot | 1x | **+265.2%** [+199.4, +364.9] | +294.1% [+251.5, +416.2] |
| wfr_yref | wfr_rot | 4x | **+361.4%** [+336.3, +421.1] | +293.3% [+224.1, +346.1] |
| wfr_ymh | wfr_rot | 1x | **-84.0%** [-87.9, -79.8] | -73.3% [-78.5, -65.2] |
| wfr_ymh | wfr_rot | 4x | **-82.4%** [-84.1, -79.2] | -81.0% [-85.7, -77.8] |
| wfr_lmh | wfr_rot | 1x | **-81.4%** [-87.9, -78.0] | -62.1% [-75.7, -35.0] |
| wfr_lmh | wfr_rot | 4x | **-83.4%** [-84.9, -81.5] | -79.1% [-83.9, -75.3] |
| wfr_lmh | wfr_ymh | 1x | +8.1% [-4.6, +40.6] | +48.0% [+20.2, +105.7] |
| wfr_lmh | wfr_ymh | 4x | **-6.5%** [-10.5, -0.7] | +16.4% [+4.5, +30.1] |
| wfr_qref | wfr_yref | 1x | **-85.9%** [-88.3, -81.7] | -77.3% [-78.5, -76.2] |
| wfr_qref | wfr_yref | 4x | **-51.9%** [-55.2, -47.9] | -61.3% [-63.6, -59.6] |
| wfr_lmh | wfr_qref | 1x | **-66.6%** [-68.7, -63.4] | -56.7% [-67.5, -36.0] |
| wfr_lmh | wfr_qref | 4x | **-92.9%** [-93.1, -91.8] | -86.4% [-87.8, -83.9] |
| wfr_lmh | ti_cold | 1x | **-82.4%** [-83.9, -79.3] | -56.7% [-65.5, -31.8] |
| wfr_lmh | ti_cold | 4x | **-82.4%** [-83.3, -81.6] | -77.8% [-80.7, -75.5] |
| wfr_lmh | abf | 1x | **-95.3%** [-95.6, -94.4] | -92.0% [-93.1, -88.8] |
| wfr_lmh | abf | 4x | **-94.0%** [-94.4, -93.1] | -93.8% [-94.4, -92.4] |
| wfr_lmh | ti_warm | 1x | **-12.7%** [-20.4, -0.0] | +47.6% [+15.0, +148.8] |
| wfr_lmh | ti_warm | 4x | **-17.5%** [-22.4, -14.0] | +3.9% [-9.8, +26.2] |
| wfr_yref | ti_cold | 1x | **+289.1%** [+270.2, +312.9] | +377.7% [+332.8, +417.8] |
| wfr_yref | ti_cold | 4x | **+405.9%** [+384.3, +430.6] | +317.5% [+302.2, +336.8] |
| wfr_yref | abf | 1x | **+8.1%** [+0.9, +15.9] | -10.3% [-16.5, -3.5] |
| wfr_yref | abf | 4x | **+76.4%** [+61.4, +89.0] | +25.8% [+15.2, +33.3] |

## Alanine dipeptide: convergence rate

| arm | e_F at max budget | late-time d log e_F / d log fe | reading |
|---|---|---|---|
| wfr_lmh | 0.5257 | -0.115 | bias floor |
| wfr_ymh | 0.5790 | -0.008 | bias floor |
| ti_warm | 0.6527 | -0.154 | partly bias-limited |
| wfr_ymap | 0.6814 | -0.485 | still converging |
| ti_cold | 2.9799 | -0.266 | partly bias-limited |
| wfr_rot | 3.3245 | -0.200 | partly bias-limited |
| wfr_shake | 3.3573 | -0.588 | still converging |
| opes | 4.6904 | -0.462 | still converging |
| wfr_qref | 7.3293 | +0.479 | bias floor |
| abf | 8.7071 | -0.373 | still converging |
| wfr_yref | 15.2039 | +0.005 | bias floor |

## Hexane: which fiber mode has to be promoted?

| arm | promoted | e_F | I_F | D_cond(phi2) | D_cond(phi3) | accept |
|---|---|---|---|---|---|---|
| RC-WFR, naive rotation lift | none | 0.0414 | 0.0486 | nan | nan | 0.000 |
| RC-WFR + Metropolis y-move, oracle proposal | phi2 (adjacent, strongly coupled) | 0.0200 | 0.0267 | nan | nan | 0.863 |
| RC-WFR + Metropolis y-move, oracle proposal | phi3 (distal, weakly coupled) | 0.0485 | 0.0546 | nan | nan | 0.891 |
| RC-WFR + Metropolis y-move, oracle proposal | both | 0.0214 | 0.0244 | nan | nan | 0.840 |
| stratified constrained TI (cold) | - | 0.0469 | 0.0647 | nan | nan | 0.000 |
| ABF, multiple walkers | - | 0.0383 | 0.0750 | nan | nan | 0.000 |

## Switch campaign: does turning transport off restore the rate?

| arm | estimator | 1.1e+08 | 4.3e+08 | late slope |
|---|---|---|---|---|
| persistent RC-WFR | all deposits | 0.0228 | 0.0208 | -0.044 |
| ABF | all deposits | 0.0322 | 0.0227 | -0.231 |
| stratified TI, cold | all deposits | 0.0363 | 0.0284 | -0.253 |
| RC-WFR, naive lift | all deposits | 0.0448 | 0.0437 | -0.088 |
| OPES / ABP | all deposits | 0.1719 | 0.1571 | -0.056 |
| WFR->TI, frozen in place, @2e+04 steps (fe 7.3e+06) | all deposits | 0.0380 | 0.0411 | +0.050 |
| WFR->TI, frozen in place, @2e+04 steps (fe 7.3e+06) | **post-switch only** | 0.0387 | 0.0412 | +0.030 |
| WFR->TI, snapped + frozen proposal, @1e+05 steps (fe 2.9e+07) | all deposits | 0.0282 | 0.0221 | -0.220 |
| WFR->TI, snapped + frozen proposal, @1e+05 steps (fe 2.9e+07) | **post-switch only** | 0.0339 | 0.0220 | -0.318 |
| WFR->TI, snapped only, @1e+05 steps (fe 2.9e+07) | all deposits | 0.0282 | 0.0221 | -0.220 |
| WFR->TI, snapped only, @1e+05 steps (fe 2.9e+07) | **post-switch only** | 0.0339 | 0.0220 | -0.318 |
| WFR->TI, snapped + frozen proposal, @4e+05 steps (fe 1.2e+08) | all deposits | 0.0228 | 0.0212 | -0.081 |
| WFR->TI, snapped + frozen proposal, @4e+05 steps (fe 1.2e+08) | **post-switch only** | 0.0228 | 0.0232 | -0.484 |

### Paired change vs persistent RC-WFR (median, 95% bootstrap CI)

| switch | estimator | budget | change in e_F |
|---|---|---|---|
| frozen in place @2e+04 | all | 1.1e+08 | **+76.3%** [+39.2, +198.1] |
| frozen in place @2e+04 | all | 4.3e+08 | **+84.3%** [+40.5, +128.5] |
| frozen in place @2e+04 | **post-switch** | 1.1e+08 | **+77.4%** [+46.1, +213.0] |
| frozen in place @2e+04 | **post-switch** | 4.3e+08 | **+84.9%** [+44.6, +132.3] |
| snapped + frozen proposal @1e+05 | all | 1.1e+08 | **+32.9%** [+15.9, +47.4] |
| snapped + frozen proposal @1e+05 | all | 4.3e+08 | **+6.8%** [+4.1, +18.1] |
| snapped + frozen proposal @1e+05 | **post-switch** | 1.1e+08 | **+48.9%** [+31.2, +77.5] |
| snapped + frozen proposal @1e+05 | **post-switch** | 4.3e+08 | **+9.5%** [+2.7, +20.1] |
| snapped only @1e+05 | all | 1.1e+08 | **+32.9%** [+15.9, +47.4] |
| snapped only @1e+05 | all | 4.3e+08 | **+6.8%** [+4.1, +18.1] |
| snapped only @1e+05 | **post-switch** | 1.1e+08 | **+48.9%** [+31.2, +77.5] |
| snapped only @1e+05 | **post-switch** | 4.3e+08 | **+9.5%** [+2.7, +20.1] |
| snapped + frozen proposal @4e+05 | all | 1.1e+08 | +0.0% [-0.0, +0.0] |
| snapped + frozen proposal @4e+05 | all | 4.3e+08 | +1.9% [-9.6, +7.2] |
| snapped + frozen proposal @4e+05 | **post-switch** | 1.1e+08 | +0.0% [-0.0, +0.0] |
| snapped + frozen proposal @4e+05 | **post-switch** | 4.3e+08 | +13.3% [-3.0, +18.3] |

## Alanine, z = (phi, psi): the complete-coordinate control (kJ/mol)

| arm | e_F | I_F | curl | coverage | vs stratified TI |
|---|---|---|---|---|---|
| stratified constrained TI, cold | 0.254 | 0.385 | 0.052 | 1.000 | - |
| RC-WFR (W + Fisher-Rao) | 0.337 | 0.582 | 0.046 | 0.786 | **+31.5%** [+5.2, +61.6] |
| RC-WFR, W only | 0.561 | 3.878 | 0.082 | 0.727 | **+114.5%** [+85.0, +147.6] |
| RC-WFR -> TI, switch @5e4 | 0.354 | 0.588 | 0.067 | 0.794 | **+40.6%** [+25.6, +67.6] |
| ABF, multiple walkers | 43.603 | 43.710 | 0.508 | 0.083 | **+17188.6%** [+15480.0, +20138.8] |

## Slow-mode ranking predicted from the run's own statistics

```json
// PEN
{
 "system": "PEN",
 "S": [
  0.08284844931659573
 ],
 "tau": [
  85372.96969002955
 ],
 "damage": [
  603844564.339456
 ],
 "ranking": [
  1
 ]
}```

```json
// HEX
{
 "system": "HEX",
 "S": [
  0.06577524283279038,
  0.0006816549139595139
 ],
 "tau": [
  62272.49998571384,
  99619.86571509554
 ],
 "damage": [
  255067463.0104099,
  6764823.559490418
 ],
 "ranking": [
  1,
  2
 ]
}```

```json
// ALA
{
 "system": "ALA",
 "S": [
  1.455382112493327
 ],
 "tau": [
  22744.68676851795
 ],
 "damage": [
  752899404.0997263
 ],
 "ranking": [
  ```

