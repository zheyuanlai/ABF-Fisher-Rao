# WCA confirmatory (dt 0.000125, N 256, T 240): ABF vs ABF+FR vs matched sham, exact (MC) reference

seeds 5100-5115 (16); verdict (frozen rule): **FR_HARMFUL**

| arm | median I_F | median e_F(T) | median own e_F' | replacements | min windowed ESS/N | max family share | round trips / replica |
|---|---|---|---|---|---|---|---|
| hist_abf | 13.9448 | 0.0095 | 0.136 | 0 | nan | nan | 2519.98 |
| hist_fr_uniform | 15.3678 | 0.0179 | 0.150 | 677 | 0.755 | 0.0684 | 2580.61 |
| hist_fr_sham | 14.6531 | 0.0096 | 0.136 | 677 | 0.764 | 0.0742 | 2524.03 |

| contrast | I_F | e_F(T) | e_F' |
|---|---|---|---|
| hist_fr_uniform vs hist_abf | +9.9 % [+2.3, +13.2] (4/16) | +131.0 % [+93.3, +194.0] (1/16) | +10.5 % [+7.0, +16.5] (2/16) |
| hist_fr_sham vs hist_abf | -0.8 % [-4.6, +5.0] (8/16) | -8.4 % [-39.8, +54.6] (8/16) | +1.0 % [-3.2, +7.4] (7/16) |
| hist_fr_uniform vs hist_fr_sham | +8.7 % [+1.1, +14.7] (3/16) | +113.2 % [+33.7, +240.6] (2/16) | +10.7 % [+5.8, +13.7] (2/16) |

pooled long-run limits (D vs exact, kT): hist_abf 0.0038, hist_fr_uniform 0.0198, hist_fr_sham 0.0000
FR tilt D(FR limit, ABF limit) = 0.0169 [up 0.0212]; sham vs ABF 0.0047
FR+sham process wall per arm vs ABF process: +4.0 %
