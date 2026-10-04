# WCA dimer replica ladder at equal force-evaluation budget -- scoreboard, dt 0.0005 (dynamics-consistent reference)

B = 491,520,000 replica-steps per arm (N x n_steps), histogram ABF (160 bins) vs + uniform FR, 16 paired seeds per N, float64, cap max(1, floor(0.02 N)), every time knob in steps. Ibar_F = (1/B) int_0^B e_F db (mean over 200 budget fractions). eps = 0.0036 = median ABF e_F(B) at N = 1024.

| N | T | ABF Ibar_F | FR Ibar_F | dIbar_F | wins | ABF e_F(B) | FR e_F(B) | d e_F(B) | wins | d e_F'(B) | ABF u_eps | FR u_eps | FR replacements | FR min windowed ESS | cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 240 | 0.0462 | 0.0516 | +12.0 % [+9.1, +19.7] | 1/16 | 0.0036 | 0.0168 | +376.9 % [+227.9, +634.9] | 0/16 | +15.9 % | inf | inf | 2618 | 810.964 | 20 |
| 256 | 960 | 0.0150 | 0.0288 | +86.2 % [+68.5, +118.5] | 0/16 | 0.0032 | 0.0211 | +562.8 % [+412.7, +647.6] | 0/16 | +19.4 % | 0.948 | inf | 3120 | 185.656 | 5 |
| 64 | 3840 | 0.0078 | 0.0210 | +160.7 % [+114.8, +187.1] | 0/16 | 0.0044 | 0.0192 | +357.8 % [+214.6, +636.6] | 0/16 | +19.6 % | inf | inf | 3644 | 37.926 | 1 |
| 16 | 15360 | 0.0072 | 0.0133 | +92.9 % [+57.4, +138.5] | 2/16 | 0.0036 | 0.0128 | +230.8 % [+136.0, +504.7] | 0/16 | +14.0 % | inf | inf | 4392 | 5.449 | 1 |
| 4 | 61440 | 0.0083 | 0.0078 | -13.0 % [-36.2, +36.0] | 10/16 | 0.0041 | 0.0047 | +6.2 % [-29.4, +37.4] | 8/16 | +4.8 % | inf | inf | 3107 | 1.000 | 1 |
| 1 | 245760 | 0.0077 | -- | -- | -- | 0.0032 | -- | -- | -- | -- | 0.920 | -- | -- | -- | -- |