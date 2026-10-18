# FR-rate calibration (rng 20261102, 4 seeds): turnover and genealogy only

Rule (frozen): secondary rate = the SMALLEST rate in {0.05, 0.10} that is safe and active; if none is active, the LARGEST safe rate in {0.05, 0.10}; if none is safe, no secondary rate. Free-energy errors of the calibration runs are NOT computed before the rate is fixed.

| rate | dt | replacements/N (median) | final ESS/N | final max ancestor frac | min ESS/N (any time) | score std | safe | active |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.00025 | 0.106 | 0.776 | 0.0088 | 0.731 | 0.418 | True | False |
| 0.05 | 0.00025 | 0.284 | 0.370 | 0.0244 | 0.272 | 0.498 | False | False |
| 0.1 | 0.00025 | 0.633 | 0.073 | 0.0518 | 0.057 | 0.548 | False | False |

**Secondary rate: None** (no safe rate: no secondary rate).
