# Ace-Val-Nme -- gate ledger

Run `abf__both__N2048__T300000__ns16__2d231e761855.npz`, 16 seeds x 2048 walkers, 300 ps. Recomputed by `scripts/close_valine.py` with the corrected diagnostics; no new dynamics.

## Support conditioning

The establishment target and the observed populations must live on the same domain, or every region reads deficient by a common factor that has nothing to do with sampling.

| check | value | requirement |
|---|---|---|
| max&nbsp;\|sum_k Q*_k(t) - 1\| | 4.44e-16 | = 0 (asserted) |
| max&nbsp;\|sum_k P_k(t) - 1\| | 2.22e-16 | = 0 (asserted) |
| walkers inside a labelled region | 0.3712 | excluded from both sides |

## Gates

| gate | threshold | measured | margin | verdict |
|---|---|---|---|---|
| V3.1 every region discovered | T_hit < 10% of run (30 ps), all seeds | worst 18.0 ps (worst first touch 5.4 ps) | 1.7x faster | PASS |
| V3.2 some region under-established | >=1 region below 50% of target for >=20% of run | worst region below half for 0.060 of run | 3.4x under threshold | **FAIL** |
| V3.4 omitted psi conditional | worst-region TV < 0.15 | 0.055 (matched cells) | below | PASS |
| accuracy of ABF's own F | -- | 0.248 kT RMSE (seeds 0.237-0.257), marginal TV 0.069 | -- | context |

V3.2 is the decisive gate and it **fails**: there is no discovered-but-under-established region, so mFR has nothing to repair.

## Per region

Worst case over seeds, not the median: a per-region median is a number no single seed has to satisfy.

| region | pilot pop | first touch (ps) | worst T_hit (ps) | worst T_est (ps) | occ/target | max rel deficit (2nd half) | below-half frac | entries (corridor-aware / naive) |
|---|---|---|---|---|---|---|---|---|
| B0 | 0.4201 | 4.2 | 6.0 | 48.0 | 0.94 | 0.25 | 0.040 | 4192 / 35208 |
| B1 | 0.1898 | 4.7 | 6.0 | 32.0 | 0.92 | 0.33 | 0.020 | 1556 / 68461 |
| B2 | 0.1840 | 4.8 | 6.0 | 50.0 | 1.08 | 0.19 | 0.046 | 2184 / 101896 |
| B3 | 0.0900 | 5.0 | 6.0 | 40.0 | 1.09 | 0.22 | 0.033 | 1655 / 0 |
| B4 | 0.0701 | 5.0 | 6.0 | 52.0 | 1.14 | 0.21 | 0.033 | 1325 / 73350 |
| B5 | 0.0380 | 4.6 | 6.0 | 46.0 | 0.89 | 0.39 | 0.040 | 849 / 71656 |
| B6 | 0.0055 | 5.3 | 18.0 | 44.0 | 0.83 | 0.56 | 0.060 | 704 / 0 |
| B7 | 0.0014 | 5.4 | 6.0 | 30.0 | 1.46 | 0.17 | 0.020 | 527 / 0 |

The naive consecutive-frame counter reports **zero** entries into B3, B6, B7 -- the regions reachable only across the unlabelled corridor above the region ceiling. The corridor-aware counter credits them, which is consistent with their finite T_hit. Regions still at zero after the correction: none.

## The omitted-coordinate check, before and after

The original check compared `p_ABF(psi | region)` against `p_pilot(psi | region)`. ABF flattens *within* a region while the pilot is Boltzmann-weighted inside it, so the two weight the region's interior differently and the statistic is non-zero even when the psi conditional at every fixed (phi, chi1) cell agrees exactly. Comparing cell by cell and aggregating with common weights removes that.

Worst-region TV: **0.280** as originally computed, **0.055** at matched cells -- against a 0.15 threshold. The condition now passes, and the failure it used to report was the confound, not the omitted coordinate.

| region | TV, matched cells | TV, as originally computed | dropped weight |
|---|---|---|---|
| B0 | 0.035 | 0.245 | 0.036 |
| B1 | 0.018 | 0.280 | 0.014 |
| B2 | 0.052 | 0.037 | 0.013 |
| B3 | 0.012 | 0.052 | 0.000 |
| B4 | 0.055 | 0.043 | 0.031 |
| B5 | 0.020 | 0.152 | 0.008 |
| B6 | 0.012 | 0.034 | 0.000 |
| B7 | 0.021 | 0.023 | 0.000 |

Common weights: bias-aware cell occupancy (model-based; run did not record per-walker CV cells). `dropped weight` is the share of the region's common weight sitting in cells with too little reference information to supply a conditional; it is reported rather than silently redistributed.
