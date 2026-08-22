# Provenance

This repository tests a NEW hypothesis and has an independent git history.  The two
closed campaigns `zheyuanlai/ABF-Fisher-Rao` (`main`, mFR-on-ABF) and its
`abp-fisher-rao` branch (SHUS/ABP+FR) are treated as validated code donors and as a
source of prior, not as parents.

## Ported numerics (conventions preserved exactly)

| here                          | from                                    |
|-------------------------------|-----------------------------------------|
| `src/rcwfr/grid.py`           | `ABP-Fisher-Rao/src/abpfr/grid.py` (+ periodic BC added) |
| `src/rcwfr/resampling.py`     | `ABP-Fisher-Rao/src/abpfr/resampling.py` |
| `src/rcwfr/fisher_rao.py`     | `ABP-Fisher-Rao/src/abpfr/fisher_rao.py` (theta backoff, ESS guard) |
| `src/rcwfr/shus.py`           | `ABP-Fisher-Rao/src/abpfr/shus.py` |
| metrics `e_F`, `I_F`, `tau_eps`, paired bootstrap | `ABP-Fisher-Rao/src/abpfr/metrics.py` |
| Gaussian-kernel / reflect-pad smoothing conventions | `ABF-Fisher-Rao/src/eb_abffr_core.py` |
| entropic-bottleneck system `EB`  | `ABF-Fisher-Rao/src/eb_abffr_core.py` |

Written fresh here: the WFR engines (`engines.py`), the Wasserstein step
(`wasserstein.py`), the shared TI mean-force estimator (`estimators.py`), the
generalized system family (`systems/base.py`), stratified TI and Hamiltonian
replica-exchange TI arms, and the row-batched sweep layout (`rowspec.py`).

## Prior carried over from the donor campaigns

* A uniform-target Fisher-Rao step computed from a histogram density IS count
  balancing.  Re-tested here as a unit test (r > 0.99) and confirmed again in the
  bias-free setting (Finding H4).
* A selection step must not outrun the rate at which clones decorrelate.
* Any comparison must be scored against a measured ESTIMATOR FLOOR; differences at
  the floor are not evidence.  This caught a false positive in the first smoke run
  of this campaign (see docs/RESULTS_LOG.md).
* Baselines must be screened at least as hard as the new arm.
