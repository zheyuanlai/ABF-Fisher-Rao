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

## Molecular phase

| what | where from |
|---|---|
| `src/rcwfr/mol/alanine.py`: `NAMES`, `BONDS`, `build_positions` (NeRF reference-minimum builder), the OpenMM topology construction and the `extract_parameters` shape | the author's earlier closed alanine campaign (`.c60-run-cc12bb9/src/alanine/`), where the torch re-implementation was validated against OpenMM to 1e-9 relative on energy and 3e-10 on forces |
| everything else in `src/rcwfr/mol/` | written for this campaign |

The alanine parity check is re-run here rather than inherited:
`scripts/mol_ala_gate.py` reports max relative error 1.19e-9 on energy and
1.67e-10 on forces over 24 thermally displaced configurations spanning
E in [1383, 5938] kJ/mol, against OpenMM's Reference platform.

Two deliberate departures from that earlier campaign:

* **uniform masses (12 amu).** Masses do not enter `e^{-beta V}`, so `F(z)` is
  exactly unchanged and the mean force becomes mass-free
  (`w = grad xi / |grad xi|^2`).  What changes is the Brownian kinetics: the
  X-H bonds no longer force `h <~ 4e-7`, and alanine's torsional diffusion per
  step lands within 11% of pentane's, so the two systems share a budget scale.
* **Brownian, not BAOAB.** The whole package is overdamped; the constrained
  sampler is projected Euler-Maruyama with SHAKE, as in the toy phase.
