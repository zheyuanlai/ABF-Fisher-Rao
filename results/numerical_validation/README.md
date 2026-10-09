# results/numerical_validation (2026-10-09/10)

Overnight numerical-validation campaign; reports in `docs/numerical_validation/` (start with `OVERNIGHT_REPORT.md`).
Raw `*.npz` are local only (gitignored); summaries, logs and figures are versioned.

| directory | what | produced by | prereg |
|---|---|---|---|
| `wca/em_impl/` | production unbiased EM (wca_numba path), dt 0.002 ... 0.000125, 64 seeds each (with the 2026-10-04 runs in `results/wca_replica_ladder/consistency/raw`) | `scripts/numerical_validation/run_wca_validation.py` | `configs/numerical_validation/wca_validation_prereg.json` |
| `wca/mc/`, `wca/mala/` | exact Gibbs arbiters (Metropolis MC 64 x 3e7 sweeps; MALA dt 2.5e-4) | same | same |
| `wca/lm/`, `wca/baoab/` | Leimkuhler-Matthews and BAOAB on the intended (unclipped) force | same | same |
| `wca/diag/` | production drift with every-step regularisation counters and energies | same | same |
| `wca/summary.json`, `wca/analysis.log`, `wca/figures/` | gates, profiles, secondary observables, sanity and convergence figures | `analyze_wca_validation.py` | |
| `wca/reference_audit.json`, `wca/rescore_history.json` | every historical reference/limit vs exact; historical runs rescored | `analyze_wca_reference_audit.py`, `rescore_wca_history.py` | |
| `wca/force_audit.json`, `wca/nve_check.json` | force-audit numbers; velocity-Verlet energy check | `wca_force_audit_numbers.py`, `wca_nve_check.py` | |
| `lta/T300/`, `lta/T150/` | umbrella MC / EM (3 dt) / LM, 16 groups x 40 windows; summary + rescoring of the published runs | `run_lta_validation.py`, `analyze_lta_validation.py` | `lta_validation_prereg.json` |
| `gateway_dt/` | accepted gateway cell at dt 4e-4 / 1e-4 / 2.5e-5 | `run_gateway_dt.py`, `analyze_gateway_dt.py` | `gateway_dt_prereg.json` |
| `wca_confirmatory/` | ABF / FR / matched sham at the validated dt 0.000125, N 256, T 240 | `run_wca_confirmatory.py`, `analyze_wca_confirmatory.py` | `wca_confirmatory_prereg.json` |
| `compute_accounting.json` | core-hours per group (CPU only; 0 GPU-h) | `compute_accounting.py` | |
