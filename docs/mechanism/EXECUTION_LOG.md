# Mechanism campaign: execution log and resource ledger

All times are UTC, 2026-10-10. Ceilings: 400 CPU core-hours and 8 GPU device-hours. Work is on `main`, per the
user's standing instruction.

## Resources at start (10:11)

* 256 cores; load average about 13–18; 1.5 TB RAM.
* GPUs (4 × H200 NVL): 0, 1 and 3 run other users' jobs (yesom, yifanchen), which are never touched. GPU 2 was
  idle.
* Disk 98 % full (78 GB free), so raw outputs are kept compact and nothing large goes into git.

## Timeline

| time | event |
|---|---|
| 10:11 | Resources checked. Existing docs, engine (`src/gateway_ladder_numba.py`), validation harness, references and the gateway gate read. |
| ~10:25 | The matched-free-energy identities verified numerically in scratch (`scratchpad/mech/derive_check.py`): E[f\|x] = F\*′ to 2e-15, the variance identity to 2e-15 relative, F constant offset to 2e-16, gradients vs FD 1e-10, for α = 1, 0.5, 0 and the shifted fibre. Confound found and preregistered: the shifted fibre's gate relaxation is 1000× slower than α = 1's, and its centre moves 4.9 σ across the gate. |
| 10:40 | **Preregistration committed and pushed (44735ea):** `docs/mechanism/SCIENTIFIC_PLAN.md` and `configs/mechanism/{matched_free_energy,conditional_relaxation,parallel_benchmark}.json`, before any engine code, validation or performance run. |
| 10:45 | Build workflow launched: engine `src/gateway_family_numba.py` + tests + `scripts/mechanism/run_cells.py`; validation harness `src/gateway_family_validation.py` + run/analyze scripts; analysis extension, cell configs, references and `cross_cell.py`. Each has multi-lens adversarial review and a fixer. |
| 10:50 | Experiment III feasibility: torch GPU engines exist for both systems (`src/gateway_core.py` with the histogram estimator; `src/lta/core_lta.py`). Planned primary backend: one GPU, one seed at a time, both arms. |

## Resource ledger

(filled as runs complete)
