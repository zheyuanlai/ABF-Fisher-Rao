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
| ~12:15 | Build workflow done: 13 agents. Engine bitwise = the original at α = 1, λ = 1 and 18/18 injected mutations caught; validation harness bitwise = the original at α = 1; the analysis extension leaves every equal-budget output byte-identical. **288 tests pass.** The review raised 4 preregistration ambiguities. |
| 12:20 | Implementation committed (c40a859). |
| 12:30 | **Amendment 1 committed and pushed (bf56da4)** before any gate, smoke or production run: V5 reads the final Γ profile; V1 uses the plan reading (strict clause reported); V4 Holm family; α = 1 / λ = 1 re-run with a 192-job bitwise comparison; cost update. |
| 12:52 | **Validation gate launched** at h = 2.5e-5 for all 7 dynamics: 245 jobs, about 57 core-h, 120 workers on cores 136–255. |
| ~13:00 | Amendment A3 (Holm over the V4 family, t₁₅ p) implemented in `analyze_validation.py`; unit test added (57 tests pass). |
| 13:05 | 2-seed smoke runs of every cell (Step 4): 72 jobs, 0 failed, 1.7 min. Not interpreted for performance. V5 ABF-bias record (`scripts/mechanism/abf_smoke.py`): all 7 dynamics PASS. The final Γ is ≤ 7.9 against a limit of 15.8; the running maximum reaches 14.4 (λ = 0.25) and 30.3 (λ = 0.1), as Amendment A1 anticipated. D1–D4 saved, and the class sums equal C_all exactly. |
| 13:10 | Analysis pipeline (`analyze_ladder`, `plot_config`) runs end to end on all 8 smoke cells: 19 figures per N, legibility passes. `scripts/mechanism/compare_alpha1_bitwise.py` written (Amendment A4). |
| 13:15 | Second build workflow launched: mechanism analysis (D1–D4, conditional laws, synthesis figures S1–S6) and the Experiment III benchmark harness (built and smoked only). |

## Resource ledger

(filled as runs complete)
