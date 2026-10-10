# Mechanism campaign: execution log and resource ledger

All times are UTC, 2026-10-10. Ceilings: 400 CPU core-hours and 8 GPU device-hours. Work is on `main`, per the
user's standing instruction.

## Resources at start (10:11)

* 256 cores; load average about 13–18; 1.5 TB RAM.
* GPUs (4 × H200 NVL): 0, 1 and 3 run other users' jobs (yesom, yifanchen), which are never touched. GPU 2 was
  idle.
* Disk 98 % full (78 GB free), so raw outputs are kept compact and nothing large goes into git.

## Timeline

Times are from git commit times and file modification times, except entries marked "~". Before 12:00 the
first draft of this table carried estimated times up to 60 minutes late; they were corrected from these sources.

| time | event |
|---|---|
| 10:11 | Resources checked. Existing docs, engine (`src/gateway_ladder_numba.py`), validation harness, references and the gateway gate read. |
| ~10:13 | The matched-free-energy identities verified numerically in scratch (`scratchpad/mech/derive_check.py`): E[f\|x] = F\*′ to 2e-15, the variance identity to 2e-15 relative, F constant offset to 2e-16, gradients vs FD 1e-10, for α = 1, 0.5, 0 and the shifted fibre. Confound found and preregistered: the shifted fibre's gate relaxation is 1000× slower than α = 1's, and its centre moves 4.9 σ across the gate. |
| 10:16 | **Preregistration committed and pushed (44735ea):** `docs/mechanism/SCIENTIFIC_PLAN.md` and `configs/mechanism/{matched_free_energy,conditional_relaxation,parallel_benchmark}.json`, before any engine code, validation or performance run. |
| ~10:20 | Build workflow launched: engine `src/gateway_family_numba.py` + tests + `scripts/mechanism/run_cells.py`; validation harness `src/gateway_family_validation.py` + run/analyze scripts; analysis extension, cell configs, references and `cross_cell.py`. Each has multi-lens adversarial review and a fixer. |
| ~10:22 | Experiment III feasibility: torch GPU engines exist for both systems (`src/gateway_core.py` with the histogram estimator; `src/lta/core_lta.py`). Planned primary backend: one GPU, one seed at a time, both arms. |
| ~11:48 | Build workflow done: 13 agents. Engine bitwise = the original at α = 1, λ = 1 and 18/18 injected mutations caught; validation harness bitwise = the original at α = 1; the analysis extension leaves every equal-budget output byte-identical. **288 tests pass.** The review raised 4 preregistration ambiguities. |
| 11:51 | Implementation committed (c40a859). |
| 11:51 | **Amendment 1 committed and pushed (bf56da4)** before any gate, smoke or production run: V5 reads the final Γ profile; V1 uses the plan reading (strict clause reported); V4 Holm family; α = 1 / λ = 1 re-run with a 192-job bitwise comparison; cost update. |
| 11:52 | **Validation gate launched** at h = 2.5e-5 for all 7 dynamics: 245 jobs, about 57 core-h, 120 workers on cores 136–255. |
| ~11:53 | Amendment A3 (Holm over the V4 family, t₁₅ p) implemented in `analyze_validation.py`; unit test added (57 tests pass). |
| 11:54–11:56 | 2-seed smoke runs of every cell (Step 4): 72 jobs, 0 failed, 1.7 min. Not interpreted for performance. V5 ABF-bias record (`scripts/mechanism/abf_smoke.py`): all 7 dynamics PASS. The final Γ is ≤ 7.9 against a limit of 15.8; the running maximum reaches 14.4 (λ = 0.25) and 30.3 (λ = 0.1), as Amendment A1 anticipated. D1–D4 saved, and the class sums equal C_all exactly. |
| 11:57–11:59 | Analysis pipeline (`analyze_ladder`, `plot_config`) runs end to end on all 8 smoke cells: 19 figures per N, legibility passes. `scripts/mechanism/compare_alpha1_bitwise.py` written (Amendment A4). |
| ~12:00 | Second build workflow launched: mechanism analysis (D1–D4, conditional laws, synthesis figures S1–S6) and the Experiment III benchmark harness (built and smoked only). |
| ~12:35 | **Second build workflow interrupted:** both builder agents hit the session's usage limit and stopped mid-work. They left partial `scripts/mechanism/mech_analysis.py`, the benchmark harness files and their tests, all unreviewed. |
| 13:26 | Validation runs complete: 245 jobs, 0 failed, 57 core-h. |
| 13:27 | **Gate analysis: PASS at h = 2.5e-5 for all 7 dynamics** (plan and strict readings agree; Holm V4: 0 of 32 rejected). |
| 13:28 | **Production launched:** Experiments I and II, 1 152 simulated runs. Includes the α = 1 rerun of Amendment A4; the 128 λ = 1 jobs are aliases of identical α = 1 runs. *(This entry and the 13:35 status message first said "1 024 runs", a miscount corrected from the ledger.)* 120 workers on cores 128–255. |
| 13:31 | Gate report `EXP1_TIMESTEP_VALIDATION.md` + figure committed and pushed (bd0c3bd). |
| ~13:33 | Interrupted build relaunched as a continuation workflow: inspect, finish and verify the partial files, then two-lens review and a fixer. |
| 14:18 | **Production complete:** 1 152 simulated runs (768 Experiment I incl. the 192 α = 1 re-runs, plus 384 Experiment II) and 128 λ = 1 aliases, 0 failed, 97.2 core-h. |
| 14:19 | **Amendment A4 check: PASS, 192/192** α = 1 re-runs bitwise equal to their equal-budget counterparts on all shared arrays (`results/mechanism/reuse_gate/alpha1_bitwise.json`). |
| 14:19–14:22 | Analysis of all 8 cells (`analyze_ladder --system gateway_family`), preregistered contrasts (`cross_cell.py` → `results/mechanism/synthesis/cross_cell.{json,md}`), per-N figures A–F (20 cell × N sets), per-cell S1–S8. **Completeness audit PASS for all 8 cells.** |
| 14:22 | Production milestone committed and pushed (2ccf40d). |

## Resource ledger

(filled as runs complete)
