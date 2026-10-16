# ERI campaign — STATUS

*Ethane in all-silica ERI: does histogram ABF + marginal Fisher–Rao birth–death beat histogram ABF?*
Started 2026-10-16. Coordinator: one Claude Code session (owns the source tree and every experiment decision).

## Current stage

**Stage 1 — repository review and literature extraction (running).**

| stage | status |
|---|---|
| 1. repo review + literature extraction (ERI model, force field, structure) | running |
| 2. engine + Gate 0 (forces, geometry, CV) | pending |
| 3. Gate 1 independent reference (umbrella/WHAM + Metropolis MC) | pending |
| 4. Gate 2 timestep ladder, Gate 3 histogram-ABF calibration | pending |
| 5. ABF-only regime screen (K ladder) + classification | pending |
| 6. FR/sham confirmation (only for eligible cells) | pending |
| 7. equal-budget ladder (only if 6 is licensed) | pending |
| 8. loadings (only after the dilute case validates) | pending |

## Standing decisions

* **Branch:** `main` (standing user preference recorded 2026-10-10: campaigns commit on `main`; no side branches).
  Nothing existing is overwritten: every new artefact lives under `docs/eri_campaign/`, `src/eri/`,
  `configs/eri_campaign/`, `scripts/eri_campaign/`, `tests/test_eri_*.py`, `results/eri_campaign/`, `cache/eri/`.
  `docs/` is git-ignored, so deliverables are added with `git add -f`.
* **Compute:** GPUs 0 and 1 carry another user's (`yesom`) jobs and are never touched. The project's standing
  device is GPU 3 (`CUDA_VISIBLE_DEVICES=3`, one process). CPU work uses a bounded number of cores on the shared
  256-core host. Ceilings for this campaign: ≤ 24 GPU device-hours, ≤ 200 CPU core-hours.

* **Two arms only (user instruction, 2026-10-16):** every comparison is histogram ABF vs histogram ABF + Fisher–Rao.
  **No sham / matched-turnover arm is designed, run or reported**; this overrides §3.3 and §6 of the campaign prompt.
  Attribution rests on the direct FR-vs-ABF contrast and on mechanism diagnostics (event fractions, genealogy,
  marginal establishment, conditional relaxation).

## Resource ledger

| date (UTC) | item | CPU core-h | GPU h |
|---|---|---|---|
| 2026-10-16 | review agents (read-only, small pytest) | < 0.5 | 0 |
| **total** | | **< 0.5** | **0** |

## Next automatic action

Reconcile the literature extraction (`review/LIT_CROSSCHECK.md`), decide exact vs surrogate status of the model,
then write `src/eri/` and the Gate 0 tests.

## How to resume

Read this file, then `docs/eri_campaign/review/*.md`, then the latest section of `docs/eri_campaign/EXECUTION_LOG.md`.
