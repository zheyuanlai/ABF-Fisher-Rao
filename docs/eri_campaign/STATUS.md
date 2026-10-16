# ERI campaign — STATUS

*Ethane in all-silica ERI: does histogram ABF + marginal Fisher–Rao birth–death beat histogram ABF?*
Started 2026-10-16. One coordinator session owns the source tree and every experiment decision.

## Current stage (2026-10-16 ~17:00 UTC)

**Stage 5, ABF-only regime screen of the dilute cell.** The data exist: K 64/256/1024 × 8 seeds. They have **not
been classified yet**: the adversarial audit (`audit/AUDIT1_*.md`) must finish first, so that any amendment to the
classifier stays blind to the screen.

| stage | status |
|---|---|
| 1. repo review + literature extraction | **done** (`review/`) |
| 2. engine + Gate 0 (forces, geometry, CV) | **done**: 16 tests pass (`tests/test_eri_model.py`) |
| 3. Gate 1 independent reference (exact MC umbrella/WHAM) | **PASS** on all six checks (`results/eri_campaign/reference/gard_tait/gate1.json`) |
| 4. Gate 2 timestep ladder | **all candidates admissible**; frozen dt 2e-4 (EM). FR-blind second clause passes (Amendment 1) |
| 4b. Gate 3 histogram ABF calibration (ABF only) | **frozen** (Amendment 1): 180 bins, trust 200, ramp 1 t.u., pre-eq 5 t.u., T 40 t.u. |
| 4c. Gate 4 FR mechanism tests | **done**: 7 tests pass (`tests/test_eri_engine.py`) |
| 5. ABF-only regime screen + classification | screen run; classification waits for the audit |
| 6. FR confirmation (eligible cells only, two arms) | pending the classification |
| 7. equal-budget ladder | only if 6 is licensed |
| 8. loadings | after the dilute cell. Expected to need a GPU engine (CPU cost about 6 core-h per N = 256 run at 2/cage) |

## Accepted model and parameters

* **Model:** "PNAS-2005-type ERI model", a labelled surrogate of Dubbeldam et al., PNAS 102, 12317 (2005).
* **Framework:** Gard & Tait 1973 (RASPA2 `ERI_SI.cif`). It is the cited structure and was selected by the frozen
  literature rule: single-bead barrier 5.96 vs the published 6.13 kBT; Schlenker 5.29 is out of the band.
* **Force field:**
  * LJ pairs from Schüring et al. 2002: O–CH3 142 K / 3.17 Å, Si–CH3 82 K / 2.12 Å, CH3–CH3 104 K / 3.78 Å;
    12 Å energy-shifted.
  * Harmonic bond 96 500 K/Å², r0 1.54 Å.
* **Box:** 39.81 × 45.97 × 45.15 Å (3a × 2√3a × 3c), 72 ERI cages.
* **T** = 600 K.
* **CV:** the tagged COM along the hexagonal a1 axis, period a/2 = 6.635 Å. Windows sit at φ = 0 and cage rows at
  φ = ±π. The CV is linear, so there is no geometric term.
* **Reference:** barrier 33.28 kJ/mol = 6.67 kBT. ΔU = 5.30 and −TΔS = 27.97 kJ/mol, so the barrier is **84 %
  entropic**. Reference noise 0.007 kJ/mol.
* **Dynamics:** overdamped EM, unit mobility, dt 2e-4 t.u.
  * Bond variance +19 % (first order).
  * D_MF vs exact MC = 0.037 kJ/mol [upper 0.059].

## Standing decisions

* **Branch:** `main` (standing user preference recorded 2026-10-10). Nothing existing is overwritten: new artefacts
  only, under `docs/eri_campaign/`, `src/eri/`, `configs/eri_campaign/`, `scripts/eri_campaign/`,
  `tests/test_eri_*.py`, `results/eri_campaign/`, `cache/eri/` and `figures/eri_campaign/`. `docs/` is git-ignored,
  so deliverables are force-added. Raw `*.npz` stay local.
* **Two arms only (user instruction, 2026-10-16):** histogram ABF vs histogram ABF + FR. **No sham arm**, ever.
* **Compute:**
  * GPUs 0 and 1 carry another user's jobs and are never touched. GPU 3 is the project device; it is idle and has
    not been used yet.
  * CPU: at most 32 concurrent single-threaded processes.
  * Ceilings: ≤ 24 GPU-h and ≤ 200 CPU core-h.

## Resource ledger (CPU core-h; GPU h)

| item | core-h | GPU h |
|---|---|---|
| review agents, tests, smokes | ~1.0 | 0 |
| Gate 1 MC (both frameworks g0–3; Gard & Tait g4–15) | 4.08 | 0 |
| Gate 2 ladder (EM 2e-4/1e-4/5e-5, LM 2e-4) | 7.74 | 0 |
| reference analyses (2000-bootstrap) | ~1.3 | 0 |
| ABF-only pilot + calibrations (warm-up, bins) | 3.25 | 0 |
| ABF-only dilute screen | 1.51 | 0 |
| **total** | **~18.9** | **0** |

## Next automatic action

When the audit returns: fix any confirmed defect (re-running what it touches), then run
`scripts/eri_campaign/classify_screen.py results/eri_campaign/screen_dilute`. After that:
* if a cell is ELIGIBLE: freeze the FR safety calibration and the seed blocks (Amendment 2), then run them;
* else: record the dilute cell as ABF-sufficient and start the loading study.

## How to resume

Read this file, `PREREGISTRATION.md` (with its amendments) and the latest entries of `EXECUTION_LOG.md`. Run the
tests:

```
NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" /home/zheyuanlai/miniconda3/envs/abffr/bin/python -m pytest -q tests/test_eri_model.py tests/test_eri_engine.py
```

Job lists live in `configs/eri_campaign/jobs/*.json`. Run them with `scripts/eri_campaign/run_jobs.py`, which is
resumable and checkpointed.
