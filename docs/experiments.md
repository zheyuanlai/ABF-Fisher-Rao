# Experiment log (NSCC ASPIRE2A)

Job ids, what changed, outcome. Newest last. Study designs live in their own docs; this is the running record.

## 2026-10-27 — C60/water reconsidered, shelved; finite-loading ethane/LTA opened

* C60/water (`docs/SPEC_c60_water.md`): the archived torch engine would need ~10 A100-days for the reference alone
  and the frozen protocol weeks in total; the conda env `envs/abffr` had been purged by scratch (631/690 OpenMM files
  gone). An OpenMM-CUDA port was started (`src/c60/omm.py`, unvalidated, branch `c60-openmm-wip`) and stopped when the
  user chose finite-loading LTA instead. Found while reading: `src/c60/dynamics.py` xi mode adds `+dV/dxi` as the
  driving force (sign error; that path never ran in production).
* Finite-loading ethane/LTA: design `docs/lta_loading/PREREGISTRATION.md`, engine `src/lta_loading/engine.py`,
  env `scripts/lta_loading/setup_env.sh` (`envs/ltafl`).
* Env: conda-forge on GPFS stalled (> 35 min extracting); rebuilt with uv + pip wheels in 3 min
  (`envs/ltafl`: Python 3.12, numpy 2.5.4, numba 0.68.0, cupy 14.2.0; CUDA 12.8.1 system toolkit).
* g1 is occupied by another project's job (one running job per queue), so this study uses g2
  (`scripts/lta_loading/pbs/gpu2_task.pbs`, two task streams). Smoke job 26028686 (V1-V3, V6) queued 21:4x.
