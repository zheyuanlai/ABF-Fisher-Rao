# Experiment log (NSCC ASPIRE 2A)

Job ids, what changed, outcome.  Keep this current: scratch is purged and the chat is not a record.

## 2026-10-29 — deca-alanine ABF vs ABF+FR re-do (branch `deca-fr-2026`)

Protocol: `docs/DECA_FR_2026_PROTOCOL.md`.  Env: `scripts/deca_fr_2026/setup_env.sh` -> `/scratch/users/nus/e1297740/envs/deca26`.
Submit: `qsub -P personal-e1297740 -l walltime=HH:MM:SS -v TASK=scripts/deca_fr_2026/tasks/<t>.sh scripts/deca_fr_2026/pbs/gpu_task.pbs`.
Logs: `/scratch/users/nus/e1297740/logs/<task>_<jobid>.log`.  Budget at start: 28 307 SU (64 SU per GPU-hour).

| date | job | task | queue | outcome |
|---|---|---|---|---|
| 2026-10-29 | 26048649 | smoke.sh | gdev | FAILED after 2 min: CPU tests 58 pass / 1 skip, then `torch.compile` -> "Cannot find a working triton installation" (uv venv had no `setuptools`, which `triton.runtime.build` imports). Fixed in `setup_env.sh`. |
| 2026-10-29 | 26049138 | smoke.sh | gdev | resubmitted after installing setuptools |
| 2026-10-29 | 26049144 | reference_build0.sh + reference_build1.sh | g2 (2 GPU, 23.5 h) | reference rebuild, builds 0 and 1 (96 x 32 x 2 ns each) |
| 2026-10-29 | 26049145 | reference_build2.sh | g1 (23.5 h) | reference rebuild, build 2 |
| 2026-10-29 | 26049146 | stage1_t0/t1/t2 + smoke | g3 (4 GPU, 6 h) | Stage I-A replay + I-B dt; Stage II-A pilots kernel h=0.04/0.02, hist 129/257; smoke |
