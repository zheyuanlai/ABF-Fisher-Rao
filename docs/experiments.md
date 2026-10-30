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
| 2026-10-29 | 26049138 | smoke.sh | gdev | tests 58 pass; A100 throughput (float64, compiled): B=128 4.2 ms/step, 512 3.35, 1024 3.2, 2048 3.6, 4096 6.5 (kernel); tiny cell FAILED: `np.savez` appended .npz to the temp init file -> rename failed. Fixed (run_cell.py). |
| 2026-10-29 | 26049146 | stage1 bundle | g3 | started 07:24 UTC with the unfixed run_cell.py -> qdel'd; resubmitted as 26049277 |
| 2026-10-29 | 26049271 | refsmoke_cpu.sh | qdev (CPU) | toy reference (4 x 3, 20k pull) to exercise save + merge on CPU |
| 2026-10-29 | 26049277 | stage1 bundle | g3 | resubmitted after the init-file fix |
| 2026-10-29 | 26049278 | smoke.sh | gdev | resubmitted after the init-file fix |
| 2026-10-29 | 26049271 | refsmoke_cpu.sh | qdev | FAILED: its output dir was under smoke/, which the g3 smoke task rm -rf'd mid-run; moved, resubmitted as 26049317 (26049316 qdel'd: task not yet fixed) |
| 2026-10-29 | 26049277 | stage1 bundle | g3 | qdel'd: run_cell.py ignored cell-level n_walkers/n_steps/rng_seed (all cells ran at K=16, 0.5 ns). Fixed; resubmitted as 26049477 (26049471 withdrawn: task split rebalanced); partial outputs deleted |
| 2026-10-29 | 26049317 | refsmoke_cpu.sh | qdev | OK: run_reference (2 toy builds, checkpoints) + merge_reference end to end on CPU |
| 2026-10-29 | 26049499 | determinism.sh | gdev | tiny ABF cell twice with the same seed: are A100 trajectories run-to-run bit-identical? (the smoke showed abf vs fr_uniform differing with zero events) |
| 2026-10-29 | 26049278 | smoke.sh | gdev | SMOKE_OK: tests, throughput, tiny two-arm cell, toy reference + merge, analysis + paired stats all run on the A100 |
| 2026-10-29 | 26049477 | stage1 t0: replay_legacy | g3 | DONE 24 min. Defect reproduced on A100: learned span on I 79.9-107.4 kT (2026-08: 86.7-110.3), 2nd-half fraction above 2.80 nm 0.91-1.00 (2026-08: 0.951-0.9996). Faithful replay. |
| 2026-10-29 | 26049499 | determinism.sh | gdev | A100 engine NOT run-to-run bit-deterministic: identical ABF repeats agree for 3.7 ps then separate (|dxi| 0.03 nm at 8 ps; PMF 0.22 kJ/mol). Arms share seed/init/noise; divergence is numerical. Noted in protocol §2. |
| 2026-10-29 | 26049477 | stage1 t0: replay_fixed | g3 | DONE 24 min. Stage I-A verdict: the half-kick fix does NOT change the 2026-08 outcome at 16 x 0.5 ns: span on I 81.2-112.7 kT (med 96.8) vs legacy 79.9-107.4 (med 94.1); paired fixed-legacy span differences +16.1, +13.0, -2.4, -8.5, +14.4, +1.2, -19.3, +8.0 kT (no systematic sign); pinned above 2.80 nm 0.972-1.000 (med 0.999) vs 0.910-1.000 (med 0.990). |
| 2026-10-29 | 26049477 | stage1 t1/t2/t3: kernel_h04, kernel_h02, hist257 | g3 | DONE 42-48 min each (K=64, 1 ns, 8 seeds). Reference-free: span med 55.2 / 63.7 / 60.3 kT (ref 72), pinned above 2.80 nm med 0.822 / 0.894 / 0.924, TV to wall-uniform med 0.41 / 0.53 / 0.54. e_F awaits the rebuilt reference. |
| 2026-10-29 | (preview) | provisional 1 ns-checkpoint F'_ref (integrated conditional mean force; span 69.6 kT, argmin 1.637 nm) | login | e_F(T) median: replay legacy 39.6 / fixed 40.2 kJ/mol (rel mean-force err 0.99 / 0.96); K=64 1 ns: kernel h04 13.5 (0.46), kernel h02 14.1 (0.52), hist257 12.0 (0.49); I_F: 40.0 / 31.2 / 13.8. Official numbers from the merged 2 ns reference. |
| 2026-10-29 | 26049477 | stage1 t2: dt1fs_K64, dt05fs_K64 | g3 | DONE 8.8 / 17.6 min (K=64, 200 ps, 8 seeds, kernel). Provisional e_F(T) med 66 vs 101 kJ/mol; 0.5 fs worse in 7/8 seeds at EVERY time (50 ps: 90 vs 138) while the walker dynamics coincide (pinned fraction by quarter 0.31/0.78/0.91/0.95 vs 0.31/0.74/0.89/0.96). Confounded: `abf_min_count` counts raw samples, so at 0.5 fs the trust ramp saturates in half the physical time and the early-bias span inflation is larger. Not an integrator verdict; superseded by the direct window test below. |
| 2026-10-29 | 26050158 | dt_windows.sh | gdev | Stage I-B direct: <f_loc|R> in 25 stiff windows x 48 replicas, 100 ps equil + 400 ps prod, dt 1 fs then 0.5 fs, same seeds |
| 2026-10-29 | 26049477 | stage1 t1: hist129 | g3 | DONE 39 min. Provisional e_F(T) med 12.7 kJ/mol, I_F 18.1, rel mean-force err 0.50. Stage I/II compute complete (job total ~1.7 h x 4 GPUs). Provisional ranking by I_F: hist257 13.8 < hist129 18.1 < kernel h02 31.2 < kernel h04 40.0; e_F(T): 12.0 < 12.7 < 13.5 < 14.1. |
| 2026-10-29 | 26049144 / 26049145 | reference builds 0,1 (g2) and 2 (g1) | g2, g1 | DONE 10:04 UTC: 2.81 h each (3072 states x 2.22 M steps), 4000 samples/replica, 0 replicas excluded in every build. ~8.4 GPU-h. |
| 2026-10-29 | 26050219 | calibration bundle (g3, held) | normal | withdrawn: g3 taken by the other project's 20 h job (26050213); re-routed to g2 + g1 |
| 2026-10-29 | 26053061 | after_reference.sh (merge + Stage I/II analysis) | qdev | merge OK: reference ACCEPTED (span 179.60 kJ = 72.00 kT, argmin 1.637, L2 0.66, drift 0.74, Gate A 0.764; F'_ref integrates to F within 2.66 kJ/mol). Stage II-A official: kernel h04 11.72/40.83, h02 13.43/32.28, hist129 11.84/18.98, hist257 10.76/14.21 (e_F(T)/I_F). FROZEN: histogram, 257 cells. plot.py crashed on the 257/129 grid mismatch (fixed; figures rendered on the login node). |
| 2026-10-29 | 26053062 | calibration (calib_t0 + calib_g2b) | g2 (3.5 h) | released after the estimator freeze: K=128, 1 ns, 8 seeds; abf + rates 0.005 / 0.02 / 0.05, R2 stop |
| 2026-10-29 | 26053063 | prod_K64_T2ns_abf | g1 (3 h) | released: production K=64 x 2 ns ABF arm (also the Stage II-B 2 ns extension) |
| 2026-10-29 | 26053327 | dt_windows.sh | g1 (2.2 h) | dual-submission of 26050158 (stuck 6 h in gdev); whichever starts first runs, the other is cancelled |
| 2026-10-29 | 26053334 | prod_K128_T2ns_abf | g1 (2.8 h) | production ABF arm, queued behind 26053063 |
| 2026-10-29 | 26053335 | prod_K256_T2ns_abf + prod_K32_T4ns_abf | g2 (4.5 h) | production ABF arms, queued behind the calibration |
| 2026-10-29 | 26053062 | calibration rate020 | g2 | DONE 44 min. Safety stop in 8/8 seeds at 130-230 ps (max lineage > 0.10), established never fired; 87-157 events/seed; ESS/K 0.16-0.28 (3/8 < 0.20). e_F(T) abf 10.8 vs fr 15.8 (FR worse 7/8); I_F 16.3 vs 19.8; FR more pinned above 2.8 nm (0.97-1.00 vs 0.90-0.96). FAILS the calibration criteria. |
| 2026-10-30 | 26056977 (26056888 withdrawn: g2 taken by pc_lt for 16 h) | calib rate002 + rate001 | g1 (2.25 h) | reduced-intensity calibration: no tested rate met the lineage criterion (0.005: safety stop in 6/8 at 325-565 ps, ESS/K 0.22-0.37; 0.02 and 0.05: safety in 8/8, accuracy worse) |
| 2026-10-30 | 26050158 / 26053327 | dt_windows.sh (ran on BOTH gdev and g1; loser not cancelled) | gdev, g1 | dt verdict: 1 fs vs 0.5 fs window mean forces differ by 3.0 kJ/mol/nm mean abs (5 % RMS rel), no window > 2 SE; dt = 1 fs accepted. T_kin 293.7 vs 298.4 K. Replica SD ~90 kJ/mol/nm at R >= 2.9 nm. |
| 2026-10-29 | 26053062 | calibration rate005, rate050 | g2 | rate005: safety stop 6/8 (325-565 ps), ESS/K 0.22-0.37, e_F(T) 12.0 vs abf 10.8, I_F 17.0 vs 16.3. rate050: safety 8/8 at 75-125 ps, e_F 14.9. "established" never fired at any rate. |
| 2026-10-29 | 26053063 / 26053334 / 26053335 | production ABF arms K64, K128, K256 (2 ns), K32 (4 ns) | g1, g1, g2 | DONE: 90 / 110 / 204 / 161 min |
| 2026-10-30 | (analysis) | production ABF arms, provisional | login | e_F(T) med 12.63 / 12.19 / 12.15 / 12.18 kJ/mol for K=32x4ns / 64 / 128 / 256 x 2 ns (seed spread +-0.4); minimum ~10.8 near 1 ns then rises to the K-independent plateau. Error located at R > 2.9 nm: ABF mean force 55-96 kJ/mol/nm below BOTH reference F' estimators; -17/-31 kJ/mol in F at 3.1-3.5 nm. Equal budget 128M: K128@1ns 11.3 < K64@2ns 12.2 < K32@4ns 12.6 < K256@0.5ns 20.8. |
| 2026-10-30 | 26056977 | calib rate002 + rate001 | g1 | DONE 04:03 UTC (46 + 45 min). rate001: no stops, ESS/K 0.60-0.78, max lineage <= 0.047 (8/8 pass); e_F 11.7 vs abf 10.8. rate002: 1/8 safety at 750 ps, ESS/K 0.37-0.56, lineage 7/8 <= 0.10; e_F 12.4. FROZEN fr_rate = 0.002 (largest rate passing in essentially all seeds). |
| 2026-10-30 | 26057261 | production FR arms K256, K32, K128, K64 (fr_uniform, rate 0.002) | g3 (4 GPU, 4.25 h) | released 13:40 UTC after the rate freeze (held since 03:40; session gap) |
| 2026-10-30 | 26064453 | calib_finish.sh | qdev | FAILED: cmp on npz files (zip timestamps differ); replaced by an array comparison, resubmitted |
| 2026-10-30 | 26064477 | prod_K256_T2ns_fr_uniform | g1 (3.8 h) | g3 job 26057261 stuck on node_pool; K256 FR arm moved to g1 (g3 job kept queued, skips completed cells) |
| 2026-10-30 | 26064478 | prod_K64_T2ns_fr_uniform | gdev (1:59) | K64 FR arm (90 min) on gdev |
| 2026-10-30 | 26064476 | calib_finish.sh | qdev | OK: official calibration table (analysis/fr_calibration_*); cond TV 2nd half worse under FR at every rate (0.19-0.22 vs 0.15); calibration inits differ chaotically between rate cells (max dx 0.7-0.8 nm): seed-matched, not trajectory-paired |
| 2026-10-30 | 26064477 / 26064478 | K256 FR (g1), K64 FR (gdev) | g1, gdev | qdel'd after 1 h / 5 min: the 4-arm g3 job 26057261 started and would have duplicated them |
