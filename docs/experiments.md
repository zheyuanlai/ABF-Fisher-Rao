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
* Smoke 26028686 failed in 2 s: the uv venv linked the login node's /usr/bin/python3.12 (absent on compute nodes);
  rebuilt with a uv-managed Python. Stage-1 chain (smoke gates -> full gates -> reference -> ABF pilot) submitted
  twice: 26029098 (g2, + bench on GPU 1) and 26029637 (gdev, <= 2 h); whichever starts first cancels the other.
  g2 also holds another project's 16 h job queued ahead of ours.
* Stage-1 rerun 26030038 (gdev) cancelled for the whole-pipeline job 26030410 (g1), which started 00:18 on 10-28.
  Stage I gates: ALL PASS (V1 5e-9, V2/V3 ~5e-16, V4 mean-force e_F 0.017 / density 0.037 / ABF 0.11-0.14,
  V5 rms z 1.19 and dU 1e-13, V6 bitwise). Reference pilot (n = 16, 4 x 1024 walkers) then STOPPED at the new
  physical check: 2 of 4096 walkers overlapped after pre-equilibration (max |F| 1.4e17). Diagnosis job
  (scripts/lta_loading/debug_preeq.py) submitted to g1 (26030565) and gdev (26030575); g1 was taken by another
  project's 4 h job right after ours ended.
* 2026-10-28 pipeline 26030627 (g1, started 01:42): Stage I ALL PASS with the A2 pre-equilibration; reference
  ACCEPTED (F_ref SE rms 0.008 kJ/mol; mean-force vs density route rms 0.046, max 0.070; loaded barrier 13.8 kT vs
  single guest 13.6 kT; reinsertion acceptance 0.5 %; pilot 150 s, frozen 484 s at ~120 ns per walker-step).
* Budget pilot (ABF N = 256, 8 seeds, 1.2M steps, 317 s): b_mid 90.6–119.8M walker-steps, median 104.4M;
  **B = 208 928 768** (n_steps 3 264 512 at N = 64 ... 204 032 at N = 1024). Stage 1 done; bench + ladder running.
* Bench (c_ss, one seed): ABF 45–151 us/step, FR +21–32 %, N 64 -> 1024. Exploration ladder (16 seeds/arm) DONE,
  then found INVALID: (1) in every flattened-bias time window the barrier sits ~0.2 kJ/mol below F_ref; (2) the
  reference's own halves disagree (barrier +0.18 vs -0.14 kJ/mol around F_ref, per-group SEM 0.03/0.008; density route
  +0.42 vs -0.10). Cause: the environment relaxes on ~1e5+ steps (one reinsertion attempt per walker per 10 steps,
  0.5 % accepted -> one relocation per guest per ~30k steps), longer than the reference burn-in (50k) and comparable to
  whole runs at large N. The jackknife and two-route acceptance cannot see a drift common to all groups (the routes
  differed by 0.046 vs the 0.05 gate -- the only hint). Confirmation 26030627 and A3 jobs 26031928/9 cancelled;
  superseded results moved to results/lta_loading/superseded_v1/ (summaries in git, batch npz on scratch).
  Fix in progress (Amendment A4): reinsertion every step, 2000 Metropolis sweeps + 20k steps pre-equilibration, an
  energy drift monitor per save, and a stationarity gate on the reference. Relaxation diagnostic: job 26031951.
* 2026-10-28 ~04:00 v2 pipeline 26032190 (g1): Stage I ALL PASS; v2 reference ACCEPTED -- halves barrier 17.103 vs
  17.070 (diff 0.033 +- 0.018), halves rms 0.015, mean-force vs density 0.0056 (v1: 0.046), SE rms 0.006; barrier
  17.09 kJ/mol = 13.70 kT (single guest 13.59 kT).
* v2 bench: ABF 59–176 us/step, FR +19–25 %. v2 budget B = 236 584 960. v2 exploration ladder DONE (16 seeds/arm):
  fixed-N paired dI_F (FR vs ABF) -3.5 % [-16.3, +11.6] (N 64), -9.7 (128), -26.0 (256), -35.9 (512), -30.3 (1024);
  median b_mid ABF 34.9M / 68.0M / 118.3M / cens 9 / cens 16, FR 42.0M / 55.0M / 74.5M / 104.7M (4 cens) / cens 8.
  Frozen selection: ABF N 64, FR N 64 (exploratory R_B 1.20 [0.48, 1.88]); A3 wall-clock arms ABF N 64 (32.4 s),
  FR N 512 (26.6 s). Confirmations running in the same job.
* v2 pipeline 26032190 DONE (TASK_EXIT 0, ~5.2 h on one A100). Primary confirmation (N 64 both, 64 seeds/arm):
  R_B 1.02 [0.81, 1.25] **unresolved**; paired: dI_F -6.1 % [-12.6, +4.8], final e_F +41.9 % [+7.5, +57.6] worse with
  FR, FR ancestor ESS/N 0.026. A3 wall-clock (ABF N 64 vs FR N 512): R_W 0.93 [0.80, 1.10] **unresolved**. Write-up:
  docs/lta_loading/RESULTS.md.
* 2026-10-28 equal-N convergence figures (no new runs): `figures/lta_loading/fig_fixedN_convergence` (ladder, every
  N) and `fig_fixedN_confirm` (N 64 confirmation seeds), e_F vs physical time per walker and vs modelled A100
  wall-clock (steps x one-seed s/step). Median time to sustained mid accuracy, t.u. / s: N 64 ABF 109 / 32.4,
  FR 131 / 48.7; N 128 106 / 35.5 vs 85.9 / 35.6; N 256 92.4 / 36.4 vs 58.2 / 28.5; N 512 ABF not reached, FR 40.9 /
  26.6; N 1024 neither; confirmation N 64 113 / 33.5 vs 115 / 42.5. Table in RESULTS.md.

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
| 2026-10-30 | 26065238 | prod_finish.sh (frozen production analysis + figures) | qdev, depend=afterany:26057261 | runs automatically when the FR arms finish |
| 2026-10-30 | 26057261 | production FR arms (progress) | g3 | K64 DONE 15:48 UTC (81 min; stop reasons 9 safety / 3 established / 4 both; median t_stop 1.06 ns), K128 DONE 16:18 UTC (110 min; 7/7/2; median t_stop 1.74 ns), K32x4ns DONE 17:05 UTC (157 min; 14 established / 2 both; median t_stop 2.14 ns). The R2 'established' trigger fires in production (t1 0.9-1.7 ns), unlike at 1 ns calibration. K256 at 85 % at 17:20 UTC (~30 min left); prod_finish 26065238 held on it |
| 2026-10-30 | 26057261 | production FR arms (all four) | g3 | DONE 17:52 UTC, TASKS_EXIT 0 0 0 0. K256 x 2 ns: 205 min (repl 2630, clip 1.6 %). Job used 3.45 h of 4.25 h. prod_finish 26065238 started on qdev at 17:52 UTC |
| 2026-10-30 | 26065238 | prod_finish.sh (frozen production analysis + figures) | qdev | OK 17:55 UTC (3 min). PRIMARY ENDPOINT NOT MET: K128 G_F = -8.1 % [-13.8, +3.8]; no cell >= +10 % (K64 +3.7 %, K256 -0.6 %, K32 +2.0 %, all CIs include 0). e_F(T) 5 % lower under FR at K >= 64 (p < 1e-4) but e_F' 1.4-2.6 % worse, cond TV 11-25 % worse, transitions 9-18 % fewer, ESS/K 0.23-0.44. Stage IV in protocol section 6 |
| 2026-10-31 | 26066576 | diag_init_both.sh: ABF from PPII and from 50/50 helix/PPII inits, K = 64 x 2 ns, 8 seeds each, ABF only (post-protocol diagnostic of the R > 2.9 nm error's path dependence; helix control = production K64_T2ns/abf) | g1 (4:15) | submitted 18:50 UTC |
| 2026-10-31 | 26066577 | diag_finish.sh (analysis + figures of the diagnostic cells) | qdev, depend=afterany:26066576 | chained |
| 2026-10-31 | 26066609 | diag_ext_ppii.sh: ABF on the extended region only ([2.6, 3.6] nm, walls 2.65/3.55, 107 cells) from a restrained PPII start (R ~ 3.2 nm, k = 3200), K = 64 x 1 ns, 8 seeds, ABF only. Reason: the unrestrained 20 ps equilibration collapses PPII to median R = 2.12 nm (job 26066576 therefore tests a collapsed-PPII init, not an extended one) | gdev (1:59) | submitted 19:25 UTC |
| 2026-10-31 | 26066609 | diag_ext_ppii.sh | gdev | DONE 19:06 UTC (45 min). DECISIVE: ABF on [2.6, 3.6] nm from a restrained PPII start gives the SAME mean force as the helix-history production ABF to < 1.5 kJ/mol/nm in every 0.2-nm interval of [2.7, 3.5] (90.2/59.9/38.3/39.9 vs 90.3/58.8/38.2/40.0; seed IQR < 1) and the same y-class mixture; both reference estimators sit 40-95 kJ/mol/nm higher on [2.9, 3.5]. The ABF side is path-independent; the discrepancy is on the umbrella side. `scripts/deca_fr_2026/diag_extended.py` -> diagnostics/analysis/ |
| 2026-10-31 | (analysis of 26050158 data) | Stage I-B windows decomposed by starting preset (replica r -> preset r mod 6 of the umbrella's diverse pool) | login | SMOKING GUN: at every centre >= 2.95 nm the helix / beta / PPII / bridge / mixed presets agree with each other and with ABF (e.g. 3.04 nm: 51-53; 3.23: 35-38; 3.41: 38-39 kJ/mol/nm, replica SD 2-8), while the LEFT-HANDED helix preset (phi, psi = 60, 40) gives 314 / 508 / 666 / 571 / 507 / 501 / 435 at 2.95-3.50 nm (SD 1-90). One sixth of the umbrella replicas are kinetically trapped strained states; they inflate the reference F' above 2.9 nm by ~ (500 - 45)/6 ~ 75 kJ/mol/nm, the size of the 'ABF error'. The reference, not ABF, is biased above 2.9 nm |
| 2026-10-31 | 26066801 | diag_windows.sh: 10 stiff windows 2.6-3.5 nm x 48 replicas, 600 ps prod in 100 ps blocks, diverse pool vs PPII-only pool (same seeds) | gdev (1:59) | submitted 19:45 UTC: confirms the trapped-preset finding and its persistence in time |
| 2026-10-31 | (analysis of reference raw builds) | per-preset decomposition of the ACCEPTED reference's conditional mean force (3 builds x 96 windows x 32 replicas; replica (w, r) -> preset (32 w + r) mod 6) | login | Confirmed on the reference itself: on [2.9, 3.1) / [3.1, 3.3) / [3.3, 3.5) the five normal presets give 59.3-60.0 / 37.1-37.6 / 40.3-41.6 kJ/mol/nm (ABF: 58.8-59.9 / 38.2-38.3 / 39.9-40.0) while the left-handed preset gives 464 / 487 / 417; pooled reference 154.5 / 102.6 / 94.2. In every window with centre >= 2.95 nm the replicas > 2.5x the window median are exactly the left-handed ones (15-18 of 96 per window, all of them) |
| 2026-10-31 | 26066818 | reference_v2.sh: re-merge the three builds with --exclude-presets 5 -> results/deca_fr_2026/reference_v2; re-score production / fr_calibration / estimator_pilot / validation against it (analysis_v2, figures_v2); diag_extended v2 | qdev | submitted 20:05 UTC |
| 2026-10-31 | 26066801 | diag_windows.sh | gdev | DONE 20:02 UTC (2 x 22 min). PPII-only pool (10 centres 2.6-3.5 nm x 48 reps, k = 8000, 600 ps): 113 / 101 / 92 / 83 / 60.0 / 44.6 / 37.2 / 35.0 / 38.8 / 49.5 kJ/mol/nm, flat in 100-ps blocks, replica SD 2-7 = the ABF values. Diverse pool: the 5 ordinary presets give the same numbers; the LEFT replicas 216 / 395 / 605 / 622 / 498 / 495 / 446 at 2.9-3.5 nm, relaxing only slowly (3.3-3.5 nm: ~550 -> ~430 over 600 ps). Reference F'(cond) at the same centres: 133 / 181 / 118 / 103 / 99 / 97 / 82 |
| 2026-10-31 | 26066818 | reference_v2.sh | qdev | OK 20:24 UTC (18 min). Corrected reference: span 52.78 kT (v1 72.0), drift 0.23 (v1 0.97), F-from-F' vs MBAR 0.19 (v1 2.66), F' vs grad 0.025 (v1 0.12). ABF e_F(T) vs v2: 0.97 / 0.90 / 0.90 / 0.79 kJ/mol at K = 64 / 128 / 256 x 2 ns / 32 x 4 ns (was 12.2). FR: 2.00 / 2.34 / 3.10 / 0.90. G_F(I_F) at K128 = -25.2 % [-31.8, -4.0]; K256 -8.0 % [-20.6, -4.1]; K32 -15.1 % [-22.1, -0.4]; K64 -3.6 % [-20.7, +6.7]. Equal budget 128M: K32@4ns 0.79 ~ K64@2ns 0.97 << K128@1ns 11.2 << K256@0.5ns 27.5 (ABF). Protocol section 6 updated (revised verdict) |
| 2026-10-31 | 26066576 / 26066577 | diag_init_both.sh + diag_finish.sh (v1 and v2 scoring) | g1 / qdev | DONE 21:08 UTC (87 + 83 min). ABF K = 64 x 2 ns from collapsed-PPII / mixed inits: e_F(T) vs v2 = 0.99 / 0.82 kJ/mol (helix control 0.97); vs v1 = 11.8 / 12.1. Extended-region cell vs v2: 0.15. Converged ABF is init independent; the v1 plateau was the reference bias |

## LRS Laplacian selection baseline (branch baseline/lrs-laplacian-gateway-lta-20261031, PR #8), NSCC 2026-11-01

Protocol and prerequisites: `docs/LRS_LAPLACIAN_BASELINE.md` (section "NSCC execution"). Jobs via
`qsub -P personal-e1297740 -l select=1:ncpus=<n>:mem=<m> -l walltime=<w> -v TASK=scripts/laplacian_baselines/nscc/tasks/<t>.sh scripts/laplacian_baselines/nscc/cpu_task.pbs`;
logs `/scratch/users/nus/e1297740/logs/<task>_<jobid>.log`.

| date | job | task | queue | outcome |
|---|---|---|---|---|
| 2026-10-31 17:40 UTC | 26074433 | validate.sh (pytest, ITE framework rebuild, 3-arm smokes of gateway / LTA / ITE) | qdev (16 cores) | 4 min. Smoke ALL OK: force evaluations equal across arms; FR and Laplacian bitwise = ABF before fr_start (LTA, ITE); events realised and growing with the scale (gateway 4 t.u. N512: FR 314 deaths, Lap 0.01 77, Lap 0.1 674; ITE N64: FR 44, Lap 0.01 36, Lap 0.1 166); ESS is ESS/K. ITE framework rebuilt from the tracked CIF, geometry = tracked JSON. pytest 5/6: `test_lta_periodic_laplacian_sign_and_fourier_frequency` demanded corr > 0.999 where the exact ratio -0.4 cos / (1 + 0.1 cos) caps it at 0.99877 (measured 0.99874) -> tolerance set to 0.998 (as the gateway test). The held pilot job 26074435 was dropped by PBS (afterok on this exit 1). |
| 2026-10-31 17:40 UTC | 26074434 | ite_reference.sh | qdev (64 cores) | FAILED in 1 min: raced the framework rebuild of 26074433 (`framework_ite_itq3.npz` missing). Resubmitted as 26074468 once the framework existed. |
| 2026-10-31 17:43 UTC | 26074468 | ite_reference.sh (exact-MC ITE references 300 K + 150 K, frozen protocol/seeds, into results/laplacian_baselines/references/) | qdev (64 cores) | 300 K: 640 windows in 205 s (3.3 core-h; archive 2.24 core-h); Gate 1 analysis + fine reference running |
| 2026-10-31 17:46 UTC | 26074505 | pilot_rerun.sh: 704 runs = ABF + FR re-runs of the 8 cells (archived seeds) + FR + Laplacian x {0.003, 0.01, 0.03, 0.1, 0.3} on 8 fresh seeds per cell | qdev (128 cores, 1:59) | started 17:46 UTC (est. 88 core-h) |
| 2026-10-31 17:46 UTC | 26074506 | validate.sh (re-run with the fixed tolerance) | qdev (16 cores) | OK 17:48 UTC: pytest 6/6, smoke 0 failures |
| 2026-10-31 17:46 UTC | 26074507 / 26074508 | lta_reference.sh: sweep umbrella/WHAM references at 150 K (validation vs exact MC), 80 K, 350 K on one A100, converted to the lta_T{T}_reference.npz layout | gdev / g1 (dual) | g1 started at once; gdev copy cancelled. 150 K running 17:47 UTC |
