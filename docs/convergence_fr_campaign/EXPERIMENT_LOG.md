# Experiment log — convergence-FR campaign

Times are UTC and come from `date -u` or git.

## 2026-10-17

* **02:52 — inspection.**
  * main = origin/main = 5223d13e. No running job of this project.
  * GPUs 0/1 belong to `yesom`; GPUs 2/3 to `hyt35` (pid 3926155, started about 00:02). GPU 3 is not available.
  * The machine has 256 cores, and other users load about 10.
* **02:57 — worktree.** Created `/home/zheyuanlai/ABF-Fisher-Rao-convergence` on the new branch
  `research/convergence-fr-stopping`.
  * A dedicated worktree, because other sessions and the user's own shell use the main checkout.
  * Ignored inputs are linked with `scripts/convergence_fr/link_data.sh`. Historical run files are read through
    `ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao`.
* **03:00 — engine.** Added `fr_stop_steps` (early-stopping FR). The tests pass (12/12):
  `tests/test_fr_stop.py` (5) and `tests/test_eri_engine.py` (7).
* **03:02 — reproduction pilot.**
  * The modified engine reproduces `results/overnight/ite/prod_confirm/fr_N1024_s0.npz` bitwise over its first
    20 000 steps (17 saves, about 10 000 FR steps), at 1 thread.
  * Throughput: 1.54 µs per molecule-step including compilation; equilibration 69 s.
  * Commit 1056ce6f.
* **03:05 — Experiment 1 atlas** (descriptive, existing data).
  * It reproduces every published contrast exactly:
    * LTA exact-reference rescoring −28.5 % / −54.4 % (150 K) and −13.7 % (300 K);
    * X1 −14.9 % / +14.0 %;
    * X2 −1.1 % / +56.8 %.
  * New observation: X2 truncated at T 50 gives ΔĪ_F −12.7 % [−14.3, −8.3].
* **03:11–03:25 — preregistration** of E1, ERI and the matrix written (`PREREGISTRATION.md`). The block analysis
  was smoke-tested on X1 data (exploratory).
* **03:13:20 — PREREGISTRATION frozen** (commit 4431ce30).
* **03:13:31 — launches.**
  * E1: 32 runs, 32 processes, 1 thread each, `jobs_E1.json`.
  * ERI calibration rung rate 2.0: 6 runs, `jobs_ERIcal_rate2.json`.
  * ITE calibration rung rate 2.0, K256 at 300 / 150 K: 6 runs, `jobs_ITEcal_rate2.json`.
  * The registry is `results/convergence_fr_campaign/logs/launches.tsv`.
* **03:17 — ITE common-rate calibration, rung 2.0** (blind, genealogy only):
  * **unsafe** at 300 K / K256: min median ESSw/K 0.293 < 0.30, lineage 0.0508 > 0.05;
  * **unsafe** at 150 K / K256: 0.258, lineage 0.078.
  * Cost 0.34 core-h; 206 s per K256 T-50 run.
  * Next rung 1.0 launched at 03:17:13 (`jobs_ITEcal_rate1.json`).
  * Consequence for the matrix: the common rate is < 2.0, so X1 cannot serve as the controlled 300 K / K1024 cell
    (prereg §5).
* **03:17 — ERI calibration K256, rung 2.0:** **SAFE** (ESSw/K 0.554, lineage 0.023, 45 replacements per t.u.).
  The selected K256 rate is 2.0, the top rung. Each K256 run took 129 s.
* **03:21 — ERI K256 production launched:** 32 runs (`jobs_ERI_rate2_K256.json`).
  * `make_jobs.py` gained `--K` so that the two replica counts can use their own calibrated rates. This is
    generation tooling only; seeds and settings are as preregistered.
* **03:18–03:21 — read-only work.**
  * Circulation diagnostics on X1/X2 (`circulation.py`). The late-half-deposit definition reproduces the overnight
    numbers exactly.
  * Block analysis of X1/X2 under the frozen primary (exploratory): X1 S_geo 1.06 [0.95, 1.15]; X2 (ABF ladder)
    0.82 [0.60, 0.95].
* **03:20–03:22 — calibration outcomes** (blind; `frcal_K*.json` in each calibration directory).

  | cell | outcome | selected rate |
  |---|---|---|
  | ITE 300 K / K256 | rung 1.0 **SAFE** (ESSw/K 0.531, lineage 0.027) | **1.0** |
  | ITE 150 K / K256 | rung 1.0 **SAFE** (0.490, 0.039) | **1.0** |
  | ERI K1024 | rung 2.0 **SAFE** (0.675, 0.006; 100 replacements per t.u.; 507 s per run) | **2.0** |

  * **Common matrix rate r_c = 1.0.** It is the largest rung safe at 300 K / K256, at 150 K / K256, and at
    300 K / K1024, where the X1 calibration found every rung up to 2.0 safe.
  * Per prereg §5, the 300 K / K1024 cell therefore needs a new block at rate 1.0. X1, at rate 2.0, cannot be
    reused as that cell.
* **03:22 — ERI K1024 production launched** (32 runs, `jobs_ERI_rate2_K1024.json`).
* **Queued** behind ERI K256: matrix M300_256 and M150_256 at rate 1.0, 16 processes each.
* **03:23:12 — ERI K256 production done** (1.168 core-h, wall 138 s; 32/32 runs).
  * Frozen analysis (`analysis/ERI256.json`): pairing 16/16 bitwise, force evaluations equal, no config difference
    apart from the method.
  * **PRIMARY S_geo 0.52 [0.46, 0.63], FR faster 0/16 → CONTRADICTED.**
  * Post-ramp Ī_F +32 % [+22, +46]; e_F(T) +113 % [+79, +236]; post-FR-start estimate +126 %.
  * FR is ahead only at 1.5–4 t.u. (up to −21 %). The crossing is at 5.2 t.u. [4.3, 6.2], 15/16 seeds.
  * ABF window establishment is 2.1 t.u. vs FR 1.9: ABF is not establishment-limited.
* **03:25 — ERI K256 mechanism** (`scripts/overnight/fr_mechanism.py`, read-only; `mechanism/fr_mechanism_ERI256.json`).
  * Pooled late-half estimate vs MC: ABF D 0.027, FR D 0.130; FR vs ABF D 0.100 (noise 0.036).
  * The difference is mostly in the flank mean force (regional RMS dΓ 0.19–0.24 kJ/mol/rad).
  * **FR distorts the cage conditional of η** (c-axis position): deposit-weighted η − MC is −0.014…−0.029 in the
    cage coarse bins vs ≤ 0.009 for ABF (RMS 0.016 vs 0.0033). u_x² is barely affected (0.0019 vs 0.0007).
  * This is the first FR-induced conditional distortion seen in these molecular cells; ITE's were clean.
* **03:26 — Amendment A1** (audit response) committed as 9b044bc8, before any E1 data was read. The audit found no
  blocker.
* **03:27:41 — E1 done** (32/32; 7.259 core-h; 812–850 s per run under load ≈ 105).
* **03:27:50 — E1 frozen analysis** (`analysis/E1.json`). All asserts pass: pairing 16/16, the seed block, 9000 FR
  opportunities per run, 0 events after the stop.
  * **PRIMARY S_geo 1.586 [1.311, 1.852], 16/16 → SUPPORTED** (97.5 %: [1.306, 1.921]).
  * H1 −37.4 % [−40.0, −29.2]; H2 e_F(T) −32.3 % [−42.6, −24.1]; H4 post-stop estimate −0.0 % [−21.0, +24.9].
  * See `EARLY_STOPPING_RESULTS.md`.
* **03:30 — Amendment A2** (E2, EB) committed as 586e8bdc. E2 and EB launched at 03:30:07.
* **03:30:31 — matrix K256 cells done**; analysed with `--ladder abf`.
  * M300_256: S_geo 1.035 [0.917, 1.212], UNRESOLVED.
  * M150_256: S_geo 1.633 [1.341, 2.727], SUPPORTED.
  * Temperature contrast, post-ramp Ī_F: +13.9 points [+4.4, +25.2] → difference PRESENT; FR helps more at 150 K.
* **03:32 — ERI K1024 done** (4.801 core-h).
  * **PRIMARY S_geo 0.510 [0.487, 0.577], 0/16 → CONTRADICTED** (Bonferroni 97.5 % [0.485, 0.598]).
  * e_F(T) +119 % [+103, +141]; post-start estimate +142 %. Cage η conditional distorted (RMS 0.017 vs ABF 0.0067).
* **03:33 — ERI early-stopping decision** (prereg §4, frozen rule).
  * At K1024 the trigger IS met: early Ī_F over [1, 10] is −2.5 % [−3.7, −1.0] and e_F(T) is +119 %, both resolved.
  * At K256 the trigger is not met: early Ī_F is −4.3 % [−8.3, +3.1].
  * The t_stop rule gives 5 t.u. (2 × the ABF establishment time of 2.1 t.u., rounded up). That is NOT below the
    crossing's 2.5 % quantile: 4.9 t.u. at K1024, 4.3 at K256.
  * **No admissible stop time, so no ERI early-stopping block is run.** FR's advantage window, about 1.5–5 t.u.,
    closes before twice ABF's establishment time.
