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
