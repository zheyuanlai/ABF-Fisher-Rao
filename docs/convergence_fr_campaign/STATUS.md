# STATUS — convergence-FR campaign (on main as a96b8532; the research branch is merged and deleted)

*Updated 2026-10-17 04:02 UTC (after the doc consistency check). Two algorithms only: histogram ABF vs the same ABF + marginal FR. The FR schedules
are continuous or early-stopping.*

| experiment | ran? | seeds | validation / integrity | result |
|---|---|---|---|---|
| 1 atlas (LTA 150/300 K, ITE 300/150 K) | **done** (descriptive) | 16 per system (historical) | reproduces every published contrast | FR reaches loose tolerances up to about 2× sooner everywhere; ITE late penalty; ITE 150 K "null" was the T 200 horizon |
| 3 **E1** ITE 300 K early-stopping (t_stop 10) | **done, confirmatory** | 16/16 pairs | engine tests 6/6 + 7/7; bitwise pilot; asserts pass | **PRIMARY SUPPORTED**: S_geo 1.59 [1.31, 1.85], 16/16; e_F(T) −32 % |
| 3 **E2** ITE 150 K early-stopping (t_stop 15, T 100) | **done, confirmatory** (A2) | 16/16 pairs | asserts pass | **PRIMARY SUPPORTED**: S_geo 1.25 [1.11, 1.58], 14/16; e_F(T) −19 %; H4 contradicted (per-seed post-stop error +15 %, no pooled bias) |
| 3 E1x t_stop ladder {5, 20} | **done** (exploratory, A3) | 4 X1 seeds | pre-stop bitwise identity with continuous FR asserted | S_geo stop 5 (median 1.43) > stop 20 (1.11) > continuous (1.04) on every seed; final error stop 5 better than stop 20 on 3 of 4 seeds; stop 10 not run on these seeds |
| 4 **ERI** 600 K continuous FR | **done, confirmatory** | 16 + 16 pairs | calibration blind (rate 2.0 safe at both K) | **CONTRADICTED at both K**: S_geo 0.52 / 0.51, 0/16; e_F(T) +113 / +119 % |
| 4 ERI early stopping | **not run** | – | frozen rule: no admissible t_stop (5 t.u. ≥ crossing quantile 4.3–4.9) | – |
| 2 matrix 300 K / 150 K at K256 | **done, confirmatory** | 16 + 16 pairs | common rate 1.0 by blind calibration (2.0 unsafe) | temperature difference PRESENT: FR helps more at 150 K (post-ramp Ī_F −22 % vs −8 %) |
| 2 matrix K1024 row | **not run** (budget, A2) | – | – | – |
| 5 circulation diagnostics | **done** (read-only) | all blocks | – | not supported as the cause of the late penalty; correction not run (the trigger is unmet or met only partly, depending on how "late error" is read, and there was no budget) |
| 6 efficiency, EB (ABF K256 × T200, equal force evaluations) | **done** (A2, descriptive) | 16 | – | ABF at K256 reaches each tolerance with 0.46–0.54× the force evaluations of FR-stop at K1024: **no equal-budget advantage** |

* **Compute.** ≈ 198.4 / 200 CPU core-h (closing balance); GPU 0 / 24 h. No GPU was available: GPUs 0–3 are held
  by other users.
* **Active jobs:** none. Every launch in `logs/launches.tsv` ended with ALLDONE.
* **Campaign state.** Complete within the authorised budget. The answers to Q1–Q7 are in `FINAL_SYNTHESIS.md`.
* **Next action, if more budget is authorised:**
  * early-stopping FR at the better ABF allocation (K256 × 200 t.u.) against EB;
  * the matrix K1024 row;
  * an online stop rule.
