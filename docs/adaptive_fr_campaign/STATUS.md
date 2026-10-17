# STATUS — adaptive-FR campaign (branch research/adaptive-fr-equal-budget)

*Updated 2026-10-17 08:06 UTC.*

* **Current task.** Experiment E, amendment E-A2: the K16 × T3200 frontier cell EB16 (32 runs, rate 1.0 by blind
  calibration, R1), launched 07:57. After it: re-run the E.3 frontier with eight configurations, the second
  wall-clock benchmark (with K16), then the final documents.
* **Completed.**
  * A: fixed stop at K256 × T200. PRIMARY UNRESOLVED, 1.28 [0.90, 1.43].
  * B: R1 developed on ITE only and frozen (c3ef3128).
  * C: R1 SUPPORTED in ITE 300 K, ITE 150 K, LTA 150 K and LTA 300 K; CONTRADICTED in ERI 600 K. R1 does not
    generalise by the preregistered definition.
  * D1, D2: mechanism, exploratory.
  * E: EA_R1 (R1 at K256 × T200) SUPPORTED, 1.31 [1.07, 1.47]. EB64 UNRESOLVED, 1.11 [0.92, 1.33]. Six-configuration
    frontier: best ABF = best FR = K64, ratio 1.02 [0.55, 1.46], UNRESOLVED. Wall-clock benchmark run 1 done.
* **Active jobs.** EB16 (pid 70191; `results/adaptive_fr_campaign/logs/EB16_rate1.log`).
* **Blockers.** None. CPU only, because every GPU belongs to other users.
* **Compute.** About 36.5 / 100 campaign core-h, before EB16 (about 9).
