# STATUS — adaptive-FR campaign (branch research/adaptive-fr-equal-budget)

*Updated 2026-10-17 06:23 UTC.*

* **Current task.** Experiment A, ITE 300 K, K256, T 200: ABF (the 16 reused EB runs) vs ABF + FR stopped at 10 t.u.
  * Preregistered in `PREREGISTRATION.md` §A.
  * The 16 FR runs are launching.
* **Completed.** Repository and hardware inspection; EB baseline verification (16/16 complete, same engine sha);
  pairing pilot (Gate F pass); 14 engine tests.
* **Active jobs.** See `results/adaptive_fr_campaign/logs/launches.tsv`.
* **Blockers.** None. CPU only, because every GPU belongs to other users.
* **Next action.** While A runs, write the A analysis (cross-directory pairing) and start Experiment B, the online
  stopping rule, on historical development data.
* **Compute.** 0.03 / 100 campaign core-h.
