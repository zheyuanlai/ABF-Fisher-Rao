# STATUS — adaptive-FR campaign (branch research/adaptive-fr-equal-budget)

*Updated 2026-10-17 08:35 UTC.*

* **Current task.** None. The campaign is complete. The final documents were corrected after an independent
  doc-vs-JSON audit (08:22–08:31) and are being committed and pushed to the research branch.
* **Completed.**
  * **A** (fixed stop at K256 × T200): PRIMARY UNRESOLVED, 1.28 [0.90, 1.43].
  * **B**: R1 developed on ITE only and frozen (c3ef3128).
  * **C**: R1 SUPPORTED in ITE 300 K (1.67), ITE 150 K (1.64), LTA 150 K (2.52) and LTA 300 K (1.46); CONTRADICTED in
    ERI 600 K (0.61). R1 does not generalise by the preregistered definition.
  * **D1, D2, D3**: mechanism, exploratory (MECHANISM.md).
  * **E**:
    * EA_R1 SUPPORTED, 1.31 [1.07, 1.47].
    * EB64 UNRESOLVED, 1.11 [0.92, 1.33]; EB16 UNRESOLVED, 0.93 [0.70, 1.29].
    * Eight-configuration frontier: best ABF = best FR = K16; best FR / best ABF 1.04 [0.74, 1.33], UNRESOLVED. The
      frontier is not bracketed below K16.
    * Wall-clock: best FR / best ABF 1.03–1.07 at 1–32 threads, all UNRESOLVED.
  * **Checks.** All analyses reproduce byte for byte (`reproduce_analyses.sh`); the engine and controller tests pass
    (35).
* **Active jobs.** None of ours.
* **Blockers.** None. Merging into `main` needs explicit permission.
* **Compute.** ≈ 44.1 / 100 campaign core-h; aggregate ≈ 242.5 / ≈ 298.4. No GPU.
