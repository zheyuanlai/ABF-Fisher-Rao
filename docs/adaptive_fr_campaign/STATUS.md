# STATUS — adaptive-FR campaign (branch research/adaptive-fr-equal-budget; amendment E-A3 on claude/priceless-rubin-9602df)

*Updated 2026-10-17 12:10 UTC.*

* **Current task.** None. Amendment E-A3 (frontier bracket below K16) and the nested best-vs-best bootstrap are complete
  and committed on `claude/priceless-rubin-9602df`; merging into `main` is the user's decision.
* **Completed.**
  * **A** (fixed stop at K256 × T200): PRIMARY UNRESOLVED, 1.28 [0.90, 1.43].
  * **B**: R1 developed on ITE only and frozen (c3ef3128).
  * **C**: R1 SUPPORTED in ITE 300 K (1.67), ITE 150 K (1.64), LTA 150 K (2.52) and LTA 300 K (1.46); CONTRADICTED in
    ERI 600 K (0.61). R1 does not generalise by the preregistered definition.
  * **D1, D2, D3**: mechanism, exploratory (MECHANISM.md).
  * **E**:
    * EA_R1 SUPPORTED, 1.31 [1.07, 1.47].
    * EB64 UNRESOLVED, 1.11 [0.92, 1.33]; EB16 UNRESOLVED, 0.93 [0.70, 1.29]; **EB8 (E-A3) UNRESOLVED, 1.03
      [0.72, 1.42]**, establishment 4.5 → 4.5 t.u. at 1 t.u. resolution.
    * Eight-configuration frontier: best ABF = best FR = K16; best FR / best ABF 1.04 [0.74, 1.33] fixed selection,
      **[0.78, 1.28] with the selection re-done per bootstrap replicate** (EFFICIENCY.md §7), UNRESOLVED.
    * **Extended frontier K1–K1024 (E-A3):** best ABF = K1 (0.170 × 10⁸), best FR = K16 (0.188); best FR / best ABF
      **1.10, fixed [0.75, 1.51], nested [0.87, 1.43], UNRESOLVED**. ABF K1 / ABF K16 0.94 [0.62, 1.38]; FR K8 / FR K16
      1.06 [0.75, 1.37]. ABF's cost is flat from K1 to K64; the frontier is explored to its floor (K1 for ABF, K8 for FR
      under the safety rules) and has no resolved minimum.
    * No K4 FR arm: R1's genealogical backstops cannot fire at K4 (preregistered criterion failed by arithmetic).
    * Wall-clock (K16–K1024): best FR / best ABF 1.03–1.07 at 1–32 threads, all UNRESOLVED.
  * **Checks.** All analyses and PNG figures reproduce byte for byte (`reproduce_analyses.sh`, 12:03 UTC);
    `E_frontier.json` keeps every published key byte-identical to 6716aefc.
* **Active jobs.** None of ours.
* **Blockers.** None.
* **Compute.** ≈ 65.9 / 100 campaign core-h (E-A3: 21.8); aggregate ≈ 264.3 / ≈ 298.4. No GPU.
