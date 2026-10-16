# ERI campaign — execution log

Times are UTC on the host clock. Every entry is written when the step happens.

## 2026-10-16

* **14:12** Campaign start. Repo state reviewed (`review/REPO_STATE_REVIEW.md`). Compute:
  * GPUs 0/1 belong to `yesom`. The project device is GPU 3, but no GPU has been used so far.
  * Host: 256 cores, load about 8–40, varying with other users.
* **14:50** User instruction: **no sham arm, ever**. Only ABF vs ABF + FR. Recorded in `STATUS.md` and in memory.
* **15:20** Literature verdict (`review/LIT_CROSSCHECK.md`): the PNAS 2005 model can only be reproduced as a labelled
  surrogate. The force field is confirmed (Schüring LJ pairs plus the Dubbeldam bond). The framework identity cannot
  be recovered, so two variants are built.
* **15:34** Frameworks built (`scripts/eri_campaign/build_frameworks.py`): Gard & Tait and Schlenker 1977. Each has
  1296 Si and 2592 O, 72 cages and 216 windows, and the coordination check passes.
* **15:40** Gate 0 tests: 16 pass (`tests/test_eri_model.py`). Committed as 3f7e10d7.
* **15:44** PREREGISTRATION.md frozen (framework rule, Gates 1–2).
* **15:47** A Gate 1 run was started and then KILLED within 1 minute, with no output analysed. The runner lacked the
  first-half accumulators that preregistered check 4 needs. They were added and the run relaunched.
* **15:51** Framework selection (§2), on groups 0–3:

  | variant | bead \|s\| barrier (kBT) | in band 6.13 ± 0.35? |
  |---|---|---|
  | Gard & Tait | 5.96 ± 0.02 | yes |
  | Schlenker | 5.29 | no |

  **PRIMARY = gard_tait**, the cited structure, and it is the only variant inside the band. The single-bead X
  profile gives 6.13.
* **16:00** Gate 1 (MC, 16 groups): **PASS on every check.**
  * D(F_density, F_MF) = 0.000 [0.008].
  * Split-half: 0.000 [0.014].
  * Equilibration: 0.013 [0.027].
  * Closure: −0.0013 ± 0.0016.
  * Noise: 0.007 kJ/mol.
  * Barrier: 33.28 kJ/mol (6.67 kBT), with ΔU = 5.30 and −TΔS = 27.97 kJ/mol, so 84 % of it is entropic.
* **16:16** Gate 2 (2000 bootstrap resamples): all four candidates are ADMISSIBLE by the frozen rule.

  | candidate | D_density [upper] | D_MF [upper] | bond-variance error |
  |---|---|---|---|
  | EM 2e-4 | 0.022 [0.048] | 0.037 [0.059] | +19.2 % |
  | EM 1e-4 | 0.000 [0.027] | 0.000 [0.029] | +8.8 % |
  | EM 5e-5 | 0.000 [0.036] | 0.000 [0.023] | +4.2 % |
  | LM 2e-4 | 0.000 [0.033] | 0.000 [0.017] | +0.04 % |

  The frozen rule gives the provisional production dt = 2e-4. It still has to pass the FR-blind second clause
  (D ≤ ⅓ of the ABF final error), which is checked at Gate 3.
* **16:05** Engine bug found in testing: the histogram ABF had **no trust rule**, and a sparse window bin could
  RUN AWAY. At FR rate 50 and warm-up 0.05 t.u., Γ went 1e3 → 1e17. ABF-only with no trust also reached
  |Γ| ≈ 1300. A per-bin trust ramp min(1, C_j/200) (the colvars `fullSamples` default) keeps |Γ| at 57–75
  kJ/mol/rad. The trust ramp is part of the frozen estimator. No clip is used.
* **16:13** ABF-only pilot: N = 64 / 256 / 1024, 40 t.u., dt 2e-4, warm-up 1 t.u., trust 200, starts at cage
  centres.
  * e_F(T) = 0.38 / 0.28 / 0.26 kJ/mol, almost independent of N.
  * The window population tracks the bias-aware target by about 4–8 t.u.
  * Dropping the deposits of t < 2 t.u. lowers the N = 1024 error from 0.264 to 0.048.
  * **Reading:** an N-independent early-deposit transient. The starts sit at cage centres, while equilibrium puts
    the molecule near the cage ends (axial τ about 1 t.u.).
  * **Remedy:** an unbiased pre-equilibration phase, `equil_steps`. It uses the IC stream, so the two arms stay
    paired, and its force evaluations are counted separately.
* **16:22** Gate 3a calibration launched: equil {0, 10k, 25k steps} × warm-up {5k, 20k} × N {64, 1024} × 2 seeds,
  ABF only.

  **Selection rule, written at 16:25 before any N = 1024 result was read:**
  * Choose the (equil, warm-up) pair that minimises the mean over N ∈ {64, 1024} of median Ī_F, each normalised by
    that N's (eq0, w5000) median.
  * Break ties within 5 % toward the shorter equilibration, then the shorter warm-up.
  * The N = 64 results were already visible. They favour equil > 0; warm-up is mixed.
* **16:43–17:10** Adversarial audit, four lenses with verification (`audit/AUDIT1_*.md`).
  * Engine, reference, WHAM and MC: correct; an independent re-implementation matches.
  * Confirmed defects:
    * the A1 establishment test has no power (F1, high);
    * the conditional-lag test has low power;
    * the classifier code had no tests;
    * the numba cache does not track cross-file changes (E1, latent; results unaffected);
    * provenance gaps in the sha recording (E2).
  * The forcefield-lens report was not returned before the session's new instruction; that lens is re-done in the
    overnight extension.
* **17:12** New user instruction ("overnight molecular extension"): exactly two arms, new candidates A–E. The
  aggregate compute ceiling is the existing one (≤ 200 core-h, ≤ 24 GPU-h in total).
* **17:25** Amendment 2 (powered tests + unit controls) frozen and committed (0e8d468a), BEFORE classifying.
* **17:27** Screen classified: **ABF-SUFFICIENT at K 64 / 256 / 1024** → STOP FR for the dilute ERI cell
  (`RESULTS.md`).
* **17:45** Forcefield/geometry audit lens returned (`audit/AUDIT1_FORCEFIELD_GEOMETRY.md`): **no result-affecting
  defect.**
  * Its independent direct quadrature of F(φ) reproduces the MC reference: shape RMS 0.024 kJ/mol, barrier 33.30 vs
    33.29, ΔU 5.298, entropic share 0.84.
  * The φ = 0 plane is open only at the 144 windows.
  * Low/info items:
    * Gard & Tait coordinates are stored rounded to 1e-4 fractional (≤ 1.3 mÅ, negligible);
    * the within-row windows sit at φ = ±π (a symmetry-equivalent hidden hop, no bias);
    * the generic orthorhombic builder lacked safety checks. Fixed: coordination and duplicate checks,
      even-permutation guard, explicit cage/window positions.
