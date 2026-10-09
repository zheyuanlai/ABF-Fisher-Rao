# Next experiments (at most three, prioritised)

*2026-10-10, after the overnight numerical-validation campaign (`OVERNIGHT_REPORT.md`). Each experiment
must be preregistered with its decision rule before it is run, as the campaign's were.*

## 1. LTA replica ladder at equal force-evaluation budget, at the validated dt (300 K and 150 K)

* **Hypothesis.** LTA's FR gain is a parallel-time (establishment) gain, not a total-compute gain.
  Concretely: at a fixed budget B = N × n_steps (the published B = 1024 × 300000 molecule-steps), the best
  ABF-only split of B across N ∈ {1024, 256, 64, 16, 4, 1} reaches an integrated error Ī_F (over budget
  fractions) **no worse** than 1024-replica ABF + FR. This is the pattern found on the gateway and on WCA.
* **Falsified if** ABF + FR at N = 1024 has a lower Ī_F than ABF at *every* N, with paired bootstrap CIs
  excluding 0. FR would then reduce total compute in LTA, the first such case in the project.
* **Scientific value.** Decisive for the wording of any acceleration claim: "faster in wall-clock with N
  workers" vs "fewer force evaluations". Without it, the paper may only claim the former.
* **Expected cost.** About one day of engineering for a numba CPU port of the LTA histogram-ABF + uniform
  FR + sham sampler. Validate it bitwise against `src/lta/core_lta.py` as was done for WCA and the
  gateway; `src/lta_validation.py` already holds the validated potential. Then about 6 N × 2 arms × 16
  seeds × ~0.3-1 core-h ≈ 100-200 CPU core-h. No GPU.
* **Necessary before publication:** **yes**, if the paper claims an acceleration of free-energy
  *computation* rather than of the establishment phase.

## 2. WCA at the validated time step

* **Hypothesis.** The 0.017 kT stationary tilt of uniform FR at the validated dt is caused by
  finite-sample *score noise* that correlates with transit (non-equilibrated) walkers. If so, it shrinks:
  * with a less noisy score: a leave-one-out KDE, or a 2× bandwidth;
  * with a larger population: N = 1024 instead of 256, at the same physical time;
  * with a post-clone decorrelation: the clone is held out of the estimator for τ_fibre ≈ 0.05 t.u.
    (W0: τ_f 3-6 steps at dt 0.002).
* **Falsified if** none of these changes moves D(FR limit, ABF limit) below 0.008 kT with CIs. The bias
  would then be intrinsic to marginal FR selection on this system, and the paper would report it as a
  limitation of the method, not of an estimator choice.
* **Scientific value.** It separates "FR is biased" from "this FR score is biased". It also tests the
  fibre-lag mechanism that explains why FR helps in starved regimes (LTA, gateway) and hurts in smoothly
  converging ones (WCA, alanine).
* **Expected cost.** dt 0.000125, T 240, 16 seeds, 4 arms (ABF, FR-LOO, FR-2bw, FR-hold), N 256: about
  60 CPU core-h, 2 h wall. Engine work: about 2 h for the LOO score and the hold-out flag in
  `src/wca_numba.py` (the gateway port already has a LOO score). N = 1024 adds about 160 core-h, or
  needs a parallelised engine.
* **Necessary before publication: no**, if WCA is reported as a validated negative control. **Yes**, if
  the paper claims that FR leaves the stationary measure unbiased (the mean-field theorem
  d/dt p(y | ξ)|_FR = 0), because this is a measured finite-N counterexample.

## 3. LTA validation at 80 K (the headline −35 % point)

* **Hypothesis.** The 80 K dynamics and umbrella reference pass the same frozen gates as 300 K and 150 K,
  and the published 80 K FR gain (−35.1 % integrated, −83 % final) survives rescoring against an exact
  umbrella-MC reference.
* **Falsified if**:
  * EM at dt 2e-4 fails the 0.03 kJ/mol gate at 80 K. That is not expected: the bond's dt·λ is
    T-independent and the LJ curvature relative to kT is the same order. But the window is the narrowest
    at 80 K and the umbrella windows the stiffest (κ = 136 kJ/mol/rad² at kT = 0.67).
  * Or the rescored FR-vs-ABF or FR-vs-sham CI includes 0.
* **Scientific value.** It closes the validation of the quoted temperature range at its strongest point.
  The 300 and 150 K results bracket the mechanism, but the paper's largest number is at 80 K.
* **Expected cost.** About 25 CPU core-h with `scripts/numerical_validation/run_lta_validation.py`, after
  adding `80.0: 136.0` to `KAPPA`. About 15 minutes wall on 100 cores.
* **Necessary before publication:** **yes, if the 80 K number is quoted**. Otherwise optional. The 225 K
  and 350 K points sit between validated temperatures and need not be redone unless quoted individually.

## Not proposed

* More WCA dt 0.002 or 0.0005 work: those cells are numerically invalid.
* Alanine (validated BAOAB integrator; neutral is existing evidence).
* New benchmarks.
* FR-parameter retuning on any system.
