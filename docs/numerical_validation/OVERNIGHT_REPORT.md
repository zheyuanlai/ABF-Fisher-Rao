# Overnight numerical-validation campaign: ABF–Fisher–Rao (2026-10-09/10)

*Branch `research/langevin-validation-oct2026`. Executed autonomously, not pushed.*

Every gate below was frozen in a committed preregistration before the data it governs was looked at:

* `configs/numerical_validation/wca_validation_prereg.json`
* `configs/numerical_validation/lta_validation_prereg.json`
* `configs/numerical_validation/wca_confirmatory_prereg.json`
* `configs/numerical_validation/gateway_dt_prereg.json`

Compute: 254 CPU core-hours on the shared node (`results/numerical_validation/compute_accounting.json`)
and **0 GPU-hours**. GPUs 0-1 carry another user's jobs and were not touched; GPUs 2-3 were not needed.

## 1. Executive summary

1. **The WCA "FR acceleration" was a numerical artefact. Refuted.**
   * The accepted WCA cell ran Euler–Maruyama at dt = 0.002. That step is *beyond the linear stability
     limit* of ordinary WCA collisions: dt·V″ = 2.05 at r = 0.95σ (V = 3 kT).
   * A per-particle force clip kept the chain bounded. That clip is not the gradient of any potential.
     It made the stationary law a different fluid: the clip binds in 29 % of steps, pairs penetrate
     inside 0.65σ in 2.8 % of steps, and a pair is closer than 0.865σ in 31 % of configurations against
     2.5e-6 under exact Gibbs.
   * The accepted reference (v2 TI) was built with the same clipped integrator. It is **0.29 kT RMS
     (0.9 kT max) away from the exact Gibbs free energy**, two independent exact samplers agreeing.
   * FR's dose-dependent stationary tilt happens to point toward that artefact reference. Hence
     "FR −24 % integrated, −51 % final, 16/16 wins". Rescored against the exact free energy, the *same
     runs* give **FR +18.5 % integrated, +34 % final, 0/16 wins**. FR is harmful at every N ≥ 16 at both
     dt 0.002 and dt 0.0005.
2. **The WCA force field and the ABF mean-force estimator are correct.**
   * The analytic gradient matches central differences at O(ε²); Newton's third law and periodicity
     hold.
   * The LRS mean force with its −2w/(βr) Jacobian term agrees with the density route to 0.0002 kT
     under exact sampling.
   * The defects are the two regularisations: the min_r clamp (force/energy inconsistent) and the
     non-conservative force clip. Both act only when the integrator overshoots.
3. **WCA dynamics gate: RESOLVED at dt* = 0.000125 (16× smaller than accepted).**
   * Production EM, clips included, reproduces exact Gibbs along z to ≤ 0.003 kT on both the density
     and mean-force routes. It fails at 0.002, 0.001 and 0.0005, and is marginal at 0.00025.
   * Independent checks agree with the MC arbiter: MALA (exact; acceptance 0.87 on the full system),
     BAOAB (γ = 1, correct configurational and momentum marginals at dt up to 0.005; velocity-Verlet
     energy error O(dt²) with no drift) and Leimkuhler–Matthews (second order, but it blows up without a
     clip at dt ≥ 0.0005).
4. **LTA passes, and its FR gain survives the exact reference.**
   * LTA's float64 EM at dt 2e-4 has no clip and sits in the stable regime (dt·λ_max ≈ 0.16-0.23). Its
     only measurable error is first order: +8.6 % bond variance and a ≤ 0.05 kJ/mol (0.02 kT) barrier
     shift.
   * The dynamics gate is ADMISSIBLE at 300 K and 150 K against umbrella Metropolis MC. The published
     reference is ADMISSIBLE at 150 K and MARGINAL at 300 K (0.034 kJ/mol, mostly its own sampling noise).
   * Rescored against exact Gibbs, FR vs ABF integrated error is **−13.7 % at 300 K and −28.5 % at
     150 K**. FR beats the matched sham directly (−13.0 %, −27.5 %), and the sham is neutral (−0.3 %,
     −1.0 %).
5. **The gateway FR gain survives timestep refinement.** Against the exact analytic F, FR vs ABF
   integrated error is −29.5 % / −28.4 % / −29.7 % at dt 4e-4 / 1e-4 / 2.5e-5 (30-32 of 32 seeds), so the
   26 % EM fibre-variance inflation at the accepted dt did not create it.
6. **Conditional WCA confirmatory at the validated dt** (ABF vs FR vs matched sham, exact reference):
   **FR_HARMFUL** (frozen rule; dt 0.000125, N 256, T 240, 16 fresh seeds; `WCA_CONFIRMATORY.md`):

   | contrast | integrated error I_F | final error e_F(T) |
   |---|---|---|
   | FR vs ABF | +9.9 % [+2.3, +13.2] (4/16 wins) | +131 % |
   | FR vs sham | +8.7 % [+1.1, +14.7] | |
   | sham vs ABF | −0.8 %, neutral | |

   FR's pooled long-run profile carries a stationary tilt of 0.017 kT relative to ABF. At N = 256 that tilt
   was 0.090 at dt 0.002 and 0.022 at dt 0.0005, so it does **not** vanish with dt: it is intrinsic to
   score-directed FR. The sham has none, so cloning per se does not cause it. **WCA is a validated
   negative control.**

## 2. What was run

| phase | experiment | outcome |
|---|---|---|
| I | independent float64 WCA model; 19 force-audit tests (gradient, Newton III, PBC, production equality, regularisation inconsistency, mean-force formula, harness bitwise equal to production, exact samplers vs analytic dimer density) | force correct; two non-gradient regularisations documented |
| II | 592 CPU jobs: production EM at 5 dt (64 seeds × 30720 t.u.), MC (64 × 3e7 sweeps), MALA (32), LM (3 dt), BAOAB (3 dt), regularisation diagnostics (5 dt), NVE energy check | dt* = 0.000125 |
| III | every historical WCA reference and pooled sampler limit vs exact Gibbs; historical FR runs rescored | v2 TI 0.29 kT off; accepted FR gain reverses |
| IV | LTA: independent numba model (4 tests); umbrella MC/EM/LM at 300 K and 150 K (6400 jobs); published runs rescored | ADMISSIBLE; FR gain survives |
| (IV′) | gateway dt refinement (96 jobs) | FR gain survives |
| V | WCA ABF / FR / sham at dt*, N 256, T 240, 16 fresh seeds (32 processes, 42 core-h) | FR_HARMFUL |

Engineering:

* `src/wca_validation.py`, `src/lta_validation.py`: new, independent models.
* `src/wca_numba.py`: matched-sham arm added, bitwise-inert for ABF/FR (40 existing tests pass, plus 3
  new).
* Every result file records commit, seed, dt, precision (float64), device (CPU) and versions.

## 3. Decision table

| scientific claim | status | evidence | remaining concern |
|---|---|---|---|
| **WCA original FR gain** (accepted −24 % / −51 %) | **Refuted** (as a free-energy acceleration) | Rescored vs exact Gibbs: +18.5 % / +34 %, 0/16 wins. FR's tilt matches the artefact reference, not the free energy (`WCA_REFERENCE_AUDIT.md` §5). | Every other dt 0.002 WCA study (kernel campaigns, FR-start ladder, movies, OT/repair) is **not established**; rescored here are the histogram confirmation and both ladders only. |
| **WCA dynamics** | **Fail** at the accepted dt 0.002; **Pass** at dt 0.000125 | Frozen gate vs MC + MALA (`WCA_LANGEVIN_VALIDATION.md`). | ⟨U⟩ (a stiff-contact observable) still carries a first-order +0.12 kT bias at dt*; admissibility covers F(z) only. Apparent convergence order 1.6 of the mean-force route is not resolved at the two smallest dt. |
| **WCA reference** | **Fail** (accepted v2 TI and v1 TI: 0.28-0.29 kT RMS); **Pass** for the new exact MC reference (noise 0.0006 kT; MALA agrees) | `WCA_REFERENCE_AUDIT.md` | The dt 0.0005 dt-consistent reference (0.0145 kT off) is not a physical reference either. |
| **WCA FR at the validated dt** | **Refuted / harmful** (FR_HARMFUL by the frozen rule: +9.9 % I_F, +131 % e_F(T); FR vs sham +8.7 %; sham neutral) | `WCA_CONFIRMATORY.md` | N = 256, not the accepted N = 1024 (cost rule fixed before dt* was known). The FR-tilt mechanism (fibre lag through score-directed cloning of transit walkers) is a hypothesis, not tested. |
| **LTA FR gain** | **Validated** at 300 K and 150 K (integrated error; the final error too at 150 K) | Dynamics ADMISSIBLE at both T; FR −13.7 % / −28.5 % vs exact reference, sham neutral, FR beats sham (`LTA_VALIDATION.md`) | 80, 225 and 350 K not re-validated; no equal-force-evaluation replica ladder for LTA; at 300 K all arms share a 0.2 kJ/mol finite-time ABF barrier bias that FR does not remove. |
| **Gateway FR gain** | **Existing evidence, now dt-robust** | −29.5 % / −28.4 % / −29.7 % at dt 4e-4 / 1e-4 / 2.5e-5 vs the exact analytic F | It is a constructed toy. At equal *force-evaluation* budget, serial ABF still beats 2048-replica ABF + FR (2026-10-04 ladder), so its gain is in parallel wall-clock, not in total compute. |
| **Alanine neutral result** | **Existing evidence** (not re-tested) | BAOAB 1 fs was validated at spec time by configurational temperature (T_conf error +0.08 ± 0.42 %, `ALANINE_SPEC.md` §2.4), and the reference shares (M, dt) with production | None from this audit. A dt artefact would not plausibly *create* neutrality. |

## 4. Answer to the research question

*After independently validating the physical dynamics and numerical references, what evidence remains
that marginal Fisher–Rao birth–death genuinely accelerates free-energy computations, under which
regimes, and at what computational cost?*

**What remains.** Two systems whose integrators and references now pass independent exact-sampling or
analytic checks:

* **Ethane in LTA**: −13.7 % integrated error at 300 K, −28.5 % at 150 K (−35 % at 80 K published, not
  re-validated).
* **The entropic gateway**: −28 to −30 % integrated, at every dt.

In both, the gain is attributable to the *direction* of the resampling: matched-turnover shams are neutral
(LTA; gateway shams in earlier campaigns).

**The regime.** In both cases ABF is *establishment-limited*: at the replica count and physical time
used, the walkers' marginal has not yet populated the bottleneck when the run starts accumulating
statistics. FR's benefit is a faster transient. It shrinks as establishment becomes easier (LTA's benefit
falls monotonically from 80 to 350 K), and it is absent where ABF converges smoothly:

* alanine: neutral across N, T, γ, CV and dose;
* the WCA dimer at a valid dt: FR is *harmful* there. ABF converges without an establishment
  bottleneck in the accepted physical time, and FR adds a 0.017 kT stationary bias.

FR does not remove ABF's own finite-time bias in the long-run limit at 300 K LTA. It does at 150 K, where
starvation is stronger.

**The cost.** Every positive is at **fixed replica count and fixed physical time**, i.e. equal force
evaluations between ABF and ABF + FR; FR's own overhead (a KDE and score every few steps) is small. The
saving is in **parallel wall-clock**, the time to accuracy of a given population, **not in total
force evaluations**. On the gateway and on WCA, the equal-budget replica ladders found that few-replica
or serial ABF reaches a given accuracy with fewer total force evaluations than 1024-2048-replica
ABF + FR. That ladder has not been run for LTA, so "FR reduces total compute" is **not established
anywhere**.

**What failed and why.** The WCA dimer, the project's most-cited positive, was a numerical artefact:

* an unstable time step;
* a non-conservative force clip that hid the instability;
* a reference built with the same integrator, so that every comparison measured agreement with the
  artefact.

The root cause is generic and worth stating in a paper: *a timestep-consistent reference cannot validate
a timestep*. The only reliable arbiters were exact samplers (Metropolis MC, MALA) and analytic references.

**What this means for the paper.** A credible claim is:

* "marginal FR accelerates the *establishment* phase of ABF in entropically or energetically starved,
  establishment-limited regimes, at equal force evaluations per replica and with negligible overhead";
* demonstrated on LTA (validated physical system) and the gateway (validated toy);
* with alanine and the validated WCA cell as negative/neutral controls;
* with the explicit caveat that at equal *total* compute a few-replica ABF can be better.

The WCA positive must be withdrawn and replaced by its validated-dt result, which is negative (FR harmful).

## 5. What remains uncertain

* WCA's FR stationary tilt is now shown to be intrinsic to FR. At N = 256 it went 0.090 → 0.022 → 0.017
  across dt 0.002 → 0.0005 → 0.000125, and the sham has none. *Why* the directed selection biases the
  conditional mean force (fibre-lag hypothesis) is not established. Nor is whether the tilt shrinks with
  N or with a better score (e.g. leave-one-out KDE).
* LTA at 80 / 225 / 350 K not re-validated (300 and 150 K bracket the mechanism); LTA equal-budget ladder
  not run.
* All LTA FR effects are at one FR rate per T, frozen before production. No retuning was done or
  considered.
* WCA dt*: the convergence order of the mean-force route is not resolved below dt 0.00025. Admissibility
  at 0.000125 rests on D ≤ 0.005 with bootstrap upper bounds 0.0074 / 0.0049, at about 1-1.5× the noise.

Next experiments: `NEXT_EXPERIMENTS.md`. Detailed documents:

* `WCA_CONFIRMATORY.md`

* `WCA_FORCE_AUDIT.md`
* `WCA_LANGEVIN_VALIDATION.md`
* `WCA_REFERENCE_AUDIT.md`
* `LTA_VALIDATION.md`

## 6. Tests

89 pass, 0 fail, 0 skipped (single-threaded CPU):

* new: `tests/test_wca_force_audit.py` (19), `tests/test_lta_validation.py` (4),
  `tests/test_wca_numba_sham.py` (3);
* existing, re-run after the sham was added: `tests/test_wca_numba.py` (40), `tests/test_gateway_numba.py`,
  `tests/test_lta_histogram.py`.

Passing tests show code equivalences and exact laws. They do not show correct statistics; that is what the
exact-sampler comparisons are for.

## 7. Process notes (honest record)

* **Tests that rejected correct code, fixed before any conclusion.** A χ² test over correlated histogram
  bins rejected the *correct* MC sampler. A numpy `a[idx] += 1` dropped duplicate indices in a law test.
  A z-convention error inverted the LTA barrier sign in a draft table.
* **Process interruptions.** A `pkill -f` pattern killed its own shell, so the engine test suite was
  re-run. A pytest run used 255 torch threads during the campaign and was killed and re-run
  single-threaded.
* **Decisions taken on frozen rules, not on outcomes.**
  * The LTA 300 K published reference is MARGINAL by the frozen rule, which treated its noise as zero;
    the noise attribution above is post-hoc.
  * The confirmatory used N = 256 by a rule fixed before dt* was known.
  * The execution was split into two processes per seed, verified bitwise identical to the 3-arm design
    and recorded as amendment 1.
