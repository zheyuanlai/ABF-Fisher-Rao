# Equal-budget replica ladders: ABF vs ABF + uniform Fisher–Rao. Final results

*2026-10-10. Branch `main`.*

| | |
|---|---|
| Preregistration | `SCIENTIFIC_PLAN.md` (commit ea14f11, before any production run) |
| Validation | `GATEWAY_TIMESTEP_VALIDATION.md`, `LTA_TIMESTEP_VALIDATION.md`, `IMPLEMENTATION_VALIDATION.md` |
| Log and ledger | `EXPERIMENT_LOG.md` |
| Interpretation | `SCIENTIFIC_INTERPRETATION.md` |
| Best-allocation confirmation (fresh seeds) | `CONFIRM_BEST_ALLOCATION.md` (prereg de3d20c, results f3dc7e2) |
| Appendix tables (absolute values, mean-based contrasts, τ in t and u, transients, discovery/establishment) | `APPENDIX_TABLES.md` (`scripts/equal_budget/report_tables.py`) |
| FR overhead breakdown | `FR_OVERHEAD.md` |
| Data locations and reproducibility | `DATA_LOCATIONS.md`, `results/equal_budget_v2/DATA_MANIFEST.json` |

All numbers below come from `results/equal_budget_v2/<system>/analysis/{summary.json,tables.md}`. Those were
written by `scripts/equal_budget/analyze_ladder.py` from the raw per-run arrays.

Conventions:
* **G = (FR − ABF)/ABF**, paired per seed: the median, with a 10 000-resample seed-bootstrap 95 % CI and the
  number of seeds in which FR is better. **Negative G means FR is better.**
* **Ī_F** is the free-energy error e_F averaged over the 200 uniform budget fractions u = b/B ∈ (0, 1].
* **Deviation from the task text, fixed at preregistration.** The task defines G_F(N) and the best allocation on
  seed *means*. The preregistration (`SCIENTIFIC_PLAN.md` §4, frozen before any data) chose the *median* of the
  per-seed paired ratios and seed *medians*, because the per-seed G is heavy-tailed at small N (SD up to 90 %).
* The mean-based versions are in `APPENDIX_TABLES.md` §2 (ratio of seed means, paired seed bootstrap) as a
  robustness check:
  * The large-N gains agree within a few points. The gateway N = 64 cell becomes significant by means
    (−27.6 % [−41.5, −11.3]).
  * LTA 300 K N = 32 becomes significantly harmful by means (+15.6 % [+6.1, +26.0]).
  * LTA 150 K N = 4 loses significance (+12.1 % [−1.6, +26.6]).
  * Mean-based best allocation, best FR − best ABF: −4.4 % [−12.0, +10.1] (300 K; FR's best moves to N = 4,
    P(FR better) = 0.68); +5.1 % [−6.9, +20.1] (150 K); +14.2 % [−1.3, +25.6] (gateway, P(FR better) = 0.04).
  * No conclusion changes. In particular no mean-based CI shows FR's best beating ABF's best.
  * The 300 K mean-based point estimate leans toward FR. The fresh-seed confirmation was preregistered at the
    median-selected N\* = 16, so it does not test FR at N = 4.
* **Further tables.** `APPENDIX_TABLES.md` also gives:
  * absolute values for every metric;
  * τ for every threshold, in physical time t and in budget fraction u;
  * per-N and per-run maximum transient improvement;
  * discovery vs establishment vs accumulated information: first arrivals, region occupation, TV_inst vs its
    finite-N floor, transitions.

  Round trips cannot be counted exactly from the saved data, because legs are not paired per walker.

## 0. Analysis pipeline

**Built, then adversarially reviewed and fixed, before reading any production data.** Code in
`scripts/equal_budget/`:
* `eqb_metrics.py`: frozen metrics;
* `analyze_ladder.py`: paired bootstrap and best-allocation bootstrap;
* `plot_config.py`: per-(N, T) figures A–F;
* `plot_synthesis.py`: S1–S8 and the cross-system X1;
* `audit_completeness.py`: the mandatory audit.

Tests: `tests/test_eqb_metrics.py` and `tests/test_audit_completeness.py`, 60 tests including an end-to-end run
on synthetic ladders.

The review raised 22 findings, all confirmed and fixed. Those that change how results are read:

* **F1, an unreachable gateway threshold.** The 180-bin P0 histogram estimator scored on the accepted
  function-space grid has a deterministic mean-force floor of **e_F′ = 0.0323** even when fed the exact
  bin-averaged mean force (it is a piecewise-constant read-out of a curved F′). So the frozen strict gateway
  threshold e_F′ ≤ 0.012 is **unreachable by construction**, and mid (0.035) leaves a statistical budget of only
  0.0136.
  * Per the prereg the thresholds were not changed. τ at strict e_F′ is reported as censored for every arm.
  * e_F′_stat = sqrt(e² − floor²) is added as a labelled descriptive companion.
  * The LTA mean-force reference is the MC per-bin conditional mean force itself, so the LTA floor is 0.
* **F2, establishment null calibration.** The instantaneous rule ("far-state fraction stays ≥ ½ target")
  fails *by chance* at small N even for an exactly uniform population. It is therefore marked UNRELIABLE where
  P(no chance failure in the second half) < 0.9: LTA N ≤ 64, gateway N ≤ 32. There the
  cumulative establishment and the first arrival are read instead.
* **F5, best-allocation ties.** τ is quantised on the save grid. Tied N now share the bootstrap credit; the
  earlier code gave N = 1 a spurious 0.54 share on identical data.
* **F7, heat-map colours.** Density 1 rendered salmon; the colormap is now tested on every draw.
* **Pairing, provenance and staleness checks** were added to the audit:
  * ABF/FR arms bitwise identical before FR's first event;
  * one engine build per system;
  * cache keyed by file sha256, config, scorer and reference;
  * stale summaries refused.

## 1. Executive summary

**Question.** At a fixed budget of force evaluations B = N × n_steps, how should ABF split work between the
number of replicas N and the per-replica time T_N = Bh/N? Does uniform-target marginal Fisher–Rao birth–death
(FR) change that answer?

**Design.**
* Three systems: the entropic gateway (12 N from 2048 to 1, 32 seeds) and ethane in LTA at 300 K and 150 K (11 N
  from 1024 to 1, 16 seeds each).
* Every N has the same B.
* Two algorithms only, histogram ABF and the same ABF + FR, paired per seed (shared initial conditions and
  Langevin noise).
* 1 408 production runs (336 + 336 + 736) plus 67 validation runs; none failed, none dropped.

**Numerics were validated before any ladder run.**
* The gateway timestep was refined to h_G = 2.5 × 10⁻⁵. The historical 4 × 10⁻⁴ samples the gate fibre 26 % too
  wide and fails the frozen gate.
* The LTA h = 2 × 10⁻⁴ was re-verified against exact Metropolis MC.
* All references are exact or MC and independent of the ladder.

**Result 1: at the same N, FR helps when N is large and T_N short.**

| system | N range | Ī_F gain | seeds |
|---|---|---|---|
| gateway | 2048–128 and 32 (N = 64 n.s.: −26 % [−46, +5], 21/32) | −22 to −41 % | 24–32/32 |
| LTA 150 K | 1024–256 | −29 to −34 % | 16/16 |
| LTA 300 K | 1024–128 | −10 to −13 % | 12–16/16 |

* In the gateway at N ≥ 128 and in LTA 150 K at N ≥ 256, FR brings the mid-accuracy free energy forward by 33–67 %. At gateway N = 64 and 32 the difference is not significant (−40 % [−63, +4] and −27 % [−60, +18]). At LTA 300 K the paired τ gain is not significant at any N (−2.8 % [−13.0, +0.6] at N = 1024), although by sign test FR is earlier at N = 1024 (10/2, p = 0.039; ABF censored in 7/16, FR in 4/16).
* At the largest N the gain persists to the end of the budget in the gateway (final e_F −50 to −61 % at
  N ≥ 512) and in LTA 150 K (−64 % at N = 1024). At LTA 300 K it is mostly transient.
* FR improves the reaction-coordinate marginal (trailing-half TV) more than the free energy at every N ≥ 32: −64 to −73 % (gateway), −27 to −40 % (150 K), −17 to −26 % (300 K). The margin is large in the gateway and at 300 K, but at LTA 150 K N ≥ 256 the TV gain (−34 to −39 %) is only 1.1–1.3× the F gain (−29 to −34 %).

**Result 2: at small N, FR is neutral on F and can be harmful.**
* Below N ≈ 16 FR is neutral on the free energy in all three systems. It is harmful at LTA 150 K N = 4 (+11.7 %
  [+1.8, +35.6]).
* In the gateway it damages the marginal it is designed to fix at N ≤ 8: integrated TV +53 to +140 %, 0–1/32
  seeds better.

**Result 3: FR does not change the best allocation.** A preregistered fresh-seed confirmation at the selected N\*
agrees: ties in LTA (FR advantage ≤ 1.2 % / ≤ 8.7 %), and ABF significantly better in the gateway (+24.0 %
[+1.1, +52.9]). At equal force evaluations, the best ABF-only allocation is
always a few long replicas (N = 2–16), and the best ABF + FR allocation is no better:

| system | best FR − best ABF (Ī_F) | best N |
|---|---|---|
| LTA 300 K | +3.4 % [−13.2, +12.4] | N = 16 for both |
| LTA 150 K | +5.2 % [−10.9, +24.1] | N = 2 for both |
| gateway | +15.9 % [−4.2, +26.6] | N = 16 for both; P(FR better) = 0.10 |

* ABF at the largest N is 8–26× worse than ABF at its best N, and FR closes only a small part of that gap.
* On the production seeds this is **outcome A in its weak form** in all three systems (the confirmation below makes
  the gateway strict): FR's best allocation does not beat ABF's best
  (P(best FR better) = 0.52 / 0.29 / 0.10 for 300 K / 150 K / gateway). The strict form ("ABF's best beats FR's
  best") is not resolved, and outcome B is not supported. It comes together with **C** (marginal improves without a matching F gain at
  intermediate N) and **D** (neutral or harmful at small N). **E** holds: the gain is larger at 150 K than at
  300 K, and the gateway's large small-N damage to the integrated marginal (+53 to +140 %) has no LTA counterpart; LTA 150 K shows only a final-TV_half penalty at N = 4 (+40 % [+6, +64], 4/16).

**Mechanism (tested, not assumed).**
* The large-N gain tracks how long ABF itself takes to establish the marginal. The cleanest test: at equal
  N ≥ 256, LTA 150 K establishes 2× later than 300 K under identical FR machinery, and gains 2.4–3.3× more.
* The gain vanishes where ABF reaches its own best error unaided (N ≤ 16–64).
* At N ≤ 8–16 the KDE score is noise (σ_KDE ≥ 1), and FR stops repairing, or damages, the marginal.
* A seed-level test of starvation agrees in LTA but has the *opposite* sign in the gateway (§8). So starvation is
  not a complete account there.

**Bottom line.** FR accelerates *per-replica physical time* for many-replica ABF that is short of it. In those
cells it is worth about 1.2–1.8× more per-replica time, i.e. wall-clock under ideal parallelism; that reading is an
inference and was not measured. It is *not* a force-evaluation accelerator. Total
cost: ≈ 165 CPU core-h in total (≈ 149 measured, including the confirmation and equivalence runs; `EXPERIMENT_LOG.md`)
of the 1 000 allowed, and 0 GPU-h.

## 2. Numerical correctness

**Force fields**
* **Gateway.** The analytic gradient equals central differences to < 1e-6 at 80 points across the gate.
* **Production force.** The force extracted from `gateway_numba.simulate` equals the analytic gradient to < 1e-7,
  and the validation EM chain is bitwise the production integrator.
* **LTA.** The independent model's forces, φ and local mean force equal `core_lta` to 1e-10 at both T.
* **Ladder engine.** `src/lta_ladder_numba.py` replays torch's own random draws to identical discrete outputs
  and 1e-9 floats. No new clip was added; the historical ±60 bias clip and ±480 deposit clip are unchanged.

**Timestep selection, gated without reading any ABF/FR outcome**
* **Gateway: h_G = 2.5 × 10⁻⁵.** It is the largest of {4e-4, 1e-4, 2.5e-5} that passes the frozen 2 %
  transverse-variance gate and the F and F′ gates.
  * The measured gate-centre fibre-variance inflation reproduces the closed form 1/(1 − ω²h/2) at every h:
    +26.0 % at 4e-4 (historical), +5.45 % at 1e-4, +1.35 % at 2.5e-5.
  * MALA gives 0 ± 0.08 %.
* **LTA: h_L = 2 × 10⁻⁴**, re-verified from the 2026-10-09 raw data (exact match).
  * EM vs exact Metropolis MC: density route D = 0.000 [upper 0.021] (300 K) and 0.000 [0.017] (150 K) kJ/mol;
    mean-force route 0.014 [0.029] and 0.011 [0.021].
  * The EM bond-variance bias (+8.6 %, textbook first order) does not reach F(φ) beyond 0.014 kJ/mol RMS.

**Gibbs and mean-force checks**
* Gateway: the flat-bias design gives −kT log p_h = F_h − F_exact. D(F) ≤ 0.0004 and D(F′) = 0.0008 at h_G,
  equal to the closed-form EM bias 0.0008.
* LTA: the MC conditional mean force has per-bin jackknife SE 0.0143 (300 K) / 0.0101 (150 K) kJ/mol/rad, and its
  circular mean is consistent with 0.

**Independent references, frozen before production**
* Gateway: the exact analytic F(x) = H(x² − 1)² + β⁻¹ log ω(x) and F′ on [−1.5, 1.5]. The EM-consistent profile
  is a secondary reference.
* LTA: fine-bin WHAM over 16 × 40 MC umbrella windows (F noise 0.0059 / 0.0035 kJ/mol), plus the MC per-bin mean
  force.
* sha256 of each reference is recorded in every summary.
* Units and gauge: LTA F in kJ/mol and F′ in kJ/mol/rad, F centred over the 180 bins. Gateway F in reduced
  energy (β = 16) and F′ in reduced energy per unit x, centred on the eval window.
* The frozen gateway reference file names only F's unit. The completeness audit flags this as a warning
  (REF_UNITS_IMPLICIT); it is documented here rather than by rewriting the frozen file.

**Remaining numerical uncertainty**
* The gateway EM floor at h_G is 8 × 10⁻⁵ in F, 25× below the strict threshold.
* The P0 read-out floor is 0.0323 in F′ (see §0 F1).
* LTA reference noise is 0.006 / 0.0035 kJ/mol in F, and the zero-noise scorer floor is 0.0075 / 0.0056 kJ/mol
  (the reference's own mean-force vs density-route consistency).
* The scorer floors are 4–5× below the LTA strict F threshold 0.03 (the reference noise is 5–9× below it), and 6–7× below the smallest Ī_F reported here (0.048 kJ/mol at 300 K, 0.041 at 150 K).
* Documentation defect: two barrier conventions for one quantity, 26.40 vs 26.79 kJ/mol at 300 K. The ladder
  never uses the scalar.

## 2b. Discovered implementation bugs and defects

None of these changed a production result of this campaign. Each was found before the affected data were read, or
it concerns historical code only.

**Historical code**
1. **`gateway_numba.simulate` with `ess_window ≤ 0`.** The ancestor labels are never initialised, the save block
   indexes garbage, and the run segfaults. Production always passed 4000, so no historical gateway result is
   affected. The new engine initialises the labels unconditionally. *Found:* Phase 0A gateway audit.
2. **`core_lta` torch-on-CPU aliasing** (`core_lta.py:373-374`, `:473`). `support_of(csum)` returns `csum` itself,
   and `.cpu().numpy()` on a CPU tensor shares memory, so every saved `eff_counts` entry equals the final csum
   (measured [976, 976, 976, 976], expected [16, 336, 656, 976]). CUDA production is correct. The numba engine
   stores copies. *Found:* Phase 0A LTA audit.
3. **Historical gateway FR drew its uniforms from the Langevin noise stream.** ABF and FR arms were therefore not
   noise-paired. The new engine uses a separate PCG64 FR stream (`IMPLEMENTATION_VALIDATION.md` §1).
4. **Historical LTA FR cap `int(0.02 N)` is 0 for N < 50.** It silently disables FR. The ladder uses the
   preregistered extension max(1, ⌊0.02N⌋), flagged at every N < 50.
5. **Documentation defect: two barrier conventions for one quantity.** 26.40 vs 26.79 kJ/mol at 300 K
   (`LTA_TIMESTEP_VALIDATION.md` §3). Not used by the ladder.

**Preregistration and analysis**

6. **Gateway strict e_F′ threshold (0.012) is unreachable by construction.** The P0 read-out floor is 0.0323.
   Found by the analysis-pipeline review before gateway data existed. Per the prereg it was not changed; it is
   reported as censored (§0 F1).
7. **Analysis-pipeline review: 22 findings, all fixed before production data were read** (§0). These include
   the uncalibrated establishment rule at small N, the best-allocation tie credit (a spurious 0.54 share for
   N = 1), the heat-map colour mapping, cache keys that ignored file content, and missing pairing/provenance checks.
8. **Engine review findings**, fixed before production (`IMPLEMENTATION_VALIDATION.md` §3):
   * the bitwise claim depended on the compile target;
   * a free deposit-clip knob;
   * per-run vs lifetime peak RSS;
   * slot- vs walker-following traces;
   * `run_job` could return another job's result;
   * a missing save-grid checkpoint.

**Process errors in this session**

9. **Tracked `tests/fixtures/` deleted.** An audit test wrote into it, and I then removed the directory with
   `rm -rf`. It was restored from git, verified, and the test paths moved to a temp directory.
10. **Wrong wait condition.** A background waiter tested for the wrong log line (the launcher appends
    `chain done`), which left 29 minutes of idle time after the gateway finished (`EXPERIMENT_LOG.md`).
11. **Final reports not version-controlled at first.** They were drafted under the gitignored `docs/*` and were
    not in any commit until the completeness critic caught it. They were then force-added.
12. **Report-verification pass.** Independent verifiers confirmed 62 overstated or wrong statements in the drafts:
    ranges that included non-significant cells, rounding, and outcome-A wording stronger than the CIs allow. All
    were corrected before the reports were committed.

## 3. Complete experiment inventory

Every planned (system, N, method, seed) completed. Nothing was dropped, re-budgeted or re-run. Cost is the
measured single-core wall time of each run from the per-run ledgers (`results/equal_budget_v2/<dir>/ledger.csv`);
core-h is the sum over seeds.

| system | T | N | T_N (t.u.) | h | walker-steps per run | method | seeds completed | status | core-h (sum) | wall per run (median s) |
|---|---|---|---|---|---|---|---|---|---|---|
| ethane/LTA | 300 K | 1024 | 60 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.21 | 275 |
| ethane/LTA | 300 K | 1024 | 60 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.23 | 273 |
| ethane/LTA | 300 K | 512 | 120 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.22 | 273 |
| ethane/LTA | 300 K | 512 | 120 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.24 | 280 |
| ethane/LTA | 300 K | 256 | 240 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.22 | 276 |
| ethane/LTA | 300 K | 256 | 240 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.25 | 282 |
| ethane/LTA | 300 K | 128 | 480 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.23 | 277 |
| ethane/LTA | 300 K | 128 | 480 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.26 | 283 |
| ethane/LTA | 300 K | 64 | 960 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.22 | 276 |
| ethane/LTA | 300 K | 64 | 960 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.25 | 282 |
| ethane/LTA | 300 K | 32 | 1920 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.23 | 277 |
| ethane/LTA | 300 K | 32 | 1920 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.26 | 284 |
| ethane/LTA | 300 K | 16 | 3840 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.22 | 275 |
| ethane/LTA | 300 K | 16 | 3840 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.26 | 284 |
| ethane/LTA | 300 K | 8 | 7680 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.23 | 277 |
| ethane/LTA | 300 K | 8 | 7680 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.29 | 290 |
| ethane/LTA | 300 K | 4 | 15360 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.24 | 278 |
| ethane/LTA | 300 K | 4 | 15360 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.33 | 299 |
| ethane/LTA | 300 K | 2 | 30720 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.22 | 276 |
| ethane/LTA | 300 K | 2 | 30720 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.40 | 315 |
| ethane/LTA | 300 K | 1 | 61440 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.24 | 279 |
| ethane/LTA | 150 K | 1024 | 60 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.23 | 278 |
| ethane/LTA | 150 K | 1024 | 60 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.25 | 281 |
| ethane/LTA | 150 K | 512 | 120 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.24 | 278 |
| ethane/LTA | 150 K | 512 | 120 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.26 | 282 |
| ethane/LTA | 150 K | 256 | 240 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.23 | 276 |
| ethane/LTA | 150 K | 256 | 240 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.25 | 279 |
| ethane/LTA | 150 K | 128 | 480 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.24 | 278 |
| ethane/LTA | 150 K | 128 | 480 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.26 | 284 |
| ethane/LTA | 150 K | 64 | 960 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.24 | 279 |
| ethane/LTA | 150 K | 64 | 960 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.27 | 285 |
| ethane/LTA | 150 K | 32 | 1920 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.24 | 278 |
| ethane/LTA | 150 K | 32 | 1920 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.27 | 286 |
| ethane/LTA | 150 K | 16 | 3840 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.23 | 276 |
| ethane/LTA | 150 K | 16 | 3840 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.28 | 287 |
| ethane/LTA | 150 K | 8 | 7680 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.24 | 278 |
| ethane/LTA | 150 K | 8 | 7680 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.30 | 292 |
| ethane/LTA | 150 K | 4 | 15360 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.24 | 280 |
| ethane/LTA | 150 K | 4 | 15360 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.34 | 301 |
| ethane/LTA | 150 K | 2 | 30720 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.23 | 278 |
| ethane/LTA | 150 K | 2 | 30720 | 0.0002 | 3.072e+08 | ABF+FR | 16/16 | complete | 1.40 | 316 |
| ethane/LTA | 150 K | 1 | 61440 | 0.0002 | 3.072e+08 | ABF | 16/16 | complete | 1.24 | 280 |
| entropic gateway | -- | 2048 | 40 | 2.5e-05 | 3.277e+09 | ABF | 32/32 | complete | 2.39 | 267 |
| entropic gateway | -- | 2048 | 40 | 2.5e-05 | 3.277e+09 | ABF+FR | 32/32 | complete | 2.41 | 269 |
| entropic gateway | -- | 1024 | 80 | 2.5e-05 | 3.277e+09 | ABF | 32/32 | complete | 2.36 | 264 |
| entropic gateway | -- | 1024 | 80 | 2.5e-05 | 3.277e+09 | ABF+FR | 32/32 | complete | 2.38 | 267 |
| entropic gateway | -- | 512 | 160 | 2.5e-05 | 3.277e+09 | ABF | 32/32 | complete | 2.34 | 262 |
| entropic gateway | -- | 512 | 160 | 2.5e-05 | 3.277e+09 | ABF+FR | 32/32 | complete | 2.34 | 263 |
| entropic gateway | -- | 256 | 320 | 2.5e-05 | 3.277e+09 | ABF | 32/32 | complete | 2.35 | 262 |
| entropic gateway | -- | 256 | 320 | 2.5e-05 | 3.277e+09 | ABF+FR | 32/32 | complete | 2.35 | 262 |
| entropic gateway | -- | 128 | 640 | 2.5e-05 | 3.277e+09 | ABF | 32/32 | complete | 2.32 | 261 |
| entropic gateway | -- | 128 | 640 | 2.5e-05 | 3.277e+09 | ABF+FR | 32/32 | complete | 2.35 | 263 |
| entropic gateway | -- | 64 | 1280 | 2.5e-05 | 3.277e+09 | ABF | 32/32 | complete | 2.33 | 259 |
| entropic gateway | -- | 64 | 1280 | 2.5e-05 | 3.277e+09 | ABF+FR | 32/32 | complete | 2.34 | 261 |
| entropic gateway | -- | 32 | 2560 | 2.5e-05 | 3.277e+09 | ABF | 32/32 | complete | 2.35 | 260 |
| entropic gateway | -- | 32 | 2560 | 2.5e-05 | 3.277e+09 | ABF+FR | 32/32 | complete | 2.38 | 263 |
| entropic gateway | -- | 16 | 5120 | 2.5e-05 | 3.277e+09 | ABF | 32/32 | complete | 2.32 | 260 |
| entropic gateway | -- | 16 | 5120 | 2.5e-05 | 3.277e+09 | ABF+FR | 32/32 | complete | 2.36 | 265 |
| entropic gateway | -- | 8 | 10240 | 2.5e-05 | 3.277e+09 | ABF | 32/32 | complete | 2.37 | 263 |
| entropic gateway | -- | 8 | 10240 | 2.5e-05 | 3.277e+09 | ABF+FR | 32/32 | complete | 2.45 | 272 |
| entropic gateway | -- | 4 | 20480 | 2.5e-05 | 3.277e+09 | ABF | 32/32 | complete | 2.35 | 265 |
| entropic gateway | -- | 4 | 20480 | 2.5e-05 | 3.277e+09 | ABF+FR | 32/32 | complete | 2.47 | 278 |
| entropic gateway | -- | 2 | 40960 | 2.5e-05 | 3.277e+09 | ABF | 32/32 | complete | 2.42 | 264 |
| entropic gateway | -- | 2 | 40960 | 2.5e-05 | 3.277e+09 | ABF+FR | 32/32 | complete | 2.63 | 291 |
| entropic gateway | -- | 1 | 81920 | 2.5e-05 | 3.277e+09 | ABF | 32/32 | complete | 2.38 | 266 |

**Validation and auxiliary runs** (not part of the ladders):

| item | runs | cost | data |
|---|---|---|---|
| gateway timestep validation (EM at 3 h, MALA, exact) | 67 | 3.5 core-h | `results/equal_budget_v2/gateway_validation/` |
| LTA numba-vs-production equivalence: N = 1024, 300 / 150 K × ABF / FR × 32 seeds, plus 48 torch-draw runs | 176 | 13.6 core-h | `results/equal_budget_v2/equivalence/` (copied from the session scratchpad; README) |
| LTA smoke test under live ABF bias (both T, N = 1024 and N = 1) | 4 | < 0.1 core-h | none saved; printed values in the transcript |
| best-allocation fresh-seed confirmation (N\* only) | 256 | 22.1 core-h | `results/equal_budget_v2/confirm_best_*/` |
| FR-overhead benchmarks (short pinned runs) | ~300 short | ≈ 1.5 core-h | `results/equal_budget_v2/fr_overhead/` |
| regeneration and bitwise-reproducibility checks (data manifest) | 7 | 0.6 core-h | scratchpad |

**Data location.** The raw per-run arrays (1 408 files, 1.16 GB) are gitignored.
* They live on the compute host under `results/equal_budget_v2/<system>/N<N>/`, with sha256 values in
  `results/equal_budget_v2/DATA_MANIFEST.json`.
* `docs/equal_budget/DATA_LOCATIONS.md` gives every data class, what is in git, its size and the exact
  regeneration commands.
* Re-running a job reproduces its arrays bit for bit (checked on 7 jobs). Only the meta (wall time, timestamps)
  differs.

## 4. Gateway results

Figures:
* `figures/equal_budget_v2/gateway/N<N>/` (19 per N: A–F; the projected-F′ panel B4 is LTA-only);
* `figures/equal_budget_v2/gateway/synthesis/` (S1–S8);
* `figures/equal_budget_v2/cross_system/synthesis/X1_cross_system_gain.png`.

Units are reduced: F in model energy (β = 16), F′ in energy per unit x.

**Free energy.** FR lowers Ī_F significantly at every N from 2048 down to 32 except N = 64, where the lean (−26.3 % [−45.6, +4.6], 21/32) is not significant. It is neutral from 16 down to 2.

| N (T_N) | ABF Ī_F | FR Ī_F | G Ī_F % | G final e_F % |
|---|---|---|---|---|
| 2048 (40) | 0.0227 | 0.0156 | **−31.4** [−34.7, −27.7] 31/32 | **−61.4** [−65.1, −55.2] 32/32 |
| 1024 (80) | 0.0141 | 0.0090 | **−35.8** [−39.4, −26.8] 29/32 | **−56.9** [−62.6, −49.7] 32/32 |
| 512 (160) | 0.0083 | 0.0050 | **−41.0** [−49.6, −33.4] 32/32 | **−50.5** [−68.0, −36.4] 26/32 |
| 256 (320) | 0.0043 | 0.0029 | **−26.9** [−40.8, −22.5] 28/32 | −6.3 [−36.5, +17.8] |
| 128 (640) | 0.0027 | 0.0018 | **−30.3** [−37.4, −17.6] 24/32 | +15.3 [−25.3, +46.3] |
| 64 (1280) | 0.0015 | 0.0013 | −26.3 [−45.6, +4.6] 21/32 | +26.4 [−0.6, +55.0] 11/32 |
| 32 (2560) | 0.0013 | 0.0010 | **−22.3** [−31.0, −4.5] 24/32 | −2.7 [−20.4, +32.1] |
| 16 (5120) | **0.00087** | 0.0010 | +10.9 [−12.5, +44.8] 14/32 | +10.8 [−13.5, +42.1] |
| 8 | 0.00092 | 0.0011 | −1.7 [−21.8, +29.3] | −16.9 [−48.7, +24.0] |
| 4 | 0.00090 | 0.0011 | +13.0 [−14.6, +27.0] 11/32 | −15.1 [−40.8, +46.3] |
| 2 | 0.00094 | 0.0011 | +13.9 [−7.0, +25.6] 11/32 | −9.4 [−40.7, +47.0] |
| 1 | 0.00092 | — | — | — |

ABF at N = 1 has final e_F 0.00038.

**Persistence**
* At N ≥ 512 the improvement of the seed-median e_F curve persists to the end of the budget: +54 to +59 % at u = 1,
  with a peak of +76 %.
* At N = 128–256 it peaks at +44 to +54 % and falls to +9 % at u = 1, i.e. mostly transient.
* At N = 64 the endpoint leans worse (+26 %, CI touching 0). This is the same intermediate-N pattern as LTA 150 K,
  where it is significant at N = 64.

**Time to accuracy**
* Mid threshold (0.0055): at N = 2048 ABF is censored in 14/32 seeds, while FR reaches it in every seed
  (median u = 0.32; 32/0 wins).
* Mid threshold at N = 1024–128: FR is 33–67 % earlier (23–32/32 wins). At N ≤ 64 there is no significant
  difference.
* Strict threshold (0.002): FR is 62 % earlier at N = 512 (28/4), 46 % at 256 (25/7) and 35 % at 128 (22/10,
  p = 0.05).

**Mean force.** e_F′ is floor-dominated, so read the paired contrasts and the floor-free companion.
* Final e_F′ is 0.0323–0.0345 at every N. At least 93.6 % of that is the P0 read-out floor 0.0323 (§0 F1), and strict
  τ is censored by construction.
* Paired Ī_F′: −20.9 % (2048), −14.7 % (1024), −12.3 % (512), −7.8 % (256), −4.4 % (128), −8.9 % (64), −0.7 %
  (32), each with ≥ 23/32 seeds. It is ≈ 0 and n.s. at N = 16–4 (−0.0 to −0.2 %) and +0.3 % [+0.0, +0.6] at N = 2.
* Floor-free companion e_F′_stat (descriptive):
  * integrated: −20 to −35 % at N = 2048–64 (−29.9 % [−34.6, −26.3] at N = 2048), then −14.7 % [−22.1, −2.5] at N = 32; n.s. at N = 16–4;
    **+9.8 % [+0.3, +19.7]** at N = 2;
  * final: −57 %, −49 % and −30 % at N = 2048, 1024 and 512; n.s. at N = 256–4; **+7.4 % [+1.6, +26.8]** at N = 2.
* τ(e_F′ mid) is 28–64 % earlier for FR at N ≥ 128.

**Reaction-coordinate establishment.** This is FR's largest effect and its largest sign change. The mean-force metrics also turn slightly but significantly positive at N = 2 (Ī_F′ +0.3 %; e_F′_stat +9.8 % integrated, +7.4 % final).
* At N ≥ 32 the integrated TV_half is −64 to −73 % (32/32); at N = 16 it is −37 %. τ(TV_half ≤ 0.1) is 75–85 %
  earlier at N ≥ 16.
* ABF at N = 2048 barely establishes the marginal within its budget: τ = 0.96 of B, censored in 6/32 seeds. FR
  establishes it at 0.17.
* **At N ≤ 8 FR damages the marginal.** Integrated TV_half is +56 % (N = 8, 1/32), +140 % (N = 4, 0/32) and +53 %
  (N = 2, 0/32); final TV_half is +167 %, +386 % and +84 %. This happens with 5 400–23 700 deaths per walker.
* At N = 8, FR reaches TV_half ≤ 0.1 earlier but then plateaus at about 0.04 while ABF continues to 0.016.
* ABF itself establishes within u ≤ 0.06 at every N ≤ 128.
* Outcome C holds at N = 16 (TV −37 %, F n.s.).

**Population and genealogy.**
* Deaths per walker are 2.6 at N = 2048, rising to 150 (N = 128), 2 670 (N = 16) and 23 700 (N = 2).
* Whole-run unique ancestors go from 426 at N = 2048 to 1 at every N ≤ 128. The windowed ESS is 0.21–0.42 at N ≥ 4 (0.50 at N = 2).
* ABF gate transitions per budget are N-independent, ≈ 2 170–2 270. FR at N ≤ 8 has more (2 360–2 730), because
  copies are made inside the gate region.

**Best allocation.**
* ABF best is N = 16 with Ī_F 0.000865 [0.000778, 0.000912]; frequencies 16: 0.40, 2: 0.17, 1: 0.17, 4: 0.14,
  8: 0.13.
* FR best is also N = 16, 0.00100 [0.00082, 0.00104].
* Best FR − best ABF = **+15.9 % [−4.2 %, +26.6 %]**, P(best FR < best ABF) = 0.10. Final e_F: +15.6 % [−14.7, +52.6].
  τ mid: tied.
* FR's best is better only on the marginal's τ: −83 % [−86, −78].
* ABF at N = 2048 is 26× worse than ABF at its best. FR at N = 2048 is 18× worse than the best ABF.
* **Outcomes A (weak form: the point estimate favours ABF, P(best FR better) = 0.10, CI includes 0; not B) + C + D.**
  This matches the 2026-10-04 ladder run at the historical h, now at the validated
  timestep.

## 5. LTA 300 K results

Figures:
* `figures/equal_budget_v2/lta_T300/N<N>/` (A–F per N);
* `figures/equal_budget_v2/lta_T300/synthesis/` (S1–S8).

**Free energy.** FR lowers Ī_F at every N ≥ 128 (16/16 seeds at N ≥ 512). Below that it is neutral, and at N = 32
borderline harmful.

| N (T_N) | 1024 (60) | 512 (120) | 256 (240) | 128 (480) | 64 (960) | 32 (1920) | 16 (3840) | 8 | 4 | 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| ABF Ī_F (kJ/mol) | 0.396 | 0.229 | 0.135 | 0.089 | 0.063 | 0.057 | **0.048** | 0.051 | 0.058 | 0.057 |
| FR Ī_F | 0.344 | 0.208 | 0.123 | 0.078 | 0.062 | 0.064 | 0.050 | 0.051 | 0.052 | 0.055 |
| G Ī_F % | **−12.5** [−13.9, −10.5] 16/16 | **−10.5** [−17.5, −7.7] 16/16 | **−9.5** [−24.8, −3.6] 12/16 | **−11.9** [−19.8, −3.6] 12/16 | −4.5 [−17.6, +19.8] | +10.8 [−0.6, +35.2] 5/16 | −1.4 [−10.2, +22.2] | +4.2 [−16, +18] | −1.5 [−13.6, +2.8] | +4.0 [−5.8, +6.6] |
| G final e_F % | −6.7 [−16.1, +2.3] | −9.0 [−16.8, +8.8] | +1.2 [−20, +20] | −11.0 [−32, +12] | +7.5 [−25, +32] | +5.5 [−17, +24] | +5.2 [−22, +25] | +5.2 | −17.1 [−26.9, +4.5] | +3.9 |

ABF at N = 1: Ī_F 0.052, final e_F 0.027.

At N = 1024 the improvement is mostly **transient**:
* The seed-median e_F curve of FR is up to 31 % below ABF (at u = 0.33), but only 9 % below at u = 1.
* The paired endpoint gain is not significant at any N.
* τ(e_F ≤ 0.094, the mid threshold) at N = 1024: ABF censored 7/16, FR 4/16. FR is earlier in 10 seeds and later
  in 2 (sign p = 0.039).
* The strict threshold (0.03) is never met by any N = 1024 arm.

**Mean force.**
* Ī_F′ gain is smaller: −7.8 % [−8.4, −5.1] (1024), −6.2 % (512), −5.8 % (256), −3.0 % [−7.5, −0.0] (128).
* At N = 32 FR is worse: +4.0 % [+0.2, +13.1], 4/16.
* No final-e_F′ contrast is significant.
* The strict e_F′ threshold (0.04) is censored for every arm at every N. Final e_F′ ≈ 0.08 at every N ≤ 128 is the
  statistical limit of this budget: 5–6× the reference SE and independent of N and method.

**Reaction-coordinate establishment.**
* FR lowers the integrated trailing-half TV to uniform by 17–26 % at every N ≥ 32 (13–16/16 seeds). It also brings
  τ(TV_half ≤ 0.1) forward by 20 % at N ≥ 256 (16/16).
* The effect fades at N = 16 (−8.7 % [−17.3, +1.6]) and is absent at N ≤ 8.
* ABF itself establishes the marginal at u = 0.20 of the budget at N = 1024 (12 t.u.), at 0.05 at N = 256, and at
  ≤ 0.03 at N ≤ 128.
* At N = 32–64 FR improves the marginal by 18 % while F is unchanged or worse: **outcome C** in that window.

**Population and genealogy.**
* FR realises 0.64 deaths per walker at N = 1024, rising to 112–154 per walker at N = 8–16 (cap 1 per
  opportunity from the finite-N extension).
* Whole-run unique ancestors collapse to 1 at N ≤ 32; the windowed ESS stays 0.5–0.8.
* Total gate transitions per budget are N-independent: ≈ 1100 per arm per seed (920 at N = 1024, where the
  initial transient costs crossings).

**Best allocation.**
* ABF best is N = 16 with Ī_F 0.0482 [0.0425, 0.0518]; bootstrap N frequency 16: 0.42, 8: 0.24, 2: 0.18.
* FR best is also N = 16, 0.0498 [0.0412, 0.0523].
* Best FR − best ABF = **+3.4 % [−13.2 %, +12.4 %]**, P(best FR < best ABF) = 0.52.
* Final e_F: +0.7 % [−17.6, +20.7]. τ mid: +31 % [−29, +73].
* **Outcomes A (weak form: a best-N tie, +3.4 % [−13.2, +12.4], P(best FR better) = 0.52; not B) + C + D.**

## 6. LTA 150 K results

Figures: `figures/equal_budget_v2/lta_T150/N<N>/` and `.../lta_T150/synthesis/`.

**Free energy.** The large-N gain is 2.4–3.3× larger than at 300 K and is persistent at N = 1024. FR is harmful at
N = 4 and neutral at N = 2.

| N (T_N) | 1024 (60) | 512 (120) | 256 (240) | 128 (480) | 64 (960) | 32 (1920) | 16 (3840) | 8 | 4 | 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| ABF Ī_F (kJ/mol) | 0.440 | 0.281 | 0.165 | 0.086 | 0.064 | 0.056 | 0.047 | 0.044 | 0.041 | **0.041** |
| FR Ī_F | 0.304 | 0.185 | 0.124 | 0.079 | 0.055 | 0.046 | 0.054 | 0.048 | 0.045 | 0.043 |
| G Ī_F % | **−30.6** [−32.8, −29.7] 16/16 | **−34.5** [−35.9, −31.0] 16/16 | **−29.1** [−34.5, −24.2] 16/16 | −8.5 [−24.7, +6.7] | **−15.7** [−21.9, −5.7] 12/16 | −11.5 [−20.9, +4.8] | +16.3 [−11.7, +29.7] 6/16 | +25.6 [−10.2, +34.9] 6/16 | **+11.7** [+1.8, +35.6] 4/16 | −7.4 [−15.8, +13.3] |
| G final e_F % | **−63.6** [−67.4, −56.8] 16/16 | **−35.0** [−45.3, −26.1] 15/16 | **−15.7** [−30.5, −1.0] 12/16 | +27.5 [−2.5, +63.3] 5/16 | **+15.9** [+5.6, +66.4] 3/16 | −16.2 [−27.8, +11.2] | +0.3 | −4.5 | +9.4 | −1.1 |

ABF at N = 1: Ī_F 0.044, final e_F 0.022.

At N = 1024 the gain persists to the end of the budget:
* The median-curve improvement peaks at 67 % (u = 0.49) and is still 64 % at u = 1.
* τ(e_F ≤ 0.13, mid): FR reaches it at u = 0.385 in all 16 seeds; ABF is censored in 8/16 (16/0 wins, −58 %).
* At N = 512, 256 and 64 FR also reaches the mid threshold 44–61 % earlier (13–16/16 seeds). At N = 128 the
  difference is not significant (−29 % [−45, +12]).
* **The endpoint flips sign at N = 64 (significant) and leans that way at N = 128.** FR's final e_F is worse at
  N = 64 (+15.9 %, CI excludes 0, 3/16) and N = 128 (+27.5 %, n.s.), although its Ī_F is lower. At N = 64 the
  early advantage is paid for at the end.

**Mean force.**
* Ī_F′: −22.4 % (1024), −20.0 % (512), −14.9 % (256), all ≥ 15/16; neutral at N ≤ 128.
* Final e_F′: −49.0 % [−53.5, −46.4] at N = 1024, 16/16.
* Strict e_F′ (0.04) is censored everywhere. Final e_F′ ≈ 0.062–0.076 at N ≤ 128.

**Reaction-coordinate establishment.**
* FR lowers the integrated TV_half by 27–40 % at every N ≥ 32 (16/16), and by 21 % at N = 16.
* It is neutral at N ≤ 8. At N = 4 FR is **worse** on the final TV_half: +40 % [+6, +64], 4/16.
* ABF reaches τ(TV_half ≤ 0.1) at u = 0.40 (N = 1024, 24 t.u.), 0.21 (512), 0.105 (256), 0.05 (128) (instantaneous establishment rule: 0.19, 0.09, 0.05, 0.0275) and
  0.03–0.055 below.
* Outcome C holds at N = 128 and N = 16: the marginal improves by 40 % / 21 %, while F does not improve.

**Population and genealogy.** As at 300 K: 0.77 deaths per walker at N = 1024 and about 150 at N = 8. Unique
ancestors collapse at N ≤ 32. Transitions per budget are ≈ 500, N-independent apart from the large-N transient
(326 at N = 1024).

**Best allocation.**
* ABF best is N = 2 with Ī_F 0.0408 [0.0333, 0.0435]; frequencies 4: 0.47, 2: 0.35, 1: 0.09, 8: 0.09.
* FR best is N = 2, 0.0429 [0.0362, 0.0452].
* Best FR − best ABF = **+5.2 % [−10.9 %, +24.1 %]**. Final e_F: +11.6 % [−11.9, +47.0]. τ mid tied: 0 %
  [−36, +71].
* Only the marginal's τ(TV_half) is better for FR at its best N: 32 vs ABF's 4, −42 % [−50, −20]. That gain does
  not carry over to F.
* **Outcomes A (weak form: +5.2 % [−10.9, +24.1], P(best FR better) = 0.29; not B) + C + D, and E relative to
  300 K** (the same-N gain is larger at 150 K).

## 7. Force-evaluation versus physical-time efficiency

**At equal force evaluations, the best ABF+FR allocation is not better than the best ABF-only allocation in
either LTA system.** In both, the optimum lies at small N, where FR has nothing to repair:
* 300 K: +3.4 % [−13.2, +12.4]; 150 K: +5.2 % [−10.9, +24.1].
* The CIs exclude a FR advantage larger than 13 % (300 K) and 11 % (150 K).
* Gateway: +15.9 % [−4.2, +26.6]. Its CI excludes a FR advantage larger than 4.2 %, and the point estimate favours ABF.

**Fresh-seed confirmation at the selected best N** (preregistered in `CONFIRM_BEST_ALLOCATION.md`, frozen at
de3d20c before running). The best-allocation bootstrap re-selects N but re-uses the production seeds. Its top
frequencies are only 0.40–0.47, so the paired contrast was re-run at N\* (the N both arms selected) with fresh
seeds, which removes the winner's curse:

| system | N\* | fresh seeds | G Ī_F | G final e_F | verdict |
|---|---|---|---|---|---|
| LTA 300 K | 16 | 32 | +6.2 % [−1.2, +22.4], 11/32 | +6.3 % [−24.7, +31.4] | tie; FR advantage ≤ 1.2 % |
| LTA 150 K | 2 | 32 | −3.0 % [−8.7, +3.3], 19/32 | −3.2 % [−17.5, +13.0] | tie; FR advantage ≤ 8.7 % |
| gateway | 16 | 64 | **+24.0 % [+1.1, +52.9]**, 22/64 | **+21.4 % [+2.0, +51.9]** | **ABF better** (Ī_F′_stat +12.0 % [+4.5, +17.2]; marginal still −31 %) |

At the best allocation, FR never beats ABF on fresh seeds. In the gateway it is significantly worse, even though
it still repairs the marginal there.

**What FR buys is per-replica physical time at large N.** Under ideal parallelism (one core per replica,
negligible communication), wall time is ∝ n_steps = B/N, so the large-N allocation finishes fastest. **This
wall-clock reading is an inference and was not measured.** The measured single-core time per run is nearly the
same at every N (≈ 260–300 s, §3), as it must be at equal B. The large-N gains are large:
* At 150 K, FR at N = 1024 (T_N = 60) reaches Ī_F 0.304. ABF reaches 0.281 at N = 512 (T_N = 120). Log-linear
  interpolation puts ABF's match at N ≈ 580, i.e. about 1.8× the per-replica physical time at the same force
  budget.
* At 300 K the equivalent factor is about 1.2× (ABF match at N ≈ 860).
* But ABF at N = 1024 is 8–11× worse than ABF at its best N. FR closes only a small part of that gap.

**Overhead.** FR's KDE, score and resampling cost is not in the walker-step budget. It was measured on pinned
cores (`FR_OVERHEAD.md`), with diagnostics off and repeated 5 times:

| system | N = 2 | N = 16 | large N |
|---|---|---|---|
| LTA | +13.9 % | +3.3 % | +1.1 % (N = 1024; about +1.3 % with diagnostics on and the production FR schedule) |
| gateway | +8.7 % | +1.5 % | +0.3 % (N = 2048) |

* FR costs about a + b·N per opportunity, so the relative overhead falls as 1/N toward about 1 % (LTA) and 0.2 %
  (gateway).
* At small N the cost is dominated by the KDE and score grid loops; resampling is ≤ 0.6 % everywhere.
* The diagnostics, which both arms carry, cost 0.8–3.1 %.
* The production-ledger ratios at large N (e.g. "−0.7 %" at LTA 300 K, N = 1024) were noise from 110 concurrent
  processes and are superseded by these measurements. In the gateway, FR at N = 2048 or 512 is matched by ABF at about 1.7× the per-replica time.

## 8. Interpretation of finite-N FR behaviour

LTA first, then the gateway. The tests are in `SCIENTIFIC_INTERPRETATION.md` §2 (`scripts/equal_budget/mechanism_analysis.py`
→ `results/equal_budget_v2/mechanism.json`).

| hypothesis | verdict |
|---|---|
| establishment starvation | **supported at large N in LTA; seed-level test contrary in the gateway** |
| insufficient independent exploration at small N | **refuted** |
| KDE score noise | **supported at N ≤ 8–16; strongest in the gateway (marginal damaged at N ≤ 8)** |
| excessive birth–death | **plausible contributor to small-N harm (LTA 150 K; gateway marginal), confounded with KDE noise** |
| loss of genealogy diversity | **not supported as a primary mechanism** |
| long-time ABF convergence without FR | **supported as a reason FR is neutral at small N; separated from KDE noise only at N = 16–64 (test M)** |

**Establishment starvation: supported at large N at 150 K and by the same-N cross-temperature test.** At 150 K, FR's F gain is large (−29 to −34 %) where ABF needs ≥ 10 % of the budget to establish the marginal (N ≥ 256) and smaller below that (N = 128: −8.5 %, n.s.; N = 64: −15.7 %). At 300 K the gain is flat at −9.5 to −12.5 % from N = 1024 (ABF 20 %) to N = 128 (ABF 3 %), so across N it does not follow ABF's establishment time there. At 300 K, starvation rests on the same-N cross-temperature test and the within-N seed test. Two tests discriminate this from the other mechanisms:
* **Same N, different temperature.** At N ≥ 256 the KDE noise is identical, deaths per walker agree within 21 %
  (within 4 % at N = 256–512), and ABF at 150 K establishes 2× later. The 150 K gain is 2.4–3.3× larger.
* **Within N, across seeds.** At N ≥ 128 the seeds whose ABF arm establishes late gain the most from FR: mean
  Spearman ρ = −0.35 (p = 0.008, 300 K) and −0.29 (p = 0.026, 150 K). The effect is absent at small N. Part of
  this correlation is mechanical, because G has ABF in its denominator.

**Insufficient independent exploration at small N: refuted.** Transitions per budget are N-independent, ABF
reaches its best error at N = 2–16, and a single LTA walker crosses at least as often as 1024 do (503 vs 326 transitions at 150 K, 1142 vs 920 at 300 K).

**KDE score noise: supported as the reason FR stops repairing the marginal at N ≤ 8–16.**
* The relative KDE noise σ_KDE exceeds 1 at N ≤ 16.
* FR's TV improvement, the thing the score is built to produce, decays from −17..−27 % at N = 32 to 0 at N ≤ 8.

**Excessive birth–death: plausible contributor to the small-N harm at 150 K only.**
* There are 110–150 deaths per walker at N = 4–16. At N = 4–8 they bring no marginal benefit; at N = 16 the integrated TV_half still improves by 21 %.
* FR is harmful at N = 4 (+11.7 %, CI excludes 0) and leans harmful at N = 8–16.
* The same death counts are neutral at 300 K.

**Loss of genealogy diversity: not supported as a primary mechanism.** Whole-run collapse to one ancestor occurs
at N ≤ 32 in both systems (two founders remain at N = 64), and FR still helps there at 150 K (N = 32: integrated TV_half −27 %, 16/16, Ī_F −11.5 % n.s.; N = 64: Ī_F −15.7 %, CI excludes 0). The median windowed ESS stays ≥ 0.5 at every N; single seeds dip to 0.40 (150 K, N = 4) and 0.36 (300 K, N = 8).

**Long-time ABF convergence without FR: supported as a reason FR is neutral at small N.** At N = 16–64 test M separates
it from KDE noise: the marginal still improves while F does not. At N ≤ 8 it is not separated from KDE noise or
birth–death excess. At N ≤ 64 (300 K) / N ≤ 32 (150 K):
* ABF is within 1.0–1.4× of its best Ī_F;
* ABF establishes the marginal within ≤ 2.5 % (300 K) / ≤ 5.5 % (150 K) of the budget.

**Gateway** (`SCIENTIFIC_INTERPRETATION.md` §2.3)

* **Starvation.** Across N the F gain persists (significant at every N except 64, where it is −26 % n.s.) from N = 2048 (ABF establishment at 0.96 of the budget) down to
  N = 32 (0.03 of the budget, ABF at 1.5× its floor), and vanishes at N ≤ 16, where ABF is at its floor.
* **The seed-level test W has the opposite sign.** At N ≥ 256, ρ = **+0.21, p = 0.019**: seeds whose ABF arm
  establishes later gain *less*. The shared-denominator artefact would push ρ negative, so this is a genuine
  contrary signal. Seed-level starvation is **not** supported in the gateway.
* **KDE noise.** σ_KDE > 1 exactly at N ≤ 8, and exactly there FR damages the marginal (0–1/32 seeds better). This
  is co-occurrence across N: consistent with KDE noise, but not separated from other N-monotone covariates.
* **Birth–death excess.** 5 400–23 700 deaths per walker accompany the damage. It cannot be separated from the KDE
  noise.
* **Genealogy.** Collapse to one ancestor already at N = 128, where FR still gives −30 %, so genealogy loss is not
  the mechanism.
* **ABF converging alone.** ABF is within 1.0–1.08× of its best at N ≤ 16, and that is exactly where the F gain
  ends.

## 9. Limitations

* **Walker steps vs true work.** The budget counts force evaluations. The LTA force (pairwise LJ to the
  framework) dominates the per-step cost, so walker-steps track CPU time within the measured FR overhead. Wall
  time on a parallel machine is ∝ B/N only under ideal parallelism. That favours large N, but it is an inference
  that was not measured here. The best-allocation answer is a statement about force evaluations, not wall time.
* **FR's extra work.** The KDE, score and resampling run every 5 steps (LTA) / 160 steps (gateway). The measured
  cost is +1.1 % (LTA, N = 1024) and +0.3 % (gateway, N = 2048), rising to +13.9 % / +8.7 % at N = 2
  (`FR_OVERHEAD.md`). It is excluded from B. Including it would only lower FR, most at the small N where the best
  allocations lie.
* **Finite-N effects.**
  * Below N = 50 the LTA FR uses the preregistered cap extension max(1, ⌊0.02N⌋). At N = 2 FR is near-degenerate:
    17–18.5 deaths per walker (17.25 at 300 K, 18.5 at 150 K), max event fraction 0.5.
  * The establishment rule is unreliable at small N (null-calibrated), and TV_inst is floor-dominated.
  * These are reported and never used for τ.
* **Conditional equilibration.** FR acts on the marginal only; by construction it leaves p(y | ξ) unchanged. In
  LTA the orthogonal degrees of freedom are fast (EM bond bias does not reach F), and in the gateway the transverse
  fibre is exactly Gaussian. Neither system tests a slow hidden coordinate.
* **Reference errors.**
  * LTA F reference noise is 0.006 / 0.0035 kJ/mol, below all reported Ī_F.
  * The mean-force reference SE (0.014 / 0.010) is 15–20 % of the final e_F′ at small N, so final e_F′ there is
    partly reference-limited. This is a common offset to both arms, so the paired G is unaffected.
  * The gateway reference is analytic. Its F′ read-out floor makes strict e_F′ unreachable (§0).
* **Transient vs persistent.**
  * At LTA 300 K the large-N e_F improvement is mostly transient: 31 % peak, 9 % at the end, endpoint G not
    significant.
  * At 150 K it is persistent at N ≥ 256 (final e_F −16 to −64 %). The endpoint flips sign at N = 64 (significant)
    and at N = 128 (n.s.).
  * The gateway is persistent at N ≥ 512 (final e_F −50 to −61 %), mostly transient at N = 128–256, and leans worse at the endpoint at N = 64.
* **Best-N stability.** The best N is not sharply identified: the top bootstrap frequency is 0.42 (300 K ABF) and
  0.47 (150 K ABF). The bootstrap re-selects N, so the CIs of the best values include this selection uncertainty.
  The optimum region is stable only at about N = 1–32. Within N = 2–16 fall 77–99 % of the resamples. ABF puts
  7–17 % of the bootstrap selections on N = 1, and FR puts 19–22 % on N = 32 at LTA 150 K and in the gateway. The
  exact N within the region is not stable.
* **Seeds.** 16 per (N, method) for LTA and 32 for the gateway. Per-seed scatter of G at small N (N ≤ 16) has an SD of about 10–65 % in LTA and 55–90 % in the gateway. The 95 %
  CI half-widths are 6–23 points (LTA) and 16–29 points (gateway), so effects below about 15–25 % are generally not
  resolvable there.

## 10. Scientific conclusions and next steps

### Answers to the research questions (task §1)

Each answer uses the full convergence history: time-to-accuracy, persistence of the median curves, and the
per-N figures A–F and S7. Integrated or final errors alone are not used.

**Q1. At fixed N, does ABF + FR converge faster than ABF?**
* **Yes at large N.**
  * Gateway: N ≥ 128 on τ(e_F mid), 33–67 % earlier.
  * LTA 150 K: N ≥ 256 and N = 64, 44–61 % earlier.
  * LTA 300 K: N = 1024 by sign test only.
* The same N carry the integrated Ī_F gains.
* The gain persists to the end of the budget only at the largest N: gateway N ≥ 512 and LTA 150 K N ≥ 256. It is
  mostly transient elsewhere (LTA 300 K, gateway N = 128–256).
* **No at small N:** neutral at N ≤ 16 in every system.

**Q2. At which N is the FR improvement largest?**
* Ī_F: N = 512 in the gateway (−41.0 % [−49.6, −33.4]) and at LTA 150 K (−34.5 % [−35.9, −31.0]); N = 1024 at
  LTA 300 K (−12.5 % [−13.9, −10.5]).
* Final e_F: the largest N in each system (−61 %, −64 %, −7 % n.s.).

**Q3. Is the benefit monotonic in N?**
* **No.** It is a broad large-N plateau with an interior peak at N = 512 (gateway, 150 K).
* Intermediate N is irregular: at 150 K, N = 128 is n.s. but N = 64 is −15.7 %; in the gateway N = 64 is n.s. but
  N = 32 is −22 %.
* At small N the sign changes on the marginal (gateway N ≤ 8) and occasionally on F (150 K N = 4).

**Q4. Is ABF + FR better than the best ABF-only allocation at equal force evaluations?**
* **No.** FR's best allocation does not beat ABF's best: +3.4 % [−13.2, +12.4], +5.2 % [−10.9, +24.1],
  +15.9 % [−4.2, +26.6].
* Both arms are best at the same small N. This is outcome A in its weak form; outcome B is not supported.
* A preregistered **fresh-seed confirmation** at the selected N\* (`CONFIRM_BEST_ALLOCATION.md`) agrees:
  * LTA ties: +6.2 % [−1.2, +22.4] at 300 K; −3.0 % [−8.7, +3.3] at 150 K;
  * the gateway is **significantly worse with FR**: +24.0 % [+1.1, +52.9]. That is strict outcome A in the
    gateway.

**Q5. Does FR establish the reaction-coordinate marginal more quickly?**
* **Yes, strongly, at N ≥ 16–32:**
  * gateway τ(TV_half ≤ 0.1) 75–85 % earlier;
  * LTA 300 K −20 % at N ≥ 256;
  * integrated TV_half −17 to −73 %.
* At N ≤ 8 it does not (LTA), or it damages the marginal (gateway).

**Q6. Does improved marginal establishment translate into better F′ and F?**
* **Only partly.**
  * At large N the F gain is 1.1–2.9× smaller than the marginal gain. The mean force improves less: Ī_F′ −4 to
    −22 %; at its converged level it is limited by statistics or read-out floors.
  * At intermediate N the marginal improves with no F gain (outcome C). Where FR's integrated F is better, the
    endpoint can reverse (150 K N = 64).

**Q7. What happens at very small N?**
* FR acts on a few replicas with a KDE score that is noise (σ_KDE ≥ 1 at N ≤ 16).
* It realises 100–24 000 deaths per walker; at N = 2 it is near-degenerate.
* The free energy is unaffected or slightly worse (150 K N = 4: +11.7 %). The marginal is unaffected (LTA) or
  damaged (gateway N ≤ 8).
* ABF alone is at its best error there.

**Q8. Do the conclusions survive rigorous numerical validation?**
* **Yes.** Every ladder ran at a timestep that passed frozen gates against exact samplers: MALA and analytic Gibbs
  for the gateway; Metropolis MC for LTA. Errors were scored against exact references.
* The gateway conclusions reproduce the 2026-10-04 ladder made at the historical, invalid h = 4 × 10⁻⁴. FR's
  large-N gain and the best-allocation verdict did not depend on that integrator error.

### Conclusions

1. **Uniform-target FR is a per-replica-time accelerator, not a force-evaluation accelerator.**
   * At fixed, large N, where ABF has not established the reaction-coordinate marginal within T_N, FR lowers the
     integrated free-energy error by 10–41 %. In the gateway and LTA 150 K it also brings mid-accuracy forward
     by 33–67 %.
   * That is equivalent to running ABF about 1.2–1.8× longer per replica.
   * Under an equal force-evaluation budget, the best allocation in all three systems is a few long ABF
     replicas (N = 2–16), and FR's best allocation does not beat it: +3.4 %, +5.2 %, +15.9 %. The CIs bound a
     possible FR advantage at ≤ 4–13 %.
   * The fresh-seed confirmation at N\* tightens this to ties in LTA (FR advantage ≤ 1.2 % at 300 K, ≤ 8.7 % at
     150 K). In the gateway ABF is significantly better (+24 % [+1, +53]).
2. **FR's marginal improvement does not translate one-for-one into free-energy accuracy.**
   * The marginal gains are 1.1–2.9× the F gains: 1.8–2.9× in the gateway (N ≥ 32) and 1.5–2.3× at LTA 300 K (N ≥ 128), but only 1.1–1.3× at LTA 150 K N ≥ 256.
   * At intermediate N the marginal improves while F does not (outcome C), and the endpoint can reverse
     (significantly at LTA 150 K N = 64, +15.9 % [+5.6, +66.4]; in the point estimate only at gateway N = 64, +26.4 % [−0.6, +55.0], and LTA 150 K N = 128, +27.5 % [−2.5, +63.3]).
   * The converged mean force is never improved beyond its statistical or read-out floor.
3. **At small N FR should be switched off.** It is neutral on F and can be harmful: LTA 150 K at N = 4, gateway
   marginal at N ≤ 8, gateway e_F′_stat at N = 2. The KDE score is noise once σ_KDE ≳ 1, which happens at N ≲ 16
   for these bandwidths.
4. **Recommendation.**
   * If the replica count is set by hardware parallelism and the per-replica time is shorter than ABF's own
     establishment time, FR is a cheap (≈ 1 % measured overhead at large N) way to cut the per-replica time needed, and
     hence wall-clock time under ideal parallelism (inferred, not measured). It is safe only away from small N.
   * Otherwise, run fewer, longer ABF replicas. FR should not be presented as reducing the computational cost of
     free-energy estimation.
   * The manuscript claim must be stated at fixed N (wall-clock) and must carry the equal-budget negative.
5. **No further experiments are needed to answer the questions posed here.** Open uncertainties, not answered by
   this design:
   * the gateway's contrary seed-level starvation test;
   * systems with slow hidden coordinates, where FR's p(y | ξ)-preserving dynamics cannot help by construction;
   * the exact location of the best N within N = 2–16.

   None of them changes conclusions 1–4.
