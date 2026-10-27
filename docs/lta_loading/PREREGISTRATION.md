# Ethane/LTA at finite loading: does the ABF + FR advantage survive interacting guests? (preregistration)

*Written 2026-10-27 on NSCC ASPIRE2A and frozen in git before any finite-loading trajectory (validation, reference,
pilot or production). Changes after the freeze are dated amendments at the end of this file.*

## 0. What was seen before writing

* The single-guest ethane/LTA results: FR (uniform target) beat ABF at every temperature of the CPU sweep, most at
  low T (`docs/LTA_HISTOGRAM_REPLICATION.md`, 150 K: ΔI_F −28 %, 16/16), and on the A100 the best FR deployment
  reached the mid accuracy faster than the best ABF deployment at 150 K (R_W = 0.72 [0.64, 0.83]) but not at 300 K
  (1.25) (`docs/nscc_a100/RESULTS.md`).
* The C60/water study was considered first and shelved for cost (`docs/C60_EXECUTION_STATE.md`; the frozen
  protocol needs weeks of A100 time, and its ~2.4 kT solvent barrier suggests an ABF-sufficient system).
* Framework geometry only (no dynamics): sodalite-cage centres are deep single-bead minima (−14 kT within 1 Å of the
  lattice corners at 150 K), separated from the α-cage network by a hard wall (> +600 kT at 3–4.5 Å from the
  corner); every point with U < −5 kT outside the sodalite cages is ≥ 5 Å from a corner.
* No finite-loading trajectory, reference, error curve or timing of any kind.

## 1. Question and claims

The single-guest model has no orthogonal environment: the only slow coordinate besides the CV is the guest's own
transverse position. At finite loading the other guests are an environment whose arrangement couples to the tagged
guest's free energy and is cloned along with it by FR. Question:

**Q1 (primary).** At 150 K and 2 ethanes per α-cage, after each method picks its best replica count N from a
ladder at equal walker-step budget B, does the best ABF + FR deployment reach the frozen mid accuracy of the tagged
guest's free energy in fewer walker-steps than the best ABF deployment, on fresh confirmation seeds?

**Q2 (secondary).** At each N, the paired FR vs ABF change of the integrated error I_F (same seeds; the two arms
share initial conditions and noise until FR starts).

**Q3 (secondary).** The modelled wall-clock version of Q1 on one A100 (§8).

A "no" or "unresolved" is a result: it would bound the single-guest positive to systems without an interacting
environment.

## 2. System (frozen)

* Framework, guest model, CV, dynamics: exactly the single-guest model of `src/lta_ladder_numba.py` (IZA LTA, 2x2x2
  pseudo-cells, a = 11.919 Å, L = 23.838 Å, 384 rigid framework O; CH3–O shifted LJ 93 K / 3.48 Å, rc 10 Å;
  harmonic bond k 400 kJ mol⁻¹ Å⁻², r0 1.54 Å; overdamped Euler–Maruyama, h = 2e-4; T = 150 K).
* **Loading: n = 16 ethanes per box (2 per α-cage).** Guest–guest CH3–CH3 shifted LJ between beads of different
  molecules: 108 K, 3.76 Å, rc 10 Å (Dubbeldam et al., PRL 93, 088302 (2004), the united-atom set of the CH3–O
  parameters already in use). No intramolecular LJ.
* **Tagged CV.** Molecule 0 carries the CV, φ = wrap(2π x_COM / a), 180 bins. ABF (and FR) act on molecule 0
  only, so the bias is a function of the CV alone and the environment's conditional law is the unbiased one; the
  free energy is the tagged guest's F(φ) at this loading (by exchange symmetry, also the one-particle density of any
  guest).
* **Environment reinsertion MC** (both arms, identical): every 10 steps, per walker, one environment guest
  (uniform among molecules 1..15) gets a new COM uniform in the box and a uniform orientation, bond length kept;
  Metropolis on the intermolecular energy. COMs within 3.5 Å of a lattice corner (sodalite cages, inaccessible
  through the 6-rings) are rejected — the standard blocking. The proposal is symmetric and independent of the bias,
  so the move leaves the biased measure invariant. **Why:** at 150 K the window barrier is 13.5 kT and unbiased
  environment guests essentially never hop during a run; without reinsertion the environment's cage occupancy would
  be frozen at its initial draw, the reference itself would be unreachable by dynamics, and the comparison would be
  between two conditional-equilibration failures. The tagged guest is never reinserted.
* **Initial conditions** (identical for both arms of a seed): every guest by the single-guest law (uniform α-cage
  centre + 0.5 Å jitter, uniform orientation); then, outside the budget, 100 placement sweeps over the environment
  (a reinsertion is taken whenever the new site does not overlap, U_inter ≤ 0), 100 Metropolis reinsertion sweeps,
  and 10 000 steps of dynamics + reinsertion with the tagged guest held fixed (Amendment A2). t = 0 then releases the tagged guest.
* **ABF and FR knobs:** inherited unchanged from the single-guest 150 K configuration
  (`configs/equal_budget_v2/lta_150K_production.json`): histogram (P0) estimator, 180 bins, bias clip 60, deposit
  clip 480, warm-up ramp 20 000 steps, production accumulators from 20 000; FR uniform target, rate 0.2, every 5
  steps from step 20 000, KDE bandwidth 0.10 rad, score clip 2, cap max(1, ⌊0.02 N⌋). Nothing is tuned for this
  system.

## 3. Engine and Stage I gates

`src/lta_loading/engine.py`: CUDA kernels through CuPy on one A100, float64, `--fmad=false`; forces, deposits
(fixed-point 2⁻⁴⁰ with integer atomics: exact and order-free), Euler–Maruyama move with counter-based Philox noise
keyed by seed, reinsertion MC; the FR score and selection are the numba engine's own `_fr_score` /
`fr_select_native` with its per-run PCG64 FR stream. R seeds of one (method, N) run batched; a seed's trajectory does
not depend on its batch.

Gates (`scripts/lta_loading/validate.py`), all must pass before the reference:
* V1 forces = −∇U by central differences (rel ≤ 1e-5); V2 forces and energy vs an independent NumPy implementation
  (≤ 1e-10); V3 at n = 1, forces, CV, bin and local force equal the numba single-guest engine (≤ 1e-12);
* V4 at n = 1 against the exact-MC single-guest reference: under its frozen bias (flat landscape) the mean-force
  route e_F ≤ 0.05 and the density route e_F ≤ 0.10 kJ/mol (a wrong bias sign or scale fails the latter by kT-sized
  errors); the ABF path at N = 1024 x 300k steps, final per-seed e_F ≤ 0.30 (single-guest CPU engine: 0.14–0.20);
* V5 reinsertion: a non-interacting environment guest reproduces the exact single-guest density at 150 K on bins
  with ≥ 400 expected samples (errors from 16 independent walker groups: rms z ≤ 1.5, |mean log ratio| ≤ 0.02,
  TV ≤ 0.02), never sits in a blocked sphere, and the kernel's ΔU of every accepted move equals the change of the
  total energy (n = 4, interacting, production pre-equilibration);
* V6 a seed run alone and inside a batch gives bitwise-identical accumulators and FR events.

## 4. Reference (before any scored run)

`scripts/lta_loading/reference.py`, seeds 30000–30003 and 30100–30107 (used nowhere else):
* pilot ABF (FR off), 4 seeds x N 1024 x 300 000 steps → bias table Γ_pilot (pooled production bins);
* frozen-bias equilibrium, 8 independent groups x N 1024 x 500 000 steps, burn-in 50 000, bias fixed at Γ_pilot.
  Because the bias depends on φ only: γ_ref = pooled conditional mean force, **F_ref = histogram_pmf(γ_ref)** (the
  scorer's functional form); the density route F_density = Â_pilot − kT log C is an independent check. Group
  jackknife standard errors.
* **Acceptance:** rms jackknife SE of F_ref ≤ 0.03 kJ/mol (a quarter of the mid threshold) and mean-force vs
  density route rms difference ≤ 0.05 kJ/mol. Failing either: the frozen-bias run is extended once by the same
  length with fresh groups; failing again stops the study.
* Reported, not gated: the loaded barrier vs the single-guest barrier (13.5 kT), MC acceptance, tagged crossings.

## 5. Budget B (ABF-only pilot, before any FR run)

ABF at N = 256, 8 pilot seeds (30200–30207), 1 200 000 steps each. **B = 2 x median b_mid**, rounded up to a
multiple of 65 536 walker-steps; if the median is censored, B = 256 x 1 200 000 and the pilot is reported. The pilot
reads ABF error curves only.

## 6. Exploration ladder and selection

* N ∈ {64, 128, 256, 512, 1024}; n_steps = B / N; both arms; 16 seeds per arm (31000–31015), the same seeds for
  both arms (paired by construction until step 20 000).
* Endpoints per run: b_thr = walker-steps N k at the first save after which e_F ≤ thr at every later save
  (thresholds 0.03 / **0.13** / 0.5 kJ/mol, inherited; inf if censored); I_F = ∫ e_F du; final e_F. e_F: the
  equal-budget LTA scorer convention against F_ref, RMS over all 180 bins after removing the mean difference.
* **Selection (frozen before confirmation):** per method, the N with the smallest median b_mid (censored = inf).

## 7. Confirmation and decision (Q1)

* Fresh seeds 32000–32063 (64 per arm) at the two selected arms.
* R_B = median b_mid(best FR) / median b_mid(best ABF), unpaired 10 000-resample bootstrap 95 % CI.
* **'FR better' iff the CI lies below 1; 'ABF better' iff above 1; otherwise 'unresolved'.** Censored runs count
  as inf, never imputed.

## 8. Secondary outputs

* Q2: at each N, paired ΔI_F (%) FR vs ABF, median and bootstrap CI over the 16 exploration seeds.
* Q3: c_ss(N, method) = steady-state seconds per step of a single-seed run (R = 1) on one A100, 3 repeats, FR
  active; modelled W_mid = (b_mid / N) x c_ss; R_W between the selected arms. The FR selection runs on the host in
  this engine (one device-to-host copy per opportunity), so c_ss(FR) is an upper bound of a device-side FR.
* Mechanism: FR replacements, ancestor ESS/N and maximum lineage share, tagged window crossings per walker,
  reinsertion acceptance, the loaded vs single-guest F.

## 9. Stopping rules and caps

* A failing Stage I gate or reference acceptance stops the study at that point (reported).
* Cap: 24 A100-hours in total. Nothing is stopped, extended or re-selected because of which method is winning.

## Amendments

### A0 (2026-10-27, before any trajectory)

V4 was first written as pooled ABF e_F ≤ 0.06 at 4 x 1024 x 200k steps. The single-guest CPU sweep shows final ABF
e_F 0.14–0.20 kJ/mol per seed at 1024 x 300k (`results/lta_histogram/comparison_T150.csv`), dominated by slow
establishment at the barrier rather than noise, so that gate would fail on statistics, not on a defect. Replaced by
the frozen-bias test above (fast-converging, sensitive to the bias sign and scale) plus a loose ABF-path gate.

### A1 (2026-10-27, after the first full Stage I run, before any reference, pilot or production run)

**What had been seen:** Stage I job 26029637: V1–V4 and V6 PASS (V2/V3 at ~5e-16; V4 mean-force route e_F 0.017,
density route 0.037, ABF path 0.11–0.14 kJ/mol). **V5 FAILED** on two clauses:

1. Density: max |log p/p_ref| = 0.106 vs the frozen 0.10. Diagnostic job 26029973 (16 independent walker groups):
   the per-bin standard error on the judged bins reaches 0.054, the worst bin is z = −2.5, rms z = 1.19 over 122
   bins, mean log ratio −0.002, TV 0.005, and the largest deviations flank the window with opposite signs. The
   0.10 gate had no noise model; a 2σ bin fluctuation exceeds it. Replaced by the noise-aware clause above
   (rms z ≤ 1.5, |mean log ratio| ≤ 0.02, TV ≤ 0.02). This is a gate change made after seeing its number and is
   recorded as such; it is an engine-validation clause, not a science endpoint.
2. ΔU consistency: worst relative mismatch 6.5e11. Cause: the test used a shortened pre-equilibration (3 sweeps
   + 500 steps); some walkers kept an overlap (U ≈ 7e27) and a difference of such totals has no significant
   digits. With the production pre-equilibration the mismatch is 8.8e-14 and every walker is physical
   (U ≤ −49 kJ/mol, max |F| 66). The test now uses the production pre-equilibration.

**Consequence adopted (robustness, both arms identical):** a walker left overlapped after pre-equilibration does
not stay merely improbable — its forces throw it where the minimum image has no precision, and it would silently
corrupt the shared estimator. So the pre-equilibration sweeps go from 20 to 40, every batch must pass a hard check
after pre-equilibration (finite, max |F| ≤ 1e4 kJ/mol/Å for every walker, else the batch raises), and every save
checks for a force blow-up (max |F| > 1e6 raises). Nothing else changes.

### A2 (2026-10-28, after Stage I passed, before any reference, pilot or production datum)

**What had been seen:** Stage I all PASS (job 26030410). The reference pilot (n = 16, 4 x 1024 walkers) then raised
at the A1 physical check: 2 of 4096 walkers overlapped after pre-equilibration. Diagnosis (job 26030575,
`results/lta_loading/validation/debug_preeq.json`, a deterministic replay): every walker starts overlapped (the IC
puts 2–3 guests on one cage centre); the first Metropolis reinsertion is accepted almost anywhere and, most of the
box being framework, typically parks the guest inside a wall; descent from there is slow (bad walkers 4096 → 2217 →
286 → 5 after 1, 10, 20, 40 sweeps), and in two walkers the remaining wall contact (bead–O 1.9 Å, |F| ~ 1.6e4) drives
Euler–Maruyama steps of ~3 Å that kept them jammed through the 10 000 dynamics steps. No walker became bad during
the dynamics.

**Change (initialisation only, both arms identical):** 100 placement sweeps in which a reinsertion is accepted
whenever the new site does not overlap (U_inter ≤ 0 kJ/mol) or by Metropolis — not detailed balance, so it is
followed by 100 Metropolis sweeps and the 10 000 steps of dynamics + reinsertion that equilibrate. The rule uses
each walker's own state only, so a seed's trajectory still does not depend on its batch. The A1 physical check is
unchanged and still binding.

