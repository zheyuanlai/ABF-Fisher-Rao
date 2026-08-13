# SPEC — NaCl ion pairing in explicit water (the published Talmazan 2025 tutorial system)

**Status: FROZEN at the commit carrying this file.** Licensed by Amendment 14 of
`docs/V2_PREREGISTRATION.md`. No NaCl trajectory may run before this file is committed, and no
clause below may be edited after the ABF-only screen runs except by a further numbered amendment
stating what changed, why, and what had been seen at the time.

**What had been seen when this was written:** the Supporting-Information archive had been
downloaded and hashed, the published model had been loaded into OpenMM, and one single-point
energy had been computed (`results/nacl/stage0/`). **No trajectory, no reference, no screen, no
gate verdict and no mFR result of any kind existed.**

This spec deliberately mirrors `docs/SPEC_methane_water.md`; identical machinery is referenced
there rather than restated. What is NaCl-specific is stated here in full.

---

## 0. The question

Reversible Na⁺/Cl⁻ ion pairing in water is the introductory ABF example of Talmazan, Fu, Zhou,
Hénin, Gumbart & Chipot, *J. Phys. Chem. B* **129**, 9913–9928 (2025), with the ion separation
as the reaction coordinate and a published **100 ns** ABF budget (`run 50000000 × 2 fs`) that the
paper reports agreeing with a 700 ns equilibrium calculation. That budget is exactly the frozen
`B_MD = N·T = 100 ns` of preregistration §8.2.

The hypothesis under test is the three-timescale window (Amendment 10):

```
  T_hit  <<  tau_perp  <<  T_est
```

with hydration-shell reorganisation (`n_NaO`, `n_ClH`, `n_bridge`) as the orthogonal physics.
The four-way classification of Amendment 10 is applied in the frozen order
Gate 0 → A → B → C → D, and the first failing gate is the verdict.

**Preregistered prior, stated to disarm it:** the published PMF suggests modest barriers
(a few kT), so ABF-sufficient is a live outcome; NaCl is *hoped* to be establishment-limited but
nothing below is tuned to make it so. Preregistration §11 fixes the reading of every outcome.

---

## 1. Physical model (frozen — extracted, never transcribed)

The model is **the published tutorial system, verbatim**: `NaCl/solvate.psf`,
`NaCl/solvate.pdb`, `NaCl/par_all22_prot.inp` (+ the CHARMM22 topology shipped in the same
archive), loaded by OpenMM's CHARMM readers. **Every number the torch engine consumes is read
back out of the OpenMM `System`** (`scripts/nacl_stage0_extract.py` →
`results/nacl/stage0/site_params.npz` and `model_manifest.json`); no force-field constant in
this repository is typed from a paper. The methane `sigma_O` lesson is the binding precedent.

Provenance: archive `NIHMS2186658-supplement-tutorial_files.zip` (PMC 13284794), sha256
`f33a8fce86bc9fb7c85afd1647a81bb66d5bd9118de2bde721796ca988a5d94c`; per-file hashes in
`cache/talmazan2025/nacl_file_hashes.sha256`.

| quantity | value (as extracted) |
|---|---|
| particles | 2465 = Na⁺ + Cl⁻ + 821 CHARMM TIP3P waters |
| Na⁺ (SOD) | `q = +1 e`, `sigma = 0.242993 nm`, `eps = 0.196230 kJ/mol`, `m = 22.9898` |
| Cl⁻ (CLA) | `q = −1 e`, `sigma = 0.404468 nm`, `eps = 0.627600 kJ/mol`, `m = 35.45` |
| water O (OT) | `q = −0.834 e`, `sigma = 0.315057 nm`, `eps = 0.636386 kJ/mol` |
| water H (HT) | `q = +0.417 e`, `sigma = 0.040001 nm`, `eps = 0.192464 kJ/mol` |
| rigid water | `r_OH = 0.09572 nm`, `r_HH = 0.15139007 nm` (θ = 104.52°), 3 constraints/water |
| mixing rule | Lorentz–Berthelot (OpenMM/CHARMM convention); **no NBFIX** (asserted) |
| temperature | 300 K (`kT = 2.494339 kJ/mol`) |
| thermostat | Langevin, `gamma = 1 ps⁻¹` (published `langevinDamping 1.0`) |
| LJ switch → cutoff | 1.00 → 1.20 nm (published `switchdist 10 / cutoff 12`) |
| electrostatics | PME |
| timestep | 2 fs (published), gated — see §1.2 |
| ensemble | NVT at the NPT-equilibrated volume — declared deviation, methane §1.3 protocol |

> **CHARMM TIP3P hydrogens carry real LJ** (`eps_H > 0`), unlike SPC/E. Every site in this
> system is both charged and LJ-active, and the engine treats that generically.

### 1.1 Declared deviations from the published protocol

Collected here in advance, none silent:

1. **NVT production at the NPT-equilibrated volume** (published production is NPT with a
   Langevin piston). Same rule and same reason as methane §1.3: the target measure must be
   exactly canonical `nu` with no barostat variable. The box is frozen from our own NPT run
   (§1.3 below), not from the tutorial's 11 ps `equilibrate.xsc` snapshot (28.9876 Å).
2. **BAOAB + M-SHAKE/RATTLE** instead of NAMD's leapfrog + SETTLE. The water angle force —
   geometrically frozen once the triangle is fully constrained, which the published
   `rigidbonds all` also enforces via SETTLE — is **removed** from the parity System; it
   contributes zero force on the constraint manifold and a constant energy.
3. **No multiple timestepping.** NAMD evaluates reciprocal PME every 2 steps
   (`fullelectfrequency 2`); we evaluate the full force every step. Strictly more accurate.
4. **PME parameters pinned, spline order 5.** NAMD used order 4, grid spacing 1.0 Å,
   tolerance 1e-5. OpenMM's PME is order-5; we pin `alpha` from tolerance 1e-5 at the 1.2 nm
   cutoff and a 30³ grid (≈ 1 Å spacing) in **both** engines, so parity does not depend on a
   box-derived default. Recorded in the manifest.
5. **No local NAMD rerun.** No NAMD binary exists on this node. Stage 0B is discharged by the
   authors' own shipped outputs (`output/abf.pmf`, 100 ns ABF; `equilibrate.*`), which are the
   literature anchor our reference is compared against externally. An approximate
   reimplementation is not called a reproduction: the claim is **the published model, exactly,
   under a declared integrator/ensemble substitution** — the same claim methane makes.
6. **Dispersion correction off in NVT** (additive constant, zero force); on for NPT only.
   Methane rule, unchanged.

### 1.2 Timestep gate

`dt = 2 fs` is the published value and the default. Stage I runs the constraint and
equipartition clauses (methane Amendment 11.3) at **1 fs and 2 fs**: constraint violation
`<= 1e-8 nm` over a production-length run, and kinetic temperature against the constrained DOF
count agreeing with OpenMM's `LangevinMiddleIntegrator` at the same `dt`. If 2 fs fails either,
production drops to 1 fs; the choice is made by this gate **before any free-energy data** and
recorded. `fullSamples`-related settings do not depend on `dt` (they are per-sample counts).

### 1.3 Box and ensemble

1. NPT (OpenMM, Langevin piston equivalent — `MonteCarloBarostat`), 1 bar, 300 K, dispersion
   correction ON, starting from the published `equilibrate.coor`/`.xsc`;
2. 0.5 ns discard, 1.0 ns average of `<V>`; `L = <V>^{1/3}` frozen and recorded;
3. all reference / screen / production is NVT at that `L`.

**Finite-size exposure, declared:** the published domain reaches 14 Å while `L/2 ≈ 14.5 Å`.
Gate, fixed in advance: the upper evaluation boundary must satisfy `R_hi <= 0.97·(L/2)`;
if the measured `L` violates it, the domain is truncated to the largest passing bin edge
**before** the reference is built, and the truncation is recorded. The minimum-image
unambiguity assertion runs with margin 0.995 (methane used 0.98 at 72 % of half-box; NaCl's
domain top sits at ~97 %, so the assertion is tightened rather than dropped).

---

## 2. Reaction coordinate, mean force, conventions

```
  xi(q) = |Q_Cl − Q_Na|  (minimum image)         — PeriodicDistanceCV, methane §2, reused
  f_loc = (1/2)(F_Na − F_Cl)·e − 2/(beta r)      — DistanceCV.local_mean_force, unchanged
  F'(r) = E[f_loc | xi = r]
  W(r)  = F(r) + 2 beta⁻¹ log r + C ,   W'(r) = F'(r) + 2/(beta r)
```

**Evaluation domain** `[0.20, 1.40] nm`, **121 grid points** (odd; spacing 0.01 nm — exactly
the published Colvars `width 0.1 Å`), subject to the §1.3 finite-size gate.

**Walls:** exactly the published `harmonicWalls`: at 0.20 and 1.40 nm with
`k = 1 kcal/mol per (0.1 Å)² = 418.4 kJ/mol per 0.01 nm² = 41 840 kJ/mol/nm²`, identical on
every arm.

**Convention check, frozen:** the published `abf.in` contains **no `hideJacobian`**, so Colvars
default applies and the shipped `output/abf.pmf` is the book's `F(r)` (121 bins, kcal/mol,
zeroed at 14 Å). Both `F` and `W` are emitted by every build and every arm, and

```
  || W'_hat − (F'_hat + 2/(beta r)) ||_inf  <  1e-10 kJ/mol/nm     (same samples, hard test)
```

**`fullSamples = 500`** (published) is the applied trust ramp: Colvars applies zero bias below
`fullSamples/2`, then ramps linearly to full at `fullSamples`. Our `abf_min_count` machinery
implements the ramp and a unit test fails if the configured value is not applied (Amendment 5
Defect 2 precedent).

---

## 3. Stage I — engine: consumed, not forked

The batched periodic engine is the methane session's deliverable
(`methane.nonbonded.PairTerms`, `methane.pme.PMEReciprocal`, `methane.dynamics.{BAOAB,
RigidWaterConstraints, PairConstraint}`, `methane.cv.PeriodicDistanceCV`). The NaCl package
(`src/nacl/`) adds **only** the model layer: parameter loading from
`results/nacl/stage0/site_params.npz`, NaCl constants (cutoff 1.2 / switch 1.0 nm, pinned PME),
hydration observables, and drivers. **PME, LJ, constraints and the integrator are not
reimplemented.** The one engine-side change NaCl needs — the `PairTerms` split-path assertion
assumes LJ-free hydrogens, false for CHARMM TIP3P — is made graceful (flag, not raise) without
touching the methane execution path.

### 3.1 Gate — engine equivalence (binding; blocks everything downstream)

Preregistration §8.1, thresholds unchanged from methane §3.2: max relative discrepancy
`< 1e-6` on `V`, `grad V`, `f_loc` and the ABF bias force against OpenMM on the same
configurations (identical pinned PME parameters), `xi` to float64 round-off, plus the rigid-water
constraint (`1e-8 nm`) and equipartition clauses at both candidate timesteps. Near-zero
components are additionally reported as absolute discrepancies.

**Configuration pool (charged-solute extension):** ≥ 16 configurations spanning contact
(~0.27 nm), barrier, solvent-separated (~0.5 nm) and dissociated (≥ 1.0 nm) separations, with
**distinct hydration structures at the same `r`** (families of §6), so the parity set exercises
Na–water, Cl–water, Na–Cl direct, reciprocal, self and exclusion terms across the states that
matter. The published `equilibrate.coor` is configuration #1.

### 3.2 Physical validation (after parity, before any free energy)

* `g_NaO(r)`, `g_ClO(r)`, `g_ClH(r)` from a short unrestrained-ion-pair-free run (ions held
  dissociated): first-shell positions physically sensible against the CHARMM TIP3P ion
  literature (sanity gate on the potential, not a curve-identity requirement);
* Na coordination number distribution well-defined and stable;
* **`R0` freeze:** the `n_ClH` and `n_ClO` switch radii are set to the first minima of these
  reference RDFs, rounded to 0.005 nm, **before any ABF or screen data exist**, and recorded in
  `results/nacl/stage0/descriptor_freeze.json`. `R0_NaO = 0.315 nm` is fixed here in advance
  (PLUMED masterclass 22.11 analysis constant; it alters no force-field term).

---

## 4. Orthogonal hydration descriptors (frozen)

Rational switch `s(x; R0) = (1 − (x/R0)⁶) / (1 − (x/R0)¹²)` — the project-standard form
(methane §5.1), applied to minimum-image distances:

```
  n_NaO    = sum_j s(|Na − O_j|;  R0_NaO)                       — primary Na descriptor
  n_ClH    = sum_{j,h} s(|Cl − H_{jh}|; R0_ClH)                 — primary Cl descriptor
  n_ClO    = sum_j s(|Cl − O_j|;  R0_ClO)                       — secondary/robustness
  n_bridge = sum_j s(|Na − O_j|; R0_NaO) · s(min_h |Cl − H_{jh}|; R0_ClH)
  Y        = (n_NaO, n_ClH, n_bridge)
```

plus a hard-count oxygen bridge (`|Na−O_j| < R0_NaO` and `|Cl−O_j| < R0_ClO`) as a robustness
diagnostic. `Y` is what Gate A distinguishability, Gate D twin decorrelation, `tau_perp` and
conditional fidelity `TV[p_method(Y|r), p_ref(Y|r)]` are all computed on. `R0` values are never
retuned after the freeze.

---

## 5. Stage II — reference, independent of ABF

**Primary: constrained mean-force TI** (methane Amendment 12.2 precedent), on the frozen
domain: grid `r = 0.20, 0.22, …, 1.40 nm` (61 points; refined by +0.01 nm inserts around the
two PMF extrema after a first pass), **4 hydration families per point** (contact-derived,
SSIP-derived, dissociated-derived, locally-equilibrated), ≥ 3 replicas per family,
`>= 50 ps` equilibration + `>= 250 ps` production per replica, **3 independent builds**.

> **Declared deviation from the methane engine split:** the TI runs **batched in the torch
> engine** (all points × families × replicas as one walker batch under `PairConstraint`),
> because 61 points × 12 replicas × 3 builds is serial-infeasible in OpenMM at this system
> size. Independence is retained by an **OpenMM CUDA spot-check**: ≥ 6 grid points × 2
> replicas re-run in OpenMM with the same protocol; `<f_loc>` must agree within combined block
> error. A disagreement stops the reference.

Acceptance (all binding):

```
  R_ref = max pairwise L2 between builds / (0.10 × span F_consensus)  <=  0.5
  W' identity of §2 to machine precision
  family disagreement per point reported, never averaged away
  OpenMM spot-check passes
  reference uncertainty propagated into the primary I_F sensitivity analysis (§4.5)
```

**External literature check (reported, not a gate):** final `F_ref` against the shipped
`output/abf.pmf` — same convention (§2), aligned at the dissociated plateau. Discrepancy is
reported with the reference uncertainty; our arms are scored **only** against our own accepted
reference, never against the tutorial curve.

**States:** CIP / SSIP / (outer) from local minima of `F_ref` with the `< 2 kT` basin-merge
rule (Amendment 3), boundaries at intervening maxima; fallback = frozen tercile partition.
Never from ABF data.

---

## 6. Gate 0 and `tau_perp` — fixed-`r` conditional machinery

At `r_k ∈ {r_CIP, r_barrier, r_SSIP, 1.20 nm}` (from **our** reference), four solvent families
per point (prepared as in methane §5.2: wet↔SSIP-derived, dry↔CIP-derived, bulk, hot), held by
`PairConstraint`, ≥ 32 replicas per family:

* **Gate 0:** cross-family spread of `<f_loc>` against `|F'_ref|`, judged against the
  calibration ladder (WCA 0.040 pass / gateway 0.036–0.189 pass / deca 0.61 fail / R15 0.564
  fail — Amendment 9: no numerical threshold, argued in `RESULT.md` in the error-carrying
  region). Fail ⇒ **conditional-equilibration-limited, STOP** (the Amendment 8 theorem).
* **`tau_perp` (family):** `inf{t : max_{a,b} TV[p_t(Y|r_k,a), p_t(Y|r_k,b)] <= 0.2}`, maxed
  over `k`.
* **`tau_perp` (clone twin, Gate D instrument):** duplicated pairs, independent noise,
  `C_Y(t) <= 1/e`. Both computed; disagreement > 2× is reported as an open finding.

**Gate A** (on the reference joint): `max over state pairs TV[p(Y|state a), p(Y|state b)]
>= 0.30`, evaluated through the frozen `Y`. Fail ⇒ STOP (a stop for the CV, not a licence to
tune).

---

## 7. Stage III — ABF-only fixed-compute regime map (preregistration §8.2, unchanged)

```
  B_MD = N · T = 100 ns   (the published ABF budget)
  N ∈ {8, 16, 32, 64}  →  T ∈ {12.5, 6.25, 3.125, 1.5625} ns
  8 ensemble seeds per cell: 4000–4007
```

* **Initial condition rule (frozen):** every walker starts in the basin containing the
  published starting configuration — `equilibrate.coor` has `r = 3.0 Å`, i.e. the
  contact/CIP basin. Each walker receives independently equilibrated solvent: ≥ 50 ps
  restrained equilibration per walker, outside the ABF budget, `assert_distinct_solvent`
  enforced. Declared bias: a contact start can only push toward discovery-limited, never
  manufacture establishment-limited.
* **Execution order:** cells run **`N = 64` first** (cheapest wall-clock per verdict at high
  batch efficiency), then 32, 16, 8; each cell's 8 seeds run **in one process on one GPU**
  (the WCA within-process determinism trap). A partial map is reported as partial.
* **Estimator:** `alkanes.interval` machinery, bandwidth ladder `h ∈ {0.008, 0.012, 0.016} nm`
  selected by the methane §6.2 deterministic rule on ABF-only + reference data, then frozen.
* **Gate B:** persistent `T_hit,k < 0.1 T` on ≥ 6/8 seeds, per relevant state.
* **Gate C:** bias-aware target `Q*_k(t) ∝ ∫_{C_k} exp(−beta[F_ref − B_t])`; under-established
  = occupancy `< 0.5 Q*_k` for a contiguous `>= 0.20 T` in the second half. Early hit + no
  deficit ⇒ ABF-sufficient, STOP. Multiple passing cells ⇒ smallest `N`, mechanically.
* **Gate D:** `lambda_rep · tau_perp <= 0.1` with activity floor `N_repl >= 0.5 N`; no active
  safe rate ⇒ C3 failure, STOP.

---

## 8. Stage IV — mFR production (only if every gate passes)

Mechanism, score, and estimator unchanged from the campaign (methane §9). Rate calibration on
seeds **4100–4103**: four rates spanning measured `lambda_rep · tau_perp ≈ {0.01, 0.03, 0.10,
0.30}`, inactive rates struck, genealogy gates `ESS_anc/N >= 0.30`, `w_max <= 0.05`, minimum
calibration `L2(F)` with the gentler rate on ties within 2 pp. **No numerical rate transfers
from WCA, the gateway, the toys or methane.**

**Arms — all five, decided now:** `abf`, `mfr_practical`, `mfr_sham`, `book_laplacian`,
`count_balancing`, on fresh matched seeds **4200–4215**. NaCl is where Q1 (does FR
directionality beat prior directed selection?) gets its molecular answer, and the WCA Q1 result
(mFR TIED by count_balancing) makes the prior-art arms mandatory, not optional. One sham per FR
arm, replaying realized replacement counts/times with random identities; the direct arm-vs-sham
contrast is the attribution statistic.

Endpoints, diagnostics, success criteria: preregistration §4.3 (median ≤ −10 %, CI upper
< −5 %, ≥ 12/16, genealogy, sham CI < 0), Q1 novelty per §4.3 with TOST for ties, `I_F` primary
with `I_W`, `I_F'`, `e_F'`, `ΔW_CIP→SSIP`, `ΔW‡`, transitions/round trips, marginal KL/TV,
conditional fidelity on `Y`, global + per-state + age-aware genealogy, clone decorrelation, and
the mandatory frozen-bias validation (~25 % budget) for any positive. A gain in `F` with
significantly worsened `p(Y|r)` fidelity is **not** a success.

---

## 9. Compute

One GPU per study, pinned with `CUDA_VISIBLE_DEVICES`, idleness re-checked and recorded before
each stage (Amendment 13/14). NaCl runs on **one** of GPUs 2/3 concurrently with methane on the
other. Whole confirmatory blocks run in single processes. Throughput (ms/step vs batch) is
measured and reported before the screen; if the frozen budget is infeasible at the measured
cost, that is an amendment, not a quiet reduction.

## 10. Stop conditions — every one is a reported result

| condition | verdict |
|---|---|
| engine-equivalence gate fails | NaCl does not run; defect reported |
| finite-size gate truncates below the SSIP basin | domain redesign amendment, or NaCl withdrawn |
| reference acceptance fails | STOP, rebuild; no screen interpreted |
| Gate 0 fails | conditional-equilibration-limited; STOP |
| Gate A fails | hydration states invisible through `r`; STOP |
| Gate B fails | discovery-limited; STOP |
| Gate C finds no deficit | **ABF-sufficient**; STOP and report |
| Gate D admits no active safe rate | C3 failure; STOP |

If NaCl stops, nothing is retuned — no temperature, box, domain, `T`, or force-field change —
per preregistration §9.

## 11. References

Talmazan, Fu, Zhou, Hénin, Gumbart, Chipot, *J. Phys. Chem. B* 129, 9913–9928 (2025),
`10.1021/acs.jpcb.5c04333` — the benchmark; SI archive is the model source.
Beglov & Roux, *J. Chem. Phys.* 100, 9050 (1994) — the SOD/CLA parameters carried by
CHARMM22 (context only; parameters are read from the files, not this paper).
Jorgensen et al., *J. Chem. Phys.* 79, 926 (1983) — TIP3P (CHARMM variant as shipped).
PLUMED masterclass 22.11 — Na–O coordination analysis constant `R0 = 0.315 nm` (analysis
only; no force-field content).
Lelièvre, Rousset & Stoltz (2010) — Chapters 3, 5: conditional mean force, ABF.
