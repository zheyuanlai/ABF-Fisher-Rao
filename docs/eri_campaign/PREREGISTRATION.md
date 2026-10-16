# ERI campaign — preregistration (validation stages)

*Frozen 2026-10-16, before any full reference profile, any Langevin-vs-MC comparison and any ABF or FR run.*
Amendments are appended at the end and never edit text above them.

**Seen before freezing.** I had seen the literature extraction (`review/`), the Gate 0 test results (16 passing)
and two design pilots, both CPU only, Schlenker and Gard & Tait, 600 K:
* an MC umbrella pilot: 36 windows, κ 300, 2 molecules × 20k steps per window. The COM-coordinate βF barriers were
  5.98 (Schlenker) and 6.61 (Gard & Tait). U(φ) rises 3.6 / 5.6 kJ/mol from cage to window. MC acceptance: bead
  0.83, translation 0.48, rotation 0.89.
* a correlation-time pilot on single windows (Schlenker). The slowest variable is the axial position in the cage,
  η = cos(4πZ/c): τ_int ≈ 310 MC steps, and 1.09 t.u. for EM at dt 1e-4.

I have NOT seen any single-bead folded profile. That is the literature-comparison observable.

## 1. Model (labelled surrogate)

"PNAS-2005-type ERI model". Its parameters are in `review/LIT_CROSSCHECK.md` §2 and the implementation in
`src/eri/model.py`:
* LJ pairs from Schüring et al. 2002 as quoted by PNAS 2005: O–CH3 142 K / 3.17 Å, Si–CH3 82 K / 2.12 Å,
  CH3–CH3 104 K / 3.78 Å.
* O and Si both interact. No charges. Rigid all-silica framework.
* Cutoff 12 Å, energy-shifted, no tail correction.
* Harmonic bond, k/kB = 96 500 K/Å², r0 = 1.54 Å.
* T = 600 K, the paper's only ERI temperature.

The CV is the tagged COM along the hexagonal a1 axis, φ = wrap(2π (X − a/4)/(a/2)). φ = 0 is at the between-row
8-ring windows and ±π at the cage rows (`model.py` docstring).

## 2. Framework selection (literature reproduction check)

There are two variants, `gard_tait` (the cited structure) and `schlenker` (the cell the PNAS box fits).

* **Literature target:** the infinite-dilution xy profile, PNAS Fig. 3 / JPCB 2006 Fig. 4a, with barrier
  **6.13 kBT**. Two independent digitizations agree to 0.03.
* **Our comparable observable:** the single-bead folded coordinate |s|, i.e. the bead's xy offset from the nearest
  8-ring window stack projected on that window's centre-to-centre line. It is computed from the MC reference by WHAM
  reweighting with both beads pooled, with 64 bins on [0, 1.05 a/(2√3)].
* **Barrier:** βΔF\* = βF(first |s| bin) − min βF.
* **Band:** |βΔF\* − 6.13| ≤ 0.35. This allows for the digitization and for the unrecoverable folding and binning
  conventions of the paper.
* **Rule:**
  * exactly one variant in the band → it is the primary framework;
  * both in the band → `gard_tait`, the cited structure;
  * neither → the closer one, and the campaign then reports that the published barrier is NOT reproduced.

  The other variant is reported as a sensitivity result only.
* **Data:** groups 0–3 of the Gate 1 design (§3), run for both variants.

## 3. Gate 1 — independent equilibrium reference (exact Metropolis MC)

* **Windows:** 40 at linspace(−π, π, 40, endpoint=False), bias (κ/2) wrap(φ − c)² with κ = 300 kJ/mol/rad².
* **Groups:** 16 independent groups, g = 0..15, seed 8_000_000 + 1000 g + w. The group is the jackknife and
  bootstrap unit.
* **Per (group, window):** 16 molecules started on the window-to-cage-axis line at φ = c. Each runs 5000 MC
  pre-steps, 5000 burn-in steps and 200 000 MC steps, sampled every 5.
* **Moves:** d_bead 0.10 Å, d_trans 0.30 Å, d_rot 0.60 rad.
* **Estimators:**
  * F_density: WHAM on 1800 fine bins (bias at fine-bin centres), summed to 180 bins, F = −kT log p.
  * F_MF: pooled conditional mean force per 180-bin, integrated with the piecewise-constant convention (the circular
    mean of Γ subtracted).
  * U(φ): pooled conditional mean potential per bin, and −TS = F − U.
* **D(A, B):** the repo's debiased RMS (LTA validation): centred difference on 180 bins; noise from a
  leave-one-group-out jackknife; 95 % upper bound from 2000 group-bootstrap resamples debiased by 2 × noise.
  Units kJ/mol.
* **Checks** (each reported; a FAIL blocks Gate 2):
  1. Overlap: every fine bin has ≥ 200 pooled samples.
  2. MC self-consistency: D(F_density, F_MF) ≤ 0.05 kJ/mol, upper ≤ 0.075.
  3. Split-half: D(groups 0–7, groups 8–15) ≤ 0.05 kJ/mol on F_density. Each half's noise is from its own jackknife.
  4. Equilibration: D(first half of each window's samples, second half) ≤ 0.05 kJ/mol. The two halves are 1–100k and
     100–200k steps.
  5. Periodic closure: the circular mean of the pooled Γ is reported. Its magnitude must be < 3 × its jackknife
     standard error; otherwise the profile is not periodic.
  6. Reference precision: the jackknife noise of F_density, RMS over bins, is reported. Every later tolerance is
     compared against it.

## 4. Gate 2 — admissible timestep (overdamped Langevin vs MC)

* **Same windows, κ, groups and starts as Gate 1**, with groups 0–15 and their own seeds 9_000_000 + 1000 g + w.
  Molecules start from 5000 exact MC steps, run 1 t.u. of burn-in, then 32 t.u. of the scheme, sampled every
  1e-3 t.u.
* **Candidates:** EM at dt 2e-4, 1e-4 and 5e-5, and LM at dt 2e-4.
* **ADMISSIBLE(dt)** requires all of:
  * D(F_density(EM dt), F_density(MC)) ≤ 0.05 kJ/mol (0.01 kT) with upper95 ≤ 0.075;
  * D(F_MF(EM dt), F_density(MC)) ≤ 0.05 with upper95 ≤ 0.075;
  * the conditional-observable check, (u_x², u_z², η, bond) per bin vs MC.

  The conditional-observable check is reported as RMS z-scores and is diagnostic, except for one hard condition:
  |Δ⟨η|φ⟩| ≤ 0.02 averaged over bins.
* **Freeze:** the production dt is the LARGEST admissible EM dt. LM is reported as a second-order arbiter.
* **Second clause, checked at Gate 3 and blind to FR:** the frozen dt's D must also be ≤ ⅓ of the median ABF-only
  final error e_F(T) at the frozen ABF configuration. If it is not, the next smaller admissible dt is taken and
  Gate 3 is repeated.
* **If no candidate passes:** smaller dt (2.5e-5), or LM if LM passes and EM does not. Production never runs at a
  failing dt.

## 5. Gate 3 — histogram ABF calibration (ABF only; FR does not exist yet)

To be specified in Amendment 1 after Gate 2. It will be frozen before the FR engine is ever run on a production
configuration.

## 6. Two-arm rule

Every comparison is histogram ABF vs histogram ABF + uniform-target FR. No sham arm (user instruction,
2026-10-16).

---

## Amendment 1 — Gate 3 frozen, ABF-only screen design and regime classifier (2026-10-16 16:50 UTC)

*Written after the Gate 1–2 results and the ABF-only calibrations (`EXECUTION_LOG.md` 16:00–16:48). No FR run on any
production configuration exists. The only FR executions so far are unit tests (64–128 walkers, ≤ 4000 steps).*

### A1.1 Gate 2 decision

All four candidates are ADMISSIBLE (2000 bootstrap resamples). The largest admissible EM dt is **2e-4**, with
D_density 0.022 [0.048] and D_MF 0.037 [0.059].

The FR-blind second clause compares D_MF with the median ABF-only final error at the frozen configuration, N = 1024,
three seeds: e_F(T) = 0.129, so ⅓ of it is 0.043. 0.037 ≤ 0.043, so the clause **PASSES**.

**Production dt = 2e-4.** Limitation, stated in advance: the upper bound 0.059 is above 0.043. Every result is
therefore also scored against the dt-consistent EM-2e-4 umbrella reference (secondary), which isolates the
statistical error from the O(dt) bias.

### A1.2 Gate 3 frozen ABF configuration (ABF only; both arms use it unchanged)

| knob | frozen value | how chosen |
|---|---|---|
| estimator | online histogram P0, n_grid 180, all deposits from step 0, own-bin force | prompt §3.1. The bin ladder (90/180/360, N 64/1024, 3 seeds) shows the plateau: 180 → 360 changes e_F(T) −4.6 % and Ī_F −0.8 %. The P0 floor is 0.0034 kJ/mol (2.6 % of e_F) |
| trust rule | per-bin min(1, C_j/200) | colvars fullSamples default. It prevents the sparse-bin runaway measured at 16:05 |
| bias ramp | min(1, step/5000), i.e. 1 t.u. | Gate 3a frozen rule (16:25) |
| pre-equilibration | 25 000 unbiased EM steps (5 t.u.) from the cage-centre starts, IC stream | Gate 3a frozen rule. Its force evaluations are counted separately in every budget |
| clip | none | max \|applied bias\| ≤ 36 (N 1024) / 64 (N 64) kJ/mol/rad vs max \|F′\| 28.5 |
| dt, scheme | EM 2e-4 | A1.1 |
| horizon | T = 40 t.u. (200 000 steps), saves every 500 steps | e_F still decreasing at T. Fixed, not tuned |
| error domain | full circle, 180 bins, uniform weight, additive constant removed | – |
| primary reference | Gate 1 MC pooled mean force on the 180 bins (P0-compatible), `metrics.reference_on_grid` | – |
| secondary reference | EM 2e-4 umbrella F_MF (dt-consistent) | A1.1 |

### A1.3 ABF-only regime screen (ABF only)

* **Design:** K ∈ {64, 256, 1024}, 8 seeds each (seed 930000 + s), the frozen configuration of A1.2, infinite
  dilution (M = 1), Gard & Tait, 600 K.
* **Regions** (from the reference profile, not from ABF data): window W = {|φ| < 0.5 rad}, which holds βF > 5.9 at
  equilibrium; cage core C = {|φ| ≥ 2.0 rad}, where F is within 0.4 kBT of its minimum. The region between them is a
  buffer.
* **True translocation:** a slot's own move takes the tagged COM from the cage core of row r into the cage core of a
  different row. This is a core-set transition, so recrossings of the barrier do not count. Clones inherit labels and
  are never events.
* **Bias-aware target:** Q\*_t ∝ exp(−β(F_ref − A_t)), with A_t the applied bias potential (ramp × trust × Γ)
  integrated.

### A1.4 Regime classifier (applied per K, ABF-only data; thresholds frozen here)

Each statistic is a median over the 8 seeds unless stated. The tests run in this order.

1. **Discovery-limited** if the median first true translocation time > 0.1 T (4 t.u.), or if any seed has no
   translocation by 0.5 T. → STOP.
2. **Conditional-equilibration-limited** if either of these holds:
   * (a) the hidden-variable relaxation time τ_int(η) of the cage axial coordinate is ≥ 0.1 T. That is 1.1 t.u. from
     the Gate 2 pilot, so this is not expected to trigger; it is re-measured from the Gate 2 EM-2e-4 time series.
   * (b) over saves in [T/2, T], the population's conditional lag d_η(t) (`metrics.conditional_lag`, weighted over
     coarse bins) has a median above 3 × its finite-N noise floor. The floor is computed from the reference
     conditional variances of η per coarse bin and the instantaneous per-bin counts.

   → STOP.
3. **Establishment-limited** requires BOTH of these:
   * (E) the longest contiguous span in [2 t.u., T] during which the window count is below 0.5 N Q\*_W − 2 sd
     (binomial) is ≥ 0.2 T = 8 t.u.;
   * (I) the deficit matters for the estimate. Window bins at T hold less than half their uniform share of the
     cumulative deposits, ((C_W/C)/(|W|/2π) < 0.5), OR the window bins carry ≥ 50 % of the final squared mean-force
     error.

   → ELIGIBLE for the FR confirmation (two arms only).
4. Otherwise **ABF-sufficient** → STOP FR for this cell.

Secondary statistics are reported for every K but do not enter the classification: TV(p̂, Q\*) on 18 coarse bins
against its exact finite-N floor; the late-deposit estimator error (deposits after 2 t.u.); the
first-window-entry time.

### A1.5 What happens next

* **If a dilute cell is ELIGIBLE:** an FR safety calibration (rate ladder {0.02, 0.05, 0.1, 0.2}, safety-only
  criteria: windowed ancestor ESS/N ≥ 0.30 and max lineage share ≤ 0.05 at the production horizon) is followed by an
  exploratory 8-seed block and a confirmatory 16-seed block. The blocks use fresh seeds. A separate amendment
  freezes them before any FR error is computed.
* **If no dilute cell is eligible:** the dilute cell is an ABF-sufficient result. The campaign proceeds to the
  loading study (§2.3 of the campaign prompt), which needs its own Gates 1–3 at each loading.

---

## Amendment 2 — classifier power repair, before the screen is classified (2026-10-16 17:25 UTC)

**Why.** The independent audit (`audit/AUDIT1_PREREG.md` F1; `audit/AUDIT1_REFERENCE_STATISTICS.md` F1/F2) showed
that the A1.4 establishment test (E) has no power:
* its per-save threshold 0.5 N Q\* − 2 sd is ≤ 0 at N = 64 (vacuous);
* at N = 256 / 1024 it needs a 74–91 % deficit;
* its null false-positive rate is 0.

The conditional-lag test 2(b) also has little power (a median of single-save ratios; η only). Both are replaced
below. The new tests have measured null and power behaviour (`tests/test_eri_metrics.py`).

**Blindness statement.**
* The coordinator has not run the classifier on, or looked at classification statistics of, `screen_dilute`.
* One auditor (statistics lens) did load the screen files to compute the positivity of the OLD threshold. It reported
  "threshold positive in 3–13 % of [2, T]" and "longest positive-threshold span 1.1–5.1 t.u.". That figure measures
  the old rule's vacuity, not the occupancy outcome. The design auditor did not read the screen.
* The calibration runs at the frozen configuration were seen by everyone. They show window occupancy at 0.9–1.2× its
  target and a window error share of 0.03–0.21.

**A2.1 Establishment (E′), replacing A1.4-3(E).**
* R(t) = window fraction / Q\*_W(t) is averaged over consecutive 2 t.u. blocks in [2, T].
* A block is deficient if mean R < 0.75 AND 1 − mean R > 3 sd_null. sd_null is the binomial null sd of the block
  mean, with n_eff = block/0.5 t.u. independent saves.
* (E′) holds if the median over seeds of the longest run of consecutive deficient blocks is ≥ 0.2 T = 8 t.u.
* Measured power (`metrics.establishment_blocks`; AR(1)-correlated synthetic occupancy, 12 t.u. deficit):

  | N | detection rate by deficit |
  |---|---|
  | 1024 | 71 % at a 30 % deficit, 99 % at 40 % |
  | 256 | 41 % at 30 %, 87 % at 40 %, 99 % at 50 % |
  | 64 | 26 % at 50 %, 57 % at 60 %, 95 % at 80 % |

  False positives: 0/200 at every N. Small-N power is limited by the binomial noise floor; this is reported, not
  hidden.
* The A1.4 (E) statistic is still computed and reported, labelled "frozen A1 rule (vacuous)".

**A2.2 Conditional lag (2b′), replacing A1.4-2(b).**
* For η and u_x², take the count-weighted RMS over coarse bins of (time-averaged population conditional mean over
  saves in [T/2, T]) − (reference conditional mean).
* Its iid floor uses n_eff = Σ_t n_b(t) · Δt/0.5 t.u. The reference conditional means and variances come from the
  Gate 2 EM-2e-4 umbrella time series (the Gate 1 accumulators store no second moments).
* (2b′) fires if either variable's statistic exceeds 3 × floor. Measured: a 0.15 lag in η at N = 256 fires; the null
  does not (test).
* (2a) is kept as frozen. A descriptive diagnostic is added, without changing the label: the ratio e_F(T) / e_F(late,
  deposits after 2 t.u.), which records the early-deposit transient (audit F2/F7).

**A2.3 Unchanged:** the discovery test, the information test (I), the classifier order and every threshold not named
above.
