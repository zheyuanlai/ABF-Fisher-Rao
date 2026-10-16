# Independent adversarial audit — overnight molecular extension (ERI 600 K, ITE 300 K, ITE 150 K)

*2026-10-16, after commit ce45ded2. Read-only audit: CPU only (`CUDA_VISIBLE_DEVICES=""`), ≤ 4 numba threads, every
run < 10 min. Repo code was imported only where noted (pure-numpy `eri.metrics`, the Gate 0 pytest with
`NUMBA_CACHE_DIR` and `PYTHONDONTWRITEBYTECODE` pointed away from the repo). Every recomputation below uses
independent scripts (own WHAM, own histogram PMF, own Q\*, own block test), kept in the session scratchpad
`.../scratchpad/audit_overnight/` (`famcheck.py`, `beadgrid.py`, `ring.py`, `starts.py`, `gates.py`, `refs.py`,
`calib.py`, `classify_indep.py`, `binlabel.py`, `t60.py`, `phase.py`, `lta_posctl.py`).*

## Verdict

**No finding changes any stated classification.** All three cells stay ABF-SUFFICIENT at K 64 / 256 / 1024 under the
frozen rules and under every alternative implementation tried. Every Gate 1/2/3 number and every classifier
statistic that was checked reproduces exactly from the raw npz files. The physics checks hold: ethane cannot leave
its channel family, the CV origin is the 8-ring plane, and the MC reference starts in family A.

Four findings do change how the results should be read or what is recommended next:
* **F1 (high, interpretation).** The frozen classifier fails its only available real positive control.
* **F2 (medium, protocol).** The region rule was not re-checked at 150 K, and both regions fail it.
* **F3 (medium, framing).** The "34–57 % early-phase error" lead mostly falls where FR cannot act, and the cell the
  HANDOFF recommends has the smallest addressable share, not the largest.
* **F4 (low–medium, protocol).** The 150 K switch to 360 bins comes from mixing two references.

## Findings

### F1 — HIGH (interpretation; labels unchanged). The frozen classifier would label the repo's validated FR-positive cell, LTA 150 K, "ABF-SUFFICIENT"

* **Where.** `scripts/eri_campaign/classify_screen.py:159` (the (I) clause). PREREGISTRATION.md:224 (Amendment 3: "A′-150
  therefore tests whether that regime appears"). RESULTS.md:13 and STATUS.md:8.
* **Evidence.** `lta_posctl.py` applies the A2 logic to the LTA histogram production
  (`results/lta_histogram/production_T150/abf.npz`, 16 seeds, T 60, 2.4 t.u. blocks). It is approximate: occupancy
  comes from the KDE `p_hat`, the applied bias is ramp × hpmf(mean_force), and the reference Γ comes from finite
  differences of the published F. In that cell FR gave −28.5 % Ī_F against exact MC (`LTA_VALIDATION.md` §4).

  | LTA | E′: median longest deficient run (threshold 0.2 T = 12) | window deposit share (needs < 0.5) | window error share (needs ≥ 0.5) | frozen label |
  |---|---|---|---|---|
  | 150 K ABF | **21.6 t.u.** (E′ fires) | 0.85 | 0.071 | **ABF-SUFFICIENT** ((I) fails) |
  | 150 K FR arm | 9.6 | 0.89 | 0.13 | – |
  | 300 K ABF (FR −13.7 %) | 9.6 | 0.92 | 0.070 | ABF-SUFFICIENT |

  * E′ works as intended: it fires in LTA 150 K, and FR halves the deficient run.
  * (I) fails by a wide margin. It cannot fire unless the deficit lasts most of the run, or unless the window
    dominates the final error. LTA 150 K satisfies neither, yet FR helped there, including at the endpoint
    (0.13 → 0.06 kJ/mol).
* **Consequence.**
  * "ABF-SUFFICIENT → FR not licensed" is a property of this gate. It is not evidence that FR would not help.
  * Amendment 3's test ("does the LTA regime appear in ITE?") cannot be answered by this instrument, because the
    instrument rejects LTA itself.
  * The ITE labels still stand on E′ alone at the frozen horizons (ITE 150 K: 8 t.u. against a 40 t.u. threshold).
    But the E′ margin depends on the horizon (`t60.py`): at an LTA-like horizon of 60 t.u. (2.4 t.u. blocks,
    12 t.u. threshold), ITE 150 K gives 6.0 / 9.6 / 9.6 t.u. at K 64 / 256 / 1024. Two of eight K 256 seeds reach 12.
  * ITE 150 K is therefore less establishment-starved than LTA 150 K (about 10 vs 22 t.u.), but not by an order of
    magnitude.
* **Fix.**
  * Add a sentence to RESULTS/STATUS: the frozen (I) clause gives a false negative on LTA 150 K, so "ABF-sufficient"
    must not be read as "FR-unnecessary".
  * Before any further preregistration that gates FR on this classifier, run the exact classifier on the LTA ABF
    data as a positive control (needs raw walker histograms), or drop or recalibrate (I).

### F2 — MEDIUM (frozen rule not applied; label unchanged). At 150 K, neither region was verified, and both fail the prereg §5 rule

* **Where.** PREREGISTRATION.md:79–81 (§5: W and C "are verified from the Gate 1 profile before the screen … if not,
  re-derived"). Amendment 3 applies §§2–6 unchanged. Amendment 2 (line 198) records the check at 300 K only.
  Amendments 4–7 contain no 150 K check.
* **Evidence.** From `results/overnight/ite150/reference/ite_itq3/reference_profile.npz`:
  * **W** = {|φ| < 0.5}: min βF = 22.30, but the rule needs ≥ βF_max − 1.5 = 22.55. **FAIL.** The re-derived W is
    |φ| < 0.454.
  * **C** = {|φ| ≥ 2}: max βF = 2.17, but the rule needs ≤ 1.5. **FAIL.**
    * The bins with βF ≤ 1.5 are φ ∈ [−3.05, −2.04] only. The 150 K cage is asymmetric (minimum at φ = −2.46), so no
      region of the form |φ| ≥ c satisfies the rule.
  * 300 K passes (14.69 ≥ 14.43; 1.04 ≤ 1.5), as recorded.
* **Consequence.**
  * Recomputed with W = |φ| < 0.454 (`classify_indep.py`, alternative "W0.45"), the result does not change:
    * longest deficient run 8 / 8 / 8 t.u.;
    * deposit share 1.02 / 1.03 / 1.04;
    * window error share 0.008 / 0.012 / 0.006.
  * C sets the engine's online translocation labels, so it cannot be redone from the saves. A stricter or asymmetric
    cage core would delay the first translocation by about the time to diffuse across the cage (≲ 3 t.u.). The
    discovery medians are 9.0 / 6.2 / 4.5 t.u. against a 20 t.u. threshold (worst seed 13.4 t.u.), so the label
    cannot plausibly flip.
  * This is still an undeclared deviation from a frozen rule.
* **Fix.** Add an amendment recording the failed check, the re-derived W, and the reason C was kept (or the
  asymmetric C), plus the recomputed statistics above.

### F3 — MEDIUM (framing of the "open lead"; changes the HANDOFF rationale). Most of the 34–57 % early-phase error share falls in the ramp, before FR can act; ITE 300 K has the smallest addressable share

* **Where.**
  * STATUS.md:23.
  * HANDOFF.md:46–49: "ITE 300 K (the largest share, about 55 %)", with FR from the end of the ramp.
  * `scripts/overnight/deficit_phase_share.py:5`: "This early phase is the only one in which marginal FR could act."
  * `fig_overnight_summary.png`, middle panel.
* **Evidence** (`phase.py`; repo `deficit_phase_share.json` reproduced exactly). The I_F share is split into the
  part accrued during the ramp [0, 1] and the part in (1, T_reach]:

  | cell | share [0, T_reach] (reported) | in [0, ramp] | first-window-entry part | **(ramp, T_reach] (addressable with fr_start = ramp)** |
  |---|---|---|---|---|
  | ERI 600 K | 0.44 / 0.34 / 0.35 | 0.21 / 0.14 / 0.12 | 0.13 / 0.01 / 0.00 | **0.23 / 0.20 / 0.22** |
  | ITE 300 K | 0.57 / 0.54 / 0.52 | 0.36 / 0.38 / 0.33 | 0.42 / 0.38 / 0.32 | **0.20 / 0.16 / 0.20** |
  | ITE 150 K | 0.43 / 0.47 / 0.40 | 0.18 / 0.22 / 0.19 | 0.31 / 0.26 / 0.19 | **0.26 / 0.24 / 0.21** |

  * e_F(0) is 10–14 kJ/mol, the RMS of F itself. Much of the early share is just the bias starting from zero.
* **Consequence.**
  * Under the frozen FR schedule (fr_start = ramp), the share FR could address is 16–26 %, not 34–57 %.
  * Even a perfect FR effect over that window gives at most about −20 % Ī_F at ITE 300 K K 1024. A realistic
    partial effect is near the −10 % useful threshold.
  * ITE 300 K has the **smallest** addressable share, not the largest. ITE 150 K and ERI are larger.
  * "The only phase FR could act" is contradicted by the repo's own LTA 150 K result, where FR also cut the endpoint
    bias.
* **Fix.**
  * Report the addressable share in STATUS, HANDOFF and the figure.
  * Base the choice of cell on it (or on an FR-from-t = 0 variant, where frontier cloning can act before the first
    window entry).
  * Remove the "only phase" sentence.

### F4 — LOW–MEDIUM (frozen rule applied as written, but its decision is a reference artefact; label unchanged). The 150 K switch to 360 bins flips under a common reference

* **Where.** PREREGISTRATION.md:74 (error reference: MC F_MF on 90/180 bins, smooth Fourier on 360) and
  PREREGISTRATION.md:324–329 (Amendment 6). `src/eri/metrics.py:252–258`.
* **Evidence** (`calib.py`; K 1024, dt 1e-4, 3 seeds; prereg-reference medians reproduced exactly):

  | | 180 bins | 360 bins | improvement of 360 over 180 | rule (> 10 %) |
  |---|---|---|---|---|
  | as applied (MC ref for 180, smooth ref for 360) | 0.0390 | 0.0348 | 10.7 % | switch |
  | common smooth reference | 0.0375 | 0.0348 | **7.1 %** | **keep 180** |

  * Paired per-seed differences under the common reference: +7 %, −2 %, −12 %.
  * Amendment 6 says the two references differ by "0.004 kJ/mol RMS at 180 bins". The measured difference is
    **0.0020 RMS (0.0040 max)**. The repo's `reference_smooth` agrees with an independent fit to 1e-8.
  * At 300 K the ladder is flat under both references (0.0649 / 0.0650 against MC; 0.0659 / 0.0650 against smooth),
    and the switch condition was correctly ruled out by the floor share.
* **Consequence.**
  * The production grid choice rests on 3.6 points of reference change.
  * The classification is not sensitive to it (`binlabel.py`). On the 150 K calib_bins runs at 180 bins:
    * longest run 8 t.u.;
    * deposit share 1.04;
    * error share 0.002;
    * these are as far from the thresholds as the 360-bin values.
* **Fix.**
  * Correct "0.004 RMS" to "0.0020 RMS (0.0040 max)".
  * Note that the switch would not occur with a common reference.
  * In future preregistrations, compare bin widths against one reference.

### F5 — LOW (physics description). The "T-independent ΔS within noise" and "ΔU unchanged" claims are statistically false

* **Where.** RESULTS.md:83 and PREREGISTRATION.md:263–264.
* **Evidence** (16-group jackknife of the barrier decomposition, same window/cage-bin definition as `barrier_stats`):

  | | ΔU (kJ/mol) | −ΔS (J mol⁻¹ K⁻¹) |
  |---|---|---|
  | 300 K | 20.614 ± 0.003 | 63.63 ± 0.03 |
  | 150 K | 20.158 ± 0.002 | 65.39 ± 0.03 |
  | difference | −0.456 ± 0.003 | 1.76 ± 0.04 (z ≈ 47) |

* **Consequence.** The change is small (about 3 %) but clearly resolved, so the barrier has a small heat-capacity
  component. Nothing downstream depends on this.
* **Fix.** Write "approximately T-independent (ΔS −63.6 vs −65.4 J mol⁻¹ K⁻¹; ΔU −0.46 kJ/mol), resolved but small".

### F6 — LOW (undeclared, conservative deviations in the dt decision; no effect)

* **Proxy instead of the screen.** Amendment 4 (line 276) says the second clause would be checked on the screen.
  Amendment 5 (line 294) instead checked it on the Gate 3b K 1024 median (0.0411), with no screen at 2e-4.
  * The 1e-4 screen gives a minimum median of 0.0397, so ⅓ × 0.0397 = 0.0132 < 0.0168. The decision would be the
    same, and the deviation is in the conservative direction.
* **D_MF compares unlike estimators.** D_MF compares Langevin F_MF with MC **F_density**. That carries the ~0.010
  kJ/mol F_density/F_MF estimator mismatch (Gate 1 check 2).
  * Against MC F_MF (independent WHAM): 150 K EM 2e-4 gives D_MF = 0.0147, still above 0.0137, so the decision stands.
  * At 1e-4, D_MF falls to 0.0076 (150 K) and 0.0088 (300 K) against the reported 0.0107 / 0.0129.
* **Fix.** Record both points in an amendment, and define D_MF against F_MF in future preregistrations.

### F7 — LOW (timeline and document-order inconsistencies; no outcome dependence)

* **Amendment 1 timestamp.** Amendment 1 (line 147) is stamped 18:50 UTC, but it was committed at 18:38:03 together
  with Amendment 2 (stamped 18:38). The Gate 3b runs are recorded at 18:32–18:37.
* **Amendment 3 timing.** Amendment 3 (line 214) says "before any 150 K computation". The 150 K Gate 1 MC files were
  written from 18:48:04 to 18:53:19. The amendment was committed at 18:49:11, and the config says "_frozen ~18:52".
  The MC reference is arm-blind, so this is cosmetic.
* **Null-control timing.** §5 (line 95) requires the null control "before the screen". At 150 K it was run after the
  screen and before the classification (line 347; the ledger puts it at 20:32–20:45, and `classification.json` is
  stamped 20:32:31). The null control is synthetic and data-blind, so the label is unaffected.
* **Fix.** Correct the timestamps and record the null-control timing as a deviation.

### F8 — LOW (reproducibility gaps)

* **Code missing from the repo.** `horizon_choice.json`, `warm_choice.json`, the Gate 3c ladder summary, and the ITE
  null-control and power simulations (Amendments 2 and 7) were produced by code that is not in the repo.
  * All the choice numbers reproduce independently, so the gap is reproducibility, not correctness.
* **Latent hazard.** `scripts/eri_campaign/summarize_calib.py:12` calls `reference_on_grid(ng)` with the default
  variant `gard_tait` and the default ERI `REF_BASE`. Used on ITE runs, it would silently score them against the
  ERI reference.
* **HANDOFF.** HANDOFF.md has no 150 K reproduction commands (the `--refbase`, `--hidden-tag` and `--seedshift 3000` /
  screen `--seedshift 1000` choices).
* **Fix.**
  * Commit the inline scripts.
  * Give `summarize_calib.py` explicit `--refbase` and `--variant` arguments (or make it raise without them).
  * Add the 150 K commands to HANDOFF.

### F9 — INFO

* **Horizon-conditional thresholds.** Discovery and establishment thresholds scale with T (0.1 T and 0.2 T).
  * At 150 K, K 64's first translocation (9.0 t.u.) is 45 % of the 20 t.u. threshold. It would be 90 % at T = 100,
    and the cell would be discovery-limited at T = 50.
  * T came from an independent rule (e(100)/e(200) = 1.65 > 1.5, which is robust), so this is legitimate. But the
    labels are conditional on T.
* **Summary-figure panel 3.** It compares e_F(T) across cells at different horizons (40 / 50 / 200 t.u.). ITE 150 K
  looks most accurate partly because its T is 4× longer.
* **Window error share.** RESULTS.md:23 says "window bins hold 1–7 %". The actual range is 0.6–7 % (ITE 150 K
  K 1024: 0.006).
* **Block count.** §5's "(≥ 25 blocks)" is in fact 24 blocks over [2, T], and the last 6 t.u. (150 K) or 0 t.u.
  (300 K) are not tested. This is immaterial.
* **η criterion.** The η criterion of Gate 2 (|Δ⟨η|φ⟩| ≤ 0.02) has little power in ITE, because η stays between 0.8
  and 1.0 inside family A. Observed differences are 0.0002.

## Checks that passed (no defect)

### 1. Physics

* **Framework build.** `perm (1, 2, 0)` is cyclic (det +1, a proper rotation), with box X = crystal b, Y = c, Z = a.
  The box is 768 Si + 1536 O and the coordination check passes.
* **Force-field parameters.** The model and both reference configs carry the Dubbeldam 2004 parameters:
  * CH3–O 93 K / 3.48 Å, oxygen only (the Si list is empty);
  * CH3–CH3 108 K / 3.76 Å;
  * bond 96 500 K/Å², r0 1.54 Å;
  * 12 Å cutoff, energy-shifted.

  The reference MC and the ABF engine use identical force-field dicts.
* **8-ring and CV origin** (`ring.py`).
  * Exactly 8 O atoms per period lie 3.24–3.50 Å from the family-A axis, at X = 3.99 / 4.25 / 4.30 Å (spread
    0.12 Å).
  * Their mean is **4.1979 Å = x_w**.
  * F(φ) peaks in the bin straddling φ = 0 at both temperatures.
  * The F asymmetry (up to 2.4 kJ/mol; cage minimum at φ = −2.46) is physical: no operation of Cmcm reverses y within
    one channel (`CAND_A_LITERATURE.md` §4.1).
* **Channel isolation** (`beadgrid.py`). An independent single-bead grid (0.2 Å) with fully periodic labelling shows:
  * family A connects to another channel class only between 100 and 120 kJ/mol (the c-glide family, via the [001]
    ring);
  * it never connects to the C-centred family up to 200 kJ/mol.
  * This confirms "> 100 kJ/mol per bead".
* **End-state check** (`famcheck.py`). Every walker's final configuration lies in family A (axis Y = c/4 + m c,
  Z = n a), for both beads and the COM, in:
  * all 48 screen runs (21 504 walkers);
  * all 96 calibration runs (52 224 walkers).

  The maximum COM distance from the axis is 3.08 Å; channel spacing is ≥ 9.8 Å.
* **MC starts** (`starts.py`, a numpy replica of `reference.window_starts`). All starts use the 12 family-A windows.
  The COM lies on the axis to 1e-15 Å, at exactly the umbrella centre, for 40 and 64 windows.
* **Reference conditional means.** The η conditional means of the MC reference (0.80–1.00) agree with family A. A
  C-centred-family sample would give η ≈ −1.
* **Gate 0.** The Gate 0 tests pass (5/5).

### 2. Gates (independent WHAM and jackknife; `gates.py`)

* **Gate 1 check 2.** D(F_density, F_MF) = 0.0099 (300 K) and 0.0103 (150 K), as reported.
* **Gate 2 at 300 K.** D_density / D_MF are reproduced to four decimals:
  * EM 2e-4: 0.0357 / 0.0301 (NOT ADMISSIBLE confirmed);
  * EM 1e-4: 0.0148 / 0.0129.
* **Gate 2 at 150 K.** D_density / D_MF:
  * EM 2e-4: 0.0161 / 0.0168;
  * EM 1e-4: 0.0026 / 0.0107.
* **Gate 3** (`calib.py`). All values reproduce:
  * horizon medians at both T (horizon rule → T 50 at 300 K, T 200 at 150 K);
  * Gate 3b scores (0.914 / 1.324 / 1.192 at 300 K; 0.933 / 1.208 / 1.235 at 150 K; the 5 % tie rule is not
    triggered);
  * the Amendment 5 consistency check (0.0390 / 0.0411 = 0.95);
  * the dt clause at 1e-4 (0.0107 ≤ 0.0132);
  * the floor share (10.2 %);
  * the per-seed bin differences (+3 / −6 / −15 %);
  * the VALIDATION 300 K ladder table.
* **Frozen prereg body.** The body was not edited after the freeze, apart from the timestamp correction.

### 3. Classifier (`classify_indep.py`, no repo code)

* **Reproduction.** For all K in both ITE cells, the following match `classification.json`:
  * first translocation;
  * longest deficient run;
  * minimum and median block ratio;
  * window deposit share;
  * window error share;
  * e_F(T), Ī_F;
  * the A1 span.

  The saved translocation counters agree with `ist` to within one save interval.
* **Grid-matched reference.** The 360-bin runs are scored against the smooth reference, as frozen; it matches an
  independent fit.
* **Robustness.** None of these alternatives brings any K near either threshold:
  * uniform target;
  * the A1 per-save rule;
  * Q\* from the full (un-ramped, un-trusted) bias;
  * re-derived W;
  * 2 t.u. blocks at 150 K.

  (I) fails by more than 10× in every variant.

### 4. Documentation, provenance and compute

* **Numbers.** Every number in the RESULTS, STATUS and VALIDATION tables that was checked matches the JSONs,
  including the ERI rows. The exceptions are those listed in F4, F5 and F9.
* **Claim typing.** No "FR-neutral" or "FR helps" claim appears anywhere. HANDOFF.md:37 explicitly forbids it.
* **Provenance.** The engine provenance hash in every screen npz (e37993f1e67b) matches the current
  `engine.py` / `model.py` / `framework.py` / `lta_ladder_numba.py`.
* **Screen configs.** The screen configs, seed blocks (950 000+ and 951 000+) and horizons match the frozen
  configuration.
* **Compute ledger.** The ledger is consistent with the recorded wall × threads and the MC/EM `elapsed` sums.
  * 150 K: MC 3.60, EM 2.14 / 4.06, bins 14.1, screen 15.0 core-h, against the ledger's 15.5 / 16.3.
  * 300 K Gate 1/2: 2.24 / 0.70 / 1.33 / 0.70 / 2.63 core-h, matching the ledger exactly.
